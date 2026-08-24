use crate::metal::{
    executor::{DispatchRecord, ExecutorSlot, Grid},
    trace, Buffer, ComputePipeline, Fence,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLSize,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

/// Shared cross-encoder output map: maps buffer pointer -> fence of the last encoder that wrote it.
/// Used by subsequent encoders to call waitForFence before reading those buffers.
pub type PrevCeOutputs = Arc<Mutex<HashMap<usize, Arc<Fence>>>>;

fn size_tuple(size: MTLSize) -> (usize, usize, usize) {
    (size.width, size.height, size.depth)
}

/// Barrier tracking state for one encoder session.
/// Owned by ComputeCommandEncoder via Arc<Mutex<>> so clones share state.
pub struct EncoderState {
    /// Buffer ptrs written since last barrier (RAW/WAW detection).
    pub prev_outputs: HashSet<usize>,
    pub next_outputs: HashSet<usize>,
    /// Buffer ptrs read since last barrier (WAR detection).
    pub prev_inputs: HashSet<usize>,
    pub next_inputs: HashSet<usize>,
    pub needs_barrier: bool,
    /// All inputs seen this encoder session (cross-encoder fence coordination).
    pub all_inputs: HashSet<usize>,
    /// All outputs seen this encoder session (registered in global map at end_encoding).
    pub all_outputs: HashSet<usize>,
    /// Fences already waited on this session, so a buffer bound repeatedly does
    /// not re-emit the same wait.
    pub waited_fences: HashSet<usize>,
    /// Name of the pipeline most recently set, so a dispatch can be attributed
    /// to a kernel. Only maintained when an executor is installed -- Metal has
    /// no way to read the bound pipeline back, and doing so per dispatch on the
    /// classical path would be cost for nobody.
    pub current_pipeline: Option<Arc<str>>,
}

impl EncoderState {
    pub fn new() -> Self {
        EncoderState {
            prev_outputs: HashSet::new(),
            next_outputs: HashSet::new(),
            prev_inputs: HashSet::new(),
            next_inputs: HashSet::new(),
            needs_barrier: false,
            all_inputs: HashSet::new(),
            all_outputs: HashSet::new(),
            waited_fences: HashSet::new(),
            current_pipeline: None,
        }
    }
}

#[derive(Clone)]
pub struct ComputeCommandEncoder {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Retained so we can register completion handlers on this CB.
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Per-encoder-session fence. Updated at end_encoding.
    pub(crate) fence: Arc<Fence>,
    /// Hazard tracking state. Arc shared between the canonical encoder in EntryState
    /// and the clone held by CommandsGuard. Uncontended in practice (CommandsGuard
    /// holds the outer Commands mutex for the entire kernel dispatch).
    pub(crate) state: Arc<Mutex<EncoderState>>,
    /// Buffer -> fence of its last writer, so a bind can wait on just that
    /// buffer instead of on every live fence.
    pub(crate) prev_ce_outputs: PrevCeOutputs,
    /// How dispatches reach the GPU (`DESIGN.md` §11.1).
    ///
    /// `ExecutorSlot::Classical` is the default and forwards to `self.raw`
    /// exactly as before this field existed, so the default path is unchanged
    /// rather than merely equivalent.
    pub(crate) executor: Arc<ExecutorSlot>,
    /// Open packed-params block, when a caller is capturing scalars instead of
    /// binding them inline (`DESIGN.md` §11.3b, issue #38).
    ///
    /// `AtomicBool` beside the buffer, rather than testing an `Option` under
    /// the lock, so the classical path pays a relaxed load and nothing else.
    /// Following #35's shape deliberately: that change kept "the classical path
    /// must not regress" *structural* by branching before doing any work, and a
    /// mutex per scalar bind would have given that up -- `DESIGN.md` §6.4a
    /// measured per-bind bookkeeping at 29.1 ns and the whole fence probe at
    /// 5.1 % of non-GPU time, so per-bind additions are exactly the shape worth
    /// not paying by default.
    pub(crate) capturing: Arc<AtomicBool>,
    pub(crate) param_capture: Arc<Mutex<ParamCapture>>,
}

/// Scalars accumulated for a packed params block, and the buffer renumbering
/// that has to accompany them.
///
/// Diverting a scalar out of the argument list leaves a hole in the buffer
/// indices: `call_rms_norm` binds `(length, elements_to_sum, src, dst, alpha,
/// eps)` at 0..5, and the packed kernel takes `(params, src, dst, alpha)` at
/// 0..3. So capture cannot only collect bytes -- it must also renumber the
/// bindings that remain, or every buffer lands one or two slots too high and
/// the kernel reads whatever was left at that index. Under
/// `HazardTrackingModeUntracked` that is a silent wrong answer (`DESIGN.md`
/// §3.5), which is the same class of failure as a bad struct offset and is
/// caught by the same bit-identical test.
///
/// `next_buffer` starts at 1 because slot 0 is the params buffer itself.
#[derive(Default)]
pub struct ParamCapture {
    bytes: Vec<u8>,
    next_buffer: usize,
    /// Buffers allocated to hold arrays that the classical path binds with
    /// `setBytes`. Handed to the caller at capture close, so their lifetime is
    /// the dispatch's rather than the capture's.
    staged: Vec<Buffer>,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> ComputeCommandEncoder {
        Self::with_executor(
            raw,
            command_buffer,
            fence,
            prev_ce_outputs,
            Arc::new(ExecutorSlot::Classical),
        )
    }

    /// As [`Self::new`], but submitting dispatches through `executor`.
    pub fn with_executor(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
        executor: Arc<ExecutorSlot>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            command_buffer,
            fence,
            state: Arc::new(Mutex::new(EncoderState::new())),
            prev_ce_outputs,
            executor,
            capturing: Arc::new(AtomicBool::new(false)),
            param_capture: Arc::new(Mutex::new(ParamCapture::default())),
        }
    }

    /// Wait on the fence of `ptr`'s last writer, if any, and only once.
    ///
    /// This replaces waiting on every live fence at encoder creation. The
    /// encoder already records every buffer it binds, so the wait can be
    /// limited to buffers this encoder actually touches.
    fn wait_for_buffer(&self, ptr: usize) {
        let fence = {
            let map = self.prev_ce_outputs.lock().unwrap();
            map.get(&ptr).cloned()
        };
        let Some(fence) = fence else { return };

        let mut state = self.state.lock().unwrap();
        if state.waited_fences.insert(Arc::as_ptr(&fence) as usize) {
            drop(state);
            self.raw.waitForFence(fence.raw());
        }
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.auto_barrier();
        trace::record_dispatch(
            size_tuple(threads_per_grid),
            size_tuple(threads_per_threadgroup),
            false,
        );
        if !self.offer_to_executor(
            Grid::Threads(threads_per_grid.into()),
            threads_per_threadgroup,
        ) {
            return;
        }
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.auto_barrier();
        trace::record_dispatch(
            size_tuple(threadgroups_per_grid),
            size_tuple(threads_per_threadgroup),
            true,
        );
        if !self.offer_to_executor(
            Grid::Threadgroups(threadgroups_per_grid.into()),
            threads_per_threadgroup,
        ) {
            return;
        }
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    /// Offer this dispatch to the executor; `true` means encode it normally.
    ///
    /// The early return on the classical path is what keeps `DESIGN.md` §11.1's
    /// "must not regress" structural: with no executor installed this is one
    /// predictable branch, and the `DispatchRecord` -- which costs an `Arc<str>`
    /// clone and a lock acquisition -- is never built.
    #[inline(always)]
    fn offer_to_executor(&self, grid: Grid, threads_per_threadgroup: MTLSize) -> bool {
        if self.executor.is_classical() {
            return true;
        }
        let kernel = self.state.lock().unwrap().current_pipeline.clone();
        self.executor.dispatch(&DispatchRecord {
            kernel,
            grid,
            threads_per_threadgroup: threads_per_threadgroup.into(),
        })
    }

    fn auto_barrier(&self) {
        let mut s = self.state.lock().unwrap();
        if s.needs_barrier {
            self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            s.needs_barrier = false;
            s.prev_outputs = std::mem::take(&mut s.next_outputs);
            s.prev_inputs = std::mem::take(&mut s.next_inputs);
        } else {
            let next_out = std::mem::take(&mut s.next_outputs);
            s.prev_outputs.extend(next_out);
            let next_in = std::mem::take(&mut s.next_inputs);
            s.prev_inputs.extend(next_in);
        }
    }

    pub fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        let index = self.capture_buffer_index(index);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(index, ptr, offset, false);
            // Read-after-write against an earlier encoder: order against that
            // buffer's last writer only.
            self.wait_for_buffer(ptr);
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_inputs.insert(ptr);
            s.all_inputs.insert(ptr);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, false);
            }
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        let index = self.capture_buffer_index(index);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(index, ptr, offset, true);
            // Write-after-write or write-after-read against an earlier encoder.
            self.wait_for_buffer(ptr);
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) || s.prev_inputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_outputs.insert(ptr);
            s.all_outputs.insert(ptr);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, true);
            }
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    /// Capture this scalar into the packed-params staging area instead of
    /// binding it inline, when a capture is open.
    ///
    /// Returns `false` when none is, which is the classical path and the
    /// default: the caller then does exactly what it did before. The relaxed
    /// load is the whole of the cost in that case -- no lock is taken.
    ///
    /// This is `DESIGN.md` §11.3b's "one function, not 51 call sites".
    /// `EncoderParam::set_param` is the only place a primitive reaches
    /// `setBytes`, so diverting it here leaves every `set_params!` site and
    /// every `call_*` entry point untouched: the caller still passes a `u32`,
    /// and where it lands becomes the encoder's business.
    #[inline(always)]
    pub(crate) fn capture_scalar(&self, bytes: &[u8], align: usize) -> bool {
        if !self.capturing.load(Ordering::Relaxed) {
            return false;
        }
        let mut cap = self.param_capture.lock().unwrap();
        // Match the layout the kernel will read: pad to the field's own
        // alignment before appending. `DESIGN.md` §15.1 -- a field at the wrong
        // offset is silent corruption, so the padding rule has to be the one
        // MSL actually applies, and `reduce_params_layout_matches_metal` is
        // what proves it is.
        while !cap.bytes.len().is_multiple_of(align) {
            cap.bytes.push(0);
        }
        cap.bytes.extend_from_slice(bytes);
        true
    }

    /// Promote a `setBytes` array to a device buffer, when capturing.
    ///
    /// `dims` and `strides` cannot join the packed struct -- their length comes
    /// from the tensor's layout -- but they do not need to: an ICB command can
    /// bind a buffer of any length, it just has no `setBytes` at all. So under
    /// capture they become a real buffer and keep their own argument slot.
    ///
    /// Allocating per call is deliberate for this change and is not what a
    /// decode path would do; see `with_packed_params` in `kernels/reduce.rs`
    /// for why that is acceptable here and what a plan-owned buffer would look
    /// like instead.
    #[inline]
    pub(crate) fn capture_array(&self, len_bytes: usize, bytes: *const c_void) -> bool {
        if !self.capturing.load(Ordering::Relaxed) {
            return false;
        }
        let index = self.capture_buffer_index(usize::MAX);
        let device = crate::metal::Device::new(self.command_buffer.device());
        let Ok(buffer) = device.new_buffer_with_data(bytes, len_bytes, crate::RESOURCE_OPTIONS)
        else {
            // Allocation failure here would otherwise bind nothing and let the
            // kernel read a stale slot. Reporting it is not possible through
            // `EncoderParam`, which returns unit, so fail loudly rather than
            // silently: this path is behind an opt-in style and is not reached
            // by any classical dispatch.
            panic!("packed-params staging allocation failed for {len_bytes} bytes");
        };
        // Bound directly rather than through `set_input_buffer`, which would
        // renumber a second time.
        let ptr = buffer.raw_ptr() as usize;
        self.wait_for_buffer(ptr);
        {
            let mut s = self.state.lock().unwrap();
            s.next_inputs.insert(ptr);
            s.all_inputs.insert(ptr);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.as_ref()), 0, index)
        }
        // The staging buffer must stay alive until the dispatch it feeds has
        // completed, not merely until it is encoded. It is parked here and
        // handed to the caller by `end_param_capture`, which holds it across
        // the dispatch -- releasing it at capture-close would drop it while the
        // GPU may still be reading, which is the in-flight-reuse failure
        // `DESIGN.md` §2.3.8b describes and no fence can see.
        self.param_capture.lock().unwrap().staged.push(buffer);
        true
    }

    /// The index a buffer should actually bind at, given any scalars already
    /// diverted out of the argument list ahead of it.
    ///
    /// Returns the caller's own index unless a capture is open.
    #[inline(always)]
    fn capture_buffer_index(&self, index: usize) -> usize {
        if !self.capturing.load(Ordering::Relaxed) {
            return index;
        }
        let mut cap = self.param_capture.lock().unwrap();
        let slot = cap.next_buffer;
        cap.next_buffer += 1;
        slot
    }

    /// Begin capturing scalars into a packed-params block.
    ///
    /// Scoped rather than persistent: [`Self::end_param_capture`] returns the
    /// bytes and closes it, so a capture cannot leak into the next dispatch.
    pub fn begin_param_capture(&self) {
        let mut cap = self.param_capture.lock().unwrap();
        cap.bytes.clear();
        cap.staged.clear();
        // Slot 0 is the params buffer, bound by the caller after the capture
        // closes, so the first real buffer goes to 1.
        cap.next_buffer = 1;
        drop(cap);
        self.capturing.store(true, Ordering::Relaxed);
    }

    /// Close a capture opened by [`Self::begin_param_capture`], returning the
    /// packed bytes and any buffers staged for arrays.
    ///
    /// The caller must hold the returned buffers until the dispatch is
    /// complete; see [`Self::capture_array`].
    ///
    /// The trailing pad matters: C++ pads a struct up to its own alignment, so
    /// `sizeof` is always a multiple of `alignof`. Without it a `{u64,u32}`
    /// would ship 12 bytes where the kernel reads 16.
    pub fn end_param_capture(&self, align: usize) -> (Vec<u8>, Vec<Buffer>) {
        self.capturing.store(false, Ordering::Relaxed);
        let mut cap = self.param_capture.lock().unwrap();
        let mut bytes = std::mem::take(&mut cap.bytes);
        let staged = std::mem::take(&mut cap.staged);
        while align != 0 && !bytes.len().is_multiple_of(align) {
            bytes.push(0);
        }
        (bytes, staged)
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        trace::record_pipeline(pipeline.name().unwrap_or("<unnamed>"));
        self.note_pipeline(pipeline);
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    /// Record the pipeline for dispatch attribution, when anyone is listening.
    #[inline(always)]
    fn note_pipeline(&self, pipeline: &ComputePipeline) {
        if self.executor.is_classical() {
            return;
        }
        self.executor.will_set_pipeline(pipeline);
        self.state.lock().unwrap().current_pipeline = pipeline.name().map(Arc::from);
    }

    /// Insert a memory barrier at buffers scope.
    pub fn insert_memory_barrier(&self) {
        self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// Wait for a fence before continuing execution.
    pub fn wait_for_fence(&self, fence: &Fence) {
        self.raw.waitForFence(fence.raw());
    }

    /// Update a fence after commands complete.
    pub fn update_fence(&self, fence: &Fence) {
        self.raw.updateFence(fence.raw());
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.updateFence(self.fence.raw());
        self.raw.endEncoding();
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        trace::record_pipeline(pipeline.name().unwrap_or("<unnamed>"));
        self.note_pipeline(pipeline);
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

/// RAII guard that pops a Metal debug group on drop. Debug groups are a stack
/// scoped to the push/pop range, so each dispatch is attributed correctly on
/// the shared concurrent encoder where `set_label` cannot.
#[cfg(feature = "debug-labels")]
pub struct DebugGroupGuard<'a> {
    encoder: &'a ComputeCommandEncoder,
}

#[cfg(feature = "debug-labels")]
impl Drop for DebugGroupGuard<'_> {
    fn drop(&mut self) {
        self.encoder.raw.popDebugGroup();
    }
}

#[cfg(feature = "debug-labels")]
impl ComputeCommandEncoder {
    /// Push a Metal debug group scoped to the returned guard.
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> DebugGroupGuard<'_> {
        self.raw.pushDebugGroup(&NSString::from_str(label));
        DebugGroupGuard { encoder: self }
    }
}

pub struct BlitCommandEncoder {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    /// Per-encoder fence, updated at end_encoding.
    fence: Arc<Fence>,
    /// Shared global cross-encoder output map.
    prev_ce_outputs: PrevCeOutputs,
    /// Buffer pointers written by this blit encoder (registered in global map at end_encoding).
    tracked_outputs: Vec<usize>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder {
            raw,
            fence,
            prev_ce_outputs,
            tracked_outputs: Vec::new(),
        }
    }

    /// Wait for a fence before continuing execution.
    pub fn wait_for_fence(&self, fence: &Fence) {
        self.raw.waitForFence(fence.raw());
    }

    /// Update a fence after commands complete.
    pub fn update_fence(&self, fence: &Fence) {
        self.raw.updateFence(fence.raw());
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;

        // Signal this blit encoder's fence after all blit commands complete
        self.update_fence(&self.fence);
        self.raw.endEncoding();

        // Register outputs so subsequent encoders can wait.
        {
            let mut map = self.prev_ce_outputs.lock().unwrap();
            for &out in &self.tracked_outputs {
                map.insert(out, Arc::clone(&self.fence));
            }
        }
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    /// Wait on the last writer of each of `ptrs`, emitting each distinct fence
    /// once.
    ///
    /// `Commands::blit_command_encoder` already waits on every fence in
    /// `live_fences` before handing this encoder out, which covers every
    /// *compute* encoder that has ended. It does not cover a prior *blit*:
    /// `end_encoding` below registers this encoder's outputs in
    /// `prev_ce_outputs` but never adds its fence to `live_fences`. So a
    /// blit-after-blit dependency is recorded only in the map, and consulting
    /// the map here is what closes that gap.
    ///
    /// Metal tolerates a repeated `waitForFence`, but `copy_from_buffer` passes
    /// two buffers that often share a writer, so the dedup avoids emitting the
    /// same wait twice on the common path.
    fn wait_for_last_writers(&self, ptrs: &[usize]) {
        use objc2_metal::MTLBlitCommandEncoder as _;

        let fences: Vec<Arc<Fence>> = {
            let map = self.prev_ce_outputs.lock().unwrap();
            let mut out: Vec<Arc<Fence>> = Vec::new();
            for ptr in ptrs {
                if let Some(f) = map.get(ptr) {
                    if !out.iter().any(|seen| Arc::ptr_eq(seen, f)) {
                        out.push(Arc::clone(f));
                    }
                }
            }
            out
        };
        for fence in fences {
            self.raw.waitForFence(fence.raw());
        }
    }

    /// Copy bytes from src to dst, ordered after the last writer of *either*.
    ///
    /// The source wait is the obvious one: the copy reads it. The destination
    /// wait matters because a copy's destination is typically a buffer the pool
    /// has just recycled, which is where a pending writer is most likely -- and
    /// under `HazardTrackingModeUntracked` a missed dependency corrupts
    /// silently rather than failing (`DESIGN.md` §3.5). The sibling
    /// `fill_buffer` already waited on its destination; this did not, and the
    /// asymmetry was not deliberate.
    ///
    /// Measured on LFM2 decode, the destination has a registered writer in 0 of
    /// its calls, so this closes a hole rather than removing an observed bug.
    ///
    /// This does **not** fix the grouped-convolution corruption: that is the
    /// buffer pool aliasing two tensors onto one in-flight allocation, which no
    /// fence can observe -- at the aliasing instant no encoder has bound the
    /// buffer, and by the time one does it looks freshly allocated
    /// (`DESIGN.md` §2.3.8b). Measured at 11/30 unstable with and without.
    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        let src_ptr = src_buffer.raw_ptr() as usize;
        let dst_ptr = dst_buffer.raw_ptr() as usize;
        self.wait_for_last_writers(&[src_ptr, dst_ptr]);

        self.tracked_outputs.push(dst_ptr);

        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        let ptr = buffer.raw_ptr() as usize;
        self.wait_for_last_writers(&[ptr]);
        self.tracked_outputs.push(ptr);

        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}
