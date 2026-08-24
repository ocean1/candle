use crate::metal::{Buffer, ComputePipeline, Fence};
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
    sync::{Arc, Mutex},
};

/// Shared cross-encoder output map: maps buffer pointer -> fence of the last encoder that wrote it.
/// Used by subsequent encoders to call waitForFence before reading those buffers.
pub type PrevCeOutputs = Arc<Mutex<HashMap<usize, Arc<Fence>>>>;

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
    /// Name of the pipeline most recently bound, so a dispatch can be attributed
    /// to a kernel when profiling. Only populated when profiling is enabled.
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
        ComputeCommandEncoder {
            raw,
            command_buffer,
            fence,
            state: Arc::new(Mutex::new(EncoderState::new())),
            prev_ce_outputs,
        }
    }

    /// Wait on the fence of `ptr`'s last writer, if any, and only once.
    ///
    /// This replaces waiting on every live fence at encoder creation. The
    /// encoder already records every buffer it binds, so the wait can be
    /// limited to buffers this encoder actually touches.
    fn wait_for_buffer(&self, ptr: usize) {
        use crate::metal::profile::{record_bind_probe, BindProbeOutcome};

        let (fence, map_len) = {
            let map = self.prev_ce_outputs.lock().unwrap();
            // Read the length under the lock the probe already holds, so the
            // instrumentation adds no synchronization of its own. `len` on a
            // `HashMap` is a field read.
            (map.get(&ptr).cloned(), map.len())
        };
        let Some(fence) = fence else {
            record_bind_probe(map_len, BindProbeOutcome::NoPendingWriter);
            return;
        };

        let mut state = self.state.lock().unwrap();
        if state.waited_fences.insert(Arc::as_ptr(&fence) as usize) {
            drop(state);
            self.raw.waitForFence(fence.raw());
            record_bind_probe(map_len, BindProbeOutcome::Waited);
        } else {
            drop(state);
            record_bind_probe(map_len, BindProbeOutcome::AlreadyWaited);
        }
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.auto_barrier();
        self.record_dispatch();
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.auto_barrier();
        self.record_dispatch();
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    /// Attribute this dispatch to the currently bound pipeline, when profiling.
    ///
    /// Both dispatch entry points funnel through here so the count cannot drift
    /// from the number of dispatches actually encoded.
    #[inline]
    fn record_dispatch(&self) {
        if !crate::metal::profile::enabled() {
            return;
        }
        // Without the inventory there is no name to fetch, so the state mutex
        // and the `Arc` clone are both skipped and the count is one atomic.
        if !crate::metal::profile::kernel_inventory_enabled() {
            crate::metal::profile::record_dispatch("");
            return;
        }
        let name = {
            let s = self.state.lock().unwrap();
            s.current_pipeline.clone()
        };
        crate::metal::profile::record_dispatch(name.as_deref().unwrap_or("<unnamed>"));
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
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            // Read-after-write against an earlier encoder: order against that
            // buffer's last writer only.
            self.wait_for_buffer(ptr);
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_inputs.insert(ptr);
            s.all_inputs.insert(ptr);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            // Write-after-write or write-after-read against an earlier encoder.
            self.wait_for_buffer(ptr);
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) || s.prev_inputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_outputs.insert(ptr);
            s.all_outputs.insert(ptr);
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

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.note_pipeline(pipeline);
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    /// Remember which kernel is bound, so the next dispatch can be attributed.
    #[inline]
    fn note_pipeline(&self, pipeline: &ComputePipeline) {
        // Gated on the inventory flag, not merely on profiling: `Arc::from(&str)`
        // allocates and copies, once per pipeline bind, and the timing path has
        // no use for the name. Attribution is opt-in so that measuring the
        // per-bind cost is not itself a per-bind allocation.
        if !crate::metal::profile::kernel_inventory_enabled() {
            return;
        }
        let name = pipeline.name().map(Arc::from);
        self.state.lock().unwrap().current_pipeline = name;
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
    /// Fences already waited on by this encoder, by identity.
    ///
    /// `Commands::blit_command_encoder` walks `live_fences` and waits on each
    /// before handing the encoder out, so this records what that blanket wait
    /// covered. A per-buffer wait can then tell whether it is adding an edge the
    /// blanket wait already has, which is what distinguishes a redundant wait
    /// from the blit-after-blit case `live_fences` cannot see (lloom #25).
    /// Only populated when profiling; the wait itself is emitted regardless.
    waited_fences: Vec<usize>,
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
            waited_fences: Vec::new(),
        }
    }

    /// Wait for a fence before continuing execution.
    pub fn wait_for_fence(&mut self, fence: &Fence) {
        if crate::metal::profile::enabled() {
            self.waited_fences
                .push(fence.raw() as *const _ as *const () as usize);
        }
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

    /// Wait on the last writer of each of `ptrs`, skipping repeats.
    ///
    /// `Commands::blit_command_encoder` already waits on every fence in
    /// `live_fences` before handing this encoder out, which covers every
    /// *compute* encoder that has ended. It does not cover a prior *blit*:
    /// `BlitCommandEncoder::end_encoding` registers its outputs in
    /// `prev_ce_outputs` but never adds its fence to `live_fences`, so a
    /// blit-after-blit dependency is visible only here. That gap is why these
    /// per-buffer waits are not redundant with the blanket one (lloom #25).
    ///
    /// Distinct fences are waited on once. Metal tolerates a repeated
    /// `waitForFence`, but two of the three call shapes below pass two buffers
    /// that frequently share a writer, and the dedup keeps that from emitting a
    /// redundant wait on the common path.
    /// Returns, per input ptr, whether it had a registered writer and whether
    /// that writer was *not* already covered by this encoder's blanket wait.
    fn wait_for_last_writers(&mut self, ptrs: &[usize]) -> Vec<(bool, bool)> {
        use objc2_metal::MTLBlitCommandEncoder as _;

        let (fences, found): (Vec<Arc<Fence>>, Vec<(bool, bool)>) = {
            let map = self.prev_ce_outputs.lock().unwrap();
            let mut out: Vec<Arc<Fence>> = Vec::new();
            let mut found = Vec::with_capacity(ptrs.len());
            for ptr in ptrs {
                match map.get(ptr) {
                    Some(f) => {
                        let raw = f.raw() as *const _ as *const () as usize;
                        let covered = self.waited_fences.contains(&raw);
                        if !out.iter().any(|seen| Arc::ptr_eq(seen, f)) {
                            out.push(Arc::clone(f));
                        }
                        found.push((true, !covered));
                    }
                    None => found.push((false, false)),
                }
            }
            (out, found)
        };
        for fence in fences {
            let raw = fence.raw() as *const _ as *const () as usize;
            if crate::metal::profile::enabled() && !self.waited_fences.contains(&raw) {
                self.waited_fences.push(raw);
            }
            self.raw.waitForFence(fence.raw());
        }
        found
    }

    /// Copy bytes from src to dst, ordered after the last writer of *either*.
    ///
    /// The source wait is the obvious one: the copy reads it. The destination
    /// wait matters because the destination of a copy is typically a buffer the
    /// pool has just recycled, which is exactly the case where a pending writer
    /// is likely -- and under `HazardTrackingModeUntracked` a missed dependency
    /// corrupts silently rather than failing (`DESIGN.md` §3.5). `fill_buffer`
    /// waits on its destination; this did not, and the asymmetry was not
    /// deliberate (lloom #25).
    ///
    /// This does **not** fix lloom #19. That corruption is the buffer pool
    /// aliasing two tensors onto one allocation while the first is in flight,
    /// which no fence can see: at the aliasing instant no encoder has bound the
    /// buffer, and by the time one does it looks freshly allocated
    /// (`DESIGN.md` §2.3.8b). Measured in PR #20 at 11/30 unstable with and
    /// without this wait.
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
        let found = self.wait_for_last_writers(&[src_ptr, dst_ptr]);
        // found[1] is the destination: (had a writer, that writer was not
        // already covered by the blanket `live_fences` wait).
        let (dst_pending, dst_uncovered) = found.get(1).copied().unwrap_or((false, false));
        crate::metal::profile::record_blit_copy(dst_pending, dst_uncovered);

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
