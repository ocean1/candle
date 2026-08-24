use crate::metal::{trace, Buffer, ComputePipeline, Fence};
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
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
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
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
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
        trace::record_pipeline(pipeline.name());
        self.raw.setComputePipelineState(pipeline.as_ref());
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
        trace::record_pipeline(pipeline.name());
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

    /// Copy bytes from src to dst. Waits on any fence that wrote to src_buffer to ensure
    /// correct ordering for HazardTrackingModeUntracked buffers.
    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        let src_ptr = src_buffer.raw_ptr() as usize;
        let fence_to_wait = {
            let map = self.prev_ce_outputs.lock().unwrap();
            map.get(&src_ptr).cloned()
        };
        if let Some(fence) = fence_to_wait {
            use objc2_metal::MTLBlitCommandEncoder as _;
            self.raw.waitForFence(fence.raw());
        }

        self.tracked_outputs.push(dst_buffer.raw_ptr() as usize);

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
        let fence_to_wait = {
            let map = self.prev_ce_outputs.lock().unwrap();
            map.get(&ptr).cloned()
        };
        if let Some(fence) = fence_to_wait {
            use objc2_metal::MTLBlitCommandEncoder as _;
            self.raw.waitForFence(fence.raw());
        }
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
