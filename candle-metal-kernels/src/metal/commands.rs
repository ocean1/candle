use crate::metal::{
    executor::ExecutorSlot, BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder,
    ComputePipeline, Device, Fence, GpuClock, PrevCeOutputs, ResidencySet,
};
use crate::MetalKernelError;
use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;

/// Callbacks run after each command buffer completes. See
/// [`Commands::on_command_buffer_complete`].
type CompletionSubscribers = Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>>;

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: MutexGuard<'a, EntryState>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.as_ref().set_compute_pipeline_state(pipeline);
    }

    #[cfg(feature = "debug-labels")]
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> crate::metal::DebugGroupGuard<'_> {
        self.as_ref().debug_group(label)
    }
}

/// RAII guard for blit command encoder operations.
pub struct BlitCommandsGuard<'a> {
    _guard: MutexGuard<'a, EntryState>,
    state: BlitCommandEncoder,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        &self.state
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        &mut self.state
    }
}

impl BlitCommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.as_mut()
            .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.as_mut().fill_buffer(buffer, range, value);
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        self.as_ref().end_encoding();
    }
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
}

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            current: cb,
            in_flight: vec![],
            current_encoder: None,
        }
    }
}

pub struct Commands {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    device: Device,
    /// Global cross-encoder output map. Maps buffer pointer to the fence of the last encoder
    /// that wrote it, enabling cross-command-buffer ordering for HazardTrackingModeUntracked.
    prev_ce_outputs: PrevCeOutputs,
    /// The distinct fences currently referenced by `prev_ce_outputs`, each with
    /// the number of buffers pointing at it.
    ///
    /// A new encoder waits on every distinct prior fence. Deriving that set from
    /// `prev_ce_outputs` means scanning one entry per live buffer and allocating
    /// a HashSet to deduplicate them, on every encoder. Maintaining it here makes
    /// encoder creation proportional to the number of distinct fences instead,
    /// which is typically small.
    live_fences: Arc<Mutex<Vec<LiveFence>>>,
    /// How far the GPU has got through the command buffers submitted here.
    ///
    /// Shared with the buffer pools, which use it to decide when a released
    /// buffer is safe to hand out again. This is the only place it advances.
    clock: Arc<GpuClock>,
    /// Run after each command buffer completes, once per command buffer.
    ///
    /// The buffer pools subscribe here to sweep in what that command buffer was
    /// holding up. Kept as a callback rather than a direct reference to the
    /// pools so that `Commands` does not need to know they exist -- it owns the
    /// clock, not the things that read it.
    on_complete: CompletionSubscribers,
    /// How dispatches reach the GPU (`DESIGN.md` §11.1). Handed to every
    /// compute encoder this creates.
    ///
    /// `Mutex` rather than a plain field because installing an executor is a
    /// setup-time action on a `&Commands` that is already shared; it is never
    /// touched per dispatch, because the encoder holds its own `Arc` to the
    /// slot from creation.
    executor: Mutex<Arc<ExecutorSlot>>,
}

/// A fence still referenced by `prev_ce_outputs`, with its buffer count.
struct LiveFence {
    /// Identity of the fence, used to match without comparing Arc contents.
    ptr: usize,
    fence: Arc<Fence>,
    buffers: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: &ResidencySet,
    ) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
        }

        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;

        Ok(Self {
            state: Mutex::new(EntryState::new(cb)),
            compute_count: AtomicUsize::new(0),
            command_queue,
            compute_per_buffer,
            device,
            prev_ce_outputs: Arc::new(Mutex::new(HashMap::new())),
            live_fences: Arc::new(Mutex::new(Vec::new())),
            clock: Arc::new(GpuClock::new()),
            on_complete: Arc::new(Mutex::new(Vec::new())),
            executor: Mutex::new(Arc::new(ExecutorSlot::Classical)),
        })
    }

    /// Install the executor that subsequent compute encoders will submit
    /// through (`DESIGN.md` §11.1).
    ///
    /// Takes effect on the next encoder created, not on one already open: an
    /// encoder holds its slot from creation so that the per-dispatch path never
    /// has to take this lock. Call before encoding the work being measured.
    pub fn set_executor(&self, executor: Arc<ExecutorSlot>) {
        if let Ok(mut slot) = self.executor.lock() {
            *slot = executor;
        }
    }

    /// The executor currently installed.
    pub fn executor(&self) -> Arc<ExecutorSlot> {
        self.executor
            .lock()
            .map(|e| Arc::clone(&e))
            .unwrap_or_else(|_| Arc::new(ExecutorSlot::Classical))
    }

    /// The clock the buffer pools decide reuse against.
    pub fn clock(&self) -> Arc<GpuClock> {
        Arc::clone(&self.clock)
    }

    /// Registers `f` to run after every command buffer completes.
    ///
    /// Must be called during device setup, before any work is encoded:
    /// subscribers added later would miss the command buffers already in
    /// flight, and a buffer parked against one of those epochs would never be
    /// swept in.
    pub fn on_command_buffer_complete<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        if let Ok(mut subs) = self.on_complete.lock() {
            subs.push(Arc::new(f));
        }
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            // No blanket wait here. HazardTrackingModeUntracked means Metal does
            // not flush GPU caches at encoder boundaries, so ordering is ours to
            // enforce -- but the encoder already records every buffer it binds,
            // so it waits per buffer in set_input_buffer / set_output_buffer
            // instead. Waiting on every live fence would order this encoder
            // after work it never touches.
            let enc = state_guard.current.compute_command_encoder_with_executor(
                &fence,
                &self.prev_ce_outputs,
                &self.executor(),
            );
            // Hazard state is per encoder, so a barrier count is only
            // interpretable against the session boundaries (`DESIGN.md` §9.2e).
            crate::metal::trace::record_encoder_begin();
            crate::metal::profile::record_encoder();
            state_guard.current_encoder = Some(enc);
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        if let Some(enc) = state_guard.current_encoder.take() {
            self.end_encoding(enc);
        }

        let fence = Arc::new(Fence::new(&self.device));
        let encoder = state_guard
            .current
            .blit_command_encoder(&fence, &self.prev_ce_outputs);

        // Wait for all prior encoder fences before any blit commands execute.
        // Required for HazardTrackingModeUntracked: GPU caches are not auto-flushed.
        //
        // This covers every *compute* encoder that has ended, because
        // `Commands::end_encoding` adds its fence to `live_fences`. It does not
        // cover a prior *blit*: `BlitCommandEncoder::end_encoding` registers its
        // outputs in `prev_ce_outputs` but never adds its fence here. That is
        // why the per-buffer waits in `copy_from_buffer` / `fill_buffer` are not
        // redundant with this one.
        {
            let fences = self.live_fences.lock().unwrap();
            for live in fences.iter() {
                encoder.wait_for_fence(&live.fence);
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: encoder,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        let to_wait = {
            let mut state = self.state.lock()?;
            if self.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&mut state, 0)?;
            }
            std::mem::take(&mut state.in_flight)
        };

        // Wait only on the last CB. Metal executes CBs in queue order, so all earlier
        // CBs are guaranteed complete when the last one is. Calling waitUntilCompleted on
        // each CB individually pays OS notification latency (~1-2ms) N times unnecessarily.
        if let Some(last) = to_wait.last() {
            Self::ensure_completed(last)?;
        }
        // Check earlier CBs for errors (no need to block — they're already done).
        for cb in &to_wait[..to_wait.len().saturating_sub(1)] {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
        }

        self.prev_ce_outputs.lock()?.clear();
        self.live_fences.lock()?.clear();

        Ok(())
    }

    /// Commit the current command buffer and wait on that specific buffer, for CPU readbacks.
    /// [`Self::wait_until_completed`] waits on the last in-flight buffer, which a concurrent
    /// `flush_and_wait` on another thread may already have taken, returning before our work ran.
    pub fn flush_and_wait_current(&self) -> Result<(), MetalKernelError> {
        let cb = {
            let mut state = self.state.lock()?;
            self.commit_swap_locked(&mut state, 0)?;
            state.in_flight.last().cloned()
        };
        if let Some(cb) = cb {
            Self::ensure_completed(&cb)?;
            // queue is FIFO: everything committed before cb is done too
            let mut state = self.state.lock()?;
            state
                .in_flight
                .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }
        Ok(())
    }

    fn commit_swap_locked(
        &self,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }

        // Close the epoch this command buffer holds and arrange for the pools to
        // hear when it finishes. Every closed epoch must be reported exactly
        // once, or a buffer parked against it waits forever -- so the handler is
        // attached on the same branch that commits, and an already-committed
        // buffer (which Metal will not accept a handler for) has its epoch
        // retired here instead.
        //
        // Both happen under the state lock, so no release can observe the new
        // epoch before the old one is accounted for.
        let epoch = self.clock.commit_epoch();
        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                let clock = Arc::clone(&self.clock);
                let subs: Vec<_> = self
                    .on_complete
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_default();
                state.current.on_completion(move || {
                    // Order matters: the clock must show the epoch finished
                    // before subscribers look, or they find nothing drainable
                    // and the buffers wait for an unrelated later completion.
                    clock.mark_completed(epoch);
                    for f in &subs {
                        f();
                    }
                });
                // Registered before commit: GPUStartTime/GPUEndTime only become
                // valid once the buffer completes, so a completion handler is
                // the only place they can be read.
                state.current.record_gpu_time_on_completion();
                state.current.commit();
            }
            // Already committed or finished, so no handler can be attached. It
            // has been waited on or will be by `flush_and_wait`; retiring the
            // epoch now keeps the clock from stalling on an epoch that will
            // never be reported.
            _ => {
                self.clock.mark_completed(epoch);
            }
        }
        let new_cb = create_command_buffer(&self.command_queue)?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                // This path commits a buffer that `commit_swap_locked` never saw,
                // so it needs its own registration or its GPU time goes unrecorded.
                cb.record_gpu_time_on_completion();
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => return Err(Self::cb_error(cb)),
            _ => unreachable!(),
        }

        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(Self::cb_error(cb));
        }

        Ok(())
    }

    fn cb_error(cb: &CommandBuffer) -> MetalKernelError {
        let msg = cb
            .error()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        MetalKernelError::CommandBufferError(msg)
    }

    fn end_encoding(&self, encoder: ComputeCommandEncoder) {
        use objc2_metal::MTLCommandEncoder as _;
        use objc2_metal::MTLComputeCommandEncoder as _;

        let all_outputs = {
            let s = encoder.state.lock().unwrap();
            s.all_outputs.clone()
        };

        {
            let mut prev_ce_outputs = self.prev_ce_outputs.lock().unwrap();
            let mut fences = self.live_fences.lock().unwrap();
            let this_ptr = Arc::as_ptr(&encoder.fence) as usize;
            let mut claimed = 0usize;

            // Register our outputs so subsequent encoders can wait for us,
            // keeping `live_fences` consistent with the map.
            for output in all_outputs.iter() {
                if let Some(prev) = prev_ce_outputs.insert(*output, encoder.fence.clone()) {
                    // Another fence owned this buffer; it loses one reference.
                    release_fence(&mut fences, Arc::as_ptr(&prev) as usize);
                }
                claimed += 1;
            }

            if claimed > 0 {
                match fences.iter_mut().find(|f| f.ptr == this_ptr) {
                    Some(live) => live.buffers += claimed,
                    None => fences.push(LiveFence {
                        ptr: this_ptr,
                        fence: encoder.fence.clone(),
                        buffers: claimed,
                    }),
                }
            }
        }

        // Signal this encoder's completion fence and end encoding.
        encoder.raw.updateFence(encoder.fence.raw());

        // Schedule cleanup of our output entries once the GPU completes.
        if !all_outputs.is_empty() {
            let fence_for_cleanup = Arc::clone(&encoder.fence);
            let map_for_cleanup = Arc::clone(&self.prev_ce_outputs);
            let fences_for_cleanup = Arc::clone(&self.live_fences);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let mut map = map_for_cleanup.lock().unwrap();
                let mut fences = fences_for_cleanup.lock().unwrap();
                let ptr = Arc::as_ptr(&fence_for_cleanup) as usize;
                for &buf in &all_outputs {
                    if let Some(f) = map.get(&buf) {
                        if Arc::ptr_eq(f, &fence_for_cleanup) {
                            map.remove(&buf);
                            release_fence(&mut fences, ptr);
                        }
                    }
                }
            });
            unsafe {
                encoder
                    .command_buffer
                    .addCompletedHandler(RcBlock::as_ptr(&block))
            };
        }

        encoder.raw.endEncoding();
    }
}

/// Drop one buffer's reference to `ptr`, forgetting the fence once none remain.
fn release_fence(fences: &mut Vec<LiveFence>, ptr: usize) {
    if let Some(idx) = fences.iter().position(|f| f.ptr == ptr) {
        fences[idx].buffers = fences[idx].buffers.saturating_sub(1);
        if fences[idx].buffers == 0 {
            fences.swap_remove(idx);
        }
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
