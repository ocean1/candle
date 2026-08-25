use crate::{DType, Result};

#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        Arena, ArenaCounters, ArenaLayout, ArenaRecorder, BlitCommandsGuard, Buffer, BufferPool,
        Commands, CommandsGuard, Device, MTLResourceOptions, PoolCounters, PoolOccupancySnapshot,
        PooledBuffer, ResidencySet, StepPlan,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use super::MetalError;

/// Unique identifier for metal devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<Commands>,

    /// Shared-storage buffer pool (`RESOURCE_OPTIONS`), for buffers the CPU
    /// also reads.
    ///
    /// Buffers return themselves here when their last handle drops; see
    /// [`candle_metal_kernels::metal::buffer_pool`]. Reclamation is automatic
    /// and requires nothing of callers.
    pub(crate) buffers: BufferPool,

    /// Private-storage pool (`PRIVATE_RESOURCE_OPTIONS`, StorageModePrivate on
    /// macOS). Intermediate compute buffers do not need CPU access, so Private
    /// avoids coherency overhead.
    ///
    /// Despite the name this is the *activation* path, and it is where reuse
    /// matters most: it takes every intermediate a forward pass produces. It
    /// also holds the resident weights, because `to_dtype` on load allocates
    /// through it -- but those are held by live tensors for the process
    /// lifetime, so they are never in the free list and never looked at by a
    /// lookup. Reclaimability separates them, not pool identity.
    pub(crate) private_buffers: BufferPool,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
    /// Residency set registered on the command queue.
    pub(crate) residency_set: Arc<ResidencySet>,

    /// The activation arena, when one is installed.
    ///
    /// `None` is the default and is the classical path: every allocation goes
    /// to the pool exactly as before, and the added cost is one `Option` test
    /// per allocation. Installing an arena changes **where an activation buffer
    /// comes from**, not how the pool decides a buffer is free (`DESIGN.md`
    /// §9.2a) -- the pool's `acquire`, `release`, free list and epoch gate are
    /// untouched.
    ///
    /// `RwLock` rather than `OnceLock` because the layout is selectable at run
    /// time: §9.3's parity check needs to run the same model under the packed
    /// layout and under the non-aliasing reference, and keeping both installable
    /// is the same discipline `ParamStyle` follows for binding styles (§11.3b).
    pub(crate) arena: Arc<RwLock<Option<Arena>>>,

    /// Observes one decode step's allocations so a plan can be derived from it.
    ///
    /// `None` except while recording. Candle is eager, so nothing declares the
    /// activation set up front -- but the dispatch sequence is stable
    /// (§11.1a.1), so one observed step describes the rest. This is §11.1a's
    /// record-then-replay, applied to allocation.
    pub(crate) arena_recorder: Arc<Mutex<Option<ArenaRecorder>>>,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);
// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = RESOURCE_OPTIONS;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    /// Destroys every buffer currently sitting in a free list.
    ///
    /// This is the old `drop_unused_buffers` sweep, and the change is that it
    /// is no longer on the allocation path. It used to run on every
    /// `wait_until_completed`, walking every buffer in both pools to discover
    /// by `strong_count` which had become free -- so the pool oscillated
    /// between growing and being swept, and the sweep *destroyed* buffers that
    /// would immediately be re-created by the next allocation.
    ///
    /// Now a free buffer is already known to be free, so this exists only to
    /// hand memory back under pressure. Nothing calls it on the hot path.
    pub fn trim_unused_buffers(&self) {
        for buffer in self.buffers.trim() {
            self.residency_set.remove(&buffer);
        }
        for buffer in self.private_buffers.trim() {
            self.residency_set.remove(&buffer);
        }
    }

    /// Scan and reuse counters for both pools. See `buffer_pool`.
    pub fn pool_counters(&self) -> (PoolCounters, PoolCounters) {
        (self.buffers.counters(), self.private_buffers.counters())
    }

    pub fn reset_pool_counters(&self) {
        self.buffers.reset_counters();
        self.private_buffers.reset_counters();
    }

    /// Live and free occupancy for both pools, shared first.
    pub fn pool_occupancy(&self) -> (PoolOccupancySnapshot, PoolOccupancySnapshot) {
        (self.buffers.occupancy(), self.private_buffers.occupancy())
    }

    pub fn command_encoder<'a>(&'a self) -> Result<CommandsGuard<'a>> {
        let command_encoder = self.commands.command_encoder().map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>> {
        let command_encoder = self
            .commands
            .blit_command_encoder()
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        self.commands
            .wait_until_completed()
            .map_err(MetalError::from)?;
        // No sweep here any more. Buffers freed while this was waiting have
        // already returned themselves to their pool, so there is nothing to
        // discover -- and destroying them, which is what the sweep did, only
        // forced the next allocation to re-create them.
        //
        // Still a drain, though, and it is not a scan: everything is complete
        // now by definition, so this hands back whatever was waiting on the GPU
        // without depending on a completion handler having been delivered yet.
        // Metal dispatches those asynchronously, so a caller that synchronizes
        // and immediately allocates would otherwise miss buffers that are
        // provably free.
        self.drain_completed_buffers();
        Ok(())
    }

    /// Returns buffers whose GPU work has finished to their free lists.
    fn drain_completed_buffers(&self) {
        self.buffers.drain_completed();
        self.private_buffers.drain_completed();
    }

    /// Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks.
    pub fn flush_and_wait_current(&self) -> Result<()> {
        self.commands
            .flush_and_wait_current()
            .map_err(MetalError::from)?;
        self.drain_completed_buffers();
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Registers buffers in the device's residency set, keeping them
    /// permanently GPU-resident instead of paying per-command-buffer residency
    /// bookkeeping. Useful for buffers candle did not allocate, e.g.
    /// `newBufferWithBytesNoCopy` views over an mmap'd weights file. No-op on
    /// systems without residency-set support.
    pub fn register_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.insert_batch(bufs);
    }

    /// Unregisters buffers previously passed to `register_buffers`, releasing
    /// the set's retain so they can be deallocated. Only unregister buffers
    /// you registered yourself, after GPU work referencing them has completed.
    pub fn unregister_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.remove_batch(bufs);
    }

    /// Returns a builder for buffer allocation. See `BufferBuilder`.
    pub fn new_buffer_builder(&self) -> BufferBuilder<'_> {
        BufferBuilder::new(self)
    }

    /// The installed arena, if any. Cheap: one lock and an `Option` clone.
    fn arena_slot(&self) -> Option<Arena> {
        self.arena.read().ok().and_then(|a| a.clone())
    }

    /// Install an activation arena built from `plan`, replacing any previous
    /// one.
    ///
    /// Allocates one `MTLBuffer` of the plan's size and registers it in the
    /// residency set -- §9.2's "residency is a CPU-side fact established once":
    /// one buffer in the set rather than 674.
    ///
    /// The arena is *additive* (§9.2a). Nothing about the pool changes, no
    /// caller is asked to release anything, and removing the arena with
    /// [`Self::clear_arena`] returns the device to the classical path exactly.
    pub fn install_arena(&self, plan: StepPlan, layout: ArenaLayout) -> Result<()> {
        let bytes = plan.arena_bytes().max(1);
        let base = self
            .device
            .new_buffer(bytes, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&base);
        let arena = Arena::new(&self.private_buffers, base, plan, layout)
            .map_err(|e| MetalError::Message(format!("arena plan rejected: {e}")))?;
        if let Ok(mut slot) = self.arena.write() {
            *slot = Some(arena);
        }
        Ok(())
    }

    /// Remove the arena. Subsequent allocations all take the pool path.
    pub fn clear_arena(&self) {
        if let Ok(mut slot) = self.arena.write() {
            *slot = None;
        }
    }

    pub fn arena(&self) -> Option<Arena> {
        self.arena_slot()
    }

    /// Mark the start of a decode step, resetting the arena's allocation
    /// ordinal.
    ///
    /// The ordinal is what ties an allocation to a slot, so it has to restart at
    /// the same point every token. No-op when no arena is installed.
    pub fn begin_decode_step(&self) {
        if let Some(a) = self.arena_slot() {
            a.begin_step();
        }
    }

    /// Mark the end of a decode step. Allocations outside a step take the pool
    /// path, which is what keeps prefill and setup on the classical route.
    pub fn end_decode_step(&self) {
        if let Some(a) = self.arena_slot() {
            a.end_step();
        }
    }

    pub fn arena_counters(&self) -> Option<ArenaCounters> {
        self.arena_slot().map(|a| a.counters())
    }

    /// Begin observing allocations so a plan can be derived from one step.
    ///
    /// Record over a *steady-state* decode step, not the first one: the first
    /// token after prefill allocates a different set, and #68 plans from
    /// `decode[1]` for the same reason.
    pub fn begin_arena_recording(&self) {
        if let Ok(mut g) = self.arena_recorder.lock() {
            *g = Some(ArenaRecorder::new());
        }
    }

    /// Stop observing and build a plan from what was seen.
    ///
    /// Returns `None` if nothing was recorded, which is a real outcome worth
    /// distinguishing from an empty plan -- it means recording was never begun
    /// or the step allocated nothing.
    pub fn finish_arena_recording(&self, layout: ArenaLayout) -> Option<StepPlan> {
        let mut g = self.arena_recorder.lock().ok()?;
        let rec = g.take()?;
        if rec.is_empty() {
            return None;
        }
        Some(rec.plan(layout))
    }

    /// Record one decode step and install the resulting arena.
    ///
    /// The whole sequence a caller needs: observe a steady-state step, derive
    /// the plan, allocate the arena, and switch to it. `step` runs the model for
    /// exactly one token.
    pub fn record_and_install_arena<F>(&self, layout: ArenaLayout, step: F) -> Result<bool>
    where
        F: FnOnce() -> Result<()>,
    {
        self.begin_arena_recording();
        let outcome = step();
        // Stop recording whether or not the step succeeded, so a failure cannot
        // leave the recorder attached and quietly accumulating.
        let plan = self.finish_arena_recording(layout);
        outcome?;
        match plan {
            Some(plan) => {
                self.install_arena(plan, layout)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Creates a new buffer (not necessarily zeroed).
    ///
    /// Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
    /// Falls back to StorageModeShared on iOS where Private is not always available.
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<PooledBuffer>> {
        let size = element_count * dtype.size_in_bytes();
        // Layer 3 (`DESIGN.md` §9.2a). The arena is offered the allocation
        // first; it serves it only inside a decode step, only while the plan
        // still has an ordinal, and only when the size matches what that ordinal
        // was planned for. Everything else -- prefill, model setup, and every
        // session-state allocation whose size grows with `kv_len` -- falls
        // through to the pool below, unchanged.
        if let Some(arena) = self.arena_slot() {
            if let Some(b) = arena.acquire(size) {
                return Ok(b);
            }
        }
        let buffer = match self.private_buffers.acquire(size) {
            Some(b) => b,
            None => {
                let alloc = buf_size(size);
                let new_buffer = self
                    .device
                    .new_buffer(alloc, PRIVATE_RESOURCE_OPTIONS)
                    .map_err(MetalError::from)?;
                self.residency_set.insert(&new_buffer);
                self.private_buffers.adopt(new_buffer, alloc)
            }
        };
        Ok(self.note_for_recording(buffer, size))
    }

    /// While a plan is being recorded, note this allocation and arrange to be
    /// told when it dies.
    ///
    /// The pair is the value's liveness interval, and the *value* is one
    /// allocation event -- never a buffer. `DESIGN.md` §9.2c: candle's pool puts
    /// 60 unrelated values in one buffer within a single token, so a recorder
    /// that keyed on buffer identity would merge them and invent lifetimes;
    /// #68's first planner did that and overstated the arena 52-fold. Nothing
    /// here consults which buffer was handed out.
    fn note_for_recording(&self, buffer: Arc<PooledBuffer>, size: usize) -> Arc<PooledBuffer> {
        let Ok(mut guard) = self.arena_recorder.lock() else {
            return buffer;
        };
        let Some(rec) = guard.as_mut() else {
            return buffer;
        };
        // The token names this allocation event. A fresh one every time, so two
        // values that share a buffer still get separate intervals.
        let token = rec.len() as u64;
        rec.record_alloc(token, size);
        drop(guard);

        // A death hook can only be attached to a handle nobody else holds yet.
        // A pool hit hands back an `Arc` that is uniquely owned at this instant,
        // so unwrapping it is sound; if it ever were not, recording simply skips
        // this value rather than failing, and the value is then treated as live
        // to the end of the step -- the conservative direction.
        let recorder = Arc::clone(&self.arena_recorder);
        match Arc::try_unwrap(buffer) {
            Ok(b) => Arc::new(b.on_death(move |_| {
                if let Ok(mut g) = recorder.lock() {
                    if let Some(rec) = g.as_mut() {
                        rec.record_free(token);
                    }
                }
            })),
            Err(shared) => shared,
        }
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<PooledBuffer>> {
        let size = element_count * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&buffer);
        // Deliberately not pooled: this exists for persistent buffers, so it
        // returns a handle with no pool behind it. Dropping it frees the buffer
        // outright rather than offering it for reuse.
        Ok(Arc::new(PooledBuffer::unpooled(buffer, size)))
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<PooledBuffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&new_buffer);

        // Deliberately not pooled.
        //
        // This path always calls `newBufferWithBytes` and never consults the
        // free list, so a buffer parked here is only reachable by an
        // `allocate_buffer` request for the same size -- and `allocate_buffer`
        // has three callers, all on quantized dequantize paths. Nothing in an
        // f16 forward pass ever asks for one.
        //
        // In practice these are weight tensors staged from safetensors, which
        // `to_dtype` immediately converts into `private_buffers`; the staging
        // buffer is then dropped and never wanted again. Measured on an LFM2
        // load that is 276 buffers holding 5145 MB. The old allocator reclaimed
        // them only as a side effect of its sweep, so with the sweep off the
        // allocation path, pooling them would double the footprint and undo
        // issue #8's 7731 -> 5509 MB.
        //
        // Freeing on drop keeps that reclamation without reintroducing a scan.
        // Note this is a property of the *path*, not of buffer size: these
        // average 18.6 MB, so a size threshold does not separate them from
        // activations.
        Ok(Arc::new(PooledBuffer::unpooled(new_buffer, size)))
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<PooledBuffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let mut blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        /*
        // Alternative impl
        if size_in_bytes > 0 {
            let encoder = self.command_encoder()?;
            call_const_fill(
                &self.device,
                &encoder,
                &self.kernels,
                "fill_u8",
                size_in_bytes,
                &buffer,
                0u8,
            )
            .map_err(crate::Error::wrap)?;
        }
        */
        Ok(buffer)
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<PooledBuffer>> {
        if let Some(b) = self.buffers.acquire(size) {
            return Ok(b);
        }
        let size = buf_size(size);
        let new_buffer = self
            .device
            .new_buffer(size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&new_buffer);
        Ok(self.buffers.adopt(new_buffer, size))
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }

        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }
}

/// Alignment applied to every buffer allocation.
///
/// This is the Apple Silicon cache line. Aligning to it means every buffer
/// starts on a line boundary, so no two buffers share a line and the memory
/// controller can coalesce accesses cleanly. It also covers the alignment any
/// dtype could need: the kernels use `float4`/`half4` (16 bytes) widely and
/// quantized block structures are wider still, all well under 128.
const BUFFER_ALIGNMENT: usize = 128;

/// Size to allocate for a request of `size` bytes.
///
/// Previously this rounded up to the next power of two, which wastes up to 2x
/// per buffer. That is not hypothetical: for shapes that sit just above a
/// power of two -- a 292-token KV cache, or an FFN intermediate of 10752 --
/// the rounding costs 75% and 52% respectively, and the pool holds the excess
/// for the process lifetime.
///
/// Aligning instead keeps the guarantees that matter (vector alignment, and a
/// size the memory controller likes) without the waste. Reuse is unaffected:
/// `find_available_buffer` accepts any pooled buffer at least as large as the
/// request, so a slightly larger buffer still satisfies a smaller one.
fn buf_size(size: usize) -> usize {
    // Never return 0: Metal rejects zero-length buffers, and callers may ask
    // for an empty tensor.
    size.max(1).next_multiple_of(BUFFER_ALIGNMENT)
}

/// Applies the [`BufferBuilder`] label, clearing any stale label on a reused pooled buffer.
#[cfg(feature = "metal-debug-labels")]
#[inline]
fn buffer_label(buffer: &Buffer, label: Option<&str>) {
    buffer.set_label(label.unwrap_or("unlabeled"));
}
#[cfg(not(feature = "metal-debug-labels"))]
#[inline]
fn buffer_label(_buffer: &Buffer, _label: Option<&str>) {}

type DataUpload<'a> = Box<dyn FnOnce(&MetalDevice) -> Result<Arc<PooledBuffer>> + 'a>;

enum BufferInit<'a> {
    Typed { elem_count: usize, dtype: DType },
    Size(usize),
    Zeros(usize),
    Data(DataUpload<'a>),
}

/// Builder for `MTLBuffer` allocations; pool reuse handled by [`MetalDevice`].
pub struct BufferBuilder<'a> {
    device: &'a MetalDevice,
    label: Option<&'a str>,
}

/// [`BufferBuilder`] with an init kind set; `build()` lives here.
pub struct ReadyBufferBuilder<'a> {
    device: &'a MetalDevice,
    init: BufferInit<'a>,
    label: Option<&'a str>,
}

impl<'a> BufferBuilder<'a> {
    fn new(device: &'a MetalDevice) -> Self {
        Self {
            device,
            label: None,
        }
    }

    /// Allocate elem_count * dtype size bytes, uninitialized, private storage.
    pub fn with_size_for(self, elem_count: usize, dtype: DType) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Typed { elem_count, dtype })
    }

    /// Allocate size bytes, uninitialized, shared storage.
    pub fn with_size(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Size(size))
    }

    /// Allocate size bytes, zero-filled, shared storage. Pool rounding may make
    /// the allocation larger than size; the extra bytes are also zeroed.
    pub fn with_zeros(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Zeros(size))
    }

    /// Allocate a shared buffer initialized from data. Always allocates; does not
    /// reuse the pool.
    pub fn with_data<T>(self, data: &'a [T]) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Data(Box::new(move |device| {
            device.new_buffer_with_data(data)
        })))
    }

    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[inline]
    fn ready(self, init: BufferInit<'a>) -> ReadyBufferBuilder<'a> {
        ReadyBufferBuilder {
            device: self.device,
            init,
            label: self.label,
        }
    }
}

impl<'a> ReadyBufferBuilder<'a> {
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn build(self) -> Result<Arc<PooledBuffer>> {
        let buffer = match self.init {
            BufferInit::Typed { elem_count, dtype } => {
                self.device.new_buffer(elem_count, dtype, "")?
            }
            BufferInit::Size(size) => self.device.allocate_buffer(size)?,
            BufferInit::Zeros(size) => self.device.allocate_zeros(size)?,
            BufferInit::Data(upload) => upload(self.device)?,
        };
        buffer_label(&buffer, self.label);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buf_size_is_aligned() {
        for size in [1usize, 2, 3, 127, 128, 129, 1000, 4096, 21504, 299008] {
            let got = buf_size(size);
            assert!(got >= size, "buf_size({size}) = {got} is too small");
            assert_eq!(
                got % BUFFER_ALIGNMENT,
                0,
                "buf_size({size}) = {got} is not {BUFFER_ALIGNMENT}-aligned"
            );
            assert!(
                got - size < BUFFER_ALIGNMENT,
                "buf_size({size}) = {got} wastes a whole alignment unit"
            );
        }
    }

    #[test]
    fn test_buf_size_never_zero() {
        // Metal rejects zero-length buffers, and an empty tensor asks for one.
        assert_eq!(buf_size(0), BUFFER_ALIGNMENT);
    }

    #[test]
    fn test_buf_size_avoids_power_of_two_waste() {
        // Shapes that sit just above a power of two are the ones that suffered:
        // a 292-token KV cache and an FFN intermediate both rounded up hard.
        assert_eq!(buf_size(299008), 299008); // was 524288, 75% waste
        assert_eq!(buf_size(21504), 21504); // was 32768, 52% waste
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element; a scalar tensor requests a
        // 2-byte buffer. It must not be rounded down.
        assert!(buf_size(2) >= 2);
    }
}
