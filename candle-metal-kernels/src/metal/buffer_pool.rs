//! An event-driven buffer pool for the Metal backend.
//!
//! # Why this exists
//!
//! The pool this replaces stored `Arc<Buffer>` in size buckets and decided a
//! buffer was reusable by testing `Arc::strong_count(b) == 1`. Because the pool
//! itself held a strong reference, that predicate was only ever true *between*
//! uses -- and nothing announced the transition. So the count had to be
//! **polled**, at two unrelated moments: a full sweep in `drop_unused_buffers`,
//! and again at every lookup in `find_available_buffer`.
//!
//! That is the actual defect. The linear scan is a consequence: with no event
//! at the instant a buffer becomes free, the only way to find a free one is to
//! look at all of them.
//!
//! # What changes
//!
//! The pool holds `Weak`, the caller holds the only strong reference, and
//! `PooledBuffer::drop` **pushes** the buffer back at the exact moment the last
//! user releases it. Lookup then pops from a free list instead of searching for
//! a free buffer.
//!
//! Reclamation stays automatic and `Drop`-driven. **No caller is required to
//! release a buffer by hand**, which is the property that makes candle's
//! allocate-and-forget ergonomics work, and the reason this is not a manual
//! `free()` API. `PooledBuffer` derefs to `Buffer`, so every existing consumer
//! -- all of which take `&Buffer` -- is unaffected.
//!
//! # Structure
//!
//! ```text
//!   free:    BTreeMap<usize, Vec<Buffer>>  size -> buffers ready to hand out
//!   pending: VecDeque<PendingBuffer>       released, waiting on the GPU
//!   live_buffers / live_bytes: usize       what callers currently hold
//! ```
//!
//! `BTreeMap` rather than `HashMap` because lookup wants the *smallest bucket
//! at least as large as the request*, which is `range(size..).next()`: one
//! ordered probe, not a scan for the minimum. Buckets that hold nothing are
//! never visited, which is what removes the pathology -- in a 400-token
//! generation the old pool accumulated 1345 bucket keys of which 1324 were
//! empty, and walked all of them on every allocation.
//!
//! # Reuse is decided on the GPU clock
//!
//! A buffer becomes reusable when **the GPU is finished with it**, not when the
//! CPU drops its last handle. Those are different instants, and the difference
//! is a correctness bug (issue #19): the CPU routinely runs a long way ahead of
//! the GPU, so a dropped buffer usually still has work outstanding on it. The
//! pool used to hand it straight back, aliasing two unrelated tensors onto one
//! allocation while the first was still being written. Measured, ~90 % of the
//! wrong values that produced were exactly another operation's correct output.
//!
//! No fence can fix that, which is why the fix is here rather than in the
//! encoder. At the instant the pool aliases two tensors, no encoder has bound
//! the buffer, so there is nothing to fence against; by the time one binds it,
//! it looks like an ordinary fresh allocation. The dependency is real and
//! invisible at every point where a fence could be emitted.
//!
//! So the release condition is GPU completion, and the CPU drop is not a second
//! predicate beside it -- it is only the signal that the buffer has no *future*
//! CPU user. One clock decides reuse. See [`GpuClock`] and `PoolInner::release`.

use super::Buffer;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Tracks how far the GPU has got through the command buffers submitted to it.
///
/// Command buffers on one queue complete **in order**, so a single monotone
/// counter is enough: if buffer N has completed, every buffer before it has
/// too. `Commands` already relies on this to wait only on the last in-flight
/// buffer.
///
/// Two counters:
///
/// - `submitted` names the command buffer work is *currently being encoded
///   into*. It is incremented when that buffer is committed and a fresh one
///   takes its place.
/// - `completed` is the highest epoch the GPU has finished.
///
/// A buffer dropped now may have been bound into any command buffer up to and
/// including `submitted`, so `submitted` is the epoch it must outlive. That is
/// conservative -- the buffer may have last been touched much earlier, or never
/// touched at all -- and deliberately so: being wrong in this direction costs a
/// little reuse latency, while being wrong in the other direction corrupts.
#[derive(Debug, Default)]
pub struct GpuClock {
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl GpuClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// The epoch a buffer dropped right now must outlive.
    pub fn current_epoch(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// Opens a new epoch, because the command buffer holding the old one has
    /// been committed. Returns the epoch just closed -- the one whose
    /// completion handler should report back to [`Self::mark_completed`].
    pub fn commit_epoch(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel)
    }

    /// Records that the GPU has finished epoch `epoch`.
    ///
    /// Stored as `epoch + 1` -- a count of finished epochs -- so that the
    /// initial zero means "nothing has completed" rather than "epoch 0 has",
    /// which are different states and would otherwise be indistinguishable.
    ///
    /// Takes a max rather than a store because completion handlers for
    /// different command buffers may be delivered on different threads;
    /// in-order execution guarantees the ordering of the *work*, not of the
    /// notification.
    pub fn mark_completed(&self, epoch: u64) {
        self.completed.fetch_max(epoch + 1, Ordering::AcqRel);
    }

    /// How many epochs have finished. Zero means none have.
    pub fn completed_count(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Whether work encoded in `epoch` is known to have finished.
    fn is_complete(&self, epoch: u64) -> bool {
        self.completed_count() > epoch
    }
}

/// A buffer that returns itself to its pool when the last handle drops.
///
/// Derefs to `Buffer`, so it substitutes for the `Arc<Buffer>` this replaces
/// everywhere the buffer is read. Cloning the `Arc<PooledBuffer>` shares the
/// handle; the return happens when the final clone drops.
pub struct PooledBuffer {
    /// `Option` only so `drop` can take ownership and hand the buffer back
    /// rather than destroying it. Always `Some` for a live handle.
    buffer: Option<Buffer>,

    /// The bucket this belongs to -- the *allocated* size, not the requested
    /// one, so it goes back where a future request of that size will find it.
    size: usize,

    /// `Weak` so a leaked buffer cannot keep the whole device alive, and so a
    /// buffer outliving its pool drops cleanly instead of resurrecting it.
    pool: Weak<PoolInner>,
}

impl PooledBuffer {
    /// A handle with no pool behind it: dropping it frees the buffer instead
    /// of offering it for reuse.
    ///
    /// For allocations that are deliberately outside the pool, so that callers
    /// which must not participate in reuse still get the same handle type.
    pub fn unpooled(buffer: Buffer, size: usize) -> Self {
        Self {
            buffer: Some(buffer),
            size,
            pool: Weak::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn buffer(&self) -> &Buffer {
        // Only `None` during `drop`, which does not call this.
        self.buffer
            .as_ref()
            .expect("PooledBuffer used after its buffer was taken in drop")
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Buffer {
        self.buffer()
    }
}

impl AsRef<Buffer> for PooledBuffer {
    fn as_ref(&self) -> &Buffer {
        self.buffer()
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for PooledBuffer {
    /// The event the old design lacked.
    ///
    /// This runs when the last `Arc<PooledBuffer>` goes away, which is the
    /// same instant `Arc::strong_count == 1` used to become true of the pooled
    /// `Arc<Buffer>` -- except that now it is a notification rather than a
    /// state to be discovered later by a scan.
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        // Pool gone: the buffer drops here and Metal frees it. Correct, and the
        // reason the back-reference is `Weak`.
        if let Some(pool) = self.pool.upgrade() {
            pool.release(buffer, self.size);
        }
    }
}

/// Statistics, maintained unconditionally because they are counter increments
/// on a path that already takes a lock. Read via `MetalDevice::pool_stats`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolCounters {
    /// Calls into `acquire`.
    pub lookups: u64,
    /// Calls that were served from a free list.
    pub hits: u64,
    /// Free-list buckets probed, summed over all lookups.
    ///
    /// The scan-length figure comparable to the old pool's "buckets walked".
    /// Bounded by 2 per lookup here: the exact-size probe, plus at most one
    /// larger bucket from the ordered range query.
    pub buckets_probed: u64,
    /// Buffers returned by `Drop`.
    pub releases: u64,
    /// Fresh `MTLBuffer` allocations.
    pub allocations: u64,
    /// Bytes handed to `newBufferWithLength`.
    pub allocated_bytes: u64,
    /// Buffers destroyed by `trim`.
    pub trimmed: u64,
    /// Calls into `trim`.
    pub trims: u64,
    /// Releases that had to wait for the GPU before becoming reusable.
    ///
    /// In steady-state decode this is essentially every release: the CPU runs
    /// far enough ahead that a dropped buffer almost always still has work
    /// outstanding. A number near zero here would mean the deferral is not
    /// doing anything, and the corruption it prevents would still be reachable.
    pub deferred: u64,
    /// Buffers moved from the pending list into a free list on GPU completion.
    pub drained: u64,
    /// Buffers currently parked waiting for the GPU.
    pub pending: u64,
}

struct PoolState {
    /// Buffers available for immediate reuse, keyed by allocated size.
    ///
    /// Ordered so lookup can ask for "smallest size >= n" directly. A bucket
    /// is removed from the map when it empties, so the map's length is the
    /// number of sizes that actually have something free -- never a graveyard
    /// of emptied keys.
    free: BTreeMap<usize, Vec<Buffer>>,

    /// Buffers currently held by a caller, and their total allocated bytes.
    ///
    /// Two counters rather than a registry of the buffers themselves. A
    /// registry would need an entry inserted on every acquire and removed on
    /// every release -- hot-path work whose only consumer is a diagnostic --
    /// and its dead entries would accumulate between trims, which is the
    /// accumulate-forever shape this change exists to remove. Incrementing a
    /// pair of counters under a lock that is already held costs nothing.
    live_buffers: usize,
    live_bytes: usize,

    /// Buffers the CPU has released but the GPU may still be using, with the
    /// epoch each must outlive.
    ///
    /// Ordered by epoch, because that is the order they become reusable in:
    /// draining is a prefix walk that stops at the first epoch still
    /// outstanding, never a scan of the whole list. Epochs are assigned from a
    /// monotone counter at push time, so the list is sorted by construction and
    /// nothing has to sort it.
    pending: std::collections::VecDeque<PendingBuffer>,

    counters: PoolCounters,
}

/// A buffer waiting for the GPU to finish with it.
struct PendingBuffer {
    buffer: Buffer,
    /// Bucket to return to -- the allocated size, as for a live handle.
    size: usize,
    /// The epoch this must outlive before it can be handed out again.
    epoch: u64,
}

pub struct PoolInner {
    state: Mutex<PoolState>,
    /// How far the GPU has got. Shared with `Commands`, which advances it.
    clock: Arc<GpuClock>,
}

impl PoolInner {
    fn new(clock: Arc<GpuClock>) -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: BTreeMap::new(),
                live_buffers: 0,
                live_bytes: 0,
                pending: std::collections::VecDeque::new(),
                counters: PoolCounters::default(),
            }),
            clock,
        }
    }

    /// Hands a buffer back. Called only from `PooledBuffer::drop`.
    ///
    /// **This is where issue #19 is fixed.** The CPU dropping its last handle
    /// says the buffer has no future CPU user; it says nothing about whether
    /// the GPU has finished with it, and in decode it usually has not. So the
    /// buffer does not go into the free list here unless the GPU is known to be
    /// done -- otherwise it is parked until the epoch it was dropped in
    /// completes.
    ///
    /// Lookup is untouched. This is a change to *when* a buffer is offered, not
    /// to how one is found.
    fn release(&self, buffer: Buffer, size: usize) {
        let epoch = self.clock.current_epoch();

        let Ok(mut state) = self.state.lock() else {
            // Poisoned: another thread panicked holding the lock. Dropping the
            // buffer here is safe -- Metal frees it -- and is better than
            // propagating a panic out of a destructor.
            return;
        };
        state.counters.releases += 1;
        state.live_buffers = state.live_buffers.saturating_sub(1);
        state.live_bytes = state.live_bytes.saturating_sub(size);

        // Always parks. There is deliberately no "the GPU is idle, hand it back
        // now" fast path: the epoch a release is stamped with is the one still
        // open, and an open epoch has by definition not completed, so such a
        // branch would be unreachable. Anything that would have taken it is
        // instead swept up by the next drain, which is O(1) amortized -- a
        // `VecDeque` push here against a `BTreeMap` entry push there, so the
        // branch would not have bought anything measurable either.
        state.counters.deferred += 1;
        state.pending.push_back(PendingBuffer {
            buffer,
            size,
            epoch,
        });
        state.counters.pending = state.pending.len() as u64;
    }

    /// Moves everything whose epoch has completed into the free lists.
    ///
    /// Called from a command buffer's completion handler -- **once per command
    /// buffer, not once per buffer**. A handler per pooled buffer would put
    /// thousands of block allocations on the decode path; this puts one, and it
    /// releases every buffer that command buffer was holding up.
    ///
    /// `pending` is ordered by epoch, so this stops at the first entry still
    /// outstanding rather than walking the rest.
    fn drain_completed(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let mut drained = 0u64;
        while let Some(front) = state.pending.front() {
            if !self.clock.is_complete(front.epoch) {
                break;
            }
            let entry = state
                .pending
                .pop_front()
                .expect("front() just returned Some");
            state.free.entry(entry.size).or_default().push(entry.buffer);
            drained += 1;
        }
        if drained > 0 {
            state.counters.drained += drained;
            state.counters.pending = state.pending.len() as u64;
        }
    }
}

/// The device-side buffer pool.
///
/// Cheap to clone; every clone refers to the same pool.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// A pool with a private clock that nothing advances.
    ///
    /// Every release then parks, because no epoch ever completes. That is the
    /// safe direction to fail, but it is not useful for a real device: use
    /// [`Self::with_clock`] with the clock `Commands` advances.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(GpuClock::new()))
    }

    /// A pool that decides reuse against `clock`.
    pub fn with_clock(clock: Arc<GpuClock>) -> Self {
        Self {
            inner: Arc::new(PoolInner::new(clock)),
        }
    }

    /// Returns every buffer whose epoch has completed to its free list.
    ///
    /// Call from a command buffer completion handler, once per command buffer.
    pub fn drain_completed(&self) {
        self.inner.drain_completed();
    }

    /// Takes a reusable buffer of at least `size` bytes, if one is free.
    ///
    /// Probes the exact-size bucket first, since an exact match cannot be
    /// improved on, then the smallest larger bucket. At most two ordered map
    /// probes, and buckets with nothing free are never looked at.
    ///
    /// `None` means the caller should allocate; it does **not** mean the pool
    /// holds nothing of that size, only that nothing of that size is free.
    pub fn acquire(&self, size: usize) -> Option<Arc<PooledBuffer>> {
        let mut state = self.inner.state.lock().ok()?;
        state.counters.lookups += 1;

        // Exact hit. Checked separately from the range query below because it
        // is the common case in decode -- every token asks for the same shapes
        // -- and because an exact match is optimal, so there is nothing to gain
        // by looking further.
        state.counters.buckets_probed += 1;
        let found = if let Some(bucket) = state.free.get_mut(&size) {
            bucket.pop().map(|b| (b, size))
        } else {
            None
        };

        // Otherwise the smallest bucket that can satisfy the request. One
        // ordered probe: `range` starts at the first key > size and stops.
        let found = match found {
            Some(hit) => Some(hit),
            None => {
                state.counters.buckets_probed += 1;
                let next = state
                    .free
                    .range_mut((std::ops::Bound::Excluded(size), std::ops::Bound::Unbounded))
                    .next()
                    .and_then(|(k, v)| v.pop().map(|b| (b, *k)));
                next
            }
        };

        let (buffer, bucket_size) = found?;

        // Drop the key when its bucket empties. This is what stops the map
        // from becoming a graveyard: the old pool's `retain` cleared a bucket's
        // Vec but left the key, so emptied keys accumulated at ~3.2 per decode
        // token and were walked forever.
        if state
            .free
            .get(&bucket_size)
            .is_some_and(|bucket| bucket.is_empty())
        {
            state.free.remove(&bucket_size);
        }

        state.counters.hits += 1;
        state.live_buffers += 1;
        state.live_bytes += bucket_size;
        Some(Arc::new(PooledBuffer {
            buffer: Some(buffer),
            size: bucket_size,
            pool: Arc::downgrade(&self.inner),
        }))
    }

    /// Wraps a freshly created `Buffer` in a pool handle.
    ///
    /// `size` must be the buffer's allocated length, since that is the bucket
    /// it returns to.
    pub fn adopt(&self, buffer: Buffer, size: usize) -> Arc<PooledBuffer> {
        if let Ok(mut state) = self.inner.state.lock() {
            state.counters.allocations += 1;
            state.counters.allocated_bytes += size as u64;
            state.live_buffers += 1;
            state.live_bytes += size;
        }
        Arc::new(PooledBuffer {
            buffer: Some(buffer),
            size,
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// Drops every currently-free buffer, returning them so the caller can
    /// unregister them from the residency set.
    ///
    /// This is the sweep, and it is now a **rare trim under memory pressure**
    /// rather than a step on the allocation path. It cannot race a live buffer:
    /// the free list holds only buffers whose last handle has already dropped.
    ///
    /// **Pending buffers are deliberately left alone.** They have been released
    /// by the CPU but the GPU may still be reading or writing them, so freeing
    /// one here would be the same use-after-free the deferral exists to
    /// prevent -- with destruction in place of aliasing. Anything genuinely
    /// finished has already been drained into a free list and is taken by the
    /// loop below; the rest is reclaimed by a later trim.
    pub fn trim(&self) -> Vec<Buffer> {
        // Sweep in anything the GPU has since finished, so a trim after a
        // synchronize reclaims what that synchronize made safe.
        self.inner.drain_completed();

        let mut freed = Vec::new();
        if let Ok(mut state) = self.inner.state.lock() {
            for (_, mut bucket) in std::mem::take(&mut state.free) {
                freed.append(&mut bucket);
            }
            state.counters.trims += 1;
            state.counters.trimmed += freed.len() as u64;
        }
        freed
    }

    pub fn counters(&self) -> PoolCounters {
        self.inner
            .state
            .lock()
            .map(|s| s.counters)
            .unwrap_or_default()
    }

    pub fn reset_counters(&self) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.counters = PoolCounters::default();
        }
    }

    /// `(live buffers, free buffers, free bytes, non-empty free buckets)`.
    ///
    /// "live" counts handles still held by a caller; "free" counts buffers
    /// sitting in the free list ready to be handed out.
    pub fn occupancy(&self) -> PoolOccupancySnapshot {
        let Ok(state) = self.inner.state.lock() else {
            return PoolOccupancySnapshot::default();
        };
        let mut free_buffers = 0;
        let mut free_bytes = 0;
        for bucket in state.free.values() {
            free_buffers += bucket.len();
            for b in bucket {
                free_bytes += b.length();
            }
        }
        PoolOccupancySnapshot {
            live_buffers: state.live_buffers,
            live_bytes: state.live_bytes,
            free_buffers,
            free_bytes,
            free_buckets: state.free.len(),
            pending_buffers: state.pending.len(),
            pending_bytes: state.pending.iter().map(|p| p.buffer.length()).sum(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PoolOccupancySnapshot {
    pub live_buffers: usize,
    pub live_bytes: usize,
    pub free_buffers: usize,
    pub free_bytes: usize,
    /// Free-list buckets that hold at least one buffer.
    ///
    /// By construction there are no others: a bucket is removed when it
    /// empties. This is the number that grew without bound before.
    pub free_buckets: usize,
    /// Released by the CPU, still waiting on the GPU. Not yet reusable.
    ///
    /// This is the memory cost of deciding reuse on the GPU clock: a buffer
    /// counted here would have been immediately reusable under the old
    /// predicate, and immediately *corruptible* with it.
    pub pending_buffers: usize,
    pub pending_bytes: usize,
}

impl PoolOccupancySnapshot {
    pub fn total_buffers(&self) -> usize {
        self.live_buffers + self.free_buffers + self.pending_buffers
    }

    pub fn total_bytes(&self) -> usize {
        self.live_bytes + self.free_bytes + self.pending_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Device;

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
    }

    /// A pool plus the clock driving it, so a test can say when the GPU
    /// finishes rather than depending on a real command queue.
    fn pool_with_clock() -> (BufferPool, Arc<GpuClock>) {
        let clock = Arc::new(GpuClock::new());
        (BufferPool::with_clock(Arc::clone(&clock)), clock)
    }

    /// Stands in for `Commands`: closes the open epoch, reports it finished,
    /// and sweeps -- what a command buffer commit and its completion handler do
    /// between them.
    fn gpu_completes(pool: &BufferPool, clock: &GpuClock) {
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();
    }

    /// Allocates through the pool the way `MetalDevice` does: try to reuse,
    /// otherwise create and adopt.
    fn alloc(pool: &BufferPool, dev: &Device, size: usize) -> Arc<PooledBuffer> {
        if let Some(b) = pool.acquire(size) {
            return b;
        }
        let raw = dev
            .new_buffer(size, crate::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        pool.adopt(raw, size)
    }

    /// Drops `b` and lets the GPU finish, so it is available for reuse. The
    /// two-step is the point of this change: a drop alone no longer suffices.
    fn release_and_complete(pool: &BufferPool, clock: &GpuClock, b: Arc<PooledBuffer>) {
        drop(b);
        gpu_completes(pool, clock);
    }

    #[test]
    fn empty_pool_misses() {
        let pool = BufferPool::new();
        assert!(pool.acquire(128).is_none());
        let c = pool.counters();
        assert_eq!(c.lookups, 1);
        assert_eq!(c.hits, 0);
        // Two probes: exact bucket, then the range query. Both find nothing.
        assert_eq!(c.buckets_probed, 2);
    }

    #[test]
    fn occupancy_of_empty_pool_is_zero() {
        let pool = BufferPool::new();
        let occ = pool.occupancy();
        assert_eq!(occ.free_buckets, 0);
        assert_eq!(occ.total_buffers(), 0);
    }

    /// Releasing is still an *event* -- nothing is swept, polled or asked -- but
    /// the event now says "the CPU is done", which is not the same as "reusable".
    /// The buffer is parked, and it is the GPU finishing that frees it.
    #[test]
    fn drop_parks_the_buffer_until_the_gpu_is_done() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let b = alloc(&pool, &dev, 1024);
        assert_eq!(pool.occupancy().free_buffers, 0, "live buffer is not free");
        assert_eq!(pool.counters().releases, 0);

        drop(b);

        assert_eq!(pool.counters().releases, 1, "Drop did not push");
        assert_eq!(pool.counters().deferred, 1, "release was not deferred");
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "buffer offered for reuse while the GPU may still be using it"
        );
        assert_eq!(pool.occupancy().pending_buffers, 1);
        assert_eq!(pool.occupancy().live_buffers, 0);

        gpu_completes(&pool, &clock);

        assert_eq!(
            pool.occupancy().free_buffers,
            1,
            "GPU completion did not free it"
        );
        assert_eq!(pool.occupancy().pending_buffers, 0);
        assert_eq!(pool.counters().drained, 1);
    }

    /// The bug, stated as a test. A dropped buffer whose GPU work is still
    /// outstanding must not be handed to a second caller -- that is the aliasing
    /// that corrupted grouped convolutions (issue #19).
    #[test]
    fn in_flight_buffer_is_not_handed_out_again() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let first = alloc(&pool, &dev, 4096);
        // The CPU is done with it; the GPU, which has not advanced, is not.
        drop(first);

        assert!(
            pool.acquire(4096).is_none(),
            "handed back a buffer with GPU work outstanding -- this is issue #19"
        );

        gpu_completes(&pool, &clock);
        assert!(
            pool.acquire(4096).is_some(),
            "buffer stayed unavailable after its work completed"
        );
    }

    /// Every release is deferred, including one made while the GPU is idle.
    ///
    /// This looks like a missed optimization and is not one. A release is
    /// stamped with the epoch that is still *open*, and an open epoch cannot
    /// have completed, so a "hand it straight back" branch would be
    /// unreachable. Pinned as a test because the branch is the obvious thing to
    /// add back, and adding it would either do nothing or -- if the condition
    /// were loosened until it did something -- reintroduce the bug.
    #[test]
    fn release_is_deferred_even_when_the_gpu_is_idle() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Everything submitted so far is finished.
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        assert!(clock.is_complete(epoch));

        let b = alloc(&pool, &dev, 1024);
        drop(b);

        assert_eq!(pool.counters().deferred, 1);
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "released against an epoch that is still open"
        );

        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// A released buffer is handed straight back rather than reallocated, and
    /// it is the same Metal object.
    #[test]
    fn released_buffer_is_reused() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Identity of the underlying Metal object, not of the wrapper: the
        // handle is necessarily a different allocation each time, and what
        // matters is that the same MTLBuffer came back.
        fn metal_id(b: &PooledBuffer) -> usize {
            let raw: &Buffer = b;
            raw.as_ref() as *const _ as *const () as usize
        }

        let first = alloc(&pool, &dev, 2048);
        let addr = metal_id(&first);
        release_and_complete(&pool, &clock, first);

        let second = alloc(&pool, &dev, 2048);
        assert_eq!(metal_id(&second), addr, "did not reuse the released buffer");
        assert_eq!(pool.counters().allocations, 1, "allocated a second time");
        assert_eq!(pool.counters().hits, 1);
    }

    /// Epochs retire in order, so a buffer parked in an earlier epoch is freed
    /// by a later completion. The drain walks a prefix and stops; it must not
    /// leave an early buffer behind because a later one is still outstanding.
    #[test]
    fn earlier_epochs_are_freed_by_later_completions() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Parked in epoch 0.
        drop(alloc(&pool, &dev, 1024));
        let e0 = clock.commit_epoch();

        // Parked in epoch 1.
        drop(alloc(&pool, &dev, 2048));
        let e1 = clock.commit_epoch();

        assert_eq!(pool.occupancy().pending_buffers, 2);

        // Epoch 0 finishes: only the first buffer is safe.
        clock.mark_completed(e0);
        pool.drain_completed();
        assert_eq!(pool.occupancy().free_buffers, 1);
        assert_eq!(pool.occupancy().pending_buffers, 1);

        clock.mark_completed(e1);
        pool.drain_completed();
        assert_eq!(pool.occupancy().free_buffers, 2);
        assert_eq!(pool.occupancy().pending_buffers, 0);
    }

    /// Reuse must not require the caller to say anything. A buffer moved into
    /// a struct, cloned, and passed around still returns itself when the last
    /// reference dies -- which is what keeps candle's allocate-and-forget
    /// ergonomics intact.
    #[test]
    fn reclamation_needs_no_caller_cooperation() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        struct Holder {
            _buffer: Arc<PooledBuffer>,
        }

        let b = alloc(&pool, &dev, 512);
        let clone = b.clone();
        let holder = Holder { _buffer: b };

        drop(clone);
        assert_eq!(
            pool.counters().releases,
            0,
            "returned while a reference was still held"
        );

        drop(holder);
        assert_eq!(
            pool.counters().releases,
            1,
            "last reference did not return it"
        );

        // Still nothing the caller has to do: the GPU finishing is what makes
        // it reusable, and that is the device's business, not the caller's.
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// A request is served by a larger free buffer when no exact match exists,
    /// which is what the old scan's "best fit" achieved -- but by one ordered
    /// probe rather than a walk over every bucket.
    #[test]
    fn larger_buffer_satisfies_smaller_request() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        release_and_complete(&pool, &clock, alloc(&pool, &dev, 4096));
        let got = pool.acquire(1024).expect("larger buffer should satisfy");
        assert_eq!(got.size(), 4096);
        assert_eq!(pool.counters().allocations, 1);
    }

    /// The smallest sufficient buffer wins, so a large one is not consumed to
    /// serve a small request while a closer fit sits free.
    ///
    /// Built by allocating all three concurrently and releasing them together:
    /// going through `alloc` one at a time would hand the 8192 back to satisfy
    /// the 2048 request, which is correct reuse but not the state under test.
    #[test]
    fn smallest_sufficient_buffer_wins() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let held: Vec<_> = [8192, 2048, 4096]
            .into_iter()
            .map(|s| alloc(&pool, &dev, s))
            .collect();
        drop(held);
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buckets, 3);

        let got = pool.acquire(1024).expect("something should satisfy");
        assert_eq!(got.size(), 2048, "did not pick the closest fit");
    }

    /// An exact match is taken without consulting any larger bucket. This is
    /// the decode-steady-state case, and it is why the scan is bounded: one
    /// probe, regardless of how many other sizes the pool holds.
    #[test]
    fn exact_match_costs_one_probe() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        for size in [512, 1024, 2048, 4096, 8192] {
            drop(alloc(&pool, &dev, size));
        }
        gpu_completes(&pool, &clock);
        pool.reset_counters();

        let got = pool.acquire(2048).expect("exact match exists");
        assert_eq!(got.size(), 2048);
        assert_eq!(
            pool.counters().buckets_probed,
            1,
            "an exact hit must not consult a second bucket"
        );
    }

    /// The pathology this change removes. The old pool's `retain` emptied a
    /// bucket's `Vec` but left the key in the map, so emptied keys accumulated
    /// -- 1324 of 1345 buckets were empty after a 400-token generation, and
    /// every allocation walked all of them. Here a bucket that empties is
    /// removed, so the map only ever holds sizes that have something free.
    #[test]
    fn emptied_buckets_do_not_accumulate() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Distinct sizes, as a growing KV cache produces.
        for i in 0..64 {
            let size = 1024 + i * 128;
            let b = alloc(&pool, &dev, size);
            release_and_complete(&pool, &clock, b);
            // Take it straight back out, emptying the bucket again.
            let b = pool.acquire(size).expect("just released");
            std::mem::forget(b);
        }

        assert_eq!(
            pool.occupancy().free_buckets,
            0,
            "emptied buckets were left behind"
        );
    }

    /// Scan length is bounded by a constant, not by pool size. This is the
    /// invariant `CONTRIBUTING.md` §3.3 and `DESIGN.md` §15.2 #10 demand of
    /// anything on the per-token path.
    #[test]
    fn lookup_cost_is_bounded_regardless_of_pool_size() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let mut held = Vec::new();
        for i in 0..256 {
            held.push(alloc(&pool, &dev, 1024 + i * 128));
        }
        drop(held);
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buckets, 256);

        pool.reset_counters();
        for i in 0..256 {
            let b = pool.acquire(1024 + i * 128).expect("exact match exists");
            std::mem::forget(b);
        }
        let c = pool.counters();
        assert_eq!(c.lookups, 256);
        assert_eq!(
            c.buckets_probed, 256,
            "expected exactly one probe per exact-match lookup, got {}",
            c.buckets_probed
        );
    }

    /// Trim destroys what is free and leaves what is live, so it can run under
    /// memory pressure without disturbing work in progress.
    #[test]
    fn trim_frees_only_what_is_free() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let live = alloc(&pool, &dev, 1024);
        release_and_complete(&pool, &clock, alloc(&pool, &dev, 2048));
        assert_eq!(pool.occupancy().free_buffers, 1);

        let freed = pool.trim();
        assert_eq!(freed.len(), 1, "trim should return the one free buffer");
        assert_eq!(pool.occupancy().free_buffers, 0);
        assert_eq!(pool.occupancy().live_buffers, 1);

        // The live one still returns itself afterwards.
        release_and_complete(&pool, &clock, live);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// Trim must not destroy a buffer the GPU may still be reading. Freeing one
    /// here would be the same defect the deferral exists to prevent, with
    /// use-after-free in place of aliasing.
    #[test]
    fn trim_leaves_buffers_the_gpu_may_still_be_using() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        drop(alloc(&pool, &dev, 2048));
        assert_eq!(pool.occupancy().pending_buffers, 1);

        let freed = pool.trim();
        assert!(
            freed.is_empty(),
            "trim destroyed a buffer with GPU work outstanding"
        );
        assert_eq!(pool.occupancy().pending_buffers, 1);

        // Once the work completes, a trim reclaims it as normal.
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        assert_eq!(pool.trim().len(), 1);
    }

    /// An unpooled handle frees its buffer outright. `new_private_buffer`
    /// depends on this: it exists for persistent allocations that must not
    /// re-enter the reuse path.
    #[test]
    fn unpooled_handle_does_not_return_to_any_pool() {
        let dev = device();
        let pool = BufferPool::new();

        let raw = dev
            .new_buffer(1024, crate::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = Arc::new(PooledBuffer::unpooled(raw, 1024));
        drop(b);

        assert_eq!(pool.occupancy().free_buffers, 0);
        assert_eq!(pool.counters().releases, 0);
    }

    /// A buffer outliving its pool must drop cleanly rather than resurrect it.
    /// This is why the back-reference is `Weak`.
    #[test]
    fn buffer_outliving_its_pool_drops_cleanly() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();
        let b = alloc(&pool, &dev, 1024);

        drop(pool);
        // Must not panic, and must not try to push into a dead pool.
        drop(b);
    }

    /// Zero must mean "nothing has completed", not "epoch 0 has". They are
    /// different states, and conflating them would free every buffer parked in
    /// the first epoch before the GPU had run anything at all -- the exact bug
    /// this change exists to fix, reintroduced by an off-by-one.
    #[test]
    fn a_fresh_clock_has_completed_nothing() {
        let clock = GpuClock::new();
        assert_eq!(clock.current_epoch(), 0);
        assert_eq!(clock.completed_count(), 0);
        assert!(
            !clock.is_complete(0),
            "epoch 0 reported finished before anything ran"
        );

        clock.mark_completed(0);
        assert!(clock.is_complete(0));
    }

    /// Completion notifications for different command buffers may arrive on
    /// different threads, so the clock must not go backwards when an older one
    /// is delivered late.
    #[test]
    fn completion_never_goes_backwards() {
        let clock = GpuClock::new();
        clock.mark_completed(5);
        clock.mark_completed(2);
        assert_eq!(
            clock.completed_count(),
            6,
            "a late report moved the clock back"
        );
        assert!(clock.is_complete(5));
    }
}
