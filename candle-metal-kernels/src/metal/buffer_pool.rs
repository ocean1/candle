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
//! `PooledBuffer::drop` **pushes** the buffer onto its bucket's free list at
//! the exact moment the last user releases it. Lookup then pops from a free
//! list instead of searching for a free buffer.
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
//!   free: BTreeMap<usize, Vec<Buffer>>     size -> buffers ready to hand out
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
//! # What this deliberately does not do
//!
//! It does not make reuse GPU-liveness-aware. `PooledBuffer::drop` fires when
//! the **CPU** releases its last reference, which is exactly when the old
//! `strong_count == 1` predicate fired, so the reuse decision is made on the
//! same clock as before and issue #19 is neither fixed nor worsened. That is
//! intentional: #19 is a correctness bug with a different fix, and bundling
//! them would make both unreviewable.
//!
//! What this *does* provide is the seam where that fix lands. `release()` is
//! the single point at which a buffer re-enters the free list, so returning a
//! buffer on GPU completion rather than on CPU drop becomes a change to *when
//! `release` is called* -- have `Drop` hand the buffer to the in-flight command
//! buffer's completion handler instead of straight to the free list -- rather
//! than a change to how lookup works. See `PoolInner::release`.

use super::Buffer;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

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

    counters: PoolCounters,
}

pub struct PoolInner {
    state: Mutex<PoolState>,
}

impl PoolInner {
    fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: BTreeMap::new(),
                live_buffers: 0,
                live_bytes: 0,
                counters: PoolCounters::default(),
            }),
        }
    }

    /// Returns a buffer to its bucket. Called only from `PooledBuffer::drop`.
    ///
    /// **This is the seam for issue #19.** The buffer becomes reusable the
    /// moment this runs, and today that is CPU-drop time -- identical in
    /// timing to the `strong_count == 1` predicate it replaces, so in-flight
    /// reuse is neither introduced nor removed here. To make reuse
    /// GPU-liveness-aware, defer *this call* until the last command buffer
    /// that touched the buffer completes; nothing about lookup changes.
    fn release(&self, buffer: Buffer, size: usize) {
        let Ok(mut state) = self.state.lock() else {
            // Poisoned: another thread panicked holding the lock. Dropping the
            // buffer here is safe -- Metal frees it -- and is better than
            // propagating a panic out of a destructor.
            return;
        };
        state.counters.releases += 1;
        state.live_buffers = state.live_buffers.saturating_sub(1);
        state.live_bytes = state.live_bytes.saturating_sub(size);
        state.free.entry(size).or_default().push(buffer);
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
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PoolInner::new()),
        }
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
    pub fn trim(&self) -> Vec<Buffer> {
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
}

impl PoolOccupancySnapshot {
    pub fn total_buffers(&self) -> usize {
        self.live_buffers + self.free_buffers
    }

    pub fn total_bytes(&self) -> usize {
        self.live_bytes + self.free_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Device;

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
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

    /// The property the whole change rests on: releasing is an *event*, not a
    /// state to be discovered later. Nothing is swept, polled or asked; the
    /// buffer is in the free list the moment the handle dies.
    #[test]
    fn drop_returns_the_buffer_immediately() {
        let dev = device();
        let pool = BufferPool::new();

        let b = alloc(&pool, &dev, 1024);
        assert_eq!(pool.occupancy().free_buffers, 0, "live buffer is not free");
        assert_eq!(pool.counters().releases, 0);

        drop(b);

        assert_eq!(pool.counters().releases, 1, "Drop did not push");
        assert_eq!(pool.occupancy().free_buffers, 1);
        assert_eq!(pool.occupancy().live_buffers, 0);
    }

    /// A released buffer is handed straight back rather than reallocated, and
    /// it is the same Metal object.
    #[test]
    fn released_buffer_is_reused() {
        let dev = device();
        let pool = BufferPool::new();

        // Identity of the underlying Metal object, not of the wrapper: the
        // handle is necessarily a different allocation each time, and what
        // matters is that the same MTLBuffer came back.
        fn metal_id(b: &PooledBuffer) -> usize {
            let raw: &Buffer = b;
            raw.as_ref() as *const _ as *const () as usize
        }

        let first = alloc(&pool, &dev, 2048);
        let addr = metal_id(&first);
        drop(first);

        let second = alloc(&pool, &dev, 2048);
        assert_eq!(metal_id(&second), addr, "did not reuse the released buffer");
        assert_eq!(pool.counters().allocations, 1, "allocated a second time");
        assert_eq!(pool.counters().hits, 1);
    }

    /// Reuse must not require the caller to say anything. A buffer moved into
    /// a struct, cloned, and passed around still returns itself when the last
    /// reference dies -- which is what keeps candle's allocate-and-forget
    /// ergonomics intact.
    #[test]
    fn reclamation_needs_no_caller_cooperation() {
        let dev = device();
        let pool = BufferPool::new();

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
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// A request is served by a larger free buffer when no exact match exists,
    /// which is what the old scan's "best fit" achieved -- but by one ordered
    /// probe rather than a walk over every bucket.
    #[test]
    fn larger_buffer_satisfies_smaller_request() {
        let dev = device();
        let pool = BufferPool::new();

        drop(alloc(&pool, &dev, 4096));
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
        let pool = BufferPool::new();

        let held: Vec<_> = [8192, 2048, 4096]
            .into_iter()
            .map(|s| alloc(&pool, &dev, s))
            .collect();
        drop(held);
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
        let pool = BufferPool::new();

        for size in [512, 1024, 2048, 4096, 8192] {
            drop(alloc(&pool, &dev, size));
        }
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
        let pool = BufferPool::new();

        // Distinct sizes, as a growing KV cache produces.
        for i in 0..64 {
            let size = 1024 + i * 128;
            let b = alloc(&pool, &dev, size);
            drop(b);
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
        let pool = BufferPool::new();

        let mut held = Vec::new();
        for i in 0..256 {
            held.push(alloc(&pool, &dev, 1024 + i * 128));
        }
        drop(held);
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
        let pool = BufferPool::new();

        let live = alloc(&pool, &dev, 1024);
        drop(alloc(&pool, &dev, 2048));
        assert_eq!(pool.occupancy().free_buffers, 1);

        let freed = pool.trim();
        assert_eq!(freed.len(), 1, "trim should return the one free buffer");
        assert_eq!(pool.occupancy().free_buffers, 0);
        assert_eq!(pool.occupancy().live_buffers, 1);

        // The live one still returns itself afterwards.
        drop(live);
        assert_eq!(pool.occupancy().free_buffers, 1);
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
        let pool = BufferPool::new();
        let b = alloc(&pool, &dev, 1024);

        drop(pool);
        // Must not panic, and must not try to push into a dead pool.
        drop(b);
    }
}
