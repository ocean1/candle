//! The activation arena: one allocation, slots at fixed offsets.
//!
//! `DESIGN.md` §9.2. Decode activation shapes are compile-time constants (§4.2),
//! so which buffer serves a given dispatch is a question that can be answered
//! once, offline, instead of by an allocator on every token.
//!
//! # What this is for, and it is not memory
//!
//! Issue #68 measured the packing: 517 activation values fit in **5 slots,
//! 68 KB**, against a 5.2 GB pool. That is 0.0013 % — packing activations saves
//! nothing worth having.
//!
//! The payoff is **dispatch stability** (§9.2c). Buffer identity varies at all
//! 674 decode dispatch positions today (§11.1a.1), because the pool hands out a
//! different allocation for the same logical slot on every token, and ICB replay
//! needs that identity fixed. The arena makes every activation a region of one
//! buffer, so the identity a dispatch binds stops depending on allocation
//! history. **Its acceptance criterion is "674 varying → 0", not bytes saved.**
//!
//! # Layer 3, and why that is not a detail
//!
//! §9.2a: the pool has three layers, and each is optional to the one above.
//!
//! 1. automatic `Drop` reclamation — the default, and it stays
//! 2. explicit control (`trim`, unpooled handles) — opt-in, on top
//! 3. **the arena** — this module
//!
//! A slot is a `PooledBuffer` handle held by the plan for the plan's lifetime,
//! so the free list simply never sees it. That is exactly how resident weights
//! are already handled: a weight's handle is never dropped, so it is never in a
//! free list and no lookup ever looks at it (§6.3a). **No new mechanism, no
//! manual `free()`, and no exemption from the reuse path** — the reuse path is
//! defined by whether anyone lets go, and the arena simply does not.
//!
//! Do not invert this. A design that required callers to release arena buffers
//! by hand would push manual lifetime management onto every caller permanently,
//! which is the property §9.2a says any replacement must preserve.
//!
//! # Liveness keys on the value, never on the buffer
//!
//! §9.2c, and it is the trap #68 hit and self-caught. Candle's pool recycles one
//! allocation across unrelated values *within a single token*: in #68's trace
//! `buf#328` is written and fully consumed 13 times per token, and `buf#47`
//! carries 60 distinct values. Keying liveness on `(buffer, offset)` merges
//! those into one interval and invents lifetimes no activation has — #68's first
//! planner did exactly that and reported **3.55 MB against the true 68 KB**, a
//! 52× overstatement.
//!
//! In candle the buffer identity is right there in the allocation call, so the
//! wrong key is the *easier* one to reach for. This module does not key on it at
//! all: a slot is chosen by **allocation ordinal within the step**, which is a
//! property of where the allocation sits in the sequence, not of what the pool
//! happened to hand out last time. See [`ArenaPlan::acquire`].
//!
//! # Session state must not enter the arena
//!
//! §9.1 and #68's finding 4. 169 dispatch positions write values that grow
//! 128 B per token — the KV cache and the `Tensor::cat` copies it moves through.
//! Those are *session state*: per-sequence, live across steps, sized by
//! `kv_len`. A slot whose occupant grows every token cannot hold a fixed offset,
//! so everything packed after it drifts; with KV left in, **969 bindings fail
//! the 674-stability check**.
//!
//! The plan enforces this by construction rather than by classification: a
//! request whose size does not match the slot recorded for that ordinal is
//! declined and falls through to the pool (see [`ArenaPlan::acquire`]). A
//! growing KV allocation changes size every token, so it declines every token.
//! `arena_size_mismatch_declines_rather_than_drifting` pins it.
//!
//! # The hazard
//!
//! §9.3: under `HazardTrackingModeUntracked` there is **no safety net**, and
//! aliasing correctness rests entirely on this module's offset arithmetic. Every
//! layout must be validated by a parity test against a non-aliasing reference
//! layout — one where every value gets its own slot and aliasing is impossible
//! by construction. [`ArenaLayout::NonAliasing`] is that reference, and it is an
//! *execution* comparison rather than a size comparison: both layouts run the
//! real model and their outputs are compared.

use super::{Buffer, BufferPool, PooledBuffer};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Alignment of every slot, in bytes.
///
/// 128 B, per §9.2 and our fork's PR #8. Two independent reasons land on the
/// same number and both must hold:
///
/// - it covers every Metal dtype (the kernels use `float4`/`half4` at 16 B and
///   quantized block structures are wider, all well under 128)
/// - `hw.cachelinesize` on this machine is **also** 128 B, so a slot boundary is
///   a CPU cache-line boundary
///
/// §9.2 is explicit that the coincidence is load-bearing in one direction:
/// **lowering this to 64 would silently introduce false sharing between adjacent
/// slots** on any CPU-side access, and the Metal-dtype reasoning alone would not
/// reveal that. Do not change it without measuring both.
pub const ARENA_ALIGNMENT: usize = 128;

/// Which layout an arena hands out.
///
/// Both are compiled and either is selectable at run time, the same discipline
/// `ParamStyle` follows for binding styles (§11.3b): keeping both live makes the
/// A/B free, and makes the parity check in §9.3 a comparison between two paths
/// that actually exist rather than between one path and an argument.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArenaLayout {
    /// The packed plan: values whose liveness intervals are disjoint share a
    /// slot. This is the layout §9.2 specifies and the one that saves the 68 KB.
    #[default]
    Packed,

    /// **The reference layout §9.3 requires.** Every value gets its own slot, so
    /// no two values ever share bytes and aliasing is impossible by
    /// construction.
    ///
    /// This is not a debugging aid, it is the oracle. The packed layout is
    /// correct only if it computes what this one computes; any offset-arithmetic
    /// error shows up as a difference between them, and under
    /// `HazardTrackingModeUntracked` there is no other detector (§9.3).
    NonAliasing,
}

/// One slot: a fixed region of the arena buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// Byte offset of the slot within the arena buffer. 128 B aligned.
    pub offset: usize,
    /// Bytes reserved for the slot -- its largest occupant.
    pub size: usize,
}

/// What a decode step's allocations map to.
///
/// Index is the **allocation ordinal within the step**: the Nth activation
/// allocation of every token gets entry N. That is the positional keying #68's
/// finding 1 asks for — "the assignment is a property of where a dispatch sits
/// in the sequence" — and it is what makes the plan independent of which buffer
/// the pool handed out.
#[derive(Clone, Debug, Default)]
pub struct StepPlan {
    slots: Vec<Slot>,
    /// Slot index per allocation ordinal, and the size that ordinal expects.
    ///
    /// The size is carried so a request that does not match can be declined
    /// rather than silently served a slot of the wrong extent -- which is how
    /// session state is kept out (see the module docs).
    by_ordinal: Vec<(usize, usize)>,
}

impl StepPlan {
    /// Build a plan from the per-ordinal `(slot, size)` assignment.
    pub fn new(slots: Vec<Slot>, by_ordinal: Vec<(usize, usize)>) -> Self {
        Self { slots, by_ordinal }
    }

    /// Total bytes the arena must reserve.
    pub fn arena_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.offset + s.size)
            .max()
            .unwrap_or(0)
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn allocations(&self) -> usize {
        self.by_ordinal.len()
    }

    /// Every slot is 128 B aligned and no two slots overlap.
    ///
    /// The overlap half is the one that matters: two slots sharing bytes would
    /// alias two values that the plan believes are disjoint, and §9.3 says
    /// nothing in the driver would catch it. Checked exhaustively rather than
    /// sampled, for the reason #68 gives -- a sampled check on an aliasing
    /// invariant is worth very little.
    pub fn check_disjoint(&self) -> Result<(), String> {
        for (i, s) in self.slots.iter().enumerate() {
            if !s.offset.is_multiple_of(ARENA_ALIGNMENT) {
                return Err(format!(
                    "slot {i} at offset {} is not {ARENA_ALIGNMENT} B aligned",
                    s.offset
                ));
            }
        }
        for (i, a) in self.slots.iter().enumerate() {
            for (j, b) in self.slots.iter().enumerate().skip(i + 1) {
                let overlap = a.offset < b.offset + b.size && b.offset < a.offset + a.size;
                if overlap {
                    return Err(format!(
                        "slots {i} ({}..{}) and {j} ({}..{}) overlap",
                        a.offset,
                        a.offset + a.size,
                        b.offset,
                        b.offset + b.size
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Counters, so "arena buffers never enter the free list" is checkable rather
/// than asserted.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArenaCounters {
    /// Allocation requests offered to the arena.
    pub offers: u64,
    /// Requests served from a slot.
    pub hits: u64,
    /// Requests declined because the size did not match the ordinal's slot.
    ///
    /// In LFM2 decode this is the session-state population -- the KV cache and
    /// its `Tensor::cat` copies, which grow 128 B per token and so never match
    /// (#68 finding 4). A zero here on a real decode would mean the size gate is
    /// not doing anything, and the drift it prevents would be reachable.
    pub declined_size: u64,
    /// Requests past the end of the plan.
    pub declined_exhausted: u64,
    /// Decode steps begun.
    pub steps: u64,
}

struct ArenaState {
    plan: StepPlan,
    layout: ArenaLayout,
    /// One handle per slot, created once and held for the plan's lifetime.
    ///
    /// **This is what makes the arena layer 3.** The handles never drop while
    /// the plan lives, so the free list never sees them and no lookup ever
    /// considers them -- exactly how resident weights already behave (§6.3a).
    /// Nothing here calls `free`.
    slot_handles: Vec<Arc<PooledBuffer>>,
    counters: ArenaCounters,
}

/// The activation arena.
///
/// Cheap to clone; every clone refers to the same arena.
#[derive(Clone)]
pub struct Arena {
    inner: Arc<ArenaInner>,
}

struct ArenaInner {
    /// The one allocation. §9.2's "the arena is one allocation": if a future
    /// class needs more, it gets its own arena rather than turning this into an
    /// array of buffers.
    base: Buffer,
    state: Mutex<ArenaState>,
    /// Allocation ordinal within the current decode step.
    ///
    /// Relaxed because a decode step is single-threaded through the model's
    /// forward pass; this counter orders allocations against each other on one
    /// thread, and nothing reads it from another.
    ordinal: AtomicUsize,
    /// Whether a step is currently open. Outside a step the arena declines
    /// everything, so prefill and setup allocate from the pool as before.
    active: AtomicU64,
}

impl Arena {
    /// Allocate the arena and build its slot handles.
    ///
    /// One `MTLBuffer` of `plan.arena_bytes()`, allocated once here and never
    /// again. The slot handles are created now and held until the arena drops.
    pub fn new(
        pool: &BufferPool,
        base: Buffer,
        plan: StepPlan,
        layout: ArenaLayout,
    ) -> Result<Self, String> {
        plan.check_disjoint()?;
        let needed = plan.arena_bytes();
        if base.length() < needed {
            return Err(format!(
                "arena buffer is {} B, plan needs {needed} B",
                base.length()
            ));
        }

        // One handle per slot, made once. `adopt` gives each a pool handle so it
        // is the same type every consumer already takes -- but because the
        // handles are held here for the arena's lifetime, none of them is ever
        // dropped, so none ever reaches a free list. Layer 3 by construction.
        let slot_handles = plan
            .slots()
            .iter()
            .map(|s| pool.adopt(base.view(s.offset, s.size), s.size))
            .collect();

        Ok(Self {
            inner: Arc::new(ArenaInner {
                base,
                state: Mutex::new(ArenaState {
                    plan,
                    layout,
                    slot_handles,
                    counters: ArenaCounters::default(),
                }),
                ordinal: AtomicUsize::new(0),
                active: AtomicU64::new(0),
            }),
        })
    }

    /// The single underlying allocation, for residency registration.
    ///
    /// §9.2's "residency is a CPU-side fact established once": one buffer in the
    /// residency set rather than 674.
    pub fn base(&self) -> &Buffer {
        &self.inner.base
    }

    pub fn layout(&self) -> ArenaLayout {
        self.inner
            .state
            .lock()
            .map(|s| s.layout)
            .unwrap_or_default()
    }

    /// Open a decode step, resetting the allocation ordinal.
    ///
    /// The ordinal is what ties an allocation to a slot, so it must restart at
    /// the same point every token or the assignment would walk. The caller marks
    /// the boundary because only the caller knows where a step begins -- the same
    /// signal the dispatch-trace harness already takes for its per-step regions.
    pub fn begin_step(&self) {
        self.inner.ordinal.store(0, Ordering::Relaxed);
        self.inner.active.store(1, Ordering::Relaxed);
        if let Ok(mut s) = self.inner.state.lock() {
            s.counters.steps += 1;
        }
    }

    /// Close the current step. Allocations outside a step go to the pool.
    pub fn end_step(&self) {
        self.inner.active.store(0, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.inner.active.load(Ordering::Relaxed) != 0
    }

    /// Offer an allocation of `size` bytes to the arena.
    ///
    /// `None` means the caller should allocate from the pool as before, and it
    /// is a normal outcome rather than a failure: everything outside a decode
    /// step, everything past the end of the plan, and every session-state
    /// allocation lands here.
    ///
    /// # Why the ordinal, and not the buffer
    ///
    /// The slot is chosen by *where this allocation falls in the step*, which is
    /// stable across tokens because the op sequence is (§11.1a.1: kernel name
    /// varies at 0 of 674 positions). Choosing by buffer identity instead would
    /// key on exactly the thing the arena exists to remove, which is the 52×
    /// error §9.2c records.
    ///
    /// # The size gate
    ///
    /// A request is served only if its size matches what the plan recorded for
    /// that ordinal. This is what keeps session state out (§9.1, #68 finding 4):
    /// the KV cache grows 128 B per token, so it matches on no token and takes
    /// the pool path every time. Serving it a fixed slot would either overflow
    /// the slot or pin a growing value at a fixed offset, and everything packed
    /// after it would drift.
    pub fn acquire(&self, size: usize) -> Option<Arc<PooledBuffer>> {
        if !self.is_active() {
            return None;
        }
        let ordinal = self.inner.ordinal.fetch_add(1, Ordering::Relaxed);
        let mut state = self.inner.state.lock().ok()?;
        state.counters.offers += 1;

        let Some(&(slot, want)) = state.plan.by_ordinal.get(ordinal) else {
            state.counters.declined_exhausted += 1;
            return None;
        };
        // Size must match what was planned for this ordinal. A larger request
        // would run past the slot; a smaller one means the sequence has drifted
        // from the plan and the ordinal no longer names what it did.
        if size != want {
            state.counters.declined_size += 1;
            return None;
        }
        state.counters.hits += 1;
        Some(Arc::clone(&state.slot_handles[slot]))
    }

    pub fn counters(&self) -> ArenaCounters {
        self.inner
            .state
            .lock()
            .map(|s| s.counters)
            .unwrap_or_default()
    }

    pub fn reset_counters(&self) {
        if let Ok(mut s) = self.inner.state.lock() {
            s.counters = ArenaCounters::default();
        }
    }

    /// Slots, for reporting and for the layout checks.
    pub fn slots(&self) -> Vec<Slot> {
        self.inner
            .state
            .lock()
            .map(|s| s.plan.slots().to_vec())
            .unwrap_or_default()
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (slots, bytes) = self
            .inner
            .state
            .lock()
            .map(|s| (s.plan.slots().len(), s.plan.arena_bytes()))
            .unwrap_or((0, 0));
        f.debug_struct("Arena")
            .field("slots", &slots)
            .field("bytes", &bytes)
            .finish()
    }
}

/// Build a plan from a recorded sequence of activation allocation sizes.
///
/// `sizes[i]` is the byte size of the i-th activation allocation in a decode
/// step, and `last_use[i]` is the ordinal of the last allocation that may still
/// be reading it. Together those are the liveness interval #68 computes by
/// refcount (§9.2b), expressed positionally.
///
/// # Ordering is size-major, and that is measured rather than inherited
///
/// §9.2 and issue #43. Slots are assigned in **size-descending** order, not in
/// start order. Linear-scan register allocation sweeps in start order and gets
/// away with it because registers are uniform in size; arena slots are not, and
/// that is the whole difference.
///
/// #68 measured both on our shapes: size-major packs to **68.00 KB** and
/// start-major to **85.00 KB**, a 1.25× difference, and the mechanism is visible
/// in the slot sizes -- start-major opens a fourth MLP-width slot because a
/// 4096 B value takes a fresh slot first and a 21504 B value later first-fits
/// into it and widens it 5.25×. Luminal hit the same mechanism at 27× on ~100
/// slots; LFM2 decode has 5 slots and two size classes, which bounds the damage
/// at about one extra wide slot.
pub fn plan_from_sizes(sizes: &[usize], last_use: &[usize], layout: ArenaLayout) -> StepPlan {
    assert_eq!(
        sizes.len(),
        last_use.len(),
        "every allocation needs a last-use ordinal"
    );

    if layout == ArenaLayout::NonAliasing {
        // The §9.3 reference: one slot per value, so no two values can share
        // bytes whatever the liveness intervals say. Deliberately not packed --
        // its whole purpose is to be the layout that cannot alias.
        let mut slots = Vec::with_capacity(sizes.len());
        let mut by_ordinal = Vec::with_capacity(sizes.len());
        let mut cursor = 0usize;
        for (i, &size) in sizes.iter().enumerate() {
            slots.push(Slot {
                offset: cursor,
                size,
            });
            by_ordinal.push((i, size));
            cursor = align_up(cursor + size, ARENA_ALIGNMENT);
        }
        return StepPlan::new(slots, by_ordinal);
    }

    // Size-major first-fit. Each slot holds values whose intervals are pairwise
    // disjoint; a slot widens to its largest occupant.
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(sizes[i]),
            i,
            std::cmp::Reverse(last_use[i].saturating_sub(i)),
        )
    });

    // Per slot: the intervals already placed in it, and its current size.
    let mut slot_intervals: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut slot_sizes: Vec<usize> = Vec::new();
    let mut assigned = vec![usize::MAX; sizes.len()];

    for &i in &order {
        let (start, end) = (i, last_use[i]);
        let mut placed = None;
        for (si, intervals) in slot_intervals.iter_mut().enumerate() {
            // Overlap is half-open in the direction this problem needs: a value
            // dies after its last read, so a value first written at exactly
            // another's last use may not share its bytes -- the producing
            // dispatch of the second reads the first.
            let clash = intervals.iter().any(|&(s, e)| !(end < s || e < start));
            if clash {
                continue;
            }
            intervals.push((start, end));
            slot_sizes[si] = slot_sizes[si].max(sizes[i]);
            placed = Some(si);
            break;
        }
        let si = match placed {
            Some(si) => si,
            None => {
                slot_intervals.push(vec![(start, end)]);
                slot_sizes.push(sizes[i]);
                slot_intervals.len() - 1
            }
        };
        assigned[i] = si;
    }

    // Slot offsets are assigned only once every size is final, since a slot
    // widens after it is opened.
    let mut slots = Vec::with_capacity(slot_sizes.len());
    let mut cursor = 0usize;
    for &size in &slot_sizes {
        slots.push(Slot {
            offset: cursor,
            size,
        });
        cursor = align_up(cursor + size, ARENA_ALIGNMENT);
    }

    let by_ordinal = assigned
        .iter()
        .enumerate()
        .map(|(i, &slot)| (slot, sizes[i]))
        .collect();

    StepPlan::new(slots, by_ordinal)
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Observes one decode step and derives the plan from it.
///
/// # Why a recorder rather than a table
///
/// The plan needs each activation's size and the interval it is live over.
/// Candle is eager (§11.1a), so nothing declares that up front -- but the
/// dispatch sequence *is* stable (§11.1a.1: kernel name varies at 0 of 674
/// positions), so one observed step describes every later one. That is the same
/// record-then-replay shape §11.1a calls "the honest first version", applied to
/// allocation rather than to dispatch.
///
/// # Liveness is keyed on the value, never on the buffer
///
/// §9.2c, and this is where the 52× error lives if it is going to. A value here
/// is **one allocation event** -- one `record_alloc`, ending at the matching
/// `record_free`. When the pool hands the same `MTLBuffer` back for a later,
/// unrelated value, that is a *new* value with its own interval, because it is a
/// new allocation event. Nothing in this recorder consults buffer identity at
/// all, which is why it cannot make the mistake: in #68's trace `buf#328` is
/// written and consumed 13 times per token, and keying on it would merge those
/// into one 3.55 MB interval against the true 68 KB.
///
/// The generation counter is implicit in the ordinal: allocation N and
/// allocation M are different values whenever `N != M`, whatever buffer either
/// one happened to get.
#[derive(Debug, Default)]
pub struct ArenaRecorder {
    /// Size of each allocation, in ordinal order.
    sizes: Vec<usize>,
    /// Ordinal at which each allocation was released. `None` while still live.
    freed_at: Vec<Option<usize>>,
    /// Allocation ordinals of values still live, keyed by the identity the
    /// caller uses to report a free.
    live: Vec<(u64, usize)>,
    /// Allocations seen so far this step.
    ordinal: usize,
}

impl ArenaRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation of `size` bytes, tagged with a caller-chosen
    /// `token` that [`Self::record_free`] will use to close it.
    ///
    /// The token identifies *this allocation event*, not the buffer: reusing a
    /// buffer produces a new event with a new token, which is exactly the
    /// value-not-buffer distinction §9.2c requires.
    pub fn record_alloc(&mut self, token: u64, size: usize) {
        let ordinal = self.ordinal;
        self.ordinal += 1;
        self.sizes.push(size);
        self.freed_at.push(None);
        self.live.push((token, ordinal));
    }

    /// Record that the value tagged `token` has been released.
    ///
    /// Its interval ends at the **last ordinal already issued**, not at the next
    /// one. The distinction is an off-by-one that decides whether the packing
    /// does anything at all: stamping the next ordinal would make every value
    /// collide with its immediate successor, so nothing would ever share a slot
    /// and the arena would degenerate into the non-aliasing reference.
    ///
    /// This matches #68's convention, where `last_use` is the position of the
    /// last dispatch that reads the value, and it pairs with an overlap test
    /// that is half-open in the same direction: a value allocated at exactly
    /// another's last use *may* take its bytes, because the earlier one is done
    /// being read by then.
    pub fn record_free(&mut self, token: u64) {
        let at = self.ordinal.saturating_sub(1);
        if let Some(pos) = self.live.iter().rposition(|&(t, _)| t == token) {
            let (_, ordinal) = self.live.remove(pos);
            // Never before its own allocation: a value freed immediately is
            // live for exactly the instant it was created.
            self.freed_at[ordinal] = Some(at.max(ordinal));
        }
    }

    /// Allocations recorded so far.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Build a plan from what was observed.
    ///
    /// A value never freed is treated as live to the end of the step, which is
    /// the conservative direction: it can then share bytes with nothing, so the
    /// plan is larger but never aliases two live values. #68 records 39 such
    /// values -- dead stores and the final logits, which leave the arena.
    pub fn plan(&self, layout: ArenaLayout) -> StepPlan {
        let end = self.ordinal.saturating_sub(1);
        let last_use: Vec<usize> = self
            .freed_at
            .iter()
            .enumerate()
            .map(|(i, f)| f.unwrap_or(end).max(i))
            .collect();
        plan_from_sizes(&self.sizes, &last_use, layout)
    }

    /// Forget everything, to record a fresh step.
    pub fn reset(&mut self) {
        self.sizes.clear();
        self.freed_at.clear();
        self.live.clear();
        self.ordinal = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Device;

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
    }

    fn arena_of(sizes: &[usize], last_use: &[usize], layout: ArenaLayout) -> (Arena, BufferPool) {
        let dev = device();
        let plan = plan_from_sizes(sizes, last_use, layout);
        let base = dev
            .new_buffer(plan.arena_bytes().max(1), crate::RESOURCE_OPTIONS)
            .expect("arena allocation");
        let pool = BufferPool::new();
        let arena = Arena::new(&pool, base, plan, layout).expect("plan is valid");
        (arena, pool)
    }

    /// Two values whose lifetimes do not overlap share a slot; two that do
    /// overlap do not. This is the packing property the 68 KB comes from.
    #[test]
    fn disjoint_values_share_a_slot_and_overlapping_ones_do_not() {
        // 0 and 2 are disjoint (0 dies at 1); 1 overlaps both.
        let sizes = [4096, 4096, 4096];
        let last_use = [1, 2, 2];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        let s: Vec<usize> = plan.by_ordinal.iter().map(|&(slot, _)| slot).collect();
        assert_ne!(s[0], s[1], "overlapping values shared a slot");
        assert_ne!(s[1], s[2], "overlapping values shared a slot");
        assert_eq!(s[0], s[2], "disjoint values did not share a slot");
        assert_eq!(plan.slots().len(), 2);
    }

    /// The §9.3 reference layout gives every value its own slot, so aliasing is
    /// impossible by construction. It must be strictly larger than the packed
    /// one on any input where packing does anything -- if they ever tie, the
    /// reference has stopped being a reference.
    #[test]
    fn non_aliasing_reference_never_shares_bytes() {
        let sizes = [4096, 4096, 4096];
        let last_use = [1, 2, 2];
        let packed = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        let reference = plan_from_sizes(&sizes, &last_use, ArenaLayout::NonAliasing);

        assert_eq!(reference.slots().len(), 3, "a value shared a slot");
        let s: Vec<usize> = reference.by_ordinal.iter().map(|&(slot, _)| slot).collect();
        assert_eq!(s, vec![0, 1, 2]);
        assert!(
            reference.arena_bytes() > packed.arena_bytes(),
            "reference {} is not larger than packed {}",
            reference.arena_bytes(),
            packed.arena_bytes()
        );
        reference.check_disjoint().expect("reference slots overlap");
    }

    /// Size-major beats start-major on the shape that produces #68's 1.25×: a
    /// small value opening a slot that a large one later widens.
    ///
    /// This is issue #43's mechanism, reproduced at the smallest size that shows
    /// it. Pinned because the sort key is three lines and reverting it to start
    /// order would look like a simplification.
    #[test]
    fn size_major_does_not_widen_a_slot_a_small_value_opened() {
        // A small value, then a large one whose interval is disjoint from it.
        // Start order places the small one first, so the large one first-fits
        // into its slot and widens it 5.25x; size-major places the large one
        // first and the small one fills in behind it.
        let sizes = [4096, 21504, 21504];
        let last_use = [0, 2, 2];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);

        // The 4096 must not have widened a slot to 21504 that then needs a
        // second 21504 beside it: two MLP-width slots is right, three is the
        // start-major pathology.
        let mlp_slots = plan.slots().iter().filter(|s| s.size == 21504).count();
        assert_eq!(
            mlp_slots,
            2,
            "expected 2 MLP-width slots, got {mlp_slots}: {:?}",
            plan.slots()
        );
    }

    /// Every slot is 128 B aligned, and the alignment is the cache line.
    ///
    /// §9.2 says lowering this to 64 would introduce false sharing between
    /// adjacent slots without the Metal-dtype reasoning revealing it, so the
    /// constant is asserted rather than left to a comment.
    #[test]
    fn slots_are_cache_line_aligned() {
        assert_eq!(ARENA_ALIGNMENT, 128);
        let sizes = [4096, 21504, 1024, 32];
        let last_use = [3, 3, 3, 3];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        for s in plan.slots() {
            assert_eq!(
                s.offset % ARENA_ALIGNMENT,
                0,
                "slot at {} is not {ARENA_ALIGNMENT} B aligned",
                s.offset
            );
        }
        plan.check_disjoint().expect("slots overlap");
    }

    /// The check that guards the aliasing invariant must be able to fail.
    /// `CONTRIBUTING.md` §3.1: a test that cannot fail is not a test, and this
    /// one guards the case §9.3 says has no safety net.
    #[test]
    fn overlap_check_catches_overlapping_slots() {
        let good = StepPlan::new(
            vec![
                Slot {
                    offset: 0,
                    size: 128,
                },
                Slot {
                    offset: 128,
                    size: 128,
                },
            ],
            vec![(0, 128), (1, 128)],
        );
        good.check_disjoint().expect("disjoint slots rejected");

        // Mutation: the second slot starts inside the first.
        let bad = StepPlan::new(
            vec![
                Slot {
                    offset: 0,
                    size: 256,
                },
                Slot {
                    offset: 128,
                    size: 128,
                },
            ],
            vec![(0, 256), (1, 128)],
        );
        let err = bad
            .check_disjoint()
            .expect_err("overlapping slots were accepted");
        assert!(err.contains("overlap"), "unexpected error: {err}");

        // Mutation: a slot that is not cache-line aligned.
        let misaligned = StepPlan::new(
            vec![Slot {
                offset: 64,
                size: 128,
            }],
            vec![(0, 128)],
        );
        let err = misaligned
            .check_disjoint()
            .expect_err("a 64 B aligned slot was accepted");
        assert!(err.contains("aligned"), "unexpected error: {err}");
    }

    /// The same ordinal gets the same slot on every step. This is the property
    /// the whole issue exists for -- 674 varying buffer identities becoming 0 --
    /// asserted at the level where it is decided.
    #[test]
    fn the_same_ordinal_gets_the_same_slot_every_step() {
        let sizes = [4096, 21504, 4096];
        let last_use = [2, 2, 2];
        let (arena, _pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        let mut per_step = Vec::new();
        for _ in 0..8 {
            arena.begin_step();
            let step: Vec<usize> = sizes
                .iter()
                .map(|&s| {
                    let b = arena.acquire(s).expect("plan covers this ordinal");
                    // Identity of the region, which is what a dispatch binds.
                    (b.base_offset(), b.length()).0
                })
                .collect();
            arena.end_step();
            per_step.push(step);
        }

        let first = &per_step[0];
        for (i, step) in per_step.iter().enumerate() {
            assert_eq!(step, first, "step {i} bound different offsets: {step:?}");
        }
    }

    /// A slot handle is never dropped while the arena lives, so it never enters
    /// a free list. That is what makes the arena layer 3 (§9.2a) rather than a
    /// change to how the pool decides a buffer is free.
    #[test]
    fn arena_buffers_never_enter_the_free_list() {
        let sizes = [4096, 21504];
        let last_use = [1, 1];
        let (arena, pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        for _ in 0..4 {
            arena.begin_step();
            for &s in &sizes {
                let b = arena.acquire(s).expect("plan covers this ordinal");
                // The caller drops its handle, exactly as a tensor would.
                drop(b);
            }
            arena.end_step();
        }

        assert_eq!(
            pool.counters().releases,
            0,
            "an arena buffer was released to the pool"
        );
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "an arena buffer reached a free list"
        );
    }

    /// Session state must not enter the arena (§9.1, #68 finding 4). A value
    /// that grows every token does not match its ordinal's planned size, so it
    /// is declined and falls through to the pool.
    ///
    /// The failure this prevents is not a size overflow but a *drift*: a slot
    /// whose occupant grows cannot hold a fixed offset, and with KV left in,
    /// 969 bindings fail the 674-stability check.
    #[test]
    fn arena_size_mismatch_declines_rather_than_drifting() {
        let sizes = [4096, 21504];
        let last_use = [1, 1];
        let (arena, _pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        for token in 0..4usize {
            arena.begin_step();
            assert!(arena.acquire(4096).is_some(), "fixed-size value declined");
            // A KV-shaped allocation: 128 B larger every token.
            assert!(
                arena.acquire(21504 + 128 * (token + 1)).is_none(),
                "a growing value was given a fixed slot"
            );
            arena.end_step();
        }

        let c = arena.counters();
        assert_eq!(c.declined_size, 4, "the size gate never engaged");
        assert_eq!(c.hits, 4);
    }

    /// Outside a decode step the arena declines everything, so prefill and
    /// model setup allocate from the pool exactly as before.
    #[test]
    fn nothing_is_served_outside_a_step() {
        let (arena, _pool) = arena_of(&[4096], &[0], ArenaLayout::Packed);
        assert!(arena.acquire(4096).is_none(), "served outside a step");
        arena.begin_step();
        assert!(arena.acquire(4096).is_some());
        arena.end_step();
        assert!(arena.acquire(4096).is_none(), "served after the step ended");
    }

    /// A step longer than the plan falls through to the pool rather than
    /// wrapping around and re-serving slot 0, which would alias two live values.
    #[test]
    fn allocations_past_the_plan_are_declined() {
        let (arena, _pool) = arena_of(&[4096], &[0], ArenaLayout::Packed);
        arena.begin_step();
        assert!(arena.acquire(4096).is_some());
        assert!(arena.acquire(4096).is_none(), "wrapped past the plan");
        assert_eq!(arena.counters().declined_exhausted, 1);
    }

    /// A view addresses its own region: its length is the slot's, not the
    /// arena's, and its CPU pointer starts at the slot.
    ///
    /// Both halves are silent-corruption guards. `fill_buffer` takes
    /// `buffer.length()` as a range, so a view reporting the parent's length
    /// would blit across every slot after it; `read_to_vec` takes `contents()`
    /// with no offset, so a view reporting the parent's pointer would read the
    /// wrong slot's bytes.
    #[test]
    fn a_view_addresses_only_its_own_region() {
        let dev = device();
        let base = dev
            .new_buffer(4096, crate::RESOURCE_OPTIONS)
            .expect("allocation");
        let parent_ptr = base.contents() as usize;

        let view = base.view(128, 256);
        assert_eq!(view.length(), 256, "view reported the parent's length");
        assert_eq!(view.base_offset(), 128);
        assert_eq!(
            view.contents() as usize,
            parent_ptr + 128,
            "view's CPU pointer did not start at its slot"
        );

        // A view of a view composes rather than resetting.
        let inner = view.view(64, 64);
        assert_eq!(inner.base_offset(), 192);
        assert_eq!(inner.length(), 64);
        assert_eq!(inner.contents() as usize, parent_ptr + 192);

        // An ordinary buffer is unaffected: offset 0, full length.
        assert_eq!(base.base_offset(), 0);
        assert_eq!(base.length(), 4096);
    }

    /// **The trap #68 hit and self-caught** (§9.2c), pinned so this port cannot
    /// re-make it.
    ///
    /// One buffer reused for three unrelated values, one after another. Keyed on
    /// the *value* -- one allocation event each -- their intervals are disjoint
    /// and they need one slot. Keyed on the *buffer* they would merge into one
    /// interval spanning the whole step, which invents a lifetime no activation
    /// has; #68's first planner did that and reported 3.55 MB against the true
    /// 68 KB, a 52x overstatement.
    ///
    /// The recorder cannot make the mistake because it never sees a buffer: the
    /// token identifies the allocation event. This test states that as a
    /// property rather than trusting the absence.
    #[test]
    fn liveness_keys_on_the_value_not_on_the_recycled_buffer() {
        let mut rec = ArenaRecorder::new();

        // Three values that happen to land in the same pooled buffer, each
        // fully consumed before the next is allocated -- the `buf#328` pattern,
        // which #68 observed 13 times per token.
        for token in 0..3u64 {
            rec.record_alloc(token, 21504);
            rec.record_free(token);
        }

        let plan = rec.plan(ArenaLayout::Packed);
        assert_eq!(
            plan.slots().len(),
            1,
            "three sequentially-dead values did not share one slot: {:?}",
            plan.slots()
        );
        assert_eq!(
            plan.arena_bytes(),
            21504,
            "peak is {} B; keying on the buffer would inflate it",
            plan.arena_bytes()
        );
        // All three ordinals resolve to the same slot, which is the positional
        // stability the arena exists for.
        let slots: Vec<usize> = plan.by_ordinal.iter().map(|&(s, _)| s).collect();
        assert_eq!(slots, vec![0, 0, 0]);
    }

    /// Values that are live at the same time must not share a slot, however
    /// they were allocated. The counterpart to the test above: it would pass
    /// trivially if the recorder simply gave everything its own slot.
    #[test]
    fn simultaneously_live_values_do_not_share() {
        let mut rec = ArenaRecorder::new();
        rec.record_alloc(0, 21504);
        rec.record_alloc(1, 21504);
        rec.record_alloc(2, 21504);
        // All three live together -- the SwiGLU moment #68 identifies as the
        // peak: silu(gate), up, and their product.
        rec.record_free(0);
        rec.record_free(1);
        rec.record_free(2);

        let plan = rec.plan(ArenaLayout::Packed);
        assert_eq!(plan.slots().len(), 3, "live values shared a slot");
        assert_eq!(plan.arena_bytes(), 3 * 21504);
    }

    /// A value never freed is live to the end of the step, so it shares with
    /// nothing. Conservative in the safe direction: a larger arena, never two
    /// live values on one slot. #68 counts 39 of these.
    #[test]
    fn a_value_never_freed_shares_with_nothing() {
        let mut rec = ArenaRecorder::new();
        rec.record_alloc(0, 4096); // never freed
        rec.record_alloc(1, 4096);
        rec.record_free(1);
        rec.record_alloc(2, 4096);
        rec.record_free(2);

        let plan = rec.plan(ArenaLayout::Packed);
        let slots: Vec<usize> = plan.by_ordinal.iter().map(|&(s, _)| s).collect();
        assert_ne!(slots[0], slots[1], "a live value shared with a later one");
        assert_ne!(slots[0], slots[2], "a live value shared with a later one");
        assert_eq!(slots[1], slots[2], "two dead values did not share");
        assert_eq!(plan.slots().len(), 2);
    }

    /// The plan's byte total is the peak, and it is what the arena allocates.
    #[test]
    fn arena_bytes_is_the_packed_peak() {
        // #68's shape: three MLP-width values live together at the SwiGLU, plus
        // the residual stream and a kv-head value.
        let sizes = [21504, 21504, 21504, 4096, 1024];
        let last_use = [4, 4, 4, 4, 4];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        // All five are live together, so none can share: 3*21504 + 4096 + 1024,
        // each slot 128 B aligned. 21504 and 4096 and 1024 are all multiples of
        // 128, so the total is exact.
        assert_eq!(plan.arena_bytes(), 3 * 21504 + 4096 + 1024);
        assert_eq!(plan.arena_bytes(), 69632, "not #68's measured 68.00 KB");
        assert_eq!(plan.slots().len(), 5);
    }
}
