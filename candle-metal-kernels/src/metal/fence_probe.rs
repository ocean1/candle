//! Counting the cross-encoder fence waits, and what a range test would do to them.
//!
//! `DESIGN.md` §9.2e gives intra-encoder hazards a precision axis --
//! [`HazardKey::Pointer`](super::encoder::HazardKey) against
//! [`HazardKey::Range`](super::encoder::HazardKey) -- and cross-encoder ordering
//! has none: `prev_ce_outputs` is `HashMap<usize, Arc<Fence>>`, keyed on the
//! buffer pointer with no offset and no length. So a read of *any* byte of a
//! buffer waits on the fence of the last encoder session that wrote *any other*
//! byte of it.
//!
//! The comment at `encoder.rs`'s `HazardKey::Pointer` justifies `Range` by
//! saying that *under an arena every activation shares one pointer, so every
//! write-then-read pair inside it becomes a false dependency*. That reasoning is
//! exactly as true across encoder sessions as within one, and only the
//! within-one case was fixed. Whether it bites is a **count**, and this module
//! is the counter.
//!
//! # Why this cannot be computed from the existing binding trace
//!
//! `trace::record_binding` carries `(index, ptr, offset, is_output)` and **no
//! length**, and an interval test needs the extent on both sides. Worse, the
//! writer's ranges are discarded at exactly the point the fence map is built:
//! `EncoderState::all_outputs` is a `HashSet<usize>` of pointers, so by
//! `end_encoding` the bytes an encoder wrote are already gone. Recording the
//! written ranges alongside is therefore new state, not a re-reading of state
//! that exists -- which is itself the finding `DESIGN.md` §6.7 L2 inverts: here
//! the information is *not* already present.
//!
//! # What is recorded, and why each field is needed
//!
//! One event per `wait_for_buffer` call, at the call's own site rather than
//! reconstructed downstream -- §9.2f's rule that a barrier count cannot be
//! simulated applies with more force here, since a wait depends on the
//! `waited_fences` dedup set, which is per session and invisible to a trace.
//!
//! * `emitted` separates the calls that actually reach `waitForFence` from the
//!   ones the per-session dedup swallows. The ceiling is sessions x distinct
//!   buffers and the arena collapses the second term toward 1, which cuts
//!   **both** ways -- it is the reason this is measured rather than argued.
//! * `reader` is the range being bound; `writer_ranges` is what the encoder
//!   holding the fence actually wrote to that same pointer. An offline range
//!   test is `reader.overlaps(w)` for any `w` -- the *same* predicate
//!   `BoundRange::overlaps` gives the intra-encoder path, so the offline test
//!   cannot drift from the one route (a) would install.
//! * `arena` marks the events whose buffer is the arena allocation, which is the
//!   population the issue is about.
//!
//! Off unless `CANDLE_METAL_FENCE_PROBE=1`. When off the cost is one relaxed
//! atomic load per call and nothing else -- no lock, no allocation -- because a
//! probe that perturbs the path it measures is `DESIGN.md` §2.4's first trap.

use super::encoder::BoundRange;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One `wait_for_buffer` call.
#[derive(Clone)]
pub struct WaitEvent {
    /// Monotonic index across the whole probe run.
    pub seq: u64,
    /// Which binding site called: see [`CallSite`].
    pub site: CallSite,
    /// Encoder session this call was made in, counted from probe start.
    ///
    /// The dedup set is per session, so a wait count is uninterpretable without
    /// it: two calls in different sessions against the same fence both emit.
    pub encoder: u64,
    /// The range being bound by the reader.
    pub reader: BoundRange,
    /// Whether the map held a fence for this pointer at all.
    pub found_fence: bool,
    /// Whether `waitForFence` was actually called -- false when the session had
    /// already waited on this fence.
    pub emitted: bool,
    /// Identity of the fence waited on, so events can be grouped by writer.
    pub fence: usize,
    /// The ranges the fence's encoder wrote to *this pointer*.
    ///
    /// Empty when no fence was found. This is the half a range-keyed
    /// `prev_ce_outputs` would have to carry, and the half nothing records
    /// today.
    pub writer_ranges: Vec<BoundRange>,
    /// Whether this pointer is the arena allocation.
    pub arena: bool,
    /// Caller-supplied marker, e.g. a decode step. Shares `trace`'s region so
    /// the two can be read against each other.
    pub region: Option<String>,
}

/// Which of the binding sites made the call.
///
/// `DESIGN.md` and issue #136 both say `wait_for_buffer` has two callers; there
/// are **three**. The third is `capture_array`, on the packed-params path, and
/// it is only reached while a param capture is open. Recorded separately rather
/// than folded in, so a count taken under `ParamStyle::Packed` cannot silently
/// attribute staging-buffer waits to the model's own reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallSite {
    /// `set_input_buffer` -- read-after-write against an earlier encoder.
    Input,
    /// `set_output_buffer` -- write-after-write, and write-after-read.
    Output,
    /// `capture_array` -- the packed-params staging buffer.
    CaptureArray,
}

impl CallSite {
    pub fn as_str(self) -> &'static str {
        match self {
            CallSite::Input => "input",
            CallSite::Output => "output",
            CallSite::CaptureArray => "capture_array",
        }
    }
}

struct ProbeState {
    events: Vec<WaitEvent>,
    next_seq: u64,
    encoder: u64,
    region: Option<String>,
    /// Pointer of the arena allocation, once one is installed.
    arena_ptr: Option<usize>,
}

fn state() -> &'static Mutex<ProbeState> {
    static STATE: OnceLock<Mutex<ProbeState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(ProbeState {
            events: Vec::new(),
            next_seq: 0,
            encoder: 0,
            region: None,
            arena_ptr: None,
        })
    })
}

/// Whether the probe was requested via `CANDLE_METAL_FENCE_PROBE`.
///
/// Read once into a `OnceLock`, for the reason `trace::enabled` gives: a
/// per-call `std::env::var` would be a lock and an allocation on a path taken
/// 1799 times per decode token (`DESIGN.md` §6.4a).
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CANDLE_METAL_FENCE_PROBE")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false)
    })
}

/// Set independently of `enabled()`, so recording can be gated to a region --
/// counting every prefill wait to answer a question about decode would bury the
/// answer.
static RECORDING: AtomicBool = AtomicBool::new(false);

/// Calls observed while recording was off, so the probe reports what it did not
/// see rather than implying it saw everything.
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// True when a call should be recorded.
#[inline]
pub fn is_recording() -> bool {
    enabled() && RECORDING.load(Ordering::Relaxed)
}

/// Turn recording on or off. No-op unless `CANDLE_METAL_FENCE_PROBE` is set.
pub fn set_recording(on: bool) {
    if enabled() {
        RECORDING.store(on, Ordering::Relaxed);
    }
}

/// Whether the probe was requested, for a harness to report when it produced
/// nothing because the variable was unset.
pub fn probe_requested() -> bool {
    enabled()
}

/// Label subsequent events, e.g. with a decode step index.
pub fn set_region(region: Option<String>) {
    if !enabled() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.region = region;
    }
}

/// Record that a new encoder session began.
///
/// The `waited_fences` dedup is per session, so the session index is what makes
/// a wait count interpretable at all.
#[inline]
pub fn note_encoder_begin() {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.encoder += 1;
    }
}

/// Tell the probe which allocation is the arena's.
///
/// Called when an arena is installed. Without it every event reports
/// `arena=false` and the attribution question -- *how many of the waits are the
/// arena's* -- cannot be answered, which would make the probe vacuous in
/// precisely the arm it exists to judge (`DESIGN.md` §9.2f).
pub fn set_arena_ptr(ptr: Option<usize>) {
    if !enabled() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.arena_ptr = ptr;
    }
}

/// Whether an arena pointer has been registered, so a harness can prove the
/// attribution arm engaged rather than reporting a zero it cannot interpret.
pub fn arena_ptr() -> Option<usize> {
    if !enabled() {
        return None;
    }
    state().lock().ok().and_then(|s| s.arena_ptr)
}

/// Record one `wait_for_buffer` call.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn record_wait(
    site: CallSite,
    reader: BoundRange,
    found_fence: bool,
    emitted: bool,
    fence: usize,
    writer_ranges: Vec<BoundRange>,
) {
    if !enabled() {
        return;
    }
    if !RECORDING.load(Ordering::Relaxed) {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if let Ok(mut s) = state().lock() {
        let seq = s.next_seq;
        s.next_seq += 1;
        let encoder = s.encoder;
        let region = s.region.clone();
        let arena = s.arena_ptr == Some(reader.ptr);
        s.events.push(WaitEvent {
            seq,
            site,
            encoder,
            reader,
            found_fence,
            emitted,
            fence,
            writer_ranges,
            arena,
            region,
        });
    }
}

/// Fence identity -> the ranges that fence's encoder wrote, per pointer.
///
/// Populated at `end_encoding`, which is where `prev_ce_outputs` is built and
/// therefore the last moment the writer's bytes are known. Entries are dropped
/// when the fence's own map entries are cleaned in the completion handler, so
/// this cannot outgrow the map it shadows.
fn writers() -> &'static Mutex<std::collections::HashMap<usize, Vec<(usize, BoundRange)>>> {
    static WRITERS: OnceLock<Mutex<std::collections::HashMap<usize, Vec<(usize, BoundRange)>>>> =
        OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Record what `fence`'s encoder wrote, so a later wait on it is decidable.
pub fn note_writer_ranges(
    fence: usize,
    ranges: &std::collections::HashMap<usize, Vec<BoundRange>>,
) {
    if !is_recording() {
        return;
    }
    if let Ok(mut w) = writers().lock() {
        let entry = w.entry(fence).or_default();
        for (ptr, rs) in ranges {
            for r in rs {
                entry.push((*ptr, *r));
            }
        }
    }
}

/// The ranges `fence`'s encoder wrote to `ptr`.
///
/// Empty when the fence predates the probe being turned on, which
/// `survives_range_test` reports as undecidable rather than as either verdict.
pub fn writer_ranges(fence: usize, ptr: usize) -> Vec<BoundRange> {
    if !is_recording() {
        return Vec::new();
    }
    match writers().lock() {
        Ok(w) => w
            .get(&fence)
            .map(|v| {
                v.iter()
                    .filter(|(p, _)| *p == ptr)
                    .map(|(_, r)| *r)
                    .collect()
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Forget `fence`'s ranges, called where its map entries are cleaned.
pub fn forget_writer(fence: usize) {
    if !enabled() {
        return;
    }
    if let Ok(mut w) = writers().lock() {
        w.remove(&fence);
    }
}

/// Every event recorded so far, draining the buffer.
pub fn take_events() -> Vec<WaitEvent> {
    if !enabled() {
        return Vec::new();
    }
    match state().lock() {
        Ok(mut s) => std::mem::take(&mut s.events),
        Err(_) => Vec::new(),
    }
}

/// Calls seen while recording was off.
pub fn skipped_count() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// What a run of events says, once summed.
///
/// Computed here rather than in the harness so the offline range test has
/// exactly one implementation, and so it is unit-testable without a GPU.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// `wait_for_buffer` calls.
    pub calls: u64,
    /// Calls that found a fence in the map.
    pub found: u64,
    /// Calls that reached `waitForFence` -- the quantity the issue asks for.
    pub emitted: u64,
    /// Emitted waits whose buffer is the arena allocation.
    pub emitted_arena: u64,
    /// Emitted waits against the arena that **survive** an offline range test:
    /// the reader's bytes overlap something the writer actually wrote.
    pub emitted_arena_survives_range: u64,
    /// Emitted waits against the arena that a range test would **drop**: the
    /// reader and the writer provably touch disjoint bytes.
    ///
    /// This is the false-dependency count, and it is the verdict.
    pub emitted_arena_false: u64,
    /// The same split over the non-arena population, for the pool arm and as a
    /// control: a false wait outside the arena is the pool's own aliasing, and
    /// §9.2f measured the intra-encoder equivalent at zero.
    pub emitted_other_survives_range: u64,
    pub emitted_other_false: u64,
    /// Emitted waits whose writer ranges were not recorded, so the range test
    /// is undecidable for them.
    ///
    /// Reported rather than folded into either side: an undecidable event
    /// counted as "survives" understates the false count and counted as "false"
    /// overstates it, and silently choosing either is the failure this field
    /// exists to prevent.
    pub emitted_undecidable: u64,
    /// Distinct encoder sessions the events span.
    pub sessions: u64,
    /// Calls per site.
    pub by_site: Vec<(CallSite, u64)>,
}

/// Whether a range-keyed `prev_ce_outputs` would still have ordered this wait.
///
/// `None` when the writer's ranges were not recorded, so the caller must decide
/// what to do with an undecidable event rather than inheriting a default.
pub fn survives_range_test(e: &WaitEvent) -> Option<bool> {
    if e.writer_ranges.is_empty() {
        return None;
    }
    Some(e.writer_ranges.iter().any(|w| w.overlaps(&e.reader)))
}

/// Sum a run of events.
pub fn summarize(events: &[WaitEvent]) -> Summary {
    let mut s = Summary::default();
    let mut sessions = std::collections::HashSet::new();
    let mut sites: Vec<(CallSite, u64)> = Vec::new();
    for e in events {
        s.calls += 1;
        sessions.insert(e.encoder);
        match sites.iter_mut().find(|(site, _)| *site == e.site) {
            Some((_, n)) => *n += 1,
            None => sites.push((e.site, 1)),
        }
        if e.found_fence {
            s.found += 1;
        }
        if !e.emitted {
            continue;
        }
        s.emitted += 1;
        if e.arena {
            s.emitted_arena += 1;
        }
        match survives_range_test(e) {
            None => s.emitted_undecidable += 1,
            Some(true) => {
                if e.arena {
                    s.emitted_arena_survives_range += 1;
                } else {
                    s.emitted_other_survives_range += 1;
                }
            }
            Some(false) => {
                if e.arena {
                    s.emitted_arena_false += 1;
                } else {
                    s.emitted_other_false += 1;
                }
            }
        }
    }
    s.sessions = sessions.len() as u64;
    sites.sort_by_key(|(site, _)| site.as_str());
    s.by_site = sites;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(ptr: usize, offset: usize, len: usize) -> BoundRange {
        BoundRange { ptr, offset, len }
    }

    fn ev(
        reader: BoundRange,
        writer_ranges: Vec<BoundRange>,
        emitted: bool,
        arena: bool,
    ) -> WaitEvent {
        WaitEvent {
            seq: 0,
            site: CallSite::Input,
            encoder: 0,
            reader,
            found_fence: !writer_ranges.is_empty(),
            emitted,
            fence: 1,
            writer_ranges,
            arena,
            region: None,
        }
    }

    #[test]
    fn disjoint_ranges_in_one_allocation_are_a_false_dependency() {
        // The shape the issue is about: reader and writer share a pointer and
        // touch no common byte, which pointer keying cannot tell apart.
        let e = ev(r(9, 0, 100), vec![r(9, 100, 100)], true, true);
        assert_eq!(survives_range_test(&e), Some(false));
        let s = summarize(&[e]);
        assert_eq!(s.emitted, 1);
        assert_eq!(s.emitted_arena, 1);
        assert_eq!(s.emitted_arena_false, 1);
        assert_eq!(s.emitted_arena_survives_range, 0);
    }

    #[test]
    fn overlapping_ranges_survive() {
        let e = ev(r(9, 50, 100), vec![r(9, 100, 100)], true, true);
        assert_eq!(survives_range_test(&e), Some(true));
        assert_eq!(summarize(&[e]).emitted_arena_survives_range, 1);
    }

    #[test]
    fn a_wait_that_was_deduped_is_not_counted_as_emitted() {
        // The ceiling is sessions x distinct buffers precisely because of this
        // dedup, so an event that did not reach `waitForFence` must not enter
        // the emitted population.
        let e = ev(r(9, 0, 100), vec![r(9, 0, 100)], false, true);
        let s = summarize(&[e]);
        assert_eq!(s.calls, 1);
        assert_eq!(s.emitted, 0);
        assert_eq!(s.emitted_arena, 0);
    }

    #[test]
    fn an_unrecorded_writer_is_undecidable_rather_than_either_verdict() {
        let e = ev(r(9, 0, 100), vec![], true, true);
        assert_eq!(survives_range_test(&e), None);
        let s = summarize(&[e]);
        assert_eq!(s.emitted_undecidable, 1);
        assert_eq!(s.emitted_arena_false, 0);
        assert_eq!(s.emitted_arena_survives_range, 0);
    }

    #[test]
    fn a_zero_length_binding_fails_toward_ordering() {
        // `BoundRange::overlaps` treats an unknown extent as the whole
        // allocation, so an event with one cannot be reported as false. The
        // offline test inherits that, which is the point of reusing the
        // predicate rather than writing a second one.
        let e = ev(r(9, 0, 0), vec![r(9, 4096, 100)], true, true);
        assert_eq!(survives_range_test(&e), Some(true));
    }

    #[test]
    fn different_allocations_never_overlap() {
        let e = ev(r(9, 0, 100), vec![r(10, 0, 100)], true, false);
        assert_eq!(survives_range_test(&e), Some(false));
    }

    #[test]
    fn sessions_and_sites_are_counted_separately() {
        let mut a = ev(r(9, 0, 10), vec![r(9, 0, 10)], true, false);
        a.encoder = 0;
        a.site = CallSite::Input;
        let mut b = ev(r(9, 0, 10), vec![r(9, 0, 10)], true, false);
        b.encoder = 1;
        b.site = CallSite::Output;
        let s = summarize(&[a, b]);
        assert_eq!(s.sessions, 2);
        assert_eq!(s.emitted, 2);
        assert_eq!(s.by_site.len(), 2);
    }
}
