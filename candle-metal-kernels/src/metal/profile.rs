//! Opt-in profiling of the Metal submission path: GPU busy time and dispatch count.
//!
//! The question this exists to answer is where a decode token's wall time goes.
//! Metal reports `GPUStartTime`/`GPUEndTime` per command buffer, so summing
//! `end - start` over every command buffer in a window gives GPU busy time, and
//! the window's wall time minus that is time the GPU was not executing --
//! which for a serial decode loop is CPU-side encode, submission and readback.
//!
//! Two properties of the measurement bound what it can conclude:
//!
//! * **Busy time is per command buffer, not per dispatch.** Dispatches inside one
//!   encoder overlap (`MTLDispatchType::Concurrent`), so a per-dispatch sum would
//!   double-count. The command buffer is the smallest unit Metal reports that
//!   does not.
//! * **Command buffers can overlap each other.** Candle commits every
//!   `CANDLE_METAL_COMPUTE_PER_BUFFER` encoders without waiting, so two buffers
//!   can be in flight at once and their intervals can intersect. Summing them
//!   would then exceed true busy time. The union of the intervals is reported
//!   alongside the sum; where they differ the union is the honest figure, and the
//!   gap between them measures how much the submission pipeline actually overlaps.
//!
//! Enable with `CANDLE_METAL_PROFILE=1`. Disabled, every hook is a cached bool
//! check and nothing is recorded -- it must be able to run in a release build
//! against the real model, since an instrumented debug build would measure a
//! different dispatch pattern than the one that ships.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Whether profiling was requested, read once from the environment.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("CANDLE_METAL_PROFILE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// One command buffer's GPU execution interval, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuInterval {
    pub start: f64,
    pub end: f64,
}

#[derive(Default)]
struct Recorder {
    /// GPU intervals of every command buffer that has completed.
    intervals: Vec<GpuInterval>,
    /// Dispatches per pipeline label, for the per-token dispatch inventory.
    dispatches: HashMap<String, u64>,
}

fn recorder() -> &'static Mutex<Recorder> {
    static RECORDER: OnceLock<Mutex<Recorder>> = OnceLock::new();
    RECORDER.get_or_init(|| Mutex::new(Recorder::default()))
}

/// Dispatches encoded since the last reset. Separate from the per-label map so
/// the count is available even where no label is supplied.
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// Command buffers committed since the last reset.
static COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
/// Compute encoders opened since the last reset. Encoder breaks are what
/// serialize the GPU (`DESIGN.md` §3.5), so the count belongs next to the
/// dispatch count rather than being inferred from it.
static ENCODERS: AtomicU64 = AtomicU64::new(0);

// ---- per-bind fence probe (lloom issue #24) --------------------------------
//
// `ComputeCommandEncoder::wait_for_buffer` runs once per bound buffer, from
// both `set_input_buffer` and `set_output_buffer`. Issue #24 asks which of its
// five steps dominates: the map mutex, the lookup, the `Arc` clone on a hit,
// the second mutex, and the `waited_fences` dedup.
//
// These are counts, not timers, and that is deliberate. A `Instant::now()` pair
// costs 43 ns on this machine, measured, against a probe whose whole body is
// tens of nanoseconds -- so timing each call would measure the timer
// (`CONTRIBUTING.md` §3.2, `DESIGN.md` §2.4). Counting is one relaxed atomic
// increment, and the per-operation cost comes from an isolated microbenchmark
// where the timer can be amortized over a long loop instead.
//
// The three counters partition every call, so `binds == miss + dedup + wait`
// is an invariant a reader can check rather than trust.

/// Calls to `wait_for_buffer`, i.e. buffers bound. Both bind sites funnel here.
static BIND_PROBES: AtomicU64 = AtomicU64::new(0);
/// Probes that found no pending writer: one mutex, one failed lookup, return.
/// The cheapest path, and the issue's hypothesis is that it is also the common
/// one ("most binds hit buffers with no pending writer at all").
static BIND_PROBE_MISSES: AtomicU64 = AtomicU64::new(0);
/// Probes that found a fence already waited on this encoder session: the full
/// cost of a hit (mutex, lookup, `Arc` clone, second mutex, `HashSet` probe)
/// for no emitted wait.
static BIND_PROBE_DEDUPED: AtomicU64 = AtomicU64::new(0);
/// Probes that emitted `waitForFence`. The only ones that produce a fence edge.
static BIND_PROBE_WAITS: AtomicU64 = AtomicU64::new(0);
/// Entries in `prev_ce_outputs` at the moment of a probe, summed, so the mean
/// map size is available. The lookup is a hash probe, so this does not set its
/// cost -- it bounds how much memory the probe touches, and it is the figure
/// that grew without bound in the issue #2 defect this map replaced.
static BIND_PROBE_MAP_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// Record one `wait_for_buffer` call and which of its three outcomes it took.
///
/// `map_len` is the size of `prev_ce_outputs` as observed under the lock that
/// the probe already holds, so reading it adds no synchronization.
#[inline]
pub fn record_bind_probe(map_len: usize, outcome: BindProbeOutcome) {
    if !enabled() {
        return;
    }
    BIND_PROBES.fetch_add(1, Ordering::Relaxed);
    BIND_PROBE_MAP_ENTRIES.fetch_add(map_len as u64, Ordering::Relaxed);
    match outcome {
        BindProbeOutcome::NoPendingWriter => &BIND_PROBE_MISSES,
        BindProbeOutcome::AlreadyWaited => &BIND_PROBE_DEDUPED,
        BindProbeOutcome::Waited => &BIND_PROBE_WAITS,
    }
    .fetch_add(1, Ordering::Relaxed);
}

/// Which path a `wait_for_buffer` call took. Exhaustive: every call is one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindProbeOutcome {
    /// No entry in `prev_ce_outputs` for this buffer.
    NoPendingWriter,
    /// A fence was found, but this encoder session had already waited on it.
    AlreadyWaited,
    /// A fence was found and `waitForFence` was emitted.
    Waited,
}

// ---- blit destination wait (lloom issue #25) -------------------------------
//
// `copy_from_buffer` waited on its source but not its destination, where
// `fill_buffer` waited on its destination. These count how often the added
// destination wait finds a pending writer at all, so "is this edge ever
// live?" is answered by measurement rather than by argument.

/// `copy_from_buffer` calls.
static BLIT_COPIES: AtomicU64 = AtomicU64::new(0);
/// Of those, calls whose destination had a registered last writer.
static BLIT_COPY_DST_PENDING: AtomicU64 = AtomicU64::new(0);
/// Of those, calls whose destination writer was a *blit* fence -- the case the
/// blanket `live_fences` wait in `blit_command_encoder` does not cover, because
/// `BlitCommandEncoder::end_encoding` never registers its fence there.
static BLIT_COPY_DST_UNCOVERED: AtomicU64 = AtomicU64::new(0);

/// Record one `copy_from_buffer` and what its destination wait found.
#[inline]
pub fn record_blit_copy(dst_had_writer: bool, dst_writer_uncovered: bool) {
    if !enabled() {
        return;
    }
    BLIT_COPIES.fetch_add(1, Ordering::Relaxed);
    if dst_had_writer {
        BLIT_COPY_DST_PENDING.fetch_add(1, Ordering::Relaxed);
    }
    if dst_writer_uncovered {
        BLIT_COPY_DST_UNCOVERED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether the per-kernel dispatch inventory is wanted, read once.
///
/// Building it costs a `String` allocation and a `HashMap` probe *per
/// dispatch* — 675 of each per decode token. Sampling the decode path showed
/// that this is ~3.6 % of forward-pass CPU samples, which is the same order as
/// the per-bind fence probe this profiler exists to measure. An instrument that
/// large cannot sit inside the measurement of something that small
/// (`CONTRIBUTING.md` §3.2, `DESIGN.md` §2.4), so the inventory is opt-in and
/// off by default: `CANDLE_METAL_PROFILE=1` counts, and
/// `CANDLE_METAL_PROFILE_KERNELS=1` additionally attributes.
pub fn kernel_inventory_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("CANDLE_METAL_PROFILE_KERNELS").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Record that a dispatch was encoded, attributed to `label`.
///
/// The count is one relaxed atomic. The per-label attribution is skipped unless
/// `CANDLE_METAL_PROFILE_KERNELS=1`, because it allocates.
#[inline]
pub fn record_dispatch(label: &str) {
    if !enabled() {
        return;
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    if !kernel_inventory_enabled() {
        return;
    }
    let mut rec = recorder().lock().unwrap();
    // `entry` on a `&str` key still allocates on the miss path only; the
    // repeated-label case that dominates decode takes the cheaper lookup.
    if let Some(c) = rec.dispatches.get_mut(label) {
        *c += 1;
    } else {
        rec.dispatches.insert(label.to_string(), 1);
    }
}

/// Record that a compute encoder was opened.
#[inline]
pub fn record_encoder() {
    if !enabled() {
        return;
    }
    ENCODERS.fetch_add(1, Ordering::Relaxed);
}

/// Record a completed command buffer's GPU interval.
#[inline]
pub fn record_command_buffer(start: f64, end: f64) {
    if !enabled() {
        return;
    }
    COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
    // A command buffer that encoded no GPU work reports zeros; recording it
    // would drag the union's lower bound back to the epoch.
    if start <= 0.0 || end <= start {
        return;
    }
    recorder()
        .lock()
        .unwrap()
        .intervals
        .push(GpuInterval { start, end });
}

/// Merge sorted intervals, returning (union duration, span from first start to last end).
///
/// Split out from [`snapshot`] so the overlap handling -- the part that is easy
/// to get wrong and the reason the union is reported at all -- is testable
/// without a GPU.
fn union_and_span(intervals: &mut [GpuInterval]) -> (f64, f64) {
    if intervals.is_empty() {
        return (0.0, 0.0);
    }
    intervals.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut union = 0.0f64;
    let mut cur = intervals[0];
    for iv in &intervals[1..] {
        if iv.start <= cur.end {
            if iv.end > cur.end {
                cur.end = iv.end;
            }
        } else {
            union += cur.end - cur.start;
            cur = *iv;
        }
    }
    union += cur.end - cur.start;

    let first_start = intervals[0].start;
    let last_end = intervals.iter().map(|i| i.end).fold(f64::MIN, f64::max);
    (union, last_end - first_start)
}

/// A profiling window's totals.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub dispatches: u64,
    pub encoders: u64,
    pub command_buffers: u64,
    /// Command buffers that reported a usable interval. Lower than
    /// `command_buffers` when some encoded no GPU work, or when the caller
    /// snapshots before every completion handler has run.
    pub timed_command_buffers: u64,
    /// Sum of per-command-buffer GPU intervals. Over-counts when buffers overlap.
    pub gpu_busy_sum_s: f64,
    /// Union of those intervals: time during which at least one command buffer
    /// was executing. This is the figure to trust.
    pub gpu_busy_union_s: f64,
    /// From the first command buffer's start to the last one's end.
    pub gpu_span_s: f64,
    /// Dispatch counts by pipeline label, descending.
    pub by_label: Vec<(String, u64)>,
    /// `wait_for_buffer` calls: one per bound buffer (lloom issue #24).
    pub bind_probes: u64,
    /// Of those, the ones that found no pending writer.
    pub bind_probe_misses: u64,
    /// Of those, the ones that found a fence already waited on this session.
    pub bind_probe_deduped: u64,
    /// Of those, the ones that emitted `waitForFence`. These are the fence
    /// edges; none may be lost by an optimization (`DESIGN.md` §2.3.2).
    pub bind_probe_waits: u64,
    /// Mean entries in `prev_ce_outputs` when a probe ran.
    pub bind_probe_mean_map_entries: f64,
    /// `copy_from_buffer` calls (lloom issue #25).
    pub blit_copies: u64,
    /// Of those, calls whose destination had a registered last writer.
    pub blit_copy_dst_pending: u64,
    /// Of those, calls whose destination writer was not already covered by the
    /// blanket `live_fences` wait -- the edge the added destination wait adds.
    pub blit_copy_dst_uncovered: u64,
}

/// Take the current totals without clearing them.
pub fn snapshot() -> Snapshot {
    let rec = recorder().lock().unwrap();

    let mut intervals = rec.intervals.clone();
    let sum: f64 = intervals.iter().map(|i| i.end - i.start).sum();
    let (union, span) = union_and_span(&mut intervals);

    let mut by_label: Vec<(String, u64)> = rec
        .dispatches
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    by_label.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let bind_probes = BIND_PROBES.load(Ordering::Relaxed);
    let map_entries = BIND_PROBE_MAP_ENTRIES.load(Ordering::Relaxed);

    Snapshot {
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        encoders: ENCODERS.load(Ordering::Relaxed),
        command_buffers: COMMAND_BUFFERS.load(Ordering::Relaxed),
        timed_command_buffers: intervals.len() as u64,
        gpu_busy_sum_s: sum,
        gpu_busy_union_s: union,
        gpu_span_s: span,
        by_label,
        bind_probes,
        bind_probe_misses: BIND_PROBE_MISSES.load(Ordering::Relaxed),
        bind_probe_deduped: BIND_PROBE_DEDUPED.load(Ordering::Relaxed),
        bind_probe_waits: BIND_PROBE_WAITS.load(Ordering::Relaxed),
        bind_probe_mean_map_entries: if bind_probes == 0 {
            0.0
        } else {
            map_entries as f64 / bind_probes as f64
        },
        blit_copies: BLIT_COPIES.load(Ordering::Relaxed),
        blit_copy_dst_pending: BLIT_COPY_DST_PENDING.load(Ordering::Relaxed),
        blit_copy_dst_uncovered: BLIT_COPY_DST_UNCOVERED.load(Ordering::Relaxed),
    }
}

/// Clear all counters, so the next window starts from zero.
///
/// Call after a device synchronization: an in-flight command buffer's completion
/// handler fires later and would otherwise land in the next window.
pub fn reset() {
    DISPATCHES.store(0, Ordering::Relaxed);
    ENCODERS.store(0, Ordering::Relaxed);
    COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    BIND_PROBES.store(0, Ordering::Relaxed);
    BIND_PROBE_MISSES.store(0, Ordering::Relaxed);
    BIND_PROBE_DEDUPED.store(0, Ordering::Relaxed);
    BIND_PROBE_WAITS.store(0, Ordering::Relaxed);
    BIND_PROBE_MAP_ENTRIES.store(0, Ordering::Relaxed);
    BLIT_COPIES.store(0, Ordering::Relaxed);
    BLIT_COPY_DST_PENDING.store(0, Ordering::Relaxed);
    BLIT_COPY_DST_UNCOVERED.store(0, Ordering::Relaxed);
    let mut rec = recorder().lock().unwrap();
    rec.intervals.clear();
    rec.dispatches.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two command buffers in flight together must not be double-counted --
    /// the whole reason the union is reported next to the sum.
    #[test]
    fn union_merges_overlapping_intervals() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
        ];
        let (union, span) = union_and_span(&mut ivs);
        // Sum would be 2 + 2 + 1 = 5; the occupied time is 3 + 1 = 4.
        assert_eq!(union, 4.0);
        assert_eq!(span, 6.0);
    }

    /// Disjoint intervals: union and sum agree, which is the case where the
    /// simpler figure would have been fine.
    #[test]
    fn union_equals_sum_when_disjoint() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 1.0,
            },
            GpuInterval {
                start: 2.0,
                end: 3.0,
            },
        ];
        let (union, span) = union_and_span(&mut ivs);
        assert_eq!(union, 2.0);
        assert_eq!(span, 3.0);
    }

    /// Input order must not matter; the sweep sorts first.
    #[test]
    fn union_is_order_independent() {
        let mut a = vec![
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
        ];
        let mut b = vec![
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
        ];
        assert_eq!(union_and_span(&mut a), union_and_span(&mut b));
    }

    /// A fully contained interval must not extend the union past the outer end.
    #[test]
    fn union_handles_containment() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 10.0,
            },
            GpuInterval {
                start: 2.0,
                end: 3.0,
            },
        ];
        let (union, _) = union_and_span(&mut ivs);
        assert_eq!(union, 10.0);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(union_and_span(&mut []), (0.0, 0.0));
    }

    /// The three bind-probe outcomes must partition every call, so that
    /// `probes == misses + deduped + waits` holds in the reported numbers. A
    /// reader checking that sum is checking the instrumentation, not trusting
    /// it, which is the point of reporting all four.
    ///
    /// Runs only when profiling is enabled, since the counters are inert
    /// otherwise; the harness sets `CANDLE_METAL_PROFILE` before the first
    /// `enabled()` call caches it.
    #[test]
    fn bind_probe_outcomes_partition_the_calls() {
        if !enabled() {
            // Not an skip-to-pass: with profiling off the counters are
            // deliberately inert, and that is asserted instead.
            record_bind_probe(7, BindProbeOutcome::Waited);
            assert_eq!(BIND_PROBES.load(Ordering::Relaxed), 0);
            return;
        }
        reset();
        record_bind_probe(10, BindProbeOutcome::NoPendingWriter);
        record_bind_probe(10, BindProbeOutcome::NoPendingWriter);
        record_bind_probe(20, BindProbeOutcome::AlreadyWaited);
        record_bind_probe(20, BindProbeOutcome::Waited);

        let s = snapshot();
        assert_eq!(s.bind_probes, 4);
        assert_eq!(s.bind_probe_misses, 2);
        assert_eq!(s.bind_probe_deduped, 1);
        assert_eq!(s.bind_probe_waits, 1);
        assert_eq!(
            s.bind_probes,
            s.bind_probe_misses + s.bind_probe_deduped + s.bind_probe_waits
        );
        assert_eq!(s.bind_probe_mean_map_entries, 15.0);
        reset();
    }
}
