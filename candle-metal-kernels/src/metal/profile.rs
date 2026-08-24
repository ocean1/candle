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

/// Record that a dispatch was encoded, attributed to `label`.
#[inline]
pub fn record_dispatch(label: &str) {
    if !enabled() {
        return;
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut rec = recorder().lock().unwrap();
    *rec.dispatches.entry(label.to_string()).or_insert(0) += 1;
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

    Snapshot {
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        encoders: ENCODERS.load(Ordering::Relaxed),
        command_buffers: COMMAND_BUFFERS.load(Ordering::Relaxed),
        timed_command_buffers: intervals.len() as u64,
        gpu_busy_sum_s: sum,
        gpu_busy_union_s: union,
        gpu_span_s: span,
        by_label,
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
}
