//! Buffer-pool instrumentation for lloom issue #21.
//!
//! `DESIGN.md` §6.7 §9 is explicit that wall-clock hides this pathology: the
//! free-list scan is CPU work in front of GPU work, and at B=1 decode it is
//! ~6 % of a token that is otherwise 94 % GPU-bound. A 1 ms move inside an
//! 18.76 ms token is within the run-to-run spread of the wall figure, so the
//! honest measurement is the scan itself.
//!
//! What is counted, and why each one is here:
//!
//! * **buckets walked** — `find_available_buffer` iterates every key in the
//!   `HashMap` with no early exit, so this is the scan's actual length. It is
//!   the number issue #7 predicted would grow when 128 B alignment replaced
//!   next-power-of-two rounding, because finer granularity means more distinct
//!   keys.
//! * **buffers examined** — the inner loop does not `break`, so a bucket costs
//!   its whole length. Separating the two makes it visible whether the scan is
//!   wide (many buckets) or deep (long buckets).
//! * **lookups, hits, misses** — a miss walked the entire pool to conclude
//!   nothing was reusable, which is the worst case and the one that then
//!   allocates.
//! * **occupancy** — buffer count and resident bytes per pool, sampled rather
//!   than accumulated. This is the footprint that must not regress past issue
//!   #8's 5509 MB.
//! * **sweeps** — `drop_unused_buffers` is O(all buffers) and *destroys*
//!   rather than recycles, so the pool oscillates between growing and being
//!   swept. Counting sweeps and what they freed shows that oscillation.
//!
//! Everything is behind `CANDLE_METAL_POOL_STATS=1`. When unset the counters
//! are a single relaxed atomic load of a `bool`, so an instrumented binary and
//! a clean one measure the same thing — which matters, because §2.4 warns that
//! the measurement tool is sometimes the cost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// Relaxed throughout: these are monotonically-increasing diagnostic counters
/// read once at the end of a run, never used to order anything.
const ORD: Ordering = Ordering::Relaxed;

pub struct PoolStats {
    enabled: AtomicBool,

    /// Calls into `find_available_buffer`.
    pub lookups: AtomicU64,
    /// Calls that found a reusable buffer.
    pub hits: AtomicU64,
    /// Bucket keys visited, summed over all lookups. The scan length.
    pub buckets_walked: AtomicU64,
    /// Individual buffers `strong_count`-tested, summed over all lookups.
    pub buffers_examined: AtomicU64,
    /// Largest single-lookup bucket walk seen.
    pub max_buckets_walked: AtomicU64,
    /// Largest single-lookup buffer examination seen.
    pub max_buffers_examined: AtomicU64,

    /// Calls into `drop_unused_buffers`.
    pub sweeps: AtomicU64,
    /// Buffers destroyed by sweeps.
    pub swept_buffers: AtomicU64,
    /// Buffers visited by sweeps, whether destroyed or not.
    pub sweep_visits: AtomicU64,

    /// Fresh `MTLBuffer` allocations, i.e. lookups that missed.
    pub allocations: AtomicU64,
    /// Bytes handed to `newBufferWithLength`.
    pub allocated_bytes: AtomicU64,
}

static STATS: OnceLock<PoolStats> = OnceLock::new();

pub fn stats() -> &'static PoolStats {
    STATS.get_or_init(|| {
        let on = std::env::var("CANDLE_METAL_POOL_STATS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        PoolStats {
            enabled: AtomicBool::new(on),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            buckets_walked: AtomicU64::new(0),
            buffers_examined: AtomicU64::new(0),
            max_buckets_walked: AtomicU64::new(0),
            max_buffers_examined: AtomicU64::new(0),
            sweeps: AtomicU64::new(0),
            swept_buffers: AtomicU64::new(0),
            sweep_visits: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
        }
    })
}

impl PoolStats {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(ORD)
    }

    /// Records one `find_available_buffer` call.
    #[inline]
    pub fn record_lookup(&self, buckets: u64, examined: u64, hit: bool) {
        if !self.enabled() {
            return;
        }
        self.lookups.fetch_add(1, ORD);
        if hit {
            self.hits.fetch_add(1, ORD);
        }
        self.buckets_walked.fetch_add(buckets, ORD);
        self.buffers_examined.fetch_add(examined, ORD);
        self.max_buckets_walked.fetch_max(buckets, ORD);
        self.max_buffers_examined.fetch_max(examined, ORD);
    }

    #[inline]
    pub fn record_sweep(&self, visits: u64, freed: u64) {
        if !self.enabled() {
            return;
        }
        self.sweeps.fetch_add(1, ORD);
        self.sweep_visits.fetch_add(visits, ORD);
        self.swept_buffers.fetch_add(freed, ORD);
    }

    #[inline]
    pub fn record_allocation(&self, bytes: u64) {
        if !self.enabled() {
            return;
        }
        self.allocations.fetch_add(1, ORD);
        self.allocated_bytes.fetch_add(bytes, ORD);
    }

    /// Resets every counter. Called between phases so prefill and decode can be
    /// reported separately -- they have very different allocation behaviour and
    /// averaging them together is how the 27 ms/token figure went wrong (§6.6).
    pub fn reset(&self) {
        for c in [
            &self.lookups,
            &self.hits,
            &self.buckets_walked,
            &self.buffers_examined,
            &self.max_buckets_walked,
            &self.max_buffers_examined,
            &self.sweeps,
            &self.swept_buffers,
            &self.sweep_visits,
            &self.allocations,
            &self.allocated_bytes,
        ] {
            c.store(0, ORD);
        }
    }

    pub fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            lookups: self.lookups.load(ORD),
            hits: self.hits.load(ORD),
            buckets_walked: self.buckets_walked.load(ORD),
            buffers_examined: self.buffers_examined.load(ORD),
            max_buckets_walked: self.max_buckets_walked.load(ORD),
            max_buffers_examined: self.max_buffers_examined.load(ORD),
            sweeps: self.sweeps.load(ORD),
            swept_buffers: self.swept_buffers.load(ORD),
            sweep_visits: self.sweep_visits.load(ORD),
            allocations: self.allocations.load(ORD),
            allocated_bytes: self.allocated_bytes.load(ORD),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStatsSnapshot {
    pub lookups: u64,
    pub hits: u64,
    pub buckets_walked: u64,
    pub buffers_examined: u64,
    pub max_buckets_walked: u64,
    pub max_buffers_examined: u64,
    pub sweeps: u64,
    pub swept_buffers: u64,
    pub sweep_visits: u64,
    pub allocations: u64,
    pub allocated_bytes: u64,
}

impl PoolStatsSnapshot {
    pub fn buckets_per_lookup(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.buckets_walked as f64 / self.lookups as f64
        }
    }

    pub fn buffers_per_lookup(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.buffers_examined as f64 / self.lookups as f64
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }

    /// One line per field, `key=value`, so a run can be diffed against another
    /// without a parser.
    pub fn report(&self, label: &str) -> String {
        format!(
            "pool_stats label={label} lookups={} hits={} hit_rate={:.4} \
             buckets_walked={} buckets_per_lookup={:.2} max_buckets_walked={} \
             buffers_examined={} buffers_per_lookup={:.2} max_buffers_examined={} \
             sweeps={} sweep_visits={} swept_buffers={} \
             allocations={} allocated_mb={:.1}",
            self.lookups,
            self.hits,
            self.hit_rate(),
            self.buckets_walked,
            self.buckets_per_lookup(),
            self.max_buckets_walked,
            self.buffers_examined,
            self.buffers_per_lookup(),
            self.max_buffers_examined,
            self.sweeps,
            self.sweep_visits,
            self.swept_buffers,
            self.allocations,
            self.allocated_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Live pool occupancy, sampled rather than accumulated.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolOccupancy {
    pub shared_buffers: usize,
    pub shared_buckets: usize,
    pub shared_empty_buckets: usize,
    pub shared_bytes: usize,
    pub private_buffers: usize,
    pub private_buckets: usize,
    pub private_empty_buckets: usize,
    pub private_bytes: usize,
}

impl PoolOccupancy {
    pub fn total_bytes(&self) -> usize {
        self.shared_bytes + self.private_bytes
    }

    pub fn total_buffers(&self) -> usize {
        self.shared_buffers + self.private_buffers
    }

    pub fn total_empty_buckets(&self) -> usize {
        self.shared_empty_buckets + self.private_empty_buckets
    }

    pub fn report(&self, label: &str) -> String {
        format!(
            "pool_occupancy label={label} \
             shared_buffers={} shared_buckets={} shared_empty_buckets={} shared_mb={:.1} \
             private_buffers={} private_buckets={} private_empty_buckets={} private_mb={:.1} \
             total_buffers={} total_buckets={} total_empty_buckets={} total_mb={:.1}",
            self.shared_buffers,
            self.shared_buckets,
            self.shared_empty_buckets,
            self.shared_bytes as f64 / (1024.0 * 1024.0),
            self.private_buffers,
            self.private_buckets,
            self.private_empty_buckets,
            self.private_bytes as f64 / (1024.0 * 1024.0),
            self.total_buffers(),
            self.shared_buckets + self.private_buckets,
            self.total_empty_buckets(),
            self.total_bytes() as f64 / (1024.0 * 1024.0),
        )
    }
}
