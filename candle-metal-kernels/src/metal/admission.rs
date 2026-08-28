//! The memory budget: predict the peak before allocating, and refuse.
//!
//! `DESIGN.md` §9.5, issue #186. This is §9.5d's admission check and §9.5k's
//! derived residual, and it is the whole of what those sections specify.
//!
//! # Why this exists, in one paragraph
//!
//! **Nothing in this engine bounded the sum of §9.1's five memory classes, and
//! the failure mode was a kernel panic — twice.** §6.3b's `set_free_budget`
//! bounds one class's *free list*; §6.2b's `DEFAULT_KV_CAPACITY` bounds one
//! class's *per-sequence reserve*. There was no mechanism above either, and
//! none anywhere that knew what the five classes sum to. The second panic
//! (§9.5a) died **during allocation** — 0.60 s user against 34.77 s system,
//! 692 467 page faults against **144 page-ins**, which is a process faulting in
//! fresh anonymous pages rather than reading a model from disk — with 60 MB
//! free and 54.91 GB wired, and with #166/#167's residency guard compiled in.
//! **That is the failure admission prevents**, and it is a different failure
//! from the one the guard closed.
//!
//! # The organizing idea: the unpredictable class is the residual
//!
//! Four terms are exactly known before anything is allocated — weights from the
//! checkpoint header (§5.5), KV from `B × capacity × 16 KiB` (§5.6's geometry,
//! *not* its table — see [`KV_BYTES_PER_TOKEN`]), conv state from §5.7, and
//! scratch from §9.1a's formula under the selected policy. What is left over is
//! a **derived** cap on everything else rather than an estimate of it:
//!
//! ```text
//! residual = budget − (weights + KV + conv + activations + scratch)
//! ```
//!
//! So the one class that grows on demand acquires a ceiling **by construction**,
//! needing no new measurement. That is what lets admission report *"this
//! configuration leaves N GB for intermediates"* instead of discovering the
//! answer by dying.
//!
//! # What this module does not close
//!
//! **§9.5f, stated here because it would be easy to over-read this code.** Every
//! reachable B=1 configuration predicts under 8.71 GB and the machine died with
//! 54.91 GB wired, so **admission as specified would have refused nothing** on
//! the run that crashed. It is necessary and it is not sufficient. §16 P0 #7 is
//! the open question, and the measurement it wants is #171's memory-by-class
//! timeline at long context — which has never been run.
//!
//! What the residual adds is a *boundary* for the second of §9.5f's three
//! candidate explanations: an allocation path growing without bound is still
//! invisible to admission, but crossing the derived ceiling is now a refusal at
//! a known site rather than a panic at an arbitrary one.

use super::buffer_pool::PoolOccupancySnapshot;
use super::scratch::{PartialsGeometry, Sizing};

/// KV bytes per token across all attention layers, from the geometry.
///
/// `2 (K,V) × 8 kv_heads × 64 head_dim × 2 B × 8 attention layers` = **16 384**.
///
/// **This is §5.6's *geometry*, not §5.6's table.** That table was computed with
/// **16 000** B/token and is therefore low by 2.4 % — `0.524288 =
/// 16000 × 32768 / 1e9` exactly, which is what identifies the arithmetic rather
/// than leaving it as a rounding question (§9.5b; the table is corrected in
/// place). The error is 50 MB at B=1 × 128k and **1.61 GB at B=32 × 128k**:
/// negligible where the table is usually quoted, and not negligible where a
/// memory budget uses it, which is how it was found. **A budget built by quoting
/// the table would under-predict in exactly the regime where the margin
/// matters**, so this constant is derived from primitives below and asserted
/// against them by `kv_geometry_is_16384_not_16000`.
pub const KV_BYTES_PER_TOKEN: usize = 2 * 8 * 64 * 2 * 8;

/// Conv state across all 22 conv layers, per sequence: `22 × 2048 × 3 × 2 B`.
///
/// §5.7, and §10.2g records that `ConvState::RotatingRing` is also this figure
/// where the sliding ring at `slack = 16` is 1.63 MiB. The larger arm is a
/// caller-supplied override rather than the default, for the reason [`Budget`]
/// gives: this crate cannot see `lfm2::Config`.
pub const CONV_STATE_BYTES: usize = 22 * 2048 * 3 * 2;

/// The activation arena at B=1, **measured** by #68 (§9.2c).
///
/// Not an estimate: 68.00 KB (69 632 B) packed, against a maximum simultaneous
/// liveness of 67.00 KB. §5.9's "tens of KB" was right and nobody had turned it
/// into a verdict.
///
/// It is also the one term that is genuinely *outside* the pools —
/// `install_arena` calls the raw device (`device.rs:485`) where weights, KV and
/// scratch all go through `acquire` (§9.5k). That asymmetry is what
/// [`unplanned_bytes`] turns on.
pub const ACTIVATION_ARENA_BYTES: usize = 69_632;

/// The default fraction of `recommendedMaxWorkingSetSize` a process may claim.
///
/// **The denominator is `recommendedMaxWorkingSetSize` and not `hw.memsize`**,
/// per §9.5c, and the choice is argued rather than defaulted. Three reasons:
/// it is the only candidate that is *about* GPU-visible memory, which is what
/// §6.3c's three-participant lifetime finding makes the relevant quantity; it
/// is **already read** (`device.rs:225`), so this needs no new capability; and
/// it degrades honestly on a machine that is not this one, where a hardcoded
/// byte count does not.
///
/// The two rejected candidates were rejected **by measurement, not by
/// preference**: free-page sampling reads a quantity every other process is
/// moving (§3.4a-ii measured `python3.13` moving **13.0 GB during a 5.4 s
/// run**), and `memoryPressure` read **False at both panics** (§9.5a) — the one
/// boolean an engine might have consulted was false when the machine died,
/// twice.
///
/// 0.65 is the figure §9.5k's table is computed at. It is a **default and not a
/// finding**: no measurement chose it, and it is a caller-settable field on
/// [`Budget`] for that reason.
pub const DEFAULT_BUDGET_FRACTION: f64 = 0.65;

/// §6.3b's measured stranding: 5231 → 13629 MB accumulated over 400 tokens.
///
/// **The yardstick a residual is graded against, and it is measured rather than
/// round.** §9.5k: a residual below this is one that a *known and already
/// observed* allocation shape can exhaust, which is what deriving the cap buys
/// — "is this configuration safe" becomes a comparison against something
/// observed rather than against a number someone picked.
///
/// It is the largest unbounded term this project has seen, and it is
/// `pending_bytes`: 11.6 buffers per token, ~21 MB/token, capped by nothing.
/// `free_bytes` is capped by `set_free_budget`; `pending_bytes` is not.
pub const OBSERVED_STRANDING_BYTES: usize = (13_629 - 5_231) * 1_000_000;

/// What one configuration is predicted to need, class by class.
///
/// Every field is bytes, and every field is computable **before the first
/// allocation** — which is the property that makes admission free (§9.5d). This
/// struct allocates nothing, reads no device state, and is a pure function of
/// the configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Footprint {
    /// §5.5, from the checkpoint header. A **fact**, with one policy lever:
    /// dropping the vision tower for a text-only request (#162).
    pub weights: usize,
    /// `B × capacity × 16 384 B`. A **policy** — the capacity is chosen — and
    /// the class that can consume the machine on its own: `B=32` at 128k is
    /// **68.719 GB of KV alone, exactly `hw.memsize`**, before a weight is
    /// loaded (§9.5b).
    pub kv: usize,
    /// `B × 264 KiB` (§5.7). A **fact** given `ConvState`.
    pub conv: usize,
    /// `B × 68 KB` (§9.2c). A **fact**, and negligible.
    pub activations: usize,
    /// §9.1a's formula under the selected policy. A **policy** —
    /// `Reserve`/`Grow`/`Bucket` — and **not** where the budget binds at any
    /// reachable configuration: 0.4 % of the total at B=1 × 128k against KV's
    /// 25 % (§9.5h).
    pub scratch: usize,
}

impl Footprint {
    /// The predicted peak: the five classes summed.
    ///
    /// **There is deliberately no `pool` row here**, and §9.5k is why. The pool
    /// is not a sixth class sitting beside the others at 0.27 GB — that figure
    /// is `set_free_budget`'s *free-list* cap, and the pool is where three of
    /// the five classes above are actually **served from**. Adding it as a row
    /// would double-count exactly what §9.5k had to read the allocation paths
    /// to rule out. What the caller needs instead is [`Admission::residual`].
    pub fn predicted(&self) -> usize {
        self.weights + self.kv + self.conv + self.activations + self.scratch
    }
}

/// The configuration admission is asked about.
///
/// # Why this type does not know about LFM2
///
/// It takes `weights` as a byte count rather than reading a checkpoint, and
/// `conv_state_bytes` as a number rather than a `ConvState`. `candle-metal-
/// kernels` is below `candle-transformers` and cannot see `lfm2::Config`;
/// putting model geometry here would be the conflation §10.1 renames the whole
/// `SequenceState` abstraction to avoid, one crate down. **Policy on the CPU,
/// numbers on the GPU** (§15.2 #8) applied to the crate boundary: the caller
/// resolves the policy and passes numbers.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Concurrent sequences. Every measurement in this project is B=1 (§13.2).
    pub batch: usize,
    /// KV tokens reserved per sequence — `DEFAULT_KV_CAPACITY`'s value, not
    /// `max_context`, because the reserve is what is *allocated* (§6.2b).
    pub kv_capacity: usize,
    /// The context the scratch class is sized for under `Reserve`.
    pub max_context: usize,
    /// Total weight bytes, from the checkpoint header (§5.5). Text-only drops
    /// the 0.86 GB vision tower and projector (#162).
    pub weights: usize,
    /// Per-sequence conv state. [`CONV_STATE_BYTES`] for `Shuffle` and
    /// `RotatingRing`; larger for `SlidingRing` at slack (§10.2g).
    pub conv_state_bytes: usize,
    /// §9.1a's sizing policy for the scratch class.
    pub scratch_sizing: Sizing,
    /// The partials geometry scratch is sized from. `n_heads` is the **query**
    /// head count — using the 8 KV heads under-sizes the class **4×** (§9.1a).
    pub partials: PartialsGeometry,
    /// Fraction of `recommendedMaxWorkingSetSize` this process may claim.
    pub fraction: f64,
}

impl Budget {
    /// A B=1 budget with the defaults every field's doc comment cites.
    pub fn new(weights: usize, kv_capacity: usize, max_context: usize) -> Self {
        Self {
            batch: 1,
            kv_capacity,
            max_context,
            weights,
            conv_state_bytes: CONV_STATE_BYTES,
            scratch_sizing: Sizing::Reserve,
            partials: PartialsGeometry::default(),
            fraction: DEFAULT_BUDGET_FRACTION,
        }
    }

    /// The five classes, each from the section that states it.
    ///
    /// Pure arithmetic: six multiplies and some adds, no device call and no
    /// allocation. That is what makes admission **free by inspection** rather
    /// than by measurement (§9.5e) — it is not on any per-token path.
    pub fn footprint(&self) -> Footprint {
        let mut partials = self.partials;
        partials.batch = self.batch;
        // §9.1a's table is per-layer-summed: the class is 8 attention layers'
        // regions. `Reserve` sizes for max_context's chunk count and allocates
        // once; `Grow` and `Bucket` size to the live kv_len, which admission
        // cannot know — so both are predicted at their worst case, which is
        // `Reserve`'s figure. Predicting `Grow` at a *smaller* number would be
        // admitting a configuration on a footprint it only has early on.
        let chunks = partials.chunks(self.max_context);
        let scratch = N_ATTENTION_LAYERS * partials.partials_bytes(chunks);
        Footprint {
            weights: self.weights,
            kv: self.batch * self.kv_capacity * KV_BYTES_PER_TOKEN,
            conv: self.batch * self.conv_state_bytes,
            activations: self.batch * ACTIVATION_ARENA_BYTES,
            scratch,
        }
    }

    /// Evaluate this configuration against `working_set` bytes.
    ///
    /// `working_set` is `MTLDevice::recommendedMaxWorkingSetSize` (§9.5c),
    /// passed in rather than read here so the arithmetic stays a pure function
    /// and is testable without a GPU.
    pub fn admit(&self, working_set: usize) -> Admission {
        let footprint = self.footprint();
        let budget = (working_set as f64 * self.fraction) as usize;
        let predicted = footprint.predicted();
        Admission {
            footprint,
            budget,
            working_set,
            fraction: self.fraction,
            // Saturating rather than wrapping: a refused configuration has a
            // negative residual conceptually, and `residual_signed` reports it.
            residual: budget.saturating_sub(predicted),
            fits: predicted <= budget,
        }
    }
}

/// Attention layers holding KV. 8 of 30 for LFM2 (§5.3, from `layer_types`).
const N_ATTENTION_LAYERS: usize = 8;

/// The verdict, with the arithmetic that produced it.
///
/// **A boolean would not be enough**, per §9.5g: the caller sees which class
/// dominates, so the next configuration is a decision rather than a guess.
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub footprint: Footprint,
    /// `fraction × working_set`.
    pub budget: usize,
    /// `recommendedMaxWorkingSetSize`, as read from the device.
    pub working_set: usize,
    pub fraction: f64,
    /// `budget − predicted`, saturating at zero. **The derived cap on
    /// everything the five classes do not name** (§9.5k) — the pool, every
    /// intermediate no planner owns, and the RoPE tables (see
    /// [`Admission::describe`]).
    pub residual: usize,
    pub fits: bool,
}

impl Admission {
    /// `budget − predicted` as a signed quantity, so a refusal can report how
    /// far over it is rather than reporting zero.
    pub fn residual_signed(&self) -> i128 {
        self.budget as i128 - self.footprint.predicted() as i128
    }

    /// Whether the residual is smaller than a measured allocation shape can
    /// consume.
    ///
    /// **Graded against §6.3b's stranding rather than a round number**
    /// ([`OBSERVED_STRANDING_BYTES`]), which is the point of deriving the cap.
    /// `B=16 ctx 128k` passes the sum-based table at **60.3 % of the machine**
    /// and fails *this*, because 3.495 GB of residual is less than the 8.398 GB
    /// a known shape has already been measured accumulating over 400 tokens
    /// (§9.5k). A refusal that did not reflect that would admit a configuration
    /// a known allocation shape can kill.
    pub fn is_tight(&self) -> bool {
        self.fits && self.residual < OBSERVED_STRANDING_BYTES
    }

    /// The report §9.5g specifies, in the shape it specifies.
    ///
    /// Refusals carry the per-class breakdown and the reductions that would fit;
    /// admissions carry the residual, **which is the more useful half because it
    /// is the case that happens**.
    pub fn describe(&self) -> String {
        let gb = |b: usize| b as f64 / 1e9;
        let f = &self.footprint;
        if self.fits {
            let mut s = format!(
                "memory budget OK: predicted peak {:.2} GB, budget {:.2} GB\n  \
                 residual {:.2} GB   for pool, intermediates, and anything unplanned\n\
                 {:>24}(§6.3b's stranding is {:.2} GB over 400 tokens)",
                gb(f.predicted()),
                gb(self.budget),
                gb(self.residual),
                "",
                gb(OBSERVED_STRANDING_BYTES),
            );
            if self.is_tight() {
                s.push_str(
                    "\n  TIGHT: the residual is below a stranding figure this project \
                     has measured.",
                );
            }
            return s;
        }
        // A refusal. §9.5g: report the arithmetic, name the dominant class, and
        // list the reductions in order of what they cost -- so the next
        // configuration is a decision rather than a guess.
        let mut s = format!(
            "memory budget exceeded: predicted peak {:.2} GB > budget {:.2} GB\n  \
             ({:.2} x recommendedMaxWorkingSetSize = {:.2} GB)\n",
            gb(f.predicted()),
            gb(self.budget),
            self.fraction,
            gb(self.working_set),
        );
        let classes = [
            ("weights", f.weights),
            ("KV", f.kv),
            ("scratch", f.scratch),
            ("conv", f.conv),
            ("act", f.activations),
        ];
        let dominant = classes
            .iter()
            .max_by_key(|(_, b)| *b)
            .map(|(n, _)| *n)
            .unwrap_or("");
        for (name, bytes) in classes {
            let mark = if name == dominant {
                "   <- dominant"
            } else {
                ""
            };
            s.push_str(&format!("    {name:<9} {:>7.2} GB{mark}\n", gb(bytes)));
        }
        s.push_str(&format!(
            "  residual {:>7.2} GB   budget - predicted   <- NEGATIVE: nothing left \
             to allocate into\n",
            self.residual_signed() as f64 / 1e9,
        ));
        s.push_str("  reductions that would fit, in order of what they cost:\n");
        // Refuse rather than silently reduce (§9.5g): these are reported for the
        // caller to choose between, never applied. A silent reduction is a
        // configuration the caller did not choose and the artifact does not
        // record, which is §2.4's "an instrument that cannot be shown to have
        // engaged" in a new quantity -- and *which* lever to give up is a policy
        // the engine does not have, since a summarization job wants context
        // where a serving job wants concurrency.
        s.push_str(&format!(
            "    halve max_context             frees {:>6.2} GB\n",
            gb(f.kv / 2 + f.scratch / 2),
        ));
        s.push_str(&format!(
            "    halve B                       frees {:>6.2} GB\n",
            gb((f.kv + f.conv + f.activations + f.scratch) / 2),
        ));
        s.push_str("    scratch Reserve -> Grow       (see §9.1a)\n");
        s.push_str("    text-only (no vision tower)   frees   0.86 GB   (#162)\n");
        s
    }
}

/// Bytes the pools hold that **no planner owns**.
///
/// # This is the one derived quantity, and why it is not `live_bytes`
///
/// §9.5k had to read the allocation paths rather than §9.1's class table to get
/// this right, and the correction is what makes the check safe:
///
/// | term | path | lands in |
/// |---|---|---|
/// | weights | `to_dtype` → `new_buffer_builder` → `new_buffer` | **`private_buffers`** |
/// | KV | `KvSlot::append` → `Tensor::zeros` → `allocate_buffer` | **`buffers`** |
/// | scratch | `ScratchPlan`, when wired | pool |
/// | **arena** | `install_arena` → the **raw** device | **outside the pools** |
///
/// So **`live_bytes` already contains the weights and the KV reserve**, and
/// comparing `live_bytes` against the residual would **refuse the weights
/// themselves**. Corroborated from a direction nobody took for this purpose:
/// §6.3a measures the pool at 5220 MB and §6.3b at 5459 MB flat at 400 tokens,
/// against 5394 MB of weights — the pool footprint *is* approximately the
/// weights, not a 256 MiB region beside them.
///
/// `free_bytes` and `pending_bytes` belong on the left because both are bytes
/// the process holds and the OS cannot have. **`pending_bytes` is the term that
/// matters**: `free_bytes` is capped by `set_free_budget`, and `pending_bytes`
/// is capped by **nothing** — it is §6.3b's stranding at ~21 MB/token, 8.398 GB
/// over 400 tokens, the largest unbounded term this project has observed. The
/// residual is what finally gives it a ceiling.
///
/// The subtraction saturates: early in a run the pools hold less than the
/// planned set (weights are still loading), and a wrapping subtraction there
/// would report a colossal `unplanned` and refuse a run that is fine.
pub fn unplanned_bytes(occupancy: &PoolOccupancySnapshot, planned_in_pool: usize) -> usize {
    let held = occupancy.live_bytes + occupancy.free_bytes + occupancy.pending_bytes;
    held.saturating_sub(planned_in_pool)
}

/// The classes that are served **by the pool**, and therefore already inside
/// `live_bytes`.
///
/// The arena is excluded because `install_arena` calls the raw device — it is
/// the one class of the five that is genuinely outside the pools (§9.5k).
pub fn planned_in_pool(footprint: &Footprint) -> usize {
    footprint.weights + footprint.kv + footprint.scratch + footprint.conv
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1e9;
    /// `hw.memsize` on the development machine (§3.4). Used as a stand-in for
    /// `recommendedMaxWorkingSetSize`, which is read from the device at runtime
    /// and cannot be assumed in a unit test.
    const MACHINE: usize = 68_719_476_736;
    /// §5.5: language model 5.394 GB + vision 0.83 + projector 0.03.
    const WEIGHTS_FULL: usize = 6_254_000_000;

    fn b1(capacity: usize, max_context: usize) -> Budget {
        Budget::new(WEIGHTS_FULL, capacity, max_context)
    }

    /// §9.5b's correction, asserted so it cannot drift back.
    ///
    /// §5.6's table was computed with 16 000 B/token where the geometry on its
    /// own preceding line derives 16 384. A budget quoting the table would
    /// under-predict by 2.4 % — 1.61 GB at B=32 × 128k — in exactly the regime
    /// where the margin matters.
    #[test]
    fn kv_geometry_is_16384_not_16000() {
        assert_eq!(
            KV_BYTES_PER_TOKEN, 16_384,
            "2 x 8 heads x 64 dim x 2 B x 8 layers"
        );
        assert_ne!(
            KV_BYTES_PER_TOKEN, 16_000,
            "§5.6's table, which is low by 2.4 %"
        );
        // The understatement at the configuration where it stops being
        // negligible, from §9.5b.
        let understated = 32 * (KV_BYTES_PER_TOKEN - 16_000) * 131_072;
        assert!(
            (understated as f64 / GB - 1.61).abs() < 0.01,
            "§9.5b: 1.61 GB at B=32 x 128k, got {:.3}",
            understated as f64 / GB
        );
    }

    /// The class table reconciles with §9.5b's, row for row.
    ///
    /// Not a restatement: this is the arithmetic §9.5b's table was drawn from,
    /// so a divergence here means one of the two moved. The tolerance is 10 MB,
    /// tight enough that a class-sized error cannot hide in it.
    #[test]
    fn footprint_reconciles_with_design_9_5b() {
        // (label, B, capacity, max_context, expected TOTAL in GB from §9.5b,
        //  minus its `pool` column -- §9.5k removed that row, so the expected
        //  figure here is the table's total less its 0.268 GB pool entry.)
        let cases = [
            ("B=1 ctx 4k", 1, 4096, 4096, 6.591 - 0.268),
            ("B=1 ctx 32k", 1, 32768, 32768, 7.068 - 0.268),
            ("B=1 ctx 128k", 1, 131072, 131072, 8.705 - 0.268),
            ("B=16 ctx 128k", 16, 131072, 131072, 41.441 - 0.268),
        ];
        for (label, batch, cap, mc, expect_gb) in cases {
            let mut b = b1(cap, mc);
            b.batch = batch;
            let got = b.footprint().predicted() as f64 / GB;
            assert!(
                (got - expect_gb).abs() < 0.01,
                "{label}: §9.5b says {expect_gb:.3} GB, got {got:.3} GB"
            );
        }
    }

    /// **Both bounds, which is the whole point** (§8.1g, and #184's precedent).
    ///
    /// An admission check that refuses everything is not a check. The two arms
    /// are asserted together in one test so neither can be dropped without the
    /// other going red.
    #[test]
    fn admits_a_configuration_that_fits_and_refuses_one_that_does_not() {
        // The configuration that ships today (§9.5b: 9.6 % of the machine).
        let ok = b1(4096, 4096).admit(MACHINE);
        assert!(ok.fits, "B=1 ctx 4k must be ADMITTED:\n{}", ok.describe());
        assert!(
            ok.residual > OBSERVED_STRANDING_BYTES,
            "and it must be ample, not merely admitted: residual {:.2} GB",
            ok.residual as f64 / GB
        );

        // B=32 at 128k: §9.5b's `KV alone = 68.719 GB, exactly hw.memsize`,
        // before a weight is loaded.
        let mut over = b1(131_072, 131_072);
        over.batch = 32;
        let refused = over.admit(MACHINE);
        assert!(!refused.fits, "B=32 ctx 128k must be REFUSED");
        assert!(
            refused.residual_signed() < 0,
            "a refusal reports a negative residual, got {}",
            refused.residual_signed()
        );
        // §9.5b: KV x B is the only class that can eat the machine on its own.
        assert!(
            refused.footprint.kv as f64 / GB > 68.0,
            "KV alone at B=32 x 128k is 68.719 GB (§9.5b), got {:.3}",
            refused.footprint.kv as f64 / GB
        );
    }

    /// **`B=16 ctx 128k` passes the sum-based table and fails the residual
    /// test** — §9.5k, and the trap the issue names explicitly.
    ///
    /// It sums to 60.3 % of the machine, which looks comfortable, and its
    /// residual is 3.495 GB against §6.3b's **measured** 8.398 GB of stranding
    /// over 400 tokens. Grading against a measured quantity rather than a round
    /// number is what makes the difference, and a refusal that did not reflect
    /// it would admit a configuration a known allocation shape can kill.
    #[test]
    fn b16_ctx128k_fits_the_sum_and_is_tight_against_measured_stranding() {
        let mut b = b1(131_072, 131_072);
        b.batch = 16;
        let a = b.admit(MACHINE);

        // It fits the sum: §9.5b puts it at 60.3 % of the machine.
        assert!(a.fits, "it passes the sum-based table:\n{}", a.describe());
        let share = a.footprint.predicted() as f64 / MACHINE as f64;
        assert!(
            (share - 0.603).abs() < 0.01,
            "§9.5b: 60.3 % of the machine, got {:.1} %",
            share * 100.0
        );

        // And it is tight: §9.5k's residual at fraction 0.65 is 3.495 GB.
        assert!(
            a.is_tight(),
            "the residual must be graded against §6.3b's measured stranding, \
             not a round number:\n{}",
            a.describe()
        );
        assert!(
            (a.residual as f64 / GB - 3.495).abs() < 0.02,
            "§9.5k: residual 3.495 GB at fraction 0.65, got {:.3} GB",
            a.residual as f64 / GB
        );
        assert!(
            a.residual < OBSERVED_STRANDING_BYTES,
            "3.495 GB residual is less than 8.398 GB of measured stranding"
        );
    }

    /// §9.5k's residual table, reproduced row for row.
    #[test]
    fn residual_reconciles_with_design_9_5k() {
        let cases = [
            ("B=1 ctx 4k", 1, 4096, 4096, 38.345),
            ("B=1 ctx 128k", 1, 131072, 131072, 36.231),
            ("B=16 ctx 32k", 16, 32768, 32768, 29.680),
            ("B=16 ctx 128k", 16, 131072, 131072, 3.495),
        ];
        for (label, batch, cap, mc, expect_gb) in cases {
            let mut b = b1(cap, mc);
            b.batch = batch;
            let got = b.admit(MACHINE).residual as f64 / GB;
            assert!(
                (got - expect_gb).abs() < 0.02,
                "{label}: §9.5k says residual {expect_gb:.3} GB, got {got:.3} GB"
            );
        }
    }

    /// **Comparing `live_bytes` against the residual would refuse the weights
    /// themselves** — §9.5k's load-bearing correction, asserted.
    ///
    /// The mutation this guards against is the *obvious* form of the check, and
    /// it is the one the issue calls out: weights land in `private_buffers` and
    /// KV in `buffers`, so both are already inside `live_bytes`.
    #[test]
    fn unplanned_subtracts_the_classes_the_pool_already_holds() {
        let f = b1(4096, 4096).footprint();
        let planned = planned_in_pool(&f);

        // A pool holding exactly the planned set and nothing else.
        let occ = PoolOccupancySnapshot {
            live_bytes: planned,
            free_bytes: 0,
            pending_bytes: 0,
            ..Default::default()
        };
        assert_eq!(
            unplanned_bytes(&occ, planned),
            0,
            "a pool holding only the planned classes has ZERO unplanned bytes; \
             comparing live_bytes against the residual would refuse the weights"
        );

        // The same pool, plus a token's worth of §6.3b stranding.
        let strand = 21 * 1_000_000;
        let occ = PoolOccupancySnapshot {
            live_bytes: planned,
            pending_bytes: strand,
            ..Default::default()
        };
        assert_eq!(
            unplanned_bytes(&occ, planned),
            strand,
            "pending_bytes is the unbounded term (§6.3b: ~21 MB/token) and must \
             be counted"
        );
    }

    /// `free_bytes` and `pending_bytes` are on the left of the subtraction.
    ///
    /// Both are bytes the process holds and the OS cannot have (§9.5k). A
    /// version counting only `live_bytes` would miss the one term that is
    /// capped by nothing.
    #[test]
    fn unplanned_counts_free_and_pending_not_only_live() {
        let occ = PoolOccupancySnapshot {
            live_bytes: 100,
            free_bytes: 20,
            pending_bytes: 3,
            ..Default::default()
        };
        assert_eq!(unplanned_bytes(&occ, 100), 23);
        // And it saturates rather than wrapping: early in a run the pool holds
        // less than the planned set, and a wrapping subtraction would report a
        // colossal figure and refuse a run that is fine.
        assert_eq!(unplanned_bytes(&occ, 10_000), 0);
    }

    /// The scratch class is sized from **query** heads, not KV heads.
    ///
    /// §9.1a: using 8 under-sizes the class **4×** — 1.03 MB at 128k where the
    /// truth is 4.12 — and §3.5 says nothing would report the overrun.
    #[test]
    fn scratch_uses_query_heads_and_reconciles_with_9_1a() {
        let f = b1(131_072, 131_072).footprint();
        // §9.1a: Reserve at 8 layers, max_context 128k, is 33.0 MiB.
        let mib = f.scratch as f64 / (1024.0 * 1024.0);
        assert!(
            (mib - 33.0).abs() < 0.1,
            "§9.1a: 33.0 MiB at 8 layers, got {mib:.2} MiB"
        );
        // The 4x error the KV-head reading would produce.
        let with_kv_heads = mib / 4.0;
        assert!(
            (with_kv_heads - 8.25).abs() < 0.1,
            "using 8 KV heads would give {with_kv_heads:.2} MiB -- the 4x \
             under-size §9.1a warns about"
        );
    }

    /// A refusal names the dominant class and does not silently reduce.
    ///
    /// §9.5g: refuse rather than reduce, because *which* lever to give up is a
    /// policy the engine does not have.
    #[test]
    fn refusal_reports_the_arithmetic_and_names_the_dominant_class() {
        let mut b = b1(131_072, 131_072);
        b.batch = 32;
        let a = b.admit(MACHINE);
        let msg = a.describe();
        assert!(msg.contains("memory budget exceeded"), "{msg}");
        assert!(
            msg.contains("KV"),
            "the dominant class must be named: {msg}"
        );
        assert!(msg.contains("<- dominant"), "{msg}");
        assert!(
            msg.contains("reductions that would fit"),
            "the caller needs the levers to choose between: {msg}"
        );
        // And the admitted case reports the residual, which §9.5g calls the
        // more useful half because it is the case that happens.
        let ok = b1(4096, 4096).admit(MACHINE).describe();
        assert!(ok.contains("memory budget OK"), "{ok}");
        assert!(ok.contains("residual"), "{ok}");
    }
}
