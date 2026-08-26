//! Record one decode step, encode the replayable subset into an
//! `MTLIndirectCommandBuffer`, and execute it by range (`DESIGN.md` §11.3).
//!
//! This is `DESIGN.md` §17 Phase 2 item 10, and it is deliberately **partial**.
//! §9.2i measures 123 of the 554 decode dispatch positions as varying across
//! steady-state steps under `Sdpa × arena`, and 48 of those need
//! `dispatchThreadgroupsWithIndirectBuffer` whatever the allocator does. Those
//! are not covered here and are not worked around: a second mechanism in the
//! same change would make a bisect impossible, which is the rule #64 split #82
//! out under and #82 declined again.
//!
//! # What "replayable" means here, and the axis that is easy to miss
//!
//! §9.2h and §9.2i classify the stream by what *varies* across steps -- buffer
//! identity, grid, binding offset. That is necessary and it is not sufficient.
//! An ICB command has no `setBytes` in any form (§3.7c), so a position whose
//! kernel binds its constants inline is unencodable **however stable it is**.
//! Coverage is therefore the intersection of two independent axes:
//!
//! * stable across steady-state steps (identity, grid, offset), and
//! * dispatched through a `_packed` entry point, so the constants live in a
//!   buffer a `setKernelBuffer` can bind.
//!
//! The second axis is what `set_default_param_style` exists for. Before it,
//! every decode dispatch bound its scalars inline and the covered set was
//! **empty** -- the packed variants from #38 through #81 were compiled and never
//! dispatched. Reporting only the first axis is what makes "123 positions stand
//! between here and replay" read as though closing that gap were enough.
//!
//! # Why record-then-replay, and what happens when the recording goes stale
//!
//! §11.1a gives the two options: record-and-validate, or have the model declare
//! a stable region. Candle is eager and declares nothing, so this records. The
//! validation is not a checksum over the recording -- it is that every replayed
//! position was *observed* to be invariant over the steps that were recorded,
//! and that the live step's own dispatches still match. A position that stops
//! matching falls back to the classical path for that step rather than replaying
//! a stale command, because under `HazardTrackingModeUntracked` a wrong buffer
//! is silent corruption rather than an error (§3.5).
use crate::metal::{Buffer, ComputePipeline};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResourceOptions, MTLSize,
};
use std::sync::{Arc, Mutex};

use super::executor::{DispatchAction, DispatchRecord, Executor, Grid, IcbRange, Size3};

/// One buffer binding, as an ICB command needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingRecord {
    pub index: usize,
    /// Kept as the pointer rather than the `Buffer`, so comparing two steps'
    /// recordings compares identity rather than a handle that may have been
    /// reissued. The `Buffer` itself is held separately, in `retained`.
    pub ptr: usize,
    pub offset: usize,
}

/// One dispatch, with everything an ICB command must carry.
///
/// `DispatchRecord` deliberately does not hold bindings -- an executor that only
/// counts should not pay for them -- so this is the replay-facing record that
/// pairs a dispatch with the binds that preceded it.
#[derive(Clone, Debug)]
pub struct StepDispatch {
    pub kernel: Option<Arc<str>>,
    pub grid: Grid,
    pub threads_per_threadgroup: Size3,
    pub bindings: Vec<BindingRecord>,
    /// The pipeline this position ran, retained so the ICB command can be
    /// encoded against it later. Metal offers no way to read a bound pipeline
    /// back off an encoder, which is why `ComputePipeline` carries its name and
    /// why this has to be captured as it happens.
    pub pipeline: Option<ComputePipeline>,
    /// Every buffer bound at this position, retained for residency.
    ///
    /// `useResource` is mandatory for anything an ICB command touches and
    /// omitting it is silent corruption rather than an error (§3.7). Unified
    /// memory can mask the omission, which makes it exactly the kind of thing to
    /// do correctly rather than discover.
    pub retained: Vec<Buffer>,
}

impl StepDispatch {
    /// Whether two recordings of the same position agree in everything an ICB
    /// command freezes.
    ///
    /// Deliberately compares the *encoded* quantities and not, say, a step
    /// index: an ICB command holds a pipeline, a grid, a threadgroup size and a
    /// set of `(index, buffer, offset)` triples, and two steps that agree on all
    /// of those are two steps one command can serve.
    fn replay_compatible(&self, other: &StepDispatch) -> bool {
        self.kernel == other.kernel
            && self.grid == other.grid
            && self.threads_per_threadgroup == other.threads_per_threadgroup
            && self.bindings == other.bindings
    }
}

/// Why a position is not replayed, kept so the executor can report coverage
/// with the reasons attached rather than as a bare count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Excluded {
    /// Bindings, grid or threadgroup size differed between recorded steps.
    Varies,
    /// The pipeline was not captured, so there is nothing to encode.
    NoPipeline,
    /// The kernel binds its constants inline, so no ICB command can hold it
    /// (§3.7c). Observed as a name without the `_packed` suffix.
    InlineConstants,
}

/// What one recording pass learned about the decode step.
#[derive(Default)]
struct Recording {
    /// Dispatches per recorded step, in submission order.
    steps: Vec<Vec<StepDispatch>>,
    /// The step being accumulated.
    current: Vec<StepDispatch>,
    /// Bindings seen since the last dispatch, i.e. those belonging to the next.
    pending: Vec<BindingRecord>,
    pending_retained: Vec<Buffer>,
    pending_pipeline: Option<ComputePipeline>,
}

/// The executor's phase. Recording, then replaying, and never back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Forwarding every dispatch and recording what it was.
    Recording,
    /// Replaying the covered positions, forwarding the rest.
    Replaying,
}

/// Coverage, as observed rather than as intended.
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub positions: usize,
    pub covered: usize,
    pub varies: usize,
    pub no_pipeline: usize,
    pub inline_constants: usize,
    /// Covered positions per kernel name, so a report can say *what* is replayed
    /// rather than how much.
    pub covered_by_kernel: std::collections::BTreeMap<String, usize>,
    pub excluded_by_kernel: std::collections::BTreeMap<(String, &'static str), usize>,
}

/// Replay a decode step's stable, packed-parameter dispatches from an ICB.
pub struct IcbExecutor {
    inner: Mutex<Inner>,
}

struct Inner {
    phase: Phase,
    /// How many steps to record before deciding what is replayable.
    ///
    /// Two is the minimum that can distinguish "stable" from "seen once", and
    /// more is better: a position that happens to agree between two consecutive
    /// steps but drifts on the third would be wrongly admitted. The default is
    /// higher than two for that reason.
    record_steps: usize,
    recording: Recording,
    plan: Option<Plan>,
    /// The ICB and its residency set, handed to the encoder at each run head.
    range: Option<Arc<IcbRange>>,
    coverage: Coverage,
    /// Positions whose live dispatch stopped matching the recording.
    ///
    /// Zero is the expected value on a steady-state decode, and a nonzero value
    /// means coverage is lower than the plan claims -- reporting the plan's
    /// figure while this is nonzero would overstate what ran.
    stale_positions: usize,
    /// Runs that went stale and are never replayed again.
    poisoned: std::collections::HashSet<usize>,
    /// Positions before this index are inside a run whose range already ran.
    suppress_until: usize,
    /// The run whose range was most recently requested, so a stale member can
    /// name the run to poison.
    run_in_flight: usize,
    /// Dispatch index within the step being executed.
    position: usize,
}

/// One maximal run of consecutive covered positions.
///
/// # Why runs, and not one range over the whole step
///
/// An ICB executes its commands as a block wherever the encoder calls
/// `executeCommandsInBuffer`. The covered set is **not** contiguous -- measured
/// on `Sdpa × arena`, 431 covered positions form **31 runs** (12/14/17 long, one
/// group per layer, no singletons; `measurements/issue-115-raw/covered-runs.txt`).
/// So a single execute call would run all 431 at one point in the step, moving
/// each of them across the uncovered dispatches that sat between them.
///
/// That is not a reordering the classical path would tolerate. Dispatches within
/// an encoder overlap and the GPU does not drain between them (§3.5), so the
/// only thing that orders a write against a later read is a barrier candle
/// emitted from the *original* interleaving. Hoisting a dispatch across one is a
/// silent wrong answer rather than an error, which is §3.5's whole point.
///
/// One execute call per run keeps every covered dispatch between the same two
/// uncovered neighbours it had classically, so the barrier structure still
/// applies to it. This is `DESIGN.md` §11.3's range trick used for the reason
/// §11.3 gives -- a computed jump into a pre-encoded list -- rather than for a
/// variable chunk count.
#[derive(Clone, Copy, Debug)]
struct Run {
    /// First dispatch position in the run.
    start: usize,
    /// First ICB command index for the run.
    command: usize,
    len: usize,
}

/// The encoded ICB and the map from dispatch position to command index.
struct Plan {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    /// `command_of[pos]` is the ICB command index for that dispatch position, or
    /// `None` when the position is not covered.
    command_of: Vec<Option<usize>>,
    /// The covered positions grouped into maximal consecutive runs, in order.
    runs: Vec<Run>,
    /// `run_at[pos]` is the run starting at that position, if one does.
    run_at: Vec<Option<usize>>,
    /// The recording each covered position was encoded from, kept so a live
    /// dispatch can be checked against it before its command is trusted.
    reference: Vec<StepDispatch>,
    /// Every buffer any encoded command binds, retained and made resident.
    resident: Vec<Buffer>,
}

// SAFETY: `Plan` holds Metal objects, which are internally thread-safe for the
// operations used here (encoding under the executor's mutex, execution on the
// command buffer's thread). The executor's `Mutex` is what actually serialises
// access; these markers only assert that moving the guard between threads is
// sound, which it is because every use goes through that lock.
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

impl IcbExecutor {
    /// Record `record_steps` decode steps, then replay what was stable.
    pub fn new(record_steps: usize) -> Arc<IcbExecutor> {
        assert!(
            record_steps >= 2,
            "recording fewer than two steps cannot distinguish a stable position from one \
             seen once"
        );
        Arc::new(IcbExecutor {
            inner: Mutex::new(Inner {
                phase: Phase::Recording,
                record_steps,
                recording: Recording::default(),
                plan: None,
                range: None,
                coverage: Coverage::default(),
                stale_positions: 0,
                poisoned: std::collections::HashSet::new(),
                suppress_until: 0,
                run_in_flight: 0,
                position: 0,
            }),
        })
    }

    /// Close the current step.
    ///
    /// Called by the harness at the token boundary, because nothing inside the
    /// encoder knows where a decode step ends -- candle packs dispatches into
    /// command buffers on its own cadence (`CANDLE_METAL_COMPUTE_PER_BUFFER`,
    /// 14 sessions per token), so an encoder session boundary is not a step
    /// boundary. §9.2f records the same distinction costing two simulations
    /// their credibility.
    pub fn end_step(&self, device: &crate::metal::Device) -> Result<(), crate::MetalKernelError> {
        // The constants pool hands position N the same buffer every step, which
        // is what makes a packed dispatch's identity stable enough to encode
        // (see `ConstantsPool`). Its cursor is per step, so it resets here --
        // the same boundary the executor's own position counter resets at, and
        // for the same reason.
        if let Some(pool) = crate::kernels::params::constants_pool() {
            pool.reset();
        }
        let mut inner = self.inner.lock().unwrap();
        inner.position = 0;
        // Per-step, not cumulative: a run's suppression window belongs to the
        // step that opened it. `poisoned` is deliberately *not* reset -- a run
        // that went stale once is not re-armed, see `dispatch_action`.
        inner.suppress_until = 0;
        inner.run_in_flight = 0;
        match inner.phase {
            Phase::Recording => {
                let step = std::mem::take(&mut inner.recording.current);
                inner.recording.steps.push(step);
                inner.recording.pending.clear();
                inner.recording.pending_retained.clear();
                inner.recording.pending_pipeline = None;
                if inner.recording.steps.len() >= inner.record_steps {
                    inner.build_plan(device)?;
                    inner.phase = Phase::Replaying;
                }
            }
            Phase::Replaying => {}
        }
        Ok(())
    }

    /// Coverage as observed, once a plan exists.
    pub fn coverage(&self) -> Coverage {
        self.inner.lock().unwrap().coverage.clone()
    }

    /// Whether a plan has been built and is being replayed.
    pub fn is_replaying(&self) -> bool {
        self.inner.lock().unwrap().phase == Phase::Replaying
    }

    /// Positions that fell back to the classical path because the live dispatch
    /// no longer matched what was recorded.
    ///
    /// Zero is the expected value on a steady-state decode. A nonzero count is
    /// not a correctness problem -- the fallback is what keeps it correct -- but
    /// it means coverage is lower than the plan claims, and reporting the plan's
    /// figure while this is nonzero would overstate what ran.
    pub fn stale_positions(&self) -> usize {
        self.inner.lock().unwrap().stale_positions
    }

    /// How many contiguous runs the covered positions form.
    ///
    /// One `executeCommandsInBuffer` is issued per run, so this is the number of
    /// execute calls a replayed step makes -- and the quantity that says whether
    /// replay is a single computed jump or many. Measured at 31 on
    /// `Sdpa × arena`; see `Run`.
    pub fn runs(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .plan
            .as_ref()
            .map_or(0, |p| p.runs.len())
    }

    /// Runs that went stale and are no longer replayed.
    pub fn poisoned_runs(&self) -> usize {
        self.inner.lock().unwrap().poisoned.len()
    }
}

impl Inner {
    /// Decide what is replayable and encode it.
    fn build_plan(&mut self, device: &crate::metal::Device) -> Result<(), crate::MetalKernelError> {
        let steps = &self.recording.steps;
        let n = steps[0].len();
        // A step whose dispatch count differs is not a steady-state step, and
        // comparing position-by-position across two different lengths would
        // align unrelated dispatches. Refuse rather than truncate.
        for (i, s) in steps.iter().enumerate() {
            if s.len() != n {
                return Err(crate::MetalKernelError::FailedToCreateResource(format!(
                    "recorded step {i} has {} dispatches against step 0's {n}; the recording \
                     window includes a non-steady-state step",
                    s.len()
                )));
            }
        }

        let mut coverage = Coverage {
            positions: n,
            ..Default::default()
        };
        let mut covered_positions = Vec::new();
        for pos in 0..n {
            let base = &steps[0][pos];
            let name = base.kernel.as_deref().unwrap_or("<unnamed>").to_string();
            let mut exclude = None;

            if base.pipeline.is_none() {
                exclude = Some(Excluded::NoPipeline);
            } else if !is_packed(&name) {
                // §3.7c: no ICB command can carry a `setBytes` scalar, so a
                // classical entry point is unencodable whatever else is true of
                // it. Checked by name because that is what the pipeline records
                // and what `ParamStyle::kernel_name` appends.
                exclude = Some(Excluded::InlineConstants);
            } else if !steps[1..].iter().all(|s| base.replay_compatible(&s[pos])) {
                exclude = Some(Excluded::Varies);
            }

            match exclude {
                None => {
                    coverage.covered += 1;
                    *coverage.covered_by_kernel.entry(name).or_default() += 1;
                    covered_positions.push(pos);
                }
                Some(reason) => {
                    let tag = match reason {
                        Excluded::Varies => {
                            coverage.varies += 1;
                            "varies"
                        }
                        Excluded::NoPipeline => {
                            coverage.no_pipeline += 1;
                            "no-pipeline"
                        }
                        Excluded::InlineConstants => {
                            coverage.inline_constants += 1;
                            "inline-constants"
                        }
                    };
                    *coverage.excluded_by_kernel.entry((name, tag)).or_default() += 1;
                }
            }
        }

        let plan = if covered_positions.is_empty() {
            None
        } else {
            Some(encode_plan(device, &steps[0], &covered_positions, n)?)
        };
        self.range = plan.as_ref().map(|p| {
            Arc::new(IcbRange {
                icb: p.icb.clone(),
                resident: p.resident.clone(),
            })
        });
        self.coverage = coverage;
        self.plan = plan;
        Ok(())
    }
}

/// Whether a kernel name is a packed entry point, i.e. whether its constants
/// arrive in a buffer.
fn is_packed(name: &str) -> bool {
    name.ends_with(crate::kernels::params::PACKED_SUFFIX)
}

/// Build the ICB and encode one command per covered position.
fn encode_plan(
    device: &crate::metal::Device,
    step: &[StepDispatch],
    covered: &[usize],
    positions: usize,
) -> Result<Plan, crate::MetalKernelError> {
    // Trap 1 (§3.7d). Checked here rather than trusted, because the failure is a
    // segfault inside `setComputePipelineState:` a few lines below and there is
    // no error to catch. `set_pipelines_support_icb` has to have been called
    // before any pipeline was built, and by this point they all have been.
    if !crate::metal::device::pipelines_support_icb() {
        return Err(crate::MetalKernelError::FailedToCreatePipeline(
            "pipelines were not built with supportIndirectCommandBuffers; encoding one into an \
             ICB segfaults at encode time (DESIGN.md §3.7d). Call \
             set_pipelines_support_icb(true) before the first dispatch"
                .to_string(),
        ));
    }

    let max_bind = covered
        .iter()
        .flat_map(|&p| step[p].bindings.iter().map(|b| b.index + 1))
        .max()
        .unwrap_or(1);

    let desc = MTLIndirectCommandBufferDescriptor::new();
    // Dispatch cannot be mixed with any other command type in one descriptor
    // (`MTLIndirectCommandBuffer.h`), so this is the only type set.
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    // Every command carries its own pipeline and its own bindings.
    // `inheritBuffers` would make them all inherit the parent encoder's
    // identically, which cannot express per-dispatch bindings -- the only reason
    // to want them (§3.7c).
    desc.setInheritBuffers(false);
    desc.setInheritPipelineState(false);
    desc.setMaxKernelBufferBindCount(max_bind);

    // SAFETY: the descriptor is fully initialised above and the count is the
    // number of commands about to be encoded.
    let icb = unsafe {
        device
            .as_ref()
            .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &desc,
                covered.len(),
                MTLResourceOptions::empty(),
            )
    }
    .ok_or_else(|| {
        crate::MetalKernelError::FailedToCreateResource(format!(
            "MTLIndirectCommandBuffer of {} commands",
            covered.len()
        ))
    })?;

    let mut command_of = vec![None; positions];
    let mut run_at = vec![None; positions];
    let mut runs: Vec<Run> = Vec::new();
    let mut reference = Vec::with_capacity(covered.len());
    let mut resident: Vec<Buffer> = Vec::new();
    let mut seen_resident = std::collections::HashSet::new();

    for (cmd_index, &pos) in covered.iter().enumerate() {
        let d = &step[pos];
        // SAFETY: `cmd_index < covered.len()`, the count the ICB was created
        // with -- the loop is over exactly that slice.
        let cmd = unsafe { icb.indirectComputeCommandAtIndex(cmd_index) };
        let pipeline = d
            .pipeline
            .as_ref()
            .expect("covered positions have a pipeline; checked in build_plan");
        cmd.setComputePipelineState(pipeline_raw(pipeline));
        for (b, buf) in d.bindings.iter().zip(d.retained.iter()) {
            // SAFETY: the buffer outlives the ICB -- it is retained in
            // `resident` below and the plan owns both -- and the index is under
            // `maxKernelBufferBindCount`, which was computed as the maximum over
            // exactly these bindings.
            unsafe { cmd.setKernelBuffer_offset_atIndex(buf.as_ref(), b.offset, b.index) };
            if seen_resident.insert(buf.raw_addr()) {
                resident.push(buf.clone());
            }
        }
        let tg = MTLSize {
            width: d.threads_per_threadgroup.width,
            height: d.threads_per_threadgroup.height,
            depth: d.threads_per_threadgroup.depth,
        };
        // The two forms are kept distinct rather than normalised. §11.1c: the
        // conversion is lossy, because `dispatchThreads` lets Metal size the
        // final partial threadgroup and a rounded replay would run more threads
        // and report a different `threads_per_threadgroup` inside the kernel.
        // Any threadgroup-wide reduction would then compute a different answer.
        match d.grid {
            Grid::Threadgroups(g) => cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: g.width,
                    height: g.height,
                    depth: g.depth,
                },
                tg,
            ),
            Grid::Threads(g) => cmd.concurrentDispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: g.width,
                    height: g.height,
                    depth: g.depth,
                },
                tg,
            ),
        }
        command_of[pos] = Some(cmd_index);
        reference.push(d.clone());

        // Extend the run in progress, or open one. `covered` is ascending and
        // `cmd_index` walks it, so a run's commands are consecutive in the ICB
        // exactly when its positions are consecutive in the step -- which is
        // what lets one `executeCommandsInBuffer` serve a whole run.
        match runs.last_mut() {
            Some(run) if run.start + run.len == pos => run.len += 1,
            _ => {
                run_at[pos] = Some(runs.len());
                runs.push(Run {
                    start: pos,
                    command: cmd_index,
                    len: 1,
                });
            }
        }
    }

    // Trap 2 (§3.7a): commands the CPU never initialised are undefined
    // behaviour. Here the ICB is created with exactly `covered.len()` commands
    // and all of them are encoded, so there is no tail -- but the reset is
    // written anyway, over the empty range, so that changing the sizing later
    // cannot silently reintroduce the hazard. `icb_unused_tail_must_be_reset`
    // is the test that pins the obligation.
    for i in covered.len()..icb.size() {
        // SAFETY: `i < icb.size()`, the count the ICB was created with.
        unsafe { icb.indirectComputeCommandAtIndex(i) }.reset();
    }

    Ok(Plan {
        icb,
        command_of,
        runs,
        run_at,
        reference,
        resident,
    })
}

fn pipeline_raw(p: &ComputePipeline) -> &ProtocolObject<dyn MTLComputePipelineState> {
    p.as_ref()
}

impl Executor for IcbExecutor {
    fn dispatch_action(&self, record: &DispatchRecord) -> DispatchAction {
        let mut inner = self.inner.lock().unwrap();
        let pos = inner.position;
        inner.position += 1;

        let bindings = std::mem::take(&mut inner.recording.pending);
        let retained = std::mem::take(&mut inner.recording.pending_retained);
        let pipeline = inner.recording.pending_pipeline.take();
        let live = StepDispatch {
            kernel: record.kernel.clone(),
            grid: record.grid,
            threads_per_threadgroup: record.threads_per_threadgroup,
            bindings,
            pipeline,
            retained,
        };

        match inner.phase {
            Phase::Recording => {
                inner.recording.current.push(live);
                // Recording forwards everything: the recorded step has to be a
                // real decode step, or the model's state diverges from the one
                // the replay was derived from.
                DispatchAction::Encode
            }
            Phase::Replaying => {
                let Some(plan) = inner.plan.as_ref() else {
                    return DispatchAction::Encode;
                };
                let Some(Some(cmd)) = plan.command_of.get(pos).copied() else {
                    // Not covered: the classical path runs it, which is what
                    // makes this executor partial rather than wrong.
                    return DispatchAction::Encode;
                };
                // The live dispatch must still be what the command was encoded
                // from. This is the validation half of §11.1a's
                // record-and-validate: replaying a command whose buffers have
                // moved is silent corruption under
                // `HazardTrackingModeUntracked` (§3.5), so a mismatch falls back
                // rather than trusting the plan.
                //
                // **The unit of that decision is the run, not the position.** A
                // run is executed by one `executeCommandsInBuffer` at its head,
                // so by the time a later member of the run is offered, its
                // command has already run. Falling back *there* would encode the
                // dispatch a second time and the position would execute twice --
                // for `badd_f16` that is a residual added twice, which is a
                // plausible wrong answer rather than a crash. So a run is
                // validated as a whole, at its head, and either replays entirely
                // or not at all.
                //
                // Positions inside a run are therefore not re-checked here: the
                // head checked them, and its verdict is recorded in
                // `suppress_until`.
                // Inside a run whose head already executed the range.
                if pos < inner.suppress_until {
                    // Still checked, and the verdict is *reported* rather than
                    // acted on: the command has already run, so encoding it now
                    // would execute this position twice -- for `badd_f16` a
                    // residual added twice, which is a plausible wrong answer
                    // rather than a crash. What the check buys is that a run
                    // going stale mid-flight becomes a nonzero `stale_positions`
                    // the harness must report, instead of silence.
                    //
                    // It is not a race that can be lost quietly: a stale member
                    // disables its run from the *next* step onward (`poisoned`),
                    // and the digest gate is what says whether the step it was
                    // first seen on was already wrong.
                    if !plan.reference[cmd].replay_compatible(&live) {
                        inner.stale_positions += 1;
                        let run_index = inner.run_in_flight;
                        inner.poisoned.insert(run_index);
                    }
                    return DispatchAction::Suppress;
                }
                let Some(run_index) = plan.run_at.get(pos).copied().flatten() else {
                    // Covered, but neither a run head nor inside a live run --
                    // so its run fell back at the head. Encode it classically.
                    let _ = cmd;
                    return DispatchAction::Encode;
                };
                // A run that ever went stale is never replayed again. One
                // position's operands moving means the recording no longer
                // describes this step, and re-arming it per step would rediscover
                // that after executing the range rather than before.
                if inner.poisoned.contains(&run_index) {
                    return DispatchAction::Encode;
                }
                if !plan.reference[cmd].replay_compatible(&live) {
                    inner.stale_positions += 1;
                    inner.poisoned.insert(run_index);
                    return DispatchAction::Encode;
                }
                let run = plan.runs[run_index];
                let range = Arc::clone(
                    inner
                        .range
                        .as_ref()
                        .expect("a plan exists, so its IcbRange was built beside it"),
                );
                // Suppress the rest of this run: their commands run as part of
                // the range being requested now.
                inner.suppress_until = run.start + run.len;
                inner.run_in_flight = run_index;
                DispatchAction::ExecuteIcb {
                    icb: range,
                    location: run.command,
                    length: run.len,
                }
            }
        }
    }

    fn will_bind_buffer(&self, index: usize, buffer: &Buffer, offset: usize, _is_output: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.recording.pending.push(BindingRecord {
            index,
            ptr: buffer.raw_addr(),
            offset,
        });
        inner.recording.pending_retained.push(buffer.clone());
    }

    fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        let mut inner = self.inner.lock().unwrap();
        inner.recording.pending_pipeline = Some(pipeline.clone());
    }
}
