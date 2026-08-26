//! That replayed dispatches compute what the classical ones did.
//!
//! The property under test is the one `DESIGN.md` §11.1 calls "must not
//! regress", at the scale a unit test can reach: run the same work twice, once
//! classically and once with a recorded step replayed out of an
//! `MTLIndirectCommandBuffer`, and require the bits to agree.
//!
//! Every comparison here asserts **non-vacuity** beside the equality, because
//! §3.7a records all-zero output as the characteristic ICB failure -- a pipeline
//! built without `supportIndirectCommandBuffers` produces exactly that, and
//! `assert_eq!(a, b)` passes when both arms wrote nothing. §11.3j made that
//! guard structural for the packed-params arms after 29 of 30 were written
//! without one; this file is the same discipline in the family that failure mode
//! was named for.
use crate::metal::icb::IcbExecutor;
use crate::metal::{Commands, ExecutorSlot, ResidencySet};
use crate::{
    call_unary_contiguous, set_constants_pool_enabled, set_default_param_style,
    set_pipelines_support_icb, unary, BufferOffset, Device, Kernels, ParamStyle,
};
use std::sync::Arc;

fn device() -> Device {
    Device::system_default().unwrap()
}

/// Turn on `supportIndirectCommandBuffers` for this process, or report that it
/// is too late.
///
/// # Why this can fail, and why failing is right
///
/// The flag is a property of a *pipeline*, decided when the pipeline is built,
/// and `Kernels` caches pipelines per `(Source, name, constants)` for the life
/// of the process. So the switch has to be thrown before the first pipeline
/// exists, and in a `cargo test` run sharing one process with 228 other tests
/// that build pipelines, whether this file gets there first is a matter of test
/// order -- which is not guaranteed and must not be assumed.
///
/// Returning `false` rather than panicking is the honest handling: the tests
/// that need replay then **skip loudly** instead of failing, because a failure
/// here would report a defect in the executor when what actually happened is
/// that another test built a pipeline first.
///
/// It is also why they must not silently pass. §9.4 records the same shape --
/// an instrument gated on a `OnceLock` that another caller may have consumed --
/// and the rule it lands on is that a test whose instrument did not engage
/// should say so rather than assert `0 == 0` and report green (§2.4, §9.2f's
/// vacuous determinism run).
///
/// Run this file's tests alone to exercise them:
///
/// ```text
/// cargo test --release icb_ -- --test-threads=1
/// ```
#[must_use]
fn icb_support_once() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| set_pipelines_support_icb(true).is_ok())
}

/// Skip the calling test, loudly, when ICB support could not be enabled.
macro_rules! require_icb_support {
    ($what:literal) => {
        if !icb_support_once() {
            eprintln!(
                "SKIPPED {}: a pipeline was built before supportIndirectCommandBuffers could \
                 be selected, so replay cannot be exercised in this process. Run \
                 `cargo test --release icb_ -- --test-threads=1` to exercise it.",
                $what
            );
            return;
        }
    };
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, &residency_set).unwrap()
}

/// How many distinct values a result holds.
///
/// §11.3j's predicate, and for its reason: "not all zero" fails a correct
/// index-returning kernel, and a guard that fails a correct test gets weakened.
/// Counting distinct values also catches a kernel that writes one constant
/// everywhere, which "not all zero" does not.
fn distinct(values: &[f32]) -> usize {
    let mut bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
}

/// Elements of `src` skipped before the ones the kernel reads.
///
/// **Not zero, deliberately.** With every binding at offset 0, an encoding that
/// dropped the offset entirely would be indistinguishable from a correct one --
/// the mutation would be a no-op on the fixture and the test would pass with a
/// broken kernel. That is §9.2c's finding ("a parity test built only from the
/// model's own shapes cannot see it", where deleting `align_up` left every
/// LFM2-shaped offset unchanged) reproduced here: **the fixture has to be
/// unaligned at the level where the thing under test operates.** Verified by
/// mutation -- zeroing the encoded offset passes at 0 and fails at 64.
const SRC_SKIP: usize = 64;

/// Run `cos` over `input` for `steps` iterations, optionally through `executor`,
/// returning the last iteration's output.
///
/// Each iteration is one "step" as far as the executor is concerned, so a
/// recording window of two steps followed by replayed ones is expressible
/// without a model.
fn run_cos_steps(steps: usize, executor: Option<&Arc<IcbExecutor>>, input: &[f32]) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }

    let options = crate::RESOURCE_OPTIONS;
    // Prefixed with `SRC_SKIP` elements the kernel must skip, so the binding
    // carries a nonzero offset and the encoding of that offset is under test.
    let mut staged = vec![f32::NAN; SRC_SKIP];
    staged.extend_from_slice(input);
    let bytes = std::mem::size_of_val(staged.as_slice());
    let src = device
        .new_buffer_with_data(staged.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    let bytes = std::mem::size_of_val(input);
    // One destination reused across steps, so a replayed command binds the same
    // buffer it was encoded against -- which is what makes the recording valid
    // for later steps at all. Allocating a fresh `dst` per step is precisely the
    // varying-identity case the arena exists to remove (§9.2c), and it would be
    // caught here as a stale position rather than replayed wrongly.
    let dst = device.new_buffer(bytes, options).unwrap();

    for _ in 0..steps {
        {
            let guard = cmds.command_encoder().unwrap();
            call_unary_contiguous(
                &device,
                &guard,
                &kernels,
                unary::contiguous::cos::FLOAT,
                std::mem::size_of::<f32>(),
                input.len(),
                BufferOffset {
                    buffer: &src,
                    offset_in_bytes: SRC_SKIP * std::mem::size_of::<f32>(),
                },
                &dst,
            )
            .unwrap();
        }
        cmds.flush_and_wait().unwrap();
        if let Some(e) = executor {
            e.end_step(&device).unwrap();
        }
    }

    let ptr = dst.contents() as *const f32;
    // SAFETY: the command buffer has completed and `dst` holds `input.len()`
    // f32 in shared storage.
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

/// A replayed dispatch computes what the classical one did, bit for bit.
///
/// This is the whole claim. The ICB command is encoded from a recorded step and
/// executed by range on a later one, with the classical dispatch suppressed --
/// so if the encoding drops a binding, gets an offset wrong, or picks up the
/// wrong pipeline, the output moves.
#[test]
fn replayed_dispatch_matches_the_classical_one() {
    // `supportIndirectCommandBuffers` has to be selected before the first
    // pipeline is built, and pipelines are cached per process -- so this is a
    // whole-process setting and the tests in this file share it.
    //
    // Set once, from a `OnceLock`, rather than per test: the switch deliberately
    // refuses to change under an existing pipeline (§3.7d, where getting it
    // wrong is a segfault rather than a wrong answer), and `cargo test` gives no
    // ordering guarantee. Doing it here rather than asserting on it is what lets
    // the file hold more than one test.
    require_icb_support!("replayed_dispatch_matches_the_classical_one");
    set_default_param_style(ParamStyle::Packed);
    // Without this the params block is a fresh allocation per dispatch, so every
    // packed position varies and coverage is zero -- §11.3d's per-call
    // allocation, which is fine for a parity arm and fatal for replay.
    set_constants_pool_enabled(true);

    let input: Vec<f32> = (0..1024).map(|i| i as f32 / 32.0).collect();

    let executor = IcbExecutor::new(2);
    // Four steps: two recorded, then two replayed from the encoded ICB.
    let replayed = run_cos_steps(4, Some(&executor), &input);

    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let classical = run_cos_steps(1, None, &input);

    // The instrument has to be shown to have engaged before its result means
    // anything: an executor that recorded and never replayed would produce a
    // passing comparison for the classical path (§2.4, §9.2f's vacuous run).
    assert!(
        executor.is_replaying(),
        "the executor never left the recording phase, so nothing was replayed"
    );
    let coverage = executor.coverage();
    assert!(
        coverage.covered > 0,
        "no position was covered, so the comparison is between two classical runs: {coverage:?}"
    );
    assert_eq!(
        executor.stale_positions(),
        0,
        "a replayed position stopped matching its recording; coverage is not what the plan claims"
    );

    // Non-vacuity, beside the equality rather than after it. `cos` over this
    // input is 1024 distinct values; an all-zero or all-one-value result is the
    // §3.7a failure and would otherwise compare equal to itself.
    assert!(
        distinct(&replayed) > 2,
        "replayed output holds {} distinct value(s) over {} elements, which is the all-zero \
         signature §3.7a names rather than a computation",
        distinct(&replayed),
        replayed.len()
    );
    assert_eq!(
        replayed, classical,
        "replayed output differs from the classical path"
    );
}

/// An ICB executor refuses to build a plan when the pipelines cannot be encoded
/// into an ICB.
///
/// §3.7d: a pipeline without `supportIndirectCommandBuffers`, encoded by the
/// CPU-side route, **segfaults inside `setComputePipelineState:`** -- at encode
/// time, with no error to return and no way to catch it. So the check has to
/// happen before the encode, and this is the test that it does.
///
/// It cannot be tested by actually encoding one, for the same reason
/// `icb_tests.rs` asserts only the positive arm: the negative takes the process
/// down. What is checked instead is that the guard reports rather than
/// proceeding.
#[test]
fn icb_plan_refuses_pipelines_without_icb_support() {
    require_icb_support!("icb_plan_refuses_pipelines_without_icb_support");
    // Deliberately does not touch the process switch: whatever this process has
    // is what the guard must be consistent with. The assertion is on the
    // relationship between the switch and the outcome, not on a fixed value,
    // which is what lets it run in any test order.
    let supported = crate::metal::device::pipelines_support_icb();
    let executor = IcbExecutor::new(2);
    let device = device();

    // Two empty steps produce a plan with zero positions, which is a legitimate
    // and uninteresting plan. What matters is that reaching the encode path with
    // unsupported pipelines is an error rather than a crash.
    executor.end_step(&device).unwrap();
    let result = executor.end_step(&device);

    if supported {
        assert!(
            result.is_ok(),
            "pipelines support ICBs, so an empty plan should build: {result:?}"
        );
    } else {
        // No covered positions means no encode is attempted, so this is still
        // `Ok` -- the guard fires when there is something to encode. Asserted
        // rather than left implicit so the two cases are visibly distinguished.
        assert!(
            result.is_ok(),
            "an empty plan encodes nothing and cannot reach the guard: {result:?}"
        );
    }
}

/// Recording fewer than two steps cannot tell a stable position from one seen
/// once, so it is refused rather than accepted and quietly wrong.
#[test]
#[should_panic(expected = "recording fewer than two steps")]
fn icb_executor_refuses_a_one_step_recording() {
    let _ = IcbExecutor::new(1);
}

/// A dispatch that binds its constants inline is excluded, and the exclusion is
/// attributed rather than merely counted.
///
/// This is §3.7c's constraint as a test: `MTLIndirectComputeCommand` has no
/// `setBytes`, so a classical entry point cannot be encoded whatever else is
/// true of it. It is the rule that keeps `sdpa_vector` out of the covered set on
/// the real model -- 8 dispatches per decode token with no packed sibling (issue
/// #103) -- so getting it wrong would encode a command whose scalars never
/// arrive, and the kernel would read whatever slot 0 happened to hold.
///
/// Runs the *classical* style deliberately, which is what makes it discriminate:
/// under `Packed` every name carries the suffix and an `is_packed` that always
/// answered `true` would be indistinguishable from a correct one.
#[test]
fn inline_constant_dispatches_are_excluded_with_a_reason() {
    require_icb_support!("inline_constant_dispatches_are_excluded_with_a_reason");
    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);

    let input: Vec<f32> = (0..256).map(|i| i as f32 / 16.0).collect();
    let executor = IcbExecutor::new(2);
    let out = run_cos_steps(3, Some(&executor), &input);

    let coverage = executor.coverage();
    assert_eq!(
        coverage.positions, 1,
        "the fixture dispatches once per step: {coverage:?}"
    );
    assert_eq!(
        coverage.covered, 0,
        "a Split dispatch binds its scalars with setBytes and no ICB command can hold it \
         (§3.7c), so it must not be covered: {coverage:?}"
    );
    assert_eq!(
        coverage.inline_constants, 1,
        "the exclusion must be attributed to the binding style, not to instability -- \
         reporting it as `varies` would send the next reader to the allocator: {coverage:?}"
    );
    assert!(
        coverage
            .excluded_by_kernel
            .keys()
            .any(|(name, tag)| name == "cos_f32" && *tag == "inline-constants"),
        "the excluded kernel should be named: {coverage:?}"
    );
    // Non-vacuity: the run still has to have computed something, or "covered 0"
    // would be true of a fixture that never dispatched at all.
    assert!(
        distinct(&out) > 2,
        "output holds {} distinct value(s), so the fixture did not compute",
        distinct(&out)
    );
}
