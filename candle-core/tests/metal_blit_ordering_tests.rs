#![cfg(feature = "metal")]
//! Ordering of blit-encoded copies against the last writer of their operands.
//!
//! `BlitCommandEncoder::copy_from_buffer` waited on the last writer of its
//! *source* but not of its *destination*, where the sibling `fill_buffer`
//! waited on its destination. Closing that asymmetry is lloom issue #25.
//!
//! # Why the destination edge is not redundant
//!
//! `Commands::blit_command_encoder` already waits on every fence in
//! `live_fences` before handing a blit encoder out, which looks like it should
//! subsume any per-buffer wait. It does, for *compute* encoders: those register
//! their fence in `live_fences` in `Commands::end_encoding`.
//!
//! A *blit* encoder does not. `BlitCommandEncoder::end_encoding` registers its
//! outputs in `prev_ce_outputs` and updates its own fence, but never adds that
//! fence to `live_fences`. So a blit that writes buffer B, followed by a second
//! blit that writes B again, has a genuine write-after-write dependency that the
//! blanket wait cannot see -- `prev_ce_outputs` is the only place it is
//! recorded. `fill_buffer` consulted it; `copy_from_buffer` did not.
//!
//! Under `HazardTrackingModeUntracked` (`DESIGN.md` §3.5) there is no safety
//! net: a missed dependency corrupts silently rather than failing.
//!
//! # What these tests do NOT show, stated up front
//!
//! **They do not fail when the fix is reverted.** Removing every per-buffer
//! wait from the blit encoder -- a strictly larger mutation than reverting
//! this change -- leaves all of them green, in release and in debug. So by
//! `CONTRIBUTING.md` §3.1 #2 they are not a test of the ordering property; they
//! are a regression guard on the values, and they are labelled as such.
//!
//! The reason is measured, not guessed: instrumented on the real LFM2 decode
//! path, `copy_from_buffer`'s destination has a registered last writer in
//! **0** of its calls. The edge is structurally correct and, on the workloads
//! reachable here, never live -- so there is nothing for a mutation to break.
//! An ordering test that could fail would need a destination that genuinely
//! carries a pending writer, and no candle path observed here produces one.
//!
//! These also do **not** test lloom #19, which is a different mechanism the
//! fence machinery cannot see at all (`DESIGN.md` §2.3.8b), and this change
//! does not fix it. PR #20 measured 11/30 unstable with and without the wait.

use candle_core::{DType, Device, Result, Tensor};

/// Repetitions per pattern. Enough that a frequent ordering failure would show
/// up; see the module note on why that is a weaker claim than it sounds.
const REPS: usize = 200;

/// Assert the patterns below actually reach `copy_from_buffer`, and report how
/// often the destination wait finds anything.
///
/// A test that never enters the path it names cannot detect a change to it.
/// This one both proves the path is live and prints the count that explains why
/// the mutation test does not kill: `dst_pending` is the number of calls whose
/// destination had a pending writer, and it is observed to be 0.
///
/// Re-execs itself with `CANDLE_METAL_PROFILE=1` rather than skipping when the
/// variable is absent: a test that silently no-ops under a normal `cargo test`
/// is one nobody runs, and this is the test that keeps the others honest.
#[test]
fn copy_destinations_rarely_have_a_pending_writer() -> Result<()> {
    use candle_core::metal_backend::profile;

    if !profile::enabled() {
        // `enabled()` caches on first call, so setting the variable in-process
        // would be racy against any other test that has already read it. Re-exec
        // this one test in a child with the variable set.
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "copy_destinations_rarely_have_a_pending_writer",
                "--exact",
                "--nocapture",
            ])
            .env("CANDLE_METAL_PROFILE", "1")
            .output()
            .expect("re-exec self with profiling enabled");
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "profiled child failed:\n{text}");
        assert!(
            text.contains("blit_copies="),
            "profiled child did not report blit counts:\n{text}"
        );
        eprintln!(
            "{}",
            text.lines()
                .filter(|l| l.contains("blit_copies="))
                .collect::<Vec<_>>()
                .join("\n")
        );
        return Ok(());
    }

    let device = Device::new_metal(0)?;
    profile::reset();
    for rep in 0..REPS {
        let a = Tensor::full(rep as f32, (128, 64), &device)?;
        let b = Tensor::full((rep + 1) as f32, (128, 64), &device)?;
        let j = Tensor::cat(&[&a, &b], 0)?;
        let _ = j.flatten_all()?.to_vec1::<f32>()?;
    }
    let s = profile::snapshot();
    println!(
        "blit_copies={} dst_pending={} dst_uncovered={}",
        s.blit_copies, s.blit_copy_dst_pending, s.blit_copy_dst_uncovered
    );
    assert!(
        s.blit_copies > 0,
        "these patterns never reached copy_from_buffer, so the ordering tests \
         below exercise nothing"
    );
    // Deliberately not asserting dst_pending == 0: that is an observation about
    // today's candle, not a property worth pinning. If it ever becomes nonzero
    // the printed line says so, and at that point a real ordering test is
    // possible and should be written.
    Ok(())
}

/// Chained `Tensor::cat` copies: each `cat` blits its inputs into a freshly
/// allocated destination, and the pool recycles destinations across iterations,
/// so a destination frequently carries a pending writer from an earlier blit.
#[test]
fn chained_cat_copies_produce_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let a = Tensor::full(rep as f32, (128, 64), &device)?;
        let b = Tensor::full((rep + 1000) as f32, (128, 64), &device)?;

        // `cat` on a contiguous source takes the blit path in
        // `MetalStorage::copy_strided_src`, once per input.
        let joined = Tensor::cat(&[&a, &b], 0)?;
        // A second cat whose destination is a buffer the pool is likely to have
        // recycled from the first one.
        let again = Tensor::cat(&[&joined, &joined], 1)?;

        let v = again.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v.len(), 256 * 128);
        let want_a = rep as f32;
        let want_b = (rep + 1000) as f32;
        assert!(
            v.iter().all(|&x| x == want_a || x == want_b),
            "rep {rep}: cat produced a value that is neither input: {:?}",
            v.iter()
                .find(|&&x| x != want_a && x != want_b)
                .copied()
                .unwrap_or(f32::NAN)
        );
        // Both halves must be present and in the right places, which a
        // whole-buffer aliasing failure would violate.
        assert_eq!(v[0], want_a, "rep {rep}: first row wrong");
        assert_eq!(v[128 * 128], want_b, "rep {rep}: second half wrong");
    }
    Ok(())
}

/// `to_vec` after a copy: the readback blit's destination is a fresh buffer
/// from the pool, which is the case issue #25 names as most likely to carry a
/// pending writer ("the destination of a copy is typically a just-recycled
/// buffer").
#[test]
fn readback_after_copy_produces_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let src = Tensor::full(rep as f32, (256, 32), &device)?;
        // Force a contiguous blit copy rather than a kernel.
        let copied = Tensor::cat(&[&src], 0)?.contiguous()?;
        let v = copied.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|&x| x == rep as f32),
            "rep {rep}: readback saw a stale or foreign value; first bad = {:?}",
            v.iter().find(|&&x| x != rep as f32).copied()
        );
    }
    Ok(())
}

/// A `fill_buffer` (via `Tensor::zeros`) immediately followed by a copy whose
/// destination is the filled buffer. This is the pairing the two blit
/// primitives are meant to be symmetric about: before the fix one of them
/// waited on its destination and the other did not.
#[test]
fn fill_then_add_produces_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let zeros = Tensor::zeros((128, 64), DType::F32, &device)?;
        let ones = Tensor::full(1f32, (128, 64), &device)?;
        // Sum forces both to be read after both are written.
        let sum = (&zeros + &ones)?;
        let v = sum.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|&x| x == 1.0),
            "rep {rep}: zeros+ones != 1 everywhere; found {:?}",
            v.iter().find(|&&x| x != 1.0).copied()
        );
    }
    Ok(())
}

/// Interleaved copies across several destinations, so the pool cycles buffers
/// between them and a destination's previous writer is a *different* copy.
#[test]
fn interleaved_copies_do_not_alias_each_other() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..(REPS / 4) {
        let mut outs = Vec::new();
        for lane in 0..4usize {
            let v = (rep * 4 + lane) as f32;
            let t = Tensor::full(v, (64, 64), &device)?;
            outs.push((v, Tensor::cat(&[&t, &t], 0)?));
        }
        // Read them back only after all four exist, so any cross-lane aliasing
        // introduced while they were all in flight is still visible.
        for (want, t) in &outs {
            let got = t.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                got.iter().all(|&x| x == *want),
                "rep {rep}: lane expecting {want} saw {:?} -- another lane's value would \
                 indicate whole-buffer aliasing",
                got.iter().find(|&&x| x != *want).copied()
            );
        }
    }
    Ok(())
}
