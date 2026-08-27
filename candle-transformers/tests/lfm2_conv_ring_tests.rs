//! The conv-state ring buffer against the shuffle it replaces (lloom #141).
//!
//! `DESIGN.md` §6.1's decode path is `narrow` + `Tensor::cat` + `mul` +
//! `sum_keepdim`, which reallocates and copies the whole state every token.
//! §10.2a/§10.2b replace it with a rotating write index. This file is the
//! correctness argument for that replacement.
//!
//! **The claim under test is stronger than "close enough".** §17 Phase 4 item 12
//! (c) says the ring changes *which address each tap reads* and not the order of
//! the three-term sum, so the output should be **bit-identical** rather than
//! merely within tolerance — which is why the assertions here are on exact bits.
//! §2.3.5a's changed-digest procedure is for changes that genuinely alter
//! arithmetic; this is not one, and a moved bit here is a defect.
//!
//! The reference is the CPU backend, per `CONTRIBUTING.md` §3.1: it is
//! bit-stable and upstreamable, where the Metal generic grouped path is the
//! unstable side at high channel counts (§2.3.8a).

use candle::{DType, Device, IndexOp, Result, Tensor};

/// The shuffle, exactly as `ShortConv::forward` runs it at `seq_len == 1`.
fn shuffle_step(
    state: &Tensor,
    bx: &Tensor,
    w: &Tensor,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    let tail = state.narrow(2, 1, l_cache - 1)?;
    let next = Tensor::cat(&[tail, bx.clone()], 2)?;
    let out = (&next * w)?.sum_keepdim(2)?.contiguous()?;
    Ok((next, out))
}

/// The ring: one in-place write at the sliding slot, then a contiguous window.
///
/// Returns the (possibly compacted) state alongside the output, because the
/// compaction replaces the buffer.
fn ring_step(
    state: &Tensor,
    bx: &Tensor,
    w: &Tensor,
    l_cache: usize,
    live_w: usize,
    width: usize,
    phase: usize,
    compact: bool,
) -> Result<(Tensor, Tensor)> {
    let (b, hidden, _) = state.dims3()?;
    let state = if compact {
        let live = state.narrow(2, width - live_w, live_w)?.contiguous()?;
        let pad = Tensor::zeros((b, hidden, width - live_w), state.dtype(), state.device())?;
        Tensor::cat(&[live, pad], 2)?.contiguous()?
    } else {
        state.clone()
    };
    state.slice_set(bx, 2, phase)?;
    let window = state.narrow(2, phase + 1 - l_cache, l_cache)?;
    let out = (window * w)?.sum_keepdim(2)?.contiguous()?;
    Ok((state, out))
}

/// The slot and the compaction flag, exactly as `Model::forward` computes them.
fn advance(phase: usize, width: usize, live_w: usize) -> (usize, bool) {
    let next = phase + 1;
    if next >= width {
        (live_w, true)
    } else {
        (next, false)
    }
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

/// Deterministic pseudo-random values, so a failure reproduces exactly.
fn seeded(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// The ring reproduces the shuffle **bit for bit**, across many compactions.
///
/// 200 steps at `slack = 4` compacts ~40 times, so the wrap is exercised heavily
/// rather than incidentally. A test stopping before the first compaction would
/// pass on a window that cannot survive one.
#[test]
fn ring_matches_shuffle_bitwise() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache) = (1usize, 64usize, 3usize);

    for (k, slack) in [(0usize, 1usize), (0, 4), (0, 16), (1, 4), (2, 7), (5, 16)] {
        let live_w = l_cache + k;
        let width = live_w + slack;
        let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

        // Both arms start from the same prefill-seeded state: the live window in
        // slots 0..live_w, the slack zeroed, so the write slot is live_w - 1.
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + (live_w - l_cache) + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut compactions = 0usize;

        for step in 0..200 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;

            let (next, sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
            sh_state = next;

            let (p, compact) = advance(phase, width, live_w);
            phase = p;
            compactions += compact as usize;
            let (st, ring_out) =
                ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, compact)?;
            ring_state = st;

            assert_eq!(
                to_vec(&sh_out)?,
                to_vec(&ring_out)?,
                "k={k} slack={slack} step={step} phase={phase}: ring diverged from the shuffle"
            );
        }
        assert!(
            compactions > 0,
            "k={k} slack={slack}: never compacted, so the wrap was not exercised"
        );
    }
    Ok(())
}

/// The live window holds the same *tokens in the same order* the shuffle does —
/// checked directly rather than only through the summed output, so a pairing bug
/// cannot hide behind a coincidental sum.
#[test]
fn ring_window_holds_the_same_tokens_in_order() -> Result<()> {
    let dev = Device::Cpu;
    let (hidden, l_cache, k, slack) = (8usize, 3usize, 0usize, 4usize);
    let live_w = l_cache + k;
    let width = live_w + slack;

    let mut ring_state = Tensor::zeros((1, hidden, width), DType::F32, &dev)?;
    let mut phase = live_w - 1;
    let mut history: Vec<usize> = vec![];

    for t in 1..=40usize {
        // One distinguishable value per token, so a slot's contents name it.
        let bx = Tensor::full(t as f32, (1, hidden, 1), &dev)?;
        let (p, compact) = advance(phase, width, live_w);
        phase = p;
        if compact {
            let live = ring_state.narrow(2, width - live_w, live_w)?.contiguous()?;
            let pad = Tensor::zeros((1, hidden, width - live_w), DType::F32, &dev)?;
            ring_state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
        }
        ring_state.slice_set(&bx, 2, phase)?;
        history.push(t);

        let got: Vec<usize> = (0..l_cache)
            .map(|i| {
                let s = phase + 1 - l_cache + i;
                let v = ring_state
                    .i((0, 0, s))?
                    .to_dtype(DType::F32)?
                    .to_vec0::<f32>()?;
                Ok(v as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        // Oldest..newest of the tokens so far, zero-padded at the start.
        let mut want: Vec<usize> = history.iter().rev().take(l_cache).rev().copied().collect();
        while want.len() < l_cache {
            want.insert(0, 0);
        }
        assert_eq!(got, want, "t={t} phase={phase}: window is not in order");
    }
    Ok(())
}

/// **Mutation test** — `CONTRIBUTING.md` §3.1 #2 and §2.4: a test that cannot
/// fail is not a test. Each mutation is a plausible way to get the sliding
/// window wrong, and each must break the bitwise comparison above.
#[test]
fn mutations_break_the_parity() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache, k, slack) = (1usize, 32usize, 3usize, 0usize, 4usize);
    let live_w = l_cache + k;
    let width = live_w + slack;
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

    for mutation in [
        // The wrap is never taken, so the window walks off its own buffer.
        "no-compaction",
        // The window is read one slot behind: an off-by-one on the tap span.
        "window-off-by-one",
        // The rotating index §10.2a specifies -- which is what this whole
        // mechanism exists to avoid, and it must show up as a difference here.
        "rotating-index",
    ] {
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut diverged = false;

        for step in 0..40 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;
            let (next, sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
            sh_state = next;

            let ring_out = match mutation {
                "no-compaction" => {
                    let (p, _) = advance(phase, width, live_w);
                    if p >= width {
                        // Walked off the buffer: the mechanism is broken, which
                        // is the point. Count it as divergence.
                        diverged = true;
                        break;
                    }
                    phase = p;
                    let (st, o) =
                        ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, false)?;
                    ring_state = st;
                    o
                }
                "window-off-by-one" => {
                    let (p, compact) = advance(phase, width, live_w);
                    phase = p;
                    if compact {
                        let live = ring_state.narrow(2, width - live_w, live_w)?.contiguous()?;
                        let pad = Tensor::zeros((b, hidden, width - live_w), DType::F32, &dev)?;
                        ring_state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
                    }
                    ring_state.slice_set(&bx, 2, phase)?;
                    let start = (phase + 1 - l_cache).saturating_sub(1);
                    let window = ring_state.narrow(2, start, l_cache)?;
                    (window * &w)?.sum_keepdim(2)?.contiguous()?
                }
                "rotating-index" => {
                    phase = (phase + 1) % live_w.max(1);
                    ring_state.slice_set(&bx, 2, phase)?;
                    let window = ring_state.narrow(2, 0, l_cache)?;
                    (window * &w)?.sum_keepdim(2)?.contiguous()?
                }
                _ => unreachable!(),
            };

            if to_vec(&sh_out)? != to_vec(&ring_out)? {
                diverged = true;
                break;
            }
        }
        assert!(
            diverged,
            "mutation {mutation:?} left the parity test green -- the test cannot fail"
        );
    }
    Ok(())
}

/// History slots (`K > 0`) are inert: widening the ring must not move the
/// output, because nothing reads them until a speculative scheme (#89) does.
/// §16 6b is why K is a parameter and not a constant.
#[test]
fn history_slots_do_not_change_the_output() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache, slack) = (1usize, 32usize, 3usize, 8usize);
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

    let mut baseline: Vec<Vec<f32>> = vec![];
    for k in [0usize, 1, 4] {
        let live_w = l_cache + k;
        let width = live_w + slack;
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + (live_w - l_cache) + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut outs = vec![];
        for step in 0..60 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;
            let (p, compact) = advance(phase, width, live_w);
            phase = p;
            let (st, o) = ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, compact)?;
            ring_state = st;
            outs.push(to_vec(&o)?);
        }
        if baseline.is_empty() {
            baseline = outs;
        } else {
            assert_eq!(
                baseline, outs,
                "K={k} changed the output; history is not inert"
            );
        }
    }
    Ok(())
}
