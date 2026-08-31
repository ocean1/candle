#![cfg(feature = "metal")]
//! `Tensor::arg_sort_last_dim` is silently wrong on Metal above 1024 columns.
//!
//! # The defect
//!
//! `ArgSort::metal_fwd` rounds `ncols` up to a power of two and calls
//! `call_arg_sort`, which dispatches **one threadgroup of `ncols_pad` threads**
//! (`candle-metal-kernels/src/kernels/sort.rs:35-42`). A Metal threadgroup holds
//! at most **1024** threads, so at LFM2's 128 000-wide vocabulary the dispatch
//! asks for 131 072 — 128× the limit — and the kernel returns zeros.
//!
//! `candle-metal-kernels` already carries the correct path:
//! `call_mlx_arg_sort` picks `multi_block_sort` when the column count needs more
//! than one block, and that crate's own `mlx_sort` test exercises it at
//! **ncols = 16 000** and passes. It has **no caller in `candle-core`**.
//!
//! # Why this is pinned as known-wrong rather than fixed here
//!
//! Routing `arg_sort_last_dim` to the multi-block path is a correctness change
//! to an entry point every backend consumer shares, and it deserves its own
//! change with its own gates — lloom `DESIGN.md` §8.1e uses exactly this shape
//! (`index_select_strided_source_is_known_wrong`) for a defect found while doing
//! something else: assert the **current wrong behaviour**, so the test turns red
//! when someone fixes it rather than sitting as a TODO nobody runs.
//!
//! # The trap these tests exist to record
//!
//! The obvious check — *"is each element ≥ the next?"* — **passes on the broken
//! kernel**, because an all-zeros output satisfies `v[0] >= v[0]` at every step.
//! §15.1 #1 already says to prefer *"at least two distinct values"* to *"not all
//! zero"*; this is that rule in a family it had not been applied to. What
//! separates them is checking that the result is a **permutation**.

use candle_core::{DType, Device, Result, Tensor};

/// Is `idx` a permutation of `0..n`? This is the check that discriminates; a
/// sortedness check alone does not.
fn is_permutation(idx: &[u32], n: usize) -> bool {
    if idx.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &i in idx {
        let i = i as usize;
        if i >= n || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

fn arg_sort_indices(device: &Device, n: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    // A fixed ramp reversed, so the correct answer is known without needing an
    // RNG: descending arg-sort of [n-1, n-2, …, 0] is [0, 1, …, n-1].
    let values: Vec<f32> = (0..n).map(|i| (n - 1 - i) as f32).collect();
    let t = Tensor::from_vec(values.clone(), (n,), device)?.to_dtype(DType::F32)?;
    let idx: Vec<u32> = t.arg_sort_last_dim(false)?.to_vec1()?;
    Ok((idx, values))
}

#[test]
fn arg_sort_is_correct_at_or_below_one_threadgroup() -> Result<()> {
    let device = Device::new_metal(0)?;
    // 1024 threads is the threadgroup limit, and `ncols_pad` is a power of two,
    // so these are the widths whose dispatch fits.
    for n in [4usize, 64, 256, 1024] {
        let (idx, _) = arg_sort_indices(&device, n)?;
        assert!(
            is_permutation(&idx, n),
            "n={n}: arg_sort returned a non-permutation at a width that fits one \
             threadgroup"
        );
        let expected: Vec<u32> = (0..n as u32).collect();
        assert_eq!(idx, expected, "n={n}: wrong order");
    }
    Ok(())
}

#[test]
fn arg_sort_above_one_threadgroup_is_known_wrong() -> Result<()> {
    let device = Device::new_metal(0)?;

    // **Asserting the defect, not the fix.** When `arg_sort_last_dim` is routed
    // to `call_mlx_arg_sort`, this test FAILS — which is the point: it is the
    // notification that the header above has become stale, and it should be
    // deleted in the same change that fixes the kernel.
    for n in [4096usize, 128_000] {
        let (idx, _) = arg_sort_indices(&device, n)?;
        assert!(
            !is_permutation(&idx, n),
            "n={n}: arg_sort_last_dim now returns a permutation above 1024 \
             columns. If this is deliberate, delete this test and the
             known-wrong note in its header — the defect it pins is fixed."
        );
    }
    Ok(())
}

#[test]
fn a_sortedness_check_alone_cannot_see_the_defect() -> Result<()> {
    let device = Device::new_metal(0)?;

    // The trap, pinned so it is not re-fallen-into. At 128 000 the output is
    // all zeros: it is NOT a permutation, and it IS "sorted" under the naive
    // predicate. A test written the obvious way would be green on a kernel that
    // computes nothing.
    let n = 128_000usize;
    let (idx, values) = arg_sort_indices(&device, n)?;

    let looks_sorted = idx
        .windows(2)
        .all(|w| values[w[0] as usize] >= values[w[1] as usize]);
    assert!(
        looks_sorted,
        "the naive sortedness predicate no longer passes; if the kernel was \
         fixed, this file's premise changed"
    );
    assert!(
        !is_permutation(&idx, n),
        "the permutation check no longer fails: the kernel was fixed and this \
         test should go with it"
    );
    Ok(())
}
