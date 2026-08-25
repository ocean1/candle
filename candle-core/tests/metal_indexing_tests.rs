//! `indexing.metal`'s four op families must agree with the CPU backend, and
//! every `(index dtype, value dtype)` pair `candle-core` can ask for must
//! exist.
//!
//! Two distinct things are checked here, and they fail for different reasons.
//!
//! **Parity** — the kernels compute the same values as the CPU backend. This is
//! the check a templating conversion needs: `indexing.metal`'s macro form
//! restated each kernel's signature beside the template it forwarded to, and
//! `DESIGN.md` §8.1c records the hazard that shape carries — reordering two
//! same-typed parameters in one of the two lists compiles, resolves, and reads
//! its bindings wrongly. `index` takes five consecutive `constant size_t &`
//! parameters, so that shape was reachable here.
//!
//! **Existence** — the pair resolves to a kernel at all. `index_select` with
//! `I64` indices over a `U8` or `U32` tensor was a runtime failure on
//! `lloom/integration` before this file existed: `candle-core` named
//! `is_i64_u8` and `is_i64_u32`, and `indexing.metal` declared neither. That is
//! the third and fourth name in the absent-variant class `DESIGN.md` §8.1b
//! tracks, after #26's 48 reduce variants and `conv`'s.
//!
//! The reference is the **CPU backend**, per `CONTRIBUTING.md` §3.1 and the
//! standing guidance in `DESIGN.md` §2.3.8a: it is bit-stable, and a Metal-side
//! reference would make this test depend on the path it is meant to validate.

#![cfg(feature = "metal")]

use candle_core::{DType, Device, IndexOp, Result, Tensor};

/// Values chosen so a transposed or misread dimension changes the answer.
///
/// A tensor of equal rows would pass a parity test with `left_size` and
/// `right_size` swapped, so the generator varies along both axes.
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32) - 50.0).collect()
}

fn cpu_and_metal() -> Option<(Device, Device)> {
    let metal = Device::new_metal(0).ok()?;
    Some((Device::Cpu, metal))
}

/// Compare a tensor against the same computation on the CPU, exactly.
///
/// Exact rather than tolerant: every op in this file *moves* values rather than
/// computing on them, so any difference is an addressing bug, not float noise.
/// `scatter_add` and `index_add` are the exceptions and are compared in `f32`
/// where the accumulation order is identical between backends.
fn assert_same(metal: &Tensor, cpu: &Tensor, what: &str) {
    let m = metal.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let c = cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        m.len(),
        c.len(),
        "{what}: metal produced {} values, cpu {}",
        m.len(),
        c.len()
    );
    assert_eq!(m, c, "{what}: metal and cpu disagree");
}

/// Every `(ids dtype, value dtype)` pair `MetalStorage::index_select` names.
///
/// Written out rather than derived so this list is the *caller's* view: it is
/// what `candle-core`'s match arms ask for, and the point of the test is to
/// compare that against what `indexing.metal` declares. Deriving it from the
/// registry would compare the registry against itself.
const INDEX_SELECT_PAIRS: &[(DType, DType)] = &[
    (DType::U8, DType::U8),
    (DType::U8, DType::U32),
    (DType::U8, DType::I64),
    (DType::U8, DType::BF16),
    (DType::U8, DType::F32),
    (DType::U8, DType::F16),
    (DType::U32, DType::U8),
    (DType::U32, DType::U32),
    (DType::U32, DType::I64),
    (DType::U32, DType::F32),
    (DType::U32, DType::F16),
    (DType::U32, DType::BF16),
    // The two that did not exist. `index_select` on a U8 or U32 tensor with
    // I64 indices was a `LoadFunctionError` from inside a forward pass.
    (DType::I64, DType::U8),
    (DType::I64, DType::U32),
    (DType::I64, DType::I64),
    (DType::I64, DType::F32),
    (DType::I64, DType::F16),
    (DType::I64, DType::BF16),
];

/// `index_select` must resolve and agree with the CPU for all 18 pairs.
///
/// The two `I64`-indexed integer pairs are the regression guard: before the
/// kernels were declared this test failed on them with a `LoadFunctionError`,
/// which is the failure a user saw at runtime.
#[test]
fn index_select_every_dtype_pair_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (6usize, 4usize);
    let raw = values(rows * cols);
    // Deliberately unsorted and repeating: a kernel that ignored `ids` and
    // copied rows in order would pass against `[0, 1, 2]`.
    let ids_raw: Vec<u32> = vec![4, 0, 3, 3, 1];

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in INDEX_SELECT_PAIRS {
        // Values must survive a round trip through the value dtype on both
        // backends, so u8 gets a range it can hold. Otherwise the comparison
        // would be against differing truncation, not against the kernel.
        let src: Vec<f32> = match val_dtype {
            DType::U8 => raw.iter().map(|v| (v.abs() % 128.0).trunc()).collect(),
            DType::U32 | DType::I64 => raw.iter().map(|v| v.abs().trunc()).collect(),
            _ => raw.clone(),
        };

        for dim in [0usize, 1usize] {
            let n = if dim == 0 { rows } else { cols };
            let ids_v: Vec<u32> = ids_raw.iter().map(|i| i % n as u32).collect();

            let build = |dev: &Device| -> Result<Tensor> {
                let t = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
                let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?.to_dtype(id_dtype)?;
                t.index_select(&ids, dim)?.to_dtype(DType::F32)
            };

            let got = build(&metal).unwrap_or_else(|e| {
                panic!("index_select ids={id_dtype:?} values={val_dtype:?} dim={dim}: {e}")
            });
            let want = build(&cpu)?;
            assert_same(
                &got,
                &want,
                &format!("index_select ids={id_dtype:?} values={val_dtype:?} dim={dim}"),
            );
            checked += 1;
        }
    }

    // Non-vacuity: 18 pairs x 2 dims. A loop that silently ran zero iterations
    // would otherwise be a green test (`DESIGN.md` §8.1b, and the guard
    // `conv_names_resolve` carries for the same reason).
    assert_eq!(checked, 36, "expected 36 index_select cases, ran {checked}");
    Ok(())
}

/// `index_select` on a non-contiguous source computes the wrong values.
///
/// **This documents a pre-existing defect rather than asserting correctness**,
/// and it is deliberately written to pass *while the bug is present* so that it
/// records the behaviour without failing a suite that has nothing to do with
/// it. Found while adding the parity tests above; not fixed here, because this
/// PR's scope is templating plus the name registry and a kernel signature
/// change is a different concern with a different failure mode.
///
/// The defect, in `indexing.metal`'s `index`:
///
/// ```text
/// get_strided_index(src_i, src_dim_size, src_dims, src_strides)
/// //                       ^^^^^^^^^^^^ num_dims is expected here
/// ```
///
/// `src_dim_size` is the extent of the *indexed dimension*; `get_strided_index`
/// wants the tensor's *rank*. The CUDA kernel for the same op passes `num_dims`
/// (`candle-kernels/src/indexing.cu:65`), so this is a Metal-only divergence
/// from the reference implementation, not a shared convention.
///
/// It is latent exactly when `rank != dims[dim]`, which is why no existing test
/// caught it: at rank 2 with a 2-long indexed dimension the two coincide. It is
/// reachable — `index_select` is the only op in `metal_backend/mod.rs` that
/// passes `is_contiguous()` through to the kernel instead of rejecting a
/// non-contiguous source, so the strided path is live.
///
/// LFM2 is unaffected: the embedding lookup is contiguous, so it takes the
/// `contiguous ? src_i : ...` fast arm and never reaches `get_strided_index`.
#[test]
fn index_select_strided_source_is_known_wrong() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let ids_v: Vec<u32> = vec![2, 0, 1, 1];
    let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), &metal)?;

    // Rank 2, indexed dimension 5 long: `num_dims` and `src_dim_size` differ,
    // which is the condition that exposes the confusion.
    let t = Tensor::from_vec(values(3 * 5), (3, 5), &metal)?.to_dtype(DType::F32)?;
    let strided = t.t()?;
    assert!(
        !strided.is_contiguous(),
        "the transposed source must be non-contiguous, or this tests nothing"
    );

    let via_strided = strided.index_select(&ids, 0)?;
    let via_contiguous = strided.contiguous()?.index_select(&ids, 0)?;

    // The contiguous arm is correct -- pinned against the CPU, so the
    // comparison below is anchored outside Metal rather than to itself.
    let cpu_t = Tensor::from_vec(values(3 * 5), (3, 5), &cpu)?
        .to_dtype(DType::F32)?
        .t()?
        .contiguous()?;
    let cpu_ids = Tensor::from_vec(ids_v, 4, &cpu)?;
    assert_same(
        &via_contiguous,
        &cpu_t.index_select(&cpu_ids, 0)?,
        "index_select contiguous arm vs cpu",
    );

    // And the strided arm disagrees with it. Asserted as *inequality* so this
    // test turns red the moment the kernel is fixed -- at which point it should
    // be rewritten as the equality it ought to be, which is the reminder.
    let s = via_strided.flatten_all()?.to_vec1::<f32>()?;
    let c = via_contiguous.flatten_all()?.to_vec1::<f32>()?;
    assert_ne!(
        s, c,
        "the strided index_select defect appears to be fixed -- if so, replace \
         this test with an equality assertion against the CPU backend"
    );
    Ok(())
}

/// `gather` must agree with the CPU across its declared dtype pairs.
#[test]
fn gather_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (4usize, 5usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U32, DType::I64),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::U8, DType::BF16),
        (DType::U8, DType::U8),
        (DType::U8, DType::U32),
        (DType::U8, DType::I64),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::BF16),
        (DType::I64, DType::U32),
        (DType::I64, DType::I64),
    ];

    // One index per output element, varying along both axes so a swapped
    // `right_size`/`left_size` changes the result.
    let idx: Vec<u32> = (0..rows * cols).map(|i| ((i * 3) % cols) as u32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let src: Vec<f32> = match val_dtype {
            DType::U8 => values(rows * cols)
                .iter()
                .map(|v| (v.abs() % 128.0).trunc())
                .collect(),
            DType::U32 | DType::I64 => values(rows * cols)
                .iter()
                .map(|v| v.abs().trunc())
                .collect(),
            _ => values(rows * cols),
        };

        let build = |dev: &Device| -> Result<Tensor> {
            let t = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(idx.clone(), (rows, cols), dev)?.to_dtype(id_dtype)?;
            t.gather(&ids, 1)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("gather ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("gather ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 16, "expected 16 gather cases, ran {checked}");
    Ok(())
}

/// `scatter_add` must agree with the CPU.
///
/// Compared in `f32` and with each destination slot written at most once, so
/// the comparison is of *addressing* rather than of accumulation order —
/// float addition is not associative (`DESIGN.md` §2.3.2) and a test that
/// depended on summation order would be checking something this kernel does not
/// promise.
#[test]
fn scatter_add_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (3usize, 4usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::U8, DType::BF16),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::BF16),
    ];

    // A permutation per row: every destination slot receives exactly one value.
    let idx: Vec<u32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| ((c + r) % cols) as u32))
        .collect();
    let src: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let build = |dev: &Device| -> Result<Tensor> {
            let dst = Tensor::zeros((rows, cols), val_dtype, dev)?;
            let s = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(idx.clone(), (rows, cols), dev)?.to_dtype(id_dtype)?;
            dst.scatter_add(&ids, &s, 1)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("scatter_add ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("scatter_add ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 10, "expected 10 scatter_add cases, ran {checked}");
    Ok(())
}

/// `index_add` must agree with the CPU.
///
/// Distinct indices for the same reason `scatter_add` uses a permutation: this
/// checks addressing, not accumulation order.
#[test]
fn index_add_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (5usize, 3usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::U32),
    ];

    // Distinct destination rows, unsorted so order is exercised.
    let ids_v: Vec<u32> = vec![3, 0, 4];
    let src: Vec<f32> = (0..ids_v.len() * cols).map(|i| (i % 9) as f32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let build = |dev: &Device| -> Result<Tensor> {
            let dst = Tensor::zeros((rows, cols), val_dtype, dev)?;
            let s = Tensor::from_vec(src.clone(), (ids_v.len(), cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?.to_dtype(id_dtype)?;
            dst.index_add(&ids, &s, 0)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("index_add ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("index_add ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 9, "expected 9 index_add cases, ran {checked}");
    Ok(())
}

/// The embedding-lookup shape specifically: `is_u32_f16`, one dispatch per
/// decode token.
///
/// `DESIGN.md` §11.3h calls this out as the only `indexing.metal` kernel on the
/// decode path, so it gets a case at the shape LFM2 actually uses — a row
/// gathered from a `[vocab, hidden]` table — rather than only at the small
/// shapes above. A whole-file conversion that broke exactly this one would
/// change generated text while every other case stayed green.
#[test]
fn embedding_lookup_shape_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    // LFM2's hidden size; a small vocab, since the width is what matters.
    let (vocab, hidden) = (64usize, 2048usize);
    let table = values(vocab * hidden);
    let ids_v: Vec<u32> = vec![37];

    let build = |dev: &Device| -> Result<Tensor> {
        let t = Tensor::from_vec(table.clone(), (vocab, hidden), dev)?.to_dtype(DType::F16)?;
        let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?;
        t.index_select(&ids, 0)?.to_dtype(DType::F32)
    };

    let got = build(&metal)?;
    assert_same(&got, &build(&cpu)?, "embedding lookup is_u32_f16");

    // And that it is the row asked for, not row 0 -- an addressing bug that
    // returned a *valid* row would otherwise pass only against the CPU's same
    // bug, which cannot happen here but is cheap to pin.
    let row = got.i(0)?.to_vec1::<f32>()?;
    let want: Vec<f32> = table[37 * hidden..38 * hidden].to_vec();
    assert_eq!(row, want, "embedding lookup returned the wrong row");
    Ok(())
}
