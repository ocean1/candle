//! FlashDecoding on the device: parity, the permuted chunk table, and the
//! merge order (`DESIGN.md` §10.4, §10.3h, issue #116).
//!
//! # The three properties, and which one needs a fixture nothing else has
//!
//! 1. **Parity against a scalar reference.** `CONTRIBUTING.md` §3.1: every
//!    kernel owes a reference implementation and a parity test against it,
//!    mutation-tested. The reference here is a single-pass softmax computed in
//!    `f64` on the CPU, which is *not* the thing being changed — §2.3.5a's
//!    load-bearing discriminator, and the only check that asks whether the
//!    answer is *right* rather than merely stable.
//!
//! 2. **The combine INDEXES its chunk table and never WALKS it** (§10.3h,
//!    §3.7f). Both visit every chunk; only the first is bit-stable, because a
//!    table's contents depend on allocation history, which under B>1 depends on
//!    what other sequences did. **At B=1 the table is the identity and the two
//!    orders coincide**, so a B=1 gate structurally cannot tell them apart —
//!    which is why [`permuted_chunk_table_is_read_by_index_not_by_walk`] builds
//!    a table that is *not* the identity. #197's probe established that a
//!    permuted arm is what separates them, and that a fixture which does not
//!    permute is testing the identity function.
//!
//! 3. **Every declared `[[host_name]]` resolves against the compiled library**
//!    (§8.1b). Not against a second copy of the list: a generator proves two
//!    lists came from one source and says nothing about whether Metal compiles
//!    the name. This is also what proves `flash_decoding.metal` compiles at
//!    all, since the offline toolchain is not installed on this machine.

use crate::kernels::params::{FlashCombineParams, FlashPartialParams};
use crate::kernels::{flash_chunk_count, ChunkTable, FlashDType, FlashKernel, FLASH_HEAD_DIMS};
use crate::metal::{Commands, ResidencySet};
use crate::{call_flash_decoding, Device, Kernels, Source};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, &residency_set).unwrap()
}

fn read_f32(buf: &crate::Buffer, n: usize) -> Vec<f32> {
    let p = buf.contents() as *const f32;
    assert!(!p.is_null(), "buffer has no CPU mapping");
    // SAFETY: shared storage holding at least `n` f32, and the caller has
    // waited for the command buffer.
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn new_f32(device: &Device, values: &[f32]) -> crate::Buffer {
    let buf = device
        .new_buffer(std::mem::size_of_val(values).max(4), SHARED)
        .unwrap();
    let dst = buf.contents() as *mut f32;
    // SAFETY: freshly allocated shared storage of exactly this length.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len()) };
    buf
}

fn new_u32(device: &Device, values: &[u32]) -> crate::Buffer {
    let buf = device
        .new_buffer(std::mem::size_of_val(values).max(4), SHARED)
        .unwrap();
    let dst = buf.contents() as *mut u32;
    // SAFETY: as `new_f32`.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len()) };
    buf
}

fn write_struct<T>(device: &Device, value: T) -> crate::Buffer {
    let buf = device.new_buffer(std::mem::size_of::<T>(), SHARED).unwrap();
    let dst = buf.contents() as *mut T;
    // SAFETY: freshly allocated shared storage sized for exactly one `T`.
    unsafe { dst.write(value) };
    buf
}

/// The scalar reference: one-pass softmax attention in `f64`.
///
/// **Deliberately not the chunked formulation.** A reference that split the same
/// way the kernel does would agree with it for the same wrong reason; this
/// computes the whole thing in one pass at higher precision, so agreement is
/// evidence that the *split and merge* are right rather than that two
/// implementations of the split agree.
///
/// `keys`/`values` are `[n_kv_heads][capacity][head_dim]`, and only the first
/// `n_keys` tokens of each head are live — the reserved-capacity shape §6.2b's
/// pre-allocated cache produces.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    scale: f32,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let mut out = vec![0f32; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let kv = h / gqa;
        let q = &queries[h * head_dim..(h + 1) * head_dim];

        let mut scores = Vec::with_capacity(n_keys);
        for t in 0..n_keys {
            let base = kv * capacity * head_dim + t * head_dim;
            let mut s = 0f64;
            for d in 0..head_dim {
                s += (q[d] as f64) * (keys[base + d] as f64);
            }
            scores.push(s * scale as f64);
        }

        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        for s in &scores {
            denom += (s - m).exp();
        }

        for d in 0..head_dim {
            let mut acc = 0f64;
            for (t, s) in scores.iter().enumerate() {
                let base = kv * capacity * head_dim + t * head_dim;
                acc += (s - m).exp() * values[base + d] as f64;
            }
            out[h * head_dim + d] = (acc / denom) as f32;
        }
    }
    out
}

/// A deterministic pseudo-random fixture. No `rand` dependency and no seed
/// plumbing: the values only have to be varied and reproducible.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// One end-to-end run, returning the kernel's output.
#[allow(clippy::too_many_arguments)]
fn run_flash(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    pages_per_chunk: usize,
    scale: f32,
    table: &ChunkTable,
) -> Vec<f32> {
    let cmds = commands(device);
    let chunk_size = page_size * pages_per_chunk;
    let n_chunks = flash_chunk_count(n_keys, chunk_size);

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &table.entries(n_chunks, pages_per_chunk));
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    // `[b, n_kv, capacity, head_dim]`: index 1 is the head stride (the RESERVED
    // capacity) and index 2 is the token stride. Those being different numbers
    // is what lets a pre-allocated cache be read without a copy (§10.3b).
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        FlashPartialParams::for_step(
            &q_shape,
            &k_shape,
            &k_stride,
            &k_stride,
            page_size,
            pages_per_chunk,
            n_chunks,
            scale,
            1.0,
        ),
    );
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    read_f32(&out, n_q_heads * head_dim)
}

/// **Every declared name resolves against the compiled library.**
///
/// §8.1b. This is also what proves `flash_decoding.metal` compiles at all: the
/// offline Metal toolchain is absent on this machine, so runtime compilation is
/// the only compiler available — which #8.1b notes *inverts the usual ranking*,
/// since the library a test interrogates is the one the GPU will be asked for.
#[test]
fn flash_names_resolve() {
    let device = device();
    let kernels = Kernels::new();
    for name in FlashKernel::all() {
        kernels
            .load_pipeline(&device, Source::FlashDecoding, name.clone())
            .unwrap_or_else(|e| {
                panic!("{name} does not resolve against flash_decoding.metal: {e}")
            });
    }
}

/// **A head dimension the file does not instantiate is refused, not resolved.**
///
/// The inverse of #103's finding, guarded from the other side: there, eight
/// head dimensions were instantiated and six reachable, so two were dead
/// instantiations paid for at every cold compile. Here the registry and the
/// instantiation list are one list, and asking for a dimension outside it is an
/// error rather than a `LoadFunctionError` from inside a forward pass.
#[test]
fn flash_refuses_an_uninstantiated_head_dim() {
    assert!(FlashKernel::partial(FlashDType::F16, 63).is_err());
    assert!(FlashKernel::combine(FlashDType::F16, 512).is_err());
    for hd in FLASH_HEAD_DIMS {
        assert!(FlashKernel::partial(FlashDType::F16, hd).is_ok());
    }
}

/// **Parity against the scalar reference, over several chunk counts.**
///
/// The chunk counts are chosen so that `n_keys` is *not* a multiple of the
/// chunk size in three of five cases: the last chunk is partial, and
/// `n_this_chunk` must walk the live count rather than the reserved one.
/// Merging over a full last chunk folds in uninitialised memory, which §9.1a
/// records as a silent wrong answer that no size check catches — so a fixture
/// whose `n_keys` always divides evenly would not exercise it.
#[test]
fn flash_matches_the_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 8;
    let n_kv_heads = 2; // GQA 4:1, as LFM2 is (§5.2)
    let capacity = 512;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // `n_keys` and the chunk size, chosen so the last chunk is partial in most.
    for &(n_keys, page_size) in &[(1usize, 32usize), (32, 32), (33, 32), (100, 32), (255, 64)] {
        let queries = fill(n_q_heads * head_dim, 1);
        let keys = fill(n_kv_heads * capacity * head_dim, 2);
        let values = fill(n_kv_heads * capacity * head_dim, 3);

        let got = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        let want = reference_attention(
            &queries, &keys, &values, n_q_heads, n_kv_heads, head_dim, capacity, n_keys, scale,
        );

        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w).abs());
        }
        assert!(
            worst < 2e-5,
            "n_keys={n_keys} page_size={page_size}: worst |delta| = {worst:e} against the f64 \
             reference; chunks={}",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **The answer does not depend on how the KV is split.**
///
/// One chunk, two chunks and eleven chunks over the *same* keys must agree to
/// float tolerance — which is the associativity property online softmax exists
/// to provide (§10.4, glossary), and the thing a wrong merge breaks.
///
/// §10.6 is the other half of it: *contiguous is paged with one page, and
/// single-pass attention is FlashDecoding with `n_chunks == 1`*. The
/// `page_size >= n_keys` arm below **is** that degenerate case, so it is tested
/// rather than argued.
#[test]
fn flash_is_invariant_to_the_chunk_count() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 512;
    let n_keys = 200;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 7);
    let keys = fill(n_kv_heads * capacity * head_dim, 8);
    let values = fill(n_kv_heads * capacity * head_dim, 9);

    let single = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        256,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    assert_eq!(
        flash_chunk_count(n_keys, 256),
        1,
        "the degenerate arm must be one chunk"
    );

    for &page_size in &[128usize, 64, 32] {
        let split = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        let mut worst = 0f32;
        for (a, b) in single.iter().zip(split.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst < 2e-5,
            "splitting into {} chunks moved the answer by {worst:e}",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **`k > 1` computes the same answer as `k = 1` at the same chunk size.**
///
/// `chunk_size = k * page_size` (§9.1d). `k = 2` at `page_size = 32` and `k = 1`
/// at `page_size = 64` are the same *computation* unit over different
/// *allocation* units, so they must agree — which is what makes `k` a real
/// parameter rather than a field nothing reads.
///
/// **This is the arm that would fail if `pages_per_chunk` were ignored.** §9.1d
/// warns that a sweep fixing `k = 1` cannot separate a page-size effect from a
/// tile-size one; carrying `k` is what makes that separable, and this test is
/// what says the field is live.
#[test]
fn flash_k_greater_than_one_agrees_with_k_one() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 256;
    let n_keys = 150;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 11);
    let keys = fill(n_kv_heads * capacity * head_dim, 12);
    let values = fill(n_kv_heads * capacity * head_dim, 13);

    // k = 1 at page 64: chunk_size 64.
    let k1 = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        64,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    // k = 2 at page 32: chunk_size 64, from two pages.
    let k2 = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        32,
        2,
        scale,
        &ChunkTable::Contiguous,
    );

    let mut worst = 0f32;
    for (a, b) in k1.iter().zip(k2.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst == 0.0,
        "k=2 x page 32 and k=1 x page 64 are the same chunking and must be bit-identical; \
         worst |delta| = {worst:e}"
    );
}

/// **The chunk table is read by INDEX and not by WALK, shown with a table that
/// is not the identity.**
///
/// This is the test §10.3h asks for and the one a B=1 gate cannot substitute
/// for. With `chunk_table[c] == c` an index and a walk are indistinguishable —
/// #197's probe had to add `page_table[3] = 6` for exactly this reason, and
/// records that *a fixture that does not permute its table is testing the
/// identity function*.
///
/// The construction: two chunks whose **pages are swapped** in the table. Chunk
/// 0 names page 1 and chunk 1 names page 0, so the kernel reads the second half
/// of the KV for chunk 0 and the first half for chunk 1. Attention over a set of
/// keys does not depend on the order the *set* is visited in — softmax is over
/// all of them — so the ANSWER must be unchanged, while a kernel that ignored
/// the table and walked `c * chunk_size` would also produce the unchanged
/// answer. **What separates them is where the partials land**, which is why the
/// assertion is on the partials rather than only on the output.
#[test]
fn permuted_chunk_table_is_read_by_index_not_by_walk() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 2;
    let n_kv_heads = 1;
    let capacity = 128;
    let page_size = 32;
    let n_keys = 64; // exactly two chunks
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 21);
    let keys = fill(n_kv_heads * capacity * head_dim, 22);
    let values = fill(n_kv_heads * capacity * head_dim, 23);

    let n_chunks = flash_chunk_count(n_keys, page_size);
    assert_eq!(n_chunks, 2);

    // The two arms differ ONLY in the table.
    let identity = run_partials(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        scale,
        &ChunkTable::Contiguous,
    );
    let permuted = run_partials(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        scale,
        &ChunkTable::Explicit(vec![1, 0]),
    );

    // Under a permuted table, chunk 0's partial must be what chunk 1's was, and
    // vice versa: the kernel resolved its range THROUGH the table.
    //
    // A kernel that ignored the table and walked `c * chunk_size` would produce
    // `identity` under both, so this comparison is capable of the other answer
    // -- which is what makes it a test rather than a display (§15.1 #1).
    let (id_maxs, perm_maxs) = (&identity.1, &permuted.1);
    assert_ne!(
        id_maxs[0], id_maxs[1],
        "the fixture is degenerate: the two chunks have equal maxima, so a swap is \
         unobservable and this test would pass under a walk"
    );
    for h in 0..n_q_heads {
        assert_eq!(
            perm_maxs[h * 2],
            id_maxs[h * 2 + 1],
            "head {h}: chunk 0 under the permuted table must read what chunk 1 read under the \
             identity -- it did not, so the table was WALKED rather than INDEXED"
        );
        assert_eq!(
            perm_maxs[h * 2 + 1],
            id_maxs[h * 2],
            "head {h}: chunk 1 under the permuted table must read what chunk 0 read under the \
             identity"
        );
    }
}

/// As [`run_flash`], returning `(partials, maxs)` so a test can assert on where
/// a chunk's work landed rather than only on the merged output.
#[allow(clippy::too_many_arguments)]
fn run_partials(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    scale: f32,
    table: &ChunkTable,
) -> (Vec<f32>, Vec<f32>) {
    let cmds = commands(device);
    let n_chunks = flash_chunk_count(n_keys, page_size);

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &table.entries(n_chunks, 1));
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, scale, 1.0,
        ),
    );
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    (
        read_f32(&partials, n_q_heads * n_chunks * head_dim),
        read_f32(&maxs, n_q_heads * n_chunks),
    )
}

/// **The same input gives the same bits, over repeated runs.**
///
/// §2.3.1's run-to-run kind, at the kernel level. It is a *necessary* check and
/// not a sufficient one — §2.3.5a: a kernel with a wrong constant is perfectly
/// deterministic — which is why it sits beside the parity test rather than
/// instead of it.
///
/// What it does catch is the failure mode §10.4 names as the most likely in this
/// design: a merge whose order depends on which chunk finished first. With 11
/// chunks over 8 heads there are 88 threadgroups whose completion order the
/// scheduler chooses, so a completion-ordered merge has ample opportunity to
/// differ between runs.
#[test]
fn flash_is_bit_stable_across_runs() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 8;
    let n_kv_heads = 2;
    let capacity = 4096;
    let n_keys = 2720; // §9.1's "largest ever measured"
    let page_size = 256; // §10.4's proposed size
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 31);
    let keys = fill(n_kv_heads * capacity * head_dim, 32);
    let values = fill(n_kv_heads * capacity * head_dim, 33);

    assert_eq!(flash_chunk_count(n_keys, page_size), 11);

    let first = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    for run in 1..5 {
        let again = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        assert_eq!(
            first,
            again,
            "run {run} differs from run 0 at kv_len={n_keys}, {} chunks -- a merge whose order \
             depends on completion is the first thing to suspect (DESIGN.md 10.4)",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **The token stride is read from the parameter, not assumed to be
/// `head_dim`.**
///
/// This is §9.1d's field — the one `sdpa_vector` could not vary, because its
/// token step is `constexpr int stride = BN * D`. Carrying it as a parameter is
/// what makes a second KV layout a different *number* rather than a different
/// kernel, and therefore what makes the arm #200 lacked buildable.
///
/// **A fixture whose token stride happens to equal `head_dim` cannot show
/// that**, and every other test in this file is such a fixture. Measured
/// rather than assumed: a mutation replacing `k_token_stride` with `D` left all
/// eight of them passing. That is #70's `align_up` lesson in a second quantity
/// (§9.2c) — *a parity test built only from the model's own shapes is blind to
/// a whole defect class*.
///
/// So this fixture pads each token's row: the KV is laid out with a stride of
/// `head_dim + PAD` and only the first `head_dim` elements of each row are the
/// real vector. The padding is filled with a value large enough that reading it
/// as data would move the answer visibly rather than subtly.
#[test]
fn token_stride_is_read_from_the_parameter() {
    let device = device();
    let kernels = Kernels::new();

    const PAD: usize = 8;
    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 128;
    let n_keys = 100;
    let page_size = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let row = head_dim + PAD;

    let queries = fill(n_q_heads * head_dim, 51);
    let tight_k = fill(n_kv_heads * capacity * head_dim, 52);
    let tight_v = fill(n_kv_heads * capacity * head_dim, 53);

    // The same values, re-laid at a padded row stride. Padding is 7.0 rather
    // than 0.0: a zero would be a plausible dot-product contribution and could
    // cancel, where 7.0 across 8 lanes cannot be mistaken for rounding.
    let mut padded_k = vec![7.0f32; n_kv_heads * capacity * row];
    let mut padded_v = vec![7.0f32; n_kv_heads * capacity * row];
    for h in 0..n_kv_heads {
        for t in 0..capacity {
            for d in 0..head_dim {
                padded_k[h * capacity * row + t * row + d] =
                    tight_k[h * capacity * head_dim + t * head_dim + d];
                padded_v[h * capacity * row + t * row + d] =
                    tight_v[h * capacity * head_dim + t * head_dim + d];
            }
        }
    }

    let n_chunks = flash_chunk_count(n_keys, page_size);
    let cmds = commands(&device);
    let q_buf = new_f32(&device, &queries);
    let k_buf = new_f32(&device, &padded_k);
    let v_buf = new_f32(&device, &padded_v);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(&device, &ChunkTable::Contiguous.entries(n_chunks, 1));
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    // Index 2 is `row`, not `head_dim`. **That is the whole fixture.**
    let k_stride = [n_kv_heads * capacity * row, capacity * row, row, 1];

    let pp = write_struct(
        &device,
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, scale, 1.0,
        ),
    );
    let cp = write_struct(&device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            &device,
            &guard,
            &kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    let got = read_f32(&out, n_q_heads * head_dim);

    // The reference reads the TIGHT layout: the padded cache holds the same
    // vectors, so a kernel honouring the stride must agree with it, and one
    // assuming `head_dim` reads the padding and cannot.
    let want = reference_attention(
        &queries, &tight_k, &tight_v, n_q_heads, n_kv_heads, head_dim, capacity, n_keys, scale,
    );
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        worst = worst.max((g - w).abs());
    }
    assert!(
        worst < 2e-5,
        "at a padded token stride ({row} against head_dim {head_dim}) the kernel diverged by \
         {worst:e} -- it is assuming head_dim rather than reading k_token_stride, which is the \
         field 9.1d asks this kernel to carry"
    );
}

/// **The combine walks chunk indices in ASCENDING order, asserted against the
/// order the kernel recorded rather than against its output.**
///
/// This is §10.4's own instruction and the test the rest of this file cannot
/// substitute for. That section records the reversal as *"caught by that
/// assertion and by nothing else"* — the bit-equality tests stayed green under
/// it, because floating-point non-associativity happened not to bite on that
/// fixture.
///
/// **Measured here rather than inherited: reversing both combine loops left all
/// seven of this family's other tests passing**, parity against the f64
/// reference included. So the claim is not that an output comparison is
/// *theoretically* insufficient; it is that on this kernel, at these shapes, it
/// *was* insufficient, and this assertion is what fired.
///
/// §2.3.3 #1 is the rule: every reduction merges in fixed index order, never
/// completion order. `CombineOrder::Completion` exists as a spelling in
/// `scratch.rs` so the rule has something to reject; this is the same
/// discipline applied to the kernel that does the merging.
#[test]
fn combine_walks_chunks_in_ascending_index_order() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 512;
    let n_keys = 300;
    let page_size = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 41);
    let keys = fill(n_kv_heads * capacity * head_dim, 42);
    let values = fill(n_kv_heads * capacity * head_dim, 43);
    let n_chunks = flash_chunk_count(n_keys, page_size);
    assert!(
        n_chunks > 2,
        "a one- or two-chunk fixture cannot show an order"
    );

    let walked = walk_order_of(
        &device, &kernels, &queries, &keys, &values, n_q_heads, n_kv_heads, head_dim, capacity,
        n_keys, page_size, scale,
    );

    let expected: Vec<u32> = (0..n_chunks as u32).collect();
    for h in 0..n_q_heads {
        let got = &walked[h * n_chunks..(h + 1) * n_chunks];
        assert_eq!(
            got,
            &expected[..],
            "head {h} walked {got:?} where 10.4 requires ascending index order {expected:?}"
        );
    }
}

/// As [`run_flash`], returning the chunk indices the combine recorded walking.
#[allow(clippy::too_many_arguments)]
fn walk_order_of(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    scale: f32,
) -> Vec<u32> {
    let cmds = commands(device);
    let n_chunks = flash_chunk_count(n_keys, page_size);

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &ChunkTable::Contiguous.entries(n_chunks, 1));
    // Poisoned, so an unwritten slot is visibly unwritten rather than a
    // plausible 0 -- the same reason 9.1a reads its partials back against a
    // poisoned buffer instead of trusting a zero.
    let walk = new_u32(device, &vec![u32::MAX; n_q_heads * n_chunks]);

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, scale, 1.0,
        ),
    );
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();

    let p = walk.contents() as *const u32;
    // SAFETY: shared storage sized for `n_q_heads * n_chunks` u32, waited.
    let got = unsafe { std::slice::from_raw_parts(p, n_q_heads * n_chunks) }.to_vec();
    assert!(
        !got.contains(&u32::MAX),
        "the combine left a slot unwritten, so the recorded order is incomplete and \
         asserting on it would assert nothing"
    );
    got
}

// The layout check is **not** here, and that is #58's mechanism working.
//
// `every_family_params_layout_matches_metal` iterates `LayoutFamily::ALL`, so
// registering `LayoutFamily::Flash` is what enrols this family — and
// `layout_registry_covers_every_family`'s exhaustive `match` makes a family
// that fails to register an `error[E0004]` rather than a silently unchecked
// one. §11.3i: *a family absent from the call-site list was never checked,
// which looks exactly like a family that passes.* Writing a per-family body
// here would put back the call site that mechanism removed.
