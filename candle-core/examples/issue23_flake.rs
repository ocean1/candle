// Issue #23 acceptance probe: does the grouped-conv corruption still happen?
//
// The workload is issue #19's (measurements/issue-19-grouped-conv-reuse.md §2):
// b=1, l=16, k=3, pad=2, `c` chunked groups=1 convolutions concatenated, run
// PROBE_REPS times per channel count. "Unstable" means an invocation produced
// more than one distinct digest for c=2048 across its reps -- the same
// operation giving different answers, which is the aliasing signature.
//
// The ascending sweep over `c` matters and is not decoration: it is what
// issue #19's reuse_probe did, and the defect is timing-sensitive, so the work
// done before reaching c=2048 is part of the conditions that reproduce it. A
// probe that starts cold at c=2048 shows ~0/30 on the *unfixed* baseline and
// therefore cannot tell whether anything was fixed.
//
// Release is clean on both branches, so debug is the correct instrument here
// (DESIGN.md §2.3.8a); nothing in this file is a performance number.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};

    let dev = Device::new_metal(0)?;
    let (b, l, k, pad) = (1usize, 16usize, 3usize, 2usize);

    let digest = |t: &Tensor| -> candle_core::Result<String> {
        let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut h: u64 = 1469598103934665603;
        for x in &v {
            for byte in x.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        Ok(format!("{h:016x}"))
    };

    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    for c in [8usize, 64, 256, 1024, 2048] {
        let n = b * c * l;
        let xv: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let wv: Vec<f32> = (0..c * k)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let x = Tensor::from_vec(xv, (b, c, l), &dev)?;
        let w = Tensor::from_vec(wv, (c, 1, k), &dev)?;

        let mut set = std::collections::BTreeSet::new();
        for _ in 0..reps {
            let xs = x.chunk(c, 1)?;
            let ws = w.chunk(c, 0)?;
            let parts = xs
                .iter()
                .zip(&ws)
                .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
                .collect::<candle_core::Result<Vec<_>>>()?;
            set.insert(digest(&Tensor::cat(&parts, 1)?)?);
        }
        // Machine-readable, so 30 invocations can be counted by a loop.
        println!("{} {}", c, set.len());
    }
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {}
