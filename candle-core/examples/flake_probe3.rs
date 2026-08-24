// Does the generic-path nondeterminism scale with the number of chunked convs?
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
    let reps = 8;
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
        println!(
            "c={c:5}  ({c} chunked convs)  distinct digests over {reps} reps: {}",
            set.len()
        );
    }
    // Also: a SINGLE non-chunked grouped conv repeated, for comparison.
    let c = 2048usize;
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
        set.insert(digest(&x.conv1d(&w, pad, 1, 1, c)?)?);
    }
    println!(
        "fused depthwise c=2048: distinct digests over {reps} reps: {}",
        set.len()
    );
    // And a single groups=1 conv over 2048 channels (one im2col+matmul, not 2048).
    let w1 = Tensor::from_vec(vec![0.5f32; c * c * k], (c, c, k), &dev)?;
    let mut set = std::collections::BTreeSet::new();
    for _ in 0..4 {
        set.insert(digest(&x.conv1d(&w1, pad, 1, 1, 1)?)?);
    }
    println!(
        "single groups=1 conv:   distinct digests over 4 reps: {}",
        set.len()
    );
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
