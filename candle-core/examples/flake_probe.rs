// Isolate which side of depthwise_conv1d_matches_generic_path is unstable.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};
    let dev = Device::new_metal(0)?;
    let (b, c, l, k, pad) = (1usize, 2048usize, 16usize, 3usize, 2usize);
    let n = b * c * l;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    let dtype = match std::env::var("PROBE_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        _ => DType::F32,
    };
    let x = Tensor::from_vec(xv, (b, c, l), &dev)?.to_dtype(dtype)?;
    let w = Tensor::from_vec(wv, (c, 1, k), &dev)?.to_dtype(dtype)?;

    let reference = |x: &Tensor, w: &Tensor| -> candle_core::Result<Tensor> {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::cat(&parts, 1)
    };

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

    // Repeat both sides in the SAME process. A stable side gives one digest.
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut fused_d = std::collections::BTreeSet::new();
    let mut ref_d = std::collections::BTreeSet::new();
    for _ in 0..reps {
        fused_d.insert(digest(&x.conv1d(&w, pad, 1, 1, c)?)?);
        ref_d.insert(digest(&reference(&x, &w)?)?);
    }
    println!("dtype={dtype:?} reps={reps}");
    println!(
        "  fused distinct digests: {} -> {:?}",
        fused_d.len(),
        fused_d
    );
    println!("  generic distinct digests: {} -> {:?}", ref_d.len(), ref_d);
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("metal feature required");
}
