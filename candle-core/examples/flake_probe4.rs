// Confirm mechanism: does forcing a sync between chunked convs remove it?
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
    let x = Tensor::from_vec(xv, (b, c, l), &dev)?;
    let w = Tensor::from_vec(wv, (c, 1, k), &dev)?;
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

    // Variant 1: no sync (the test's shape).
    let mut s1 = std::collections::BTreeSet::new();
    for _ in 0..reps {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        s1.insert(digest(&Tensor::cat(&parts, 1)?)?);
    }
    println!("no sync between chunks:       {} distinct", s1.len());

    // Variant 2: synchronize the device after every chunk conv. If the cause is
    // CPU run-ahead recycling pool buffers still in flight, this removes it.
    let mut s2 = std::collections::BTreeSet::new();
    for _ in 0..reps {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let mut parts = Vec::with_capacity(c);
        for (xi, wi) in xs.iter().zip(&ws) {
            let p = xi.conv1d(wi, pad, 1, 1, 1)?;
            dev.synchronize()?;
            parts.push(p);
        }
        s2.insert(digest(&Tensor::cat(&parts, 1)?)?);
    }
    println!("device.synchronize per chunk: {} distinct", s2.len());
    println!("  (if variant 2 == 1 and variant 1 > 1, the cause is in-flight buffer reuse)");
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
