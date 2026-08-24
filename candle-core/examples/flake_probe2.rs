// Narrow the generic-path nondeterminism to a specific op.
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
    let dtype = DType::F32;
    let x = Tensor::from_vec(xv, (b, c, l), &dev)?.to_dtype(dtype)?;
    let w = Tensor::from_vec(wv, (c, 1, k), &dev)?.to_dtype(dtype)?;

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

    // A: chunk + cat only, no conv at all. Isolates chunk/cat from conv.
    let mut a = std::collections::BTreeSet::new();
    for _ in 0..reps {
        let xs = x.chunk(c, 1)?;
        a.insert(digest(&Tensor::cat(&xs, 1)?)?);
    }
    println!("A chunk+cat (no conv):        {} distinct", a.len());

    // B: per-chunk conv1d, CONTIGUOUS-ified weight chunk. Tests the kernel_c bug.
    let mut bset = std::collections::BTreeSet::new();
    for _ in 0..reps {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.contiguous()?.conv1d(&wi.contiguous()?, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        bset.insert(digest(&Tensor::cat(&parts, 1)?)?);
    }
    println!("B per-chunk conv, contiguous: {} distinct", bset.len());

    // C: exactly what the test does (non-contiguous views).
    let mut cset = std::collections::BTreeSet::new();
    for _ in 0..reps {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        cset.insert(digest(&Tensor::cat(&parts, 1)?)?);
    }
    println!("C test's exact reference:     {} distinct", cset.len());

    // D: is the weight chunk actually non-contiguous?
    let ws = w.chunk(c, 0)?;
    println!(
        "   weight chunk[1] contiguous? {}  layout={:?}",
        ws[1].is_contiguous(),
        ws[1].layout().stride()
    );
    let xs = x.chunk(c, 1)?;
    println!(
        "   input  chunk[1] contiguous? {}  layout={:?}",
        xs[1].is_contiguous(),
        xs[1].layout().stride()
    );
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
