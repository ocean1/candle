// Is the harness the cost? Time the per-conv Rust/dispatch overhead against
// the GPU work, by comparing a 22-conv batch to a 1-conv batch.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use std::time::Instant;
    let dev = Device::new_metal(0)?;
    let (c, k, pad) = (2048usize, 3usize, 2usize);
    for &l in &[34usize, 736, 4096] {
        let n = c * l;
        let xv: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let wv: Vec<f32> = (0..c * k).map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0).collect();
        let x = Tensor::from_vec(xv, (1, c, l), &dev)?.to_dtype(DType::F16)?;
        let w = Tensor::from_vec(wv, (c, 1, k), &dev)?.to_dtype(DType::F16)?;
        for _ in 0..20 { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
        dev.synchronize()?;
        let mut res = vec![];
        for reps in [1usize, 22, 44] {
            let mut v = vec![];
            for _ in 0..100 {
                let t = Instant::now();
                for _ in 0..reps { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
                dev.synchronize()?;
                v.push(t.elapsed().as_secs_f64()*1e3);
            }
            v.sort_by(|a,b| a.partial_cmp(b).unwrap());
            res.push((reps, v[10]));
        }
        let (_, t1) = res[0]; let (_, t22) = res[1]; let (_, t44) = res[2];
        // Linear fit through the 22 and 44 points gives the marginal per-conv
        // cost; the intercept is fixed per-batch overhead (mostly synchronize).
        let marginal = (t44 - t22) / 22.0;
        println!("l_in={l:5}  p10: 1 conv {t1:.3} ms | 22 convs {t22:.3} ms | 44 convs {t44:.3} ms");
        println!("          marginal per-conv (from 22->44) = {marginal:.4} ms; \
                  fixed per-batch overhead ~= {:.3} ms", t22 - 22.0*marginal);
    }
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
