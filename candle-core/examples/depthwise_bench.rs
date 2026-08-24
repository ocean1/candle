// Isolated timing for the depthwise conv1d kernel at LFM2's prefill shapes.
//
// The whole-generation A/B cannot resolve this: LFM2 calls the depthwise kernel
// 22 times per forward pass, at prefill only, against a decode loop that
// dominates wall time and never touches it. This times the kernel directly.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use std::time::Instant;

    let dev = Device::new_metal(0)?;
    let dtype = match std::env::var("BENCH_DTYPE").as_deref() {
        Ok("f32") => DType::F32,
        _ => DType::F16,
    };
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let label = std::env::var("BENCH_LABEL").unwrap_or_default();

    // (c, l_in, k, pad) — LFM2 is c=2048, k=3, pad=2 (conv_L_cache - 1).
    // l_in sweeps prompt length: 34 is the probe's prompt, 736 the DESIGN.md
    // reference prefill, 4096 a long-context chunk.
    for &(c, l, k, pad) in &[
        (2048usize, 34usize, 3usize, 2usize),
        (2048, 128, 3, 2),
        (2048, 736, 3, 2),
        (2048, 4096, 3, 2),
    ] {
        let n = c * l;
        let xv: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let wv: Vec<f32> = (0..c * k).map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0).collect();
        let x = Tensor::from_vec(xv, (1, c, l), &dev)?.to_dtype(dtype)?;
        let w = Tensor::from_vec(wv, (c, 1, k), &dev)?.to_dtype(dtype)?;

        // Warm the pipeline cache and let the GPU reach steady state.
        for _ in 0..20 { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
        dev.synchronize()?;

        // Time in batches of 22, matching one LFM2 forward pass's conv layers,
        // so the number is directly comparable to a per-forward-pass cost.
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            for _ in 0..22 { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
            dev.synchronize()?;
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = samples[samples.len() / 2];
        let p10 = samples[samples.len() / 10];
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let sd = (samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
        println!(
            "{label:8} {dtype:?} c={c} l_in={l} k={k} pad={pad}  22-conv batch: \
             median {med:.3} ms  p10 {p10:.3}  mean {mean:.3}  sd {sd:.3}  (n={iters})"
        );
    }
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
