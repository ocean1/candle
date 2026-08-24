// Does per-tap branch DIVERGENCE cost anything here, as distinct from branching?
//
// DESIGN.md 3.3: the expensive case is lanes within a simdgroup taking
// different paths, not a branch as such. Which lanes diverge is decided by
// l_out: tid is linear and l_idx = tid % l_out, so a simdgroup diverges only
// when it straddles an l_out boundary. That makes divergence a function of
// l_out mod 32, which this sweeps directly.
//
// If divergence were the dominant cost, l_out values that make every simdgroup
// straddle a boundary (small l_out) would be far worse per element than l_out
// values where almost none do (large l_out, few boundaries).
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use std::time::Instant;
    let dev = Device::new_metal(0)?;
    let (c, k, pad) = (2048usize, 3usize, 2usize);
    let label = std::env::var("BENCH_LABEL").unwrap_or_default();
    println!("{label}  divergent-simdgroup fraction vs cost per output element");
    println!("{:>7} {:>7} {:>10} {:>14} {:>16}", "l_in", "l_out", "div_frac", "ms/conv(p10)", "ns/output_elem");
    // l_in chosen so l_out lands at, just under, and just over multiples of 32.
    for &l_in in &[30usize, 32, 34, 62, 64, 158, 160, 254, 256, 510, 512, 1022, 1024, 2046, 2048] {
        let l_out = l_in + 2*pad - (k-1);
        // A simdgroup (32 consecutive tid) diverges iff it contains an l_out
        // boundary, i.e. iff some lane wraps l_idx. Fraction ~ min(1, 32/l_out)
        // plus the always-divergent first/last positions of each row.
        let n_tid = c * l_out;
        let n_sg = n_tid.div_ceil(32);
        let mut div = 0usize;
        for sg in 0..n_sg {
            let (lo, hi) = (sg*32, (sg*32+31).min(n_tid-1));
            let mut pats = std::collections::HashSet::new();
            for kk in 0..k {
                for tid in lo..=hi {
                    let l_idx = tid % l_out;
                    let pos = l_idx + kk;
                    pats.insert(if pos < pad { 0u8 } else if pos - pad >= l_in { 2 } else { 1 });
                }
            }
            if pats.len() > 1 { div += 1; }
        }
        let frac = div as f64 / n_sg as f64;

        let nel = c * l_in;
        let xv: Vec<f32> = (0..nel).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let wv: Vec<f32> = (0..c * k).map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0).collect();
        let x = Tensor::from_vec(xv, (1, c, l_in), &dev)?.to_dtype(DType::F16)?;
        let w = Tensor::from_vec(wv, (c, 1, k), &dev)?.to_dtype(DType::F16)?;
        for _ in 0..20 { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
        dev.synchronize()?;
        let mut v = vec![];
        for _ in 0..100 {
            let t = Instant::now();
            for _ in 0..22 { let _ = x.conv1d(&w, pad, 1, 1, c)?; }
            dev.synchronize()?;
            v.push(t.elapsed().as_secs_f64()*1e3/22.0);
        }
        v.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let ms = v[10];
        let ns_per = ms*1e6/(c*l_out) as f64;
        println!("{l_in:>7} {l_out:>7} {:>9.1}% {ms:>13.4} {ns_per:>16.4}", frac*100.0);
    }
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
