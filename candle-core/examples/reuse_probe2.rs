// Which chunk is corrupted, and what does the corrupt value look like?
//
// flake_probe3 hashes the concatenated result, so it says "differs" but not
// where or how. This runs the same chunked grouped conv repeatedly, keeps the
// per-chunk outputs, and diffs each repetition against a CPU reference. The
// shape of the wrong values is the discriminator:
//
//   - garbage / other-chunk data  -> the buffer was overwritten by a later op
//   - stale (a previous rep's value) -> the read happened before the write
//   - zeros                        -> read of a never-written recycled buffer
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::metal_backend::device::reuse_probe;
    use candle_core::{DType, Device, Tensor};

    reuse_probe::enable();
    let dev = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let c: usize = std::env::var("PROBE_C")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let (b, l, k, pad) = (1usize, 16usize, 3usize, 2usize);
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let n = b * c * l;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    // CPU reference: bit-stable, and the parity target CONTRIBUTING.md 3.1 wants.
    let xc = Tensor::from_vec(xv.clone(), (b, c, l), &cpu)?;
    let wc = Tensor::from_vec(wv.clone(), (c, 1, k), &cpu)?;
    let refv = {
        let xs = xc.chunk(c, 1)?;
        let ws = wc.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::cat(&parts, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    };

    let x = Tensor::from_vec(xv, (b, c, l), &dev)?;
    let w = Tensor::from_vec(wv, (c, 1, k), &dev)?;

    let l_out = l + 2 * pad - (k - 1);
    println!("c={c} l_out={l_out} reps={reps}  (per-chunk block = {l_out} values)");

    for rep in 0..reps {
        let xs = x.chunk(c, 1)?;
        let ws = w.chunk(c, 0)?;
        let parts = xs
            .iter()
            .zip(&ws)
            .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let got = Tensor::cat(&parts, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let mut bad_chunks: Vec<usize> = Vec::new();
        let mut first: Option<(usize, f32, f32)> = None;
        for (i, (g, r)) in got.iter().zip(&refv).enumerate() {
            if (g - r).abs() > 1e-4 {
                let ch = i / l_out;
                if bad_chunks.last() != Some(&ch) {
                    bad_chunks.push(ch);
                }
                if first.is_none() {
                    first = Some((i, *g, *r));
                }
            }
        }
        // Is a wrong value equal to some OTHER chunk's correct value? That would
        // mean the recycled buffer still held a different convolution's output.
        let mut aliased = 0usize;
        for (i, (g, r)) in got.iter().zip(&refv).enumerate() {
            if (g - r).abs() > 1e-4 {
                let off = i % l_out;
                for other in 0..c {
                    if other != i / l_out && (refv[other * l_out + off] - g).abs() <= 1e-6 {
                        aliased += 1;
                        break;
                    }
                }
            }
        }
        let zeros = got
            .iter()
            .zip(&refv)
            .filter(|(g, r)| (*g - *r).abs() > 1e-4 && g.abs() < 1e-12)
            .count();
        match first {
            None => println!("rep {rep}: MATCHES cpu reference"),
            Some((i, g, r)) => println!(
                "rep {rep}: {} bad chunks of {c}; first bad idx {i} (chunk {}, off {}): got {g} want {r}; \
                 of the wrong values {aliased} equal ANOTHER chunk's correct value, {zeros} are zero",
                bad_chunks.len(),
                i / l_out,
                i % l_out
            ),
        }
    }
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
