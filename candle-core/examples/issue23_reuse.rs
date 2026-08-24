// Issue #23 acceptance probe: is a buffer ever handed out with GPU work
// outstanding on it?
//
// This is the mechanistic counterpart to `issue23_flake`. The flake rate is a
// timing-sensitive stochastic quantity -- measured on the unfixed baseline it
// ranged from 2/60 to 16/60 across samples of the same binary -- so "it went
// down" is not by itself evidence (`CONTRIBUTING.md` §1.3). This measures the
// mechanism instead, and the mechanism is not stochastic: either the pool hands
// back buffers the GPU has not finished with, or it does not.
//
// Issue #19 measured 17406 of 17409 pool hits (100.0 % at c>=1024) returning a
// buffer whose last writer had not completed. The criterion is that this is 0.
//
// The counter is taken at the instant of the hit: the pool remembers which
// epoch each free buffer was waiting on, and `acquire` asks whether that epoch
// has completed *now*. The reuse rate is reported alongside, because a "fix"
// that simply stopped reusing anything would also read 0 here and would be
// worthless -- the hit rate has to stay high for the number to mean anything.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor};

    let dev = Device::new_metal(0)?;
    let Device::Metal(metal) = &dev else {
        return Ok(());
    };
    let (b, l, k, pad) = (1usize, 16usize, 3usize, 2usize);

    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    println!("reps={reps}");
    println!(
        "{:>6}  {:>10}  {:>9}  {:>24}  {:>16}",
        "c", "pool-hits", "hit-rate", "hits-with-pending-writer", "hits-no-epoch"
    );

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

        metal.reset_pool_counters();
        for _ in 0..reps {
            let xs = x.chunk(c, 1)?;
            let ws = w.chunk(c, 0)?;
            let parts = xs
                .iter()
                .zip(&ws)
                .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let _ = Tensor::cat(&parts, 1)?;
        }

        let (s, p) = metal.pool_counters();
        let hits = s.hits + p.hits;
        let lookups = s.lookups + p.lookups;
        let rate = if lookups == 0 {
            0.0
        } else {
            hits as f64 / lookups as f64 * 100.0
        };
        println!(
            "{c:>6}  {hits:>10}  {rate:>8.1}%  {:>24}  {:>16}",
            s.probe_hits_with_pending_writer + p.probe_hits_with_pending_writer,
            s.probe_hits_without_epoch + p.probe_hits_without_epoch,
        );
    }
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {}
