// Does the pool hand back a buffer whose last writer is still in flight, and if
// so, can the bind-time per-buffer wait find that writer's fence?
//
// Fork PR #10 waits per buffer at bind time by looking the bound buffer up in
// prev_ce_outputs. Entries go in at end_encoding and come out in the command
// buffer's completion handler. So the wait is correct exactly when a recycled
// buffer's last writer is still registered at the instant it is re-bound.
//
// POOL_HITS counts every time find_available_buffer returned a pooled buffer.
// HITS_WITH_PENDING_WRITER counts the subset whose pointer was still in
// prev_ce_outputs at that moment -- i.e. reuse of a buffer with GPU work
// outstanding, which is exactly the case the wait has to cover.
#[cfg(feature = "metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::metal_backend::device::reuse_probe;
    use candle_core::{DType, Device, Tensor};

    reuse_probe::enable();
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

    println!("reps={reps}");
    println!(
        "{:>6}  {:>8}  {:>10}  {:>24}  {:>7}",
        "c", "distinct", "pool-hits", "hits-with-pending-writer", "pct"
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
        reuse_probe::reset();
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
        let (hits, pending, read_open, written_open) = reuse_probe::snapshot();
        let pct = if hits == 0 {
            0.0
        } else {
            100.0 * pending as f64 / hits as f64
        };
        println!(
            "{c:>6}  {:>8}  {hits:>10}  {pending:>24}  {pct:>6.1}%  open-read={read_open} open-write={written_open}",
            set.len()
        );
    }
    Ok(())
}
#[cfg(not(feature = "metal"))]
fn main() {}
