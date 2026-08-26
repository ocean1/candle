//! A GPU load generator, to run alongside the determinism probe.
//!
//! The determinism question (lloom issue #5) is specifically about whether
//! results depend on occupancy and scheduling. On an idle machine the probe's
//! dispatches essentially have the GPU to themselves, so scheduling-dependent
//! behaviour has no opportunity to appear. This process exists to take that
//! opportunity away: it runs large matmuls back to back in a separate process,
//! contending for the same GPU.
//!
//! It deliberately does *not* coordinate with the probe. Uncoordinated
//! contention is the point — the probe's kernels should interleave with this
//! one's arbitrarily.
//!
//! ```bash
//! cargo run --release --example metal-load --features metal -- --seconds 600
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, Device, Tensor};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// How long to keep the GPU busy.
    #[arg(long, default_value_t = 600)]
    seconds: u64,

    /// Square matrix dimension per matmul.
    #[arg(long, default_value_t = 2048)]
    dim: usize,

    /// Report achieved throughput every N iterations, as evidence the load is
    /// actually running rather than silently erroring out.
    #[arg(long, default_value_t = 200)]
    report_every: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_metal(0)?;

    let a = Tensor::randn(0f32, 1f32, (args.dim, args.dim), &device)?.to_dtype(DType::F16)?;
    let b = Tensor::randn(0f32, 1f32, (args.dim, args.dim), &device)?.to_dtype(DType::F16)?;

    let start = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(args.seconds);
    let mut iters: usize = 0;

    println!(
        "metal-load: {}x{} f16 matmuls for {}s",
        args.dim, args.dim, args.seconds
    );

    while start.elapsed() < deadline {
        let c = a.matmul(&b)?;
        // Force the result to be realized, so the work cannot be elided and the
        // queue does not simply grow without executing. The reduction is taken
        // in f32 because candle's `to_scalar::<f32>` will not accept an f16
        // tensor.
        let _ = c.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        iters += 1;

        if iters % args.report_every == 0 {
            let secs = start.elapsed().as_secs_f64();
            let gflop = 2.0 * (args.dim as f64).powi(3) * iters as f64 / 1e9;
            println!(
                "metal-load: {iters} matmuls, {secs:.1}s, {:.1} GFLOP/s",
                gflop / secs
            );
        }
    }

    println!("metal-load: done, {iters} matmuls in {:?}", start.elapsed());
    Ok(())
}
