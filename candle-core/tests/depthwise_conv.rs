//! The fused depthwise conv1d must agree with the generic grouped path.
//!
//! The fused kernel writes straight to `(b, c, l_out)` and skips im2col, so a
//! layout or padding mistake would show up as plausible-but-wrong numbers
//! rather than a failure. These compare against the path it replaces.

#![cfg(feature = "metal")]

// Fused depthwise conv1d must match the generic grouped path exactly.
use candle_core::{DType, Device, Tensor};

fn reference(x: &Tensor, w: &Tensor, pad: usize, groups: usize) -> candle_core::Result<Tensor> {
    // Force the generic path by running each group separately.
    let xs = x.chunk(groups, 1)?;
    let ws = w.chunk(groups, 0)?;
    let parts = xs
        .iter()
        .zip(&ws)
        .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Tensor::cat(&parts, 1)
}

fn check(dev: &Device, dtype: DType, b: usize, c: usize, l: usize, k: usize, pad: usize) {
    let n = b * c * l;
    let xv: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
    let wv: Vec<f32> = (0..c * k).map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0).collect();

    let x = Tensor::from_vec(xv, (b, c, l), dev).unwrap().to_dtype(dtype).unwrap();
    let w = Tensor::from_vec(wv, (c, 1, k), dev).unwrap().to_dtype(dtype).unwrap();

    let fused = x.conv1d(&w, pad, 1, 1, c).unwrap();
    let refr = reference(&x, &w, pad, c).unwrap();

    assert_eq!(fused.dims(), refr.dims(), "shape mismatch {dtype:?} b{b} c{c} l{l} k{k} pad{pad}");

    let a = fused.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let e = refr.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
    let mut worst = 0f32;
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        let d = (x - y).abs();
        if d > worst { worst = d; }
        assert!(d <= tol, "{dtype:?} b{b} c{c} l{l} k{k} pad{pad}: idx {i} {x} vs {y}");
    }
    println!("  ok {dtype:?} b={b} c={c} l={l} k={k} pad={pad} worst={worst:.2e}");
}

#[test]
fn depthwise_conv1d_matches_generic_path() {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for dtype in [DType::F32, DType::F16] {
        // LFM2's real shape (k=3, pad=2) plus edge cases.
        check(&dev, dtype, 1, 2048, 16, 3, 2);
        check(&dev, dtype, 1, 8, 5, 3, 2);
        check(&dev, dtype, 2, 4, 7, 3, 2);
        check(&dev, dtype, 1, 4, 1, 3, 2);   // single position
        check(&dev, dtype, 1, 3, 9, 1, 0);   // k=1, no padding
        check(&dev, dtype, 1, 5, 6, 5, 0);   // wide kernel, no padding
        check(&dev, dtype, 3, 16, 12, 3, 1);
    }
    println!("all depthwise conv parity checks passed");
}
