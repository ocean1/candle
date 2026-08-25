use super::*;
use crate::metal::{Commands, ResidencySet};
use core::ffi::c_void;
use half::{bf16, f16};
use rand::prelude::SliceRandom;
use rand::{rng, Rng};
use std::sync::Arc;
use std::thread;

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    Commands::new(queue, &residency_set).unwrap()
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

#[test]
fn pipeline_cache_distinguishes_sources() {
    let device = device();
    let kernels = Kernels::new();

    // Prime the cache with a name that is not present in the binary library.
    kernels
        .load_pipeline(&device, Source::Unary, "cos_f32")
        .unwrap();
    assert!(matches!(
        kernels.load_pipeline(&device, Source::Binary, "cos_f32"),
        Err(MetalKernelError::LoadFunctionError(_))
    ));
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: 0,
    };
    let output = new_buffer(&device, v);
    call_unary_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        v.len(),
        input,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

fn run_binary<T: Clone, S: ToString>(x: &[T], y: &[T], name: S) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device
        .new_buffer(std::mem::size_of_val(x), options)
        .unwrap();
    call_binary_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        x.len(),
        BufferOffset::zero_offset(&left),
        BufferOffset::zero_offset(&right),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, x.len())
}

fn run_strided<T: Clone>(
    v: &[T],
    kernel: unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: offset,
    };
    let output_b = new_buffer(&device, v);
    let output = BufferOffset {
        buffer: &output_b,
        offset_in_bytes: 0,
    };
    let kernels = Kernels::new();
    call_unary_strided(
        &device, &encoder, &kernels, kernel, shape, input, strides, output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output_b, v.len())
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

    let v = vec![1.0f32; 10_000];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.954, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::HALF);
    assert_eq!(approx_f16(results, 3), expected);
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.955, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn silu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.27, 0.0, 0.73, 1.76, 2.86, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn silu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.269, 0.0, 0.731, 1.762, 2.858, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, "badd_f32");
    let expected: Vec<_> = left
        .iter()
        .zip(right.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    macro_rules! binary_op {
        ($opname:ident, $dtype:ident, $opexpr:expr) => {{
            let results = run_binary(
                &lhs,
                &rhs,
                concat!(stringify!($opname), "_", stringify!($dtype)),
            );
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&$dtype, &$dtype)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }
    binary_op!(badd, bf16, |x, y| x + y);
    binary_op!(bsub, bf16, |x, y| x - y);
    binary_op!(bmul, bf16, |x, y| x * y);
    binary_op!(bdiv, bf16, |x, y| x / y);
    binary_op!(bminimum, bf16, |x: bf16, y| x.min(y));
    binary_op!(bmaximum, bf16, |x: bf16, y| x.max(y));
}

fn run_cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let options = RESOURCE_OPTIONS;
    let size = v.len() * std::mem::size_of::<U>();
    let output = device.new_buffer(size, options).unwrap();

    call_cast_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        v.len(),
        BufferOffset::zero_offset(&input),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn cast_f32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f32 -> f16
    let results: Vec<half::f16> = run_cast(&v_f32, "cast_f32_f16");
    assert_eq!(results, v_f16);

    // f32 -> bf16
    let results: Vec<bf16> = run_cast(&v_f32, "cast_f32_bf16");
    assert_eq!(results, v_bf16);

    // f32 -> u32
    let results: Vec<u32> = run_cast(&v_f32, "cast_f32_u32");
    assert_eq!(results, v_u32);

    // f32 -> u8
    let results: Vec<u8> = run_cast(&v_f32, "cast_f32_u8");
    assert_eq!(results, v_u8);

    // f32 -> i64
    let results: Vec<i64> = run_cast(&v_f32, "cast_f32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_f16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f16 -> f32
    let results: Vec<f32> = run_cast(&v_f16, "cast_f16_f32");
    assert_eq!(results, v_f32);

    // f16 -> bf16
    let results: Vec<bf16> = run_cast(&v_f16, "cast_f16_bf16");
    assert_eq!(results, v_bf16);

    // f16 -> u32
    let results: Vec<u32> = run_cast(&v_f16, "cast_f16_u32");
    assert_eq!(results, v_u32);

    // f16 -> u8
    let results: Vec<u8> = run_cast(&v_f16, "cast_f16_u8");
    assert_eq!(results, v_u8);

    // f16 -> i64
    let results: Vec<i64> = run_cast(&v_f16, "cast_f16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_bf16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // bf16 -> f32
    let results: Vec<f32> = run_cast(&v_bf16, "cast_bf16_f32");
    assert_eq!(results, v_f32);

    // bf16 -> f16
    let results: Vec<f16> = run_cast(&v_bf16, "cast_bf16_f16");
    assert_eq!(results, v_f16);

    // bf16 -> u32
    let results: Vec<u32> = run_cast(&v_bf16, "cast_bf16_u32");
    assert_eq!(results, v_u32);

    // bf16 -> u8
    let results: Vec<u8> = run_cast(&v_bf16, "cast_bf16_u8");
    assert_eq!(results, v_u8);

    // bf16 -> i64
    let results: Vec<i64> = run_cast(&v_bf16, "cast_bf16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u32 -> f32
    let results: Vec<f32> = run_cast(&v_u32, "cast_u32_f32");
    assert_eq!(results, v_f32);

    // u32 -> f16
    let results: Vec<f16> = run_cast(&v_u32, "cast_u32_f16");
    assert_eq!(results, v_f16);

    // u32 -> bf16
    let results: Vec<bf16> = run_cast(&v_u32, "cast_u32_bf16");
    assert_eq!(results, v_bf16);

    // u32 -> u8
    let results: Vec<u8> = run_cast(&v_u32, "cast_u32_u8");
    assert_eq!(results, v_u8);

    // u32 -> i64
    let results: Vec<i64> = run_cast(&v_u32, "cast_u32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u8() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u8 -> f32
    let results: Vec<f32> = run_cast(&v_u8, "cast_u8_f32");
    assert_eq!(results, v_f32);

    // u8 -> f16
    let results: Vec<f16> = run_cast(&v_u8, "cast_u8_f16");
    assert_eq!(results, v_f16);

    // u8 -> bf16
    let results: Vec<bf16> = run_cast(&v_u8, "cast_u8_bf16");
    assert_eq!(results, v_bf16);

    // u8 -> u32
    let results: Vec<u32> = run_cast(&v_u8, "cast_u8_u32");
    assert_eq!(results, v_u32);

    // u8 -> i64
    let results: Vec<i64> = run_cast(&v_u8, "cast_u8_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_i64() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // i64 -> f32
    let results: Vec<f32> = run_cast(&v_i64, "cast_i64_f32");
    assert_eq!(results, v_f32);

    // i64 -> f16
    let results: Vec<f16> = run_cast(&v_i64, "cast_i64_f16");
    assert_eq!(results, v_f16);

    // i64 -> bf16
    let results: Vec<bf16> = run_cast(&v_i64, "cast_i64_bf16");
    assert_eq!(results, v_bf16);

    // i64 -> u32
    let results: Vec<u32> = run_cast(&v_i64, "cast_i64_u32");
    assert_eq!(results, v_u32);

    // i64 -> u8
    let results: Vec<u8> = run_cast(&v_i64, "cast_i64_u8");
    assert_eq!(results, v_u8);
}

fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let size = v.len();

    call_affine(
        &device,
        &encoder,
        &kernels,
        "affine_f32",
        size_of::<T>(),
        size,
        BufferOffset::zero_offset(&input),
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

fn run_affine_strided<T: Clone>(
    v: &[T],
    shape: &[usize],
    strides: &[usize],
    mul: f64,
    add: f64,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    call_affine_strided(
        &device,
        &encoder,
        &kernels,
        "affine_f32_strided",
        shape,
        BufferOffset::zero_offset(&input),
        strides,
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let len: usize = shape.iter().product();
    read_to_vec(&output, len)
}

#[test]
fn affine() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

    let input = [1.0f32; 40_000];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6; 40_000]);
}

#[test]
fn affine_strided() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let shape = [4];
    let strides = [2];
    let result = run_affine_strided(&input, &shape, &strides, mul, add);
    // 1 on 2
    assert_eq!(result, vec![2.6, 5.6, 8.6, 11.6]);
}

fn run_mlx_sort<T: Clone>(v: &[T], ncols: usize) -> Vec<u32> {
    let nrows = v.len() / ncols;
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let indexes = vec![0u32; v.len()];
    let output = new_buffer(&device, &indexes);

    call_mlx_arg_sort(
        &device,
        &encoder,
        &kernels,
        DType::F32,
        nrows,
        ncols,
        BufferOffset::zero_offset(&input),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn mlx_sort() {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let input: Vec<_> = (0..8).map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [0, 1, 2, 3, 0, 1, 2, 3]);
    let input: Vec<_> = (0..8).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [3, 2, 1, 0, 3, 2, 1, 0]);
    let input: Vec<_> = (0..1000).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 200);
    let out: Vec<_> = (0..200).rev().collect();
    assert_eq!(&result[..200], out);
    assert_eq!(&result[200..400], out);
    assert_eq!(&result[400..600], out);
    assert_eq!(&result[600..800], out);
    assert_eq!(&result[800..], out);

    // Multi-block test
    let ncols = 16000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(299792458);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let input: Vec<f32> = (0..ncols * 16).map(|_| normal.sample(&mut rng)).collect();
    let result = run_mlx_sort(&input, ncols);
    for start in 0..16 {
        let slice = &input[start * ncols..(start + 1) * ncols];
        let result = &result[start * ncols..(start + 1) * ncols];
        let mut perm: Vec<usize> = (0..ncols).collect();
        perm.sort_by(|i1, i2| slice[*i1].total_cmp(&slice[*i2]));
        let perm: Vec<_> = perm.into_iter().map(|v| v as u32).collect();
        assert_eq!(perm, result);
    }
}

#[test]
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let stride = [1, 2];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn index_select_strided() {
    let embedding = (0..16).map(|x| x as f32).collect::<Vec<_>>();
    let shape = [2, 2];
    let stride = [2, 4];
    let ids = [0u32];
    let dim = 0;
    let result = run_index_select_strided(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![0.0, 4.0]);
}

#[test]
fn index_select_f16() {
    let embedding: Vec<_> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f16");
    assert_eq!(
        approx_f16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u8_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_u8_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_i64_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0i64, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_i64_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_dim1() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 7.0, 8.0f32, 7.0, 9.0, 10.0, 9.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        true,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

fn run_index_select_strided<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        false,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let results = run(&v, unary::contiguous::cos::HALF);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

fn run_reduce<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(out_length * core::mem::size_of::<U>(), options)
        .unwrap();
    let shape = vec![in_length];
    match call_reduce_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        &shape,
        out_length,
        BufferOffset::zero_offset(&input),
        &output,
    ) {
        Ok(_) => {}
        Err(e) => {
            println!("{e}");
            panic!();
        }
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    call_last_softmax(
        &device,
        &encoder,
        &kernels,
        name,
        v.len(),
        last_dim,
        &input,
        0,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

const fn create_array<const N: usize>() -> [f32; N] {
    let mut array: [f32; N] = [0.0; N];
    let mut i = 1;
    while i <= N {
        array[i - 1] = i as f32;
        i += 1;
    }
    array
}

const fn correct_sum<const N: usize, const D: usize>() -> [f32; D] {
    let mut sum = 0;
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        sum += i;
        i += 1;
        if i > j * N / D {
            results[j - 1] = sum as f32;
            j += 1;
            sum = 0;
        }
    }
    results
}

const fn correct_max<const N: usize, const D: usize>() -> [f32; D] {
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        i += 1;
        if i > j * (N / D) {
            results[j - 1] = (i - 1) as f32;
            j += 1;
        }
    }
    results
}

fn correct_argmax<const N: usize, const D: usize>(arr: [f32; N]) -> [u32; D] {
    let mut max = 0.0;
    let mut max_index: u32 = 0;
    let mut results: [u32; D] = [0; D];
    let mut i = 0;
    let mut j = 1;
    while i <= N {
        if i >= (j * N / D) {
            results[j - 1] = max_index;
            max = 0.0;
            max_index = 0;
            j += 1;
        }
        if i == N {
            break;
        }
        if arr[i] > max {
            max = arr[i];
            max_index = i as u32;
        }
        i += 1;
    }
    results
}

fn reduce_sum_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_sum_f32");
    assert_eq!(approx(results, 4), correct_sum::<N, D>());
}

fn reduce_max_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_max_f32");
    assert_eq!(approx(results, 4), correct_max::<N, D>());
}

fn reduce_argmax_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results: Vec<u32> = run_reduce(&v, N, D, "fast_argmax_f32");
    assert_eq!(results, correct_argmax::<N, D>(v));
}

#[test]
fn reduce_sum1() {
    reduce_sum_case::<9, 1>();
    reduce_sum_case::<6, 1>();
    reduce_sum_case::<10, 1>();
    reduce_sum_case::<64, 1>();
    reduce_sum_case::<128, 1>();
    reduce_sum_case::<256, 1>();
    reduce_sum_case::<512, 1>();
    reduce_sum_case::<1024, 1>();
    reduce_sum_case::<2048, 1>();
    reduce_sum_case::<4096, 1>();
}

#[test]
fn reduce_sum2() {
    reduce_sum_case::<6, 2>();
    reduce_sum_case::<10, 2>();
    reduce_sum_case::<64, 2>();
    reduce_sum_case::<128, 2>();
    reduce_sum_case::<256, 2>();
    reduce_sum_case::<512, 2>();
    reduce_sum_case::<1024, 2>();
    reduce_sum_case::<2048, 2>();
    reduce_sum_case::<4096, 2>();
}

#[test]
fn reduce_max() {
    reduce_max_case::<6, 1>();
    reduce_max_case::<9, 1>();
    reduce_max_case::<10, 1>();
    reduce_max_case::<64, 1>();
    reduce_max_case::<128, 1>();
    reduce_max_case::<256, 1>();
    reduce_max_case::<512, 1>();
    reduce_max_case::<1024, 1>();
    reduce_max_case::<2048, 1>();
    reduce_max_case::<4096, 1>();

    reduce_max_case::<6, 2>();
    reduce_max_case::<10, 2>();
    reduce_max_case::<64, 2>();
    reduce_max_case::<128, 2>();
    reduce_max_case::<256, 2>();
    reduce_max_case::<512, 2>();
    reduce_max_case::<1024, 2>();
    reduce_max_case::<2048, 2>();
    reduce_max_case::<4096, 2>();

    reduce_max_case::<6, 3>();
    reduce_max_case::<10, 3>();
    reduce_max_case::<64, 3>();
    reduce_max_case::<128, 3>();
    reduce_max_case::<256, 3>();
    reduce_max_case::<512, 3>();
    reduce_max_case::<1024, 3>();
    reduce_max_case::<2048, 3>();
    reduce_max_case::<4096, 3>();
}

#[test]
fn reduce_argmax() {
    reduce_argmax_case::<6, 1>();
    reduce_argmax_case::<9, 1>();
    reduce_argmax_case::<10, 1>();
    reduce_argmax_case::<64, 1>();
    reduce_argmax_case::<128, 1>();
    reduce_argmax_case::<256, 1>();
    reduce_argmax_case::<512, 1>();
    reduce_argmax_case::<1024, 1>();
    reduce_argmax_case::<2048, 1>();
}

#[test]
fn reduce_argmax2() {
    reduce_argmax_case::<6, 2>();
    reduce_argmax_case::<10, 2>();
    reduce_argmax_case::<64, 2>();
    reduce_argmax_case::<128, 2>();
    reduce_argmax_case::<256, 2>();
    reduce_argmax_case::<512, 2>();
    reduce_argmax_case::<1024, 2>();
    reduce_argmax_case::<2048, 2>();
    reduce_argmax_case::<4096, 2>();
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let last_dim = 4096;
    let n = 200;
    let mut v = vec![0.0; n * last_dim];
    for i in 0..n {
        v[i * last_dim] = 20.0;
    }
    let results = run_softmax(&v, last_dim, "softmax_f32");
    let results = approx(results, 4);
    assert_eq!(
        results.iter().map(|&s| s.round() as usize).sum::<usize>(),
        n
    );
    assert_eq!(results[0], 1.0);
    assert_eq!(results[1], 0.0);
    assert_eq!(results[last_dim], 1.0);
    assert_eq!(results[2 * last_dim], 1.0);

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f16");
    assert_eq!(
        approx_f16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2332, 0.6338]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_bf16");
    assert_eq!(
        approx_bf16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0859, 0.2324, 0.6328]
    );
}

#[allow(clippy::too_many_arguments)]
fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let length = cond.len();
    let cond = device
        .new_buffer_with_data(
            cond.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(cond),
            options,
        )
        .unwrap();
    let left = device
        .new_buffer_with_data(
            left_true.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();
    let right = device
        .new_buffer_with_data(
            right_false.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();

    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    let cond = BufferOffset {
        buffer: &cond,
        offset_in_bytes: cond_offset,
    };
    let left = BufferOffset {
        buffer: &left,
        offset_in_bytes: left_offset,
    };
    let right = BufferOffset {
        buffer: &right,
        offset_in_bytes: cond_offset,
    };
    call_where_cond(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        shape,
        cond,
        &cond_stride,
        true,
        left,
        &left_stride,
        true,
        right,
        &cond_stride,
        true,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}
#[test]
fn where_cond_u32_f32() {
    let shape = vec![6];
    let cond = vec![0u32, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u32_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}

#[allow(clippy::too_many_arguments)]
fn run_mlx_gemm<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    lhs_offset: usize,
    rhs: &[T],
    rhs_stride: &[usize],
    rhs_offset: usize,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let lhs = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(lhs),
            options,
        )
        .unwrap();
    let rhs = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(rhs),
            options,
        )
        .unwrap();
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    call_mlx_gemm(
        &device,
        &encoder,
        &kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        &lhs,
        rhs_stride,
        rhs_offset,
        &rhs,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn mlx_gemm() {
    let (b, m, n, k) = (1, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
    );

    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![
            20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0, 488.0,
            518.0, 548.0, 578.0
        ]
    );

    // OFFSET
    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    // Manually set batch_size=1 and offset 12 elements * 4 the number of bytes for f32
    let results = run_mlx_gemm(
        GemmDType::F32,
        (1, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        12 * 4,
    );
    assert_eq!(
        approx(results, 4),
        vec![56.0, 59.0, 62.0, 65.0, 200.0, 212.0, 224.0, 236.0]
    );

    // bgemm sanity test
    {
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<bf16> = (0..b * m * k).map(|f| bf16::from_f32(f as f32)).collect();
        let rhs: Vec<bf16> = (0..b * n * k).map(|f| bf16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::BF16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_bf16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }

    {
        // hgemm sanity test
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::F16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_f16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }
}

fn run_random<T: Clone>(name: &'static str, seed: u64, length: usize, a: f32, b: f32) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();

    let seed = device
        .new_buffer_with_data(
            &seed as *const u64 as *const core::ffi::c_void,
            std::mem::size_of::<u64>(),
            options,
        )
        .unwrap();

    if name.starts_with("rand_uniform") {
        call_random_uniform(
            &device, &encoder, &kernels, name, a, b, length, &seed, &output,
        )
        .unwrap();
    } else {
        call_random_normal(
            &device, &encoder, &kernels, name, a, b, length, &seed, &output,
        )
        .unwrap();
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn random() {
    fn calc_mean(data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>();
        let count = data.len();
        assert!(count > 0);
        sum / count as f32
    }

    fn calc_stddev(data: &[f32]) -> f32 {
        let mean = calc_mean(data);
        let count = data.len();
        assert!(count > 0);

        let variance = data
            .iter()
            .map(|value| {
                let diff = mean - *value;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;

        variance.sqrt()
    }

    let shape = [1024, 10];

    let length = shape.iter().product::<usize>();
    let seed = 299792458u64;

    let min = -30.0;
    let max = 30.0;
    let mean = 100.0;
    let stddev = 50.0;

    macro_rules! validate_random {
        ($type:ty) => {
            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_uniform_", stringify!($type)),
                seed,
                length,
                min,
                max,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            results.iter().for_each(|v| {
                assert!(*v >= min && *v <= max);
            });
            assert!(calc_mean(&results) > -1.0 && calc_mean(&results) < 1.0);

            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_normal_", stringify!($type)),
                seed,
                length,
                mean,
                stddev,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            assert!((calc_mean(&results) - mean).abs() < mean / 10.0);
            assert!((calc_stddev(&results) - stddev).abs() < stddev / 10.0);
        };
    }

    validate_random!(f32);
    validate_random!(f16);
    validate_random!(bf16);
}

fn run_scatter_add<T: Clone, I: Clone + std::fmt::Debug>(
    input: &[T],
    ids: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);
    let ids_buffer = new_buffer(&device, ids);
    let output = device
        .new_buffer(std::mem::size_of_val(input), options)
        .unwrap();
    call_scatter(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        BufferOffset::zero_offset(&output),
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, input.len())
}

#[test]
fn scatter_add() {
    let ids_u8 = [0u8, 0, 1, 0, 2, 2, 3, 3];
    let ids_u32 = [0u32, 0, 1, 0, 2, 2, 3, 3];
    let ids_i64 = [0i64, 0, 1, 0, 2, 2, 3, 3];

    let input_f32 = [5.0f32, 1.0, 7.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let input_f16 = input_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let input_bf16 = input_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim1_f32 = vec![8.0, 7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let output_dim1_f16 = output_dim1_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim1_bf16 = output_dim1_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim2_f32 = vec![5.0, 3.0, 7.0, 0.0, 3.0, 2.0, 1.0, 3.0];
    let output_dim2_f16 = output_dim2_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim2_bf16 = output_dim2_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    for (shape, output_f32, output_f16, output_bf16) in [
        (vec![8], output_dim1_f32, output_dim1_f16, output_dim1_bf16),
        (
            vec![4, 2],
            output_dim2_f32,
            output_dim2_f16,
            output_dim2_bf16,
        ),
    ] {
        for results in [
            run_scatter_add(&input_f32, &ids_u8, &shape, 0, "sa_u8_f32"),
            run_scatter_add(&input_f32, &ids_u32, &shape, 0, "sa_u32_f32"),
            run_scatter_add(&input_f32, &ids_i64, &shape, 0, "sa_i64_f32"),
        ] {
            assert_eq!(results, output_f32);
        }
        for results in [
            run_scatter_add(&input_f16, &ids_u8, &shape, 0, "sa_u8_f16"),
            run_scatter_add(&input_f16, &ids_u32, &shape, 0, "sa_u32_f16"),
            run_scatter_add(&input_f16, &ids_i64, &shape, 0, "sa_i64_f16"),
        ] {
            assert_eq!(results, output_f16);
        }
        for results in [
            run_scatter_add(&input_bf16, &ids_u8, &shape, 0, "sa_u8_bf16"),
            run_scatter_add(&input_bf16, &ids_u32, &shape, 0, "sa_u32_bf16"),
            run_scatter_add(&input_bf16, &ids_i64, &shape, 0, "sa_i64_bf16"),
        ] {
            assert_eq!(results, output_bf16);
        }
    }
}

fn run_index_add<T: Clone, I: Clone + std::fmt::Debug>(
    left: &[T],
    right: &[T],
    indices: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input_buffer = new_buffer(&device, right);
    let output = new_buffer(&device, left);
    let indices_buffer = new_buffer(&device, indices);
    call_index_add(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&indices_buffer),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, left.len())
}

#[test]
fn index_add() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let indices = vec![0u32, 1, 0, 1, 0, 1];
    let shape = vec![6];

    // u32, f32
    {
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, f16
    {
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, bf16
    {
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f32
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, bf16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f32
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, bf16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

fn run_pool2d<T: Clone>(
    v: &[T],
    (w_k, h_k): (usize, usize),
    (w_stride, h_stride): (usize, usize),
    shape: &[usize],
    strides: &[usize],
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out_w = (shape[2] - w_k) / w_stride + 1;
    let out_h = (shape[3] - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();
    call_pool2d(
        &device, &encoder, &kernels, name, shape, strides, out_w, out_h, w_k, h_k, w_stride,
        h_stride, &input, &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn max_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 7.0, 13.0, 15.0];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f32",
    );
    let expected = vec![
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<f16> = (0..16).map(|v| f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| f16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<bf16> = (0..16).map(|v| bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_bf16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| bf16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u8",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u32",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    input_stride: &[usize],
    kernel: &[T],
    kernel_shape: &[usize],
    kernel_stride: &[usize],
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let c_out = kernel_shape[1];
    let k_size = kernel_shape[2];
    let b_size = input_shape[0];
    let l_in = input_shape[2];
    let l_out = (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1;
    let dst_el = c_out * l_out * b_size;

    let input = new_buffer(&device, input);
    let kernel = new_buffer(&device, kernel);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();

    call_conv_transpose1d(
        &device,
        &encoder,
        &kernels,
        name,
        dilation,
        stride,
        padding,
        out_padding,
        c_out,
        l_out,
        b_size,
        input_shape,
        input_stride,
        kernel_shape,
        kernel_stride,
        &input,
        0,
        &kernel,
        0,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn conv_transpose1d_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel = vec![1.0f32, 2.0, 3.0, 4.0];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f32",
    );

    let expected = vec![1., 4., 10., 20., 25., 24., 16.];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_f16() {
    let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_bf16() {
    let input: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_bf16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u8() {
    let input: Vec<u8> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u8> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u8",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u32() {
    let input: Vec<u32> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u32> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u32",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn const_fill() {
    fn constant_fill<T: Clone + EncoderParam>(name: &'static str, len: usize, value: T) -> Vec<T> {
        let dev = device();
        let kernels = Kernels::new();
        let commands = commands(&dev);
        let encoder = commands.command_encoder().unwrap();
        let buffer = dev
            .new_buffer(len * std::mem::size_of::<T>(), RESOURCE_OPTIONS)
            .unwrap();
        call_const_fill(&dev, &encoder, &kernels, name, len, &buffer, value).unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        read_to_vec::<T>(&buffer, len)
    }
    fn test<T: Clone + Copy + EncoderParam + PartialEq + std::fmt::Debug, F: FnOnce(f32) -> T>(
        name: &'static str,
        f: F,
    ) {
        let len = rand::rng().random_range(2..16) * rand::rng().random_range(4..16);
        let value = rand::rng().random_range(1. ..19.);
        let value = f(value);
        let v = constant_fill::<T>(name, len, value);
        assert_eq!(v, vec![value; len])
    }
    test::<u8, _>("fill_u8", |v| v as u8);
    test::<u32, _>("fill_u32", |v| v as u32);
    test::<i64, _>("fill_i64", |v| v as i64);
    test::<f16, _>("fill_f16", f16::from_f32);
    test::<bf16, _>("fill_bf16", bf16::from_f32);
    test::<f32, _>("fill_f32", |v| v);
}

#[test]
fn commands_creation_and_encoder() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Commands::new(queue, &residency_set).unwrap();

    let encoder = commands.command_encoder().unwrap();
    drop(encoder);
}

#[test]
fn commands_concurrent_acquisition() {
    std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "2");

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Arc::new(Commands::new(queue, &residency_set).unwrap());

    let mut handles = vec![];

    for _ in 0..16 {
        let c = Arc::clone(&commands);
        handles.push(thread::spawn(move || {
            let encoder = c.command_encoder().unwrap();
            drop(encoder);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    commands.wait_until_completed().unwrap();
}

#[test]
fn residency_set_batch_insert_remove() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = set.raw() else {
        // Residency sets are unsupported on this device/OS; the set no-ops.
        return;
    };

    let bufs: Vec<Buffer> = (0..3).map(|i| new_buffer(&device, &[i as f32])).collect();
    let base = raw.allocationCount();

    set.insert_batch(&bufs);
    assert_eq!(raw.allocationCount(), base + bufs.len());
    set.remove_batch(&bufs);
    assert_eq!(raw.allocationCount(), base);

    // Empty batches are valid and leave the set untouched.
    set.insert_batch(std::iter::empty());
    set.remove_batch(std::iter::empty());
    assert_eq!(raw.allocationCount(), base);
}

/// Every name [`ConvKernel`] declares must exist in the compiled `conv.metal`
/// library.
///
/// This is the check that makes the table in `conv_names.rs` load-bearing
/// rather than decorative. The two sides — the `[[host_name]]` strings that
/// `conv.metal`'s macros expand to, and the Rust strings callers pass to
/// `load_pipeline` — were previously hand-synced with nothing comparing them,
/// so a rename or a dtype added to one side only surfaced as a runtime
/// `LoadFunctionError` in the middle of a forward pass.
///
/// `conv.metal` is compiled from source at runtime, so the library this test
/// queries is the same one a real dispatch resolves against: the oracle is the
/// actual compiled metallib, not a second copy of the list.
#[test]
fn conv_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ConvKernel::ALL {
        for (suffix, name) in family.variants() {
            kernels
                .load_pipeline(&device, Source::Conv, name)
                .unwrap_or_else(|e| {
                    panic!(
                        "conv.metal has no kernel named {name:?} \
                         (declared by ConvKernel::{} for {suffix:?}): {e:?}",
                        family.stem(),
                    )
                });
            checked += 1;
        }
    }

    // Guards against the table being emptied or a family silently dropping all
    // its variants — an all-green run over zero names would otherwise pass.
    // 5 dtypes each for im2col1d, im2col, col2im1d, conv_transpose1d,
    // upsample_nearest2d, upsample_bilinear2d, max_pool2d and avg_pool2d, plus
    // 3 each for the float-only conv1d_depthwise and conv_transpose2d, and 3
    // each for the three k_size-specialized depthwise families (k = 2, 3, 4).
    assert_eq!(
        checked, 55,
        "expected 55 declared conv variants, found {checked}"
    );
}

/// Each declared name must be its family's stem, then its dtype suffix, then
/// the family's tail, and each suffix must appear once per family.
///
/// The names are stored verbatim so they can be grepped against `conv.metal`,
/// which means a row could pair one family's suffix with another family's name
/// and still resolve — `conv_names_resolve` would pass while
/// [`ConvKernel::name`] handed callers the wrong kernel. Checking the spelling
/// against the stem closes that, and the duplicate check catches a copy-pasted
/// row that shadows the one below it.
///
/// The tail carries a compile-tier axis for families that have one (`_k3` on
/// the `k_size`-specialized depthwise variants). It is checked here rather than
/// exempted, so those names are held to the same full spelling rule as the
/// rest — a `_k3` row that resolved to a `_k2` kernel is exactly the
/// wrong-kernel case this test exists to catch, and it would be silent.
#[test]
fn conv_names_match_their_stem_and_suffix() {
    for family in ConvKernel::ALL {
        let mut seen: Vec<&str> = Vec::new();
        for (suffix, name) in family.variants() {
            assert_eq!(
                name,
                format!("{}_{suffix}{}", family.stem(), family.tail()),
                "ConvKernel::{}{} declares {name:?} for {suffix:?}",
                family.stem(),
                family.tail(),
            );
            assert!(
                !seen.contains(&suffix),
                "ConvKernel::{} declares {suffix:?} twice",
                family.stem(),
            );
            seen.push(suffix);
        }
        assert!(
            !seen.is_empty(),
            "ConvKernel::{} declares no variants",
            family.stem(),
        );
    }
}

/// A dtype a family does not declare must not produce a name.
///
/// `i64` and `f64` are instantiated nowhere in `conv.metal`, and the float-only
/// families reject the integer suffixes. Returning `None` is what keeps an
/// unsupported dtype from reaching `load_pipeline` as a string that will fail
/// to resolve — the caller reports it against its own dtype enum instead.
#[test]
fn conv_undeclared_dtype_has_no_name() {
    for family in ConvKernel::ALL {
        for suffix in ["i64", "f64", "i32", "not_a_dtype"] {
            assert_eq!(
                family.name(suffix),
                None,
                "ConvKernel::{} named {suffix:?}, which conv.metal does not instantiate",
                family.stem(),
            );
        }
    }

    // The float-only families specifically: `conv.metal` declares no integer
    // instantiation for either, and both are on the LFM2 decode path.
    assert_eq!(ConvKernel::CONV1D_DEPTHWISE.name("u8"), None);
    assert_eq!(ConvKernel::CONV1D_DEPTHWISE.name("u32"), None);
    assert_eq!(ConvKernel::CONV_TRANSPOSE2D.name("u8"), None);
    assert_eq!(ConvKernel::CONV_TRANSPOSE2D.name("u32"), None);

    // And the positive case, so the assertions above are not vacuous.
    assert_eq!(
        ConvKernel::CONV1D_DEPTHWISE.name("f16"),
        Some("conv1d_depthwise_f16")
    );
}

/// Every name [`ReduceKernel`] declares must exist in the compiled
/// `reduce.metal` library.
///
/// The `reduce.metal` counterpart of `conv_names_resolve`, and it earned its
/// place immediately: converting the file's `impl_*` macros to template
/// instantiations introduced `init_reduce` rows whose `#op` stringized to the
/// operator *type* (`Sum`), emitting `fast_Sum_f32` where every caller asks for
/// `fast_sum_f32`. That compiles and links; all 48 reduction variants were
/// simply absent from the library. Without this test the first symptom would
/// have been a `LoadFunctionError` inside an LFM2 forward pass.
#[test]
fn reduce_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ReduceKernel::ALL {
        for (suffix, name) in family.variants() {
            kernels
                .load_pipeline(&device, Source::Reduce, name)
                .unwrap_or_else(|e| {
                    panic!(
                        "reduce.metal has no kernel named {name:?} \
                         (declared by ReduceKernel::{}{} for {suffix:?}): {e:?}",
                        family.stem(),
                        family.tail(),
                    )
                });
            checked += 1;
        }
    }

    // Guards against the table being emptied or a family silently dropping all
    // its variants — an all-green run over zero names would otherwise pass.
    // 6 dtypes each for sum, mul, min, max, argmin and argmax in both their
    // contiguous and strided forms (12 families), plus 3 float dtypes each for
    // softmax, rmsnorm, layernorm, rope, rope_i and rope_thd.
    assert_eq!(
        checked, 90,
        "expected 90 declared reduce variants, found {checked}"
    );
}

/// Each declared name must be its family's stem, then its dtype suffix, then
/// the family's tail, and each suffix must appear once per family.
///
/// Resolution alone cannot catch a row that names a *different* real kernel.
/// That matters more here than it did for conv, because the strided and
/// contiguous forms of a reduction differ by a suffix and take different
/// argument lists: a `_strided` row pointing at the contiguous kernel resolves
/// perfectly well and then reads a `strides` argument that was never bound.
#[test]
fn reduce_names_match_their_stem_and_suffix() {
    for family in ReduceKernel::ALL {
        let mut seen: Vec<&str> = Vec::new();
        for (suffix, name) in family.variants() {
            assert_eq!(
                name,
                format!("{}_{suffix}{}", family.stem(), family.tail()),
                "ReduceKernel::{}{} declares {name:?} for {suffix:?}",
                family.stem(),
                family.tail(),
            );
            assert!(
                !seen.contains(&suffix),
                "ReduceKernel::{}{} declares {suffix:?} twice",
                family.stem(),
                family.tail(),
            );
            seen.push(suffix);
        }
        assert!(
            !seen.is_empty(),
            "ReduceKernel::{}{} declares no variants",
            family.stem(),
            family.tail(),
        );
    }
}

/// A dtype a family does not declare must not produce a name.
///
/// Softmax, the norms and RoPE are float-only in `reduce.metal`, so the integer
/// suffixes must be refused here rather than reaching `load_pipeline`.
#[test]
fn reduce_undeclared_dtype_has_no_name() {
    for family in [
        ReduceKernel::SOFTMAX,
        ReduceKernel::RMSNORM,
        ReduceKernel::LAYERNORM,
        ReduceKernel::ROPE,
        ReduceKernel::ROPE_I,
        ReduceKernel::ROPE_THD,
    ] {
        for suffix in ["u8", "u32", "i64", "f64", "not_a_dtype"] {
            assert_eq!(
                family.name(suffix),
                None,
                "ReduceKernel::{} named {suffix:?}, which reduce.metal does not instantiate",
                family.stem(),
            );
        }
    }

    // f64 is instantiated nowhere in reduce.metal, including for the families
    // that do carry the integer dtypes.
    for family in ReduceKernel::ALL {
        assert_eq!(family.name("f64"), None);
    }

    // And the positive cases, so the assertions above are not vacuous.
    assert_eq!(ReduceKernel::SUM.name("f16"), Some("fast_sum_f16"));
    assert_eq!(
        ReduceKernel::SUM_STRIDED.name("f16"),
        Some("fast_sum_f16_strided")
    );
}

// ---------------------------------------------------------------------------
// Packed parameters (issue #38)
//
// `reduce.metal` carries every kernel twice: the classical form binding each
// scalar with `setBytes`, and a `_packed` form taking one `device const
// Params*`. Only the second is expressible in an ICB command, which has no
// `setBytes` in any form (`DESIGN.md` §3.7b).
//
// Three things have to hold, and they fail in different ways:
//
//  * the two sides agree on **layout** — a field at the wrong offset is a
//    plausible wrong answer, not a crash (`DESIGN.md` §3.5, §15.1)
//  * every `_packed` name **resolves** — a missing instantiation compiles and
//    links, and #26 shipped exactly that defect for 48 variants
//  * the two forms compute **bit-identical** output — which is what proves the
//    body really is shared rather than merely similar

/// The device's own view of every packed struct's layout, against Rust's.
///
/// Dispatches `reduce_params_layout`, which writes `sizeof` and each field's
/// `offsetof` *as the compiled kernel sees them*. Comparing those against
/// `size_of`/`offset_of!` on the `#[repr(C)]` mirrors checks the two sides
/// against each other, where a `static_assert` on either side alone would only
/// prove that side is self-consistent.
///
/// This is the check that catches the hazards #38 names — a vector type
/// over-aligning, a `bool` sized differently — for any struct added later.
#[test]
fn reduce_params_layout_matches_metal() {
    use crate::kernels::params::{expected_layout, LAYOUT_KERNEL, LAYOUT_SLOTS};

    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let out = device
        .new_buffer(LAYOUT_SLOTS * core::mem::size_of::<u32>(), RESOURCE_OPTIONS)
        .unwrap();

    let pipeline = kernels
        .load_pipeline(&device, Source::Reduce, LAYOUT_KERNEL)
        .unwrap();
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_output_buffer(0, Some(&out), 0);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let device_side: Vec<u32> = read_to_vec(&out, LAYOUT_SLOTS);
    let mut disagreements = Vec::new();
    for (slot, (what, rust)) in expected_layout().into_iter().enumerate() {
        if device_side[slot] != rust {
            disagreements.push(format!(
                "  {what}: metal {} vs rust {rust}",
                device_side[slot]
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "packed parameter layout differs between reduce.metal and params.rs.\n\
         A field at the wrong offset is silent corruption, not a crash:\n{}",
        disagreements.join("\n")
    );
}

/// The layout check must be able to fail.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test, and this one guards a defect that is otherwise invisible.
///
/// The mutation is applied to a **copy of `reduce.metal`** compiled at runtime,
/// so the real source is untouched. It **swaps `ReduceParams.src_numel` and
/// `ReduceParams.num_dims`** -- two adjacent fields of the same type. That case
/// is chosen deliberately over inserting or removing a field, because it is the
/// one nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `reduce.metal` still
///   passes;
/// * both fields are `uint`, so the brace initializer still type-checks --
///   inserting a field of a different type is rejected by C++11 narrowing
///   before it ever reaches a GPU, which is a real layer of defence but not
///   this one;
/// * the kernel runs and produces plausible numbers from the wrong values.
///
/// It is the same shape as the defect `DESIGN.md` §8.1c warns about for
/// `im2col`'s eight consecutive `constant size_t` parameters: reordering two
/// same-typed arguments is silent.
#[test]
fn reduce_params_layout_check_detects_a_moved_field() {
    use crate::kernels::params::{expected_layout, LAYOUT_KERNEL, LAYOUT_SLOTS};

    let original = "struct ReduceParams {\n    uint src_numel;\n    uint num_dims;\n    uint el_per_block;\n};";
    let swapped = "struct ReduceParams {\n    uint num_dims;\n    uint src_numel;\n    uint el_per_block;\n};";
    assert!(
        crate::source::REDUCE.contains(original),
        "the struct this test mutates has been reworded; update the mutation"
    );
    let mutant = crate::source::REDUCE.replace(original, swapped);

    let device = device();
    let library = device.new_library_with_source(&mutant, None).unwrap();
    let function = library.get_function(LAYOUT_KERNEL, None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out = device
        .new_buffer(LAYOUT_SLOTS * core::mem::size_of::<u32>(), RESOURCE_OPTIONS)
        .unwrap();
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_output_buffer(0, Some(&out), 0);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let device_side: Vec<u32> = read_to_vec(&out, LAYOUT_SLOTS);
    let disagreements: Vec<String> = expected_layout()
        .into_iter()
        .enumerate()
        .filter(|(slot, (_, rust))| device_side[*slot] != *rust)
        .map(|(slot, (what, rust))| format!("{what}: metal {} vs rust {rust}", device_side[slot]))
        .collect();

    // Both swapped fields must be reported, and the struct's size must not be
    // -- which is exactly why `sizeof` alone is not a sufficient check.
    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("ReduceParams.src_numel")),
        "swapping src_numel must be reported; got {disagreements:?}"
    );
    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("ReduceParams.num_dims")),
        "swapping num_dims must be reported; got {disagreements:?}"
    );
    assert!(
        !disagreements
            .iter()
            .any(|d| d.starts_with("sizeof(ReduceParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {disagreements:?}"
    );
}

/// Every classical name has a `_packed` counterpart in the compiled library.
///
/// The same argument as `reduce_names_resolve`, applied to the new axis: an
/// absent instantiation is not a compile error on either side, and the first
/// symptom would be a `LoadFunctionError` inside a forward pass. #26 shipped
/// precisely that for 48 variants, caught only by loading against the library.
#[test]
fn packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ReduceKernel::ALL {
        for (suffix, name) in family.variants() {
            let packed = crate::kernels::params::packed_name(name);
            kernels
                .load_pipeline(&device, Source::Reduce, packed.clone())
                .unwrap_or_else(|e| {
                    panic!(
                        "reduce.metal has no kernel named {packed:?} \
                         (the packed form of {name:?}, ReduceKernel::{}{} for {suffix:?}): {e:?}",
                        family.stem(),
                        family.tail(),
                    )
                });
            checked += 1;
        }
    }
    assert_eq!(
        checked, 90,
        "expected a packed counterpart for all 90 declared variants, found {checked}"
    );
}

/// Run a reduction both ways and return `(split, packed)`.
fn reduce_both_ways<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> (Vec<U>, Vec<U>) {
    let device = device();
    let kernels = Kernels::new();
    let shape = vec![in_length];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, v);
        let output = device
            .new_buffer(out_length * core::mem::size_of::<U>(), RESOURCE_OPTIONS)
            .unwrap();
        call_reduce_contiguous_with(
            &device,
            &encoder,
            &kernels,
            name,
            &shape,
            out_length,
            BufferOffset::zero_offset(&input),
            &output,
            style,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<U>(&output, out_length));
    }
    let packed = outs.pop().unwrap();
    let split = outs.pop().unwrap();
    (split, packed)
}

/// `fast_sum_f16` — 22 of the 674 dispatches in a decode token — must give the
/// same bits either way.
///
/// Bit-identical rather than approximately equal, deliberately. The two
/// variants are instantiated from one body, so anything but an exact match
/// means the binding style changed what the kernel computed, which is the whole
/// question this issue asks. `DESIGN.md` §2.3 makes the same argument for the
/// engine as a whole.
#[test]
fn packed_matches_split_for_sum_f16() {
    let v: Vec<f16> = (0..1024)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let (split, packed) = reduce_both_ways::<f16, f16>(&v, 1024, 8, "fast_sum_f16");
    assert_eq!(
        split.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        packed.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "fast_sum_f16 and fast_sum_f16_packed disagree"
    );
}

#[test]
fn packed_matches_split_for_sum_f32() {
    let v: Vec<f32> = (0..2048)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let (split, packed) = reduce_both_ways::<f32, f32>(&v, 2048, 16, "fast_sum_f32");
    assert_eq!(
        split.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        packed.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "fast_sum_f32 and fast_sum_f32_packed disagree"
    );
}

/// `argmax` exercises the `indexed<T>` accumulator and the arg-reduce entry
/// points, which are a separate pair of wrappers from the plain reductions.
#[test]
fn packed_matches_split_for_argmax_f32() {
    let v: Vec<f32> = (0..512)
        .map(|i| ((i * 61 % 97) as f32 - 48.0) / 7.0)
        .collect();
    let (split, packed) = reduce_both_ways::<f32, u32>(&v, 512, 4, "fast_argmax_f32");
    assert_eq!(
        split, packed,
        "fast_argmax_f32 and its packed form disagree"
    );
}

/// `softmax_f32` — 8 dispatches per decode token, one per attention layer.
#[test]
fn packed_matches_split_for_softmax_f32() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f32> = (0..1024)
        .map(|i| ((i * 29 % 83) as f32 - 41.0) / 13.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        call_last_softmax_with(
            &device,
            &encoder,
            &kernels,
            "softmax_f32",
            v.len(),
            128,
            &input,
            0,
            &output,
            style,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, v.len()));
    }
    assert_eq!(
        outs[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        outs[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "softmax_f32 and softmax_f32_packed disagree"
    );
}

/// `rmsnorm_f16` — 77 of the 674 dispatches in a decode token, the single
/// largest contributor from this file, and the kernel `DESIGN.md` §3.7b names
/// as the clearest case of a dispatch an ICB cannot express today.
#[test]
fn packed_matches_split_for_rmsnorm_f16() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f16> = (0..2048)
        .map(|i| f16::from_f32(((i * 53 % 89) as f32 - 44.0) / 17.0))
        .collect();
    let alpha: Vec<f16> = (0..256)
        .map(|i| f16::from_f32(1.0 + (i % 7) as f32 / 32.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let alpha_buf = new_buffer(&device, &alpha);
        let output = new_buffer(&device, &v);
        call_rms_norm_with(
            &device,
            &encoder,
            &kernels,
            "rmsnorm_f16",
            v.len(),
            256,
            1e-5,
            &input,
            0,
            &alpha_buf,
            0,
            &output,
            style,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, v.len()));
    }
    assert_eq!(
        outs[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        outs[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "rmsnorm_f16 and rmsnorm_f16_packed disagree"
    );
}

/// `rope_f16` — 16 dispatches per decode token, and the only family here whose
/// scalars are `size_t`. That makes it the one that would break first if the
/// Rust mirror had been narrowed to `u32`, so it is worth its own arm rather
/// than being taken as covered by the `uint` families.
#[test]
fn packed_matches_split_for_rope_f16() {
    let device = device();
    let kernels = Kernels::new();
    let (bh, td, d) = (8usize, 64usize, 64usize);
    let n = bh * td;
    let v: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 41 % 79) as f32 - 39.0) / 11.0))
        .collect();
    let cs: Vec<f16> = (0..(td * d / 2))
        .map(|i| f16::from_f32((i % 13) as f32 / 13.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let cos = new_buffer(&device, &cs);
        let sin = new_buffer(&device, &cs);
        let output = new_buffer(&device, &v);
        call_rope_with(
            &device, &encoder, &kernels, "rope_f16", bh, td, d, 0, &input, 0, &cos, 0, &sin, 0,
            &output, style,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, n));
    }
    assert_eq!(
        outs[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        outs[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "rope_f16 and rope_f16_packed disagree"
    );
}

/// A strided reduction, so the `dims` *and* `strides` arrays both take the
/// promoted-to-a-buffer path rather than `setBytes`.
///
/// This is the case the packed form does not fully absorb: an array's length
/// comes from the tensor's layout, so it cannot be a struct field. It stays a
/// separate binding, which an ICB can express — the constraint is `setBytes`,
/// not buffer count.
#[test]
fn packed_matches_split_for_sum_f32_strided() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f32> = (0..1024)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let shape = vec![4usize, 8, 32];
    let strides = vec![256usize, 32, 1];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = device
            .new_buffer(32 * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        call_reduce_strided_with(
            &device,
            &encoder,
            &kernels,
            "fast_sum_f32_strided",
            &shape,
            &strides,
            32,
            BufferOffset::zero_offset(&input),
            &output,
            style,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, 32));
    }
    assert_eq!(
        outs[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        outs[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "fast_sum_f32_strided and its packed form disagree"
    );
}

// ---------------------------------------------------------------------------
// gemv carries both binding styles (issue #41)
//
// `gemv` is 183 of the 674 dispatches per decode token -- the second-largest
// single contributor after `copy2d_f16` -- and unlike the elementwise families
// these are the matmuls, where the weight traffic happens (`DESIGN.md` §13.4a).
// So these tests check two distinct things: that the two styles agree bit for
// bit, and that the struct's layout agrees across the language boundary.
// ---------------------------------------------------------------------------

/// Run one GEMV in a chosen binding style.
///
/// Goes through `call_mlx_gemv_with` rather than reimplementing the tile
/// selection, so both arms pick the same `[[host_name]]` stem and differ only
/// in the `_packed` suffix and how the scalars are bound.
#[allow(clippy::too_many_arguments)]
fn run_mlx_gemv_style<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    rhs: &[T],
    rhs_stride: &[usize],
    style: ParamStyle,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let lhs_buf = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(lhs),
            options,
        )
        .unwrap();
    let rhs_buf = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(rhs),
            options,
        )
        .unwrap();
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();

    call_mlx_gemv_with(
        &device,
        &encoder,
        &kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        0,
        &lhs_buf,
        rhs_stride,
        0,
        &rhs_buf,
        &output,
        style,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

/// The two binding styles compute the same bits.
///
/// Bit-identical rather than approximately equal, deliberately: both entry
/// points are instantiated from one body (`gemv_body`), so anything but an
/// exact match means the binding style changed what the kernel computed, which
/// is the question issue #41 asks.
///
/// The shapes cover both kernels and both tile-selection branches that LFM2
/// decode reaches: `n == 1` takes `gemv` and `m == 1` takes `gemv_t`, and the
/// `in_vec_size`/`out_vec_size` ratios pick different `(bm, bn, sm, sn)` rows.
#[test]
fn packed_matches_split_for_gemv() {
    // n == 1 -> gemv (mat[M,K] x vec[K]); K=2048 is LFM2's hidden size.
    for (b, m, n, k) in [
        (1usize, 2048usize, 1usize, 2048usize),
        (1, 512, 1, 64),
        (2, 128, 1, 256),
    ] {
        let lhs: Vec<f32> = (0..b * m * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let rhs: Vec<f32> = (0..b * n * k)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 24.0)
            .collect();
        let split = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, 1, n],
            ParamStyle::Split,
        );
        let packed = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, 1, n],
            ParamStyle::Packed,
        );
        assert_eq!(
            split.iter().map(|x: &f32| x.to_bits()).collect::<Vec<_>>(),
            packed.iter().map(|x: &f32| x.to_bits()).collect::<Vec<_>>(),
            "gemv and its packed form disagree at (b,m,n,k)=({b},{m},{n},{k})"
        );
    }
}

/// As above, for the transposed kernel (`m == 1` -> `gemv_t`).
///
/// Kept separate so a failure names which of the two kernels broke; they have
/// different bodies and different threadgroup-reduction paths.
#[test]
fn packed_matches_split_for_gemv_t() {
    for (b, m, n, k) in [
        (1usize, 1usize, 2048usize, 2048usize),
        (1, 1, 512, 128),
        (2, 1, 64, 256),
    ] {
        let lhs: Vec<f32> = (0..b * m * k)
            .map(|i| ((i * 41 % 89) as f32 - 44.0) / 22.0)
            .collect();
        let rhs: Vec<f32> = (0..b * n * k)
            .map(|i| ((i * 29 % 83) as f32 - 41.0) / 20.0)
            .collect();
        let split = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, n, 1],
            ParamStyle::Split,
        );
        let packed = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, n, 1],
            ParamStyle::Packed,
        );
        assert_eq!(
            split.iter().map(|x: &f32| x.to_bits()).collect::<Vec<_>>(),
            packed.iter().map(|x: &f32| x.to_bits()).collect::<Vec<_>>(),
            "gemv_t and its packed form disagree at (b,m,n,k)=({b},{m},{n},{k})"
        );
    }
}

/// The f16 path, which is what LFM2 decode actually dispatches.
///
/// 167 of the 183 gemv dispatches per token are f16 (issue #41's inventory), so
/// an f32-only parity test would leave the decode path itself unchecked.
#[test]
fn packed_matches_split_for_gemv_f16() {
    let (b, m, n, k) = (1usize, 2048usize, 1usize, 2048usize);
    let lhs: Vec<f16> = (0..b * m * k)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 50.0))
        .collect();
    let rhs: Vec<f16> = (0..b * n * k)
        .map(|i| f16::from_f32(((i * 53 % 97) as f32 - 48.0) / 48.0))
        .collect();
    let split = run_mlx_gemv_style(
        GemmDType::F16,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        &rhs,
        &[n * k, 1, n],
        ParamStyle::Split,
    );
    let packed = run_mlx_gemv_style(
        GemmDType::F16,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        &rhs,
        &[n * k, 1, n],
        ParamStyle::Packed,
    );
    assert_eq!(
        split.iter().map(|x: &f16| x.to_bits()).collect::<Vec<_>>(),
        packed.iter().map(|x: &f16| x.to_bits()).collect::<Vec<_>>(),
        "gemv_float16 and its packed form disagree"
    );
}

/// Every classical gemv name has a `_packed` counterpart in the compiled
/// library, and vice versa.
///
/// The same argument as `packed_names_resolve` for `reduce.metal`: an absent
/// instantiation is a compile error on neither side, and the first symptom
/// would be a `LoadFunctionError` inside a forward pass. #26 shipped exactly
/// that for 48 variants, caught only by loading against the library
/// (`DESIGN.md` §8.1b).
///
/// The axis lists are restated here rather than derived from the `.metal`
/// macros, which is the point: if a tile configuration is added to one and not
/// the other, this fails. 144 classical names, 144 packed.
#[test]
fn gemv_packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    // (bm, bn, sm, sn, tm, tn), matching `instantiate_gemv_blocks`.
    let gemv_blocks = [
        (1, 8, 1, 32, 4, 4),
        (1, 8, 1, 32, 1, 4),
        (1, 1, 8, 4, 4, 4),
        (1, 1, 8, 4, 1, 4),
        (4, 1, 1, 32, 1, 4),
        (4, 1, 1, 32, 4, 4),
        (8, 1, 1, 32, 4, 4),
    ];
    // matching `instantiate_gemv_t_blocks`.
    let gemv_t_blocks = [
        (1, 2, 8, 4, 4, 1),
        (1, 2, 8, 4, 4, 4),
        (1, 4, 8, 4, 4, 4),
        (1, 16, 8, 4, 4, 4),
        (1, 16, 4, 8, 4, 4),
    ];

    let mut checked = 0usize;
    for (prefix, blocks) in [("gemv", &gemv_blocks[..]), ("gemv_t", &gemv_t_blocks[..])] {
        for dtype in ["float32", "float16", "bfloat16"] {
            for &(bm, bn, sm, sn, tm, tn) in blocks.iter() {
                for nc in [0, 1] {
                    for axpby in [0, 1] {
                        let classical = format!(
                            "{prefix}_{dtype}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{nc}_axpby{axpby}"
                        );
                        let packed = crate::kernels::params::packed_name(&classical);
                        kernels
                            .load_pipeline(&device, Source::Gemv, classical.clone())
                            .unwrap_or_else(|e| panic!("{classical} does not resolve: {e:?}"));
                        kernels
                            .load_pipeline(&device, Source::Gemv, packed.clone())
                            .unwrap_or_else(|e| panic!("{packed} does not resolve: {e:?}"));
                        checked += 2;
                    }
                }
            }
        }
    }
    assert_eq!(
        checked, 288,
        "expected 144 classical + 144 packed gemv variants"
    );
}

/// `GemvParams`'s layout agrees between `gemv.metal` and `params.rs`.
///
/// A `static_assert` on either side proves only that side is self-consistent;
/// this ships the *compiled kernel's own* `sizeof` and field offsets across the
/// boundary and compares them against Rust's `size_of`/`offset_of!`. A field at
/// the wrong offset does not crash -- the kernel reads a well-formed number
/// from the wrong place and computes a plausible wrong answer (`DESIGN.md`
/// §3.5, §15.1).
#[test]
fn gemv_params_layout_matches_metal() {
    use crate::kernels::params::{gemv_expected_layout, GEMV_LAYOUT_KERNEL, GEMV_LAYOUT_SLOTS};

    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let out = device
        .new_buffer(
            GEMV_LAYOUT_SLOTS * core::mem::size_of::<u32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();

    let pipeline = kernels
        .load_pipeline(&device, Source::Gemv, GEMV_LAYOUT_KERNEL)
        .unwrap();
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_output_buffer(0, Some(&out), 0);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let device_side: Vec<u32> = read_to_vec(&out, GEMV_LAYOUT_SLOTS);
    let mut disagreements = Vec::new();
    for (slot, (what, rust)) in gemv_expected_layout().into_iter().enumerate() {
        if device_side[slot] != rust {
            disagreements.push(format!(
                "  {what}: metal {} vs rust {rust}",
                device_side[slot]
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "packed parameter layout differs between gemv.metal and params.rs.\n\
         A field at the wrong offset is silent corruption, not a crash:\n{}",
        disagreements.join("\n")
    );
}

/// The gemv layout check must be able to fail.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test. The mutation is applied to a **copy of `gemv.metal`**
/// compiled at runtime, so the real source is untouched.
///
/// It **swaps `GemvParams.in_vec_size` and `.out_vec_size`** -- two adjacent
/// fields of the same type, which is the case nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `gemv.metal` still
///   passes;
/// * both are `int`, so no narrowing diagnostic fires -- inserting a field of a
///   *different* type is rejected at compile time by C++11 narrowing on the
///   brace initializer, a real layer of defence but not this one;
/// * the kernel runs and computes a plausible wrong answer, since swapping the
///   loop bound with the output length is arithmetic that still terminates.
///
/// That last point is why this specific pair was chosen over any other adjacent
/// pair: it is the one whose confusion would corrupt a matmul silently rather
/// than crash it.
#[test]
fn gemv_params_layout_check_detects_a_moved_field() {
    use crate::kernels::params::{gemv_expected_layout, GEMV_LAYOUT_KERNEL, GEMV_LAYOUT_SLOTS};

    let original = "struct GemvParams {\n  int in_vec_size;\n  int out_vec_size;";
    let swapped = "struct GemvParams {\n  int out_vec_size;\n  int in_vec_size;";
    assert!(
        crate::source::GEMV.contains(original),
        "the struct this test mutates has been reworded; update the mutation"
    );
    let mutant = crate::source::GEMV.replace(original, swapped);

    let device = device();
    let library = device.new_library_with_source(&mutant, None).unwrap();
    let function = library.get_function(GEMV_LAYOUT_KERNEL, None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out = device
        .new_buffer(
            GEMV_LAYOUT_SLOTS * core::mem::size_of::<u32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_output_buffer(0, Some(&out), 0);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let device_side: Vec<u32> = read_to_vec(&out, GEMV_LAYOUT_SLOTS);
    let disagreements: Vec<String> = gemv_expected_layout()
        .into_iter()
        .enumerate()
        .filter(|(slot, (_, rust))| device_side[*slot] != *rust)
        .map(|(slot, (what, rust))| format!("{what}: metal {} vs rust {rust}", device_side[slot]))
        .collect();

    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("GemvParams.in_vec_size")),
        "swapping in_vec_size must be reported; got {disagreements:?}"
    );
    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("GemvParams.out_vec_size")),
        "swapping out_vec_size must be reported; got {disagreements:?}"
    );
    assert!(
        !disagreements
            .iter()
            .any(|d| d.starts_with("sizeof(GemvParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {disagreements:?}"
    );
}

/// The packed entry point does not change the inner loop's shape.
///
/// This is the check issue #41 asks for that the elementwise families did not
/// need. `gemv` is where the weight traffic happens and decode is bandwidth-
/// bound at 88 % of a streaming roofline (`DESIGN.md` §13.4a), so an added
/// per-iteration load or a register spill here is a real regression rather
/// than noise.
///
/// `maxTotalThreadsPerThreadgroup` is computed by the compiler from
/// registers-per-thread, so it is a direct read on register pressure
/// (`DESIGN.md` §3.2: registers/thread -> occupancy -> latency hiding). If the
/// packed form held an extra live value or spilled, this is where it would
/// show. `staticThreadgroupMemoryLength` covers the other half — the
/// `tgp_mem_size` the template computed.
///
/// Why it should hold, which this measures rather than assumes: both entry
/// points pass the scalars **by value** into `gemv_body`, whose parameters are
/// `const int`/`const float`, and `GEMVKernel::run` takes them by value again.
/// So the loop bound and the row stride are in registers before the first
/// iteration under either style, and neither `constant int&` nor
/// `device GemvParams*` is dereferenced inside the loop. Promoting them to a
/// buffer moves *where the one load happens*, not how often.
///
/// There is no `metal-objdump` in this Xcode, so the loop body cannot be
/// diffed directly; this is the strongest check the runtime exposes. The
/// standalone form is `measurements/probes/issue-41/regpressure.m`.
#[test]
fn gemv_packed_does_not_change_occupancy() {
    let device = device();
    let kernels = Kernels::new();

    // The four the decode inventory names, plus the bm8 f16 row that carries
    // the most dispatches. Named explicitly rather than swept so a failure
    // says which variant regressed.
    let decode_path = [
        "gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0",
    ];

    let mut differences = Vec::new();
    for classical in decode_path {
        let packed = crate::kernels::params::packed_name(classical);
        let pc = kernels
            .load_pipeline(&device, Source::Gemv, classical)
            .unwrap();
        let pp = kernels
            .load_pipeline(&device, Source::Gemv, packed.clone())
            .unwrap();
        if pc.max_total_threads_per_threadgroup() != pp.max_total_threads_per_threadgroup() {
            differences.push(format!(
                "{classical}: maxTotalThreadsPerThreadgroup {} vs {} packed",
                pc.max_total_threads_per_threadgroup(),
                pp.max_total_threads_per_threadgroup()
            ));
        }
    }

    assert!(
        differences.is_empty(),
        "the packed entry point changed occupancy, which means register pressure \
         in the matmul inner loop moved:\n{}",
        differences.join("\n")
    );
}
