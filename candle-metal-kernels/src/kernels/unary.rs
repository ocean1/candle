use crate::kernels::macros::ops;
use crate::kernels::params::{
    begin_packed_params, finish_packed_params, Copy2dParams, ParamStyle, UnaryParams,
    UnaryStridedParams,
};
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, Kernels,
    MetalKernelError, Output, Source,
};
use crate::{get_block_dims, get_tile_size, linear_split};
use objc2_metal::MTLSize;

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `unary_params_layout_matches_metal` is what proves those
/// mirrors agree with `unary.metal`.
const UNARY_PARAMS_ALIGN: usize = core::mem::align_of::<UnaryParams>();
const UNARY_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<UnaryStridedParams>();
const COPY2D_PARAMS_ALIGN: usize = core::mem::align_of::<Copy2dParams>();

ops!(
    cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf, tanh,
    recip, silu, sign, sigmoid, const_set
);

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_unary_contiguous_with(
        device,
        ep,
        kernels,
        kernel_name,
        dtype_size,
        length,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_unary_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "unary {} elems={length}", kernel_name.0);

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (length, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_unary_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        strides,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_unary_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "unary_strided {} elems={length}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            &input,
            Output::from_buffer_offset(&output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_const_set_contiguous_with(
        device,
        ep,
        kernels,
        kernel_name,
        dtype_size,
        length,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_const_set_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_const_set_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "const_set {} elems={length}", kernel_name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (length, input, Output::from_buffer_offset(&output))
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_const_set_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        strides,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_const_set_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_const_set_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "const_set_strided {} elems={length}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            input,
            Output::from_buffer_offset(&output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub mod copy2d {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy2d_f32");
    pub const HALF: Kernel = Kernel("copy2d_f16");
    pub const BFLOAT: Kernel = Kernel("copy2d_bf16");
    pub const I64: Kernel = Kernel("copy2d_i64");
    pub const I32: Kernel = Kernel("copy2d_i32");
    pub const I16: Kernel = Kernel("copy2d_i16");
    pub const U32: Kernel = Kernel("copy2d_u32");
    pub const U8: Kernel = Kernel("copy2d_u8");
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
) -> Result<(), MetalKernelError> {
    call_copy2d_with(
        device,
        ep,
        kernels,
        name,
        input,
        output,
        d1,
        d2,
        src_s,
        dst_s,
        src_o_in_bytes,
        dst_o_in_bytes,
        ParamStyle::default(),
    )
}

/// As [`call_copy2d`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_copy2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "copy2d {} d1={d1} d2={d2}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            d1 as i64,
            d2 as i64,
            src_s as i64,
            dst_s as i64,
            (input, src_o_in_bytes),
            Output::with_offset(output, dst_o_in_bytes)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, COPY2D_PARAMS_ALIGN)?;

    let grid_dims = MTLSize {
        width: d1,
        height: d2,
        depth: 1,
    };
    let group_dims = get_block_dims(d1, d2, 1);
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}
