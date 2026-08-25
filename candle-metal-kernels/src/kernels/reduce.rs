use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source, RESOURCE_OPTIONS,
};
use objc2_metal::MTLSize;

use crate::kernels::params::{
    NormParams, ReduceParams, RopeIParams, RopeParams, RopeThdParams, SoftmaxParams,
};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `reduce_params_layout_matches_metal` is what proves those
/// mirrors agree with `reduce.metal`.
const REDUCE_PARAMS_ALIGN: usize = core::mem::align_of::<ReduceParams>();
const SOFTMAX_PARAMS_ALIGN: usize = core::mem::align_of::<SoftmaxParams>();
const NORM_PARAMS_ALIGN: usize = core::mem::align_of::<NormParams>();
const ROPE_I_PARAMS_ALIGN: usize = core::mem::align_of::<RopeIParams>();
const ROPE_PARAMS_ALIGN: usize = core::mem::align_of::<RopeParams>();
const ROPE_THD_PARAMS_ALIGN: usize = core::mem::align_of::<RopeThdParams>();

// A width mismatch this change surfaced, recorded because it is latent on the
// classical path rather than introduced here.
//
// `softmax_kernel`, `rms_norm_kernel` and `layer_norm_kernel` declare their
// scalars as `constant uint &` -- 4 bytes -- but `call_last_softmax`,
// `call_rms_norm` and `call_layer_norm` bound them as `usize`, which is 8. The
// classical path survives that only because `setBytes` writes 8 bytes into the
// argument slot and a little-endian `uint` read takes the low 4, which are the
// right ones for any length that fits in 32 bits.
//
// Packing turns it into a real error rather than a benign one: the block is
// built from each value's own width, so two `usize` fields occupy 16 bytes
// where the struct is 8, and every field after the first lands wrong. The
// parity test caught it on the first run, which is the argument for having it.
//
// The fix is to bind what the kernel declares -- `as u32`, which is what the
// reduce and arg-reduce families already did. That is a no-op for the classical
// path (the same low four bytes reach the same slot), and it is why the LFM2
// digests are unchanged.

/// How a kernel's scalars reach it.
///
/// `Split` is what candle has always done: one `setBytes` per scalar. `Packed`
/// puts them in a device buffer instead, which is the only form an ICB command
/// can express (`DESIGN.md` §3.7b, issue #38).
///
/// Both are compiled into the same metallib from the same kernel body, so this
/// selects a `[[host_name]]` and nothing more -- a compile-tier variant axis in
/// the sense of `DESIGN.md` §7.1, alongside dtype. Keeping both is what makes
/// the A/B free: same inputs, two pipelines, compare outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParamStyle {
    /// Inline constants via `setBytes`. The default and the correctness bar.
    #[default]
    Split,
    /// One `device const Params*`, bindable by `setKernelBuffer`.
    Packed,
}

impl ParamStyle {
    /// The `[[host_name]]` to load for this style.
    ///
    /// `_packed` is appended after the dtype and any `_strided`, matching the
    /// `init_*` macros in `reduce.metal`. The resolution test in `tests.rs`
    /// checks both spellings against the compiled library rather than against
    /// each other, which is `DESIGN.md` §8.1b's argument and what caught a
    /// whole family of absent names during #26.
    ///
    /// Returns `KernelName` rather than a string so the classical path keeps
    /// its `&'static str` and allocates nothing: the pipeline cache is keyed on
    /// this, and it is what the per-token path hits (§15.2 #10).
    fn kernel_name(self, classical: &'static str) -> crate::kernel::KernelName {
        match self {
            ParamStyle::Split => crate::kernel::KernelName::from(classical),
            ParamStyle::Packed => {
                crate::kernel::KernelName::from(crate::kernels::params::packed_name(classical))
            }
        }
    }
}

/// Bind the scalars `f` sets, either inline or as one packed buffer.
///
/// The packed block is built by letting `f` run exactly as it does on the
/// classical path -- `EncoderParam::set_param` diverts each scalar into the
/// capture instead of calling `setBytes` -- so the two styles cannot disagree
/// about which values are bound or in what order. That is the property worth
/// having: a hand-written packing struct beside a `set_params!` call is two
/// declarations of one thing, and `DESIGN.md` §8.1b is about not having those.
///
/// The staging buffers are per-call, which is deliberate for this change and is
/// *not* what a decode path should do. This exists to prove the mechanism and
/// make the A/B free; a real ICB path wants a plan-owned constants buffer
/// (`DESIGN.md` §4.4, §15.2 #8) written once and re-pointed per step, which is
/// the same object `KvDescriptor` is in §10.5. Recorded here rather than hidden,
/// because an allocation per dispatch would violate §15.2 #10 if it ever became
/// the default -- and it is why no performance claim is made for `Packed`.
///
/// The returned buffers must outlive the dispatch, so the caller holds them
/// until after `dispatch_thread_groups` rather than dropping them here.
#[must_use = "the staging buffers must outlive the dispatch"]
fn begin_packed_params(
    encoder: &ComputeCommandEncoder,
    style: ParamStyle,
) -> Option<&ComputeCommandEncoder> {
    match style {
        ParamStyle::Split => None,
        ParamStyle::Packed => {
            encoder.begin_param_capture();
            Some(encoder)
        }
    }
}

/// Close a capture, upload the packed block, and bind it at slot 0.
///
/// Returns every buffer that has to stay alive until the dispatch completes:
/// the params block itself, plus any array promoted out of `setBytes`.
fn finish_packed_params(
    device: &Device,
    encoder: &ComputeCommandEncoder,
    style: ParamStyle,
    align: usize,
) -> Result<Vec<Buffer>, MetalKernelError> {
    if style == ParamStyle::Split {
        return Ok(Vec::new());
    }
    let (bytes, mut staged) = encoder.end_param_capture(align);
    let params = device.new_buffer_with_data(
        bytes.as_ptr() as *const std::ffi::c_void,
        bytes.len(),
        RESOURCE_OPTIONS,
    )?;
    // Slot 0, after the capture has closed, so this is not renumbered.
    encoder.set_input_buffer(0, Some(&params), 0);
    staged.push(params);
    Ok(staged)
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_reduce_contiguous_with(
        device,
        ep,
        kernels,
        kernel_name,
        shape,
        out_length,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_reduce_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "reduce {kernel_name} length={length} out_length={out_length}"
    );

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            work_per_threadgroup as u32,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, REDUCE_PARAMS_ALIGN)?;

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_reduce_strided_with(
        device,
        ep,
        kernels,
        kernel_name,
        shape,
        strides,
        out_length,
        input,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "reduce_strided {kernel_name} length={length} out_length={out_length}"
    );

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            strides.as_slice(),
            work_per_threadgroup as u32,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, REDUCE_PARAMS_ALIGN)?;

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_last_softmax_with(
        device,
        ep,
        kernels,
        kernel_name,
        length,
        elements,
        input,
        input_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let work_per_threadgroup = elements;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "softmax {kernel_name} length={length} elements={elements}"
    );

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length as u32,
            work_per_threadgroup as u32,
            (input, input_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SOFTMAX_PARAMS_ALIGN)?;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_rms_norm_with(
        device,
        ep,
        kernels,
        kernel_name,
        length,
        elements_to_sum,
        eps,
        input,
        input_offset,
        alpha,
        alpha_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "rms_norm {kernel_name} length={length} elements_to_sum={elements_to_sum}"
    );

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length as u32,
            elements_to_sum as u32,
            (input, input_offset),
            Output::new(output),
            (alpha, alpha_offset),
            eps
        )
    );
    let _staged = finish_packed_params(device, encoder, style, NORM_PARAMS_ALIGN)?;
    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_layer_norm_with(
        device,
        ep,
        kernels,
        kernel_name,
        length,
        elements_to_sum,
        eps,
        input,
        input_offset,
        alpha,
        alpha_offset,
        beta,
        beta_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "layer_norm {kernel_name} length={length} elements_to_sum={elements_to_sum}"
    );

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length as u32,
            elements_to_sum as u32,
            (input, input_offset),
            Output::new(output),
            (alpha, alpha_offset),
            (beta, beta_offset),
            eps
        )
    );
    let _staged = finish_packed_params(device, encoder, style, NORM_PARAMS_ALIGN)?;

    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_rope_i_with(
        device,
        ep,
        kernels,
        kernel_name,
        bh,
        td,
        stride_b,
        src,
        src_offset,
        cos,
        cos_offset,
        sin,
        sin_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope_i {kernel_name} bh={bh} td={td}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            bh,
            td,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, ROPE_I_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_rope_thd_with(
        device,
        ep,
        kernels,
        kernel_name,
        b,
        t,
        h,
        d,
        stride_b,
        src,
        src_offset,
        cos,
        cos_offset,
        sin,
        sin_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope_thd {kernel_name} b={b} t={t} h={h} d={d}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            b,
            t,
            h,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, ROPE_THD_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_rope_with(
        device,
        ep,
        kernels,
        kernel_name,
        bh,
        td,
        d,
        stride_b,
        src,
        src_offset,
        cos,
        cos_offset,
        sin,
        sin_offset,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope {kernel_name} bh={bh} td={td} d={d}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            bh,
            td,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, ROPE_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
