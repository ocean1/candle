//! Packed parameter structs for the kernel families that carry both binding
//! styles, and the layout checks that keep them honest.
//!
//! `reduce.metal` came first (issue #38); `unary`, `binary`, `cast` and
//! `affine` followed (issue #40), then `gemv` (issue #41) and `conv` (issue
//! #42). Each family declares its own structs and its own layout kernel,
//! because each `.metal` file is compiled into its own library -- a kernel in
//! one cannot see a struct in another.
//!
//! # Why these exist
//!
//! `MTLIndirectComputeCommand` has no `setBytes` in any form (`DESIGN.md`
//! §3.7b), so a kernel that takes its scalars as `constant uint &n` cannot be
//! encoded into an ICB at all. `setKernelBuffer` is the only binding primitive
//! an ICB command has, so the scalars have to arrive in a buffer.
//!
//! `reduce.metal` therefore carries every kernel in two forms: the classical
//! one, unchanged, binding each scalar with `setBytes`; and a `_packed` one
//! taking a single `device const Params*`. Both are instantiated from the same
//! body, so this is a compile-tier variant axis in the sense of `DESIGN.md`
//! §7.1 — the same kind of thing as dtype — rather than a migration.
//!
//! # Why the layout is checked rather than trusted
//!
//! A field at the wrong offset does not crash. The kernel reads a well-formed
//! number from the wrong place and computes a plausible wrong answer, which
//! under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
//! §15.1 both single out. MSL is C++14 with standard layout rules so the two
//! sides *should* agree, but "should" is what the check is for.
//!
//! Both hazards `#38` names are avoided by construction here and are worth
//! stating, because the next family converted may not be so lucky:
//!
//! * **Vector types over-align.** `float3` occupies 16 bytes in MSL, not 12
//!   (`packed_float3` exists for exactly this). No field below is a vector.
//! * **`bool` is 1 byte in MSL**, and `EncoderParam` has a `primitive!(bool)`.
//!   No field below is a `bool`.
//!
//! `size_t` in MSL is 8 bytes, which is why the RoPE structs use `u64` and are
//! 8-aligned where the reduction structs are 4-aligned. Narrowing them to `u32`
//! would have been a numeric change smuggled in beside a binding change.
//!
//! # How a family registers its check
//!
//! [`LayoutFamily`] at the foot of this file. A family is a variant, and
//! [`LayoutFamily::descriptor`] matches exhaustively, so **a family that does
//! not register does not compile** (issue #58). Before that the registration was
//! a call site in `tests.rs`, and a family omitted there was simply never
//! checked — indistinguishable from one that passed.

use crate::metal::{Buffer, ComputeCommandEncoder, Device};
use crate::{MetalKernelError, RESOURCE_OPTIONS};

/// Scalars bound by the reduce and arg-reduce entry points.
///
/// `dims` and `strides` are deliberately absent: their length is a property of
/// the tensor's layout, not of the struct, so they stay separate bindings. An
/// ICB can express that — `setKernelBuffer` binds a buffer of any length — it
/// just cannot express a `setBytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReduceParams {
    pub src_numel: u32,
    pub num_dims: u32,
    pub el_per_block: u32,
}

/// Scalars bound by `softmax_kernel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftmaxParams {
    pub src_numel: u32,
    pub el_per_block: u32,
}

/// Scalars bound by `rms_norm_kernel` and `layer_norm_kernel`.
///
/// One struct for both because they bind the same three scalars; the norms
/// differ in their *buffer* arguments (`layer_norm` takes a `beta`), which are
/// bindings either way and so are not part of this.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormParams {
    pub src_numel: u32,
    pub el_per_block: u32,
    pub eps: f32,
}

/// Scalars bound by `rope_i_kernel`. `u64` mirrors MSL's 8-byte `size_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeIParams {
    pub bh: u64,
    pub td: u64,
    pub stride_b: u64,
}

/// Scalars bound by `rope_kernel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeParams {
    pub bh: u64,
    pub td: u64,
    pub d: u64,
    pub stride_b: u64,
}

/// Scalars bound by `rope_thd_kernel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeThdParams {
    pub b: u64,
    pub t: u64,
    pub h: u64,
    pub d: u64,
    pub stride_b: u64,
}

/// Scalars bound by `gemv` and `gemv_t` (issue #41).
///
/// Field order mirrors the classical argument order in `call_mlx_gemv`, which
/// is what lets the packed block be built by letting the existing `set_params!`
/// run and diverting each scalar as it passes: only one argument list exists,
/// so the two binding styles cannot disagree about what is bound or in what
/// order.
///
/// Every field is a 4-byte scalar, so this is 28 bytes at 4-byte alignment with
/// no padding. `alpha`/`beta` are `f32` between `i32`s deliberately — that is
/// the classical order, and reordering to group by type would be a second
/// declaration of the same thing.
///
/// `batch_shape` and the three stride arrays are *not* fields: their length is
/// a property of the call, so they stay separate bindings. An ICB can express
/// that — `setKernelBuffer` binds a buffer of any length — since the constraint
/// is `setBytes`, not buffer count (`DESIGN.md` §11.3d).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemvParams {
    pub in_vec_size: i32,
    pub out_vec_size: i32,
    pub matrix_ld: i32,
    pub alpha: f32,
    pub beta: f32,
    pub batch_ndim: i32,
    pub bias_stride: i32,
}

// ---------------------------------------------------------------------------
// `conv.metal` (issue #42).
//
// Ten families, 59 `constant &` parameters -- more than any other file bar
// `reduce`. Two things distinguish them from the reduce structs above and are
// worth stating where they are decided:
//
// * **`size_t` throughout, so these are 8-aligned** where the reduction structs
//   are 4-aligned `uint`. The one exception is `UpsampleBilinear2dParams`.
// * **`UpsampleBilinear2dParams` mixes widths.** Three `bool` (1 byte in MSL, as
//   in Rust) and two `float` sit between two `size_t`, so five of its seven
//   fields land at an offset the padding rule decides rather than at a multiple
//   of a field width. It exercises both hazards issue #38 names -- sub-word
//   types and mixed widths -- in one struct, where #40's `AffineParams` (12
//   bytes of fields, `sizeof` 16) exercises the padding half alone. Neither is
//   obvious by inspection, which is the argument for checking offsets across
//   the boundary rather than asserting them on one side.
//
// Field order mirrors each kernel's classical argument list exactly. That is
// load-bearing rather than stylistic: the packed block is built by letting the
// existing `set_params!` call run and diverting each scalar as it passes, so a
// struct whose order differs from the argument list would silently misread every
// field after the first divergence.
// ---------------------------------------------------------------------------

/// Scalars bound by `im2col`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2colParams {
    pub dst_numel: u64,
    pub h_out: u64,
    pub w_out: u64,
    pub h_k: u64,
    pub w_k: u64,
    pub stride: u64,
    pub padding: u64,
    pub dilation: u64,
}

/// Scalars bound by `col2im1d`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Col2im1dParams {
    pub dst_el: u64,
    pub l_out: u64,
    pub l_in: u64,
    pub c_out: u64,
    pub k_size: u64,
    pub stride: u64,
}

/// Scalars bound by `im2col1d`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2col1dParams {
    pub dst_numel: u64,
    pub l_out: u64,
    pub l_k: u64,
    pub stride: u64,
    pub padding: u64,
    pub dilation: u64,
}

/// Scalars bound by `upsample_nearest2d`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpsampleNearest2dParams {
    pub w_out: u64,
    pub h_out: u64,
    pub w_scale: f32,
    pub h_scale: f32,
}

/// Scalars bound by `upsample_bilinear2d`.
///
/// The mixed-width struct. `bool` is 1 byte on both sides, and the two `f32`
/// pad to 4 after it, so the offsets are 0, 8, 16, 17, 20, 24, 28 and the whole
/// thing pads to 32 for its own 8-byte alignment. None of those numbers is a
/// field width; every one is a padding rule, which is why
/// `conv_params_layout_matches_metal` ships them across the boundary instead of
/// either side asserting its own.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpsampleBilinear2dParams {
    pub w_out: u64,
    pub h_out: u64,
    pub align_corners: bool,
    pub has_scale_h: bool,
    pub scale_h_factor: f32,
    pub has_scale_w: bool,
    pub scale_w_factor: f32,
}

/// Scalars bound by `max_pool2d` and `avg_pool2d`.
///
/// One struct for both because they bind the same four scalars. They differ in
/// their accumulator, which is a template parameter on the Metal side and not
/// part of the binding — the integer `avg_pool2d` instantiations accumulate in
/// their own type rather than widening, and that must not move (`DESIGN.md`
/// §8.1c).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool2dParams {
    pub w_k: u64,
    pub h_k: u64,
    pub w_stride: u64,
    pub h_stride: u64,
}

/// Scalars bound by `conv_transpose1d`.
///
/// `out_padding` is bound but never read by the kernel. It stays a field
/// because the struct must mirror the argument list; dropping it would move
/// `dilation` by 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose1dParams {
    pub l_out: u64,
    pub stride: u64,
    pub padding: u64,
    pub out_padding: u64,
    pub dilation: u64,
}

/// Scalars bound by `conv_transpose2d`. `out_padding` is unread, as above.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose2dParams {
    pub w_out: u64,
    pub h_out: u64,
    pub stride: u64,
    pub padding: u64,
    pub out_padding: u64,
    pub dilation: u64,
}

/// Scalars bound by `conv1d_depthwise`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dDepthwiseParams {
    pub dst_numel: u64,
    pub l_out: u64,
    pub k_size: u64,
    pub stride: u64,
    pub padding: u64,
    pub dilation: u64,
}

/// Scalars bound by `conv1d_depthwise_k`, whose `k_size`, `stride` and
/// `dilation` are compile-time or preconditions rather than bindings.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dDepthwiseKParams {
    pub dst_numel: u64,
    pub l_out: u64,
    pub padding: u64,
}

/// The kernel in `conv.metal` that reports the device-side layout.
pub const CONV_LAYOUT_KERNEL: &str = "conv_params_layout";

/// What each slot of [`CONV_LAYOUT_KERNEL`]'s output means, and what Rust
/// computes for it.
///
/// Same shape as [`expected_reduce_layout`], and deliberately a separate
/// function rather than an extension of it: `conv.metal` and `reduce.metal` are
/// compiled as separate libraries, so their layout kernels are separate
/// dispatches and each family's slots have to be addressed against its own.
pub fn expected_conv_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<Im2colParams>(), 8);
    debug_assert_eq!(align_of::<Col2im1dParams>(), 8);
    debug_assert_eq!(align_of::<Im2col1dParams>(), 8);
    debug_assert_eq!(align_of::<UpsampleNearest2dParams>(), 8);
    debug_assert_eq!(align_of::<UpsampleBilinear2dParams>(), 8);
    debug_assert_eq!(align_of::<Pool2dParams>(), 8);
    debug_assert_eq!(align_of::<ConvTranspose1dParams>(), 8);
    debug_assert_eq!(align_of::<ConvTranspose2dParams>(), 8);
    debug_assert_eq!(align_of::<Conv1dDepthwiseParams>(), 8);
    debug_assert_eq!(align_of::<Conv1dDepthwiseKParams>(), 8);

    vec![
        ("sizeof(Im2colParams)", size_of::<Im2colParams>() as u32),
        (
            "Im2colParams.dst_numel",
            offset_of!(Im2colParams, dst_numel) as u32,
        ),
        ("Im2colParams.h_out", offset_of!(Im2colParams, h_out) as u32),
        ("Im2colParams.w_out", offset_of!(Im2colParams, w_out) as u32),
        ("Im2colParams.h_k", offset_of!(Im2colParams, h_k) as u32),
        ("Im2colParams.w_k", offset_of!(Im2colParams, w_k) as u32),
        (
            "Im2colParams.stride",
            offset_of!(Im2colParams, stride) as u32,
        ),
        (
            "Im2colParams.padding",
            offset_of!(Im2colParams, padding) as u32,
        ),
        (
            "Im2colParams.dilation",
            offset_of!(Im2colParams, dilation) as u32,
        ),
        ("sizeof(Col2im1dParams)", size_of::<Col2im1dParams>() as u32),
        (
            "Col2im1dParams.dst_el",
            offset_of!(Col2im1dParams, dst_el) as u32,
        ),
        (
            "Col2im1dParams.l_out",
            offset_of!(Col2im1dParams, l_out) as u32,
        ),
        (
            "Col2im1dParams.l_in",
            offset_of!(Col2im1dParams, l_in) as u32,
        ),
        (
            "Col2im1dParams.c_out",
            offset_of!(Col2im1dParams, c_out) as u32,
        ),
        (
            "Col2im1dParams.k_size",
            offset_of!(Col2im1dParams, k_size) as u32,
        ),
        (
            "Col2im1dParams.stride",
            offset_of!(Col2im1dParams, stride) as u32,
        ),
        ("sizeof(Im2col1dParams)", size_of::<Im2col1dParams>() as u32),
        (
            "Im2col1dParams.dst_numel",
            offset_of!(Im2col1dParams, dst_numel) as u32,
        ),
        (
            "Im2col1dParams.l_out",
            offset_of!(Im2col1dParams, l_out) as u32,
        ),
        ("Im2col1dParams.l_k", offset_of!(Im2col1dParams, l_k) as u32),
        (
            "Im2col1dParams.stride",
            offset_of!(Im2col1dParams, stride) as u32,
        ),
        (
            "Im2col1dParams.padding",
            offset_of!(Im2col1dParams, padding) as u32,
        ),
        (
            "Im2col1dParams.dilation",
            offset_of!(Im2col1dParams, dilation) as u32,
        ),
        (
            "sizeof(UpsampleNearest2dParams)",
            size_of::<UpsampleNearest2dParams>() as u32,
        ),
        (
            "UpsampleNearest2dParams.w_out",
            offset_of!(UpsampleNearest2dParams, w_out) as u32,
        ),
        (
            "UpsampleNearest2dParams.h_out",
            offset_of!(UpsampleNearest2dParams, h_out) as u32,
        ),
        (
            "UpsampleNearest2dParams.w_scale",
            offset_of!(UpsampleNearest2dParams, w_scale) as u32,
        ),
        (
            "UpsampleNearest2dParams.h_scale",
            offset_of!(UpsampleNearest2dParams, h_scale) as u32,
        ),
        (
            "sizeof(UpsampleBilinear2dParams)",
            size_of::<UpsampleBilinear2dParams>() as u32,
        ),
        (
            "UpsampleBilinear2dParams.w_out",
            offset_of!(UpsampleBilinear2dParams, w_out) as u32,
        ),
        (
            "UpsampleBilinear2dParams.h_out",
            offset_of!(UpsampleBilinear2dParams, h_out) as u32,
        ),
        (
            "UpsampleBilinear2dParams.align_corners",
            offset_of!(UpsampleBilinear2dParams, align_corners) as u32,
        ),
        (
            "UpsampleBilinear2dParams.has_scale_h",
            offset_of!(UpsampleBilinear2dParams, has_scale_h) as u32,
        ),
        (
            "UpsampleBilinear2dParams.scale_h_factor",
            offset_of!(UpsampleBilinear2dParams, scale_h_factor) as u32,
        ),
        (
            "UpsampleBilinear2dParams.has_scale_w",
            offset_of!(UpsampleBilinear2dParams, has_scale_w) as u32,
        ),
        (
            "UpsampleBilinear2dParams.scale_w_factor",
            offset_of!(UpsampleBilinear2dParams, scale_w_factor) as u32,
        ),
        ("sizeof(Pool2dParams)", size_of::<Pool2dParams>() as u32),
        ("Pool2dParams.w_k", offset_of!(Pool2dParams, w_k) as u32),
        ("Pool2dParams.h_k", offset_of!(Pool2dParams, h_k) as u32),
        (
            "Pool2dParams.w_stride",
            offset_of!(Pool2dParams, w_stride) as u32,
        ),
        (
            "Pool2dParams.h_stride",
            offset_of!(Pool2dParams, h_stride) as u32,
        ),
        (
            "sizeof(ConvTranspose1dParams)",
            size_of::<ConvTranspose1dParams>() as u32,
        ),
        (
            "ConvTranspose1dParams.l_out",
            offset_of!(ConvTranspose1dParams, l_out) as u32,
        ),
        (
            "ConvTranspose1dParams.stride",
            offset_of!(ConvTranspose1dParams, stride) as u32,
        ),
        (
            "ConvTranspose1dParams.padding",
            offset_of!(ConvTranspose1dParams, padding) as u32,
        ),
        (
            "ConvTranspose1dParams.out_padding",
            offset_of!(ConvTranspose1dParams, out_padding) as u32,
        ),
        (
            "ConvTranspose1dParams.dilation",
            offset_of!(ConvTranspose1dParams, dilation) as u32,
        ),
        (
            "sizeof(ConvTranspose2dParams)",
            size_of::<ConvTranspose2dParams>() as u32,
        ),
        (
            "ConvTranspose2dParams.w_out",
            offset_of!(ConvTranspose2dParams, w_out) as u32,
        ),
        (
            "ConvTranspose2dParams.h_out",
            offset_of!(ConvTranspose2dParams, h_out) as u32,
        ),
        (
            "ConvTranspose2dParams.stride",
            offset_of!(ConvTranspose2dParams, stride) as u32,
        ),
        (
            "ConvTranspose2dParams.padding",
            offset_of!(ConvTranspose2dParams, padding) as u32,
        ),
        (
            "ConvTranspose2dParams.out_padding",
            offset_of!(ConvTranspose2dParams, out_padding) as u32,
        ),
        (
            "ConvTranspose2dParams.dilation",
            offset_of!(ConvTranspose2dParams, dilation) as u32,
        ),
        (
            "sizeof(Conv1dDepthwiseParams)",
            size_of::<Conv1dDepthwiseParams>() as u32,
        ),
        (
            "Conv1dDepthwiseParams.dst_numel",
            offset_of!(Conv1dDepthwiseParams, dst_numel) as u32,
        ),
        (
            "Conv1dDepthwiseParams.l_out",
            offset_of!(Conv1dDepthwiseParams, l_out) as u32,
        ),
        (
            "Conv1dDepthwiseParams.k_size",
            offset_of!(Conv1dDepthwiseParams, k_size) as u32,
        ),
        (
            "Conv1dDepthwiseParams.stride",
            offset_of!(Conv1dDepthwiseParams, stride) as u32,
        ),
        (
            "Conv1dDepthwiseParams.padding",
            offset_of!(Conv1dDepthwiseParams, padding) as u32,
        ),
        (
            "Conv1dDepthwiseParams.dilation",
            offset_of!(Conv1dDepthwiseParams, dilation) as u32,
        ),
        (
            "sizeof(Conv1dDepthwiseKParams)",
            size_of::<Conv1dDepthwiseKParams>() as u32,
        ),
        (
            "Conv1dDepthwiseKParams.dst_numel",
            offset_of!(Conv1dDepthwiseKParams, dst_numel) as u32,
        ),
        (
            "Conv1dDepthwiseKParams.l_out",
            offset_of!(Conv1dDepthwiseKParams, l_out) as u32,
        ),
        (
            "Conv1dDepthwiseKParams.padding",
            offset_of!(Conv1dDepthwiseKParams, padding) as u32,
        ),
    ]
}

// ---------------------------------------------------------------------------
// `indexing.metal` (issue #81).
//
// Five families, 28 `constant &` parameters across five kernel signatures. The
// last decode-path family to carry both styles: `is_u32_f16` is the LFM2
// embedding lookup, one dispatch per token (`DESIGN.md` §11.3h).
//
// **This is the family where the `bool` hazard fires.** `DESIGN.md` §11.3b
// names two layout hazards -- vector types over-aligning, and `bool` at 1 byte
// -- and #40 recorded the second as not firing in any of the four elementwise
// families, checked by enumerating their parameters rather than assumed.
// `index` takes a `constant bool &contiguous` between five `size_t` and a
// sixth, so [`IndexParams`] is **56 bytes where its fields sum to 49**, with
// seven bytes of padding after the `bool`. `conv.metal`'s
// `UpsampleBilinear2dParams` is the only other struct in the crate with a
// `bool` in it, and it is off the decode path entirely.
//
// A width mismatch of the kind #38 found (`usize` bound to a `constant uint &`,
// benign under `setBytes` and wrong once packed) **does not occur here**,
// checked rather than assumed: `indexing.metal` declares 29 `constant size_t &`
// and one `constant bool &`, `indexing.rs` binds `usize` and `bool`
// respectively, and there is no `uint` in the file. The `()` hazard #41 found
// -- an unbound argument consuming no slot and shifting every later binding --
// does not occur either: none of the four `call_*` passes `()`.
//
// Field order mirrors each kernel's classical argument list exactly. That is
// load-bearing rather than stylistic: the packed block is built by letting the
// existing `set_params!` call run and diverting each scalar as it passes, so a
// struct whose order differs from the argument list would silently misread
// every field after the first divergence.
// ---------------------------------------------------------------------------

/// Scalars bound by `index` — `index_select`, and the family containing
/// `is_u32_f16`, the LFM2 embedding lookup.
///
/// **56 bytes, not 49.** `contiguous` is a `bool` at offset 40 and
/// `src_num_dims` is a `size_t`, so it cannot start until 48. Neither number is
/// obvious by inspection, which is the argument for shipping them across the
/// boundary rather than asserting them on each side.
///
/// `src_dims` and `src_strides` are deliberately absent: their length is a
/// property of the tensor's layout, not of the struct, so they stay separate
/// bindings. An ICB can express that — `setKernelBuffer` binds a buffer of any
/// length — since the constraint is `setBytes`, not buffer count (`DESIGN.md`
/// §11.3d).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexParams {
    pub dst_size: u64,
    pub left_size: u64,
    pub src_dim_size: u64,
    pub right_size: u64,
    pub ids_size: u64,
    pub contiguous: bool,
    pub src_num_dims: u64,
}

/// Scalars bound by `gather`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherParams {
    pub dst_size: u64,
    pub left_size: u64,
    pub src_dim_size: u64,
    pub right_size: u64,
    pub ids_size: u64,
}

/// Scalars bound by `scatter` and `scatter_add`.
///
/// One struct for both because they bind the same five scalars; they differ in
/// whether the destination is assigned or accumulated into, which is the kernel
/// body rather than the binding. The same reasoning as [`NormParams`] serving
/// both norms and [`Pool2dParams`] serving both pools.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScatterParams {
    pub dst_size: u64,
    pub left_size: u64,
    pub src_dim_size: u64,
    pub right_size: u64,
    pub dst_dim_size: u64,
}

/// Scalars bound by `index_add`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexAddParams {
    pub dst_size: u64,
    pub left_size: u64,
    pub src_dim_size: u64,
    pub right_size: u64,
    pub dst_dim_size: u64,
    pub ids_dim_size: u64,
}

/// The kernel in `indexing.metal` that reports the device-side layout.
pub const INDEXING_LAYOUT_KERNEL: &str = "indexing_params_layout";

/// What each slot of [`INDEXING_LAYOUT_KERNEL`]'s output means, and what Rust
/// computes for it.
///
/// As [`expected_reduce_layout`], for `indexing.metal`. The interesting rows
/// are [`IndexParams`]'s: `contiguous` at 40 and `src_num_dims` at 48 with a
/// `sizeof` of 56, which is the padding rule rather than any field width.
pub fn expected_indexing_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<IndexParams>(), 8);
    debug_assert_eq!(align_of::<GatherParams>(), 8);
    debug_assert_eq!(align_of::<ScatterParams>(), 8);
    debug_assert_eq!(align_of::<IndexAddParams>(), 8);

    vec![
        ("sizeof(IndexParams)", size_of::<IndexParams>() as u32),
        (
            "IndexParams.dst_size",
            offset_of!(IndexParams, dst_size) as u32,
        ),
        (
            "IndexParams.left_size",
            offset_of!(IndexParams, left_size) as u32,
        ),
        (
            "IndexParams.src_dim_size",
            offset_of!(IndexParams, src_dim_size) as u32,
        ),
        (
            "IndexParams.right_size",
            offset_of!(IndexParams, right_size) as u32,
        ),
        (
            "IndexParams.ids_size",
            offset_of!(IndexParams, ids_size) as u32,
        ),
        (
            "IndexParams.contiguous",
            offset_of!(IndexParams, contiguous) as u32,
        ),
        (
            "IndexParams.src_num_dims",
            offset_of!(IndexParams, src_num_dims) as u32,
        ),
        ("sizeof(GatherParams)", size_of::<GatherParams>() as u32),
        (
            "GatherParams.dst_size",
            offset_of!(GatherParams, dst_size) as u32,
        ),
        (
            "GatherParams.left_size",
            offset_of!(GatherParams, left_size) as u32,
        ),
        (
            "GatherParams.src_dim_size",
            offset_of!(GatherParams, src_dim_size) as u32,
        ),
        (
            "GatherParams.right_size",
            offset_of!(GatherParams, right_size) as u32,
        ),
        (
            "GatherParams.ids_size",
            offset_of!(GatherParams, ids_size) as u32,
        ),
        ("sizeof(ScatterParams)", size_of::<ScatterParams>() as u32),
        (
            "ScatterParams.dst_size",
            offset_of!(ScatterParams, dst_size) as u32,
        ),
        (
            "ScatterParams.left_size",
            offset_of!(ScatterParams, left_size) as u32,
        ),
        (
            "ScatterParams.src_dim_size",
            offset_of!(ScatterParams, src_dim_size) as u32,
        ),
        (
            "ScatterParams.right_size",
            offset_of!(ScatterParams, right_size) as u32,
        ),
        (
            "ScatterParams.dst_dim_size",
            offset_of!(ScatterParams, dst_dim_size) as u32,
        ),
        ("sizeof(IndexAddParams)", size_of::<IndexAddParams>() as u32),
        (
            "IndexAddParams.dst_size",
            offset_of!(IndexAddParams, dst_size) as u32,
        ),
        (
            "IndexAddParams.left_size",
            offset_of!(IndexAddParams, left_size) as u32,
        ),
        (
            "IndexAddParams.src_dim_size",
            offset_of!(IndexAddParams, src_dim_size) as u32,
        ),
        (
            "IndexAddParams.right_size",
            offset_of!(IndexAddParams, right_size) as u32,
        ),
        (
            "IndexAddParams.dst_dim_size",
            offset_of!(IndexAddParams, dst_dim_size) as u32,
        ),
        (
            "IndexAddParams.ids_dim_size",
            offset_of!(IndexAddParams, ids_dim_size) as u32,
        ),
    ]
}

/// The kernel in `reduce.metal` that reports the device-side layout.
pub const REDUCE_LAYOUT_KERNEL: &str = "reduce_params_layout";

/// The kernel in `gemv.metal` that reports the device-side layout.
///
/// A second kernel rather than more slots on [`REDUCE_LAYOUT_KERNEL`]: the
/// layout check has to load from the library that actually defines the struct,
/// and `gemv.metal` and `reduce.metal` are separate `Source`s compiled
/// independently (`kernel.rs:109`). Keeping them separate is also what lets the
/// families compose rather than share one fixed slot count.
pub const GEMV_LAYOUT_KERNEL: &str = "gemv_params_layout";

/// What each slot of [`GEMV_LAYOUT_KERNEL`] means, and what Rust computes for
/// it.
///
/// As [`expected_reduce_layout`], for `gemv.metal`. Written as data rather than
/// as a sequence of assertions so a mismatch reports *which* field disagrees —
/// the whole point being that one wrong offset is otherwise invisible: the
/// kernel reads a well-formed number from the wrong place and computes a
/// plausible wrong answer (`DESIGN.md` §3.5, §15.1).
pub fn expected_gemv_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<GemvParams>(), 4);

    vec![
        ("sizeof(GemvParams)", size_of::<GemvParams>() as u32),
        (
            "GemvParams.in_vec_size",
            offset_of!(GemvParams, in_vec_size) as u32,
        ),
        (
            "GemvParams.out_vec_size",
            offset_of!(GemvParams, out_vec_size) as u32,
        ),
        (
            "GemvParams.matrix_ld",
            offset_of!(GemvParams, matrix_ld) as u32,
        ),
        ("GemvParams.alpha", offset_of!(GemvParams, alpha) as u32),
        ("GemvParams.beta", offset_of!(GemvParams, beta) as u32),
        (
            "GemvParams.batch_ndim",
            offset_of!(GemvParams, batch_ndim) as u32,
        ),
        (
            "GemvParams.bias_stride",
            offset_of!(GemvParams, bias_stride) as u32,
        ),
    ]
}

/// What each slot of [`REDUCE_LAYOUT_KERNEL`]'s output means, and what Rust
/// computes for it.
///
/// The ordering is declared here and in the kernel, and
/// `reduce_params_layout_matches_metal` is what fails if they drift. Written as
/// data rather than as a sequence of assertions so a mismatch reports *which*
/// field disagrees rather than just that something did — the whole point being
/// that a single wrong offset is otherwise invisible.
pub fn expected_reduce_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    // Alignment is not shipped per slot -- it is `static_assert`ed on the Metal
    // side and checked here against the same constants -- but a size that
    // matches while alignment does not would still pad differently in an array,
    // so it is asserted rather than assumed.
    debug_assert_eq!(align_of::<ReduceParams>(), 4);
    debug_assert_eq!(align_of::<SoftmaxParams>(), 4);
    debug_assert_eq!(align_of::<NormParams>(), 4);
    debug_assert_eq!(align_of::<RopeIParams>(), 8);
    debug_assert_eq!(align_of::<RopeParams>(), 8);
    debug_assert_eq!(align_of::<RopeThdParams>(), 8);

    vec![
        ("sizeof(ReduceParams)", size_of::<ReduceParams>() as u32),
        (
            "ReduceParams.src_numel",
            offset_of!(ReduceParams, src_numel) as u32,
        ),
        (
            "ReduceParams.num_dims",
            offset_of!(ReduceParams, num_dims) as u32,
        ),
        (
            "ReduceParams.el_per_block",
            offset_of!(ReduceParams, el_per_block) as u32,
        ),
        ("sizeof(SoftmaxParams)", size_of::<SoftmaxParams>() as u32),
        (
            "SoftmaxParams.src_numel",
            offset_of!(SoftmaxParams, src_numel) as u32,
        ),
        (
            "SoftmaxParams.el_per_block",
            offset_of!(SoftmaxParams, el_per_block) as u32,
        ),
        ("sizeof(NormParams)", size_of::<NormParams>() as u32),
        (
            "NormParams.src_numel",
            offset_of!(NormParams, src_numel) as u32,
        ),
        (
            "NormParams.el_per_block",
            offset_of!(NormParams, el_per_block) as u32,
        ),
        ("NormParams.eps", offset_of!(NormParams, eps) as u32),
        ("sizeof(RopeIParams)", size_of::<RopeIParams>() as u32),
        ("RopeIParams.bh", offset_of!(RopeIParams, bh) as u32),
        ("RopeIParams.td", offset_of!(RopeIParams, td) as u32),
        (
            "RopeIParams.stride_b",
            offset_of!(RopeIParams, stride_b) as u32,
        ),
        ("sizeof(RopeParams)", size_of::<RopeParams>() as u32),
        ("RopeParams.bh", offset_of!(RopeParams, bh) as u32),
        ("RopeParams.td", offset_of!(RopeParams, td) as u32),
        ("RopeParams.d", offset_of!(RopeParams, d) as u32),
        (
            "RopeParams.stride_b",
            offset_of!(RopeParams, stride_b) as u32,
        ),
        ("sizeof(RopeThdParams)", size_of::<RopeThdParams>() as u32),
        ("RopeThdParams.b", offset_of!(RopeThdParams, b) as u32),
        ("RopeThdParams.t", offset_of!(RopeThdParams, t) as u32),
        ("RopeThdParams.h", offset_of!(RopeThdParams, h) as u32),
        ("RopeThdParams.d", offset_of!(RopeThdParams, d) as u32),
        (
            "RopeThdParams.stride_b",
            offset_of!(RopeThdParams, stride_b) as u32,
        ),
    ]
}

// ---------------------------------------------------------------------------
// The elementwise families (issue #40): unary, binary, cast, affine.
//
// Structurally these are one family in four files -- every kernel is a
// `constant size_t &dim`, an optional `(num_dims, dims, strides)` triple, zero
// to two `float` scalars, and pointers. Two things distinguish them from
// `reduce.metal`'s structs and are the reason the cross-boundary check matters
// more here, not less:
//
// * **`size_t`, not `uint`.** These are 8-byte, 8-aligned where the reduction
//   structs were 4-byte, 4-aligned.
// * **Mixed widths, hence real padding.** `affine` pairs `size_t` with `float`,
//   so `AffineParams` is 16 bytes rather than the 12 its fields sum to, and
//   `AffineStridedParams` is 24 rather than 20. `reduce.metal` had no struct
//   that padded.
//
// The `bool` hazard `#38` flags does **not** fire in any of the four: no kernel
// takes a `bool` scalar. `binary.metal`'s comparison families produce `bool`
// *outputs*, which is a `device U*` buffer binding and never reaches a packed
// struct. Checked by enumerating every `constant &` parameter in the four
// files, not assumed.
// ---------------------------------------------------------------------------

/// Scalars bound by `unary_kernel`, `const_set` and their cast/binary analogues.
///
/// One struct shape serves the contiguous entry point of all three of `unary`,
/// `binary` and `cast`, but each file declares its own so that a change to one
/// family cannot silently move another's fields. They are separate types here
/// for the same reason.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnaryParams {
    pub dim: u64,
}

/// Scalars bound by `unary_kernel_strided` and `const_set_strided`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnaryStridedParams {
    pub dim: u64,
    pub num_dims: u64,
}

/// Scalars bound by `copy2d` -- 140 of the 674 dispatches in a decode token,
/// the largest single kernel in the trace (`DESIGN.md` §11.2).
///
/// `i64`, not `u64`: the kernel declares `constant int64_t &`, and these are
/// compared signed against `idx.x`/`idx.y`. Mirroring them as unsigned would be
/// a numeric change smuggled in beside a binding change.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy2dParams {
    pub d1: i64,
    pub d2: i64,
    pub src_s: i64,
    pub dst_s: i64,
}

/// Scalars bound by `binary_kernel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryParams {
    pub dim: u64,
}

/// Scalars bound by `binary_kernel_strided`.
///
/// This family binds *three* arrays (`dims`, `left_strides`, `right_strides`)
/// where the others bind two. They stay separate bindings either way -- an ICB
/// can express a buffer of any length; what it cannot express is `setBytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryStridedParams {
    pub dim: u64,
    pub num_dims: u64,
}

/// Scalars bound by `cast_kernel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastParams {
    pub dim: u64,
}

/// Scalars bound by `cast_kernel_strided`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastStridedParams {
    pub dim: u64,
    pub num_dims: u64,
}

/// Scalars bound by `affine_kernel`.
///
/// 16 bytes, not 12: `size_t` forces 8-byte alignment, so the two trailing
/// `float`s are followed by no padding but the struct is already a multiple of
/// 8. Stated because it is the first packed struct in this crate where
/// `sizeof` is not the sum of the field widths.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineParams {
    pub dim: u64,
    pub mul: f32,
    pub add: f32,
}

/// Scalars bound by `affine_kernel_strided`. 24 bytes: 8 + 8 + 4 + 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineStridedParams {
    pub dim: u64,
    pub num_dims: u64,
    pub mul: f32,
    pub add: f32,
}

/// Scalars bound by `powf_kernel` and `elu_kernel`, which take one float where
/// `affine` takes two.
///
/// **16 bytes, not 12**: `{u64, f32}` pads out to the struct's own 8-byte
/// alignment. A separate struct rather than reusing `AffineParams` with a dead
/// `add`, because the capture's length is decided by what the call site binds:
/// these sites bind one float, so the block is 16 bytes, and a kernel expecting
/// `AffineParams` would read a fourth field that was never written.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleParams {
    pub dim: u64,
    pub mul: f32,
}

/// Scalars bound by `powf_kernel_strided` and `elu_kernel_strided`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleStridedParams {
    pub dim: u64,
    pub num_dims: u64,
    pub mul: f32,
}

/// The kernel in `unary.metal` that reports the device-side layout.
pub const UNARY_LAYOUT_KERNEL: &str = "unary_params_layout";

/// The kernel in `binary.metal` that reports the device-side layout.
pub const BINARY_LAYOUT_KERNEL: &str = "binary_params_layout";

/// The kernel in `cast.metal` that reports the device-side layout.
pub const CAST_LAYOUT_KERNEL: &str = "cast_params_layout";

/// The kernel in `affine.metal` that reports the device-side layout.
pub const AFFINE_LAYOUT_KERNEL: &str = "affine_params_layout";

/// What each slot of [`UNARY_LAYOUT_KERNEL`]'s output means.
pub fn expected_unary_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<UnaryParams>(), 8);
    debug_assert_eq!(align_of::<UnaryStridedParams>(), 8);
    debug_assert_eq!(align_of::<Copy2dParams>(), 8);

    vec![
        ("sizeof(UnaryParams)", size_of::<UnaryParams>() as u32),
        ("UnaryParams.dim", offset_of!(UnaryParams, dim) as u32),
        (
            "sizeof(UnaryStridedParams)",
            size_of::<UnaryStridedParams>() as u32,
        ),
        (
            "UnaryStridedParams.dim",
            offset_of!(UnaryStridedParams, dim) as u32,
        ),
        (
            "UnaryStridedParams.num_dims",
            offset_of!(UnaryStridedParams, num_dims) as u32,
        ),
        ("sizeof(Copy2dParams)", size_of::<Copy2dParams>() as u32),
        ("Copy2dParams.d1", offset_of!(Copy2dParams, d1) as u32),
        ("Copy2dParams.d2", offset_of!(Copy2dParams, d2) as u32),
        ("Copy2dParams.src_s", offset_of!(Copy2dParams, src_s) as u32),
        ("Copy2dParams.dst_s", offset_of!(Copy2dParams, dst_s) as u32),
    ]
}

/// What each slot of [`BINARY_LAYOUT_KERNEL`]'s output means.
pub fn expected_binary_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<BinaryParams>(), 8);
    debug_assert_eq!(align_of::<BinaryStridedParams>(), 8);

    vec![
        ("sizeof(BinaryParams)", size_of::<BinaryParams>() as u32),
        ("BinaryParams.dim", offset_of!(BinaryParams, dim) as u32),
        (
            "sizeof(BinaryStridedParams)",
            size_of::<BinaryStridedParams>() as u32,
        ),
        (
            "BinaryStridedParams.dim",
            offset_of!(BinaryStridedParams, dim) as u32,
        ),
        (
            "BinaryStridedParams.num_dims",
            offset_of!(BinaryStridedParams, num_dims) as u32,
        ),
    ]
}

/// What each slot of [`CAST_LAYOUT_KERNEL`]'s output means.
pub fn expected_cast_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<CastParams>(), 8);
    debug_assert_eq!(align_of::<CastStridedParams>(), 8);

    vec![
        ("sizeof(CastParams)", size_of::<CastParams>() as u32),
        ("CastParams.dim", offset_of!(CastParams, dim) as u32),
        (
            "sizeof(CastStridedParams)",
            size_of::<CastStridedParams>() as u32,
        ),
        (
            "CastStridedParams.dim",
            offset_of!(CastStridedParams, dim) as u32,
        ),
        (
            "CastStridedParams.num_dims",
            offset_of!(CastStridedParams, num_dims) as u32,
        ),
    ]
}

/// What each slot of [`AFFINE_LAYOUT_KERNEL`]'s output means.
///
/// This is the one where the numbers are not obvious by inspection, which is
/// the argument for shipping them across the boundary rather than asserting
/// them on each side: `sizeof(AffineParams)` is 16 where its fields sum to 12,
/// and `sizeof(ScaleStridedParams)` is 24 where they sum to 20.
pub fn expected_affine_layout() -> Vec<(&'static str, u32)> {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<AffineParams>(), 8);
    debug_assert_eq!(align_of::<AffineStridedParams>(), 8);
    debug_assert_eq!(align_of::<ScaleParams>(), 8);
    debug_assert_eq!(align_of::<ScaleStridedParams>(), 8);

    vec![
        ("sizeof(AffineParams)", size_of::<AffineParams>() as u32),
        ("AffineParams.dim", offset_of!(AffineParams, dim) as u32),
        ("AffineParams.mul", offset_of!(AffineParams, mul) as u32),
        ("AffineParams.add", offset_of!(AffineParams, add) as u32),
        (
            "sizeof(AffineStridedParams)",
            size_of::<AffineStridedParams>() as u32,
        ),
        (
            "AffineStridedParams.dim",
            offset_of!(AffineStridedParams, dim) as u32,
        ),
        (
            "AffineStridedParams.num_dims",
            offset_of!(AffineStridedParams, num_dims) as u32,
        ),
        (
            "AffineStridedParams.mul",
            offset_of!(AffineStridedParams, mul) as u32,
        ),
        (
            "AffineStridedParams.add",
            offset_of!(AffineStridedParams, add) as u32,
        ),
        ("sizeof(ScaleParams)", size_of::<ScaleParams>() as u32),
        ("ScaleParams.dim", offset_of!(ScaleParams, dim) as u32),
        ("ScaleParams.mul", offset_of!(ScaleParams, mul) as u32),
        (
            "sizeof(ScaleStridedParams)",
            size_of::<ScaleStridedParams>() as u32,
        ),
        (
            "ScaleStridedParams.dim",
            offset_of!(ScaleStridedParams, dim) as u32,
        ),
        (
            "ScaleStridedParams.num_dims",
            offset_of!(ScaleStridedParams, num_dims) as u32,
        ),
        (
            "ScaleStridedParams.mul",
            offset_of!(ScaleStridedParams, mul) as u32,
        ),
    ]
}

// ---------------------------------------------------------------------------
// The binding-style axis itself, shared by every family that carries both.
//
// Introduced for `reduce.metal` (issue #38) and lifted here when the four
// elementwise families (issue #40) and `gemv` (issue #41) followed. One
// declaration rather than one per family: the capture protocol is the same for
// all of them, and copies of it would be the hand-sync `DESIGN.md` §8.1b exists
// to remove.
//
// #40 and #41 were written in parallel against `lloom/integration` and lifted
// the same three items to the same place independently. Converging on one
// location and one spelling is what made that merge a union rather than a
// conflict; the two `kernel_name` variants below are the only part that is
// genuinely per-family, and they differ in what the call site *has* to offer,
// not in what they mean.
// ---------------------------------------------------------------------------

/// How a kernel's scalars reach it.
///
/// `Split` is what candle has always done: one `setBytes` per scalar. `Packed`
/// puts them in a device buffer instead, which is the only form an ICB command
/// can express (`DESIGN.md` §3.7c, issue #38).
///
/// Both are compiled into the same metallib from the same kernel body, so this
/// selects a `[[host_name]]` and nothing more — a compile-tier variant axis in
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
    /// `_packed` is appended after the dtype and any further name segments —
    /// `_strided`, an indexer suffix, or `gemv`'s seven tile parameters —
    /// matching the instantiation macros in every `.metal` file that carries
    /// both styles. The resolution test in `tests.rs` checks both spellings
    /// against the compiled library rather than against each other, which is
    /// `DESIGN.md` §8.1b's argument and what caught a whole family of absent
    /// names during #26.
    ///
    /// Returns `KernelName` rather than a string so the classical path keeps
    /// its `&'static str` and allocates nothing: the pipeline cache is keyed on
    /// this, and it is what the per-token path hits (§15.2 #10).
    pub(crate) fn kernel_name(self, classical: &'static str) -> crate::kernel::KernelName {
        match self {
            ParamStyle::Split => crate::kernel::KernelName::from(classical),
            ParamStyle::Packed => crate::kernel::KernelName::from(packed_name(classical)),
        }
    }

    /// As [`Self::kernel_name`], for the call sites whose classical name is
    /// already an owned `String` rather than a `&'static str`.
    ///
    /// `binary.metal`'s entry points build their name from a dtype and an
    /// indexer suffix at the call site (`MetalStorage::binary`'s `format!`,
    /// which `DESIGN.md` §8.1b names), so there is no `&'static str` to take.
    /// Splitting the two cases here keeps the `&'static str` path allocation-free
    /// rather than making every family pay for the one that cannot be.
    pub(crate) fn kernel_name_owned(self, classical: String) -> crate::kernel::KernelName {
        match self {
            ParamStyle::Split => crate::kernel::KernelName::from(classical),
            ParamStyle::Packed => crate::kernel::KernelName::from(packed_name(&classical)),
        }
    }

    /// The name segment this style appends, for call sites that build their
    /// `[[host_name]]` with `format!` rather than having a name to hand at all.
    ///
    /// `call_mlx_gemv` is one, and it is a third case rather than a variant of
    /// [`Self::kernel_name_owned`]: its name carries seven tile parameters
    /// chosen from the shapes
    /// (`gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`), so the suffix has
    /// to be interpolated *into* the `format!` that builds it rather than
    /// appended to a finished string. Returning the segment rather than a whole
    /// name keeps the one spelling of `_packed` in [`PACKED_SUFFIX`], which is
    /// the point of all three.
    pub(crate) fn name_suffix(self) -> &'static str {
        match self {
            ParamStyle::Split => "",
            ParamStyle::Packed => PACKED_SUFFIX,
        }
    }
}

/// Bind the scalars the following `set_params!` sets, either inline or as one
/// packed buffer.
///
/// The packed block is built by letting that call run exactly as it does on the
/// classical path — `EncoderParam::set_param` diverts each scalar into the
/// capture instead of calling `setBytes` — so the two styles cannot disagree
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
/// the default — and it is why no performance claim is made for `Packed`.
///
/// The returned buffers must outlive the dispatch, so the caller holds them
/// until after the dispatch is encoded rather than dropping them here.
#[must_use = "the staging buffers must outlive the dispatch"]
pub(crate) fn begin_packed_params(
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
pub(crate) fn finish_packed_params(
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

/// The segment appended to a classical `[[host_name]]` to name its packed
/// counterpart.
///
/// Declared once, here, and used by both `ParamStyle::kernel_name` (which
/// selects the pipeline) and `packed_names_resolve` (which checks the name
/// exists). Spelling it in two places would be the same hand-sync `DESIGN.md`
/// §8.1b exists to remove, at a smaller scale.
pub const PACKED_SUFFIX: &str = "_packed";

/// The `_packed` counterpart of a classical `[[host_name]]`.
///
/// A function rather than a table because the rule is uniform: `_packed` is a
/// name segment appended after the dtype and any `_strided`, exactly as those
/// are. What makes that safe to assume rather than merely hope is
/// `packed_names_resolve`, which loads every result against the compiled
/// library — the check `DESIGN.md` §8.1b argues for and that #26 demonstrated
/// the need for.
pub fn packed_name(classical: &str) -> String {
    format!("{classical}{PACKED_SUFFIX}")
}

// ---------------------------------------------------------------------------
// The layout registry (issue #58).
//
// Every family above carries a packed-parameter struct, a `<family>_params_layout`
// kernel that reports the device's view of it, and an `expected_*_layout()`
// giving Rust's. What was missing is anything that makes *checking* them
// mandatory: before this, each family added a const, a function, and a call site
// in `tests.rs`, and the call site is the one that fails silently. A family left
// out of it is not checked, and an unchecked family is indistinguishable from a
// passing one.
//
// # Why an enum and not a slice
//
// A `&[(name, fn)]` slice removes the hand-maintained *call-site list*, which is
// what the issue asks for literally. It does not remove the failure mode: a new
// family can still be omitted from the slice, and the omission is still silent.
// That relocates the defect rather than closing it.
//
// `LayoutFamily` closes it because the `match` in [`LayoutFamily::descriptor`]
// is exhaustive. Adding a variant without an arm is `error[E0004]: non-exhaustive
// patterns`, so a family that fails to register **cannot compile**, which is the
// acceptance criterion stated as a property of the language rather than of
// anyone's diligence. [`LayoutFamily::ALL`] is then the derived list the checker
// iterates, and `layout_registry_covers_every_family` asserts it names every
// variant — the one thing the compiler cannot check, since a missing entry in
// `ALL` is a short array and not a type error.
//
// # Why the slot counts are gone
//
// `LAYOUT_SLOTS`, `CONV_LAYOUT_SLOTS` and the four elementwise ones were each a
// hand-maintained integer that had to agree with the length of an array declared
// six lines below it. The descriptor takes the length from the array instead, so
// the count cannot drift from the thing it counts. The device side still gets
// checked against it: the buffer is sized from `expected.len()` and the kernel
// writes exactly that many slots.
// ---------------------------------------------------------------------------

/// A kernel family that carries packed parameters, and therefore owes a layout
/// check.
///
/// Registration is by *existence of a variant*: [`LayoutFamily::descriptor`]
/// matches exhaustively, so a new family that does not describe itself is a
/// compile error rather than a silently unchecked family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayoutFamily {
    /// `reduce.metal` — issue #38. Reductions, softmax, the two norms, RoPE.
    Reduce,
    /// `unary.metal` — issue #40. Includes `copy2d`, the largest single kernel
    /// in the decode trace (`DESIGN.md` §11.2).
    Unary,
    /// `binary.metal` — issue #40.
    Binary,
    /// `cast.metal` — issue #40.
    Cast,
    /// `affine.metal` — issue #40. The first structs that pad.
    Affine,
    /// `gemv.metal` — issue #41. 183 dispatches per decode token.
    Gemv,
    /// `conv.metal` — issue #42. The mixed-width structs.
    Conv,
    /// `indexing.metal` — issue #81. The last decode-path family, and the one
    /// where the `bool` hazard §11.3b names finally fires.
    Indexing,
}

/// Everything the layout check needs to know about one family.
///
/// `expected` is a `Vec` rather than a fixed-size array because the families
/// have different slot counts (5 to 65) and a trait object over arrays of
/// differing length is not expressible. The cost is one allocation per test
/// dispatch, which is a test-only path and not on the per-token path §15.2 #10
/// governs.
pub struct LayoutDescriptor {
    /// The family, for error messages.
    pub family: LayoutFamily,
    /// Which `.metal` library defines the structs. Each is compiled separately,
    /// so a layout kernel can only see its own file's structs.
    pub source: crate::Source,
    /// The `[[host_name]]` of the kernel that reports the device-side layout.
    pub kernel: &'static str,
    /// What Rust computes for each slot the kernel writes, in order.
    pub expected: Vec<(&'static str, u32)>,
}

impl LayoutDescriptor {
    /// How many `u32` slots this family's layout kernel writes.
    ///
    /// Derived from `expected` rather than declared beside it, so the count and
    /// the thing it counts cannot drift apart.
    pub fn slots(&self) -> usize {
        self.expected.len()
    }
}

impl LayoutFamily {
    /// Every family. The checker iterates this; `layout_registry_covers_every
    /// _family` asserts it is complete.
    ///
    /// Complete coverage of *this* array is the only part the compiler cannot
    /// enforce, which is why it has its own test. The `descriptor` match below
    /// is what the compiler does enforce, and it is the stronger half: a family
    /// can be missing from `ALL` and still be a well-formed program, but it
    /// cannot be missing from `descriptor` at all.
    pub const ALL: &'static [LayoutFamily] = &[
        LayoutFamily::Reduce,
        LayoutFamily::Unary,
        LayoutFamily::Binary,
        LayoutFamily::Cast,
        LayoutFamily::Affine,
        LayoutFamily::Gemv,
        LayoutFamily::Conv,
        LayoutFamily::Indexing,
    ];

    /// The name of the `.metal` file this family lives in, for error messages.
    pub fn metal_file(self) -> &'static str {
        match self {
            LayoutFamily::Reduce => "reduce.metal",
            LayoutFamily::Unary => "unary.metal",
            LayoutFamily::Binary => "binary.metal",
            LayoutFamily::Cast => "cast.metal",
            LayoutFamily::Affine => "affine.metal",
            LayoutFamily::Gemv => "gemv.metal",
            LayoutFamily::Conv => "conv.metal",
            LayoutFamily::Indexing => "indexing.metal",
        }
    }

    /// This family's layout kernel, source, and expected slots.
    ///
    /// **This match is the registration mechanism.** It is exhaustive, so a new
    /// `LayoutFamily` variant without an arm here does not compile — which is
    /// what makes "a family that fails to register is a compile error" true by
    /// construction rather than by convention.
    pub fn descriptor(self) -> LayoutDescriptor {
        let (source, kernel, expected) = match self {
            LayoutFamily::Reduce => (
                crate::Source::Reduce,
                REDUCE_LAYOUT_KERNEL,
                expected_reduce_layout(),
            ),
            LayoutFamily::Unary => (
                crate::Source::Unary,
                UNARY_LAYOUT_KERNEL,
                expected_unary_layout(),
            ),
            LayoutFamily::Binary => (
                crate::Source::Binary,
                BINARY_LAYOUT_KERNEL,
                expected_binary_layout(),
            ),
            LayoutFamily::Cast => (
                crate::Source::Cast,
                CAST_LAYOUT_KERNEL,
                expected_cast_layout(),
            ),
            LayoutFamily::Affine => (
                crate::Source::Affine,
                AFFINE_LAYOUT_KERNEL,
                expected_affine_layout(),
            ),
            LayoutFamily::Gemv => (
                crate::Source::Gemv,
                GEMV_LAYOUT_KERNEL,
                expected_gemv_layout(),
            ),
            LayoutFamily::Conv => (
                crate::Source::Conv,
                CONV_LAYOUT_KERNEL,
                expected_conv_layout(),
            ),
            LayoutFamily::Indexing => (
                crate::Source::Indexing,
                INDEXING_LAYOUT_KERNEL,
                expected_indexing_layout(),
            ),
        };
        LayoutDescriptor {
            family: self,
            source,
            kernel,
            expected,
        }
    }
}
