//! Packed parameter structs for the kernel families that carry both binding
//! styles, and the layout checks that keep them honest.
//!
//! `reduce.metal` came first (issue #38); `unary`, `binary`, `cast` and
//! `affine` followed (issue #40). Each family declares its own structs and its
//! own layout kernel, because each `.metal` file is compiled into its own
//! library -- a kernel in one cannot see a struct in another.
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

use crate::{Buffer, ComputeCommandEncoder, Device, MetalKernelError, RESOURCE_OPTIONS};

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

/// The kernel in `reduce.metal` that reports the device-side layout.
pub const LAYOUT_KERNEL: &str = "reduce_params_layout";

/// How many `u32` slots [`LAYOUT_KERNEL`] writes.
pub const LAYOUT_SLOTS: usize = 26;

/// What each slot of [`LAYOUT_KERNEL`]'s output means, and what Rust computes
/// for it.
///
/// The ordering is declared here and in the kernel, and
/// `reduce_params_layout_matches_metal` is what fails if they drift. Written as
/// data rather than as a sequence of assertions so a mismatch reports *which*
/// field disagrees rather than just that something did — the whole point being
/// that a single wrong offset is otherwise invisible.
pub fn expected_layout() -> [(&'static str, u32); LAYOUT_SLOTS] {
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

    [
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
/// How many `u32` slots [`UNARY_LAYOUT_KERNEL`] writes.
pub const UNARY_LAYOUT_SLOTS: usize = 10;

/// The kernel in `binary.metal` that reports the device-side layout.
pub const BINARY_LAYOUT_KERNEL: &str = "binary_params_layout";
/// How many `u32` slots [`BINARY_LAYOUT_KERNEL`] writes.
pub const BINARY_LAYOUT_SLOTS: usize = 5;

/// The kernel in `cast.metal` that reports the device-side layout.
pub const CAST_LAYOUT_KERNEL: &str = "cast_params_layout";
/// How many `u32` slots [`CAST_LAYOUT_KERNEL`] writes.
pub const CAST_LAYOUT_SLOTS: usize = 5;

/// The kernel in `affine.metal` that reports the device-side layout.
pub const AFFINE_LAYOUT_KERNEL: &str = "affine_params_layout";
/// How many `u32` slots [`AFFINE_LAYOUT_KERNEL`] writes.
pub const AFFINE_LAYOUT_SLOTS: usize = 16;

/// What each slot of [`UNARY_LAYOUT_KERNEL`]'s output means.
pub fn expected_unary_layout() -> [(&'static str, u32); UNARY_LAYOUT_SLOTS] {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<UnaryParams>(), 8);
    debug_assert_eq!(align_of::<UnaryStridedParams>(), 8);
    debug_assert_eq!(align_of::<Copy2dParams>(), 8);

    [
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
pub fn expected_binary_layout() -> [(&'static str, u32); BINARY_LAYOUT_SLOTS] {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<BinaryParams>(), 8);
    debug_assert_eq!(align_of::<BinaryStridedParams>(), 8);

    [
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
pub fn expected_cast_layout() -> [(&'static str, u32); CAST_LAYOUT_SLOTS] {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<CastParams>(), 8);
    debug_assert_eq!(align_of::<CastStridedParams>(), 8);

    [
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
pub fn expected_affine_layout() -> [(&'static str, u32); AFFINE_LAYOUT_SLOTS] {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<AffineParams>(), 8);
    debug_assert_eq!(align_of::<AffineStridedParams>(), 8);
    debug_assert_eq!(align_of::<ScaleParams>(), 8);
    debug_assert_eq!(align_of::<ScaleStridedParams>(), 8);

    [
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
// Introduced for `reduce.metal` (issue #38) and lifted here when `unary`,
// `binary`, `cast` and `affine` followed (issue #40). One declaration rather
// than five: the capture protocol is the same for all of them, and five copies
// of it would be the hand-sync `DESIGN.md` §8.1b exists to remove.
// ---------------------------------------------------------------------------

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
    /// `_packed` is appended after the dtype and any `_strided` or indexer
    /// suffix, matching the `init_*` macros in every `.metal` file that carries
    /// both styles. The resolution test in `tests.rs`
    /// checks both spellings against the compiled library rather than against
    /// each other, which is `DESIGN.md` §8.1b's argument and what caught a
    /// whole family of absent names during #26.
    ///
    /// Returns `KernelName` rather than a string so the classical path keeps
    /// its `&'static str` and allocates nothing: the pipeline cache is keyed on
    /// this, and it is what the per-token path hits (§15.2 #10).
    pub(crate) fn kernel_name(self, classical: &'static str) -> crate::kernel::KernelName {
        match self {
            ParamStyle::Split => crate::kernel::KernelName::from(classical),
            ParamStyle::Packed => {
                crate::kernel::KernelName::from(crate::kernels::params::packed_name(classical))
            }
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
            ParamStyle::Packed => {
                crate::kernel::KernelName::from(crate::kernels::params::packed_name(&classical))
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
