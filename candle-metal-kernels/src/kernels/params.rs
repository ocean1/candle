//! Packed parameter structs for `reduce.metal`, and the layout check that keeps
//! them honest.
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

/// The kernel in `reduce.metal` that reports the device-side layout.
pub const LAYOUT_KERNEL: &str = "reduce_params_layout";

/// How many `u32` slots [`LAYOUT_KERNEL`] writes.
pub const LAYOUT_SLOTS: usize = 26;

/// The kernel in `gemv.metal` that reports the device-side layout.
///
/// A second kernel rather than more slots on [`LAYOUT_KERNEL`]: the layout
/// check has to load from the library that actually defines the struct, and
/// `gemv.metal` and `reduce.metal` are separate `Source`s compiled
/// independently (`kernel.rs:109`). Keeping them separate is also what lets the
/// families compose rather than share one fixed slot count.
pub const GEMV_LAYOUT_KERNEL: &str = "gemv_params_layout";

/// How many `u32` slots [`GEMV_LAYOUT_KERNEL`] writes.
pub const GEMV_LAYOUT_SLOTS: usize = 8;

/// What each slot of [`GEMV_LAYOUT_KERNEL`] means, and what Rust computes for
/// it.
///
/// As [`expected_layout`], for `gemv.metal`. Written as data rather than as a
/// sequence of assertions so a mismatch reports *which* field disagrees — the
/// whole point being that one wrong offset is otherwise invisible: the kernel
/// reads a well-formed number from the wrong place and computes a plausible
/// wrong answer (`DESIGN.md` §3.5, §15.1).
pub fn gemv_expected_layout() -> [(&'static str, u32); GEMV_LAYOUT_SLOTS] {
    use core::mem::{align_of, offset_of, size_of};

    debug_assert_eq!(align_of::<GemvParams>(), 4);

    [
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
// The binding-style axis itself, shared by every family that carries both.
//
// Introduced for `reduce.metal` (issue #38) and lifted here when `gemv`
// followed (issue #41). One declaration rather than one per family: the capture
// protocol is the same for all of them, and copies of it would be the hand-sync
// `DESIGN.md` §8.1b exists to remove.
//
// Issue #40 lifts the same three items to the same place for the four
// elementwise families. The two changes were written in parallel against
// `lloom/integration`; converging on one location and one spelling is
// deliberate, so the merge is a union rather than a conflict.
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
    /// `_packed` is appended after the dtype and any further name segments,
    /// matching the instantiation macros in every `.metal` file that carries
    /// both styles. The resolution test in `tests.rs` checks both spellings
    /// against the compiled library rather than against each other, which is
    /// `DESIGN.md` §8.1b's argument and what caught a whole family of absent
    /// names during #26.
    ///
    /// Returns `KernelName` rather than a string so the classical path keeps
    /// its `&'static str` and allocates nothing: the pipeline cache is keyed on
    /// this, and it is what the per-token path hits (§15.2 #10).
    #[allow(dead_code)]
    pub(crate) fn kernel_name(self, classical: &'static str) -> crate::kernel::KernelName {
        match self {
            ParamStyle::Split => crate::kernel::KernelName::from(classical),
            ParamStyle::Packed => crate::kernel::KernelName::from(packed_name(classical)),
        }
    }

    /// The name segment this style appends, for call sites that build their
    /// `[[host_name]]` with `format!` rather than having a `&'static str`.
    ///
    /// `call_mlx_gemv` is one: its name carries seven tile parameters chosen
    /// from the shapes (`gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`), so
    /// there is no static string to append to. Returning the segment rather
    /// than a whole name keeps the one spelling of `_packed` in
    /// [`PACKED_SUFFIX`], which is the point.
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
