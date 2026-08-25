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
