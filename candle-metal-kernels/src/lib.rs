pub mod err;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

pub use err::MetalKernelError;
pub use kernel::Kernels;
// `ParamStyle` used to arrive via `reduce::*`, where it was declared when
// `reduce.metal` was the only family carrying both binding styles. It moved to
// `kernels::params` when `conv.metal` followed (issue #42), so it is named
// explicitly here; the spelling callers use is unchanged.
pub use kernels::{
    affine::*, arena_alloc::*, call_binary_contiguous, call_binary_contiguous_with,
    call_binary_strided, call_binary_strided_with, call_mlx_gemm, call_mlx_gemv_with, cast::*,
    convolution::*, fill::*, indexing::*, quantized::*, random::*, reduce::*, sdpa::*, sort::*,
    ternary::*, unary, unary::*, ConvKernel, GemmDType, GgmlDType, IndexingKernel, ParamStyle,
    ReduceKernel,
};
// The arena's GPU-side allocator vocabulary (`DESIGN.md` §9.2d, issue #70).
// Named rather than arriving through a glob, because `ARENA_DECLINED` is a
// cross-language constant -- `arena_alloc.metal` writes the same sentinel, and
// `arena_alloc_reports_alignment` is what checks the two agree.
pub use metal::{ArenaCursor, ArenaOffsets, ARENA_DECLINED};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function,
    Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use source::Source;
use utils::{get_block_dims, get_tile_size, linear_split, EncoderParam, EncoderProvider};
pub use utils::{BufferOffset, Output};

pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[cfg(test)]
mod tests;
