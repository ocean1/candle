//! How a dispatch reaches the GPU.
//!
//! `DESIGN.md` §11.1 asks for the classical command-buffer path and an ICB path
//! behind one interface, so the two can be A/B'd on the same workload. This is
//! that interface. The classical path is the default and the correctness bar;
//! it is also what already worked, so the trait is interposed in front of
//! working code rather than duplicating it.
//!
//! # Where the seam is, and why here
//!
//! Every dispatch in the crate funnels through
//! [`ComputeCommandEncoder::dispatch_thread_groups`] or `dispatch_threads` --
//! verified, no kernel reaches `MTLComputeCommandEncoder` directly. So a trait
//! at the point of *submission* sees every dispatch without touching any of the
//! 51 `call_*` entry points or the 57 sites that set pipeline state. That
//! matters more than elegance: a seam that required editing every kernel would
//! have to be re-reviewed against every kernel, and `DESIGN.md` §11.1's "must
//! not regress" is much harder to argue when the diff is that wide.
//!
//! # What an executor may and may not assume
//!
//! An executor sees dispatches in submission order and may record, forward, or
//! defer them. It may **not** change what a dispatch computes: bindings are
//! already on the encoder by the time [`Executor::dispatch`] is called, because
//! `set_params!` binds as it walks its argument list. An executor that wants to
//! replay must therefore capture bindings as they happen, which is what
//! [`Executor::will_bind_buffer`] exists for -- see `IcbFeasibility` below for
//! why no such executor is included here.
use crate::metal::{Buffer, ComputePipeline};
use objc2_metal::MTLSize;

/// One dispatch, as the executor sees it.
///
/// Deliberately not a borrow of the encoder: an executor that records for
/// replay needs these values to outlive the call, and one that forwards
/// immediately does not care either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchRecord {
    /// Kernel function name, when the pipeline carried one (it does whenever
    /// the pipeline came through `Kernels::load_pipeline`).
    pub kernel: Option<std::sync::Arc<str>>,
    /// Threadgroups per grid, or threads per grid for a `dispatch_threads`.
    pub grid: Grid,
    pub threads_per_threadgroup: Size3,
}

/// Which of Metal's two dispatch forms a record came from.
///
/// Kept distinct rather than normalised to threadgroups because the conversion
/// is lossy: `dispatchThreads` lets Metal size the final partial threadgroup,
/// and a replay that rounded it to threadgroups would compute a different
/// number of threads for any grid that is not a multiple of the threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    Threadgroups(Size3),
    Threads(Size3),
}

/// `MTLSize` without the Metal dependency, so a record is comparable and
/// hashable. `MTLSize` is three `NSUInteger`s and derives neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Size3 {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl From<MTLSize> for Size3 {
    fn from(s: MTLSize) -> Self {
        Size3 {
            width: s.width,
            height: s.height,
            depth: s.depth,
        }
    }
}

/// How a dispatch is submitted to the GPU.
///
/// Implementors are consulted on the hot path -- 675 times per LFM2 decode
/// token (`DESIGN.md` §11.2) -- so every method has a default that does
/// nothing. An executor that only wants to count dispatches pays for nothing
/// else, and adding a method later does not break implementors.
///
/// `DESIGN.md` §15.2 #9 forbids `dyn Trait` on the per-token path. This trait
/// is deliberately shaped so the default path never dispatches dynamically:
/// [`Classical`] is a zero-sized type whose methods are empty, and
/// [`ExecutorSlot`] resolves to it by a branch on an enum rather than a vtable.
pub trait Executor: Send + Sync {
    /// Called before a dispatch is forwarded to the encoder.
    ///
    /// Returning `false` suppresses the underlying dispatch, which is what a
    /// record-only executor wants. Returning `true` (the default) leaves the
    /// classical behaviour exactly as it was.
    fn dispatch(&self, _record: &DispatchRecord) -> bool {
        true
    }

    /// Called when a buffer is bound, before the bind reaches Metal.
    ///
    /// A replaying executor needs this because bindings are applied as
    /// `set_params!` walks its arguments, so by `dispatch` time they are on the
    /// encoder and no longer enumerable.
    fn will_bind_buffer(&self, _index: usize, _buffer: &Buffer, _offset: usize, _is_output: bool) {}

    /// Called when pipeline state is set, before it reaches Metal.
    fn will_set_pipeline(&self, _pipeline: &ComputePipeline) {}
}

/// The default: forward every dispatch to the command encoder unchanged.
///
/// Zero-sized, and every method is the trait default, so selecting this costs
/// one branch on an enum discriminant and nothing else. This is what makes
/// "the classical path must not regress" (`DESIGN.md` §11.1) a structural
/// property rather than a measured hope -- there is no other code on it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Classical;

impl Executor for Classical {}

/// Which executor a device is using.
///
/// An enum rather than `Box<dyn Executor>` because of §15.2 #9: the decode path
/// must not pay a virtual call per dispatch. The cost of an executor being
/// installed is a discriminant test that predicts perfectly, and `Classical`
/// falls through to the same code that ran before this module existed.
///
/// Not `#[non_exhaustive]`: an ICB variant will be added here when one is
/// buildable (see [`IcbFeasibility`]), and making that addition visible to
/// downstream matches is the point.
#[derive(Default)]
pub enum ExecutorSlot {
    /// Classical command buffers. The default, and the A/B baseline.
    #[default]
    Classical,
    /// An executor supplied by a caller, for recording or experiment.
    ///
    /// This one *is* a dynamic dispatch and is intended for measurement
    /// harnesses, not for the decode path. It is behind the same enum so that
    /// installing one cannot accidentally become the default.
    Custom(std::sync::Arc<dyn Executor>),
}

impl ExecutorSlot {
    /// True when no executor is installed, so callers can skip building a
    /// [`DispatchRecord`] at all.
    ///
    /// Building a record costs an `Arc<str>` clone and three struct copies.
    /// That is small, but it is per-dispatch on a path where `DESIGN.md` §6.4a
    /// measured the whole per-bind fence probe at 5.1 % of non-GPU time, so
    /// "small and per-dispatch" is exactly the shape worth not paying by
    /// default.
    #[inline(always)]
    pub fn is_classical(&self) -> bool {
        matches!(self, ExecutorSlot::Classical)
    }

    #[inline(always)]
    pub fn dispatch(&self, record: &DispatchRecord) -> bool {
        match self {
            ExecutorSlot::Classical => true,
            ExecutorSlot::Custom(e) => e.dispatch(record),
        }
    }

    #[inline(always)]
    pub fn will_bind_buffer(&self, index: usize, buffer: &Buffer, offset: usize, is_output: bool) {
        match self {
            ExecutorSlot::Classical => {}
            ExecutorSlot::Custom(e) => e.will_bind_buffer(index, buffer, offset, is_output),
        }
    }

    #[inline(always)]
    pub fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        match self {
            ExecutorSlot::Classical => {}
            ExecutorSlot::Custom(e) => e.will_set_pipeline(pipeline),
        }
    }
}

impl std::fmt::Debug for ExecutorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorSlot::Classical => f.write_str("Classical"),
            ExecutorSlot::Custom(_) => f.write_str("Custom"),
        }
    }
}

/// Why the ICB path is not implemented here, recorded at the seam it would
/// occupy so it is not rediscovered.
///
/// `DESIGN.md` §11.1a and issue #32 establish that GPU-side ICB encoding works
/// on this machine, including a GPU-written execution range, and treat the
/// remaining obstacles as buffer identity (fixed by issues #21/#23) and
/// `kv_len`-dependent grids. There is a third, and it is not mentioned in
/// either: **an ICB command cannot carry an inline constant.**
///
/// `MTLIndirectComputeCommand` is 11 methods, enumerated from the Objective-C
/// runtime rather than read from a header:
///
/// ```text
/// setComputePipelineState:                        setBarrier
/// setKernelBuffer:offset:atIndex:                 clearBarrier
/// setKernelBuffer:offset:attributeStride:atIndex: setImageblockWidth:height:
/// concurrentDispatchThreadgroups:...              setStageInRegion:
/// concurrentDispatchThreads:...                   setThreadgroupMemoryLength:atIndex:
/// reset
/// ```
///
/// There is no `setBytes` in any form; `respondsToSelector:` is `NO` for both
/// `setKernelBytes:length:atIndex:` and `setBytes:length:atIndex:`. The same
/// holds for the render-side protocol, so this is Metal's design rather than an
/// omission.
///
/// Candle binds inline constants at essentially every dispatch -- 56
/// `set_params!` sites, and `EncoderParam` lowers every primitive to
/// `setBytes`. `call_rms_norm` is the clearest case: 77 of the 675 dispatches
/// in a decode token, passing `length`, `elements_to_sum` and `eps` inline.
/// None of the three is expressible in an ICB command.
///
/// `inheritBuffers` does not rescue it: it makes commands inherit the parent
/// encoder's bindings *identically*, so it cannot express constants that differ
/// per dispatch, which is the only reason to want them.
///
/// So an ICB path requires promoting every inline constant into a device
/// buffer -- changing every kernel signature and every `.metal` source that
/// reads one. That is the invasive, cross-cutting refactor `DESIGN.md` §8.1b
/// records upstream declining, and it is a prerequisite for the ICB path
/// rather than part of it. It is recorded here, and in `DESIGN.md` §11.3a, as
/// the next thing that must land before this enum grows an `Icb` variant.
#[derive(Debug)]
pub struct IcbFeasibility;
