use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Kernel function name, carried so a dispatch can be attributed to a kernel
    /// when profiling (`profile`). `MTLComputePipelineState` has a `label`, but
    /// candle never sets it and reading it back through the Obj-C bridge on a hot
    /// path would cost more than keeping the name we already had at load time.
    ///
    /// `Arc<str>` rather than `String` because pipelines are cloned out of the
    /// cache on every kernel call; cloning a refcount is free where cloning the
    /// name would not be.
    name: Option<Arc<str>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> ComputePipeline {
        ComputePipeline { raw, name: None }
    }

    /// Attach the kernel function name this pipeline was built from.
    pub fn with_name(mut self, name: impl AsRef<str>) -> ComputePipeline {
        self.name = Some(Arc::from(name.as_ref()));
        self
    }

    /// The kernel function name, if it was recorded at load time.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
