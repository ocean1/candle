use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Kernel name this pipeline was built from.
    ///
    /// `MTLComputePipelineState` carries no readable function name, and candle
    /// never sets a label, so the name is captured where it is already known —
    /// at pipeline creation — rather than reconstructed at dispatch time. Held
    /// as `Arc<str>` because pipelines are cloned out of the pipeline cache on
    /// every kernel call and the name would otherwise be reallocated each time.
    name: Arc<str>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        name: impl Into<Arc<str>>,
    ) -> ComputePipeline {
        ComputePipeline {
            raw,
            name: name.into(),
        }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
