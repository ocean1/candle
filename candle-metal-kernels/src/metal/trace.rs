//! Per-dispatch recording of the Metal command stream.
//!
//! Answers "is the decode dispatch sequence identical across tokens?" — the
//! question that gates whether an `MTLIndirectCommandBuffer` can be encoded once
//! and replayed, since an ICB is only replayable if the command sequence is
//! identical between invocations.
//!
//! Off unless `CANDLE_METAL_TRACE=1`, checked once into a `OnceLock`. When off
//! the cost is one relaxed atomic load per dispatch and nothing else: no
//! allocation, no lock, no formatting. That matters because the trace has to be
//! able to run in a release build against the real model — an instrumented
//! debug build would measure a different dispatch pattern than the one shipped.
//!
//! # Buffer identity
//!
//! Raw `MTLBuffer` pointers are recorded through a stable interning table rather
//! than verbatim. Addresses are meaningless across processes and would make
//! every run's trace differ for no reason; the interned id is assigned in order
//! of first use, so two runs that allocate the same logical buffers in the same
//! order produce identical traces. That is exactly the distinction the ICB
//! question needs — "a different buffer" has to be separable from "the same
//! buffer at a different address".
//!
//! The raw address is emitted alongside the id, since the allocator question in
//! `DESIGN.md` §9 wants to know whether a logical slot is backed by a stable
//! allocation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One buffer binding recorded against the dispatch that consumed it.
#[derive(Clone, PartialEq, Eq)]
pub struct Binding {
    /// Argument-table index the buffer was bound at.
    pub index: usize,
    /// Interned buffer identity, stable within a process.
    pub buffer_id: u64,
    /// Raw `MTLBuffer` address, for the allocator-stability question.
    pub buffer_addr: usize,
    /// Byte offset into the buffer.
    pub offset: usize,
    /// Whether the binding was made as an input or an output.
    pub is_output: bool,
}

/// A single `dispatchThreads`/`dispatchThreadgroups` call and its bound state.
#[derive(Clone)]
pub struct Dispatch {
    /// Monotonic index across the whole trace.
    pub seq: u64,
    /// Kernel name from the pipeline that was current at dispatch time.
    pub pipeline: String,
    /// `(width, height, depth)` of the grid, in the unit the dispatch used.
    pub grid: (usize, usize, usize),
    /// `(width, height, depth)` threads per threadgroup.
    pub threadgroup: (usize, usize, usize),
    /// True when dispatched as threadgroups rather than threads. The two are
    /// different units, so a trace that conflated them could report a false
    /// difference between two identical dispatches.
    pub by_threadgroups: bool,
    /// Buffers bound since the previous dispatch, in binding order.
    pub bindings: Vec<Binding>,
    /// Caller-supplied marker this dispatch falls under, e.g. a decode step.
    pub region: Option<String>,
}

impl Dispatch {
    /// The part of a dispatch that must match for an ICB slot to be replayable:
    /// same kernel, same grid, same bindings. Excludes `seq` and `region`, which
    /// are position labels rather than command content.
    fn signature_parts(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(format!(
            "{} grid={:?} tg={:?} tgmode={}",
            self.pipeline, self.grid, self.threadgroup, self.by_threadgroups
        ))
        .chain(self.bindings.iter().map(|b| {
            format!(
                "  [{}] {} buf#{} off={}",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.buffer_id,
                b.offset
            )
        }))
    }

    /// Full command content, including buffer identity. Two dispatches with the
    /// same signature are replayable from one ICB slot.
    pub fn signature(&self) -> String {
        self.signature_parts().collect::<Vec<_>>().join("\n")
    }

    /// Command content with buffer identity erased, keeping offsets and the
    /// input/output role.
    ///
    /// Separating this from `signature` is what distinguishes the issue's second
    /// outcome ("differs only in buffer identity" — an allocator problem, fixable)
    /// from its third ("differs structurally" — the §14.4 trigger). Collapsing
    /// them into one comparison would report the harder verdict for the easier
    /// fault.
    pub fn shape_signature(&self) -> String {
        std::iter::once(format!(
            "{} grid={:?} tg={:?} tgmode={}",
            self.pipeline, self.grid, self.threadgroup, self.by_threadgroups
        ))
        .chain(self.bindings.iter().map(|b| {
            format!(
                "  [{}] {} off={}",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.offset
            )
        }))
        .collect::<Vec<_>>()
        .join("\n")
    }

    /// Kernel name and binding slots only — no grid, no offsets, no buffers.
    ///
    /// Separates "the op sequence changed" from "the same ops ran over more
    /// data". Grid size is a dispatch-tier parameter (`DESIGN.md` §7.1), so a
    /// sequence whose kernels are fixed and whose grids merely scale with
    /// `kv_len` is a materially different finding — and a much more recoverable
    /// one — than a sequence whose kernels differ.
    pub fn kernel_signature(&self) -> String {
        std::iter::once(self.pipeline.clone())
            .chain(self.bindings.iter().map(|b| {
                format!(
                    "  [{}] {}",
                    b.index,
                    if b.is_output { "out" } else { "in " }
                )
            }))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

struct TraceState {
    dispatches: Vec<Dispatch>,
    /// Bindings accumulated since the last dispatch.
    pending: Vec<Binding>,
    /// Pipeline name set by the most recent `set_compute_pipeline_state`.
    current_pipeline: Option<String>,
    /// Buffer address -> interned id.
    buffer_ids: HashMap<usize, u64>,
    next_buffer_id: u64,
    region: Option<String>,
    next_seq: u64,
}

impl TraceState {
    fn new() -> Self {
        TraceState {
            dispatches: Vec::new(),
            pending: Vec::new(),
            current_pipeline: None,
            buffer_ids: HashMap::new(),
            next_buffer_id: 0,
            region: None,
            next_seq: 0,
        }
    }

    fn intern(&mut self, addr: usize) -> u64 {
        if let Some(id) = self.buffer_ids.get(&addr) {
            return *id;
        }
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffer_ids.insert(addr, id);
        id
    }
}

fn state() -> &'static Mutex<TraceState> {
    static STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(TraceState::new()))
}

/// Whether tracing was requested via `CANDLE_METAL_TRACE`.
///
/// Read once. A per-dispatch `std::env::var` would be a lock and an allocation
/// on the hot path, which is precisely the kind of measurement-tool cost
/// `DESIGN.md` §2.4 warns about.
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CANDLE_METAL_TRACE")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false)
    })
}

/// Set independently of `enabled()` so recording can be gated to a region of
/// interest — tracing every prefill dispatch of a 700-token prompt to answer a
/// question about decode would bury the answer.
static RECORDING: AtomicBool = AtomicBool::new(false);

/// Dispatches observed while recording was off, so the trace can report what it
/// did not see rather than implying it saw everything.
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// True when a dispatch should be recorded.
#[inline]
pub fn is_recording() -> bool {
    enabled() && RECORDING.load(Ordering::Relaxed)
}

/// Turn recording on or off. No-op unless `CANDLE_METAL_TRACE` is set.
pub fn set_recording(on: bool) {
    if enabled() {
        RECORDING.store(on, Ordering::Relaxed);
    }
}

/// Label subsequent dispatches, e.g. with a decode step index.
pub fn set_region(region: Option<String>) {
    if !enabled() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.region = region;
    }
}

/// Record the pipeline that later dispatches will run.
#[inline]
pub fn record_pipeline(name: &str) {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.current_pipeline = Some(name.to_string());
    }
}

/// Record a buffer binding against the dispatch that will consume it.
#[inline]
pub fn record_binding(index: usize, addr: usize, offset: usize, is_output: bool) {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        let buffer_id = s.intern(addr);
        s.pending.push(Binding {
            index,
            buffer_id,
            buffer_addr: addr,
            offset,
            is_output,
        });
    }
}

/// Record a dispatch, consuming the bindings accumulated since the last one.
///
/// Bindings are cleared even when not recording, so that turning recording on
/// mid-stream does not attribute a previous dispatch's bindings to the first
/// recorded one.
#[inline]
pub fn record_dispatch(
    grid: (usize, usize, usize),
    threadgroup: (usize, usize, usize),
    by_threadgroups: bool,
) {
    if !enabled() {
        return;
    }
    if !RECORDING.load(Ordering::Relaxed) {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut s) = state().lock() {
            s.pending.clear();
        }
        return;
    }
    if let Ok(mut s) = state().lock() {
        let seq = s.next_seq;
        s.next_seq += 1;
        let bindings = std::mem::take(&mut s.pending);
        let pipeline = s
            .current_pipeline
            .clone()
            // A dispatch with no pipeline recorded would be a gap in the trace,
            // so it is named rather than dropped.
            .unwrap_or_else(|| "<unknown>".to_string());
        let region = s.region.clone();
        s.dispatches.push(Dispatch {
            seq,
            pipeline,
            grid,
            threadgroup,
            by_threadgroups,
            bindings,
            region,
        });
    }
}

/// Every dispatch recorded so far.
pub fn take_dispatches() -> Vec<Dispatch> {
    if !enabled() {
        return Vec::new();
    }
    match state().lock() {
        Ok(mut s) => std::mem::take(&mut s.dispatches),
        Err(_) => Vec::new(),
    }
}

/// Dispatches seen while recording was off.
pub fn skipped_count() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// Whether `CANDLE_METAL_TRACE` requested tracing, for a harness to report when
/// it produced nothing because the variable was unset.
pub fn trace_requested() -> bool {
    enabled()
}
