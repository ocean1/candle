//! Where does an LFM2 decode token's time actually go? CPU or GPU, and why.
//!
//! Measurement harness for lloom issue #7 / `DESIGN.md` §3.4, §6.6, §11.2,
//! §16 P0 #1 and #4.
//!
//! Three numbers, measured in one run so they describe the same execution:
//!
//! * **wall time per decode token** — measured independently rather than
//!   inherited from the ~27 ms/token recorded in `DESIGN.md` §6.6.
//! * **GPU busy time per decode token** — the union of every command buffer's
//!   `GPUStartTime`..`GPUEndTime` in the window, via `CANDLE_METAL_PROFILE=1`.
//! * **dispatch count per decode token** — counted, not inherited from §11.2's
//!   ~240 estimate.
//!
//! `wall - gpu_busy` is the time the GPU was not executing. In a serial decode
//! loop with nothing to overlap against, that is CPU-side encode, submission and
//! readback: the answer to §16 P0 #1.
//!
//! # Why prefill is measured separately
//!
//! Prefill runs one forward pass over the whole prompt and decode runs one per
//! token. Averaging them together makes ms/token a function of prompt length,
//! which is how a decode figure ends up describing something else. The prompt's
//! forward pass is timed into its own bucket and excluded from the decode
//! average.
//!
//! # Why a warmup token is excluded
//!
//! The first decode token pays for pipeline compilation, buffer-pool growth and
//! page faults on freshly mmap'd weights. Those are real costs, but they are
//! one-time, and a steady-state figure is what the roofline in §16 P0 #4 is
//! being compared against. `--warmup` tokens are timed and reported, then left
//! out of the steady-state average.
//!
//! ```bash
//! CANDLE_METAL_PROFILE=1 cargo run --release --features metal \
//!   --example lfm2-decode-profile -- --n 200
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::lfm2::{Cache, Config, ConvState, LayerType, Lfm2Config, Model};
use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// The serialized run record (lloom issue #171).
///
/// Written **at exit**, beside the `RESULT` line. The only thing it does inside
/// a timed window is stamp each finished token with `CLOCK_UPTIME_RAW`, which is
/// what lets the memory timeline be plotted against token index instead of
/// against a reconstruction.
mod run_record;

/// One `sysctl -n` read, for the machine fields of the run record.
fn sysctl_str(key: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The 1-minute load average.
///
/// Recorded at the start **and** the end of a run, because one reading cannot
/// show a build starting midway through -- `lloom-probe`'s `Conditions` makes
/// the same split for the same reason. `.bench/README.md` §1.2: a number taken
/// on this machine without stating contention is not interpretable, and #40's
/// null control read **+65.8 %** with two other agents building.
fn load_average() -> Option<f64> {
    extern "C" {
        fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
    }
    let mut a = [0.0f64; 3];
    // SAFETY: `getloadavg` writes at most `nelem` doubles into the buffer, and
    // we pass the true length of a stack array we own.
    let n = unsafe { getloadavg(a.as_mut_ptr(), 3) };
    (n > 0).then_some(a[0])
}

/// Access to the Metal profiling counters, with a no-op stand-in off Metal.
///
/// The shim keeps the call sites unconditional: without it every snapshot and
/// reset would need its own `#[cfg]`, and a `cfg`-heavy measurement loop is
/// exactly where a "which build did this number come from?" mistake hides.
#[cfg(feature = "metal")]
mod profile {
    pub use candle::metal_backend::profile::{enabled, reset, snapshot};
}

#[cfg(not(feature = "metal"))]
mod profile {
    #[derive(Clone, Debug, Default)]
    pub struct Snapshot {
        pub dispatches: u64,
        pub encoders: u64,
        pub command_buffers: u64,
        pub timed_command_buffers: u64,
        pub gpu_busy_sum_s: f64,
        pub gpu_busy_union_s: f64,
        pub gpu_span_s: f64,
        pub by_label: Vec<(String, u64)>,
    }
    pub fn enabled() -> bool {
        false
    }
    pub fn reset() {}
    pub fn snapshot() -> Snapshot {
        Snapshot::default()
    }
}

/// The resolved variant configuration for one run (`DESIGN.md` §7.1, #105).
///
/// Parsed and validated once, before the clock starts, so an unsatisfiable
/// combination fails immediately rather than after a five-minute measurement.
/// `announce` prints it, because a timing figure whose configuration is not
/// stated is the defect #105 exists to remove: today's `.bench/` rows are
/// all-defaults and nothing says so.
struct Axes {
    /// Whether the arena is installed at all.
    arena: bool,
    /// Decode steps recorded on the pool before the arena is installed.
    record_steps: usize,
    #[cfg(feature = "metal")]
    layout: candle::metal_backend::ArenaLayout,
    /// #70's GPU bump allocator, rather than #69's CPU plan.
    gpu_offsets: bool,
    /// The attention arm, echoed for the run line.
    attn: String,
    /// The KV-append arm, echoed for the run line (issue #142).
    kv_append: String,
    /// The conv-state arm (#141), echoed for the run line.
    conv_state: ConvState,
}

impl Axes {
    fn resolve(args: &Args, _device: &Device) -> Result<Self> {
        #[cfg(feature = "metal")]
        let layout = match args.arena_layout.as_str() {
            "packed" => candle::metal_backend::ArenaLayout::Packed,
            "non-aliasing" | "nonaliasing" | "reference" => {
                candle::metal_backend::ArenaLayout::NonAliasing
            }
            other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
        };

        let gpu_offsets = match args.arena_offsets.as_str() {
            "cpu" => false,
            "gpu" => true,
            other => anyhow::bail!("unknown --arena-offsets {other:?}; want cpu or gpu"),
        };
        if gpu_offsets {
            if !args.arena {
                anyhow::bail!(
                    "--arena-offsets gpu needs --arena: there is nothing to allocate over"
                );
            }
            // Refused rather than downgraded (#70, §9.2g). A packed plan reuses
            // slots, so its offsets are not monotone and a forward-only cursor
            // cannot reproduce them; falling back to the CPU here would report
            // #69's numbers under this flag's name.
            #[cfg(feature = "metal")]
            if layout != candle::metal_backend::ArenaLayout::NonAliasing {
                anyhow::bail!(
                    "--arena-offsets gpu needs --arena-layout non-aliasing: a packed plan reuses \
                     slots, so its offsets are not monotone and no bump allocator can reproduce \
                     them"
                );
            }
        }

        // An explicit flag wins; otherwise the selector is left alone so
        // `CANDLE_METAL_HAZARD_KEY` is honoured. Calling the setter
        // unconditionally consumes the `OnceLock` guarding the env switch and
        // silently pins the default -- which it did in #69, and which made a
        // determinism run that believed it was testing route (a) actually test
        // the default (§2.4, §9.2f).
        #[cfg(feature = "metal")]
        if let Some(requested) = args.hazard_key.as_deref() {
            let key = match requested {
                "pointer" | "ptr" => candle::metal_backend::HazardKey::Pointer,
                "range" => candle::metal_backend::HazardKey::Range,
                other => anyhow::bail!("unknown --hazard-key {other:?}; want pointer or range"),
            };
            candle::metal_backend::set_hazard_key(key);
        }
        #[cfg(not(feature = "metal"))]
        if args.hazard_key.is_some() {
            anyhow::bail!("--hazard-key needs a Metal build");
        }

        if args.arena && !matches!(_device, Device::Metal(_)) {
            anyhow::bail!("--arena needs a Metal device");
        }

        Ok(Self {
            arena: args.arena,
            record_steps: args.arena_record_steps,
            #[cfg(feature = "metal")]
            layout,
            gpu_offsets,
            attn: args.attn.clone(),
            kv_append: args.kv_append.clone(),
            conv_state: ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?,
        })
    }

    /// The configuration as one line, in the form the registry keys on.
    fn config_line(&self) -> String {
        // ParamStyle is not selectable from here: nothing on the LFM2 path
        // dispatches the packed variants (§11.3k), so a flag would name an arm
        // that does not run. Recorded as `split` rather than omitted, so the
        // line states every axis §7.1 has.
        //
        // `Executor` and `ReplayBarriers` are stated for that same reason and
        // are likewise not selectable here. This harness deliberately carries no
        // `--icb` (§11.3n): with the flag present, the timing §17 Phase 2 item
        // 10 forbids would be one argument away and nothing in the output would
        // say the arm is invalid. `ReplayBarriers` only has an effect under an
        // installed ICB executor, so on this harness it is `Always` by
        // construction -- recorded so a row cannot be read as having been taken
        // under an unstated arm, which is #99's diagnosis and the failure
        // `MathMode` embodied for the life of the project by being in neither
        // the registry nor a config line.
        format!(
            "ParamStyle=Split ArenaLayout={} ArenaOffsets={} HazardKey={} AttnImpl={} \
             KvAppend={} ConvState={} ScratchSizing=none Executor=Classical \
             ReplayBarriers=Always",
            if self.arena {
                #[cfg(feature = "metal")]
                {
                    match self.layout {
                        candle::metal_backend::ArenaLayout::Packed => "Packed",
                        candle::metal_backend::ArenaLayout::NonAliasing => "NonAliasing",
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    "none"
                }
            } else {
                "none(pool)"
            },
            if self.gpu_offsets { "Gpu" } else { "Cpu" },
            Self::hazard_key_name(),
            if self.attn == "sdpa" {
                "Sdpa"
            } else {
                "Generic"
            },
            if self.kv_append == "in-place" {
                "InPlace"
            } else {
                "Cat"
            },
            match self.conv_state {
                ConvState::Shuffle => "Shuffle".to_string(),
                ConvState::SlidingRing { k, slack } => {
                    format!("SlidingRing(k={k},slack={slack})")
                }
                ConvState::RotatingRing { k } => format!("RotatingRing(k={k})"),
            },
        )
    }

    /// Every axis `.bench/configurations.md` §1 declares, as `(axis, arm)`.
    ///
    /// # Why this is not `config_line()` split on spaces
    ///
    /// `config_line`'s own comment says it "states every axis §7.1 has", and it
    /// does not: the registry declares **eleven** and that line emits **eight**.
    /// `Executor`, `BarrierScope` and `MathMode` are missing. That is not a
    /// cosmetic gap — it is precisely the shape of the failure #122 recorded,
    /// where `MathMode` was a real, switchable axis absent from both the
    /// registry and the `config=[…]` line and was therefore invisible for the
    /// life of the project (`DESIGN.md` §2.3.9).
    ///
    /// `.bench/configurations.md` §1 states the consequence exactly: the
    /// `config=[…]` line "catches an axis added to the harness and not to this
    /// file. It does **not** catch an axis added to neither."
    ///
    /// So this function reports all eleven, reading each from the mechanism that
    /// actually decides it rather than from a flag — `MathMode` from the same
    /// environment variable `kernel.rs:203` reads, `Executor` from the device.
    /// A run recorded through this cannot be silently pooled with one taken
    /// under a different math mode, which is the merge the store must refuse.
    ///
    /// `config_line` is deliberately left alone: it is what every committed
    /// artifact's `RESULT` line looks like, and changing its shape would break
    /// the ingest of the corpus that already exists.
    fn axis_pairs(&self) -> Vec<(String, String)> {
        let p = |k: &str, v: &str| (k.to_string(), v.to_string());
        vec![
            // Not selectable here -- nothing on the LFM2 path dispatches the
            // packed variants (§11.3k) -- but stated, because an axis that is
            // pinned at a default is still an axis the run was taken under.
            p("ParamStyle", "Split"),
            p("Executor", Self::executor_name()),
            p(
                "ArenaLayout",
                &if self.arena {
                    #[cfg(feature = "metal")]
                    {
                        match self.layout {
                            candle::metal_backend::ArenaLayout::Packed => "Packed".to_string(),
                            candle::metal_backend::ArenaLayout::NonAliasing => {
                                "NonAliasing".to_string()
                            }
                        }
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        "none".to_string()
                    }
                } else {
                    "none(pool)".to_string()
                },
            ),
            p("ArenaOffsets", if self.gpu_offsets { "Gpu" } else { "Cpu" }),
            p("HazardKey", Self::hazard_key_name()),
            p(
                "AttnImpl",
                if self.attn == "sdpa" {
                    "Sdpa"
                } else {
                    "Generic"
                },
            ),
            p(
                "KvAppend",
                if self.kv_append == "in-place" {
                    "InPlace"
                } else {
                    "Cat"
                },
            ),
            p(
                "ConvState",
                &match self.conv_state {
                    ConvState::Shuffle => "Shuffle".to_string(),
                    ConvState::SlidingRing { k, slack } => {
                        format!("SlidingRing(k={k},slack={slack})")
                    }
                    ConvState::RotatingRing { k } => format!("RotatingRing(k={k})"),
                },
            ),
            // Only selects anything under `Executor=Icb`, which this harness
            // does not take. Stated at its default rather than omitted: an
            // unstated axis and an axis at its default are different facts, and
            // conflating them is what this whole function exists to stop.
            p("BarrierScope", "RunStart"),
            p("ScratchSizing", "none"),
            p("MathMode", Self::math_mode_name()),
        ]
    }

    /// The math mode this build will compile its kernels under.
    ///
    /// Read from the same environment variable `candle-metal-kernels`'
    /// `kernel.rs:203` reads, so the recorded value is the one that decides
    /// codegen rather than a flag we happen to pass. Note the off arm is
    /// `Relaxed`, **not** `Safe` — `MTLMathMode` has three values and neither
    /// arm available today is the strict one (§2.3.5, §2.3.9).
    fn math_mode_name() -> &'static str {
        match std::env::var("CANDLE_METAL_ENABLE_FAST_MATH").as_deref() {
            Ok("0") | Ok("false") | Ok("no") => "Relaxed+Precise",
            // Metal's own default is `Fast`, so an unset variable is the fast
            // arm rather than an unknown one: candle's `get_env_bool(.., true)`
            // declines to depart from Metal, and the `else` branch is the
            // departure (§2.3.9).
            _ => "Fast",
        }
    }

    fn executor_name() -> &'static str {
        // This harness never selects the ICB executor -- #115 ships `--icb` on
        // the trace harness, and §17 Phase 2 item 10 forbids timing a
        // 78 %-coverage executor against a full baseline anyway.
        "Classical"
    }

    fn hazard_key_name() -> &'static str {
        #[cfg(feature = "metal")]
        {
            match candle::metal_backend::hazard_key() {
                candle::metal_backend::HazardKey::Pointer => "Pointer",
                candle::metal_backend::HazardKey::Range => "Range",
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a"
        }
    }

    fn announce(&self) {
        eprintln!("configuration: {}", self.config_line());
    }
}

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode CPU/GPU split and dispatch-count profile")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Decode tokens to generate and time.
    #[arg(long, short = 'n', default_value_t = 200)]
    n: usize,

    /// Leading decode tokens excluded from the steady-state average.
    ///
    /// They still run and are still reported; they are just not what the
    /// roofline comparison is about.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// 0 selects argmax. Sampling choice does not affect the timing, but fixing
    /// it keeps the generated text identical across runs so a timing change
    /// cannot be a different-text change in disguise.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    dtype: Option<String>,

    /// Print every token's wall and GPU time, not just the summary.
    #[arg(long)]
    per_token: bool,

    /// Label printed with the result, to tag a run inside a batch.
    #[arg(long, default_value = "")]
    label: String,

    // ---- the run store (lloom issue #171) -------------------------------
    /// Append a serialized run record to this JSONL file, at exit.
    ///
    /// Nothing is written without it, and a write failure is a warning rather
    /// than a failed run: the measurement has already happened by then.
    #[arg(long)]
    run_store: Option<String>,

    /// Run id, shared with the `lloom-sample` telemetry for the same run so the
    /// two files can be joined without guessing.
    #[arg(long)]
    run_id: Option<String>,

    /// Path of the `lloom-sample` telemetry recorded around this run, recorded
    /// in the run record so a reader can find it.
    #[arg(long)]
    telemetry_path: Option<String>,

    // ---- variant axes (`DESIGN.md` §7.1) --------------------------------
    //
    // The same selectors `lfm2-dispatch-trace` carries, so a timing run and a
    // dispatch-count run can name the same configuration. Without them a
    // ms/token figure is silently all-defaults, which is what makes today's
    // `.bench/` rows unreproducible (#105).
    //
    // Each defaults to the shipped default, so an unflagged run is the
    // configuration that produced 18.763 ms/token.
    /// Serve decode activations from an activation arena (`DESIGN.md` §9.2).
    ///
    /// The first `--arena-record-steps` decode steps run on the pool to derive
    /// the plan; the arena is installed after them. Those steps are excluded
    /// from the steady-state average in addition to `--warmup`, since they
    /// measure a different allocator.
    #[arg(long)]
    arena: bool,

    /// `packed` (#69's, slots reused by liveness) or `non-aliasing` (§9.3's
    /// reference layout, every value in its own slot).
    #[arg(long, default_value = "packed")]
    arena_layout: String,

    /// Where arena offsets are computed: `cpu` (#69) or `gpu` (#70's bump
    /// allocator). Requires `--arena` and `--arena-layout non-aliasing`.
    #[arg(long, default_value = "cpu")]
    arena_offsets: String,

    /// How intra-encoder hazards are keyed: `pointer` (candle's default) or
    /// `range` (#69's route (a), interval overlap).
    ///
    /// Left unset the selector is untouched, so `CANDLE_METAL_HAZARD_KEY` is
    /// honoured -- calling the setter unconditionally consumes the `OnceLock`
    /// and silently pins the default, which is the vacuous-instrument failure
    /// §2.4 records from #69.
    #[arg(long)]
    hazard_key: Option<String>,

    /// LFM2 decode attention: `generic` (the default path) or `sdpa` (#97's
    /// GQA-native `sdpa_vector`).
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV tokens per page for `--attn flash` — the **allocation** granularity.
    ///
    /// §10.4 proposes 256 and marks it **UNVERIFIED**. A flag rather than a
    /// constant because §10.3d establishes page size as two dispatch-tier
    /// numbers, so 16, 256 and 1024 are one field apart — which is what
    /// *"must not foreclose it"* requires. Verifying which wins is #61's axis.
    #[arg(long, default_value_t = 256)]
    flash_page_size: usize,

    /// Pages per chunk for `--attn flash` — the `k` of
    /// `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk equal **by fiat**; §9.1d records that
    /// a page (allocation) and a tile (computation) are optimised against
    /// disjoint cost functions, so **a sweep holding `k = 1` cannot separate a
    /// page-size effect from a tile-size one**. 1 is what §10.4 specifies; the
    /// flag is what makes the two separable.
    #[arg(long, default_value_t = 1)]
    flash_k: usize,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset -- issue #142).
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// How decode writes conv state: `shuffle` (§6.1's `narrow` + `Tensor::cat`,
    /// the default), `ring[:K[:slack]]` (the sliding window, §10.2e) or
    /// `rotating[:K]` (§10.2a's rotating index, §10.2g).
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// Decode steps recorded before the arena is installed.
    ///
    /// Two, not one: comparing two steps is what separates an activation from
    /// session state, since an allocation whose size moved between them is
    /// sized by `kv_len` (§9.1, #68 finding 4).
    #[arg(long, default_value_t = 2)]
    arena_record_steps: usize,
}

/// Normalize an LFM2.5-VL `config.json` into candle's schema.
///
/// Same normalization as the `lfm2-determinism` probe, and for the same reason:
/// the harness has to measure the configuration that actually runs. `rope_theta`
/// hides in `rope_parameters` and candle would default it to 10000; candle
/// recomputes `intermediate_size` as 8192 while the FFN weights are
/// `[10752, 2048]`. Both change how much memory a token moves, so both matter to
/// a bandwidth argument.
fn parse_config(raw: &str) -> Result<Config> {
    let root: Value = serde_json::from_str(raw).context("parsing config.json")?;
    let text = root.get("text_config").unwrap_or(&root);
    let mut obj = text
        .as_object()
        .context("expected a JSON object for the model config")?
        .clone();

    if !obj.contains_key("rope_theta") {
        if let Some(theta) = obj
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .cloned()
        {
            obj.insert("rope_theta".into(), theta);
        }
    }

    if !obj.contains_key("tie_embedding") {
        if let Some(v) = obj.get("tie_word_embeddings").cloned() {
            obj.insert("tie_embedding".into(), v);
        }
    }

    for key in ["bos_token_id", "eos_token_id"] {
        if !obj.contains_key(key) {
            if let Some(v) = root.get(key).cloned() {
                obj.insert(key.into(), v);
            }
        }
    }

    let normalized = Value::Object(obj);
    let base: Lfm2Config = serde_json::from_value(normalized.clone())
        .context("config.json does not match candle's LFM2 config schema")?;
    let mut config = base.into_config(false);

    if let Some(stated) = normalized
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    {
        config.intermediate_size = stated;
    }

    Ok(config)
}

fn tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    anyhow::ensure!(
        header_len > 0 && header_len < 100 * 1024 * 1024,
        "implausible safetensors header length {header_len}"
    );

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let parsed: Value = serde_json::from_slice(&header)?;
    Ok(parsed
        .as_object()
        .context("safetensors header is not a JSON object")?
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let raw = std::fs::read_to_string(&index)?;
    let parsed: Value = serde_json::from_str(&raw)?;
    let map = parsed
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("safetensors index has no weight_map")?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

fn default_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-3B/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// Bytes of language-model weights, computed from the loaded config rather than
/// quoted from `DESIGN.md` §5.5.
///
/// Recomputing it here means the roofline denominator tracks whatever
/// configuration actually ran: if `intermediate_size` had silently fallen back
/// to candle's 8192, the byte count would change and the reconciliation would
/// visibly stop working, instead of quietly comparing against the wrong number.
///
/// Counts each weight once. `tie_word_embeddings` is true for this checkpoint,
/// so the embedding matrix is also the lm_head and is not double counted.
fn language_weight_bytes(config: &Config, layer_is_attn: &[bool], dtype_bytes: usize) -> usize {
    let d = config.hidden_size;
    let ffn = config.intermediate_size;
    let kv_dim = config.num_key_value_heads * (d / config.num_attention_heads);

    let mut params = 0usize;
    // Tied embedding: also the lm_head.
    params += config.vocab_size * d;
    // Final norm.
    params += d;

    for &is_attn in layer_is_attn {
        // Both layer kinds: two norms and the SwiGLU triple.
        params += 2 * d;
        params += 2 * ffn * d; // w1 (gate), w3 (up)
        params += ffn * d; // w2 (down)

        if is_attn {
            params += d * d; // q_proj
            params += 2 * kv_dim * d; // k_proj, v_proj
            params += d * d; // out_proj
            params += 2 * (d / config.num_attention_heads); // q/k layernorm
        } else {
            params += 3 * d * d; // in_proj [3d, d], already fused BCx
            params += d * 3; // depthwise conv, k=3
            params += d * d; // out_proj
        }
    }

    params * dtype_bytes
}

fn main() -> Result<()> {
    let args = Args::parse();

    let profiling = profile::enabled();

    // Before anything loads, so it is the load the run *began* under. The
    // closing reading is taken at the record, and the pair is what shows a
    // build arriving midway through.
    let load_at_start = load_average();

    // Cloned rather than moved out: `args` is read again below, both to resolve
    // the variant axes and to print them on the RESULT line.
    let model_dir = args.model_dir.clone().or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // f16 on Metal is what ambrogio loads. Measuring f32 here would measure a
    // path nothing runs, and would move twice the bytes per token.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => match &device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        },
    };

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_raw)?;

    // `AttnImpl` is a construction-tier axis on the config (§7.1, #97), so it
    // is selected before the model is built and both arms stay compiled.
    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    config.attn_impl = match args.attn.as_str() {
        "generic" => candle_transformers::models::lfm2::AttnImpl::Generic,
        "sdpa" => candle_transformers::models::lfm2::AttnImpl::Sdpa,
        // Issue #116. Selectable and never a default: 10.4's argument for it
        // is structural rather than measured, and the kv_len at which it pays
        // is #61's to find.
        "flash" => candle_transformers::models::lfm2::AttnImpl::FlashDecoding,
        other => anyhow::bail!(
            "--attn must be `generic`, `sdpa` or `flash`, got `{other}`"
        ),
    };
    config.flash_page_size = args.flash_page_size;
    config.flash_pages_per_chunk = args.flash_k;
    // Refused rather than defaulted, for the same reason as `--attn`.
    config.kv_append = match args.kv_append.as_str() {
        "cat" => candle_transformers::models::lfm2::KvAppend::Cat,
        "in-place" => candle_transformers::models::lfm2::KvAppend::InPlace,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::new(&config, vb).context("constructing LFM2 model")?;
    let mut cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    // ---- variant axes, validated before the clock starts -----------------
    //
    // Parsed and checked here so an unsatisfiable combination fails before the
    // model runs rather than after a five-minute measurement.
    let axes = Axes::resolve(&args, &device)?;
    axes.announce();

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let prompt_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    // ---- prefill, timed into its own bucket ------------------------------

    // Synchronize before starting the clock so no earlier work (weight upload,
    // cache allocation) lands inside the window.
    device.synchronize()?;
    if profiling {
        profile::reset();
    }

    let prefill_start = std::time::Instant::now();
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = model
        .forward(&input, 0, &mut cache)
        .context("prefill forward pass")?
        .squeeze(0)?;
    // Force the prefill to complete before the clock stops; candle is
    // asynchronous, so without this the time recorded is submission, not
    // execution.
    device.synchronize()?;
    let prefill_wall = prefill_start.elapsed();

    let prefill_profile = if profiling {
        let s = profile::snapshot();
        profile::reset();
        Some(s)
    } else {
        None
    };

    let mut kv_len = prompt_ids.len();

    // ---- decode, one timed window per token ------------------------------

    // Per token: (wall seconds, gpu busy seconds, dispatches).
    let mut steps: Vec<(f64, f64, u64)> = Vec::with_capacity(args.n);
    // Per token, for the run record: `CLOCK_UPTIME_RAW` at the end of the token
    // and the `kv_len` it ran at. Pre-reserved, so the push inside the loop
    // cannot reallocate mid-window; the stamp itself is one
    // `clock_gettime_nsec_np` against a 24 MHz timebase (`DESIGN.md` §3.4b).
    let mut step_stamps: Vec<(u64, usize)> = Vec::with_capacity(args.n);
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n);
    let mut hit_eos = false;
    let mut last_token_kernels: Vec<(String, u64)> = Vec::new();

    #[cfg(feature = "metal")]
    let metal_device = match &device {
        Device::Metal(d) => Some(d.clone()),
        _ => None,
    };
    while tokens.len() < args.n {
        let next = logits_processor.sample(&logits).context("sampling")?;
        if eos_ids.contains(&next) {
            // The point is a steady-state decode measurement, so generation
            // continues past the model's natural stopping point. Recorded
            // because past EOS the text degenerates, and a reader should know
            // the tail was not on-distribution.
            hit_eos = true;
        }
        tokens.push(next);

        // Arena recording spans the first `record_steps` decode steps.
        //
        // `next_arena_recording_step` between them is load-bearing rather than
        // bookkeeping: a plan needs **two** steps, because comparing them is what
        // separates an activation from session state -- an allocation whose size
        // moved between them is sized by `kv_len` and must not enter the arena
        // (§9.1, #68 finding 4). Recording one step makes the planner refuse and
        // return no allocations, which presents as an arena arm that silently
        // ran on the pool.
        #[cfg(feature = "metal")]
        if axes.arena {
            if let Some(dev) = metal_device.as_ref() {
                let step = tokens.len() - 1;
                if step == 0 {
                    dev.begin_arena_recording();
                } else if step < axes.record_steps {
                    dev.next_arena_recording_step();
                }
                dev.begin_decode_step();
            }
        }

        let step_start = std::time::Instant::now();
        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        logits = model
            .forward(&input, kv_len, &mut cache)
            .context("decode forward pass")?
            .squeeze(0)?;
        // One synchronization per token. This is what makes the window a single
        // token rather than a submission queue, and it is also what the decode
        // loop does anyway: sampling reads the logits back to the CPU, so the
        // serialization is inherent to the workload, not an artifact of
        // measuring it.
        device.synchronize()?;
        let wall = step_start.elapsed().as_secs_f64();

        let (gpu, disp) = if profiling {
            let s = profile::snapshot();
            profile::reset();
            // Kept from the last iteration: the counters are reset per token, so
            // after the loop there is nothing left to read. This is one token's
            // kernel mix, which is what the inventory should be.
            last_token_kernels = s.by_label.clone();
            (s.gpu_busy_union_s, s.dispatches)
        } else {
            (0.0, 0)
        };

        // The arena is derived from the first `record_steps` decode steps and
        // installed after them, so those steps run on the pool and are excluded
        // from the steady-state average below. Placed after the timing snapshot
        // so the install itself is not counted into a token.
        #[cfg(feature = "metal")]
        if axes.arena {
            if let Some(dev) = metal_device.as_ref() {
                dev.end_decode_step();
                // `tokens.len() - 1` is this step's index, so the install fires
                // on the last recorded step -- the same `record_steps - 1`
                // condition the dispatch-trace harness uses.
                if tokens.len() - 1 == axes.record_steps - 1 {
                    let excluded = dev.arena_recording_excluded();
                    let by_test = dev.arena_recording_excluded_by_test();
                    let plan = dev
                        .finish_arena_recording(axes.layout)
                        .context("arena recording produced no allocations")?;
                    let (covered, total) = plan.covered();
                    if let Some((n_excl, n_all)) = excluded {
                        eprintln!(
                            "arena: {n_all} allocations recorded over {} steps, \
                             {n_excl} excluded as session state",
                            axes.record_steps
                        );
                        // Split by detector, because neither subsumes the other
                        // and a zero on either side is a result: size growth
                        // finds the KV cache, cross-step liveness finds the conv
                        // state, which never grows (§5.7, §9.2c).
                        if let Some((by_size, by_step)) = by_test {
                            eprintln!(
                                "arena:   {by_size} caught by size growth (kv_len), \
                                 {by_step} by outliving the step"
                            );
                        }
                    }
                    let monotone = plan.is_bump_reproducible();
                    eprintln!(
                        "arena: {covered} of {total} ordinals served -> {} slots, {} B ({:?})",
                        plan.slots().len(),
                        plan.arena_bytes(),
                        axes.layout,
                    );
                    dev.install_arena(plan, axes.layout)
                        .context("installing the arena")?;
                    if axes.gpu_offsets {
                        // Checked against the plan that was recorded, not
                        // against the layout's name: a plan that somehow is not
                        // monotone must fail before a wrong offset can be bound
                        // rather than after (§9.3 -- there is no other
                        // detector).
                        if !monotone {
                            anyhow::bail!(
                                "the recorded plan's offsets are not monotone, so a bump \
                                 allocator cannot reproduce them"
                            );
                        }
                        let served = dev
                            .install_gpu_arena_offsets()
                            .context("computing arena offsets on the GPU")?;
                        // The engagement check §2.4 requires: this reads `Gpu`
                        // only after the table was verified equal to the plan
                        // element-wise, so it is evidence the GPU allocator ran
                        // and agreed -- not that a flag was passed.
                        eprintln!(
                            "arena: offsets computed on the GPU and verified against the plan \
                             ({served} ordinals); source now {:?}",
                            dev.arena_offsets()
                        );
                    }
                }
            }
        }

        steps.push((wall, gpu, disp));
        // Stamped after the timing snapshot, so the clock read is outside the
        // window it describes rather than inside it.
        step_stamps.push((run_record::now_ns(), kv_len));
        kv_len += 1;
    }

    // ---- summarize -------------------------------------------------------

    // Under `--arena` the first `record_steps` tokens run on the pool, because
    // the arena does not exist until its plan does. Averaging them in would mix
    // two allocators into one ms/token figure and attribute the difference to
    // the arena -- the confound this harness exists to remove. They are dropped
    // in addition to `--warmup`, not instead of it: they are a different
    // allocator, where warmup is a cold one.
    let arena_skip = if axes.arena { axes.record_steps } else { 0 };
    let warmup = args.warmup.max(arena_skip).min(steps.len());
    let steady = &steps[warmup..];
    anyhow::ensure!(
        !steady.is_empty(),
        "no steady-state tokens left after {warmup} excluded tokens \
         (--warmup {}, arena recording {arena_skip}); raise --n",
        args.warmup
    );

    let n_steady = steady.len() as f64;
    let wall_mean: f64 = steady.iter().map(|s| s.0).sum::<f64>() / n_steady;
    let gpu_mean: f64 = steady.iter().map(|s| s.1).sum::<f64>() / n_steady;
    let disp_mean: f64 = steady.iter().map(|s| s.2 as f64).sum::<f64>() / n_steady;

    // Standard deviation, so a mean is never reported without its spread.
    let wall_sd = (steady
        .iter()
        .map(|s| (s.0 - wall_mean).powi(2))
        .sum::<f64>()
        / n_steady)
        .sqrt();
    let gpu_sd = (steady.iter().map(|s| (s.1 - gpu_mean).powi(2)).sum::<f64>() / n_steady).sqrt();

    let mut walls: Vec<f64> = steady.iter().map(|s| s.0).collect();
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wall_min = walls[0];
    let wall_med = walls[walls.len() / 2];
    let wall_max = walls[walls.len() - 1];

    // Dispatch counts must be identical token to token if the sequence is
    // stable; report the range so a reader can see whether it was.
    let disp_min = steady.iter().map(|s| s.2).min().unwrap_or(0);
    let disp_max = steady.iter().map(|s| s.2).max().unwrap_or(0);

    // `layer_types` is the authoritative source for which layers hold attention
    // (`DESIGN.md` §5.2: `full_attn_idxs` is null in this checkpoint, so indices
    // must not be hardcoded).
    let layer_is_attn: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| matches!(t, LayerType::FullAttention))
        .collect();
    let n_attn = layer_is_attn.iter().filter(|b| **b).count();
    let weight_bytes = language_weight_bytes(&config, &layer_is_attn, dtype.size_in_bytes());

    println!("=== machine / configuration ===");
    println!("dtype                 {dtype:?}");
    println!(
        "device                {}",
        if args.cpu { "cpu" } else { "metal" }
    );
    println!(
        "layers                {} ({} attention, {} conv)",
        config.num_hidden_layers,
        n_attn,
        config.num_hidden_layers - n_attn
    );
    println!(
        "hidden / ffn          {} / {}",
        config.hidden_size, config.intermediate_size
    );
    println!("vocab                 {}", config.vocab_size);
    println!(
        "lm weight bytes       {weight_bytes} ({:.3} GB)",
        weight_bytes as f64 / 1e9
    );
    println!(
        "profiling             {}",
        if profiling {
            "on (CANDLE_METAL_PROFILE=1)"
        } else {
            "OFF -- rerun with CANDLE_METAL_PROFILE=1 for the CPU/GPU split"
        }
    );
    println!();

    println!("=== prefill ({} tokens) ===", prompt_ids.len());
    println!(
        "wall                  {:.2} ms",
        prefill_wall.as_secs_f64() * 1e3
    );
    if let Some(p) = &prefill_profile {
        println!("gpu busy (union)      {:.2} ms", p.gpu_busy_union_s * 1e3);
        println!("gpu busy (sum)        {:.2} ms", p.gpu_busy_sum_s * 1e3);
        println!("dispatches            {}", p.dispatches);
        println!("encoders              {}", p.encoders);
        println!(
            "command buffers       {} ({} timed)",
            p.command_buffers, p.timed_command_buffers
        );
    }
    println!();

    println!(
        "=== decode, steady state ({} tokens, {} warmup excluded) ===",
        steady.len(),
        warmup
    );
    println!(
        "wall / token          {:.3} ms  (sd {:.3}, min {:.3}, med {:.3}, max {:.3})",
        wall_mean * 1e3,
        wall_sd * 1e3,
        wall_min * 1e3,
        wall_med * 1e3,
        wall_max * 1e3
    );
    println!("throughput            {:.2} tok/s", 1.0 / wall_mean);
    if profiling {
        println!(
            "gpu busy / token      {:.3} ms  (sd {:.3})",
            gpu_mean * 1e3,
            gpu_sd * 1e3
        );
        println!(
            "non-gpu / token       {:.3} ms  ({:.1}% of wall)",
            (wall_mean - gpu_mean) * 1e3,
            100.0 * (wall_mean - gpu_mean) / wall_mean
        );
        println!(
            "dispatches / token    {:.1}  (min {}, max {})",
            disp_mean, disp_min, disp_max
        );
        if disp_mean > 0.0 {
            println!(
                "non-gpu / dispatch    {:.1} us",
                (wall_mean - gpu_mean) * 1e6 / disp_mean
            );
        }
        println!();
        println!("=== roofline ===");
        println!("weights per token     {:.3} GB", weight_bytes as f64 / 1e9);
        if gpu_mean > 0.0 {
            println!(
                "implied bandwidth     {:.1} GB/s  (weights / gpu busy)",
                weight_bytes as f64 / gpu_mean / 1e9
            );
        }
        println!(
            "implied bw vs wall    {:.1} GB/s  (weights / wall)",
            weight_bytes as f64 / wall_mean / 1e9
        );
    }
    println!();

    if args.per_token {
        println!("=== per token ===");
        for (i, (w, g, d)) in steps.iter().enumerate() {
            let tag = if i < warmup { " [warmup]" } else { "" };
            println!(
                "{i:4}  wall {:8.3} ms  gpu {:8.3} ms  dispatches {d}{tag}",
                w * 1e3,
                g * 1e3
            );
        }
        println!();
    }

    if profiling && !last_token_kernels.is_empty() {
        // One decode token's kernel mix, captured before the per-token reset.
        let total: u64 = last_token_kernels.iter().map(|(_, c)| c).sum();
        println!("=== kernels in one decode token ({total} dispatches) ===");
        for (name, count) in &last_token_kernels {
            println!("{count:6}  {name}");
        }
        println!();
    }

    // The configuration is part of the RESULT line, not a separate note.
    // #99's diagnosis was that the harness emits a near-complete cache key
    // missing the commit and the variant axes; both are here, so a row can name
    // the configuration it was taken under rather than being read as
    // all-defaults by whoever finds it later.
    println!(
        "RESULT label={} n={} warmup={} wall_ms_per_token={:.4} gpu_ms_per_token={:.4} \
         nongpu_ms_per_token={:.4} dispatches_per_token={:.1} prefill_ms={:.2} \
         prompt_tokens={} weight_bytes={} dtype={:?} hit_eos={} profiling={} \
         config=[{}] candle_commit={} seed={} temp={} top_p={}",
        args.label,
        args.n,
        warmup,
        wall_mean * 1e3,
        gpu_mean * 1e3,
        (wall_mean - gpu_mean) * 1e3,
        disp_mean,
        prefill_wall.as_secs_f64() * 1e3,
        prompt_ids.len(),
        weight_bytes,
        dtype,
        hit_eos,
        profiling,
        axes.config_line(),
        // Passed in by the caller rather than read from git here.
        //
        // A `git rev-parse` at runtime names the tree the binary is *run* from,
        // which is not necessarily the tree it was *built* from -- and a commit
        // field that is silently wrong is worse than one that is absent, since
        // #99's whole point is that this field decides whether a cached result
        // is valid. `unknown` is the honest default; the run script supplies it.
        std::env::var("LLOOM_CANDLE_COMMIT").unwrap_or_else(|_| "unknown".into()),
        args.seed,
        args.temperature,
        args.top_p,
    );

    // ---- the run record, at exit (lloom issue #171) ----------------------
    //
    // Deliberately here: after the last token, after `device.synchronize()`,
    // after every mean is computed, and after `RESULT` is printed. Nothing
    // below runs inside a measured window.
    //
    // Absent `--run-store`, nothing is written and nothing fails. `lloom-sample`
    // decided the same way and gives the reason: telemetry failure must degrade
    // to "no telemetry", never to "failed measurement".
    if let Some(store) = args.run_store.as_deref() {
        let m = |v: f64| run_record::Measured::Value(v);
        // With profiling off the harness cannot compute a GPU split -- the
        // figures come from `GPUStartTime`/`GPUEndTime`, which only the
        // profiling path collects. Recorded NEVER RUN rather than as the 0.0
        // the `RESULT` line prints, which is the defect the corpus already
        // carries: a real zero standing where "not measured" was meant.
        let (gpu_metric, nongpu_metric, disp_metric) = if profiling {
            (
                m(gpu_mean * 1e3),
                m((wall_mean - gpu_mean) * 1e3),
                m(disp_mean),
            )
        } else {
            (
                run_record::Measured::NeverRun,
                run_record::Measured::NeverRun,
                run_record::Measured::NeverRun,
            )
        };

        let rec = run_record::RunRecord {
            run_id: args.run_id.clone().unwrap_or_else(|| {
                format!(
                    "{}-{}",
                    if args.label.is_empty() {
                        "run"
                    } else {
                        &args.label
                    },
                    std::process::id()
                )
            }),
            harness: "lfm2-decode-profile",
            label: args.label.clone(),
            candle_commit: std::env::var("LLOOM_CANDLE_COMMIT")
                .ok()
                .filter(|v| !v.is_empty() && v != "unknown"),
            lloom_commit: std::env::var("LLOOM_COMMIT")
                .ok()
                .filter(|v| !v.is_empty() && v != "unknown"),
            binary_uuid: run_record::self_uuid(),
            binary_path: std::env::current_exe()
                .ok()
                .map(|p| p.display().to_string()),
            axes: axes.axis_pairs(),
            machine: sysctl_str("hw.model"),
            macos_build: sysctl_str("kern.osversion"),
            load_start: load_at_start,
            load_end: load_average(),
            under_lease: std::env::var("LLOOM_ARB_LEASE").is_ok(),
            n: args.n,
            warmup,
            seed: args.seed,
            temperature: args.temperature,
            top_p: args.top_p,
            prompt_tokens: prompt_ids.len(),
            kv_len_first: step_stamps.first().map(|s| s.1).unwrap_or(0),
            kv_len_last: step_stamps.last().map(|s| s.1).unwrap_or(0),
            dtype: format!("{dtype:?}"),
            profiling_compiled: cfg!(feature = "metal"),
            profiling_enabled: profiling,
            wall_ms_per_token: m(wall_mean * 1e3),
            gpu_ms_per_token: gpu_metric,
            nongpu_ms_per_token: nongpu_metric,
            dispatches_per_token: disp_metric,
            prefill_ms: m(prefill_wall.as_secs_f64() * 1e3),
            weight_bytes: m(weight_bytes as f64),
            steps: steps
                .iter()
                .zip(step_stamps.iter())
                .enumerate()
                .map(|(i, ((wall, gpu, disp), (t_end_ns, kv)))| run_record::Step {
                    index: i,
                    warmup: i < warmup,
                    wall_ms: wall * 1e3,
                    gpu_ms: if profiling { Some(gpu * 1e3) } else { None },
                    dispatches: if profiling { Some(*disp) } else { None },
                    t_end_ns: *t_end_ns,
                    kv_len: *kv,
                })
                .collect(),
            telemetry_path: args.telemetry_path.clone(),
        };
        match rec.emit(store) {
            Ok(()) => eprintln!("run record appended to {store}"),
            // A failed write is reported and does not fail the run: the
            // measurement already happened and its `RESULT` line is printed.
            Err(e) => eprintln!("WARNING: could not write the run record to {store}: {e}"),
        }
    }

    Ok(())
}
