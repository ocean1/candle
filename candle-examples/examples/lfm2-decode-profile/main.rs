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
use candle_transformers::models::lfm2::{Cache, Config, LayerType, Lfm2Config, Model};
use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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
        pub bind_probes: u64,
        pub bind_probe_misses: u64,
        pub bind_probe_deduped: u64,
        pub bind_probe_waits: u64,
        pub bind_probe_mean_map_entries: f64,
        pub blit_copies: u64,
        pub blit_copy_dst_pending: u64,
        pub blit_copy_dst_uncovered: u64,
    }
    pub fn enabled() -> bool {
        false
    }
    pub fn reset() {}
    pub fn snapshot() -> Snapshot {
        Snapshot::default()
    }
}

/// One token's per-bind fence-probe counts (lloom issue #24).
///
/// `probes` is one call to `ComputeCommandEncoder::wait_for_buffer`, i.e. one
/// bound buffer, and the other three partition it: every probe either found no
/// pending writer, found a fence this encoder had already waited on, or emitted
/// a `waitForFence`. Reporting all four lets a reader check the partition
/// instead of trusting the instrumentation.
#[derive(Clone, Copy, Debug, Default)]
struct BindCounts {
    probes: u64,
    misses: u64,
    deduped: u64,
    waits: u64,
    mean_map_entries: f64,
    /// `copy_from_buffer` calls and what their destination wait found (#25).
    blit_copies: u64,
    blit_dst_pending: u64,
    blit_dst_uncovered: u64,
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

    let model_dir = args.model_dir.or_else(default_model_dir).context(
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
    let config = parse_config(&config_raw)?;

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
    // Per token, in the same order: the per-bind fence-probe counts (issue #24).
    // Kept in a parallel vector rather than widened into the tuple above so the
    // steady-state arithmetic that consumes `steps` is untouched.
    let mut binds: Vec<BindCounts> = Vec::with_capacity(args.n);
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n);
    let mut hit_eos = false;
    let mut last_token_kernels: Vec<(String, u64)> = Vec::new();

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
            binds.push(BindCounts {
                probes: s.bind_probes,
                misses: s.bind_probe_misses,
                deduped: s.bind_probe_deduped,
                waits: s.bind_probe_waits,
                mean_map_entries: s.bind_probe_mean_map_entries,
                blit_copies: s.blit_copies,
                blit_dst_pending: s.blit_copy_dst_pending,
                blit_dst_uncovered: s.blit_copy_dst_uncovered,
            });
            (s.gpu_busy_union_s, s.dispatches)
        } else {
            binds.push(BindCounts::default());
            (0.0, 0)
        };

        steps.push((wall, gpu, disp));
        kv_len += 1;
    }

    // ---- summarize -------------------------------------------------------

    let warmup = args.warmup.min(steps.len());
    let steady = &steps[warmup..];
    anyhow::ensure!(
        !steady.is_empty(),
        "no steady-state tokens left after {warmup} warmup tokens; raise --n"
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

        // ---- per-bind fence probe (lloom issue #24) ----------------------
        //
        // Counts only. The per-operation price comes from the `bind_cost`
        // microbenchmark, because a timer pair costs ~43 ns against a probe
        // body of the same order (`CONTRIBUTING.md` §3.2).
        let steady_binds = &binds[warmup.min(binds.len())..];
        if !steady_binds.is_empty() && steady_binds.iter().any(|b| b.probes > 0) {
            let nb = steady_binds.len() as f64;
            let probes = steady_binds.iter().map(|b| b.probes as f64).sum::<f64>() / nb;
            let misses = steady_binds.iter().map(|b| b.misses as f64).sum::<f64>() / nb;
            let deduped = steady_binds.iter().map(|b| b.deduped as f64).sum::<f64>() / nb;
            let waits = steady_binds.iter().map(|b| b.waits as f64).sum::<f64>() / nb;
            let map_entries = steady_binds.iter().map(|b| b.mean_map_entries).sum::<f64>() / nb;
            let probe_min = steady_binds.iter().map(|b| b.probes).min().unwrap_or(0);
            let probe_max = steady_binds.iter().map(|b| b.probes).max().unwrap_or(0);

            println!("=== per-bind fence probe (issue #24) ===");
            println!(
                "wait_for_buffer calls {probes:.1} / token  (min {probe_min}, max {probe_max})"
            );
            if disp_mean > 0.0 {
                println!("  per dispatch        {:.2}", probes / disp_mean);
            }
            let pct = |x: f64| {
                if probes > 0.0 {
                    100.0 * x / probes
                } else {
                    0.0
                }
            };
            println!(
                "  no pending writer   {misses:.1}  ({:.1}%)  -- one mutex, one failed lookup",
                pct(misses)
            );
            println!(
                "  hit, already waited {deduped:.1}  ({:.1}%)  -- full probe cost, no edge emitted",
                pct(deduped)
            );
            println!(
                "  waitForFence issued {waits:.1}  ({:.1}%)  -- the actual fence edges",
                pct(waits)
            );
            println!("  mean map entries    {map_entries:.1}");

            // lloom #25: how often the destination wait added to
            // `copy_from_buffer` actually finds an edge. `uncovered` is the
            // subset the blanket `live_fences` wait in `blit_command_encoder`
            // does not already provide, i.e. the edge this change adds.
            let copies = steady_binds
                .iter()
                .map(|b| b.blit_copies as f64)
                .sum::<f64>()
                / nb;
            let dst_pending = steady_binds
                .iter()
                .map(|b| b.blit_dst_pending as f64)
                .sum::<f64>()
                / nb;
            let dst_uncovered = steady_binds
                .iter()
                .map(|b| b.blit_dst_uncovered as f64)
                .sum::<f64>()
                / nb;
            println!("  copy_from_buffer    {copies:.1} / token");
            println!("    dst had a writer  {dst_pending:.1}");
            println!("    of those, not already covered by the blanket wait  {dst_uncovered:.1}");
            // The partition is an invariant of the instrumentation; print the
            // check rather than asserting it, so a violation is visible in the
            // artifact instead of aborting a 200-token run.
            let sum = misses + deduped + waits;
            println!(
                "  partition check     {sum:.1} == {probes:.1}  {}",
                if (sum - probes).abs() < 1e-9 {
                    "ok"
                } else {
                    "MISMATCH"
                }
            );
            println!();
        }

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
    } else if profiling {
        // Say so rather than leaving a silently absent section: the inventory
        // costs a String allocation per dispatch, which is the same order as
        // the per-bind costs this profile measures, so it is off by default.
        println!(
            "=== kernels in one decode token ===\n(off; \
             set CANDLE_METAL_PROFILE_KERNELS=1 to attribute dispatches to kernels.\n\
             It allocates per dispatch, so it is excluded from timed runs.)\n"
        );
    }

    let steady_binds = &binds[warmup.min(binds.len())..];
    let nb = steady_binds.len().max(1) as f64;
    let bind_mean = |f: fn(&BindCounts) -> u64| -> f64 {
        steady_binds.iter().map(|b| f(b) as f64).sum::<f64>() / nb
    };

    println!(
        "RESULT label={} n={} warmup={} wall_ms_per_token={:.4} gpu_ms_per_token={:.4} \
         nongpu_ms_per_token={:.4} dispatches_per_token={:.1} prefill_ms={:.2} \
         prompt_tokens={} weight_bytes={} dtype={:?} hit_eos={} profiling={} \
         bind_probes_per_token={:.1} bind_misses_per_token={:.1} \
         bind_deduped_per_token={:.1} bind_waits_per_token={:.1} \
         bind_mean_map_entries={:.1} blit_copies_per_token={:.1} \
         blit_dst_pending_per_token={:.1} blit_dst_uncovered_per_token={:.1}",
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
        bind_mean(|b| b.probes),
        bind_mean(|b| b.misses),
        bind_mean(|b| b.deduped),
        bind_mean(|b| b.waits),
        steady_binds.iter().map(|b| b.mean_map_entries).sum::<f64>() / nb,
        bind_mean(|b| b.blit_copies),
        bind_mean(|b| b.blit_dst_pending),
        bind_mean(|b| b.blit_dst_uncovered),
    );

    Ok(())
}
