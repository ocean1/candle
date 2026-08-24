//! Is candle's decode dispatch sequence identical across tokens?
//!
//! Measurement harness for lloom issue #6 / `DESIGN.md` §11.1a, §16 P0 #2.
//!
//! An `MTLIndirectCommandBuffer` is only replayable if the command sequence is
//! identical between invocations. Candle is eager — ops execute as called, and
//! nothing guarantees the next forward pass issues the same dispatches against
//! the same buffers. This records (pipeline, buffers, grid) per dispatch across
//! N decode tokens and diffs consecutive tokens.
//!
//! The model driver is the one from the determinism probe
//! (`candle-examples/examples/lfm2-determinism`, lloom issue #5), so this
//! measures the same decode path that was shown to be bit-deterministic rather
//! than a second, differently-wrong reimplementation of it.
//!
//! Three outcomes are distinguished, because they have different consequences:
//!
//! * **identical** — replay is possible; the executor work is justified.
//! * **differs only in buffer identity** — the allocator hands out different
//!   buffers for the same logical slot. Fixable, and it makes the allocator a
//!   prerequisite rather than a parallel track.
//! * **differs structurally** — kernel, grid or binding shape changes. Replay
//!   would need the model to declare stability.
//!
//! ```bash
//! CANDLE_METAL_TRACE=1 cargo run --release --example lfm2-dispatch-trace -- --n 24
//! ```
//!
//! Without `CANDLE_METAL_TRACE=1` the instrumentation is inert and the harness
//! says so rather than reporting an empty trace as a result.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::lfm2::{Cache, Config, Lfm2Config, Model};
use clap::Parser;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use candle::metal_backend::trace;

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode dispatch-sequence probe")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt fed to the model.
    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Number of decode tokens to trace.
    #[arg(long, short = 'n', default_value_t = 24)]
    n: usize,

    /// Sampling seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Sampling temperature. 0 selects argmax.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling cutoff.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Weight dtype. Defaults to f16 on Metal, matching how ambrogio loads it.
    #[arg(long)]
    dtype: Option<String>,

    /// Print the full recorded sequence for these decode steps, comma separated.
    ///
    /// The raw record is what makes the verdict checkable rather than believed,
    /// so at least two consecutive steps are worth dumping.
    #[arg(long, default_value = "1,2")]
    dump_steps: String,

    /// Write the full trace of every step to this file.
    #[arg(long)]
    dump_all: Option<PathBuf>,

    /// Also trace the prefill pass, reported separately.
    ///
    /// Prefill runs a different shape (`seq` = prompt length against `seq` = 1),
    /// so mixing it into the decode comparison would manufacture a difference
    /// that says nothing about replayability across decode tokens.
    #[arg(long)]
    include_prefill: bool,
}

/// Normalize an LFM2.5-VL `config.json` into candle's schema.
///
/// Mirrors ambrogio's `parse_lfm2_config` and the determinism probe's copy: the
/// language config is nested under `text_config`, `rope_theta` lives in
/// `rope_parameters` where candle would default it to 10000, and candle
/// recomputes `intermediate_size` as 8192 while the FFN weights are
/// `[10752, 2048]`, so the stated value has to win.
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

/// Tensor names, read from the safetensors header without mapping the weights.
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

/// One decode step's recorded dispatches.
struct Step {
    label: String,
    dispatches: Vec<trace::Dispatch>,
}

impl Step {
    fn signature(&self) -> Vec<String> {
        self.dispatches.iter().map(|d| d.signature()).collect()
    }

    fn shape_signature(&self) -> Vec<String> {
        self.dispatches
            .iter()
            .map(|d| d.shape_signature())
            .collect()
    }

    fn kernel_signature(&self) -> Vec<String> {
        self.dispatches
            .iter()
            .map(|d| d.kernel_signature())
            .collect()
    }
}

/// How two steps differ.
///
/// Finer than the issue's three outcomes on purpose. "Structural" turned out to
/// cover two findings with opposite consequences for the ICB thesis: an op
/// sequence that genuinely changes shape, versus one whose kernels and binding
/// slots are fixed and only whose *grids* scale with `kv_len`. The latter is a
/// dispatch-tier parameter (`DESIGN.md` §7.1) and is replayable if the grid can
/// be made indirect; reporting it as "structural" would retire the ICB thesis on
/// a fault that does not warrant it.
#[derive(PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
enum Diff {
    /// Byte-for-byte the same commands, including buffers.
    Identical,
    /// Same kernels, grids, slots and offsets; different backing buffers.
    BufferIdentityOnly,
    /// Same kernel sequence and binding slots; some grids or offsets scale.
    GridOrOffsetScaling,
    /// The kernel sequence itself differs.
    KernelSequence,
}

fn compare(a: &Step, b: &Step) -> Diff {
    if a.signature() == b.signature() {
        Diff::Identical
    } else if a.shape_signature() == b.shape_signature() {
        Diff::BufferIdentityOnly
    } else if a.kernel_signature() == b.kernel_signature() {
        Diff::GridOrOffsetScaling
    } else {
        Diff::KernelSequence
    }
}

fn render(step: &Step) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== {} : {} dispatches ===\n",
        step.label,
        step.dispatches.len()
    ));
    for (i, d) in step.dispatches.iter().enumerate() {
        out.push_str(&format!(
            "{i:>4}  {} grid={:?} tg={:?} {}\n",
            d.pipeline,
            d.grid,
            d.threadgroup,
            if d.by_threadgroups {
                "threadgroups"
            } else {
                "threads"
            }
        ));
        for b in &d.bindings {
            out.push_str(&format!(
                "        [{}] {} buf#{} @{:#x} off={}\n",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.buffer_id,
                b.buffer_addr,
                b.offset
            ));
        }
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !trace::trace_requested() {
        anyhow::bail!(
            "CANDLE_METAL_TRACE is not set, so no dispatches would be recorded.\n\
             Re-run with:  CANDLE_METAL_TRACE=1 cargo run --release --example \
             lfm2-dispatch-trace -- --n {}",
            args.n
        );
    }

    let model_dir = args.model_dir.or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = Device::new_metal(0)?;

    // f16 on Metal is what ambrogio loads and what the determinism probe
    // measured. Tracing a dtype nothing runs would trace a different kernel mix.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => DType::F16,
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

    // The checkpoint's chat template, reduced for a single user turn. Fed a bare
    // string LFM2 emits EOS at step 0 and there is no decode to trace.
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

    let mut steps: Vec<Step> = Vec::new();
    let mut kv_len = 0usize;

    // Prefill. Traced only on request and never mixed into the decode
    // comparison: `seq` is the prompt length here and 1 in decode, so the two
    // are different command sequences by construction.
    // Each traced window covers exactly one `model.forward`, and nothing else.
    //
    // Two things have to be kept out of it or the count is not the model's.
    // Sampling runs its own kernels (softmax, sort, affine) and is not part of
    // the forward pass. And candle is lazy on Metal, so the forward's dispatches
    // are not all encoded until something reads the result back — a readback
    // inside the window would encode the tail of the forward but also add cast
    // dispatches of its own.
    //
    // `device.synchronize()` resolves this: it forces the queued work to be
    // submitted and completed without encoding any compute of its own, so the
    // window closes on a drained queue containing only the forward.
    let trace_window = |device: &Device| -> Result<()> {
        device.synchronize()?;
        Ok(())
    };

    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    if args.include_prefill {
        trace::set_region(Some("prefill".to_string()));
        trace::set_recording(true);
    }
    let mut logits = model
        .forward(&input, kv_len, &mut cache)
        .context("prefill forward pass")?
        .squeeze(0)?;
    trace_window(&device)?;
    if args.include_prefill {
        trace::set_recording(false);
        let dispatches = trace::take_dispatches();
        steps.push(Step {
            label: "prefill".to_string(),
            dispatches,
        });
    }
    kv_len += prompt_ids.len();

    let mut tokens: Vec<u32> = Vec::new();
    for step in 0..args.n {
        let next = logits_processor.sample(&logits).context("sampling")?;
        if eos_ids.contains(&next) {
            eprintln!(
                "note: EOS at decode step {step}; traced {} decode steps",
                steps
                    .iter()
                    .filter(|s| s.label.starts_with("decode"))
                    .count()
            );
            break;
        }
        tokens.push(next);

        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;

        // Drain anything sampling left queued, so it is attributed to the step
        // it belongs to rather than to the window opened next.
        trace_window(&device)?;

        let label = format!("decode[{step}] kv_len={kv_len}");
        trace::set_region(Some(label.clone()));
        trace::set_recording(true);
        logits = model
            .forward(&input, kv_len, &mut cache)
            .context("decode forward pass")?
            .squeeze(0)?;
        trace_window(&device)?;
        trace::set_recording(false);

        let dispatches = trace::take_dispatches();
        steps.push(Step { label, dispatches });
        kv_len += 1;
    }

    trace::set_region(None);

    let decode_steps: Vec<&Step> = steps
        .iter()
        .filter(|s| s.label.starts_with("decode"))
        .collect();

    anyhow::ensure!(
        decode_steps.len() >= 2,
        "need at least 2 decode steps to diff; got {}",
        decode_steps.len()
    );

    println!("== dispatch counts per decode token ==");
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for s in &decode_steps {
        *counts.entry(s.dispatches.len()).or_default() += 1;
        println!("{:<28} {} dispatches", s.label, s.dispatches.len());
    }
    if let Some(prefill) = steps.iter().find(|s| s.label == "prefill") {
        println!(
            "{:<28} {} dispatches (not compared against decode)",
            prefill.label,
            prefill.dispatches.len()
        );
    }
    println!("\ndistinct decode dispatch counts: {counts:?}");

    // Kernel histogram: what the per-token dispatch budget is actually spent on.
    // §11.2 estimates ~240 from 30 layers x ~8 ops; the issue asks for the
    // counted number rather than the inherited estimate.
    println!("\n== kernels per decode token (step 1) ==");
    let mut per_kernel: BTreeMap<&str, usize> = BTreeMap::new();
    for d in &decode_steps[1].dispatches {
        *per_kernel.entry(d.pipeline.as_str()).or_default() += 1;
    }
    let mut hist: Vec<(&&str, &usize)> = per_kernel.iter().collect();
    hist.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, n) in hist {
        println!("{n:>5}  {name}");
    }

    // Consecutive diffs. Step 0 is included in the report but the verdict is
    // taken over steps >= 1: the first decode token follows prefill and can
    // legitimately differ (cache warm-up, first-use allocations) without saying
    // anything about whether steady-state decode is replayable.
    println!("\n== consecutive diffs ==");
    let mut verdicts: Vec<(String, Diff)> = Vec::new();
    for pair in decode_steps.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let diff = compare(a, b);
        println!("{:<28} -> {:<28} {:?}", a.label, b.label, diff);
        verdicts.push((format!("{} -> {}", a.label, b.label), diff));
    }

    // Which dispatch positions actually move, aggregated over every steady-state
    // pair rather than reported from the first difference found. The first
    // difference is whichever position happens to come earliest, which says
    // nothing about how much of the sequence is unstable or why.
    println!("\n== what varies across steady-state decode steps ==");
    let n = decode_steps[1].dispatches.len();
    let uniform_len = decode_steps.iter().skip(1).all(|s| s.dispatches.len() == n);
    if uniform_len {
        let mut grid_varies: Vec<usize> = Vec::new();
        let mut offset_varies: Vec<usize> = Vec::new();
        let mut buffer_varies: Vec<usize> = Vec::new();
        let mut kernel_varies: Vec<usize> = Vec::new();
        for i in 0..n {
            let base = &decode_steps[1].dispatches[i];
            let rest = || decode_steps.iter().skip(2).map(|s| &s.dispatches[i]);
            if rest().any(|d| d.pipeline != base.pipeline) {
                kernel_varies.push(i);
            }
            if rest().any(|d| {
                d.grid != base.grid
                    || d.threadgroup != base.threadgroup
                    || d.by_threadgroups != base.by_threadgroups
            }) {
                grid_varies.push(i);
            }
            if rest().any(|d| {
                d.bindings
                    .iter()
                    .map(|b| b.offset)
                    .ne(base.bindings.iter().map(|b| b.offset))
            }) {
                offset_varies.push(i);
            }
            if rest().any(|d| {
                d.bindings
                    .iter()
                    .map(|b| b.buffer_id)
                    .ne(base.bindings.iter().map(|b| b.buffer_id))
            }) {
                buffer_varies.push(i);
            }
        }

        let summarize = |what: &str, idx: &[usize]| {
            println!("{what:<22} {:>4} of {n} dispatches", idx.len());
            if !idx.is_empty() {
                let mut by_kernel: BTreeMap<&str, usize> = BTreeMap::new();
                for &i in idx {
                    *by_kernel
                        .entry(decode_steps[1].dispatches[i].pipeline.as_str())
                        .or_default() += 1;
                }
                let mut v: Vec<_> = by_kernel.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
                for (k, c) in v {
                    println!("{:>26}  {c:>4}  {k}", "");
                }
            }
        };
        summarize("kernel name varies:", &kernel_varies);
        summarize("grid varies:", &grid_varies);
        summarize("binding offset varies:", &offset_varies);
        summarize("buffer identity varies:", &buffer_varies);

        // A grid that grows by a constant amount per token is `kv_len` reaching
        // the dispatch, which is the shape §6.2 predicts for the KV `cat`.
        if let Some(&i) = grid_varies.first() {
            let seq: Vec<usize> = decode_steps
                .iter()
                .skip(1)
                .map(|s| s.dispatches[i].grid.1)
                .collect();
            let deltas: Vec<i64> = seq.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect();
            println!(
                "\nexample growing dispatch: #{i} {} grid.height {:?}",
                decode_steps[1].dispatches[i].pipeline,
                &seq[..seq.len().min(8)]
            );
            println!("  per-token delta: {:?}", &deltas[..deltas.len().min(7)]);
        }
    } else {
        println!("dispatch counts differ between steps; per-position comparison skipped");
    }

    let steady: Vec<&(String, Diff)> = verdicts.iter().skip(1).collect();
    let worst = steady
        .iter()
        .map(|(_, d)| *d)
        .max()
        .unwrap_or(Diff::Identical);
    let verdict = match worst {
        Diff::Identical => "IDENTICAL",
        Diff::BufferIdentityOnly => "BUFFER_IDENTITY_ONLY",
        Diff::GridOrOffsetScaling => "STABLE_KERNEL_SEQUENCE_VARYING_GRIDS",
        Diff::KernelSequence => "STRUCTURAL",
    };

    println!("\n== verdict ==");
    println!(
        "steady-state decode (steps 1..{}): {verdict}",
        decode_steps.len() - 1
    );
    println!(
        "first transition (decode[0] -> decode[1]): {:?}",
        verdicts[0].1
    );
    println!(
        "dispatches per decode token: {}",
        if counts.len() == 1 {
            format!("{}", decode_steps[1].dispatches.len())
        } else {
            format!("varies: {counts:?}")
        }
    );
    println!(
        "dispatches seen while not recording: {}",
        trace::skipped_count()
    );

    let dump: Vec<usize> = args
        .dump_steps
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .context("parsing --dump-steps")?;
    for i in dump {
        if let Some(s) = decode_steps.get(i) {
            println!("\n{}", render(s));
        }
    }

    if let Some(path) = args.dump_all {
        let mut out = String::new();
        for s in &steps {
            out.push_str(&render(s));
            out.push('\n');
        }
        std::fs::write(&path, out).with_context(|| format!("writing {}", path.display()))?;
        println!("full trace written to {}", path.display());
    }

    println!("\ngenerated {} tokens", tokens.len());
    Ok(())
}
