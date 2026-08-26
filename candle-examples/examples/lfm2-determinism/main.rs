//! Is LFM2 decode bit-deterministic on Apple silicon, including under GPU load?
//!
//! Measurement harness for lloom issue #5 / `DESIGN.md` §2.3.8, §16 P0 #5.
//!
//! The question is whether the *hardware* is deterministic: same dispatch, same
//! inputs, same bits, regardless of occupancy, thermal state and concurrent GPU
//! work. Asking it of the real model rather than a synthetic reduction kernel
//! exercises the actual dispatch mix, the actual allocator behaviour and the
//! actual fence pattern, which is where nondeterminism would plausibly come
//! from.
//!
//! Two digests are reported per run, and the distinction matters:
//!
//! * **tokens** — SHA-256 over the generated token ids. This is what the issue
//!   asks for, but it is the *weaker* signal: sampling quantizes logits to a
//!   token, so a bit-level difference usually maps to the same token and is
//!   invisible. It only becomes visible when a difference happens to straddle
//!   a sampling boundary.
//! * **logits** — SHA-256 over the raw little-endian f32 bits of every logit
//!   vector at every step, folded in step order. This sees a single flipped
//!   mantissa bit at any of the 128000 * n_steps values, whether or not it ever
//!   changes a token.
//!
//! A logits digest that is stable across runs is a far stronger claim than a
//! stable token stream, so both are printed and both are compared.
//!
//! ```bash
//! cargo run --release --example lfm2-determinism -- \
//!   --model-dir /path/to/LFM2.5-VL-3B --n 500
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod sha256;

use anyhow::{Context, Result};
use candle::metal_backend::ArenaLayout;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::lfm2::{Cache, Config, Lfm2Config, Model};
use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode determinism probe")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt fed to the model. Fixed across runs by default.
    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, short = 'n', default_value_t = 500)]
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

    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,

    /// Weight dtype. Defaults to f16 on Metal, matching how ambrogio loads it.
    #[arg(long)]
    dtype: Option<String>,

    /// Feed the prompt verbatim instead of wrapping it in the chat template.
    ///
    /// Without the template LFM2 emits EOS at the first step and then degenerates,
    /// which measures nothing; the wrapped form is what ambrogio actually sends.
    #[arg(long)]
    raw_prompt: bool,

    /// Emit every token id and its per-step logits digest.
    #[arg(long)]
    dump_tokens: bool,

    /// Label printed with the result, to tag a run inside a batch.
    #[arg(long, default_value = "")]
    label: String,

    /// Verify the bundled SHA-256 against known vectors, then exit.
    #[arg(long)]
    self_test: bool,

    /// Ignore EOS and generate the full token budget.
    ///
    /// Note that past its natural stopping point this checkpoint falls into a
    /// two-token cycle, which is a weak test: a self-reinforcing attractor
    /// would likely reproduce even under real nondeterminism. Prefer
    /// `--turns` for long streams.
    #[arg(long)]
    ignore_eos: bool,

    /// Number of conversational turns to run against one KV cache.
    ///
    /// `DESIGN.md` §2.3.6 calls long multi-turn the strongest determinism test:
    /// beyond amplifying bit differences it exercises KV reuse, a growing and
    /// non-monotone `kv_len`, and changing chunk counts — the paths where
    /// order-dependence would actually live. Each turn appends a follow-up
    /// prompt to the same cache, so the stream stays on-distribution instead of
    /// degenerating the way `--ignore-eos` does.
    #[arg(long, default_value_t = 1)]
    turns: usize,

    /// Serve decode activations from an activation arena (`DESIGN.md` §9.2).
    ///
    /// Two steady-state decode steps are recorded to derive the plan, the arena
    /// is installed, and everything after runs against it.
    #[arg(long)]
    arena: bool,

    /// Arena layout: `packed` or `non-aliasing` (§9.3's reference).
    #[arg(long, default_value = "non-aliasing")]
    arena_layout: String,

    /// Where arena offsets come from: `cpu` (#69) or `gpu` (issue #70).
    ///
    /// **This flag is why the gate exists.** The GPU allocator's per-step reset
    /// is ordered by a device-scope fence plus candle's barrier, and §3.5 says a
    /// wrongly-ordered reset is silent corruption rather than an error -- no
    /// barrier count can detect it, so the digest is the only available detector
    /// (§2.3.2). A single forward pass will not do: §15.1 #7 requires N runs of
    /// a long multi-turn generation, which additionally exercises KV reuse and a
    /// growing `kv_len`.
    #[arg(long, default_value = "cpu")]
    arena_offsets: String,
}

/// Follow-up prompts for multi-turn mode, cycled in order.
///
/// Fixed and content-free with respect to the measurement: they exist to keep
/// the model generating real text across a long conversation, not to test any
/// particular capability.
const FOLLOW_UPS: [&str; 8] = [
    "Expand on that, and give a concrete example.",
    "What are the main tradeoffs involved?",
    "How did this evolve historically?",
    "What do people most often get wrong about it?",
    "Compare that with the main alternative approach.",
    "What would you measure to tell if it is working well?",
    "Summarize the key points so far.",
    "What is the next thing someone should learn about this?",
];

/// Normalize an LFM2.5-VL `config.json` into candle's schema.
///
/// Mirrors ambrogio's `parse_lfm2_config`, because the harness must measure the
/// configuration that actually runs, not a nearby one:
///
/// * the language config is nested under `text_config`,
/// * `rope_theta` lives in `rope_parameters` and candle would otherwise default
///   it to 10000 where this checkpoint uses 1e6,
/// * candle recomputes `intermediate_size` as 8192 while the FFN weights are
///   `[10752, 2048]`, so the stated value has to win.
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

/// Default HF cache location for the checkpoint this project targets.
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

fn main() -> Result<()> {
    let args = Args::parse();

    if args.self_test {
        sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
        println!("sha256 self-test: OK (NIST vectors + 1e6*'a' + streaming)");
        return Ok(());
    }

    let model_dir = args.model_dir.or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // f16 on Metal is what ambrogio loads: LFM2 ships bf16, but Metal's bf16
    // kernel coverage is patchy and unsupported ops fall back to the CPU
    // silently. Measuring f32 here would measure a path nothing runs.
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
    // A VL checkpoint nests the language stack under `model.language_model.`;
    // candle's text-only loader asks for `model.*`.
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

    // The checkpoint's chat_template.jinja reduces, for a single user turn with
    // no system prompt and no tools, to exactly this. Constructing it directly
    // avoids a Jinja dependency while still feeding the model the framing it
    // was trained on — fed a bare string it emits EOS at step 0.
    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    };

    let prompt_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    let mut token_hasher = sha256::Sha256::new();
    let mut logits_hasher = sha256::Sha256::new();
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n);
    let mut per_step: Vec<(u32, String, usize)> = Vec::with_capacity(args.n);

    let start = std::time::Instant::now();

    // kv_len at the moment each token is produced. Reported alongside the
    // divergence position, because a drift that localizes to a page boundary or
    // a chunk-count change is the debugging signal (§2.3.6).
    let mut kv_len = 0usize;
    let mut hit_eos = false;
    let mut turns_run = 0usize;

    // Arena state (`DESIGN.md` §9.2; issues #69, #70).
    let metal_device = match &device {
        Device::Metal(d) => Some(d.clone()),
        _ => None,
    };
    let arena_layout = match args.arena_layout.as_str() {
        "packed" => ArenaLayout::Packed,
        "non-aliasing" | "nonaliasing" | "reference" => ArenaLayout::NonAliasing,
        other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
    };
    let gpu_offsets = match args.arena_offsets.as_str() {
        "cpu" => false,
        "gpu" => true,
        other => anyhow::bail!("unknown --arena-offsets {other:?}; want cpu or gpu"),
    };
    if gpu_offsets {
        if !args.arena {
            anyhow::bail!("--arena-offsets gpu needs --arena");
        }
        if arena_layout != ArenaLayout::NonAliasing {
            // Refused rather than downgraded: a packed plan reuses slots, so its
            // offsets are not monotone and no forward-only cursor can reproduce
            // them. Falling back silently would report the CPU path's digest
            // under the GPU path's name, which is the failure §2.4 records.
            anyhow::bail!(
                "--arena-offsets gpu needs --arena-layout non-aliasing: a packed plan reuses \
                 slots, so its offsets are not monotone"
            );
        }
    }
    /// Decode steps recorded before the arena is installed. Two, because
    /// comparing two is what separates an activation from session state.
    const ARENA_RECORD_STEPS: usize = 2;
    let mut decode_steps = 0usize;

    'turns: for turn in 0..args.turns.max(1) {
        // Turn 0 prefills the initial prompt; later turns append a follow-up to
        // the same cache, so KV state is reused rather than rebuilt.
        let turn_ids: Vec<u32> = if turn == 0 {
            prompt_ids.clone()
        } else {
            let follow_up = FOLLOW_UPS[(turn - 1) % FOLLOW_UPS.len()];
            let text = format!("<|im_start|>user\n{follow_up}<|im_end|>\n<|im_start|>assistant\n");
            tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing follow-up: {e}"))?
                .get_ids()
                .to_vec()
        };

        let input = Tensor::new(turn_ids.as_slice(), &device)?.unsqueeze(0)?;
        let mut logits = model
            .forward(&input, kv_len, &mut cache)
            .context("prefill forward pass")?
            .squeeze(0)?;
        kv_len += turn_ids.len();
        turns_run = turn + 1;

        loop {
            if tokens.len() >= args.n {
                break 'turns;
            }

            // Digest the logits before sampling: this is the quantity the
            // hardware actually produced, whereas the token is that quantity
            // after a threshold has thrown most of the information away.
            let step_logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut step_bytes = Vec::with_capacity(step_logits.len() * 4);
            for v in &step_logits {
                step_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            let step_digest = sha256::digest_hex(&step_bytes);
            logits_hasher.update(&step_bytes);

            let next = logits_processor.sample(&logits).context("sampling")?;

            if eos_ids.contains(&next) && !args.ignore_eos {
                // End of this turn. With more turns to run the conversation
                // continues; otherwise generation is done.
                hit_eos = true;
                if turn + 1 < args.turns.max(1) {
                    break;
                }
                break 'turns;
            }

            token_hasher.update(&next.to_le_bytes());
            tokens.push(next);
            per_step.push((next, step_digest, kv_len));

            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;

            // Arena bookkeeping, if one was asked for (`DESIGN.md` §9.2, issues
            // #69 and #70). The first two decode steps are recorded rather than
            // served: two are the minimum that can tell an activation from
            // session state, since what separates them is a size that moved
            // between steps (§9.1, #68 finding 4).
            if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
                match decode_steps {
                    0 => dev.begin_arena_recording(),
                    n if n < ARENA_RECORD_STEPS => dev.next_arena_recording_step(),
                    _ => {}
                }
                dev.begin_decode_step();
            }

            logits = model
                .forward(&input, kv_len, &mut cache)
                .context("decode forward pass")?
                .squeeze(0)?;

            if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
                dev.end_decode_step();
                if decode_steps == ARENA_RECORD_STEPS - 1 {
                    let plan = dev
                        .finish_arena_recording(arena_layout)
                        .context("arena recording produced no allocations")?;
                    let monotone = plan.is_bump_reproducible();
                    let (covered, total) = plan.covered();
                    dev.install_arena(plan, arena_layout)
                        .context("installing the arena")?;
                    eprintln!("arena: {covered} of {total} ordinals served ({arena_layout:?})");

                    if gpu_offsets {
                        if !monotone {
                            anyhow::bail!(
                                "the recorded plan's offsets are not monotone, so a bump \
                                 allocator cannot reproduce them"
                            );
                        }
                        let served = dev
                            .install_gpu_arena_offsets()
                            .context("computing arena offsets on the GPU")?;
                        // The engagement check §2.4 requires, and #69's vacuous
                        // determinism run is why it is here: a digest reported
                        // under a flag that silently did nothing is a passing
                        // result for the path the flag was meant to replace.
                        // `arena_offsets()` reads `Gpu` only after the table was
                        // verified equal to the plan element-wise.
                        eprintln!(
                            "arena: offsets computed on the GPU and verified ({served} ordinals); \
                             source now {:?}",
                            dev.arena_offsets()
                        );
                    }
                }
                decode_steps += 1;
            }

            kv_len += 1;
        }
    }

    let elapsed = start.elapsed();
    let token_digest = sha256::hex(&token_hasher.finalize());
    let logits_digest = sha256::hex(&logits_hasher.finalize());

    if args.dump_tokens {
        for (i, (tok, digest, kv)) in per_step.iter().enumerate() {
            println!("step {i} token {tok} kv_len {kv} logits {digest}");
        }
    }

    // One machine-greppable line per run, so a batch of runs collapses to a
    // sort|uniq over digests.
    println!(
        "RESULT label={} n={} tokens_sha256={} logits_sha256={} prompt_tokens={} generated={} \
         turns={} final_kv_len={} hit_eos={} dtype={:?} device={} seed={} temp={} top_p={} \
         elapsed_ms={} tok_per_s={:.2}",
        args.label,
        args.n,
        token_digest,
        logits_digest,
        prompt_ids.len(),
        tokens.len(),
        turns_run,
        kv_len,
        hit_eos,
        dtype,
        if args.cpu { "cpu" } else { "metal" },
        args.seed,
        args.temperature,
        args.top_p,
        elapsed.as_millis(),
        tokens.len() as f64 / elapsed.as_secs_f64(),
    );

    // The raw stream, so the reported hash is independently checkable with
    // `shasum` rather than trusted.
    println!(
        "TOKENS {}",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    Ok(())
}
