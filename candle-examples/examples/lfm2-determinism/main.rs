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
//! # This harness reports no throughput, deliberately
//!
//! `lfm2-decode-profile` is the one that measures throughput, and its
//! `wall_ms_per_token` is a steady-state per-token mean with warmup excluded
//! under argmax. Nothing here is comparable to it: this runs multi-turn with
//! sampling on, counts prefill and warmup, and lets `kv_len` grow across turns.
//!
//! The `elapsed_ms` below is a duration and not a rate. **Do not divide it by
//! `generated`** — that reconstructs the whole-generation average
//! `measurements/issue-7-reconciliation.md` §1 exists to discredit, and it is
//! what the removed `tok_per_s` field was (lloom #102).
//!
//! The reason it cannot be fixed by relabelling is structural: two arms of a
//! determinism A/B generate different numbers of tokens *by construction*,
//! because when the digests diverge the arms take different paths and stop at
//! different points. So `elapsed_ms` is not comparable between arms either, and
//! neither is anything derived from it.
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
use candle_transformers::models::lfm2::{
    AttnImpl, Cache, Config, ConvState, KvAppend, Lfm2Config, Model,
};
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

    /// Write every step's raw f32 logits to this path, as
    /// `step <i> <v0> <v1> ...` one line per step.
    ///
    /// **A digest cannot answer "does the error grow".** §2.3.5a's second
    /// discriminator is that a reduction-order difference is ulp-scale and does
    /// *not* grow with reduction length — which is a statement about magnitude,
    /// and a SHA-256 destroys magnitude by construction: two logit vectors one
    /// ulp apart and two that are unrelated both give unequal 64-hex strings.
    /// This dumps the values so the difference between two arms can be
    /// subtracted rather than merely observed to exist (issue #141, §10.2g).
    #[arg(long)]
    dump_logits: Option<String>,

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

    /// Which attention implementation decode takes: `generic` (the default, and
    /// the arm that must still reproduce the digests recorded from §8.1b through
    /// §11.3k) or `sdpa` (the GQA-native `sdpa_vector` kernel, issue #97).
    ///
    /// The `sdpa` arm is *expected* to produce a different pair, for the reasons
    /// recorded in `measurements/issue-97-prediction.md` before it was run:
    /// online softmax rescales the accumulator per key, `fast::exp` is not
    /// candle's `exp`, and the score is a `simd_sum`. What it must not do is
    /// produce a pair that *varies between runs* -- the kernel walks keys by
    /// index with no float atomics and no completion-order merge, so §2.3.3 #1
    /// holds by construction and a varying digest would mean something else is
    /// wrong.
    ///
    /// **Carried here beside the arena flags so a combination can be gated.**
    /// `bench/issue-70-det` had the arena arms and no `--attn`;
    /// `bench/issue-97-det` had `--attn` and no arena. Neither could gate
    /// `Sdpa × arena`, which is why #105's C6 reports counts rather than digests.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset -- issue #142).
    ///
    /// **The digests must not move across this axis**, which is what makes it
    /// different from `--attn`. `Sdpa` changes the arithmetic and legitimately
    /// moves them (§2.3.8c); `in-place` changes only *where the bytes live*, so
    /// a moved digest here is a defect rather than a variant (§2.3.5a). Gate it
    /// against the pair for whichever `--attn` arm is selected, not against the
    /// canonical generic pair.
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// How decode writes conv state: `shuffle` (the default, §6.1's
    /// `narrow` + `Tensor::cat`), `ring[:K[:slack]]` (the sliding window,
    /// §10.2e) or `rotating[:K]` (§10.2a's rotating index, §10.2g).
    ///
    /// **The two ring arms differ in what this harness should expect, and the
    /// difference is the point.** `ring` slides, so its live window is
    /// `l_cache` contiguous slots in the shuffle's own order: the summation
    /// order is unchanged and the digests must be **unmoved** — a moved digest
    /// there is a defect. `rotating` rotates which slot holds the newest token,
    /// which rotates the order `sum_keepdim` walks, so its digests **move
    /// deliberately** — the shape `--attn sdpa` has (§2.3.8c). §10.2g records
    /// the discriminator runs that separate that from a computational bug.
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// How every kernel's scalars reach it: `split` (the default, one `setBytes`
    /// per scalar) or `packed` (one `device const Params*`, issue #115).
    ///
    /// The families converted in #38 through #81 each carry both entry points
    /// from one body, and until now **nothing selected the packed ones** --
    /// `candle-core` calls only the classical `call_*`, which pass
    /// `ParamStyle::default()`. So the packed variants were compiled, checked by
    /// their own parity arms, and never dispatched by the model.
    ///
    /// This flag makes them reachable, and the digest is why it exists. Packing
    /// moves a scalar from an inline `setBytes` into a struct field at a
    /// computed offset, and §11.3d records that a field at the wrong offset does
    /// not crash -- the kernel reads a well-formed number from the wrong place
    /// and computes a plausible wrong answer. The per-family layout checks
    /// compare `sizeof` and every offset across the language boundary; what they
    /// cannot show is that the *whole decode path* still computes the same thing
    /// when every family switches at once. Only the §15.1 #7 gate does that.
    ///
    /// Expected result is therefore **bit-identical**, not merely stable. Both
    /// styles are instantiated from the same kernel body and differ only in how
    /// the arguments arrive, so unlike `--attn` this changes no arithmetic: a
    /// digest that moves here is a defect, not a variant (§2.3.5a).
    #[arg(long, default_value = "split")]
    param_style: String,

    /// Replay the stable, packed subset of a decode step from an
    /// `MTLIndirectCommandBuffer` (issue #115, `DESIGN.md` §17 Phase 2 item 10).
    ///
    /// **This flag is why the gate exists for this axis.** Replay suppresses a
    /// classical dispatch and runs a pre-encoded command in its place, and an
    /// ICB's commands are `ConcurrentDispatch` with no barrier between them
    /// (§3.5) -- so the ordering candle's `auto_barrier` emits has to be
    /// re-expressed on the commands themselves. A missing edge there is silent
    /// corruption rather than an error, and no barrier count can detect it: the
    /// count would simply be lower and still look plausible.
    ///
    /// Expected result is **bit-identical to the same configuration without
    /// `--icb`**. Replay changes where a dispatch is encoded, not what it
    /// computes, so a moved digest is a defect and not a variant (§2.3.5a).
    ///
    /// Needs `--param-style packed`: an ICB command cannot carry an inline
    /// constant (§3.7c), so under `split` every position is excluded and the run
    /// would gate an executor that replayed nothing.
    #[arg(long)]
    icb: bool,

    /// How far back an ICB command's barrier scan looks: `run-start` or
    /// `since-barrier`.
    ///
    /// `DESIGN.md` §11.3l's open question 3. `run-start` (the default, and what
    /// #115 shipped) emits a barrier wherever a command conflicts with anything
    /// earlier in its run. `since-barrier` scans back only to the previous
    /// barrier, which is what candle's `auto_barrier` does -- it emits one and
    /// then *replaces* `prev_outputs`, so an edge an earlier barrier already
    /// covers is not re-ordered.
    ///
    /// Both must produce the same digest: `setBarrier` orders a command after
    /// every command before it, so the transitive order is identical and only
    /// the number of commands carrying the flag differs. A moved digest here is
    /// a defect (§2.3.5a) and, per §3.5, the barrier count cannot detect one --
    /// which is why this is gated on the digest and not on the count.
    #[arg(long, default_value = "run-start")]
    icb_barrier_scope: String,

    /// Whether candle's own `auto_barrier` still fires at a replayed position:
    /// `always` or `skip-replayed` (issue #144).
    ///
    /// `DESIGN.md` §11.3p. `auto_barrier` is the first statement of
    /// `dispatch_thread_groups` and `offer_to_executor` is consulted after it,
    /// so a replayed position emits candle's barrier *and* the ICB's -- §11.3n
    /// measures the 401 as **additive**, for 906 orderings against the classical
    /// arm's 505. §11.3p attributes the 505 and finds **393 fire at covered
    /// non-head positions**, where the ICB has already encoded ordering.
    ///
    /// `always` is the default and is byte-for-byte what ships.
    /// `skip-replayed` suppresses candle's barrier at a **non-head member of a
    /// run already in flight** -- narrower than "the position is covered", for
    /// the three reasons §11.3p gives and which
    /// `measurements/issue-144-predicate.md` argues at edge level.
    ///
    /// **This removes ordering edges**, so under `HazardTrackingModeUntracked` a
    /// wrongly-removed one is silent corruption rather than an error (§3.5) --
    /// the same standing as `HazardKey::Range`. Both arms must produce the
    /// canonical digests; a moved digest here is a defect and not a variant
    /// (§2.3.5a), and the barrier count cannot detect one because it would
    /// simply be lower and still plausible (§2.4).
    #[arg(long, default_value = "always")]
    icb_replay_barriers: String,
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
    let mut config = parse_config(&config_raw)?;

    // Refused rather than defaulted. A silent fallback would report the generic
    // path's digests under the sdpa arm's name, and the whole point of this run
    // is to tell those two apart -- #69's vacuous determinism run is exactly the
    // failure this avoids (§2.4).
    config.attn_impl = match args.attn.as_str() {
        "generic" => AttnImpl::Generic,
        "sdpa" => AttnImpl::Sdpa,
        other => anyhow::bail!("--attn must be `generic` or `sdpa`, got `{other}`"),
    };
    println!("attention implementation: {:?}", config.attn_impl);
    // Refused rather than defaulted, for the same reason as `--attn`: an
    // unrecognized value that silently pinned the default would report the
    // `cat` arm's digest under the `in-place` arm's name, which is #69's
    // vacuous run (§2.4, §9.2f).
    config.kv_append = match args.kv_append.as_str() {
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    println!("kv append: {:?}", config.kv_append);

    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    println!("conv state: {:?}", config.conv_state);

    // The binding-style axis (issue #115). Set before any dispatch, and read
    // back from the crate rather than echoed from `args`, so the line printed is
    // what the kernels will actually use: an A/B behind a switch owes a check
    // that the two arms differ in something observable (§2.4, §9.2f).
    #[cfg(feature = "metal")]
    {
        use candle_metal_kernels::{default_param_style, set_default_param_style, ParamStyle};
        let want = match args.param_style.as_str() {
            "split" => ParamStyle::Split,
            "packed" => ParamStyle::Packed,
            other => anyhow::bail!("--param-style must be `split` or `packed`, got `{other}`"),
        };
        set_default_param_style(want);
        let got = default_param_style();
        // Not `assert_eq!` against `want` alone: that would pass if the setter
        // and the getter agreed with each other while neither reached the
        // kernels. The load-bearing evidence that this arm engaged is the
        // kernel-name census `lfm2-dispatch-trace` prints -- every packed
        // dispatch resolves a `*_packed` `[[host_name]]` -- and this is the
        // cheap precondition for it.
        anyhow::ensure!(
            got == want,
            "--param-style {want:?} did not take effect; default_param_style() reports {got:?}"
        );
        println!("param style: {got:?}");
    }

    // ICB replay (issue #115). Both switches have to be thrown before the first
    // pipeline is built: `supportIndirectCommandBuffers` is a property of a
    // pipeline and pipelines are cached per process, so one built earlier keeps
    // the old setting -- and encoding that into an ICB is §3.7d's segfault at
    // encode time, with no error to catch.
    #[cfg(feature = "metal")]
    if args.icb {
        anyhow::ensure!(
            args.param_style == "packed",
            "--icb needs --param-style packed: an ICB command cannot carry an inline constant \
             (DESIGN.md §3.7c), so under `split` every position is excluded and this would \
             gate an executor that replayed nothing"
        );
        candle_metal_kernels::set_pipelines_support_icb(true)
            .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
        candle_metal_kernels::set_constants_pool_enabled(true);
        println!("icb: enabled");
    }
    #[cfg(not(feature = "metal"))]
    anyhow::ensure!(!args.icb, "--icb needs the `metal` feature");
    // Refused rather than ignored off Metal: silently accepting `--param-style
    // packed` on a CPU build and reporting a passing digest for the default path
    // is #69's vacuous run exactly (§2.4).
    #[cfg(not(feature = "metal"))]
    anyhow::ensure!(
        args.param_style == "split",
        "--param-style {:?} needs the `metal` feature; this binary has no packed entry points",
        args.param_style
    );

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
    let mut logits_sink = match args.dump_logits.as_deref() {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).with_context(|| format!("creating {p}"))?,
        )),
        None => None,
    };

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

    // Steps the ICB executor records before deciding what is replayable. Three
    // rather than the minimum of two: a position that happens to agree between
    // two consecutive steps but drifts on the third would be admitted by a
    // two-step window, and admitting one wrongly is silent corruption (§3.5)
    // rather than a visible failure.
    #[cfg(feature = "metal")]
    const ICB_RECORD_STEPS: usize = 3;
    #[cfg(feature = "metal")]
    let icb_barrier_scope = match args.icb_barrier_scope.as_str() {
        "run-start" => candle_metal_kernels::BarrierScope::RunStart,
        "since-barrier" => candle_metal_kernels::BarrierScope::SinceBarrier,
        other => anyhow::bail!(
            "--icb-barrier-scope must be `run-start` or `since-barrier`, got `{other}`"
        ),
    };
    #[cfg(feature = "metal")]
    let icb_replay_barriers = match args.icb_replay_barriers.as_str() {
        "always" => candle_metal_kernels::ReplayBarriers::Always,
        "skip-replayed" => candle_metal_kernels::ReplayBarriers::SkipReplayed,
        other => anyhow::bail!(
            "--icb-replay-barriers must be `always` or `skip-replayed`, got `{other}`"
        ),
    };
    #[cfg(feature = "metal")]
    let icb_executor = args.icb.then(|| {
        candle_metal_kernels::IcbExecutor::configured(
            ICB_RECORD_STEPS,
            icb_barrier_scope,
            icb_replay_barriers,
        )
    });
    #[cfg(feature = "metal")]
    let icb_installed_at = if args.arena { ARENA_RECORD_STEPS } else { 0 };

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

            if let Some(sink) = logits_sink.as_mut() {
                use std::io::Write;
                let mut line = String::with_capacity(step_logits.len() * 12);
                line.push_str(&format!("step {}", per_step.len()));
                for v in &step_logits {
                    // Hex of the bit pattern: exact, so the comparison is over
                    // the values the model produced rather than over a decimal
                    // rendering of them.
                    line.push_str(&format!(" {:08x}", v.to_bits()));
                }
                line.push('\n');
                sink.write_all(line.as_bytes())
                    .context("writing --dump-logits")?;
            }

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

            // Install the ICB executor once the operands are stable. Under
            // `--arena` that is after its recording steps: recording earlier
            // would capture the pool's per-step buffer identities and exclude
            // every position as varying, for a reason that is not the
            // executor's.
            #[cfg(feature = "metal")]
            if let (Some(exec), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
                if decode_steps == icb_installed_at {
                    dev.set_executor(std::sync::Arc::new(
                        candle_metal_kernels::metal::ExecutorSlot::Custom(exec.clone()),
                    ));
                }
            }

            logits = model
                .forward(&input, kv_len, &mut cache)
                .context("decode forward pass")?
                .squeeze(0)?;

            // Close the ICB step here: after the forward pass and before the
            // next iteration samples, so a step is exactly the forward pass.
            // Sampling adds a `softmax_f32` and an `affine_f32`, and including
            // them in some windows and not others makes position N a different
            // dispatch in different steps -- which reads as "everything varies"
            // rather than as a misaligned boundary.
            #[cfg(feature = "metal")]
            if let (Some(exec), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
                if decode_steps >= icb_installed_at {
                    exec.end_step(dev.metal_device())
                        .map_err(|e| anyhow::anyhow!("closing ICB step {decode_steps}: {e}"))?;
                }
            }

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

    // What the ICB executor actually replayed.
    //
    // The digest below is only evidence about replay if replay *happened*, and a
    // flag being passed does not establish that -- #69's determinism run under a
    // new hazard key reported a passing digest for the unchanged path, and was
    // caught by checking that a quantity had moved rather than by trusting the
    // flag (§2.4, §9.2f). A nonzero `covered` here is that quantity: it is the
    // number of dispatches this run did *not* encode classically.
    #[cfg(feature = "metal")]
    if let Some(exec) = icb_executor.as_ref() {
        let cov = exec.coverage();
        println!(
            "ICB replaying={} covered={} of {} runs={} barriers_encoded={} stale={} \
             poisoned_runs={} discarded_windows={} excluded_varies={} excluded_inline_consts={}",
            exec.is_replaying(),
            cov.covered,
            cov.positions,
            exec.runs(),
            exec.encoded_barriers(),
            exec.stale_positions(),
            exec.poisoned_runs(),
            exec.discarded_windows(),
            cov.varies,
            cov.inline_constants,
        );
        println!("ICB barrier scope: {icb_barrier_scope:?}");
        // Issue #144. Printed beside the scope because the two axes are
        // independent and a reader comparing arms needs both: `BarrierScope`
        // governs the ordering the ICB *encodes*, `ReplayBarriers` the ordering
        // candle *still emits* beside it (§11.3n -- additive, not substitutive).
        //
        // `barriers_suppressed` is the quantity that shows the switch engaged,
        // which is the check #69's vacuous determinism run earned (§2.4, §9.2f):
        // both arms there silently ran the default and the "changed" arm
        // reported a passing digest for the unchanged path. **It is engagement
        // proof and never the correctness argument** -- a wrongly-suppressed
        // edge leaves this higher and the barrier count lower, both plausible.
        // The correctness evidence is this harness's own digest.
        println!(
            "ICB replay barriers: {icb_replay_barriers:?}  suppressed={}",
            candle_metal_kernels::metal::trace::barriers_suppressed(),
        );
        // Per kernel, with the reason attached, because the totals above cannot
        // distinguish a position that became *covered* from one that merely
        // changed which bucket excludes it. #103 gave `sdpa_vector` a packed
        // sibling and `excluded_inline_consts` went 8 -> 0 while `covered`
        // stayed at 433: the 8 moved to `varies`. A reader comparing only the
        // totals would read the first half as a win and never see the second.
        for ((kernel, reason), n) in &cov.excluded_by_kernel {
            println!("ICB excluded {n:4} {kernel:<40} {reason}");
        }
        anyhow::ensure!(
            exec.is_replaying() && cov.covered > 0,
            "--icb was passed but nothing was replayed ({} covered of {}), so the digest below \
             would be the classical path's wearing this arm's name",
            cov.covered,
            cov.positions
        );
        anyhow::ensure!(
            exec.stale_positions() == 0,
            "{} replayed positions went stale, so coverage is lower than reported and the \
             digest describes a partially-classical run",
            exec.stale_positions()
        );
    }

    // One machine-greppable line per run, so a batch of runs collapses to a
    // sort|uniq over digests.
    //
    // There is deliberately no throughput field here, and `elapsed_ms` is not one
    // (lloom #102). This harness measures digests; `lfm2-decode-profile` measures
    // throughput, and only its figure is comparable to anything in `.bench/`.
    //
    // A `tok_per_s` was printed until 2026-08-26 and it was a whole-generation
    // average -- `generated / elapsed` -- so prefill, sampling, warmup and a
    // `kv_len` that grows across turns were all inside it.
    // `measurements/issue-7-reconciliation.md` §1 was written to discredit exactly
    // that shape, and it survived here for another twenty-odd issues and misled two
    // readers. Worked from one of its own runs: 267 tokens over 8272 ms reads
    // 32.28 tok/s; backing out three prefills at 90.22 ms leaves 29.97 ms/token
    // against §6.6's 18.763 ms steady-state decode, a factor of 1.60 that prefill
    // alone does not explain.
    //
    // The trap is not that the number is imprecise, it is that the two arms of a
    // determinism A/B generate *different numbers of tokens by construction*: when
    // the digests diverge the arms take different paths and stop at different
    // points (#97's were 267 and 401). So `elapsed_ms` is incomparable between arms
    // and any rate derived from it is incomparable too, sitting exactly where a
    // reader looks for a comparison.
    //
    // `elapsed_ms` stays because §2.3.8's evidence is that wall time varied while
    // the bits did not; it is a duration, not a rate, and dividing it by
    // `generated` reconstructs the discredited figure. Measured while removing the
    // field: every run taken for lloom #102 produced one digest pair, with wall
    // clocks spanning 8291 to 20085 ms as the machine's load moved -- a 2.4x
    // spread on the removed quantity. Sharpest inside a single three-run gate, one
    // binary, minutes apart: 31.90, 13.41 and 13.34 tok/s, from bit-identical
    // output -- and the preceding gate reproduced the same shape, so a 2.4x swing
    // within one gate happened twice out of two
    // (`measurements/issue-102-tok-per-s-removed.md` §4.1).
    println!(
        "RESULT label={} n={} tokens_sha256={} logits_sha256={} prompt_tokens={} generated={} \
         turns={} final_kv_len={} hit_eos={} dtype={:?} device={} seed={} temp={} top_p={} \
         elapsed_ms={}",
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
