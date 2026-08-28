//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // FFN dimension configuration
    #[serde(default = "default_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f32,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: usize,
}

fn default_num_key_value_heads() -> usize {
    8
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    128000
}

fn default_conv_l_cache() -> usize {
    3
}

fn default_ffn_dim_multiplier() -> f32 {
    1.0
}

fn default_block_multiple_of() -> usize {
    256
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Compute the actual intermediate size for the FFN.
    /// LFM2 uses: hidden_size * 4 * block_ffn_dim_multiplier, rounded to block_multiple_of
    fn compute_intermediate_size(&self) -> usize {
        let base_size = (self.hidden_size as f32 * 4.0 * self.block_ffn_dim_multiplier) as usize;
        let multiple = self.block_multiple_of;
        base_size.div_ceil(multiple) * multiple
    }

    pub fn into_config(self, use_flash_attn: bool) -> Config {
        // Use computed intermediate size (matches actual weights) instead of config field
        let intermediate_size = self.compute_intermediate_size();
        Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types: self.layer_types,
            tie_embedding: self.tie_embedding,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            use_flash_attn,
            attn_impl: AttnImpl::default(),
            kv_append: KvAppend::default(),
            conv_state: ConvState::default(),
        }
    }
}

/// How the decode path writes conv state.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms, in the shape §9.1a uses
/// for `ScratchSizing`: every arm is compiled, the selection is one field, and a
/// regression is revertible without a rebuild.
///
/// # Three arms, and the third moves the digests deliberately
///
/// §10.2a specifies "a rotating base into a `(l_cache + K)`-wide buffer" and
/// predicts the wrap is nearly free because the phase is uniform across the
/// simdgroup (§3.3). The uniformity argument is sound; what that section gets
/// wrong is the premise it rests on. It argues the ring "sums the same three
/// products in the same tap order, differing only in which address each tap
/// reads", and the second clause is false: `sum_keepdim` accumulates in *slot*
/// order, so rotating which slot holds which token rotates the summation order.
/// Float addition is not associative, so the low bits move.
///
/// **That makes the rotating arm a different summation order, not a defect —
/// established by measurement, not by assumption** (issue #141, `DESIGN.md`
/// §10.2g). §2.3.5a names CPU-backend parity as the only load-bearing
/// discriminator between a legitimate reduction-order change and a computational
/// bug, and the rotating arm passes it: exact against an independent f64
/// reference on the reduction it reorders, and its LFM2 error does not grow over
/// a full generation. So it is an axis in the shape `AttnImpl::Sdpa` is
/// (§2.3.8c) — the digests move, deliberately, and the text does not.
///
/// - [`ConvState::SlidingRing`] keeps the window contiguous and in the shuffle's
///   own slot order, so its output is **bit-identical** and it costs a periodic
///   compaction — §10.2a's own second fallback ("a doubled buffer"), generalised
///   with `slack` as how much doubling is paid for.
/// - [`ConvState::RotatingRing`] is §10.2a as specified. It never compacts, so
///   its dispatch count is **constant** and its write offsets take one of
///   `l_cache + k` fixed values — both of which the sliding arm gives up
///   (§10.2e). It moves the digests.
///
/// Neither dominates: the sliding arm buys bit-identity with `slack` bytes and a
/// non-constant dispatch count, the rotating arm buys a constant count and fixed
/// offsets with a moved digest. `Shuffle` remains the default, so an
/// unconfigured caller keeps the path every recorded digest belongs to.
///
/// # Why `k` is a parameter and not a constant
///
/// §16 6b: the ring's resident footprint crosses the snapshot's at
/// `K = l_cache`, so K is a real memory decision once a speculative scheme
/// (#89) can supply an acceptance-rate distribution to size it against — and
/// until then any specific `K > 0` is a number nobody measured. `K = 0` is the
/// default, so the live window is exactly `l_cache` wide and the mechanism costs
/// no resident bytes beyond `slack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvState {
    /// `narrow` + `Tensor::cat` — reallocates and copies the whole state every
    /// token (`DESIGN.md` §6.1). The correctness bar the rings are compared
    /// against, and the arm every recorded digest belongs to.
    #[default]
    Shuffle,
    /// A sliding write index into an `l_cache + k + slack`-wide buffer
    /// (`DESIGN.md` §10.2a, §10.2b). The `cat` disappears: the write becomes one
    /// in-place `slice_set`, and the read is a `narrow` that never wraps.
    ///
    /// Bit-identical to `Shuffle`, because the live window is always `l_cache`
    /// contiguous slots in the shuffle's own order. Costs a compaction every
    /// `slack` tokens, which makes the per-token dispatch count non-constant.
    SlidingRing { k: usize, slack: usize },
    /// A rotating write index into an `l_cache + k`-wide buffer — §10.2a as
    /// originally specified.
    ///
    /// No slack, no compaction, and the write lands at one of `l_cache + k`
    /// fixed offsets, so the per-token dispatch count is constant. **It moves
    /// the digests**, because the summation order rotates with the slot the
    /// newest token occupies; that is a reduction-order change and not a bug —
    /// see this enum's note, and `DESIGN.md` §10.2g for the discriminator runs.
    RotatingRing { k: usize },
}

/// Slack slots when `--conv-state ring` is given without one.
///
/// 16 puts the amortised wrap cost at one extra copy per 17 tokens — 0.06
/// dispatches per conv layer per token against the 2 the shuffle pays — while
/// costing 16 × 2048 × 2 B × 22 = 1.375 MiB of resident state. Past ~64 the
/// curve is flat and the memory is not.
const DEFAULT_RING_SLACK: usize = 16;

impl ConvState {
    /// Extra history slots beyond the live window.
    fn history(&self) -> usize {
        match self {
            ConvState::Shuffle => 0,
            ConvState::SlidingRing { k, .. } | ConvState::RotatingRing { k } => *k,
        }
    }

    /// Total slots allocated: the live window, the speculative history, and —
    /// for the sliding arm only — the slack the window slides through before it
    /// must be compacted. The rotating arm needs none, which is its point.
    fn width(&self, l_cache: usize) -> usize {
        match self {
            ConvState::Shuffle => l_cache,
            ConvState::SlidingRing { k, slack } => l_cache + k + slack,
            ConvState::RotatingRing { k } => l_cache + k,
        }
    }

    /// Parse a harness flag.
    ///
    /// `shuffle` | `ring[:<K>[:<slack>]]` (the sliding arm; `ring` is kept as
    /// the spelling every recorded #141 artifact uses) | `rotating[:<K>]`.
    ///
    /// Lives here rather than in each harness so the harnesses that carry the
    /// flag cannot drift apart on what a spelling means -- the shape §8.1b
    /// argues for, applied to a flag instead of a kernel name.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        if s == "shuffle" {
            return Ok(ConvState::Shuffle);
        }
        let num = |v: &str, what: &str| {
            v.parse::<usize>()
                .map_err(|_| format!("--conv-state: {what} must be an integer, got {v:?}"))
        };
        // Split `<name>[:<field>...]` once, so both arms share one parse of the
        // field list and cannot disagree about how `K` is spelled.
        fn fields<'a>(rest: &'a str, s: &str) -> std::result::Result<Vec<&'a str>, String> {
            match rest {
                "" => Ok(vec![]),
                r => Ok(r
                    .strip_prefix(':')
                    .ok_or_else(|| format!("--conv-state: expected `<arm>:...`, got {s:?}"))?
                    .split(':')
                    .collect()),
            }
        }

        if let Some(rest) = s.strip_prefix("rotating") {
            let parts = fields(rest, s)?;
            let k = match parts.as_slice() {
                [] => 0,
                [k] => num(k, "K")?,
                // No `slack` field: the rotating window never slides, so slack
                // has no meaning here and accepting it silently would let a
                // caller believe they had configured something.
                _ => {
                    return Err(format!(
                        "--conv-state: `rotating` takes at most `:<K>` (it has no slack \
                         -- the window rotates in place), got {s:?}"
                    ))
                }
            };
            return Ok(ConvState::RotatingRing { k });
        }

        let Some(rest) = s.strip_prefix("ring") else {
            return Err(format!(
                "--conv-state must be `shuffle`, `ring`, `ring:<K>`, \
                 `ring:<K>:<slack>`, `rotating` or `rotating:<K>`, got {s:?}"
            ));
        };
        let parts = fields(rest, s)?;
        let (k, slack) = match parts.as_slice() {
            [] => (0, DEFAULT_RING_SLACK),
            [k] => (num(k, "K")?, DEFAULT_RING_SLACK),
            [k, slack] => (num(k, "K")?, num(slack, "slack")?),
            _ => return Err(format!("--conv-state: too many fields in {s:?}")),
        };
        if slack == 0 {
            // A zero-slack window cannot advance, so it would compact on every
            // token -- strictly worse than the shuffle it replaces. `rotating`
            // is the arm for "no slack at all", and it is spelled separately.
            return Err("--conv-state: slack must be >= 1".to_string());
        }
        Ok(ConvState::SlidingRing { k, slack })
    }
}

/// Which attention implementation the decode path takes.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms: it selects a kernel at
/// model-construction time and reaches the GPU as nothing at all. Both arms are
/// compiled, so the A/B is free and a regression is one field away — the same
/// discipline as `ParamStyle` and `ArenaLayout`.
///
/// This is deliberately *not* a `use_flash_attn`-style bool. A third arm
/// (`call_sdpa_vector_2pass`, the chunked variant) is a later issue, and a bool
/// would have to be widened into an enum at that point anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttnImpl {
    /// `repeat_kv` + F32 upcast + `matmul` + `softmax_last_dim`.
    ///
    /// The default, and the correctness bar every other arm is compared
    /// against. Works on every backend.
    #[default]
    Generic,
    /// `candle_nn::ops::sdpa`, which is GQA-native and accumulates in F32
    /// internally, so neither `repeat_kv` nor the upcast is needed.
    ///
    /// Metal only, and only for `seq_len == 1`; every other case falls back to
    /// `Generic` rather than failing, so this is safe to set unconditionally.
    Sdpa,
}

/// How the KV cache grows as decode appends a token.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms, following `AttnImpl`'s
/// shape: both arms are compiled, the old one is the default, and the A/B is one
/// field apart. §7.2 places it here rather than at compile tier because it
/// changes neither addressing inside a kernel nor registers per thread — the
/// kernel sees a different `n` and a different stride, which are dispatch-tier
/// numbers in a descriptor (§15.2 #8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvAppend {
    /// `Tensor::cat(&[cache, &new], 2)` — reallocate and copy the whole cache,
    /// every layer, every token.
    ///
    /// The default and the correctness bar. `DESIGN.md` §6.2 measures what it
    /// costs: the copy grows with `kv_len`, and at 128k it is ~2 GB per token
    /// on top of reading it.
    #[default]
    Cat,
    /// Write the new token in place at a moving offset inside a pre-allocated
    /// buffer, and read the cache back as a `narrow` of it.
    ///
    /// The `cat` disappears, the copy becomes constant-size, and — the property
    /// §11.1a.1 actually wants — the buffer identity stops changing, because
    /// `narrow` shares storage and only rewrites the layout.
    InPlace,
}

/// How much KV a sequence may accumulate before [`KvAppend::InPlace`] declines.
///
/// **This value is not chosen on evidence, and that is deliberate.** See the
/// `Cache::new_with` documentation for why, and for what would decide it.
pub const DEFAULT_KV_CAPACITY: usize = 4096;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub use_flash_attn: bool,
    /// Defaults to `Generic`, so `into_config` and every existing caller keep
    /// the path they had without naming it.
    pub attn_impl: AttnImpl,
    /// Defaults to `Cat`, for the same reason `attn_impl` defaults to `Generic`.
    pub kv_append: KvAppend,
    /// Defaults to `Shuffle`, so every existing caller keeps §6.1's path.
    pub conv_state: ConvState,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// One attention layer's K or V, pre-allocated once and written at a moving
/// offset.
///
/// `DESIGN.md` §6.2's `Tensor::cat` reallocates and copies the whole cache every
/// token; this writes only the new token's bytes and hands back a `narrow` of
/// the buffer. Three consequences, and the third is the one §11.1a.1 is after:
///
/// * the `cat` is gone, so the `copy2d` per layer per token drops from 2 to 1;
/// * the surviving `copy2d` is **constant-size** — one token's `[1, 8, 1, 64]`
///   — where the `cat` copied `kv_len` tokens and grew by 64 rows per step;
/// * **the buffer identity stops changing.** `narrow` clones the storage `Arc`
///   and rewrites only the layout, so every token binds the same `MTLBuffer` at
///   the same base. That is the property ICB replay needs and the reason this
///   is worth more than the copy it removes.
///
/// The read is a *view*, never a copy. `call_sdpa_vector` takes `k_l.stride()`
/// and `n = k_shape[2]`, so a narrowed cache is exactly the shape it is written
/// to consume (`candle-metal-kernels/src/kernels/sdpa.rs`); the generic arm's
/// `repeat_kv` goes through `Tensor::cat`, which handles a strided source.
#[derive(Debug, Clone)]
struct KvSlot {
    /// `[b_sz, n_kv_heads, capacity, head_dim]`, allocated on first append.
    ///
    /// `None` until then, because the batch size is not known at `Cache::new`.
    /// This mirrors `candle_nn::kv_cache::Cache`, deliberately: that type is the
    /// same mechanism and was read before this was written (see the note on
    /// `Cache::new_with` for why it is not reused directly).
    all_data: Option<Tensor>,
    /// How many tokens of `all_data` are live. The `narrow` length.
    len: usize,
    capacity: usize,
}

impl KvSlot {
    fn new(capacity: usize) -> Self {
        Self {
            all_data: None,
            len: 0,
            capacity,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
        self.all_data = None;
    }

    /// Append `src` along dim 2 and return the live prefix as a view.
    ///
    /// Returns `Err` rather than reallocating when the capacity is exhausted —
    /// see `Cache::new_with` for why that is a loud failure and not a `Grow`.
    fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(2)?;
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[2] = self.capacity;
            self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }
        let all_data = self.all_data.as_ref().unwrap();
        if self.len + seq_len > self.capacity {
            candle::bail!(
                "lfm2 KV cache exhausted: {} + {} > capacity {}. \
                 Raise `Config::kv_capacity`; see `DESIGN.md` §6.2a on why no policy is chosen.",
                self.len,
                seq_len,
                self.capacity
            )
        }
        // `slice_set` requires both sides contiguous and writes one `copy2d` of
        // exactly `src`'s bytes at `offset * block_size` (`tensor_cat.rs`). The
        // caller has already made `src` contiguous.
        all_data.slice_set(src, 2, self.len)?;
        self.len += seq_len;
        all_data.narrow(2, 0, self.len)
    }
}

/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    // KV cache for attention layers: (key, value) per layer
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // The `KvAppend::InPlace` arm's storage, one (K, V) pair per layer.
    // Held beside `kvs` rather than replacing it so that both arms stay
    // compiled and the unselected one allocates nothing.
    kv_slots: Vec<(KvSlot, KvSlot)>,
    kv_append: KvAppend,
    // Conv state cache for convolution layers
    conv_states: Vec<Option<Tensor>>,
    /// The ring's rotating write index, in slots.
    ///
    /// **Deliberately here and not in a params struct** (`DESIGN.md` §10.2c,
    /// §10.5, §11.3): it is per-sequence *conv state*, so it migrates and evicts
    /// with the sequence exactly as the bytes do, and putting it in a struct
    /// named for KV is the conflation §10.1 renames the abstraction to avoid.
    ///
    /// One index for all 22 conv layers rather than one each: every layer is
    /// written exactly once per token, so their phases are equal at every point
    /// a layer reads them. Per-layer indices would be 22 copies of one number.
    conv_phase: usize,
    /// Set by `Model::forward` when the write slot has run out of slack, so the
    /// conv layers compact their window before writing this token.
    ///
    /// **The decision lives in exactly one place on purpose.** An earlier draft
    /// had `Model::forward` wrap the index *and* the layers test `phase >=
    /// width` — two owners of one decision, which is a shape that cannot be
    /// right: the wrap made the layers' test unreachable, so the compaction
    /// never ran and the window silently read stale slots from the 17th token
    /// on. Caught by the bitwise parity test, which is what it is for.
    conv_compact: bool,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        Self::new_with(use_kv_cache, dtype, config, device, DEFAULT_KV_CAPACITY)
    }

    /// As [`Cache::new`], with the [`KvAppend::InPlace`] capacity given
    /// explicitly.
    ///
    /// # The capacity is not decided, and this says so rather than picking
    ///
    /// A moving write offset needs a bound on how far it may move. `DESIGN.md`
    /// §9.1a faced the identical question for the scratch class and built
    /// `Reserve`, `Grow` and `Bucket` while **choosing none**, because the axis
    /// that separates them is `kv_len` and the largest value this project has
    /// ever recorded is **2720** against a 128k target (§13.2). This issue hits
    /// the same wall, and the same answer is the honest one.
    ///
    /// What the three policies would cost here, computed from §5.6's 16 KiB per
    /// token across all 8 attention layers at B=1, f16:
    ///
    /// | `kv_len` | `Reserve` (128k) | `Grow` | `Bucket` |
    /// |---|---|---|---|
    /// | 2,720 — the largest ever measured | 2048 MiB | 42.5 MiB | 64 MiB |
    /// | 32,768 | 2048 MiB | 512 MiB | 512 MiB |
    /// | 131,072 | 2048 MiB | 2048 MiB | 2048 MiB |
    ///
    /// They converge at the target and differ by ~48× in the regime we can
    /// actually reach — which is precisely the shape that makes a measurement
    /// here uninformative about the choice.
    ///
    /// **What is chosen instead is a parameter with a default and a loud
    /// failure.** `DEFAULT_KV_CAPACITY` is 4096: above the 2720 ceiling with
    /// headroom, 64 MiB against a 5.2 GB pool (§6.3b), and small enough that
    /// nobody mistakes it for a considered long-context answer. Exceeding it is
    /// an error, not a silent reallocation — a `Grow` arm would put the
    /// `Tensor::cat` this change exists to remove back on the path at exactly
    /// the moment the cache is largest, and would move the buffer identity that
    /// §11.1a.1 wants stable. Failing loudly keeps both properties true or
    /// visibly absent, which is `DESIGN.md` §2.4's rule about instruments that
    /// cannot be shown to have engaged.
    ///
    /// **What would decide it** is a context-length curve — issue #61, and
    /// `DESIGN.md` §13.2 records that no preset has evidence for the same
    /// reason. Concretely: ms/token and peak footprint at `kv_len` across
    /// 2k/8k/32k/128k, with the three policies as arms. Until that exists, any
    /// value here is a guess, and this one is labelled as one.
    ///
    /// # Why not `candle_nn::kv_cache::KvCache`
    ///
    /// It is the same mechanism — `slice_set` at a moving offset, `narrow` to
    /// read — and it was read before this was written; seven models in
    /// `candle-transformers` already use it. It is not reused here because its
    /// `append` implements exactly the `Grow` arm this function declines: on
    /// exhaustion it does `Tensor::cat` with a fresh block
    /// (`candle-nn/src/kv_cache.rs`), which reintroduces the reallocation and
    /// moves the buffer identity. Taking `KvCache` would have been the smaller
    /// diff and would have silently chosen the policy this section exists to
    /// leave open. Making that a *choice* rather than an inheritance is the
    /// reason for the local type, and it is a difference of about forty lines.
    pub fn new_with(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
        kv_capacity: usize,
    ) -> Result<Self> {
        let theta = calculate_default_inv_freq(config);
        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            kv_slots: (0..num_layers)
                .map(|_| (KvSlot::new(kv_capacity), KvSlot::new(kv_capacity)))
                .collect(),
            kv_append: config.kv_append,
            conv_states: vec![None; num_layers],
            conv_phase: 0,
            conv_compact: false,
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    /// Drop every sequence-state tensor, returning the cache to its pre-prefill
    /// state.
    ///
    /// Both KV arms are reset, whichever is selected. For `InPlace` that means
    /// the buffer is dropped rather than the length being rewound: a new
    /// sequence may have a different batch size, so keeping the allocation
    /// would pin `b_sz` from the previous one.
    pub fn clear(&mut self) {
        self.kvs.iter_mut().for_each(|v| *v = None);
        self.kv_slots.iter_mut().for_each(|(k, v)| {
            k.reset();
            v.reset();
        });
        self.conv_states.iter_mut().for_each(|v| *v = None);
        self.conv_phase = 0;
        self.conv_compact = false;
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

/// MLP layer with SwiGLU activation.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        // LFM2 uses w1 (gate), w3 (up), w2 (down) naming convention
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("w1"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("w3"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("w2"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Attention layer with per-head QK normalization and RoPE.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    attn_impl: AttnImpl,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(
            num_attention_heads * head_dim,
            hidden_size,
            vb.pp("out_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            attn_impl: cfg.attn_impl,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    /// Whether this call can take the fused `sdpa_vector` kernel.
    ///
    /// Every condition is one `candle_nn::ops::sdpa` would otherwise `bail!` on
    /// (`candle-nn/src/ops.rs`, `Sdpa::metal_fwd`), so a false here is a
    /// fallback and never a failure. They are checked rather than assumed:
    ///
    /// * **Metal.** `Sdpa::cpu_fwd` bails outright; there is no CPU arm.
    /// * **`seq_len == 1`.** `supports_sdpa_vector` requires it. Prefill is
    ///   `call_sdpa_full`, whose mask handling differs — deliberately out of
    ///   scope, see the type-level note on `AttnImpl::Sdpa`.
    /// * **dtype.** The kernel is instantiated for f16/bf16/f32 only.
    ///
    /// `head_dim == 64` and `32 % 8 == 0` also hold for LFM2 (§5.2) and are
    /// checked by `ops::sdpa` itself; they are not re-asserted here, because a
    /// second copy of a condition is a second thing to keep in sync (§8.1b).
    fn sdpa_applies(&self, q: &Tensor, seq_len: usize) -> bool {
        self.attn_impl == AttnImpl::Sdpa
            && q.device().is_metal()
            && seq_len == 1
            && matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) then transpose to (batch, num_heads, seq, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply per-head QK normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // Handle KV cache
        let (k, v) = if !cache.use_kv_cache {
            (k, v)
        } else {
            match cache.kv_append {
                KvAppend::Cat => {
                    let (k, v) = match &cache.kvs[block_idx] {
                        Some((k_cache, v_cache)) if index_pos > 0 => {
                            let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                            let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                            (k, v)
                        }
                        _ => (k, v),
                    };
                    cache.kvs[block_idx] = Some((k.clone(), v.clone()));
                    (k, v)
                }
                KvAppend::InPlace => {
                    // `slice_set` requires a contiguous source. `k` comes from
                    // `rope`, which returns a fresh contiguous tensor; `v` was
                    // made contiguous above. Asserted rather than assumed,
                    // because a non-contiguous source is an error here and a
                    // silent full copy under `Tensor::cat`.
                    let k = k.contiguous()?;
                    let v = v.contiguous()?;
                    let (k_slot, v_slot) = &mut cache.kv_slots[block_idx];
                    // A turn boundary re-prefills without clearing, so this is
                    // a `seq_len > 1` append onto a non-empty slot. It is the
                    // same call — the offset simply advances by more than one.
                    let k = k_slot.append(&k)?;
                    let v = v_slot.append(&v)?;
                    (k, v)
                }
            }
        };

        // The fused kernel is GQA-native — it derives `gqa_factor` from the head
        // counts and indexes `kv_head_idx = head_idx / gqa_factor` in registers
        // — and it accumulates in F32 internally. So on that arm `repeat_kv` and
        // the F32 upcast below are not merely unnecessary, they are the two
        // things being removed (`DESIGN.md` §6.2, §8.1 principle 4). Both
        // therefore have to stay *inside* the arm that needs them.
        let y = if self.sdpa_applies(&q, seq_len) {
            // Scale is applied to `q` inside the kernel before the dot product,
            // matching `att / sqrt(head_dim)` below. No mask: `seq_len == 1`
            // attends to the whole cache, which is the same reason the generic
            // arm skips `masked_fill` in that case.
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1f32 / (self.head_dim as f32).sqrt(),
                1.0,
            )?
        } else if self.use_flash_attn {
            let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
            let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            // Expand KV heads to match query heads
            let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
            let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&y)
    }
}

/// Short convolution layer for efficient sequence processing.
#[derive(Debug, Clone)]
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
    /// One weight permutation per rotation phase, built at load time for
    /// `ConvState::RotatingRing` and `None` otherwise. See `ShortConv::new`.
    rotating_weights: Option<Vec<Tensor>>,
    conv_state: ConvState,
    span: tracing::Span,
}

impl ShortConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let l_cache = cfg.conv_l_cache;

        // in_proj projects to 3 * hidden_size for B, C, X components
        let in_proj = linear(hidden_size, 3 * hidden_size, vb.pp("in_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        // Conv weight shape: (hidden_size, 1, l_cache) or (hidden_size, l_cache)
        let conv_weight = vb.get((hidden_size, 1, l_cache), "conv.weight")?;

        // The rotating arm's per-phase weights, precomputed at load time.
        //
        // **This is what makes the rotating form compute the right convolution
        // rather than a different one.** A rotating buffer's slot `s` holds the
        // token of age `(s + width - phase - 1) mod width`, so multiplying slot
        // `s` by weight `s` -- which is what a naive rotation does -- pairs each
        // tap with the wrong weight. Measured, that is an **O(1)** error (~1e7
        // ulp), not a rounding: it is a different operator, and it agrees with
        // the shuffle only on the one phase in `width` where the rotation is the
        // identity.
        //
        // Permuting the weights by phase restores the pairing exactly, and the
        // residual against the shuffle is then bounded by **1 ulp** and flat in
        // run length -- a reduction-order difference and nothing more
        // (`DESIGN.md` §10.2g). The weight is a `[hidden, l_cache]` constant, so
        // all `width` permutations are built once here and cost nothing per
        // token; §10.2a lists this as its third fallback.
        let rotating_weights = match cfg.conv_state {
            ConvState::RotatingRing { .. } => {
                let width = cfg.conv_state.width(l_cache);
                let w = conv_weight.squeeze(1)?;
                let mut per_phase = Vec::with_capacity(width);
                for phase in 0..width {
                    // Slot `s` must meet the weight of the token it holds. Ages
                    // run 0 = oldest .. l_cache-1 = newest, and the newest sits
                    // at `phase`, so slot `s` holds age `s + width - phase - 1`
                    // (mod width) -- taken over the live window only.
                    let mut cols = Vec::with_capacity(width);
                    for s in 0..width {
                        let age = (s + width - phase - 1) % width;
                        // Slots outside the live window hold history (K > 0),
                        // which nothing reads yet; give them a zero column so a
                        // future reader cannot silently pick up a live weight.
                        let col = if age < l_cache {
                            w.narrow(1, age, 1)?
                        } else {
                            Tensor::zeros((hidden_size, 1), w.dtype(), w.device())?
                        };
                        cols.push(col);
                    }
                    per_phase.push(Tensor::cat(&cols, 1)?.contiguous()?);
                }
                Some(per_phase)
            }
            _ => None,
        };

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            rotating_weights,
            l_cache,
            hidden_size,
            conv_state: cfg.conv_state,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    fn forward(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        // Project input to B, C, X components
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;

        // Element-wise multiply B and X
        let bx = (b * &x_proj)?.contiguous()?;

        // Prepare conv weight: squeeze to (hidden_size, l_cache) for element-wise, or keep for Conv1d
        let conv_weight = self.conv_weight.squeeze(1)?;

        let conv_out = if seq_len == 1 {
            match self.conv_state {
                ConvState::RotatingRing { .. } => {
                    // §10.2a as specified: the window IS the buffer, and the
                    // write index rotates through it. No slack, so no compaction
                    // and a constant dispatch count -- and the write lands at one
                    // of `width` fixed offsets rather than at a sliding one.
                    //
                    // **The read is the whole buffer against a phase-permuted
                    // weight**, which is what keeps it the same convolution. A
                    // naive rotation multiplies slot `s` by weight `s` and so
                    // pairs each tap with the wrong weight -- an O(1) error, not
                    // a rounding. With the permutation the products are exactly
                    // the shuffle's; only the order `sum_keepdim` accumulates
                    // them in differs, so the output moves within 1 ulp and the
                    // LFM2 digests move with it. That is a reduction-order
                    // change, discharged against §2.3.5a's discriminators rather
                    // than assumed -- `DESIGN.md` §10.2g.
                    let width = self.conv_state.width(self.l_cache);
                    let state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => {
                            Tensor::zeros((b_sz, self.hidden_size, width), bx.dtype(), bx.device())?
                        }
                    };

                    // Advanced once per token by `Model::forward`, for the same
                    // reason the sliding arm's is: 22 conv layers share one index.
                    let phase = cache.conv_phase;
                    state.slice_set(&bx, 2, phase)?;

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // Built at load time, one per phase -- so the per-token cost
                    // is an index rather than a permutation (§15.2 #10: the
                    // answer is computed when the plan is built, not per token).
                    let w = self
                        .rotating_weights
                        .as_ref()
                        .expect("rotating weights are built whenever the arm is RotatingRing")
                        .get(phase)
                        .expect("phase is taken modulo width, so it indexes the table");
                    (state * w.unsqueeze(0)?)?.sum_keepdim(2)?.contiguous()?
                }
                ConvState::SlidingRing { .. } => {
                    let width = self.conv_state.width(self.l_cache);
                    // The ring is written in place, so it must exist as storage
                    // before the write rather than being produced by it.
                    let mut state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => {
                            Tensor::zeros((b_sz, self.hidden_size, width), bx.dtype(), bx.device())?
                        }
                    };

                    // `conv_phase` is the write slot, advanced once per token by
                    // `Model::forward` -- not here, because all 22 conv layers
                    // share one index and advancing it per layer would move it 22
                    // times per token. `conv_compact` comes from the same place,
                    // so this layer never decides where the window sits.
                    let phase = cache.conv_phase;

                    // The window has run out of slack: slide it back to the
                    // front. **This is the entire wrap cost** (`DESIGN.md`
                    // §16 6a), and it is paid once every `slack` tokens rather
                    // than on every token. The live window keeps its slot order
                    // across the move, so the compaction is invisible to the
                    // arithmetic.
                    if cache.conv_compact {
                        let live_w = self.l_cache + self.conv_state.history();
                        let live = state.narrow(2, width - live_w, live_w)?.contiguous()?;
                        let pad = Tensor::zeros(
                            (b_sz, self.hidden_size, width - live_w),
                            bx.dtype(),
                            bx.device(),
                        )?;
                        state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
                    }

                    // The whole point: one in-place write at the sliding slot.
                    // `slice_set` is a single `copy2d` into storage that already
                    // exists, so the `narrow` and the `cat` both disappear -- 44
                    // dispatch positions per token become 22 (§10.2b).
                    state.slice_set(&bx, 2, phase)?;

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // The live window is `l_cache` *contiguous* slots ending at
                    // the write, in the same order the shuffle presents them --
                    // which is what keeps the three-term sum's accumulation order
                    // unchanged and the output bit-identical. A rotating index
                    // would not: see `ConvState`'s note on §10.2a.
                    let window = state.narrow(2, phase + 1 - self.l_cache, self.l_cache)?;
                    (window * conv_weight.unsqueeze(0)?)?
                        .sum_keepdim(2)?
                        .contiguous()?
                }
                ConvState::Shuffle => {
                    // Token-by-token generation: use cached state
                    let mut state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => Tensor::zeros(
                            (b_sz, self.hidden_size, self.l_cache),
                            bx.dtype(),
                            bx.device(),
                        )?,
                    };

                    // Shift cache and add new token
                    if self.l_cache > 1 {
                        let tail = state.narrow(2, 1, self.l_cache - 1)?;
                        state = Tensor::cat(&[tail, bx.clone()], 2)?;
                    } else {
                        state = bx.clone();
                    }

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // Apply convolution as element-wise multiply and sum
                    (state * conv_weight.unsqueeze(0)?)?
                        .sum_keepdim(2)?
                        .contiguous()?
                }
            }
        } else {
            // Prefill: use Conv1d
            let conv = Conv1d::new(
                self.conv_weight.clone(),
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: self.hidden_size,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx)?;
            out = out.narrow(2, 0, seq_len)?;

            // Update cache with last l_cache tokens
            if cache.use_kv_cache && self.l_cache > 0 {
                let start = seq_len.saturating_sub(self.l_cache);
                let cache_len = seq_len - start;
                let mut cache_src = bx.narrow(2, start, cache_len)?;
                if cache_len < self.l_cache {
                    let pad = self.l_cache - cache_len;
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, pad),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                // Under the ring the buffer is `l_cache + K + slack` wide and
                // decode writes into it in place -- so prefill must hand over
                // storage of the *full* width, with the live window at the front
                // (write slot `live_w - 1`, which is what `Model::forward` seeds)
                // and everything after it zeroed. Prefill is off the per-token
                // path, so this `cat` is paid once per prompt, not once per token.
                let width = self.conv_state.width(self.l_cache);
                if width > self.l_cache {
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, width - self.l_cache),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[cache_src, zeros], 2)?;
                }
                cache.conv_states[block_idx] = Some(cache_src.contiguous()?);
            }

            out
        };

        // Multiply by C and project output
        let conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

/// Unified decoder layer supporting both attention and convolution.
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(Box<Attention>),
    ShortConv(ShortConv),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        // LFM2 uses operator_norm and ffn_norm naming
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        // LFM2 uses feed_forward naming for MLP
        let mlp = Mlp::new(cfg, vb.pp("feed_forward"))?;

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"))?),
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            kind,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let x = match &self.kind {
            LayerKind::Attention(attn) => attn.forward(&x, index_pos, block_idx, cache)?,
            LayerKind::ShortConv(conv) => conv.forward(&x, block_idx, cache)?,
        };

        let x = (x + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
    conv_state: ConvState,
    l_cache: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            dtype: vb.dtype(),
            conv_state: cfg.conv_state,
            l_cache: cfg.conv_l_cache,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // The ring's write slot, advanced once per forward pass and read by all
        // 22 conv layers (`DESIGN.md` §10.2c -- it is conv state, so it lives in
        // `Cache` beside the bytes it indexes rather than in a params struct).
        //
        // Prefill writes the state directly rather than through the ring, and
        // leaves the newest token in the last live slot -- which is slot
        // `l_cache + K - 1`. Decode advances from there. Advancing *before* the
        // layers run rather than after is what makes the first decode token
        // after a prefill land past the newest token instead of overwriting it.
        //
        // **One owner for the wrap decision.** This computes both the slot and
        // whether the layers must compact to reach it; the layers only obey. An
        // earlier draft split it -- this wrapped the index while the layers
        // tested `phase >= width` -- and the wrap made the layers' test
        // unreachable, so the compaction never ran.
        if let ConvState::RotatingRing { .. } = self.conv_state {
            // The rotating arm never compacts -- that is the whole difference.
            // The index wraps modulo the buffer width, which is what makes the
            // dispatch count constant and the write offsets a fixed finite set.
            let width = self.conv_state.width(self.l_cache);
            cache.conv_compact = false;
            cache.conv_phase = if seq_len == 1 {
                (cache.conv_phase + 1) % width
            } else {
                // Prefill lays the live window down at slots `0..l_cache`, so
                // the newest token is at `l_cache - 1` -- the same seeding the
                // sliding arm uses, and for the same reason.
                self.l_cache - 1
            };
        } else if let ConvState::SlidingRing { .. } = self.conv_state {
            let width = self.conv_state.width(self.l_cache);
            let live_w = self.l_cache + self.conv_state.history();
            if seq_len == 1 {
                let next = cache.conv_phase + 1;
                cache.conv_compact = next >= width;
                // After a compaction the live window sits at the front, so the
                // next write lands immediately after it.
                cache.conv_phase = if cache.conv_compact { live_w } else { next };
            } else {
                // Seeded, not advanced: prefill's own write defines the slot.
                //
                // Prefill lays the live window down at slots `0..l_cache` and
                // zeroes the rest, so the newest token is at `l_cache - 1` --
                // **not** `live_w - 1`. The two coincide at `K = 0` and diverge
                // for any `K > 0`, so spelling it `l_cache` is what keeps the
                // history slots from being read as live.
                cache.conv_compact = false;
                cache.conv_phase = self.l_cache - 1;
            }
        }

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (block_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }

        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two arms' shapes and contents, built the way `Attention::forward`
    /// builds them: `[b_sz, n_kv_heads, seq, head_dim]`, appended along dim 2.
    fn token(step: usize, dev: &Device) -> Result<Tensor> {
        // Distinct values per step and per element, so a transposed or
        // misaligned write is visible rather than accidentally equal. #11.3l's
        // finding 3 is the rule: a fixture whose values coincide proves nothing.
        let n: Vec<f32> = (0..2 * 3).map(|i| (step * 100 + i) as f32).collect();
        Tensor::from_vec(n, (1, 2, 1, 3), dev)
    }

    /// The reference: what `Tensor::cat` produces, step by step.
    fn cat_reference(steps: usize, dev: &Device) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for s in 0..steps {
            let t = token(s, dev)?;
            acc = Some(match acc {
                None => t,
                Some(a) => Tensor::cat(&[&a, &t], 2)?.contiguous()?,
            });
        }
        acc.ok_or_else(|| candle::Error::Msg("no steps".into()))
    }

    /// **The parity test.** In-place append must equal `Tensor::cat` exactly.
    ///
    /// Bit-equality rather than a tolerance: this moves bytes and does no
    /// arithmetic, so `DESIGN.md` §2.3.5a's changed-digest procedure does not
    /// apply and anything but equality is a defect.
    #[test]
    fn in_place_append_matches_cat() -> Result<()> {
        let dev = Device::Cpu;
        for steps in 1..=6 {
            let mut slot = KvSlot::new(8);
            let mut got = None;
            for s in 0..steps {
                got = Some(slot.append(&token(s, &dev)?)?);
            }
            let got = got.unwrap();
            let want = cat_reference(steps, &dev)?;
            assert_eq!(got.dims(), want.dims(), "shape at {steps} steps");
            assert_eq!(
                got.flatten_all()?.to_vec1::<f32>()?,
                want.flatten_all()?.to_vec1::<f32>()?,
                "values at {steps} steps"
            );
        }
        Ok(())
    }

    /// A turn boundary: `seq_len > 1` appended onto a non-empty slot.
    ///
    /// This is the case `lfm2-smoke`'s turn 2 exercises and the one §11.1a's
    /// single-turn limitation is the precedent for — a second prefill writes
    /// several positions at a non-zero offset. It is the same `slice_set` call
    /// with a larger `src`, and the point of the test is that nothing special
    /// happens.
    #[test]
    fn in_place_survives_a_multi_token_append_at_a_nonzero_offset() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        // A 3-token "prefill" on top of 2 decoded tokens.
        let prefill = Tensor::cat(&[&token(2, &dev)?, &token(3, &dev)?, &token(4, &dev)?], 2)?
            .contiguous()?;
        let got = slot.append(&prefill)?;
        let want = cat_reference(5, &dev)?;
        assert_eq!(got.dims(), &[1, 2, 5, 3]);
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    /// The read is a *view*, and that is the property §11.1a.1 wants.
    ///
    /// A `narrow` clones the storage `Arc` and rewrites only the layout, so one
    /// allocation backs every step and the binding is the same `MTLBuffer` every
    /// token. If this silently became a copy, the dispatch count and every
    /// digest would be **unchanged** and only the buffer identity would move —
    /// which no gate in this project can see. Hence a test.
    ///
    /// Observed by aliasing rather than by pointer equality: `same_storage` is
    /// `pub(crate)` to `candle-core`, and widening a core API for one assertion
    /// is a larger change than the assertion is worth. A later `append` writes
    /// through the backing buffer, so if the earlier return value was a view it
    /// sees nothing (the write lands past its length) while its *own* bytes stay
    /// readable — and if it were a copy, `all_data` and the view would have
    /// diverged. The discriminating check is that the backing buffer carries
    /// every step's bytes at the offsets the views reported.
    #[test]
    fn the_read_is_a_view_of_the_preallocated_buffer() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        let first = slot.append(&token(0, &dev)?)?;
        let first_before = first.flatten_all()?.to_vec1::<f32>()?;
        let second = slot.append(&token(1, &dev)?)?;

        // The step-2 write must not have disturbed step 1's region.
        assert_eq!(
            first.flatten_all()?.to_vec1::<f32>()?,
            first_before,
            "appending must not rewrite the live prefix"
        );

        // Both views must be prefixes of one buffer: read the backing store at
        // full capacity and check each view equals its own leading slice.
        let backing = slot.all_data.as_ref().unwrap();
        for (steps, view) in [(1usize, &first), (2usize, &second)] {
            let want = backing
                .narrow(2, 0, steps)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                view.flatten_all()?.to_vec1::<f32>()?,
                want,
                "the {steps}-step view must be the buffer's own prefix"
            );
        }
        Ok(())
    }

    /// Exhaustion is an error, not a silent reallocation.
    ///
    /// `candle_nn::kv_cache::Cache` grows by `Tensor::cat` here; this declines,
    /// because growing would put back the reallocation this change removes and
    /// would move the buffer identity, both at the moment the cache is largest.
    /// See `Cache::new_with` for why no policy is chosen.
    #[test]
    fn exceeding_capacity_fails_loudly() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(2);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        let err = slot.append(&token(2, &dev)?).unwrap_err().to_string();
        assert!(
            err.contains("KV cache exhausted"),
            "expected a capacity error, got: {err}"
        );
        Ok(())
    }

    /// `clear()` must reach the in-place slots too.
    ///
    /// Resetting only `kvs` would leave a stale buffer that the next sequence
    /// appends *after*, so its first token would read `len` zeros in front of
    /// it — a wrong answer that every shape check passes.
    #[test]
    fn reset_returns_the_slot_to_its_pre_prefill_state() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        slot.reset();
        let got = slot.append(&token(0, &dev)?)?;
        assert_eq!(got.dims(), &[1, 2, 1, 3], "a reset slot starts at length 1");
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            cat_reference(1, &dev)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
