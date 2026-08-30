//! Does the speculative verify path work at `B > 1`? — issue #258
//!
//! `DESIGN.md` §13.4b established that `B` enters kernel selection: at `m == 1`
//! the weight matmuls dispatch GEMV and at `m >= 2` a 32×32-tiled GEMM. #258's
//! census establishes the same switch on the **K** axis, at the same tile —
//! `gemm_nt_f16_f16_32_32_16_2_2`, 167 of them, identical to #249's B=2 arm.
//!
//! That makes `m = K × B` the quantity, and the iso-M experiment #258 proposes
//! — hold `K × B = 32` and split it three ways — needs `forward_all` to accept
//! a `[B, seq_len]` input. `Model::forward_all` takes `dims2()` (`lfm2.rs`),
//! so the **shape is expressible**; whether it *works* is a different claim and
//! is what this file establishes, by execution rather than by reading.
//!
//! It runs on the CPU. The question is about shapes and about which rows come
//! out equal, not about a kernel, and the CPU backend is bit-stable and needs
//! no GPU (`CONTRIBUTING.md` §3.1). A **kernel-selection** claim would need
//! Metal and is not made here.

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};

/// A two-layer LFM2 — one conv layer, one attention layer — small enough to
/// build from random weights and large enough to exercise both paths.
///
/// Duplicated from `lfm2_spec_conv_rollback.rs` rather than shared: integration
/// tests are separate crates, and a `mod common` would be a third file for two
/// callers. The config is pinned by `Config`'s own field list, so a drift
/// between the copies is a compile error rather than a silent divergence.
fn tiny_config(conv_state: ConvState) -> Config {
    Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        kv_append: KvAppend::InPlace,
        conv_state,
        memory_budget: None,
    }
}

fn build(cfg: &Config, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Model::new(cfg, vb)
}

/// **The question #258 asks second, and it is answered here rather than
/// assumed from `dims2()`.**
///
/// A verify pass at `B` rows and `k` positions must produce `[B, k, vocab]`,
/// and — fed `B` identical rows — every row must carry the same logits as the
/// `B = 1` pass over the same tokens. That second half is #249's per-row check
/// (§13.4b), applied to `forward_all` instead of to `forward`: at `B = N` with
/// identical prompts every row must reproduce the `B = 1` stream, and it is a
/// stronger gate than a shape assertion because a shape can be right while the
/// rows are mixed.
#[test]
fn forward_all_runs_at_b_greater_than_one_and_rows_agree_with_b1() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(ConvState::RotatingRing { k });
    let model = build(&cfg, &dev)?;

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let window: Vec<u32> = vec![6, 7, 8, 9];
    assert_eq!(window.len(), k);

    // --- B = 1 reference ----------------------------------------------------
    let mut c1 = Cache::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut c1)?;
    let tok1 = c1.advance(k)?;
    let inp = Tensor::new(window.as_slice(), &dev)?.reshape((1, k))?;
    let want = model.forward_all(&inp, prompt.len(), &mut c1)?;
    c1.resolve(tok1, k)?;
    assert_eq!(
        want.dims(),
        &[1, k, cfg.vocab_size],
        "B=1 forward_all must return [B, seq_len, vocab]"
    );

    // --- B = 3, every row the same tokens -----------------------------------
    //
    // Identical rows is what makes the comparison a statement about the batch
    // dimension rather than about the tokens: any per-row difference is the
    // batching, because the inputs cannot account for one.
    let b = 3usize;
    let mut cb = Cache::new(true, DType::F32, &cfg, &dev)?;
    let rows: Vec<u32> = prompt
        .iter()
        .cycle()
        .take(b * prompt.len())
        .copied()
        .collect();
    let inp = Tensor::new(rows.as_slice(), &dev)?.reshape((b, prompt.len()))?;
    model.forward(&inp, 0, &mut cb)?;

    let tokb = cb.advance(k)?;
    let rows: Vec<u32> = window.iter().cycle().take(b * k).copied().collect();
    let inp = Tensor::new(rows.as_slice(), &dev)?.reshape((b, k))?;
    let got = model.forward_all(&inp, prompt.len(), &mut cb)?;
    cb.resolve(tokb, k)?;

    assert_eq!(
        got.dims(),
        &[b, k, cfg.vocab_size],
        "B>1 forward_all must return [B, seq_len, vocab]"
    );

    // Every row equals the B=1 pass, to f32 tolerance. Not bit-equality: the
    // batched arm reduces over a different tensor shape, which is the same
    // class of difference §2.3.5a classifies as a reordering, and this test is
    // about whether the rows are the *same computation* rather than about the
    // low bits.
    let want0 = want.i(0)?.flatten_all()?.to_vec1::<f32>()?;
    for r in 0..b {
        let row = got.i(r)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(row.len(), want0.len());
        let worst = row
            .iter()
            .zip(&want0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-4,
            "row {r} of a B={b} verify pass diverges from the B=1 pass by {worst:e}"
        );
    }
    Ok(())
}

/// **The mutation this file owes**, and it is what makes the test above a test
/// rather than a display (`CONTRIBUTING.md` §3.1 #2).
///
/// Feeding one row a *different* window must make the comparison fail. Without
/// this arm, a `forward_all` that broadcast row 0 over the batch — the exact
/// defect the row check exists to catch — would pass the test above, because
/// every row would equal the B=1 result by construction.
#[test]
fn the_row_check_can_fail_when_a_row_differs() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(ConvState::RotatingRing { k });
    let model = build(&cfg, &dev)?;

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let b = 2usize;
    let mut cb = Cache::new(true, DType::F32, &cfg, &dev)?;
    let rows: Vec<u32> = prompt
        .iter()
        .cycle()
        .take(b * prompt.len())
        .copied()
        .collect();
    let inp = Tensor::new(rows.as_slice(), &dev)?.reshape((b, prompt.len()))?;
    model.forward(&inp, 0, &mut cb)?;

    // Row 0 gets `window`, row 1 gets a different one.
    let tokb = cb.advance(k)?;
    let rows: Vec<u32> = vec![6, 7, 8, 9, 10, 11, 12, 13];
    let inp = Tensor::new(rows.as_slice(), &dev)?.reshape((b, k))?;
    let got = model.forward_all(&inp, prompt.len(), &mut cb)?;
    cb.resolve(tokb, k)?;

    let r0 = got.i(0)?.flatten_all()?.to_vec1::<f32>()?;
    let r1 = got.i(1)?.flatten_all()?.to_vec1::<f32>()?;
    let worst = r0
        .iter()
        .zip(&r1)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst > 1e-4,
        "two rows fed different tokens produced the same logits (worst {worst:e}), so the \
         row comparison in this file cannot distinguish a per-row result from a broadcast one"
    );
    Ok(())
}
