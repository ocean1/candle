//! Admission against a real Metal device (lloom #186, `DESIGN.md` §9.5l).
//!
//! The unit tests in `lfm2_admission_tests.rs` are pure arithmetic against a
//! `hw.memsize` stand-in. **This one builds a `Cache` on the device**, which is
//! the only thing that exercises three properties the arithmetic cannot:
//!
//! * that `admit_memory_budget` is *reached* from `Cache::new_with` at all;
//! * that it runs **before** the RoPE tables, which are allocated in that same
//!   function and are the reason the ordering matters (§9.5k);
//! * that the real denominator is used, which is **not** `hw.memsize` —
//!   `recommendedMaxWorkingSetSize` reads **55.663 GB** on this machine against
//!   `hw.memsize`'s 68.719 (§9.5l).
//!
//! **This was worth writing, and the reason is recorded rather than implied.**
//! The unit tests were green while admission was compiling to its non-Metal
//! stub in every harness — `candle-examples`' `metal` feature did not propagate
//! `candle-transformers/metal`, so the `#[cfg(feature = "metal")]` arm was
//! never built. Nothing that tested the *arithmetic* could have seen that; only
//! building a `Cache` and watching it refuse can.

#![cfg(feature = "metal")]

use candle::{DType, Device};
use candle_transformers::models::lfm2::{Cache, Config, LayerType, MemoryBudget};

/// §5.5's language-model weights, which is what a text-only LFM2 load carries.
const WEIGHTS: usize = 5_394_397_184;

fn config(max_position_embeddings: usize) -> Config {
    Config {
        vocab_size: 128_000,
        hidden_size: 2048,
        intermediate_size: 10_752,
        num_hidden_layers: 30,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        norm_eps: 1e-5,
        rope_theta: 1e6,
        max_position_embeddings,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv; 30],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        kv_append: Default::default(),
        conv_state: Default::default(),
        memory_budget: None,
    }
}

/// **Both bounds, on the device** (§8.1g, #184): a configuration inside the
/// budget builds a `Cache`, and one outside it fails to.
#[test]
fn admits_what_fits_and_refuses_what_does_not_on_the_device() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };

    // ADMITTED: B=1 at the shipped capacity. §9.5b puts this well inside.
    let mut ok = config(4096);
    ok.memory_budget = Some(MemoryBudget::new(WEIGHTS));
    Cache::new_with(true, DType::F16, &ok, &device, 4096)
        .expect("the shipping configuration must be admitted");

    // REFUSED: B=32 at 128k, where §9.5b's KV alone is 68.719 GB -- more than
    // this machine's whole working set, before a weight is loaded.
    let mut over = config(131_072);
    let mut budget = MemoryBudget::new(WEIGHTS);
    budget.batch = 32;
    over.memory_budget = Some(budget);
    let err = Cache::new_with(true, DType::F16, &over, &device, 131_072)
        .expect_err("B=32 x 128k must be refused");
    let msg = err.to_string();

    // §9.5g: the refusal carries the arithmetic that produced it, names the
    // dominant class, and lists the levers -- so the next configuration is a
    // decision rather than a guess.
    assert!(msg.contains("memory budget exceeded"), "{msg}");
    assert!(
        msg.contains("<- dominant"),
        "names the class to reduce: {msg}"
    );
    assert!(msg.contains("reductions that would fit"), "{msg}");
    // And the RoPE tables are named, since they are allocated in this very
    // function and are not one of §9.1's five classes (§9.5k).
    assert!(msg.contains("RoPE"), "{msg}");
}

/// **Admission off by default builds exactly as it did before.**
///
/// The off-arm on the device rather than in the abstract: a `Config` with no
/// budget must reach the same `Cache` it always did, including the RoPE tables
/// admission would otherwise have been asked about.
#[test]
fn no_budget_means_no_refusal_on_the_device() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // A configuration admission would refuse outright -- if it were consulted.
    let cfg = config(131_072);
    assert!(cfg.memory_budget.is_none());
    Cache::new_with(true, DType::F16, &cfg, &device, 131_072)
        .expect("with no budget set, nothing is refused however large");
}
