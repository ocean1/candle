//! Device-level behaviour of the Metal buffer pool (lloom issue #21).
//!
//! `candle-metal-kernels` tests the pool data structure directly. These test
//! the properties that only appear once `MetalDevice` is driving it: which
//! allocation paths participate in reuse, that reclamation needs nothing from
//! the caller, and that the free list does not grow without bound.

#![cfg(feature = "metal")]

use candle_core::{DType, Device, Result, Tensor};

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// The property the change exists to provide: a tensor's buffer is offered for
/// reuse the moment the tensor dies, with no sweep, no scan, and nothing asked
/// of the caller.
#[test]
fn dropping_a_tensor_returns_its_buffer() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Warm the pool so the sizes under test already exist.
    drop(Tensor::zeros((256, 256), DType::F32, &device)?);

    let releases = |m: &candle_core::MetalDevice| {
        let (s, p) = m.pool_counters();
        s.releases + p.releases
    };

    let before = releases(metal);
    let t = Tensor::zeros((256, 256), DType::F32, &device)?;
    let during = releases(metal);
    drop(t);
    let after = releases(metal);

    assert!(
        after > during,
        "dropping a tensor did not return its buffer: {during} -> {after}"
    );
    assert!(after > before);
    Ok(())
}

/// Repeating identical work must not grow the pool. This is the decode steady
/// state in miniature: the same shapes every iteration, so after the first pass
/// every allocation should be served from the free list.
#[test]
fn repeated_identical_work_stops_allocating() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let a = Tensor::ones((128, 128), DType::F32, &device)?;
    let b = Tensor::ones((128, 128), DType::F32, &device)?;

    // Several warm-up iterations, so every intermediate shape has been seen.
    for _ in 0..8 {
        let c = ((&a + &b)? * 2.0)?;
        let _ = c.sum_all()?;
    }

    metal.reset_pool_counters();
    for _ in 0..32 {
        let c = ((&a + &b)? * 2.0)?;
        let _ = c.sum_all()?;
    }
    let (shared, private) = metal.pool_counters();

    let lookups = shared.lookups + private.lookups;
    let allocations = shared.allocations + private.allocations;
    assert!(lookups > 0, "no pool lookups happened at all");
    assert_eq!(
        allocations, 0,
        "steady-state work allocated {allocations} new buffers over {lookups} lookups; \
         the pool is not reusing"
    );
    Ok(())
}

/// Scan length must not grow with how long the process has been running.
///
/// The old pool failed this: `retain` emptied a bucket's `Vec` but left its key
/// in the map, so keys accumulated at ~3.2 per decode token and every lookup
/// walked all of them -- 1345 buckets, 1324 of them empty, after 400 tokens.
/// The cost was O(work done), not O(live buffers), and it never converged.
#[test]
fn free_list_does_not_accumulate_empty_buckets() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Ever-changing shapes, the way a growing KV cache produces a new size
    // every few tokens.
    for n in 1..200usize {
        let t = Tensor::zeros((n, 64), DType::F32, &device)?;
        let _ = (&t + 1.0)?;
    }

    let (shared, private) = metal.pool_occupancy();
    for (name, occ) in [("shared", shared), ("private", private)] {
        assert!(
            occ.free_buckets <= occ.free_buffers,
            "{name} pool has {} free buckets holding only {} buffers, so some are empty",
            occ.free_buckets,
            occ.free_buffers
        );
    }
    Ok(())
}

/// Per-lookup scan length is bounded by a small constant regardless of how many
/// distinct sizes the pool holds -- the invariant `CONTRIBUTING.md` §3.3 and
/// `DESIGN.md` §15.2 #10 require of anything on the per-token path.
#[test]
fn lookup_cost_does_not_grow_with_pool_size() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    for n in 1..256usize {
        let _ = Tensor::zeros((n, 32), DType::F32, &device)?;
    }

    metal.reset_pool_counters();
    for n in 1..256usize {
        let _ = Tensor::zeros((n, 32), DType::F32, &device)?;
    }
    let (shared, private) = metal.pool_counters();

    let lookups = shared.lookups + private.lookups;
    let probed = shared.buckets_probed + private.buckets_probed;
    assert!(lookups > 0, "no lookups recorded");

    // Two probes per lookup is the structural maximum: the exact-size bucket,
    // then one ordered range query. The old scan walked every bucket.
    let per_lookup = probed as f64 / lookups as f64;
    assert!(
        per_lookup <= 2.0,
        "expected at most 2 bucket probes per lookup, got {per_lookup:.2} \
         ({probed} probes over {lookups} lookups)"
    );
    Ok(())
}

/// Buffers uploaded with data are not parked in the free list.
///
/// They come from `newBufferWithBytes` and are only reachable by an
/// `allocate_buffer` request of exactly their size, which nothing in a normal
/// forward pass issues. On an LFM2 load, pooling them held 5145 MB for a reuse
/// that never comes -- the old allocator reclaimed them only as a side effect
/// of its sweep, so with the sweep off the allocation path they must not be
/// retained (issue #8's 7731 -> 5509 MB depends on it).
#[test]
fn uploaded_buffers_are_not_retained_for_reuse() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let data: Vec<f32> = vec![1.0; 64 * 1024];
    metal.reset_pool_counters();
    let (shared_before, _) = metal.pool_occupancy();

    for _ in 0..16 {
        let t = Tensor::from_slice(&data, (64, 1024), &device)?;
        drop(t);
    }

    let (shared_after, _) = metal.pool_occupancy();
    assert_eq!(
        shared_after.free_bytes, shared_before.free_bytes,
        "uploaded buffers were retained in the free list: {} -> {} bytes free",
        shared_before.free_bytes, shared_after.free_bytes
    );
    Ok(())
}

/// Reclamation must not require anything of the caller. A buffer moved into a
/// struct, cloned, and handed around still comes back on its own -- which is
/// what keeps candle's allocate-and-forget ergonomics, and why this is not a
/// manual `free()` API.
#[test]
fn reclamation_requires_nothing_of_the_caller() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    drop(Tensor::zeros((64, 64), DType::F32, &device)?);

    struct Owned {
        _tensor: Tensor,
    }

    let t = Tensor::zeros((64, 64), DType::F32, &device)?;
    let shared_clone = t.clone();
    let owned = Owned { _tensor: t };

    let releases = |m: &candle_core::MetalDevice| {
        let (s, p) = m.pool_counters();
        s.releases + p.releases
    };

    let before = releases(metal);
    drop(shared_clone);
    let mid = releases(metal);
    assert_eq!(
        mid, before,
        "buffer returned while another reference was still live"
    );

    drop(owned);
    let after = releases(metal);
    assert!(
        after > mid,
        "buffer did not return when its last reference dropped"
    );
    Ok(())
}

/// `trim` hands memory back without disturbing live tensors.
#[test]
fn trim_frees_the_free_list_and_spares_live_tensors() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let live = Tensor::zeros((128, 128), DType::F32, &device)?;
    drop(Tensor::zeros((256, 256), DType::F32, &device)?);

    let free_buffers = |m: &candle_core::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.free_buffers + p.free_buffers
    };

    assert!(free_buffers(metal) > 0, "expected something free to trim");

    metal.trim_unused_buffers();

    assert_eq!(free_buffers(metal), 0, "trim left buffers in the free list");

    // The live tensor is untouched and still usable.
    let sum = (&live + 1.0)?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(sum, (128 * 128) as f32);
    Ok(())
}
