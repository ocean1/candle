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

/// Dropping a tensor hands its buffer back to the pool with no sweep, no scan,
/// and nothing asked of the caller.
///
/// "Hands back" is not "makes reusable": the buffer is parked until the GPU
/// finishes with it. What this pins is that the release still happens on drop,
/// automatically -- `trim_does_not_free_buffers_still_in_flight` and
/// `in_flight_buffer_is_not_handed_out_again` cover the second half.
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
/// state in miniature: the same shapes every iteration, so once the pool is
/// warm every allocation should be served from the free list.
///
/// The warm-up is longer than it looks like it needs to be, and deliberately.
/// Buffers are released on **GPU completion**, so the working set has to cover
/// not just one iteration's tensors but every iteration still in flight. Until
/// it does, a lookup can legitimately miss and allocate. What must be true --
/// and is asserted here -- is that this converges and then stops: measured over
/// 2000 iterations the allocation count freezes and the footprint is flat.
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

    // Long enough for the in-flight working set to be fully populated, not just
    // for every intermediate *shape* to have been seen once.
    for _ in 0..128 {
        let c = ((&a + &b)? * 2.0)?;
        let _ = c.sum_all()?;
    }

    metal.reset_pool_counters();
    for _ in 0..128 {
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

/// The pool must reach a bounded size and stay there.
///
/// Deferring release to GPU completion holds more buffers at once than
/// releasing on CPU drop did, which is the cost of the change. The cost has to
/// be a constant -- the depth of the in-flight window -- and not something that
/// grows with how long the process runs, which is the shape issue #21 removed
/// from the free-list scan and must not be reintroduced in the footprint.
#[test]
fn pool_footprint_converges_and_stays_flat() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let a = Tensor::ones((128, 128), DType::F32, &device)?;
    let b = Tensor::ones((128, 128), DType::F32, &device)?;

    let footprint = |m: &candle_core::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.total_bytes() + p.total_bytes()
    };

    // Warm up past the point where the in-flight working set is populated.
    for _ in 0..200 {
        let c = ((&a + &b)? * 2.0)?;
        let _ = c.sum_all()?;
    }
    let settled = footprint(metal);

    for _ in 0..1000 {
        let c = ((&a + &b)? * 2.0)?;
        let _ = c.sum_all()?;
    }
    let after = footprint(metal);

    assert_eq!(
        after, settled,
        "pool grew from {settled} to {after} bytes over 1000 further iterations \
         of identical work; the in-flight window is not bounded"
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
    // A dropped buffer is not free until the GPU is done with it, so a trim
    // has nothing to reclaim until the work it was part of has completed.
    metal.wait_until_completed()?;

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

/// The property that stops `trim` from becoming a use-after-free.
///
/// A buffer the CPU has released but the GPU may still be reading is *not* in
/// the free list, so trim leaves it alone. Destroying it would be the same
/// defect the deferral exists to prevent, with a freed allocation in place of
/// an aliased one.
#[test]
fn trim_does_not_free_buffers_still_in_flight() -> Result<()> {
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Queue work and drop its result without synchronizing, so the buffers are
    // released by the CPU while the GPU may still be using them.
    for _ in 0..16 {
        let t = Tensor::zeros((256, 256), DType::F32, &device)?;
        drop((&t + 1.0)?);
    }

    let pending = |m: &candle_core::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.pending_buffers + p.pending_buffers
    };

    let before = pending(metal);
    assert!(before > 0, "expected buffers awaiting the GPU");

    metal.trim_unused_buffers();

    assert_eq!(
        pending(metal),
        before,
        "trim destroyed {} buffer(s) the GPU may still be using",
        before - pending(metal)
    );
    Ok(())
}
