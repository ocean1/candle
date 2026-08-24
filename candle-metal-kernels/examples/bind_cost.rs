//! What does candle's per-bind bookkeeping actually cost, component by component?
//!
//! Measurement harness for lloom issue #24 / `DESIGN.md` §6.4, §6.7 L4b,
//! §16 P0 #1. Issue #7 established that a decode token spends 1.151 ms outside
//! the GPU (1.251 ms on `lloom/integration`) while raw dispatch encoding
//! accounts for only 0.068 ms of it, and attributed the rest to the per-bind
//! `prev_ce_outputs` probe and the allocator scan **by elimination**. Issue #21
//! removed the scan. This prices what is left.
//!
//! # Why this is a separate program from the in-decode counters
//!
//! `Instant::now()` costs ~18 ns on this machine and a start/stop pair ~43 ns,
//! measured. `wait_for_buffer`'s whole body is of that order, so timing it in
//! place would mostly measure the clock -- the failure `CONTRIBUTING.md` §3.2
//! and `DESIGN.md` §2.4 name ("check whether the measurement tool is the
//! cost"). The decode-path instrumentation therefore only *counts*, at one
//! relaxed atomic each, and this program supplies the per-operation price by
//! running each component in a loop long enough to amortize one timer pair over
//! millions of iterations.
//!
//! Multiplying the two gives the attribution. It is a model, and it is stated
//! as one: it assumes the components cost the same in a tight loop as they do
//! interleaved with Metal calls, which is optimistic about cache residency.
//! §"validation" below closes that gap by removing real work from the real
//! decode path and checking the predicted saving against the measured one.
//!
//! # What is priced
//!
//! The per-bind path in `ComputeCommandEncoder::set_input_buffer` is:
//!
//! ```text
//!   wait_for_buffer(ptr):
//!     1. lock prev_ce_outputs           <- mutex
//!     2. map.get(&ptr)                  <- hash lookup
//!     3. .cloned()                      <- Arc clone, on a hit only
//!     4. lock state                     <- second mutex, on a hit only
//!     5. waited_fences.insert(..)       <- HashSet probe, on a hit only
//!   then, unconditionally:
//!     6. lock state                     <- mutex again
//!     7. prev_outputs.contains(&ptr)    <- HashSet probe
//!     8. next_inputs.insert(ptr)        <- HashSet insert
//!     9. all_inputs.insert(ptr)         <- HashSet insert (never read; §6.7 L2)
//!    10. setBuffer_offset_atIndex       <- the actual Metal call
//! ```
//!
//! Steps 6-9 are not in issue #24's list, and they are per-bind too. Pricing
//! them separately is the difference between "the probe is 30 % of the gap" and
//! "per-bind bookkeeping is 30 % of the gap, of which the probe is half".
//!
//! ```bash
//! cargo run --release --features metal --example bind_cost
//! ```

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Iterations per timed loop. Large enough that one timer pair (~43 ns) is
/// under 0.01 % of the measured interval even for the cheapest component.
const ITERS: u64 = 2_000_000;
/// Repeats of each measurement, so a spread is reported rather than one number.
const REPEATS: usize = 5;

/// Entries in the simulated `prev_ce_outputs`. Measured live during LFM2 decode
/// and reported by `lfm2-decode-profile` as `mean_map_entries`; the default here
/// is set from that measurement, and `--map-entries` overrides it so the
/// sensitivity of a hash probe to table size can be checked rather than assumed.
const DEFAULT_MAP_ENTRIES: usize = 320;

fn main() {
    let map_entries = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--map-entries")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(DEFAULT_MAP_ENTRIES);

    println!("=== per-bind component cost (lloom issue #24) ===");
    println!("iterations per loop   {ITERS}");
    println!("repeats               {REPEATS}");
    println!("map entries           {map_entries}");
    println!();

    // A realistic key population: buffer pointers are 16-byte-aligned heap
    // addresses, not dense integers, so using dense ones would give the hasher
    // a friendlier distribution than it sees in practice.
    let keys: Vec<usize> = (0..map_entries).map(|i| 0x1_0000_0000 + i * 4096).collect();

    let mut results: Vec<(&str, Vec<f64>)> = Vec::new();

    // ---- 0. the timer itself, so every figure below can be read net of it ---
    results.push(("baseline: empty loop", bench(black_box)));

    // ---- 1. uncontended mutex lock/unlock --------------------------------
    {
        let m = Mutex::new(0u64);
        results.push((
            "mutex lock+unlock",
            bench(|i| {
                let mut g = m.lock().unwrap();
                *g = i;
                black_box(*g)
            }),
        ));
    }

    // ---- 2. hash lookup, miss and hit ------------------------------------
    {
        let mut map: HashMap<usize, Arc<u64>> = HashMap::new();
        for (n, k) in keys.iter().enumerate() {
            map.insert(*k, Arc::new(n as u64));
        }
        // `get(..).is_some()` rather than `contains_key`, deliberately, and
        // clippy's `unnecessary_get_then_check` is allowed for that reason: the
        // code being priced is `map.get(&ptr).cloned()`, which fetches the
        // value. `contains_key` can skip the value load, so it would measure a
        // cheaper operation than the one `wait_for_buffer` performs.
        //
        // Miss: a key not in the map, which is the common case if the issue's
        // "most binds hit buffers with no pending writer" hypothesis holds.
        #[allow(clippy::unnecessary_get_then_check)]
        results.push((
            "hashmap get (miss)",
            bench(|i| black_box(map.get(&(0xdead_0000 + i as usize)).is_some() as u64)),
        ));
        // Hit, without the clone, so the clone can be priced on its own.
        let kn = keys.len();
        #[allow(clippy::unnecessary_get_then_check)]
        results.push((
            "hashmap get (hit)",
            bench(|i| black_box(map.get(&keys[i as usize % kn]).is_some() as u64)),
        ));
        // Hit plus the `Arc` clone the current code performs unconditionally on
        // a hit. The clone is an atomic increment now and a decrement at drop.
        results.push((
            "hashmap get (hit) + Arc clone+drop",
            bench(|i| {
                let v = map.get(&keys[i as usize % kn]).cloned();
                black_box(v.is_some() as u64)
            }),
        ));
    }

    // ---- 3. the full probe body, miss and hit-deduped ---------------------
    // This is what `wait_for_buffer` costs end to end, excluding the Metal
    // `waitForFence` call itself (which only happens on a genuine first wait).
    {
        let map: Arc<Mutex<HashMap<usize, Arc<u64>>>> = Arc::new(Mutex::new(
            keys.iter()
                .enumerate()
                .map(|(n, k)| (*k, Arc::new(n as u64)))
                .collect(),
        ));
        let state: Arc<Mutex<HashSet<usize>>> = Arc::new(Mutex::new(
            // `waited_fences` after a few dozen distinct fences have been seen.
            (0..64).map(|i| 0x2_0000_0000 + i * 64).collect(),
        ));

        results.push((
            "wait_for_buffer (no pending writer)",
            bench(|i| {
                let fence = {
                    let m = map.lock().unwrap();
                    m.get(&(0xdead_0000 + i as usize)).cloned()
                };
                black_box(fence.is_some() as u64)
            }),
        ));

        let kn = keys.len();
        results.push((
            "wait_for_buffer (hit, already waited)",
            bench(|i| {
                let fence = {
                    let m = map.lock().unwrap();
                    m.get(&keys[i as usize % kn]).cloned()
                };
                let Some(f) = fence else { return black_box(0) };
                let mut s = state.lock().unwrap();
                let inserted = s.insert(Arc::as_ptr(&f) as usize);
                if inserted {
                    // Keep the set from growing without bound across ITERS
                    // iterations, which would turn this into a rehash benchmark.
                    s.remove(&(Arc::as_ptr(&f) as usize));
                }
                black_box(inserted as u64)
            }),
        ));
    }

    // ---- 4. the hazard bookkeeping after the probe (steps 6-9) ------------
    // Not in issue #24's list, but on the same per-bind path.
    {
        let state: Arc<Mutex<HazardState>> = Arc::new(Mutex::new(HazardState {
            prev_outputs: keys.iter().copied().collect(),
            next_inputs: HashSet::new(),
            all_inputs: keys.iter().copied().collect(),
        }));
        let kn = keys.len();
        results.push((
            "hazard bookkeeping (contains + 2 inserts)",
            bench(|i| {
                let ptr = keys[i as usize % kn];
                let mut s = state.lock().unwrap();
                let hit = s.prev_outputs.contains(&ptr);
                s.next_inputs.insert(ptr);
                s.all_inputs.insert(ptr);
                black_box(hit as u64)
            }),
        ));
    }

    // ---- report -----------------------------------------------------------
    println!(
        "{:<44} {:>10} {:>10} {:>10}",
        "component", "mean ns", "sd ns", "min ns"
    );
    let baseline = mean(&results[0].1);
    for (name, samples) in &results {
        let m = mean(samples);
        let sd = stddev(samples, m);
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        println!("{name:<44} {m:>10.2} {sd:>10.2} {min:>10.2}");
    }
    println!();
    println!("loop baseline is {baseline:.2} ns; subtract it to get net cost.");
    println!();

    println!("=== net of loop baseline ===");
    for (name, samples) in &results[1..] {
        println!("{:<44} {:>10.2} ns", name, mean(samples) - baseline);
    }
    println!();

    // A machine-readable line, so the report's arithmetic can be re-derived.
    print!("RESULT_BIND_COST map_entries={map_entries}");
    for (name, samples) in &results {
        let key: String = name
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        print!(" {}={:.3}", key, mean(samples) - baseline);
    }
    println!();
}

struct HazardState {
    prev_outputs: HashSet<usize>,
    next_inputs: HashSet<usize>,
    all_inputs: HashSet<usize>,
}

/// Run `f` `ITERS` times, `REPEATS` times over, returning ns/iteration each.
///
/// One timer pair per repeat, never per iteration: at ~43 ns a pair, timing
/// inside the loop would dominate every component measured here.
fn bench<F: FnMut(u64) -> u64>(mut f: F) -> Vec<f64> {
    // One untimed pass so the caches and any lazy allocation are warm; a cold
    // first repeat would show up as spread and be mistaken for variance.
    for i in 0..(ITERS / 10) {
        black_box(f(i));
    }
    (0..REPEATS)
        .map(|_| {
            let t0 = Instant::now();
            let mut acc = 0u64;
            for i in 0..ITERS {
                acc = acc.wrapping_add(f(i));
            }
            let el = t0.elapsed().as_secs_f64();
            black_box(acc);
            el * 1e9 / ITERS as f64
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stddev(xs: &[f64], m: f64) -> f64 {
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}
