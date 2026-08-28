//! Serializing one run record at exit (lloom issue #171).
//!
//! # Why this is written by hand rather than by pulling in `lloom-runs`
//!
//! The reader lives in the lloom repo (`tools/lloom-runs`) and this harness
//! lives in candle. A path dependency across the two would make candle
//! unbuildable without lloom checked out beside it, which
//! `CONTRIBUTING.md` §4.1's "keep changes independent and revertible" rules out
//! — and the coupling would run the wrong way, since candle is the thing being
//! upstreamed.
//!
//! What crosses the boundary is a **format**, not a crate: one JSON object per
//! line, whose field names are the contract. `lloom-runs` parses it; nothing
//! here links against it. The format is pinned from the reader's side by
//! `tools/lloom-runs/tests/` fixtures, so a drift is a test failure there rather
//! than a silent mismatch.
//!
//! # Where it runs
//!
//! [`RunRecord::emit`] is called next to the `RESULT` line — after the last
//! token, after `device.synchronize()`, after the means are computed. Nothing in
//! this file executes inside a timed window except [`now_ns`], which the loop
//! calls once per token to stamp the step it just finished.
//!
//! That one call is the whole in-window cost, and it is a `clock_gettime_nsec_np`
//! against a 24 MHz timebase — the same call `lloom-sample` makes to bracket
//! every read. Measured rather than asserted: see `--record-self-cost`.

use std::fmt::Write as _;

/// `CLOCK_UPTIME_RAW` in nanoseconds — `DESIGN.md` §3.4b's domain A.
///
/// The same clock `lloom-sample` brackets its IOReport reads in, so a token
/// stamped here and a telemetry sample land on one axis with no conversion and
/// no reconstruction. **Not `CLOCK_MONOTONIC_RAW`**, which on Darwin includes
/// time asleep where this does not — the opposite of Linux, and 39.38 hours of
/// it on this machine.
#[cfg(target_os = "macos")]
pub fn now_ns() -> u64 {
    extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
    }
    const CLOCK_UPTIME_RAW: u32 = 8;
    // SAFETY: the call takes a clock id and returns a u64; it touches no memory
    // we own and cannot fail for a compile-time-known id.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

#[cfg(not(target_os = "macos"))]
pub fn now_ns() -> u64 {
    0
}

/// The binary's own Mach-O `LC_UUID`.
///
/// #163 identified a panicking process by UUID when five sibling
/// `lfm2-decode-profile` builds existed on this machine and neither a name nor
/// a commit could have separated them (`DESIGN.md` §6.3c). Recorded per run so a
/// future panic log can be matched against the store rather than against a guess.
#[cfg(target_os = "macos")]
pub fn self_uuid() -> Option<String> {
    #[repr(C)]
    struct MachHeader64 {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }
    #[repr(C)]
    struct LoadCommand {
        cmd: u32,
        cmdsize: u32,
    }
    extern "C" {
        fn _dyld_get_image_header(image_index: u32) -> *const MachHeader64;
    }
    const LC_UUID: u32 = 0x1b;

    // SAFETY: image 0 is the main executable, mapped for the process's lifetime.
    // We walk exactly the `ncmds` load commands the header declares, reading
    // only within each command's own `cmdsize`.
    unsafe {
        let header = _dyld_get_image_header(0);
        if header.is_null() {
            return None;
        }
        let mut cmd = (header as *const u8).add(std::mem::size_of::<MachHeader64>());
        for _ in 0..(*header).ncmds {
            let lc = cmd as *const LoadCommand;
            if (*lc).cmd == LC_UUID {
                let b = *(cmd.add(8) as *const [u8; 16]);
                return Some(format!(
                    "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-\
                     {:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                    b[12], b[13], b[14], b[15]
                ));
            }
            cmd = cmd.add((*lc).cmdsize as usize);
        }
        None
    }
}

#[cfg(not(target_os = "macos"))]
pub fn self_uuid() -> Option<String> {
    None
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

fn opt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("\"{}\"", esc(s)),
        None => "null".into(),
    }
}

/// A measured quantity, or an honest statement that it was not measured.
///
/// `NeverRun` is a *value*, not `None`. The corpus already contains the failure
/// this prevents: `gpu_ms_per_token=0.0000` appears on every run taken with
/// profiling off, a real zero standing where "not measured" was meant. Three
/// registries arrived at this independently — `.bench/`'s `unmeasured`,
/// `lloom-probe`'s `UNCHECKED`, #99's "not measured ≠ measured-unchanged".
#[derive(Clone, Copy)]
pub enum Measured {
    Value(f64),
    NeverRun,
}

impl Measured {
    fn json(self) -> String {
        match self {
            Measured::Value(v) => format!("{{\"state\":\"value\",\"value\":{v}}}"),
            Measured::NeverRun => "{\"state\":\"never_run\"}".into(),
        }
    }
}

/// One decode step.
pub struct Step {
    pub index: usize,
    pub warmup: bool,
    pub wall_ms: f64,
    pub gpu_ms: Option<f64>,
    pub dispatches: Option<u64>,
    /// `CLOCK_UPTIME_RAW` at the end of this token.
    ///
    /// This field is why the memory timeline can be plotted against token index.
    /// `lloom-sample`'s plotter currently *reconstructs* the decode timeline by
    /// cumulative sum anchored at the child's exit, because the profiler prints
    /// no clock reading; its README names the fix as "a timestamp printed by the
    /// profiler itself — a candle change, deliberately out of scope for this
    /// issue". This is that change. `DESIGN.md` §3.4a-i finding 4 records an
    /// anchored reconstruction moving a figure **15 %**.
    pub t_end_ns: u64,
    pub kv_len: usize,
}

/// Everything the store needs about one run.
pub struct RunRecord {
    pub run_id: String,
    pub harness: &'static str,
    pub label: String,
    pub candle_commit: Option<String>,
    pub lloom_commit: Option<String>,
    pub binary_uuid: Option<String>,
    pub binary_path: Option<String>,
    /// `(axis, arm)` for every axis this build can resolve.
    ///
    /// **Completeness against `.bench/configurations.md` §1 is the reader's job,
    /// not this list's.** An axis absent here becomes `UNRECORDED` in the store —
    /// a value that never merges with a recorded arm — which is what makes a run
    /// taken today safe to group against one taken after a twelfth axis exists.
    /// The failure being designed out is `MathMode`: a real axis, in neither the
    /// registry nor the `config=[…]` line, invisible for the life of the project
    /// (`DESIGN.md` §2.3.9).
    pub axes: Vec<(String, String)>,
    pub machine: String,
    pub macos_build: String,
    pub load_start: Option<f64>,
    pub load_end: Option<f64>,
    pub under_lease: bool,
    pub n: usize,
    pub warmup: usize,
    pub seed: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub prompt_tokens: usize,
    pub kv_len_first: usize,
    pub kv_len_last: usize,
    pub dtype: String,
    pub profiling_compiled: bool,
    pub profiling_enabled: bool,
    pub wall_ms_per_token: Measured,
    pub gpu_ms_per_token: Measured,
    pub nongpu_ms_per_token: Measured,
    pub dispatches_per_token: Measured,
    pub prefill_ms: Measured,
    pub weight_bytes: Measured,
    pub steps: Vec<Step>,
    pub telemetry_path: Option<String>,
}

impl RunRecord {
    /// Serializes to one JSON line and appends it to `path`.
    ///
    /// Called **at exit**, beside the `RESULT` line. JSONL and append-only, so
    /// two agents writing one store interleave whole records, and a process that
    /// dies mid-run leaves every completed record readable — which matters here
    /// more than usual, since this project has lost two sessions to kernel
    /// panics (#163) and the run that crashes is the interesting one.
    pub fn emit(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;

        let mut s = String::with_capacity(4096 + self.steps.len() * 96);
        s.push_str("{\"schema_version\":1");
        let _ = write!(s, ",\"run_id\":\"{}\"", esc(&self.run_id));
        let _ = write!(s, ",\"harness\":\"{}\"", self.harness);
        let _ = write!(s, ",\"label\":\"{}\"", esc(&self.label));
        let _ = write!(
            s,
            ",\"started_unix_s\":{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        );

        let _ = write!(
            s,
            ",\"provenance\":{{\"candle_commit\":{},\"lloom_commit\":{},\
             \"binary\":{{\"uuid\":{},\"path\":{},\"mtime_epoch_s\":null}}}}",
            opt_str(self.candle_commit.as_deref()),
            opt_str(self.lloom_commit.as_deref()),
            opt_str(self.binary_uuid.as_deref()),
            opt_str(self.binary_path.as_deref()),
        );

        s.push_str(",\"axes\":{\"axes\":{");
        for (i, (k, v)) in self.axes.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "\"{}\":{{\"kind\":\"Arm\",\"arm\":\"{}\"}}", esc(k), esc(v));
        }
        s.push_str("}}");

        let f = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_else(|| "null".into());
        let _ = write!(
            s,
            ",\"conditions\":{{\"machine\":\"{}\",\"macos_build\":\"{}\",\
             \"load_average_start\":{},\"load_average_end\":{},\"gpu_holders\":[],\
             \"arbiter_idle\":null,\"under_arbiter_lease\":{}}}",
            esc(&self.machine),
            esc(&self.macos_build),
            f(self.load_start),
            f(self.load_end),
            self.under_lease,
        );

        let _ = write!(
            s,
            ",\"params\":{{\"n\":{},\"warmup\":{},\"seed\":{},\"temperature\":{},\
             \"top_p\":{},\"prompt_tokens\":{},\"kv_len_first\":{},\"kv_len_last\":{},\
             \"dtype\":\"{}\"}}",
            self.n,
            self.warmup,
            self.seed,
            self.temperature,
            self.top_p,
            self.prompt_tokens,
            self.kv_len_first,
            self.kv_len_last,
            esc(&self.dtype),
        );

        let _ = write!(
            s,
            ",\"profiling_compiled\":{},\"profiling_enabled\":{}",
            self.profiling_compiled, self.profiling_enabled
        );

        let _ = write!(
            s,
            ",\"metrics\":{{\"wall_ms_per_token\":{},\"gpu_ms_per_token\":{},\
             \"nongpu_ms_per_token\":{},\"dispatches_per_token\":{},\
             \"barriers_per_token\":{},\"prefill_ms\":{},\"weight_bytes\":{},\
             \"peak_phys_footprint_bytes\":{}}}",
            self.wall_ms_per_token.json(),
            self.gpu_ms_per_token.json(),
            self.nongpu_ms_per_token.json(),
            self.dispatches_per_token.json(),
            // This harness does not count barriers -- `lfm2-dispatch-trace`
            // does. Spelled NEVER RUN rather than defaulted to zero, which is
            // the whole reason `Measured` is not `Option<f64>`.
            Measured::NeverRun.json(),
            self.prefill_ms.json(),
            self.weight_bytes.json(),
            Measured::NeverRun.json(),
        );

        s.push_str(",\"pair\":null,\"steps\":[");
        for (i, st) in self.steps.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(
                s,
                "{{\"index\":{},\"warmup\":{},\"wall_ms\":{:.6},\"gpu_ms\":{},\
                 \"dispatches\":{},\"t_end_ns\":{},\"kv_len\":{}}}",
                st.index,
                st.warmup,
                st.wall_ms,
                st.gpu_ms.map(|v| format!("{v:.6}")).unwrap_or("null".into()),
                st.dispatches.map(|v| v.to_string()).unwrap_or("null".into()),
                st.t_end_ns,
                st.kv_len,
            );
        }
        s.push(']');

        let _ = write!(
            s,
            ",\"telemetry_path\":{},\"tokens_sha256\":null,\"logits_sha256\":null}}\n",
            opt_str(self.telemetry_path.as_deref())
        );

        if let Some(dir) = std::path::Path::new(path).parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(s.as_bytes())?;
        file.flush()
    }
}
