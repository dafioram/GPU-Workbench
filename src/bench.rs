//! Benchmarks and stress tests. Everything runs on a worker thread and reports
//! back through a channel so the UI (or CLI) stays responsive.

use crate::gpu::{self, Gpu, fmt_bytes};
use crate::sensors::clock;
use crate::shaders;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;
/// Target wall time for one batched submission. Kept well under the 2 s
/// Windows TDR watchdog so a slow GPU never gets reset mid-test.
const BATCH_TARGET: f64 = 0.15;
/// Timed work allowed per test, in seconds, after warm-up. A fast GPU never gets
/// near it: its batches are short, and its fixed batch counts run out first. A
/// slow device, such as the Windows software renderer, stops after a batch or
/// two instead of running for minutes.
const MEASURE_BUDGET: f64 = 20.0;

/// True once a test has at least one timed sample and has spent its budget.
fn budget_reached(spent: f64, samples: usize) -> bool {
    samples >= 1 && spent >= MEASURE_BUDGET
}

/// Why the FP16 matmul is skipped. On DX12, wgpu advertises FP16 only with
/// shader model 6.2 or newer, which FXC cannot compile for (it stops at 5.1).
fn f16_skip_detail(dx12: bool, dxc: bool) -> String {
    match (dx12, dxc) {
        (true, false) => "adapter does not support SHADER_F16. This DX12 run uses FXC, which \
             cannot compile FP16 shaders. Put dxcompiler.dll next to the program to \
             enable it (see the README)."
            .into(),
        (true, true) => "adapter does not support SHADER_F16, even with DXC loaded: the \
             driver reports neither shader model 6.2 nor native 16-bit shader support."
            .into(),
        (false, _) => "adapter does not support SHADER_F16".into(),
    }
}

/// How far the stress throughput fell from start to end, in words. A speed-up
/// reads as a gain, not as a negative drop.
fn trend_text(fall_pct: f64) -> String {
    if fall_pct >= 0.0 {
        format!("{fall_pct:.1}% drop")
    } else {
        format!("{:.1}% gain", -fall_pct)
    }
}

/// Allowed values for each setting. The GUI sliders and the CLI both use these,
/// so the two can't drift apart.
pub mod limits {
    use std::ops::RangeInclusive;

    /// Transfer buffer size in MiB.
    pub const TRANSFER_MB: RangeInclusive<u64> = 16..=4096;
    /// Matrix size, which must also be a multiple of 64. 256 is the smallest
    /// size the watchdog planner shrinks to, so anything below it is refused.
    pub const MATMUL_N: RangeInclusive<u32> = 256..=8192;
    /// VRAM to fill and verify, in MiB.
    pub const VRAM_MB: RangeInclusive<u64> = 64..=262144;
    /// Patterns per VRAM run.
    pub const VRAM_PASSES: RangeInclusive<u32> = 1..=6;
    /// Sustained stress duration in seconds.
    pub const STRESS_SECS: RangeInclusive<f64> = 5.0..=3600.0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Test {
    Upload,
    Readback,
    DeviceCopy,
    ComputeBandwidth,
    MatmulF32,
    MatmulF16,
    Vram,
    Stress,
}

impl Test {
    pub const ALL: [Test; 8] = [
        Test::Upload,
        Test::Readback,
        Test::DeviceCopy,
        Test::ComputeBandwidth,
        Test::MatmulF32,
        Test::MatmulF16,
        Test::Vram,
        Test::Stress,
    ];
    /// The quick suite skips the long-running VRAM and stress tests.
    pub const QUICK: [Test; 6] = [
        Test::Upload,
        Test::Readback,
        Test::DeviceCopy,
        Test::ComputeBandwidth,
        Test::MatmulF32,
        Test::MatmulF16,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Test::Upload => "Upload (host → GPU)",
            Test::Readback => "Readback (GPU → host)",
            Test::DeviceCopy => "Copy (GPU → GPU)",
            Test::ComputeBandwidth => "Compute bandwidth",
            Test::MatmulF32 => "Matmul FP32",
            Test::MatmulF16 => "Matmul FP16",
            Test::Vram => "VRAM fill + verify",
            Test::Stress => "Sustained stress",
        }
    }

    /// One-paragraph explanation, shown as a tooltip and in the CLI help.
    pub fn help(self) -> &'static str {
        match self {
            Test::Upload => {
                "Copies a buffer from system RAM to the GPU (queue.write_buffer) and reports GB/s. \
                Measures the PCIe link plus driver staging overhead. Size: 'Transfer size'."
            }
            Test::Readback => {
                "Copies a buffer from the GPU back to system RAM (copy to a mappable buffer, map, memcpy). \
                Usually slower than upload. Size: 'Transfer size'."
            }
            Test::DeviceCopy => {
                "Copies one GPU buffer to another with the driver's copy command; read + write bytes \
                are both counted. Some drivers use a slower copy engine here than shaders achieve."
            }
            Test::ComputeBandwidth => {
                "A shader streams one buffer into another, 16 bytes per thread. The best estimate of \
                usable VRAM bandwidth; typically 60-90% of the card's spec."
            }
            Test::MatmulF32 => {
                "Multiplies two N x N FP32 matrices with a tiled shader and reports GFLOPS. 256 results are \
                checked exactly against the CPU. Size: 'Matmul N'."
            }
            Test::MatmulF16 => {
                "Same as Matmul FP32 in half precision. Needs SHADER_F16 support, which older GPUs and \
                OpenGL lack (skipped there)."
            }
            Test::Vram => {
                "Fills video memory with a pattern, then checks every word, repeated with different patterns. \
                Catches bad VRAM and flags memory that the driver quietly placed in system RAM."
            }
            Test::Stress => {
                "Runs matmul back to back for 'Stress duration' and plots throughput. A falling line means \
                thermal or power throttling; the result is re-checked at the end to catch errors under heat."
            }
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Test::Upload => "upload",
            Test::Readback => "readback",
            Test::DeviceCopy => "copy",
            Test::ComputeBandwidth => "bandwidth",
            Test::MatmulF32 => "matmul32",
            Test::MatmulF16 => "matmul16",
            Test::Vram => "vram",
            Test::Stress => "stress",
        }
    }

    pub fn from_key(k: &str) -> Option<Test> {
        Test::ALL.into_iter().find(|t| t.key() == k)
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub transfer_mb: u64,
    pub matmul_n: u32,
    /// Size the VRAM test from a fresh free-memory reading taken when it starts.
    pub vram_auto: bool,
    /// Used when `vram_auto` is false.
    pub vram_mb: u64,
    pub vram_passes: u32,
    pub stress_secs: f64,
    pub stress_n: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            transfer_mb: 256,
            matmul_n: 2048,
            vram_auto: true,
            vram_mb: 1024,
            vram_passes: 3,
            stress_secs: 60.0,
            stress_n: 2048,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Status {
    Pass,
    /// Completed without errors, but the result needs a caveat (see detail).
    Warn,
    Fail,
    Skipped,
    /// The user pressed Stop before the test finished. Not a failure.
    Stopped,
}

impl Status {
    pub fn label(&self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
            Status::Skipped => "SKIP",
            Status::Stopped => "STOP",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestResult {
    pub test: Test,
    pub value: f64,
    pub unit: &'static str,
    pub detail: String,
    pub status: Status,
}

pub enum Msg {
    Log(String),
    Started(Test),
    Progress(f32),
    Result(TestResult),
    Stress {
        t: f64,
        gflops: f64,
    },
    /// The run could not start (e.g. device creation failed).
    Error(String),
    Done,
}

/// Starts a run on a background thread. Set `stop` to cancel between steps.
pub fn spawn(
    adapter: usize,
    tests: Vec<Test>,
    cfg: Config,
    tx: Sender<Msg>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // A driver-level panic inside wgpu must not leave the UI stuck in
        // "running": report it and still send Done.
        let tx2 = tx.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let log_tx = tx.clone();
            let log = move |s: String| {
                let _ = log_tx.send(Msg::Log(s));
            };
            match gpu::open(adapter, log) {
                Ok(g) => {
                    let _ = tx.send(Msg::Log(format!(
                        "Opened {} via {}",
                        g.info.name,
                        gpu::backend_str(g.info.backend)
                    )));
                    let b = Bench {
                        g,
                        tx: tx.clone(),
                        stop,
                        cut: AtomicBool::new(false),
                        matmul_kernels: RefCell::default(),
                    };
                    for t in tests {
                        if b.stopped() {
                            break;
                        }
                        if b.g.lost.load(Ordering::Relaxed) {
                            let _ = tx.send(Msg::Error(
                                "The GPU was reset by the driver (device lost), so the remaining tests were skipped. \
                                 On Windows this usually means a dispatch hit the 2 s watchdog; \
                                 try a smaller Matmul N or another backend."
                                    .into(),
                            ));
                            break;
                        }
                        let _ = tx.send(Msg::Started(t));
                        b.cut.store(false, Ordering::Relaxed);
                        let outcome = b.run(t, &cfg);
                        let r = finish(t, outcome, b.cut.load(Ordering::Relaxed));
                        let _ = tx.send(Msg::Result(r));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(e));
                }
            }
        }));
        if let Err(p) = outcome {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            let _ = tx2.send(Msg::Error(format!("GPU driver/wgpu panic: {msg}")));
        }
        let _ = tx2.send(Msg::Done);
    })
}

/// Decides what to report for one test. `cut` means a loop stopped early
/// because the user pressed Stop.
///
/// - A cut test is STOP, unless its own check failed (that stays FAIL).
/// - A cut test with no finished measurement (an error, or a non-finite value
///   from zero samples) reports no number.
/// - Anything not cut is reported as it came out, so real errors stay FAIL.
pub fn finish(test: Test, outcome: Result<TestResult, String>, cut: bool) -> TestResult {
    if !cut {
        return match outcome {
            Ok(r) => r,
            Err(e) => TestResult {
                test,
                value: 0.0,
                unit: "",
                detail: e,
                status: Status::Fail,
            },
        };
    }
    match outcome {
        // The test's own check failed: that is still a failure.
        Ok(r) if matches!(r.status, Status::Fail) => r,
        // Part of the work finished: report it, marked as stopped early.
        Ok(r) if r.value.is_finite() => TestResult {
            status: Status::Stopped,
            detail: format!("stopped early: {}", r.detail),
            ..r
        },
        // Stopped before any measurement finished: no number to show.
        _ => TestResult {
            test,
            value: 0.0,
            unit: "",
            detail: "stopped before it finished".into(),
            status: Status::Stopped,
        },
    }
}

struct Bench {
    g: Gpu,
    tx: Sender<Msg>,
    stop: Arc<AtomicBool>,
    /// Set when a loop stops early because of Stop, so the result can say so.
    cut: AtomicBool,
    /// The matmul shader, compiled once per run for FP32 (index 0) and FP16
    /// (index 1), so the Stress test doesn't compile it again.
    matmul_kernels: RefCell<[Option<Rc<Kernel>>; 2]>,
}

#[derive(Clone, Copy)]
enum Bind {
    ReadOnly,
    Storage,
    Uniform,
}

/// Matmul buffers plus the host inputs kept for the CPU cross-check.
struct Matmul {
    kernel: Rc<Kernel>,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    c: wgpu::Buffer,
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    n: u32,
    /// Tile rows per dispatch.
    band: u32,
    /// Dispatches per queue submission.
    per_submit: usize,
    /// Explains any automatic size reduction.
    note: String,
}

/// A, B, C on the GPU plus the host copies of A and B.
type MatmulBuffers = (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, Vec<f32>, Vec<f32>);

/// Longest a single matmul dispatch / submission should run, estimated from a
/// small probe. Far under the 2 s Windows watchdog, because the estimate is
/// rough and old GPUs can't preempt a running dispatch.
const MATMUL_DISPATCH_MAX: f64 = 0.03;
const MATMUL_SUBMIT_MAX: f64 = 0.12;
const MATMUL_BAND_LIMIT: f64 = 0.2;

/// How many tile rows go in one dispatch and how many dispatches go in one
/// submit, given the measured time per tile row. Keeps each dispatch under
/// MATMUL_DISPATCH_MAX and each submit under MATMUL_SUBMIT_MAX.
fn plan_band(row_secs: f64, tiles: u32) -> (u32, usize) {
    let row = row_secs.max(1e-9);
    let band = ((MATMUL_DISPATCH_MAX / row) as u32).clamp(1, tiles.max(1));
    let per_submit = ((MATMUL_SUBMIT_MAX / (row * band as f64)) as usize).max(1);
    (band, per_submit)
}

/// Band size and dispatches per submit. On a software adapter (the WARP device)
/// one dispatch covers all tile rows: with one dispatch per band, WARP wrote
/// only the last band, while a single dispatch over all rows came out exact.
/// Real GPUs keep the watchdog-sized bands.
fn matmul_plan(row_secs: f64, tiles: u32, software: bool) -> (u32, usize) {
    if software {
        (tiles, 1)
    } else {
        plan_band(row_secs, tiles)
    }
}

/// Throughput of the fastest pass, in GB/s: `bytes` moved in the shortest time.
fn best_pass_gbs(bytes: u64, pass_secs: &[f64]) -> f64 {
    let fastest = pass_secs.iter().cloned().fold(f64::INFINITY, f64::min);
    bytes as f64 / fastest.max(1e-9) / 1e9
}

/// One character per 64x64 tile of C, from one sampled entry per tile: '.' if
/// it is exactly zero (never written), 'o' if it matches the CPU, 'x' if not.
/// Shows whether whole tile rows or columns were skipped, or all are wrong.
fn tile_map(gpu: &[f32], ha: &[f32], hb: &[f32], n: usize, f16: bool) -> Vec<String> {
    let tiles = n / 64;
    (0..tiles)
        .map(|ti| {
            (0..tiles)
                .map(|tj| {
                    let (r, c) = (ti * 64 + 17, tj * 64 + 9);
                    let got = gpu[r * n + c] as f64;
                    if got == 0.0 {
                        return '.';
                    }
                    let want: f64 = (0..n)
                        .map(|k| ha[r * n + k] as f64 * hb[k * n + c] as f64)
                        .sum();
                    let tol = if f16 { 0.5 + want.abs() * 2e-3 } else { 1e-3 };
                    if (got - want).abs() <= tol { 'o' } else { 'x' }
                })
                .collect()
        })
        .collect()
}

/// Says what a wrong C entry holds. `terms` are the products a[r,k] * b[k,c]
/// and `gpu` is the value the shader wrote. If it equals the sum with one K
/// block of the matmul kernel (TK = 16) left out or counted twice, or with a
/// prefix of the K range left out, the shader skipped or repeated that work.
/// Otherwise the value matches no such partial sum.
fn explain_bad(gpu: f64, terms: &[f64]) -> String {
    const TOL: f64 = 1e-3;
    const TK: usize = 16;
    if gpu.abs() <= TOL {
        return "is zero (never written)".into();
    }
    let total: f64 = terms.iter().sum();
    let mut hits = Vec::new();
    let mut prefix = 0.0;
    for (i, block) in terms.chunks(TK).enumerate() {
        let k0 = i * TK;
        let k1 = k0 + block.len();
        let sum: f64 = block.iter().sum();
        if k0 > 0 && (prefix - gpu).abs() <= TOL {
            hits.push(format!("only k < {k0} summed"));
        }
        if k0 > 0 && (total - prefix - gpu).abs() <= TOL {
            hits.push(format!("k < {k0} missing"));
        }
        if (total - sum - gpu).abs() <= TOL {
            hits.push(format!("k {k0}..{k1} missing"));
        }
        if (total + sum - gpu).abs() <= TOL {
            hits.push(format!("k {k0}..{k1} counted twice"));
        }
        prefix += sum;
    }
    if hits.is_empty() {
        "matches no simple partial sum: the values themselves are wrong".into()
    } else {
        hits.truncate(3);
        hits.join("; ")
    }
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Times a closure that submits work and waits; returns seconds.
fn timed(f: impl FnOnce()) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64()
}

/// Formats with ~3 significant digits so both 2.1 and 41 000 read well.
pub fn num(v: f64) -> String {
    let a = v.abs();
    if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

fn summarize(times: &[f64]) -> (f64, f64) {
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = times.iter().sum::<f64>() / times.len().max(1) as f64;
    (best, avg)
}

/// Splits a 1-D element count into a 2-D grid of 256-wide workgroups.
fn grid(elements: u64) -> (u32, u32) {
    let groups = elements.div_ceil(shaders::WG as u64).max(1);
    let x = groups.min(65535);
    let y = groups.div_ceil(x);
    (x as u32, y as u32)
}

impl Bench {
    /// True once Stop is pressed. Seeing it also marks the running test as cut
    /// short, so its result is reported as STOP.
    fn stopped(&self) -> bool {
        let s = self.stop.load(Ordering::Relaxed);
        if s {
            self.cut.store(true, Ordering::Relaxed);
        }
        s
    }

    fn log(&self, s: impl Into<String>) {
        let _ = self.tx.send(Msg::Log(s.into()));
    }

    fn progress(&self, f: f32) {
        let _ = self.tx.send(Msg::Progress(f.clamp(0.0, 1.0)));
    }

    fn wait(&self) {
        let _ = self.g.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn run(&self, t: Test, cfg: &Config) -> Result<TestResult, String> {
        match t {
            Test::Upload => self.upload(cfg),
            Test::Readback => self.readback(cfg),
            Test::DeviceCopy => self.device_copy(cfg),
            Test::ComputeBandwidth => self.compute_bandwidth(cfg),
            Test::MatmulF32 => self.matmul(cfg.matmul_n, false),
            Test::MatmulF16 => self.matmul(cfg.matmul_n, true),
            Test::Vram => self.vram(cfg),
            Test::Stress => self.stress(cfg),
        }
    }

    // ---------- helpers ----------

    /// Creates a buffer, returning an error instead of logging one if the
    /// driver reports out-of-memory or the size is invalid.
    fn try_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Result<wgpu::Buffer, String> {
        self.create_checked(size, usage, label, false)
    }

    /// A buffer that starts out holding `data`. Written at creation instead of
    /// with queue.write_buffer: on the WARP device, the per-band uniforms of the
    /// matmul all ended up holding the last band's offset.
    fn try_buffer_with(
        &self,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Result<wgpu::Buffer, String> {
        let buf = self.create_checked(data.len() as u64, usage, label, true)?;
        buf.slice(..)
            .get_mapped_range_mut()
            .map_err(|e| format!("{e}"))?
            .copy_from_slice(data);
        buf.unmap();
        Ok(buf)
    }

    /// Creates a buffer; an out-of-memory or validation failure becomes an error.
    fn create_checked(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
        mapped: bool,
    ) -> Result<wgpu::Buffer, String> {
        let oom = self
            .g
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = self
            .g
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let buf = self.g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: mapped,
        });
        let v = pollster::block_on(val.pop());
        let o = pollster::block_on(oom.pop());
        if let Some(e) = o.or(v) {
            return Err(format!("allocating {} failed: {e}", fmt_bytes(size)));
        }
        Ok(buf)
    }

    fn kernel(&self, src: &str, entry: &str, binds: &[Bind]) -> Result<Kernel, String> {
        let scope = self
            .g
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .g
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let entries: Vec<_> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match b {
                        Bind::ReadOnly => wgpu::BufferBindingType::Storage { read_only: true },
                        Bind::Storage => wgpu::BufferBindingType::Storage { read_only: false },
                        Bind::Uniform => wgpu::BufferBindingType::Uniform,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = self
            .g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
        let pl = self
            .g
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .g
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("shader '{entry}' failed: {e}"));
        }
        Ok(Kernel { pipeline, layout })
    }

    fn bind(&self, k: &Kernel, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &k.layout,
            entries: &entries,
        })
    }

    fn dispatch(&self, k: &Kernel, bg: &wgpu::BindGroup, groups: (u32, u32), reps: u32) {
        let mut enc = self.g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, bg, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(groups.0, groups.1, 1);
            }
        }
        self.g.queue.submit([enc.finish()]);
    }

    /// Copies `size` bytes of `src` into host memory.
    fn read_back(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
        let staging = self.try_buffer(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "readback",
        )?;
        let mut enc = self.g.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.g.queue.submit([enc.finish()]);
        self.map_read(&staging)
    }

    fn map_read(&self, staging: &wgpu::Buffer) -> Result<Vec<u8>, String> {
        let slice = staging.slice(..);
        let (tx, rx) = channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|_| "map timed out".to_string())?
            .map_err(|e| format!("map failed: {e}"))?;
        let out = slice
            .get_mapped_range()
            .map_err(|e| format!("{e}"))?
            .to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Size for the transfer tests, clamped to what one buffer may hold.
    fn transfer_size(&self, cfg: &Config, storage: bool) -> u64 {
        let mut max = self.g.limits.max_buffer_size;
        if storage {
            max = max.min(self.g.limits.max_storage_buffer_binding_size);
        }
        (cfg.transfer_mb * MIB).min(max) / 256 * 256
    }

    /// Runs `f(reps)` once to calibrate, then picks reps so a batch takes
    /// about BATCH_TARGET seconds. Returns (reps, per-batch times).
    fn calibrated(&self, iters: usize, mut f: impl FnMut(u32)) -> (u32, Vec<f64>) {
        // Warm-up: pipeline creation, lazy allocation. If it already took a whole
        // batch (a slow device), it doubles as the calibration instead of a
        // second slow call.
        let warm = timed(|| f(1));
        let t1 = if warm >= BATCH_TARGET {
            warm
        } else {
            timed(|| f(1)).max(1e-6)
        };
        let reps = ((BATCH_TARGET / t1.max(1e-6)) as u32).clamp(1, 500);
        let mut times = Vec::new();
        let mut spent = 0.0;
        for i in 0..iters {
            if self.stopped() {
                break;
            }
            let t = timed(|| f(reps));
            times.push(t);
            spent += t;
            self.progress((i + 1) as f32 / iters as f32);
            if budget_reached(spent, times.len()) {
                self.budget_note(times.len(), iters);
                break;
            }
        }
        (reps, times)
    }

    fn budget_note(&self, taken: usize, planned: usize) {
        self.log(format!(
            "time budget of {MEASURE_BUDGET:.1} s reached: {taken} of {planned} timed batches run"
        ));
    }

    // ---------- tests ----------

    fn upload(&self, cfg: &Config) -> Result<TestResult, String> {
        let size = self.transfer_size(cfg, false);
        let buf = self.try_buffer(
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            "upload",
        )?;
        let data = vec![0xA5u8; size as usize];
        let (reps, times) = self.calibrated(8, |reps| {
            for _ in 0..reps {
                self.g.queue.write_buffer(&buf, 0, &data);
            }
            self.g.queue.submit([]);
            self.wait();
        });
        let (best, avg) = summarize(&times);
        let gbs = |t: f64| (size * reps as u64) as f64 / t / 1e9;
        Ok(TestResult {
            test: Test::Upload,
            value: gbs(avg),
            unit: "GB/s",
            detail: format!(
                "{} per copy, best {} GB/s (queue.write_buffer, includes staging memcpy)",
                fmt_bytes(size),
                num(gbs(best))
            ),
            status: Status::Pass,
        })
    }

    fn readback(&self, cfg: &Config) -> Result<TestResult, String> {
        let size = self.transfer_size(cfg, false);
        let src = self.try_buffer(
            size,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "src",
        )?;
        let staging = self.try_buffer(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "staging",
        )?;
        let mut host = vec![0u8; size as usize];
        let mut times = Vec::new();
        let mut spent = 0.0;
        for i in 0..9 {
            if self.stopped() {
                break;
            }
            let t = timed(|| {
                let mut enc = self.g.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&src, 0, &staging, 0, size);
                self.g.queue.submit([enc.finish()]);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                self.wait();
                if let Ok(view) = slice.get_mapped_range() {
                    host.copy_from_slice(&view);
                }
                staging.unmap();
            });
            if i > 0 {
                times.push(t); // first iteration is warm-up
                spent += t;
                if budget_reached(spent, times.len()) {
                    self.budget_note(times.len(), 8);
                    break;
                }
            }
            self.progress(i as f32 / 8.0);
        }
        let (best, avg) = summarize(&times);
        let gbs = |t: f64| size as f64 / t / 1e9;
        Ok(TestResult {
            test: Test::Readback,
            value: gbs(avg),
            unit: "GB/s",
            detail: format!(
                "{} per copy, best {} GB/s (copy + map + memcpy)",
                fmt_bytes(size),
                num(gbs(best))
            ),
            status: Status::Pass,
        })
    }

    fn device_copy(&self, cfg: &Config) -> Result<TestResult, String> {
        let size = self.transfer_size(cfg, false);
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let a = self.try_buffer(size, usage, "a")?;
        let b = self.try_buffer(size, usage, "b")?;
        let (reps, times) = self.calibrated(8, |reps| {
            let mut enc = self.g.device.create_command_encoder(&Default::default());
            for _ in 0..reps {
                enc.copy_buffer_to_buffer(&a, 0, &b, 0, size);
            }
            self.g.queue.submit([enc.finish()]);
            self.wait();
        });
        let (best, avg) = summarize(&times);
        let gbs = |t: f64| 2.0 * (size * reps as u64) as f64 / t / 1e9;
        Ok(TestResult {
            test: Test::DeviceCopy,
            value: gbs(avg),
            unit: "GB/s",
            detail: format!(
                "{} buffers, best {} GB/s (read + write counted)",
                fmt_bytes(size),
                num(gbs(best))
            ),
            status: Status::Pass,
        })
    }

    fn compute_bandwidth(&self, cfg: &Config) -> Result<TestResult, String> {
        let size = self.transfer_size(cfg, true) / 16 * 16;
        let k = self.kernel(shaders::COPY, "main", &[Bind::ReadOnly, Bind::Storage])?;
        let usage = wgpu::BufferUsages::STORAGE;
        let a = self.try_buffer(size, usage, "src")?;
        let b = self.try_buffer(size, usage, "dst")?;
        let bg = self.bind(&k, &[&a, &b]);
        let groups = grid(size / 16);
        let (reps, times) = self.calibrated(8, |reps| {
            self.dispatch(&k, &bg, groups, reps);
            self.wait();
        });
        let (best, avg) = summarize(&times);
        let gbs = |t: f64| 2.0 * (size * reps as u64) as f64 / t / 1e9;
        Ok(TestResult {
            test: Test::ComputeBandwidth,
            value: gbs(avg),
            unit: "GB/s",
            detail: format!(
                "vec4 copy kernel over {}, best {} GB/s",
                fmt_bytes(size),
                num(gbs(best))
            ),
            status: Status::Pass,
        })
    }

    /// Matrices are filled with multiples of 1/4 in [-1, 1], so every product
    /// and partial sum is exactly representable — GPU and CPU results must
    /// agree exactly in FP32 (and almost always in FP16).
    fn matmul_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut x = seed | 1;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x % 9) as f32 - 4.0) * 0.25
        };
        let a = (0..n * n).map(|_| next()).collect();
        let b = (0..n * n).map(|_| next()).collect();
        (a, b)
    }

    /// Allocates A, B, C (and fills A, B) for an N x N matmul.
    fn matmul_buffers(&self, n: u32, f16: bool) -> Result<MatmulBuffers, String> {
        let elem = if f16 { 2u64 } else { 4 };
        let (ha, hb) = Self::matmul_inputs(n as usize, 0x1234_5678 + n as u64);
        let bytes = (n as u64) * (n as u64) * elem;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let a = self.try_buffer(bytes, usage, "A")?;
        let b = self.try_buffer(bytes, usage, "B")?;
        let c = self.try_buffer(bytes, usage, "C")?;
        if f16 {
            let conv = |v: &[f32]| {
                v.iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect::<Vec<_>>()
            };
            self.g
                .queue
                .write_buffer(&a, 0, bytemuck::cast_slice(&conv(&ha)));
            self.g
                .queue
                .write_buffer(&b, 0, bytemuck::cast_slice(&conv(&hb)));
        } else {
            self.g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&ha));
            self.g.queue.write_buffer(&b, 0, bytemuck::cast_slice(&hb));
        }
        Ok((a, b, c, ha, hb))
    }

    fn dims_bind(
        &self,
        k: &Kernel,
        bufs: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        n: u32,
        row_off: u32,
    ) -> Result<wgpu::BindGroup, String> {
        let dims = self.try_buffer_with(
            bytemuck::cast_slice(&[n, row_off, 0, 0]),
            wgpu::BufferUsages::UNIFORM,
            "dims",
        )?;
        Ok(self.bind(k, &[bufs.0, bufs.1, bufs.2, &dims]))
    }

    /// Times a tiny 256x256 matmul to estimate throughput, then picks N, the
    /// band height and how many dispatches go in one submission so nothing
    /// runs long enough to trigger a driver reset. The probe underestimates
    /// speed (small sizes don't fill the GPU), which errs on the safe side.
    fn matmul_setup(&self, n: u32, f16: bool) -> Result<Matmul, String> {
        let setup_t0 = Instant::now();
        let elem = if f16 { 2u64 } else { 4 };
        let max_n = ((self.g.limits.max_storage_buffer_binding_size / elem) as f64).sqrt() as u32;
        let mut n = (n.min(max_n) / 64 * 64).max(64);
        let k = self.matmul_kernel(f16)?;

        // --- probe
        const PN: u32 = 256;
        let (pa, pb, pc, _, _) = self.matmul_buffers(PN, f16)?;
        let probe = self.dims_bind(&k, (&pa, &pb, &pc), PN, 0)?;
        let run_probe = || {
            self.dispatch(&k, &probe, (PN / 64, PN / 64), 1);
            self.wait();
        };
        run_probe();
        let t = (0..3)
            .map(|_| timed(run_probe))
            .fold(f64::INFINITY, f64::min);
        let rate = 2.0 * (PN as f64).powi(3) / t.max(1e-6); // FLOP/s
        drop((pa, pb, pc, probe));

        // --- plan
        let band_time = |n: u32| 2.0 * (n as f64) * (n as f64) * 64.0 / rate;
        let requested = n;
        // The shrink guards against GPU watchdog resets. A software adapter has
        // no such watchdog and keeps the size asked for, so a slow probe on a
        // CPU device cannot turn a 2048 run into a 1024 one.
        let software = self.g.info.device_type == wgpu::DeviceType::Cpu;
        while !software && band_time(n) > MATMUL_BAND_LIMIT && n > PN {
            n = (n / 2 / 64 * 64).max(PN);
        }
        let mut note = String::new();
        if n != requested {
            note = format!(" (auto-reduced from N={requested} to stay under the GPU watchdog)");
            self.log(format!(
                "Matmul: probe ran at {} GFLOPS, so N={requested} could stall the GPU too long; using N={n}",
                num(rate / 1e9)
            ));
        }
        let tiles = n / 64;
        let (band, per_submit) = plan_band(band_time(n), tiles);

        // --- real buffers
        let (a, b, c, ha, hb) = self.matmul_buffers(n, f16)?;
        self.log(format!(
            "Matmul N={n}: probe {} GFLOPS, {band} tile rows per dispatch, {per_submit} dispatches per submit (setup {:.1} s)",
            num(rate / 1e9),
            setup_t0.elapsed().as_secs_f64()
        ));
        Ok(Matmul {
            kernel: k,
            a,
            b,
            c,
            a_host: ha,
            b_host: hb,
            n,
            band,
            per_submit,
            note,
        })
    }

    /// The matmul pipeline for FP32 or FP16, compiled on first use and reused
    /// for the rest of the run.
    fn matmul_kernel(&self, f16: bool) -> Result<Rc<Kernel>, String> {
        let slot = f16 as usize;
        let cached = self.matmul_kernels.borrow()[slot].clone();
        if let Some(k) = cached {
            return Ok(k);
        }
        let t0 = Instant::now();
        let k = Rc::new(self.kernel(
            &shaders::matmul(f16),
            "main",
            &[Bind::ReadOnly, Bind::ReadOnly, Bind::Storage, Bind::Uniform],
        )?);
        self.log(format!(
            "Matmul {} shader compiled in {:.2} s",
            if f16 { "FP16" } else { "FP32" },
            t0.elapsed().as_secs_f64()
        ));
        self.matmul_kernels.borrow_mut()[slot] = Some(k.clone());
        Ok(k)
    }

    /// The probe runs a 256 matmul, which barely fills a fast GPU. Its speed
    /// estimate can be 30x too low, which makes every dispatch far shorter than
    /// the limits allow. So time one real matmul (safe, since it uses the
    /// conservative plan) and re-plan from that measurement.
    fn refine_plan(&self, m: &mut Matmul) {
        let tiles = m.n / 64;
        let secs = timed(|| self.run_matmul(m, 1));
        let software = self.g.info.device_type == wgpu::DeviceType::Cpu;
        let (band, per_submit) = matmul_plan(secs / tiles as f64, tiles, software);
        self.log(format!(
            "Matmul plan from a measured {:.2} ms per tile row: {band} tile rows per dispatch, {per_submit} dispatches per submit{}",
            secs / tiles as f64 * 1e3,
            if software { " (software adapter)" } else { "" }
        ));
        m.band = band;
        m.per_submit = per_submit;
    }

    /// Runs `reps` full matmuls as bands of tile rows, splitting the work
    /// over several submissions, then waits for all of it.
    fn run_matmul(&self, m: &Matmul, reps: u32) {
        let tiles = m.n / 64;
        let mut bands = Vec::new();
        for _ in 0..reps {
            let mut off = 0;
            while off < tiles {
                let rows = m.band.min(tiles - off);
                bands.push((off, rows));
                off += rows;
            }
        }
        for chunk in bands.chunks(m.per_submit) {
            // Each band's bind group is made just before its dispatch. Ones made
            // at setup all read the last band's row offset on WARP, while a bind
            // group made right before a single dispatch came out exact.
            let mut planned = Vec::with_capacity(chunk.len());
            for &(off, rows) in chunk {
                match self.dims_bind(&m.kernel, (&m.a, &m.b, &m.c), m.n, off) {
                    Ok(bind) => planned.push((bind, rows)),
                    Err(e) => self.log(format!("  band at tile row {off} not run: {e}")),
                }
            }
            let mut enc = self.g.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&m.kernel.pipeline);
                for (bind, rows) in &planned {
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(tiles, *rows, 1);
                }
            }
            self.g.queue.submit([enc.finish()]);
        }
        self.wait();
    }

    /// Diagnostic for a failed check: runs the whole matmul as one dispatch
    /// with a freshly made row offset of 0, then checks it again. If that
    /// passes, the separate per-band dispatches are at fault; if it fails the
    /// same way, the kernel or the copy back is.
    fn single_dispatch_retry(&self, m: &Matmul, f16: bool) {
        self.log("  retry as one dispatch over all tile rows, with a fresh row offset of 0");
        let tiles = m.n / 64;
        let bind = match self.dims_bind(&m.kernel, (&m.a, &m.b, &m.c), m.n, 0) {
            Ok(b) => b,
            Err(e) => {
                self.log(format!("  retry skipped: {e}"));
                return;
            }
        };
        let mut enc = self.g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&m.kernel.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(tiles, tiles, 1);
        }
        self.g.queue.submit([enc.finish()]);
        self.wait();
        match self.matmul_check(&m.c, &m.a_host, &m.b_host, m.n, f16) {
            Ok((bad, _)) => self.log(format!("  retry check: {bad}/256 bad")),
            Err(e) => self.log(format!("  retry check failed: {e}")),
        }
    }

    /// Compares 256 sampled entries of C against a CPU dot product.
    fn matmul_check(
        &self,
        c: &wgpu::Buffer,
        ha: &[f32],
        hb: &[f32],
        n: u32,
        f16: bool,
    ) -> Result<(usize, f64), String> {
        let elem = if f16 { 2 } else { 4 };
        let raw = self.read_back(c, (n as u64) * (n as u64) * elem)?;
        let gpu: Vec<f32> = if f16 {
            bytemuck::cast_slice::<u8, half::f16>(&raw)
                .iter()
                .map(|h| h.to_f32())
                .collect()
        } else {
            bytemuck::cast_slice::<u8, f32>(&raw).to_vec()
        };
        let n = n as usize;
        let mut bad = 0;
        let mut max_err = 0.0f64;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..256 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let (r, col) = ((x as usize) % n, ((x >> 32) as usize) % n);
            let terms: Vec<f64> = (0..n)
                .map(|k| ha[r * n + k] as f64 * hb[k * n + col] as f64)
                .collect();
            let want: f64 = terms.iter().sum();
            let got = gpu[r * n + col] as f64;
            let err = (got - want).abs();
            max_err = max_err.max(err);
            // FP32 sums are exact here; FP16 may round once partial sums pass 128.
            let tol = if f16 { 0.5 + want.abs() * 2e-3 } else { 1e-3 };
            if err > tol || err.is_nan() {
                bad += 1;
                // The first few misses say which part of the K sum went wrong.
                if bad <= 3 {
                    self.log(format!(
                        "  wrong C[{r}][{col}]: GPU {got:.4}, CPU {want:.4}: {}",
                        explain_bad(got, &terms)
                    ));
                }
            }
        }
        if bad > 0 {
            // Which 64x64 tiles of C were written, and whether they are right.
            self.log("  tile map: one character per 64x64 tile of C; . never written, o matches the CPU, x wrong");
            for row in tile_map(&gpu, ha, hb, n, f16) {
                self.log(format!("    {row}"));
            }
        }
        Ok((bad, max_err))
    }

    fn matmul(&self, n: u32, f16: bool) -> Result<TestResult, String> {
        let test = if f16 {
            Test::MatmulF16
        } else {
            Test::MatmulF32
        };
        if f16 && !self.g.has_f16 {
            let dx12 = self.g.info.backend == wgpu::Backend::Dx12;
            return Ok(TestResult {
                test,
                value: 0.0,
                unit: "",
                detail: f16_skip_detail(dx12, gpu::dxc_loadable()),
                status: Status::Skipped,
            });
        }
        let mut m = self.matmul_setup(n, f16)?;
        self.refine_plan(&mut m);
        let n = m.n;
        let (reps, times) = self.calibrated(6, |reps| self.run_matmul(&m, reps));
        let (best, avg) = summarize(&times);
        let flops = 2.0 * (n as f64).powi(3) * reps as f64;
        let (bad, max_err) = self.matmul_check(&m.c, &m.a_host, &m.b_host, n, f16)?;
        if bad > 0 && !self.stopped() {
            self.single_dispatch_retry(&m, f16);
        }
        Ok(TestResult {
            test,
            value: flops / avg / 1e9,
            unit: "GFLOPS",
            detail: format!(
                "N={n}, best {} GFLOPS, check {}/256 bad (max err {max_err:.3}){}",
                num(flops / best / 1e9),
                bad,
                m.note
            ),
            status: if bad == 0 { Status::Pass } else { Status::Fail },
        })
    }

    /// Free VRAM right now, from the most reliable source available:
    /// the DXGI per-process budget on Windows, else NVML/sysfs free memory.
    /// Returns (bytes, description of where the number came from).
    fn vram_available(&self) -> Option<(u64, String)> {
        let mut best: Option<(u64, String)> = None;
        match crate::dxgi::query(&self.g.info) {
            Ok(m) => {
                self.log(format!(
                    "DXGI: budget {} for this process, using {} VRAM + {} system RAM, card has {}",
                    fmt_bytes(m.budget),
                    fmt_bytes(m.local_usage),
                    fmt_bytes(m.nonlocal_usage),
                    fmt_bytes(m.dedicated)
                ));
                best = Some((m.available(), "Windows (DXGI) VRAM budget".into()));
            }
            // Empty on platforms without DXGI, where there is nothing to say.
            Err(why) if !why.is_empty() => self.log(why),
            Err(_) => {}
        }
        let sensor = crate::sensors::Sensor::for_adapter(&self.g.info);
        if let Some(free) = sensor.vram_free() {
            let src = sensor.static_info().source;
            self.log(format!("{src}: {} VRAM free", fmt_bytes(free)));
            // Use the smaller figure when both exist; either one being
            // exceeded leads to paging.
            if best.as_ref().is_none_or(|(b, _)| free < *b) {
                best = Some((free, format!("free VRAM ({src})")));
            }
        }
        best
    }

    fn vram(&self, cfg: &Config) -> Result<TestResult, String> {
        // Measure right before allocating: earlier readings go stale as other
        // apps (and earlier tests) allocate and free memory.
        let available = self.vram_available();
        let dxgi_before = crate::dxgi::query(&self.g.info).ok();
        let target = if cfg.vram_auto {
            match &available {
                Some((free, src)) => {
                    let t = (*free as f64 * 0.9) as u64 / MIB * MIB;
                    self.log(format!(
                        "VRAM test: auto size {} (90% of {} from {src})",
                        fmt_bytes(t),
                        fmt_bytes(*free)
                    ));
                    t
                }
                None => {
                    self.log("VRAM test: free VRAM unknown on this system; auto size falls back to 1 GiB");
                    1024 * MIB
                }
            }
        } else {
            let t = cfg.vram_mb * MIB;
            if let Some((free, src)) = &available
                && t > *free
            {
                self.log(format!(
                    "WARNING: VRAM target {} exceeds {} available ({src}); expect part of it in system RAM",
                    fmt_bytes(t),
                    fmt_bytes(*free)
                ));
            }
            t
        };
        let target = target.max(64 * MIB);
        // ~32 chunks, so the "how much was really VRAM" estimate is fine-grained
        // (about 3% of the target) without thousands of tiny allocations.
        let chunk = (target / 32)
            .next_multiple_of(16 * MIB)
            .clamp(32 * MIB, 256 * MIB)
            .min(self.g.limits.max_storage_buffer_binding_size)
            .min(self.g.limits.max_buffer_size)
            / 1024
            * 1024;
        let k_fill = self.kernel(
            shaders::PATTERN,
            "fill",
            &[Bind::Storage, Bind::Uniform, Bind::Storage],
        )?;
        let k_verify = self.kernel(
            shaders::PATTERN,
            "verify",
            &[Bind::Storage, Bind::Uniform, Bind::Storage],
        )?;
        let params = self.try_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "params",
        )?;
        let errs = self.try_buffer(
            8,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "errs",
        )?;

        // Allocate chunks until the target is reached or the driver refuses.
        let mut chunks: Vec<wgpu::Buffer> = Vec::new();
        let mut total = 0u64;
        let mut note = String::new();
        while total < target && !self.stopped() {
            let size = chunk.min(target - total) / 1024 * 1024;
            if size == 0 {
                break;
            }
            match self.try_buffer(size, wgpu::BufferUsages::STORAGE, "vram") {
                Ok(b) => {
                    chunks.push(b);
                    total += size;
                    self.progress(0.1 * total as f32 / target as f32);
                }
                Err(e) => {
                    note = format!(" — allocation stopped early: {e}");
                    break;
                }
            }
        }
        self.log(format!(
            "VRAM test: allocated {} in {} chunks{note}",
            fmt_bytes(total),
            chunks.len()
        ));
        if chunks.is_empty() {
            return Err("could not allocate any VRAM".into());
        }

        let fill_bgs: Vec<_> = chunks
            .iter()
            .map(|b| self.bind(&k_fill, &[b, &params, &errs]))
            .collect();
        let verify_bgs: Vec<_> = chunks
            .iter()
            .map(|b| self.bind(&k_verify, &[b, &params, &errs]))
            .collect();
        let seeds = [
            0xA5A5_A5A5u32,
            0x5A5A_5A5A,
            0xDEAD_BEEF,
            0x0F0F_F0F0,
            0x1234_5678,
            0xFFFF_0000,
        ];
        let passes = cfg.vram_passes.clamp(1, seeds.len() as u32) as usize;
        let mut total_errors = 0u64;
        let mut first_bad: Option<String> = None;
        // Seconds per pass for filling and for verifying all the chunks.
        let mut pass_fill: Vec<f64> = Vec::new();
        let mut pass_verify: Vec<f64> = Vec::new();
        // Best fill speed seen per chunk (bytes/s), to spot chunks the driver
        // quietly placed in system RAM instead of VRAM.
        let mut chunk_speed = vec![0.0f64; chunks.len()];
        let steps = (passes * chunks.len() * 2) as f32;
        let mut step = 0.0;

        for (pass, &seed) in seeds.iter().take(passes).enumerate() {
            let (mut fill_t, mut verify_t) = (0.0f64, 0.0f64);
            self.g
                .queue
                .write_buffer(&params, 0, bytemuck::cast_slice(&[seed, 0, 0, 0]));
            // Fill everything first so the data must survive while the rest
            // of VRAM is being written.
            for (i, bg) in fill_bgs.iter().enumerate() {
                if self.stopped() {
                    return Err("stopped".into());
                }
                let t = timed(|| {
                    self.dispatch(&k_fill, bg, grid(chunks[i].size() / 4), 1);
                    self.wait();
                });
                fill_t += t;
                chunk_speed[i] = chunk_speed[i].max(chunks[i].size() as f64 / t.max(1e-9));
                step += 1.0;
                self.progress(0.1 + 0.9 * step / steps);
            }
            for (i, bg) in verify_bgs.iter().enumerate() {
                if self.stopped() {
                    return Err("stopped".into());
                }
                self.g
                    .queue
                    .write_buffer(&errs, 0, bytemuck::cast_slice(&[0u32, u32::MAX]));
                verify_t += timed(|| {
                    self.dispatch(&k_verify, bg, grid(chunks[i].size() / 4), 1);
                    self.wait();
                });
                let e = self.read_back(&errs, 8)?;
                let e: &[u32] = bytemuck::cast_slice(&e);
                if e[0] > 0 {
                    total_errors += e[0] as u64;
                    let offset = i as u64 * chunk + e[1] as u64 * 4;
                    self.log(format!(
                        "pass {}: {} bad words in chunk {i}, first at offset {:#x}",
                        pass + 1,
                        e[0],
                        offset
                    ));
                    first_bad.get_or_insert(format!("first at {:#x}", offset));
                }
                step += 1.0;
                self.progress(0.1 + 0.9 * step / steps);
            }
            self.log(format!(
                "VRAM pass {}: fill {} GB/s, verify {} GB/s",
                pass + 1,
                num(total as f64 / fill_t.max(1e-9) / 1e9),
                num(total as f64 / verify_t.max(1e-9) / 1e9)
            ));
            pass_fill.push(fill_t);
            pass_verify.push(verify_t);
        }

        // Chunks under 40% of the fastest chunk's speed are almost certainly
        // not in VRAM (system RAM over PCIe is several times slower). Tiny
        // chunks are skipped because fixed overhead dominates their timing.
        // On Windows, DXGI says directly how much of this process's GPU memory
        // sits in system RAM; growth during the test is spillover.
        let spilled = match (dxgi_before, crate::dxgi::query(&self.g.info).ok()) {
            (Some(a), Some(b)) => {
                let grew = b.nonlocal_usage.saturating_sub(a.nonlocal_usage);
                self.log(format!(
                    "DXGI during test: {} in VRAM, {} in system RAM ({} more than before the test)",
                    fmt_bytes(b.local_usage),
                    fmt_bytes(b.nonlocal_usage),
                    fmt_bytes(grew)
                ));
                grew
            }
            _ => 0,
        };
        let fastest = chunk_speed.iter().cloned().fold(0.0, f64::max);
        let is_slow =
            |speed: &[f64], i: usize| chunks[i].size() >= chunk / 2 && speed[i] < 0.4 * fastest;
        // Re-time suspects a few more times so a one-off OS hiccup during a
        // ~1 ms fill can't fake a slow chunk. The data is already verified,
        // so overwriting it here is harmless.
        for i in 0..chunks.len() {
            if is_slow(&chunk_speed, i) && !self.stopped() {
                for _ in 0..3 {
                    let t = timed(|| {
                        self.dispatch(&k_fill, &fill_bgs[i], grid(chunks[i].size() / 4), 1);
                        self.wait();
                    });
                    chunk_speed[i] = chunk_speed[i].max(chunks[i].size() as f64 / t.max(1e-9));
                }
            }
        }
        let slow: Vec<usize> = (0..chunks.len())
            .filter(|&i| is_slow(&chunk_speed, i))
            .collect();
        self.log(format!(
            "VRAM per-chunk fill speed (GB/s): {}",
            chunk_speed
                .iter()
                .map(|v| num(v / 1e9))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        // Spillover is reported if either signal sees it: slow chunks, or
        // DXGI counting at least one chunk's worth of new system-RAM usage.
        let spill_seen = spilled >= chunk;
        let paging = if slow.is_empty() && !spill_seen {
            String::new()
        } else {
            let fast_bytes: u64 = (0..chunks.len())
                .filter(|i| !slow.contains(i))
                .map(|i| chunks[i].size())
                .sum();
            // Best estimate of what really fit in VRAM.
            let fit = if spill_seen {
                fast_bytes.min(total.saturating_sub(spilled))
            } else {
                fast_bytes
            } / MIB;
            let mut why = Vec::new();
            if !slow.is_empty() {
                why.push(format!(
                    "{} of {} chunks filled at under 40% of the fastest chunk's speed",
                    slow.len(),
                    chunks.len()
                ));
            }
            if spill_seen {
                why.push(format!(
                    "Windows reports {} of this test's memory in system RAM",
                    fmt_bytes(spilled)
                ));
            }
            self.log(format!(
                "WARNING: part of the test ran in system RAM, not VRAM ({}). About {fit} MiB fit in VRAM; \
                 set 'VRAM to test' to that (or use Auto) for a pure VRAM test.",
                why.join("; ")
            ));
            format!(" — only ~{fit} MiB fit in VRAM; rest spilled to system RAM")
        };

        // Speeds come from the fastest pass. Averaging in a slow first pass
        // (first-touch allocation) would report a figure no chunk reached.
        Ok(TestResult {
            test: Test::Vram,
            value: total as f64 / MIB as f64,
            unit: "MiB tested",
            detail: format!(
                "{passes} passes, {total_errors} errors{}, best pass: fill {} GB/s, verify {} GB/s{note}{paging}",
                first_bad.map(|s| format!(" ({s})")).unwrap_or_default(),
                num(best_pass_gbs(total, &pass_fill)),
                num(best_pass_gbs(total, &pass_verify)),
            ),
            status: if total_errors > 0 {
                Status::Fail
            } else if !paging.is_empty() {
                Status::Warn
            } else {
                Status::Pass
            },
        })
    }

    fn stress(&self, cfg: &Config) -> Result<TestResult, String> {
        let mut m = self.matmul_setup(cfg.stress_n, false)?;
        self.refine_plan(&mut m);
        let n = m.n;
        let flops_per = 2.0 * (n as f64).powi(3);
        self.run_matmul(&m, 1);
        let t1 = timed(|| self.run_matmul(&m, 1));
        let reps = ((0.1 / t1.max(1e-6)) as u32).clamp(1, 500);
        let start = Instant::now();
        let mut samples = Vec::new();
        while start.elapsed().as_secs_f64() < cfg.stress_secs && !self.stopped() {
            let t = timed(|| self.run_matmul(&m, reps));
            let gflops = flops_per * reps as f64 / t / 1e9;
            samples.push(gflops);
            let _ = self.tx.send(Msg::Stress { t: clock(), gflops });
            self.progress((start.elapsed().as_secs_f64() / cfg.stress_secs) as f32);
        }
        if samples.is_empty() {
            return Err("stopped".into());
        }
        let avg = samples.iter().sum::<f64>() / samples.len() as f64;
        let tenth = (samples.len() / 10).max(1);
        let head = samples[..tenth].iter().sum::<f64>() / tenth as f64;
        let tail = samples[samples.len() - tenth..].iter().sum::<f64>() / tenth as f64;
        let fall = (head - tail) / head * 100.0;
        // The result must still be correct after minutes at full load.
        let (bad, _) = self.matmul_check(&m.c, &m.a_host, &m.b_host, n, false)?;
        Ok(TestResult {
            test: Test::Stress,
            value: avg,
            unit: "GFLOPS avg",
            detail: format!(
                "{:.0}s at N={n}, start {} → end {} GFLOPS ({}), check {bad}/256 bad{}",
                start.elapsed().as_secs_f64(),
                num(head),
                num(tail),
                trend_text(fall),
                m.note
            ),
            status: if bad == 0 { Status::Pass } else { Status::Fail },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_format() {
        assert_eq!(num(2.123), "2.12");
        assert_eq!(num(31.73), "31.7");
        assert_eq!(num(41234.0), "41234");
    }

    #[test]
    fn grid_covers_large_buffers() {
        for elems in [1u64, 255, 256, 257, 65535 * 256, 65535 * 256 + 1, 1 << 28] {
            let (x, y) = grid(elems);
            assert!(x <= 65535 && y <= 65535);
            assert!(x as u64 * y as u64 * shaders::WG as u64 >= elems);
        }
    }

    /// The CPU check relies on every product and partial sum being exact in
    /// f32: inputs are multiples of 1/4 in [-1, 1].
    #[test]
    fn matmul_inputs_are_exact_in_f32() {
        let n = 128;
        let (a, b) = Bench::matmul_inputs(n, 42);
        assert!(
            a.iter()
                .chain(&b)
                .all(|v| (v * 4.0).fract() == 0.0 && v.abs() <= 1.0)
        );
        let mut f32_sum = 0.0f32;
        let mut f64_sum = 0.0f64;
        for k in 0..n {
            f32_sum += a[k] * b[k * n];
            f64_sum += a[k] as f64 * b[k * n] as f64;
        }
        assert_eq!(f32_sum as f64, f64_sum);
    }

    #[test]
    fn test_keys_round_trip() {
        for t in Test::ALL {
            assert_eq!(Test::from_key(t.key()), Some(t));
        }
    }

    fn outcome(status: Status, value: f64) -> Result<TestResult, String> {
        Ok(TestResult {
            test: Test::MatmulF32,
            value,
            unit: "GFLOPS",
            detail: "d".into(),
            status,
        })
    }

    #[test]
    fn errors_without_stop_stay_failures() {
        let r = finish(Test::Vram, Err("could not allocate".into()), false);
        assert!(matches!(r.status, Status::Fail));
        assert_eq!(r.detail, "could not allocate");
    }

    #[test]
    fn stop_is_reported_as_stop_not_fail() {
        let r = finish(Test::Vram, Err("stopped".into()), true);
        assert!(matches!(r.status, Status::Stopped));
        assert_eq!(r.status.label(), "STOP");
    }

    #[test]
    fn partial_result_after_stop_keeps_its_number() {
        let r = finish(Test::Stress, outcome(Status::Pass, 12.5), true);
        assert!(matches!(r.status, Status::Stopped));
        assert_eq!(r.value, 12.5);
        assert!(r.detail.starts_with("stopped early"));
    }

    #[test]
    fn stop_with_no_finished_measurement_shows_no_number() {
        // With zero timed batches the rate is infinite, which must not be shown.
        let r = finish(Test::Upload, outcome(Status::Pass, f64::INFINITY), true);
        assert!(matches!(r.status, Status::Stopped));
        assert_eq!(r.value, 0.0);
    }

    #[test]
    fn failed_check_stays_fail_even_when_stopped() {
        let r = finish(Test::MatmulF32, outcome(Status::Fail, 3.0), true);
        assert!(matches!(r.status, Status::Fail));
    }

    #[test]
    fn plan_keeps_dispatches_and_submits_under_their_limits() {
        // A slow card: one tile row takes 7 ms, so only a few rows fit a dispatch.
        let (band, per_submit) = plan_band(0.007, 32);
        assert_eq!((band, per_submit), (4, 4));
        // A fast card: a row takes 0.2 ms, so the whole matrix fits one dispatch.
        let (band, per_submit) = plan_band(0.0002, 32);
        assert_eq!(band, 32);
        assert!(band as f64 * 0.0002 <= MATMUL_DISPATCH_MAX);
        assert!(per_submit as f64 * band as f64 * 0.0002 <= MATMUL_SUBMIT_MAX);
        // Never zero, even for a row that takes longer than a whole dispatch.
        assert_eq!(plan_band(1.0, 32), (1, 1));
    }

    #[test]
    fn software_adapters_get_one_dispatch_over_all_rows() {
        assert_eq!(matmul_plan(1.0, 32, true), (32, 1));
        assert_eq!(matmul_plan(0.0002, 32, false), plan_band(0.0002, 32));
    }

    #[test]
    fn best_pass_is_the_fastest_pass() {
        // 1 GB in 0.5 s and then 0.25 s: the fastest pass is 4 GB/s.
        let gbs = best_pass_gbs(1_000_000_000, &[0.5, 0.25]);
        assert!((gbs - 4.0).abs() < 1e-9, "{gbs}");
    }

    #[test]
    fn tile_map_marks_missing_and_wrong_tiles() {
        // 128x128 of ones times ones: every entry is 128. Tile (0,1) is never
        // written, tile (1,0) holds a wrong value.
        let n = 128;
        let ones = vec![1.0f32; n * n];
        let mut gpu = vec![128.0f32; n * n];
        for r in 0..64 {
            for c in 64..128 {
                gpu[r * n + c] = 0.0;
            }
        }
        for r in 64..128 {
            for c in 0..64 {
                gpu[r * n + c] = 5.0;
            }
        }
        assert_eq!(tile_map(&gpu, &ones, &ones, n, false), ["o.", "xo"]);
    }

    #[test]
    fn wrong_entries_are_traced_to_the_k_range() {
        let ones = vec![1.0; 64]; // total 64, four K blocks of 16
        assert!(explain_bad(32.0, &ones).contains("only k < 32 summed"));
        assert!(explain_bad(48.0, &ones).contains("k 0..16 missing"));
        assert!(explain_bad(80.0, &ones).contains("counted twice"));
        assert!(explain_bad(0.0, &ones).contains("never written"));
        assert!(explain_bad(7.0, &ones).contains("no simple partial sum"));
    }

    #[test]
    fn time_budget_needs_a_sample_and_the_full_budget() {
        assert!(!budget_reached(MEASURE_BUDGET * 10.0, 0));
        assert!(!budget_reached(MEASURE_BUDGET - 0.1, 3));
        assert!(budget_reached(MEASURE_BUDGET, 1));
    }

    #[test]
    fn fp16_skip_names_the_cause_on_dx12() {
        assert!(f16_skip_detail(true, false).contains("dxcompiler.dll"));
        assert!(f16_skip_detail(true, false).contains("FXC"));
        assert!(f16_skip_detail(true, true).contains("shader model 6.2"));
        assert_eq!(
            f16_skip_detail(false, false),
            "adapter does not support SHADER_F16"
        );
    }

    #[test]
    fn stress_trend_reads_as_drop_or_gain() {
        assert_eq!(trend_text(9.94), "9.9% drop");
        assert_eq!(trend_text(0.0), "0.0% drop");
        assert_eq!(trend_text(-3.2), "3.2% gain");
    }
}
