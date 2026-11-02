//! Headless mode: `gpu-workbench-cli [options]` (or `gpu-workbench --cli`).

use crate::bench::{self, Config, Msg, Status, Test, limits};
use crate::gpu::{self, fmt_bytes};
use crate::sensors::Sensor;
use std::fmt::Display;
use std::ops::RangeInclusive;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Prints a line and appends it to the session log.
macro_rules! out {
    () => { out!("") };
    ($($t:tt)*) => {{
        let s = format!($($t)*);
        // Ignore a closed pipe (e.g. `| head`) instead of panicking.
        let _ = std::io::Write::write_fmt(&mut std::io::stdout(), format_args!("{s}\n"));
        crate::logfile::line(&s);
    }};
}

/// Same, to stderr.
macro_rules! err {
    ($($t:tt)*) => {{
        let s = format!($($t)*);
        let _ = std::io::Write::write_fmt(&mut std::io::stderr(), format_args!("{s}\n"));
        crate::logfile::line(&format!("ERROR: {s}"));
    }};
}

const HELP: &str = "\
gpu-workbench — GPU info, benchmarks and stress tests

USAGE:
    gpu-workbench-cli --list          list adapters and exit
    gpu-workbench-cli [OPTIONS]       run tests headless (no options: quick suite)
    gpu-workbench                     launch the GUI

OPTIONS (numbers outside the ranges are refused):
    --adapter N        adapter index from --list (default: best GPU found)
    --tests LIST       comma list: upload,readback,copy,bandwidth,matmul32,
                       matmul16,vram,stress | quick | all   (default: quick)
    --transfer-mb N    buffer size for transfer tests, 16-4096 (default 256)
    --matmul-n N       matrix size, 256-8192, multiple of 64 (default 2048)
    --vram-mb N        VRAM to fill and verify, 64-262144 (default: auto, 90% of
                       the free VRAM measured when the test starts, else 1024)
    --vram-passes N    patterns per VRAM run, 1-6 (default 3)
    --stress-secs N    sustained stress duration, 5-3600 (default 60)
    --stress-n N       matrix size for the stress test, 256-8192, multiple of 64
                       (default 2048)

ENV:
    WGPU_BACKEND=vulkan|dx12   restrict backends
";

/// Options after parsing and range checks.
#[derive(Debug)]
pub struct Args {
    pub cfg: Config,
    pub adapter: Option<usize>,
    pub tests: Vec<Test>,
    pub list_only: bool,
    pub help: bool,
}

/// Parses the command line. Every number is checked against the range the GUI
/// allows, so a typo or a zero is refused instead of silently running with a
/// default (or with 0, which used to report a meaningless PASS).
pub fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut out = Args {
        cfg: Config::default(),
        adapter: None,
        tests: Test::QUICK.to_vec(),
        list_only: false,
        help: false,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let flag = flag.as_str();
        match flag {
            "--cli" => {}
            "--list" => out.list_only = true,
            "-h" | "--help" => out.help = true,
            "--adapter" => out.adapter = Some(number(flag, value(&mut it, flag)?, 0..=usize::MAX)?),
            "--transfer-mb" => {
                out.cfg.transfer_mb = number(flag, value(&mut it, flag)?, limits::TRANSFER_MB)?
            }
            "--matmul-n" => {
                out.cfg.matmul_n = tiled(flag, value(&mut it, flag)?, limits::MATMUL_N)?
            }
            "--vram-mb" => {
                out.cfg.vram_mb = number(flag, value(&mut it, flag)?, limits::VRAM_MB)?;
                out.cfg.vram_auto = false;
            }
            "--vram-passes" => {
                out.cfg.vram_passes = number(flag, value(&mut it, flag)?, limits::VRAM_PASSES)?
            }
            "--stress-secs" => {
                out.cfg.stress_secs = number(flag, value(&mut it, flag)?, limits::STRESS_SECS)?
            }
            "--stress-n" => {
                out.cfg.stress_n = tiled(flag, value(&mut it, flag)?, limits::MATMUL_N)?
            }
            "--tests" => out.tests = parse_tests(value(&mut it, flag)?)?,
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    Ok(out)
}

/// The value after a flag, or an error saying the flag is missing one.
fn value<'a>(it: &mut std::slice::Iter<'a, String>, flag: &str) -> Result<&'a str, String> {
    it.next()
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// Parses `raw` as a number in `range`. Errors name the flag, the bad value,
/// and the allowed range.
fn number<T>(flag: &str, raw: &str, range: RangeInclusive<T>) -> Result<T, String>
where
    T: FromStr + PartialOrd + Copy + Display,
{
    let v: T = raw
        .parse()
        .map_err(|_| format!("{flag}: '{raw}' is not a number"))?;
    if range.contains(&v) {
        Ok(v)
    } else {
        Err(format!(
            "{flag}: {v} is out of range, use {}-{}",
            range.start(),
            range.end()
        ))
    }
}

/// Like `number`, but the value must also be a multiple of 64 (the matmul tile).
fn tiled(flag: &str, raw: &str, range: RangeInclusive<u32>) -> Result<u32, String> {
    let n = number(flag, raw, range)?;
    if n.is_multiple_of(64) {
        Ok(n)
    } else {
        Err(format!("{flag}: {n} must be a multiple of 64"))
    }
}

fn parse_tests(v: &str) -> Result<Vec<Test>, String> {
    match v {
        "all" => Ok(Test::ALL.to_vec()),
        "quick" => Ok(Test::QUICK.to_vec()),
        _ => v
            .split(',')
            .map(|k| Test::from_key(k.trim()).ok_or_else(|| format!("unknown test '{k}'")))
            .collect(),
    }
}

pub fn main(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(p) => p,
        Err(e) => {
            err!("{e} (see --help)");
            return 2;
        }
    };
    if parsed.help {
        print!("{HELP}");
        println!("\nTESTS:");
        for t in Test::ALL {
            println!("    {:<10} {}", t.key(), t.help());
        }
        return 0;
    }
    let Args {
        cfg,
        adapter,
        tests,
        list_only,
        ..
    } = parsed;

    if let Some(p) = crate::logfile::path() {
        out!("Log: {}", p.display());
    }
    let adapters = gpu::enumerate();
    if adapters.is_empty() {
        err!("No Vulkan/DX12 adapters found. Check GPU drivers.");
        return 1;
    }
    out!("Adapters:");
    for a in &adapters {
        out!("  {}", a.label());
    }
    if list_only {
        return 0;
    }

    let idx = adapter.unwrap_or_else(|| default_adapter(&adapters));
    let Some(sel) = adapters.get(idx) else {
        err!("adapter #{idx} does not exist");
        return 2;
    };
    print_info(sel);

    let sensor = Sensor::for_adapter(&sel.info);
    let si = sensor.static_info();
    out!("Telemetry: {}", si.source);
    for (k, v) in &si.extra {
        out!("  {k:<20} {v}");
    }
    // Full per-reading status goes to the log file only, to keep output short.
    for d in sensor.diagnose() {
        crate::logfile::line(&format!("    sensor {d}"));
    }

    out!();
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let handle = bench::spawn(idx, tests, cfg, tx, stop);
    let mut failed = false;
    for msg in rx {
        match msg {
            Msg::Log(s) => out!("  · {s}"),
            Msg::Error(s) => {
                err!("ERROR: {s}");
                failed = true;
            }
            Msg::Started(t) => out!("==> {}", t.name()),
            Msg::Progress(_) | Msg::Stress { .. } => {}
            Msg::Result(r) => {
                if matches!(r.status, Status::Fail) {
                    failed = true;
                }
                let tag = r.status.label();
                if matches!(r.status, Status::Skipped) || r.unit.is_empty() {
                    out!("  [{tag}] {}", r.detail);
                } else {
                    out!(
                        "  [{tag}] {} {}  — {}",
                        bench::num(r.value),
                        r.unit,
                        r.detail
                    );
                }
                if let Some(s) = sensor.available().then(|| sensor.sample(0.0)) {
                    let mut parts = Vec::new();
                    if let Some(t) = s.temp_c {
                        parts.push(format!("{t:.0}°C"));
                    }
                    if let Some(p) = s.power_w {
                        parts.push(format!("{p:.0} W"));
                    }
                    if let Some(c) = s.core_mhz {
                        parts.push(format!("{c:.0} MHz"));
                    }
                    if !parts.is_empty() {
                        out!("         now: {}", parts.join(", "));
                    }
                }
            }
            Msg::Done => break,
        }
    }
    let _ = handle.join();
    if failed { 1 } else { 0 }
}

/// Prefers adapters with compute, then discrete > integrated > others, and
/// Vulkan/DX12 over OpenGL for the same kind of device.
pub fn default_adapter(adapters: &[gpu::AdapterSummary]) -> usize {
    let rank = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    adapters
        .iter()
        .min_by_key(|a| (!a.has_compute(), rank(a.effective_type()), a.is_gl()))
        .map(|a| a.index)
        .unwrap_or(0)
}

fn print_info(a: &gpu::AdapterSummary) {
    let i = &a.info;
    let l = &a.limits;
    out!("\nSelected #{}: {}", a.index, i.name);
    out!(
        "  Compute shaders {}",
        if a.has_compute() {
            "yes"
        } else {
            "NO (needs GL 4.3 on OpenGL)"
        }
    );
    out!(
        "  Vendor/device   {} ({:#06x}) / {:#06x}",
        a.vendor_name(),
        i.vendor,
        i.device
    );
    out!(
        "  Type / backend  {} / {}",
        gpu::device_type_str(a.effective_type()),
        gpu::backend_str(i.backend)
    );
    out!(
        "  Driver          {}",
        format!("{} {}", i.driver, i.driver_info).trim()
    );
    out!("  Max buffer      {}", fmt_bytes(l.max_buffer_size));
    out!(
        "  Max storage bind {}",
        fmt_bytes(l.max_storage_buffer_binding_size)
    );
    out!(
        "  Workgroup max   {} invocations, {} shared",
        l.max_compute_invocations_per_workgroup,
        fmt_bytes(l.max_compute_workgroup_storage_size as u64)
    );
    out!(
        "  FP16 shaders    {}",
        a.features.contains(wgpu::Features::SHADER_F16)
    );
    out!(
        "  Timestamp query {}",
        a.features.contains(wgpu::Features::TIMESTAMP_QUERY)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backend, DeviceType, DownlevelCapabilities, DownlevelFlags};

    fn adapter(
        index: usize,
        device_type: DeviceType,
        backend: Backend,
        compute: bool,
    ) -> gpu::AdapterSummary {
        gpu::AdapterSummary {
            index,
            info: wgpu::AdapterInfo {
                name: format!("test adapter {index}"),
                vendor: 0x10DE,
                device: 0,
                device_type,
                device_pci_bus_id: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                backend,
                subgroup_min_size: 0,
                subgroup_max_size: 0,
                transient_saves_memory: None,
                limit_bucket: None,
            },
            limits: wgpu::Limits::default(),
            features: wgpu::Features::empty(),
            downlevel: DownlevelCapabilities {
                flags: if compute {
                    DownlevelFlags::COMPUTE_SHADERS
                } else {
                    DownlevelFlags::empty()
                },
                ..Default::default()
            },
        }
    }

    #[test]
    fn default_adapter_needs_compute_first() {
        let list = [
            adapter(0, DeviceType::DiscreteGpu, Backend::Vulkan, false),
            adapter(1, DeviceType::IntegratedGpu, Backend::Vulkan, true),
        ];
        assert_eq!(default_adapter(&list), 1);
    }

    #[test]
    fn default_adapter_prefers_discrete_over_integrated() {
        let list = [
            adapter(0, DeviceType::IntegratedGpu, Backend::Vulkan, true),
            adapter(1, DeviceType::DiscreteGpu, Backend::Vulkan, true),
        ];
        assert_eq!(default_adapter(&list), 1);
    }

    #[test]
    fn default_adapter_prefers_vulkan_over_opengl() {
        let list = [
            adapter(0, DeviceType::DiscreteGpu, Backend::Gl, true),
            adapter(1, DeviceType::DiscreteGpu, Backend::Vulkan, true),
        ];
        assert_eq!(default_adapter(&list), 1);
    }

    fn parse(line: &str) -> Result<Args, String> {
        let args: Vec<String> = line.split_whitespace().map(String::from).collect();
        parse_args(&args)
    }

    #[test]
    fn refuses_zero_tiny_and_non_numeric_values() {
        for bad in [
            "--transfer-mb 0",
            "--transfer-mb 8",
            "--vram-mb 0",
            "--vram-passes 0",
            "--vram-passes 7",
            "--stress-secs 0",
            "--stress-secs 4.9",
            "--stress-n 0",
            "--stress-n 128",
            "--matmul-n 0",
            "--matmul-n 100",
            "--matmul-n 300",
            "--adapter abc",
            "--adapter -1",
            "--transfer-mb",
        ] {
            assert!(parse(bad).is_err(), "'{bad}' should be refused");
        }
    }

    #[test]
    fn error_names_the_flag_and_range() {
        let e = parse("--stress-secs 0").unwrap_err();
        assert!(e.contains("--stress-secs") && e.contains("5-3600"), "{e}");
        // 300 is in range but not a whole number of 64-wide tiles.
        let e = parse("--matmul-n 300").unwrap_err();
        assert!(e.contains("multiple of 64"), "{e}");
        // 100 is below the minimum, so the range message comes first.
        let e = parse("--matmul-n 100").unwrap_err();
        assert!(e.contains("256-8192"), "{e}");
    }

    #[test]
    fn accepts_defaults_and_minimums() {
        parse("--transfer-mb 256 --matmul-n 2048 --vram-passes 3 --stress-secs 60 --stress-n 2048 --adapter 0")
            .unwrap();
        let a = parse("--transfer-mb 16 --matmul-n 256 --stress-secs 5 --stress-n 256").unwrap();
        assert_eq!(a.cfg.transfer_mb, 16);
        assert_eq!(a.cfg.stress_secs, 5.0);
    }

    #[test]
    fn vram_mb_switches_off_auto_sizing() {
        let a = parse("--vram-mb 512").unwrap();
        assert!(!a.cfg.vram_auto);
        assert_eq!(a.cfg.vram_mb, 512);
        assert!(parse("").unwrap().cfg.vram_auto);
    }
}
