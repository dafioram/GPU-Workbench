//! Headless mode: `gpu-workbench --cli [options]`.

use crate::bench::{self, Config, Msg, Status, Test};
use crate::gpu::{self, fmt_bytes};
use crate::sensors::Sensor;
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
    gpu-workbench                 launch the GUI
    gpu-workbench --list          list adapters and exit
    gpu-workbench --cli [OPTIONS] run tests headless

OPTIONS:
    --adapter N        adapter index from --list (default: first discrete GPU)
    --tests LIST       comma list: upload,readback,copy,bandwidth,matmul32,
                       matmul16,vram,stress | quick | all   (default: quick)
    --transfer-mb N    buffer size for transfer tests (default 256)
    --matmul-n N       matrix size, multiple of 64 (default 2048)
    --vram-mb N        VRAM to fill and verify (default: auto, 90% of the free
                       VRAM measured when the test starts, else 1024)
    --vram-passes N    patterns per VRAM run, 1-6 (default 3)
    --stress-secs N    sustained stress duration (default 60)
    --stress-n N       matrix size used by the stress test (default 2048)

ENV:
    WGPU_BACKEND=vulkan|dx12   restrict backends
";

pub fn main(args: &[String]) -> i32 {
    let mut cfg = Config::default();
    let mut adapter: Option<usize> = None;
    let mut tests: Vec<Test> = Test::QUICK.to_vec();
    let mut list_only = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_default();
        match a.as_str() {
            "--cli" => {}
            "--list" => list_only = true,
            "-h" | "--help" => {
                print!("{HELP}");
                println!("\nTESTS:");
                for t in Test::ALL {
                    println!("    {:<10} {}", t.key(), t.help());
                }
                return 0;
            }
            "--adapter" => adapter = val().parse().ok(),
            "--transfer-mb" => cfg.transfer_mb = val().parse().unwrap_or(cfg.transfer_mb),
            "--matmul-n" => cfg.matmul_n = val().parse().unwrap_or(cfg.matmul_n),
            "--vram-mb" => {
                cfg.vram_mb = val().parse().unwrap_or(cfg.vram_mb);
                cfg.vram_auto = false;
            }
            "--vram-passes" => cfg.vram_passes = val().parse().unwrap_or(cfg.vram_passes),
            "--stress-secs" => cfg.stress_secs = val().parse().unwrap_or(cfg.stress_secs),
            "--stress-n" => cfg.stress_n = val().parse().unwrap_or(cfg.stress_n),
            "--tests" => {
                let v = val();
                tests = match v.as_str() {
                    "all" => Test::ALL.to_vec(),
                    "quick" => Test::QUICK.to_vec(),
                    _ => {
                        let mut out = Vec::new();
                        for k in v.split(',') {
                            match Test::from_key(k.trim()) {
                                Some(t) => out.push(t),
                                None => {
                                    err!("unknown test '{k}'");
                                    return 2;
                                }
                            }
                        }
                        out
                    }
                }
            }
            other => {
                err!("unknown option '{other}'\n\n{HELP}");
                return 2;
            }
        }
    }

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
}
