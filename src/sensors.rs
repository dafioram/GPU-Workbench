//! Hardware telemetry: NVML for NVIDIA (Windows + Linux), amdgpu sysfs for
//! AMD on Linux. Anything else reports "not available".

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One telemetry reading. Every field is optional because support varies.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub t: f64,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub core_mhz: Option<f64>,
    pub mem_mhz: Option<f64>,
    pub fan_pct: Option<f64>,
}

/// Static facts the sensor library knows that wgpu doesn't.
#[derive(Clone, Debug, Default)]
pub struct StaticInfo {
    pub source: String,
    pub extra: Vec<(String, String)>,
}

#[allow(clippy::large_enum_variant)] // one instance per app; boxing buys nothing
enum Backend {
    Nvml {
        nvml: Nvml,
        index: u32,
    },
    #[cfg(target_os = "linux")]
    AmdSysfs {
        dev: std::path::PathBuf,
    },
    None,
}

pub struct Sensor {
    backend: Backend,
    /// Why no sensor backend is available (shown in the UI and log).
    note: Option<String>,
}

struct VramInfo {
    used: u64,
    total: u64,
}

/// VRAM usage via NVML. nvml-wrapper calls nvmlDeviceGetMemoryInfo_v2, which
/// older drivers (e.g. 47x for Kepler) don't export; fall back to the original
/// nvmlDeviceGetMemoryInfo, which every NVML version has.
fn nvml_memory(d: &nvml_wrapper::Device) -> Result<VramInfo, String> {
    let v2_err = match d.memory_info() {
        Ok(m) => {
            return Ok(VramInfo {
                used: m.used,
                total: m.total,
            });
        }
        Err(e) => e.to_string(),
    };
    let Ok(f) = d.nvml().lib().nvmlDeviceGetMemoryInfo.as_ref() else {
        return Err(v2_err);
    };
    let mut m = nvml_wrapper_sys::bindings::nvmlMemory_t {
        total: 0,
        free: 0,
        used: 0,
    };
    // SAFETY: `f` is the NVML v1 entry point loaded by nvml-wrapper, `handle`
    // is a live device handle from the same library, `m` is a valid out-pointer.
    let rc = unsafe { f(d.handle(), &mut m) };
    if rc == 0 {
        Ok(VramInfo {
            used: m.used,
            total: m.total,
        })
    } else {
        Err(format!("{v2_err}; v1 fallback returned NVML error {rc}"))
    }
}

/// Normalises PCI bus ids so "00000000:01:00.0" and "0000:01:00.0" compare equal.
fn bus_key(id: &str) -> String {
    let id = id.trim().to_ascii_lowercase();
    let parts: Vec<&str> = id.split(':').collect();
    if parts.len() >= 2 {
        parts[parts.len() - 2..].join(":")
    } else {
        id
    }
}

impl Sensor {
    /// Finds the sensor matching a wgpu adapter: by PCI bus id when wgpu reports
    /// one, otherwise by name, otherwise the first device of that vendor.
    pub fn for_adapter(info: &wgpu::AdapterInfo) -> Sensor {
        let mut note = None;
        if crate::gpu::is_nvidia(info) {
            match Self::try_nvml(info) {
                Ok(s) => return s,
                Err(e) => note = Some(e),
            }
        }
        #[cfg(target_os = "linux")]
        if crate::gpu::is_amd(info)
            && let Some(s) = Self::try_amd_sysfs(info)
        {
            return s;
        }
        Sensor {
            backend: Backend::None,
            note,
        }
    }

    fn try_nvml(info: &wgpu::AdapterInfo) -> Result<Sensor, String> {
        let nvml = Nvml::init().map_err(|e| format!("NVML failed to load: {e}"))?;
        let count = nvml
            .device_count()
            .map_err(|e| format!("NVML device count failed: {e}"))?;
        let want_bus = bus_key(&info.device_pci_bus_id);
        let mut by_name = None;
        let mut first = None;
        for i in 0..count {
            let Ok(d) = nvml.device_by_index(i) else {
                continue;
            };
            first.get_or_insert(i);
            if !info.device_pci_bus_id.is_empty()
                && let Ok(pci) = d.pci_info()
                && bus_key(&pci.bus_id) == want_bus
            {
                return Ok(Sensor {
                    backend: Backend::Nvml { nvml, index: i },
                    note: None,
                });
            }
            if by_name.is_none()
                && d.name()
                    .map(|n| info.name.contains(&n) || n.contains(&info.name))
                    .unwrap_or(false)
            {
                by_name = Some(i);
            }
        }
        let index = by_name
            .or(first)
            .ok_or_else(|| format!("NVML found {count} devices but none could be opened"))?;
        Ok(Sensor {
            backend: Backend::Nvml { nvml, index },
            note: None,
        })
    }

    #[cfg(target_os = "linux")]
    fn try_amd_sysfs(info: &wgpu::AdapterInfo) -> Option<Sensor> {
        let want_bus = bus_key(&info.device_pci_bus_id);
        let mut first = None;
        for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let dev = entry.path().join("device");
            let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
            if vendor.trim() != "0x1002" || !dev.join("mem_info_vram_total").exists() {
                continue;
            }
            let bus = std::fs::canonicalize(&dev)
                .ok()
                .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
                .unwrap_or_default();
            if !want_bus.is_empty() && bus_key(&bus) == want_bus {
                return Some(Sensor {
                    backend: Backend::AmdSysfs { dev },
                    note: None,
                });
            }
            first.get_or_insert(dev);
        }
        first.map(|dev| Sensor {
            backend: Backend::AmdSysfs { dev },
            note: None,
        })
    }

    pub fn available(&self) -> bool {
        !matches!(self.backend, Backend::None)
    }

    pub fn static_info(&self) -> StaticInfo {
        match &self.backend {
            Backend::Nvml { nvml, index } => {
                let mut extra = Vec::new();
                if let Ok(v) = nvml.sys_driver_version() {
                    extra.push(("Driver".into(), v));
                }
                if let Ok(d) = nvml.device_by_index(*index) {
                    if let Ok(m) = nvml_memory(&d) {
                        extra.push(("VRAM".into(), crate::gpu::fmt_bytes(m.total)));
                    }
                    if let Ok(cc) = d.cuda_compute_capability() {
                        extra.push((
                            "Compute capability".into(),
                            format!("{}.{}", cc.major, cc.minor),
                        ));
                    }
                    if let (Ok(g), Ok(m)) = (
                        d.max_clock_info(Clock::Graphics),
                        d.max_clock_info(Clock::Memory),
                    ) {
                        extra.push(("Max clocks".into(), format!("{g} MHz core / {m} MHz mem")));
                    }
                    if let Ok(p) = d.enforced_power_limit() {
                        extra.push(("Power limit".into(), format!("{:.0} W", p as f64 / 1000.0)));
                    }
                    if let (Ok(g), Ok(w)) = (d.current_pcie_link_gen(), d.current_pcie_link_width())
                    {
                        extra.push(("PCIe link".into(), format!("Gen{g} x{w}")));
                    }
                    if let Ok(pci) = d.pci_info() {
                        extra.push(("PCI bus".into(), pci.bus_id));
                    }
                }
                StaticInfo {
                    source: "NVML".into(),
                    extra,
                }
            }
            #[cfg(target_os = "linux")]
            Backend::AmdSysfs { dev } => {
                let mut extra = Vec::new();
                if let Some(t) = read_u64(&dev.join("mem_info_vram_total")) {
                    extra.push(("VRAM".into(), crate::gpu::fmt_bytes(t)));
                }
                if let Ok(w) = std::fs::read_to_string(dev.join("current_link_width")) {
                    let s =
                        std::fs::read_to_string(dev.join("current_link_speed")).unwrap_or_default();
                    extra.push(("PCIe link".into(), format!("{} x{}", s.trim(), w.trim())));
                }
                if let Ok(p) = std::fs::canonicalize(dev) {
                    extra.push(("sysfs".into(), p.display().to_string()));
                }
                StaticInfo {
                    source: "amdgpu sysfs".into(),
                    extra,
                }
            }
            Backend::None => StaticInfo {
                source: self
                    .note
                    .clone()
                    .unwrap_or_else(|| "none (NVML on NVIDIA, sysfs on Linux AMD)".into()),
                extra: vec![],
            },
        }
    }

    pub fn sample(&self, t: f64) -> Sample {
        let mut s = Sample {
            t,
            ..Default::default()
        };
        match &self.backend {
            Backend::Nvml { nvml, index } => {
                if let Ok(d) = nvml.device_by_index(*index) {
                    // Some legacy GeForce drivers answer 0 instead of "not supported";
                    // a running GPU is never at 0 °C.
                    s.temp_c = d
                        .temperature(TemperatureSensor::Gpu)
                        .ok()
                        .filter(|&v| v > 0)
                        .map(|v| v as f64);
                    s.power_w = d.power_usage().ok().map(|mw| mw as f64 / 1000.0);
                    s.power_limit_w = d.enforced_power_limit().ok().map(|mw| mw as f64 / 1000.0);
                    if let Ok(u) = d.utilization_rates() {
                        s.util_pct = Some(u.gpu as f64);
                        s.mem_util_pct = Some(u.memory as f64);
                    }
                    if let Ok(m) = nvml_memory(&d) {
                        s.vram_used = Some(m.used);
                        s.vram_total = Some(m.total);
                    }
                    s.core_mhz = d.clock_info(Clock::Graphics).ok().map(|v| v as f64);
                    s.mem_mhz = d.clock_info(Clock::Memory).ok().map(|v| v as f64);
                    s.fan_pct = d.fan_speed(0).ok().map(|v| v as f64);
                }
            }
            #[cfg(target_os = "linux")]
            Backend::AmdSysfs { dev } => {
                s.vram_used = read_u64(&dev.join("mem_info_vram_used"));
                s.vram_total = read_u64(&dev.join("mem_info_vram_total"));
                s.util_pct = read_u64(&dev.join("gpu_busy_percent")).map(|v| v as f64);
                s.mem_util_pct = read_u64(&dev.join("mem_busy_percent")).map(|v| v as f64);
                s.core_mhz = read_active_clock(&dev.join("pp_dpm_sclk"));
                s.mem_mhz = read_active_clock(&dev.join("pp_dpm_mclk"));
                if let Some(hw) = hwmon_dir(dev) {
                    s.temp_c = read_u64(&hw.join("temp1_input")).map(|v| v as f64 / 1000.0);
                    s.power_w = read_u64(&hw.join("power1_average"))
                        .or_else(|| read_u64(&hw.join("power1_input")))
                        .map(|uw| uw as f64 / 1e6);
                    s.power_limit_w = read_u64(&hw.join("power1_cap")).map(|uw| uw as f64 / 1e6);
                    if let (Some(p), Some(m)) =
                        (read_u64(&hw.join("pwm1")), read_u64(&hw.join("pwm1_max")))
                        && m > 0
                    {
                        s.fan_pct = Some(p as f64 * 100.0 / m as f64);
                    }
                }
            }
            Backend::None => {}
        }
        s
    }

    /// One line per reading: its value, or the driver's error for it. Logged
    /// when an adapter is selected so missing sensors can be diagnosed.
    pub fn diagnose(&self) -> Vec<String> {
        fn show<T: std::fmt::Debug, E: std::fmt::Display>(name: &str, r: Result<T, E>) -> String {
            match r {
                Ok(v) => format!("{name}: {v:?}"),
                Err(e) => format!("{name}: unavailable ({e})"),
            }
        }
        match &self.backend {
            Backend::Nvml { nvml, index } => {
                let mut v = vec![show("driver", nvml.sys_driver_version())];
                match nvml.device_by_index(*index) {
                    Ok(d) => {
                        v.push(show("name", d.name()));
                        v.push(show("temperature", d.temperature(TemperatureSensor::Gpu)));
                        v.push(show("power (mW)", d.power_usage()));
                        v.push(show("power limit (mW)", d.enforced_power_limit()));
                        v.push(show(
                            "utilization",
                            d.utilization_rates().map(|u| (u.gpu, u.memory)),
                        ));
                        v.push(show("memory", nvml_memory(&d).map(|m| (m.used, m.total))));
                        v.push(show("core clock", d.clock_info(Clock::Graphics)));
                        v.push(show("memory clock", d.clock_info(Clock::Memory)));
                        v.push(show("fan", d.fan_speed(0)));
                    }
                    Err(e) => v.push(format!("device #{index}: unavailable ({e})")),
                }
                v
            }
            #[cfg(target_os = "linux")]
            Backend::AmdSysfs { dev } => vec![format!("amdgpu sysfs at {}", dev.display())],
            Backend::None => vec![self.static_info().source],
        }
    }

    /// Free VRAM in bytes, if known — used to size the VRAM test.
    pub fn vram_free(&self) -> Option<u64> {
        let s = self.sample(0.0);
        Some(s.vram_total?.saturating_sub(s.vram_used?))
    }
}

#[cfg(target_os = "linux")]
fn read_u64(p: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn hwmon_dir(dev: &std::path::Path) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dev.join("hwmon"))
        .ok()?
        .flatten()
        .next()
        .map(|e| e.path())
}

/// pp_dpm_* lists levels like "1: 1800Mhz *"; the starred one is active.
#[cfg(target_os = "linux")]
fn read_active_clock(p: &std::path::Path) -> Option<f64> {
    let text = std::fs::read_to_string(p).ok()?;
    let line = text.lines().find(|l| l.trim_end().ends_with('*'))?;
    let mhz = line.split_whitespace().nth(1)?.to_ascii_lowercase();
    mhz.trim_end_matches("mhz").parse().ok()
}

/// Seconds since the program started; shared time base for every plot.
pub fn clock() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Rolling telemetry history shared between the sampler thread and the UI.
pub type History = Arc<Mutex<Vec<Sample>>>;

/// Samples every `period` until `stop` is set. Keeps the last ~10 minutes.
pub fn spawn_sampler(
    sensor: Sensor,
    history: History,
    period: Duration,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let max_len = (600.0 / period.as_secs_f64()) as usize;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            let s = sensor.sample(clock());
            if let Ok(mut h) = history.lock() {
                h.push(s);
                if h.len() > max_len {
                    let excess = h.len() - max_len;
                    h.drain(..excess);
                }
            }
            std::thread::sleep(period);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_ids_match_across_formats() {
        assert_eq!(bus_key("00000000:01:00.0"), bus_key("0000:01:00.0"));
        assert_eq!(bus_key("0000:0A:00.0"), "0a:00.0");
        assert_ne!(bus_key("0000:01:00.0"), bus_key("0000:02:00.0"));
    }
}
