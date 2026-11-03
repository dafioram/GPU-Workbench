//! Adapter discovery and device creation.

use std::sync::Arc;

/// Everything we show about an adapter, captured once at startup.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub index: usize,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    pub downlevel: wgpu::DownlevelCapabilities,
}

impl AdapterSummary {
    /// Compute shaders are always there on Vulkan/DX12/Metal; on OpenGL they
    /// need GL 4.3 (or GLES 3.1), which older drivers may not expose.
    pub fn has_compute(&self) -> bool {
        compute_ok(&self.info, &self.downlevel)
    }

    /// wgpu can't tell discrete from unknown on OpenGL and reports "Other";
    /// treat known desktop GPU brands as discrete so they're preferred.
    pub fn effective_type(&self) -> wgpu::DeviceType {
        let t = self.info.device_type;
        if t == wgpu::DeviceType::Other && self.is_gl() && looks_discrete(&self.info) {
            wgpu::DeviceType::DiscreteGpu
        } else {
            t
        }
    }

    pub fn is_gl(&self) -> bool {
        self.info.backend == wgpu::Backend::Gl
    }

    pub fn label(&self) -> String {
        format!(
            "#{} {} ({}, {}{})",
            self.index,
            self.info.name,
            backend_str(self.info.backend),
            device_type_str(self.effective_type()),
            if self.has_compute() {
                ""
            } else {
                ", no compute"
            }
        )
    }

    pub fn vendor_name(&self) -> &'static str {
        vendor_name(self.info.vendor)
    }
}

pub fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1414 => "Microsoft",
        0x10005 => "Mesa",
        _ => "Unknown",
    }
}

/// Desktop GL version from the driver string, e.g. "4.6.0 NVIDIA 470.x" -> (4, 6).
/// Returns None for GLES or unparsable strings.
pub fn gl_version(info: &wgpu::AdapterInfo) -> Option<(u32, u32)> {
    parse_gl_version(&format!("{} {}", info.driver, info.driver_info))
}

fn parse_gl_version(text: &str) -> Option<(u32, u32)> {
    if text.contains("OpenGL ES") {
        return None;
    }
    let ver = text.split_whitespace().find(|w| w.contains('.'))?;
    let mut it = ver.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it
        .next()?
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some((major, minor))
}

/// wgpu's compute flag, plus a GL >= 4.3 check: some drivers expose the
/// compute extension on older GL versions but can't run wgpu's GL path.
pub fn compute_ok(info: &wgpu::AdapterInfo, downlevel: &wgpu::DownlevelCapabilities) -> bool {
    if !downlevel
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        return false;
    }
    if info.backend == wgpu::Backend::Gl
        && let Some(v) = gl_version(info)
    {
        return v >= (4, 3);
    }
    true
}

pub fn is_nvidia(info: &wgpu::AdapterInfo) -> bool {
    let n = info.name.to_lowercase();
    info.vendor == 0x10DE
        || ["nvidia", "geforce", "quadro"]
            .iter()
            .any(|k| n.contains(k))
}

/// Old AMD/ATI GL drivers report "ATI Technologies", which wgpu doesn't map.
pub fn is_amd(info: &wgpu::AdapterInfo) -> bool {
    let n = info.name.to_lowercase();
    info.vendor == 0x1002 || ["radeon", "amd", "firepro"].iter().any(|k| n.contains(k))
}

fn looks_discrete(info: &wgpu::AdapterInfo) -> bool {
    let n = info.name.to_lowercase();
    let apu = [
        "radeon graphics",
        "vega 3",
        "vega 6",
        "vega 8",
        "vega 11",
        "radeon(tm) r",
    ]
    .iter()
    .any(|k| n.contains(k));
    (is_nvidia(info) || is_amd(info)) && !apu
}

pub fn backend_str(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Gl => "OpenGL",
        _ => "other",
    }
}

pub fn device_type_str(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "CPU/software",
        wgpu::DeviceType::Other => "other",
    }
}

/// Vulkan + DX12 (+ Metal) first, with OpenGL as the fallback for GPUs too old
/// for Vulkan/DX12. GL compute needs OpenGL 4.3. Override with
/// WGPU_BACKEND=vulkan,dx12,gl.
pub fn backends() -> wgpu::Backends {
    wgpu::Backends::from_env().unwrap_or(
        wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL | wgpu::Backends::GL,
    )
}

pub fn instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = backends();
    wgpu::Instance::new(desc)
}

/// Lists adapters. The same physical GPU can appear once per backend
/// (e.g. Vulkan and DX12 on Windows); each is listed so they can be compared.
///
/// Runs on its own thread: the OpenGL backend makes a GL context current while
/// probing, which clashes with the UI's own GL context on the main thread.
/// A panic while probing (buggy legacy drivers) yields an empty list.
pub fn enumerate() -> Vec<AdapterSummary> {
    std::thread::spawn(enumerate_here)
        .join()
        .unwrap_or_else(|_| {
            eprintln!("adapter enumeration crashed; check GPU drivers or set WGPU_BACKEND");
            Vec::new()
        })
}

fn enumerate_here() -> Vec<AdapterSummary> {
    let instance = instance();
    let adapters = pollster::block_on(instance.enumerate_adapters(backends()));
    adapters
        .into_iter()
        .enumerate()
        .map(|(index, a)| AdapterSummary {
            index,
            info: a.get_info(),
            limits: a.limits(),
            features: a.features(),
            downlevel: a.get_downlevel_capabilities(),
        })
        .collect()
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub has_f16: bool,
    /// Set by the driver's device-lost callback (e.g. after a watchdog reset).
    pub lost: Arc<std::sync::atomic::AtomicBool>,
}

/// Opens a device on adapter `index` with the adapter's full limits so the
/// benchmarks can use large buffers. Uncaptured errors and device loss are
/// forwarded to `log` instead of panicking.
pub fn open<L>(index: usize, log: L) -> Result<Gpu, String>
where
    L: Fn(String) + Send + Sync + Clone + 'static,
{
    let instance = instance();
    let adapter = pollster::block_on(instance.enumerate_adapters(backends()))
        .into_iter()
        .nth(index)
        .ok_or_else(|| format!("adapter #{index} not found"))?;

    let info = adapter.get_info();
    if info.backend == wgpu::Backend::Dx12
        && let Some(note) = dx12_compiler_note()
    {
        log(note);
    }
    if !compute_ok(&info, &adapter.get_downlevel_capabilities()) {
        return Err(format!(
            "{} has no compute shader support over {} (driver reports \"{}\"). \
             The OpenGL path needs GL 4.3+; a newer driver may help, otherwise this card \
             would need an OpenCL backend.",
            info.name,
            backend_str(info.backend),
            info.driver_info.trim()
        ));
    }
    let limits = adapter.limits();
    let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
    let mut features = wgpu::Features::empty();
    if has_f16 {
        features |= wgpu::Features::SHADER_F16;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("workbench"),
        required_features: features,
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .map_err(|e| {
        let mut msg = format!("request_device failed: {e}");
        if info.backend == wgpu::Backend::Gl {
            msg += &format!(
                " — OpenGL driver reports \"{}\"; the GL path needs 4.3+ with compute shaders",
                info.driver_info.trim()
            );
        }
        msg
    })?;

    let l1 = log.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        l1(format!("GPU error: {e}"));
    }));
    let lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let lost2 = lost.clone();
    device.set_device_lost_callback(move |reason, msg| {
        lost2.store(true, std::sync::atomic::Ordering::Relaxed);
        log(format!("DEVICE LOST ({reason:?}): {msg}"));
    });

    Ok(Gpu {
        device,
        queue,
        info,
        limits,
        has_f16,
        lost,
    })
}

/// Which compiler the DX12 backend will use for shaders, for the log. wgpu's
/// default ("Auto") loads dxcompiler.dll and falls back to FXC without saying
/// so. This makes the same check and reports the result.
#[cfg(windows)]
pub fn dx12_compiler_note() -> Option<String> {
    if let Ok(v) = std::env::var("WGPU_DX12_COMPILER")
        && !v.eq_ignore_ascii_case("auto")
    {
        return Some(format!(
            "DX12 shader compiler forced by WGPU_DX12_COMPILER={v}"
        ));
    }
    Some(if dxc_loadable() {
        "DX12 shaders: DXC (dxcompiler.dll found)".into()
    } else {
        "DX12 shaders: dxcompiler.dll not found, falling back to FXC (slower). \
         Put dxcompiler.dll next to the program; see the README."
            .into()
    })
}

#[cfg(not(windows))]
pub fn dx12_compiler_note() -> Option<String> {
    None
}

/// Whether wgpu can load dxcompiler.dll. Uses the same name and search order,
/// so the answer matches what wgpu will do.
#[cfg(windows)]
fn dxc_loadable() -> bool {
    use windows::Win32::Foundation::FreeLibrary;
    use windows::Win32::System::LibraryLoader::LoadLibraryW;
    use windows::core::w;
    // SAFETY: loads a DLL by name and frees it again straight away.
    unsafe {
        match LoadLibraryW(w!("dxcompiler.dll")) {
            Ok(module) => {
                let _ = FreeLibrary(module);
                true
            }
            Err(_) => false,
        }
    }
}

pub fn fmt_bytes(b: u64) -> String {
    const K: f64 = 1024.0;
    let b = b as f64;
    if b >= K * K * K {
        format!("{:.2} GiB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.1} MiB", b / (K * K))
    } else if b >= K {
        format!("{:.1} KiB", b / K)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_versions() {
        assert_eq!(parse_gl_version(" 4.6.0 NVIDIA 470.256.02"), Some((4, 6)));
        assert_eq!(
            parse_gl_version(" 4.5 (Core Profile) Mesa 25.2.8"),
            Some((4, 5))
        );
        assert_eq!(parse_gl_version("4.1 ATI-4.6.21"), Some((4, 1)));
        assert_eq!(parse_gl_version("OpenGL ES 3.2 Mesa"), None);
        assert_eq!(parse_gl_version("no version here"), None);
    }

    #[test]
    fn bytes() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(256 * 1024 * 1024), "256.0 MiB");
        assert_eq!(fmt_bytes(2 * 1024 * 1024 * 1024), "2.00 GiB");
    }
}
