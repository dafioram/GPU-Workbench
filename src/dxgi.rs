//! Windows only: video memory budget and usage from DXGI.
//!
//! `Budget` is how much VRAM Windows will let *this process* keep resident
//! before it starts moving its memory to system RAM. That is the number the
//! VRAM test should stay under, and it is more accurate than "free VRAM",
//! which ignores what Windows reserves. Usage is tracked per process across
//! APIs, so it also covers memory allocated through Vulkan.

#[derive(Clone, Copy, Debug)]
pub struct VideoMemory {
    /// Per-process VRAM budget set by Windows (bytes).
    pub budget: u64,
    /// VRAM this process is using now (bytes).
    pub local_usage: u64,
    /// System RAM this process's GPU allocations occupy (bytes).
    pub nonlocal_usage: u64,
    /// Total dedicated VRAM on the card (bytes).
    pub dedicated: u64,
}

impl VideoMemory {
    /// What this process can still allocate before Windows starts paging.
    pub fn available(&self) -> u64 {
        self.budget.saturating_sub(self.local_usage)
    }
}

/// One hardware adapter as DXGI describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub vendor: u32,
    pub device: u32,
    pub name: String,
}

/// Picks the DXGI adapter that matches a wgpu adapter, or None when it can't
/// tell which one it is.
///
/// 1. The same PCI vendor and device id. Vulkan and DX12 always report these.
/// 2. Otherwise the only adapter with the same name (case and spacing ignored).
///    OpenGL can report no device id, and sometimes no vendor id either.
/// 3. Otherwise the only adapter from the same vendor, if the vendor is known.
///
/// Anything ambiguous gives None rather than a guess: a budget from the wrong
/// card is worse than no budget.
pub fn pick_match(cands: &[Candidate], vendor: u32, device: u32, name: &str) -> Option<usize> {
    if vendor != 0
        && let Some(i) = cands
            .iter()
            .position(|c| c.vendor == vendor && c.device == device)
    {
        return Some(i);
    }
    let want = normalize(name);
    let by_name: Vec<usize> = (0..cands.len())
        .filter(|&i| !want.is_empty() && normalize(&cands[i].name) == want)
        .collect();
    if let [only] = by_name[..] {
        return Some(only);
    }
    if vendor != 0 {
        let same_vendor: Vec<usize> = (0..cands.len())
            .filter(|&i| cands[i].vendor == vendor)
            .collect();
        if let [only] = same_vendor[..] {
            return Some(only);
        }
    }
    None
}

/// Lowercases and collapses runs of spaces, so "AMD  Radeon" equals "amd radeon".
fn normalize(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// DXGI's adapter description: UTF-16, NUL-terminated, in a fixed-size buffer.
pub fn wide_name(units: &[u16]) -> String {
    let end = units.iter().position(|&c| c == 0).unwrap_or(units.len());
    String::from_utf16_lossy(&units[..end])
}

/// Finds the DXGI adapter matching `info` and reads its memory counters. None
/// on other platforms, or when no adapter can be matched (the log says why).
#[cfg(windows)]
pub fn query(info: &wgpu::AdapterInfo) -> Option<VideoMemory> {
    match imp::query(info) {
        Ok(m) => m,
        Err(e) => {
            crate::logfile::line(&format!("DXGI query failed: {e}"));
            None
        }
    }
}

#[cfg(not(windows))]
pub fn query(_info: &wgpu::AdapterInfo) -> Option<VideoMemory> {
    None
}

#[cfg(windows)]
mod imp {
    use super::{Candidate, VideoMemory, pick_match, wide_name};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory1,
    };
    use windows::core::Interface;

    pub fn query(info: &wgpu::AdapterInfo) -> windows::core::Result<Option<VideoMemory>> {
        // SAFETY: plain COM calls on interfaces we own; out-structs are
        // zero-initialised and sized by the bindings.
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
            // Hardware adapters only, in DXGI order: `adapters[i]` is `cands[i]`.
            let mut adapters: Vec<IDXGIAdapter1> = Vec::new();
            let mut cands: Vec<Candidate> = Vec::new();
            let mut i = 0;
            loop {
                let adapter = match factory.EnumAdapters1(i) {
                    Ok(a) => a,
                    Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                    Err(e) => return Err(e),
                };
                i += 1;
                let desc = adapter.GetDesc1()?;
                if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                    continue;
                }
                cands.push(Candidate {
                    vendor: desc.VendorId,
                    device: desc.DeviceId,
                    name: wide_name(&desc.Description),
                });
                adapters.push(adapter);
            }
            let Some(idx) = pick_match(&cands, info.vendor, info.device, &info.name) else {
                let listed: Vec<String> = cands
                    .iter()
                    .map(|c| format!("'{}' ({:04x}:{:04x})", c.name, c.vendor, c.device))
                    .collect();
                crate::logfile::line(&format!(
                    "DXGI: no adapter matches '{}' ({:04x}:{:04x}); DXGI reports {}",
                    info.name,
                    info.vendor,
                    info.device,
                    listed.join(", ")
                ));
                return Ok(None);
            };
            let adapter = &adapters[idx];
            let desc = adapter.GetDesc1()?;
            let a3: IDXGIAdapter3 = adapter.cast()?;
            let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)?;
            let mut nonlocal = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut nonlocal)?;
            Ok(Some(VideoMemory {
                budget: local.Budget,
                local_usage: local.CurrentUsage,
                nonlocal_usage: nonlocal.CurrentUsage,
                dedicated: desc.DedicatedVideoMemory as u64,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(vendor: u32, device: u32, name: &str) -> Candidate {
        Candidate {
            vendor,
            device,
            name: name.into(),
        }
    }

    /// A machine with a Radeon and Microsoft's software renderer (which
    /// `query` drops before matching, so it is not in real lists).
    fn list() -> Vec<Candidate> {
        vec![
            cand(0x1002, 0x73df, "AMD Radeon RX 6750 XT"),
            cand(0x1414, 0x008c, "Microsoft Basic Render Driver"),
        ]
    }

    #[test]
    fn exact_pci_ids_win() {
        assert_eq!(
            pick_match(&list(), 0x1002, 0x73df, "AMD Radeon RX 6750 XT"),
            Some(0)
        );
    }

    #[test]
    fn opengl_without_ids_matches_by_name() {
        assert_eq!(pick_match(&list(), 0, 0, "AMD Radeon RX 6750 XT"), Some(0));
    }

    #[test]
    fn name_match_ignores_case_and_spacing() {
        assert_eq!(
            pick_match(&list(), 0x1002, 0, "amd  radeon rx 6750 xt"),
            Some(0)
        );
    }

    #[test]
    fn ambiguous_names_match_nothing() {
        let two = vec![
            cand(0x1002, 0x73df, "AMD Radeon RX 6750 XT"),
            cand(0x1002, 0x73df, "AMD Radeon RX 6750 XT"),
        ];
        assert_eq!(pick_match(&two, 0, 0, "AMD Radeon RX 6750 XT"), None);
    }

    #[test]
    fn a_unique_vendor_is_the_last_resort() {
        assert_eq!(pick_match(&list(), 0x1002, 0x1234, "Radeon"), Some(0));
    }

    #[test]
    fn unrelated_adapters_match_nothing() {
        assert_eq!(pick_match(&list(), 0x8086, 0x1234, "Intel UHD"), None);
    }

    #[test]
    fn wide_names_stop_at_the_terminator() {
        let w: Vec<u16> = "AMD Radeon"
            .encode_utf16()
            .chain([0u16, 0x41, 0x42])
            .collect();
        assert_eq!(wide_name(&w), "AMD Radeon");
    }
}
