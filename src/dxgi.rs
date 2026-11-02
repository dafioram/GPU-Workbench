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

/// Finds the DXGI adapter matching `info` (by PCI vendor + device ID) and
/// reads its memory counters. None on other platforms or if no match.
#[cfg(windows)]
pub fn query(info: &wgpu::AdapterInfo) -> Option<VideoMemory> {
    imp::query(info).ok().flatten()
}

#[cfg(not(windows))]
pub fn query(_info: &wgpu::AdapterInfo) -> Option<VideoMemory> {
    None
}

#[cfg(windows)]
mod imp {
    use super::VideoMemory;
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
            let mut exact: Option<IDXGIAdapter1> = None;
            let mut same_vendor: Vec<IDXGIAdapter1> = Vec::new();
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
                if desc.VendorId == info.vendor && desc.DeviceId == info.device {
                    exact = Some(adapter);
                    break;
                }
                if desc.VendorId == info.vendor {
                    same_vendor.push(adapter);
                }
            }
            // OpenGL adapters often have no device ID; accept a vendor match
            // only when it is unambiguous.
            let adapter = match exact {
                Some(a) => a,
                None if same_vendor.len() == 1 => same_vendor.remove(0),
                None => return Ok(None),
            };
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
