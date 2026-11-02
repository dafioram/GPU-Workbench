//! WGSL compute kernels used by the benchmarks.
//!
//! Every 1-D kernel uses a 256-wide workgroup and a 2-D dispatch so it can
//! cover buffers larger than `65535 * 256` elements; the linear index is
//! rebuilt from `num_workgroups`.

pub const WG: u32 = 256;

/// Streams `src` into `dst` as vec4<f32> (16 bytes per thread).
pub const COPY: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i < arrayLength(&dst)) {
        dst[i] = src[i];
    }
}
"#;

/// Writes a seeded hash pattern into VRAM, then checks it back.
/// `errs[0]` counts mismatching words, `errs[1]` keeps the first bad index.
pub const PATTERN: &str = r#"
struct Params { seed: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> errs: array<atomic<u32>, 2>;

fn pattern(i: u32, seed: u32) -> u32 {
    var x = (i * 0x9E3779B9u) ^ seed;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i < arrayLength(&buf)) {
        buf[i] = pattern(i, p.seed);
    }
}

@compute @workgroup_size(256)
fn verify(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i < arrayLength(&buf)) {
        if (buf[i] != pattern(i, p.seed)) {
            atomicAdd(&errs[0], 1u);
            atomicMin(&errs[1], i);
        }
    }
}
"#;

/// Tiled matrix multiply C = A * B for square N x N matrices (N % 64 == 0).
/// 16x16 threads, each computing a 4x4 block -> 64x64 output tile per workgroup.
///
/// The 4x4 accumulator is held in four vec4s and only ever indexed with
/// constant swizzles (.x/.y/.z/.w). Indexing a private array with a loop
/// variable makes some Vulkan drivers (e.g. NVIDIA Kepler) spill it to slow
/// local memory, which made this kernel many times slower than on DX12.
///
/// `row_off` lets one matmul be split across several dispatches (bands of tile
/// rows) so no single dispatch runs long enough to trip the GPU watchdog.
/// `SCALAR` is replaced with f32 or f16; `ENABLE` with the f16 extension line.
const MATMUL_TEMPLATE: &str = r#"
ENABLE
struct Dims { n: u32, row_off: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<storage, read> a: array<SCALAR>;
@group(0) @binding(1) var<storage, read> b: array<SCALAR>;
@group(0) @binding(2) var<storage, read_write> c: array<SCALAR>;
@group(0) @binding(3) var<uniform> d: Dims;

const TS: u32 = 64u;
const TK: u32 = 16u;
var<workgroup> As: array<SCALAR, 1024>; // [k][row], TK x TS
var<workgroup> Bs: array<SCALAR, 1024>; // [k][col], TK x TS

fn store4(base: u32, v: vec4<SCALAR>) {
    c[base] = v.x;
    c[base + 16u] = v.y;
    c[base + 32u] = v.z;
    c[base + 48u] = v.w;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let n = d.n;
    let row0 = (wid.y + d.row_off) * TS;
    let col0 = wid.x * TS;
    let tid = lid.y * 16u + lid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc0 = vec4<SCALAR>(SCALAR(0));
    var acc1 = vec4<SCALAR>(SCALAR(0));
    var acc2 = vec4<SCALAR>(SCALAR(0));
    var acc3 = vec4<SCALAR>(SCALAR(0));

    for (var k0 = 0u; k0 < n; k0 += TK) {
        for (var e = 0u; e < 4u; e++) {
            let idx = tid + e * 256u;
            let ar = idx / TK;
            let ak = idx % TK;
            As[ak * TS + ar] = a[(row0 + ar) * n + k0 + ak];
            let bk = idx / TS;
            let bc = idx % TS;
            Bs[bk * TS + bc] = b[(k0 + bk) * n + col0 + bc];
        }
        workgroupBarrier();

        for (var k = 0u; k < TK; k++) {
            let ao = k * TS + ty;
            let bo = k * TS + tx;
            let av = vec4<SCALAR>(As[ao], As[ao + 16u], As[ao + 32u], As[ao + 48u]);
            let bv = vec4<SCALAR>(Bs[bo], Bs[bo + 16u], Bs[bo + 32u], Bs[bo + 48u]);
            acc0 = fma(vec4<SCALAR>(av.x), bv, acc0);
            acc1 = fma(vec4<SCALAR>(av.y), bv, acc1);
            acc2 = fma(vec4<SCALAR>(av.z), bv, acc2);
            acc3 = fma(vec4<SCALAR>(av.w), bv, acc3);
        }
        workgroupBarrier();
    }

    let base = (row0 + ty) * n + col0 + tx;
    store4(base, acc0);
    store4(base + 16u * n, acc1);
    store4(base + 32u * n, acc2);
    store4(base + 48u * n, acc3);
}
"#;

pub fn matmul(f16: bool) -> String {
    if f16 {
        MATMUL_TEMPLATE
            .replace("ENABLE", "enable f16;")
            .replace("SCALAR", "f16")
    } else {
        MATMUL_TEMPLATE
            .replace("ENABLE", "")
            .replace("SCALAR", "f32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};

    /// Parses and validates WGSL with naga, the same front end wgpu uses, so
    /// syntax and type errors are caught without a GPU.
    fn check(src: &str, caps: Capabilities) {
        let module = wgpu::naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        Validator::new(ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    }

    #[test]
    fn kernels_validate() {
        check(COPY, Capabilities::empty());
        check(PATTERN, Capabilities::empty());
        check(&matmul(false), Capabilities::empty());
    }

    #[test]
    fn f16_matmul_validates_with_f16_capability() {
        check(&matmul(true), Capabilities::SHADER_FLOAT16);
    }
}
