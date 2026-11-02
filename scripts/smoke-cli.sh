#!/usr/bin/env bash
# Headless smoke test: runs every benchmark through the CLI on software Vulkan
# (lavapipe) and software OpenGL (llvmpipe). Fails if the binary exits non-zero,
# a correctness check reports errors, or the session log is missing.
#
# Usage: scripts/smoke-cli.sh [path/to/gpu-workbench-cli]
# Needs mesa-vulkan-drivers and libgl1-mesa-dri (see .github/workflows/build.yml).
set -euo pipefail

BIN=${1:-target/release/gpu-workbench-cli}

# Keep session logs out of the real home directory.
XDG_STATE_HOME=$(mktemp -d)
export XDG_STATE_HOME

fail() {
  echo "SMOKE FAIL: $*" >&2
  exit 1
}

echo "== --list"
listing=$("$BIN" --list)
echo "$listing"
grep -q "Adapters:" <<<"$listing" || fail "--list printed no adapter list"

TESTS=upload,readback,copy,bandwidth,matmul32,matmul16,vram,stress
ARGS=(--cli --tests "$TESTS" --transfer-mb 64 --matmul-n 256 --vram-mb 128
  --vram-passes 1 --stress-secs 5 --stress-n 256)

for backend in vulkan gl; do
  echo "== CLI, WGPU_BACKEND=$backend"
  out=$(mktemp)
  # With pipefail, a non-zero exit from the binary fails the script here.
  WGPU_BACKEND=$backend "$BIN" "${ARGS[@]}" | tee "$out"
  grep -q "check 0/256 bad" "$out" || fail "$backend: matmul correctness line missing or failing"
  grep -q ", 0 errors" "$out" || fail "$backend: VRAM verify line missing or failing"
  if grep -q "\[FAIL\]" "$out"; then
    fail "$backend: a test reported FAIL"
  fi
done

# Every run writes a session log; the CLI run must have recorded its matmul.
logs=$(find "$XDG_STATE_HOME" -name 'gpu-workbench_*_cli.log')
[ -n "$logs" ] || fail "no CLI session log was written"
grep -q "Matmul N=256" $logs || fail "session log does not record the matmul run"

echo "CLI smoke OK"
