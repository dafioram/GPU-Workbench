#!/usr/bin/env bash
# GUI smoke test: starts the window under a virtual X display, checks it is
# still running after a few seconds, and checks the session log for panics.
# Runs once with the glow (OpenGL) renderer and once with the wgpu renderer.
#
# Usage: xvfb-run -a scripts/smoke-gui.sh [path/to/gpu-workbench]
set -euo pipefail

BIN=${1:-target/release/gpu-workbench}
: "${DISPLAY:?run under xvfb-run, e.g. xvfb-run -a scripts/smoke-gui.sh}"

XDG_STATE_HOME=$(mktemp -d)
export XDG_STATE_HOME
out=$(mktemp)

fail() {
  echo "SMOKE FAIL: $*" >&2
  cat "$out" >&2
  exit 1
}

for renderer in glow wgpu; do
  echo "== GUI, WORKBENCH_RENDERER=$renderer"
  WORKBENCH_RENDERER=$renderer "$BIN" >"$out" 2>&1 &
  pid=$!
  sleep 15
  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid" || true
    fail "GUI exited early with $renderer renderer"
  fi
  kill "$pid"
  wait "$pid" || true
  if grep -rq "PANIC" "$XDG_STATE_HOME"; then
    grep -rh "PANIC" "$XDG_STATE_HOME" >&2
    fail "GUI logged a panic with $renderer renderer"
  fi
  echo "still running after 15 s, no panic"
done

echo "GUI smoke OK"
