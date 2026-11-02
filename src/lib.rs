//! GPU Workbench: adapter discovery, benchmarks, telemetry, and the GUI and CLI
//! front ends. Two binaries use this library:
//!
//! - `gpu-workbench`: the GUI (on Windows a GUI-subsystem program, so it has no
//!   console; `--cli` still works when run from a terminal).
//! - `gpu-workbench-cli`: a console program for PowerShell, cmd and scripts. The
//!   shell waits for it and receives its exit code.

pub mod app;
pub mod bench;
pub mod cli;
pub mod dxgi;
pub mod gpu;
pub mod logfile;
pub mod sensors;
pub mod shaders;
