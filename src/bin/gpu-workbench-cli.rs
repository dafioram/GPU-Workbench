//! Console build of the command-line interface, for PowerShell, cmd and scripts.
//!
//! It is a console program, so the shell waits for it and `$LASTEXITCODE` (or
//! `%ERRORLEVEL%`) carries the result. `gpu-workbench.exe` is a GUI program on
//! Windows, and shells don't wait for those.

use gpu_workbench::{cli, logfile};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a == "-h" || a == "--help") {
        logfile::init("cli");
    }
    std::process::exit(cli::main(&args));
}
