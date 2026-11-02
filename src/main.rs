// Hide the console window for the GUI build on Windows (release only, so
// `cargo run` still shows logs). The CLI re-attaches to the parent console.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod bench;
mod cli;
mod dxgi;
mod gpu;
mod logfile;
mod sensors;
mod shaders;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        #[cfg(windows)]
        attach_console();
        if !args.iter().any(|a| a == "-h" || a == "--help") {
            logfile::init("cli");
        }
        std::process::exit(cli::main(&args));
    }
    logfile::init("gui");
    if let Err(e) = app::run() {
        logfile::line(&format!("GUI failed to start: {e}"));
        eprintln!("GUI failed to start: {e}\nTry the headless mode: gpu-workbench --cli");
        std::process::exit(1);
    }
}

/// With windows_subsystem = "windows" there is no console; attach to the
/// terminal that launched us so CLI output is visible.
#[cfg(windows)]
fn attach_console() {
    unsafe extern "system" {
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        AttachConsole(u32::MAX); // ATTACH_PARENT_PROCESS
    }
}
