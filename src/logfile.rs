//! Session log written to disk, one file per launch.
//!
//! Location:
//!   Windows: %LOCALAPPDATA%\gpu-workbench\logs
//!   Linux:   $XDG_STATE_HOME/gpu-workbench/logs (default ~/.local/state/...)
//! Falls back to a `logs` folder next to the executable, then the temp dir.
//! Every line is flushed immediately so a driver crash that kills the process
//! still leaves everything up to that point on disk. The newest 30 files are kept.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

struct Log {
    file: Option<File>,
    path: Option<PathBuf>,
}

static LOG: OnceLock<Mutex<Log>> = OnceLock::new();
const KEEP: usize = 30;

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    #[cfg(windows)]
    if let Some(d) = std::env::var_os("LOCALAPPDATA") {
        v.push(PathBuf::from(d).join("gpu-workbench").join("logs"));
    }
    #[cfg(not(windows))]
    {
        if let Some(d) = std::env::var_os("XDG_STATE_HOME") {
            v.push(PathBuf::from(d).join("gpu-workbench").join("logs"));
        } else if let Some(h) = std::env::var_os("HOME") {
            v.push(PathBuf::from(h).join(".local/state/gpu-workbench/logs"));
        }
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        v.push(dir.join("logs"));
    }
    v.push(std::env::temp_dir().join("gpu-workbench-logs"));
    v
}

/// Opens the session log. `mode` ("gui" / "cli") goes in the file name.
/// Safe to call more than once; only the first call does anything.
pub fn init(mode: &str) -> Option<PathBuf> {
    let log = LOG.get_or_init(|| {
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        for dir in candidates() {
            if fs::create_dir_all(&dir).is_err() {
                continue;
            }
            let path = dir.join(format!("gpu-workbench_{stamp}_{mode}.log"));
            if let Ok(file) = File::create(&path) {
                prune(&dir);
                return Mutex::new(Log {
                    file: Some(file),
                    path: Some(path),
                });
            }
        }
        Mutex::new(Log {
            file: None,
            path: None,
        })
    });
    let path = log.lock().ok().and_then(|l| l.path.clone());
    line(&format!(
        "gpu-workbench {} ({} {}) started",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    install_panic_hook();
    path
}

pub fn path() -> Option<PathBuf> {
    LOG.get()?.lock().ok()?.path.clone()
}

/// Appends one timestamped line (multi-line text is indented).
pub fn line(text: &str) {
    let Some(log) = LOG.get() else { return };
    let Ok(mut log) = log.lock() else { return };
    let Some(f) = log.file.as_mut() else { return };
    let t = chrono::Local::now().format("%H:%M:%S%.3f");
    let body = text.trim_end().replace('\n', "\n             ");
    let _ = writeln!(f, "[{t}] {body}");
    let _ = f.flush();
}

/// Removes all but the newest KEEP logs in `dir`.
fn prune(dir: &Path) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut logs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("gpu-workbench_") && n.ends_with(".log"))
        })
        .collect();
    logs.sort(); // names start with a sortable timestamp
    if logs.len() > KEEP {
        for p in &logs[..logs.len() - KEEP] {
            let _ = fs::remove_file(p);
        }
    }
}

/// Records panics (including ones caught on the benchmark thread) in the log,
/// then runs the default hook so they still print to the console.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_string();
            line(&format!("PANIC on thread '{thread}': {info}"));
            default(info);
        }));
    });
}

/// Opens the folder holding the log in the system file manager.
pub fn open_folder() {
    let Some(dir) = path().and_then(|p| p.parent().map(Path::to_path_buf)) else {
        return;
    };
    #[cfg(windows)]
    let cmd = "explorer";
    #[cfg(not(windows))]
    let cmd = "xdg-open";
    let _ = std::process::Command::new(cmd).arg(dir).spawn();
}
