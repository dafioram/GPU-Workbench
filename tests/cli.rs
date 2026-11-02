//! Runs the built binary to check its command-line surface and exit codes.
//! None of these need a GPU, so they run the same on every CI runner.

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    // Keep the session log out of the developer's real state directory
    // (XDG on Linux, LOCALAPPDATA on Windows).
    let state = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("state");
    Command::new(env!("CARGO_BIN_EXE_gpu-workbench"))
        .args(args)
        .env("XDG_STATE_HOME", &state)
        .env("LOCALAPPDATA", &state)
        .output()
        .expect("failed to start gpu-workbench")
}

#[test]
fn help_lists_every_test_key() {
    let out = run(&["--help"]);
    assert!(out.status.success(), "--help exited {:?}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    for key in [
        "upload",
        "readback",
        "copy",
        "bandwidth",
        "matmul32",
        "matmul16",
        "vram",
        "stress",
    ] {
        assert!(text.contains(key), "--help does not mention '{key}'");
    }
}

#[test]
fn unknown_option_is_a_usage_error() {
    let out = run(&["--cli", "--no-such-option"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown option"));
}

#[test]
fn unknown_test_is_a_usage_error() {
    let out = run(&["--cli", "--tests", "upload,nope"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown test 'nope'"));
}
