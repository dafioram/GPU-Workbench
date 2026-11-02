//! Runs the built binaries to check their command-line surface and exit codes.
//! None of these need a GPU, so they run the same on every CI runner.

use std::process::{Command, Output};

/// Runs one of the binaries with `args`, with the session log kept in a
/// scratch directory rather than the developer's real state directory.
fn run_with(bin: &str, args: &[&str]) -> Output {
    let state = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("state");
    Command::new(bin)
        .args(args)
        .env("XDG_STATE_HOME", &state)
        .env("LOCALAPPDATA", &state)
        .output()
        .expect("failed to start gpu-workbench")
}

/// The console CLI, which is what scripts and PowerShell use.
fn cli(args: &[&str]) -> Output {
    run_with(env!("CARGO_BIN_EXE_gpu-workbench-cli"), args)
}

#[test]
fn help_lists_every_test_key() {
    let out = cli(&["--help"]);
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
fn gui_binary_still_answers_help() {
    let out = run_with(env!("CARGO_BIN_EXE_gpu-workbench"), &["--help"]);
    assert!(
        out.status.success(),
        "gpu-workbench --help exited {:?}",
        out.status
    );
}

#[test]
fn unknown_option_is_a_usage_error() {
    let out = cli(&["--no-such-option"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown option"));
}

#[test]
fn unknown_test_is_a_usage_error() {
    let out = cli(&["--tests", "upload,nope"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown test 'nope'"));
}

#[test]
fn zero_and_out_of_range_values_are_refused_before_any_work() {
    // Each of these must fail with exit code 2 and name the flag, and must not
    // start a run (so no adapter is opened).
    for (args, flag) in [
        (&["--stress-secs", "0"][..], "--stress-secs"),
        (&["--transfer-mb", "0"][..], "--transfer-mb"),
        (&["--matmul-n", "100"][..], "--matmul-n"),
        (&["--adapter", "abc"][..], "--adapter"),
    ] {
        let out = cli(args);
        assert_eq!(out.status.code(), Some(2), "{args:?} should exit 2");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains(flag),
            "{args:?}: stderr does not name {flag}: {err}"
        );
    }
}
