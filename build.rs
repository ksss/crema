//! Exports `CREMA_GIT_INFO` (` (<short hash> <commit date>)`, or empty when git
//! is unavailable) so `crema --version` can tell which commit a binary came from.
//!
//! The value is read only by `src/main.rs`: a HEAD move re-runs this script and
//! recompiles the bin crate, while the lib crate stays fresh.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        // Set by git hooks and some tooling; would point at another repo.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Emit `rerun-if-changed` only for paths that exist: cargo treats a missing
/// path as always stale and would re-run this script on every build.
fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    // Always emitted so cargo never falls back to "re-run on any file change".
    println!("cargo:rerun-if-changed=build.rs");

    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let info = commit_info(&dir).unwrap_or_default();
    println!("cargo:rustc-env=CREMA_GIT_INFO={info}");
}

fn commit_info(dir: &Path) -> Option<String> {
    // git walks up to parent directories, so a `.git`-less source tree nested
    // inside another repo would otherwise report that repo's HEAD.
    let toplevel = PathBuf::from(git(dir, &["rev-parse", "--show-toplevel"])?);
    if toplevel.canonicalize().ok()? != dir.canonicalize().ok()? {
        return None;
    }

    let hash = git(dir, &["rev-parse", "--short=8", "HEAD"])?;
    let date = git(dir, &["log", "-1", "--format=%cs"])?;

    // In a linked worktree HEAD lives in the per-worktree git dir while branch
    // refs and packed-refs live in the common dir, so watch both.
    let git_dir = dir.join(git(dir, &["rev-parse", "--git-dir"])?);
    let common_dir = dir.join(git(dir, &["rev-parse", "--git-common-dir"])?);
    watch(&git_dir.join("HEAD"));
    watch(&common_dir.join("packed-refs"));
    // A commit on a branch that only lives in packed-refs creates a new loose
    // ref after this script ran, so there was nothing to watch. The HEAD
    // reflog is appended on every commit / checkout / reset.
    watch(&git_dir.join("logs/HEAD"));
    if let Some(head_ref) = git(dir, &["symbolic-ref", "-q", "HEAD"]) {
        watch(&common_dir.join(head_ref));
    }

    Some(format!(" ({hash} {date})"))
}
