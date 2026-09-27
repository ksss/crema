//! Port of rbs `FileFinder.each_file` (`lib/rbs/file_finder.rb`): the
//! one walk every `.rbs` directory ingest shares, so the environment
//! load, the sig-path change-detection set, and the G-layer freshness
//! manifest can never disagree about which files a directory contains.
//!
//! Rules mirrored from rbs:
//! - `Pathname#glob("**/*.rbs")` never descends a symlinked directory
//!   (the root itself may be a symlink — `rbs collection install` links
//!   local-source install dirs). A symlinked `.rbs` *file* is still
//!   yielded.
//! - With `skip_hidden`, any directory component below the root whose
//!   name starts with `_` is pruned (`relative_path_from(path).ascend
//!   .drop(1)` — the root's own name is never inspected, and a `_x.rbs`
//!   file name is not a directory component). rbs `EnvironmentLoader`
//!   passes `skip_hidden: !source.is_a?(Pathname)`: core / library /
//!   collection sources skip, user `-I` dirs do not.
//! - Results are sorted by path string (`paths.sort_by!(&:to_s)`).
//! - A subdirectory (or entry) that cannot be read is skipped, like
//!   `Dir.glob` does; only an unreadable root is an error, so callers
//!   keep crema's existing "failed to load <dir>" warning.

use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Every `.rbs` file under `dir`, sorted. `Err` only when `dir` itself
/// cannot be listed.
pub fn each_file(dir: &Path, skip_hidden: bool) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Cannot read directory {}: {}", dir.display(), e))?;
    let mut out = Vec::new();
    collect(entries, skip_hidden, &mut out);
    // Byte order of the whole path string, not `Path`'s component-wise
    // `Ord` (which puts `a/b.rbs` before `a-b.rbs`).
    out.sort_by(|a, b| a.as_os_str().as_bytes().cmp(b.as_os_str().as_bytes()));
    Ok(out)
}

fn collect(entries: std::fs::ReadDir, skip_hidden: bool, out: &mut Vec<PathBuf>) {
    for entry in entries.flatten() {
        let path = entry.path();
        // `file_type()` reports the link itself (no follow), which is
        // what lets a symlinked dir be pruned without a second stat.
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            if skip_hidden
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with('_'))
            {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(&path) {
                collect(entries, skip_hidden, out);
            }
        } else if file_type.is_symlink() && path.is_dir() {
            continue;
        } else if path.extension().is_some_and(|ext| ext == "rbs") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, content: &str) {
        fs::write(dir.join(name), content).unwrap();
    }

    fn layout() -> tempfile::TempDir {
        // root/
        //   a.rbs, _top.rbs, README.md
        //   _hid/y.rbs
        //   real/b.rbs, real/_deep/c.rbs
        //   link -> real          (symlink dir)
        //   loop -> root          (symlink dir, self-loop)
        //   file_link.rbs -> a.rbs (symlink file)
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, "a.rbs", "");
        write(root, "_top.rbs", "");
        write(root, "README.md", "");
        fs::create_dir(root.join("_hid")).unwrap();
        write(&root.join("_hid"), "y.rbs", "");
        fs::create_dir_all(root.join("real/_deep")).unwrap();
        write(&root.join("real"), "b.rbs", "");
        write(&root.join("real/_deep"), "c.rbs", "");
        std::os::unix::fs::symlink(root.join("real"), root.join("link")).unwrap();
        std::os::unix::fs::symlink(root, root.join("loop")).unwrap();
        std::os::unix::fs::symlink(root.join("a.rbs"), root.join("file_link.rbs")).unwrap();
        tmp
    }

    fn rel(root: &Path, paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn skip_hidden_prunes_underscore_dirs_at_every_depth_but_not_files() {
        let tmp = layout();
        let files = each_file(tmp.path(), true).unwrap();
        assert_eq!(
            rel(tmp.path(), &files),
            ["_top.rbs", "a.rbs", "file_link.rbs", "real/b.rbs"]
        );
    }

    #[test]
    fn without_skip_hidden_underscore_dirs_are_read_but_symlink_dirs_still_are_not() {
        let tmp = layout();
        let files = each_file(tmp.path(), false).unwrap();
        assert_eq!(
            rel(tmp.path(), &files),
            [
                "_hid/y.rbs",
                "_top.rbs",
                "a.rbs",
                "file_link.rbs",
                "real/_deep/c.rbs",
                "real/b.rbs",
            ]
        );
    }

    #[test]
    fn root_may_be_a_symlink_and_its_own_underscore_name_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("_real");
        fs::create_dir(&real).unwrap();
        write(&real, "a.rbs", "");
        let link = tmp.path().join("rootlink");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let files = each_file(&link, true).unwrap();
        assert_eq!(rel(&link, &files), ["a.rbs"]);
    }

    #[test]
    fn sorts_by_path_string_like_rbs() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir(tmp.path().join("a")).unwrap();
        write(&tmp.path().join("a"), "b.rbs", "");
        write(tmp.path(), "a-b.rbs", "");
        let files = each_file(tmp.path(), true).unwrap();
        assert_eq!(rel(tmp.path(), &files), ["a-b.rbs", "a/b.rbs"]);
    }

    #[test]
    fn unreadable_subdir_is_skipped_and_siblings_kept() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "a.rbs", "");
        fs::create_dir(tmp.path().join("bad")).unwrap();
        write(&tmp.path().join("bad"), "c.rbs", "");
        fs::create_dir(tmp.path().join("ok")).unwrap();
        write(&tmp.path().join("ok"), "b.rbs", "");
        fs::set_permissions(tmp.path().join("bad"), fs::Permissions::from_mode(0o000)).unwrap();
        let files = each_file(tmp.path(), true);
        fs::set_permissions(tmp.path().join("bad"), fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(rel(tmp.path(), &files.unwrap()), ["a.rbs", "ok/b.rbs"]);
    }

    #[test]
    fn unreadable_root_is_an_error() {
        let err = each_file(Path::new("/nonexistent/crema_file_finder"), true).unwrap_err();
        assert!(err.starts_with("Cannot read directory"), "{err}");
    }
}
