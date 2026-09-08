//! Atomic file-write helper shared by every cache / snapshot writer.
//!
//! Call sites — `gem_dir_cache` and `snapshot::write` — previously
//! inlined the same three-step dance (create parent dirs → `fs::write`
//! a `.tmp` sibling → `fs::rename` onto the final path). POSIX rename
//! is atomic within a filesystem, so a concurrent reader observes
//! either the old file or the new one but never a partial write.
//! Extract that dance here so a fix (fsync, error text, temp-file
//! cleanup) touches one place instead of each caller.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Write `bytes` to `final_path` atomically: create the parent
/// directory chain if missing, write the payload to `<final_path>.tmp`,
/// then rename onto `final_path`.
///
/// The caller owns pre-write cleanup that is specific to its cache
/// shape (e.g. removing a stale-directory cache path *before* calling
/// this, so the rename lands on a fresh slot). Keeping such special
/// cases in the caller keeps this helper the single, unspecialized
/// bytes-in-file-out primitive.
pub fn write_atomic_bytes(final_path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = with_tmp_suffix(final_path);
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Return `<path>.tmp`. Kept as a separate fn so a future test that
/// wants to observe the intermediate file (e.g. asserting it is
/// cleaned up on a rename failure) can name the exact path
/// [`write_atomic_bytes`] would.
fn with_tmp_suffix(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}
