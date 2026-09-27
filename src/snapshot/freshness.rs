//! G-layer file freshness manifest: (path, mtime, size) for every `.rbs`
//! file the cold build loaded, baked into the snapshot so a warm hit can
//! detect direct edits to the G layer (fake_rbs core swaps during
//! debugging, `gem install` overwriting a pinned gem's sig, ...) that the
//! invalidation key cannot see — the key only covers lockfile / crema
//! version / crema.toml / sig-path-set content
//! (`g_snapshot_g_layer_freshness_check`).
//!
//! Detection is metadata-only (stat, not content read/hash): a 2026-08-09
//! measurement found content hashing costs ~7ms/4.7MB (read-dominated,
//! survives switching hash functions) against a stat scan's ~0.2ms for
//! 250 files, so freshness trades a small false-negative risk (mtime
//! restored to its old value while content changes — not a realistic
//! editor/git/gem-install edit path) for staying inside the 10ms budget
//! (ADR-0028).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct GFileStat {
    pub path: String,
    pub mtime_secs: u64,
    pub mtime_nanos: u32,
    pub size: u64,
}

/// Baked at cold time, compared against a fresh [`scan`] of `dirs` on
/// every warm attempt ([`is_fresh`]). `dirs` are the G-layer roots the
/// cold build actually called `load_dir` on (core, resolved libraries,
/// collection-lock gems) — `files` is every `.rbs` file found under them.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GFreshnessManifest {
    pub dirs: Vec<String>,
    pub files: Vec<GFileStat>,
}

/// Stat every `.rbs` file under `dirs`, recursively. A directory that
/// cannot be read (missing, permission denied) contributes no files
/// instead of failing the scan — mirrors the cold ingest loop's existing
/// `is_dir()` / `load_dir` error handling, which likewise degrades to
/// "nothing loaded from here" rather than aborting the run.
pub fn scan(dirs: &[PathBuf]) -> GFreshnessManifest {
    let mut files = Vec::new();
    for dir in dirs {
        collect(dir, &mut files);
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    GFreshnessManifest {
        dirs: dirs
            .iter()
            .map(|d| d.to_string_lossy().into_owned())
            .collect(),
        files,
    }
}

fn collect(dir: &Path, out: &mut Vec<GFileStat>) {
    // Same walk as the cold build's `load_dir` on a G-layer dir
    // (`skip_hidden: true`: `_` dirs pruned, symlinked dirs never
    // descended), so the manifest is exactly the loaded file set.
    for path in crate::file_finder::each_file(dir, true).unwrap_or_default() {
        // A file that vanishes or turns unreadable between the
        // listing and this `stat` is dropped from the manifest rather
        // than erroring the scan — it will show up as a set difference
        // against the stored manifest, which `is_fresh` already treats
        // as a miss.
        if let Ok(meta) = fs::metadata(&path) {
            let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
            let since_epoch = mtime.duration_since(UNIX_EPOCH).unwrap_or_default();
            out.push(GFileStat {
                path: path.to_string_lossy().into_owned(),
                mtime_secs: since_epoch.as_secs(),
                mtime_nanos: since_epoch.subsec_nanos(),
                size: meta.len(),
            });
        }
    }
}

/// Re-[`scan`] `stored.dirs` and compare against `stored.files`. Any
/// added/removed/edited file (path set differs, or a shared path's
/// mtime/size differs) means the G layer changed since the snapshot was
/// written — the caller should treat that the same as an invalidation-key
/// mismatch (cold rebuild).
pub fn is_fresh(stored: &GFreshnessManifest) -> bool {
    let dirs: Vec<PathBuf> = stored.dirs.iter().map(PathBuf::from).collect();
    scan(&dirs).files == stored.files
}

