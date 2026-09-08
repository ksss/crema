//! G-snapshot read path: file read + header/key validation, with every
//! failure folded to `None` (silent full-rebuild fallback).
//!
//! [`read_g_snapshot_backend`] is the production warm path (lazy
//! backend, ADR-0028 slice 2a-2); [`read_g_snapshot`] is the slice 1c
//! eager rebuild, kept as the reference decode for roundtrip tests.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::name::NameTable;
use crate::snapshot::backend::GSnapshotBackend;
use crate::snapshot::freshness::{self, GFreshnessManifest};
use crate::snapshot::invalidation::InvalidationKey;
use crate::snapshot::rebuild::{RebuiltGSnapshot, rebuild};
use crate::snapshot::write::{
    FRESH_LEN_OFFSET, FRESH_OFF_OFFSET, G_SNAPSHOT_MAGIC, G_SNAPSHOT_SCHEMA_VERSION, HEADER_LEN,
};

/// Read `cache_path` and open it as a lazy snapshot backend after
/// verifying the header, `expected_key`, and the G-layer freshness
/// manifest (`g_snapshot_g_layer_freshness_check`: a stat scan of the
/// baked G-layer directories must reproduce the baked (path, mtime,
/// size) list exactly). All failures fold to `None` (silent cold
/// rebuild). These checks run before [`GSnapshotBackend::open`] interns
/// anything, so a stale snapshot leaves `names` untouched.
pub fn read_g_snapshot_backend(
    cache_path: &Path,
    expected_key: &InvalidationKey,
    names: &NameTable,
) -> Option<Arc<GSnapshotBackend>> {
    let timing = std::env::var_os("CREMA_DEBUG_SNAPSHOT_TIMING").is_some_and(|v| v == "1");
    let miss = |reason: &dyn std::fmt::Display| {
        if timing {
            eprintln!("g-snapshot: miss ({reason})");
        }
    };
    let t0 = Instant::now();
    let bytes = match fs::read(cache_path) {
        Ok(b) => b,
        Err(e) => {
            miss(&format_args!("{}: {}", cache_path.display(), e));
            return None;
        }
    };
    if bytes.len() < HEADER_LEN {
        miss(&"file shorter than header");
        return None;
    }
    if u64::from_le_bytes(bytes[0..8].try_into().unwrap()) != G_SNAPSHOT_MAGIC {
        miss(&"bad snapshot magic");
        return None;
    }
    if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != G_SNAPSHOT_SCHEMA_VERSION {
        miss(&"unsupported schema version");
        return None;
    }
    if bytes[16..48] != expected_key.0 {
        miss(&"invalidation key mismatch");
        return None;
    }
    let fresh_off = u64::from_le_bytes(
        bytes[FRESH_OFF_OFFSET..FRESH_OFF_OFFSET + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let fresh_len = u64::from_le_bytes(
        bytes[FRESH_LEN_OFFSET..FRESH_LEN_OFFSET + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let fresh_end = match fresh_off
        .checked_add(fresh_len)
        .filter(|&end| fresh_off >= HEADER_LEN && end <= bytes.len())
    {
        Some(end) => end,
        None => {
            miss(&"freshness manifest section out of bounds");
            return None;
        }
    };
    let manifest: GFreshnessManifest = match bincode::deserialize(&bytes[fresh_off..fresh_end]) {
        Ok(m) => m,
        Err(e) => {
            miss(&format_args!("corrupt freshness manifest: {e}"));
            return None;
        }
    };
    if !freshness::is_fresh(&manifest) {
        miss(&"g-layer file changed since snapshot");
        return None;
    }
    let len = bytes.len();
    match GSnapshotBackend::open(bytes, names) {
        Ok(backend) => {
            if timing {
                eprintln!(
                    "g-snapshot: warm hit ({} bytes, open {:.1}ms, lazy backend)",
                    len,
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
            Some(Arc::new(backend))
        }
        Err(e) => {
            miss(&e);
            None
        }
    }
}

/// Read `cache_path`, rebuild the frozen gem environment, and verify the
/// stored invalidation key matches `expected_key`.
///
/// Every failure (missing file, bad magic, unsupported schema version,
/// corrupt payload, key mismatch) returns `None` so the caller falls back
/// to a full cold build — which rewrites the snapshot. Nothing is printed:
/// diagnostics must stay byte-identical with the snapshot-off path. Set
/// `CREMA_DEBUG_SNAPSHOT_TIMING=1` for an opt-in stderr breakdown.
///
/// The whole file is read eagerly (`fs::read`, not mmap): slice 1c decodes
/// every entry anyway, so mapping lazily buys nothing; the lazy mmap-probe
/// backend is slice 2's territory.
pub fn read_g_snapshot(
    cache_path: &Path,
    expected_key: &InvalidationKey,
) -> Option<RebuiltGSnapshot> {
    let timing = std::env::var_os("CREMA_DEBUG_SNAPSHOT_TIMING").is_some_and(|v| v == "1");
    let t0 = Instant::now();
    let bytes = match fs::read(cache_path) {
        Ok(b) => b,
        Err(e) => {
            if timing {
                eprintln!("g-snapshot: miss ({}: {})", cache_path.display(), e);
            }
            return None;
        }
    };
    let t_read = t0.elapsed();
    match rebuild(&bytes) {
        Ok(r) if r.key == *expected_key => {
            if timing {
                eprintln!(
                    "g-snapshot: warm hit ({} bytes, read {:.1}ms, rebuild {:.1}ms)",
                    bytes.len(),
                    t_read.as_secs_f64() * 1000.0,
                    (t0.elapsed() - t_read).as_secs_f64() * 1000.0,
                );
            }
            Some(r)
        }
        Ok(_) => {
            if timing {
                eprintln!("g-snapshot: miss (invalidation key mismatch)");
            }
            None
        }
        Err(e) => {
            if timing {
                eprintln!("g-snapshot: miss ({e})");
            }
            None
        }
    }
}
