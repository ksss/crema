//! Walk up from a `project_root` to the directory that holds
//! `Gemfile.lock` — the file crema uses as its cache invalidation
//! source and g-snapshot key input. Bundler's own anchor is `Gemfile`
//! (from which it derives `Gemfile.lock = Gemfile + ".lock"`); under
//! the standard layout the two anchors converge on the same
//! directory. [`discover_bundle_root`] anchors on the lockfile
//! directly because what invalidation actually needs is the
//! lockfile's *content*, and a `Gemfile` without a resolved
//! `Gemfile.lock` has nothing to hash into the key yet.
//!
//! Cache-invalidation and g-snapshot key computation run off
//! `project_root` (not `cwd`) — ADR-0028 Decision 4 requires
//! subdirectory execution and root execution to produce the same key,
//! and `cwd` differs between the two. A flat join of
//! `project_root/Gemfile.lock` would miss a separated layout (the
//! Gemfile.lock above the `crema.toml` directory) and never invalidate
//! when the real lockfile changes, hence the walk-up.
//!
//! The bundler side starts from the same place: `ResolverAnchor` in
//! `main.rs` decides bundler mode with `find_gemfile` from
//! `project_root` and runs the bundler probe there, so bundler's own
//! walk-up (from its cwd) reaches the lockfile this walk-up hashes.
//!
//! # Stop marker
//!
//! `.git` matches [`crate::config::Config::discover_walking_up`] and
//! [`crate::rbs_collection::discover_lockfile`]. Consistency across the
//! three walk-up paths means a `crema.toml` monorepo subproject cannot
//! silently pick up an *unrelated* ancestor's Gemfile.lock; the `.git`
//! boundary is what makes it safe to walk past `project_root`.

use std::path::{Path, PathBuf};


/// Filename of a bundler manifest.
pub const GEMFILE: &str = "Gemfile";

/// Filename of a bundler lockfile — the resolver's anchor and the
/// invariant this walk-up preserves.
pub const GEMFILE_LOCK: &str = "Gemfile.lock";

/// Walk up from `project_root` looking for the directory that holds
/// `Gemfile.lock` — the file whose *content* is the cache invalidation
/// source and g-snapshot key input. Returns the *directory* (callers
/// join `GEMFILE_LOCK` / `GEMFILE` themselves).
///
/// Anchor rationale: bundler's own resolver anchors on `Gemfile` (it
/// derives `Gemfile.lock = Gemfile + ".lock"`). Under the standard
/// layout the two anchors converge — `Gemfile` and `Gemfile.lock`
/// share a directory. crema anchors on `Gemfile.lock` directly because
/// what we actually consume is the *lockfile content*; a project with
/// a `Gemfile` but no resolved `Gemfile.lock` has nothing to hash into
/// the invalidation key yet.
///
/// Returns `None` when no `Gemfile.lock` is found before the walk
/// stops at a `.git` boundary or reaches the filesystem root. Also
/// returns `None` if `project_root` cannot be canonicalized — the
/// sibling walkers ([`crate::config::Config::discover_walking_up`],
/// [`crate::rbs_collection::discover_lockfile`]) surface that as
/// `Err`; here it collapses to `None` because the caller sites
/// (cache invalidation, snapshot key) already treat missing lockfile
/// as benign. That deliberate divergence keeps the return shape
/// simple for callers that cannot meaningfully propagate a walk
/// error.
pub fn discover_bundle_root(project_root: &Path) -> Option<PathBuf> {
    let start = project_root.canonicalize().ok()?;
    let mut current = start.as_path();
    loop {
        if current.join(GEMFILE_LOCK).is_file() {
            return Some(current.to_path_buf());
        }
        // Mirror the `.git` guard in `Config::discover_walking_up` /
        // `rbs_collection::discover_lockfile`. Permission errors on the
        // probe collapse to "stop the walk" (via `unwrap_or(true)`) —
        // silently sailing past an unreadable `.git` is the failure
        // mode this rule exists to prevent.
        if current.join(".git").try_exists().unwrap_or(true) {
            return None;
        }
        current = current.parent()?;
    }
}
