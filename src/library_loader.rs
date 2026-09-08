use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::gem_dir_cache::ResolvedGemDirs;


/// stdlib RBS lives under `${rbs_gem_dir}/stdlib/<name>/<version>/`. Every
/// library bundled in rbs uses the single version directory `0`, so crema
/// hardcodes it. Version selection only matters once a git/rubygems source
/// with multiple versions is added (tracked in `low_rbs_collection.md`).
const STDLIB_VERSION: &str = "0";

/// Shown once per run (not once per missing gem) alongside a genuinely
/// missing rubygems-lock entry (`resolve_lock_gems`'s "not found"
/// warning), pointing at the full resolver conversation dump so
/// troubleshooting a gem-resolution failure doesn't require reading
/// crema's source. Printed by the caller (main.rs), not pushed into
/// `ResolvedLibraries::warnings` itself, so `resolve_lock_gems` stays a
/// pure resolution function and its warning count stays one-per-miss.
pub const RESOLVER_DEBUG_HINT: &str =
    "hint: re-run with CREMA_DEBUG_RESOLVER=1 to see the gem resolver's conversation";

/// Result of resolving a set of library names into the RBS directories to
/// load. Returned by both `resolve` (the `crema.toml` `libraries = [...]`
/// path with manifest expansion) and the `resolve_lock_*_gems` helpers
/// (the `rbs collection` lock paths, no manifest expansion). `dirs`
/// preserves source order with each entry appearing once. `warnings`
/// carries one message per entry whose directory could not be found,
/// surfaced on stderr by the caller — crema continues as a best-effort
/// preflight checker rather than aborting, where rbs raises
/// `UnknownLibraryError` for the libraries path and
/// `CollectionNotAvailable` / `UnknownLibraryError` for the lock paths.
///
/// `claimed` holds every library name this pass took ownership of
/// (manifest-expanded deps included), found or not — mirroring rbs
/// `EnvironmentLoader#add`, which inserts into its `libs` set before
/// resolution. Later passes seed their dedup from the earlier passes'
/// `claimed` so the load-order rule (crema.toml → lock rubygems → lock
/// stdlib → lock git) wins over a second load of the same name.
/// Path-traversal-rejected lock entries are never claimed.
pub struct ResolvedLibraries {
    pub dirs: Vec<PathBuf>,
    pub warnings: Vec<String>,
    pub claimed: HashSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    Rubygems,
    Stdlib,
    Git,
    Local,
}

impl SourceKind {
    pub fn load_error_label(self) -> &'static str {
        match self {
            SourceKind::Rubygems => "collection gem",
            SourceKind::Stdlib => "stdlib collection gem",
            SourceKind::Git => "git collection gem",
            SourceKind::Local => "local collection gem",
        }
    }
}

pub struct ResolvedCollectionLock {
    pub dirs: Vec<(SourceKind, PathBuf)>,
    pub warnings: Vec<String>,
}

pub struct GitLockSource<'a> {
    pub lockfile_dir: &'a Path,
    pub lockfile_path: &'a Path,
    pub entries: &'a [(&'a str, &'a str)],
}

pub struct LocalLockSource<'a> {
    pub lockfile_dir: &'a Path,
    pub lockfile_path: &'a Path,
    pub entries: &'a [(&'a str, &'a str)],
}

/// Resolve the `libraries` list against the gem-dir map, mirroring rbs's
/// `EnvironmentLoader#each_dir` order: gem-sig (`${gem_dir}/sig/`) is
/// tried first, then `${rbs_gem_dir}/stdlib/<name>/0/`. A library
/// without an entry in `gem_dirs.libraries` (or with `None`) skips the
/// gem-sig step and falls through to stdlib.
///
/// Both paths read `manifest.yaml` for transitive `dependencies`,
/// matching rbs `Collection::Sources::Rubygems#manifest_of` and
/// `Stdlib`'s behavior: a library is expanded only the first time it is
/// seen, giving dedup and cycle termination in one step.
pub fn resolve(gem_dirs: &ResolvedGemDirs, libraries: &[String]) -> ResolvedLibraries {
    let mut seen: HashSet<String> = HashSet::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    for library in libraries {
        add(gem_dirs, library, &mut seen, &mut dirs, &mut warnings);
    }

    ResolvedLibraries {
        dirs,
        warnings,
        claimed: seen,
    }
}

fn add(
    gem_dirs: &ResolvedGemDirs,
    name: &str,
    seen: &mut HashSet<String>,
    dirs: &mut Vec<PathBuf>,
    warnings: &mut Vec<String>,
) {
    if !seen.insert(name.to_string()) {
        return;
    }

    let Some(dir) = resolve_one(gem_dirs, name) else {
        warnings.push(format!(
            "warning: cannot find type definitions for library: {}",
            name
        ));
        return;
    };

    dirs.push(dir.clone());

    let manifest = dir.join("manifest.yaml");
    match std::fs::read_to_string(&manifest) {
        Ok(content) => {
            for dep in parse_manifest_deps(&content) {
                add(gem_dirs, &dep, seen, dirs, warnings);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            warnings.push(format!(
                "warning: cannot read manifest for library {}: {}",
                name, e
            ));
        }
    }
}

/// Find the sig directory for a single library. gem-sig wins over
/// stdlib when both exist; a missing or `None` library entry skips
/// straight to the stdlib lookup.
fn resolve_one(gem_dirs: &ResolvedGemDirs, name: &str) -> Option<PathBuf> {
    if let Some(sig) = resolve_gem_sig(gem_dirs, name) {
        return Some(sig);
    }

    let stdlib_dir = gem_dirs
        .rbs_gem_dir
        .join("stdlib")
        .join(name)
        .join(STDLIB_VERSION);
    if stdlib_dir.is_dir() {
        Some(stdlib_dir)
    } else {
        None
    }
}

/// Gem-sig-only lookup: `${gem_dir}/sig/` if the name has a gem_dir
/// entry and that `sig/` directory exists. No stdlib fallback —
/// rubygems-source lock entries explicitly mean "this gem ships its
/// own RBS", so a stdlib backup would silently drift from what the
/// lock pins. Mirrors the first half of `resolve_one`, reused
/// independently by `resolve_lock_gems`.
fn resolve_gem_sig(gem_dirs: &ResolvedGemDirs, name: &str) -> Option<PathBuf> {
    // rbs is hoisted out of `libraries` into `rbs_gem_dir` (see
    // gem_dir_cache.rs). Without this special case, a collection-lock
    // entry of `name: rbs` falls through to `libraries.get("rbs")` →
    // None and emits a bogus "not found" warning even when the gem is
    // installed. The hoist stays as is; we just teach the lookup
    // about it.
    if name == "rbs" {
        let sig = gem_dirs.rbs_gem_dir.join("sig");
        return if sig.is_dir() { Some(sig) } else { None };
    }
    if let Some(Some(gem_dir)) = gem_dirs.libraries.get(name) {
        let sig = gem_dir.join("sig");
        if sig.is_dir() {
            return Some(sig);
        }
    }
    None
}

/// Resolve rubygems-source gems from an `rbs collection` lockfile.
/// Returns the sig directories to load, in lockfile order, skipping
/// names already claimed by earlier passes (`already_loaded` —
/// `crema.toml` `libraries` including their manifest-expanded deps).
///
/// Lock-derived loads differ from `crema.toml libraries` in two ways:
///   1. **No manifest expansion.** The lockfile is the transitive
///      closure already (rbs `EnvironmentLoader#add_collection` calls
///      `add(..., resolve_dependencies: false)` for the same reason),
///      so reading `manifest.yaml` here would double-load deps and
///      break the lock's pin.
///   2. **No stdlib fallback.** A `Source::Rubygems` entry asserts the
///      gem ships its own RBS; falling back to `stdlib/<name>/0/`
///      would silently substitute the bundled copy when the gem is
///      uninstalled.
///
/// Missing entries (gem not installed, or installed without a `sig/`)
/// produce a warning on `warnings` and are otherwise skipped — the lock
/// path runs in best-effort mode so one install hole does not block the
/// rest of the type check. Path-traversal-rejected names stay silent on
/// purpose: those mean the lockfile itself is malformed (rbs CLI never
/// emits such entries) rather than an install miss the user can fix.
pub fn resolve_lock_gems(
    gem_dirs: &ResolvedGemDirs,
    already_loaded: &HashSet<String>,
    lock_gem_names: &[&str],
) -> ResolvedLibraries {
    let mut dirs = Vec::new();
    let mut warnings = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for name in lock_gem_names {
        if !is_valid_lock_path_component(name) {
            continue;
        }
        if already_loaded.contains(*name) {
            continue;
        }
        if !seen.insert(name.to_string()) {
            continue;
        }
        if let Some(sig) = resolve_gem_sig(gem_dirs, name) {
            dirs.push(sig);
        } else {
            warnings.push(format!("warning: rbs collection: gem '{}' not found", name));
        }
    }
    ResolvedLibraries {
        dirs,
        warnings,
        claimed: seen,
    }
}

/// Lock entries are joined into install paths verbatim (stdlib:
/// `${rbs_gem_dir}/stdlib/<name>/0/`, git:
/// `<lockfile_dir>/<lockfile.path>/<name>/<version>/`), so anything other
/// than a single path component would let a malformed lockfile reach
/// outside the install root. rbs gem names and versions are bare
/// identifiers (letters, digits, `-`, `_`, `.`), so the cheapest correct
/// check is "no path separators, not empty, not a parent reference".
/// Both name and version pass through this gate.
fn is_valid_lock_path_component(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains('/') && !s.contains('\\')
}

/// Resolve stdlib-source gems from an `rbs collection` lockfile.
/// Returns the sig directories to load, in lockfile order, skipping
/// names already claimed by earlier passes (`already_loaded` —
/// `crema.toml` `libraries` with manifest-expanded deps, plus lock
/// rubygems names, so earlier paths win when both list the same gem).
///
/// The lock entry's `version` field is ignored: rbs
/// `LockfileGenerator#assign_stdlib` writes `"0"` for every stdlib
/// entry, and rbs ships exactly one `stdlib/<name>/0/` dir per gem,
/// so resolution always points at the `STDLIB_VERSION` directory.
/// A missing `stdlib/<name>/0/` is reported on `warnings` and otherwise
/// skipped, matching the sibling `resolve_lock_gems`. Path-traversal-
/// rejected names stay silent for the same reason.
///
/// Like `resolve_lock_gems`, this does NOT expand `manifest.yaml`
/// dependencies: the lockfile is the transitive closure (rbs
/// `EnvironmentLoader#add_collection` uses
/// `resolve_dependencies: false` for the same reason).
pub fn resolve_lock_stdlib_gems(
    gem_dirs: &ResolvedGemDirs,
    already_loaded: &HashSet<String>,
    lock_gem_names: &[&str],
) -> ResolvedLibraries {
    let mut dirs = Vec::new();
    let mut warnings = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for name in lock_gem_names {
        if !is_valid_lock_path_component(name) {
            continue;
        }
        if already_loaded.contains(*name) {
            continue;
        }
        if !seen.insert(name.to_string()) {
            continue;
        }
        let stdlib_dir = gem_dirs
            .rbs_gem_dir
            .join("stdlib")
            .join(name)
            .join(STDLIB_VERSION);
        if stdlib_dir.is_dir() {
            dirs.push(stdlib_dir);
        } else {
            warnings.push(format!(
                "warning: rbs collection: stdlib gem '{}' not found at {}",
                name,
                stdlib_dir.display()
            ));
        }
    }
    ResolvedLibraries {
        dirs,
        warnings,
        claimed: seen,
    }
}

/// Resolve git-source gems from an `rbs collection` lockfile. Returns
/// the install directories to load, in lockfile order, skipping any
/// entry whose `name` is already covered by `already_loaded`
/// (`crema.toml libraries` with manifest-expanded deps, plus lock
/// rubygems/stdlib names claimed by earlier passes, so earlier paths
/// win). Dedup is
/// name-only — the lockfile generator never emits two entries with the
/// same name + different versions for the same source, and across
/// sources the goal is to prevent duplicate class declarations
/// regardless of version.
///
/// Git-source entries point at the on-disk dir
/// `<lockfile_dir>/<lockfile_path>/<name>/<version>/`. crema does not
/// clone or fetch; that is `rbs collection install`'s responsibility.
/// `lockfile_path` is the `path` field of the lockfile, joined verbatim
/// — if a user sets it to an absolute path, `PathBuf::join` overrides
/// the left-hand side per Rust's path semantics.
///
/// Both `name` and `version` are joined into the install path verbatim,
/// so both pass through `is_valid_lock_path_component` to guard against
/// a malformed lockfile escaping the install root.
///
/// Like the rubygems/stdlib siblings, this does NOT expand
/// `manifest.yaml` dependencies: the lockfile is the transitive closure
/// (rbs `EnvironmentLoader#add_collection` uses
/// `resolve_dependencies: false` for the same reason). Missing dirs
/// produce a warning on `warnings` and are otherwise skipped. Path-
/// traversal-rejected name/version pairs stay silent for the same
/// "malformed lockfile, not an install hole" reason as the siblings.
pub fn resolve_lock_git_gems(
    already_loaded: &HashSet<String>,
    lockfile_dir: &Path,
    lockfile_path: &Path,
    git_entries: &[(&str, &str)],
) -> ResolvedLibraries {
    let mut dirs = Vec::new();
    let mut warnings = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let install_root = lockfile_dir.join(lockfile_path);
    for (name, version) in git_entries {
        if !is_valid_lock_path_component(name) || !is_valid_lock_path_component(version) {
            continue;
        }
        if already_loaded.contains(*name) {
            continue;
        }
        if !seen.insert((*name).to_string()) {
            continue;
        }
        let gem_dir = install_root.join(name).join(version);
        if gem_dir.is_dir() {
            dirs.push(gem_dir);
        } else {
            warnings.push(format!(
                "warning: rbs collection: git gem '{}' ({}) not found at {}",
                name,
                version,
                gem_dir.display()
            ));
        }
    }
    ResolvedLibraries {
        dirs,
        warnings,
        claimed: seen,
    }
}

/// Resolve local-source gems from an `rbs collection` lockfile. The
/// install dir matches `Source::Git`
/// (`<lockfile_dir>/<lockfile_path>/<name>/<version>/`) because rbs CLI's
/// `Collection::Sources::Local#install` symlinks that dir to the user's
/// `Source::Local.path`. crema reads the install dir directly; `std::fs`
/// follows the symlink so no symlink-specific code is needed here.
///
/// Same gates as the git sibling: `is_valid_lock_path_component` on both
/// `name` and `version`, dedup against `already_loaded`, missing dirs
/// (dangling symlink, never-installed entry) produce a warning and are
/// otherwise skipped. Path-traversal-rejected entries stay silent for
/// the same "malformed lockfile, not an install hole" reason.
pub fn resolve_lock_local_gems(
    already_loaded: &HashSet<String>,
    lockfile_dir: &Path,
    lockfile_path: &Path,
    entries: &[(&str, &str)],
) -> ResolvedLibraries {
    let mut dirs = Vec::new();
    let mut warnings = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let install_root = lockfile_dir.join(lockfile_path);
    for (name, version) in entries {
        if !is_valid_lock_path_component(name) || !is_valid_lock_path_component(version) {
            continue;
        }
        if already_loaded.contains(*name) {
            continue;
        }
        if !seen.insert((*name).to_string()) {
            continue;
        }
        let gem_dir = install_root.join(name).join(version);
        if gem_dir.is_dir() {
            dirs.push(gem_dir);
        } else {
            warnings.push(format!(
                "warning: rbs collection: local gem '{}' ({}) not found at {}",
                name,
                version,
                gem_dir.display()
            ));
        }
    }
    ResolvedLibraries {
        dirs,
        warnings,
        claimed: seen,
    }
}

pub fn resolve_collection_lock(
    gem_dirs: &ResolvedGemDirs,
    claimed: &HashSet<String>,
    lock_rubygems_refs: &[&str],
    lock_stdlib_refs: &[&str],
    git_source: Option<GitLockSource<'_>>,
    local_source: Option<LocalLockSource<'_>>,
) -> ResolvedCollectionLock {
    let lock_rubygems = resolve_lock_gems(gem_dirs, claimed, lock_rubygems_refs);

    let mut dirs: Vec<(SourceKind, PathBuf)> = lock_rubygems
        .dirs
        .iter()
        .cloned()
        .map(|dir| (SourceKind::Rubygems, dir))
        .collect();
    let mut warnings = lock_rubygems.warnings;

    let mut stdlib_claimed_names = claimed.clone();
    stdlib_claimed_names.extend(lock_rubygems.claimed.iter().cloned());
    let lock_stdlib = resolve_lock_stdlib_gems(gem_dirs, &stdlib_claimed_names, lock_stdlib_refs);
    dirs.extend(
        lock_stdlib
            .dirs
            .iter()
            .cloned()
            .map(|dir| (SourceKind::Stdlib, dir)),
    );
    warnings.extend(lock_stdlib.warnings);

    let mut loaded_names = stdlib_claimed_names;
    loaded_names.extend(lock_stdlib.claimed.iter().cloned());

    if let Some(git_source) = git_source {
        let lock_git = resolve_lock_git_gems(
            &loaded_names,
            git_source.lockfile_dir,
            git_source.lockfile_path,
            git_source.entries,
        );
        loaded_names.extend(lock_git.claimed.iter().cloned());
        dirs.extend(lock_git.dirs.into_iter().map(|dir| (SourceKind::Git, dir)));
        warnings.extend(lock_git.warnings);
    }

    if let Some(local_source) = local_source {
        let lock_local = resolve_lock_local_gems(
            &loaded_names,
            local_source.lockfile_dir,
            local_source.lockfile_path,
            local_source.entries,
        );
        dirs.extend(
            lock_local
                .dirs
                .into_iter()
                .map(|dir| (SourceKind::Local, dir)),
        );
        warnings.extend(lock_local.warnings);
    }

    ResolvedCollectionLock { dirs, warnings }
}

/// Extract the dependency library names from a stdlib `manifest.yaml`.
///
/// The format is uniform across every bundled manifest: a single
/// `dependencies:` top-level key whose entries are `- name: <lib>`. crema
/// reads exactly that shape instead of pulling in a YAML crate. Only
/// `- name:` entries nested under the `dependencies:` key are collected, so
/// a `name:` field belonging to some other top-level key is not mistaken
/// for a dependency. A manifest without a `dependencies` key (or no
/// manifest at all) yields `[]`.
fn parse_manifest_deps(content: &str) -> Vec<String> {
    let mut deps: Vec<String> = Vec::new();
    let mut in_dependencies = false;
    for line in content.lines() {
        // A non-blank line with no leading whitespace opens a new top-level
        // key; entries only count while we are inside `dependencies:`.
        if !line.is_empty() && !line.starts_with(char::is_whitespace) {
            in_dependencies = line.trim_end() == "dependencies:";
            continue;
        }
        if !in_dependencies {
            continue;
        }
        if let Some(rest) = line.trim().strip_prefix("- name:") {
            let name = rest.trim().trim_matches(|c| c == '"' || c == '\'');
            if !name.is_empty() {
                deps.push(name.to_string());
            }
        }
    }
    deps
}
