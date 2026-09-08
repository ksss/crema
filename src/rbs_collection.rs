use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;


const CONFIG_FILE: &str = "rbs_collection.yaml";
const LOCK_FILE: &str = "rbs_collection.lock.yaml";

/// Port of rbs `Collection::Config::Lockfile`. `path` is the collection
/// install directory relative to the lockfile's own directory (rbs:
/// `Lockfile#fullpath = lockfile_dir + path`). crema does not yet
/// resolve git/local sources, so `path` is parsed for round-trip
/// fidelity but only used by the deferred child todos.
#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct Lockfile {
    pub path: PathBuf,
    #[serde(default)]
    pub gemfile_lock_path: Option<PathBuf>,
    #[serde(default)]
    pub gems: Vec<LockGem>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct LockGem {
    pub name: String,
    pub version: String,
    pub source: Source,
}

/// Mirrors rbs `Collection::Sources::*`. `Rubygems`, `Stdlib`, and
/// `Git` are all wired into gem-dir resolution; `Local` is still
/// parsed for schema round-trip but its resolution is deferred to a
/// follow-up.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Source {
    Rubygems,
    Stdlib,
    Git {
        name: String,
        remote: String,
        revision: String,
        #[serde(default)]
        repo_dir: Option<String>,
    },
    Local {
        path: String,
    },
}

/// A parsed lockfile together with the directory it was found in.
/// `lockfile_dir` is needed to resolve `Lockfile.path` (collection
/// install dir) and `Source::Local.path` against. Both are deferred to
/// child todos, but the field is exposed now so downstream code does
/// not need a second discovery pass.
#[derive(Debug, PartialEq, Eq)]
pub struct DiscoveredLockfile {
    pub lockfile: Lockfile,
    pub lockfile_dir: PathBuf,
}

impl DiscoveredLockfile {
    /// Path to the lockfile itself. `lockfile_dir` is the source of
    /// truth and the filename is owned by this module, so callers
    /// that need the file path (cache-staleness checks, error
    /// messages) ask here rather than re-deriving `LOCK_FILE`.
    pub fn lockfile_path(&self) -> PathBuf {
        self.lockfile_dir.join(LOCK_FILE)
    }
}

#[derive(Debug)]
pub enum LockError {
    /// I/O error keyed by the path it failed on and a short label naming
    /// the operation that triggered it. Without these the error message
    /// would just say "cannot read rbs collection lockfile" no matter
    /// whether the failure was canonicalizing the start dir, reading the
    /// lockfile, or checking the `.git` stop marker — all three hit
    /// `io::Error` through different syscalls.
    Io {
        op: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_yaml::Error,
    },
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Io { op, path, source } => write!(
                f,
                "error: cannot {} for rbs collection at {}: {}",
                op,
                path.display(),
                source
            ),
            LockError::Parse { path, source } => write!(
                f,
                "error: invalid rbs collection lockfile {}: {}",
                path.display(),
                source
            ),
        }
    }
}

/// Walk up from `start_dir` looking for `rbs_collection.yaml`, stopping
/// at `.git` (file or directory). When the config is found, read the
/// sibling `rbs_collection.lock.yaml`; if the lock is absent the
/// function returns `Ok(None)` silently (the project has a config but
/// `rbs collection install` was not run — rbs CLI also treats this as
/// "no collection loaded" rather than erroring).
///
/// `.git` is a tighter stop boundary than rbs's filesystem-root walk,
/// matching `Config::discover_walking_up` so monorepo subprojects do
/// not pick up an unrelated parent's lockfile.
///
/// `start_dir` is canonicalized once so `.` becomes absolute;
/// intermediate ancestors are not re-canonicalized to preserve
/// symlinked layouts such as git worktrees (the same rule
/// `Config::discover_walking_up` follows).
pub fn discover_lockfile(start_dir: &Path) -> Result<Option<DiscoveredLockfile>, LockError> {
    let start = start_dir.canonicalize().map_err(|source| LockError::Io {
        op: "canonicalize start dir",
        path: start_dir.to_path_buf(),
        source,
    })?;
    let mut current = start.as_path();
    loop {
        let config_path = current.join(CONFIG_FILE);
        // `try_exists` propagates permission errors as `Err` rather
        // than collapsing them to "not present"; otherwise walk-up
        // would sail past an unreadable config and pick up an
        // unrelated ancestor's lockfile.
        let config_present = config_path.try_exists().map_err(|source| LockError::Io {
            op: "check rbs_collection.yaml presence",
            path: config_path.clone(),
            source,
        })?;
        if config_present {
            let lock_path = current.join(LOCK_FILE);
            return match fs::read_to_string(&lock_path) {
                Ok(content) => {
                    let lockfile = parse_lockfile(&lock_path, &content)?;
                    Ok(Some(DiscoveredLockfile {
                        lockfile,
                        lockfile_dir: current.to_path_buf(),
                    }))
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(source) => Err(LockError::Io {
                    op: "read rbs_collection.lock.yaml",
                    path: lock_path,
                    source,
                }),
            };
        }
        let git_marker = current.join(".git");
        if git_marker.try_exists().map_err(|source| LockError::Io {
            op: "check .git marker",
            path: git_marker.clone(),
            source,
        })? {
            return Ok(None);
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => return Ok(None),
        }
    }
}

/// Load a lockfile from an explicit config path (CLI `--collection CONFIG`
/// or `crema.toml collection_config = "..."`). The lockfile path is
/// derived from `config_path` via `derive_lockfile_path`, matching the
/// rbs/steep convention (`RBS::Collection::Config.to_lockfile_path`).
///
/// Unlike `discover_lockfile`, NotFound on the derived lockfile is a
/// hard error — the user named the config file explicitly and silent
/// fallthrough would mask typos / forgotten `rbs collection install`.
/// crema does not parse the config file itself; it is only used to
/// derive the lockfile path (same as rbs CLI's behavior at the load
/// site — see `RBS::CLI::LibraryOptions#loader`).
///
/// `lockfile_dir` is the parent directory of the derived lockfile;
/// downstream resolution (git/local source dirs) uses it as the base
/// for relative paths inside the lock. When the derived lockfile has
/// no parent component (e.g. `config_path` is a bare filename like
/// `rbs_collection.yaml` and the derived `rbs_collection.lock.yaml`
/// has no parent), `lockfile_dir` falls back to `PathBuf::from(".")`.
pub fn load_from_config(config_path: &Path) -> Result<DiscoveredLockfile, LockError> {
    let lock_path = derive_lockfile_path(config_path);
    let content = fs::read_to_string(&lock_path).map_err(|source| LockError::Io {
        op: "read derived lockfile",
        path: lock_path.clone(),
        source,
    })?;
    let lockfile = parse_lockfile(&lock_path, &content)?;
    let lockfile_dir = lock_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(DiscoveredLockfile {
        lockfile,
        lockfile_dir,
    })
}

/// Port of rbs `RBS::Collection::Config.to_lockfile_path`:
/// `config_path.sub_ext('.lock' + config_path.extname)`. Inserts
/// `.lock` before the final extension (or appends `.lock` when the
/// config has no extension). Examples:
///   `foo.yaml` → `foo.lock.yaml`
///   `foo.yml`  → `foo.lock.yml`
///   `foo`      → `foo.lock`
///   `foo.bar.yaml` → `foo.bar.lock.yaml`
///
/// Trailing-dot edge case: Rust's `Path::extension` treats `foo.` as
/// extension-less, so this returns `foo.lock` — diverging from Ruby's
/// `Pathname("foo.").sub_ext(".lock")` which preserves the dot
/// (`foo..lock`). Filesystem-wise trailing dots are anomalous and not
/// expected in collection config filenames.
fn derive_lockfile_path(config_path: &Path) -> PathBuf {
    let new_ext: std::ffi::OsString = match config_path.extension() {
        Some(ext) => {
            let mut s = std::ffi::OsString::from("lock.");
            s.push(ext);
            s
        }
        None => std::ffi::OsString::from("lock"),
    };
    config_path.with_extension(new_ext)
}

fn parse_lockfile(path: &Path, content: &str) -> Result<Lockfile, LockError> {
    serde_yaml::from_str(content).map_err(|e| LockError::Parse {
        path: path.to_path_buf(),
        source: e,
    })
}

impl Lockfile {
    /// Names of gems whose `source` is `Rubygems`. The lock-derived
    /// load path only resolves these in the minimum-skeleton todo —
    /// other variants are silent-skipped until the child todos fill
    /// them in.
    pub fn rubygems_gem_names(&self) -> Vec<&str> {
        self.gems
            .iter()
            .filter(|g| matches!(g.source, Source::Rubygems))
            .map(|g| g.name.as_str())
            .collect()
    }

    /// `(name, version)` pairs for gems whose `source` is `Rubygems`.
    /// The Ruby gem-dir resolver passes `version` as the second argument
    /// to `Gem::Specification.find_by_name` so the lock pin (not just the
    /// name) drives spec selection — without it a host with multiple
    /// installed versions of the same gem can return a `gem_dir` that
    /// disagrees with the lockfile.
    pub fn rubygems_gem_entries(&self) -> Vec<(&str, &str)> {
        self.gems
            .iter()
            .filter(|g| matches!(g.source, Source::Rubygems))
            .map(|g| (g.name.as_str(), g.version.as_str()))
            .collect()
    }

    /// Names of gems whose `source` is `Stdlib`. The lock entry's
    /// `version` field is intentionally ignored: rbs
    /// `LockfileGenerator#assign_stdlib` always writes `"0"`, and rbs
    /// ships every stdlib sig under a single `stdlib/<name>/0/` dir.
    pub fn stdlib_gem_names(&self) -> Vec<&str> {
        self.gems
            .iter()
            .filter(|g| matches!(g.source, Source::Stdlib))
            .map(|g| g.name.as_str())
            .collect()
    }

    /// `(name, version)` pairs for gems whose `source` is `Git`. Both
    /// components are used verbatim in the on-disk install path
    /// (`<lockfile_dir>/<lockfile.path>/<name>/<version>/`), so the
    /// caller must validate them against path traversal — see
    /// `library_loader::is_valid_lock_path_component`.
    pub fn git_gem_entries(&self) -> Vec<(&str, &str)> {
        self.gems
            .iter()
            .filter(|g| matches!(g.source, Source::Git { .. }))
            .map(|g| (g.name.as_str(), g.version.as_str()))
            .collect()
    }

    /// `(name, version)` pairs for gems whose `source` is `Local`. The
    /// install path matches `Source::Git`
    /// (`<lockfile_dir>/<lockfile.path>/<name>/<version>/`) because rbs
    /// CLI's `Collection::Sources::Local#install` puts a symlink there
    /// pointing at `Source::Local.path`. crema reads the install dir
    /// and `std::fs` follows the symlink, so `Source::Local.path` is
    /// parse-round-trip-only and not exposed here.
    pub fn local_gem_entries(&self) -> Vec<(&str, &str)> {
        self.gems
            .iter()
            .filter(|g| matches!(g.source, Source::Local { .. }))
            .map(|g| (g.name.as_str(), g.version.as_str()))
            .collect()
    }
}
