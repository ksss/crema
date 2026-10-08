//! Gemfile.lock reader for the bundler-mode gem-dir resolver.
//!
//! Bundler decides two different things: *where to look* (install
//! roots — `BUNDLE_PATH`, `.bundle/config`, deployment, `path.system`)
//! and *which gem at which version* (the lockfile). Only the first
//! needs Ruby; crema asks a probe for the roots (see
//! [`parse_probe_output`]) and reads the second from `Gemfile.lock` as
//! text, so a cold run never pays for `bundle exec`'s resolution.
//!
//! The section / source rules are ported from bundler's
//! `LockfileParser` and `Source::{Rubygems,Path,Git}` (4.0.x): a spec
//! line is `    name (version[-platform])` under a `GEM` / `PATH` / `GIT`
//! header; a PATH / GIT gem lives where its `.gemspec` is found by
//! `glob` (default [`DEFAULT_GLOB`]); a GIT checkout is
//! `<bundle_path>/bundler/gems/<base_name>-<revision[0..12]>`.
//! Other sections (`PLUGIN SOURCE`, `DEPENDENCIES`, `CHECKSUMS`, ...)
//! carry no spec crema can place, so their gems read as not locked.

use std::path::{Component, Path, PathBuf};


pub const DEFAULT_GLOB: &str = "{,*,*/*}.gemspec";

/// What the probe reports about bundler's effective configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundlerRoots {
    pub lockfile: PathBuf,
    /// Directory of the Gemfile; PATH sources' `remote:` is relative to it.
    pub bundle_root: PathBuf,
    pub bundle_path: PathBuf,
    /// `Gem.path` after `Bundler.configure`: the only roots searched for
    /// GEM-source gems (never wider than bundler itself).
    pub gem_path: Vec<PathBuf>,
}

/// The probe prints `lockfile=`, `root=`, `bundle_path=` once and
/// `gem_path=` once per root, in Gem.path order.
pub fn parse_probe_output(stdout: &str) -> Result<BundlerRoots, String> {
    let mut lockfile = None;
    let mut bundle_root = None;
    let mut bundle_path = None;
    let mut gem_path = Vec::new();
    for line in stdout.lines().filter(|l| !l.is_empty()) {
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "error: malformed bundler probe output line: {line}"
            ));
        };
        match key {
            "lockfile" => lockfile = Some(PathBuf::from(value)),
            "root" => bundle_root = Some(PathBuf::from(value)),
            "bundle_path" => bundle_path = Some(PathBuf::from(value)),
            "gem_path" => gem_path.push(PathBuf::from(value)),
            _ => {
                return Err(format!(
                    "error: malformed bundler probe output line: {line}"
                ));
            }
        }
    }
    match (lockfile, bundle_root, bundle_path) {
        (Some(lockfile), Some(bundle_root), Some(bundle_path)) => Ok(BundlerRoots {
            lockfile,
            bundle_root,
            bundle_path,
            gem_path,
        }),
        _ => Err("error: incomplete bundler probe output".to_string()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SourceKind {
    Rubygems,
    Path { remote: String },
    Git { remote: String, revision: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Source {
    kind: SourceKind,
    glob: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LockedSpec {
    name: String,
    version: String,
    platform: Option<String>,
    source: usize,
}

impl LockedSpec {
    fn full_name(&self) -> String {
        match &self.platform {
            Some(p) => format!("{}-{}-{}", self.name, self.version, p),
            None => format!("{}-{}", self.name, self.version),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lockfile {
    sources: Vec<Source>,
    specs: Vec<LockedSpec>,
}

/// Outcome of placing one gem on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    /// The lockfile does not list the gem; bundler hides it, so it is a miss.
    NotLocked,
    /// Listed, but no install root holds that version.
    NotInstalled {
        version: String,
    },
    Installed {
        version: String,
        dir: PathBuf,
    },
}

impl Lockfile {
    pub fn parse(text: &str) -> Lockfile {
        let mut lock = Lockfile::default();
        // `None` outside GEM / PATH / GIT (DEPENDENCIES, PLATFORMS, ...).
        let mut current: Option<usize> = None;
        let mut in_specs = false;
        for raw in text.lines() {
            let line = raw.trim_end();
            if line.is_empty() {
                continue;
            }
            if !line.starts_with(' ') {
                in_specs = false;
                let kind = match line {
                    "GEM" => Some(SourceKind::Rubygems),
                    "PATH" => Some(SourceKind::Path {
                        remote: String::new(),
                    }),
                    "GIT" => Some(SourceKind::Git {
                        remote: String::new(),
                        revision: String::new(),
                    }),
                    _ => None,
                };
                current = kind.map(|kind| {
                    lock.sources.push(Source {
                        kind,
                        glob: DEFAULT_GLOB.to_string(),
                    });
                    lock.sources.len() - 1
                });
                continue;
            }
            let Some(idx) = current else { continue };
            if !in_specs {
                if line == "  specs:" {
                    in_specs = true;
                } else if let Some((key, value)) = line.trim_start().split_once(": ") {
                    let source = &mut lock.sources[idx];
                    match (key, &mut source.kind) {
                        ("glob", _) => source.glob = value.to_string(),
                        ("remote", SourceKind::Path { remote })
                        | ("remote", SourceKind::Git { remote, .. }) => {
                            *remote = value.to_string();
                        }
                        ("revision", SourceKind::Git { revision, .. }) => {
                            *revision = value.to_string();
                        }
                        _ => {}
                    }
                }
                continue;
            }
            // Spec rows sit at exactly 4 spaces; 6 spaces is a dependency row.
            let Some(spec_line) = line.strip_prefix("    ") else {
                continue;
            };
            if spec_line.starts_with(' ') {
                continue;
            }
            let Some((name, rest)) = spec_line.split_once(" (") else {
                continue;
            };
            let Some(version_platform) = rest.strip_suffix(')') else {
                continue;
            };
            let (version, platform) = match version_platform.split_once('-') {
                Some((v, p)) => (v, Some(p.to_string())),
                None => (version_platform, None),
            };
            lock.specs.push(LockedSpec {
                name: name.to_string(),
                version: version.to_string(),
                platform,
                source: idx,
            });
        }
        lock
    }

    pub fn locate(&self, name: &str, roots: &BundlerRoots) -> Located {
        let mut candidates: Vec<&LockedSpec> =
            self.specs.iter().filter(|s| s.name == name).collect();
        let Some(first) = candidates.first() else {
            return Located::NotLocked;
        };
        let version = first.version.clone();
        let source = &self.sources[first.source];
        let dir = match &source.kind {
            SourceKind::Rubygems => {
                // A platform variant is the more specific install, so it is
                // tried first within each root; root order stays Gem.path's.
                candidates.sort_by_key(|s| s.platform.is_none());
                roots.gem_path.iter().find_map(|root| {
                    candidates
                        .iter()
                        .map(|s| root.join("gems").join(s.full_name()))
                        .find(|d| d.is_dir())
                })
            }
            SourceKind::Path { remote } => gemspec_dir(
                &lexical_join(&roots.bundle_root, remote),
                &source.glob,
                name,
            ),
            SourceKind::Git { remote, revision } => {
                let rev: String = revision.chars().take(12).collect();
                let checkout = roots.bundle_path.join("bundler/gems").join(format!(
                    "{}-{}",
                    git_base_name(remote),
                    rev
                ));
                gemspec_dir(&checkout, &source.glob, name)
            }
        };
        match dir {
            Some(dir) => Located::Installed { version, dir },
            None => Located::NotInstalled { version },
        }
    }
}

/// `Source::Git#base_name`: `File.basename(uri.sub(%r{^(\w+://)?([^/:]+:)?
/// (//\w*/)?(\w*/)*}, ""), ".git")`. The `sub` matters only for scp-style
/// `user@host:name.git`, where a plain split on `/` would keep the host.
fn git_base_name(uri: &str) -> String {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut rest = uri;
    let word_len = rest.find(|c: char| !is_word(c)).unwrap_or(rest.len());
    if word_len > 0 && rest[word_len..].starts_with("://") {
        rest = &rest[word_len + 3..];
    }
    let host_len = rest.find(['/', ':']).unwrap_or(rest.len());
    if host_len > 0 && rest[host_len..].starts_with(':') {
        rest = &rest[host_len + 1..];
    }
    if let Some(after) = rest.strip_prefix("//") {
        let word_len = after.find(|c: char| !is_word(c)).unwrap_or(after.len());
        if after[word_len..].starts_with('/') {
            rest = &after[word_len + 1..];
        }
    }
    loop {
        let word_len = rest.find(|c: char| !is_word(c)).unwrap_or(rest.len());
        if rest[word_len..].starts_with('/') {
            rest = &rest[word_len + 1..];
        } else {
            break;
        }
    }
    let trimmed = rest.trim_end_matches('/');
    let base = trimmed.rsplit('/').next().unwrap_or(trimmed);
    match base.strip_suffix(".git") {
        Some(stem) if !stem.is_empty() => stem.to_string(),
        _ => base.to_string(),
    }
}

/// `File.expand_path(remote, root)` without touching the filesystem:
/// `.` and `..` are folded lexically, symlinks are left alone.
fn lexical_join(root: &Path, remote: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for component in root.join(remote).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Directory of the `<name>.gemspec` that `glob` finds under `base`
/// (`Source::Path#load_spec_files`). Bundler lets the shallower file win
/// when two gemspecs name the same gem, so shallow first here too.
fn gemspec_dir(base: &Path, glob: &str, name: &str) -> Option<PathBuf> {
    let matcher = globset::GlobBuilder::new(glob)
        .literal_separator(true)
        .build()
        .ok()?
        .compile_matcher();
    let max_depth = if glob.contains("**") {
        8
    } else {
        glob.matches('/').count() + 1
    };
    let wanted = format!("{name}.gemspec");
    let mut best: Option<(usize, PathBuf)> = None;
    let mut stack = vec![(base.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            // Ruby's `*` does not match dotfiles, so dot directories are
            // never entered.
            if file_name.to_string_lossy().starts_with('.') {
                continue;
            }
            let path = entry.path();
            // `is_dir` follows symlinks like Ruby's `Dir.glob`; the dir entry's
            // own file type would not, hiding symlinked monorepo members.
            if path.is_dir() {
                if depth + 1 < max_depth {
                    stack.push((path, depth + 1));
                }
            } else if file_name.to_string_lossy() == wanted
                && path
                    .strip_prefix(base)
                    .is_ok_and(|rel| matcher.is_match(rel))
                // `read_dir` order is unspecified, so equal-depth ties go to
                // the smaller path.
                && best
                    .as_ref()
                    .is_none_or(|(d, b)| (depth, &dir) < (*d, b))
            {
                best = Some((depth, dir.clone()));
            }
        }
    }
    best.map(|(_, dir)| dir)
}
