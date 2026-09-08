use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use globset::{Glob, GlobSet, GlobSetBuilder};

use std::io::{BufWriter, Write};

use crema::config::{CollectionCliOverride, CollectionMode, Config, ConfigError};
use crema::definition_builder::{ConsultationLog, DefinitionBuilder};
use crema::diagnostic::{DiagnosticEmitter, DiagnosticKind, join_did_you_mean};
use crema::gem_dir_cache::ResolvedGemDirs;
use crema::rbs_collection::{self, DiscoveredLockfile};
use crema::type_checker::{CheckOptions, check_source_with_log};
use crema::validator;

mod prompts;

#[derive(Parser)]
#[command(name = "crema", version, about = "An AI-agent-first Ruby type checker")]
struct Cli {
    /// Path to a config file to load instead of `./crema.toml`. When
    /// set, walk-up discovery is bypassed entirely (no merge). Relative
    /// paths inside the file (e.g. `sig = [...]`) are still resolved
    /// from the current working directory, not the config file's dir.
    /// (This asymmetry is intentional: auto-discovery without `--config`
    /// walks up to the nearest `crema.toml` or `.git` boundary and
    /// resolves relative paths against the discovered file's directory.)
    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum DocCommands {
    /// Reference for crema.toml keys
    Config,
    /// Reference for the `.crema/extract.json` schema written by `crema extract`
    Extract,
    /// Reference for diagnostic codes: bare form lists every code as JSONL,
    /// or pass a code (e.g. `Ruby::NoMethod`) to get its markdown doc
    Diagnostic {
        /// Diagnostic code to document, for example Ruby::NoMethod. Omit
        /// to list every known code as JSONL.
        code: Option<String>,
    },
}

#[derive(Subcommand)]
enum InternalSnapshotCommands {
    /// Build the G-layer (gem environment) snapshot and write it to
    /// `.crema/cache/g_snapshot_v1.bin` under the project root
    Dump {
        /// Project root to load the gem environment from
        project_root: PathBuf,
    },
}

#[derive(Subcommand)]
enum InternalCommands {
    /// G-snapshot maintenance
    Snapshot {
        #[command(subcommand)]
        command: InternalSnapshotCommands,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Type check Ruby files against RBS definitions
    Check {
        /// Ruby files to check
        files: Vec<PathBuf>,

        /// Evaluate inline Ruby code (mutually exclusive with files)
        #[arg(short = 'e')]
        eval: Option<String>,

        /// RBS signature directory or `.rbs` file to load (repeatable).
        /// Directories are walked recursively for `.rbs` files; a path
        /// pointing at a single `.rbs` file loads just that file.
        ///
        /// **Replaces** any `sig` list in `crema.toml` (spec Design Goal 4,
        /// CLI overrides config). Use `--add-sig` to append instead. Use
        /// `crema.toml` `sig` for project-wide entries.
        #[arg(long = "sig", value_name = "PATH")]
        sig_dirs: Vec<PathBuf>,

        /// Extra RBS signature directory or `.rbs` file to load
        /// (repeatable), **appended** to the `crema.toml` `sig` list.
        /// Complementary to `--sig`, which replaces. Ignored (with a
        /// warning) when `--sig` is also given.
        #[arg(long = "add-sig", value_name = "PATH")]
        add_sig_dirs: Vec<PathBuf>,

        /// Print type checker decisions to stderr for debugging
        #[arg(long)]
        verbose: bool,

        /// Whether to read `# @rbs` inline annotations from .rb files
        /// (default: true). Set to false to use only sig/ as source of truth.
        /// Note: when false, sig/ should declare all classes/modules used in
        /// the checked .rb files; otherwise unresolved constant diagnostics
        /// will surface.
        #[arg(long, action = clap::ArgAction::Set, num_args = 1)]
        inline: Option<bool>,

        /// Explicit path to an `rbs_collection.yaml` (collection
        /// config). The lockfile (`rbs_collection.lock.yaml` by default)
        /// is derived from this path by inserting `.lock` before the
        /// final extension, matching `RBS::Collection::Config.to_lockfile_path`
        /// (rbs / steep convention). Overrides walk-up discovery and any
        /// `crema.toml collection_config` setting. The derived lockfile
        /// must exist; missing lockfile is a hard error.
        #[arg(long = "collection", value_name = "CONFIG")]
        collection: Option<PathBuf>,

        /// Disable the gem (G-layer) snapshot cache. By default crema
        /// caches the gem environment as a binary snapshot under
        /// `.crema/cache/g_snapshot_v1.bin` and rebuilds from it on warm
        /// runs; passing `--no-g-snapshot` takes the full-rebuild path
        /// on every run (neither reads nor writes the snapshot file).
        /// Escape hatch for isolating regressions and comparing timings
        /// against the pre-snapshot behaviour.
        #[arg(long = "no-g-snapshot")]
        no_g_snapshot: bool,

        /// Skip reading the G-snapshot and rebuild it from the gem RBS
        /// sources (same semantics as an invalidation-key miss). Escape
        /// hatch for stale snapshots when gem sig files change without a
        /// lockfile change (e.g. an edited `.gem_rbs_collection`); the
        /// invalidation key fingerprints the lockfile, not sig contents.
        /// Has no effect when combined with `--no-g-snapshot` since the
        /// snapshot layer is disabled entirely in that mode.
        #[arg(long = "refresh-g-snapshot")]
        refresh_g_snapshot: bool,

        /// The output format is tamped (stably projected).
        /// The keys for each record are reduced to only `file`, `code`,
        /// and `fingerprint`, and are sorted in this order.
        /// This output is ideal for use as a baseline.
        #[arg(long = "tamp")]
        tamp: bool,
    },
    /// Export check-internal facts (definitions, implements,
    /// method-call sites, consulted symbols) as one JSON document at
    /// `.crema/extract.json`; stdout carries a one-line machine-readable
    /// summary. Scope and environment come from crema.toml exactly like
    /// `crema check`. With `-e`, the document for the snippet alone goes
    /// to stdout instead (nothing is written to `.crema/`).
    Extract {
        /// Extract from inline Ruby code instead of the config scope:
        /// the whole document (`files` holding just the `-e`
        /// pseudo-file) is printed to stdout and `.crema/extract.json`
        /// is left untouched. The scope's RBS environment is still
        /// built, so the snippet sees project types.
        #[arg(short = 'e')]
        eval: Option<String>,
    },
    /// Reference documentation (crema.toml keys, diagnostic codes)
    Doc {
        #[command(subcommand)]
        command: DocCommands,
    },
    /// Internal validation commands — not part of the user-facing surface
    #[command(name = "_internal", hide = true)]
    Internal {
        #[command(subcommand)]
        command: InternalCommands,
    },
}

const G_SNAPSHOT_FILE: &str = ".crema/cache/g_snapshot_v1.bin";
// `GEMFILE` / `GEMFILE_LOCK` moved to `crema::bundle_root` so the
// `project_root` walk-up and the flat sites share one source of truth
// for the filenames. `find_gemfile` still walks from `cwd`, so it
// keeps using the shared constant via `use crema::bundle_root::GEMFILE`.
use crema::bundle_root::{GEMFILE, GEMFILE_LOCK, discover_bundle_root};

/// Build the Command used to spawn the Ruby gem-dir resolver. Split out
/// so the bundler-vs-plain-ruby selection can be exercised in unit tests
/// without spawning a child process.
fn build_resolver_command(use_bundler: bool, script: &str) -> std::process::Command {
    let mut cmd = if use_bundler {
        let mut c = std::process::Command::new("bundle");
        c.args(["exec", "ruby", "-e", script]);
        c
    } else {
        let mut c = std::process::Command::new("ruby");
        c.args(["-e", script]);
        c
    };
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd
}

/// Debug-only (`CREMA_DEBUG_RESOLVER=1`): prints the resolver
/// subprocess's argv as a single-quoted, copy-paste-runnable shell
/// command. Single quotes (not double) so the multi-line `-e` script
/// survives literally; the script has no embedded single quotes to
/// escape, but `shell_quote` handles that case anyway for robustness.
fn eprint_resolver_spawn(cmd: &std::process::Command) {
    let mut line = shell_quote(&cmd.get_program().to_string_lossy());
    for arg in cmd.get_args() {
        line.push(' ');
        line.push_str(&shell_quote(&arg.to_string_lossy()));
    }
    eprintln!("resolver: spawn {}", line);
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Escapes backslash/newline/CR/tab so a value dumped under
/// `CREMA_DEBUG_RESOLVER=1` can't be visually mistaken for a line break
/// or field separator in the surrounding debug output (see the stdin
/// dump in `resolve_gem_dirs`, the only site whose text isn't already
/// controlled by crema itself).
fn escape_control_chars(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn find_gemfile(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .map(|dir| dir.join(GEMFILE))
        .find(|path| path.is_file())
}

fn should_use_bundler(bundle_gemfile: Option<&OsStr>, cwd: &Path) -> bool {
    match bundle_gemfile {
        Some(path) if !path.is_empty() => true,
        _ => find_gemfile(cwd).is_some(),
    }
}

/// Print a G-construction warning immediately (so a cold run's stderr
/// is unchanged) and record it verbatim so a later warm `g_snapshot`
/// hit can replay the identical line — G construction (gem resolution,
/// library loading, collection lock) only runs on a miss, so anything
/// it would have warned about must be captured now or never surface
/// again on a warm hit.
fn warn_and_record(warnings: &mut Vec<String>, message: String) {
    eprintln!("{}", message);
    warnings.push(message);
}

/// Resolve gem dirs for `rbs` and every entry in `entries` in a single
/// Ruby invocation. Each entry is `(name, Some(version))` when the
/// caller wants the spec pinned (rbs-collection rubygems-source gems
/// carry their lock pin here), or `(name, None)` when only the name is
/// known (`crema.toml libraries = [...]` and `rbs` itself).
///
/// Entries are written to Ruby's stdin as `name\tversion\n` (version
/// empty when `None`); the script writes `name=path\n` for each. When
/// a pinned lookup misses (`Gem::MissingSpecError`, which also covers
/// the `MissingSpecVersionError` subclass raised for a wrong version),
/// the script retries unpinned and, if that succeeds *and* the
/// fallback actually ships type definitions (a `core/` dir for `rbs`
/// itself, `sig/` for everything else — mirroring rbs's own
/// `EnvironmentLoader.gem_sig_path`'s `path.directory?` gate before its
/// analogous stale-pin self-heal in `add_collection`), reports the
/// self-healed dir with the installed version as a third
/// tab-separated field (`name=path\tinstalled_version`) so the caller
/// can warn about the stale pin. An empty path means neither the
/// pinned lookup nor a usable fallback was found. When Bundler would
/// discover a Gemfile from cwd or `BUNDLE_GEMFILE`, the resolver is
/// spawned under `bundle exec` so bundler-managed gems (vendor/bundle,
/// `BUNDLE_PATH`) are visible and the rbs gem itself resolves to the
/// version the host project pins. If `bundle` is not on PATH, we warn
/// and retry with plain `ruby`.
fn resolve_gem_dirs(
    entries: &[(String, Option<String>)],
    warnings: &mut Vec<String>,
) -> Result<ResolvedGemDirs, String> {
    let debug_resolver = std::env::var_os("CREMA_DEBUG_RESOLVER").is_some_and(|v| v == "1");
    let script = r##"
$stdin.each_line do |line|
  name, version = line.chomp.split("\t", 2)
  next if name.nil? || name.empty?
  pinned = version && !version.empty?
  begin
    spec = pinned ? Gem::Specification.find_by_name(name, version) : Gem::Specification.find_by_name(name)
    puts "#{name}=#{spec.gem_dir}"
  rescue Gem::MissingSpecError
    if pinned
      begin
        fallback = Gem::Specification.find_by_name(name)
        sig_dir = name == "rbs" ? "core" : "sig"
        if File.directory?(File.join(fallback.gem_dir, sig_dir))
          puts "#{name}=#{fallback.gem_dir}\t#{fallback.version}"
          next
        end
      rescue Gem::MissingSpecError
      end
    end
    puts "#{name}="
  end
end
"##;

    let cwd =
        std::env::current_dir().map_err(|e| format!("error: failed to get current dir: {}", e))?;
    let bundle_gemfile = std::env::var_os("BUNDLE_GEMFILE");
    let use_bundler = should_use_bundler(bundle_gemfile.as_deref(), &cwd);
    let mut cmd = build_resolver_command(use_bundler, script);
    if debug_resolver {
        eprint_resolver_spawn(&cmd);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        // Fall back to plain ruby only when `bundle` is missing from PATH.
        // Other spawn errors (EACCES, ENOEXEC, ...) point at a misconfigured
        // bundler install that the user needs to see, not silently work around.
        Err(e) if use_bundler && e.kind() == std::io::ErrorKind::NotFound => {
            warn_and_record(
                warnings,
                format!(
                    "warning: bundle not found on PATH (falling back to ruby): {}",
                    e
                ),
            );
            let mut fallback = build_resolver_command(false, script);
            if debug_resolver {
                eprint_resolver_spawn(&fallback);
            }
            fallback
                .spawn()
                .map_err(|e| format!("error: failed to spawn ruby: {}", e))?
        }
        Err(e) if use_bundler => {
            return Err(format!("error: failed to spawn bundle: {}", e));
        }
        Err(e) => return Err(format!("error: failed to spawn ruby: {}", e)),
    };

    let mut payload = String::new();
    for (name, version) in entries {
        payload.push_str(name);
        payload.push('\t');
        if let Some(v) = version {
            payload.push_str(v);
        }
        payload.push('\n');
    }
    if debug_resolver {
        // Dumped from the structured `entries`, not by re-splitting
        // `payload` on `\n` — a `crema.toml` `libraries` name or a
        // lock-pinned version is unvalidated free text (src/config.rs's
        // `libraries: Vec<String>`), and an embedded newline there would
        // otherwise fabricate an extra, misleading "resolver: stdin"
        // line when troubleshooting via this exact flag.
        for (name, version) in entries {
            eprintln!(
                "resolver: stdin {}\t{}",
                escape_control_chars(name),
                version
                    .as_deref()
                    .map(escape_control_chars)
                    .unwrap_or_default()
            );
        }
    }

    // Hand stdin to a writer thread so `wait_with_output` can drain
    // stdout/stderr in parallel. Without this, a large payload can
    // fill Ruby's stdout pipe buffer (typically 64 KB) before the
    // parent finishes writing stdin, deadlocking both sides.
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "error: ruby stdin unavailable".to_string())?;
    let writer = std::thread::spawn(move || stdin.write_all(payload.as_bytes()));

    let output = child
        .wait_with_output()
        .map_err(|e| format!("error: failed to wait for ruby: {}", e))?;

    writer
        .join()
        .map_err(|_| "error: ruby stdin writer panicked".to_string())?
        .map_err(|e| format!("error: failed to write to ruby stdin: {}", e))?;

    if debug_resolver {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            eprintln!("resolver: stdout {}", line);
        }
        // Unlike the failure path below, this dumps stderr even on
        // success — the whole point of CREMA_DEBUG_RESOLVER is to
        // surface what the normal path silently discards.
        for line in String::from_utf8_lossy(&output.stderr).lines() {
            eprintln!("resolver: stderr {}", line);
        }
        // `ExitStatus`'s own `Display` already reads "exit status: N".
        eprintln!("resolver: {}", output.status);
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("error: ruby exited with failure: {}", stderr));
    }

    let stdout = std::str::from_utf8(&output.stdout)
        .map_err(|e| format!("error: ruby stdout is not UTF-8: {}", e))?;

    let mut rbs_gem_dir: Option<PathBuf> = None;
    let mut libraries: HashMap<String, Option<PathBuf>> = HashMap::new();
    let mut stale: HashMap<String, String> = HashMap::new();
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        let Some((name, rest)) = line.split_once('=') else {
            return Err(format!(
                "error: malformed ruby resolver output line: {}",
                line
            ));
        };
        // A self-healed stale pin carries the installed version as a
        // third tab-separated field (see the resolver script above).
        // The script only ever pairs this with a real path, but the
        // check stays defensive rather than trusting the wire format.
        let (path, installed_version) = match rest.split_once('\t') {
            Some((path, version)) => (path, Some(version)),
            None => (rest, None),
        };
        if name == "rbs" {
            if !path.is_empty() {
                rbs_gem_dir = Some(PathBuf::from(path));
                if let Some(version) = installed_version {
                    stale.insert(name.to_string(), version.to_string());
                }
            }
        } else if path.is_empty() {
            libraries.insert(name.to_string(), None);
        } else {
            libraries.insert(name.to_string(), Some(PathBuf::from(path)));
            if let Some(version) = installed_version {
                stale.insert(name.to_string(), version.to_string());
            }
        }
    }

    let rbs_gem_dir = rbs_gem_dir.ok_or_else(|| {
        format!(
            "warning: rbs gem not found. Install rbs gem to enable type checking.\n{}",
            crema::library_loader::RESOLVER_DEBUG_HINT
        )
    })?;
    Ok(ResolvedGemDirs {
        rbs_gem_dir,
        libraries,
        stale,
    })
}

/// Resolve gem dirs via Ruby every call — no persistent cache. `entries`
/// carries each library as `(name, Some(version))` for lock-pinned
/// rubygems gems or `(name, None)` for crema.toml libraries that have
/// no version information. `rbs` is always added to the resolution set
/// (as `None`, so bundler's pin wins when present) so the core RBS
/// path is part of the same map. `rbs_collection_lock` names the
/// discovered `rbs_collection.lock.yaml` path (if any), used only to
/// name the lockfile in a stale-pin warning message.
///
/// Callers only reach this on a `g_snapshot` miss (`main`'s
/// `g_snapshot_warm` gate) — a warm hit skips G construction entirely,
/// including this resolve, and replays the warnings recorded in the
/// snapshot instead. `warnings` accumulates every line this call warns
/// about so the caller can bake them into that snapshot.
///
/// Baked warnings are only as fresh as `compute_g_snapshot_key`'s
/// invalidation key (crema version, lockfile/crema.toml content, sig
/// paths) — none of which change when the *installed* gem state does
/// (`bundle install` with an already-correct lockfile, or a plain `gem
/// install`). A resolved-but-later-fixed "bundle not found on PATH" or
/// stale-pin warning keeps replaying verbatim until something touches
/// one of those key inputs or `--refresh-g-snapshot` is passed. This
/// mirrors the pre-existing tradeoff for the G layer's actual RBS
/// content (a warm hit never re-reads gem sig files either); it is not
/// new to warning replay.
fn get_gem_dirs(
    entries: &[(String, Option<String>)],
    rbs_collection_lock: Option<&Path>,
    warnings: &mut Vec<String>,
) -> Result<ResolvedGemDirs, String> {
    // `entries` already carries a pinned `rbs` tuple when the lock file
    // lists rbs itself as a rubygems source — prepending an unpinned
    // `("rbs", None)` in that case would send two `rbs` lines to the
    // resolver and, before the self-heal fix above, discard a
    // successful unpinned resolution behind a later empty pinned one.
    let resolver_input: Vec<(String, Option<String>)> =
        if entries.iter().any(|(name, _)| name == "rbs") {
            entries.to_vec()
        } else {
            std::iter::once(("rbs".to_string(), None))
                .chain(entries.iter().cloned())
                .collect()
        };
    let resolved = resolve_gem_dirs(&resolver_input, warnings)?;
    emit_stale_pin_warnings(&resolved, entries, rbs_collection_lock, warnings);
    Ok(resolved)
}

/// Record one warning per lock-pinned gem whose exact pinned version
/// isn't installed but a self-healed fallback (installed) version
/// resolved instead (`gem_dirs.stale`, populated by `resolve_gem_dirs`).
fn emit_stale_pin_warnings(
    gem_dirs: &ResolvedGemDirs,
    entries: &[(String, Option<String>)],
    rbs_collection_lock: Option<&Path>,
    warnings: &mut Vec<String>,
) {
    if gem_dirs.stale.is_empty() {
        return;
    }
    let lock_name = rbs_collection_lock
        .and_then(|p| p.file_name())
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "rbs_collection.lock.yaml".to_string());
    let sync_command = collection_update_command();

    let mut names: Vec<&String> = gem_dirs.stale.keys().collect();
    names.sort();
    for name in names {
        let Some(pin_version) = entries
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.as_deref())
        else {
            continue;
        };
        let installed_version = &gem_dirs.stale[name];
        warn_and_record(
            warnings,
            format!(
                "warning: {} pins {} {} but {} is installed. Run `{}` to sync.",
                lock_name, name, pin_version, installed_version, sync_command
            ),
        );
    }
}

/// The command to suggest for re-syncing a stale lock pin: bundler
/// wraps it when a Gemfile is in play *and* `bundle` is actually on
/// PATH. A Gemfile alone isn't enough — `resolve_gem_dirs` falls back
/// to plain `ruby` (with its own warning) when `bundle` is missing, so
/// suggesting `bundle exec ...` in that case would tell the user to
/// run a command that was just shown not to work.
fn collection_update_command() -> &'static str {
    let use_bundler = std::env::current_dir()
        .is_ok_and(|cwd| should_use_bundler(std::env::var_os("BUNDLE_GEMFILE").as_deref(), &cwd))
        && bundle_on_path(std::env::var_os("PATH").as_deref());
    if use_bundler {
        "bundle exec rbs collection update"
    } else {
        "rbs collection update"
    }
}

/// Cheap PATH scan for a `bundle` executable — mirrors the check
/// `Command::new("bundle").spawn()` performs internally, without
/// spawning a process, so `collection_update_command` can agree with
/// `resolve_gem_dirs`'s own NotFound fallback.
fn bundle_on_path(path_env: Option<&OsStr>) -> bool {
    let Some(paths) = path_env else {
        return false;
    };
    std::env::split_paths(paths).any(|dir| dir.join("bundle").is_file())
}

/// Prints `collection_lock.warnings` to stderr and records each printed
/// line into `warnings` (see `warn_and_record`), adding
/// `library_loader::RESOLVER_DEBUG_HINT` once (not once per gem) right
/// after the first genuinely-missing rubygems-lock entry. The other
/// warning shapes in this list (stdlib/git/local "not found at <path>",
/// a claimed-but-missing sig) point at different remedies, so they
/// don't trigger the hint.
fn print_collection_lock_warnings(warnings: &mut Vec<String>, collection_warnings: &[String]) {
    let mut hinted = false;
    for w in collection_warnings {
        warn_and_record(warnings, w.clone());
        if !hinted && w.starts_with("warning: rbs collection: gem '") && w.ends_with("' not found")
        {
            warn_and_record(
                warnings,
                crema::library_loader::RESOLVER_DEBUG_HINT.to_string(),
            );
            hinted = true;
        }
    }
}

/// Reject sig paths that don't exist, aren't directories, and aren't `.rbs`
/// files. `label` names the source ("--sig" or "crema.toml sig") so the error
/// attributes back to the user-facing input that produced it. A single
/// `metadata()` call avoids the two-syscall `exists() + is_dir()` pattern.
fn validate_sig_dirs(dirs: &[PathBuf], label: &str) {
    for dir in dirs {
        match fs::metadata(dir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) if dir.extension().is_some_and(|ext| ext == "rbs") => {}
            Ok(_) => {
                eprintln!(
                    "error: {} path is not a directory or .rbs file: {}",
                    label,
                    dir.display()
                );
                process::exit(2);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("error: {} path not found: {}", label, dir.display());
                process::exit(2);
            }
            Err(e) => {
                eprintln!(
                    "error: {} path is not accessible: {}: {}",
                    label,
                    dir.display(),
                    e
                );
                process::exit(2);
            }
        }
    }
}

/// Walk `dir` recursively and append every `.rb` file into `out`. Kept
/// structurally symmetric with `EnvironmentDraft::load_dir` (the `.rbs`
/// walker); read errors warn and skip the subtree rather than abort, since
/// a permission glitch in one directory should not kill the whole check.
/// Resolve CLI `check` arguments into a flat list of `.rb` files: directories
/// are walked recursively (`.rb` only), files pass through unchanged, and a
/// missing path is fatal so the user sees their typo. Exits the process on
/// I/O errors rather than returning, matching the surrounding CLI style.
fn expand_targets(targets: &[PathBuf]) -> Vec<PathBuf> {
    let mut expanded = Vec::new();
    for target in targets {
        match fs::metadata(target) {
            Ok(m) if m.is_dir() => {
                collect_rb_files(target, &mut expanded);
            }
            Ok(_) => expanded.push(target.clone()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("error: file not found: {}", target.display());
                process::exit(2);
            }
            Err(e) => {
                eprintln!("error: cannot access {}: {}", target.display(), e);
                process::exit(2);
            }
        }
    }
    expanded
}

/// Recursively collect every `.rb` file under `dir` into `out`. An
/// unreadable directory or entry warns and is skipped — a partial walk
/// under-represents `out` rather than aborting the run.
fn collect_rb_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: cannot read directory {}: {}", dir.display(), e);
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("warning: cannot read entry in {}: {}", dir.display(), e);
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            collect_rb_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rb") {
            out.push(path);
        }
    }
}

/// Canonicalize `path` (resolve `.`/`..`/symlinks) or exit 2. Existence
/// is assumed already validated by the caller (`expand_targets`
/// / `fs::metadata`); a failure here means a race (removed between the
/// existence check and this call) rather than a user typo, but it still
/// gets the same fatal treatment as every other CLI path error.
fn canonicalize_or_exit(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|e| {
        eprintln!("error: cannot resolve {}: {}", path.display(), e);
        process::exit(2);
    })
}

/// ADR-0029 §4: print the copy-pasteable "add a `check` field" error and
/// exit 2. `config_missing` prepends a line naming the more specific
/// cause (no crema.toml found at all, vs. one found without `check`)
/// and, in that case, appends a hint pointing to zero-config `-e` — the
/// only trial-run path that works without crema.toml (ADR-0029 §5
/// amendment).
fn print_no_check_target_configured_and_exit(config_missing: bool) -> ! {
    if config_missing {
        eprintln!("error: no crema.toml found");
        eprintln!();
    }
    eprintln!("error: no check target configured");
    eprintln!();
    eprintln!("Add a `check` field to crema.toml:");
    eprintln!();
    eprintln!("    check = [\"app\", \"lib\"]");
    eprintln!();
    eprintln!("See https://github.com/ksss/crema for details.");
    if config_missing {
        eprintln!();
        eprintln!("hint: to try crema without a config file, use `-e`:");
        eprintln!();
        eprintln!("    crema check -e '1 + 1'");
    }
    process::exit(2);
}

/// Compile `patterns` (already glob-syntax-validated by
/// `Config::resolve_with_cli_collecting_warnings`) into a matcher.
fn build_glob_set(patterns: &[String]) -> GlobSet {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            Glob::new(pattern).expect(
                "glob syntax was validated in Config::resolve_with_cli_collecting_warnings",
            ),
        );
    }
    builder
        .build()
        .expect("individually-valid globs always compose into a valid GlobSet")
}

/// `ignore`/`sig_ignore` glob patterns match against `path` relative to
/// `project_root` (the directory `crema.toml` was found in — the same
/// basis `check`/`sig` entries resolve against), not `path` verbatim or
/// cwd-relative. Falls back to matching `path` as given when it does not
/// live under `project_root` (CLI `--sig`/`--add-sig` entries are
/// cwd-relative, an existing asymmetry this does not attempt to fix).
///
/// A `patterns` entry with no glob metacharacter additionally matches
/// every path nested under it, component-wise (`rel.starts_with`, not a
/// string prefix — `"lib/rbs/test"` must not match the sibling
/// `"lib/rbs/tester.rb"`). This is disk-independent (no `fs::metadata`
/// call, unlike `check`'s own directory expansion) and mirrors how
/// Steepfile's `ignore` treats a bare directory name — so a metachar-free
/// entry in a Steepfile `ignore` list carries over verbatim. A glob-syntax
/// entry does not: `globset`'s `*` does not cross `/` the way Steep's
/// `Pattern` recursively expands every entry, so a Steepfile entry with a
/// wildcard may still need widening by hand.
fn is_ignored(path: &Path, project_root: &Path, globs: &GlobSet, patterns: &[String]) -> bool {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    if globs.is_match(rel) {
        return true;
    }
    patterns
        .iter()
        .filter(|p| !has_glob_metachar(p))
        .any(|p| rel.starts_with(p))
}

/// Whether `pattern` uses `globset`/glob syntax rather than naming a
/// literal path. Deliberately simple (no escape-sequence awareness) —
/// when in doubt this should say "yes it's a glob", since [`is_ignored`]'s
/// plain-path directory-prefix matching is additive to glob matching, not
/// a replacement for it.
fn has_glob_metachar(pattern: &str) -> bool {
    pattern.contains(['*', '?', '[', '{'])
}

/// ADR-0029 §3: turn CLI positional targets into a diagnostic-output
/// filter over `scope` (the expanded, canonicalized, `ignore`-subtracted
/// environment-scope file list). Returns `None` for bare `crema check`
/// (no filtering).
///
/// A file target must resolve inside `scope` — exit 2 otherwise ("`<path>`
/// must be a subset of the configured scope", ADR-0029 §3). When the
/// target is a member of `ignored_files` (a file that was part of
/// `check`'s scope before `ignore` subtracted it — not just any path
/// that happens to match the glob, which could name something never in
/// scope to begin with), the error names that cause instead of the
/// generic "outside scope" message — the todo's decided semantics treat
/// an explicit ignored-file target the same way `git add` rejects a
/// `.gitignore`d path: no silent skip. A directory target is expanded
/// recursively and silently intersected with `scope` — files outside
/// scope (ignored or otherwise) under that directory are dropped rather
/// than rejected (directories are permissive, files are strict, per the
/// ADR). A target that does not exist on disk exits 2, matching
/// `expand_targets`'s existing NotFound handling.
fn build_diagnostic_filter(
    cli_targets: &[PathBuf],
    scope: &[PathBuf],
    ignored_files: &HashSet<PathBuf>,
) -> Option<HashSet<PathBuf>> {
    if cli_targets.is_empty() {
        return None;
    }
    let scope_set: HashSet<PathBuf> = scope.iter().cloned().collect();
    let mut filter = HashSet::new();
    for target in cli_targets {
        let metadata = match fs::metadata(target) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("error: file not found: {}", target.display());
                process::exit(2);
            }
            Err(e) => {
                eprintln!("error: cannot access {}: {}", target.display(), e);
                process::exit(2);
            }
        };
        if metadata.is_dir() {
            let mut dir_files = Vec::new();
            collect_rb_files(target, &mut dir_files);
            for f in dir_files {
                let canon = canonicalize_or_exit(&f);
                if scope_set.contains(&canon) {
                    filter.insert(canon);
                }
            }
        } else {
            let canon = canonicalize_or_exit(target);
            if !scope_set.contains(&canon) {
                if ignored_files.contains(&canon) {
                    eprintln!(
                        "error: {} is ignored (matches a crema.toml `ignore` entry)",
                        target.display()
                    );
                } else {
                    eprintln!(
                        "error: {} is outside the configured check scope",
                        target.display()
                    );
                }
                process::exit(2);
            }
            filter.insert(canon);
        }
    }
    Some(filter)
}

/// Discover an `rbs_collection.lock.yaml` by walking up from cwd,
/// stopping at `.git`. Errors are formatted into a single string so
/// the caller can route them through the same `error:`/`warning:`
/// stderr convention as `get_gem_dirs`. A missing config (no
/// `rbs_collection.yaml` along the walk) and a config without a lock
/// both return `Ok(None)` silently — rbs CLI parity.
fn discover_lockfile_from_cwd() -> Result<Option<DiscoveredLockfile>, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("error: cannot read current dir: {}", e))?;
    rbs_collection::discover_lockfile(&cwd).map_err(|e| e.to_string())
}

/// Load file config according to the top-level `--config <PATH>` flag.
/// When the flag is set, the named path is loaded verbatim (NotFound is
/// a hard exit-2 error). When absent, fall back to cwd discovery
/// (NotFound silently yields None).
fn load_file_config(cli_config: Option<&Path>) -> Option<Config> {
    load_file_config_with_dir(cli_config).0
}

fn load_file_config_with_dir(cli_config: Option<&Path>) -> (Option<Config>, PathBuf) {
    let result = match cli_config {
        Some(path) => Config::load_from_file(path).map(|cfg| {
            let dir = if path.is_absolute() {
                path.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."))
            } else {
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                cwd.join(path)
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or(cwd)
            };
            (Some(cfg), dir)
        }),
        None => {
            let cwd = std::env::current_dir().map_err(ConfigError::Io);
            match cwd {
                Ok(cwd) => Config::discover_walking_up(&cwd).map(|found| match found {
                    Some((cfg, dir)) => (Some(cfg), dir),
                    None => (None, cwd),
                }),
                Err(err) => Err(err),
            }
        }
    };
    match result {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            process::exit(2);
        }
    }
}

/// G-snapshot invalidation key over the current run's inputs (ADR-0028
/// Decision 4: lockfiles, crema version, crema.toml, sig path set).
/// All project resources (Gemfile.lock, crema.toml) are read from
/// `project_root` so subdirectory execution and root execution compute
/// the same key.
fn compute_g_snapshot_key(
    sig_dirs: &[PathBuf],
    rbs_collection_lock: Option<&Path>,
    project_root: &Path,
) -> crema::snapshot::invalidation::InvalidationKey {
    let gemfile_lock_content =
        discover_bundle_root(project_root).and_then(|dir| fs::read(dir.join(GEMFILE_LOCK)).ok());
    let rbs_collection_lock_content = rbs_collection_lock.and_then(|p| fs::read(p).ok());
    let crema_toml_content = fs::read(project_root.join("crema.toml")).ok();
    let mut sig_paths: Vec<String> = sig_dirs
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    sig_paths.sort();
    let sig_path_refs: Vec<&str> = sig_paths.iter().map(String::as_str).collect();
    let version = crema::snapshot::invalidation::crema_version();
    crema::snapshot::invalidation::compute(&crema::snapshot::invalidation::InvalidationInputs {
        crema_version: &version,
        gemfile_lock_content: gemfile_lock_content.as_deref(),
        rbs_collection_lock_content: rbs_collection_lock_content.as_deref(),
        crema_toml_content: crema_toml_content.as_deref(),
        sig_paths: &sig_path_refs,
    })
}

/// Recursively collect every `.rbs` file under `dir` into `out`, same walk
/// order as [`crema::environment::draft::EnvironmentDraft::load_dir`] but
/// enumeration-only (no read/parse) — ADR-0028 S8 change detection needs
/// the current path set, not file content.
fn collect_rbs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rbs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rbs") {
            out.push(path);
        }
    }
}

/// Every `.rbs` file `--sig` would load: each `sig_dirs` entry expanded the
/// same way the cold ingest loop resolves it (`is_dir` → recursive walk,
/// otherwise the file itself).
fn sig_file_targets(sig_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in sig_dirs {
        if path.is_dir() {
            collect_rbs_files(path, &mut out);
        } else {
            out.push(path.clone());
        }
    }
    out
}

/// `_internal snapshot dump`: load the project's gem environment (rbs
/// core + crema.toml libraries + rbs collection — the G layer, no sig/
/// and no inline), freeze it, and write the snapshot with its
/// invalidation key to `.crema/cache/g_snapshot_v1.bin`.
///
/// This is the write-path validation vehicle for ADR-0028 slice 1b:
/// the check pipeline never calls it; slice 1c wires reading and decides
/// when checks write snapshots. The gem-loading sequence intentionally
/// mirrors `Commands::Check` (kept duplicated because slice scope
/// forbids touching the check path; slice 1c reconciles the two).
fn run_snapshot_dump(project_root: &Path, cli_config: Option<&Path>) {
    if let Err(e) = std::env::set_current_dir(project_root) {
        eprintln!(
            "error: cannot enter project root {}: {}",
            project_root.display(),
            e
        );
        process::exit(2);
    }

    let file_config = load_file_config(cli_config);
    let resolved = match Config::resolve_with_cli_collecting_warnings(
        file_config,
        &[],
        &[],
        None,
        CollectionCliOverride::Absent,
    ) {
        Ok((r, warnings)) => {
            for w in &warnings {
                eprintln!("{}", w);
            }
            r
        }
        Err(e) => {
            eprintln!("{}", e);
            process::exit(2);
        }
    };

    let lockfile = match &resolved.collection {
        CollectionMode::Auto => match discover_lockfile_from_cwd() {
            Ok(opt) => opt,
            Err(msg) => {
                eprintln!("{}", msg);
                process::exit(2);
            }
        },
        CollectionMode::ConfigPath(p) => match rbs_collection::load_from_config(p) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("{}", e);
                process::exit(2);
            }
        },
    };

    let lock_rubygems_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .rubygems_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let lock_stdlib_names: Vec<String> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .stdlib_gem_names()
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();
    let lock_git_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .git_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let lock_local_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .local_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut all_entries: Vec<(String, Option<String>)> = resolved
        .libraries
        .iter()
        .map(|name| (name.clone(), None))
        .collect();
    {
        let mut added: HashSet<String> = resolved.libraries.iter().cloned().collect();
        for (name, version) in &lock_rubygems_entries {
            if added.insert(name.clone()) {
                all_entries.push((name.clone(), Some(version.clone())));
            }
        }
    }

    let mut draft = crema::environment::draft::EnvironmentDraft::new();
    let rbs_collection_lock_path = lockfile.as_ref().map(|d| d.lockfile_path());
    let mut g_warnings: Vec<String> = Vec::new();
    // Every directory `draft.load_dir` is called on below, so the
    // freshness manifest baked into the snapshot can be stat-rescanned
    // on a later warm attempt (`g_snapshot_g_layer_freshness_check`).
    let mut g_dirs: Vec<PathBuf> = Vec::new();
    match get_gem_dirs(
        &all_entries,
        rbs_collection_lock_path.as_deref(),
        &mut g_warnings,
    ) {
        Ok(gem_dirs) => {
            let core_dir = gem_dirs.rbs_gem_dir.join("core");
            if core_dir.is_dir() {
                g_dirs.push(core_dir.clone());
                if let Err(e) = draft.load_dir(&core_dir) {
                    warn_and_record(
                        &mut g_warnings,
                        format!("warning: failed to load core RBS: {}", e),
                    );
                }
            }

            let libs = crema::library_loader::resolve(&gem_dirs, &resolved.libraries);
            for w in &libs.warnings {
                warn_and_record(&mut g_warnings, w.clone());
            }
            for dir in &libs.dirs {
                g_dirs.push(dir.clone());
                if let Err(e) = draft.load_dir(dir) {
                    warn_and_record(
                        &mut g_warnings,
                        format!("warning: failed to load library {}: {}", dir.display(), e),
                    );
                }
            }

            let lock_rubygems_refs: Vec<&str> = lock_rubygems_entries
                .iter()
                .map(|(n, _)| n.as_str())
                .collect();
            let lock_stdlib_refs: Vec<&str> =
                lock_stdlib_names.iter().map(String::as_str).collect();
            let lock_git_refs: Vec<(&str, &str)> = lock_git_entries
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let git_source =
                lockfile
                    .as_ref()
                    .map(|discovered| crema::library_loader::GitLockSource {
                        lockfile_dir: &discovered.lockfile_dir,
                        lockfile_path: &discovered.lockfile.path,
                        entries: &lock_git_refs,
                    });
            let lock_local_refs: Vec<(&str, &str)> = lock_local_entries
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let local_source =
                lockfile
                    .as_ref()
                    .map(|discovered| crema::library_loader::LocalLockSource {
                        lockfile_dir: &discovered.lockfile_dir,
                        lockfile_path: &discovered.lockfile.path,
                        entries: &lock_local_refs,
                    });
            let collection_lock = crema::library_loader::resolve_collection_lock(
                &gem_dirs,
                &libs.claimed,
                &lock_rubygems_refs,
                &lock_stdlib_refs,
                git_source,
                local_source,
            );
            print_collection_lock_warnings(&mut g_warnings, &collection_lock.warnings);
            for (source, dir) in &collection_lock.dirs {
                g_dirs.push(dir.clone());
                if let Err(e) = draft.load_dir(dir) {
                    warn_and_record(
                        &mut g_warnings,
                        format!(
                            "warning: failed to load {} {}: {}",
                            source.load_error_label(),
                            dir.display(),
                            e
                        ),
                    );
                }
            }
        }
        Err(msg) => {
            let is_error = msg.starts_with("error:");
            warn_and_record(&mut g_warnings, msg);
            if is_error {
                process::exit(2);
            }
        }
    }

    // Stat the G-layer directories now, right after the load loop above
    // finished reading them — not after `draft.build()`, which can run
    // long enough that a concurrent edit lands in the gap and gets
    // stat-scanned as "fresh" for content `build()` never saw. Scanning
    // here means any such race instead makes the *next* run's manifest
    // comparison miss (one wasted rebuild, self-correcting) rather than
    // baking a stale-forever pair into this snapshot.
    let manifest = crema::snapshot::freshness::scan(&g_dirs);

    let frozen = match draft.build() {
        Ok(env) => std::sync::Arc::new(env),
        Err((e, names)) => {
            eprintln!(
                "error: failed to build environment: {}",
                e.format_with(&names)
            );
            process::exit(2);
        }
    };

    let key = compute_g_snapshot_key(
        &resolved.sig_dirs,
        rbs_collection_lock_path.as_deref(),
        project_root,
    );

    let cache_path = project_root.join(G_SNAPSHOT_FILE);
    let cache_path = cache_path.as_path();
    if let Err(e) =
        crema::snapshot::write::write_g_snapshot(cache_path, &key, &frozen, &g_warnings, &manifest)
    {
        eprintln!(
            "error: failed to write snapshot {}: {}",
            cache_path.display(),
            e
        );
        process::exit(2);
    }
    let written = fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "wrote {} bytes to {}, key={}",
        written,
        cache_path.display(),
        key.to_hex()
    );
}

/// Render one verify-mode divergence (ADR-0032 Decision 6) to stderr,
/// with enough detail to identify the recording bug without rerunning:
/// the file, both pre-join diagnostic lists, every lost query by name,
/// and stale stored-only projection hashes.
fn report_verify_divergence(
    path: &str,
    stored: &crema::incremental::FileEntry,
    fresh: &[crema::diagnostic::Diagnostic],
    div: &crema::incremental::VerifyDivergence,
) {
    let json = |d: &crema::diagnostic::Diagnostic| {
        serde_json::to_string(d).unwrap_or_else(|_| format!("{d:?}"))
    };
    eprintln!("incremental verify: divergence in {path}");
    if div.diagnostics_differ {
        eprintln!(
            "  diagnostics differ: {} stored vs {} fresh (the replay would have been wrong)",
            stored.diagnostics.len(),
            fresh.len()
        );
        for d in &stored.diagnostics {
            eprintln!("    stored: {}", json(d));
        }
        for d in fresh {
            eprintln!("    fresh:  {}", json(d));
        }
    }
    for k in &div.missing_keys {
        eprintln!("  query consulted by the fresh check but missing from the stored set: {k:?}");
    }
    for h in &div.stale_hashes {
        eprintln!("  stored consulted hash with no fresh counterpart: {h:#018x}");
    }
}

/// Everything `Commands::Check` carries, bundled so `run_check` can be
/// shared by `crema check` and `crema extract` — the latter runs the
/// same environment-building pipeline with default flags, a suppressed
/// diagnostic stream, and its own per-file phase (see the [`RunMode`]
/// branch after the env phase inside `run_check`).
struct CheckInvocation {
    files: Vec<PathBuf>,
    eval: Option<String>,
    sig_dirs: Vec<PathBuf>,
    add_sig_dirs: Vec<PathBuf>,
    verbose: bool,
    inline: Option<bool>,
    collection: Option<PathBuf>,
    no_g_snapshot: bool,
    refresh_g_snapshot: bool,
    tamp: bool,
}

/// Which per-file phase `run_check` runs after the shared env phase.
/// `Check` streams diagnostics; `Extract` suppresses them (the emitter
/// writes to a sink) and runs its own collection + output.
enum RunMode {
    Check,
    Extract,
}

/// The `crema check` pipeline (config → gem env → sig/inline ingest →
/// build → validate → per-file check), extracted from `main`'s match arm
/// verbatim so `crema extract` can share every phase up to and
/// including env construction. `mode` switches the output contract:
/// `Extract` discards diagnostics (the emitter writes to a sink) and
/// runs its own per-file phase.
fn run_check(cli_config: Option<&Path>, args: CheckInvocation, mode: RunMode) {
    // Anchors the check-timing line: `setup` is everything before
    // ingest (config, collection, gem dirs, snapshot open).
    let t_run = std::time::Instant::now();
    let CheckInvocation {
        files: cli_targets,
        eval,
        sig_dirs,
        add_sig_dirs,
        verbose,
        inline,
        collection,
        no_g_snapshot,
        refresh_g_snapshot,
        tamp,
    } = args;
    // Captured once, ahead of `eval`'s move into the `sources`
    // construction below (ADR-0029 §5) — every later gate reads
    // this instead of re-borrowing the (by-then-moved) `eval`.
    let eval_active = eval.is_some();
    if eval_active && !cli_targets.is_empty() {
        eprintln!("error: -e and files are mutually exclusive");
        process::exit(2);
    }

    let (file_config, project_root) = load_file_config_with_dir(cli_config);

    // ADR-0029 §4: the environment scope now lives in crema.toml's
    // `check` field, so a missing crema.toml or a missing `check`
    // field is a hard error — there is no more "no files
    // specified" fallback message (bare `crema check` is now the
    // primary entry point, scoped entirely by config).
    //
    // ADR-0029 §5 amendment: zero-config `-e` is the one exception
    // — a `crema check -e '<code>'` invocation with NO crema.toml
    // proceeds with an empty scope (no snapshot persistence,
    // no `.crema/` writes; see `has_config` gating below). The
    // trial-run UX would otherwise reject first-contact users
    // with an exit-2 wall, violating the zero-config trial-run goal.
    // A crema.toml that exists but lacks `check` is NOT covered
    // by this exception — the user is clearly set up with a
    // config file and deserves the config-completeness error.
    let has_config = file_config.is_some();
    match &file_config {
        None if !eval_active => print_no_check_target_configured_and_exit(true),
        None => { /* zero-config -e: fall through with empty scope */ }
        Some(cfg) if cfg.check.is_none() => print_no_check_target_configured_and_exit(false),
        _ => {}
    }

    // ADR-0029 §5 amendment: zero-config `-e` must not write to
    // `.crema/` — a trial-run invocation is invasively-hostile
    // otherwise (user tried crema once, now has an unexplained
    // cache directory in cwd). Force the G-snapshot write off by
    // shadowing the opt-in flag; `get_gem_dirs` itself never
    // writes to disk (no persistent gem-dir cache), so this is
    // the only gate needed.
    let no_g_snapshot = no_g_snapshot || !has_config;

    if let Some(cfg) = &file_config {
        validate_sig_dirs(&cfg.sig, "crema.toml sig");
    }
    validate_sig_dirs(&sig_dirs, "--sig");
    validate_sig_dirs(&add_sig_dirs, "--add-sig");

    // `--collection CONFIG` is treated as cwd-relative and
    // absolutized so downstream code does not need to know
    // about the CLI's working directory.
    let cli_collection = if let Some(p) = collection {
        let absolute = if p.is_absolute() {
            p
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(p),
                Err(e) => {
                    eprintln!("error: cannot read current dir: {}", e);
                    process::exit(2);
                }
            }
        };
        CollectionCliOverride::ConfigPath(absolute)
    } else {
        CollectionCliOverride::Absent
    };

    let mut resolved = match Config::resolve_with_cli_collecting_warnings(
        file_config,
        &sig_dirs,
        &add_sig_dirs,
        inline,
        cli_collection,
    ) {
        Ok((r, warnings)) => {
            for w in &warnings {
                eprintln!("{}", w);
            }
            r
        }
        Err(e) => {
            eprintln!("{}", e);
            process::exit(2);
        }
    };

    let ignore_set = build_glob_set(&resolved.ignore);
    // `sig_ignore` subtracts here, once, regardless of a sig
    // entry's origin (config `sig` / `--sig` / `--add-sig`,
    // already merged into `resolved.sig_dirs`) — every
    // downstream consumer (every one of which reads
    // `resolved.sig_dirs`) sees the same already-filtered,
    // already-flattened file list for free.
    //
    // Gated on a non-empty `sig_ignore`: flattening directory
    // entries into individual files also changes a malformed
    // `.rbs`'s fault-isolation granularity (a bad file now only
    // drops its own decls, not its whole directory's — `load_dir`
    // aborts its recursive walk on the first `Err`, `load_file`
    // doesn't). That's an unrelated side effect an unconfigured
    // `sig_ignore` must not introduce; every project without it
    // keeps the pre-existing `draft.load_dir` per-directory
    // behavior untouched below.
    if !resolved.sig_ignore.is_empty() {
        let sig_ignore_set = build_glob_set(&resolved.sig_ignore);
        let mut dropped = Vec::new();
        resolved.sig_dirs = sig_file_targets(&resolved.sig_dirs)
            .into_iter()
            .filter(|p| {
                let ignored = is_ignored(p, &project_root, &sig_ignore_set, &resolved.sig_ignore);
                if ignored {
                    dropped.push(p.clone());
                }
                !ignored
            })
            .collect();
        // Surfaced regardless of a dropped entry's origin (config
        // `sig` vs. `--sig`/`--add-sig`) — an explicit `--sig
        // <file>` matching `sig_ignore` would otherwise vanish
        // with no diagnostic, indistinguishable from a typo.
        for p in &dropped {
            eprintln!("warning: {} is ignored by sig_ignore", p.display());
        }
    }

    // ADR-0029 §1: the environment scope is `resolved.check`
    // (guaranteed `Some` — checked above, before `file_config` was
    // moved into `resolve_with_cli_collecting_warnings`), expanded
    // the same way CLI targets used to be and canonicalized so
    // every downstream file identity (sources, diagnostics,
    // snapshot fingerprints) agrees on one representation
    // regardless of how the CLI filter argument was spelled
    // (`./foo.rb` vs `foo.rb`, symlinked dirs, etc — ADR-0029
    // slice S3 design point 5's "most bug-prone" seam).
    //
    // ADR-0029 §5 (slice S4): `-e` mode builds this exact same
    // scope — the eval code is a separate, unpersisted source
    // added on top (see the `sources` construction below), never
    // a substitute for it. This is what lets `-e` see project
    // classes/sig exactly like `crema check` does.
    // ADR-0029 §5 amendment: zero-config `-e` (crema.toml absent)
    // has no `check` scope at all — the eval pseudo-source is the
    // only thing type-checked. `unwrap_or_default` gives that path
    // an empty Vec; the ADR-0029 gate above still guarantees
    // `Some(_)` in every other invocation.
    let check_field = resolved.check.clone().unwrap_or_default();
    let expanded = expand_targets(&check_field);
    // Canonicalize the full pre-`ignore` expansion first (and
    // dedup — `check`'s entries are a union, ADR-0029 §1: a
    // directory and a file it already contains, or two
    // overlapping directories, must not double-ingest the same
    // file, which would otherwise surface as a spurious
    // `Ruby::DuplicatedMethodDefinitionError`), *then* subtract
    // `ignore` — so `files` and `ignored_files` below share one
    // canonicalization pass. Matching `ignore` against the
    // nominal (pre-canonicalize) path first and re-matching the
    // canonical path later (in `build_diagnostic_filter`) let a
    // symlinked ignored subtree agree with itself on inclusion
    // but disagree on *why* a CLI target naming it was excluded.
    let mut seen = HashSet::new();
    let all_files: Vec<PathBuf> = expanded
        .iter()
        .map(|p| canonicalize_or_exit(p))
        .filter(|p| seen.insert(p.clone()))
        .collect();
    let mut files: Vec<PathBuf> = Vec::new();
    // Exactly the files `ignore` subtracted from `check`'s scope
    // — not "any path matching the glob" (a `lib/x.rb` that was
    // never in `check` to begin with could coincidentally match
    // an unrelated `ignore` entry). `build_diagnostic_filter`
    // uses this set to give a precise "is ignored" error only
    // for CLI targets that were actually excluded this way.
    let mut ignored_files: HashSet<PathBuf> = HashSet::new();
    for p in all_files {
        if is_ignored(&p, &project_root, &ignore_set, &resolved.ignore) {
            ignored_files.insert(p);
        } else {
            files.push(p);
        }
    }

    // ADR-0029 §3/§5: CLI positional targets are a diagnostic
    // output filter over the config scope; `-e` mode filters down
    // to just the eval pseudo-file instead (its own diagnostics
    // only — scope files are still built and validated, but
    // never surfaced, matching §5's "type check only" contract).
    let diagnostic_filter = if eval_active {
        let mut filter = HashSet::new();
        filter.insert(PathBuf::from("-e"));
        Some(filter)
    } else {
        build_diagnostic_filter(&cli_targets, &files, &ignored_files)
    };

    // ADR-0017 Phase 5b switchover: production runs through the
    // Phase 5a `crate::definition_builder::DefinitionBuilder` built from a
    // frozen `Environment`. The `LegacyEnvironmentBuffer` is kept
    // alive only so the parallel-push entry points
    // (`load_dir_with_draft`, `load_inline_annotations_with_draft`)
    // can route declarations into the draft alongside their legacy
    // sinks; we never call `unresolved.resolve()` here. The draft
    let mut draft = crema::environment::draft::EnvironmentDraft::new();

    let lockfile = match &resolved.collection {
        CollectionMode::Auto => match discover_lockfile_from_cwd() {
            Ok(opt) => opt,
            Err(msg) => {
                eprintln!("{}", msg);
                process::exit(2);
            }
        },
        CollectionMode::ConfigPath(p) => match rbs_collection::load_from_config(p) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("{}", e);
                eprintln!(
                    "hint: --collection / collection_config names the rbs collection \
                         config file (rbs_collection.yaml); the lockfile is derived by \
                         inserting .lock before the final extension. Run \
                         `rbs collection install` to generate the lockfile."
                );
                process::exit(2);
            }
        },
    };
    let lock_rubygems_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .rubygems_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    // Stdlib-source gems are NOT fed to the Ruby resolver: they live
    // under `${rbs_gem_dir}/stdlib/<name>/0/` and need no gem path
    // lookup. They are picked up purely by `resolve_lock_stdlib_gems`.
    let lock_stdlib_names: Vec<String> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .stdlib_gem_names()
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();
    // Git-source gems also bypass the Ruby resolver: their install
    // dir is `<lockfile_dir>/<lockfile.path>/<name>/<version>/`,
    // populated by `rbs collection install`. crema only reads.
    let lock_git_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .git_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    // Local-source gems share the git install layout because rbs
    // CLI's `Collection::Sources::Local#install` symlinks each
    // entry into the same `<lockfile_dir>/<lockfile.path>/<name>/
    // <version>/` slot.
    let lock_local_entries: Vec<(String, String)> = lockfile
        .as_ref()
        .map(|d| {
            d.lockfile
                .local_gem_entries()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Entries passed to the Ruby resolver: `crema.toml` libraries
    // first (load order: crema.toml → collection) carry no
    // version, then lock rubygems entries carry their pin so
    // `Gem::Specification.find_by_name(name, version)` selects
    // the spec the lockfile names. This set only feeds the gem
    // path lookup — load dedup is seeded from each pass's
    // `claimed` below, which also covers manifest-expanded deps.
    let mut all_entries: Vec<(String, Option<String>)> = resolved
        .libraries
        .iter()
        .map(|name| (name.clone(), None))
        .collect();
    {
        let mut added: HashSet<String> = resolved.libraries.iter().cloned().collect();
        for (name, version) in &lock_rubygems_entries {
            if added.insert(name.clone()) {
                all_entries.push((name.clone(), Some(version.clone())));
            }
        }
    }

    let rbs_collection_lock_path = lockfile.as_ref().map(|d| d.lockfile_path());

    // ADR-0028 slice 2a-2: opt-in G-snapshot, lazy backend. On a
    // warm hit the gem RBS load below is skipped entirely; the
    // snapshot becomes a lazy probe backend attached to the fresh
    // A-layer draft (sig/ + inline + infusion insert on top, and
    // `build` grafts the layers per name). On a miss the cold
    // path loads as usual, freezes the G-only draft, persists it,
    // and re-opens the just-encoded bytes through the same
    // backend join point — cold and warm share one code path.
    let g_snapshot_key = if no_g_snapshot {
        None
    } else {
        Some(compute_g_snapshot_key(
            &resolved.sig_dirs,
            rbs_collection_lock_path.as_deref(),
            &project_root,
        ))
    };
    let g_snapshot_path = project_root.join(G_SNAPSHOT_FILE);

    let infusion_active = resolved.infusion.activesupport
        || resolved.infusion.activemodel
        || resolved.infusion.activerecord
        || resolved.config_infusion.is_some()
        || resolved.active_decorator_infusion.is_some();
    // Shared by every infusion ingest site below — one build so
    // they all see identical inflection rules.
    let inflector_owner =
        crema::infusion_collector::build_for_config(resolved.inflections.as_ref());

    let mut g_snapshot_warm = false;
    // --refresh-g-snapshot skips the read so the run takes the
    // cold path below: full G rebuild, snapshot rewrite, and the
    // same backend join point (refresh ≡ key-mismatch miss).
    if !refresh_g_snapshot
        && let Some(key) = &g_snapshot_key
        && let Some(backend) =
            crema::snapshot::read::read_g_snapshot_backend(&g_snapshot_path, key, draft.names())
    {
        // The G-construction warnings baked at cold time (stale
        // pin, missing library, load failure, ...) never fire on
        // this run — replay them verbatim so a warm hit stays
        // indistinguishable from a cold one on stderr.
        for w in backend.warnings() {
            eprintln!("{}", w);
        }
        draft.attach_g_backend(backend);
        g_snapshot_warm = true;
    }

    let mut g_warnings: Vec<String> = Vec::new();
    // Every directory `draft.load_dir` is called on below, so the
    // freshness manifest baked into the snapshot can be stat-rescanned
    // on a later warm attempt (`g_snapshot_g_layer_freshness_check`).
    let mut g_dirs: Vec<PathBuf> = Vec::new();
    if !g_snapshot_warm {
        match get_gem_dirs(
            &all_entries,
            rbs_collection_lock_path.as_deref(),
            &mut g_warnings,
        ) {
            Ok(gem_dirs) => {
                let core_dir = gem_dirs.rbs_gem_dir.join("core");

                if core_dir.is_dir() {
                    g_dirs.push(core_dir.clone());
                    if let Err(e) = draft.load_dir(&core_dir) {
                        warn_and_record(
                            &mut g_warnings,
                            format!("warning: failed to load core RBS: {}", e),
                        );
                    }
                }

                // Libraries from `crema.toml` `libraries = [...]`,
                // resolved gem-sig first then stdlib fallback
                // (matching rbs `EnvironmentLoader#each_dir`).
                let libs = crema::library_loader::resolve(&gem_dirs, &resolved.libraries);
                for w in &libs.warnings {
                    warn_and_record(&mut g_warnings, w.clone());
                }
                for dir in &libs.dirs {
                    g_dirs.push(dir.clone());
                    if let Err(e) = draft.load_dir(dir) {
                        warn_and_record(
                            &mut g_warnings,
                            format!("warning: failed to load library {}: {}", dir.display(), e),
                        );
                    }
                }

                let lock_rubygems_refs: Vec<&str> = lock_rubygems_entries
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect();
                let lock_stdlib_refs: Vec<&str> =
                    lock_stdlib_names.iter().map(String::as_str).collect();
                let lock_git_refs: Vec<(&str, &str)> = lock_git_entries
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();
                let git_source =
                    lockfile
                        .as_ref()
                        .map(|discovered| crema::library_loader::GitLockSource {
                            lockfile_dir: &discovered.lockfile_dir,
                            lockfile_path: &discovered.lockfile.path,
                            entries: &lock_git_refs,
                        });
                let lock_local_refs: Vec<(&str, &str)> = lock_local_entries
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();
                let local_source =
                    lockfile
                        .as_ref()
                        .map(|discovered| crema::library_loader::LocalLockSource {
                            lockfile_dir: &discovered.lockfile_dir,
                            lockfile_path: &discovered.lockfile.path,
                            entries: &lock_local_refs,
                        });
                let collection_lock = crema::library_loader::resolve_collection_lock(
                    &gem_dirs,
                    &libs.claimed,
                    &lock_rubygems_refs,
                    &lock_stdlib_refs,
                    git_source,
                    local_source,
                );
                print_collection_lock_warnings(&mut g_warnings, &collection_lock.warnings);
                for (source, dir) in &collection_lock.dirs {
                    g_dirs.push(dir.clone());
                    if let Err(e) = draft.load_dir(dir) {
                        warn_and_record(
                            &mut g_warnings,
                            format!(
                                "warning: failed to load {} {}: {}",
                                source.load_error_label(),
                                dir.display(),
                                e
                            ),
                        );
                    }
                }
            }
            Err(msg) => {
                let is_error = msg.starts_with("error:");
                warn_and_record(&mut g_warnings, msg);
                if is_error {
                    process::exit(2);
                }
            }
        }

        if let Some(key) = &g_snapshot_key
            && !draft.has_resolve_type_names_bypass()
        {
            // Cold path with the flag on: the draft holds exactly
            // the G layer here (sig/ and inline come later). Gem
            // envs containing `# resolve-type-names: false` files
            // are never snapshotted — their frozen form holds raw
            // names a rebuild would wrongly re-resolve, so such
            // projects stay on the plain path (identical to flag
            // off) instead of risking divergent warm output.
            //
            // Stat `g_dirs` now, before `build()` — not after. A
            // concurrent edit landing in `build()`'s window would
            // otherwise get scanned as "fresh" for content `build()`
            // never saw, baking a stale-forever (env, manifest) pair.
            // Scanning here means such a race instead makes the
            // *next* run's comparison miss (one wasted rebuild).
            let g_manifest = crema::snapshot::freshness::scan(&g_dirs);
            let g_env = match draft.build() {
                Ok(env) => std::sync::Arc::new(env),
                Err((e, names)) => {
                    eprintln!(
                        "error: failed to build environment: {}",
                        e.format_with(&names)
                    );
                    process::exit(2);
                }
            };
            let bytes =
                crema::snapshot::write::encode_g_snapshot(key, &g_env, &g_warnings, &g_manifest);
            // Write failures are silent: a warning would make the
            // flag-on run's stderr diverge from the flag-off run.
            let write_result = crema::snapshot::write::write_atomic_bytes(&g_snapshot_path, &bytes);
            if std::env::var_os("CREMA_DEBUG_SNAPSHOT_TIMING").is_some_and(|v| v == "1") {
                match &write_result {
                    Ok(()) => eprintln!("g-snapshot: cold write"),
                    Err(e) => eprintln!("g-snapshot: cold write failed: {}", e),
                }
            }
            drop(write_result);
            // Re-open the just-encoded bytes through the lazy
            // backend so cold and warm runs share one join point
            // (and thus one output shape, byte for byte).
            let mut fresh = crema::environment::draft::EnvironmentDraft::new();
            match crema::snapshot::backend::GSnapshotBackend::open(bytes, fresh.names()) {
                Ok(backend) => {
                    fresh.attach_g_backend(std::sync::Arc::new(backend));
                    draft = fresh;
                }
                Err(_) => {
                    // Bytes we just encoded should always open;
                    // still, crema is a preflight checker whose
                    // snapshot layer must never fail a check, so
                    // an encode/open drift degrades to the slice
                    // 1c eager re-draft instead of aborting.
                    // The unwrap is sound: `encode_g_snapshot`'s
                    // internal Arc clones are dropped on return.
                    let g_env = std::sync::Arc::try_unwrap(g_env)
                        .unwrap_or_else(|_| unreachable!("g_env has no other refs here"));
                    draft = match crema::environment::draft::EnvironmentDraft::from_frozen(g_env) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("error: failed to re-draft gem environment: {}", e);
                            process::exit(2);
                        }
                    };
                }
            }
        }
    }

    let snapshot_timing = std::env::var_os("CREMA_DEBUG_SNAPSHOT_TIMING").is_some_and(|v| v == "1");
    let d_setup = t_run.elapsed();
    let t_ingest = std::time::Instant::now();
    // Buffer for environment-level diagnostics (parser / inline /
    // infusion / validate). Flushed in canonical order right
    // before the per-file check phase begins.
    let mut env_diags: Vec<crema::diagnostic::Diagnostic> = Vec::new();

    // Full sig ingest. Existence/type were validated upfront; any
    // load failure here is I/O (unreadable file, bad UTF-8, etc.).
    // Each entry is either a directory (walked recursively) or a
    // single `.rbs` file.
    for path in &resolved.sig_dirs {
        let result = if path.is_dir() {
            draft.load_dir(path)
        } else {
            draft.load_file(path)
        };
        if let Err(e) = result {
            eprintln!("warning: failed to load sig {}: {}", path.display(), e);
        }
    }

    // Read all sources: Layer 3 (below) type-checks every
    // requested file every run.
    // ADR-0029 §5: `-e` code is appended on top of the scope
    // files rather than replacing them — the scope still gets
    // built into the environment exactly like a bare `crema
    // check`, and the eval pseudo-file participates in the same
    // Pass 0/1 (its declarations join the in-memory environment
    // so a self-contained snippet can define + call its own
    // classes). What `-e` never does is *persist*: every
    // snapshot write below is gated on `!eval_active`. It keeps
    // the pseudo-path `-e` both for diagnostic display and for
    // Location tracking (the build-layer validator resolves `-e`
    // via the in-memory source cache below).
    let mut sources: Vec<(PathBuf, Arc<[u8]>)> = files
        .iter()
        .map(|file| {
            let source = match std::fs::read(file) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: cannot read {}: {}", file.display(), e);
                    process::exit(2);
                }
            };
            (file.clone(), Arc::from(source))
        })
        .collect();
    if let Some(code) = eval {
        sources.push((PathBuf::from("-e"), Arc::from(code.into_bytes())));
    }

    fn flush_env_diagnostics<W: std::io::Write>(
        env_diags: &mut Vec<crema::diagnostic::Diagnostic>,
        emitter: &mut DiagnosticEmitter<W>,
    ) {
        crema::diagnostic::sort_by_canonical_order(env_diags);
        for diag in env_diags.iter() {
            emitter.emit(diag);
        }
        env_diags.clear();
    }

    // Two output streams share one BufWriter:
    //   1. Environment-level diagnostics (parser / inline /
    //      infusion / validate) buffer into `env_diags` below
    //      and get sorted into `(file, start_byte, code,
    //      message)` order right before the check phase begins
    //      — see `crema::diagnostic::sort_by_canonical_order`.
    //   2. Per-file check diagnostics stream out as each file
    //      finishes (Layer 3 loop below), preserving the file-
    //      completion responsiveness AI agents rely on.
    // The macro layer boundary (env phase entirely before
    // check phase) is preserved by construction; within the
    // env phase the byte-order sort supersedes the historical
    // "inline parse → build validate" sub-layer emission order.
    let stdout = std::io::stdout();
    // `"file"` values are displayed cwd-relative (pre-ADR-0029
    // representation, ADR-0005's jq-filter workflow) even though
    // the paths are canonical/absolute internally. Canonicalize
    // the cwd so it string-prefix-matches those canonical paths
    // (`getcwd` already resolves symlinks on the platforms crema
    // targets; the extra canonicalize covers the rest).
    let display_base = std::env::current_dir()
        .ok()
        .and_then(|d| d.canonicalize().ok());
    // ADR-0029 determined this filter's view is the whole
    // observable output (diagnostics *and* exit code), so files
    // outside it can skip Layer 3 entirely — kept here (cloned
    // before the emitter takes ownership) to drive that skip
    // and to suppress the incremental cache write below.
    let check_filter = diagnostic_filter.clone();
    // Extract mode owns stdout for its own output; the
    // check pipeline's diagnostics still flow through the emitter
    // machinery (same code path, same error exits) but land in a sink.
    let emitter_out: Box<dyn Write> = if matches!(mode, RunMode::Check) {
        Box::new(stdout.lock())
    } else {
        Box::new(std::io::sink())
    };
    let mut emitter = DiagnosticEmitter::new(
        resolved.format,
        BufWriter::new(emitter_out),
        resolved.diagnostic.clone(),
    )
    .with_filter(diagnostic_filter)
    .with_display_base(display_base)
    .with_tamped(tamp);
    for (file, source) in &sources {
        emitter.register_source(file.clone(), Arc::clone(source));
    }

    // Pass 0 (parse, shared): every check target's AST is needed
    // for Layer 3 regardless of warm/cold. Inline-annotation
    // collection (which mutates the draft) branches below by
    // mode. The file path is threaded through even in `-e` mode
    // so inline mixins carry a Location keyed on the `-e`
    // pseudo-path; the build validator reads the in-memory
    // source from `source_cache` below to produce accurate
    // `file:line` on arity diagnostics.
    let mut parsed_sources = Vec::with_capacity(sources.len());
    for (file, source) in &sources {
        let source = &source[..];
        let parse_result = ruby_prism::parse(source);

        // Prism returns an AST even when the source is
        // syntactically invalid. Walking that recovery AST is
        // unsafe: receiver-less CallNodes like `{a!: }` (which
        // Ruby itself rejects with SyntaxError) get type-checked
        // as real calls and surface as NoMethod diagnostics on
        // input ruby itself refuses to run. Mirror Steep's
        // policy ("Ruby rejects → type checker stays silent")
        // by emitting one `Ruby::SyntaxError` (anchored at the
        // first prism error) and skipping inline collection
        // and type checking for this file. Build / validator
        // still see other files' declarations.
        if let Some(first_err) = parse_result.errors().next() {
            let offset = first_err.location().start_offset();
            let message = first_err.message().to_string();
            let location = crema::diagnostic::Diagnostic::location_for_byte_range(
                file.clone(),
                source,
                offset,
                first_err.location().end_offset(),
            );
            let diag = crema::diagnostic::Diagnostic::at(
                location,
                DiagnosticKind::SyntaxError { message },
            );
            env_diags.push(diag);
            continue;
        }

        parsed_sources.push(ParsedSource {
            file: file.as_path(),
            source,
            parse_result: Some(parse_result),
            content_hash: None,
        });
    }

    if snapshot_timing {
        eprintln!("ingest: read+parse {}us", t_ingest.elapsed().as_micros());
    }
    let t_inline = std::time::Instant::now();
    // Mutate the draft for every file, the eval pseudo-file
    // included (ADR-0029 §5 rework — eval declarations live
    // in the in-memory environment; only snapshot persistence
    // is eval-gated).
    for parsed in &parsed_sources {
        let inline_diags = if resolved.inline {
            crema::inline_parser::load_inline_annotations(
                parsed.source,
                parsed.ast(),
                Some(parsed.file),
                &mut draft,
            )
        } else {
            crema::inline_parser::parse_inline_annotations(
                parsed.source,
                parsed.ast(),
                Some(parsed.file),
                &mut draft,
            )
        };
        env_diags.extend(inline_diags);
    }

    if infusion_active {
        let infusion_sources: Vec<_> = parsed_sources
            .iter()
            .map(|parsed| crema::infusion_collector::SourceUnit {
                source: parsed.source,
                parse_result: parsed.ast(),
                file: Some(parsed.file),
            })
            .collect();
        if let Some(config_options) = resolved.config_infusion.as_ref() {
            match crema::infusion_collector::config::load(config_options, &mut draft) {
                Ok(warnings) => env_diags.extend(warnings),
                Err(err) => {
                    eprintln!("error: invalid [infusion.config]: {}", err);
                    // Flush anything Pass 0 already pushed into
                    // `env_diags` — syntax / inline-parse errors
                    // from earlier files must not vanish just
                    // because a later config load blew up.
                    flush_env_diagnostics(&mut env_diags, &mut emitter);
                    emitter.finish();
                    drop(emitter);
                    process::exit(2);
                }
            }
        }
        let schema = if resolved.infusion.activerecord {
            // Pre-pass: `self.table_name =` lives in the model files and the
            // schema dumps parse independently of them — collect both up
            // front; load_all emits the schema declarations once the
            // post-concern-expansion enum set is known.
            let (schema, schema_diags) = crema::infusion_collector::activerecord::prepare_schema(
                &project_root,
                &infusion_sources,
            );
            env_diags.extend(schema_diags);
            Some(schema)
        } else {
            None
        };
        env_diags.extend(crema::infusion_collector::load_all_with_schema(
            &infusion_sources,
            &mut draft,
            resolved.infusion,
            &inflector_owner,
            schema,
        ));
        // Last: decorator modules take their self-type from a
        // model that may be declared in any other file, so every
        // Ruby-source declaration has to be in the draft first.
        if let Some(table) = resolved.active_decorator_infusion.as_ref() {
            crema::infusion_collector::active_decorator::synthesize(
                &mut draft,
                &project_root,
                &table.decorator_suffix,
            );
        }
    }

    if snapshot_timing {
        eprintln!(
            "ingest: inline+infusion {}us",
            t_inline.elapsed().as_micros()
        );
    }
    let d_ingest = t_ingest.elapsed();
    // ADR-0032 Decision 1: the incremental scheduler is a
    // crema.toml opt-in and never active for `-e` (nothing
    // persists there — same gate as the snapshot writes). Every
    // cache read/write, the fingerprint pass, and the per-file
    // content hashing live behind this flag; the default path
    // pays nothing new.
    let incremental_active = resolved.incremental.active() && !eval_active;
    // ADR-0032 Decision 6: shadow-recheck replay-classified
    // files and treat any divergence as a recording bug —
    // reported on stderr (never the JSONL stream, which always
    // carries the fresh side) and failing the run.
    let verify_active = resolved.incremental.verify() && !eval_active;
    let incremental_debug = std::env::var_os("CREMA_DEBUG_INCREMENTAL").is_some_and(|v| v == "1");
    let incremental_cache_path = project_root.join(crema::incremental::INCREMENTAL_CACHE_FILE);
    let t_incremental = std::time::Instant::now();
    // Scheduler stage 1: the previous generation off disk. Read
    // here, before the draft is frozen, because its content
    // hashes are what the AST release below keys on.
    let cached_generation = incremental_active.then(|| {
        // The cache key is the G-snapshot invalidation key
        // (crema version + lockfile contents + crema.toml
        // content + sig path set — ADR-0032 Decision 3's G-layer
        // matching, which also folds in every crema.toml-borne
        // check setting). Recomputed only when --no-g-snapshot
        // withheld the one computed above; the two caches stay
        // independently usable.
        let key = g_snapshot_key.unwrap_or_else(|| {
            compute_g_snapshot_key(
                &resolved.sig_dirs,
                rbs_collection_lock_path.as_deref(),
                &project_root,
            )
        });
        crema::incremental::CachedGeneration::load(&incremental_cache_path, key, resolved.inline)
    });
    let mut d_incremental_prepare = t_incremental.elapsed();

    // AST release: a file's AST is dead once inline collection is
    // over unless Layer 3 type-checks it. Two kinds of file
    // provably won't be: one outside the diagnostic filter (the
    // loop skips it outright), and one whose bytes match the
    // previous generation (a replay candidate — the loop re-emits
    // its stored diagnostics). The second is provisional: a probe
    // hit in `dispose` can still demote it to recheck, in which
    // case the loop re-parses from `source`. Releasing here, before
    // env construction, is the point — the footprint peak sits at
    // the end of `Scheduler::prepare`, and the C-side prism heap
    // holds these nodes at ~10x the source size, so a large warm
    // project's peak drops by the whole AST. The two conditions
    // are independent: a filtered-out file is released whether or
    // not the scheduler is on, while the replay-candidate release
    // needs a valid previous generation — so a cold run (no cache),
    // a run without the scheduler, and verify mode (every file
    // rechecked) all keep every in-filter AST, and this pass is a
    // plain hash pre-computation for them. Extract reads every AST,
    // so it is exempt.
    let mut dropped_ast_count = 0usize;
    if matches!(mode, RunMode::Check) {
        let mut dead_asts = Vec::new();
        for parsed in &mut parsed_sources {
            let in_filter = check_filter
                .as_ref()
                .is_none_or(|filter| filter.contains(parsed.file));
            parsed.content_hash = (in_filter && incremental_active)
                .then(|| crema::incremental::content_hash(parsed.source));
            let replay_candidate = !verify_active
                && match (&cached_generation, parsed.content_hash) {
                    (Some(generation), Some(hash)) => {
                        generation.content_unchanged(&parsed.file.to_string_lossy(), hash)
                    }
                    _ => false,
                };
            if (!in_filter || replay_candidate)
                && let Some(ast) = parsed.parse_result.take()
            {
                dead_asts.push(ast);
            }
        }
        dropped_ast_count = dead_asts.len();
        drop(dead_asts);
    }

    let t_build = std::time::Instant::now();

    // Build a source cache for the build-layer validator so Location
    // → line/column works for `.rb` files (including `-e`) without
    // hitting disk. `.rbs` files read by env.load_dir aren't in
    // memory here; the validator falls back to std::fs::read for
    // those.
    let mut source_cache: FxHashMap<std::path::PathBuf, &[u8]> = FxHashMap::default();
    for (file, bytes) in &sources {
        source_cache.insert(file.to_path_buf(), &bytes[..]);
    }

    let (env, d_build, d_defbuild, d_validate) = {
        let frozen = match draft.build() {
            Ok(env) => std::sync::Arc::new(env),
            Err((e, names)) => {
                eprintln!(
                    "error: failed to build environment: {}",
                    e.format_with(&names)
                );
                // Pass 0 pushed inline-parse / syntax diagnostics
                // into `env_diags`; the infusion loaders above
                // may have added more. Flush them in canonical
                // order before exit so the run that just failed
                // still tells the user which diagnostics were
                // already computed.
                flush_env_diagnostics(&mut env_diags, &mut emitter);
                emitter.finish();
                drop(emitter);
                process::exit(2);
            }
        };
        let d_build_cold = t_build.elapsed();
        let t_defbuild = std::time::Instant::now();
        let env = DefinitionBuilder::from_environment(Arc::clone(&frozen));
        let d_defbuild_cold = t_defbuild.elapsed();

        let t_validate = std::time::Instant::now();
        let (_, validate_diags) = validator::full_validate(&env, &source_cache);
        env_diags.extend(validate_diags);
        let d_validate_cold = t_validate.elapsed();

        (env, d_build_cold, d_defbuild_cold, d_validate_cold)
    };

    // Environment-level phase done — flush the buffered
    // diagnostics in canonical order (see
    // `sort_by_canonical_order`) before switching to the
    // per-file streaming check phase below.
    flush_env_diagnostics(&mut env_diags, &mut emitter);

    match &mode {
        RunMode::Extract => {
            emitter.finish();
            drop(emitter);
            run_extract(
                &env,
                &sources,
                &parsed_sources,
                &resolved,
                &project_root,
                eval_active,
            );
            return;
        }
        RunMode::Check => {}
    }

    let t_check = std::time::Instant::now();
    // Sub-phase accumulators for the check-timing line: time spent in
    // the `did_you_mean` join and in JSONL emission, so a warm run's
    // replay cost can be told apart from the type-check proper.
    let mut d_join = std::time::Duration::ZERO;
    let mut d_emit = std::time::Duration::ZERO;
    let mut d_cache_write = std::time::Duration::ZERO;

    // Dev-only trigger for per-file consultation recording
    // (ADR-0032 Decision 2, axis 3). The real on/off switch is
    // `crema.toml`'s `incremental` flag (cache_io's territory,
    // Decision 1); this env var exists only so this axis's own
    // JSONL-idempotency and perf-overhead checks can turn
    // recording on without that wiring existing yet. Mirrors the
    // `CREMA_DEBUG_SNAPSHOT_TIMING` precedent above.
    let record_consultation =
        std::env::var_os("CREMA_DEBUG_CONSULTATION_LOG").is_some_and(|v| v == "1");
    let mut consulted_key_total = 0usize;

    let mut verify_divergent = 0usize;
    let mut reparsed_ast_count = 0usize;
    // Scheduler stage 2: fingerprint diff + probes against the
    // frozen env. This is where the footprint peaks (every
    // declaration resolves for the fingerprint pass) — hence the
    // AST release above sits before it, not after.
    let t_incremental = std::time::Instant::now();
    let mut scheduler = cached_generation.map(|generation| generation.prepare(env.env_arc()));
    d_incremental_prepare += t_incremental.elapsed();
    let mut incremental_files: Vec<crema::incremental::FileEntry> = Vec::new();
    let mut replayed_count = 0usize;
    let mut rechecked_count = 0usize;

    // Layer 3 (type check): per-file type checking. Stream each
    // file's diagnostics out as soon as the file finishes; there
    // is no shared budget across files. Under the incremental
    // scheduler each file either replays (stored diagnostics
    // re-emitted through the same join + emit path a fresh check
    // takes) or rechecks with consultation recording on — in the
    // same walk order either way, so the JSONL stream stays
    // bit-identical to a full check (ADR-0032 Decision 1).
    for parsed in &mut parsed_sources {
        // Files outside the diagnostic filter never reach the
        // output or the exit code (ADR-0029 §3/§5), so their
        // per-file check is unobservable work. Env construction
        // above still saw every scope file — only Layer 3 is
        // elided. Skipped files also stay out of
        // `incremental_files`, which is safe because filtered
        // runs never write the cache (see below).
        if let Some(filter) = &check_filter
            && !filter.contains(parsed.file)
        {
            continue;
        }
        let current_hash = parsed.content_hash;
        if let Some(sched) = scheduler.as_mut() {
            let hash = current_hash.expect("incremental_active implies hash");
            let path_str = parsed.file.to_string_lossy();
            if let Some(entry) = sched.dispose(&path_str, hash) {
                if verify_active {
                    let log = ConsultationLog::new();
                    let fresh = check_source_with_log(
                        &env,
                        parsed.file.to_path_buf(),
                        parsed.source,
                        parsed.ast(),
                        CheckOptions {
                            verbose,
                            inline: resolved.inline,
                        },
                        Some(&log),
                    );
                    let entries = log.into_entries();
                    match crema::incremental::verify_file(&entry, &fresh, &entries) {
                        Some(div) => {
                            report_verify_divergence(&path_str, &entry, &fresh, &div);
                            verify_divergent += 1;
                            // The fresh entry replaces the diverged
                            // one in the next generation (counted as
                            // a recheck so `into_payload` sees the
                            // cache as dirty): one verify run heals
                            // the cache.
                            rechecked_count += 1;
                            incremental_files.push(crema::incremental::FileEntry {
                                path: entry.path,
                                content_hash: hash,
                                consulted: crema::incremental::project_consulted_entries(&entries),
                                diagnostics: fresh.clone(),
                            });
                        }
                        None => {
                            replayed_count += 1;
                            incremental_files.push(entry);
                        }
                    }
                    // Emit the fresh side either way — identical to
                    // the stored side when clean, and never the
                    // forged bytes when diverged, so the JSONL
                    // contract holds without consulting the verdict.
                    let mut diagnostics = fresh;
                    let t_join = std::time::Instant::now();
                    if resolved.did_you_mean {
                        join_did_you_mean(&env, &mut diagnostics);
                    }
                    d_join += t_join.elapsed();
                    let t_emit = std::time::Instant::now();
                    for diag in &diagnostics {
                        emitter.emit(diag);
                    }
                    d_emit += t_emit.elapsed();
                    continue;
                }
                let mut diagnostics = entry.diagnostics.clone();
                let t_join = std::time::Instant::now();
                if resolved.did_you_mean {
                    join_did_you_mean(&env, &mut diagnostics);
                }
                d_join += t_join.elapsed();
                let t_emit = std::time::Instant::now();
                for diag in &diagnostics {
                    emitter.emit(diag);
                }
                d_emit += t_emit.elapsed();
                replayed_count += 1;
                incremental_files.push(entry);
                continue;
            }
        }
        // A replay candidate demoted to recheck by a probe hit had
        // its AST released above; the same bytes parse to the same
        // tree, so re-parsing here is invisible to the output.
        if parsed.parse_result.is_none() {
            parsed.parse_result = Some(ruby_prism::parse(parsed.source));
            reparsed_ast_count += 1;
        }
        let log = (incremental_active || record_consultation).then(ConsultationLog::new);
        let mut diagnostics = check_source_with_log(
            &env,
            parsed.file.to_path_buf(),
            parsed.source,
            parsed.ast(),
            CheckOptions {
                verbose,
                inline: resolved.inline,
            },
            log.as_ref(),
        );
        if let Some(log) = log {
            let entries = log.into_entries();
            if record_consultation {
                consulted_key_total += entries.len();
            }
            if incremental_active {
                rechecked_count += 1;
                // Persisted pre-join: `did_you_mean` is a
                // computed column recomputed against the live
                // env at every emission (Decision 5a), so the
                // stored fact must not carry it.
                incremental_files.push(crema::incremental::FileEntry {
                    path: parsed.file.to_string_lossy().into_owned(),
                    content_hash: current_hash.expect("incremental_active implies hash"),
                    consulted: crema::incremental::project_consulted_entries(&entries),
                    diagnostics: diagnostics.clone(),
                });
            }
        }
        // ADR-0032 Decision 5a: check only records the fact that a
        // constant failed to resolve (name + candidate scope);
        // `did_you_mean` is a computed column joined against the
        // env in scope here, right before emission — the same
        // join a cached/replayed diagnostic would go through.
        let t_join = std::time::Instant::now();
        if resolved.did_you_mean {
            join_did_you_mean(&env, &mut diagnostics);
        }
        d_join += t_join.elapsed();
        let t_emit = std::time::Instant::now();
        for diag in &diagnostics {
            emitter.emit(diag);
        }
        d_emit += t_emit.elapsed();
    }

    if let Some(sched) = scheduler {
        // Invariant: only a full-view run advances the cache
        // generation. A filtered run skipped some files above;
        // persisting a fresh fingerprint table next to their
        // stale entries would make the next bare run see an
        // empty diff and replay outdated diagnostics.
        let partial = check_filter.is_some();
        let payload = if partial {
            None
        } else {
            sched.into_payload(incremental_files, rechecked_count)
        };
        let wrote = payload.is_some();
        let t_cache_write = std::time::Instant::now();
        if let Some((key, payload)) = payload
            && let Err(e) = crema::incremental::write_cache(&incremental_cache_path, &key, &payload)
        {
            eprintln!("warning: failed to write incremental cache: {e}");
        }
        d_cache_write = t_cache_write.elapsed();
        if incremental_debug {
            eprintln!("ast: {dropped_ast_count} dropped, {reparsed_ast_count} reparsed");
            eprintln!(
                "incremental: {replayed_count} replayed, {rechecked_count} rechecked, \
                     prepare {:.3}ms, cache {}",
                d_incremental_prepare.as_secs_f64() * 1000.0,
                if wrote {
                    "written"
                } else if partial {
                    "skipped (partial run)"
                } else {
                    "unchanged"
                },
            );
        }
    }

    if record_consultation {
        eprintln!(
            "consultation-log: {consulted_key_total} keys recorded across {} files",
            parsed_sources.len()
        );
    }

    if snapshot_timing {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        eprintln!(
            "check-timing: total {:.1}ms setup {:.1}ms ingest {:.1}ms build {:.1}ms defbuild {:.1}ms validate {:.1}ms check {:.1}ms (join {:.1}ms emit {:.1}ms) cache-write {:.1}ms",
            ms(t_run.elapsed()),
            ms(d_setup),
            ms(d_ingest),
            ms(d_build),
            ms(d_defbuild),
            ms(d_validate),
            ms(t_check.elapsed()),
            ms(d_join),
            ms(d_emit),
            ms(d_cache_write)
        );
        if let Some(backend) = env.env().g_backend() {
            eprintln!(
                "g-snapshot: decoded {} entries lazily",
                backend.decode_count()
            );
        }
    }

    let had_any_error = emitter.had_any_error();
    emitter.finish();
    drop(emitter);

    // A verify divergence is a crema bug, not a project
    // diagnostic — it fails the run even when the JSONL stream
    // is clean, so automated dogfooding cannot miss it.
    if had_any_error || verify_divergent > 0 {
        process::exit(1);
    }
}

/// One check target parsed by Pass 0 (shared by the check loop and the
/// extract loop).
struct ParsedSource<'a> {
    file: &'a Path,
    source: &'a [u8],
    /// `None` once the AST has been released ahead of the check loop
    /// (a content-unchanged or filtered-out file — see the release
    /// pre-pass in `run_check`); a recheck re-parses it on demand. Every reader after that point goes
    /// through `ast()`, so a read-after-release is a loud panic rather
    /// than a silently skipped file.
    parse_result: Option<ruby_prism::ParseResult<'a>>,
    /// xxh3 of `source`, computed once in the release pre-pass for
    /// every file the incremental scheduler will classify; `None`
    /// otherwise (scheduler off, or file outside the check filter).
    content_hash: Option<u64>,
}

impl<'a> ParsedSource<'a> {
    fn ast(&self) -> &ruby_prism::ParseResult<'a> {
        self.parse_result
            .as_ref()
            .expect("AST read after it was released ahead of the check loop")
    }
}

/// The per-file phase of `crema extract` (invoked from `run_check`
/// after env construction): run the site-collecting check on every
/// scope file with consultation recording on, join in the per-file
/// definitions from the environment, write the whole document to
/// `.crema/extract.json`, and print the one-line summary.
///
/// In `-e` mode (`eval_active`) the document is narrowed to the `-e`
/// pseudo-file and printed to stdout instead: no `.crema/extract.json`
/// write, no summary line, so stdout stays parsable as one JSON value.
///
/// Every file is checked fresh — the incremental cache is neither read
/// nor narrowed here. Extract is opt-in and allowed to be slower than
/// `crema check`; recording costs exist only on this path.
fn run_extract(
    env: &DefinitionBuilder,
    sources: &[(PathBuf, Arc<[u8]>)],
    parsed_sources: &[ParsedSource<'_>],
    resolved: &crema::config::ResolvedConfig,
    project_root: &Path,
    eval_active: bool,
) {
    // Same cwd-relative display rule as the diagnostic emitter; `root`
    // records the base so the document stays
    // self-describing wherever it is consumed.
    let display_base = std::env::current_dir()
        .ok()
        .and_then(|d| d.canonicalize().ok());
    let display = |path: &Path| -> String {
        display_base
            .as_deref()
            .and_then(|base| path.strip_prefix(base).ok())
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    };

    // Keyed by `sources` (the whole check scope), not `parsed_sources`:
    // a file whose parse failed never reaches `parsed_sources`, but it
    // must still appear in the document — silently dropping it would
    // make "syntax-broken file" indistinguishable from "file with no
    // records" for consumers (crema-review finding). Such a file gets
    // an entry with its content hash and empty record arrays.
    let mut files: std::collections::BTreeMap<String, crema::extract::FileRecord> =
        std::collections::BTreeMap::new();
    // The `-e` pseudo-path `run_check` appended on top of the scope
    // (ADR-0029 §5). In `-e` mode it is the only file the document may
    // contain, so the scope files are skipped here: extract mode sinks
    // its diagnostics, so a record nobody reads is the sole thing their
    // check would produce.
    let eval_file = Path::new("-e");
    for (file, bytes) in sources {
        if eval_active && file != eval_file {
            continue;
        }
        let mut record = crema::extract::FileRecord {
            content_hash: format!("{:016x}", crema::incremental::content_hash(bytes)),
            ..Default::default()
        };
        if let Some(parsed) = parsed_sources.iter().find(|p| p.file == file) {
            let log = crema::definition_builder::ConsultationLog::new();
            let (method_call, implements, constant, signature_types) =
                crema::type_checker::check_source_extract(
                    env,
                    parsed.file.to_path_buf(),
                    parsed.source,
                    parsed.ast(),
                    CheckOptions {
                        verbose: false,
                        inline: resolved.inline,
                    },
                    Some(&log),
                );
            record.method_call = method_call;
            record.implements = implements;
            record.constant = constant;
            record.consulted = crema::extract::consulted_symbols(
                &log.into_entries(),
                &signature_types,
                env.env().names(),
            );
        }
        files.insert(display(file), record);
    }

    // Definitions pass: every file that contributed declarations
    // (`path_index` = A-layer only, so gem/core files never appear).
    // Check targets reuse their in-memory bytes; sig files are read
    // back from disk for hash computation.
    let names = env.env().names();
    for file_name in env.env().path_index_files() {
        let path_string = names.resolve(file_name);
        let path = Path::new(&path_string);
        if eval_active && path != eval_file {
            continue;
        }
        let source: std::borrow::Cow<'_, [u8]> =
            match parsed_sources.iter().find(|p| p.file == path) {
                Some(parsed) => std::borrow::Cow::Borrowed(parsed.source),
                None => match std::fs::read(path) {
                    Ok(bytes) => std::borrow::Cow::Owned(bytes),
                    // Unreadable index entries (file-less test sources,
                    // files deleted mid-run) contribute no definitions.
                    Err(_) => continue,
                },
            };
        let definitions = crema::extract::file_definitions(env, file_name);
        if definitions.is_empty() {
            continue;
        }
        let entry = files
            .entry(display(path))
            .or_insert_with(|| crema::extract::FileRecord {
                content_hash: format!("{:016x}", crema::incremental::content_hash(&source)),
                ..Default::default()
            });
        entry.definitions = definitions;
    }

    let root = display_base
        .as_deref()
        .unwrap_or(project_root)
        .to_string_lossy()
        .into_owned();
    let output = crema::extract::ExtractOutput {
        version: crema::extract::EXTRACT_SCHEMA_VERSION,
        root,
        files,
    };
    let bytes = serde_json::to_vec(&output).expect("extract document serialize is infallible");
    if eval_active {
        // Scratch query: stdout carries the document itself, so nothing
        // else may share it (no summary line) and nothing is persisted
        // — an existing `.crema/extract.json` stays as it was, and a
        // zero-config run never creates `.crema/`.
        let mut out = std::io::stdout().lock();
        if let Err(err) = out.write_all(&bytes).and_then(|()| out.write_all(b"\n"))
            && err.kind() != std::io::ErrorKind::BrokenPipe
        {
            eprintln!("error: failed to write extract document: {err}");
            process::exit(2);
        }
        return;
    }
    let out_path = project_root.join(crema::extract::EXTRACT_FILE);
    if let Some(parent) = out_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("error: failed to create {}: {e}", parent.display());
        process::exit(2);
    }
    if let Err(e) = crema::atomic_write::write_atomic_bytes(&out_path, &bytes) {
        eprintln!("error: failed to write {}: {e}", out_path.display());
        process::exit(2);
    }

    // Hand-ordered keys (path, files, bytes), values through serde_json
    // for escaping.
    println!(
        "{{\"path\":{},\"files\":{},\"bytes\":{}}}",
        serde_json::to_string(&display(&out_path)).expect("string serialize is infallible"),
        output.files.len(),
        bytes.len(),
    );
}

fn main() {
    let cli = Cli::parse();
    let cli_config = cli.config.clone();

    match cli.command {
        Commands::Doc { command } => match command {
            DocCommands::Config => {
                print!("{}", prompts::DOC_CONFIG);
            }
            DocCommands::Extract => {
                print!("{}", prompts::DOC_EXTRACT);
            }
            DocCommands::Diagnostic { code: Some(code) } => {
                if let Some(doc) = DiagnosticKind::doc_for_code(&code) {
                    print!("{}", doc);
                } else {
                    eprintln!("error: unknown diagnostic code `{}`", code);
                    eprintln!("hint: run `crema doc diagnostic` to inspect known codes");
                    process::exit(2);
                }
            }
            DocCommands::Diagnostic { code: None } => {
                let file_config = load_file_config(cli_config.as_deref());
                let diagnostic = match Config::resolve_with_cli_collecting_warnings(
                    file_config,
                    &[],
                    &[],
                    None,
                    CollectionCliOverride::Absent,
                ) {
                    Ok((resolved, warnings)) => {
                        for w in &warnings {
                            eprintln!("{}", w);
                        }
                        resolved.diagnostic
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        process::exit(2);
                    }
                };

                let stdout = std::io::stdout();
                let mut out = BufWriter::new(stdout.lock());
                for (code, description) in DiagnosticKind::ALL_CODES
                    .iter()
                    .zip(DiagnosticKind::ALL_DESCRIPTIONS)
                {
                    let severity = diagnostic.severity_for_code(code);
                    let entry = serde_json::json!({
                        "code": code,
                        "description": description,
                        "severity": severity.as_str(),
                    });
                    // Match the diagnostic emitter's BrokenPipe convention:
                    // `crema doc diagnostic | head` closes the pipe early,
                    // which is a clean exit, not a crash.
                    if let Err(err) = writeln!(out, "{}", entry) {
                        if err.kind() == std::io::ErrorKind::BrokenPipe {
                            break;
                        }
                        panic!("write diagnostic code entry: {}", err);
                    }
                }
            }
        },
        Commands::Internal { command } => match command {
            InternalCommands::Snapshot { command } => match command {
                InternalSnapshotCommands::Dump { project_root } => {
                    run_snapshot_dump(&project_root, cli_config.as_deref());
                }
            },
        },
        Commands::Check {
            files,
            eval,
            sig_dirs,
            add_sig_dirs,
            verbose,
            inline,
            collection,
            no_g_snapshot,
            refresh_g_snapshot,
            tamp,
        } => run_check(
            cli_config.as_deref(),
            CheckInvocation {
                files,
                eval,
                sig_dirs,
                add_sig_dirs,
                verbose,
                inline,
                collection,
                no_g_snapshot,
                refresh_g_snapshot,
                tamp,
            },
            RunMode::Check,
        ),
        Commands::Extract { eval } => run_check(
            cli_config.as_deref(),
            CheckInvocation {
                files: Vec::new(),
                eval,
                sig_dirs: Vec::new(),
                add_sig_dirs: Vec::new(),
                verbose: false,
                inline: None,
                collection: None,
                no_g_snapshot: false,
                refresh_g_snapshot: false,
                tamp: false,
            },
            RunMode::Extract,
        ),
    }
}


// mimalloc as the global allocator: measured 1.34x (macOS) / ~1.2x (Linux
// glibc) on `crema check lib` against the steep repo with identical output.
// Only Rust-side allocations are affected; the rbs C parser keeps using
// the system malloc. Lives in main.rs so the lib does not impose it.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
