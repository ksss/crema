use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::{Parser, Subcommand};
use globset::{Glob, GlobSet, GlobSetBuilder};

use std::io::{BufWriter, Write};

use crema::config::{CONFIG_FILE, CollectionCliOverride, CollectionMode, Config, ConfigError};
use crema::definition_builder::DefinitionBuilder;
use crema::diagnostic::{
    DiagnosticEmitter, DiagnosticKind, DiagnosticRenderer, RenderedDiagnostic, join_did_you_mean,
};
use crema::gem_dir_cache::ResolvedGemDirs;
use crema::ingest_cache::{DirEntry, DirListing, FileStat};
use crema::rbs_collection::{self, DiscoveredLockfile};
use crema::type_checker::check_source;
use crema::validator;

mod prompts;

#[derive(Parser)]
#[command(
    name = "crema",
    version = concat!(env!("CARGO_PKG_VERSION"), env!("CREMA_GIT_INFO")),
    about = "An AI-agent-first Ruby type checker"
)]
struct Cli {
    /// Path to a config file to load instead of `./crema.toml`. When
    /// set, walk-up discovery is bypassed entirely (no merge). The file
    /// is treated as if it were the project's `crema.toml`: its
    /// directory is the project root, so relative paths inside it (e.g.
    /// `sig = [...]`, `check = [...]`), the `.crema` directory and
    /// the `file` paths in the output are all relative to that directory,
    /// not the current one. CLI path arguments (`--sig`, positional
    /// filters) stay relative to the current directory.
    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    /// Spawn the gem resolver with plain `ruby` instead of `bundle exec
    /// ruby`, even when a Gemfile is discoverable. Lets `crema extract`
    /// run without `bundle install` (e.g. in CI); gem-provided types
    /// are then not resolved, only rbs core and globally installed
    /// gems. Snapshots built with and without this flag never share a
    /// cache entry.
    #[arg(long, global = true)]
    no_bundler: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum DocCommands {
    /// Reference for crema.toml keys
    Config,
    /// Reference for the document schema printed by `crema extract`
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
    /// `.crema/g_snapshot_v1.bin` under the project root
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

        /// Script mode: type check one stdlib-only Ruby script with no
        /// project setup. `crema.toml` is never looked for (neither
        /// read nor ignored — the walk-up does not run), the
        /// environment is rbs core plus every stdlib signature shipped
        /// with the installed rbs gem plus the rbs gem's own `sig/` (and
        /// the `sig/` of its gemspec runtime dependencies, e.g. prism),
        /// inline annotations are on, the
        /// gem resolver is plain `ruby` (never `bundle exec`), and
        /// nothing is written to disk (no `.crema/`, no snapshot read
        /// or write). Scripts that need gems or project code belong in
        /// a project with a `crema.toml` instead. Cannot be combined
        /// with files, `-e`, `--config`, `--sig`, `--add-sig`,
        /// `--collection`, `--inline`, `--no-g-snapshot` or
        /// `--refresh-g-snapshot`.
        #[arg(long = "script", value_name = "FILE")]
        script: Option<PathBuf>,

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
        /// `.crema/g_snapshot_v1.bin` and rebuilds from it on warm
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
        /// This output is ideal for use as a baseline. Never filtered
        /// by `crema.toml`'s `baseline` (it is the full projection
        /// `--update-baseline` writes).
        #[arg(long = "tamp")]
        tamp: bool,

        /// Rewrite the baseline file named by `crema.toml`'s `baseline`
        /// with this run's diagnostics (same rows and order as
        /// `--tamp`) and exit 0. Whole-scope only: cannot be combined
        /// with file arguments, `-e`, `--tamp` or `--no-baseline`.
        #[arg(long = "update-baseline")]
        update_baseline: bool,

        /// Ignore `crema.toml`'s `baseline` for this run: every
        /// diagnostic is reported, including grandfathered ones.
        #[arg(long = "no-baseline")]
        no_baseline: bool,

        /// Number of threads for the parallel phases (parse, inline
        /// collection and type check). Defaults to `CREMA_THREADS` if
        /// set, else every available core. `1` runs single-threaded,
        /// which is the yardstick for CPU-time comparisons; the default
        /// thread count is the yardstick for wall-clock time.
        #[arg(long = "threads", value_name = "N")]
        threads: Option<usize>,
    },
    /// Export check-internal facts (definitions, implements,
    /// method-call sites, consulted symbols) as one JSON document on
    /// stdout. The document is large (tens of MB on a big project), so
    /// redirect or pipe it. Scope and environment come from crema.toml
    /// exactly like `crema check`. With `-e`, the document covers the
    /// snippet alone.
    Extract {
        /// Extract from inline Ruby code instead of the config scope:
        /// `files` holds just the `-e` pseudo-file. The scope's RBS
        /// environment is still built, so the snippet sees project
        /// types.
        #[arg(short = 'e')]
        eval: Option<String>,

        /// Number of threads for the parallel phases (parse, inline
        /// collection and the per-file check). Defaults to
        /// `CREMA_THREADS` if set, else every available core. `1` runs
        /// single-threaded, which is the yardstick for CPU-time
        /// comparisons; the default thread count is the yardstick for
        /// wall-clock time.
        #[arg(long = "threads", value_name = "N")]
        threads: Option<usize>,
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

const G_SNAPSHOT_FILE: &str = ".crema/g_snapshot_v1.bin";
// `GEMFILE` / `GEMFILE_LOCK` live in `crema::bundle_root` so the
// `project_root` walk-up and `find_gemfile` share one source of truth
// for the filenames.
use crema::bundle_root::{GEMFILE, GEMFILE_LOCK, discover_bundle_root};

/// Where gem resolution starts, so a run from any directory of the
/// project reads the lockfile a run from the project root reads.
///
/// The ruby child runs in `project_root`: bundler searches its Gemfile
/// up from its own cwd (`Bundler::SharedHelpers#search_up`), and the
/// G-snapshot key predicts the lockfile from the project root, so the
/// two must start from the same place. `BUNDLE_GEMFILE` /
/// `BUNDLE_LOCKFILE` keep bundler's meaning — a relative value names a
/// file under the cwd crema was started from — so they are absolutized
/// against that cwd before the child is moved.
struct ResolverAnchor {
    project_root: PathBuf,
    bundle_gemfile: Option<PathBuf>,
    bundle_lockfile: Option<PathBuf>,
}

impl ResolverAnchor {
    /// Read `BUNDLE_GEMFILE` / `BUNDLE_LOCKFILE` and the cwd from the
    /// process.
    fn from_env(project_root: &Path) -> Self {
        let cwd = std::env::current_dir().ok();
        Self::new(
            project_root,
            std::env::var_os("BUNDLE_GEMFILE").as_deref(),
            std::env::var_os("BUNDLE_LOCKFILE").as_deref(),
            cwd.as_deref(),
        )
    }

    /// An empty value is unset, as bundler treats it. Without a `cwd` a
    /// relative value is kept as given.
    fn new(
        project_root: &Path,
        bundle_gemfile: Option<&OsStr>,
        bundle_lockfile: Option<&OsStr>,
        cwd: Option<&Path>,
    ) -> Self {
        let absolutize = |value: Option<&OsStr>| {
            value.filter(|v| !v.is_empty()).map(|v| match cwd {
                Some(cwd) => cwd.join(v),
                None => PathBuf::from(v),
            })
        };
        ResolverAnchor {
            project_root: project_root.to_path_buf(),
            bundle_gemfile: absolutize(bundle_gemfile),
            bundle_lockfile: absolutize(bundle_lockfile),
        }
    }

    fn uses_bundler(&self, no_bundler: bool) -> bool {
        resolver_uses_bundler(
            no_bundler,
            self.bundle_gemfile.as_deref().map(Path::as_os_str),
            &self.project_root,
        )
    }

    /// The Gemfile.lock whose content feeds the G-snapshot key: the one
    /// bundler reads (`Bundler::SharedHelpers#default_lockfile`) when
    /// the env names it, else the walk-up from the project root. Under
    /// `--no-bundler` the env is not consulted, as the resolver ignores
    /// it.
    fn gemfile_lock(&self, no_bundler: bool) -> Option<PathBuf> {
        if !no_bundler {
            if let Some(lock) = &self.bundle_lockfile {
                return Some(lock.clone());
            }
            if let Some(gemfile) = &self.bundle_gemfile {
                return Some(if gemfile.file_name() == Some(OsStr::new("gems.rb")) {
                    gemfile.with_file_name("gems.locked")
                } else {
                    let mut lock = gemfile.clone().into_os_string();
                    lock.push(".lock");
                    PathBuf::from(lock)
                });
            }
        }
        discover_bundle_root(&self.project_root).map(|dir| dir.join(GEMFILE_LOCK))
    }

    /// Run `cmd` from the project root with the absolutized env.
    fn apply(&self, cmd: &mut std::process::Command) {
        cmd.current_dir(&self.project_root);
        if let Some(gemfile) = &self.bundle_gemfile {
            cmd.env("BUNDLE_GEMFILE", gemfile);
        }
        if let Some(lock) = &self.bundle_lockfile {
            cmd.env("BUNDLE_LOCKFILE", lock);
        }
    }
}

/// Build the Command used to spawn a Ruby gem-dir script (the plain
/// `name=path` resolver or the bundler probe). Always plain `ruby`:
/// bundler mode no longer runs under `bundle exec`, it reads
/// `Gemfile.lock` itself and asks Ruby only where bundler looks.
fn build_resolver_command(script: &str, anchor: &ResolverAnchor) -> std::process::Command {
    let mut cmd = std::process::Command::new("ruby");
    cmd.args(["-e", script]);
    anchor.apply(&mut cmd);
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

fn find_gemfile(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|dir| dir.join(GEMFILE))
        .find(|path| path.is_file())
}

fn should_use_bundler(bundle_gemfile: Option<&OsStr>, start: &Path) -> bool {
    match bundle_gemfile {
        Some(path) if !path.is_empty() => true,
        _ => find_gemfile(start).is_some(),
    }
}

/// `should_use_bundler` gated by the `--no-bundler` opt-out: the flag
/// wins over both `BUNDLE_GEMFILE` and Gemfile discovery, so `bundle`
/// is never consulted (neither spawned nor suggested) when it is set.
fn resolver_uses_bundler(no_bundler: bool, bundle_gemfile: Option<&OsStr>, start: &Path) -> bool {
    !no_bundler && should_use_bundler(bundle_gemfile, start)
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

/// Bundler's answer to "where do gems live", printed one `key=value` per
/// line (`gem_path` once per root). Run under plain `ruby`:
/// `Bundler.configure` makes `Gem.path` bundler's effective search roots
/// (`BUNDLE_PATH` set means vendor only, never the shared gems); without
/// it `Gem.path` would also show system gems and be more permissive than
/// bundler. Which gem at which version comes from `Gemfile.lock`, read in
/// Rust (`gem_lockfile`).
const BUNDLER_PROBE_SCRIPT: &str = r##"
require "bundler"
Bundler.configure
puts "lockfile=#{Bundler.default_lockfile}"
puts "root=#{Bundler.root}"
puts "bundle_path=#{Bundler.bundle_path}"
Gem.path.each { |path| puts "gem_path=#{path}" }
"##;

/// What a ruby script reads on stdin, plus the lines to echo for it
/// under `CREMA_DEBUG_RESOLVER=1`.
struct ResolverStdin {
    payload: String,
    debug_lines: Vec<String>,
}

/// Spawn `ruby -e <script>`, feed `stdin` (or nothing), and return its
/// stdout. A non-zero exit is `Err("error: ruby exited with failure: ..")`.
/// `stdout_label` names the child's stdout lines in the debug dump
/// (`stdout` for the resolver, `probe` for the bundler probe) so they
/// are never mistaken for the final `name=path` result.
fn run_ruby_script(
    script: &str,
    anchor: &ResolverAnchor,
    extra_args: &[&str],
    stdin: Option<ResolverStdin>,
    stdout_label: &str,
    debug_resolver: bool,
) -> Result<String, String> {
    let mut cmd = build_resolver_command(script, anchor);
    cmd.args(extra_args);
    if stdin.is_none() {
        cmd.stdin(std::process::Stdio::null());
    }
    if debug_resolver {
        eprint_resolver_spawn(&cmd);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("error: failed to spawn ruby: {}", e))?;

    let mut writer = None;
    if let Some(stdin) = stdin {
        if debug_resolver {
            for line in &stdin.debug_lines {
                eprintln!("{line}");
            }
        }
        // Hand stdin to a writer thread so `wait_with_output` can drain
        // stdout/stderr in parallel. Without this, a large payload can
        // fill Ruby's stdout pipe buffer (typically 64 KB) before the
        // parent finishes writing stdin, deadlocking both sides.
        let mut pipe = child
            .stdin
            .take()
            .ok_or_else(|| "error: ruby stdin unavailable".to_string())?;
        writer = Some(std::thread::spawn(move || {
            pipe.write_all(stdin.payload.as_bytes())
        }));
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("error: failed to wait for ruby: {}", e))?;

    if let Some(writer) = writer {
        writer
            .join()
            .map_err(|_| "error: ruby stdin writer panicked".to_string())?
            .map_err(|e| format!("error: failed to write to ruby stdin: {}", e))?;
    }

    if debug_resolver {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            eprintln!("resolver: {} {}", stdout_label, line);
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

    String::from_utf8(output.stdout).map_err(|e| format!("error: ruby stdout is not UTF-8: {}", e))
}

/// Resolve gem dirs for `rbs` and every entry in `entries`. Each entry is
/// `(name, Some(version))` when the caller wants the spec pinned
/// (rbs-collection rubygems-source gems carry their lock pin here), or
/// `(name, None)` when only the name is known (`crema.toml libraries =
/// [...]` and `rbs` itself).
///
/// Two protocols, chosen by `resolver_uses_bundler`:
///
/// - **bundler mode** (a Gemfile is discoverable from the project root
///   or `BUNDLE_GEMFILE`, and not `--no-bundler`): see
///   [`resolve_gem_dirs_from_lockfile`]. `bundle` is never spawned.
/// - **plain mode** (no Gemfile, or `--no-bundler`, or script mode): one
///   `ruby -e` over the whole `entries` list, below.
///
/// Plain mode writes entries to Ruby's stdin as `name\tversion\n`
/// (version empty when `None`); the script writes `name=path\n` for each.
/// When a pinned lookup misses (`Gem::MissingSpecError`, which also
/// covers the `MissingSpecVersionError` subclass raised for a wrong
/// version), the script retries unpinned and, if that succeeds *and* the
/// fallback actually ships type definitions (a `core/` dir for `rbs`
/// itself, `sig/` for everything else — mirroring rbs's own
/// `EnvironmentLoader.gem_sig_path`'s `path.directory?` gate before its
/// analogous stale-pin self-heal in `add_collection`), reports the
/// self-healed dir with the installed version as a third tab-separated
/// field (`name=path\tinstalled_version`) so the caller can warn about
/// the stale pin. An empty path means neither the pinned lookup nor a
/// usable fallback was found.
///
/// `rbs_runtime_deps` (script mode only) additionally walks the rbs
/// gemspec's `runtime_dependencies` transitively and reports every
/// dependency that ships a `sig/` dir and has no same-named
/// `${rbs_gem_dir}/stdlib/` entry as `dep:name=path` (today: prism,
/// which rbs's own sig references but `sig/manifest.yaml` does not
/// list — manifests only name stdlib deps, gem deps live in the
/// gemspec). This is a deliberate divergence from rbs's own
/// `Collection::Sources::Base#dependencies_of`, which only chases
/// `sig/manifest.yaml`: following that alone leaves `Prism::*`
/// unresolved. The gem names come from the installed gemspec, never
/// from a table in crema, so an rbs release that changes its
/// dependencies is followed without a code change. The normal path
/// never passes the flag, so its stdin / stdout stay byte-identical.
fn resolve_gem_dirs(
    entries: &[(String, Option<String>)],
    anchor: &ResolverAnchor,
    no_bundler: bool,
    rbs_runtime_deps: bool,
    warnings: &mut Vec<String>,
) -> Result<ResolvedGemDirs, String> {
    let debug_resolver = std::env::var_os("CREMA_DEBUG_RESOLVER").is_some_and(|v| v == "1");
    if anchor.uses_bundler(no_bundler) {
        // Script mode forces `no_bundler`, so the two never meet.
        debug_assert!(!rbs_runtime_deps);
        return resolve_gem_dirs_from_lockfile(entries, anchor, debug_resolver);
    }
    let script = r##"
$stdin.each_line do |line|
  name, version = line.chomp.split("\t", 2)
  next if name.nil? || name.empty?
  pinned = version && !version.empty?
  begin
    spec = pinned ? Gem::Specification.find_by_name(name, version) : Gem::Specification.find_by_name(name)
    puts "#{name}=#{spec.gem_dir}"
    if name == "rbs" && ARGV.include?("--rbs-runtime-deps")
      seen = {}
      queue = spec.runtime_dependencies.map(&:name)
      deps = []
      until queue.empty?
        dep = queue.shift
        next if seen[dep]
        seen[dep] = true
        begin
          dep_spec = Gem::Specification.find_by_name(dep)
        rescue Gem::MissingSpecError
          deps << "dep:#{dep}="
          next
        end
        queue.concat(dep_spec.runtime_dependencies.map(&:name))
        next if File.directory?(File.join(spec.gem_dir, "stdlib", dep))
        next unless File.directory?(File.join(dep_spec.gem_dir, "sig"))
        deps << "dep:#{dep}=#{dep_spec.gem_dir}"
      end
      puts deps.sort
    end
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

    let mut payload = String::new();
    for (name, version) in entries {
        payload.push_str(name);
        payload.push('\t');
        if let Some(v) = version {
            payload.push_str(v);
        }
        payload.push('\n');
    }
    // Dumped from the structured `entries`, not by re-splitting
    // `payload` on `\n` — a `crema.toml` `libraries` name or a
    // lock-pinned version is unvalidated free text (src/config.rs's
    // `libraries: Vec<String>`), and an embedded newline there would
    // otherwise fabricate an extra, misleading "resolver: stdin"
    // line when troubleshooting via this exact flag.
    let debug_lines = entries
        .iter()
        .map(|(name, version)| {
            format!(
                "resolver: stdin {}\t{}",
                escape_control_chars(name),
                version
                    .as_deref()
                    .map(escape_control_chars)
                    .unwrap_or_default()
            )
        })
        .collect();
    let extra_args: &[&str] = if rbs_runtime_deps {
        // `--` ends ruby's own option parsing so the flag lands in ARGV.
        &["--", "--rbs-runtime-deps"]
    } else {
        &[]
    };
    let stdout = run_ruby_script(
        script,
        anchor,
        extra_args,
        Some(ResolverStdin {
            payload,
            debug_lines,
        }),
        "stdout",
        debug_resolver,
    )?;

    let mut rbs_gem_dir: Option<PathBuf> = None;
    let mut libraries: HashMap<String, Option<PathBuf>> = HashMap::new();
    let mut stale: HashMap<String, String> = HashMap::new();
    let mut rbs_runtime_dep_dirs: Vec<(String, PathBuf)> = Vec::new();
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(dep) = line.strip_prefix("dep:") {
            let Some((name, path)) = dep.split_once('=') else {
                return Err(format!(
                    "error: malformed ruby resolver output line: {}",
                    line
                ));
            };
            if path.is_empty() {
                // rubygems normally guarantees runtime deps are
                // installed; a miss is worth a line, not a silent skip.
                warn_and_record(
                    warnings,
                    format!(
                        "warning: rbs runtime dependency {} is not installed; its signatures are not loaded",
                        name
                    ),
                );
            } else {
                rbs_runtime_dep_dirs.push((name.to_string(), PathBuf::from(path)));
            }
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

    finish_resolved(rbs_gem_dir, libraries, stale, rbs_runtime_dep_dirs)
}

/// `rbs` is the one gem every run needs: without it there is no core RBS,
/// so a missing `rbs_gem_dir` is the (warning-prefixed) missing-gem error
/// whichever protocol produced the result.
fn finish_resolved(
    rbs_gem_dir: Option<PathBuf>,
    libraries: HashMap<String, Option<PathBuf>>,
    stale: HashMap<String, String>,
    rbs_runtime_deps: Vec<(String, PathBuf)>,
) -> Result<ResolvedGemDirs, String> {
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
        rbs_runtime_deps,
    })
}

/// Bundler-mode resolution: one probe for bundler's roots, then
/// `Gemfile.lock` read as text and each gem placed by `gem_lockfile`.
/// Replaces `bundle exec ruby`, whose cost is dominated by bundler
/// re-resolving the whole lockfile in every cold run.
///
/// The lockfile is the authority on versions. A lock-pinned entry whose
/// pin differs from the locked version self-heals to the locked version
/// (reported in `stale`) only if that install ships type definitions
/// (`core/` for `rbs`, `sig/` otherwise — the same gate as the plain
/// resolver's fallback). A gem the lockfile does not list is a miss, as
/// bundler hides it. A gem crema reads that the lockfile lists but no
/// probe root holds is an error: crema cannot type-check against code
/// that is not installed, and `bundle install` is the fix. Gems crema
/// does not read are never looked at.
fn resolve_gem_dirs_from_lockfile(
    entries: &[(String, Option<String>)],
    anchor: &ResolverAnchor,
    debug_resolver: bool,
) -> Result<ResolvedGemDirs, String> {
    let probe_stdout = run_ruby_script(
        BUNDLER_PROBE_SCRIPT,
        anchor,
        &[],
        None,
        "probe",
        debug_resolver,
    )?;
    let roots = crema::gem_lockfile::parse_probe_output(&probe_stdout)?;
    let lock_text = match fs::read_to_string(&roots.lockfile) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "error: {} not found. Run `bundle install` to create it (or pass --no-bundler to resolve gems without bundler)",
                roots.lockfile.display()
            ));
        }
        Err(e) => {
            return Err(format!(
                "error: failed to read {}: {}",
                roots.lockfile.display(),
                e
            ));
        }
    };
    let lock = crema::gem_lockfile::Lockfile::parse(&lock_text);

    let mut rbs_gem_dir: Option<PathBuf> = None;
    let mut libraries: HashMap<String, Option<PathBuf>> = HashMap::new();
    let mut stale: HashMap<String, String> = HashMap::new();
    let mut not_installed: Vec<String> = Vec::new();
    for (name, pin) in entries {
        use crema::gem_lockfile::Located;
        let mut healed_from: Option<String> = None;
        let found = match lock.locate(name, &roots) {
            Located::NotLocked => None,
            Located::NotInstalled { version } => {
                not_installed.push(format!("{name} ({version})"));
                continue;
            }
            Located::Installed { version, dir } => {
                let pinned_elsewhere = pin
                    .as_deref()
                    .is_some_and(|p| !p.is_empty() && p != version);
                let sig_dir = if name == "rbs" { "core" } else { "sig" };
                if pinned_elsewhere && !dir.join(sig_dir).is_dir() {
                    None
                } else {
                    if pinned_elsewhere {
                        healed_from = Some(version);
                    }
                    Some(dir)
                }
            }
        };
        if debug_resolver {
            eprintln!(
                "resolver: stdout {}={}{}",
                name,
                found
                    .as_ref()
                    .map(|d| d.display().to_string())
                    .unwrap_or_default(),
                healed_from
                    .as_ref()
                    .map(|v| format!("\t{v}"))
                    .unwrap_or_default()
            );
        }
        if name == "rbs" {
            if let Some(dir) = found {
                rbs_gem_dir = Some(dir);
                if let Some(version) = healed_from {
                    stale.insert(name.clone(), version);
                }
            }
        } else {
            let has_dir = found.is_some();
            libraries.insert(name.clone(), found);
            if let (true, Some(version)) = (has_dir, healed_from) {
                stale.insert(name.clone(), version);
            }
        }
    }
    if !not_installed.is_empty() {
        return Err(format!(
            "error: {} lists gems that are not installed: {}. Run `bundle install` (or pass --no-bundler to resolve gems without bundler)",
            roots.lockfile.display(),
            not_installed.join(", ")
        ));
    }
    finish_resolved(rbs_gem_dir, libraries, stale, Vec::new())
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
/// invalidation key (crema version and binary identity,
/// lockfile/crema.toml content, sig paths) — none of which change when the *installed* gem state does
/// (`bundle install` with an already-correct lockfile, or a plain `gem
/// install`). A resolved-but-later-fixed stale-pin
/// warning keeps replaying verbatim until something touches
/// one of those key inputs or `--refresh-g-snapshot` is passed. This
/// mirrors the pre-existing tradeoff for the G layer's actual RBS
/// content (a warm hit never re-reads gem sig files either); it is not
/// new to warning replay.
fn get_gem_dirs(
    entries: &[(String, Option<String>)],
    rbs_collection_lock: Option<&Path>,
    anchor: &ResolverAnchor,
    no_bundler: bool,
    rbs_runtime_deps: bool,
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
    let resolved = resolve_gem_dirs(
        &resolver_input,
        anchor,
        no_bundler,
        rbs_runtime_deps,
        warnings,
    )?;
    emit_stale_pin_warnings(
        &resolved,
        entries,
        rbs_collection_lock,
        anchor,
        no_bundler,
        warnings,
    );
    Ok(resolved)
}

/// Every `${rbs_gem_dir}/stdlib/<name>/0/` directory, sorted by name so
/// load order (and thus any duplicate-declaration error text) is
/// stable across runs. The `0/` level is rbs's stdlib version slot
/// (`Collection::Sources::Stdlib` / `EnvironmentLoader`); only that
/// slot exists today. A missing or unreadable `stdlib/` yields nothing.
fn stdlib_sig_dirs(rbs_gem_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(rbs_gem_dir.join("stdlib")) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path().join("0"))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Record one warning per lock-pinned gem whose exact pinned version
/// isn't installed but a self-healed fallback (installed) version
/// resolved instead (`gem_dirs.stale`, populated by `resolve_gem_dirs`).
fn emit_stale_pin_warnings(
    gem_dirs: &ResolvedGemDirs,
    entries: &[(String, Option<String>)],
    rbs_collection_lock: Option<&Path>,
    anchor: &ResolverAnchor,
    no_bundler: bool,
    warnings: &mut Vec<String>,
) {
    if gem_dirs.stale.is_empty() {
        return;
    }
    let lock_name = rbs_collection_lock
        .and_then(|p| p.file_name())
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "rbs_collection.lock.yaml".to_string());
    let sync_command =
        collection_update_command(no_bundler, anchor, std::env::var_os("PATH").as_deref());

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
/// PATH. A Gemfile alone isn't enough — crema itself never spawns
/// `bundle`, so nothing has shown it works, and suggesting `bundle exec
/// ...` on a machine without it would point the user at a command that
/// does not exist. Likewise under `--no-bundler` (`no_bundler`): bundler
/// was deliberately bypassed, so the suggestion follows the resolver and
/// drops the `bundle exec`. The Gemfile is looked for from the project
/// root, as the resolver does.
fn collection_update_command(
    no_bundler: bool,
    anchor: &ResolverAnchor,
    path_env: Option<&OsStr>,
) -> &'static str {
    let use_bundler = anchor.uses_bundler(no_bundler) && bundle_on_path(path_env);
    if use_bundler {
        "bundle exec rbs collection update"
    } else {
        "rbs collection update"
    }
}

/// Cheap PATH scan for a `bundle` executable, without spawning a
/// process, for `collection_update_command`.
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

/// A `.rb` file found by [`collect_rb_files`]. `joinable` is true when
/// every component from the walk root down to the file is a non-symlink
/// entry whose type `read_dir` reported — then `realpath(path)` is
/// `realpath(walk root)` + `path` relative to the root (the walk never
/// descends a symlinked directory and `read_dir` never yields `.`/`..`),
/// so [`canonicalize_walked`] can skip the per-file realpath (ADR-0035
/// Decision 2).
struct WalkedRb {
    path: PathBuf,
    joinable: bool,
}

/// One `check` entry resolved by [`expand_targets`]: a file passed
/// through as given, or a directory with the `.rb` files walked under it.
enum ExpandedTarget {
    File(PathBuf),
    Dir { root: PathBuf, files: Vec<WalkedRb> },
}

/// The directory listings a walk produced, for the ingest cache's
/// write-back (see [`DirListing`]), plus how many it took from the
/// cache instead of `read_dir`.
#[derive(Default)]
struct WalkedDirs {
    listings: Vec<DirListing>,
    reused: usize,
}

/// Resolve CLI `check` arguments: directories are walked recursively
/// (`.rb` only), files pass through unchanged, and a missing path is
/// fatal so the user sees their typo. Exits the process on I/O errors
/// rather than returning, matching the surrounding CLI style. `cached`
/// is the previous run's directory listings (ingest cache), consulted
/// by [`walk_rb_dir`]; every listing this walk ends up with goes to
/// `dirs`.
fn expand_targets(
    targets: &[PathBuf],
    cached: Option<&FxHashMap<String, DirListing>>,
    dirs: &mut WalkedDirs,
) -> Vec<ExpandedTarget> {
    let mut expanded = Vec::new();
    for target in targets {
        match fs::metadata(target) {
            Ok(m) if m.is_dir() => {
                let mut files = Vec::new();
                collect_rb_files(target, true, cached, &mut files, dirs);
                expanded.push(ExpandedTarget::Dir {
                    root: target.clone(),
                    files,
                });
            }
            Ok(_) => expanded.push(ExpandedTarget::File(target.clone())),
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

/// Canonicalize every file of [`expand_targets`]'s result, in order.
/// Runs after the whole walk so every walk warning is printed before a
/// resolve error exits.
fn canonicalize_expanded(expanded: &[ExpandedTarget]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for target in expanded {
        match target {
            ExpandedTarget::File(path) => out.push(canonicalize_or_exit(path)),
            ExpandedTarget::Dir { root, files } => canonicalize_walked(root, files, &mut out),
        }
    }
    out
}

/// Canonicalize the files [`collect_rb_files`] found under `root`: a
/// joinable file costs no syscall (one realpath of `root`, made lazily on
/// the first one), any other is resolved on its own.
fn canonicalize_walked(root: &Path, files: &[WalkedRb], out: &mut Vec<PathBuf>) {
    let mut canonical_root: Option<PathBuf> = None;
    for file in files {
        if file.joinable {
            let base = canonical_root.get_or_insert_with(|| canonicalize_or_exit(root));
            let rel = file
                .path
                .strip_prefix(root)
                .expect("collect_rb_files only yields paths under its root");
            out.push(base.join(rel));
        } else {
            out.push(canonicalize_or_exit(&file.path));
        }
    }
}

/// Recursively collect every `.rb` file under `dir` into `out`. An
/// unreadable directory or entry warns and is skipped — a partial walk
/// under-represents `out` rather than aborting the run. `dir` is walked
/// by its nominal spelling, so warnings name the path the user wrote.
/// `joinable` says whether `dir` itself is reachable from the walk root
/// through non-symlink entries only (see [`WalkedRb`]).
///
/// Subdirectories are walked on the global pool, so the pool must exist
/// before the first walk (`run_check` builds it up front). Files and
/// warnings still come out in the serial `read_dir` DFS order: each
/// directory's results are concatenated in entry order, and the
/// warnings are printed once the whole walk is done.
fn collect_rb_files(
    dir: &Path,
    joinable: bool,
    cached: Option<&FxHashMap<String, DirListing>>,
    out: &mut Vec<WalkedRb>,
    dirs: &mut WalkedDirs,
) {
    let mut walked = Vec::new();
    walk_rb_dir(dir, joinable, cached, &mut walked);
    for item in walked {
        match item {
            Walked::Rb(file) => out.push(file),
            Walked::Warning(warning) => eprintln!("{warning}"),
            Walked::Dir { listing, reused } => {
                dirs.listings.push(listing);
                dirs.reused += reused as usize;
            }
        }
    }
}

/// One result of [`walk_rb_dir`], in walk order. A directory's listing
/// comes out before anything found under it.
enum Walked {
    Rb(WalkedRb),
    Warning(String),
    Dir { listing: DirListing, reused: bool },
}

/// One entry of a directory being walked, before its subtree is.
enum WalkEntry {
    Done(Walked),
    Dir { path: PathBuf, joinable: bool },
}

/// Walk one directory: from its cached listing when `cached` has one
/// under the same spelling with the same stat, else by `read_dir`. The
/// stat is taken first so a change landing before the `read_dir` is
/// seen as stale next run (see the `ingest_cache` module doc).
fn walk_rb_dir(
    dir: &Path,
    joinable: bool,
    cached: Option<&FxHashMap<String, DirListing>>,
    out: &mut Vec<Walked>,
) {
    use rayon::prelude::*;

    let stat = fs::metadata(dir).ok().map(|m| FileStat::of(&m));
    // A non-UTF-8 directory path is simply never cached.
    let key = dir.to_str();
    let cached_listing = match (cached, stat, key) {
        (Some(cached), Some(stat), Some(key)) => cached.get(key).filter(|l| l.stat == stat),
        _ => None,
    };
    let mut items = Vec::new();
    let listing = if let Some(listing) = cached_listing {
        for entry in &listing.entries {
            let path = dir.join(&entry.name);
            let entry_joinable = entry.joinable && joinable;
            if entry.is_dir {
                items.push(WalkEntry::Dir {
                    path,
                    joinable: entry_joinable,
                });
            } else {
                items.push(WalkEntry::Done(Walked::Rb(WalkedRb {
                    path,
                    joinable: entry_joinable,
                })));
            }
        }
        Some(Walked::Dir {
            listing: DirListing {
                path: listing.path.clone(),
                stat: listing.stat,
                entries: listing.entries.clone(),
            },
            reused: true,
        })
    } else {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                out.push(Walked::Warning(format!(
                    "warning: cannot read directory {}: {}",
                    dir.display(),
                    e
                )));
                return;
            }
        };
        // `None` once the directory turns out not to be cacheable: a
        // warning (not stored, so the next run warns again) or a
        // non-UTF-8 entry name.
        let mut fresh: Option<Vec<DirEntry>> = Some(Vec::new());
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    items.push(WalkEntry::Done(Walked::Warning(format!(
                        "warning: cannot read entry in {}: {}",
                        dir.display(),
                        e
                    ))));
                    fresh = None;
                    continue;
                }
            };
            let path = entry.path();
            // `entry.file_type()` reads the type `read_dir` already
            // returned — usually free of an extra stat, unlike
            // `path.is_dir()` (always one). A symlink costs one stat to
            // tell dir from file: a symlinked directory is pruned (Steep's
            // `**/*.rb` glob never descends one — Ruby's `**` skips symlink
            // dirs, which also keeps a self-loop link from walking until
            // PATH_MAX), a symlinked `.rb` file is still collected (same
            // rule as `file_finder::each_file` on the `.rbs` side).
            //
            // `entry_joinable` is false for a symlinked file and for an
            // entry whose type is unknown: the `Err` arm's `is_dir()`
            // follows links, so it may even descend a symlinked directory,
            // and nothing below it can be derived from the root.
            let (is_dir, inherits_joinable) = match entry.file_type() {
                Ok(ft) if ft.is_symlink() => {
                    if path.is_dir() {
                        continue;
                    }
                    (false, false)
                }
                Ok(ft) => (ft.is_dir(), true),
                Err(_) => (path.is_dir(), false),
            };
            let entry_joinable = inherits_joinable && joinable;
            let is_rb = !is_dir && path.extension().is_some_and(|ext| ext == "rb");
            if !is_dir && !is_rb {
                continue;
            }
            if let Some(fresh_entries) = &mut fresh {
                match entry.file_name().to_str() {
                    Some(name) => fresh_entries.push(DirEntry {
                        name: name.to_string(),
                        is_dir,
                        joinable: inherits_joinable,
                    }),
                    None => fresh = None,
                }
            }
            if is_dir {
                items.push(WalkEntry::Dir {
                    path,
                    joinable: entry_joinable,
                });
            } else {
                items.push(WalkEntry::Done(Walked::Rb(WalkedRb {
                    path,
                    joinable: entry_joinable,
                })));
            }
        }
        match (fresh, stat, key) {
            (Some(entries), Some(stat), Some(key)) => Some(Walked::Dir {
                listing: DirListing {
                    path: key.to_string(),
                    stat,
                    entries,
                },
                reused: false,
            }),
            _ => None,
        }
    };
    out.extend(listing);
    // Each subtree goes into a Vec of its own and is appended in entry
    // order, whichever finishes first.
    let subtrees: Vec<Vec<Walked>> = items
        .par_iter()
        .map(|item| match item {
            WalkEntry::Dir { path, joinable } => {
                let mut subtree = Vec::new();
                walk_rb_dir(path, *joinable, cached, &mut subtree);
                subtree
            }
            WalkEntry::Done(_) => Vec::new(),
        })
        .collect();
    for (item, mut subtree) in items.into_iter().zip(subtrees) {
        match item {
            WalkEntry::Done(walked) => out.push(walked),
            WalkEntry::Dir { .. } => out.append(&mut subtree),
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
/// and, in that case, appends a hint listing both trial-run paths that
/// work without crema.toml: zero-config `-e` (ADR-0029 §5 amendment) and
/// `--script FILE`. The hint does not depend on whether files were
/// passed — bare `crema check` and `crema check a.rb` print the same text.
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
        eprintln!("hint: to try crema without a config file:");
        eprintln!();
        eprintln!("    crema check -e '1 + 1'        # inline Ruby");
        eprintln!(
            "    crema check --script FILE     # one script using only the stdlib (no gems or project files)"
        );
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
    cached_dirs: Option<&FxHashMap<String, DirListing>>,
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
            // The scope walk already produced the listings of every
            // directory under `check`; a CLI directory is usually one of
            // them, so its listings are read but not written back.
            let mut dir_files = Vec::new();
            let mut dirs = WalkedDirs::default();
            collect_rb_files(target, true, cached_dirs, &mut dir_files, &mut dirs);
            let mut canonical = Vec::new();
            canonicalize_walked(target, &dir_files, &mut canonical);
            for canon in canonical {
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

/// Discover an `rbs_collection.lock.yaml` by walking up from the
/// project root, stopping at `.git` — so a subdirectory run (or
/// `--config` naming another project) reads the lock a root run reads,
/// the one the G-snapshot key hashes. Errors are formatted into a single
/// string so the caller can route them through the same
/// `error:`/`warning:` stderr convention as `get_gem_dirs`. A missing
/// config (no `rbs_collection.yaml` along the walk) and a config without
/// a lock both return `Ok(None)` silently — rbs CLI parity.
fn discover_collection_lockfile(project_root: &Path) -> Result<Option<DiscoveredLockfile>, String> {
    rbs_collection::discover_lockfile(project_root).map_err(|e| e.to_string())
}

/// Load file config according to the top-level `--config <PATH>` flag.
/// When the flag is set, the named path is loaded verbatim (NotFound is
/// a hard exit-2 error). When absent, fall back to cwd discovery
/// (NotFound silently yields None).
fn load_file_config(cli_config: Option<&Path>) -> Option<Config> {
    load_file_config_with_dir(cli_config).config
}

/// Base directory every output path (`"file"`, fingerprint material,
/// baseline rows, extract `files` keys) is relativized against: the
/// canonicalized crema.toml dir, so the output does not depend on which
/// subdirectory crema runs from (ADR-0029 §7 note 3, amended
/// 2026-09-30). `-e` without a config and `--script` already carry
/// `project_root` = cwd. Canonicalized so it string-prefix-matches the
/// canonical paths diagnostics carry internally; falls back to the
/// canonicalized cwd, and to `None` (paths left untouched) when both
/// fail. Files outside the base keep their absolute path.
fn output_display_base(project_root: &Path) -> Option<PathBuf> {
    project_root.canonicalize().ok().or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|d| d.canonicalize().ok())
    })
}

/// The config a run loaded, where it came from, and the project root it
/// anchors. `--config <PATH>` and walk-up discovery are the same here:
/// `project_root` is the loaded file's directory and every relative path
/// inside the file has already been resolved against it. With no config
/// file, `project_root` is the cwd and `path` is `None`.
struct LoadedFileConfig {
    config: Option<Config>,
    project_root: PathBuf,
    /// The file `config` was read from — the one the G-snapshot key must
    /// hash, whatever its name.
    path: Option<PathBuf>,
}

fn load_file_config_with_dir(cli_config: Option<&Path>) -> LoadedFileConfig {
    let result = match cli_config {
        Some(path) => Config::load_from_file(path).map(|(cfg, dir)| {
            let file = dir.join(path.file_name().unwrap_or(path.as_os_str()));
            LoadedFileConfig {
                config: Some(cfg),
                project_root: dir,
                path: Some(file),
            }
        }),
        None => {
            let cwd = std::env::current_dir().map_err(ConfigError::Io);
            match cwd {
                Ok(cwd) => Config::discover_walking_up(&cwd).map(|found| match found {
                    Some((cfg, dir)) => LoadedFileConfig {
                        config: Some(cfg),
                        path: Some(dir.join(CONFIG_FILE)),
                        project_root: dir,
                    },
                    None => LoadedFileConfig {
                        config: None,
                        project_root: cwd,
                        path: None,
                    },
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
/// Decision 4: lockfiles, crema version, crema.toml, sig path set) plus
/// the `--no-bundler` resolver mode. The crema version is refined to the
/// running binary's size + mtime
/// ([`crema::snapshot::invalidation::crema_version`]), so another build
/// of the same version misses rather than replaying this build's caches.
/// Gemfile.lock is the one the resolver reads
/// ([`ResolverAnchor::gemfile_lock`]): searched from the project root
/// so subdirectory execution and root execution compute the same key,
/// or named by `BUNDLE_GEMFILE` / `BUNDLE_LOCKFILE`. The config
/// content is read from `config_path`, the file the run actually loaded
/// (`--config` names any file; `None` when no config was loaded).
fn compute_g_snapshot_key(
    sig_dirs: &[PathBuf],
    rbs_collection_lock: Option<&Path>,
    anchor: &ResolverAnchor,
    config_path: Option<&Path>,
    no_bundler: bool,
) -> crema::snapshot::invalidation::InvalidationKey {
    let gemfile_lock_content = anchor
        .gemfile_lock(no_bundler)
        .and_then(|lock| fs::read(lock).ok());
    let rbs_collection_lock_content = rbs_collection_lock.and_then(|p| fs::read(p).ok());
    let crema_toml_content = config_path.and_then(|p| fs::read(p).ok());
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
        no_bundler,
    })
}

/// Every `.rbs` file under a user sig dir, same walk as
/// [`crema::environment::draft::EnvironmentDraft::load_dir`] with
/// `skip_hidden: false` but enumeration-only (no read/parse) — ADR-0028 S8
/// change detection needs the current path set, not file content. An
/// unreadable root contributes nothing, as before.
fn collect_rbs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    out.extend(crema::file_finder::each_file(dir, false).unwrap_or_default());
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
/// invalidation key to `.crema/g_snapshot_v1.bin`.
///
/// This is the write-path validation vehicle for ADR-0028 slice 1b:
/// the check pipeline never calls it; slice 1c wires reading and decides
/// when checks write snapshots. The gem-loading sequence intentionally
/// mirrors `Commands::Check` (kept duplicated because slice scope
/// forbids touching the check path; slice 1c reconciles the two).
fn run_snapshot_dump(project_root: &Path, cli_config: Option<&Path>, no_bundler: bool) {
    // Taken before the chdir: a relative `BUNDLE_GEMFILE` /
    // `BUNDLE_LOCKFILE` names a file under the directory crema was
    // started from, as in `run_check`.
    let launch_cwd = std::env::current_dir().ok();
    if let Err(e) = std::env::set_current_dir(project_root) {
        eprintln!(
            "error: cannot enter project root {}: {}",
            project_root.display(),
            e
        );
        process::exit(2);
    }
    // `project_root` may be relative (`dump .`); the Gemfile walk-up
    // needs the absolute directory to see its ancestors.
    let absolute_root = std::env::current_dir().unwrap_or_else(|_| project_root.to_path_buf());
    let anchor = ResolverAnchor::new(
        &absolute_root,
        std::env::var_os("BUNDLE_GEMFILE").as_deref(),
        std::env::var_os("BUNDLE_LOCKFILE").as_deref(),
        launch_cwd.as_deref(),
    );

    let LoadedFileConfig {
        config: file_config,
        path: config_path,
        ..
    } = load_file_config_with_dir(cli_config);
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
        CollectionMode::Auto => match discover_collection_lockfile(project_root) {
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
        &anchor,
        no_bundler,
        false,
        &mut g_warnings,
    ) {
        Ok(gem_dirs) => {
            let core_dir = gem_dirs.rbs_gem_dir.join("core");
            if core_dir.is_dir() {
                g_dirs.push(core_dir.clone());
                if let Err(e) = draft.load_dir(&core_dir, true) {
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
                if let Err(e) = draft.load_dir(dir, true) {
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
                if let Err(e) = draft.load_dir(dir, true) {
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
        &anchor,
        config_path.as_deref(),
        no_bundler,
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

/// Everything `Commands::Check` carries, bundled so `run_check` can be
/// shared by `crema check` and `crema extract` — the latter runs the
/// same environment-building pipeline with default flags, a suppressed
/// diagnostic stream, and its own per-file phase (see the [`RunMode`]
/// branch after the env phase inside `run_check`).
struct CheckInvocation {
    files: Vec<PathBuf>,
    eval: Option<String>,
    script: Option<PathBuf>,
    sig_dirs: Vec<PathBuf>,
    add_sig_dirs: Vec<PathBuf>,
    inline: Option<bool>,
    collection: Option<PathBuf>,
    no_g_snapshot: bool,
    refresh_g_snapshot: bool,
    tamp: bool,
    update_baseline: bool,
    no_baseline: bool,
    threads: Option<usize>,
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
fn run_check(cli_config: Option<&Path>, no_bundler: bool, args: CheckInvocation, mode: RunMode) {
    // Anchors the check-timing line: `setup` is everything before
    // ingest (config, collection, gem dirs, snapshot open).
    let t_run = std::time::Instant::now();
    let CheckInvocation {
        files: cli_targets,
        eval,
        script,
        sig_dirs,
        add_sig_dirs,
        inline,
        collection,
        no_g_snapshot,
        refresh_g_snapshot,
        tamp,
        update_baseline,
        no_baseline,
        threads,
    } = args;
    // Pool size for the parallel phases — the `.rb` walk, ingest
    // (ADR-0033) and check (ADR-0034): `--threads N` wins, then `CREMA_THREADS=<n>` (what
    // the perf gate exports so User time keeps measuring algorithmic
    // cost single-threaded), then rayon's default (available
    // parallelism).
    //
    // Main joins the pool as its thread 0 (ADR-0035). With a one-thread
    // pool main then runs every ingest and check job itself, so the
    // ASTs and declarations are allocated on the thread that later
    // drops them: allocating on a worker and freeing on main costs the
    // check phase ~3% (mimalloc frees across threads), and the
    // single-thread run is the perf gate's and the Steep / Sorbet
    // comparison's yardstick.
    // Built before anything touches rayon: the first rayon call would
    // create a default global pool, and `build_global` would then fail —
    // dropping `--threads` / `CREMA_THREADS` and main's thread 0. Every
    // phase stays correct without either, so a release build carries
    // on; a debug build (what the CLI tests run) asserts, so moving any
    // rayon use ahead of this point fails the suite.
    let threads = threads.or_else(|| {
        std::env::var_os("CREMA_THREADS")
            .and_then(|v| v.to_str().and_then(|v| v.parse::<usize>().ok()))
    });
    let pool_built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.unwrap_or(0))
        .stack_size(POOL_STACK_BYTES)
        .use_current_thread()
        .build_global();
    debug_assert!(
        pool_built.is_ok(),
        "rayon ran before the global pool was built: {pool_built:?}"
    );
    // Captured once, ahead of `eval`'s move into the `sources`
    // construction below (ADR-0029 §5) — every later gate reads
    // this instead of re-borrowing the (by-then-moved) `eval`.
    let eval_active = eval.is_some();
    if eval_active && !cli_targets.is_empty() {
        eprintln!("error: -e and files are mutually exclusive");
        process::exit(2);
    }
    // `--update-baseline` must see the whole scope (a filtered subset
    // would make every other row "new" on the next run) and owns the
    // tamp rows (stdout vs file, and disabling vs updating contradict).
    if update_baseline {
        let conflicts = [
            ("files", !cli_targets.is_empty()),
            ("-e", eval_active),
            ("--tamp", tamp),
            ("--no-baseline", no_baseline),
        ];
        for (name, set) in conflicts {
            if set {
                eprintln!("error: --update-baseline cannot be combined with {}", name);
                process::exit(2);
            }
        }
    }
    // Script mode (`--script FILE`): every flag below would be
    // silently overridden by the mode's fixed choices (no config, no
    // sig, inline on, no snapshot), so their co-use is an error rather
    // than a no-op (specs/config.md Design Goal 5).
    let script_active = script.is_some();
    if script_active {
        let conflicts: [(&str, bool); 11] = [
            ("files", !cli_targets.is_empty()),
            ("-e", eval_active),
            ("--config", cli_config.is_some()),
            ("--sig", !sig_dirs.is_empty()),
            ("--add-sig", !add_sig_dirs.is_empty()),
            ("--collection", collection.is_some()),
            ("--inline", inline.is_some()),
            ("--no-g-snapshot", no_g_snapshot),
            ("--refresh-g-snapshot", refresh_g_snapshot),
            ("--update-baseline", update_baseline),
            ("--no-baseline", no_baseline),
        ];
        for (name, set) in conflicts {
            if set {
                eprintln!("error: --script cannot be combined with {}", name);
                process::exit(2);
            }
        }
        // One script means one file: `expand_targets` below would
        // otherwise walk a directory and quietly turn script mode into
        // a scope-less multi-file check (crema-review finding).
        if let Some(path) = &script
            && !path.is_file()
        {
            eprintln!("error: --script expects a Ruby file: {}", path.display());
            process::exit(2);
        }
    }

    // Script mode never runs the crema.toml walk-up: the project's
    // config is not "found and ignored", it is not looked for at all
    // (a malformed crema.toml in cwd must not break a script check).
    // `project_root` only feeds paths that are disabled below
    // (snapshot) and `ignore` matching over an
    // empty list, so cwd is a fine stand-in.
    let LoadedFileConfig {
        config: file_config,
        project_root,
        path: config_path,
    } = if script_active {
        let cwd = std::env::current_dir().unwrap_or_else(|e| {
            eprintln!("error: cannot read current dir: {}", e);
            process::exit(2);
        });
        LoadedFileConfig {
            config: None,
            project_root: cwd,
            path: None,
        }
    } else {
        load_file_config_with_dir(cli_config)
    };
    let resolver_anchor = ResolverAnchor::from_env(&project_root);

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
        None if !eval_active && !script_active => print_no_check_target_configured_and_exit(true),
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
    // Script mode has no project to persist into either.
    let no_g_snapshot = no_g_snapshot || !has_config;
    // Script mode's resolver is always plain `ruby`: a script has no
    // bundle, and a Gemfile discoverable from cwd belongs to whatever
    // project the user happens to be standing in.
    let no_bundler = no_bundler || script_active;

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

    // Baseline (crema.toml `baseline`): resolve the file once, up
    // front, so a missing or malformed file fails before any checking
    // happens (spec Design Goal 5 — never a silent no-op). Extract mode
    // discards diagnostics and never consults it. `--no-baseline`
    // disables matching for this run only.
    // `--tamp` is the raw projection `--update-baseline` writes, so it
    // bypasses matching the same way `--no-baseline` does.
    let baseline_path = match (&mode, &resolved.baseline) {
        (RunMode::Check, Some(path)) if !no_baseline && !tamp => Some(path.clone()),
        _ => None,
    };
    if update_baseline && baseline_path.is_none() {
        eprintln!("error: --update-baseline needs `baseline = true` (or a path) in crema.toml");
        process::exit(2);
    }
    let baseline = match (&baseline_path, update_baseline) {
        (Some(path), false) => match load_baseline(path) {
            Ok(b) => Some(b),
            Err(msg) => {
                eprintln!("{}", msg);
                process::exit(2);
            }
        },
        _ => None,
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

    // ADR-0017 Phase 5b switchover: production runs through the
    // Phase 5a `crate::definition_builder::DefinitionBuilder` built from a
    // frozen `Environment`. The `LegacyEnvironmentBuffer` is kept
    // alive only so the parallel-push entry points
    // (`load_dir_with_draft`, `load_inline_annotations_with_draft`)
    // can route declarations into the draft alongside their legacy
    // sinks; we never call `unresolved.resolve()` here. The draft
    let mut draft = crema::environment::draft::EnvironmentDraft::new();

    let lockfile = match &resolved.collection {
        // Script mode: an rbs_collection.lock.yaml in cwd belongs to
        // the surrounding project, not to the script.
        CollectionMode::Auto if script_active => None,
        CollectionMode::Auto => match discover_collection_lockfile(&project_root) {
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
    // The ingest cache (ADR-0036 Decision 4-1) keys on the same inputs
    // plus the CLI inline override, so the base key is computed once
    // whether or not the G snapshot is in play.
    let base_invalidation_key = compute_g_snapshot_key(
        &resolved.sig_dirs,
        rbs_collection_lock_path.as_deref(),
        &resolver_anchor,
        config_path.as_deref(),
        no_bundler,
    );
    let g_snapshot_key = (!no_g_snapshot).then_some(base_invalidation_key);
    let g_snapshot_path = project_root.join(G_SNAPSHOT_FILE);
    let ingest_cache_key = crema::ingest_cache::derive_key(&base_invalidation_key, resolved.inline);
    let ingest_cache_path = project_root.join(crema::ingest_cache::INGEST_CACHE_FILE);

    // Per-file ingest cache (ADR-0036 Decision 4-1), opened before the
    // scope walk so the walk can reuse its directory listings. Lookup
    // serves only files outside the diagnostic filter: a target's AST
    // is needed for Layer 3, so it is always read and parsed. A run
    // without a filter (every file is a target) looks nothing up but
    // still writes, so a project-wide check warms the cache the editor
    // hook's single-file checks then hit. Writing shares the G
    // snapshot's gates: no crema.toml, no persistence; `-e` never
    // persists; extract reads every AST and neither reads nor writes
    // the cache. The lookup condition is `diagnostic_filter.is_some()`
    // spelled from its inputs, since the filter is built after the walk.
    let ingest_cache_lookup =
        matches!(mode, RunMode::Check) && (eval_active || !cli_targets.is_empty());
    let ingest_cache_write = matches!(mode, RunMode::Check) && has_config && !eval_active;
    let mut ingest_cache_cold_reason: Option<String> = None;
    let mut ingest_cache_records: FxHashMap<String, crema::ingest_cache::Record> =
        FxHashMap::default();
    let mut ingest_cache_dirs: FxHashMap<String, DirListing> = FxHashMap::default();
    let t_ingest_cache_open = std::time::Instant::now();
    if ingest_cache_lookup {
        match crema::ingest_cache::read_file(&ingest_cache_path, &ingest_cache_key) {
            Ok(contents) => {
                ingest_cache_records.reserve(contents.records.len());
                for record in contents.records {
                    ingest_cache_records.insert(record.path.clone(), record);
                }
                ingest_cache_dirs.reserve(contents.dirs.len());
                for listing in contents.dirs {
                    ingest_cache_dirs.insert(listing.path.clone(), listing);
                }
            }
            Err(reason) => ingest_cache_cold_reason = Some(reason),
        }
    }
    let d_ingest_cache_open = t_ingest_cache_open.elapsed();
    let mut walked_dirs = WalkedDirs::default();

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
    // Script mode: the scope is exactly the one script (the mode's
    // whole point is that no `check` field exists to expand).
    let check_field = match &script {
        Some(path) => vec![path.clone()],
        None => resolved.check.clone().unwrap_or_default(),
    };
    let expanded = expand_targets(
        &check_field,
        ingest_cache_lookup.then_some(&ingest_cache_dirs),
        &mut walked_dirs,
    );
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
    let all_files: Vec<PathBuf> = canonicalize_expanded(&expanded)
        .into_iter()
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
        build_diagnostic_filter(
            &cli_targets,
            &files,
            &ignored_files,
            ingest_cache_lookup.then_some(&ingest_cache_dirs),
        )
    };

    let infusion_active = resolved.infusion.activesupport
        || resolved.infusion.activemodel
        || resolved.infusion.activerecord
        || resolved.infusion.sidekiq
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
        // Opening G interned every gem symbol; fold before ingest probes them.
        draft.compact_names();
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
            &resolver_anchor,
            no_bundler,
            script_active,
            &mut g_warnings,
        ) {
            Ok(gem_dirs) => {
                let core_dir = gem_dirs.rbs_gem_dir.join("core");

                if core_dir.is_dir() {
                    g_dirs.push(core_dir.clone());
                    if let Err(e) = draft.load_dir(&core_dir, true) {
                        warn_and_record(
                            &mut g_warnings,
                            format!("warning: failed to load core RBS: {}", e),
                        );
                    }
                }

                // Script mode: every `${rbs_gem_dir}/stdlib/<name>/0/`
                // dir, loaded by path. Deliberately not through the
                // name-based `library_loader::resolve` below — that
                // is gem-sig first, and an installed gem shipping its
                // own `sig/` for a stdlib name (nkf, bigdecimal, ...)
                // would then collide with the stdlib copy. The stdlib
                // set itself loads together cleanly (pinned by
                // `test_script_mode_installed_rbs_stdlib_set_loads_together`).
                // No manifest expansion is needed: everything is
                // loaded, so every dependency is already present.
                if script_active {
                    for dir in stdlib_sig_dirs(&gem_dirs.rbs_gem_dir) {
                        g_dirs.push(dir.clone());
                        if let Err(e) = draft.load_dir(&dir, true) {
                            warn_and_record(
                                &mut g_warnings,
                                format!("warning: failed to load stdlib {}: {}", dir.display(), e),
                            );
                        }
                    }
                    // Plus the rbs gem's own `sig/` (rbs is a Ruby 4.0
                    // bundled gem, so scripts driving `RBS::*` are in
                    // scope) and the `sig/` of every gemspec runtime
                    // dependency it references (`gem_dirs.rbs_runtime_deps`,
                    // resolved by the Ruby resolver — no gem name is
                    // hard-coded here). Deps first so rbs's references
                    // to them resolve within one environment build.
                    let rbs_sig = gem_dirs.rbs_gem_dir.join("sig");
                    let script_gem_dirs = gem_dirs
                        .rbs_runtime_deps
                        .iter()
                        .map(|(_, dir)| dir.join("sig"))
                        .chain(std::iter::once(rbs_sig));
                    for dir in script_gem_dirs {
                        g_dirs.push(dir.clone());
                        if let Err(e) = draft.load_dir(&dir, true) {
                            warn_and_record(
                                &mut g_warnings,
                                format!("warning: failed to load gem sig {}: {}", dir.display(), e),
                            );
                        }
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
                    if let Err(e) = draft.load_dir(dir, true) {
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
                    if let Err(e) = draft.load_dir(dir, true) {
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
    // single `.rbs` file. The files are kept as the user's own sig
    // files for `validate_decl_namespaces`, which must not walk
    // library decls.
    let mut user_sig_files: Vec<PathBuf> = Vec::new();
    for path in &resolved.sig_dirs {
        let result = if path.is_dir() {
            // User sig dirs are rbs `-I` dirs: `_` subdirs are read.
            // Same stop-at-first-failure shape as `load_dir`.
            crema::file_finder::each_file(path, false).and_then(|files| {
                for file in files {
                    draft.load_file(&file)?;
                    user_sig_files.push(file);
                }
                Ok(())
            })
        } else {
            draft
                .load_file(path)
                .map(|()| user_sig_files.push(path.clone()))
        };
        if let Err(e) = result {
            eprintln!("warning: failed to load sig {}: {}", path.display(), e);
        }
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
    //   2. Per-file check diagnostics are rendered by the pool
    //      thread that checked the file and written by main in walk
    //      order, each file as soon as it and every file before it
    //      are done (Layer 3 below) — the responsiveness AI agents
    //      rely on, without the output order depending on which
    //      file finishes first.
    // The macro layer boundary (env phase entirely before
    // check phase) is preserved by construction; within the
    // env phase the byte-order sort supersedes the historical
    // "inline parse → build validate" sub-layer emission order.
    let stdout = std::io::stdout();
    // `"file"` values (and the fingerprints / baseline rows built from
    // them) are displayed relative to the crema.toml dir even though
    // the paths are canonical/absolute internally — see
    // `output_display_base`.
    let display_base = output_display_base(&project_root);
    // ADR-0029 determined this filter's view is the whole
    // observable output (diagnostics *and* exit code), so files
    // outside it can skip Layer 3 entirely — kept here (cloned
    // before the emitter takes ownership) to drive that skip
    // and to keep a filtered run's baseline staleness report quiet.
    let check_filter = diagnostic_filter.clone();
    // Extract mode owns stdout for its own output; the
    // check pipeline's diagnostics still flow through the emitter
    // machinery (same code path, same error exits) but land in a sink.
    // `--update-baseline` prints nothing either: its rows go to the
    // baseline file at the normal end of the run, and an exit-2 path
    // (which only calls `finish()`) must neither print them nor touch
    // the file.
    let emitter_out: Box<dyn Write> = if matches!(mode, RunMode::Check) && !update_baseline {
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
    // `--update-baseline` collects the same rows `--tamp` streams; the
    // writer above is a sink, so only the explicit drain at the normal
    // end of the run gets them.
    .with_tamped(tamp || update_baseline)
    .with_baseline(baseline);

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
    // via the in-memory source cache below). The file path is
    // threaded through even in `-e` mode so inline mixins carry a
    // Location keyed on the `-e` pseudo-path.
    //
    // Pass 0 (read + prism parse) and the inline collector run on
    // the rayon pool, one read -> parse -> collect task per file
    // (ADR-0033, ADR-0035). Every check target's AST is needed for
    // Layer 3 regardless of warm/cold. The read is part of the task
    // rather than a serial loop on main: where file syscalls are slow
    // (endpoint security on open / stat) a serial read dominated the
    // ingest, and spreading it costs little elsewhere.
    // Each pool thread collects against its own `NameTable`:
    // `Symbol` / `TypeName` ids are content-addressed (ADR-0025),
    // so a worker's declarations are valid in main's table once the
    // worker's interners are merged in below, and the one
    // positional `Name` a worker needs (the file path) is interned
    // here, on main, before dispatch. Results are put back in walk
    // order by index, so the sequential insert loop below sees
    // files in the same order the old single-thread loop did.
    let inline_mode = resolved.inline;
    let eval_code: Option<Arc<[u8]>> = eval.map(|code| Arc::from(code.into_bytes()));
    let eval_path = PathBuf::from("-e");
    let n = files.len() + usize::from(eval_code.is_some());
    let path_at = |i: usize| -> &Path {
        if i < files.len() {
            files[i].as_path()
        } else {
            eval_path.as_path()
        }
    };
    // One slot per file: the task that reads it fills it, and the AST
    // borrows it for as long as the AST lives (through the check phase).
    // The `-e` code is never read, so its slot is filled up front.
    let slots: Vec<std::sync::OnceLock<Arc<[u8]>>> =
        (0..n).map(|_| std::sync::OnceLock::new()).collect();
    if let Some(code) = &eval_code {
        let _ = slots[files.len()].set(Arc::clone(code));
    }
    let units: Vec<crema::inline_parser::SourceFile<'_>> = (0..n)
        .map(|i| crema::inline_parser::SourceFile::intern(path_at(i), draft.names()))
        .collect();
    // One slot per pool thread, addressed by `current_thread_index` —
    // each lock is only ever taken by its own thread, so it is
    // uncontended.
    let pool_threads = rayon::current_num_threads();
    let mut workers: Vec<std::sync::Mutex<IngestWorker<'_>>> = (0..pool_threads)
        .map(|_| std::sync::Mutex::new(IngestWorker::default()))
        .collect();

    // Files are handed to the pool in batches of 32 rather than one
    // task per file (6,941 spawns -> ~220 on the gitlab workload).
    // Measured: -4% User time at 6 threads, same wall time. The bulk
    // of the extra CPU a parallel run shows over a single-threaded
    // one is not spawn overhead but the E-cores' slower parse counted
    // in CPU-seconds.
    const INGEST_BATCH: usize = 32;
    debug_assert_eq!(
        ingest_cache_lookup,
        matches!(mode, RunMode::Check) && check_filter.is_some(),
        "the lookup gate restates the diagnostic filter's condition"
    );
    // The previous run's listings served the walk; drop them before
    // the ingest phase (the write-back uses this run's listings).
    drop(ingest_cache_dirs);
    let ingest_cache = (ingest_cache_lookup || ingest_cache_write).then_some(IngestCacheInput {
        lookup: ingest_cache_lookup.then_some(&ingest_cache_records),
        filter: check_filter.as_ref(),
        write: ingest_cache_write,
        verify: snapshot_timing,
        scope_files: files.len(),
    });
    let input = IngestInput {
        slots: &slots,
        units: &units,
        inline_mode,
        infusion: infusion_active.then_some(InfusionCollectOptions {
            options: resolved.infusion,
            inflector: &inflector_owner,
        }),
        cache: ingest_cache,
    };
    rayon::in_place_scope(|s| {
        let workers = &workers;
        for start in (0..n).step_by(INGEST_BATCH) {
            let batch = start..(start + INGEST_BATCH).min(n);
            s.spawn(move |_| run_ingest_batch(workers, input, batch));
        }
    });
    // Tasks run in no particular order, so a read failure is reported
    // only once every task is done, and it is the first one in walk
    // order — the same file a serial read would have stopped at.
    let first_read_error = workers
        .iter_mut()
        .flat_map(|w| {
            std::mem::take(
                &mut w
                    .get_mut()
                    .expect("ingest worker slot poisoned")
                    .read_errors,
            )
        })
        .min_by_key(|(i, _)| *i);
    if let Some((i, e)) = first_read_error {
        eprintln!("error: cannot read {}: {}", path_at(i).display(), e);
        process::exit(2);
    }

    // A file served from the ingest cache has no bytes in memory: the
    // emitter and the validator fall back to reading the file if a
    // diagnostic ever needs its text (it never does for a file outside
    // the filter — see `build_diagnostic_filter`).
    let sources: Vec<(PathBuf, Arc<[u8]>)> = (0..n)
        .filter_map(|i| {
            slots[i]
                .get()
                .map(|bytes| (path_at(i).to_path_buf(), Arc::clone(bytes)))
        })
        .collect();
    // Nothing is emitted before this point, so registering the
    // sources after the read is equivalent to registering them
    // before it.
    for (file, source) in &sources {
        emitter.register_source(file.clone(), Arc::clone(source));
    }
    let mut ingested: Vec<Option<IngestedFile<'_>>> = (0..n).map(|_| None).collect();
    let mut cache_stats = IngestCacheStats::default();
    for worker in workers {
        let IngestWorker {
            names,
            results,
            cache_stats: worker_stats,
            ..
        } = worker.into_inner().expect("ingest worker slot poisoned");
        draft.names().merge(names);
        cache_stats.add(&worker_stats);
        for (i, result) in results {
            ingested[i] = Some(result);
        }
    }
    // The workers' tables were merged in above; fold before the passes
    // that follow read them.
    draft.compact_names();

    if snapshot_timing && matches!(mode, RunMode::Check) {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let note = match (&ingest_cache_cold_reason, ingest_cache_lookup) {
            (Some(reason), _) => format!(" cold ({reason})"),
            (None, false) => " (no lookup: every file is a target)".to_string(),
            (None, true) => String::new(),
        };
        let uncacheable = if cache_stats.uncacheable > 0 {
            format!(" uncacheable {}", cache_stats.uncacheable)
        } else {
            String::new()
        };
        // `open` is wall time on main (read + record index); `stat`,
        // `decode` and `verify` are summed over the workers. `verify`
        // (reading every hit file back to check its content hash) only
        // runs under this env, so it inflates the `ingest:` line below
        // by about the read cost the cache saves.
        // `dirs`: directories whose listing came from the cache vs.
        // read again (both are written back).
        eprintln!(
            "ingest-cache: hit {} / miss {} (open {:.1}ms, stat {:.1}ms, decode {:.1}ms, verify {:.1}ms) dirs reused {} / walked {}{}{}",
            cache_stats.hits,
            cache_stats.misses,
            ms(d_ingest_cache_open),
            ms(cache_stats.stat_time),
            ms(cache_stats.decode_time),
            ms(cache_stats.verify_time),
            walked_dirs.reused,
            walked_dirs.listings.len() - walked_dirs.reused,
            uncacheable,
            note
        );
    }
    // Write back the scope files' records (see `ingest_cache_write_back`).
    // The cache is a lookup by path, so a file left out only misses next
    // run. Write failures are silent, as for the G snapshot: stderr must
    // not depend on the cache.
    if ingest_cache_write {
        let outcomes = ingested
            .iter_mut()
            .enumerate()
            .take(files.len())
            .map(|(i, file)| {
                let file = file.as_mut().expect("every file is ingested exactly once");
                (
                    path_at(i),
                    std::mem::replace(&mut file.cache, IngestCacheOutcome::Off),
                )
            });
        let records = ingest_cache_write_back(outcomes, &mut ingest_cache_records);
        let contents = crema::ingest_cache::Contents {
            dirs: std::mem::take(&mut walked_dirs.listings),
            records,
        };
        let bytes = crema::ingest_cache::encode_file(&ingest_cache_key, &contents);
        let _ = crema::snapshot::write::write_atomic_bytes(&ingest_cache_path, &bytes);
    }
    drop(ingest_cache_records);

    let mut parsed_sources = Vec::with_capacity(n);
    let mut collected = Vec::with_capacity(n);
    // Indexed by walk index (a `None` per syntax-error file), which is
    // the `source_index` the workers stamped on their concern records.
    let mut infusion_collected: Vec<Option<crema::infusion_collector::CollectedSource<'_>>> =
        (0..n).map(|_| None).collect();
    for (i, ingested) in ingested.into_iter().enumerate() {
        let IngestedFile {
            parsed, payload, ..
        } = ingested.expect("every file is ingested exactly once");
        match payload {
            crema::ingest_cache::IngestPayload::SyntaxError(diag) => env_diags.push(diag),
            crema::ingest_cache::IngestPayload::Parsed {
                decls,
                diags,
                infusion,
            } => {
                // A cache hit has neither bytes nor AST: its source is
                // read on demand if an insert-time diagnostic needs it.
                let (source, parse_result) = match parsed {
                    Some((source, ast)) => {
                        (crema::source_ref::SourceRef::Bytes(source), Some(ast.0))
                    }
                    None => (
                        crema::source_ref::SourceRef::Lazy {
                            slot: &slots[i],
                            path: path_at(i),
                        },
                        None,
                    ),
                };
                parsed_sources.push(ParsedSource {
                    file: path_at(i),
                    source,
                    parse_result,
                });
                collected.push((decls, diags));
                infusion_collected[i] = infusion;
            }
        }
    }

    if snapshot_timing {
        eprintln!(
            "ingest: read+parse+collect {}us",
            t_ingest.elapsed().as_micros()
        );
    }
    let t_inline = std::time::Instant::now();
    // Insert the collected declarations into the draft, single-threaded
    // and in walk order, for every file — the eval pseudo-file
    // included (ADR-0029 §5 rework — eval declarations live in the
    // in-memory environment; only snapshot persistence is eval-gated).
    // Sig mode (`inline = false`) ran no collector (see `ingest_file`),
    // so every file's lists are empty there.
    for (parsed, (decls, mut diags)) in parsed_sources.iter().zip(collected) {
        for decl in &decls {
            draft.insert_ruby_decl(decl, parsed.source, Some(parsed.file), &mut diags);
        }
        env_diags.extend(diags);
    }

    if infusion_active {
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
            // The schema dumps parse independently of the Ruby sources
            // (their `self.table_name =` overrides were collected per file
            // on the ingest workers); load_collected emits the schema
            // declarations once the post-concern-expansion enum set is
            // known.
            let (schema, schema_diags) =
                crema::infusion_collector::activerecord::prepare_schema(&project_root);
            env_diags.extend(schema_diags);
            Some(schema)
        } else {
            None
        };
        env_diags.extend(crema::infusion_collector::load_collected(
            infusion_collected,
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
            "ingest: insert+infusion {}us",
            t_inline.elapsed().as_micros()
        );
    }
    let d_ingest = t_ingest.elapsed();
    // AST release: a file's AST is dead once inline collection is
    // over unless Layer 3 type-checks it. A file outside the
    // diagnostic filter provably won't be (the check loop skips it
    // outright), so its AST is released here, before env
    // construction: the C-side prism heap holds these nodes at ~10x
    // the source size, and a single-file check (ADR-0029) would
    // otherwise carry every other scope file's tree through the
    // footprint peak. Extract reads every AST, so it is exempt.
    if matches!(mode, RunMode::Check)
        && let Some(filter) = &check_filter
    {
        let mut dead_asts = Vec::new();
        for parsed in &mut parsed_sources {
            if !filter.contains(parsed.file)
                && let Some(ast) = parsed.parse_result.take()
            {
                dead_asts.push(ast);
            }
        }
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
        // ADR-0036 Decision 4-2: the diagnostics-only env walks (the
        // defbuild dup scans, every validator sub-check) start from the
        // filter's files when there is one — their diagnostics for any
        // other file would be dropped by the emitter's filter anyway,
        // so under a single-file check they were ~2/5 of the wall spent
        // before the check phase for nothing. The filter's paths are
        // canonical, as are the names the ingest interned decls under
        // (`SourceFile::intern`), and `-e` is both the filter entry and
        // the eval unit's name, so the lookup is exact: a filter file
        // the index does not know simply contributes no seed.
        let scan_scope = match &check_filter {
            Some(filter) => {
                let names = frozen.names();
                frozen.scan_scope_for_files(
                    filter
                        .iter()
                        .filter_map(|path| names.lookup(&path.to_string_lossy())),
                )
            }
            None => crema::environment::ScanScope::Whole,
        };
        let env = DefinitionBuilder::from_environment_scoped(Arc::clone(&frozen), &scan_scope);
        let d_defbuild_cold = t_defbuild.elapsed();

        let t_validate = std::time::Instant::now();
        let (_, validate_diags) = validator::full_validate_scoped(&env, &source_cache, &scan_scope);
        env_diags.extend(validate_diags);
        // The user's own decls: sig files plus every scope `.rb`
        // (inline decls; `-e` included), as the names their decls were
        // interned under. Library sigs are not walked. Under a filter,
        // only the filter's files: a `.rbs` or non-target `.rb` decl's
        // namespace diagnostic could never reach the output.
        let names = env.names();
        let user_files: Vec<crema::name::Name> = match &check_filter {
            Some(filter) => filter
                .iter()
                .filter_map(|path| names.lookup(&path.to_string_lossy()))
                .collect(),
            None => user_sig_files
                .iter()
                .filter_map(|path| names.lookup(&path.to_string_lossy()))
                .chain(units.iter().map(|unit| unit.name))
                .collect(),
        };
        env_diags.extend(validator::validate_decl_namespaces(&env, user_files));
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
                &mut parsed_sources,
                &project_root,
                eval_active,
                snapshot_timing,
            );
            return;
        }
        RunMode::Check => {}
    }

    let t_check = std::time::Instant::now();
    // Sub-phase accumulators for the check-timing line: time spent in
    // the `did_you_mean` join and in JSONL emission, so their cost can
    // be told apart from the type-check proper.
    let mut d_join = std::time::Duration::ZERO;
    let mut d_emit = std::time::Duration::ZERO;

    // Layer 3 (type check): one job per file, run on the pool
    // (ADR-0034 Decision 4). Main commits the results in walk order as
    // the finished prefix grows, so each file's diagnostics still go
    // out as soon as it and every file before it are done, and the
    // JSONL stream stays bit-identical to a serial check whatever the
    // thread count.
    let mut jobs = Vec::new();
    for parsed in &mut parsed_sources {
        // Files outside the diagnostic filter never reach the
        // output or the exit code (ADR-0029 §3/§5), so their
        // per-file check is unobservable work. Env construction
        // above still saw every scope file — only Layer 3 is
        // elided.
        if let Some(filter) = &check_filter
            && !filter.contains(parsed.file)
        {
            continue;
        }
        jobs.push(CheckJob {
            file: parsed.file,
            source: parsed.source.bytes(),
            ast: SendParseResult(
                parsed
                    .parse_result
                    .take()
                    .expect("an in-filter file keeps its AST until its check job"),
            ),
        });
    }
    let (renderer, committer) = emitter.split();
    let context = CheckContext {
        env: &env,
        renderer,
        did_you_mean: resolved.did_you_mean,
    };
    run_in_walk_order(
        jobs,
        |job| run_check_job(context, job),
        |outcome| {
            let t_commit = std::time::Instant::now();
            for rendered in outcome.rendered {
                committer.commit(rendered);
            }
            d_emit += outcome.d_render + t_commit.elapsed();
            d_join += outcome.d_join;
        },
    );

    if snapshot_timing {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        eprintln!(
            "check-timing: total {:.1}ms setup {:.1}ms ingest {:.1}ms build {:.1}ms defbuild {:.1}ms validate {:.1}ms check {:.1}ms (join {:.1}ms emit {:.1}ms)",
            ms(t_run.elapsed()),
            ms(d_setup),
            ms(d_ingest),
            ms(d_build),
            ms(d_defbuild),
            ms(d_validate),
            ms(t_check.elapsed()),
            ms(d_join),
            ms(d_emit)
        );
        if let Some(backend) = env.env().g_backend() {
            eprintln!(
                "g-snapshot: decoded {} entries lazily",
                backend.decode_count()
            );
        }
    }

    let had_any_error = emitter.had_any_error();
    if update_baseline {
        let path = baseline_path
            .as_ref()
            .expect("--update-baseline was gated on a resolved baseline path");
        if let Err(err) = write_baseline(path, emitter.take_tamp_lines()) {
            eprintln!("error: cannot write baseline {}: {}", path.display(), err);
            process::exit(2);
        }
    } else if emitter.stale_baseline_count() > 0 && check_filter.is_none() {
        // A filtered run never sees the rest of the scope, so its
        // unmatched rows say nothing about staleness.
        eprintln!(
            "baseline: {} entries no longer reported; run `crema check --update-baseline` to drop them",
            emitter.stale_baseline_count()
        );
    }
    emitter.finish();
    drop(emitter);

    // `--update-baseline` records the diagnostics instead of failing
    // on them.
    if had_any_error && !update_baseline {
        process::exit(1);
    }
}

/// Read a baseline file into remaining-absorb counts keyed by the
/// canonical tamp row. Each line must be exactly a
/// `{file, code, fingerprint}` object; anything else names its line
/// number so a hand edit or a merge conflict marker is caught, not
/// silently treated as "no such diagnostic".
fn load_baseline(path: &Path) -> Result<HashMap<String, usize>, String> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Row {
        file: String,
        code: String,
        fingerprint: String,
    }
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "error: baseline file not found: {}\n  run `crema check --update-baseline` to create it",
                path.display()
            ));
        }
        Err(e) => {
            return Err(format!(
                "error: cannot read baseline {}: {}",
                path.display(),
                e
            ));
        }
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (idx, line) in content.lines().enumerate() {
        let row: Row = serde_json::from_str(line).map_err(|e| {
            format!(
                "error: malformed baseline row at {}:{}: {}",
                path.display(),
                idx + 1,
                e
            )
        })?;
        *counts
            .entry(crema::diagnostic::render_tamp_line(
                &row.file,
                &row.code,
                &row.fingerprint,
            ))
            .or_insert(0) += 1;
    }
    Ok(counts)
}

/// Write the sorted tamp rows as the baseline file. An empty run
/// still writes the (empty) file so `baseline = true` never trips its
/// own missing-file error afterwards.
fn write_baseline(path: &Path, lines: Vec<String>) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut content = String::new();
    for line in lines {
        content.push_str(&line);
        content.push('\n');
    }
    std::fs::write(path, content)
}

/// Stack size of the pool threads that main spawns: the usual size of
/// the main thread's own stack. The checker recurses on expression
/// depth, and any pool thread may pick up any file, so with a smaller
/// stack (Rust's default for spawned threads is 2MB) an input that
/// checks fine on one thread would overflow on several. Fixed on
/// purpose: it is a correctness bound, not a tuning knob.
const POOL_STACK_BYTES: usize = 8 * 1024 * 1024;

/// One check target parsed by Pass 0 (shared by the check phase and the
/// extract check pass).
struct ParsedSource<'a> {
    file: &'a Path,
    source: crema::source_ref::SourceRef<'a>,
    /// `None` once the AST has been released ahead of the check phase
    /// (a filtered-out file — see the release pre-pass in `run_check`)
    /// or moved into the file's check or extract job. Every reader
    /// `take()`s it with an `expect`, so a read-after-release is a loud
    /// panic rather than a silently skipped file.
    parse_result: Option<ruby_prism::ParseResult<'a>>,
}

/// A prism `ParseResult` that may be moved off the thread that parsed it.
///
/// `ParseResult` holds `NonNull` pointers into the C parser and node
/// tree, which makes it `!Send` by default. Everything those pointers
/// reach is heap-owned by the result itself (freed in its `Drop`); prism
/// keeps no thread-local or global parser state. A result changes
/// threads twice, each time by move: the ingest worker that parsed it
/// hands it to main (infusion, the release pre-pass), and main hands it
/// to the check job that reads it and then frees it. There is one owner
/// at any time, so the moves are sound. `Sync` is deliberately not
/// claimed: nothing shares an AST across threads.
struct SendParseResult<'a>(ruby_prism::ParseResult<'a>);

// SAFETY: see the type doc — heap-only state, single owner at a time,
// moved (never shared) across the worker → main → worker boundaries.
unsafe impl Send for SendParseResult<'_> {}

/// What one parallel-ingest worker produces for one check target.
struct IngestedFile<'a> {
    /// The bytes and AST of a file parsed this run. `None` for a file
    /// served from the ingest cache (never read) and for a file prism
    /// rejected (no AST to keep).
    parsed: Option<(&'a [u8], SendParseResult<'a>)>,
    /// The collect output: the inline collector's declarations and
    /// diagnostics (insert-time diagnostics are appended on main) and
    /// the rails infusion collect half, walked on the same worker right
    /// after the inline collector so the AST is walked twice there
    /// rather than once there and once on main. A prism error is one
    /// `Ruby::SyntaxError` and the file takes no further part in the
    /// run (see `ingest_file`).
    payload: crema::ingest_cache::IngestPayload<'a>,
    cache: IngestCacheOutcome,
}

/// The records to write back for the scope files `outcomes` lists in
/// walk order: a hit's stored record (taken out of `stored`), a miss's
/// freshly encoded one. A file with neither (uncacheable, or a stat
/// that failed under the read) is left out and misses next run; only
/// it pays for that. Stored records of files no longer in scope are
/// never taken, so a deleted file drops out of the cache.
fn ingest_cache_write_back<'p>(
    outcomes: impl IntoIterator<Item = (&'p Path, IngestCacheOutcome)>,
    stored: &mut FxHashMap<String, crema::ingest_cache::Record>,
) -> Vec<crema::ingest_cache::Record> {
    outcomes
        .into_iter()
        .filter_map(|(path, outcome)| match outcome {
            IngestCacheOutcome::Hit => stored.remove(path.to_string_lossy().as_ref()),
            IngestCacheOutcome::Miss(record) => record,
            IngestCacheOutcome::Off => None,
        })
        .collect()
}

/// How the ingest cache saw one file.
enum IngestCacheOutcome {
    /// Served from the cache; its record is reused for the write-back.
    Hit,
    /// Parsed this run. `Some` carries the record to write back; `None`
    /// when the run is not writing, the file could not be stat'ed, or
    /// its payload could not be encoded (uncacheable).
    Miss(Option<crema::ingest_cache::Record>),
    /// Not a cache subject (the `-e` pseudo-file, or no cache active).
    Off,
}

/// The ingest cache's inputs to every ingest task.
#[derive(Clone, Copy)]
struct IngestCacheInput<'a> {
    /// Records read from the cache file, by canonical path. `None` when
    /// this run looks nothing up (no diagnostic filter).
    lookup: Option<&'a FxHashMap<String, crema::ingest_cache::Record>>,
    /// Files in here are check targets and are always parsed.
    filter: Option<&'a HashSet<PathBuf>>,
    /// Encode a record for every parsed file, for the write-back.
    write: bool,
    /// `CREMA_DEBUG_SNAPSHOT_TIMING=1`: read hit files back and report
    /// a content-hash mismatch on stderr.
    verify: bool,
    /// Walk indices at or past this are not scope files (`-e`).
    scope_files: usize,
}

/// Hit / miss counts and the time the lookup itself took, summed over
/// the workers that did it.
#[derive(Default)]
struct IngestCacheStats {
    hits: usize,
    misses: usize,
    /// Parsed files whose payload could not be encoded (a `Name` other
    /// than the file's own): never cached, re-parsed every run. Zero
    /// unless a collector starts stamping foreign names; the debug line
    /// is the only place a release build reports it.
    uncacheable: usize,
    stat_time: std::time::Duration,
    decode_time: std::time::Duration,
    verify_time: std::time::Duration,
}

impl IngestCacheStats {
    fn add(&mut self, other: &IngestCacheStats) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.uncacheable += other.uncacheable;
        self.stat_time += other.stat_time;
        self.decode_time += other.decode_time;
        self.verify_time += other.verify_time;
    }
}

/// What the infusion collect half needs per file, shared read-only by
/// every ingest worker. `None` when no infusion is active.
#[derive(Clone, Copy)]
struct InfusionCollectOptions<'a> {
    options: crema::config::InfusionOptions,
    inflector: &'a crema::infusion_collector::Inflector,
}

/// What every ingest task reads, indexed by walk index. `slots[i]` is
/// filled by the task that reads file `i` (or up front for `-e`).
#[derive(Clone, Copy)]
struct IngestInput<'a, 'c> {
    slots: &'a [std::sync::OnceLock<Arc<[u8]>>],
    units: &'a [crema::inline_parser::SourceFile<'a>],
    inline_mode: bool,
    infusion: Option<InfusionCollectOptions<'a>>,
    /// `None` when neither reading nor writing the ingest cache. Its
    /// own lifetime: the records are borrowed only for the pool's
    /// duration, while `'a` (the sources and ASTs) outlives the write-back
    /// that consumes them.
    cache: Option<IngestCacheInput<'c>>,
}

/// One pool thread's share of the ingest: the `NameTable` it collects
/// against, the files it finished and the files it could not read,
/// each tagged with their walk index.
struct IngestWorker<'a> {
    names: crema::name::NameTable,
    results: Vec<(usize, IngestedFile<'a>)>,
    read_errors: Vec<(usize, std::io::Error)>,
    cache_stats: IngestCacheStats,
}

impl Default for IngestWorker<'_> {
    fn default() -> Self {
        IngestWorker {
            names: crema::name::NameTable::new(),
            results: Vec::new(),
            read_errors: Vec::new(),
            cache_stats: IngestCacheStats::default(),
        }
    }
}

/// Read and `ingest_file` the files `batch` (walk indices) on the
/// calling pool thread's own [`IngestWorker`] slot. A file that cannot
/// be read is recorded rather than reported: which task fails first
/// depends on scheduling, the report must not.
fn run_ingest_batch<'a>(
    workers: &[std::sync::Mutex<IngestWorker<'a>>],
    input: IngestInput<'a, '_>,
    batch: std::ops::Range<usize>,
) {
    let idx = rayon::current_thread_index().expect("ingest tasks run on pool threads");
    let mut worker = workers[idx].lock().expect("ingest worker slot poisoned");
    let IngestInput {
        slots,
        units,
        inline_mode,
        infusion,
        cache,
    } = input;
    for i in batch {
        let unit = units[i];
        let cache = cache.filter(|c| i < c.scope_files);
        // The stat is taken before the read so a write that lands
        // between the two leaves a record the next run misses on,
        // rather than a stale parse filed under the new mtime.
        let stat = cache.and_then(|_| {
            let t = std::time::Instant::now();
            let stat = std::fs::metadata(unit.path)
                .ok()
                .map(|m| crema::ingest_cache::FileStat::of(&m));
            worker.cache_stats.stat_time += t.elapsed();
            stat
        });
        if let Some(c) = cache
            && let Some(lookup) = c.lookup
            && !c.filter.is_some_and(|f| f.contains(unit.path))
            && let Some(record) = lookup.get(unit.path.to_string_lossy().as_ref())
            && stat == Some(record.stat)
        {
            let t = std::time::Instant::now();
            let decoded = crema::ingest_cache::decode_payload(record, &worker.names, unit.name);
            worker.cache_stats.decode_time += t.elapsed();
            if let Some(mut payload) = decoded {
                if let crema::ingest_cache::IngestPayload::Parsed {
                    infusion: Some(collected),
                    ..
                } = &mut payload
                {
                    collected.rebind(
                        i,
                        crema::source_ref::SourceRef::Lazy {
                            slot: &slots[i],
                            path: unit.path,
                        },
                        Some(unit.path),
                    );
                }
                if c.verify {
                    let t = std::time::Instant::now();
                    if let Ok(bytes) = std::fs::read(unit.path)
                        && xxhash_rust::xxh3::xxh3_64(&bytes) != record.content_hash
                    {
                        eprintln!(
                            "ingest-cache: verify mismatch for {} (same mtime and size, different content)",
                            unit.path.display()
                        );
                    }
                    worker.cache_stats.verify_time += t.elapsed();
                }
                worker.cache_stats.hits += 1;
                worker.results.push((
                    i,
                    IngestedFile {
                        parsed: None,
                        payload,
                        cache: IngestCacheOutcome::Hit,
                    },
                ));
                continue;
            }
        }
        let source: &'a [u8] = match slots[i].get() {
            Some(bytes) => bytes,
            None => match std::fs::read(unit.path) {
                Ok(bytes) => slots[i].get_or_init(|| Arc::from(bytes)),
                Err(e) => {
                    worker.read_errors.push((i, e));
                    continue;
                }
            },
        };
        let (parsed, payload) = ingest_file(i, source, unit, &worker.names, inline_mode, infusion);
        let cache = match cache {
            Some(c) => {
                worker.cache_stats.misses += 1;
                let record = if c.write {
                    stat.and_then(|stat| {
                        let encoded =
                            crema::ingest_cache::encode_payload(&worker.names, unit.name, &payload);
                        if encoded.is_none() {
                            worker.cache_stats.uncacheable += 1;
                        }
                        let encoded = encoded?;
                        Some(crema::ingest_cache::Record {
                            path: unit.path.to_string_lossy().into_owned(),
                            stat,
                            content_hash: xxhash_rust::xxh3::xxh3_64(source),
                            self_name: encoded.self_name,
                            strings: encoded.strings,
                            payload: encoded.payload,
                        })
                    })
                } else {
                    None
                };
                IngestCacheOutcome::Miss(record)
            }
            None => IngestCacheOutcome::Off,
        };
        worker.results.push((
            i,
            IngestedFile {
                parsed,
                payload,
                cache,
            },
        ));
    }
}

/// Parse one check target and run the inline collector (inline mode
/// only) — and, when an infusion is active, the infusion collector — on
/// it, against the worker's own `names` (ADR-0033). Pure with respect
/// to the draft.
/// `index` is the file's walk index; it is the `source_index` the
/// infusion insert half resolves concern files through.
fn ingest_file<'a>(
    index: usize,
    source: &'a [u8],
    file: crema::inline_parser::SourceFile<'a>,
    names: &crema::name::NameTable,
    inline_mode: bool,
    infusion: Option<InfusionCollectOptions<'a>>,
) -> (
    Option<(&'a [u8], SendParseResult<'a>)>,
    crema::ingest_cache::IngestPayload<'a>,
) {
    let parse_result = ruby_prism::parse(source);

    // Prism returns an AST even when the source is syntactically
    // invalid. Walking that recovery AST is unsafe: receiver-less
    // CallNodes like `{a!: }` (which Ruby itself rejects with
    // SyntaxError) get type-checked as real calls and surface as
    // NoMethod diagnostics on input ruby itself refuses to run.
    // Mirror Steep's policy ("Ruby rejects → type checker stays
    // silent") by emitting one `Ruby::SyntaxError` (anchored at the
    // first prism error) and skipping inline collection and type
    // checking for this file. Build / validator still see other
    // files' declarations.
    if let Some(first_err) = parse_result.errors().next() {
        let offset = first_err.location().start_offset();
        let message = first_err.message().to_string();
        let location = crema::diagnostic::Diagnostic::location_for_byte_range(
            file.path.to_path_buf(),
            source,
            offset,
            first_err.location().end_offset(),
        );
        return (
            None,
            crema::ingest_cache::IngestPayload::SyntaxError(crema::diagnostic::Diagnostic::at(
                location,
                DiagnosticKind::SyntaxError { message },
            )),
        );
    }

    // Sig mode reads declarations from `sig/` only (ADR-0027: the
    // inline class), so the collector — and every diagnostic it would
    // push — does not run. The checker still reads standard annotations
    // (`expr #: T`, `foo #: [T]`) from comments on its own.
    let (decls, diags) = if inline_mode {
        crema::inline_parser::collect_inline_declarations(source, &parse_result, Some(file), names)
    } else {
        (Vec::new(), Vec::new())
    };
    let infusion = infusion.map(|infusion| {
        crema::infusion_collector::collect_source(
            names,
            index,
            Some(file.name),
            source,
            Some(file.path),
            &parse_result,
            infusion.options,
            infusion.inflector,
            inline_mode,
        )
    });
    (
        Some((source, SendParseResult(parse_result))),
        crema::ingest_cache::IngestPayload::Parsed {
            decls,
            diags,
            infusion,
        },
    )
}

/// One file's share of the check phase. Built on main in walk order,
/// run on whichever pool thread picks it up.
struct CheckJob<'a> {
    file: &'a Path,
    source: &'a [u8],
    /// The job owns the AST and frees it as soon as the file's check
    /// is over.
    ast: SendParseResult<'a>,
}

/// One `crema extract` file: what the check job needs, plus the AST it
/// owns and frees (same ownership as `CheckJob`).
struct ExtractJob<'a> {
    file: &'a Path,
    source: &'a [u8],
    ast: SendParseResult<'a>,
}

/// What an extract job hands back to main: the file's records, ready
/// to drop into its `FileRecord`. Nothing here touched shared state.
struct ExtractOutcome<'a> {
    file: &'a Path,
    method_call: Vec<crema::extract::MethodCallRecord>,
    implements: Vec<crema::extract::ImplementsRecord>,
    constant: Vec<crema::extract::ConstantRecord>,
    consulted: Vec<String>,
}

/// Runs the site-collecting check on one file with consultation
/// recording on. The log is per file and lives inside the job, so
/// nothing is shared across threads but `env`.
fn run_extract_job<'a>(env: &DefinitionBuilder, job: ExtractJob<'a>) -> ExtractOutcome<'a> {
    let ExtractJob { file, source, ast } = job;
    let log = crema::definition_builder::ConsultationLog::new();
    let (method_call, implements, constant, signature_types) =
        crema::type_checker::check_source_extract(
            env,
            file.to_path_buf(),
            source,
            &ast.0,
            Some(&log),
        );
    // Freed here, on the thread that read it.
    drop(ast);
    let consulted = crema::extract::consulted_symbols(
        &log.into_entries(),
        &signature_types,
        &constant,
        env.env().names(),
    );
    ExtractOutcome {
        file,
        method_call,
        implements,
        constant,
        consulted,
    }
}

/// What every check job reads.
#[derive(Clone, Copy)]
struct CheckContext<'a> {
    env: &'a DefinitionBuilder,
    renderer: &'a DiagnosticRenderer,
    did_you_mean: bool,
}

/// What a check job hands back to main. Nothing here has touched the
/// run's ordered state (baseline counts, exit flag, the output): main
/// applies it in walk order.
struct CheckOutcome {
    rendered: Vec<RenderedDiagnostic>,
    d_join: std::time::Duration,
    d_render: std::time::Duration,
}

/// Check one file and render its diagnostics.
fn run_check_job(context: CheckContext<'_>, job: CheckJob<'_>) -> CheckOutcome {
    let CheckJob { file, source, ast } = job;
    let mut diagnostics = check_source(context.env, file.to_path_buf(), source, &ast.0);
    // Freed here, on the thread that checked it, rather than kept
    // until every file is done.
    drop(ast);
    // ADR-0032 Decision 5a: check only records the fact that a
    // constant failed to resolve (name + candidate scope);
    // `did_you_mean` is a computed column joined against the
    // env in scope here, right before rendering.
    let t_join = std::time::Instant::now();
    if context.did_you_mean {
        join_did_you_mean(context.env, &mut diagnostics);
    }
    let d_join = t_join.elapsed();
    let t_render = std::time::Instant::now();
    let rendered = diagnostics
        .iter()
        .filter_map(|diag| context.renderer.render(diag))
        .collect();
    CheckOutcome {
        rendered,
        d_join,
        d_render: t_render.elapsed(),
    }
}

/// Run `work` on every job on the rayon pool and hand the results to
/// `commit` on the calling thread, one at a time, in job order.
///
/// Jobs are queued first-in first-out, and while the next result in
/// order is not in yet the caller runs queued jobs itself. In a
/// one-thread pool the caller is the only thread (ADR-0035), so the
/// run alternates "work i, commit i" like a serial loop; in a larger
/// pool the finished prefix is committed while later jobs still run.
///
/// A panic in `work` or `commit` propagates out of this call, with the
/// outcome a serial loop would leave behind: when job `i` panics, every
/// result before `i` is committed and nothing from `i` on is; jobs
/// after `i` that have not started are skipped. When `commit` panics,
/// every job that has not started is skipped.
fn run_in_walk_order<J: Send, R: Send>(
    jobs: Vec<J>,
    work: impl Fn(J) -> R + Sync,
    mut commit: impl FnMut(R),
) {
    /// Lowers `first_panic` to `index` when a panic unwinds through
    /// the scope holding it.
    struct RecordPanic<'a> {
        first_panic: &'a AtomicUsize,
        index: usize,
    }
    impl Drop for RecordPanic<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.first_panic.fetch_min(self.index, Ordering::Relaxed);
            }
        }
    }

    let total = jobs.len();
    // The lowest index of a job that panicked; jobs after it are not
    // worth running.
    let first_panic = AtomicUsize::new(usize::MAX);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, R)>();
    rayon::in_place_scope_fifo(|scope| {
        let (work, first_panic) = (&work, &first_panic);
        // A panic in `commit` below: no job is worth running any more.
        let _record = RecordPanic {
            first_panic,
            index: 0,
        };
        for (index, job) in jobs.into_iter().enumerate() {
            let tx = tx.clone();
            scope.spawn_fifo(move |_| {
                if index > first_panic.load(Ordering::Relaxed) {
                    return;
                }
                let _record = RecordPanic { first_panic, index };
                // The receiver outlives the scope, so the send cannot fail.
                let _ = tx.send((index, work(job)));
            });
        }
        // From here on only the jobs hold a sender: when the last one is
        // gone, a `recv` for a result that never came fails instead of
        // blocking.
        drop(tx);
        let mut arrived: Vec<Option<R>> = (0..total).map(|_| None).collect();
        let mut next = 0;
        while next < total {
            for (index, result) in rx.try_iter() {
                arrived[index] = Some(result);
            }
            while let Some(result) = arrived.get_mut(next).and_then(Option::take) {
                commit(result);
                next += 1;
            }
            if next == total {
                break;
            }
            // Waiting alone would stall a one-thread pool: run a queued
            // job here. Only when none is left to run (the remaining ones
            // are on other threads) block until the next result arrives.
            if matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                continue;
            }
            match rx.recv() {
                Ok((index, result)) => arrived[index] = Some(result),
                // Every job is over and a result is missing: a job
                // panicked. The scope re-raises that panic on the way out.
                Err(_) => break,
            }
        }
    });
}

/// The per-file phase of `crema extract` (invoked from `run_check`
/// after env construction): run the site-collecting check on every
/// scope file with consultation recording on, join in the per-file
/// definitions from the environment, and print the whole document to
/// stdout as one JSON value. Nothing is persisted.
///
/// In `-e` mode (`eval_active`) the document is narrowed to the `-e`
/// pseudo-file.
///
/// Extract is opt-in and allowed to be slower than `crema check`;
/// recording costs exist only on this path.
fn run_extract(
    env: &DefinitionBuilder,
    sources: &[(PathBuf, Arc<[u8]>)],
    parsed_sources: &mut [ParsedSource<'_>],
    project_root: &Path,
    eval_active: bool,
    timing: bool,
) {
    let t_extract = std::time::Instant::now();
    // Same crema.toml-dir-relative display rule as the diagnostic
    // emitter; `root` records the base so the document stays
    // self-describing wherever it is consumed.
    let display_base = output_display_base(project_root);
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
        files.insert(
            display(file),
            crema::extract::FileRecord {
                content_hash: format!("{:016x}", crema::extract::content_hash(bytes)),
                ..Default::default()
            },
        );
    }

    // Check pass: one job per parsed file on the pool, the same shape
    // as `crema check`'s Layer 3 (ADR-0034 Decision 4). Each job owns
    // its AST and frees it once its records are collected; the
    // definitions pass below reads source bytes only. Main folds the
    // outcomes into `files` in walk order — the map is keyed by path,
    // so the document would come out the same in any order, but walk
    // order keeps the fold's cost profile identical to a serial loop.
    let mut jobs = Vec::new();
    for parsed in parsed_sources.iter_mut() {
        if eval_active && parsed.file != eval_file {
            continue;
        }
        jobs.push(ExtractJob {
            file: parsed.file,
            source: parsed.source.bytes(),
            ast: SendParseResult(
                parsed
                    .parse_result
                    .take()
                    .expect("extract keeps every AST until its check job"),
            ),
        });
    }
    run_in_walk_order(
        jobs,
        |job| run_extract_job(env, job),
        |outcome| {
            // Every parsed file is a scope file, so its entry was made
            // above; a miss would mean `parsed_sources` outgrew
            // `sources`, and a hash-less entry is the loud form of it.
            let record = files.entry(display(outcome.file)).or_default();
            record.method_call = outcome.method_call;
            record.implements = outcome.implements;
            record.consulted = outcome.consulted;
            record.constant = outcome.constant;
        },
    );
    let d_check = t_extract.elapsed();

    // Definitions pass: every file that contributed declarations
    // (`path_index` = A-layer only, so gem/core files never appear).
    // Check targets reuse their in-memory bytes; sig files are read
    // back from disk for hash computation.
    // Path -> in-memory bytes, built once: a linear scan per index
    // entry would be quadratic in the scope size (12,040 files on
    // gitlab cost ~9s that way).
    let source_by_path: FxHashMap<&Path, &[u8]> = sources
        .iter()
        .map(|(file, bytes)| (file.as_path(), &bytes[..]))
        .collect();
    let names = env.env().names();
    for file_name in env.env().path_index_files() {
        let path_string = names.resolve(file_name);
        let path = Path::new(&path_string);
        if eval_active && path != eval_file {
            continue;
        }
        let source: std::borrow::Cow<'_, [u8]> = match source_by_path.get(path) {
            Some(bytes) => std::borrow::Cow::Borrowed(bytes),
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
                content_hash: format!("{:016x}", crema::extract::content_hash(&source)),
                ..Default::default()
            });
        entry.definitions = definitions;
    }
    let d_definitions = t_extract.elapsed() - d_check;

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
    let t_serialize = std::time::Instant::now();
    let bytes = serde_json::to_vec(&output).expect("extract document serialize is infallible");
    let d_serialize = t_serialize.elapsed();
    // stdout carries the document itself and nothing else (no summary
    // line), in every mode: nothing is persisted, and a zero-config
    // `-e` run never creates `.crema/`. A consumer closing the pipe
    // early (`| head`) is not an error.
    let mut out = std::io::stdout().lock();
    if let Err(err) = out.write_all(&bytes).and_then(|()| out.write_all(b"\n"))
        && err.kind() != std::io::ErrorKind::BrokenPipe
    {
        eprintln!("error: failed to write extract document: {err}");
        process::exit(2);
    }
    if timing {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        eprintln!(
            "extract-timing: total {:.1}ms check {:.1}ms definitions {:.1}ms serialize {:.1}ms write {:.1}ms ({} bytes)",
            ms(t_extract.elapsed()),
            ms(d_check),
            ms(d_definitions),
            ms(d_serialize),
            ms(t_extract.elapsed() - d_check - d_definitions - d_serialize),
            bytes.len(),
        );
    }
}

fn main() {
    let cli = Cli::parse();
    let cli_config = cli.config.clone();
    let no_bundler = cli.no_bundler;

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
                    run_snapshot_dump(&project_root, cli_config.as_deref(), no_bundler);
                }
            },
        },
        Commands::Check {
            files,
            eval,
            script,
            sig_dirs,
            add_sig_dirs,
            inline,
            collection,
            no_g_snapshot,
            refresh_g_snapshot,
            tamp,
            update_baseline,
            no_baseline,
            threads,
        } => run_check(
            cli_config.as_deref(),
            no_bundler,
            CheckInvocation {
                files,
                eval,
                script,
                sig_dirs,
                add_sig_dirs,
                inline,
                collection,
                no_g_snapshot,
                refresh_g_snapshot,
                tamp,
                update_baseline,
                no_baseline,
                threads,
            },
            RunMode::Check,
        ),
        Commands::Extract { eval, threads } => run_check(
            cli_config.as_deref(),
            no_bundler,
            CheckInvocation {
                files: Vec::new(),
                eval,
                script: None,
                sig_dirs: Vec::new(),
                add_sig_dirs: Vec::new(),
                inline: None,
                collection: None,
                no_g_snapshot: false,
                refresh_g_snapshot: false,
                tamp: false,
                update_baseline: false,
                no_baseline: false,
                threads,
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
