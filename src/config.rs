use std::collections::HashMap;
use std::path::{Path, PathBuf};

use globset::Glob;
use serde::Deserialize;

use crate::diagnostic::{DiagnosticKind, Severity};


const CONFIG_FILE: &str = "crema.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Jsonl,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub sig: Vec<PathBuf>,
    /// Type-check target (ADR-0029): directories and `.rb` files whose
    /// union forms the environment scope (= the A snapshot's input
    /// set). Mirrors `sig`'s shape and path-resolution rules exactly
    /// (relative entries absolutized against the `crema.toml` directory
    /// during walk-up discovery, not absolutized under `--config`).
    /// `None` means the field is absent from the file (ADR-0029 §4: a
    /// hard error at the CLI layer); `Some(vec![])` means an explicit
    /// `check = []`, a valid empty scope. Consumed by `main.rs`'s
    /// `Commands::Check` handler (ADR-0029 slice S3).
    #[serde(default)]
    pub check: Option<Vec<PathBuf>>,
    /// Glob patterns (matched against paths relative to the `crema.toml`
    /// directory) subtracted from `check`'s walked `.rb` set:
    /// `walk(check) - glob(ignore)`. A file excluded this way supplies no
    /// decls (inline or otherwise) to the environment — this is a set
    /// subtraction, not a diagnostic-suppression filter.
    #[serde(default)]
    pub ignore: Vec<String>,
    /// Glob patterns subtracted from the final `sig` file set, the same
    /// way `ignore` subtracts from `check`: `walk(sig) - glob(sig_ignore)`.
    /// The subtraction step itself does not special-case a sig entry's
    /// origin (this field, `--sig`, or `--add-sig` — all three are
    /// merged into one list before this runs); a dropped entry is
    /// warned about regardless of where it came from. The glob's
    /// matching *basis* still inherits `--sig`/`--add-sig`'s existing
    /// cwd-relative-vs-`sig`'s-project-root-relative asymmetry — see
    /// `main.rs`'s `is_ignored`.
    #[serde(default)]
    pub sig_ignore: Vec<String>,
    /// Libraries to load by name (`libraries = ["json", "prism"]`). Each
    /// name is resolved gem-sig first (`${gem_dir}/sig/`) and then
    /// `${rbs_gem_dir}/stdlib/<name>/0/` as fallback, with `manifest.yaml`
    /// dependencies pulled in transitively from whichever side hit.
    /// See `crate::library_loader`.
    #[serde(default)]
    pub libraries: Vec<String>,
    pub inline: Option<bool>,
    /// Opt into `did_you_mean` suggestions on `UnknownConstant`
    /// (default false). The suggestion join is a spell-check over every
    /// constant in scope per diagnostic and dominated the warm check on
    /// large trees, so it is off unless a project asks for it
    /// (ADR-0032, amended 2026-09-06).
    pub did_you_mean: Option<bool>,
    /// Opt into the incremental check cache (ADR-0032 Decision 1,
    /// default false). When true, `crema check` persists per-file
    /// consulted-key sets, content hashes, the per-name fingerprint
    /// table, and structured diagnostics under `.crema/cache/`, and
    /// skips re-checking files no changed name can reach. Cache presence
    /// never changes the JSONL output (bit-identical to a full check) —
    /// it is purely an internal cost optimization. Config-file only by
    /// design: no CLI flag (Decision 1 names crema.toml as the switch).
    ///
    /// `"verify"` (ADR-0032 Decision 6) additionally shadow-rechecks
    /// every replay-classified file and reports any divergence from the
    /// stored entry on stderr with exit 1 — the `-Z
    /// incremental-verify-ich` sibling for watching real edit patterns.
    pub incremental: Option<IncrementalSetting>,
    /// Raw `[diagnostic]` table. `preset = "..."` plus per-code
    /// overrides (`"Ruby::NoMethod" = "warning"` etc.). Validated and
    /// resolved into `DiagnosticConfig` by `resolve_with_cli`; see
    /// `validate_diagnostic_table` for the rules applied to unknown
    /// keys, unknown presets, and invalid values.
    #[serde(default)]
    pub diagnostic: Option<HashMap<String, toml::Value>>,
    /// Path to an `rbs_collection.yaml` (config) file. The lockfile is
    /// derived from this path via `rbs_collection::load_from_config`
    /// (rbs/steep convention: `.lock` inserted before the final
    /// extension). CLI flag `--collection CONFIG` takes precedence; in
    /// its absence, this field opts the project into a specific config
    /// path. Lowered to `CollectionMode` by `resolve_with_cli`.
    /// Relative paths are absolutized against the directory containing
    /// `crema.toml` during walk-up discovery; explicit `--config <PATH>`
    /// loading does not absolutize (matches the `sig` paradigm).
    #[serde(default)]
    pub collection_config: Option<PathBuf>,
    /// Raw `[infusion]` table. `[infusion.rails] enabled = true` enables the
    /// Rails DSL preset. Unknown names are rejected by serde so typos surface
    /// as ConfigError::Parse.
    #[serde(default)]
    pub infusion: Option<InfusionTable>,
}

/// `[infusion]` table shape. Public config uses framework-level presets;
/// resolved options keep the lower-level rule-family switches internal.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InfusionTable {
    #[serde(default)]
    pub rails: Option<RailsInfusionTable>,
    #[serde(default)]
    pub config: Option<ConfigInfusionTable>,
    #[serde(default)]
    pub active_decorator: Option<ActiveDecoratorInfusionTable>,
    #[serde(default)]
    pub paranoia: Option<ParanoiaInfusionTable>,
}

/// `[infusion.rails]` table shape. Held as a sub-table (not a scalar bool)
/// so future Rails-specific options (e.g. `[infusion.rails.inflections]`)
/// can extend this shape without another breaking rename.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RailsInfusionTable {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub inflections: Option<InflectionsTable>,
}

/// `[infusion.rails.inflections]` table shape. Locale-keyed sub-tables;
/// only `en` is supported, others are rejected by `deny_unknown_fields`
/// so future locale opens (`ja`, `fr`, ...) surface as a clear parse
/// error rather than a silent no-op.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InflectionsTable {
    #[serde(default)]
    pub en: Option<InflectionsLocaleTable>,
}

/// `[infusion.rails.inflections.en]` table shape. Mirrors the
/// `inflect.irregular` / `inflect.acronym` directives in a Rails app's
/// `config/initializers/inflections.rb`. `plural` / `singular` regex
/// pairs and `uncountable` are intentionally not supported: an
/// uncountable word is expressible as `inflect.irregular 'x', 'x'`, so
/// the extra surface was dropped, and regex pairs are deferred until
/// real-world `inflections.rb` usage shows they are needed.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InflectionsLocaleTable {
    #[serde(default)]
    pub irregular: Vec<[String; 2]>,
    #[serde(default)]
    pub acronym: Vec<String>,
}

/// `[infusion.config]` table shape. Parameterizes the `Settings` (or
/// user-named) constant synthesized from YAML config files: which name
/// to mount under (`const_name`), which top-level keys to drop
/// (`except_keys`), and which YAML files to merge in order (`files`).
/// See `crate::infusion_collector::config` for the lowering.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConfigInfusionTable {
    #[serde(default = "default_config_const_name")]
    pub const_name: String,
    #[serde(default)]
    pub except_keys: Vec<String>,
    #[serde(default)]
    pub files: Vec<PathBuf>,
}

fn default_config_const_name() -> String {
    "Settings".to_string()
}

/// `[infusion.active_decorator]` table shape. Parameterizes the
/// [active_decorator](https://github.com/amatsuda/active_decorator) gem's
/// decorator-module self-type synthesis: `decorator_suffix` mirrors the
/// gem's own `ActiveDecorator.config.decorator_suffix`
/// (`lib/active_decorator/config.rb`, default `'Decorator'`).
///
/// The search root is fixed at `app/decorators` and is deliberately not
/// configurable: the gem's generator always writes there, and the fixed
/// root is what lets the pass ignore every `*Decorator` module coming
/// from gem code. See `crate::infusion_collector::active_decorator`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActiveDecoratorInfusionTable {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_decorator_suffix")]
    pub decorator_suffix: String,
}

fn default_decorator_suffix() -> String {
    "Decorator".to_string()
}

/// `[infusion.paranoia]` table shape. The
/// [paranoia](https://github.com/rubysherpas/paranoia) gem's
/// `acts_as_paranoid` options (`column:`, `sentinel_value:`, ...) do not
/// change any RBS signature in the gem's collection types, so the table
/// carries only the opt-in switch. Requires `[infusion.rails]
/// enabled = true`: the synthesized mixins reference
/// `<Model>::ActiveRecord_Relation`, which only the rails preset's AR
/// synthesis declares (`resolve_with_cli_collecting_warnings` rejects
/// the standalone combination as `ConfigError::InvalidParanoia`).
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParanoiaInfusionTable {
    #[serde(default)]
    pub enabled: bool,
}

/// Resolved view of `[infusion]`. Plain `Copy` bool fields so the
/// infusion collector can take it by value without lifetime entanglements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InfusionOptions {
    pub activesupport: bool,
    pub activemodel: bool,
    pub activerecord: bool,
    /// `[infusion.paranoia] enabled = true`. Never true without the
    /// rails preset — the standalone combination is rejected at config
    /// resolution (`ConfigError::InvalidParanoia`).
    pub paranoia: bool,
}

impl InfusionOptions {
    /// True when the resolved config's `[infusion.rails] enabled = true`
    /// preset is in effect. `ResolvedConfig::from_file` fans that single
    /// `enabled` boolean out to all three individual flags identically
    /// (see the `rails_infusion` block in `from_file`), so their
    /// simultaneous truth uniquely identifies the rails-on state. Used
    /// by the zeitwerk namespace-synthesis pass, which must not fire
    /// under standalone `activesupport`/`activemodel`/`activerecord`
    /// gating (compact-declaration namespace synthesis is a Rails-only
    /// concession — plain Ruby treats `class Staff::X` as a NameError).
    pub fn rails_enabled(&self) -> bool {
        self.activesupport && self.activemodel && self.activerecord
    }
}

/// Resolved view of the collection-loading decision. `Auto` is walk-up
/// discovery; `Path(p)` names a config file whose sibling lockfile is
/// loaded explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionMode {
    /// Walk-up discovery — `rbs_collection.discover_lockfile` searches
    /// for `rbs_collection.yaml` and reads its sibling
    /// `rbs_collection.lock.yaml`, stopping at the `.git` boundary.
    /// This is intentionally narrower than the rbs CLI, which walks all
    /// the way to filesystem root looking for the config file.
    Auto,
    /// Use the given config path directly, bypassing walk-up. The
    /// lockfile is derived from the config path via
    /// `rbs_collection::load_from_config`. NotFound on the derived
    /// lockfile is a hard error (unlike `Auto`'s silent skip).
    ConfigPath(PathBuf),
}

/// Built-in preset that supplies the base severity for every
/// diagnostic kind. Overrides in `[diagnostic]` win over the preset
/// on a per-code basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Use `DiagnosticKind::default_severity()` — the Steep-parity
    /// table baked into the crate.
    Default,
    /// Promote every diagnostic to `Severity::Error`.
    AllError,
    /// Demote every diagnostic to `Severity::Ignore`. Combine with
    /// per-code overrides to build "only the codes I name" configs.
    AllIgnore,
}

impl Preset {
    pub fn severity_for(self, kind: &DiagnosticKind) -> Severity {
        self.severity_for_code(kind.code())
    }

    pub fn severity_for_code(self, code: &str) -> Severity {
        // Dev-only diagnostic kinds (currently only `Crema::NotImplementedYet`)
        // are off under every preset — even `all_error` must not promote
        // them. The user opts in via explicit `[diagnostic]` override,
        // which is applied by `DiagnosticConfig::severity_for_code`
        // *before* this preset fallback.
        if DiagnosticKind::is_dev_only_code(code) {
            return Severity::Ignore;
        }
        match self {
            Preset::Default => DiagnosticKind::default_severity_for_code(code),
            Preset::AllError => DiagnosticKind::all_error_severity_for_code(code),
            Preset::AllIgnore => DiagnosticKind::all_ignore_severity_for_code(code),
        }
    }

    pub fn from_config_str(s: &str) -> Option<Preset> {
        match s {
            "default" => Some(Preset::Default),
            "all_error" => Some(Preset::AllError),
            "all_ignore" => Some(Preset::AllIgnore),
            _ => None,
        }
    }
}

/// Resolved view of the `[diagnostic]` table. The emitter consults
/// `severity_for(kind)` to decide the final tier (which may be
/// `Severity::Ignore`, meaning "drop before writing").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticConfig {
    pub preset: Preset,
    /// Keyed by `DiagnosticKind::code()` strings (e.g. `"Ruby::NoMethod"`).
    pub overrides: HashMap<String, Severity>,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        DiagnosticConfig {
            preset: Preset::Default,
            overrides: HashMap::new(),
        }
    }
}

impl DiagnosticConfig {
    pub fn severity_for(&self, kind: &DiagnosticKind) -> Severity {
        self.severity_for_code(kind.code())
    }

    /// Code-string-keyed form of `severity_for`, for callers that hold a
    /// `code()` string rather than a `DiagnosticKind` (e.g. `crema
    /// diagnostic list` iterating `DiagnosticKind::ALL_CODES`).
    pub fn severity_for_code(&self, code: &str) -> Severity {
        self.overrides
            .get(code)
            .copied()
            .unwrap_or_else(|| self.preset.severity_for_code(code))
    }
}

/// Raw `incremental` value as written in crema.toml: `true`/`false` or
/// the mode string `"verify"`. Untagged so the boolean form stays
/// backward compatible; the string is validated in `resolve` (unknown
/// strings are a `ConfigError`, not a silent fallback).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum IncrementalSetting {
    Enabled(bool),
    Named(String),
}

/// Resolved incremental mode. `Verify` implies everything `On` does,
/// plus the shadow recheck of replay-classified files (ADR-0032
/// Decision 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementalMode {
    Off,
    On,
    Verify,
}

impl IncrementalMode {
    /// Whether the incremental cache machinery runs at all.
    pub fn active(self) -> bool {
        !matches!(self, IncrementalMode::Off)
    }

    /// Whether replay-classified files are shadow-rechecked.
    pub fn verify(self) -> bool {
        matches!(self, IncrementalMode::Verify)
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    /// A `[diagnostic]` entry's value or `preset` name failed
    /// validation (unknown preset, unknown severity string, non-string
    /// value). Unknown diagnostic codes are *not* surfaced through this
    /// variant — they emit a stderr warning instead.
    InvalidDiagnostic(String),
    /// `[infusion.rails.inflections.en]` carried an unsupported pair —
    /// empty string in an `irregular` entry, or a `[singular, plural]`
    /// pair whose first letters disagree case-insensitively (the
    /// `add_irregular` implementation only supports the matched-first-
    /// letter branch that all Rails defaults sit on).
    InvalidInflections(String),
    /// `[infusion.active_decorator]` carried a value the provider cannot
    /// act on — an empty `decorator_suffix`.
    InvalidActiveDecorator(String),
    /// `[infusion.paranoia] enabled = true` without `[infusion.rails]
    /// enabled = true`. The synthesized mixins reference
    /// `<Model>::ActiveRecord_Relation`, which only AR synthesis
    /// declares, so the standalone combination would be a half-working
    /// silent no-op, and config must never silently no-op.
    InvalidParanoia(String),
    /// An `ignore` or `sig_ignore` entry is not a syntactically valid
    /// glob pattern.
    InvalidGlob(String),
    /// `incremental` carried a string other than `"verify"`.
    InvalidIncremental(String),
    /// A path explicitly named by the user (`--config <PATH>`) does not
    /// exist. Distinguished from `Io` because `discover()` silently
    /// treats NotFound as "no config" (Ok(None)), whereas an explicit
    /// `--config` path must be a hard error.
    NotFound {
        path: PathBuf,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "error: cannot read crema.toml: {}", e),
            ConfigError::Parse(e) => write!(f, "error: invalid crema.toml: {}", e),
            ConfigError::InvalidDiagnostic(msg) => {
                write!(f, "error: invalid crema.toml [diagnostic]: {}", msg)
            }
            ConfigError::InvalidInflections(msg) => write!(
                f,
                "error: invalid crema.toml [infusion.rails.inflections.en]: {}",
                msg
            ),
            ConfigError::InvalidActiveDecorator(msg) => write!(
                f,
                "error: invalid crema.toml [infusion.active_decorator]: {}",
                msg
            ),
            ConfigError::InvalidParanoia(msg) => {
                write!(f, "error: invalid crema.toml [infusion.paranoia]: {}", msg)
            }
            ConfigError::InvalidGlob(msg) => write!(f, "error: invalid crema.toml glob: {}", msg),
            ConfigError::InvalidIncremental(value) => write!(
                f,
                "error: invalid crema.toml `incremental` value {:?}: expected true, false, or \"verify\"",
                value
            ),
            ConfigError::NotFound { path } => {
                write!(f, "error: config file not found: {}", path.display())
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedConfig {
    pub format: Format,
    pub sig_dirs: Vec<PathBuf>,
    /// ADR-0029 check target, propagated verbatim from `Config::check`
    /// (no CLI merge here — the CLI argument is a diagnostic filter,
    /// not an environment-scope input; see `Config::check`'s doc for
    /// the `None`/`Some(vec![])` distinction). `main.rs`'s
    /// `Commands::Check` handler treats `None` as a hard error
    /// (ADR-0029 §4) before ever constructing a `ResolvedConfig`, so by
    /// the time callers hold one in the `crema check` path, this field
    /// is `Some`.
    pub check: Option<Vec<PathBuf>>,
    /// Validated `ignore` glob patterns, verbatim from `Config::ignore`.
    /// `main.rs`'s `Commands::Check` handler compiles these into a
    /// `globset::GlobSet` and subtracts matches from the walked `check`
    /// file list.
    pub ignore: Vec<String>,
    /// Validated `sig_ignore` glob patterns, verbatim from
    /// `Config::sig_ignore`. Subtracted from the final `sig` file set
    /// regardless of whether an entry came from this config's `sig`,
    /// `--sig`, or `--add-sig`.
    pub sig_ignore: Vec<String>,
    pub libraries: Vec<String>,
    pub inline: bool,
    /// Resolved `did_you_mean` (`Config::did_you_mean`, default false).
    /// Gates `join_did_you_mean` in `main.rs`'s check handler.
    pub did_you_mean: bool,
    /// Resolved `incremental` mode (`Config::incremental`, default
    /// false). Read by `main.rs`'s check handler to gate every cache
    /// read/write — the default path must not touch `.crema/cache/`'s
    /// incremental artifact at all.
    pub incremental: IncrementalMode,
    pub diagnostic: DiagnosticConfig,
    pub collection: CollectionMode,
    pub infusion: InfusionOptions,
    pub config_infusion: Option<ConfigInfusionTable>,
    /// `[infusion.active_decorator]`, lowered to `None` when the table is
    /// absent *or* carries `enabled = false`. Downstream code gates on
    /// `is_some()` alone and never re-reads `enabled`, so the opt-in
    /// decision lives in exactly one place.
    pub active_decorator_infusion: Option<ActiveDecoratorInfusionTable>,
    /// User-supplied `inflect.irregular` / `inflect.acronym` from
    /// `[infusion.rails.inflections.en]`. Held even when
    /// `infusion.rails.enabled = false` (parse-accept, apply-skip
    /// gated downstream by whether the AR pipeline runs).
    pub inflections: Option<InflectionsLocaleTable>,
}

impl Config {
    /// Discover a `crema.toml` by walking up from the current working
    /// directory until either the file is found or a `.git` entry marks
    /// the repository root. Returns `Ok(None)` when no `crema.toml`
    /// exists along that path. Relative `sig` entries in the discovered
    /// config are absolutized against the directory the config was
    /// found in. See `discover_walking_up` for the full algorithm.
    pub fn discover() -> Result<Option<Config>, ConfigError> {
        let cwd = std::env::current_dir().map_err(ConfigError::Io)?;
        Ok(Self::discover_walking_up(&cwd)?.map(|(cfg, _dir)| cfg))
    }

    /// Walk up from `start_dir` looking for `crema.toml`. At each step:
    /// 1. If `<current>/crema.toml` exists, parse it, absolutize its
    ///    relative `sig` entries against `<current>`, and return the
    ///    config along with `<current>` (the directory it was found in).
    /// 2. Otherwise, if `<current>/.git` exists (file or directory),
    ///    treat `<current>` as the repository root and stop — return
    ///    `Ok(None)` rather than ascending past it.
    /// 3. Otherwise ascend to the parent and repeat. If no parent
    ///    exists (filesystem root), return `Ok(None)`.
    ///
    /// `start_dir` is canonicalized once so `.` resolves to an absolute
    /// path; intermediate ancestors are not re-canonicalized, preserving
    /// symlinked layouts such as git worktrees.
    ///
    /// nearest-wins: the first `crema.toml` along the walk is returned;
    /// configs higher in the tree are never merged.
    pub fn discover_walking_up(start_dir: &Path) -> Result<Option<(Config, PathBuf)>, ConfigError> {
        let start = start_dir.canonicalize().map_err(ConfigError::Io)?;
        let mut current = start.as_path();
        loop {
            let candidate = current.join(CONFIG_FILE);
            match std::fs::read_to_string(&candidate) {
                Ok(content) => {
                    let mut cfg = Self::parse(&content)?;
                    let dir = current.to_path_buf();
                    for path in cfg.sig.iter_mut() {
                        if path.is_relative() {
                            *path = dir.join(&path);
                        }
                    }
                    if let Some(check) = cfg.check.as_mut() {
                        for path in check.iter_mut() {
                            if path.is_relative() {
                                *path = dir.join(&path);
                            }
                        }
                    }
                    if let Some(p) = cfg.collection_config.as_mut()
                        && p.is_relative()
                    {
                        *p = dir.join(&p);
                    }
                    if let Some(infusion) = cfg.infusion.as_mut()
                        && let Some(config) = infusion.config.as_mut()
                    {
                        for path in config.files.iter_mut() {
                            if path.is_relative() {
                                *path = dir.join(&path);
                            }
                        }
                    }
                    return Ok(Some((cfg, dir)));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(ConfigError::Io(e)),
            }

            // Stop at the repository root (`.git` may be a directory or
            // a file — submodules use a file pointing to the gitdir).
            // `try_exists` propagates permission errors as Err rather
            // than silently treating them as "not present", which would
            // let walk-up sail past a real `.git` and pick up a remote
            // ancestor's crema.toml.
            if current.join(".git").try_exists().map_err(ConfigError::Io)? {
                return Ok(None);
            }

            match current.parent() {
                Some(parent) => current = parent,
                None => return Ok(None),
            }
        }
    }

    /// Single-directory discovery that does *not* walk up. Used as a
    /// test helper and by callers that want to inspect exactly one
    /// directory. Production `crema check` paths go through
    /// `discover` → `discover_walking_up`.
    pub fn discover_in(dir: &Path) -> Result<Option<Config>, ConfigError> {
        let path = dir.join(CONFIG_FILE);
        match std::fs::read_to_string(&path) {
            Ok(content) => Ok(Some(Self::parse(&content)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ConfigError::Io(e)),
        }
    }

    /// Load a config file from an explicit path (`--config <PATH>`).
    /// Unlike `discover()`, NotFound is a hard error here because the
    /// user explicitly named this file.
    pub fn load_from_file(path: &Path) -> Result<Config, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(content) => Self::parse(&content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ConfigError::NotFound {
                path: path.to_path_buf(),
            }),
            Err(e) => Err(ConfigError::Io(e)),
        }
    }

    pub fn parse(s: &str) -> Result<Config, ConfigError> {
        toml::from_str(s).map_err(ConfigError::Parse)
    }

    /// Merge CLI overrides onto a file-loaded config, injecting defaults for
    /// any value still unspecified. CLI `Some` wins; CLI `None` falls through to
    /// the file value, then to a built-in default.
    ///
    /// `sig` semantics follow the config design goal that CLI overrides
    /// config (MUST):
    ///
    /// - `--sig` (`cli_sig`) — when non-empty, **replaces** the project's
    ///   `sig` list wholesale. The file-side entries are dropped.
    /// - `--add-sig` (`cli_add_sig`) — appends to the project's `sig` list.
    ///   This is the escape hatch for the "config plus extras" workflow the
    ///   old concatenating `--sig` served, without violating Goal 4.
    /// - Both together — `--sig` wins and `--add-sig` is ignored with a
    ///   warning surfaced through the return value. Rationale: a
    ///   simultaneous "replace" and "append" is contradictory; picking one
    ///   silently would be more surprising than saying so.
    pub fn resolve_with_cli(
        file: Option<Config>,
        cli_sig: &[PathBuf],
        cli_add_sig: &[PathBuf],
        cli_inline: Option<bool>,
        cli_collection: CollectionCliOverride,
    ) -> Result<ResolvedConfig, ConfigError> {
        let (cfg, _warnings) = Self::resolve_with_cli_collecting_warnings(
            file,
            cli_sig,
            cli_add_sig,
            cli_inline,
            cli_collection,
        )?;
        Ok(cfg)
    }

    /// Like `resolve_with_cli`, but also returns the list of
    /// `[diagnostic]` warnings (currently: unknown code keys, and the
    /// `--sig` + `--add-sig` co-usage warning). The caller decides how
    /// to surface them (stderr in main, captured vec in tests).
    pub fn resolve_with_cli_collecting_warnings(
        file: Option<Config>,
        cli_sig: &[PathBuf],
        cli_add_sig: &[PathBuf],
        cli_inline: Option<bool>,
        cli_collection: CollectionCliOverride,
    ) -> Result<(ResolvedConfig, Vec<String>), ConfigError> {
        let file = file.unwrap_or_default();
        validate_glob_patterns("ignore", &file.ignore)?;
        validate_glob_patterns("sig_ignore", &file.sig_ignore)?;
        let (sig_dirs, sig_warnings) = resolve_sig_dirs(file.sig, cli_sig, cli_add_sig);
        let (diagnostic, mut warnings) = match file.diagnostic {
            Some(table) => validate_diagnostic_table(table)?,
            None => (DiagnosticConfig::default(), Vec::new()),
        };
        warnings.extend(sig_warnings);
        let collection = resolve_collection_mode(cli_collection, file.collection_config);
        let rails_ref = file.infusion.as_ref().and_then(|d| d.rails.as_ref());
        let rails_infusion = rails_ref.is_some_and(|r| r.enabled);
        let inflections = rails_ref
            .and_then(|r| r.inflections.as_ref())
            .and_then(|t| t.en.as_ref())
            .cloned();
        if let Some(en) = inflections.as_ref() {
            validate_inflections_locale_table(en)?;
        }
        let active_decorator_ref = file
            .infusion
            .as_ref()
            .and_then(|d| d.active_decorator.as_ref());
        if let Some(t) = active_decorator_ref {
            validate_active_decorator_table(t)?;
        }
        let active_decorator_infusion = active_decorator_ref.filter(|t| t.enabled).cloned();
        let paranoia_infusion = file
            .infusion
            .as_ref()
            .and_then(|d| d.paranoia.as_ref())
            .is_some_and(|t| t.enabled);
        if paranoia_infusion && !rails_infusion {
            return Err(ConfigError::InvalidParanoia(
                "requires [infusion.rails] enabled = true (the synthesized mixins \
                 reference <Model>::ActiveRecord_Relation)"
                    .to_string(),
            ));
        }
        let config_infusion = file.infusion.and_then(|d| d.config);
        let infusion = InfusionOptions {
            activesupport: rails_infusion,
            activemodel: rails_infusion,
            activerecord: rails_infusion,
            paranoia: paranoia_infusion,
        };
        Ok((
            ResolvedConfig {
                format: Format::Jsonl,
                sig_dirs,
                check: file.check,
                ignore: file.ignore,
                sig_ignore: file.sig_ignore,
                libraries: file.libraries,
                inline: cli_inline.or(file.inline).unwrap_or(true),
                did_you_mean: file.did_you_mean.unwrap_or(false),
                incremental: match &file.incremental {
                    None | Some(IncrementalSetting::Enabled(false)) => IncrementalMode::Off,
                    Some(IncrementalSetting::Enabled(true)) => IncrementalMode::On,
                    Some(IncrementalSetting::Named(s)) if s == "verify" => IncrementalMode::Verify,
                    Some(IncrementalSetting::Named(s)) => {
                        return Err(ConfigError::InvalidIncremental(s.clone()));
                    }
                },
                diagnostic,
                collection,
                infusion,
                config_infusion,
                active_decorator_infusion,
                inflections,
            },
            warnings,
        ))
    }
}

/// CLI-side input for `collection_config` resolution.
/// `ConfigPath(PathBuf)` is `--collection CONFIG`; `Absent` means the
/// flag was not set (fall back to toml or walk-up). The unset variant
/// is named `Absent` rather than `None` to avoid visual collision with
/// `Option::None` at call sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionCliOverride {
    Absent,
    ConfigPath(PathBuf),
}

/// Merge CLI override and `collection_config` toml field into a
/// `CollectionMode`. Precedence: CLI > toml > walk-up.
///
/// There is no "disable collection loading" variant. The legacy
/// `--no-collection` flag and `[collection].disable` toml field were
/// removed as crema-specific concepts; opting out is now expressed by
/// the absence of a config file (no `--collection`, no
/// `collection_config` key, and no `rbs_collection.yaml` along walk-up).
fn resolve_collection_mode(
    cli: CollectionCliOverride,
    toml_config: Option<PathBuf>,
) -> CollectionMode {
    match cli {
        CollectionCliOverride::ConfigPath(p) => CollectionMode::ConfigPath(p),
        CollectionCliOverride::Absent => match toml_config {
            Some(p) => CollectionMode::ConfigPath(p),
            None => CollectionMode::Auto,
        },
    }
}

/// Merge the file-side `sig` list with `--sig` (override) and
/// `--add-sig` (append). Precedence follows the config design goal that
/// CLI overrides config: CLI override wins over file, and the append flag is a
/// separate opt-in on top of the file entries.
///
/// - `cli_sig` non-empty → `sig_dirs = cli_sig`. File entries are
///   dropped, matching CLI-precedence semantics. `cli_add_sig` is
///   ignored and a warning is emitted so the user notices the flag
///   contradiction (asking for both replace and append is meaningless).
/// - `cli_sig` empty → `sig_dirs = file_sig ++ cli_add_sig`. When the
///   append flag is also empty, this reduces to `file_sig` verbatim
///   (the "no CLI sig options" case).
fn resolve_sig_dirs(
    file_sig: Vec<PathBuf>,
    cli_sig: &[PathBuf],
    cli_add_sig: &[PathBuf],
) -> (Vec<PathBuf>, Vec<String>) {
    if !cli_sig.is_empty() {
        let mut warnings = Vec::new();
        if !cli_add_sig.is_empty() {
            warnings.push(
                "warning: --add-sig ignored because --sig replaces the sig list wholesale"
                    .to_string(),
            );
        }
        return (cli_sig.to_vec(), warnings);
    }
    let mut dirs = file_sig;
    dirs.extend(cli_add_sig.iter().cloned());
    (dirs, Vec::new())
}

/// Reject `ignore`/`sig_ignore` entries that are not syntactically valid
/// globs, so a typo'd pattern fails fast at config-resolve time rather
/// than silently never matching anything at check time. An empty entry
/// is rejected outright rather than left as a silent no-op: `is_ignored`
/// treats a metachar-free pattern as a path-component prefix, and an
/// empty path is a prefix of every path (`Path::starts_with("")` is
/// `true` for any non-empty path), which would silently ignore the
/// entire project rather than nothing.
fn validate_glob_patterns(label: &str, patterns: &[String]) -> Result<(), ConfigError> {
    for pattern in patterns {
        if pattern.is_empty() {
            return Err(ConfigError::InvalidGlob(format!(
                "{} entry cannot be an empty string",
                label
            )));
        }
        Glob::new(pattern).map_err(|e| {
            ConfigError::InvalidGlob(format!("{} entry {:?}: {}", label, pattern, e))
        })?;
    }
    Ok(())
}

/// Reject inflections inputs that the inflector cannot represent.
/// Empty strings would `panic!` inside `add_irregular`'s `chars().next()`
/// pull, so we catch them here before they reach the runtime inflector.
fn validate_inflections_locale_table(en: &InflectionsLocaleTable) -> Result<(), ConfigError> {
    for word in &en.acronym {
        if word.is_empty() {
            return Err(ConfigError::InvalidInflections(
                "acronym entries cannot be empty strings".to_string(),
            ));
        }
    }
    for pair in &en.irregular {
        let [singular, plural] = pair;
        if singular.is_empty() || plural.is_empty() {
            return Err(ConfigError::InvalidInflections(format!(
                "irregular pair {:?} cannot contain an empty string",
                pair
            )));
        }
    }
    Ok(())
}

/// Reject an empty `decorator_suffix`. Every module name ends with the
/// empty string, so the gem's own `end_with?` test would match all of
/// them and `delete_suffix("")` would hand each module itself as its
/// model — a degenerate reading nobody wants. Validated whether or not
/// the provider is enabled, like the inflections table: a value crema
/// cannot act on is a config error, not a silent no-op (config design
/// goal: no silent no-ops).
fn validate_active_decorator_table(
    table: &ActiveDecoratorInfusionTable,
) -> Result<(), ConfigError> {
    if table.decorator_suffix.is_empty() {
        return Err(ConfigError::InvalidActiveDecorator(
            "decorator_suffix cannot be an empty string".to_string(),
        ));
    }
    Ok(())
}

/// Validate the raw `[diagnostic]` TOML table and lower it into a
/// `DiagnosticConfig`. Returns the resolved config alongside any
/// non-fatal warnings (unknown diagnostic codes — typos that crema
/// surfaces but does not exit on, so a downstream upgrade adding a
/// new code does not break older configs).
pub fn validate_diagnostic_table(
    raw: HashMap<String, toml::Value>,
) -> Result<(DiagnosticConfig, Vec<String>), ConfigError> {
    let mut preset = Preset::Default;
    let mut overrides: HashMap<String, Severity> = HashMap::new();
    let mut warnings: Vec<String> = Vec::new();

    for (key, value) in raw {
        if key == "preset" {
            let s = value.as_str().ok_or_else(|| {
                ConfigError::InvalidDiagnostic(format!(
                    "preset must be a string, got {}",
                    value.type_str()
                ))
            })?;
            preset = Preset::from_config_str(s).ok_or_else(|| {
                ConfigError::InvalidDiagnostic(format!(
                    "unknown preset {:?}; expected one of \"default\", \"all_error\", \"all_ignore\"",
                    s
                ))
            })?;
            continue;
        }

        let s = value.as_str().ok_or_else(|| {
            ConfigError::InvalidDiagnostic(format!(
                "value for {:?} must be a string, got {}",
                key,
                value.type_str()
            ))
        })?;
        let severity = Severity::from_config_str(s).ok_or_else(|| {
            ConfigError::InvalidDiagnostic(format!(
                "invalid severity {:?} for {:?}; expected one of \"error\", \"warning\", \"information\", \"hint\", \"ignore\"",
                s, key
            ))
        })?;
        if !DiagnosticKind::is_known_code(&key) {
            warnings.push(format!(
                "warning: unknown diagnostic code {:?} in crema.toml [diagnostic]; ignoring",
                key
            ));
            continue;
        }
        if DiagnosticKind::is_always_error_code(&key) {
            return Err(ConfigError::InvalidDiagnostic(format!(
                "{:?} cannot be overridden; it is always reported as error",
                key
            )));
        }
        overrides.insert(key, severity);
    }

    Ok((DiagnosticConfig { preset, overrides }, warnings))
}
