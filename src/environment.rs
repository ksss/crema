use crate::name::Name;
use crate::type_name::TypeName;
use crate::type_param::TypeVarKey;
use crate::types::Ty;

/// Provenance of a per-decl file slot (ADR-0028 provenance-enum
/// refactor). Replaces the overloaded `Option<Name>` that used to let
/// three unrelated producers collapse onto the same `None`: G-snapshot
/// decode (`crate::snapshot::backend`), infusion synthesis
/// (`crate::infusion_collector::active_record_synthesis` /
/// `zeitwerk_synthesis` / `config`), and file-less test sources
/// (`EnvironmentDraft::load_rbs_source`). That overload caused a real
/// regression (commit a5541d63): the S1c G double-merge guard read
/// `file.is_none()` as "G-origin" and wrongly matched a synthesized
/// decl too, skipping a gem merge it should have run.
///
/// Consumers that only want "the real path, if any" (diagnostics,
/// `path_index`, per-file `use`/resolve-flag lookup) go through
/// [`Self::file`] instead of matching every variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclOrigin {
    /// Declared in a real source file (RBS or Ruby) at the carried path.
    Path(Name),
    /// Merged in from an attached G-snapshot backend
    /// (`crate::snapshot::backend::GSnapshotBackend`); the mirror does not
    /// carry an entry-level file slot, so this decl's own AST
    /// `source_file` (when it has one) is the only path available.
    GSnapshot,
    /// Synthesized by an infusion collector (`active_record_synthesis`,
    /// `zeitwerk_synthesis`, `config`) rather than parsed from a file.
    /// Carries which collector produced it — the granularity
    /// `DuplicatedMethodDefinition`'s `duplicate_source` surfaces when the
    /// colliding member has no file to point at (`mid_dup_method_infusion_provenance`).
    ///
    /// The second slot is the *owner file* (ADR-0028 S2 relaxation, todo
    /// `infusion_rb_edit_synthesized_ownership`): the single `.rb` file
    /// whose contents this synthesized decl is derived from, when one
    /// exists — e.g. `active_record_synthesis` tags a model's
    /// `ActiveRecord_Relation` bundle with the file declaring the model.
    /// Owner-tagged decls are registered in `path_index` under the owner
    /// (see [`Self::index_file`]) so a warm differential `unload` of that
    /// file finds and strips them for re-synthesis. `None` means the decl
    /// is derived from more than one file (or from global inputs like
    /// `db/schema.rb`) — such decls survive `unload`, which is sound only
    /// because the warm-eligibility probe sent any edit that could
    /// invalidate them down the cold path.
    Synthesized(InfusionUnit, Option<Name>),
    /// No path and no other classification — e.g. inline RBS source fed
    /// through `EnvironmentDraft::load_rbs_source` in tests.
    Unspecified,
}

impl DeclOrigin {
    /// The real path backing this decl, or `None` for every non-`Path`
    /// variant. Compatibility accessor for consumers that only care
    /// whether a path exists, not why it's absent.
    pub fn file(self) -> Option<Name> {
        match self {
            DeclOrigin::Path(name) => Some(name),
            DeclOrigin::GSnapshot | DeclOrigin::Synthesized(..) | DeclOrigin::Unspecified => None,
        }
    }

    /// The file this decl is *indexed and unloaded under*: the real path
    /// for `Path`, the owner file for owner-tagged `Synthesized` decls.
    /// This is `path_index`'s and `unload`'s notion of ownership —
    /// broader than [`Self::file`], which keeps answering "the path the
    /// decl was parsed from" (a synthesized decl has none) for
    /// diagnostics and per-file `use`/resolve lookups.
    pub fn index_file(self) -> Option<Name> {
        match self {
            DeclOrigin::Path(name) => Some(name),
            DeclOrigin::Synthesized(_, owner) => owner,
            DeclOrigin::GSnapshot | DeclOrigin::Unspecified => None,
        }
    }
}

/// Which infusion collector synthesized a `DeclOrigin::Synthesized` decl.
/// The granularity is "config unit" — coarser than an individual rule
/// (e.g. a single Rails association macro), finer than "crema synthesis"
/// as an undifferentiated whole. Matches the collector module that
/// constructs the decl, not the `[infusion.*]` config flag that gates it
/// (`Zeitwerk` has no config flag of its own — it fires under the
/// combined `rails_enabled()` gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InfusionUnit {
    ActiveRecord,
    Zeitwerk,
    Config,
    Paranoia,
    ActionMailer,
    Sidekiq,
}

impl InfusionUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            InfusionUnit::ActiveRecord => "activerecord",
            InfusionUnit::Zeitwerk => "zeitwerk",
            InfusionUnit::Config => "config",
            InfusionUnit::Paranoia => "paranoia",
            InfusionUnit::ActionMailer => "actionmailer",
            InfusionUnit::Sidekiq => "sidekiq",
        }
    }
}

impl From<Option<Name>> for DeclOrigin {
    /// `Some` becomes `Path`; `None` becomes `Unspecified` — the default
    /// classification for producers that only know "real path or no
    /// path" (RBS/Ruby loaders). G-snapshot decode and infusion
    /// synthesis construct their variants directly instead of going
    /// through this conversion.
    fn from(file: Option<Name>) -> Self {
        match file {
            Some(name) => DeclOrigin::Path(name),
            None => DeclOrigin::Unspecified,
        }
    }
}

/// A type alias definition with its resolved body and type parameters.
#[derive(Debug, Clone)]
pub struct TypeAliasEntry {
    pub params: Vec<TypeVarKey>,
    pub body: Ty,
}

/// Whether an RBS declaration is a class, module, or interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKindLocal {
    Class,
    Module,
    Interface,
}

/// Whether a `name = target` alias stands for a class or a module.
/// Matches RBS's ClassAliasEntry / ModuleAliasEntry distinction in
/// `lib/rbs/environment.rb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKindLocal {
    ClassAlias,
    ModuleAlias,
}

/// Records a single `class X = Y` or `module X = Y` declaration.
///
/// `old_name` is resolved to an absolute name at load time and may itself
/// point at another alias. Chains are followed lazily by
/// [`DefinitionBuilder::normalize_module_name`] with cycle detection.
#[derive(Debug, Clone)]
pub struct ClassAliasEntry {
    pub kind: AliasKindLocal,
    pub old_name: TypeName,
}

/// A reference to a superclass or a mixin (include/extend/prepend) with its
/// applied type arguments.
///
/// The `args` are interned `Ty` values, possibly containing `TypeVariable`
/// entries from the *referring* class's type parameter scope. When walking
/// the ancestor chain, these args are substituted through the accumulated
/// bindings to produce concrete types for the target's scope.
///
/// `args` shape invariant: arity-valid refs (`min_count <= args.len() <=
/// params.len()`) are padded to `params.len()` by `mixin_ref` using each
/// param's `default_type` (rbs `TypeParam.normalize_args`). Arity-invalid
/// refs (excess or insufficient) keep the raw user-supplied length so
/// `validate_applied_type_args` can report the original `got` count.
/// Consumers that zip `args` with the target's params (`ancestor_bindings`)
/// rely on the padded shape; consumers that count `args` for diagnostics
/// (`validator::check_one`) rely on the raw shape for the arity-invalid arm.
///
/// `location` points at the declaration site (the `include M[T]` line in
/// `.rbs`, or the `include M #[T]` line in `.rb`). Populated by both
/// loaders; `None` only for refs that have no source (implicit supers
/// filled by `finalize_implicit_superclasses`). Consumed by the build-layer
/// validator (ADR-0013) to attach `file:line` to arity diagnostics.
///
/// Phase 6: `location` carries file identity via [`crate::location::SourceLocation`].
#[derive(Debug, Clone)]
pub struct MixinRef {
    pub name: TypeName,
    pub args: Vec<Ty>,
    pub location: Option<crate::location::SourceLocation>,
}

impl MixinRef {
    pub fn bare(name: TypeName) -> Self {
        MixinRef {
            name,
            args: vec![],
            location: None,
        }
    }
}

pub mod draft;
pub(crate) mod extras_state;
pub mod frozen;
pub mod invalidation;
pub mod resolution;
pub mod ruby_decl;
pub mod use_map;

/// `crate::environment::Environment` is the frozen ast-aggregation layer
/// produced by `EnvironmentDraft::build` (ADR-0017).
pub use frozen::Environment;
pub use frozen::ScanScope;
