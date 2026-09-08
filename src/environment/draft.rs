//! Draft-side environment entries — mutable construction-time forms
//! that pair an `Arc<ast::declarations::*>` with the namespace context it was
//! declared in.
//!
//! Phase 2 of ADR-0017. The frozen counterparts (`ClassEntry`,
//! `ModuleEntry`, `InterfaceEntry`, `Environment`) and
//! `EnvironmentDraft::build` arrive in Phase 4. Stage B introduces
//! the per-name draft entry types (this module's contents); Stage C
//! adds [`EnvironmentDraft`] on top.
//!
//! Two ADR-0017 design points show up here:
//!
//! - **Per-decl context, not per-member context.** Each
//!   `(Context, Arc<Class>)` pair attaches the namespace stack to
//!   the declaration as a whole. All members inside that decl share
//!   the same context — encoded by the type, not by convention.
//! - **`Arc<ast::declarations::*>` everywhere.** Entries hold shared references
//!   so the future frozen `Environment` and `Definition` layers can
//!   share allocations without reparsing.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::ast::TypeParam;
use crate::ast::declarations::{
    ClassAliasDeclaration as ClassAlias, ClassDeclaration as Class,
    ConstantDeclaration as Constant, GlobalDeclaration as Global,
    InterfaceDeclaration as Interface, ModuleAliasDeclaration as ModuleAlias,
    ModuleDeclaration as Module, TypeAliasDeclaration as TypeAlias,
};
use crate::ast::directives::Directive;
use crate::ast::ruby::annotations::InlineAliasKind;
use crate::ast::ruby::declarations::{
    ClassDecl as RubyClassDecl, ClassModuleAliasDecl as RubyClassModuleAliasDecl,
    ModuleDecl as RubyModuleDecl,
};
use crate::environment::DeclOrigin;
use crate::environment::extras_state::PathIndexState;
use crate::environment::frozen::GOverlap;
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassAliasEntry as FrozenClassAliasEntry, ClassDeclaration, ClassEntry,
    ClassOrModule, ClassOrModuleAliasEntry, Environment, InterfaceEntry, ModuleAliasDeclaration,
    ModuleAliasEntry as FrozenModuleAliasEntry, ModuleDeclaration, ModuleEntry,
    NormalizeModuleNameResult,
};
use crate::environment::resolution::{self, FlattenedDecls, TypeNameResolver};
use crate::environment::use_map::UseMap;
use crate::location::RubyLocation;
use crate::name::{Name, NameTable, Symbol};
use crate::snapshot::backend::{GClassKind, GSnapshotBackend};
use crate::type_name::TypeName;


/// The namespace stack visible to a declaration at the point it was
/// loaded — outer scopes first, innermost last. Held as a shared slice
/// so that nested decls can reuse the parent's allocation rather than
/// rebuilding it per child.
pub type Context = Arc<[TypeName]>;

/// Draft-side discriminator over the two declaration sources for an
/// open class. `Signature` is an RBS-loaded `ast::declarations::Class`;
/// `Ruby` is an inline-annotated `ast::ruby::declarations::ClassDecl`.
/// Mirrors the frozen-side
/// [`crate::environment::frozen::ClassDeclaration`] but holds the draft
/// `Arc` payloads so the build step can rewrite raws without touching
/// the originals.
#[derive(Debug, Clone)]
pub enum ClassDeclarationDraft {
    Signature(Arc<Class>),
    Ruby(Arc<RubyClassDecl>),
}

#[derive(Debug, Clone)]
pub enum ModuleDeclarationDraft {
    Signature(Arc<Module>),
    Ruby(Arc<RubyModuleDecl>),
}

/// Mutable draft for a single class name.
///
/// Mirrors `RBS::Environment::ClassEntry` (rbs's ClassEntry is itself
/// a thin wrapper around `context_decls`). Multiple decls for the
/// same class — open class — push onto the same draft via
/// [`push_signature`](Self::push_signature) /
/// [`push_ruby`](Self::push_ruby); identity-level validation
/// (`type_params` consistency, `super_class` agreement) runs at
/// `EnvironmentDraft::build` time in Phase 4 and is not enforced here.
#[derive(Debug, Clone)]
pub struct ClassEntryDraft {
    pub name: TypeName,
    pub context_decls: Vec<(DeclOrigin, Context, ClassDeclarationDraft)>,
}

impl ClassEntryDraft {
    pub fn new(name: TypeName) -> Self {
        Self {
            name,
            context_decls: Vec::new(),
        }
    }

    pub fn push_signature(&mut self, file: DeclOrigin, context: Context, decl: Arc<Class>) {
        self.context_decls
            .push((file, context, ClassDeclarationDraft::Signature(decl)));
    }

    pub fn push_ruby(&mut self, file: DeclOrigin, context: Context, decl: Arc<RubyClassDecl>) {
        self.context_decls
            .push((file, context, ClassDeclarationDraft::Ruby(decl)));
    }
}

/// Mutable draft for a single module name.
///
/// Mirrors `RBS::Environment::ModuleEntry`. Same shape as
/// [`ClassEntryDraft`] but distinct at the type level so class-only
/// fields (super_class) and module-only fields (self_types) cannot be
/// referenced through a tag-discriminated value.
#[derive(Debug, Clone)]
pub struct ModuleEntryDraft {
    pub name: TypeName,
    pub context_decls: Vec<(DeclOrigin, Context, ModuleDeclarationDraft)>,
}

impl ModuleEntryDraft {
    pub fn new(name: TypeName) -> Self {
        Self {
            name,
            context_decls: Vec::new(),
        }
    }

    pub fn push_signature(&mut self, file: DeclOrigin, context: Context, decl: Arc<Module>) {
        self.context_decls
            .push((file, context, ModuleDeclarationDraft::Signature(decl)));
    }

    pub fn push_ruby(&mut self, file: DeclOrigin, context: Context, decl: Arc<RubyModuleDecl>) {
        self.context_decls
            .push((file, context, ModuleDeclarationDraft::Ruby(decl)));
    }
}

/// Mutable draft for a single interface name.
///
/// Mirrors `RBS::Environment::InterfaceEntry`. Unlike classes and
/// modules, interfaces are not open: a single decl per name is the
/// whole entry, so `context_decls` collapses to one `(context, decl)`
/// pair and there is no `push` method.
#[derive(Debug, Clone)]
pub struct InterfaceEntryDraft {
    pub name: TypeName,
    pub file: DeclOrigin,
    pub context: Context,
    pub decl: Arc<Interface>,
}

/// Discriminated union over class- and module-shaped drafts.
///
/// `EnvironmentDraft::class_decls` will key from `TypeName` to this
/// enum, matching `RBS::Environment#class_decls`'s
/// `Hash[TypeName, ModuleEntry | ClassEntry]`. Single-source-of-truth
/// at the type level: a value tagged `Class` cannot be used where the
/// caller expects a module.
#[derive(Debug, Clone)]
pub enum ClassOrModuleDraft {
    Class(ClassEntryDraft),
    Module(ModuleEntryDraft),
}

/// Draft-side single-decl entry. Generic over the decl payload so
/// the same struct serves both type-alias and constant maps; globals
/// use the dedicated [`GlobalEntry`] instead because their key is a
/// [`Name`], not a [`TypeName`].
///
/// Distinct from the frozen-side [`SingleEntry`] because the draft
/// stores the draft-side [`TypeName`] for duplicate-detection keys;
/// the frozen counterpart holds a resolved [`TypeName`] (Phase 4d).
#[derive(Debug, Clone)]
pub struct SingleEntryDraft<D> {
    pub name: TypeName,
    pub file: DeclOrigin,
    pub context: Context,
    pub decl: Arc<D>,
}

/// Frozen single-decl entry — the post-build counterpart of
/// [`SingleEntryDraft`]. `name` is a structured [`TypeName`] resolved
/// at `EnvironmentDraft::build` time.
#[derive(Debug, Clone, PartialEq)]
pub struct SingleEntry<D> {
    pub name: TypeName,
    pub file: DeclOrigin,
    pub context: Context,
    pub decl: Arc<D>,
}

/// Phase 2 helpers: convenience aliases that fix the decl payload.
pub type TypeAliasEntry = SingleEntry<TypeAlias>;
pub type ConstantEntry = SingleEntry<Constant>;
pub type TypeAliasEntryDraft = SingleEntryDraft<TypeAlias>;
pub type ConstantEntryDraft = SingleEntryDraft<Constant>;

/// Single-decl entry for global declarations.
///
/// Distinct from [`SingleEntry`] because globals are keyed by [`Name`]
/// (the bare identifier) rather than by [`TypeName`]; the ADR-0017
/// `SingleEntry<D>` shape with a `name: TypeName` field cannot
/// represent that without a structural lie.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalEntry {
    pub name: Symbol,
    pub file: DeclOrigin,
    pub context: Context,
    pub decl: Arc<Global>,
}

/// Class- or module-alias declaration payload.
///
/// `Class` / `Module` variants carry RBS-side `class A = B` / `module A = B`
/// declarations, whose backing structs (`ClassAlias` / `ModuleAlias`) differ
/// by kind. `Ruby` carries inline-Ruby `Foo = Bar #: class-alias` /
/// `#: module-alias` declarations; both kinds share the single
/// `RubyClassModuleAliasDecl` struct, so the class/module distinction is read
/// from `decl.annotation.kind()` rather than split into two variants. The
/// frozen side keeps that distinction on the wrapping entry
/// (`ClassAliasDeclaration::Ruby` / `ModuleAliasDeclaration::Ruby`); the draft
/// defers it until `build_class_alias_entry` reads the kind.
#[derive(Debug, Clone)]
pub enum ClassAliasDraft {
    Class(Arc<ClassAlias>),
    Module(Arc<ModuleAlias>),
    Ruby(Arc<RubyClassModuleAliasDecl>),
}

impl ClassAliasDraft {
    pub fn new_name(&self) -> &TypeName {
        match self {
            ClassAliasDraft::Class(decl) => &decl.new_name,
            ClassAliasDraft::Module(decl) => &decl.new_name,
            ClassAliasDraft::Ruby(decl) => &decl.new_name,
        }
    }

    /// Raw path string of the alias RHS (`old_name`). Mirrors rbs
    /// `class_alias_decls[name].decl.old_name` access; the raw form is
    /// what the resolver re-parses when following the alias chain.
    ///
    /// For the Ruby variants, `decl.old_name(names)` is documented to
    /// return `None` only when the inline collector has already dropped
    /// the declaration with a diagnostic; reaching this method with a
    /// missing RHS is a contract break, so we surface it as a panic
    /// rather than silently aliasing to `""`.
    pub fn old_name_raw(&self, names: &NameTable) -> String {
        match self {
            ClassAliasDraft::Class(decl) => names.resolve(decl.old_name),
            ClassAliasDraft::Module(decl) => names.resolve(decl.old_name),
            ClassAliasDraft::Ruby(decl) => names.resolve(decl.old_name(names).expect(
                "Ruby alias decl reached old_name_raw without RHS — inline collector should have dropped it",
            )),
        }
    }
}

/// Class- or module-alias entry.
///
/// Mirrors `RBS::Environment::ClassAliasEntry` /
/// `RBS::Environment::ModuleAliasEntry`, collapsed with the
/// [`ClassAliasDraft`] value carries the class/module distinction.
///
/// Lives in [`crate::environment::draft`] alongside the other Phase 2
/// draft entries; the same name in [`crate::environment`] is a
/// pre-ADR-0017 type and stays untouched until Phase 4 retires it.
#[derive(Debug, Clone)]
pub struct ClassAliasEntry {
    pub name: TypeName,
    pub file: DeclOrigin,
    pub context: Context,
    pub decl: ClassAliasDraft,
}

/// Key into [`EnvironmentDraft::path_index`] / `Environment::path_index`
/// (ADR-0028 Decision 1's crema-only path→entry reverse index). Wraps the
/// declared name with the decl map it lives in, mirroring the six
/// `EnvironmentDraft` maps (`class_decls`, `interface_decls`,
/// `class_alias_decls`, `type_alias_decls`, `constant_decls`,
/// `global_decls`) one-for-one. A bare [`TypeName`] cannot serve as the key
/// on its own: `class_decls`, `constant_decls`, and `class_alias_decls`
/// share one Ruby constant namespace (see
/// [`EnvironmentDraft::check_constant_namespace_free`]), so the same name
/// text can only ever claim one of the three, but the *map* it landed in
/// still has to be recorded to look it back up.
///
/// `ClassOrModule` covers both `class_decls` variants
/// ([`ClassOrModuleDraft::Class`] and [`ClassOrModuleDraft::Module`]),
/// and `ClassAlias` covers both `class A = B` and `module A = B` aliases —
/// each pair shares a single underlying map, so the key does too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathIndexKey {
    ClassOrModule(TypeName),
    Interface(TypeName),
    ClassAlias(TypeName),
    TypeAlias(TypeName),
    Constant(TypeName),
    Global(Symbol),
}

/// Top-level mutable draft environment.
///
/// Mirrors `RBS::Environment` in mutable form. Producers (Phase 3 in
/// [`crate::ast_builder`], [`crate::rbs_loader`], and
/// [`crate::inline_parser`]) call `insert_*` for each declaration
/// they extract. Phase 4 will add `build(self) -> Result<Environment,
/// BuildError>` to freeze the draft into the post-build form;
/// Stage C deliberately leaves `build` unimplemented so the
/// build/frozen axis is introduced in one piece in Phase 4.
///
/// Stage C only enforces the validations that fall out naturally at
/// insertion time: a name cannot be claimed by both a class and a
/// module, and single-decl kinds (interface, type alias, constant,
/// global, class alias) reject any second decl. Cross-decl checks
/// that need to compare members of the same name (`type_params`
/// agreement, `super_class` agreement, `normalize_module_name`
/// precompute, `primary_decl` selection) belong to `build` and are
/// listed as `BuildError` variants for that future call site.
///
/// `names` owns the [`NameTable`] for this draft and is moved into the
/// frozen [`Environment`] at build time. Callers building `Name` /
/// `TypeName` values for use against this draft must do so through
/// `draft.names()` so the same symbol universe survives the build move.
#[derive(Debug)]
pub struct EnvironmentDraft {
    pub(crate) class_decls: FxHashMap<TypeName, ClassOrModuleDraft>,
    pub(crate) interface_decls: FxHashMap<TypeName, InterfaceEntryDraft>,
    pub(crate) class_alias_decls: FxHashMap<TypeName, ClassAliasEntry>,
    pub(crate) type_alias_decls: FxHashMap<TypeName, TypeAliasEntryDraft>,
    pub(crate) constant_decls: FxHashMap<TypeName, ConstantEntryDraft>,
    pub(crate) global_decls: FxHashMap<Symbol, GlobalEntry>,
    pub(crate) sources: Vec<Option<RubyLocation>>,
    pub(crate) names: NameTable,
    /// `use` directives collected per source file. Keyed by the
    /// `file` field carried on each declaration's [`RubyLocation`]. A
    /// `None`-`file` declaration (typically a test's inline RBS
    /// source loaded via [`Self::load_rbs_source`]) consults
    /// [`Self::default_directives`] instead. The
    /// [`crate::ast::directives::UseClause`]s here drive the
    /// `UseMap` built at [`Self::build`] time.
    pub(crate) file_directives: FxHashMap<Name, Vec<crate::ast::directives::Directive>>,
    /// `use` directives for sources that have no `Name` file
    /// identity (inline RBS in tests). Acts as the file-less
    /// fallback bucket so a single test source can still drive
    /// `use`-aware resolution.
    pub(crate) default_directives: Vec<crate::ast::directives::Directive>,
    /// Per-file `# resolve-type-names:` flag (rbs
    /// `AST::Directives::ResolveTypeNamesDirective`). An entry is recorded only
    /// when the file carries the magic comment; absence implies the
    /// default `true` (resolve as normal). `false` makes [`Self::build`]
    /// feed that file's raw AST through to the frozen environment
    /// without rewriting any type names.
    pub(crate) file_resolve_type_names: FxHashMap<Name, bool>,
    /// `resolve_type_names` flag for the file-less inline-RBS path
    /// (mirrors [`Self::default_directives`]). Defaults to `true`.
    pub(crate) default_resolve_type_names: bool,
    /// Lazy G-layer snapshot the A-layer declarations are grafted onto
    /// (ADR-0028 slice 2a-2). When set, the insert-time duplicate checks
    /// also consult the snapshot's key sets (so `class Foo` over a gem
    /// `module Foo` errors exactly like a combined draft would), and
    /// [`Self::build`] feeds the snapshot's names into the resolver,
    /// merges reopened entries per name, and hands the backend to the
    /// frozen [`Environment`].
    pub(crate) g: Option<Arc<GSnapshotBackend>>,
    /// Reverse index from a source file to the [`PathIndexKey`]s it
    /// contributed (ADR-0028 Decision 1's crema-only extension over rbs's
    /// O(env) `unload`). Populated incrementally by the `insert_*` methods
    /// as each declaration lands, not by a `build`-time walk, so it stays
    /// O(inserted) rather than O(env).
    ///
    /// A [`DeclOrigin`] with no [`DeclOrigin::file`] records nothing: such
    /// a declaration (a test's inline RBS source loaded via
    /// [`Self::load_rbs_source`], or a Ruby decl re-drafted through
    /// [`Self::from_frozen`]) has no path a future `unload(paths)` could
    /// ever pass in, so there is no key to index it under.
    ///
    /// G-layer (gem snapshot) entries are never recorded here — only
    /// A-layer `insert_*` calls populate this map; the G snapshot has its
    /// own persisted structures (ADR-0028 Decision 3).
    ///
    /// Two-layer (ADR-0028 F6 remainder): an unload-carried draft keeps
    /// the frozen environment's baseline behind the overlay; fresh
    /// drafts are overlay-only.
    pub(crate) path_index: PathIndexState,
    /// COW overlay carrying the surviving frozen decl maps from
    /// [`Environment::unload`] (ADR-0028 Decision 2, slice S1b). `None` for
    /// every other draft (fresh `EnvironmentDraft::new()`, G-only builds,
    /// `from_frozen`). When `Some`, [`Self::build`] seeds each Pass 2 output
    /// map with the corresponding overlay map before resolving and merging
    /// in the freshly inserted draft entries, and the `insert_*` duplicate
    /// checks additionally consult it — mirroring the existing `self.g`
    /// checks one for one, but eagerly folded into the final maps at build
    /// time rather than staying lazy (overlay has no persisted backend to
    /// defer to).
    pub(crate) overlay: Option<Box<FrozenOverlay>>,
}

/// The six frozen decl maps surviving an [`Environment::unload`] call,
/// carried into a draft as a COW overlay (ADR-0028 Decision 2).
///
/// Field-for-field mirror of the six maps on
/// [`crate::environment::frozen::Environment`]. Entries here are already
/// resolved (they came from a previously built `Environment`); `build`
/// seeds them straight into its Pass 2 output and never re-resolves them —
/// the crema encoding of rbs `resolve_type_names(only:)`'s pass-through of
/// unchanged decls.
#[derive(Debug, Default)]
pub(crate) struct FrozenOverlay {
    pub(crate) class_decls: FxHashMap<TypeName, ClassOrModule>,
    pub(crate) interface_decls: FxHashMap<TypeName, InterfaceEntry>,
    pub(crate) class_alias_decls: FxHashMap<TypeName, ClassOrModuleAliasEntry>,
    pub(crate) type_alias_decls: FxHashMap<TypeName, TypeAliasEntry>,
    pub(crate) constant_decls: FxHashMap<TypeName, ConstantEntry>,
    pub(crate) global_decls: FxHashMap<Symbol, GlobalEntry>,
    /// `class_decls` keys [`Environment::unload`](crate::environment::frozen::Environment::unload)
    /// stripped of their G-origin (`DeclOrigin::GSnapshot`) decl(s) because
    /// the key is also declared by the attached G snapshot (ADR-0028 slice S1c).
    /// `EnvironmentDraft::build`'s G per-name merge step re-merges these
    /// unconditionally, on top of its default rule of only merging entries
    /// that don't already carry a G-origin decl — see that step's doc
    /// comment for why both signals are checked.
    pub(crate) g_remerge: FxHashSet<TypeName>,
    /// Carried resolver table (ADR-0028 slice S2), already delta-updated
    /// by [`Environment::unload`](crate::environment::frozen::Environment::unload).
    /// [`EnvironmentDraft::build`] moves these out (via `mem::take`) as
    /// Pass 1's starting point instead of re-walking every surviving decl
    /// map — see that method's doc comment.
    pub(crate) all_names: FxHashSet<TypeName>,
    pub(crate) aliases: FxHashMap<TypeName, (String, Context)>,
}

impl FrozenOverlay {
    /// Class-or-module kind of an overlay-carried name — the open-class
    /// kind check's view of the surviving A layer.
    fn class_kind(&self, name: &TypeName) -> Option<GClassKind> {
        self.class_decls.get(name).map(|entry| match entry {
            ClassOrModule::Class(_) => GClassKind::Class,
            ClassOrModule::Module(_) => GClassKind::Module,
        })
    }
}

/// Errors emitted by [`EnvironmentDraft`] insertion and (Phase 4) by
/// `build`.
///
/// Stage C uses only [`BuildError::DuplicatedDeclaration`]. The other
/// variants are declared up-front so Phase 4 can extend their
/// payloads without churning the enum's identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A name is claimed by two declarations whose kinds cannot
    /// coexist — e.g. a class and a module, or two type aliases for
    /// the same name. Open-class merging (multiple class decls for
    /// the same name) is not a duplicate; those accumulate on the
    /// same [`ClassEntryDraft`].
    DuplicatedDeclaration { name: TypeName },
    /// A name is claimed by two type aliases or constants where the
    /// key would normally be a [`TypeName`] but the actual conflict
    /// happens on a [`Name`] (currently only globals).
    DuplicatedGlobal { name: Symbol },
    /// Open-class decls for the same name disagree on
    /// `type_params`. Phase 4 fills in the per-decl detail; Stage C
    /// only declares the variant.
    GenericParameterMismatch { name: TypeName },
    /// Open-class decls disagree on `super_class`. Same Phase 4
    /// expansion plan as [`BuildError::GenericParameterMismatch`].
    SuperclassConflict { name: TypeName },
}

impl BuildError {
    /// Render the error with interned symbols resolved through `names`.
    /// The bare `Debug` impl prints `Symbol(<spur-id>)` which is opaque to
    /// humans and AIs alike (`src/name.rs:54`); this helper resolves every
    /// embedded name so the message reads like
    /// `DuplicatedDeclaration { name: ::JSON::_Reader (Interface) }`.
    pub fn format_with(&self, names: &NameTable) -> String {
        // Roots cannot be declared, so a missing kind never reaches a
        // BuildError in practice; label it explicitly rather than
        // leaking `Some(...)` if it ever does.
        let kind_label = |name: &TypeName| match names.type_name_kind(*name) {
            Some(kind) => format!("{kind:?}"),
            None => String::from("Root"),
        };
        match self {
            BuildError::DuplicatedDeclaration { name } => {
                format!(
                    "DuplicatedDeclaration {{ name: {} ({}) }}",
                    names.resolve(name),
                    kind_label(name),
                )
            }
            BuildError::DuplicatedGlobal { name } => {
                format!("DuplicatedGlobal {{ name: {} }}", names.resolve(*name))
            }
            BuildError::GenericParameterMismatch { name } => {
                format!(
                    "GenericParameterMismatch {{ name: {} ({}) }}",
                    names.resolve(name),
                    kind_label(name),
                )
            }
            BuildError::SuperclassConflict { name } => {
                format!(
                    "SuperclassConflict {{ name: {} ({}) }}",
                    names.resolve(name),
                    kind_label(name),
                )
            }
        }
    }
}

impl Default for EnvironmentDraft {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvironmentDraft {
    pub fn new() -> Self {
        Self::new_with_names(NameTable::new())
    }

    /// Same as [`Self::new`] but carries an existing [`NameTable`] instead
    /// of starting a fresh one. [`Environment::unload`](crate::environment::frozen::Environment::unload)
    /// uses this to hand the frozen environment's table forward — every
    /// `Name` / `TypeName` reachable through the surviving overlay entries
    /// was interned against it (ADR-0028 slice S1b).
    pub(crate) fn new_with_names(names: NameTable) -> Self {
        Self {
            class_decls: FxHashMap::default(),
            interface_decls: FxHashMap::default(),
            class_alias_decls: FxHashMap::default(),
            type_alias_decls: FxHashMap::default(),
            constant_decls: FxHashMap::default(),
            global_decls: FxHashMap::default(),
            sources: Vec::new(),
            names,
            file_directives: FxHashMap::default(),
            default_directives: Vec::new(),
            file_resolve_type_names: FxHashMap::default(),
            default_resolve_type_names: true,
            g: None,
            path_index: PathIndexState::default(),
            overlay: None,
        }
    }

    /// Record the [`directives`](crate::ast::directives::Directive)
    /// observed in one source file. A `None` `file` routes the
    /// directives into [`Self::default_directives`] for the file-less
    /// inline-RBS path; otherwise they accumulate per `Name` in
    /// [`Self::file_directives`].
    ///
    /// `ResolveTypeNamesDirective` directives also update the per-file
    /// resolve-flag map; mirroring rbs's `magic_comment` semantics, the
    /// directive remains in the directives list but [`Self::build`]
    /// consults the flag map (not the list) to decide whether to skip
    /// resolution.
    pub fn add_directives(
        &mut self,
        file: Option<Name>,
        directives: &[crate::ast::directives::Directive],
    ) {
        match file {
            Some(name) => {
                self.file_directives
                    .entry(name)
                    .or_default()
                    .extend(directives.iter().cloned());
                for dir in directives {
                    if let crate::ast::directives::Directive::ResolveTypeNames(rtn) = dir {
                        self.file_resolve_type_names.insert(name, rtn.value);
                    }
                }
            }
            None => {
                self.default_directives.extend(directives.iter().cloned());
                for dir in directives {
                    if let crate::ast::directives::Directive::ResolveTypeNames(rtn) = dir {
                        self.default_resolve_type_names = rtn.value;
                    }
                }
            }
        }
    }

    pub fn names(&self) -> &NameTable {
        &self.names
    }

    /// Record that the draft observed a new source (RBS sig file or
    /// Ruby source). Stage C accepts every source unconditionally;
    /// Phase 4 may reject duplicates or contradictory provenance.
    pub fn add_source(&mut self, location: Option<RubyLocation>) -> Result<(), BuildError> {
        self.sources.push(location);
        Ok(())
    }

    /// Reject names already claimed by another declaration kind in
    /// the same constant namespace. `class_decls`, `constant_decls`,
    /// and `class_alias_decls` all key by the
    /// [`Kind::Class`](crate::type_name::Kind::Class)
    /// slot, which Ruby treats as a single namespace; a name there
    /// can only be claimed by one of them. `interface_decls` and
    /// `type_alias_decls` use distinct kinds (`Interface`, `Alias`)
    /// and are independent.
    ///
    /// With a G-snapshot attached the same rules extend to the gem
    /// layer's key sets — a combined draft would have held those
    /// entries directly, so the graft reproduces the identical
    /// [`BuildError`] at the identical insert (ADR-0028 slice 2a-2).
    fn check_constant_namespace_free(
        &self,
        name: &TypeName,
        skip_class_decls: bool,
        skip_constant_decls: bool,
        skip_class_alias_decls: bool,
    ) -> Result<(), BuildError> {
        if !skip_class_decls && self.class_decls.contains_key(name) {
            return Err(BuildError::DuplicatedDeclaration { name: *name });
        }
        if !skip_constant_decls && self.constant_decls.contains_key(name) {
            return Err(BuildError::DuplicatedDeclaration { name: *name });
        }
        if !skip_class_alias_decls && self.class_alias_decls.contains_key(name) {
            return Err(BuildError::DuplicatedDeclaration { name: *name });
        }
        if let Some(g) = &self.g {
            if !skip_class_decls && g.class_contains(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
            if !skip_constant_decls && g.constant_contains(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
            if !skip_class_alias_decls && g.alias_contains(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
        }
        if let Some(overlay) = &self.overlay {
            if !skip_class_decls && overlay.class_decls.contains_key(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
            if !skip_constant_decls && overlay.constant_decls.contains_key(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
            if !skip_class_alias_decls && overlay.class_alias_decls.contains_key(name) {
                return Err(BuildError::DuplicatedDeclaration { name: *name });
            }
        }
        Ok(())
    }

    /// Attach the lazy G-snapshot backend this draft's declarations are
    /// layered on top of. Must precede every A-layer insert so the
    /// cross-layer duplicate checks see the gem names.
    pub fn attach_g_backend(&mut self, g: Arc<GSnapshotBackend>) {
        debug_assert!(
            self.class_decls.is_empty()
                && self.interface_decls.is_empty()
                && self.class_alias_decls.is_empty()
                && self.type_alias_decls.is_empty()
                && self.constant_decls.is_empty()
                && self.global_decls.is_empty(),
            "attach_g_backend must precede every insert"
        );
        self.g = Some(g);
    }

    /// Cross-layer arm of the open-class kind check: reopening a gem
    /// class as a class (or module as module) is the merge path; the
    /// opposite kind is the duplicate a combined draft reports.
    fn check_g_class_kind(&self, name: &TypeName, want: GClassKind) -> Result<(), BuildError> {
        if let Some(g) = &self.g
            && let Some(kind) = g.class_kind(name)
            && kind != want
        {
            return Err(BuildError::DuplicatedDeclaration { name: *name });
        }
        Ok(())
    }

    /// Overlay arm of the open-class kind check (ADR-0028 slice S1b),
    /// symmetric to [`Self::check_g_class_kind`]: reopening an
    /// overlay-carried class as a class (or module as module) is the merge
    /// path `build` runs later; the opposite kind is a duplicate.
    fn check_overlay_class_kind(
        &self,
        name: &TypeName,
        want: GClassKind,
    ) -> Result<(), BuildError> {
        if let Some(kind) = self.overlay.as_ref().and_then(|o| o.class_kind(name))
            && kind != want
        {
            return Err(BuildError::DuplicatedDeclaration { name: *name });
        }
        Ok(())
    }

    /// Record that `key` was contributed by `file` (ADR-0028 `path_index`).
    /// Only [`DeclOrigin::Path`] is recorded — see [`Self::path_index`].
    /// Every `insert_*` method calls this on each success path, after the
    /// corresponding decl map has already accepted the declaration.
    fn record_path(&mut self, file: DeclOrigin, key: PathIndexKey) {
        record_path_index(&mut self.path_index, file, key);
    }

    /// Append a `class` declaration. Open-class merging into an
    /// existing [`ClassEntryDraft`] is the success path; a name
    /// already taken by a [`ClassOrModuleDraft::Module`], a
    /// constant, or a class alias is a duplicate.
    pub fn insert_class_decl(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<Class>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        // class_decls is checked manually below for the open-class merge;
        // skip it in the cross-map check.
        self.check_constant_namespace_free(&name, true, false, false)?;
        self.check_g_class_kind(&name, GClassKind::Class)?;
        self.check_overlay_class_kind(&name, GClassKind::Class)?;
        match self.class_decls.get_mut(&name) {
            Some(ClassOrModuleDraft::Class(entry)) => {
                entry.push_signature(file, context, decl);
                self.record_path(file, PathIndexKey::ClassOrModule(name));
                Ok(())
            }
            Some(ClassOrModuleDraft::Module(_)) => Err(BuildError::DuplicatedDeclaration { name }),
            None => {
                let mut entry = ClassEntryDraft::new(name);
                entry.push_signature(file, context, decl);
                self.class_decls
                    .insert(name, ClassOrModuleDraft::Class(entry));
                self.record_path(file, PathIndexKey::ClassOrModule(name));
                Ok(())
            }
        }
    }

    /// Append a Ruby (inline-annotated) `class` declaration. Same
    /// open-class merging rules as [`insert_class_decl`]; the source
    /// kind (Signature vs Ruby) is recorded per decl in the entry's
    /// `context_decls` list.
    ///
    /// `name_raw` is the qualified name of the decl. The inline
    /// collector pre-qualifies Ruby decl names against the enclosing
    /// scope, so callers should pass `decl.class_name.clone()` here.
    pub fn insert_ruby_class_decl(
        &mut self,
        file: DeclOrigin,
        context: Context,
        name_raw: TypeName,
        decl: Arc<RubyClassDecl>,
    ) -> Result<(), BuildError> {
        self.check_constant_namespace_free(&name_raw, true, false, false)?;
        self.check_g_class_kind(&name_raw, GClassKind::Class)?;
        self.check_overlay_class_kind(&name_raw, GClassKind::Class)?;
        match self.class_decls.get_mut(&name_raw) {
            Some(ClassOrModuleDraft::Class(entry)) => {
                entry.push_ruby(file, context, decl);
                self.record_path(file, PathIndexKey::ClassOrModule(name_raw));
                Ok(())
            }
            Some(ClassOrModuleDraft::Module(_)) => {
                Err(BuildError::DuplicatedDeclaration { name: name_raw })
            }
            None => {
                let mut entry = ClassEntryDraft::new(name_raw);
                entry.push_ruby(file, context, decl);
                self.class_decls
                    .insert(name_raw, ClassOrModuleDraft::Class(entry));
                self.record_path(file, PathIndexKey::ClassOrModule(name_raw));
                Ok(())
            }
        }
    }

    /// Append a `module` declaration. Symmetric to
    /// [`insert_class_decl`](Self::insert_class_decl) — a name held by
    /// a class, constant, or class alias is a duplicate.
    pub fn insert_module_decl(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<Module>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        self.check_constant_namespace_free(&name, true, false, false)?;
        self.check_g_class_kind(&name, GClassKind::Module)?;
        self.check_overlay_class_kind(&name, GClassKind::Module)?;
        match self.class_decls.get_mut(&name) {
            Some(ClassOrModuleDraft::Module(entry)) => {
                entry.push_signature(file, context, decl);
                self.record_path(file, PathIndexKey::ClassOrModule(name));
                Ok(())
            }
            Some(ClassOrModuleDraft::Class(_)) => Err(BuildError::DuplicatedDeclaration { name }),
            None => {
                let mut entry = ModuleEntryDraft::new(name);
                entry.push_signature(file, context, decl);
                self.class_decls
                    .insert(name, ClassOrModuleDraft::Module(entry));
                self.record_path(file, PathIndexKey::ClassOrModule(name));
                Ok(())
            }
        }
    }

    /// Append a Ruby (inline-annotated) `module` declaration.
    pub fn insert_ruby_module_decl(
        &mut self,
        file: DeclOrigin,
        context: Context,
        name_raw: TypeName,
        decl: Arc<RubyModuleDecl>,
    ) -> Result<(), BuildError> {
        self.check_constant_namespace_free(&name_raw, true, false, false)?;
        self.check_g_class_kind(&name_raw, GClassKind::Module)?;
        self.check_overlay_class_kind(&name_raw, GClassKind::Module)?;
        match self.class_decls.get_mut(&name_raw) {
            Some(ClassOrModuleDraft::Module(entry)) => {
                entry.push_ruby(file, context, decl);
                self.record_path(file, PathIndexKey::ClassOrModule(name_raw));
                Ok(())
            }
            Some(ClassOrModuleDraft::Class(_)) => {
                Err(BuildError::DuplicatedDeclaration { name: name_raw })
            }
            None => {
                let mut entry = ModuleEntryDraft::new(name_raw);
                entry.push_ruby(file, context, decl);
                self.class_decls
                    .insert(name_raw, ClassOrModuleDraft::Module(entry));
                self.record_path(file, PathIndexKey::ClassOrModule(name_raw));
                Ok(())
            }
        }
    }

    /// Insert an `interface` declaration. Interfaces are not open;
    /// any duplicate name fails immediately.
    pub fn insert_interface_decl(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<Interface>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        if self.interface_decls.contains_key(&name)
            || self.g.as_ref().is_some_and(|g| g.interface_contains(&name))
            || self
                .overlay
                .as_ref()
                .is_some_and(|o| o.interface_decls.contains_key(&name))
        {
            return Err(BuildError::DuplicatedDeclaration { name });
        }
        self.interface_decls.insert(
            name,
            InterfaceEntryDraft {
                name,
                file,
                context,
                decl,
            },
        );
        self.record_path(file, PathIndexKey::Interface(name));
        Ok(())
    }

    pub fn insert_type_alias(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<TypeAlias>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        if self.type_alias_decls.contains_key(&name)
            || self
                .g
                .as_ref()
                .is_some_and(|g| g.type_alias_contains(&name))
            || self
                .overlay
                .as_ref()
                .is_some_and(|o| o.type_alias_decls.contains_key(&name))
        {
            return Err(BuildError::DuplicatedDeclaration { name });
        }
        self.type_alias_decls.insert(
            name,
            SingleEntryDraft {
                name,
                file,
                context,
                decl,
            },
        );
        self.record_path(file, PathIndexKey::TypeAlias(name));
        Ok(())
    }

    pub fn insert_constant(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<Constant>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        self.check_constant_namespace_free(&name, false, false, false)?;
        self.constant_decls.insert(
            name,
            SingleEntryDraft {
                name,
                file,
                context,
                decl,
            },
        );
        self.record_path(file, PathIndexKey::Constant(name));
        Ok(())
    }

    pub fn insert_global(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<Global>,
    ) -> Result<(), BuildError> {
        let name = decl.name;
        if self.global_decls.contains_key(&name)
            || self.g.as_ref().is_some_and(|g| g.global_contains(&name))
            || self
                .overlay
                .as_ref()
                .is_some_and(|o| o.global_decls.contains_key(&name))
        {
            return Err(BuildError::DuplicatedGlobal { name });
        }
        self.global_decls.insert(
            name,
            GlobalEntry {
                name,
                file,
                context,
                decl,
            },
        );
        self.record_path(file, PathIndexKey::Global(name));
        Ok(())
    }

    pub fn insert_class_alias(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<ClassAlias>,
    ) -> Result<(), BuildError> {
        let name = decl.new_name;
        self.check_constant_namespace_free(&name, false, false, false)?;
        self.class_alias_decls.insert(
            name,
            ClassAliasEntry {
                name,
                file,
                context,
                decl: ClassAliasDraft::Class(decl),
            },
        );
        self.record_path(file, PathIndexKey::ClassAlias(name));
        Ok(())
    }

    pub fn insert_module_alias(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<ModuleAlias>,
    ) -> Result<(), BuildError> {
        let name = decl.new_name;
        self.check_constant_namespace_free(&name, false, false, false)?;
        self.class_alias_decls.insert(
            name,
            ClassAliasEntry {
                name,
                file,
                context,
                decl: ClassAliasDraft::Module(decl),
            },
        );
        self.record_path(file, PathIndexKey::ClassAlias(name));
        Ok(())
    }

    pub fn insert_ruby_alias(
        &mut self,
        file: DeclOrigin,
        context: Context,
        decl: Arc<RubyClassModuleAliasDecl>,
    ) -> Result<(), BuildError> {
        let new_name = decl.new_name;
        self.check_constant_namespace_free(&new_name, false, false, false)?;
        self.class_alias_decls.insert(
            new_name,
            ClassAliasEntry {
                name: new_name,
                file,
                context,
                decl: ClassAliasDraft::Ruby(decl),
            },
        );
        self.record_path(file, PathIndexKey::ClassAlias(new_name));
        Ok(())
    }

    /// True when any loaded file opted into `# resolve-type-names: false`
    /// (rbs `Directives::ResolveTypeNames`). Under that directive `build`
    /// passes the file's AST through with raw (possibly relative) type
    /// names; a snapshot of the resulting frozen env would lose the flag
    /// and re-resolve those names on rebuild — the one case where the
    /// snapshot roundtrip is not behavior-preserving. Callers use this to
    /// skip snapshotting such environments (see `main.rs` cold path).
    pub fn has_resolve_type_names_bypass(&self) -> bool {
        !self.default_resolve_type_names || self.file_resolve_type_names.values().any(|v| !*v)
    }

    /// Re-draft a frozen G-layer environment so app-layer declarations
    /// (sig/ + inline + infusion) can be inserted on top and built into
    /// one combined environment (ADR-0028 slice 1c).
    ///
    /// Only context-length-1 pairs (top-level declarations) are
    /// re-inserted: the draft never holds flatten-generated entries —
    /// [`Self::build`] regenerates them from the parents' member trees.
    /// Re-inserting the flattened pairs (context length >= 2) would
    /// double-register every nested declaration.
    ///
    /// Dropped relative to `env`: `sources` (nothing outside tests
    /// consumes them; dropping on both the cold and warm snapshot paths
    /// keeps the two byte-identical) and `normalized_module_names`
    /// (recomputed by `build`). The per-decl [`DeclOrigin`] slot survives
    /// the round trip: Signature decls re-insert with `s.source_file`
    /// converted through `DeclOrigin::from` (the AST's own copy, which is
    /// what resolver selection reads), Ruby decls (no `source_file` field
    /// on their AST) re-insert with the frozen entry's carried
    /// `DeclOrigin` unchanged — see ADR-0028 slice S1a.
    ///
    /// `use` directives and per-file resolve flags are also gone, which
    /// is safe because `build` on the re-draft re-resolves an already
    /// resolved AST: absolute names are idempotent (`UseMap::resolve` and
    /// the resolver walk both short-circuit on absolute), and names left
    /// unresolved under full resolution stay unresolved without the maps.
    /// The one non-idempotent case — raw names preserved by
    /// `# resolve-type-names: false` — is excluded at the snapshot write
    /// site via [`Self::has_resolve_type_names_bypass`].
    ///
    /// Errors are pre-formatted with the environment's `NameTable`
    /// (which is consumed either way).
    pub fn from_frozen(env: Environment) -> Result<Self, String> {
        let Environment {
            class_decls,
            interface_decls,
            class_alias_decls,
            type_alias_decls,
            constant_decls,
            global_decls,
            sources: _,
            names,
            normalized_module_names: _,
            g: _,
            g_overlap: _,
            // Dropped like the other derived fields above: the insert_*
            // calls below repopulate it from scratch as they re-run for
            // every context-length-1 pair.
            path_index: _,
            // Dropped like `path_index`: `from_frozen` re-inserts every
            // decl, so `build`'s Pass 1 walk regenerates the resolver
            // table from scratch (no overlay is attached here to carry it
            // forward — ADR-0028 slice S2's delta path is `unload`-only).
            all_names: _,
            aliases: _,
        } = env;

        let mut draft = EnvironmentDraft {
            class_decls: FxHashMap::default(),
            interface_decls: FxHashMap::default(),
            class_alias_decls: FxHashMap::default(),
            type_alias_decls: FxHashMap::default(),
            constant_decls: FxHashMap::default(),
            global_decls: FxHashMap::default(),
            sources: Vec::new(),
            names,
            file_directives: FxHashMap::default(),
            default_directives: Vec::new(),
            file_resolve_type_names: FxHashMap::default(),
            default_resolve_type_names: true,
            g: None,
            path_index: PathIndexState::default(),
            overlay: None,
        };

        for (name, com) in class_decls {
            match com {
                ClassOrModule::Class(e) => {
                    for (file, ctx, decl) in e.context_decls.iter() {
                        if ctx.len() != 1 {
                            continue;
                        }
                        let r = match decl {
                            ClassDeclaration::Signature(s) => draft.insert_class_decl(
                                s.source_file.into(),
                                Arc::clone(ctx),
                                Arc::clone(s),
                            ),
                            ClassDeclaration::Ruby(r) => draft.insert_ruby_class_decl(
                                *file,
                                Arc::clone(ctx),
                                name,
                                Arc::clone(r),
                            ),
                        };
                        if let Err(e) = r {
                            return Err(e.format_with(draft.names()));
                        }
                    }
                }
                ClassOrModule::Module(e) => {
                    for (file, ctx, decl) in e.context_decls.iter() {
                        if ctx.len() != 1 {
                            continue;
                        }
                        let r = match decl {
                            ModuleDeclaration::Signature(s) => draft.insert_module_decl(
                                s.source_file.into(),
                                Arc::clone(ctx),
                                Arc::clone(s),
                            ),
                            ModuleDeclaration::Ruby(r) => draft.insert_ruby_module_decl(
                                *file,
                                Arc::clone(ctx),
                                name,
                                Arc::clone(r),
                            ),
                        };
                        if let Err(e) = r {
                            return Err(e.format_with(draft.names()));
                        }
                    }
                }
            }
        }

        for (_, e) in interface_decls {
            if e.context.len() != 1 {
                continue;
            }
            let file = e.decl.source_file.into();
            if let Err(err) = draft.insert_interface_decl(file, e.context, e.decl) {
                return Err(err.format_with(draft.names()));
            }
        }

        for (_, e) in class_alias_decls {
            match e {
                ClassOrModuleAliasEntry::Class(a) => {
                    if a.context.len() != 1 {
                        continue;
                    }
                    let ruby_file = a.file;
                    match a.decl {
                        ClassAliasDeclaration::Signature(s) => {
                            let file = s.source_file.into();
                            if let Err(err) = draft.insert_class_alias(file, a.context, s) {
                                return Err(err.format_with(draft.names()));
                            }
                        }
                        ClassAliasDeclaration::Ruby(r) => {
                            if let Err(err) = draft.insert_ruby_alias(ruby_file, a.context, r) {
                                return Err(err.format_with(draft.names()));
                            }
                        }
                    }
                }
                ClassOrModuleAliasEntry::Module(a) => {
                    if a.context.len() != 1 {
                        continue;
                    }
                    let ruby_file = a.file;
                    match a.decl {
                        ModuleAliasDeclaration::Signature(s) => {
                            let file = s.source_file.into();
                            if let Err(err) = draft.insert_module_alias(file, a.context, s) {
                                return Err(err.format_with(draft.names()));
                            }
                        }
                        ModuleAliasDeclaration::Ruby(r) => {
                            if let Err(err) = draft.insert_ruby_alias(ruby_file, a.context, r) {
                                return Err(err.format_with(draft.names()));
                            }
                        }
                    }
                }
            }
        }

        for (_, e) in type_alias_decls {
            if e.context.len() != 1 {
                continue;
            }
            let file = e.decl.source_file.into();
            if let Err(err) = draft.insert_type_alias(file, e.context, e.decl) {
                return Err(err.format_with(draft.names()));
            }
        }

        for (_, e) in constant_decls {
            if e.context.len() != 1 {
                continue;
            }
            if let Err(err) = draft.insert_constant(e.file, e.context, e.decl) {
                return Err(err.format_with(draft.names()));
            }
        }

        for (_, e) in global_decls {
            if e.context.len() != 1 {
                continue;
            }
            if let Err(err) = draft.insert_global(e.file, e.context, e.decl) {
                return Err(err.format_with(draft.names()));
            }
        }

        Ok(draft)
    }

    /// Freeze the draft into an immutable [`Environment`].
    ///
    /// Runs the identity-level validation that cannot be enforced at
    /// insertion time: cross-decl type-param consistency for open
    /// classes and modules. After `build`, the returned value has
    /// `primary_decl` selected, `sources` boxed, per-name entries
    /// converted to their frozen variants, and every reference-position
    /// [`TypeName`] in the AST resolved to its absolute form via
    /// [`Environment::names`].
    ///
    /// Phase 4d Stage 3 adds rbs `resolve_type_names` semantics: every
    /// reference-position type-name slot in the AST (super_class /
    /// mixin / self_types / type_param bounds + defaults / method types
    /// / attr / type-alias body / constant / global / alias old_name)
    /// is rewritten to its absolute resolved form against the draft's
    /// declared name set. Nested `class` / `module` / `interface`
    /// declarations are flattened: the outer decl's `members` keeps the
    /// nested `Arc`, and an additional top-level entry is registered
    /// under the qualified name.
    // `NameTable` carries the pre-interned `BuiltinNames` (~24 `TypeName`s)
    // and a `Rodeo` buffer, pushing the `Err` variant past the lint's
    // 128-byte threshold. Boxing would force every caller into a two-step
    // deref pattern; the `Err` path is rare (only on environment build
    // failure) so eat the local `allow` instead.
    #[allow(clippy::result_large_err)]
    pub fn build(mut self) -> Result<Environment, (BuildError, NameTable)> {
        let g_backend = self.g.clone();
        // Taken (not cloned) up front: Pass 1 moves the carried resolver
        // table out of it, Pass 2 seeds its six decl maps by destructuring
        // the rest, so ownership needs to outlive both steps but nothing
        // here may be duplicated (ADR-0028 Decision 1's O(delta) goal —
        // cloning the whole overlay here would silently reintroduce an
        // O(env) copy).
        let mut overlay = self.overlay.take();
        let names: NameTable = self.names;

        // Wrap the body in an immediately-invoked closure so `?` can
        // continue to propagate `BuildError` internally, while the outer
        // function returns the table alongside it on `Err`. The closure
        // captures `&names` by reference (every helper takes `&NameTable`);
        // the `Ok` path uses `NameTable::default()` as a placeholder and
        // we stitch the real table back on after the closure returns.
        // See `BuildError::format_with` for why the table needs to survive
        // the `Err` arm.
        let body_result = (|| -> Result<Environment, BuildError> {
            // ---- Pass 1: collect all_names + aliases for the resolver ----
            //
            // Mirrors rbs `Resolver::TypeNameResolver.build`: all_names holds
            // real declared names (class / module / interface / type-alias /
            // constant), aliases holds class_alias_decls keyed by alias
            // new_name with (raw old_name, declaration context). Top-level
            // class_alias_decls land here directly; nested aliases are picked
            // up by collect_*_names while walking the member tree.
            //
            // ADR-0028 slice S2: with an overlay attached, the starting
            // point is the carried resolver table — already delta-updated
            // by `Environment::unload` — instead of an empty set. This is
            // what keeps this pass O(inserted) rather than O(env): the
            // loops below only walk `self.class_decls` etc. (the fresh
            // draft insertions), never the overlay's surviving decls.
            // `mem::take` leaves the overlay's copy empty so the later
            // `overlay.map(|o| *o).unwrap_or_default()` destructure (Pass
            // 2) doesn't see stale data.
            // ADR-0028 F6 remainder: `take_materialized` folds the
            // carried two-layer states back into owned collections —
            // O(env) on this change-run-only path, the same rows the
            // warm open used to decode unconditionally before F6.
            let (mut all_names, mut aliases): (
                FxHashSet<TypeName>,
                FxHashMap<TypeName, (String, Context)>,
            ) = match overlay.as_deref_mut() {
                Some(o) => (
                    std::mem::take(&mut o.all_names),
                    std::mem::take(&mut o.aliases),
                ),
                None => (FxHashSet::default(), FxHashMap::default()),
            };
            for entry in self.class_decls.values() {
                match entry {
                    ClassOrModuleDraft::Class(c) => {
                        for (_file, ctx, decl) in &c.context_decls {
                            match decl {
                                ClassDeclarationDraft::Signature(s) => {
                                    resolution::collect_class_names(
                                        s,
                                        &names,
                                        &mut all_names,
                                        &mut aliases,
                                        ctx,
                                    );
                                }
                                ClassDeclarationDraft::Ruby(r) => {
                                    resolution::collect_ruby_class_names(
                                        r,
                                        &names,
                                        &mut all_names,
                                        &mut aliases,
                                        ctx,
                                    );
                                }
                            }
                        }
                    }
                    ClassOrModuleDraft::Module(m) => {
                        for (_file, ctx, decl) in &m.context_decls {
                            match decl {
                                ModuleDeclarationDraft::Signature(s) => {
                                    resolution::collect_module_names(
                                        s,
                                        &names,
                                        &mut all_names,
                                        &mut aliases,
                                        ctx,
                                    );
                                }
                                ModuleDeclarationDraft::Ruby(r) => {
                                    resolution::collect_ruby_module_names(
                                        r,
                                        &names,
                                        &mut all_names,
                                        &mut aliases,
                                        ctx,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            for entry in self.interface_decls.values() {
                resolution::collect_interface_names(&entry.decl, &names, &mut all_names);
            }
            for entry in self.type_alias_decls.values() {
                all_names.insert(entry.name);
            }
            // `constant_decls` deliberately does not feed `all_names`. Mirrors the
            // rbs split between `TypeNameResolver` (type-name space) and
            // `Resolver::ConstantResolver` (constant-name space): a constant
            // (`FOO: Integer`) is a value declaration, not a type, so surfacing
            // it as a resolvable type name would let `try_resolve("FOO", Class)`
            // succeed for a name that has no type interpretation. The frozen
            // `Environment::constant_decls` still owns the data; the
            // ConstantResolver port consumes it independently.
            //
            // Top-level class_alias_decls go to `aliases` (rbs parity); the
            // alias new_name is *not* added to all_names — class_alias is a
            // name-substitution directive, not a real type.
            for entry in self.class_alias_decls.values() {
                let old_raw = entry.decl.old_name_raw(&names);
                aliases.insert(entry.name, (old_raw, Arc::clone(&entry.context)));
            }

            // G-snapshot fallback (ADR-0028 slice 2a-2): a combined draft
            // would have walked the gem declarations through the loops
            // above; the graft supplies the same sets from the snapshot's
            // key partition (declared classes / modules / interfaces /
            // type aliases — constants stay excluded, see the comment
            // above) and its eagerly decoded alias entries. The gem AST
            // itself is already resolved and is not re-resolved here.
            if let Some(g) = &g_backend {
                all_names.extend(g.resolver_names());
                for (name, entry) in g.alias_entries() {
                    let (old_name, context) = match entry {
                        ClassOrModuleAliasEntry::Class(e) => match &e.decl {
                            ClassAliasDeclaration::Signature(s) => (s.old_name, e.context()),
                            ClassAliasDeclaration::Ruby(_) => {
                                unreachable!("gem snapshots hold signature decls only")
                            }
                        },
                        ClassOrModuleAliasEntry::Module(e) => match &e.decl {
                            ModuleAliasDeclaration::Signature(s) => (s.old_name, e.context()),
                            ModuleAliasDeclaration::Ruby(_) => {
                                unreachable!("gem snapshots hold signature decls only")
                            }
                        },
                    };
                    aliases.insert(*name, (names.resolve(old_name), Arc::clone(context)));
                }
            }

            // No overlay fallback here (ADR-0028 slice S2 removed it): the
            // carried resolver table seeded at the top of Pass 1 already
            // holds every name the overlay's surviving decls declare —
            // `Environment::unload` delta-updated it, so a freshly
            // inserted decl referencing an untouched overlay name (e.g. a
            // reopened file's `super_class` pointing at a class nobody
            // unloaded) resolves against that carried entry without this
            // pass re-walking the overlay's maps.

            // ---- Build use-directive resolution state ----
            //
            // Mirrors rbs's `resolve_type_names`:
            // - one `UseMap::Table` keyed by every declared TypeName,
            //   precomputed children for wildcard lookup
            // - one `UseMap` per source file (`file_use_maps`) plus the
            //   file-less inline-RBS fallback (`default_use_map`), each
            //   populated by walking that source's `use` clauses
            //
            // `Table::known_types`/`children` are only read from
            // `UseMap::build_map`'s wildcard arm (`use Foo::*`) — a single
            // clause (`use Foo` / `use Foo as Bar`) never touches the
            // table. When this draft has no `use` directive at all, the
            // O(all_names) clone + `compute_children` walk below is pure
            // waste (ADR-0028 slice S2): every inline-Ruby-only warm run
            // hits this, since `.rb` files never call `add_directives`.
            // Draft sessions that *do* carry a `use` directive (any file
            // with directives, or the file-less default path some tests
            // use) still pay the full O(all_names) cost on every build,
            // overlay or not — a known remaining debt, not addressed here.
            let needs_use_table = !self.file_directives.is_empty()
                || self
                    .default_directives
                    .iter()
                    .any(|d| matches!(d, Directive::Use(_)));
            let use_table = if needs_use_table {
                let mut t = crate::environment::use_map::Table::new();
                t.known_types = all_names.clone();
                t.compute_children(&names);
                Arc::new(t)
            } else {
                Arc::new(crate::environment::use_map::Table::new())
            };
            let file_use_maps: FxHashMap<Name, UseMap> = self
                .file_directives
                .iter()
                .map(|(file_name, directives)| {
                    let mut map = UseMap::new(Arc::clone(&use_table));
                    for dir in directives {
                        if let Directive::Use(use_dir) = dir {
                            for clause in &use_dir.clauses {
                                map.build_map(clause, &names);
                            }
                        }
                    }
                    (*file_name, map)
                })
                .collect();
            let default_use_map: UseMap = {
                let mut map = UseMap::new(Arc::clone(&use_table));
                for dir in &self.default_directives {
                    if let Directive::Use(use_dir) = dir {
                        for clause in &use_dir.clauses {
                            map.build_map(clause, &names);
                        }
                    }
                }
                map
            };
            // Snapshot the resolve-type-names flags before `self` is
            // destructured below; the resolver-selector closure consults
            // them per-file.
            let file_resolve_type_names = self.file_resolve_type_names.clone();
            let default_resolve_type_names = self.default_resolve_type_names;

            // Helper: pick the right `UseMap` for a declaration based on
            // its source location. `None` source falls back to the
            // file-less default; an unknown `Name` (no directives in that
            // file) falls back to the default too.
            let use_map_for = |file: Option<Name>| -> &UseMap {
                match file {
                    Some(name) => file_use_maps.get(&name).unwrap_or(&default_use_map),
                    None => &default_use_map,
                }
            };

            let resolver = TypeNameResolver::new(&all_names, &aliases, &names);

            // Helper: pick the right `TypeNameResolver` for a declaration
            // based on its source file. When the file (or the file-less
            // default) opted into `# resolve-type-names: false`, the
            // returned resolver is in bypass mode so every `try_resolve*`
            // call returns `None` and the raw AST passes through unchanged.
            let resolver_for = |file: Option<Name>| -> TypeNameResolver<'_> {
                let resolve = match file {
                    Some(f) => file_resolve_type_names.get(&f).copied().unwrap_or(true),
                    None => default_resolve_type_names,
                };
                if resolve {
                    resolver.with_use_map(Some(use_map_for(file)))
                } else {
                    resolver.with_bypass()
                }
            };

            // ---- Pass 2: build frozen entries with resolved AST ----
            //
            // Every output map is seeded from the overlay (ADR-0028 slice
            // S1b) before any draft entry is resolved into it. Overlay
            // entries are never re-resolved (rbs `resolve_type_names(only:)`
            // pass-through semantics); the seed just makes them visible to
            // the merge / duplicate-detection paths below exactly as if
            // they were inserted earlier in the same build:
            // - open-class reopens (`class_decls.insert(resolved_key, ...)`
            //   below, and `merge_nested_class`/`merge_nested_module` in the
            //   flatten loop) find the overlay entry already occupying the
            //   key and merge into it via the existing `Entry::Occupied` arms.
            // - single-decl kinds (interface / type_alias / constant / class
            //   alias) find the key already occupied and their existing
            //   duplicate checks (`insert_single_or_duplicate` /
            //   `insert_alias_or_duplicate`) reject the collision, with no
            //   new overlay-specific code needed in the flatten loop.
            let FrozenOverlay {
                mut class_decls,
                mut interface_decls,
                mut class_alias_decls,
                mut type_alias_decls,
                mut constant_decls,
                mut global_decls,
                g_remerge: overlay_g_remerge,
                // Already moved out at the top of Pass 1 (`mem::take`);
                // both fields are empty defaults by this point.
                all_names: _,
                aliases: _,
            } = overlay.map(|o| *o).unwrap_or_default();
            // Names whose G decl was already merged into `class_decls` by a
            // prior `build()` and survived `Environment::unload` untouched
            // (the ADR-0028 slice S1c double-merge guard below). Captured
            // here, before any fresh A-layer entry merges into `class_decls`
            // (Pass 0/1 declarations, infusion synthesis) — those carry
            // `DeclOrigin::Synthesized` rather than `DeclOrigin::GSnapshot`
            // (e.g. `active_record_synthesis`'s synthesized model/relation
            // decls), so `has_g_origin_decl` no longer confuses the two
            // kinds even checked against the live post-merge entry; the
            // pre-merge snapshot is kept anyway to avoid coupling this
            // guard's correctness to merge order.
            let overlay_g_origin: FxHashSet<TypeName> = class_decls
                .iter()
                .filter(|(_, entry)| has_g_origin_decl(entry))
                .map(|(name, _)| *name)
                .collect();

            let mut nested = FlattenedDecls::default();

            // Top-level class/module entries. Each draft (Context, Arc<Class>)
            // is rewritten to a resolved Arc<Class>; nested class/module/interface
            // members are pulled out into `nested` for later flat-map insertion.
            for (raw_name, entry) in self.class_decls {
                let resolved_key = raw_name;
                let frozen = match entry {
                    ClassOrModuleDraft::Class(draft) => {
                        let resolved_decls: Vec<(DeclOrigin, Context, ClassDeclarationDraft)> =
                            draft
                                .context_decls
                                .iter()
                                .map(|(file, ctx, decl)| {
                                    let resolved = match decl {
                                        ClassDeclarationDraft::Signature(s) => {
                                            let r = resolver_for(file.file());
                                            ClassDeclarationDraft::Signature(
                                                resolution::resolve_class_recursive(
                                                    s,
                                                    ctx,
                                                    &r,
                                                    &names,
                                                    &mut nested,
                                                    *file,
                                                ),
                                            )
                                        }
                                        ClassDeclarationDraft::Ruby(r_decl) => {
                                            // Ruby files have no `use` directive; pass
                                            // the file-less default so any Ruby-only
                                            // file resolves with no aliases.
                                            let r = resolver.with_use_map(Some(&default_use_map));
                                            ClassDeclarationDraft::Ruby(
                                                resolution::resolve_ruby_class_recursive(
                                                    r_decl,
                                                    ctx,
                                                    &r,
                                                    &names,
                                                    &mut nested,
                                                    *file,
                                                ),
                                            )
                                        }
                                    };
                                    (*file, Arc::clone(ctx), resolved)
                                })
                                .collect();
                        let resolved_draft = ClassEntryDraft {
                            name: draft.name,
                            context_decls: resolved_decls,
                        };
                        ClassOrModule::Class(build_class_entry(resolved_key, resolved_draft)?)
                    }
                    ClassOrModuleDraft::Module(draft) => {
                        let resolved_decls: Vec<(DeclOrigin, Context, ModuleDeclarationDraft)> =
                            draft
                                .context_decls
                                .iter()
                                .map(|(file, ctx, decl)| {
                                    let resolved = match decl {
                                        ModuleDeclarationDraft::Signature(s) => {
                                            let r = resolver_for(file.file());
                                            ModuleDeclarationDraft::Signature(
                                                resolution::resolve_module_recursive(
                                                    s,
                                                    ctx,
                                                    &r,
                                                    &names,
                                                    &mut nested,
                                                    *file,
                                                ),
                                            )
                                        }
                                        ModuleDeclarationDraft::Ruby(r_decl) => {
                                            let r = resolver.with_use_map(Some(&default_use_map));
                                            ModuleDeclarationDraft::Ruby(
                                                resolution::resolve_ruby_module_recursive(
                                                    r_decl,
                                                    ctx,
                                                    &r,
                                                    &names,
                                                    &mut nested,
                                                    *file,
                                                ),
                                            )
                                        }
                                    };
                                    (*file, Arc::clone(ctx), resolved)
                                })
                                .collect();
                        let resolved_draft = ModuleEntryDraft {
                            name: draft.name,
                            context_decls: resolved_decls,
                        };
                        ClassOrModule::Module(build_module_entry(resolved_key, resolved_draft)?)
                    }
                };
                // If the overlay seeded this key (an open-class reopen of a
                // retained decl), merge rather than overwrite — same
                // ordering and validation as a G-snapshot reopen
                // (`merge_g_class_or_module`'s doc comment): overlay-side
                // decls first, freshly drafted decls after.
                match class_decls.remove(&resolved_key) {
                    Some(existing) => {
                        let merged = merge_g_class_or_module(&resolved_key, &existing, frozen)?;
                        class_decls.insert(resolved_key, merged);
                    }
                    None => {
                        class_decls.insert(resolved_key, frozen);
                    }
                }
            }

            for (raw_name, draft) in self.interface_decls {
                let resolved = raw_name;
                let r = resolver_for(draft.file.file());
                let resolved_decl = resolution::resolve_interface_recursive(
                    &draft.decl,
                    &draft.context,
                    &r,
                    &names,
                );
                interface_decls.insert(
                    resolved,
                    InterfaceEntry {
                        name: resolved,
                        file: draft.file,
                        context: draft.context,
                        decl: resolved_decl,
                    },
                );
            }

            // Flatten nested class/module/interface decls. Same qualified name
            // appearing in two different parents (or via open class) merges
            // into a single entry; rbs's flatten map allows this. Inline
            // Ruby decls flatten the same way — the absolute `class_name` /
            // `module_name` has been pre-built by the inline collector, so
            // the flat entry's key comes straight from there.
            for (key, ctx, decl, file) in nested.classes {
                merge_nested_class(
                    &mut class_decls,
                    key,
                    file,
                    ctx,
                    ClassDeclaration::Signature(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassOrModule(key));
            }
            for (key, ctx, decl, file) in nested.modules {
                merge_nested_module(
                    &mut class_decls,
                    key,
                    file,
                    ctx,
                    ModuleDeclaration::Signature(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassOrModule(key));
            }
            for (key, ctx, decl, file) in nested.ruby_classes {
                merge_nested_class(
                    &mut class_decls,
                    key,
                    file,
                    ctx,
                    ClassDeclaration::Ruby(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassOrModule(key));
            }
            for (key, ctx, decl, file) in nested.ruby_modules {
                merge_nested_module(
                    &mut class_decls,
                    key,
                    file,
                    ctx,
                    ModuleDeclaration::Ruby(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassOrModule(key));
            }
            for (key, ctx, decl, file) in nested.interfaces {
                merge_nested_interface(&mut interface_decls, key, file, ctx, decl);
                record_path_index(&mut self.path_index, file, PathIndexKey::Interface(key));
            }

            // Per-name merge of reopened gem entries (ADR-0028 slice
            // 2a-2). Every A-layer class/module name also present in the
            // snapshot decodes its G entry once and rebuilds the combined
            // entry — re-running the primary-decl selection and the
            // type-params consistency check across both layers, exactly
            // like a combined draft build. Names only in G stay lazy.
            //
            // Double-merge guard (ADR-0028 slice S1c): an overlay-seeded
            // entry that survived `Environment::unload` untouched already
            // carries its G-origin (`DeclOrigin::GSnapshot`) decl(s) from a
            // prior build — merging again would duplicate them.
            // `needs_g_merge` skips those via `overlay_g_origin`, captured
            // above from the overlay before any fresh A-layer entry could
            // merge in. `overlay_g_remerge` is the one exception: `unload`
            // explicitly stripped that entry's G-origin decl(s) (because it
            // removed the file that used to carry the merge) and flags it
            // here so the merge runs even though the entry is no longer in
            // `overlay_g_origin`.
            let mut g_overlap = GOverlap::default();
            if let Some(g) = &g_backend {
                let reopened: Vec<TypeName> = class_decls
                    .keys()
                    .filter(|k| g.class_contains(k))
                    .copied()
                    .collect();
                g_overlap.class = reopened.len();
                for name in reopened {
                    let needs_g_merge =
                        overlay_g_remerge.contains(&name) || !overlay_g_origin.contains(&name);
                    if !needs_g_merge {
                        continue;
                    }
                    let a_entry = class_decls.remove(&name).expect("key collected above");
                    let g_entry = g.class_entry(&name).expect("class_contains checked above");
                    let merged = merge_g_class_or_module(&name, g_entry, a_entry)?;
                    class_decls.insert(name, merged);
                }
                // Interfaces are single-decl; a flatten-generated A
                // interface shadows the G one (`merge_nested_interface`'s
                // most-recent-wins), so the layered view just counts the
                // overlap for `len` and lets the A map win on lookup.
                g_overlap.interface = interface_decls
                    .keys()
                    .filter(|k| g.interface_contains(k))
                    .count();
            }

            // Extended (not rebuilt) so the overlay seed above survives:
            // a colliding key here would mean an `insert_class_alias` /
            // `insert_module_alias` / `insert_ruby_alias` call slipped past
            // the overlay duplicate check in `check_constant_namespace_free`.
            class_alias_decls.extend(self.class_alias_decls.into_iter().map(
                |(raw_name, draft)| {
                    let resolved = raw_name;
                    let resolved_decl = match draft.decl {
                        ClassAliasDraft::Class(c) => {
                            let r = resolver_for(draft.file.file());
                            ClassAliasDraft::Class(Arc::new(resolution::resolve_class_alias_decl(
                                &c,
                                &draft.context,
                                &r,
                                &names,
                            )))
                        }
                        ClassAliasDraft::Module(m) => {
                            let r = resolver_for(draft.file.file());
                            ClassAliasDraft::Module(Arc::new(
                                resolution::resolve_module_alias_decl(
                                    &m,
                                    &draft.context,
                                    &r,
                                    &names,
                                ),
                            ))
                        }
                        // Mirror rbs `Environment#resolve_ruby_decl`
                        // (environment.rb:771-783): absolutize the alias rhs
                        // against the enclosing context so a sibling reference
                        // inside `module C; X = ...; Y = X #: class-alias; end`
                        // resolves to `::C::X` rather than the top-level
                        // fallback `::X`. Same helper covers nested decls in
                        // resolution::resolve_ruby_member.
                        ClassAliasDraft::Ruby(d) => {
                            // Ruby files have no `use` directive.
                            let r = resolver.with_use_map(Some(&default_use_map));
                            ClassAliasDraft::Ruby(Arc::new(resolution::resolve_ruby_alias_decl(
                                &d,
                                &draft.context,
                                &r,
                                &names,
                            )))
                        }
                    };
                    let resolved_draft = ClassAliasEntry {
                        name: draft.name,
                        file: draft.file,
                        context: draft.context,
                        decl: resolved_decl,
                    };
                    (resolved, build_class_alias_entry(resolved, resolved_draft))
                },
            ));

            // Same overlay-seeded extend as `class_alias_decls` above.
            type_alias_decls.extend(self.type_alias_decls.into_iter().map(|(raw_name, draft)| {
                let resolved = raw_name;
                let r = resolver_for(draft.file.file());
                let resolved_decl = Arc::new(resolution::resolve_type_alias_decl(
                    &draft.decl,
                    &draft.context,
                    &r,
                    &names,
                ));
                (
                    resolved,
                    SingleEntry {
                        name: resolved,
                        file: draft.file,
                        context: draft.context,
                        decl: resolved_decl,
                    },
                )
            }));

            // Same overlay-seeded extend as `class_alias_decls` above.
            constant_decls.extend(self.constant_decls.into_iter().map(|(raw_name, draft)| {
                let resolved = raw_name;
                let r = resolver_for(draft.file.file());
                let resolved_decl = Arc::new(resolution::resolve_constant_decl(
                    &draft.decl,
                    &draft.context,
                    &r,
                    &names,
                ));
                (
                    resolved,
                    SingleEntry {
                        name: resolved,
                        file: draft.file,
                        context: draft.context,
                        decl: resolved_decl,
                    },
                )
            }));

            // Flatten nested type-alias / constant / class-alias / module-alias
            // decls discovered during the recursive walk. These are single-decl
            // per name; collision against an existing top-level entry (or
            // between two flatten paths) is a duplicate.
            // Cross-layer arm of the single-decl collision rule: a
            // combined draft would have found the gem entry in the local
            // map, so the graft consults the snapshot key sets alongside.
            let g_type_alias_taken = |key: &TypeName| {
                g_backend
                    .as_ref()
                    .is_some_and(|g| g.type_alias_contains(key))
            };
            let g_constant_taken =
                |key: &TypeName| g_backend.as_ref().is_some_and(|g| g.constant_contains(key));
            let g_alias_taken =
                |key: &TypeName| g_backend.as_ref().is_some_and(|g| g.alias_contains(key));

            for (key, ctx, decl, file) in nested.type_aliases {
                if g_type_alias_taken(&key) {
                    return Err(duplicate_decl_error(&key, &names));
                }
                insert_single_or_duplicate(&mut type_alias_decls, key, file, ctx, decl, &names)?;
                record_path_index(&mut self.path_index, file, PathIndexKey::TypeAlias(key));
            }
            for (key, ctx, decl, file) in nested.constants {
                if g_constant_taken(&key) {
                    return Err(duplicate_decl_error(&key, &names));
                }
                insert_single_or_duplicate(&mut constant_decls, key, file, ctx, decl, &names)?;
                record_path_index(&mut self.path_index, file, PathIndexKey::Constant(key));
            }
            for (key, ctx, decl, file) in nested.class_aliases {
                if g_alias_taken(&key) {
                    return Err(duplicate_decl_error(&key, &names));
                }
                insert_alias_or_duplicate(
                    &mut class_alias_decls,
                    key,
                    file,
                    ctx,
                    ClassAliasDraft::Class(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassAlias(key));
            }
            for (key, ctx, decl, file) in nested.module_aliases {
                if g_alias_taken(&key) {
                    return Err(duplicate_decl_error(&key, &names));
                }
                insert_alias_or_duplicate(
                    &mut class_alias_decls,
                    key,
                    file,
                    ctx,
                    ClassAliasDraft::Module(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassAlias(key));
            }
            for (key, ctx, decl, file) in nested.ruby_aliases {
                if g_alias_taken(&key) {
                    return Err(duplicate_decl_error(&key, &names));
                }
                insert_alias_or_duplicate(
                    &mut class_alias_decls,
                    key,
                    file,
                    ctx,
                    ClassAliasDraft::Ruby(decl),
                    &names,
                )?;
                record_path_index(&mut self.path_index, file, PathIndexKey::ClassAlias(key));
            }

            // Same overlay-seeded extend as `class_alias_decls` above.
            global_decls.extend(self.global_decls.into_iter().map(|(name, entry)| {
                let r = resolver_for(entry.file.file());
                let resolved_decl = Arc::new(resolution::resolve_global_decl(
                    &entry.decl,
                    &entry.context,
                    &r,
                    &names,
                ));
                (
                    name,
                    GlobalEntry {
                        name: entry.name,
                        file: entry.file,
                        context: entry.context,
                        decl: resolved_decl,
                    },
                )
            }));

            // Merge the eagerly decoded gem aliases into the frozen map:
            // `class_alias_decls` (and the normalize table below) stay
            // `Built` in the layered environment, so alias chains that
            // cross layers — and gem aliases whose dangling target the A
            // layer now declares — fold with plain map lookups.
            if let Some(g) = &g_backend {
                for (name, entry) in g.alias_entries() {
                    class_alias_decls
                        .entry(*name)
                        .or_insert_with(|| entry.clone());
                }
            }

            let normalized_module_names = precompute_normalized_module_names(
                &class_alias_decls,
                |name| {
                    class_decls.contains_key(name)
                        || g_backend.as_ref().is_some_and(|g| g.class_contains(name))
                },
                &names,
            );

            Ok(Environment {
                class_decls,
                interface_decls,
                class_alias_decls,
                type_alias_decls,
                constant_decls,
                global_decls,
                sources: self.sources.into_boxed_slice(),
                // Stitched to the real `names` after the closure returns; we
                // can't move from the borrowed `names` here.
                names: NameTable::default(),
                normalized_module_names,
                g: g_backend.clone(),
                g_overlap,
                path_index: self.path_index,
                all_names,
                aliases,
            })
        })();

        match body_result {
            Ok(mut env) => {
                env.names = names;
                Ok(env)
            }
            Err(e) => Err((e, names)),
        }
    }
}

/// Shared body of [`EnvironmentDraft::record_path`], factored into a free
/// function so `build`'s flatten loop (where `self.class_decls` etc. have
/// already been moved out field-by-field, ruling out a `&mut self` method
/// call on the partially-moved `self`) can record `path_index` entries for
/// nested decls without borrowing the whole draft.
fn record_path_index(path_index: &mut PathIndexState, file: DeclOrigin, key: PathIndexKey) {
    // `index_file`, not `file`: owner-tagged synthesized decls (ADR-0028
    // S2 relaxation) are indexed under their owner `.rb` file so a warm
    // `unload` of that file finds and strips them for re-synthesis.
    if let Some(file) = file.index_file() {
        path_index.add(file, key);
    }
}

/// Walk every `class_alias_decls` key once, following its chain into
/// `class_decls`. The result table only stores non-trivial outcomes — keys
/// that are not aliases get the `Normalized(self)` answer from
/// [`Environment::normalize_module_name_result`]'s table-miss fallback.
///
/// Cycles are detected with a per-walk visited set; rbs's reference impl
/// only follows one hop and would never loop, but crema folds whole chains
/// here so the consumer sees an `O(1)` answer post-build.
///
/// `pub(crate)` (not private) so the snapshot decode path
/// can recompute this table directly from the decoded `class_alias_decls` /
/// `class_decls` instead of persisting it on disk — recompute is O(aliases)
/// (one walk per `class_alias_decls` key), cheap enough that storing a
/// redundant copy in the snapshot would only add drift risk (ADR-0028 S6a-2).
pub(crate) fn precompute_normalized_module_names(
    class_alias_decls: &FxHashMap<TypeName, ClassOrModuleAliasEntry>,
    class_contains: impl Fn(&TypeName) -> bool,
    names: &NameTable,
) -> FxHashMap<TypeName, NormalizeModuleNameResult> {
    let mut table =
        FxHashMap::with_capacity_and_hasher(class_alias_decls.len(), Default::default());
    for original in class_alias_decls.keys() {
        let mut visited: FxHashSet<TypeName> = FxHashSet::default();
        let mut current = *original;
        let result = loop {
            if !visited.insert(current) {
                break NormalizeModuleNameResult::Cycle {
                    original: *original,
                };
            }
            match class_alias_decls.get(&current) {
                Some(entry) => match alias_target_name(entry, names) {
                    Some(next) => current = next,
                    None => {
                        break NormalizeModuleNameResult::UnknownTarget {
                            original: *original,
                            target: current,
                        };
                    }
                },
                None => {
                    if class_contains(&current) {
                        break NormalizeModuleNameResult::Normalized(current);
                    } else {
                        break NormalizeModuleNameResult::UnknownTarget {
                            original: *original,
                            target: current,
                        };
                    }
                }
            }
        };
        table.insert(*original, result);
    }
    table
}

/// Extract the alias target as a [`TypeName`]. Returns `None` only when
/// the Ruby alias decl has neither inferred nor explicit `old_name` text
/// — the precompute treats this as `UnknownTarget` so future
/// Ruby-alias wiring trips a real test rather than silently
/// identity-folding.
///
/// Re-derives the alias target through `decl.old_name(names)` because
/// Phase 4d's `resolve_ruby_alias_decl` writes the resolved absolute
/// form back into `infered_old_name` (when no annotation is present)
/// — the precompute reads whichever side currently holds the target
/// and preserves its `namespace.absolute()` signal, so unresolved
/// relative rhs (e.g. `Sibling` inside `::Container`) remain relative
/// and surface in `UnknownTypeName` diagnostics with the name the user
/// wrote.
fn alias_target_name(entry: &ClassOrModuleAliasEntry, names: &NameTable) -> Option<TypeName> {
    let ruby_target = |decl: &RubyClassModuleAliasDecl| decl.old_name(names);
    match entry {
        ClassOrModuleAliasEntry::Class(e) => match &e.decl {
            ClassAliasDeclaration::Signature(s) => Some(s.old_name),
            ClassAliasDeclaration::Ruby(d) => ruby_target(d),
        },
        ClassOrModuleAliasEntry::Module(e) => match &e.decl {
            ModuleAliasDeclaration::Signature(s) => Some(s.old_name),
            ModuleAliasDeclaration::Ruby(d) => ruby_target(d),
        },
    }
}

/// Insert a flattened nested-class entry into the frozen class_decls map.
/// If the qualified name is already present (open class via merged
/// declarations or a same-name parent + nested), append to the existing
/// `context_decls`. `primary_decl` follows the usual selection rule.
/// Insert a flattened nested-class entry into the frozen `class_decls`
/// map. If the qualified name is already present (open-class merge),
/// append to the existing `context_decls`; otherwise create a new
/// `ClassEntry`. `primary_decl` follows rbs's "first decl with a
/// `super_class` wins; else the first decl" rule applied across
/// Signature *and* Ruby variants.
///
/// Returns `BuildError::DuplicatedDeclaration` when the existing entry
/// at the same name is a `Module` (the AST disallows class/module
/// collision within a single parent's `members`, but flatten sees both
/// the Outer-level entry and any same-named nested decl, so the check
/// stays here).
fn merge_nested_class(
    map: &mut FxHashMap<TypeName, ClassOrModule>,
    key: TypeName,
    file: DeclOrigin,
    context: Context,
    decl: ClassDeclaration,
    names: &NameTable,
) -> Result<(), BuildError> {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(v) => {
            v.insert(ClassOrModule::Class(ClassEntry {
                name: key,
                context_decls: Box::from([(file, context, decl.clone())]),
                primary_decl: decl,
            }));
            Ok(())
        }
        Entry::Occupied(mut o) => match o.get_mut() {
            ClassOrModule::Class(entry) => {
                let mut decls: Vec<(DeclOrigin, Context, ClassDeclaration)> =
                    entry.context_decls.to_vec();
                decls.push((file, context, decl));
                // Re-validate type_params across all decls (including the
                // newly merged one). Same rule as `build_class_entry`:
                // signature decls' params must agree under alpha-renaming;
                // Ruby decls have no decl-level params and are skipped.
                validate_type_params(
                    &key,
                    decls.iter().filter_map(|(_, _, d)| match d {
                        ClassDeclaration::Signature(s) => Some(s.type_params.as_slice()),
                        ClassDeclaration::Ruby(_) => None,
                    }),
                )?;
                let primary_index = decls
                    .iter()
                    .position(|(_, _, d)| class_decl_has_super(d))
                    .unwrap_or(0);
                entry.primary_decl = decls[primary_index].2.clone();
                entry.context_decls = decls.into_boxed_slice();
                Ok(())
            }
            ClassOrModule::Module(_) => Err(duplicate_decl_error(&key, names)),
        },
    }
}

fn merge_nested_module(
    map: &mut FxHashMap<TypeName, ClassOrModule>,
    key: TypeName,
    file: DeclOrigin,
    context: Context,
    decl: ModuleDeclaration,
    names: &NameTable,
) -> Result<(), BuildError> {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(v) => {
            v.insert(ClassOrModule::Module(ModuleEntry {
                name: key,
                context_decls: Box::from([(file, context, decl.clone())]),
                primary_decl: decl,
            }));
            Ok(())
        }
        Entry::Occupied(mut o) => match o.get_mut() {
            ClassOrModule::Module(entry) => {
                let mut decls: Vec<(DeclOrigin, Context, ModuleDeclaration)> =
                    entry.context_decls.to_vec();
                decls.push((file, context, decl));
                validate_type_params(
                    &key,
                    decls.iter().filter_map(|(_, _, d)| match d {
                        ModuleDeclaration::Signature(s) => Some(s.type_params.as_slice()),
                        ModuleDeclaration::Ruby(_) => None,
                    }),
                )?;
                entry.primary_decl = decls[0].2.clone();
                entry.context_decls = decls.into_boxed_slice();
                Ok(())
            }
            ClassOrModule::Class(_) => Err(duplicate_decl_error(&key, names)),
        },
    }
}

fn insert_single_or_duplicate<D>(
    map: &mut FxHashMap<TypeName, SingleEntry<D>>,
    key: TypeName,
    file: DeclOrigin,
    context: Context,
    decl: Arc<D>,
    names: &NameTable,
) -> Result<(), BuildError> {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(v) => {
            v.insert(SingleEntry {
                name: key,
                file,
                context,
                decl,
            });
            Ok(())
        }
        Entry::Occupied(_) => Err(duplicate_decl_error(&key, names)),
    }
}

fn insert_alias_or_duplicate(
    map: &mut FxHashMap<TypeName, ClassOrModuleAliasEntry>,
    key: TypeName,
    file: DeclOrigin,
    context: Context,
    decl: ClassAliasDraft,
    names: &NameTable,
) -> Result<(), BuildError> {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(v) => {
            let entry = ClassAliasEntry {
                name: key,
                file,
                context,
                decl,
            };
            v.insert(build_class_alias_entry(key, entry));
            Ok(())
        }
        Entry::Occupied(_) => Err(duplicate_decl_error(&key, names)),
    }
}

/// Whether `entry` already carries a G-origin decl — S1a's decode
/// invariant is that a G-origin decl always carries
/// [`DeclOrigin::GSnapshot`]. Used by [`EnvironmentDraft::build`] to
/// snapshot which overlay-seeded entries were already G-merged by a
/// prior build (ADR-0028 slice S1c), *before* any fresh A-layer entry
/// (Pass 0/1 declarations, infusion synthesis — e.g.
/// `active_record_synthesis`'s synthesized model/relation decls) merges
/// into the same map. Those synthetic decls carry [`DeclOrigin::Synthesized`]
/// instead — a distinct variant, so this check no longer needs to be
/// evaluated only against the overlay's pre-merge state to avoid
/// misclassifying them (kept anyway; see the call site's doc comment for
/// why the *timing* still matters for `overlay_g_remerge`).
fn has_g_origin_decl(entry: &ClassOrModule) -> bool {
    match entry {
        ClassOrModule::Class(c) => c
            .context_decls()
            .iter()
            .any(|(file, _, _)| matches!(file, DeclOrigin::GSnapshot)),
        ClassOrModule::Module(m) => m
            .context_decls()
            .iter()
            .any(|(file, _, _)| matches!(file, DeclOrigin::GSnapshot)),
    }
}

/// Merge a reopened gem entry with its A-layer counterpart (ADR-0028
/// slice 2a-2), reproducing the pair order a combined draft build
/// yields: top-level (context length 1) pairs land in insert order —
/// gems load before app sigs — while flatten-generated pairs are
/// appended after every top-level entry. Hence: G top-level, A
/// top-level, G flattened, A flattened. Getting this order right is
/// what keeps primary-decl selection (and the member walk order every
/// consumer sees) byte-identical with the snapshot-off path.
fn merge_g_class_or_module(
    key: &TypeName,
    g_entry: &ClassOrModule,
    a_entry: ClassOrModule,
) -> Result<ClassOrModule, BuildError> {
    let is_top_level = |c: &Context| c.len() == 1;
    match (g_entry, a_entry) {
        (ClassOrModule::Class(g), ClassOrModule::Class(a)) => {
            let (g1, g2): (Vec<_>, Vec<_>) = g
                .context_decls
                .iter()
                .cloned()
                .partition(|(_, c, _)| is_top_level(c));
            let (a1, a2): (Vec<_>, Vec<_>) = a
                .context_decls
                .into_vec()
                .into_iter()
                .partition(|(_, c, _)| is_top_level(c));
            let decls: Vec<(DeclOrigin, Context, ClassDeclaration)> =
                g1.into_iter().chain(a1).chain(g2).chain(a2).collect();
            validate_type_params(
                key,
                decls.iter().filter_map(|(_, _, d)| match d {
                    ClassDeclaration::Signature(s) => Some(s.type_params.as_slice()),
                    ClassDeclaration::Ruby(_) => None,
                }),
            )?;
            let primary_index = decls
                .iter()
                .position(|(_, _, d)| class_decl_has_super(d))
                .unwrap_or(0);
            Ok(ClassOrModule::Class(ClassEntry {
                name: *key,
                primary_decl: decls[primary_index].2.clone(),
                context_decls: decls.into_boxed_slice(),
            }))
        }
        (ClassOrModule::Module(g), ClassOrModule::Module(a)) => {
            let (g1, g2): (Vec<_>, Vec<_>) = g
                .context_decls
                .iter()
                .cloned()
                .partition(|(_, c, _)| is_top_level(c));
            let (a1, a2): (Vec<_>, Vec<_>) = a
                .context_decls
                .into_vec()
                .into_iter()
                .partition(|(_, c, _)| is_top_level(c));
            let decls: Vec<(DeclOrigin, Context, ModuleDeclaration)> =
                g1.into_iter().chain(a1).chain(g2).chain(a2).collect();
            validate_type_params(
                key,
                decls.iter().filter_map(|(_, _, d)| match d {
                    ModuleDeclaration::Signature(s) => Some(s.type_params.as_slice()),
                    ModuleDeclaration::Ruby(_) => None,
                }),
            )?;
            // Modules have no super_class, so rbs's primary selection
            // degenerates to "first decl" (`build_module_entry` does the
            // same) — the asymmetry with the Class arm is deliberate.
            Ok(ClassOrModule::Module(ModuleEntry {
                name: *key,
                primary_decl: decls[0].2.clone(),
                context_decls: decls.into_boxed_slice(),
            }))
        }
        // A flatten-generated nested decl of the opposite kind; the
        // top-level insert path already rejects the direct form.
        _ => Err(BuildError::DuplicatedDeclaration { name: *key }),
    }
}

pub(crate) fn class_decl_has_super(d: &ClassDeclaration) -> bool {
    match d {
        ClassDeclaration::Signature(s) => s.super_class.is_some(),
        ClassDeclaration::Ruby(r) => r.super_class.is_some(),
    }
}

fn duplicate_decl_error(key: &TypeName, _names: &NameTable) -> BuildError {
    BuildError::DuplicatedDeclaration { name: *key }
}

fn merge_nested_interface(
    map: &mut FxHashMap<TypeName, InterfaceEntry>,
    key: TypeName,
    file: DeclOrigin,
    context: Context,
    decl: Arc<Interface>,
) {
    // Interface is single-decl per name; if a flatten pass produces a
    // duplicate, the most recent one wins. Phase 4d does not validate
    // this — the duplicate-detection rule lives at insertion time on the
    // draft side, and nested interfaces have no insertion-time
    // collision check (they live inside a parent's members). Stage E /
    // Phase 5 may revisit.
    map.insert(
        key,
        InterfaceEntry {
            name: key,
            file,
            context,
            decl,
        },
    );
}

fn build_class_entry(name: TypeName, draft: ClassEntryDraft) -> Result<ClassEntry, BuildError> {
    let ClassEntryDraft {
        name: raw_name,
        context_decls,
    } = draft;

    debug_assert!(
        !context_decls.is_empty(),
        "ClassEntryDraft created without any decl"
    );

    // Type-param compatibility is an RBS-side identity invariant; only
    // signature decls participate. Inline Ruby decls have no
    // declaration-level type parameter list, so they're skipped here.
    validate_type_params(
        &raw_name,
        context_decls.iter().filter_map(|(_, _, d)| match d {
            ClassDeclarationDraft::Signature(s) => Some(s.type_params.as_slice()),
            ClassDeclarationDraft::Ruby(_) => None,
        }),
    )?;

    let primary_index = context_decls
        .iter()
        .position(|(_, _, decl)| match decl {
            ClassDeclarationDraft::Signature(s) => s.super_class.is_some(),
            ClassDeclarationDraft::Ruby(r) => r.super_class.is_some(),
        })
        .unwrap_or(0);
    let primary_decl = class_draft_to_frozen(&context_decls[primary_index].2);

    let context_decls = context_decls
        .into_iter()
        .map(|(file, ctx, decl)| (file, ctx, class_draft_to_frozen(&decl)))
        .collect();

    Ok(ClassEntry {
        name,
        context_decls,
        primary_decl,
    })
}

fn class_draft_to_frozen(draft: &ClassDeclarationDraft) -> ClassDeclaration {
    match draft {
        ClassDeclarationDraft::Signature(s) => ClassDeclaration::Signature(Arc::clone(s)),
        ClassDeclarationDraft::Ruby(r) => ClassDeclaration::Ruby(Arc::clone(r)),
    }
}

fn module_draft_to_frozen(draft: &ModuleDeclarationDraft) -> ModuleDeclaration {
    match draft {
        ModuleDeclarationDraft::Signature(s) => ModuleDeclaration::Signature(Arc::clone(s)),
        ModuleDeclarationDraft::Ruby(r) => ModuleDeclaration::Ruby(Arc::clone(r)),
    }
}

fn build_module_entry(name: TypeName, draft: ModuleEntryDraft) -> Result<ModuleEntry, BuildError> {
    let ModuleEntryDraft {
        name: raw_name,
        context_decls,
    } = draft;

    debug_assert!(
        !context_decls.is_empty(),
        "ModuleEntryDraft created without any decl"
    );

    validate_type_params(
        &raw_name,
        context_decls.iter().filter_map(|(_, _, d)| match d {
            ModuleDeclarationDraft::Signature(s) => Some(s.type_params.as_slice()),
            ModuleDeclarationDraft::Ruby(_) => None,
        }),
    )?;

    let primary_decl = module_draft_to_frozen(&context_decls[0].2);

    let context_decls = context_decls
        .into_iter()
        .map(|(file, ctx, decl)| (file, ctx, module_draft_to_frozen(&decl)))
        .collect();

    Ok(ModuleEntry {
        name,
        context_decls,
        primary_decl,
    })
}

fn build_class_alias_entry(name: TypeName, draft: ClassAliasEntry) -> ClassOrModuleAliasEntry {
    let ClassAliasEntry {
        name: _raw_name,
        file,
        context,
        decl,
    } = draft;
    match decl {
        ClassAliasDraft::Class(decl) => ClassOrModuleAliasEntry::Class(FrozenClassAliasEntry {
            name,
            file,
            context,
            decl: ClassAliasDeclaration::Signature(decl),
        }),
        ClassAliasDraft::Module(decl) => ClassOrModuleAliasEntry::Module(FrozenModuleAliasEntry {
            name,
            file,
            context,
            decl: ModuleAliasDeclaration::Signature(decl),
        }),
        ClassAliasDraft::Ruby(decl) => match decl.annotation.kind() {
            InlineAliasKind::Class => ClassOrModuleAliasEntry::Class(FrozenClassAliasEntry {
                name,
                file,
                context,
                decl: ClassAliasDeclaration::Ruby(decl),
            }),
            InlineAliasKind::Module => ClassOrModuleAliasEntry::Module(FrozenModuleAliasEntry {
                name,
                file,
                context,
                decl: ModuleAliasDeclaration::Ruby(decl),
            }),
        },
    }
}

fn validate_type_params<'a>(
    name: &TypeName,
    mut params: impl Iterator<Item = &'a [TypeParam]>,
) -> Result<(), BuildError> {
    let Some(first) = params.next() else {
        return Ok(());
    };
    for other in params {
        if !type_params_match(first, other) {
            return Err(BuildError::GenericParameterMismatch { name: *name });
        }
    }
    Ok(())
}

/// Mirrors rbs's `validate_type_params` literal-equality semantics.
///
/// rbs's `AST::TypeParam.rename` uses an identity substitution
/// (`{new_names[i] -> Var(new_names[i])}`), so type variable
/// references inside bounds and defaults pass through unchanged.
/// What rbs effectively compares is param count plus per-index
/// variance / unchecked / upper_bound / lower_bound / default_type
/// equality; the param's outer name is replaced by `rename` and so is
/// trivially equal. The `Type` AST derives `PartialEq`, so the
/// per-field check collapses to a few `==` comparisons.
///
/// Known divergence from rbs: rbs's `Types::*#==` ignores `location`,
/// but crema's `Type::PartialEq` is field-by-field including the
/// `location` field. Today this is dormant — every
/// `ast_builder::build_type` site leaves `location = None`, so
/// RBS-loaded bounds always compare equal on structure alone. If
/// `build_type` ever starts attaching locations, `type_params_match`
/// will need a location-blind comparison.
fn type_params_match(first: &[TypeParam], other: &[TypeParam]) -> bool {
    if first.len() != other.len() {
        return false;
    }
    first.iter().zip(other.iter()).all(|(a, b)| {
        a.variance == b.variance
            && a.unchecked == b.unchecked
            && a.upper_bound == b.upper_bound
            && a.lower_bound == b.lower_bound
            && a.default_type == b.default_type
    })
}
