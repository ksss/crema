//! Frozen environment — the immutable post-build form of
//! [`crate::environment::draft::EnvironmentDraft`].
//!
//! Mirrors `RBS::Environment` (see ADR-0017).
//! Per-name entries hold both `Signature(Arc<ast::declarations::*>)` and
//! `Ruby(Arc<ast::ruby::declarations::*>)` decls side by side; inline
//! declarations are *not* lowered into RBS declaration AST.
//!
//! `instance_methods` / `singleton_methods` / `superclasses` / `includes` /
//! `prepends` / `extends` live on [`crate::definition_builder::DefinitionBuilder`],
//! not here.

use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Arc;

use crate::snapshot::backend::GSnapshotBackend;

use crate::ast::declarations::{
    ClassAliasDeclaration as ClassAlias, ClassDeclaration as Class,
    ConstantDeclaration as Constant, InterfaceDeclaration as Interface,
    ModuleAliasDeclaration as ModuleAlias, ModuleDeclaration as Module, ModuleSelf as SelfType,
    TypeAliasDeclaration as TypeAlias,
};
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::declarations::{
    ClassDecl as RubyClassDecl, ClassModuleAliasDecl as RubyClassModuleAliasDecl,
    ModuleDecl as RubyModuleDecl,
};
use crate::ast::types::{
    AliasType, BaseType, BlockType, ClassInstanceType, ClassSingletonType, Function, FunctionParam,
    FunctionType, InterfaceType, IntersectionType, KeywordParam, LiteralType, OptionalType,
    ProcType, RecordField, RecordType, TupleType, Type, UnionType, VariableType,
};
use crate::environment::DeclOrigin;

use crate::environment::draft::{
    Context, EnvironmentDraft, FrozenOverlay, GlobalEntry, PathIndexKey, SingleEntry,
    class_decl_has_super,
};
use crate::environment::extras_state::PathIndexState;
use crate::location::{LocationRange, RubyLocation, SourceLocation};
use crate::name::{Name, NameTable, Symbol};
use crate::type_name::TypeName;

/// Outcome of looking up `class A = B` style alias chains.
///
/// Mirrors RBS's `normalize_module_name?: (TypeName) -> (TypeName | nil | false)`
/// (see `lib/rbs/environment.rb` and `sig/environment.rbs`) but encodes the
/// three nil/false reasons explicitly so callers can react without having to
/// re-do the lookup. `Normalized(name)` is the "happy path" — the chain
/// terminates at a real class/module.
///
/// Used by [`Environment::normalize_module_name_result`]; the convenience
/// [`Environment::normalize_module_name`] folds the three non-`Normalized`
/// variants back to the original input name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizeModuleNameResult {
    /// Alias chain (or non-alias identity) terminates at a name present in
    /// `class_decls`.
    Normalized(TypeName),
    /// Chain terminates at a name that is in neither `class_alias_decls` nor
    /// `class_decls`. `original` is the queried key; `target` is the
    /// dangling end of the chain.
    UnknownTarget {
        original: TypeName,
        target: TypeName,
    },
    /// Chain revisits a name (`A = B`, `B = A`). `original` is the queried
    /// key.
    Cycle { original: TypeName },
    /// Queried name is not a class/module-kind `TypeName` (i.e. the
    /// `TypeName::Alias` or `TypeName::Interface` variant). One of the
    /// shapes rbs's `normalize_module_name?` collapses into `nil` —
    /// crema splits the nil cases out (`UnknownTarget`, `NotClassOrModule`)
    /// so callers do not have to re-do the lookup to figure out *why*
    /// normalization didn't apply. `NotClassOrModule` specifically replaces
    /// rbs's `raise "Class/module name is expected"` panic with a state
    /// value; future "chain ended in `interface_decls` / `type_alias_decls`"
    /// shapes can land here too if a caller ever needs to distinguish them.
    NotClassOrModule { original: TypeName },
}


/// Mirrors rbs's `AST::Declarations::Class | AST::Ruby::Declarations::ClassDecl`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassDeclaration {
    Signature(Arc<Class>),
    Ruby(Arc<RubyClassDecl>),
}

/// Mirrors rbs's `AST::Declarations::Module | AST::Ruby::Declarations::ModuleDecl`.
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleDeclaration {
    Signature(Arc<Module>),
    Ruby(Arc<RubyModuleDecl>),
}

/// Mirrors rbs's `AST::Declarations::ClassAlias | AST::Ruby::Declarations::ClassModuleAliasDecl`.
///
/// `RubyClassModuleAliasDecl` is shared between class and module aliases
/// on the rbs side; the class/module distinction lives on the wrapping
/// [`ClassOrModuleAliasEntry`], not on the inner node.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassAliasDeclaration {
    Signature(Arc<ClassAlias>),
    Ruby(Arc<RubyClassModuleAliasDecl>),
}

/// Mirrors rbs's `AST::Declarations::ModuleAlias | AST::Ruby::Declarations::ClassModuleAliasDecl`.
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleAliasDeclaration {
    Signature(Arc<ModuleAlias>),
    Ruby(Arc<RubyClassModuleAliasDecl>),
}

/// Mirrors `RBS::Environment::ClassEntry`.
///
/// `primary_decl` is the rbs primary-decl selection cached as a field
/// (first decl with a `super_class`; else first decl), so reads cost
/// one Arc bump.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassEntry {
    pub(crate) name: TypeName,
    pub(crate) context_decls: Box<[(DeclOrigin, Context, ClassDeclaration)]>,
    pub(crate) primary_decl: ClassDeclaration,
}

impl ClassEntry {
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    pub fn context_decls(&self) -> &[(DeclOrigin, Context, ClassDeclaration)] {
        &self.context_decls
    }

    pub fn primary_decl(&self) -> &ClassDeclaration {
        &self.primary_decl
    }
}

/// Mirrors `RBS::Environment::ModuleEntry`. Distinct from [`ClassEntry`]
/// so `self_types` (module-only) and `super_class` (class-only) cannot be
/// reached through one tag-discriminated value.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleEntry {
    pub(crate) name: TypeName,
    pub(crate) context_decls: Box<[(DeclOrigin, Context, ModuleDeclaration)]>,
    pub(crate) primary_decl: ModuleDeclaration,
}

impl ModuleEntry {
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    pub fn context_decls(&self) -> &[(DeclOrigin, Context, ModuleDeclaration)] {
        &self.context_decls
    }

    pub fn primary_decl(&self) -> &ModuleDeclaration {
        &self.primary_decl
    }

    /// Mirrors rbs `RBS::Environment::ModuleEntry#self_types`:
    /// `each_decl.flat_map(&:self_types).uniq`. Aggregates explicit
    /// `module M : T` declarations across every reopens, deduping on
    /// `(name, args)` since rbs `Module::Self#==` ignores location.
    /// Signature decls contribute their `self_types` field directly;
    /// Ruby decls contribute via [`RubyModuleDecl::self_types`], which
    /// filters `@rbs module-self: T` (`Members::ModuleSelfMember`) out
    /// of the body — matching rbs's
    /// `AST::Ruby::Declarations::ModuleDecl#self_types`.
    pub fn self_types(&self) -> Vec<SelfType> {
        let mut aggregated: Vec<SelfType> = Vec::new();
        let push_unique = |aggregated: &mut Vec<SelfType>, st: SelfType| {
            if !aggregated
                .iter()
                .any(|existing| same_self_type(existing, &st))
            {
                aggregated.push(st);
            }
        };
        for (_, _, decl) in self.context_decls.iter() {
            match decl {
                ModuleDeclaration::Signature(m) => {
                    for st in &m.self_types {
                        push_unique(&mut aggregated, st.clone());
                    }
                }
                ModuleDeclaration::Ruby(m) => {
                    for st in m.self_types() {
                        push_unique(&mut aggregated, st);
                    }
                }
            }
        }
        aggregated
    }
}

fn same_self_type(a: &SelfType, b: &SelfType) -> bool {
    a.name == b.name && types_eq_ignore_location(&a.args, &b.args)
}

fn types_eq_ignore_location(a: &[Type], b: &[Type]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| strip_location(x) == strip_location(y))
}

/// Clone `ty` with every embedded location set to `None`. Used to
/// give `==` rbs-equivalent semantics (rbs `Type#==` ignores location, see
/// `lib/rbs/types.rb`). The crema `Type` `PartialEq` derive still includes
/// location, so dedup paths normalize through this helper. Future-proofed by
/// listing every `Type` variant explicitly: a new variant will fail to
/// compile here and force the caller to decide whether the new shape carries
/// a `location` field.
pub(super) fn strip_location(ty: &Type) -> Type {
    match ty {
        Type::ClassInstance(t) => Type::ClassInstance(ClassInstanceType {
            name: t.name,
            args: t.args.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::ClassSingleton(t) => Type::ClassSingleton(ClassSingletonType {
            name: t.name,
            args: t.args.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Interface(t) => Type::Interface(InterfaceType {
            name: t.name,
            args: t.args.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Alias(t) => Type::Alias(AliasType {
            name: t.name,
            args: t.args.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Variable(t) => Type::Variable(VariableType {
            name: t.name,
            location: None,
        }),
        Type::Union(t) => Type::Union(UnionType {
            types: t.types.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Intersection(t) => Type::Intersection(IntersectionType {
            types: t.types.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Optional(t) => Type::Optional(OptionalType {
            ty: Box::new(strip_location(&t.ty)),
            location: None,
        }),
        Type::Tuple(t) => Type::Tuple(TupleType {
            types: t.types.iter().map(strip_location).collect(),
            location: None,
        }),
        Type::Record(t) => Type::Record(RecordType {
            fields: t
                .fields
                .iter()
                .map(|f| RecordField {
                    key: f.key.clone(),
                    ty: strip_location(&f.ty),
                    required: f.required,
                })
                .collect(),
            location: None,
        }),
        Type::Proc(proc_type) => Type::Proc(Box::new(ProcType {
            function: strip_function_type_location(&proc_type.function),
            self_type: proc_type
                .self_type
                .as_ref()
                .map(|t| Box::new(strip_location(t))),
            block: proc_type.block.as_ref().map(strip_block_location),
            location: None,
        })),
        Type::Literal(t) => Type::Literal(LiteralType {
            literal: t.literal.clone(),
            location: None,
        }),
        Type::Base(b) => Type::Base(BaseType {
            kind: b.kind.clone(),
            location: None,
        }),
    }
}

pub(super) fn strip_param_location(p: &FunctionParam) -> FunctionParam {
    FunctionParam {
        ty: Box::new(strip_location(&p.ty)),
        name: p.name,
        location: None,
    }
}

pub(super) fn strip_keyword_param_location(kp: &KeywordParam) -> KeywordParam {
    KeywordParam {
        name: kp.name,
        param: strip_param_location(&kp.param),
    }
}

pub(super) fn strip_function_type_location(ft: &Function) -> Function {
    match ft {
        Function::Typed(f) => Function::Typed(FunctionType {
            required_positionals: f
                .required_positionals
                .iter()
                .map(strip_param_location)
                .collect(),
            optional_positionals: f
                .optional_positionals
                .iter()
                .map(strip_param_location)
                .collect(),
            rest_positionals: f
                .rest_positionals
                .as_ref()
                .map(|p| Box::new(strip_param_location(p))),
            trailing_positionals: f
                .trailing_positionals
                .iter()
                .map(strip_param_location)
                .collect(),
            required_keywords: f
                .required_keywords
                .iter()
                .map(strip_keyword_param_location)
                .collect(),
            optional_keywords: f
                .optional_keywords
                .iter()
                .map(strip_keyword_param_location)
                .collect(),
            rest_keywords: f
                .rest_keywords
                .as_ref()
                .map(|p| Box::new(strip_param_location(p))),
            return_type: Box::new(strip_location(&f.return_type)),
        }),
        Function::Untyped(u) => {
            let mut cloned = u.clone();
            cloned.return_type = Box::new(strip_location(&cloned.return_type));
            Function::Untyped(cloned)
        }
    }
}

pub(super) fn strip_block_location(b: &BlockType) -> BlockType {
    BlockType {
        required: b.required,
        function: strip_function_type_location(&b.function),
        self_type: b.self_type.as_ref().map(|t| Box::new(strip_location(t))),
    }
}

/// Mirrors `RBS::Environment::InterfaceEntry`. Single-decl per name;
/// no inline Ruby counterpart in rbs today.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceEntry {
    pub(crate) name: TypeName,
    pub(crate) file: DeclOrigin,
    pub(crate) context: Context,
    pub(crate) decl: Arc<Interface>,
}

impl InterfaceEntry {
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    /// Real path for `DeclOrigin::Path`; `None` otherwise. See
    /// [`DeclOrigin::file`].
    pub fn file(&self) -> Option<Name> {
        self.file.file()
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn decl(&self) -> &Arc<Interface> {
        &self.decl
    }
}

/// Value type for [`Environment::class_decls`], matching rbs's
/// `Hash[TypeName, ModuleEntry | ClassEntry]`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassOrModule {
    Class(ClassEntry),
    Module(ModuleEntry),
}

/// Mirrors `RBS::Environment::ClassAliasEntry`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassAliasEntry {
    pub(crate) name: TypeName,
    pub(crate) file: DeclOrigin,
    pub(crate) context: Context,
    pub(crate) decl: ClassAliasDeclaration,
}

impl ClassAliasEntry {
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    /// Real path for `DeclOrigin::Path`; `None` otherwise. See
    /// [`DeclOrigin::file`].
    pub fn file(&self) -> Option<Name> {
        self.file.file()
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn decl(&self) -> &ClassAliasDeclaration {
        &self.decl
    }
}

/// Mirrors `RBS::Environment::ModuleAliasEntry`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleAliasEntry {
    pub(crate) name: TypeName,
    pub(crate) file: DeclOrigin,
    pub(crate) context: Context,
    pub(crate) decl: ModuleAliasDeclaration,
}

impl ModuleAliasEntry {
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    /// Real path for `DeclOrigin::Path`; `None` otherwise. See
    /// [`DeclOrigin::file`].
    pub fn file(&self) -> Option<Name> {
        self.file.file()
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn decl(&self) -> &ModuleAliasDeclaration {
        &self.decl
    }
}

/// Value type for [`Environment::class_alias_decls`], matching rbs's
/// `Hash[TypeName, ModuleAliasEntry | ClassAliasEntry]`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassOrModuleAliasEntry {
    Class(ClassAliasEntry),
    Module(ModuleAliasEntry),
}

impl ClassOrModuleAliasEntry {
    /// Location of the alias rhs (the `old_name` reference). Used by the
    /// build-layer validator (ADR-0013) to attach `file:line` to the
    /// `UnknownTypeName` diagnostic when the alias chain dangles.
    /// Mirrors rbs's `decl.location&.[](:old_name)`.
    ///
    /// Returns `None` for inline aliases that pre-date the
    /// `ClassModuleAliasDecl::old_name_location` plumbing (test
    /// fixtures synthesized via `test_support`).
    pub fn old_name_location(&self) -> Option<LocationRange> {
        match self {
            ClassOrModuleAliasEntry::Class(e) => match &e.decl {
                ClassAliasDeclaration::Signature(s) => s.location.map(|l| l.old_name_range),
                ClassAliasDeclaration::Ruby(_) => None,
            },
            ClassOrModuleAliasEntry::Module(e) => match &e.decl {
                ModuleAliasDeclaration::Signature(s) => s.location.map(|l| l.old_name_range),
                ModuleAliasDeclaration::Ruby(_) => None,
            },
        }
    }

    pub fn old_name_source_location(&self, names: &NameTable) -> Option<SourceLocation> {
        let (sf, range): (Option<Name>, Option<LocationRange>) = match self {
            ClassOrModuleAliasEntry::Class(e) => match &e.decl {
                ClassAliasDeclaration::Signature(s) => {
                    (s.source_file, s.location.map(|l| l.old_name_range))
                }
                ClassAliasDeclaration::Ruby(_) => (None, None),
            },
            ClassOrModuleAliasEntry::Module(e) => match &e.decl {
                ModuleAliasDeclaration::Signature(s) => {
                    (s.source_file, s.location.map(|l| l.old_name_range))
                }
                ModuleAliasDeclaration::Ruby(_) => (None, None),
            },
        };
        sf.zip(range).map(|(f, r)| SourceLocation {
            file: PathBuf::from(names.resolve(f)),
            range: r,
        })
    }
}

/// Mirrors `RBS::Environment`. Per-name aggregation only — flat method /
/// mixin / superclass maps live on
/// [`crate::definition_builder::DefinitionBuilder`].
///
/// Fields are `pub(crate)` so the only public way to construct an
/// `Environment` is `EnvironmentDraft::build`. After build the value is
/// guaranteed to have passed identity-level validation
/// (`primary_decl` selection, type-params consistency); exposing fields
/// for direct construction would let callers bypass that and produce an
/// inconsistent frozen value.
///
/// `names` is the [`NameTable`] moved from the draft at build time. Every
/// interned `Name` / `TypeName` reachable through this `Environment` was
/// produced against this table; callers must build map-lookup keys
/// against `env.names()`, not a fresh `NameTable` (see ADR-0017 Phase 4d).
///
/// `Clone` deep-copies every map (including `names`, via `NameTable`'s own
/// `Clone`) — added for `_internal incr-bench` (ADR-0028 P3 measurement),
/// which needs an owned, independent copy to feed [`Self::unload`] per
/// iteration without consuming the generation kept alive for comparison.
/// No other caller clones an `Environment`; the check pipeline builds and
/// discards one per run.
#[derive(Debug, Clone)]
pub struct Environment {
    pub(crate) class_decls: FxHashMap<TypeName, ClassOrModule>,
    pub(crate) interface_decls: FxHashMap<TypeName, InterfaceEntry>,
    pub(crate) class_alias_decls: FxHashMap<TypeName, ClassOrModuleAliasEntry>,
    pub(crate) type_alias_decls: FxHashMap<TypeName, SingleEntry<TypeAlias>>,
    pub(crate) constant_decls: FxHashMap<TypeName, SingleEntry<Constant>>,
    pub(crate) global_decls: FxHashMap<Symbol, GlobalEntry>,
    pub(crate) sources: Box<[Option<RubyLocation>]>,
    pub(crate) names: NameTable,
    /// Precomputed alias-chain folds. Populated by `EnvironmentDraft::build`
    /// for every key in `class_alias_decls`; non-alias names are looked up
    /// through the `Normalized(self)` fallback in
    /// [`Environment::normalize_module_name_result`].
    pub(crate) normalized_module_names: FxHashMap<TypeName, NormalizeModuleNameResult>,
    /// Lazy G-layer backend (ADR-0028 slice 2a-2). `Some` only for
    /// environments grafted onto a gem snapshot: the maps above then hold
    /// the A layer (plus per-name merges of reopened G names) and lookups
    /// fall through to the snapshot probe. `None` for plain builds — every
    /// accessor stays a direct map borrow.
    pub(crate) g: Option<Arc<GSnapshotBackend>>,
    /// Per-kind count of names present in both the A-layer map and the
    /// snapshot (reopened classes/modules, flatten-shadowed interfaces).
    /// Subtracted in [`Decls::len`] so the layered view counts each name
    /// once. Kinds whose cross-layer duplicates are build errors have no
    /// overlap and are omitted.
    pub(crate) g_overlap: GOverlap,
    /// Reverse index from a source file to the [`PathIndexKey`]s it
    /// contributed. Carried over verbatim from
    /// `EnvironmentDraft::path_index` at build time — see that field's doc
    /// for the population and `None`-file rules (ADR-0028 Decision 1).
    /// Consumed by [`Self::unload`] (ADR-0028 slice S1b). Two-layer
    /// (ADR-0028 F6 remainder): a warm-decoded environment borrows the
    /// snapshot's flat table and only touched files land in the overlay.
    pub(crate) path_index: PathIndexState,
    /// Resolver table (rbs `Resolver::TypeNameResolver.build`'s
    /// `all_names`): every real declared `TypeName` — class / module /
    /// interface / type-alias, flattened to include nested decls — plus
    /// the G-layer's `resolver_names()` when a snapshot is attached.
    /// Persisted here so [`Self::unload`] can delta-update it instead of
    /// `EnvironmentDraft::build` re-walking every retained decl on the
    /// next build (ADR-0028 slice S2). Constants are excluded, matching
    /// `EnvironmentDraft::build`'s Pass 1 (a constant is a value
    /// declaration, not a type). Two-layer — see [`Self::path_index`].
    pub(crate) all_names: FxHashSet<TypeName>,
    /// Resolver table's alias half (rbs `TypeNameResolver.build`'s
    /// `aliases`): every `class_alias_decls` key (top-level and nested)
    /// mapped to its raw RHS text and declaration context, plus the
    /// G-layer's `alias_entries()` when a snapshot is attached. See
    /// [`Self::all_names`] for the persistence rationale. Two-layer —
    /// see [`Self::path_index`].
    pub(crate) aliases: FxHashMap<TypeName, (String, Context)>,
    /// `included do` / `prepended do` block → final include / prepend
    /// targets, per ActiveSupport::Concern module, computed by the
    /// infusion pipeline's concern expansion
    /// (`infusion_collector::load_collected`). Read by
    /// `DefinitionBuilder::concern_block_targets` so the type checker can
    /// walk a concern block body under each target's singleton context.
    /// Kept as its own table rather than derived from the synthetic
    /// decls: a block whose only calls synthesize no member (`validates`)
    /// leaves no decl behind, yet still needs its targets. An entry with
    /// zero targets is recorded too — the checker skips that body, as
    /// Rails never runs it. A-layer only (concerns are a Ruby-only
    /// mechanism), so never persisted in a G snapshot.
    pub(crate) concern_block_targets: FxHashMap<TypeName, Vec<ConcernBlockTargets>>,
}

/// One concern block's expansion result — see
/// [`Environment::concern_block_targets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcernBlockTargets {
    /// File the concern module (and its block) is declared in. `None`
    /// for a file-less source (tests' inline units) — matches any file.
    pub source_file: Option<Name>,
    /// Byte range of the block node (`do ... end`), the same range the
    /// synthetic decls carry as `name_location`.
    pub location: PrismByteRange,
    /// Final include / prepend targets, in concern-site iteration order,
    /// deduplicated. Intermediate concern modules never appear.
    pub targets: Vec<TypeName>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GOverlap {
    pub(crate) class: usize,
    pub(crate) interface: usize,
}

/// Read-only view over one decl namespace of a frozen [`Environment`].
///
/// Two backends (ADR-0028 slice 2a-2): `Built` borrows a plain map;
/// `Layered` overlays the A-layer map on the lazy G-snapshot probe.
/// Point lookups check the A map first (merged reopened entries live
/// there), then fall through to a per-entry lazy decode. Full scans
/// (`iter` / `keys` / `len`) materialize the snapshot side once and
/// memoize it — the remaining full-scan callers are slice 2b's target.
/// rbs has no counterpart — `RBS::Environment` exposes plain Hash
/// attributes.
///
/// Method names and signatures mirror `FxHashMap` so call sites read
/// identically; lookups return `&'env` references tied to the
/// environment, not to this short-lived `Copy` handle.
#[derive(Debug)]
pub struct Decls<'env, K, V> {
    backend: DeclsBackend<'env, K, V>,
}

// Manual `Copy`/`Clone`: the derive would bound `K: Copy, V: Copy`,
// but every field is a borrow or fn pointer regardless of K/V.
impl<K, V> Clone for Decls<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V> Copy for Decls<'_, K, V> {}

/// Per-kind window into the [`GSnapshotBackend`]: plain fn pointers so
/// [`Decls`] needs no trait bound (and namespaces without a snapshot
/// side, like `class_alias_decls`, never construct one).
#[derive(Debug)]
struct GView<'env, K, V> {
    backend: &'env GSnapshotBackend,
    get: for<'e, 'k> fn(&'e GSnapshotBackend, &'k K) -> Option<&'e V>,
    contains: for<'k> fn(&GSnapshotBackend, &'k K) -> bool,
    len: fn(&GSnapshotBackend) -> usize,
    all: for<'e> fn(&'e GSnapshotBackend) -> &'e FxHashMap<K, V>,
    /// Key-table walk without entry decode (ADR-0028 slice 2b-3);
    /// backs [`Decls::keys`] so keys-only callers never materialize.
    keys: for<'e> fn(&'e GSnapshotBackend) -> Box<dyn Iterator<Item = &'e K> + 'e>,
}

impl<K, V> Clone for GView<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V> Copy for GView<'_, K, V> {}

#[derive(Debug)]
enum DeclsBackend<'env, K, V> {
    Built(&'env FxHashMap<K, V>),
    Layered {
        a: &'env FxHashMap<K, V>,
        g: GView<'env, K, V>,
        /// Names present in both the (logical) A layer and G; subtracted
        /// from `len`.
        overlap: usize,
    },
}

impl<K, V> Clone for DeclsBackend<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, V> Copy for DeclsBackend<'_, K, V> {}

impl<'env, K: Eq + Hash, V> Decls<'env, K, V> {
    fn built(map: &'env FxHashMap<K, V>) -> Self {
        Decls {
            backend: DeclsBackend::Built(map),
        }
    }

    fn layered(a: &'env FxHashMap<K, V>, g: GView<'env, K, V>, overlap: usize) -> Self {
        Decls {
            backend: DeclsBackend::Layered { a, g, overlap },
        }
    }

    pub fn get(&self, key: &K) -> Option<&'env V> {
        match self.backend {
            DeclsBackend::Built(map) => map.get(key),
            DeclsBackend::Layered { a, g, .. } => a.get(key).or_else(|| (g.get)(g.backend, key)),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        match self.backend {
            DeclsBackend::Built(map) => map.contains_key(key),
            DeclsBackend::Layered { a, g, .. } => {
                a.contains_key(key) || (g.contains)(g.backend, key)
            }
        }
    }

    /// A-layer-only membership (never G). The layered counterpart of
    /// probing the raw `class_decls` field — see [`Self::a_iter`] for
    /// when A-only semantics are correct.
    pub fn a_contains_key(&self, key: &K) -> bool {
        match self.backend {
            DeclsBackend::Built(map) => map.contains_key(key),
            DeclsBackend::Layered { a, .. } => a.contains_key(key),
        }
    }

    pub fn len(&self) -> usize {
        match self.backend {
            DeclsBackend::Built(map) => map.len(),
            DeclsBackend::Layered { a, g, overlap } => a.len() + (g.len)(g.backend) - overlap,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Key walk without entry decode: the `Layered` arm reads the
    /// backend's key tables directly instead of materializing the
    /// snapshot side the way [`Self::iter`] does. Same key set as
    /// `iter().map(|(k, _)| k)`, but the snapshot-phase order may
    /// differ (both are hash orders) — callers must not depend on it.
    pub fn keys(&self) -> DeclsKeys<'env, K, V> {
        match self.backend {
            DeclsBackend::Built(map) => DeclsKeys {
                a: map.keys(),
                shadow: None,
                g: None,
            },
            DeclsBackend::Layered { a, g, .. } => DeclsKeys {
                a: a.keys(),
                shadow: Some(a),
                g: Some((g.keys)(g.backend)),
            },
        }
    }

    /// Key walk over the A layer only (no G phase, no decode). See
    /// [`Self::a_iter`] for when A-only semantics are correct.
    pub fn a_keys(&self) -> impl Iterator<Item = &'env K> {
        match self.backend {
            DeclsBackend::Built(map) => map.keys(),
            DeclsBackend::Layered { a, .. } => a.keys(),
        }
    }

    /// Iterate only the A-layer entries. Under `Built` this is
    /// equivalent to [`Self::iter`] (there is no G side); under
    /// `Layered` the snapshot G phase is skipped entirely — the
    /// backend's decode-all is avoided and callers observe only the
    /// A layer.
    ///
    /// **Use only when G entries are structurally irrelevant** to the
    /// caller's question — e.g. `synthetic_method_context_targets`,
    /// which searches `class_decls` for `ClassDeclaration::Ruby` /
    /// `ModuleDeclaration::Ruby` inline entries. The G layer is
    /// `.rbs`-only (the G-only draft encoding in `main.rs` panics on
    /// Ruby decls; see slice 1a doc), so Ruby decls that could match
    /// live entirely in A — including A-side reopens of G classes,
    /// which are merged into the A entry by the layering step, not
    /// preserved as separate G rows.
    ///
    /// A caller that needs G presence for anything else (name-set
    /// membership, canonical decl lookup, etc.) must go through
    /// [`Self::iter`] or [`Self::keys`] instead.
    pub fn a_iter(&self) -> DeclsAIter<'env, K, V> {
        match self.backend {
            DeclsBackend::Built(map) => DeclsAIter { a: map.iter() },
            DeclsBackend::Layered { a, .. } => DeclsAIter { a: a.iter() },
        }
    }

    pub fn iter(&self) -> DeclsIter<'env, K, V> {
        match self.backend {
            DeclsBackend::Built(map) => DeclsIter {
                a: DeclsAIter { a: map.iter() },
                g: None,
            },
            DeclsBackend::Layered { a, g, .. } => DeclsIter {
                a: DeclsAIter { a: a.iter() },
                g: Some(((g.all)(g.backend).iter(), a)),
            },
        }
    }
}

/// Iterator over the A layer of a [`Decls`] view.
pub struct DeclsAIter<'env, K, V> {
    a: std::collections::hash_map::Iter<'env, K, V>,
}

impl<'env, K: Eq + Hash, V> Iterator for DeclsAIter<'env, K, V> {
    type Item = (&'env K, &'env V);

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next()
    }
}

/// G phase of [`DeclsIter`]: the materialized G map's iterator plus the
/// A map used to filter out shadowed names.
type GIterPhase<'env, K, V> = (
    std::collections::hash_map::Iter<'env, K, V>,
    &'env FxHashMap<K, V>,
);

/// Iterator over a [`Decls`] view: A-layer entries first, then the
/// G-snapshot entries whose names the A layer does not shadow.
pub struct DeclsIter<'env, K, V> {
    a: DeclsAIter<'env, K, V>,
    g: Option<GIterPhase<'env, K, V>>,
}

impl<'env, K: Eq + Hash, V> Iterator for DeclsIter<'env, K, V> {
    type Item = (&'env K, &'env V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.a.next() {
            return Some(item);
        }
        let (g, a_map) = self.g.as_mut()?;
        g.find(|(k, _)| !a_map.contains_key(k))
    }
}

/// Key iterator over a [`Decls`] view: A-layer keys, then the G key
/// table's keys the A layer does not shadow. Unlike [`DeclsIter`], the
/// G phase never decodes an entry.
pub struct DeclsKeys<'env, K, V> {
    a: std::collections::hash_map::Keys<'env, K, V>,
    /// A map used to shadow the G phase.
    shadow: Option<&'env FxHashMap<K, V>>,
    g: Option<Box<dyn Iterator<Item = &'env K> + 'env>>,
}

impl<'env, K: Eq + Hash, V> Iterator for DeclsKeys<'env, K, V> {
    type Item = &'env K;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(k) = self.a.next() {
            return Some(k);
        }
        let g = self.g.as_mut()?;
        let a_map = self.shadow?;
        g.find(|k| !a_map.contains_key(k))
    }
}

impl<'env, K: Eq + Hash, V> IntoIterator for Decls<'env, K, V> {
    type Item = (&'env K, &'env V);
    type IntoIter = DeclsIter<'env, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Environment {
    fn class_g_view(g: &GSnapshotBackend) -> GView<'_, TypeName, ClassOrModule> {
        GView {
            backend: g,
            get: GSnapshotBackend::class_entry,
            contains: GSnapshotBackend::class_contains,
            len: GSnapshotBackend::class_len,
            all: GSnapshotBackend::class_all,
            keys: GSnapshotBackend::class_key_iter,
        }
    }

    /// Concern block → final targets table, keyed by the concern module.
    /// See the field doc for provenance and consumers.
    pub fn concern_block_targets(&self) -> &FxHashMap<TypeName, Vec<ConcernBlockTargets>> {
        &self.concern_block_targets
    }

    pub fn class_decls(&self) -> Decls<'_, TypeName, ClassOrModule> {
        match &self.g {
            None => Decls::built(&self.class_decls),
            Some(g) => Decls::layered(
                &self.class_decls,
                Self::class_g_view(g),
                self.g_overlap.class,
            ),
        }
    }

    pub fn interface_decls(&self) -> Decls<'_, TypeName, InterfaceEntry> {
        match &self.g {
            None => Decls::built(&self.interface_decls),
            Some(g) => Decls::layered(
                &self.interface_decls,
                Self::interface_g_view(g),
                self.g_overlap.interface,
            ),
        }
    }

    fn interface_g_view(g: &GSnapshotBackend) -> GView<'_, TypeName, InterfaceEntry> {
        GView {
            backend: g,
            get: GSnapshotBackend::interface_entry,
            contains: GSnapshotBackend::interface_contains,
            len: GSnapshotBackend::interface_len,
            all: GSnapshotBackend::interface_all,
            keys: GSnapshotBackend::interface_key_iter,
        }
    }

    /// Always `Built`: the graft decodes every snapshot alias eagerly
    /// (the resolver and the normalize precompute need each alias's
    /// `old_name` up front) and merges them into the A-side map.
    pub fn class_alias_decls(&self) -> Decls<'_, TypeName, ClassOrModuleAliasEntry> {
        Decls::built(&self.class_alias_decls)
    }

    pub fn type_alias_decls(&self) -> Decls<'_, TypeName, SingleEntry<TypeAlias>> {
        match &self.g {
            None => Decls::built(&self.type_alias_decls),
            Some(g) => Decls::layered(&self.type_alias_decls, Self::type_alias_g_view(g), 0),
        }
    }

    fn type_alias_g_view(g: &GSnapshotBackend) -> GView<'_, TypeName, SingleEntry<TypeAlias>> {
        GView {
            backend: g,
            get: GSnapshotBackend::type_alias_entry,
            contains: GSnapshotBackend::type_alias_contains,
            len: GSnapshotBackend::type_alias_len,
            all: GSnapshotBackend::type_alias_all,
            keys: GSnapshotBackend::type_alias_key_iter,
        }
    }

    pub fn constant_decls(&self) -> Decls<'_, TypeName, SingleEntry<Constant>> {
        match &self.g {
            None => Decls::built(&self.constant_decls),
            Some(g) => Decls::layered(&self.constant_decls, Self::constant_g_view(g), 0),
        }
    }

    fn constant_g_view(g: &GSnapshotBackend) -> GView<'_, TypeName, SingleEntry<Constant>> {
        GView {
            backend: g,
            get: GSnapshotBackend::constant_entry,
            contains: GSnapshotBackend::constant_contains,
            len: GSnapshotBackend::constant_len,
            all: GSnapshotBackend::constant_all,
            keys: GSnapshotBackend::constant_key_iter,
        }
    }

    pub fn global_decls(&self) -> Decls<'_, Symbol, GlobalEntry> {
        match &self.g {
            None => Decls::built(&self.global_decls),
            Some(g) => Decls::layered(&self.global_decls, Self::global_g_view(g), 0),
        }
    }

    fn global_g_view(g: &GSnapshotBackend) -> GView<'_, Symbol, GlobalEntry> {
        GView {
            backend: g,
            get: GSnapshotBackend::global_entry,
            contains: GSnapshotBackend::global_contains,
            len: GSnapshotBackend::global_len,
            all: GSnapshotBackend::global_all,
            keys: GSnapshotBackend::global_key_iter,
        }
    }

    /// The lazy G-snapshot backend, when this environment was grafted
    /// onto one. Exposed for the `CREMA_DEBUG_SNAPSHOT_TIMING=1`
    /// decode-count line in `main`.
    pub fn g_backend(&self) -> Option<&Arc<GSnapshotBackend>> {
        self.g.as_ref()
    }

    /// A-layer membership for class/module names — used by the
    /// incremental validators' "has an A participant" filters.
    pub(crate) fn a_class_contains(&self, name: &TypeName) -> bool {
        self.class_decls.contains_key(name)
    }

    pub fn sources(&self) -> &[Option<RubyLocation>] {
        &self.sources
    }

    pub fn names(&self) -> &NameTable {
        &self.names
    }

    /// The [`PathIndexKey`]s `file` contributed, or `None` if `file`
    /// contributed nothing (never loaded, or fully unloaded). Exposed for
    /// `_internal incr-bench` (ADR-0028 P3 measurement) cache-warming; the
    /// check pipeline has no caller. Owned (not a borrow): a baseline hit
    /// decodes its flat row on the spot (ADR-0028 F6 remainder).
    pub fn path_index_for(&self, file: Name) -> Option<FxHashSet<PathIndexKey>> {
        self.path_index.get(file)
    }

    /// The [`ScanScope`] for a CLI diagnostic filter over `files`: the
    /// union of every [`PathIndexKey`] those files contributed. A file
    /// the index does not know (no declarations, never loaded) adds
    /// nothing, so the scope may be empty — the diagnostics-only walks
    /// then visit no owner at all, which is the right answer for a run
    /// whose filter would drop every env-level diagnostic anyway.
    pub fn scan_scope_for_files(&self, files: impl IntoIterator<Item = Name>) -> ScanScope {
        let mut keys: FxHashSet<PathIndexKey> = FxHashSet::default();
        for file in files {
            if let Some(file_keys) = self.path_index.get(file) {
                keys.extend(file_keys);
            }
        }
        ScanScope::Files(keys)
    }

    /// Every path-backed file this environment currently has declarations
    /// for. ADR-0028 S8: warm change detection needs this superset of the
    /// persisted `fingerprints` baseline, since a file added
    /// by a prior *differential* run never rewrites that baseline (only a
    /// cold write or compaction does) but does land in `path_index`.
    pub fn path_index_files(&self) -> impl Iterator<Item = Name> + '_ {
        self.path_index.files()
    }

    /// Detailed alias-resolution outcome. Returns
    /// [`NormalizeModuleNameResult::NotClassOrModule`] when `name` is the
    /// `Alias` or `Interface` variant of [`TypeName`]; otherwise consults
    /// the precomputed table built by `EnvironmentDraft::build`. Names not
    /// present in the table are reported as
    /// [`NormalizeModuleNameResult::Normalized`] with the input — non-alias
    /// keys do not get precompute entries, mirroring rbs's behaviour where
    /// `normalize_module_name?` returns the input when the name lives in
    /// `class_decls` directly.
    pub fn normalize_module_name_result(&self, name: &TypeName) -> NormalizeModuleNameResult {
        if !self.names.is_class(*name) {
            return NormalizeModuleNameResult::NotClassOrModule { original: *name };
        }
        match self.normalized_module_names.get(name) {
            Some(result) => result.clone(),
            None => NormalizeModuleNameResult::Normalized(*name),
        }
    }

    /// Compatibility accessor that mirrors rbs's
    /// `normalize_module_name(name) = normalize_module_name?(name) || name`.
    /// Cycle / unknown-target / wrong-kind cases fall back to the input
    /// `name`, matching the legacy `DefinitionBuilder::normalize_module_name`
    /// contract.
    pub fn normalize_module_name(&self, name: &TypeName) -> TypeName {
        match self.normalize_module_name_result(name) {
            NormalizeModuleNameResult::Normalized(t) => t,
            NormalizeModuleNameResult::UnknownTarget { original, .. }
            | NormalizeModuleNameResult::Cycle { original }
            | NormalizeModuleNameResult::NotClassOrModule { original } => original,
        }
    }

    /// Mirrors rbs `Environment#normalize_type_name`
    /// (`normalize_type_name?(name) || name`): a class-kind name goes
    /// through [`Self::normalize_module_name`] whole; an alias /
    /// interface-kind name keeps its own last segment and has only its
    /// namespace normalized (`A::t` with `class A = ::Target` becomes
    /// `::Target::t`). When the namespace cannot be normalized (cycle,
    /// unknown alias target, undeclared parent) the input is returned
    /// unchanged, exactly like the `|| name` fallback.
    ///
    /// Resolution (relative → absolute, rbs `absolute_type_name`) is a
    /// separate step that runs first; this only rewrites alias names
    /// to their targets. Called by type lowering so no lowered `Ty`
    /// carries a class alias name (Steep displays the normalized name).
    pub fn normalize_type_name(&self, name: TypeName) -> TypeName {
        if self.names.is_class(name) {
            return self.normalize_module_name(&name);
        }
        let Some(parent) = self.names.type_name_parent(name) else {
            return name;
        };
        if self.names.type_name_is_root(parent) {
            return name;
        }
        let Some(last) = self.names.last_segment(name) else {
            return name;
        };
        match self.normalize_module_name_result(&parent) {
            NormalizeModuleNameResult::Normalized(p) => self.names.append_type_name(p, last),
            _ => name,
        }
    }

    /// Mirrors `RBS::Environment#unload` (rbs `lib/rbs/environment.rb`,
    /// bottom of the file): rebuild an environment with every declaration
    /// contributed by `paths` removed — equivalent to a full build that
    /// never saw those files. `paths` holds interned file [`Name`]s (rbs's
    /// `unload` also accepts a `Buffer` there; crema has no buffer
    /// indirection at this layer, so callers resolve to `Name` first).
    ///
    /// crema-only extension (ADR-0028 Decision 1): rbs's `unload` is
    /// O(env) — it re-adds every retained source to a fresh `Environment`.
    /// This uses [`Self::path_index`] to edit the frozen maps in place, in
    /// O(decls contributed by `paths`), and returns the result as a COW
    /// overlay on a fresh [`EnvironmentDraft`] (ADR-0028 Decision 2) rather
    /// than a rebuilt `Environment` — the caller inserts the replacement
    /// declarations (if any) and calls [`EnvironmentDraft::build`], which
    /// seeds its output maps from the overlay and merges in the new
    /// entries. `names` (the [`NameTable`]) is carried into the returned
    /// draft; `sources` is dropped, matching [`EnvironmentDraft::from_frozen`]
    /// (no consumer reads it — see that method's doc comment).
    ///
    /// G-backed environments (ADR-0028 slice S1c): `self.g` is carried into
    /// the returned draft (mirroring `path_index` and `overlay` above) so
    /// `EnvironmentDraft::build` can re-graft it. A `class_decls` /
    /// `interface_decls` key that a G snapshot also declares can hold a
    /// per-name merge of G-origin decls ([`DeclOrigin::GSnapshot`])
    /// alongside the A-layer ones — see [`remove_class_or_module_decls`]
    /// for how those are stripped and re-merge is requested via
    /// [`FrozenOverlay::g_remerge`].
    pub fn unload(mut self, paths: &FxHashSet<Name>) -> EnvironmentDraft {
        // Inverted iteration vs the pre-F6-remainder `retain` (walk
        // `paths`, not every indexed file): removes exactly the same
        // entries — files in `paths` that hold at least one key — in
        // O(paths) probes instead of O(env).
        let mut affected: FxHashSet<PathIndexKey> = FxHashSet::default();
        for &file in paths {
            if let Some(keys) = self.path_index.remove_take(file) {
                affected.extend(keys);
            }
        }

        let Environment {
            mut class_decls,
            mut interface_decls,
            mut class_alias_decls,
            mut type_alias_decls,
            mut constant_decls,
            mut global_decls,
            sources: _,
            names,
            normalized_module_names: _,
            g,
            g_overlap: _,
            path_index,
            mut all_names,
            mut aliases,
            concern_block_targets,
        } = self;

        // Resolver-table delta (ADR-0028 slice S2): a name drops out of
        // `all_names`/`aliases` only when its decl map entry disappears
        // entirely — an open-class reopen that merely loses one
        // contributor (or a `ClassOrModule` name still declared by the
        // attached G snapshot) keeps its resolver-table membership,
        // mirroring how the name would still resolve in a fresh combined
        // build.
        let mut g_remerge: FxHashSet<TypeName> = FxHashSet::default();
        for key in affected {
            match key {
                PathIndexKey::ClassOrModule(name) => {
                    let removed = remove_class_or_module_decls(
                        &mut class_decls,
                        name,
                        paths,
                        g.as_deref(),
                        &mut g_remerge,
                    );
                    if removed {
                        all_names.remove(&name);
                    }
                }
                PathIndexKey::Interface(name) => {
                    if interface_decls
                        .get(&name)
                        .is_some_and(|e| e.file.index_file().is_some_and(|f| paths.contains(&f)))
                    {
                        interface_decls.remove(&name);
                        all_names.remove(&name);
                    }
                }
                PathIndexKey::ClassAlias(name) => {
                    let owned_by_paths = class_alias_decls.get(&name).is_some_and(|e| {
                        let file = match e {
                            ClassOrModuleAliasEntry::Class(c) => c.file,
                            ClassOrModuleAliasEntry::Module(m) => m.file,
                        };
                        file.index_file().is_some_and(|f| paths.contains(&f))
                    });
                    if owned_by_paths {
                        class_alias_decls.remove(&name);
                        aliases.remove(&name);
                    }
                }
                PathIndexKey::TypeAlias(name) => {
                    if type_alias_decls
                        .get(&name)
                        .is_some_and(|e| e.file.index_file().is_some_and(|f| paths.contains(&f)))
                    {
                        type_alias_decls.remove(&name);
                        all_names.remove(&name);
                    }
                }
                PathIndexKey::Constant(name) => {
                    if constant_decls
                        .get(&name)
                        .is_some_and(|e| e.file.index_file().is_some_and(|f| paths.contains(&f)))
                    {
                        constant_decls.remove(&name);
                    }
                }
                PathIndexKey::Global(sym) => {
                    if global_decls
                        .get(&sym)
                        .is_some_and(|e| e.file.index_file().is_some_and(|f| paths.contains(&f)))
                    {
                        global_decls.remove(&sym);
                    }
                }
            }
        }

        let mut draft = EnvironmentDraft::new_with_names(names);
        draft.path_index = path_index;
        draft.g = g;
        // Entries are keyed by the concern's own file: dropping the
        // unloaded files' entries mirrors dropping their decls. A
        // surviving entry's targets may still name a class declared in an
        // unloaded file — `load_collected` recomputes the whole table
        // when the infusion pipeline re-runs over the draft.
        draft.concern_block_targets = concern_block_targets
            .into_iter()
            .map(|(concern, entries)| {
                let kept: Vec<_> = entries
                    .into_iter()
                    .filter(|e| !e.source_file.is_some_and(|f| paths.contains(&f)))
                    .collect();
                (concern, kept)
            })
            .filter(|(_, entries)| !entries.is_empty())
            .collect();

        draft.overlay = Some(Box::new(FrozenOverlay {
            class_decls,
            interface_decls,
            class_alias_decls,
            type_alias_decls,
            constant_decls,
            global_decls,
            g_remerge,
            all_names,
            aliases,
        }));
        draft
    }
}

/// Remove every `paths`-owned decl from the `ClassOrModule` entry at `name`,
/// recomputing `primary_decl` from whatever remains (same selection rule as
/// `crate::environment::draft`'s `merge_nested_class` / `merge_nested_module`:
/// first decl with a `super_class` wins for classes, plain first decl for
/// modules). Drops the entry entirely when no decls survive. A no-op if
/// `name` is not present (a `path_index` key for an already-processed
/// removal, or a name never claimed).
///
/// G interplay (ADR-0028 slice S1c): when `name` is also declared by the
/// G snapshot (`g`), the entry's [`DeclOrigin::GSnapshot`] decl(s) are the
/// per-name merge `EnvironmentDraft::build` grafted in (S1a's decode
/// invariant). Those are always stripped first — unconditionally, ahead of
/// the `paths` filter — and `name` is recorded into `g_remerge` so `build`
/// knows to re-merge the G side into whatever A decls remain, rather than
/// leaving a half-degraded entry that silently drops the gem's
/// contribution. Non-`Path` decls of other origins (infusion-synthesized,
/// file-less test sources) are not G-origin and survive the strip.
///
/// Returns `true` when the entry was removed from `map` entirely (no
/// decls survived) — the caller (`Environment::unload`, ADR-0028 slice
/// S2) uses this to know whether `name` must also drop out of the
/// persisted resolver table (`all_names`); a name that keeps at least one
/// surviving decl is still declared and stays in the table.
fn remove_class_or_module_decls(
    map: &mut FxHashMap<TypeName, ClassOrModule>,
    name: TypeName,
    paths: &FxHashSet<Name>,
    g: Option<&GSnapshotBackend>,
    g_remerge: &mut FxHashSet<TypeName>,
) -> bool {
    use std::collections::hash_map::Entry;
    let Entry::Occupied(mut o) = map.entry(name) else {
        return false;
    };
    // `index_file`: a synthesized decl owned by an unloaded `.rb` file is
    // stripped alongside the file's own decls (ADR-0028 S2 relaxation) —
    // the warm caller re-synthesizes it from the reinserted source.
    let owned_by_paths = |file: &DeclOrigin| file.index_file().is_some_and(|f| paths.contains(&f));
    let g_reopened = g.is_some_and(|g| g.class_contains(&name));
    let keep = |file: &DeclOrigin| {
        if g_reopened && matches!(file, DeclOrigin::GSnapshot) {
            return false;
        }
        !owned_by_paths(file)
    };
    match o.get_mut() {
        ClassOrModule::Class(entry) => {
            let remaining: Vec<_> = entry
                .context_decls
                .iter()
                .filter(|(file, _, _)| keep(file))
                .cloned()
                .collect();
            if remaining.is_empty() {
                o.remove();
                return true;
            }
            if g_reopened {
                g_remerge.insert(name);
            }
            let primary_index = remaining
                .iter()
                .position(|(_, _, d)| class_decl_has_super(d))
                .unwrap_or(0);
            entry.primary_decl = remaining[primary_index].2.clone();
            entry.context_decls = remaining.into_boxed_slice();
        }
        ClassOrModule::Module(entry) => {
            let remaining: Vec<_> = entry
                .context_decls
                .iter()
                .filter(|(file, _, _)| keep(file))
                .cloned()
                .collect();
            if remaining.is_empty() {
                o.remove();
                return true;
            }
            if g_reopened {
                g_remerge.insert(name);
            }
            entry.primary_decl = remaining[0].2.clone();
            entry.context_decls = remaining.into_boxed_slice();
        }
    }
    false
}

/// Which declarations the diagnostics-only environment walks start from
/// (ADR-0036 Decision 4-2). Those walks — `DefinitionBuilder`'s
/// construction-time method / variable dup scans and every
/// `validator::full_validate` sub-check — exist only to produce
/// diagnostics; the type checker never reads their results. Under a CLI
/// file filter (ADR-0029: positional targets filter the *output*, scope
/// is `crema.toml`'s `check`) every diagnostic they produce for an owner
/// outside the filter's files is dropped at render time, so building it
/// is wasted work: on a single-file check those walks were ~2/5 of the
/// wall before the check phase even began.
///
/// `Files` restricts each walk to the owners `path_index` attributes to
/// the filter's files. That is output-preserving because a diagnostic
/// located in file `F` always comes from an owner that has a declaration
/// in `F` — a method / variable dup anchors on a member of one of the
/// owner's decls, an arity violation on a mixin clause of the host's
/// decl, an ancestor / type-alias cycle on its anchor participant's
/// primary decl, an alias-target error on the alias decl itself — and
/// `path_index` records an owner under every file holding one of its
/// decls, infusion's concern expansions included (they are inserted under
/// the concern's file). Cross-file relations are still followed from
/// those seeds (reopens are walked through the owner's merged entry,
/// cycles through the reachable graph), so what the seeds produce is
/// identical to what the whole walk produced for them.
///
/// Walk *order* is also preserved: a scoped walk iterates the same maps
/// as the whole walk and skips non-members, rather than iterating the
/// seed set — `sort_by_canonical_order` breaks ties by insertion order,
/// and two owners can emit diagnostics with an identical sort key (one
/// concern `def` expanded into several models anchors every dup on the
/// same concern-file location).
#[derive(Debug, Clone)]
pub enum ScanScope {
    /// Every declaration — a project-wide check, `extract`,
    /// `--update-baseline`: no filter, so every diagnostic reaches output.
    Whole,
    /// Only the owners the filter's files declared, as `path_index` keys.
    /// Built by [`Environment::scan_scope_for_files`].
    Files(FxHashSet<PathIndexKey>),
}

impl ScanScope {
    /// Whether a walk should visit the owner `key` names.
    pub fn admits(&self, key: PathIndexKey) -> bool {
        match self {
            ScanScope::Whole => true,
            ScanScope::Files(keys) => keys.contains(&key),
        }
    }
}
