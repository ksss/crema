//! ADR-0032 Decision 3/4: per-name shape fingerprint pass + diff +
//! changed-set expansion — the pure computation layer of incremental
//! change detection.
//!
//! Given a fresh [`Environment`], [`compute_fingerprint_table`] hashes
//! each A-layer declaration's *type surface shape* (never body, never
//! location — both are excluded by design). [`RawDiff::compute`] diffs two
//! such tables into added/removed/changed [`FingerprintKey`]s.
//! [`expand_changed_set`] walks a fresh [`AncestorGraph`] built over the
//! same env to pull in descendants (superclass/mixin propagation) and
//! dangling-reference referrers (deletion propagation), producing the
//! key set a consultation-log matcher can use.
//!
//! Persistence, per-file consulted-set probing, and wiring into `crema
//! check` are out of scope here (`incremental_cache_io_scheduler`); this
//! module never touches disk and is never called from the default check
//! path.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet};
use xxhash_rust::xxh3::Xxh3;

use crate::ast::declarations::{AsMember, Member as SigMember};
use crate::ast::members::IvarName;
use crate::ast::method_type::MethodType;
use crate::ast::ruby::members::{
    BlockEntry, DefMemberOrigin, DocStyle, ExplicitAnnotation, Member as RubyMember,
    PositionalEntry, SplatRestEntry, TypeAnnotations,
};
use crate::ast::types::{BlockType, Function, Type};
use crate::ast::{TypeParam, Variance};
use crate::definition::ancestor_graph::{AncestorGraph, Node};
use crate::environment::DeclOrigin;
use crate::environment::frozen::{
    self, ClassAliasDeclaration, ClassDeclaration, ClassOrModule, ClassOrModuleAliasEntry,
    Environment, ModuleAliasDeclaration, ModuleDeclaration,
};
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;

/// One fingerprinted slot in the A-layer environment. `Type`/`Member`
/// cover class, module, *and* interface declarations uniformly (both
/// have a type-level shape and member-level shapes); the other four
/// variants each correspond one-to-one with a single-value decl table
/// (ADR-0032 Decision 3's four spike-untested tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FingerprintKey {
    /// Type-level shape: superclass/self-types, type params, and the
    /// aggregate of every `include`/`extend`/`prepend` (or, for an
    /// interface, embedded interfaces).
    Type(TypeName),
    /// Member-level shape: one method/attr/ivar/cvar's signature,
    /// keyed by its owning name and its own symbol.
    Member(TypeName, Symbol),
    ClassAlias(TypeName),
    TypeAlias(TypeName),
    Constant(TypeName),
    Global(Symbol),
}

impl FingerprintKey {
    /// The owning `TypeName`, when this key has one — every variant
    /// except `Global`, which is a bare identifier with no place in the
    /// (class/module/interface-only) [`AncestorGraph`].
    pub fn type_name(&self) -> Option<TypeName> {
        match self {
            FingerprintKey::Type(n)
            | FingerprintKey::Member(n, _)
            | FingerprintKey::ClassAlias(n)
            | FingerprintKey::TypeAlias(n)
            | FingerprintKey::Constant(n) => Some(*n),
            FingerprintKey::Global(_) => None,
        }
    }
}

pub type FingerprintTable = FxHashMap<FingerprintKey, u64>;

/// Compute the fingerprint table for every A-layer declaration in `env`.
/// Walks each of the six decl tables' A-layer view only (`a_iter` — G
/// (gem snapshot) entries are out of scope, ADR-0032 Decision 3: G is
/// matched by lockfile key, not fingerprinted).
pub fn compute_fingerprint_table(env: &Environment) -> FingerprintTable {
    let mut table = FingerprintTable::default();
    let names = env.names();

    for (name, entry) in env.class_decls().a_iter() {
        hash_class_or_module(*name, entry, names, &mut table);
    }
    for (name, entry) in env.interface_decls().a_iter() {
        hash_interface(*name, entry.decl(), names, &mut table);
    }
    for (name, entry) in env.class_alias_decls().a_iter() {
        hash_class_alias(*name, entry, names, &mut table);
    }
    for (name, entry) in env.type_alias_decls().a_iter() {
        let mut h = Xxh3::new();
        hash_type(&entry.decl.ty, &mut h);
        table.insert(FingerprintKey::TypeAlias(*name), h.digest());
    }
    for (name, entry) in env.constant_decls().a_iter() {
        let mut h = Xxh3::new();
        hash_type(&entry.decl.ty, &mut h);
        table.insert(FingerprintKey::Constant(*name), h.digest());
    }
    for (name, entry) in env.global_decls().a_iter() {
        let mut h = Xxh3::new();
        hash_type(&entry.decl.ty, &mut h);
        table.insert(FingerprintKey::Global(*name), h.digest());
    }

    table
}

/// Added/removed/changed [`FingerprintKey`]s between two tables (the diff
/// against the previous table). A key present in both with an unequal hash is `changed`;
/// absent in `old` is `added`; absent in `new` is `removed`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RawDiff {
    pub added: FxHashSet<FingerprintKey>,
    pub removed: FxHashSet<FingerprintKey>,
    pub changed: FxHashSet<FingerprintKey>,
}

impl RawDiff {
    pub fn compute(old: &FingerprintTable, new: &FingerprintTable) -> RawDiff {
        let mut diff = RawDiff::default();
        for (key, new_hash) in new {
            match old.get(key) {
                None => {
                    diff.added.insert(*key);
                }
                Some(old_hash) if old_hash != new_hash => {
                    diff.changed.insert(*key);
                }
                Some(_) => {}
            }
        }
        for key in old.keys() {
            if !new.contains_key(key) {
                diff.removed.insert(*key);
            }
        }
        diff
    }

    /// Every key touched by this diff, regardless of category.
    pub fn all(&self) -> impl Iterator<Item = &FingerprintKey> {
        self.added.iter().chain(&self.removed).chain(&self.changed)
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Expand `raw` through `graph` (built over the *new* env — the same one
/// [`RawDiff`]'s `new` table came from) into a changed key set usable for
/// consultation-log matching.
///
/// Seeds from *every* raw-changed key's owning `TypeName` — not just
/// `FingerprintKey::Type` (type-level) entries, but `Member`/`ClassAlias`
/// too. This is deliberately broader than "this name's ancestry
/// changed": a `Member(Base, helper)`-only change (no type-level change
/// to `Base` at all) still needs to seed expansion from `Base`, because
/// a descendant `Sub < Base` that never overrides `helper` resolves it
/// by walking up to `Base` — the consultation-log query key such a
/// resolution records is keyed by the *querying* receiver (`Sub`, per
/// ADR-0032 Decision 2: keyed by receiver type name), not by `Base`, so only
/// propagating `Base`'s own change through the graph lets a matcher find
/// `Sub`'s query at all (crema-review adversarial pass flagged this
/// breadth as inconsistent with an earlier, narrower version of this doc
/// comment — the code was already intentionally this broad; the comment
/// was wrong, not the behavior). The trade-off is real (a member-only
/// change now walks the full descendant/interface-includer graph from
/// its owner, not just that one key) but over-invalidation is the safe
/// direction ADR-0032 accepts throughout.
///
/// For every seed `TypeName`, pulls in:
/// - every graph descendant (instance and singleton side) — covers
///   superclass/mixin swaps rippling to subclasses/includers, *and* a
///   fully-deleted class/module, because [`AncestorGraph`] registers a
///   referrer's superclass edge unconditionally (never gated on the
///   target actually resolving — verified by this module's own
///   `class_full_deletion_expands_to_referencing_descendant` test)
/// - every unresolved referrer — covers a deleted *mixin* target
///   (`include M`/`extend M`/`prepend M` do gate edge creation on
///   resolution, unlike superclass; a referrer whose `M` just
///   disappeared shows up only in the fresh graph's `unresolved` reverse
///   index, mirroring `AncestorGraph::extend_seeds`'s own existence-change
///   rule)
///
/// Each descendant/referrer found is folded in as `FingerprintKey::Type`
/// regardless of what kind of raw-changed key seeded it — a type-level
/// marker is what a consultation-log matcher needs to invalidate
/// anything that consulted the descendant's own resolution.
pub fn expand_changed_set(graph: &AncestorGraph, raw: &RawDiff) -> FxHashSet<FingerprintKey> {
    let mut expanded: FxHashSet<FingerprintKey> = raw.all().copied().collect();

    let seeds: FxHashSet<TypeName> = expanded
        .iter()
        .filter_map(FingerprintKey::type_name)
        .collect();

    let mut affected: FxHashSet<TypeName> = FxHashSet::default();
    for name in &seeds {
        for node in [Node::InstanceNode(*name), Node::SingletonNode(*name)] {
            affected.extend(
                graph
                    .each_descendant(&node)
                    .into_iter()
                    .map(Node::type_name),
            );
        }
        affected.extend(graph.each_unresolved_referrer(name).map(Node::type_name));
    }

    for name in affected {
        expanded.insert(FingerprintKey::Type(name));
    }

    expanded
}

// ---------------------------------------------------------------------
// Per-table hashing
// ---------------------------------------------------------------------

/// Accumulates one A-layer decl's member-level shapes. Each contributing
/// call computes its own member's digest in an isolated `Xxh3` and folds
/// it into the entry via `wrapping_add` rather than feeding a shared
/// streaming hasher directly — `Xxh3` is a non-commutative streaming
/// hash, so two members sharing a key (e.g. `def foo` and `def self.foo`
/// both keyed by the bare `foo` `Symbol` — [`FingerprintKey::Member`] is
/// deliberately the todo's literal `(TypeName, Symbol)` 2-tuple, not
/// `(TypeName, Symbol, MethodKind)`) previously produced a digest that
/// depended on declaration order alone, so a pure reordering refactor
/// with no semantic change would report a false `changed` (caught by
/// crema-review adversarial pass). `wrapping_add` is commutative and
/// associative and — unlike XOR — never cancels two identical
/// contributions back to the zero state, so two reopens defining the
/// same member identically still combine to a distinguishable digest.
type MemberDigests = FxHashMap<Symbol, u64>;

fn contribute_member<H>(digests: &mut MemberDigests, key: Symbol, build: H)
where
    H: FnOnce(&mut Xxh3),
{
    let mut h = Xxh3::new();
    build(&mut h);
    let entry = digests.entry(key).or_insert(0);
    *entry = entry.wrapping_add(h.digest());
}

/// Whether `origin` is A-layer content proper, as opposed to G (gem
/// snapshot) content grafted into the same `context_decls` slice when an
/// A-layer decl reopens a gem class/module
/// (`environment::draft::merge_g_class_or_module`). ADR-0032 Decision 3
/// tracks G separately by lockfile key, never by fingerprint — a
/// reopened gem class whose *gem* content changed (a version bump) must
/// not perturb this decl's fingerprint, and reopening a large gem class
/// must not balloon this decl's hashed member set (caught by
/// crema-review adversarial pass; this module's own header doc already
/// claimed this exclusion, but the `context_decls` walk hadn't
/// implemented it).
fn is_a_layer_origin(origin: DeclOrigin) -> bool {
    !matches!(origin, DeclOrigin::GSnapshot)
}

fn hash_class_or_module(
    name: TypeName,
    entry: &ClassOrModule,
    names: &NameTable,
    table: &mut FingerprintTable,
) {
    let mut type_hasher = Xxh3::new();
    let mut member_digests: MemberDigests = FxHashMap::default();

    match entry {
        ClassOrModule::Class(class_entry) => {
            match class_entry.primary_decl() {
                ClassDeclaration::Signature(c) => {
                    hash_type_params(&c.type_params, &mut type_hasher);
                    hash_option_super(
                        c.super_class.as_ref().map(|s| (s.name, s.args.as_slice())),
                        &mut type_hasher,
                    );
                }
                ClassDeclaration::Ruby(c) => {
                    let sup = c.super_class.as_ref().map(|s| {
                        let args: &[Type] = s
                            .type_annotation
                            .as_ref()
                            .map(|a| a.type_args.as_slice())
                            .unwrap_or(&[]);
                        (s.type_name, args)
                    });
                    hash_option_super(sup, &mut type_hasher);
                }
            }
            for (origin, _, decl) in class_entry.context_decls() {
                if !is_a_layer_origin(*origin) {
                    continue;
                }
                match decl {
                    ClassDeclaration::Signature(c) => {
                        let mut current_visibility = crate::ast::members::Visibility::Public;
                        for cm in &c.members {
                            if let Some(m) = cm.as_member() {
                                hash_sig_member(
                                    m,
                                    &mut type_hasher,
                                    &mut member_digests,
                                    names,
                                    &mut current_visibility,
                                );
                            }
                        }
                    }
                    ClassDeclaration::Ruby(c) => {
                        for m in &c.members {
                            hash_ruby_member(m, &mut type_hasher, &mut member_digests, names);
                        }
                    }
                }
            }
        }
        ClassOrModule::Module(module_entry) => {
            hash_type_params_for_primary_module(module_entry, &mut type_hasher);
            hash_self_types(
                module_entry
                    .self_types()
                    .iter()
                    .map(|s| (s.name, s.args.as_slice())),
                &mut type_hasher,
            );
            for (origin, _, decl) in module_entry.context_decls() {
                if !is_a_layer_origin(*origin) {
                    continue;
                }
                match decl {
                    ModuleDeclaration::Signature(m) => {
                        let mut current_visibility = crate::ast::members::Visibility::Public;
                        for cm in &m.members {
                            if let Some(mem) = cm.as_member() {
                                hash_sig_member(
                                    mem,
                                    &mut type_hasher,
                                    &mut member_digests,
                                    names,
                                    &mut current_visibility,
                                );
                            }
                        }
                    }
                    ModuleDeclaration::Ruby(m) => {
                        for mem in &m.members {
                            hash_ruby_member(mem, &mut type_hasher, &mut member_digests, names);
                        }
                    }
                }
            }
        }
    }

    table.insert(FingerprintKey::Type(name), type_hasher.digest());
    for (sym, digest) in member_digests {
        table.insert(FingerprintKey::Member(name, sym), digest);
    }
}

/// `ModuleEntry::self_types()` already aggregates across every
/// `context_decls` reopen (rbs `Environment::ModuleEntry#self_types`:
/// `each_decl.flat_map(&:self_types).uniq`) — used directly by
/// [`hash_class_or_module`]'s module branch instead of reading only
/// `primary_decl()`'s own self-types, which would silently ignore a `:
/// SelfType` constraint declared in a non-primary reopen (caught by
/// crema-review spec-consistency pass). Type params have no such
/// cross-reopen aggregator (rbs treats the primary decl as their sole
/// source, mirroring `class_super_or_default`'s primary-only read for
/// superclass), so this only handles type params.
fn hash_type_params_for_primary_module<H: Hasher>(
    module_entry: &crate::environment::frozen::ModuleEntry,
    h: &mut H,
) {
    match module_entry.primary_decl() {
        ModuleDeclaration::Signature(m) => hash_type_params(&m.type_params, h),
        ModuleDeclaration::Ruby(_) => {
            // Ruby-origin `ModuleDecl` carries no `type_params` field
            // (inline declarations don't support explicit generic type
            // parameters today) — nothing to hash.
        }
    }
}

fn hash_option_super<H: Hasher>(sup: Option<(TypeName, &[Type])>, h: &mut H) {
    match sup {
        Some((sup_name, args)) => {
            true.hash(h);
            sup_name.hash(h);
            args.len().hash(h);
            for a in args {
                hash_type(a, h);
            }
        }
        None => false.hash(h),
    }
}

fn hash_self_types<'a, H: Hasher>(
    self_types: impl Iterator<Item = (TypeName, &'a [Type])>,
    h: &mut H,
) {
    let mut entries: Vec<(TypeName, &[Type])> = self_types.collect();
    entries.sort_by_key(|(n, _)| *n);
    entries.len().hash(h);
    for (n, args) in entries {
        n.hash(h);
        args.len().hash(h);
        for a in args {
            hash_type(a, h);
        }
    }
}

fn hash_interface(
    name: TypeName,
    decl: &crate::ast::declarations::InterfaceDeclaration,
    names: &NameTable,
    table: &mut FingerprintTable,
) {
    let mut type_hasher = Xxh3::new();
    let mut member_digests: MemberDigests = FxHashMap::default();
    // Interfaces have no `public`/`private` markers in RBS syntax (every
    // member is implicitly public), so a fixed `Public` ambient is
    // correct here — unlike `hash_class_or_module`, there is no per-decl
    // walk that could ever flip it.
    let mut current_visibility = crate::ast::members::Visibility::Public;

    hash_type_params(&decl.type_params, &mut type_hasher);
    for m in &decl.members {
        hash_sig_member(
            m,
            &mut type_hasher,
            &mut member_digests,
            names,
            &mut current_visibility,
        );
    }

    table.insert(FingerprintKey::Type(name), type_hasher.digest());
    for (sym, digest) in member_digests {
        table.insert(FingerprintKey::Member(name, sym), digest);
    }
}

/// `ClassModuleAliasDecl::old_name` resolves the annotation's explicit
/// target text over `infered_old_name` (mirrors rbs
/// `AST::Ruby::Declarations::ClassModuleAliasDecl#old_name`) — hashing
/// `infered_old_name` directly missed every retarget expressed purely
/// through the annotation, since `infered_old_name` can legitimately
/// stay `None` in that shape both before and after the retarget (caught
/// by crema-review spec-consistency pass).
fn hash_class_alias(
    name: TypeName,
    entry: &ClassOrModuleAliasEntry,
    names: &NameTable,
    table: &mut FingerprintTable,
) {
    let mut h = Xxh3::new();
    match entry {
        ClassOrModuleAliasEntry::Class(e) => match e.decl() {
            ClassAliasDeclaration::Signature(s) => s.old_name.hash(&mut h),
            ClassAliasDeclaration::Ruby(r) => r.old_name(names).hash(&mut h),
        },
        ClassOrModuleAliasEntry::Module(e) => match e.decl() {
            ModuleAliasDeclaration::Signature(s) => s.old_name.hash(&mut h),
            ModuleAliasDeclaration::Ruby(r) => r.old_name(names).hash(&mut h),
        },
    }
    table.insert(FingerprintKey::ClassAlias(name), h.digest());
}

/// Type-level shape only — `include`/`extend`/`prepend` folded together
/// with superclass/self-types/type-params into one hasher, matching
/// ADR-0032 Decision 3's type-level grouping (superclass / include / type params).
fn hash_type_params<H: Hasher>(params: &[TypeParam], h: &mut H) {
    params.len().hash(h);
    for p in params {
        hash_type_param(p, h);
    }
}

fn hash_type_param<H: Hasher>(p: &TypeParam, h: &mut H) {
    p.name.hash(h);
    hash_variance(p.variance, h);
    hash_option_type(p.upper_bound.as_ref(), h);
    hash_option_type(p.lower_bound.as_ref(), h);
    hash_option_type(p.default_type.as_ref(), h);
    p.unchecked.hash(h);
}

fn hash_variance<H: Hasher>(v: Variance, h: &mut H) {
    match v {
        Variance::Invariant => 0u8.hash(h),
        Variance::Covariant => 1u8.hash(h),
        Variance::Contravariant => 2u8.hash(h),
    }
}

fn hash_option_type<H: Hasher>(ty: Option<&Type>, h: &mut H) {
    match ty {
        Some(t) => {
            true.hash(h);
            hash_type(t, h);
        }
        None => false.hash(h),
    }
}

/// Location-blind type hash: strip every embedded location (`frozen`'s
/// `strip_location`, already location-blind-`Type::eq`'s workhorse) and
/// hash the stripped clone via `Type`'s own `derive(Hash)` — every
/// stripped location is uniformly `None`, so the derive no longer sees
/// byte offsets, only shape.
fn hash_type<H: Hasher>(ty: &Type, h: &mut H) {
    frozen::strip_location(ty).hash(h);
}

fn hash_function<H: Hasher>(f: &Function, h: &mut H) {
    frozen::strip_function_type_location(f).hash(h);
}

fn hash_block<H: Hasher>(b: &BlockType, h: &mut H) {
    frozen::strip_block_location(b).hash(h);
}

fn hash_method_type<H: Hasher>(mt: &MethodType, h: &mut H) {
    hash_type_params(&mt.type_params, h);
    hash_function(&mt.function, h);
    match &mt.block {
        Some(b) => {
            true.hash(h);
            hash_block(b, h);
        }
        None => false.hash(h),
    }
}

fn hash_ivar_name<H: Hasher>(name: &IvarName, h: &mut H) {
    match name {
        IvarName::Unspecified => 0u8.hash(h),
        IvarName::Empty => 1u8.hash(h),
        IvarName::Name(s) => {
            2u8.hash(h);
            s.hash(h);
        }
    }
}

/// Fold one Signature-origin (`.rbs`) member into the type-level hasher
/// (mixins) or its owning member-level digest (method/attr/ivar/cvar).
///
/// `Public`/`Private` flip `current_visibility` — an *ambient* marker
/// that only `attr_reader`/`attr_writer`/`attr_accessor` with no
/// explicit `visibility:` actually consult
/// (`definition_builder::attribute_visibility`'s `AttributeKind::Instance`
/// arm; bare `def` always resolves through
/// `Visibility::from_ast_or_default`, which ignores ambient state
/// entirely — verified by reading both call sites before assuming the
/// generic "any subsequent method/attr" framing crema-review
/// spec-consistency pass first reported). `Alias` populates a real
/// method entry under `new_name` (`definition::MethodOrigin::Alias`,
/// `method_builder.rs`) and is hashed as such — the doc comment here
/// used to claim `Alias` was a no-shape marker like `Public`/`Private`;
/// that was wrong (crema-review, both agents independently).
fn hash_sig_member<H: Hasher>(
    m: &SigMember,
    type_hasher: &mut H,
    member_digests: &mut MemberDigests,
    names: &NameTable,
    current_visibility: &mut crate::ast::members::Visibility,
) {
    match m {
        SigMember::Include(inc) => {
            "include".hash(type_hasher);
            inc.name.hash(type_hasher);
            inc.args.len().hash(type_hasher);
            for a in &inc.args {
                hash_type(a, type_hasher);
            }
        }
        SigMember::Extend(inc) => {
            "extend".hash(type_hasher);
            inc.name.hash(type_hasher);
            inc.args.len().hash(type_hasher);
            for a in &inc.args {
                hash_type(a, type_hasher);
            }
        }
        SigMember::Prepend(inc) => {
            "prepend".hash(type_hasher);
            inc.name.hash(type_hasher);
            inc.args.len().hash(type_hasher);
            for a in &inc.args {
                hash_type(a, type_hasher);
            }
        }
        SigMember::MethodDefinition(def) => {
            contribute_member(member_digests, def.name, |h| {
                def.kind.hash(h);
                def.overloading.hash(h);
                def.visibility.hash(h);
                def.overloads.len().hash(h);
                for o in &def.overloads {
                    hash_method_type(&o.method_type, h);
                }
            });
        }
        SigMember::AttrReader(a) => hash_sig_attr(
            a.name,
            &a.ty,
            a.kind,
            a.visibility,
            &a.ivar_name,
            *current_visibility,
            member_digests,
        ),
        SigMember::AttrWriter(a) => hash_sig_attr(
            a.name,
            &a.ty,
            a.kind,
            a.visibility,
            &a.ivar_name,
            *current_visibility,
            member_digests,
        ),
        SigMember::AttrAccessor(a) => hash_sig_attr(
            a.name,
            &a.ty,
            a.kind,
            a.visibility,
            &a.ivar_name,
            *current_visibility,
            member_digests,
        ),
        SigMember::InstanceVariable(v) => {
            contribute_member(member_digests, v.name, |h| hash_type(&v.ty, h));
        }
        SigMember::ClassInstanceVariable(v) => {
            contribute_member(member_digests, v.name, |h| hash_type(&v.ty, h));
        }
        SigMember::ClassVariable(v) => {
            contribute_member(member_digests, v.name, |h| hash_type(&v.ty, h));
        }
        SigMember::Alias(a) => {
            contribute_member(member_digests, a.new_name, |h| {
                a.old_name.hash(h);
                a.kind.hash(h);
            });
        }
        SigMember::Public(_) => *current_visibility = crate::ast::members::Visibility::Public,
        SigMember::Private(_) => *current_visibility = crate::ast::members::Visibility::Private,
    }
    let _ = names; // Signature-origin names are already `Symbol` — no interning needed here.
}

#[allow(clippy::too_many_arguments)]
fn hash_sig_attr(
    name: Symbol,
    ty: &Type,
    kind: crate::ast::members::AttributeKind,
    visibility: Option<crate::ast::members::Visibility>,
    ivar_name: &IvarName,
    current_visibility: crate::ast::members::Visibility,
    member_digests: &mut MemberDigests,
) {
    // Mirrors `definition_builder::attribute_visibility`: an
    // `AttributeKind::Instance` attr with no explicit `visibility:`
    // resolves to the *ambient* `current_visibility`, not a fixed
    // default — an unmarked `private`/`public` toggle upstream of this
    // attr genuinely changes its effective visibility (crema-review
    // spec-consistency pass). `Singleton` attrs, like bare `def`, never
    // consult ambient state (`Visibility::from_ast_or_default`).
    let effective_visibility = match (kind, visibility) {
        (crate::ast::members::AttributeKind::Instance, None) => current_visibility,
        (_, Some(v)) => v,
        (crate::ast::members::AttributeKind::Singleton, None) => {
            crate::ast::members::Visibility::Public
        }
    };
    contribute_member(member_digests, name, |h| {
        kind.hash(h);
        effective_visibility.hash(h);
        hash_ivar_name(ivar_name, h);
        hash_type(ty, h);
    });
}

/// Ruby-origin (inline `# @rbs`) counterpart of [`hash_sig_member`].
/// `DefMemberOrigin` is not part of the hash — a synthetic
/// (concern-included/prepended) def's shape already changes whenever the
/// concern module it was synthesized from changes, so hashing origin
/// would only add noise, not signal. Ruby-origin syntax has no
/// `public`/`private` markers or `alias` keyword inside inline `# @rbs`
/// bodies (those are plain Ruby statements, not members this AST layer
/// models), so unlike the Signature-origin sibling there is no ambient
/// visibility or alias case to handle here.
fn hash_ruby_member<H: Hasher>(
    m: &RubyMember,
    type_hasher: &mut H,
    member_digests: &mut MemberDigests,
    names: &NameTable,
) {
    match m {
        RubyMember::Include(inc) => hash_ruby_mixin("include", &inc.mixin, type_hasher),
        RubyMember::Extend(inc) => hash_ruby_mixin("extend", &inc.mixin, type_hasher),
        RubyMember::Prepend(inc) | RubyMember::SingletonPrepend(inc) => {
            hash_ruby_mixin("prepend", &inc.mixin, type_hasher)
        }
        RubyMember::Def(def) => {
            let sym = names.intern_symbol(&def.name);
            contribute_member(member_digests, sym, |h| {
                def.kind.hash(h);
                let _ = DefMemberOrigin::Real; // origin intentionally excluded, see doc comment
                hash_method_type_annotation(&def.method_type, h);
                hash_option_comment_block(def.leading_comment.as_ref(), h);
                // `initialize` is the one method whose body carries
                // type-relevant facts: synthesized ivar declarations
                // derive from these `@ivar = param` pairs, so adding /
                // removing / reordering them must invalidate dependents.
                // Empty (= every non-initialize def) hashes as a length-0
                // sequence, preserving the "body edits don't move the
                // fingerprint" pin for ordinary methods.
                def.ivar_param_pairs.hash(h);
            });
        }
        RubyMember::AttrReader(a) => hash_ruby_attr(&a.attribute, member_digests, names),
        RubyMember::AttrWriter(a) => hash_ruby_attr(&a.attribute, member_digests, names),
        RubyMember::AttrAccessor(a) => hash_ruby_attr(&a.attribute, member_digests, names),
        RubyMember::InstanceVariable(v) => {
            let sym = names.intern_symbol(&v.annotation.name);
            contribute_member(member_digests, sym, |h| hash_type(&v.annotation.ty, h));
        }
        RubyMember::ModuleSelf(s) => {
            "self_type".hash(type_hasher);
            s.annotation.name.hash(type_hasher);
            s.annotation.args.len().hash(type_hasher);
            for a in &s.annotation.args {
                hash_type(a, type_hasher);
            }
        }
        RubyMember::Declaration(_) => {
            // Nested class/module/etc declarations are their own
            // top-level A-layer entries (walked independently by
            // `compute_fingerprint_table`'s own table loops) — hashing
            // them again here would double-count the same shape under
            // two different keys.
        }
    }
}

/// rbs's own `type_fingerprint` design (`docs/type_fingerprint.md`:
/// "Documentation comments are considered type related information")
/// folds a Ruby-origin member's leading comment text into its
/// fingerprint as a deliberate catch-all — doc-style `# @rbs` tags live
/// inside this same comment block, and not every nuance is reachable
/// through the parsed `DocStyle`/`TypeAnnotations` structure this
/// module already hashes structurally. Text only, never `location`
/// (`CommentLine` carries a `PrismByteRange` that must stay excluded).
fn hash_option_comment_block<H: Hasher>(
    comment: Option<&crate::ast::ruby::comment_block::CommentBlock>,
    h: &mut H,
) {
    match comment {
        Some(block) => {
            true.hash(h);
            block.comments.len().hash(h);
            for line in &block.comments {
                line.text.hash(h);
            }
        }
        None => false.hash(h),
    }
}

fn hash_ruby_mixin<H: Hasher>(
    tag: &'static str,
    mixin: &crate::ast::ruby::members::MixinMember,
    h: &mut H,
) {
    tag.hash(h);
    mixin.module_name.hash(h);
    match &mixin.annotation {
        Some(ann) => {
            true.hash(h);
            ann.type_args.len().hash(h);
            for a in &ann.type_args {
                hash_type(a, h);
            }
        }
        None => false.hash(h),
    }
}

fn hash_ruby_attr(
    attr: &crate::ast::ruby::members::AttributeMember,
    member_digests: &mut MemberDigests,
    names: &NameTable,
) {
    for name_node in &attr.name_nodes {
        let sym = names.intern_symbol(&name_node.name);
        contribute_member(member_digests, sym, |h| {
            // Raw annotation text, not a parsed `Type` — Ruby-origin
            // attrs store `type_text: Option<String>` pre-parse (see
            // `AttributeMember` doc comment); hashing the text directly
            // is location-free by construction, no strip needed.
            attr.type_text.hash(h);
        });
    }
}

fn hash_method_type_annotation<H: Hasher>(
    ann: &crate::ast::ruby::members::MethodTypeAnnotation,
    h: &mut H,
) {
    match &ann.type_annotations {
        TypeAnnotations::Array(overloads) => {
            0u8.hash(h);
            overloads.len().hash(h);
            for o in overloads {
                hash_explicit_annotation(o, h);
            }
        }
        TypeAnnotations::DocStyle(doc) => {
            1u8.hash(h);
            hash_doc_style(doc, h);
        }
        TypeAnnotations::None => 2u8.hash(h),
    }
}

fn hash_explicit_annotation<H: Hasher>(ann: &ExplicitAnnotation, h: &mut H) {
    match ann {
        ExplicitAnnotation::Colon(c) => {
            0u8.hash(h);
            hash_method_type(&c.method_type, h);
        }
        ExplicitAnnotation::MethodTypes(mts) => {
            1u8.hash(h);
            mts.overloads.len().hash(h);
            for o in &mts.overloads {
                hash_method_type(&o.method_type, h);
            }
        }
    }
}

fn hash_doc_style<H: Hasher>(doc: &DocStyle, h: &mut H) {
    match &doc.return_type_annotation {
        Some(r) => {
            true.hash(h);
            hash_type(&r.return_type, h);
        }
        None => false.hash(h),
    }
    hash_positional_entries(&doc.required_positionals, h);
    hash_positional_entries(&doc.optional_positionals, h);
    hash_positional_entries(&doc.trailing_positionals, h);
    match &doc.rest_positionals {
        Some(SplatRestEntry::Annotated(a)) => {
            0u8.hash(h);
            hash_type(&a.param_type, h);
        }
        Some(SplatRestEntry::ByName(n)) => {
            1u8.hash(h);
            n.hash(h);
        }
        Some(SplatRestEntry::Unnamed) => 2u8.hash(h),
        None => 3u8.hash(h),
    }
    let mut required_keywords: Vec<&(String, PositionalEntry)> =
        doc.required_keywords.iter().collect();
    required_keywords.sort_by(|a, b| a.0.cmp(&b.0));
    required_keywords.len().hash(h);
    for (name, entry) in required_keywords {
        name.hash(h);
        hash_positional_entry(entry, h);
    }
    let mut optional_keywords: Vec<&(String, PositionalEntry)> =
        doc.optional_keywords.iter().collect();
    optional_keywords.sort_by(|a, b| a.0.cmp(&b.0));
    optional_keywords.len().hash(h);
    for (name, entry) in optional_keywords {
        name.hash(h);
        hash_positional_entry(entry, h);
    }
    match &doc.rest_keywords {
        Some(crate::ast::ruby::members::DoubleSplatRestEntry::Annotated(a)) => {
            0u8.hash(h);
            hash_type(&a.param_type, h);
        }
        Some(crate::ast::ruby::members::DoubleSplatRestEntry::ByName(n)) => {
            1u8.hash(h);
            n.hash(h);
        }
        Some(crate::ast::ruby::members::DoubleSplatRestEntry::Unnamed) => 2u8.hash(h),
        None => 3u8.hash(h),
    }
    match &doc.block {
        Some(BlockEntry::Annotated(b)) => {
            0u8.hash(h);
            hash_function(&b.function, h);
        }
        Some(BlockEntry::ByName(n)) => {
            1u8.hash(h);
            n.hash(h);
        }
        Some(BlockEntry::Unnamed) => 2u8.hash(h),
        None => 3u8.hash(h),
    }
}

fn hash_positional_entries<H: Hasher>(entries: &[PositionalEntry], h: &mut H) {
    entries.len().hash(h);
    for e in entries {
        hash_positional_entry(e, h);
    }
}

fn hash_positional_entry<H: Hasher>(entry: &PositionalEntry, h: &mut H) {
    match entry {
        PositionalEntry::Annotated(a) => {
            0u8.hash(h);
            hash_type(&a.param_type, h);
        }
        PositionalEntry::ByName(n) => {
            1u8.hash(h);
            n.hash(h);
        }
    }
}

