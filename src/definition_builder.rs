//! Definition layer — lazy-on-demand memoized `Definition`s on top of
//! the frozen [`Environment`], plus the method / variable / alias
//! resolution helpers that walk the linearized ancestor chains.
//!
//! Mirrors `RBS::DefinitionBuilder` (see ADR-0015, ADR-0019, ADR-0020). Each class / module declaration produces two
//! `Definition`s (instance side and singleton side); each interface
//! declaration produces one. Populated lazily on the first
//! `build_instance` / `build_singleton` / `build_interface` call and
//! memoized in `Arc`-wrapped caches (ADR-0020). Method resolution
//! walks ancestors via [`AncestorBuilder::instance_ancestors`] /
//! `singleton_ancestors` / `interface_ancestors` and binds the root's
//! type params with [`InstanceAncestors::apply`].
//! ADR-0009 is preserved at the data-shape level: `Definition.methods`
//! stores own-class entries only; no ancestor-merged method tables are
//! pre-built. The free functions below (`lookup_instance_method`,
//! `expand_alias`, etc.) keep crema's ADR-0009 lazy ancestor walk: no
//! ancestor-merged method tables, lookup walks the linearized chain on
//! demand.

use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::OnceCell;
use std::cell::RefCell;
use std::sync::Arc;

use crate::ast::MethodKind;
use crate::ast::TypeParam as AstTypeParam;
use crate::ast::annotation::Annotation;
use crate::ast::declarations::{AsMember, Member};
use crate::ast::members::{AliasKind, AliasMember, AttributeKind, IvarName};
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::members::{DefMemberOrigin, Member as RubyMember};
use crate::ast::types::Type as AstType;
use crate::definition::ancestor_builder::{self, Ancestor, AncestorBuilder, AncestorSource};
use crate::definition::lowering_maps::LoweringMaps;
use crate::definition::method::deprecated_annotation;
use crate::definition::method_builder;
use crate::definition::{
    ConstantContext, ConstantResolver, Definition, LoweringEnv, MemberRef, Method,
    ResolverConstant, TypeDef, Variable, VariableDuplication, VariableDuplicationKind,
    VariableSource,
};
use crate::environment::draft::EnvironmentDraft;
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassDeclaration, ClassOrModule, ClassOrModuleAliasEntry,
    InterfaceEntry, ModuleAliasDeclaration, ModuleDeclaration,
};
use crate::environment::ruby_decl;
use crate::environment::{DeclKindLocal, DeclOrigin, Environment, InfusionUnit};
use crate::location::{DuplicateSource, LocationRange, SourceLocation};
use crate::name::{Name, NameTable, Symbol};
use crate::substitution::Substitution;
use crate::type_name::TypeName;
use crate::type_param::{TypeParam, TypeParamScope, TypeVarKey, TypeVarScope};
use crate::types::{
    Function, FunctionType, MethodType, Ty, Type, TypeTable, UntypedFunction, Visibility,
};
use std::path::PathBuf;

/// Key for the subtype-check memoization. Collects the four
/// `SubtypeChecker` inputs that fully determine the answer to `check`:
/// the relation (`sub`, `sup`) and the context (`self_bound`,
/// `type_variables_are_wildcards`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubtypeCacheKey {
    pub(crate) sub: Ty,
    pub(crate) sup: Ty,
    pub(crate) self_bound: Option<Ty>,
    pub(crate) type_variables_are_wildcards: bool,
}

type SubtypeCache = FxHashMap<SubtypeCacheKey, bool>;

/// On-demand memoized per-class definition cache built on top of a frozen
/// [`Environment`]. Mirrors `RBS::DefinitionBuilder`.
///
/// `build_instance` / `build_singleton` / `build_interface` are lazy:
/// a `Definition` is built the first time a given `TypeName` is requested
/// and then cached inside a `RefCell<FxHashMap>` for subsequent calls
/// (ADR-0020). `RefCell` keeps the public API as `&self` while allowing
/// interior mutation; it is `!Sync`, preserving the single-thread
/// assumption (concurrency is out of scope per ADR-0020).
#[derive(Debug)]
pub struct DefinitionBuilder {
    env: Arc<Environment>,
    /// Shared with [`Self::constant_resolver`] so the resolver can keep
    /// hitting the same ancestor cache without taking a borrow on the
    /// builder. Wrapped in [`Arc`] for shared ownership; consumers see a
    /// `&AncestorBuilder` through [`Self::ancestor_builder`] and stay
    /// unaffected.
    ancestor_builder: Arc<AncestorBuilder>,
    /// rbs `Resolver::ConstantResolver` port. Built once from
    /// [`Self::ancestor_builder`]; the resolver borrows the same Arc so
    /// `&self` lookup methods can walk ancestors without juggling
    /// references at call sites.
    constant_resolver: ConstantResolver,
    /// Shared declared-name membership + class-alias map for on-the-fly
    /// `LoweringEnv` construction. Built once in `from_environment`;
    /// `Arc`-shared so per-lookup instances can be assembled without
    /// re-walking every declaration.
    lowering_maps: Arc<LoweringMaps>,
    /// Lazy cache: one entry per class/module for the instance side.
    /// Populated on first call to `build_instance`; empty at construction.
    instance_definition_cache: RefCell<FxHashMap<TypeName, Arc<Definition>>>,
    /// Lazy cache: one entry per class/module for the singleton side.
    /// Populated on first call to `build_singleton`; empty at construction.
    singleton_definition_cache: RefCell<FxHashMap<TypeName, Arc<Definition>>>,
    /// Lazy cache: one entry per interface.
    /// Populated on first call to `build_interface`; empty at construction.
    interface_definition_cache: RefCell<FxHashMap<TypeName, Arc<Definition>>>,
    /// Memoized `SubtypeChecker::check` results keyed by the four inputs
    /// that fully determine the answer:
    /// `(sub, sup, self_bound, type_variables_are_wildcards)`.
    /// Mirrors `Steep::Subtyping::Cache` (`subtyping/cache.rb`): both
    /// positive and negative results are stored. No invalidation since
    /// the underlying environment is immutable post-build (ADR-0020).
    subtype_cache: RefCell<SubtypeCache>,
    /// Memoized `expand_alias` results: input `Ty` → fully-expanded `Ty`
    /// (after fixpoint alias-to-alias chasing). The function is called
    /// from both the dispatch path (`resolve_call_target_at` /
    /// `check_no_method_at`) and the utility path (`literal_access`,
    /// `inference.rs` hint expansion, `method_resolver::lookup_method`'s
    /// `Type::Alias` arm), so the in-function cache amortizes across all
    /// callers. Cyclic aliases hitting `ALIAS_EXPANSION_LIMIT` are stored
    /// as the bottomed-out `Type::Alias` itself, mirroring the function's
    /// return contract.
    expand_alias_cache: RefCell<FxHashMap<Ty, Ty>>,
    /// Memoized `normalize_receiver` results: input `Ty` → fully-normalized
    /// `Ty` (alias expand + Optional/Bool sugar widen + alias-of-union
    /// flatten, applied at the dispatch boundary). Keyed on the
    /// pre-normalization `Ty`. Independent of `expand_alias_cache` because
    /// the normalize result diverges (widen + flatten) and is only safe to
    /// reuse at dispatch sites, not at utility sites where widen would
    /// change semantics.
    normalize_receiver_cache: RefCell<FxHashMap<Ty, Ty>>,
    /// Resolved type-param cache keyed by class / module / interface
    /// `TypeName`. Mirrors the legacy `DefinitionBuilder.type_params`
    /// shape so Phase 5b consumers (subtyping / type_checker) can pull
    /// the same `Vec<TypeParam>` they used to read off the legacy
    /// builder. Populated on demand from the `Signature` variant only —
    /// the inline (Ruby) path stays Phase 5b's responsibility, so a
    /// class declared only via inline annotations resolves to `None`.
    type_params: TypeParamsCache,
    /// Duplicate attr/method definitions detected during a lightweight
    /// member-only scan at `from_environment` time (ADR-0020). Each entry
    /// is `(method_name, duplicate_location, original_location)`, where
    /// `duplicate_location` anchors the diagnostic at the redefining site
    /// (rbs's `member.location`) and `original_location` refers back to
    /// the first-encountered definition (rbs's `original_member.location`)
    /// — surfaced as the diagnostic's `duplicate_source` field so the
    /// user sees both sides of the clash.
    ///
    /// Mirrors `RBS::DuplicatedMethodDefinitionError` but collected as
    /// structured data so the build-layer validator can emit them
    /// alongside other diagnostics without re-walking the environment.
    method_dups: PerNameCache<MethodDupEntry>,
    /// Recursive-alias cycles (self-loops and multi-step SCCs) detected
    /// during the same construction-time member walk that populates
    /// `method_dups`. Mirrors `RBS::RecursiveAliasDefinitionError`.
    alias_cycles: PerNameCache<AliasCycleEntry>,
    /// Duplicate variable declarations detected from `Variable.parent_variable`
    /// chains during the construction-time scan.
    variable_dups: PerNameCache<VariableDuplication>,
    /// `(method, kind) -> injection-target owners` index over synthetic
    /// ActiveSupport::Concern `def`s (`DefMemberOrigin::SyntheticConcernIncluded`
    /// / `SyntheticConcernPrepended`), built by a single A-layer walk
    /// (`scan_synthetic_concern_index`) at construction — replaces
    /// `TypeChecker::synthetic_method_context_targets`'s old per-`def`-node
    /// full A walk (ADR-0032 Decision 5b). Read by
    /// [`Self::synthetic_concern_targets`].
    ///
    /// Unlike `method_dups` / `variable_dups`, [`Self::update`] does *not*
    /// carry this over incrementally: `synthetic_concern_targets`'s result
    /// order drives `TypeChecker::visit_def_node`'s per-target check loop,
    /// which streams per-file diagnostics straight to output with no
    /// re-sort downstream (unlike `method_dups`/`variable_dups`, whose
    /// consumers always re-sort). An `except`-owner carry-over would
    /// reorder a bucket's tail by `FxHashSet` iteration order instead of
    /// `a_iter` position, so `update()`'s output could diverge from a
    /// cold build's for the same env — breaking ADR-0032 Decision 1's
    /// cache-on/off idempotence contract. A full rescan costs one more
    /// `class_decls()` walk per `update()` call, no worse than what
    /// `from_environment` already pays once.
    synthetic_concern_index: FxHashMap<(Symbol, MethodKind), Vec<SyntheticConcernTarget>>,
}

impl DefinitionBuilder {
    /// Build a `DefinitionBuilder` from a frozen [`Environment`].
    ///
    /// Allocates empty lazy caches; no `Definition` is built here.
    /// Per ADR-0020, all definition work is deferred to the first call
    /// of `build_instance` / `build_singleton` / `build_interface`.
    ///
    /// A lightweight member-only dup scan runs at construction so that
    /// `DuplicatedMethodDefinition` diagnostics remain complete even for
    /// classes that are never actually touched during type checking.
    pub fn from_environment(env: Arc<Environment>) -> Self {
        let ancestor_builder = Arc::new(AncestorBuilder::new(Arc::clone(&env)));
        let lowering = LoweringEnv::from_environment(&env, ancestor_builder.types());
        let (method_dups, alias_cycles) =
            scan_method_builder_diagnostics(&env, &lowering, ancestor_builder.types());
        let variable_dups = scan_variable_dups(&env);
        let synthetic_concern_index = scan_synthetic_concern_index(&env);
        let constant_resolver = ConstantResolver::new(Arc::clone(&ancestor_builder));
        let lowering_maps = Arc::clone(&lowering.maps);

        Self {
            env,
            ancestor_builder,
            constant_resolver,
            lowering_maps,
            instance_definition_cache: RefCell::new(FxHashMap::default()),
            singleton_definition_cache: RefCell::new(FxHashMap::default()),
            interface_definition_cache: RefCell::new(FxHashMap::default()),
            subtype_cache: RefCell::new(FxHashMap::default()),
            expand_alias_cache: RefCell::new(FxHashMap::default()),
            normalize_receiver_cache: RefCell::new(FxHashMap::default()),
            type_params: TypeParamsCache::default(),
            method_dups,
            alias_cycles,
            variable_dups,
            synthetic_concern_index,
        }
    }

    /// Port of `RBS::DefinitionBuilder#update` (`lib/rbs/definition_builder.rb:1018-1034`,
    /// ADR-0028 Decision 1 / S4). Builds a `DefinitionBuilder` for `env`
    /// (a new generation of a previously-built `Environment`) that carries
    /// over every cached entry not in `except`, instead of recomputing the
    /// whole environment the way [`Self::from_environment`] does.
    ///
    /// `except` is the invalidated `TypeName` set — e.g.
    /// [`crate::environment::invalidation::invalidated_names`]'s
    /// `type_names` field. `ancestor_builder` must already reflect `env`
    /// (typically `self.ancestor_builder().update(env, except)` —
    /// see that method's doc for why crema, unlike rbs, ports an
    /// `AncestorBuilder` update at all).
    ///
    /// Three fields are *not* carried over, each for a different reason:
    /// - `subtype_cache` / `expand_alias_cache` / `normalize_receiver_cache`:
    ///   forbidden outright (ADR-0028 Decision 3 — no persistence for a
    ///   cache keyed by `Ty`, since crema has no "changed `TypeName` ->
    ///   affected `Ty` keys" reverse index to invalidate by).
    /// - `type_params`: rbs has no correspondent (crema-only convenience
    ///   cache read via [`TypeParamsCache::get_or_compute`]). Its backing
    ///   [`crate::snapshot::append_map::AppendMap`] is insert-only by
    ///   safety invariant (see that type's doc) with no removal op, so a
    ///   partial except-aware carry is not implementable without touching
    ///   that invariant; starting fresh is the safe choice and cheap
    ///   (on-demand, same as a first build).
    /// - `constant_resolver`: rbs's `Resolver::ConstantResolver` has no
    ///   `update` either (checked: absent from
    ///   `lib/rbs/resolver/constant_resolver.rb`), but a fresh rebuild
    ///   would be an O(env) `ConstantTable::new` walk on every changed
    ///   run — instead the table is delta-updated via
    ///   [`ConstantTable::update`] (a performance layer outside the rbs-port contract,
    ///   layer, same framing as F4c): only owners touched by `except` /
    ///   `constants` recompute, lookup semantics are unchanged.
    ///
    /// `lowering_maps` is *not* recomputed here: `ancestor_builder` (per
    /// the contract above) has already delta-patched it via
    /// [`AncestorBuilder::update`](crate::definition::ancestor_builder::AncestorBuilder::update)
    /// (ADR-0028 S4b), so `ancestor_builder.lowering()` is reused
    /// verbatim — recomputing here would be the same O(env) work twice
    /// for an identical result (both start from the same `env`).
    pub fn update(
        &self,
        env: Arc<Environment>,
        except: &FxHashSet<TypeName>,
        constants: &FxHashSet<TypeName>,
        ancestor_builder: Arc<AncestorBuilder>,
    ) -> Self {
        let types = ancestor_builder.types();
        let lowering = ancestor_builder.lowering();

        let mut method_dups = self.method_dups.without(except);
        let mut alias_cycles = self.alias_cycles.without(except);
        let mut variable_dups = self.variable_dups.without(except);
        for &owner in except {
            if let Some(entry) = env.class_decls().get(&owner) {
                let (_, _, dups, cycles) =
                    extract_class_or_module_methods(&env, &lowering, types, &owner, entry);
                method_dups.push(owner, dups);
                alias_cycles.push(owner, cycles);

                let (instance_variables, singleton_instance_variables) =
                    extract_class_or_module_variable_dup_candidates(&env, &owner, entry);
                let mut owner_var_dups = Vec::new();
                collect_variable_dups_from_map(
                    &instance_variables,
                    &mut owner_var_dups,
                    env.names(),
                );
                collect_variable_dups_from_map(
                    &singleton_instance_variables,
                    &mut owner_var_dups,
                    env.names(),
                );
                variable_dups.push(owner, owner_var_dups);
            } else if let Some(entry) = env.interface_decls().get(&owner) {
                let (_, dups, cycles) =
                    extract_interface_methods(&env, &lowering, types, &owner, entry);
                method_dups.push(owner, dups);
                alias_cycles.push(owner, cycles);
            }
        }
        // Full rescan, not an incremental carry-over — see the field doc
        // comment on `synthetic_concern_index` for why.
        let synthetic_concern_index = scan_synthetic_concern_index(&env);

        let instance_definition_cache =
            carry_over_definitions(&self.instance_definition_cache, except);
        let singleton_definition_cache =
            carry_over_definitions(&self.singleton_definition_cache, except);
        let interface_definition_cache =
            carry_over_definitions(&self.interface_definition_cache, except);

        let constant_resolver = ConstantResolver::from_updated(
            Arc::clone(&ancestor_builder),
            self.constant_resolver
                .table()
                .update(&self.env, &env, except, constants),
        );
        let lowering_maps = Arc::clone(&lowering.maps);

        Self {
            env,
            ancestor_builder,
            constant_resolver,
            lowering_maps,
            instance_definition_cache,
            singleton_definition_cache,
            interface_definition_cache,
            subtype_cache: RefCell::new(FxHashMap::default()),
            expand_alias_cache: RefCell::new(FxHashMap::default()),
            normalize_receiver_cache: RefCell::new(FxHashMap::default()),
            type_params: TypeParamsCache::default(),
            method_dups,
            alias_cycles,
            variable_dups,
            synthetic_concern_index,
        }
    }

    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// The shared frozen environment handle itself — for callers that
    /// need an owned `Arc` (e.g. `AncestorGraph::new`) over the same env
    /// this builder reads.
    pub fn env_arc(&self) -> &Arc<Environment> {
        &self.env
    }

    pub fn ancestor_builder(&self) -> &AncestorBuilder {
        &self.ancestor_builder
    }

    /// rbs `Resolver::ConstantResolver`. Constant-name lookup against
    /// the lexical scope and ancestor chain, with cache parity.
    pub fn constant_resolver(&self) -> &ConstantResolver {
        &self.constant_resolver
    }

    /// Look up a previously stored subtype-check result. Returns `None`
    /// on cache miss; the caller is expected to compute the result and
    /// feed it back via [`Self::store_subtype_result`]. Mirrors
    /// `Steep::Subtyping::Cache#[]` (`subtyping/cache.rb:15`).
    pub(crate) fn cached_subtype_result(&self, key: &SubtypeCacheKey) -> Option<bool> {
        self.subtype_cache.borrow().get(key).copied()
    }

    /// Memoize the result of a subtype check. The relation is
    /// deterministic given the key inputs, so a repeated write for the
    /// same key is idempotent. Mirrors `Steep::Subtyping::Cache#[]=`
    /// (`subtyping/cache.rb:20`).
    pub(crate) fn store_subtype_result(&self, key: SubtypeCacheKey, result: bool) {
        self.subtype_cache.borrow_mut().insert(key, result);
    }

    /// Cache-state inspector for [`expand_alias`] tests. Returns the
    /// previously-stored expansion of `ty`, or `None` if `expand_alias`
    /// has not been invoked on this input yet.
    #[cfg(test)]
    pub(crate) fn cached_expand_alias(&self, ty: Ty) -> Option<Ty> {
        self.expand_alias_cache.borrow().get(&ty).copied()
    }

    /// Cache-state inspector for [`normalize_receiver`] tests. Returns
    /// the previously-stored normalization of `ty`, or `None` if
    /// `normalize_receiver` has not been invoked on this input yet.
    #[cfg(test)]
    pub(crate) fn cached_normalize_receiver(&self, ty: Ty) -> Option<Ty> {
        self.normalize_receiver_cache.borrow().get(&ty).copied()
    }

    /// Instance-side `Definition` for a class / module.
    ///
    /// Mirrors `RBS::DefinitionBuilder#build_instance`. On-demand
    /// memoized (ADR-0020): first call builds and caches; subsequent
    /// calls return the cached `Arc`. Returns `None` for unknown names.
    pub fn build_instance(&self, name: &TypeName) -> Option<Arc<Definition>> {
        if let Some(arc) = self.instance_definition_cache.borrow().get(name).cloned() {
            return Some(arc);
        }
        self.build_class_pair(name).map(|(inst, _)| inst)
    }

    /// Singleton-side `Definition` for a class / module.
    ///
    /// Mirrors `RBS::DefinitionBuilder#build_singleton`. On-demand
    /// memoized; shares build work with `build_instance` so calling
    /// either side populates both caches for that `TypeName`.
    pub fn build_singleton(&self, name: &TypeName) -> Option<Arc<Definition>> {
        if let Some(arc) = self.singleton_definition_cache.borrow().get(name).cloned() {
            return Some(arc);
        }
        self.build_class_pair(name).map(|(_, sing)| sing)
    }

    /// `Definition` for an interface.
    ///
    /// Mirrors `RBS::DefinitionBuilder#build_interface`. On-demand
    /// memoized (ADR-0020).
    pub fn build_interface(&self, name: &TypeName) -> Option<Arc<Definition>> {
        if let Some(arc) = self.interface_definition_cache.borrow().get(name).cloned() {
            return Some(arc);
        }
        let entry = self.env.interface_decls().get(name)?;
        let lowering = self.make_lowering_env();
        let arc = Arc::new(build_one_interface_definition(
            &self.env,
            &self.ancestor_builder,
            &lowering,
            &self.type_params,
            name,
            entry,
        ));
        self.interface_definition_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&arc));
        Some(arc)
    }

    /// Look up an instance-side method definition owned by `name` (no
    /// ancestor walk). Shields callers from the `Arc<Definition>` +
    /// `FxHashMap<Symbol, Method>` access shape so resolver code only
    /// references `Method`.
    fn own_instance_method(&self, name: &TypeName, method: Symbol) -> Option<Method> {
        self.build_instance(name)
            .and_then(|d| d.methods.get(&method).cloned())
    }

    /// Singleton-side counterpart of [`Self::own_instance_method`].
    fn own_singleton_method(&self, name: &TypeName, method: Symbol) -> Option<Method> {
        self.build_singleton(name)
            .and_then(|d| d.methods.get(&method).cloned())
    }

    /// Interface counterpart of [`Self::own_instance_method`]. Used by
    /// the instance walker as a fallback when a mixed-in interface owns
    /// the method.
    fn own_interface_method(&self, name: &TypeName, method: Symbol) -> Option<Method> {
        self.build_interface(name)
            .and_then(|d| d.methods.get(&method).cloned())
    }

    /// Look up `method` on the chain entry `name`, naming `includer` as
    /// the implementer when the entry turns out to be a mixed-in
    /// interface.
    ///
    /// rbs settles this at build time: `define_instance` recurses per
    /// type and `import_methods` stamps that type's *direct* interface
    /// includes with `implemented_in: module_name`
    /// (`definition_builder.rb:671-678`). crema walks a linearized
    /// chain at lookup time instead, where an interface entry no longer
    /// records who pulled it in — so the walker tracks the nearest
    /// non-interface ancestor it has passed and hands it in here. The
    /// linearization places an included interface after the type that
    /// includes it, which is what makes that the same type rbs names.
    fn own_method_at_chain_entry(
        &self,
        name: &TypeName,
        method: Symbol,
        includer: Option<TypeName>,
    ) -> Option<Method> {
        self.own_instance_method(name, method).or_else(|| {
            self.own_interface_method(name, method).map(|mut m| {
                if let Some(includer) = includer {
                    m.stamp_interface_implementer(includer);
                }
                m
            })
        })
    }

    /// True when `name` is an interface entry, so the instance walkers
    /// can keep their "nearest non-interface ancestor" cursor for
    /// [`Self::own_method_at_chain_entry`] without duplicating the
    /// decl-table probe.
    fn is_interface_entry(&self, name: &TypeName) -> bool {
        self.declared_kind_by_type_name(name) == Some(DeclKindLocal::Interface)
    }

    /// Whether `chain` is walked from a class that also `prepend`s an
    /// interface, deciding the seed of the walkers' includer cursor.
    /// See [`Self::own_method_at_chain_entry`] for what the cursor is
    /// for.
    fn seed_includer<'a>(&self, chain: impl Iterator<Item = &'a Ancestor>) -> Option<TypeName> {
        chain.into_iter().find_map(|a| match a {
            Ancestor::Instance { name, .. } if !self.is_interface_entry(name) => Some(*name),
            _ => None,
        })
    }

    /// Walk a module entry's `OneAncestors.self_types` and try
    /// `lookup_instance_method` on each. Returns `None` when `name`
    /// is not a module entry (so callers can chain via `.or_else`
    /// without an entry-kind check).
    ///
    /// `skip` is the set of TypeNames already present in the caller's
    /// ancestor chain. Self_types whose name appears there are bypassed:
    /// the chain walk visits them directly, so re-entering through the
    /// self_type fallback would loop on common builtin shapes like
    /// `class Object include Kernel` + `module Kernel : Object`.
    /// Visibility is preserved because the same name will be lookup-ed
    /// during the chain walk it belongs to.
    ///
    /// Reproduces rbs's recursive `define_instance` merge of self_type
    /// methods (lib/rbs/definition_builder.rb:233-247 + 85-116) at
    /// lookup time, so crema can keep the own-only invariant on
    /// `Definition.methods` while delivering the same visibility (per
    /// the `high_self_type_consumer_port` design decision: fix point
    /// C, lookup-time sub-walk).
    fn try_self_types_instance_method(
        &self,
        name: &TypeName,
        method: Symbol,
        skip: &FxHashSet<TypeName>,
    ) -> Option<Method> {
        if self.declared_kind_by_type_name(name) != Some(DeclKindLocal::Module) {
            return None;
        }
        let one = self.ancestor_builder.one_instance_ancestors_arc(name);
        // Iterate in reverse: rbs's build_instance line 233-247 merges
        // `entry.self_types.each { definition.methods.merge!(...) }`, and
        // `Hash#merge!` overwrites colliding keys with the latter entry's
        // value, so `module M : A, B` resolves to B over A. Reverse walk
        // here returns the later self_type first, matching that precedence.
        for self_type in one.self_types.iter().rev() {
            if skip.contains(&self_type.name) {
                continue;
            }
            // Interned IDs carry no kind field, so the historical
            // "coerce-to-Class then probe interface_decls to recover
            // the kind" dance is gone: the spelling-derived ID lands in
            // whichever decl table declared it, and
            // `lookup_instance_method`'s chain walker tries
            // `own_interface_method` for each entry anyway.
            if let Some(m) = self.lookup_instance_method(&self_type.name, method) {
                return Some(m);
            }
        }
        // Steep `type_construction.rb:388-413` builds the module body's
        // self type as `AST::Types::Intersection.build([Object, *self_types, M])`.
        // The self_types loop above covers the middle terms; the Object
        // base must be tried as a final fallback so Kernel methods stay
        // visible when self_types are interface-only (rbs's
        // `module_self_types_or_default` excludes Object whenever the
        // user wrote any explicit self_type). `chain_names` skip handles
        // Object/Kernel cycle shapes.
        let object = &self.names().builtins().object.clone();
        if !skip.contains(object)
            && let Some(m) = self.lookup_instance_method(object, method)
        {
            return Some(m);
        }
        None
    }

    /// `args`-aware counterpart of [`Self::try_self_types_instance_method`].
    /// Each `MixinRef` carries pre-resolved `args` in the module's own
    /// type-param scope (e.g. `module M[T] : Foo[T]` stores `[T]` where
    /// `T` is M's class param). The caller hands in `anc_args` — the
    /// args under which the module was reached as an ancestor — and we
    /// substitute M's params before recursing into Foo, so `include M[Integer]`
    /// resolves Foo's `T` to `Integer`.
    fn try_self_types_instance_method_with_args(
        &self,
        name: &TypeName,
        anc_args: &[Ty],
        method: Symbol,
        skip: &FxHashSet<TypeName>,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        if self.declared_kind_by_type_name(name) != Some(DeclKindLocal::Module) {
            return None;
        }
        let one = self.ancestor_builder.one_instance_ancestors_arc(name);
        let bindings = self.ancestor_bindings(name, anc_args);
        let subst = Substitution::from_mapping(bindings);
        // Same "later wins" iteration as the args-free helper.
        for self_type in one.self_types.iter().rev() {
            if skip.contains(&self_type.name) {
                continue;
            }
            let st_args: Vec<Ty> = self_type
                .args
                .iter()
                .map(|t| subst.apply(*t, self.ancestor_builder.types()))
                .collect();
            // Re-dispatch through `lookup_instance_method_with_args` so
            // alias resolution + overload accumulation are shared with
            // the direct-lookup path. `st_args` already substitutes the
            // module's own params (e.g. `module M[U] : _IFace[U]` carries
            // `[U]` in `self_type.args`, mapped by `subst` above), so
            // generic interface params resolve correctly. No kind
            // recovery needed — the interned ID is spelling-derived.
            if let Some(found) =
                self.lookup_instance_method_with_args(&self_type.name, &st_args, method)
            {
                return Some(found);
            }
        }
        // Steep `Object & ...self_types & M` Intersection's Object base.
        // See `try_self_types_instance_method` for the full rationale.
        // `Object` carries no type params so the args slice is empty;
        // bindings come back empty too.
        let object = &self.names().builtins().object.clone();
        if !skip.contains(object)
            && let Some(found) = self.lookup_instance_method_with_args(object, &[], method)
        {
            return Some(found);
        }
        None
    }

    /// Build and cache both instance and singleton sides for a class or module.
    /// Always populates both caches together so a subsequent call to
    /// either `build_instance` or `build_singleton` finds a cache hit.
    ///
    /// For class entries, the singleton side runs a final typed-`.new`
    /// injection (see [`Self::bake_typed_new`]) that mirrors
    /// `RBS::DefinitionBuilder#build_singleton` (lib/rbs/definition_builder.rb:343-447).
    /// The instance Arc is cached *before* the injection so the synthesis
    /// can call `lookup_instance_method_with_args(name, …, :initialize)`
    /// without re-entering this same `build_class_pair`.
    fn build_class_pair(&self, name: &TypeName) -> Option<(Arc<Definition>, Arc<Definition>)> {
        let entry = self.env.class_decls().get(name)?;
        let lowering = self.make_lowering_env();
        let (instance_def, mut singleton_def, _dups) = build_one_class_definition(
            &self.env,
            &self.ancestor_builder,
            &lowering,
            &self.type_params,
            name,
            entry,
        );
        let instance_arc = Arc::new(instance_def);
        self.instance_definition_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&instance_arc));

        if matches!(entry, ClassOrModule::Class(_)) {
            self.bake_typed_new(name, &mut singleton_def);
        }

        let singleton_arc = Arc::new(singleton_def);
        self.singleton_definition_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&singleton_arc));
        Some((instance_arc, singleton_arc))
    }

    /// Port of `RBS::DefinitionBuilder#build_singleton`'s typed `.new`
    /// injection (lib/rbs/definition_builder.rb:343-447). When the class
    /// has no own `def self.new` and no ancestor has shadowed the chain-
    /// flattened `Class#new` with a typed override, synthesise a typed
    /// `.new` from `#initialize` and insert it into the singleton table.
    ///
    /// crema-specific: rbs's `definition.methods` is chain-flattened by
    /// `build_singleton0`, so rbs reads `methods[:new]` and checks
    /// `defs.all? { d.defined_in == ::Class }` directly. crema keeps
    /// own-only tables, so the same check is reproduced by walking
    /// ancestor chain *starting from index 1* — the entry at index 0 is
    /// `Singleton { name: self }`, which would re-enter the in-flight
    /// `build_class_pair(self)` if we routed it through
    /// [`Self::lookup_singleton_method`].
    fn bake_typed_new(&self, name: &TypeName, singleton_def: &mut Definition) {
        let new_sym = self.env.names().intern_symbol("new");

        // Own `def self.new` (or anything that landed in the own singleton
        // table) wins outright. rbs's `methods[:new]` would already be the
        // non-`::Class` override in that case.
        if singleton_def.methods.contains_key(&new_sym) {
            return;
        }

        // Walk chain[1..] for an inherited `:new`. The rbs check is
        // `defs.all? { defined_in == ::Class }` — true when the only thing
        // in the chain is the untyped `Class#new` from core.rbs. crema does
        // not load core.rbs and bakes a synthesised typed_new at every
        // class entry (`MemberRef::Synthesized`), so the inherited entry
        // can either be the rbs core declaration *or* a parent's already-
        // baked synth. Both are "no real override"; only a chain entry
        // with a non-synthesised member (a real `def self.new` /
        // `attr_*` / Ruby `def`) blocks synthesis.
        if let Some(inherited) = self.lookup_singleton_method_skip_self(name, new_sym) {
            let class_tn = &self.env.names().builtins().class;
            let is_real_override = inherited.defs.iter().any(|td| {
                !matches!(td.member, MemberRef::Synthesized) && &td.defined_in != class_tn
            });
            if is_real_override {
                return;
            }
        }

        let class_type_params = self.class_type_params_by_type_name(name);
        let args =
            type_params_as_variable_args(class_type_params, name, self.ancestor_builder.types());
        let instance_ty = self.ancestor_builder.types().intern(Type::ClassInstance {
            name: *name,
            args: args.clone(),
        });
        let init_sym = self.env.names().intern_symbol("initialize");
        let synth = match self.lookup_instance_method_with_args(name, &args, init_sym) {
            Some((init, bindings)) => {
                // BasicObject#initialize non-inheritance (soutaro-approved
                // rule, mirror of `lookup_method_target`'s narrowing on the
                // type-checker side): when the resolved `#initialize` came
                // entirely from `BasicObject#initialize: () -> void` AND the
                // target class declares an inline unannotated
                // `def initialize`, fall through to `synthesize_untyped_new`
                // instead of baking the `() -> void` into a zero-arg `.new`.
                // Empty classes (no inline `def initialize`) keep the
                // `synthesize_new_from_initialize` path so `Foo.new(1)` still
                // reports `UnexpectedPositionalArgument` (the true positive
                // on an empty class must survive). Annotated inline initialize
                // and typed intermediate ancestors do not hit this branch
                // because their `defined_in` is not `::BasicObject`.
                if defs_all_from_basic_object(ConsultationView::new(self, None), &init)
                    && class_has_inline_def_initialize(self, name)
                {
                    synthesize_untyped_new(instance_ty, *name)
                } else {
                    // Apply chain bindings (e.g. `Base.T -> Integer` for
                    // `Child < Base[Integer]`) before lowering. The baked
                    // overload lives at the receiver's own singleton table,
                    // where the chain walk's `Singleton { name: receiver }`
                    // arm returns no bindings — so the substitution has to
                    // happen here, not at the call site.
                    //
                    // `with_self_type(instance_ty)` ports rbs PR #2831:
                    // `DefinitionBuilder#build_singleton` runs `map_type`
                    // over the full method signature so any
                    // `Types::Bases::Self` inside `#initialize` (block
                    // params, args, return) is rewritten to the receiver's
                    // `ClassInstance`. `class` / `instance` keywords stay
                    // untouched — PR #2831 scoped the rewrite to
                    // `Bases::Self` only.
                    let subst = Substitution::from_mapping(bindings).with_self_type(instance_ty);
                    let substituted_defs =
                        substitute_method_defs(&init.defs, &subst, self.ancestor_builder.types());
                    let substituted_init = Method::from_defs(substituted_defs, init.accessibility);
                    synthesize_new_from_initialize(
                        instance_ty,
                        &substituted_init,
                        class_type_params.map(Vec::as_slice).unwrap_or(&[]),
                    )
                }
            }
            None => synthesize_untyped_new(instance_ty, *name),
        };
        singleton_def.methods.insert(new_sym, synth);
    }

    /// Same chain walk as [`Self::lookup_singleton_method`] but skips
    /// index 0 (which is `Singleton { name: class }`). Used only by
    /// [`Self::bake_typed_new`] during the in-flight build of `class`'s
    /// own pair, where the cache for `class` does not yet hold a singleton
    /// `Arc` and routing through `lookup_singleton_method` would re-enter
    /// `build_class_pair(class)`. Discards the bindings tuple — the gate
    /// only cares about `defs[*].member` / `defs[*].defined_in`.
    fn lookup_singleton_method_skip_self(
        &self,
        class: &TypeName,
        method: Symbol,
    ) -> Option<Method> {
        let chain = self.ancestor_builder.singleton_ancestors(class);
        self.walk_singleton_chain(
            chain.ancestors.iter().filter(
                |ancestor| !matches!(ancestor, Ancestor::Singleton { name } if name == class),
            ),
            method,
            None,
        )
        .map(|(m, _)| m)
    }

    /// Resolve an unresolved [`crate::ast::types::Type`] (as produced by
    /// `ast_builder::build_node_type_assertion`) into an interned `Ty`.
    /// Used by the type checker when consuming inline `#: T` annotations
    /// at expression sites, where lowering must happen after the
    /// environment is frozen.
    ///
    /// `context` is the lexical class/module nesting at the assertion
    /// site, formatted the same way `type_builder::build_type` expects:
    /// each entry is the absolute `Name` of one enclosing scope
    /// (`Some(::A)`, `Some(::A::B)`). Inline assertions parse fresh AST
    /// at check time, so unlike the frozen-environment lowering path
    /// (which sees AST already canonicalized to absolute paths), the
    /// resolver here must walk this stack to canonicalize relative
    /// names like `Inner` against the enclosing namespace.
    ///
    /// `scope` carries any in-scope type parameters; pass
    /// [`TypeParamScope::new`] when no class/method type params apply.
    pub fn lower_ast_type(
        &self,
        ast_ty: &crate::ast::types::Type,
        context: &[Option<Name>],
        scope: &crate::type_param::TypeParamScope,
    ) -> Ty {
        let lowering = self.make_lowering_env();
        crate::definition_builder::type_builder::build_type(
            ast_ty,
            context,
            &lowering.maps,
            lowering.names,
            lowering.types,
            scope,
        )
    }

    /// Construct a call-frame `LoweringEnv` by sharing the pre-built `Arc`.
    /// Cheap: one `Arc::clone` call plus borrows of `names` and `types`.
    fn make_lowering_env(&self) -> LoweringEnv<'_> {
        LoweringEnv::from_maps(
            Arc::clone(&self.lowering_maps),
            self.env.names(),
            self.ancestor_builder.types(),
        )
    }

    /// Expand a type alias to its body, substituting `args` for the
    /// alias's type parameters. Mirrors rbs `Environment#type_alias_decls`
    /// direct access + on-the-fly lowering (ADR-0019: no cache layer).
    ///
    /// Returns `None` when `name` is not declared as a type alias, or when
    /// `args.len() != params.len()` (mirrors rbs `expand_alias2` raise).
    pub fn expand_type_alias(&self, name: &TypeName, args: &[Ty]) -> Option<Ty> {
        let entry = self.env.type_alias_decls().get(name)?;
        let lowering = self.make_lowering_env();
        let class_scope = build_class_param_scope(&entry.decl.type_params, name);
        // Bindings must use the same scope that the lowered body sees —
        // `build_class_param_scope` keys references with `Class(name)`, so
        // we mint params under the same scope to keep the substitution
        // lookup live (ADR-0023).
        let params: Vec<TypeVarKey> = entry
            .decl
            .type_params
            .iter()
            .map(|tp| {
                class_scope
                    .get(&tp.name)
                    .cloned()
                    .unwrap_or_else(|| TypeVarKey::new(tp.name, TypeVarScope::Free))
            })
            .collect();
        if params.len() != args.len() {
            return None;
        }
        let body = lowering.build_type(&entry.decl.ty, &class_scope);
        if params.is_empty() {
            return Some(body);
        }
        let bindings: FxHashMap<TypeVarKey, Ty> = params
            .iter()
            .zip(args.iter())
            .map(|(param, &arg)| (param.clone(), arg))
            .collect();
        let subst = Substitution::from_mapping(bindings);
        Some(subst.apply(body, self.ancestor_builder.types()))
    }

    /// Look up a constant's lowered `Ty` by `TypeName`. On-the-fly lowering
    /// of the raw `Constant` decl held on `Environment` (ADR-0019).
    ///
    /// Returns `None` when `name` is not declared as a constant.
    pub fn lookup_constant_type(&self, name: &TypeName) -> Option<Ty> {
        let entry = self.env.constant_decls().get(name)?;
        let lowering = self.make_lowering_env();
        Some(lowering.build_type(&entry.decl.ty, &TypeParamScope::default()))
    }

    /// Deprecated-annotation message for a declared constant. Same
    /// shape as [`global_deprecated_message`](Self::global_deprecated_message).
    pub fn constant_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let entry = self.env.constant_decls().get(name)?;
        deprecated_annotation(&entry.decl.annotations, self.env.names())
    }

    /// Look up a global's lowered `Ty` by name string (`"$stdout"` etc).
    /// On-the-fly lowering of the raw `Global` decl held on `Environment`
    /// (ADR-0019). Returns `None` when `name` is not declared as a global.
    pub fn lookup_global(&self, name_str: &str) -> Option<Ty> {
        let n = self.env.names().lookup_symbol(name_str)?;
        let entry = self.env.global_decls().get(&n)?;
        let lowering = self.make_lowering_env();
        Some(lowering.build_type(&entry.decl.ty, &TypeParamScope::default()))
    }

    /// Deprecated-annotation message for a declared global, if any.
    /// `None` covers both "not declared" and "declared but not
    /// deprecated"; `Some(None)` is deprecated with no message; `Some(Some(msg))`
    /// carries the message text.
    pub fn global_deprecated_message(&self, sym: Symbol) -> Option<Option<String>> {
        let entry = self.env.global_decls().get(&sym)?;
        deprecated_annotation(&entry.decl.annotations, self.env.names())
    }

    /// Deprecated-annotation message for a declared class/module, scanning
    /// annotations across all of its `context_decls` (own declaration
    /// sites, not ancestors). Same `Option<Option<String>>` shape as
    /// [`global_deprecated_message`](Self::global_deprecated_message).
    pub fn class_or_module_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let entry = self.env.class_decls().get(name)?;
        let annotations: Vec<Annotation> = match entry {
            ClassOrModule::Class(c) => c
                .context_decls()
                .iter()
                .flat_map(|(_, _, decl)| match decl {
                    ClassDeclaration::Signature(s) => s.annotations.clone(),
                    ClassDeclaration::Ruby(_) => Vec::new(),
                })
                .collect(),
            ClassOrModule::Module(m) => m
                .context_decls()
                .iter()
                .flat_map(|(_, _, decl)| match decl {
                    ModuleDeclaration::Signature(s) => s.annotations.clone(),
                    ModuleDeclaration::Ruby(_) => Vec::new(),
                })
                .collect(),
        };
        deprecated_annotation(&annotations, self.env.names())
    }

    /// Deprecated-annotation message for a declared class/module alias
    /// (`class New = Old`). Same shape as
    /// [`class_or_module_deprecated_message`](Self::class_or_module_deprecated_message).
    pub fn class_alias_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let entry = self.env.class_alias_decls().get(name)?;
        let annotations: Vec<Annotation> = match entry {
            ClassOrModuleAliasEntry::Class(e) => match e.decl() {
                ClassAliasDeclaration::Signature(s) => s.annotations.clone(),
                ClassAliasDeclaration::Ruby(_) => Vec::new(),
            },
            ClassOrModuleAliasEntry::Module(e) => match e.decl() {
                ModuleAliasDeclaration::Signature(s) => s.annotations.clone(),
                ModuleAliasDeclaration::Ruby(_) => Vec::new(),
            },
        };
        deprecated_annotation(&annotations, self.env.names())
    }

    /// Duplicate method definitions collected during class-cache build.
    /// Each entry is `(method_name, location_of_duplicate)`. Used by the
    /// build-layer validator to emit `DuplicatedMethodDefinition` diagnostics.
    /// Flat, insertion-order view of every duplicate-method finding.
    /// Collected on demand from the per-owner cache; kept as an owned
    /// `Vec` so callers that want `.len()` / `[i]` / `.iter()` can use
    /// them uniformly. Per-owner lookup goes through
    /// [`Self::method_dups_for`].
    pub fn method_dups(&self) -> Vec<MethodDupEntry> {
        self.method_dups
            .iter()
            .flat_map(|(_, e)| e.to_vec())
            .collect()
    }

    /// Duplicate methods for a single owner. `None` for owners the
    /// scan never saw a dup on (i.e. every well-behaved class). Used
    /// by consumers that want to hand-craft per-name views (tests,
    /// and future `update(except:)`).
    #[allow(dead_code)]
    pub(crate) fn method_dups_for(&self, owner: &TypeName) -> Option<Vec<MethodDupEntry>> {
        self.method_dups.get(owner).map(|v| v.to_vec())
    }

    /// Recursive-alias cycles detected during the same construction-time
    /// scan that populates [`Self::method_dups`]. Used by the build-layer
    /// validator to emit `RecursiveAliasDefinition` diagnostics.
    pub(crate) fn alias_cycles(&self) -> Vec<AliasCycleEntry> {
        self.alias_cycles
            .iter()
            .flat_map(|(_, e)| e.to_vec())
            .collect()
    }

    pub fn variable_dups(&self) -> Vec<VariableDuplication> {
        self.variable_dups
            .iter()
            .flat_map(|(_, e)| e.to_vec())
            .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn variable_dups_for(&self, owner: &TypeName) -> Option<Vec<VariableDuplication>> {
        self.variable_dups.get(owner).map(|v| v.to_vec())
    }

    /// ADR-0032 Decision 5b: injection-target owners for a synthetic
    /// ActiveSupport::Concern `def` (index-backed replacement for
    /// `TypeChecker::synthetic_method_context_targets`'s old per-`def`-node
    /// full A-layer walk). `(method, kind)` selects the candidate bucket;
    /// `location` / `source_file` narrow it to the exact `def` (the same
    /// concern `def` can inject into several classes, all sharing one
    /// source location — `source_file: None` on a candidate means "any
    /// file", mirroring the old `Option::is_none_or` match). `current` is
    /// excluded: a concern module never synthesizes onto itself. Preserves
    /// the original A-layer iteration order (candidates are pushed in
    /// `a_iter` order at construction) and dedups by owner, matching the
    /// old walk's `seen: FxHashSet` guard.
    pub(crate) fn synthetic_concern_targets(
        &self,
        method: Symbol,
        kind: MethodKind,
        location: PrismByteRange,
        source_file: Option<Name>,
        current: Option<TypeName>,
    ) -> Vec<TypeName> {
        let Some(candidates) = self.synthetic_concern_index.get(&(method, kind)) else {
            return Vec::new();
        };
        let mut seen = FxHashSet::default();
        let mut targets = Vec::new();
        for candidate in candidates {
            if Some(candidate.owner) == current || candidate.location != location {
                continue;
            }
            if let Some(file) = candidate.source_file
                && Some(file) != source_file
            {
                continue;
            }
            if seen.insert(candidate.owner) {
                targets.push(candidate.owner);
            }
        }
        targets
    }

    /// Shortcut for `self.env().names()`. Free functions reach the
    /// `NameTable` through the receiver, so they keep doing so here.
    pub fn names(&self) -> &NameTable {
        self.env.names()
    }

    /// Shortcut for `self.ancestor_builder().types()`. Same reason as
    /// [`names`](Self::names).
    pub fn types(&self) -> &TypeTable {
        self.ancestor_builder.types()
    }

    /// Mirror of legacy `DefinitionBuilder::declared_kind(&str)` — class
    /// / module / interface lookup against the frozen environment.
    pub fn declared_kind(&self, class_name: &str) -> Option<DeclKindLocal> {
        let n = self.names().lookup(class_name)?;
        let tn = self.declared_type_name_by_name(n)?;
        self.declared_kind_by_type_name(&tn)
    }

    /// `TypeName`-keyed variant of [`declared_kind`](Self::declared_kind).
    /// Class / module aliases (`module YAML = Psych`) are normalized
    /// through the frozen environment's alias table so the alias resolves
    /// to the target's kind, mirroring legacy `DefinitionBuilder::declared_kind`.
    pub fn declared_kind_by_type_name(&self, tn: &TypeName) -> Option<DeclKindLocal> {
        if let Some(com) = self.env.class_decls().get(tn) {
            return Some(match com {
                ClassOrModule::Class(_) => DeclKindLocal::Class,
                ClassOrModule::Module(_) => DeclKindLocal::Module,
            });
        }
        if self.env.interface_decls().contains_key(tn) {
            return Some(DeclKindLocal::Interface);
        }
        if self.env.class_alias_decls().contains_key(tn) {
            let normalized = self.env.normalize_module_name(tn);
            if &normalized != tn {
                return self.declared_kind_by_type_name(&normalized);
            }
        }
        None
    }

    /// Flat existence check against `class_decls` (no alias fallback,
    /// unlike [`declared_kind_by_type_name`](Self::declared_kind_by_type_name)).
    /// Kept as its own table-scoped query: ADR-0032 Decision 2's
    /// recording view records one consulted key per table, so a caller
    /// that only cares about one table shouldn't also produce
    /// consultation records for the others.
    pub fn is_declared_class(&self, name: &TypeName) -> bool {
        self.env.class_decls().contains_key(name)
    }

    /// Flat existence check against `class_alias_decls` (no target
    /// normalization, unlike `declared_kind_by_type_name`'s alias
    /// fallback). Same table-granularity reasoning as
    /// [`is_declared_class`](Self::is_declared_class).
    pub fn is_declared_class_alias(&self, name: &TypeName) -> bool {
        self.env.class_alias_decls().contains_key(name)
    }

    /// Existence check against `interface_decls`.
    pub fn is_declared_interface(&self, name: &TypeName) -> bool {
        self.env.interface_decls().contains_key(name)
    }

    /// Existence check against `type_alias_decls`.
    pub fn is_declared_type_alias(&self, name: &TypeName) -> bool {
        self.env.type_alias_decls().contains_key(name)
    }

    /// Existence check against `constant_decls`.
    pub fn is_declared_constant(&self, name: &TypeName) -> bool {
        self.env.constant_decls().contains_key(name)
    }

    /// Class/module tag for a `class_decls` entry — a flat lookup with
    /// no interface check and no alias-normalization fallback (unlike
    /// [`declared_kind_by_type_name`](Self::declared_kind_by_type_name)).
    /// Safe wherever the caller already knows `name` is (or would be) a
    /// direct class/module declaration — e.g. the class currently being
    /// type-checked — and just needs the Class-vs-Module tag.
    pub fn class_or_module_kind(&self, name: &TypeName) -> Option<DeclKindLocal> {
        match self.env.class_decls().get(name)? {
            ClassOrModule::Class(_) => Some(DeclKindLocal::Class),
            ClassOrModule::Module(_) => Some(DeclKindLocal::Module),
        }
    }

    /// Convert a flat `Name` to the `TypeName` under which a class /
    /// module / interface is declared. Mirrors legacy
    /// `DefinitionBuilder::declared_type_name_by_name`. Returns `None`
    /// when no declaration is found in the frozen environment.
    pub fn declared_type_name_by_name(&self, name: Name) -> Option<TypeName> {
        let s = self.names().resolve(name);
        let class_tn = self.names().parse_type_name(&s);
        if self.env.class_decls().contains_key(&class_tn)
            || self.env.class_alias_decls().contains_key(&class_tn)
        {
            return Some(class_tn);
        }
        let interface_tn = self.names().parse_type_name(&s);
        if self.env.interface_decls().contains_key(&interface_tn) {
            return Some(interface_tn);
        }
        None
    }

    pub fn declared_type_name_by_type_name(&self, name: TypeName) -> Option<TypeName> {
        if self.env.class_decls().contains_key(&name)
            || self.env.class_alias_decls().contains_key(&name)
            || self.env.interface_decls().contains_key(&name)
        {
            return Some(name);
        }

        self.declared_type_name_by_name(self.names().intern(&self.names().resolve(name)))
    }

    /// `Name`-in / `Name`-out wrapper around the frozen environment's
    /// `TypeName`-based normalize. Folds the `class X = Y` alias chain.
    pub fn normalize_module_name(&self, name: Name) -> Name {
        let tn = self.names().parse_type_name(&self.names().resolve(name));
        let normalized = self.env.normalize_module_name(&tn);
        self.names().intern(&self.names().resolve(normalized))
    }

    /// Resolved type parameters of a class / module / interface, keyed
    /// by flat `Name`. Defers to [`class_type_params_by_type_name`].
    pub fn class_type_params_by_name(&self, name: Name) -> Option<&Vec<TypeParam>> {
        let tn = self.declared_type_name_by_name(name)?;
        self.class_type_params_by_type_name(&tn)
    }

    /// The lazy type-params cache, for building a `validator::AncestryEnv`
    /// that shares this builder's resolution state.
    pub(crate) fn type_params_cache(&self) -> &TypeParamsCache {
        &self.type_params
    }

    /// Resolved type parameters keyed by `TypeName`. Class / module aliases
    /// resolve through the alias target before consulting the declaration map.
    /// Mirrors legacy `DefinitionBuilder::class_type_params_by_type_name`.
    pub fn class_type_params_by_type_name(&self, name: &TypeName) -> Option<&Vec<TypeParam>> {
        let normalized = self.env.normalize_module_name(name);
        self.type_params
            .get_or_compute(&self.env, &self.make_lowering_env(), &normalized)
    }

    /// Build a class instance type with no type arguments from a fully-
    /// qualified [`TypeName`]. Convenience over
    /// `types().class_instance(name)` for callers holding a
    /// `DefinitionBuilder` reference.
    pub fn class_instance_type(&self, name: TypeName) -> Ty {
        self.types().class_instance(name)
    }

    /// Internal: pull the direct (1-step) superclass `TypeName` from
    /// the `OneAncestors` view, ignoring the singleton arm because
    /// `superclass_*` callers all want the instance chain's super.
    fn direct_superclass_name(&self, class_name: &TypeName) -> Option<TypeName> {
        let one_instance = self.ancestor_builder.one_instance_ancestors_arc(class_name);
        match one_instance.super_class.as_ref()? {
            Ancestor::Instance { name, .. } => Some(*name),
            Ancestor::Singleton { .. } => None,
        }
    }

    /// Method names required by an interface. Mirrors legacy
    /// `interface_method_names_by_name`. Returns an owned `Vec` because
    /// the type-view `Definition` carries `methods: FxHashMap`, not a
    /// declaration-order `Vec<Symbol>` field.
    pub fn interface_method_names_by_name(&self, name: Name) -> Option<Vec<Symbol>> {
        let tn = self.names().parse_type_name(&self.names().resolve(name));
        self.interface_method_names_by_type_name(&tn)
    }

    /// TypeName-keyed variant of [`interface_method_names_by_name`].
    pub fn interface_method_names_by_type_name(&self, name: &TypeName) -> Option<Vec<Symbol>> {
        self.build_interface(name)
            .map(|d| d.methods.keys().copied().collect())
    }

    /// Return the superclass `TypeName` of `class_name`, or `None` when
    /// no superclass is recorded (e.g. `::BasicObject`).
    pub fn superclass_type_name(&self, class_name: &TypeName) -> Option<TypeName> {
        self.direct_superclass_name(class_name)
    }

    /// Whether the superclass chain rooted at `class_name` reaches
    /// `::BasicObject` (or a `::Module` terminal for modules). Mirrors
    /// legacy `has_complete_ancestor_chain`. Used by the type checker to
    /// gate `NoMethod` diagnostics — incomplete chains imply the user
    /// did not load every signature, so the absence of a method is not
    /// reliable evidence of a real error.
    pub fn has_complete_ancestor_chain(&self, class_name: &TypeName) -> bool {
        use rustc_hash::FxHashSet;
        let basic_object = &self.names().builtins().basic_object;
        let module_class = &self.names().builtins().module;
        let mut current = self.env.normalize_module_name(class_name);
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(current) {
                return true;
            }
            let Some(kind) = self.declared_kind_by_type_name(&current) else {
                return false;
            };
            match self.direct_superclass_name(&current) {
                Some(sup) => current = sup,
                None => {
                    if &current == basic_object {
                        return true;
                    }
                    if kind == DeclKindLocal::Module {
                        current = *module_class;
                        continue;
                    }
                    return false;
                }
            }
        }
    }

    /// Walk the instance ancestor chain of `class` and resolve `method`,
    /// honoring the `def foo: ...` overload-extension trailer.
    ///
    /// When the first ancestor that owns `method` has `overloading: true`,
    /// the walk continues and appends each subsequent ancestor's
    /// same-name overloads onto the result, stopping at the first
    /// ancestor whose own definition has `overloading: false` (the
    /// canonical method declaration). The order is leaf-first: own
    /// overloads come first, then each parent's overloads in walk order
    /// — matching rbs's `Method#members` merge.
    ///
    /// Returns `None` when no ancestor owns `method`. The returned
    /// `Method` carries the leaf's `visibility` / `source`;
    /// only `overloads` accumulates across the chain. Args
    /// applied to the receiver are *not* substituted here — callers
    /// (type checker) hold the receiver's `Ty` and run the
    /// substitution themselves.
    pub fn lookup_instance_method(&self, class: &TypeName, method: Symbol) -> Option<Method> {
        let chain = self
            .ancestor_builder
            .instance_ancestors(class)
            .apply(&[], self.ancestor_builder.types());
        // Pre-collect every Instance-arm name in this chain. The self_types
        // fallback skips names already present here, so cycles like
        // `Object includes Kernel` + `module Kernel : Object` (Ruby's
        // canonical builtin shape) don't loop -- the chain walk visits
        // the name directly anyway.
        let chain_names: FxHashSet<TypeName> = chain
            .iter()
            .filter_map(|a| match a {
                Ancestor::Instance { name, .. } => Some(*name),
                _ => None,
            })
            .collect();
        let mut acc: Option<Method> = None;
        // Nearest non-interface ancestor passed so far — the implementer
        // a mixed-in interface's methods are attributed to. Seeded with
        // the first non-interface entry rather than `None` because a
        // `prepend`ed entry linearizes in *front* of self, so the walk
        // can meet an interface before it has passed any class. Without
        // the seed such a method would keep `implemented_in: None` and
        // no query could name the call site.
        let mut includer: Option<TypeName> = self.seed_includer(chain.iter());
        for ancestor in &chain {
            let Ancestor::Instance { name, .. } = ancestor else {
                continue;
            };
            // The chain mixes class / module entries (`build_instance`) and
            // interface entries (`build_interface`, when an `include
            // _Comparable`-style mixin is in source). Both must be looked up:
            // skipping interfaces would silently lose method resolution for
            // any signature method declared on a mixed-in interface.
            //
            // self_types are intentionally NOT consulted here; they fire
            // only as the chain-final fallback below so include's methods
            // win over the included module's self_type (rbs `merge!`
            // precedence in build_instance line 233-247 + 250).
            if !self.is_interface_entry(name) {
                includer = Some(*name);
            }
            let m = self.own_method_at_chain_entry(name, method, includer);
            let Some(mut m) = m else {
                continue;
            };
            if m.is_unresolved_alias() {
                // Unresolved alias placeholder produced by
                // `flush_bucket_via_sorter`. Re-walk this same chain
                // starting from the alias-declaring class with the
                // alias's `old_name`, then stamp owner = declaring class
                // on the target overloads. Mirrors rbs `define_method`
                // lines 700-721. Silent skip on missing target (ADR-0010);
                // `continue` (not `?`) so a missing target only hides
                // the alias entry — the ancestor walk continues and a
                // same-named method on a deeper ancestor remains visible.
                let am = m
                    .alias_member
                    .clone()
                    .expect("is_unresolved_alias guarantees alias_member is Some");
                let Some(target) = self.lookup_instance_method(name, am.old_name) else {
                    continue;
                };
                m = self.build_resolved_alias(&target, &am, name);
            }
            let is_overloading = m.is_overloading();
            match acc.as_mut() {
                None => acc = Some(m),
                Some(prev) => {
                    // Merge child + parent annotations and re-apply
                    // rbs step C across all defs (lookup-time equivalent
                    // of rbs `define_method` lines 891-967).
                    let parent_anns = m.annotations;
                    prev.defs.extend(m.defs);
                    prev.drop_unannotated_placeholder_defs();
                    merge_inherited_annotations(prev, parent_anns);
                }
            }
            if !is_overloading {
                return acc;
            }
        }
        // Phase 2: chain walk found nothing in own/interface tables. Try
        // each module ancestor's self_types as the final fallback (rbs
        // build_instance line 233-247: self_types methods land in
        // `definition.methods` first, then own/include `merge!` over
        // them, so they show up only when own/include miss).
        // Walk chain in forward order: the receiver's own self_types
        // come first, then include's, matching the merge-order from
        // shallowest to deepest in rbs.
        if acc.is_none() {
            for ancestor in &chain {
                let Ancestor::Instance { name, .. } = ancestor else {
                    continue;
                };
                if let Some(m) = self.try_self_types_instance_method(name, method, &chain_names) {
                    return Some(m);
                }
            }
        }
        acc
    }

    /// Build the resolved alias Method to return from the lookup walk.
    /// Mirrors rbs `define_method` lines 711-723: copy the target's
    /// `defs` while rewriting `defined_in` to the alias-declaring class
    /// (and `implemented_in` to the same when the declaring decl
    /// actually implements; interfaces only declare, so they keep
    /// `implemented_in: None`), prefer special-accessibility on the
    /// alias new_name (e.g. `initialize`), else inherit the target's
    /// accessibility, and link `alias_of` / `alias_member` /
    /// `annotations` so downstream consumers see the rbs-aligned shape.
    fn build_resolved_alias(
        &self,
        target: &Method,
        alias_member: &Arc<AliasMember>,
        declaring_class: &TypeName,
    ) -> Method {
        let is_interface = matches!(
            self.declared_kind_by_type_name(declaring_class),
            Some(DeclKindLocal::Interface)
        );
        let implemented_in = if is_interface {
            None
        } else {
            Some(*declaring_class)
        };
        // rbs `define_method` line 715-723: `defs.map { defn.update(...) }`
        // preserves the target's per-overload annotations through
        // `defn.update` (TypeDef::update at definition.rb:73-78). The alias's
        // `Method.annotations` is set to the alias's own annotations (line
        // 723), and step C (lines 966-968) then uniformly overwrites
        // every TypeDef's `member_annotations` with that value.
        let defs: Vec<TypeDef> = target
            .defs
            .iter()
            .map(|td| TypeDef {
                type_: td.type_.clone(),
                member: td.member.clone(),
                defined_in: *declaring_class,
                implemented_in,
                member_annotations: alias_member.annotations.clone(),
                overload_annotations: td.overload_annotations.clone(),
            })
            .collect();
        let is_instance = alias_member.kind == AliasKind::Instance;
        let accessibility = special_accessibility(is_instance, alias_member.new_name, self.names())
            .unwrap_or(target.accessibility);
        Method {
            defs,
            accessibility,
            super_method: None,
            alias_of: Some(Arc::new(target.clone())),
            alias_member: Some(Arc::clone(alias_member)),
            annotations: alias_member.annotations.clone(),
        }
    }

    /// Materialize the `T -> Ty` bindings contributed by an ancestor:
    /// zip the ancestor's class-level type params with the
    /// receiver-applied `anc_args`. Returns an empty map when the
    /// ancestor declares no type params, which the lookup walks use as
    /// the "no substitution needed" signal to skip the substitute path.
    fn ancestor_bindings(&self, name: &TypeName, anc_args: &[Ty]) -> FxHashMap<TypeVarKey, Ty> {
        self.class_type_params_by_type_name(name)
            .map(|params| {
                params
                    .iter()
                    .zip(anc_args.iter())
                    .map(|(p, &ty)| (p.name.clone(), ty))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Look up an instance method with applied receiver args, returning
    /// the cloned `Method` together with the bindings in scope at the
    /// level where it was found.
    ///
    /// `args` are the type arguments at the call site (e.g. `[String]`
    /// for `Array[String].new.push(x)`). [`InstanceAncestors::apply`]
    /// folds them across the linearized chain once at the root, so the
    /// per-ancestor `args` in the returned chain are already in the
    /// ancestor's own type-param scope -- we just zip them with the
    /// ancestor's params to rebuild the bindings.
    ///
    /// The bindings are pinned at the first ancestor that owns
    /// `method`; subsequent overload entries along the chain extend
    /// `defs` but do not change the binding scope, matching the
    /// args-free [`Self::lookup_instance_method`] ascent rule.
    pub fn lookup_instance_method_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self
            .ancestor_builder
            .instance_ancestors(class)
            .apply(args, self.ancestor_builder.types());
        self.resolve_instance_method_in_chain(&chain, method)
    }

    /// Interface counterpart of [`Self::lookup_instance_method_with_args`].
    ///
    /// `interface_ancestors` returns the same [`InstanceAncestors`] shape
    /// as the class side, with each chain element being an
    /// `Ancestor::Instance` carrying the interface name and its applied
    /// args. The shared [`Self::resolve_instance_method_in_chain`] already
    /// falls back to `own_interface_method` when `own_instance_method`
    /// misses, so the walk reuses without modification.
    ///
    /// Mirrors Steep `Interface::Builder#object_shape` for the
    /// `AST::Types::Name::Interface` arm.
    pub fn lookup_interface_method_with_args(
        &self,
        interface: &TypeName,
        args: &[Ty],
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self
            .ancestor_builder
            .interface_ancestors(interface)
            .apply(args, self.ancestor_builder.types());
        self.resolve_instance_method_in_chain(&chain, method)
    }

    /// Resolve `method` by walking `chain`, an already-`apply`-substituted
    /// instance ancestor linearization. Factored out of
    /// [`Self::lookup_instance_method_with_args`] so `super` resolution can
    /// pass a chain slice starting after the defining class. Passing the
    /// full chain reproduces the pre-refactor behavior exactly.
    ///
    /// `chain_names` (the self_type cycle-break set) is computed from the
    /// passed slice, so a `super` caller's slice intentionally excludes the
    /// defining class and everything before it. That matches `super`
    /// semantics — the phase-2 self_types fallback must only reach ancestors
    /// strictly above the defining class — and stays safe for cycle-breaking
    /// because any self_type cycle that the fallback could chase lives among
    /// the slice's own ancestors.
    pub(crate) fn resolve_instance_method_in_chain(
        &self,
        chain: &[Ancestor],
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        // See `lookup_instance_method` for the rationale of pre-collecting
        // chain names to break Object/Kernel-style self_type cycles.
        let chain_names: FxHashSet<TypeName> = chain
            .iter()
            .filter_map(|a| match a {
                Ancestor::Instance { name, .. } => Some(*name),
                _ => None,
            })
            .collect();
        let mut acc: Option<(Method, FxHashMap<TypeVarKey, Ty>)> = None;
        // See `lookup_instance_method` — same cursor, same seed, same
        // reason.
        let mut includer: Option<TypeName> = self.seed_includer(chain.iter());
        for ancestor in chain {
            let Ancestor::Instance {
                name,
                args: anc_args,
                ..
            } = ancestor
            else {
                continue;
            };
            // self_types are intentionally NOT consulted here; they fire
            // only as the chain-final fallback below so include's methods
            // win over the included module's self_type (rbs `merge!`
            // precedence in build_instance line 233-247 + 250).
            if !self.is_interface_entry(name) {
                includer = Some(*name);
            }
            let m = self.own_method_at_chain_entry(name, method, includer);
            let Some(mut m) = m else {
                continue;
            };
            // Unresolved alias resolution. When this is the first method
            // found in the walk (`acc.is_none()`), the alias's resolved
            // bindings replace the alias-declaring class bindings, so
            // generics on the target ancestor (e.g. `class P[T]; def
            // foo: () -> T; class C < P[Integer]; alias bar foo`) thread
            // through to the caller. In the rare case alias is found
            // mid-accumulate, the existing pinned bindings win — the
            // alias's resolved defs go through the standard
            // `substitute_method_defs` extend path. Missing target
            // silently hides the alias entry (`continue`) instead of
            // collapsing the whole walk; deeper ancestors with the same
            // name remain reachable.
            let mut alias_bindings_override: Option<FxHashMap<TypeVarKey, Ty>> = None;
            if m.is_unresolved_alias() {
                let am = m
                    .alias_member
                    .clone()
                    .expect("is_unresolved_alias guarantees alias_member is Some");
                let Some((target, target_bindings)) =
                    self.lookup_instance_method_with_args(name, anc_args, am.old_name)
                else {
                    continue;
                };
                m = self.build_resolved_alias(&target, &am, name);
                if acc.is_none() {
                    alias_bindings_override = Some(target_bindings);
                }
            }
            let is_overloading = m.is_overloading();
            let bindings =
                alias_bindings_override.unwrap_or_else(|| self.ancestor_bindings(name, anc_args));
            match acc.as_mut() {
                None => acc = Some((m, bindings)),
                Some((prev, _)) => {
                    // Mirror rbs `defn.sub(subst)`: bind the ancestor's
                    // class-level type params into the appended defs so
                    // a `(T) -> T` overload inherited under
                    // `Child < Parent[Integer]` lands as `(Integer) -> Integer`.
                    let subst = Substitution::from_mapping(bindings);
                    let substituted =
                        substitute_method_defs(&m.defs, &subst, self.ancestor_builder.types());
                    prev.defs.extend(substituted);
                    prev.drop_unannotated_placeholder_defs();
                    merge_inherited_annotations(prev, m.annotations);
                }
            }
            if !is_overloading {
                return acc;
            }
        }
        // Phase 2: chain-final self_types fallback. See
        // `lookup_instance_method` for the rationale (rbs merge order).
        if acc.is_none() {
            for ancestor in chain {
                let Ancestor::Instance {
                    name,
                    args: anc_args,
                    ..
                } = ancestor
                else {
                    continue;
                };
                if let Some(found) = self.try_self_types_instance_method_with_args(
                    name,
                    anc_args,
                    method,
                    &chain_names,
                ) {
                    return Some(found);
                }
            }
        }
        acc
    }

    /// Resolve the `super`-target method issued from within `self_type`'s
    /// `method` definition, returning the method and its ancestor bindings
    /// like the sibling `lookup_*` entries (the caller projects the return
    /// type / selects an overload).
    ///
    /// `super` resolves the same-named method starting *after* the defining
    /// class in the linearization (Ruby/rbs semantics), so this walks the
    /// ancestor chain of `self_type`, finds the defining class's own entry,
    /// and resolves `method` in the remaining slice. `prepend` (defining
    /// class is preceded by the prepended module) and `include` (the module
    /// follows the class) both fall out of the linearization order.
    ///
    /// A module definition's `super` finds nothing: the host that
    /// `include`/`prepend`s the module is not in the module's own ancestors,
    /// and at module-body check time that host is statically unknown (any
    /// number of classes may mix the module in, each resolving `super`
    /// differently). Ruby defers the target to the use-site host; a static
    /// checker cannot, so it stays unresolved — matching Steep.
    ///
    /// `None` when `self_type` is not a concrete class/singleton, the
    /// defining class is absent from its own linearization, or no ancestor
    /// above it defines `method`; callers map `None` to `untyped`.
    pub fn lookup_super_method(
        &self,
        self_type: Ty,
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        match &self.types().resolve(self_type) {
            Type::ClassInstance { name, args } => {
                let chain = self
                    .ancestor_builder
                    .instance_ancestors(name)
                    .apply(args, self.ancestor_builder.types());
                let skip = chain
                    .iter()
                    .position(|a| matches!(a, Ancestor::Instance { name: n, .. } if n == name))?;
                self.resolve_instance_method_in_chain(&chain[skip + 1..], method)
            }
            Type::ClassSingleton { name } => {
                let chain = self.ancestor_builder.singleton_ancestors(name);
                let skip = chain
                    .ancestors
                    .iter()
                    .position(|a| matches!(a, Ancestor::Singleton { name: n } if n == name))?;
                self.walk_singleton_chain(chain.ancestors[skip + 1..].iter(), method, None)
            }
            _ => None,
        }
    }

    /// Look up a singleton method with bindings.
    ///
    /// Singleton receivers carry no type arguments, so there are no caller
    /// args to fold. Bindings come from each ancestor's own `args` field
    /// (e.g. `extend Bag[Integer]` contributes `T -> Integer`).
    ///
    /// Mirrors the structure of [`Self::lookup_instance_method_with_args`]
    /// but uses `singleton_ancestors` directly (no `apply` call) and handles
    /// both `Ancestor::Instance` (extend / super instance-flip) and
    /// `Ancestor::Singleton` arms.
    pub fn lookup_singleton_method(
        &self,
        class: &TypeName,
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self.ancestor_builder.singleton_ancestors(class);
        self.walk_singleton_chain(chain.ancestors.iter(), method, Some(class))
    }

    /// Shared body for the public [`Self::lookup_singleton_method`] and
    /// the build-time [`Self::lookup_singleton_method_skip_self`]. Callers
    /// pass a chain iterator already trimmed to the slice they want walked.
    fn walk_singleton_chain<'a>(
        &self,
        ancestors: impl Iterator<Item = &'a Ancestor>,
        method: Symbol,
        reentrant_root: Option<&TypeName>,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let mut acc: Option<(Method, FxHashMap<TypeVarKey, Ty>)> = None;
        let mut includer: Option<TypeName> = None;
        for ancestor in ancestors {
            if acc.is_some()
                && matches!(
                    (ancestor, reentrant_root),
                    (Ancestor::Singleton { name }, Some(root)) if name == root
                )
            {
                continue;
            }
            let (mut m, bindings) = match ancestor {
                Ancestor::Instance {
                    name,
                    args: anc_args,
                    ..
                } => {
                    if !self.is_interface_entry(name) {
                        includer = Some(*name);
                    }
                    let m = self.own_method_at_chain_entry(name, method, includer);
                    let Some(m) = m else { continue };
                    let bindings = self.ancestor_bindings(name, anc_args);
                    (m, bindings)
                }
                Ancestor::Singleton { name } => {
                    // `extend`ed entries land on the singleton side, so
                    // a class reached here is the implementer for any
                    // interface extended into it. Set the cursor before
                    // the `continue`: the class implements the extended
                    // interface whether or not it also defines `method`
                    // itself.
                    includer = Some(*name);
                    let m = self.own_singleton_method(name, method);
                    let Some(m) = m else { continue };
                    (m, FxHashMap::default())
                }
            };
            // Same alias resolution as the instance walker: the
            // singleton bucket shares `flush_bucket_via_sorter`, so
            // singleton-side `alias self.bar self.foo` lands here as an
            // unresolved placeholder. Resolve via the matching lookup
            // path (instance ancestor → instance method, singleton
            // ancestor → singleton method) so target overloads and
            // owner stamps mirror rbs `define_method`. Missing target
            // hides the alias entry via `continue`.
            if m.is_unresolved_alias() {
                let am = m
                    .alias_member
                    .clone()
                    .expect("is_unresolved_alias guarantees alias_member is Some");
                m = match ancestor {
                    Ancestor::Instance { name, .. } => {
                        let Some(target) = self.lookup_instance_method(name, am.old_name) else {
                            continue;
                        };
                        self.build_resolved_alias(&target, &am, name)
                    }
                    Ancestor::Singleton { name } => {
                        let Some((target, _)) = self.lookup_singleton_method(name, am.old_name)
                        else {
                            continue;
                        };
                        self.build_resolved_alias(&target, &am, name)
                    }
                };
            }
            let is_overloading = m.is_overloading();
            match acc.as_mut() {
                None => acc = Some((m, bindings)),
                Some((prev, _)) => {
                    // Same eager substitution as the instance walker:
                    // `extend Bag[Integer]` contributes `T -> Integer`,
                    // and the inherited overload must reach the type
                    // checker already bound.
                    let subst = Substitution::from_mapping(bindings);
                    let substituted =
                        substitute_method_defs(&m.defs, &subst, self.ancestor_builder.types());
                    prev.defs.extend(substituted);
                    prev.drop_unannotated_placeholder_defs();
                    merge_inherited_annotations(prev, m.annotations);
                }
            }
            if !is_overloading {
                return acc;
            }
        }
        acc
    }

    /// Look up an instance variable (`@foo`) with applied receiver args.
    ///
    /// Variable-side counterpart of [`Self::lookup_instance_method_with_args`]:
    /// [`InstanceAncestors::apply`] folds `args` across the linearized chain
    /// once at the root, so each ancestor's `args` are already in its own
    /// type-param scope. The first ancestor that owns `var` wins (variables
    /// have no overloading ascent); its bindings come from zipping that
    /// ancestor's params with the applied `args`. Interface ancestors carry
    /// no instance variables, so reading `build_instance` skips them.
    pub fn lookup_instance_variable_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self
            .ancestor_builder
            .instance_ancestors(class)
            .apply(args, self.ancestor_builder.types());
        for ancestor in variable_priority_walk(&chain) {
            let Ancestor::Instance {
                name,
                args: anc_args,
                ..
            } = ancestor
            else {
                continue;
            };
            if let Some(var_def) = self
                .build_instance(name)
                .and_then(|d| d.instance_variables.get(&var).cloned())
            {
                return Some((var_def, self.ancestor_bindings(name, anc_args)));
            }
        }
        None
    }

    /// Look up a class variable (`@@foo`). Walks the same instance-side
    /// linearized chain as [`Self::lookup_instance_variable_with_args`] but
    /// reads `class_variables`. Interface ancestors carry no class variables,
    /// so they are skipped by reading only `build_instance`.
    ///
    /// Singleton-side `Definition`s carry an empty `class_variables` in crema,
    /// so `@@foo` at a class-method `self` is routed through this instance-side
    /// walk by `resolve_cvar_at_self`.
    ///
    /// **Lookup-time subst divergence.** rbs `define_instance` inserts
    /// `ClassVariable` members with `member.type` raw, with no `Substitution`
    /// applied at populate time (`definition_builder.rb:188`). That diverges
    /// from rbs's `InstanceVariable` handling, where `define_instance`
    /// substitutes eagerly (`definition_builder.rb:148, 182`). crema applies
    /// subst at lookup time uniformly for both ivar and cvar via the returned
    /// `bindings` + caller-side `Substitution::apply`. The visible behavior is
    /// equivalent when class variables stay outside generic-arg scope (the
    /// common case Ruby invites by design: class variables share a single
    /// slot across the inheritance tree, so they rarely reference a class
    /// type parameter). For declarations that *do* reference a class type
    /// parameter, crema returns the bound type while rbs returns the raw
    /// type variable and leaves binding to downstream consumers.
    pub fn lookup_class_variable_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self
            .ancestor_builder
            .instance_ancestors(class)
            .apply(args, self.ancestor_builder.types());
        for ancestor in variable_priority_walk(&chain) {
            let Ancestor::Instance {
                name,
                args: anc_args,
                ..
            } = ancestor
            else {
                continue;
            };
            if let Some(var_def) = self
                .build_instance(name)
                .and_then(|d| d.class_variables.get(&var).cloned())
            {
                return Some((var_def, self.ancestor_bindings(name, anc_args)));
            }
        }
        None
    }

    /// Look up a class instance variable (`self.@foo`) from the singleton-side
    /// linearized chain. Mirrors the two-arm structure of
    /// [`Self::walk_singleton_chain`]:
    ///
    /// - `Ancestor::Singleton` arm: reads `build_singleton(name).instance_variables`
    ///   with empty bindings. rbs `build_singleton0` raw-merges the super_class's
    ///   `instance_variables` without applying a substitution
    ///   (`definition_builder.rb:269-279`), so a `self.@foo: T` inherited
    ///   through `class Child < Base[Integer]` reaches the caller as the
    ///   unbound type variable T, not Integer.
    /// - `Ancestor::Instance` arm: contributed by `extend M[args]` (and the
    ///   include chain expanded inside M by `singleton_ancestors`). rbs
    ///   `build_singleton0` runs `tapp_subst` for each extended module
    ///   (`definition_builder.rb:281-288`); the port mirrors that via
    ///   `ancestor_bindings(name, anc_args)`.
    pub fn lookup_class_instance_variable_with_args(
        &self,
        class: &TypeName,
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let chain = self.ancestor_builder.singleton_ancestors(class);
        for ancestor in &chain.ancestors {
            let (var_def, bindings) = match ancestor {
                Ancestor::Singleton { name } => {
                    let Some(var_def) = self
                        .build_singleton(name)
                        .and_then(|d| d.instance_variables.get(&var).cloned())
                    else {
                        continue;
                    };
                    (var_def, FxHashMap::default())
                }
                Ancestor::Instance {
                    name,
                    args: anc_args,
                    ..
                } => {
                    let Some(var_def) = self
                        .build_instance(name)
                        .and_then(|d| d.instance_variables.get(&var).cloned())
                    else {
                        continue;
                    };
                    (var_def, self.ancestor_bindings(name, anc_args))
                }
            };
            return Some((var_def, bindings));
        }
        None
    }

    /// One-shot constructor: parse RBS bytes through the loader and
    /// build a `DefinitionBuilder` from the resulting environment.
    pub fn from_rbs_source(source: &[u8]) -> Result<Self, String> {
        let mut draft = EnvironmentDraft::new();
        draft.load_rbs_source(source)?;
        let env = Arc::new(
            draft
                .build()
                .map_err(|(e, names)| format!("draft build: {}", e.format_with(&names)))?,
        );
        Ok(Self::from_environment(env))
    }

    /// Same as [`from_rbs_source`] for an on-disk `.rbs` file —
    /// Location entries record the file path.
    pub fn from_rbs_file(path: &std::path::Path) -> Result<Self, String> {
        let mut draft = EnvironmentDraft::new();
        draft.load_file(path)?;
        let env = Arc::new(
            draft
                .build()
                .map_err(|(e, names)| format!("draft build: {}", e.format_with(&names)))?,
        );
        Ok(Self::from_environment(env))
    }

    /// Same as [`from_rbs_source`] for a directory tree — recurses
    /// over every `.rbs` file and accumulates them into a single
    /// draft before building.
    pub fn from_rbs_dir(dir: &std::path::Path) -> Result<Self, String> {
        let mut draft = EnvironmentDraft::new();
        draft.load_dir(dir)?;
        let env = Arc::new(
            draft
                .build()
                .map_err(|(e, names)| format!("draft build: {}", e.format_with(&names)))?,
        );
        Ok(Self::from_environment(env))
    }
}

/// Apply `subst` to every [`MethodType`] in `defs`, returning a new
/// vector. Used both by [`DefinitionBuilder::bake_typed_new`] when
/// synthesizing a class's own `.new` from `initialize`, and by the
/// lookup-walk extend paths in
/// [`DefinitionBuilder::lookup_instance_method_with_args`] /
/// [`DefinitionBuilder::walk_singleton_chain`] where an ancestor's
/// `MethodType` carries class-level type parameters that have to be
/// bound before joining the caller's overload chain (rbs's eager
/// `defn.sub(subst)` regimen, applied at lookup time per ADR-0019).
///
/// Method-level `type_params` are preserved untouched — they live in a
/// distinct scope inferred per call site. Annotation slots
/// (`overload_annotations` / `member_annotations`) are carried through:
/// rbs's `Method::TypeDef#update` (definition.rb:73-78) preserves them
/// across `defn.update(type: ...)`, and `Method#sub` (line 167-174) /
/// `Method#map_type` (line 176-181) both route their per-def rewrite
/// through `defn.update`. crema mirrors that by cloning the source
/// annotation vectors here.
pub(crate) fn substitute_method_defs(
    defs: &[TypeDef],
    subst: &Substitution,
    types: &TypeTable,
) -> Vec<TypeDef> {
    defs.iter()
        .map(|td| {
            let type_ = subst.apply_function_type(&td.type_.type_, types);
            let block = td.type_.block.as_ref().map(|b| subst.apply_block(b, types));
            let mut new_td = TypeDef::new(
                MethodType {
                    type_params: td.type_.type_params.clone(),
                    type_,
                    block,
                },
                td.member.clone(),
                td.defined_in,
                td.implemented_in,
            );
            new_td.member_annotations = td.member_annotations.clone();
            new_td.overload_annotations = td.overload_annotations.clone();
            new_td
        })
        .collect()
}

/// Merge a parent ancestor's annotations into the running `prev`
/// `Method` during a lookup walk, then re-apply rbs step C uniformly
/// across every entry of `prev.defs`.
///
/// Mirrors rbs `define_method`'s combined effect at lookup time:
/// - lines 891 / 946 (`when DefMember` unannotated child + `when nil`
///   overloading-only child): inherit the parent's `Method.annotations`
///   by `replace` + later `concat` of overloading extras
/// - lines 966-968 (step C): every TypeDef's `member_annotations` is
///   overwritten with the final `Method.annotations`
///
/// crema's lookup walk reaches the same end state without the build-phase
/// `existing_method` fetch rbs uses — defs are already extended by the
/// caller, and this helper folds in the parent's annotations plus
/// re-applies step C across the freshly merged def set.
///
/// Annotation order is `[child, parent]` (the natural walk order) rather
/// than rbs's `[parent, child]`. `each_annotation` consumers (and crema's
/// only consumer to date, `Method::is_pure`) test set membership, so
/// the difference is invisible to callers.
fn merge_inherited_annotations(
    prev: &mut Method,
    parent_annotations: Vec<crate::ast::annotation::Annotation>,
) {
    prev.annotations.extend(parent_annotations);
    let final_ann = prev.annotations.clone();
    for td in prev.defs.iter_mut() {
        td.member_annotations = final_ann.clone();
    }
}

/// Collected duplicate method definitions: each entry is
/// `(method_name, duplicate_location, duplicate_source)`. The duplicate
/// location anchors the diagnostic (rbs `member.location`) and is `None`
/// when neither colliding member has a file identity. `duplicate_source`
/// is the other side — a real location, an infusion-synthesized origin
/// (`DuplicateSource::Synthesized`), or `None` when there is nothing left
/// to point at (see [`resolve_dup_locations`]).
pub(crate) type MethodDupEntry = (String, Option<SourceLocation>, Option<DuplicateSource>);
pub(crate) type MethodDups = Vec<MethodDupEntry>;

/// Per-owner diagnostic cache — the shape rbs's `MethodBuilder`
/// (`lib/rbs/method_builder.rb`) uses to key its result table by
/// declared class / module / interface name. `map` is the source of
/// truth for existence; `order` records the first-insertion sequence
/// so [`Self::iter`] reproduces the walk-order the flat-`Vec`
/// predecessor produced. That is what keeps
/// `Ruby::DuplicatedMethodDefinitionError` / `RecursiveAliasDefinition`
/// / `InstanceVariableDuplication` diagnostic outputs byte-identical
/// before / after the reshape.
///
/// [`Self::without`] is the per-owner "drop" operation `DefinitionBuilder::update`
/// (ADR-0028 S4) needs — it does not reuse `Hash#delete` (rbs's
/// `RBS::DefinitionBuilder#update` at `lib/rbs/definition_builder.rb:1016-1030`
/// uses `merge!` + `Hash#delete` on plain Ruby Hashes) because this
/// two-field (`map` + insertion order) shape has no O(1) "remove from
/// `order`" op; instead it rebuilds `order` by skipping the dropped
/// owners, which is the net-equivalent of merge-then-delete without the
/// two-step dance.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct PerNameCache<T> {
    map: FxHashMap<TypeName, Vec<T>>,
    order: Vec<TypeName>,
}

impl<T> PerNameCache<T> {
    pub(crate) fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            order: Vec::new(),
        }
    }

    /// Append `entries` to the owner's bucket. Empty inputs are dropped
    /// so unrelated owners do not pollute [`Self::iter`]'s owner
    /// sequence — a wall of `(owner, [])` tuples would be dead weight
    /// in emission and a distraction in per-name debugging.
    pub(crate) fn push(&mut self, owner: TypeName, entries: Vec<T>) {
        if entries.is_empty() {
            return;
        }
        match self.map.get_mut(&owner) {
            Some(existing) => existing.extend(entries),
            None => {
                self.order.push(owner);
                self.map.insert(owner, entries);
            }
        }
    }

    /// Iterate `(owner, entries)` in first-insertion order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (TypeName, &[T])> + '_ {
        self.order
            .iter()
            .filter_map(move |n| self.map.get(n).map(|v| (*n, v.as_slice())))
    }

    /// Look up a single owner's entries. `None` for owners the scan
    /// never saw a finding on.
    #[allow(dead_code)]
    pub(crate) fn get(&self, owner: &TypeName) -> Option<&[T]> {
        self.map.get(owner).map(|v| v.as_slice())
    }

    /// Copy of `self` with every `except` owner dropped. Used by
    /// [`DefinitionBuilder::update`] to carry over per-owner findings for
    /// unaffected owners; the caller then `push`es freshly re-scanned
    /// entries for the `except` owners onto the result.
    pub(crate) fn without(&self, except: &FxHashSet<TypeName>) -> Self
    where
        T: Clone,
    {
        let mut result = Self::new();
        for &owner in &self.order {
            if except.contains(&owner) {
                continue;
            }
            if let Some(entries) = self.map.get(&owner) {
                result.push(owner, entries.to_vec());
            }
        }
        result
    }
}

/// Copy of a `Definition` cache with every `except` `TypeName` dropped.
/// Shared by [`DefinitionBuilder::update`]'s three definition caches
/// (instance / singleton / interface) — each is a `TypeName -> Arc<Definition>`
/// map where a dropped entry is simply rebuilt lazily on the next
/// `build_instance` / `build_singleton` / `build_interface` call, mirroring
/// rbs's `instance_cache.delete(name)` (`definition_builder.rb:1027-1031`).
fn carry_over_definitions(
    cache: &RefCell<FxHashMap<TypeName, Arc<Definition>>>,
    except: &FxHashSet<TypeName>,
) -> RefCell<FxHashMap<TypeName, Arc<Definition>>> {
    let mut result = FxHashMap::default();
    for (name, def) in cache.borrow().iter() {
        if !except.contains(name) {
            result.insert(*name, Arc::clone(def));
        }
    }
    RefCell::new(result)
}

/// Resolve a [`MemberRef`]'s file identity and range to a
/// [`SourceLocation`]. Returns `None` when either piece is missing —
/// today that covers `Synthesized` (the crema-only `Class#new` member
/// variant) and any member whose `source_file` is unset (in-memory test
/// RBS, or an infusion-synthesized `MethodDefinition`, whose `location`/
/// `source_file` are both structurally `None`). Callers that need to
/// tell "no location for a real reason" apart from "this member is
/// crema's own infusion synthesis" go through [`member_provenance`]
/// instead, which checks the member's `DeclOrigin` first.
fn resolve_member_source(
    member: &crate::definition::MemberRef,
    names: &crate::name::NameTable,
) -> Option<SourceLocation> {
    member
        .source_file()
        .zip(member.location())
        .map(|(f, range)| SourceLocation {
            file: PathBuf::from(names.resolve(f)),
            range,
        })
}

/// Classification of one side of a duplicate-method pair, before the
/// diagnostic-facing [`DuplicateSource`] shape is picked. `Real` is a
/// real source file the user can jump to; `Synthesized` is crema's own
/// infusion output — no file to point at, but the collector is known
/// from the member's `DeclOrigin` (never guessed from the missing
/// location, so a real-file collision can never be mislabeled as
/// synthesized); `Unknown` covers every other file-less case
/// (`DeclOrigin::Unspecified`, in-memory test RBS).
#[derive(Clone)]
enum Provenance {
    Real(SourceLocation),
    Synthesized(InfusionUnit),
    Unknown,
}

impl Provenance {
    fn into_duplicate_source(self) -> Option<DuplicateSource> {
        match self {
            Provenance::Real(loc) => Some(DuplicateSource::Location(loc)),
            Provenance::Synthesized(infusion) => Some(DuplicateSource::Synthesized { infusion }),
            Provenance::Unknown => None,
        }
    }
}

/// Classify one duplicate-pair member: `origin` is checked first (an
/// infusion-synthesized decl is always `Synthesized`, regardless of
/// what `member`'s own location fields happen to hold), falling back to
/// [`resolve_member_source`] for every other origin.
fn member_provenance(
    member: &crate::definition::MemberRef,
    origin: DeclOrigin,
    names: &crate::name::NameTable,
) -> Provenance {
    if let DeclOrigin::Synthesized(infusion, _) = origin {
        return Provenance::Synthesized(infusion);
    }
    match resolve_member_source(member, names) {
        Some(loc) => Provenance::Real(loc),
        None => Provenance::Unknown,
    }
}

/// Pick the primary (diagnostic-anchoring) location and its
/// `duplicate_source` companion out of a duplicate pair. The anchor
/// always prefers a real, editable location over a synthesized one —
/// `dup` (the later-encountered member, `err.members` past index 0)
/// anchors when it is `Real`; otherwise `original` (the first-
/// encountered member) anchors if it is `Real`. This keeps the anchor
/// on the user's editable side and `duplicate_source` on the other
/// side regardless of which member the infusion collector happened to
/// insert first (the "fixed direction" invariant: which side anchors
/// never depends on insertion order).
///
/// When neither side anchors a real location, `duplicate_source` still
/// surfaces synthesis provenance if either side has it (`dup` preferred
/// on a tie) — a synthesized collision is worth naming even when there
/// is nothing to jump to.
fn resolve_dup_locations(
    dup: Provenance,
    original: Provenance,
) -> (Option<SourceLocation>, Option<DuplicateSource>) {
    if let Provenance::Real(loc) = &dup {
        return (Some(loc.clone()), original.into_duplicate_source());
    }
    if let Provenance::Real(loc) = &original {
        return (Some(loc.clone()), dup.into_duplicate_source());
    }
    let source = dup
        .into_duplicate_source()
        .or_else(|| original.into_duplicate_source());
    (None, source)
}

/// Project a [`method_builder::MethodBuilder`]'s collected dup errors
/// into [`MethodDups`]. Shared by the class/module and interface
/// collectors ([`extract_class_or_module_methods`] /
/// [`extract_interface_methods`]) — both walk the same
/// `err.members`/`err.origins` shape.
fn collect_method_dups(
    errors: &[method_builder::DuplicatedMethodDefinitionError],
    names: &crate::name::NameTable,
) -> MethodDups {
    errors
        .iter()
        .flat_map(|err| {
            let mut pairs = err.members.iter().zip(err.origins.iter());
            let original_provenance = pairs
                .next()
                .map(|(m, o)| member_provenance(m, *o, names))
                .unwrap_or(Provenance::Unknown);
            pairs.map(move |(m, o)| {
                let dup_provenance = member_provenance(m, *o, names);
                let (location, source) =
                    resolve_dup_locations(dup_provenance, original_provenance.clone());
                (names.resolve(err.method_name), location, source)
            })
        })
        .collect()
}

/// Collected recursive-alias cycles. One entry per cycle (self-loop or
/// multi-step SCC). Mirrors rbs `RecursiveAliasDefinitionError` (which
/// raises once per cycle, carrying the full `defs` array). Names are
/// pre-resolved to strings so the projection layer (`validator.rs`)
/// stays decoupled from `NameTable`.
pub(crate) type AliasCycles = Vec<AliasCycleEntry>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AliasCycleEntry {
    /// Fully-qualified owner type (e.g. `"::Foo"`, `"::_Bar"`).
    pub type_name: String,
    /// Names of every alias entry participating in the cycle.
    /// `len() == 1` for a self-loop, `>= 2` for a multi-step SCC.
    pub alias_names: Vec<String>,
    /// Location of the cycle's first alias declaration (rbs `first_def.original.location`).
    pub primary_location: Option<SourceLocation>,
}

fn extract_class_or_module_methods(
    env: &Environment,
    lowering: &LoweringEnv<'_>,
    types: &TypeTable,
    name: &TypeName,
    entry: &ClassOrModule,
) -> (
    FxHashMap<Symbol, Method>,
    FxHashMap<Symbol, Method>,
    MethodDups,
    AliasCycles,
) {
    let names = env.names();
    let owner_context = legacy_context_from_type_name(name, names);

    let primary_params: &[AstTypeParam] = match entry {
        ClassOrModule::Class(c) => ancestor_builder::primary_signature_class_type_params(c),
        ClassOrModule::Module(m) => ancestor_builder::primary_signature_module_type_params(m),
    };
    let primary_scope = build_class_param_scope(primary_params, name);

    // 1 type 1 bucket: MethodBuilder walks every reopen decl and
    // aggregates all members (including aliases) into one Methods per
    // receiver kind. AST-level subst pre-rewrites each reopen decl's
    // type params onto the primary decl's names so the bucket holds
    // primary-scope members directly.
    let mut mb = method_builder::MethodBuilder::new(env, types);
    mb.build_instance(name);
    mb.build_singleton(name);

    let dups: MethodDups = collect_method_dups(mb.errors(), names);

    let mut instance: FxHashMap<Symbol, Method> = FxHashMap::default();
    let mut singleton: FxHashMap<Symbol, Method> = FxHashMap::default();
    let mut cycles: AliasCycles = Vec::new();

    let instance_bucket = mb
        .instance_methods()
        .get(name)
        .expect("MethodBuilder::build_instance must populate the cache for `name`");
    flush_bucket_via_sorter(
        instance_bucket,
        name,
        crate::type_param::MethodKind::Instance,
        names,
        lowering,
        &primary_scope,
        &owner_context,
        None,
        Some(*name),
        &mut instance,
        &mut cycles,
    );
    let singleton_bucket = mb
        .singleton_methods()
        .get(name)
        .expect("MethodBuilder::build_singleton must populate the cache for `name`");
    flush_bucket_via_sorter(
        singleton_bucket,
        name,
        crate::type_param::MethodKind::Singleton,
        names,
        lowering,
        &primary_scope,
        &owner_context,
        None,
        Some(*name),
        &mut singleton,
        &mut cycles,
    );

    (instance, singleton, dups, cycles)
}

/// Port of `RBS::DefinitionBuilder#special_accessibility`
/// (`lib/rbs/definition_builder.rb:973-977`). Returns `Some(Private)`
/// when an instance-side alias targets one of the 5 special method
/// names that rbs always forces private; returns `None` otherwise so
/// the caller falls back to the target method's accessibility.
fn special_accessibility(is_instance: bool, name: Symbol, names: &NameTable) -> Option<Visibility> {
    if !is_instance {
        return None;
    }
    matches!(
        names.resolve(name).as_str(),
        "initialize"
            | "initialize_copy"
            | "initialize_clone"
            | "initialize_dup"
            | "respond_to_missing?"
    )
    .then_some(Visibility::Private)
}

/// Sort a 1-type bucket via [`method_builder::Sorter`] (alias DAG SCC
/// order) and project each entry into `dest`. Mirrors rbs's
/// `Methods#each` loop body: a singleton SCC's alias bucket clones the
/// target Method from `dest`; a non-alias bucket lowers via
/// [`lower_bucket_defn_to_method`] (rbs `define_method`). Cyclic SCCs
/// (size > 1) are pushed into `cycles` for `validate_alias_cycles` to
/// project into `RecursiveAliasDefinition` diagnostics — matching rbs's
/// `RBS::RecursiveAliasDefinitionError`. The bucket entry is still
/// dropped so lookup-time has nothing to recurse on.
///
/// When a bucket has `originals.len() > 1` (dup, always accompanied by a
/// `DuplicatedMethodDefinition` diagnostic), `is_alias()` uses only
/// `originals.first()`, so the lower branch is push-order sensitive.
/// rbs raises before reaching lower in this case, leaving post-dup
/// resolution undefined by spec; crema's first-wins behaviour is the
/// deliberate choice (see `mid_alias_method_same_name_collision`).
#[expect(clippy::too_many_arguments)]
fn flush_bucket_via_sorter(
    bucket: &method_builder::Methods,
    owner: &TypeName,
    method_kind: crate::type_param::MethodKind,
    names: &NameTable,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
    context: &[Option<Name>],
    accessibility_override: Option<Visibility>,
    implemented_in: Option<TypeName>,
    dest: &mut FxHashMap<Symbol, Method>,
    cycles: &mut AliasCycles,
) {
    let sorter = method_builder::Sorter::new(&bucket.methods);
    sorter.each_strongly_connected_component(|scc| {
        if scc.len() > 1 {
            // Cyclic alias chain (true SCC). rbs raises
            // `RecursiveAliasDefinitionError`; crema collects a diagnostic
            // here and still skips lowering so lookup-time has no entry
            // to recurse on. Per rbs convention, `primary_location` is
            // the first participant's alias-member location.
            let primary_location = match scc.first().and_then(|d| d.originals.first()) {
                Some(MemberRef::Alias(alias)) => {
                    alias
                        .source_file
                        .zip(alias.location)
                        .map(|(f, l)| SourceLocation {
                            file: PathBuf::from(names.resolve(f)),
                            range: l.range,
                        })
                }
                _ => None,
            };
            cycles.push(AliasCycleEntry {
                type_name: names.resolve(owner),
                alias_names: scc.iter().map(|d| names.resolve(d.name)).collect(),
                primary_location,
            });
            return;
        }
        let defn = scc[0];
        if defn.is_alias() {
            if let Some(MemberRef::Alias(alias)) = defn.originals.first() {
                // `alias foo foo` self-loop. rbs raises
                // `UnknownMethodAliasError` here (cli_test.rb:686, verified
                // via `bundle exec rbs validate` on 2026-06-01) because its
                // TSort returns the self-edge as a size-1 SCC and the
                // downstream `define_method` lookup misses the alias's own
                // entry. crema keeps the silent skip until that diagnostic
                // is ported.
                if alias.new_name == alias.old_name {
                    return;
                }
                // Resolution is deferred to lookup time. The lookup walk
                // (`DefinitionBuilder::lookup_instance_method` /
                // `_with_args`) detects the unresolved-alias shape and
                // re-walks the same ancestor chain for `old_name`,
                // unifying same-class and cross-class alias paths under
                // ADR-0009 (no eager ancestor merge).
                let is_instance = alias.kind == AliasKind::Instance;
                let accessibility = special_accessibility(is_instance, alias.new_name, names)
                    .unwrap_or_else(|| accessibility_override.unwrap_or(Visibility::Public));
                dest.insert(
                    alias.new_name,
                    Method::unresolved_alias(Arc::clone(alias), accessibility),
                );
            }
        } else {
            let method = lower_bucket_defn_to_method(
                defn,
                owner,
                method_kind,
                names,
                lowering,
                class_scope,
                context,
                accessibility_override,
                implemented_in,
            );
            dest.insert(defn.name, method);
        }
    });
}

/// Lower a single non-alias bucket entry into a [`Method`]. Mirrors
/// the per-bucket body of rbs `define_method`: when `originals` has
/// a canonical entry, build its overloads as the base; then walk
/// `overloads` (extras-only members from the `| ...` form) and
/// prepend each one's overloads. The bucket is alias-free at this
/// point — alias resolution happens in [`flush_bucket_via_sorter`].
#[expect(clippy::too_many_arguments)]
fn lower_bucket_defn_to_method(
    defn: &method_builder::methods::Definition,
    owner: &TypeName,
    method_kind: crate::type_param::MethodKind,
    names: &NameTable,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
    context: &[Option<Name>],
    accessibility_override: Option<Visibility>,
    implemented_in: Option<TypeName>,
) -> Method {
    // `next_overload_index` is the running u16 counter ADR-0023 Phase 3
    // expects: base overloads claim 0..base_len, then each extras lowering
    // continues from there so a `(T) -> T | (T, T) -> T` and a follow-up
    // extras `| (T, T, T) -> T` keep overload-index identity contiguous
    // across the lowering boundary.
    // Bucket accessibility order, mirroring rbs `define_method`'s
    // `original.visibility || special_accessibility || method.accessibility`:
    //
    // 1. `defn.accessibilities.first()` — per-method value recorded at
    //    push time. method_builder already folds
    //    `special_instance_visibility` over the push for both
    //    class-side (`push_signature_members`) and interface-side
    //    (`build_interface_inner`) paths, and attribute pushes carry
    //    explicit-vs-default through `attribute_visibility`. So a
    //    non-empty entry here is authoritative — including an explicit
    //    `Public` on an attr_reader, which must beat special.
    //
    // 2. Instance-side special fallback. `build_method` skips
    //    `accessibilities.push` when `overloading` is true, so a
    //    `def initialize: () -> void | ...` lands here with an empty
    //    bucket. Re-apply `special_instance_visibility` so the
    //    overload-only path doesn't silently fall back to the default.
    //
    // 3. `accessibility_override` — bucket-level default the interface
    //    flush path passes in (`:public`). Used when the bucket has
    //    neither a recorded accessibility nor a special-name hit.
    //
    // 4. `Visibility::Public` — last-resort default for empty buckets
    //    on the class/module path (no override).
    let bucket_accessibility = defn
        .accessibilities
        .first()
        .copied()
        .or_else(|| {
            if matches!(method_kind, crate::type_param::MethodKind::Instance) {
                special_instance_visibility(&names.resolve(defn.name))
            } else {
                None
            }
        })
        .or(accessibility_override)
        .unwrap_or(Visibility::Public);
    let (mut defs, accessibility, mut next_overload_index, mut method_annotations) =
        if let Some(orig) = defn.originals.first() {
            let base = lower_member_to_method(
                orig,
                defn.name,
                bucket_accessibility,
                owner,
                method_kind,
                0,
                names,
                lowering,
                class_scope,
                context,
                implemented_in,
            );
            let base_len = u16::try_from(base.defs.len()).unwrap_or(u16::MAX);
            // Trust `base.accessibility`: `lower_member_to_method` already
            // folds per-attribute `special_instance_visibility` for
            // AttrReader/AttrAccessor, and the Method/RubyDef branches
            // passthrough `bucket_accessibility` (computed above with
            // special precedence intact). No further override here — that
            // keeps the method path and the alias path
            // (`flush_bucket_via_sorter` line 1400) consistent: in both
            // paths special wins over `accessibility_override`.
            (base.defs, base.accessibility, base_len, base.annotations)
        } else if !defn.overloads.is_empty() {
            (Vec::new(), bucket_accessibility, 0, Vec::new())
        } else {
            return Method::from_defs(Vec::new(), bucket_accessibility);
        };
    for extras_member in &defn.overloads {
        let extras = lower_member_to_method(
            extras_member,
            defn.name,
            accessibility,
            owner,
            method_kind,
            next_overload_index,
            names,
            lowering,
            class_scope,
            context,
            implemented_in,
        );
        next_overload_index = next_overload_index
            .saturating_add(u16::try_from(extras.defs.len()).unwrap_or(u16::MAX));
        // Step B: rbs `define_method` line 963 — concat each overloading
        // extras' member-level annotations onto the bucket's running
        // Method.annotations. Each extras's per-overload annotations
        // already ride on `extras.defs` via the step-A wiring in
        // `lower_member_to_method`, so the splice carries them through.
        method_annotations.extend(extras.annotations);
        defs.splice(0..0, extras.defs);
    }
    // Step C: rbs `define_method` lines 966-968 — every TypeDef's
    // `member_annotations` is set to the final `Method.annotations`,
    // uniformly. This is the rule that ties `each_annotation`
    // (definition.rb:89-94) to `Method.annotations` across all defs.
    for td in defs.iter_mut() {
        td.member_annotations = method_annotations.clone();
    }
    let mut method = Method::from_defs(defs, accessibility);
    method.annotations = method_annotations;
    // An unannotated inline `def` for a method this class also declares
    // in `sig/` arrives as an overloading extras member and splices to
    // the front, so its `(?) -> untyped` placeholder has to go before it
    // shadows the declared signature (rbs `define_method:884` replaces
    // rather than prepends).
    method.drop_unannotated_placeholder_defs();
    method
}

fn untyped_method_type() -> MethodType {
    MethodType {
        type_params: vec![],
        type_: FunctionType::Untyped(UntypedFunction {
            return_type: Ty::UNTYPED,
        }),
        block: None,
    }
}

fn legacy_context_from_type_name(name: &TypeName, names: &NameTable) -> Vec<Option<Name>> {
    let mut context = Vec::new();
    let mut path = String::from("::");
    for (index, segment) in names.type_name_segments(*name).iter().enumerate() {
        if index > 0 {
            path.push_str("::");
        }
        path.push_str(&names.resolve(*segment));
        context.push(Some(names.intern(&path)));
    }
    context
}

/// Build a per-decl `TypeParamScope` that **alpha-renames** the decl's
/// type-param names onto the primary decl's. The resulting scope maps
/// each raw name from `decl_params` to the scoped Name of the
/// position-corresponding param in `primary_params`, so lowering a
/// secondary decl's `def foo: (B)` mints `Type::TypeVariable(primary.A)`
/// directly — no post-pass substitution required.
///
/// Arity mismatch fallback: if `decl_params` is longer than
/// `primary_params`, the extra entries map to their own scoped Name
/// (no rename target available). This shouldn't normally happen — the
/// build-phase `validate_type_params` rejects open-class arity drift —
/// but the fallback keeps lowering total instead of panicking on a
/// mismatch that slipped through.
fn build_alpha_renamed_param_scope(
    decl_params: &[AstTypeParam],
    primary_params: &[AstTypeParam],
    owner: &TypeName,
    _names: &NameTable,
) -> TypeParamScope {
    let mut scope = TypeParamScope::default();
    for (i, decl_tp) in decl_params.iter().enumerate() {
        // Reopen alpha-rename (mirrors rbs ClassEntry: the i-th param in a
        // reopened `class Foo[B]` is normalised onto the primary spelling
        // from the head decl `class Foo[A]`). The map key stays at the
        // reopen-side spelling so source references inside the reopen body
        // hit it, but the value's raw is the primary spelling so the
        // resulting `TypeVarKey` matches what `build_class_param_scope`
        // mints on the head decl.
        let target_name = match primary_params.get(i) {
            Some(primary_tp) => primary_tp.name,
            None => decl_tp.name,
        };
        scope.insert(
            decl_tp.name,
            TypeVarKey::new(target_name, TypeVarScope::Class(*owner)),
        );
    }
    scope
}

/// Map each class type-param to its `Type::TypeVariable` Ty so the
/// derived `self_type` of a Definition view carries `Foo[E]` instead
/// of `Foo[]` for generic classes. Mirrors rbs `build_instance`'s
/// `args: type_params.map { Variable.new }`. Also used by
/// [`DefinitionBuilder::bake_typed_new`] to compute the receiver's
/// instance type for the synthesised `.new`.
pub(crate) fn type_params_as_variable_args(
    type_params: Option<&Vec<TypeParam>>,
    owner: &TypeName,
    types: &TypeTable,
) -> Vec<Ty> {
    type_params
        .map(|params| {
            params
                .iter()
                .map(|tp| {
                    types.intern(Type::TypeVariable {
                        raw: tp.name.raw,
                        scope: TypeVarScope::Class(*owner),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Build a typed `.new` whose overloads copy `#initialize`'s parameters and
/// block but whose return type is rewritten to the receiver instance.
///
/// Every synthesised `TypeDef` uses `MemberRef::Synthesized` rather than
/// inheriting `#initialize`'s `member` back-ref. The marker is read by
/// [`DefinitionBuilder::bake_typed_new`] to recognise that an ancestor's
/// already-baked `:new` is NOT a real `def self.new` override — without it,
/// `class Child < Base[Integer]` would inherit `Base`'s synthesised
/// typed_new and never get its own substitution applied. The trade-off is
/// that verbose-log source lookups for `.new` no longer trace back to
/// `#initialize`'s AST location (rbs reaches that location through the
/// real `Class#new` declaration in core.rbs; crema doesn't load core.rbs
/// yet — the long-term fix lives alongside `MemberRef::Synthesized`).
fn synthesize_new_from_initialize(
    instance_ty: Ty,
    init: &Method,
    class_type_params: &[TypeParam],
) -> Method {
    let defs = init
        .defs
        .iter()
        .map(|td| {
            let type_ = match &td.type_.type_ {
                FunctionType::Typed(f) => FunctionType::Typed(Function {
                    return_type: instance_ty,
                    ..f.clone()
                }),
                FunctionType::Untyped(_) => FunctionType::Untyped(UntypedFunction {
                    return_type: instance_ty,
                }),
            };
            // Mirror rbs build_singleton (lib/rbs/definition_builder.rb:397-398):
            // class params come first, method params follow. The intersect /
            // fresh-rename branch (rbs L374-395) is unreachable here because
            // `scope_class_param` and `scope_method_param` always assign
            // distinct prefixes (`Owner@P` vs `Owner#method@P`), so the
            // raw-name collision rbs guards against cannot happen.
            let type_params: Vec<TypeParam> = class_type_params
                .iter()
                .cloned()
                .chain(td.type_.type_params.iter().cloned())
                .collect();
            let method_type = MethodType {
                type_params,
                type_,
                block: td.type_.block.clone(),
            };
            TypeDef::new(
                method_type,
                MemberRef::Synthesized,
                td.defined_in,
                td.implemented_in,
            )
        })
        .collect();
    // `.new` is the public constructor even when `#initialize` is private
    // (which it always is by Ruby / RBS convention).
    Method::from_defs(defs, Visibility::Public)
}

/// Build an untyped `.new: (?) -> receiver` used when no `#initialize` is
/// available. Matches the permissive behavior the prior `Constructor` path
/// had when `initializer: None` (accepts any args without checking).
fn synthesize_untyped_new(instance_ty: Ty, defined_in: TypeName) -> Method {
    let method_type = MethodType {
        type_params: vec![],
        type_: FunctionType::Untyped(UntypedFunction {
            return_type: instance_ty,
        }),
        block: None,
    };
    Method::from_defs(
        vec![TypeDef::new(
            method_type,
            MemberRef::Synthesized,
            defined_in,
            Some(defined_in),
        )],
        Visibility::Public,
    )
}

/// Resolve the backing-ivar Symbol for an [`Attribute`], mirroring
/// rbs's `member.ivar_name || :"@#{member.name}"` rule (and rbs's
/// `false` opt-out, which crema represents as [`IvarName::Empty`]).
fn attribute_ivar_symbol(
    ivar_name: &IvarName,
    attr_name: Symbol,
    names: &NameTable,
) -> Option<Symbol> {
    match ivar_name {
        IvarName::Unspecified => {
            let s = format!("@{}", names.resolve(attr_name));
            Some(names.intern_symbol(&s))
        }
        IvarName::Name(sym) => Some(*sym),
        IvarName::Empty => None,
    }
}

/// Port of rbs's `insert_variable`. Overwrites any existing entry,
/// but threads the previous entry through the new entry's
/// `parent_variable` field so a follow-up duplicate-validation pass
/// (`mid_variable_duplicate_validation.md`) can walk the chain and
/// raise `InstanceVariableDuplicationError`-shaped diagnostics.
fn insert_variable(
    dest: &mut FxHashMap<Symbol, Variable>,
    name: Symbol,
    ty: Ty,
    declared_in: TypeName,
    source: VariableSource,
) {
    let parent = dest.remove(&name).map(Box::new);
    dest.insert(
        name,
        Variable {
            parent_variable: parent,
            ty,
            declared_in,
            source,
        },
    );
}

#[derive(Clone, Copy)]
enum VariableTypeBuilder<'a, 'b> {
    Lowered {
        lowering: &'a LoweringEnv<'b>,
        class_scope: &'a TypeParamScope,
    },
    Untyped,
}

impl<'a, 'b> VariableTypeBuilder<'a, 'b> {
    fn lowered(lowering: &'a LoweringEnv<'b>, class_scope: &'a TypeParamScope) -> Self {
        Self::Lowered {
            lowering,
            class_scope,
        }
    }

    fn untyped() -> Self {
        Self::Untyped
    }

    fn build(self, ast_ty: &AstType) -> Ty {
        match self {
            Self::Lowered {
                lowering,
                class_scope,
            } => lowering.build_type(ast_ty, class_scope),
            Self::Untyped => Ty::UNTYPED,
        }
    }
}

impl VariableSource {
    fn is_attr_backing(&self) -> bool {
        matches!(
            self,
            VariableSource::AttrReader(_)
                | VariableSource::AttrWriter(_)
                | VariableSource::AttrAccessor(_)
                | VariableSource::RubyAttrReader(_)
                | VariableSource::RubyAttrWriter(_)
                | VariableSource::RubyAttrAccessor(_)
        )
    }

    fn location(&self) -> Option<LocationRange> {
        match self {
            VariableSource::InstanceVariable(v) => v.location.map(|l| l.range),
            VariableSource::ClassInstanceVariable(v) => v.location.map(|l| l.range),
            VariableSource::ClassVariable(v) => v.location.map(|l| l.range),
            VariableSource::AttrReader(v) => v.location.map(|l| l.range),
            VariableSource::AttrWriter(v) => v.location.map(|l| l.range),
            VariableSource::AttrAccessor(v) => v.location.map(|l| l.range),
            VariableSource::RubyAttrReader(_)
            | VariableSource::RubyAttrWriter(_)
            | VariableSource::RubyAttrAccessor(_) => None,
            VariableSource::RubyInstanceVariable(_) => None,
            VariableSource::InferredInitializeParam(_) => None,
        }
    }

    fn source_file(&self) -> Option<Name> {
        match self {
            VariableSource::InstanceVariable(v) => v.source_file,
            VariableSource::ClassInstanceVariable(v) => v.source_file,
            VariableSource::ClassVariable(v) => v.source_file,
            VariableSource::AttrReader(v) => v.source_file,
            VariableSource::AttrWriter(v) => v.source_file,
            VariableSource::AttrAccessor(v) => v.source_file,
            VariableSource::RubyAttrReader(_)
            | VariableSource::RubyAttrWriter(_)
            | VariableSource::RubyAttrAccessor(_)
            | VariableSource::RubyInstanceVariable(_)
            | VariableSource::InferredInitializeParam(_) => None,
        }
    }

    fn source_location(&self, names: &NameTable) -> Option<SourceLocation> {
        self.source_file()
            .zip(self.location())
            .map(|(f, range)| SourceLocation {
                file: PathBuf::from(names.resolve(f)),
                range,
            })
    }

    fn ruby_source_location(&self) -> Option<crate::location::RubyLocation> {
        match self {
            VariableSource::RubyInstanceVariable(v) => v.annotation.source_location,
            _ => None,
        }
    }
}

fn is_explicit_instance_variable_source(source: &VariableSource) -> bool {
    matches!(
        source,
        VariableSource::InstanceVariable(_) | VariableSource::RubyInstanceVariable(_)
    )
}

fn validate_variable_dup(
    name: Symbol,
    var: &Variable,
    names: &NameTable,
) -> Option<VariableDuplication> {
    var.parent_variable.as_ref()?;

    let mut first = None;
    let mut second = None;
    let mut current = Some(var);
    while let Some(v) = current {
        if !v.source.is_attr_backing() {
            if first.is_none() {
                first = Some(v);
            } else {
                second = Some(v);
                break;
            }
        }
        current = v.parent_variable.as_deref();
    }

    let l = first?;
    let r = second?;
    if l.declared_in != r.declared_in {
        return None;
    }

    let kind = match &l.source {
        VariableSource::InstanceVariable(_) | VariableSource::RubyInstanceVariable(_) => {
            if is_explicit_instance_variable_source(&r.source) {
                VariableDuplicationKind::Instance
            } else {
                return None;
            }
        }
        VariableSource::ClassInstanceVariable(_) => {
            if matches!(&r.source, VariableSource::ClassInstanceVariable(_)) {
                VariableDuplicationKind::ClassInstance
            } else {
                return None;
            }
        }
        _ => return None,
    };

    Some(VariableDuplication {
        kind,
        type_name: l.declared_in,
        variable_name: name,
        location: l.source.source_location(names),
        ruby_source_location: l.source.ruby_source_location(),
    })
}

fn collect_variable_dups_from_map(
    variables: &FxHashMap<Symbol, Variable>,
    dups: &mut Vec<VariableDuplication>,
    names: &NameTable,
) {
    for (&name, var) in variables {
        if let Some(dup) = validate_variable_dup(name, var, names) {
            dups.push(dup);
        }
    }
}

/// Returned by [`extract_class_or_module_variables`]:
/// `(instance_side instance_variables, singleton_side instance_variables,
///   instance_side class_variables)`.
type ExtractedVariables = (
    FxHashMap<Symbol, Variable>,
    FxHashMap<Symbol, Variable>,
    FxHashMap<Symbol, Variable>,
);

/// Walk every signature / Ruby declaration of `entry` and aggregate
/// own-class variable declarations into three buckets, matching rbs's
/// `define_instance` + `build_singleton0` variable arms:
///
/// - `@foo: T`, instance-side `attr_*` -> instance-side `instance_variables`
/// - `self.@foo: T`, singleton-side `attr_*` -> singleton-side `instance_variables`
/// - `@@foo: T` -> instance-side `class_variables`
///
/// Own-only (ADR-0009 / ADR-0019 extension): ancestor variables are
/// not merged here. [`crate::definition_builder::lookup_instance_variable_with_args`]
/// walks the linearized ancestor chain at lookup time.
fn extract_class_or_module_variables(
    env: &Environment,
    lowering: &LoweringEnv<'_>,
    name: &TypeName,
    entry: &ClassOrModule,
    instance_methods: &FxHashMap<Symbol, Method>,
) -> ExtractedVariables {
    let names = env.names();
    let primary_params: &[AstTypeParam] = match entry {
        ClassOrModule::Class(c) => ancestor_builder::primary_signature_class_type_params(c),
        ClassOrModule::Module(m) => ancestor_builder::primary_signature_module_type_params(m),
    };
    let primary_scope = build_class_param_scope(primary_params, name);

    let mut inst_vars: FxHashMap<Symbol, Variable> = FxHashMap::default();
    let mut singleton_inst_vars: FxHashMap<Symbol, Variable> = FxHashMap::default();
    let mut class_vars: FxHashMap<Symbol, Variable> = FxHashMap::default();

    match entry {
        ClassOrModule::Class(c) => {
            for (_, _, decl) in c.context_decls() {
                match decl {
                    ClassDeclaration::Signature(decl) => collect_signature_variables(
                        &decl.members,
                        &decl.type_params,
                        primary_params,
                        name,
                        names,
                        lowering,
                        &mut inst_vars,
                        &mut singleton_inst_vars,
                        &mut class_vars,
                    ),
                    ClassDeclaration::Ruby(decl) => collect_ruby_variables(
                        &decl.members,
                        VariableTypeBuilder::lowered(lowering, &primary_scope),
                        name,
                        names,
                        &mut inst_vars,
                    ),
                }
            }
        }
        ClassOrModule::Module(m) => {
            for (_, _, decl) in m.context_decls() {
                match decl {
                    ModuleDeclaration::Signature(decl) => collect_signature_variables(
                        &decl.members,
                        &decl.type_params,
                        primary_params,
                        name,
                        names,
                        lowering,
                        &mut inst_vars,
                        &mut singleton_inst_vars,
                        &mut class_vars,
                    ),
                    ModuleDeclaration::Ruby(decl) => collect_ruby_variables(
                        &decl.members,
                        VariableTypeBuilder::lowered(lowering, &primary_scope),
                        name,
                        names,
                        &mut inst_vars,
                    ),
                }
            }
        }
    }

    synthesize_initialize_param_ivars(entry, instance_methods, name, names, &mut inst_vars);

    (inst_vars, singleton_inst_vars, class_vars)
}

/// Sorbet-style instance variable inference (crema extension, no rbs
/// counterpart): synthesize a declaration for each `@ivar = param` fact
/// harvested from the Ruby `initialize` body, typed from the already
/// resolved `initialize` method type. Because methods are built before
/// variables in `build_one_class_definition`, both the inline-annotation
/// route and the RBS-file-signature route feed the same resolved
/// `MethodType` — the two are symmetric by construction.
///
/// Runs after the explicit-declaration loop and inserts only absent
/// names, so an explicit declaration (RBS `@foo: T`, inline
/// `@rbs @foo: T`, attr backing) always wins and no `parent_variable`
/// duplicate chain is ever created for inferred entries.
fn synthesize_initialize_param_ivars(
    entry: &ClassOrModule,
    instance_methods: &FxHashMap<Symbol, Method>,
    owner: &TypeName,
    names: &NameTable,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
) {
    use crate::ast::ruby::members::InitializeParamRef;

    // Exactly one Ruby `initialize` DefMember carrying facts; a reopened
    // class with two initialize bodies is ambiguous — skip entirely.
    fn initialize_candidates<'a>(
        members: &'a [RubyMember],
        candidates: &mut Vec<&'a crate::ast::ruby::members::DefMember>,
    ) {
        for member in members {
            if let RubyMember::Def(d) = member
                && d.kind == crate::ast::MethodKind::Instance
                && d.name == "initialize"
                && !d.ivar_param_pairs.is_empty()
            {
                candidates.push(d);
            }
        }
    }
    let mut candidates: Vec<&crate::ast::ruby::members::DefMember> = Vec::new();
    match entry {
        ClassOrModule::Class(c) => {
            for (_, _, decl) in c.context_decls() {
                if let ClassDeclaration::Ruby(decl) = decl {
                    initialize_candidates(&decl.members, &mut candidates);
                }
            }
        }
        ClassOrModule::Module(m) => {
            for (_, _, decl) in m.context_decls() {
                if let ModuleDeclaration::Ruby(decl) = decl {
                    initialize_candidates(&decl.members, &mut candidates);
                }
            }
        }
    }
    let [def] = candidates.as_slice() else {
        return;
    };

    // Overloaded initialize (RBS `... | ...`, or multiple defs merged)
    // has no unique param type — skip. Untyped functions carry no param
    // types at all.
    let init_sym = names.intern_symbol("initialize");
    let Some(method) = instance_methods.get(&init_sym) else {
        return;
    };
    let [type_def] = method.defs.as_slice() else {
        return;
    };
    let method_type = &type_def.type_;

    let def_arc = Arc::new((*def).clone());
    for pair in &def.ivar_param_pairs {
        let ty = match &pair.param {
            InitializeParamRef::RequiredPositional(i) => {
                method_type.required_positionals().get(*i).copied()
            }
            InitializeParamRef::OptionalPositional(i) => {
                method_type.optional_positionals().get(*i).copied()
            }
            InitializeParamRef::Keyword(kw) => method_type
                .required_keywords()
                .iter()
                .chain(method_type.optional_keywords())
                .find(|(n, _)| n == kw)
                .map(|&(_, ty)| ty),
        };
        // Arity mismatch between the Ruby def and the resolved signature
        // (or an unannotated param) yields no type — never synthesize an
        // untyped declaration (it would only swallow
        // `UnknownInstanceVariable` without adding detection power).
        let Some(ty) = ty else {
            continue;
        };
        if ty.is_untyped() {
            continue;
        }
        let ivar_sym = names.intern_symbol(&pair.ivar_name);
        if inst_vars.contains_key(&ivar_sym) {
            continue;
        }
        inst_vars.insert(
            ivar_sym,
            Variable {
                parent_variable: None,
                ty,
                declared_in: *owner,
                source: VariableSource::InferredInitializeParam(def_arc.clone()),
            },
        );
    }
}

#[expect(clippy::too_many_arguments)]
fn collect_signature_variables<M: AsMember>(
    members: &[M],
    decl_params: &[AstTypeParam],
    primary_params: &[AstTypeParam],
    owner: &TypeName,
    names: &NameTable,
    lowering: &LoweringEnv<'_>,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
    singleton_inst_vars: &mut FxHashMap<Symbol, Variable>,
    class_vars: &mut FxHashMap<Symbol, Variable>,
) {
    let class_scope = build_alpha_renamed_param_scope(decl_params, primary_params, owner, names);
    let ty_builder = VariableTypeBuilder::lowered(lowering, &class_scope);
    collect_signature_variables_with(
        members,
        owner,
        names,
        ty_builder,
        inst_vars,
        singleton_inst_vars,
        Some(class_vars),
    );
}

fn collect_signature_variables_untyped<M: AsMember>(
    members: &[M],
    owner: &TypeName,
    names: &NameTable,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
    singleton_inst_vars: &mut FxHashMap<Symbol, Variable>,
) {
    collect_signature_variables_with(
        members,
        owner,
        names,
        VariableTypeBuilder::untyped(),
        inst_vars,
        singleton_inst_vars,
        None,
    );
}

fn collect_signature_variables_with<M: AsMember>(
    members: &[M],
    owner: &TypeName,
    names: &NameTable,
    ty_builder: VariableTypeBuilder<'_, '_>,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
    singleton_inst_vars: &mut FxHashMap<Symbol, Variable>,
    mut class_vars: Option<&mut FxHashMap<Symbol, Variable>>,
) {
    for wrapper in members {
        let Some(member) = wrapper.as_member() else {
            continue;
        };
        match member {
            Member::InstanceVariable(v) => {
                let ty = ty_builder.build(&v.ty);
                insert_variable(
                    inst_vars,
                    v.name,
                    ty,
                    *owner,
                    VariableSource::InstanceVariable(Arc::new(v.clone())),
                );
            }
            Member::ClassInstanceVariable(v) => {
                let ty = ty_builder.build(&v.ty);
                insert_variable(
                    singleton_inst_vars,
                    v.name,
                    ty,
                    *owner,
                    VariableSource::ClassInstanceVariable(Arc::new(v.clone())),
                );
            }
            Member::ClassVariable(v) => {
                let Some(class_vars) = class_vars.as_deref_mut() else {
                    continue;
                };
                let ty = ty_builder.build(&v.ty);
                insert_variable(
                    class_vars,
                    v.name,
                    ty,
                    *owner,
                    VariableSource::ClassVariable(Arc::new(v.clone())),
                );
            }
            Member::AttrReader(r) => collect_attr_ivar(
                &r.ivar_name,
                r.name,
                &r.ty,
                r.kind,
                owner,
                names,
                ty_builder,
                inst_vars,
                singleton_inst_vars,
                VariableSource::AttrReader(Arc::new(r.clone())),
            ),
            Member::AttrWriter(w) => collect_attr_ivar(
                &w.ivar_name,
                w.name,
                &w.ty,
                w.kind,
                owner,
                names,
                ty_builder,
                inst_vars,
                singleton_inst_vars,
                VariableSource::AttrWriter(Arc::new(w.clone())),
            ),
            Member::AttrAccessor(a) => collect_attr_ivar(
                &a.ivar_name,
                a.name,
                &a.ty,
                a.kind,
                owner,
                names,
                ty_builder,
                inst_vars,
                singleton_inst_vars,
                VariableSource::AttrAccessor(Arc::new(a.clone())),
            ),
            _ => {}
        }
    }
}

/// Shared body for `AttrReader` / `AttrWriter` / `AttrAccessor` ivar
/// registration. The `source` variant is built by the caller from the
/// concrete attr struct so the rbs `case ... when AttrReader` / etc.
/// discrimination is preserved on the resulting [`Variable.source`].
#[allow(clippy::too_many_arguments)]
fn collect_attr_ivar(
    ivar_name: &IvarName,
    attr_name: Symbol,
    attr_ty: &crate::ast::types::Type,
    attr_kind: AttributeKind,
    owner: &TypeName,
    names: &NameTable,
    ty_builder: VariableTypeBuilder<'_, '_>,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
    singleton_inst_vars: &mut FxHashMap<Symbol, Variable>,
    source: VariableSource,
) {
    let Some(ivar) = attribute_ivar_symbol(ivar_name, attr_name, names) else {
        return;
    };
    let ty = ty_builder.build(attr_ty);
    let target = match attr_kind {
        AttributeKind::Instance => inst_vars,
        AttributeKind::Singleton => singleton_inst_vars,
    };
    insert_variable(target, ivar, ty, *owner, source);
}

/// Walk inline (Ruby-side) members and aggregate `@rbs @ivar: T`
/// annotations plus `attr_reader` / `attr_writer` / `attr_accessor`
/// backing ivars into the instance-side variable bucket.
fn collect_ruby_variables(
    members: &[RubyMember],
    ty_builder: VariableTypeBuilder<'_, '_>,
    owner: &TypeName,
    names: &NameTable,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
) {
    for member in members {
        match member {
            RubyMember::AttrReader(r) => collect_ruby_attr_ivars(
                &r.attribute,
                ty_builder,
                owner,
                names,
                inst_vars,
                VariableSource::RubyAttrReader(Arc::new(r.clone())),
            ),
            RubyMember::AttrWriter(w) => collect_ruby_attr_ivars(
                &w.attribute,
                ty_builder,
                owner,
                names,
                inst_vars,
                VariableSource::RubyAttrWriter(Arc::new(w.clone())),
            ),
            RubyMember::AttrAccessor(a) => collect_ruby_attr_ivars(
                &a.attribute,
                ty_builder,
                owner,
                names,
                inst_vars,
                VariableSource::RubyAttrAccessor(Arc::new(a.clone())),
            ),
            RubyMember::InstanceVariable(iv) => {
                let ann = &iv.annotation;
                let name_sym = names.intern_symbol(&ann.name);
                let ty = ty_builder.build(&ann.ty);
                insert_variable(
                    inst_vars,
                    name_sym,
                    ty,
                    *owner,
                    VariableSource::RubyInstanceVariable(Arc::new(iv.clone())),
                );
            }
            _ => {}
        }
    }
}

fn collect_ruby_attr_ivars(
    attribute: &crate::ast::ruby::members::AttributeMember,
    ty_builder: VariableTypeBuilder<'_, '_>,
    owner: &TypeName,
    names: &NameTable,
    inst_vars: &mut FxHashMap<Symbol, Variable>,
    source: VariableSource,
) {
    let ty = match &attribute.type_text {
        Some(type_text) => lower_ruby_attr_type_text(type_text, names, ty_builder),
        None => Ty::UNTYPED,
    };
    for name in attribute.names() {
        let ivar = format!("@{name}");
        insert_variable(
            inst_vars,
            names.intern_symbol(&ivar),
            ty,
            *owner,
            source.clone(),
        );
    }
}

fn lower_ruby_attr_type_text(
    type_text: &str,
    names: &NameTable,
    ty_builder: VariableTypeBuilder<'_, '_>,
) -> Ty {
    let Ok(ast_ty) = ruby_decl::parse_rbs_type(type_text.as_bytes(), names) else {
        return Ty::UNTYPED;
    };
    ty_builder.build(&ast_ty)
}

/// Method-side counterpart of [`lower_ruby_attr_type_text`]. Lowers an
/// inline attribute's optional `#: T` annotation into the getter/setter
/// type. `None` → `Ty::UNTYPED` (rbs/docs/inline.md §"Unannotated
/// attributes"); parse failure → `Ty::UNTYPED`. The build-side
/// re-parse is diagnostic-free; the failing parse has already been
/// surfaced by `check_attr_annotations` during load (one
/// `AnnotationSyntaxError` per attribute call), so collapsing to
/// `UNTYPED` here suppresses downstream cascades without double-emitting.
fn ruby_attr_member_lowered_type(
    attribute: &crate::ast::ruby::members::AttributeMember,
    names: &NameTable,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
) -> Ty {
    let Some(text) = attribute.type_text.as_ref() else {
        return Ty::UNTYPED;
    };
    let Ok(ast_ty) = ruby_decl::parse_rbs_type(text.as_bytes(), names) else {
        return Ty::UNTYPED;
    };
    lowering.build_type(&ast_ty, class_scope)
}

/// Build the instance-side and singleton-side [`Definition`]s for a single
/// class or module entry, plus collect any duplicate-method diagnostics for
/// that entry.
fn build_one_class_definition(
    env: &Environment,
    ancestor_builder: &AncestorBuilder,
    lowering: &LoweringEnv<'_>,
    type_params_cache: &TypeParamsCache,
    name: &TypeName,
    entry: &ClassOrModule,
) -> (Definition, Definition, MethodDups) {
    let types = ancestor_builder.types();

    let (instance_methods, singleton_methods, dups, _cycles) =
        extract_class_or_module_methods(env, lowering, types, name, entry);

    let (instance_variables, singleton_instance_variables, class_variables) =
        extract_class_or_module_variables(env, lowering, name, entry, &instance_methods);

    let instance_args = type_params_as_variable_args(
        type_params_cache.get_or_compute(env, lowering, name),
        name,
        types,
    );
    let instance_self_type = types.intern(Type::ClassInstance {
        name: *name,
        args: instance_args,
    });
    let instance_def = Definition {
        type_name: *name,
        self_type: instance_self_type,
        ancestors: OnceCell::new(),
        methods: instance_methods,
        instance_variables,
        class_variables,
    };

    let singleton_def = Definition {
        type_name: *name,
        self_type: types.class_singleton(*name),
        ancestors: OnceCell::new(),
        methods: singleton_methods,
        // rbs `build_singleton0` stores `self.@foo: T` (ClassInstanceVariable)
        // and singleton-side `attr_*`'s ivar in the singleton-side
        // `instance_variables` (definition_builder.rb:301-317).
        instance_variables: singleton_instance_variables,
        // rbs `build_singleton0` copies the instance-side
        // `class_variables` into the singleton-side via
        // `definition.class_variables.replace(instance_definition.class_variables)`
        // (definition_builder.rb:321-322) so callers reading either
        // face of `self` see the same `@@foo` map. crema diverges:
        // the instance-side Definition owns the only copy, and
        // `lookup_class_variable_with_args` walks instance-side
        // ancestors regardless of whether `self` was a
        // `ClassInstance` or a `ClassSingleton` at the call site.
        // Mapping `resolve_cvar_at_self`'s `ClassSingleton` arm
        // back through the instance side preserves the lookup
        // result without the rbs-style duplicate populate.
        class_variables: FxHashMap::default(),
    };

    (instance_def, singleton_def, dups)
}

/// Method-builder diagnostic scan run once at construction: calls
/// `extract_class_or_module_methods` for every class / module declaration
/// to collect duplicate method names AND recursive-alias cycles. Full
/// method extraction cost applies; `Definition`s themselves are not built.
///
/// ADR-0020: lazy populate builds definitions on demand, so the eager loop
/// that previously collected dups as a side effect is gone. This scan runs
/// once at `from_environment` time to keep both `method_dups` and
/// `alias_cycles` complete (matching rbs's `MethodBuilder` two-error
/// surface: `DuplicatedMethodDefinitionError` + `RecursiveAliasDefinitionError`).
fn scan_method_builder_diagnostics(
    env: &Environment,
    lowering: &LoweringEnv<'_>,
    types: &TypeTable,
) -> (PerNameCache<MethodDupEntry>, PerNameCache<AliasCycleEntry>) {
    let mut all_dups: PerNameCache<MethodDupEntry> = PerNameCache::new();
    let mut all_cycles: PerNameCache<AliasCycleEntry> = PerNameCache::new();
    match env.g_backend() {
        // The G side was scanned once at cold time and baked into the
        // snapshot; walk only the A layer and splice the baked groups,
        // skipping owners the A map redeclares — their merged entries
        // (G decls + A decls) carry both layers' decls, so keeping the
        // baked group too would double-report (ADR-0028 slice 2b).
        //
        // Ordering: cold and warm both read the groups in the order the
        // cold scan serialized them, so cold == warm holds structurally.
        // Parity with flag-off ordering is NOT guaranteed — the layered
        // walk this replaces already emitted G-side diagnostics at a
        // different position than the combined map's iteration order
        // (same multiset; verified on cfp-app 2026-07-07).
        //
        // `&env.class_decls` is the raw A-layer field on purpose:
        // `.class_decls()` would re-merge the G side we splice from the
        // baked groups below.
        Some(g) => {
            let baked = g.baked();
            for (name, entry) in env.class_decls().a_iter() {
                let (_, _, dups, cycles) =
                    extract_class_or_module_methods(env, lowering, types, name, entry);
                all_dups.push(*name, dups);
                all_cycles.push(*name, cycles);
            }
            for (owner, dups, cycles) in &baked.classes {
                if !env.a_class_contains(owner) {
                    all_dups.push(*owner, dups.clone());
                    all_cycles.push(*owner, cycles.clone());
                }
            }
            for (name, entry) in env.interface_decls().a_iter() {
                let (_, dups, cycles) =
                    extract_interface_methods(env, lowering, types, name, entry);
                all_dups.push(*name, dups);
                all_cycles.push(*name, cycles);
            }
            for (owner, dups, cycles) in &baked.interfaces {
                if !env.interface_decls().a_contains_key(owner) {
                    all_dups.push(*owner, dups.clone());
                    all_cycles.push(*owner, cycles.clone());
                }
            }
        }
        None => {
            for (name, entry) in env.class_decls() {
                let (_, _, dups, cycles) =
                    extract_class_or_module_methods(env, lowering, types, name, entry);
                all_dups.push(*name, dups);
                all_cycles.push(*name, cycles);
            }
            for (name, entry) in env.interface_decls() {
                let (_, dups, cycles) =
                    extract_interface_methods(env, lowering, types, name, entry);
                all_dups.push(*name, dups);
                all_cycles.push(*name, cycles);
            }
        }
    }
    (all_dups, all_cycles)
}

fn scan_variable_dups(env: &Environment) -> PerNameCache<VariableDuplication> {
    let names = env.names();
    let mut all_dups: PerNameCache<VariableDuplication> = PerNameCache::new();
    let mut owner_buf: Vec<VariableDuplication> = Vec::new();
    match env.g_backend() {
        Some(g) => {
            for (name, entry) in env.class_decls().a_iter() {
                let (instance_variables, singleton_instance_variables) =
                    extract_class_or_module_variable_dup_candidates(env, name, entry);
                collect_variable_dups_from_map(&instance_variables, &mut owner_buf, names);
                collect_variable_dups_from_map(
                    &singleton_instance_variables,
                    &mut owner_buf,
                    names,
                );
                all_dups.push(*name, std::mem::take(&mut owner_buf));
            }
            for (owner, dups) in &g.baked().variables {
                if !env.a_class_contains(owner) {
                    all_dups.push(*owner, dups.clone());
                }
            }
        }
        None => {
            for (name, entry) in env.class_decls() {
                owner_buf.clear();
                let (instance_variables, singleton_instance_variables) =
                    extract_class_or_module_variable_dup_candidates(env, name, entry);
                collect_variable_dups_from_map(&instance_variables, &mut owner_buf, names);
                collect_variable_dups_from_map(
                    &singleton_instance_variables,
                    &mut owner_buf,
                    names,
                );
                all_dups.push(*name, std::mem::take(&mut owner_buf));
            }
        }
    }
    all_dups
}

/// A candidate injection-target owner in [`DefinitionBuilder::synthetic_concern_index`].
/// `location` / `source_file` identify the originating concern `def` node
/// (shared across every class the concern injects into, since they are
/// copies of the same node) and are matched against a checked `def` node's
/// own location/source_file in [`DefinitionBuilder::synthetic_concern_targets`].
#[derive(Debug, Clone, Copy)]
struct SyntheticConcernTarget {
    owner: TypeName,
    location: PrismByteRange,
    source_file: Option<Name>,
}

/// Synthetic-concern index scan, run in full on every construction —
/// [`DefinitionBuilder::from_environment`] and [`DefinitionBuilder::update`]
/// alike (see the field doc comment on `synthetic_concern_index` for why
/// `update` does not carry this over incrementally like its sibling
/// caches). Walks the A-only layer and indexes every `RubyMember::Def`
/// with `DefMemberOrigin::SyntheticConcernIncluded` / `SyntheticConcernPrepended`
/// by `(method, kind)`, replacing `TypeChecker::synthetic_method_context_targets`'s
/// old per-`def`-node full A walk (ADR-0032 Decision 5b). Unlike
/// `scan_method_builder_diagnostics` / `scan_variable_dups`, there is no
/// baked-G splice branch: concerns are a Ruby-only mechanism (G is
/// `.rbs`-only by construction), so a G entry can never carry a synthetic
/// concern member — the `Some(g)` branch below only needs the A side.
fn scan_synthetic_concern_index(
    env: &Environment,
) -> FxHashMap<(Symbol, MethodKind), Vec<SyntheticConcernTarget>> {
    let names = env.names();
    let mut index: FxHashMap<(Symbol, MethodKind), Vec<SyntheticConcernTarget>> =
        FxHashMap::default();
    match env.g_backend() {
        Some(_) => {
            for (name, entry) in env.class_decls().a_iter() {
                index_synthetic_concern_defs_of(*name, entry, names, &mut index);
            }
        }
        None => {
            for (name, entry) in env.class_decls() {
                index_synthetic_concern_defs_of(*name, entry, names, &mut index);
            }
        }
    }
    index
}

/// Indexes synthetic concern `def`s declared directly on `owner` (own
/// decls only, matching the old per-`def`-node walk's `context_decls()`
/// scope). Mirrors the old `ruby_members_include_def`'s origin filter.
/// `DefMemberOrigin::SyntheticConcerning` is deliberately excluded: those
/// defs are checked lexically in their owner's instance context and need
/// no retargeting (see the variant's doc comment).
fn index_synthetic_concern_defs_of(
    owner: TypeName,
    entry: &ClassOrModule,
    names: &NameTable,
    index: &mut FxHashMap<(Symbol, MethodKind), Vec<SyntheticConcernTarget>>,
) {
    let mut push_members = |members: &[RubyMember]| {
        for member in members {
            let RubyMember::Def(def) = member else {
                continue;
            };
            if !matches!(
                def.origin,
                DefMemberOrigin::SyntheticConcernIncluded
                    | DefMemberOrigin::SyntheticConcernPrepended
            ) {
                continue;
            }
            index
                .entry((names.intern_symbol(&def.name), def.kind))
                .or_default()
                .push(SyntheticConcernTarget {
                    owner,
                    location: def.location,
                    source_file: def.source_file,
                });
        }
    };
    match entry {
        ClassOrModule::Class(class) => {
            for (_, _, decl) in class.context_decls() {
                if let ClassDeclaration::Ruby(decl) = decl {
                    push_members(&decl.members);
                }
            }
        }
        ClassOrModule::Module(module) => {
            for (_, _, decl) in module.context_decls() {
                if let ModuleDeclaration::Ruby(decl) = decl {
                    push_members(&decl.members);
                }
            }
        }
    }
}

/// G-only scan results grouped per owner, in cold-scan emission order.
/// Baked into the snapshot by `encode_g_snapshot` and spliced back by
/// the warm scans above. Only owners with at least one finding are
/// recorded — empty groups carry no ordering information the splice
/// needs, since groups splice as contiguous blocks.
pub(crate) struct RoastedGScan {
    pub(crate) classes: Vec<(TypeName, MethodDups, AliasCycles)>,
    pub(crate) interfaces: Vec<(TypeName, MethodDups, AliasCycles)>,
    pub(crate) variables: Vec<(TypeName, Vec<VariableDuplication>)>,
    /// G-only ancestor cycles, sorted by anchor name (the SCC pass emits
    /// them sorted). `participants` keys the warm-side exclusion: a cycle
    /// any of whose members the A map redeclares is re-found (possibly
    /// reshaped by the reopen) by the A-roots walk and must not splice.
    pub(crate) ancestor_cycles: Vec<BakedAncestorCycle>,
    /// G-only mixin-arity violations grouped per host, in the cold walk's
    /// emission order (class hosts, then interface hosts). Hosts the A
    /// map redeclares are excluded at splice time — the merged entry's
    /// probe re-checks both layers' mixins.
    pub(crate) arity: Vec<(TypeName, Vec<BakedArityViolation>)>,
    /// G-only class/module super-edges for infusion's AR model detection
    /// (ADR-0028 slice 2b-4). Two arms feed the same set fixpoint on the
    /// warm side, hence [`SuperEdges`]'s single `resolved` + `unresolved`
    /// struct: unlike the diagnostic groups this carries no ordering
    /// contract, and updating one arm without the other is a hazard the
    /// bundling prevents. Filled by `roast_g_super_edges` at the encode
    /// call site (like the validator fields), left empty by
    /// `roast_g_scan`.
    pub(crate) super_edges: SuperEdges,
    /// G-construction stderr warnings (stale pin, missing library, load
    /// failure, ...) — unrelated to the scan above (they're known
    /// before `env` even exists), carried here purely so
    /// `encode_g_snapshot` can bake them into the same snapshot blob.
    /// Filled at the encode call site, left empty by `roast_g_scan`.
    pub(crate) warnings: Vec<String>,
}

/// Cold-time super-edge scan output, bundled so the two arms never
/// drift out of step. Both feed the same warm-side model-detection
/// fixpoint; the distinction is where the G-only resolver could bind
/// the reference (`resolved`) versus where the reference points at an
/// A-only name and must be re-resolved after splice (`unresolved`).
#[derive(Default)]
pub(crate) struct SuperEdges {
    /// Edges the G-only resolver bound to an absolute name.
    pub(crate) resolved: Vec<(TypeName, TypeName)>,
    /// Gem super references cold could not bind. Each record carries
    /// the raw super `TypeName` plus the declaration context so the
    /// warm splice can retry against the live A ∪ G resolver (Rails-
    /// engine `class GemModel < ApplicationRecord` idiom).
    pub(crate) unresolved: Vec<BakedUnresolvedSuperEdge>,
}

/// A gem super reference cold could not resolve against G-only names.
/// Warm-side splice retries [`Self::raw_super`] against the live
/// A ∪ G resolver with [`Self::context`] as the walker's stack; success
/// contributes a bridge edge for `class`, failure preserves the raw
/// relative reference.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BakedUnresolvedSuperEdge {
    pub(crate) class: TypeName,
    pub(crate) raw_super: TypeName,
    pub(crate) context: Vec<TypeName>,
}

/// One collapsed `RecursiveAncestor` finding from the cold G-only walk:
/// the resolved diagnostic fields plus the participant set the warm
/// splice needs for cross-layer exclusion.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BakedAncestorCycle {
    pub(crate) participants: Vec<TypeName>,
    pub(crate) type_name: String,
    pub(crate) chain: Vec<String>,
    pub(crate) primary_source: Option<SourceLocation>,
}

/// One `MixinTypeArgumentArityMismatch` finding from the cold G-only walk.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BakedArityViolation {
    /// One of "superclass" / "include" / "extend" / "prepend".
    pub(crate) kind: &'static str,
    pub(crate) target: String,
    pub(crate) class: String,
    pub(crate) expected: String,
    pub(crate) got: usize,
    pub(crate) location: Option<SourceLocation>,
}

/// Run the construction-time scans over a G-only frozen environment and
/// group the results per owner (ADR-0028 slice 2b, "roast"). Cold path
/// only. The `TypeTable` is a throwaway: method extraction interns `Ty`s
/// into it, but nothing `Ty`-bearing escapes into the returned data
/// (Decision 3's persistence ban).
pub(crate) fn roast_g_scan(env: &Environment) -> RoastedGScan {
    let types = TypeTable::new();
    let lowering = LoweringEnv::from_environment(env, &types);
    let names = env.names();
    let mut classes = Vec::new();
    let mut variables = Vec::new();
    for (name, entry) in env.class_decls() {
        let (_, _, dups, cycles) =
            extract_class_or_module_methods(env, &lowering, &types, name, entry);
        if !dups.is_empty() || !cycles.is_empty() {
            classes.push((*name, dups, cycles));
        }
        let (instance_variables, singleton_instance_variables) =
            extract_class_or_module_variable_dup_candidates(env, name, entry);
        let mut var_dups = Vec::new();
        collect_variable_dups_from_map(&instance_variables, &mut var_dups, names);
        collect_variable_dups_from_map(&singleton_instance_variables, &mut var_dups, names);
        if !var_dups.is_empty() {
            variables.push((*name, var_dups));
        }
    }
    let mut interfaces = Vec::new();
    for (name, entry) in env.interface_decls() {
        let (_, dups, cycles) = extract_interface_methods(env, &lowering, &types, name, entry);
        if !dups.is_empty() || !cycles.is_empty() {
            interfaces.push((*name, dups, cycles));
        }
    }
    RoastedGScan {
        classes,
        interfaces,
        variables,
        // The validator roast needs an `AncestorBuilder` (Arc-owned env),
        // so `validator::roast_g_validators` fills these two at the
        // encode call site rather than here. `super_edges` and
        // `unresolved_super_edges` are likewise filled there by
        // `roast_g_super_edges` (it needs its own G-only resolver,
        // unrelated to the method/variable scans above).
        ancestor_cycles: Vec::new(),
        arity: Vec::new(),
        super_edges: SuperEdges::default(),
        warnings: Vec::new(),
    }
}

fn extract_class_or_module_variable_dup_candidates(
    env: &Environment,
    name: &TypeName,
    entry: &ClassOrModule,
) -> (FxHashMap<Symbol, Variable>, FxHashMap<Symbol, Variable>) {
    let names = env.names();
    let mut inst_vars: FxHashMap<Symbol, Variable> = FxHashMap::default();
    let mut singleton_inst_vars: FxHashMap<Symbol, Variable> = FxHashMap::default();

    match entry {
        ClassOrModule::Class(c) => {
            for (_, _, decl) in c.context_decls() {
                match decl {
                    ClassDeclaration::Signature(decl) => collect_signature_variables_untyped(
                        &decl.members,
                        name,
                        names,
                        &mut inst_vars,
                        &mut singleton_inst_vars,
                    ),
                    ClassDeclaration::Ruby(decl) => collect_ruby_variables(
                        &decl.members,
                        VariableTypeBuilder::untyped(),
                        name,
                        names,
                        &mut inst_vars,
                    ),
                }
            }
        }
        ClassOrModule::Module(m) => {
            for (_, _, decl) in m.context_decls() {
                match decl {
                    ModuleDeclaration::Signature(decl) => collect_signature_variables_untyped(
                        &decl.members,
                        name,
                        names,
                        &mut inst_vars,
                        &mut singleton_inst_vars,
                    ),
                    ModuleDeclaration::Ruby(decl) => collect_ruby_variables(
                        &decl.members,
                        VariableTypeBuilder::untyped(),
                        name,
                        names,
                        &mut inst_vars,
                    ),
                }
            }
        }
    }

    (inst_vars, singleton_inst_vars)
}

/// Build the [`Definition`] for a single interface entry. Mirrors
/// `RBS::DefinitionBuilder#build_interface` — `ancestors` carries the
/// linearized `interface_ancestors` (an `InstanceAncestors`).
fn build_one_interface_definition(
    env: &Environment,
    ancestor_builder: &AncestorBuilder,
    lowering: &LoweringEnv<'_>,
    type_params_cache: &TypeParamsCache,
    name: &TypeName,
    entry: &InterfaceEntry,
) -> Definition {
    let types = ancestor_builder.types();
    let (methods, _dups, _cycles) = extract_interface_methods(env, lowering, types, name, entry);
    let args = type_params_as_variable_args(
        type_params_cache.get_or_compute(env, lowering, name),
        name,
        types,
    );
    let self_type = types.intern(Type::Interface { name: *name, args });
    Definition {
        type_name: *name,
        self_type,
        ancestors: OnceCell::new(),
        methods,
        instance_variables: FxHashMap::default(),
        class_variables: FxHashMap::default(),
    }
}

/// Build the interface's `methods` map AND collect method-builder
/// diagnostics (dup methods + recursive-alias cycles) in one pass.
/// Mirrors [`extract_class_or_module_methods`] for the interface kind
/// so that both `build_one_interface_definition` (lazy populate) and
/// [`scan_method_builder_diagnostics`] (eager diagnostic pass) can
/// reuse the same per-bucket logic. Interfaces use only one bucket
/// (no instance / singleton split) because rbs `build_interface` does.
fn extract_interface_methods(
    env: &Environment,
    lowering: &LoweringEnv<'_>,
    types: &TypeTable,
    name: &TypeName,
    entry: &InterfaceEntry,
) -> (FxHashMap<Symbol, Method>, MethodDups, AliasCycles) {
    let names = env.names();
    let class_scope = build_class_param_scope(&entry.decl().type_params, name);

    let mut mb = method_builder::MethodBuilder::new(env, types);
    mb.build_interface(name);
    let bucket = mb
        .interface_methods()
        .get(name)
        .expect("MethodBuilder::build_interface must populate the cache for `name`");

    let dups: MethodDups = collect_method_dups(mb.errors(), names);

    let mut methods: FxHashMap<Symbol, Method> = FxHashMap::default();
    let mut cycles: AliasCycles = Vec::new();
    // Interfaces share the class/module flush helper. rbs `build_interface`
    // calls `build_method(accessibility: :public)` and rbs `define_method`
    // pipes `implemented_in: nil` for interfaces (declaration-only types).
    // We mirror both by passing `accessibility_override = Some(Public)`
    // and `implemented_in = None`. The `context` argument feeds the
    // RubyDef lowering branch only — interfaces never carry RubyDef
    // members (classify_member rejects them), so an empty slice is safe.
    flush_bucket_via_sorter(
        bucket,
        name,
        crate::type_param::MethodKind::Instance,
        names,
        lowering,
        &class_scope,
        &[],
        Some(Visibility::Public),
        None,
        &mut methods,
        &mut cycles,
    );
    (methods, dups, cycles)
}

/// Per-name lazy cache of resolved declaration `type_params`, keyed by
/// class / module / interface `TypeName`. Powers
/// [`DefinitionBuilder::class_type_params_by_type_name`] which Phase 5b
/// consumers consult during method-level type-arg threading.
///
/// Replaces the former eager whole-environment scan (ADR-0028 slice
/// 2b-3): resolving a G name now decodes just that snapshot entry, and
/// the resolved `Vec<TypeParam>` — which contains `Ty` and therefore
/// can never be baked into the snapshot (ADR-0028 Decision 3) — is
/// computed on first demand per run. `AppendMap` keeps the handed-out
/// `&Vec<TypeParam>` stable across later inserts, so the read API shape
/// matches the old `FxHashMap::get`.
///
/// Only the `Signature` variant contributes — the inline (Ruby) path is
/// Phase 5b territory (see [`DefinitionBuilder::type_params`] doc);
/// those names cache `None`.
#[derive(Default)]
pub(crate) struct TypeParamsCache {
    cache: crate::snapshot::append_map::AppendMap<TypeName, Option<Vec<TypeParam>>>,
}

impl std::fmt::Debug for TypeParamsCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeParamsCache")
            .field("cached", &self.cache.len())
            .finish()
    }
}

impl TypeParamsCache {
    pub(crate) fn get_or_compute(
        &self,
        env: &Environment,
        lowering: &LoweringEnv<'_>,
        name: &TypeName,
    ) -> Option<&Vec<TypeParam>> {
        let entry = self
            .cache
            .get_or_try_insert_with(*name, || {
                Ok::<_, std::convert::Infallible>(compute_type_params(env, lowering, name))
            })
            .expect("compute is infallible");
        entry.as_ref()
    }
}

fn compute_type_params(
    env: &Environment,
    lowering: &LoweringEnv<'_>,
    name: &TypeName,
) -> Option<Vec<TypeParam>> {
    if let Some(com) = env.class_decls().get(name) {
        let raws: &[AstTypeParam] = match com {
            ClassOrModule::Class(c) => match c.primary_decl() {
                ClassDeclaration::Signature(d) => &d.type_params[..],
                ClassDeclaration::Ruby(_) => return None,
            },
            ClassOrModule::Module(m) => match m.primary_decl() {
                ModuleDeclaration::Signature(d) => &d.type_params[..],
                ModuleDeclaration::Ruby(_) => return None,
            },
        };
        let scope = build_class_param_scope(raws, name);
        return Some(ancestor_builder::ast_type_params_to_resolved(
            raws, lowering, &scope,
        ));
    }
    if let Some(entry) = env.interface_decls().get(name) {
        let raws = &entry.decl().type_params[..];
        let scope = build_class_param_scope(raws, name);
        return Some(ancestor_builder::ast_type_params_to_resolved(
            raws, lowering, &scope,
        ));
    }
    None
}

/// Build a `TypeParamScope` for the given class-level params. Each raw
/// param name maps to a scoped (`<owner>@<raw>`) interned `Name`,
/// matching the alpha-rename rule the legacy buffer uses
/// (`scope_class_param`); inner method-level params layer on top via
/// `build_method_type`.
pub(crate) fn build_class_param_scope(
    type_params: &[AstTypeParam],
    owner: &TypeName,
) -> TypeParamScope {
    let mut scope = TypeParamScope::default();
    for tp in type_params {
        scope.insert(
            tp.name,
            TypeVarKey::new(tp.name, TypeVarScope::Class(*owner)),
        );
    }
    scope
}

/// Force-private rule for instance-side methods. Mirrors rbs's
/// `special_accessibility(is_instance, method_name)` in
/// `lib/rbs/definition_builder.rb:973`. Only triggers on the
/// instance side; singleton-side `initialize` (rare in source) is
/// not force-privated.
///
/// crema differs from rbs in one corner: rbs respects an explicit
/// `private`/`public` annotation on the method (`original.visibility ||
/// special || method.accessibility`), but crema's `AstVisibility` is
/// already `Public | Private` with no unspecified state — the
/// annotation-vs-default distinction is gone before this layer sees
/// the method. We therefore force-private these names unconditionally,
/// which matches rbs in the common case (annotations are rare and
/// almost never combined with `initialize`-family names).
pub(crate) fn special_instance_visibility(name: &str) -> Option<Visibility> {
    matches!(
        name,
        "initialize"
            | "initialize_copy"
            | "initialize_clone"
            | "initialize_dup"
            | "respond_to_missing?"
    )
    .then_some(Visibility::Private)
}

/// Effective attribute visibility, folding the surrounding
/// `private` / `public` marker for instance attrs. Mirrors the
/// `DefinitionBuilder#build_instance` visibility rule:
/// - Singleton attrs ignore surrounding markers (default `Public`).
/// - Instance attrs honour explicit annotation; otherwise inherit
///   `current_visibility`.
///
/// Type lowering is deferred to `lower_member_to_method` so the
/// recursive walk runs once per attribute (at flush time), not once
/// per bucket-push site.
pub(crate) fn attribute_visibility(
    kind: AttributeKind,
    visibility: Option<crate::ast::Visibility>,
    current_visibility: Visibility,
) -> Visibility {
    match kind {
        AttributeKind::Singleton => Visibility::from_ast_or_default(visibility),
        AttributeKind::Instance => visibility
            .map(Visibility::from_ast)
            .unwrap_or(current_visibility),
    }
}

/// Build a `Method` from a single `MemberRef`, using the originating
/// `class_scope` / `context` to lower types. Mirrors the per-variant
/// lowering branches that used to live inline in
/// `process_signature_method_members` and `insert_ruby_def_method`.
///
/// `implemented_in` mirrors rbs `define_method(implemented_in:)`: pass
/// `Some(owner)` for class / module buckets (the type is the
/// implementor) and `None` for interface buckets (interfaces only
/// declare, they do not implement). `defined_in` always stays
/// `owner.clone()` regardless.
#[expect(clippy::too_many_arguments)]
fn lower_member_to_method(
    member: &MemberRef,
    bucket_name: Symbol,
    accessibility: Visibility,
    owner: &TypeName,
    method_kind: crate::type_param::MethodKind,
    start_index: u16,
    names: &NameTable,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
    context: &[Option<Name>],
    implemented_in: Option<TypeName>,
) -> Method {
    match member {
        MemberRef::Method(md) => {
            let overloads = lowering.build_overloads(
                &md.overloads,
                class_scope,
                Some((owner, md.name, method_kind)),
                start_index,
            );
            // Step A: zip each built MethodType with its source AST overload
            // so per-overload annotations land on the resulting TypeDef.
            // Matches rbs `define_method` lines 740-741.
            let defs: Vec<TypeDef> = overloads
                .into_iter()
                .zip(md.overloads.iter())
                .map(|(mt, ast_overload)| {
                    let mut td = TypeDef::new(mt, member.clone(), *owner, implemented_in);
                    td.overload_annotations = ast_overload.annotations.clone();
                    td
                })
                .collect();
            let mut method = Method::from_defs(defs, accessibility);
            method.annotations = md.annotations.clone();
            method
        }
        MemberRef::AttrReader(r) => {
            let ty = lowering.build_type(&r.ty, class_scope);
            let effective = if r.visibility.is_none() && matches!(r.kind, AttributeKind::Instance) {
                special_instance_visibility(&names.resolve(r.name)).unwrap_or(accessibility)
            } else {
                accessibility
            };
            let defs = vec![TypeDef::new(
                MethodType::getter(ty),
                member.clone(),
                *owner,
                implemented_in,
            )];
            let mut method = Method::from_defs(defs, effective);
            method.annotations = r.annotations.clone();
            method
        }
        MemberRef::AttrWriter(w) => {
            let ty = lowering.build_type(&w.ty, class_scope);
            let defs = vec![TypeDef::new(
                MethodType::setter(ty),
                member.clone(),
                *owner,
                implemented_in,
            )];
            let mut method = Method::from_defs(defs, accessibility);
            method.annotations = w.annotations.clone();
            method
        }
        MemberRef::AttrAccessor(a) => {
            let ty = lowering.build_type(&a.ty, class_scope);
            let name_str = names.resolve(bucket_name);
            let is_writer = name_str.ends_with('=');
            let (method_type, effective) = if is_writer {
                (MethodType::setter(ty), accessibility)
            } else {
                let effective =
                    if a.visibility.is_none() && matches!(a.kind, AttributeKind::Instance) {
                        special_instance_visibility(&names.resolve(a.name)).unwrap_or(accessibility)
                    } else {
                        accessibility
                    };
                (MethodType::getter(ty), effective)
            };
            let defs = vec![TypeDef::new(
                method_type,
                member.clone(),
                *owner,
                implemented_in,
            )];
            let mut method = Method::from_defs(defs, effective);
            method.annotations = a.annotations.clone();
            method
        }
        MemberRef::RubyDef(def) => {
            // rbs `ast/ruby/members.rb:509-519`: a def carrying no
            // annotation still yields one overload, `(?) -> untyped`.
            // It is owned by this class, so `defined_in` is `owner`
            // exactly like an annotated def — that ownership is what
            // lets `crema extract` attribute an unannotated method's
            // call sites (`method_call[].symbol`).
            //
            // rbs `define_method:884` discards these defs wholesale
            // when a super method exists (`empty? && existing_method`);
            // the ancestor walk does the same through
            // `Method::drop_unannotated_placeholder_defs`. Until an
            // ancestor replaces it this def *is* the method's type, and
            // `is_overloading` (which reads `method_type.is_empty()`)
            // keeps the walk going so the parent still gets its chance.
            if def.method_type.is_empty() {
                let defs = vec![TypeDef::new(
                    untyped_method_type(),
                    member.clone(),
                    *owner,
                    implemented_in,
                )];
                return Method::from_defs(defs, accessibility);
            }
            let name = names.intern_symbol(&def.name);
            let overloads = lowering
                .build_overloads_from_annotation(
                    &def.method_type,
                    context,
                    class_scope,
                    Some((owner, name, method_kind)),
                )
                .unwrap_or_else(|| vec![untyped_method_type()]);
            let defs = overloads
                .into_iter()
                .map(|mt| TypeDef::new(mt, member.clone(), *owner, implemented_in))
                .collect();
            Method::from_defs(defs, accessibility)
        }
        MemberRef::RubyAttrReader(r) => {
            let ty = ruby_attr_member_lowered_type(&r.attribute, names, lowering, class_scope);
            let effective =
                special_instance_visibility(&names.resolve(bucket_name)).unwrap_or(accessibility);
            let defs = vec![TypeDef::new(
                MethodType::getter(ty),
                member.clone(),
                *owner,
                implemented_in,
            )];
            Method::from_defs(defs, effective)
        }
        MemberRef::RubyAttrWriter(w) => {
            let ty = ruby_attr_member_lowered_type(&w.attribute, names, lowering, class_scope);
            let defs = vec![TypeDef::new(
                MethodType::setter(ty),
                member.clone(),
                *owner,
                implemented_in,
            )];
            Method::from_defs(defs, accessibility)
        }
        MemberRef::RubyAttrAccessor(a) => {
            let ty = ruby_attr_member_lowered_type(&a.attribute, names, lowering, class_scope);
            let name_str = names.resolve(bucket_name);
            let (method_type, effective) = if name_str.ends_with('=') {
                (MethodType::setter(ty), accessibility)
            } else {
                let effective = special_instance_visibility(&name_str).unwrap_or(accessibility);
                (MethodType::getter(ty), effective)
            };
            let defs = vec![TypeDef::new(
                method_type,
                member.clone(),
                *owner,
                implemented_in,
            )];
            Method::from_defs(defs, effective)
        }
        MemberRef::Synthesized => Method::from_defs(Vec::new(), accessibility),
        MemberRef::Alias(_) => unreachable!(
            "alias members are resolved in the bucket-iter path \
             (Sorter::each_strongly_connected_component) by cloning the \
             target Method, not via lower_member_to_method"
        ),
    }
}

// --- existing free functions: method / variable / alias resolution helpers ---

/// Walk an applied instance ancestor chain in variable-lookup priority order:
/// own (SelfDecl) first, then the prepend segment, then include+super.
///
/// rbs `define_instance` populates variables with own last (so own wins).
/// The linearized chain is prepend → self → include → super, so split at
/// SelfDecl to reconstruct the rbs priority without altering the chain shape.
fn variable_priority_walk(chain: &[Ancestor]) -> impl Iterator<Item = &Ancestor> {
    let self_pos = chain
        .iter()
        .position(|a| {
            matches!(
                a,
                Ancestor::Instance {
                    source: AncestorSource::SelfDecl,
                    ..
                }
            )
        })
        .expect("instance chain must contain SelfDecl");
    std::iter::once(&chain[self_pos])
        .chain(chain[..self_pos].iter())
        .chain(chain[self_pos + 1..].iter())
}

fn class_key(env: &DefinitionBuilder, name: Name) -> TypeName {
    env.names().parse_type_name(&env.names().resolve(name))
}

fn declared_type_name_key(env: ConsultationView, name: TypeName) -> TypeName {
    env.declared_type_name_by_type_name(name).unwrap_or(name)
}

/// Compute the `instance` and `class` base types for a method call
/// receiver. Used to build a `Substitution` that resolves `instance` /
/// `class` occurrences inside the method signature.
///
/// - instance receiver `Foo[A]`   → (`Foo[A]`, `singleton(Foo)`)
/// - singleton receiver `singleton(Foo)` with generic params → (`Foo[untyped, ...]`, `singleton(Foo)`)
/// - anything else → (`None`, `None`), leaving the keyword untouched.
pub fn base_types_for_receiver(env: ConsultationView, receiver: Ty) -> (Option<Ty>, Option<Ty>) {
    match env.types().resolve(receiver) {
        Type::ClassInstance { name, .. } => {
            (Some(receiver), Some(env.types().class_singleton(*name)))
        }
        Type::ClassSingleton { name } => {
            let args = env
                .class_type_params_by_type_name(name)
                .map(|params| crate::type_param::apply_defaults(params, &[], env.types()))
                .unwrap_or_default();
            let instance = env
                .types()
                .intern(Type::ClassInstance { name: *name, args });
            (Some(instance), Some(receiver))
        }
        _ => (None, None),
    }
}

/// Build a call-site `Substitution` carrying the caller-supplied
/// type-variable `bindings` plus the three base keywords (`self` /
/// `instance` / `class`) populated from `receiver`'s shape via
/// [`base_types_for_receiver`].
///
/// `self_type` is always set to `receiver`. `instance` / `class` are
/// only set when [`base_types_for_receiver`] returns them — receivers
/// without a class shape (Tuple, Record, Literal, ...) leave those
/// keywords as fallthrough.
pub fn substitution_for_receiver(
    env: ConsultationView,
    receiver: Ty,
    bindings: FxHashMap<TypeVarKey, Ty>,
) -> Substitution {
    let (instance_ty, class_ty) = base_types_for_receiver(env, receiver);
    let mut subst = Substitution::from_mapping(bindings).with_self_type(receiver);
    if let Some(it) = instance_ty {
        subst = subst.with_instance_type(it);
    }
    if let Some(ct) = class_ty {
        subst = subst.with_class_type(ct);
    }
    subst
}

/// Look up an instance method. Walks the ancestor chain.
pub fn lookup_instance_method(
    env: ConsultationView,
    class_name: &str,
    method_name: &str,
) -> Option<Method> {
    let mn = env.names().lookup_symbol(method_name)?;
    let cn = env.names().lookup(class_name)?;
    lookup_instance_method_by_name(env, cn, mn)
}

/// Look up a singleton (class) method by string name. Thin `&str` boundary
/// over the linearized [`DefinitionBuilder::lookup_singleton_method`]: it
/// interns the method symbol, parses + normalizes the class name, then
/// delegates the chain walk and drops the bindings the method version
/// returns. Callers that need bindings call the method version directly.
///
/// `extend M` contributes M's instance methods (transitively via M's own
/// `include` chain) as singleton methods of the current class; `include`
/// does not affect singleton lookup. Those semantics live in
/// `singleton_ancestors`, which the method version consumes.
pub fn lookup_singleton_method(
    env: &DefinitionBuilder,
    class_name: &str,
    method_name: &str,
) -> Option<Method> {
    // `intern_symbol` (not `lookup_symbol`): typed `.new` is now baked
    // into `singleton_def.methods` by `bake_typed_new`, which interns
    // "new" lazily on its first run. If we early-returned on
    // `lookup_symbol("new") -> None` here, callers reaching this entry
    // point *before* any `build_singleton` had been triggered would
    // miss the to-be-baked entry. Interning is a no-op when the symbol
    // already exists; for genuinely-unknown method names the downstream
    // `methods.get(&mn)` still returns None.
    let mn = env.names().intern_symbol(method_name);
    let raw = env.names().parse_type_name(class_name);
    lookup_singleton_method_by_type_name(ConsultationView::new(env, None), raw, mn)
}

/// Like [`resolve_singleton_method`] but returns the type-variable
/// bindings that were in scope at the matched method's level, ready to
/// feed into `substitute_type_vars`. Thin adapter over
/// [`resolve_singleton_method_with_type_name_args`] for callers that
/// still hold a `Name`-keyed class reference.
pub fn resolve_singleton_method_with_args(
    env: &DefinitionBuilder,
    class_name: Name,
    method_name: Symbol,
) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
    let raw = env
        .names()
        .parse_type_name(&env.names().resolve(class_name));
    resolve_singleton_method_with_type_name_args(ConsultationView::new(env, None), raw, method_name)
}

pub fn resolve_singleton_method_with_type_name_args(
    env: ConsultationView,
    class_name: TypeName,
    method_name: Symbol,
) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
    let class_name = normalized_declared_type_name(env, class_name);
    if let Some(found) = env.lookup_singleton_method(&class_name, method_name) {
        return Some(found);
    }

    let metaclass = match env.declared_kind_by_type_name(&class_name)? {
        DeclKindLocal::Class => "::Class",
        DeclKindLocal::Module => "::Module",
        DeclKindLocal::Interface => return None,
    };

    lookup_instance_method(env, metaclass, &env.names().resolve(method_name))
        .map(|def| (def, FxHashMap::default()))
}

/// Look up a method for a singleton type `singleton(class_name)`.
///
/// Resolution order (mirrors RBS `ancestor_builder.rb`):
/// 1. Walk singleton_methods along the superclass chain
/// 2. Fall through to the metaclass's instance_methods:
///    - class → `::Class` → `::Module` → `::Object` → ...
///    - module → `::Module` → `::Object` → ...
pub fn lookup_method_for_singleton(
    env: &DefinitionBuilder,
    class_name: &str,
    method_name: &str,
) -> Option<Method> {
    if let Some(def) = lookup_singleton_method(env, class_name, method_name) {
        return Some(def);
    }

    let cn = env.names().lookup(class_name)?;
    let metaclass = match env.declared_kind_by_type_name(&class_key(env, cn))? {
        DeclKindLocal::Class => "::Class",
        DeclKindLocal::Module => "::Module",
        DeclKindLocal::Interface => return None,
    };

    lookup_instance_method(ConsultationView::new(env, None), metaclass, method_name)
}

/// Resolve a method on a singleton type. Runs the chain walk and the
/// metaclass fallback; typed `.new` synthesis is no longer special-cased
/// here — `DefinitionBuilder::bake_typed_new` writes the synth into the
/// singleton table during `build_class_pair`, so the chain walk above
/// picks it up like any other method.
pub fn resolve_singleton_method(
    env: &DefinitionBuilder,
    class_name: &str,
    method_name: &str,
) -> Option<Method> {
    if let Some(def) = lookup_singleton_method(env, class_name, method_name) {
        return Some(def);
    }

    let cn = env.names().lookup(class_name)?;
    let metaclass = match env.declared_kind_by_type_name(&class_key(env, cn))? {
        DeclKindLocal::Class => "::Class",
        DeclKindLocal::Module => "::Module",
        DeclKindLocal::Interface => return None,
    };

    lookup_instance_method(ConsultationView::new(env, None), metaclass, method_name)
}

/// Look up an instance method by [`Name`] (avoids string round-trip).
pub fn lookup_instance_method_by_name(
    env: ConsultationView,
    class_name: Name,
    method_name: Symbol,
) -> Option<Method> {
    let raw = env
        .names()
        .parse_type_name(&env.names().resolve(class_name));
    lookup_instance_method_by_type_name(env, raw, method_name)
}

pub fn lookup_singleton_method_by_name(
    env: &DefinitionBuilder,
    class_name: Name,
    method_name: Symbol,
) -> Option<Method> {
    let raw = env
        .names()
        .parse_type_name(&env.names().resolve(class_name));
    lookup_singleton_method_by_type_name(ConsultationView::new(env, None), raw, method_name)
}

pub fn lookup_instance_method_by_type_name(
    env: ConsultationView,
    class_name: TypeName,
    method_name: Symbol,
) -> Option<Method> {
    let class_name = normalized_declared_type_name(env, class_name);
    env.lookup_instance_method(&class_name, method_name)
}

/// True when every overload of `method` was declared in `::BasicObject`.
/// Used by both the type-checker (`lookup_method_target`) and the `.new`
/// synthesis path (`bake_typed_new`) to detect the ancestor walk that
/// resolves an unresolved `initialize` to `BasicObject#initialize: () -> void`,
/// so the soutaro-approved non-inheritance narrowing can suppress the
/// downstream `MethodParameterMismatch` / synthetic `.new` false positives.
///
/// Empty `defs` is treated as "no BasicObject match" — the caller keeps
/// its default "no method target" behavior instead of accidentally
/// suppressing an unrelated resolution.
pub fn defs_all_from_basic_object(env: ConsultationView, method: &Method) -> bool {
    if method.defs.is_empty() {
        return false;
    }
    let basic_object = env.names().builtins().basic_object;
    method.defs.iter().all(|td| td.defined_in == basic_object)
}

/// True when the receiver class `class_name` OR any of its instance
/// ancestors (super classes, included / prepended modules) carries an
/// inline `def initialize` in a Ruby-side declaration.
///
/// Signal used by `bake_typed_new` to distinguish the two cases the
/// synthetic-`.new` path otherwise cannot tell apart once
/// `defs_all_from_basic_object` fires:
///
/// - `class Foo; end` — no inline `def initialize` anywhere on the chain,
///   keep the `synthesize_new_from_initialize` path so `Foo.new(1)` still
///   errors against `BasicObject#initialize: () -> void` (the true
///   positive on an empty class must survive).
/// - `class Foo; def initialize(a, b:); end; end`
///   / `class Bar < Foo; end` / `class Baz; include M; end` (where M
///   declares an inline `def initialize`) — the synthetic `.new` must
///   fall through to `synthesize_untyped_new`, matching Ruby's runtime
///   semantics where `Bar.new(...)` / `Baz.new(...)` dispatches to the
///   inherited or included `initialize`.
///
/// Walking the full instance-ancestor chain (not just the receiver's own
/// declarations) mirrors the override-side narrowing in
/// `lookup_method_target`: both paths ask "did the ancestor walk skip
/// over an inline unannotated `def initialize` before landing on
/// `BasicObject#initialize: () -> void`?", just at different call sites.
/// Restricting the check to the receiver alone (`class_decls.get(name)`)
/// leaves subclass / include-mixin cases producing the same
/// `UnexpectedPositionalArgument` false positives the direct-class fix
/// was meant to eliminate.
pub fn class_has_inline_def_initialize(env: &DefinitionBuilder, class_name: &TypeName) -> bool {
    let ancestors = env
        .ancestor_builder()
        .instance_ancestors(class_name)
        .apply(&[], env.ancestor_builder().types());
    ancestors.iter().any(|ancestor| match ancestor {
        crate::definition::ancestor_builder::Ancestor::Instance { name, .. } => {
            declaration_has_inline_def_initialize(env, name)
        }
        crate::definition::ancestor_builder::Ancestor::Singleton { .. } => false,
    })
}

fn declaration_has_inline_def_initialize(env: &DefinitionBuilder, name: &TypeName) -> bool {
    let Some(entry) = env.env().class_decls().get(name) else {
        return false;
    };
    match entry {
        ClassOrModule::Class(class_entry) => {
            class_entry
                .context_decls()
                .iter()
                .any(|(_, _, decl)| match decl {
                    ClassDeclaration::Ruby(ruby_decl) => {
                        members_have_inline_def_initialize(&ruby_decl.members)
                    }
                    ClassDeclaration::Signature(_) => false,
                })
        }
        ClassOrModule::Module(module_entry) => {
            module_entry
                .context_decls()
                .iter()
                .any(|(_, _, decl)| match decl {
                    ModuleDeclaration::Ruby(ruby_decl) => {
                        members_have_inline_def_initialize(&ruby_decl.members)
                    }
                    ModuleDeclaration::Signature(_) => false,
                })
        }
    }
}

fn members_have_inline_def_initialize(members: &[RubyMember]) -> bool {
    members.iter().any(|member| match member {
        RubyMember::Def(def) => {
            def.name == "initialize" && def.kind == crate::ast::MethodKind::Instance
        }
        _ => false,
    })
}

pub fn lookup_singleton_method_by_type_name(
    env: ConsultationView,
    class_name: TypeName,
    method_name: Symbol,
) -> Option<Method> {
    let class_name = normalized_declared_type_name(env, class_name);
    env.lookup_singleton_method(&class_name, method_name)
        .map(|(m, _)| m)
}

// `env.builder.env()` (not `env.env()`): the view intentionally does not
// forward raw `Environment` access (that is exactly the bypass ADR-0032
// Decision 2 closes off for the checker system). This free function lives
// in the same module as `ConsultationView`, so it can still reach the
// builder's raw environment directly for this one normalization step —
// the result only feeds `declared_type_name_key`, which itself only feeds
// a later recorded query, so recording this step too would double-count
// the same fact under two keys.
fn normalized_declared_type_name(env: ConsultationView, class_name: TypeName) -> TypeName {
    let normalized = env.builder.env().normalize_module_name(&class_name);
    declared_type_name_key(env, normalized)
}

/// Expand a type alias reference to its body, following alias-to-alias
/// chains to a fixpoint (`type path = ::path` where `::path` is itself an
/// alias). For generic aliases, substitutes type parameters with the
/// provided args. If the type is not an alias, returns it unchanged.
///
/// The depth bound guards against recursive aliases (`type a = a`), which
/// rbs rejects via type-alias regularity validation but crema does not
/// (ADR-0013): a cyclic body would otherwise loop forever here.
///
/// Memoized on the input `Ty` via [`DefinitionBuilder::expand_alias_cache`]
/// because the function is hot in `rbs/lib` checks where the same alias
/// (`type fields = ...`) is dispatched on repeatedly. Cyclic aliases that
/// hit `ALIAS_EXPANSION_LIMIT` are also cached, mirroring the function's
/// bottom-out return.
pub fn expand_alias(env: ConsultationView, ty: Ty) -> Ty {
    // `env.builder.expand_alias_cache` (not a view-forwarded accessor):
    // this is the function's own memoization cache, not a checker-facing
    // query — same same-module private-field access as
    // `normalized_declared_type_name`'s `env.builder.env()` above.
    if let Some(hit) = env.builder.expand_alias_cache.borrow().get(&ty).copied() {
        return hit;
    }
    let original = ty;
    let mut current = ty;
    for _ in 0..ALIAS_EXPANSION_LIMIT {
        let Type::Alias { name, args } = env.types().resolve(current) else {
            env.builder
                .expand_alias_cache
                .borrow_mut()
                .insert(original, current);
            return current;
        };
        current = env.expand_type_alias(name, args).unwrap_or(Ty::UNTYPED);
    }
    env.builder
        .expand_alias_cache
        .borrow_mut()
        .insert(original, current);
    current
}

/// Maximum alias-to-alias hops `expand_alias` follows before giving up,
/// protecting against cyclic aliases in unvalidated RBS input.
const ALIAS_EXPANSION_LIMIT: usize = 100;

/// Whether `ty` (after alias expansion) resolves to `Type::Record`.
pub fn resolves_to_record(env: ConsultationView, ty: Ty) -> bool {
    matches!(
        env.types().resolve(expand_alias(env, ty)),
        Type::Record { .. }
    )
}

/// Expand `T?` / `Bool` into the explicit union form
/// (`T | nil` / `TrueClass | FalseClass`), recursing into nested
/// `Type::Union` so members are kept de-duplicated by intern id.
///
/// Free-function form of the former `TypeChecker::widen_optional_bool_to_union`
/// — moved here so `normalize_receiver` can call it without going through a
/// `TypeChecker` `&self` borrow. The function reads `env.types()` /
/// `env.names()` / `env.class_instance_type()` only; no `TypeChecker`-side
/// state is consulted.
pub fn widen_optional_bool(env: ConsultationView, ty: Ty) -> Ty {
    let mut members: Vec<Ty> = Vec::new();
    expand_optional_bool_sugar(env, ty, &mut members);
    if members.len() == 1 {
        members[0]
    } else {
        env.types().intern(Type::Union(members))
    }
}

/// Recursive helper for [`widen_optional_bool`]. Flattens `Type::Optional`
/// / `Type::Bool` (and `Type::Union` members) into `out`, deduping per
/// intern id so `Optional(Optional(T))` does not emit `nil` twice.
fn expand_optional_bool_sugar(env: ConsultationView, ty: Ty, out: &mut Vec<Ty>) {
    fn push_unique(out: &mut Vec<Ty>, t: Ty) {
        if !out.contains(&t) {
            out.push(t);
        }
    }
    match env.types().resolve(ty) {
        Type::Optional(inner) => {
            expand_optional_bool_sugar(env, *inner, out);
            push_unique(out, Ty::NIL);
        }
        Type::Bool => {
            let names = env.names();
            let true_ty = env.class_instance_type(names.builtins().true_class);
            let false_ty = env.class_instance_type(names.builtins().false_class);
            push_unique(out, true_ty);
            push_unique(out, false_ty);
        }
        Type::Union(members) => {
            for &m in members {
                expand_optional_bool_sugar(env, m, out);
            }
        }
        _ => push_unique(out, ty),
    }
}

/// Expand each member's alias and flatten any nested unions for
/// per-component dispatch. Receivers like `Integer | RBS::Types::t`
/// (where `t` itself is a union of variants) reach the union arm with
/// `t` still wrapped as `Type::Alias`; per-component method lookup needs
/// the alias members spread back into the outer iteration.
///
/// Termination: `expand_alias` is bounded by `ALIAS_EXPANSION_LIMIT`, but
/// a union containing a self-referential alias (`type a = ::String | ::a`)
/// would re-enter the same alias on every recursion. The `visited` set
/// keys on pre-expansion `Ty` so each alias is followed at most once per
/// flatten call.
///
/// Free-function form of the former `type_checker::calls::flatten_alias_union_members`
/// — moved here for use by [`normalize_receiver`].
pub fn flatten_alias_union_members(env: ConsultationView, members: &[Ty]) -> Vec<Ty> {
    let mut out = Vec::new();
    let mut visited = FxHashSet::default();
    flatten_alias_union_into(env, members, &mut out, &mut visited);
    out
}

fn flatten_alias_union_into(
    env: ConsultationView,
    members: &[Ty],
    out: &mut Vec<Ty>,
    visited: &mut FxHashSet<Ty>,
) {
    for &m in members {
        if !visited.insert(m) {
            continue;
        }
        let expanded = expand_alias(env, m);
        match env.types().resolve(expanded) {
            Type::Union(inner) => flatten_alias_union_into(env, inner, out, visited),
            _ => out.push(expanded),
        }
    }
}

/// Normalize a receiver `Ty` for dispatch: alias expand (fixpoint) →
/// Optional/Bool sugar widen → alias-of-union flatten. Run at dispatch
/// boundary entry (`resolve_call_target_at` / `check_no_method_at`) so
/// the three stages don't repeat for every receiver kind arm below.
///
/// Memoized on the input `Ty` via
/// [`DefinitionBuilder::normalize_receiver_cache`]. Independent of
/// [`expand_alias_cache`] because the result diverges (widen + flatten
/// stages) and is only safe to reuse at dispatch sites — utility callers
/// (`element_access_specialization`, `inference.rs` hint expansion) must
/// keep calling [`expand_alias`] directly to preserve their unwidened
/// semantics.
///
/// Mirrors Steep's per-call receiver normalization (`raw_shape` Alias
/// arm and sugar handling in `factory.rb` / `builder.rb`) under
/// ADR-0021's "no Shape layer" constraint — the cache wraps the
/// per-call function rather than the full shape.
pub fn normalize_receiver(env: ConsultationView, ty: Ty) -> Ty {
    // `env.builder.normalize_receiver_cache`: this function's own
    // memoization cache, not a checker-facing query (see `expand_alias`'s
    // comment on the same pattern).
    if let Some(hit) = env
        .builder
        .normalize_receiver_cache
        .borrow()
        .get(&ty)
        .copied()
    {
        return hit;
    }
    let expanded = expand_alias(env, ty);
    let widened = widen_optional_bool(env, expanded);
    let normalized = match env.types().resolve(widened) {
        Type::Union(members) => {
            let flattened = flatten_alias_union_members(env, members);
            if flattened.len() == 1 {
                flattened[0]
            } else {
                env.types().intern(Type::Union(flattened))
            }
        }
        _ => widened,
    };
    env.builder
        .normalize_receiver_cache
        .borrow_mut()
        .insert(ty, normalized);
    normalized
}

// ============================================================================
// Consultation view (ADR-0032 Decision 2)
// ============================================================================
//
// `ConsultationView` fronts `DefinitionBuilder`'s public query API for the
// checker system (`TypeChecker` / `SubtypeChecker` / `narrowing` /
// `pure_call_env`), which hold a `ConsultationView` by value where they used
// to hold a `&DefinitionBuilder` directly. Every query method the view forwards has
// the same name as its `DefinitionBuilder` counterpart, so existing call
// sites (`self.env.lookup_instance_method(...)`, free functions taking
// `env: &DefinitionBuilder`, etc.) keep compiling unchanged against the view
// type — any query surface the view does *not* forward simply fails to
// compile at its call site, which is how completeness is enforced here
// rather than by a runtime assertion.
//
// Only queries actually reachable from the checker system are forwarded.
// `lookup_constant_type`, `build_instance` / `build_singleton` /
// `build_interface`, `method_dups` / `variable_dups`, and the `Name`-keyed
// siblings of several `TypeName`-keyed queries have zero callers from
// checker code today (verified by grep across `src/type_checker/`,
// `src/subtyping.rs`, `src/narrowing.rs`, `src/pure_call_env.rs`, and the
// free-function layer below) and are left on `DefinitionBuilder` only.

/// Key for the dedup'd per-file consultation set (ADR-0032 Decision 2).
/// Built from process-stable names only (`Symbol` / `TypeName` / ...) —
/// never a `Ty` handle, interner index, or raw `Name` (see
/// [`Name`]'s doc comment: unlike `Symbol`/`TypeName`, `Name` ids are
/// first-seen-order, not content-addressed, so they cannot survive a
/// process restart as a stable cache key).
///
/// The key identifies *which fact was consulted*, not the fact's value —
/// hit vs. miss is tracked separately by [`ConsultationLog`]. Two queries
/// with the same argument shape but different meaning (e.g.
/// `IsDeclaredClass` vs. `IsDeclaredInterface`, both keyed on a lone
/// `TypeName`) get distinct variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConsultedKey {
    ExpandTypeAlias(TypeName),
    ConstantDeprecated(TypeName),
    GlobalLookup(Symbol),
    GlobalDeprecated(Symbol),
    ClassOrModuleDeprecated(TypeName),
    ClassAliasDeprecated(TypeName),
    DeclaredKind(TypeName),
    IsDeclaredClass(TypeName),
    IsDeclaredClassAlias(TypeName),
    IsDeclaredInterface(TypeName),
    IsDeclaredTypeAlias(TypeName),
    IsDeclaredConstant(TypeName),
    ClassOrModuleKind(TypeName),
    ClassTypeParams(TypeName),
    InterfaceMethodNames(TypeName),
    SuperclassTypeName(TypeName),
    HasCompleteAncestorChain(TypeName),
    InstanceAncestors(TypeName),
    SingletonAncestors(TypeName),
    OneInstanceAncestors(TypeName),
    MethodResolution {
        receiver: TypeName,
        method: Symbol,
    },
    InterfaceMethodResolution {
        interface: TypeName,
        method: Symbol,
    },
    SingletonMethodResolution {
        class: TypeName,
        method: Symbol,
    },
    SuperInstanceMethod {
        class: TypeName,
        method: Symbol,
    },
    SuperSingletonMethod {
        class: TypeName,
        method: Symbol,
    },
    InstanceVariable {
        class: TypeName,
        var: Symbol,
    },
    ClassVariable {
        class: TypeName,
        var: Symbol,
    },
    ClassInstanceVariable {
        class: TypeName,
        var: Symbol,
    },
    SyntheticConcernTargets {
        method: Symbol,
        kind: MethodKind,
        location: PrismByteRange,
        /// Resolved from the raw `Name` at the call site — see the
        /// [`ConsultedKey`] doc comment for why the id itself cannot be
        /// used as a key.
        source_file: Option<String>,
        current: Option<TypeName>,
    },
    ConstantResolution {
        name: Symbol,
        context: ConstantContext,
    },
    ConstantResolutionInNamespace {
        scope: Option<TypeName>,
        name: Symbol,
    },
    ConstantResolutionChild {
        module: TypeName,
        name: Symbol,
    },
}

/// Recording sink for one [`ConsultationView`]. Maps each consulted key to
/// whether the query hit (`true`) or missed (`false`) — this both dedups
/// (a repeated query overwrites the same entry) and keeps the hit/miss
/// outcome inspectable, satisfying ADR-0032 Decision 2's requirement to
/// record misses too (negative dependency) without a separate representation.
#[derive(Debug, Default)]
pub struct ConsultationLog {
    entries: RefCell<FxHashMap<ConsultedKey, bool>>,
}

impl ConsultationLog {
    pub fn new() -> Self {
        Self::default()
    }

    fn record(&self, key: ConsultedKey, hit: bool) {
        self.entries.borrow_mut().insert(key, hit);
    }

    /// Drains the recorded set for persistence (cache_io) or inspection.
    /// Consumes `self` — a log is per-file and one-shot, never replayed.
    pub fn into_entries(self) -> FxHashMap<ConsultedKey, bool> {
        self.entries.into_inner()
    }
}

/// Per-file recording front for [`DefinitionBuilder`]'s query API. Checker
/// holders (`TypeChecker` / `SubtypeChecker` / narrowing / `pure_call_env`)
/// take a `ConsultationView` by value where they used to take a
/// `&DefinitionBuilder` — just two references (`Copy`), so it is passed
/// around the same way `Ty` / `Symbol` / `TypeName` already are in this
/// codebase, rather than through an extra layer of borrowing.
///
/// `log` is `None` on the default (incremental off) path — every recording
/// method degrades to a plain forward-and-discard call, so the only
/// per-query cost paid there is one `Option` check.
#[derive(Clone, Copy)]
pub struct ConsultationView<'a> {
    builder: &'a DefinitionBuilder,
    log: Option<&'a ConsultationLog>,
}

impl<'a> ConsultationView<'a> {
    pub fn new(builder: &'a DefinitionBuilder, log: Option<&'a ConsultationLog>) -> Self {
        Self { builder, log }
    }

    /// `key` is a closure, not a value: `log: None` (the default,
    /// incremental-off path) must not pay for constructing a `ConsultedKey`
    /// it will immediately discard. Eagerly-evaluated arguments would defeat
    /// that — `ConstantContext` clones a heap-allocated `Box<[TypeName]>`,
    /// and `lookup_global`'s key used to force a permanent interner entry
    /// for names that were never actually declared. Laziness makes both
    /// costs disappear entirely when nothing is listening.
    fn record(&self, key: impl FnOnce() -> ConsultedKey, hit: bool) {
        if let Some(log) = self.log {
            log.record(key(), hit);
        }
    }

    // -- recorded queries --------------------------------------------------

    pub fn expand_type_alias(&self, name: &TypeName, args: &[Ty]) -> Option<Ty> {
        let result = self.builder.expand_type_alias(name, args);
        self.record(|| ConsultedKey::ExpandTypeAlias(*name), result.is_some());
        result
    }

    pub fn constant_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let result = self.builder.constant_deprecated_message(name);
        self.record(|| ConsultedKey::ConstantDeprecated(*name), result.is_some());
        result
    }

    pub fn lookup_global(&self, name_str: &str) -> Option<Ty> {
        let result = self.builder.lookup_global(name_str);
        // `intern_symbol` (not the non-interning `lookup_symbol`) only
        // inside the closure — outside `log: Some(_)`, an undeclared
        // global reference must not leave a permanent interner entry.
        self.record(
            || ConsultedKey::GlobalLookup(self.builder.names().intern_symbol(name_str)),
            result.is_some(),
        );
        result
    }

    pub fn global_deprecated_message(&self, sym: Symbol) -> Option<Option<String>> {
        let result = self.builder.global_deprecated_message(sym);
        self.record(|| ConsultedKey::GlobalDeprecated(sym), result.is_some());
        result
    }

    pub fn class_or_module_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let result = self.builder.class_or_module_deprecated_message(name);
        self.record(
            || ConsultedKey::ClassOrModuleDeprecated(*name),
            result.is_some(),
        );
        result
    }

    pub fn class_alias_deprecated_message(&self, name: &TypeName) -> Option<Option<String>> {
        let result = self.builder.class_alias_deprecated_message(name);
        self.record(
            || ConsultedKey::ClassAliasDeprecated(*name),
            result.is_some(),
        );
        result
    }

    pub fn declared_kind_by_type_name(&self, tn: &TypeName) -> Option<DeclKindLocal> {
        let result = self.builder.declared_kind_by_type_name(tn);
        self.record(|| ConsultedKey::DeclaredKind(*tn), result.is_some());
        result
    }

    pub fn is_declared_class(&self, name: &TypeName) -> bool {
        let result = self.builder.is_declared_class(name);
        self.record(|| ConsultedKey::IsDeclaredClass(*name), result);
        result
    }

    pub fn is_declared_class_alias(&self, name: &TypeName) -> bool {
        let result = self.builder.is_declared_class_alias(name);
        self.record(|| ConsultedKey::IsDeclaredClassAlias(*name), result);
        result
    }

    pub fn is_declared_interface(&self, name: &TypeName) -> bool {
        let result = self.builder.is_declared_interface(name);
        self.record(|| ConsultedKey::IsDeclaredInterface(*name), result);
        result
    }

    pub fn is_declared_type_alias(&self, name: &TypeName) -> bool {
        let result = self.builder.is_declared_type_alias(name);
        self.record(|| ConsultedKey::IsDeclaredTypeAlias(*name), result);
        result
    }

    pub fn is_declared_constant(&self, name: &TypeName) -> bool {
        let result = self.builder.is_declared_constant(name);
        self.record(|| ConsultedKey::IsDeclaredConstant(*name), result);
        result
    }

    pub fn class_or_module_kind(&self, name: &TypeName) -> Option<DeclKindLocal> {
        let result = self.builder.class_or_module_kind(name);
        self.record(|| ConsultedKey::ClassOrModuleKind(*name), result.is_some());
        result
    }

    pub fn class_type_params_by_type_name(&self, name: &TypeName) -> Option<&'a Vec<TypeParam>> {
        let result = self.builder.class_type_params_by_type_name(name);
        self.record(|| ConsultedKey::ClassTypeParams(*name), result.is_some());
        result
    }

    pub fn interface_method_names_by_type_name(&self, name: &TypeName) -> Option<Vec<Symbol>> {
        let result = self.builder.interface_method_names_by_type_name(name);
        self.record(
            || ConsultedKey::InterfaceMethodNames(*name),
            result.is_some(),
        );
        result
    }

    pub fn superclass_type_name(&self, class_name: &TypeName) -> Option<TypeName> {
        let result = self.builder.superclass_type_name(class_name);
        self.record(
            || ConsultedKey::SuperclassTypeName(*class_name),
            result.is_some(),
        );
        result
    }

    pub fn has_complete_ancestor_chain(&self, class_name: &TypeName) -> bool {
        let result = self.builder.has_complete_ancestor_chain(class_name);
        self.record(
            || ConsultedKey::HasCompleteAncestorChain(*class_name),
            result,
        );
        result
    }

    pub fn lookup_instance_method(&self, class: &TypeName, method: Symbol) -> Option<Method> {
        let result = self.builder.lookup_instance_method(class, method);
        self.record(
            || ConsultedKey::MethodResolution {
                receiver: *class,
                method,
            },
            result.is_some(),
        );
        result
    }

    pub fn lookup_instance_method_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let result = self
            .builder
            .lookup_instance_method_with_args(class, args, method);
        self.record(
            || ConsultedKey::MethodResolution {
                receiver: *class,
                method,
            },
            result.is_some(),
        );
        result
    }

    pub fn lookup_interface_method_with_args(
        &self,
        interface: &TypeName,
        args: &[Ty],
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let result = self
            .builder
            .lookup_interface_method_with_args(interface, args, method);
        self.record(
            || ConsultedKey::InterfaceMethodResolution {
                interface: *interface,
                method,
            },
            result.is_some(),
        );
        result
    }

    pub fn lookup_super_method(
        &self,
        self_type: Ty,
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let result = self.builder.lookup_super_method(self_type, method);
        match self.builder.types().resolve(self_type) {
            Type::ClassInstance { name, .. } => self.record(
                || ConsultedKey::SuperInstanceMethod {
                    class: *name,
                    method,
                },
                result.is_some(),
            ),
            Type::ClassSingleton { name } => self.record(
                || ConsultedKey::SuperSingletonMethod {
                    class: *name,
                    method,
                },
                result.is_some(),
            ),
            _ => {}
        }
        result
    }

    pub fn lookup_singleton_method(
        &self,
        class: &TypeName,
        method: Symbol,
    ) -> Option<(Method, FxHashMap<TypeVarKey, Ty>)> {
        let result = self.builder.lookup_singleton_method(class, method);
        self.record(
            || ConsultedKey::SingletonMethodResolution {
                class: *class,
                method,
            },
            result.is_some(),
        );
        result
    }

    pub fn lookup_instance_variable_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let result = self
            .builder
            .lookup_instance_variable_with_args(class, args, var);
        self.record(
            || ConsultedKey::InstanceVariable { class: *class, var },
            result.is_some(),
        );
        result
    }

    pub fn lookup_class_variable_with_args(
        &self,
        class: &TypeName,
        args: &[Ty],
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let result = self
            .builder
            .lookup_class_variable_with_args(class, args, var);
        self.record(
            || ConsultedKey::ClassVariable { class: *class, var },
            result.is_some(),
        );
        result
    }

    pub fn lookup_class_instance_variable_with_args(
        &self,
        class: &TypeName,
        var: Symbol,
    ) -> Option<(Variable, FxHashMap<TypeVarKey, Ty>)> {
        let result = self
            .builder
            .lookup_class_instance_variable_with_args(class, var);
        self.record(
            || ConsultedKey::ClassInstanceVariable { class: *class, var },
            result.is_some(),
        );
        result
    }

    pub(crate) fn synthetic_concern_targets(
        &self,
        method: Symbol,
        kind: MethodKind,
        location: PrismByteRange,
        source_file: Option<Name>,
        current: Option<TypeName>,
    ) -> Vec<TypeName> {
        let result =
            self.builder
                .synthetic_concern_targets(method, kind, location, source_file, current);
        self.record(
            || ConsultedKey::SyntheticConcernTargets {
                method,
                kind,
                location,
                source_file: source_file.map(|n| self.builder.names().resolve(n)),
                current,
            },
            !result.is_empty(),
        );
        result
    }

    /// Replaces raw `constant_resolver().resolve(...)` access — checker
    /// code held the raw `&ConstantResolver` across several calls, which
    /// would bypass the view entirely.
    pub fn resolve_constant(
        &self,
        name: Symbol,
        context: &ConstantContext,
    ) -> Option<ResolverConstant> {
        let result = self.builder.constant_resolver().resolve(name, context);
        self.record(
            || ConsultedKey::ConstantResolution {
                name,
                context: context.clone(),
            },
            result.is_some(),
        );
        result
    }

    pub fn resolve_constant_in_namespace(
        &self,
        scope: Option<&TypeName>,
        name: Symbol,
    ) -> Option<ResolverConstant> {
        let result = self
            .builder
            .constant_resolver()
            .resolve_in_namespace(scope, name);
        self.record(
            || ConsultedKey::ConstantResolutionInNamespace {
                scope: scope.copied(),
                name,
            },
            result.is_some(),
        );
        result
    }

    pub fn resolve_constant_child(
        &self,
        module_name: &TypeName,
        name: Symbol,
    ) -> Option<ResolverConstant> {
        let result = self
            .builder
            .constant_resolver()
            .resolve_child(module_name, name);
        self.record(
            || ConsultedKey::ConstantResolutionChild {
                module: *module_name,
                name,
            },
            result.is_some(),
        );
        result
    }

    // -- non-recorded passthroughs ------------------------------------------
    //
    // Interner access, the builder's own internal caches (the 6 tables
    // ADR-0032 Decision 2 says the view must not alter), and normalization
    // steps that only feed a later recorded query (so recording them too
    // would double-count the same fact under two keys).

    pub fn names(&self) -> &'a NameTable {
        self.builder.names()
    }

    pub fn types(&self) -> &'a TypeTable {
        self.builder.types()
    }

    /// Recorded (unlike the interner passthroughs above): the linearized
    /// chain is a direct function of `name`'s decl graph, so a caller
    /// that only walks ancestors — e.g. `SubtypeChecker`'s
    /// `Sub <: Base` check, which never records a method key on the sub
    /// type — must still be invalidated when `name` is reparented.
    pub fn instance_ancestors(&self, name: &TypeName) -> Arc<ancestor_builder::InstanceAncestors> {
        let result = self.builder.ancestor_builder().instance_ancestors(name);
        self.record(|| ConsultedKey::InstanceAncestors(*name), true);
        result
    }

    /// See [`Self::instance_ancestors`].
    pub fn singleton_ancestors(
        &self,
        name: &TypeName,
    ) -> Arc<ancestor_builder::SingletonAncestors> {
        let result = self.builder.ancestor_builder().singleton_ancestors(name);
        self.record(|| ConsultedKey::SingletonAncestors(*name), true);
        result
    }

    /// See [`Self::instance_ancestors`].
    pub fn one_instance_ancestors_arc(
        &self,
        name: &TypeName,
    ) -> Arc<ancestor_builder::OneAncestors> {
        let result = self
            .builder
            .ancestor_builder()
            .one_instance_ancestors_arc(name);
        self.record(|| ConsultedKey::OneInstanceAncestors(*name), true);
        result
    }

    pub fn class_instance_type(&self, name: TypeName) -> Ty {
        self.builder.class_instance_type(name)
    }

    pub fn declared_type_name_by_type_name(&self, name: TypeName) -> Option<TypeName> {
        self.builder.declared_type_name_by_type_name(name)
    }

    /// Not decomposed into `ConsultedKey`s (out of scope for this todo —
    /// ADR-0032's own enumerated query surface does not cover it either).
    /// AST-type lowering internally consults declared type-alias / type-param
    /// shape, so this is a real gap for axis 3 to close, not an oversight
    /// left silently: flagged in the axis 2 done todo for axis 3 to pick up.
    pub fn lower_ast_type(
        &self,
        ast_ty: &crate::ast::types::Type,
        context: &[Option<Name>],
        scope: &TypeParamScope,
    ) -> Ty {
        self.builder.lower_ast_type(ast_ty, context, scope)
    }

    pub(crate) fn cached_subtype_result(&self, key: &SubtypeCacheKey) -> Option<bool> {
        self.builder.cached_subtype_result(key)
    }

    pub(crate) fn store_subtype_result(&self, key: SubtypeCacheKey, result: bool) {
        self.builder.store_subtype_result(key, result);
    }
}

pub mod type_builder;


