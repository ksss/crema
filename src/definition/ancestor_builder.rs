//! Ancestor resolution for the new Definition layer.
//!
//! Mirrors rbs `RBS::DefinitionBuilder::AncestorBuilder` with two
//! entrypoint families:
//!
//! - `one_*_ancestors` returns 1-step [`OneAncestors`] (used during the
//!   build phase) — direct super_class, mixin refs, self_types.
//! - `instance_ancestors` / `singleton_ancestors` / `interface_ancestors`
//!   return a fully linearized [`InstanceAncestors`] / [`SingletonAncestors`]
//!   with the recursive walk *and* type-arg substitution baked in. Callers
//!   then `apply(args)` to bind the root's type params at lookup time.
//!
//! Validation that rbs performs at this layer (NoMixinFoundError /
//! MixinClassError / InvalidTypeApplicationError) is intentionally not
//! ported. ADR-0010 keeps such diagnostics on the legacy build path
//! (ADR-0013); unknown mixin targets here are silently skipped.
//!
//! Walk strategy: the linearized ancestors are built eagerly and held in a
//! `Vec`. ADR-0009's lazy-walk preservation is satisfied at the data-shape
//! level (`Definition.methods` holds own-class entries only); whether the
//! walk itself is lazy is independent. Caller-side `Vec::iter().find()`
//! still short-circuits at the iterator boundary if a method-resolution
//! caller wants early termination, but no walk-time short-circuit is
//! provided.

use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::Arc;

use super::MixinRef;
use super::ancestor_graph::Node;
use super::lowering_maps::LoweringMaps;
use super::type_lowering::{LoweringEnv, build_lowering_maps, class_alias_old_name};
use crate::ast::TypeParam as AstTypeParam;
use crate::ast::declarations::{AsMember, Member};
use crate::ast::ruby::members::{Member as RubyMember, MixinMember as RubyMixinMember};
use crate::ast::types::Type as AstType;
use crate::environment::frozen::{
    ClassDeclaration, ClassEntry, ClassOrModule, Environment, ModuleDeclaration, ModuleEntry,
    NormalizeModuleNameResult,
};
use crate::location::{LocationRange, SourceLocation};
use crate::name::NameTable;
use crate::substitution::Substitution;
use crate::type_name::{Kind, TypeName};
use crate::type_param::{
    TypeParam, TypeParamScope, TypeVarKey, TypeVarScope, Variance, pad_ancestor_args,
};
use crate::types::{Ty, Type, TypeTable};
use std::path::PathBuf;

/// Common fields shared by Include/Extend/Prepend, extracted locally so the
/// three `bucket_*` functions share a single `&MixinCore` parameter type.
struct MixinCore {
    name: TypeName,
    args: Vec<AstType>,
    source_file: Option<crate::name::Name>,
    location: Option<LocationRange>,
}

/// 1-step ancestor data — direct super_class, mixin refs, and
/// (for modules) self_types. Mirrors rbs `OneAncestors`.
///
/// `super_class` is an [`Ancestor`] (not a [`MixinRef`]) so that singleton
/// chains can encode the rbs flip rule: a `BasicObject` singleton's super
/// is `Ancestor::Instance(::Class)` (instance chain), a module
/// singleton's super is `Ancestor::Instance(::Module)`, and an ordinary
/// class singleton's super is `Ancestor::Singleton(super_name)`. The
/// linearized walker dispatches on this variant to pick the right side.
#[derive(Debug, Clone, Default)]
pub struct OneAncestors {
    pub type_name: Option<TypeName>,
    pub params: Vec<TypeParam>,
    pub super_class: Option<Ancestor>,
    pub self_types: Vec<MixinRef>,
    pub included_modules: Vec<MixinRef>,
    pub included_interfaces: Vec<MixinRef>,
    pub prepended_modules: Vec<MixinRef>,
    pub extended_modules: Vec<MixinRef>,
    pub extended_interfaces: Vec<MixinRef>,
}

/// How an [`Ancestor::Instance`] entered the chain. Mirrors rbs's
/// `Definition::Ancestor::Instance#source` which is one of `nil`,
/// `:super`, an `AST::Members::Include`, or an `AST::Members::Prepend`.
///
/// rbs uses `nil` as a sentinel meaning "the self_ancestor of this very
/// chain"; `fill_ancestor_source` then overwrites it with the right
/// relationship when the inner chain is folded into an outer one (e.g.
/// `M` is `nil` in M's own chain, but becomes `Include` in C's chain
/// when `class C; include M; end`). Rust drops the nullability and
/// uses [`Self::SelfDecl`] as the same sentinel.
///
/// Singleton ancestors do not carry a source — `extend` walks the
/// singleton chain via [`SingletonAncestors`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AncestorSource {
    /// This ancestor is the chain's self_ancestor (rbs's `nil` source).
    /// Overwritten by [`fill_ancestor_source`] when folded into an outer
    /// chain.
    SelfDecl,
    /// `< Super` (rbs's `:super` symbol).
    Super,
    /// `include Mod` / `include Iface` (rbs's `AST::Members::Include`
    /// or `AST::Ruby::Members::IncludeMember`).
    Include,
    /// `prepend Mod` (rbs's `AST::Members::Prepend` or
    /// `AST::Ruby::Members::PrependMember`).
    Prepend,
    /// `extend Mod` / `extend Iface` (rbs's `AST::Members::Extend` or
    /// `AST::Ruby::Members::ExtendMember`). Lives on instance ancestors
    /// produced for the *extended module's own chain* — folded onto the
    /// singleton chain root of the extender via
    /// [`SingletonAncestors`]. Constant lookup walks
    /// `instance_ancestors` only and so naturally skips this variant.
    Extend,
}

/// One node in a linearized ancestor chain. Mirrors
/// `RBS::Definition::Ancestor::Instance | Singleton`. The `source`
/// field on [`Self::Instance`] records the relationship that brought
/// the ancestor into the outermost chain (super / include / prepend /
/// self) and is consumed by passes that need to filter the chain (the
/// most prominent being constant lookup, which excludes `prepend` per
/// rbs `Resolver::ConstantResolver`).
#[derive(Debug, Clone)]
pub enum Ancestor {
    Instance {
        name: TypeName,
        args: Vec<Ty>,
        source: AncestorSource,
    },
    Singleton {
        name: TypeName,
    },
}

impl Ancestor {
    /// Project an `Ancestor::Instance` edge into the `MixinRef` shape
    /// callers (arity validator, lookup wrappers) already use for
    /// `include` / `extend` / `prepend` edges. Returns `None` for the
    /// `Singleton` arm because singleton edges do not carry type
    /// arguments and would not produce a meaningful `MixinRef`.
    /// `location` is `None` because ancestor edges are synthesized from
    /// the linearization, not parsed from a source `Mixin` member.
    pub fn to_instance_mixin_ref(&self) -> Option<MixinRef> {
        match self {
            Ancestor::Instance { name, args, .. } => Some(MixinRef {
                name: *name,
                args: args.clone(),
                location: None,
            }),
            Ancestor::Singleton { .. } => None,
        }
    }
}

/// Fully linearized ancestors for an instance side (class, module, or
/// interface). Mirrors `RBS::Definition::InstanceAncestors`.
///
/// `params` is the root's type-param name list; [`Self::apply`] uses it
/// to build a [`Substitution`] for the caller-supplied args and rewrite
/// every `Ancestor::Instance.args` accordingly. Mixin / super args are
/// already substituted into the root's scope at build time, so a single
/// top-level apply propagates concrete args to the entire chain.
#[derive(Debug, Clone)]
pub struct InstanceAncestors {
    pub type_name: TypeName,
    pub params: Vec<TypeVarKey>,
    pub ancestors: Vec<Ancestor>,
}

impl InstanceAncestors {
    /// Substitute `args` into every `Ancestor::Instance.args` along the
    /// chain by binding `self.params -> args`. Length mismatch is silent
    /// per ADR-0010: extra args are dropped, missing args leave the
    /// corresponding param unsubstituted (the type variable survives).
    pub fn apply(&self, args: &[Ty], types: &TypeTable) -> Vec<Ancestor> {
        let subst = build_param_subst(&self.params, args);
        self.ancestors
            .iter()
            .map(|a| sub_ancestor(a, &subst, types))
            .collect()
    }
}

/// Fully linearized ancestors for the singleton side. Mirrors
/// `RBS::Definition::SingletonAncestors`. Singleton ancestors do not
/// propagate type args, so there is no `params` field and no `apply`.
#[derive(Debug, Clone)]
pub struct SingletonAncestors {
    pub type_name: TypeName,
    pub ancestors: Vec<Ancestor>,
}

/// Whether the build-phase mixin walk fills the instance buckets
/// (include / prepend) or the singleton buckets (extend) of the
/// receiving [`OneAncestors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Instance,
    Singleton,
}

/// `Clone` deep-copies every cache (`RefCell` clone copies the inner
/// map) — added for the same reason as [`Environment`]'s own `Clone`
/// (ADR-0028 `_internal incr-bench`, and now [`AncestorGraph`]'s
/// bench-only per-iteration clone — see its doc): a caller that keeps an
/// old generation alive alongside a delta-updated new one needs an
/// independent copy rather than sharing interior-mutable cache state.
///
/// [`AncestorGraph`]: super::ancestor_graph::AncestorGraph
#[derive(Debug, Clone)]
pub struct AncestorBuilder {
    env: Arc<Environment>,
    /// `Arc`-shared (not owned) so [`Self::update`] can hand the same
    /// interner to the new generation's builder: `TypeTable::intern` is
    /// append-only, so every `Ty` cached inside a carried-over
    /// `OneAncestors` / `InstanceAncestors` / `SingletonAncestors` stays
    /// valid against a table that keeps growing across generations. This
    /// is the shared interner every `Ty` in the process indexes into, not
    /// the persisted Ty-*keyed cache* ADR-0028 Decision 3 forbids (that
    /// decision targets invalidation-unsound caches like `subtype_cache`).
    types: Arc<TypeTable>,
    /// `Arc`-shared declared-name membership + class-alias map for cheap
    /// `LoweringEnv` reconstruction. Built once in [`Self::new`]; every
    /// `lowering()` call goes through [`LoweringEnv::from_maps`] instead
    /// of re-walking every declaration.
    lowering_maps: Arc<LoweringMaps>,
    /// Name-keyed memoization of [`Self::one_instance_ancestors`].
    /// `OneAncestors` is a function of the canonical (post-`normalize_module_name`)
    /// `TypeName` and the frozen environment, both of which are immutable
    /// over `AncestorBuilder`'s lifetime, so cache invalidation is not a
    /// concern. The cache mirrors rbs `@one_instance_ancestors_cache` in
    /// `lib/rbs/definition_builder/ancestor_builder.rb`.
    one_instance_cache: RefCell<FxHashMap<TypeName, Arc<OneAncestors>>>,
    one_singleton_cache: RefCell<FxHashMap<TypeName, Arc<OneAncestors>>>,
    one_interface_cache: RefCell<FxHashMap<TypeName, Arc<OneAncestors>>>,
    /// Full-linearization memoization, mirroring rbs
    /// `@instance_ancestors_cache` / `@singleton_ancestors_cache` /
    /// `@interface_ancestors_cache`. Keyed by the same canonical
    /// `TypeName` as the `one_*_cache` and equally invalidation-free over
    /// the frozen environment. For acyclic ancestor graphs — the only
    /// shape rbs accepts — a type's linearization is a pure function of
    /// that type plus the environment, so caching it is sound.
    ///
    /// Cyclic ancestors (mutual includes) are malformed input that rbs
    /// rejects with `RecursiveAncestorError`. crema does not raise; the
    /// `visited` guard returns an empty chain at the re-entrant node, and
    /// the cache may then store a build-order-dependent partial for the
    /// cyclic types. This divergence is left unaddressed: it is bounded to
    /// declarations rbs would not accept, never panics, and termination
    /// still holds via `visited`.
    instance_ancestors_cache: RefCell<FxHashMap<TypeName, Arc<InstanceAncestors>>>,
    singleton_ancestors_cache: RefCell<FxHashMap<TypeName, Arc<SingletonAncestors>>>,
    interface_ancestors_cache: RefCell<FxHashMap<TypeName, Arc<InstanceAncestors>>>,
    /// Companion cache (crema-only, ADR-0028 S3b-2): for every `node`
    /// whose one-ancestors computation dropped a mixin/super reference
    /// because `resolve_mixin_target` found no declaration for the
    /// target (ADR-0010's silent skip), the set of *normalized* target
    /// names that node could not resolve. Not part of `OneAncestors`
    /// (that rbs-ported shape stays as-is) — a separate side channel
    /// populated alongside `one_instance_cache` / `one_singleton_cache`
    /// / `one_interface_cache` (same population trigger, same
    /// `except`-driven carry-over in [`Self::update`]).
    ///
    /// Recording the *normalized* name (not the literal written
    /// reference) matters for a dangling alias: `module A = ::Missing;
    /// class C; include A; end` records `::Missing` here, not `A` — so
    /// that when `::Missing` is later declared, `AncestorGraph`'s
    /// target-keyed reverse index (built from this cache, see
    /// `AncestorGraph::unresolved`) can find `C` from `::Missing` alone,
    /// without needing to know `A` was ever involved.
    ///
    unresolved: RefCell<FxHashMap<Node, FxHashSet<TypeName>>>,
}

impl AncestorBuilder {
    pub fn new(env: Arc<Environment>) -> Self {
        Self::with_types(env, TypeTable::new())
    }

    pub fn with_types(env: Arc<Environment>, types: TypeTable) -> Self {
        let (all_names, class_aliases) = build_lowering_maps(&env, env.names());
        Self {
            env,
            types: Arc::new(types),
            lowering_maps: Arc::new(LoweringMaps::from_owned_maps(all_names, class_aliases)),
            one_instance_cache: RefCell::new(FxHashMap::default()),
            one_singleton_cache: RefCell::new(FxHashMap::default()),
            one_interface_cache: RefCell::new(FxHashMap::default()),
            instance_ancestors_cache: RefCell::new(FxHashMap::default()),
            singleton_ancestors_cache: RefCell::new(FxHashMap::default()),
            interface_ancestors_cache: RefCell::new(FxHashMap::default()),
            unresolved: RefCell::new(FxHashMap::default()),
        }
    }

    /// The shared lowering maps — test-only today, to compare a
    /// delta-patched copy against a from-scratch build.
    #[cfg(test)]
    pub(crate) fn lowering_parts(&self) -> &Arc<LoweringMaps> {
        &self.lowering_maps
    }

    /// crema-only: rbs's `AncestorBuilder` has no `update` counterpart —
    /// `RBS::DefinitionBuilder#update` (`lib/rbs/definition_builder.rb:1018-1034`)
    /// takes an already-built `ancestor_builder:` from its caller and never
    /// constructs one incrementally itself (confirmed: no `def update` in
    /// `lib/rbs/definition_builder/ancestor_builder.rb`). This method is
    /// what `DefinitionBuilder::update` calls to obtain a genuinely
    /// incremental `ancestor_builder` instead of falling back to
    /// `AncestorBuilder::new` (an O(env) rebuild); its merge-then-delete-except
    /// shape mirrors the two rbs-ported siblings (`MethodBuilder#update`,
    /// `DefinitionBuilder#update`) for consistency, not a literal rbs port.
    ///
    /// `except` is the invalidated `TypeName` set (see
    /// [`DefinitionBuilder::update`](crate::definition_builder::DefinitionBuilder::update)).
    /// `self.types` is shared (not rebuilt) with the returned builder —
    /// see the field's doc for why that is sound.
    ///
    /// `lowering_maps` is delta-patched from `self`'s copy (ADR-0028 S4b,
    /// S2 follow-up): touching only `except` plus `changed_constants`
    /// (`InvalidationResult::constants`; `except`/
    /// `InvalidationResult::type_names` deliberately excludes constants,
    /// see that field's doc, but this patch's name-membership check needs
    /// them too) instead of re-walking every declaration the way
    /// [`build_lowering_maps`] does. Correctness depends on `except` ∪
    /// `changed_constants` being a superset of every `TypeName` whose
    /// class/module/interface/type-alias/class-alias/constant declaration
    /// entry appeared or disappeared between the old and new environment
    /// — [`crate::environment::invalidation::invalidated_names`] already
    /// guarantees this for its own two fields.
    pub fn update(
        &self,
        env: Arc<Environment>,
        except: &FxHashSet<TypeName>,
        changed_constants: &FxHashSet<TypeName>,
    ) -> Self {
        let mut new_maps = (*self.lowering_maps).clone();
        let names = env.names();
        for &owner in except.iter().chain(changed_constants) {
            let n = names.intern(&names.resolve(owner));
            let declared = env.class_decls().contains_key(&owner)
                || env.interface_decls().contains_key(&owner)
                || env.type_alias_decls().contains_key(&owner)
                || env.constant_decls().contains_key(&owner)
                || env.class_alias_decls().contains_key(&owner);
            if declared {
                new_maps.insert_name(n);
            } else {
                new_maps.remove_name(n);
            }

            match env
                .class_alias_decls()
                .get(&owner)
                .and_then(|entry| class_alias_old_name(entry, names))
            {
                Some(old_n) => {
                    new_maps.insert_alias(n, old_n);
                }
                None => {
                    new_maps.remove_alias(n);
                }
            }
        }
        // Carry the `unresolved` cache forward minus the `except`ed
        // nodes; `record_unresolved` below re-populates any of them this
        // pass actually re-walks.
        let mut new_unresolved = self.unresolved.borrow().clone();
        for &name in except {
            for node in [Node::InstanceNode(name), Node::SingletonNode(name)] {
                new_unresolved.remove(&node);
            }
        }
        let result = Self {
            env,
            types: Arc::clone(&self.types),
            lowering_maps: Arc::new(new_maps),
            one_instance_cache: RefCell::new(FxHashMap::default()),
            one_singleton_cache: RefCell::new(FxHashMap::default()),
            one_interface_cache: RefCell::new(FxHashMap::default()),
            instance_ancestors_cache: RefCell::new(FxHashMap::default()),
            singleton_ancestors_cache: RefCell::new(FxHashMap::default()),
            interface_ancestors_cache: RefCell::new(FxHashMap::default()),
            unresolved: RefCell::new(new_unresolved),
        };
        carry_over_arc_cache(&self.one_instance_cache, &result.one_instance_cache, except);
        carry_over_arc_cache(
            &self.one_singleton_cache,
            &result.one_singleton_cache,
            except,
        );
        carry_over_arc_cache(
            &self.one_interface_cache,
            &result.one_interface_cache,
            except,
        );
        carry_over_arc_cache(
            &self.instance_ancestors_cache,
            &result.instance_ancestors_cache,
            except,
        );
        carry_over_arc_cache(
            &self.singleton_ancestors_cache,
            &result.singleton_ancestors_cache,
            except,
        );
        carry_over_arc_cache(
            &self.interface_ancestors_cache,
            &result.interface_ancestors_cache,
            except,
        );
        result
    }

    /// Cloned snapshot of `node`'s unresolved miss set — the set of
    /// normalized target names `node`'s own-ancestors computation could
    /// not resolve. Read by [`AncestorGraph`](super::ancestor_graph::AncestorGraph)
    /// to build/delta-update its target-keyed `unresolved` reverse index.
    pub(super) fn unresolved_misses(&self, node: Node) -> FxHashSet<TypeName> {
        self.unresolved
            .borrow()
            .get(&node)
            .cloned()
            .unwrap_or_default()
    }

    pub fn env(&self) -> &Environment {
        &self.env
    }

    pub fn types(&self) -> &TypeTable {
        &self.types
    }

    pub(crate) fn lowering(&self) -> LoweringEnv<'_> {
        LoweringEnv::from_maps(
            Arc::clone(&self.lowering_maps),
            self.env.names(),
            &self.types,
        )
    }

    /// Resolve own super_class / included / prepended / extended for an
    /// instance of `name`. Mirrors rbs
    /// `AncestorBuilder#one_instance_ancestors`. Memoized per
    /// canonical `TypeName` — the returned value deep-clones the cache
    /// entry's five mixin `Vec`s, so callers that only borrow the result
    /// are better served by [`Self::one_instance_ancestors_arc`].
    pub fn one_instance_ancestors(&self, name: &TypeName) -> OneAncestors {
        (*self.one_instance_ancestors_arc(name)).clone()
    }

    /// Resolve own singleton-side mixin chain for `name`. Memoized; see
    /// [`Self::one_instance_ancestors`].
    pub fn one_singleton_ancestors(&self, name: &TypeName) -> OneAncestors {
        (*self.one_singleton_ancestors_arc(name)).clone()
    }

    /// Resolve own included_interfaces for an interface `name`. Memoized;
    /// see [`Self::one_instance_ancestors`].
    pub fn one_interface_ancestors(&self, name: &TypeName) -> OneAncestors {
        (*self.one_interface_ancestors_arc(name)).clone()
    }

    /// `Arc`-sharing variant of [`Self::one_instance_ancestors`] that
    /// hands back the cache entry directly instead of deep-cloning its
    /// five mixin `Vec`s. Prefer this when the caller only borrows.
    pub fn one_instance_ancestors_arc(&self, name: &TypeName) -> Arc<OneAncestors> {
        let normalized = self.env.normalize_module_name(name);
        self.one_instance_cached(&normalized)
    }

    /// `Arc`-sharing variant of [`Self::one_singleton_ancestors`]; see
    /// [`Self::one_instance_ancestors_arc`].
    pub fn one_singleton_ancestors_arc(&self, name: &TypeName) -> Arc<OneAncestors> {
        let normalized = self.env.normalize_module_name(name);
        self.one_singleton_cached(&normalized)
    }

    /// `Arc`-sharing variant of [`Self::one_interface_ancestors`]; see
    /// [`Self::one_instance_ancestors_arc`].
    pub fn one_interface_ancestors_arc(&self, name: &TypeName) -> Arc<OneAncestors> {
        self.one_interface_cached(name)
    }

    /// Cached `Arc` view of [`Self::one_instance_ancestors`]. `name` must
    /// be the canonical (post-`env.normalize_module_name`) `TypeName`;
    /// passing a `class A = ::Real`-style alias produces a duplicate
    /// cache entry under the alias key.
    fn one_instance_cached(&self, name: &TypeName) -> Arc<OneAncestors> {
        if let Some(v) = self.one_instance_cache.borrow().get(name) {
            return Arc::clone(v);
        }
        let lowering = self.lowering();
        let mut misses = FxHashSet::default();
        let computed = Arc::new(self.one_instance_ancestors_with(name, &lowering, &mut misses));
        self.one_instance_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&computed));
        self.record_unresolved(Node::InstanceNode(*name), misses);
        computed
    }

    fn one_singleton_cached(&self, name: &TypeName) -> Arc<OneAncestors> {
        if let Some(v) = self.one_singleton_cache.borrow().get(name) {
            return Arc::clone(v);
        }
        let lowering = self.lowering();
        let mut misses = FxHashSet::default();
        let computed = Arc::new(self.one_singleton_ancestors_with(name, &lowering, &mut misses));
        self.one_singleton_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&computed));
        self.record_unresolved(Node::SingletonNode(*name), misses);
        computed
    }

    fn one_interface_cached(&self, name: &TypeName) -> Arc<OneAncestors> {
        if let Some(v) = self.one_interface_cache.borrow().get(name) {
            return Arc::clone(v);
        }
        let mut misses = FxHashSet::default();
        let computed = Arc::new(self.compute_one_interface_ancestors(name, &mut misses));
        self.one_interface_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&computed));
        self.record_unresolved(Node::InstanceNode(*name), misses);
        computed
    }

    /// Merge `misses` (ADR-0028 S3b-2) into [`Self::unresolved`] under
    /// `node`, skipping empty sets so the cache never holds an entry with
    /// no misses (matches the invariant the other `one_*_cache`s'
    /// insert-on-compute pattern keeps implicitly).
    fn record_unresolved(&self, node: Node, misses: FxHashSet<TypeName>) {
        if !misses.is_empty() {
            self.unresolved.borrow_mut().insert(node, misses);
        }
    }

    fn compute_one_interface_ancestors(
        &self,
        name: &TypeName,
        misses: &mut FxHashSet<TypeName>,
    ) -> OneAncestors {
        let entry = match self.env.interface_decls().get(name) {
            Some(e) => e,
            None => return OneAncestors::default(),
        };
        let decl = entry.decl();
        let scope = crate::definition_builder::build_class_param_scope(&decl.type_params, name);
        let lowering = self.lowering();
        let mut one = OneAncestors {
            type_name: Some(*name),
            params: ast_type_params_to_resolved(&decl.type_params, &lowering, &scope),
            ..OneAncestors::default()
        };

        // rbs `mixin_ancestors0` (lib/rbs/definition_builder/ancestor_builder.rb:350-)
        // is called with only `included_interfaces` non-nil for an interface
        // host, so:
        //   - `Include` whose target is itself an interface flows into
        //     `included_interfaces`;
        //   - `Include` of a class-shaped target is silently dropped
        //     (ADR-0010, mirroring rbs's `included_modules` nil branch);
        //   - `Extend` / `Prepend` are silently dropped (rbs `extended_*` /
        //     `prepended_modules` nil branches).
        //
        // A `Class`-kind resolution here (target exists, wrong shape for
        // an interface host) is *not* an ADR-0028 S3b-2 unresolved-target
        // miss: class-shaped and interface-shaped names are lexically
        // disjoint (interface names start with `_`), so this branch can
        // never flip to `Interface` by a later declaration change — only
        // a genuine `resolve_mixin_target` failure (target doesn't exist
        // at all) is recorded into `misses`.
        for member in &decl.members {
            if let Member::Include(inc) = member {
                let core = MixinCore {
                    name: inc.name,
                    args: inc.args.clone(),
                    source_file: inc.source_file,
                    location: inc.location.map(|l| l.range),
                };
                if let Some((mref, MixinTargetKind::Interface)) =
                    resolve_mixin_target(&core, &lowering, &self.env, &scope, misses)
                {
                    one.included_interfaces.push(mref);
                }
            }
        }
        one
    }

    /// Build-phase variant of [`Self::one_instance_ancestors`] that
    /// reuses a caller-supplied [`LoweringEnv`]. `name` must be the
    /// canonical post-`normalize_module_name` form — the cache helpers
    /// pre-normalize at the cache-key step, so this body skips the
    /// double normalize.
    pub(super) fn one_instance_ancestors_with(
        &self,
        name: &TypeName,
        lowering: &LoweringEnv<'_>,
        misses: &mut FxHashSet<TypeName>,
    ) -> OneAncestors {
        let entry = match self.env.class_decls().get(name) {
            Some(e) => e,
            None => return OneAncestors::default(),
        };

        let names = self.env.names();
        let class_scope = entry_class_param_scope(entry, name, names);
        let mut one = OneAncestors {
            type_name: Some(*name),
            ..OneAncestors::default()
        };

        match entry {
            ClassOrModule::Class(class_entry) => {
                one.params = class_type_params(class_entry, lowering, &class_scope);
                if !is_basic_object(name, names) {
                    let sup = class_super_or_default(
                        class_entry,
                        lowering,
                        &self.env,
                        names,
                        &class_scope,
                    );
                    one.super_class = Some(Ancestor::Instance {
                        name: sup.name,
                        args: sup.args,
                        source: AncestorSource::Super,
                    });
                }
                walk_class_mixins(
                    class_entry,
                    name,
                    lowering,
                    &self.env,
                    Side::Instance,
                    &class_scope,
                    &mut one,
                    names,
                    misses,
                );
            }
            ClassOrModule::Module(module_entry) => {
                one.params = module_type_params(module_entry, lowering, &class_scope);
                one.self_types = module_self_types_or_default(
                    module_entry,
                    name,
                    lowering,
                    &self.env,
                    names,
                    &class_scope,
                );
                walk_module_mixins(
                    module_entry,
                    name,
                    lowering,
                    &self.env,
                    Side::Instance,
                    &class_scope,
                    &mut one,
                    names,
                    misses,
                );
            }
        }
        one
    }

    /// Same canonical-name precondition as [`Self::one_instance_ancestors_with`].
    pub(super) fn one_singleton_ancestors_with(
        &self,
        name: &TypeName,
        lowering: &LoweringEnv<'_>,
        misses: &mut FxHashSet<TypeName>,
    ) -> OneAncestors {
        let entry = match self.env.class_decls().get(name) {
            Some(e) => e,
            None => return OneAncestors::default(),
        };

        let names = self.env.names();
        let class_scope = entry_class_param_scope(entry, name, names);
        let mut one = OneAncestors {
            type_name: Some(*name),
            ..OneAncestors::default()
        };

        match entry {
            ClassOrModule::Class(class_entry) => {
                if is_basic_object(name, names) {
                    // BasicObject's singleton routes to ::Class's instance chain.
                    one.super_class = Some(Ancestor::Instance {
                        name: names.builtins().class,
                        args: Vec::new(),
                        source: AncestorSource::Super,
                    });
                } else {
                    let sup = class_super_or_default(
                        class_entry,
                        lowering,
                        &self.env,
                        names,
                        &class_scope,
                    );
                    one.super_class = Some(Ancestor::Singleton { name: sup.name });
                }
                walk_class_mixins(
                    class_entry,
                    name,
                    lowering,
                    &self.env,
                    Side::Singleton,
                    &class_scope,
                    &mut one,
                    names,
                    misses,
                );
            }
            ClassOrModule::Module(module_entry) => {
                // Module singletons inherit ::Module's instance chain.
                one.super_class = Some(Ancestor::Instance {
                    name: names.builtins().module,
                    args: Vec::new(),
                    source: AncestorSource::Super,
                });
                walk_module_mixins(
                    module_entry,
                    name,
                    lowering,
                    &self.env,
                    Side::Singleton,
                    &class_scope,
                    &mut one,
                    names,
                    misses,
                );
            }
        }
        one
    }

    /// Build the fully linearized instance ancestors of `name`.
    ///
    /// Mirrors rbs `AncestorBuilder#instance_ancestors`. Mixin / super
    /// args are substituted into the root's scope at build time, so a
    /// single [`InstanceAncestors::apply`] at the top binds concrete
    /// args throughout the chain.
    pub fn instance_ancestors(&self, name: &TypeName) -> Arc<InstanceAncestors> {
        self.build_instance_ancestors(name, &mut FxHashSet::default())
    }

    /// Build the fully linearized singleton ancestors of `name`.
    pub fn singleton_ancestors(&self, name: &TypeName) -> Arc<SingletonAncestors> {
        self.build_singleton_ancestors(name, &mut FxHashSet::default())
    }

    /// Build the fully linearized interface ancestors of `name`.
    pub fn interface_ancestors(&self, name: &TypeName) -> Arc<InstanceAncestors> {
        self.build_interface_ancestors(name, &mut FxHashSet::default())
    }

    fn build_instance_ancestors(
        &self,
        name: &TypeName,
        visited: &mut FxHashSet<TypeName>,
    ) -> Arc<InstanceAncestors> {
        let normalized = self.env.normalize_module_name(name);

        // Full-linearization cache check, mirroring rbs ancestor_builder.rb:493.
        // Runs *before* the cycle guard so a completed build is reused when
        // reached recursively. For acyclic graphs the stored chain is the
        // canonical one (cycle divergence is documented on the cache field).
        if let Some(cached) = self.instance_ancestors_cache.borrow().get(&normalized) {
            return Arc::clone(cached);
        }

        let one = self.one_instance_cached(&normalized);
        let params: Vec<TypeVarKey> = one.params.iter().map(|tp| tp.name.clone()).collect();

        // Stack-style cycle guard (mirrors rbs's `building_ancestors` push/pop):
        // present-during-this-recursion only, *not* a deduplicating cache.
        // Removing on return is what lets the same module appear twice when
        // it is mixed in along two different chains (e.g. `class B < A;
        // include M; end` where A also includes M — Ruby yields M in both
        // positions). A persistent set would silently drop the second M.
        // The empty result returned here is the cyclic-input divergence
        // documented on the cache field (rbs raises instead).
        if !visited.insert(normalized) {
            return Arc::new(InstanceAncestors {
                type_name: normalized,
                params,
                ancestors: Vec::new(),
            });
        }

        let self_args: Vec<Ty> = params
            .iter()
            .map(|k| {
                self.types.intern(Type::TypeVariable {
                    raw: k.raw,
                    scope: k.scope.clone(),
                })
            })
            .collect();
        let self_ancestor = Ancestor::Instance {
            name: normalized,
            args: self_args,
            source: AncestorSource::SelfDecl,
        };

        let mut ancestors: Vec<Ancestor> = Vec::new();

        if let Some(Ancestor::Instance {
            name: super_name,
            args: super_args,
            ..
        }) = &one.super_class
        {
            let super_ia = self.build_instance_ancestors(super_name, visited);
            let mut super_subs = super_ia.apply(super_args, &self.types);
            fill_ancestor_source(&mut super_subs, super_name, AncestorSource::Super);
            prepend_in_place(&mut ancestors, super_subs);
        }

        for mixin in &one.included_modules {
            let mod_ia = self.build_instance_ancestors(&mixin.name, visited);
            let mut mod_subs = mod_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut mod_subs, &mixin.name, AncestorSource::Include);
            prepend_in_place(&mut ancestors, mod_subs);
        }

        for mixin in &one.included_interfaces {
            let iface_ia = self.build_interface_ancestors(&mixin.name, visited);
            let mut iface_subs = iface_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut iface_subs, &mixin.name, AncestorSource::Include);
            prepend_in_place(&mut ancestors, iface_subs);
        }

        ancestors.insert(0, self_ancestor);

        for mixin in &one.prepended_modules {
            let mod_ia = self.build_instance_ancestors(&mixin.name, visited);
            let mut mod_subs = mod_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut mod_subs, &mixin.name, AncestorSource::Prepend);
            prepend_in_place(&mut ancestors, mod_subs);
        }

        visited.remove(&normalized);
        let result = Arc::new(InstanceAncestors {
            type_name: normalized,
            params,
            ancestors,
        });
        self.instance_ancestors_cache
            .borrow_mut()
            .insert(normalized, Arc::clone(&result));
        result
    }

    fn build_singleton_ancestors(
        &self,
        name: &TypeName,
        visited: &mut FxHashSet<TypeName>,
    ) -> Arc<SingletonAncestors> {
        let normalized = self.env.normalize_module_name(name);

        // Full-linearization cache check before the cycle guard. See
        // `build_instance_ancestors`.
        if let Some(cached) = self.singleton_ancestors_cache.borrow().get(&normalized) {
            return Arc::clone(cached);
        }

        let self_ancestor = Ancestor::Singleton { name: normalized };

        // Stack-style cycle guard. See `build_instance_ancestors`.
        if !visited.insert(normalized) {
            return Arc::new(SingletonAncestors {
                type_name: normalized,
                ancestors: Vec::new(),
            });
        }

        let one = self.one_singleton_cached(&normalized);
        let mut ancestors: Vec<Ancestor> = Vec::new();

        match &one.super_class {
            Some(Ancestor::Instance {
                name: super_name,
                args: super_args,
                ..
            }) => {
                // BasicObject / Module singleton flips to instance chain.
                let super_ia = self.build_instance_ancestors(super_name, &mut FxHashSet::default());
                let mut super_subs = super_ia.apply(super_args, &self.types);
                fill_ancestor_source(&mut super_subs, super_name, AncestorSource::Super);
                prepend_in_place(&mut ancestors, super_subs);
            }
            Some(Ancestor::Singleton { name: super_name }) => {
                let super_sa = self.build_singleton_ancestors(super_name, visited);
                prepend_in_place(&mut ancestors, super_sa.ancestors.clone());
            }
            None => {}
        }

        for mixin in &one.extended_modules {
            let mod_ia = self.build_instance_ancestors(&mixin.name, &mut FxHashSet::default());
            let mut mod_subs = mod_ia.apply(&mixin.args, &self.types);
            // Match rbs L607-614: `singleton_ancestors`' extended_modules
            // loop fills the chain root with `mod.source`, i.e. the
            // extend relationship. Deeper ancestors inside the extended
            // module's own chain keep their original (Include / Super)
            // tag — `fill_ancestor_source` rewrites only the root.
            fill_ancestor_source(&mut mod_subs, &mixin.name, AncestorSource::Extend);
            prepend_in_place(&mut ancestors, mod_subs);
        }
        for mixin in &one.extended_interfaces {
            let iface_ia = self.build_interface_ancestors(&mixin.name, &mut FxHashSet::default());
            let mut iface_subs = iface_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut iface_subs, &mixin.name, AncestorSource::Extend);
            prepend_in_place(&mut ancestors, iface_subs);
        }

        ancestors.insert(0, self_ancestor);

        for mixin in &one.prepended_modules {
            let mod_ia = self.build_instance_ancestors(&mixin.name, &mut FxHashSet::default());
            let mut mod_subs = mod_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut mod_subs, &mixin.name, AncestorSource::Prepend);
            prepend_in_place(&mut ancestors, mod_subs);
        }

        visited.remove(&normalized);
        let result = Arc::new(SingletonAncestors {
            type_name: normalized,
            ancestors,
        });
        self.singleton_ancestors_cache
            .borrow_mut()
            .insert(normalized, Arc::clone(&result));
        result
    }

    fn build_interface_ancestors(
        &self,
        name: &TypeName,
        visited: &mut FxHashSet<TypeName>,
    ) -> Arc<InstanceAncestors> {
        // Full-linearization cache check before the cycle guard. Interfaces
        // are keyed by `name` directly (no `normalize_module_name`), matching
        // `one_interface_cached`. See `build_instance_ancestors`.
        if let Some(cached) = self.interface_ancestors_cache.borrow().get(name) {
            return Arc::clone(cached);
        }

        let one = self.one_interface_cached(name);
        let params: Vec<TypeVarKey> = one.params.iter().map(|tp| tp.name.clone()).collect();

        // Stack-style cycle guard. See `build_instance_ancestors`.
        if !visited.insert(*name) {
            return Arc::new(InstanceAncestors {
                type_name: *name,
                params,
                ancestors: Vec::new(),
            });
        }

        let self_args: Vec<Ty> = params
            .iter()
            .map(|k| {
                self.types.intern(Type::TypeVariable {
                    raw: k.raw,
                    scope: k.scope.clone(),
                })
            })
            .collect();
        let self_ancestor = Ancestor::Instance {
            name: *name,
            args: self_args,
            source: AncestorSource::SelfDecl,
        };

        let mut ancestors: Vec<Ancestor> = Vec::new();

        for mixin in &one.included_interfaces {
            let iface_ia = self.build_interface_ancestors(&mixin.name, visited);
            let mut iface_subs = iface_ia.apply(&mixin.args, &self.types);
            fill_ancestor_source(&mut iface_subs, &mixin.name, AncestorSource::Include);
            prepend_in_place(&mut ancestors, iface_subs);
        }

        ancestors.insert(0, self_ancestor);

        visited.remove(name);
        let result = Arc::new(InstanceAncestors {
            type_name: *name,
            params,
            ancestors,
        });
        self.interface_ancestors_cache
            .borrow_mut()
            .insert(*name, Arc::clone(&result));
        result
    }
}

/// Copy every `src` entry not in `except` into `dst`. Shared by
/// [`AncestorBuilder::update`]'s six `TypeName`-keyed `Arc` caches.
fn carry_over_arc_cache<V>(
    src: &RefCell<FxHashMap<TypeName, Arc<V>>>,
    dst: &RefCell<FxHashMap<TypeName, Arc<V>>>,
    except: &FxHashSet<TypeName>,
) {
    let mut dst_map = dst.borrow_mut();
    for (name, value) in src.borrow().iter() {
        if !except.contains(name) {
            dst_map.insert(*name, Arc::clone(value));
        }
    }
}

fn prepend_in_place(target: &mut Vec<Ancestor>, mut prefix: Vec<Ancestor>) {
    prefix.append(target);
    *target = prefix;
}

fn build_param_subst(params: &[TypeVarKey], args: &[Ty]) -> Substitution {
    let mut bindings: FxHashMap<TypeVarKey, Ty> = FxHashMap::default();
    for (p, a) in params.iter().zip(args.iter()) {
        bindings.insert(p.clone(), *a);
    }
    Substitution::from_mapping(bindings)
}

fn sub_ancestor(a: &Ancestor, subst: &Substitution, types: &TypeTable) -> Ancestor {
    match a {
        Ancestor::Instance { name, args, source } => {
            if args.is_empty() {
                Ancestor::Instance {
                    name: *name,
                    args: Vec::new(),
                    source: *source,
                }
            } else {
                let new_args: Vec<Ty> = args.iter().map(|&t| subst.apply(t, types)).collect();
                Ancestor::Instance {
                    name: *name,
                    args: new_args,
                    source: *source,
                }
            }
        }
        Ancestor::Singleton { name } => Ancestor::Singleton { name: *name },
    }
}

/// Overwrite the `SelfDecl` source on the chain's root ancestor with
/// the relationship that brought this chain into an outer chain.
/// Mirrors rbs `fill_ancestor_source` (definition_builder/ancestor_builder.rb
/// L664-675): only ancestors whose name matches `root` and whose source is
/// still the `SelfDecl` sentinel are rewritten — already-tagged ancestors
/// (from a deeper fold) keep their source.
fn fill_ancestor_source(chain: &mut [Ancestor], root: &TypeName, new_source: AncestorSource) {
    for ancestor in chain.iter_mut() {
        if let Ancestor::Instance { name, source, .. } = ancestor
            && name == root
            && *source == AncestorSource::SelfDecl
        {
            *source = new_source;
        }
    }
}

fn is_basic_object(name: &TypeName, names: &NameTable) -> bool {
    name == &names.builtins().basic_object
}

pub(crate) fn primary_signature_class_type_params(entry: &ClassEntry) -> &[AstTypeParam] {
    match entry.primary_decl() {
        ClassDeclaration::Signature(c) => &c.type_params,
        ClassDeclaration::Ruby(_) => &[],
    }
}

pub(crate) fn primary_signature_module_type_params(entry: &ModuleEntry) -> &[AstTypeParam] {
    match entry.primary_decl() {
        ModuleDeclaration::Signature(m) => &m.type_params,
        ModuleDeclaration::Ruby(_) => &[],
    }
}

fn class_type_params(
    entry: &ClassEntry,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
) -> Vec<TypeParam> {
    ast_type_params_to_resolved(
        primary_signature_class_type_params(entry),
        lowering,
        class_scope,
    )
}

fn module_type_params(
    entry: &ModuleEntry,
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
) -> Vec<TypeParam> {
    ast_type_params_to_resolved(
        primary_signature_module_type_params(entry),
        lowering,
        class_scope,
    )
}

/// Lower the AST-side `TypeParam` list to the resolved-side `TypeParam`,
/// folding the `upper_bound` / `lower_bound` / `default_type` AST types
/// into `Ty` via `lowering.build_type`. `class_scope` carries the same
/// type-param names so a bound that references a sibling param (e.g.
/// `class C[T, U < T]`) resolves into a `TypeVariable` instead of an
/// untyped fallback.
///
/// `TypeParam.name` is set to the **alpha-renamed scoped Name** (e.g.
/// `::Foo.T`) — the same Name that `lowering.build_type` mints inside
/// `Type::TypeVariable`. This keeps `InstanceAncestors::apply` honest:
/// the substitution it builds keys on the same Name it is about to
/// rewrite. Using the raw `tp.name` here would silently break chain
/// substitution because the apply'd subst would never match.
pub(crate) fn ast_type_params_to_resolved(
    raws: &[AstTypeParam],
    lowering: &LoweringEnv<'_>,
    class_scope: &TypeParamScope,
) -> Vec<TypeParam> {
    raws.iter()
        .map(|tp| {
            let key = class_scope
                .get(&tp.name)
                .cloned()
                .unwrap_or_else(|| TypeVarKey::new(tp.name, TypeVarScope::Free));
            let mut p = TypeParam::new(key, Variance::from_ast(tp.variance));
            p.unchecked = tp.unchecked;
            p.upper_bound = tp
                .upper_bound
                .as_ref()
                .map(|t| lowering.build_type(t, class_scope));
            p.lower_bound = tp
                .lower_bound
                .as_ref()
                .map(|t| lowering.build_type(t, class_scope));
            p.default_type = tp
                .default_type
                .as_ref()
                .map(|t| lowering.build_type(t, class_scope));
            p
        })
        .collect()
}

fn entry_class_param_scope(
    entry: &ClassOrModule,
    name: &TypeName,
    _names: &NameTable,
) -> TypeParamScope {
    let raws = match entry {
        ClassOrModule::Class(c) => primary_signature_class_type_params(c),
        ClassOrModule::Module(m) => primary_signature_module_type_params(m),
    };
    crate::definition_builder::build_class_param_scope(raws, name)
}

/// Unified projection of a class's `super` reference.
///
/// rbs's `RBS::AST::Declarations::Class::Super` (signature side) and
/// `RBS::AST::Ruby::Declarations::ClassDecl::SuperClass` (Ruby side)
/// carry the same logical information — target name, optional type
/// args, optional source location — but spell their fields
/// differently (`args: Vec<Type>` vs `type_annotation.type_args`,
/// `location: Option<Location>` vs `byte_range: Option<PrismByteRange>`).
/// This view collapses the two into one shape so [`mixin_ref`] is
/// invoked once and the signature / Ruby branches stay structurally
/// identical.
struct SuperView<'a> {
    name: TypeName,
    args: &'a [crate::ast::types::Type],
    source_file: Option<crate::name::Name>,
    location: Option<LocationRange>,
}

fn class_super_or_default(
    entry: &ClassEntry,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    names: &NameTable,
    class_scope: &TypeParamScope,
) -> MixinRef {
    let view = match entry.primary_decl() {
        ClassDeclaration::Signature(c) => c.super_class.as_ref().map(|s| SuperView {
            name: s.name,
            args: &s.args,
            source_file: s.source_file,
            location: s.location.map(|l| l.range),
        }),
        ClassDeclaration::Ruby(c) => c.super_class.as_ref().map(|s| SuperView {
            name: s.type_name,
            args: s
                .type_annotation
                .as_ref()
                .map(|a| a.type_args.as_slice())
                .unwrap_or(&[]),
            source_file: None,
            location: None,
        }),
    };
    if let Some(v) = view
        && names.type_name_is_absolute(v.name)
    {
        return mixin_ref(
            v.name,
            v.args,
            v.source_file,
            v.location,
            lowering,
            env,
            class_scope,
        );
    }
    MixinRef::bare(names.builtins().object)
}

fn module_self_types_or_default(
    entry: &ModuleEntry,
    name: &TypeName,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    names: &NameTable,
    class_scope: &TypeParamScope,
) -> Vec<MixinRef> {
    // rbs `ModuleEntry#self_types` aggregates `each_decl.flat_map(&:self_types).uniq`,
    // aligning each decl's type params onto the primary decl's spelling
    // (`align_params`) before the `uniq`. Reading only `primary_decl()`
    // mis-builds reopens like Kernel, where the self-type-bearing decl is
    // not necessarily the first one loaded; lowering every decl under the
    // primary scope leaves a reopen-spelled `_Each[Elem]` unbound.
    let primary_params = primary_signature_module_type_params(entry);
    let mut resolved: Vec<MixinRef> = Vec::new();
    let mut push_unique = |mref: MixinRef| {
        if !resolved
            .iter()
            .any(|existing| existing.name == mref.name && existing.args == mref.args)
        {
            resolved.push(mref);
        }
    };
    for (_, _, decl) in entry.context_decls() {
        let (self_types, decl_scope): (
            Vec<crate::ast::declarations::ModuleSelf>,
            std::borrow::Cow<'_, TypeParamScope>,
        ) = match decl {
            ModuleDeclaration::Signature(m) => (
                m.self_types.clone(),
                std::borrow::Cow::Owned(decl_param_scope(
                    &m.type_params,
                    primary_params,
                    name,
                    names,
                )),
            ),
            ModuleDeclaration::Ruby(m) => (m.self_types(), std::borrow::Cow::Borrowed(class_scope)),
        };
        for st in self_types
            .iter()
            .filter(|st| names.type_name_is_absolute(st.name))
        {
            push_unique(mixin_ref(
                st.name,
                &st.args,
                st.source_file,
                st.location.map(|l| l.range),
                lowering,
                env,
                &decl_scope,
            ));
        }
    }
    if resolved.is_empty() {
        vec![MixinRef::bare(names.builtins().object)]
    } else {
        resolved
    }
}

/// Per-decl scope for lowering a reopen's mixin / self_types args: the
/// decl's own type-param spelling is alpha-renamed onto the primary
/// decl's (rbs `ClassEntry#align_params`, applied by
/// `AncestorBuilder#mixin_ancestors` and `ModuleEntry#self_types`).
fn decl_param_scope(
    decl_params: &[AstTypeParam],
    primary_params: &[AstTypeParam],
    owner: &TypeName,
    names: &NameTable,
) -> TypeParamScope {
    crate::definition_builder::build_alpha_renamed_param_scope(
        decl_params,
        primary_params,
        owner,
        names,
    )
}

fn mixin_ref(
    name: TypeName,
    args: &[crate::ast::types::Type],
    source_file: Option<crate::name::Name>,
    location: Option<LocationRange>,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    class_scope: &TypeParamScope,
) -> MixinRef {
    let lowered: Vec<Ty> = args
        .iter()
        .map(|t| lowering.build_type(t, class_scope))
        .collect();
    let padded = pad_with_target_defaults(&name, lowered, lowering, env);
    let source_location = source_file.zip(location).map(|(f, range)| SourceLocation {
        file: PathBuf::from(env.names().resolve(f)),
        range,
    });
    MixinRef {
        name,
        args: padded,
        location: source_location,
    }
}

/// Apply `pad_ancestor_args` using the target's declared type params, mirroring
/// rbs `AST::TypeParam.normalize_args` at every mixin / super lowering site
/// (`lib/rbs/definition_builder/ancestor_builder.rb:224, 253, 257, 365, ...`).
/// Targets unknown to the environment fall through unchanged so ADR-0010's
/// silent-skip behaviour for unresolved names is preserved.
fn pad_with_target_defaults(
    name: &TypeName,
    args: Vec<Ty>,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
) -> Vec<Ty> {
    let Some(params) = target_resolved_type_params(name, lowering, env) else {
        return args;
    };
    pad_ancestor_args(&params, &args, lowering.types)
}

/// Resolve the declared, fully-lowered `Vec<TypeParam>` for a mixin target.
/// Class / module names go through `normalize_module_name` to follow
/// `class A = ::Real` aliases; interface targets are looked up by their
/// original name since interfaces are not aliased through the same map.
fn target_resolved_type_params(
    name: &TypeName,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
) -> Option<Vec<TypeParam>> {
    let names = env.names();
    let normalized = env.normalize_module_name(name);
    if let Some(entry) = env.class_decls().get(&normalized) {
        let scope = entry_class_param_scope(entry, &normalized, names);
        return Some(match entry {
            ClassOrModule::Class(c) => class_type_params(c, lowering, &scope),
            ClassOrModule::Module(m) => module_type_params(m, lowering, &scope),
        });
    }
    if let Some(entry) = env.interface_decls().get(name) {
        let decl = entry.decl();
        let scope = crate::definition_builder::build_class_param_scope(&decl.type_params, name);
        return Some(ast_type_params_to_resolved(
            &decl.type_params,
            lowering,
            &scope,
        ));
    }
    None
}

#[derive(Clone, Copy, Debug)]
enum MixinTargetKind {
    Class,
    Interface,
}

impl From<MixinTargetKind> for Kind {
    fn from(k: MixinTargetKind) -> Self {
        match k {
            MixinTargetKind::Class => Kind::Class,
            MixinTargetKind::Interface => Kind::Interface,
        }
    }
}

/// Decide whether a mixin's target lives in `class_decls` (after alias
/// normalization) or `interface_decls`. Returns `None` if neither map
/// has the name — the build phase silently skips unknown targets per
/// ADR-0010. Also returns `None` for relative (unresolved) `TypeName`s.
fn classify_mixin_target(env: &Environment, name: &TypeName) -> Option<MixinTargetKind> {
    if !env.names().type_name_is_absolute(*name) {
        return None;
    }
    let normalized = env.normalize_module_name(name);
    if env.class_decls().contains_key(&normalized) {
        return Some(MixinTargetKind::Class);
    }
    if env.interface_decls().contains_key(name) {
        return Some(MixinTargetKind::Interface);
    }
    None
}

/// Whether `name` currently classifies as a mixin/super target in `env`
/// — same notion of existence as [`classify_mixin_target`], but safe to
/// call for a `name` this generation's `env` never interned at all.
///
/// `classify_mixin_target` (via `type_name_is_absolute` /
/// `normalize_module_name`) indexes the `TypeNameInterner`'s
/// `entries` map directly and panics ("no entry found for key") if
/// `name` is missing from it — a pre-existing bug when comparing across
/// two independently-diverged `Environment` generations (concurrent fix
/// in progress, see `environment::invalidation`'s seed computation
/// comment). Declaration-map membership (`class_decls` /
/// `interface_decls` / `class_alias_decls`) is a plain hashmap probe
/// that never touches the interner, and `name` can only be a key in one
/// of those maps if *this* `env`'s own build interned it — so checking
/// membership first before ever calling `classify_mixin_target` avoids
/// the panic without needing to fix it here. Used by
/// [`AncestorGraph::update`](super::ancestor_graph::AncestorGraph::update)'s
/// seed-extension rule (existence of a seed name changing between
/// generations pulls in its old graph children / unresolved referrers).
pub(super) fn mixin_target_exists(env: &Environment, name: TypeName) -> bool {
    if env.class_decls().contains_key(&name)
        || env.interface_decls().contains_key(&name)
        || env.class_alias_decls().contains_key(&name)
    {
        classify_mixin_target(env, &name).is_some()
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_class_mixins(
    entry: &ClassEntry,
    name: &TypeName,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    names: &NameTable,
    misses: &mut FxHashSet<TypeName>,
) {
    let primary_params = primary_signature_class_type_params(entry);
    for (_, _, decl) in entry.context_decls() {
        match decl {
            ClassDeclaration::Signature(c) => {
                let decl_scope = decl_param_scope(&c.type_params, primary_params, name, names);
                for member in &c.members {
                    dispatch_mixin_member(
                        member,
                        lowering,
                        env,
                        side,
                        &decl_scope,
                        class_scope,
                        one,
                        misses,
                    );
                }
            }
            ClassDeclaration::Ruby(c) => {
                for member in &c.members {
                    dispatch_ruby_mixin_member(
                        member,
                        lowering,
                        env,
                        side,
                        class_scope,
                        one,
                        names,
                        misses,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_module_mixins(
    entry: &ModuleEntry,
    name: &TypeName,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    names: &NameTable,
    misses: &mut FxHashSet<TypeName>,
) {
    let primary_params = primary_signature_module_type_params(entry);
    for (_, _, decl) in entry.context_decls() {
        match decl {
            ModuleDeclaration::Signature(m) => {
                let decl_scope = decl_param_scope(&m.type_params, primary_params, name, names);
                for member in &m.members {
                    dispatch_mixin_member(
                        member,
                        lowering,
                        env,
                        side,
                        &decl_scope,
                        class_scope,
                        one,
                        misses,
                    );
                }
            }
            ModuleDeclaration::Ruby(m) => {
                for member in &m.members {
                    dispatch_ruby_mixin_member(
                        member,
                        lowering,
                        env,
                        side,
                        class_scope,
                        one,
                        names,
                        misses,
                    );
                }
            }
        }
    }
}

/// Route a single member to the bucket function for its mixin variant.
/// The dispatch was previously a `(legacy mixin kind, Side, Kind)`
/// 3-tuple match inside one `bucket_mixin`; with Include / Extend /
/// Prepend now distinct AST variants, the keyword discriminator lives
/// at the type level and only `(Side, Kind)` filtering remains
/// inside each bucket function.
///
/// `decl_scope` is the per-decl alpha-renamed scope used for `include` /
/// `prepend` args; `extend` args are lowered under the primary
/// `class_scope` unchanged because rbs `mixin_ancestors0` applies
/// `align_params` only to Include / Prepend (`ancestor_builder.rb:363,
/// 392` vs the bare `member.args` at `:400`) — class type params are
/// not in scope on the singleton side, so there is nothing to align
/// (rbs `validate` rejects `extend M[E]` with `NoTypeFoundError`).
#[allow(clippy::too_many_arguments)]
fn dispatch_mixin_member<M: AsMember>(
    wrapper: &M,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    decl_scope: &TypeParamScope,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    misses: &mut FxHashSet<TypeName>,
) {
    let Some(member) = wrapper.as_member() else {
        return;
    };
    match member {
        Member::Include(inc) => {
            bucket_include(
                &MixinCore {
                    name: inc.name,
                    args: inc.args.clone(),
                    source_file: inc.source_file,
                    location: inc.location.map(|l| l.range),
                },
                lowering,
                env,
                side,
                decl_scope,
                one,
                misses,
            );
        }
        Member::Extend(ext) => {
            bucket_extend(
                &MixinCore {
                    name: ext.name,
                    args: ext.args.clone(),
                    source_file: ext.source_file,
                    location: ext.location.map(|l| l.range),
                },
                lowering,
                env,
                side,
                class_scope,
                one,
                misses,
            );
        }
        Member::Prepend(pre) => {
            bucket_prepend(
                &MixinCore {
                    name: pre.name,
                    args: pre.args.clone(),
                    source_file: pre.source_file,
                    location: pre.location.map(|l| l.range),
                },
                lowering,
                env,
                side,
                decl_scope,
                one,
                misses,
            );
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ruby_mixin_member(
    member: &RubyMember,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    names: &NameTable,
    misses: &mut FxHashSet<TypeName>,
) {
    match member {
        RubyMember::Include(inc) => {
            if let Some(mixin) = mixin_from_ruby(&inc.mixin, names) {
                bucket_include(&mixin, lowering, env, side, class_scope, one, misses);
            }
        }
        RubyMember::Extend(ext) => {
            if let Some(mixin) = mixin_from_ruby(&ext.mixin, names) {
                bucket_extend(&mixin, lowering, env, side, class_scope, one, misses);
            }
        }
        RubyMember::Prepend(pre) => {
            if let Some(mixin) = mixin_from_ruby(&pre.mixin, names) {
                bucket_prepend(&mixin, lowering, env, side, class_scope, one, misses);
            }
        }
        RubyMember::SingletonPrepend(pre) => {
            if side == Side::Singleton
                && let Some(mixin) = mixin_from_ruby(&pre.mixin, names)
                && let Some((mref, _target_kind)) =
                    resolve_mixin_target(&mixin, lowering, env, class_scope, misses)
            {
                one.prepended_modules.push(mref);
            }
        }
        _ => {}
    }
}

/// Build a [`MixinCore`] view from a Ruby-side [`RubyMixinMember`].
///
/// After the build phase, `module_name` is an absolute path string (e.g.
/// `"::M"`). Returns `None` for relative names so the caller can silently
/// skip them, matching rbs's behaviour for unresolved mixin targets
/// (ADR-0010). The `starts_with("::")` guard is required before calling
/// `parse_class_name` because that function panics on relative names.
fn mixin_from_ruby(member: &RubyMixinMember, names: &NameTable) -> Option<MixinCore> {
    if !member.module_name.starts_with("::") {
        return None;
    }
    let name = names.parse_type_name(&member.module_name);
    let args = member
        .annotation
        .as_ref()
        .map(|a| a.type_args.clone())
        .unwrap_or_default();
    Some(MixinCore {
        name,
        args,
        source_file: None,
        location: None,
    })
}

/// Resolve the mixin target name and lower its type arguments. Returns
/// `None` when the target is unknown — ADR-0010 silently skips such
/// mixins. Shared by all three bucket functions.
///
/// `misses` (ADR-0028 S3b-2): on a `None` return for a genuinely
/// absolute target name, records `env.normalize_module_name(&mixin.name)`
/// — the normalized identity a future declaration would need to match
/// for this reference to newly resolve (see the `unresolved` field doc
/// on [`AncestorBuilder`] for why the *normalized* name, not the literal
/// `mixin.name`, is the right key). A relative name is a parse-time
/// artifact with no addressable identity — matches
/// `classify_mixin_target`'s own early return and is not recorded.
fn resolve_mixin_target(
    mixin: &MixinCore,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    class_scope: &TypeParamScope,
    misses: &mut FxHashSet<TypeName>,
) -> Option<(MixinRef, MixinTargetKind)> {
    let Some(target_kind) = classify_mixin_target(env, &mixin.name) else {
        if env.names().type_name_is_absolute(mixin.name) {
            misses.insert(unresolved_target_name(env, mixin.name));
        }
        return None;
    };
    let mref = mixin_ref(
        mixin.name,
        &mixin.args,
        mixin.source_file,
        mixin.location,
        lowering,
        env,
        class_scope,
    );
    Some((mref, target_kind))
}

/// The identity a future declaration would need to match for `name`
/// (a mixin/super reference [`classify_mixin_target`] just failed to
/// resolve) to newly resolve.
///
/// **Not** `env.normalize_module_name(&name)` — that convenience
/// wrapper folds a dangling alias chain (`module A = ::Missing` where
/// `::Missing` is undeclared) back to `A` itself (the *original* query
/// name), per [`NormalizeModuleNameResult::UnknownTarget`]'s doc. `A`'s
/// own declaration is untouched by `::Missing` later getting declared,
/// so recording under `A` would never let a future existence-check on
/// `::Missing` find this referrer. [`Environment::normalize_module_name_result`]
/// exposes the chain's *true* dangling end (`UnknownTarget.target`,
/// possibly several aliases deep) — that is the name whose future
/// declaration actually changes this resolution's outcome, so it is the
/// right key. Every other outcome (`Normalized` / `Cycle` /
/// `NotClassOrModule`) already matches what `normalize_module_name`
/// itself would return, so this only special-cases `UnknownTarget`.
fn unresolved_target_name(env: &Environment, name: TypeName) -> TypeName {
    match env.normalize_module_name_result(&name) {
        NormalizeModuleNameResult::UnknownTarget { target, .. } => target,
        NormalizeModuleNameResult::Normalized(t) => t,
        NormalizeModuleNameResult::Cycle { original }
        | NormalizeModuleNameResult::NotClassOrModule { original } => original,
    }
}

/// `include` contributes only to the instance-side ancestor chain.
/// Unlike `prepend`, class-shaped and interface-shaped targets land in
/// distinct buckets so cycle / linearization checks downstream can
/// inspect them independently.
fn bucket_include(
    mixin: &MixinCore,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    misses: &mut FxHashSet<TypeName>,
) {
    if side != Side::Instance {
        return;
    }
    let Some((mref, target_kind)) = resolve_mixin_target(mixin, lowering, env, class_scope, misses)
    else {
        return;
    };
    match target_kind {
        MixinTargetKind::Class => one.included_modules.push(mref),
        MixinTargetKind::Interface => one.included_interfaces.push(mref),
    }
}

/// `extend` contributes only to the singleton-side ancestor chain.
/// As with `include`, class-shaped and interface-shaped targets are
/// kept in separate buckets.
fn bucket_extend(
    mixin: &MixinCore,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    misses: &mut FxHashSet<TypeName>,
) {
    if side != Side::Singleton {
        return;
    }
    let Some((mref, target_kind)) = resolve_mixin_target(mixin, lowering, env, class_scope, misses)
    else {
        return;
    };
    match target_kind {
        MixinTargetKind::Class => one.extended_modules.push(mref),
        MixinTargetKind::Interface => one.extended_interfaces.push(mref),
    }
}

/// `prepend` contributes only to the instance-side ancestor chain.
/// Unlike `include` / `extend`, class-shaped and interface-shaped
/// targets share the same `prepended_modules` bucket — the rbs gem
/// does not distinguish them on the prepend side.
fn bucket_prepend(
    mixin: &MixinCore,
    lowering: &LoweringEnv<'_>,
    env: &Environment,
    side: Side,
    class_scope: &TypeParamScope,
    one: &mut OneAncestors,
    misses: &mut FxHashSet<TypeName>,
) {
    if side != Side::Instance {
        return;
    }
    let Some((mref, _target_kind)) =
        resolve_mixin_target(mixin, lowering, env, class_scope, misses)
    else {
        return;
    };
    one.prepended_modules.push(mref);
}
