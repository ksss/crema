//! Port of `RBS::Resolver::ConstantResolver` — constant-name lookup that
//! mirrors Ruby's lexical scope and ancestor chain semantics.
//!
//! This module is the constant-name counterpart to
//! [`crate::environment::resolution::TypeNameResolver`]. Where the latter
//! resolves *type* names against `all_names` + `aliases`, this one resolves
//! *constant* names — class / module / module-alias / constant decls —
//! against a lexical scope context and the ancestor chain.
//!
//! Mirrors `rbs/lib/rbs/resolver/constant_resolver.rb`:
//!
//! - [`ConstantTable`] (rbs `Resolver::ConstantResolver::Table`):
//!   precomputes three maps — per-module child constants, top-level
//!   constants, and the "class as a constant" entries.
//! - [`ConstantResolver`] (rbs `Resolver::ConstantResolver`): public
//!   `resolve` / `constants` / `resolve_child` / `children` API, with the
//!   same two cache lanes (context-key, module-key) for repeat lookups.
//! - [`ConstantContext`] (rbs `Resolver::context` — `nil | [parent, last]`):
//!   the lexical scope passed to [`ConstantResolver::resolve`]. rbs's
//!   linked-list shape is flattened to an outer-first slice; the trailing
//!   element corresponds to rbs's `last`, the rest to `parent`. Empty
//!   represents rbs's `nil` (top-level only).
//!
//! Sources are filtered through [`super::AncestorSource`]: child constant
//! lookup walks `instance_ancestors` and pulls children only from `Include`
//! / `Super` / `SelfDecl` ancestors, deliberately skipping `Prepend` to
//! match rbs (constants in a prepended module are *not* visible to the
//! including class — rbs `constant_resolver.rb` L151-156).

use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use crate::environment::Environment;
use crate::environment::frozen::NormalizeModuleNameResult;
use crate::name::{NameTable, Symbol};
use crate::snapshot::append_map::AppendMap;
use crate::type_name::TypeName;
use crate::type_param::TypeParamScope;
use crate::types::Ty;

use super::ancestor_builder::{Ancestor, AncestorBuilder, AncestorSource};

/// Lexical scope passed to constant lookup.
///
/// Mirrors rbs's `Resolver::context = nil | [parent_context, last_typename]`.
/// rbs's recursive linked list is represented here as an outer-first
/// slice: index 0 is the outermost scope, the trailing element is rbs's
/// `last`, and an empty `ConstantContext` is rbs's `nil` (top-level
/// only). Stored in a `Box<[TypeName]>` so the value is owned and can
/// serve as a `FxHashMap` key without lifetime entanglement —
/// [`ConstantResolver`]'s `context_constants_cache` consumes the
/// `PartialEq + Eq + Hash` impl for cache keying.
///
/// [`ConstantResolver`]: super::ConstantResolver
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct ConstantContext {
    scopes: Box<[TypeName]>,
}

impl ConstantContext {
    /// Top-level scope (rbs's `nil` context).
    pub fn toplevel() -> Self {
        Self::default()
    }

    /// Build a new context by pushing `scope` onto `parent` as the
    /// innermost (rbs's `last`) typename. Cloning `parent`'s slice is
    /// `O(n)` but contexts are short (lexical nesting depth) and the
    /// resulting value is structurally interned by the cache, so the
    /// cost amortizes across repeat lookups.
    pub fn with_scope(parent: &Self, scope: TypeName) -> Self {
        let mut next = Vec::with_capacity(parent.scopes.len() + 1);
        next.extend(parent.scopes.iter().cloned());
        next.push(scope);
        Self {
            scopes: next.into_boxed_slice(),
        }
    }

    /// rbs's `last` — the innermost lexical scope, or `None` for
    /// top-level.
    pub fn last(&self) -> Option<&TypeName> {
        self.scopes.last()
    }

    /// rbs's `parent` — drop the innermost scope. `None` when already
    /// at top-level (consistent with rbs's `nil` parent of `nil`).
    pub fn parent(&self) -> Option<Self> {
        if self.scopes.is_empty() {
            None
        } else {
            let (_, init) = self.scopes.split_last()?;
            Some(Self {
                scopes: init.to_vec().into_boxed_slice(),
            })
        }
    }

    /// Outer-first slice of the lexical scope stack. Empty slice means
    /// top-level.
    pub fn scopes(&self) -> &[TypeName] {
        &self.scopes
    }
}

/// A resolved constant value. Mirrors rbs
/// `Resolver::ConstantResolver::Constant` — name plus the type the
/// constant evaluates to plus a back-reference to the source that
/// produced it. The back-reference (`origin`) is what lets downstream
/// consumers ask "is this a module-as-constant or a real `Foo: T` decl?".
#[derive(Debug, Clone)]
pub struct ResolverConstant {
    /// Absolute name of the constant. For `constant_of_module` this is
    /// the class/module name itself; for `constant_of_constant` this is
    /// the constant's qualified name. For aliases the rbs port preserves
    /// the *alias* name here (not the normalized target) — see
    /// rbs L34-46.
    pub name: TypeName,
    /// Static type the constant evaluates to. For modules this is
    /// `singleton(name)`; for constants this is the lowered `decl.ty`.
    pub ty: Ty,
    /// Provenance — discriminates module / module-alias / constant.
    pub origin: ConstantOrigin,
}

/// Which kind of declaration produced a [`ResolverConstant`]. Mirrors
/// the rbs distinction between `constant_of_module` (class / module /
/// class-alias / module-alias decl) and `constant_of_constant`
/// (`FOO: T` constant decl).
///
/// `Constant` carries no payload today — downstream consumers only
/// branch on the variant (e.g. to detect an intermediate value mid
/// path resolution). If location / comment info becomes load-bearing
/// later, reintroduce the `Arc<Constant>` payload then.
#[derive(Debug, Clone)]
pub enum ConstantOrigin {
    /// class / module / class-alias / module-alias declaration. The
    /// associated `TypeName` carries the alias-normalized target if
    /// the constant originated from a class-alias / module-alias.
    Module { target: TypeName },
    /// `FOO: T` constant declaration.
    Constant,
}

/// Compact name-set entry that stands in for a [`ResolverConstant`]
/// inside [`ConstantTable`]'s children / toplevel / constants maps.
///
/// Splits the two roles Pass 4 used to fuse together — name-set
/// membership vs. `.ty` lowering. Membership is eager (`Copy`, small
/// enum), while `.ty` is deferred to [`ConstantTable::hydrate`] so
/// `constant_of_constant` entries only decode their backing
/// `constant_decls` entry when a caller actually resolves the name.
///
/// Modules keep an eager form (`class_singleton(name)` is a cheap
/// intern that never touches the snapshot backend), and only
/// `Constant` needs the lazy hydration path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantEntry {
    /// `constant_of_module`: class / module / class-alias / module-alias.
    /// `name` mirrors rbs's alias-preserving `Constant.name` (the alias
    /// itself for aliases, the decl name otherwise); `target` is the
    /// alias-normalized destination (equal to `name` for non-aliases).
    Module { name: TypeName, target: TypeName },
    /// `constant_of_constant`: a plain `FOO: T` decl. The `.ty` is
    /// lowered lazily via [`ConstantTable::hydrate`] and cached in
    /// `ConstantTable::constant_ty_cache`.
    Constant { name: TypeName },
}

/// Precomputed constant-name table — port of
/// `RBS::Resolver::ConstantResolver::Table` (constant_resolver.rb L7-83).
///
/// Two indices — see [`ConstantTableRepr`]:
///
/// - `children_table` — for every class / module name, the map of
///   directly-nested constants visible *inside* that scope. Keyed by
///   the leaf symbol so a lookup against a lexical scope is `O(1)`.
/// - `toplevel` — constants whose namespace is empty (rbs's
///   `toplevel`). The `class_decls` walk routes top-level
///   class / module decls here; the `class_alias_decls` walk routes
///   top-level class/module aliases here; the `constant_decls` walk
///   routes top-level `FOO: T` here.
///
/// rbs's third index, `constants_table` (the `singleton(name)`
/// self-entry `constants_itself` uses to answer "what is `A` evaluated
/// as a constant inside `class A`'s body?"), has no field here — see
/// [`Self::constant`]'s doc.
///
/// [`ConstantResolver`]: super::ConstantResolver
#[derive(Debug)]
pub struct ConstantTable {
    repr: ConstantTableRepr,
    /// Per-name lazy cache for `constant_of_constant` `.ty` lowering.
    /// `Module` entries hydrate cheaply through
    /// `TypeTable::class_singleton` and never touch this cache; only
    /// `Constant` variants hit it, at most once per name, decoding the
    /// backing `constant_decls` entry and lowering its `decl.ty`.
    constant_ty_cache: ConstantTyCache,
}

/// Per-owner constants map, shared via `Arc` so both
/// [`ConstantTableRepr`] variants return the same cheap-to-clone type
/// from `children`/`toplevel`.
type OwnerConstants = Arc<FxHashMap<Symbol, ConstantEntry>>;

/// Backing storage for [`ConstantTable`]: `Owned` is the full in-memory
/// build [`ConstantTable::new`] always produces (cold build,
/// `DefinitionBuilder::update`'s per-generation rebuild — see that
/// method's doc for why it cannot avoid this).
#[derive(Debug)]
enum ConstantTableRepr {
    Owned {
        /// `Arc`-wrapped so [`ConstantTable::update`] can share the whole
        /// map into an [`ConstantTableRepr::Updated`] base without an
        /// O(owners) clone.
        children_table: Arc<FxHashMap<TypeName, OwnerConstants>>,
        toplevel: OwnerConstants,
    },
    /// Delta-updated table for a changed run: a shared base (the
    /// previous generation's full tables) plus per-owner recomputed
    /// overlays. Produced only by [`ConstantTable::update`]; chained
    /// updates flatten their overlays into one layer so a long-lived
    /// process never stacks more than base + one overlay.
    Updated {
        base: UpdatedBase,
        /// Recomputed owners: `Some(map)` replaces the base entry,
        /// `None` is a tombstone (the owner slot does not exist in the
        /// new generation — mirrors `ConstantTable::new` producing no
        /// `children_table` key for it).
        children_overlay: FxHashMap<TypeName, Option<OwnerConstants>>,
        /// Per-symbol toplevel patches over the base's toplevel map,
        /// applied lazily in [`ConstantTable::toplevel`].
        toplevel_patch: FxHashMap<Symbol, Option<ConstantEntry>>,
        toplevel_cache: OnceLock<OwnerConstants>,
    },
}

/// Base layer of [`ConstantTableRepr::Updated`]: where an un-overlaid
/// owner lookup falls through to.
#[derive(Debug, Clone)]
struct UpdatedBase {
    children_table: Arc<FxHashMap<TypeName, OwnerConstants>>,
    toplevel: OwnerConstants,
}

/// Per-name lazy cache for `constant_of_constant` `.ty` values.
///
/// Mirrors [`crate::definition_builder::TypeParamsCache`]'s shape (an
/// `AppendMap`-backed wrapper with a manual `Debug` impl and a single
/// `get_or_compute` accessor) so the two sibling per-name caches
/// present the same surface. The pattern keeps `ConstantTable`'s
/// `#[derive(Debug)]` valid and hides the `Ok::<_, Infallible>` /
/// `.expect("infallible compute")` boilerplate at the boundary.
struct ConstantTyCache {
    cache: AppendMap<TypeName, Ty>,
}

impl std::fmt::Debug for ConstantTyCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstantTyCache")
            .field("cached", &self.cache.len())
            .finish()
    }
}

impl ConstantTyCache {
    fn new() -> Self {
        Self {
            cache: AppendMap::default(),
        }
    }

    /// Return the cached `.ty` for `name`, lowering it on first hit.
    /// `name` must have been recorded from `env.constant_decls().keys()`
    /// during `ConstantTable::new`; a missing decl at lookup time
    /// signals a broken invariant (the frozen `Environment` mutated
    /// under us) and panics rather than silently synthesizing a value.
    ///
    /// Lowers through `builder.lowering()` — the `Arc<LoweringMaps>` the
    /// builder already holds — rather than `LoweringEnv::from_environment`,
    /// which re-walks every declaration (`build_lowering_maps`, O(env)
    /// with a String round-trip per name). Paying that walk once per
    /// constant made it the dominant cost of a cold check on large
    /// environments (gitlab: 87% of main-thread CPU).
    fn get_or_compute(&self, name: TypeName, builder: &AncestorBuilder) -> Ty {
        *self
            .cache
            .get_or_try_insert_with(name, || {
                let decl = builder.env().constant_decls().get(&name).expect(
                    "ConstantEntry::Constant name was recorded from \
                     env.constant_decls().keys(); decl must still be present",
                );
                let lowering = builder.lowering();
                Ok::<_, std::convert::Infallible>(
                    lowering.build_type(&decl.decl.ty, &TypeParamScope::default()),
                )
            })
            .expect("infallible compute")
    }
}

impl ConstantTable {
    /// Walk the frozen environment and build `children_table` /
    /// `toplevel`. Mirrors rbs `Table#initialize`
    /// (constant_resolver.rb L10-61), minus the `constants_table` bookkeeping
    /// — see [`Self::constant`]'s doc for why that index isn't built here:
    ///
    /// 1. Empty `children_table` slots for every class / module key.
    /// 2. Walk `class_decls`: register `constant_of_module` under the
    ///    parent's children_table (or `toplevel` for namespace-empty
    ///    names).
    /// 3. Walk `class_alias_decls`: resolve the alias chain, register
    ///    the alias name + normalized target under children_table or
    ///    toplevel.
    /// 4. Walk `constant_decls`: register `constant_of_constant` under
    ///    children_table or toplevel — via `.keys()` so the entry
    ///    payloads stay decode-free until a caller resolves a name.
    ///    The `.ty` lowering is deferred to [`Self::hydrate`].
    ///
    /// `O(env)` — every caller of this constructor
    /// (`DefinitionBuilder::from_environment`/`::update`) pays that cost
    /// every time.
    pub fn new(builder: &AncestorBuilder) -> Self {
        let env = builder.env();

        let mut children_table: FxHashMap<TypeName, FxHashMap<Symbol, ConstantEntry>> =
            FxHashMap::with_capacity_and_hasher(env.class_decls().len(), Default::default());
        let mut toplevel: FxHashMap<Symbol, ConstantEntry> = FxHashMap::default();

        // Pass 1: pre-seed children_table so every class/module name has
        // a (possibly empty) children map. rbs builds this up-front so
        // later `or raise` lookups never miss.
        for name in env.class_decls().keys() {
            children_table.insert(*name, FxHashMap::default());
        }

        // Pass 2: class / module decls themselves become constants of
        // their parent scope.
        for name in env.class_decls().keys() {
            let entry = ConstantEntry::Module {
                name: *name,
                target: *name,
            };
            insert_into_scope(name, entry, &mut children_table, &mut toplevel, env);
        }

        // Pass 3: class/module aliases resolve through
        // `normalize_module_name_result`. Cycles, unknown targets, and
        // wrong-kind names are skipped (rbs L35: `or next`).
        for alias_name in env.class_alias_decls().keys() {
            let target = match env.normalize_module_name_result(alias_name) {
                NormalizeModuleNameResult::Normalized(t) => t,
                NormalizeModuleNameResult::UnknownTarget { .. }
                | NormalizeModuleNameResult::Cycle { .. }
                | NormalizeModuleNameResult::NotClassOrModule { .. } => continue,
            };
            // The target must point at a real class/module decl (rbs's
            // `module_class_entry(... normalized: true)` returns nil
            // otherwise, triggering `next`). Skip if the resolved name
            // is not in class_decls.
            if !env.class_decls().contains_key(&target) {
                continue;
            }
            let entry = ConstantEntry::Module {
                name: *alias_name,
                target,
            };
            insert_into_scope(alias_name, entry, &mut children_table, &mut toplevel, env);
        }

        // Pass 4: plain `FOO: T` constants. Iterate `.keys()` — the
        // decode-free key walk — and record only the name; `.ty`
        // lowering is deferred to `hydrate` on first `resolve` /
        // `resolve_child`. ADR-0028 Decision 3 (Ty is not
        // snapshot-persistable) rules out roasting the lowered form,
        // so per-name lazy is the only warm-run reduction available.
        for name in env.constant_decls().keys() {
            let entry = ConstantEntry::Constant { name: *name };
            insert_into_scope(name, entry, &mut children_table, &mut toplevel, env);
        }

        Self {
            repr: ConstantTableRepr::Owned {
                children_table: Arc::new(
                    children_table
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect(),
                ),
                toplevel: Arc::new(toplevel),
            },
            constant_ty_cache: ConstantTyCache::new(),
        }
    }

    /// Materialize a [`ResolverConstant`] from a stored entry, using
    /// `builder`'s env + type table for the `constant_of_constant`
    /// lowering path.
    ///
    /// - `Module` entries reconstruct `class_singleton(name)` on the
    ///   fly (an intern, no decode) and never allocate cache slots.
    /// - `Constant` entries decode the backing `constant_decls` entry
    ///   and lower `decl.ty` on the first hit, then reuse the cached
    ///   [`Ty`] for every subsequent lookup. `Ty` is `Copy` so the
    ///   cache holds `Ty` directly, not an `Arc`.
    fn hydrate(&self, entry: &ConstantEntry, builder: &AncestorBuilder) -> ResolverConstant {
        match *entry {
            ConstantEntry::Module { name, target } => {
                let ty = builder.types().class_singleton(name);
                ResolverConstant {
                    name,
                    ty,
                    origin: ConstantOrigin::Module { target },
                }
            }
            ConstantEntry::Constant { name } => {
                let ty = self.constant_ty_cache.get_or_compute(name, builder);
                ResolverConstant {
                    name,
                    ty,
                    origin: ConstantOrigin::Constant,
                }
            }
        }
    }

    /// Direct-children map for `name`. Returns `None` when `name` is
    /// not a known class / module — rbs's `children_table` only has
    /// pre-seeded slots for `class_decls` keys.
    pub fn children(&self, name: &TypeName) -> Option<OwnerConstants> {
        match &self.repr {
            ConstantTableRepr::Owned { children_table, .. } => children_table.get(name).cloned(),
            ConstantTableRepr::Updated {
                base,
                children_overlay,
                ..
            } => {
                if let Some(patched) = children_overlay.get(name) {
                    return patched.as_ref().map(Arc::clone);
                }
                base.children_table.get(name).cloned()
            }
        }
    }

    /// "Class as a constant" entry — what `A` evaluates to inside its
    /// own body. Aliases are deliberately absent here (rbs L34-46).
    /// Currently unused externally; kept for parity with rbs's
    /// `constants_table` API in case a downstream consumer needs the
    /// self-entry directly.
    ///
    /// Not a stored field (ADR-0028 F4c): [`Self::new`]'s Pass 2 is the
    /// *only* writer of rbs's `constants_table`, and every entry it
    /// inserts is `Module { name, target: name }` for `name` in
    /// `env.class_decls().keys()` — Pass 3/4 never touch it. That makes
    /// `constants_table` a pure projection of data the caller already
    /// has (`Environment::class_decls`), so it is computed on demand
    /// here instead of persisted as a fourth flat table.
    #[allow(dead_code)]
    pub fn constant(&self, name: &TypeName, env: &Environment) -> Option<ConstantEntry> {
        env.class_decls()
            .contains_key(name)
            .then_some(ConstantEntry::Module {
                name: *name,
                target: *name,
            })
    }

    /// Top-level constants (rbs `toplevel`). Re-exposed so the resolver
    /// can splice them into `::Object`'s children for class-context
    /// lookups (rbs L182-184, L191-194). `pub(crate)` (was
    /// `pub(super)`) so the differential tests can compare a
    /// delta-updated table's toplevel against a full rebuild's.
    /// `Updated` applies its per-symbol patches lazily.
    pub(crate) fn toplevel(&self) -> OwnerConstants {
        match &self.repr {
            ConstantTableRepr::Owned { toplevel, .. } => Arc::clone(toplevel),
            ConstantTableRepr::Updated {
                base,
                toplevel_patch,
                toplevel_cache,
                ..
            } => Arc::clone(toplevel_cache.get_or_init(|| {
                let base_map = Arc::clone(&base.toplevel);
                if toplevel_patch.is_empty() {
                    return base_map;
                }
                let mut map = (*base_map).clone();
                for (sym, patch) in toplevel_patch {
                    match patch {
                        Some(entry) => {
                            map.insert(*sym, *entry);
                        }
                        None => {
                            map.remove(sym);
                        }
                    }
                }
                Arc::new(map)
            })),
        }
    }

    /// `true` when this table was delta-updated via [`Self::update`]
    /// instead of walked fresh via [`Self::new`]. The changed-warm-run
    /// counterpart pin — a test asserting this after
    /// `DefinitionBuilder::update` proves the O(env)
    /// `ConstantTable::new` walk never fired for that generation.
    #[cfg(test)]
    pub(crate) fn is_updated(&self) -> bool {
        matches!(self.repr, ConstantTableRepr::Updated { .. })
    }

    /// Delta-update this table to `new_env`'s generation, recomputing
    /// only owners touched by the changed-name sets instead of
    /// re-walking the whole environment the way [`Self::new`] does
    /// (ADR-0028 Decision 3's "no O(env) on a changed run" applied to
    /// the constant table; follow-up to F4c which fixed the *unchanged*
    /// run).
    ///
    /// `except` is the invalidated type-name set
    /// (`InvalidationResult::type_names` — a superset of the seed names
    /// whose table contribution can actually differ; extra descendants
    /// recompute to identical values) and `constants` the changed
    /// `FOO: T` names (`InvalidationResult::constants`). `old_env` must
    /// be the generation this table was built for and `new_env` the one
    /// being switched to; both share the same content-addressed
    /// `NameTable` lineage, so `TypeName`s compare across them.
    ///
    /// Correctness of the touched-owner derivation (why this is closed
    /// over the changed sets, with no full-walk fallback):
    ///
    /// 1. A decl whose own file changed is in `except` / `constants`
    ///    (invalidation seed rule 1); its old and new contribution
    ///    slots are marked directly.
    /// 2. The only way an *unchanged* decl's entry moves is its parent
    ///    namespace's alias resolution changing (`insert_into_scope`
    ///    files entries under `normalize_module_name(parent)`). The
    ///    alias name itself is in `except` (its decl changed), so
    ///    [`Self::update`] detects the owner change and re-marks every
    ///    entry of both the old and the new owner — the previous
    ///    generation's own children map serves as the reverse index
    ///    "which decls contributed here" (each [`ConstantEntry`] carries
    ///    its alias-preserving decl name), so no extra persisted index
    ///    is needed.
    /// 3. A marked slot is then recomputed from `new_env` alone: the
    ///    candidate decl names are the marked names plus the owner's
    ///    direct child plus aliased parents' children (via a reverse
    ///    alias index built from `class_alias_decls` — O(aliases), not
    ///    O(env)), evaluated in [`Self::new`]'s pass-precedence order
    ///    (constant > alias > class/module, mirroring later-pass-wins).
    pub(crate) fn update(
        &self,
        old_env: &Environment,
        new_env: &Environment,
        except: &FxHashSet<TypeName>,
        constants: &FxHashSet<TypeName>,
    ) -> Self {
        // Reverse alias index over the new generation: normalized
        // target -> alias names. Only resolvable aliases whose target
        // is a real class/module contribute entries (mirrors `new`'s
        // Pass 3 skip conditions).
        let mut rev_alias: FxHashMap<TypeName, Vec<TypeName>> = FxHashMap::default();
        for alias in new_env.class_alias_decls().keys() {
            if let NormalizeModuleNameResult::Normalized(target) =
                new_env.normalize_module_name_result(alias)
                && new_env.class_decls().contains_key(&target)
            {
                rev_alias.entry(target).or_default().push(*alias);
            }
        }

        // Phase 1: mark every (owner, symbol) slot whose value may have
        // changed, remembering which changed decl names hit it.
        let mut toplevel_slots: FxHashMap<Symbol, FxHashSet<TypeName>> = FxHashMap::default();
        let mut children_slots: FxHashMap<TypeName, FxHashMap<Symbol, FxHashSet<TypeName>>> =
            FxHashMap::default();

        for &name in except.iter().chain(constants.iter()) {
            mark_slot(old_env, name, &mut toplevel_slots, &mut children_slots);
            mark_slot(new_env, name, &mut toplevel_slots, &mut children_slots);
        }

        for &name in except.iter().chain(constants.iter()) {
            // Pre-seed lifecycle: a class/module gaining or losing its
            // decl gains or loses its (possibly empty) children slot.
            if old_env.class_decls().contains_key(&name)
                || new_env.class_decls().contains_key(&name)
            {
                children_slots.entry(name).or_default();
            }

            // Alias moves (doc point 2): when `name`'s children-owner
            // resolution differs between generations, every entry of
            // both owners is re-marked so unchanged decls filed under
            // the alias's namespace relocate.
            let alias_involved = old_env.class_alias_decls().contains_key(&name)
                || new_env.class_alias_decls().contains_key(&name);
            if alias_involved {
                let owner_old = children_owner_for(old_env, name);
                let owner_new = children_owner_for(new_env, name);
                if owner_old != owner_new {
                    let mut remark = Vec::new();
                    for owner in [owner_old, owner_new] {
                        if let Some(entries) = self.children(&owner) {
                            remark.extend(entries.values().map(entry_decl_name));
                        }
                    }
                    for decl in remark {
                        mark_slot(old_env, decl, &mut toplevel_slots, &mut children_slots);
                        mark_slot(new_env, decl, &mut toplevel_slots, &mut children_slots);
                    }
                }
            }
        }

        // Phase 2: flatten the previous layer's overlay (keeps the repr
        // at base + one overlay across chained updates), then recompute
        // every marked slot from `new_env`.
        let (base, mut children_overlay, mut toplevel_patch) = match &self.repr {
            ConstantTableRepr::Owned {
                children_table,
                toplevel,
            } => (
                UpdatedBase {
                    children_table: Arc::clone(children_table),
                    toplevel: Arc::clone(toplevel),
                },
                FxHashMap::default(),
                FxHashMap::default(),
            ),
            ConstantTableRepr::Updated {
                base,
                children_overlay,
                toplevel_patch,
                ..
            } => (
                base.clone(),
                children_overlay.clone(),
                toplevel_patch.clone(),
            ),
        };

        for (owner, slots) in children_slots {
            let mut map: FxHashMap<Symbol, ConstantEntry> = self
                .children(&owner)
                .map(|m| (*m).clone())
                .unwrap_or_default();
            for (sym, marked) in slots {
                match recompute_children_slot(new_env, owner, sym, &marked, &rev_alias) {
                    Some(entry) => {
                        map.insert(sym, entry);
                    }
                    None => {
                        map.remove(&sym);
                    }
                }
            }
            // An owner slot exists in a fresh walk iff it is pre-seeded
            // (a class/module decl) or some entry landed in it
            // (`insert_into_scope`'s `or_default`).
            let exists = new_env.class_decls().contains_key(&owner) || !map.is_empty();
            children_overlay.insert(owner, exists.then(|| Arc::new(map)));
        }

        for (sym, marked) in toplevel_slots {
            toplevel_patch.insert(sym, recompute_toplevel_slot(new_env, sym, &marked));
        }

        Self {
            repr: ConstantTableRepr::Updated {
                base,
                children_overlay,
                toplevel_patch,
                toplevel_cache: OnceLock::new(),
            },
            constant_ty_cache: ConstantTyCache::new(),
        }
    }
}

/// `true` when `env` declares `name` in any of the three maps
/// [`ConstantTable::new`] walks (class/module, class-alias, constant) —
/// i.e. when `name` can contribute a table entry on that side. An
/// unresolvable alias still returns `true`: it contributed nothing, but
/// marking its slot is harmless (the recompute yields the same result).
fn declares_constant_source(env: &Environment, name: TypeName) -> bool {
    env.class_decls().contains_key(&name)
        || env.class_alias_decls().contains_key(&name)
        || env.constant_decls().contains_key(&name)
}

/// Mark `name`'s contribution slot on `env`'s side (see
/// [`ConstantTable::update`] Phase 1). No-op when `env` does not declare
/// `name` — the slot for a purely-removed or purely-added name is marked
/// via the side that does declare it. Owner keys use `env`'s own alias
/// normalization so old-side and new-side contributions land on the
/// exact keys the respective full walk would have used.
fn mark_slot(
    env: &Environment,
    name: TypeName,
    toplevel_slots: &mut FxHashMap<Symbol, FxHashSet<TypeName>>,
    children_slots: &mut FxHashMap<TypeName, FxHashMap<Symbol, FxHashSet<TypeName>>>,
) {
    if !declares_constant_source(env, name) {
        return;
    }
    let names = env.names();
    let Some(last) = names.last_segment(name) else {
        return;
    };
    match parent_typename(name, names) {
        None => {
            toplevel_slots.entry(last).or_default().insert(name);
        }
        Some(parent) => {
            let owner = env.normalize_module_name(&parent);
            children_slots
                .entry(owner)
                .or_default()
                .entry(last)
                .or_default()
                .insert(name);
        }
    }
}

/// The `children_table` owner key that entries nested under `name`'s
/// namespace file into on `env`'s side: the normalized alias target for
/// a resolvable alias, `name` itself otherwise (a class/module, an
/// unresolvable alias, or an undeclared namespace all key by `name` —
/// `insert_into_scope`'s `normalize_module_name` is a no-op for them).
fn children_owner_for(env: &Environment, name: TypeName) -> TypeName {
    if env.class_alias_decls().contains_key(&name)
        && let NormalizeModuleNameResult::Normalized(target) =
            env.normalize_module_name_result(&name)
        && env.class_decls().contains_key(&target)
    {
        return target;
    }
    name
}

/// The alias-preserving decl name a stored [`ConstantEntry`] came from.
fn entry_decl_name(entry: &ConstantEntry) -> TypeName {
    match *entry {
        ConstantEntry::Module { name, .. } | ConstantEntry::Constant { name } => name,
    }
}

/// The entry `name` contributes on `new_env`'s side, evaluated in
/// [`ConstantTable::new`]'s pass-precedence order — a later pass
/// overwrites an earlier one in the full walk, so precedence here is
/// constant (Pass 4) > alias (Pass 3) > class/module (Pass 2).
fn contribution_at(new_env: &Environment, name: TypeName) -> Option<ConstantEntry> {
    if new_env.constant_decls().contains_key(&name) {
        return Some(ConstantEntry::Constant { name });
    }
    if new_env.class_alias_decls().contains_key(&name)
        && let NormalizeModuleNameResult::Normalized(target) =
            new_env.normalize_module_name_result(&name)
        && new_env.class_decls().contains_key(&target)
    {
        return Some(ConstantEntry::Module { name, target });
    }
    if new_env.class_decls().contains_key(&name) {
        return Some(ConstantEntry::Module { name, target: name });
    }
    None
}

/// Recompute one `children_table[owner][sym]` slot from `new_env`.
/// Candidates are every decl name that could file here: the owner's
/// direct child, each alias-of-owner's child (distinct textual parents
/// normalizing to `owner`), and the marked changed names. Iterated in
/// sorted order so a (pathological) multi-contributor slot resolves
/// deterministically — the full walk's own winner there is
/// map-iteration-order dependent, so there is no single "correct" value
/// to mirror.
fn recompute_children_slot(
    new_env: &Environment,
    owner: TypeName,
    sym: Symbol,
    marked: &FxHashSet<TypeName>,
    rev_alias: &FxHashMap<TypeName, Vec<TypeName>>,
) -> Option<ConstantEntry> {
    let names = new_env.names();
    let mut candidates: Vec<TypeName> = Vec::with_capacity(marked.len() + 2);
    candidates.push(names.append_type_name(owner, sym));
    if let Some(aliases) = rev_alias.get(&owner) {
        for alias in aliases {
            candidates.push(names.append_type_name(*alias, sym));
        }
    }
    candidates.extend(marked.iter().copied());
    candidates.sort_unstable_by_key(|t| t.get());
    candidates.dedup();

    let files_here = |c: &TypeName| {
        names.last_segment(*c) == Some(sym)
            && parent_typename(*c, names).map(|p| new_env.normalize_module_name(&p)) == Some(owner)
    };
    slot_winner(new_env, &candidates, files_here)
}

/// Recompute one `toplevel[sym]` slot from `new_env`. Top-level entries
/// have no parent namespace, so alias-parent contributors cannot exist;
/// candidates are the marked names plus the absolute-root direct child.
fn recompute_toplevel_slot(
    new_env: &Environment,
    sym: Symbol,
    marked: &FxHashSet<TypeName>,
) -> Option<ConstantEntry> {
    let names = new_env.names();
    let mut candidates: Vec<TypeName> = Vec::with_capacity(marked.len() + 1);
    candidates.push(names.append_type_name(names.absolute_root(), sym));
    candidates.extend(marked.iter().copied());
    candidates.sort_unstable_by_key(|t| t.get());
    candidates.dedup();

    let files_here =
        |c: &TypeName| names.last_segment(*c) == Some(sym) && parent_typename(*c, names).is_none();
    slot_winner(new_env, &candidates, files_here)
}

/// Pass-precedence winner across `candidates` (see [`contribution_at`]):
/// checks all candidates for a Pass-4 constant first, then Pass-3
/// aliases, then Pass-2 class/module decls.
fn slot_winner(
    new_env: &Environment,
    candidates: &[TypeName],
    files_here: impl Fn(&TypeName) -> bool,
) -> Option<ConstantEntry> {
    let mut alias_hit: Option<ConstantEntry> = None;
    let mut class_hit: Option<ConstantEntry> = None;
    for c in candidates {
        if !files_here(c) {
            continue;
        }
        match contribution_at(new_env, *c) {
            Some(entry @ ConstantEntry::Constant { .. }) => return Some(entry),
            Some(entry @ ConstantEntry::Module { name, target }) => {
                if name == target {
                    class_hit.get_or_insert(entry);
                } else {
                    alias_hit.get_or_insert(entry);
                }
            }
            None => {}
        }
    }
    alias_hit.or(class_hit)
}

/// Insert `entry` into the right scope-level map. `name.namespace`
/// being empty means top-level (rbs `toplevel`); otherwise the parent
/// TypeName is reconstructed from the namespace's trailing segment so
/// the entry lands in `children_table[parent]`. A missing parent is
/// structurally unreachable (Pass 1 pre-seeds every class_decl slot),
/// so it surfaces as a panic.
///
/// `env.normalize_module_name(&parent)` mirrors rbs L40 so that nested
/// names under an aliased namespace (e.g. `class A = ::B; A::C` —
/// `A::C` resolves to `B::C`) land under the *normalized* parent
/// (`::B`) rather than the alias name (`::A`). Without normalization,
/// `children(::B)` would miss `C` because the entry would be filed
/// under `::A`. For non-alias parents the normalization is a no-op
/// (the precomputed table returns the input).
fn insert_into_scope(
    name: &TypeName,
    entry: ConstantEntry,
    children_table: &mut FxHashMap<TypeName, FxHashMap<Symbol, ConstantEntry>>,
    toplevel: &mut FxHashMap<Symbol, ConstantEntry>,
    env: &Environment,
) {
    let names = env.names();
    let last = names
        .last_segment(*name)
        .expect("constant name has at least one segment");
    match parent_typename(*name, names) {
        None => {
            toplevel.insert(last, entry);
        }
        Some(parent) => {
            let normalized_parent = env.normalize_module_name(&parent);
            children_table
                .entry(normalized_parent)
                .or_default()
                .insert(last, entry);
        }
    }
}

/// The parent class/module TypeName of `name`. Mirrors rbs
/// `name.namespace.to_type_name`. Returns `None` for top-level names
/// (parent is a namespace root).
fn parent_typename(name: TypeName, names: &NameTable) -> Option<TypeName> {
    let parent = names.type_name_parent(name)?;
    (!names.type_name_is_root(parent)).then_some(parent)
}

/// Cache value type: an immutable per-context (or per-module) snapshot
/// of the merged constants map. `Arc` shares the snapshot cheaply
/// across repeat lookups.
///
/// Values are the compact [`ConstantEntry`], not `ResolverConstant` —
/// this keeps the merged snapshots decode-free so caching a scope's
/// visible name set does not force `.ty` lowering for every name in
/// scope. Hydration happens at the resolver's public API boundary
/// (`resolve` / `resolve_child` / `resolve_in_namespace`) via
/// [`ConstantTable::hydrate`].
type ConstantsMap = FxHashMap<Symbol, ConstantEntry>;

/// Port of `RBS::Resolver::ConstantResolver` (constant_resolver.rb
/// L85-216). Constant-name lookup that honors lexical scope plus the
/// ancestor chain, with the two cache lanes rbs maintains:
///
/// - `context_constants_cache` keyed by [`ConstantContext`]: every
///   `resolve(name, context)` / `constants(context)` hit reuses the
///   merged map for that lexical scope.
/// - `child_constants_cache` keyed by [`TypeName`]: every
///   `resolve_child(module, name)` / `children(module)` hit reuses the
///   merged ancestor-folded children map for that class/module.
///
/// Both caches use `RefCell` for interior mutability so the public API
/// stays on `&self` — matching rbs's `@cache = {}` and letting
/// `DefinitionBuilder` keep the resolver as a plain field. Returns are
/// `Arc<ConstantsMap>` so repeat hits do not clone the entire map.
#[derive(Debug)]
pub struct ConstantResolver {
    /// Shared with [`super::DefinitionBuilder`]; held so `&self`
    /// lookups can reach `instance_ancestors` without a separate
    /// borrow.
    builder: Arc<AncestorBuilder>,
    table: ConstantTable,
    /// Precomputed `::Object` `TypeName`. The hot-path filters
    /// (`load_child_constants`, `constants_from_ancestors`,
    /// `constants_itself`) check `ancestor.name == object` and reuse
    /// this rather than reparsing `"::Object"` per call.
    object: TypeName,
    context_constants_cache: RefCell<FxHashMap<ConstantContext, Arc<ConstantsMap>>>,
    child_constants_cache: RefCell<FxHashMap<TypeName, Arc<ConstantsMap>>>,
}

impl ConstantResolver {
    /// Build the resolver from an [`AncestorBuilder`], walking the
    /// environment to build [`ConstantTable`] fresh (`O(env)` — see that
    /// type's `new` doc). The table is precomputed eagerly so the only
    /// work `resolve` / `constants` / `resolve_child` / `children` do is
    /// to merge and cache snapshots.
    pub fn new(builder: Arc<AncestorBuilder>) -> Self {
        let table = ConstantTable::new(&builder);
        Self::build(builder, table)
    }

    /// Build the resolver over a [`ConstantTable`] delta-updated from the
    /// previous generation's table ([`ConstantTable::update`]) instead of
    /// walked fresh. Called by `DefinitionBuilder::update`. The
    /// resolver's merge caches start empty (they are per-generation,
    /// same as a fresh build).
    pub(crate) fn from_updated(builder: Arc<AncestorBuilder>, table: ConstantTable) -> Self {
        Self::build(builder, table)
    }

    /// The underlying [`ConstantTable`], so `DefinitionBuilder::update`
    /// can delta-update it via [`ConstantTable::update`].
    pub(crate) fn table(&self) -> &ConstantTable {
        &self.table
    }

    fn build(builder: Arc<AncestorBuilder>, table: ConstantTable) -> Self {
        let object = builder.env().names().builtins().object;
        Self {
            builder,
            table,
            object,
            context_constants_cache: RefCell::new(FxHashMap::default()),
            child_constants_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// rbs `resolve(name, context:)` (L95-98). Looks up `name` in the
    /// merged context constants map, then hydrates the matching entry
    /// into a full [`ResolverConstant`] — this is where `.ty` lowering
    /// for a `constant_of_constant` name is deferred to.
    pub fn resolve(&self, name: Symbol, context: &ConstantContext) -> Option<ResolverConstant> {
        self.constants(context)
            .get(&name)
            .map(|entry| self.table.hydrate(entry, &self.builder))
    }

    /// rbs `constants(context)` (L100-106). Returns the entire merged
    /// snapshot for the given lexical scope. The first call for a
    /// context computes the snapshot from `load_context_constants` and
    /// stores it; subsequent calls return the cached `Arc`.
    pub fn constants(&self, context: &ConstantContext) -> Arc<ConstantsMap> {
        if let Some(cached) = self.context_constants_cache.borrow().get(context) {
            return Arc::clone(cached);
        }
        let computed = Arc::new(self.load_context_constants(context));
        self.context_constants_cache
            .borrow_mut()
            .insert(context.clone(), Arc::clone(&computed));
        computed
    }

    /// rbs `resolve_child(module_name, name)` (L108-110). Looks up
    /// `name` in `module_name`'s ancestor-folded children map, then
    /// hydrates the matching entry.
    pub fn resolve_child(&self, module_name: &TypeName, name: Symbol) -> Option<ResolverConstant> {
        self.children(module_name)
            .get(&name)
            .map(|entry| self.table.hydrate(entry, &self.builder))
    }

    /// Look up `name` strictly in the current lexical namespace — no
    /// parent walk-up. When `scope` is `Some`, defers to
    /// `resolve_child` (ancestor-folded children of that scope). When
    /// `scope` is `None` (top-level), there is no parent scope to walk
    /// up *to*, so the regular toplevel `resolve` is correct — and
    /// crucially it sees the toplevel-only entries (class/module
    /// aliases, bare `CONST: T` decls) that `resolve_child(::Object)`
    /// would miss, since rbs lands those in the `toplevel` table, not
    /// in `::Object`'s children. Used by the constant-write diagnostic,
    /// which must not let a *lexical parent's* same-named declaration
    /// shadow a fresh inner-scope write.
    pub fn resolve_in_namespace(
        &self,
        scope: Option<&TypeName>,
        name: Symbol,
    ) -> Option<ResolverConstant> {
        match scope {
            Some(target) => self.resolve_child(target, name),
            None => self.resolve(name, &ConstantContext::toplevel()),
        }
    }

    /// rbs `children(module_name)` (L112-120). Returns the
    /// ancestor-folded children map for `module_name` (normalized
    /// through `normalize_module_name`, matching rbs L113).
    pub fn children(&self, module_name: &TypeName) -> Arc<ConstantsMap> {
        let normalized = self.builder.env().normalize_module_name(module_name);
        if let Some(cached) = self.child_constants_cache.borrow().get(&normalized) {
            return Arc::clone(cached);
        }
        let computed = Arc::new(self.load_child_constants(&normalized));
        self.child_constants_cache
            .borrow_mut()
            .insert(normalized, Arc::clone(&computed));
        computed
    }

    /// rbs `load_context_constants(context)` (L122-136). Three-step
    /// merge: ancestors of the innermost scope (or Object when context
    /// is empty), the lexical scope chain itself, then the scope's own
    /// self-constant. Subsequent merges overwrite earlier ones — Ruby's
    /// inner-shadow-outer semantics expressed as
    /// `FxHashMap::insert` order.
    fn load_context_constants(&self, context: &ConstantContext) -> ConstantsMap {
        let mut consts: ConstantsMap = FxHashMap::default();

        match context.last() {
            Some(typename) => self.constants_from_ancestors(typename, &mut consts),
            None => {
                // rbs L126-130 always routes through `::Object` here.
                // When `::Object` is in the environment, take the same
                // path so the ancestor-walk side effects (e.g. children
                // of `::Kernel` if `::Object` includes it) stay
                // consistent with rbs. When it's *not* present
                // (minimum-rbs test envs that don't inject built-ins),
                // the rbs path would `raise`; crema falls back to
                // splicing `toplevel` in directly so top-level
                // resolves still work in those envs. Mirrors the
                // robustness guarantee crema gives the type checker:
                // missing built-ins degrade to soft misses, not
                // panics.
                if self.builder.env().class_decls().contains_key(&self.object) {
                    self.seed_toplevel_via_object(&mut consts);
                } else {
                    merge_into(&mut consts, &self.table.toplevel());
                }
            }
        }

        // rbs returns `nil` from `constants_from_context` for a broken
        // context (missing children_table entry mid-walk). In Rust we
        // surface the partial result — diagnostic passes catch the
        // missing reference upstream — but the conditional return path
        // is mirrored so future divergence stays visible.
        let _ = self.constants_from_context(context, &mut consts);

        self.constants_itself(context, &mut consts);
        consts
    }

    /// rbs `load_child_constants(name)` (L138-161). Walks the
    /// normalized instance-ancestors chain in reverse and merges each
    /// `Include` / `Super` / `SelfDecl` ancestor's children. `Prepend`
    /// ancestors are deliberately skipped (rbs case L151-156 does not
    /// list `AST::Members::Prepend`) — prepended modules' constants
    /// are *not* visible from the including class.
    fn load_child_constants(&self, name: &TypeName) -> ConstantsMap {
        let mut constants: ConstantsMap = FxHashMap::default();

        // rbs L142: `if table.children(name)` — keys missing from
        // children_table return an empty map. Pre-seeded for every
        // class_decl in ConstantTable::new, so a miss here means
        // `name` was never declared (e.g. a synthesized name).
        if self.table.children(name).is_none() {
            return constants;
        }

        let target_is_object = name == &self.object;
        let chain = self.builder.instance_ancestors(name);
        for ancestor in chain.ancestors.iter().rev() {
            let Ancestor::Instance {
                name: ancestor_name,
                source,
                ..
            } = ancestor
            else {
                continue;
            };
            // rbs L146-149: Object's children are merged into Object's
            // own child-constants map but skipped for *other* roots —
            // they're already routed through `constants_from_ancestors`
            // for class entries.
            if ancestor_name == &self.object && !target_is_object {
                continue;
            }
            match source {
                AncestorSource::Include | AncestorSource::Super | AncestorSource::SelfDecl => {
                    if let Some(children) = self.table.children(ancestor_name) {
                        merge_into(&mut constants, &children);
                    }
                }
                AncestorSource::Prepend | AncestorSource::Extend => {}
            }
        }

        constants
    }

    /// rbs `constants_from_context(context, constants:)` (L163-176).
    /// Recurses parent-first then merges the innermost scope's
    /// children. Returns `false` if any segment is missing from
    /// `children_table` (rbs's `or return false`); the caller's value
    /// would be `nil` in Ruby — Rust callers fold a `false` into a
    /// silent partial merge.
    fn constants_from_context(&self, context: &ConstantContext, consts: &mut ConstantsMap) -> bool {
        let Some(parent) = context.parent() else {
            return true;
        };
        if !self.constants_from_context(&parent, consts) {
            return false;
        }
        let Some(last) = context.last() else {
            return true;
        };
        let normalized = self.builder.env().normalize_module_name(last);
        match self.table.children(&normalized) {
            Some(children) => {
                merge_into(consts, &children);
                true
            }
            None => false,
        }
    }

    /// rbs `constants_from_ancestors(module_name, constants:)`
    /// (L178-199). Seeds the merge with Object's children + toplevel
    /// (when the root is a class/module entry), then walks
    /// `instance_ancestors(module_name)` in reverse. Inside the walk,
    /// hitting `::Object` from a class entry re-merges toplevel into
    /// Object's effective children (rbs L191-194 — the explicit
    /// "insert toplevel constants as ::Object's constants" comment).
    fn constants_from_ancestors(&self, name: &TypeName, consts: &mut ConstantsMap) {
        // Mirror rbs `module_class_entry(name, normalized: true)`
        // (L181). When `name` is the innermost lexical scope and is an
        // alias, `class_decls.get(name)` misses (aliases live in
        // `class_alias_decls`). Normalizing routes the lookup to the
        // alias's real target so Object/toplevel seeding still fires
        // for an alias-only context.
        let name = self.builder.env().normalize_module_name(name);
        let entry_is_class_or_module = self.builder.env().class_decls().contains_key(&name);
        let entry_is_class = matches!(
            self.builder.env().class_decls().get(&name),
            Some(crate::environment::frozen::ClassOrModule::Class(_))
        );

        if entry_is_class_or_module {
            if let Some(object_children) = self.table.children(&self.object) {
                merge_into(consts, &object_children);
            }
            merge_into(consts, &self.table.toplevel());
        } else {
            // Undefined scope: rbs raises here (L179 `or raise`).
            // crema seeds toplevel so references inside an unknown
            // module/class body resolve against top-level rather than
            // producing cascading UnknownConstant false positives.
            if let Some(object_children) = self.table.children(&self.object) {
                merge_into(consts, &object_children);
            }
            merge_into(consts, &self.table.toplevel());
            return;
        }

        let chain = self.builder.instance_ancestors(&name);
        for ancestor in chain.ancestors.iter().rev() {
            let Ancestor::Instance {
                name: ancestor_name,
                source,
                ..
            } = ancestor
            else {
                continue;
            };
            match source {
                AncestorSource::Include | AncestorSource::Super | AncestorSource::SelfDecl => {
                    let Some(children) = self.table.children(ancestor_name) else {
                        continue;
                    };
                    merge_into(consts, &children);
                    // rbs L188-194: when the chain reaches `::Object`
                    // from a class entry, toplevel is folded into
                    // Object's effective children. Inserting toplevel
                    // here (after `children`) keeps the same final
                    // map as rbs's per-ancestor `local` clone without
                    // allocating a fresh per-ancestor FxHashMap.
                    if ancestor_name == &self.object && entry_is_class {
                        merge_into(consts, &self.table.toplevel());
                    }
                }
                AncestorSource::Prepend | AncestorSource::Extend => {}
            }
        }
    }

    /// Helper for `load_context_constants`'s `None` branch: route
    /// through `::Object`'s ancestor chain to merge Object's children
    /// and toplevel into the constants map. Mirrors rbs's
    /// `constants_from_ancestors(Object, ...)`.
    fn seed_toplevel_via_object(&self, consts: &mut ConstantsMap) {
        let object = self.object;
        self.constants_from_ancestors(&object, consts);
    }

    /// rbs `constants_itself(context, constants:)` (L201-216). Adds
    /// the context's innermost scope as a constant of itself, mirroring
    /// rbs's "`A` inside `class A`" semantics.
    fn constants_itself(&self, context: &ConstantContext, consts: &mut ConstantsMap) {
        let Some(typename) = context.last() else {
            return;
        };
        let names = self.builder.env().names();
        let Some(last) = names.last_segment(*typename) else {
            return;
        };
        let entry = match parent_typename(*typename, names) {
            None => self.table.toplevel().get(&last).copied(),
            Some(parent) => self
                .table
                .children(&parent)
                .and_then(|map| map.get(&last).copied()),
        };
        if let Some(entry) = entry {
            consts.insert(last, entry);
        }
    }
}

/// Merge every `(Symbol, ConstantEntry)` pair from `src` into `dst`.
/// Mirrors rbs's `consts.merge!(table)` — the same `insert` order that
/// gives Ruby's inner-shadow-outer semantics (later writes win). Used
/// by every ancestor / context merge step so the loop pattern is
/// expressed once. `ConstantEntry` is `Copy`, so the values are moved
/// bitwise without touching `.ty` — the merge stays decode-free.
fn merge_into(dst: &mut ConstantsMap, src: &ConstantsMap) {
    for (sym, entry) in src {
        dst.insert(*sym, *entry);
    }
}

