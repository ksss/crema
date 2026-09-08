//! ADR-0028 Decision 4: invalidation seed computation + propagation.
//!
//! Pure set computation over two frozen [`Environment`] generations (an
//! "old" one and a "new" one derived from it via `unload` + reinsert +
//! `build`) plus their [`AncestorGraph`]s. This slice stops at computing
//! the sets — wiring the result into `DefinitionBuilder::update(except:)`
//! cache invalidation is slice S4's job, and this module does not touch
//! `DefinitionBuilder` at all.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::definition::ancestor_graph::{AncestorGraph, Node};
use crate::environment::draft::PathIndexKey;
use crate::environment::frozen::Environment;
use crate::name::{Name, Symbol};
use crate::type_name::TypeName;

/// Output of [`invalidated_names`]: names whose cached definitions may be
/// affected by `changed_paths` (ADR-0028 Decision 4).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InvalidationResult {
    /// Class/module/interface/type-alias names invalidated by seed +
    /// `AncestorGraph` descendant propagation. Nested decls need no
    /// separate handling — a nested name is just another
    /// [`PathIndexKey`] entry under the same file, so it lands in the seed
    /// set the same way a top-level name does.
    pub type_names: FxHashSet<TypeName>,
    /// Constant names declared by `changed_paths` (old or new side).
    /// Constants have no `AncestorGraph` node and do not propagate — this
    /// is a direct path filter, not type-name invalidation (ADR-0028
    /// Decision 4).
    pub constants: FxHashSet<TypeName>,
    /// Global variable names declared by `changed_paths` (old or new
    /// side). Same direct path-filter treatment as `constants`.
    pub globals: FxHashSet<Symbol>,
}

/// Compute [`InvalidationResult`] for `changed_paths`, building a fresh
/// [`AncestorGraph`] over `old_env` and `new_env`. Convenience wrapper over
/// [`invalidated_names_with_graphs`] for callers that have not already
/// built the graphs.
///
/// Each `AncestorGraph::new` call is an O(env) walk (ADR-0028 Decision 3
/// accepts this for the current in-memory slice); persisting the graph or
/// delta-updating it across generations is future-slice work, which is why
/// [`invalidated_names_with_graphs`] exists as the graph-accepting entry
/// point a later caller can use to avoid rebuilding.
pub fn invalidated_names(
    old_env: &Arc<Environment>,
    new_env: &Arc<Environment>,
    changed_paths: &FxHashSet<Name>,
) -> InvalidationResult {
    let old_graph = AncestorGraph::new(Arc::clone(old_env));
    let new_graph = AncestorGraph::new(Arc::clone(new_env));
    invalidated_names_with_graphs(old_env, new_env, &old_graph, &new_graph, changed_paths)
}

/// Same as [`invalidated_names`] but takes pre-built graphs so a caller
/// that already has one (or, in a future slice, a persisted/delta graph)
/// does not pay a second O(env) build. `old_graph` must have been built
/// over `old_env` and `new_graph` over `new_env` — this is not checked.
pub fn invalidated_names_with_graphs(
    old_env: &Environment,
    new_env: &Environment,
    old_graph: &AncestorGraph,
    new_graph: &AncestorGraph,
    changed_paths: &FxHashSet<Name>,
) -> InvalidationResult {
    let seeds = invalidation_seeds(old_env, new_env, changed_paths);
    let mut result = InvalidationResult {
        type_names: FxHashSet::default(),
        constants: seeds.constants,
        globals: seeds.globals,
    };

    // Propagate (rule 3): old graph ∪ new graph descendants of every seed,
    // both instance and singleton node — mixin removal is only reachable
    // through the old graph's edge (the new graph has already dropped it),
    // so skipping either side would under-invalidate.
    for &name in &seeds.type_names {
        result.type_names.insert(name);
        for node in [Node::InstanceNode(name), Node::SingletonNode(name)] {
            for graph in [old_graph, new_graph] {
                for descendant in graph.each_descendant(&node) {
                    result.type_names.insert(node_type_name(descendant));
                }
            }
        }
    }

    result
}

/// Seed-only half of [`InvalidationResult`] (ADR-0028 Decision 4, rules
/// 1-2) — every type-name-shaped `path_index` key under a changed file
/// plus alias old/new normalize targets, *before* `AncestorGraph`
/// descendant propagation (rule 3, [`invalidated_names_with_graphs`]-only).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InvalidationSeeds {
    pub type_names: FxHashSet<TypeName>,
    pub constants: FxHashSet<TypeName>,
    pub globals: FxHashSet<Symbol>,
}

/// Compute [`InvalidationSeeds`] for `changed_paths`. Split out from
/// [`invalidated_names_with_graphs`] (ADR-0028 S3b-2) so
/// `AncestorGraph::update`'s delta-update path can drive off the same
/// seed set a fresh `invalidated_names_with_graphs` call would use,
/// without either duplicating this computation or paying for
/// `AncestorGraph::new`'s O(env) rebuild just to get seeds.
pub fn invalidation_seeds(
    old_env: &Environment,
    new_env: &Environment,
    changed_paths: &FxHashSet<Name>,
) -> InvalidationSeeds {
    let mut result = InvalidationSeeds::default();

    // Seed (ADR-0028 Decision 4, rule 1): every type-name-shaped
    // `path_index` key under a changed file, on both the old and the new
    // side — the union catches both additions (new side only) and
    // removals (old side only).
    for &file in changed_paths {
        for env in [old_env, new_env] {
            let Some(keys) = env.path_index.get(file) else {
                continue;
            };
            for key in &keys {
                match *key {
                    PathIndexKey::ClassOrModule(name)
                    | PathIndexKey::Interface(name)
                    | PathIndexKey::TypeAlias(name)
                    | PathIndexKey::ClassAlias(name) => {
                        result.type_names.insert(name);
                    }
                    PathIndexKey::Constant(name) => {
                        result.constants.insert(name);
                    }
                    PathIndexKey::Global(sym) => {
                        result.globals.insert(sym);
                    }
                }
            }
        }
    }

    // Alias seeds (rule 2): old/new normalize targets for every seed name.
    // `Environment::normalize_module_name` is a documented no-op (returns
    // the input unchanged) for non-alias `TypeName` kinds and for a side
    // where the name is not — or is no longer — an alias entry, so calling
    // it unconditionally on every seed (not just the `ClassAlias`-kind
    // ones) is safe *for names each side actually declares*: it only ever
    // adds real alias targets or names already in `seeds`. This also
    // covers alias deletion (only the old side resolves to a real target;
    // the new side degrades to the input, already seeded).
    //
    // `checked_normalize_module_name` (concurrent fix in progress — see
    // its own doc) guards the case that comment doesn't cover: a seed
    // that is a brand-new declaration only one side's `Environment` ever
    // interned (e.g. a nested class introduced by this exact edit).
    // Calling `normalize_module_name` on that side's `env` with a name
    // from the *other* side panics ("no entry found for key" in
    // `TypeNameInterner`) rather than returning the input unchanged.
    let mut alias_targets: FxHashSet<TypeName> = FxHashSet::default();
    for &name in &result.type_names {
        alias_targets.extend(checked_normalize_module_name(old_env, name));
        alias_targets.extend(checked_normalize_module_name(new_env, name));
    }
    result.type_names.extend(alias_targets);

    result
}

/// Guard [`Environment::normalize_module_name`] against a `TypeName`
/// `env` never interned at all — `None` in that case instead of the
/// panic `normalize_module_name` (via `is_class` → `TypeNameInterner`)
/// hits for such a name. This is a **local, minimal guard**, not a fix
/// for that panic: a `name` unknown to `env`'s declaration maps cannot
/// be a resolvable class-alias on this side anyway (a real alias must
/// have been declared, hence interned, by this side's own build), so
/// skipping the call rather than fixing `normalize_module_name` itself
/// loses nothing here. The underlying bug (comparing `TypeName`s minted
/// by two independently-diverged `Environment` generations) is tracked
/// and being fixed in a separate concurrent session.
fn checked_normalize_module_name(env: &Environment, name: TypeName) -> Option<TypeName> {
    let known = env.class_decls().contains_key(&name)
        || env.interface_decls().contains_key(&name)
        || env.class_alias_decls().contains_key(&name);
    known.then(|| env.normalize_module_name(&name))
}

fn node_type_name(node: Node) -> TypeName {
    match node {
        Node::InstanceNode(name) | Node::SingletonNode(name) => name,
    }
}

