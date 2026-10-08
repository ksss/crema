//! Ancestor graph over the frozen [`Environment`] — a parent/child index
//! of every one-step ancestor edge, used to answer "what depends on type
//! `X`" for invalidation propagation (ADR-0028).
//!
//! Mirrors rbs `RBS::AncestorGraph` (`lib/rbs/ancestor_graph.rb`).

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use super::ancestor_builder::mixin_target_exists;
use super::{Ancestor, AncestorBuilder, OneAncestors};
use crate::environment::frozen::Environment;
use crate::type_name::TypeName;

/// One node of the graph: an instance-side or singleton-side type.
/// Mirrors rbs `AncestorGraph::InstanceNode` / `AncestorGraph::SingletonNode`
/// (both `Struct.new(:type_name, keyword_init: true)` in the source,
/// collapsed here into one enum since Rust has no anonymous struct
/// equality to lean on).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    InstanceNode(TypeName),
    SingletonNode(TypeName),
}

impl Node {
    /// The underlying referring/target `TypeName`, regardless of side.
    pub(crate) fn type_name(self) -> TypeName {
        match self {
            Node::InstanceNode(name) | Node::SingletonNode(name) => name,
        }
    }
}

/// Parent/child adjacency over [`Node`]s, built once from an
/// [`Environment`] and its [`AncestorBuilder`]. Mirrors rbs
/// `AncestorGraph`.
///
/// `Clone` (crema-only, ADR-0028 S3b-2, same rationale as
/// [`Environment`]'s and [`AncestorBuilder`]'s own `Clone`): a bench or
/// test that keeps a persisted graph alive across many delta-update
/// trials needs an independent copy per trial since [`Self::update`]
/// consumes `self`.
#[derive(Debug, Clone)]
pub struct AncestorGraph {
    env: Arc<Environment>,
    ancestor_builder: AncestorBuilder,
    parents: FxHashMap<Node, FxHashSet<Node>>,
    children: FxHashMap<Node, FxHashSet<Node>>,
    /// crema-only extension (ADR-0028 S3b-2): reverse index from a
    /// *normalized* mixin/super target name that some node's
    /// one-ancestors computation referenced but could not resolve
    /// (ADR-0010 silent skip — ordinarily because the target simply
    /// doesn't exist yet) to the set of referring nodes.
    ///
    /// This is the "addition direction" half of delta-update
    /// correctness: a seed-only edge replace only re-walks nodes whose
    /// *own* declaration changed, so a node that references a
    /// not-yet-declared name is never touched when that name later gets
    /// declared — its edge would silently never appear. Looking up the
    /// newly-declared name here after [`Self::build`] finds every such
    /// referrer so [`Self::update`] can re-walk it too. See
    /// `AncestorBuilder`'s own `unresolved` field doc for where entries
    /// are recorded and why the key is the *normalized* target name.
    unresolved: FxHashMap<TypeName, FxHashSet<Node>>,
}

impl AncestorGraph {
    /// Build the graph with a fresh [`AncestorBuilder`] for `env`. Mirrors
    /// rbs `AncestorGraph.new(env:)`, which defaults `ancestor_builder:` to
    /// `DefinitionBuilder::AncestorBuilder.new(env: env)`.
    pub fn new(env: Arc<Environment>) -> Self {
        let ancestor_builder = AncestorBuilder::new(Arc::clone(&env));
        Self::with_ancestor_builder(env, ancestor_builder)
    }

    /// Build the graph reusing a caller-supplied [`AncestorBuilder`].
    /// Mirrors rbs `AncestorGraph.new(env:, ancestor_builder:)`.
    pub fn with_ancestor_builder(env: Arc<Environment>, ancestor_builder: AncestorBuilder) -> Self {
        let mut graph = Self {
            env,
            ancestor_builder,
            parents: FxHashMap::default(),
            children: FxHashMap::default(),
            unresolved: FxHashMap::default(),
        };
        graph.build();
        graph
    }

    pub fn env(&self) -> &Environment {
        &self.env
    }

    pub fn ancestor_builder(&self) -> &AncestorBuilder {
        &self.ancestor_builder
    }

    /// Mirrors rbs `AncestorGraph#build`. Walks every class/module and
    /// interface name in `env` — including G-layer (gem snapshot) decls,
    /// since `Environment::class_decls` / `interface_decls` return the
    /// layered A+G view (see `frozen::Decls`) and `AncestorBuilder`'s
    /// `one_*_ancestors` lookups resolve against that same layered view.
    /// This is a full-environment walk, matching rbs's own `each_key`
    /// scan; ADR-0028 accepts the O(n) build cost for this initial slice.
    fn build(&mut self) {
        self.parents.clear();
        self.children.clear();
        self.unresolved.clear();

        let class_names: Vec<TypeName> = self.env.class_decls().keys().copied().collect();
        for type_name in class_names {
            let one_instance = self.ancestor_builder.one_instance_ancestors_arc(&type_name);
            Self::build_node_ancestors(
                &mut self.parents,
                &mut self.children,
                Node::InstanceNode(type_name),
                &one_instance,
            );
            let one_singleton = self
                .ancestor_builder
                .one_singleton_ancestors_arc(&type_name);
            Self::build_node_ancestors(
                &mut self.parents,
                &mut self.children,
                Node::SingletonNode(type_name),
                &one_singleton,
            );
            Self::insert_unresolved_for(&mut self.unresolved, &self.ancestor_builder, type_name);
        }

        let interface_names: Vec<TypeName> = self.env.interface_decls().keys().copied().collect();
        for type_name in interface_names {
            let one_interface = self
                .ancestor_builder
                .one_interface_ancestors_arc(&type_name);
            Self::build_node_ancestors(
                &mut self.parents,
                &mut self.children,
                Node::InstanceNode(type_name),
                &one_interface,
            );
            Self::insert_unresolved_for(&mut self.unresolved, &self.ancestor_builder, type_name);
        }
    }

    /// Mirrors rbs `AncestorGraph#build_ancestors`, driven by
    /// `OneAncestors#each_ancestor` (`ancestor_builder.rb:29-44`): the
    /// `super_class` edge plus every `self_types` / `included_modules` /
    /// `included_interfaces` / `prepended_modules` / `extended_modules` /
    /// `extended_interfaces` entry, each registered as a parent of `node`.
    fn build_node_ancestors(
        parents: &mut FxHashMap<Node, FxHashSet<Node>>,
        children: &mut FxHashMap<Node, FxHashSet<Node>>,
        node: Node,
        one: &OneAncestors,
    ) {
        if let Some(ancestor) = &one.super_class {
            let parent = ancestor_node(ancestor);
            register(parents, children, node, parent);
        }
        for mixin in one
            .self_types
            .iter()
            .chain(one.included_modules.iter())
            .chain(one.included_interfaces.iter())
            .chain(one.prepended_modules.iter())
            .chain(one.extended_modules.iter())
            .chain(one.extended_interfaces.iter())
        {
            register(parents, children, node, Node::InstanceNode(mixin.name));
        }
    }

    /// Mirrors rbs `AncestorGraph#each_parent`.
    pub fn each_parent(&self, node: &Node) -> impl Iterator<Item = Node> {
        edges_of(&self.parents, *node)
    }

    /// Mirrors rbs `AncestorGraph#each_child`.
    pub fn each_child(&self, node: &Node) -> impl Iterator<Item = Node> {
        edges_of(&self.children, *node)
    }

    /// Every ancestor reachable from `node`'s parent edges, each yielded
    /// once. Mirrors rbs `AncestorGraph#each_ancestor`.
    pub fn each_ancestor(&self, node: &Node) -> Vec<Node> {
        let mut yielded = FxHashSet::default();
        let mut out = Vec::new();
        self.walk_ancestors(node, &mut yielded, &mut out);
        out
    }

    fn walk_ancestors(&self, node: &Node, yielded: &mut FxHashSet<Node>, out: &mut Vec<Node>) {
        for parent in self.each_parent(node) {
            if yielded.insert(parent) {
                out.push(parent);
                self.walk_ancestors(&parent, yielded, out);
            }
        }
    }

    /// Every descendant reachable from `node`'s child edges, each yielded
    /// once. Mirrors rbs `AncestorGraph#each_descendant`.
    pub fn each_descendant(&self, node: &Node) -> Vec<Node> {
        let mut yielded = FxHashSet::default();
        let mut out = Vec::new();
        self.walk_descendants(node, &mut yielded, &mut out);
        out
    }

    fn walk_descendants(&self, node: &Node, yielded: &mut FxHashSet<Node>, out: &mut Vec<Node>) {
        for child in self.each_child(node) {
            if yielded.insert(child) {
                out.push(child);
                self.walk_descendants(&child, yielded, out);
            }
        }
    }

    /// Referring nodes for `target` whose one-ancestors computation
    /// could not resolve it (ADR-0028 S3b-2). See the `unresolved` field
    /// doc.
    pub fn each_unresolved_referrer(&self, target: &TypeName) -> impl Iterator<Item = Node> {
        edges_of(&self.unresolved, *target)
    }

    /// Delta-update the graph across an environment generation instead
    /// of [`Self::new`]'s full O(env) rebuild (ADR-0028 S3b-2,
    /// crema-only — rbs's `AncestorGraph` has no update counterpart).
    ///
    /// `seeds` is the invalidation seed set for this generation (see
    /// `environment::invalidation`'s seed computation: every `TypeName`
    /// whose own declaration, or whose containing alias's normalize
    /// target, changed between `self.env()` and `new_env`). The
    /// re-walked set is `seeds` extended by one rule: for every seed
    /// `N` whose standing as a mixin/super target
    /// ([`mixin_target_exists`]) differs between `self.env()` and
    /// `new_env` — a plain class/module/interface being added or
    /// removed, or an alias starting or ceasing to resolve to something
    /// real — pull in both `N`'s old graph children (deletion
    /// direction: nodes whose edge to `N` must disappear) and `N`'s old
    /// [`Self::each_unresolved_referrer`] (addition direction: nodes
    /// whose reference to `N` previously dropped and should now
    /// resolve into a new edge).
    ///
    /// The extension is one level deep and does not cascade: a referrer
    /// pulled in this way is re-walked because it *named* a name whose
    /// existence changed, not because its own existence changed — so
    /// re-walking it cannot in turn flip any *other* name's existence
    /// and trigger a further round.
    ///
    /// A seed that is itself a class-alias name (e.g. `module A = M1`
    /// retargeting to `M2`) needs no extra rule: every edge this graph
    /// records uses the *unnormalized* reference name a member actually
    /// wrote (`mixin_ref`/`class_super_or_default` never normalize), so
    /// `C; include A; end`'s edge is `C -> A` regardless of what `A`
    /// currently resolves to — retargeting between two names that both
    /// already exist changes no edge at all. This was verified
    /// empirically (independent old/new builds produce byte-identical
    /// `children` sets for the alias node) before dropping the
    /// originally-drafted "re-walk the old normalize target's children"
    /// rule as redundant with the existence-change rule above, which
    /// already covers alias deletion and dangling-alias resolution.
    pub fn update(self, new_env: Arc<Environment>, seeds: &FxHashSet<TypeName>) -> AncestorGraph {
        let extended = self.extend_seeds(&new_env, seeds);

        let AncestorGraph {
            ancestor_builder,
            mut parents,
            mut children,
            mut unresolved,
            ..
        } = self;

        for &name in &extended {
            for node in [Node::InstanceNode(name), Node::SingletonNode(name)] {
                Self::remove_node_out_edges(&mut parents, &mut children, node);
                Self::remove_unresolved_for(&mut unresolved, &ancestor_builder, node);
            }
        }

        let new_ancestor_builder = ancestor_builder.update(Arc::clone(&new_env), &extended);

        // Rebuild only for names `new_env` actually walks as a
        // class_names/interface_names source (mirrors `Self::build`'s
        // two loops exactly) — an `extended` member that is an alias
        // (or was deleted, or was never a real declaration to begin
        // with) must produce *no* edges under its own name.
        // `one_instance_ancestors_arc`/`one_singleton_ancestors_arc`
        // normalize their argument internally, so calling them
        // unconditionally on an alias name like `A` would silently
        // register `A`'s edges using its *normalize target*'s
        // super/mixins instead of leaving `A` unwalked — a spurious
        // node a fresh build never produces (caught by this slice's own
        // differential tests: `alias_retarget_includer_unchanged` /
        // `dangling_alias_resolves`).
        for &name in &extended {
            if new_env.class_decls().contains_key(&name) {
                let one_instance = new_ancestor_builder.one_instance_ancestors_arc(&name);
                Self::build_node_ancestors(
                    &mut parents,
                    &mut children,
                    Node::InstanceNode(name),
                    &one_instance,
                );
                let one_singleton = new_ancestor_builder.one_singleton_ancestors_arc(&name);
                Self::build_node_ancestors(
                    &mut parents,
                    &mut children,
                    Node::SingletonNode(name),
                    &one_singleton,
                );
            }
            if new_env.interface_decls().contains_key(&name) {
                let one_interface = new_ancestor_builder.one_interface_ancestors_arc(&name);
                Self::build_node_ancestors(
                    &mut parents,
                    &mut children,
                    Node::InstanceNode(name),
                    &one_interface,
                );
            }
            Self::insert_unresolved_for(&mut unresolved, &new_ancestor_builder, name);
        }

        AncestorGraph {
            env: new_env,
            ancestor_builder: new_ancestor_builder,
            parents,
            children,
            unresolved,
        }
    }

    /// `seeds` extended per [`Self::update`]'s existence-change rule.
    /// Reads only `self`'s already-built `children` / `unresolved` maps
    /// and safe declaration-map probes on `old_env`/`new_env`
    /// ([`mixin_target_exists`]) — bounded by `seeds` and their direct
    /// graph neighbors, never an O(env) walk.
    fn extend_seeds(
        &self,
        new_env: &Environment,
        seeds: &FxHashSet<TypeName>,
    ) -> FxHashSet<TypeName> {
        let mut extended = seeds.clone();
        for &name in seeds {
            let existed_old = mixin_target_exists(&self.env, name);
            let existed_new = mixin_target_exists(new_env, name);
            if existed_old == existed_new {
                continue;
            }
            for node in [Node::InstanceNode(name), Node::SingletonNode(name)] {
                extended.extend(self.each_child(&node).map(Node::type_name));
            }
            extended.extend(self.each_unresolved_referrer(&name).map(Node::type_name));
        }
        extended
    }

    /// Remove every edge where `node` is the *child* (i.e. the edges
    /// `node`'s own one-ancestors computation produced) from both
    /// `parents` and the mirrored `children` side. Edges where `node`
    /// is a *parent* (other nodes' references to it) belong to those
    /// other nodes' own computations and are untouched here.
    ///
    /// An emptied entry is dropped rather than left behind, matching a
    /// fresh [`Self::build`] (which never inserts one) so a delta-updated
    /// graph stays structurally identical to a fresh rebuild (pinned by
    /// this module's own differential tests).
    fn remove_node_out_edges(
        parents: &mut FxHashMap<Node, FxHashSet<Node>>,
        children: &mut FxHashMap<Node, FxHashSet<Node>>,
        node: Node,
    ) {
        let old_parents = parents.remove(&node).unwrap_or_default();
        for parent in old_parents {
            let set = children.entry(parent).or_default();
            set.remove(&node);
            if set.is_empty() {
                children.remove(&parent);
            }
        }
    }

    /// Insert `ancestor_builder`'s currently-cached misses for `node`'s
    /// instance and singleton sides into `unresolved`. Shared by
    /// [`Self::build`] (called once per class/interface name against
    /// the fully-populated builder) and [`Self::update`] (called once
    /// per extended name against the freshly re-walked builder).
    fn insert_unresolved_for(
        unresolved: &mut FxHashMap<TypeName, FxHashSet<Node>>,
        ancestor_builder: &AncestorBuilder,
        type_name: TypeName,
    ) {
        for node in [
            Node::InstanceNode(type_name),
            Node::SingletonNode(type_name),
        ] {
            for target in ancestor_builder.unresolved_misses(node) {
                unresolved.entry(target).or_default().insert(node);
            }
        }
    }

    /// Remove `node` from every `unresolved` entry it referrers under,
    /// using `ancestor_builder`'s (pre-update) cached misses for `node`
    /// to know exactly which target keys to touch — bounded by `node`'s
    /// own miss count, not a scan over all of `unresolved`. An emptied
    /// entry is dropped — see [`Self::remove_node_out_edges`]'s doc.
    fn remove_unresolved_for(
        unresolved: &mut FxHashMap<TypeName, FxHashSet<Node>>,
        ancestor_builder: &AncestorBuilder,
        node: Node,
    ) {
        for target in ancestor_builder.unresolved_misses(node) {
            let set = unresolved.entry(target).or_default();
            set.remove(&node);
            if set.is_empty() {
                unresolved.remove(&target);
            }
        }
    }
}

/// Shared read shape for the three adjacency maps.
fn edges_of<K: std::hash::Hash + Eq + Copy>(
    map: &FxHashMap<K, FxHashSet<Node>>,
    key: K,
) -> std::vec::IntoIter<Node> {
    map.get(&key)
        .map(|set| set.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
}

fn ancestor_node(ancestor: &Ancestor) -> Node {
    match ancestor {
        Ancestor::Instance { name, .. } => Node::InstanceNode(*name),
        Ancestor::Singleton { name } => Node::SingletonNode(*name),
    }
}

/// Mirrors rbs `AncestorGraph#register(parent:, child:)`.
fn register(
    parents: &mut FxHashMap<Node, FxHashSet<Node>>,
    children: &mut FxHashMap<Node, FxHashSet<Node>>,
    child: Node,
    parent: Node,
) {
    parents.entry(child).or_default().insert(parent);
    children.entry(parent).or_default().insert(child);
}

