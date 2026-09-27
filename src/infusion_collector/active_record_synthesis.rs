//! ActiveRecord per-model relation and association synthesis.
//!
//! Ported from orthoses-rails' `Orthoses::ActiveRecord::Relation#call`
//! (`@strict=false` path) and its `HasMany#call`/`BelongsTo#call`/
//! `HasOne#call` siblings at
//! `lib/orthoses/active_record/{relation,has_many,belongs_to,has_one}.rb`
//! in <https://github.com/ksss/orthoses-rails>.
//! orthoses is a well-considered reference implementation; do not diverge
//! (adding methods, changing return types, widening the collection scope)
//! without a corresponding change on the orthoses side.
//!
//! Faithful port of the `@strict=false` path:
//!   1. `<Model>::GeneratedRelationMethods` module gets `def name: (?) -> untyped`
//!      for every own singleton method (orthoses `klass.singleton_methods(false)`).
//!      `scope`-derived methods are the exception: orthoses scope.rb writes the
//!      typed `(<params>) -> <Model>::ActiveRecord_Relation` definition on both
//!      the model and this module, so they arrive via [`ActiveRecordScope`]
//!      instead of the untyped sweep.
//!   2. `<Model>::ActiveRecord_Relation` and
//!      `<Model>::ActiveRecord_Associations_CollectionProxy` classes with the
//!      three include lines.
//!   3. `<Model>::GeneratedAssociationMethods` module gets one method per
//!      `has_many`/`belongs_to`/`has_one` declaration (all three orthoses
//!      loaders funnel into the same `<Model>::GeneratedAssociationMethods`
//!      module + guarded `include` line pattern).
//!   4. Model class body gets `extend _ActiveRecord_Relation_ClassMethods[...]`
//!      plus `include <Model>::GeneratedAssociationMethods` — orthoses does
//!      NOT re-declare the model's own singleton methods on the model class
//!      body itself (the class-method surface is delivered by the extend'd
//!      interface, not by per-method redeclaration).

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, ClassSuper as Super, Declaration, Member,
    ModuleDeclaration as Module, ModuleMember,
};
use crate::ast::members::{
    ExtendMember as Extend, IncludeMember as Include, MethodDefinitionMember as MethodDefinition,
    MethodDefinitionOverload, Visibility,
};
use crate::ast::method_type::MethodType;
use crate::ast::ruby::members::{Member as RubyMember, MethodTypeAnnotation, TypeAnnotations};
use crate::ast::types::{
    BaseType, BaseTypeKind, ClassInstanceType, Function, Type as AstType, UntypedFunctionType,
};
use crate::ast::{MethodKind, TypeParam};
use crate::definition_builder::{BakedUnresolvedSuperEdge, SuperEdges};
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, Context, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::frozen::{ClassDeclaration, ClassOrModule, Environment, ModuleDeclaration};
use crate::environment::resolution::TypeNameResolver;
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::activerecord::{
    ActiveRecordAssociation, ActiveRecordAssociationKind, ActiveRecordAssociationTargetName,
    ActiveRecordEnumMapping, ActiveRecordScope, enum_mapping_mirror_method_type, scope_method_type,
};
use crate::infusion_collector::inflector::Inflector;
use crate::name::{Name, NameTable, Symbol};
use crate::snapshot::backend::GSnapshotBackend;
use crate::type_name::TypeName;

pub(crate) const ACTIVE_RECORD_BASE: &str = "::ActiveRecord::Base";
const ACTIVE_RECORD_RELATION: &str = "::ActiveRecord::Relation";
const ACTIVE_RECORD_COLLECTION_PROXY: &str = "::ActiveRecord::Associations::CollectionProxy";
const ACTIVE_RECORD_RELATION_INTERFACE: &str = "::_ActiveRecord_Relation";
const ACTIVE_RECORD_RELATION_CLASS_METHODS_INTERFACE: &str =
    "::_ActiveRecord_Relation_ClassMethods";
const ENUMERABLE: &str = "::Enumerable";

/// `batch_files`: the `.rb` files of the current collector batch, used to
/// pick each model's owner file for synthesized-decl ownership tracking
/// (ADR-0028 S2 relaxation). The owner is the *unique* batch file
/// contributing a decl to the model; models fed by several batch files
/// (or none — gem/sig-only models) get no owner and their synthesized
/// decls can only be refreshed by a cold rebuild.
/// `expansion_contributions` (S2d): `(target, concern file)` pairs whose
/// decls are concern-expansion output — excluded from the uniqueness
/// check, since the concern warm path re-creates them itself. A genuine
/// reopen in a non-emitting file still disqualifies the owner. The pair
/// granularity is file-level, so a genuine reopen *inside* an emitting
/// concern file is excluded too and the owner tag still lands — safe
/// because concern-source eligibility then fails projection equality
/// (the fresh parse re-declares the class the old side excluded as
/// expansion output), so no warm path ever consumes the tag for it.
#[allow(clippy::too_many_arguments)] // cold-pass globals, same shape as synthesize_model below
pub(crate) fn synthesize(
    draft: &mut EnvironmentDraft,
    associations: &[ActiveRecordAssociation],
    scopes: &[ActiveRecordScope],
    enum_mappings: &[ActiveRecordEnumMapping],
    inflector: &Inflector,
    batch_files: &FxHashSet<Name>,
    expansion_contributions: &FxHashSet<(TypeName, Name)>,
) {
    let models = active_record_model_names(draft);
    if models.is_empty() {
        return;
    }

    let all_names = collect_declared_class_module_names(draft);
    let model_names = models.iter().copied().collect::<FxHashSet<_>>();
    let nested_class_methods = collect_nested_class_methods(draft, &model_names);
    let context = Arc::from([draft.names().absolute_root()]);
    for model in models {
        let owner = model_owner(draft, model, batch_files, expansion_contributions);
        synthesize_model(
            draft,
            Arc::clone(&context),
            model,
            owner,
            associations,
            scopes,
            enum_mappings,
            &all_names,
            &model_names,
            &nested_class_methods,
            inflector,
        );
    }
}

/// The single batch `.rb` file whose edit invalidates `model`'s
/// synthesized decls, or `None` when no unique such file exists.
/// Non-batch contributors (the `db/schema.rb` trigger co-decl, sig/gem
/// reopens) don't disqualify the owner: their presence makes the file
/// warm-ineligible on its own (`rb_touch_model_warm_eligible` rejects
/// foreign non-trigger co-decls), so an owner tag that is never
/// exercised warm stays harmless.
fn model_owner(
    draft: &EnvironmentDraft,
    model: TypeName,
    batch_files: &FxHashSet<Name>,
    expansion_contributions: &FxHashSet<(TypeName, Name)>,
) -> Option<Name> {
    let entry = draft.class_decls.get(&model)?;
    let ClassOrModuleDraft::Class(entry) = entry else {
        return None;
    };
    let mut owner = None;
    for (origin, _, _) in &entry.context_decls {
        let Some(file) = origin.file() else { continue };
        if !batch_files.contains(&file) {
            continue;
        }
        if expansion_contributions.contains(&(model, file)) {
            continue;
        }
        match owner {
            None => owner = Some(file),
            Some(prev) if prev == file => {}
            Some(_) => return None,
        }
    }
    owner
}

pub(crate) fn active_record_model_names(draft: &EnvironmentDraft) -> Vec<TypeName> {
    descendant_class_names(draft, ACTIVE_RECORD_BASE)
}

/// Every class declared in the draft (A layer) or baked into the G
/// snapshot whose super chain reaches `base` (an absolute name such as
/// `::ActiveRecord::Base`), sorted by display name. Shared by the
/// `ActiveRecord::Base` model fixpoint and the `ActionMailer::Base`
/// mailer fixpoint, which need the same A ∪ G super-edge walk with a
/// different root.
pub(crate) fn descendant_class_names(draft: &EnvironmentDraft, base: &str) -> Vec<TypeName> {
    let names = draft.names();
    let base = names.parse_type_name(base);
    let all_names = collect_declared_class_module_names(draft);
    let aliases = FxHashMap::default();
    let resolver = TypeNameResolver::new(&all_names, &aliases, names);
    let mut candidates = Vec::new();
    let mut supers = Vec::new();

    // Warm walks the live A draft with a merged A ∪ G resolver, so a
    // fallback here means the reference is unknown in either layer —
    // the raw relative would never bridge in the fixpoint. Pass `None`
    // so the walker skips both the allocation and the push instead of
    // shovelling into a throwaway sink.
    for entry in draft.class_decls.values() {
        collect_super_edges(
            entry,
            names,
            &resolver,
            &mut Vec::new(),
            &mut SuperEdgeSink::ResolvedOnly(&mut supers),
        );
    }
    // The G super-edges were resolved once at cold time and baked into
    // the snapshot (ADR-0028 slice 2b-4); splicing them here reaches
    // every gem super without decoding a single backend entry. Live A
    // edges above and these baked G edges feed one set fixpoint, so their
    // union is order-independent.
    //
    // Cold's G-only resolver could not see the A layer, so any gem super
    // referencing an A-only name (Rails engine `class GemModel <
    // ApplicationRecord` idiom) was baked as an unresolved record
    // instead: `(class, raw_super, decl_context)`. Retry each against
    // the live A ∪ G resolver so those bridges reach the model
    // fixpoint the same way flag-off does.
    if let Some(g) = draft.g.as_ref() {
        let baked = g.baked();
        supers.extend(baked.super_edges.resolved.iter().copied());
        for edge in &baked.super_edges.unresolved {
            let resolved = resolver
                .try_resolve_typename(edge.raw_super, &edge.context)
                .unwrap_or(edge.raw_super);
            supers.push((edge.class, resolved));
        }
    }

    let mut models = FxHashSet::default();
    loop {
        let before = models.len();
        for (name, super_name) in &supers {
            if *super_name == base || models.contains(super_name) {
                models.insert(*name);
            }
        }
        if models.len() == before {
            break;
        }
    }

    candidates.extend(models);
    candidates.sort_by_key(|name| names.display_type_name(*name));
    candidates
}

pub(crate) fn collect_declared_class_module_names(draft: &EnvironmentDraft) -> FxHashSet<TypeName> {
    let mut names = FxHashSet::default();
    for (name, entry) in &draft.class_decls {
        names.insert(*name);
        collect_nested_declared_names(entry, &mut names);
    }
    if let Some(g) = draft.g.as_ref() {
        collect_g_declared_names(g, &mut names);
    }
    names
}

pub(crate) fn collect_nested_declared_names(
    entry: &ClassOrModuleDraft,
    out: &mut FxHashSet<TypeName>,
) {
    match entry {
        ClassOrModuleDraft::Class(entry) => {
            for (_file, _context, decl) in &entry.context_decls {
                match decl {
                    ClassDeclarationDraft::Signature(decl) => {
                        collect_signature_member_names(&decl.members, out)
                    }
                    ClassDeclarationDraft::Ruby(decl) => {
                        collect_ruby_member_names(&decl.members, out)
                    }
                }
            }
        }
        ClassOrModuleDraft::Module(entry) => {
            for (_file, _context, decl) in &entry.context_decls {
                match decl {
                    ModuleDeclarationDraft::Signature(decl) => {
                        collect_signature_module_member_names(&decl.members, out)
                    }
                    ModuleDeclarationDraft::Ruby(decl) => {
                        collect_ruby_member_names(&decl.members, out)
                    }
                }
            }
        }
    }
}

fn collect_signature_member_names(members: &[ClassMember], out: &mut FxHashSet<TypeName>) {
    for member in members {
        let ClassMember::Declaration(decl) = member else {
            continue;
        };
        collect_signature_decl_name(decl, out);
    }
}

fn collect_signature_module_member_names(members: &[ModuleMember], out: &mut FxHashSet<TypeName>) {
    for member in members {
        let ModuleMember::Declaration(decl) = member else {
            continue;
        };
        collect_signature_decl_name(decl, out);
    }
}

fn collect_signature_decl_name(decl: &Declaration, out: &mut FxHashSet<TypeName>) {
    match decl {
        Declaration::Class(decl) => {
            out.insert(decl.name);
            collect_signature_member_names(&decl.members, out);
        }
        Declaration::Module(decl) => {
            out.insert(decl.name);
            collect_signature_module_member_names(&decl.members, out);
        }
        Declaration::Interface(decl) => {
            out.insert(decl.name);
        }
        Declaration::ClassAlias(_)
        | Declaration::ModuleAlias(_)
        | Declaration::TypeAlias(_)
        | Declaration::Constant(_)
        | Declaration::Global(_) => {}
    }
}

fn collect_ruby_member_names(members: &[RubyMember], out: &mut FxHashSet<TypeName>) {
    for member in members {
        let RubyMember::Declaration(decl) = member else {
            continue;
        };
        match decl {
            crate::ast::ruby::declarations::Declaration::Class(decl) => {
                out.insert(decl.class_name);
                collect_ruby_member_names(&decl.members, out);
            }
            crate::ast::ruby::declarations::Declaration::Module(decl) => {
                out.insert(decl.module_name);
                collect_ruby_member_names(&decl.members, out);
            }
            crate::ast::ruby::declarations::Declaration::Constant(_)
            | crate::ast::ruby::declarations::Declaration::ClassModuleAlias(_) => {}
        }
    }
}

/// Sink for the super-edge walker family. Bundles the walker's two
/// outputs — resolved absolute edges and unresolved gem-super records —
/// into a single mutable argument, so a new output shape (e.g. a
/// skip-reason counter) needs only one signature touch instead of eight.
///
/// The two variants encode the walker's two callers:
///
/// - [`Self::ResolvedOnly`] — warm's live A ∪ G walk
///   (`active_record_model_names`). A miss there means the reference is
///   unknown in either layer and could never bridge in the model
///   fixpoint, so unresolved records would be pure noise.
/// - [`Self::Split`] — cold's G-only walk (`roast_g_super_edges`).
///   Unresolved records are kept so warm splice can rerun them against
///   the merged A ∪ G resolver (`high_infusion_ar_a_layer_super_bridging`).
pub(crate) enum SuperEdgeSink<'a> {
    ResolvedOnly(&'a mut Vec<(TypeName, TypeName)>),
    Split {
        resolved: &'a mut Vec<(TypeName, TypeName)>,
        unresolved: &'a mut Vec<BakedUnresolvedSuperEdge>,
    },
}

impl<'a> SuperEdgeSink<'a> {
    /// Resolve `class < raw_super` and route the outcome — hit into
    /// `resolved`, miss (when the variant carries an unresolved slot)
    /// into `unresolved`. Both variants share the resolution logic;
    /// dispatch is centralized here so [`SuperEdgeSink`] is the sole
    /// gate for edge accumulation.
    fn record(
        &mut self,
        class: TypeName,
        raw_super: TypeName,
        resolver: &TypeNameResolver<'_>,
        context: &[TypeName],
    ) {
        match resolver.try_resolve_typename(raw_super, context) {
            Some(resolved_super) => match self {
                SuperEdgeSink::ResolvedOnly(resolved) => resolved.push((class, resolved_super)),
                SuperEdgeSink::Split { resolved, .. } => resolved.push((class, resolved_super)),
            },
            None => {
                if let SuperEdgeSink::Split { unresolved, .. } = self {
                    unresolved.push(BakedUnresolvedSuperEdge {
                        class,
                        raw_super,
                        context: context.to_vec(),
                    });
                }
            }
        }
    }
}

fn collect_super_edges(
    entry: &ClassOrModuleDraft,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    match entry {
        ClassOrModuleDraft::Class(entry) => {
            for (_file, decl_context, decl) in &entry.context_decls {
                context.clear();
                context.extend(decl_context.iter().copied());
                match decl {
                    ClassDeclarationDraft::Signature(decl) => {
                        collect_signature_class_super_edges(decl, names, resolver, context, sink)
                    }
                    ClassDeclarationDraft::Ruby(decl) => {
                        collect_ruby_class_super_edges(decl, names, resolver, context, sink)
                    }
                }
            }
        }
        ClassOrModuleDraft::Module(entry) => {
            for (_file, decl_context, decl) in &entry.context_decls {
                context.clear();
                context.extend(decl_context.iter().copied());
                match decl {
                    ModuleDeclarationDraft::Signature(decl) => {
                        collect_signature_module_super_edges(decl, names, resolver, context, sink)
                    }
                    ModuleDeclarationDraft::Ruby(decl) => {
                        collect_ruby_module_super_edges(decl, names, resolver, context, sink)
                    }
                }
            }
        }
    }
}

fn collect_signature_class_super_edges(
    decl: &Class,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    if let Some(super_class) = &decl.super_class {
        sink.record(decl.name, super_class.name, resolver, context.as_slice());
    }
    context.push(decl.name);
    for member in &decl.members {
        if let ClassMember::Declaration(decl) = member {
            collect_signature_decl_super_edges(decl, names, resolver, context, sink);
        }
    }
    context.pop();
}

fn collect_signature_module_super_edges(
    decl: &Module,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    context.push(decl.name);
    for member in &decl.members {
        if let ModuleMember::Declaration(decl) = member {
            collect_signature_decl_super_edges(decl, names, resolver, context, sink);
        }
    }
    context.pop();
}

fn collect_signature_decl_super_edges(
    decl: &Declaration,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    match decl {
        Declaration::Class(decl) => {
            collect_signature_class_super_edges(decl, names, resolver, context, sink)
        }
        Declaration::Module(decl) => {
            collect_signature_module_super_edges(decl, names, resolver, context, sink)
        }
        Declaration::Interface(_)
        | Declaration::ClassAlias(_)
        | Declaration::ModuleAlias(_)
        | Declaration::TypeAlias(_)
        | Declaration::Constant(_)
        | Declaration::Global(_) => {}
    }
}

fn collect_ruby_class_super_edges(
    decl: &crate::ast::ruby::declarations::ClassDecl,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    if let Some(super_class) = &decl.super_class {
        sink.record(
            decl.class_name,
            super_class.type_name,
            resolver,
            context.as_slice(),
        );
    }
    context.push(decl.class_name);
    for member in &decl.members {
        collect_ruby_member_super_edges(member, names, resolver, context, sink);
    }
    context.pop();
}

fn collect_ruby_module_super_edges(
    decl: &crate::ast::ruby::declarations::ModuleDecl,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    context.push(decl.module_name);
    for member in &decl.members {
        collect_ruby_member_super_edges(member, names, resolver, context, sink);
    }
    context.pop();
}

fn collect_ruby_member_super_edges(
    member: &RubyMember,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    let RubyMember::Declaration(decl) = member else {
        return;
    };
    match decl {
        crate::ast::ruby::declarations::Declaration::Class(decl) => {
            collect_ruby_class_super_edges(decl, names, resolver, context, sink)
        }
        crate::ast::ruby::declarations::Declaration::Module(decl) => {
            collect_ruby_module_super_edges(decl, names, resolver, context, sink)
        }
        crate::ast::ruby::declarations::Declaration::Constant(_)
        | crate::ast::ruby::declarations::Declaration::ClassModuleAlias(_) => {}
    }
}

/// Collect the `(class, resolved_super)` edges one built class/module
/// entry declares: its own super plus every nested member decl's super
/// (the leaf collectors are shared with the draft-side walker, which
/// still needs the recursion). Called once per entry by
/// [`roast_g_super_edges`]. Splits by resolver outcome — resolved edges
/// go to `out`, unresolved gem-super references go to `out_unresolved`
/// with their declaration context for a warm-side retry.
fn collect_class_or_module_super_edges(
    entry: &ClassOrModule,
    names: &NameTable,
    resolver: &TypeNameResolver<'_>,
    context: &mut Vec<TypeName>,
    sink: &mut SuperEdgeSink<'_>,
) {
    match entry {
        ClassOrModule::Class(class_entry) => {
            for (_file, decl_context, decl) in class_entry.context_decls().iter() {
                context.clear();
                context.extend(decl_context.iter().copied());
                match decl {
                    ClassDeclaration::Signature(decl) => {
                        collect_signature_class_super_edges(decl, names, resolver, context, sink)
                    }
                    ClassDeclaration::Ruby(decl) => {
                        collect_ruby_class_super_edges(decl, names, resolver, context, sink)
                    }
                }
            }
        }
        ClassOrModule::Module(module_entry) => {
            for (_file, decl_context, decl) in module_entry.context_decls().iter() {
                context.clear();
                context.extend(decl_context.iter().copied());
                match decl {
                    ModuleDeclaration::Signature(decl) => {
                        collect_signature_module_super_edges(decl, names, resolver, context, sink)
                    }
                    ModuleDeclaration::Ruby(decl) => {
                        collect_ruby_module_super_edges(decl, names, resolver, context, sink)
                    }
                }
            }
        }
    }
}

/// Cold-path roast of the G layer's class/module super-edges (ADR-0028
/// slice 2b-4). Resolves each gem super reference against G-only names —
/// the edge set warm `active_record_model_names` used to recompute by
/// decoding every backend entry — and hands two vecs to
/// `encode_g_snapshot` to bake: `(resolved edges, unresolved records)`.
/// The warm read splices the resolved edges with live A super-edges and
/// retries each unresolved record against the merged A ∪ G resolver.
///
/// Cold-only. `debug_assert!` enforces the invariant so a future warm
/// caller cannot silently re-trigger the decode-all via the layered
/// `env.class_decls().iter()` path.
///
/// Resolution is G-only because the A layer does not exist yet at cold
/// time. Flag-off resolves gem supers in a single merged layer, so a
/// gem class super-referencing a name that lives only in the A layer
/// (Rails-engine `class GemModel < ApplicationRecord` idiom) misses
/// here — the raw reference plus its declaration context are captured
/// as an unresolved record instead of collapsing to a relative
/// `TypeName` fallback. `active_record_model_names` re-runs each
/// unresolved record against the live A ∪ G resolver at warm splice
/// time so the model-detection fixpoint reaches the same bridge
/// flag-off would.
pub(crate) fn roast_g_super_edges(env: &Environment) -> SuperEdges {
    debug_assert!(
        env.g_backend().is_none(),
        "roast_g_super_edges is cold-only; a layered env re-triggers class_all decode-all",
    );
    let names = env.names();
    let all_names: FxHashSet<TypeName> = env.class_decls().keys().copied().collect();
    let aliases = FxHashMap::default();
    let resolver = TypeNameResolver::new(&all_names, &aliases, names);
    let mut context = Vec::new();
    let mut edges = SuperEdges::default();
    {
        let mut sink = SuperEdgeSink::Split {
            resolved: &mut edges.resolved,
            unresolved: &mut edges.unresolved,
        };
        for (_name, entry) in env.class_decls().iter() {
            collect_class_or_module_super_edges(entry, names, &resolver, &mut context, &mut sink);
        }
    }
    // Nested classes are walked twice (as their own entry + through the
    // parent's member recursion), so the raw Vec carries duplicates. The
    // warm fixpoint absorbs them, but they bloat the snapshot payload
    // and the field's doc explicitly disclaims any ordering contract, so
    // a canonicalized sort+dedup is unblocked. The unresolved side
    // dedups by full record (class + raw_super + context vec) — nested
    // classes emit identical unresolved records via both walk paths.
    edges.resolved.sort_unstable();
    edges.resolved.dedup();
    edges.unresolved.sort_unstable();
    edges.unresolved.dedup();
    edges
}

fn collect_g_declared_names(g: &GSnapshotBackend, out: &mut FxHashSet<TypeName>) {
    // The snapshot stores a *built* environment, so every nested class /
    // module the draft side finds by walking members is already a
    // first-class entry — the open-time key table reproduces the
    // recursive walk without decoding (ADR-0028 slice 2b-3). The draft
    // walk also sweeps in member-declared interface names, but those are
    // inert here: `TypeName` is kind-tagged, so an interface name can
    // never match the class/module candidates this set is resolved
    // against.
    out.extend(g.class_key_iter().copied());
}

#[allow(clippy::too_many_arguments)] // internal synthesis entry; the args are the cold pass's globals, bundling them adds indirection without a reusable shape
fn synthesize_model(
    draft: &mut EnvironmentDraft,
    context: Context,
    model: TypeName,
    owner: Option<Name>,
    associations: &[ActiveRecordAssociation],
    scopes: &[ActiveRecordScope],
    enum_mappings: &[ActiveRecordEnumMapping],
    all_names: &FxHashSet<TypeName>,
    model_names: &FxHashSet<TypeName>,
    nested_class_methods: &NestedClassMethods,
    inflector: &Inflector,
) {
    let names = draft.names();
    let relation = nested_name(names, model, "ActiveRecord_Relation");
    let collection_proxy = nested_name(names, model, "ActiveRecord_Associations_CollectionProxy");
    let generated_relation_methods = nested_name(names, model, "GeneratedRelationMethods");
    let generated_association_methods = nested_name(names, model, "GeneratedAssociationMethods");
    let class_method_names = collect_class_method_names(draft, model, nested_class_methods);
    let association_methods = collect_association_methods(
        draft.names(),
        model,
        associations,
        all_names,
        model_names,
        inflector,
    );

    let relation_decl = Arc::new(Class {
        name: relation,
        type_params: Vec::new(),
        super_class: Some(super_class(names, ACTIVE_RECORD_RELATION)),
        members: relation_members(names, model, generated_relation_methods),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let collection_proxy_decl = Arc::new(Class {
        name: collection_proxy,
        type_params: Vec::new(),
        super_class: Some(super_class(names, ACTIVE_RECORD_COLLECTION_PROXY)),
        members: relation_members(names, model, generated_relation_methods),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    // Orthoses scope.rb mirrors each scope on GeneratedRelationMethods with
    // the same typed definition it wrote as `def self.<name>` on the model.
    // Typed model-side scope defs no longer match the untyped sweep's
    // `TypeAnnotations::None` filter, so the name filter below only guards
    // the handwritten `def self.<name>` + `scope :<name>` collision. Last
    // scope declaration wins, matching the runtime redefinition.
    let mut scope_methods: Vec<(Symbol, MethodType)> = Vec::new();
    for scope in scopes.iter().filter(|scope| scope.owner == model) {
        let name = names.intern_symbol(&scope.name);
        let method_type = scope_method_type(names, model, scope.params.as_ref());
        if let Some(existing) = scope_methods.iter_mut().find(|(n, _)| *n == name) {
            existing.1 = method_type;
        } else {
            scope_methods.push((name, method_type));
        }
    }
    // Enum mappings mirror the same way as scopes: typed on the model
    // (so the untyped sweep no longer sees them) and re-supplied typed
    // here. Last declaration wins, same as the runtime redefinition.
    let mut enum_mapping_methods: Vec<(Symbol, MethodType)> = Vec::new();
    for mapping in enum_mappings
        .iter()
        .filter(|mapping| mapping.owner == model)
    {
        let name = names.intern_symbol(&mapping.name);
        let method_type = enum_mapping_mirror_method_type(names, mapping);
        if let Some(existing) = enum_mapping_methods.iter_mut().find(|(n, _)| *n == name) {
            existing.1 = method_type;
        } else {
            enum_mapping_methods.push((name, method_type));
        }
    }
    let scope_names: FxHashSet<Symbol> = scope_methods
        .iter()
        .map(|(name, _)| *name)
        .chain(enum_mapping_methods.iter().map(|(name, _)| *name))
        .collect();
    let generated_relation_methods_decl = Arc::new(Module {
        name: generated_relation_methods,
        type_params: Vec::new(),
        self_types: Vec::new(),
        members: class_method_names
            .iter()
            .filter(|name| !scope_names.contains(name))
            .map(|name| ModuleMember::Member(method_returning_untyped(*name, MethodKind::Instance)))
            .chain(scope_methods.into_iter().map(|(name, method_type)| {
                ModuleMember::Member(method_with_type(name, MethodKind::Instance, method_type))
            }))
            .chain(enum_mapping_methods.into_iter().map(|(name, method_type)| {
                ModuleMember::Member(method_with_type(name, MethodKind::Instance, method_type))
            }))
            .collect(),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let generated_association_methods_decl = Arc::new(Module {
        name: generated_association_methods,
        type_params: Vec::new(),
        self_types: Vec::new(),
        members: association_methods
            .into_iter()
            .map(ModuleMember::Member)
            .collect(),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });

    // Orthoses `@strict=false` puts a *single* line on the model class body:
    // `extend _ActiveRecord_Relation_ClassMethods[Model, Relation, PK]`. It
    // does not re-declare the model's own singleton methods on the class
    // body — those live only in `GeneratedRelationMethods` (i.e. reachable
    // as instance methods on the model's Relation). Do not add per-method
    // singleton decls here without a corresponding change on the orthoses
    // side: they collide with the model's own `def self.foo` declarations
    // and lie about the return type.
    let mut model_members = vec![ClassMember::Member(Member::Extend(Extend {
        name: names.parse_type_name(ACTIVE_RECORD_RELATION_CLASS_METHODS_INTERFACE),
        args: vec![
            class_instance(model),
            class_instance(relation),
            untyped_type(),
        ],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    }))];
    model_members.push(include_member(generated_association_methods, Vec::new()));
    let model_decl = Arc::new(Class {
        name: model,
        type_params: Vec::new(),
        super_class: None,
        members: model_members,
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });

    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveRecord, owner),
        Arc::clone(&context),
        relation_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveRecord, owner),
        Arc::clone(&context),
        collection_proxy_decl,
    );
    let _ = draft.insert_module_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveRecord, owner),
        Arc::clone(&context),
        generated_relation_methods_decl,
    );
    let _ = draft.insert_module_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveRecord, owner),
        Arc::clone(&context),
        generated_association_methods_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveRecord, owner),
        context,
        model_decl,
    );
}

fn relation_members(
    names: &NameTable,
    model: TypeName,
    generated_relation_methods: TypeName,
) -> Vec<ClassMember> {
    vec![
        include_member(generated_relation_methods, Vec::new()),
        include_member(
            names.parse_type_name(ACTIVE_RECORD_RELATION_INTERFACE),
            vec![class_instance(model), untyped_type()],
        ),
        include_member(
            names.parse_type_name(ENUMERABLE),
            vec![class_instance(model)],
        ),
    ]
}

/// Untyped singleton method names of every model declared *nested*
/// inside another class/module body (`module Admin; class User < ...;
/// def self.foo`), keyed by the model's resolved name. The A draft holds
/// nested declarations only inside the parent's member tree (flattening
/// happens in `build`), so these are reachable only by walking every
/// entry; building the map once per `synthesize` keeps that walk O(env)
/// instead of O(models × env).
type NestedClassMethods = FxHashMap<TypeName, FxHashSet<Symbol>>;

fn collect_class_method_names(
    draft: &EnvironmentDraft,
    model: TypeName,
    nested_class_methods: &NestedClassMethods,
) -> Vec<Symbol> {
    let mut out = FxHashSet::default();
    if let Some(ClassOrModuleDraft::Class(entry)) = draft.class_decls.get(&model) {
        for (_file, _context, decl) in &entry.context_decls {
            match decl {
                ClassDeclarationDraft::Signature(decl) => {
                    collect_signature_class_methods(&decl.members, &mut out)
                }
                ClassDeclarationDraft::Ruby(decl) => {
                    collect_ruby_class_methods(draft.names(), &decl.members, &mut out)
                }
            }
        }
    }
    if let Some(nested) = nested_class_methods.get(&model) {
        out.extend(nested.iter().copied());
    }
    // A point lookup suffices for the G side: the snapshot is a *built*
    // environment, so a model reopened or declared nested anywhere is
    // already merged into its single entry, whose `context_decls` bundle
    // every declaration site's class methods (ADR-0028 slice 2b-4). The A
    // draft is not yet flattened, so its nested sites come from the
    // once-per-synthesize `NestedClassMethods` map above. Pinned by
    // `snapshot_ar_gem_nested_model_class_methods_byte_identical`.
    if let Some(g) = draft.g.as_ref()
        && let Some(ClassOrModule::Class(entry)) = g.class_entry(&model)
    {
        for (_file, _context, decl) in entry.context_decls().iter() {
            match decl {
                ClassDeclaration::Signature(decl) => {
                    collect_signature_class_methods(&decl.members, &mut out)
                }
                ClassDeclaration::Ruby(decl) => {
                    collect_ruby_class_methods(draft.names(), &decl.members, &mut out)
                }
            }
        }
    }
    let mut out: Vec<_> = out.into_iter().collect();
    out.sort_by_key(|name| draft.names().resolve(*name).to_string());
    out
}

fn collect_nested_class_methods(
    draft: &EnvironmentDraft,
    models: &FxHashSet<TypeName>,
) -> NestedClassMethods {
    let names = draft.names();
    let mut out = NestedClassMethods::default();
    for entry in draft.class_decls.values() {
        match entry {
            ClassOrModuleDraft::Class(entry) => {
                for (_file, _context, decl) in &entry.context_decls {
                    match decl {
                        ClassDeclarationDraft::Signature(decl) => {
                            collect_signature_nested_class_methods(&decl.members, models, &mut out)
                        }
                        ClassDeclarationDraft::Ruby(decl) => collect_ruby_nested_class_methods(
                            names,
                            &decl.members,
                            models,
                            &mut out,
                        ),
                    }
                }
            }
            ClassOrModuleDraft::Module(entry) => {
                for (_file, _context, decl) in &entry.context_decls {
                    match decl {
                        ModuleDeclarationDraft::Signature(decl) => {
                            collect_signature_module_nested_class_methods(
                                &decl.members,
                                models,
                                &mut out,
                            )
                        }
                        ModuleDeclarationDraft::Ruby(decl) => collect_ruby_nested_class_methods(
                            names,
                            &decl.members,
                            models,
                            &mut out,
                        ),
                    }
                }
            }
        }
    }
    out
}

fn collect_signature_module_nested_class_methods(
    members: &[ModuleMember],
    models: &FxHashSet<TypeName>,
    out: &mut NestedClassMethods,
) {
    for member in members {
        let ModuleMember::Declaration(decl) = member else {
            continue;
        };
        collect_signature_decl_nested_class_methods(decl, models, out);
    }
}

fn collect_signature_nested_class_methods(
    members: &[ClassMember],
    models: &FxHashSet<TypeName>,
    out: &mut NestedClassMethods,
) {
    for member in members {
        let ClassMember::Declaration(decl) = member else {
            continue;
        };
        collect_signature_decl_nested_class_methods(decl, models, out);
    }
}

fn collect_signature_decl_nested_class_methods(
    decl: &Declaration,
    models: &FxHashSet<TypeName>,
    out: &mut NestedClassMethods,
) {
    match decl {
        Declaration::Class(decl) => {
            if models.contains(&decl.name) {
                collect_signature_class_methods(&decl.members, out.entry(decl.name).or_default());
            }
            collect_signature_nested_class_methods(&decl.members, models, out);
        }
        Declaration::Module(decl) => {
            collect_signature_module_nested_class_methods(&decl.members, models, out)
        }
        Declaration::Interface(_)
        | Declaration::ClassAlias(_)
        | Declaration::ModuleAlias(_)
        | Declaration::TypeAlias(_)
        | Declaration::Constant(_)
        | Declaration::Global(_) => {}
    }
}

fn collect_ruby_nested_class_methods(
    names: &NameTable,
    members: &[RubyMember],
    models: &FxHashSet<TypeName>,
    out: &mut NestedClassMethods,
) {
    for member in members {
        let RubyMember::Declaration(decl) = member else {
            continue;
        };
        match decl {
            crate::ast::ruby::declarations::Declaration::Class(decl) => {
                if models.contains(&decl.class_name) {
                    collect_ruby_class_methods(
                        names,
                        &decl.members,
                        out.entry(decl.class_name).or_default(),
                    );
                }
                collect_ruby_nested_class_methods(names, &decl.members, models, out);
            }
            crate::ast::ruby::declarations::Declaration::Module(decl) => {
                collect_ruby_nested_class_methods(names, &decl.members, models, out);
            }
            crate::ast::ruby::declarations::Declaration::Constant(_)
            | crate::ast::ruby::declarations::Declaration::ClassModuleAlias(_) => {}
        }
    }
}

fn collect_signature_class_methods(members: &[ClassMember], out: &mut FxHashSet<Symbol>) {
    for member in members {
        if let ClassMember::Member(Member::MethodDefinition(method)) = member
            && method.kind == MethodKind::Singleton
        {
            out.insert(method.name);
        }
    }
}

fn collect_ruby_class_methods(
    names: &NameTable,
    members: &[RubyMember],
    out: &mut FxHashSet<Symbol>,
) {
    for member in members {
        if let RubyMember::Def(def) = member
            && def.kind == MethodKind::Singleton
            && matches!(
                def.method_type,
                MethodTypeAnnotation {
                    type_annotations: TypeAnnotations::None
                }
            )
        {
            out.insert(names.intern_symbol(&def.name));
        }
    }
}

fn collect_association_methods(
    names: &NameTable,
    model: TypeName,
    associations: &[ActiveRecordAssociation],
    all_names: &FxHashSet<TypeName>,
    model_names: &FxHashSet<TypeName>,
    inflector: &Inflector,
) -> Vec<Member> {
    associations
        .iter()
        .filter(|association| association.owner == model)
        .filter_map(|association| {
            let return_type = association_return_type(
                names,
                model,
                association,
                all_names,
                model_names,
                inflector,
            )?;
            Some(method_returning(
                names.intern_symbol(&association.name),
                MethodKind::Instance,
                return_type,
            ))
        })
        .collect()
}

fn association_return_type(
    names: &NameTable,
    owner: TypeName,
    association: &ActiveRecordAssociation,
    all_names: &FxHashSet<TypeName>,
    model_names: &FxHashSet<TypeName>,
    inflector: &Inflector,
) -> Option<TypeName> {
    let target =
        association_target_type(names, owner, association, all_names, model_names, inflector)?;
    match association.kind {
        ActiveRecordAssociationKind::BelongsTo | ActiveRecordAssociationKind::HasOne => {
            Some(target)
        }
        ActiveRecordAssociationKind::HasMany => Some(nested_name(
            names,
            target,
            "ActiveRecord_Associations_CollectionProxy",
        )),
    }
}

pub(crate) fn association_target_type(
    names: &NameTable,
    owner: TypeName,
    association: &ActiveRecordAssociation,
    all_names: &FxHashSet<TypeName>,
    model_names: &FxHashSet<TypeName>,
    inflector: &Inflector,
) -> Option<TypeName> {
    if association.polymorphic {
        return None;
    }
    if let Some(target) = &association.target_name {
        let target = match target {
            ActiveRecordAssociationTargetName::Absolute(target) => *target,
            ActiveRecordAssociationTargetName::Relative(raw) => {
                resolve_association_target(names, owner, raw, all_names)?
            }
        };
        return model_names.contains(&target).then_some(target);
    }
    let base_name = match association.kind {
        ActiveRecordAssociationKind::BelongsTo | ActiveRecordAssociationKind::HasOne => {
            association.name.clone()
        }
        ActiveRecordAssociationKind::HasMany => inflector.singularize(&association.name),
    };
    let class_name = inflector.camelize(&base_name);
    resolve_association_target(names, owner, &class_name, all_names)
        .filter(|target| model_names.contains(target))
}

fn resolve_association_target(
    names: &NameTable,
    owner: TypeName,
    class_name: &str,
    all_names: &FxHashSet<TypeName>,
) -> Option<TypeName> {
    let mut namespace = names.type_name_parent(owner);
    while let Some(current) = namespace {
        let candidate = append_relative_type_name(names, current, class_name)?;
        if all_names.contains(&candidate) {
            return Some(candidate);
        }
        namespace = names.type_name_parent(current);
    }
    let root_candidate = names.parse_type_name(&format!("::{class_name}"));
    all_names
        .contains(&root_candidate)
        .then_some(root_candidate)
}

fn append_relative_type_name(names: &NameTable, base: TypeName, raw: &str) -> Option<TypeName> {
    if raw.starts_with("::") {
        return None;
    }
    let mut current = base;
    for segment in raw.split("::") {
        if segment.is_empty() {
            return None;
        }
        current = names.append_type_name(current, names.intern_symbol(segment));
    }
    Some(current)
}

pub(crate) fn nested_name(names: &NameTable, owner: TypeName, segment: &str) -> TypeName {
    let segment = names.intern_symbol(segment);
    names.append_type_name(owner, segment)
}

fn super_class(names: &NameTable, raw: &str) -> Super {
    Super {
        name: names.parse_type_name(raw),
        args: Vec::new(),
        location: None,
        source_file: None,
    }
}

pub(crate) fn include_member(name: TypeName, args: Vec<AstType>) -> ClassMember {
    ClassMember::Member(Member::Include(Include {
        name,
        args,
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    }))
}

fn method_with_type(name: Symbol, kind: MethodKind, method_type: MethodType) -> Member {
    Member::MethodDefinition(MethodDefinition {
        name,
        kind,
        overloads: vec![MethodDefinitionOverload {
            method_type,
            annotations: Vec::new(),
        }],
        annotations: Vec::new(),
        overloading: false,
        visibility: Some(Visibility::Public),
        location: None,
        source_file: None,
        comment: None,
    })
}

pub(crate) fn method_returning(name: Symbol, kind: MethodKind, return_type: TypeName) -> Member {
    Member::MethodDefinition(MethodDefinition {
        name,
        kind,
        overloads: vec![MethodDefinitionOverload {
            method_type: MethodType {
                type_params: Vec::<TypeParam>::new(),
                function: Function::Untyped(UntypedFunctionType {
                    return_type: Box::new(class_instance(return_type)),
                }),
                block: None,
                location: None,
            },
            annotations: Vec::new(),
        }],
        annotations: Vec::new(),
        overloading: false,
        visibility: Some(Visibility::Public),
        location: None,
        source_file: None,
        comment: None,
    })
}

/// Orthoses relation.rb:26 emits `def #{name}: (?) -> untyped` — a
/// permissive fallback that survives arity mismatch and does not lie about
/// the return type. Do not change the return type to a concrete class
/// (e.g. Relation) without a corresponding change on the orthoses side —
/// the source of truth is `Orthoses::ActiveRecord::Relation#call`.
fn method_returning_untyped(name: Symbol, kind: MethodKind) -> Member {
    Member::MethodDefinition(MethodDefinition {
        name,
        kind,
        overloads: vec![MethodDefinitionOverload {
            method_type: MethodType {
                type_params: Vec::<TypeParam>::new(),
                function: Function::Untyped(UntypedFunctionType {
                    return_type: Box::new(untyped_type()),
                }),
                block: None,
                location: None,
            },
            annotations: Vec::new(),
        }],
        annotations: Vec::new(),
        overloading: false,
        visibility: Some(Visibility::Public),
        location: None,
        source_file: None,
        comment: None,
    })
}

pub(crate) fn class_instance(name: TypeName) -> AstType {
    AstType::ClassInstance(ClassInstanceType {
        name,
        args: Vec::new(),
        location: None,
    })
}

fn untyped_type() -> AstType {
    AstType::Base(BaseType {
        kind: BaseTypeKind::Any { todo: false },
        location: None,
    })
}
