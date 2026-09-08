//! Build-layer validator (ADR-0013).
//!
//! Runs after all `.rbs` and inline sources have been loaded into the
//! `DefinitionBuilder` and after `finalize_implicit_superclasses`, but *before*
//! `finalize_type_param_defaults` so arity checks see the as-written args
//! rather than default-filled ones.
//!
//! Each sub-check returns its own `Vec<Diagnostic>`; the top-level
//! `validate` concatenates them in layer order.
//!
//! The scope of this layer is diagnostics that are only decidable once the
//! merged environment exists (arity today; future checks such as inline
//! self-ref defaults or unknown-name resolution will land here). Deep
//! RBS-internal analysis that `rbs validate` already provides is still
//! out of scope per ADR-0010.

use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::ast::types::Type as AstType;
use crate::definition::ancestor_builder::{Ancestor, AncestorBuilder};
use crate::definition::{MixinRef, VariableDuplicationKind};
use crate::definition_builder::{
    BakedAncestorCycle, BakedArityViolation, DefinitionBuilder, PerNameCache, TypeParamsCache,
};
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::frozen::{
    ClassDeclaration, ClassOrModule, Environment, ModuleDeclaration, NormalizeModuleNameResult,
};
use crate::location::{RubyLocation, SourceLocation};
use crate::name::NameTable;
use crate::snapshot::backend::GSnapshotBackend;
use crate::type_name::TypeName;
use crate::type_param::TypeParam;

/// In-memory source lookup used when translating `RubyLocation` byte offsets to
/// line numbers. Keys are the canonical `PathBuf` for each source file.
/// Callers pass the `.rb` sources they already hold and anything Ruby-side
/// generated (e.g. `-e` inline code with no on-disk counterpart). Falls back
/// to reading from disk for files absent from the map (typical `.rbs` files
/// that `env.load_dir` reads then discards).
pub type SourceCache<'a> = FxHashMap<PathBuf, &'a [u8]>;

/// Persisted differential state for [`validate_update`] (ADR-0028 S5v):
/// one entry per sub-check that needs cross-generation state to avoid an
/// O(env) re-walk. The other four sub-checks need no entry here:
/// `method_dups` / `alias_cycles` / `variable_dups` are already
/// O(diagnostic count) via [`DefinitionBuilder`]'s carried `PerNameCache`s
/// (read straight off the already-updated builder — see
/// `validate_method_dups` et al.), and `alias_targets` stays a full O(env)
/// walk every call because it measured negligible (see
/// `validate_alias_targets`'s doc comment for the measurement).
pub struct ValidationState {
    arity: ArityState,
    ancestors: AncestorCycleState,
    type_aliases: TypeAliasCycleState,
}

/// Run every build-layer check against `env` and return their combined
/// diagnostics in the order the sub-checks below are invoked (applied
/// type args → `validate_method_dups` → `validate_alias_cycles` →
/// recursive type aliases → recursive ancestors → `validate_variable_dups`
/// → `validate_alias_targets` — see [`full_validate`] for the exact call
/// sequence). All sub-checks run to completion — there is no shared
/// budget that would short-circuit later layers.
///
/// `sources` supplies in-memory bytes for any file that is not backed by
/// disk (e.g. `-e` mode) and — as a perf convenience — for files main
/// already has in memory. Pass an empty map if all files are on disk.
pub fn validate(env: &DefinitionBuilder, sources: &SourceCache<'_>) -> Vec<Diagnostic> {
    full_validate(env, sources).1
}

/// Same diagnostics as [`validate`] (identical content and order), plus
/// the [`ValidationState`] the recompute produced as a byproduct — the
/// first-generation state producer [`validate_update`] seeds from
/// (mirrors [`full_ancestor_cycles`] / [`full_type_alias_cycles`]'s
/// existing "the full path also returns state" role, ADR-0028 Decision 4
/// S5, extended to the whole validator in S5v).
pub fn full_validate(
    env: &DefinitionBuilder,
    sources: &SourceCache<'_>,
) -> (ValidationState, Vec<Diagnostic>) {
    let ctx = AncestryEnv::from_builder(env);
    let environment = env.env();

    let (arity, mut diagnostics) = match environment.g_backend() {
        Some(g) => full_applied_type_args_a(&ctx, g),
        None => full_applied_type_args(&ctx),
    };
    diagnostics.extend(validate_method_dups(env, sources));
    diagnostics.extend(validate_alias_cycles(env, sources));
    let (type_aliases, type_alias_diags) = full_type_alias_cycles(env);
    diagnostics.extend(type_alias_diags);
    let (ancestors, ancestor_diags) = match environment.g_backend() {
        Some(g) => full_ancestor_cycles_a(&ctx, g),
        None => full_ancestor_cycles(&ctx),
    };
    diagnostics.extend(ancestor_diags);
    diagnostics.extend(validate_variable_dups(env, sources));
    diagnostics.extend(validate_alias_targets(env, sources));

    (
        ValidationState {
            arity,
            ancestors,
            type_aliases,
        },
        diagnostics,
    )
}

/// Incremental counterpart to [`full_validate`] (ADR-0028 S5v): given the
/// previous generation's [`ValidationState`] and the invalidated-name set
/// between generations
/// ([`InvalidationResult::type_names`](crate::environment::invalidation::InvalidationResult::type_names)),
/// recompute [`validate`]'s diagnostics for the new generation without an
/// O(env) re-walk of the two recursive-cycle validators or the
/// applied-type-args arity check.
///
/// `env` must already reflect the new generation
/// (`DefinitionBuilder::update`) — `validate_method_dups` /
/// `validate_alias_cycles` / `validate_variable_dups` read `env`'s own
/// carried `PerNameCache`s directly, so they need no separate state
/// threading here. Diagnostic *content* matches a from-scratch
/// [`full_validate`] on the same generation; diagnostic *order* does not
/// (the differential path re-derives per-owner/per-cycle order from
/// carry-over + recompute, not a single fresh env walk) — callers that
/// care about order (e.g. `_internal incr-bench`'s differential gate)
/// already re-sort with `sort_by_canonical_order` before comparing.
pub fn validate_update(
    old_state: &ValidationState,
    invalidated: &FxHashSet<TypeName>,
    env: &DefinitionBuilder,
    sources: &SourceCache<'_>,
) -> (ValidationState, Vec<Diagnostic>) {
    let ctx = AncestryEnv::from_builder(env);
    let environment = env.env();

    let (arity, mut diagnostics) = match environment.g_backend() {
        Some(g) => incremental_applied_type_args_a(&old_state.arity, invalidated, &ctx, g),
        None => incremental_applied_type_args(&old_state.arity, invalidated, &ctx),
    };
    diagnostics.extend(validate_method_dups(env, sources));
    diagnostics.extend(validate_alias_cycles(env, sources));
    let (type_aliases, type_alias_diags) =
        incremental_type_alias_cycles(&old_state.type_aliases, invalidated, env);
    diagnostics.extend(type_alias_diags);
    let (ancestors, ancestor_diags) = match environment.g_backend() {
        Some(g) => incremental_ancestor_cycles_a(&old_state.ancestors, invalidated, &ctx, g),
        None => incremental_ancestor_cycles(&old_state.ancestors, invalidated, &ctx),
    };
    diagnostics.extend(ancestor_diags);
    diagnostics.extend(validate_variable_dups(env, sources));
    diagnostics.extend(validate_alias_targets(env, sources));

    (
        ValidationState {
            arity,
            ancestors,
            type_aliases,
        },
        diagnostics,
    )
}

/// Emit `UnknownTypeName` for every class- or module-alias whose rhs
/// (`old_name`) does not resolve to a declared class or module.
/// Mirrors RBS's `NoTypeFoundError` raised in
/// `Validator#validate_class_alias` (`lib/rbs/validator.rb:161`), and
/// matches Steep's `Diagnostic::Signature::UnknownTypeName` surface.
///
/// Walks `Environment::normalized_module_names`, which the build phase
/// already populated with one `NormalizeModuleNameResult` per alias key.
/// Only the `UnknownTarget` arm is reported here — `Cycle` and the
/// (not yet generated) kind-mismatch case are sibling todos and reuse
/// distinct diagnostic shapes.
fn validate_alias_targets(env: &DefinitionBuilder, _sources: &SourceCache<'_>) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let environment = env.env();
    for (alias_key, result) in &environment.normalized_module_names {
        let NormalizeModuleNameResult::UnknownTarget { target, .. } = result else {
            continue;
        };
        let Some(entry) = environment.class_alias_decls().get(alias_key) else {
            continue;
        };
        let location = resolve_or_default(entry.old_name_source_location(env.names()).as_ref());
        out.push(Diagnostic {
            scope: None,
            kind: DiagnosticKind::UnknownTypeName {
                name: env.names().resolve(target),
            },
            location,
        });
    }
    out
}

/// Emit `RecursiveAncestor` diagnostics for cyclic ancestor graphs
/// (super / include / prepend / self_types). Mirrors rbs's
/// `RecursiveAncestorError.check!` (`lib/rbs/errors.rb` L134) which
/// raises during `instance_ancestors` build for any class whose
/// `building_ancestors` stack revisits a name.
///
/// crema's build layer (`build_instance_ancestors`) absorbs cycles
/// silently via a `visited` set to avoid stack overflow, so the
/// diagnostic surface lives here in the validator instead. The walk is
/// independent from build (it never touches `instance_ancestors_cache`):
/// class / module nodes consult `one_instance_ancestors` and follow
/// the same four edge kinds rbs's cycle check considers, with
/// self_types as a cycle-detection-only neighbour (per rbs
/// `instance_ancestors` L527, self_types are checked for cycles but
/// never folded into the chain); interface nodes consult
/// `one_interface_ancestors` and follow only `included_interfaces`,
/// mirroring rbs's `interface_ancestors` shape.
///
/// Roots cover both `class_decls` and `interface_decls`. Pure
/// interface-to-interface cycles are unreachable from any class root,
/// so seeding interface entries is required for `interface _A;
/// include _B; interface _B; include _A` to surface at all.
///
/// Why crema diverges from rbs's "one diagnostic per affected class"
/// surface: rbs emits a separate `RecursiveAncestorError` for every
/// class whose ancestor build touches a cycle (e.g. `class Object;
/// include Foo; end` causes one error for every standard-library
/// class). For an AI consumer that is noise — the cycle, not the
/// downstream class count, is the actionable signal — so each cycle
/// is collapsed to a single diagnostic, keyed by the lexicographically
/// smallest participant.
///
/// The walk also guards inline annotations: `rbs validate` doesn't
/// cover `.rb` inline declarations, so without this pass cyclic
/// inline ancestors would have no detection path at all.
///
/// Full recompute for the flag-off (no G backend) case, factored out so
/// it can also serve as [`incremental_ancestor_cycles`]'s first-generation
/// state producer (ADR-0028 Decision 4, S5: "the full path also returns
/// state — the first run just produces it as a byproduct of computing
/// diagnostics normally"). [`full_ancestor_cycles_a`] is the G-backend-
/// attached counterpart, wired in by [`full_validate`] / [`validate_update`]
/// via `Environment::g_backend`.
///
/// The diagnostic stream is deterministic regardless of `FxHashMap`
/// iteration order because [`strongly_connected_components`] sorts its
/// own output.
pub(crate) fn full_ancestor_cycles(ctx: &AncestryEnv) -> (AncestorCycleState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let roots: Vec<TypeName> = environment
        .class_decls()
        .keys()
        .chain(environment.interface_decls().keys())
        .cloned()
        .collect();
    let state: AncestorCycleState = ancestor_scc_cycles(ctx, roots, |_| true);
    let diagnostics = state.iter().map(|c| ancestor_cycle_diagnostic(c)).collect();
    (state, diagnostics)
}

/// Persisted state for [`incremental_ancestor_cycles`] /
/// [`incremental_ancestor_cycles_a`]: every currently known ancestor cycle
/// in collapsed-diagnostic form. `Arc`-wrapped so an unaffected cycle can
/// be carried into the next generation by cloning the handle (cheap, and
/// lets a test assert non-recomputation via `Arc::ptr_eq`) instead of the
/// `BakedAncestorCycle` value.
pub(crate) type AncestorCycleState = Vec<Arc<BakedAncestorCycle>>;

/// BFS-from-`seeds` + Tarjan SCC body shared by [`full_ancestor_cycles`] /
/// [`incremental_ancestor_cycles`] (flag-off) and
/// [`full_ancestor_cycles_a`] / [`incremental_ancestor_cycles_a`]
/// (G-backend-attached). A found component becomes a state entry when it
/// is cyclic (`len >= 2`) and passes `keep` — callers use `keep` to add
/// the "must touch `invalidated`" (incremental) or "must have an A
/// participant" (G-backend) conditions on top of plain cyclicity.
fn ancestor_scc_cycles(
    ctx: &AncestryEnv,
    seeds: Vec<TypeName>,
    keep: impl Fn(&[TypeName]) -> bool,
) -> Vec<Arc<BakedAncestorCycle>> {
    let names = ctx.names();
    let graph = ancestor_graph(ctx, seeds);
    strongly_connected_components(&graph, names)
        .iter()
        .filter(|component| component.len() >= 2 && keep(component))
        .map(|component| Arc::new(component_to_baked(ctx, &graph, component)))
        .collect()
}

/// Incrementally update an [`AncestorCycleState`] for a new environment
/// generation given the invalidated-name set that changed between
/// generations (ADR-0028 Decision 4, S5). `ctx` must view the *new*
/// generation; `old_state` is the previous generation's result (from this
/// function or from [`full_ancestor_cycles`] on the first run).
///
/// # Why this is sound
///
/// (Supersedes the original S5 claim that an edge `X→Y` in the
/// cycle-detection graph is derived *only* from `X`'s own declaration —
/// S3b found the counter-example: [`ancestor_successors`] also drops
/// edges to a target that isn't declared yet, so `Y`'s own *existence*
/// changing (not just `X`'s declaration) can change whether the `X→Y`
/// edge exists at all.)
///
/// The surviving argument is endpoint-based, not `X`-only: a changed edge
/// `X→Y` comes from either "`X`'s own decl changed" or "`Y`'s existence
/// changed", and either way **both endpoints land in `invalidated`** — `X`
/// directly if its own file changed, `Y` directly as a `path_index` seed
/// for its (dis)appearance, and the *other* endpoint via `AncestorGraph`
/// descendant propagation (`InvalidationResult` rule 3: everything
/// reachable from a seed in the old or new ancestor graph). A cycle is a
/// set of mutually reachable nodes connected only by edges between its
/// own members; if a cycle is newly formed or newly broken between
/// generations, at least one of its edges must have changed, and *both*
/// that edge's endpoints are themselves cycle members. So **every cycle
/// that appears or disappears contains at least one invalidated name** —
/// symmetrically, a cycle with no invalidated participant has every edge
/// unchanged and is still exactly the same cycle in the new generation.
///
/// This argument assumes edges are computed against an up-to-date view of
/// both endpoints' declarations. It does *not* cover ADR-0028's separately
/// accepted staleness window (ADR-0028 accepts shadowing that runs in
/// the reverse direction): `Environment::unload` + `build` only
/// re-resolves *freshly (re)inserted* declarations (`EnvironmentDraft::build`'s
/// Pass 1, `src/environment/draft.rs`) — a referrer that is never itself
/// touched keeps whatever resolution (including "unresolved") its mixin
/// refs got the generation it was last (re)inserted, even after a target
/// it references is later declared. The ADR names the deletion-direction
/// case explicitly; the addition-direction case (a dangling `include`
/// becoming resolvable) is the same staleness class but currently has no
/// differential-test coverage here — a candidate for its own follow-up,
/// not an S5v regression (verified to reproduce identically on
/// `unload`+`build` alone, with no [`validator`](crate::validator)
/// involvement at all).
///
/// This licenses the update:
/// 1. Drop every `old_state` cycle that shares a participant with
///    `invalidated` (it may have changed shape or vanished; recomputed in
///    step 2 if it still exists).
/// 2. Re-run Tarjan SCC, but only over the successor-reachable subgraph
///    seeded at `invalidated` — sound because a cycle a changed node is
///    part of is, by the cycle definition, reachable from that node by
///    following successor edges all the way around back to it, so the
///    forward search from the seed necessarily covers the whole cycle.
/// 3. Keep only the components found in step 2 that actually contain an
///    invalidated participant. Components found there that don't (e.g. an
///    untouched cycle purely downstream of an invalidated node, rediscovered
///    as a byproduct of the reachability search) are already present in
///    the untouched fraction kept in step 1 — this filter is what keeps
///    the merge duplicate-free without a separate dedup pass.
///
/// `invalidated` is expected to be
/// [`InvalidationResult::type_names`](crate::environment::invalidation::InvalidationResult::type_names)
/// (S3) as-is, including its descendant propagation. A descendant whose
/// own edges are unaffected by the edit just gets harmlessly rediscovered
/// by step 2 and re-admitted by step 3 (identical to what it was), at
/// bounded — not O(env) — cost.
pub(crate) fn incremental_ancestor_cycles(
    old_state: &AncestorCycleState,
    invalidated: &FxHashSet<TypeName>,
    ctx: &AncestryEnv,
) -> (AncestorCycleState, Vec<Diagnostic>) {
    let mut merged: AncestorCycleState = old_state
        .iter()
        .filter(|cycle| !cycle.participants.iter().any(|p| invalidated.contains(p)))
        .cloned()
        .collect();

    let seeds: Vec<TypeName> = invalidated.iter().copied().collect();
    merged.extend(ancestor_scc_cycles(ctx, seeds, |component| {
        component.iter().any(|p| invalidated.contains(p))
    }));
    merged.sort_by(|a, b| a.type_name.cmp(&b.type_name));

    let diagnostics = merged
        .iter()
        .map(|c| ancestor_cycle_diagnostic(c))
        .collect();
    (merged, diagnostics)
}

/// G-backend-attached counterpart of [`full_ancestor_cycles`] (ADR-0028
/// S5v): seeds are the raw A-map names only (`environment.class_decls` /
/// `interface_decls` fields, not the A+G-merged accessor methods); the
/// graph still expands into the G chain via per-name lazy probes
/// (O(touched)), so cross-layer cycles are found. G-only cycles come
/// from `g`'s baked roast instead. The exclusion is component-wise: the
/// walk keeps only components with an A participant, the splice
/// ([`ancestor_cycle_diagnostics_with_g_splice`]) keeps only baked cycles
/// without one — a cycle whose member the A layer reopens may gain
/// edges, so the re-walked shape wins over the stale baked one.
///
/// The persisted [`AncestorCycleState`] holds only the A-participating
/// (walked) cycles — the G-only baked splice is cheap to recompute every
/// call (O(#baked cycles), the G environment never changes within a
/// differential session) so it is not part of the carried state.
pub(crate) fn full_ancestor_cycles_a(
    ctx: &AncestryEnv,
    g: &GSnapshotBackend,
) -> (AncestorCycleState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let names = ctx.names();
    let mut seeds: Vec<TypeName> = environment
        .class_decls()
        .a_keys()
        .chain(environment.interface_decls().a_keys())
        .cloned()
        .collect();
    seeds.sort_by_key(|n| names.resolve(n));
    let in_a = |n: &TypeName| {
        environment.a_class_contains(n) || environment.interface_decls().a_contains_key(n)
    };
    let state: AncestorCycleState =
        ancestor_scc_cycles(ctx, seeds, |component| component.iter().any(in_a));
    let diagnostics = ancestor_cycle_diagnostics_with_g_splice(&state, g, in_a);
    (state, diagnostics)
}

/// G-backend-attached counterpart of [`incremental_ancestor_cycles`]
/// (ADR-0028 S5v). Same three-step update as the flag-off version, with
/// an extra "must have an A participant" component filter (mirrors
/// [`full_ancestor_cycles_a`]'s walk) so a purely-G-internal cycle
/// rediscovered as a byproduct of the bounded search is not persisted
/// into state — it is either already excluded from `g`'s baked splice (a
/// no-op, since G never changes mid-session) or would double-report
/// alongside it.
pub(crate) fn incremental_ancestor_cycles_a(
    old_state: &AncestorCycleState,
    invalidated: &FxHashSet<TypeName>,
    ctx: &AncestryEnv,
    g: &GSnapshotBackend,
) -> (AncestorCycleState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let in_a = |n: &TypeName| {
        environment.a_class_contains(n) || environment.interface_decls().a_contains_key(n)
    };

    let mut merged: AncestorCycleState = old_state
        .iter()
        .filter(|cycle| !cycle.participants.iter().any(|p| invalidated.contains(p)))
        .cloned()
        .collect();

    let seeds: Vec<TypeName> = invalidated.iter().copied().collect();
    merged.extend(ancestor_scc_cycles(ctx, seeds, |component| {
        component.iter().any(|p| invalidated.contains(p)) && component.iter().any(in_a)
    }));
    merged.sort_by(|a, b| a.type_name.cmp(&b.type_name));

    let diagnostics = ancestor_cycle_diagnostics_with_g_splice(&merged, g, in_a);
    (merged, diagnostics)
}

/// Merge `state`'s ancestor-cycle diagnostics with `g`'s G-only baked
/// cycles (`BakedAncestorCycle`s whose participants have no A member —
/// already excluded from `state` by the `full_ancestor_cycles_a` /
/// `incremental_ancestor_cycles_a` "must have an A participant" component
/// filter). Both streams are sorted by anchor `type_name` (SCC output
/// order for `state`, roast-time SCC order for `g.baked()`), so a
/// two-pointer merge reproduces the flag-off, globally name-sorted
/// diagnostic order.
fn ancestor_cycle_diagnostics_with_g_splice(
    state: &AncestorCycleState,
    g: &GSnapshotBackend,
    in_a: impl Fn(&TypeName) -> bool,
) -> Vec<Diagnostic> {
    let baked: Vec<&BakedAncestorCycle> = g
        .baked()
        .ancestor_cycles
        .iter()
        .filter(|c| !c.participants.iter().any(&in_a))
        .collect();
    let mut out = Vec::with_capacity(state.len() + baked.len());
    let (mut w, mut b) = (state.iter().peekable(), baked.iter().peekable());
    loop {
        match (w.peek(), b.peek()) {
            (Some(wc), Some(bc)) => {
                if wc.type_name <= bc.type_name {
                    out.push(ancestor_cycle_diagnostic(w.next().unwrap()));
                } else {
                    out.push(ancestor_cycle_diagnostic(b.next().unwrap()));
                }
            }
            (Some(_), None) => out.push(ancestor_cycle_diagnostic(w.next().unwrap())),
            (None, Some(_)) => out.push(ancestor_cycle_diagnostic(b.next().unwrap())),
            (None, None) => break,
        }
    }
    out
}

/// The subset of `DefinitionBuilder` the ancestor-shaped validators need.
/// Lets the cold roast run the same walk over a G-only environment
/// without constructing a full `DefinitionBuilder`, whose eager scans
/// would double the cold path's O(G) work.
pub(crate) struct AncestryEnv<'a> {
    env: &'a Environment,
    ancestors: &'a AncestorBuilder,
    type_params: &'a TypeParamsCache,
}

impl<'a> AncestryEnv<'a> {
    /// `pub(crate)` (rather than private) so the ADR-0028 S5 differential
    /// harness (`environment::differential_tests`) can build a "new
    /// generation" view to hand to [`incremental_ancestor_cycles`]
    /// without constructing a throwaway `DefinitionBuilder` wrapper of
    /// its own.
    pub(crate) fn from_builder(b: &'a DefinitionBuilder) -> Self {
        Self {
            env: b.env(),
            ancestors: b.ancestor_builder(),
            type_params: b.type_params_cache(),
        }
    }

    pub(crate) fn names(&self) -> &NameTable {
        self.env.names()
    }

    /// Mirrors `DefinitionBuilder::declared_kind_by_type_name`'s reach
    /// (class / interface direct hits plus alias-normalized hits) as a
    /// presence check.
    fn is_declared(&self, tn: &TypeName) -> bool {
        if self.env.class_decls().contains_key(tn) || self.env.interface_decls().contains_key(tn) {
            return true;
        }
        if self.env.class_alias_decls().contains_key(tn) {
            let normalized = self.env.normalize_module_name(tn);
            if &normalized != tn {
                return self.is_declared(&normalized);
            }
        }
        false
    }

    /// Mirrors `DefinitionBuilder::class_type_params_by_type_name`.
    fn type_params_of(&self, tn: &TypeName) -> Option<&Vec<TypeParam>> {
        let normalized = self.env.normalize_module_name(tn);
        self.type_params
            .get_or_compute(self.env, &self.ancestors.lowering(), &normalized)
    }
}

/// Run the two ancestor-shaped validators over a G-only frozen
/// environment and return their findings in baked form (ADR-0028 slice
/// 2b-2). Cold path only; the `AncestorBuilder` and type-params cache
/// are throwaways whose `Ty`s never escape into the returned data.
pub(crate) fn roast_g_validators(
    env: &Arc<Environment>,
) -> (
    Vec<BakedAncestorCycle>,
    Vec<(TypeName, Vec<BakedArityViolation>)>,
) {
    let ancestors = AncestorBuilder::new(Arc::clone(env));
    let type_params = TypeParamsCache::default();
    let ctx = AncestryEnv {
        env,
        ancestors: &ancestors,
        type_params: &type_params,
    };
    let names = env.names();

    let mut roots: Vec<TypeName> = env
        .class_decls()
        .keys()
        .chain(env.interface_decls().keys())
        .cloned()
        .collect();
    roots.sort_by_key(|n| names.resolve(n));
    let graph = ancestor_graph(&ctx, roots);
    let cycles: Vec<BakedAncestorCycle> = strongly_connected_components(&graph, names)
        .iter()
        .filter(|component| component.len() >= 2)
        .map(|component| component_to_baked(&ctx, &graph, component))
        .collect();

    let mut arity: Vec<(TypeName, Vec<BakedArityViolation>)> = Vec::new();
    for class_n in env.class_decls().keys() {
        let violations = class_host_arity(&ctx, class_n);
        if !violations.is_empty() {
            arity.push((*class_n, violations));
        }
    }
    for iface_n in env.interface_decls().keys() {
        let violations = interface_host_arity(&ctx, iface_n);
        if !violations.is_empty() {
            arity.push((*iface_n, violations));
        }
    }
    (cycles, arity)
}

/// Successor-closure graph from `seeds`: a worklist expansion over
/// [`ancestor_successors`]. With every declared name as a seed this is
/// the full ancestor graph (flag-off / cold roast); with A-map seeds it
/// is the A-reachable subgraph, whose cost is O(touched) thanks to the
/// per-name lazy one-ancestors probes. With `invalidated`-name seeds
/// (ADR-0028 Decision 4, S5) it is the bounded search space
/// [`incremental_ancestor_cycles`] runs Tarjan over.
fn ancestor_graph(ctx: &AncestryEnv, seeds: Vec<TypeName>) -> FxHashMap<TypeName, Vec<TypeName>> {
    reachable_subgraph(seeds, |name| ancestor_successors(ctx, name))
}

/// BFS worklist expansion computing the successor-reachable subgraph from
/// `seeds`, generic over the edge function so both [`ancestor_graph`]
/// (ancestor edges) and [`incremental_type_alias_cycles`] (type-alias
/// dependency edges) can share it.
fn reachable_subgraph(
    seeds: Vec<TypeName>,
    successors_of: impl Fn(&TypeName) -> Vec<TypeName>,
) -> FxHashMap<TypeName, Vec<TypeName>> {
    let mut graph: FxHashMap<TypeName, Vec<TypeName>> = FxHashMap::default();
    let mut queue: std::collections::VecDeque<TypeName> = seeds.into();
    while let Some(node) = queue.pop_front() {
        if graph.contains_key(&node) {
            continue;
        }
        let successors = successors_of(&node);
        for next in &successors {
            if !graph.contains_key(next) {
                queue.push_back(*next);
            }
        }
        graph.insert(node, successors);
    }
    graph
}

fn ancestor_successors(ctx: &AncestryEnv, name: &TypeName) -> Vec<TypeName> {
    let environment = ctx.env;
    let normalized = environment.normalize_module_name(name);
    let in_class = environment.class_decls().contains_key(&normalized);
    let in_interface = environment.interface_decls().contains_key(&normalized);
    if !in_class && !in_interface {
        // Unknown name (e.g. a relative or unresolved target). The arity
        // validator skips unknown targets the same way.
        return Vec::new();
    }

    let mut successors: Vec<TypeName> = Vec::new();
    if in_class {
        let one = ctx.ancestors.one_instance_ancestors_arc(&normalized);
        if let Some(Ancestor::Instance { name: sup, .. }) = &one.super_class {
            successors.push(*sup);
        }
        for m in &one.included_modules {
            successors.push(m.name);
        }
        for m in &one.included_interfaces {
            successors.push(m.name);
        }
        for m in &one.prepended_modules {
            successors.push(m.name);
        }
        for m in &one.self_types {
            successors.push(m.name);
        }
    } else {
        // Interface entries fold only `included_interfaces`; rbs's
        // `interface_ancestors` (lib/rbs/definition_builder/ancestor_builder.rb)
        // mirrors this restriction — no super, no module include, no
        // self_types.
        let one = ctx.ancestors.one_interface_ancestors_arc(&normalized);
        for m in &one.included_interfaces {
            successors.push(m.name);
        }
    }

    successors
        .into_iter()
        .filter_map(|next| {
            let next_normalized = environment.normalize_module_name(&next);
            let next_in_graph = environment.class_decls().contains_key(&next_normalized)
                || environment.interface_decls().contains_key(&next_normalized);
            if !next_in_graph {
                return None;
            }
            // Keep the existing single-node ancestor self-edge policy:
            // synthetic Object self-super and explicit self-include stay
            // silent, while multi-node SCCs are reported below.
            if next_normalized == normalized {
                return None;
            }
            Some(next_normalized)
        })
        .collect()
}

fn component_to_baked(
    ctx: &AncestryEnv,
    graph: &FxHashMap<TypeName, Vec<TypeName>>,
    component: &[TypeName],
) -> BakedAncestorCycle {
    let names = ctx.names();
    let anchor = *component
        .iter()
        .min_by_key(|n| names.resolve(**n))
        .expect("component non-empty: SCC caller filters to cyclic components");
    let chain: Vec<String> = component_closed_walk(graph, component, anchor, names)
        .into_iter()
        .map(|n| names.resolve(n))
        .collect();

    BakedAncestorCycle {
        participants: component.to_vec(),
        type_name: names.resolve(anchor),
        chain,
        primary_source: primary_decl_location(ctx, &anchor),
    }
}

fn ancestor_cycle_diagnostic(cycle: &BakedAncestorCycle) -> Diagnostic {
    Diagnostic {
        scope: None,
        kind: DiagnosticKind::RecursiveAncestor {
            type_name: cycle.type_name.clone(),
            chain: cycle.chain.clone(),
            primary_source: cycle.primary_source.clone(),
        },
        location: resolve_or_default(cycle.primary_source.as_ref()),
    }
}

fn component_closed_walk(
    graph: &FxHashMap<TypeName, Vec<TypeName>>,
    component: &[TypeName],
    anchor: TypeName,
    names: &NameTable,
) -> Vec<TypeName> {
    let mut targets: Vec<TypeName> = component.iter().copied().filter(|n| *n != anchor).collect();
    targets.sort_by_key(|n| names.resolve(n));

    let mut walk = vec![anchor];
    let mut current = anchor;
    for target in targets {
        append_path_in_component(graph, component, current, target, names, &mut walk);
        current = target;
    }
    append_path_in_component(graph, component, current, anchor, names, &mut walk);
    walk
}

fn append_path_in_component(
    graph: &FxHashMap<TypeName, Vec<TypeName>>,
    component: &[TypeName],
    start: TypeName,
    goal: TypeName,
    names: &NameTable,
    walk: &mut Vec<TypeName>,
) {
    if start == goal {
        return;
    }

    let component_set: FxHashSet<TypeName> = component.iter().copied().collect();
    let mut queue = std::collections::VecDeque::from([start]);
    let mut seen = FxHashSet::default();
    let mut prev: FxHashMap<TypeName, TypeName> = FxHashMap::default();
    seen.insert(start);

    while let Some(node) = queue.pop_front() {
        let mut successors = graph.get(&node).cloned().unwrap_or_default();
        successors.sort_by_key(|n| names.resolve(n));
        successors.dedup();
        for next in successors {
            if !component_set.contains(&next) || !seen.insert(next) {
                continue;
            }
            prev.insert(next, node);
            if next == goal {
                let mut path = vec![goal];
                let mut cursor = goal;
                while cursor != start {
                    cursor = prev[&cursor];
                    path.push(cursor);
                }
                path.reverse();
                walk.extend(path.into_iter().skip(1));
                return;
            }
            queue.push_back(next);
        }
    }
}

fn strongly_connected_components(
    graph: &FxHashMap<TypeName, Vec<TypeName>>,
    names: &NameTable,
) -> Vec<Vec<TypeName>> {
    struct Tarjan<'a> {
        graph: &'a FxHashMap<TypeName, Vec<TypeName>>,
        names: &'a NameTable,
        next_index: usize,
        indices: FxHashMap<TypeName, usize>,
        lowlinks: FxHashMap<TypeName, usize>,
        stack: Vec<TypeName>,
        on_stack: FxHashSet<TypeName>,
        components: Vec<Vec<TypeName>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, node: TypeName) {
            let index = self.next_index;
            self.next_index += 1;
            self.indices.insert(node, index);
            self.lowlinks.insert(node, index);
            self.stack.push(node);
            self.on_stack.insert(node);

            let mut successors = self.graph.get(&node).cloned().unwrap_or_default();
            successors.sort_by_key(|n| self.names.resolve(n));
            successors.dedup();

            for next in successors {
                if !self.indices.contains_key(&next) {
                    self.visit(next);
                    let next_lowlink = self.lowlinks[&next];
                    let node_lowlink = self.lowlinks[&node];
                    self.lowlinks.insert(node, node_lowlink.min(next_lowlink));
                } else if self.on_stack.contains(&next) {
                    let next_index = self.indices[&next];
                    let node_lowlink = self.lowlinks[&node];
                    self.lowlinks.insert(node, node_lowlink.min(next_index));
                }
            }

            if self.lowlinks[&node] != self.indices[&node] {
                return;
            }

            let mut component = Vec::new();
            while let Some(member) = self.stack.pop() {
                self.on_stack.remove(&member);
                component.push(member);
                if member == node {
                    break;
                }
            }
            component.sort_by_key(|n| self.names.resolve(n));
            self.components.push(component);
        }
    }

    // Visitation order does not affect which SCCs are found (that is a
    // structural property of the graph, not of traversal order) — only
    // the presentation does, and that is already made deterministic below
    // by sorting each component's members and then the component list
    // itself, both O(#cycles) instead of O(#roots). So the root list here
    // is intentionally left in whatever order `graph.keys()` yields
    // (ADR-0028 Decision 4, S5 "sort smell" fix: `resolve`+`sort` moves
    // from every root to just the handful of reported cycles).
    let roots: Vec<TypeName> = graph.keys().copied().collect();

    let mut tarjan = Tarjan {
        graph,
        names,
        next_index: 0,
        indices: FxHashMap::default(),
        lowlinks: FxHashMap::default(),
        stack: Vec::new(),
        on_stack: FxHashSet::default(),
        components: Vec::new(),
    };

    for root in roots {
        if !tarjan.indices.contains_key(&root) {
            tarjan.visit(root);
        }
    }

    tarjan
        .components
        .sort_by_key(|component| names.resolve(component[0]));
    tarjan.components
}

fn has_self_edge(graph: &FxHashMap<TypeName, Vec<TypeName>>, node: TypeName) -> bool {
    graph
        .get(&node)
        .is_some_and(|successors| successors.contains(&node))
}

/// `RubyClassDecl` / `RubyModuleDecl` track byte ranges per member but
/// have no decl-level `LocationRange`, so inline anchors degrade to `None` —
/// the diagnostic still surfaces with the empty path / line 1 fallback
/// that `resolve_or_default` produces for locationless inputs.
fn primary_decl_location(ctx: &AncestryEnv, name: &TypeName) -> Option<SourceLocation> {
    let environment = ctx.env;
    let names = ctx.names();
    if let Some(class_or_module) = environment.class_decls().get(name) {
        return match class_or_module {
            ClassOrModule::Class(entry) => match entry.primary_decl() {
                ClassDeclaration::Signature(c) => {
                    c.source_file.zip(c.location).map(|(f, l)| SourceLocation {
                        file: PathBuf::from(names.resolve(f)),
                        range: l.range,
                    })
                }
                ClassDeclaration::Ruby(_) => None,
            },
            ClassOrModule::Module(entry) => match entry.primary_decl() {
                ModuleDeclaration::Signature(m) => {
                    m.source_file.zip(m.location).map(|(f, l)| SourceLocation {
                        file: PathBuf::from(names.resolve(f)),
                        range: l.range,
                    })
                }
                ModuleDeclaration::Ruby(_) => None,
            },
        };
    }
    let entry = environment.interface_decls().get(name)?;
    let decl = entry.decl();
    decl.source_file
        .zip(decl.location)
        .map(|(f, l)| SourceLocation {
            file: PathBuf::from(names.resolve(f)),
            range: l.range,
        })
}

/// Emit `RecursiveAliasDefinition` diagnostics for cyclic alias chains
/// collected during the same construction-time scan that produces
/// `method_dups`. Mirrors the `RBS::RecursiveAliasDefinitionError` raise
/// in `method_builder.rb`'s `Methods#each`.
fn validate_alias_cycles(env: &DefinitionBuilder, _sources: &SourceCache<'_>) -> Vec<Diagnostic> {
    env.alias_cycles()
        .into_iter()
        .map(|cycle| {
            let location = resolve_or_default(cycle.primary_location.as_ref());
            Diagnostic {
                scope: None,
                kind: DiagnosticKind::RecursiveAliasDefinition {
                    type_name: cycle.type_name.clone(),
                    alias_names: cycle.alias_names.clone(),
                    primary_source: cycle.primary_location.clone(),
                },
                location,
            }
        })
        .collect()
}

/// Emit `RecursiveTypeAlias` diagnostics for cyclic type alias graphs.
/// Mirrors rbs's `Validator#validate_type_alias` raise of
/// `RecursiveTypeAliasError` (`lib/rbs/validator.rb` L63-68), which keys on
/// `TypeAliasDependency#circular_definition?` (`lib/rbs/type_alias_dependency.rb`
/// L65-78). The dependency graph follows the same transparency rule as
/// rbs's `direct_dependency`: `Union`, `Intersection`, and `Optional` nodes
/// are walked through, every other constructor (`ClassInstance`, `Tuple`,
/// `Record`, `Proc`, `Interface`, etc.) is opaque. That is why
/// `type a = Array[a]`, `type a = [a, a]`, `type a = ^(a) -> a`, and
/// `type a = { foo: a }` are all regular and emit nothing.
///
/// crema collapses every cycle to a single diagnostic keyed by the
/// lexicographically smallest participant, matching
/// [`full_ancestor_cycles`] and unlike rbs (which raises per-entry).
///
/// Full recompute of every type-alias cycle in `env`, factored out so it
/// can also serve as [`incremental_type_alias_cycles`]'s first-generation
/// state producer (ADR-0028 Decision 4, S5; see [`full_ancestor_cycles`]
/// for why the full path returns state at all). The graph is built
/// directly over every declared type alias rather than via
/// [`reachable_subgraph`] — every alias is already a root, so there is no
/// bounded-vs-full distinction to make here (unlike the incremental entry
/// point below). Unlike the ancestor validator, there is no G-backend
/// split — type-alias cycles have no G-baked sibling (see
/// [`BakedTypeAliasCycle`]'s doc), so this single function covers both
/// backends and [`full_validate`] / [`validate_update`] call it
/// unconditionally.
pub(crate) fn full_type_alias_cycles(
    env: &DefinitionBuilder,
) -> (TypeAliasCycleState, Vec<Diagnostic>) {
    let environment = env.env();
    let names = env.names();

    let roots: Vec<TypeName> = environment.type_alias_decls().keys().cloned().collect();
    let mut graph: FxHashMap<TypeName, Vec<TypeName>> = FxHashMap::default();
    for root in &roots {
        graph.insert(*root, type_alias_successors(env, root));
    }

    let state: TypeAliasCycleState = strongly_connected_components(&graph, names)
        .iter()
        .filter(|component| component.len() >= 2 || has_self_edge(&graph, component[0]))
        .map(|component| Arc::new(build_type_alias_cycle(env, component)))
        .collect();
    let diagnostics = state
        .iter()
        .map(|c| type_alias_cycle_diagnostic(c))
        .collect();
    (state, diagnostics)
}

/// One collapsed `RecursiveTypeAlias` finding, persisted across
/// generations for [`incremental_type_alias_cycles`]. Mirrors
/// [`BakedAncestorCycle`]'s role for the ancestor validator. Not the same
/// thing as the pre-existing `MBakedAliasCycle` (`snapshot/write.rs`) —
/// that one is the serialized form of `AliasCycleEntry`, i.e. the *method*
/// `alias` keyword's cycle, which `validate_alias_cycles` (a different,
/// already per-name-differentiable validator) handles. This struct is for
/// `type X = ...` type-alias cycles, crema-only with no G-baked sibling.
///
/// `participants` / `anchor` are read only by [`incremental_type_alias_cycles`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BakedTypeAliasCycle {
    pub(crate) participants: Vec<TypeName>,
    /// Resolved name of the lexicographically smallest participant — the
    /// sort/identity key that reproduces `strongly_connected_components`'s
    /// component-order output, mirroring [`BakedAncestorCycle::type_name`]'s
    /// role. Named `anchor` rather than `type_name` because this
    /// validator's diagnostic has no singular "type_name" field of its
    /// own (it reports `alias_names`, plural).
    pub(crate) anchor: String,
    pub(crate) alias_names: Vec<String>,
    pub(crate) primary_source: Option<SourceLocation>,
}

/// Persisted state for [`incremental_type_alias_cycles`]. See
/// [`AncestorCycleState`] for why entries are `Arc`-wrapped.
pub(crate) type TypeAliasCycleState = Vec<Arc<BakedTypeAliasCycle>>;

fn build_type_alias_cycle(env: &DefinitionBuilder, component: &[TypeName]) -> BakedTypeAliasCycle {
    let names = env.names();
    let mut alias_names: Vec<String> = component.iter().map(|n| names.resolve(n)).collect();
    alias_names.sort();

    let anchor_name = *component
        .iter()
        .min_by_key(|n| names.resolve(**n))
        .expect("component non-empty: SCC caller filters to cyclic components");
    let anchor = names.resolve(anchor_name);

    let primary_source = env
        .env()
        .type_alias_decls()
        .get(&anchor_name)
        .and_then(|entry| {
            entry
                .decl
                .source_file
                .zip(entry.decl.location)
                .map(|(f, l)| SourceLocation {
                    file: PathBuf::from(env.names().resolve(f)),
                    range: l.name_range,
                })
        });

    BakedTypeAliasCycle {
        participants: component.to_vec(),
        anchor,
        alias_names,
        primary_source,
    }
}

fn type_alias_cycle_diagnostic(cycle: &BakedTypeAliasCycle) -> Diagnostic {
    let location = resolve_or_default(cycle.primary_source.as_ref());
    Diagnostic {
        scope: None,
        kind: DiagnosticKind::RecursiveTypeAlias {
            alias_names: cycle.alias_names.clone(),
            primary_source: cycle.primary_source.clone(),
        },
        location,
    }
}

/// Incremental counterpart to [`full_type_alias_cycles`]. See
/// [`incremental_ancestor_cycles`] for the full soundness argument —
/// identical reasoning applies with [`type_alias_successors`] in place of
/// `ancestor_successors`: a type alias's dependency edges are derived only
/// from its own `type X = ...` body, so an edge changes iff its owning
/// alias is in `invalidated`, and the same three-step
/// drop-affected/bounded-research/keep-invalidated-touched update applies.
/// No G-backend split, same as [`full_type_alias_cycles`].
pub(crate) fn incremental_type_alias_cycles(
    old_state: &TypeAliasCycleState,
    invalidated: &FxHashSet<TypeName>,
    env: &DefinitionBuilder,
) -> (TypeAliasCycleState, Vec<Diagnostic>) {
    let names = env.names();

    let mut merged: TypeAliasCycleState = old_state
        .iter()
        .filter(|cycle| !cycle.participants.iter().any(|p| invalidated.contains(p)))
        .cloned()
        .collect();

    let seeds: Vec<TypeName> = invalidated.iter().copied().collect();
    let graph = reachable_subgraph(seeds, |name| type_alias_successors(env, name));
    for component in strongly_connected_components(&graph, names) {
        let is_cycle = component.len() >= 2 || has_self_edge(&graph, component[0]);
        if !is_cycle || !component.iter().any(|p| invalidated.contains(p)) {
            continue;
        }
        merged.push(Arc::new(build_type_alias_cycle(env, &component)));
    }
    merged.sort_by(|a, b| a.anchor.cmp(&b.anchor));

    let diagnostics = merged
        .iter()
        .map(|c| type_alias_cycle_diagnostic(c))
        .collect();
    (merged, diagnostics)
}

fn type_alias_successors(env: &DefinitionBuilder, name: &TypeName) -> Vec<TypeName> {
    let environment = env.env();
    let Some(entry) = environment.type_alias_decls().get(name) else {
        return Vec::new();
    };

    let mut deps: Vec<TypeName> = Vec::new();
    collect_direct_alias_deps(&entry.decl.ty, &mut deps);
    deps.into_iter()
        .filter(|next| environment.type_alias_decls().contains_key(next))
        .collect()
}

/// Mirrors rbs `TypeAliasDependency#direct_dependency`
/// (`lib/rbs/type_alias_dependency.rb` L65-78): walk through `Union`,
/// `Intersection`, `Optional`; record `Alias` names; treat every other
/// type constructor as opaque. The walker therefore distinguishes
/// circular (`type a = a?`) from regular (`type a = Array[a]`) the same
/// way rbs does.
fn collect_direct_alias_deps(body: &AstType, out: &mut Vec<TypeName>) {
    match body {
        AstType::Union(t) => {
            for ty in &t.types {
                collect_direct_alias_deps(ty, out);
            }
        }
        AstType::Intersection(t) => {
            for ty in &t.types {
                collect_direct_alias_deps(ty, out);
            }
        }
        AstType::Optional(t) => {
            collect_direct_alias_deps(&t.ty, out);
        }
        AstType::Alias(t) => {
            out.push(t.name);
        }
        _ => {}
    }
}

/// Emit `DuplicatedMethodDefinition` diagnostics for duplicate method definitions
/// collected during class-cache population. Mirrors the
/// `RBS::DuplicatedMethodDefinitionError` raise in `method_builder.rb#validate!`.
fn validate_method_dups(env: &DefinitionBuilder, _sources: &SourceCache<'_>) -> Vec<Diagnostic> {
    env.method_dups()
        .into_iter()
        .map(|(method_name, dup_loc, original_loc)| {
            let location = resolve_or_default(dup_loc.as_ref());
            Diagnostic {
                scope: None,
                kind: DiagnosticKind::DuplicatedMethodDefinition {
                    method_name: method_name.clone(),
                    duplicate_source: original_loc.clone(),
                },
                location,
            }
        })
        .collect()
}

fn validate_variable_dups(env: &DefinitionBuilder, sources: &SourceCache<'_>) -> Vec<Diagnostic> {
    env.variable_dups()
        .into_iter()
        .map(|dup| {
            let location = match dup.ruby_source_location {
                Some(loc) => resolve_ruby_location(env, sources, loc),
                None => resolve_or_default(dup.location.as_ref()),
            };
            let type_name = env.names().resolve(dup.type_name);
            let variable_name = env.names().resolve(dup.variable_name);
            let kind = match dup.kind {
                VariableDuplicationKind::Instance => DiagnosticKind::InstanceVariableDuplication {
                    type_name,
                    variable_name,
                    duplicate_source: dup.location.clone(),
                },
                VariableDuplicationKind::ClassInstance => {
                    DiagnosticKind::ClassInstanceVariableDuplication {
                        type_name,
                        variable_name,
                        duplicate_source: dup.location.clone(),
                    }
                }
            };
            Diagnostic {
                scope: None,
                kind,
                location,
            }
        })
        .collect()
}

/// Verify every `include M[X, Y]`, `extend M[X]`, `prepend M[X]`, and
/// `class A < B[X]` supplies a type-argument count within the target's
/// `[min_arity, max_arity]` range. Also covers `interface _Sub include
/// _Base[X]` — interface hosts can only carry `include` mixins (rbs's
/// C parser rejects `extend` / `prepend` inside interfaces), so the
/// interface arm only inspects `included_interfaces`.
///
/// Mirrors RBS's `InvalidTypeApplicationError.check!`
/// (`lib/rbs/definition_builder.rb`), which raises an exception instead of
/// producing a diagnostic. crema surfaces the same condition as a
/// structured `MixinTypeArgumentArityMismatch` Diagnostic with location so
/// it shows up in the same stream as type-check errors.
///
/// Targets that aren't declared (e.g. built-in classes the user didn't
/// load, or implicit supers from `finalize_implicit_superclasses`) are
/// skipped. MixinRefs without a source location — implicit refs — are
/// also skipped, since arity 0/0 on those is always correct and any
/// violation would indicate a bug in how the ref was constructed rather
/// than user-actionable feedback.
///
/// Persisted state for [`incremental_applied_type_args`] /
/// [`incremental_applied_type_args_a`]: every currently known mixin-arity
/// violation, grouped per owner — the same per-owner cache shape
/// `DefinitionBuilder` already uses for `method_dups` / `alias_cycles` /
/// `variable_dups` ([`PerNameCache`]).
pub(crate) type ArityState = PerNameCache<BakedArityViolation>;

/// Full recompute for the flag-off (no G backend) case, factored out so
/// it can also serve as [`incremental_applied_type_args`]'s
/// first-generation state producer (ADR-0028 S5v, the same "full path
/// also returns state" pattern as [`full_ancestor_cycles`]).
/// [`full_applied_type_args_a`] is the G-backend-attached counterpart,
/// wired in by [`full_validate`] / [`validate_update`] via
/// `Environment::g_backend`.
pub(crate) fn full_applied_type_args(ctx: &AncestryEnv) -> (ArityState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let mut state = ArityState::new();
    for class_n in environment.class_decls().keys() {
        state.push(*class_n, class_host_arity(ctx, class_n));
    }
    for iface_n in environment.interface_decls().keys() {
        state.push(*iface_n, interface_host_arity(ctx, iface_n));
    }
    let diagnostics = state
        .iter()
        .flat_map(|(_, violations)| violations.iter().map(arity_diagnostic).collect::<Vec<_>>())
        .collect();
    (state, diagnostics)
}

/// Incremental counterpart to [`full_applied_type_args`] (ADR-0028 S5v).
/// Every arity violation for an owner is derived only from that owner's
/// own mixin refs (`class_host_arity` / `interface_host_arity` walk only
/// the owner's *one-step* ancestors) plus the mixin target's type-param
/// arity — and a target's own type-param change makes the target itself
/// `invalidated`, which (via `AncestorGraph` descendant propagation,
/// ADR-0028 Decision 4 rule 3) also invalidates every owner referencing
/// it as a mixin. So recomputing exactly the `invalidated` owners and
/// carrying every other owner's violations forward is sound: no
/// non-invalidated owner's violation set can have changed.
pub(crate) fn incremental_applied_type_args(
    old_state: &ArityState,
    invalidated: &FxHashSet<TypeName>,
    ctx: &AncestryEnv,
) -> (ArityState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let mut state = old_state.without(invalidated);
    for &owner in invalidated {
        if environment.class_decls().contains_key(&owner) {
            state.push(owner, class_host_arity(ctx, &owner));
        } else if environment.interface_decls().contains_key(&owner) {
            state.push(owner, interface_host_arity(ctx, &owner));
        }
    }
    let diagnostics = state
        .iter()
        .flat_map(|(_, violations)| violations.iter().map(arity_diagnostic).collect::<Vec<_>>())
        .collect();
    (state, diagnostics)
}

/// G-backend-attached counterpart of [`full_applied_type_args`]
/// (ADR-0028 S5v): A-restricted walk + baked splice, mirroring the scan
/// splice in `scan_method_builder_diagnostics`. The raw A-map fields are
/// walked (their merged entries carry both layers' mixins, so a
/// reopened G host's G-side violation is re-found there), and baked
/// groups for hosts the A map redeclares are excluded to avoid the
/// double.
pub(crate) fn full_applied_type_args_a(
    ctx: &AncestryEnv,
    g: &GSnapshotBackend,
) -> (ArityState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let mut state = ArityState::new();
    for class_n in environment.class_decls().a_keys() {
        state.push(*class_n, class_host_arity(ctx, class_n));
    }
    for iface_n in environment.interface_decls().a_keys() {
        state.push(*iface_n, interface_host_arity(ctx, iface_n));
    }
    let diagnostics = applied_type_args_diagnostics_with_g_splice(&state, g, environment);
    (state, diagnostics)
}

/// G-backend-attached counterpart of [`incremental_applied_type_args`]
/// (ADR-0028 S5v). See that function's doc for why recomputing only
/// `invalidated` owners is sound; the state here is A-only (mirroring
/// [`full_applied_type_args_a`]'s walk), so an `invalidated` name outside
/// both raw A-map fields (e.g. a G-only name reached via descendant
/// propagation) contributes no state entry — its arity, if any, is
/// already covered by `g`'s baked splice, unaffected since G never
/// changes mid-session.
pub(crate) fn incremental_applied_type_args_a(
    old_state: &ArityState,
    invalidated: &FxHashSet<TypeName>,
    ctx: &AncestryEnv,
    g: &GSnapshotBackend,
) -> (ArityState, Vec<Diagnostic>) {
    let environment = ctx.env;
    let mut state = old_state.without(invalidated);
    for &owner in invalidated {
        if environment.a_class_contains(&owner) {
            state.push(owner, class_host_arity(ctx, &owner));
        } else if environment.interface_decls().a_contains_key(&owner) {
            state.push(owner, interface_host_arity(ctx, &owner));
        }
    }
    let diagnostics = applied_type_args_diagnostics_with_g_splice(&state, g, environment);
    (state, diagnostics)
}

/// Append `g`'s baked arity violations for hosts the A map does not
/// redeclare after `state`'s own diagnostics — matches
/// [`full_applied_type_args_a`]'s original inline splice order (A-walk
/// entries first, then baked; no by-name merge sort, unlike the ancestor
/// validator's [`ancestor_cycle_diagnostics_with_g_splice`]).
fn applied_type_args_diagnostics_with_g_splice(
    state: &ArityState,
    g: &GSnapshotBackend,
    environment: &Environment,
) -> Vec<Diagnostic> {
    let mut diagnostics: Vec<Diagnostic> = state
        .iter()
        .flat_map(|(_, violations)| violations.iter().map(arity_diagnostic).collect::<Vec<_>>())
        .collect();
    for (owner, violations) in &g.baked().arity {
        if !environment.a_class_contains(owner)
            && !environment.interface_decls().a_contains_key(owner)
        {
            diagnostics.extend(violations.iter().map(arity_diagnostic));
        }
    }
    diagnostics
}

fn class_host_arity(ctx: &AncestryEnv, class_n: &TypeName) -> Vec<BakedArityViolation> {
    let mut out = Vec::new();
    let one_instance = ctx.ancestors.one_instance_ancestors_arc(class_n);
    let one_singleton = ctx.ancestors.one_singleton_ancestors_arc(class_n);

    if let Some(mixin) = one_instance
        .super_class
        .as_ref()
        .and_then(Ancestor::to_instance_mixin_ref)
        && let Some(v) = check_one(ctx, class_n, &mixin, "superclass")
    {
        out.push(v);
    }
    for m in one_instance
        .included_modules
        .iter()
        .chain(one_instance.included_interfaces.iter())
    {
        if let Some(v) = check_one(ctx, class_n, m, "include") {
            out.push(v);
        }
    }
    for m in one_singleton
        .extended_modules
        .iter()
        .chain(one_singleton.extended_interfaces.iter())
    {
        if let Some(v) = check_one(ctx, class_n, m, "extend") {
            out.push(v);
        }
    }
    for m in &one_instance.prepended_modules {
        if let Some(v) = check_one(ctx, class_n, m, "prepend") {
            out.push(v);
        }
    }
    out
}

fn interface_host_arity(ctx: &AncestryEnv, iface_n: &TypeName) -> Vec<BakedArityViolation> {
    let mut out = Vec::new();
    let one = ctx.ancestors.one_interface_ancestors_arc(iface_n);
    for m in &one.included_interfaces {
        if let Some(v) = check_one(ctx, iface_n, m, "include") {
            out.push(v);
        }
    }
    out
}

fn arity_diagnostic(v: &BakedArityViolation) -> Diagnostic {
    Diagnostic {
        scope: None,
        kind: DiagnosticKind::MixinTypeArgumentArityMismatch {
            kind: v.kind.to_string(),
            target: v.target.clone(),
            class: v.class.clone(),
            expected: v.expected.clone(),
            got: v.got,
        },
        location: resolve_or_default(v.location.as_ref()),
    }
}

fn check_one(
    ctx: &AncestryEnv,
    host_n: &TypeName,
    mixin: &MixinRef,
    kind: &'static str,
) -> Option<BakedArityViolation> {
    // Skip targets we don't know about — arity is only meaningful when we
    // can see the target's declared params. This also gracefully handles
    // implicit `Object` supers added by `finalize_implicit_superclasses`
    // for classes loaded without RBS.
    if !ctx.is_declared(&mixin.name) {
        return None;
    }
    let params = ctx.type_params_of(&mixin.name);
    let max_arity = params.map(|p| p.len()).unwrap_or(0);
    // ADR-0017 §"Single source of truth: `type_params` from `primary_decl`":
    // `type_params_min_arity` is derived on demand as the count of leading
    // params with no default.
    let min_arity = params
        .map(|p| p.iter().take_while(|tp| tp.default_type.is_none()).count())
        .unwrap_or(max_arity);
    let got = mixin.args.len();
    // Matches RBS's `InvalidTypeApplicationError.check!` rule:
    // `min_arity <= got <= max_arity` (default-typed params can fill the
    // tail). Bare references (`got == 0`) must still satisfy the
    // min_arity floor — e.g. `include Enumerable` (no args) fails
    // because `Enumerable[Elem]` has no default for `Elem`.
    if min_arity <= got && got <= max_arity {
        return None;
    }
    let expected = if min_arity == max_arity {
        format!("{}", max_arity)
    } else {
        format!("{}..{}", min_arity, max_arity)
    };

    Some(BakedArityViolation {
        kind,
        target: ctx.names().resolve(mixin.name),
        class: ctx.names().resolve(host_n),
        expected,
        got,
        location: mixin.location.clone(),
    })
}

/// Translate an optional `SourceLocation` to a diagnostic location. `None`
/// falls back to an empty path and a 0-width range at byte 0. Production
/// `.rbs`/`.rb`/`-e` paths set a location; the fallback is for in-memory
/// tests and implicit refs that cannot be anchored to source.
fn resolve_or_default(loc: Option<&SourceLocation>) -> SourceLocation {
    let Some(sl) = loc else {
        return SourceLocation {
            file: PathBuf::new(),
            range: crate::location::LocationRange::new(0, 0, 0, 0),
        };
    };
    sl.clone()
}

/// Translate a byte-range `RubyLocation` into a `SourceLocation`.
fn resolve_ruby_location(
    env: &DefinitionBuilder,
    sources: &SourceCache<'_>,
    loc: RubyLocation,
) -> SourceLocation {
    let file_str = env.names().resolve(loc.file);
    let path = PathBuf::from(&file_str);
    if let Some(source) = sources.get(&path) {
        Diagnostic::location_for_byte_range(
            path,
            source,
            loc.start_byte as usize,
            loc.end_byte as usize,
        )
    } else if let Ok(source) = std::fs::read(&path) {
        Diagnostic::location_for_byte_range(
            path,
            &source,
            loc.start_byte as usize,
            loc.end_byte as usize,
        )
    } else {
        SourceLocation {
            file: path,
            range: crate::location::LocationRange::new(
                loc.start_byte,
                loc.start_byte,
                loc.end_byte,
                loc.end_byte,
            ),
        }
    }
}

