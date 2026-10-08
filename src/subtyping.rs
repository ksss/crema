use crate::definition::{Ancestor, AncestorSource, Method};
use crate::definition_builder::{self, ConsultationView, SubtypeCacheKey};
use crate::name::Symbol;
use crate::substitution::Substitution;
use crate::type_name::TypeName;
use crate::type_param::TypeVarKey;
use crate::types::{Block, FunctionType, Literal, RecordKey, Ty, Type};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;

/// One bound-constraint violation discovered by
/// `SubtypeChecker::check_type_arg_bounds`. The caller converts this into
/// a `DiagnosticKind::TypeArgumentBoundViolation` (string-formatting the
/// types) — keeping raw `Ty` here lets callers choose the display form.
#[derive(Debug, Clone)]
pub struct BoundViolation {
    pub param_name: Symbol,
    pub bound_kind: crate::diagnostic::BoundKind,
    pub bound: Ty,
    pub actual: Ty,
}

/// Cap on `SubtypeChecker::check`'s in-flight recursion depth
/// (number of in-flight relations at entry), mirroring Steep's
/// `Subtyping::Check::ABORT_LIMIT` (`subtyping/check.rb:4`, default 50).
/// Coinductive self-referential shapes (e.g. a block param typed
/// `Box[E, self]` checked against an interface block param typed `self`)
/// re-derive a structurally *new* `Ty` on every recursion — one more layer
/// of wrapping each time — so the exact-match in-flight guard below
/// never revisits the same key and never short-circuits. Steep hits the
/// same growth and fails closed past this depth (`Result::Failure::
/// LoopAbort`, verified via steep-playground 2026-07-21: `Box[Integer,
/// untyped] <: _EachEntryLike[Integer]` fires `Ruby::ArgumentTypeMismatch`,
/// not silent, not a crash) rather than accepting past it — mirror that
/// direction here too, not a gradual/untyped accept. This cap counts total
/// in-flight `check()` frames, not self-referential growth specifically, so
/// a legitimately deep (51+ level) but non-recursive type comparison would
/// also abort — the same trade-off Steep's own `ABORT_LIMIT` makes.
///
/// A relation that hits the shared memo returns without pushing a frame, so
/// whether a deep comparison reaches this cap depends on what earlier queries
/// memoized. Only non-diverging comparisons nested 50+ levels are affected,
/// and the order changes only false positives, never missed errors. This is
/// an accepted limitation (ADR-0034), not a spec.
const SUBTYPE_ASSUMPTION_ABORT_LIMIT: usize = 50;

/// Checks subtype relations between types.
pub struct SubtypeChecker<'a> {
    env: ConsultationView<'a>,
    type_variables_are_wildcards: bool,
    self_bound: Option<Ty>,
    search: RefCell<Search>,
}

/// Per-query state of the coinductive search behind
/// [`SubtypeChecker::check`].
///
/// Invariant: every `provisional` dependency names a frame that is still
/// on `frames` — when a frame ends, the results resting on it are either
/// confirmed into the shared memo, discarded, or re-pointed at the outer
/// frame the ending one itself rests on.
#[derive(Default)]
struct Search {
    /// In-flight relations, outermost first. A relation re-entered while
    /// in flight is assumed to hold (coinduction).
    frames: Vec<Frame>,
    /// `key -> index into frames` for the in-flight relations.
    in_flight: FxHashMap<SubtypeCacheKey, usize>,
    /// Relations that came out `true` only by assuming an in-flight one,
    /// in the order they finished. Not shared: they are sound only once
    /// the frame they rest on also finishes `true`.
    provisional: Vec<SubtypeCacheKey>,
    /// `key -> lowest frame index it rests on`, for each `provisional` key.
    provisional_rests_on: FxHashMap<SubtypeCacheKey, usize>,
    /// Relations that came out `false` through the depth cap during the
    /// current outermost query. Not shared (a shallower start may prove
    /// them), but reused within the query: a self-wrapping shape reaches
    /// each level's key once per overload, and recomputing it every time
    /// branches exponentially before the cap is hit.
    aborted_false: FxHashSet<SubtypeCacheKey>,
}

struct Frame {
    key: SubtypeCacheKey,
    /// Lowest frame index a `true` inside this frame was derived from by
    /// assumption (`usize::MAX` when none).
    rests_on: usize,
    /// Whether a `false` inside this frame came from the depth cap.
    aborted: bool,
    /// `provisional.len()` when this frame was pushed.
    provisional_start: usize,
}

impl Search {
    fn rest_innermost_on(&mut self, index: usize) {
        if let Some(frame) = self.frames.last_mut() {
            frame.rests_on = frame.rests_on.min(index);
        }
    }

    fn drain_provisional_from(&mut self, start: usize) -> Vec<SubtypeCacheKey> {
        let drained: Vec<_> = self.provisional.drain(start..).collect();
        for key in &drained {
            self.provisional_rests_on.remove(key);
        }
        drained
    }
}

impl<'a> SubtypeChecker<'a> {
    pub fn new(env: ConsultationView<'a>) -> Self {
        SubtypeChecker {
            env,
            type_variables_are_wildcards: true,
            self_bound: None,
            search: RefCell::default(),
        }
    }

    pub fn preserving_type_variables(env: ConsultationView<'a>) -> Self {
        SubtypeChecker {
            env,
            type_variables_are_wildcards: false,
            self_bound: None,
            search: RefCell::default(),
        }
    }

    /// Bind the concrete `self` type for SUB-side `self <: X` widening,
    /// mirroring Steep's `self_type` context in `subtyping/check.rb:417`.
    /// A self-bearing check context (method-body return, argument passing)
    /// supplies its `current_self_type()`; env-only contexts leave it `None`
    /// so `Ty::SELF_TYPE` stays opaque.
    pub fn with_self_bound(mut self, self_type: Ty) -> Self {
        self.self_bound = Some(self_type);
        self
    }

    /// Returns true if `sub` is a subtype of `sup`.
    ///
    /// Wraps [`Self::check_uncached`] with an env-shared memoization layer
    /// keyed by `(sub, sup, self_bound, type_variables_are_wildcards)`,
    /// like `Steep::Subtyping::Check#check_type` consults `@cache`
    /// (`subtyping/check.rb:195`). The untyped short-circuit runs ahead of
    /// the cache lookup since it's O(1) and avoids polluting the cache with
    /// trivial entries.
    ///
    /// The memo is shared across files (and threads, ADR-0034), so it only
    /// takes values that do not depend on this query's context — unlike
    /// Steep, which stores whatever the in-flight search produced. The
    /// relation is monotone (sub-results combine only through `all`/`any`,
    /// never negated), which splits the results in two:
    /// - `true` derived by assuming an in-flight relation holds only if that
    ///   relation also ends `true`; it is kept per query until then
    /// - `false` derived from the depth cap may be `true` from a shallower
    ///   start; it is never stored
    ///
    /// Every other result equals what an empty search would compute.
    pub fn check(&self, sub: Ty, sup: Ty) -> bool {
        if sub.is_untyped() || sup.is_untyped() {
            return true;
        }

        // Fail closed past the depth cap, ahead of the cache lookup (mirrors
        // Steep's `check_type`, which returns `Failure(LoopAbort)` before
        // consulting `@cache`).
        if self.search.borrow().frames.len() >= SUBTYPE_ASSUMPTION_ABORT_LIMIT {
            if let Some(frame) = self.search.borrow_mut().frames.last_mut() {
                frame.aborted = true;
            }
            return false;
        }

        let cache_key = SubtypeCacheKey {
            sub,
            sup,
            self_bound: self.self_bound,
            type_variables_are_wildcards: self.type_variables_are_wildcards,
        };
        if let Some(cached) = self.env.cached_subtype_result(&cache_key) {
            return cached;
        }

        {
            let mut search = self.search.borrow_mut();
            let search = &mut *search;
            if let Some(&index) = search
                .provisional_rests_on
                .get(&cache_key)
                .or_else(|| search.in_flight.get(&cache_key))
            {
                search.rest_innermost_on(index);
                return true;
            }
            if search.aborted_false.contains(&cache_key) {
                if let Some(frame) = search.frames.last_mut() {
                    frame.aborted = true;
                }
                return false;
            }
            let index = search.frames.len();
            search.in_flight.insert(cache_key, index);
            let provisional_start = search.provisional.len();
            search.frames.push(Frame {
                key: cache_key,
                rests_on: usize::MAX,
                aborted: false,
                provisional_start,
            });
        }

        self.env.begin_capture();
        let result = self.check_uncached(sub, sup);
        self.finish_frame(result);
        result
    }

    /// Pop the innermost frame and settle what it (and everything that
    /// rested on it) may contribute to the shared memo.
    ///
    /// A stored relation carries the consultations captured over its
    /// frame. The relations established together with a frame were each
    /// proven inside it, and each reaches the frame's relation back
    /// through the assumption it rested on, so a fresh check of any of
    /// them would make the same consultations: they all carry the frame's.
    fn finish_frame(&self, result: bool) {
        let consultations = self.env.end_capture();
        let mut search = self.search.borrow_mut();
        let frame = search.frames.pop().expect("finish_frame without a frame");
        search.in_flight.remove(&frame.key);
        let index = search.frames.len();
        if index == 0 {
            search.aborted_false.clear();
        } else if !result && frame.aborted {
            search.aborted_false.insert(frame.key);
        }

        if !result {
            // A `false` holds in any context (monotonicity), but the `true`s
            // derived while this relation was assumed are now unfounded.
            search.drain_provisional_from(frame.provisional_start);
            drop(search);
            if frame.aborted {
                if let Some(parent) = self.search.borrow_mut().frames.last_mut() {
                    parent.aborted = true;
                }
            } else {
                self.env
                    .store_subtype_result(frame.key, false, consultations);
            }
            return;
        }

        if frame.rests_on >= index {
            // Rests on nothing outside itself: this relation and every result
            // assumed under it are established.
            let established = search.drain_provisional_from(frame.provisional_start);
            drop(search);
            for key in established {
                self.env
                    .store_subtype_result(key, true, consultations.clone());
            }
            self.env
                .store_subtype_result(frame.key, true, consultations);
            return;
        }

        let rests_on = frame.rests_on;
        let Search {
            provisional,
            provisional_rests_on,
            ..
        } = &mut *search;
        for key in &provisional[frame.provisional_start..] {
            if let Some(dep) = provisional_rests_on.get_mut(key)
                && *dep >= index
            {
                *dep = rests_on;
            }
        }
        provisional.push(frame.key);
        provisional_rests_on.insert(frame.key, rests_on);
        search.rest_innermost_on(rests_on);
    }

    fn check_uncached(&self, sub: Ty, sup: Ty) -> bool {
        // Expand type aliases before comparison
        let sub = definition_builder::expand_alias(self.env, sub);
        let sup = definition_builder::expand_alias(self.env, sup);

        // `check` short-circuits `untyped` before expansion; an alias that
        // expands to `untyped` (`type u = untyped`) only shows it here.
        if sub.is_untyped() || sup.is_untyped() {
            return true;
        }

        if sub == sup {
            return true;
        }

        let types = self.env.types();
        let sub_type = types.resolve(sub);
        let sup_type = types.resolve(sup);

        match sup_type {
            Type::Top => return true,
            Type::Void => return true,
            _ => {}
        }

        if matches!(sub_type, Type::Bottom) {
            return true;
        }

        let names = self.env.names();

        if let Type::Literal(lit) = sub_type {
            let widened = self
                .env
                .class_instance_type(*lit.class_typename(self.env.names().builtins()));
            if self.check(widened, sup) {
                return true;
            }
        }

        if matches!(sup_type, Type::Bool) && matches!(sub_type, Type::Literal(Literal::Bool(_))) {
            return true;
        }

        let builtins = names.builtins();
        if matches!(sup_type, Type::Bool)
            && let Type::ClassInstance { name, .. } = sub_type
            && builtins.is_bool_class(*name)
        {
            return true;
        }

        if matches!(sup_type, Type::Nil)
            && let Type::ClassInstance { name, .. } = sub_type
        {
            return *name == builtins.nil_class;
        }
        if matches!(sub_type, Type::Nil)
            && let Type::ClassInstance { name, .. } = sup_type
        {
            return *name == builtins.nil_class;
        }

        if let Type::Union(members) = sub_type {
            if members.iter().any(|m| m.is_untyped()) {
                return true;
            }
            return members.iter().all(|&m| self.check(m, sup));
        }

        if let Type::Optional(inner) = sub_type {
            if inner.is_untyped() {
                return true;
            }
            return self.check(*inner, sup) && self.check(Ty::NIL, sup);
        }

        // Intersection on the sup side: every member must be matched. Evaluated
        // before the sub-side rule so `A <: B & C` checks each conjunct even
        // when `A` is itself an intersection — the sub-side `any` rule would
        // otherwise short-circuit on a partial match.
        if let Type::Intersection(members) = sup_type {
            return members.iter().all(|&m| self.check(sub, m));
        }

        // Sup-side Union/Optional are decomposed before sub-side Intersection
        // for the same reason sup-side Intersection above runs before sub-side
        // Intersection: the sub-side `any` rule would otherwise short-circuit
        // on a partial match and miss the case where the whole intersection
        // matches a sup branch directly (`(A & B) <: (A & B)?` reduces to
        // `(A & B) <: (A & B)` only if Optional is peeled first).
        if let Type::Union(members) = sup_type {
            return members.iter().any(|&m| self.check(sub, m));
        }

        if let Type::Optional(inner) = sup_type {
            return self.check(sub, *inner) || self.check(sub, Ty::NIL);
        }

        // Intersection on the sub side: a value typed as `A & B` can be used
        // wherever either `A` or `B` fits.
        if let Type::Intersection(members) = sub_type {
            return members.iter().any(|&m| self.check(m, sup));
        }

        // SUB-side `self <: X` widening (mirrors Steep subtyping/check.rb:417).
        // In a self-bearing context (`self_bound` is `Some`), expand `self` to
        // the concrete bound and recurse. SUP-side `X <: self` gets no arm:
        // only `sub == sup` (handled above) passes, keeping `self` strict so
        // a concrete like `Box[untyped]` returned from a `-> self` body is
        // rejected. Placed AFTER sup-side union/optional/intersection
        // decomposition so `self <: self?` peels the optional first
        // (`self <: self` passes) instead of widening `self` to a concrete
        // that the strict SUP-side `self` then rejects — matching Steep, where
        // `sub_type.is_a?(Self)` is handled after the union/interface arms.
        // The `bound != sub` guard avoids self-recursion if the bound ever
        // resolves back to `self`.
        if matches!(sub_type, Type::SelfType)
            && let Some(bound) = self.self_bound
            && bound != sub
        {
            return self.check(bound, sup);
        }

        if let Type::Tuple(sub_members) = &sub_type {
            match &sup_type {
                Type::Tuple(sup_members) => {
                    if sub_members.len() != sup_members.len() {
                        return false;
                    }
                    return sub_members
                        .iter()
                        .zip(sup_members.iter())
                        .all(|(&s, &t)| self.check(s, t));
                }
                // Tuples widen to `Array[union-of-elements]` for any non-tuple sup.
                // This lets inheritance, interface conformance, and gradual typing
                // all reuse the existing ClassInstance machinery uniformly. An
                // interface sup gets a second chance below with `self` kept as
                // the tuple (Steep's `Any(widened, shape)` for a Tuple sub).
                _ => {
                    let array_ty = self.tuple_as_array(sub_members);
                    if self.check(array_ty, sup) {
                        return true;
                    }
                    if !matches!(sup_type, Type::Interface { .. }) {
                        return false;
                    }
                }
            }
        }

        if let Type::Record { fields: sub_fields } = &sub_type {
            match &sup_type {
                Type::Record { fields: sup_fields } => {
                    return self.check_record_structural(sub_fields, sup_fields);
                }
                // Same strategy as Tuple: widen to `Hash[Symbol, union]` and reuse
                // ClassInstance-based inheritance/interface/Hash-method logic.
                _ => {
                    let hash_ty = self.record_as_hash(sub_fields);
                    if self.check(hash_ty, sup) {
                        return true;
                    }
                    if !matches!(sup_type, Type::Interface { .. }) {
                        return false;
                    }
                }
            }
        }

        if let Type::Interface {
            name: iface_name,
            args: iface_args,
        } = &sup_type
        {
            return self.check_interface_conformance(sub, iface_name, iface_args);
        }

        if let (
            Type::ClassInstance {
                name: sub_name,
                args: sub_args,
            },
            Type::ClassInstance {
                name: sup_name,
                args: sup_args,
            },
        ) = (&sub_type, &sup_type)
        {
            let chain = self.env.instance_ancestors(sub_name);
            let ancestors = chain.apply(sub_args, self.env.types());
            for ancestor in &ancestors {
                if let Ancestor::Instance { name, args, source } = ancestor
                    && name == sup_name
                {
                    let actual_args = if *source == AncestorSource::SelfDecl {
                        sub_args
                    } else {
                        args
                    };
                    if self.check_type_args(sup_name, actual_args, sup_args) {
                        return true;
                    }
                }
            }
            return false;
        }

        // singleton(Sub) <: singleton(Sup) iff Sub's class hierarchy reaches Sup.
        // ClassSingleton carries no type args (Sorbet's `singleton(X)[T]` is ignored
        // by the parser), so no args comparison is needed.
        if let (Type::ClassSingleton { name: sub_name }, Type::ClassSingleton { name: sup_name }) =
            (&sub_type, &sup_type)
        {
            return self.check_inheritance(sub_name, sup_name);
        }

        // singleton(Sub) <: ClassInstance(Sup) — walk the linearized
        // singleton ancestors so the class/module routing (BasicObject
        // boundary flips to `::Class`, module singletons flip to
        // `::Module`) stays delegated to `AncestorBuilder` instead of
        // being re-encoded as well-known names here.
        if let (
            Type::ClassSingleton { name: sub_name },
            Type::ClassInstance {
                name: sup_name,
                args: sup_args,
            },
        ) = (&sub_type, &sup_type)
        {
            let chain = self.env.singleton_ancestors(sub_name);
            for ancestor in &chain.ancestors {
                if let Ancestor::Instance { name, args, .. } = ancestor
                    && name == sup_name
                {
                    return self.check_type_args(sup_name, args, sup_args);
                }
            }
            return false;
        }

        if let (Type::Proc { .. }, Type::ClassInstance { name: _, args: _ }) =
            (&sub_type, &sup_type)
        {
            let proc_instance = self
                .env
                .class_instance_type(self.env.names().builtins().proc);
            return self.check(proc_instance, sup);
        }

        if let (
            Type::Proc {
                type_: sub_ft,
                self_type: sub_self,
                block: sub_block,
            },
            Type::Proc {
                type_: sup_ft,
                self_type: sup_self,
                block: sup_block,
            },
        ) = (&sub_type, &sup_type)
        {
            return self.check_proc(sub_ft, sub_self, sub_block, sup_ft, sup_self, sup_block);
        }

        if matches!(sub_type, Type::TypeVariable { .. })
            || matches!(sup_type, Type::TypeVariable { .. })
        {
            return self.type_variables_are_wildcards;
        }

        false
    }

    /// Check Proc subtyping.
    /// - Return: covariant
    /// - Parameters: contravariant
    /// - `self_type`: contravariant (`super_self <: sub_self`), matching
    ///   Steep's `check_self_type_binding`
    /// - `block`: when both sides have block info, presence/required/function
    ///   must match; when sub has no block info (`None`) against a sup that
    ///   has a block, we treat it as permissive — crema does not yet infer
    ///   block types for lambdas with `&block` params, so `block: None` means
    ///   "inference gap", not "no block". This is intentionally wide.
    /// - If either function is `UntypedFunction`, parameter checks are skipped
    fn check_proc(
        &self,
        sub_ft: &FunctionType,
        sub_self: &Option<Ty>,
        sub_block: &Option<Block>,
        sup_ft: &FunctionType,
        sup_self: &Option<Ty>,
        sup_block: &Option<Block>,
    ) -> bool {
        if !self.check(sub_ft.return_type(), sup_ft.return_type()) {
            return false;
        }

        if !self.check_self_type_binding(sub_self, sup_self) {
            return false;
        }

        if !self.check_optional_block(sub_block, sup_block) {
            return false;
        }

        match (sub_ft, sup_ft) {
            (FunctionType::Untyped(_), _) | (_, FunctionType::Untyped(_)) => true,
            (FunctionType::Typed(sub_f), FunctionType::Typed(sup_f)) => {
                self.check_proc_params(sub_f, sup_f)
            }
        }
    }

    /// Contravariant subtype check for an optional `[self: T]` binding.
    /// A block's `self_type` is supplied by the caller when invoking the
    /// block, so it behaves like a function parameter — the substitutability
    /// direction flips: `Some(super_self) <: Some(sub_self)`.
    fn check_self_type_binding(&self, sub_self: &Option<Ty>, sup_self: &Option<Ty>) -> bool {
        match (sub_self, sup_self) {
            (None, None) => true,
            (Some(sub), Some(sup)) => self.check(*sup, *sub),
            _ => false,
        }
    }

    fn check_optional_block(&self, sub_block: &Option<Block>, sup_block: &Option<Block>) -> bool {
        match (sub_block, sup_block) {
            (None, None) => true,
            (Some(sub), Some(sup)) => {
                if sub.required != sup.required {
                    return false;
                }
                if !self.check_self_type_binding(&sub.self_type, &sup.self_type) {
                    return false;
                }
                match (&sub.type_, &sup.type_) {
                    (FunctionType::Untyped(_), _) | (_, FunctionType::Untyped(_)) => true,
                    (FunctionType::Typed(sub_f), FunctionType::Typed(sup_f)) => {
                        self.check(sub_f.return_type, sup_f.return_type)
                            && self.check_proc_params(sub_f, sup_f)
                    }
                }
            }
            // sub has no block info: treat as permissive. `block: None` means
            // crema could not infer the block type (e.g. unannotated lambda
            // with `&block` param), not that the proc has no block.
            (None, Some(_)) => true,
            (Some(_), None) => false,
        }
    }

    /// Contravariant parameter check for Proc.
    /// Shapes must match: same number of required/optional/trailing positionals,
    /// same keyword names, and consistent rest slots. Types are compared with
    /// the relation reversed (sup param <: sub param).
    fn check_proc_params(
        &self,
        sub_f: &crate::types::Function,
        sup_f: &crate::types::Function,
    ) -> bool {
        if sub_f.required_positionals.len() != sup_f.required_positionals.len()
            || sub_f.optional_positionals.len() != sup_f.optional_positionals.len()
            || sub_f.trailing_positionals.len() != sup_f.trailing_positionals.len()
            || sub_f.rest_positional.is_some() != sup_f.rest_positional.is_some()
            || sub_f.rest_keyword.is_some() != sup_f.rest_keyword.is_some()
        {
            return false;
        }

        for (sub_p, sup_p) in sub_f
            .required_positionals
            .iter()
            .zip(&sup_f.required_positionals)
        {
            if !self.check(*sup_p, *sub_p) {
                return false;
            }
        }
        for (sub_p, sup_p) in sub_f
            .optional_positionals
            .iter()
            .zip(&sup_f.optional_positionals)
        {
            if !self.check(*sup_p, *sub_p) {
                return false;
            }
        }
        for (sub_p, sup_p) in sub_f
            .trailing_positionals
            .iter()
            .zip(&sup_f.trailing_positionals)
        {
            if !self.check(*sup_p, *sub_p) {
                return false;
            }
        }
        if let (Some(sub_rest), Some(sup_rest)) = (sub_f.rest_positional, sup_f.rest_positional)
            && !self.check(sup_rest, sub_rest)
        {
            return false;
        }

        let sub_req_names: FxHashSet<&str> = sub_f
            .required_keywords
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        let sup_req_names: FxHashSet<&str> = sup_f
            .required_keywords
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        let sub_opt_names: FxHashSet<&str> = sub_f
            .optional_keywords
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        let sup_opt_names: FxHashSet<&str> = sup_f
            .optional_keywords
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();

        if sub_req_names != sup_req_names || sub_opt_names != sup_opt_names {
            return false;
        }

        for (name, sub_ty) in &sub_f.required_keywords {
            let sup_ty = sup_f
                .required_keywords
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, t)| *t)
                .unwrap();
            if !self.check(sup_ty, *sub_ty) {
                return false;
            }
        }
        for (name, sub_ty) in &sub_f.optional_keywords {
            let sup_ty = sup_f
                .optional_keywords
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, t)| *t)
                .unwrap();
            if !self.check(sup_ty, *sub_ty) {
                return false;
            }
        }
        if let (Some(sub_rest), Some(sup_rest)) = (sub_f.rest_keyword, sup_f.rest_keyword)
            && !self.check(sup_rest, sub_rest)
        {
            return false;
        }

        true
    }

    /// Check that each inferred/applied type argument in `bindings` satisfies
    /// the upper/lower bound of its corresponding `TypeParam`. Returns one
    /// `BoundViolation` per violated bound (upper and lower are separate
    /// entries when both fail on the same param).
    ///
    /// Bounds that reference other type parameters (e.g. `[U, T < Array[U]]`)
    /// are substituted against the same `bindings` before comparison —
    /// otherwise the free `TypeVariable` would behave as a wildcard and
    /// silently accept any concrete argument.
    ///
    /// `untyped` always satisfies every bound (gradual typing — matches Steep
    /// and the rest of crema's subtype relation). Params without an entry in
    /// `bindings` are skipped: inference left them unbound, which is a
    /// separate issue handled upstream.
    pub fn check_type_arg_bounds(
        &self,
        params: &[crate::type_param::TypeParam],
        bindings: &FxHashMap<TypeVarKey, Ty>,
    ) -> Vec<BoundViolation> {
        let mut violations = Vec::new();
        let subst = crate::substitution::Substitution::from_mapping(bindings.clone());
        let types = self.env.types();
        for param in params {
            let Some(&actual) = bindings.get(&param.name) else {
                // param.name is now a TypeVarKey; the bindings map uses
                // the same key, so the lookup works without conversion.
                continue;
            };
            if actual.is_untyped() {
                continue;
            }
            if let Some(upper) = param.upper_bound {
                let upper_subst = subst.apply(upper, types);
                if !self.check(actual, upper_subst) {
                    violations.push(BoundViolation {
                        param_name: param.name.raw,
                        bound_kind: crate::diagnostic::BoundKind::Upper,
                        bound: upper_subst,
                        actual,
                    });
                }
            }
            if let Some(lower) = param.lower_bound {
                let lower_subst = subst.apply(lower, types);
                if !self.check(lower_subst, actual) {
                    violations.push(BoundViolation {
                        param_name: param.name.raw,
                        bound_kind: crate::diagnostic::BoundKind::Lower,
                        bound: lower_subst,
                        actual,
                    });
                }
            }
        }
        violations
    }

    /// Check type arguments using the declared variance of `class_name`'s
    /// type parameters. Empty args on either side are treated as compatible
    /// (gradual typing). If the class's type params are unknown, fall back
    /// to covariant comparison to preserve pre-variance behavior.
    fn check_type_args(&self, class_name: &TypeName, sub_args: &[Ty], sup_args: &[Ty]) -> bool {
        use crate::type_param::Variance;

        if sub_args.is_empty() || sup_args.is_empty() {
            return true;
        }

        let params = self.env.class_type_params_by_type_name(class_name);

        for (i, (&sub_arg, &sup_arg)) in sub_args.iter().zip(sup_args.iter()).enumerate() {
            let variance = params
                .and_then(|ps| ps.get(i))
                .map(|p| p.variance)
                .unwrap_or(Variance::Covariant);
            match variance {
                Variance::Covariant => {
                    if !self.check(sub_arg, sup_arg) {
                        return false;
                    }
                }
                Variance::Contravariant => {
                    if !self.check(sup_arg, sub_arg) {
                        return false;
                    }
                }
                Variance::Invariant => {
                    if !self.check(sub_arg, sup_arg) || !self.check(sup_arg, sub_arg) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a type structurally conforms to an interface.
    /// The sub type must have all methods that the interface declares,
    /// with compatible signatures (return type covariance, parameter compatibility).
    fn check_interface_conformance(
        &self,
        sub: Ty,
        iface_name: &TypeName,
        iface_args: &[Ty],
    ) -> bool {
        let Some(required_methods) = self.env.interface_method_names_by_type_name(iface_name)
        else {
            // Unknown interface — treat as compatible (gradual typing)
            return true;
        };

        let types = self.env.types();
        let sub_type = types.resolve(sub);

        // Whether the sub side is a class itself (`singleton(C)`) rather than
        // an instance. Singleton conformance walks the class's own singleton
        // methods (`def self.foo`) instead of the instance-method chain.
        // `sub_args` carries the receiver's own type-arg slice (e.g.
        // `Foo[String]` -> `[::String]`); empty for singleton receivers.
        let (sub_class_name, sub_args, is_singleton) = match &sub_type {
            Type::ClassInstance { name, args } => (*name, args.clone(), false),
            Type::Interface { name, args } => (*name, args.clone(), false),
            Type::ClassSingleton { name } => (*name, Vec::new(), true),
            // `nil` is a singleton value of `::NilClass`; walk the
            // NilClass method chain so `T?` = `T | nil` satisfies any
            // interface whose methods live on Kernel / Object. Without
            // this arm the sub-side Union decomposition at `check()`
            // fails whenever a member reduces to `Type::Nil` (Steep
            // parity: `Interface::Builder#shape` bridges `AST::Types::
            // Nil` to `NilClass` at `steep/lib/steep/interface/
            // builder.rb`).
            Type::Nil => (self.env.names().builtins().nil_class, Vec::new(), false),
            // Structural types take their methods from the class they widen
            // to, while `sub` (the structural type itself) stays the `self`
            // those methods are substituted with. Steep `raw_shape` builds
            // the same split: `object_shape(Array)` with `self_type: tuple`.
            Type::Literal(lit) => {
                let widened = self
                    .env
                    .class_instance_type(*lit.class_typename(self.env.names().builtins()));
                let Some((name, args)) = class_instance_parts(types, widened) else {
                    return false;
                };
                (name, args, false)
            }
            Type::Tuple(members) => {
                let Some((name, args)) = class_instance_parts(types, self.tuple_as_array(members))
                else {
                    return false;
                };
                (name, args, false)
            }
            Type::Record { fields } => {
                let Some((name, args)) = class_instance_parts(types, self.record_as_hash(fields))
                else {
                    return false;
                };
                (name, args, false)
            }
            _ => return false,
        };

        // Rewrite `Type::SelfType` (and `Type::InstanceType` on the singleton
        // path) inside the sub method's signatures to concrete receiver types
        // before comparing. `SubtypeChecker::check` has no arm for either, so
        // without these substitutions any contravariant/covariant check against
        // `self` or `instance` falls through to `false` and conformance fails
        // for shapes like `Exception#exception: (?self) -> self` or
        // `Exception.exception: (?...) -> instance`. Mirrors the substitution
        // `DefinitionBuilder::build_singleton` applies to `#initialize` when
        // folding `Bases::Self`.
        let singleton_instance_ty = if is_singleton {
            Some(types.intern(Type::ClassInstance {
                name: sub_class_name,
                args: Vec::new(),
            }))
        } else {
            None
        };

        for method_name in &required_methods {
            let method_name = *method_name;
            // Bindings-aware lookup. The args-free
            // `lookup_instance_method_by_type_name` returns method defs whose
            // type variables (class params on the receiver or any ancestor in
            // the chain) stay free; `SubtypeChecker::check` then treats them
            // as wildcards (`type_variables_are_wildcards: true`) and any
            // mismatch falls through silently. The `_with_args` / singleton
            // variants fold the receiver's `args` and per-ancestor bindings
            // before returning, and the bindings tuple they hand back covers
            // the leaf ancestor whose own defs are not yet substituted (see
            // `lookup_instance_method_with_args` doc).
            let sub_lookup = if is_singleton {
                self.env
                    .lookup_singleton_method(&sub_class_name, method_name)
            } else {
                self.env
                    .lookup_instance_method_with_args(&sub_class_name, &sub_args, method_name)
            };
            let Some((sub_method, sub_bindings)) = sub_lookup else {
                return false;
            };
            let Some(iface_method) = definition_builder::lookup_instance_method_by_type_name(
                self.env,
                *iface_name,
                method_name,
            ) else {
                return false;
            };

            let mut sub_subst = Substitution::from_mapping(sub_bindings).with_self_type(sub);
            if let Some(instance_ty) = singleton_instance_ty {
                sub_subst = sub_subst.with_instance_type(instance_ty);
            }

            // sup interface side: a generic interface like
            // `_Get[T]; def get: () -> T` would leave `T` free in
            // `iface_method.defs`, and the wildcard arm in
            // `SubtypeChecker::check` would accept any sub return type.
            // Build the bindings directly by zipping the interface's own
            // class-level params with `iface_args`; `lookup_instance_method_*`
            // does not work here because `one_instance_ancestors_with` only
            // populates params for `class_decls` entries, leaving the
            // interface self-ancestor with empty params (and hence empty
            // ancestor_bindings).
            let iface_bindings: FxHashMap<TypeVarKey, Ty> = self
                .env
                .class_type_params_by_type_name(iface_name)
                .map(|params| {
                    params
                        .iter()
                        .zip(iface_args.iter())
                        .map(|(p, &ty)| (p.name.clone(), ty))
                        .collect()
                })
                .unwrap_or_default();
            // The interface's own `self` binds to the interface type itself
            // (Steep parity: `Interface::Builder#interface_subst` and the
            // `super_type.is_a?(Interface)` arm in `subtyping/check.rb` both
            // build the sup-side shape with `self_type: <the interface
            // type>`, not the candidate receiver). This lets an inherited
            // method like `Sub#dup: () -> Base` (no literal `self`, `Base`
            // via ancestry) satisfy `interface _Dupable; def dup: () -> self;
            // end`: the return-type check recurses into `check(Base,
            // _Dupable)`, and if `Base` also only has `dup: () -> Base`, that
            // recursion re-derives the exact relation already in
            // the in-flight relations (pushed by the outer `check()` call for
            // `Sub <: _Dupable`), so the coinductive assumption closes the
            // loop — mirroring Steep's `assumptions.member?(relation)`
            // short-circuit. Binding directly to `sub` instead (the naive
            // fix) only covers methods that literally return `self`; it
            // rejects the inherited-ancestor case because `check(Base, Sub)`
            // is a false ancestor-direction check, not a fresh interface
            // relation the assumption machinery can catch.
            let iface_ty = types.intern(Type::Interface {
                name: *iface_name,
                args: iface_args.to_vec(),
            });
            let iface_subst =
                Substitution::from_mapping(iface_bindings.clone()).with_self_type(iface_ty);

            // Positional params (and, since the block-return fix, block
            // *return* type specifically — see `check_block_params`) get a
            // *second* substitution that binds the interface's own `self`
            // to `sub` (the candidate receiver) instead of `iface_ty`.
            // Contravariance flips `self`'s position: `def ==: (self) ->
            // bool` in argument position means "whatever the caller's own
            // type turns out to be", which is `sub` from the checking
            // candidate's point of view, not the interface type itself.
            // Block return position has the same shape. Block *param*
            // position does NOT use this substitution — it lands the
            // interface side on the SUP side of `check()` either way, but
            // swapping it to `self→sub` there replaces the existing
            // structural SUP-side Interface arm with a narrower nominal
            // check and rejects real-Steep-accepted programs (verified
            // 2026-07-19; see `check_block_params`'s doc comment). Verified
            // against real Steep (`bundle exec steep check`
            // + a `TracePoint` trace of `check_type0`, 2026-07-17): Steep
            // resolves this shape not through a dedicated structural rule but
            // through its coinductive `same_type?` assumption-flip
            // short-circuit (`subtyping/check.rb:666`), whose result is then
            // written into the *unscoped* subtype cache
            // (`subtyping/check.rb:195-207`). That leaks: checking
            // `Sink.new.take_eq(C.new)` first caches `_Eq <: C => true`, and
            // a later, unrelated `Sink.new.take_c(some_Eq_value)` in the same
            // file then wrongly passes off that stale cache entry — order
            // dependent and unsound (reproduced directly against a local
            // checkout of <https://github.com/soutaro/steep>, not inferred
            // from source reading). Substituting `self` to `sub` up front avoids the
            // interface-as-sub shape entirely for this position, so no
            // coinductive/caching trick is needed and the false-accept can't
            // happen. Return position keeps the `iface_ty` substitution
            // above unchanged — it depends on the coinductive assumption
            // closing over `Sub <: interface` (see the comment above), which
            // this substitution does not touch.
            let iface_subst_for_params =
                Substitution::from_mapping(iface_bindings).with_self_type(sub);

            let substituted_sub_defs =
                definition_builder::substitute_method_defs(&sub_method.defs, &sub_subst, types);
            let sub_method_substituted =
                Method::from_defs(substituted_sub_defs, sub_method.accessibility);

            let substituted_iface_defs =
                definition_builder::substitute_method_defs(&iface_method.defs, &iface_subst, types);
            let iface_method_substituted =
                Method::from_defs(substituted_iface_defs, iface_method.accessibility);

            let substituted_iface_defs_for_params = definition_builder::substitute_method_defs(
                &iface_method.defs,
                &iface_subst_for_params,
                types,
            );
            let iface_method_substituted_for_params = Method::from_defs(
                substituted_iface_defs_for_params,
                iface_method.accessibility,
            );

            if !self.check_method_compatibility(
                &sub_method_substituted,
                &iface_method_substituted,
                &iface_method_substituted_for_params,
            ) {
                return false;
            }
        }

        true
    }

    /// Check if the sub method's overloads are compatible with the interface method's overloads.
    /// At least one sub overload must be compatible with each interface overload.
    ///
    /// `iface_method` and `iface_method_for_params` are the same overloads
    /// substituted two different ways (self→iface_ty vs. self→sub, see
    /// `check_interface_conformance`); `method_types()` walks `defs` in the
    /// same declaration order for both, so the zip pairs up matching
    /// overloads.
    fn check_method_compatibility(
        &self,
        sub_method: &Method,
        iface_method: &Method,
        iface_method_for_params: &Method,
    ) -> bool {
        for (iface_ol, iface_ol_for_params) in iface_method
            .method_types()
            .zip(iface_method_for_params.method_types())
        {
            let compatible = sub_method.method_types().any(|sub_ol| {
                self.check_overload_compatibility(sub_ol, iface_ol, iface_ol_for_params)
            });
            if !compatible {
                return false;
            }
        }
        true
    }

    /// Check if a single sub overload is compatible with a single interface overload.
    ///
    /// The interface defines the contract callers follow.
    /// The sub method must accept any call the interface allows.
    ///
    /// - Block: checked **first** — a block-shape mismatch is a cheap,
    ///   non-recursive check that can reject a self-wrapping overload
    ///   (e.g. `Enumerable#each_entry: () -> Enumerator[E, self]`) before
    ///   the return-type check below ever recurses into it. Steep orders
    ///   the same way: `Subtyping::Check#check_method_type` resolves
    ///   `expand_block_given` before calling `check_function` (which
    ///   covers params/return), short-circuiting entirely on mismatch
    ///   (`subtyping/check.rb:868-884`). Checking return type first (the
    ///   prior order here) let a block-less overload's self-referential
    ///   return type recurse indefinitely before ever reaching the block
    ///   check that would have rejected it on shape alone. Within the
    ///   block itself: relation reversed, same rules as params, but block
    ///   *param* position is checked against `iface_ol` (self→iface_ty —
    ///   this lands the interface side on the SUP side of `check()`,
    ///   where the existing SUP-side Interface arm handles it
    ///   structurally) while block *return* position is checked against
    ///   `iface_ol_for_params` (self→sub — it lands the interface side on
    ///   the SUB side instead, same shape as Method params, and has no
    ///   arm without the substitution). `self→sub` is not correct for
    ///   block param position — see `check_block_params`
    /// - Return type: **covariant** (sub_return <: iface_return)
    /// - Parameters: **contravariant** (iface_param <: sub_param), skipped
    ///   entirely when either side is `(?) -> T` (untyped function), checked
    ///   against `iface_ol_for_params` (self→sub) rather than `iface_ol`
    ///   (self→iface_ty) — see `check_interface_conformance`
    fn check_overload_compatibility(
        &self,
        sub_ol: &crate::types::MethodType,
        iface_ol: &crate::types::MethodType,
        iface_ol_for_params: &crate::types::MethodType,
    ) -> bool {
        if !self.check_block_params(sub_ol, iface_ol, iface_ol_for_params) {
            return false;
        }

        if !self.check(sub_ol.return_type(), iface_ol.return_type()) {
            return false;
        }

        // `(?) -> T` on either side: params are not checked, only the
        // return type (and block) — Steep `check_function` compares
        // params only when both sides have them; same rule as
        // `check_proc`. Not `max_arity() == None`, which a typed rest
        // positional also returns.
        if sub_ol.is_untyped_function() || iface_ol.is_untyped_function() {
            return true;
        }

        if !self.check_positional_params(sub_ol, iface_ol_for_params) {
            return false;
        }

        if !self.check_keyword_params(sub_ol, iface_ol) {
            return false;
        }

        true
    }

    /// Check positional parameter compatibility.
    ///
    /// Interface callers provide between `iface.required.len()` and
    /// `iface.required.len() + iface.optional.len()` args (unlimited if rest).
    /// Sub must accept that entire range.
    fn check_positional_params(
        &self,
        sub_ol: &crate::types::MethodType,
        iface_ol: &crate::types::MethodType,
    ) -> bool {
        let sub_min = sub_ol.min_arity();
        let sub_max = sub_ol.max_arity();

        let iface_min = iface_ol.min_arity();
        let iface_max = iface_ol.max_arity();

        if sub_min > iface_min {
            return false;
        }

        match (sub_max, iface_max) {
            (Some(sm), Some(im)) if sm < im => return false,
            (Some(_), None) => return false,
            _ => {}
        }

        // When iface has rest (max=None), check up to iface's explicit params only
        let check_count =
            iface_max.unwrap_or(iface_ol.min_arity() + iface_ol.optional_positionals().len());

        for i in 0..check_count {
            let iface_param = positional_param_at(iface_ol, i);
            let sub_param = positional_param_at(sub_ol, i);

            match (sub_param, iface_param) {
                (Some(sp), Some(ip)) => {
                    if !self.check(ip, sp) {
                        return false;
                    }
                }
                (None, Some(_)) => return false,
                _ => {}
            }
        }

        if let (Some(iface_rest), Some(sub_rest)) =
            (iface_ol.rest_positional(), sub_ol.rest_positional())
            && !self.check(iface_rest, sub_rest)
        {
            return false;
        }

        true
    }

    /// Check keyword parameter compatibility.
    ///
    /// - Every keyword the interface may provide (required + optional), sub must accept
    /// - Every keyword sub requires must be guaranteed by the interface (in iface's required)
    /// - Types: contravariant (iface_kw_type <: sub_kw_type)
    fn check_keyword_params(
        &self,
        sub_ol: &crate::types::MethodType,
        iface_ol: &crate::types::MethodType,
    ) -> bool {
        for (name, iface_ty) in iface_ol.required_keywords() {
            if let Some(&sub_ty) = find_keyword(sub_ol, name) {
                if !self.check(*iface_ty, sub_ty) {
                    return false;
                }
            } else if sub_ol.rest_keyword().is_some() {
            } else {
                return false;
            }
        }

        for (name, iface_ty) in iface_ol.optional_keywords() {
            if let Some(&sub_ty) = find_keyword(sub_ol, name) {
                if !self.check(*iface_ty, sub_ty) {
                    return false;
                }
            } else if sub_ol.rest_keyword().is_some() {
            } else {
                return false;
            }
        }

        for (name, _) in sub_ol.required_keywords() {
            let in_iface_required = iface_ol.required_keywords().iter().any(|(n, _)| n == name);
            if !in_iface_required && iface_ol.rest_keyword().is_none() {
                return false;
            }
        }

        if let (Some(iface_rest), Some(sub_rest)) = (iface_ol.rest_keyword(), sub_ol.rest_keyword())
            && !self.check(iface_rest, sub_rest)
        {
            return false;
        }

        true
    }

    /// Check block parameter compatibility.
    ///
    /// If iface has a required block, sub must accept a block.
    /// If sub has a required block, iface must provide one.
    /// Block types are checked with the relation reversed (contravariant wrapper).
    ///
    /// Takes both `iface_ol` (self→iface_ty) and `iface_ol_for_params`
    /// (self→sub). Block param position and `self_type_binding` keep using
    /// `iface_ol`, matching pre-existing behavior: they land the interface
    /// side on the SUP side of `check()`, where the existing SUP-side
    /// Interface arm resolves conformance structurally (e.g. an ancestor
    /// class whose block-param type independently satisfies the interface,
    /// not just the exact substituted type — swapping this to `self→sub`
    /// was tried and rejects real-Steep-accepted programs because it
    /// replaces that structural check with a narrower nominal one, verified
    /// against real Steep, 2026-07-19). Block *return* position uses
    /// `iface_ol_for_params` instead, because it lands the interface side
    /// on the SUB side (same shape as Method positional params, see
    /// `check_overload_compatibility`), which has no `self→iface_ty` arm
    /// and false-rejects without the substitution.
    fn check_block_params(
        &self,
        sub_ol: &crate::types::MethodType,
        iface_ol: &crate::types::MethodType,
        iface_ol_for_params: &crate::types::MethodType,
    ) -> bool {
        match (&sub_ol.block, &iface_ol.block) {
            (None, None) => true,
            (Some(_), None) => sub_ol.block.as_ref().is_none_or(|b| !b.required),
            (None, Some(iface_block)) => !iface_block.required,
            (Some(sub_block), Some(iface_block)) => {
                // `iface_ol_for_params` is `iface_ol`'s defs re-substituted
                // with a different `Substitution`, not re-derived from a
                // different method — its block presence mirrors `iface_ol`'s.
                let iface_block_for_params = iface_ol_for_params
                    .block
                    .as_ref()
                    .expect("iface_ol_for_params has a block whenever iface_ol does");

                if !self.check(
                    iface_block_for_params.return_type(),
                    sub_block.return_type(),
                ) {
                    return false;
                }

                for (&sub_bp, &iface_bp) in
                    sub_block.params().iter().zip(iface_block.params().iter())
                {
                    if !self.check(sub_bp, iface_bp) {
                        return false;
                    }
                }

                self.check_self_type_binding(&sub_block.self_type, &iface_block.self_type)
            }
        }
    }

    /// Widen a tuple's element list to the covariant element type for `Array[T]`:
    /// - `[]` → `bot` (every `Array[T]` accepts an empty tuple)
    /// - `[A]` → `A`
    /// - `[A, B]` → `A | B`
    fn tuple_element_union(&self, members: &[Ty]) -> Ty {
        if members.is_empty() {
            Ty::BOTTOM
        } else if members.len() == 1 {
            members[0]
        } else {
            self.env.types().intern(Type::Union(members.to_vec()))
        }
    }

    /// Build `Array[element_union]` so tuple subtyping can reuse ClassInstance logic.
    fn tuple_as_array(&self, members: &[Ty]) -> Ty {
        let types = self.env.types();
        let array_name = self.env.names().builtins().array;
        let elem = self.tuple_element_union(members);
        types.intern(Type::ClassInstance {
            name: array_name,
            args: vec![elem],
        })
    }

    /// Structural record subtyping.
    ///
    /// - `sup.required k: T` → sub must have required `k: S` with `S <: T`
    /// - `sup.optional ?k: T` → if sub has `k: S`, then `S <: T`; sub may omit `k`
    /// - sub may have extra fields not in sup (structural openness)
    fn check_record_structural(
        &self,
        sub_fields: &[(RecordKey, Ty, bool)],
        sup_fields: &[(RecordKey, Ty, bool)],
    ) -> bool {
        for (sup_key, sup_ty, sup_required) in sup_fields {
            match sub_fields.iter().find(|(k, _, _)| k == sup_key) {
                Some((_, sub_ty, sub_required)) => {
                    if !self.check(*sub_ty, *sup_ty) {
                        return false;
                    }
                    if *sup_required && !sub_required {
                        return false;
                    }
                }
                None => {
                    if *sup_required {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Widen a record to `Hash[key_class_union, value_union]` so record-vs-
    /// non-record subtyping can reuse ClassInstance logic (Hash#methods, Hash
    /// inheritance). The key union is built from each field's `RecordKey`
    /// widened class (Symbol→::Symbol, String→::String, …). An empty record
    /// becomes `Hash[bot, bot]`, a subtype of any `Hash[K, V]`.
    fn record_as_hash(&self, fields: &[(RecordKey, Ty, bool)]) -> Ty {
        let types = self.env.types();
        let key_ty = record_key_union(fields, self.env);
        let field_tys: Vec<Ty> = fields.iter().map(|(_, ty, _)| *ty).collect();
        let value_ty = if field_tys.is_empty() {
            Ty::BOTTOM
        } else if field_tys.len() == 1 {
            field_tys[0]
        } else {
            types.intern(Type::Union(field_tys))
        };
        let hash_name = self.env.names().builtins().hash;
        types.intern(Type::ClassInstance {
            name: hash_name,
            args: vec![key_ty, value_ty],
        })
    }

    fn check_inheritance(&self, sub_name: &TypeName, sup_name: &TypeName) -> bool {
        let mut current = *sub_name;
        let mut visited = FxHashSet::default();
        loop {
            if current == *sup_name {
                return true;
            }
            if !visited.insert(current) {
                return false;
            }
            match self.env.superclass_type_name(&current) {
                Some(parent) => current = parent,
                None => return false,
            }
        }
    }
}

/// `(name, args)` of a widened structural type, which is always a
/// `ClassInstance` (`Array[...]` / `Hash[...]` / a literal's class).
fn class_instance_parts(types: &crate::types::TypeTable, ty: Ty) -> Option<(TypeName, Vec<Ty>)> {
    match types.resolve(ty) {
        Type::ClassInstance { name, args } => Some((*name, args.clone())),
        _ => None,
    }
}

/// Get the positional parameter type at a given index,
/// walking required → optional → rest → trailing.
///
/// Trailing positionals are anchored to the END of the argument list,
/// but for type-level comparison we lay them out after rest in the index space:
/// [required...] [optional...] [rest...] [trailing...]
fn positional_param_at(ol: &crate::types::MethodType, index: usize) -> Option<Ty> {
    if index < ol.required_positionals().len() {
        Some(ol.required_positionals()[index])
    } else {
        let after_req = index - ol.required_positionals().len();
        if after_req < ol.optional_positionals().len() {
            Some(ol.optional_positionals()[after_req])
        } else if ol.rest_positional().is_some() && ol.trailing_positionals().is_empty() {
            ol.rest_positional()
        } else if ol.rest_positional().is_some() {
            // With rest + trailing: rest absorbs middle, trailing are at the end.
            // In index space: after optional, rest fills until trailing starts.
            // We can't know the exact boundary statically, so return rest for
            // middle indices. This is used for type checking, not arity.
            ol.rest_positional()
        } else {
            None
        }
    }
}

/// Find a keyword parameter by name in a method's required or optional keywords.
fn find_keyword<'a>(ol: &'a crate::types::MethodType, name: &str) -> Option<&'a Ty> {
    ol.required_keywords()
        .iter()
        .chain(ol.optional_keywords().iter())
        .find(|(n, _)| n == name)
        .map(|(_, ty)| ty)
}

/// Build the key-side type for widening a record to a `Hash[K, V]`.
///
/// Collects the class-widened key for each field (`RecordKey::widen_class_name`),
/// dedups so `Hash[Symbol, V]` stays a `ClassInstance` rather than a singleton
/// union, and collapses an empty record to `BOTTOM`.
pub(crate) fn record_key_union(fields: &[(RecordKey, Ty, bool)], env: ConsultationView) -> Ty {
    let names = env.names();
    let types = env.types();
    if fields.is_empty() {
        return Ty::BOTTOM;
    }
    let mut key_tys: Vec<Ty> = Vec::new();
    for (key, _, _) in fields {
        let class_name = key.widen_class_name();
        let ty = types.class_instance(names.parse_type_name(class_name));
        if !key_tys.contains(&ty) {
            key_tys.push(ty);
        }
    }
    if key_tys.len() == 1 {
        key_tys[0]
    } else {
        types.intern(Type::Union(key_tys))
    }
}

