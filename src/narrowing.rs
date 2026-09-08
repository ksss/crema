//! Pure set-operation primitives for flow-sensitive narrowing.
//!
//! `narrow(ty, target, env)` returns the intersection of `ty` with `target`.
//! `subtract(ty, target, env)` returns `ty` with `target` removed.
//!
//! Both return `Ty::BOTTOM` when the resulting set is empty. Whether the
//! caller treats `Bottom` as an `UnreachableBranch` diagnostic or folds it
//! into a runtime fallback (e.g. binding the variable to `untyped` to avoid
//! over-propagation) is a checker-layer policy, not the concern of this
//! module. See ADR-0022 §(2).
//!
//! `target` is a **single type** — `Union`/`Intersection` decomposition for
//! compound conditions belongs to the upstream env-algebra (ADR-0022 §(1)).
//!
//! Member selection uses a two-tier strategy: an intern-id equality check
//! first (the common case for `is_a?(C)` where `C` already appears in the
//! union), falling back to `SubtypeChecker::check` only on miss. This keeps
//! the hot path at integer comparisons and limits the linearisation walk to
//! the genuinely cross-hierarchy cases (`is_a?(Numeric)` over `Integer`).

use crate::definition_builder::{ConsultationView, expand_alias};
use crate::subtyping::SubtypeChecker;
use crate::types::{Literal, Ty, Type, union_of, union_of_many};

/// Return the intersection of `ty` with `target`.
///
/// Empty intersection → `Ty::BOTTOM`. `Untyped` on the input narrows to
/// `target`; `Untyped` on the target acts as identity (everything matches).
pub fn narrow(ty: Ty, target: Ty, env: ConsultationView) -> Ty {
    let ty = expand_alias(env, ty);
    let target = expand_alias(env, target);
    if ty == target {
        return ty;
    }
    if ty == Ty::BOTTOM || target == Ty::BOTTOM {
        return Ty::BOTTOM;
    }
    if target.is_untyped() {
        return ty;
    }
    if ty.is_untyped() {
        return target;
    }
    if target == Ty::TOP {
        return ty;
    }
    if ty == Ty::TOP {
        return target;
    }

    let types = env.types();
    if let (Type::Bool, Type::Literal(Literal::Bool(_))) =
        (types.resolve(ty), types.resolve(target))
    {
        return target;
    }
    match types.resolve(ty) {
        Type::Union(members) => {
            let kept: Vec<Ty> = members
                .iter()
                .copied()
                .map(|m| narrow(m, target, env))
                .collect();
            union_of_many(&kept, env.types())
        }
        Type::Optional(inner) => {
            // `Optional(T)` ≡ `T | nil`.
            let lhs = narrow(*inner, target, env);
            let rhs = narrow(Ty::NIL, target, env);
            union_of(lhs, rhs, env.types())
        }
        _ => {
            let checker = SubtypeChecker::new(env);
            if checker.check(ty, target) {
                ty
            } else if checker.check(target, ty) {
                // `ty` is broader than `target`: the intersection is the
                // smaller side. Returning `ty` here would violate the
                // "narrow = intersection" contract — Steep matches this:
                // `x: Object; is_a?(String)` narrows `x` to `String`.
                target
            } else {
                Ty::BOTTOM
            }
        }
    }
}

/// Return `ty` with `target` removed.
///
/// Empty difference → `Ty::BOTTOM`. `Untyped` on the input is left intact
/// (`untyped without T` has no representation in RBS, so we conservatively
/// keep the whole set). `Untyped` on the target removes nothing (a target
/// of unknown extent cannot be safely excluded).
pub fn subtract(ty: Ty, target: Ty, env: ConsultationView) -> Ty {
    let ty = expand_alias(env, ty);
    let target = expand_alias(env, target);
    if ty == target {
        return Ty::BOTTOM;
    }
    if ty == Ty::BOTTOM {
        return Ty::BOTTOM;
    }
    if target == Ty::BOTTOM {
        return ty;
    }
    if ty.is_untyped() {
        return ty;
    }
    if target.is_untyped() {
        return ty;
    }
    if target == Ty::TOP {
        return Ty::BOTTOM;
    }

    let types = env.types();
    if let (Type::Bool, Type::Literal(Literal::Bool(value))) =
        (types.resolve(ty), types.resolve(target))
    {
        return types.intern(Type::Literal(Literal::Bool(!value)));
    }
    match types.resolve(ty) {
        Type::Union(members) => {
            let kept: Vec<Ty> = members
                .iter()
                .copied()
                .map(|m| subtract(m, target, env))
                .collect();
            union_of_many(&kept, env.types())
        }
        Type::Optional(inner) => {
            let lhs = subtract(*inner, target, env);
            let rhs = subtract(Ty::NIL, target, env);
            union_of(lhs, rhs, env.types())
        }
        _ => {
            let checker = SubtypeChecker::new(env);
            if checker.check(ty, target) {
                Ty::BOTTOM
            } else {
                ty
            }
        }
    }
}

