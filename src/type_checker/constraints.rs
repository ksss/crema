use rustc_hash::{FxHashMap, FxHashSet};

use crate::type_param::TypeVarKey;
use crate::types::{Ty, Type, TypeTable};

/// Constraint set for method-level type parameter inference.
///
/// Two complementary bound kinds:
/// - **lower** (`X ⊇ ty`) — Phase A: produced by call-site arg unification
///   and by block-body return unification. Seed-derived information.
/// - **upper** (`X ⊆ ty`) — Phase B: produced by unifying the overload's
///   return type against a context hint (e.g. a trailing `#:` assertion).
///   Intent-derived information.
///
/// Resolution rule: when both bounds exist for the same unknown, the upper
/// bound wins. The hint is the call-site intent and overrides the seed; this
/// is what makes `with_object([]) do |i, xs| ... end #: Array[Integer]` bind
/// `xs: Array[Integer]` rather than `Array[untyped]`.
///
/// Genuine conflicts (`lower` not `<:` `upper` in either direction) are
/// detected in `apply_hint_override` via a bidirectional `SubtypeChecker`
/// gate and reported as `Ruby::UnsatisfiableConstraint`. The upper bound
/// (hint) is still adopted so downstream inference continues from user intent.
///
/// Expected lifecycle: build a fresh `Constraints` per call site, seed with
/// `unknowns` (the method-level type param names), accumulate via `add_lower`
/// and `add_upper`, then call `solution` to produce a `{name -> Ty}` map for
/// substitution.
#[derive(Debug, Clone)]
pub(super) struct Constraints {
    unknowns: FxHashSet<TypeVarKey>,
    lower: FxHashMap<TypeVarKey, Vec<Ty>>,
    upper: FxHashMap<TypeVarKey, Vec<Ty>>,
}

impl Constraints {
    pub(super) fn new(unknowns: impl IntoIterator<Item = TypeVarKey>) -> Self {
        Constraints {
            unknowns: unknowns.into_iter().collect(),
            lower: FxHashMap::default(),
            upper: FxHashMap::default(),
        }
    }

    /// Register `name ⊇ ty`. No-op if `name` is not declared as an unknown —
    /// callers may pass any `TypeVariable` they encounter and the set decides
    /// whether to record it.
    pub(super) fn add_lower(&mut self, name: TypeVarKey, ty: Ty) {
        if !self.unknowns.contains(&name) {
            return;
        }
        self.lower.entry(name).or_default().push(ty);
    }

    /// Register `name ⊆ ty`. Hint-driven; takes precedence over `lower`
    /// bounds in `solution` (Phase B intent overrides Phase A seed).
    ///
    /// Phase B Stage B overrides bindings directly via `apply_hint_override`.
    /// `Ruby::UnsatisfiableConstraint` conflict detection is also implemented
    /// there (bidirectional SubtypeChecker gate). This API is kept for a
    /// future structural refactor that routes hint upper bounds through this
    /// solver — which would enable Steep-compatible derivation chain output
    /// and a cleaner separation between constraint accumulation and checking.
    #[allow(dead_code)]
    pub(super) fn add_upper(&mut self, name: TypeVarKey, ty: Ty) {
        if !self.unknowns.contains(&name) {
            return;
        }
        self.upper.entry(name).or_default().push(ty);
    }

    /// Collapse accumulated bounds into a single `{name -> Ty}` map.
    ///
    /// Selection per unknown:
    /// - any `upper` present → fold uppers (Phase B intent wins over seeds)
    /// - else any `lower` present → fold lowers (Phase A behavior)
    /// - else omit (callers fall back to `UNTYPED`)
    ///
    /// Folding within a side: one bound → that bound verbatim, multiple →
    /// their union (deduped, flattened).
    pub(super) fn solution(&self, types: &TypeTable) -> FxHashMap<TypeVarKey, Ty> {
        let mut out = FxHashMap::default();
        for name in &self.unknowns {
            let bounds = match (self.upper.get(name), self.lower.get(name)) {
                (Some(u), _) if !u.is_empty() => u,
                (_, Some(l)) if !l.is_empty() => l,
                _ => continue,
            };
            let mut flat: Vec<Ty> = Vec::new();
            for &b in bounds {
                push_flat_union(b, &mut flat, types);
            }
            let resolved = if flat.len() == 1 {
                flat[0]
            } else {
                types.intern(Type::Union(flat))
            };
            out.insert(name.clone(), resolved);
        }
        out
    }
}

fn push_flat_union(ty: Ty, out: &mut Vec<Ty>, types: &TypeTable) {
    match types.resolve(ty) {
        Type::Union(members) => {
            for &m in members {
                push_flat_union(m, out, types);
            }
        }
        _ => {
            if !out.contains(&ty) {
                out.push(ty);
            }
        }
    }
}

