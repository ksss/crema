//! Multi-scrutinee narrowing environment used by `analyze_condition`.
//!
//! `CondEnv` carries the set of scrutinees that a predicate narrows on
//! its truthy or falsy branch. A "scrutinee" is either a bare local
//! variable ([`Scrutinee::Lvar`]) or a pure method-call expression keyed
//! by its structural [`PureKey`] ([`Scrutinee::Pure`]) — the two share
//! the same narrowing machinery because `narrow` / `subtract` (ADR-0022
//! §(2)) operate on the scrutinee's `Ty` regardless of how its identity
//! is recorded.
//!
//! The "no entry = original ctx type" invariant lets `join_branch`'s
//! "intersect on names, union on types" semantics reproduce Steep's
//! observed path-merge behavior (ADR-0022 §(1)): a scrutinee present in
//! only one branch reverts to its original type via the ctx lookup,
//! which equals "type-union with the unchanged side."
//!
//! Operations:
//! - `merge_sequential` — `a && b` truthy / `a || b` falsy. The rhs env
//!   was evaluated under `self` already in scope, so same-key entries
//!   are overwritten by `rhs` (rhs is the stricter narrow on top).
//! - `join_branch` — `a && b` falsy / `a || b` truthy. Only keys
//!   present in both envs survive; their types are unioned.
//!
//! An env also records whether its side is unreachable: some scrutinee
//! narrowed to `bot` (Steep's `LogicTypeInterpreter` result
//! `unreachable` flag). The flag survives `merge_sequential` from either
//! side and `join_branch` only from both, so a later narrow of the same
//! key cannot revive a dead side; `&&` / `||` then reset it from their
//! left operand ([`CondEnv::with_unreachable`]). Dead arms read such a
//! scrutinee as `untyped` ([`CondEnv::planted`]) and drop out of the
//! join.

use rustc_hash::FxHashMap;

use crate::definition_builder::ConsultationView;
use crate::name::Name;
use crate::pure_call_env::PureKey;
use crate::types::{Ty, union_of};

/// The narrow target: either a local variable read or a pure call
/// expression. Carries the discriminator the type-checker needs to
/// route a write back to the right env slot — `Lvar` writes go through
/// [`crate::context::Context::set_local_variable`] (lvar scope), `Pure`
/// writes go through `pure_call_env_mut().set` (pure-call cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scrutinee {
    Lvar(Name),
    Pure(PureKey),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CondEnv {
    entries: FxHashMap<Scrutinee, Ty>,
    unreachable: bool,
}

impl CondEnv {
    pub(crate) fn empty() -> Self {
        CondEnv {
            entries: FxHashMap::default(),
            unreachable: false,
        }
    }

    pub(crate) fn single(scrutinee: Scrutinee, ty: Ty) -> Self {
        let mut entries = FxHashMap::default();
        entries.insert(scrutinee, ty);
        CondEnv {
            entries,
            unreachable: ty == Ty::BOTTOM,
        }
    }

    #[cfg(test)]
    pub(crate) fn single_lvar(name: Name, ty: Ty) -> Self {
        Self::single(Scrutinee::Lvar(name), ty)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether this side can never run: a scrutinee narrowed to `bot`.
    pub(crate) fn is_unreachable(&self) -> bool {
        self.unreachable
    }

    /// Override the reachability of a composed env. `&&` / `||` take it
    /// from the left operand alone (Steep's `:and` / `:or`).
    pub(crate) fn with_unreachable(mut self, unreachable: bool) -> Self {
        self.unreachable = unreachable;
        self
    }

    /// The type an arm plants for an entry, `None` to leave the binding
    /// alone. A scrutinee narrowed to `bot` reads as `untyped` inside a
    /// dead arm (Steep types the value `untyped` there), so the arm's
    /// diagnostics are not about an artifact of the narrow. In a live
    /// arm such an entry only comes from the right operand of `&&` /
    /// `||`, and the arm keeps the scrutinee's type as Steep does for a
    /// narrow that finds no type.
    pub(crate) fn planted(&self, ty: Ty) -> Option<Ty> {
        match ty {
            Ty::BOTTOM if self.unreachable => Some(Ty::UNTYPED),
            Ty::BOTTOM => None,
            _ => Some(ty),
        }
    }

    #[cfg(test)]
    pub(crate) fn lookup_lvar(&self, name: Name) -> Option<Ty> {
        self.entries.get(&Scrutinee::Lvar(name)).copied()
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&Scrutinee, Ty)> + '_ {
        self.entries.iter().map(|(k, &t)| (k, t))
    }

    /// Iterate entries in the order required by every branch / narrow
    /// dispatch: every `Scrutinee::Lvar` first, then every
    /// `Scrutinee::Pure`. The ordering is significant because the lvar
    /// write path (`Context::set_local_variable` /
    /// `Context::enter_narrow`) invalidates pure-cache entries whose
    /// key references the narrowed name. Running lvar writes before
    /// pure writes means the pure narrows we plant in the same pass
    /// are not the collateral damage of their own siblings'
    /// invalidations. Mirrors Steep's `refine_types` atomicity at
    /// `type_env.rb:137-157`.
    ///
    /// Centralises the rule on this side so callers
    /// (`with_cond_narrowing`, `with_cond_branch`) don't each carry an
    /// independent copy of "iterate twice, lvar-then-pure" -- a future
    /// change to invalidation semantics has a single application point.
    pub(crate) fn iter_lvars_then_pures(&self) -> impl Iterator<Item = (&Scrutinee, Ty)> + '_ {
        let lvars = self
            .entries
            .iter()
            .filter(|(k, _)| matches!(k, Scrutinee::Lvar(_)))
            .map(|(k, &t)| (k, t));
        let pures = self
            .entries
            .iter()
            .filter(|(k, _)| matches!(k, Scrutinee::Pure(_)))
            .map(|(k, &t)| (k, t));
        lvars.chain(pures)
    }

    /// Sequentially compose two envs assuming `rhs` was evaluated with
    /// `self` already in scope. Same-key entries are overwritten by
    /// `rhs`.
    /// A `bot` entry in `rhs` keeps `self`'s entry for the key: the
    /// later narrow found no type, which ends that narrow (`unreachable`)
    /// without undoing the earlier one, as Steep keeps the env.
    pub(crate) fn merge_sequential(mut self, rhs: Self) -> Self {
        for (key, ty) in rhs.entries {
            if ty == Ty::BOTTOM && self.entries.contains_key(&key) {
                continue;
            }
            self.entries.insert(key, ty);
        }
        self.unreachable |= rhs.unreachable;
        self
    }

    /// Join two envs at a branch confluence. Only keys present in BOTH
    /// envs survive; their types are unioned. Keys appearing in only
    /// one env are dropped — they revert to the ctx's original type at
    /// lookup time, matching Steep's type-union join semantics.
    pub(crate) fn join_branch(self, rhs: Self, env: ConsultationView) -> Self {
        let mut entries = FxHashMap::default();
        for (key, lhs_ty) in &self.entries {
            if let Some(&rhs_ty) = rhs.entries.get(key) {
                entries.insert(*key, union_of(*lhs_ty, rhs_ty, env.types()));
            }
        }
        CondEnv {
            entries,
            unreachable: self.unreachable && rhs.unreachable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition_builder::DefinitionBuilder;
    use crate::name::NameTable;
    use crate::pure_call_env::{PureKeyTable, PureNode};

    const RBS: &str = r#"
class Integer
end
class String
end
"#;

    fn fixture() -> (DefinitionBuilder, NameTable, Name, Name) {
        let env = DefinitionBuilder::from_rbs_source(RBS.as_bytes()).unwrap();
        let names = NameTable::new();
        let x = names.intern("x");
        let y = names.intern("y");
        (env, names, x, y)
    }

    fn integer_ty(env: &DefinitionBuilder) -> Ty {
        env.class_instance_type(env.names().builtins().integer)
    }

    fn string_ty(env: &DefinitionBuilder) -> Ty {
        env.class_instance_type(env.names().builtins().string)
    }

    #[test]
    fn test_empty_env_has_no_entries() {
        let env = CondEnv::empty();
        assert!(env.is_empty());
        assert_eq!(env.iter().count(), 0);
    }

    #[test]
    fn test_single_env_carries_one_entry() {
        let (env, _names, x, _y) = fixture();
        let int = integer_ty(&env);
        let cond = CondEnv::single_lvar(x, int);
        assert!(!cond.is_empty());
        assert_eq!(cond.lookup_lvar(x), Some(int));
    }

    #[test]
    fn test_lookup_missing_name_returns_none() {
        let (env, _names, x, y) = fixture();
        let int = integer_ty(&env);
        let cond = CondEnv::single_lvar(x, int);
        assert_eq!(cond.lookup_lvar(y), None);
    }

    #[test]
    fn test_merge_sequential_overwrites_same_name() {
        let (env, _names, x, _y) = fixture();
        let int = integer_ty(&env);
        let str_ty = string_ty(&env);
        let lhs = CondEnv::single_lvar(x, int);
        let rhs = CondEnv::single_lvar(x, str_ty);
        let merged = lhs.merge_sequential(rhs);
        assert_eq!(merged.lookup_lvar(x), Some(str_ty));
    }

    #[test]
    fn test_merge_sequential_unions_distinct_names() {
        let (env, _names, x, y) = fixture();
        let int = integer_ty(&env);
        let str_ty = string_ty(&env);
        let lhs = CondEnv::single_lvar(x, int);
        let rhs = CondEnv::single_lvar(y, str_ty);
        let merged = lhs.merge_sequential(rhs);
        assert_eq!(merged.lookup_lvar(x), Some(int));
        assert_eq!(merged.lookup_lvar(y), Some(str_ty));
    }

    #[test]
    fn test_join_branch_keeps_only_common_names() {
        let (env, _names, x, y) = fixture();
        let int = integer_ty(&env);
        let str_ty = string_ty(&env);
        let lhs = CondEnv::single_lvar(x, int);
        let rhs = CondEnv::single_lvar(y, str_ty);
        let view = ConsultationView::new(&env, None);
        let joined = lhs.join_branch(rhs, view);
        assert!(joined.is_empty(), "x and y disjoint should drop both");
    }

    #[test]
    fn test_join_branch_same_type_keeps_one_entry() {
        let (env, _names, x, _y) = fixture();
        let int = integer_ty(&env);
        let lhs = CondEnv::single_lvar(x, int);
        let rhs = CondEnv::single_lvar(x, int);
        let view = ConsultationView::new(&env, None);
        let joined = lhs.join_branch(rhs, view);
        assert_eq!(joined.lookup_lvar(x), Some(int));
    }

    #[test]
    fn test_join_branch_distinct_types_routes_through_union_of() {
        // The CondEnv wiring contract: same-key entries on a branch
        // confluence are unioned via `types::union_of`. Member shape,
        // flatten, dedup, Bottom drop, commutativity and absorption are
        // exercised at the types::union_of layer (see src/types/tests.rs);
        // here we pin only that the join goes through that helper.
        let (env, _names, x, _y) = fixture();
        let int = integer_ty(&env);
        let str_ty = string_ty(&env);
        let lhs = CondEnv::single_lvar(x, int);
        let rhs = CondEnv::single_lvar(x, str_ty);
        let view = ConsultationView::new(&env, None);
        let joined = lhs.join_branch(rhs, view);
        assert_eq!(
            joined.lookup_lvar(x),
            Some(union_of(int, str_ty, env.types()))
        );
    }

    #[test]
    fn test_unreachable_survives_merge_from_either_side_and_join_from_both() {
        let (env, _names, x, y) = fixture();
        let int = integer_ty(&env);
        let dead = || CondEnv::single_lvar(x, Ty::BOTTOM);
        let live = || CondEnv::single_lvar(y, int);
        assert!(dead().is_unreachable());
        assert!(!live().is_unreachable());
        assert!(dead().merge_sequential(live()).is_unreachable());
        assert!(live().merge_sequential(dead()).is_unreachable());
        // A later narrow of the same key does not revive a dead side.
        let revived = dead().merge_sequential(CondEnv::single_lvar(x, int));
        assert!(revived.is_unreachable());
        let view = ConsultationView::new(&env, None);
        assert!(!dead().join_branch(live(), view).is_unreachable());
        let view = ConsultationView::new(&env, None);
        assert!(dead().join_branch(dead(), view).is_unreachable());
    }

    #[test]
    fn test_pure_scrutinee_round_trip() {
        let (env, names, _x, _y) = fixture();
        let str_ty = string_ty(&env);
        let account = names.intern("account");
        let phone = names.intern_symbol("phone");
        let keys = PureKeyTable::new();
        let key = keys.intern(PureNode::Send(keys.intern(PureNode::Lvar(account)), phone));
        let cond = CondEnv::single(Scrutinee::Pure(key), str_ty);
        let entries: Vec<_> = cond.iter().collect();
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].0, Scrutinee::Pure(k) if k == &key));
    }
}
