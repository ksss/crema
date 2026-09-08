//! Pure-call cache: the sequential type-env axis for method-return
//! narrowing.
//!
//! Companion to [`crate::context::Context`]'s lvar-scope binding store.
//! Where `Context.scopes` carries `Name → Ty` for local variables,
//! `PureCallEnv` carries `PureKey → Ty` for the structurally-equal pure
//! method-call expressions whose returns have already been narrowed.
//! The two axes are orthogonal: a pure entry is dropped when one of the
//! lvar names it references is reassigned (see
//! [`PureCallEnv::invalidate_by_lvar`]) or when a constant assignment
//! forces constant-rooted entries to be cleared conservatively. Branch
//! confluence on each axis happens independently.
//!
//! "Pure" here is decided by the caller via [`crate::definition::Method::is_pure`];
//! this module is purely the cache substrate.
//!
//! Mirrors Steep's `TypeEnv#pure_method_calls`
//! (`lib/steep/type_inference/type_env.rb:42-310`). The key contract
//! follows ADR-0024:
//!   - cache key is structural (value-based identity), so the same
//!     expression at condition site and body site shares an entry
//!   - invalidation walks the key's syntax tree looking for an lvar
//!     read of the assigned name; the assigned value is irrelevant
//!   - branch confluence keeps only keys present in BOTH arms, with
//!     types unioned via [`crate::types::union_of`]
//!
//! Scope of this module (ADR-0024 Phase 1 plus constant singleton calls):
//!   - keys cover bare local variables, `self`, constant paths, and chained
//!     method sends with no arguments and no block. Instance variables are
//!     deliberately absent — they need separate invalidation rules and
//!     belong to a follow-up todo (`mid_pure_narrowing_extension`).

use rustc_hash::FxHashMap;

use crate::definition_builder::ConsultationView;
use crate::name::{Name, Symbol};
use crate::type_name::TypeName;
use crate::types::{Ty, union_of};

/// Structural identity of a pure receiver/call expression.
///
/// Two `PureKey`s compare equal iff their syntactic shapes match —
/// `c.phone` written in a condition position equals the same `c.phone`
/// written inside the if body, so the narrowed type propagates.
///
/// The recursive `Send` variant naturally encodes chains: `a.b.c` is
/// `Send(Send(Lvar(a), b), c)`. There is no depth bound at this layer.
/// Constant singleton receivers use the resolved absolute type name, so
/// `RBS.logger_output` is `Send(ConstPath(::RBS), logger_output)`.
///
/// `Lvar(Name)` uses the lvar interner; `Send`'s method-name slot uses
/// the `Symbol` interner — these are different namespaces in crema even
/// though both wrap the same backing `Spur` (see `src/name.rs`).
///
/// `SelfRef` is the normalized base for self-rooted calls. Both implicit
/// self (`(send nil :path)`) and explicit `self.path` (`(send (self) :path)`)
/// collapse to `Send(SelfRef, path)`, so the two spellings share a cache
/// entry. Steep keys on raw AST nodes and keeps them separate; crema's
/// `PureKey` is a normalized structural form, so folding self into one
/// base is consistent with this layer (ADR-0024).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PureKey {
    Lvar(Name),
    SelfRef,
    ConstPath(TypeName),
    Send(Box<PureKey>, Symbol),
}

impl PureKey {
    /// True iff this key's syntax tree contains a `Lvar(name)` somewhere.
    /// Used by [`PureCallEnv::invalidate_by_lvar`] to find entries whose
    /// receiver chain references the reassigned variable. `SelfRef` never
    /// references an lvar, so a self-rooted entry is never dropped by lvar
    /// reassignment (self is invariant within a method body; block-entry
    /// self change is out of Phase 1 scope).
    pub fn contains_lvar(&self, name: Name) -> bool {
        match self {
            PureKey::Lvar(n) => *n == name,
            PureKey::SelfRef | PureKey::ConstPath(_) => false,
            PureKey::Send(recv, _) => recv.contains_lvar(name),
        }
    }

    pub fn contains_const_path(&self) -> bool {
        match self {
            PureKey::ConstPath(_) => true,
            PureKey::Lvar(_) | PureKey::SelfRef => false,
            PureKey::Send(recv, _) => recv.contains_const_path(),
        }
    }
}

/// Flat `PureKey → Ty` store. Lifetime parallels Steep's
/// `pure_method_calls`: not scope-stacked (lvar bindings are scope-stacked
/// in `Context.scopes`; pure entries are not, mirroring Steep's flat
/// hash).
#[derive(Debug, Clone, Default)]
pub struct PureCallEnv {
    entries: FxHashMap<PureKey, Ty>,
}

impl PureCallEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &PureKey) -> Option<Ty> {
        self.entries.get(key).copied()
    }

    pub fn set(&mut self, key: PureKey, ty: Ty) {
        self.entries.insert(key, ty);
    }

    /// Remove a single key. Used by [`crate::context::Context::exit_pure_narrow`]
    /// when restoring the "no prior entry" state.
    pub fn remove(&mut self, key: &PureKey) -> Option<Ty> {
        self.entries.remove(key)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PureKey, Ty)> + '_ {
        self.entries.iter().map(|(k, &t)| (k, t))
    }

    /// Drop every entry whose key references `name`. Returns the number
    /// of entries dropped — useful for assertions in tests.
    ///
    /// Mirrors Steep's `invalidated_pure_nodes` / `pure_node_invalidation`
    /// pair (`type_env.rb:286-309`), collapsed into one pass since we
    /// have no need to emit `(call, nil)` placeholders — crema's
    /// downstream consumers only inspect presence.
    pub fn invalidate_by_lvar(&mut self, name: Name) -> usize {
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.contains_lvar(name));
        before - self.entries.len()
    }

    pub fn invalidate_const_paths(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.contains_const_path());
        before - self.entries.len()
    }

    /// Branch-confluence join: only keys present in BOTH arms survive,
    /// and their types are unioned via [`union_of`]. Mirrors Steep's
    /// per-arm `common_pure_nodes` intersection plus `Union.build`
    /// merge at `type_env.rb:243-255`.
    ///
    /// Callers passing arm snapshots collected by the if/else machinery
    /// get the "if-only arm drops, both-arms unions" semantics for free.
    pub fn join_at_branch(self, other: Self, env: ConsultationView) -> Self {
        let mut joined = FxHashMap::default();
        for (key, lhs_ty) in &self.entries {
            if let Some(&rhs_ty) = other.entries.get(key) {
                joined.insert(key.clone(), union_of(*lhs_ty, rhs_ty, env.types()));
            }
        }
        Self { entries: joined }
    }
}

