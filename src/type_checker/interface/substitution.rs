//! Port of `Steep::Interface::Substitution` for inference-compose usage.
//!
//! Carries a `dictionary` mapping type variable names to concrete types, plus
//! optional overrides for `instance_type`, `module_type`, and `self_type`.
//!
//! **Not** the same as `crate::substitution::Substitution` (`RBS::Substitution`,
//! declarative apply). This struct models Steep's solver accumulator: new
//! substitutions are *composed* into the accumulator via `merge_mut`, which
//! re-applies the incoming substitution to existing dictionary values before
//! merging (function-composition semantics, not simple key-value union).
//!
//! # Steep source
//! `lib/steep/interface/substitution.rb`

use rustc_hash::{FxHashMap, FxHashSet};

use crate::substitution::Substitution as RbsSubstitution;
use crate::type_param::TypeVarKey;
use crate::types::{MethodType, Ty, TypeTable};

/// Port of `Steep::Interface::Substitution`.
///
/// Used by the Shape / inference layer to accumulate type-variable bindings
/// during constraint solving. Field names mirror Steep's implementation.
#[derive(Debug, Default, Clone)]
pub struct Substitution {
    pub dictionary: FxHashMap<TypeVarKey, Ty>,
    pub instance_type: Option<Ty>,
    pub module_type: Option<Ty>,
    pub self_type: Option<Ty>,
}

impl Substitution {
    /// Port of `Steep::Interface::Substitution.empty`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Port of `Steep::Interface::Substitution#merge!`.
    ///
    /// Composes `other` into `self` with the following semantics (matching
    /// Steep's `merge!(s, overwrite:)` implementation):
    ///
    /// 1. Each existing value in `self.dictionary` is re-substituted by `other`
    ///    (`transform_values! {|ty| ty.subst(s) }` in Steep).
    /// 2. Keys from `other.dictionary` are merged in.  On conflict:
    ///    - `overwrite: true`  → `other` wins (Steep `overwrite: true`)
    ///    - `overwrite: false` → panics (Steep `raise "Duplicated key"`)
    /// 3. `instance_type` / `module_type` / `self_type` are *composed*, not
    ///    overwritten: an existing axis is re-substituted by `other`
    ///    (Steep `@instance_type = instance_type.subst(s) if instance_type`),
    ///    and a `None` axis (Steep's marker singleton) takes `other`'s axis.
    ///    Axes never raise on conflict — only the dictionary does.
    pub fn merge_mut(&mut self, other: &Substitution, overwrite: bool, types: &TypeTable) {
        let adapter = other.to_rbs_adapter();

        // Step 1: re-substitute existing dictionary values by `other`
        // (Steep `dictionary.transform_values! {|ty| ty.subst(s) }`).
        for val in self.dictionary.values_mut() {
            *val = adapter.apply(*val, types);
        }

        // Step 2: merge keys from `other`.
        for (key, &val) in &other.dictionary {
            if self.dictionary.contains_key(key) && !overwrite {
                panic!("Duplicated key in Substitution::merge_mut: {:?}", key);
            }
            self.dictionary.insert(key.clone(), val);
        }

        // Step 3: compose the base-keyword axes.
        self.instance_type = compose_axis(self.instance_type, other.instance_type, &adapter, types);
        self.module_type = compose_axis(self.module_type, other.module_type, &adapter, types);
        self.self_type = compose_axis(self.self_type, other.self_type, &adapter, types);
    }

    /// Bridge this Steep-side substitution to an `RBS::Substitution` so the
    /// declarative `apply` machinery can be reused. `module_type` maps to the
    /// rbs `class_type` axis because both represent the class/module-as-object
    /// for the `class` RBS keyword.
    pub(crate) fn to_rbs_adapter(&self) -> RbsSubstitution {
        let mut adapter = RbsSubstitution::from_mapping(self.dictionary.clone());
        if let Some(it) = self.instance_type {
            adapter = adapter.with_instance_type(it);
        }
        if let Some(mt) = self.module_type {
            adapter = adapter.with_class_type(mt);
        }
        if let Some(st) = self.self_type {
            adapter = adapter.with_self_type(st);
        }
        adapter
    }

    /// Apply this substitution to a single type. Port of the `type.subst(s)`
    /// call inside Steep `Shape#subst`.
    pub fn apply_type(&self, ty: Ty, types: &TypeTable) -> Ty {
        self.to_rbs_adapter().apply(ty, types)
    }

    /// Apply this substitution to a method type. Port of Steep
    /// `MethodType#subst`: method-level `type_params` are removed from the
    /// substitution first so a method's own bound variables are never replaced
    /// by an outer binding of the same name.
    ///
    /// `type_params` are carried through unchanged (including their bounds).
    /// This matches Steep, whose `MethodType#subst` reconstructs with
    /// `type_params: type_params`; it differs from rbs's `MethodType#sub`,
    /// which also `map_type`s each param's bound. crema follows Steep here
    /// because this is the Shape/interface layer's substitution path.
    pub fn apply_method_type(&self, mt: &MethodType, types: &TypeTable) -> MethodType {
        let mut adapter = self.to_rbs_adapter();
        if !mt.type_params.is_empty() {
            let bound: FxHashSet<TypeVarKey> =
                mt.type_params.iter().map(|p| p.name.clone()).collect();
            adapter.mapping.retain(|k, _| !bound.contains(k));
        }
        MethodType {
            type_params: mt.type_params.clone(),
            type_: adapter.apply_function_type(&mt.type_, types),
            block: mt.block.as_ref().map(|b| adapter.apply_block(b, types)),
        }
    }
}

/// Compose a single base-keyword axis. Port of Steep
/// `@axis = axis.subst(s) if axis`: an existing axis (`Some`) is re-substituted
/// by `other` via `adapter`; a `None` axis stands for Steep's marker singleton,
/// whose `subst(s)` yields `other`'s axis (`incoming`).
fn compose_axis(
    dst: Option<Ty>,
    incoming: Option<Ty>,
    adapter: &RbsSubstitution,
    types: &TypeTable,
) -> Option<Ty> {
    match dst {
        Some(v) => Some(adapter.apply(v, types)),
        None => incoming,
    }
}

