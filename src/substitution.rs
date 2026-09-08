//! Port of `RBS::Substitution` for declarative type application.
//!
//! Carries a mapping from type variable names to concrete types, plus optional
//! overrides for the three base keywords (`self`, `instance`, `class`). The
//! four-axis form extends the rbs two-axis form to cover crema's needs;
//! `self_type` and `class_type` are crema extensions not present in rbs.
//!
//! For inference-compose usage (Shape layer), see
//! `crate::type_checker::interface::Substitution`.

use rustc_hash::FxHashMap;

use crate::type_param::TypeVarKey;
use crate::types::{
    Block, Function, FunctionType, MethodType, RecordKey, Ty, Type, TypeTable, UntypedFunction,
};

/// Port of `RBS::Substitution`.
///
/// Declarative substitution: apply a fixed set of bindings to a type.
/// Unbound variables and unset base keywords are preserved unchanged.
#[derive(Debug, Default, Clone)]
pub struct Substitution {
    pub mapping: FxHashMap<TypeVarKey, Ty>,
    pub self_type: Option<Ty>,
    pub instance_type: Option<Ty>,
    pub class_type: Option<Ty>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_mapping(mapping: FxHashMap<TypeVarKey, Ty>) -> Self {
        Self {
            mapping,
            self_type: None,
            instance_type: None,
            class_type: None,
        }
    }

    pub fn with_self_type(mut self, self_type: Ty) -> Self {
        self.self_type = Some(self_type);
        self
    }

    pub fn with_instance_type(mut self, instance_type: Ty) -> Self {
        self.instance_type = Some(instance_type);
        self
    }

    pub fn with_class_type(mut self, class_type: Ty) -> Self {
        self.class_type = Some(class_type);
        self
    }

    /// Port of `RBS::Substitution#apply`.
    ///
    /// Substitutes type variables and base keywords in `ty`.
    /// Recurses into composite types (ClassInstance, Union, etc.).
    pub fn apply(&self, ty: Ty, types: &TypeTable) -> Ty {
        match types.resolve(ty) {
            Type::TypeVariable { raw, scope } => {
                let key = TypeVarKey {
                    raw: *raw,
                    scope: scope.clone(),
                };
                self.mapping.get(&key).copied().unwrap_or(ty)
            }
            Type::SelfType => self.self_type.unwrap_or(ty),
            Type::InstanceType => self.instance_type.unwrap_or(ty),
            Type::ClassType => self.class_type.unwrap_or(ty),
            Type::ClassInstance { name, args } => {
                let new_args: Vec<Ty> = args.iter().map(|&a| self.apply(a, types)).collect();
                types.intern(Type::ClassInstance {
                    name: *name,
                    args: new_args,
                })
            }
            Type::Union(members) => {
                let new_members: Vec<Ty> = members.iter().map(|&m| self.apply(m, types)).collect();
                types.intern(Type::Union(new_members))
            }
            Type::Intersection(members) => {
                let new_members: Vec<Ty> = members.iter().map(|&m| self.apply(m, types)).collect();
                types.intern(Type::Intersection(new_members))
            }
            Type::Optional(inner) => {
                let new_inner = self.apply(*inner, types);
                types.intern(Type::Optional(new_inner))
            }
            Type::Interface { name, args } => {
                let new_args: Vec<Ty> = args.iter().map(|&a| self.apply(a, types)).collect();
                types.intern(Type::Interface {
                    name: *name,
                    args: new_args,
                })
            }
            Type::Alias { name, args } => {
                let new_args: Vec<Ty> = args.iter().map(|&a| self.apply(a, types)).collect();
                types.intern(Type::Alias {
                    name: *name,
                    args: new_args,
                })
            }
            Type::ClassSingleton { .. } => ty,
            Type::Tuple(members) => {
                let new_members: Vec<Ty> = members.iter().map(|&m| self.apply(m, types)).collect();
                types.intern(Type::Tuple(new_members))
            }
            Type::Record { fields } => {
                let new_fields: Vec<(RecordKey, Ty, bool)> = fields
                    .iter()
                    .map(|(key, ty, required)| (key.clone(), self.apply(*ty, types), *required))
                    .collect();
                types.intern(Type::Record { fields: new_fields })
            }
            Type::Proc {
                type_,
                self_type,
                block,
            } => {
                let new_type = self.apply_function_type(type_, types);
                let new_self = self_type.map(|st| self.apply(st, types));
                let new_block = block.as_ref().map(|b| self.apply_block(b, types));
                types.intern(Type::Proc {
                    type_: new_type,
                    self_type: new_self,
                    block: new_block,
                })
            }
            _ => ty,
        }
    }

    pub fn apply_block(&self, block: &Block, types: &TypeTable) -> Block {
        Block {
            required: block.required,
            type_: self.apply_function_type(&block.type_, types),
            self_type: block.self_type.map(|st| self.apply(st, types)),
        }
    }

    /// Apply this substitution to a whole `MethodType`. Carries `type_params`
    /// through unchanged: method-level type parameters live in `scope=Method`
    /// keys that don't collide with outer (`scope=Class`) bindings, so no
    /// `retain` step is needed (matches the existing `substitute_method_defs`
    /// helper in `definition_builder`).
    ///
    /// The scope-based disambiguation is what protects method-level params
    /// here — `TypeVarKey` is `(raw, scope)`, so a class-level `T` and a
    /// method-level `[T]` are distinct keys even when their raw names
    /// match. Callers that construct test-only `Free`-scoped keys with
    /// colliding raw names will see the method-level param substituted,
    /// because no scope distinction exists to disambiguate them.
    pub fn apply_method_type(&self, mt: &MethodType, types: &TypeTable) -> MethodType {
        MethodType {
            type_params: mt.type_params.clone(),
            type_: self.apply_function_type(&mt.type_, types),
            block: mt.block.as_ref().map(|b| self.apply_block(b, types)),
        }
    }

    pub fn apply_function_type(&self, ft: &FunctionType, types: &TypeTable) -> FunctionType {
        match ft {
            FunctionType::Untyped(u) => FunctionType::Untyped(UntypedFunction {
                return_type: self.apply(u.return_type, types),
            }),
            FunctionType::Typed(f) => FunctionType::Typed(Function {
                required_positionals: f
                    .required_positionals
                    .iter()
                    .map(|&t| self.apply(t, types))
                    .collect(),
                optional_positionals: f
                    .optional_positionals
                    .iter()
                    .map(|&t| self.apply(t, types))
                    .collect(),
                rest_positional: f.rest_positional.map(|t| self.apply(t, types)),
                trailing_positionals: f
                    .trailing_positionals
                    .iter()
                    .map(|&t| self.apply(t, types))
                    .collect(),
                required_keywords: f
                    .required_keywords
                    .iter()
                    .map(|(k, t)| (k.clone(), self.apply(*t, types)))
                    .collect(),
                optional_keywords: f
                    .optional_keywords
                    .iter()
                    .map(|(k, t)| (k.clone(), self.apply(*t, types)))
                    .collect(),
                rest_keyword: f.rest_keyword.map(|t| self.apply(t, types)),
                return_type: self.apply(f.return_type, types),
            }),
        }
    }
}

