//! Mirrors `RBS::Substitution` (`lib/rbs/substitution.rb`).
//!
//! AST-level substitution that rewrites `TypeVariable` occurrences in
//! an AST [`Type`] tree. Used by `MethodBuilder::build_instance` to
//! rename a reopen-decl's class type params onto the primary decl's
//! type params before bucket-pushing the member, matching rbs's
//! `member.update(overloads: member.overloads.map { |o| o.sub(subst) })`
//! pattern (`lib/rbs/definition_builder/method_builder.rb:115-128`).
//!
//! crema's lowered `Ty` is rewritten by a sibling [`crate::types::Substitution`]
//! (`src/types.rs`). The two substitutions live at separate layers
//! because rbs's `RBS::Types::*` doubles as both AST and resolved
//! while crema splits the two — we need one substitution per layer.
//!
//! Method-level type-param shadowing mirrors rbs `MethodType#sub`'s
//! `s.without(*type_param_names)` step (`lib/rbs/method_type.rb:33-48`):
//! [`Substitution::apply_method_type`] accepts a `&NameTable` and removes
//! method-level param names from the substitution before applying it, so a
//! class-level `T → U` mapping cannot overwrite a method-scoped `[T]`.

use rustc_hash::FxHashMap;

use super::{
    AliasType, BlockType, ClassInstanceType, ClassSingletonType, Function, FunctionParam,
    InterfaceType, IntersectionType, OptionalType, ProcType, RecordField, RecordType, TupleType,
    Type, UnionType, UntypedFunctionType, VariableType,
};
use crate::ast::TypeParam;
use crate::ast::members::MethodDefinitionOverload;
use crate::ast::method_type::MethodType;
use crate::ast::types::{FunctionType, KeywordParam};
use crate::name::{NameTable, Symbol};

/// Mirrors `RBS::Substitution`.
///
/// Maps class-level type-param `Symbol`s to the AST `Type` they should
/// be replaced with. Built by [`Substitution::build`] from a per-decl
/// type-param list and the primary decl's type-param list; applied
/// recursively over an AST `Type` tree by [`Substitution::apply_type`].
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    map: FxHashMap<Symbol, Type>,
}

impl Substitution {
    /// Empty substitution. Mirrors `RBS::Substitution.new`.
    pub fn empty() -> Self {
        Substitution::default()
    }

    /// Mirrors `RBS::Substitution.build(variables, types)` in the
    /// special case `types = primary_params.map { TypeVariable.new(name:) }`.
    /// Builds a substitution that maps each `decl_params[i].name` to
    /// `TypeVariable(primary_params[i].name)`, producing the rename
    /// that turns a reopen decl's member into a primary-scope member.
    ///
    /// Arity-mismatch fallback: when `decl_params` is longer than
    /// `primary_params`, the extra entries are mapped to themselves
    /// (no rename target). Mirrors the same fallback in crema's
    /// `build_alpha_renamed_param_scope`; the build-phase
    /// `validate_type_params` normally rejects open-class arity drift
    /// before this is reached.
    pub fn build(decl_params: &[TypeParam], primary_params: &[TypeParam]) -> Self {
        let mut subst = Self::empty();
        for (i, decl_tp) in decl_params.iter().enumerate() {
            let target_name = match primary_params.get(i) {
                Some(p) => p.name,
                None => decl_tp.name,
            };
            subst.add(
                decl_tp.name,
                Type::Variable(VariableType {
                    name: target_name,
                    location: None,
                }),
            );
        }
        subst
    }

    /// Mirrors `RBS::Substitution#empty?`.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Mirrors `RBS::Substitution#add(from:, to:)`.
    pub fn add(&mut self, from: Symbol, to: Type) {
        self.map.insert(from, to);
    }

    /// Mirrors `RBS::Substitution#without(*vars)`. Returns a clone of
    /// this substitution with the named entries removed, used by
    /// `RBS::MethodType#sub` for method-level type-param shadowing.
    pub fn without(&self, names: &[Symbol]) -> Self {
        let mut sub = self.clone();
        for name in names {
            sub.map.remove(name);
        }
        sub
    }

    /// Mirrors `RBS::Types::*#sub`. Recursive walk over the AST `Type`
    /// tree that rewrites every `TypeVariable` whose name is bound in
    /// the substitution map. Non-`TypeVariable` leaves are returned
    /// unchanged; composite types map their child types through
    /// `apply_type` and preserve their own `location`.
    pub fn apply_type(&self, ty: &Type) -> Type {
        if self.is_empty() {
            return ty.clone();
        }
        match ty {
            Type::Variable(VariableType { name, location }) => match self.map.get(name) {
                // Preserve the source `location` from `ty` when the
                // substitution value is itself a `Variable`. Without
                // this, `build` populates the map with location-less
                // `Variable`s (the rename target's source location is
                // unknown at build time), so an unchanged identity rename
                // (primary decl applies its own subst) would erase every
                // type-param occurrence's source span.
                Some(Type::Variable(VariableType {
                    name: target_name, ..
                })) => Type::Variable(VariableType {
                    name: *target_name,
                    location: *location,
                }),
                Some(other) => other.clone(),
                None => ty.clone(),
            },
            Type::ClassInstance(ClassInstanceType {
                name,
                args,
                location,
            }) => Type::ClassInstance(ClassInstanceType {
                name: *name,
                args: args.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::ClassSingleton(ClassSingletonType {
                name,
                args,
                location,
            }) => Type::ClassSingleton(ClassSingletonType {
                name: *name,
                args: args.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::Interface(InterfaceType {
                name,
                args,
                location,
            }) => Type::Interface(InterfaceType {
                name: *name,
                args: args.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::Alias(AliasType {
                name,
                args,
                location,
            }) => Type::Alias(AliasType {
                name: *name,
                args: args.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::Union(UnionType { types, location }) => Type::Union(UnionType {
                types: types.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::Intersection(IntersectionType { types, location }) => {
                Type::Intersection(IntersectionType {
                    types: types.iter().map(|t| self.apply_type(t)).collect(),
                    location: *location,
                })
            }
            Type::Optional(OptionalType {
                ty: inner,
                location,
            }) => Type::Optional(OptionalType {
                ty: Box::new(self.apply_type(inner)),
                location: *location,
            }),
            Type::Tuple(TupleType { types, location }) => Type::Tuple(TupleType {
                types: types.iter().map(|t| self.apply_type(t)).collect(),
                location: *location,
            }),
            Type::Record(RecordType { fields, location }) => Type::Record(RecordType {
                fields: fields
                    .iter()
                    .map(|f| RecordField {
                        key: f.key.clone(),
                        ty: self.apply_type(&f.ty),
                        required: f.required,
                    })
                    .collect(),
                location: *location,
            }),
            Type::Proc(proc_type) => Type::Proc(Box::new(ProcType {
                function: self.apply_function_type(&proc_type.function),
                self_type: proc_type
                    .self_type
                    .as_ref()
                    .map(|st| Box::new(self.apply_type(st))),
                block: proc_type.block.as_ref().map(|b| self.apply_block(b)),
                location: proc_type.location,
            })),
            Type::Literal(_) | Type::Base(_) => ty.clone(),
        }
    }

    /// Mirrors `RBS::Types::Function#sub` / `UntypedFunction#sub` —
    /// applies the substitution element-wise across every type slot of
    /// the function signature.
    pub fn apply_function_type(&self, ft: &Function) -> Function {
        if self.is_empty() {
            return ft.clone();
        }
        match ft {
            Function::Typed(f) => Function::Typed(FunctionType {
                required_positionals: f
                    .required_positionals
                    .iter()
                    .map(|p| self.apply_param(p))
                    .collect(),
                optional_positionals: f
                    .optional_positionals
                    .iter()
                    .map(|p| self.apply_param(p))
                    .collect(),
                rest_positionals: f
                    .rest_positionals
                    .as_ref()
                    .map(|p| Box::new(self.apply_param(p))),
                trailing_positionals: f
                    .trailing_positionals
                    .iter()
                    .map(|p| self.apply_param(p))
                    .collect(),
                required_keywords: f
                    .required_keywords
                    .iter()
                    .map(|kp| self.apply_keyword_param(kp))
                    .collect(),
                optional_keywords: f
                    .optional_keywords
                    .iter()
                    .map(|kp| self.apply_keyword_param(kp))
                    .collect(),
                rest_keywords: f
                    .rest_keywords
                    .as_ref()
                    .map(|p| Box::new(self.apply_param(p))),
                return_type: Box::new(self.apply_type(&f.return_type)),
            }),
            Function::Untyped(u) => Function::Untyped(UntypedFunctionType {
                return_type: Box::new(self.apply_type(&u.return_type)),
            }),
        }
    }

    fn apply_param(&self, p: &FunctionParam) -> FunctionParam {
        FunctionParam {
            ty: Box::new(self.apply_type(&p.ty)),
            name: p.name,
            location: p.location,
        }
    }

    fn apply_keyword_param(&self, kp: &KeywordParam) -> KeywordParam {
        KeywordParam {
            name: kp.name,
            param: self.apply_param(&kp.param),
        }
    }

    /// Mirrors `RBS::Types::Block#sub`.
    pub fn apply_block(&self, b: &BlockType) -> BlockType {
        BlockType {
            required: b.required,
            function: self.apply_function_type(&b.function),
            self_type: b.self_type.as_ref().map(|t| Box::new(self.apply_type(t))),
        }
    }

    /// Mirrors rbs's `member.overloads.map { |o| o.sub(subst) }` —
    /// applies the substitution to each overload's `method_type` while
    /// keeping the per-overload annotation slot untouched.
    pub fn apply_overloads(
        &self,
        overloads: &[MethodDefinitionOverload],
        names: &NameTable,
    ) -> Vec<MethodDefinitionOverload> {
        overloads
            .iter()
            .map(|o| MethodDefinitionOverload {
                method_type: self.apply_method_type(&o.method_type, names),
                annotations: o.annotations.clone(),
            })
            .collect()
    }

    /// Mirrors `RBS::MethodType#sub`. Rewrites the function signature, the
    /// block, and the bounds of method-level type params after removing
    /// method-scope names from the substitution (shadowing step).
    pub fn apply_method_type(&self, mt: &MethodType, _names: &NameTable) -> MethodType {
        if self.is_empty() {
            return mt.clone();
        }
        // rbs: sub = s.without(*type_param_names)
        let shadowed: Vec<Symbol> = mt.type_params.iter().map(|tp| tp.name).collect();
        let inner = if shadowed.is_empty() {
            self.clone()
        } else {
            self.without(&shadowed)
        };
        // rbs: return self if sub.empty?
        if inner.is_empty() {
            return mt.clone();
        }
        let type_params = mt
            .type_params
            .iter()
            .map(|tp| TypeParam {
                name: tp.name,
                variance: tp.variance,
                upper_bound: tp.upper_bound.as_ref().map(|b| inner.apply_type(b)),
                lower_bound: tp.lower_bound.as_ref().map(|b| inner.apply_type(b)),
                default_type: tp.default_type.as_ref().map(|b| inner.apply_type(b)),
                unchecked: tp.unchecked,
                location: tp.location,
            })
            .collect();
        MethodType {
            type_params,
            function: inner.apply_function_type(&mt.function),
            block: mt.block.as_ref().map(|b| inner.apply_block(b)),
            location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variance;
    use crate::name::NameTable;
    use crate::type_name::TypeName;

    fn class_tp(name: Symbol) -> TypeParam {
        TypeParam {
            name,
            variance: Variance::Invariant,
            upper_bound: None,
            lower_bound: None,
            default_type: None,
            unchecked: false,
            location: None,
        }
    }

    fn type_var(name: Symbol) -> Type {
        Type::Variable(VariableType {
            name,
            location: None,
        })
    }

    fn fp(ty: Type) -> FunctionParam {
        FunctionParam {
            ty: Box::new(ty),
            name: None,
            location: None,
        }
    }

    fn class_instance(name: TypeName, args: Vec<Type>) -> Type {
        Type::ClassInstance(ClassInstanceType {
            name,
            args,
            location: None,
        })
    }

    #[test]
    fn apply_type_rewrites_matching_type_variable() {
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let rewritten = subst.apply_type(&type_var(t));
        assert_eq!(rewritten, type_var(a));
    }

    #[test]
    fn apply_type_recurses_into_class_instance_args() {
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let array_name = names.parse_type_name("::Array");
        let array_t = class_instance(array_name, vec![type_var(t)]);
        let rewritten = subst.apply_type(&array_t);

        assert_eq!(rewritten, class_instance(array_name, vec![type_var(a)]));
    }

    #[test]
    fn apply_type_leaves_unmapped_type_variable_unchanged() {
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let u = names.intern_symbol("U");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let rewritten = subst.apply_type(&type_var(u));
        assert_eq!(rewritten, type_var(u));
    }

    #[test]
    fn without_drops_named_entries() {
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let pruned = subst.without(&[t]);
        assert!(pruned.is_empty());
    }

    #[test]
    fn apply_type_preserves_source_location_when_rewriting_typevariable() {
        use crate::location::LocationRange;
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let loc = LocationRange {
            start_char: 5,
            start_byte: 5,
            end_char: 6,
            end_byte: 6,
        };
        let original = Type::Variable(VariableType {
            name: t,
            location: Some(loc),
        });
        let rewritten = subst.apply_type(&original);
        assert_eq!(
            rewritten,
            Type::Variable(VariableType {
                name: a,
                location: Some(loc),
            }),
            "TypeVariable rename keeps the original `location` so diagnostics remain pointable"
        );
    }

    #[test]
    fn apply_method_type_rewrites_function_signature_with_subst() {
        use crate::ast::method_type::MethodType;
        use crate::ast::types::{Function, FunctionType};

        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let u = names.intern_symbol("U");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let mt = MethodType {
            type_params: vec![TypeParam::new(u)],
            function: Function::Typed(FunctionType {
                required_positionals: vec![fp(type_var(t))],
                optional_positionals: vec![],
                rest_positionals: None,
                trailing_positionals: vec![],
                required_keywords: vec![],
                optional_keywords: vec![],
                rest_keywords: None,
                return_type: Box::new(type_var(t)),
            }),
            block: None,
            location: None,
        };
        let rewritten = subst.apply_method_type(&mt, &names);
        let Function::Typed(f) = rewritten.function else {
            panic!("apply_method_type must preserve Function::Typed");
        };
        assert_eq!(f.required_positionals, vec![fp(type_var(a))]);
        assert_eq!(*f.return_type, type_var(a));
        assert_eq!(
            rewritten.type_params.len(),
            1,
            "method-level type_params are preserved (name `U`)"
        );
    }

    #[test]
    fn apply_overloads_rewrites_each_overload_method_type() {
        use crate::ast::members::MethodDefinitionOverload;
        use crate::ast::method_type::MethodType;
        use crate::ast::types::{Function, FunctionType};

        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let overloads = vec![MethodDefinitionOverload {
            method_type: MethodType {
                type_params: vec![],
                function: Function::Typed(FunctionType {
                    required_positionals: vec![fp(type_var(t))],
                    optional_positionals: vec![],
                    rest_positionals: None,
                    trailing_positionals: vec![],
                    required_keywords: vec![],
                    optional_keywords: vec![],
                    rest_keywords: None,
                    return_type: Box::new(type_var(t)),
                }),
                block: None,
                location: None,
            },
            annotations: vec![],
        }];
        let rewritten = subst.apply_overloads(&overloads, &names);
        assert_eq!(rewritten.len(), 1);
        let Function::Typed(f) = &rewritten[0].method_type.function else {
            panic!("apply_overloads preserves Function::Typed");
        };
        assert_eq!(f.required_positionals, vec![fp(type_var(a))]);
        assert_eq!(*f.return_type, type_var(a));
    }

    /// class subst {T → A}, method-level [T] (T) -> T
    /// T is shadowed by method param: function body must remain T (not A).
    #[test]
    fn apply_method_type_shadows_same_name_method_param() {
        use crate::ast::method_type::MethodType;
        use crate::ast::types::{Function, FunctionType};

        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let mt = MethodType {
            type_params: vec![TypeParam::new(t)],
            function: Function::Typed(FunctionType {
                required_positionals: vec![fp(type_var(t))],
                optional_positionals: vec![],
                rest_positionals: None,
                trailing_positionals: vec![],
                required_keywords: vec![],
                optional_keywords: vec![],
                rest_keywords: None,
                return_type: Box::new(type_var(t)),
            }),
            block: None,
            location: None,
        };
        let rewritten = subst.apply_method_type(&mt, &names);
        let Function::Typed(f) = rewritten.function else {
            panic!();
        };
        assert_eq!(
            f.required_positionals,
            vec![fp(type_var(t))],
            "method-level T must not be rewritten by class subst"
        );
        assert_eq!(
            *f.return_type,
            type_var(t),
            "method-level T must not be rewritten by class subst"
        );
    }

    /// class subst {T → A}, method-level [U] (T) -> U
    /// T is NOT shadowed (no method param named T): T rewrites to A.
    /// U is shadowed: U stays U (method-level param).
    #[test]
    fn apply_method_type_shadows_only_matching_method_param() {
        use crate::ast::method_type::MethodType;
        use crate::ast::types::{Function, FunctionType};

        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let u = names.intern_symbol("U");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let mt = MethodType {
            type_params: vec![TypeParam::new(u)],
            function: Function::Typed(FunctionType {
                required_positionals: vec![fp(type_var(t))],
                optional_positionals: vec![],
                rest_positionals: None,
                trailing_positionals: vec![],
                required_keywords: vec![],
                optional_keywords: vec![],
                rest_keywords: None,
                return_type: Box::new(type_var(u)),
            }),
            block: None,
            location: None,
        };
        let rewritten = subst.apply_method_type(&mt, &names);
        let Function::Typed(f) = rewritten.function else {
            panic!();
        };
        assert_eq!(
            f.required_positionals,
            vec![fp(type_var(a))],
            "class-level T must rewrite to A when not shadowed"
        );
        assert_eq!(
            *f.return_type,
            type_var(u),
            "method-level U is not in class subst so stays U"
        );
        assert_eq!(rewritten.type_params.len(), 1);
    }

    /// class subst {T → A}, method-level [U < T] () -> U
    /// upper_bound T must rewrite to A via shadowed sub applied to bounds.
    #[test]
    fn apply_method_type_rewrites_method_param_bound_via_shadowed_sub() {
        use crate::ast::method_type::MethodType;
        use crate::ast::types::{Function, FunctionType};

        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let a = names.intern_symbol("A");
        let u = names.intern_symbol("U");
        let subst = Substitution::build(&[class_tp(t)], &[class_tp(a)]);

        let mt = MethodType {
            type_params: vec![TypeParam {
                name: u,
                variance: Variance::Invariant,
                upper_bound: Some(type_var(t)),
                lower_bound: None,
                default_type: None,
                unchecked: false,
                location: None,
            }],
            function: Function::Typed(FunctionType {
                required_positionals: vec![],
                optional_positionals: vec![],
                rest_positionals: None,
                trailing_positionals: vec![],
                required_keywords: vec![],
                optional_keywords: vec![],
                rest_keywords: None,
                return_type: Box::new(type_var(u)),
            }),
            block: None,
            location: None,
        };
        let rewritten = subst.apply_method_type(&mt, &names);
        assert_eq!(rewritten.type_params.len(), 1);
        assert_eq!(
            rewritten.type_params[0].upper_bound.as_ref(),
            Some(&type_var(a)),
            "upper_bound T must be rewritten to A by the shadowed sub"
        );
    }

    #[test]
    fn empty_subst_is_a_clone_on_apply() {
        let names = NameTable::new();
        let t = names.intern_symbol("T");
        let subst = Substitution::empty();

        let rewritten = subst.apply_type(&type_var(t));
        assert_eq!(rewritten, type_var(t));
    }
}
