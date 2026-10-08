use rustc_hash::FxHashMap;

use crate::name::Symbol;
use crate::rbs_raw::TypeParamVariance;
use crate::substitution::Substitution;
use crate::type_name::TypeName;
use crate::types::{
    Block, Function, FunctionType, MethodType, RecordKey, Ty, Type, TypeTable, UntypedFunction,
};


/// Map from a raw RBS-written type-param name (e.g. `T`) to the full
/// [`TypeVarKey`] it should be lowered to.
///
/// The key is what the lookup sees in source — the raw symbol as written
/// at the use site. The value is what the type variable identity actually
/// resolves to: the canonical raw plus its declaration scope.
///
/// Two raws can differ: when a re-opened declaration uses a different
/// spelling for the same param slot (`class Foo[A]` then `class Foo[B]`),
/// the second `B` is rewritten to the primary `A` via this map, so every
/// downstream reference shares one identity (per the primary).
///
/// Walking an RBS type (`node_to_type`) consults this map when it hits an
/// `AstType::TypeVariable` to attach both the canonical raw and the
/// declaration scope. Lookups miss for stray refs outside any declared
/// scope — those land in [`TypeVarScope::Free`].
pub type TypeParamScope = FxHashMap<Symbol, TypeVarKey>;

/// Scope a type variable belongs to. Each variant identifies a distinct
/// declaration site so that `T` written in one declaration cannot alias `T`
/// written in another (per ADR-0023).
///
/// `Method` carries `kind` and `overload_index` so that `def a` /
/// `def self.a` and per-overload `[T]`s are distinguishable identities, even
/// when they share the same raw spelling.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TypeVarScope {
    /// Stray reference outside any declared scope. Behaves like a free
    /// variable; treated as `untyped` by downstream consumers that do not
    /// have a binding.
    Free,
    Class(TypeName),
    Interface(TypeName),
    Alias(TypeName),
    Method {
        class: TypeName,
        method: Symbol,
        kind: MethodKind,
        overload_index: u16,
    },
}

/// Identifies whether a method-level type param was declared on an instance
/// (`def a`) or a singleton (`def self.a`) method. The two share table
/// space in different `Definition` slots, but the `[T]` declared on each
/// must still be a distinct variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MethodKind {
    Instance,
    Singleton,
}

/// Full identity of a type variable: the raw user-written name plus the
/// scope that disambiguates it.
///
/// Used as the payload of [`Type::TypeVariable`] and as the key type for
/// substitution mappings and per-call bindings maps. Two `TypeVarKey`s
/// compare equal iff they came from the same declaration site, so a `T`
/// from one method does not collide with a `T` from another.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TypeVarKey {
    pub raw: Symbol,
    pub scope: TypeVarScope,
}

impl TypeVarKey {
    pub fn new(raw: Symbol, scope: TypeVarScope) -> Self {
        Self { raw, scope }
    }

    /// Build a `Free`-scoped key — used when a type variable reference is
    /// not bound by any enclosing declaration scope.
    pub fn free(raw: Symbol) -> Self {
        Self {
            raw,
            scope: TypeVarScope::Free,
        }
    }
}

/// Mirrors RBS's `:invariant` / `:covariant` / `:contravariant` symbols.
/// Default is `Invariant` — matches RBS when the `in` / `out` keyword is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Variance {
    Invariant,
    Covariant,
    Contravariant,
}

impl Variance {
    pub fn from_rbs(v: TypeParamVariance) -> Self {
        match v {
            TypeParamVariance::Invariant => Variance::Invariant,
            TypeParamVariance::Covariant => Variance::Covariant,
            TypeParamVariance::Contravariant => Variance::Contravariant,
        }
    }

    pub fn from_ast(v: crate::ast::Variance) -> Self {
        match v {
            crate::ast::Variance::Invariant => Variance::Invariant,
            crate::ast::Variance::Covariant => Variance::Covariant,
            crate::ast::Variance::Contravariant => Variance::Contravariant,
        }
    }
}

/// Type parameter declaration attached to a class, module, interface, or
/// type alias. Shape mirrors `RBS::AST::TypeParam` so that future phases
/// (bounds, default, unchecked) can populate the existing slots without
/// restructuring call sites.
///
/// Phase B populates only `name` and `variance`. Other fields are reserved
/// and always carry their zero value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeParam {
    pub name: TypeVarKey,
    pub variance: Variance,
    pub upper_bound: Option<Ty>,
    pub lower_bound: Option<Ty>,
    pub default_type: Option<Ty>,
    pub unchecked: bool,
}

impl TypeParam {
    pub fn new(name: TypeVarKey, variance: Variance) -> Self {
        TypeParam {
            name,
            variance,
            upper_bound: None,
            lower_bound: None,
            default_type: None,
            unchecked: false,
        }
    }

    /// Test-friendly constructor: build a `TypeParam` whose name is a
    /// `Free`-scoped `TypeVarKey`. Production code should always pick the
    /// correct scope variant via [`TypeParam::new`].
    #[cfg(test)]
    pub fn free(raw: Symbol, variance: Variance) -> Self {
        Self::new(TypeVarKey::free(raw), variance)
    }
}

/// Fill missing type arguments using each param's `default_type`, falling
/// back to `UNTYPED` when no default exists. Length is always `params.len()`.
///
/// More permissive than RBS gem's `TypeParam.normalize_args`, which leaves
/// args untouched when `args.len() < min_count`. Here each param is resolved
/// independently so callers always get a complete `Vec<Ty>`, which keeps
/// downstream type operations (subtype, lookup) from hitting arity-shaped
/// special cases.
///
/// Defaults may reference earlier params (`class Map[K, V = K]`) — each
/// resolved slot is added to the substitution before the next param's
/// default is substituted.
///
/// Excess args (`args.len() > params.len()`) are dropped; arity diagnostics
/// are the responsibility of `validate_applied_type_args`, not this helper.
/// Walk a `Ty` and apply `apply_defaults` to every `ClassInstance`,
/// `Interface`, and `Alias` args list. Traverses into all composite type
/// forms (Union, Optional, Tuple, Record, Proc, etc.) so nested generic
/// references (e.g. `Array[Cache[Integer]]`) are normalized all the way
/// down.
///
/// `params_of(name)` returns the declared type parameters for a class,
/// module, interface, or alias — or `None` when the target is unknown
/// (built-in types the user hasn't loaded, etc.). Unknown targets are
/// traversed but their args are left as-is.
pub fn normalize_class_instance_defaults<F>(ty: Ty, types: &TypeTable, params_of: &F) -> Ty
where
    F: Fn(&TypeName) -> Option<Vec<TypeParam>>,
{
    let walk = |t: Ty| normalize_class_instance_defaults(t, types, params_of);
    let fill = |name: &TypeName, args: &[Ty]| -> Vec<Ty> {
        let walked: Vec<Ty> = args.iter().map(|&a| walk(a)).collect();
        if let Some(params) = params_of(name) {
            apply_defaults(&params, &walked, types)
        } else {
            walked
        }
    };
    match types.resolve(ty) {
        Type::ClassInstance { name, args } => types.intern(Type::ClassInstance {
            args: fill(name, args),
            name: *name,
        }),
        Type::Interface { name, args } => types.intern(Type::Interface {
            args: fill(name, args),
            name: *name,
        }),
        Type::Alias { name, args } => types.intern(Type::Alias {
            args: fill(name, args),
            name: *name,
        }),
        Type::Union(members) => {
            types.intern(Type::Union(members.iter().map(|&m| walk(m)).collect()))
        }
        Type::Intersection(members) => types.intern(Type::Intersection(
            members.iter().map(|&m| walk(m)).collect(),
        )),
        Type::Optional(inner) => types.intern(Type::Optional(walk(*inner))),
        Type::Tuple(members) => {
            types.intern(Type::Tuple(members.iter().map(|&m| walk(m)).collect()))
        }
        Type::Record { fields } => {
            let new_fields: Vec<(RecordKey, Ty, bool)> = fields
                .iter()
                .map(|(k, v, req)| (k.clone(), walk(*v), *req))
                .collect();
            types.intern(Type::Record { fields: new_fields })
        }
        Type::Proc {
            type_,
            self_type,
            block,
        } => {
            let new_func = normalize_function_type_defaults(type_, types, params_of);
            let new_self = self_type.map(walk);
            let new_block = block
                .as_ref()
                .map(|b| normalize_block_defaults(b, types, params_of));
            types.intern(Type::Proc {
                type_: new_func,
                self_type: new_self,
                block: new_block,
            })
        }
        _ => ty,
    }
}

fn normalize_function_type_defaults<F>(
    ft: &FunctionType,
    types: &TypeTable,
    params_of: &F,
) -> FunctionType
where
    F: Fn(&TypeName) -> Option<Vec<TypeParam>>,
{
    match ft {
        FunctionType::Typed(f) => {
            let walk = |t: Ty| normalize_class_instance_defaults(t, types, params_of);
            FunctionType::Typed(Function {
                required_positionals: f.required_positionals.iter().copied().map(walk).collect(),
                optional_positionals: f.optional_positionals.iter().copied().map(walk).collect(),
                rest_positional: f.rest_positional.map(walk),
                trailing_positionals: f.trailing_positionals.iter().copied().map(walk).collect(),
                required_keywords: f
                    .required_keywords
                    .iter()
                    .map(|(k, t)| (k.clone(), walk(*t)))
                    .collect(),
                optional_keywords: f
                    .optional_keywords
                    .iter()
                    .map(|(k, t)| (k.clone(), walk(*t)))
                    .collect(),
                rest_keyword: f.rest_keyword.map(walk),
                return_type: walk(f.return_type),
            })
        }
        FunctionType::Untyped(u) => FunctionType::Untyped(UntypedFunction {
            return_type: normalize_class_instance_defaults(u.return_type, types, params_of),
        }),
    }
}

/// Normalize defaults in every `Ty` referenced by a `MethodType`: param
/// types (all positional/keyword slots), return type, and the optional
/// block's function type and self_type binding. The method's own
/// `type_params` are left alone — defaults on method-level type params
/// are an inference concern, handled separately.
pub fn normalize_method_type_defaults<F>(
    mt: &MethodType,
    types: &TypeTable,
    params_of: &F,
) -> MethodType
where
    F: Fn(&TypeName) -> Option<Vec<TypeParam>>,
{
    MethodType {
        type_params: mt.type_params.clone(),
        type_: normalize_function_type_defaults(&mt.type_, types, params_of),
        block: mt
            .block
            .as_ref()
            .map(|b| normalize_block_defaults(b, types, params_of)),
    }
}

fn normalize_block_defaults<F>(b: &Block, types: &TypeTable, params_of: &F) -> Block
where
    F: Fn(&TypeName) -> Option<Vec<TypeParam>>,
{
    Block {
        required: b.required,
        type_: normalize_function_type_defaults(&b.type_, types, params_of),
        self_type: b
            .self_type
            .map(|t| normalize_class_instance_defaults(t, types, params_of)),
    }
}

/// Fill missing type arguments using each param's `default_type`, falling
/// back to `UNTYPED` when no default exists.
///
/// More permissive than RBS gem's `TypeParam.normalize_args`, which leaves
/// args untouched when `args.len() < min_count`. Here each param is resolved
/// independently so callers always get a fully-shaped `Vec<Ty>` for the
/// first `params.len()` slots, which keeps downstream type operations
/// (subtype, lookup) from hitting arity-shaped special cases.
///
/// Defaults may reference earlier params (`class Map[K, V = K]`) — each
/// resolved slot is added to the substitution before the next param's
/// default is substituted.
///
/// Excess args (`args.len() > params.len()`) are **preserved at the tail**:
/// this helper never drops user-written type arguments. Arity diagnostics
/// are the responsibility of `validate_applied_type_args`, and they need
/// the original args to report excess. Returning a length of
/// `args.len().max(params.len())` keeps the two concerns separable
/// regardless of which pass runs first.
pub fn apply_defaults(params: &[TypeParam], args: &[Ty], types: &TypeTable) -> Vec<Ty> {
    let out_len = args.len().max(params.len());
    let mut result: Vec<Ty> = Vec::with_capacity(out_len);
    let mut subst = Substitution::new();
    for (i, param) in params.iter().enumerate() {
        let ty = if let Some(&arg) = args.get(i) {
            arg
        } else if let Some(default) = param.default_type {
            subst.apply(default, types)
        } else {
            Ty::UNTYPED
        };
        subst.mapping.insert(param.name.clone(), ty);
        result.push(ty);
    }
    if args.len() > params.len() {
        result.extend_from_slice(&args[params.len()..]);
    }
    result
}

/// Strict counterpart of [`apply_defaults`] for ancestor / mixin args, mirroring
/// rbs `RBS::AST::TypeParam.normalize_args` (`lib/rbs/ast/type_param.rb`): pad
/// missing args from each param's `default_type` only when the supplied count
/// is in `[min_count, params.len()]`, where `min_count` is the count of
/// default-less params. Outside that window the args are returned unchanged
/// so `validate_applied_type_args` can surface a
/// `MixinTypeArgumentArityMismatch` against the original shape.
///
/// `min_count` counts *every* default-less param, matching rbs's
/// `params.count { _1.default_type.nil? }` (`ast/type_param.rb:201` and
/// `errors.rb:84`). rbs's `TypeParam.validate` enforces that defaults are
/// trailing, so `count`-of-default-less is identical to leading-default-less
/// for valid input; we mirror the rbs formula literally to stay defensive if
/// non-trailing defaults ever leak past validation.
///
/// Unlike [`apply_defaults`], this never substitutes `UNTYPED` for an unfilled
/// no-default slot. The arity gate above guarantees those slots already have
/// a user-supplied arg or a `default_type`, so the inner branch is unreachable
/// in well-formed inputs; the `UNTYPED` fallback exists only to keep the
/// helper total without panicking on malformed callers.
pub fn pad_ancestor_args(params: &[TypeParam], args: &[Ty], types: &TypeTable) -> Vec<Ty> {
    let min_count = params.iter().filter(|p| p.default_type.is_none()).count();
    if args.len() < min_count || args.len() > params.len() {
        return args.to_vec();
    }
    let mut result: Vec<Ty> = Vec::with_capacity(params.len());
    let mut subst = Substitution::new();
    for (i, param) in params.iter().enumerate() {
        let ty = if let Some(&arg) = args.get(i) {
            arg
        } else if let Some(default) = param.default_type {
            subst.apply(default, types)
        } else {
            Ty::UNTYPED
        };
        subst.mapping.insert(param.name.clone(), ty);
        result.push(ty);
    }
    result
}
