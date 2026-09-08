//! Per-call receiver-context method resolver (ADR-0021).
//!
//! Dispatches on the receiver `Ty` and returns the method to apply at
//! the call site, alongside the type bindings and the receiver class.
//! This is the single home for receiver-class-dependent rewrites: a
//! `Kernel#class` call on a `ClassInstance(C)` receiver narrows the
//! return type from `Class` to `singleton(C)` (Steep's
//! `replace_kernel_class` equivalent).
//!
//! Distinct from `crate::resolver`, which ports rbs's name resolution
//! pipeline (`lib/rbs/resolver/`). This module is the Steep
//! `Interface::Builder` analogue and lives in the type_checker layer
//! to make the namespace separation explicit.

use rustc_hash::FxHashMap;

use crate::definition::Method;
use crate::definition_builder::{
    ConsultationView, expand_alias, resolve_singleton_method_with_type_name_args,
};
use crate::name::Symbol;
use crate::subtyping::record_key_union;
use crate::type_name::TypeName;
use crate::type_param::TypeVarKey;
use crate::types::{FunctionType, RecordKey, Ty, Type};

/// Result of a successful per-call resolution.
pub struct ResolvedMethod {
    /// The method to apply at the call site. Cloned from the
    /// `DefinitionBuilder` so receiver-class-dependent rewrites (e.g.
    /// `Kernel#class` -> `singleton(C)`) can be baked in without
    /// touching the underlying definition.
    pub method: Method,
    /// Type-parameter bindings collected along the ancestor walk.
    pub bindings: FxHashMap<TypeVarKey, Ty>,
    /// The receiver class identity. Passed back so call-site diagnostics
    /// can name the actual receiver (`receiver_class` field on
    /// `CallTarget::Method`).
    pub receiver_class: TypeName,
}

/// Look up a method given a receiver type and a method-name symbol.
///
/// Returns `None` when:
/// - the receiver kind is not yet handled (Union, Self, Var, Bool —
///   future slices), or
/// - the method itself does not resolve.
///
/// Handled receiver kinds: `ClassInstance`, `ClassSingleton`,
/// `Literal`, `Tuple`, `Record`, `Nil`, `Proc`, `Alias`, `Intersection`.
/// The composite types and `Nil` widen to their class-instance
/// counterparts (`Nil` → `::NilClass`, `Proc` → `::Proc`) and recurse.
/// `Alias` is peeled via `expand_alias` and the expansion is dispatched
/// recursively; a cyclic alias (`expand_alias` returns the alias
/// unchanged after `ALIAS_EXPANSION_LIMIT` hops) returns `None` rather
/// than recursing forever.
///
/// The `Proc` widen resolves methods against the RBS `::Proc` class,
/// then `overwrite_proc_call` replaces `.call` / `.[]` with the
/// receiver's own signature (Steep `proc_shape`).
///
/// No `untyped` fallback: lookup failure is reported as `None` so
/// callers can surface the underlying diagnostic instead of having it
/// swallowed by gradual typing.
pub fn lookup_method(
    env: ConsultationView,
    receiver: Ty,
    method_name: Symbol,
) -> Option<ResolvedMethod> {
    let resolved = env.types().resolve(receiver);
    match resolved {
        Type::ClassInstance { name, args } => {
            let (method, bindings) =
                env.lookup_instance_method_with_args(name, args, method_name)?;
            let method = maybe_rewrite_kernel_class(env, method, method_name, name);
            Some(ResolvedMethod {
                method,
                bindings,
                receiver_class: *name,
            })
        }
        Type::ClassSingleton { name } => {
            let (method, bindings) =
                resolve_singleton_method_with_type_name_args(env, *name, method_name)?;
            Some(ResolvedMethod {
                method,
                bindings,
                receiver_class: *name,
            })
        }
        Type::Interface { name, args } => {
            let (method, bindings) =
                env.lookup_interface_method_with_args(name, args, method_name)?;
            Some(ResolvedMethod {
                method,
                bindings,
                receiver_class: *name,
            })
        }
        Type::Literal(lit) => {
            let widened = env.class_instance_type(*lit.class_typename(env.names().builtins()));
            lookup_method(env, widened, method_name)
        }
        Type::Tuple(members) => {
            let widened = widen_tuple_to_array(env, members);
            lookup_method(env, widened, method_name)
        }
        Type::Record { fields } => {
            let widened = widen_record_to_hash(env, fields);
            lookup_method(env, widened, method_name)
        }
        Type::Nil => {
            let widened = env.class_instance_type(env.names().builtins().nil_class);
            lookup_method(env, widened, method_name)
        }
        Type::Proc { type_, block, .. } => {
            // Steep `interface/builder.rb:157-159` widens `Types::Proc` to
            // `object_shape(AST::Builtin::Proc.module_name)`, then
            // `proc_shape` (`builder.rb:718`) overwrites `.call` / `.[]`
            // with a single overload built from the proc's own signature.
            // Methods like `lambda?` / `arity` / `curry` keep resolving
            // through the RBS `::Proc` class.
            let type_ = type_.clone();
            let block = block.clone();
            let widened = env.class_instance_type(env.names().builtins().proc);
            let resolved = lookup_method(env, widened, method_name)?;
            Some(overwrite_proc_call(
                env,
                resolved,
                method_name,
                type_,
                block,
            ))
        }
        Type::Intersection(members) => {
            // Steep `intersection_shape.methods.merge!` keeps the new
            // (last) entry on collision. Iterating in reverse and
            // returning on the first hit therefore lets the rightmost
            // member's signature win. Members that lack the method
            // simply fall through; the call succeeds as long as at
            // least one member has it (any-of dispatch).
            for member in members.iter().rev() {
                if let Some(resolved) = lookup_method(env, *member, method_name) {
                    return Some(resolved);
                }
            }
            None
        }
        Type::Alias { .. } => {
            // Steep `Interface::Builder#raw_shape` expands `Name::Alias` to its
            // body and recurses; method lookup happens on the underlying class.
            // Cyclic aliases (`type a = b; type b = a`) trip
            // `ALIAS_EXPANSION_LIMIT` in `expand_alias` and stay as an alias —
            // bail to `None` so the caller routes the call to its existing
            // conservative path rather than re-entering this arm forever.
            let expanded = expand_alias(env, receiver);
            if matches!(env.types().resolve(expanded), Type::Alias { .. }) {
                return None;
            }
            lookup_method(env, expanded, method_name)
        }
        _ => None,
    }
}

/// Widen a tuple to `Array[element_union]` for method lookup.
pub(super) fn widen_tuple_to_array(env: ConsultationView, members: &[Ty]) -> Ty {
    let types = env.types();
    let elem = match members {
        [] => Ty::BOTTOM,
        [single] => *single,
        _ => types.intern(Type::Union(members.to_vec())),
    };
    let array_name = env.names().builtins().array;
    types.intern(Type::ClassInstance {
        name: array_name,
        args: vec![elem],
    })
}

/// Widen a record to `Hash[key_class_union, value_union]` for method lookup.
pub(super) fn widen_record_to_hash(env: ConsultationView, fields: &[(RecordKey, Ty, bool)]) -> Ty {
    let types = env.types();
    let key_ty = record_key_union(fields, env);
    let field_tys: Vec<Ty> = fields.iter().map(|(_, ty, _)| *ty).collect();
    let value_ty = match field_tys.as_slice() {
        [] => Ty::BOTTOM,
        [single] => *single,
        _ => types.intern(Type::Union(field_tys)),
    };
    let hash_name = env.names().builtins().hash;
    types.intern(Type::ClassInstance {
        name: hash_name,
        args: vec![key_ty, value_ty],
    })
}

/// Overwrite `.call` / `.[]` on a `Type::Proc` receiver with a single
/// overload built from the proc's own signature.
///
/// Mirrors Steep `Interface::Builder#proc_shape`
/// (`lib/steep/interface/builder.rb:718-739`): the `::Proc` object shape
/// is merged in first, then `methods[:[]]` and `methods[:call]` are
/// *replaced* by `MethodType.new(type_params: [], type: proc.type,
/// block: proc.block)`. Every other name (`lambda?`, `arity`, `yield`,
/// `===`, ...) keeps the class-declared signature.
///
/// `self_type` (the `[self: T]` binding) is deliberately not carried
/// onto the overload: Steep passes only `proc.type` and `proc.block`,
/// and `MethodType` has no slot for a self binding.
///
/// The synthesized `TypeDef` reuses the `member` / `defined_in` /
/// `implemented_in` identity of the `::Proc` def it replaces, so
/// `super_method` resolution and annotation reads keep working; only
/// the `MethodType` differs.
fn overwrite_proc_call(
    env: ConsultationView,
    resolved: ResolvedMethod,
    method_name: Symbol,
    type_: FunctionType,
    block: Option<crate::types::Block>,
) -> ResolvedMethod {
    let names = env.names();
    if method_name != names.intern_symbol("call") && method_name != names.intern_symbol("[]") {
        return resolved;
    }
    let ResolvedMethod {
        mut method,
        bindings,
        receiver_class,
    } = resolved;
    let Some(mut td) = method.defs.first().cloned() else {
        return ResolvedMethod {
            method,
            bindings,
            receiver_class,
        };
    };
    td.type_ = crate::types::MethodType {
        type_params: vec![],
        type_,
        block,
    };
    method.defs = vec![td];
    ResolvedMethod {
        method,
        bindings,
        receiver_class,
    }
}

/// Rewrite `Kernel#class`'s return type from `Class` to
/// `singleton(receiver_class)` for every def site whose `defined_in` is
/// `::Kernel`. User-overridden `def class` defs keep their original
/// declared return type because the gate is applied per-def.
///
/// Mirrors Steep `Interface::Builder#replace_kernel_class`
/// (`lib/steep/interface/builder.rb:845`), which iterates `TypeDef`s
/// and rewrites only the Kernel-owned overload. A class that mixes a
/// local `def class: () -> ...` overload with the inherited Kernel one
/// (via the `...` trailer) thus narrows the Kernel-side overload while
/// preserving the user-declared return type on the local overload.
///
/// The rewrite has to happen at lookup time, not at definition build,
/// because `Kernel`'s own definition cannot know who included it
/// (ADR-0021 Decision).
fn maybe_rewrite_kernel_class(
    env: ConsultationView,
    method: Method,
    method_name: Symbol,
    receiver_class: &TypeName,
) -> Method {
    let class_sym = env.names().intern_symbol("class");
    if method_name != class_sym {
        return method;
    }
    let kernel = &env.names().builtins().kernel;
    if !method.defs.iter().any(|td| &td.defined_in == kernel) {
        return method;
    }
    let singleton_ty = env.types().class_singleton(*receiver_class);
    let mut rewritten = method;
    for td in &mut rewritten.defs {
        if &td.defined_in != kernel {
            continue;
        }
        let original = &td.type_;
        let new_function = match &original.type_ {
            FunctionType::Typed(f) => {
                let mut f = f.clone();
                f.return_type = singleton_ty;
                FunctionType::Typed(f)
            }
            FunctionType::Untyped(u) => {
                let mut u = u.clone();
                u.return_type = singleton_ty;
                FunctionType::Untyped(u)
            }
        };
        td.type_ = crate::types::MethodType {
            type_params: original.type_params.clone(),
            type_: new_function,
            block: original.block.clone(),
        };
    }
    rewritten
}
