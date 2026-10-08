//! A total order on interned types that reads only their content.
//!
//! `Ty` ids are handed out in intern order, which depends on which file was
//! checked first (and, under parallel check, on thread interleaving). Union
//! and intersection members are sorted by this order instead, so the member
//! sequence — and every consumer that picks "the first member" or "the last
//! member" — is the same in every run.
//!
//! The leaves are content-addressed (`TypeName`, `Symbol` are xxh3 ids;
//! strings and bools compare as themselves), and children are compared
//! recursively, so no intern id is ever read. Interning keeps one `Ty` per
//! structurally distinct `Type`, so the order returns `Equal` exactly when
//! the two handles are the same `Ty`.

use std::cmp::Ordering;

use super::{Block, Function, FunctionType, Literal, Ty, Type, TypeTable};
use crate::type_param::{MethodKind, TypeVarScope};

/// Compare two types by content. See the module docs.
pub fn cmp_by_content(a: Ty, b: Ty, types: &TypeTable) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (ta, tb) = (types.resolve(a), types.resolve(b));
    variant_rank(ta)
        .cmp(&variant_rank(tb))
        .then_with(|| cmp_same_variant(ta, tb, types))
}

fn variant_rank(ty: &Type) -> u8 {
    match ty {
        Type::ClassInstance { .. } => 0,
        Type::ClassSingleton { .. } => 1,
        Type::Union(_) => 2,
        Type::Intersection(_) => 3,
        Type::Optional(_) => 4,
        Type::Void => 5,
        Type::Nil => 6,
        Type::Bool => 7,
        Type::Untyped => 8,
        Type::Top => 9,
        Type::Bottom => 10,
        Type::SelfType => 11,
        Type::InstanceType => 12,
        Type::ClassType => 13,
        Type::Literal(_) => 14,
        Type::TypeVariable { .. } => 15,
        Type::Interface { .. } => 16,
        Type::Alias { .. } => 17,
        Type::Proc { .. } => 18,
        Type::Tuple(_) => 19,
        Type::Record { .. } => 20,
    }
}

fn cmp_same_variant(a: &Type, b: &Type, types: &TypeTable) -> Ordering {
    match (a, b) {
        (
            Type::ClassInstance { name: n1, args: a1 },
            Type::ClassInstance { name: n2, args: a2 },
        )
        | (Type::Interface { name: n1, args: a1 }, Type::Interface { name: n2, args: a2 })
        | (Type::Alias { name: n1, args: a1 }, Type::Alias { name: n2, args: a2 }) => {
            n1.cmp(n2).then_with(|| cmp_tys(a1, a2, types))
        }
        (Type::ClassSingleton { name: n1 }, Type::ClassSingleton { name: n2 }) => n1.cmp(n2),
        (Type::Union(m1), Type::Union(m2))
        | (Type::Intersection(m1), Type::Intersection(m2))
        | (Type::Tuple(m1), Type::Tuple(m2)) => cmp_tys(m1, m2, types),
        (Type::Optional(t1), Type::Optional(t2)) => cmp_by_content(*t1, *t2, types),
        (Type::Literal(l1), Type::Literal(l2)) => cmp_literal(l1, l2),
        (Type::TypeVariable { raw: r1, scope: s1 }, Type::TypeVariable { raw: r2, scope: s2 }) => {
            r1.cmp(r2).then_with(|| cmp_scope(s1, s2))
        }
        (
            Type::Proc {
                type_: f1,
                self_type: s1,
                block: b1,
            },
            Type::Proc {
                type_: f2,
                self_type: s2,
                block: b2,
            },
        ) => cmp_function_type(f1, f2, types)
            .then_with(|| cmp_opt_ty(*s1, *s2, types))
            .then_with(|| cmp_opt(b1.as_ref(), b2.as_ref(), |x, y| cmp_block(x, y, types))),
        (Type::Record { fields: f1 }, Type::Record { fields: f2 }) => {
            cmp_seq(f1, f2, |(k1, t1, r1), (k2, t2, r2)| {
                k1.cmp(k2)
                    .then_with(|| cmp_by_content(*t1, *t2, types))
                    .then_with(|| r1.cmp(r2))
            })
        }
        (Type::Void, Type::Void)
        | (Type::Nil, Type::Nil)
        | (Type::Bool, Type::Bool)
        | (Type::Untyped, Type::Untyped)
        | (Type::Top, Type::Top)
        | (Type::Bottom, Type::Bottom)
        | (Type::SelfType, Type::SelfType)
        | (Type::InstanceType, Type::InstanceType)
        | (Type::ClassType, Type::ClassType) => Ordering::Equal,
        _ => unreachable!("cmp_same_variant is only called on equal variant ranks"),
    }
}

/// Lexicographic order over two sequences, shorter-is-less on a shared prefix.
fn cmp_seq<T>(a: &[T], b: &[T], mut cmp: impl FnMut(&T, &T) -> Ordering) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let o = cmp(x, y);
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

fn cmp_opt<T>(a: Option<T>, b: Option<T>, cmp: impl FnOnce(T, T) -> Ordering) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => cmp(x, y),
    }
}

fn cmp_tys(a: &[Ty], b: &[Ty], types: &TypeTable) -> Ordering {
    cmp_seq(a, b, |x, y| cmp_by_content(*x, *y, types))
}

fn cmp_opt_ty(a: Option<Ty>, b: Option<Ty>, types: &TypeTable) -> Ordering {
    cmp_opt(a, b, |x, y| cmp_by_content(x, y, types))
}

fn cmp_literal(a: &Literal, b: &Literal) -> Ordering {
    fn rank(l: &Literal) -> u8 {
        match l {
            Literal::Integer(_) => 0,
            Literal::String(_) => 1,
            Literal::Symbol(_) => 2,
            Literal::Bool(_) => 3,
        }
    }
    match (a, b) {
        (Literal::Integer(x), Literal::Integer(y))
        | (Literal::String(x), Literal::String(y))
        | (Literal::Symbol(x), Literal::Symbol(y)) => x.cmp(y),
        (Literal::Bool(x), Literal::Bool(y)) => x.cmp(y),
        _ => rank(a).cmp(&rank(b)),
    }
}

fn cmp_scope(a: &TypeVarScope, b: &TypeVarScope) -> Ordering {
    fn rank(s: &TypeVarScope) -> u8 {
        match s {
            TypeVarScope::Free => 0,
            TypeVarScope::Class(_) => 1,
            TypeVarScope::Interface(_) => 2,
            TypeVarScope::Alias(_) => 3,
            TypeVarScope::Method { .. } => 4,
        }
    }
    fn kind_rank(k: MethodKind) -> u8 {
        match k {
            MethodKind::Instance => 0,
            MethodKind::Singleton => 1,
        }
    }
    match (a, b) {
        (TypeVarScope::Class(x), TypeVarScope::Class(y))
        | (TypeVarScope::Interface(x), TypeVarScope::Interface(y))
        | (TypeVarScope::Alias(x), TypeVarScope::Alias(y)) => x.cmp(y),
        (
            TypeVarScope::Method {
                class: c1,
                method: m1,
                kind: k1,
                overload_index: i1,
            },
            TypeVarScope::Method {
                class: c2,
                method: m2,
                kind: k2,
                overload_index: i2,
            },
        ) => c1
            .cmp(c2)
            .then_with(|| m1.cmp(m2))
            .then_with(|| kind_rank(*k1).cmp(&kind_rank(*k2)))
            .then_with(|| i1.cmp(i2)),
        _ => rank(a).cmp(&rank(b)),
    }
}

fn cmp_function_type(a: &FunctionType, b: &FunctionType, types: &TypeTable) -> Ordering {
    match (a, b) {
        (FunctionType::Typed(f1), FunctionType::Typed(f2)) => cmp_function(f1, f2, types),
        (FunctionType::Untyped(u1), FunctionType::Untyped(u2)) => {
            cmp_by_content(u1.return_type, u2.return_type, types)
        }
        (FunctionType::Typed(_), FunctionType::Untyped(_)) => Ordering::Less,
        (FunctionType::Untyped(_), FunctionType::Typed(_)) => Ordering::Greater,
    }
}

fn cmp_function(a: &Function, b: &Function, types: &TypeTable) -> Ordering {
    let cmp_keywords = |x: &[(String, Ty)], y: &[(String, Ty)]| {
        cmp_seq(x, y, |(n1, t1), (n2, t2)| {
            n1.cmp(n2).then_with(|| cmp_by_content(*t1, *t2, types))
        })
    };
    cmp_tys(&a.required_positionals, &b.required_positionals, types)
        .then_with(|| cmp_tys(&a.optional_positionals, &b.optional_positionals, types))
        .then_with(|| cmp_opt_ty(a.rest_positional, b.rest_positional, types))
        .then_with(|| cmp_tys(&a.trailing_positionals, &b.trailing_positionals, types))
        .then_with(|| cmp_keywords(&a.required_keywords, &b.required_keywords))
        .then_with(|| cmp_keywords(&a.optional_keywords, &b.optional_keywords))
        .then_with(|| cmp_opt_ty(a.rest_keyword, b.rest_keyword, types))
        .then_with(|| cmp_by_content(a.return_type, b.return_type, types))
}

fn cmp_block(a: &Block, b: &Block, types: &TypeTable) -> Ordering {
    a.required
        .cmp(&b.required)
        .then_with(|| cmp_function_type(&a.type_, &b.type_, types))
        .then_with(|| cmp_opt_ty(a.self_type, b.self_type, types))
}
