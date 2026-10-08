//! Unresolved type parameter — mirrors `RBS::AST::TypeParam`.
//!
//! Single struct shared by class / module / interface / type-alias
//! declarations and by method types. rbs Ruby uses the same
//! `RBS::AST::TypeParam` for both (`sig/method_types.rbs:16` declares
//! `type_params: Array[AST::TypeParam]`), and rbs Rust follows suit
//! (`rust/ruby-rbs/src/ast/type_param.rs`). Method-level parameters in
//! source forbid `variance` / `default_type` / bounds combinations
//! through the parser, but the AST node carries the full shape and
//! leaves enforcement to upstream syntax validation.

use crate::ast::types::Type;
use crate::location::TypeParamLocation;
use crate::name::Symbol;

/// Variance keyword on a type parameter.
///
/// Defaults to `Invariant` when source has neither `in` nor `out`.
/// Distinct from [`crate::type_param::Variance`] (resolved side) so
/// the AST layer stays free of `crate::types` dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Variance {
    Invariant,
    Covariant,
    Contravariant,
}

/// Unresolved type parameter. Mirrors `RBS::AST::TypeParam` and
/// structurally parallels rbs Rust `ast::TypeParam`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TypeParam {
    pub name: Symbol,
    pub variance: Variance,
    pub upper_bound: Option<Type>,
    pub lower_bound: Option<Type>,
    pub default_type: Option<Type>,
    pub unchecked: bool,
    pub location: Option<TypeParamLocation>,
}

impl TypeParam {
    /// Construct a bare type parameter with `Invariant` variance and
    /// no bounds / default — the shape produced when a method-level
    /// `[T]` is parsed.
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            variance: Variance::Invariant,
            upper_bound: None,
            lower_bound: None,
            default_type: None,
            unchecked: false,
            location: None,
        }
    }
}
