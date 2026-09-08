//! String-building helpers for rbs_loader's declaration walk.
//!
//! Phase 2 of ADR-0014 moved the actual `resolve_relative_type_name`
//! logic into [`crate::resolver::type_name_resolver`]. What remains here
//! is the lightweight `qualified_name` concatenation and the
//! `ResolveContext` alias that the loader threads through the
//! declaration recursion — neither does name resolution against an
//! environment, so they sit outside the resolver module proper.

use crate::name::Name;

/// Resolution context for relative type names.
/// Represents the nesting stack of module/class declarations.
///
/// Each entry is `Some(absolute_name)` for a declared module/class,
/// or `None` for an undeclared module (RBS inline `false` case).
///
/// Example: `module OpenSSL; module SSL; class SSLContext`
///   `[Some(::OpenSSL), Some(::OpenSSL::SSL), Some(::OpenSSL::SSL::SSLContext)]`
pub type ResolveContext = Vec<Option<Name>>;

pub(crate) fn qualified_name(ns_prefix: &str, name: &str) -> String {
    if name.starts_with("::") {
        name.to_string()
    } else if ns_prefix.is_empty() {
        format!("::{}", name)
    } else {
        format!("{}::{}", ns_prefix, name)
    }
}
