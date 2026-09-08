//! Top-level directives carried in a single `.rbs` source file.
//!
//! Mirrors `RBS::AST::Directives::*` (`rbs/lib/rbs/ast/directives.rb`,
//! `rbs/sig/directives.rbs`). Each `.rbs` source file may start with
//! a sequence of directives that affect how short type names resolve
//! inside that file's scope.
//!
//! Two variants are modeled:
//! - [`UseDirective`] mirrors `RBS::AST::Directives::Use` (file-scoped short
//!   name aliases). Resolution is handled by
//!   [`crate::environment::use_map::UseMap`].
//! - [`ResolveTypeNamesDirective`] mirrors
//!   `RBS::AST::Directives::ResolveTypeNames` (the `# resolve-type-names:
//!   false` magic comment). When `value` is `false`, the draft build
//!   skips type-name resolution for that source file and feeds the raw
//!   AST through unchanged.

use crate::location::{
    ResolveTypeNamesDirectiveLocation, UseDirectiveLocation, UseSingleClauseLocation,
    UseWildcardClauseLocation,
};
use crate::name::Symbol;
use crate::type_name::TypeName;

/// One top-level directive in a `.rbs` file.
///
/// Mirrors the rbs `AST::Directives::Base` hierarchy.
#[derive(Debug, Clone)]
pub enum Directive {
    Use(UseDirective),
    ResolveTypeNames(ResolveTypeNamesDirective),
}

/// `use Foo, Foo::Bar as FBar, Foo::Baz::*` — one `use` line with one
/// or more clauses.
///
/// Mirrors `RBS::AST::Directives::Use`.
#[derive(Debug, Clone)]
pub struct UseDirective {
    pub clauses: Vec<UseClause>,
    pub location: Option<UseDirectiveLocation>,
}

/// One clause inside a `use` directive.
///
/// Mirrors `RBS::AST::Directives::Use::clause` (the type alias for
/// `UseSingleClause | UseWildcardClause`).
#[derive(Debug, Clone)]
pub enum UseClause {
    Single(UseSingleClause),
    Wildcard(UseWildcardClause),
}

/// `use Foo::Bar` or `use Foo::Bar as X`.
///
/// Mirrors `RBS::AST::Directives::Use::SingleClause`. `new_name` is
/// `Some` when an `as` keyword is present, otherwise the short name
/// is taken from the last segment of `type_name`.
#[derive(Debug, Clone)]
pub struct UseSingleClause {
    pub type_name: TypeName,
    pub new_name: Option<Symbol>,
    pub location: Option<UseSingleClauseLocation>,
}

/// `use Foo::Bar::*`.
///
/// Mirrors `RBS::AST::Directives::Use::WildcardClause`. `namespace`
/// designates the parent under which all known type names become
/// directly accessible by their short name.
#[derive(Debug, Clone)]
pub struct UseWildcardClause {
    pub namespace: TypeName,
    pub location: Option<UseWildcardClauseLocation>,
}

/// `# resolve-type-names: false` (or `true`).
///
/// Mirrors `RBS::AST::Directives::ResolveTypeNames`. When `value` is
/// `false`, the draft build skips type-name resolution for the
/// containing source file (the raw AST is fed through to the frozen
/// environment unchanged). When `value` is `true`, the directive is
/// equivalent to its absence (default behavior).
#[derive(Debug, Clone)]
pub struct ResolveTypeNamesDirective {
    pub value: bool,
    pub location: Option<ResolveTypeNamesDirectiveLocation>,
}
