//! Mirrors `RBS::AST::Annotation` (lib/rbs/ast/annotation.rb).

use crate::location::LocationRange;
use crate::name::Symbol;

/// A `%a{...}` source-level annotation written above an rbs signature.
///
/// Mirrors `RBS::AST::Annotation`. `string` is the annotation body
/// verbatim (rbs strips the surrounding `%a{}`); semantic
/// interpretation (`noreturn`, etc.) belongs to downstream layers.
/// `location` is `Option` to allow synthetic annotations from
/// non-source paths, following the convention in
/// [`crate::ast::declarations`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Annotation {
    pub string: Symbol,
    pub location: Option<LocationRange>,
}
