//! Mirrors `RBS::AST::Comment` (lib/rbs/ast/comment.rb).

use crate::location::LocationRange;
use crate::name::Symbol;

/// Mirrors `RBS::AST::Comment`. `string` is the full comment body
/// (rbs strips leading `#` and whitespace); equality is by `string`
/// only — `location` is carried for diagnostics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Comment {
    pub string: Symbol,
    pub location: Option<LocationRange>,
}
