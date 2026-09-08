//! Unresolved AST layer.
//!
//! Phase 1 of the rbs-isomorphic parser layer described in
//! ADR-0014. The modules under
//! `ast::` mirror `RBS::Types::*` and `RBS::AST::Ruby::*` — values produced
//! directly by parsers before any name resolution against an DefinitionBuilder has
//! taken place.
//!
//! The resolved counterpart lives in [`crate::types`] (`Ty`, interned). The
//! two layers **coexist** during the migration; converting one to the other
//! is the job of a later phase (a pure `ast::Type → Ty` resolver).
//!
//! Phase 1 introduced `ast::types`. Phase 2 added `ast::method_type`
//! (unresolved method signatures). Phase 3 adds
//! `ast::ruby::{annotations, members}` for leading inline annotations and
//! the `MethodTypeAnnotation` classifier. `ast::ruby::declarations` will
//! arrive in a later phase.
pub mod annotation;
pub mod comment;
pub mod declarations;
pub mod directives;
pub mod members;
pub mod method_type;
pub mod ruby;
pub mod type_param;
pub mod types;

pub use type_param::{TypeParam, Variance};

use crate::ast::declarations::Declaration;
use crate::ast::directives::Directive;

/// One `.rbs` source file's top-level structure.
///
/// Mirrors `RBS::Source::RBS` (`rbs/lib/rbs/source.rb`) at the
/// directives + declarations level. The `buffer` field that rbs
/// carries is out of scope: crema's RBS AST location bookkeeping lives in
/// file-less ranges per node, so there is no file-level buffer handle to
/// keep alongside.
#[derive(Debug, Clone)]
pub struct Source {
    pub directives: Vec<Directive>,
    pub declarations: Vec<Declaration>,
}

pub use members::{MethodKind, Visibility};
