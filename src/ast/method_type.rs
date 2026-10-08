//! Unresolved method types — mirrors `RBS::MethodType`.
//!
//! Produced by `crate::ast_builder::build_method_type` from an rbs-wrapped
//! node, without name resolution. A `definition_builder` pass later turns
//! this into the resolved [`crate::types::MethodType`].
//!
//! Method-level type parameters share the unified [`crate::ast::TypeParam`]
//! shape (rbs Ruby `RBS::AST::TypeParam` is the same struct for class- and
//! method-level use; `sig/method_types.rbs:16` declares
//! `type_params: Array[AST::TypeParam]`). The parser is responsible for
//! rejecting method-scope variance / defaults.

use crate::ast::TypeParam;
use crate::ast::types::{BlockType, Function};
use crate::location::MethodTypeLocation;


/// Unresolved method type. Mirrors `RBS::MethodType` and structurally
/// parallels the resolved [`crate::types::MethodType`]: same three fields
/// (type parameters, function signature, optional block) but with owned
/// `ast::*` children instead of interned `Ty` handles.
///
/// The `function` field corresponds to rbs's `RBS::MethodType#type`; the
/// resolved layer renames it to `type_` (Rust keyword escape) while this
/// layer renames it semantically.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MethodType {
    pub type_params: Vec<TypeParam>,
    pub function: Function,
    pub block: Option<BlockType>,
    pub location: Option<MethodTypeLocation>,
}
