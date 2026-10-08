//! Thin adapter from a frozen [`Environment`] to the
//! [`crate::definition_builder::type_builder`] helpers.
//!
//! The frozen-Environment AST is **already name-resolved** (ADR-0017
//! Phase 4d): every reference-position `TypeName` in the AST holds an
//! absolute path, so the adapter passes an empty `context` slice. The
//! environment is still consulted while lowering: the
//! `TypeNameResolver` built over `Environment::all_names` / `aliases`
//! follows a class alias in namespace position, and
//! `Environment::normalize_type_name` rewrites a final alias name to
//! its target (`class A = ::Real` → references to `::A` lower to
//! `::Real`). Callers that parse fresh AST at check time (inline
//! assertions) go through [`LoweringEnv::build_type_in_context`] with
//! the lexical nesting as `context`.

use crate::ast::members::MethodDefinitionOverload as AstOverload;
use crate::ast::method_type::MethodType as AstMethodType;
use crate::ast::ruby::members::MethodTypeAnnotation;
use crate::ast::types::Type as AstType;
use crate::definition_builder::type_builder;
use crate::environment::Environment;
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;
use crate::type_param::TypeParamScope;
use crate::types::{MethodType, Ty, TypeTable};

/// Lowering view over a frozen [`Environment`].
///
/// Every field is a borrow, so building one per call site is free;
/// `DefinitionBuilder` / `AncestorBuilder` construct it on demand from
/// the `Arc<Environment>` they already hold.
pub struct LoweringEnv<'a> {
    pub env: &'a Environment,
    pub names: &'a NameTable,
    pub types: &'a TypeTable,
}

impl<'a> LoweringEnv<'a> {
    /// Build a `LoweringEnv` from the frozen environment plus the
    /// caller's `TypeTable` (the table that will receive the interned
    /// `Ty`s).
    pub fn from_environment(env: &'a Environment, types: &'a TypeTable) -> Self {
        Self {
            env,
            names: env.names(),
            types,
        }
    }

    pub fn build_type(&self, ast_ty: &AstType, type_param_scope: &TypeParamScope) -> Ty {
        self.build_type_in_context(ast_ty, &[], type_param_scope)
    }

    /// Lower an AST type whose names may still be relative, resolving
    /// them against `context` (innermost last, the shape the type
    /// checker's cref stack has).
    pub fn build_type_in_context(
        &self,
        ast_ty: &AstType,
        context: &[TypeName],
        type_param_scope: &TypeParamScope,
    ) -> Ty {
        type_builder::build_type(
            ast_ty,
            context,
            self.env,
            self.names,
            self.types,
            type_param_scope,
        )
    }

    pub fn build_method_type(
        &self,
        ast_mt: &AstMethodType,
        type_param_scope: &TypeParamScope,
        method_owner: Option<(&TypeName, Symbol, crate::type_param::MethodKind)>,
        overload_index: u16,
    ) -> MethodType {
        type_builder::build_method_type(
            ast_mt,
            &[],
            self.env,
            self.names,
            self.types,
            type_param_scope,
            method_owner,
            overload_index,
        )
    }

    pub fn build_overloads(
        &self,
        ast_overloads: &[AstOverload],
        type_param_scope: &TypeParamScope,
        method_owner: Option<(&TypeName, Symbol, crate::type_param::MethodKind)>,
        start_index: u16,
    ) -> Vec<MethodType> {
        type_builder::build_overloads(
            ast_overloads,
            &[],
            self.env,
            self.names,
            self.types,
            type_param_scope,
            method_owner,
            start_index,
        )
    }

    pub fn build_overloads_from_annotation(
        &self,
        annotation: &MethodTypeAnnotation,
        type_param_scope: &TypeParamScope,
        method_owner: Option<(&TypeName, Symbol, crate::type_param::MethodKind)>,
    ) -> Option<Vec<MethodType>> {
        type_builder::build_overloads_from_annotation_with_scope(
            annotation,
            &[],
            self.env,
            self.names,
            self.types,
            type_param_scope,
            method_owner,
        )
    }
}
