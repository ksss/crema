//! Thin adapter from a frozen [`Environment`] to the existing
//! [`crate::definition_builder::type_builder`] helpers.
//!
//! The frozen-Environment AST is **already name-resolved** (ADR-0017
//! Phase 4d): every reference-position `TypeName` in the AST holds an
//! absolute path. So the adapter passes the legacy helper an empty
//! `context` slice — relative-name resolution becomes a no-op for AST
//! that has no relative names left. `class_aliases` is still populated
//! because alias substitution runs against absolute names too
//! (`class A = ::Real` → references to `::A` inside type expressions
//! must rewrite to `::Real` at lower time).

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::ast::members::MethodDefinitionOverload as AstOverload;
use crate::ast::method_type::MethodType as AstMethodType;
use crate::ast::ruby::members::MethodTypeAnnotation;
use crate::ast::types::Type as AstType;
use crate::definition::lowering_maps::LoweringMaps;
use crate::definition_builder::type_builder;
use crate::environment::Environment;
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassOrModuleAliasEntry, ModuleAliasDeclaration,
};
use crate::name::{Name, NameTable, Symbol};
use crate::type_name::TypeName;
use crate::type_param::TypeParamScope;
use crate::types::{MethodType, Ty, TypeTable};

/// Render `tn` as an absolute path string. Relative-form TypeNames
/// (resolution fell back to the source form because the target was
/// not in the environment) are lifted to absolute via `to_absolute`,
/// preserving the lowering map's pre-TypeName invariant that every
/// key/value is an absolute path string interned via `NameTable`.
/// Already-absolute names pass through unchanged.
fn absolutize_path(tn: TypeName, names: &NameTable) -> String {
    names.resolve(names.to_absolute(tn))
}

/// Lowering view over a frozen [`Environment`].
///
/// Built once per `DefinitionBuilder::from_environment` call and reused
/// for every type lowering inside the build phase. `maps` is `Arc`-shared
/// so cheap clone-backed instances can be constructed for on-the-fly
/// lookups without re-walking every decl.
pub struct LoweringEnv<'a> {
    /// Declared-name membership (`all_names`) plus `class A = ::Real`
    /// alias targets (`class_aliases`) that `type_builder` consults while
    /// lowering `Type::ClassInstance` / `Interface` / `Alias` /
    /// `ClassSingleton` nodes. See [`LoweringMaps`]'s own doc for the
    /// baseline+overlay+tombstone shape.
    pub maps: Arc<LoweringMaps>,
    pub names: &'a NameTable,
    pub types: &'a TypeTable,
}

impl<'a> LoweringEnv<'a> {
    /// Build a `LoweringEnv` from the frozen environment plus the
    /// caller's `TypeTable` (the table that will receive the interned
    /// `Ty`s).
    pub fn from_environment(env: &'a Environment, types: &'a TypeTable) -> Self {
        let names = env.names();
        let (all_names, class_aliases) = build_lowering_maps(env, names);
        Self {
            maps: Arc::new(LoweringMaps::from_owned_maps(all_names, class_aliases)),
            names,
            types,
        }
    }

    /// Construct a cheap clone-backed instance that shares `maps` from an
    /// already-built `Arc`. Used by `DefinitionBuilder` on-the-fly lookup
    /// functions to avoid re-walking every declaration on each call.
    pub fn from_maps(maps: Arc<LoweringMaps>, names: &'a NameTable, types: &'a TypeTable) -> Self {
        Self { maps, names, types }
    }

    pub fn build_type(&self, ast_ty: &AstType, type_param_scope: &TypeParamScope) -> Ty {
        type_builder::build_type(
            ast_ty,
            &[],
            &self.maps,
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
            &self.maps,
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
            &self.maps,
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
        context: &[Option<Name>],
        type_param_scope: &TypeParamScope,
        method_owner: Option<(&TypeName, Symbol, crate::type_param::MethodKind)>,
    ) -> Option<Vec<MethodType>> {
        type_builder::build_overloads_from_annotation_with_scope(
            annotation,
            context,
            &self.maps,
            self.names,
            self.types,
            type_param_scope,
            method_owner,
        )
    }
}

/// Build the `all_names` and `class_aliases` maps from a frozen environment.
/// Extracted so `from_environment` and `DefinitionBuilder` share the same
/// construction logic.
pub fn build_lowering_maps(
    env: &Environment,
    names: &NameTable,
) -> (FxHashSet<Name>, FxHashMap<Name, Name>) {
    let cap = env.class_decls().len()
        + env.interface_decls().len()
        + env.type_alias_decls().len()
        + env.constant_decls().len()
        + env.class_alias_decls().len();
    let mut all_names = FxHashSet::with_capacity_and_hasher(cap, Default::default());
    for tn in env
        .class_decls()
        .keys()
        .chain(env.interface_decls().keys())
        .chain(env.type_alias_decls().keys())
        .chain(env.constant_decls().keys())
        .chain(env.class_alias_decls().keys())
    {
        all_names.insert(names.intern(&names.resolve(tn)));
    }
    let mut class_aliases: FxHashMap<Name, Name> =
        FxHashMap::with_capacity_and_hasher(env.class_alias_decls().len(), Default::default());
    for (new_tn, entry) in env.class_alias_decls() {
        let Some(old_n) = class_alias_old_name(entry, names) else {
            continue;
        };
        let new_n = names.intern(&names.resolve(new_tn));
        class_aliases.insert(new_n, old_n);
    }
    (all_names, class_aliases)
}

/// The old-side target `Name` of a `class_alias_decls` entry, or `None`
/// when it is a Ruby-side alias whose target could not be resolved (rbs
/// parity: such an entry contributes nothing to `class_aliases`). Shared
/// by [`build_lowering_maps`] (full walk) and
/// [`AncestorBuilder::update`](crate::definition::ancestor_builder::AncestorBuilder::update)
/// (per-name patch) so both compute an identical mapping.
pub(crate) fn class_alias_old_name(
    entry: &ClassOrModuleAliasEntry,
    names: &NameTable,
) -> Option<Name> {
    let owned_old: String;
    let old_raw: &str = match entry {
        ClassOrModuleAliasEntry::Class(e) => match &e.decl {
            ClassAliasDeclaration::Signature(s) => {
                owned_old = names.resolve(s.old_name);
                &owned_old
            }
            ClassAliasDeclaration::Ruby(d) => {
                owned_old = absolutize_path(d.old_name(names)?, names);
                &owned_old
            }
        },
        ClassOrModuleAliasEntry::Module(e) => match &e.decl {
            ModuleAliasDeclaration::Signature(s) => {
                owned_old = names.resolve(s.old_name);
                &owned_old
            }
            ModuleAliasDeclaration::Ruby(d) => {
                owned_old = absolutize_path(d.old_name(names)?, names);
                &owned_old
            }
        },
    };
    Some(names.intern(old_raw))
}
