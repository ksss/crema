//! Top-level AST declarations collected from inline-annotated Ruby sources.
//!
//! Mirrors `RBS::AST::Ruby::Declarations::*`
//! (`lib/rbs/ast/ruby/declarations.rb`). Unlike the flat `RawInlineMember`
//! enum that preceded this module, each declaration owns its nested
//! [`Member`]s, matching rbs's tree shape so that resolver passes and
//! the future `env.add_source` / `resolve_type_names` 2-step (Phase 7 of
//! ADR-0014) can walk one value without rebuilding the tree.
//!
//! Phase 5 of ADR-0014.
//!
//! [`ClassModuleAliasDecl`] is the single declaration shape for both
//! `#: class-alias` and `#: module-alias`, with class/module dispatch
//! living on [`crate::ast::ruby::annotations::AliasAnnotation`]'s
//! variant. The frozen [`crate::environment::frozen::ClassAliasDeclaration::Ruby`] /
//! [`crate::environment::frozen::ModuleAliasDeclaration::Ruby`] variants
//! reuse the same struct, mirroring rbs's `Environment#class_alias_decls` /
//! `#module_alias_decls` split at storage level only.

use crate::ast::declarations::ModuleSelf as SelfType;
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::annotations::{AliasAnnotation, TypeApplicationAnnotation};
use crate::ast::ruby::comment_block::CommentBlock;
use crate::ast::ruby::members::Member;
use crate::location::RubyLocation;
use crate::name::NameTable;
use crate::type_name::TypeName;
use std::sync::Arc;

/// A top-level declaration in an inline-annotated Ruby file.
///
/// Mirrors the subclasses of `RBS::AST::Ruby::Declarations::Base`.
/// Declarations may nest (a `ClassDecl` can contain nested `ClassDecl`s
/// as members via [`Member::Declaration`]).
///
/// Class / module payloads are `Arc` like the signature-side
/// [`crate::ast::declarations::Declaration`]: after `EnvironmentDraft::build`
/// the parent's `members` and the flattened env entry share one resolved
/// decl, mirroring rbs `Environment#resolve_ruby_decl` which stores the
/// resolved object itself in the parent's members.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Declaration {
    Class(Arc<ClassDecl>),
    Module(Arc<ModuleDecl>),
    Constant(ConstantDecl),
    /// `Foo = Bar #: class-alias` / `Foo = Bar #: module-alias` —
    /// class- or module-alias from an inline annotation. The class/module
    /// distinction is carried by the payload's
    /// [`ClassModuleAliasDecl::annotation`] variant, mirroring rbs's
    /// `decl.annotation.is_a?(ClassAliasAnnotation)` dispatch from
    /// `RBS::AST::Ruby::Declarations::ClassModuleAliasDecl`.
    ClassModuleAlias(ClassModuleAliasDecl),
}

/// A `class Foo < Bar` declaration with its body lowered to AST members.
///
/// Mirrors `RBS::AST::Ruby::Declarations::ClassDecl`. `class_name` is the
/// fully-qualified canonical name with `namespace.is_absolute() == true`,
/// so callers do not need to re-derive it from an enclosing scope stack.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassDecl {
    pub class_name: TypeName,
    pub name_location: PrismByteRange,
    pub super_class: Option<SuperClass>,
    pub members: Vec<Member>,
    /// crema extension, not part of the rbs `ClassDecl` contract (rbs's
    /// `InlineParser` never builds a decl from such a body): `true` when
    /// `members` came from a `Const = Class.new(...) do ... end` /
    /// `Struct.new(...) do` / `Data.define(...) do` block instead of a
    /// `class` keyword body. The block owns `self` and `def` for
    /// `class_name`, but Ruby's cref (`Module.nesting`) inside it is the
    /// *enclosing* scope, so the environment resolves the members'
    /// annotation and mixin names in the decl's own context rather than
    /// extending it with `class_name`.
    pub block_body: bool,
}

/// The explicit superclass reference of a `ClassDecl`, if any.
///
/// Mirrors `RBS::AST::Ruby::Declarations::ClassDecl::SuperClass`.
///
/// `type_name` carries the name as written. At collect time it preserves
/// the source-form absoluteness (`namespace.is_absolute() == false` for
/// `Bar`, `true` for `::Foo`); `EnvironmentDraft::build` rewrites it to
/// the absolute form via `resolve_ruby_class_recursive`. `type_annotation`
/// is the owned snapshot of a trailing `#[T, U]` annotation on the super
/// reference (rbs `type_annotation: TypeApplicationAnnotation`); the
/// annotation's own `location` field carries the byte range of the `#[T]`
/// comment, removing the need for a separate range slot. `byte_range` is
/// the byte range of the superclass name reference itself in the Ruby
/// source (mirrors rbs's `type_name_location`) and feeds `file:line` on
/// arity diagnostics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SuperClass {
    pub type_name: TypeName,
    pub type_annotation: Option<TypeApplicationAnnotation>,
    pub byte_range: Option<PrismByteRange>,
}

/// A `module Foo` declaration with its body lowered to AST members.
///
/// Mirrors `RBS::AST::Ruby::Declarations::ModuleDecl`. `module_name` is
/// absolute (`namespace.is_absolute() == true`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModuleDecl {
    pub module_name: TypeName,
    pub name_location: PrismByteRange,
    pub members: Vec<Member>,
}

impl ModuleDecl {
    /// Mirrors rbs `AST::Ruby::Declarations::ModuleDecl#self_types`:
    /// filter `Members::ModuleSelfMember` out of `members` and lift each
    /// into a `Module::Self` (crema's [`SelfType`]).
    ///
    /// `SelfType::location` is set to `None`: `ModuleSelfMember` carries
    /// only a `PrismByteRange` (no file id), and the frozen layer has no
    /// build context to lift it into a [`Location`]. The same precedent
    /// is used by the super-class Ruby branch in
    /// `ancestor_builder::class_super_or_default` (see the
    /// `// PrismByteRange → Location conversion is intentionally
    /// skipped` comment there). Dedup at the [`crate::environment::frozen::ModuleEntry::self_types`]
    /// site goes through `strip_location`, so this `None` does not
    /// affect aggregation semantics.
    pub fn self_types(&self) -> Vec<SelfType> {
        self.members
            .iter()
            .filter_map(|member| match member {
                Member::ModuleSelf(m) => Some(SelfType {
                    name: m.annotation.name,
                    args: m.annotation.args.clone(),
                    location: None,
                    source_file: None,
                }),
                _ => None,
            })
            .collect()
    }
}

/// A `CONST = value` declaration.
///
/// Mirrors `RBS::AST::Ruby::Declarations::ConstantDecl`.
///
/// `type_text` carries the body of an optional trailing `#: T` comment
/// (the resolver parses it lazily via `parse_rbs_type`). When absent,
/// `value_kind` drives literal-based inference (e.g. an integer literal
/// resolves to `::Integer`). `annotation_range` is populated only when
/// `type_text.is_some()` and attaches diagnostics on parse failure.
/// `leading_comment` mirrors `RBS::AST::Ruby::Declarations::ConstantDecl#leading_comment`;
/// the inline collector currently leaves it as `None` until comment-block
/// collection is wired through.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstantDecl {
    pub constant_name: TypeName,
    pub name_location: PrismByteRange,
    pub type_text: Option<String>,
    pub annotation_range: Option<PrismByteRange>,
    pub value_kind: ConstantValueKind,
    pub leading_comment: Option<CommentBlock>,
}

/// `Foo = Bar` class- or module-alias assignment.
///
/// Mirrors `RBS::AST::Ruby::Declarations::ClassModuleAliasDecl`. The
/// class/module distinction is carried by [`Self::annotation`]'s variant
/// (`AliasAnnotation::Class` / `AliasAnnotation::Module`), matching rbs's
/// `decl.annotation.is_a?(ClassAliasAnnotation)` dispatch.
///
/// `new_name` is absolute (`namespace.is_absolute() == true`). The
/// annotation's optional `type_name_text` corresponds to rbs's
/// `annotation.type_name`. Either that or `infered_old_name` (inferred
/// from the right-hand side constant) must be present; the inline
/// collector reports `InlineClassAliasMissingTypeName` and drops the
/// declaration when both are absent. `infered_old_name` carries the
/// source-form absoluteness at collect time; `resolve_ruby_alias_decl`
/// rewrites it to absolute.
///
/// `leading_comment` mirrors
/// `RBS::AST::Ruby::Declarations::ClassModuleAliasDecl#leading_comment`;
/// the inline collector currently leaves it as `None` until comment-block
/// collection is wired through.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassModuleAliasDecl {
    pub new_name: TypeName,
    pub name_location: PrismByteRange,
    pub infered_old_name: Option<TypeName>,
    pub annotation: AliasAnnotation,
    pub byte_range: Option<PrismByteRange>,
    /// Location used for build-layer diagnostics such as
    /// `UnknownTypeName`. Populated by the inline parser at collect time
    /// from `self.file` and `byte_range` (the whole-assignment range —
    /// matches rbs's degraded inline behavior where
    /// `decl.location&.[](:old_name)` returns nil and the validator
    /// falls back to the assignment location).
    ///
    /// crema-specific simplification: rbs's `ClassModuleAliasDecl`
    /// holds `@buffer` + `@node` (Prism node) from which file and
    /// byte ranges are derived on demand. crema currently extracts
    /// byte_range eagerly and carries file via this Location field.
    /// When the buffer/node port is revisited (richer rbs Location
    /// with named sub-ranges), this field should be reconsidered.
    pub old_name_location: Option<RubyLocation>,
    pub leading_comment: Option<CommentBlock>,
}

impl ClassModuleAliasDecl {
    /// Mirrors `RBS::AST::Ruby::Declarations::ClassModuleAliasDecl#old_name`:
    /// the explicit annotation argument wins over the inferred right-hand-side
    /// constant. Returns `None` only when both fields are absent — the inline
    /// collector treats that as a diagnostic and drops the declaration before
    /// it reaches this method.
    ///
    /// The annotation's `type_name_text` is still a `String` (its
    /// `TypeName` lift lives in
    /// `mid_owned_class_module_alias_annotation.md`), so this helper
    /// re-parses it on demand against `names`. `infered_old_name` is
    /// already a `TypeName` and is cloned through.
    pub fn old_name(&self, names: &NameTable) -> Option<TypeName> {
        if let Some(text) = self.annotation.type_name_text() {
            Some(names.parse_type_name(text))
        } else {
            self.infered_old_name
        }
    }
}

/// The kind of value assigned to a constant, used for literal type
/// inference when no `#: T` annotation is present.
///
/// Kept deliberately coarse (no payload) because the resolver only needs
/// to map the kind to a well-known type — richer AST reuse is not a
/// goal of Phase 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConstantValueKind {
    Integer,
    Float,
    String,
    True,
    False,
    Symbol,
    Nil,
    Other,
}
