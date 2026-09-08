//! Unified declaration AST. Mirrors `RBS::AST::Declarations::*`.
//!
//! Where RBS-loaded and Ruby-loaded declarations merge into a single
//! `Arc<ClassDeclaration>` / `Arc<ModuleDeclaration>` / `Arc<InterfaceDeclaration>` shape
//! that [`crate::environment::draft::EnvironmentDraft`] holds uniformly.
//!
//! Rationale and the full data-shape map live in ADR-0017 (sections
//! "Data structures" and "Layer correspondence").
//!
//! `ast::ruby::declarations` and `ast::ruby::members` predate this
//! module and serve as the Ruby-collector intermediate form
//! (`qualified_name: String`, byte-range based). They route through
//! this layer separately and are intentionally left untouched here.

use std::sync::Arc;

use crate::ast::TypeParam;
use crate::ast::annotation::Annotation;
use crate::ast::comment::Comment;
use crate::ast::members::{
    AliasMember, AttrAccessorMember, AttrReaderMember, AttrWriterMember,
    ClassInstanceVariableMember, ClassVariableMember, ExtendMember, IncludeMember,
    InstanceVariableMember, MethodDefinitionMember, PrependMember, PrivateMember, PublicMember,
};
use crate::ast::types::Type;
use crate::location::{
    AliasDeclarationLocation, ClassDeclarationLocation, ClassSuperLocation,
    ConstantDeclarationLocation, GlobalDeclarationLocation, InterfaceDeclarationLocation,
    ModuleDeclarationLocation, ModuleSelfLocation, TypeAliasDeclarationLocation,
};
use crate::name::{Name, Symbol};
use crate::type_name::TypeName;


/// `class C[X, ...] < Super ... end`.
///
/// Mirrors `RBS::AST::Declarations::Class`. Members own their
/// individual decl form; class-level attributes (`super_class`,
/// `type_params`) live on the decl rather than inside members so that
/// downstream code does not have to scan a `Vec<...>` for them.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassDeclaration {
    pub name: TypeName,
    pub type_params: Vec<TypeParam>,
    pub super_class: Option<ClassSuper>,
    pub members: Vec<ClassMember>,
    pub annotations: Vec<Annotation>,
    pub location: Option<ClassDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `module M[X, ...] : SelfType, ... end`.
///
/// Mirrors `RBS::AST::Declarations::Module`. `self_types` lists the
/// post-`:` self-type constraints; empty when the source has no
/// self-type clause.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDeclaration {
    pub name: TypeName,
    pub type_params: Vec<TypeParam>,
    pub self_types: Vec<ModuleSelf>,
    pub members: Vec<ModuleMember>,
    pub annotations: Vec<Annotation>,
    pub location: Option<ModuleDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `interface _I[X, ...] ... end`.
///
/// Mirrors `RBS::AST::Declarations::Interface`. Interfaces accept a
/// strict subset of class/module members (only methods); the
/// dedicated [`Member`] enum (without nested-declaration variants) encodes
/// that restriction at the type level.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceDeclaration {
    pub name: TypeName,
    pub type_params: Vec<TypeParam>,
    pub members: Vec<Member>,
    pub annotations: Vec<Annotation>,
    pub location: Option<InterfaceDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `type Name[X, ...] = T`.
///
/// Mirrors `RBS::AST::Declarations::TypeAlias`. RBS allows full
/// class-level type-param syntax on aliases, including variance
/// (`type foo[out T] = ...`), `unchecked`, bounds, and defaults
/// (`type foo[unchecked out T = String] = ...`); the AST therefore
/// reuses the same [`TypeParam`] shape as classes and modules.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAliasDeclaration {
    pub name: TypeName,
    pub type_params: Vec<TypeParam>,
    pub ty: Type,
    pub annotations: Vec<Annotation>,
    pub location: Option<TypeAliasDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `Name: T` at the top level.
///
/// Mirrors `RBS::AST::Declarations::Constant`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantDeclaration {
    pub name: TypeName,
    pub ty: Type,
    pub annotations: Vec<Annotation>,
    pub location: Option<ConstantDeclarationLocation>,
    pub comment: Option<Comment>,
}

/// `$name: T`.
///
/// Mirrors `RBS::AST::Declarations::Global`. The leading `$` is part
/// of the source syntax; the AST stores the bare identifier (e.g.
/// `Name` for `$DEBUG`).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalDeclaration {
    pub name: Symbol,
    pub ty: Type,
    pub annotations: Vec<Annotation>,
    pub location: Option<GlobalDeclarationLocation>,
    pub comment: Option<Comment>,
}

/// `class New = Old`.
///
/// Mirrors `RBS::AST::Declarations::ClassAlias`. Fields are held
/// directly (no shared `AliasDecl` wrapper) following the rbs Rust
/// flatten convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassAliasDeclaration {
    pub new_name: TypeName,
    pub old_name: TypeName,
    pub annotations: Vec<Annotation>,
    pub location: Option<AliasDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `module New = Old`.
///
/// Mirrors `RBS::AST::Declarations::ModuleAlias`. Fields are held
/// directly (no shared `AliasDecl` wrapper) following the rbs Rust
/// flatten convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleAliasDeclaration {
    pub new_name: TypeName,
    pub old_name: TypeName,
    pub annotations: Vec<Annotation>,
    pub location: Option<AliasDeclarationLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// The set of members that can appear inside a class, module, or interface body,
/// excluding nested declarations.
///
/// Mirrors `RBS::AST::Members::t` from the rbs Rust port. Nested declarations
/// (`ClassMember::Declaration`, `ModuleMember::Declaration`) are held in the
/// wrapper enums rather than here, because interfaces do not accept them.
///
/// `super_class` (class-only) and `self_types` (module-only) live on the parent
/// decl struct rather than in members; both are at most one per declaration and
/// downstream code accesses them by name.
#[derive(Debug, Clone, PartialEq)]
pub enum Member {
    MethodDefinition(MethodDefinitionMember),
    Include(IncludeMember),
    Extend(ExtendMember),
    Prepend(PrependMember),
    AttrReader(AttrReaderMember),
    AttrWriter(AttrWriterMember),
    AttrAccessor(AttrAccessorMember),
    Public(PublicMember),
    Private(PrivateMember),
    Alias(AliasMember),
    /// Signature-side `@foo: T`. Populated into the instance-side
    /// Definition's `instance_variables`. See [`crate::definition::Variable`].
    InstanceVariable(InstanceVariableMember),
    /// Signature-side `self.@foo: T`. Populated into the singleton-side
    /// Definition's `instance_variables` (not `class_variables`),
    /// matching rbs `build_singleton0`.
    ClassInstanceVariable(ClassInstanceVariableMember),
    /// Signature-side `@@foo: T`. Populated into the instance-side
    /// Definition's `class_variables`. rbs's `build_singleton0` does
    /// not propagate these to the singleton side.
    ClassVariable(ClassVariableMember),
}

/// A single member of a class body.
///
/// Mirrors `RBS::AST::Declarations::Class::member` from the rbs Rust port.
/// Classes accept all [`Member`] forms plus any nested [`Declaration`].
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ClassMember {
    Member(Member),
    Declaration(Declaration),
}

/// A single member of a module body.
///
/// Mirrors `RBS::AST::Declarations::Module::member` from the rbs Rust port.
/// Modules accept all [`Member`] forms plus any nested [`Declaration`].
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ModuleMember {
    Member(Member),
    Declaration(Declaration),
}

/// Access the inner [`Member`] of a `ClassMember` or `ModuleMember`.
///
/// Enables generic functions that process member lists without duplicating
/// code for the two wrapper types. This trait has no equivalent in the rbs
/// Rust port; it is a crema-only helper introduced to avoid duplicating
/// consumer logic for the two wrapper enums.
pub trait AsMember {
    fn as_member(&self) -> Option<&Member>;
}

impl AsMember for ClassMember {
    fn as_member(&self) -> Option<&Member> {
        if let ClassMember::Member(m) = self {
            Some(m)
        } else {
            None
        }
    }
}

impl AsMember for ModuleMember {
    fn as_member(&self) -> Option<&Member> {
        if let ModuleMember::Member(m) = self {
            Some(m)
        } else {
            None
        }
    }
}

impl From<ClassMember> for ModuleMember {
    fn from(m: ClassMember) -> Self {
        match m {
            ClassMember::Member(m) => ModuleMember::Member(m),
            ClassMember::Declaration(d) => ModuleMember::Declaration(d),
        }
    }
}

/// `class C < Super` / `class C < Super[T]`.
///
/// Mirrors `RBS::AST::Declarations::Class::Super`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassSuper {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<ClassSuperLocation>,
    pub source_file: Option<Name>,
}

/// `module M : SelfType` / `module M : SelfType[T]`.
///
/// Mirrors `RBS::AST::Declarations::Module::Self`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleSelf {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<ModuleSelfLocation>,
    pub source_file: Option<Name>,
}

/// One top-level declaration in a `.rbs` file.
///
/// Mirrors `RBS::AST::Declarations::t`
/// (`Class | Module | Interface | Constant | Global | TypeAlias |
/// ClassAlias | ModuleAlias`, declared in `rbs/sig/declarations.rbs:4`).
/// The eight variants are kept in the same order as rbs's union so
/// readers can cross-reference the two layers without translating
/// names.
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Class(Arc<ClassDeclaration>),
    Module(Arc<ModuleDeclaration>),
    Interface(Arc<InterfaceDeclaration>),
    Constant(Arc<ConstantDeclaration>),
    Global(Arc<GlobalDeclaration>),
    TypeAlias(Arc<TypeAliasDeclaration>),
    ClassAlias(Arc<ClassAliasDeclaration>),
    ModuleAlias(Arc<ModuleAliasDeclaration>),
}
