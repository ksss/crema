//! Mirrors `RBS::AST::Members` (lib/rbs/ast/members.rb).
//!
//! Flat, suffix-named structs matching the rbs Rust port
//! (`rbs/rust/ruby-rbs/src/ast/members.rs`). Each struct holds its
//! fields directly (no shared payload structs like `Mixin`, `Attribute`,
//! or `LocationOnly`).

use crate::ast::annotation::Annotation;
use crate::ast::comment::Comment;
use crate::ast::method_type::MethodType;
use crate::ast::types::Type;
use crate::location::{
    AliasMemberLocation, AttributeMemberLocation, LocationRange, MethodDefinitionLocation,
    MixinMemberLocation, VariableMemberLocation,
};
use crate::name::{Name, Symbol};
use crate::type_name::TypeName;

/// `public` / `private` keyword.
///
/// Mirrors `RBS::AST::Members::Public` / `Private` plus the
/// per-method visibility annotation. Defaulted to `Public` by the
/// resolver when no explicit marker is in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Visibility {
    Public,
    Private,
}

/// Whether a method or alias targets the instance, the singleton,
/// or both.
///
/// - `Instance` — `def name` (defined on the instance only).
/// - `Singleton` — `def self.name` (defined on the singleton class).
/// - `SingletonInstance` — `def self?.name` (RBS shorthand that
///   defines the same signature on both the instance and the
///   singleton). The build layer fans this out into separate
///   instance and singleton definitions, matching rbs's
///   `:singleton_instance` handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MethodKind {
    Instance,
    Singleton,
    SingletonInstance,
}

/// Receiver-side discriminator for attributes.
///
/// Mirrors `RBS::AST::Members::AttributeKind`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
/// RBS rejects `singleton?`-style shorthand for attrs so only
/// `Instance` and `Singleton` are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AttributeKind {
    Instance,
    Singleton,
}

/// Receiver-side discriminator for method aliases.
///
/// Mirrors `RBS::AST::Members::AliasKind`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AliasKind {
    Instance,
    Singleton,
}

/// Instance-variable backing for attr members.
///
/// Mirrors `RBS::AST::Members::IvarName`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
///
/// - `Unspecified` — implicit `@name` from attr's name (`attr_reader foo: T`).
/// - `Name(sym)` — explicit `attr_reader foo (@bar): T`.
/// - `Empty` — `attr_reader foo (): T` (function only, no ivar).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IvarName {
    Unspecified,
    Empty,
    Name(Symbol),
}

/// A single overload of a method definition.
///
/// Mirrors `RBS::AST::Members::MethodDefinitionOverload`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`). RBS pairs each overload
/// with per-overload annotations (e.g. `%a{noreturn}` written above
/// one specific signature line), so the port keeps the annotation
/// alongside the method type rather than collapsing the pair into a
/// bare `MethodType`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MethodDefinitionOverload {
    pub method_type: MethodType,
    pub annotations: Vec<Annotation>,
}

/// Mirrors `RBS::AST::Members::MethodDefinitionMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
///
/// `visibility` is `Option<Visibility>` to mirror RBS's nilable field —
/// the source-level distinction between bare `def foo` and explicitly
/// annotated `private def foo`. Today the rbs_loader populates
/// `Some(_)` eagerly by folding the surrounding `private` / `public`
/// marker (and method-kind defaults) at build time, so downstream
/// code observes a concrete visibility on every method.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MethodDefinitionMember {
    pub name: Symbol,
    pub kind: MethodKind,
    pub overloads: Vec<MethodDefinitionOverload>,
    pub annotations: Vec<Annotation>,
    pub overloading: bool,
    pub visibility: Option<Visibility>,
    pub location: Option<MethodDefinitionLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `include M` inside a class / module body.
///
/// Mirrors `RBS::AST::Members::IncludeMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncludeMember {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub annotations: Vec<Annotation>,
    pub location: Option<MixinMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `extend M` inside a class / module body.
///
/// Mirrors `RBS::AST::Members::ExtendMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExtendMember {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub annotations: Vec<Annotation>,
    pub location: Option<MixinMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `prepend M` inside a class / module body.
///
/// Mirrors `RBS::AST::Members::PrependMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PrependMember {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub annotations: Vec<Annotation>,
    pub location: Option<MixinMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// `attr_reader name: T` with optional explicit ivar and visibility.
///
/// Mirrors `RBS::AST::Members::AttrReaderMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
///
/// `visibility` mirrors RBS's nilable field (`:public | :private | nil`):
/// `None` for unannotated attributes, `Some(_)` for explicit annotations.
/// The build layer preserves raw values without folding against surrounding
/// markers; folding is deferred to the definition layer, mirroring
/// RBS gem's `DefinitionBuilder`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttrReaderMember {
    pub name: Symbol,
    pub ty: Type,
    pub ivar_name: IvarName,
    pub kind: AttributeKind,
    pub annotations: Vec<Annotation>,
    pub location: Option<AttributeMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
    pub visibility: Option<Visibility>,
}

/// `attr_writer name: T` with optional explicit ivar and visibility.
///
/// Mirrors `RBS::AST::Members::AttrWriterMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttrWriterMember {
    pub name: Symbol,
    pub ty: Type,
    pub ivar_name: IvarName,
    pub kind: AttributeKind,
    pub annotations: Vec<Annotation>,
    pub location: Option<AttributeMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
    pub visibility: Option<Visibility>,
}

/// `attr_accessor name: T` with optional explicit ivar and visibility.
///
/// Mirrors `RBS::AST::Members::AttrAccessorMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttrAccessorMember {
    pub name: Symbol,
    pub ty: Type,
    pub ivar_name: IvarName,
    pub kind: AttributeKind,
    pub annotations: Vec<Annotation>,
    pub location: Option<AttributeMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
    pub visibility: Option<Visibility>,
}

/// Bare `public` visibility marker.
///
/// Mirrors `RBS::AST::Members::PublicMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PublicMember {
    pub location: Option<LocationRange>,
}

/// Bare `private` visibility marker.
///
/// Mirrors `RBS::AST::Members::PrivateMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PrivateMember {
    pub location: Option<LocationRange>,
}

/// `alias new_name old_name` inside a class / module body.
///
/// Mirrors `RBS::AST::Members::AliasMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
///
/// `visibility` is absent — rbs does not allow explicit visibility on
/// alias declarations (unlike MethodDefinition and AttrReader/Writer/Accessor).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AliasMember {
    pub new_name: Symbol,
    pub old_name: Symbol,
    pub kind: AliasKind,
    pub annotations: Vec<Annotation>,
    pub location: Option<AliasMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// Signature-side `@foo: T` declaration.
///
/// Mirrors `RBS::AST::Members::InstanceVariableMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
/// Populated into the instance-side Definition's `instance_variables`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InstanceVariableMember {
    pub name: Symbol,
    pub ty: Type,
    pub location: Option<VariableMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// Signature-side `self.@foo: T` declaration.
///
/// Mirrors `RBS::AST::Members::ClassInstanceVariableMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
/// Declares an instance variable on the class object itself (singleton-side
/// instance). rbs's `build_singleton0` registers these in the singleton-side
/// Definition's `instance_variables` (not `class_variables`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassInstanceVariableMember {
    pub name: Symbol,
    pub ty: Type,
    pub location: Option<VariableMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}

/// Signature-side `@@foo: T` declaration.
///
/// Mirrors `RBS::AST::Members::ClassVariableMember`
/// (`rbs/rust/ruby-rbs/src/ast/members.rs`).
/// Stored on the instance-side `class_variables`; rbs's `build_singleton0`
/// does not propagate these to the singleton side.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassVariableMember {
    pub name: Symbol,
    pub ty: Type,
    pub location: Option<VariableMemberLocation>,
    pub source_file: Option<Name>,
    pub comment: Option<Comment>,
}
