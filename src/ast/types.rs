//! Unresolved type AST — mirrors `RBS::Types::*`.
//!
//! These values are produced by parsers before name resolution. They differ
//! from the resolved [`crate::types::Type`] / [`crate::types::Ty`] pair in two
//! ways:
//!
//! 1. Names use [`crate::type_name::TypeName`] with a possibly-relative
//!    [`crate::type_name::Namespace`] (`absolute: false`). Turning it into a
//!    canonical absolute [`crate::name::Name`] is the resolver's job.
//! 2. Children are **inlined**, not interned: `Box<Type>` / `Vec<Type>` rather
//!    than `Ty` handles. Phase 1 chooses owned enums over interning because
//!    the AST is short-lived (per-file parse output) and intern tables would
//!    only add complexity until a measurable need appears.
//!
//! The shape is deliberately structurally parallel to `crate::types::Type` so
//! a Phase 2 resolver can convert variant-for-variant. Naming matches rbs
//! (`RBS::Types::*`) first and crema's existing `Ty` naming second, per the
//! migration plan in ADR-0014.
//!
//! Each variant carries `location: Option<Location>` mirroring rbs's
//! per-type-node `attr_reader :location` pattern (`lib/rbs/types.rb`). The
//! resolved layer (`crate::types::Type`) deliberately omits location because
//! interned values are shared across call sites; per-occurrence locations live
//! on surrounding diagnostic / argument records.

use crate::location::{
    AliasLocation, ClassInstanceLocation, ClassSingletonLocation, FunctionParamLocation,
    InterfaceLocation, LocationRange,
};
use crate::name::Symbol;
use crate::type_name::TypeName;

pub mod substitution;


/// An unresolved type AST node. Mirrors `RBS::Types::*`.
///
/// Structurally parallels [`crate::types::Type`]: same set of variants, same
/// roles, but with relative-or-absolute [`TypeName`] in place of resolved
/// `Name`, owned recursion (`Box<Type>` / `Vec<Type>`) in place of interned
/// `Ty`, and per-node `location` mirroring rbs's type-node location pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Base(BaseType),
    Variable(VariableType),
    ClassSingleton(ClassSingletonType),
    Interface(InterfaceType),
    ClassInstance(ClassInstanceType),
    Alias(AliasType),
    Tuple(TupleType),
    Record(RecordType),
    Optional(OptionalType),
    Union(UnionType),
    Intersection(IntersectionType),
    Proc(Box<ProcType>),
    Literal(LiteralType),
}

/// `RBS::Types::Bases::*` — the nine base types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BaseType {
    pub kind: BaseTypeKind,
    pub location: Option<LocationRange>,
}

/// Variants of [`BaseType`], mirroring `RBS::Types::Bases::*`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BaseTypeKind {
    Bool,
    Void,
    /// `RBS::Types::Bases::Any`. The `todo` flag mirrors the FFI field; all
    /// input currently sets it `false`. The rbs keyword is `untyped`; `any` is
    /// a deprecated synonym.
    Any {
        todo: bool,
    },
    Nil,
    Top,
    Bottom,
    /// `RBS::Types::Bases::Self`. `SelfType` avoids collision with the Rust
    /// keyword `Self`.
    SelfType,
    Instance,
    Class,
}

/// `RBS::Types::Variable` — a type variable like `T`, `U`.
///
/// rbs `Variable` holds a `Symbol` (not a `TypeName`); crema mirrors that.
/// Disambiguation from `ClassInstance` happens at the parser / resolver
/// boundary: the parser marks the node as a variable when it can (e.g.
/// inside a generic parameter list), and the resolver reclassifies
/// otherwise-ambiguous `ClassInstance` nodes against the enclosing scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariableType {
    pub name: Symbol,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::ClassSingleton` — `singleton(Foo)`.
///
/// The `args` field mirrors rbs Rust `ClassSingletonType::args` and is
/// populated by the AST builder from the FFI layer. The resolved
/// `Type::ClassSingleton` drops args (singleton types are not parameterised
/// at the type-checker level).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassSingletonType {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<ClassSingletonLocation>,
}

/// `RBS::Types::Interface` — `_ToStr`, `_Each[String, void]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InterfaceType {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<InterfaceLocation>,
}

/// `RBS::Types::ClassInstance` — `Array`, `Array[Integer]`, `::Foo::Bar[T]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassInstanceType {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<ClassInstanceLocation>,
}

/// `RBS::Types::Alias` — `int`, `array[String]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasType {
    pub name: TypeName,
    pub args: Vec<Type>,
    pub location: Option<AliasLocation>,
}

/// `RBS::Types::Tuple` — fixed-length heterogeneous array `[A, B, C]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleType {
    pub types: Vec<Type>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Record` — `{ foo: A, ?bar: B, "baz" => C }`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordType {
    pub fields: Vec<RecordField>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Optional` — `A?` (syntactic sugar for `A | nil`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptionalType {
    pub ty: Box<Type>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Union` — `A | B | ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnionType {
    pub types: Vec<Type>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Intersection` — `A & B & ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntersectionType {
    pub types: Vec<Type>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Proc` — `^(A) -> B`, optionally with `[self: T]` binding
/// and a trailing block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProcType {
    pub function: Function,
    pub block: Option<BlockType>,
    pub self_type: Option<Box<Type>>,
    pub location: Option<LocationRange>,
}

/// `RBS::Types::Literal` — `1`, `"hello"`, `:sym`, `true`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LiteralType {
    pub literal: Literal,
    pub location: Option<LocationRange>,
}

/// Mirrors [`crate::types::Literal`]. Reproduced here so the ast layer is
/// free of dependencies on the resolved `Ty` module — a Phase 2 resolver
/// converts variant-for-variant.
///
/// Mirrors `rbs Rust` `ast::types::Literal`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Integer(String),
    String(String),
    Symbol(Symbol),
    Bool(bool),
}

/// Mirrors `rbs Rust` `ast::types::RecordKey`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RecordKey {
    Symbol(Symbol),
    String(String),
    Integer(String),
    Bool(bool),
}

/// `RBS::Types::Record::Field` — one entry in a record type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordField {
    pub key: RecordKey,
    pub ty: Type,
    pub required: bool,
}

/// `RBS::Types::Function | RBS::Types::UntypedFunction`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Function {
    Typed(FunctionType),
    Untyped(UntypedFunctionType),
}

/// `RBS::Types::Function` — the full parameter layout plus a return type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub required_positionals: Vec<FunctionParam>,
    pub optional_positionals: Vec<FunctionParam>,
    pub rest_positionals: Option<Box<FunctionParam>>,
    pub trailing_positionals: Vec<FunctionParam>,
    pub required_keywords: Vec<KeywordParam>,
    pub optional_keywords: Vec<KeywordParam>,
    pub rest_keywords: Option<Box<FunctionParam>>,
    pub return_type: Box<Type>,
}

/// `RBS::Types::Function::Param` — a positional parameter with optional name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParam {
    pub ty: Box<Type>,
    pub name: Option<Symbol>,
    pub location: Option<FunctionParamLocation>,
}

/// `RBS::Types::Function::KeywordParam` — a keyword parameter with name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeywordParam {
    pub name: Symbol,
    pub param: FunctionParam,
}

/// `RBS::Types::UntypedFunction` — `(?) -> T`, accepts any arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UntypedFunctionType {
    pub return_type: Box<Type>,
}

/// `RBS::Types::Block` — `{ (A) -> B }` or `{ (A) [self: T] -> B }`.
///
/// `required = false` corresponds to the `?{ ... }` form.
/// `self_type` is the `[self: T]` binding when present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockType {
    pub required: bool,
    pub function: Function,
    pub self_type: Option<Box<Type>>,
}
