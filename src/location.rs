use std::path::PathBuf;

use crate::environment::InfusionUnit;
use crate::name::Name;

/// Byte and character range in the source buffer, without file identity.
///
/// Mirrors `LocationRange` in rbs Rust (`ruby-rbs/src/ast/location.rs`).
/// Four-field form preserves char offsets alongside byte offsets so that
/// column-number diagnostics remain possible without re-scanning.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LocationRange {
    pub start_char: u32,
    pub start_byte: u32,
    pub end_char: u32,
    pub end_byte: u32,
}

impl LocationRange {
    pub fn new(start_char: u32, start_byte: u32, end_char: u32, end_byte: u32) -> Self {
        Self {
            start_char,
            start_byte,
            end_char,
            end_byte,
        }
    }
}

/// A source location that includes both a file path and a range within it.
///
/// Used by consumer layers (diagnostics, definitions, environment) that need
/// to report or store the file alongside the byte/char range. The ast/ layer
/// stores only [`LocationRange`] (file-less); consumers attach the file when
/// they build a `SourceLocation`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub range: LocationRange,
}

/// The other side of a `Ruby::DuplicatedMethodDefinitionError` collision:
/// either a real source location the user can jump to, or crema's own
/// infusion synthesis (no file to point at, but the collector is known).
/// Collapsing both onto a bare `Option<SourceLocation>` (the pre-
/// `mid_dup_method_infusion_provenance` shape) made a collision with a
/// synthesized method indistinguishable from "no location for other
/// reasons" — this type keeps that distinction on the wire.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DuplicateSource {
    Location(SourceLocation),
    Synthesized { infusion: InfusionUnit },
}

/// File-bearing location with the same shape as rbs Ruby's `RBS::Location`.
///
/// `RubyLocation` keeps file identity as an interned [`Name`], matching the
/// rbs Ruby side where `RBS::Location` carries a `Buffer`. It is used only
/// where crema still needs this interned-file + byte-range shape, such as
/// ast/ruby nodes, source registration, and inline Ruby diagnostic plumbing.
/// The rbs Rust port path (`ast/`) uses file-less [`LocationRange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RubyLocation {
    pub file: Name,
    pub start_byte: u32,
    pub end_byte: u32,
}

// ──────────────────────────────────────────────────────────────────────────
// Typed location structs — mirrors `ruby-rbs/src/ast/location.rs`
// ──────────────────────────────────────────────────────────────────────────

/// ```rbs
/// foo
/// ^^^ name
///
/// foo[bar, baz]
/// ^^^           name
///    ^^^^^^^^^^ args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AliasLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// Foo
/// ^^^ name
///
/// Foo[Bar, Baz]
/// ^^^           name
///    ^^^^^^^^^^ args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClassInstanceLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// singleton(::Foo)
///           ^^^^^  name
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClassSingletonLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// _Foo
/// ^^^^ name
///
/// _Foo[Bar, Baz]
/// ^^^^           name
///     ^^^^^^^^^^ args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct InterfaceLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// String name
///        ^^^^ name
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FunctionParamLocation {
    pub range: LocationRange,
    pub name_range: Option<LocationRange>,
}

/// ```rbs
/// () -> void
/// ^^^^^^^^^^     type
///
/// [A] () { () -> A } -> A
/// ^^^                      type_params
///     ^^^^^^^^^^^^^^^^^^^  type
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MethodTypeLocation {
    pub range: LocationRange,
    pub type_range: LocationRange,
    pub type_params_range: Option<LocationRange>,
}

/// ```rbs
/// Key
/// ^^^ name
///
/// unchecked out Elem < _ToJson > bot = untyped
/// ^^^^^^^^^                                        unchecked
///           ^^^                                    variance
///               ^^^^                               name
///                    ^^^^^^^^^                     upper_bound
///                              ^^^^^               lower_bound
///                                      ^^^^^^^^    default
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TypeParamLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub variance_range: Option<LocationRange>,
    pub unchecked_range: Option<LocationRange>,
    pub upper_bound_range: Option<LocationRange>,
    pub lower_bound_range: Option<LocationRange>,
    pub default_range: Option<LocationRange>,
}

/// ```rbs
/// String
/// ^^^^^^  name
///
/// Array[String]
/// ^^^^^         name
///      ^^^^^^^^ args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClassSuperLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// class Foo end
/// ^^^^^         keyword
///       ^^^     name
///           ^^^ end
///
/// class Foo[A] < String end
/// ^^^^^                     keyword
///       ^^^                 name
///          ^^^              type_params
///              ^            lt
///                       ^^^ end
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClassDeclarationLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub end_range: LocationRange,
    pub type_params_range: Option<LocationRange>,
    pub lt_range: Option<LocationRange>,
}

/// ```rbs
/// _Each[String]
/// ^^^^^         name
///      ^^^^^^^^ args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModuleSelfLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// module Foo end
/// ^^^^^^         keyword
///        ^^^     name
///            ^^^ end
///
/// module Foo[A] : BasicObject end
/// ^^^^^^                          keyword
///        ^^^                      name
///           ^^^                   type_params
///               ^                 colon
///                 ^^^^^^^^^^^     self_types
///                             ^^^ end
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModuleDeclarationLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub end_range: LocationRange,
    pub type_params_range: Option<LocationRange>,
    pub colon_range: Option<LocationRange>,
    pub self_types_range: Option<LocationRange>,
}

/// ```rbs
/// interface _Foo end
/// ^^^^^^^^^          keyword
///           ^^^^     name
///                ^^^ end
///
/// interface _Bar[A, B] end
/// ^^^^^^^^^                keyword
///           ^^^^           name
///               ^^^^^^     type_params
///                      ^^^ end
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct InterfaceDeclarationLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub end_range: LocationRange,
    pub type_params_range: Option<LocationRange>,
}

/// ```rbs
/// type loc[T] = Location[T, bot]
/// ^^^^                            keyword
///      ^^^                        name
///         ^^^                     type_params
///             ^                   eq
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TypeAliasDeclarationLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub eq_range: LocationRange,
    pub type_params_range: Option<LocationRange>,
}

/// ```rbs
/// VERSION: String
/// ^^^^^^^         name
///        ^        colon
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ConstantDeclarationLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub colon_range: LocationRange,
}

/// ```rbs
/// $SIZE: String
/// ^^^^^         name
///      ^        colon
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GlobalDeclarationLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub colon_range: LocationRange,
}

/// ```rbs
/// module Foo = Bar
/// ^^^^^^             keyword
///        ^^^         new_name
///            ^       eq
///              ^^^   old_name
///
/// class Foo = Bar
/// ^^^^^              keyword
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AliasDeclarationLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub new_name_range: LocationRange,
    pub eq_range: LocationRange,
    pub old_name_range: LocationRange,
}

/// ```rbs
/// def foo: () -> void
/// ^^^                    keyword
///     ^^^                name
///
/// private def self.bar: () -> void | ...
/// ^^^^^^^                                  visibility
///         ^^^                              keyword
///             ^^^^^                        kind
///                  ^^^                     name
///                                    ^^^   overloading
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MethodDefinitionLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub kind_range: Option<LocationRange>,
    pub overloading_range: Option<LocationRange>,
    pub visibility_range: Option<LocationRange>,
}

/// ```rbs
/// @foo: String
/// ^^^^            name
///     ^           colon
///
/// self.@all: Array[String]
/// ^^^^^                        kind
///      ^^^^                    name
///          ^                   colon
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VariableMemberLocation {
    pub range: LocationRange,
    pub name_range: LocationRange,
    pub colon_range: LocationRange,
    pub kind_range: Option<LocationRange>,
}

/// ```rbs
/// include Foo
/// ^^^^^^^       keyword
///         ^^^   name
///
/// include Array[String]
/// ^^^^^^^                keyword
///         ^^^^^          name
///              ^^^^^^^^  args
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MixinMemberLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub args_range: Option<LocationRange>,
}

/// ```rbs
/// attr_reader name: String
/// ^^^^^^^^^^^                  keyword
///             ^^^^             name
///                 ^            colon
///
/// public attr_accessor self.name (@foo) : String
/// ^^^^^^                                           visibility
///        ^^^^^^^^^^^^^                             keyword
///                      ^^^^^                       kind
///                           ^^^^                   name
///                                ^^^^^^            ivar
///                                 ^^^^             ivar_name
///                                       ^          colon
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AttributeMemberLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub name_range: LocationRange,
    pub colon_range: LocationRange,
    pub kind_range: Option<LocationRange>,
    pub ivar_range: Option<LocationRange>,
    pub ivar_name_range: Option<LocationRange>,
    pub visibility_range: Option<LocationRange>,
}

/// ```rbs
/// alias foo bar
/// ^^^^^           keyword
///       ^^^       new_name
///           ^^^   old_name
///
/// alias self.foo self.bar
/// ^^^^^                      keyword
///       ^^^^^                new_kind
///            ^^^             new_name
///                ^^^^^       old_kind
///                     ^^^    old_name
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AliasMemberLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub new_name_range: LocationRange,
    pub old_name_range: LocationRange,
    pub new_kind_range: Option<LocationRange>,
    pub old_kind_range: Option<LocationRange>,
}

/// ```rbs
/// use Foo
/// ^^^       keyword
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct UseDirectiveLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
}

/// ```rbs
/// Foo::Bar
/// ^^^^^^^^       type_name
///
/// Foo::Bar as X
///          ^^    keyword
///             ^  new_name
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct UseSingleClauseLocation {
    pub range: LocationRange,
    pub type_name_range: LocationRange,
    pub keyword_range: Option<LocationRange>,
    pub new_name_range: Option<LocationRange>,
}

/// ```rbs
/// Foo::Bar::*
/// ^^^^^^^^^^    namespace
///           ^   star
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct UseWildcardClauseLocation {
    pub range: LocationRange,
    pub namespace_range: LocationRange,
    pub star_range: LocationRange,
}

/// ```rbs
/// # resolve-type-names: false
///   ^^^^^^^^^^^^^^^^^^          keyword
///                     ^         colon
///                       ^^^^^   value
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ResolveTypeNamesDirectiveLocation {
    pub range: LocationRange,
    pub keyword_range: LocationRange,
    pub colon_range: LocationRange,
    pub value_range: LocationRange,
}
