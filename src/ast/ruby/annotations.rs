//! Parsed-and-translated leading inline annotations.
//!
//! Mirrors the variants of `RBS::AST::Ruby::Annotations::*` that the
//! existing `src/inline_parser.rs::resolve_def_overloads` discriminates
//! via `rbs_node_type::RBS_AST_RUBY_ANNOTATIONS_*`.
//!
//! Other rbs annotation variants are intentionally absent from
//! [`LeadingAnnotation`]: they are ignored by `resolve_def_overloads` today
//! and stay out of Phase 3's scope to preserve observable behavior.
//! `crate::ast_builder::build_leading_annotation` returns `None` for those
//! kinds.

use crate::ast::annotation::Annotation;
use crate::ast::members::MethodDefinitionOverload;
use crate::ast::method_type::MethodType;
use crate::ast::ruby::PrismByteRange;
use crate::ast::types::{Function, Type};
use crate::location::RubyLocation;
use crate::type_name::TypeName;

/// Whether an inline alias declaration was annotated as a class alias
/// (`#: class-alias`) or a module alias (`#: module-alias`). Derived from
/// [`AliasAnnotation`]'s variant — kept as a standalone enum so the
/// diagnostic surface (`InlineClassAliasMissingTypeName { kind }`) can
/// name the alias kind in error messages without re-reading the owned
/// annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InlineAliasKind {
    Class,
    Module,
}


/// A `# @rbs …` or `#: …` paragraph, translated from the C-side
/// annotation node into crema's unresolved ast layer.
///
/// Each variant wraps a dedicated `*Annotation` struct that carries the
/// `Base.location` / `Base.prefix_location` shared by every rbs
/// annotation, plus variant-specific sub-locations. This mirrors rbs's
/// `RBS::AST::Ruby::Annotations::Base` inheritance hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LeadingAnnotation {
    /// `#: (T) -> U` — a single method-type assertion.
    ColonMethodType(Box<ColonMethodTypeAnnotation>),
    /// `# @rbs (T) -> U` or `# @rbs (A) -> B | (C) -> D` — one or more
    /// overload signatures.
    MethodTypes(MethodTypesAnnotation),
    /// `# @rbs foo: T` — per-parameter type annotation.
    ParamType(ParamTypeAnnotation),
    /// `# @rbs return: T` — return-type annotation.
    ReturnType(ReturnTypeAnnotation),
    /// `# @rbs @foo: T` — instance-variable annotation.
    InstanceVariable(InstanceVariableAnnotation),
    /// `# @rbs &block: T` — block-parameter annotation.
    BlockParamType(BlockParamTypeAnnotation),
    /// `# @rbs *name: T` (name optional) — rest-positional annotation.
    SplatParamType(SplatParamTypeAnnotation),
    /// `# @rbs **name: T` (name optional) — rest-keyword annotation.
    DoubleSplatParamType(DoubleSplatParamTypeAnnotation),
    /// `# @rbs skip` — requests that the method's leading annotations
    /// be discarded and the method fall back to `untyped`.
    Skip(SkipAnnotation),
    /// `# @rbs module-self: NAME[ARGS]` — module-self constraint.
    ModuleSelf(ModuleSelfAnnotation),
}

impl LeadingAnnotation {
    /// The byte range of the annotation in the source file.
    /// Mirrors `RBS::AST::Ruby::Annotations::Base#location`.
    pub fn location(&self) -> PrismByteRange {
        match self {
            LeadingAnnotation::ColonMethodType(a) => a.location,
            LeadingAnnotation::MethodTypes(a) => a.location,
            LeadingAnnotation::ParamType(a) => a.location,
            LeadingAnnotation::ReturnType(a) => a.location,
            LeadingAnnotation::InstanceVariable(a) => a.location,
            LeadingAnnotation::BlockParamType(a) => a.location,
            LeadingAnnotation::SplatParamType(a) => a.location,
            LeadingAnnotation::DoubleSplatParamType(a) => a.location,
            LeadingAnnotation::Skip(a) => a.location,
            LeadingAnnotation::ModuleSelf(a) => a.location,
        }
    }
}

/// `#: (T) -> U` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::ColonMethodTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColonMethodTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub annotations: Vec<Annotation>,
    pub method_type: MethodType,
}

/// `# @rbs (T) -> U` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::MethodTypesAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MethodTypesAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub overloads: Vec<MethodDefinitionOverload>,
    pub vertical_bar_locations: Vec<PrismByteRange>,
    pub dot3_location: Option<PrismByteRange>,
}

/// `# @rbs foo: T` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::ParamTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParamTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub name_location: PrismByteRange,
    pub colon_location: PrismByteRange,
    pub name: String,
    pub param_type: Type,
    pub comment_location: Option<PrismByteRange>,
}

/// `# @rbs return: T` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::ReturnTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReturnTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub return_location: PrismByteRange,
    pub colon_location: PrismByteRange,
    pub return_type: Type,
    pub comment_location: Option<PrismByteRange>,
}

/// `# @rbs *name: T` (name optional) leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::SplatParamTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SplatParamTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub star_location: PrismByteRange,
    pub name_location: Option<PrismByteRange>,
    pub colon_location: PrismByteRange,
    pub name: Option<String>,
    pub param_type: Type,
    pub comment_location: Option<PrismByteRange>,
}

/// `# @rbs **name: T` (name optional) leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::DoubleSplatParamTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DoubleSplatParamTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub star2_location: PrismByteRange,
    pub name_location: Option<PrismByteRange>,
    pub colon_location: PrismByteRange,
    pub name: Option<String>,
    pub param_type: Type,
    pub comment_location: Option<PrismByteRange>,
}

/// The subset of [`LeadingAnnotation`] variants that are valid as
/// parameter-type annotations inside `DocStyle::build`.
///
/// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation::DocStyle::param_type_annotation`
/// (the 4-union type alias in rbs `sig/ast/ruby/members.rbs:23-26`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParamTypeKindAnnotation {
    Param(ParamTypeAnnotation),
    Splat(SplatParamTypeAnnotation),
    DoubleSplat(DoubleSplatParamTypeAnnotation),
    Block(BlockParamTypeAnnotation),
}

/// `# @rbs skip` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::SkipAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SkipAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub skip_location: PrismByteRange,
    pub comment_location: Option<PrismByteRange>,
}

/// Test-only constructors that fill location fields with `(0, 0)`. Hide
/// the boilerplate from fixtures so test bodies stay focused on the
/// behaviorally relevant fields (`name`, `param_type`, etc.). Production
/// code must build the structs literally so the location channel can be
/// audited statically.
#[cfg(test)]
impl ParamTypeAnnotation {
    pub fn for_test(name: impl Into<String>, param_type: Type) -> Self {
        ParamTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            name_location: (0, 0),
            colon_location: (0, 0),
            name: name.into(),
            param_type,
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl ReturnTypeAnnotation {
    pub fn for_test(return_type: Type) -> Self {
        ReturnTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            return_location: (0, 0),
            colon_location: (0, 0),
            return_type,
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl SplatParamTypeAnnotation {
    /// `name_location` is derived from `name.is_some()` so the test fixture
    /// keeps rbs's `Some(name) ⇒ Some(name_location)` invariant: rbs parses
    /// the splat name from the same `name_location` range, so a `Some` name
    /// never coexists with a missing location at runtime.
    pub fn for_test(name: Option<String>, param_type: Type) -> Self {
        let name_location = name.as_ref().map(|_| (0, 0));
        SplatParamTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            star_location: (0, 0),
            name_location,
            colon_location: (0, 0),
            name,
            param_type,
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl DoubleSplatParamTypeAnnotation {
    /// `name_location` is derived from `name.is_some()` so the test fixture
    /// keeps rbs's `Some(name) ⇒ Some(name_location)` invariant.
    pub fn for_test(name: Option<String>, param_type: Type) -> Self {
        let name_location = name.as_ref().map(|_| (0, 0));
        DoubleSplatParamTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            star2_location: (0, 0),
            name_location,
            colon_location: (0, 0),
            name,
            param_type,
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl BlockParamTypeAnnotation {
    pub fn for_test(name: Option<String>, function: Function, required: bool) -> Self {
        let name_location = name.as_ref().map(|_| (0, 0));
        BlockParamTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            ampersand_location: (0, 0),
            name_location,
            colon_location: (0, 0),
            question_location: (!required).then_some((0, 0)),
            type_location: (0, 0),
            name,
            function,
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl SkipAnnotation {
    pub fn for_test() -> Self {
        SkipAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            skip_location: (0, 0),
            comment_location: None,
        }
    }
}

#[cfg(test)]
impl ColonMethodTypeAnnotation {
    pub fn for_test(annotations: Vec<Annotation>, method_type: MethodType) -> Self {
        ColonMethodTypeAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            annotations,
            method_type,
        }
    }
}

#[cfg(test)]
impl MethodTypesAnnotation {
    pub fn for_test(
        overloads: Vec<MethodDefinitionOverload>,
        vertical_bar_locations: Vec<PrismByteRange>,
        dot3_location: Option<PrismByteRange>,
    ) -> Self {
        MethodTypesAnnotation {
            location: (0, 0),
            prefix_location: (0, 0),
            overloads,
            vertical_bar_locations,
            dot3_location,
        }
    }
}

/// `#[T]` / `#[T, U]` trailing annotation on a mixin reference or a
/// class's super reference, owned-snapshot form.
///
/// Mirrors `RBS::AST::Ruby::Annotations::TypeApplicationAnnotation`.
/// `location` corresponds to rbs's `Base.location` (the whole-annotation
/// byte range, including the `#[` prefix and the closing `]`).
/// rbs's `prefix_location`, `close_bracket_location`, and
/// `comma_locations` are not ported here — adding them requires
/// trailing-annotation parser changes and is left as a follow-up.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TypeApplicationAnnotation {
    pub type_args: Vec<Type>,
    pub location: PrismByteRange,
}

/// `@rbs @ivar: T` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::InstanceVariableAnnotation`
/// for the payload needed by `InstanceVariableMember`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InstanceVariableAnnotation {
    pub name: String,
    pub location: PrismByteRange,
    pub name_location: PrismByteRange,
    pub source_location: Option<RubyLocation>,
    pub ty: Type,
}

/// `@rbs &block: T` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::BlockParamTypeAnnotation`
/// for the payload needed by doc-style method annotations.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockParamTypeAnnotation {
    pub location: PrismByteRange,
    pub prefix_location: PrismByteRange,
    pub ampersand_location: PrismByteRange,
    pub name_location: Option<PrismByteRange>,
    pub colon_location: PrismByteRange,
    pub question_location: Option<PrismByteRange>,
    pub type_location: PrismByteRange,
    pub name: Option<String>,
    pub function: Function,
    pub comment_location: Option<PrismByteRange>,
}

/// `@rbs module-self: NAME[ARGS]` leading annotation.
///
/// Mirrors `RBS::AST::Ruby::Annotations::ModuleSelfAnnotation`
/// for the payload needed by `ModuleSelfMember`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModuleSelfAnnotation {
    pub name: TypeName,
    pub name_location: PrismByteRange,
    pub args: Vec<Type>,
    pub location: PrismByteRange,
}

/// `#: class-alias` / `#: module-alias` trailing annotation on a constant
/// assignment, owned-snapshot form.
///
/// Mirrors `RBS::AST::Ruby::Annotations::AliasAnnotation` and its two
/// concrete subclasses `ClassAliasAnnotation` / `ModuleAliasAnnotation`.
/// The class/module distinction is encoded by the enum variant — same
/// shape as rbs's `is_a?(ClassAliasAnnotation)` dispatch from
/// `RBS::AST::Ruby::Declarations::ClassModuleAliasDecl#annotation`.
///
/// Carries the minimal set of fields the current resolver and
/// diagnostic paths consume: the annotation byte-range and, when
/// written, the explicit `type_name` argument
/// (`#: class-alias Foo`'s `Foo`). rbs's `keyword_location` and
/// `type_name_location` sub-locations are not ported here — adding
/// them is a follow-up child todo so the inline parser's trailing
/// annotation parser can be extended in one pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AliasAnnotation {
    Class(AliasAnnotationFields),
    Module(AliasAnnotationFields),
}

/// Shared payload for [`AliasAnnotation::Class`] / [`AliasAnnotation::Module`].
///
/// Mirrors the fields rbs's `AliasAnnotation` superclass exposes that
/// crema currently consumes. `type_name_text` corresponds to
/// `annotation.type_name` and is `None` when the user wrote a bare
/// `#: class-alias` / `#: module-alias` without an explicit target.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AliasAnnotationFields {
    pub location: PrismByteRange,
    pub type_name_text: Option<String>,
}

impl AliasAnnotation {
    /// Build the variant matching `kind` with the given fields. Folds the
    /// `InlineAliasKind` → `AliasAnnotation` mapping in one place so the
    /// inline collector does not repeat the match.
    pub fn new(kind: InlineAliasKind, fields: AliasAnnotationFields) -> AliasAnnotation {
        match kind {
            InlineAliasKind::Class => AliasAnnotation::Class(fields),
            InlineAliasKind::Module => AliasAnnotation::Module(fields),
        }
    }

    pub fn kind(&self) -> InlineAliasKind {
        match self {
            AliasAnnotation::Class(_) => InlineAliasKind::Class,
            AliasAnnotation::Module(_) => InlineAliasKind::Module,
        }
    }

    pub fn type_name_text(&self) -> Option<&str> {
        self.fields().type_name_text.as_deref()
    }

    pub fn location(&self) -> PrismByteRange {
        self.fields().location
    }

    fn fields(&self) -> &AliasAnnotationFields {
        match self {
            AliasAnnotation::Class(f) | AliasAnnotation::Module(f) => f,
        }
    }

    /// Rebuild the annotation with the same variant kind and location but
    /// a different `type_name_text`. Mirrors rbs's
    /// `AliasAnnotation#map_type_name` block-form rewrite; used by
    /// `resolve_ruby_alias_decl` to absolutize the type-name reference
    /// against an enclosing context.
    pub fn map_type_name_text(&self, type_name_text: Option<String>) -> AliasAnnotation {
        match self {
            AliasAnnotation::Class(f) => AliasAnnotation::Class(AliasAnnotationFields {
                location: f.location,
                type_name_text,
            }),
            AliasAnnotation::Module(f) => AliasAnnotation::Module(AliasAnnotationFields {
                location: f.location,
                type_name_text,
            }),
        }
    }
}
