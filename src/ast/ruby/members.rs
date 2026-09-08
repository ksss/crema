//! Ruby member AST. Mirrors `RBS::AST::Ruby::Members::*`.
//!
//! [`MethodTypeAnnotation`] classifies a method's leading annotations into
//! either explicit method-type signatures (`#:` / `# @rbs (T) -> U`) or a
//! doc-style assembly of per-parameter / return / rest annotations, and
//! offers an `overloads` accessor that resolves the classified form into
//! concrete `types::MethodType` values.
//!
//! [`Member`], [`DefMember`], [`AttributeMember`], and [`MixinMember`] are
//! ports of the concrete member classes in `lib/rbs/ast/ruby/members.rb`.
//! They form the leaves of the Ruby-side declaration tree defined in
//! `crate::ast::ruby::declarations`.

use rustc_hash::FxHashMap;

use ruby_prism::DefNode;

use crate::ast::MethodKind;
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::annotations::{
    BlockParamTypeAnnotation, ColonMethodTypeAnnotation, DoubleSplatParamTypeAnnotation,
    InstanceVariableAnnotation, LeadingAnnotation, MethodTypesAnnotation, ModuleSelfAnnotation,
    ParamTypeAnnotation, ParamTypeKindAnnotation, ReturnTypeAnnotation, SplatParamTypeAnnotation,
    TypeApplicationAnnotation,
};
use crate::ast::ruby::comment_block::CommentBlock;
use crate::ast::ruby::declarations::Declaration;
use crate::ast_builder;
use crate::inline_parser::TrailingAnnotation;
use crate::name::{Name, NameTable};
use crate::rbs_raw::Parser as RbsParser;


/// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodTypeAnnotation {
    pub type_annotations: TypeAnnotations,
}

/// The classification of annotations found on a method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeAnnotations {
    /// At least one explicit method-type annotation (`#: ...` or
    /// `# @rbs (T) -> U`). Stored in source order.
    Array(Vec<ExplicitAnnotation>),
    /// No explicit method-type annotation, but one or more doc-style
    /// annotations (`# @rbs foo: T`, `# @rbs return: T`, ...) which
    /// `DocStyle::build` has assembled against the def's parameter
    /// names. Boxed so this variant does not widen every `Member` slot
    /// to DocStyle's inline rest/block annotation width.
    DocStyle(Box<DocStyle>),
    /// No usable annotations. `overloads` returns `None` so the caller
    /// falls back to its default untyped MethodType.
    None,
}

/// Explicit method-type shape — either a single `#:` assertion or a
/// `# @rbs (…) -> …` possibly with `|`-separated overloads.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum ExplicitAnnotation {
    /// `#: (T) -> U`
    Colon(ColonMethodTypeAnnotation),
    /// `# @rbs (A) -> B | (C) -> D` — one or more overloads.
    MethodTypes(MethodTypesAnnotation),
}

/// Doc-style annotations assembled against the def's parameter names.
///
/// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation::DocStyle`.
/// Each positional / keyword slot is either `Annotated(ParamTypeAnnotation)`
/// (the source wrote `@rbs name: Type`) or `ByName(String)` (no annotation
/// was supplied and the slot falls back to untyped when resolved).
/// This preserves arity so the resolver can build a `Function` whose
/// slot count matches the def's formal parameter list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocStyle {
    /// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation::DocStyle#return_type_annotation`.
    pub return_type_annotation: Option<Box<ReturnTypeAnnotation>>,
    pub required_positionals: Vec<PositionalEntry>,
    pub optional_positionals: Vec<PositionalEntry>,
    /// Mirrors rbs's `rest_positionals: Annotations::SplatParamTypeAnnotation | Symbol | true | nil`.
    pub rest_positionals: Option<SplatRestEntry>,
    pub trailing_positionals: Vec<PositionalEntry>,
    pub required_keywords: Vec<(String, PositionalEntry)>,
    pub optional_keywords: Vec<(String, PositionalEntry)>,
    /// Mirrors rbs's `rest_keywords: Annotations::DoubleSplatParamTypeAnnotation | Symbol | true | nil`.
    pub rest_keywords: Option<DoubleSplatRestEntry>,
    /// Mirrors rbs's `block: Annotations::BlockParamTypeAnnotation | Symbol | true | nil`.
    pub block: Option<BlockEntry>,
}

/// A non-rest parameter slot in [`DocStyle`].
///
/// Mirrors rbs's `Array[Annotations::ParamTypeAnnotation | Symbol]` element type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum PositionalEntry {
    /// `@rbs foo: Integer` — the annotation object is preserved so
    /// location survives to Stage D emission.
    Annotated(ParamTypeAnnotation),
    /// Only the def's parameter name is known; no annotation. The
    /// resolver fills this with `Ty::UNTYPED`.
    ByName(String),
}

/// A rest-positional parameter slot in [`DocStyle`].
///
/// Mirrors rbs's `rest_positionals: Annotations::SplatParamTypeAnnotation | Symbol | true | nil`
/// by combining `Option<SplatRestEntry>` with three variants:
///
/// - `None` — the def has no rest parameter (rbs `nil`)
/// - `Some(Annotated(_))` — annotation matched (rbs `annotation`)
/// - `Some(ByName(_))` — named rest param, no matching annotation (rbs `Symbol`)
/// - `Some(Unnamed)` — anonymous rest param (`*`), no annotation (rbs `true`)
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum SplatRestEntry {
    Annotated(SplatParamTypeAnnotation),
    ByName(String),
    Unnamed,
}

/// A rest-keyword parameter slot in [`DocStyle`].
///
/// Mirrors rbs's `rest_keywords: Annotations::DoubleSplatParamTypeAnnotation | Symbol | true | nil`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum DoubleSplatRestEntry {
    Annotated(DoubleSplatParamTypeAnnotation),
    ByName(String),
    Unnamed,
}

/// A block parameter slot in [`DocStyle`].
///
/// Mirrors rbs's `block: Annotations::BlockParamTypeAnnotation | Symbol | true | nil`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum BlockEntry {
    Annotated(BlockParamTypeAnnotation),
    ByName(String),
    Unnamed,
}

/// Outcome of resolving a def's trailing annotation, distinguishing
/// "annotation consumed", "annotation present but unused" and
/// "annotation present but failed empty-body parse". The third arm exists
/// so callers can emit `AnnotationSyntaxError` for empty `#:` instead of
/// the generic `UnusedInlineAnnotation` (Steep parity: empty `#:` on def
/// trailing return surfaces as `RBS::InlineDiagnostic`).
pub(crate) enum TrailingResolution<'a> {
    None,
    Unused(TrailingAnnotation<'a>),
    ParseFailed {
        range: PrismByteRange,
        parser_error: String,
    },
}

impl MethodTypeAnnotation {
    /// Build the method-type annotation from a def's surrounding
    /// `CommentBlock`s. Mirrors rbs's `MethodTypeAnnotation.build` 1:1.
    ///
    /// `variables` are reserved — threaded into `CommentBlock::each_paragraph`
    /// but ignored downstream until the C-side parser accepts them.
    /// `source` is needed for slicing trailing annotation bodies.
    ///
    /// Returns a 3-tuple mirroring rbs:
    ///
    /// 1. the built `MethodTypeAnnotation`
    /// 2. unused leading annotations (shadowed by explicit method types,
    ///    duplicates, or name mismatches)
    /// 3. the trailing-annotation resolution outcome (consumed / unused /
    ///    empty-body parse failure).
    pub(crate) fn build<'a>(
        leading_block: Option<&'a CommentBlock>,
        trailing_block: Option<&'a CommentBlock>,
        variables: &[String],
        node: &DefNode<'_>,
        source: &'a [u8],
        names: &NameTable,
    ) -> (Self, Vec<LeadingAnnotation>, TrailingResolution<'a>) {
        let leading_annotations: Vec<LeadingAnnotation> = leading_block
            .map(|b| b.each_paragraph(variables, names))
            .unwrap_or_default();
        let trailing = trailing_block.and_then(|b| b.trailing_annotation(source));
        let (trailing_return_annotation, trailing_resolution) = match trailing {
            Some(TrailingAnnotation::NodeTypeAssertion { range, type_text }) => {
                match build_trailing_return_annotation(range, type_text, names) {
                    Some(annotation) => (Some(annotation), TrailingResolution::None),
                    None if type_text.trim_ascii().is_empty() => (
                        None,
                        TrailingResolution::ParseFailed {
                            range,
                            parser_error: RbsParser::parse_type(type_text.as_bytes())
                                .err()
                                .unwrap_or_default(),
                        },
                    ),
                    None => (
                        None,
                        TrailingResolution::Unused(TrailingAnnotation::NodeTypeAssertion {
                            range,
                            type_text,
                        }),
                    ),
                }
            }
            Some(other) => (None, TrailingResolution::Unused(other)),
            None => (None, TrailingResolution::None),
        };
        let (mta, unused) = Self::classify_leading_annotations_with_return_type(
            &leading_annotations,
            node,
            trailing_return_annotation,
        );
        (mta, unused, trailing_resolution)
    }

    /// Classify a slice of already-parsed leading annotations against
    /// the def's parameter layout. Split from [`build`] so test code
    /// (and any future caller that already has parsed annotations) can
    /// hit the classification logic without constructing a
    /// `CommentBlock`. Internal API; production callers must go through
    /// [`build`].
    #[cfg(test)]
    pub(super) fn classify_leading_annotations(
        annotations: &[LeadingAnnotation],
        node: &DefNode<'_>,
    ) -> (Self, Vec<LeadingAnnotation>) {
        Self::classify_leading_annotations_with_return_type(annotations, node, None)
    }

    fn classify_leading_annotations_with_return_type(
        annotations: &[LeadingAnnotation],
        node: &DefNode<'_>,
        initial_return_annotation: Option<ReturnTypeAnnotation>,
    ) -> (Self, Vec<LeadingAnnotation>) {
        let mut explicit: Option<Vec<ExplicitAnnotation>> = None;
        let mut return_annotation: Option<ReturnTypeAnnotation> = initial_return_annotation;
        // Buffer param-like annotations in declaration order so we can
        // later hand them to DocStyle::build in one pass.
        let mut param_kind_annotations: Vec<ParamTypeKindAnnotation> = Vec::new();
        let mut unused: Vec<LeadingAnnotation> = Vec::new();

        for a in annotations {
            match a {
                LeadingAnnotation::Skip(_) => {}
                LeadingAnnotation::ColonMethodType(annotation) => {
                    explicit
                        .get_or_insert_with(Vec::new)
                        .push(ExplicitAnnotation::Colon((**annotation).clone()));
                }
                LeadingAnnotation::MethodTypes(annotation) => {
                    explicit
                        .get_or_insert_with(Vec::new)
                        .push(ExplicitAnnotation::MethodTypes(annotation.clone()));
                }
                LeadingAnnotation::ReturnType(annotation) => {
                    if explicit.is_some() || return_annotation.is_some() {
                        // Shadowed by an earlier explicit method-type
                        // annotation, or a duplicate `return:` (first
                        // wins). rbs collects both into the same
                        // `unused_annotations` array.
                        unused.push(a.clone());
                    } else {
                        return_annotation = Some(annotation.clone());
                    }
                }
                LeadingAnnotation::InstanceVariable(_) => {
                    unused.push(a.clone());
                }
                LeadingAnnotation::BlockParamType(ann) => {
                    if explicit.is_some() {
                        unused.push(a.clone());
                    } else {
                        param_kind_annotations.push(ParamTypeKindAnnotation::Block(ann.clone()));
                    }
                }
                LeadingAnnotation::ParamType(ann) => {
                    if explicit.is_some() {
                        unused.push(a.clone());
                    } else {
                        param_kind_annotations.push(ParamTypeKindAnnotation::Param(ann.clone()));
                    }
                }
                LeadingAnnotation::SplatParamType(ann) => {
                    if explicit.is_some() {
                        unused.push(a.clone());
                    } else {
                        param_kind_annotations.push(ParamTypeKindAnnotation::Splat(ann.clone()));
                    }
                }
                LeadingAnnotation::DoubleSplatParamType(ann) => {
                    if explicit.is_some() {
                        unused.push(a.clone());
                    } else {
                        param_kind_annotations
                            .push(ParamTypeKindAnnotation::DoubleSplat(ann.clone()));
                    }
                }
                LeadingAnnotation::ModuleSelf(_) => {
                    unused.push(a.clone());
                }
            }
        }

        let type_annotations = match explicit {
            Some(arr) if !arr.is_empty() => TypeAnnotations::Array(arr),
            _ => {
                if return_annotation.is_some() || !param_kind_annotations.is_empty() {
                    let (doc, unused_from_doc) =
                        DocStyle::build(&param_kind_annotations, return_annotation, node);
                    // Convert ParamTypeKindAnnotation back to LeadingAnnotation so
                    // the caller always sees a uniform Vec<LeadingAnnotation>.
                    unused.extend(unused_from_doc.into_iter().map(|pka| match pka {
                        ParamTypeKindAnnotation::Param(ann) => LeadingAnnotation::ParamType(ann),
                        ParamTypeKindAnnotation::Splat(ann) => {
                            LeadingAnnotation::SplatParamType(ann)
                        }
                        ParamTypeKindAnnotation::DoubleSplat(ann) => {
                            LeadingAnnotation::DoubleSplatParamType(ann)
                        }
                        ParamTypeKindAnnotation::Block(ann) => {
                            LeadingAnnotation::BlockParamType(ann)
                        }
                    }));
                    TypeAnnotations::DocStyle(Box::new(doc))
                } else {
                    TypeAnnotations::None
                }
            }
        };

        (MethodTypeAnnotation { type_annotations }, unused)
    }

    /// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation#overloading?`
    /// (`lib/rbs/ast/ruby/members.rb:524-533`). True when at least one
    /// explicit `# @rbs (...) -> ...` annotation carries the `| ...`
    /// trailer (its `dot3_location` is `Some`).
    pub fn overloading(&self) -> bool {
        match &self.type_annotations {
            TypeAnnotations::Array(annotations) => annotations.iter().any(|a| match a {
                ExplicitAnnotation::MethodTypes(m) => m.dot3_location.is_some(),
                ExplicitAnnotation::Colon(_) => false,
            }),
            TypeAnnotations::DocStyle(_) | TypeAnnotations::None => false,
        }
    }

    /// Mirrors `RBS::AST::Ruby::Members::MethodTypeAnnotation#empty?`
    /// (`lib/rbs/ast/ruby/members.rb:484-486`). True iff the def has
    /// no explicit method-type or doc-style annotations at all — the
    /// signal that `define_method` treats as "inherit from parent if
    /// any" (`lib/rbs/definition_builder.rb:881`).
    pub fn is_empty(&self) -> bool {
        matches!(self.type_annotations, TypeAnnotations::None)
    }
}

fn build_trailing_return_annotation(
    range: PrismByteRange,
    type_text: &str,
    names: &NameTable,
) -> Option<ReturnTypeAnnotation> {
    let return_type = ast_builder::parse_trailing_type_text(type_text, names)?;
    Some(ReturnTypeAnnotation {
        location: range,
        prefix_location: (range.0, range.0.saturating_add(2)),
        return_location: (0, 0),
        colon_location: (range.0.saturating_add(1), range.0.saturating_add(2)),
        return_type,
        comment_location: Some(range),
    })
}

impl DocStyle {
    /// Assemble a `DocStyle` from a list of param-shape annotations plus
    /// an optional return-type annotation, matching annotations against
    /// the def's `Prism::DefNode` parameter layout. Mirrors rbs's
    /// `RBS::AST::Ruby::Members::MethodTypeAnnotation::DocStyle.build`.
    ///
    /// Unmatched annotations (duplicates, or name mismatches against
    /// the def's parameter list) are returned as `ParamTypeKindAnnotation`
    /// so callers can wrap them back into `LeadingAnnotation` and
    /// recover the source location.
    fn build(
        param_annotations: &[ParamTypeKindAnnotation],
        return_annotation: Option<ReturnTypeAnnotation>,
        node: &DefNode<'_>,
    ) -> (Self, Vec<ParamTypeKindAnnotation>) {
        let mut unused: Vec<ParamTypeKindAnnotation> = Vec::new();
        // rbs `param_annotations` hash: name -> ParamTypeAnnotation. The
        // map owns the annotation until a def parameter consumes it via
        // `positional_entry`; survivors at the end become unused.
        let mut param_map: FxHashMap<String, ParamTypeAnnotation> = FxHashMap::default();
        let mut splat: Option<SplatParamTypeAnnotation> = None;
        let mut double_splat: Option<DoubleSplatParamTypeAnnotation> = None;
        let mut block: Option<BlockParamTypeAnnotation> = None;

        for a in param_annotations {
            match a {
                ParamTypeKindAnnotation::Param(ann) => {
                    if param_map.contains_key(&ann.name) {
                        unused.push(ParamTypeKindAnnotation::Param(ann.clone()));
                    } else {
                        param_map.insert(ann.name.clone(), ann.clone());
                    }
                }
                ParamTypeKindAnnotation::Splat(ann) => {
                    if splat.is_some() {
                        unused.push(ParamTypeKindAnnotation::Splat(ann.clone()));
                    } else {
                        splat = Some(ann.clone());
                    }
                }
                ParamTypeKindAnnotation::DoubleSplat(ann) => {
                    if double_splat.is_some() {
                        unused.push(ParamTypeKindAnnotation::DoubleSplat(ann.clone()));
                    } else {
                        double_splat = Some(ann.clone());
                    }
                }
                ParamTypeKindAnnotation::Block(ann) => {
                    if block.is_some() {
                        unused.push(ParamTypeKindAnnotation::Block(ann.clone()));
                    } else {
                        block = Some(ann.clone());
                    }
                }
            }
        }

        let mut doc = DocStyle {
            return_type_annotation: return_annotation.map(Box::new),
            ..DocStyle::default()
        };

        if let Some(params) = node.parameters() {
            doc.required_positionals = params
                .requireds()
                .iter()
                .filter_map(|p| {
                    p.as_required_parameter_node()
                        .map(|rp| prism_name_to_string(rp.name().as_slice()))
                })
                .map(|n| positional_entry(&n, &mut param_map))
                .collect();

            doc.optional_positionals = params
                .optionals()
                .iter()
                .filter_map(|p| {
                    p.as_optional_parameter_node()
                        .map(|op| prism_name_to_string(op.name().as_slice()))
                })
                .map(|n| positional_entry(&n, &mut param_map))
                .collect();

            doc.rest_positionals = splat_rest_match(
                splat.take(),
                params
                    .rest()
                    .and_then(|r| r.as_rest_parameter_node())
                    .map(|rp| rp.name().map(|n| prism_name_to_string(n.as_slice()))),
                &mut unused,
            );

            doc.trailing_positionals = params
                .posts()
                .iter()
                .filter_map(|p| {
                    p.as_required_parameter_node()
                        .map(|rp| prism_name_to_string(rp.name().as_slice()))
                })
                .map(|n| positional_entry(&n, &mut param_map))
                .collect();

            doc.required_keywords = params
                .keywords()
                .iter()
                .filter_map(|k| {
                    k.as_required_keyword_parameter_node()
                        .map(|rk| prism_name_to_string(rk.name().as_slice()))
                })
                .map(|n| {
                    let entry = positional_entry(&n, &mut param_map);
                    (n, entry)
                })
                .collect();

            doc.optional_keywords = params
                .keywords()
                .iter()
                .filter_map(|k| {
                    k.as_optional_keyword_parameter_node()
                        .map(|ok| prism_name_to_string(ok.name().as_slice()))
                })
                .map(|n| {
                    let entry = positional_entry(&n, &mut param_map);
                    (n, entry)
                })
                .collect();

            doc.rest_keywords = double_splat_rest_match(
                double_splat.take(),
                params
                    .keyword_rest()
                    .and_then(|kr| kr.as_keyword_rest_parameter_node())
                    .map(|krp| krp.name().map(|n| prism_name_to_string(n.as_slice()))),
                &mut unused,
            );

            doc.block = block_match(
                block.take(),
                params
                    .block()
                    .map(|bp| bp.name().map(|n| prism_name_to_string(n.as_slice()))),
                &mut unused,
            );
        } else {
            doc.rest_positionals = splat_rest_match(splat.take(), None, &mut unused);
            doc.rest_keywords = double_splat_rest_match(double_splat.take(), None, &mut unused);
        }

        if let Some(ann) = block.take() {
            doc.block = Some(BlockEntry::Annotated(ann));
        }

        // Survivors in param_map couldn't be placed on any def
        // parameter — their original annotations become unused.
        // Sort by name so diagnostics are deterministic, matching the
        // previous behavior (rbs has insertion-ordered Hash; crema's
        // FxHashMap doesn't, so we sort).
        let mut leftover: Vec<(String, ParamTypeAnnotation)> = param_map.into_iter().collect();
        leftover.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        for (_, ann) in leftover {
            unused.push(ParamTypeKindAnnotation::Param(ann));
        }

        (doc, unused)
    }
}

fn positional_entry(
    name: &str,
    param_map: &mut FxHashMap<String, ParamTypeAnnotation>,
) -> PositionalEntry {
    match param_map.remove(name) {
        Some(ann) => PositionalEntry::Annotated(ann),
        None => PositionalEntry::ByName(name.to_string()),
    }
}

/// Match a `*` annotation against the def's rest-positional slot.
///
/// Mirrors rbs's 4-state `rest_positionals: annotation | Symbol | true | nil`
/// selection at `lib/rbs/ast/ruby/members.rb:106-112`.
fn splat_rest_match(
    annotation: Option<SplatParamTypeAnnotation>,
    def_rest: Option<Option<String>>,
    unused: &mut Vec<ParamTypeKindAnnotation>,
) -> Option<SplatRestEntry> {
    match (annotation, def_rest) {
        // Named rest with annotation: match on name (or unnamed annotation).
        (Some(ann), Some(Some(d))) => match &ann.name {
            None => Some(SplatRestEntry::Annotated(ann)),
            Some(n) if n == &d => Some(SplatRestEntry::Annotated(ann)),
            Some(_) => {
                unused.push(ParamTypeKindAnnotation::Splat(ann));
                Some(SplatRestEntry::ByName(d.clone()))
            }
        },
        // Named rest, no annotation.
        (None, Some(Some(d))) => Some(SplatRestEntry::ByName(d)),
        // Anonymous rest with annotation: rbs's `rest.name.nil?` branch
        // accepts any annotation name (including named ones), because the
        // def has no rest name to collide with.
        (Some(ann), Some(None)) => Some(SplatRestEntry::Annotated(ann)),
        // Anonymous rest, no annotation — rbs's `true` sentinel.
        (None, Some(None)) => Some(SplatRestEntry::Unnamed),
        // No rest slot — annotation, if any, is unused.
        (Some(ann), None) => {
            unused.push(ParamTypeKindAnnotation::Splat(ann));
            None
        }
        (None, None) => None,
    }
}

/// Match a `**` annotation against the def's rest-keyword slot.
///
/// Mirrors rbs's 4-state `rest_keywords: annotation | Symbol | true | nil`
/// selection at `lib/rbs/ast/ruby/members.rb:145-151`.
fn double_splat_rest_match(
    annotation: Option<DoubleSplatParamTypeAnnotation>,
    def_rest: Option<Option<String>>,
    unused: &mut Vec<ParamTypeKindAnnotation>,
) -> Option<DoubleSplatRestEntry> {
    match (annotation, def_rest) {
        (Some(ann), Some(Some(d))) => match &ann.name {
            None => Some(DoubleSplatRestEntry::Annotated(ann)),
            Some(n) if n == &d => Some(DoubleSplatRestEntry::Annotated(ann)),
            Some(_) => {
                unused.push(ParamTypeKindAnnotation::DoubleSplat(ann));
                Some(DoubleSplatRestEntry::ByName(d.clone()))
            }
        },
        (None, Some(Some(d))) => Some(DoubleSplatRestEntry::ByName(d)),
        (Some(ann), Some(None)) => Some(DoubleSplatRestEntry::Annotated(ann)),
        (None, Some(None)) => Some(DoubleSplatRestEntry::Unnamed),
        (Some(ann), None) => {
            unused.push(ParamTypeKindAnnotation::DoubleSplat(ann));
            None
        }
        (None, None) => None,
    }
}

/// Match a `&block` annotation against the def's block slot.
///
/// Mirrors rbs's block selection in `DocStyle.build`: an annotation can
/// attach to a matching block parameter, to an unnamed block parameter, or
/// to a method with no explicit block parameter.
fn block_match(
    annotation: Option<BlockParamTypeAnnotation>,
    def_block: Option<Option<String>>,
    unused: &mut Vec<ParamTypeKindAnnotation>,
) -> Option<BlockEntry> {
    match (annotation, def_block) {
        (Some(ann), Some(Some(d))) => match &ann.name {
            None => Some(BlockEntry::Annotated(ann)),
            Some(n) if n == &d => Some(BlockEntry::Annotated(ann)),
            Some(_) => {
                unused.push(ParamTypeKindAnnotation::Block(ann));
                Some(BlockEntry::ByName(d.clone()))
            }
        },
        (None, Some(Some(d))) => Some(BlockEntry::ByName(d)),
        (Some(ann), Some(None)) => Some(BlockEntry::Annotated(ann)),
        (None, Some(None)) => Some(BlockEntry::Unnamed),
        (Some(ann), None) => Some(BlockEntry::Annotated(ann)),
        (None, None) => None,
    }
}

fn prism_name_to_string(name: &[u8]) -> String {
    String::from_utf8_lossy(name).to_string()
}

/// A single member of a class or module body.
///
/// Mirrors the subclasses of `RBS::AST::Ruby::Members::Base` plus the
/// nested-declaration case (a class inside a module inside a class).
/// Nesting is represented by wrapping a [`Declaration`] in
/// [`Member::Declaration`] so that tree walks can recurse through
/// declarations without switching representation.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Member {
    /// A nested `class` / `module` / `CONST =` declaration.
    Declaration(Declaration),
    /// A `def foo(...)` method definition.
    Def(DefMember),
    /// An `attr_reader` declaration.
    AttrReader(AttrReaderMember),
    /// An `attr_writer` declaration.
    AttrWriter(AttrWriterMember),
    /// An `attr_accessor` declaration.
    AttrAccessor(AttrAccessorMember),
    /// An `include` call.
    Include(IncludeMember),
    /// An `extend` call.
    Extend(ExtendMember),
    /// A `prepend` call.
    Prepend(PrependMember),
    /// A synthetic prepend into the owner singleton class.
    SingletonPrepend(PrependMember),
    /// An instance variable annotation.
    InstanceVariable(InstanceVariableMember),
    /// A module-self annotation.
    ModuleSelf(ModuleSelfMember),
}

impl Member {
    pub fn location_start(&self) -> u32 {
        match self {
            Member::Declaration(_) => u32::MAX,
            Member::Def(m) => m.location.0,
            Member::AttrReader(m) => m.attribute.location.0,
            Member::AttrWriter(m) => m.attribute.location.0,
            Member::AttrAccessor(m) => m.attribute.location.0,
            Member::Include(m) => m.mixin.location.0,
            Member::Extend(m) => m.mixin.location.0,
            Member::Prepend(m) => m.mixin.location.0,
            Member::SingletonPrepend(m) => m.mixin.location.0,
            Member::InstanceVariable(m) => m.annotation.location.0,
            Member::ModuleSelf(m) => m.annotation.location.0,
        }
    }
}

/// A method definition collected from the Ruby AST.
///
/// Mirrors `RBS::AST::Ruby::Members::DefMember`.
///
/// Rust does not keep rbs's `node: Prism::DefNode` field because
/// Prism nodes borrow from the parse result. Instead, crema snapshots
/// the rbs-visible accessors derived from that node (`location` and
/// `name_location`) while keeping `method_type` as the already-built
/// [`MethodTypeAnnotation`].
#[derive(Debug, Clone, PartialEq)]
pub struct DefMember {
    pub name: String,
    pub kind: MethodKind,
    pub location: PrismByteRange,
    pub name_location: PrismByteRange,
    pub method_type: MethodTypeAnnotation,
    pub leading_comment: Option<CommentBlock>,
    pub origin: DefMemberOrigin,
    pub source_file: Option<Name>,
    /// crema extension, not part of the rbs `DefMember` contract:
    /// syntactic `@ivar = param` facts harvested from an instance-side
    /// `initialize` body (unconditional top-level statements whose RHS
    /// is a direct param reference), in source order. Empty for every
    /// other method. Consumed by the definition builder to synthesize
    /// instance-variable declarations from the initialize signature
    /// (Sorbet-style instance variable inference).
    pub ivar_param_pairs: Vec<InitializeIvarParamPair>,
}

/// One `@ivar = param` fact inside `def initialize`. `param` locates the
/// parameter positionally so the definition builder can index into the
/// resolved `MethodType` (whose positional params carry no names).
///
/// Deliberately location-free: the struct is hashed into the incremental
/// fingerprint (ADR-0032) so that adding/removing/reordering eligible
/// assignments invalidates dependents, and a byte range would make pure
/// code motion perturb the fingerprint spuriously.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InitializeIvarParamPair {
    /// Ivar name including the `@` sigil.
    pub ivar_name: String,
    pub param: InitializeParamRef,
}

/// Positional reference into an `initialize` method type's parameter
/// lists. Only the Sorbet-inferable kinds are representable —
/// rest / block / post params are deliberately out of scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InitializeParamRef {
    RequiredPositional(usize),
    OptionalPositional(usize),
    Keyword(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefMemberOrigin {
    Real,
    SyntheticConcernIncluded,
    SyntheticConcernPrepended,
    /// A `def` written directly in a `concerning :Topic do ... end` block
    /// (not nested under `included do` / `prepended do`): a member of the
    /// synthesized `<owner>::<Topic>` module, collected by the infusion
    /// pipeline because the inline parser's block descent stops at
    /// `concerning`. Unlike the two variants above it is NOT indexed by
    /// `scan_synthetic_concern_index` — its source node sits lexically
    /// inside the owner's class body, so the checker's ordinary walk
    /// already checks it in the owner's instance context (the module is
    /// really included/prepended there) and no context retargeting is
    /// needed.
    SyntheticConcerning,
}

/// Shared payload for `attr_reader` / `attr_writer` / `attr_accessor`.
///
/// Mirrors `RBS::AST::Ruby::Members::AttributeMember`.
///
/// `type_text` is the body of a trailing `#: T` comment (with no
/// surrounding whitespace); `None` when the attr has no annotation and
/// should fall back to untyped. `annotation_range` is populated only
/// when `type_text.is_some()` and attaches diagnostics on parse failure.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeMember {
    pub location: PrismByteRange,
    pub name_nodes: Vec<AttributeNameNode>,
    pub type_text: Option<String>,
    pub annotation_range: Option<PrismByteRange>,
    /// Interned path of the source file that produced this attribute, or
    /// `None` when the inline parser was invoked without a file identity
    /// (in-memory tests). Consumed by [`crate::definition::method::MemberRef`]
    /// to attach file identity to diagnostics emitted against Ruby-side
    /// attr / Struct-synthesised accessor members (e.g. the multi-source
    /// dup between a sig `attr_reader` and a rb `Struct.new` accessor).
    pub source_file: Option<Name>,
}

impl AttributeMember {
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.name_nodes.iter().map(|node| node.name.as_str())
    }

    pub fn name_locations(&self) -> impl Iterator<Item = PrismByteRange> + '_ {
        self.name_nodes.iter().map(|node| node.location)
    }
}

/// Owned view of the Prism symbol nodes stored in rbs's `name_nodes`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeNameNode {
    pub name: String,
    pub location: PrismByteRange,
}

/// Mirrors `RBS::AST::Ruby::Members::AttrReaderMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrReaderMember {
    pub attribute: AttributeMember,
}

/// Mirrors `RBS::AST::Ruby::Members::AttrWriterMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrWriterMember {
    pub attribute: AttributeMember,
}

/// Mirrors `RBS::AST::Ruby::Members::AttrAccessorMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrAccessorMember {
    pub attribute: AttributeMember,
}

/// Shared payload for `include` / `extend` / `prepend`.
///
/// Mirrors `RBS::AST::Ruby::Members::MixinMember`.
///
/// `module_name` is the module name as written in the Ruby source
/// (`"Helper"`, `"Foo::Bar"`, `"::Abs::M"`); resolving it against the
/// enclosing class's scope is the resolver's job.
#[derive(Debug, Clone, PartialEq)]
pub struct MixinMember {
    pub module_name: String,
    pub location: PrismByteRange,
    pub name_location: PrismByteRange,
    pub annotation: Option<TypeApplicationAnnotation>,
}

/// Mirrors `RBS::AST::Ruby::Members::IncludeMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct IncludeMember {
    pub mixin: MixinMember,
}

/// Mirrors `RBS::AST::Ruby::Members::ExtendMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendMember {
    pub mixin: MixinMember,
}

/// Mirrors `RBS::AST::Ruby::Members::PrependMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrependMember {
    pub mixin: MixinMember,
}

/// Mirrors `RBS::AST::Ruby::Members::InstanceVariableMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceVariableMember {
    pub annotation: InstanceVariableAnnotation,
}

/// Mirrors `RBS::AST::Ruby::Members::ModuleSelfMember`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleSelfMember {
    pub annotation: ModuleSelfAnnotation,
}
