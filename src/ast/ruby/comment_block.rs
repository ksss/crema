use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::annotations::LeadingAnnotation;
use crate::ast_builder;
use crate::inline_parser::{TrailingAnnotation, classify_trailing_range};
use crate::name::NameTable;
use crate::rbs_raw::Parser as RbsParser;

/// A source comment block attached to a Ruby declaration or member.
///
/// Mirrors `RBS::AST::Ruby::CommentBlock` for the payload crema keeps
/// after Prism parsing: the original comment byte range plus the text
/// normalized for inline annotation parsing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommentBlock {
    pub comments: Vec<CommentLine>,
}

impl CommentBlock {
    pub fn new(comments: Vec<CommentLine>) -> Option<Self> {
        if comments.is_empty() {
            None
        } else {
            Some(CommentBlock { comments })
        }
    }

    /// Iterate this block's comments and yield each parsed
    /// `LeadingAnnotation`. Mirrors rbs's `each_paragraph(variables)`
    /// for the subset crema needs: each comment line is run through
    /// `RbsParser::parse_inline_leading` and successfully parsed
    /// annotations are yielded in declaration order.
    ///
    /// Parse failures and non-annotation paragraphs are silently
    /// skipped. rbs's `AnnotationSyntaxError` path is not yet ported
    /// — see todo `mid_method_type_annotation_build_signature.md` D3.
    ///
    /// `_variables` corresponds to rbs's `variables: Array[Symbol]`
    /// (type-variable names visible while parsing). rbs's own inline
    /// parse path passes `[]` everywhere (`inline_parser.rb:225`), so
    /// crema also passes `[]` to stay aligned. The argument is reserved
    /// for future parity if rbs ever wires type-variable declarations
    /// into its inline parse path.
    pub(crate) fn each_paragraph(
        &self,
        _variables: &[String],
        names: &NameTable,
    ) -> Vec<LeadingAnnotation> {
        self.comments
            .iter()
            .filter_map(|line| {
                let text = &line.text;
                let (parser, node) = RbsParser::parse_inline_leading(text.as_bytes()).ok()?;
                let mut annotation =
                    ast_builder::build_leading_annotation(&parser, node, text.as_bytes(), names)?;
                offset_leading_annotation_location(&mut annotation, line.location.0);
                if let LeadingAnnotation::InstanceVariable(ivar) = &mut annotation {
                    ivar.name_location = offset_range(ivar.name_location, line.location.0);
                }
                if let LeadingAnnotation::BlockParamType(block) = &mut annotation {
                    block.ampersand_location =
                        offset_range(block.ampersand_location, line.location.0);
                    block.name_location = block
                        .name_location
                        .map(|range| offset_range(range, line.location.0));
                    block.colon_location = offset_range(block.colon_location, line.location.0);
                    block.question_location = block
                        .question_location
                        .map(|range| offset_range(range, line.location.0));
                    block.type_location = offset_range(block.type_location, line.location.0);
                    block.comment_location = block
                        .comment_location
                        .map(|range| offset_range(range, line.location.0));
                }
                Some(annotation)
            })
            .collect()
    }

    /// Classify the block's first comment line as a trailing inline
    /// annotation. Mirrors rbs's `CommentBlock#trailing_annotation`
    /// for the def-trailing path: the def's `# … #: T` line is wrapped
    /// in a single-line `CommentBlock` and decoded here.
    ///
    /// Returns `None` when the block is empty or the first line is not
    /// a recognized trailing prefix (`#:` / `#[`).
    pub(crate) fn trailing_annotation<'a>(
        &self,
        source: &'a [u8],
    ) -> Option<TrailingAnnotation<'a>> {
        let line = self.comments.first()?;
        classify_trailing_range(source, line.location)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommentLine {
    pub location: PrismByteRange,
    pub text: String,
}

fn offset_range(range: PrismByteRange, offset: u32) -> PrismByteRange {
    (range.0 + offset, range.1 + offset)
}

fn offset_leading_annotation_location(a: &mut LeadingAnnotation, offset: u32) {
    match a {
        LeadingAnnotation::ColonMethodType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::MethodTypes(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::ParamType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::ReturnType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::InstanceVariable(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::BlockParamType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::SplatParamType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::DoubleSplatParamType(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::Skip(a) => a.location = offset_range(a.location, offset),
        LeadingAnnotation::ModuleSelf(a) => {
            a.location = offset_range(a.location, offset);
            a.name_location = offset_range(a.name_location, offset);
        }
    }
}
