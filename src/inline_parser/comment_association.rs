use rustc_hash::FxHashMap;

use crate::ast::ruby::comment_block::{CommentBlock, CommentLine};
use crate::ast::ruby::{LineIndex, PrismByteRange, byte_offset_to_column, is_line_head_comment};

/// A trailing inline annotation, classified by its prefix.
///
/// Mirrors rbs's `AST::Ruby::Annotations::NodeTypeAssertion` and
/// `TypeApplicationAnnotation` at the boundary between comment lookup and
/// annotation parsing. The enclosed `&str` is a borrow into the original
/// source; downstream callers pass it to `RbsParser::parse_inline_trailing`
/// or consume it directly as a type text.
///
/// - `NodeTypeAssertion`: the whitespace-trimmed body of `#: T` (e.g. `"String"`
///   from `"#: String"`). Callers that already expect the historical trailing
///   type text can use this directly.
/// - `TypeApplication`: the bracketed body of `#[T1, T2]` including the
///   brackets (e.g. `"[String]"`). Pass straight to
///   `RbsParser::parse_inline_trailing` for bracket-aware parsing.
pub(crate) enum TrailingAnnotation<'a> {
    /// `#: T` — `range` spans the entire comment (including the `#:`
    /// prefix), so diagnostics can report the comment's location even
    /// after the type body has been stripped for parsing.
    NodeTypeAssertion {
        range: PrismByteRange,
        type_text: &'a str,
    },
    /// `#[T1, T2]` — same convention: `range` is the full comment, `body`
    /// is the bracketed payload (`"[String, Integer]"`).
    TypeApplication {
        range: PrismByteRange,
        body: &'a str,
    },
    /// `#$ T` (Steep-style block type argument) — `range` is the full comment,
    /// `body` is the trimmed type text after `$` (e.g. `"Integer"` from `"#$ Integer"`).
    /// Used as a type-arg hint on the selector line of a method call with a block
    /// (e.g. `inject(nil) {|acc, x| #$ Integer ...}`), mirroring Steep's
    /// `AST::Node::TypeApplication.parse` regex `/\A\$\s*(.+)/`.
    DollarTypeApplication {
        range: PrismByteRange,
        body: &'a str,
    },
    /// `#: class-alias` (implicit form) or `#: class-alias Foo` (explicit form).
    /// `type_name_text` is `None` for the implicit form; the trimmed name
    /// (e.g. `"Foo"` or `"Foo::Bar"`) for the explicit form.
    ClassAlias {
        range: PrismByteRange,
        type_name_text: Option<&'a str>,
    },
    /// `#: module-alias` (implicit form) or `#: module-alias Foo` (explicit form).
    ModuleAlias {
        range: PrismByteRange,
        type_name_text: Option<&'a str>,
    },
}

impl TrailingAnnotation<'_> {
    /// The byte range of the trailing annotation in the source file,
    /// spanning the entire comment (including the `#:` / `#[` prefix).
    pub(crate) fn range(&self) -> PrismByteRange {
        match self {
            TrailingAnnotation::NodeTypeAssertion { range, .. } => *range,
            TrailingAnnotation::TypeApplication { range, .. } => *range,
            TrailingAnnotation::DollarTypeApplication { range, .. } => *range,
            TrailingAnnotation::ClassAlias { range, .. } => *range,
            TrailingAnnotation::ModuleAlias { range, .. } => *range,
        }
    }
}

/// Maps source lines to inline annotation locations.
///
/// Ported from `RBS::InlineParser::CommentAssociation` — stores byte ranges
/// into the original source rather than copies of the comment text, so
/// queries slice on demand. Leading annotations are cached in two shapes:
/// a per-line map for enclosed-range scanners, and a per-block index for
/// upward leading walks (`leading_block_for`). Trailing annotations are
/// classified and sliced at query time.
pub(crate) struct CommentAssociation {
    /// line -> (byte_start, byte_end) of a trailing comment (`#:` or `#[`).
    /// Includes the `#` prefix; `trailing_annotation` slices it out.
    trailing_ranges: FxHashMap<usize, (u32, u32)>,
    /// Whether any trailing comment is a type-application form (`#[T]` or
    /// `#$ T`). Set once at build time so `lookup_callsite_type_args` can
    /// skip its per-call line scan on files that hold only `#:` assertions
    /// (the overwhelmingly common case), which it never consumes.
    has_trailing_type_applications: bool,
    /// line -> (byte range of the source comment, formatted inline-parser
    /// input) for `#:` or `# @rbs` leading annotations. The parser input
    /// strips the `#` prefix (e.g. `": (A) -> B"` or `"@rbs skip"`).
    /// The byte range is kept so `AnnotationSyntaxError` diagnostics can
    /// report the line/column of the original comment, not the formatted
    /// parser input.
    ///
    /// Used by enclosed-range scanners (`mark_enclosed_leading_lines`,
    /// `collect_enclosed_instance_variable_members`) that iterate every
    /// line inside a node and ask per-line whether an annotation lives
    /// there. Upward leading walks use `leading_blocks_by_end_line`
    /// instead so plain-comment paragraphs don't break the walk.
    leading: FxHashMap<usize, (PrismByteRange, String)>,
    /// CommentBlocks indexed by their last line. A block is the maximal
    /// run of consecutive line-head comments that share the same start
    /// column — mirrors rbs `RBS::AST::Ruby::CommentBlock.build` exactly.
    /// `leading_block_for(node_start_line)` looks up the block whose
    /// last comment line is `node_start_line - 1`, so callers obtain
    /// the entire upward sequence (plain paragraphs and annotation
    /// paragraphs alike) in one shot.
    leading_blocks_by_end_line: FxHashMap<usize, CommentBlock>,
}

impl CommentAssociation {
    pub(crate) fn from_source(
        source: &[u8],
        line_index: &LineIndex,
        parse_result: &ruby_prism::ParseResult<'_>,
    ) -> Self {
        let mut trailing_ranges = FxHashMap::default();
        let mut has_trailing_type_applications = false;
        let mut leading = FxHashMap::default();
        let mut leading_blocks_by_end_line = FxHashMap::default();

        // Block-builder state — accumulates consecutive same-column
        // line-head comments per rbs `CommentBlock.build`. A comment is
        // appended to the current block iff its line is `end + 1` and
        // its column matches `block_column`; otherwise the current block
        // is flushed and a new block starts at this comment.
        let mut block_lines: Vec<CommentLine> = Vec::new();
        let mut block_end_line: usize = 0;
        let mut block_column: usize = 0;

        for comment in parse_result.comments() {
            let text = comment.text();
            let text_str = match std::str::from_utf8(text) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let loc = comment.location();
            let line = line_index.line(loc.start_offset());

            // Record the raw byte range for any trailing-candidate prefix
            // (`#:` type assertion or `#[` type application). Dispatch between
            // the two variants happens in `trailing_annotation`.
            if text_str.starts_with("#:")
                || text_str.starts_with("#[")
                || text_str.starts_with("#$")
            {
                trailing_ranges.insert(line, (loc.start_offset() as u32, loc.end_offset() as u32));
                if text_str.starts_with("#[") || text_str.starts_with("#$") {
                    has_trailing_type_applications = true;
                }
            }

            let range = (loc.start_offset() as u32, loc.end_offset() as u32);
            // Only treat a comment as a *leading* annotation when it sits
            // at the head of its line (preceded only by whitespace). Trailing
            // annotations like `attr_reader :name #: String` would otherwise
            // also get picked up as leading candidates for the method on the
            // next line, producing spurious AnnotationSyntaxError reports
            // once the parse failure stops being silently swallowed.
            if !is_line_head_comment(source, loc.start_offset()) {
                // A non-line-head comment ends any in-flight block: rbs's
                // `CommentBlock.build` rejects such comments from the
                // grouping logic (`block_comments << comments.shift` is
                // gated on `start_line_slice.index(/\S/)` being nil).
                flush_block(
                    &mut leading_blocks_by_end_line,
                    &mut block_lines,
                    block_end_line,
                );
                continue;
            }
            // Compute the inline-parser input once and reuse it for both
            // the per-line `leading` map and the block-level `CommentLine`.
            // Empty string means the comment is not an inline annotation
            // (plain comment, or `#:` / `# @rbs` with no body) — it stays
            // in the block (so paragraph iteration still sees its location)
            // but is omitted from the per-line map (which only indexes
            // parseable annotations).
            let parser_input = parser_input_for_comment(text_str);
            if !parser_input.is_empty() {
                leading.insert(line, (range, parser_input.clone()));
            }

            // Block bookkeeping — build the rbs `CommentBlock` shape.
            let column = byte_offset_to_column(source, loc.start_offset());
            let can_append =
                !block_lines.is_empty() && block_end_line + 1 == line && block_column == column;
            if !can_append {
                flush_block(
                    &mut leading_blocks_by_end_line,
                    &mut block_lines,
                    block_end_line,
                );
                block_column = column;
            }
            block_lines.push(CommentLine {
                location: range,
                text: parser_input,
            });
            block_end_line = line;
        }
        flush_block(
            &mut leading_blocks_by_end_line,
            &mut block_lines,
            block_end_line,
        );

        CommentAssociation {
            trailing_ranges,
            has_trailing_type_applications,
            leading,
            leading_blocks_by_end_line,
        }
    }

    /// CommentBlock whose last comment lives on `node_start_line - 1`,
    /// i.e. the leading block immediately preceding a node that begins
    /// on `node_start_line`. Returns `None` when no contiguous line-head
    /// comment ends on that line (gap, column mismatch, or no comment
    /// at all).
    ///
    /// Mirrors rbs's `comments.leading_block(node)` query at
    /// `RBS::InlineParser::CommentAssociation`. The returned block
    /// contains every line in the rbs sense — plain paragraphs as well
    /// as `@rbs` / `#:` annotations — so callers iterate paragraphs via
    /// `CommentBlock::each_paragraph` and pattern-match on annotation
    /// kind rather than stopping at the first plain line.
    pub(crate) fn leading_block_for(&self, node_start_line: usize) -> Option<&CommentBlock> {
        if node_start_line <= 1 {
            return None;
        }
        self.leading_blocks_by_end_line.get(&(node_start_line - 1))
    }

    pub(crate) fn trailing_annotation<'a>(
        &self,
        source: &'a [u8],
        line: usize,
    ) -> Option<TrailingAnnotation<'a>> {
        let &(s, e) = self.trailing_ranges.get(&line)?;
        classify_trailing_range(source, (s, e))
    }

    /// Whether the source has any trailing `#:` / `#[` comment at all.
    /// Lets per-node visitors short-circuit before computing line
    /// numbers (a linear scan over the source bytes) on files that hold
    /// no inline annotations.
    pub(crate) fn has_trailing_annotations(&self) -> bool {
        !self.trailing_ranges.is_empty()
    }

    /// Whether the source has any type-application trailing comment (`#[T]`
    /// or `#$ T`). Lets `lookup_callsite_type_args` skip its per-call line
    /// scan on files that hold only `#:` assertions, which it never reads.
    pub(crate) fn has_trailing_type_applications(&self) -> bool {
        self.has_trailing_type_applications
    }

    pub(super) fn leading_annotation(&self, line: usize) -> Option<(PrismByteRange, &str)> {
        self.leading
            .get(&line)
            .map(|(range, s)| (*range, s.as_str()))
    }

    /// Raw byte range of the trailing comment recorded on `line`, if any.
    /// Used by callers that want to wrap the comment as a `CommentBlock`
    /// for downstream rbs-port processing (e.g. `MethodTypeAnnotation::build`).
    pub(super) fn trailing_range(&self, line: usize) -> Option<(u32, u32)> {
        self.trailing_ranges.get(&line).copied()
    }
}

/// Push the current accumulator into the per-end-line block map and
/// reset it. No-op when no lines are pending (`block_lines.is_empty()`),
/// so callers may call it unconditionally to flush boundaries.
fn flush_block(
    blocks: &mut FxHashMap<usize, CommentBlock>,
    block_lines: &mut Vec<CommentLine>,
    block_end_line: usize,
) {
    if block_lines.is_empty() {
        return;
    }
    let lines = std::mem::take(block_lines);
    if let Some(block) = CommentBlock::new(lines) {
        blocks.insert(block_end_line, block);
    }
}

/// Normalize a raw comment text into the inline-parser input format
/// (`": T"` for `#: T`, `"@rbs ..."` for `# @rbs ...`). Returns the
/// empty string for plain comments and for empty annotation prefixes —
/// `each_paragraph` parses on this text and an empty input fails the
/// parse cleanly so plain paragraphs are dropped without producing a
/// spurious annotation.
fn parser_input_for_comment(text_str: &str) -> String {
    if let Some(rest) = text_str.strip_prefix("#:") {
        let trimmed = rest.trim();
        if trimmed.is_empty() {
            String::new()
        } else {
            format!(": {}", trimmed)
        }
    } else if let Some(rest) = strip_rbs_prefix(text_str) {
        let trimmed = rest.trim();
        if trimmed.is_empty() {
            String::new()
        } else {
            format!("@rbs {}", trimmed)
        }
    } else {
        String::new()
    }
}

/// Classify a single trailing-annotation comment (`#: T` / `#[T]` /
/// `#: class-alias` / `#: module-alias`) from its byte range, returning
/// the same `TrailingAnnotation` enum that `CommentAssociation` produces.
///
/// Shared by `CommentAssociation::trailing_annotation` and the rbs-port
/// `CommentBlock::trailing_annotation` so the two query paths classify
/// trailing payloads identically.
pub(crate) fn classify_trailing_range(
    source: &[u8],
    (s, e): (u32, u32),
) -> Option<TrailingAnnotation<'_>> {
    let range: PrismByteRange = (s, e);
    let raw = source.get(s as usize..e as usize)?;
    let body = raw.strip_prefix(b"#")?;
    if let Some(rest) = body.strip_prefix(b":") {
        let trimmed = rest.trim_ascii();
        if trimmed.is_empty() {
            // Empty `#:` body — surface it as a `NodeTypeAssertion` carrying
            // an empty `type_text` so the existing parse paths (load-phase
            // `parse_rbs_type` for constant/attr, build-phase
            // `parse_trailing_type_text` for def) naturally fail and emit
            // `AnnotationSyntaxError`. The expression-assertion path in
            // `type_checker::visitor` adds an explicit empty-body silent
            // branch to match Steep's `ast/node/type_assertion.rb` parity.
            return Some(TrailingAnnotation::NodeTypeAssertion {
                range,
                type_text: "",
            });
        }
        let text = std::str::from_utf8(trimmed).ok()?;
        // Mirror rbs `kCLASSALIAS` / `kMODULEALIAS` keyword tokens:
        // `class-alias` / `module-alias` followed by either end-of-input
        // or whitespace + an explicit type name.
        if let Some(rest) = strip_alias_keyword(text, "class-alias") {
            return Some(TrailingAnnotation::ClassAlias {
                range,
                type_name_text: rest,
            });
        }
        if let Some(rest) = strip_alias_keyword(text, "module-alias") {
            return Some(TrailingAnnotation::ModuleAlias {
                range,
                type_name_text: rest,
            });
        }
        Some(TrailingAnnotation::NodeTypeAssertion {
            range,
            type_text: text,
        })
    } else if body.starts_with(b"[") {
        let text = std::str::from_utf8(body).ok()?;
        Some(TrailingAnnotation::TypeApplication { range, body: text })
    } else if let Some(rest) = body.strip_prefix(b"$") {
        let trimmed = rest.trim_ascii();
        if trimmed.is_empty() {
            return None;
        }
        let text = std::str::from_utf8(trimmed).ok()?;
        Some(TrailingAnnotation::DollarTypeApplication { range, body: text })
    } else {
        None
    }
}

/// Strip an alias keyword (`class-alias` / `module-alias`) from the start of
/// the trimmed `#:`-body. Returns:
/// - `Some(None)` for the implicit form (keyword alone, no trailing content)
/// - `Some(Some(name))` for the explicit form (keyword followed by a name)
/// - `None` when the body does not start with the keyword as a whole token
///
/// A trailing identifier character (e.g. `class-aliased`) defeats the match
/// to avoid swallowing unrelated annotations whose body happens to share a
/// prefix.
fn strip_alias_keyword<'a>(text: &'a str, keyword: &str) -> Option<Option<&'a str>> {
    let rest = text.strip_prefix(keyword)?;
    match rest.chars().next() {
        None => Some(None),
        Some(c) if c.is_ascii_whitespace() => {
            let trimmed = rest.trim_ascii();
            if trimmed.is_empty() {
                Some(None)
            } else if !is_valid_alias_target(trimmed) {
                // `: class-alias _Foo` (interface name) or `: class-alias element`
                // (lowercase / type variable) is rejected by rbs's annotation
                // parser. Reduce to NodeTypeAssertion fall-through so the bad
                // body fails downstream parsing rather than masquerading as a
                // valid alias target.
                None
            } else {
                Some(Some(trimmed))
            }
        }
        _ => None,
    }
}

/// A class/module alias target is a `::`-separated chain of constant names
/// (each segment starts with an ASCII uppercase letter and contains only
/// alphanumerics or underscores). Mirrors rbs's class-name lexing: interface
/// names (`_Foo`), type variables (`element`), and arbitrary identifiers are
/// rejected.
fn is_valid_alias_target(text: &str) -> bool {
    let body = text.strip_prefix("::").unwrap_or(text);
    if body.is_empty() {
        return false;
    }
    body.split("::").all(|segment| {
        let mut chars = segment.chars();
        match chars.next() {
            Some(c) if c.is_ascii_uppercase() => {
                chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        }
    })
}

/// Strip a `# @rbs` or `#@rbs` prefix (word-boundary aware) and return the
/// remaining content. Returns `None` if the comment is not an `@rbs`-style
/// doc annotation.
fn strip_rbs_prefix(text: &str) -> Option<&str> {
    let after_hash = text.strip_prefix('#')?;
    let after_space = after_hash.strip_prefix(' ').unwrap_or(after_hash);
    let after_at = after_space.strip_prefix("@rbs")?;
    // Require a word boundary: either end-of-input or a non-identifier char.
    match after_at.chars().next() {
        None => Some(""),
        Some(c) if !c.is_alphanumeric() && c != '_' => Some(after_at),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(source: &str) -> Option<TrailingAnnotation<'_>> {
        let bytes = source.as_bytes();
        let parse_result = ruby_prism::parse(bytes);
        let line_index = LineIndex::from_source(bytes);
        let assoc = CommentAssociation::from_source(bytes, &line_index, &parse_result);
        // Comments live on a single source line; harvest them by scanning
        // every line for the first non-empty annotation.
        let line_count = bytes.iter().filter(|&&b| b == b'\n').count() + 1;
        for line in 1..=line_count {
            if let Some(a) = assoc.trailing_annotation(bytes, line) {
                return Some(a);
            }
        }
        None
    }

    #[test]
    fn class_alias_implicit() {
        let src = "Foo = Bar #: class-alias\n";
        match classify(src) {
            Some(TrailingAnnotation::ClassAlias { type_name_text, .. }) => {
                assert_eq!(type_name_text, None);
            }
            other => panic!("expected ClassAlias, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn class_alias_explicit() {
        let src = "Foo = bar #: class-alias Object\n";
        match classify(src) {
            Some(TrailingAnnotation::ClassAlias { type_name_text, .. }) => {
                assert_eq!(type_name_text, Some("Object"));
            }
            other => panic!("expected ClassAlias, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn class_alias_explicit_qualified() {
        let src = "Foo = bar #: class-alias Outer::Inner\n";
        match classify(src) {
            Some(TrailingAnnotation::ClassAlias { type_name_text, .. }) => {
                assert_eq!(type_name_text, Some("Outer::Inner"));
            }
            other => panic!("expected ClassAlias, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn module_alias_implicit() {
        let src = "Foo = Bar #: module-alias\n";
        match classify(src) {
            Some(TrailingAnnotation::ModuleAlias { type_name_text, .. }) => {
                assert_eq!(type_name_text, None);
            }
            other => panic!("expected ModuleAlias, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn module_alias_explicit() {
        let src = "Foo = bar #: module-alias Kernel\n";
        match classify(src) {
            Some(TrailingAnnotation::ModuleAlias { type_name_text, .. }) => {
                assert_eq!(type_name_text, Some("Kernel"));
            }
            other => panic!("expected ModuleAlias, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn type_assertion_unaffected() {
        let src = "Foo = 1 #: Integer\n";
        match classify(src) {
            Some(TrailingAnnotation::NodeTypeAssertion { type_text, .. }) => {
                assert_eq!(type_text, "Integer");
            }
            other => panic!("expected NodeTypeAssertion, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn class_aliased_prefix_is_not_class_alias() {
        // The substring `class-alias` is followed by `d`, an identifier
        // character, so the keyword does not match — fall through to
        // NodeTypeAssertion (which then fails downstream parsing, but is
        // not silently treated as an alias).
        let src = "Foo = 1 #: class-aliased\n";
        match classify(src) {
            Some(TrailingAnnotation::NodeTypeAssertion { type_text, .. }) => {
                assert_eq!(type_text, "class-aliased");
            }
            other => panic!("expected NodeTypeAssertion, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn type_application_unaffected() {
        let src = "include M #[String]\n";
        match classify(src) {
            Some(TrailingAnnotation::TypeApplication { body, .. }) => {
                assert_eq!(body, "[String]");
            }
            other => panic!("expected TypeApplication, got {:?}", other.is_some()),
        }
    }

    fn type_app_flag(source: &str) -> bool {
        let bytes = source.as_bytes();
        let parse_result = ruby_prism::parse(bytes);
        let line_index = LineIndex::from_source(bytes);
        CommentAssociation::from_source(bytes, &line_index, &parse_result)
            .has_trailing_type_applications()
    }

    #[test]
    fn type_application_flag_distinguishes_kinds() {
        assert!(!type_app_flag("x = [] #: Array[Integer]\n"));
        assert!(!type_app_flag("foo(1)\n"));
        assert!(type_app_flag("foo(1) #[Integer]\n"));
        assert!(type_app_flag("inject(nil) {|a, b| #$ Integer\n  a }\n"));
    }
}
