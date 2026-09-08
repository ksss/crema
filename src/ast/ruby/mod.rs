//! AST for Ruby-side inline annotations.
//!
//! Mirrors `RBS::AST::Ruby::*` in the rbs repo. These nodes are
//! produced by `crate::ast_builder` from rbs_raw C structures after
//! the inline parser has split a comment block into annotation
//! paragraphs. Downstream `MethodTypeAnnotation::build` classifies and
//! assembles them without touching any environment.
//!
//! Phase 3/5 of ADR-0014.

pub mod annotations;
pub mod comment_block;
pub mod declarations;
pub mod members;


/// Byte range `[start, end)` captured from a Prism node's location.
///
/// Lives on the AST layer (not on `inline_parser`) because both
/// `declarations` and `members` carry these ranges for later diagnostic
/// attachment. Translated to `RubyLocation` by `inline_parser` once a file
/// identity is available at registration time.
pub type PrismByteRange = (u32, u32);

/// Precomputed newline-offset table for O(log n) byte-offset → line
/// lookups. Built once per source on long-lived visitors
/// (`TypeChecker`, `InlineCollector`) so per-call
/// line lookups in hot paths like `lookup_callsite_type_args` don't
/// re-scan the source on every Prism node.
pub struct LineIndex {
    /// `u32` to match Prism's location encoding.
    newlines: Vec<u32>,
}

impl LineIndex {
    pub fn from_source(source: &[u8]) -> Self {
        let mut newlines = Vec::new();
        for (i, &b) in source.iter().enumerate() {
            if b == b'\n' {
                newlines.push(i as u32);
            }
        }
        LineIndex { newlines }
    }

    /// 1-based line number containing `offset`. Saturates to the last
    /// line for any offset past EOF — matches the legacy
    /// [`byte_offset_to_line`] semantics that existing callers depend on.
    pub fn line(&self, offset: usize) -> usize {
        self.newlines.partition_point(|&n| (n as usize) < offset) + 1
    }
}

/// 1-based line number containing `offset`, by counting newlines up to
/// that point in the original Ruby source bytes. Shared between the
/// inline collector (Prism node → line) and the Ruby declaration
/// loader (annotation byte range → diagnostic line).
///
/// One-shot wrapper that builds a fresh [`LineIndex`] internally —
/// prefer reusing a shared [`LineIndex`] on callers that query the same
/// source repeatedly.
pub fn byte_offset_to_line(source: &[u8], offset: usize) -> usize {
    LineIndex::from_source(source).line(offset)
}

/// Byte column (0-based) of `offset` on its line — the byte distance from
/// the byte immediately after the previous `\n` (or from the start of the
/// source) to `offset`. Used by `CommentAssociation` to enforce rbs's
/// `CommentBlock.build` rule that two consecutive line-head comments must
/// share the same start column to be grouped into one block.
pub fn byte_offset_to_column(source: &[u8], offset: usize) -> usize {
    let mut i = offset.min(source.len());
    let mut column = 0usize;
    while i > 0 {
        let prev = i - 1;
        if source[prev] == b'\n' {
            break;
        }
        column += 1;
        i = prev;
    }
    column
}

/// True when the byte at `offset` is preceded only by whitespace on its
/// line (i.e. nothing but spaces/tabs back to the previous newline or
/// the start of the source). Used by `CommentAssociation` to gate
/// "leading" vs "trailing" classification — trailing comments on a code
/// line must not be harvested as leading candidates for the following
/// statement.
pub(crate) fn is_line_head_comment(source: &[u8], offset: usize) -> bool {
    let mut i = offset;
    while i > 0 {
        i -= 1;
        match source[i] {
            b'\n' => return true,
            b' ' | b'\t' => continue,
            _ => return false,
        }
    }
    true
}
