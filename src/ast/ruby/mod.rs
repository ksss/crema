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
    /// Parallel to `newlines`: number of chars in `source[..=newlines[i]]`,
    /// i.e. the char offset of the head of line `i + 2`. Lets
    /// [`char_offset`](Self::char_offset) count only within one line
    /// instead of from the start of the file on every diagnostic emit.
    line_head_chars: Vec<u32>,
    /// Length of the longest valid UTF-8 prefix of the source. Any prefix
    /// extending past it is not valid UTF-8, and its char offset falls back
    /// to the byte offset (see [`char_offset`](Self::char_offset)).
    valid_up_to: usize,
}

impl LineIndex {
    pub fn from_source(source: &[u8]) -> Self {
        let mut newlines = Vec::new();
        let mut line_head_chars = Vec::new();
        let mut chars = 0u32;
        for (i, &b) in source.iter().enumerate() {
            // Every char starts with a non-continuation byte, so counting
            // those equals `chars().count()` on valid UTF-8.
            if (b & 0xC0) != 0x80 {
                chars += 1;
            }
            if b == b'\n' {
                newlines.push(i as u32);
                line_head_chars.push(chars);
            }
        }
        let valid_up_to = match std::str::from_utf8(source) {
            Ok(_) => source.len(),
            Err(e) => e.valid_up_to(),
        };
        LineIndex {
            newlines,
            line_head_chars,
            valid_up_to,
        }
    }

    /// Char offset of byte `offset` in `source` — the number of chars in
    /// `source[..offset]`. `source` must be the slice this index was built
    /// from. O(line length): the chars before the current line come from
    /// the precomputed per-line table.
    ///
    /// Semantics mirror `from_utf8(&source[..offset]).map(chars().count())
    /// .unwrap_or(offset)`: once the prefix contains an invalid UTF-8 byte
    /// the result is the byte offset itself, so a file with one bad byte
    /// reports char == byte for everything after it. Offsets past EOF
    /// saturate to `source.len()`. Offsets inside a multi-byte char are
    /// outside the contract (Prism never produces them) but do not panic.
    pub fn char_offset(&self, source: &[u8], offset: usize) -> u32 {
        let offset = offset.min(source.len());
        if offset > self.valid_up_to {
            return offset as u32;
        }
        let line = self.newlines.partition_point(|&n| (n as usize) < offset);
        let (head_byte, head_chars) = if line == 0 {
            (0, 0)
        } else {
            (
                self.newlines[line - 1] as usize + 1,
                self.line_head_chars[line - 1],
            )
        };
        let within_line = source[head_byte..offset]
            .iter()
            .filter(|&&b| (b & 0xC0) != 0x80)
            .count() as u32;
        head_chars + within_line
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
