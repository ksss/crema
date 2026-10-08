//! A file's bytes, either already in memory or readable on demand.
//!
//! The ingest phase reads and parses every file it has no cached
//! collect output for; a file served from the per-file ingest cache
//! (ADR-0036 Decision 4-1) is never read. The insert phase still wants
//! that file's bytes in a few places — the line/column of an insert-time
//! diagnostic (`DuplicatedDeclaration`, a broken `#: T` on an attribute
//! or constant) — so those sites take a `SourceRef` and read the file
//! only when such a diagnostic is actually built.

use std::path::Path;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy)]
pub enum SourceRef<'a> {
    /// Bytes the caller already holds (a file parsed this run, or an
    /// in-memory test source).
    Bytes(&'a [u8]),
    /// Bytes that will be read from `path` on first use and parked in
    /// `slot` — the same per-file slot the ingest loop fills for a file
    /// it parses, so a later reader of the slot finds them too.
    Lazy {
        slot: &'a OnceLock<Arc<[u8]>>,
        path: &'a Path,
    },
}

impl<'a> SourceRef<'a> {
    /// The bytes, reading the file if this is the first use of a lazy
    /// source. A file that cannot be read yields an empty slice: the
    /// only consumers map byte offsets to line/column, which then
    /// degrade to the clamped offset instead of failing the run.
    pub fn bytes(self) -> &'a [u8] {
        match self {
            SourceRef::Bytes(b) => b,
            SourceRef::Lazy { slot, path } => {
                slot.get_or_init(|| Arc::from(std::fs::read(path).unwrap_or_default()))
            }
        }
    }
}

impl<'a> From<&'a [u8]> for SourceRef<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        SourceRef::Bytes(bytes)
    }
}

impl Default for SourceRef<'_> {
    fn default() -> Self {
        SourceRef::Bytes(&[])
    }
}
