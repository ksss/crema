//! Content-addressed string interner producing [`SymbolId`]s.
//!
//! Port of `ruby-rbs/src/interner.rs` (rbs PR #2964; see ADR-0025).
//!
//! Each [`SymbolId`] is the `xxh3_64` hash of the interned bytes, so the
//! same string always produces the same `SymbolId` — regardless of which
//! [`StringInterner`] (and therefore which thread) it was interned in. To merge
//! per-thread interners into one, just take the union of their backing
//! maps; no `Remap` walk is needed.
//!
//! ```
//! use crema::interner::StringInterner;
//!
//! let mut a = StringInterner::new();
//! let mut b = StringInterner::new();
//! let a_string = a.intern("String");
//! let a_int = a.intern("Integer");
//! let b_string = b.intern("String");
//! let b_array = b.intern("Array");
//!
//! // Same content ⇒ same id across independent interners.
//! assert_eq!(a_string, b_string);
//!
//! let mut global = StringInterner::new();
//! global.merge(a);
//! global.merge(b);
//!
//! assert_eq!(global.resolve(a_int), "Integer");
//! assert_eq!(global.resolve(b_array), "Array");
//! ```

use crate::ids::SymbolId;
use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;


/// Interns strings and assigns each the content-addressed [`SymbolId`]
/// `xxh3_64(s.as_bytes())`.
///
/// One `StringInterner` per thread during parallel work, then [`merge`] them all
/// into a single destination `StringInterner` for the final shared view.
///
/// [`merge`]: Self::merge
#[derive(Default, Clone)]
pub struct StringInterner {
    map: FxHashMap<SymbolId, Box<str>>,
}

impl StringInterner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the content-addressed [`SymbolId`] for `s`, allocating
    /// storage only when `s` is new to this interner.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        let id = SymbolId::from_hash(xxh3_64(s.as_bytes()));
        self.map.entry(id).or_insert_with(|| Box::<str>::from(s));
        id
    }

    /// Returns the string previously interned for `id`.
    ///
    /// # Panics
    /// If `id` was not issued by this interner (or one merged into it).
    #[must_use]
    pub fn resolve(&self, id: SymbolId) -> &str {
        &self.map[&id]
    }

    /// Returns the string for `id`, or `None` if it was never interned here.
    #[must_use]
    pub fn try_resolve(&self, id: SymbolId) -> Option<&str> {
        self.map.get(&id).map(|s| &**s)
    }

    /// Returns the number of interned strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Move every entry from `other` into `self`. Because IDs are
    /// content-addressed, entries already present in `self` are kept; new
    /// ones are absorbed without reallocating their `Box<str>` storage.
    pub fn merge(&mut self, other: StringInterner) {
        for (id, boxed) in other.map {
            self.map.entry(id).or_insert(boxed);
        }
    }
}
