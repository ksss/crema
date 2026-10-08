//! Content-addressed, type-tagged 64-bit IDs.
//!
//! Port of `ruby-rbs/src/ids.rs` (rbs PR #2964; see ADR-0025).
//!
//! Different ID domains (interned strings, type names, ...) share the same
//! underlying `NonZeroU64` representation but are distinguished at the
//! type level via a phantom tag, so a `SymbolId` cannot be silently used
//! where a `TypeName` is expected.
//!
//! The value of an ID is the 64-bit xxh3 hash of its content (with `0`
//! folded to `1` to keep the niche optimization on `Option<Id<T>>`).
//! Because IDs are derived from content, two independently-built
//! interners that see the same value assign the same ID — merging is
//! just a `HashMap` union, no remap walk needed.
//!
//! ```
//! use crema::ids::{Id, SymbolId};
//! enum OtherTag {}
//! type OtherId = Id<OtherTag>;
//!
//! fn takes_symbol(_: SymbolId) {}
//! // takes_symbol(OtherId::from_hash(0)); // compile error
//! ```

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU64;


/// A 64-bit content-addressed ID tagged with a domain marker `T`.
///
/// The tag is a zero-sized type parameter — typically an uninhabited enum —
/// used only to distinguish ID domains at the type level.
pub struct Id<T> {
    raw: NonZeroU64,
    _tag: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    /// Wrap a 64-bit hash as an `Id`. A hash of `0` is folded to `1`
    /// so the representation stays non-zero (enabling niche optimization
    /// in `Option<Id<T>>`). Collisions on the folded value are vanishingly
    /// rare in 2^64 space.
    #[must_use]
    pub fn from_hash(h: u64) -> Self {
        let raw = NonZeroU64::new(h).unwrap_or(NonZeroU64::new(1).unwrap());
        Self {
            raw,
            _tag: PhantomData,
        }
    }

    /// Wrap a pre-validated non-zero value.
    #[must_use]
    pub fn from_raw(raw: NonZeroU64) -> Self {
        Self {
            raw,
            _tag: PhantomData,
        }
    }

    /// The underlying non-zero 64-bit value.
    #[must_use]
    pub fn raw(self) -> NonZeroU64 {
        self.raw
    }

    /// The underlying 64-bit value as a plain integer.
    #[must_use]
    pub fn get(self) -> u64 {
        self.raw.get()
    }
}

// Manual trait impls — `#[derive(...)]` on a struct with `PhantomData<T>`
// would add unnecessary bounds on `T`, but `Id<T>` is always copyable,
// hashable, etc. regardless of the tag.

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("Id");
        write!(f, "{}({:#018x})", tag, self.raw.get())
    }
}

// Serialized as the bare 64-bit value. Persisting the raw id is sound
// because it is content-addressed (xxh3 of content, never a seeded
// process-local hash) — the same logical name resolves to the same id in
// any process, the property the G-snapshot decode path already relies on
// (`Symbol::from_raw_id`). Manual impls for the same reason the derives
// above are manual: no bounds on `T`.
// The ingest cache additionally needs to know which symbols / type
// names a payload references (their strings travel with the record);
// its codec scope is told about every id serialized while it is active
// and is a no-op otherwise.
impl<T: 'static> serde::Serialize for Id<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let tag = std::any::TypeId::of::<T>();
        if tag == std::any::TypeId::of::<SymbolTag>() {
            crate::ingest_cache::encoding_symbol(self.raw.get());
        } else if tag == std::any::TypeId::of::<TypeNameTag>() {
            crate::ingest_cache::encoding_type_name(self.raw.get());
        }
        serializer.serialize_u64(self.raw.get())
    }
}

impl<'de, T> serde::Deserialize<'de> for Id<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = u64::deserialize(deserializer)?;
        Ok(Id::from_hash(raw))
    }
}

/// Tag for IDs produced by the string interner.
pub enum SymbolTag {}

/// Identifier for an interned string. Content-addressed: `xxh3_64` of the
/// string bytes.
pub type SymbolId = Id<SymbolTag>;

/// Tag for IDs produced by the type-name interner.
pub enum TypeNameTag {}

/// Identifier for an interned type name. Content-addressed: derived from
/// the parent type name's hash and the last segment's hash.
pub type TypeName = Id<TypeNameTag>;
