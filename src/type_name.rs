//! Content-addressed flyweight type names.
//!
//! Port of `ruby-rbs/src/type_name.rs` (rbs PR #2964; see ADR-0025).
//!
//! Unlike the Ruby implementation, which distinguishes `RBS::Namespace`
//! (a path of class names plus an `absolute` flag) from `RBS::TypeName`
//! (a Namespace plus a trailing name), this module folds both into a
//! single [`TypeName`]:
//!
//! - An empty path is a namespace root (either `::` or `""`).
//! - A non-empty path's last segment is what Ruby calls the trailing
//!   "name", and its [`Kind`] is derived from that segment's first
//!   character.
//!
//! Each [`TypeName`] is a 64-bit content-addressed id derived from its
//! parent's id and its last segment's [`SymbolId`]. Because the recipe is
//! deterministic, two independently-built [`TypeNameInterner`]s assign the
//! same id to the same logical name — merging is just a `HashMap` union.
//! Two pre-interned roots cover the absolute / relative split via fixed
//! sentinel hashes.
//!
//! ```
//! use crema::interner::StringInterner;
//! use crema::type_name::{Kind, TypeNameInterner};
//!
//! let mut strings = StringInterner::new();
//! let mut names = TypeNameInterner::new();
//!
//! let foo = names.parse(&mut strings, "::RBS::Foo");
//! let foo_again = names.parse(&mut strings, "::RBS::Foo");
//! assert_eq!(foo, foo_again);                         // flyweighted
//! assert_eq!(names.kind(foo, &strings), Some(Kind::Class));
//! assert_eq!(names.display(foo, &strings), "::RBS::Foo");
//! ```

use crate::ids::SymbolId;
pub use crate::ids::TypeName;
use crate::interner::StringInterner;
use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;


/// Fixed sentinel hash for the absolute (`::`) namespace root. Chosen so
/// no realistic `xxh3_64` content collision is expected; any value would
/// do as long as it differs from `RELATIVE_ROOT_HASH`.
const ABSOLUTE_ROOT_HASH: u64 = 0xA850_1075_0001_0001;

/// Fixed sentinel hash for the relative (`""`) namespace root.
const RELATIVE_ROOT_HASH: u64 = 0x8E1A_71FE_0001_0001;

/// Content-addressing recipe for a `parent::segment` child id. `pub(crate)`
/// so [`crate::name::NameTable`] (ADR-0028 F2) can compute a would-be
/// child's id up front, before deciding whether it already lives in an
/// attached A-snapshot baseline.
pub(crate) fn child_hash(parent: TypeName, segment: SymbolId) -> u64 {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&parent.get().to_le_bytes());
    buf[8..].copy_from_slice(&segment.get().to_le_bytes());
    xxh3_64(&buf)
}

/// Kind of a [`TypeName`], derived from the first character of its last
/// segment. `None` is returned for an empty type name (a namespace root).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Kind {
    Class,
    Alias,
    Interface,
}

/// An entry rooted in the absolute namespace (`::`).
#[derive(Copy, Clone, Debug)]
struct AbsoluteTypeNameEntry {
    /// `None` for the absolute root.
    parent: Option<TypeName>,
    /// `None` for the absolute root.
    segment: Option<SymbolId>,
}

/// An entry rooted in the relative namespace (`""`).
#[derive(Copy, Clone, Debug)]
struct RelativeTypeNameEntry {
    /// `None` for the relative root.
    parent: Option<TypeName>,
    /// `None` for the relative root.
    segment: Option<SymbolId>,
}

#[derive(Copy, Clone, Debug)]
enum Entry {
    Absolute(AbsoluteTypeNameEntry),
    Relative(RelativeTypeNameEntry),
}

impl Entry {
    fn parent(self) -> Option<TypeName> {
        match self {
            Self::Absolute(e) => e.parent,
            Self::Relative(e) => e.parent,
        }
    }

    fn segment(self) -> Option<SymbolId> {
        match self {
            Self::Absolute(e) => e.segment,
            Self::Relative(e) => e.segment,
        }
    }

    fn is_absolute(self) -> bool {
        matches!(self, Self::Absolute(_))
    }
}

/// Interner that flyweights [`TypeName`]s with content-addressed ids.
///
/// Build new names by walking down from a root:
///
/// ```
/// use crema::interner::StringInterner;
/// use crema::type_name::TypeNameInterner;
///
/// let mut strings = StringInterner::new();
/// let mut names = TypeNameInterner::new();
/// let rbs = strings.intern("RBS");
/// let foo = strings.intern("Foo");
///
/// let abs = names.absolute_root();
/// let n1 = names.append(abs, rbs);
/// let n2 = names.append(n1, foo);
/// assert_eq!(names.display(n2, &strings), "::RBS::Foo");
/// ```
#[derive(Clone)]
pub struct TypeNameInterner {
    entries: FxHashMap<TypeName, Entry>,
    relative_root: TypeName,
    absolute_root: TypeName,
}

impl Default for TypeNameInterner {
    fn default() -> Self {
        let relative_root = TypeName::from_hash(RELATIVE_ROOT_HASH);
        let absolute_root = TypeName::from_hash(ABSOLUTE_ROOT_HASH);
        let mut entries = FxHashMap::default();
        entries.insert(
            relative_root,
            Entry::Relative(RelativeTypeNameEntry {
                parent: None,
                segment: None,
            }),
        );
        entries.insert(
            absolute_root,
            Entry::Absolute(AbsoluteTypeNameEntry {
                parent: None,
                segment: None,
            }),
        );
        Self {
            entries,
            relative_root,
            absolute_root,
        }
    }
}

impl TypeNameInterner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of interned entries, including the two roots. Mirrors
    /// `StringInterner::len` — used by [`crate::name::NameTable`]'s
    /// baseline short-circuit tests (ADR-0028 F2) to check that resolving
    /// or appending a baseline-backed `TypeName` leaves the overlay
    /// untouched.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Never true: the two roots are always present (see [`Default`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The empty relative type name (`""` — `Namespace.empty` in Ruby).
    #[must_use]
    pub fn relative_root(&self) -> TypeName {
        self.relative_root
    }

    /// The empty absolute type name (`"::"` — `Namespace.root` in Ruby).
    #[must_use]
    pub fn absolute_root(&self) -> TypeName {
        self.absolute_root
    }

    /// Returns [`absolute_root`] when `absolute` is true, else
    /// [`relative_root`].
    ///
    /// [`absolute_root`]: Self::absolute_root
    /// [`relative_root`]: Self::relative_root
    #[must_use]
    pub fn root(&self, absolute: bool) -> TypeName {
        if absolute {
            self.absolute_root
        } else {
            self.relative_root
        }
    }

    /// Returns the type name `parent::segment`. Content-addressed:
    /// identical inputs return the same [`TypeName`] across any
    /// [`TypeNameInterner`].
    pub fn append(&mut self, parent: TypeName, segment: SymbolId) -> TypeName {
        let id = TypeName::from_hash(child_hash(parent, segment));
        if self.entries.contains_key(&id) {
            return id;
        }
        let parent_entry = self
            .entries
            .get(&parent)
            .copied()
            .expect("parent TypeName must be interned");
        let entry = if parent_entry.is_absolute() {
            Entry::Absolute(AbsoluteTypeNameEntry {
                parent: Some(parent),
                segment: Some(segment),
            })
        } else {
            Entry::Relative(RelativeTypeNameEntry {
                parent: Some(parent),
                segment: Some(segment),
            })
        };
        self.entries.insert(id, entry);
        id
    }

    /// Builds a type name by appending each `segment` in order to `base`.
    pub fn extend<I>(&mut self, base: TypeName, segments: I) -> TypeName
    where
        I: IntoIterator<Item = SymbolId>,
    {
        segments.into_iter().fold(base, |p, s| self.append(p, s))
    }

    /// Non-panicking sibling of [`Self::parent`]/[`Self::last_segment`]/
    /// [`Self::is_absolute`]: `None` if `name` has no entry here, instead
    /// of panicking. [`crate::name::NameTable`] (ADR-0028 F2) uses this to
    /// probe the overlay after an attached baseline misses, where
    /// "neither layer has it" is an expected outcome rather than a bug.
    #[must_use]
    pub(crate) fn try_entry(
        &self,
        name: TypeName,
    ) -> Option<(Option<TypeName>, Option<SymbolId>, bool)> {
        self.entries
            .get(&name)
            .map(|e| (e.parent(), e.segment(), e.is_absolute()))
    }

    /// Insert `parent::segment`'s entry directly under `id`, using an
    /// already-known `absolute` flag instead of [`Self::append`]'s own
    /// `parent` lookup (ADR-0028 F2: [`crate::name::NameTable::append_type_name`]
    /// resolves `absolute` itself through its baseline-aware
    /// `type_name_is_absolute`, since `parent` may live only in an
    /// attached A-snapshot baseline this interner — the session overlay —
    /// has no entry for). No-op if `id` is already present, same
    /// idempotence as `append`.
    pub(crate) fn insert_child(
        &mut self,
        id: TypeName,
        parent: TypeName,
        segment: SymbolId,
        absolute: bool,
    ) {
        if self.entries.contains_key(&id) {
            return;
        }
        let entry = if absolute {
            Entry::Absolute(AbsoluteTypeNameEntry {
                parent: Some(parent),
                segment: Some(segment),
            })
        } else {
            Entry::Relative(RelativeTypeNameEntry {
                parent: Some(parent),
                segment: Some(segment),
            })
        };
        self.entries.insert(id, entry);
    }

    /// Returns the parent of `name`, or `None` if `name` is one of the
    /// two roots.
    #[must_use]
    pub fn parent(&self, name: TypeName) -> Option<TypeName> {
        self.entries[&name].parent()
    }

    /// Returns the last segment of `name`, or `None` if `name` is one of
    /// the two roots.
    #[must_use]
    pub fn last_segment(&self, name: TypeName) -> Option<SymbolId> {
        self.entries[&name].segment()
    }

    #[must_use]
    pub fn is_absolute(&self, name: TypeName) -> bool {
        self.entries[&name].is_absolute()
    }

    /// True for an empty path (the relative or absolute root).
    #[must_use]
    pub fn is_root(&self, name: TypeName) -> bool {
        self.entries[&name].parent().is_none()
    }

    /// Number of segments in `name`.
    #[must_use]
    pub fn depth(&self, name: TypeName) -> usize {
        let mut depth = 0;
        let mut cur = name;
        while let Some(parent) = self.parent(cur) {
            depth += 1;
            cur = parent;
        }
        depth
    }

    /// Returns the segments of `name` from root to leaf.
    #[must_use]
    pub fn segments(&self, name: TypeName) -> Vec<SymbolId> {
        let mut buf = Vec::with_capacity(self.depth(name));
        let mut cur = name;
        loop {
            let entry = self.entries[&cur];
            let Some(parent) = entry.parent() else {
                break;
            };
            let seg = entry.segment().expect("non-root TypeName has a segment");
            buf.push(seg);
            cur = parent;
        }
        buf.reverse();
        buf
    }

    /// Returns the same type name with `absolute = true`, sharing the path.
    pub fn to_absolute(&mut self, name: TypeName) -> TypeName {
        if self.is_absolute(name) {
            return name;
        }
        let segs = self.segments(name);
        self.extend(self.absolute_root, segs)
    }

    /// Returns the same type name with `absolute = false`, sharing the path.
    pub fn to_relative(&mut self, name: TypeName) -> TypeName {
        if !self.is_absolute(name) {
            return name;
        }
        let segs = self.segments(name);
        self.extend(self.relative_root, segs)
    }

    /// Ruby `TypeName#+` semantics: if `tail` is absolute, return `tail`;
    /// otherwise concatenate `head`'s segments + `tail`'s segments under
    /// `head`'s absolute flag.
    pub fn concat(&mut self, head: TypeName, tail: TypeName) -> TypeName {
        if self.is_absolute(tail) {
            return tail;
        }
        let tail_segs = self.segments(tail);
        self.extend(head, tail_segs)
    }

    /// Kind of the trailing segment. `None` for roots.
    #[must_use]
    pub fn kind(&self, name: TypeName, strings: &StringInterner) -> Option<Kind> {
        let seg = self.last_segment(name)?;
        let bytes = strings.resolve(seg).as_bytes();
        let first = *bytes.first()?;
        Some(if first == b'_' {
            Kind::Interface
        } else if first.is_ascii_uppercase() {
            Kind::Class
        } else {
            Kind::Alias
        })
    }

    /// Render `name` in the canonical RBS string form
    /// (e.g. `::RBS::Foo`, `Foo::bar`).
    #[must_use]
    pub fn display(&self, name: TypeName, strings: &StringInterner) -> String {
        let segs = self.segments(name);
        let absolute = self.is_absolute(name);
        let mut s = String::new();
        if absolute {
            s.push_str("::");
        }
        for (i, seg) in segs.iter().enumerate() {
            if i > 0 {
                s.push_str("::");
            }
            s.push_str(strings.resolve(*seg));
        }
        s
    }

    /// Parse an RBS type-name string into a [`TypeName`], interning any
    /// new segments into `strings`.
    ///
    /// Empty `source` returns the relative root; `"::"` returns the
    /// absolute root.
    pub fn parse(&mut self, strings: &mut StringInterner, source: &str) -> TypeName {
        let absolute = source.starts_with("::");
        let trimmed = source.strip_prefix("::").unwrap_or(source);
        let mut current = self.root(absolute);
        for part in trimmed.split("::") {
            if part.is_empty() {
                continue;
            }
            let seg = strings.intern(part);
            current = self.append(current, seg);
        }
        current
    }

    /// Move every entry from `other` into `self`. Because IDs are
    /// content-addressed, the two interners' roots and any shared paths
    /// already have the same ids; this is a plain `HashMap` union.
    pub fn merge(&mut self, other: TypeNameInterner) {
        for (id, entry) in other.entries {
            self.entries.entry(id).or_insert(entry);
        }
    }

    /// Refine `name` to an [`AbsoluteTypeName`] if it is absolute.
    #[must_use]
    pub fn try_as_absolute(&self, name: TypeName) -> Option<AbsoluteTypeName> {
        self.is_absolute(name).then_some(AbsoluteTypeName(name))
    }

    /// Refine `name` to an [`AbsoluteClassTypeName`] if it is absolute and
    /// its last segment denotes a class.
    #[must_use]
    pub fn try_as_absolute_class(
        &self,
        name: TypeName,
        strings: &StringInterner,
    ) -> Option<AbsoluteClassTypeName> {
        (self.is_absolute(name) && self.kind(name, strings) == Some(Kind::Class))
            .then_some(AbsoluteClassTypeName(name))
    }

    /// Refine `name` to an [`AbsoluteAliasTypeName`] if it is absolute and
    /// its last segment denotes an alias.
    #[must_use]
    pub fn try_as_absolute_alias(
        &self,
        name: TypeName,
        strings: &StringInterner,
    ) -> Option<AbsoluteAliasTypeName> {
        (self.is_absolute(name) && self.kind(name, strings) == Some(Kind::Alias))
            .then_some(AbsoluteAliasTypeName(name))
    }

    /// Refine `name` to an [`AbsoluteInterfaceTypeName`] if it is absolute
    /// and its last segment denotes an interface.
    #[must_use]
    pub fn try_as_absolute_interface(
        &self,
        name: TypeName,
        strings: &StringInterner,
    ) -> Option<AbsoluteInterfaceTypeName> {
        (self.is_absolute(name) && self.kind(name, strings) == Some(Kind::Interface))
            .then_some(AbsoluteInterfaceTypeName(name))
    }
}

/// A [`TypeName`] guaranteed to be absolute.
///
/// Construct via [`TypeNameInterner::try_as_absolute`]; widen back to a
/// plain [`TypeName`] via [`From`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AbsoluteTypeName(TypeName);

impl AbsoluteTypeName {
    #[must_use]
    pub fn as_type_name(self) -> TypeName {
        self.0
    }
}

impl From<AbsoluteTypeName> for TypeName {
    fn from(value: AbsoluteTypeName) -> Self {
        value.0
    }
}

/// A [`TypeName`] guaranteed to be absolute and of [`Kind::Class`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AbsoluteClassTypeName(TypeName);

impl AbsoluteClassTypeName {
    #[must_use]
    pub fn as_type_name(self) -> TypeName {
        self.0
    }
}

impl From<AbsoluteClassTypeName> for TypeName {
    fn from(value: AbsoluteClassTypeName) -> Self {
        value.0
    }
}

impl From<AbsoluteClassTypeName> for AbsoluteTypeName {
    fn from(value: AbsoluteClassTypeName) -> Self {
        AbsoluteTypeName(value.0)
    }
}

/// A [`TypeName`] guaranteed to be absolute and of [`Kind::Alias`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AbsoluteAliasTypeName(TypeName);

impl AbsoluteAliasTypeName {
    #[must_use]
    pub fn as_type_name(self) -> TypeName {
        self.0
    }
}

impl From<AbsoluteAliasTypeName> for TypeName {
    fn from(value: AbsoluteAliasTypeName) -> Self {
        value.0
    }
}

impl From<AbsoluteAliasTypeName> for AbsoluteTypeName {
    fn from(value: AbsoluteAliasTypeName) -> Self {
        AbsoluteTypeName(value.0)
    }
}

/// A [`TypeName`] guaranteed to be absolute and of [`Kind::Interface`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AbsoluteInterfaceTypeName(TypeName);

impl AbsoluteInterfaceTypeName {
    #[must_use]
    pub fn as_type_name(self) -> TypeName {
        self.0
    }
}

impl From<AbsoluteInterfaceTypeName> for TypeName {
    fn from(value: AbsoluteInterfaceTypeName) -> Self {
        value.0
    }
}

impl From<AbsoluteInterfaceTypeName> for AbsoluteTypeName {
    fn from(value: AbsoluteInterfaceTypeName) -> Self {
        AbsoluteTypeName(value.0)
    }
}
