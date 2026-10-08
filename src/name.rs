use std::fmt;
use std::num::NonZeroU32;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use xxhash_rust::xxh3::xxh3_64;

use crate::once_table::{OnceStore, OnceTable};

use crate::ids::SymbolId;
use crate::interner::StringInterner;
use crate::type_name::{Kind, TypeName, TypeNameInterner};

/// An interned name, a sequential (first-seen-order) id assigned by the
/// self-rolled two-layer interner in [`NameTable`] (ADR-0028 F3,
/// replacing `lasso::Rodeo` — `Name`'s id is NOT content-addressed like
/// [`Symbol`]'s or [`TypeName`]'s, see [`NameOverlay`]'s doc). Copy-cheap,
/// comparison-cheap.
///
/// Backed by a `NonZeroU32`, so `Name` has the same niche-optimised size
/// as `u32` and `Option<Name>` fits in 4 bytes — the same layout
/// `lasso::Spur` gave it before this port.
///
/// `Name` is the catch-all interned-string type for things that are not
/// language identifiers in the RBS sense — file paths, local variable
/// names, and other strings that have no rbs-side counterpart. For class
/// path segments, method names, type-parameter names, and other
/// language-identifier values, use [`Symbol`] instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Name(NonZeroU32);

// Serialized as the bare overlay id. A `Name` is positional, so the id
// only means something to the table that minted it; the ingest cache
// (the one persisted consumer) remaps it through its codec scope — see
// `ingest_cache::encoding_name` / `decoding_name`.
impl serde::Serialize for Name {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        crate::ingest_cache::encoding_name(self.0.get());
        serializer.serialize_u32(self.0.get())
    }
}

impl<'de> serde::Deserialize<'de> for Name {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = u32::deserialize(deserializer)?;
        let raw = crate::ingest_cache::decoding_name(raw).ok_or_else(|| {
            serde::de::Error::custom("Name other than the file's own in an ingest record")
        })?;
        NonZeroU32::new(raw)
            .map(Name)
            .ok_or_else(|| serde::de::Error::custom("Name id must be non-zero"))
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Name({})", self.0.get())
    }
}

impl Name {
    /// Wrap a raw [`NameOverlay`] id (always `>= 1`, see its doc) as a
    /// `Name`.
    fn from_overlay_id(id: u32) -> Name {
        Name(NonZeroU32::new(id).expect("NameOverlay ids start at 1"))
    }

    /// The raw overlay id, for the ingest cache's `Name` remap
    /// (`ingest_cache`). Meaningless outside the table that minted it.
    pub(crate) fn overlay_id(self) -> u32 {
        self.0.get()
    }
}

/// An interned Ruby-level identifier, port of the values that RBS holds
/// as Ruby `Symbol`s (class / module / interface name segments, method
/// names, type-parameter names, type-variable names, global names).
///
/// Backed by a content-addressed [`SymbolId`] from the ported
/// [`StringInterner`] (rbs #2964; ADR-0025), so the same string yields
/// the same `Symbol` in any interner — the property the TypeName
/// interner's `xxh3(parent_id, segment_id)` recipe builds on. The type
/// distinction from [`Name`] is enforced at the Rust level: a `Name`
/// cannot be passed where a `Symbol` is expected, and vice versa.
/// Mirrors the rbs design where `Namespace#path` is `Array[Symbol]` and
/// method-keyed tables such as `RBS::Definition#methods` use `Symbol`
/// as the `Hash` key (the method name itself is not held as a field on
/// `Definition::Method`).
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Symbol(SymbolId);

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({:#018x})", self.0.get())
    }
}

impl Symbol {
    /// Reconstruct a `Symbol` from its raw content-addressed id without
    /// looking it up in any interner (ADR-0028 F1: replaces the old
    /// `DecodeTables` id-to-`Symbol` lookup table). Content addressing
    /// (ADR-0025) makes this exact whenever `raw` really is a symbol id
    /// that was written by `sym_id_of`/`StringInterner::intern` — which
    /// the flat payload's own `flat::Reader::try_open` validation (plus,
    /// on the G path, `DecodeTables::intern_from`'s per-symbol drift
    /// check) establishes before decode reaches a call site that needs
    /// this. A `raw` that was never actually interned resolves to a
    /// panic on first display, the same "corrupt payload, delete
    /// `.crema`" contract the snapshot backends document for
    /// other post-open decode failures.
    pub(crate) fn from_raw_id(raw: u64) -> Symbol {
        Symbol(SymbolId::from_hash(raw))
    }
}

/// Self-rolled interner backing [`Name`] (ADR-0028 F3, replacing
/// `lasso::Rodeo`). Unlike [`Symbol`]/[`TypeName`], `Name`'s id is NOT
/// content-addressed — it is assigned by insertion order, starting at 1.
///
/// `entries` holds the interned strings in insertion order — physical
/// index `i` holds id `i + 1`. `index` maps the xxh3 of a string to its
/// id; a hit is confirmed against `entries`, so two strings sharing a
/// hash panic instead of sharing an id. Reads and hits take no lock;
/// minting a new id takes `mint`, looks the string up again under it,
/// then fills `entries` before publishing the id in `index`, so ids stay
/// dense and one string never gets two ids even across threads.
#[derive(Default)]
struct NameOverlay {
    entries: OnceStore<Box<str>>,
    index: OnceTable<u32>,
    /// Number of ids handed out. Written only under `mint`.
    len: AtomicU32,
    mint: Mutex<()>,
}

impl Clone for NameOverlay {
    /// Re-interns in id order, so every `Name` keeps its id.
    fn clone(&self) -> Self {
        let fresh = NameOverlay::new();
        for id in 1..=self.len() as u32 {
            fresh.intern(self.resolve(id));
        }
        fresh
    }
}

impl NameOverlay {
    fn new() -> Self {
        NameOverlay::default()
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    fn resolve(&self, id: u32) -> &str {
        self.entries
            .get((id - 1) as usize)
            .unwrap_or_else(|| panic!("Name not interned: {id}"))
    }

    fn find(&self, hash: u64, s: &str) -> Option<u32> {
        let id = *self.index.get(hash)?;
        assert!(self.resolve(id) == s, "xxh3 collision between Name strings");
        Some(id)
    }

    fn lookup(&self, s: &str) -> Option<u32> {
        self.find(xxh3_64(s.as_bytes()), s)
    }

    fn intern(&self, s: &str) -> u32 {
        let hash = xxh3_64(s.as_bytes());
        if let Some(id) = self.find(hash, s) {
            return id;
        }
        let _mint = self.mint.lock().unwrap();
        if let Some(id) = self.find(hash, s) {
            return id;
        }
        let len = self.len.load(Ordering::Relaxed);
        self.entries.set(len as usize, Box::from(s));
        let id = len + 1;
        self.len.store(id, Ordering::Release);
        self.index.insert(hash, id);
        id
    }
}

/// A string interner that maps strings to `Name` IDs.
/// Same string always maps to the same `Name`.
///
/// `Send + Sync`: every method takes `&self`, and threads sharing one
/// table (the frozen `Environment` during a parallel check, ADR-0034)
/// see each other's interned `Name`s / `Symbol`s / `TypeName`s. Reads and
/// hits take no lock; only adding a new entry is serialized. See
/// [`NameOverlay`] for the `Name` id space and [`crate::once_table`] for
/// the storage.
#[derive(Clone)]
pub struct NameTable {
    names: NameOverlay,
    /// Content-addressed interner backing [`Symbol`]. Separate from the
    /// overlay backing [`Name`]: `Symbol` follows the rbs Rust
    /// representation (ADR-0025) while `Name` has no rbs counterpart.
    symbols: StringInterner,
    /// Parent-chain interner backing [`TypeName`] (rbs #2964 port). The
    /// methods below delegate so call sites keep the `&NameTable`
    /// plumbing; the ported interner API itself stays in
    /// [`crate::type_name`].
    type_names: TypeNameInterner,
    builtins: BuiltinNames,
}

impl fmt::Debug for NameTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NameTable({} names, {} symbols)",
            self.names.len(),
            self.symbols.len()
        )
    }
}

impl NameTable {
    pub fn new() -> Self {
        let names = NameOverlay::new();
        let symbols = StringInterner::new();
        let type_names = TypeNameInterner::new();
        let builtins = BuiltinNames::pre_intern(&symbols, &type_names);
        NameTable {
            names,
            symbols,
            type_names,
            builtins,
        }
    }

    /// Fold the growth chains of the three tables into single segments.
    pub(crate) fn compact(&mut self) {
        self.symbols.compact();
        self.type_names.compact();
        self.names.index.compact();
    }

    fn resolve_name(&self, name: Name) -> &str {
        self.names.resolve(name.0.get())
    }

    /// Resolve `name`'s raw entry as `(parent, segment, absolute)`.
    /// `None` if `name` was never interned.
    fn tn_lookup(&self, name: TypeName) -> Option<(Option<TypeName>, Option<Symbol>, bool)> {
        self.type_names
            .try_entry(name)
            .map(|(p, s, a)| (p, s.map(Symbol), a))
    }

    /// Parse an RBS type-name string (`::Foo::Bar`, `Foo::bar`, ...) into
    /// an interned [`TypeName`]. Empty input yields the relative root,
    /// `"::"` the absolute root.
    ///
    pub fn parse_type_name(&self, s: &str) -> TypeName {
        let absolute = s.starts_with("::");
        let trimmed = s.strip_prefix("::").unwrap_or(s);
        let mut current = self.type_name_root(absolute);
        for part in trimmed.split("::") {
            if part.is_empty() {
                continue;
            }
            let seg = self.intern_symbol(part);
            current = self.append_type_name(current, seg);
        }
        current
    }

    /// The absolute root namespace `::` as a [`TypeName`].
    pub fn absolute_root(&self) -> TypeName {
        self.type_names.absolute_root()
    }

    /// The relative empty namespace `""` as a [`TypeName`].
    pub fn relative_root(&self) -> TypeName {
        self.type_names.relative_root()
    }

    /// [`absolute_root`](Self::absolute_root) when `absolute`, else
    /// [`relative_root`](Self::relative_root).
    pub fn type_name_root(&self, absolute: bool) -> TypeName {
        self.type_names.root(absolute)
    }

    /// Returns `parent::segment`. Resolves `parent`'s `absolute` flag
    /// through [`Self::type_name_is_absolute`] and inserts directly via
    /// [`TypeNameInterner::insert_child`], bypassing
    /// [`TypeNameInterner::append`]'s own parent lookup.
    pub fn append_type_name(&self, parent: TypeName, segment: Symbol) -> TypeName {
        let id = TypeName::from_hash(crate::type_name::child_hash(parent, segment.0));
        let absolute = self.type_name_is_absolute(parent);
        self.type_names
            .insert_child(id, parent, segment.0, absolute);
        id
    }

    /// Appends each segment in order to `base`.
    pub fn extend_type_name(
        &self,
        base: TypeName,
        segments: impl IntoIterator<Item = Symbol>,
    ) -> TypeName {
        segments
            .into_iter()
            .fold(base, |p, s| self.append_type_name(p, s))
    }

    /// The parent of `name`, or `None` for the two roots.
    pub fn type_name_parent(&self, name: TypeName) -> Option<TypeName> {
        self.tn_lookup(name)
            .unwrap_or_else(|| panic!("TypeName not interned: {name:?}"))
            .0
    }

    /// The last segment of `name`, or `None` for the two roots.
    pub fn last_segment(&self, name: TypeName) -> Option<Symbol> {
        self.tn_lookup(name)
            .unwrap_or_else(|| panic!("TypeName not interned: {name:?}"))
            .1
    }

    pub fn type_name_is_absolute(&self, name: TypeName) -> bool {
        self.tn_lookup(name)
            .unwrap_or_else(|| panic!("TypeName not interned: {name:?}"))
            .2
    }

    /// True for an empty path (the relative or absolute root).
    pub fn type_name_is_root(&self, name: TypeName) -> bool {
        self.type_name_parent(name).is_none()
    }

    /// The segments of `name` from root to leaf.
    pub fn type_name_segments(&self, name: TypeName) -> Vec<Symbol> {
        let mut buf = Vec::new();
        let mut cur = name;
        loop {
            let (parent, seg, _) = self
                .tn_lookup(cur)
                .unwrap_or_else(|| panic!("TypeName not interned: {cur:?}"));
            let Some(parent) = parent else { break };
            buf.push(seg.expect("non-root TypeName has a segment"));
            cur = parent;
        }
        buf.reverse();
        buf
    }

    /// The same path with `absolute = true`.
    pub fn to_absolute(&self, name: TypeName) -> TypeName {
        if self.type_name_is_absolute(name) {
            return name;
        }
        let segs = self.type_name_segments(name);
        self.extend_type_name(self.absolute_root(), segs)
    }

    /// The same path with `absolute = false`.
    pub fn to_relative(&self, name: TypeName) -> TypeName {
        if !self.type_name_is_absolute(name) {
            return name;
        }
        let segs = self.type_name_segments(name);
        self.extend_type_name(self.relative_root(), segs)
    }

    /// Ruby `TypeName#+` semantics (see [`TypeNameInterner::concat`]).
    pub fn concat_type_name(&self, head: TypeName, tail: TypeName) -> TypeName {
        if self.type_name_is_absolute(tail) {
            return tail;
        }
        let tail_segs = self.type_name_segments(tail);
        self.extend_type_name(head, tail_segs)
    }

    /// [`Kind`] of the trailing segment, derived from its first
    /// character. `None` for roots.
    pub fn type_name_kind(&self, name: TypeName) -> Option<Kind> {
        self.type_names.kind(name, &self.symbols)
    }

    pub fn is_class(&self, name: TypeName) -> bool {
        self.type_name_kind(name) == Some(Kind::Class)
    }

    pub fn is_alias(&self, name: TypeName) -> bool {
        self.type_name_kind(name) == Some(Kind::Alias)
    }

    pub fn is_interface(&self, name: TypeName) -> bool {
        self.type_name_kind(name) == Some(Kind::Interface)
    }

    /// Render `name` in the canonical RBS string form (`::Foo::Bar`,
    /// `Foo::bar`). Successor of the retired `TypeName::to_path_string`.
    pub fn display_type_name(&self, name: TypeName) -> String {
        self.type_names.display(name, &self.symbols)
    }

    /// Pre-interned [`TypeName`]s for the rbs builtin classes / modules.
    /// Port of `RBS::BuiltinNames` (`rbs/lib/rbs/builtin_names.rb`).
    pub(crate) fn builtins(&self) -> &BuiltinNames {
        &self.builtins
    }

    /// Absorb the content-addressed interners of `other` (a worker
    /// table from the parallel ingest, ADR-0033). `Symbol` / `TypeName`
    /// ids are xxh3-derived (ADR-0025), so entries already present here
    /// coincide with `other`'s and the union is a plain map merge; no
    /// id remapping of the data the worker produced is needed.
    ///
    /// `Name` ids are positional, so a worker must never have interned
    /// one: the caller pre-interns every `Name` the worker needs (file
    /// paths) and hands them over. A non-empty overlay in `other` means
    /// the worker minted ids main cannot resolve — refuse loudly rather
    /// than let dangling `Name`s reach a diagnostic.
    pub fn merge(&self, other: NameTable) {
        assert!(
            other.names.len() == 0,
            "merged NameTable must not carry positional `Name` entries"
        );
        self.symbols.merge(other.symbols);
        self.type_names.merge(other.type_names);
    }

    /// Intern a string, returning the same `Name` for equal strings.
    pub fn intern(&self, s: &str) -> Name {
        Name::from_overlay_id(self.names.intern(s))
    }

    /// Intern a string as a [`Symbol`] (RBS-level Ruby identifier).
    /// Goes through the content-addressed [`StringInterner`], so the same
    /// string yields the same `Symbol` in any `NameTable`.
    pub fn intern_symbol(&self, s: &str) -> Symbol {
        Symbol(self.symbols.intern(s))
    }

    /// The string a `Name` / `Symbol` was interned from, borrowed from the
    /// table. Entries never move while the table is shared (only
    /// `compact`, which takes `&mut self`, rebuilds the index), so the
    /// borrow lives as long as `&self`. A [`TypeName`] is not a single
    /// interned string; render it with [`display_type_name`](Self::display_type_name).
    pub fn resolve<N: ResolvableName>(&self, name: N) -> &str {
        name.resolve_with(self)
    }

    /// Look up a string without interning. Returns `None` if not yet
    /// interned.
    pub fn lookup(&self, s: &str) -> Option<Name> {
        self.names.lookup(s).map(Name::from_overlay_id)
    }

    /// Look up a string as a [`Symbol`] without interning. Returns `None`
    /// if not yet interned through [`intern_symbol`](Self::intern_symbol).
    /// Strings interned only as [`Name`] are not visible here — the two
    /// backings are separate.
    pub fn lookup_symbol(&self, s: &str) -> Option<Symbol> {
        // Mirrors StringInterner::intern's id recipe (xxh3 of the bytes);
        // the presence check keeps "never interned" answering None.
        let id = SymbolId::from_hash(xxhash_rust::xxh3::xxh3_64(s.as_bytes()));
        self.symbols.try_resolve(id).map(|_| Symbol(id))
    }

    /// Look up a name that is expected to be interned. Panics if not found.
    pub fn name(&self, s: &str) -> Name {
        self.lookup(s)
            .unwrap_or_else(|| panic!("Expected interned name: {}", s))
    }

    /// Look up a [`Symbol`] that is expected to be interned. Panics if not
    /// found. Companion to [`name`](Self::name) for the Symbol side.
    pub fn name_symbol(&self, s: &str) -> Symbol {
        self.lookup_symbol(s)
            .unwrap_or_else(|| panic!("Expected interned symbol: {}", s))
    }
}

impl Default for NameTable {
    fn default() -> Self {
        Self::new()
    }
}

/// An id that maps back to exactly one interned string in a [`NameTable`].
pub trait ResolvableName {
    fn resolve_with(self, names: &NameTable) -> &str;
}

impl ResolvableName for Name {
    fn resolve_with(self, names: &NameTable) -> &str {
        names.resolve_name(self)
    }
}

/// Shared by every `Symbol`-resolving `ResolvableName` impl below.
fn resolve_symbol_id(names: &NameTable, id: SymbolId) -> &str {
    names.symbols.resolve(id)
}

impl ResolvableName for Symbol {
    fn resolve_with(self, names: &NameTable) -> &str {
        resolve_symbol_id(names, self.0)
    }
}

impl ResolvableName for crate::type_param::TypeVarKey {
    fn resolve_with(self, names: &NameTable) -> &str {
        resolve_symbol_id(names, self.raw.0)
    }
}

impl ResolvableName for &crate::type_param::TypeVarKey {
    fn resolve_with(self, names: &NameTable) -> &str {
        resolve_symbol_id(names, self.raw.0)
    }
}

/// Port of `RBS::BuiltinNames` (`rbs/lib/rbs/builtin_names.rb`): pre-interned
/// absolute [`TypeName`]s for Ruby's builtin classes / modules. Each field is
/// a `kind = Class` `TypeName` rooted at `::` with a single name segment.
///
/// Built once at [`NameTable::new`] and exposed via [`NameTable::builtins`].
/// Compare with `==` rather than calling `parse_absolute` again at use sites.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BuiltinNames {
    pub basic_object: TypeName,
    pub object: TypeName,
    pub kernel: TypeName,
    pub class: TypeName,
    pub module: TypeName,
    pub string: TypeName,
    pub comparable: TypeName,
    pub enumerable: TypeName,
    pub array: TypeName,
    pub hash: TypeName,
    pub range: TypeName,
    pub enumerator: TypeName,
    pub set: TypeName,
    pub symbol: TypeName,
    pub integer: TypeName,
    pub float: TypeName,
    pub regexp: TypeName,
    pub true_class: TypeName,
    pub false_class: TypeName,
    pub numeric: TypeName,
    /// crema includes NilClass following `Steep::AST::Builtin`
    /// (`steep/lib/steep/ast/builtin.rb`); `rbs::BuiltinNames` does not.
    pub nil_class: TypeName,
    /// `::Proc` builtin class name.
    pub proc: TypeName,
}

impl BuiltinNames {
    fn pre_intern(strings: &StringInterner, names: &TypeNameInterner) -> Self {
        let root = names.absolute_root();
        let cn = |s: &str| {
            let sym = strings.intern(s);
            names.append(root, sym)
        };
        Self {
            basic_object: cn("BasicObject"),
            object: cn("Object"),
            kernel: cn("Kernel"),
            class: cn("Class"),
            module: cn("Module"),
            string: cn("String"),
            comparable: cn("Comparable"),
            enumerable: cn("Enumerable"),
            array: cn("Array"),
            hash: cn("Hash"),
            range: cn("Range"),
            enumerator: cn("Enumerator"),
            set: cn("Set"),
            symbol: cn("Symbol"),
            integer: cn("Integer"),
            float: cn("Float"),
            regexp: cn("Regexp"),
            true_class: cn("TrueClass"),
            false_class: cn("FalseClass"),
            numeric: cn("Numeric"),
            nil_class: cn("NilClass"),
            proc: cn("Proc"),
        }
    }

    /// True iff `name` is `::TrueClass` or `::FalseClass`. These
    /// two pair up to form the `bool` widening / display target;
    /// callers checking for the pair should go through this helper
    /// so a future split (e.g. tristate bool) has a single edit point.
    pub fn is_bool_class(&self, name: TypeName) -> bool {
        name == self.true_class || name == self.false_class
    }
}

