//! Lazy G-snapshot backend — point probe + per-entry lazy decode over
//! the flat payload (ADR-0028 slice 2a-2, Decision 3). The payload is
//! owned as a heap `Vec` today; swapping it for an mmap is a possible
//! later slice, the probe layout is already mmap-compatible.
//!
//! Replaces the eager warm path (decode-all → re-draft → full rebuild)
//! with a value that the frozen [`Environment`] holds alongside the
//! A-layer maps:
//!
//! - **eager, O(names + symbols + type-names)**: interning the global
//!   tables into the runtime `NameTable` ([`DecodeTables::intern_from`],
//!   with content-addressed id verification), the entry-index kind scan
//!   (4–8 bytes per entry, no payload decode), and decoding every
//!   class/module-alias entry (the resolver and the normalize table
//!   need every alias `old_name` up front; alias count is independent
//!   of entry count)
//! - **lazy, per reference**: decl entry payloads. A point lookup
//!   decodes only the referenced entry and memoizes the frozen value in
//!   that key's slot. The key set is fixed at open, so every slot exists
//!   up front: a probe fills one, never adds one. A filled slot is read
//!   without a lock and the handed-out `&V` lives as long as the
//!   backend, so the backend is `Send + Sync` (ADR-0034 Decision 1).
//!   Two threads racing on an empty slot both decode; the first store
//!   wins and the other value is dropped (Decision 3, no waiting)
//! - **full scans**: the first `*_all` call materializes every kind's
//!   map in one pass and memoizes them together. Unlike a point probe,
//!   concurrent callers wait for that one pass instead of decoding in
//!   parallel. The remaining full-scan callers (DefinitionBuilder eager
//!   scans, validators) pay one decode-all per process until slice 2b
//!   differentializes them

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;

use crate::definition::{VariableDuplication, VariableDuplicationKind};
use crate::definition_builder::{
    AliasCycleEntry, AliasCycles, BakedAncestorCycle, BakedArityViolation,
    BakedUnresolvedSuperEdge, MethodDups, RoastedGScan,
};
use crate::environment::DeclOrigin;
use crate::environment::draft::{Context, GlobalEntry, SingleEntry};
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassAliasEntry, ClassEntry, ClassOrModule, ClassOrModuleAliasEntry,
    InterfaceEntry, ModuleAliasDeclaration, ModuleAliasEntry, ModuleDeclaration, ModuleEntry,
};
use crate::location::{DuplicateSource, LocationRange, RubyLocation, SourceLocation};
use crate::name::{NameTable, Symbol};
use crate::snapshot::flat;
use crate::snapshot::mirror::{
    MBakedDiagnostics, MBakedLoc, MBakedMethodGroup, MClassOrModule, MEntry,
};
use crate::snapshot::rebuild::{
    DecodeTables, Decoder, EntryMemo, EntrySource, NestedDecls, RebuildError, corrupt,
};
use crate::snapshot::write::{DIAG_LEN_OFFSET, DIAG_OFF_OFFSET, HEADER_LEN};
use crate::type_name::TypeName;

use crate::ast::declarations::{
    ConstantDeclaration as Constant, TypeAliasDeclaration as TypeAlias,
};
use crate::environment::frozen::ClassDeclaration;

/// Kind of a class-namespace snapshot entry, recovered from the
/// bincode variant tags without decoding the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GClassKind {
    Class,
    Module,
}

// bincode 1.x (legacy `serialize`/`deserialize` config: little-endian,
// fixint) encodes an enum's variant index as a `u32` prefix. The outer
// tag mirrors `MEntry`'s declaration order; `MEntry::ClassOrModule`
// nests `MClassOrModule`, whose own tag follows at bytes 4..8. Pinned
// by `tag_scan_matches_full_decode` in the module tests so a mirror or
// bincode change fails loudly instead of mis-partitioning kinds.
const TAG_CLASS_OR_MODULE: u32 = 0;
const TAG_INTERFACE: u32 = 1;
const TAG_CLASS_ALIAS: u32 = 2;
const TAG_TYPE_ALIAS: u32 = 3;
const TAG_CONSTANT: u32 = 4;
const TAG_GLOBAL: u32 = 5;
const TAG_INNER_CLASS: u32 = 0;
const TAG_INNER_MODULE: u32 = 1;

/// The variant tag a flat entry payload opens with (see the constants
/// above).
pub(crate) fn entry_tag(bytes: &[u8], at: usize) -> Result<u32, RebuildError> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| corrupt("entry payload shorter than its variant tag"))
}

fn baked_loc(l: &MBakedLoc) -> SourceLocation {
    SourceLocation {
        file: l.file.clone(),
        range: LocationRange::new(l.range.0, l.range.1, l.range.2, l.range.3),
    }
}

/// Rehydrate the baked diagnostics: owner ids reconstruct directly into
/// runtime `TypeName`s (content-addressed, ADR-0025/ADR-0028 F2), and
/// symbol / name strings re-intern into the shared `NameTable`.
fn baked_from_mirror(
    m: MBakedDiagnostics,
    names: &NameTable,
) -> Result<RoastedGScan, RebuildError> {
    let tn = |id: u64| -> Result<TypeName, RebuildError> { Ok(TypeName::from_hash(id)) };
    let method_group =
        |g: &MBakedMethodGroup| -> Result<(TypeName, MethodDups, AliasCycles), RebuildError> {
            let dups: MethodDups = g
                .dups
                .iter()
                .map(|(name, dup_loc, original_loc)| {
                    (
                        name.clone(),
                        dup_loc.as_ref().map(baked_loc),
                        original_loc
                            .as_ref()
                            .map(|l| DuplicateSource::Location(baked_loc(l))),
                    )
                })
                .collect();
            let cycles: AliasCycles = g
                .cycles
                .iter()
                .map(|c| AliasCycleEntry {
                    type_name: c.type_name.clone(),
                    alias_names: c.alias_names.clone(),
                    primary_location: c.primary_location.as_ref().map(baked_loc),
                })
                .collect();
            Ok((tn(g.owner)?, dups, cycles))
        };

    let classes = m
        .classes
        .iter()
        .map(method_group)
        .collect::<Result<Vec<_>, _>>()?;
    let interfaces = m
        .interfaces
        .iter()
        .map(method_group)
        .collect::<Result<Vec<_>, _>>()?;
    let variables = m
        .variables
        .iter()
        .map(
            |g| -> Result<(TypeName, Vec<VariableDuplication>), RebuildError> {
                let dups = g
                    .dups
                    .iter()
                    .map(|d| -> Result<VariableDuplication, RebuildError> {
                        let kind = match d.kind {
                            0 => VariableDuplicationKind::Instance,
                            1 => VariableDuplicationKind::ClassInstance,
                            k => return Err(corrupt(format!("unknown baked variable kind {k}"))),
                        };
                        Ok(VariableDuplication {
                            kind,
                            type_name: tn(g.owner)?,
                            variable_name: names.intern_symbol(&d.variable_name),
                            location: d.location.as_ref().map(baked_loc),
                            ruby_source_location: d.ruby_source_location.as_ref().map(
                                |(file, start, end)| RubyLocation {
                                    file: names.intern(file),
                                    start_byte: *start,
                                    end_byte: *end,
                                },
                            ),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((tn(g.owner)?, dups))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let ancestor_cycles = m
        .ancestor_cycles
        .iter()
        .map(|c| -> Result<BakedAncestorCycle, RebuildError> {
            Ok(BakedAncestorCycle {
                participants: c
                    .participants
                    .iter()
                    .map(|id| tn(*id))
                    .collect::<Result<Vec<_>, _>>()?,
                type_name: c.type_name.clone(),
                chain: c.chain.clone(),
                primary_source: c.primary_source.as_ref().map(baked_loc),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let arity = m
        .arity
        .iter()
        .map(
            |g| -> Result<(TypeName, Vec<BakedArityViolation>), RebuildError> {
                let violations = g
                    .violations
                    .iter()
                    .map(|v| -> Result<BakedArityViolation, RebuildError> {
                        let kind = match v.kind {
                            0 => "superclass",
                            1 => "include",
                            2 => "extend",
                            3 => "prepend",
                            k => return Err(corrupt(format!("unknown baked arity kind {k}"))),
                        };
                        Ok(BakedArityViolation {
                            kind,
                            target: v.target.clone(),
                            class: v.class.clone(),
                            expected: v.expected.clone(),
                            got: v.got as usize,
                            location: v.location.as_ref().map(baked_loc),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((tn(g.owner)?, violations))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    // Both endpoints of a super-edge are content-addressed ids the cold
    // walk drew from declared entry keys or the decl's raw super
    // reference, so each is already registered in the tn table.
    let resolved = m
        .super_edges
        .resolved
        .iter()
        .map(|(class, super_name)| Ok((tn(*class)?, tn(*super_name)?)))
        .collect::<Result<Vec<_>, RebuildError>>()?;
    let unresolved = m
        .super_edges
        .unresolved
        .iter()
        .map(|edge| {
            let context = edge
                .context
                .iter()
                .map(|c| tn(*c))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(BakedUnresolvedSuperEdge {
                class: tn(edge.class)?,
                raw_super: tn(edge.raw_super)?,
                context,
            })
        })
        .collect::<Result<Vec<_>, RebuildError>>()?;

    Ok(RoastedGScan {
        classes,
        interfaces,
        variables,
        ancestor_cycles,
        arity,
        super_edges: crate::definition_builder::SuperEdges {
            resolved,
            unresolved,
        },
        warnings: m.warnings.clone(),
    })
}

/// Read-side counterpart of the frozen gem environment: owns the raw
/// snapshot bytes and serves per-name entries on demand.
///
/// Decode failures after `open` succeeded panic: the payload structure
/// was validated up front (`flat::Reader::try_open`), the invalidation
/// key matched, and writes are atomic — a bincode error here means the
/// file was corrupted in place, and silently dropping gem declarations
/// would produce wrong diagnostics. Deleting `.crema` recovers.
/// The snapshot file image (64-byte header + flat payload) a backend
/// reads from. The warm path maps the file (`read_g_snapshot_backend`)
/// so its pages stay file-backed instead of being copied into the heap;
/// the cold path hands over the bytes it just encoded.
pub enum SnapshotImage {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for SnapshotImage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            SnapshotImage::Owned(v) => v,
            SnapshotImage::Mapped(m) => m,
        }
    }
}

impl From<Vec<u8>> for SnapshotImage {
    fn from(v: Vec<u8>) -> Self {
        SnapshotImage::Owned(v)
    }
}

impl From<memmap2::Mmap> for SnapshotImage {
    fn from(m: memmap2::Mmap) -> Self {
        SnapshotImage::Mapped(m)
    }
}

pub struct GSnapshotBackend {
    /// Full snapshot file image (64-byte header + flat payload).
    bytes: SnapshotImage,
    tables: DecodeTables,
    /// One slot per entry the snapshot lists, filled on first probe.
    /// The key set is complete at open, so a probe never inserts a key:
    /// it only fills the slot, and a filled slot is read without a lock.
    class_kinds: FxHashMap<TypeName, (GClassKind, OnceLock<ClassOrModule>)>,
    interface_keys: FxHashMap<TypeName, OnceLock<InterfaceEntry>>,
    type_alias_keys: FxHashMap<TypeName, OnceLock<SingleEntry<TypeAlias>>>,
    constant_keys: FxHashMap<TypeName, OnceLock<SingleEntry<Constant>>>,
    /// Global probe needs the snapshot-side id; runtime `Symbol` is not
    /// itself the flat id (the flat id hashes the resolved string).
    global_ids: FxHashMap<Symbol, (u64, OnceLock<GlobalEntry>)>,
    /// Eagerly decoded `class A = B` / `module A = B` entries.
    alias_entries: FxHashMap<TypeName, ClassOrModuleAliasEntry>,
    /// G-only scan diagnostics baked at cold time (ADR-0028 slice 2b),
    /// decoded eagerly at open — size is O(findings), not O(entries).
    baked: RoastedGScan,

    /// Decode-all result, every kind at once (see `materialize_all`).
    all: OnceLock<Materialized>,

    /// Entry payload deserialization count (nested children included),
    /// bumped by every `EntrySource::Flat` fetch and the eager alias
    /// pass, for the `CREMA_DEBUG_SNAPSHOT_TIMING=1` observability line
    /// and the laziness regression tests.
    decode_count: AtomicUsize,
}

struct Materialized {
    class: FxHashMap<TypeName, ClassOrModule>,
    interface: FxHashMap<TypeName, InterfaceEntry>,
    type_alias: FxHashMap<TypeName, SingleEntry<TypeAlias>>,
    constant: FxHashMap<TypeName, SingleEntry<Constant>>,
    global: FxHashMap<Symbol, GlobalEntry>,
}

/// Fill `slot` with `compute()` unless it is already filled, and return
/// the stored value. `compute` runs outside the slot: two threads may
/// both decode the same entry, the first `set` wins and the loser's
/// value is dropped. Nothing waits on another thread's decode.
fn fill<V>(slot: &OnceLock<V>, compute: impl FnOnce() -> V) -> &V {
    if let Some(v) = slot.get() {
        return v;
    }
    let _ = slot.set(compute());
    slot.get().expect("set just above")
}

impl std::fmt::Debug for GSnapshotBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GSnapshotBackend")
            .field("bytes", &self.bytes.len())
            .field("classes", &self.class_kinds.len())
            .field("interfaces", &self.interface_keys.len())
            .field("type_aliases", &self.type_alias_keys.len())
            .field("constants", &self.constant_keys.len())
            .field("globals", &self.global_ids.len())
            .field("aliases", &self.alias_entries.len())
            .field("decoded", &self.decode_count())
            .finish()
    }
}

impl GSnapshotBackend {
    /// Open a snapshot file image whose header (magic / schema version /
    /// invalidation key) the caller has already verified. Interns the
    /// global tables into `names` — the same `NameTable` the A-layer
    /// draft will use, so ids are shared across layers.
    pub fn open(
        bytes: impl Into<SnapshotImage>,
        names: &NameTable,
    ) -> Result<GSnapshotBackend, RebuildError> {
        let bytes = bytes.into();
        if bytes.len() < HEADER_LEN {
            return Err(corrupt(format!(
                "file shorter than header: {} bytes",
                bytes.len()
            )));
        }
        let (tables, class_kinds, interface_keys, type_alias_keys, constant_keys, global_ids) = {
            let reader = flat::Reader::try_open(&bytes[HEADER_LEN..]).map_err(corrupt)?;
            let tables = DecodeTables::intern_from(&reader, names)?;

            let mut class_kinds = FxHashMap::default();
            let mut interface_keys = FxHashMap::default();
            let mut type_alias_keys = FxHashMap::default();
            let mut constant_keys = FxHashMap::default();
            let mut global_ids = FxHashMap::default();
            let tn = |id: u64| -> Result<TypeName, RebuildError> { Ok(TypeName::from_hash(id)) };
            for (id, payload) in reader.iter_entries() {
                match entry_tag(payload, 0)? {
                    TAG_CLASS_OR_MODULE => {
                        let kind = match entry_tag(payload, 4)? {
                            TAG_INNER_CLASS => GClassKind::Class,
                            TAG_INNER_MODULE => GClassKind::Module,
                            t => return Err(corrupt(format!("unknown class/module tag {t}"))),
                        };
                        class_kinds.insert(tn(id)?, (kind, OnceLock::new()));
                    }
                    TAG_INTERFACE => {
                        interface_keys.insert(tn(id)?, OnceLock::new());
                    }
                    TAG_CLASS_ALIAS => {} // decoded eagerly below
                    TAG_TYPE_ALIAS => {
                        type_alias_keys.insert(tn(id)?, OnceLock::new());
                    }
                    TAG_CONSTANT => {
                        constant_keys.insert(tn(id)?, OnceLock::new());
                    }
                    TAG_GLOBAL => {
                        global_ids.insert(Symbol::from_raw_id(id), (id, OnceLock::new()));
                    }
                    t => return Err(corrupt(format!("unknown entry tag {t}"))),
                }
            }
            (
                tables,
                class_kinds,
                interface_keys,
                type_alias_keys,
                constant_keys,
                global_ids,
            )
        };

        let baked = {
            let diag_off = u64::from_le_bytes(
                bytes[DIAG_OFF_OFFSET..DIAG_OFF_OFFSET + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let diag_len = u64::from_le_bytes(
                bytes[DIAG_LEN_OFFSET..DIAG_LEN_OFFSET + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let end = diag_off
                .checked_add(diag_len)
                .filter(|&end| diag_off >= HEADER_LEN && end <= bytes.len())
                .ok_or_else(|| corrupt("baked diagnostics section out of bounds"))?;
            let mirror: MBakedDiagnostics =
                bincode::deserialize(&bytes[diag_off..end]).map_err(RebuildError::BincodeDecode)?;
            baked_from_mirror(mirror, names)?
        };

        let mut backend = GSnapshotBackend {
            bytes,
            tables,
            class_kinds,
            interface_keys,
            type_alias_keys,
            constant_keys,
            global_ids,
            alias_entries: FxHashMap::default(),
            baked,
            all: OnceLock::new(),
            decode_count: AtomicUsize::new(0),
        };
        backend.alias_entries = backend.decode_alias_entries()?;
        Ok(backend)
    }

    fn reader(&self) -> flat::Reader<'_> {
        flat::Reader::open(&self.bytes[HEADER_LEN..])
    }

    fn count_decodes(&self, n: usize) {
        self.decode_count.fetch_add(n, Ordering::Relaxed);
    }

    pub fn decode_count(&self) -> usize {
        self.decode_count.load(Ordering::Relaxed)
    }

    /// Baked G-only scan diagnostics, spliced by the warm
    /// `DefinitionBuilder` scans in place of a G-layer walk.
    pub(crate) fn baked(&self) -> &RoastedGScan {
        &self.baked
    }

    /// The G-construction stderr warnings baked at cold time — `pub`
    /// (unlike `baked`) so `main.rs`, in the separate binary crate,
    /// can replay them verbatim on a warm hit.
    pub fn warnings(&self) -> &[String] {
        &self.baked.warnings
    }

    /// Decode the raw entry payload for `id` and convert it with `f`.
    /// The mirror-payload `memo` is caller-scoped: a nested child
    /// fetched while converting its parent is not deserialized twice
    /// within the same memo's lifetime, but nothing deserialized here
    /// outlives that memo — point probes pass a fresh one per call,
    /// the decode-all pass shares one across its whole walk (see
    /// [`Self::materialize_all`]).
    fn decode_entry_with<T>(
        &self,
        id: u64,
        dec: &mut Decoder<'_>,
        memo: &EntryMemo,
        f: impl FnOnce(&mut Decoder<'_>, &EntrySource<'_>, &MEntry) -> Result<T, RebuildError>,
    ) -> Result<Option<T>, RebuildError> {
        let reader = self.reader();
        let source = EntrySource::Flat {
            reader: &reader,
            memo,
            decodes: &self.decode_count,
        };
        let Some(entry) = source.get(id)? else {
            return Ok(None);
        };
        let value = f(dec, &source, &entry)?;
        Ok(Some(value))
    }

    fn decode_alias_entries(
        &self,
    ) -> Result<FxHashMap<TypeName, ClassOrModuleAliasEntry>, RebuildError> {
        let reader = self.reader();
        let mut out = FxHashMap::default();
        let mut decoded = 0usize;
        for (id, payload) in reader.iter_entries() {
            if entry_tag(payload, 0)? != TAG_CLASS_ALIAS {
                continue;
            }
            let entry: MEntry =
                bincode::deserialize(payload).map_err(RebuildError::BincodeDecode)?;
            decoded += 1;
            let MEntry::ClassAlias(e) = &entry else {
                return Err(corrupt(format!(
                    "entry {id:#x} tag says class-alias but payload decodes otherwise"
                )));
            };
            let dec = Decoder::new(&self.tables);
            let name = TypeName::from_hash(id);
            let context = dec.context(&e.context)?;
            let frozen = match e.kind {
                0 => ClassOrModuleAliasEntry::Class(ClassAliasEntry {
                    name,
                    file: DeclOrigin::GSnapshot,
                    context,
                    decl: ClassAliasDeclaration::Signature(Arc::new(
                        dec.class_alias_decl(&e.decl)?,
                    )),
                }),
                1 => ClassOrModuleAliasEntry::Module(ModuleAliasEntry {
                    name,
                    file: DeclOrigin::GSnapshot,
                    context,
                    decl: ModuleAliasDeclaration::Signature(Arc::new(
                        dec.module_alias_decl(&e.decl)?,
                    )),
                }),
                k => return Err(corrupt(format!("unknown class-alias kind {k}"))),
            };
            out.insert(name, frozen);
        }
        self.count_decodes(decoded);
        Ok(out)
    }

    /// Warm point probe: the decoded entry carries its own members only —
    /// nested decls stay behind their `MDeclRef` and are read from their
    /// own entries (`Decoder::lazy`).
    fn decode_class_entry(&self, name: TypeName) -> Result<Option<ClassOrModule>, RebuildError> {
        let mut dec = Decoder::lazy(&self.tables);
        let memo = EntryMemo::default();
        self.decode_class_entry_with(name, &mut dec, &memo)
    }

    /// `dec` may carry the shared nested-decl cache (decode-all pass):
    /// a flatten pair whose declaration was already converted while
    /// decoding its parent reuses that Arc instead of re-converting the
    /// subtree. Pair indexes align because a child's pairs are appended
    /// in exactly the order its parents' decl refs consume them.
    fn decode_class_entry_with(
        &self,
        name: TypeName,
        dec: &mut Decoder<'_>,
        memo: &EntryMemo,
    ) -> Result<Option<ClassOrModule>, RebuildError> {
        use crate::ast::declarations::Declaration;
        let id = name.get();
        self.decode_entry_with(id, dec, memo, |dec, source, entry| match entry {
            MEntry::ClassOrModule(MClassOrModule::Class(e)) => {
                let mut pairs: Vec<(DeclOrigin, Context, ClassDeclaration)> =
                    Vec::with_capacity(e.context_decls.len());
                for (i, (ctx, d)) in e.context_decls.iter().enumerate() {
                    let decl = match dec.nested_decl((id, i)) {
                        Some(Declaration::Class(arc)) => ClassDeclaration::Signature(arc),
                        Some(_) => return Err(corrupt("nested-decl cache kind mismatch")),
                        None => {
                            ClassDeclaration::Signature(Arc::new(dec.class_decl(source, d, ctx)?))
                        }
                    };
                    // G-snapshot mirror doesn't carry the per-decl file
                    // slot (byte format frozen; see mirror.rs doc) — the
                    // decl's own `source_file` stands in for Signature
                    // decls, so the entry-level slot records the G
                    // provenance itself.
                    pairs.push((DeclOrigin::GSnapshot, dec.context(ctx)?, decl));
                }
                let primary_decl = pairs
                    .get(e.primary as usize)
                    .map(|(_, _, d)| d.clone())
                    .ok_or_else(|| corrupt(format!("primary index {} out of range", e.primary)))?;
                Ok(ClassOrModule::Class(ClassEntry {
                    name,
                    context_decls: pairs.into_boxed_slice(),
                    primary_decl,
                }))
            }
            MEntry::ClassOrModule(MClassOrModule::Module(e)) => {
                let mut pairs: Vec<(DeclOrigin, Context, ModuleDeclaration)> =
                    Vec::with_capacity(e.context_decls.len());
                for (i, (ctx, d)) in e.context_decls.iter().enumerate() {
                    let decl = match dec.nested_decl((id, i)) {
                        Some(Declaration::Module(arc)) => ModuleDeclaration::Signature(arc),
                        Some(_) => return Err(corrupt("nested-decl cache kind mismatch")),
                        None => {
                            ModuleDeclaration::Signature(Arc::new(dec.module_decl(source, d, ctx)?))
                        }
                    };
                    pairs.push((DeclOrigin::GSnapshot, dec.context(ctx)?, decl));
                }
                let primary_decl = pairs
                    .get(e.primary as usize)
                    .map(|(_, _, d)| d.clone())
                    .ok_or_else(|| corrupt(format!("primary index {} out of range", e.primary)))?;
                Ok(ClassOrModule::Module(ModuleEntry {
                    name,
                    context_decls: pairs.into_boxed_slice(),
                    primary_decl,
                }))
            }
            _ => Err(corrupt(format!(
                "entry for {name:?} is not a class/module payload"
            ))),
        })
    }

    /// Depth of the type-name parent chain — used to materialize parents
    /// before children so nested-decl conversions are shared, not
    /// repeated per ancestor level.
    fn tn_depth(&self, id: u64) -> usize {
        let reader = self.reader();
        let mut depth = 0;
        let mut cur = id;
        while let Some((parent, _, _)) = reader.tn_entry(cur) {
            if parent == 0 {
                break;
            }
            depth += 1;
            cur = parent;
        }
        depth
    }

    fn expect_decoded<T>(&self, what: &str, r: Result<Option<T>, RebuildError>) -> T {
        match r {
            Ok(Some(v)) => v,
            Ok(None) => panic!(
                "g-snapshot backend: {what} listed in entry index but absent on probe \
                 (delete .crema to recover)"
            ),
            Err(e) => panic!(
                "g-snapshot backend: {what} failed to decode after open-time validation: {e} \
                 (delete .crema to recover)"
            ),
        }
    }

    // ---- per-kind point lookups ----

    pub fn class_entry(&self, name: &TypeName) -> Option<&ClassOrModule> {
        if let Some(all) = self.all.get() {
            return all.class.get(name);
        }
        let (_, slot) = self.class_kinds.get(name)?;
        Some(fill(slot, || {
            self.expect_decoded("class entry", self.decode_class_entry(*name))
        }))
    }

    pub fn interface_entry(&self, name: &TypeName) -> Option<&InterfaceEntry> {
        if let Some(all) = self.all.get() {
            return all.interface.get(name);
        }
        let slot = self.interface_keys.get(name)?;
        Some(fill(slot, || self.interface_entry_uncached(name)))
    }

    pub fn type_alias_entry(&self, name: &TypeName) -> Option<&SingleEntry<TypeAlias>> {
        if let Some(all) = self.all.get() {
            return all.type_alias.get(name);
        }
        let slot = self.type_alias_keys.get(name)?;
        Some(fill(slot, || self.type_alias_entry_uncached(name)))
    }

    pub fn constant_entry(&self, name: &TypeName) -> Option<&SingleEntry<Constant>> {
        if let Some(all) = self.all.get() {
            return all.constant.get(name);
        }
        let slot = self.constant_keys.get(name)?;
        Some(fill(slot, || self.constant_entry_uncached(name)))
    }

    pub fn global_entry(&self, name: &Symbol) -> Option<&GlobalEntry> {
        if let Some(all) = self.all.get() {
            return all.global.get(name);
        }
        let (id, slot) = self.global_ids.get(name)?;
        Some(fill(slot, || self.global_entry_uncached(name, *id)))
    }

    // ---- per-kind decode bodies, shared by point probes and `*_all` ----

    fn interface_entry_uncached(&self, name: &TypeName) -> InterfaceEntry {
        self.interface_entry_in(name, &EntryMemo::default())
    }

    fn interface_entry_in(&self, name: &TypeName, memo: &EntryMemo) -> InterfaceEntry {
        if let Some(v) = self.interface_keys.get(name).and_then(OnceLock::get) {
            return v.clone();
        }
        let mut dec = Decoder::new(&self.tables);
        let decoded =
            self.decode_entry_with(name.get(), &mut dec, memo, |dec, _, entry| match entry {
                MEntry::Interface(e) => Ok(InterfaceEntry {
                    name: *name,
                    file: DeclOrigin::GSnapshot,
                    context: dec.context(&e.context)?,
                    decl: Arc::new(dec.interface_decl(&e.decl)?),
                }),
                _ => Err(corrupt("entry tag/payload mismatch for interface")),
            });
        self.expect_decoded("interface entry", decoded)
    }

    fn type_alias_entry_uncached(&self, name: &TypeName) -> SingleEntry<TypeAlias> {
        self.type_alias_entry_in(name, &EntryMemo::default())
    }

    fn type_alias_entry_in(&self, name: &TypeName, memo: &EntryMemo) -> SingleEntry<TypeAlias> {
        if let Some(v) = self.type_alias_keys.get(name).and_then(OnceLock::get) {
            return v.clone();
        }
        let mut dec = Decoder::new(&self.tables);
        let decoded =
            self.decode_entry_with(name.get(), &mut dec, memo, |dec, _, entry| match entry {
                MEntry::TypeAlias(e) => Ok(SingleEntry {
                    name: *name,
                    file: DeclOrigin::GSnapshot,
                    context: dec.context(&e.context)?,
                    decl: Arc::new(dec.type_alias_decl(&e.decl)?),
                }),
                _ => Err(corrupt("entry tag/payload mismatch for type alias")),
            });
        self.expect_decoded("type-alias entry", decoded)
    }

    fn constant_entry_uncached(&self, name: &TypeName) -> SingleEntry<Constant> {
        self.constant_entry_in(name, &EntryMemo::default())
    }

    fn constant_entry_in(&self, name: &TypeName, memo: &EntryMemo) -> SingleEntry<Constant> {
        if let Some(v) = self.constant_keys.get(name).and_then(OnceLock::get) {
            return v.clone();
        }
        let mut dec = Decoder::new(&self.tables);
        let decoded =
            self.decode_entry_with(name.get(), &mut dec, memo, |dec, _, entry| match entry {
                MEntry::Constant(e) => Ok(SingleEntry {
                    name: *name,
                    file: DeclOrigin::GSnapshot,
                    context: dec.context(&e.context)?,
                    decl: Arc::new(dec.constant_decl(&e.decl)?),
                }),
                _ => Err(corrupt("entry tag/payload mismatch for constant")),
            });
        self.expect_decoded("constant entry", decoded)
    }

    fn global_entry_uncached(&self, name: &Symbol, id: u64) -> GlobalEntry {
        self.global_entry_in(name, id, &EntryMemo::default())
    }

    fn global_entry_in(&self, name: &Symbol, id: u64, memo: &EntryMemo) -> GlobalEntry {
        if let Some(v) = self.global_ids.get(name).and_then(|(_, slot)| slot.get()) {
            return v.clone();
        }
        let mut dec = Decoder::new(&self.tables);
        let decoded = self.decode_entry_with(id, &mut dec, memo, |dec, _, entry| match entry {
            MEntry::Global(e) => Ok(GlobalEntry {
                name: *name,
                // `MGlobalEntry` is the one G-only shape that persists a
                // real file, so a stored path survives as `Path`; only a
                // pathless payload falls back to the G provenance marker.
                file: match dec.opt_name(e.file)? {
                    Some(f) => DeclOrigin::Path(f),
                    None => DeclOrigin::GSnapshot,
                },
                context: dec.context(&e.context)?,
                decl: Arc::new(dec.global_decl(&e.decl)?),
            }),
            _ => Err(corrupt("entry tag/payload mismatch for global")),
        });
        self.expect_decoded("global entry", decoded)
    }

    // ---- contains / len (no decode) ----

    pub fn class_kind(&self, name: &TypeName) -> Option<GClassKind> {
        self.class_kinds.get(name).map(|(kind, _)| *kind)
    }

    pub fn class_contains(&self, name: &TypeName) -> bool {
        self.class_kinds.contains_key(name)
    }

    pub fn interface_contains(&self, name: &TypeName) -> bool {
        self.interface_keys.contains_key(name)
    }

    pub fn type_alias_contains(&self, name: &TypeName) -> bool {
        self.type_alias_keys.contains_key(name)
    }

    pub fn constant_contains(&self, name: &TypeName) -> bool {
        self.constant_keys.contains_key(name)
    }

    pub fn global_contains(&self, name: &Symbol) -> bool {
        self.global_ids.contains_key(name)
    }

    pub fn alias_contains(&self, name: &TypeName) -> bool {
        self.alias_entries.contains_key(name)
    }

    pub fn class_len(&self) -> usize {
        self.class_kinds.len()
    }

    pub fn interface_len(&self) -> usize {
        self.interface_keys.len()
    }

    pub fn type_alias_len(&self) -> usize {
        self.type_alias_keys.len()
    }

    pub fn constant_len(&self) -> usize {
        self.constant_keys.len()
    }

    pub fn global_len(&self) -> usize {
        self.global_ids.len()
    }

    // ---- keys (no decode; ADR-0028 slice 2b-3) ----
    //
    // Boxed so `GView` can hold one fn-pointer shape per kind even
    // though the underlying storages differ (map keys vs set iter).

    pub fn class_key_iter(&self) -> Box<dyn Iterator<Item = &TypeName> + '_> {
        Box::new(self.class_kinds.keys())
    }

    pub fn interface_key_iter(&self) -> Box<dyn Iterator<Item = &TypeName> + '_> {
        Box::new(self.interface_keys.keys())
    }

    pub fn type_alias_key_iter(&self) -> Box<dyn Iterator<Item = &TypeName> + '_> {
        Box::new(self.type_alias_keys.keys())
    }

    pub fn constant_key_iter(&self) -> Box<dyn Iterator<Item = &TypeName> + '_> {
        Box::new(self.constant_keys.keys())
    }

    pub fn global_key_iter(&self) -> Box<dyn Iterator<Item = &Symbol> + '_> {
        Box::new(self.global_ids.keys())
    }

    // ---- resolver inputs (Pass 1 fallback, no decode) ----

    /// Names the A-layer resolver must treat as declared: classes,
    /// modules, interfaces, and type aliases. Deliberately excludes
    /// constants (rbs keeps them out of the type-name space; see the
    /// Pass 1 comment in `EnvironmentDraft::build`) and alias new-names
    /// (they feed the `aliases` map instead).
    pub fn resolver_names(&self) -> impl Iterator<Item = TypeName> + '_ {
        self.class_kinds
            .keys()
            .chain(self.interface_keys.keys())
            .chain(self.type_alias_keys.keys())
            .copied()
    }

    /// Eagerly decoded `class A = B` / `module A = B` entries, keyed by
    /// the alias new-name. Feeds the resolver `aliases` map, the
    /// combined normalize precompute, and the merged
    /// `class_alias_decls` map of the grafted environment.
    pub fn alias_entries(&self) -> &FxHashMap<TypeName, ClassOrModuleAliasEntry> {
        &self.alias_entries
    }

    // ---- full-scan materialization (decode-all, memoized) ----

    /// Build every kind's map in one pass sharing one payload memo: a
    /// nested interface / type alias / constant pulled in while
    /// converting its enclosing class is not deserialized again on its
    /// own kind's turn, so materializing costs at most one
    /// deserialization per entry. All five maps sit behind one
    /// `OnceLock` so a single thread runs the whole pass — with one lock
    /// per kind, a thread that lost the class pass could win a later
    /// kind and re-decode what the winner's (thread-local) memo already
    /// held. The memo is dropped with the pass. Every `*_all` accessor
    /// routes through here, so asking for one kind materializes all of
    /// them — acceptable because decode-all is a cold-only / test path,
    /// never a warm point probe.
    fn materialize_all(&self) -> &Materialized {
        self.all.get_or_init(|| {
            let memo = EntryMemo::default();
            // Parents first (see `tn_depth`), one decoder for the whole
            // pass so nested-decl consumption cursors and the shared
            // conversion cache span every entry — the same walk order a
            // draft-flatten build effectively performs.
            let mut names: Vec<TypeName> = self.class_kinds.keys().copied().collect();
            names.sort_by_cached_key(|n| self.tn_depth(n.get()));
            let nested = NestedDecls::default();
            let mut dec = Decoder::with_nested_cache(&self.tables, &nested);
            // Never reuse a filled point-probe slot here: its value came
            // from a lazy point probe (`Decoder::lazy`), whose parent decl
            // omits its nested decls — decode-all must hand out the
            // rbs-shaped inline form, so every entry is re-decoded with the
            // pass-local nested cache. Once `all` is set, the point probes
            // read from it and the slots are dead weight.
            let class = names
                .into_iter()
                .map(|name| {
                    let v = self.expect_decoded(
                        "class entry",
                        self.decode_class_entry_with(name, &mut dec, &memo),
                    );
                    (name, v)
                })
                .collect();
            let interface = self
                .interface_keys
                .keys()
                .map(|name| (*name, self.interface_entry_in(name, &memo)))
                .collect();
            let type_alias = self
                .type_alias_keys
                .keys()
                .map(|name| (*name, self.type_alias_entry_in(name, &memo)))
                .collect();
            let constant = self
                .constant_keys
                .keys()
                .map(|name| (*name, self.constant_entry_in(name, &memo)))
                .collect();
            let global = self
                .global_ids
                .iter()
                .map(|(name, (id, _))| (*name, self.global_entry_in(name, *id, &memo)))
                .collect();
            Materialized {
                class,
                interface,
                type_alias,
                constant,
                global,
            }
        })
    }

    pub fn class_all(&self) -> &FxHashMap<TypeName, ClassOrModule> {
        &self.materialize_all().class
    }

    pub fn interface_all(&self) -> &FxHashMap<TypeName, InterfaceEntry> {
        &self.materialize_all().interface
    }

    pub fn type_alias_all(&self) -> &FxHashMap<TypeName, SingleEntry<TypeAlias>> {
        &self.materialize_all().type_alias
    }

    pub fn constant_all(&self) -> &FxHashMap<TypeName, SingleEntry<Constant>> {
        &self.materialize_all().constant
    }

    pub fn global_all(&self) -> &FxHashMap<Symbol, GlobalEntry> {
        &self.materialize_all().global
    }
}

