//! Per-file ingest cache (ADR-0036 Decision 4-1): the collect output of
//! every scope file, persisted so a run that only needs some files'
//! ASTs (a `crema check <file>` from an editor hook) can skip the read,
//! prism parse, inline collect and infusion collect of every file that
//! has not changed since the last run.
//!
//! What a file's ingest output depends on is exactly its bytes, the
//! crema.toml-derived collect options and the binary, so the cache is a
//! pure function of (key, file content). Freshness is judged by stat
//! (mtime + size), the same trade cargo and ninja make: a content hash
//! would cost the read the cache exists to avoid. The hash is still
//! stored per record for `CREMA_DEBUG_SNAPSHOT_TIMING=1` verification.
//!
//! Layout: one file, `.crema/ingest_cache_v1.bin`, a 48-byte
//! header (magic, schema version, invalidation key) followed by a
//! bincode [`Contents`]: the scope walk's directory listings, then one
//! [`Record`] per file, both in walk order. Each record keeps its
//! payload as opaque bytes, so opening the file costs one `Vec<u8>` per
//! record and only the hits are decoded (on the ingest worker, in
//! parallel).
//!
//! The directory listings let a lookup run skip the scope's `read_dir`s
//! too: a directory whose stat (mtime + size) matches its listing is
//! taken from the cache, any other one is read again. A directory's
//! mtime moves when an entry is added, removed or renamed in it (POSIX;
//! APFS / ext4 / xfs keep nanoseconds), not when a file's content
//! changes — content freshness stays with the per-file stat of the
//! records. Known limit, not worked around: a filesystem with
//! one-second mtime granularity (HFS+) can miss an entry added within
//! the same second as the previous walk. A listing is taken before its
//! `read_dir` is issued, so a change landing between the two is seen as
//! a stale listing next run rather than a fresh one with old contents.
//!
//! Two ids inside a payload are not stable across runs and are fixed up
//! through a thread-local codec scope that the `serde` impls of `Name`
//! and `Id<T>` consult:
//!
//! - `Name` is positional (`NameOverlay`). The only `Name` a worker can
//!   stamp into its output is the file's own path (pre-interned on main),
//!   so a record stores that one id and decode substitutes the current
//!   run's `Name` for it. Any other `Name` is a bug: encode asserts in
//!   debug builds and refuses to cache the file in release; decode
//!   rejects the record.
//! - `Symbol` / `TypeName` ids are content-addressed and therefore
//!   stable, but the strings behind them live in a `NameTable`, and a
//!   file served from the cache is the only place its names may occur.
//!   Encode collects every id the payload serializes and stores their
//!   strings; decode re-interns them into the worker's table first.

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::ast::ruby::declarations::Declaration;
use crate::diagnostic::Diagnostic;
use crate::infusion_collector::CollectedSource;
use crate::name::{Name, NameTable, Symbol};
use crate::snapshot::invalidation::InvalidationKey;
use crate::type_name::TypeName;


pub const INGEST_CACHE_FILE: &str = ".crema/ingest_cache_v1.bin";
/// Identifies a crema ingest-cache file ("C5EA 1A6E" ≈ crema ingest).
pub const INGEST_CACHE_MAGIC: u64 = 0xC5EA_1A6E_0000_0001;
/// 2: directory listings were added in front of the records.
pub const INGEST_CACHE_SCHEMA_VERSION: u32 = 2;
/// magic u64 LE | schema_version u32 LE | reserved u32 = 0 | key 32 bytes
pub const HEADER_LEN: usize = 48;

/// What one scope file's ingest produced: the inline collector's
/// declarations and diagnostics plus, when an infusion is active, the
/// infusion collector's output. `SyntaxError` is cached too — a file
/// prism rejects contributes nothing but its diagnostic, and that fact
/// is as stable as any other collect result.
#[derive(Serialize, Deserialize)]
pub enum IngestPayload<'a> {
    SyntaxError(Diagnostic),
    Parsed {
        decls: Vec<Declaration>,
        diags: Vec<Diagnostic>,
        infusion: Option<CollectedSource<'a>>,
    },
}

/// (mtime, size) of a file, taken before it is read so a write landing
/// between the stat and the read makes the next run miss rather than
/// serve the stale parse under the new stat.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileStat {
    pub mtime_secs: u64,
    pub mtime_nanos: u32,
    pub size: u64,
}

impl FileStat {
    pub fn of(meta: &fs::Metadata) -> Self {
        let since_epoch = meta
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        FileStat {
            mtime_secs: since_epoch.as_secs(),
            mtime_nanos: since_epoch.subsec_nanos(),
            size: meta.len(),
        }
    }
}

/// One file's entry in the cache file.
#[derive(Serialize, Deserialize, Debug)]
pub struct Record {
    /// Canonical path, as the scope walk produced it.
    pub path: String,
    pub stat: FileStat,
    /// xxh3 of the bytes the payload was collected from. Not consulted
    /// for freshness; `CREMA_DEBUG_SNAPSHOT_TIMING=1` reads hit files
    /// back and reports a mismatch.
    pub content_hash: u64,
    /// The raw id of the file's own `Name` when the payload was encoded.
    pub self_name: u32,
    /// bincode of [`StringTable`].
    pub strings: Vec<u8>,
    /// bincode of [`IngestPayload`].
    pub payload: Vec<u8>,
}

/// Everything the cache file stores after its header.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Contents {
    /// Every directory the scope walk read, in walk order.
    pub dirs: Vec<DirListing>,
    pub records: Vec<Record>,
}

/// What the scope walk saw in one directory: the entries it would act
/// on (`.rb` files and subdirectories), in `read_dir` order, with the
/// stat the directory had just before it was read. Entries the walk
/// ignores (other files, symlinked directories) are not stored. A
/// directory that produced a walk warning, or has an entry name that is
/// not UTF-8, is not stored either — it is read again every run.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirListing {
    /// The directory by the spelling the walk reached it with (the
    /// `check` entry's path, then `read_dir` names joined onto it).
    pub path: String,
    /// Stat of the directory itself, taken before its `read_dir`.
    pub stat: FileStat,
    pub entries: Vec<DirEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    /// Whether the entry inherits its parent directory's `joinable`
    /// (a plain entry) or is never joinable (a symlinked file, or an
    /// entry whose type `read_dir` could not report). Relative to the
    /// parent so the same listing serves a walk from any root.
    pub joinable: bool,
}

/// The strings behind every `Symbol` / `TypeName` a payload serializes.
/// A type name is its root (absolute or relative) plus segment indices
/// into `symbols`; re-interning them in that order rebuilds the same
/// content-addressed ids.
#[derive(Serialize, Deserialize, Default)]
struct StringTable {
    symbols: Vec<String>,
    type_names: Vec<(bool, Vec<u32>)>,
}

/// The ingest cache keys on everything the G snapshot keys on (crema.toml
/// content, binary identity, ...) plus the one collect input crema.toml
/// does not carry: the CLI `--inline` override. Sharing the G inputs
/// over-invalidates on a lockfile change, which costs one cold ingest.
pub fn derive_key(base: &InvalidationKey, inline: bool) -> InvalidationKey {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(b"ingest-cache-v1");
    buf.extend_from_slice(&base.0);
    buf.push(inline as u8);
    let mut key = [0u8; 32];
    for seed in 0..4u64 {
        let h = xxhash_rust::xxh3::xxh3_64_with_seed(&buf, seed);
        key[seed as usize * 8..(seed as usize + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    InvalidationKey(key)
}

pub fn encode_file(key: &InvalidationKey, contents: &Contents) -> Vec<u8> {
    let mut out = vec![0u8; HEADER_LEN];
    out[0..8].copy_from_slice(&INGEST_CACHE_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&INGEST_CACHE_SCHEMA_VERSION.to_le_bytes());
    out[16..48].copy_from_slice(&key.0);
    bincode::serialize_into(&mut out, contents).expect("ingest contents serialize into a Vec");
    out
}

/// Read and validate the cache file. Every failure is a reason string
/// for the caller's debug line; the run itself must not notice (silent
/// cold, same contract as the G snapshot).
pub fn read_file(path: &Path, expected_key: &InvalidationKey) -> Result<Contents, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    if bytes.len() < HEADER_LEN {
        return Err("file shorter than header".into());
    }
    if u64::from_le_bytes(bytes[0..8].try_into().unwrap()) != INGEST_CACHE_MAGIC {
        return Err("bad ingest cache magic".into());
    }
    if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != INGEST_CACHE_SCHEMA_VERSION {
        return Err("unsupported schema version".into());
    }
    if bytes[16..48] != expected_key.0 {
        return Err("invalidation key mismatch".into());
    }
    bincode::deserialize(&bytes[HEADER_LEN..]).map_err(|e| format!("corrupt records: {e}"))
}

/// The encoded halves of a record that depend on the payload.
pub struct EncodedPayload {
    pub self_name: u32,
    pub strings: Vec<u8>,
    pub payload: Vec<u8>,
}

/// Serialize `payload` for the file whose own `Name` is `self_name`,
/// resolving the strings it references through `names` (the table the
/// payload was collected against). `None` when the payload carries a
/// `Name` other than `self_name` — such a file is simply not cached.
pub fn encode_payload(
    names: &NameTable,
    self_name: Name,
    payload: &IngestPayload<'_>,
) -> Option<EncodedPayload> {
    SCOPE.with(|scope| {
        *scope.borrow_mut() = Some(Scope::Encode {
            self_name,
            symbols: Vec::new(),
            seen_symbols: FxHashSet::default(),
            type_names: Vec::new(),
            seen_type_names: FxHashSet::default(),
            foreign_name: false,
        });
    });
    let payload_bytes = bincode::serialize(payload);
    let scope = SCOPE
        .with(|scope| scope.borrow_mut().take())
        .expect("encode scope is set above");
    let payload = payload_bytes.expect("ingest payload serializes into a Vec");
    let Scope::Encode {
        mut symbols,
        type_names,
        foreign_name,
        ..
    } = scope
    else {
        unreachable!("encode scope was installed above")
    };
    debug_assert!(
        !foreign_name,
        "ingest payload for {} carries a Name other than the file's own",
        names.resolve(self_name)
    );
    if foreign_name {
        return None;
    }
    // A type name's segments are symbols too; make sure the table lists
    // them even when the payload never serialized the bare segment.
    let mut index: FxHashMap<Symbol, u32> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, i as u32))
        .collect();
    let mut table_type_names = Vec::with_capacity(type_names.len());
    for tn in type_names {
        let absolute = names.type_name_is_absolute(tn);
        let segments = names
            .type_name_segments(tn)
            .into_iter()
            .map(|seg| {
                *index.entry(seg).or_insert_with(|| {
                    symbols.push(seg);
                    symbols.len() as u32 - 1
                })
            })
            .collect();
        table_type_names.push((absolute, segments));
    }
    let table = StringTable {
        symbols: symbols
            .iter()
            .map(|s| names.resolve(*s).to_string())
            .collect(),
        type_names: table_type_names,
    };
    Some(EncodedPayload {
        self_name: name_raw(self_name),
        strings: bincode::serialize(&table).expect("string table serializes into a Vec"),
        payload,
    })
}

/// Decode `record`'s payload for the current run, where the file's own
/// `Name` is `self_name`, interning the payload's strings into `names`
/// first. `None` on any inconsistency (a record is then treated as a
/// miss). The returned `CollectedSource`, if any, still needs
/// [`CollectedSource::rebind`] for its walk index and source.
pub fn decode_payload<'a>(
    record: &Record,
    names: &NameTable,
    self_name: Name,
) -> Option<IngestPayload<'a>> {
    let table: StringTable = bincode::deserialize(&record.strings).ok()?;
    let symbols: Vec<Symbol> = table
        .symbols
        .iter()
        .map(|s| names.intern_symbol(s))
        .collect();
    for (absolute, segments) in &table.type_names {
        let mut segs = Vec::with_capacity(segments.len());
        for &i in segments {
            segs.push(*symbols.get(i as usize)?);
        }
        names.extend_type_name(names.type_name_root(*absolute), segs);
    }
    SCOPE.with(|scope| {
        *scope.borrow_mut() = Some(Scope::Decode {
            stored_self: record.self_name,
            self_name,
        });
    });
    let decoded = bincode::deserialize::<IngestPayload<'a>>(&record.payload);
    SCOPE.with(|scope| scope.borrow_mut().take());
    decoded.ok()
}

enum Scope {
    Encode {
        self_name: Name,
        symbols: Vec<Symbol>,
        seen_symbols: FxHashSet<Symbol>,
        type_names: Vec<TypeName>,
        seen_type_names: FxHashSet<TypeName>,
        foreign_name: bool,
    },
    Decode {
        stored_self: u32,
        self_name: Name,
    },
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

fn name_raw(name: Name) -> u32 {
    name.overlay_id()
}

/// Called by `Name`'s `Serialize` with the raw overlay id about to be
/// written.
pub(crate) fn encoding_name(raw: u32) {
    SCOPE.with(|scope| {
        if let Some(Scope::Encode {
            self_name,
            foreign_name,
            ..
        }) = scope.borrow_mut().as_mut()
            && name_raw(*self_name) != raw
        {
            *foreign_name = true;
        }
    });
}

/// Called by `Name`'s `Deserialize` with the raw id read: the id to
/// construct, or `None` when a decode scope is active and `raw` is not
/// the recorded self name.
pub(crate) fn decoding_name(raw: u32) -> Option<u32> {
    SCOPE.with(|scope| match scope.borrow().as_ref() {
        Some(Scope::Decode {
            stored_self,
            self_name,
        }) => (*stored_self == raw).then(|| name_raw(*self_name)),
        Some(Scope::Encode { .. }) | None => Some(raw),
    })
}

/// Called by `Id<SymbolTag>`'s `Serialize`.
pub(crate) fn encoding_symbol(raw: u64) {
    SCOPE.with(|scope| {
        if let Some(Scope::Encode {
            symbols,
            seen_symbols,
            ..
        }) = scope.borrow_mut().as_mut()
        {
            let sym = Symbol::from_raw_id(raw);
            if seen_symbols.insert(sym) {
                symbols.push(sym);
            }
        }
    });
}

/// Called by `Id<TypeNameTag>`'s `Serialize`.
pub(crate) fn encoding_type_name(raw: u64) {
    SCOPE.with(|scope| {
        if let Some(Scope::Encode {
            type_names,
            seen_type_names,
            ..
        }) = scope.borrow_mut().as_mut()
        {
            let tn = TypeName::from_hash(raw);
            if seen_type_names.insert(tn) {
                type_names.push(tn);
            }
        }
    });
}
