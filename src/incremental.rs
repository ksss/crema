//! ADR-0032 Decision 1/4: the incremental check cache — persistence and
//! per-file recheck/replay classification.
//!
//! Persists exactly the four artifact kinds Decision 1 allows: per-file
//! content hashes, per-file consulted key sets (projected to stable
//! `u64`s, see below), the per-name fingerprint table, and structured
//! check-phase diagnostics. Never a type environment, never a `Ty` —
//! the cache feeds only the scheduler (which files to skip); the checker
//! always reads a fresh env.
//!
//! Every persisted hash is a content-addressed interner id
//! (`TypeName`/`Symbol`, xxh3-derived — ADR-0025), never a
//! `#[derive(Hash)]` (SipHash) value: the same logical name maps to the
//! same id in any process, the property the G-snapshot decode path
//! already builds on (`Symbol::from_raw_id`).
//!
//! Any read failure — missing file, short file, wrong magic/version/key,
//! bincode error — degrades to `None`, which the caller treats as "no
//! cache": a silent full recheck. Corruption can never surface as a
//! diagnostic or a panic (Decision 4: version bump = full recheck
//! fallback keeps migration cost at zero).

use std::path::Path;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::definition::ancestor_graph::AncestorGraph;
use crate::definition_builder::ConsultedKey;
use crate::diagnostic::Diagnostic;
use crate::environment::fingerprint::{
    FingerprintKey, FingerprintTable, RawDiff, compute_fingerprint_table, expand_changed_set,
};
use crate::environment::frozen::Environment;
use crate::name::{NameTable, Symbol};
use crate::snapshot::invalidation::InvalidationKey;
use crate::snapshot::write::write_atomic_bytes;
use crate::type_name::TypeName;


/// Cache location, sibling of the G snapshot under the same
/// "delete `.crema/cache` to recover" operational contract.
pub const INCREMENTAL_CACHE_FILE: &str = ".crema/cache/incremental_v1.bin";

/// Identifies a crema incremental check cache ("C5EA 1CAC" ≈ crema
/// incremental cache).
pub const CACHE_MAGIC: u64 = 0xC5EA_1CAC_0000_0001;

/// Bump on any layout-affecting change to [`CachePayload`] or a type it
/// transitively serializes (notably `Diagnostic`/`DiagnosticKind` —
/// bincode has no schema evolution, and a same-version misparse is the
/// one failure mode the header cannot catch). Also bump when the
/// *meaning* of the recorded consulted set changes even though the wire
/// format doesn't: a cache written before a new `ConsultedKey` family
/// existed carries consulted columns missing those entries, and reading
/// it would replay exactly the stale diagnostics the new keys were
/// added to prevent (v1→v2: ancestor-walk queries became recorded).
/// v2→v3: `FingerprintKey::ConcernTargets` and
/// `ConsultedKey::ConcernBlockTargets` (concern block bodies checked per
/// include target); a v2 cache has neither column.
pub const CACHE_SCHEMA_VERSION: u32 = 3;

/// Probe id for a [`FingerprintKey::ConcernTargets`] change, shared by
/// [`probe_hashes`] and the `ConsultedKey::ConcernBlockTargets`
/// projection. Distinct from the bare name id so a concern's target-set
/// change reaches only the file that walked its block, not every file
/// that merely resolved the module (a `DeclaredKind` / include probe
/// carries the bare id).
pub fn concern_targets_probe_id(concern: TypeName) -> u64 {
    concern.get().rotate_left(32) ^ 0x434f_4e43_4552_4e21 // "CONCERN!"
}

/// Layout: magic u64 LE | schema_version u32 LE | reserved u32 = 0 |
/// invalidation key 32 bytes. Mirrors the G-snapshot header shape.
pub const HEADER_LEN: usize = 48;

/// Everything one run persists for the next.
#[derive(Debug, Serialize, Deserialize)]
pub struct CachePayload {
    /// The resolved `inline` flag the cached run checked under. Part of
    /// ADR-0032 Decision 3's set of inputs that change check behavior
    /// without changing the environment:
    /// crema.toml's `inline` is already folded into the invalidation key
    /// via the config file's content hash, but CLI `--inline` overrides
    /// the file without touching it — so the *resolved* value must be
    /// matched separately. A mismatch is treated like a key mismatch
    /// (full recheck).
    pub inline: bool,
    /// The previous run's fingerprint table (ADR-0032 Decision 3), one
    /// row per A-layer decl slot, sorted by [`record_sort_key`] so the
    /// serialized bytes are deterministic.
    pub fingerprints: Vec<FingerprintRecord>,
    /// One entry per type-checked file, sorted by path. A file with no
    /// entry (new file, previously syntax-broken file) is always
    /// rechecked; an entry with an empty `diagnostics` list replays as
    /// clean (tsc convention: absence of stored diagnostics = clean).
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FingerprintRecord {
    pub key: FingerprintKey,
    pub fingerprint: u64,
    /// Last path segment of the key's owning `TypeName`, captured at
    /// save time. Probe material for keys *removed* from the fresh env:
    /// their `TypeName` may no longer be interned there, so the fresh
    /// `NameTable` cannot recover the trailing constant symbol that
    /// negative-dependency matching needs (a file that failed to resolve
    /// constant `Foo` recorded only the bare symbol — see
    /// [`probe_hashes`]). `None` for `Member`/`Global` rows (their
    /// probe material is contained in the key itself).
    pub last_segment: Option<Symbol>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileEntry {
    /// Canonical absolute path as the check loop walked it, UTF-8 lossy.
    /// Compared by string equality; any drift (project moved, non-UTF-8
    /// path) misses the entry and safely rechecks.
    pub path: String,
    /// xxh3 of the source bytes. A mismatch (edited file) rechecks
    /// unconditionally — replayed byte ranges are only ever applied to
    /// bit-identical sources.
    pub content_hash: u64,
    /// Sorted, deduped name-component projection of the file's consulted
    /// keys (see [`project_consulted_entries`]).
    pub consulted: Vec<u64>,
    /// Check-phase diagnostics as produced by `check_source` — i.e.
    /// *before* `join_did_you_mean`. The suggestion column is computed
    /// against the fresh env at every emission, replayed or not (ADR-0032
    /// Decision 5a), so persisting it would be both redundant and stale.
    pub diagnostics: Vec<Diagnostic>,
}

/// Scheduler verdict for one previously-cached file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileDisposition {
    Recheck,
    Replay,
}

pub fn content_hash(source: &[u8]) -> u64 {
    xxh3_64(source)
}

pub fn encode_cache(key: &InvalidationKey, payload: &CachePayload) -> Vec<u8> {
    let body = bincode::serialize(payload).expect("CachePayload serialize is infallible");
    let mut bytes = Vec::with_capacity(HEADER_LEN + body.len());
    bytes.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&CACHE_SCHEMA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&[0u8; 4]);
    bytes.extend_from_slice(&key.0);
    bytes.extend_from_slice(&body);
    bytes
}

/// Atomically place the cache at `path` (tmp + rename, same recipe as
/// the G snapshot), creating `.crema/cache/` if needed.
pub fn write_cache(
    path: &Path,
    key: &InvalidationKey,
    payload: &CachePayload,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_atomic_bytes(path, &encode_cache(key, payload))
}

/// Read and validate the cache. `None` on any mismatch or corruption —
/// the caller falls back to a full recheck, never an error.
pub fn read_cache(path: &Path, key: &InvalidationKey) -> Option<CachePayload> {
    let bytes = std::fs::read(path).ok()?;
    decode_cache(&bytes, key)
}

pub fn decode_cache(bytes: &[u8], key: &InvalidationKey) -> Option<CachePayload> {
    if bytes.len() < HEADER_LEN {
        return None;
    }
    if bytes[0..8] != CACHE_MAGIC.to_le_bytes() {
        return None;
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    if version != CACHE_SCHEMA_VERSION {
        return None;
    }
    if bytes[16..48] != key.0 {
        return None;
    }
    bincode::deserialize(&bytes[HEADER_LEN..]).ok()
}

/// Project one file's consulted key set onto the persistent matching
/// space: a sorted, deduped set of content-addressed ids.
///
/// The projection drops the query family — `IsDeclaredClass(::Foo)` and
/// `MethodResolution { receiver: ::Foo, .. }` both project to `::Foo`'s
/// id. Cross-family over-matching is over-invalidation, the safe
/// direction ADR-0032 accepts throughout (Decision 4: project onto name
/// components).
pub fn project_consulted_entries(entries: &FxHashMap<ConsultedKey, bool>) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::with_capacity(entries.len() * 2);
    for key in entries.keys() {
        project_consulted_key(key, &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Name components contributed by one consulted key.
///
/// Two deliberate asymmetries:
///
/// - Method-family keys project only the receiver `TypeName`, not the
///   method `Symbol`. A method's addition/removal/sig change surfaces as
///   `Member(owner, m)` whose probe carries the owner's id, and
///   `expand_changed_set` already translates ancestor-side changes into
///   every descendant's own name — the name these keys record. Adding
///   the bare method symbol would instead invalidate every file calling
///   *any* same-named method anywhere (`each`, `new`, ...), destroying
///   selectivity for no soundness gain.
/// - Constant/global-family keys project their bare `Symbol` (plus every
///   scope `TypeName` they carry). Unlike the method family, a
///   constant-resolution key does *not* name the decl that would satisfy
///   it — a miss on `Foo` records only the symbol and the lexical
///   context, so a later `class Foo` (whose fingerprint key is a
///   `TypeName`) can only reach this file through the trailing-segment
///   symbol column (ADR-0032 Decision 4: constant queries are matched on
///   two columns, symbol name and context scope name).
///
/// `SyntheticConcernTargets` projects its method symbol: it is an index
/// query over member-name occurrences (ADR-0032 Decision 5), matched by
/// `Member(_, m)` probes carrying the bare member symbol.
fn project_consulted_key(key: &ConsultedKey, out: &mut Vec<u64>) {
    use ConsultedKey::*;
    match key {
        ExpandTypeAlias(n)
        | ConstantDeprecated(n)
        | ClassOrModuleDeprecated(n)
        | ClassAliasDeprecated(n)
        | DeclaredKind(n)
        | IsDeclaredClass(n)
        | IsDeclaredClassAlias(n)
        | IsDeclaredInterface(n)
        | IsDeclaredTypeAlias(n)
        | IsDeclaredConstant(n)
        | ClassOrModuleKind(n)
        | ClassTypeParams(n)
        | InterfaceMethodNames(n)
        | SuperclassTypeName(n)
        | HasCompleteAncestorChain(n)
        | InstanceAncestors(n)
        | SingletonAncestors(n)
        | OneInstanceAncestors(n) => {
            out.push(n.get());
        }
        GlobalLookup(s) | GlobalDeprecated(s) => out.push(s.raw_id()),
        MethodResolution { receiver: n, .. }
        | InterfaceMethodResolution { interface: n, .. }
        | SingletonMethodResolution { class: n, .. }
        | SuperInstanceMethod { class: n, .. }
        | SuperSingletonMethod { class: n, .. }
        | InstanceVariable { class: n, .. }
        | ClassVariable { class: n, .. }
        | ClassInstanceVariable { class: n, .. } => out.push(n.get()),
        SyntheticConcernTargets {
            method, current, ..
        } => {
            out.push(method.raw_id());
            if let Some(n) = current {
                out.push(n.get());
            }
        }
        // Matched by `FingerprintKey::ConcernTargets(concern)`'s probe
        // (`environment::fingerprint::hash_class_or_module` emits it), so
        // a target joining or leaving reaches this file and no other.
        ConcernBlockTargets { concern, .. } => out.push(concern_targets_probe_id(*concern)),

        ConstantResolution { name, context } => {
            out.push(name.raw_id());
            for scope in context.scopes() {
                out.push(scope.get());
            }
        }
        ConstantResolutionInNamespace { scope, name } => {
            out.push(name.raw_id());
            if let Some(n) = scope {
                out.push(n.get());
            }
        }
        ConstantResolutionChild { module, name } => {
            out.push(name.raw_id());
            out.push(module.get());
        }
    }
}

/// Serialize-ready rows for `table`, sorted for deterministic bytes.
/// `names` must be the `NameTable` of the env `table` was computed from
/// (every key's name is interned there, so `last_segment` is total).
pub fn fingerprint_records(table: &FingerprintTable, names: &NameTable) -> Vec<FingerprintRecord> {
    let mut out: Vec<FingerprintRecord> = table
        .iter()
        .map(|(key, fp)| FingerprintRecord {
            key: *key,
            fingerprint: *fp,
            last_segment: match key {
                FingerprintKey::Type(n)
                | FingerprintKey::ClassAlias(n)
                | FingerprintKey::TypeAlias(n)
                | FingerprintKey::Constant(n) => names.last_segment_if_interned(*n),
                FingerprintKey::Member(..)
                | FingerprintKey::Global(_)
                | FingerprintKey::ConcernTargets(_) => None,
            },
        })
        .collect();
    out.sort_unstable_by_key(|r| record_sort_key(&r.key));
    out
}

fn record_sort_key(key: &FingerprintKey) -> (u8, u64, u64) {
    match key {
        FingerprintKey::Type(n) => (0, n.get(), 0),
        FingerprintKey::Member(n, m) => (1, n.get(), m.raw_id()),
        FingerprintKey::ClassAlias(n) => (2, n.get(), 0),
        FingerprintKey::TypeAlias(n) => (3, n.get(), 0),
        FingerprintKey::Constant(n) => (4, n.get(), 0),
        FingerprintKey::Global(s) => (5, s.raw_id(), 0),
        FingerprintKey::ConcernTargets(n) => (6, n.get(), 0),
    }
}

/// Rebuild the diffable table plus the removed-key probe-material side
/// map from deserialized rows.
pub fn records_to_table(
    records: &[FingerprintRecord],
) -> (FingerprintTable, FxHashMap<FingerprintKey, Symbol>) {
    let mut table = FingerprintTable::default();
    let mut last_segments = FxHashMap::default();
    for r in records {
        table.insert(r.key, r.fingerprint);
        if let Some(seg) = r.last_segment {
            last_segments.insert(r.key, seg);
        }
    }
    (table, last_segments)
}

/// Lower the expanded changed key set into the probe hashes matched
/// against every file's `consulted` column.
///
/// Per changed key:
/// - `Type`/`ClassAlias`/`TypeAlias`/`Constant(n)` → `n`'s id plus the
///   trailing segment's symbol id (the constant-resolution column — a
///   file that only *missed* on the name recorded the bare symbol, not
///   the `TypeName`). The segment comes from the fresh `NameTable` when
///   the name is still interned, else from the cached side map (removed
///   decls).
/// - `Member(n, m)` → `n`'s id (method-family keys record the receiver)
///   plus `m`'s id (`SyntheticConcernTargets` matches by member-name
///   occurrence).
/// - `Global(s)` → `s`'s id.
pub fn probe_hashes(
    expanded: &FxHashSet<FingerprintKey>,
    names: &NameTable,
    cached_last_segments: &FxHashMap<FingerprintKey, Symbol>,
) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::with_capacity(expanded.len() * 2);
    for key in expanded {
        match key {
            FingerprintKey::Type(n)
            | FingerprintKey::ClassAlias(n)
            | FingerprintKey::TypeAlias(n)
            | FingerprintKey::Constant(n) => {
                out.push(n.get());
                let seg = names
                    .last_segment_if_interned(*n)
                    .or_else(|| cached_last_segments.get(key).copied());
                if let Some(seg) = seg {
                    out.push(seg.raw_id());
                }
            }
            FingerprintKey::Member(n, m) => {
                out.push(n.get());
                out.push(m.raw_id());
            }
            FingerprintKey::Global(s) => out.push(s.raw_id()),
            FingerprintKey::ConcernTargets(n) => out.push(concern_targets_probe_id(*n)),
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Classify one cached file against the current run. Content-hash
/// mismatch (edited file) rechecks before any probe — replay is only
/// ever valid against bit-identical sources. Otherwise the spike's
/// linear MVP: one `binary_search` into the file's sorted consulted set
/// per probe hash, short-circuiting on the first hit.
pub fn classify(entry: &FileEntry, current_hash: u64, probes: &[u64]) -> FileDisposition {
    if entry.content_hash != current_hash {
        return FileDisposition::Recheck;
    }
    for p in probes {
        if entry.consulted.binary_search(p).is_ok() {
            return FileDisposition::Recheck;
        }
    }
    FileDisposition::Replay
}

/// What the verify mode found when a replay-classified file's stored
/// entry disagreed with its shadow recheck (ADR-0032 Decision 6). Any
/// divergence is a consultation-log recording bug: replay is only ever
/// applied to bit-identical sources under the same invalidation key, so
/// a correct log makes the stored entry a pure function of inputs that
/// have not changed.
#[derive(Debug)]
pub struct VerifyDivergence {
    /// Stored pre-join diagnostics differ from the fresh check's
    /// (compared before `join_did_you_mean`, so suggestion recomputation
    /// can never register as a divergence).
    pub diagnostics_differ: bool,
    /// Queries the fresh check consulted whose projection is absent from
    /// the stored set — the recording gap itself, named. Sorted by debug
    /// rendering for deterministic reports.
    ///
    /// Detection operates in projection space, which is exactly the
    /// resolution replay soundness needs: invalidation probes match
    /// projected hashes, so a per-family recording loss whose hashes
    /// another recorded key still supplies (e.g. a dropped
    /// `MethodResolution { receiver: Foo, .. }` shadowed by a recorded
    /// `IsDeclaredClass(Foo)`) cannot cause a stale replay — the
    /// surviving hash still routes the file to recheck. What it *can*
    /// hide is the lost key's family identity; the report names whichever
    /// fresh keys contributed an unmatched hash, not every dropped call
    /// site.
    pub missing_keys: Vec<ConsultedKey>,
    /// Stored projection hashes no fresh consulted key produces. The
    /// key's identity is unrecoverable (the projection is one-way), so
    /// these are reported as bare hashes.
    pub stale_hashes: Vec<u64>,
}

/// Compare one replay-classified file's stored entry against its shadow
/// recheck. `None` means the entry verified clean. Diagnostics are
/// compared through their bincode encoding — the same bytes the cache
/// round-trips — which sidesteps a `PartialEq` derive across the whole
/// `DiagnosticKind` tree.
pub fn verify_file(
    stored: &FileEntry,
    fresh_diagnostics: &[Diagnostic],
    fresh_consulted: &FxHashMap<ConsultedKey, bool>,
) -> Option<VerifyDivergence> {
    let stored_bytes =
        bincode::serialize(&stored.diagnostics).expect("Diagnostic serialize is infallible");
    let fresh_bytes =
        bincode::serialize(fresh_diagnostics).expect("Diagnostic serialize is infallible");
    let diagnostics_differ = stored_bytes != fresh_bytes;

    let fresh_projection = project_consulted_entries(fresh_consulted);
    let mut missing_keys: Vec<ConsultedKey> = Vec::new();
    let mut scratch: Vec<u64> = Vec::new();
    for key in fresh_consulted.keys() {
        scratch.clear();
        project_consulted_key(key, &mut scratch);
        if scratch
            .iter()
            .any(|h| stored.consulted.binary_search(h).is_err())
        {
            missing_keys.push(key.clone());
        }
    }
    missing_keys.sort_unstable_by_key(|k| format!("{k:?}"));
    let stale_hashes: Vec<u64> = stored
        .consulted
        .iter()
        .copied()
        .filter(|h| fresh_projection.binary_search(h).is_err())
        .collect();

    if !diagnostics_differ && missing_keys.is_empty() && stale_hashes.is_empty() {
        return None;
    }
    Some(VerifyDivergence {
        diagnostics_differ,
        missing_keys,
        stale_hashes,
    })
}

/// One check run's scheduler state: the loaded (or absent) previous
/// generation plus everything precomputed for per-file classification.
/// Built once between env construction and the check loop; consumed by
/// [`Scheduler::into_payload`] after the loop to produce the next
/// generation.
pub struct Scheduler {
    key: InvalidationKey,
    inline: bool,
    /// Previous generation's per-file entries, drained as files replay.
    /// Empty when the cache was missing or invalid.
    entries: FxHashMap<String, FileEntry>,
    probes: Vec<u64>,
    cache_valid: bool,
    diff_empty: bool,
    old_file_count: usize,
    fresh_records: Vec<FingerprintRecord>,
}

/// The previous generation as read off disk — stage 1 of the scheduler,
/// built before the environment exists. It can answer the content-hash
/// half of [`classify`] on its own ("did this file's bytes change?"),
/// which is what lets `run_check` release the prism AST of every
/// unchanged file ahead of env construction (the footprint peak) instead
/// of after it. The dependency half (probe hits) needs the fresh
/// fingerprint table and so waits for [`CachedGeneration::prepare`].
///
/// A file this stage calls unchanged may still be rechecked once the
/// probes are known; the caller re-parses it then. A file it calls
/// changed is never replayed, so its AST must be kept.
pub struct CachedGeneration {
    key: InvalidationKey,
    inline: bool,
    /// `None` when the cache was missing or invalid (a cold run).
    previous: Option<PreviousGeneration>,
}

struct PreviousGeneration {
    entries: FxHashMap<String, FileEntry>,
    fingerprints: Vec<FingerprintRecord>,
    old_file_count: usize,
}

impl CachedGeneration {
    /// Read + validate the cache. Needs nothing but the key, so it can
    /// run right after parsing, before the draft is frozen.
    pub fn load(cache_path: &Path, key: InvalidationKey, inline: bool) -> CachedGeneration {
        let previous = read_cache(cache_path, &key)
            .filter(|p| p.inline == inline)
            .map(|payload| {
                let old_file_count = payload.files.len();
                let entries: FxHashMap<String, FileEntry> = payload
                    .files
                    .into_iter()
                    .map(|f| (f.path.clone(), f))
                    .collect();
                PreviousGeneration {
                    entries,
                    fingerprints: payload.fingerprints,
                    old_file_count,
                }
            });
        CachedGeneration {
            key,
            inline,
            previous,
        }
    }

    /// The content-hash half of [`classify`]: `true` when the previous
    /// generation has this file with the same bytes. Only a necessary
    /// condition for replay — a probe hit in [`Scheduler::dispose`] can
    /// still turn it into a recheck.
    pub fn content_unchanged(&self, path: &str, current_hash: u64) -> bool {
        self.previous
            .as_ref()
            .and_then(|prev| prev.entries.get(path))
            .is_some_and(|entry| entry.content_hash == current_hash)
    }

    /// Stage 2: compute the fresh fingerprint table, diff, expand, and
    /// lower to probe hashes. The `AncestorGraph` (O(env) walk) is only
    /// built when the diff is non-empty — the common no-change warm run
    /// skips it entirely, keeping the fixed cost to cache read +
    /// fingerprint pass (the budget of ADR-0032's third verification
    /// condition).
    pub fn prepare(self, env: &Arc<Environment>) -> Scheduler {
        let fresh_table = compute_fingerprint_table(env);
        let names = env.names();
        let (cache_valid, entries, probes, diff_empty, old_file_count) = match self.previous {
            None => (false, FxHashMap::default(), Vec::new(), false, 0),
            Some(prev) => {
                let (old_table, last_segments) = records_to_table(&prev.fingerprints);
                let raw = RawDiff::compute(&old_table, &fresh_table);
                let diff_empty = raw.is_empty();
                let probes = if diff_empty {
                    Vec::new()
                } else {
                    let graph = AncestorGraph::new(Arc::clone(env));
                    let expanded = expand_changed_set(&graph, &raw);
                    probe_hashes(&expanded, names, &last_segments)
                };
                (true, prev.entries, probes, diff_empty, prev.old_file_count)
            }
        };
        let fresh_records = fingerprint_records(&fresh_table, names);
        Scheduler {
            key: self.key,
            inline: self.inline,
            entries,
            probes,
            cache_valid,
            diff_empty,
            old_file_count,
            fresh_records,
        }
    }
}

impl Scheduler {
    /// Both stages at once, for callers that have the environment in
    /// hand already (tests); `run_check` runs them separately so the
    /// AST release can sit between them.
    pub fn prepare(
        cache_path: &Path,
        key: InvalidationKey,
        inline: bool,
        env: &Arc<Environment>,
    ) -> Scheduler {
        CachedGeneration::load(cache_path, key, inline).prepare(env)
    }

    pub fn key(&self) -> &InvalidationKey {
        &self.key
    }

    pub fn cache_valid(&self) -> bool {
        self.cache_valid
    }

    /// Decide one file: `Some(entry)` means replay (the cached entry is
    /// handed back, drained from the scheduler, for the caller to emit
    /// and to carry into the next generation); `None` means recheck (no
    /// entry, edited content, or a probe hit).
    pub fn dispose(&mut self, path: &str, current_hash: u64) -> Option<FileEntry> {
        if !self.cache_valid {
            return None;
        }
        let entry = self.entries.get(path)?;
        match classify(entry, current_hash, &self.probes) {
            FileDisposition::Recheck => None,
            FileDisposition::Replay => self.entries.remove(path),
        }
    }

    /// Assemble the next cache generation, or `None` when this run
    /// changed nothing — same valid cache, empty fingerprint diff, zero
    /// rechecks, same file set — and the bytes on disk are already
    /// identical to what would be written.
    pub fn into_payload(
        self,
        mut files: Vec<FileEntry>,
        rechecked: usize,
    ) -> Option<(InvalidationKey, CachePayload)> {
        let dirty = !self.cache_valid
            || !self.diff_empty
            || rechecked > 0
            || files.len() != self.old_file_count;
        if !dirty {
            return None;
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Some((
            self.key,
            CachePayload {
                inline: self.inline,
                fingerprints: self.fresh_records,
                files,
            },
        ))
    }
}
