//! G-snapshot disk write path: 64-byte header + flat payload, placed
//! atomically via `<path>.tmp` + rename (ADR-0028 Decision 5).

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::definition_builder::{self, RoastedGScan};
use crate::environment::frozen::Environment;
use crate::infusion_collector::active_record_synthesis;
use crate::location::{DuplicateSource, SourceLocation};
use crate::name::NameTable;
use crate::snapshot::freshness::GFreshnessManifest;
use crate::snapshot::invalidation::InvalidationKey;
use crate::snapshot::mirror::{
    MBakedAliasCycle, MBakedAncestorCycle, MBakedArityGroup, MBakedArityViolation,
    MBakedDiagnostics, MBakedLoc, MBakedMethodGroup, MBakedUnresolvedSuperEdge, MBakedVariableDup,
    MBakedVariableGroup, MEntry,
};
use crate::snapshot::{convert, flat};
use crate::validator;

/// Identifies a crema G-snapshot file ("C5EA 05A9" ≈ crema snap).
pub const G_SNAPSHOT_MAGIC: u64 = 0xC5EA_05A9_0000_0001;
/// v8: header grows two words (`FRESH_OFF_OFFSET`/`FRESH_LEN_OFFSET`)
/// locating a new freshness-manifest section — (path, mtime, size) for
/// every G-layer `.rbs` file the cold build loaded, checked on every warm
/// attempt so direct edits to the G layer are detected even though the
/// invalidation key never changes for them
/// (`g_snapshot_g_layer_freshness_check`). v7: baked diagnostics gain
/// `warnings` — the G-construction stderr
/// lines (stale pin, missing library, load failure, ...) a cold run
/// would have printed, baked in so a warm hit can replay them
/// (`gem_dirs_cache_removal_warning_replay` — the gem-dirs disk cache
/// this replaced could no longer keep surfacing them itself).
/// v6: `super_edges` and `unresolved_super_edges` bundle into one
/// `MSuperEdges` field. The layout on disk (bincode) changes because
/// two Vec<...> collapse into a struct that still serializes them, so
/// old readers reject the payload. v5: baked diagnostics gain
/// `unresolved_super_edges` so warm can bridge gem supers into A-only
/// names (`high_infusion_ar_a_layer_super_bridging`). v4: infusion's
/// G super-edges join the baked
/// diagnostics section (ADR-0028 slice 2b-4), changing its bincode
/// layout. v3 added validator findings (ancestor cycles, mixin arity,
/// slice 2b-2); v2 appended the diagnostics section after the flat
/// payload, located via the header's diag offset/len words. Older
/// versions fail the check and fall back to a cold rebuild.
pub const G_SNAPSHOT_SCHEMA_VERSION: u32 = 8;
pub const HEADER_LEN: usize = 80;
/// Absolute file offset of the baked-diagnostics section, stored in the
/// header words that were reserved in v1.
pub const DIAG_OFF_OFFSET: usize = 48;
pub const DIAG_LEN_OFFSET: usize = 56;
/// Absolute file offset of the freshness-manifest section (v8).
pub const FRESH_OFF_OFFSET: usize = 64;
pub const FRESH_LEN_OFFSET: usize = 72;

/// Layout: magic u64 LE | schema_version u32 LE | reserved u32 = 0 |
/// invalidation key 32 bytes | diag section offset u64 LE | diag
/// section len u64 LE | freshness-manifest section offset u64 LE |
/// freshness-manifest section len u64 LE.
pub fn build_header(key: &InvalidationKey) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..8].copy_from_slice(&G_SNAPSHOT_MAGIC.to_le_bytes());
    h[8..12].copy_from_slice(&G_SNAPSHOT_SCHEMA_VERSION.to_le_bytes());
    h[16..48].copy_from_slice(&key.0);
    h
}

/// Serialize the frozen gem environment and place it at `cache_path`.
/// Payload recipe: mirror-convert, bincode each entry individually, and
/// lay the global tables + entry index out with `flat::Writer` — the
/// same shape the spike's cmd_build produced (read side is slice 1c).
/// `warnings` is the G-construction stderr lines the cold run that
/// produced `env` already printed — baked in so a later warm hit can
/// replay them (see `G_SNAPSHOT_SCHEMA_VERSION`'s v7 note). `manifest` is
/// the G-layer freshness manifest (`G_SNAPSHOT_SCHEMA_VERSION`'s v8 note).
pub fn write_g_snapshot(
    cache_path: &Path,
    key: &InvalidationKey,
    env: &Arc<Environment>,
    warnings: &[String],
    manifest: &GFreshnessManifest,
) -> io::Result<()> {
    let bytes = encode_g_snapshot(key, env, warnings, manifest);
    write_atomic_bytes(cache_path, &bytes)
}

/// Serialize the frozen gem environment into the full snapshot file image
/// (header + flat payload). Split from [`write_g_snapshot`] so the cold
/// path can re-open the exact bytes it just wrote through the lazy
/// snapshot backend without a disk round-trip (ADR-0028 slice 2a-2).
pub fn encode_g_snapshot(
    key: &InvalidationKey,
    env: &Arc<Environment>,
    warnings: &[String],
    manifest: &GFreshnessManifest,
) -> Vec<u8> {
    let mut baked = definition_builder::roast_g_scan(env);
    // The validator roast needs an `AncestorBuilder`, which shares the
    // env by `Arc` — hence this function's `&Arc<Environment>` parameter.
    let (ancestor_cycles, arity) = validator::roast_g_validators(env);
    baked.ancestor_cycles = ancestor_cycles;
    baked.arity = arity;
    // Infusion's AR model detection reads these instead of decoding every
    // gem class at warm time (ADR-0028 slice 2b-4). Cold's G-only
    // resolver bakes edges that resolve here; edges that reference an
    // A-only name fall into `SuperEdges::unresolved` so the warm splice
    // can rerun them against the merged A ∪ G resolver.
    baked.super_edges = active_record_synthesis::roast_g_super_edges(env);
    baked.warnings = warnings.to_vec();
    let diag_blob = bincode::serialize(&baked_to_mirror(&baked, env.names()))
        .expect("MBakedDiagnostics serialize is infallible");
    let manifest_blob =
        bincode::serialize(manifest).expect("GFreshnessManifest serialize is infallible");
    let (snap, _stats) = convert::convert(env);

    let mut entries: Vec<(u64, Vec<u8>)> = Vec::new();
    let encode = |e: &MEntry| bincode::serialize(e).expect("MEntry serialize is infallible");
    for (id, e) in snap.class_decls {
        entries.push((id, encode(&MEntry::ClassOrModule(e))));
    }
    for (id, e) in snap.interface_decls {
        entries.push((id, encode(&MEntry::Interface(e))));
    }
    for (id, e) in snap.class_alias_decls {
        entries.push((id, encode(&MEntry::ClassAlias(e))));
    }
    for (id, e) in snap.type_alias_decls {
        entries.push((id, encode(&MEntry::TypeAlias(e))));
    }
    for (id, e) in snap.constant_decls {
        entries.push((id, encode(&MEntry::Constant(e))));
    }
    for (id, e) in snap.global_decls {
        entries.push((id, encode(&MEntry::Global(e))));
    }

    let mut writer = flat::Writer::new();
    writer.write_names(&snap.names);
    writer.write_symbols(&snap.symbols);
    writer.write_type_names(&snap.type_names);
    writer.write_normalized(&snap.normalized);
    writer.write_entries(&entries);
    let payload = writer.finish();

    let mut bytes =
        Vec::with_capacity(HEADER_LEN + payload.len() + manifest_blob.len() + diag_blob.len());
    bytes.extend_from_slice(&build_header(key));
    bytes.extend_from_slice(&payload);
    // Freshness manifest before diagnostics so the diag section keeps
    // spanning the file tail (existing byte-format pin in
    // `write_g_snapshot_roundtrip`).
    let fresh_off = bytes.len() as u64;
    bytes.extend_from_slice(&manifest_blob);
    bytes[FRESH_OFF_OFFSET..FRESH_OFF_OFFSET + 8].copy_from_slice(&fresh_off.to_le_bytes());
    bytes[FRESH_LEN_OFFSET..FRESH_LEN_OFFSET + 8]
        .copy_from_slice(&(manifest_blob.len() as u64).to_le_bytes());
    let diag_off = bytes.len() as u64;
    bytes.extend_from_slice(&diag_blob);
    bytes[DIAG_OFF_OFFSET..DIAG_OFF_OFFSET + 8].copy_from_slice(&diag_off.to_le_bytes());
    bytes[DIAG_LEN_OFFSET..DIAG_LEN_OFFSET + 8]
        .copy_from_slice(&(diag_blob.len() as u64).to_le_bytes());
    bytes
}

pub(crate) fn baked_loc(l: &SourceLocation) -> MBakedLoc {
    MBakedLoc {
        file: l.file.clone(),
        range: convert::range(l.range),
    }
}

fn baked_to_mirror(baked: &RoastedGScan, names: &NameTable) -> MBakedDiagnostics {
    let method_group = |(owner, dups, cycles): &(
        crate::type_name::TypeName,
        definition_builder::MethodDups,
        definition_builder::AliasCycles,
    )| {
        MBakedMethodGroup {
        owner: owner.get(),
        dups: dups
            .iter()
            .map(|(name, dup_loc, original_loc)| {
                (
                    name.clone(),
                    dup_loc.as_ref().map(baked_loc),
                    original_loc.as_ref().map(|src| match src {
                        DuplicateSource::Location(loc) => baked_loc(loc),
                        DuplicateSource::Synthesized { .. } => unreachable!(
                            "G-snapshot dup scan only ever sees real gem files — infusion never runs on gem RBS"
                        ),
                    }),
                )
            })
            .collect(),
        cycles: cycles
            .iter()
            .map(|c| MBakedAliasCycle {
                type_name: c.type_name.clone(),
                alias_names: c.alias_names.clone(),
                primary_location: c.primary_location.as_ref().map(baked_loc),
            })
            .collect(),
    }
    };
    let arity_kind = |kind: &'static str| -> u8 {
        match kind {
            "superclass" => 0,
            "include" => 1,
            "extend" => 2,
            "prepend" => 3,
            other => unreachable!("unknown arity kind {other}"),
        }
    };
    MBakedDiagnostics {
        classes: baked.classes.iter().map(method_group).collect(),
        interfaces: baked.interfaces.iter().map(method_group).collect(),
        ancestor_cycles: baked
            .ancestor_cycles
            .iter()
            .map(|c| MBakedAncestorCycle {
                participants: c.participants.iter().map(|p| p.get()).collect(),
                type_name: c.type_name.clone(),
                chain: c.chain.clone(),
                primary_source: c.primary_source.as_ref().map(baked_loc),
            })
            .collect(),
        arity: baked
            .arity
            .iter()
            .map(|(owner, violations)| MBakedArityGroup {
                owner: owner.get(),
                violations: violations
                    .iter()
                    .map(|v| MBakedArityViolation {
                        kind: arity_kind(v.kind),
                        target: v.target.clone(),
                        class: v.class.clone(),
                        expected: v.expected.clone(),
                        got: v.got as u64,
                        location: v.location.as_ref().map(baked_loc),
                    })
                    .collect(),
            })
            .collect(),
        variables: baked
            .variables
            .iter()
            .map(|(owner, dups)| MBakedVariableGroup {
                owner: owner.get(),
                dups: dups
                    .iter()
                    .map(|d| MBakedVariableDup {
                        kind: match d.kind {
                            crate::definition::VariableDuplicationKind::Instance => 0,
                            crate::definition::VariableDuplicationKind::ClassInstance => 1,
                        },
                        variable_name: names.resolve(d.variable_name).to_string(),
                        location: d.location.as_ref().map(baked_loc),
                        ruby_source_location: d
                            .ruby_source_location
                            .as_ref()
                            .map(|r| (names.resolve(r.file).to_string(), r.start_byte, r.end_byte)),
                    })
                    .collect(),
            })
            .collect(),
        super_edges: crate::snapshot::mirror::MSuperEdges {
            resolved: baked
                .super_edges
                .resolved
                .iter()
                .map(|(class, super_name)| (class.get(), super_name.get()))
                .collect(),
            unresolved: baked
                .super_edges
                .unresolved
                .iter()
                .map(|edge| MBakedUnresolvedSuperEdge {
                    class: edge.class.get(),
                    raw_super: edge.raw_super.get(),
                    context: edge.context.iter().map(|c| c.get()).collect(),
                })
                .collect(),
        },
        warnings: baked.warnings.clone(),
    }
}

/// Atomically place a pre-encoded snapshot image at `cache_path`.
///
/// Thin re-export around [`crate::atomic_write::write_atomic_bytes`],
/// kept as a `pub fn` so the existing external call site in `main.rs`
/// (`crema::snapshot::write::write_atomic_bytes`) does not have to
/// learn the new module path in the same PR that dedups the impl.
pub fn write_atomic_bytes(cache_path: &Path, bytes: &[u8]) -> io::Result<()> {
    crate::atomic_write::write_atomic_bytes(cache_path, bytes)
}
