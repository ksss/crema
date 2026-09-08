//! Name resolution helper — canonicalizes a relative type name against
//! the set of declared names plus the nesting context, with class-alias
//! chains normalized along the way.
//!
//! Mirrors `RBS::Resolver::TypeNameResolver#resolve_namespace0` in rbs.
//! Moved here in Phase 2 from `crate::name_resolver`, which now retains
//! only the string-building `qualified_name` and the `ResolveContext`
//! type alias that the rbs_loader declaration walk relies on.

use crate::definition::lowering_maps::LoweringMaps;
use crate::name::{Name, NameTable};


/// Resolve a potentially relative type name to an absolute [`Name`].
///
/// When a resolved segment turns out to be a `class X = Y` alias, the
/// alias chain is followed before appending the next segment. This
/// mirrors rbs `Resolver::TypeNameResolver#resolve_namespace0`:
/// otherwise `Queue::Foo` (with `class Queue = Thread::Queue`) would
/// look up `::Queue::Foo` and miss the real class. The same head/tail
/// walk runs for both relative and absolute (`::Q::Inner`) names, so
/// `class B = ::Q::Inner` with `class Q = Foo` normalizes to
/// `::Foo::Inner` rather than the literal `::Q::Inner`.
pub(crate) fn resolve_relative_type_name(
    relative_name: &str,
    context: &[Option<Name>],
    maps: &LoweringMaps,
    names: &NameTable,
) -> Name {
    if let Some(stripped) = relative_name.strip_prefix("::") {
        let parts: Vec<&str> = stripped.split("::").collect();
        let head = parts[0];
        let tail = &parts[1..];
        let head_name = names.intern(&format!("::{}", head));
        let head_resolved = normalize_through_aliases(head_name, maps);
        return walk_tail(head_resolved, tail, maps, names)
            .unwrap_or_else(|| names.intern(relative_name));
    }

    let parts: Vec<&str> = relative_name.split("::").collect();
    let head = parts[0];
    let tail = &parts[1..];

    if let Some(resolved) = resolve_head_in_context(head, context, maps, names) {
        let head_resolved = normalize_through_aliases(resolved, maps);
        walk_tail(head_resolved, tail, maps, names)
            .unwrap_or_else(|| names.intern(&format!("::{}", relative_name)))
    } else {
        names.intern(&format!("::{}", relative_name))
    }
}

/// Walk tail segments under an already-resolved head, normalizing each
/// step through the alias map. Returns `None` when a segment is missing
/// from `maps` so the caller can apply its own literal fallback.
fn walk_tail(
    mut current: Name,
    tail: &[&str],
    maps: &LoweringMaps,
    names: &NameTable,
) -> Option<Name> {
    for &part in tail {
        let candidate_str = format!("{}::{}", names.resolve(current), part);
        let candidate = names.intern(&candidate_str);
        if maps.contains(candidate) {
            current = normalize_through_aliases(candidate, maps);
        } else {
            return None;
        }
    }
    Some(current)
}

/// Follow the `class X = Y` alias chain with cycle detection.
///
/// Kept local to the resolver so callers don't need to thread the whole
/// `DefinitionBuilder` through. `DefinitionBuilder::normalize_module_name` uses the
/// same logic against `self.class_alias_decls`.
fn normalize_through_aliases(name: Name, maps: &LoweringMaps) -> Name {
    let mut current = name;
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(old_name) = maps.alias(current) {
        if !visited.insert(current) {
            return name;
        }
        current = old_name;
    }
    current
}

fn resolve_head_in_context(
    head: &str,
    context: &[Option<Name>],
    maps: &LoweringMaps,
    names: &NameTable,
) -> Option<Name> {
    for entry in context.iter().rev() {
        match entry {
            Some(ns_name) => {
                let candidate_str = format!("{}::{}", names.resolve(*ns_name), head);
                let candidate = names.intern(&candidate_str);
                if maps.contains(candidate) {
                    return Some(candidate);
                }
            }
            None => continue,
        }
    }

    let toplevel_str = format!("::{}", head);
    let toplevel = names.intern(&toplevel_str);
    if maps.contains(toplevel) {
        Some(toplevel)
    } else {
        None
    }
}
