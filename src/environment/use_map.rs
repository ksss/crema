//! `use` directive resolution map.
//!
//! Port of `RBS::Environment::UseMap`
//! (`rbs/lib/rbs/environment/use_map.rb`). Two pieces:
//!
//! - [`Table`] — file-crossing precomputed index of every declared
//!   [`TypeName`], grouped by parent namespace (itself a [`TypeName`]
//!   under the folded representation). Required so that wildcard
//!   clauses (`use Foo::*`) can enumerate the children without
//!   re-scanning the whole environment per file.
//! - [`UseMap`] — per-file alias table built up by
//!   [`build_map`](UseMap::build_map) over the file's
//!   [`UseClause`]s. [`resolve`](UseMap::resolve) is the same
//!   two-arm logic as rbs:
//!   - bare short name `T` → look up `@map[T]`
//!   - prefixed short name `A::B::C` → rewrite the leading `A`
//!     through `@map`, then keep the trailing `B::C` segments
//!
//! Absolute [`TypeName`]s are never rewritten (rbs's `resolve?`
//! returns `nil` for them).

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::ast::directives::UseClause;
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;

/// File-crossing index of every declared [`TypeName`].
///
/// `known_types` is the set of *real* declared names (one per
/// class / module / interface / type-alias / constant decl, mirroring
/// rbs's `class_decls.keys` + alias decls + type alias decls +
/// interface decls). [`compute_children`](Table::compute_children)
/// inverts the set into `children`, an index from parent namespace
/// (a [`TypeName`]) to the children declared under it — the data
/// shape `UseWildcardClause` needs.
#[derive(Debug, Default, Clone)]
pub struct Table {
    pub known_types: FxHashSet<TypeName>,
    pub children: FxHashMap<TypeName, FxHashSet<TypeName>>,
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    /// Populate `children` from `known_types`. Mirrors rbs
    /// `UseMap::Table#compute_children`.
    pub fn compute_children(&mut self, names: &NameTable) {
        self.children.clear();
        for tn in &self.known_types {
            let Some(parent) = names.type_name_parent(*tn) else {
                continue;
            };
            if names.type_name_is_root(parent) {
                continue;
            }
            self.children.entry(parent).or_default().insert(*tn);
        }
    }
}

/// Per-file alias table backed by a shared [`Table`].
///
/// `Arc<Table>` because every file in an environment shares the same
/// pre-computed children index; the per-file state is just the
/// `Symbol → TypeName` mapping built by [`build_map`](Self::build_map).
#[derive(Debug, Clone)]
pub struct UseMap {
    map: FxHashMap<Symbol, TypeName>,
    table: Arc<Table>,
}

impl UseMap {
    pub fn new(table: Arc<Table>) -> Self {
        Self {
            map: FxHashMap::default(),
            table,
        }
    }

    /// Incorporate one clause into the map. Mirrors rbs
    /// `UseMap#build_map`.
    pub fn build_map(&mut self, clause: &UseClause, names: &NameTable) {
        match clause {
            UseClause::Single(sc) => {
                let absolute = names.to_absolute(sc.type_name);
                let Some(last) = names.last_segment(sc.type_name) else {
                    return;
                };
                let alias_key = sc.new_name.unwrap_or(last);
                self.map.insert(alias_key, absolute);
            }
            UseClause::Wildcard(wc) => {
                let abs_ns = names.to_absolute(wc.namespace);
                if let Some(children) = self.table.children.get(&abs_ns) {
                    for child in children {
                        if let Some(last) = names.last_segment(*child) {
                            self.map.insert(last, *child);
                        }
                    }
                }
            }
        }
    }

    /// Try to rewrite `type_name` via this file's use map.
    ///
    /// Returns `None` when `type_name` is already absolute (rbs's
    /// `resolve?` short-circuits on `absolute?`) or when neither the
    /// bare name nor the leading segment of a relative path is in
    /// the map.
    pub fn resolve(&self, type_name: TypeName, names: &NameTable) -> Option<TypeName> {
        if names.type_name_is_absolute(type_name) {
            return None;
        }

        let segments = names.type_name_segments(type_name);
        let (head, rest) = segments.split_first()?;
        if rest.is_empty() {
            // bare short name `T` — look up `@map[T]`.
            return self.map.get(head).copied();
        }

        // Prefixed short name `A::B::C` — rewrite the leading `A`
        // via the map, then re-attach the trailing segments.
        let head_resolved = *self.map.get(head)?;
        Some(names.extend_type_name(head_resolved, rest.iter().copied()))
    }

    /// `resolve(type_name).unwrap_or(type_name)`.
    /// Mirrors rbs `UseMap#resolve` (the non-`?` version that always
    /// returns a `TypeName`).
    pub fn resolve_or_self(&self, type_name: TypeName, names: &NameTable) -> TypeName {
        self.resolve(type_name, names).unwrap_or(type_name)
    }
}

