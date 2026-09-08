//! Shared wrapper (barista slice S1, `lowering_maps_clone_debt`) around
//! the two collections `AncestorBuilder` carries for lowering
//! (`all_names` membership, `class_aliases` lookup). Held behind an `Arc`
//! so `AncestorBuilder::update` can share them with the builder it
//! derives instead of deep-cloning both collections per update.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::name::Name;

/// `AncestorBuilder`'s `all_names` membership set and `class_aliases`
/// lookup map, kept together so callers share one `Arc`.
#[derive(Debug, Clone, Default)]
pub struct LoweringMaps {
    all_names: FxHashSet<Name>,
    class_aliases: FxHashMap<Name, Name>,
}

impl LoweringMaps {
    pub fn from_owned_maps(
        all_names: FxHashSet<Name>,
        class_aliases: FxHashMap<Name, Name>,
    ) -> Self {
        LoweringMaps {
            all_names,
            class_aliases,
        }
    }

    pub fn contains(&self, name: Name) -> bool {
        self.all_names.contains(&name)
    }

    pub fn alias(&self, name: Name) -> Option<Name> {
        self.class_aliases.get(&name).copied()
    }

    pub fn insert_name(&mut self, name: Name) {
        self.all_names.insert(name);
    }

    pub fn remove_name(&mut self, name: Name) {
        self.all_names.remove(&name);
    }

    pub fn insert_alias(&mut self, key: Name, target: Name) {
        self.class_aliases.insert(key, target);
    }

    pub fn remove_alias(&mut self, key: Name) {
        self.class_aliases.remove(&key);
    }

    pub fn iter_all_names(&self) -> impl Iterator<Item = Name> + '_ {
        self.all_names.iter().copied()
    }

    pub fn iter_class_aliases(&self) -> impl Iterator<Item = (Name, Name)> + '_ {
        self.class_aliases.iter().map(|(&k, &v)| (k, v))
    }
}

