//! Owned wrapper around the file -> contributed-declaration reverse
//! index [`crate::environment::frozen::Environment`] carries, so the
//! draft/frozen pair and `unload` share one vocabulary for it
//! (`add`, `remove_take`, `files`) instead of open-coding map
//! operations at every site.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::environment::draft::PathIndexKey;
use crate::name::Name;

/// `Environment::path_index` (file -> contributed [`PathIndexKey`]s).
#[derive(Clone, Debug, Default)]
pub(crate) struct PathIndexState {
    map: FxHashMap<Name, FxHashSet<PathIndexKey>>,
}

impl PathIndexState {
    pub(crate) fn get(&self, file: Name) -> Option<FxHashSet<PathIndexKey>> {
        self.map.get(&file).cloned()
    }

    /// Record one key contributed by `file` (the draft-build write path,
    /// `record_path_index`).
    pub(crate) fn add(&mut self, file: Name, key: PathIndexKey) {
        self.map.entry(file).or_default().insert(key);
    }

    #[cfg(test)]
    pub(crate) fn contains_key(&self, file: Name) -> bool {
        self.map.contains_key(&file)
    }

    /// Remove `file`, returning the key set it held (the `unload` drain
    /// wants the affected keys). `None` when `file` was absent.
    pub(crate) fn remove_take(&mut self, file: Name) -> Option<FxHashSet<PathIndexKey>> {
        self.map.remove(&file)
    }

    /// Every file currently holding at least one key.
    pub(crate) fn files(&self) -> impl Iterator<Item = Name> + '_ {
        self.map.keys().copied()
    }

    /// The whole index as a plain map, for differential tests that
    /// compare a delta-updated environment against a fresh build.
    #[cfg(test)]
    pub(crate) fn materialize(&self) -> FxHashMap<Name, FxHashSet<PathIndexKey>> {
        self.map.clone()
    }
}
