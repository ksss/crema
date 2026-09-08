//! Append-only map that hands out references outliving the borrow of the
//! map itself — the cache primitive behind the lazy snapshot backend
//! (ADR-0028 slice 2a-2).

use std::cell::RefCell;
use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxHashMap;

/// Insert-only hash map with interior mutability.
///
/// `get` / `get_or_try_insert_with` return `&V` tied to `&self` (not to a
/// `RefCell` guard), so a caller holding a decoded snapshot entry can keep
/// using it while later probes insert more entries.
///
/// # Safety invariants
///
/// The lifetime extension below is sound because:
///
/// - values live behind `Arc`, so growing the underlying `FxHashMap`
///   never moves the pointed-to `V`. `Arc` (not `Box`) on purpose: the
///   handed-out `&V` points into the `Arc`'s own heap allocation, which
///   only ever sees shared access — rehash moves and map mutation touch
///   the map's buffer, a different allocation, so Stacked Borrows keeps
///   the raw pointer valid (a `Box` would assert uniqueness on every
///   move and invalidate it);
/// - entries are never removed and never overwritten (first value wins
///   on reentrant races), so the `Arc` refcount never reaches zero while
///   the map lives;
/// - the type is `!Sync` (`RefCell`), matching crema's single-threaded
///   checker (see `Cargo.toml` `arc_with_non_send_sync` note).
pub struct AppendMap<K, V> {
    inner: RefCell<FxHashMap<K, Arc<V>>>,
}

impl<K: Eq + Hash + Copy, V> Default for AppendMap<K, V> {
    fn default() -> Self {
        AppendMap {
            inner: RefCell::new(FxHashMap::default()),
        }
    }
}

impl<K: Eq + Hash + Copy, V> AppendMap<K, V> {
    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let map = self.inner.borrow();
        map.get(key).map(|arc| unsafe { &*Arc::as_ptr(arc) })
    }

    /// Insert `value` unless the key is already present (first value
    /// wins, keeping previously handed-out references valid).
    pub fn insert_first(&self, key: K, value: V) {
        self.inner
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| Arc::new(value));
    }

    /// Return the cached value for `key`, computing and inserting it when
    /// absent. `compute` runs with no `RefCell` borrow held, so it may
    /// reentrantly probe this map (nested snapshot entries decode their
    /// children through the same cache).
    pub fn get_or_try_insert_with<E>(
        &self,
        key: K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<&V, E> {
        if let Some(v) = self.get(&key) {
            return Ok(v);
        }
        let value = compute()?;
        self.insert_first(key, value);
        Ok(self.get(&key).expect("inserted just above, never removed"))
    }
}
