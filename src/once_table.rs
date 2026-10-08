//! Insert-only tables whose reads take no lock (ADR-0034 Decision 2).
//!
//! [`OnceTable`] is an open-addressing hash table keyed by an already
//! well-mixed `u64` (an xxh3 content address). Every slot is a
//! `OnceLock`, so a filled slot is read with one atomic load and a `&V`
//! handed out stays valid while other threads keep inserting. Growth
//! never moves entries: a full segment is left in place and a larger one
//! is chained after it. [`OnceTable::compact`] (needs `&mut`) folds the
//! chain back into one segment.
//!
//! Adding a key that is not there yet takes the table's `grow` lock and
//! looks the key up again under it, so two threads interning the same
//! new key agree on one slot. Hits never touch the lock.
//!
//! [`OnceStore`] is the positional counterpart: an append-only vector
//! whose filled slots are read the same way.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};


const BASE_CAP: usize = 64;
const SEGS: usize = 16;

struct Seg<V> {
    slots: Box<[OnceLock<(u64, V)>]>,
    len: AtomicUsize,
}

impl<V> Seg<V> {
    fn new(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Seg {
            slots: (0..cap).map(|_| OnceLock::new()).collect(),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn find(&self, key: u64) -> Option<&V> {
        let mask = self.slots.len() - 1;
        let mut i = key as usize & mask;
        loop {
            match self.slots[i].get() {
                None => return None,
                Some((k, v)) if *k == key => return Some(v),
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    /// Callers serialize `put` (the table's `grow` lock, or `&mut`), so
    /// the first empty slot of the probe sequence is free to take.
    fn put(&self, key: u64, value: V) -> &V {
        // A segment without room may have no empty slot left, and the
        // probe below would then spin forever.
        assert!(self.has_room(), "put into a full segment");
        let mask = self.slots.len() - 1;
        let mut i = key as usize & mask;
        while self.slots[i].get().is_some() {
            i = (i + 1) & mask;
        }
        assert!(self.slots[i].set((key, value)).is_ok(), "unserialized put");
        self.len.fetch_add(1, Ordering::Relaxed);
        &self.slots[i].get().expect("just set").1
    }

    fn has_room(&self) -> bool {
        self.len.load(Ordering::Relaxed) * 2 < self.slots.len()
    }
}

pub(crate) struct OnceTable<V> {
    segs: [OnceLock<Seg<V>>; SEGS],
    /// Held only while adding a key that is not there yet. Lookups and
    /// `get_or_insert_with` hits never touch it.
    grow: Mutex<()>,
}

impl<V> Default for OnceTable<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> OnceTable<V> {
    pub(crate) fn new() -> Self {
        OnceTable {
            segs: std::array::from_fn(|_| OnceLock::new()),
            grow: Mutex::new(()),
        }
    }

    fn with_capacity_for(len: usize) -> Self {
        let table = Self::new();
        // Strictly more than twice `len`, so `has_room` holds after `len` puts.
        let cap = (len * 2 + 1).next_power_of_two().max(BASE_CAP);
        let _ = table.segs[0].set(Seg::new(cap));
        table
    }

    pub(crate) fn len(&self) -> usize {
        self.segs
            .iter()
            .map_while(|s| s.get())
            .map(|s| s.len.load(Ordering::Relaxed))
            .sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(crate) fn get(&self, key: u64) -> Option<&V> {
        for seg in &self.segs {
            let seg = seg.get()?;
            if let Some(v) = seg.find(key) {
                return Some(v);
            }
        }
        None
    }

    pub(crate) fn contains_key(&self, key: u64) -> bool {
        self.get(key).is_some()
    }

    /// The value stored under `key`, inserting `make()` when absent.
    pub(crate) fn get_or_insert_with(&self, key: u64, make: impl FnOnce() -> V) -> &V {
        if let Some(v) = self.get(key) {
            return v;
        }
        let _grow = self.grow.lock().unwrap();
        let mut tail = 0;
        let mut total = 0;
        for (i, seg) in self.segs.iter().enumerate() {
            let Some(seg) = seg.get() else { break };
            if let Some(v) = seg.find(key) {
                return v;
            }
            tail = i;
            total += seg.len.load(Ordering::Relaxed);
        }
        loop {
            let seg = self.segs[tail]
                .get_or_init(|| Seg::new((total * 8).next_power_of_two().max(BASE_CAP)));
            if seg.has_room() {
                return seg.put(key, make());
            }
            tail += 1;
            assert!(tail < SEGS, "OnceTable segment chain exhausted");
        }
    }

    pub(crate) fn insert(&self, key: u64, value: V) {
        self.get_or_insert_with(key, || value);
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u64, &V)> {
        self.segs
            .iter()
            .map_while(|s| s.get())
            .flat_map(|s| s.slots.iter())
            .filter_map(|slot| slot.get().map(|(k, v)| (*k, v)))
    }

    pub(crate) fn into_entries(self) -> impl Iterator<Item = (u64, V)> {
        self.segs
            .into_iter()
            .filter_map(|s| s.into_inner())
            .flat_map(|s| s.slots.into_vec())
            .filter_map(|slot| slot.into_inner())
    }

    /// Fold the segment chain into one segment sized for the current
    /// entries, so every later hit costs one probe sequence.
    pub(crate) fn compact(&mut self) {
        if self.segs[1].get().is_none() && self.segs[0].get().is_none_or(|s| s.has_room()) {
            return;
        }
        let old = std::mem::take(self);
        let fresh = Self::with_capacity_for(old.len());
        let seg = fresh.segs[0].get().expect("just set");
        for (k, v) in old.into_entries() {
            seg.put(k, v);
        }
        *self = fresh;
    }
}

impl<V: Clone> Clone for OnceTable<V> {
    fn clone(&self) -> Self {
        // Collect first: other threads may insert while we iterate, and the
        // copy must be sized for what we actually saw.
        let entries: Vec<(u64, V)> = self.iter().map(|(k, v)| (k, v.clone())).collect();
        let fresh = Self::with_capacity_for(entries.len());
        let seg = fresh.segs[0].get().expect("just set");
        for (k, v) in entries {
            seg.put(k, v);
        }
        fresh
    }
}

/// Append-only vector whose filled slots are read without a lock. Chunk
/// `k` holds `BASE_CAP << k` slots, so nothing is allocated until the
/// first push and no element ever moves. Pushes must be serialized by
/// the caller.
pub(crate) struct OnceStore<T> {
    chunks: [OnceLock<Box<[OnceLock<T>]>>; 26],
}

impl<T> Default for OnceStore<T> {
    fn default() -> Self {
        OnceStore {
            chunks: std::array::from_fn(|_| OnceLock::new()),
        }
    }
}

impl<T> OnceStore<T> {
    fn locate(index: usize) -> (usize, usize) {
        let n = index / BASE_CAP + 1;
        let chunk = (usize::BITS - 1 - n.leading_zeros()) as usize;
        (chunk, index - BASE_CAP * ((1 << chunk) - 1))
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        let (chunk, offset) = Self::locate(index);
        self.chunks[chunk].get()?[offset].get()
    }

    pub(crate) fn set(&self, index: usize, value: T) {
        let (chunk, offset) = Self::locate(index);
        let slots = self.chunks[chunk]
            .get_or_init(|| (0..BASE_CAP << chunk).map(|_| OnceLock::new()).collect());
        assert!(
            slots[offset].set(value).is_ok(),
            "OnceStore slot written twice"
        );
    }
}
