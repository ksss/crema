//! Insert-only hash map whose reads take no lock (ADR-0034 Decision 3).
//!
//! The generic-key sibling of [`crate::once_table::OnceTable`], holding
//! the derived caches that grow from empty while files are checked.
//! Every slot is a `OnceLock<(K, V)>`, so a filled slot is read with one
//! atomic load and a `&V` handed out stays valid while other threads keep
//! inserting.
//!
//! Growth copies. When the newest table is half full, a table twice its
//! size receives a clone of every entry and is then published, so a
//! lookup probes one table only. Older tables stay allocated until the
//! map is dropped, because references handed out earlier point into
//! them.
//!
//! First write wins. [`OnceMap::insert_first`] takes the `grow` lock,
//! looks the key up again under it, and hands back whichever value the
//! map holds afterwards — the caller's or an earlier one. Values are
//! computed before the call, outside the lock, so a computation may
//! re-enter the same map and two threads may compute the same key; the
//! loser's value is dropped.

use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use rustc_hash::FxBuildHasher;


const BASE_CAP: usize = 64;
/// Each generation doubles the previous one, so this bounds the map far
/// beyond addressable memory.
const GENERATIONS: usize = 32;

struct Table<K, V> {
    slots: Box<[OnceLock<(K, V)>]>,
    /// `64 - log2(slots.len())`: the top bits of the mixed hash pick the
    /// first slot of a probe sequence.
    shift: u32,
    len: AtomicUsize,
}

impl<K, V> Table<K, V> {
    fn new(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Table {
            slots: (0..cap).map(|_| OnceLock::new()).collect(),
            shift: 64 - cap.trailing_zeros(),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn start(&self, hash: u64) -> usize {
        (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn find(&self, hash: u64, eq: impl Fn(&K) -> bool) -> Option<&V> {
        let mask = self.slots.len() - 1;
        let mut i = self.start(hash);
        loop {
            match self.slots[i].get() {
                None => return None,
                Some((k, v)) if eq(k) => return Some(v),
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    /// Callers serialize `put` (the map's `grow` lock), so the first
    /// empty slot of the probe sequence is free to take.
    fn put(&self, hash: u64, key: K, value: V) -> &V {
        // A table without room may have no empty slot left, and the
        // probe below would then spin forever.
        assert!(self.has_room(), "put into a full table");
        let mask = self.slots.len() - 1;
        let mut i = self.start(hash);
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

    fn entries(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots
            .iter()
            .filter_map(|slot| slot.get().map(|(k, v)| (k, v)))
    }
}

pub(crate) struct OnceMap<K, V> {
    /// Every table built so far, smallest first. Only the newest one is
    /// read; the rest keep earlier `&V`s alive.
    tables: [OnceLock<Table<K, V>>; GENERATIONS],
    /// How many of `tables` are published. A table is counted only after
    /// it holds every entry of its predecessor.
    published: AtomicUsize,
    /// Held only while adding a key. Lookups never touch it.
    grow: Mutex<()>,
}

impl<K, V> Default for OnceMap<K, V> {
    fn default() -> Self {
        OnceMap {
            tables: std::array::from_fn(|_| OnceLock::new()),
            published: AtomicUsize::new(0),
            grow: Mutex::new(()),
        }
    }
}

impl<K, V> std::fmt::Debug for OnceMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnceMap").field("len", &self.len()).finish()
    }
}

impl<K, V> OnceMap<K, V> {
    #[inline]
    fn newest(&self) -> Option<&Table<K, V>> {
        let n = self.published.load(Ordering::Acquire);
        if n == 0 {
            None
        } else {
            self.tables[n - 1].get()
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.newest().map_or(0, |t| t.len.load(Ordering::Relaxed))
    }

    /// Every entry, in no particular order. An insert racing with the
    /// walk may or may not be seen.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.newest().into_iter().flat_map(Table::entries)
    }
}

impl<K: Hash + Eq, V> OnceMap<K, V> {
    #[inline]
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.get_by(key, |k| k == key)
    }

    /// Lookup through a borrowed form of the key, for keys whose owned
    /// form would have to be allocated just to ask. `probe` must hash the
    /// same as the stored `K` it stands for; `eq` decides the match.
    #[inline]
    pub(crate) fn get_by<Q: Hash + ?Sized>(
        &self,
        probe: &Q,
        eq: impl Fn(&K) -> bool,
    ) -> Option<&V> {
        self.newest()?.find(FxBuildHasher.hash_one(probe), eq)
    }
}

impl<K: Hash + Eq + Clone, V: Clone> OnceMap<K, V> {
    /// Store `value` under `key` unless the key is already there, and
    /// return the value the map holds afterwards.
    pub(crate) fn insert_first(&self, key: K, value: V) -> &V {
        let hash = FxBuildHasher.hash_one(&key);
        let _grow = self.grow.lock().unwrap();
        let n = self.published.load(Ordering::Relaxed);
        let newest = match n {
            0 => None,
            _ => Some(self.tables[n - 1].get().expect("published table")),
        };
        if let Some(table) = newest {
            if let Some(v) = table.find(hash, |k| *k == key) {
                return v;
            }
            if table.has_room() {
                return table.put(hash, key, value);
            }
        }
        assert!(n < GENERATIONS, "OnceMap generations exhausted");
        let cap = newest.map_or(BASE_CAP, |t| t.slots.len() * 2);
        let next = self.tables[n].get_or_init(|| Table::new(cap));
        for (k, v) in newest.into_iter().flat_map(Table::entries) {
            next.put(FxBuildHasher.hash_one(k), k.clone(), v.clone());
        }
        let v = next.put(hash, key, value);
        self.published.store(n + 1, Ordering::Release);
        v
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for OnceMap<K, V> {
    fn clone(&self) -> Self {
        let fresh = Self::default();
        for (k, v) in self.iter() {
            fresh.insert_first(k.clone(), v.clone());
        }
        fresh
    }
}
