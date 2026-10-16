//! Insert-only hash map whose reads take no lock (ADR-0034 Decision 3).
//!
//! The generic-key sibling of [`crate::once_table::OnceTable`], holding
//! the derived caches that grow from empty while files are checked.
//!
//! Two parts. The **store** holds every `(K, V)` exactly once, in
//! 64-slot chunks that are appended and never moved, so a `&V` handed
//! out stays valid while other threads keep inserting. The **index**
//! maps a hash to a store id: an open-addressing table of 8-byte slots
//! (hash tag + id), read with one atomic load. When the newest index is
//! half full, a table twice its size receives every id and is then
//! published, so a lookup probes one table only. Older index tables
//! stay allocated until the map is dropped, because a reader may still
//! be probing one; they hold ids, not entries, so the copy clones
//! nothing and the retained bytes are 8 per slot.
//!
//! First write wins. [`OnceMap::insert_first`] takes the `grow` lock,
//! looks the key up again under it, and hands back whichever value the
//! map holds afterwards — the caller's or an earlier one. Values are
//! computed before the call, outside the lock, so a computation may
//! re-enter the same map and two threads may compute the same key; the
//! loser's value is dropped.

use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use rustc_hash::FxBuildHasher;


const BASE_CAP: usize = 64;
/// Each generation doubles the previous one, so this bounds the map far
/// beyond addressable memory.
const GENERATIONS: usize = 32;

/// Entries per store chunk. Also the store's only waste: at most one
/// chunk per map is partly empty.
const CHUNK: usize = 64;

/// A store chunk: `CHUNK` write-once entries.
type Chunk<K, V> = Box<[OnceLock<(K, V)>]>;

/// One directory level: `2^l` chunk pointers, allocated when the level
/// is first reached.
type Level<K, V> = Box<[OnceLock<Chunk<K, V>>]>;

/// Append-only entry storage. Chunk `c` lives in level `log2(c + 1)`,
/// so level `l` holds `2^l` chunk pointers and the directory grows
/// without ever moving a chunk or a pointer a reader may be following.
struct Store<K, V> {
    levels: [OnceLock<Level<K, V>>; GENERATIONS],
    /// Entries whose index slot is published. Ids below it are readable.
    len: AtomicU32,
}

impl<K, V> Store<K, V> {
    fn new() -> Self {
        Store {
            levels: std::array::from_fn(|_| OnceLock::new()),
            len: AtomicU32::new(0),
        }
    }

    #[inline]
    fn locate(id: u32) -> (usize, usize, usize) {
        let chunk = id as usize / CHUNK;
        let level = (usize::BITS - 1 - (chunk + 1).leading_zeros()) as usize;
        (level, chunk + 1 - (1 << level), id as usize % CHUNK)
    }

    #[inline]
    fn get(&self, id: u32) -> &(K, V) {
        let (level, chunk, slot) = Self::locate(id);
        self.levels[level]
            .get()
            .and_then(|level| level[chunk].get())
            .and_then(|chunk| chunk[slot].get())
            .expect("published id has an entry")
    }

    /// Write the entry for `id`. Callers serialize `put` (the map's
    /// `grow` lock) and hand out ids in order, so the chunk either
    /// exists or is the next one to allocate.
    fn put(&self, id: u32, key: K, value: V) -> &(K, V) {
        let (level, chunk, slot) = Self::locate(id);
        let level = self.levels[level]
            .get_or_init(|| (0..1usize << level).map(|_| OnceLock::new()).collect());
        let chunk = level[chunk].get_or_init(|| (0..CHUNK).map(|_| OnceLock::new()).collect());
        assert!(chunk[slot].set((key, value)).is_ok(), "unserialized put");
        chunk[slot].get().expect("just set")
    }

    #[cfg(test)]
    fn bytes(&self) -> usize {
        let chunk_bytes = CHUNK * std::mem::size_of::<OnceLock<(K, V)>>();
        self.levels
            .iter()
            .filter_map(OnceLock::get)
            .map(|level| {
                let ptrs = level.len() * std::mem::size_of::<OnceLock<Chunk<K, V>>>();
                ptrs + level.iter().filter(|c| c.get().is_some()).count() * chunk_bytes
            })
            .sum()
    }
}

/// Hash → store id. A slot is `0` when empty, otherwise the low 32 bits
/// of the key's hash in the high half and `id + 1` in the low half.
struct Index {
    slots: Box<[AtomicU64]>,
    /// `64 - log2(slots.len())`: the top bits of the mixed hash pick the
    /// first slot of a probe sequence.
    shift: u32,
    len: AtomicUsize,
}

#[inline]
fn encode(hash: u64, id: u32) -> u64 {
    ((hash as u32 as u64) << 32) | (id as u64 + 1)
}

impl Index {
    fn new(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Index {
            slots: (0..cap).map(|_| AtomicU64::new(0)).collect(),
            shift: 64 - cap.trailing_zeros(),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn start(&self, hash: u64) -> usize {
        (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// Probe for `hash`; `matches(id)` reads the store entry and decides.
    #[inline]
    fn find(&self, hash: u64, matches: impl Fn(u32) -> bool) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let tag = hash as u32;
        let mut i = self.start(hash);
        loop {
            let slot = self.slots[i].load(Ordering::Acquire);
            if slot == 0 {
                return None;
            }
            let id = (slot as u32).wrapping_sub(1);
            if (slot >> 32) as u32 == tag && matches(id) {
                return Some(id);
            }
            i = (i + 1) & mask;
        }
    }

    /// Callers serialize `put` (the map's `grow` lock), so the first
    /// empty slot of the probe sequence is free to take.
    fn put(&self, hash: u64, id: u32) {
        // A table without room may have no empty slot left, and the
        // probe below would then spin forever.
        assert!(self.has_room(), "put into a full index");
        let mask = self.slots.len() - 1;
        let mut i = self.start(hash);
        while self.slots[i].load(Ordering::Relaxed) != 0 {
            i = (i + 1) & mask;
        }
        self.slots[i].store(encode(hash, id), Ordering::Release);
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    fn has_room(&self) -> bool {
        self.len.load(Ordering::Relaxed) * 2 < self.slots.len()
    }

    fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().filter_map(|slot| {
            let s = slot.load(Ordering::Relaxed);
            (s != 0).then(|| (s as u32).wrapping_sub(1))
        })
    }
}

pub(crate) struct OnceMap<K, V> {
    store: Store<K, V>,
    /// Every index built so far, smallest first. Only the newest one is
    /// read; the rest may still be under a reader's probe.
    indexes: [OnceLock<Index>; GENERATIONS],
    /// How many of `indexes` are published. An index is counted only
    /// after it holds every id of its predecessor.
    published: AtomicUsize,
    /// Held only while adding a key. Lookups never touch it.
    grow: Mutex<()>,
}

impl<K, V> Default for OnceMap<K, V> {
    fn default() -> Self {
        OnceMap {
            store: Store::new(),
            indexes: std::array::from_fn(|_| OnceLock::new()),
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
    fn newest(&self) -> Option<&Index> {
        let n = self.published.load(Ordering::Acquire);
        if n == 0 {
            None
        } else {
            self.indexes[n - 1].get()
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.store.len.load(Ordering::Acquire) as usize
    }

    /// Every entry, in insertion order. An insert racing with the walk
    /// may or may not be seen.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        (0..self.store.len.load(Ordering::Acquire)).map(|id| {
            let (k, v) = self.store.get(id);
            (k, v)
        })
    }

    /// Bytes held by the index tables and the store, for the test that
    /// pins the overhead per entry. Not a public measure.
    #[cfg(test)]
    fn debug_bytes(&self) -> usize {
        let index: usize = self
            .indexes
            .iter()
            .filter_map(OnceLock::get)
            .map(|t| t.slots.len() * std::mem::size_of::<AtomicU64>())
            .sum();
        index + self.store.bytes()
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
        let hash = FxBuildHasher.hash_one(probe);
        let id = self.newest()?.find(hash, |id| eq(&self.store.get(id).0))?;
        Some(&self.store.get(id).1)
    }

    /// Store `value` under `key` unless the key is already there, and
    /// return the value the map holds afterwards.
    pub(crate) fn insert_first(&self, key: K, value: V) -> &V {
        let hash = FxBuildHasher.hash_one(&key);
        let _grow = self.grow.lock().unwrap();
        let n = self.published.load(Ordering::Relaxed);
        let newest = match n {
            0 => None,
            _ => Some(self.indexes[n - 1].get().expect("published index")),
        };
        if let Some(id) = newest.and_then(|i| i.find(hash, |id| self.store.get(id).0 == key)) {
            return &self.store.get(id).1;
        }
        let id = self.store.len.load(Ordering::Relaxed);
        let entry = self.store.put(id, key, value);
        match newest {
            Some(index) if index.has_room() => index.put(hash, id),
            _ => {
                assert!(n < GENERATIONS, "OnceMap generations exhausted");
                let cap = newest.map_or(BASE_CAP, |t| t.slots.len() * 2);
                let next = self.indexes[n].get_or_init(|| Index::new(cap));
                for old in newest.into_iter().flat_map(Index::ids) {
                    next.put(FxBuildHasher.hash_one(&self.store.get(old).0), old);
                }
                next.put(hash, id);
                self.published.store(n + 1, Ordering::Release);
            }
        }
        self.store.len.store(id + 1, Ordering::Release);
        &entry.1
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
