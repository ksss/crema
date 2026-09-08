// Ported from ADR-0028 spike 1 verbatim; refactor forbidden by slice 1a scope.
#![allow(clippy::new_without_default, clippy::manual_is_multiple_of)]

//! Layout B (hybrid) flat snapshot layout.
//!
//! Global tables (name/symbol interner, type-name table, normalized-alias
//! table, entry index) are stored as flat open-addressed hash tables that can
//! be probed directly on the mmap'd bytes — no deserialization, no hashmap
//! rebuild. Decl entry payloads are bincode-encoded individually and decoded
//! per entry on reference.
//!
//! All ids are content-addressed non-zero u64, so `0` marks an empty slot.
//! Tables use power-of-two capacity, load factor <= 0.5, linear probing,
//! slot seed = id (ids are already xxh3 hashes).

use crate::snapshot::mirror::*;

pub const MAGIC: u64 = 0x5350_494B_4531_4200; // "SPIKE1B\0"

const H_MAGIC: usize = 0;
const H_NAMES_COUNT: usize = 1;
const H_NAMES_OFFSETS: usize = 2;
const H_NAMES_BLOB: usize = 3;
const H_SYM_CAP: usize = 4;
const H_SYM_TABLE: usize = 5;
const H_SYM_BLOB: usize = 6;
const H_TN_CAP: usize = 7;
const H_TN_TABLE: usize = 8;
const H_NORM_CAP: usize = 9;
const H_NORM_TABLE: usize = 10;
const H_IDX_CAP: usize = 11;
const H_IDX_TABLE: usize = 12;
const H_ENTRIES_BLOB: usize = 13;
const H_ENTRIES_LEN: usize = 14;
const HEADER_WORDS: usize = 16;

fn cap_for(n: usize) -> usize {
    (n.max(1) * 2).next_power_of_two()
}

pub struct Writer {
    pub buf: Vec<u8>,
    header: [u64; HEADER_WORDS],
}

impl Writer {
    pub fn new() -> Self {
        let mut w = Writer {
            buf: vec![0u8; HEADER_WORDS * 8],
            header: [0; HEADER_WORDS],
        };
        w.header[H_MAGIC] = MAGIC;
        w
    }

    fn align8(&mut self) {
        while self.buf.len() % 8 != 0 {
            self.buf.push(0);
        }
    }

    fn mark(&mut self) -> u64 {
        self.align8();
        self.buf.len() as u64
    }

    pub fn write_names(&mut self, names: &[String]) {
        self.header[H_NAMES_COUNT] = names.len() as u64;
        self.header[H_NAMES_OFFSETS] = self.mark();
        let mut off = 0u32;
        for n in names {
            self.buf.extend_from_slice(&off.to_le_bytes());
            off += n.len() as u32;
        }
        self.buf.extend_from_slice(&off.to_le_bytes());
        self.header[H_NAMES_BLOB] = self.mark();
        for n in names {
            self.buf.extend_from_slice(n.as_bytes());
        }
    }

    /// Slot: id u64, off u32, len u32 (16 bytes). Strings in a blob.
    pub fn write_symbols(&mut self, symbols: &FxMap<Box<str>>) {
        let cap = cap_for(symbols.len());
        self.header[H_SYM_CAP] = cap as u64;
        let table_off = self.mark();
        self.buf.resize(self.buf.len() + cap * 16, 0);
        self.header[H_SYM_TABLE] = table_off;

        let mut blob: Vec<u8> = Vec::new();
        for (id, s) in symbols {
            let off = blob.len() as u32;
            blob.extend_from_slice(s.as_bytes());
            let mut slot = (*id as usize) & (cap - 1);
            loop {
                let p = table_off as usize + slot * 16;
                if u64::from_le_bytes(self.buf[p..p + 8].try_into().unwrap()) == 0 {
                    self.buf[p..p + 8].copy_from_slice(&id.to_le_bytes());
                    self.buf[p + 8..p + 12].copy_from_slice(&off.to_le_bytes());
                    self.buf[p + 12..p + 16].copy_from_slice(&(s.len() as u32).to_le_bytes());
                    break;
                }
                slot = (slot + 1) & (cap - 1);
            }
        }
        self.header[H_SYM_BLOB] = self.mark();
        self.buf.extend_from_slice(&blob);
    }

    /// Slot: id u64, parent u64, segment u64, flags u64 (32 bytes).
    pub fn write_type_names(&mut self, tns: &FxMap<MTnEntry>) {
        let cap = cap_for(tns.len());
        self.header[H_TN_CAP] = cap as u64;
        let table_off = self.mark();
        self.buf.resize(self.buf.len() + cap * 32, 0);
        self.header[H_TN_TABLE] = table_off;
        for (id, (parent, seg, abs)) in tns {
            let mut slot = (*id as usize) & (cap - 1);
            loop {
                let p = table_off as usize + slot * 32;
                if u64::from_le_bytes(self.buf[p..p + 8].try_into().unwrap()) == 0 {
                    self.buf[p..p + 8].copy_from_slice(&id.to_le_bytes());
                    self.buf[p + 8..p + 16].copy_from_slice(&parent.to_le_bytes());
                    self.buf[p + 16..p + 24].copy_from_slice(&seg.to_le_bytes());
                    self.buf[p + 24..p + 32].copy_from_slice(&(*abs as u64).to_le_bytes());
                    break;
                }
                slot = (slot + 1) & (cap - 1);
            }
        }
    }

    /// Slot: id u64, tag u64, aux u64 (24 bytes).
    pub fn write_normalized(&mut self, norm: &FxMap<MNormResult>) {
        let cap = cap_for(norm.len());
        self.header[H_NORM_CAP] = cap as u64;
        let table_off = self.mark();
        self.buf.resize(self.buf.len() + cap * 24, 0);
        self.header[H_NORM_TABLE] = table_off;
        // `original` on UnknownTarget/Cycle/NotClassOrModule is discarded
        // (aux=0 or stored as `target`) because it always equals the outer
        // slot key `id` — see draft::precompute_normalized_module_names
        // where the loop key is inserted as `original`. Decoders recover
        // it from `id` and don't need aux.
        for (id, r) in norm {
            let (tag, aux): (u64, u64) = match r {
                MNormResult::Normalized(t) => (1, *t),
                MNormResult::UnknownTarget { target, .. } => (2, *target),
                MNormResult::Cycle { .. } => (3, 0),
                MNormResult::NotClassOrModule { .. } => (4, 0),
            };
            let mut slot = (*id as usize) & (cap - 1);
            loop {
                let p = table_off as usize + slot * 24;
                if u64::from_le_bytes(self.buf[p..p + 8].try_into().unwrap()) == 0 {
                    self.buf[p..p + 8].copy_from_slice(&id.to_le_bytes());
                    self.buf[p + 8..p + 16].copy_from_slice(&tag.to_le_bytes());
                    self.buf[p + 16..p + 24].copy_from_slice(&aux.to_le_bytes());
                    break;
                }
                slot = (slot + 1) & (cap - 1);
            }
        }
    }

    /// entries: (id, encoded bincode bytes). Index slot: id u64,
    /// off u64, len u64 (24 bytes).
    pub fn write_entries(&mut self, entries: &[(u64, Vec<u8>)]) {
        let cap = cap_for(entries.len());
        self.header[H_IDX_CAP] = cap as u64;
        let table_off = self.mark();
        self.buf.resize(self.buf.len() + cap * 24, 0);
        self.header[H_IDX_TABLE] = table_off;

        let mut blob: Vec<u8> = Vec::new();
        for (id, bytes) in entries {
            let off = blob.len() as u64;
            blob.extend_from_slice(bytes);
            let mut slot = (*id as usize) & (cap - 1);
            loop {
                let p = table_off as usize + slot * 24;
                if u64::from_le_bytes(self.buf[p..p + 8].try_into().unwrap()) == 0 {
                    self.buf[p..p + 8].copy_from_slice(&id.to_le_bytes());
                    self.buf[p + 8..p + 16].copy_from_slice(&off.to_le_bytes());
                    self.buf[p + 16..p + 24].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
                    break;
                }
                slot = (slot + 1) & (cap - 1);
            }
        }
        self.header[H_ENTRIES_BLOB] = self.mark();
        self.header[H_ENTRIES_LEN] = blob.len() as u64;
        self.buf.extend_from_slice(&blob);
    }

    pub fn finish(mut self) -> Vec<u8> {
        for (i, w) in self.header.iter().enumerate() {
            self.buf[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        self.buf
    }
}

pub struct Reader<'a> {
    data: &'a [u8],
    header: [u64; HEADER_WORDS],
}

impl<'a> Reader<'a> {
    pub fn open(data: &'a [u8]) -> Self {
        let mut header = [0u64; HEADER_WORDS];
        for (i, w) in header.iter_mut().enumerate() {
            *w = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        }
        assert_eq!(header[H_MAGIC], MAGIC, "bad magic");
        Reader { data, header }
    }

    /// Validating sibling of [`Self::open`] for untrusted bytes (slice 1c
    /// read path). Checks every section extent, every blob range referenced
    /// from a table slot, and blob UTF-8 up front so the panicking accessors
    /// and the `iter_*` methods below cannot go out of bounds afterwards.
    /// O(file) — acceptable because the read path decodes eagerly anyway.
    pub fn try_open(data: &'a [u8]) -> Result<Self, String> {
        if data.len() < HEADER_WORDS * 8 {
            return Err(format!("payload shorter than header: {} bytes", data.len()));
        }
        let mut header = [0u64; HEADER_WORDS];
        for (i, w) in header.iter_mut().enumerate() {
            *w = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        }
        if header[H_MAGIC] != MAGIC {
            return Err(format!("bad flat magic: {:#x}", header[H_MAGIC]));
        }
        let len = data.len();
        let extent = |label: &str, off: u64, size: u64| -> Result<usize, String> {
            let end = off
                .checked_add(size)
                .ok_or_else(|| format!("{label}: offset overflow"))?;
            if end as usize > len {
                return Err(format!("{label}: extent {end} exceeds payload {len}"));
            }
            Ok(off as usize)
        };
        for (label, cap_word, table_word, slot) in [
            ("symbols", H_SYM_CAP, H_SYM_TABLE, 16u64),
            ("type_names", H_TN_CAP, H_TN_TABLE, 32),
            ("normalized", H_NORM_CAP, H_NORM_TABLE, 24),
            ("entry index", H_IDX_CAP, H_IDX_TABLE, 24),
        ] {
            let cap = header[cap_word];
            if cap == 0 || !cap.is_power_of_two() {
                return Err(format!("{label}: capacity {cap} is not a power of two"));
            }
            extent(label, header[table_word], cap * slot)?;
        }
        let r = Reader { data, header };

        let names_count = header[H_NAMES_COUNT];
        extent(
            "name offsets",
            header[H_NAMES_OFFSETS],
            (names_count + 1) * 4,
        )?;
        let names_blob = header[H_NAMES_BLOB] as usize;
        let mut prev = 0u32;
        for i in 0..=names_count as usize {
            let off = r.u32_at(header[H_NAMES_OFFSETS] as usize + i * 4);
            if off < prev {
                return Err(format!("name offsets not monotonic at {i}"));
            }
            prev = off;
        }
        extent("name blob", header[H_NAMES_BLOB], prev as u64)?;
        for i in 0..names_count as u32 {
            let a = r.u32_at(header[H_NAMES_OFFSETS] as usize + i as usize * 4) as usize;
            let b = r.u32_at(header[H_NAMES_OFFSETS] as usize + (i as usize + 1) * 4) as usize;
            std::str::from_utf8(&data[names_blob + a..names_blob + b])
                .map_err(|e| format!("name {i} is not UTF-8: {e}"))?;
        }

        let sym_blob = header[H_SYM_BLOB];
        for slot in 0..header[H_SYM_CAP] as usize {
            let p = header[H_SYM_TABLE] as usize + slot * 16;
            if r.u64_at(p) == 0 {
                continue;
            }
            let off = r.u32_at(p + 8) as u64;
            let slen = r.u32_at(p + 12) as u64;
            let start = extent("symbol blob", sym_blob + off, slen)?;
            std::str::from_utf8(&data[start..start + slen as usize])
                .map_err(|e| format!("symbol at slot {slot} is not UTF-8: {e}"))?;
        }

        let entries_blob = header[H_ENTRIES_BLOB];
        extent("entries blob", entries_blob, header[H_ENTRIES_LEN])?;
        for slot in 0..header[H_IDX_CAP] as usize {
            let p = header[H_IDX_TABLE] as usize + slot * 24;
            if r.u64_at(p) == 0 {
                continue;
            }
            let off = r.u64_at(p + 8);
            let elen = r.u64_at(p + 16);
            if off
                .checked_add(elen)
                .is_none_or(|end| end > header[H_ENTRIES_LEN])
            {
                return Err(format!("entry at slot {slot} exceeds entries blob"));
            }
        }
        Ok(r)
    }

    #[inline]
    fn u64_at(&self, off: usize) -> u64 {
        u64::from_le_bytes(self.data[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.data[off..off + 4].try_into().unwrap())
    }

    pub fn lookup_entry(&self, id: u64) -> Option<&'a [u8]> {
        let cap = self.header[H_IDX_CAP] as usize;
        let table = self.header[H_IDX_TABLE] as usize;
        let blob = self.header[H_ENTRIES_BLOB] as usize;
        let mut slot = (id as usize) & (cap - 1);
        loop {
            let p = table + slot * 24;
            let sid = self.u64_at(p);
            if sid == 0 {
                return None;
            }
            if sid == id {
                let off = self.u64_at(p + 8) as usize;
                let len = self.u64_at(p + 16) as usize;
                return Some(&self.data[blob + off..blob + off + len]);
            }
            slot = (slot + 1) & (cap - 1);
        }
    }

    pub fn tn_entry(&self, id: u64) -> Option<(u64, u64, bool)> {
        let cap = self.header[H_TN_CAP] as usize;
        let table = self.header[H_TN_TABLE] as usize;
        let mut slot = (id as usize) & (cap - 1);
        loop {
            let p = table + slot * 32;
            let sid = self.u64_at(p);
            if sid == 0 {
                return None;
            }
            if sid == id {
                return Some((
                    self.u64_at(p + 8),
                    self.u64_at(p + 16),
                    self.u64_at(p + 24) != 0,
                ));
            }
            slot = (slot + 1) & (cap - 1);
        }
    }

    pub fn sym_str(&self, id: u64) -> Option<&'a str> {
        let cap = self.header[H_SYM_CAP] as usize;
        let table = self.header[H_SYM_TABLE] as usize;
        let blob = self.header[H_SYM_BLOB] as usize;
        let mut slot = (id as usize) & (cap - 1);
        loop {
            let p = table + slot * 16;
            let sid = self.u64_at(p);
            if sid == 0 {
                return None;
            }
            if sid == id {
                let off = self.u32_at(p + 8) as usize;
                let len = self.u32_at(p + 12) as usize;
                return Some(
                    std::str::from_utf8(&self.data[blob + off..blob + off + len]).unwrap(),
                );
            }
            slot = (slot + 1) & (cap - 1);
        }
    }

    pub fn normalized(&self, id: u64) -> Option<(u64, u64)> {
        let cap = self.header[H_NORM_CAP] as usize;
        let table = self.header[H_NORM_TABLE] as usize;
        let mut slot = (id as usize) & (cap - 1);
        loop {
            let p = table + slot * 24;
            let sid = self.u64_at(p);
            if sid == 0 {
                return None;
            }
            if sid == id {
                return Some((self.u64_at(p + 8), self.u64_at(p + 16)));
            }
            slot = (slot + 1) & (cap - 1);
        }
    }

    pub fn name_str(&self, id: u32) -> &'a str {
        let count = self.header[H_NAMES_COUNT] as usize;
        assert!((id as usize) < count);
        let offsets = self.header[H_NAMES_OFFSETS] as usize;
        let blob = self.header[H_NAMES_BLOB] as usize;
        let a = self.u32_at(offsets + id as usize * 4) as usize;
        let b = self.u32_at(offsets + (id as usize + 1) * 4) as usize;
        std::str::from_utf8(&self.data[blob + a..blob + b]).unwrap()
    }

    pub fn names_len(&self) -> usize {
        self.header[H_NAMES_COUNT] as usize
    }

    /// Full-slot scan of the symbol table (slice 1c rebuild). Order is
    /// slot order, deterministic for a given file but meaningless.
    pub fn iter_symbols(&self) -> impl Iterator<Item = (u64, &'a str)> + '_ {
        let cap = self.header[H_SYM_CAP] as usize;
        let table = self.header[H_SYM_TABLE] as usize;
        let blob = self.header[H_SYM_BLOB] as usize;
        (0..cap).filter_map(move |slot| {
            let p = table + slot * 16;
            let id = self.u64_at(p);
            if id == 0 {
                return None;
            }
            let off = self.u32_at(p + 8) as usize;
            let len = self.u32_at(p + 12) as usize;
            let s = std::str::from_utf8(&self.data[blob + off..blob + off + len]).unwrap();
            Some((id, s))
        })
    }

    pub fn iter_type_names(&self) -> impl Iterator<Item = (u64, MTnEntry)> + '_ {
        let cap = self.header[H_TN_CAP] as usize;
        let table = self.header[H_TN_TABLE] as usize;
        (0..cap).filter_map(move |slot| {
            let p = table + slot * 32;
            let id = self.u64_at(p);
            if id == 0 {
                return None;
            }
            Some((
                id,
                (
                    self.u64_at(p + 8),
                    self.u64_at(p + 16),
                    self.u64_at(p + 24) != 0,
                ),
            ))
        })
    }

    /// Yields `(id, (tag, aux))` in the encoding `write_normalized` uses.
    pub fn iter_normalized(&self) -> impl Iterator<Item = (u64, (u64, u64))> + '_ {
        let cap = self.header[H_NORM_CAP] as usize;
        let table = self.header[H_NORM_TABLE] as usize;
        (0..cap).filter_map(move |slot| {
            let p = table + slot * 24;
            let id = self.u64_at(p);
            if id == 0 {
                return None;
            }
            Some((id, (self.u64_at(p + 8), self.u64_at(p + 16))))
        })
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (u64, &'a [u8])> + '_ {
        let cap = self.header[H_IDX_CAP] as usize;
        let table = self.header[H_IDX_TABLE] as usize;
        let blob = self.header[H_ENTRIES_BLOB] as usize;
        (0..cap).filter_map(move |slot| {
            let p = table + slot * 24;
            let id = self.u64_at(p);
            if id == 0 {
                return None;
            }
            let off = self.u64_at(p + 8) as usize;
            let len = self.u64_at(p + 16) as usize;
            Some((id, &self.data[blob + off..blob + off + len]))
        })
    }

    /// Walk the type-name chain resolving each segment string — includes the
    /// symbol-table probes so touched-entry benchmarks pay the same global
    /// table costs a real check would.
    pub fn display_tn(&self, id: u64) -> String {
        let mut segs: Vec<u64> = Vec::new();
        let mut cur = id;
        let absolute = loop {
            let (parent, seg, abs) = self.tn_entry(cur).expect("tn present");
            if parent == 0 {
                break abs;
            }
            segs.push(seg);
            cur = parent;
        };
        segs.reverse();
        let mut s = String::new();
        if absolute {
            s.push_str("::");
        }
        for (i, seg) in segs.iter().enumerate() {
            if i > 0 {
                s.push_str("::");
            }
            s.push_str(self.sym_str(*seg).expect("sym present"));
        }
        s
    }
}
