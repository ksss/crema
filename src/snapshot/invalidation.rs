//! G-snapshot invalidation key: a 256-bit content fingerprint over every
//! input whose change must force a snapshot rebuild (ADR-0028 Decision 4:
//! lockfiles, crema version, crema.toml, sig path set — plus the gem
//! resolver mode, since `--no-bundler` can point G at a different rbs
//! install than bundler does).

use xxhash_rust::xxh3::xxh3_64_with_seed;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidationKey(pub [u8; 32]);

impl InvalidationKey {
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Inputs are fed in this struct's field order (canonical); reordering
/// fields is a schema change. `None` and `Some(b"")` hash differently
/// (absent marker byte vs zero-length content).
pub struct InvalidationInputs<'a> {
    pub crema_version: &'a str,
    pub gemfile_lock_content: Option<&'a [u8]>,
    pub rbs_collection_lock_content: Option<&'a [u8]>,
    pub crema_toml_content: Option<&'a [u8]>,
    /// Normalized and sorted by the caller (contract; not re-sorted here).
    pub sig_paths: &'a [&'a str],
    /// `--no-bundler`: the resolver was pinned to plain `ruby`. Keyed on
    /// the flag, not on which program actually ran, so a bundler-mode
    /// run that fell back to `ruby` (bundle missing from PATH) still
    /// shares its key with the bundler-mode runs it belongs to.
    pub no_bundler: bool,
}

/// The `crema_version` input for [`compute`]. Production builds always
/// use the compiled crate version; debug builds honor
/// `CREMA_VERSION_OVERRIDE` so e2e tests can force a key mismatch
/// without rebuilding the binary.
pub fn crema_version() -> String {
    if cfg!(debug_assertions)
        && let Ok(v) = std::env::var("CREMA_VERSION_OVERRIDE")
    {
        return v;
    }
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn compute(inputs: &InvalidationInputs<'_>) -> InvalidationKey {
    hash_buf(&base_buf(inputs))
}

fn base_buf(inputs: &InvalidationInputs<'_>) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    feed(&mut buf, 0, Some(inputs.crema_version.as_bytes()));
    feed(&mut buf, 1, inputs.gemfile_lock_content);
    feed(&mut buf, 2, inputs.rbs_collection_lock_content);
    feed(&mut buf, 3, inputs.crema_toml_content);
    buf.push(4);
    buf.extend_from_slice(&(inputs.sig_paths.len() as u64).to_le_bytes());
    for p in inputs.sig_paths {
        buf.extend_from_slice(&(p.len() as u64).to_le_bytes());
        buf.extend_from_slice(p.as_bytes());
    }
    buf.push(5);
    buf.push(inputs.no_bundler as u8);
    buf
}

fn hash_buf(buf: &[u8]) -> InvalidationKey {
    let mut key = [0u8; 32];
    for seed in 0..4u64 {
        let h = xxh3_64_with_seed(buf, seed);
        key[seed as usize * 8..(seed as usize + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    InvalidationKey(key)
}

/// Length-prefixed so adjacent inputs cannot alias by concatenation;
/// tag + absent marker keeps `None` distinct from empty content.
fn feed(buf: &mut Vec<u8>, tag: u8, content: Option<&[u8]>) {
    buf.push(tag);
    match content {
        Some(c) => {
            buf.push(1);
            buf.extend_from_slice(&(c.len() as u64).to_le_bytes());
            buf.extend_from_slice(c);
        }
        None => buf.push(0),
    }
}
