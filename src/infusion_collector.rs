//! Infusion collector — synthesizes method declarations from Ruby
//! metaprogramming idioms (e.g. activesupport's `cattr_accessor` /
//! `mattr_accessor`, Rails association macros). Runs on every parsed
//! Ruby source independently of the inline annotation reader, gated
//! by the `[infusion.rails] enabled = true` preset in `crema.toml`.
//!
//! ActiveSupport::Concern expansion is shared by the first Rails DSL
//! rules so that normal class bodies and synthetic Concern bodies pass
//! through the same InfusionBody rule application path.

pub mod active_decorator;
pub mod active_record_synthesis;
pub mod activemodel;
pub mod activerecord;
pub mod activesupport;
pub mod config;
pub(crate) mod inflector;
pub(crate) mod paranoia;
pub(crate) mod pipeline;
mod zeitwerk_synthesis;

pub use inflector::{Inflector, OwnedOrDefault, build_for_config};
pub use pipeline::{
    SourceUnit, collect, load, load_all, load_all_with_options, load_all_with_schema,
};
