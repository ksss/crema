//! Resolver layer — rbs's `resolve_type_names` phase, pulled out of the
//! fused `type_converter` so the definition_builder can call it as a
//! stand-alone helper.
//!
//! Mirrors `lib/rbs/resolver/` in rbs. Phase 2 of ADR-0014.

pub mod type_name_resolver;
