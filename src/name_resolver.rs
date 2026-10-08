//! String-building helper for rbs_loader's declaration walk.
//!
//! `qualified_name` only concatenates a namespace prefix and a name; it
//! does no resolution against an environment, which is why it sits
//! outside `environment::resolution` (the rbs `Resolver` port).

pub(crate) fn qualified_name(ns_prefix: &str, name: &str) -> String {
    if name.starts_with("::") {
        name.to_string()
    } else if ns_prefix.is_empty() {
        format!("::{}", name)
    } else {
        format!("{}::{}", ns_prefix, name)
    }
}
