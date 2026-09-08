//! Peel `_ = ...` / `Foo = _ = ...` write wrappers off an assignment's
//! RHS to reach the underlying `CallNode`.
//!
//! Three call sites peel these wrappers on the way to syntactic
//! dispatch on `Struct.new` / `Class.new` / `Data.define`. They picked
//! different gate semantics for **good reason** — this module makes
//! that reason explicit in one doc so the difference is not read as
//! an accidental divergence:
//!
//! - **Type-checker visitor** (`type_checker/visitor.rs`) — the
//!   assignment's RHS gets type-checked. Only Steep's
//!   `SPECIAL_LVAR_NAMES` idiom (`_` / `__any__` / `__skip__`) reaches
//!   here in practice (that's the standard cast-silencer pattern
//!   `Const = _ = Struct.new(...)` used across the steep repo), and
//!   `__skip__` is explicitly excluded so the
//!   "do-not-type-check-the-RHS" contract from
//!   `Steep::TypeConstruction#:lvasgn` is preserved.
//! - **Inline parser** (`inline_parser.rs`) and
//!   **class_new_recognizer** (`class_new_recognizer.rs`) — static
//!   declaration registration only. No RHS type-check happens here,
//!   so the `__skip__` contract does not apply and gating on
//!   `SPECIAL_LVAR_NAMES` would silently drop
//!   `Const = user_lvar = Struct.new(...)` for no reason. Any
//!   local-variable name is accepted, and deeper chains are unwrapped
//!   recursively.
//!
//! Deliberately keeping the two gate semantics side-by-side rather
//! than unifying them: the `__skip__` contract belongs to type-check
//! callers only. Forcing it on the parser side would silently drop a
//! declaration; forcing "any lvar" on the type-check side would
//! violate the `__skip__` contract.

use ruby_prism::{CallNode, LocalVariableWriteNode, Node};

use crate::type_checker::is_special_lvar_name;

/// Result of peeling wrappers off a value node. `cast` is the outermost
/// `LocalVariableWriteNode` seen (when any); the visitor caller uses it
/// to attach the trailing `#: T` annotation on the intermediate
/// assignment (the outer `ConstantWriteNode` is excluded from top-level
/// statement-assertion eligibility, so the lvar cast is the only site
/// that sees a trailing assertion on this line). The multi-recurse
/// callers ignore it — for them the reason to peel is only to reach
/// the `CallNode` for syntactic dispatch.
pub(crate) struct UnwrappedCastRhs<'pr> {
    pub(crate) cast: Option<LocalVariableWriteNode<'pr>>,
    pub(crate) call: CallNode<'pr>,
}

/// Callable predicate on a local-variable name's bytes.
type LvarGate = fn(&[u8]) -> bool;

/// How the walker should traverse wrappers around the target call.
/// The three call sites picked different combinations; see this
/// module's doc for the reasoning.
pub(crate) struct CastUnwrapConfig {
    /// Which local-variable names are peeled through. The predicate is
    /// consulted for each `LocalVariableWriteNode` wrapper.
    pub(crate) lvar_gate: LvarGate,
    /// Whether to also peel the outer `ConstantWriteNode` wrapper. The
    /// type-checker visitor already sees the RHS (i.e. the constant
    /// write is peeled by the caller before `unwrap_cast` runs), so it
    /// passes `false`; the parser-side callers receive the whole node
    /// and pass `true`.
    pub(crate) peel_constant_write: bool,
    /// Whether to walk arbitrarily deep wrapper chains. `false` peels
    /// one lvar level and stops (the visitor's contract: deeper chains
    /// nothing in practice exercises are not guessed at). `true` keeps
    /// peeling until a non-wrapper node is reached (the parser-side
    /// contract: static registration is safe under any depth).
    pub(crate) recurse: bool,
}

impl CastUnwrapConfig {
    /// Type-checker visitor: single-level, `SPECIAL_LVAR_NAMES` gate
    /// excluding `__skip__`, no constant-write peel.
    pub(crate) const VISITOR: Self = Self {
        lvar_gate: special_lvar_excluding_skip,
        peel_constant_write: false,
        recurse: false,
    };

    /// Inline parser / class_new_recognizer: multi-level, any lvar name,
    /// peel constant-write wrappers too.
    pub(crate) const PARSER: Self = Self {
        lvar_gate: any_lvar,
        peel_constant_write: true,
        recurse: true,
    };
}

fn special_lvar_excluding_skip(name: &[u8]) -> bool {
    is_special_lvar_name(name) && name != b"__skip__"
}

fn any_lvar(_: &[u8]) -> bool {
    true
}

/// Peel wrappers off `value` per `config` and return the underlying
/// `CallNode` plus (when a lvar wrapper was seen) the outermost cast
/// node. Returns `None` when the shape does not bottom out in a call
/// under the given gate.
pub(crate) fn unwrap_cast<'pr>(
    value: &Node<'pr>,
    config: &CastUnwrapConfig,
) -> Option<UnwrappedCastRhs<'pr>> {
    if let Some(call) = value.as_call_node() {
        return Some(UnwrappedCastRhs { cast: None, call });
    }
    if config.peel_constant_write
        && let Some(write) = value.as_constant_write_node()
    {
        return unwrap_cast(&write.value(), config);
    }
    let cast = value.as_local_variable_write_node()?;
    let name_bytes = cast.name().as_slice();
    if !(config.lvar_gate)(name_bytes) {
        return None;
    }
    if config.recurse {
        let inner = unwrap_cast(&cast.value(), config)?;
        return Some(UnwrappedCastRhs {
            cast: inner.cast.or(Some(cast)),
            call: inner.call,
        });
    }
    let call = cast.value().as_call_node()?;
    Some(UnwrappedCastRhs {
        cast: Some(cast),
        call,
    })
}

