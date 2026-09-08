use ruby_prism::{CallNode, Node};

use crate::data_struct_recognizer::is_static_named_receiver;

/// True iff the call's receiver is bare or rooted `Class` and the
/// method name is `new`. Callers layer their own argument-shape gate
/// on top of this syntactic core.
pub(crate) fn is_class_dot_new(call: &CallNode<'_>) -> bool {
    let Some(receiver) = call.receiver() else {
        return false;
    };
    if call.name().as_slice() != b"new" {
        return false;
    }
    is_static_named_receiver(&receiver, b"Class")
}

/// `&Node` entry point for callers that have not yet descended to a
/// `CallNode`. Returns the call when [`is_class_dot_new`] holds.
///
/// Wrapper peeling (through `ConstantWriteNode` / `LocalVariableWriteNode`,
/// including deeper `Foo = _ = Class.new(...)` chains) is delegated to
/// `crate::cast_unwrap`; the parser-side registration path there takes
/// any lvar name because no RHS type-check is at stake.
pub(crate) fn class_dot_new_call<'n>(node: &Node<'n>) -> Option<CallNode<'n>> {
    let unwrapped =
        crate::cast_unwrap::unwrap_cast(node, &crate::cast_unwrap::CastUnwrapConfig::PARSER)?;
    if !is_class_dot_new(&unwrapped.call) {
        return None;
    }
    Some(unwrapped.call)
}
