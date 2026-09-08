use ruby_prism::{CallNode, Node};

/// Which construction call was recognized: `Struct.new` or `Data.define`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataStructConstructionKind {
    Struct,
    Data,
}

/// Recognize the receiver+method core of a Struct/Data construction
/// call. Returns the kind when the receiver is bare or rooted `Struct`
/// / `Data` and the method name matches (`new` for Struct, `define`
/// for Data). Callers layer their own argument-shape gate on top of
/// this syntactic core (mirrors [`crate::class_new_recognizer`]).
pub(crate) fn data_struct_construction_kind(
    call: &CallNode<'_>,
) -> Option<DataStructConstructionKind> {
    let receiver = call.receiver()?;
    let method = call.name().as_slice();
    if method == b"new" && is_static_named_receiver(&receiver, b"Struct") {
        return Some(DataStructConstructionKind::Struct);
    }
    if method == b"define" && is_static_named_receiver(&receiver, b"Data") {
        return Some(DataStructConstructionKind::Data);
    }
    None
}

/// Match the receiver against `<Name>` (bare) or `::<Name>` (rooted).
/// A nested path like `Foo::Struct` is rejected — only the top-level
/// constant counts as the syntactic anchor for the pattern. Shared
/// with [`crate::class_new_recognizer`] so the two recognizers stay
/// drift-free on what counts as a static top-level constant.
pub(crate) fn is_static_named_receiver(receiver: &Node<'_>, expected: &[u8]) -> bool {
    if let Some(c) = receiver.as_constant_read_node() {
        return c.name().as_slice() == expected;
    }
    if let Some(p) = receiver.as_constant_path_node()
        && p.parent().is_none()
        && let Some(name) = p.name()
    {
        return name.as_slice() == expected;
    }
    false
}
