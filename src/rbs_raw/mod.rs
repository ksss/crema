//! Safe wrapper over ruby-rbs-sys raw FFI bindings.
//!
//! All unsafe code for accessing RBS C parser structures is contained
//! in this module (and the auto-generated rbs_raw_generated.rs).
//! The rest of crema should never need unsafe blocks for RBS access.

use std::marker::PhantomData;

use ruby_rbs_sys::bindings::*;

mod rbs_raw_generated;
pub use rbs_raw_generated::*;


/// Payload view for leading inline annotation nodes.
///
/// The raw annotation structs live in the rbs C arena owned by
/// [`Parser`]. This enum keeps only raw child pointers plus owned
/// parameter names sliced from the inline annotation input.
#[derive(Debug)]
pub enum InlineLeadingAnnotationKind<'a> {
    ColonMethodType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        annotations: RawNodeList<'a>,
        method_type: *const rbs_node,
    },
    MethodTypes {
        location: (u32, u32),
        prefix_location: (u32, u32),
        overloads: RawNodeList<'a>,
        vertical_bar_locations: RawLocationRangeList<'a>,
        dot3_location: Option<(u32, u32)>,
    },
    Skip {
        location: (u32, u32),
        prefix_location: (u32, u32),
        skip_location: (u32, u32),
        comment_location: Option<(u32, u32)>,
    },
    ReturnType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        return_location: (u32, u32),
        colon_location: (u32, u32),
        return_type: *const rbs_node,
        comment_location: Option<(u32, u32)>,
    },
    ParamType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        name_location: (u32, u32),
        colon_location: (u32, u32),
        name: String,
        param_type: *const rbs_node,
        comment_location: Option<(u32, u32)>,
    },
    BlockParamType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        ampersand_location: (u32, u32),
        name_location: Option<(u32, u32)>,
        colon_location: (u32, u32),
        question_location: Option<(u32, u32)>,
        type_location: (u32, u32),
        name: Option<String>,
        type_: *const rbs_node,
        comment_location: Option<(u32, u32)>,
    },
    InstanceVariable {
        location: (u32, u32),
        prefix_location: (u32, u32),
        name_location: (u32, u32),
        colon_location: (u32, u32),
        name: String,
        type_: *const rbs_node,
    },
    SplatParamType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        star_location: (u32, u32),
        name_location: Option<(u32, u32)>,
        colon_location: (u32, u32),
        name: Option<String>,
        param_type: *const rbs_node,
        comment_location: Option<(u32, u32)>,
    },
    DoubleSplatParamType {
        location: (u32, u32),
        prefix_location: (u32, u32),
        star2_location: (u32, u32),
        name_location: Option<(u32, u32)>,
        colon_location: (u32, u32),
        name: Option<String>,
        param_type: *const rbs_node,
        comment_location: Option<(u32, u32)>,
    },
    ModuleSelf {
        location: (u32, u32),
        prefix_location: (u32, u32),
        keyword_location: (u32, u32),
        colon_location: (u32, u32),
        name: *const rbs_type_name,
        name_location: (u32, u32),
        args: RawNodeList<'a>,
    },
    Unsupported(AnnotationKind),
}

/// Payload view for trailing inline annotation nodes.
#[derive(Debug)]
pub enum InlineTrailingAnnotationKind<'a> {
    NodeTypeAssertion { type_node: *const rbs_node },
    TypeApplication { type_args: RawNodeList<'a> },
    Unsupported(AnnotationKind),
}

// Inline annotation parse entry points.
//
// These are declared in crema instead of being consumed from ruby-rbs-sys
// bindgen output, because the functions are not yet in rbs-sys's allowlist.
// The return parameter is typed as `*mut rbs_node` rather than the actual
// `rbs_ast_ruby_annotations_t *` union. This relies on the design invariant
// that every annotation variant begins with `rbs_node_t base` as its first
// field, so the two pointer types alias at the same address. Dispatch by
// `(*node).type_` and cast to the concrete variant struct as needed.
unsafe extern "C" {
    pub fn rbs_parse_inline_leading_annotation(
        parser: *mut rbs_parser_t,
        annotation: *mut *mut rbs_node,
    ) -> bool;

    pub fn rbs_parse_inline_trailing_annotation(
        parser: *mut rbs_parser_t,
        annotation: *mut *mut rbs_node,
    ) -> bool;
}

/// Wrapper for rbs_node_list_t pointer, providing safe iteration.
///
/// The `'a` lifetime is bound to the owning [`Parser`], so the compiler
/// rejects any use that outlives the parser (which would be use-after-free
/// because the C arena is freed on `Parser` drop).
///
/// Valid use — the list is consumed while the parser is alive:
///
/// ```
/// use crema::rbs_raw::Parser;
/// let (parser, sig) = Parser::parse_signature(b"class C end\n").unwrap();
/// let decls = parser.signature_declarations(sig);
/// let _ = decls.iter().next();
/// ```
///
/// Invalid use — leaking the list past the parser fails to compile:
///
/// ```compile_fail
/// use crema::rbs_raw::Parser;
/// let decls = {
///     let (parser, sig) = Parser::parse_signature(b"class C end\n").unwrap();
///     parser.signature_declarations(sig)
/// };
/// // ERROR: `parser` does not live long enough
/// let _ = decls.iter().next();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RawNodeList<'a> {
    ptr: *mut rbs_node_list,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RawNodeList<'a> {
    /// Construct from a raw pointer.
    ///
    /// # Safety
    /// The caller asserts that `ptr` is valid for the lifetime `'a`
    /// (typically the parser that owns the RBS AST arena).
    pub unsafe fn from_raw(ptr: *mut rbs_node_list) -> Self {
        RawNodeList {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn iter(&self) -> RawNodeListIter<'a> {
        if self.ptr.is_null() {
            RawNodeListIter {
                current: std::ptr::null_mut(),
                _marker: PhantomData,
            }
        } else {
            RawNodeListIter {
                current: unsafe { (*self.ptr).head },
                _marker: PhantomData,
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

pub struct RawNodeListIter<'a> {
    current: *mut rbs_node_list_node,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for RawNodeListIter<'a> {
    type Item = *const rbs_node;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            let node = unsafe { (*self.current).node as *const rbs_node };
            self.current = unsafe { (*self.current).next };
            Some(node)
        }
    }
}

/// Wrapper for rbs_location_range_list_t pointer, providing safe iteration.
#[derive(Debug, Clone, Copy)]
pub struct RawLocationRangeList<'a> {
    ptr: *mut rbs_location_range_list_t,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RawLocationRangeList<'a> {
    /// Construct from a raw pointer.
    ///
    /// # Safety
    /// The caller asserts that `ptr` is valid for the lifetime `'a`.
    pub unsafe fn from_raw(ptr: *mut rbs_location_range_list_t) -> Self {
        RawLocationRangeList {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn iter(&self) -> RawLocationRangeListIter<'a> {
        if self.ptr.is_null() {
            RawLocationRangeListIter {
                current: std::ptr::null_mut(),
                _marker: PhantomData,
            }
        } else {
            RawLocationRangeListIter {
                current: unsafe { (*self.ptr).head },
                _marker: PhantomData,
            }
        }
    }
}

pub struct RawLocationRangeListIter<'a> {
    current: *mut rbs_location_range_list_node_t,
    _marker: PhantomData<&'a ()>,
}

impl Iterator for RawLocationRangeListIter<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            let node = unsafe { &*self.current };
            self.current = node.next;
            Some(raw_location_range(node.range))
        }
    }
}

/// Wrapper for rbs_hash_t pointer, providing safe iteration.
///
/// The `'a` lifetime is bound to the owning [`Parser`]; see [`RawNodeList`].
#[derive(Debug, Clone, Copy)]
pub struct RawHash<'a> {
    ptr: *mut rbs_hash,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RawHash<'a> {
    /// Construct from a raw pointer.
    ///
    /// # Safety
    /// The caller asserts that `ptr` is valid for the lifetime `'a`.
    pub unsafe fn from_raw(ptr: *mut rbs_hash) -> Self {
        RawHash {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn iter(&self) -> RawHashIter<'a> {
        if self.ptr.is_null() {
            RawHashIter {
                current: std::ptr::null_mut(),
                _marker: PhantomData,
            }
        } else {
            RawHashIter {
                current: unsafe { (*self.ptr).head },
                _marker: PhantomData,
            }
        }
    }
}

pub struct RawHashIter<'a> {
    current: *mut rbs_hash_node,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for RawHashIter<'a> {
    type Item = (*const rbs_node, *const rbs_node);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            let key = unsafe { (*self.current).key as *const rbs_node };
            let value = unsafe { (*self.current).value as *const rbs_node };
            self.current = unsafe { (*self.current).next };
            Some((key, value))
        }
    }
}

/// Wrapper for rbs_string_t, providing safe access.
///
/// The `'a` lifetime is bound to the owning [`Parser`].
#[derive(Debug, Clone, Copy)]
pub struct RawString<'a> {
    raw: rbs_string_t,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RawString<'a> {
    /// Construct from a raw `rbs_string_t`.
    ///
    /// # Safety
    /// The caller asserts that the backing bytes are valid for the lifetime `'a`.
    pub unsafe fn from_raw(raw: rbs_string_t) -> Self {
        RawString {
            raw,
            _marker: PhantomData,
        }
    }

    pub fn as_str(&self) -> &'a str {
        unsafe {
            let start = self.raw.start as *const u8;
            let end = self.raw.end as *const u8;
            let len = end.offset_from(start) as usize;
            let bytes = std::slice::from_raw_parts(start, len);
            std::str::from_utf8_unchecked(bytes)
        }
    }
}

/// RBS parser wrapper. Owns the parser and frees it on drop.
pub struct Parser {
    parser: *mut rbs_parser_t,
    // Keep source alive as long as parser lives
    _source: Vec<u8>,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl Parser {
    /// Borrow the source bytes this parser was built from.
    ///
    /// Used by ast_builder helpers that need to read location-range
    /// slices (e.g. extracting a `@rbs name: T` param name out of its
    /// source location). The slice lives for as long as `self`.
    pub fn source_bytes(&self) -> &[u8] {
        &self._source
    }

    /// Parse an RBS signature string.
    /// Returns the parser and a pointer to the parsed signature.
    pub fn parse_signature(source: &[u8]) -> Result<(Self, *mut rbs_signature_t), String> {
        let source_vec = source.to_vec();
        let start_ptr = source_vec.as_ptr() as *const std::os::raw::c_char;
        let end_ptr = unsafe { start_ptr.add(source_vec.len()) } as *const std::os::raw::c_char;

        let rbs_string = unsafe { rbs_string_new(start_ptr, end_ptr) };
        let encoding_ptr = unsafe {
            &rbs_encodings[rbs_encoding_type_t::RBS_ENCODING_UTF_8 as usize]
                as *const rbs_encoding_t
        };
        let parser =
            unsafe { rbs_parser_new(rbs_string, encoding_ptr, 0, source_vec.len() as i32) };

        if parser.is_null() {
            return Err("Failed to create RBS parser".to_string());
        }

        let mut signature: *mut rbs_signature_t = std::ptr::null_mut();
        let success = unsafe { rbs_parse_signature(parser, &mut signature) };

        if !success || unsafe { !(*parser).error.is_null() } {
            let error_msg = unsafe {
                if (*parser).error.is_null() {
                    "Unknown parse error".to_string()
                } else {
                    // Read error message from parser (C string)
                    let err = &*(*parser).error;
                    std::ffi::CStr::from_ptr(err.message)
                        .to_string_lossy()
                        .to_string()
                }
            };
            unsafe { rbs_parser_free(parser) };
            return Err(error_msg);
        }

        Ok((
            Parser {
                parser,
                _source: source_vec,
            },
            signature,
        ))
    }

    /// Resolve a constant pool symbol to a string.
    pub fn resolve_constant(&self, symbol: *const rbs_ast_symbol) -> &str {
        unsafe { self.resolve_constant_id((*symbol).constant_id) }
    }

    pub fn resolve_constant_id(&self, constant_id: rbs_constant_id_t) -> &str {
        unsafe {
            let constant = rbs_constant_pool_id_to_constant(
                &(*self.parser).constant_pool as *const rbs_constant_pool_t
                    as *mut rbs_constant_pool_t,
                constant_id,
            );
            let bytes = std::slice::from_raw_parts((*constant).start, (*constant).length as usize);
            std::str::from_utf8_unchecked(bytes)
        }
    }

    /// Convert a namespace pointer to a qualified string (e.g., "::Foo::Bar::"
    /// or "Foo::"). The trailing `::` reflects rbs's `Namespace#to_s`
    /// convention; a namespace always ends with the separator even
    /// when empty (the empty absolute namespace is "::"). A null
    /// pointer maps to the empty relative namespace.
    pub fn namespace_to_string(&self, ns: *const rbs_namespace) -> String {
        unsafe {
            if ns.is_null() {
                return String::new();
            }
            let ns_ref = &*ns;
            let path_list = RawNodeList::from_raw(ns_ref.path);
            let mut parts: Vec<String> = Vec::new();
            for node in path_list.iter() {
                if (*node).type_ == rbs_node_type::RBS_AST_SYMBOL {
                    let sym = node as *const rbs_ast_symbol;
                    parts.push(self.resolve_constant(sym).to_string());
                }
            }
            let joined = if parts.is_empty() {
                String::new()
            } else {
                format!("{}::", parts.join("::"))
            };
            if ns_ref.absolute {
                format!("::{}", joined)
            } else {
                joined
            }
        }
    }

    /// Convert a type_name pointer to a qualified string (e.g., "::Foo::Bar").
    pub fn type_name_to_string(&self, type_name: *const rbs_type_name) -> String {
        unsafe {
            let tn = &*type_name;
            let ns = tn.rbs_namespace;
            let name = self.resolve_constant(tn.name as *const rbs_ast_symbol);

            let mut parts: Vec<String> = Vec::new();

            if !ns.is_null() {
                let ns_ref = &*ns;
                let path_list = RawNodeList::from_raw(ns_ref.path);
                for node in path_list.iter() {
                    if (*node).type_ == rbs_node_type::RBS_AST_SYMBOL {
                        let sym = node as *const rbs_ast_symbol;
                        parts.push(self.resolve_constant(sym).to_string());
                    }
                }
            }

            parts.push(name.to_string());

            let absolute = !ns.is_null() && (*ns).absolute;
            if absolute {
                format!("::{}", parts.join("::"))
            } else {
                parts.join("::")
            }
        }
    }

    /// Get declarations list from a parsed signature.
    ///
    /// The returned `RawNodeList` borrows from `self` so the compiler
    /// rejects any use that outlives this `Parser`.
    pub fn signature_declarations(&self, sig: *mut rbs_signature_t) -> RawNodeList<'_> {
        unsafe { RawNodeList::from_raw((*sig).declarations) }
    }

    /// Top-level `use ...` (and friends) directives carried at the
    /// signature root, parallel to [`signature_declarations`]. Mirrors
    /// `RBS::Source::RBS#directives`.
    pub fn signature_directives(&self, sig: *mut rbs_signature_t) -> RawNodeList<'_> {
        unsafe { RawNodeList::from_raw((*sig).directives) }
    }

    /// Try to read a node as a symbol string. Returns None if not a symbol node.
    pub fn node_as_symbol(&self, node: *const rbs_node) -> Option<&str> {
        unsafe {
            if (*node).type_ == rbs_node_type::RBS_AST_SYMBOL {
                Some(self.resolve_constant(node as *const rbs_ast_symbol))
            } else {
                None
            }
        }
    }

    /// Classify a leading inline annotation and expose the fields needed
    /// by the unresolved AST builder.
    ///
    /// `source` must be the exact byte string passed to
    /// [`Parser::parse_inline_leading`]. rbs stores param names as
    /// location ranges, not symbol nodes, so this method slices them from
    /// that source.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn classify_inline_leading_annotation<'a>(
        &'a self,
        node: *const rbs_node,
        source: &[u8],
    ) -> InlineLeadingAnnotationKind<'a> {
        match classify_annotation(self, node) {
            AnnotationKind::ColonMethodTypeAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_colon_method_type_annotation_t)
                };
                InlineLeadingAnnotationKind::ColonMethodType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    annotations: unsafe { RawNodeList::from_raw(n.annotations) },
                    method_type: n.method_type as *const rbs_node,
                }
            }
            AnnotationKind::MethodTypesAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_method_types_annotation_t)
                };
                InlineLeadingAnnotationKind::MethodTypes {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    overloads: unsafe { RawNodeList::from_raw(n.overloads) },
                    vertical_bar_locations: unsafe {
                        RawLocationRangeList::from_raw(n.vertical_bar_locations)
                    },
                    dot3_location: non_empty_raw_location_range(n.dot3_location),
                }
            }
            AnnotationKind::SkipAnnotation => {
                let n = unsafe { &*(node as *const rbs_ast_ruby_annotations_skip_annotation_t) };
                InlineLeadingAnnotationKind::Skip {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    skip_location: raw_location_range(n.skip_location),
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::ReturnTypeAnnotation => {
                let n =
                    unsafe { &*(node as *const rbs_ast_ruby_annotations_return_type_annotation_t) };
                InlineLeadingAnnotationKind::ReturnType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    return_location: raw_location_range(n.return_location),
                    colon_location: raw_location_range(n.colon_location),
                    return_type: n.return_type as *const rbs_node,
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::ParamTypeAnnotation => {
                let n =
                    unsafe { &*(node as *const rbs_ast_ruby_annotations_param_type_annotation_t) };
                InlineLeadingAnnotationKind::ParamType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    name_location: raw_location_range(n.name_location),
                    colon_location: raw_location_range(n.colon_location),
                    name: location_source(source, n.name_location).unwrap_or_default(),
                    param_type: n.param_type as *const rbs_node,
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::InstanceVariableAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_instance_variable_annotation_t)
                };
                InlineLeadingAnnotationKind::InstanceVariable {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    name_location: raw_location_range(n.ivar_name_location),
                    colon_location: raw_location_range(n.colon_location),
                    name: location_source(source, n.ivar_name_location).unwrap_or_default(),
                    type_: n.type_ as *const rbs_node,
                }
            }
            AnnotationKind::BlockParamTypeAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_block_param_type_annotation_t)
                };
                InlineLeadingAnnotationKind::BlockParamType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    ampersand_location: raw_location_range(n.ampersand_location),
                    name_location: non_empty_raw_location_range(n.name_location),
                    colon_location: raw_location_range(n.colon_location),
                    question_location: non_empty_raw_location_range(n.question_location),
                    type_location: raw_location_range(n.type_location),
                    name: location_source(source, n.name_location),
                    type_: n.type_ as *const rbs_node,
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::SplatParamTypeAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_splat_param_type_annotation_t)
                };
                InlineLeadingAnnotationKind::SplatParamType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    star_location: raw_location_range(n.star_location),
                    name_location: non_empty_raw_location_range(n.name_location),
                    colon_location: raw_location_range(n.colon_location),
                    name: location_source(source, n.name_location),
                    param_type: n.param_type as *const rbs_node,
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::DoubleSplatParamTypeAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_double_splat_param_type_annotation_t)
                };
                InlineLeadingAnnotationKind::DoubleSplatParamType {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    star2_location: raw_location_range(n.star2_location),
                    name_location: non_empty_raw_location_range(n.name_location),
                    colon_location: raw_location_range(n.colon_location),
                    name: location_source(source, n.name_location),
                    param_type: n.param_type as *const rbs_node,
                    comment_location: non_empty_raw_location_range(n.comment_location),
                }
            }
            AnnotationKind::ModuleSelfAnnotation => {
                let n =
                    unsafe { &*(node as *const rbs_ast_ruby_annotations_module_self_annotation_t) };
                let name_ptr = n.name as *const rbs_type_name;
                let name_location = unsafe { raw_location_range((*name_ptr).base.location) };
                InlineLeadingAnnotationKind::ModuleSelf {
                    location: raw_location_range(n.base.location),
                    prefix_location: raw_location_range(n.prefix_location),
                    keyword_location: raw_location_range(n.keyword_location),
                    colon_location: raw_location_range(n.colon_location),
                    name: name_ptr,
                    name_location,
                    args: unsafe { RawNodeList::from_raw(n.args) },
                }
            }
            other => InlineLeadingAnnotationKind::Unsupported(other),
        }
    }

    /// Classify a trailing inline annotation and expose the fields
    /// needed by the unresolved AST builder.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn classify_inline_trailing_annotation<'a>(
        &'a self,
        node: *const rbs_node,
    ) -> InlineTrailingAnnotationKind<'a> {
        match classify_annotation(self, node) {
            AnnotationKind::NodeTypeAssertion => {
                let n =
                    unsafe { &*(node as *const rbs_ast_ruby_annotations_node_type_assertion_t) };
                InlineTrailingAnnotationKind::NodeTypeAssertion {
                    type_node: n.type_ as *const rbs_node,
                }
            }
            AnnotationKind::TypeApplicationAnnotation => {
                let n = unsafe {
                    &*(node as *const rbs_ast_ruby_annotations_type_application_annotation_t)
                };
                InlineTrailingAnnotationKind::TypeApplication {
                    type_args: unsafe { RawNodeList::from_raw(n.type_args) },
                }
            }
            other => InlineTrailingAnnotationKind::Unsupported(other),
        }
    }

    /// Get the type name from a ClassSuper node.
    pub fn class_super_type_name(
        &self,
        super_class: *const rbs_ast_declarations_class_super,
    ) -> String {
        unsafe { self.type_name_to_string((*super_class).name as *const rbs_type_name) }
    }

    /// Read an Integer node's string representation.
    pub fn integer_string_repr(&self, node: *const rbs_node) -> Option<String> {
        unsafe {
            if (*node).type_ == rbs_node_type::RBS_AST_INTEGER {
                let int_node = &*(node as *const rbs_ast_integer);
                Some(
                    RawString::from_raw(int_node.string_representation)
                        .as_str()
                        .to_string(),
                )
            } else {
                None
            }
        }
    }

    /// Read a String node's string value.
    pub fn string_value(&self, node: *const rbs_node) -> Option<String> {
        unsafe {
            if (*node).type_ == rbs_node_type::RBS_AST_STRING {
                let str_node = &*(node as *const rbs_ast_string);
                Some(RawString::from_raw(str_node.string).as_str().to_string())
            } else {
                None
            }
        }
    }

    /// Read a Bool node's value.
    pub fn bool_value(&self, node: *const rbs_node) -> Option<bool> {
        unsafe {
            if (*node).type_ == rbs_node_type::RBS_AST_BOOL {
                let bool_node = &*(node as *const rbs_ast_bool);
                Some(bool_node.value)
            } else {
                None
            }
        }
    }

    /// Read a Symbol node's bytes (for Symbol literals).
    pub fn symbol_bytes(&self, node: *const rbs_node) -> Option<Vec<u8>> {
        unsafe {
            if (*node).type_ == rbs_node_type::RBS_AST_SYMBOL {
                let sym = node as *const rbs_ast_symbol;
                let s = self.resolve_constant(sym);
                Some(s.as_bytes().to_vec())
            } else {
                None
            }
        }
    }

    /// Classify a block pointer as TypeKind::Block by casting to node.
    pub fn classify_block(&self, block: *const rbs_types_block) -> TypeKind<'_> {
        classify_type(self, block as *const rbs_node)
    }

    /// Build a `RubyLocation` from an RBS node's base location and a file name.
    ///
    /// Returns `None` when `file` is `None` or the node has an RBS null range
    /// (`start_char == -1`). This is the translation layer that pairs the
    /// parser's byte offsets with a file identity — analogous to
    /// `rbs_translation_context_t` in `ext/rbs_extension/ast_translation.c`.
    pub fn node_location(
        node: *const rbs_node,
        file: Option<crate::name::Name>,
    ) -> Option<crate::location::RubyLocation> {
        let file = file?;
        let range = unsafe { (*node).location };
        if range.start_char < 0 {
            return None;
        }
        Some(crate::location::RubyLocation {
            file,
            start_byte: range.start_byte as u32,
            end_byte: range.end_byte as u32,
        })
    }

    /// Parse an RBS type string (e.g. "String", "Array[Integer]").
    /// Returns the parser and a pointer to the parsed type node.
    ///
    /// Internally wraps the source in a trailing type-assertion annotation
    /// (`": <source>"`) and unwraps the inner type. This keeps the function
    /// working against rbs's inline parser entry points, which are already
    /// in the bindgen allowlist; `rbs_parse_type` was dropped from the
    /// allowlist once inline annotation parsing covered every call site.
    pub fn parse_type(source: &[u8]) -> Result<(Self, *const rbs_node), String> {
        let mut wrapped = Vec::with_capacity(source.len() + 2);
        wrapped.extend_from_slice(b": ");
        wrapped.extend_from_slice(source);

        let (parser, annotation_node) = Self::parse_inline_trailing(&wrapped)?;

        let type_node = unsafe {
            if (*annotation_node).type_
                != rbs_node_type::RBS_AST_RUBY_ANNOTATIONS_NODE_TYPE_ASSERTION
            {
                return Err("expected type assertion annotation".to_string());
            }
            let assertion =
                annotation_node as *const rbs_ast_ruby_annotations_node_type_assertion_t;
            (*assertion).type_ as *const rbs_node
        };
        Ok((parser, type_node))
    }

    /// Parse an inline leading annotation (body without the `#` prefix).
    ///
    /// Accepts the comment body stripped of the leading `#`, e.g.
    /// `": (String) -> Integer"` or `"@rbs foo: String"` or `"@rbs skip"`.
    /// The returned node's `type_` discriminates the variant:
    ///
    /// - `RBS_AST_RUBY_ANNOTATIONS_COLON_METHOD_TYPE_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_METHOD_TYPES_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_PARAM_TYPE_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_RETURN_TYPE_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_SKIP_ANNOTATION`
    /// - and other `@rbs` doc-style variants
    pub fn parse_inline_leading(source: &[u8]) -> Result<(Self, *const rbs_node), String> {
        Self::parse_inline_impl(source, true)
    }

    /// Parse an inline trailing annotation (body without the `#` prefix).
    ///
    /// Accepts the comment body stripped of the leading `#`, e.g.
    /// `": String"` for a type assertion or `"[String, Integer]"` for a
    /// type application. The returned node's `type_` discriminates:
    ///
    /// - `RBS_AST_RUBY_ANNOTATIONS_NODE_TYPE_ASSERTION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_TYPE_APPLICATION_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_COLON_METHOD_TYPE_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_CLASS_ALIAS_ANNOTATION`
    /// - `RBS_AST_RUBY_ANNOTATIONS_MODULE_ALIAS_ANNOTATION`
    pub fn parse_inline_trailing(source: &[u8]) -> Result<(Self, *const rbs_node), String> {
        Self::parse_inline_impl(source, false)
    }

    fn parse_inline_impl(source: &[u8], leading: bool) -> Result<(Self, *const rbs_node), String> {
        let source_vec = source.to_vec();
        let start_ptr = source_vec.as_ptr() as *const std::os::raw::c_char;
        let end_ptr = unsafe { start_ptr.add(source_vec.len()) } as *const std::os::raw::c_char;

        let rbs_string = unsafe { rbs_string_new(start_ptr, end_ptr) };
        let encoding_ptr = unsafe {
            &rbs_encodings[rbs_encoding_type_t::RBS_ENCODING_UTF_8 as usize]
                as *const rbs_encoding_t
        };
        let parser =
            unsafe { rbs_parser_new(rbs_string, encoding_ptr, 0, source_vec.len() as i32) };

        if parser.is_null() {
            return Err("Failed to create RBS parser".to_string());
        }

        let mut node: *mut rbs_node = std::ptr::null_mut();
        let success = unsafe {
            if leading {
                rbs_parse_inline_leading_annotation(parser, &mut node)
            } else {
                rbs_parse_inline_trailing_annotation(parser, &mut node)
            }
        };

        if !success || node.is_null() || unsafe { !(*parser).error.is_null() } {
            let error_msg = unsafe {
                if (*parser).error.is_null() {
                    "Failed to parse inline annotation".to_string()
                } else {
                    let err = &*(*parser).error;
                    std::ffi::CStr::from_ptr(err.message)
                        .to_string_lossy()
                        .to_string()
                }
            };
            unsafe { rbs_parser_free(parser) };
            return Err(error_msg);
        }

        Ok((
            Parser {
                parser,
                _source: source_vec,
            },
            node as *const rbs_node,
        ))
    }
}

impl Drop for Parser {
    fn drop(&mut self) {
        if !self.parser.is_null() {
            unsafe { rbs_parser_free(self.parser) };
        }
    }
}

fn location_source(source: &[u8], location: rbs_location_range) -> Option<String> {
    let start = usize::try_from(location.start_byte).ok()?;
    let end = usize::try_from(location.end_byte).ok()?;
    if start >= end || end > source.len() {
        return None;
    }
    std::str::from_utf8(source.get(start..end)?)
        .ok()
        .map(str::to_string)
}

fn raw_location_range(location: rbs_location_range) -> (u32, u32) {
    (
        u32::try_from(location.start_byte).unwrap_or(0),
        u32::try_from(location.end_byte).unwrap_or(0),
    )
}

fn non_empty_raw_location_range(location: rbs_location_range) -> Option<(u32, u32)> {
    if location.start_byte < location.end_byte {
        Some(raw_location_range(location))
    } else {
        None
    }
}

pub(crate) fn ffi_range_to_location_range(
    location: rbs_location_range,
) -> crate::location::LocationRange {
    crate::location::LocationRange {
        start_char: u32::try_from(location.start_char).unwrap_or(0),
        start_byte: u32::try_from(location.start_byte).unwrap_or(0),
        end_char: u32::try_from(location.end_char).unwrap_or(0),
        end_byte: u32::try_from(location.end_byte).unwrap_or(0),
    }
}

pub(crate) fn ffi_range_to_optional_location_range(
    location: rbs_location_range,
) -> Option<crate::location::LocationRange> {
    if location.start_byte < location.end_byte {
        Some(ffi_range_to_location_range(location))
    } else {
        None
    }
}

pub(crate) fn node_location_range(node: *const rbs_node) -> Option<crate::location::LocationRange> {
    let range = unsafe { (*node).location };
    if range.start_char < 0 {
        return None;
    }
    Some(ffi_range_to_location_range(range))
}
