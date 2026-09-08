//! Build AST layer — converts `rbs_raw` wrapper nodes into [`crate::ast`]
//! values without touching any [`crate::environment::DefinitionBuilder`].
//!
//! In rbs's 5-phase pipeline (parse → build AST → collect → resolve →
//! check) this module is the "build AST" phase. The actual parse is done
//! by `ruby-rbs-sys`; crema never spells "parse" in its own module names.
//!
//! Phase 2 of ADR-0014. Downstream `definition_builder` code takes these
//! pure `ast::*` values plus an environment to produce resolved `Ty` /
//! `MethodType` values.

use crate::ast::annotation::Annotation;
use crate::ast::members::MethodDefinitionOverload;
use crate::ast::method_type::MethodType;
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::annotations::{
    BlockParamTypeAnnotation, ColonMethodTypeAnnotation, DoubleSplatParamTypeAnnotation,
    InstanceVariableAnnotation, LeadingAnnotation, MethodTypesAnnotation, ModuleSelfAnnotation,
    ParamTypeAnnotation, ReturnTypeAnnotation, SkipAnnotation, SplatParamTypeAnnotation,
    TypeApplicationAnnotation,
};
use crate::ast::types::{
    AliasType, BaseType, BaseTypeKind, BlockType, ClassInstanceType, ClassSingletonType, Function,
    FunctionParam, FunctionType, InterfaceType, IntersectionType, KeywordParam, Literal,
    LiteralType, OptionalType, ProcType, RecordField, RecordKey, RecordType, TupleType, Type,
    UnionType, UntypedFunctionType, VariableType,
};
use crate::ast::{TypeParam, Variance};
use crate::name::{Name, NameTable};
use crate::rbs_raw::{
    self, InlineLeadingAnnotationKind, InlineTrailingAnnotationKind, MemberKind, Parser,
    RawNodeList, StructuralKind, TypeKind, classify_member, classify_structural, classify_type,
};


/// Build an unresolved [`Type`] AST node from an `rbs_raw` node.
///
/// Returns `None` when the source shape cannot be represented in the
/// current ast layer (for example a record whose key is a non-literal
/// expression). Callers in this module fill such slots with `Ty::UNTYPED`
/// — a silent subtype of everything at runtime.
pub fn build_type(
    parser: &Parser,
    node: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Option<Type> {
    match classify_type(parser, node) {
        TypeKind::ClassInstance { name, args } => {
            let raw = parser.type_name_to_string(name);
            let type_name = names.parse_type_name(&raw);
            Some(Type::ClassInstance(ClassInstanceType {
                name: type_name,
                args: build_type_list(parser, args, names),
                location: None,
            }))
        }
        TypeKind::Interface { name, args } => {
            let raw = parser.type_name_to_string(name);
            let type_name = names.parse_type_name(&raw);
            Some(Type::Interface(InterfaceType {
                name: type_name,
                args: build_type_list(parser, args, names),
                location: None,
            }))
        }
        TypeKind::Alias { name, args } => {
            let raw = parser.type_name_to_string(name);
            let type_name = names.parse_type_name(&raw);
            Some(Type::Alias(AliasType {
                name: type_name,
                args: build_type_list(parser, args, names),
                location: None,
            }))
        }
        TypeKind::ClassSingleton { name, args } => {
            let name_str = parser.type_name_to_string(name);
            let type_name = names.parse_type_name(&name_str);
            Some(Type::ClassSingleton(ClassSingletonType {
                name: type_name,
                args: build_type_list(parser, args, names),
                location: None,
            }))
        }
        TypeKind::Variable { name } => {
            let raw = parser.resolve_constant(name).to_string();
            Some(Type::Variable(VariableType {
                name: names.intern_symbol(&raw),
                location: None,
            }))
        }
        TypeKind::Union { types } => {
            let members = build_type_list(parser, types, names);
            if members.len() == 1 {
                // rbs parses `A | A` as a single-member Union; fold it to
                // the inner type to match the resolved-side invariant.
                members.into_iter().next()
            } else {
                Some(Type::Union(UnionType {
                    types: members,
                    location: None,
                }))
            }
        }
        TypeKind::Intersection { types } => {
            let members = build_type_list(parser, types, names);
            if members.len() == 1 {
                members.into_iter().next()
            } else {
                Some(Type::Intersection(IntersectionType {
                    types: members,
                    location: None,
                }))
            }
        }
        TypeKind::Optional { type_: inner } => build_type(parser, inner, names).map(|t| {
            Type::Optional(OptionalType {
                ty: Box::new(t),
                location: None,
            })
        }),
        TypeKind::Tuple { types } => Some(Type::Tuple(TupleType {
            types: build_type_list(parser, types, names),
            location: None,
        })),
        TypeKind::Record { all_fields } => {
            let mut fields: Vec<RecordField> = Vec::new();
            for (key_node, value_node) in all_fields.iter() {
                let key = build_record_key(parser, key_node, names)?;
                let TypeKind::RecordField {
                    type_: field_type_ptr,
                    required,
                } = classify_type(parser, value_node)
                else {
                    return None;
                };
                let field_ty = build_type(parser, field_type_ptr, names)?;
                fields.push(RecordField {
                    key,
                    ty: field_ty,
                    required,
                });
            }
            // Sort so structurally equal records compare equal regardless
            // of declaration order; matches the resolved-side invariant.
            fields.sort_by(|a, b| a.key.cmp(&b.key));
            Some(Type::Record(RecordType {
                fields,
                location: None,
            }))
        }
        TypeKind::Proc {
            type_: func_ptr,
            block: block_opt,
            self_type: self_type_opt,
        } => {
            let function = build_function_type(parser, func_ptr, names);
            let self_type = self_type_opt.map(|st| {
                Box::new(
                    build_type(parser, st, names).unwrap_or(Type::Base(BaseType {
                        kind: BaseTypeKind::Any { todo: false },
                        location: None,
                    })),
                )
            });
            let block = block_opt.and_then(|bp| build_block(parser, bp, names));
            Some(Type::Proc(Box::new(ProcType {
                function,
                block,
                self_type,
                location: None,
            })))
        }
        TypeKind::Literal { literal } => {
            if let Some(s) = parser.integer_string_repr(literal) {
                Some(Type::Literal(LiteralType {
                    literal: Literal::Integer(s),
                    location: None,
                }))
            } else if let Some(s) = parser.string_value(literal) {
                Some(Type::Literal(LiteralType {
                    literal: Literal::String(s),
                    location: None,
                }))
            } else if let Some(bytes) = parser.symbol_bytes(literal) {
                let s = String::from_utf8_lossy(&bytes).to_string();
                Some(Type::Literal(LiteralType {
                    literal: Literal::Symbol(names.intern_symbol(&s)),
                    location: None,
                }))
            } else {
                parser.bool_value(literal).map(|b| {
                    Type::Literal(LiteralType {
                        literal: Literal::Bool(b),
                        location: None,
                    })
                })
            }
        }
        TypeKind::Bool => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Bool,
            location: None,
        })),
        TypeKind::Void => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Void,
            location: None,
        })),
        TypeKind::Nil => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Nil,
            location: None,
        })),
        TypeKind::Any { .. } => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        })),
        TypeKind::Top => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Top,
            location: None,
        })),
        TypeKind::Bottom => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Bottom,
            location: None,
        })),
        TypeKind::SelfType => Some(Type::Base(BaseType {
            kind: BaseTypeKind::SelfType,
            location: None,
        })),
        TypeKind::Instance => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Instance,
            location: None,
        })),
        TypeKind::Class => Some(Type::Base(BaseType {
            kind: BaseTypeKind::Class,
            location: None,
        })),
        _ => None,
    }
}

/// Build an unresolved [`MethodType`] from an `rbs_method_type_t` node.
///
/// Returns `None` when the node is not a well-formed method type (for
/// example missing the structural `Method` shape).
pub fn build_method_type(
    parser: &Parser,
    mt_ptr: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Option<MethodType> {
    let StructuralKind::Method {
        type_params: tp_list,
        type_: func_ptr,
        block: block_opt,
    } = classify_structural(parser, mt_ptr)
    else {
        return None;
    };

    let type_params = tp_list
        .iter()
        .filter_map(|tp| build_type_param(parser, tp, names))
        .collect();

    let function = build_function_type(parser, func_ptr, names);
    let block = block_opt.and_then(|bp| build_block(parser, bp, names));

    Some(MethodType {
        type_params,
        function,
        block,
        location: None,
    })
}

/// Build the `[T, U]` list of unresolved type params on a method type.
///
/// Callers handle the resolution side (alpha-renaming to method scope,
/// merging with the enclosing class scope). Here we only translate
/// syntactic shape plus optional bounds.
fn build_type_param(
    parser: &Parser,
    tp_node: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Option<TypeParam> {
    let StructuralKind::TypeParam {
        name,
        upper_bound,
        lower_bound,
        ..
    } = classify_structural(parser, tp_node)
    else {
        return None;
    };

    let name_sym = names.intern_symbol(parser.resolve_constant(name));
    // A bound that is syntactically present but whose body isn't
    // representable in the ast layer must stay as `Some(_)` — not
    // collapse back to `None`. Otherwise the resolved MethodType loses
    // the fact that the user wrote a bound at all.
    let upper = upper_bound.map(|n| {
        build_type(parser, n, names).unwrap_or(Type::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        }))
    });
    let lower = lower_bound.map(|n| {
        build_type(parser, n, names).unwrap_or(Type::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        }))
    });

    Some(TypeParam {
        name: name_sym,
        variance: Variance::Invariant,
        upper_bound: upper,
        lower_bound: lower,
        default_type: None,
        unchecked: false,
        location: None,
    })
}

/// Convert an rbs_raw `RawNodeList` of `%a{...}` annotation nodes into
/// the AST `Vec<Annotation>` carried by rbs AST ports.
///
/// Source order is preserved. The rbs C parser only ever places
/// Annotation nodes in these lists; a non-Annotation variant means
/// rbs_raw's `classify_structural` has drifted out of sync with the
/// rbs C schema. We trip a `debug_assert` so test builds catch that
/// drift, and silently skip in release so a checker run never crashes
/// on unexpected input.
pub fn build_annotations(
    parser: &Parser,
    list: RawNodeList<'_>,
    _file: Option<Name>,
    names: &NameTable,
) -> Vec<Annotation> {
    list.iter()
        .filter_map(|node| match classify_structural(parser, node) {
            StructuralKind::Annotation { string } => Some(Annotation {
                string: names.intern_symbol(string.as_str()),
                location: None,
            }),
            _ => {
                debug_assert!(
                    false,
                    "build_annotations: rbs_raw delivered a non-Annotation node \
                     in an annotations list; classify_structural is out of sync \
                     with the rbs C schema",
                );
                None
            }
        })
        .collect()
}

/// Iterate raw overload children and build each [`Overload`] (method
/// type plus per-overload annotations).
///
/// `file` lets each annotation location carry the source file.
pub fn build_overloads(
    parser: &Parser,
    overloads_list: RawNodeList,
    file: Option<Name>,
    names: &NameTable,
) -> Vec<MethodDefinitionOverload> {
    overloads_list
        .iter()
        .filter_map(|overload_ptr| {
            let MemberKind::MethodDefinitionOverload {
                method_type: mt_ptr,
                annotations,
            } = classify_member(parser, overload_ptr)
            else {
                return None;
            };
            let method_type = build_method_type(parser, mt_ptr, names)?;
            let annotations = build_annotations(parser, annotations, file, names);
            Some(MethodDefinitionOverload {
                method_type,
                annotations,
            })
        })
        .collect()
}

/// Build the unresolved AST representation of a leading inline
/// annotation (`#:` / `# @rbs ...`).
///
/// `source` must be the exact input passed to
/// [`Parser::parse_inline_leading`], because param names are stored in
/// the raw rbs node as byte ranges into that input.
pub fn build_leading_annotation(
    parser: &Parser,
    node: *const ruby_rbs_sys::bindings::rbs_node,
    source: &[u8],
    names: &NameTable,
) -> Option<LeadingAnnotation> {
    match parser.classify_inline_leading_annotation(node, source) {
        InlineLeadingAnnotationKind::ColonMethodType {
            location,
            prefix_location,
            annotations,
            method_type,
        } => build_method_type(parser, method_type, names).map(|method_type| {
            LeadingAnnotation::ColonMethodType(Box::new(ColonMethodTypeAnnotation {
                location,
                prefix_location,
                annotations: build_annotations(parser, annotations, None, names),
                method_type,
            }))
        }),
        InlineLeadingAnnotationKind::MethodTypes {
            location,
            prefix_location,
            overloads,
            vertical_bar_locations,
            dot3_location,
        } => Some(LeadingAnnotation::MethodTypes(MethodTypesAnnotation {
            location,
            prefix_location,
            overloads: build_overloads(parser, overloads, None, names),
            vertical_bar_locations: vertical_bar_locations.iter().collect(),
            dot3_location,
        })),
        InlineLeadingAnnotationKind::Skip {
            location,
            prefix_location,
            skip_location,
            comment_location,
        } => Some(LeadingAnnotation::Skip(SkipAnnotation {
            location,
            prefix_location,
            skip_location,
            comment_location,
        })),
        InlineLeadingAnnotationKind::ReturnType {
            location,
            prefix_location,
            return_location,
            colon_location,
            return_type,
            comment_location,
        } => {
            let return_type =
                build_type(parser, return_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));
            Some(LeadingAnnotation::ReturnType(ReturnTypeAnnotation {
                location,
                prefix_location,
                return_location,
                colon_location,
                return_type,
                comment_location,
            }))
        }
        InlineLeadingAnnotationKind::ParamType {
            location,
            prefix_location,
            name_location,
            colon_location,
            name,
            param_type,
            comment_location,
        } => {
            let param_type =
                build_type(parser, param_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));
            Some(LeadingAnnotation::ParamType(ParamTypeAnnotation {
                location,
                prefix_location,
                name_location,
                colon_location,
                name,
                param_type,
                comment_location,
            }))
        }
        InlineLeadingAnnotationKind::InstanceVariable {
            location,
            name_location,
            name,
            type_,
            ..
        } => {
            let ty = build_type(parser, type_, names).unwrap_or(Type::Base(BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            }));
            Some(LeadingAnnotation::InstanceVariable(
                InstanceVariableAnnotation {
                    name,
                    location,
                    name_location,
                    source_location: None,
                    ty,
                },
            ))
        }
        InlineLeadingAnnotationKind::BlockParamType {
            location,
            prefix_location,
            ampersand_location,
            name_location,
            colon_location,
            question_location,
            type_location,
            name,
            type_,
            comment_location,
        } => Some(LeadingAnnotation::BlockParamType(
            BlockParamTypeAnnotation {
                location,
                prefix_location,
                ampersand_location,
                name_location,
                colon_location,
                question_location,
                type_location,
                name,
                function: build_function_type(parser, type_, names),
                comment_location,
            },
        )),
        InlineLeadingAnnotationKind::SplatParamType {
            location,
            prefix_location,
            star_location,
            name_location,
            colon_location,
            name,
            param_type,
            comment_location,
        } => {
            let param_type =
                build_type(parser, param_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));
            Some(LeadingAnnotation::SplatParamType(
                SplatParamTypeAnnotation {
                    location,
                    prefix_location,
                    star_location,
                    name_location,
                    colon_location,
                    name,
                    param_type,
                    comment_location,
                },
            ))
        }
        InlineLeadingAnnotationKind::DoubleSplatParamType {
            location,
            prefix_location,
            star2_location,
            name_location,
            colon_location,
            name,
            param_type,
            comment_location,
        } => {
            let param_type =
                build_type(parser, param_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));
            Some(LeadingAnnotation::DoubleSplatParamType(
                DoubleSplatParamTypeAnnotation {
                    location,
                    prefix_location,
                    star2_location,
                    name_location,
                    colon_location,
                    name,
                    param_type,
                    comment_location,
                },
            ))
        }
        InlineLeadingAnnotationKind::ModuleSelf {
            location,
            name,
            name_location,
            args,
            ..
        } => {
            let raw = parser.type_name_to_string(name);
            let type_name = names.parse_type_name(&raw);
            Some(LeadingAnnotation::ModuleSelf(ModuleSelfAnnotation {
                name: type_name,
                name_location,
                args: build_type_list(parser, args, names),
                location,
            }))
        }
        InlineLeadingAnnotationKind::Unsupported(_) => None,
    }
}

/// Build the unresolved AST representation of a trailing type
/// application annotation (`#[T]` / `#[T, U]`). `location` is the byte
/// range of the whole annotation comment in the surrounding Ruby source
/// (mirrors rbs's `Base.location`).
pub fn build_type_application_annotation(
    parser: &Parser,
    node: *const ruby_rbs_sys::bindings::rbs_node,
    location: PrismByteRange,
    names: &NameTable,
) -> Option<TypeApplicationAnnotation> {
    match parser.classify_inline_trailing_annotation(node) {
        InlineTrailingAnnotationKind::TypeApplication { type_args } => {
            let type_args = build_type_list(parser, type_args, names);
            Some(TypeApplicationAnnotation {
                type_args,
                location,
            })
        }
        InlineTrailingAnnotationKind::NodeTypeAssertion { .. } => None,
        InlineTrailingAnnotationKind::Unsupported(_) => None,
    }
}

pub fn parse_trailing_type_application_text(
    body: &str,
    location: PrismByteRange,
    names: &NameTable,
) -> Option<TypeApplicationAnnotation> {
    let (parser, node) = Parser::parse_inline_trailing(body.as_bytes()).ok()?;
    build_type_application_annotation(&parser, node, location, names)
}

pub fn build_node_type_assertion(
    parser: &Parser,
    node: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Option<Type> {
    match parser.classify_inline_trailing_annotation(node) {
        InlineTrailingAnnotationKind::NodeTypeAssertion { type_node } => {
            build_type(parser, type_node, names)
        }
        InlineTrailingAnnotationKind::TypeApplication { .. }
        | InlineTrailingAnnotationKind::Unsupported(_) => None,
    }
}

/// Parse a trailing-annotation type body (the `T` in `#: T`) and build
/// its unresolved [`Type`] AST. The body is reformulated with the
/// `": "` prefix that `RbsParser::parse_inline_trailing` expects.
///
/// Two callers share this triplet: the method-return annotation builder
/// in [`crate::ast::ruby::members`] and the type checker's inline
/// assignment-assertion path. Keeping the `": "` contract in one place
/// means a future change to the rbs trailing-parser convention only
/// touches a single call site.
pub fn parse_trailing_type_text(type_text: &str, names: &NameTable) -> Option<Type> {
    let mut source = Vec::with_capacity(type_text.len() + 2);
    source.extend_from_slice(b": ");
    source.extend_from_slice(type_text.as_bytes());
    let (parser, node) = Parser::parse_inline_trailing(&source).ok()?;
    build_node_type_assertion(&parser, node, names)
}

fn build_type_list(parser: &Parser, nodes: RawNodeList, names: &NameTable) -> Vec<Type> {
    // Each child slot must be filled. Unrepresentable nested types fall
    // back to `Type::Untyped` rather than being dropped — arity must
    // be preserved (nested drops would silently change
    // `Array[Integer, Weird]` into `Array[Integer]`).
    nodes
        .iter()
        .map(|n| {
            build_type(parser, n, names).unwrap_or(Type::Base(BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            }))
        })
        .collect()
}

fn build_function_type(
    parser: &Parser,
    func_ptr: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Function {
    match classify_type(parser, func_ptr) {
        TypeKind::Function {
            required_positionals,
            optional_positionals,
            rest_positionals,
            trailing_positionals,
            required_keywords,
            optional_keywords,
            rest_keywords,
            forwarding,
            return_type,
        } => {
            let return_ty =
                build_type(parser, return_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));

            // `...` (RBS param forwarding, e.g. `(String, ...) -> void`) is an
            // experimental rbs syntax disabled by default
            // (`rbs_parser_options_t::enable_forwarding_params`); crema's
            // parser calls pass no options, so `forwarding` is always null
            // today. Handled defensively rather than left as an unreachable
            // panic: rbs itself defines no call-site arity semantics for it
            // (it only records the marker node; see rbs lib/rbs/types.rb), so
            // fall back to `Function::Untyped` — the same "accepts any
            // arguments" shape used below — rather than adding a `forwarding`
            // field to `FunctionType` that every construction site would need
            // to thread through.
            if forwarding.is_some() {
                return Function::Untyped(UntypedFunctionType {
                    return_type: Box::new(return_ty),
                });
            }

            let required = build_param_type_list(parser, required_positionals, names);
            let optional = build_param_type_list(parser, optional_positionals, names);
            let rest = rest_positionals.map(|p| Box::new(build_param(parser, p, names)));
            let trailing = build_param_type_list(parser, trailing_positionals, names);
            let required_kw = build_keyword_param_list(parser, required_keywords, names);
            let optional_kw = build_keyword_param_list(parser, optional_keywords, names);
            let rest_kw = rest_keywords.map(|p| Box::new(build_param(parser, p, names)));

            Function::Typed(FunctionType {
                required_positionals: required,
                optional_positionals: optional,
                rest_positionals: rest,
                trailing_positionals: trailing,
                required_keywords: required_kw,
                optional_keywords: optional_kw,
                rest_keywords: rest_kw,
                return_type: Box::new(return_ty),
            })
        }
        TypeKind::UntypedFunction { return_type } => {
            let return_ty =
                build_type(parser, return_type, names).unwrap_or(Type::Base(BaseType {
                    kind: BaseTypeKind::Any { todo: false },
                    location: None,
                }));
            Function::Untyped(UntypedFunctionType {
                return_type: Box::new(return_ty),
            })
        }
        // Fallback when the underlying node is neither `FunctionType` nor
        // `UntypedFunctionType`. Structurally unreachable under today's rbs
        // parser, but kept honest for future variants.
        //
        // Must be `Function::Untyped`, not `Typed`: subtyping only
        // skips arity / keyword shape comparison when the function type
        // is `Untyped` (the `(?)` form). A `Typed` fallback with an empty
        // parameter list would participate in shape checks and could
        // produce cascade diagnostics.
        _ => Function::Untyped(UntypedFunctionType {
            return_type: Box::new(Type::Base(BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            })),
        }),
    }
}

fn build_block(
    parser: &Parser,
    block_ptr: *const ruby_rbs_sys::bindings::rbs_types_block,
    names: &NameTable,
) -> Option<BlockType> {
    let TypeKind::Block {
        type_: func_ptr,
        required,
        self_type,
    } = parser.classify_block(block_ptr)
    else {
        return None;
    };

    let function = build_function_type(parser, func_ptr, names);
    let self_type = self_type.map(|st| {
        Box::new(
            build_type(parser, st, names).unwrap_or(Type::Base(BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            })),
        )
    });

    Some(BlockType {
        required,
        function,
        self_type,
    })
}

fn build_param_type_list(
    parser: &Parser,
    params: RawNodeList,
    names: &NameTable,
) -> Vec<FunctionParam> {
    params
        .iter()
        .map(|p| build_param(parser, p, names))
        .collect()
}

fn build_param(
    parser: &Parser,
    param: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> FunctionParam {
    // Both fallback paths below land on `Type::Untyped` — the inner
    // one for a `FunctionParam` whose body cannot be represented, and the
    // outer one for a node that is not even a `FunctionParam` (structurally
    // unreachable today but kept honest for future variants).
    if let TypeKind::FunctionParam { type_, name } = classify_type(parser, param) {
        let ty = build_type(parser, type_, names).unwrap_or(Type::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        }));
        let param_name = name.map(|sym| names.intern_symbol(parser.resolve_constant(sym)));
        FunctionParam {
            ty: Box::new(ty),
            name: param_name,
            location: None,
        }
    } else {
        FunctionParam {
            ty: Box::new(Type::Base(BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            })),
            name: None,
            location: None,
        }
    }
}

fn build_keyword_param_list(
    parser: &Parser,
    keywords: rbs_raw::RawHash,
    names: &NameTable,
) -> Vec<KeywordParam> {
    keywords
        .iter()
        .map(|(key, value)| {
            let key_name = parser.node_as_symbol(key).unwrap_or("");
            let name = names.intern_symbol(key_name);
            let param = build_param(parser, value, names);
            KeywordParam { name, param }
        })
        .collect()
}

fn build_record_key(
    parser: &Parser,
    node: *const ruby_rbs_sys::bindings::rbs_node,
    names: &NameTable,
) -> Option<RecordKey> {
    if let Some(s) = parser.node_as_symbol(node) {
        return Some(RecordKey::Symbol(names.intern_symbol(s)));
    }
    if let Some(s) = parser.string_value(node) {
        return Some(RecordKey::String(s));
    }
    if let Some(s) = parser.integer_string_repr(node) {
        return Some(RecordKey::Integer(s));
    }
    if let Some(b) = parser.bool_value(node) {
        return Some(RecordKey::Bool(b));
    }
    None
}
