//! Build resolved [`Ty`] / [`MethodType`] values from unresolved
//! [`crate::ast`] nodes plus an environment.
//!
//! Pure ast → Ty resolver: callers parse rbs source via `ast_builder`
//! to obtain `ast::Type` / `ast::MethodType`, then call into this
//! module with the frozen environment (its `TypeNameResolver` tables
//! and class-alias normalization), the intern tables and the type-param
//! scope to mint interned `Ty` / `MethodType`. Used by `Environment::resolve` to walk the
//! buffered ast::* declarations exactly once after every source has
//! been observed.
//!
//! Phase 2 of ADR-0014. The Phase 6/7 flip removed the
//! `*_from_raw` composition helpers that fused parsing + building
//! into one call — that fusion is now spelled out by callers as
//! `ast_builder::build_*(...).unwrap_or(<Ty::UNTYPED ast fallback>)`
//! plus a `build_type` / `build_method_type` / `build_overloads`
//! invocation.


use crate::ast::members::MethodDefinitionOverload as AstOverload;
use crate::ast::method_type::MethodType as AstMethodType;
use crate::ast::ruby::members::{
    BlockEntry, DocStyle, DoubleSplatRestEntry, ExplicitAnnotation, MethodTypeAnnotation,
    PositionalEntry, SplatRestEntry, TypeAnnotations,
};
use crate::ast::types::{
    BaseTypeKind, BlockType as AstBlock, Function as AstFunctionType, FunctionType as AstFunction,
    Literal as AstLiteral, RecordField as AstRecordField, RecordKey as AstRecordKey,
    Type as AstType, UntypedFunctionType as AstUntypedFunction,
};
use crate::environment::Environment;
use crate::environment::resolution::{TypeNameResolver, absolute_type_name_typename};
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;
use crate::type_param::{
    MethodKind, TypeParam, TypeParamScope, TypeVarKey, TypeVarScope, Variance,
};
use crate::types::{
    Block, Function, FunctionType, Literal, MethodType, RecordKey, Ty, Type, TypeTable,
    UntypedFunction,
};

/// Resolve an unresolved [`AstType`] into an interned [`Ty`].
#[allow(clippy::too_many_arguments)]
pub fn build_type(
    ast_ty: &AstType,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> Ty {
    match ast_ty {
        AstType::ClassInstance(t) => {
            if let Some(key) =
                scoped_type_variable_from_class_instance(&t.name, &t.args, type_param_scope, names)
            {
                return types.intern(Type::TypeVariable {
                    raw: key.raw,
                    scope: key.scope,
                });
            }
            let resolved = lower_type_name(t.name, context, env);
            let arg_tys = build_type_vec(&t.args, context, env, names, types, type_param_scope);
            types.intern(Type::ClassInstance {
                name: resolved,
                args: arg_tys,
            })
        }
        AstType::Interface(t) => {
            let resolved = lower_type_name(t.name, context, env);
            let arg_tys = build_type_vec(&t.args, context, env, names, types, type_param_scope);
            types.intern(Type::Interface {
                name: resolved,
                args: arg_tys,
            })
        }
        AstType::Alias(t) => {
            let resolved = lower_type_name(t.name, context, env);
            let arg_tys = build_type_vec(&t.args, context, env, names, types, type_param_scope);
            types.intern(Type::Alias {
                name: resolved,
                args: arg_tys,
            })
        }
        AstType::ClassSingleton(t) => {
            let resolved = lower_type_name(t.name, context, env);
            types.intern(Type::ClassSingleton { name: resolved })
        }
        AstType::Variable(t) => {
            // Honor any alpha-rename recorded in the scope: the *key* in the
            // map is the raw written at the use site (`B`), but the *value*
            // carries the canonical raw (`A`) plus its declaration scope.
            let key = type_param_scope
                .get(&t.name)
                .cloned()
                .unwrap_or_else(|| TypeVarKey::new(t.name, TypeVarScope::Free));
            types.intern(Type::TypeVariable {
                raw: key.raw,
                scope: key.scope,
            })
        }
        AstType::Union(t) => {
            let built = build_type_vec(&t.types, context, env, names, types, type_param_scope);
            if built.len() == 1 {
                built[0]
            } else {
                types.intern(Type::Union(built))
            }
        }
        AstType::Intersection(t) => {
            let built = build_type_vec(&t.types, context, env, names, types, type_param_scope);
            if built.len() == 1 {
                built[0]
            } else {
                types.intern(Type::Intersection(built))
            }
        }
        AstType::Optional(t) => {
            let inner_ty = build_type(&t.ty, context, env, names, types, type_param_scope);
            types.intern(Type::Optional(inner_ty))
        }
        AstType::Tuple(t) => {
            let built = build_type_vec(&t.types, context, env, names, types, type_param_scope);
            types.intern(Type::Tuple(built))
        }
        AstType::Record(t) => {
            let mut resolved: Vec<(RecordKey, Ty, bool)> = t
                .fields
                .iter()
                .map(|f: &AstRecordField| {
                    let key = build_record_key(&f.key, names);
                    let value = build_type(&f.ty, context, env, names, types, type_param_scope);
                    (key, value, f.required)
                })
                .collect();
            resolved.sort_by(|a, b| a.0.cmp(&b.0));
            types.intern(Type::Record { fields: resolved })
        }
        AstType::Proc(proc_type) => {
            let func = build_function_type(
                &proc_type.function,
                context,
                env,
                names,
                types,
                type_param_scope,
            );
            let self_ty = proc_type
                .self_type
                .as_ref()
                .map(|st| build_type(st, context, env, names, types, type_param_scope));
            let blk = proc_type
                .block
                .as_ref()
                .map(|b| build_block(b, context, env, names, types, type_param_scope));
            types.intern(Type::Proc {
                type_: func,
                self_type: self_ty,
                block: blk,
            })
        }
        AstType::Literal(t) => match &t.literal {
            AstLiteral::Integer(s) => types.intern(Type::Literal(Literal::Integer(
                normalize_rbs_integer_string(s),
            ))),
            other => types.intern(Type::Literal(build_literal(other, names))),
        },
        AstType::Base(b) => match &b.kind {
            BaseTypeKind::Void => Ty::VOID,
            BaseTypeKind::Nil => Ty::NIL,
            BaseTypeKind::Bool => Ty::BOOL,
            BaseTypeKind::Any { .. } => Ty::UNTYPED,
            BaseTypeKind::Top => Ty::TOP,
            BaseTypeKind::Bottom => Ty::BOTTOM,
            BaseTypeKind::SelfType => Ty::SELF_TYPE,
            BaseTypeKind::Instance => Ty::INSTANCE_TYPE,
            BaseTypeKind::Class => Ty::CLASS_TYPE,
        },
    }
}

/// Resolve an unresolved [`AstMethodType`] into a [`MethodType`].
///
/// `method_owner = Some((class_qname, method_name))` alpha-renames the
/// method's own type params (`[A, B]`) to `<class>#<method>@<raw>` and
/// merges that scope on top of `type_param_scope` (the outer class-level
/// scope) so method params shadow same-named class params within the
/// method's signature. `None` leaves method params raw — used for Proc
/// types (`^(T) -> T` nested inside a class signature) and inline
/// annotations that don't yet thread owner context.
#[allow(clippy::too_many_arguments)]
pub fn build_method_type(
    ast_mt: &AstMethodType,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
    method_owner: Option<(&TypeName, Symbol, MethodKind)>,
    overload_index: u16,
) -> MethodType {
    // Pass 1: build the method-level scope first so bounds can reference
    // sibling method params, and so the merged scope below correctly shadows
    // any class-level entry with the same raw name.
    //
    // ADR-0023 Phase 3: each method-level param gets a `TypeVarScope::Method`
    // identity that includes the 0-based `overload_index`. Two `[T]`s on the
    // same `def` but in different overloads are now distinct variables.
    let mut merged_scope: TypeParamScope = type_param_scope.clone();
    let mut scoped_keys: Vec<TypeVarKey> = Vec::with_capacity(ast_mt.type_params.len());
    for tp in &ast_mt.type_params {
        let raw = tp.name;
        let scope = match method_owner {
            Some((class_q, method_n, kind)) => TypeVarScope::Method {
                class: *class_q,
                method: method_n,
                kind,
                overload_index,
            },
            None => TypeVarScope::Free,
        };
        let key = TypeVarKey::new(raw, scope);
        merged_scope.insert(raw, key.clone());
        scoped_keys.push(key);
    }

    let type_params: Vec<TypeParam> = ast_mt
        .type_params
        .iter()
        .enumerate()
        .map(|(i, tp)| {
            let scoped = scoped_keys[i].clone();
            let mut param = TypeParam::new(scoped, Variance::Invariant);
            if let Some(upper) = &tp.upper_bound {
                param.upper_bound =
                    Some(build_type(upper, context, env, names, types, &merged_scope));
            }
            if let Some(lower) = &tp.lower_bound {
                param.lower_bound =
                    Some(build_type(lower, context, env, names, types, &merged_scope));
            }
            param
        })
        .collect();

    let function = build_function_type(&ast_mt.function, context, env, names, types, &merged_scope);
    let block = ast_mt
        .block
        .as_ref()
        .map(|b| build_block(b, context, env, names, types, &merged_scope));

    MethodType {
        type_params,
        type_: function,
        block,
    }
}

/// Resolve a `Vec<AstOverload>` slice into `Vec<MethodType>`.
///
/// Per-overload annotations on each [`AstOverload`] are not propagated
/// past this boundary — the resolved [`crate::definition::Method`]
/// has no slot for them. Build-layer code that needs annotations should
/// read them from the AST overload directly.
#[allow(clippy::too_many_arguments)]
pub fn build_overloads(
    ast_overloads: &[AstOverload],
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
    method_owner: Option<(&TypeName, Symbol, MethodKind)>,
    start_index: u16,
) -> Vec<MethodType> {
    ast_overloads
        .iter()
        .enumerate()
        .map(|(i, o)| {
            build_method_type(
                &o.method_type,
                context,
                env,
                names,
                types,
                type_param_scope,
                method_owner,
                start_index.saturating_add(u16::try_from(i).unwrap_or(u16::MAX)),
            )
        })
        .collect()
}

/// Resolve a classified [`MethodTypeAnnotation`] into concrete
/// `Vec<MethodType>` values, or `None` when the caller should fall back
/// to an untyped default.
///
/// This is the "resolve" counterpart to `MethodTypeAnnotation::build`
/// — take the unresolved classification plus an environment and
/// produce resolved method types. Replaces the inner logic of
/// `src/inline_parser.rs::resolve_def_overloads`.
#[allow(clippy::too_many_arguments)]
pub fn build_overloads_from_annotation_with_scope(
    mta: &MethodTypeAnnotation,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
    method_owner: Option<(&TypeName, Symbol, MethodKind)>,
) -> Option<Vec<MethodType>> {
    match &mta.type_annotations {
        TypeAnnotations::Array(arr) => {
            let mut result = Vec::new();
            for annot in arr {
                match annot {
                    ExplicitAnnotation::Colon(annotation) => {
                        let idx = u16::try_from(result.len()).unwrap_or(u16::MAX);
                        result.push(build_method_type(
                            &annotation.method_type,
                            context,
                            env,
                            names,
                            types,
                            type_param_scope,
                            method_owner,
                            idx,
                        ));
                    }
                    ExplicitAnnotation::MethodTypes(annotation) => {
                        for overload in &annotation.overloads {
                            let idx = u16::try_from(result.len()).unwrap_or(u16::MAX);
                            result.push(build_method_type(
                                &overload.method_type,
                                context,
                                env,
                                names,
                                types,
                                type_param_scope,
                                method_owner,
                                idx,
                            ));
                        }
                    }
                }
            }
            // Depth-defense against `MethodTypes([])` surviving into
            // TypeAnnotations::Array: a `Some([])` here would bypass
            // the caller's untyped fallback and leave the method with
            // zero overloads. ast_builder guards the usual source, but
            // stay safe if this struct is constructed directly (tests,
            // future callers).
            if result.is_empty() {
                None
            } else {
                Some(result)
            }
        }
        TypeAnnotations::DocStyle(doc) => {
            let mt = build_doc_style_method_type(doc, context, env, names, types, type_param_scope);
            Some(vec![mt])
        }
        TypeAnnotations::None => None,
    }
}

fn build_doc_style_method_type(
    doc: &DocStyle,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> MethodType {
    let return_ty = match &doc.return_type_annotation {
        Some(ann) => build_type(
            &ann.return_type,
            context,
            env,
            names,
            types,
            type_param_scope,
        ),
        None => Ty::UNTYPED,
    };

    let mut func = Function::empty(return_ty);

    let resolve_entry = |entry: &PositionalEntry| match entry {
        PositionalEntry::Annotated(ann) => build_type(
            &ann.param_type,
            context,
            env,
            names,
            types,
            type_param_scope,
        ),
        PositionalEntry::ByName(_) => Ty::UNTYPED,
    };

    let resolve_splat_rest = |entry: &SplatRestEntry| match entry {
        SplatRestEntry::Annotated(ann) => build_type(
            &ann.param_type,
            context,
            env,
            names,
            types,
            type_param_scope,
        ),
        SplatRestEntry::ByName(_) | SplatRestEntry::Unnamed => Ty::UNTYPED,
    };

    let resolve_double_splat_rest = |entry: &DoubleSplatRestEntry| match entry {
        DoubleSplatRestEntry::Annotated(ann) => build_type(
            &ann.param_type,
            context,
            env,
            names,
            types,
            type_param_scope,
        ),
        DoubleSplatRestEntry::ByName(_) | DoubleSplatRestEntry::Unnamed => Ty::UNTYPED,
    };

    let resolve_block = |entry: &BlockEntry| match entry {
        BlockEntry::Annotated(ann) => Block {
            required: ann.question_location.is_none(),
            type_: build_function_type(&ann.function, context, env, names, types, type_param_scope),
            self_type: None,
        },
        BlockEntry::ByName(_) | BlockEntry::Unnamed => Block {
            required: false,
            type_: FunctionType::Untyped(UntypedFunction {
                return_type: Ty::UNTYPED,
            }),
            self_type: None,
        },
    };

    func.required_positionals = doc
        .required_positionals
        .iter()
        .map(&resolve_entry)
        .collect();
    func.optional_positionals = doc
        .optional_positionals
        .iter()
        .map(&resolve_entry)
        .collect();
    func.rest_positional = doc.rest_positionals.as_ref().map(&resolve_splat_rest);
    func.trailing_positionals = doc
        .trailing_positionals
        .iter()
        .map(&resolve_entry)
        .collect();
    func.required_keywords = doc
        .required_keywords
        .iter()
        .map(|(n, e)| (n.clone(), resolve_entry(e)))
        .collect();
    func.optional_keywords = doc
        .optional_keywords
        .iter()
        .map(|(n, e)| (n.clone(), resolve_entry(e)))
        .collect();
    func.rest_keyword = doc.rest_keywords.as_ref().map(&resolve_double_splat_rest);

    MethodType {
        type_params: vec![],
        type_: FunctionType::Typed(func),
        block: doc.block.as_ref().map(resolve_block),
    }
}

/// Resolve one reference-position type name against `context` and
/// normalize class aliases away, in the two steps rbs keeps separate:
/// `Environment#absolute_type_name` (relative → absolute; an alias
/// `new_name` is kept as is) followed by
/// `Environment#normalize_type_name` (alias → target). A name the
/// environment does not declare comes back unchanged, in its source
/// form, so an unresolved reference stays observable downstream (a
/// relative raw keeps `namespace.is_absolute() == false`).
fn lower_type_name(name: TypeName, context: &[TypeName], env: &Environment) -> TypeName {
    let resolver = TypeNameResolver::new(&env.all_names, &env.aliases, env.names());
    let resolved = absolute_type_name_typename(&resolver, name, context);
    env.normalize_type_name(resolved)
}

fn scoped_type_variable_from_class_instance(
    name: &TypeName,
    args: &[AstType],
    type_param_scope: &TypeParamScope,
    names: &NameTable,
) -> Option<TypeVarKey> {
    let parent = names.type_name_parent(*name)?;
    if names.type_name_is_root(parent) && args.is_empty() {
        // Same alpha-rename treatment as in `node_to_type`'s `TypeVariable`
        // arm: return the value verbatim so reopen-side spellings collapse
        // onto the primary identity.
        type_param_scope.get(&names.last_segment(*name)?).cloned()
    } else {
        None
    }
}

fn build_type_vec(
    nodes: &[AstType],
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> Vec<Ty> {
    nodes
        .iter()
        .map(|n| build_type(n, context, env, names, types, type_param_scope))
        .collect()
}

fn build_function_type(
    ast_ft: &AstFunctionType,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> FunctionType {
    match ast_ft {
        AstFunctionType::Typed(f) => FunctionType::Typed(build_function(
            f,
            context,
            env,
            names,
            types,
            type_param_scope,
        )),
        AstFunctionType::Untyped(u) => FunctionType::Untyped(build_untyped_function(
            u,
            context,
            env,
            names,
            types,
            type_param_scope,
        )),
    }
}

fn build_function(
    ast_f: &AstFunction,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> Function {
    let required_positionals: Vec<Ty> = ast_f
        .required_positionals
        .iter()
        .map(|p| build_type(&p.ty, context, env, names, types, type_param_scope))
        .collect();
    let optional_positionals: Vec<Ty> = ast_f
        .optional_positionals
        .iter()
        .map(|p| build_type(&p.ty, context, env, names, types, type_param_scope))
        .collect();
    let rest_positional = ast_f
        .rest_positionals
        .as_ref()
        .map(|p| build_type(&p.ty, context, env, names, types, type_param_scope));
    let trailing_positionals: Vec<Ty> = ast_f
        .trailing_positionals
        .iter()
        .map(|p| build_type(&p.ty, context, env, names, types, type_param_scope))
        .collect();
    let required_keywords: Vec<(String, Ty)> = ast_f
        .required_keywords
        .iter()
        .map(|kp| {
            (
                names.resolve(kp.name).to_string(),
                build_type(&kp.param.ty, context, env, names, types, type_param_scope),
            )
        })
        .collect();
    let optional_keywords: Vec<(String, Ty)> = ast_f
        .optional_keywords
        .iter()
        .map(|kp| {
            (
                names.resolve(kp.name).to_string(),
                build_type(&kp.param.ty, context, env, names, types, type_param_scope),
            )
        })
        .collect();
    let rest_keyword = ast_f
        .rest_keywords
        .as_ref()
        .map(|p| build_type(&p.ty, context, env, names, types, type_param_scope));
    let return_type = build_type(
        &ast_f.return_type,
        context,
        env,
        names,
        types,
        type_param_scope,
    );

    Function {
        required_positionals,
        optional_positionals,
        rest_positional,
        trailing_positionals,
        required_keywords,
        optional_keywords,
        rest_keyword,
        return_type,
    }
}

fn build_untyped_function(
    ast_u: &AstUntypedFunction,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> UntypedFunction {
    let return_type = build_type(
        &ast_u.return_type,
        context,
        env,
        names,
        types,
        type_param_scope,
    );
    UntypedFunction { return_type }
}

fn build_block(
    ast_b: &AstBlock,
    context: &[TypeName],
    env: &Environment,
    names: &NameTable,
    types: &TypeTable,
    type_param_scope: &TypeParamScope,
) -> Block {
    let function = build_function_type(
        &ast_b.function,
        context,
        env,
        names,
        types,
        type_param_scope,
    );
    let self_type = ast_b
        .self_type
        .as_ref()
        .map(|t| build_type(t, context, env, names, types, type_param_scope));
    Block {
        required: ast_b.required,
        type_: function,
        self_type,
    }
}

/// Normalizes an RBS integer-literal lexeme (`rbs_ast_integer_t
/// .string_representation`, e.g. `"1_000"`, `"+7"`, `"-007"`) into the
/// canonical decimal string used as `Literal::Integer`/`RecordKey::Integer`'s
/// payload. rbs's own C parser keeps this text as the raw lexeme (only
/// whitespace-stripped, per `rbs_string_strip_whitespace` in
/// `parser.c`'s `tINTEGER` case) and normalizes it to a real value only when
/// bridging to Ruby, via `String#to_i` (`ast_translation.c`). RBS integer
/// syntax has no `0x`/`0b` prefixes (`number = [0-9] [0-9_]*` in
/// `lexer.re`), so — unlike the prism/Ruby-source side — the only
/// differences from canonical form are underscore separators, a leading
/// `+`, and leading zeros; strip them the same way `to_i` would.
fn normalize_rbs_integer_string(s: &str) -> String {
    let (negative, unsigned) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let digits: String = unsigned.chars().filter(|c| *c != '_').collect();
    let magnitude = digits.trim_start_matches('0');
    let magnitude = if magnitude.is_empty() { "0" } else { magnitude };
    if negative && magnitude != "0" {
        format!("-{magnitude}")
    } else {
        magnitude.to_string()
    }
}

fn build_record_key(ast_key: &AstRecordKey, names: &NameTable) -> RecordKey {
    match ast_key {
        AstRecordKey::Symbol(s) => RecordKey::Symbol(names.resolve(*s).to_string()),
        AstRecordKey::String(s) => RecordKey::String(s.clone()),
        AstRecordKey::Integer(s) => RecordKey::Integer(normalize_rbs_integer_string(s)),
        AstRecordKey::Bool(b) => RecordKey::Bool(*b),
    }
}

fn build_literal(ast_lit: &AstLiteral, names: &NameTable) -> Literal {
    match ast_lit {
        AstLiteral::Integer(_) => unreachable!("Integer literal handled in build_type"),
        AstLiteral::String(s) => Literal::String(s.clone()),
        AstLiteral::Symbol(s) => Literal::Symbol(names.resolve(*s).to_string()),
        AstLiteral::Bool(b) => Literal::Bool(*b),
    }
}
