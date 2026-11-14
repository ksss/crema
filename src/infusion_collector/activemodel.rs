//! ActiveModel DSL rules.
//!
//! `attribute :name, :type` synthesis is ported from orthoses-rails'
//! `Orthoses::ActiveModel::Attributes#call`
//! (`lib/orthoses/active_model/attributes.rb` in
//! <https://github.com/ksss/orthoses-rails>):
//!
//! | orthoses (runtime trace)                                 | this pass (static)                                   |
//! |----------------------------------------------------------|------------------------------------------------------|
//! | trace `ActiveModel::Attributes::ClassMethods#attribute`  | receiver-less `attribute` in a class whose include sites or super chain reach `::ActiveModel::Attributes` |
//! | `DEFAULT_TYPES[cast_type] \|\| 'untyped'`                | [`cast_type_name`]                                   |
//! | `module <Owner>::ActiveModelGeneratedAttributeMethods`   | same module                                          |
//! | `def name: () -> T?` / `def name=: (untyped) -> untyped` | same defs                                            |
//! | `include <Owner>::ActiveModelGeneratedAttributeMethods`  | same include on the owner                            |
//!
//! Do not add methods or change their types without a corresponding
//! change on the orthoses side.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::MethodKind;
use crate::ast::declarations::{
    ClassDeclaration as Class, ModuleDeclaration as Module, ModuleMember,
};
use crate::ast::method_type::MethodType;
use crate::ast::ruby::members::Member;
use crate::ast::types::{
    BaseType, BaseTypeKind, Function, FunctionParam, FunctionType, OptionalType, Type,
};
use crate::environment::draft::{Context, EnvironmentDraft};
use crate::environment::resolution::TypeNameResolver;
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    class_instance, collect_declared_class_module_names, descendants_of, include_member,
    method_with_type, nested_name,
};
use crate::infusion_collector::pipeline::{InfusionCall, push_def};
use crate::infusion_collector::sidekiq::{alias_map, resolve_mixin_argument};
use crate::name::NameTable;
use crate::type_name::TypeName;

const ACTIVE_MODEL_ATTRIBUTES: &str = "::ActiveModel::Attributes";
const GENERATED_ATTRIBUTE_METHODS: &str = "ActiveModelGeneratedAttributeMethods";

pub(crate) fn collect_call(members: &mut Vec<Member>, call: &InfusionCall) {
    collect_has_secure_password_call(members, call);
}

fn collect_has_secure_password_call(members: &mut Vec<Member>, call: &InfusionCall) {
    if call.name != "has_secure_password" {
        return;
    }
    let attribute = call
        .symbol_args
        .first()
        .map(|arg| (arg.name.as_str(), arg.location))
        .unwrap_or(("password", call.location));
    let reader = attribute.0.to_string();
    let writer = format!("{}=", attribute.0);
    let confirmation_writer = format!("{}_confirmation=", attribute.0);
    let authenticator = format!("authenticate_{}", attribute.0);

    for name in [reader, writer, confirmation_writer, authenticator] {
        push_def(
            members,
            name,
            MethodKind::Instance,
            call.location,
            attribute.1,
        );
    }
    if attribute.0 == "password" {
        push_def(
            members,
            "authenticate".to_string(),
            MethodKind::Instance,
            call.location,
            attribute.1,
        );
    }
}

/// One `attribute :name[, :cast_type]` call in a class body. A candidate
/// only: [`synthesize`] keeps it when the owner reaches
/// `ActiveModel::Attributes`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ActiveModelAttribute {
    pub(crate) owner: TypeName,
    pub(crate) name: String,
    /// The second positional argument when it is a symbol literal.
    pub(crate) cast_type: Option<String>,
}

/// The collector records `attribute`'s first positional symbol as
/// `symbol_args[0]` and a symbol second positional as `symbol_args[1]`.
pub(crate) fn attribute_from_call(
    owner: TypeName,
    call: &InfusionCall,
) -> Option<ActiveModelAttribute> {
    if call.name != "attribute" {
        return None;
    }
    let mut args = call.symbol_args.iter();
    let name = args.next()?.name.clone();
    Some(ActiveModelAttribute {
        owner,
        name,
        cast_type: args.next().map(|arg| arg.name.clone()),
    })
}

/// orthoses `Attributes::DEFAULT_TYPES`. `:datetime` is `::Time` here,
/// not the `ActiveSupport::TimeWithZone` ActiveRecord columns read as:
/// `ActiveModel::Type::DateTime` casts to a plain `Time`.
fn cast_type_name(cast_type: &str) -> Option<&'static str> {
    Some(match cast_type {
        "big_integer" | "integer" => "::Integer",
        "binary" | "immutable_string" | "string" => "::String",
        "boolean" => "bool",
        "date" => "::Date",
        "datetime" | "time" => "::Time",
        "decimal" => "::BigDecimal",
        "float" => "::Float",
        _ => return None,
    })
}

/// One receiver-less `include X` site, as the collector recorded it.
pub(crate) struct ActiveModelIncludeSite<'a> {
    pub(crate) target: TypeName,
    pub(crate) raw_module_name: &'a str,
    pub(crate) lexical_context: &'a [TypeName],
}

/// Declare `<Owner>::ActiveModelGeneratedAttributeMethods` and include it
/// in every owner whose `attribute` calls reach
/// `ActiveModel::Attributes#attribute`: the owner itself includes
/// `ActiveModel::Attributes`, or a superclass does (orthoses traces the
/// runtime call, so inherited `attribute` is covered there too).
///
/// The reader and writer live in the generated module rather than on the
/// owner, as in Rails, so a `def name; super; end` on the owner reaches
/// them.
pub(crate) fn synthesize(
    draft: &mut EnvironmentDraft,
    attributes: &[ActiveModelAttribute],
    sites: &[ActiveModelIncludeSite<'_>],
) {
    if attributes.is_empty() {
        return;
    }
    let includers: FxHashSet<TypeName> = {
        let names = draft.names();
        let target = names.parse_type_name(ACTIVE_MODEL_ATTRIBUTES);
        let all_names = collect_declared_class_module_names(draft);
        let aliases = alias_map(draft);
        let resolver = TypeNameResolver::new(&all_names, &aliases, names);
        sites
            .iter()
            .filter(|site| {
                resolve_mixin_argument(
                    &resolver,
                    &aliases,
                    names,
                    site.raw_module_name,
                    site.lexical_context,
                ) == Some(target)
            })
            .map(|site| site.target)
            .collect()
    };
    if includers.is_empty() {
        return;
    }
    let descendants = descendants_of(draft, &includers);

    // Last declaration of a name wins, as the runtime redefinition does;
    // owners and names keep their first-seen order.
    let mut owners: Vec<TypeName> = Vec::new();
    let mut by_owner: FxHashMap<TypeName, Vec<&ActiveModelAttribute>> = FxHashMap::default();
    for attribute in attributes {
        if !includers.contains(&attribute.owner) && !descendants.contains(&attribute.owner) {
            continue;
        }
        let entry = by_owner.entry(attribute.owner).or_insert_with(|| {
            owners.push(attribute.owner);
            Vec::new()
        });
        match entry.iter_mut().find(|a| a.name == attribute.name) {
            Some(existing) => *existing = attribute,
            None => entry.push(attribute),
        }
    }

    let context: Context = Arc::from([draft.names().absolute_root()]);
    for owner in owners {
        synthesize_owner(draft, Arc::clone(&context), owner, &by_owner[&owner]);
    }
}

fn synthesize_owner(
    draft: &mut EnvironmentDraft,
    context: Context,
    owner: TypeName,
    attributes: &[&ActiveModelAttribute],
) {
    let names = draft.names();
    let module_name = nested_name(names, owner, GENERATED_ATTRIBUTE_METHODS);
    let members = attributes
        .iter()
        .flat_map(|attribute| {
            let reader = reader_method_type(names, attribute.cast_type.as_deref());
            [
                (attribute.name.clone(), reader),
                (format!("{}=", attribute.name), writer_method_type()),
            ]
        })
        .map(|(name, method_type)| {
            ModuleMember::Member(method_with_type(
                names.intern_symbol(&name),
                MethodKind::Instance,
                method_type,
            ))
        })
        .collect();
    let module_decl = Arc::new(Module {
        name: module_name,
        type_params: Vec::new(),
        self_types: Vec::new(),
        members,
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let owner_decl = Arc::new(Class {
        name: owner,
        type_params: Vec::new(),
        super_class: None,
        members: vec![include_member(module_name, Vec::new())],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let _ = draft.insert_module_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveModel, None),
        Arc::clone(&context),
        module_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::ActiveModel, None),
        context,
        owner_decl,
    );
}

/// `() -> T?`, where an unknown cast type is plain `untyped` (orthoses
/// writes `untyped?`, which means the same).
fn reader_method_type(names: &NameTable, cast_type: Option<&str>) -> MethodType {
    let return_type = match cast_type.and_then(cast_type_name) {
        Some("bool") => optional(Type::Base(BaseType {
            kind: BaseTypeKind::Bool,
            location: None,
        })),
        Some(raw) => optional(class_instance(names.parse_type_name(raw))),
        None => untyped(),
    };
    method_type(Vec::new(), return_type)
}

/// `(untyped) -> untyped`.
fn writer_method_type() -> MethodType {
    let param = FunctionParam {
        ty: Box::new(untyped()),
        name: None,
        location: None,
    };
    method_type(vec![param], untyped())
}

fn method_type(required_positionals: Vec<FunctionParam>, return_type: Type) -> MethodType {
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals,
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(return_type),
        }),
        block: None,
        location: None,
    }
}

fn optional(ty: Type) -> Type {
    Type::Optional(OptionalType {
        ty: Box::new(ty),
        location: None,
    })
}

fn untyped() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Any { todo: false },
        location: None,
    })
}
