//! dry-initializer `param` / `option` synthesis.
//!
//! [dry-initializer](https://github.com/dry-rb/dry-initializer) classes
//! write `extend Dry::Initializer` and then declare their constructor
//! arguments with `param` / `option`; the gem `class_eval`s a reader per
//! argument and an initializer into a per-class anonymous mixin at
//! runtime. Statically neither exists, so readers are `Ruby::NoMethod`
//! and every `new` argument is unexpected.
//!
//! Static counterpart of the gem's builders (`lib/dry/initializer/`):
//!
//! | dry-initializer (runtime)                           | this pass (static)                                  |
//! |-----------------------------------------------------|-----------------------------------------------------|
//! | `Config#mixin`, an anonymous module the class includes | `<Owner>::DryInitializerMixin` + include on the owner |
//! | `Builders::Reader` (`attr_reader` / `def target`)   | `def <target>: () -> untyped`, private when `reader: :private` |
//! | `Builders::Attribute` (`@target = ...`)             | `@<target>: untyped`                                |
//! | `Builders::Signature` + `Mixin::Root#initialize`    | `def initialize: (untyped.., ?untyped.., *untyped, req: untyped, ?opt: untyped, **untyped) -> void` |
//! | `Config#final_definitions`                          | superclass definitions first, a redefinition keeps the parent's position |
//!
//! Everything is `untyped` on purpose: the `type` argument is a coercer
//! (a callable), not a type, and the value a reader returns is whatever
//! the coercer produced. The point of this pass is that the names exist.
//!
//! The members live in a generated module, as at runtime, so that an
//! owner's `def name; super; end` reaches the synthesized reader.
//!
//! The initializer is tolerant by construction — surplus positionals and
//! unknown keywords are dropped silently by the gem — so the only
//! arguments it can reject are missing required ones. `**untyped` is
//! emitted even without any `option`: the gem then has no `**` and a
//! keyword argument lands in `*` as a trailing hash, which still succeeds.
//!
//! Arguments that are not literals fall to the side that cannot produce
//! a false positive: an unreadable `param` name turns the initializer of
//! that class and its descendants into `(*untyped, **untyped)`, an
//! unreadable `optional:` / `required:` or a `**splat` makes the
//! argument optional, an unreadable `as:` drops the reader.
//!
//! The `::Dry::Initializer` module itself is *not* synthesized — it comes
//! from the user's rbs collection, and when it is absent this pass does
//! nothing (infusion design goal: crema does not ship type definitions).

use std::sync::Arc;

use ruby_prism::{CallNode, Node};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::MethodKind;
use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, Member, ModuleDeclaration as Module, ModuleMember,
};
use crate::ast::members::{ExtendMember as Extend, InstanceVariableMember, Visibility};
use crate::ast::method_type::MethodType;
use crate::ast::types::{
    BaseType, BaseTypeKind, Function, FunctionParam, FunctionType, KeywordParam, Type,
};
use crate::environment::draft::{Context, EnvironmentDraft};
use crate::environment::resolution::TypeNameResolver;
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    collect_declared_class_module_names, include_member, method_with_type, nested_name,
    resolved_super_edges,
};
use crate::infusion_collector::pipeline::constant_node_to_string;
use crate::infusion_collector::sidekiq::{alias_map, resolve_mixin_argument};
use crate::name::NameTable;
use crate::type_name::TypeName;

const DRY_INITIALIZER: &str = "::Dry::Initializer";
const MIXIN: &str = "DryInitializerMixin";

/// What one class body (or one Concern `included` block) says to
/// dry-initializer, in source order.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DryInitializerBody {
    pub(crate) extends: Vec<ExtendSite>,
    pub(crate) definitions: Vec<Definition>,
    /// The body writes its own `def initialize`, which shadows the
    /// gem's (`Mixin::Root` sits behind the class in the MRO).
    pub(crate) defines_initialize: bool,
}

impl DryInitializerBody {
    pub(crate) fn is_empty(&self) -> bool {
        self.extends.is_empty() && self.definitions.is_empty() && !self.defines_initialize
    }
}

/// A receiver-less `extend X` or `extend X[...]`: `X` as written.
/// Whether it names `::Dry::Initializer` is decided at synthesis, with
/// Ruby's constant lookup.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ExtendSite {
    pub(crate) raw_module_name: String,
    /// `extend Dry::Initializer[undefined: false]`: the argument is a
    /// call, so no `extend` member exists for it anywhere else and this
    /// pass has to supply one.
    pub(crate) bracket: bool,
}

/// `Dry::Initializer::Definition`, reduced to what shapes a signature.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Definition {
    pub(crate) option: bool,
    /// `None` when the first argument is not a symbol / string literal.
    pub(crate) source: Option<String>,
    /// Reader and ivar name. `None` when it cannot be read (`as:` is not
    /// a literal, or `source` is unknown) or is not a method name.
    pub(crate) target: Option<String>,
    pub(crate) optional: bool,
    pub(crate) reader: Reader,
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Reader {
    Public,
    Private,
    None,
}

/// A class body's dry-initializer facts, with the cref its `extend`
/// arguments were written under (outer-first, without the root).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DryInitializerSite {
    pub(crate) target: TypeName,
    pub(crate) lexical_context: Vec<TypeName>,
    /// The body is a Concern `included` block expanded onto `target`.
    pub(crate) from_concern: bool,
    pub(crate) body: DryInitializerBody,
}

/// The module a receiver-less `extend` call names, when it has the one
/// shape this pass reads: a single constant, or a single `Constant[...]`.
pub(crate) fn extend_site(node: &CallNode<'_>) -> Vec<ExtendSite> {
    let Some(arguments) = node.arguments() else {
        return Vec::new();
    };
    arguments
        .arguments()
        .iter()
        .filter_map(|arg| {
            if let Some(raw_module_name) = constant_node_to_string(&arg) {
                return Some(ExtendSite {
                    raw_module_name,
                    bracket: false,
                });
            }
            let call = arg.as_call_node()?;
            if call.name().as_slice() != b"[]" {
                return None;
            }
            Some(ExtendSite {
                raw_module_name: constant_node_to_string(&call.receiver()?)?,
                bracket: true,
            })
        })
        .collect()
}

/// Static counterpart of `Dry::Initializer::Dispatchers.call` for one
/// receiver-less `param` / `option` call.
pub(crate) fn definition_from_call(node: &CallNode<'_>, option: bool) -> Definition {
    let mut args = node
        .arguments()
        .map(|arguments| arguments.arguments().iter().collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter();
    let source = args.next().as_ref().and_then(literal_name);

    let mut target = source.clone();
    let mut optional = false;
    let mut has_default = false;
    let mut required = false;
    let mut reader = Reader::Public;
    for arg in args {
        let Some(hash) = arg.as_keyword_hash_node() else {
            continue;
        };
        for element in hash.elements().iter() {
            let Some(assoc) = element.as_assoc_node() else {
                // `**opts`: any key may be in there.
                optional = true;
                continue;
            };
            let Some(key) = assoc.key().as_symbol_node() else {
                continue;
            };
            let value = assoc.value();
            match key.unescaped() {
                b"as" => target = literal_name(&value),
                b"optional" => optional |= !is_falsy(&value),
                b"default" => has_default |= !is_falsy(&value),
                b"required" => required = value.as_true_node().is_some(),
                b"reader" => reader = reader_from(&value),
                _ => {}
            }
        }
    }
    // `PrepareTarget` downcases the name and rejects anything that is not
    // a Ruby method name; the gem's own generated code then breaks on a
    // `?` suffix, so only plain identifiers are worth a reader.
    let target = target
        .map(|name| name.to_lowercase())
        .filter(|name| is_identifier(name));
    Definition {
        option,
        source,
        target,
        // `PrepareDefault` + `PrepareOptional`.
        optional: (optional || has_default) && !required,
        reader,
    }
}

fn literal_name(node: &Node<'_>) -> Option<String> {
    if let Some(symbol) = node.as_symbol_node() {
        return Some(String::from_utf8_lossy(symbol.unescaped()).to_string());
    }
    node.as_string_node()
        .map(|string| String::from_utf8_lossy(string.unescaped()).to_string())
}

fn is_falsy(node: &Node<'_>) -> bool {
    node.as_false_node().is_some() || node.as_nil_node().is_some()
}

/// `PrepareReader`: `false` / `nil` skip the reader, `:private` makes it
/// private. `:protected` has no RBS counterpart and stays public, like
/// every value that cannot be read — the name existing matters more than
/// its visibility.
fn reader_from(node: &Node<'_>) -> Reader {
    if is_falsy(node) {
        return Reader::None;
    }
    match literal_name(node).as_deref() {
        Some("private") => Reader::Private,
        _ => Reader::Public,
    }
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Declare `<Owner>::DryInitializerMixin` and include it in every class
/// whose singleton reaches `::Dry::Initializer` — it extends the module
/// itself (directly, through `Dry::Initializer[...]`, or through an
/// expanded Concern `included` block) or a superclass does.
///
/// Only classes are passed in: the gem's `extended` hook asks the
/// extender for its `superclass`, so a module raises at runtime.
pub(crate) fn synthesize(draft: &mut EnvironmentDraft, sites: &[DryInitializerSite]) {
    if sites.is_empty() {
        return;
    }
    let mut roots: FxHashSet<TypeName> = FxHashSet::default();
    let mut bracket_roots: Vec<TypeName> = Vec::new();
    {
        let names = draft.names();
        let dry_initializer = names.parse_type_name(DRY_INITIALIZER);
        let all_names = collect_declared_class_module_names(draft);
        if !all_names.contains(&dry_initializer) {
            return;
        }
        let aliases = alias_map(draft);
        let resolver = TypeNameResolver::new(&all_names, &aliases, names);
        for site in sites {
            for extend in &site.body.extends {
                if resolve_mixin_argument(
                    &resolver,
                    &aliases,
                    names,
                    &extend.raw_module_name,
                    &site.lexical_context,
                ) != Some(dry_initializer)
                {
                    continue;
                }
                roots.insert(site.target);
                if extend.bracket && !bracket_roots.contains(&site.target) {
                    bracket_roots.push(site.target);
                }
            }
        }
    }
    if roots.is_empty() {
        return;
    }

    let supers: FxHashMap<TypeName, TypeName> = resolved_super_edges(draft).into_iter().collect();
    let mut owners: Vec<TypeName> = Vec::new();
    let mut by_owner: FxHashMap<TypeName, Vec<&DryInitializerBody>> = FxHashMap::default();
    // Concern blocks first: they run at the `include` line, which by
    // convention opens the class body, ahead of the class's own calls.
    for from_concern in [true, false] {
        for site in sites
            .iter()
            .filter(|site| site.from_concern == from_concern)
        {
            by_owner
                .entry(site.target)
                .or_insert_with(|| {
                    owners.push(site.target);
                    Vec::new()
                })
                .push(&site.body);
        }
    }

    let context: Context = Arc::from([draft.names().absolute_root()]);
    for owner in owners {
        let Some(chain) = extending_chain(owner, &roots, &supers) else {
            continue;
        };
        let own: Vec<&Definition> = by_owner[&owner]
            .iter()
            .flat_map(|body| body.definitions.iter())
            .collect();
        if own.is_empty() && !roots.contains(&owner) {
            continue;
        }
        let bodies_of = |class: &TypeName| by_owner.get(class).into_iter().flatten().copied();
        let initialize = (!chain
            .iter()
            .any(|class| bodies_of(class).any(|body| body.defines_initialize)))
        .then(|| {
            initialize_method_type(
                draft.names(),
                chain
                    .iter()
                    .flat_map(|class| bodies_of(class).flat_map(|body| body.definitions.iter())),
            )
        });
        synthesize_owner(
            draft,
            Arc::clone(&context),
            owner,
            &own,
            initialize,
            bracket_roots.contains(&owner),
        );
    }
}

/// `owner`'s super chain from the topmost class whose singleton reaches
/// `::Dry::Initializer` down to `owner` itself — the classes whose
/// definitions `Config#final_definitions` merges. `None` when `owner`
/// never reaches the module.
fn extending_chain(
    owner: TypeName,
    roots: &FxHashSet<TypeName>,
    supers: &FxHashMap<TypeName, TypeName>,
) -> Option<Vec<TypeName>> {
    let mut chain = Vec::new();
    let mut topmost_root = None;
    let mut current = Some(owner);
    while let Some(class) = current {
        if chain.contains(&class) {
            break;
        }
        chain.push(class);
        if roots.contains(&class) {
            topmost_root = Some(chain.len());
        }
        current = supers.get(&class).copied();
    }
    chain.truncate(topmost_root?);
    chain.reverse();
    Some(chain)
}

/// `Builders::Signature` over `Config#final_definitions`: `definitions`
/// run from the topmost class down, and a redefinition replaces the
/// earlier entry of the same source name in place.
fn initialize_method_type<'a>(
    names: &NameTable,
    definitions: impl Iterator<Item = &'a Definition>,
) -> MethodType {
    let mut merged: Vec<&Definition> = Vec::new();
    let mut opaque = false;
    for definition in definitions {
        let Some(source) = definition.source.as_deref() else {
            // An unreadable `option` only adds a keyword `**untyped`
            // already accepts; an unreadable `param` hides the arity.
            opaque |= !definition.option;
            continue;
        };
        match merged
            .iter_mut()
            .find(|existing| existing.source.as_deref() == Some(source))
        {
            Some(existing) => *existing = definition,
            None => merged.push(definition),
        }
    }

    let positional = || FunctionParam {
        ty: Box::new(untyped()),
        name: None,
        location: None,
    };
    let mut function = FunctionType {
        required_positionals: Vec::new(),
        optional_positionals: Vec::new(),
        rest_positionals: Some(Box::new(positional())),
        trailing_positionals: Vec::new(),
        required_keywords: Vec::new(),
        optional_keywords: Vec::new(),
        rest_keywords: Some(Box::new(positional())),
        return_type: Box::new(Type::Base(BaseType {
            kind: BaseTypeKind::Void,
            location: None,
        })),
    };
    if !opaque {
        for definition in merged {
            let source = definition.source.as_deref().unwrap_or_default();
            match (definition.option, definition.optional) {
                (false, false) => function.required_positionals.push(positional()),
                (false, true) => function.optional_positionals.push(positional()),
                (true, optional) => {
                    // A source such as `""` or `1` is a valid hash key
                    // but not a keyword; `**untyped` takes it.
                    if !is_identifier(source) {
                        continue;
                    }
                    let keyword = KeywordParam {
                        name: names.intern_symbol(source),
                        param: positional(),
                    };
                    if optional {
                        function.optional_keywords.push(keyword);
                    } else {
                        function.required_keywords.push(keyword);
                    }
                }
            }
        }
    }
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(function),
        block: None,
        location: None,
    }
}

fn synthesize_owner(
    draft: &mut EnvironmentDraft,
    context: Context,
    owner: TypeName,
    definitions: &[&Definition],
    initialize: Option<MethodType>,
    bracket: bool,
) {
    let names = draft.names();
    let module_name = nested_name(names, owner, MIXIN);

    // Each `param` / `option` re-`undef`s and redefines its target, so
    // the last definition of a name decides the reader.
    let mut readers: Vec<(&str, Reader)> = Vec::new();
    for definition in definitions {
        let Some(target) = definition.target.as_deref() else {
            continue;
        };
        match readers.iter_mut().find(|(name, _)| *name == target) {
            Some(existing) => existing.1 = definition.reader,
            None => readers.push((target, definition.reader)),
        }
    }

    let mut members: Vec<ModuleMember> = Vec::new();
    for (target, reader) in &readers {
        members.push(ModuleMember::Member(Member::InstanceVariable(
            InstanceVariableMember {
                name: names.intern_symbol(&format!("@{target}")),
                ty: untyped(),
                location: None,
                source_file: None,
                comment: None,
            },
        )));
        let visibility = match reader {
            Reader::Public => Visibility::Public,
            Reader::Private => Visibility::Private,
            Reader::None => continue,
        };
        let mut method = method_with_type(
            names.intern_symbol(target),
            MethodKind::Instance,
            MethodType {
                type_params: Vec::new(),
                function: Function::Typed(FunctionType {
                    required_positionals: Vec::new(),
                    optional_positionals: Vec::new(),
                    rest_positionals: None,
                    trailing_positionals: Vec::new(),
                    required_keywords: Vec::new(),
                    optional_keywords: Vec::new(),
                    rest_keywords: None,
                    return_type: Box::new(untyped()),
                }),
                block: None,
                location: None,
            },
        );
        if let Member::MethodDefinition(definition) = &mut method {
            definition.visibility = Some(visibility);
        }
        members.push(ModuleMember::Member(method));
    }
    if let Some(method_type) = initialize {
        members.push(ModuleMember::Member(method_with_type(
            names.intern_symbol("initialize"),
            MethodKind::Instance,
            method_type,
        )));
    }

    let mut owner_members = vec![include_member(module_name, Vec::new())];
    if bracket {
        owner_members.push(ClassMember::Member(Member::Extend(Extend {
            name: names.parse_type_name(DRY_INITIALIZER),
            args: Vec::new(),
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        })));
    }
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
        members: owner_members,
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let _ = draft.insert_module_decl(
        DeclOrigin::Synthesized(InfusionUnit::DryInitializer, None),
        Arc::clone(&context),
        module_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::DryInitializer, None),
        context,
        owner_decl,
    );
}

fn untyped() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Any { todo: false },
        location: None,
    })
}
