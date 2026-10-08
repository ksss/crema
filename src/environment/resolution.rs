//! Phase 4d structured-`TypeName` resolver.
//!
//! Port of rbs's `Resolver::TypeNameResolver` against the ADR-0017
//! `TypeName` (`Namespace` + `Name` + `Kind`) representation.
//! Consumed by [`crate::environment::draft::EnvironmentDraft::build`] (Pass
//! 2, every reference-position name in the frozen AST) and by type
//! lowering (`definition_builder::type_builder::lower_type_name`), which
//! builds one over `Environment::all_names` / `aliases` for the names
//! that are parsed fresh at check time (inline assertions).
//!
//! # Two-set structure
//!
//! Mirrors rbs's `all_names` / `aliases` separation. `all_names` is the
//! set of *real* declared `TypeName`s (class / module / interface /
//! type-alias / constant decls). `aliases` is the `class_alias_decls`
//! map keyed by alias `new_name`, valued by `(raw old_name, declaration
//! context)`. The two sets are disjoint, reflecting RBS semantics that a
//! `class Foo = Bar` is a name-substitution directive rather than a new
//! type. Resolution consults `all_names` first (`has_type_name?`), then
//! `aliases` (`aliased_name?`); when an alias is hit as a namespace
//! segment, [`TypeNameResolver::normalize_alias_chain`] re-resolves the
//! RHS in its declaration context, threading a `visited` set for cycle
//! detection (mirrors rbs `normalize_namespace`).
//!
//! Final-type alias normalization (`Foo → Bar` when `Foo` is the
//! complete reference) intentionally stays with Phase 4e
//! (`normalize_module_name`): [`TypeNameResolver::resolve_simple_name`]
//! returns the alias `new_name` unchanged so Phase 4e remains the single
//! owner of that transformation. Unresolved references fall back to a
//! relative `TypeName` (unresolved signal,
//! `namespace.is_absolute() == false`).
//!
//! Type-variable disambiguation is the parser/ast_builder's job
//! (`Type::TypeVariable` is a separate variant); this resolver only
//! processes `ClassInstance` / `ClassSingleton` / `Alias` / `Interface`
//! slots and leaves `TypeVariable` untouched.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::ast::TypeParam;
use crate::ast::declarations::{
    ClassAliasDeclaration as ClassAlias, ClassDeclaration as Class, ClassMember,
    ClassSuper as Super, ConstantDeclaration as Constant, Declaration, GlobalDeclaration as Global,
    InterfaceDeclaration as Interface, Member, ModuleAliasDeclaration as ModuleAlias,
    ModuleDeclaration as Module, ModuleMember, ModuleSelf as SelfType,
    TypeAliasDeclaration as TypeAlias,
};
use crate::ast::members::{
    AttrAccessorMember, AttrReaderMember, AttrWriterMember, ExtendMember, IncludeMember,
    MethodDefinitionMember, MethodDefinitionOverload, PrependMember,
};
use crate::ast::method_type::MethodType;
use crate::ast::ruby::declarations::{
    ClassDecl as RubyClassDecl, ClassModuleAliasDecl as RubyClassModuleAliasDecl,
    Declaration as RubyDeclaration, ModuleDecl as RubyModuleDecl, SuperClass as RubySuperClass,
};
use crate::ast::ruby::members::{
    BlockEntry, DoubleSplatRestEntry, ExplicitAnnotation, ExtendMember as RubyExtendMember,
    IncludeMember as RubyIncludeMember, Member as RubyMember, MethodTypeAnnotation,
    MixinMember as RubyMixinMember, PositionalEntry, PrependMember as RubyPrependMember,
    SplatRestEntry, TypeAnnotations,
};
use crate::ast::types::{
    AliasType, BlockType, ClassInstanceType, ClassSingletonType, Function, FunctionParam,
    FunctionType, InterfaceType, IntersectionType, KeywordParam, OptionalType, ProcType,
    RecordField, RecordType, TupleType, Type, UnionType, UntypedFunctionType,
};
use crate::environment::DeclOrigin;
use crate::environment::use_map::UseMap;
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;

/// Resolve a raw type-name string against the enclosing context.
///
/// `all_names` holds every *real* declared `TypeName` (class / module /
/// interface / type-alias / constant decls; nested decls included after
/// prefix walking). `aliases` holds `class_alias_decls`, keyed by the
/// absolute alias `new_name` and valued by the raw `old_name` plus its
/// declaration context. The two sets are disjoint (mirroring rbs).
///
/// Unknown raws fall back to a relative `TypeName` (unresolved signal)
/// via [`absolute_type_name`].
pub(crate) struct TypeNameResolver<'a> {
    all_names: &'a FxHashSet<TypeName>,
    aliases: &'a FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    names: &'a NameTable,
    /// File-scoped `use` directive alias table. When present,
    /// [`try_resolve`](Self::try_resolve) and
    /// [`try_resolve_typename`](Self::try_resolve_typename) consult it
    /// before falling back to lexical resolution, mirroring rbs's
    /// `Environment#absolute_type_name`:
    /// `resolver.resolve_type_name(map.resolve(name), context:)`.
    /// Per-file callers rebuild the resolver via
    /// [`with_use_map`](Self::with_use_map) so the same `all_names` /
    /// `aliases` references are reused.
    use_map: Option<&'a UseMap>,
    /// When set, every `try_resolve*` returns `None` immediately,
    /// causing callers to fall back to a raw relative `TypeName` and
    /// effectively bypassing type-name resolution for the current
    /// source file. Mirrors rbs's `# resolve-type-names: false`
    /// branch (`decls = source.declarations`); see
    /// [`crate::ast::directives::ResolveTypeNamesDirective`].
    bypass: bool,
}

impl<'a> TypeNameResolver<'a> {
    pub(crate) fn new(
        all_names: &'a FxHashSet<TypeName>,
        aliases: &'a FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
        names: &'a NameTable,
    ) -> Self {
        Self {
            all_names,
            aliases,
            names,
            use_map: None,
            bypass: false,
        }
    }

    /// Return a copy of this resolver with `use_map` attached. Cheap:
    /// every field is a reference. Per-file resolution flows build a
    /// resolver once, then call `with_use_map(...)` per source file
    /// before walking that file's declarations.
    pub(crate) fn with_use_map(&self, use_map: Option<&'a UseMap>) -> Self {
        Self {
            all_names: self.all_names,
            aliases: self.aliases,
            names: self.names,
            use_map,
            bypass: self.bypass,
        }
    }

    /// Return a copy of this resolver in **bypass mode**: every
    /// `try_resolve*` call returns `None` so the AST keeps the raw
    /// (parser-emitted) type names. Used for files that opt out of
    /// resolution via `# resolve-type-names: false`.
    pub(crate) fn with_bypass(&self) -> Self {
        Self {
            all_names: self.all_names,
            aliases: self.aliases,
            names: self.names,
            use_map: None,
            bypass: true,
        }
    }

    /// Attempt to resolve `raw` against `context`. Returns `Some` when `raw`
    /// matches a declared `TypeName` in `all_names` or an alias `new_name`
    /// in `aliases`; `None` when no match is found.
    ///
    /// The interned-ID representation derives kind from the trailing
    /// segment's spelling (rbs #2964), so the resolver no longer threads a
    /// `Kind` argument: a raw `foo` can only ever match an alias decl, a
    /// raw `_Foo` an interface decl, exactly as in rbs Ruby.
    ///
    /// Callers that always need a `TypeName` should go through
    /// [`absolute_type_name`], which wraps the `None` case with
    /// [`relative_typename_from_raw`].
    pub(crate) fn try_resolve(&self, raw: &str, context: &[TypeName]) -> Option<TypeName> {
        if self.bypass {
            return None;
        }
        // File-scoped use directive rewriting (mirrors rbs's
        // `Environment#absolute_type_name`, which calls
        // `resolver.resolve_type_name(map.resolve(name), context:)`).
        if let Some(use_map) = self.use_map {
            let tentative = self.names.parse_type_name(raw);
            if let Some(rewritten) = use_map.resolve(tentative, self.names)
                && self.contains_or_aliased(rewritten)
            {
                return Some(rewritten);
            }
        }

        let (is_absolute, parts) = parse_raw_path(raw);
        if parts.is_empty() {
            return None;
        }

        if parts.len() == 1 {
            let head = self.names.intern_symbol(parts[0]);
            self.resolve_simple_name(head, is_absolute, context)
        } else {
            let (last, ns_parts) = parts.split_last().unwrap();
            let mut visited = FxHashSet::default();
            let ns_resolved =
                self.walk_class_segments(is_absolute, ns_parts, context, &mut visited)?;
            let last_name = self.names.intern_symbol(last);
            let candidate = self.names.append_type_name(ns_resolved, last_name);
            if self.contains_or_aliased(candidate) {
                Some(candidate)
            } else {
                None
            }
        }
    }

    /// Segment-aware sibling of [`Self::try_resolve`]: resolves an
    /// already-built `TypeName` against `context` without the
    /// stringify-then-reparse round trip the `&str` API would impose
    /// on callers that already hold a `TypeName` (`SuperClass.type_name`,
    /// `ClassModuleAliasDecl.infered_old_name`). The semantics match
    /// `try_resolve(&names.display_type_name(tn), tn.kind, context)` —
    /// `names.type_name_is_absolute(tn)` provides the `is_absolute` signal
    /// rbs's resolver consults — but the segments stay as interned
    /// `Symbol`s the whole way through.
    pub(crate) fn try_resolve_typename(
        &self,
        tn: TypeName,
        context: &[TypeName],
    ) -> Option<TypeName> {
        if self.bypass {
            return None;
        }
        // File-scoped use directive rewriting; see [`try_resolve`].
        if let Some(use_map) = self.use_map
            && let Some(rewritten) = use_map.resolve(tn, self.names)
            && self.contains_or_aliased(rewritten)
        {
            return Some(rewritten);
        }

        let is_absolute = self.names.type_name_is_absolute(tn);
        // Roots carry no last segment and cannot name a type.
        let last = self.names.last_segment(tn)?;
        let parent = self
            .names
            .type_name_parent(tn)
            .expect("non-root TypeName has a parent");

        if self.names.type_name_is_root(parent) {
            self.resolve_simple_name(last, is_absolute, context)
        } else {
            let segments = self.names.type_name_segments(parent);
            let mut visited = FxHashSet::default();
            let ns_resolved =
                self.walk_class_segments_syms(is_absolute, &segments, context, &mut visited)?;
            let candidate = self.names.append_type_name(ns_resolved, last);
            if self.contains_or_aliased(candidate) {
                Some(candidate)
            } else {
                None
            }
        }
    }

    /// Resolve a single segment with the requested `kind`. Mirrors rbs's
    /// `resolve_type_name` (the no-tail branch of `resolve`), consulting
    /// both `all_names` and `aliases` so an alias `new_name` resolves to
    /// itself (Phase 4e takes ownership of `new_name → old_name`
    /// normalization).
    fn resolve_simple_name(
        &self,
        head: Symbol,
        is_absolute: bool,
        context: &[TypeName],
    ) -> Option<TypeName> {
        if is_absolute {
            let tn = self
                .names
                .append_type_name(self.names.absolute_root(), head);
            return self.contains_or_aliased(tn).then_some(tn);
        }
        for ns in context.iter().rev() {
            let candidate = self.names.append_type_name(*ns, head);
            if self.contains_or_aliased(candidate) {
                return Some(candidate);
            }
        }
        let candidate = self
            .names
            .append_type_name(self.names.absolute_root(), head);
        self.contains_or_aliased(candidate).then_some(candidate)
    }

    /// Walk the namespace prefix of a multi-segment path, treating each
    /// segment as a class name. Mirrors rbs's `resolve_namespace0`:
    /// the head and each intermediate segment are run through
    /// [`Self::normalize_alias_chain`], so namespace prefixes that are
    /// class aliases are transparently rewritten to the alias target.
    fn walk_class_segments(
        &self,
        is_absolute: bool,
        parts: &[&str],
        context: &[TypeName],
        visited: &mut FxHashSet<TypeName>,
    ) -> Option<TypeName> {
        let parts_sym: Vec<Symbol> = parts.iter().map(|s| self.names.intern_symbol(s)).collect();
        self.walk_class_segments_syms(is_absolute, &parts_sym, context, visited)
    }

    /// Pre-interned counterpart to [`Self::walk_class_segments`]: walks
    /// a namespace prefix already expressed as `[Symbol]` (the shape
    /// `TypeName::namespace.segments()` returns), skipping the
    /// `intern_symbol` per-segment lookup the `&str` path performs.
    fn walk_class_segments_syms(
        &self,
        is_absolute: bool,
        parts: &[Symbol],
        context: &[TypeName],
        visited: &mut FxHashSet<TypeName>,
    ) -> Option<TypeName> {
        let head_name = parts[0];
        let tail = &parts[1..];

        let head_tn = if is_absolute {
            let tn = self
                .names
                .append_type_name(self.names.absolute_root(), head_name);
            self.contains_or_aliased(tn).then_some(tn)
        } else {
            self.resolve_head_in_context(head_name, context)
        }?;

        let mut current = self.normalize_alias_chain(head_tn, visited)?;

        for &part_name in tail {
            let candidate = self.names.append_type_name(current, part_name);
            if !self.contains_or_aliased(candidate) {
                return None;
            }
            current = self.normalize_alias_chain(candidate, visited)?;
        }
        Some(current)
    }

    /// Resolve a head class segment by walking `context` outermost-last,
    /// mirroring rbs `resolve_head_namespace`. Each candidate is checked
    /// against both `all_names` and `aliases` (`has_type_name? ||
    /// aliased_name?`).
    fn resolve_head_in_context(&self, head: Symbol, context: &[TypeName]) -> Option<TypeName> {
        for ns in context.iter().rev() {
            let candidate = self.names.append_type_name(*ns, head);
            if self.contains_or_aliased(candidate) {
                return Some(candidate);
            }
        }
        let candidate = self
            .names
            .append_type_name(self.names.absolute_root(), head);
        self.contains_or_aliased(candidate).then_some(candidate)
    }

    /// Normalize a (possibly) class-alias `TypeName` through its alias
    /// chain. Returns `name` unchanged when it is not an alias, the
    /// resolved chain target when it is, and `None` when a cycle is
    /// detected (`name` already in `visited`) or the alias RHS fails to
    /// resolve. Mirrors rbs `normalize_namespace`.
    fn normalize_alias_chain(
        &self,
        name: TypeName,
        visited: &mut FxHashSet<TypeName>,
    ) -> Option<TypeName> {
        if visited.contains(&name) {
            return None;
        }
        let (old_raw, old_context) = match self.aliases.get(&name) {
            None => return Some(name),
            Some((r, c)) => (r.clone(), Arc::clone(c)),
        };
        visited.insert(name);
        let result = self.resolve_as_class_namespace(&old_raw, &old_context, visited);
        visited.remove(&name);
        result
    }

    /// Resolve a raw string as a class-kind namespace target, threading
    /// `visited` so alias chains stay cycle-safe. Used by
    /// [`Self::normalize_alias_chain`] to re-resolve an alias RHS in its
    /// declaration context.
    fn resolve_as_class_namespace(
        &self,
        raw: &str,
        context: &[TypeName],
        visited: &mut FxHashSet<TypeName>,
    ) -> Option<TypeName> {
        let (is_absolute, parts) = parse_raw_path(raw);
        if parts.is_empty() {
            return None;
        }
        if parts.len() == 1 {
            let head_sym = self.names.intern_symbol(parts[0]);
            let head_tn = if is_absolute {
                let tn = self
                    .names
                    .append_type_name(self.names.absolute_root(), head_sym);
                self.contains_or_aliased(tn).then_some(tn)
            } else {
                self.resolve_head_in_context(head_sym, context)
            }?;
            self.normalize_alias_chain(head_tn, visited)
        } else {
            self.walk_class_segments(is_absolute, &parts, context, visited)
        }
    }

    fn contains_or_aliased(&self, tn: TypeName) -> bool {
        self.all_names.contains(&tn) || self.aliases.contains_key(&tn)
    }
}

/// Split a raw path string into `(is_absolute, segments)`.
fn parse_raw_path(raw: &str) -> (bool, Vec<&str>) {
    if let Some(stripped) = raw.strip_prefix("::") {
        if stripped.is_empty() {
            (true, Vec::new())
        } else {
            (true, stripped.split("::").collect())
        }
    } else if raw.is_empty() {
        (false, Vec::new())
    } else {
        (false, raw.split("::").collect())
    }
}

/// Aggregate resolver: mirrors `RBS::Environment#absolute_type_name`.
///
/// Calls `resolver.try_resolve` and, on `None`, falls back to
/// [`relative_typename_from_raw`]. A raw that starts with `::` is treated
/// as user-declared absolute and the fallback preserves that; an unprefixed
/// raw returns a *relative* `TypeName` (`namespace.is_absolute() == false`),
/// which is the structural signal for "unresolved reference" consumed by
/// downstream diagnostic passes.
pub(crate) fn absolute_type_name(
    resolver: &TypeNameResolver,
    raw: &str,
    context: &[TypeName],
    names: &NameTable,
) -> TypeName {
    resolver
        .try_resolve(raw, context)
        .unwrap_or_else(|| relative_typename_from_raw(raw, names))
}

/// Segment-aware sibling of [`absolute_type_name`]: same fallback
/// contract (return the source-form `TypeName` unchanged when
/// `try_resolve_typename` finds no match) without stringifying a
/// `TypeName` the caller already holds.
pub(crate) fn absolute_type_name_typename(
    resolver: &TypeNameResolver,
    name: TypeName,
    context: &[TypeName],
) -> TypeName {
    resolver.try_resolve_typename(name, context).unwrap_or(name)
}

/// Build a `TypeName` from a raw string without resolution, preserving the
/// absolute/relative distinction encoded in the leading `::`.
///
/// Used as the fallback branch of [`absolute_type_name`], mirroring rbs's
/// `|| type_name` — the original name is returned unchanged so the caller
/// can detect unresolved references via `namespace.is_absolute()`.
fn relative_typename_from_raw(raw: &str, names: &NameTable) -> TypeName {
    names.parse_type_name(raw)
}

// =====================================================================
// AST-to-AST resolution: walk a draft declaration tree and produce a new
// tree where every reference-position TypeName is rewritten to its
// resolved absolute form. Decl-side names pass through unchanged.
// =====================================================================

/// Walk a class declaration and populate two collections:
/// - `all_names` receives every real `TypeName` (this class + nested
///   class/module/interface, plus nested type-alias / constant names).
/// - `aliases` receives every nested class/module alias keyed by alias
///   `new_name`, valued by `(raw old_name string, declaration context)`.
///   The context chain is `parent_context` extended by this class's own
///   namespace so the resolver re-resolves the alias RHS from inside the
///   declaration site.
///
/// Top-level `class_alias_decls` are added by `EnvironmentDraft::build`
/// directly (their context is the top-level chain); nested aliases are
/// surfaced only through `Class.members` and reach `aliases` here.
pub(crate) fn collect_class_names(
    class: &Class,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    parent_context: &[TypeName],
) {
    all_names.insert(class.name);
    let inner_context = extend_context(parent_context, class.name);
    for member in &class.members {
        if let ClassMember::Declaration(d) = member {
            collect_declaration_names(d, names, all_names, aliases, &inner_context);
        }
    }
}

pub(crate) fn collect_module_names(
    module: &Module,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    parent_context: &[TypeName],
) {
    all_names.insert(module.name);
    let inner_context = extend_context(parent_context, module.name);
    for member in &module.members {
        if let ModuleMember::Declaration(d) = member {
            collect_declaration_names(d, names, all_names, aliases, &inner_context);
        }
    }
}

pub(crate) fn collect_interface_names(
    iface: &Interface,
    _names: &NameTable,
    acc: &mut FxHashSet<TypeName>,
) {
    acc.insert(iface.name);
}

fn collect_declaration_names(
    decl: &Declaration,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    context: &[TypeName],
) {
    match decl {
        Declaration::Class(c) => collect_class_names(c, names, all_names, aliases, context),
        Declaration::Module(m) => collect_module_names(m, names, all_names, aliases, context),
        Declaration::Interface(i) => collect_interface_names(i, names, all_names),
        Declaration::TypeAlias(t) => {
            all_names.insert(t.name);
        }
        // Constants are values, not types. Mirrors the rbs split between
        // `TypeNameResolver` (type-name space) and `Resolver::ConstantResolver`
        // (constant-name space); a constant name must not surface as a
        // resolvable type name.
        Declaration::Constant(_) | Declaration::Global(_) => {}
        Declaration::ClassAlias(a) => {
            aliases.insert(
                a.new_name,
                (names.display_type_name(a.old_name), context_to_arc(context)),
            );
        }
        Declaration::ModuleAlias(a) => {
            aliases.insert(
                a.new_name,
                (names.display_type_name(a.old_name), context_to_arc(context)),
            );
        }
    }
}

/// Resolve a single `Type` AST node, walking into composite types and
/// rewriting each reference-position `TypeName` to its canonicalized
/// absolute form. Type variables and base types pass through unchanged.
pub(crate) fn resolve_type(
    ty: &Type,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Type {
    match ty {
        Type::ClassInstance(t) => Type::ClassInstance(ClassInstanceType {
            name: resolve_type_name_slot(&t.name, context, resolver, names),
            args: resolve_type_list(&t.args, context, resolver, names),
            location: t.location,
        }),
        Type::ClassSingleton(t) => Type::ClassSingleton(ClassSingletonType {
            name: resolve_type_name_slot(&t.name, context, resolver, names),
            args: resolve_type_list(&t.args, context, resolver, names),
            location: t.location,
        }),
        Type::Interface(t) => Type::Interface(InterfaceType {
            name: resolve_type_name_slot(&t.name, context, resolver, names),
            args: resolve_type_list(&t.args, context, resolver, names),
            location: t.location,
        }),
        Type::Alias(t) => Type::Alias(AliasType {
            name: resolve_type_name_slot(&t.name, context, resolver, names),
            args: resolve_type_list(&t.args, context, resolver, names),
            location: t.location,
        }),
        Type::Union(t) => Type::Union(UnionType {
            types: resolve_type_list(&t.types, context, resolver, names),
            location: t.location,
        }),
        Type::Intersection(t) => Type::Intersection(IntersectionType {
            types: resolve_type_list(&t.types, context, resolver, names),
            location: t.location,
        }),
        Type::Optional(t) => Type::Optional(OptionalType {
            ty: Box::new(resolve_type(&t.ty, context, resolver, names)),
            location: t.location,
        }),
        Type::Tuple(t) => Type::Tuple(TupleType {
            types: resolve_type_list(&t.types, context, resolver, names),
            location: t.location,
        }),
        Type::Record(t) => Type::Record(RecordType {
            fields: t
                .fields
                .iter()
                .map(|f| RecordField {
                    key: f.key.clone(),
                    ty: resolve_type(&f.ty, context, resolver, names),
                    required: f.required,
                })
                .collect(),
            location: t.location,
        }),
        Type::Proc(proc_type) => Type::Proc(Box::new(ProcType {
            function: resolve_function_type(&proc_type.function, context, resolver, names),
            self_type: proc_type
                .self_type
                .as_ref()
                .map(|t| Box::new(resolve_type(t, context, resolver, names))),
            block: proc_type
                .block
                .as_ref()
                .map(|b| resolve_block(b, context, resolver, names)),
            location: proc_type.location,
        })),
        // Terminals: type variable references and base types pass through.
        Type::Variable(_) | Type::Literal(_) | Type::Base(_) => ty.clone(),
    }
}

fn resolve_type_list(
    list: &[Type],
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Vec<Type> {
    list.iter()
        .map(|t| resolve_type(t, context, resolver, names))
        .collect()
}

fn resolve_type_name_slot(
    slot: &TypeName,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> TypeName {
    let raw = names.display_type_name(*slot);
    absolute_type_name(resolver, &raw, context, names)
}

fn resolve_function_type(
    f: &Function,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Function {
    let resolve_param = |p: &FunctionParam| FunctionParam {
        ty: Box::new(resolve_type(&p.ty, context, resolver, names)),
        name: p.name,
        location: p.location,
    };
    let resolve_keyword_param = |kp: &KeywordParam| KeywordParam {
        name: kp.name,
        param: resolve_param(&kp.param),
    };
    match f {
        Function::Typed(func) => Function::Typed(FunctionType {
            required_positionals: func
                .required_positionals
                .iter()
                .map(&resolve_param)
                .collect(),
            optional_positionals: func
                .optional_positionals
                .iter()
                .map(&resolve_param)
                .collect(),
            rest_positionals: func
                .rest_positionals
                .as_ref()
                .map(|p| Box::new(resolve_param(p))),
            trailing_positionals: func
                .trailing_positionals
                .iter()
                .map(&resolve_param)
                .collect(),
            required_keywords: func
                .required_keywords
                .iter()
                .map(&resolve_keyword_param)
                .collect(),
            optional_keywords: func
                .optional_keywords
                .iter()
                .map(&resolve_keyword_param)
                .collect(),
            rest_keywords: func
                .rest_keywords
                .as_ref()
                .map(|p| Box::new(resolve_param(p))),
            return_type: Box::new(resolve_type(&func.return_type, context, resolver, names)),
        }),
        Function::Untyped(u) => Function::Untyped(UntypedFunctionType {
            return_type: Box::new(resolve_type(&u.return_type, context, resolver, names)),
        }),
    }
}

fn resolve_block(
    b: &BlockType,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> BlockType {
    BlockType {
        required: b.required,
        function: resolve_function_type(&b.function, context, resolver, names),
        self_type: b
            .self_type
            .as_ref()
            .map(|t| Box::new(resolve_type(t, context, resolver, names))),
    }
}

/// Resolve an `ast::method_type::MethodType`. Walks type-param bounds
/// and the function/block signatures.
pub(crate) fn resolve_method_type(
    mt: &MethodType,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> MethodType {
    MethodType {
        type_params: mt
            .type_params
            .iter()
            .map(|tp| resolve_decl_type_param(tp, context, resolver, names))
            .collect(),
        function: resolve_function_type(&mt.function, context, resolver, names),
        block: mt
            .block
            .as_ref()
            .map(|b| resolve_block(b, context, resolver, names)),
        location: None,
    }
}

/// Resolve a [`TypeParam`]'s bounds and default type.
fn resolve_decl_type_param(
    tp: &TypeParam,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> TypeParam {
    TypeParam {
        name: tp.name,
        variance: tp.variance,
        upper_bound: tp
            .upper_bound
            .as_ref()
            .map(|t| resolve_type(t, context, resolver, names)),
        lower_bound: tp
            .lower_bound
            .as_ref()
            .map(|t| resolve_type(t, context, resolver, names)),
        default_type: tp
            .default_type
            .as_ref()
            .map(|t| resolve_type(t, context, resolver, names)),
        unchecked: tp.unchecked,
        location: tp.location,
    }
}

fn resolve_super(
    m: &Super,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Super {
    let raw = names.display_type_name(m.name);
    Super {
        name: absolute_type_name(resolver, &raw, context, names),
        args: resolve_type_list(&m.args, context, resolver, names),
        location: m.location,
        source_file: m.source_file,
    }
}

fn resolve_self_type(
    m: &SelfType,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> SelfType {
    let raw = names.display_type_name(m.name);
    SelfType {
        name: absolute_type_name(resolver, &raw, context, names),
        args: resolve_type_list(&m.args, context, resolver, names),
        location: m.location,
        source_file: m.source_file,
    }
}

fn resolve_include(
    inc: &IncludeMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> IncludeMember {
    let raw = names.display_type_name(inc.name);
    IncludeMember {
        name: absolute_type_name(resolver, &raw, context, names),
        args: resolve_type_list(&inc.args, context, resolver, names),
        annotations: inc.annotations.clone(),
        location: inc.location,
        source_file: inc.source_file,
        comment: inc.comment.clone(),
    }
}

fn resolve_extend(
    ext: &ExtendMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> ExtendMember {
    let raw = names.display_type_name(ext.name);
    ExtendMember {
        name: absolute_type_name(resolver, &raw, context, names),
        args: resolve_type_list(&ext.args, context, resolver, names),
        annotations: ext.annotations.clone(),
        location: ext.location,
        source_file: ext.source_file,
        comment: ext.comment.clone(),
    }
}

fn resolve_prepend(
    pre: &PrependMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> PrependMember {
    let raw = names.display_type_name(pre.name);
    PrependMember {
        name: absolute_type_name(resolver, &raw, context, names),
        args: resolve_type_list(&pre.args, context, resolver, names),
        annotations: pre.annotations.clone(),
        location: pre.location,
        source_file: pre.source_file,
        comment: pre.comment.clone(),
    }
}

fn resolve_method_definition(
    md: &MethodDefinitionMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> MethodDefinitionMember {
    MethodDefinitionMember {
        name: md.name,
        kind: md.kind,
        overloads: md
            .overloads
            .iter()
            .map(|o| MethodDefinitionOverload {
                method_type: resolve_method_type(&o.method_type, context, resolver, names),
                annotations: o.annotations.clone(),
            })
            .collect(),
        annotations: md.annotations.clone(),
        overloading: md.overloading,
        visibility: md.visibility,
        location: md.location,
        source_file: md.source_file,
        comment: md.comment.clone(),
    }
}

fn resolve_attr_reader(
    a: &AttrReaderMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> AttrReaderMember {
    AttrReaderMember {
        name: a.name,
        ty: resolve_type(&a.ty, context, resolver, names),
        kind: a.kind,
        ivar_name: a.ivar_name.clone(),
        annotations: a.annotations.clone(),
        location: a.location,
        source_file: a.source_file,
        comment: a.comment.clone(),
        visibility: a.visibility,
    }
}

fn resolve_attr_accessor(
    a: &AttrAccessorMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> AttrAccessorMember {
    AttrAccessorMember {
        name: a.name,
        ty: resolve_type(&a.ty, context, resolver, names),
        kind: a.kind,
        ivar_name: a.ivar_name.clone(),
        annotations: a.annotations.clone(),
        location: a.location,
        source_file: a.source_file,
        comment: a.comment.clone(),
        visibility: a.visibility,
    }
}

fn resolve_attr_writer(
    a: &AttrWriterMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> AttrWriterMember {
    AttrWriterMember {
        name: a.name,
        ty: resolve_type(&a.ty, context, resolver, names),
        kind: a.kind,
        ivar_name: a.ivar_name.clone(),
        annotations: a.annotations.clone(),
        location: a.location,
        source_file: a.source_file,
        comment: a.comment.clone(),
        visibility: a.visibility,
    }
}

pub(crate) fn resolve_constant_decl(
    c: &Constant,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Constant {
    Constant {
        name: c.name,
        ty: resolve_type(&c.ty, context, resolver, names),
        annotations: c.annotations.clone(),
        location: c.location,
        comment: c.comment.clone(),
    }
}

pub(crate) fn resolve_global_decl(
    g: &Global,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Global {
    Global {
        name: g.name,
        ty: resolve_type(&g.ty, context, resolver, names),
        annotations: g.annotations.clone(),
        location: g.location,
        comment: g.comment.clone(),
    }
}

pub(crate) fn resolve_type_alias_decl(
    t: &TypeAlias,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> TypeAlias {
    TypeAlias {
        name: t.name,
        type_params: t
            .type_params
            .iter()
            .map(|tp| resolve_decl_type_param(tp, context, resolver, names))
            .collect(),
        ty: resolve_type(&t.ty, context, resolver, names),
        annotations: t.annotations.clone(),
        location: t.location,
        source_file: t.source_file,
        comment: t.comment.clone(),
    }
}

pub(crate) fn resolve_class_alias_decl(
    a: &ClassAlias,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> ClassAlias {
    let old_raw = names.display_type_name(a.old_name);
    ClassAlias {
        new_name: a.new_name,
        old_name: absolute_type_name(resolver, &old_raw, context, names),
        annotations: a.annotations.clone(),
        location: a.location,
        source_file: a.source_file,
        comment: a.comment.clone(),
    }
}

pub(crate) fn resolve_module_alias_decl(
    a: &ModuleAlias,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> ModuleAlias {
    let old_raw = names.display_type_name(a.old_name);
    ModuleAlias {
        new_name: a.new_name,
        old_name: absolute_type_name(resolver, &old_raw, context, names),
        annotations: a.annotations.clone(),
        location: a.location,
        source_file: a.source_file,
        comment: a.comment.clone(),
    }
}

/// Walk a class declaration recursively. Produces a resolved
/// `Arc<Class>` whose top-level `super_class`, mixin members, type-body
/// members, and inner-decl members all carry rewritten raws. Inner
/// `class` / `module` / `interface` declarations are also collected
/// into the `nested` map (keyed by their `TypeName`) so the
/// `EnvironmentDraft::build` flatten flow can register them as
/// top-level entries.
pub(crate) fn resolve_class_recursive(
    decl: &Arc<Class>,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> Arc<Class> {
    let self_ns = decl.name;
    let inner_context = extend_context(context, self_ns);

    let resolved_super = decl
        .super_class
        .as_ref()
        // rbs uses the **outer** context for super_class refs, not the
        // inner (post-`class` push) one. Mirrors environment.rb's
        // `outer_context` vs `inner_context` distinction.
        .map(|m| resolve_super(m, context, resolver, names));

    let resolved_type_params = decl
        .type_params
        .iter()
        .map(|tp| resolve_decl_type_param(tp, &inner_context, resolver, names))
        .collect();

    let resolved_members = decl
        .members
        .iter()
        .map(|m| resolve_class_member(m, &inner_context, resolver, names, nested, file))
        .collect();

    Arc::new(Class {
        name: decl.name,
        type_params: resolved_type_params,
        super_class: resolved_super,
        members: resolved_members,
        annotations: decl.annotations.clone(),
        location: decl.location,
        source_file: decl.source_file,
        comment: decl.comment.clone(),
    })
}

pub(crate) fn resolve_module_recursive(
    decl: &Arc<Module>,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> Arc<Module> {
    let self_ns = decl.name;
    let inner_context = extend_context(context, self_ns);

    let resolved_self_types = decl
        .self_types
        .iter()
        // `module M : _Foo` accepts both class/module and interface
        // self-types; the spelling-derived kind covers both.
        .map(|m| resolve_self_type(m, &inner_context, resolver, names))
        .collect();

    let resolved_type_params = decl
        .type_params
        .iter()
        .map(|tp| resolve_decl_type_param(tp, &inner_context, resolver, names))
        .collect();

    let resolved_members = decl
        .members
        .iter()
        .map(|m| resolve_module_member(m, &inner_context, resolver, names, nested, file))
        .collect();

    Arc::new(Module {
        name: decl.name,
        type_params: resolved_type_params,
        self_types: resolved_self_types,
        members: resolved_members,
        annotations: decl.annotations.clone(),
        location: decl.location,
        source_file: decl.source_file,
        comment: decl.comment.clone(),
    })
}

pub(crate) fn resolve_interface_recursive(
    decl: &Arc<Interface>,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Arc<Interface> {
    let self_ns = decl.name;
    let inner_context = extend_context(context, self_ns);

    let resolved_type_params = decl
        .type_params
        .iter()
        .map(|tp| resolve_decl_type_param(tp, &inner_context, resolver, names))
        .collect();

    let resolved_members = decl
        .members
        .iter()
        .map(|m| resolve_member(m, &inner_context, resolver, names))
        .collect();

    Arc::new(Interface {
        name: decl.name,
        type_params: resolved_type_params,
        members: resolved_members,
        annotations: decl.annotations.clone(),
        location: decl.location,
        source_file: decl.source_file,
        comment: decl.comment.clone(),
    })
}

fn resolve_member(
    member: &Member,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> Member {
    match member {
        Member::MethodDefinition(md) => {
            Member::MethodDefinition(resolve_method_definition(md, context, resolver, names))
        }
        Member::Include(m) => Member::Include(resolve_include(m, context, resolver, names)),
        Member::Extend(m) => Member::Extend(resolve_extend(m, context, resolver, names)),
        Member::Prepend(m) => Member::Prepend(resolve_prepend(m, context, resolver, names)),
        Member::AttrReader(a) => {
            Member::AttrReader(resolve_attr_reader(a, context, resolver, names))
        }
        Member::AttrAccessor(a) => {
            Member::AttrAccessor(resolve_attr_accessor(a, context, resolver, names))
        }
        Member::AttrWriter(a) => {
            Member::AttrWriter(resolve_attr_writer(a, context, resolver, names))
        }
        Member::Alias(a) => Member::Alias(a.clone()),
        Member::InstanceVariable(v) => {
            use crate::ast::members::InstanceVariableMember;
            Member::InstanceVariable(InstanceVariableMember {
                name: v.name,
                ty: resolve_type(&v.ty, context, resolver, names),
                location: v.location,
                source_file: v.source_file,
                comment: v.comment.clone(),
            })
        }
        Member::ClassInstanceVariable(v) => {
            use crate::ast::members::ClassInstanceVariableMember;
            Member::ClassInstanceVariable(ClassInstanceVariableMember {
                name: v.name,
                ty: resolve_type(&v.ty, context, resolver, names),
                location: v.location,
                source_file: v.source_file,
                comment: v.comment.clone(),
            })
        }
        Member::ClassVariable(v) => {
            use crate::ast::members::ClassVariableMember;
            Member::ClassVariable(ClassVariableMember {
                name: v.name,
                ty: resolve_type(&v.ty, context, resolver, names),
                location: v.location,
                source_file: v.source_file,
                comment: v.comment.clone(),
            })
        }
        Member::Public(p) => Member::Public(p.clone()),
        Member::Private(p) => Member::Private(p.clone()),
    }
}

fn resolve_member_declaration(
    decl: &Declaration,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> Declaration {
    match decl {
        Declaration::Constant(c) => {
            let resolved = resolve_constant_decl(c, context, resolver, names);
            let key = resolved.name;
            let arc = Arc::new(resolved);
            nested
                .constants
                .push((key, context_to_arc(context), Arc::clone(&arc), file));
            Declaration::Constant(arc)
        }
        Declaration::Global(g) => {
            Declaration::Global(Arc::new(resolve_global_decl(g, context, resolver, names)))
        }
        Declaration::TypeAlias(t) => {
            let resolved = resolve_type_alias_decl(t, context, resolver, names);
            let key = resolved.name;
            let arc = Arc::new(resolved);
            nested
                .type_aliases
                .push((key, context_to_arc(context), Arc::clone(&arc), file));
            Declaration::TypeAlias(arc)
        }
        Declaration::ClassAlias(a) => {
            let resolved = resolve_class_alias_decl(a, context, resolver, names);
            let key = resolved.new_name;
            let arc = Arc::new(resolved);
            nested
                .class_aliases
                .push((key, context_to_arc(context), Arc::clone(&arc), file));
            Declaration::ClassAlias(arc)
        }
        Declaration::ModuleAlias(a) => {
            let resolved = resolve_module_alias_decl(a, context, resolver, names);
            let key = resolved.new_name;
            let arc = Arc::new(resolved);
            nested
                .module_aliases
                .push((key, context_to_arc(context), Arc::clone(&arc), file));
            Declaration::ModuleAlias(arc)
        }
        Declaration::Class(c) => {
            let resolved = resolve_class_recursive(c, context, resolver, names, nested, file);
            let key = resolved.name;
            nested
                .classes
                .push((key, context_to_arc(context), Arc::clone(&resolved), file));
            Declaration::Class(resolved)
        }
        Declaration::Module(m) => {
            let resolved = resolve_module_recursive(m, context, resolver, names, nested, file);
            let key = resolved.name;
            nested
                .modules
                .push((key, context_to_arc(context), Arc::clone(&resolved), file));
            Declaration::Module(resolved)
        }
        Declaration::Interface(i) => {
            let resolved = resolve_interface_recursive(i, context, resolver, names);
            let key = resolved.name;
            nested
                .interfaces
                .push((key, context_to_arc(context), Arc::clone(&resolved), file));
            Declaration::Interface(resolved)
        }
    }
}

fn resolve_class_member(
    member: &ClassMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> ClassMember {
    match member {
        ClassMember::Member(m) => ClassMember::Member(resolve_member(m, context, resolver, names)),
        ClassMember::Declaration(d) => ClassMember::Declaration(resolve_member_declaration(
            d, context, resolver, names, nested, file,
        )),
    }
}

fn resolve_module_member(
    member: &ModuleMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> ModuleMember {
    match member {
        ModuleMember::Member(m) => {
            ModuleMember::Member(resolve_member(m, context, resolver, names))
        }
        ModuleMember::Declaration(d) => ModuleMember::Declaration(resolve_member_declaration(
            d, context, resolver, names, nested, file,
        )),
    }
}

/// One flattened nested decl: qualified key, declaration context, the decl
/// itself, and the [`DeclOrigin`] it was found under — the same value the
/// parent decl carried, threaded through so `EnvironmentDraft::build`'s
/// flatten loop can record `path_index` entries for nested decls too; see
/// [`FlattenedDecls`].
type NestedDecl<D> = (TypeName, Arc<[TypeName]>, Arc<D>, DeclOrigin);

/// Side-collection of nested decls discovered while walking a class /
/// module tree. The build flow drains this into the frozen
/// `Environment.{class,interface,type_alias,constant,class_alias}_decls`
/// maps as additional top-level entries — rbs-style flatten where the
/// outer decl's `members` list keeps the same nested `Arc`, but the
/// flat maps also get separate entries keyed by the qualified name.
#[derive(Default)]
pub(crate) struct FlattenedDecls {
    pub(crate) classes: Vec<NestedDecl<Class>>,
    pub(crate) modules: Vec<NestedDecl<Module>>,
    pub(crate) interfaces: Vec<NestedDecl<Interface>>,
    pub(crate) ruby_classes: Vec<NestedDecl<RubyClassDecl>>,
    pub(crate) ruby_modules: Vec<NestedDecl<RubyModuleDecl>>,
    pub(crate) type_aliases: Vec<NestedDecl<TypeAlias>>,
    pub(crate) constants: Vec<NestedDecl<Constant>>,
    pub(crate) class_aliases: Vec<NestedDecl<ClassAlias>>,
    pub(crate) module_aliases: Vec<NestedDecl<ModuleAlias>>,
    pub(crate) ruby_aliases: Vec<NestedDecl<RubyClassModuleAliasDecl>>,
}

/// Build a fresh `Vec<TypeName>` whose path is `context` followed by
/// `inner`. Returns `Vec` (not `Arc`) because callers reuse the chain
/// for multiple inner traversals before any single result needs sharing.
fn extend_context(context: &[TypeName], inner: TypeName) -> Vec<TypeName> {
    let mut v: Vec<TypeName> = context.to_vec();
    v.push(inner);
    v
}

/// Clone `context` into a shareable `Arc<[TypeName]>`. Use when the
/// context will be stored alongside an alias entry, draft entry, or
/// flattened decl record — anywhere the chain outlives the current
/// stack frame and needs cheap clones from there on.
fn context_to_arc(context: &[TypeName]) -> Arc<[TypeName]> {
    Arc::from(context.to_vec())
}

// =====================================================================
// Inline Ruby decl resolution (Phase 4d Stage 4)
// =====================================================================
//
// Resolves the reference-side `SuperClass.type_name` / Ruby mixin
// `module_name` / `ClassModuleAliasDecl.infered_old_name` fields that
// the inline collector stores in their source-form (relative or
// absolute) TypeName shape. The deferred-parse annotation slot
// (`type_text`) stays verbatim. Ruby def `method_type` and mixin
// `annotation` are already parsed, but Phase 4d still does not resolve
// names inside them.
// Name resolution within an annotation body is a Phase 5
// responsibility tied to annotation parsing.

pub(crate) fn collect_ruby_class_names(
    decl: &RubyClassDecl,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    parent_context: &[TypeName],
) {
    let self_tn = decl.class_name;
    let inner_context = ruby_class_inner_context(decl, parent_context);
    all_names.insert(self_tn);
    for member in &decl.members {
        collect_ruby_member_names(member, names, all_names, aliases, &inner_context);
    }
}

/// The context a Ruby class decl's members resolve in. rbs
/// `resolve_ruby_decl` always uses `[context, full_name]`; crema's
/// `Class.new` / `Struct.new` / `Data.define` block decls
/// (`ClassDecl::block_body`) keep the enclosing `context`, because Ruby's
/// cref inside such a block is the scope around the block, not the new
/// class (`Module.nesting` there does not include it).
fn ruby_class_inner_context(decl: &RubyClassDecl, context: &[TypeName]) -> Vec<TypeName> {
    if decl.block_body {
        context.to_vec()
    } else {
        extend_context(context, decl.class_name)
    }
}

pub(crate) fn collect_ruby_module_names(
    decl: &RubyModuleDecl,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    parent_context: &[TypeName],
) {
    let self_tn = decl.module_name;
    let inner_context = extend_context(parent_context, self_tn);
    all_names.insert(self_tn);
    for member in &decl.members {
        collect_ruby_member_names(member, names, all_names, aliases, &inner_context);
    }
}

fn collect_ruby_member_names(
    member: &RubyMember,
    names: &NameTable,
    all_names: &mut FxHashSet<TypeName>,
    aliases: &mut FxHashMap<TypeName, (String, Arc<[TypeName]>)>,
    context: &[TypeName],
) {
    match member {
        RubyMember::Declaration(d) => match d {
            RubyDeclaration::Class(c) => {
                collect_ruby_class_names(c, names, all_names, aliases, context)
            }
            RubyDeclaration::Module(m) => {
                collect_ruby_module_names(m, names, all_names, aliases, context)
            }
            // Constants are values, not types. Mirrors the RBS-side
            // `ClassMember::Declaration(Constant)` arm: a constant name must not
            // surface as a resolvable type name (see ConstantResolver port).
            RubyDeclaration::Constant(_) => {}
            RubyDeclaration::ClassModuleAlias(a) => {
                let new_name = a.new_name;
                let old_tn = a
                    .old_name(names)
                    .expect("nested Ruby alias decl reached collect without RHS — inline collector should have dropped it");
                let old_raw = names.display_type_name(old_tn);
                aliases.insert(new_name, (old_raw, context_to_arc(context)));
            }
        },
        // Defs / attrs / mixins do not introduce class-namespace names.
        RubyMember::Def(_)
        | RubyMember::AttrReader(_)
        | RubyMember::AttrWriter(_)
        | RubyMember::AttrAccessor(_)
        | RubyMember::Include(_)
        | RubyMember::Extend(_)
        | RubyMember::Prepend(_)
        | RubyMember::SingletonPrepend(_)
        | RubyMember::InstanceVariable(_)
        | RubyMember::ModuleSelf(_) => {}
    }
}

pub(crate) fn resolve_ruby_class_recursive(
    decl: &Arc<RubyClassDecl>,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> Arc<RubyClassDecl> {
    let inner_context = ruby_class_inner_context(decl, context);

    // SuperClass.type_name carries the source-form name written in the
    // Ruby source (relative or absolute). Resolve against the *outer*
    // context (rbs convention).
    let resolved_super = decl.super_class.as_ref().map(|s| RubySuperClass {
        type_name: absolute_type_name_typename(resolver, s.type_name, context),
        type_annotation: s.type_annotation.clone(),
        byte_range: s.byte_range,
    });

    let resolved_members = decl
        .members
        .iter()
        .map(|m| resolve_ruby_member(m, &inner_context, resolver, names, nested, file))
        .collect();

    Arc::new(RubyClassDecl {
        class_name: decl.class_name,
        name_location: decl.name_location,
        super_class: resolved_super,
        members: resolved_members,
        block_body: decl.block_body,
    })
}

pub(crate) fn resolve_ruby_module_recursive(
    decl: &Arc<RubyModuleDecl>,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> Arc<RubyModuleDecl> {
    let self_ns = decl.module_name;
    let inner_context = extend_context(context, self_ns);

    let resolved_members = decl
        .members
        .iter()
        .map(|m| resolve_ruby_member(m, &inner_context, resolver, names, nested, file))
        .collect();

    Arc::new(RubyModuleDecl {
        module_name: decl.module_name,
        name_location: decl.name_location,
        members: resolved_members,
    })
}

fn resolve_ruby_member(
    member: &RubyMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
    nested: &mut FlattenedDecls,
    file: DeclOrigin,
) -> RubyMember {
    match member {
        RubyMember::Declaration(d) => match d {
            // The parent's members and the flattened entry share one Arc
            // (rbs `resolve_ruby_decl` stores the resolved object itself in
            // the parent's members). A deep clone here would duplicate every
            // nested decl once per nesting level.
            RubyDeclaration::Class(c) => {
                let resolved =
                    resolve_ruby_class_recursive(c, context, resolver, names, nested, file);
                let key = resolved.class_name;
                nested.ruby_classes.push((
                    key,
                    context_to_arc(context),
                    Arc::clone(&resolved),
                    file,
                ));
                RubyMember::Declaration(RubyDeclaration::Class(resolved))
            }
            RubyDeclaration::Module(m) => {
                let resolved =
                    resolve_ruby_module_recursive(m, context, resolver, names, nested, file);
                let key = resolved.module_name;
                nested.ruby_modules.push((
                    key,
                    context_to_arc(context),
                    Arc::clone(&resolved),
                    file,
                ));
                RubyMember::Declaration(RubyDeclaration::Module(resolved))
            }
            RubyDeclaration::Constant(c) => {
                // ConstantDecl carries `type_text` (deferred-parse). Phase 4d
                // does not parse annotations, so the constant value passes
                // through unchanged.
                RubyMember::Declaration(RubyDeclaration::Constant(c.clone()))
            }
            RubyDeclaration::ClassModuleAlias(a) => {
                let resolved = resolve_ruby_alias_decl(a, context, resolver, names);
                let arc = Arc::new(resolved);
                let key = arc.new_name;
                nested
                    .ruby_aliases
                    .push((key, context_to_arc(context), Arc::clone(&arc), file));
                RubyMember::Declaration(RubyDeclaration::ClassModuleAlias((*arc).clone()))
            }
        },
        RubyMember::Include(m) => RubyMember::Include(RubyIncludeMember {
            mixin: resolve_ruby_mixin(&m.mixin, context, resolver, names),
        }),
        RubyMember::Extend(m) => RubyMember::Extend(RubyExtendMember {
            mixin: resolve_ruby_mixin(&m.mixin, context, resolver, names),
        }),
        RubyMember::Prepend(m) => RubyMember::Prepend(RubyPrependMember {
            mixin: resolve_ruby_mixin(&m.mixin, context, resolver, names),
        }),
        RubyMember::SingletonPrepend(m) => RubyMember::SingletonPrepend(RubyPrependMember {
            mixin: resolve_ruby_mixin(&m.mixin, context, resolver, names),
        }),
        // rbs `resolve_ruby_member` `DefMember` arm: `method_type.map_type_name`
        // against the enclosing context. The def's annotation is already
        // parsed at collect time, so its type names are absolutized here
        // like a signature `def`'s — the definition builder then lowers
        // absolute names without consulting its owner-derived context.
        RubyMember::Def(d) => {
            let mut resolved = d.clone();
            resolved.method_type =
                resolve_ruby_method_type_annotation(&d.method_type, context, resolver, names);
            RubyMember::Def(resolved)
        }
        // Attributes carry deferred-parse annotation text only (`#: T`
        // after `attr_reader :x`); the text is parsed at lowering time.
        RubyMember::AttrReader(a) => RubyMember::AttrReader(a.clone()),
        RubyMember::AttrWriter(a) => RubyMember::AttrWriter(a.clone()),
        RubyMember::AttrAccessor(a) => RubyMember::AttrAccessor(a.clone()),
        // rbs `resolve_ruby_member` `InstanceVariableMember` arm.
        RubyMember::InstanceVariable(a) => {
            let mut resolved = a.clone();
            resolved.annotation.ty = resolve_type(&a.annotation.ty, context, resolver, names);
            RubyMember::InstanceVariable(resolved)
        }
        RubyMember::ModuleSelf(a) => {
            // Rewrite the annotation's `name` to its absolute form so
            // `ancestor_builder::module_self_types_or_default`'s
            // `namespace.is_absolute()` filter keeps the entry. Args
            // are absolutized against the same enclosing context so a
            // relative type-name reference inside `Foo[Other]` (when
            // the annotation sits under `module Outer`) becomes
            // `::Outer::Other` rather than leaking out to the root
            // namespace at downstream lowering time.
            let mut resolved = a.clone();
            resolved.annotation.name =
                absolute_type_name_typename(resolver, a.annotation.name, context);
            resolved.annotation.args =
                resolve_type_list(&a.annotation.args, context, resolver, names);
            RubyMember::ModuleSelf(resolved)
        }
    }
}

/// Port of the `method_type.map_type_name` half of rbs `resolve_ruby_member`'s
/// `DefMember` arm: rewrite every type name inside a Ruby def's annotation
/// to its absolute form. Explicit shapes (`#:` / `# @rbs (T) -> U`) hold a
/// signature-style `MethodType` and reuse [`resolve_method_type`]; the
/// doc-style shape resolves each annotated slot's `Type` in place.
fn resolve_ruby_method_type_annotation(
    annotation: &MethodTypeAnnotation,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> MethodTypeAnnotation {
    let type_annotations = match &annotation.type_annotations {
        TypeAnnotations::Array(explicit) => TypeAnnotations::Array(
            explicit
                .iter()
                .map(|e| match e {
                    ExplicitAnnotation::Colon(colon) => {
                        let mut resolved = colon.clone();
                        resolved.method_type =
                            resolve_method_type(&colon.method_type, context, resolver, names);
                        ExplicitAnnotation::Colon(resolved)
                    }
                    ExplicitAnnotation::MethodTypes(mts) => {
                        let mut resolved = mts.clone();
                        resolved.overloads = mts
                            .overloads
                            .iter()
                            .map(|o| MethodDefinitionOverload {
                                method_type: resolve_method_type(
                                    &o.method_type,
                                    context,
                                    resolver,
                                    names,
                                ),
                                annotations: o.annotations.clone(),
                            })
                            .collect();
                        ExplicitAnnotation::MethodTypes(resolved)
                    }
                })
                .collect(),
        ),
        TypeAnnotations::DocStyle(doc) => {
            let resolve_ty = |ty: &Type| resolve_type(ty, context, resolver, names);
            let resolve_positional = |entry: &PositionalEntry| match entry {
                PositionalEntry::Annotated(p) => {
                    let mut resolved = p.clone();
                    resolved.param_type = resolve_ty(&p.param_type);
                    PositionalEntry::Annotated(resolved)
                }
                PositionalEntry::ByName(name) => PositionalEntry::ByName(name.clone()),
            };
            let resolve_positionals = |entries: &[PositionalEntry]| -> Vec<PositionalEntry> {
                entries.iter().map(resolve_positional).collect()
            };
            let resolve_keywords =
                |entries: &[(String, PositionalEntry)]| -> Vec<(String, PositionalEntry)> {
                    entries
                        .iter()
                        .map(|(name, entry)| (name.clone(), resolve_positional(entry)))
                        .collect()
                };
            let mut resolved = (**doc).clone();
            resolved.return_type_annotation = doc.return_type_annotation.as_ref().map(|r| {
                let mut resolved = (**r).clone();
                resolved.return_type = resolve_ty(&r.return_type);
                Box::new(resolved)
            });
            resolved.required_positionals = resolve_positionals(&doc.required_positionals);
            resolved.optional_positionals = resolve_positionals(&doc.optional_positionals);
            resolved.trailing_positionals = resolve_positionals(&doc.trailing_positionals);
            resolved.required_keywords = resolve_keywords(&doc.required_keywords);
            resolved.optional_keywords = resolve_keywords(&doc.optional_keywords);
            if let Some(SplatRestEntry::Annotated(s)) = &doc.rest_positionals {
                let mut splat = s.clone();
                splat.param_type = resolve_ty(&s.param_type);
                resolved.rest_positionals = Some(SplatRestEntry::Annotated(splat));
            }
            if let Some(DoubleSplatRestEntry::Annotated(s)) = &doc.rest_keywords {
                let mut splat = s.clone();
                splat.param_type = resolve_ty(&s.param_type);
                resolved.rest_keywords = Some(DoubleSplatRestEntry::Annotated(splat));
            }
            if let Some(BlockEntry::Annotated(b)) = &doc.block {
                let mut block = b.clone();
                block.function = resolve_function_type(&b.function, context, resolver, names);
                resolved.block = Some(BlockEntry::Annotated(block));
            }
            TypeAnnotations::DocStyle(Box::new(resolved))
        }
        TypeAnnotations::None => TypeAnnotations::None,
    };
    MethodTypeAnnotation { type_annotations }
}

/// Port of rbs `Environment#resolve_ruby_decl`'s `ClassModuleAliasDecl` arm
/// (`environment.rb:771-783`): rewrite both the inferred and the annotation's
/// explicit `type_name` to their absolute form against the enclosing
/// `context`. rbs returns a brand-new `ClassModuleAliasDecl` whose
/// `infered_old_name` and `annotation.type_name` are absolute; crema mirrors
/// that by re-building both the `infered_old_name` TypeName and the
/// `annotation` owned snapshot. The annotation's `type_name_text` stays
/// String — its TypeName lift lives in
/// `mid_owned_class_module_alias_annotation.md`.
pub(crate) fn resolve_ruby_alias_decl(
    decl: &RubyClassModuleAliasDecl,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> RubyClassModuleAliasDecl {
    let resolve_text =
        |text: &str| names.display_type_name(absolute_type_name(resolver, text, context, names));
    let resolved_annotation = decl
        .annotation
        .map_type_name_text(decl.annotation.type_name_text().map(resolve_text));
    RubyClassModuleAliasDecl {
        new_name: decl.new_name,
        name_location: decl.name_location,
        infered_old_name: decl
            .infered_old_name
            .map(|tn| absolute_type_name_typename(resolver, tn, context)),
        annotation: resolved_annotation,
        byte_range: decl.byte_range,
        old_name_location: decl.old_name_location,
        leading_comment: decl.leading_comment.clone(),
    }
}

fn resolve_ruby_mixin(
    mixin: &RubyMixinMember,
    context: &[TypeName],
    resolver: &TypeNameResolver,
    names: &NameTable,
) -> RubyMixinMember {
    // `module_name` is the relative-form module reference written at
    // the include/extend/prepend call site. Interface mixins resolve via
    // the spelling-derived kind.
    let resolved = absolute_type_name(resolver, &mixin.module_name, context, names);
    RubyMixinMember {
        module_name: names.display_type_name(resolved),
        location: mixin.location,
        name_location: mixin.name_location,
        annotation: mixin.annotation.clone(),
    }
}

