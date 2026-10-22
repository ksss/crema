//! Gem-layer references the cold G-only resolver left relative, and
//! their re-resolution against the merged A ∪ G names at build time.
//!
//! rbs resolves every declaration with one resolver built from the whole
//! environment (`Environment#resolve_type_names`), so a gem decl's
//! `class Widget < AppBase` binds to an `AppBase` the app declares. The
//! G snapshot resolves the gem layer alone at cold time and bakes the
//! result; a reference that needed an A-layer name stays relative there
//! (the unresolved signal). Cold therefore records, per G entry, every
//! relative reference with the context and `use` clauses it was resolved
//! under ([`collect`]). The build then retries those references against
//! the live resolver and re-resolves only the entries where one now binds
//! ([`reresolve`]) — absolute names pass through the resolver unchanged,
//! so only the relative ones move. The cost follows the number of
//! recorded references, not the size of the gem layer (ADR-0028
//! Decision 3).
//!
//! A gem file under `# resolve-type-names: false` never reaches here: a
//! gem layer holding one is not snapshotted at all (`main.rs`).

use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::ast::TypeParam;
use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, InterfaceDeclaration as Interface, Member,
    ModuleDeclaration as Module, ModuleMember,
};
use crate::ast::directives::UseClause;
use crate::ast::method_type::MethodType;
use crate::ast::types::{BlockType, Function, Type};
use crate::environment::draft::{GlobalEntry, SingleEntry};
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassAliasEntry, ClassDeclaration, ClassEntry, ClassOrModule,
    ClassOrModuleAliasEntry, Environment, InterfaceEntry, ModuleAliasDeclaration, ModuleAliasEntry,
    ModuleDeclaration, ModuleEntry,
};
use crate::environment::resolution::{self, TypeNameResolver};
use crate::environment::use_map::{Table, UseMap};
use crate::name::{Name, NameTable, Symbol};
use crate::snapshot::backend::GSnapshotBackend;
use crate::type_name::TypeName;

use crate::ast::declarations::{
    ConstantDeclaration as Constant, TypeAliasDeclaration as TypeAlias,
};

/// The G entry a [`GRelativeEntry`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GEntryKey {
    Class(TypeName),
    Interface(TypeName),
    TypeAlias(TypeName),
    Constant(TypeName),
    Global(Symbol),
    /// `class A = B` / `module A = B`.
    Alias(TypeName),
}

/// Every G entry holding a relative reference, baked at cold time.
#[derive(Debug, Clone, Default)]
pub(crate) struct GRelativeRefs {
    /// `use` clauses of each gem file that holds a relative reference
    /// and has any; [`GRelativeDecl::use_group`] indexes this.
    pub(crate) use_groups: Vec<Vec<UseClause>>,
    pub(crate) entries: Vec<GRelativeEntry>,
}

impl GRelativeRefs {
    /// Whether a `use Foo::*` clause is among the groups: its expansion
    /// needs the merged name table's children index.
    pub(crate) fn has_wildcard_use(&self) -> bool {
        self.use_groups
            .iter()
            .flatten()
            .any(|c| matches!(c, UseClause::Wildcard(_)))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GRelativeEntry {
    pub(crate) key: GEntryKey,
    pub(crate) decls: Vec<GRelativeDecl>,
}

/// One declaration of a G entry that holds relative references.
#[derive(Debug, Clone)]
pub(crate) struct GRelativeDecl {
    /// Position in the entry's `context_decls`; 0 for single-decl kinds.
    pub(crate) index: u32,
    pub(crate) use_group: Option<u32>,
    /// The declaration's context — the outer one for a class / module /
    /// interface (what its super class resolves against).
    pub(crate) context: Vec<TypeName>,
    /// References resolved against [`Self::context`].
    pub(crate) outer_refs: Vec<TypeName>,
    /// References resolved against [`Self::context`] plus the entry's own
    /// name (a class / module / interface body).
    pub(crate) inner_refs: Vec<TypeName>,
}

impl GRelativeDecl {
    fn is_empty(&self) -> bool {
        self.outer_refs.is_empty() && self.inner_refs.is_empty()
    }
}

// =====================================================================
// Cold: collect the relative references of the G-only environment
// =====================================================================

/// Record every relative reference in the frozen gem-layer `env`, with
/// the `use` clauses of the file each declaration came from
/// (`file_uses`, the G draft's directives — the frozen env drops them).
/// Nested declarations are their own entries, so each decl is walked
/// without descending into them.
pub(crate) fn collect(
    env: &Environment,
    file_uses: &FxHashMap<Name, Vec<UseClause>>,
) -> GRelativeRefs {
    let names = env.names();
    let mut out = Collector {
        file_uses,
        group_of: FxHashMap::default(),
        refs: GRelativeRefs::default(),
    };

    for (name, entry) in env.class_decls() {
        let decls: Vec<GRelativeDecl> = match entry {
            ClassOrModule::Class(e) => e
                .context_decls()
                .iter()
                .enumerate()
                .filter_map(|(i, (file, ctx, d))| match d {
                    ClassDeclaration::Signature(s) => {
                        let mut sink = RefSink::new(names);
                        if let Some(sup) = &s.super_class {
                            sink.name(sup.name);
                            sink.types(&sup.args);
                        }
                        let mut inner = RefSink::new(names);
                        inner.class(s);
                        out.decl(i, file.file(), ctx, sink, inner)
                    }
                    ClassDeclaration::Ruby(_) => None,
                })
                .collect(),
            ClassOrModule::Module(e) => e
                .context_decls()
                .iter()
                .enumerate()
                .filter_map(|(i, (file, ctx, d))| match d {
                    ModuleDeclaration::Signature(s) => {
                        let mut inner = RefSink::new(names);
                        inner.module(s);
                        out.decl(i, file.file(), ctx, RefSink::new(names), inner)
                    }
                    ModuleDeclaration::Ruby(_) => None,
                })
                .collect(),
        };
        out.entry(GEntryKey::Class(*name), decls);
    }
    for (name, e) in env.interface_decls() {
        let mut inner = RefSink::new(names);
        inner.interface(&e.decl);
        let decl = out.decl(0, e.file.file(), &e.context, RefSink::new(names), inner);
        out.entry(GEntryKey::Interface(*name), decl.into_iter().collect());
    }
    for (name, e) in env.type_alias_decls() {
        let mut sink = RefSink::new(names);
        sink.type_params(&e.decl.type_params);
        sink.ty(&e.decl.ty);
        let decl = out.decl(0, e.file.file(), &e.context, sink, RefSink::new(names));
        out.entry(GEntryKey::TypeAlias(*name), decl.into_iter().collect());
    }
    for (name, e) in env.constant_decls() {
        let mut sink = RefSink::new(names);
        sink.ty(&e.decl.ty);
        let decl = out.decl(0, e.file.file(), &e.context, sink, RefSink::new(names));
        out.entry(GEntryKey::Constant(*name), decl.into_iter().collect());
    }
    for (name, e) in env.global_decls() {
        let mut sink = RefSink::new(names);
        sink.ty(&e.decl.ty);
        let decl = out.decl(0, e.file.file(), &e.context, sink, RefSink::new(names));
        out.entry(GEntryKey::Global(*name), decl.into_iter().collect());
    }
    for (name, e) in env.class_alias_decls() {
        let (file, context, old_name) = match e {
            ClassOrModuleAliasEntry::Class(a) => match &a.decl {
                ClassAliasDeclaration::Signature(s) => (a.file, &a.context, s.old_name),
                ClassAliasDeclaration::Ruby(_) => continue,
            },
            ClassOrModuleAliasEntry::Module(a) => match &a.decl {
                ModuleAliasDeclaration::Signature(s) => (a.file, &a.context, s.old_name),
                ModuleAliasDeclaration::Ruby(_) => continue,
            },
        };
        let mut sink = RefSink::new(names);
        sink.name(old_name);
        let decl = out.decl(0, file.file(), context, sink, RefSink::new(names));
        out.entry(GEntryKey::Alias(*name), decl.into_iter().collect());
    }

    // Hash-map walk order is not stable across builds; sort so the same
    // gem layer bakes the same bytes.
    out.refs.entries.sort_by_cached_key(|e| match e.key {
        GEntryKey::Class(t) => (0, t.get(), String::new()),
        GEntryKey::Interface(t) => (1, t.get(), String::new()),
        GEntryKey::TypeAlias(t) => (2, t.get(), String::new()),
        GEntryKey::Constant(t) => (3, t.get(), String::new()),
        GEntryKey::Global(s) => (4, 0, names.resolve(s).to_string()),
        GEntryKey::Alias(t) => (5, t.get(), String::new()),
    });
    out.refs
}

struct Collector<'a> {
    file_uses: &'a FxHashMap<Name, Vec<UseClause>>,
    group_of: FxHashMap<Name, u32>,
    refs: GRelativeRefs,
}

impl Collector<'_> {
    fn decl(
        &mut self,
        index: usize,
        file: Option<Name>,
        context: &[TypeName],
        outer: RefSink,
        inner: RefSink,
    ) -> Option<GRelativeDecl> {
        let decl = GRelativeDecl {
            index: index as u32,
            use_group: None,
            context: context.to_vec(),
            outer_refs: outer.refs,
            inner_refs: inner.refs,
        };
        if decl.is_empty() {
            return None;
        }
        let use_group = file.and_then(|f| {
            let clauses = self.file_uses.get(&f).filter(|c| !c.is_empty())?;
            Some(*self.group_of.entry(f).or_insert_with(|| {
                self.refs.use_groups.push(clauses.clone());
                (self.refs.use_groups.len() - 1) as u32
            }))
        });
        Some(GRelativeDecl { use_group, ..decl })
    }

    fn entry(&mut self, key: GEntryKey, decls: Vec<GRelativeDecl>) {
        if !decls.is_empty() {
            self.refs.entries.push(GRelativeEntry { key, decls });
        }
    }
}

/// Collects the distinct relative names of the reference positions the
/// resolver rewrites (`resolution::resolve_*`): the same walk, read-only.
struct RefSink<'a> {
    names: &'a NameTable,
    refs: Vec<TypeName>,
}

impl<'a> RefSink<'a> {
    fn new(names: &'a NameTable) -> Self {
        RefSink {
            names,
            refs: Vec::new(),
        }
    }

    fn name(&mut self, tn: TypeName) {
        if !self.names.type_name_is_absolute(tn) && !self.refs.contains(&tn) {
            self.refs.push(tn);
        }
    }

    fn types(&mut self, list: &[Type]) {
        for t in list {
            self.ty(t);
        }
    }

    fn ty(&mut self, ty: &Type) {
        match ty {
            Type::ClassInstance(t) => {
                self.name(t.name);
                self.types(&t.args);
            }
            Type::ClassSingleton(t) => {
                self.name(t.name);
                self.types(&t.args);
            }
            Type::Interface(t) => {
                self.name(t.name);
                self.types(&t.args);
            }
            Type::Alias(t) => {
                self.name(t.name);
                self.types(&t.args);
            }
            Type::Union(t) => self.types(&t.types),
            Type::Intersection(t) => self.types(&t.types),
            Type::Optional(t) => self.ty(&t.ty),
            Type::Tuple(t) => self.types(&t.types),
            Type::Record(t) => {
                for f in &t.fields {
                    self.ty(&f.ty);
                }
            }
            Type::Proc(p) => {
                self.function(&p.function);
                if let Some(s) = &p.self_type {
                    self.ty(s);
                }
                if let Some(b) = &p.block {
                    self.block(b);
                }
            }
            Type::Variable(_) | Type::Literal(_) | Type::Base(_) => {}
        }
    }

    fn function(&mut self, f: &Function) {
        match f {
            Function::Typed(func) => {
                let params = func
                    .required_positionals
                    .iter()
                    .chain(&func.optional_positionals)
                    .chain(func.rest_positionals.as_deref())
                    .chain(&func.trailing_positionals)
                    .chain(func.required_keywords.iter().map(|k| &k.param))
                    .chain(func.optional_keywords.iter().map(|k| &k.param))
                    .chain(func.rest_keywords.as_deref());
                for p in params {
                    self.ty(&p.ty);
                }
                self.ty(&func.return_type);
            }
            Function::Untyped(u) => self.ty(&u.return_type),
        }
    }

    fn block(&mut self, b: &BlockType) {
        self.function(&b.function);
        if let Some(s) = &b.self_type {
            self.ty(s);
        }
    }

    fn method_type(&mut self, mt: &MethodType) {
        self.type_params(&mt.type_params);
        self.function(&mt.function);
        if let Some(b) = &mt.block {
            self.block(b);
        }
    }

    fn type_params(&mut self, params: &[TypeParam]) {
        for tp in params {
            for t in [&tp.upper_bound, &tp.lower_bound, &tp.default_type]
                .into_iter()
                .flatten()
            {
                self.ty(t);
            }
        }
    }

    fn member(&mut self, m: &Member) {
        match m {
            Member::MethodDefinition(md) => {
                for o in &md.overloads {
                    self.method_type(&o.method_type);
                }
            }
            Member::Include(m) => {
                self.name(m.name);
                self.types(&m.args);
            }
            Member::Extend(m) => {
                self.name(m.name);
                self.types(&m.args);
            }
            Member::Prepend(m) => {
                self.name(m.name);
                self.types(&m.args);
            }
            Member::AttrReader(a) => self.ty(&a.ty),
            Member::AttrAccessor(a) => self.ty(&a.ty),
            Member::AttrWriter(a) => self.ty(&a.ty),
            Member::InstanceVariable(v) => self.ty(&v.ty),
            Member::ClassInstanceVariable(v) => self.ty(&v.ty),
            Member::ClassVariable(v) => self.ty(&v.ty),
            Member::Alias(_) | Member::Public(_) | Member::Private(_) => {}
        }
    }

    /// Body of a class: type params and members (the super class is an
    /// outer-context reference, collected by the caller).
    fn class(&mut self, c: &Class) {
        self.type_params(&c.type_params);
        for m in &c.members {
            if let ClassMember::Member(m) = m {
                self.member(m);
            }
        }
    }

    fn module(&mut self, m: &Module) {
        self.type_params(&m.type_params);
        for s in &m.self_types {
            self.name(s.name);
            self.types(&s.args);
        }
        for m in &m.members {
            if let ModuleMember::Member(m) = m {
                self.member(m);
            }
        }
    }

    fn interface(&mut self, i: &Interface) {
        self.type_params(&i.type_params);
        for m in &i.members {
            self.member(m);
        }
    }
}

// =====================================================================
// Build: re-resolve the entries whose references now bind
// =====================================================================

/// G entries re-resolved against the A ∪ G names, keyed like the frozen
/// maps they shadow.
#[derive(Debug, Default)]
pub(crate) struct GReresolved {
    pub(crate) classes: FxHashMap<TypeName, ClassOrModule>,
    pub(crate) interfaces: FxHashMap<TypeName, InterfaceEntry>,
    pub(crate) type_aliases: FxHashMap<TypeName, SingleEntry<TypeAlias>>,
    pub(crate) constants: FxHashMap<TypeName, SingleEntry<Constant>>,
    pub(crate) globals: FxHashMap<Symbol, GlobalEntry>,
    pub(crate) aliases: FxHashMap<TypeName, ClassOrModuleAliasEntry>,
}

/// Retry `g`'s baked relative references against `resolver` (built over
/// the merged A ∪ G names) and re-resolve every entry where at least one
/// binds now. Entries whose references still fail stay lazy in `g` —
/// only the selected ones are decoded. `use_table` backs the gem files'
/// `use` clauses; it must hold the merged names when
/// [`GRelativeRefs::has_wildcard_use`].
pub(crate) fn reresolve(
    g: &GSnapshotBackend,
    resolver: &TypeNameResolver,
    use_table: &Arc<Table>,
    names: &NameTable,
) -> GReresolved {
    let refs = &g.baked().relative_refs;
    let mut out = GReresolved::default();
    if refs.entries.is_empty() {
        return out;
    }
    let use_maps: Vec<UseMap> = refs
        .use_groups
        .iter()
        .map(|clauses| {
            let mut map = UseMap::new(Arc::clone(use_table));
            for clause in clauses {
                map.build_map(clause, names);
            }
            map
        })
        .collect();
    let resolver_of = |d: &GRelativeDecl| {
        resolver.with_use_map(d.use_group.and_then(|i| use_maps.get(i as usize)))
    };

    for entry in &refs.entries {
        let own_name = match entry.key {
            GEntryKey::Class(t) | GEntryKey::Interface(t) => Some(t),
            _ => None,
        };
        let binds_now = entry.decls.iter().any(|d| {
            let r = resolver_of(d);
            let outer = || {
                d.outer_refs
                    .iter()
                    .any(|n| r.try_resolve_typename(*n, &d.context).is_some())
            };
            let inner = || {
                let Some(own) = own_name else { return false };
                let mut ctx = d.context.clone();
                ctx.push(own);
                d.inner_refs
                    .iter()
                    .any(|n| r.try_resolve_typename(*n, &ctx).is_some())
            };
            outer() || inner()
        });
        if !binds_now {
            continue;
        }
        match entry.key {
            GEntryKey::Class(name) => {
                if let Some(e) = g.class_entry(&name) {
                    out.classes.insert(
                        name,
                        reresolve_class_or_module(e, entry, &resolver_of, names),
                    );
                }
            }
            GEntryKey::Interface(name) => {
                if let (Some(e), Some(d)) = (g.interface_entry(&name), entry.decls.first()) {
                    let decl = resolution::resolve_interface_recursive(
                        &e.decl,
                        &e.context,
                        &resolver_of(d),
                        names,
                    );
                    out.interfaces
                        .insert(name, InterfaceEntry { decl, ..e.clone() });
                }
            }
            GEntryKey::TypeAlias(name) => {
                if let (Some(e), Some(d)) = (g.type_alias_entry(&name), entry.decls.first()) {
                    let decl = Arc::new(resolution::resolve_type_alias_decl(
                        &e.decl,
                        &e.context,
                        &resolver_of(d),
                        names,
                    ));
                    out.type_aliases
                        .insert(name, SingleEntry { decl, ..e.clone() });
                }
            }
            GEntryKey::Constant(name) => {
                if let (Some(e), Some(d)) = (g.constant_entry(&name), entry.decls.first()) {
                    let decl = Arc::new(resolution::resolve_constant_decl(
                        &e.decl,
                        &e.context,
                        &resolver_of(d),
                        names,
                    ));
                    out.constants
                        .insert(name, SingleEntry { decl, ..e.clone() });
                }
            }
            GEntryKey::Global(name) => {
                if let (Some(e), Some(d)) = (g.global_entry(&name), entry.decls.first()) {
                    let decl = Arc::new(resolution::resolve_global_decl(
                        &e.decl,
                        &e.context,
                        &resolver_of(d),
                        names,
                    ));
                    out.globals.insert(name, GlobalEntry { decl, ..e.clone() });
                }
            }
            GEntryKey::Alias(name) => {
                if let (Some(e), Some(d)) = (g.alias_entries().get(&name), entry.decls.first()) {
                    let r = resolver_of(d);
                    let resolved = match e {
                        ClassOrModuleAliasEntry::Class(a) => match &a.decl {
                            ClassAliasDeclaration::Signature(s) => {
                                ClassOrModuleAliasEntry::Class(ClassAliasEntry {
                                    decl: ClassAliasDeclaration::Signature(Arc::new(
                                        resolution::resolve_class_alias_decl(
                                            s, &a.context, &r, names,
                                        ),
                                    )),
                                    ..a.clone()
                                })
                            }
                            ClassAliasDeclaration::Ruby(_) => continue,
                        },
                        ClassOrModuleAliasEntry::Module(a) => match &a.decl {
                            ModuleAliasDeclaration::Signature(s) => {
                                ClassOrModuleAliasEntry::Module(ModuleAliasEntry {
                                    decl: ModuleAliasDeclaration::Signature(Arc::new(
                                        resolution::resolve_module_alias_decl(
                                            s, &a.context, &r, names,
                                        ),
                                    )),
                                    ..a.clone()
                                })
                            }
                            ModuleAliasDeclaration::Ruby(_) => continue,
                        },
                    };
                    out.aliases.insert(name, resolved);
                }
            }
        }
    }
    out
}

/// Re-resolve the recorded decls of one class / module entry, keeping
/// the others (they hold no relative reference) and the primary-decl
/// position (re-resolution never adds or drops a super class).
fn reresolve_class_or_module<'r>(
    entry: &ClassOrModule,
    rel: &GRelativeEntry,
    resolver_of: &impl Fn(&GRelativeDecl) -> TypeNameResolver<'r>,
    names: &NameTable,
) -> ClassOrModule {
    match entry {
        ClassOrModule::Class(e) => {
            let primary = e
                .context_decls
                .iter()
                .position(|(_, _, d)| match (d, &e.primary_decl) {
                    (ClassDeclaration::Signature(a), ClassDeclaration::Signature(b)) => {
                        Arc::ptr_eq(a, b)
                    }
                    _ => false,
                })
                .unwrap_or(0);
            let mut decls = e.context_decls.to_vec();
            for d in &rel.decls {
                if let Some((_, ctx, ClassDeclaration::Signature(s))) =
                    decls.get_mut(d.index as usize)
                {
                    *s = resolution::resolve_class_shallow(s, ctx, &resolver_of(d), names);
                }
            }
            ClassOrModule::Class(ClassEntry {
                name: e.name,
                primary_decl: decls[primary].2.clone(),
                context_decls: decls.into_boxed_slice(),
            })
        }
        ClassOrModule::Module(e) => {
            let primary = e
                .context_decls
                .iter()
                .position(|(_, _, d)| match (d, &e.primary_decl) {
                    (ModuleDeclaration::Signature(a), ModuleDeclaration::Signature(b)) => {
                        Arc::ptr_eq(a, b)
                    }
                    _ => false,
                })
                .unwrap_or(0);
            let mut decls = e.context_decls.to_vec();
            for d in &rel.decls {
                if let Some((_, ctx, ModuleDeclaration::Signature(s))) =
                    decls.get_mut(d.index as usize)
                {
                    *s = resolution::resolve_module_shallow(s, ctx, &resolver_of(d), names);
                }
            }
            ClassOrModule::Module(ModuleEntry {
                name: e.name,
                primary_decl: decls[primary].2.clone(),
                context_decls: decls.into_boxed_slice(),
            })
        }
    }
}
