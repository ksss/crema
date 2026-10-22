//! `[infusion.class_methods]` — the `X::ClassMethods` idiom for modules
//! that exist only as RBS.
//!
//! `ActiveSupport::Concern` (and the hand-written
//! `def self.included(base) base.extend ClassMethods end` it codifies)
//! extends a module's nested `ClassMethods` onto every includer at
//! runtime. RBS has no member for "including me extends you", so a
//! gem's RBS can only hand-write the `extend` on the gem's own classes;
//! an application class that writes `include ActiveModel::Model` gets
//! nothing, and every `validates` on it is a `Ruby::NoMethod`.
//!
//! This pass writes that `extend` for the includer, as a Ruby decl
//! anchored at the `include` call (`prepend M` gets a singleton
//! prepend instead). The rules follow the runtime:
//!
//! - **What counts**: a module on the include chain that declares a
//!   nested module named exactly `ClassMethods`, whatever the module's
//!   source (gem RBS, `sig/`, inline Ruby, or the strict Concern pass's
//!   own `class_methods do` synthesis). `extend ActiveSupport::Concern`
//!   is not required — most hand-written RBS omits it.
//! - **How far the chain goes**: `include` members only, never
//!   `prepend`. A module declared in Ruby is followed only when it
//!   extends `ActiveSupport::Concern`: `append_features` forwards a
//!   Concern's dependencies to the final includer, a plain module's
//!   `include` lands on the plain module itself. A module known only
//!   from RBS cannot be told apart, so it is followed unconditionally
//!   (`ActiveModel::Model → Validations`).
//! - **Order**: dependencies first, the included module's own
//!   `ClassMethods` last. The ancestor builder prepends each `extend`
//!   in member order, so the last one is closest to the singleton —
//!   the same MRO Rails produces by including dependencies before
//!   extending its own `ClassMethods`.
//! - **Targets**: classes and modules alike, except a module that is
//!   itself a Ruby Concern: Rails defers there until the final
//!   includer, which the chain walk reaches from that includer's site.
//! - **Dedupe**: an `extend` / singleton prepend the target already
//!   carries — hand-written, from the strict Concern pass, from the
//!   sidekiq pass — is not emitted again. Nothing is ever removed.
//!
//! Only the `include` / `prepend` sites the collector recorded from
//! Ruby are triggers; an `include` written in RBS is the RBS author's,
//! who can write the `extend` next to it. What a `self.included` hook
//! does beyond `base.extend ClassMethods` (sidekiq's `base.include
//! Options`) is not interpreted (specs/infusion.md Non-Goal).

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::declarations::{ClassDeclaration as SigClass, ModuleDeclaration as SigModule};
use crate::ast::declarations::{
    ClassMember as SigClassMember, Declaration as SigDeclaration, Member as SigMember,
    ModuleMember as SigModuleMember,
};
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::declarations::{ClassDecl, Declaration, ModuleDecl};
use crate::ast::ruby::members::{ExtendMember, Member, MixinMember, PrependMember};
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::frozen::{ClassDeclaration, ClassOrModule, ModuleDeclaration};
use crate::environment::resolution::TypeNameResolver;
use crate::infusion_collector::active_record_synthesis::collect_declared_class_module_names;
use crate::infusion_collector::pipeline::{
    ConcernSiteKind, InfusionOwnerKind, SyntheticClassMethodsMixinKind,
};
use crate::infusion_collector::sidekiq::{
    AliasMap, alias_map, normalize_alias, resolve_mixin_argument,
};
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;

const CLASS_METHODS: &str = "ClassMethods";

/// One receiver-less `include X` / `prepend X` in a class or module
/// body, as the collector recorded it, plus the index of the file it
/// came from (so the caller can insert the result with that file's
/// source).
pub(crate) struct ClassMethodsSite {
    pub(crate) source_index: usize,
    pub(crate) target: TypeName,
    pub(crate) target_kind: InfusionOwnerKind,
    pub(crate) kind: ConcernSiteKind,
    /// `X` as written, and the cref it was written under (outer-first,
    /// without the root).
    pub(crate) raw_module_name: String,
    pub(crate) lexical_context: Vec<TypeName>,
    pub(crate) location: PrismByteRange,
}

/// For every site, the `extend` (or singleton prepend) members the
/// chain walk produces, as one Ruby decl on the site's target. Returned
/// with the site's `source_index` rather than inserted here: the draft
/// is only read during the walk, and the caller owns the sources.
///
/// `ruby_concerns` are the modules the collector saw `extend
/// ActiveSupport::Concern` in — the Ruby modules whose includes are
/// forwarded, and the targets that receive nothing themselves.
pub(crate) fn synthesize(
    draft: &EnvironmentDraft,
    sites: &[ClassMethodsSite],
    ruby_concerns: &FxHashSet<TypeName>,
) -> Vec<(usize, Declaration)> {
    if sites.is_empty() {
        return Vec::new();
    }
    let names = draft.names();
    let all_names = collect_declared_class_module_names(draft);
    let aliases = alias_map(draft);
    let resolver = TypeNameResolver::new(&all_names, &aliases, names);
    let mut walker = ChainWalker {
        draft,
        names,
        resolver: &resolver,
        aliases: &aliases,
        ruby_concerns,
        class_methods: names.intern_symbol(CLASS_METHODS),
        modules: FxHashMap::default(),
    };
    // Mixins each target already has, seeded from its existing decls
    // on first use and grown with every emission, so two sites on one
    // target (`include A` then `include B`) cannot double up either.
    let mut emitted: FxHashMap<TypeName, FxHashSet<(TypeName, SyntheticClassMethodsMixinKind)>> =
        FxHashMap::default();
    let mut out = Vec::new();
    for site in sites {
        if ruby_concerns.contains(&site.target) {
            continue;
        }
        let Some(module) = resolve_mixin_argument(
            &resolver,
            &aliases,
            names,
            &site.raw_module_name,
            &site.lexical_context,
        ) else {
            continue;
        };
        if module == site.target {
            continue;
        }
        let mut chain = Vec::new();
        walker.expand(module, &mut Vec::new(), &mut chain);
        if chain.is_empty() {
            continue;
        }
        let mixin_kind = site.kind.class_methods_mixin_kind();
        let existing = emitted
            .entry(site.target)
            .or_insert_with(|| walker.existing_mixins(site.target));
        let members: Vec<Member> = chain
            .into_iter()
            .filter(|class_methods| existing.insert((*class_methods, mixin_kind)))
            .map(|class_methods| {
                let mixin = MixinMember {
                    module_name: names.display_type_name(class_methods),
                    location: site.location,
                    name_location: site.location,
                    annotation: None,
                };
                match mixin_kind {
                    SyntheticClassMethodsMixinKind::Extend => {
                        Member::Extend(ExtendMember { mixin })
                    }
                    SyntheticClassMethodsMixinKind::SingletonPrepend => {
                        Member::SingletonPrepend(PrependMember { mixin })
                    }
                }
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let decl = match site.target_kind {
            InfusionOwnerKind::Class => Declaration::Class(Arc::new(ClassDecl {
                class_name: site.target,
                name_location: site.location,
                super_class: None,
                members,
                block_body: false,
            })),
            InfusionOwnerKind::Module => Declaration::Module(Arc::new(ModuleDecl {
                module_name: site.target,
                name_location: site.location,
                members,
            })),
        };
        out.push((site.source_index, decl));
    }
    out
}

/// What the chain walk needs to know about one module, gathered once
/// per module across every declaration site (draft reopenings and the
/// G snapshot's merged entry).
struct ModuleInfo {
    /// Resolved `include` members in declaration order.
    includes: Vec<TypeName>,
    /// Declares a nested `ClassMethods` module somewhere.
    has_class_methods: bool,
    /// Has at least one Ruby declaration — the collector could see
    /// whether it is a Concern, so its includes are forwarded only if
    /// it is one.
    ruby_declared: bool,
}

struct ChainWalker<'a> {
    draft: &'a EnvironmentDraft,
    names: &'a NameTable,
    resolver: &'a TypeNameResolver<'a>,
    aliases: &'a AliasMap,
    ruby_concerns: &'a FxHashSet<TypeName>,
    class_methods: Symbol,
    /// Per-module memo: `None` for a name that is not a module.
    modules: FxHashMap<TypeName, Option<Arc<ModuleInfo>>>,
}

impl ChainWalker<'_> {
    /// Depth-first over `include` members, dependencies before the
    /// module's own `ClassMethods`, each `ClassMethods` once. `visiting`
    /// cuts include cycles (`A include B; B include A`).
    fn expand(&mut self, module: TypeName, visiting: &mut Vec<TypeName>, out: &mut Vec<TypeName>) {
        if visiting.contains(&module) {
            return;
        }
        let Some(info) = self.module_info(module) else {
            return;
        };
        if !info.ruby_declared || self.ruby_concerns.contains(&module) {
            visiting.push(module);
            for dependency in &info.includes {
                self.expand(*dependency, visiting, out);
            }
            visiting.pop();
        }
        if info.has_class_methods {
            let class_methods = self.names.append_type_name(module, self.class_methods);
            if !out.contains(&class_methods) {
                out.push(class_methods);
            }
        }
    }

    fn module_info(&mut self, module: TypeName) -> Option<Arc<ModuleInfo>> {
        if let Some(cached) = self.modules.get(&module) {
            return cached.clone();
        }
        let info = self.gather(module).map(Arc::new);
        self.modules.insert(module, info.clone());
        info
    }

    fn gather(&self, module: TypeName) -> Option<ModuleInfo> {
        let class_methods = self.names.append_type_name(module, self.class_methods);
        let mut includes = Vec::new();
        let mut has_class_methods = false;
        let mut ruby_declared = false;
        let mut found = false;
        for decl in draft_decls(self.draft, module) {
            found = true;
            match decl {
                DraftDecl::SigClass(..) | DraftDecl::RubyClass(..) => return None,
                DraftDecl::SigModule(decl, inner) => {
                    for member in &decl.members {
                        match member {
                            SigModuleMember::Member(SigMember::Include(include)) => {
                                if let Some(name) =
                                    self.resolver.try_resolve_typename(include.name, &inner)
                                {
                                    includes.push(self.normalize(name));
                                }
                            }
                            SigModuleMember::Declaration(SigDeclaration::Module(nested))
                                if nested.name == class_methods =>
                            {
                                has_class_methods = true;
                            }
                            _ => {}
                        }
                    }
                }
                DraftDecl::RubyModule(decl, inner) => {
                    ruby_declared = true;
                    for member in &decl.members {
                        match member {
                            Member::Include(include) => {
                                if let Some(name) = self
                                    .resolver
                                    .try_resolve(&include.mixin.module_name, &inner)
                                {
                                    includes.push(self.normalize(name));
                                }
                            }
                            Member::Declaration(Declaration::Module(nested))
                                if nested.module_name == class_methods =>
                            {
                                has_class_methods = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        // A `ClassMethods` the strict Concern pass synthesized from a
        // `class_methods do` block is a draft entry of its own, not a
        // member of its parent.
        if matches!(
            self.draft.class_decls.get(&class_methods),
            Some(ClassOrModuleDraft::Module(_))
        ) {
            has_class_methods = true;
        }
        if let Some(g) = self.draft.g.as_ref() {
            match g.class_entry(&module) {
                Some(ClassOrModule::Module(entry)) => {
                    found = true;
                    // The snapshot is a built environment: member names
                    // are already absolute, and every nested module is
                    // a flat entry.
                    for (_origin, _context, decl) in entry.context_decls().iter() {
                        match decl {
                            ModuleDeclaration::Signature(decl) => {
                                for member in &decl.members {
                                    if let SigModuleMember::Member(SigMember::Include(include)) =
                                        member
                                    {
                                        includes.push(self.normalize(include.name));
                                    }
                                }
                            }
                            ModuleDeclaration::Ruby(decl) => {
                                ruby_declared = true;
                                for member in &decl.members {
                                    if let Member::Include(include) = member {
                                        includes.push(self.normalize(
                                            self.names.parse_type_name(&include.mixin.module_name),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    if g.class_contains(&class_methods) {
                        has_class_methods = true;
                    }
                }
                Some(ClassOrModule::Class(_)) => return None,
                None => {}
            }
        }
        found.then_some(ModuleInfo {
            includes,
            has_class_methods,
            ruby_declared,
        })
    }

    fn normalize(&self, name: TypeName) -> TypeName {
        normalize_alias(name, self.aliases, self.resolver)
    }

    /// The `extend` / singleton-prepend mixins `target` already carries
    /// across all its declarations, resolved the way `build` will
    /// resolve them, so a synthesized duplicate is recognized whatever
    /// spelling the existing one uses.
    fn existing_mixins(
        &self,
        target: TypeName,
    ) -> FxHashSet<(TypeName, SyntheticClassMethodsMixinKind)> {
        let mut out = FxHashSet::default();
        for decl in draft_decls(self.draft, target) {
            match decl {
                DraftDecl::SigClass(decl, inner) => {
                    self.push_sig_mixins(sig_class_members(&decl.members), &inner, &mut out)
                }
                DraftDecl::SigModule(decl, inner) => {
                    self.push_sig_mixins(sig_module_members(&decl.members), &inner, &mut out)
                }
                DraftDecl::RubyClass(decl, inner) => {
                    self.push_ruby_mixins(&decl.members, &inner, &mut out)
                }
                DraftDecl::RubyModule(decl, inner) => {
                    self.push_ruby_mixins(&decl.members, &inner, &mut out)
                }
            }
        }
        if let Some(g) = self.draft.g.as_ref()
            && let Some(entry) = g.class_entry(&target)
        {
            // Built environment: names are absolute, any context does.
            let root = [self.names.absolute_root()];
            match entry {
                ClassOrModule::Class(entry) => {
                    for (_origin, _context, decl) in entry.context_decls().iter() {
                        match decl {
                            ClassDeclaration::Signature(decl) => self.push_sig_mixins(
                                sig_class_members(&decl.members),
                                &root,
                                &mut out,
                            ),
                            ClassDeclaration::Ruby(decl) => {
                                self.push_ruby_mixins(&decl.members, &root, &mut out)
                            }
                        }
                    }
                }
                ClassOrModule::Module(entry) => {
                    for (_origin, _context, decl) in entry.context_decls().iter() {
                        match decl {
                            ModuleDeclaration::Signature(decl) => self.push_sig_mixins(
                                sig_module_members(&decl.members),
                                &root,
                                &mut out,
                            ),
                            ModuleDeclaration::Ruby(decl) => {
                                self.push_ruby_mixins(&decl.members, &root, &mut out)
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn push_sig_mixins<'m>(
        &self,
        members: impl Iterator<Item = &'m SigMember>,
        inner: &[TypeName],
        out: &mut FxHashSet<(TypeName, SyntheticClassMethodsMixinKind)>,
    ) {
        for member in members {
            if let SigMember::Extend(extend) = member
                && let Some(name) = self.resolver.try_resolve_typename(extend.name, inner)
            {
                out.insert((self.normalize(name), SyntheticClassMethodsMixinKind::Extend));
            }
        }
    }

    fn push_ruby_mixins(
        &self,
        members: &[Member],
        inner: &[TypeName],
        out: &mut FxHashSet<(TypeName, SyntheticClassMethodsMixinKind)>,
    ) {
        for member in members {
            let (mixin, kind) = match member {
                Member::Extend(extend) => (&extend.mixin, SyntheticClassMethodsMixinKind::Extend),
                Member::SingletonPrepend(prepend) => (
                    &prepend.mixin,
                    SyntheticClassMethodsMixinKind::SingletonPrepend,
                ),
                _ => continue,
            };
            if let Some(name) = self.resolver.try_resolve(&mixin.module_name, inner) {
                out.insert((self.normalize(name), kind));
            }
        }
    }
}

fn sig_class_members(members: &[SigClassMember]) -> impl Iterator<Item = &SigMember> {
    members.iter().filter_map(|m| match m {
        SigClassMember::Member(m) => Some(m),
        SigClassMember::Declaration(_) => None,
    })
}

fn sig_module_members(members: &[SigModuleMember]) -> impl Iterator<Item = &SigMember> {
    members.iter().filter_map(|m| match m {
        SigModuleMember::Member(m) => Some(m),
        SigModuleMember::Declaration(_) => None,
    })
}

/// One declaration of a class or module in the A draft, with the
/// context its own members resolve in (the declaring context plus every
/// enclosing name and the decl's own — rbs's `inner_context`).
enum DraftDecl<'a> {
    SigClass(&'a SigClass, Vec<TypeName>),
    SigModule(&'a SigModule, Vec<TypeName>),
    RubyClass(&'a ClassDecl, Vec<TypeName>),
    RubyModule(&'a ModuleDecl, Vec<TypeName>),
}

/// Every declaration of `name` in the draft. Before `build` only
/// top-level names are entries in `class_decls`; a nested `module
/// Sidekiq::Job` (RBS or Ruby) lives in `module Sidekiq`'s members, so
/// this walks down from the outermost enclosing entry through each of
/// its declaration sites. Constants are the one thing the Ruby insert
/// path flattens, and they are not looked for here.
fn draft_decls(draft: &EnvironmentDraft, name: TypeName) -> Vec<DraftDecl<'_>> {
    let names = draft.names();
    let root = names.absolute_root();
    // Outermost first, `name` last.
    let mut path = Vec::new();
    let mut current = name;
    loop {
        if current == root {
            break;
        }
        path.push(current);
        match names.type_name_parent(current) {
            Some(parent) => current = parent,
            None => break,
        }
    }
    path.reverse();
    let Some((outer, rest)) = path.split_first() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match draft.class_decls.get(outer) {
        Some(ClassOrModuleDraft::Class(entry)) => {
            for (_origin, context, decl) in &entry.context_decls {
                let inner = inner_context(context, *outer);
                match decl {
                    ClassDeclarationDraft::Signature(decl) => {
                        descend(DraftDecl::SigClass(decl, inner), rest, &mut out)
                    }
                    ClassDeclarationDraft::Ruby(decl) => {
                        descend(DraftDecl::RubyClass(decl, inner), rest, &mut out)
                    }
                }
            }
        }
        Some(ClassOrModuleDraft::Module(entry)) => {
            for (_origin, context, decl) in &entry.context_decls {
                let inner = inner_context(context, *outer);
                match decl {
                    ModuleDeclarationDraft::Signature(decl) => {
                        descend(DraftDecl::SigModule(decl, inner), rest, &mut out)
                    }
                    ModuleDeclarationDraft::Ruby(decl) => {
                        descend(DraftDecl::RubyModule(decl, inner), rest, &mut out)
                    }
                }
            }
        }
        None => {}
    }
    out
}

/// Walk `rest` (the remaining nesting, outermost first) down through
/// `decl`'s members; push every leaf reached. A name declared twice at
/// one level (reopened inside the same body) yields every site.
fn descend<'a>(decl: DraftDecl<'a>, rest: &[TypeName], out: &mut Vec<DraftDecl<'a>>) {
    let Some((next, rest)) = rest.split_first() else {
        out.push(decl);
        return;
    };
    match decl {
        DraftDecl::SigClass(decl, inner) => {
            for member in &decl.members {
                if let SigClassMember::Declaration(nested) = member {
                    descend_sig(nested, &inner, *next, rest, out);
                }
            }
        }
        DraftDecl::SigModule(decl, inner) => {
            for member in &decl.members {
                if let SigModuleMember::Declaration(nested) = member {
                    descend_sig(nested, &inner, *next, rest, out);
                }
            }
        }
        DraftDecl::RubyClass(decl, inner) => descend_ruby(&decl.members, &inner, *next, rest, out),
        DraftDecl::RubyModule(decl, inner) => descend_ruby(&decl.members, &inner, *next, rest, out),
    }
}

fn descend_sig<'a>(
    nested: &'a SigDeclaration,
    outer: &[TypeName],
    next: TypeName,
    rest: &[TypeName],
    out: &mut Vec<DraftDecl<'a>>,
) {
    match nested {
        SigDeclaration::Class(decl) if decl.name == next => descend(
            DraftDecl::SigClass(decl, inner_context(outer, next)),
            rest,
            out,
        ),
        SigDeclaration::Module(decl) if decl.name == next => descend(
            DraftDecl::SigModule(decl, inner_context(outer, next)),
            rest,
            out,
        ),
        _ => {}
    }
}

fn descend_ruby<'a>(
    members: &'a [Member],
    outer: &[TypeName],
    next: TypeName,
    rest: &[TypeName],
    out: &mut Vec<DraftDecl<'a>>,
) {
    for member in members {
        let Member::Declaration(nested) = member else {
            continue;
        };
        match nested {
            Declaration::Class(decl) if decl.class_name == next => descend(
                DraftDecl::RubyClass(decl, inner_context(outer, next)),
                rest,
                out,
            ),
            Declaration::Module(decl) if decl.module_name == next => descend(
                DraftDecl::RubyModule(decl, inner_context(outer, next)),
                rest,
                out,
            ),
            _ => {}
        }
    }
}

/// `outer` plus `owner`: the context `owner`'s own members resolve in.
fn inner_context(outer: &[TypeName], owner: TypeName) -> Vec<TypeName> {
    let mut context = outer.to_vec();
    context.push(owner);
    context
}

#[cfg(test)]
mod tests {
    use super::*;

    const M_RBS: &str = "\
module M
  module ClassMethods
    def cm: () -> Integer
  end
end
";

    /// One `include M` site on `::Foo`, from `ruby` loaded as the
    /// inline-annotated declarations of the single checked file.
    fn run(ruby: &str) -> Vec<(usize, Declaration)> {
        let mut draft = EnvironmentDraft::new();
        draft.load_rbs_source(M_RBS.as_bytes()).unwrap();
        let parse_result = crate::test_fixture::parse_fixture(ruby.as_bytes());
        let diags = crate::inline_parser::load_inline_annotations(
            ruby.as_bytes(),
            &parse_result,
            None,
            &mut draft,
        );
        assert!(diags.is_empty(), "{diags:?}");
        let names = draft.names();
        let site = ClassMethodsSite {
            source_index: 0,
            target: names.parse_type_name("::Foo"),
            target_kind: InfusionOwnerKind::Class,
            kind: ConcernSiteKind::Include,
            raw_module_name: "M".to_string(),
            lexical_context: vec![names.parse_type_name("::Foo")],
            location: (0, 0),
        };
        synthesize(&draft, &[site], &FxHashSet::default())
    }

    fn extend_names(decls: &[(usize, Declaration)]) -> Vec<String> {
        decls
            .iter()
            .flat_map(|(_, decl)| match decl {
                Declaration::Class(decl) => decl.members.iter(),
                Declaration::Module(decl) => decl.members.iter(),
                _ => unreachable!(),
            })
            .filter_map(|member| match member {
                Member::Extend(extend) => Some(extend.mixin.module_name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The dedupe is only observable here: a duplicate `extend` is
    /// silent at check time (the ancestor builder does not dedupe mixins
    /// and the second one changes nothing), so the pin is on what the
    /// pass emits.
    #[test]
    fn hand_written_extend_suppresses_the_synthesized_one() {
        let without = run("class Foo\n  include M\nend\n");
        assert_eq!(extend_names(&without), ["::M::ClassMethods"]);

        let with = run("class Foo\n  include M\n  extend M::ClassMethods\nend\n");
        assert!(
            with.is_empty(),
            "nothing to add when the extend is already written: {:?}",
            extend_names(&with)
        );
    }
}
