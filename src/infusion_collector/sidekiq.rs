//! sidekiq `include Sidekiq::Job` mixin synthesis.
//!
//! [sidekiq](https://github.com/sidekiq/sidekiq) job classes write
//! `include Sidekiq::Job`; the gem's `Sidekiq::Job.included(base)` hook
//! then wires the class-level API onto the includer at runtime. RBS has
//! no member that says "including me extends you", so gem_rbs_collection
//! leaves that `extend` for every job class to hand-write. This pass
//! writes it for them.
//!
//! The synthesized members are the static counterpart of the two
//! `included` hooks in sidekiq's `lib/sidekiq/job.rb`:
//!
//! | sidekiq (runtime hook, in execution order)      | this pass (static)                                 |
//! |-------------------------------------------------|----------------------------------------------------|
//! | `Job.included` → `base.include(Options)`        | `include ::Sidekiq::Job::Options`                  |
//! | ↳ `Options.included` → `base.extend(ClassMethods)` | `extend ::Sidekiq::Job::Options::ClassMethods`  |
//! | `Job.included` → `base.extend(ClassMethods)`    | `extend ::Sidekiq::Job::ClassMethods`              |
//!
//! The two `extend`s are emitted in that execution order on purpose:
//! a later `extend` sits closer to the singleton in Ruby's MRO, and the
//! ancestor builder mirrors that (each extended module is prepended in
//! member order), so `Job::ClassMethods` shadows `Options::ClassMethods`
//! exactly as it does at runtime — sidekiq relies on it, its
//! `Job::ClassMethods#sidekiq_options` is a bare `super` into the
//! `Options` one.
//!
//! `Options.included` also calls `sidekiq_class_attribute` for three
//! internal accessors; those are sidekiq's own bookkeeping, absent from
//! the collection types, and deliberately not synthesized.
//!
//! The `::Sidekiq::Job` module types themselves are *not* synthesized —
//! they come from the user's rbs collection, and when they are absent
//! the include site never resolves to `::Sidekiq::Job`, so this pass
//! does nothing and the original `Ruby::NoMethod` diagnostics stay
//! (infusion design goal: crema does not ship type definitions).
//!
//! Include sites are matched after RBS class/module alias normalization
//! (`module Worker = Job` in the collection), so the legacy
//! `include Sidekiq::Worker` spelling reaches the same synthesis without
//! a second name being hard-coded here.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, Declaration, Member, ModuleMember,
};
use crate::ast::members::ExtendMember as Extend;
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, Context, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassOrModuleAliasEntry, ModuleAliasDeclaration,
};
use crate::environment::resolution::TypeNameResolver;
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    collect_declared_class_module_names, include_member,
};
use crate::name::NameTable;
use crate::type_name::TypeName;

const SIDEKIQ_JOB: &str = "::Sidekiq::Job";
const SIDEKIQ_JOB_CLASS_METHODS: &str = "::Sidekiq::Job::ClassMethods";
const SIDEKIQ_JOB_OPTIONS: &str = "::Sidekiq::Job::Options";
const SIDEKIQ_JOB_OPTIONS_CLASS_METHODS: &str = "::Sidekiq::Job::Options::ClassMethods";

/// One receiver-less `include X` in a class body, as the collector
/// recorded it: the including class and the namespace-qualified
/// candidates for `X` (innermost first).
pub(crate) struct SidekiqSite {
    pub(crate) target: TypeName,
    pub(crate) module_names: Vec<TypeName>,
}

/// Attach the three sidekiq mixin members to every class whose include
/// site resolves (through RBS aliases) to `::Sidekiq::Job`.
///
/// Only class targets are considered by the caller: Ruby fires
/// `included` on a module includer too, but the resulting `extend`
/// lands on that module's singleton and never reaches the classes that
/// include the module later — synthesizing there would hide a real
/// runtime `NoMethodError`.
pub(crate) fn synthesize(draft: &mut EnvironmentDraft, sites: &[SidekiqSite]) {
    if sites.is_empty() {
        return;
    }
    let targets = {
        let names = draft.names();
        let job = names.parse_type_name(SIDEKIQ_JOB);
        let all_names = collect_declared_class_module_names(draft);
        let aliases = alias_map(draft);
        let resolver = TypeNameResolver::new(&all_names, &aliases, names);
        let mut seen = FxHashSet::default();
        let mut targets = Vec::new();
        for site in sites {
            // First candidate that names a declared class/module or an
            // alias — the same innermost-first order Ruby's constant
            // lookup uses for the include argument.
            let Some(resolved) = site
                .module_names
                .iter()
                .find_map(|candidate| resolver.try_resolve_typename(*candidate, &[]))
            else {
                continue;
            };
            if normalize_alias(resolved, &aliases, &resolver) != job {
                continue;
            }
            if seen.insert(site.target) {
                targets.push(site.target);
            }
        }
        targets
    };
    if targets.is_empty() {
        return;
    }
    let context: Context = Arc::from([draft.names().absolute_root()]);
    for target in targets {
        synthesize_class(draft, Arc::clone(&context), target);
    }
}

type AliasMap = FxHashMap<TypeName, (String, Context)>;

/// `TypeNameResolver` alias input over both layers: the A draft's alias
/// decls (raw `old_name` + declaring context) and the G snapshot's
/// already-resolved alias entries. Same construction as the freeze-time
/// resolver in `EnvironmentDraft::build`, rebuilt here because this
/// pass runs before freeze — which is also why the draft's nested
/// signature aliases (`module Sidekiq; module Worker = Job; end`) have
/// to be walked out of their parents' members: `build`'s flatten pass
/// is what lifts them into `class_alias_decls`, and it has not run yet.
fn alias_map(draft: &EnvironmentDraft) -> AliasMap {
    let names = draft.names();
    let mut aliases = FxHashMap::default();
    for entry in draft.class_alias_decls.values() {
        aliases.insert(
            entry.name,
            (entry.decl.old_name_raw(names), Arc::clone(&entry.context)),
        );
    }
    for (name, entry) in &draft.class_decls {
        match entry {
            ClassOrModuleDraft::Class(entry) => {
                for (_file, context, decl) in &entry.context_decls {
                    if let ClassDeclarationDraft::Signature(decl) = decl {
                        let inner = nested_context(context, *name);
                        for member in &decl.members {
                            if let ClassMember::Declaration(decl) = member {
                                collect_nested_alias(decl, &inner, names, &mut aliases);
                            }
                        }
                    }
                }
            }
            ClassOrModuleDraft::Module(entry) => {
                for (_file, context, decl) in &entry.context_decls {
                    if let ModuleDeclarationDraft::Signature(decl) = decl {
                        let inner = nested_context(context, *name);
                        for member in &decl.members {
                            if let ModuleMember::Declaration(decl) = member {
                                collect_nested_alias(decl, &inner, names, &mut aliases);
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(g) = draft.g.as_ref() {
        for (name, entry) in g.alias_entries() {
            let (old_name, context) = match entry {
                ClassOrModuleAliasEntry::Class(e) => match &e.decl {
                    ClassAliasDeclaration::Signature(s) => (s.old_name, e.context()),
                    ClassAliasDeclaration::Ruby(_) => continue,
                },
                ClassOrModuleAliasEntry::Module(e) => match &e.decl {
                    ModuleAliasDeclaration::Signature(s) => (s.old_name, e.context()),
                    ModuleAliasDeclaration::Ruby(_) => continue,
                },
            };
            aliases.insert(
                *name,
                (names.display_type_name(old_name), Arc::clone(context)),
            );
        }
    }
    aliases
}

fn nested_context(outer: &Context, owner: TypeName) -> Context {
    let mut context = outer.to_vec();
    context.push(owner);
    Arc::from(context)
}

/// Nested signature aliases keep an absolute `new_name` from the
/// parser (like nested class / module names); only the RHS is raw and
/// needs the declaring context to resolve.
fn collect_nested_alias(
    decl: &Declaration,
    context: &Context,
    names: &NameTable,
    aliases: &mut AliasMap,
) {
    match decl {
        Declaration::ClassAlias(a) => {
            aliases.insert(
                a.new_name,
                (names.display_type_name(a.old_name), Arc::clone(context)),
            );
        }
        Declaration::ModuleAlias(a) => {
            aliases.insert(
                a.new_name,
                (names.display_type_name(a.old_name), Arc::clone(context)),
            );
        }
        Declaration::Class(c) => {
            let inner = nested_context(context, c.name);
            for member in &c.members {
                if let ClassMember::Declaration(decl) = member {
                    collect_nested_alias(decl, &inner, names, aliases);
                }
            }
        }
        Declaration::Module(m) => {
            let inner = nested_context(context, m.name);
            for member in &m.members {
                if let ModuleMember::Declaration(decl) = member {
                    collect_nested_alias(decl, &inner, names, aliases);
                }
            }
        }
        Declaration::Interface(_)
        | Declaration::TypeAlias(_)
        | Declaration::Constant(_)
        | Declaration::Global(_) => {}
    }
}

/// Follow `name` through the alias chain to the declared class/module
/// it stands for (`::Sidekiq::Worker` → `::Sidekiq::Job`). Stops at the
/// first non-alias, at an unresolvable alias target, or on a cycle —
/// in every case returning the last name reached, which the caller
/// compares against `::Sidekiq::Job` and simply fails to match.
fn normalize_alias(
    name: TypeName,
    aliases: &AliasMap,
    resolver: &TypeNameResolver<'_>,
) -> TypeName {
    let mut visited = FxHashSet::default();
    let mut current = name;
    while let Some((raw, context)) = aliases.get(&current) {
        if !visited.insert(current) {
            break;
        }
        match resolver.try_resolve(raw, context) {
            Some(next) => current = next,
            None => break,
        }
    }
    current
}

fn synthesize_class(draft: &mut EnvironmentDraft, context: Context, target: TypeName) {
    let names = draft.names();
    let options = names.parse_type_name(SIDEKIQ_JOB_OPTIONS);
    let extend = |raw: &str| {
        ClassMember::Member(Member::Extend(Extend {
            name: names.parse_type_name(raw),
            args: Vec::new(),
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        }))
    };
    let decl = Arc::new(Class {
        name: target,
        type_params: Vec::new(),
        super_class: None,
        members: vec![
            include_member(options, Vec::new()),
            extend(SIDEKIQ_JOB_OPTIONS_CLASS_METHODS),
            extend(SIDEKIQ_JOB_CLASS_METHODS),
        ],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::Sidekiq, None),
        context,
        decl,
    );
}
