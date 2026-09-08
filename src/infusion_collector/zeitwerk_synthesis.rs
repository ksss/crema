//! Zeitwerk implicit-namespace synthesis.
//!
//! Rails apps (Zeitwerk autoloader) autovivify empty `Module` objects
//! for directory-based namespaces at load time
//! (`zeitwerk::loader::callbacks#on_dir_autoloaded` calls
//! `cref.set(Module.new)`). Crema ingests every declaration up front,
//! so it does not need to model Zeitwerk's inflector / collapse / ignore
//! logic — it needs only the resulting *fact* that an implicit namespace
//! exists. That fact is already visible in the source: any compact
//! declaration `class Staff::PagesController` at the top level demands
//! a `Staff` namespace, and so do all proper prefixes of every
//! Ruby-sourced constant name in the draft.
//!
//! This pass walks those proper prefixes and inserts an empty
//! signature-side `Module` for any prefix that has no declaration
//! anywhere (draft or G-layer snapshot). Explicit declarations are
//! preserved verbatim: this pass never overwrites, downgrades, or
//! merges — it only fills gaps. Consequently `class Admin; end` (an
//! explicit class) plus `class Admin::X` leaves `Admin` a class, and
//! `module Staff` (explicit module) plus `class Staff::X` leaves
//! `Staff` a module with its original members intact.
//!
//! Gated by [`crate::config::InfusionOptions::rails_enabled`]. In
//! standalone / non-Rails mode compact-declaration prefixes remain
//! `Ruby::UnknownConstant`, preserving plain-Ruby NameError detection.
//!
//! Non-goals: `.rbs`-sourced prefixes are excluded on purpose — a
//! namespace-less `.rbs` (`class Broken::X end` with no `module Broken`)
//! is silently accepted by rbs today, and this pass does not "rescue"
//! that mistake.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::ast::declarations::ModuleDeclaration as Module;
use crate::ast::ruby::declarations::ModuleDecl as RubyModuleDecl;
use crate::ast::ruby::declarations::{ClassDecl as RubyClassDecl, Declaration as RubyDeclaration};
use crate::ast::ruby::members::Member as RubyMember;
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::type_name::TypeName;

/// Synthesize empty `Module` declarations for proper prefixes of
/// Ruby-sourced class/module declarations that lack any explicit
/// declaration. Idempotent within a single pass (repeated prefixes are
/// visited once) and safe against a G snapshot (prefixes already claimed
/// by the gem layer are skipped).
pub(crate) fn synthesize(draft: &mut EnvironmentDraft) {
    let ruby_names = collect_ruby_source_names(draft);
    if ruby_names.is_empty() {
        return;
    }

    // Enumerate everything already declared in this draft (top-level
    // keys + nested Ruby-inline decls buried in each entry's members).
    // Nested Ruby-inline class/module names are not yet lifted into
    // `draft.class_decls` — that happens in `build()`'s flatten step
    // via `resolve_ruby_{class,module}_recursive`. Checking only top-
    // level keys would miss `module Outer; class Middle; end; end`
    // and synthesize `::Outer::Middle` as a Module, colliding with the
    // real Class decl at flatten time.
    let declared = collect_declared_names(draft);

    let missing_prefixes = collect_missing_prefixes(draft, &ruby_names, &declared);
    if missing_prefixes.is_empty() {
        return;
    }

    let context = Arc::from([draft.names().absolute_root()]);
    for name in missing_prefixes {
        let synthetic = Arc::new(Module {
            name,
            type_params: Vec::new(),
            self_types: Vec::new(),
            members: Vec::new(),
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        });
        // `insert_module_decl` re-runs `check_constant_namespace_free`
        // and `check_g_class_kind`; a rare race with a same-name
        // constant / alias (not covered by `collect_missing_prefixes`)
        // would surface as a `BuildError` here. Dropping it silently
        // matches the "explicit wins / never overwrite" contract — the
        // explicit declaration wins, no diagnostic is produced.
        let _ = draft.insert_module_decl(
            DeclOrigin::Synthesized(InfusionUnit::Zeitwerk, None),
            Arc::clone(&context),
            synthetic,
        );
    }
}

fn collect_ruby_source_names(draft: &EnvironmentDraft) -> Vec<TypeName> {
    let mut out = Vec::new();
    for entry in draft.class_decls.values() {
        match entry {
            ClassOrModuleDraft::Class(e) => {
                for (_, _, decl) in &e.context_decls {
                    if let ClassDeclarationDraft::Ruby(r) = decl {
                        out.push(r.class_name);
                        walk_ruby_class_source_names(r, &mut out);
                    }
                }
            }
            ClassOrModuleDraft::Module(e) => {
                for (_, _, decl) in &e.context_decls {
                    if let ModuleDeclarationDraft::Ruby(r) = decl {
                        out.push(r.module_name);
                        walk_ruby_module_source_names(r, &mut out);
                    }
                }
            }
        }
    }
    out
}

fn walk_ruby_class_source_names(decl: &RubyClassDecl, out: &mut Vec<TypeName>) {
    for member in &decl.members {
        let RubyMember::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => {
                out.push(c.class_name);
                walk_ruby_class_source_names(c, out);
            }
            RubyDeclaration::Module(m) => {
                out.push(m.module_name);
                walk_ruby_module_source_names(m, out);
            }
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}

fn walk_ruby_module_source_names(decl: &RubyModuleDecl, out: &mut Vec<TypeName>) {
    for member in &decl.members {
        let RubyMember::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => {
                out.push(c.class_name);
                walk_ruby_class_source_names(c, out);
            }
            RubyDeclaration::Module(m) => {
                out.push(m.module_name);
                walk_ruby_module_source_names(m, out);
            }
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}

/// Every absolute `TypeName` that is already claimed in the draft,
/// including nested Ruby-inline decls that have not yet been lifted
/// into `class_decls` by `build()`'s flatten step. Used by
/// `collect_missing_prefixes` to skip prefixes that would collide with
/// an existing declaration when merged during build.
fn collect_declared_names(draft: &EnvironmentDraft) -> FxHashSet<TypeName> {
    let mut set: FxHashSet<TypeName> = FxHashSet::default();
    set.extend(draft.class_decls.keys().copied());
    set.extend(draft.constant_decls.keys().copied());
    set.extend(draft.class_alias_decls.keys().copied());
    set.extend(draft.type_alias_decls.keys().copied());
    set.extend(draft.interface_decls.keys().copied());
    for entry in draft.class_decls.values() {
        walk_nested_ruby_declared(entry, &mut set);
    }
    set
}

fn walk_nested_ruby_declared(entry: &ClassOrModuleDraft, set: &mut FxHashSet<TypeName>) {
    match entry {
        ClassOrModuleDraft::Class(e) => {
            for (_, _, decl) in &e.context_decls {
                if let ClassDeclarationDraft::Ruby(r) = decl {
                    walk_ruby_class_declared(r, set);
                }
            }
        }
        ClassOrModuleDraft::Module(e) => {
            for (_, _, decl) in &e.context_decls {
                if let ModuleDeclarationDraft::Ruby(r) = decl {
                    walk_ruby_module_declared(r, set);
                }
            }
        }
    }
}

fn walk_ruby_class_declared(decl: &RubyClassDecl, set: &mut FxHashSet<TypeName>) {
    for member in &decl.members {
        let RubyMember::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => {
                set.insert(c.class_name);
                walk_ruby_class_declared(c, set);
            }
            RubyDeclaration::Module(m) => {
                set.insert(m.module_name);
                walk_ruby_module_declared(m, set);
            }
            RubyDeclaration::Constant(c) => {
                set.insert(c.constant_name);
            }
            RubyDeclaration::ClassModuleAlias(a) => {
                set.insert(a.new_name);
            }
        }
    }
}

fn walk_ruby_module_declared(decl: &RubyModuleDecl, set: &mut FxHashSet<TypeName>) {
    for member in &decl.members {
        let RubyMember::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => {
                set.insert(c.class_name);
                walk_ruby_class_declared(c, set);
            }
            RubyDeclaration::Module(m) => {
                set.insert(m.module_name);
                walk_ruby_module_declared(m, set);
            }
            RubyDeclaration::Constant(c) => {
                set.insert(c.constant_name);
            }
            RubyDeclaration::ClassModuleAlias(a) => {
                set.insert(a.new_name);
            }
        }
    }
}

fn collect_missing_prefixes(
    draft: &EnvironmentDraft,
    ruby_names: &[TypeName],
    declared: &FxHashSet<TypeName>,
) -> Vec<TypeName> {
    let mut missing = Vec::new();
    let mut seen: FxHashSet<TypeName> = FxHashSet::default();
    for name in ruby_names {
        let mut cur = *name;
        while let Some(parent) = draft.names().type_name_parent(cur) {
            if draft.names().type_name_is_root(parent) {
                break;
            }
            cur = parent;
            if !seen.insert(parent) {
                // Ancestors of an already-visited prefix are also
                // visited — no re-work needed. Bottom-up walk guarantees
                // this invariant.
                break;
            }
            if !prefix_already_declared(draft, parent, declared) {
                missing.push(parent);
            }
        }
    }
    missing
}

fn prefix_already_declared(
    draft: &EnvironmentDraft,
    name: TypeName,
    declared: &FxHashSet<TypeName>,
) -> bool {
    if declared.contains(&name) {
        return true;
    }
    if let Some(g) = draft.g.as_ref()
        && g.class_contains(&name)
    {
        return true;
    }
    false
}
