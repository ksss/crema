//! active_decorator decorator-module self-type synthesis.
//!
//! The [active_decorator](https://github.com/amatsuda/active_decorator)
//! gem extends a model instance with a same-named decorator module at
//! render time (`obj.extend UserDecorator`), so decorator bodies are
//! written as if the model's methods were their own:
//!
//! ```ruby
//! # app/decorators/user_decorator.rb
//! module UserDecorator
//!   def decorated_name = name + "!"   # `name` is User#name
//! end
//! ```
//!
//! Statically that `name` has no receiver, so every decorator method
//! becomes a `Ruby::NoMethod` false positive. RBS already expresses the
//! fact exactly — `module UserDecorator : User` — and crema already
//! resolves through module self-types. This pass only has to *attach*
//! that self-type, which it does by pushing the same
//! [`Member::ModuleSelf`] the inline `# @rbs module-self:` reader pushes
//! (`crate::inline_parser`), so both routes share one downstream path.
//!
//! The port target is `Orthoses::ActiveDecorator`, which resolves the
//! same facts at runtime. Each of its dynamic probes has a static
//! counterpart here, which is why the gem's semantics survive the
//! translation without executing Ruby (infusion design goal: no Ruby
//! execution):
//!
//! | orthoses (runtime)                    | this pass (static)               |
//! |---------------------------------------|----------------------------------|
//! | `ObjectSpace.each_object(Module)`     | walk the draft's decl trees      |
//! | `const_source_location` under the dir | `DeclOrigin::Path` under it      |
//! | `name.end_with?(decorator_suffix)`    | same, on the last segment        |
//! | `Class === mod` rejects classes       | only `ModuleDecl` nodes qualify  |
//! | `model_name.constantize` succeeds     | the name is declared in D or G   |
//!
//! The walk descends into nested declarations rather than reading
//! `draft.class_decls` keys alone: at this point in the pipeline only
//! *top-level* Ruby declarations are keyed there, and a nested
//! `module Admin; module UserDecorator` is still a `Member::Declaration`
//! inside its parent (`EnvironmentDraft::build` is what flattens them).
//! `zeitwerk_synthesis`'s `walk_ruby_*_declared` descends for the same
//! reason.
//!
//! The search root is fixed at `app/decorators`. The gem's generator
//! always writes there, and the fixed root is load-bearing rather than
//! incidental: it is what keeps every `*Decorator` module owned by gem
//! code out of scope. Widening it would silently relax unrelated
//! modules' types.
//!
//! Non-goals, both of which are dynamic in the gem and therefore out of
//! reach for static infusion: `ActiveDecorator::Helpers`'s
//! `method_missing` delegation to the view context, and `decorator_for`'s
//! `base_class` walk for STI subclasses.

use std::path::Path;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::ruby::annotations::ModuleSelfAnnotation;
use crate::ast::ruby::declarations::{
    ClassDecl as RubyClassDecl, Declaration as RubyDeclaration, ModuleDecl as RubyModuleDecl,
};
use crate::ast::ruby::members::{Member, ModuleSelfMember};
use crate::environment::DeclOrigin;
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::name::NameTable;
use crate::type_name::TypeName;

/// Directory, relative to the project root, that decorator modules are
/// read from. Fixed rather than configurable — see the module docs.
const DECORATORS_DIR: &str = "app/decorators";

/// Attach `self` types to decorator modules declared under
/// `<project_root>/app/decorators`.
///
/// A module qualifies when its name ends with `decorator_suffix`, the
/// name with that suffix removed is declared somewhere in the type
/// environment (the draft or an attached G snapshot), and it does not
/// already carry a self-type. Anything else is skipped silently: a
/// `FooDecorator` with no `Foo` is indistinguishable from a module that
/// merely happens to end in `Decorator`, and reporting on it would fire
/// on unrelated code.
///
/// Runs after every Ruby-source declaration has been inserted, since
/// the model may be declared in a file this pass has no other reason to
/// visit.
///
/// `decorator_suffix` is non-empty — `validate_active_decorator_table`
/// rejects an empty one at config load, since every name ends with it.
pub fn synthesize(draft: &mut EnvironmentDraft, project_root: &Path, decorator_suffix: &str) {
    // Canonicalized so the prefix test compares like with like: check
    // targets reach the draft canonicalized (`main.rs`'s
    // `canonicalize_or_exit`). A project with no `app/decorators` fails
    // here and exits — the common non-Rails case, at one `stat`.
    let Ok(decorators_root) = project_root.join(DECORATORS_DIR).canonicalize() else {
        return;
    };

    let scan = scan_draft(draft, &decorators_root);
    let targets = resolve_targets(draft, &scan, decorator_suffix);
    if targets.is_empty() {
        return;
    }
    attach_self_types(draft, &decorators_root, &targets);
}

/// What one read-only pass over the draft's declaration trees yields.
///
/// All three sets are collected before any mutation, so every "is the
/// model declared?" / "was a self-type written by hand?" test observes
/// the same draft: a synthesized self-type must never become input to
/// another module's decision.
#[derive(Default)]
struct Scan {
    /// Every class and module name declared in Ruby source, nested ones
    /// included. Seeded with `class_decls`'s keys, which additionally
    /// cover RBS-sourced declarations (flattened at insert time).
    declared: FxHashSet<TypeName>,
    /// Modules that already declare a self-type, in RBS (`module M : X`)
    /// or inline (`# @rbs module-self: X`). Explicit wins: this pass
    /// fills gaps only, mirroring `zeitwerk_synthesis`'s never-overwrite
    /// contract.
    self_typed: FxHashSet<TypeName>,
    /// Modules whose declaration sits under `app/decorators`. Classes
    /// never enter: RBS self-types are a module-only construct, and the
    /// gem rejects classes too (`unless Class === d` in `decorator_for`).
    candidates: FxHashSet<TypeName>,
}

fn scan_draft(draft: &EnvironmentDraft, decorators_root: &Path) -> Scan {
    let mut scan = Scan {
        declared: draft.class_decls.keys().copied().collect(),
        ..Scan::default()
    };
    let names = draft.names();
    for (name, entry) in &draft.class_decls {
        match entry {
            ClassOrModuleDraft::Class(e) => {
                for (origin, _, decl) in &e.context_decls {
                    if let ClassDeclarationDraft::Ruby(r) = decl {
                        let under = origin_is_under(names, origin, decorators_root);
                        scan_class_decl(r, under, &mut scan);
                    }
                }
            }
            ClassOrModuleDraft::Module(e) => {
                for (origin, _, decl) in &e.context_decls {
                    match decl {
                        ModuleDeclarationDraft::Signature(d) => {
                            if !d.self_types.is_empty() {
                                scan.self_typed.insert(*name);
                            }
                        }
                        ModuleDeclarationDraft::Ruby(r) => {
                            let under = origin_is_under(names, origin, decorators_root);
                            scan_module_decl(r, under, &mut scan);
                        }
                    }
                }
            }
        }
    }
    scan
}

fn scan_class_decl(decl: &RubyClassDecl, under: bool, scan: &mut Scan) {
    scan.declared.insert(decl.class_name);
    scan_members(&decl.members, under, scan);
}

fn scan_module_decl(decl: &RubyModuleDecl, under: bool, scan: &mut Scan) {
    scan.declared.insert(decl.module_name);
    if !decl.self_types().is_empty() {
        scan.self_typed.insert(decl.module_name);
    }
    if under {
        scan.candidates.insert(decl.module_name);
    }
    scan_members(&decl.members, under, scan);
}

/// `under` propagates unchanged into nested declarations: a nested decl
/// lives in its parent's file, which is the file the origin describes.
fn scan_members(members: &[Member], under: bool, scan: &mut Scan) {
    for member in members {
        let Member::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => scan_class_decl(c, under, scan),
            RubyDeclaration::Module(m) => scan_module_decl(m, under, scan),
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}

/// `decorator module -> model it decorates` for every candidate that
/// survives the suffix, explicit-self-type, and model-exists tests.
fn resolve_targets(
    draft: &EnvironmentDraft,
    scan: &Scan,
    decorator_suffix: &str,
) -> FxHashMap<TypeName, TypeName> {
    scan.candidates
        .iter()
        .filter(|name| !scan.self_typed.contains(name))
        .filter_map(|&name| {
            let model = strip_suffix(draft, name, decorator_suffix)?;
            (scan.declared.contains(&model) || g_declares(draft, model)).then_some((name, model))
        })
        .collect()
}

fn origin_is_under(names: &NameTable, origin: &DeclOrigin, root: &Path) -> bool {
    let DeclOrigin::Path(file) = origin else {
        return false;
    };
    Path::new(&names.resolve(*file)).starts_with(root)
}

/// `::Admin::UserDecorator` → `::Admin::User`. The suffix is removed
/// from the last segment only, so the namespace is preserved and a
/// module named exactly the suffix (`::Decorator`) yields nothing.
fn strip_suffix(draft: &EnvironmentDraft, name: TypeName, suffix: &str) -> Option<TypeName> {
    let names = draft.names();
    let last = names.last_segment(name)?;
    let last = names.resolve(last);
    let stem = last.strip_suffix(suffix)?;
    if stem.is_empty() {
        return None;
    }
    let parent = names.type_name_parent(name)?;
    Some(names.append_type_name(parent, names.intern_symbol(stem)))
}

/// Whether the G snapshot declares `name` — the gem-layer half of the
/// static stand-in for orthoses's `constantize` succeeding. The draft
/// half lives in [`Scan::declared`].
fn g_declares(draft: &EnvironmentDraft, name: TypeName) -> bool {
    draft.g.as_ref().is_some_and(|g| g.class_contains(&name))
}

/// Push `Member::ModuleSelf` onto each targeted module declaration.
///
/// Only declarations under `app/decorators` are visited, so a decorator
/// module reopened elsewhere keeps that other file's decl untouched —
/// self-types aggregate per module, so one attachment suffices.
fn attach_self_types(
    draft: &mut EnvironmentDraft,
    decorators_root: &Path,
    targets: &FxHashMap<TypeName, TypeName>,
) {
    // Split borrow: `names` and `class_decls` are disjoint fields, which
    // is what lets the origin test run inside a `values_mut` walk.
    let names = &draft.names;
    for entry in draft.class_decls.values_mut() {
        match entry {
            ClassOrModuleDraft::Class(e) => {
                for (origin, _, decl) in e.context_decls.iter_mut() {
                    let ClassDeclarationDraft::Ruby(r) = decl else {
                        continue;
                    };
                    if !origin_is_under(names, origin, decorators_root) {
                        continue;
                    }
                    // The draft entry shares its Arc with the collector's
                    // top-level Vec until that Vec is dropped; synthesize
                    // runs after that, so make_mut must not copy here.
                    debug_assert_eq!(Arc::strong_count(r), 1, "decl shared before synthesize");
                    attach_in_members(&mut Arc::make_mut(r).members, targets);
                }
            }
            ClassOrModuleDraft::Module(e) => {
                for (origin, _, decl) in e.context_decls.iter_mut() {
                    let ModuleDeclarationDraft::Ruby(r) = decl else {
                        continue;
                    };
                    if !origin_is_under(names, origin, decorators_root) {
                        continue;
                    }
                    debug_assert_eq!(Arc::strong_count(r), 1, "decl shared before synthesize");
                    attach_in_module(Arc::make_mut(r), targets);
                }
            }
        }
    }
}

/// The byte range of the module's own name stands in for the
/// annotation's location: `RubyModuleDecl::self_types` drops it
/// (`location: None`) when lifting members into self-types, so no
/// diagnostic can point at it and no synthetic span escapes.
fn attach_in_module(decl: &mut RubyModuleDecl, targets: &FxHashMap<TypeName, TypeName>) {
    if let Some(&self_type_name) = targets.get(&decl.module_name) {
        let name_location = decl.name_location;
        decl.members.push(Member::ModuleSelf(ModuleSelfMember {
            annotation: ModuleSelfAnnotation {
                name: self_type_name,
                name_location,
                args: Vec::new(),
                location: name_location,
            },
        }));
    }
    attach_in_members(&mut decl.members, targets);
}

fn attach_in_members(members: &mut [Member], targets: &FxHashMap<TypeName, TypeName>) {
    for member in members.iter_mut() {
        let Member::Declaration(d) = member else {
            continue;
        };
        match d {
            // Collect stage: the nested Arc is still unique here (sharing
            // only starts at draft.build()), so make_mut does not copy. A
            // silent copy would detach the parent's member from the env
            // entry, so fail loudly if that ordering is ever broken.
            RubyDeclaration::Class(c) => {
                debug_assert_eq!(Arc::strong_count(c), 1, "nested decl shared before collect");
                attach_in_members(&mut Arc::make_mut(c).members, targets)
            }
            RubyDeclaration::Module(m) => {
                debug_assert_eq!(Arc::strong_count(m), 1, "nested decl shared before collect");
                attach_in_module(Arc::make_mut(m), targets)
            }
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}
