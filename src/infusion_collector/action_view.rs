//! View-context self-types for the helper modules under `app/helpers`.
//!
//! A Rails helper module is included *into* the view class, so its own
//! ancestors are empty, yet its method bodies run with `self` = the view
//! and call view-context methods with no receiver:
//!
//! ```ruby
//! # app/helpers/theme_helper.rb
//! module ThemeHelper
//!   def title = content_tag(:h1, t("title"))   # ActionView::Helpers
//!   def home = root_url                        # route helper
//! end
//! ```
//!
//! Statically every such call is a `Ruby::NoMethod` false positive. RBS
//! already has the construct for "inside this module `self` is a T" — the
//! module self-type — so this pass declares one per helper module:
//!
//! ```rbs
//! module ThemeHelper : ::ActionDispatch::Routing::RouteSet::UrlHelpersModule,
//!                      ::ActionDispatch::Routing::RouteSet::PathHelpersModule,
//!                      ::ActionView::Base            # when declared
//! end
//! ```
//!
//! Which modules are helpers follows Rails' own rule rather than a guess:
//! `AbstractController::Helpers::ClassMethods#all_helpers_from_path`
//! globs `app/helpers/**/*_helper.rb` and camelizes each relative path
//! into the module it includes (`admin/roles_helper.rb` →
//! `Admin::RolesHelper`). The static counterpart is a Ruby module declared
//! in such a file whose last name segment ends in `Helper` — every
//! camelized `*_helper` path does. The suffix test is what leaves the
//! `Admin` namespace module of a nested helper alone.
//!
//! A self-type rather than include edges (`include ActionView::Helpers`):
//! an include is inherited by everything that includes the helper, and
//! helper modules are routinely included outside views (serializers,
//! services, controllers). Such a class's own `def fields` / `def request`
//! would then be checked against the same-named view method. A self-type
//! adds nothing to the helper's ancestors; it only changes what `self` is
//! inside the helper's own bodies.
//!
//! An RBS self-type is also a constraint on includers (`include M`
//! requires the includer to be a T), and that half does not hold here:
//! the same non-view includers would each be an `RBS::ModuleSelfTypeError`.
//! The declaration is inserted with a `DeclOrigin::Synthesized` origin,
//! which marks its self-types as assumptions the includer check skips
//! (`OneAncestors::assumed_self_types`). The assumption is unsound for a
//! helper method that calls a view method while included into a non-view
//! class; Rails raises `NoMethodError` there and crema stays silent.
//!
//! Out of reach, by construction: methods a controller exports with
//! `helper_method`, and calls from one helper module into another (the
//! view class includes every helper; the declared `ActionView::Base` does
//! not).
//!
//! Not read: `config.action_controller.include_all_helpers`, a custom
//! `helpers_path`, and engines' own `app/helpers`. The root is fixed for
//! the same reason `active_decorator`'s is — it is what keeps gem-owned
//! `*Helper` modules out of scope.

use std::path::Path;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::ast::declarations::{ModuleDeclaration as Module, ModuleSelf};
use crate::ast::ruby::declarations::{
    ClassDecl as RubyClassDecl, Declaration as RubyDeclaration, ModuleDecl as RubyModuleDecl,
};
use crate::ast::ruby::members::Member;
use crate::config::InfusionOptions;
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, Context, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::action_dispatch::{PATH_HELPERS_MODULE, URL_HELPERS_MODULE};
use crate::infusion_collector::active_record_synthesis::collect_declared_class_module_names;
use crate::name::NameTable;
use crate::type_name::TypeName;

/// Rails' default `helpers_path`, relative to the project root. Fixed
/// rather than configurable — see the module docs.
const HELPERS_DIR: &str = "app/helpers";
/// The file-name half of `all_helpers_from_path`'s glob.
const HELPER_FILE_SUFFIX: &str = "_helper.rb";
/// What every camelized `*_helper` path ends in.
const HELPER_MODULE_SUFFIX: &str = "Helper";
/// The class every view context is an instance of (a subclass of it,
/// built by `ActionView::Base.with_empty_template_cache`).
const VIEW_CONTEXT: &str = "::ActionView::Base";

/// Declare the view context as the self-type of every helper module under
/// `<project_root>/app/helpers`. No-op unless the rails preset is on.
///
/// Runs after `load_collected`: every Ruby-source declaration must be in
/// the draft, and so must the route helper modules that `emit_routes`
/// declares (unconditionally under the rails preset, which is why they
/// need no existence test while `ActionView::Base` does — a project that
/// loads no actionview signatures must not grow a self-type naming an
/// undeclared class).
pub fn synthesize(draft: &mut EnvironmentDraft, project_root: &Path, options: InfusionOptions) {
    if !options.rails_enabled() {
        return;
    }
    // Canonicalized so the prefix test compares like with like: check
    // targets reach the draft canonicalized. A project with no
    // `app/helpers` fails here and exits at the cost of one `stat`.
    let Ok(helpers_root) = project_root.join(HELPERS_DIR).canonicalize() else {
        return;
    };
    let helpers = helper_modules(draft, &helpers_root);
    if helpers.is_empty() {
        return;
    }

    let names = draft.names();
    // `view_context_class` includes `routes.url_helpers(supports_path)`,
    // i.e. both modules. Named directly so they arrive without actionview
    // signatures too. The view class goes last: later self-types win
    // lookup, as the class's own methods win over the modules it includes.
    let mut self_types = vec![
        names.parse_type_name(URL_HELPERS_MODULE),
        names.parse_type_name(PATH_HELPERS_MODULE),
    ];
    let view_context = names.parse_type_name(VIEW_CONTEXT);
    if collect_declared_class_module_names(draft).contains(&view_context) {
        self_types.push(view_context);
    }

    let context: Context = Arc::from([names.absolute_root()]);
    // Load-bearing: a `Synthesized` origin is what exempts these
    // self-types from the includer check (see the module docs).
    let origin = DeclOrigin::Synthesized(InfusionUnit::ActionView, None);
    for helper in helpers {
        let decl = Module {
            name: helper,
            type_params: Vec::new(),
            self_types: self_types
                .iter()
                .map(|&name| ModuleSelf {
                    name,
                    args: Vec::new(),
                    location: None,
                    source_file: None,
                })
                .collect(),
            members: Vec::new(),
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        };
        let _ = draft.insert_module_decl(origin, Arc::clone(&context), Arc::new(decl));
    }
}

/// Every Ruby module declared in a helper file whose name ends in
/// `Helper`, nested ones included, each once.
///
/// The walk descends into nested declarations rather than reading
/// `draft.class_decls` keys alone: before `EnvironmentDraft::build` only
/// top-level Ruby declarations are keyed there, and the
/// `module Admin; module RolesHelper` of a nested helper is still a member
/// of its parent.
fn helper_modules(draft: &EnvironmentDraft, helpers_root: &Path) -> Vec<TypeName> {
    let names = draft.names();
    let mut found = Found::default();
    for entry in draft.class_decls.values() {
        match entry {
            ClassOrModuleDraft::Class(e) => {
                for (origin, _, decl) in &e.context_decls {
                    if let ClassDeclarationDraft::Ruby(r) = decl
                        && is_helper_file(names, origin, helpers_root)
                    {
                        scan_class(names, r, &mut found);
                    }
                }
            }
            ClassOrModuleDraft::Module(e) => {
                for (origin, _, decl) in &e.context_decls {
                    if let ModuleDeclarationDraft::Ruby(r) = decl
                        && is_helper_file(names, origin, helpers_root)
                    {
                        scan_module(names, r, &mut found);
                    }
                }
            }
        }
    }
    found.modules
}

#[derive(Default)]
struct Found {
    modules: Vec<TypeName>,
    seen: FxHashSet<TypeName>,
}

/// `Dir["#{helpers_root}/**/*_helper.rb"]`, applied to a declaration's file.
fn is_helper_file(names: &NameTable, origin: &DeclOrigin, helpers_root: &Path) -> bool {
    let DeclOrigin::Path(file) = origin else {
        return false;
    };
    let file = names.resolve(*file);
    file.ends_with(HELPER_FILE_SUFFIX) && Path::new(file).starts_with(helpers_root)
}

/// A class is never a helper (`helper` takes modules only), but a helper
/// module may be nested in one.
fn scan_class(names: &NameTable, decl: &RubyClassDecl, found: &mut Found) {
    scan_members(names, &decl.members, found);
}

fn scan_module(names: &NameTable, decl: &RubyModuleDecl, found: &mut Found) {
    let is_helper = names
        .last_segment(decl.module_name)
        .is_some_and(|last| names.resolve(last).ends_with(HELPER_MODULE_SUFFIX));
    if is_helper && found.seen.insert(decl.module_name) {
        found.modules.push(decl.module_name);
    }
    scan_members(names, &decl.members, found);
}

fn scan_members(names: &NameTable, members: &[Member], found: &mut Found) {
    for member in members {
        let Member::Declaration(d) = member else {
            continue;
        };
        match d {
            RubyDeclaration::Class(c) => scan_class(names, c, found),
            RubyDeclaration::Module(m) => scan_module(names, m, found),
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}
