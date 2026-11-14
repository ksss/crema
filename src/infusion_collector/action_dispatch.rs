//! ActionDispatch routes provider: the named-route helper names
//! (`users_path` / `users_url`, ...) that `config/routes.rb` generates.
//!
//! Rails defines every `*_path` / `*_url` helper at boot by evaluating the
//! routing DSL (`ActionDispatch::Routing::Mapper`), so none of them has a
//! static declaration and every call is a `Ruby::NoMethod` false positive.
//! This module evaluates the DSL statically: it walks the
//! `Rails.application.routes.draw do ... end` block (and the files
//! `draw :name` pulls in from `config/routes/`) with a scope stack that
//! mirrors `Mapper::Scope`, and applies the naming rules of
//! `Mapper::Resources#name_for_action` to produce the set of helper base
//! names. The member / signature synthesis is the second half
//! ([`emit_routes`]); this half only decides *which names exist*.
//!
//! Port boundaries (actionpack 8.1 `action_dispatch/routing/mapper.rb`):
//!
//! - `name_for_action` (:2082), `prefix_name_for_action` (:2070),
//!   `Mapper.normalize_name` (:407) and `Scope#action_name` (:2476) are
//!   ported as written.
//! - `Resource` / `SingletonResource` (:1306 / :1418): `name = as || entities`,
//!   `member_name` / `collection_name`, `default_actions` (never
//!   `api_only` — `config/application.rb` is not read, so the `new` / `edit`
//!   helpers are over-approximated for API-only apps).
//! - `resources` / `resource` (:1663 / :1490) with `apply_common_behavior_for`
//!   (:1928): multiple names, `shallow:`, nesting under a resource scope and
//!   the residual `scope(...)` wrap; the canonical `collection` / `new` /
//!   `member` route emission (`set_member_mappings_for_resource` :2108).
//! - `scope` (:989) / `namespace` (:1097): only `as:` / `shallow_prefix:` /
//!   `only:` / `except:` influence names; `path:` / `module:` /
//!   `controller:` do not and are ignored.
//! - `match` / `get` / ... (:1837, :2131 `map_match`, :2195 `add_route`):
//!   a path with a character outside `[\w\-/]` (a dynamic `:segment`, a
//!   `.`) yields no automatic name; `as:` names it anyway; `as: nil` /
//!   `as: false` suppresses the name.
//! - `root` (:1903), `concern` / `concerns` (:2297 / :2313), `draw` (:1816,
//!   `config/routes/<name>.rb` — railties `engine.rb:597`), `mount` with an
//!   explicit `as:` (:605; the implicit `railtie_name` lives in the gem and
//!   is skipped), `direct` (:2374).
//!
//! Deterministic-input boundary (specs/infusion.md Design Goal 2): nothing
//! is executed. Every branch of a conditional is taken (the name set is a
//! safe over-approximation for the NoMethod suppression this exists for;
//! the same source still yields the same set), any argument that is not a
//! literal makes that one call contribute nothing and surfaces as
//! `InfusionProviderSkipped`, and a gem DSL that mints names from its own
//! rules (`devise_for`, `use_doorkeeper`) is reported the same way rather
//! than guessed at.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ruby_prism::{CallNode, DefNode, LambdaNode, Node, Visit};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, Member as SigMember, ModuleDeclaration as Module,
    ModuleMember,
};
use crate::ast::members::{
    ExtendMember as Extend, MethodDefinitionMember as MethodDefinition, MethodDefinitionOverload,
    MethodKind, Visibility,
};
use crate::ast::method_type::MethodType;
use crate::ast::types::{Function, FunctionParam, FunctionType};
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::draft::{Context, EnvironmentDraft};
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    class_instance, collect_declared_class_module_names, include_member, untyped_type,
};
use crate::infusion_collector::inflector::Inflector;
use crate::infusion_collector::pipeline::InfusionOwnerKind;
use crate::type_name::TypeName;

pub(crate) const PROVIDER: &str = "actiondispatch";

/// The two helper modules, mirroring `NamedRouteCollection#url_helpers_module`
/// / `#path_helpers_module` (`route_set.rb:548`): Rails includes both into
/// controllers and views but only the url one into mailers
/// (`include_path_helpers = false`).
pub(crate) const URL_HELPERS_MODULE: &str = "::ActionDispatch::Routing::RouteSet::UrlHelpersModule";
pub(crate) const PATH_HELPERS_MODULE: &str =
    "::ActionDispatch::Routing::RouteSet::PathHelpersModule";
/// gem_rbs_collection's name for the anonymous module `RouteSet#url_helpers`
/// returns; given the helpers too (include + extend, as Rails does at
/// `route_set.rb:596`) when the project's signatures declare it.
const ANONYMOUS_MODULE: &str = "::ActionDispatch::Routing::RouteSet::RouteSet_AnonymousModule";
/// Where Rails includes the helpers at boot, and whether the path helpers
/// come along: controllers (`abstract_controller/railties/routes_helpers.rb`),
/// views (`action_view/rendering.rb:66`), mailers (`action_mailer/railtie.rb:39`,
/// url only) and integration tests (`action_dispatch/testing/integration.rb:368`).
const INCLUDERS: &[(&str, bool)] = &[
    ("::ActionController::Base", true),
    ("::ActionView::Base", true),
    ("::ActionMailer::Base", false),
    ("::ActionDispatch::IntegrationTest", true),
];
const RETURN_TYPE: &str = "::String";

/// A class-body `include` / `extend Rails.application.routes.url_helpers`
/// site, collected per file (`pipeline::Collector`) and resolved to the
/// synthesized modules at emission.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct UrlHelpersSite {
    pub(crate) target: TypeName,
    pub(crate) target_kind: InfusionOwnerKind,
    pub(crate) extend: bool,
    /// `url_helpers` / `url_helpers(true)` → both modules; `url_helpers(false)` → url only.
    pub(crate) supports_path: bool,
}

/// `Rails.application.routes.url_helpers` (optionally `(true)` /
/// `(false)`) as the sole argument of a mixin call: `Some(supports_path)`.
pub(crate) fn url_helpers_mixin_argument(node: &CallNode<'_>) -> Option<bool> {
    let arguments = node.arguments()?;
    let args: Vec<_> = arguments.arguments().iter().collect();
    if args.len() != 1 {
        return None;
    }
    let call = args[0].as_call_node()?;
    if call.name().as_slice() != b"url_helpers" {
        return None;
    }
    let supports_path = match call.arguments() {
        None => true,
        Some(arguments) => {
            let inner: Vec<_> = arguments.arguments().iter().collect();
            if inner.len() != 1 {
                return None;
            }
            if inner[0].as_true_node().is_some() {
                true
            } else if inner[0].as_false_node().is_some() {
                false
            } else {
                return None;
            }
        }
    };
    let routes = call.receiver()?;
    let routes = routes.as_call_node()?;
    if routes.name().as_slice() != b"routes" || routes.arguments().is_some() {
        return None;
    }
    let application = routes.receiver()?;
    let application = application.as_call_node()?;
    if application.name().as_slice() != b"application" || application.arguments().is_some() {
        return None;
    }
    let rails = application.receiver()?;
    let is_rails = if let Some(constant) = rails.as_constant_read_node() {
        constant.name().as_slice() == b"Rails"
    } else if let Some(path) = rails.as_constant_path_node() {
        path.parent().is_none() && path.name().is_some_and(|name| name.as_slice() == b"Rails")
    } else {
        false
    };
    is_rails.then_some(supports_path)
}

/// Declare the helper modules and attach them. Runs under the rails
/// preset whether or not a routes file exists: the modules are what the
/// `include Rails.application.routes.url_helpers` sites resolve to, so
/// they must exist even when empty. Attachments to gem classes
/// (`ActionController::Base`, ...) happen only when the environment
/// already declares the class — a project without actionpack signatures
/// does not grow a ghost `ActionController::Base`.
pub(crate) fn emit_routes(
    routes: Option<&PreparsedRoutes>,
    sites: &[UrlHelpersSite],
    draft: &mut EnvironmentDraft,
) {
    // Everything that needs the name table is built first (the table
    // borrows the draft); the inserts run after.
    let pending = {
        let names = draft.names();
        // Every class / module the environment knows: the A layer with
        // its nested RBS declarations (only top-level ones are keyed in
        // the draft before `build`) and the G snapshot.
        let declared = collect_declared_class_module_names(draft);
        let url_module = names.parse_type_name(URL_HELPERS_MODULE);
        let path_module = names.parse_type_name(PATH_HELPERS_MODULE);
        let string = names.parse_type_name(RETURN_TYPE);
        let empty_names = Vec::new();
        let route_names = routes.map_or(&empty_names, |routes| &routes.names);
        let mut pending: Vec<Pending> = Vec::new();

        // Namespace: `module ActionDispatch; module Routing; class RouteSet`.
        // Open-class merges when the gem signatures declare them, fresh
        // (empty) declarations when they do not.
        for namespace in ["::ActionDispatch", "::ActionDispatch::Routing"] {
            pending.push(Pending::Module(Arc::new(empty_module(
                names.parse_type_name(namespace),
            ))));
        }
        pending.push(Pending::Class(Arc::new(empty_class(
            names.parse_type_name("::ActionDispatch::Routing::RouteSet"),
        ))));

        // The two modules: `name_url` on one, `name_path` on the other,
        // each `(*untyped, **untyped) -> String` — Rails'
        // `handle_positional_args` fills missing segments from the current
        // request and routes extra positionals into `:format`, so arity is
        // not statically checkable.
        for (module, suffix) in [(url_module, "_url"), (path_module, "_path")] {
            // `url_for` / `polymorphic_path` stay with the gem signatures'
            // own `UrlFor` / `PolymorphicRoutes` (RouteSet_AnonymousModule
            // includes them). Pulling them in here would also pull their
            // `initialize` into every includer, and a class's own Ruby
            // `initialize` is then checked against it.
            let mut members: Vec<ModuleMember> = Vec::new();
            for name in route_names {
                let symbol = names.intern_symbol(&format!("{name}{suffix}"));
                members.push(ModuleMember::Member(helper_method(symbol, string)));
            }
            pending.push(Pending::Module(Arc::new(Module {
                name: module,
                type_params: Vec::new(),
                self_types: Vec::new(),
                members,
                annotations: Vec::new(),
                location: None,
                source_file: None,
                comment: None,
            })));
        }

        // Where Rails includes them at boot.
        for (includer, with_path) in INCLUDERS {
            let includer = names.parse_type_name(includer);
            if !declared.contains(&includer) {
                continue;
            }
            let mut members = vec![include_member(url_module, Vec::new())];
            if *with_path {
                members.push(include_member(path_module, Vec::new()));
            }
            pending.push(Pending::Class(Arc::new(Class {
                name: includer,
                type_params: Vec::new(),
                super_class: None,
                members,
                annotations: Vec::new(),
                location: None,
                source_file: None,
                comment: None,
            })));
        }

        // `Rails.application.routes.url_helpers.foo_path`: the anonymous
        // module both includes and extends the helpers (`route_set.rb:596-599`).
        let anonymous = names.parse_type_name(ANONYMOUS_MODULE);
        if declared.contains(&anonymous) {
            pending.push(Pending::Module(Arc::new(Module {
                name: anonymous,
                type_params: Vec::new(),
                self_types: Vec::new(),
                members: vec![
                    ModuleMember::Member(include_sig_member(url_module)),
                    ModuleMember::Member(include_sig_member(path_module)),
                    ModuleMember::Member(extend_sig_member(url_module)),
                    ModuleMember::Member(extend_sig_member(path_module)),
                ],
                annotations: Vec::new(),
                location: None,
                source_file: None,
                comment: None,
            })));
        }

        // Explicit `include` / `extend Rails.application.routes.url_helpers`.
        for site in sites {
            let mixin = |module| {
                if site.extend {
                    extend_sig_member(module)
                } else {
                    include_sig_member(module)
                }
            };
            let mut members: Vec<SigMember> = vec![mixin(url_module)];
            if site.supports_path {
                members.push(mixin(path_module));
            }
            pending.push(match site.target_kind {
                InfusionOwnerKind::Class => Pending::Class(Arc::new(Class {
                    name: site.target,
                    type_params: Vec::new(),
                    super_class: None,
                    members: members.into_iter().map(ClassMember::Member).collect(),
                    annotations: Vec::new(),
                    location: None,
                    source_file: None,
                    comment: None,
                })),
                InfusionOwnerKind::Module => Pending::Module(Arc::new(Module {
                    name: site.target,
                    type_params: Vec::new(),
                    self_types: Vec::new(),
                    members: members.into_iter().map(ModuleMember::Member).collect(),
                    annotations: Vec::new(),
                    location: None,
                    source_file: None,
                    comment: None,
                })),
            });
        }
        pending
    };

    let context: Context = Arc::from([draft.names().absolute_root()]);
    let origin = DeclOrigin::Synthesized(InfusionUnit::ActionDispatch, None);
    for decl in pending {
        match decl {
            Pending::Class(decl) => {
                let _ = draft.insert_class_decl(origin, Arc::clone(&context), decl);
            }
            Pending::Module(decl) => {
                let _ = draft.insert_module_decl(origin, Arc::clone(&context), decl);
            }
        }
    }
}

enum Pending {
    Class(Arc<Class>),
    Module(Arc<Module>),
}

fn empty_module(name: TypeName) -> Module {
    Module {
        name,
        type_params: Vec::new(),
        self_types: Vec::new(),
        members: Vec::new(),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    }
}

fn empty_class(name: TypeName) -> Class {
    Class {
        name,
        type_params: Vec::new(),
        super_class: None,
        members: Vec::new(),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    }
}

fn include_sig_member(name: TypeName) -> SigMember {
    match include_member(name, Vec::new()) {
        ClassMember::Member(member) => member,
        ClassMember::Declaration(_) => unreachable!("include_member builds a Member"),
    }
}

fn extend_sig_member(name: TypeName) -> SigMember {
    SigMember::Extend(Extend {
        name,
        args: Vec::new(),
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    })
}

fn untyped_param() -> Box<FunctionParam> {
    Box::new(FunctionParam {
        ty: Box::new(untyped_type()),
        name: None,
        location: None,
    })
}

/// `def <name>: (*untyped, **untyped) -> String`.
fn helper_method(name: crate::name::Symbol, string: TypeName) -> SigMember {
    SigMember::MethodDefinition(MethodDefinition {
        name,
        kind: MethodKind::Instance,
        overloads: vec![MethodDefinitionOverload {
            method_type: MethodType {
                type_params: Vec::new(),
                function: Function::Typed(FunctionType {
                    required_positionals: Vec::new(),
                    optional_positionals: Vec::new(),
                    rest_positionals: Some(untyped_param()),
                    trailing_positionals: Vec::new(),
                    required_keywords: Vec::new(),
                    optional_keywords: Vec::new(),
                    rest_keywords: Some(untyped_param()),
                    return_type: Box::new(class_instance(string)),
                }),
                block: None,
                location: None,
            },
            annotations: Vec::new(),
        }],
        annotations: Vec::new(),
        overloading: false,
        visibility: Some(Visibility::Public),
        location: None,
        source_file: None,
        comment: None,
    })
}

/// The helper base names (`users`, `edit_user`, ...) `config/routes.rb`
/// declares, in first-definition order. Empty for a project without
/// `config/routes.rb`: the helper modules are declared either way (the
/// explicit mixin sites resolve to them), just with no names.
pub struct PreparsedRoutes {
    pub(crate) names: Vec<String>,
}

impl PreparsedRoutes {
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

const CANONICAL_ACTIONS: &[&str] = &["index", "create", "new", "show", "update", "destroy"];

/// `Resource.default_actions(false)` / `SingletonResource.default_actions(false)`.
const RESOURCES_DEFAULT_ACTIONS: &[&str] = &[
    "index", "create", "new", "show", "update", "destroy", "edit",
];
const RESOURCE_DEFAULT_ACTIONS: &[&str] = &["show", "create", "update", "destroy", "new", "edit"];

/// Gem DSLs that define named routes by rules that live in the gem.
/// Reported, never guessed.
const GEM_ROUTE_DSLS: &[&str] = &["devise_for", "use_doorkeeper"];

/// Evaluate `<project_root>/config/routes.rb` (plus every `draw`n file
/// under `<project_root>/config/routes/`). A missing routes file is a
/// silent empty result; a parse error in any file skips that file with a
/// diagnostic and leaves the others' names in place.
pub fn prepare_routes(
    project_root: &Path,
    inflector: &Inflector,
) -> (PreparsedRoutes, Vec<Diagnostic>) {
    let routes_path = project_root.join("config").join("routes.rb");
    let Ok(root_source) = std::fs::read(&routes_path) else {
        return (PreparsedRoutes { names: Vec::new() }, Vec::new());
    };
    let draw_dir = project_root.join("config").join("routes");
    let mut diagnostics = Vec::new();
    let files = RouteFiles::load(
        project_root,
        routes_path,
        root_source,
        &draw_dir,
        &mut diagnostics,
    );
    let names = evaluate(&files, inflector, &mut diagnostics);
    (PreparsedRoutes { names }, diagnostics)
}

/// The routes file and every file a `draw :name` reaches, parsed up front
/// so that a `concern` defined in one file can be expanded inside another
/// (the block nodes borrow their file's parse result).
struct RouteFiles {
    project_root: PathBuf,
    paths: Vec<PathBuf>,
    sources: Vec<Vec<u8>>,
    /// `draw :name` → index into `paths` / `sources`.
    draws: FxHashMap<String, usize>,
}

impl RouteFiles {
    fn load(
        project_root: &Path,
        routes_path: PathBuf,
        root_source: Vec<u8>,
        draw_dir: &Path,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> RouteFiles {
        let mut files = RouteFiles {
            project_root: project_root.to_path_buf(),
            paths: vec![routes_path],
            sources: vec![root_source],
            draws: FxHashMap::default(),
        };
        let mut pending = 0;
        while pending < files.sources.len() {
            let draw_names = {
                let parse_result = ruby_prism::parse(&files.sources[pending]);
                let mut scan = DrawScan::default();
                scan.visit(&parse_result.node());
                scan.names
            };
            pending += 1;
            for name in draw_names {
                if files.draws.contains_key(&name) || !draw_name_is_safe(&name) {
                    continue;
                }
                let path = draw_dir.join(format!("{name}.rb"));
                match std::fs::read(&path) {
                    Ok(source) => {
                        files.paths.push(path);
                        files.sources.push(source);
                        files.draws.insert(name, files.sources.len() - 1);
                    }
                    Err(_) => {
                        // Reported at the `draw` call site during evaluation,
                        // where the location is known.
                    }
                }
            }
        }
        let _ = diagnostics;
        files
    }

    fn subject(&self, index: usize) -> String {
        self.paths[index]
            .strip_prefix(&self.project_root)
            .unwrap_or(&self.paths[index])
            .display()
            .to_string()
    }
}

/// `draw :name` resolves to `config/routes/<name>.rb`; Rails allows a
/// subdirectory (`draw "admin/x"`) but nothing that leaves the directory.
fn draw_name_is_safe(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name
            .split('/')
            .any(|segment| segment.is_empty() || segment == "..")
}

/// Finds every literal `draw :name` so the files can be read before
/// evaluation starts.
#[derive(Default)]
struct DrawScan {
    names: Vec<String>,
}

impl<'pr> Visit<'pr> for DrawScan {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if node.receiver().is_none()
            && node.name().as_slice() == b"draw"
            && let Some(name) = positional_literal_plain(node, 0)
        {
            self.names.push(name);
        }
        ruby_prism::visit_call_node(self, node);
    }
}

fn evaluate(
    files: &RouteFiles,
    inflector: &Inflector,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<String> {
    let parse_results: Vec<_> = files
        .sources
        .iter()
        .map(|source| ruby_prism::parse(source))
        .collect();
    let mut parsed_ok = vec![true; parse_results.len()];
    for (index, parse_result) in parse_results.iter().enumerate() {
        if let Some(first_err) = parse_result.errors().next() {
            parsed_ok[index] = false;
            let location = first_err.location();
            diagnostics.push(Diagnostic {
                scope: None,
                kind: DiagnosticKind::InfusionProviderSkipped {
                    provider: PROVIDER.to_string(),
                    subject: files.subject(index),
                    reason: format!(
                        "routes parse error: {}",
                        String::from_utf8_lossy(first_err.message())
                    ),
                },
                location: Diagnostic::location_for_byte_range(
                    files.paths[index].clone(),
                    &files.sources[index],
                    location.start() as usize,
                    location.end() as usize,
                ),
            });
        }
    }
    if !parsed_ok[0] {
        return Vec::new();
    }
    // `concern :name do ... end` bodies from every file, so a concern
    // defined in `routes.rb` expands inside a `draw`n file. Rails requires
    // the definition to precede the use; this accepts either order.
    let mut concerns = FxHashMap::default();
    for (index, parse_result) in parse_results.iter().enumerate() {
        if !parsed_ok[index] {
            continue;
        }
        let mut scan = ConcernScan {
            file_index: index,
            concerns: &mut concerns,
        };
        scan.visit(&parse_result.node());
    }
    let mut evaluator = Evaluator {
        files,
        parse_results: &parse_results,
        parsed_ok: &parsed_ok,
        inflector,
        file_index: 0,
        scopes: vec![Frame::default()],
        concerns: &concerns,
        names: Vec::new(),
        seen: FxHashSet::default(),
        diagnostics,
        draw_depth: 0,
        visiting_concerns: FxHashSet::default(),
        visiting_draws: FxHashSet::default(),
        bindings: Vec::new(),
        with_options: Vec::new(),
    };
    // Only the `draw do ... end` block body is the DSL; whatever else the
    // file does at top level (constants, helper lambdas) is left alone.
    let root = parse_results[0].node();
    let mut finder = DrawBlockFinder { body: None };
    finder.visit(&root);
    if let Some(body) = finder.body {
        evaluator.visit(&body);
    }
    evaluator.names
}

/// Collects `concern :name do ... end` bodies.
struct ConcernScan<'a, 'pr> {
    file_index: usize,
    concerns: &'a mut FxHashMap<String, (usize, Node<'pr>)>,
}

impl<'a, 'pr> Visit<'pr> for ConcernScan<'a, 'pr> {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if node.receiver().is_none()
            && node.name().as_slice() == b"concern"
            && let Some(name) = positional_literal_plain(node, 0)
            && let Some(block) = node.block().and_then(|block| block.as_block_node())
            && let Some(body) = block.body()
        {
            self.concerns.insert(name, (self.file_index, body));
            return;
        }
        ruby_prism::visit_call_node(self, node);
    }
}

/// `Rails.application.routes.draw do ... end`: the first `draw` call with
/// a block (any receiver — `Rails.application.routes` is not resolved).
struct DrawBlockFinder<'pr> {
    body: Option<Node<'pr>>,
}

impl<'pr> Visit<'pr> for DrawBlockFinder<'pr> {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if self.body.is_some() {
            return;
        }
        if node.name().as_slice() == b"draw"
            && let Some(block) = node.block().and_then(|block| block.as_block_node())
            && let Some(body) = block.body()
        {
            self.body = Some(body);
            return;
        }
        ruby_prism::visit_call_node(self, node);
    }
}

/// `Mapper::Scope#scope_level`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    Top,
    Resources,
    Resource,
    Nested,
    Collection,
    Member,
    New,
    Root,
}

impl Level {
    /// `Scope#resource_scope?` (`RESOURCE_SCOPES`).
    fn is_resource_scope(self) -> bool {
        matches!(self, Level::Resources | Level::Resource)
    }

    /// `Scope#resource_method_scope?` (`RESOURCE_METHOD_SCOPES`).
    fn is_resource_method_scope(self) -> bool {
        matches!(self, Level::Collection | Level::Member | Level::New)
    }
}

/// `Resource` / `SingletonResource`: the parts that decide names.
#[derive(Clone, Debug)]
struct ResourceInfo {
    /// `Resource#name` — `as || entities`.
    singular: String,
    plural: String,
    singleton: bool,
    /// `Resource#shallow?` (`@scope[:shallow]` at construction).
    shallow: bool,
    actions: Vec<String>,
}

impl ResourceInfo {
    fn member_name(&self) -> &str {
        &self.singular
    }

    /// `Resource#collection_name`: `singular == plural ? "#{plural}_index" : plural`;
    /// `SingletonResource` aliases it to `singular`.
    fn collection_name(&self) -> String {
        if self.singleton {
            self.singular.clone()
        } else if self.singular == self.plural {
            format!("{}_index", self.plural)
        } else {
            self.plural.clone()
        }
    }

    fn has_action(&self, action: &str) -> bool {
        self.actions.iter().any(|a| a == action)
    }
}

/// One `Mapper::Scope` node. `Scope#new(hash)` merges the parent frame
/// with the new keys, so a child frame starts as a clone of its parent.
#[derive(Clone, Default, Debug)]
struct Frame {
    level: Option<Level>,
    /// `@scope[:as]`.
    as_prefix: Option<String>,
    /// `@scope[:shallow_prefix]`.
    shallow_prefix: Option<String>,
    /// `@scope[:shallow]`.
    shallow: bool,
    /// `@scope[:scope_level_resource]`.
    resource: Option<ResourceInfo>,
    /// `@scope[:action_options]` — `scope only: / except:`.
    action_options: Option<ActionOptions>,
}

impl Frame {
    fn level(&self) -> Level {
        self.level.unwrap_or(Level::Top)
    }
}

#[derive(Clone, Debug, Default)]
struct ActionOptions {
    only: Option<Vec<String>>,
    except: Option<Vec<String>>,
}

/// `as:` as written on a route.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AsOption {
    /// Not given (`DEFAULT`): auto-name from the action / path.
    Default,
    /// `as: :name` / `as: "name"`.
    Explicit(String),
    /// `as: nil` / `as: false`: no helper.
    None,
}

struct Evaluator<'pr> {
    files: &'pr RouteFiles,
    parse_results: &'pr [ruby_prism::ParseResult<'pr>],
    parsed_ok: &'pr [bool],
    inflector: &'pr Inflector,
    file_index: usize,
    scopes: Vec<Frame>,
    /// `concern :name do ... end` bodies, by name; expanded at `concerns`.
    concerns: &'pr FxHashMap<String, (usize, Node<'pr>)>,
    names: Vec<String>,
    seen: FxHashSet<String>,
    diagnostics: &'pr mut Vec<Diagnostic>,
    /// Guards `draw` / `concerns` recursion: depth, plus the concerns and
    /// draw files currently being expanded (a self-recursive concern would
    /// otherwise expand exponentially before the depth cap bites).
    draw_depth: usize,
    visiting_concerns: FxHashSet<String>,
    visiting_draws: FxHashSet<String>,
    /// Block parameters bound by `%w(...).each { |path| ... }`, innermost last.
    bindings: Vec<(String, String)>,
    /// `with_options(...) do ... end` argument lists, innermost last; a
    /// keyword absent from a call is looked up here.
    with_options: Vec<ruby_prism::ArgumentsNode<'pr>>,
}

const MAX_DRAW_DEPTH: usize = 32;

/// Options read off a `resources` / `resource` call.
#[derive(Clone, Debug, Default)]
struct ResourceOptions {
    as_name: Option<String>,
    only: Option<Vec<String>>,
    except: Option<Vec<String>>,
    /// `shallow: true` / `shallow: false` / absent.
    shallow: Option<bool>,
    shallow_prefix: Option<String>,
    concerns: Vec<String>,
}

impl<'pr> Evaluator<'pr> {
    fn frame(&self) -> &Frame {
        self.scopes.last().expect("scope stack is never empty")
    }

    /// `with_scope_level(kind)` / `@scope.new(...)`: push a child frame
    /// derived from the current one.
    fn push(&mut self, edit: impl FnOnce(&mut Frame)) {
        let mut frame = self.frame().clone();
        edit(&mut frame);
        self.scopes.push(frame);
    }

    fn pop(&mut self) {
        self.scopes.pop();
        debug_assert!(!self.scopes.is_empty());
    }

    fn with_level(&mut self, level: Level, body: impl FnOnce(&mut Self)) {
        self.push(|f| f.level = Some(level));
        body(self);
        self.pop();
    }

    fn parent_resource(&self) -> Option<&ResourceInfo> {
        self.frame().resource.as_ref()
    }

    /// `Resources#shallow?`: `!parent_resource.singleton? && @scope[:shallow]`.
    fn is_shallow(&self) -> bool {
        match self.parent_resource() {
            Some(resource) => !resource.singleton && self.frame().shallow,
            None => false,
        }
    }

    /// `shallow_nesting_depth >= 1`: some frame on the chain carries a
    /// shallow resource (frames inherit, so counting nodes is only an
    /// existence test here).
    fn has_shallow_ancestor(&self) -> bool {
        self.scopes
            .iter()
            .any(|frame| frame.resource.as_ref().is_some_and(|r| r.shallow))
    }

    fn skip(&mut self, node: &CallNode<'pr>, reason: String) {
        let subject = format!(
            "{}: {}",
            self.files.subject(self.file_index),
            String::from_utf8_lossy(node.name().as_slice())
        );
        let location = node.location();
        self.diagnostics.push(Diagnostic {
            scope: None,
            kind: DiagnosticKind::InfusionProviderSkipped {
                provider: PROVIDER.to_string(),
                subject,
                reason,
            },
            location: Diagnostic::location_for_byte_range(
                self.files.paths[self.file_index].clone(),
                &self.files.sources[self.file_index],
                location.start() as usize,
                location.end() as usize,
            ),
        });
    }

    fn add_name(&mut self, name: String) {
        if self.seen.insert(name.clone()) {
            self.names.push(name);
        }
    }

    fn visit_block_body(&mut self, node: &CallNode<'pr>) {
        if let Some(block) = node.block().and_then(|block| block.as_block_node())
            && let Some(body) = block.body()
        {
            self.visit(&body);
        }
    }

    // ----- naming (Resources#name_for_action and friends) -----

    /// `canonical_action?(action)`.
    fn is_canonical_action(&self, action: Option<&str>) -> bool {
        self.frame().level().is_resource_method_scope()
            && action.is_some_and(|action| CANONICAL_ACTIONS.contains(&action))
    }

    /// `prefix_name_for_action(as, action)`.
    fn prefix_name_for_action(&self, as_option: &AsOption, action: Option<&str>) -> Option<String> {
        let prefix = match as_option {
            AsOption::Explicit(name) => Some(name.as_str()),
            AsOption::Default if !self.is_canonical_action(action) => action,
            _ => None,
        }?;
        if prefix == "/" || prefix.is_empty() {
            return None;
        }
        Some(normalize_name(&prefix.replace('-', "_")))
    }

    /// `name_for_action(as, action)`. Returns the helper base name, or
    /// `None` when the route gets no helper.
    fn name_for_action(&self, as_option: &AsOption, action: Option<&str>) -> Option<String> {
        let prefix = self.prefix_name_for_action(as_option, action);
        let name_prefix = self.frame().as_prefix.clone();
        let (collection_name, member_name) = match self.parent_resource() {
            Some(resource) => {
                if *as_option == AsOption::Default && action.is_none() {
                    return None;
                }
                (
                    Some(resource.collection_name()),
                    Some(resource.member_name().to_string()),
                )
            }
            None => (None, None),
        };
        let parts: Vec<Option<String>> = match self.frame().level() {
            Level::Nested => vec![name_prefix, prefix],
            Level::Collection => vec![prefix, name_prefix, collection_name],
            Level::New => vec![prefix, Some("new".to_string()), name_prefix, member_name],
            Level::Member => vec![prefix, name_prefix, member_name],
            Level::Root => vec![name_prefix, collection_name, prefix],
            _ => vec![name_prefix, member_name, prefix],
        };
        let candidate: Vec<&str> = parts
            .iter()
            .filter_map(|part| part.as_deref())
            .filter(|part| !part.is_empty())
            .collect();
        if candidate.is_empty() {
            return None;
        }
        let candidate = candidate.join("_");
        if *as_option == AsOption::Default {
            let first = candidate.as_bytes()[0];
            if !(first == b'_' || first.is_ascii_alphabetic()) {
                return None;
            }
            if self.seen.contains(&candidate) {
                return None;
            }
        }
        Some(candidate)
    }

    /// `add_route(action, ..., as, ...)`: the name half. `action` is the
    /// route's path string or action symbol as written.
    fn add_route(&mut self, action: &str, as_option: &AsOption) {
        let action = action
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'/')
            .then_some(action);
        if *as_option == AsOption::None {
            return;
        }
        if let Some(name) = self.name_for_action(as_option, action) {
            self.add_name(name);
        }
    }

    // ----- scope-level helpers -----

    /// `member { }`: `shallow_scope` resets the `as` prefix.
    fn member(&mut self, body: impl FnOnce(&mut Self)) {
        let shallow = self.is_shallow();
        self.push(|f| f.level = Some(Level::Member));
        if shallow {
            self.push(|f| f.as_prefix = f.shallow_prefix.clone());
        }
        body(self);
        if shallow {
            self.pop();
        }
        self.pop();
    }

    fn collection(&mut self, body: impl FnOnce(&mut Self)) {
        self.with_level(Level::Collection, body);
    }

    fn new_scope(&mut self, body: impl FnOnce(&mut Self)) {
        self.with_level(Level::New, body);
    }

    /// `nested { }`: shallow resets the prefix, then
    /// `scope(as: parent_resource.member_name)`.
    fn nested(&mut self, body: impl FnOnce(&mut Self)) {
        let shallow = self.is_shallow() && self.has_shallow_ancestor();
        let member_name = self
            .parent_resource()
            .map(|r| r.member_name().to_string())
            .unwrap_or_default();
        self.push(|f| f.level = Some(Level::Nested));
        if shallow {
            self.push(|f| f.as_prefix = f.shallow_prefix.clone());
        }
        // `scope(as:)` under a nested level: `nested_scope?` is true, so
        // `shallow_prefix` is not touched.
        self.push(|f| f.as_prefix = Some(merge_as(f.as_prefix.as_deref(), &member_name)));
        body(self);
        self.pop();
        if shallow {
            self.pop();
        }
        self.pop();
    }

    /// `decomposed_match` without `on:` — the implicit wrapping by scope level.
    fn decomposed_match(&mut self, action: &str, as_option: &AsOption) {
        match self.frame().level() {
            Level::Resources => self.nested(|e| e.add_route(action, as_option)),
            Level::Resource => self.member(|e| e.add_route(action, as_option)),
            _ => self.add_route(action, as_option),
        }
    }

    // ----- DSL verbs -----

    /// `match` / `get` / `post` / `put` / `patch` / `delete`.
    fn map_match(&mut self, node: &CallNode<'pr>) {
        let path_or_action = match self.positional_literal(node, 0) {
            Some(value) => value,
            None => match string_key_mapping(node) {
                Some(path) => path,
                None => {
                    if node.arguments().is_some() {
                        self.skip(
                            node,
                            "route path is not a string or symbol literal".to_string(),
                        );
                    }
                    return;
                }
            },
        };
        let Some(as_option) = self.as_option(node) else {
            self.skip(node, "`as:` is not a literal".to_string());
            return;
        };
        match self.keyword_symbol(node, "on").as_deref() {
            Some("member") => self.member(|e| e.add_route(&path_or_action, &as_option)),
            Some("collection") => self.collection(|e| e.add_route(&path_or_action, &as_option)),
            Some("new") => self.new_scope(|e| e.add_route(&path_or_action, &as_option)),
            _ => self.decomposed_match(&path_or_action, &as_option),
        }
    }

    fn root(&mut self, node: &CallNode<'pr>) {
        let as_option = match self.as_option(node) {
            Some(AsOption::Default) | None => AsOption::Explicit("root".to_string()),
            Some(other) => other,
        };
        if self.frame().level() == Level::Resources {
            self.with_level(Level::Root, |e| e.add_route("/", &as_option));
        } else {
            self.add_route("/", &as_option);
        }
    }

    fn mount(&mut self, node: &CallNode<'pr>) {
        let as_option = match self.as_option(node) {
            Some(AsOption::Explicit(name)) => AsOption::Explicit(name),
            Some(AsOption::None) => return,
            _ => {
                self.skip(
                    node,
                    "mount without `as:` takes its name from the engine's railtie_name".to_string(),
                );
                return;
            }
        };
        let at = self
            .keyword_literal(node, "at")
            .or_else(|| string_key_mapping(node))
            .unwrap_or_else(|| "/".to_string());
        self.add_route(&at, &as_option);
    }

    fn direct(&mut self, node: &CallNode<'pr>) {
        match self.positional_literal(node, 0) {
            Some(name) => self.add_name(name),
            None => self.skip(node, "direct name is not a literal".to_string()),
        }
    }

    /// `scope` / `namespace` share the merge; `namespace` defaults `as` to
    /// its name and `shallow_prefix` to `as`.
    fn scope(&mut self, node: &CallNode<'pr>, namespace: bool) {
        if namespace && self.frame().level().is_resource_scope() {
            // `Resources#namespace`: `nested { super }`.
            self.nested(|e| e.scope_inner(node, true));
        } else {
            self.scope_inner(node, namespace);
        }
    }

    fn scope_inner(&mut self, node: &CallNode<'pr>, namespace: bool) {
        let as_value = match self.keyword_node(node, "as") {
            None => {
                if namespace {
                    match self.positional_literal(node, 0) {
                        Some(name) => Some(name),
                        None => {
                            self.skip(node, "namespace name is not a literal".to_string());
                            return;
                        }
                    }
                } else {
                    None
                }
            }
            Some(value) => match self.literal_string(&value) {
                Some(name) => Some(name),
                None if value.as_nil_node().is_some() || value.as_false_node().is_some() => None,
                None => {
                    self.skip(node, "`as:` is not a literal".to_string());
                    return;
                }
            },
        };
        let shallow_prefix = match self.keyword_node(node, "shallow_prefix") {
            Some(value) => match self.literal_string(&value) {
                Some(prefix) => Some(prefix),
                None => {
                    self.skip(node, "`shallow_prefix:` is not a literal".to_string());
                    return;
                }
            },
            None if namespace => as_value.clone(),
            None => None,
        };
        let only = self.keyword_symbol_list(node, "only");
        let except = self.keyword_symbol_list(node, "except");
        let shallow = self.keyword_bool(node, "shallow");
        let nested_scope = self.frame().level() == Level::Nested;
        self.push(|f| {
            if let Some(as_value) = &as_value {
                // `options[:shallow_prefix] ||= options[:as]` unless nested.
                if !nested_scope && shallow_prefix.is_none() {
                    f.shallow_prefix = Some(merge_as(f.shallow_prefix.as_deref(), as_value));
                }
                f.as_prefix = Some(merge_as(f.as_prefix.as_deref(), as_value));
            }
            if let Some(prefix) = &shallow_prefix {
                f.shallow_prefix = Some(merge_as(f.shallow_prefix.as_deref(), prefix));
            }
            if only.is_some() || except.is_some() {
                f.action_options = Some(ActionOptions { only, except });
            }
            if let Some(shallow) = shallow {
                f.shallow = shallow;
            }
        });
        self.visit_block_body(node);
        self.pop();
    }

    fn resources(&mut self, node: &CallNode<'pr>, singleton: bool) {
        let mut entities = Vec::new();
        let mut positional = 0;
        if let Some(arguments) = node.arguments() {
            for arg in arguments.arguments().iter() {
                if arg.as_keyword_hash_node().is_some() {
                    continue;
                }
                positional += 1;
                match self.literal_string(&arg) {
                    Some(name) => entities.push(name),
                    None => {
                        self.skip(
                            node,
                            "resource name is not a string or symbol literal".to_string(),
                        );
                        return;
                    }
                }
            }
        }
        if positional == 0 {
            return;
        }
        let Some(options) = self.resource_options(node) else {
            return;
        };
        for entity in entities {
            self.resources_one(node, singleton, &entity, options.clone());
        }
    }

    fn resource_options(&mut self, node: &CallNode<'pr>) -> Option<ResourceOptions> {
        let mut options = ResourceOptions::default();
        if let Some(value) = self.keyword_node(node, "as") {
            match self.literal_string(&value) {
                Some(name) => options.as_name = Some(name),
                None => {
                    self.skip(node, "`as:` is not a literal".to_string());
                    return None;
                }
            }
        }
        for (key, slot) in [("only", &mut options.only), ("except", &mut options.except)] {
            if let Some(value) = self.keyword_node(node, key) {
                match self.symbol_list(&value) {
                    Some(list) => *slot = Some(list),
                    None => {
                        self.skip(node, format!("`{key}:` is not a list of symbol literals"));
                        return None;
                    }
                }
            }
        }
        if let Some(value) = self.keyword_node(node, "shallow") {
            match (
                value.as_true_node().is_some(),
                value.as_false_node().is_some(),
            ) {
                (true, _) => options.shallow = Some(true),
                (_, true) => options.shallow = Some(false),
                _ => {
                    self.skip(node, "`shallow:` is not a boolean literal".to_string());
                    return None;
                }
            }
        }
        if let Some(value) = self.keyword_node(node, "shallow_prefix") {
            match self.literal_string(&value) {
                Some(prefix) => options.shallow_prefix = Some(prefix),
                None => {
                    self.skip(node, "`shallow_prefix:` is not a literal".to_string());
                    return None;
                }
            }
        }
        if let Some(value) = self.keyword_node(node, "concerns") {
            match self.symbol_list(&value) {
                Some(list) => options.concerns = list,
                None => {
                    self.skip(
                        node,
                        "`concerns:` is not a list of symbol literals".to_string(),
                    );
                    return None;
                }
            }
        }
        Some(options)
    }

    /// `apply_common_behavior_for` + the resource body, for one entity.
    fn resources_one(
        &mut self,
        node: &CallNode<'pr>,
        singleton: bool,
        entity: &str,
        mut options: ResourceOptions,
    ) {
        if options.shallow == Some(true) {
            options.shallow = None;
            self.push(|f| f.shallow = true);
            self.resources_one(node, singleton, entity, options);
            self.pop();
            return;
        }
        if self.frame().level().is_resource_scope() {
            self.nested(|e| e.resources_one(node, singleton, entity, options));
            return;
        }
        if options.shallow_prefix.is_some() || options.shallow == Some(false) {
            // The residual `scope(**scope_options)` wrap.
            let shallow_prefix = options.shallow_prefix.take();
            let shallow = options.shallow.take();
            let nested_scope = self.frame().level() == Level::Nested;
            self.push(|f| {
                if let Some(prefix) = &shallow_prefix {
                    let _ = nested_scope;
                    f.shallow_prefix = Some(merge_as(f.shallow_prefix.as_deref(), prefix));
                }
                if let Some(shallow) = shallow {
                    f.shallow = shallow;
                }
            });
            self.resources_one(node, singleton, entity, options);
            self.pop();
            return;
        }

        let level = if singleton {
            Level::Resource
        } else {
            Level::Resources
        };
        let default_actions = if singleton {
            RESOURCE_DEFAULT_ACTIONS
        } else {
            RESOURCES_DEFAULT_ACTIONS
        };
        // `apply_action_options`: the call's own `only:` / `except:` win;
        // otherwise the enclosing `scope only:/except:` applies, limited to
        // the actions this resource kind has.
        let (only, except) = if options.only.is_some() || options.except.is_some() {
            (options.only.clone(), options.except.clone())
        } else if let Some(scope_options) = &self.frame().action_options {
            let applicable = |list: &Option<Vec<String>>| {
                list.as_ref().map(|list| {
                    list.iter()
                        .filter(|a| default_actions.contains(&a.as_str()))
                        .cloned()
                        .collect::<Vec<_>>()
                })
            };
            (
                applicable(&scope_options.only),
                applicable(&scope_options.except),
            )
        } else {
            (None, None)
        };
        // `Resource#actions`: `only` narrows, then `except` removes.
        let available: Vec<String> = match &only {
            Some(only) => only.clone(),
            None => default_actions.iter().map(|a| a.to_string()).collect(),
        };
        let actions: Vec<String> = match &except {
            Some(except) => available
                .into_iter()
                .filter(|a| !except.contains(a))
                .collect(),
            None => available,
        };
        let name = options
            .as_name
            .clone()
            .unwrap_or_else(|| entity.to_string());
        let (singular, plural) = if singleton {
            (name.clone(), self.inflector.pluralize(&name))
        } else {
            (self.inflector.singularize(&name), name.clone())
        };
        let resource = ResourceInfo {
            singular,
            plural,
            singleton,
            shallow: self.frame().shallow,
            actions,
        };
        self.push(|f| {
            f.level = Some(level);
            f.resource = Some(resource);
        });
        self.visit_block_body(node);
        let concerns = options.concerns.clone();
        self.expand_concerns(node, &concerns);
        if singleton {
            // `resource`: new, member mappings, then `collection { post :create }`.
            if self.parent_resource().is_some_and(|r| r.has_action("new")) {
                self.new_scope(|e| e.add_route("new", &AsOption::Default));
            }
            self.set_member_mappings();
            if self
                .parent_resource()
                .is_some_and(|r| r.has_action("create"))
            {
                self.collection(|e| e.add_route("create", &AsOption::Default));
            }
        } else {
            self.collection(|e| {
                if e.parent_resource().is_some_and(|r| r.has_action("index")) {
                    e.add_route("index", &AsOption::Default);
                }
                if e.parent_resource().is_some_and(|r| r.has_action("create")) {
                    e.add_route("create", &AsOption::Default);
                }
            });
            if self.parent_resource().is_some_and(|r| r.has_action("new")) {
                self.new_scope(|e| e.add_route("new", &AsOption::Default));
            }
            self.set_member_mappings();
        }
        self.pop();
    }

    /// `set_member_mappings_for_resource`.
    fn set_member_mappings(&mut self) {
        self.member(|e| {
            for action in ["edit", "show", "update", "destroy"] {
                if e.parent_resource().is_some_and(|r| r.has_action(action)) {
                    e.add_route(action, &AsOption::Default);
                }
            }
        });
    }

    fn concern(&mut self, node: &CallNode<'pr>) {
        // Bodies were collected by `ConcernScan`; only report a name that
        // cannot be matched at a use site.
        if self.positional_literal(node, 0).is_none() {
            self.skip(node, "concern name is not a literal".to_string());
        }
    }

    fn concerns_call(&mut self, node: &CallNode<'pr>) {
        let mut names = Vec::new();
        if let Some(arguments) = node.arguments() {
            for arg in arguments.arguments().iter() {
                if arg.as_keyword_hash_node().is_some() {
                    continue;
                }
                match self.symbol_list(&arg) {
                    Some(list) => names.extend(list),
                    None => {
                        self.skip(node, "concern name is not a literal".to_string());
                        return;
                    }
                }
            }
        }
        self.expand_concerns(node, &names);
    }

    fn expand_concerns(&mut self, node: &CallNode<'pr>, names: &[String]) {
        for name in names {
            let Some((file_index, body)) = self.concerns.get(name) else {
                self.skip(node, format!("concern `{name}` is not defined"));
                continue;
            };
            if self.draw_depth >= MAX_DRAW_DEPTH {
                return;
            }
            if !self.visiting_concerns.insert(name.clone()) {
                self.skip(
                    node,
                    format!("concern `{name}` is used inside its own definition"),
                );
                continue;
            }
            self.draw_depth += 1;
            let saved = std::mem::replace(&mut self.file_index, *file_index);
            self.visit(body);
            self.file_index = saved;
            self.draw_depth -= 1;
            self.visiting_concerns.remove(name);
        }
    }

    fn draw(&mut self, node: &CallNode<'pr>) {
        let Some(name) = self.positional_literal(node, 0) else {
            self.skip(node, "draw name is not a literal".to_string());
            return;
        };
        if !draw_name_is_safe(&name) {
            self.skip(node, format!("draw name `{name}` leaves config/routes"));
            return;
        }
        let Some(&index) = self.files.draws.get(&name) else {
            self.skip(node, format!("config/routes/{name}.rb was not found"));
            return;
        };
        if !self.parsed_ok[index] || self.draw_depth >= MAX_DRAW_DEPTH {
            return;
        }
        if !self.visiting_draws.insert(name.clone()) {
            self.skip(node, format!("config/routes/{name}.rb draws itself"));
            return;
        }
        self.draw_depth += 1;
        let saved = std::mem::replace(&mut self.file_index, index);
        let root = self.parse_results[index].node();
        self.visit(&root);
        self.file_index = saved;
        self.draw_depth -= 1;
        self.visiting_draws.remove(&name);
    }

    /// `<array literal>.each { |x| ... }`: bind `x` to each element and walk
    /// the body once per element. A non-literal collection, or a block
    /// that is not a single plain parameter, is reported and skipped.
    fn each_over_literals(&mut self, node: &CallNode<'pr>, receiver: &Node<'pr>) {
        if !matches!(
            node.name().as_slice(),
            b"each" | b"each_with_index" | b"map"
        ) {
            return;
        }
        let Some(block) = node.block().and_then(|block| block.as_block_node()) else {
            return;
        };
        let Some(body) = block.body() else {
            return;
        };
        let elements: Option<Vec<String>> = receiver.as_array_node().and_then(|array| {
            array
                .elements()
                .iter()
                .map(|elem| self.literal_string(&elem))
                .collect()
        });
        let param = block
            .parameters()
            .and_then(|params| params.as_block_parameters_node())
            .and_then(|params| params.parameters())
            .and_then(|params| {
                let requireds: Vec<_> = params.requireds().iter().collect();
                if requireds.len() == 1
                    && params.optionals().iter().next().is_none()
                    && params.rest().is_none()
                    && params.posts().iter().next().is_none()
                    && params.keywords().iter().next().is_none()
                {
                    requireds[0]
                        .as_required_parameter_node()
                        .map(|p| String::from_utf8_lossy(p.name().as_slice()).to_string())
                } else {
                    None
                }
            });
        match (elements, param) {
            (Some(elements), Some(param)) if node.name().as_slice() == b"each" => {
                for element in elements {
                    self.bindings.push((param.clone(), element));
                    self.visit(&body);
                    self.bindings.pop();
                }
            }
            _ => self.skip(
                node,
                "iteration is not `<literal array>.each { |x| ... }`".to_string(),
            ),
        }
    }

    // ----- literal lookups (bindings and `with_options` aware) -----

    fn literal_string(&self, node: &Node<'_>) -> Option<String> {
        if let Some(read) = node.as_local_variable_read_node() {
            let name = read.name().as_slice();
            return self
                .bindings
                .iter()
                .rev()
                .find(|(bound, _)| bound.as_bytes() == name)
                .map(|(_, value)| value.clone());
        }
        literal_string_plain(node)
    }

    fn positional_literal(&self, node: &CallNode<'_>, index: usize) -> Option<String> {
        let arguments = node.arguments()?;
        let arg = arguments
            .arguments()
            .iter()
            .filter(|arg| arg.as_keyword_hash_node().is_none())
            .nth(index)?;
        self.literal_string(&arg)
    }

    /// The value of keyword `name`: on the call itself (last occurrence
    /// wins), else from the innermost enclosing `with_options` that sets it.
    fn keyword_node(&self, node: &CallNode<'pr>, name: &str) -> Option<Node<'pr>> {
        if let Some(arguments) = node.arguments()
            && let Some(found) = keyword_in_arguments(&arguments, name)
        {
            return Some(found);
        }
        self.with_options
            .iter()
            .rev()
            .find_map(|arguments| keyword_in_arguments(arguments, name))
    }

    fn keyword_literal(&self, node: &CallNode<'pr>, name: &str) -> Option<String> {
        self.literal_string(&self.keyword_node(node, name)?)
    }

    fn keyword_symbol(&self, node: &CallNode<'pr>, name: &str) -> Option<String> {
        self.keyword_literal(node, name)
    }

    fn keyword_bool(&self, node: &CallNode<'pr>, name: &str) -> Option<bool> {
        let value = self.keyword_node(node, name)?;
        if value.as_true_node().is_some() {
            Some(true)
        } else if value.as_false_node().is_some() {
            Some(false)
        } else {
            None
        }
    }

    /// `[:a, :b]`, `%i[a b]` or a lone `:a`; `None` for anything else.
    fn symbol_list(&self, node: &Node<'_>) -> Option<Vec<String>> {
        if let Some(single) = self.literal_string(node) {
            return Some(vec![single]);
        }
        let array = node.as_array_node()?;
        let mut list = Vec::new();
        for elem in array.elements().iter() {
            list.push(self.literal_string(&elem)?);
        }
        Some(list)
    }

    fn keyword_symbol_list(&self, node: &CallNode<'pr>, name: &str) -> Option<Vec<String>> {
        self.symbol_list(&self.keyword_node(node, name)?)
    }

    /// `as:` as written: absent → `Default`; literal → `Explicit`; `nil` /
    /// `false` → `None`; anything else → `Option::None` (not a literal).
    fn as_option(&self, node: &CallNode<'pr>) -> Option<AsOption> {
        let Some(value) = self.keyword_node(node, "as") else {
            return Some(AsOption::Default);
        };
        if let Some(name) = self.literal_string(&value) {
            return Some(AsOption::Explicit(name));
        }
        if value.as_nil_node().is_some() || value.as_false_node().is_some() {
            return Some(AsOption::None);
        }
        None
    }
}

impl<'pr> Visit<'pr> for Evaluator<'pr> {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if let Some(receiver) = node.receiver() {
            // `%w(/a /b).each { |path| get path, to: ... }` is static
            // iteration over literals; everything else with a receiver
            // (`Rails.env.development?`, `Module.new { ... }`,
            // `map.resources`) is not the routing DSL and is not descended
            // into.
            self.each_over_literals(node, &receiver);
            return;
        }
        let name = node.name().as_slice();
        match name {
            b"with_options" => {
                if let Some(arguments) = node.arguments() {
                    self.with_options.push(arguments);
                    self.visit_block_body(node);
                    self.with_options.pop();
                } else {
                    self.visit_block_body(node);
                }
            }
            b"resources" => self.resources(node, false),
            b"resource" => self.resources(node, true),
            b"get" | b"post" | b"put" | b"patch" | b"delete" | b"match" | b"options" | b"head" => {
                self.map_match(node)
            }
            b"root" => self.root(node),
            b"namespace" => self.scope(node, true),
            b"scope" => self.scope(node, false),
            b"member" => {
                if self.frame().level().is_resource_scope() {
                    self.member(|e| e.visit_block_body(node));
                }
            }
            b"collection" => {
                if self.frame().level().is_resource_scope() {
                    self.collection(|e| e.visit_block_body(node));
                }
            }
            b"new" => {
                if self.frame().level().is_resource_scope() {
                    self.new_scope(|e| e.visit_block_body(node));
                }
            }
            b"nested" => {
                if self.frame().level().is_resource_scope() {
                    self.nested(|e| e.visit_block_body(node));
                }
            }
            b"shallow" => {
                self.push(|f| f.shallow = true);
                self.visit_block_body(node);
                self.pop();
            }
            b"concern" => self.concern(node),
            b"concerns" => self.concerns_call(node),
            b"draw" => self.draw(node),
            b"mount" => self.mount(node),
            b"direct" => self.direct(node),
            // `resolve` maps models to existing helpers; defines none.
            b"resolve" => {}
            _ => {
                if GEM_ROUTE_DSLS.contains(&String::from_utf8_lossy(name).as_ref()) {
                    self.skip(
                        node,
                        "named routes of this gem DSL are defined by the gem, not by routes.rb"
                            .to_string(),
                    );
                    return;
                }
                // `constraints` / `defaults` / `controller` / `devise_scope` /
                // `authenticate` / ...: wrappers that change no name. Their
                // block bodies are routes in the current scope.
                self.visit_block_body(node);
            }
        }
    }

    fn visit_def_node(&mut self, _node: &DefNode<'pr>) {}

    fn visit_lambda_node(&mut self, _node: &LambdaNode<'pr>) {}
}

// ----- literal helpers -----

/// `Mapper.normalize_name`: `normalize_path(name)[1..-1].tr("/", "_")`.
fn normalize_name(name: &str) -> String {
    let mut path = String::with_capacity(name.len() + 1);
    path.push('/');
    path.push_str(name);
    let mut squeezed = String::with_capacity(path.len());
    let mut last_slash = false;
    for c in path.chars() {
        if c == '/' {
            if last_slash {
                continue;
            }
            last_slash = true;
        } else {
            last_slash = false;
        }
        squeezed.push(c);
    }
    let trimmed = if squeezed == "/" {
        squeezed.as_str()
    } else {
        squeezed.trim_end_matches('/')
    };
    trimmed[1..].replace('/', "_")
}

/// `merge_as_scope` / `merge_shallow_prefix_scope`.
fn merge_as(parent: Option<&str>, child: &str) -> String {
    match parent {
        Some(parent) => format!("{parent}_{child}"),
        None => child.to_string(),
    }
}

/// A symbol or plain string literal.
fn literal_string_plain(node: &Node<'_>) -> Option<String> {
    if let Some(symbol) = node.as_symbol_node() {
        return Some(String::from_utf8_lossy(symbol.unescaped()).to_string());
    }
    node.as_string_node()
        .map(|string| String::from_utf8_lossy(string.unescaped()).to_string())
}

fn positional_literal_plain(node: &CallNode<'_>, index: usize) -> Option<String> {
    let arguments = node.arguments()?;
    let arg = arguments
        .arguments()
        .iter()
        .filter(|arg| arg.as_keyword_hash_node().is_none())
        .nth(index)?;
    literal_string_plain(&arg)
}

/// The value of keyword `name` in an argument list (last occurrence wins).
fn keyword_in_arguments<'pr>(
    arguments: &ruby_prism::ArgumentsNode<'pr>,
    name: &str,
) -> Option<Node<'pr>> {
    let mut found = None;
    for arg in arguments.arguments().iter() {
        let Some(kw_hash) = arg.as_keyword_hash_node() else {
            continue;
        };
        for elem in kw_hash.elements().iter() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            let Some(sym) = assoc.key().as_symbol_node() else {
                continue;
            };
            if sym.unescaped() == name.as_bytes() {
                found = Some(assoc.value());
            }
        }
    }
    found
}

/// `get "path" => "controller#action"` / `mount App => "/path"`: the first
/// string-keyed pair of the keyword hash. For `mount` the string is the
/// value; for `match` it is the key.
fn string_key_mapping(node: &CallNode<'_>) -> Option<String> {
    let arguments = node.arguments()?;
    for arg in arguments.arguments().iter() {
        let Some(kw_hash) = arg.as_keyword_hash_node() else {
            continue;
        };
        for elem in kw_hash.elements().iter() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            if let Some(key) = assoc.key().as_string_node() {
                return Some(String::from_utf8_lossy(key.unescaped()).to_string());
            }
            if assoc.key().as_symbol_node().is_none()
                && let Some(value) = assoc.value().as_string_node()
            {
                return Some(String::from_utf8_lossy(value.unescaped()).to_string());
            }
        }
    }
    None
}

