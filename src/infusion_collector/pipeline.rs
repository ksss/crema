//! Shared DSL collector pipeline.

use std::path::Path;
use std::sync::Arc;

use ruby_prism::{
    CallNode, ClassNode, ConstantPathNode, DefNode, ModuleNode, Node, SingletonClassNode, Visit,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::MethodKind;
use crate::ast::ruby::annotations::LeadingAnnotation;
use crate::ast::ruby::declarations::{ClassDecl, Declaration, ModuleDecl};
use crate::ast::ruby::members::{
    AttrAccessorMember, AttrReaderMember, AttrWriterMember, AttributeMember, AttributeNameNode,
    DefMember, DefMemberOrigin, ExtendMember, IncludeMember, Member, MethodTypeAnnotation,
    MixinMember, PrependMember, TrailingResolution, TypeAnnotations,
};
use crate::ast::ruby::{LineIndex, PrismByteRange};
use crate::config::InfusionOptions;
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::draft::EnvironmentDraft;
use crate::environment::frozen::ConcernBlockTargets;

use crate::environment::ruby_decl::build_annotation_syntax_error;
use crate::infusion_collector::activerecord::{
    ActiveRecordAssociation, ActiveRecordEnumMapping, ActiveRecordScope,
};
use crate::infusion_collector::inflector::{self, Inflector};
use crate::infusion_collector::{
    active_record_synthesis, activemodel, activerecord, activesupport,
};
use crate::inline_parser::{
    CommentAssociation, TrailingAnnotation, collect_consecutive_leading, collect_trailing_block,
    prism_location_range, push_class_abs_path,
};
use crate::name::{Name, NameTable};
use crate::rbs_raw::Parser as RbsParser;
use crate::source_ref::SourceRef;
use crate::type_name::TypeName;

/// Walk `parse_result` and build declarations whose members are
/// synthesized by enabled DSL rules.
pub fn collect(
    source: &[u8],
    file: Option<&Path>,
    parse_result: &ruby_prism::ParseResult<'_>,
    names: &NameTable,
) -> (Vec<Declaration>, Vec<Diagnostic>) {
    let root = parse_result.node();
    let mut collector = Collector::new(
        names,
        0,
        None,
        source,
        file,
        parse_result,
        activesupport_options(),
        inflector::default_en(),
        true,
    );
    collector.visit(&root);
    (collector.top_level, collector.diagnostics)
}

pub struct SourceUnit<'a> {
    pub source: &'a [u8],
    pub parse_result: &'a ruby_prism::ParseResult<'a>,
    pub file: Option<&'a Path>,
}

/// `collect` + draft merge. Files whose `parse_result` carries any
/// Prism error are skipped (same stance as
/// `inline_parser::load_inline_annotations`).
pub fn load(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<&Path>,
    draft: &mut EnvironmentDraft,
) -> Vec<Diagnostic> {
    load_all(
        &[SourceUnit {
            source,
            parse_result,
            file,
        }],
        draft,
    )
}

pub fn load_all<'a>(sources: &[SourceUnit<'a>], draft: &mut EnvironmentDraft) -> Vec<Diagnostic> {
    load_all_with_options(
        sources,
        draft,
        activesupport_options(),
        inflector::default_en(),
    )
}

/// Collects in inline mode; `load_all_with_schema` takes the mode.
pub fn load_all_with_options<'a>(
    sources: &[SourceUnit<'a>],
    draft: &mut EnvironmentDraft,
    options: InfusionOptions,
    inflector: &Inflector,
) -> Vec<Diagnostic> {
    load_all_with_schema(sources, draft, options, inflector, true, None)
}

/// `load_all_with_options` plus the pre-parsed ActiveRecord schema
/// (`activerecord::prepare_schema`). The schema declarations are emitted
/// here — after concern expansion, so the column-accessor suppression sees
/// every enum the model ends up with, and before AR synthesis, so the
/// synthesized decls still observe the schema classes in the draft.
/// `inline_mode` is passed through to [`collect_source`].
pub fn load_all_with_schema<'a>(
    sources: &[SourceUnit<'a>],
    draft: &mut EnvironmentDraft,
    options: InfusionOptions,
    inflector: &Inflector,
    inline_mode: bool,
    schema: Option<activerecord::PreparsedSchema>,
) -> Vec<Diagnostic> {
    let collected = sources
        .iter()
        .enumerate()
        .map(|(source_index, unit)| {
            if unit.parse_result.errors().next().is_some() {
                return None;
            }
            let source_file = unit
                .file
                .map(|file| draft.names().intern(&file.to_string_lossy()));
            Some(collect_source(
                draft.names(),
                source_index,
                source_file,
                unit.source,
                unit.file,
                unit.parse_result,
                options,
                inflector,
                inline_mode,
            ))
        })
        .collect();
    load_collected(collected, draft, options, inflector, schema)
}

/// The collect half of `load_all_with_schema` for one file: walk the
/// AST and return everything the insert half needs, owned (no AST
/// borrow). Runs on a parallel-ingest worker (ADR-0033), so `names` may
/// be a worker table: `source_file` must be pre-interned by the caller
/// and no `Name` is minted here. `source_index` is the position the
/// caller will give this result in the `Vec` passed to
/// [`load_collected`]; concern expansion resolves the concern's file
/// through it.
///
/// `inline_mode` is the CLI's `--inline`. In sig mode the members a
/// `class_methods do` block synthesizes are collected as if their
/// `#:` / `# @rbs` annotations were absent (ADR-0027: method and
/// attribute annotations are the inline class), so the file's comments
/// are not read at all.
#[allow(clippy::too_many_arguments)]
pub fn collect_source<'a>(
    names: &NameTable,
    source_index: usize,
    source_file: Option<Name>,
    source: &'a [u8],
    file: Option<&'a Path>,
    parse_result: &ruby_prism::ParseResult<'_>,
    options: InfusionOptions,
    inflector: &Inflector,
    inline_mode: bool,
) -> CollectedSource<'a> {
    let mut collector = Collector::new(
        names,
        source_index,
        source_file,
        source,
        file,
        parse_result,
        options,
        inflector,
        inline_mode,
    );
    let root = parse_result.node();
    collector.visit(&root);
    let mut diagnostics = collector.diagnostics;
    let table_name_assignments = if options.activerecord {
        activerecord::collect_table_name_assignments(&root, file, source, &mut diagnostics)
    } else {
        Vec::new()
    };
    CollectedSource {
        source: SourceRef::Bytes(source),
        file,
        top_level: collector.top_level,
        active_record_associations: collector.active_record_associations,
        active_record_scopes: collector.active_record_scopes,
        active_record_enum_mappings: collector.active_record_enum_mappings,
        paranoia_models: collector.paranoia_models,
        concerns: collector.concerns,
        concern_sites: collector.concern_sites,
        table_name_assignments,
        diagnostics,
    }
}

impl<'a> CollectedSource<'a> {
    /// Attach a decoded record to the current run: the file's walk index
    /// (stamped on every concern, as `collect_source` did), its path and
    /// its on-demand source.
    pub fn rebind(&mut self, source_index: usize, source: SourceRef<'a>, file: Option<&'a Path>) {
        self.source = source;
        self.file = file;
        for concern in &mut self.concerns {
            concern.source_index = source_index;
        }
    }
}

/// The insert half of `load_all_with_schema`: draft insertion, concern
/// expansion, schema emission and ActiveRecord / paranoia / zeitwerk
/// synthesis, single-threaded. `collected[i]` is the result of
/// [`collect_source`] called with `source_index == i`, or `None` for a
/// file that was not collected (Prism error), so the indices concern
/// expansion carries stay valid.
pub fn load_collected<'a>(
    collected: Vec<Option<CollectedSource<'a>>>,
    draft: &mut EnvironmentDraft,
    options: InfusionOptions,
    inflector: &Inflector,
    schema: Option<activerecord::PreparsedSchema>,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut collected = collected;
    for source in collected.iter_mut().flatten() {
        diagnostics.append(&mut source.diagnostics);
    }
    let collected_at = |index: usize| -> &CollectedSource<'a> {
        collected[index]
            .as_ref()
            .expect("concern source_index must point at a collected file")
    };

    for source in collected.iter().flatten() {
        for decl in &source.top_level {
            draft.insert_ruby_decl(decl, source.source, source.file, &mut diagnostics);
        }
        for concern in &source.concerns {
            let Some(class_methods) = concern.class_methods.as_ref() else {
                continue;
            };
            let Some(decl) = class_methods.to_declaration() else {
                continue;
            };
            draft.insert_ruby_decl(&decl, source.source, source.file, &mut diagnostics);
        }
    }

    let mut concerns = FxHashMap::default();
    let mut concern_names = FxHashSet::default();
    for source in collected.iter().flatten() {
        for concern in &source.concerns {
            concern_names.insert(concern.name);
            concerns.insert(concern.name, concern);
        }
    }

    // Every concern block gets an entry up front, targets or not: the
    // type checker skips a block with zero targets (Rails never runs it)
    // and must tell that apart from a block the pipeline never collected
    // (which keeps the default module-singleton walk).
    let mut concern_block_targets: FxHashMap<TypeName, Vec<ConcernBlockTargets>> =
        FxHashMap::default();
    for source in collected.iter().flatten() {
        let source_file = source
            .file
            .map(|f| draft.names().intern(&f.to_string_lossy()));
        for concern in &source.concerns {
            for body in [
                concern.included_body.as_ref(),
                concern.prepended_body.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                concern_block_targets
                    .entry(concern.name)
                    .or_default()
                    .push(ConcernBlockTargets {
                        source_file,
                        location: body.name_location,
                        targets: Vec::new(),
                    });
            }
        }
    }

    let mut emitted_bodies = FxHashSet::default();
    let mut emitted_class_methods = FxHashSet::default();

    // Expansion contributions per target, keyed by the emitting concern
    // file: `model_owner` must not count them as a second declaring file
    // (they are re-creatable from the concern side — ADR-0028 S2d), while
    // a genuine cross-file reopen still disqualifies the owner.
    let mut expansion_contributions: FxHashSet<(TypeName, Name)> = FxHashSet::default();
    let mut active_record_associations = collected
        .iter()
        .flatten()
        .flat_map(|source| source.active_record_associations.iter().cloned())
        .collect::<Vec<_>>();
    let mut active_record_scopes = collected
        .iter()
        .flatten()
        .flat_map(|source| source.active_record_scopes.iter().cloned())
        .collect::<Vec<_>>();
    let mut active_record_enum_mappings = collected
        .iter()
        .flatten()
        .flat_map(|source| source.active_record_enum_mappings.iter().cloned())
        .collect::<Vec<_>>();
    let mut paranoia_models = collected
        .iter()
        .flatten()
        .flat_map(|source| source.paranoia_models.iter().copied())
        .collect::<Vec<_>>();
    for source in collected.iter().flatten() {
        for site in &source.concern_sites {
            let Some(concern) = resolve_concern(&concerns, &site.module_names) else {
                continue;
            };
            if site.target == concern.name {
                continue;
            }
            if concern_names.contains(&site.target) {
                continue;
            }
            let mut synthetic_bodies = Vec::new();
            expand_concern_bodies(
                concern,
                &concerns,
                ConcernExpansionTarget {
                    target: site.target,
                    target_kind: site.target_kind,
                },
                site.kind.body_kind(),
                &mut Vec::new(),
                &mut emitted_bodies,
                &mut synthetic_bodies,
            );
            for synthetic in synthetic_bodies {
                let associations = synthetic
                    .body
                    .active_record_associations(draft.names(), options);
                let scopes = synthetic.body.active_record_scopes(options);
                let enum_mappings = synthetic
                    .body
                    .active_record_enum_mappings(options, inflector);
                // Extend before `apply_body`'s empty-member bail: a
                // concern body whose only call is `acts_as_paranoid`
                // yields no members (and thus no decl) but still marks
                // its include target as a paranoia model.
                paranoia_models.extend(synthetic.body.paranoia_model(options));
                // Also before the bail, for the same reason: the target
                // list must not depend on whether any member came out.
                if let Some(entry) =
                    concern_block_targets
                        .get_mut(&synthetic.concern)
                        .and_then(|entries| {
                            entries
                                .iter_mut()
                                .find(|e| e.location == synthetic.body.name_location)
                        })
                    && !entry.targets.contains(&site.target)
                {
                    entry.targets.push(site.target);
                }

                let Some(decl) = Collector::apply_body(
                    synthetic.body,
                    Vec::new(),
                    options,
                    inflector,
                    draft.names(),
                ) else {
                    continue;
                };
                active_record_associations.extend(associations);
                active_record_scopes.extend(scopes);
                active_record_enum_mappings.extend(enum_mappings);
                let concern_source = collected_at(synthetic.source_index);
                if let Some(f) = concern_source.file {
                    expansion_contributions
                        .insert((site.target, draft.names().intern(&f.to_string_lossy())));
                }
                draft.insert_ruby_decl(
                    &decl,
                    concern_source.source,
                    concern_source.file,
                    &mut diagnostics,
                );
            }
            let mut class_method_extends = Vec::new();
            expand_concern_class_methods(
                concern,
                &concerns,
                ConcernExpansionTarget {
                    target: site.target,
                    target_kind: site.target_kind,
                },
                site.kind.class_methods_mixin_kind(),
                &mut Vec::new(),
                &mut emitted_class_methods,
                &mut class_method_extends,
            );
            for synthetic in class_method_extends {
                let Some(decl) = synthetic.to_declaration(draft.names()) else {
                    continue;
                };
                let concern_source = collected_at(synthetic.source_index);
                if let Some(f) = concern_source.file {
                    expansion_contributions
                        .insert((site.target, draft.names().intern(&f.to_string_lossy())));
                }
                draft.insert_ruby_decl(
                    &decl,
                    concern_source.source,
                    concern_source.file,
                    &mut diagnostics,
                );
            }
        }
    }

    draft.set_concern_block_targets(concern_block_targets);

    if options.activerecord {
        if let Some(schema) = schema.as_ref() {
            let mut enums_by_attr: FxHashMap<TypeName, FxHashSet<String>> = FxHashMap::default();
            for mapping in &active_record_enum_mappings {
                enums_by_attr
                    .entry(mapping.owner)
                    .or_default()
                    .insert(mapping.attr_name.clone());
            }
            let table_names = activerecord::table_name_overrides(
                collected
                    .iter()
                    .flatten()
                    .flat_map(|source| &source.table_name_assignments),
            );
            activerecord::emit_schema(
                schema,
                &table_names,
                draft,
                inflector,
                &enums_by_attr,
                &mut diagnostics,
            );
        }
        let batch_files: FxHashSet<Name> = collected
            .iter()
            .flatten()
            .filter_map(|source| {
                source
                    .file
                    .map(|file| draft.names().intern(&file.to_string_lossy()))
            })
            .collect();
        active_record_synthesis::synthesize(
            draft,
            &active_record_associations,
            &active_record_scopes,
            &active_record_enum_mappings,
            inflector,
            &batch_files,
            &expansion_contributions,
        );
        // After AR synthesis so the mixin targets
        // (`<Model>::ActiveRecord_Relation` etc.) exist in the draft.
        // `options.paranoia` implies `options.activerecord` (config
        // resolution rejects the standalone combination), so nesting
        // under the activerecord gate loses no case.
        if options.paranoia {
            super::paranoia::synthesize(draft, &paranoia_models);
        }
    }

    // sidekiq: class-body `include Sidekiq::Job` sites, matched after
    // alias normalization inside the pass. Module targets are dropped
    // here — see `sidekiq::synthesize` for why.
    if options.sidekiq {
        let sites: Vec<super::sidekiq::SidekiqSite> = collected
            .iter()
            .flatten()
            .flat_map(|source| source.concern_sites.iter())
            .filter(|site| {
                matches!(site.kind, ConcernSiteKind::Include)
                    && matches!(site.target_kind, InfusionOwnerKind::Class)
            })
            .map(|site| super::sidekiq::SidekiqSite {
                target: site.target,
                module_names: site.module_names.clone(),
            })
            .collect();
        super::sidekiq::synthesize(draft, &sites);
    }

    // Zeitwerk-style implicit namespace synthesis (see
    // `crate::infusion_collector::zeitwerk_synthesis`). Runs after every
    // Ruby-source declaration (top-level + concern expansion +
    // active-record synth) has been inserted, so proper-prefix
    // enumeration sees the full draft. Gated on rails-on: standalone
    // activerecord/activemodel/activesupport must not synthesize
    // namespaces, since compact `class A::B` outside a Rails app is a
    // plain-Ruby NameError we intentionally preserve.
    if options.rails_enabled() {
        super::zeitwerk_synthesis::synthesize(draft);
        // Reads the Ruby `def`s of every `ActionMailer::Base` descendant,
        // so it too runs after every Ruby-source declaration is in.
        super::action_mailer::synthesize(draft);
    }

    diagnostics
}

fn activesupport_options() -> InfusionOptions {
    InfusionOptions {
        activesupport: true,
        activemodel: false,
        activerecord: false,
        paranoia: false,
        sidekiq: false,
    }
}

/// One file's collect-half output, handed from [`collect_source`] to
/// [`load_collected`]. Owned throughout (no AST borrow), so it can cross
/// the ingest worker → main boundary.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CollectedSource<'a> {
    /// Not persisted: a cached file's bytes are read on demand
    /// ([`CollectedSource::rebind`]).
    #[serde(skip)]
    source: SourceRef<'a>,
    /// Not persisted: re-bound to the current run's path on decode.
    #[serde(skip)]
    file: Option<&'a Path>,
    top_level: Vec<Declaration>,
    active_record_associations: Vec<ActiveRecordAssociation>,
    active_record_scopes: Vec<ActiveRecordScope>,
    active_record_enum_mappings: Vec<ActiveRecordEnumMapping>,
    paranoia_models: Vec<TypeName>,
    concerns: Vec<ConcernDef>,
    concern_sites: Vec<ConcernSite>,
    /// `self.table_name =` assignments as `(absolute model path, table)`
    /// in source order; empty unless the activerecord provider is on.
    table_name_assignments: Vec<(String, String)>,
    /// Collection-time diagnostics, drained by `load_collected` in
    /// file order before any insert-time diagnostic is pushed.
    diagnostics: Vec<Diagnostic>,
}

struct Collector<'a> {
    names: &'a NameTable,
    options: InfusionOptions,
    inflector: &'a Inflector,
    source_index: usize,
    source_file: Option<Name>,
    source: &'a [u8],
    file: Option<&'a Path>,
    line_index: LineIndex,
    /// `None` in sig mode: the only reader is `ClassMethodsCollector`,
    /// whose annotations sig mode does not read.
    comments: Option<CommentAssociation>,
    class_stack: Vec<String>,
    scope_stack: Vec<ScopeFrame>,
    top_level: Vec<Declaration>,
    active_record_associations: Vec<ActiveRecordAssociation>,
    active_record_scopes: Vec<ActiveRecordScope>,
    active_record_enum_mappings: Vec<ActiveRecordEnumMapping>,
    paranoia_models: Vec<TypeName>,
    diagnostics: Vec<Diagnostic>,
    concerns: Vec<ConcernDef>,
    concern_sites: Vec<ConcernSite>,
}

struct ScopeFrame {
    body: InfusionBody,
    nested_decls: Vec<Declaration>,
    dependencies: Vec<ConcernDependency>,
    extends_concern: bool,
    class_methods_module: Option<ClassMethodsModuleRef>,
    included_body: Option<InfusionBody>,
    prepended_body: Option<InfusionBody>,
    class_methods: Option<ClassMethodsBody>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct InfusionBody {
    owner: TypeName,
    owner_kind: InfusionOwnerKind,
    name_location: PrismByteRange,
    calls: Vec<InfusionCall>,
    members: Vec<Member>,
    origin: InfusionBodyOrigin,
}

impl InfusionBody {
    fn active_record_associations(
        &self,
        names: &NameTable,
        options: InfusionOptions,
    ) -> Vec<ActiveRecordAssociation> {
        if !options.activerecord || !matches!(self.owner_kind, InfusionOwnerKind::Class) {
            return Vec::new();
        }
        self.calls
            .iter()
            .filter_map(|call| activerecord::association_from_call(names, self.owner, call))
            .collect()
    }

    fn active_record_scopes(&self, options: InfusionOptions) -> Vec<ActiveRecordScope> {
        if !options.activerecord || !matches!(self.owner_kind, InfusionOwnerKind::Class) {
            return Vec::new();
        }
        // One interleaved pass in call order: a `scope :name` after an
        // `enum` value of the same name (or vice versa) must win the
        // last-wins dedup downstream exactly as the runtime redefinition
        // does. Chaining two per-kind passes would order all enum scopes
        // after all real scopes regardless of source order.
        self.calls
            .iter()
            .flat_map(|call| {
                activerecord::scope_from_call(self.owner, call)
                    .into_iter()
                    .chain(activerecord::enum_scopes_from_call(self.owner, call))
            })
            .collect()
    }

    fn active_record_enum_mappings(
        &self,
        options: InfusionOptions,
        inflector: &Inflector,
    ) -> Vec<ActiveRecordEnumMapping> {
        if !options.activerecord || !matches!(self.owner_kind, InfusionOwnerKind::Class) {
            return Vec::new();
        }
        self.calls
            .iter()
            .filter_map(|call| activerecord::enum_mapping_from_call(self.owner, call, inflector))
            .collect()
    }

    /// The body's owner when it carries a receiver-less
    /// `acts_as_paranoid` call — a candidate for paranoia mixin
    /// synthesis (`super::paranoia`). Candidate only: the synthesis
    /// pass still requires the owner to be an `ActiveRecord::Base`
    /// descendant, mirroring orthoses-paranoia's
    /// `ActiveRecord::Base.acts_as_paranoid` trace target.
    fn paranoia_model(&self, options: InfusionOptions) -> Option<TypeName> {
        if !options.paranoia || !matches!(self.owner_kind, InfusionOwnerKind::Class) {
            return None;
        }
        self.calls
            .iter()
            .any(|call| call.name == "acts_as_paranoid")
            .then_some(self.owner)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum InfusionOwnerKind {
    Class,
    Module,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum InfusionBodyOrigin {
    Real,
    SyntheticConcernIncluded,
    SyntheticConcernPrepended,
    /// The synthesized `<owner>::<Topic>` module body of a
    /// `concerning :Topic do ... end` call. `def`s collected while this
    /// origin is active become `DefMemberOrigin::SyntheticConcerning`
    /// members of the synthesized module — see that variant's doc for why
    /// they need no check-context retargeting (their source nodes are
    /// checked lexically in the owner's instance context).
    SyntheticConcerningModule,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionCall {
    pub(crate) name: String,
    pub(crate) symbol_args: Vec<InfusionSymbolArg>,
    pub(crate) enum_values: Vec<InfusionEnumValue>,
    pub(crate) keyword_bools: Vec<InfusionKeywordBool>,
    pub(crate) keyword_strings: Vec<InfusionKeywordString>,
    /// `scope` calls only: the body lambda's parameter structure when the
    /// body is a `->` literal, `None` for non-literal bodies (constant,
    /// method call, `lambda { }`) — those fall back to `(?)`. Mirrors
    /// orthoses scope.rb's runtime `body.to_proc.parameters`, taken
    /// statically (best-effort, user decision 2026-08-26).
    pub(crate) scope_lambda_params: Option<InfusionScopeParams>,
    pub(crate) location: PrismByteRange,
}

/// Parameter structure of a `scope` body lambda — the static counterpart
/// of Ruby `Proc#parameters` kinds (`req`/`opt`/`rest`/`post`(= trailing)/
/// `keyreq`/`key`/`keyrest`/`block`). Only structure is carried; every
/// param is typed `untyped` at synthesis (orthoses `parameters_to_type`).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionScopeParams {
    pub(crate) requireds: Vec<String>,
    pub(crate) optionals: Vec<String>,
    pub(crate) rest: bool,
    pub(crate) trailings: Vec<String>,
    pub(crate) required_keywords: Vec<String>,
    pub(crate) optional_keywords: Vec<String>,
    pub(crate) keyrest: bool,
    pub(crate) block: bool,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionSymbolArg {
    pub(crate) name: String,
    pub(crate) location: PrismByteRange,
}

/// One `enum` label with its statically-known value literal. `value` is
/// `None` for non-literal values (constant reference, method call) —
/// consumers fall back to `untyped` for the value-typed surface only
/// (todo enum_typed_signature: labels still type the string/symbol side).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionEnumValue {
    pub(crate) name: String,
    pub(crate) location: PrismByteRange,
    pub(crate) value: Option<EnumLiteral>,
}

/// A statically-known `enum` value literal. Integers are carried as
/// decimal strings, matching `ast::types::Literal::Integer(String)`.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum EnumLiteral {
    Int(String),
    Str(String),
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionKeywordBool {
    pub(crate) name: String,
    pub(crate) value: Option<bool>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InfusionKeywordString {
    pub(crate) name: String,
    pub(crate) value: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ConcernDef {
    name: TypeName,
    /// The walk index of the file this concern was collected from — a
    /// property of the run, not of the file, so a cached record's value
    /// is overwritten on decode ([`CollectedSource::rebind`]).
    source_index: usize,
    dependencies: Vec<ConcernDependency>,
    class_methods_module: Option<ClassMethodsModuleRef>,
    included_body: Option<InfusionBody>,
    prepended_body: Option<InfusionBody>,
    class_methods: Option<ClassMethodsBody>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ConcernDependency {
    module_name_candidates: Vec<TypeName>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ConcernSite {
    target: TypeName,
    target_kind: InfusionOwnerKind,
    kind: ConcernSiteKind,
    module_names: Vec<TypeName>,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum ConcernSiteKind {
    Include,
    Prepend,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ConcernBodyKind {
    Included,
    Prepended,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AttributeCallSiteKind {
    Reader,
    Writer,
    Accessor,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SyntheticClassMethodsMixinKind {
    Extend,
    SingletonPrepend,
}

struct SyntheticBody {
    source_index: usize,
    /// The concern module whose block `body` was cloned from — the key
    /// `load_collected` records the block's targets under.
    concern: TypeName,
    body: InfusionBody,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ClassMethodsBody {
    module_name: TypeName,
    name_location: PrismByteRange,
    members: Vec<Member>,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct ClassMethodsModuleRef {
    module_name: TypeName,
    location: PrismByteRange,
}

struct SyntheticClassMethodsExtend {
    source_index: usize,
    target: TypeName,
    target_kind: InfusionOwnerKind,
    mixin_kind: SyntheticClassMethodsMixinKind,
    class_methods_module: TypeName,
    location: PrismByteRange,
}

struct ClassMethodsCollector<'a> {
    names: &'a NameTable,
    /// Pre-interned `file`, stamped on attr members. Never interned here:
    /// on a parallel-ingest worker `names` is a table that must not
    /// mint positional `Name`s (see `NameTable::merge`).
    source_file: Option<Name>,
    source: &'a [u8],
    file: Option<&'a Path>,
    line_index: &'a LineIndex,
    /// `None` in sig mode: members are collected as if unannotated.
    comments: Option<&'a CommentAssociation>,
    members: Vec<Member>,
    diagnostics: Vec<Diagnostic>,
}

struct IncludedSingletonClassCollector {
    members: Vec<Member>,
    origin: DefMemberOrigin,
    source_file: Option<Name>,
}

#[derive(Clone, Copy)]
struct ConcernExpansionTarget {
    target: TypeName,
    target_kind: InfusionOwnerKind,
}

fn resolve_concern<'a>(
    concerns: &FxHashMap<TypeName, &'a ConcernDef>,
    candidates: &[TypeName],
) -> Option<&'a ConcernDef> {
    candidates
        .iter()
        .find_map(|name| concerns.get(name).copied())
}

fn expand_concern_bodies(
    concern: &ConcernDef,
    concerns: &FxHashMap<TypeName, &ConcernDef>,
    expansion_target: ConcernExpansionTarget,
    body_kind: ConcernBodyKind,
    visiting: &mut Vec<TypeName>,
    emitted: &mut FxHashSet<(TypeName, TypeName, ConcernBodyKind)>,
    out: &mut Vec<SyntheticBody>,
) {
    if visiting.contains(&concern.name) {
        return;
    }
    if !emitted.insert((expansion_target.target, concern.name, body_kind)) {
        return;
    }

    visiting.push(concern.name);
    for dependency in &concern.dependencies {
        let Some(dependency) = resolve_concern(concerns, &dependency.module_name_candidates) else {
            continue;
        };
        expand_concern_bodies(
            dependency,
            concerns,
            expansion_target,
            body_kind,
            visiting,
            emitted,
            out,
        );
    }
    visiting.pop();

    if let Some(body) = concern.body(body_kind) {
        out.push(SyntheticBody {
            source_index: concern.source_index,
            concern: concern.name,
            body: body.synthetic_for(
                expansion_target.target,
                expansion_target.target_kind,
                body_kind.origin(),
            ),
        });
    }
}

fn expand_concern_class_methods(
    concern: &ConcernDef,
    concerns: &FxHashMap<TypeName, &ConcernDef>,
    expansion_target: ConcernExpansionTarget,
    mixin_kind: SyntheticClassMethodsMixinKind,
    visiting: &mut Vec<TypeName>,
    emitted: &mut FxHashSet<(TypeName, TypeName, SyntheticClassMethodsMixinKind)>,
    out: &mut Vec<SyntheticClassMethodsExtend>,
) {
    if visiting.contains(&concern.name) {
        return;
    }
    if !emitted.insert((expansion_target.target, concern.name, mixin_kind)) {
        return;
    }

    visiting.push(concern.name);
    for dependency in &concern.dependencies {
        let Some(dependency) = resolve_concern(concerns, &dependency.module_name_candidates) else {
            continue;
        };
        expand_concern_class_methods(
            dependency,
            concerns,
            expansion_target,
            mixin_kind,
            visiting,
            emitted,
            out,
        );
    }
    visiting.pop();

    if let Some(class_methods) = concern.class_methods_module {
        out.push(SyntheticClassMethodsExtend {
            source_index: concern.source_index,
            target: expansion_target.target,
            target_kind: expansion_target.target_kind,
            mixin_kind,
            class_methods_module: class_methods.module_name,
            location: class_methods.location,
        });
    }
}

impl ClassMethodsBody {
    fn to_declaration(&self) -> Option<Declaration> {
        if self.members.is_empty() {
            return None;
        }
        Some(Declaration::Module(Arc::new(ModuleDecl {
            module_name: self.module_name,
            name_location: self.name_location,
            members: self.members.clone(),
        })))
    }
}

impl ConcernSiteKind {
    fn body_kind(self) -> ConcernBodyKind {
        match self {
            ConcernSiteKind::Include => ConcernBodyKind::Included,
            ConcernSiteKind::Prepend => ConcernBodyKind::Prepended,
        }
    }

    fn class_methods_mixin_kind(self) -> SyntheticClassMethodsMixinKind {
        match self {
            ConcernSiteKind::Include => SyntheticClassMethodsMixinKind::Extend,
            ConcernSiteKind::Prepend => SyntheticClassMethodsMixinKind::SingletonPrepend,
        }
    }
}

impl ConcernBodyKind {
    fn origin(self) -> InfusionBodyOrigin {
        match self {
            ConcernBodyKind::Included => InfusionBodyOrigin::SyntheticConcernIncluded,
            ConcernBodyKind::Prepended => InfusionBodyOrigin::SyntheticConcernPrepended,
        }
    }
}

impl InfusionBodyOrigin {
    fn synthetic_def_origin(self) -> Option<DefMemberOrigin> {
        match self {
            InfusionBodyOrigin::Real => None,
            InfusionBodyOrigin::SyntheticConcernIncluded => {
                Some(DefMemberOrigin::SyntheticConcernIncluded)
            }
            InfusionBodyOrigin::SyntheticConcerningModule => {
                Some(DefMemberOrigin::SyntheticConcerning)
            }
            InfusionBodyOrigin::SyntheticConcernPrepended => {
                Some(DefMemberOrigin::SyntheticConcernPrepended)
            }
        }
    }
}

impl ConcernDef {
    fn body(&self, kind: ConcernBodyKind) -> Option<&InfusionBody> {
        match kind {
            ConcernBodyKind::Included => self.included_body.as_ref(),
            ConcernBodyKind::Prepended => self.prepended_body.as_ref(),
        }
    }
}

impl SyntheticClassMethodsExtend {
    fn to_declaration(&self, names: &NameTable) -> Option<Declaration> {
        let mixin = MixinMember {
            module_name: names.display_type_name(self.class_methods_module),
            location: self.location,
            name_location: self.location,
            annotation: None,
        };
        let member = match self.mixin_kind {
            SyntheticClassMethodsMixinKind::Extend => Member::Extend(ExtendMember { mixin }),
            SyntheticClassMethodsMixinKind::SingletonPrepend => {
                Member::SingletonPrepend(PrependMember { mixin })
            }
        };
        match self.target_kind {
            InfusionOwnerKind::Class => Some(Declaration::Class(Arc::new(ClassDecl {
                class_name: self.target,
                name_location: self.location,
                super_class: None,
                members: vec![member],
                block_body: false,
            }))),
            InfusionOwnerKind::Module => Some(Declaration::Module(Arc::new(ModuleDecl {
                module_name: self.target,
                name_location: self.location,
                members: vec![member],
            }))),
        }
    }
}

impl<'a> ClassMethodsCollector<'a> {
    fn new(
        names: &'a NameTable,
        source_file: Option<Name>,
        source: &'a [u8],
        file: Option<&'a Path>,
        line_index: &'a LineIndex,
        comments: Option<&'a CommentAssociation>,
    ) -> Self {
        Self {
            names,
            source_file,
            source,
            file,
            line_index,
            comments,
            members: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn unused_annotation_diagnostic(&self, range: PrismByteRange) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::UnusedInlineAnnotation,
            location: Diagnostic::location_for_byte_range(
                self.file.map(|p| p.to_path_buf()).unwrap_or_default(),
                self.source,
                range.0 as usize,
                range.1 as usize,
            ),
        }
    }

    fn report_unused_leading(&mut self, unused: Vec<LeadingAnnotation>) {
        for a in unused {
            let diag = self.unused_annotation_diagnostic(a.location());
            self.diagnostics.push(diag);
        }
    }
}

impl IncludedSingletonClassCollector {
    fn new(origin: DefMemberOrigin, source_file: Option<Name>) -> Self {
        Self {
            members: Vec::new(),
            origin,
            source_file,
        }
    }
}

impl<'pr, 'a> Visit<'pr> for ClassMethodsCollector<'a> {
    fn visit_class_node(&mut self, _node: &ClassNode<'pr>) {}

    fn visit_module_node(&mut self, _node: &ModuleNode<'pr>) {}

    fn visit_def_node(&mut self, node: &DefNode<'pr>) {
        let kind = match node.receiver() {
            Some(receiver) if receiver.as_self_node().is_some() => MethodKind::Singleton,
            Some(_) => return,
            None => MethodKind::Instance,
        };
        let (leading_block, trailing_block) = match self.comments {
            Some(comments) => {
                let def_start_line = self.line_index.line(node.location().start_offset());
                (
                    collect_consecutive_leading(comments, def_start_line),
                    collect_trailing_block(self.source, comments, def_start_line),
                )
            }
            None => (None, None),
        };
        let (method_type, unused_leading, trailing_resolution) = MethodTypeAnnotation::build(
            leading_block.as_ref(),
            trailing_block.as_ref(),
            &[],
            node,
            self.source,
            self.names,
        );
        self.report_unused_leading(unused_leading);
        match trailing_resolution {
            TrailingResolution::ParseFailed {
                range,
                parser_error,
            } => {
                let diag =
                    build_annotation_syntax_error(self.source, self.file, range, parser_error);
                self.diagnostics.push(diag);
            }
            TrailingResolution::Unused(trailing) => {
                let diag = self.unused_annotation_diagnostic(trailing.range());
                self.diagnostics.push(diag);
            }
            TrailingResolution::None => {}
        }
        self.members.push(Member::Def(DefMember {
            visibility: None,
            ivar_param_pairs: Vec::new(),
            name: String::from_utf8_lossy(node.name().as_slice()).to_string(),
            kind,
            location: prism_location_range(node.location()),
            name_location: prism_location_range(node.name_loc()),
            method_type,
            leading_comment: None,
            origin: DefMemberOrigin::Real,
            source_file: None,
        }));
    }

    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if node.receiver().is_some() {
            return;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        let call_kind = match name.as_ref() {
            "attr_reader" => AttributeCallSiteKind::Reader,
            "attr_writer" => AttributeCallSiteKind::Writer,
            "attr_accessor" => AttributeCallSiteKind::Accessor,
            _ => return,
        };
        let Some(arguments) = node.arguments() else {
            return;
        };
        let mut name_nodes = Vec::new();
        for arg in arguments.arguments().iter() {
            let Some(symbol) = arg.as_symbol_node() else {
                continue;
            };
            name_nodes.push(AttributeNameNode {
                name: String::from_utf8_lossy(symbol.unescaped()).to_string(),
                location: prism_location_range(symbol.location()),
            });
        }
        if name_nodes.is_empty() {
            return;
        }
        let trailing = self.comments.and_then(|comments| {
            let end_line = self
                .line_index
                .line(node.location().end_offset().saturating_sub(1));
            comments.trailing_annotation(self.source, end_line)
        });
        let (type_text, annotation_range) = match trailing {
            Some(TrailingAnnotation::NodeTypeAssertion { range, type_text }) => {
                (Some(type_text.to_string()), Some(range))
            }
            Some(TrailingAnnotation::TypeApplication { range, body }) => {
                if let Err(err) = RbsParser::parse_inline_trailing(body.as_bytes()) {
                    let diag = build_annotation_syntax_error(self.source, self.file, range, err);
                    self.diagnostics.push(diag);
                }
                (None, None)
            }
            _ => (None, None),
        };
        let attribute = AttributeMember {
            visibility: None,
            location: prism_location_range(node.location()),
            name_nodes,
            type_text,
            annotation_range,
            source_file: self.source_file,
        };
        let member = match call_kind {
            AttributeCallSiteKind::Reader => Member::AttrReader(AttrReaderMember { attribute }),
            AttributeCallSiteKind::Writer => Member::AttrWriter(AttrWriterMember { attribute }),
            AttributeCallSiteKind::Accessor => {
                Member::AttrAccessor(AttrAccessorMember { attribute })
            }
        };
        self.members.push(member);
    }

    fn visit_singleton_class_node(&mut self, _node: &SingletonClassNode<'pr>) {}
}

impl<'pr> Visit<'pr> for IncludedSingletonClassCollector {
    fn visit_class_node(&mut self, _node: &ClassNode<'pr>) {}

    fn visit_module_node(&mut self, _node: &ModuleNode<'pr>) {}

    fn visit_def_node(&mut self, node: &DefNode<'pr>) {
        if node.receiver().is_some() {
            return;
        }
        push_def_with_origin(
            &mut self.members,
            String::from_utf8_lossy(node.name().as_slice()).to_string(),
            MethodKind::Singleton,
            prism_location_range(node.location()),
            prism_location_range(node.name_loc()),
            self.origin,
            self.source_file,
        );
    }

    fn visit_call_node(&mut self, _node: &CallNode<'pr>) {}

    fn visit_singleton_class_node(&mut self, _node: &SingletonClassNode<'pr>) {}
}

impl<'a> Collector<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        names: &'a NameTable,
        source_index: usize,
        source_file: Option<Name>,
        source: &'a [u8],
        file: Option<&'a Path>,
        parse_result: &ruby_prism::ParseResult<'_>,
        options: InfusionOptions,
        inflector: &'a Inflector,
        inline_mode: bool,
    ) -> Self {
        let line_index = LineIndex::from_source(source);
        let comments =
            inline_mode.then(|| CommentAssociation::from_source(source, &line_index, parse_result));
        Collector {
            names,
            options,
            inflector,
            source_index,
            source_file,
            source,
            file,
            line_index,
            comments,
            class_stack: vec![],
            scope_stack: vec![],
            top_level: vec![],
            active_record_associations: vec![],
            active_record_scopes: vec![],
            active_record_enum_mappings: vec![],
            paranoia_models: vec![],
            diagnostics: vec![],
            concerns: vec![],
            concern_sites: vec![],
        }
    }

    fn current_frame_mut(&mut self) -> Option<&mut ScopeFrame> {
        self.scope_stack.last_mut()
    }

    fn push_member(members: &mut Vec<Member>, m: Member) {
        members.push(m);
    }

    fn attach_declaration(&mut self, decl: Declaration) {
        match self.scope_stack.last_mut() {
            Some(parent) => parent.nested_decls.push(decl),
            None => self.top_level.push(decl),
        }
    }

    fn collect_active_record_associations(&mut self, body: &InfusionBody) {
        self.active_record_associations
            .extend(body.active_record_associations(self.names, self.options));
        self.active_record_scopes
            .extend(body.active_record_scopes(self.options));
        self.active_record_enum_mappings
            .extend(body.active_record_enum_mappings(self.options, self.inflector));
        self.paranoia_models
            .extend(body.paranoia_model(self.options));
    }

    fn current_class_typename(&self) -> Option<TypeName> {
        let abs_path = self.class_stack.last()?;
        Some(self.names.parse_type_name(abs_path))
    }

    fn type_name_candidates_in_current_namespace(&self, raw: &str) -> Vec<TypeName> {
        let name = self.names.parse_type_name(raw);
        if self.names.type_name_is_absolute(name) {
            return vec![name];
        }
        let mut candidates = Vec::new();
        if let Some(owner) = self.current_class_typename() {
            candidates.push(self.names.concat_type_name(owner, name));
            if let Some(parent) = self.names.type_name_parent(owner) {
                let parent_candidate = self.names.concat_type_name(parent, name);
                if parent_candidate != candidates[0] {
                    candidates.push(parent_candidate);
                }
            }
        } else {
            candidates.push(
                self.names
                    .concat_type_name(self.names.absolute_root(), name),
            );
        }
        candidates
    }

    fn find_existing_class_methods_module(
        &self,
        owner: TypeName,
        nested_decls: &[Declaration],
    ) -> Option<ClassMethodsModuleRef> {
        nested_decls.iter().find_map(|decl| {
            let Declaration::Module(module) = decl else {
                return None;
            };
            if self.names.type_name_parent(module.module_name) != Some(owner) {
                return None;
            }
            let segment = self.names.last_segment(module.module_name)?;
            if self.names.resolve(segment) != "ClassMethods" {
                return None;
            }
            Some(ClassMethodsModuleRef {
                module_name: module.module_name,
                location: module.name_location,
            })
        })
    }

    fn collect_existing_class_methods_module(
        &mut self,
        module_name: TypeName,
        name_location: PrismByteRange,
    ) {
        let Some(parent) = self.scope_stack.last_mut() else {
            return;
        };
        if self.names.type_name_parent(module_name) != Some(parent.body.owner) {
            return;
        }
        let Some(segment) = self.names.last_segment(module_name) else {
            return;
        };
        if self.names.resolve(segment) != "ClassMethods" {
            return;
        }
        parent.class_methods_module = Some(ClassMethodsModuleRef {
            module_name,
            location: name_location,
        });
    }

    fn collect_call(&mut self, node: &CallNode<'_>) -> bool {
        if self.collect_concern_call(node) {
            return true;
        }
        if self.collect_synthetic_attr_call(node) {
            return true;
        }
        let Some(frame) = self.current_frame_mut() else {
            return self.is_supported_dsl_call(node);
        };
        if node.receiver().is_some() {
            return false;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        if !matches!(
            name.as_ref(),
            "cattr_reader"
                | "mattr_reader"
                | "cattr_writer"
                | "mattr_writer"
                | "cattr_accessor"
                | "mattr_accessor"
                | "class_attribute"
                | "has_secure_password"
                | "belongs_to"
                | "has_one"
                | "has_many"
                | "enum"
                | "scope"
                | "acts_as_paranoid"
        ) {
            return false;
        }
        let mut symbol_args = Vec::new();
        let mut enum_values = Vec::new();
        let mut keyword_bools = Vec::new();
        let mut keyword_strings = Vec::new();
        let mut scope_lambda_params = None;
        if let Some(arguments) = node.arguments() {
            let args = arguments.arguments();
            if matches!(
                name.as_ref(),
                "has_secure_password" | "belongs_to" | "has_one" | "has_many" | "scope"
            ) {
                let mut iter = args.iter();
                let first_arg = iter.next();
                if name == "has_secure_password"
                    && first_arg.as_ref().is_some_and(|arg| {
                        arg.as_symbol_node().is_none() && arg.as_keyword_hash_node().is_none()
                    })
                {
                    return true;
                }
                if name == "scope" {
                    let Some(body_arg) = iter.next() else {
                        return true;
                    };
                    if body_arg.as_keyword_hash_node().is_some() {
                        return true;
                    }
                    scope_lambda_params = lambda_scope_params(&body_arg);
                }
                if let Some(arg) = first_arg
                    && let Some(sym) = arg.as_symbol_node()
                {
                    symbol_args.push(InfusionSymbolArg {
                        name: String::from_utf8_lossy(sym.unescaped()).to_string(),
                        location: prism_location_range(sym.location()),
                    });
                }
                for arg in iter {
                    collect_keyword_bools(&arg, &mut keyword_bools);
                    collect_keyword_strings(&arg, &mut keyword_strings);
                }
                frame.body.calls.push(InfusionCall {
                    name: name.to_string(),
                    symbol_args,
                    enum_values,
                    keyword_bools,
                    keyword_strings,
                    scope_lambda_params: scope_lambda_params.take(),
                    location: prism_location_range(node.location()),
                });
                return true;
            }
            if name == "enum" {
                let mut iter = args.iter();
                let Some(first_arg) = iter.next() else {
                    return true;
                };
                let Some(name_arg) = first_arg.as_symbol_node() else {
                    return true;
                };
                symbol_args.push(InfusionSymbolArg {
                    name: String::from_utf8_lossy(name_arg.unescaped()).to_string(),
                    location: prism_location_range(name_arg.location()),
                });
                if let Some(values_arg) = iter.next() {
                    collect_enum_values(&values_arg, &mut enum_values);
                    collect_keyword_bools(&values_arg, &mut keyword_bools);
                    collect_keyword_strings(&values_arg, &mut keyword_strings);
                }
                for arg in iter {
                    collect_keyword_bools(&arg, &mut keyword_bools);
                    collect_keyword_strings(&arg, &mut keyword_strings);
                }
                frame.body.calls.push(InfusionCall {
                    name: name.to_string(),
                    symbol_args,
                    enum_values,
                    keyword_bools,
                    keyword_strings,
                    scope_lambda_params: scope_lambda_params.take(),
                    location: prism_location_range(node.location()),
                });
                return true;
            }
            for arg in args.iter() {
                if let Some(sym) = arg.as_symbol_node() {
                    symbol_args.push(InfusionSymbolArg {
                        name: String::from_utf8_lossy(sym.unescaped()).to_string(),
                        location: prism_location_range(sym.location()),
                    });
                    continue;
                }
                collect_keyword_bools(&arg, &mut keyword_bools);
                collect_keyword_strings(&arg, &mut keyword_strings);
            }
        }
        frame.body.calls.push(InfusionCall {
            name: name.to_string(),
            symbol_args,
            enum_values,
            keyword_bools,
            keyword_strings,
            scope_lambda_params: scope_lambda_params.take(),
            location: prism_location_range(node.location()),
        });
        true
    }

    fn collect_synthetic_attr_call(&mut self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }
        // Hoist `source_file` read before the frame borrow so both
        // `self.source_file` and `frame` can coexist without a second
        // `&mut self` — the `AttributeMember` below needs the former
        // while `frame` still holds the latter.
        let source_file = self.source_file;
        let Some(frame) = self.current_frame_mut() else {
            return false;
        };
        if frame.body.origin.synthetic_def_origin().is_none() {
            return false;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        let member_kind = match name.as_ref() {
            "attr_reader" => AttributeCallSiteKind::Reader,
            "attr_writer" => AttributeCallSiteKind::Writer,
            "attr_accessor" => AttributeCallSiteKind::Accessor,
            _ => return false,
        };
        let Some(arguments) = node.arguments() else {
            return true;
        };
        let mut name_nodes = Vec::new();
        for arg in arguments.arguments().iter() {
            let Some(symbol) = arg.as_symbol_node() else {
                continue;
            };
            name_nodes.push(AttributeNameNode {
                name: String::from_utf8_lossy(symbol.unescaped()).to_string(),
                location: prism_location_range(symbol.location()),
            });
        }
        if name_nodes.is_empty() {
            return true;
        }
        let attribute = AttributeMember {
            visibility: None,
            location: prism_location_range(node.location()),
            name_nodes,
            type_text: None,
            annotation_range: None,
            source_file,
        };
        let member = match member_kind {
            AttributeCallSiteKind::Reader => Member::AttrReader(AttrReaderMember { attribute }),
            AttributeCallSiteKind::Writer => Member::AttrWriter(AttrWriterMember { attribute }),
            AttributeCallSiteKind::Accessor => {
                Member::AttrAccessor(AttrAccessorMember { attribute })
            }
        };
        frame.body.members.push(member);
        true
    }

    fn is_supported_dsl_call(&self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        matches!(
            name.as_ref(),
            "cattr_reader"
                | "mattr_reader"
                | "cattr_writer"
                | "mattr_writer"
                | "cattr_accessor"
                | "mattr_accessor"
                | "class_attribute"
                | "has_secure_password"
                | "belongs_to"
                | "has_one"
                | "has_many"
                | "enum"
                | "scope"
                | "acts_as_paranoid"
        )
    }

    fn collect_concern_call(&mut self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        match name.as_ref() {
            "extend" => self.collect_extend_concern(node),
            "include" => self.collect_concern_site(node, ConcernSiteKind::Include),
            "prepend" => self.collect_concern_site(node, ConcernSiteKind::Prepend),
            "included" => self.collect_concern_block(node, ConcernBodyKind::Included),
            "prepended" => self.collect_concern_block(node, ConcernBodyKind::Prepended),
            "class_methods" => self.collect_class_methods_block(node),
            // Unlike the arms above, `concerning` only claims the call
            // when the argument shape is one it can synthesize from
            // (literal topic + block); unrecognized shapes fall back to
            // the caller's general block descent so `def`s inside are
            // still collected onto the enclosing owner.
            "concerning" => return self.collect_concerning_call(node),
            _ => return false,
        }
        true
    }

    fn collect_extend_concern(&mut self, node: &CallNode<'_>) {
        let Some(frame) = self.current_frame_mut() else {
            return;
        };
        let Some(arguments) = node.arguments() else {
            return;
        };
        if arguments.arguments().len() != 1 {
            return;
        }
        let Some(arg) = arguments.arguments().iter().next() else {
            return;
        };
        if matches!(
            constant_node_to_string(&arg).as_deref(),
            Some("ActiveSupport::Concern" | "::ActiveSupport::Concern")
        ) {
            frame.extends_concern = true;
        }
    }

    fn collect_concern_site(&mut self, node: &CallNode<'_>, kind: ConcernSiteKind) {
        let Some(frame) = self.scope_stack.last() else {
            return;
        };
        let Some(arguments) = node.arguments() else {
            return;
        };
        if arguments.arguments().len() != 1 {
            return;
        }
        let Some(arg) = arguments.arguments().iter().next() else {
            return;
        };
        let Some(raw_module_name) = constant_node_to_string(&arg) else {
            return;
        };
        let module_names = self.type_name_candidates_in_current_namespace(&raw_module_name);
        self.concern_sites.push(ConcernSite {
            target: frame.body.owner,
            target_kind: frame.body.owner_kind,
            kind,
            module_names: module_names.clone(),
        });
        if let Some(frame) = self.current_frame_mut() {
            let dependency = ConcernDependency {
                module_name_candidates: module_names,
            };
            match kind {
                ConcernSiteKind::Include => frame.dependencies.push(dependency),
                ConcernSiteKind::Prepend => frame.dependencies.insert(0, dependency),
            }
        }
    }

    fn collect_concern_block(&mut self, node: &CallNode<'_>, body_kind: ConcernBodyKind) {
        let Some(block_node) = node.block() else {
            return;
        };
        let Some(block) = block_node.as_block_node() else {
            return;
        };
        let Some(block_body) = block.body() else {
            return;
        };
        let Some(parent) = self.scope_stack.last() else {
            return;
        };
        let synthetic_body = InfusionBody {
            owner: parent.body.owner,
            owner_kind: parent.body.owner_kind,
            name_location: prism_location_range(block.location()),
            calls: Vec::new(),
            members: Vec::new(),
            origin: body_kind.origin(),
        };
        self.scope_stack.push(ScopeFrame {
            body: synthetic_body,
            nested_decls: Vec::new(),
            dependencies: Vec::new(),
            extends_concern: false,
            class_methods_module: None,
            included_body: None,
            prepended_body: None,
            class_methods: None,
        });
        self.visit(&block_body);
        let frame = self.scope_stack.pop().unwrap();
        if let Some(parent) = self.current_frame_mut() {
            match body_kind {
                ConcernBodyKind::Included if parent.included_body.is_none() => {
                    parent.included_body = Some(frame.body);
                }
                ConcernBodyKind::Prepended if parent.prepended_body.is_none() => {
                    parent.prepended_body = Some(frame.body);
                }
                _ => {}
            }
        }
    }

    fn collect_class_methods_block(&mut self, node: &CallNode<'_>) {
        let Some(block_node) = node.block() else {
            return;
        };
        let Some(block) = block_node.as_block_node() else {
            return;
        };
        let Some(block_body) = block.body() else {
            return;
        };
        let Some(parent) = self.scope_stack.last() else {
            return;
        };
        let class_methods_segment = self.names.intern_symbol("ClassMethods");
        let module_name = self
            .names
            .append_type_name(parent.body.owner, class_methods_segment);
        let mut collector = ClassMethodsCollector::new(
            self.names,
            self.source_file,
            self.source,
            self.file,
            &self.line_index,
            self.comments.as_ref(),
        );
        collector.visit(&block_body);
        let members = collector.members;
        let diagnostics = collector.diagnostics;
        self.diagnostics.extend(diagnostics);
        if members.is_empty() {
            return;
        }
        if let Some(parent) = self.current_frame_mut() {
            parent.class_methods_module = Some(ClassMethodsModuleRef {
                module_name,
                location: prism_location_range(block.location()),
            });
            let class_methods = parent
                .class_methods
                .get_or_insert_with(|| ClassMethodsBody {
                    module_name,
                    name_location: prism_location_range(block.location()),
                    members: Vec::new(),
                });
            class_methods.members.extend(members);
        }
    }

    /// `concerning :Topic do ... end` — Rails sugar (F1 in the ready todo)
    /// for `const_set :Topic, Module.new { extend ActiveSupport::Concern;
    /// module_eval(&block) }` followed by `include` (or `prepend`, with
    /// `prepend: true`) on the enclosing owner. Synthesizes the
    /// `<owner>::<Topic>` module declaration directly, registers it as a
    /// `ConcernDef` so nested `included do` / `class_methods do` reuse the
    /// existing Concern expansion pipeline, and registers a `ConcernSite`
    /// on the owner so the module actually gets included/prepended.
    ///
    /// The module deliberately carries no `ModuleSelf(owner)` member even
    /// though Ruby's `concerning` guarantees the owner is the include
    /// target: `self_types` participate in RBS ancestor-chain
    /// construction, so "self-typed as the owner AND included by the
    /// owner" validates as `RBS::RecursiveAncestorError` in crema, rbs,
    /// and Steep alike (F12 in the todo). Owner-context visibility for
    /// the block's `def`s needs no declaration-side edge at all — their
    /// source nodes sit lexically inside the owner's class body, so the
    /// checker already checks them in the owner's instance context.
    ///
    /// Returns whether the call was claimed: `false` for shapes it cannot
    /// synthesize from (dynamic topic, no block), letting the caller's
    /// general block descent collect the block's contents onto the
    /// enclosing owner instead of dropping them.
    fn collect_concerning_call(&mut self, node: &CallNode<'_>) -> bool {
        let Some((topic_name, topic_location, prepend)) = concerning_topic_and_prepend(node) else {
            return false;
        };
        let Some(block_node) = node.block() else {
            return false;
        };
        let Some(block) = block_node.as_block_node() else {
            return false;
        };
        // No enclosing class/module frame (top-level `concerning`, which
        // Ruby itself rejects at runtime): don't claim, so the general
        // descent still collects the block's contents somewhere instead
        // of dropping them entirely.
        let Some(parent) = self.scope_stack.last() else {
            return false;
        };
        let owner = parent.body.owner;
        let owner_kind = parent.body.owner_kind;
        let topic_segment = self.names.intern_symbol(&topic_name);
        let module_name = self.names.append_type_name(owner, topic_segment);

        self.scope_stack.push(ScopeFrame {
            body: InfusionBody {
                owner: module_name,
                owner_kind: InfusionOwnerKind::Module,
                name_location: topic_location,
                calls: Vec::new(),
                members: Vec::new(),
                origin: InfusionBodyOrigin::SyntheticConcerningModule,
            },
            nested_decls: Vec::new(),
            dependencies: Vec::new(),
            extends_concern: false,
            class_methods_module: None,
            included_body: None,
            prepended_body: None,
            class_methods: None,
        });
        // A zero-statement block parses as `body: None`; Rails still
        // const_sets and includes the module for `concerning :C do end`,
        // so synthesis must not depend on the body's presence.
        if let Some(block_body) = block.body() {
            self.visit(&block_body);
        }
        let frame = self.scope_stack.pop().unwrap();

        let class_methods_module = frame
            .class_methods_module
            .or_else(|| self.find_existing_class_methods_module(module_name, &frame.nested_decls));
        self.concerns.push(ConcernDef {
            name: module_name,
            source_index: self.source_index,
            dependencies: frame.dependencies.clone(),
            class_methods_module,
            included_body: frame.included_body.clone(),
            prepended_body: frame.prepended_body.clone(),
            class_methods: frame.class_methods.clone(),
        });
        self.collect_active_record_associations(&frame.body);

        // Extend is unconditional (F1: every `concerning` module extends
        // Concern), unlike `apply_body`'s natural members which are
        // dropped entirely when empty — an empty `concerning :C do end`
        // still needs it.
        let mixin = MixinMember {
            module_name: "::ActiveSupport::Concern".to_string(),
            location: topic_location,
            name_location: topic_location,
            annotation: None,
        };
        let mut members = vec![Member::Extend(ExtendMember { mixin })];
        for call in &frame.body.calls {
            if self.options.activesupport {
                activesupport::collect_call(&mut members, call);
            }
            if self.options.activemodel {
                activemodel::collect_call(&mut members, call);
            }
            if self.options.activerecord {
                activerecord::collect_call(
                    &mut members,
                    call,
                    self.inflector,
                    self.names,
                    frame.body.owner,
                );
            }
        }
        members.extend(frame.body.members.clone());
        members.extend(frame.nested_decls.into_iter().map(Member::Declaration));

        self.attach_declaration(Declaration::Module(Arc::new(ModuleDecl {
            module_name,
            name_location: topic_location,
            members,
        })));

        let kind = if prepend {
            ConcernSiteKind::Prepend
        } else {
            ConcernSiteKind::Include
        };
        self.concern_sites.push(ConcernSite {
            target: owner,
            target_kind: owner_kind,
            kind,
            module_names: vec![module_name],
        });
        // `ConcernSite`/`ConcernDependency` below only drive the
        // included-do/prepended-do body-expansion bookkeeping
        // (`expand_concern_bodies`) — for a *literal* `include Foo`, that
        // relationship is redundant with the real `Member::Include`
        // inline_parser.rs already attached from the source text.
        // `concerning` has no such literal statement, so the actual
        // mixin relationship (needed for ordinary ancestor-chain lookup
        // of `def foo_bar` etc.) has to be synthesized here too.
        let owner_mixin = MixinMember {
            module_name: self.names.display_type_name(module_name),
            location: topic_location,
            name_location: topic_location,
            annotation: None,
        };
        let owner_member = match kind {
            ConcernSiteKind::Include => Member::Include(IncludeMember { mixin: owner_mixin }),
            ConcernSiteKind::Prepend => Member::Prepend(PrependMember { mixin: owner_mixin }),
        };
        if let Some(parent) = self.current_frame_mut() {
            let dependency = ConcernDependency {
                module_name_candidates: vec![module_name],
            };
            match kind {
                ConcernSiteKind::Include => parent.dependencies.push(dependency),
                ConcernSiteKind::Prepend => parent.dependencies.insert(0, dependency),
            }
            parent.body.members.push(owner_member);
        }
        true
    }

    fn apply_body(
        body: InfusionBody,
        nested_decls: Vec<Declaration>,
        options: InfusionOptions,
        inflector: &Inflector,
        names: &NameTable,
    ) -> Option<Declaration> {
        let mut members = Vec::new();
        debug_assert!(matches!(
            body.origin,
            InfusionBodyOrigin::Real
                | InfusionBodyOrigin::SyntheticConcernIncluded
                | InfusionBodyOrigin::SyntheticConcernPrepended
        ));
        for call in &body.calls {
            if options.activesupport {
                activesupport::collect_call(&mut members, call);
            }
            if options.activemodel {
                activemodel::collect_call(&mut members, call);
            }
            if options.activerecord {
                activerecord::collect_call(&mut members, call, inflector, names, body.owner);
            }
        }
        members.extend(body.members);
        members.extend(nested_decls.into_iter().map(Member::Declaration));
        if members.is_empty() {
            return None;
        }
        match body.owner_kind {
            InfusionOwnerKind::Class => Some(Declaration::Class(Arc::new(ClassDecl {
                class_name: body.owner,
                name_location: body.name_location,
                super_class: None,
                members,
                block_body: false,
            }))),
            InfusionOwnerKind::Module => Some(Declaration::Module(Arc::new(ModuleDecl {
                module_name: body.owner,
                name_location: body.name_location,
                members,
            }))),
        }
    }
}

/// Extracts the parameter structure of a `scope` body when it is a `->`
/// literal. `None` means "fall back to `(?)`": non-lambda bodies, plus
/// literal shapes whose proc-parameter mapping is not representable here
/// (destructuring `->((a, b))`, implicit rest `->(a,)`, `...` forwarding).
/// Numbered (`_1`) and `it` params map to required positionals, matching
/// `lambda#parameters` (`[[:req, :_1]]`).
fn lambda_scope_params(node: &Node<'_>) -> Option<InfusionScopeParams> {
    let lambda = node.as_lambda_node()?;
    let mut out = InfusionScopeParams::default();
    let Some(params_node) = lambda.parameters() else {
        return Some(out);
    };
    if let Some(numbered) = params_node.as_numbered_parameters_node() {
        for i in 1..=numbered.maximum() {
            out.requireds.push(format!("_{i}"));
        }
        return Some(out);
    }
    if params_node.as_it_parameters_node().is_some() {
        out.requireds.push("it".to_string());
        return Some(out);
    }
    let block_params = params_node.as_block_parameters_node()?;
    let Some(params) = block_params.parameters() else {
        return Some(out);
    };
    for param in params.requireds().iter() {
        let required = param.as_required_parameter_node()?;
        out.requireds
            .push(String::from_utf8_lossy(required.name().as_slice()).to_string());
    }
    for param in params.optionals().iter() {
        let optional = param.as_optional_parameter_node()?;
        out.optionals
            .push(String::from_utf8_lossy(optional.name().as_slice()).to_string());
    }
    if let Some(rest) = params.rest() {
        rest.as_rest_parameter_node()?;
        out.rest = true;
    }
    for param in params.posts().iter() {
        let post = param.as_required_parameter_node()?;
        out.trailings
            .push(String::from_utf8_lossy(post.name().as_slice()).to_string());
    }
    for param in params.keywords().iter() {
        if let Some(keyword) = param.as_required_keyword_parameter_node() {
            out.required_keywords
                .push(String::from_utf8_lossy(keyword.name().as_slice()).to_string());
        } else if let Some(keyword) = param.as_optional_keyword_parameter_node() {
            out.optional_keywords
                .push(String::from_utf8_lossy(keyword.name().as_slice()).to_string());
        } else {
            return None;
        }
    }
    if let Some(keyword_rest) = params.keyword_rest() {
        if keyword_rest.as_keyword_rest_parameter_node().is_some() {
            out.keyrest = true;
        } else if keyword_rest.as_no_keywords_parameter_node().is_none() {
            return None;
        }
    }
    if params.block().is_some() {
        out.block = true;
    }
    Some(out)
}

pub(crate) fn push_def(
    members: &mut Vec<Member>,
    name: String,
    kind: MethodKind,
    location: PrismByteRange,
    name_location: PrismByteRange,
) {
    push_def_with_origin(
        members,
        name,
        kind,
        location,
        name_location,
        DefMemberOrigin::Real,
        None,
    );
}

fn push_def_with_origin(
    members: &mut Vec<Member>,
    name: String,
    kind: MethodKind,
    location: PrismByteRange,
    name_location: PrismByteRange,
    origin: DefMemberOrigin,
    source_file: Option<Name>,
) {
    Collector::push_member(
        members,
        Member::Def(DefMember {
            visibility: None,
            ivar_param_pairs: Vec::new(),
            name,
            kind,
            location,
            name_location,
            method_type: MethodTypeAnnotation {
                type_annotations: TypeAnnotations::None,
            },
            leading_comment: None,
            origin,
            source_file,
        }),
    );
}

/// Parses `concerning`'s argument list: a required literal topic name
/// (symbol or string, see [`literal_topic_name`]) followed by an
/// optional `prepend:` keyword. Returns `None` for any shape that isn't
/// at least `concerning :Topic` (dynamic topic, missing args, etc.) —
/// mirrors the other DSL-arg parsers in this file, which silently skip
/// forms they don't recognize rather than diagnosing them. A `prepend:`
/// value that isn't the literal `true` (e.g. a dynamic expression) is
/// treated as `false` — an accepted approximation, same default-on-
/// unrecognized-shape stance as the rest of this file's keyword parsing.
pub(crate) fn concerning_topic_and_prepend(
    node: &CallNode<'_>,
) -> Option<(String, PrismByteRange, bool)> {
    let arguments = node.arguments()?;
    let mut iter = arguments.arguments().iter();
    let topic_arg = iter.next()?;
    let (topic_name, topic_location) = literal_topic_name(&topic_arg)?;
    let mut prepend = false;
    for arg in iter {
        let Some(kw_hash) = arg.as_keyword_hash_node() else {
            continue;
        };
        for elem in kw_hash.elements().iter() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            let Some(key) = assoc.key().as_symbol_node() else {
                continue;
            };
            if key.unescaped() == b"prepend" {
                prepend = assoc.value().as_true_node().is_some();
            }
        }
    }
    Some((topic_name, topic_location, prepend))
}

/// A `concerning`/`concern` topic that is statically known: a symbol or
/// string literal (Rails passes either straight to `const_set`).
/// Interpolated forms parse as distinct node types and fall out as
/// `None`. `inline_parser::is_infusion_owned_block_call` gates its
/// block-descent stop on the same shapes — keep the two in sync, or
/// `def`s inside an unclaimed `concerning` block vanish from both the
/// owner and the (never-synthesized) module.
fn literal_topic_name(node: &Node<'_>) -> Option<(String, PrismByteRange)> {
    if let Some(sym) = node.as_symbol_node() {
        return Some((
            String::from_utf8_lossy(sym.unescaped()).to_string(),
            prism_location_range(sym.location()),
        ));
    }
    let string = node.as_string_node()?;
    Some((
        String::from_utf8_lossy(string.unescaped()).to_string(),
        prism_location_range(string.location()),
    ))
}

fn collect_keyword_bools(node: &Node<'_>, out: &mut Vec<InfusionKeywordBool>) {
    let Some(kw_hash) = node.as_keyword_hash_node() else {
        return;
    };
    for elem in kw_hash.elements().iter() {
        let Some(assoc) = elem.as_assoc_node() else {
            continue;
        };
        let Some(sym) = assoc.key().as_symbol_node() else {
            continue;
        };
        let value = assoc.value();
        let value_bool = if value.as_true_node().is_some() {
            Some(true)
        } else if value.as_false_node().is_some() {
            Some(false)
        } else {
            None
        };
        out.push(InfusionKeywordBool {
            name: String::from_utf8_lossy(sym.unescaped()).to_string(),
            value: value_bool,
        });
    }
}

fn collect_keyword_strings(node: &Node<'_>, out: &mut Vec<InfusionKeywordString>) {
    let Some(kw_hash) = node.as_keyword_hash_node() else {
        return;
    };
    for elem in kw_hash.elements().iter() {
        let Some(assoc) = elem.as_assoc_node() else {
            continue;
        };
        let Some(sym) = assoc.key().as_symbol_node() else {
            continue;
        };
        let value = assoc.value();
        let value = if let Some(sym) = value.as_symbol_node() {
            String::from_utf8_lossy(sym.unescaped()).to_string()
        } else if let Some(string) = value.as_string_node() {
            String::from_utf8_lossy(string.unescaped()).to_string()
        } else {
            continue;
        };
        out.push(InfusionKeywordString {
            name: String::from_utf8_lossy(sym.unescaped()).to_string(),
            value,
        });
    }
}

fn collect_enum_values(node: &Node<'_>, out: &mut Vec<InfusionEnumValue>) {
    if let Some(array) = node.as_array_node() {
        // Rails assigns array labels their index (`each_with_index`), so
        // the value literal is the position, same as orthoses enum.rb.
        for (index, elem) in array.elements().iter().enumerate() {
            let Some(sym) = elem.as_symbol_node() else {
                continue;
            };
            out.push(InfusionEnumValue {
                name: String::from_utf8_lossy(sym.unescaped()).to_string(),
                location: prism_location_range(sym.location()),
                value: Some(EnumLiteral::Int(index.to_string())),
            });
        }
    } else if let Some(hash) = node.as_hash_node() {
        for elem in hash.elements().iter() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            let Some(sym) = assoc.key().as_symbol_node() else {
                continue;
            };
            out.push(InfusionEnumValue {
                name: String::from_utf8_lossy(sym.unescaped()).to_string(),
                location: prism_location_range(sym.location()),
                value: enum_value_literal(&assoc.value()),
            });
        }
    } else if let Some(kw_hash) = node.as_keyword_hash_node() {
        for elem in kw_hash.elements().iter() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            let Some(sym) = assoc.key().as_symbol_node() else {
                continue;
            };
            let name = String::from_utf8_lossy(sym.unescaped()).to_string();
            if is_enum_option_name(&name) {
                continue;
            }
            out.push(InfusionEnumValue {
                name,
                location: prism_location_range(sym.location()),
                value: enum_value_literal(&assoc.value()),
            });
        }
    }
}

/// The value side of one `enum` hash pair when it is a plain literal.
/// Integers outside u32 (or negative) are treated as non-literal — no
/// real enum column carries them, and the fallback (`untyped` value
/// surface) stays sound.
fn enum_value_literal(node: &Node<'_>) -> Option<EnumLiteral> {
    if let Some(int_node) = node.as_integer_node() {
        let integer = int_node.value();
        let (negative, digits) = integer.to_u32_digits();
        if negative || digits.len() > 1 {
            return None;
        }
        return Some(EnumLiteral::Int(
            digits.first().copied().unwrap_or(0).to_string(),
        ));
    }
    node.as_string_node()
        .map(|s| EnumLiteral::Str(String::from_utf8_lossy(s.unescaped()).to_string()))
}

fn is_enum_option_name(name: &str) -> bool {
    matches!(
        name,
        "prefix" | "suffix" | "scopes" | "default" | "instance_methods" | "validate"
    )
}

impl InfusionBody {
    fn synthetic_for(
        &self,
        owner: TypeName,
        owner_kind: InfusionOwnerKind,
        origin: InfusionBodyOrigin,
    ) -> Self {
        InfusionBody {
            owner,
            owner_kind,
            name_location: self.name_location,
            calls: self.calls.clone(),
            members: self.members.clone(),
            origin,
        }
    }
}

fn constant_node_to_string(node: &Node<'_>) -> Option<String> {
    if let Some(constant) = node.as_constant_read_node() {
        Some(String::from_utf8_lossy(constant.name().as_slice()).to_string())
    } else {
        node.as_constant_path_node()
            .and_then(|path| constant_path_to_string(&path))
    }
}

fn constant_path_to_string(node: &ConstantPathNode<'_>) -> Option<String> {
    let child_name = String::from_utf8_lossy(node.name()?.as_slice()).to_string();
    match node.parent() {
        Some(parent) => {
            if let Some(constant) = parent.as_constant_read_node() {
                let parent_name = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                Some(format!("{}::{}", parent_name, child_name))
            } else if let Some(path) = parent.as_constant_path_node() {
                Some(format!(
                    "{}::{}",
                    constant_path_to_string(&path)?,
                    child_name
                ))
            } else {
                None
            }
        }
        None => Some(format!("::{}", child_name)),
    }
}

impl<'pr, 'a> Visit<'pr> for Collector<'a> {
    fn visit_class_node(&mut self, node: &ClassNode<'pr>) {
        if push_class_abs_path(&mut self.class_stack, &node.constant_path()).is_none() {
            // `NonConstantClassName` is owned by the inline parser; stay
            // silent here so a single source line never gets two
            // copies of the same syntax diagnostic.
            return;
        }
        let class_name = self.current_class_typename().unwrap();
        let name_location =
            crate::inline_parser::prism_location_range(node.constant_path().location());
        self.scope_stack.push(ScopeFrame {
            body: InfusionBody {
                owner: class_name,
                owner_kind: InfusionOwnerKind::Class,
                name_location,
                calls: Vec::new(),
                members: Vec::new(),
                origin: InfusionBodyOrigin::Real,
            },
            nested_decls: Vec::new(),
            dependencies: Vec::new(),
            extends_concern: false,
            class_methods_module: None,
            included_body: None,
            prepended_body: None,
            class_methods: None,
        });
        ruby_prism::visit_class_node(self, node);
        let frame = self.scope_stack.pop().unwrap();
        self.class_stack.pop();
        if frame.extends_concern {
            let class_methods_module = frame.class_methods_module.or_else(|| {
                self.find_existing_class_methods_module(frame.body.owner, &frame.nested_decls)
            });
            self.concerns.push(ConcernDef {
                name: frame.body.owner,
                source_index: self.source_index,
                dependencies: frame.dependencies.clone(),
                class_methods_module,
                included_body: frame.included_body.clone(),
                prepended_body: frame.prepended_body.clone(),
                class_methods: frame.class_methods.clone(),
            });
        }
        self.collect_active_record_associations(&frame.body);
        let Some(decl) = Self::apply_body(
            frame.body,
            frame.nested_decls,
            self.options,
            self.inflector,
            self.names,
        ) else {
            // No synthesized methods, no nested decls — nothing to
            // contribute. Suppress the empty re-declaration so a plain
            // `class Foo; end` (without any DSL call) does not get
            // phantom-registered through the DSL pass. The build layer
            // would merge an empty shell harmlessly, but in
            // `inline = false` + no-sig setups it would still introduce
            // a class entry the user never asked the DSL pass to add.
            return;
        };
        self.attach_declaration(decl);
    }

    fn visit_module_node(&mut self, node: &ModuleNode<'pr>) {
        if push_class_abs_path(&mut self.class_stack, &node.constant_path()).is_none() {
            return;
        }
        let module_name = self.current_class_typename().unwrap();
        let name_location =
            crate::inline_parser::prism_location_range(node.constant_path().location());
        self.collect_existing_class_methods_module(module_name, name_location);
        self.scope_stack.push(ScopeFrame {
            body: InfusionBody {
                owner: module_name,
                owner_kind: InfusionOwnerKind::Module,
                name_location,
                calls: Vec::new(),
                members: Vec::new(),
                origin: InfusionBodyOrigin::Real,
            },
            nested_decls: Vec::new(),
            dependencies: Vec::new(),
            extends_concern: false,
            class_methods_module: None,
            included_body: None,
            prepended_body: None,
            class_methods: None,
        });
        ruby_prism::visit_module_node(self, node);
        let frame = self.scope_stack.pop().unwrap();
        self.class_stack.pop();
        if frame.extends_concern {
            let class_methods_module = frame.class_methods_module.or_else(|| {
                self.find_existing_class_methods_module(frame.body.owner, &frame.nested_decls)
            });
            self.concerns.push(ConcernDef {
                name: frame.body.owner,
                source_index: self.source_index,
                dependencies: frame.dependencies.clone(),
                class_methods_module,
                included_body: frame.included_body.clone(),
                prepended_body: frame.prepended_body.clone(),
                class_methods: frame.class_methods.clone(),
            });
        }
        self.collect_active_record_associations(&frame.body);
        let Some(decl) = Self::apply_body(
            frame.body,
            frame.nested_decls,
            self.options,
            self.inflector,
            self.names,
        ) else {
            return;
        };
        self.attach_declaration(decl);
    }

    fn visit_singleton_class_node(&mut self, node: &SingletonClassNode<'pr>) {
        let source_file = self.source_file;
        let Some(frame) = self.current_frame_mut() else {
            return;
        };
        let Some(origin) = frame.body.origin.synthetic_def_origin() else {
            return;
        };
        if node.expression().as_self_node().is_none() {
            return;
        }
        let Some(body) = node.body() else {
            return;
        };
        let mut collector = IncludedSingletonClassCollector::new(origin, source_file);
        collector.visit(&body);
        frame.body.members.extend(collector.members);
    }

    fn visit_def_node(&mut self, node: &DefNode<'pr>) {
        let source_file = self.source_file;
        if let Some(frame) = self.current_frame_mut()
            && let Some(origin) = frame.body.origin.synthetic_def_origin()
            && (node.receiver().is_none()
                || node
                    .receiver()
                    .is_some_and(|receiver| receiver.as_self_node().is_some()))
        {
            let kind = if node.receiver().is_some() {
                MethodKind::Singleton
            } else {
                MethodKind::Instance
            };
            push_def_with_origin(
                &mut frame.body.members,
                String::from_utf8_lossy(node.name().as_slice()).to_string(),
                kind,
                prism_location_range(node.location()),
                prism_location_range(node.name_loc()),
                origin,
                source_file,
            );
        }
        // Method bodies run at call time, not while the class/module body
        // is evaluated, so body-level DSL directives are not collected here.
    }

    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if !self.collect_call(node) {
            ruby_prism::visit_call_node(self, node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_body(owner: TypeName, attr: &str, origin: InfusionBodyOrigin) -> InfusionBody {
        InfusionBody {
            owner,
            owner_kind: InfusionOwnerKind::Module,
            name_location: (0, 0),
            calls: vec![InfusionCall {
                name: "cattr_accessor".to_string(),
                symbol_args: vec![InfusionSymbolArg {
                    name: attr.to_string(),
                    location: (0, 0),
                }],
                enum_values: Vec::new(),
                keyword_bools: Vec::new(),
                keyword_strings: Vec::new(),
                scope_lambda_params: None,
                location: (0, 0),
            }],
            members: Vec::new(),
            origin,
        }
    }

    fn test_class_methods(names: &NameTable, owner: TypeName, method: &str) -> ClassMethodsBody {
        ClassMethodsBody {
            module_name: names.append_type_name(owner, names.intern_symbol("ClassMethods")),
            name_location: (0, 0),
            members: vec![Member::Def(DefMember {
                visibility: None,
                ivar_param_pairs: Vec::new(),
                name: method.to_string(),
                kind: MethodKind::Instance,
                location: (0, 0),
                name_location: (0, 0),
                method_type: MethodTypeAnnotation {
                    type_annotations: TypeAnnotations::Array(Vec::new()),
                },
                leading_comment: None,
                origin: DefMemberOrigin::Real,
                source_file: None,
            })],
        }
    }

    #[test]
    fn expand_concern_included_bodies_orders_dependencies_before_owner() {
        let names = NameTable::new();
        let timestamped_name = names.parse_type_name("::Timestamped");
        let auditable_name = names.parse_type_name("::Auditable");
        let user_name = names.parse_type_name("::User");
        let timestamped = ConcernDef {
            name: timestamped_name,
            source_index: 0,
            dependencies: Vec::new(),
            class_methods_module: None,
            included_body: Some(test_body(
                timestamped_name,
                "timestamped",
                InfusionBodyOrigin::SyntheticConcernIncluded,
            )),
            prepended_body: None,
            class_methods: None,
        };
        let auditable = ConcernDef {
            name: auditable_name,
            source_index: 1,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![timestamped_name],
            }],
            class_methods_module: None,
            included_body: Some(test_body(
                auditable_name,
                "audited",
                InfusionBodyOrigin::SyntheticConcernIncluded,
            )),
            prepended_body: None,
            class_methods: None,
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(timestamped_name, &timestamped);
        concerns.insert(auditable_name, &auditable);

        let mut out = Vec::new();
        expand_concern_bodies(
            &auditable,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            ConcernBodyKind::Included,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let emitted: Vec<_> = out
            .iter()
            .map(|synthetic| synthetic.body.calls[0].symbol_args[0].name.as_str())
            .collect();
        assert_eq!(emitted, vec!["timestamped", "audited"]);
        assert_eq!(out[0].body.owner, user_name);
        assert_eq!(out[1].body.owner, user_name);
    }

    #[test]
    fn expand_concern_included_bodies_skips_cycles_but_keeps_each_body_once() {
        let names = NameTable::new();
        let a_name = names.parse_type_name("::A");
        let b_name = names.parse_type_name("::B");
        let user_name = names.parse_type_name("::User");
        let a = ConcernDef {
            name: a_name,
            source_index: 0,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![b_name],
            }],
            class_methods_module: None,
            included_body: Some(test_body(
                a_name,
                "a_attr",
                InfusionBodyOrigin::SyntheticConcernIncluded,
            )),
            prepended_body: None,
            class_methods: None,
        };
        let b = ConcernDef {
            name: b_name,
            source_index: 1,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![a_name],
            }],
            class_methods_module: None,
            included_body: Some(test_body(
                b_name,
                "b_attr",
                InfusionBodyOrigin::SyntheticConcernIncluded,
            )),
            prepended_body: None,
            class_methods: None,
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(a_name, &a);
        concerns.insert(b_name, &b);

        let mut out = Vec::new();
        expand_concern_bodies(
            &a,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            ConcernBodyKind::Included,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let emitted: Vec<_> = out
            .iter()
            .map(|synthetic| synthetic.body.calls[0].symbol_args[0].name.as_str())
            .collect();
        assert_eq!(emitted, vec!["b_attr", "a_attr"]);
    }

    #[test]
    fn collect_concern_dependencies_unshifts_prepends_before_includes() {
        let names = NameTable::new();
        let source = b"\
module Auditable
  extend ActiveSupport::Concern
  include ::IncludedDep
  prepend ::PrependedDep
end
";
        let parse_result = ruby_prism::parse(source);
        let mut collector = Collector::new(
            &names,
            0,
            None,
            source,
            None,
            &parse_result,
            activesupport_options(),
            inflector::default_en(),
            true,
        );
        collector.visit(&parse_result.node());

        assert_eq!(collector.concerns.len(), 1);
        let dependencies = &collector.concerns[0].dependencies;
        let first = dependencies[0].module_name_candidates[0];
        let second = dependencies[1].module_name_candidates[0];
        assert_eq!(names.display_type_name(first), "::PrependedDep");
        assert_eq!(names.display_type_name(second), "::IncludedDep");
    }

    #[test]
    fn expand_concern_prepended_bodies_orders_dependencies_before_owner() {
        let names = NameTable::new();
        let timestamped_name = names.parse_type_name("::Timestamped");
        let auditable_name = names.parse_type_name("::Auditable");
        let user_name = names.parse_type_name("::User");
        let timestamped = ConcernDef {
            name: timestamped_name,
            source_index: 0,
            dependencies: Vec::new(),
            class_methods_module: None,
            included_body: None,
            prepended_body: Some(test_body(
                timestamped_name,
                "timestamped",
                InfusionBodyOrigin::SyntheticConcernPrepended,
            )),
            class_methods: None,
        };
        let auditable = ConcernDef {
            name: auditable_name,
            source_index: 1,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![timestamped_name],
            }],
            class_methods_module: None,
            included_body: None,
            prepended_body: Some(test_body(
                auditable_name,
                "audited",
                InfusionBodyOrigin::SyntheticConcernPrepended,
            )),
            class_methods: None,
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(timestamped_name, &timestamped);
        concerns.insert(auditable_name, &auditable);

        let mut out = Vec::new();
        expand_concern_bodies(
            &auditable,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            ConcernBodyKind::Prepended,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let emitted: Vec<_> = out
            .iter()
            .map(|synthetic| synthetic.body.calls[0].symbol_args[0].name.as_str())
            .collect();
        assert_eq!(emitted, vec!["timestamped", "audited"]);
        assert!(out.iter().all(|synthetic| matches!(
            synthetic.body.origin,
            InfusionBodyOrigin::SyntheticConcernPrepended
        )));
    }

    #[test]
    fn expand_concern_class_methods_orders_dependencies_before_owner_extends() {
        let names = NameTable::new();
        let timestamped_name = names.parse_type_name("::Timestamped");
        let auditable_name = names.parse_type_name("::Auditable");
        let user_name = names.parse_type_name("::User");
        let timestamped = ConcernDef {
            name: timestamped_name,
            source_index: 0,
            dependencies: Vec::new(),
            class_methods_module: Some(ClassMethodsModuleRef {
                module_name: names
                    .append_type_name(timestamped_name, names.intern_symbol("ClassMethods")),
                location: (0, 0),
            }),
            included_body: None,
            prepended_body: None,
            class_methods: Some(test_class_methods(&names, timestamped_name, "timestamped")),
        };
        let auditable = ConcernDef {
            name: auditable_name,
            source_index: 1,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![timestamped_name],
            }],
            class_methods_module: Some(ClassMethodsModuleRef {
                module_name: names
                    .append_type_name(auditable_name, names.intern_symbol("ClassMethods")),
                location: (0, 0),
            }),
            included_body: None,
            prepended_body: None,
            class_methods: Some(test_class_methods(&names, auditable_name, "audited")),
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(timestamped_name, &timestamped);
        concerns.insert(auditable_name, &auditable);

        let mut out = Vec::new();
        expand_concern_class_methods(
            &auditable,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            SyntheticClassMethodsMixinKind::Extend,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let emitted: Vec<_> = out
            .iter()
            .map(|synthetic| names.display_type_name(synthetic.class_methods_module))
            .collect();
        assert_eq!(
            emitted,
            vec![
                "::Timestamped::ClassMethods".to_string(),
                "::Auditable::ClassMethods".to_string(),
            ]
        );
        assert!(out.iter().all(|synthetic| synthetic.target == user_name));
        let Some(Declaration::Class(decl)) = out[0].to_declaration(&names) else {
            panic!("expected synthetic class declaration");
        };
        assert_eq!(decl.members.len(), 1);
        let Member::Extend(extend) = &decl.members[0] else {
            panic!("expected extend member");
        };
        assert_eq!(extend.mixin.module_name, "::Timestamped::ClassMethods");
    }

    #[test]
    fn expand_concern_class_methods_can_prepend_to_target_singleton() {
        let names = NameTable::new();
        let auditable_name = names.parse_type_name("::Auditable");
        let user_name = names.parse_type_name("::User");
        let auditable = ConcernDef {
            name: auditable_name,
            source_index: 0,
            dependencies: Vec::new(),
            class_methods_module: Some(ClassMethodsModuleRef {
                module_name: names
                    .append_type_name(auditable_name, names.intern_symbol("ClassMethods")),
                location: (0, 0),
            }),
            included_body: None,
            prepended_body: None,
            class_methods: Some(test_class_methods(&names, auditable_name, "audited")),
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(auditable_name, &auditable);

        let mut out = Vec::new();
        expand_concern_class_methods(
            &auditable,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            SyntheticClassMethodsMixinKind::SingletonPrepend,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let Some(Declaration::Class(decl)) = out[0].to_declaration(&names) else {
            panic!("expected synthetic class declaration");
        };
        let Member::SingletonPrepend(prepend) = &decl.members[0] else {
            panic!("expected singleton prepend member");
        };
        assert_eq!(prepend.mixin.module_name, "::Auditable::ClassMethods");
    }

    #[test]
    fn expand_concern_class_methods_skips_cycles_but_keeps_each_extend_once() {
        let names = NameTable::new();
        let a_name = names.parse_type_name("::A");
        let b_name = names.parse_type_name("::B");
        let user_name = names.parse_type_name("::User");
        let a = ConcernDef {
            name: a_name,
            source_index: 0,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![b_name],
            }],
            class_methods_module: Some(ClassMethodsModuleRef {
                module_name: names.append_type_name(a_name, names.intern_symbol("ClassMethods")),
                location: (0, 0),
            }),
            included_body: None,
            prepended_body: None,
            class_methods: Some(test_class_methods(&names, a_name, "a_method")),
        };
        let b = ConcernDef {
            name: b_name,
            source_index: 1,
            dependencies: vec![ConcernDependency {
                module_name_candidates: vec![a_name],
            }],
            class_methods_module: Some(ClassMethodsModuleRef {
                module_name: names.append_type_name(b_name, names.intern_symbol("ClassMethods")),
                location: (0, 0),
            }),
            included_body: None,
            prepended_body: None,
            class_methods: Some(test_class_methods(&names, b_name, "b_method")),
        };
        let mut concerns = FxHashMap::default();
        concerns.insert(a_name, &a);
        concerns.insert(b_name, &b);

        let mut out = Vec::new();
        expand_concern_class_methods(
            &a,
            &concerns,
            ConcernExpansionTarget {
                target: user_name,
                target_kind: InfusionOwnerKind::Class,
            },
            SyntheticClassMethodsMixinKind::Extend,
            &mut Vec::new(),
            &mut FxHashSet::default(),
            &mut out,
        );

        let emitted: Vec<_> = out
            .iter()
            .map(|synthetic| names.display_type_name(synthetic.class_methods_module))
            .collect();
        assert_eq!(
            emitted,
            vec![
                "::B::ClassMethods".to_string(),
                "::A::ClassMethods".to_string(),
            ]
        );
    }
}

/// `collect_source` on a parallel-ingest worker table (ADR-0033): the
/// walk must not mint a positional `Name`, and its content-addressed
/// ids must survive `NameTable::merge` regardless of the order the
/// worker interned things in.
#[cfg(test)]
mod worker_collect_tests {
    use super::*;

    // `attr_reader` inside `class_methods do ... end` is the one attr
    // site the infusion collector builds itself (`ClassMethodsCollector`),
    // which used to re-intern the file path instead of using the
    // pre-interned `source_file`. An attr in the class body alone would
    // not reach it.
    const SOURCE: &[u8] = b"\
module Trackable
  extend ActiveSupport::Concern
  class_methods do
    attr_reader :foo
  end
end

class User < ApplicationRecord
  include Trackable
  attr_reader :bar
  has_many :bars
end
";

    fn rails_options() -> InfusionOptions {
        InfusionOptions {
            activesupport: true,
            activemodel: true,
            activerecord: true,
            paranoia: false,
            sidekiq: false,
        }
    }

    fn collect_on_worker<'a>(worker: &NameTable, source_file: Name) -> CollectedSource<'a> {
        let parse_result = ruby_prism::parse(SOURCE);
        assert!(parse_result.errors().next().is_none());
        collect_source(
            worker,
            0,
            Some(source_file),
            SOURCE,
            Some(Path::new("app/models/user.rb")),
            &parse_result,
            rails_options(),
            inflector::default_en(),
            true,
        )
    }

    #[test]
    fn worker_collect_mints_no_positional_name() {
        let main = NameTable::new();
        let source_file = main.intern("app/models/user.rb");
        let worker = NameTable::new();
        let collected = collect_on_worker(&worker, source_file);
        let class_methods = collected
            .concerns
            .iter()
            .find_map(|concern| concern.class_methods.as_ref())
            .expect("Trackable.class_methods must be collected");
        let attr = class_methods
            .members
            .iter()
            .find_map(|member| match member {
                Member::AttrReader(attr) => Some(&attr.attribute),
                _ => None,
            })
            .expect("attr_reader :foo must be collected as a ClassMethods member");
        assert_eq!(attr.source_file, Some(source_file));
        // Panics ("must not carry positional `Name` entries") if the
        // walk interned the path on the worker table.
        main.merge(worker);
    }

    #[test]
    fn worker_collect_ids_survive_merge_in_either_intern_order() {
        let orders: [&[&str]; 2] = [&["Bar", "User", "foo"], &["foo", "User", "Bar"]];
        for order in orders {
            let main = NameTable::new();
            let source_file = main.intern("app/models/user.rb");
            let worker = NameTable::new();
            for s in order {
                worker.intern_symbol(s);
            }
            let collected = collect_on_worker(&worker, source_file);
            main.merge(worker);

            let class_names: Vec<String> = collected
                .top_level
                .iter()
                .filter_map(|decl| match decl {
                    Declaration::Class(class) => Some(main.display_type_name(class.class_name)),
                    _ => None,
                })
                .collect();
            assert_eq!(class_names, vec!["::User".to_string()], "order={order:?}");
            let concern_names: Vec<String> = collected
                .concerns
                .iter()
                .map(|concern| {
                    let class_methods = concern
                        .class_methods
                        .as_ref()
                        .expect("Trackable.class_methods must be collected");
                    format!(
                        "{} {}",
                        main.display_type_name(concern.name),
                        main.display_type_name(class_methods.module_name)
                    )
                })
                .collect();
            assert_eq!(
                concern_names,
                vec!["::Trackable ::Trackable::ClassMethods".to_string()],
                "order={order:?}"
            );
            let associations: Vec<String> = collected
                .active_record_associations
                .iter()
                .map(|assoc| format!("{} {}", main.display_type_name(assoc.owner), assoc.name))
                .collect();
            assert_eq!(
                associations,
                vec!["::User bars".to_string()],
                "order={order:?}"
            );
        }
    }
}
