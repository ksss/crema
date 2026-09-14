//! `crema extract` — single-JSON export of check-internal facts.
//!
//! The check pipeline computes definition, reference-resolution, and
//! consultation information and normally discards it. `crema extract`
//! runs the same pipeline (`run_check` in main.rs, shared with `crema
//! check`) and prints the per-file facts as one JSON document on
//! stdout so external programs (Ruby + jq) can build find-references / dead-code
//! / coverage / layering tools without crema growing a subcommand per
//! use case (`crema references` was such a subcommand; it was removed
//! once `method_call[].symbol` answered the same question).
//!
//! Five record kinds per file, all symbol strings in absolute rbs
//! syntax (`::User#save`, `::User.new`, `::Billing::Invoice`):
//!
//! - `definitions` — repo-declared symbols only (the A-layer
//!   `path_index`; gem/core declarations never appear).
//! - `implements` — Ruby `def` sites and constant-assignment sites, on
//!   the lexical symbol axis (see [`ImplementsRecord`]; v6 added the
//!   `constant` kind). The implementation counterpart of
//!   `definitions`: in a sig-separated project `.rb` files contribute
//!   no declarations, so without this array "which file implements
//!   this symbol" is unanswerable from the export.
//! - `method_call` — every call/super site the check touched, tagged
//!   with a `state` variant (v4, after Steep's
//!   `TypeInference::MethodCall` Typed / Untyped / NoMethodError /
//!   Error): `typed` (resolved, no diagnostic), `error` (resolved but
//!   the check diagnosed the call), `no_method_error` (receiver type
//!   resolved, method missing), `untyped` (receiver untyped, no
//!   resolution attempted — the honesty column). See
//!   [`MethodCallRecord`].
//! - `constant` — every constant *read* site the check touched, tagged
//!   with a `state` variant (v5, the constant sibling of
//!   `method_call`'s taxonomy): `typed` / `error` /
//!   `unknown_constant` / `untyped`. Read sites plus the superclass
//!   position (`class Sub < Base`, v7); writes (`X = 1` targets,
//!   or-writes) are out of scope. See [`ConstantRecord`].
//! - `consulted` — every symbol the check touched (deduped, no
//!   positions): the checker's environment queries plus the types each
//!   def signature references (annotation lowering resolves those in
//!   the env phase, so the query log alone cannot see them). Includes
//!   implicit dependencies (ancestor chains, alias expansion) that
//!   never appear as source occurrences — dependency-graph consumers
//!   must read this, not `method_call`. Strictly "what the *check*
//!   consulted": extract-mode bookkeeping performs no queries of its
//!   own, so checking a file with or without extract consults the same
//!   set (`return_type` is `null` where the check computed no type,
//!   rather than synthesized).
//!
//! Global variables (`$foo`) are intentionally absent from v1 output:
//! every emitted symbol is absolute (`::`-prefixed) rbs syntax, and
//! globals have no such form.
//!
//! The internal `ConsultedKey` enum never leaks: the public schema is
//! rbs vocabulary only (the same lingua-franca rule diagnostics
//! follow).

use std::cell::RefCell;
use std::collections::BTreeMap;

use serde::Serialize;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::definition_builder::{ConsultedKey, DefinitionBuilder};
use crate::environment::draft::PathIndexKey;
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassDeclaration, ClassOrModule, ClassOrModuleAliasEntry,
    ModuleAliasDeclaration, ModuleDeclaration,
};
use crate::name::{Name, NameTable};
use crate::type_name::TypeName;


/// Version of the extract document schema.
pub const EXTRACT_SCHEMA_VERSION: u32 = 7;

/// `state` variant of a [`MethodCallRecord`], after Steep's
/// `TypeInference::MethodCall` class hierarchy (`method_call.rb`:
/// `Typed` / `Untyped` / `NoMethodError` / `Error`).
pub mod method_call_state {
    /// Resolved, and the check emitted no diagnostic for the call.
    pub const TYPED: &str = "typed";
    /// Resolution succeeded but the check diagnosed the call itself
    /// (argument mismatch, unresolved overloading, visibility, block
    /// misuse). A diagnostic inside a block body does not error the
    /// outer call — the inner site carries its own record.
    pub const ERROR: &str = "error";
    /// Receiver type resolved but the method was not found (the
    /// `NoMethod` diagnostic fired). Absent from v1-v3 output.
    pub const NO_METHOD_ERROR: &str = "no_method_error";
    /// Receiver was `untyped`, so no resolution was attempted. The
    /// honesty column: consumers can report "N confirmed + M unknown"
    /// instead of silently dropping these sites.
    pub const UNTYPED: &str = "untyped";
}

/// One call/super site. Field order is the JSON key order. Field
/// names follow Steep `TypeInference::MethodCall::Base` (`method_name`
/// / `receiver_type` / `return_type`).
#[derive(Debug, Clone, Serialize)]
pub struct MethodCallRecord {
    /// Variant tag — see [`method_call_state`].
    pub state: &'static str,
    /// `"call"` / `"super"` — [`ReferenceKind::as_str`].
    pub kind: &'static str,
    /// Start of the call node's source span (never a diagnostic's
    /// narrowed span).
    pub start_byte: u32,
    /// Exclusive end of the site's source range (prism node span).
    pub end_byte: u32,
    /// Bare name of the called method (no owner qualification — the
    /// `untyped` / `no_method_error` states have none to offer).
    pub method_name: String,
    /// Static receiver type at the site, rbs syntax (same rendering
    /// as diagnostics). For `kind: "super"` the enclosing self class;
    /// `"untyped"` for the `untyped` state.
    pub receiver_type: String,
    /// Type of the expression at this site as the check computed it —
    /// the call's return type, safe-navigation nil widen included,
    /// union receivers joined (a site with several `symbol` records
    /// shares one `return_type`). `null` when the check itself never
    /// computed a type for the site (a call reached only through the
    /// plain tree descent, where its value is discarded); a diagnosed
    /// site follows the same rule. Extract reports what the check
    /// computed and runs no inference of its own.
    pub return_type: Option<String>,
    /// `typed` / `error` states only: landing definition in absolute
    /// rbs syntax (`::Parent#foo`) — the `implemented_in` owner, not
    /// the receiver's class. One record per landing (union receivers
    /// produce several records sharing a span). A call through an
    /// interface-typed receiver has `implemented_in: None` (an
    /// interface declares without implementing); extract records the
    /// declaring interface (`::_Speak#speak`) so the dependency is not
    /// lost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
}

/// `state` variant of a [`ConstantRecord`] — the constant sibling of
/// [`method_call_state`]. No Steep port exists for this axis (Steep has
/// no constant counterpart of its `TypeInference::MethodCall`
/// hierarchy), so the variant names follow crema's own diagnostics.
pub mod constant_state {
    /// Resolved, and the check emitted no diagnostic for the read.
    pub const TYPED: &str = "typed";
    /// Resolution succeeded but the check diagnosed the read itself —
    /// today that is `Ruby::DeprecatedReference` only.
    pub const ERROR: &str = "error";
    /// Resolution was attempted and failed (the `Ruby::UnknownConstant`
    /// diagnostic fired) — the constant version of `no_method_error`.
    pub const UNKNOWN_CONSTANT: &str = "unknown_constant";
    /// Resolution was never completed: a dynamic-parent path
    /// (`expr::CONST`) that cannot be statically walked, or an
    /// `untyped`-typed intermediate that short-circuits the walk. The
    /// honesty column, as in `method_call`.
    pub const UNTYPED: &str = "untyped";
}

/// One constant read site (v5), or the superclass position of a
/// `class Sub < Base` declaration (v7 — the check resolves it in the
/// enclosing scope exactly like a read). Write targets never record.
/// Field order is the JSON key order.
#[derive(Debug, Clone, Serialize)]
pub struct ConstantRecord {
    /// Variant tag — see [`constant_state`].
    pub state: &'static str,
    /// Start of the constant node's source span (a path node spans the
    /// whole `A::B::C`).
    pub start_byte: u32,
    /// Exclusive end of the site's source range.
    pub end_byte: u32,
    /// The constant as written in source (`Billing::MAX`, with the
    /// leading `::` when the path is absolute — the same round-trip
    /// discipline as the `UnknownConstant` diagnostic's `path` field).
    pub path: String,
    /// `typed` / `error` states only: the resolved constant in absolute
    /// rbs syntax (`::Billing::MAX`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Type the check computed for the read (rbs syntax, same rendering
    /// as diagnostics). Omitted when the check computed none (the
    /// `unknown_constant` / `untyped` states).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub constant_type: Option<String>,
}

/// One Ruby `def` site or constant assignment (`FOO = 1`, `A::B = 2`):
/// the symbol this file implements. The symbol is the *lexical* axis —
/// the enclosing class/module (top-level defs are `::Object`, Ruby's
/// actual owner; top-level constants are rooted `::X`, rbs's spelling)
/// plus the name — regardless of whether, or where, a signature
/// declares it. A def whose signature resolves to an ancestor
/// (`class Sub < Foo; def bar` with only `Foo#bar` declared) still
/// records `::Sub#bar`: implements answers "what does this file's
/// source define", not "which declaration was it checked against".
///
/// Constant symbols follow the inline collector's rule so the two join
/// on the same string: `qualify_under(innermost class, path)` with a
/// path write concatenated as written (`A::B = 2` inside `class Foo`
/// is `::Foo::A::B`, never resolved) and a rooted write kept as is.
/// `||=` / `+=` and dynamic-parent writes (`obj.foo::BAR = 1`) have no
/// record.
#[derive(Debug, Clone, Serialize)]
pub struct ImplementsRecord {
    pub symbol: String,
    /// Same vocabulary as [`DefinitionRecord`]s: `instance_method` /
    /// `singleton_method` / `constant` (v6).
    pub kind: &'static str,
    pub start_byte: u32,
    /// Exclusive end of the whole `def` node (keyword through `end`)
    /// or of the whole write node (name through the end of the RHS).
    pub end_byte: u32,
}

/// One repo-declared symbol. Type-level records span the declaration
/// head (keyword through name); method records span the defining
/// member.
#[derive(Debug, Clone, Serialize)]
pub struct DefinitionRecord {
    pub symbol: String,
    /// rbs declaration vocabulary: `class` / `module` / `interface` /
    /// `class_alias` / `module_alias` / `type_alias` / `constant` /
    /// `instance_method` / `singleton_method`.
    pub kind: &'static str,
    pub start_byte: u32,
    pub end_byte: u32,
    /// Method records only: the declared signature (overloads joined
    /// with ` | `), rbs syntax.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method_type: Option<String>,
}

/// Per-file record bundle. Field order is the JSON key order.
#[derive(Debug, Default, Serialize)]
pub struct FileRecord {
    /// xxh3 of the file bytes as fixed-width lowercase hex, for
    /// consumers to detect staleness against the working tree.
    pub content_hash: String,
    pub definitions: Vec<DefinitionRecord>,
    /// Ruby `def` sites (lexical symbol axis) — the implementation
    /// counterpart of `definitions`, which only sees the declaration
    /// space and is empty for `.rb` files in sig-separated projects.
    pub implements: Vec<ImplementsRecord>,
    /// Every call/super site, tagged with a `state` variant (v4).
    pub method_call: Vec<MethodCallRecord>,
    /// Every constant read site, tagged with a `state` variant (v5).
    pub constant: Vec<ConstantRecord>,
    /// Deduped, sorted symbol set the check consulted (positionless).
    pub consulted: Vec<String>,
}

/// The whole extract document.
#[derive(Debug, Serialize)]
pub struct ExtractOutput {
    pub version: u32,
    /// Base directory the `files` keys are relative to.
    pub root: String,
    pub files: BTreeMap<String, FileRecord>,
}

/// The site kind column of [`MethodCallRecord`]: a plain call, or a
/// `super` dispatch from inside a method body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    Call,
    Super,
}

impl ReferenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ReferenceKind::Call => "call",
            ReferenceKind::Super => "super",
        }
    }
}

/// Site sink handed to the type checker in extract mode (every
/// resolved site is recorded, no query filter). Interior mutability
/// mirrors `TypeChecker::diagnostics`.
#[derive(Debug, Default)]
pub struct ExtractSitesCollector {
    method_calls: RefCell<Vec<MethodCallRecord>>,
    implements: RefCell<Vec<ImplementsRecord>>,
    constants: RefCell<Vec<ConstantRecord>>,
    /// Types referenced by the file's def signatures — a consulted
    /// input the [`ConsultationLog`] cannot see (annotation lowering
    /// resolved them during the env phase). Set-typed: signatures
    /// repeat types freely and only membership matters.
    signature_types: RefCell<FxHashSet<TypeName>>,
}

impl ExtractSitesCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_method_call(&self, record: MethodCallRecord) {
        self.method_calls.borrow_mut().push(record);
    }

    pub fn push_implements(&self, record: ImplementsRecord) {
        self.implements.borrow_mut().push(record);
    }

    pub fn push_constant(&self, record: ConstantRecord) {
        self.constants.borrow_mut().push(record);
    }

    pub fn push_signature_type(&self, name: TypeName) {
        self.signature_types.borrow_mut().insert(name);
    }

    /// Drains the collected sites, sorted by source position for
    /// deterministic output (signature types come back unsorted — they
    /// merge into the consulted string set, which sorts at render
    /// time).
    pub fn into_parts(
        self,
    ) -> (
        Vec<MethodCallRecord>,
        Vec<ImplementsRecord>,
        Vec<ConstantRecord>,
        Vec<TypeName>,
    ) {
        let mut method_calls = self.method_calls.into_inner();
        method_calls.sort_by(|a, b| {
            a.start_byte
                .cmp(&b.start_byte)
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        let mut implements = self.implements.into_inner();
        implements.sort_by_key(|i| i.start_byte);
        // `end_byte` breaks the tie a receiver read inside a
        // dynamic-parent path creates (`Billing.name::Dyn`: the inner
        // `Billing` and the whole-path record share a start).
        let mut constants = self.constants.into_inner();
        constants.sort_by_key(|c| (c.start_byte, c.end_byte));
        let signature_types = self.signature_types.into_inner().into_iter().collect();
        (method_calls, implements, constants, signature_types)
    }
}

/// Project one file's consulted inputs onto absolute rbs symbol
/// strings: sorted, deduped, internal enum vocabulary erased. Two
/// sources merge here: the pre-projection [`ConsultedKey`] entries
/// (the u64 cache projection would have dropped the names) and
/// `signature_types` — the types the file's def signatures reference,
/// which the log cannot see because annotation lowering resolved them
/// during the env phase.
///
/// Misses are included alongside hits — a query that came back empty
/// is still a dependency of this file's diagnostics (the same negative
/// dependency ADR-0032 records for invalidation).
///
/// Method-family keys contribute both the receiver type (`::User`) and
/// the qualified method (`::User#discount_rate` / `::User.new`) —
/// unlike the cache's u64 projection, which drops the method symbol
/// for selectivity reasons that don't apply to a self-describing
/// export. Keys carrying only a bare `Symbol` (globals, unresolved
/// constant names) contribute their scope type names only: a bare
/// symbol has no absolute rbs spelling.
pub fn consulted_symbols(
    entries: &FxHashMap<ConsultedKey, bool>,
    signature_types: &[TypeName],
    names: &NameTable,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tn in signature_types {
        out.push(names.display_type_name(*tn));
    }
    let type_name = |n: TypeName| names.display_type_name(n);
    let qualified = |n: TypeName, separator: &str, m: crate::name::Symbol| {
        format!(
            "{}{}{}",
            names.display_type_name(n),
            separator,
            names.resolve(m)
        )
    };
    for key in entries.keys() {
        use ConsultedKey::*;
        match key {
            ExpandTypeAlias(n)
            | ConstantDeprecated(n)
            | ClassOrModuleDeprecated(n)
            | ClassAliasDeprecated(n)
            | DeclaredKind(n)
            | IsDeclaredClass(n)
            | IsDeclaredClassAlias(n)
            | IsDeclaredInterface(n)
            | IsDeclaredTypeAlias(n)
            | IsDeclaredConstant(n)
            | ClassOrModuleKind(n)
            | ClassTypeParams(n)
            | InterfaceMethodNames(n)
            | SuperclassTypeName(n)
            | HasCompleteAncestorChain(n)
            | InstanceAncestors(n)
            | SingletonAncestors(n)
            | OneInstanceAncestors(n) => out.push(type_name(*n)),
            GlobalLookup(_) | GlobalDeprecated(_) => {}
            MethodResolution {
                receiver: n,
                method: m,
            }
            | InterfaceMethodResolution {
                interface: n,
                method: m,
            }
            | SuperInstanceMethod {
                class: n,
                method: m,
            } => {
                out.push(type_name(*n));
                out.push(qualified(*n, "#", *m));
            }
            SingletonMethodResolution {
                class: n,
                method: m,
            }
            | SuperSingletonMethod {
                class: n,
                method: m,
            } => {
                out.push(type_name(*n));
                out.push(qualified(*n, ".", *m));
            }
            InstanceVariable { class: n, .. }
            | ClassVariable { class: n, .. }
            | ClassInstanceVariable { class: n, .. } => out.push(type_name(*n)),
            SyntheticConcernTargets { current, .. } => {
                if let Some(n) = current {
                    out.push(type_name(*n));
                }
            }
            ConstantResolution { context, .. } => {
                for scope in context.scopes() {
                    out.push(type_name(*scope));
                }
            }
            ConstantResolutionInNamespace { scope, .. } => {
                if let Some(n) = scope {
                    out.push(type_name(*n));
                }
            }
            ConstantResolutionChild { module, .. } => out.push(type_name(*module)),
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Enumerate the definitions `file` contributed to the environment.
///
/// The entry point is the A-layer `path_index` (file → contributed
/// declaration keys), which is what keeps gem/core declarations out:
/// G-snapshot entries are never indexed there. Type-level records span
/// the declaration's *name* token; method records span the defining
/// member (`MemberRef::location`), which is also what silently excludes
/// `MemberRef::Synthesized` methods — they have no source anchor.
///
/// `PathIndexKey::Global` contributes nothing: every extract symbol is
/// absolute rbs syntax and globals (`$foo`) have no such form (v1
/// scope, see the module doc).
pub fn file_definitions(env: &DefinitionBuilder, file: Name) -> Vec<DefinitionRecord> {
    let environment = env.env();
    let names = environment.names();
    let Some(keys) = environment.path_index_for(file) else {
        return Vec::new();
    };
    let mut records: Vec<DefinitionRecord> = Vec::new();
    let push = |symbol: String, kind: &'static str, span: (u32, u32)| DefinitionRecord {
        symbol,
        kind,
        start_byte: span.0,
        end_byte: span.1,
        method_type: None,
    };
    for key in &keys {
        match key {
            PathIndexKey::ClassOrModule(tn) => {
                let symbol = names.display_type_name(*tn);
                match environment.class_decls().get(tn) {
                    Some(ClassOrModule::Class(entry)) => {
                        for (origin, _, decl) in entry.context_decls() {
                            if origin.file() != Some(file) {
                                continue;
                            }
                            if let Some(span) = class_decl_name_span(decl) {
                                records.push(push(symbol.clone(), "class", span));
                            }
                        }
                    }
                    Some(ClassOrModule::Module(entry)) => {
                        for (origin, _, decl) in entry.context_decls() {
                            if origin.file() != Some(file) {
                                continue;
                            }
                            if let Some(span) = module_decl_name_span(decl) {
                                records.push(push(symbol.clone(), "module", span));
                            }
                        }
                    }
                    None => {}
                }
                if let Some(definition) = env.build_instance(tn) {
                    method_definition_records(env, *tn, file, false, &definition, &mut records);
                }
                if let Some(definition) = env.build_singleton(tn) {
                    method_definition_records(env, *tn, file, true, &definition, &mut records);
                }
            }
            PathIndexKey::Interface(tn) => {
                if let Some(entry) = environment.interface_decls().get(tn)
                    && entry.file() == Some(file)
                    && let Some(location) = entry.decl().location
                {
                    records.push(push(
                        names.display_type_name(*tn),
                        "interface",
                        (location.name_range.start_byte, location.name_range.end_byte),
                    ));
                }
                if let Some(definition) = env.build_interface(tn) {
                    method_definition_records(env, *tn, file, false, &definition, &mut records);
                }
            }
            PathIndexKey::ClassAlias(tn) => {
                if let Some(entry) = environment.class_alias_decls().get(tn) {
                    let symbol = names.display_type_name(*tn);
                    match entry {
                        ClassOrModuleAliasEntry::Class(e) => {
                            if e.file() == Some(file)
                                && let Some(span) = class_alias_name_span(e.decl())
                            {
                                records.push(push(symbol, "class_alias", span));
                            }
                        }
                        ClassOrModuleAliasEntry::Module(e) => {
                            if e.file() == Some(file)
                                && let Some(span) = module_alias_name_span(e.decl())
                            {
                                records.push(push(symbol, "module_alias", span));
                            }
                        }
                    }
                }
            }
            PathIndexKey::TypeAlias(tn) => {
                if let Some(entry) = environment.type_alias_decls().get(tn)
                    && entry.file.file() == Some(file)
                    && let Some(location) = entry.decl.location
                {
                    records.push(push(
                        names.display_type_name(*tn),
                        "type_alias",
                        (location.name_range.start_byte, location.name_range.end_byte),
                    ));
                }
            }
            PathIndexKey::Constant(tn) => {
                if let Some(entry) = environment.constant_decls().get(tn)
                    && entry.file.file() == Some(file)
                    && let Some(location) = entry.decl.location
                {
                    records.push(push(
                        names.display_type_name(*tn),
                        "constant",
                        (location.name_range.start_byte, location.name_range.end_byte),
                    ));
                }
            }
            PathIndexKey::Global(_) => {}
        }
    }
    records.sort_by(|a, b| {
        a.start_byte
            .cmp(&b.start_byte)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    records
}

/// Method definitions `file` contributed to `owner`'s own method table
/// (`Definition.methods` is own-methods-only — inherited entries live
/// on the ancestor's own table, so each definition is listed exactly
/// once, under its declaring type).
fn method_definition_records(
    env: &DefinitionBuilder,
    owner: TypeName,
    file: Name,
    singleton: bool,
    definition: &crate::definition::Definition,
    records: &mut Vec<DefinitionRecord>,
) {
    let names = env.names();
    let types = env.types();
    let separator = if singleton { "." } else { "#" };
    let kind = if singleton {
        "singleton_method"
    } else {
        "instance_method"
    };
    for (sym, method) in &definition.methods {
        let symbol = format!(
            "{}{}{}",
            names.display_type_name(owner),
            separator,
            names.resolve(*sym)
        );
        // An `alias new old` method anchors at the alias member itself:
        // its `defs` are clones of the *target's* defs, whose members
        // point at the target's source (possibly another file).
        if let Some(alias_member) = &method.alias_member {
            if alias_member.source_file == Some(file)
                && let Some(location) = alias_member.location
            {
                records.push(DefinitionRecord {
                    symbol,
                    kind,
                    start_byte: location.range.start_byte,
                    end_byte: location.range.end_byte,
                    method_type: None,
                });
            }
            continue;
        }
        // One record per defining member: a multi-overload `def` lowers
        // to one TypeDef per overload sharing a member, joined back with
        // ` | ` here.
        let mut spans: Vec<((u32, u32), Vec<String>)> = Vec::new();
        for td in &method.defs {
            if td.member.source_file() != Some(file) {
                continue;
            }
            let Some(range) = td.member.location() else {
                continue;
            };
            let key = (range.start_byte, range.end_byte);
            let rendered = crate::type_checker::display_method_type(&td.type_, types, names);
            match spans.iter_mut().find(|(k, _)| *k == key) {
                Some((_, overloads)) => overloads.push(rendered),
                None => spans.push((key, vec![rendered])),
            }
        }
        for ((start_byte, end_byte), overloads) in spans {
            records.push(DefinitionRecord {
                symbol: symbol.clone(),
                kind,
                start_byte,
                end_byte,
                method_type: Some(overloads.join(" | ")),
            });
        }
    }
}

fn class_decl_name_span(decl: &ClassDeclaration) -> Option<(u32, u32)> {
    match decl {
        ClassDeclaration::Signature(c) => c
            .location
            .map(|l| (l.name_range.start_byte, l.name_range.end_byte)),
        ClassDeclaration::Ruby(r) => Some(r.name_location),
    }
}

fn module_decl_name_span(decl: &ModuleDeclaration) -> Option<(u32, u32)> {
    match decl {
        ModuleDeclaration::Signature(m) => m
            .location
            .map(|l| (l.name_range.start_byte, l.name_range.end_byte)),
        ModuleDeclaration::Ruby(r) => Some(r.name_location),
    }
}

fn class_alias_name_span(decl: &ClassAliasDeclaration) -> Option<(u32, u32)> {
    match decl {
        ClassAliasDeclaration::Signature(a) => a
            .location
            .map(|l| (l.new_name_range.start_byte, l.new_name_range.end_byte)),
        ClassAliasDeclaration::Ruby(r) => Some(r.name_location),
    }
}

fn module_alias_name_span(decl: &ModuleAliasDeclaration) -> Option<(u32, u32)> {
    match decl {
        ModuleAliasDeclaration::Signature(a) => a
            .location
            .map(|l| (l.new_name_range.start_byte, l.new_name_range.end_byte)),
        ModuleAliasDeclaration::Ruby(r) => Some(r.name_location),
    }
}
