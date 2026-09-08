use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::ast::ruby::byte_offset_to_line;
use crate::config::{DiagnosticConfig, Format};
use crate::definition::ConstantContext;
use crate::definition_builder::DefinitionBuilder;
use crate::location::{DuplicateSource, LocationRange, SourceLocation};
use crate::type_name::TypeName;

/// Diagnostic severity tier, mirroring LSP `DiagnosticSeverity` 1..4
/// (Steep `Diagnostic::LSPFormatter::{ERROR,WARNING,INFORMATION,HINT}`)
/// plus a crema-specific `Ignore` for diagnostics suppressed by the
/// `crema.toml` `[diagnostic]` config. Only `Error` participates in the
/// process exit code; warning / information / hint surface for user
/// attention but leave exit status at 0; `Ignore` is dropped before
/// reaching the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Information,
    Hint,
    Ignore,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Information => "information",
            Severity::Hint => "hint",
            Severity::Ignore => "ignore",
        }
    }

    /// Parse a TOML string value for `[diagnostic]` config entries.
    /// Accepts only the five canonical lowercase forms; anything else
    /// must be surfaced as a config error by the caller.
    pub fn from_config_str(s: &str) -> Option<Severity> {
        match s {
            "error" => Some(Severity::Error),
            "warning" => Some(Severity::Warning),
            "information" => Some(Severity::Information),
            "hint" => Some(Severity::Hint),
            "ignore" => Some(Severity::Ignore),
            _ => None,
        }
    }
}

pub use crate::ast::ruby::annotations::InlineAliasKind;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub location: SourceLocation,
    /// Enclosing lexical scope used as fingerprint material, e.g.
    /// `::Foo::Bar#baz` (instance method) or `::Foo::Bar.baz`
    /// (singleton method). `None` for diagnostics without a method
    /// scope concept (sig-side validation, parse errors); those hash
    /// with empty scope material. Filled at the `push_diagnostic`
    /// funnel for the type-checker path, never at construction sites.
    pub scope: Option<String>,
}

// serde is derived for the incremental check cache (ADR-0032 Decision 1:
// structured diagnostics are one of the four persisted artifact kinds).
// bincode gives no schema evolution — any layout-affecting edit to this
// enum (or a nested field type) must bump
// `crate::incremental::CACHE_SCHEMA_VERSION` so old caches fall back to a
// full recheck instead of misparsing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DiagnosticKind {
    ArgumentTypeMismatch {
        method_name: String,
        param_index: Option<usize>,
        keyword: Option<String>,
        expected: String,
        actual: String,
        /// rbs `TypeDef#defined_in` of the resolved signature, fully
        /// qualified via `NameTable::display_type_name`. `None` only for
        /// serialize-compat (JSON key omission); every emit site resolves
        /// a concrete owner.
        defined_in: Option<String>,
    },
    InsufficientPositionalArguments {
        method_name: String,
        expected: usize,
        actual: usize,
        defined_in: Option<String>,
    },
    UnexpectedPositionalArgument {
        method_name: String,
        expected: usize,
        actual: usize,
        defined_in: Option<String>,
    },
    UnexpectedKeywordArgument {
        method_name: String,
        keyword: String,
        defined_in: Option<String>,
    },
    UnexpectedTypeArgument {
        method_name: String,
        method_type: String,
        type_arg: String,
    },
    InsufficientTypeArguments {
        method_name: String,
        method_type: String,
        expected: usize,
        actual: usize,
    },
    MethodBodyTypeMismatch {
        method_name: String,
        expected: String,
        actual: String,
    },
    InsufficientKeywordArguments {
        method_name: String,
        keyword: String,
        defined_in: Option<String>,
    },
    BlockBodyTypeMismatch {
        method_name: String,
        expected: String,
        actual: String,
    },
    RequiredBlockMissing {
        method_name: String,
    },
    /// `receiver.method` where `receiver`'s type has no such method.
    /// Mirrors Steep's `Ruby::NoMethod` (`steep/lib/steep/diagnostic/ruby.rb`
    /// `class NoMethod`).
    ///
    /// Display intentionally omits the receiver type that Steep embeds
    /// (Steep: `"Type `#{type}` does not have method `#{method}`"`);
    /// `receiver_type` is exposed as a structured JSON key only so the
    /// message stays role-only and avoids duplicating long unions
    /// (dogfood against rbs/lib saw 21-member, 741-char receivers).
    NoMethod {
        method_name: String,
        receiver_type: String,
        /// For a `Type::Union` receiver, the display strings of members
        /// that lack the method (one per missing member). Empty for
        /// non-union receivers, intersection receivers (where every
        /// member lacks the method by construction so the list would be
        /// redundant with `receiver_type`), and the union
        /// block-survives-drop sub-case where every member carries the
        /// method but their block clauses fail to combine. The empty
        /// case is omitted from JSON output so consumers can rely on
        /// the key's presence as a signal that member-level info is
        /// available. Each entry comes from the same `display_type`
        /// path that produces `receiver_type`, so the strings are
        /// directly comparable as substrings of `receiver_type` (no
        /// alternate formatting). Steep `Ruby::NoMethod` (`type +
        /// method` only) has no equivalent — crema extends here for
        /// narrowing-fix hints.
        missing_from: Vec<String>,
    },
    ClassModuleMismatch {
        name: String,
        ruby_kind: String,
        rbs_kind: String,
    },
    MethodArityMismatch {
        method_name: String,
        ruby_arity: String,
        rbs_arity: String,
    },
    /// `def foo(...); bar(...); end` where the caller's forwarded
    /// signature does not match the callee's parameter list. Mirrors
    /// Steep's `Ruby::IncompatibleArgumentForwarding`
    /// (`lib/steep/diagnostic/ruby.rb` `IncompatibleArgumentForwarding`).
    /// `caller_signature` / `callee_signature` show the two signatures
    /// involved (formatted to mirror Steep's display); `mismatch_kind`
    /// distinguishes arity vs element-type failure so downstream tools
    /// can filter by category without parsing the message.
    IncompatibleArgumentForwarding {
        method_name: String,
        caller_signature: String,
        callee_signature: String,
        mismatch_kind: ForwardingMismatchKind,
    },
    MethodParameterMismatch {
        method_name: String,
        param_name: String,
        ruby_kind: String,
        rbs_kind: String,
    },
    /// A Ruby parameter has a different kind from the RBS declaration where
    /// the Ruby side is the looser one (optarg / restarg / kwoptarg / kwrestarg).
    /// Mirrors Steep's `Ruby::DifferentMethodParameterKind`
    /// (`steep/lib/steep/diagnostic/ruby.rb` `class DifferentMethodParameterKind`).
    /// Kept distinct from [`MethodParameterMismatch`] so downstream tools can
    /// filter by the Steep classification (Ruby strict vs loose).
    DifferentMethodParameterKind {
        method_name: String,
        param_name: String,
        ruby_kind: String,
        rbs_kind: String,
    },
    /// A call whose receiver has multiple overloads but none matches the
    /// supplied argument types. Mirrors Steep's `Ruby::UnresolvedOverloading`
    /// (`steep/lib/steep/diagnostic/ruby.rb` `class UnresolvedOverloading`).
    ///
    /// Display intentionally omits the receiver type that Steep embeds
    /// (Steep: `"Cannot find compatible overloading of method `#{method_name}`
    /// of type `#{receiver_type}`"`); `receiver_type` is exposed as a
    /// structured JSON key only, mirroring `NoMethod`'s rationale.
    ///
    /// `call_arguments` is a crema-only extension — Steep's
    /// `UnresolvedOverloading` carries only `method_types`, leaving the
    /// caller to read the call site themselves via IDE hover. crema has
    /// no hover, so the call shape is rendered as an rbs-style signature
    /// fragment (e.g. `(::String, base: ::Pathname)`, `(::Integer) { ... }`)
    /// so AI agents can compare it letter-for-letter against the overload
    /// list without re-reading the source. `method_types` lists every
    /// overload of the resolved method without `def NAME:` prefix — the
    /// `def NAME:` head is reattached once by the text Display so JSON
    /// consumers stay free to reformat. Both keys are filled at the four
    /// emit sites in `type_checker/calls.rs`; union receivers flatten
    /// every component's overloads into a single list.
    UnresolvedOverloading {
        method_name: String,
        receiver_type: String,
        call_arguments: String,
        method_types: Vec<String>,
    },
    UnknownTupleIndex {
        index: i64,
        tuple_length: usize,
    },
    UnknownRecordKey {
        key: String,
        known_keys: Vec<String>,
    },
    TypeArgumentBoundViolation {
        container_name: String,
        param_name: String,
        bound_kind: BoundKind,
        bound: String,
        actual: String,
    },
    /// The type text of a `#: T` / `#[T]` / `# @rbs ...` annotation is
    /// syntactically broken.
    /// Mirrors `RBS::InlineParser::Diagnostic::AnnotationSyntaxError`
    /// (`lib/rbs/inline_parser.rb` L38). See ADR-0011.
    AnnotationSyntaxError {
        /// The annotation as written in the source (e.g. `"#: Strng<>>"`).
        annotation_text: String,
        /// Underlying rbs C parser message, if the wrapper could capture it.
        /// `None` means only `annotation_text` is shown — the generic
        /// "Syntax error in inline annotation" wording still surfaces the
        /// problem, just without the parser-level detail.
        parser_error: Option<String>,
    },
    /// `Foo = some_function() #: class-alias` / `#: module-alias` —
    /// the annotation has no explicit type-name argument and the
    /// right-hand side is not a constant, so no alias target can be
    /// inferred. Mirrors
    /// `RBS::InlineParser::Diagnostic::ClassModuleAliasDeclarationMissingTypeName`.
    InlineClassAliasMissingTypeName {
        /// Whether the annotation was `#: class-alias` (`Class`) or
        /// `#: module-alias` (`Module`); chooses the diagnostic message
        /// wording.
        kind: InlineAliasKind,
    },
    /// `include A, B` / `extend A, B` / `prepend A, B`.
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::MixinMultipleArguments`.
    MixinMultipleArguments,
    /// A private method was called with an explicit non-self receiver.
    ///
    /// Kept distinct from `NoMethod` (Steep collapses both) so downstream
    /// agents can grep by category without parsing the message.
    PrivateMethodCall {
        method_name: String,
        receiver_type: String,
    },
    /// The class declaration path `class <expr>::Foo` contains a dynamic part.
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::NonConstantClassName`
    /// (`lib/rbs/inline_parser.rb` L107).
    NonConstantClassName,
    /// The module declaration path `module <expr>::Foo` contains a dynamic part.
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::NonConstantModuleName`
    /// (`lib/rbs/inline_parser.rb` L129).
    NonConstantModuleName,
    /// The superclass expression in `class A < <expr>` is not a constant path.
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::NonConstantSuperClassName`
    /// (`lib/rbs/inline_parser.rb` L537).
    NonConstantSuperClassName,
    /// A leading or trailing inline annotation exists but is not consumed by
    /// the method definition (e.g. parameter name mismatch, shadowed by an
    /// explicit method-type annotation, or written as trailing where only a
    /// `T` assertion is accepted).
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::UnusedInlineAnnotation`
    /// (`lib/rbs/inline_parser.rb` L37, L506-519`).
    UnusedInlineAnnotation,
    /// A top-level inline Ruby method definition has no RBS declaration target.
    ///
    /// Mirrors `RBS::InlineParser::Diagnostic::TopLevelMethodDefinition`.
    TopLevelMethodDefinition,
    /// `class << self` inside another inline `class << self`, or `def self.foo`
    /// inside inline `class << self`, targets the singleton class of a
    /// singleton class, which RBS cannot represent.
    NestedSingletonScope,
    /// Top-level inline `class << self` opens the singleton class of `main`,
    /// which RBS cannot represent.
    TopLevelSingletonScope,
    /// Inline `class << expr` where `expr` is not `self` opens an anonymous
    /// singleton class, which RBS cannot represent.
    NonSelfSingletonScope,
    /// Mirrors `RBS::DuplicatedMethodDefinitionError` (`lib/rbs/errors.rb` L251).
    /// `overloading: true` entries are exempt (intentional multi-overload merge).
    DuplicatedMethodDefinition {
        method_name: String,
        /// The other side of the collision: a real location (rbs's
        /// `original_member.location`) the user can jump to, or crema's
        /// own infusion synthesis when the colliding member has no file
        /// (`mid_dup_method_infusion_provenance`). The redefining site
        /// is carried by the surrounding `Diagnostic.location`;
        /// `duplicate_source` always points at the *other* side, never
        /// at itself.
        duplicate_source: Option<DuplicateSource>,
    },
    /// Mirrors `RBS::DuplicatedDeclarationError` for declaration-kind
    /// conflicts detected while loading inline Ruby declarations.
    DuplicatedDeclaration {
        name: String,
        detail: String,
        conflicting_source: Option<SourceLocation>,
    },
    /// Mirrors `RBS::RecursiveAliasDefinitionError` (`lib/rbs/errors.rb` L448).
    /// Reported for alias chains that loop back on themselves: a self-loop
    /// (`alias a a`) or a multi-step SCC (`alias a b; alias b a`,
    /// `alias a b; alias b c; alias c a`, …). rbs raises a single exception
    /// per cycle covering all participants; crema follows the same surface
    /// (1 diagnostic per cycle, `alias_names` lists every participant).
    RecursiveAliasDefinition {
        /// Fully-qualified owner type (e.g. `"::Foo"`).
        type_name: String,
        /// Names of every alias entry that participates in the cycle.
        /// One entry for a self-loop, two-plus for a multi-step SCC.
        /// Order follows the bucket-walk order of
        /// `method_builder::Sorter::each_strongly_connected_component`.
        alias_names: Vec<String>,
        /// Location of the cycle's first alias declaration. Mirrors rbs's
        /// `RecursiveAliasDefinitionError#location` (= `first_def.original.location`).
        primary_source: Option<SourceLocation>,
    },
    /// Mirrors `RBS::RecursiveTypeAliasError` (`lib/rbs/errors.rb` L527).
    /// Emitted by the build-layer validator when one or more type alias
    /// declarations form a cycle on the rbs `direct_dependency` graph
    /// (Union / Intersection / Optional are transparent, every other
    /// constructor is opaque — `type a = Array[a]` is regular, but
    /// `type a = a?` is not). rbs raises a separate exception per alias
    /// entry that participates in a cycle; crema collapses every cycle to
    /// a single diagnostic anchored on its lexicographically smallest
    /// participant, matching the surface of [`Self::RecursiveAncestor`].
    RecursiveTypeAlias {
        /// Names of every alias entry that participates in the cycle,
        /// sorted lexicographically. One entry for a self-loop
        /// (`type a = a`), two-plus for a multi-step SCC
        /// (`type a = b; type b = a`).
        alias_names: Vec<String>,
        /// Location of the anchor alias's `name_location`
        /// (= `alias_names[0]`'s declaration). Mirrors rbs's
        /// `RecursiveTypeAliasError#location`.
        primary_source: Option<SourceLocation>,
    },
    /// Mirrors `RBS::RecursiveAncestorError` (`lib/rbs/errors.rb` L110).
    /// Emitted by the build-layer validator when the ancestor graph
    /// (super / include / prepend / self_types) of any declared class or
    /// module contains a cycle. rbs raises one error per class whose
    /// `instance_ancestors` build hits the cycle; crema collapses every
    /// cycle to a single diagnostic so noise stays low for AI consumers
    /// (the same cycle would otherwise emit one diagnostic per class
    /// transitively involved — e.g. `Object include Foo` triggers it for
    /// every class in the standard library).
    RecursiveAncestor {
        /// Fully-qualified anchor name (the cycle participant with the
        /// lexicographically smallest `TypeName`). Used by `method_name()`
        /// so the diagnostic carries a stable identifier without needing a
        /// new accessor.
        type_name: String,
        /// Cycle path expressed as fully-qualified names, closed by
        /// repeating the first participant at the end. Matches rbs's
        /// `Detected recursive ancestors: ::A < ::B < ::A` message format.
        chain: Vec<String>,
        /// Location of the anchor's primary declaration. Mirrors rbs's
        /// `RecursiveAncestorError#location` (= `entry.primary_decl.location`).
        primary_source: Option<SourceLocation>,
    },
    InstanceVariableDuplication {
        type_name: String,
        variable_name: String,
        duplicate_source: Option<SourceLocation>,
    },
    ClassInstanceVariableDuplication {
        type_name: String,
        variable_name: String,
        duplicate_source: Option<SourceLocation>,
    },
    /// `include M[X, Y]`, `class A < B[X]`, etc. supplies a wrong number of
    /// type arguments. Emitted by the build-layer validator (ADR-0013)
    /// against both `.rbs` and inline origins, distinguished only by
    /// location. `expected` is either "N" or "N..M" (default-typed tail).
    MixinTypeArgumentArityMismatch {
        /// "superclass" | "include" | "extend" | "prepend"
        kind: String,
        /// Fully-qualified name of the mixin / superclass target (e.g. "::Enumerable").
        target: String,
        /// Fully-qualified name of the class/module doing the mixin (e.g. "::Bag").
        class: String,
        /// Expected arity as a human string: "1" or "1..2".
        expected: String,
        /// Actual number of type arguments supplied.
        got: usize,
    },
    /// A type name referenced from the build layer (alias rhs today;
    /// later super class / mixin / module-self / type references) is
    /// not declared anywhere in the environment. Emitted by the
    /// build-layer validator (ADR-0013).
    ///
    /// Mirrors Steep's `Diagnostic::Signature::UnknownTypeName`
    /// (`Cannot find type \`X\``), which collapses RBS's
    /// `NoTypeFoundError` / `NoSuperclassFoundError` /
    /// `NoMixinFoundError` / `NoSelfTypeFoundError` into a single
    /// diagnostic. crema follows the same collapse. The originating
    /// context (alias / super / mixin / self-type) is identifiable
    /// from `file:line`; this variant carries only the dangling name
    /// to match Steep's surface.
    UnknownTypeName {
        /// Dangling name (absolute when written with `::` or resolved
        /// against an enclosing context; relative when left as a raw
        /// segment by `TypeNameResolver::try_resolve`).
        name: String,
    },
    /// `x = expr #: T` where neither `actual <: T` nor `T <: actual`
    /// (with `actual` = the hint-applied type of `expr`).
    ///
    /// Mirrors Steep's `Diagnostic::Ruby::FalseAssertion`
    /// (`lib/steep/diagnostic/ruby.rb`). Reported at `Severity::Hint`
    /// per Steep default.
    FalseAssertion {
        /// Hint-applied (post-bidirectional) type of the assigned expression.
        natural: String,
        /// Asserted type from `#: T`.
        asserted: String,
    },
    /// The type of an expression could not be determined and fell back to
    /// `untyped`. Mirrors Steep's `Diagnostic::Ruby::FallbackAny`
    /// (`lib/steep/type_construction.rb` `fallback_to_any` helper).
    ///
    /// Reported at `Severity::Hint` matching Steep's default config.
    /// crema extends the message with a parenthesized `reason` so
    /// AI agents can identify the exact cause without re-running.
    FallbackAny {
        /// Human-readable cause of the untyped fallback.
        reason: String,
    },
    /// A `yield` expression appears in a method that has no block parameter
    /// in its RBS declaration. Mirrors Steep's `Ruby::UnexpectedYield`.
    UnexpectedYield {
        method_name: String,
    },
    /// `super` / `super(args)` whose target cannot be resolved on the
    /// receiver's ancestor chain above the defining class. Mirrors Steep's
    /// `Ruby::UnexpectedSuper` (`steep/lib/steep/diagnostic/ruby.rb`
    /// `class UnexpectedSuper`), reported at `Severity::Information` to
    /// match Steep's `default` preset (`UnexpectedSuper => :information`).
    UnexpectedSuper {
        method_name: String,
    },
    /// The lower bound (from seed arg) and upper bound (from trailing hint)
    /// for a method-level type parameter have no subtype relation in either
    /// direction, so no consistent type can be assigned.
    ///
    /// Mirrors Steep's `Ruby::UnsatisfiableConstraint`.
    UnsatisfiableConstraint {
        /// Human-readable display of the lower bound (seed-derived).
        lower: String,
        /// Human-readable display of the upper bound (hint-derived).
        upper: String,
        /// Name of the conflicting type parameter (e.g. `"U"`).
        type_param: String,
        /// Display of the overload's method type signature (context for the reader).
        method_type: String,
    },
    /// A constant does not resolve against the lexical scope + ancestor
    /// chain. Mirrors Steep's `Diagnostic::Ruby::UnknownConstant`
    /// (`steep/lib/steep/diagnostic/ruby.rb`,
    /// `Cannot find the declaration of #{kind}: \`#{name}\``). `kind`
    /// tracks Steep's `@kind` symbol: a bare read keeps `:constant`,
    /// while a class declaration name / superclass uses `.class!` and a
    /// module declaration name uses `.module!`.
    UnknownConstant {
        /// Unresolved constant name as written (`Foo`).
        name: String,
        /// Full constant expression as written, reconstructed from path
        /// segments (`Foo::Bar` or `::Foo::Bar`).
        path: String,
        /// What the occurrence is: a bare read (`Constant`), a class
        /// declaration name or superclass (`Class`), or a module
        /// declaration name (`Module`). Selects the message wording.
        kind: ConstantKind,
        /// Typo suggestions drawn from the constant names in scope at the
        /// failure site, best match first (see `crate::spell_checker`).
        /// ADR-0032 Decision 5a: check only records the *fact* that this
        /// name failed to resolve (`name` + `candidate_scope`); this
        /// field starts empty and is filled by [`join_did_you_mean`] at
        /// the output boundary, computed fresh against the env in scope
        /// there. A diagnostic that skips the join (e.g. an internal
        /// helper that doesn't call it) legitimately has this empty —
        /// callers that need suggestions must run the join.
        did_you_mean: Vec<String>,
        /// Namespaces walked for the failed lookup. A top-level lexical
        /// context is represented as `"::"`. Unlike `did_you_mean`, this
        /// is a fact of the failure site's own lexical context (fixed by
        /// the file's own source), not the env, so it stays computed at
        /// check time.
        searched_namespaces: Vec<String>,
        /// Where to draw `did_you_mean` candidates from, computed once at
        /// check time from the failure site's context and carried as data
        /// so [`join_did_you_mean`] can recompute suggestions against any
        /// env — including a fresh one at replay time, not just the env
        /// that was in scope when this diagnostic was first produced
        /// (ADR-0032 Decision 5a's "fact vs. computed column" split).
        candidate_scope: CandidateScope,
    },

    /// `case x; when A; when B; end` left `x` partially uncovered.
    /// Crema's side-effect-free exhaustiveness check (ADR-0022): once
    /// every `when` resolves to a Class/Module narrow target, the
    /// subtract residue of the scrutinee is the set of values no branch
    /// would catch. If that residue is non-empty and no `else` clause is
    /// present, this fires. Steep does not emit this diagnostic; crema
    /// surfaces it at `Hint` by default so the gap is visible during
    /// preflight without failing the check — dogfooding may promote it
    /// later if false-positive risk stays low.
    NonExhaustiveCase {
        /// Scrutinee's full type display (e.g. `::Integer | ::String | ::Symbol`).
        /// Kept alongside `residue_type` so the reader can distinguish the
        /// original union from the leftover, instead of inferring it from
        /// the unrelated source code.
        scrutinee_type: String,
        /// Residue type display (e.g. `::Symbol`, `(::Symbol | nil)`).
        residue_type: String,
    },

    /// `case x; when A; when B; else; ... end` covered every value of
    /// `x` already, so the `else` branch is statically unreachable
    /// (ADR-0022 § exhaustiveness). Mirrors Steep's
    /// `Ruby::UnreachableValueBranch`; `Hint` by default to match
    /// Steep's `default` preset.
    UnreachableValueBranch,

    /// RHS of an ivar / cvar / gvar assignment is incompatible with the
    /// variable's declared type (e.g. `@x: Integer` with `@x = "string"`).
    /// Mirrors Steep's `Diagnostic::Ruby::IncompatibleAssignment`
    /// (`lib/steep/type_construction.rb` `ivasgn` / `gvasgn` / `cvasgn`).
    ///
    /// Inline `#: T` assertion does not override the declaration here —
    /// the ivar/cvar/gvar declaration is always the source of truth
    /// (Steep parity, measured 2026-06-06). lvar assertion override
    /// stays under `FalseAssertion`.
    ///
    /// Default severity = `Error` (stricter than Steep's `Hint` default,
    /// matching crema's `UnknownConstant` policy: preflight signals worth
    /// failing on).
    IncompatibleAssignment {
        /// Display of the declared (LHS) type, e.g. `::Integer`.
        lhs_type: String,
        /// Display of the inferred RHS type, e.g. `::String`.
        rhs_type: String,
    },

    /// Multiple assignment (`a, b = rhs`) where the RHS's `to_ary`
    /// resolves but returns a non-expandable type (neither a Tuple nor
    /// `Array[T]`, e.g. `Integer`). Ruby itself raises `TypeError` at
    /// runtime via `rb_check_array_type`, so the silent scalar
    /// fallback would hide a real bug. Mirrors Steep's
    /// `Ruby::MultipleAssignmentConversionError`
    /// (`steep/lib/steep/diagnostic/ruby.rb` `MultipleAssignmentConversionError`);
    /// `Error` suffix dropped per crema naming convention.
    ///
    /// When emitted, all LHS targets (leading / splat / trailing) are
    /// bound to `untyped` to suppress cascading `NilClass` NoMethod
    /// diagnostics on subsequent uses.
    MultipleAssignmentConversion {
        /// Display of the RHS value type (Steep's `original_type`), e.g.
        /// `::BadAry`.
        original_type: String,
        /// Display of the type returned by `to_ary` (Steep's
        /// `returned_type`), e.g. `::Integer`.
        returned_type: String,
    },

    /// Write to an instance variable that has no declaration on the
    /// current `self` (instance-side or singleton-side ivar bucket on
    /// the linearized ancestor chain). Mirrors Steep's
    /// `Diagnostic::Ruby::UnknownInstanceVariable`.
    ///
    /// crema's read side currently silently falls back to `untyped` for
    /// undeclared ivars; this asymmetry is intentional and tracked as a
    /// follow-up (see todo `Scope outside`).
    UnknownInstanceVariable {
        /// Ivar name as written, including the leading `@` (e.g. `@x`).
        name: String,
    },

    /// Write to a class variable with no declaration on the current
    /// `self`'s class-variable bucket. crema-only diagnostic — Steep has
    /// no equivalent (`Diagnostic::Ruby::UnknownClassVariable` does not
    /// exist in `steep/lib/steep/diagnostic/ruby.rb`); Steep does not
    /// check class variable references at all.
    UnknownClassVariable {
        /// Class variable name including the leading `@@` (e.g. `@@x`).
        name: String,
    },

    /// Write to a global variable that has no declaration in any loaded
    /// RBS signature. Mirrors Steep's
    /// `Diagnostic::Ruby::UnknownGlobalVariable`.
    UnknownGlobalVariable {
        /// Global variable name including the leading `$` (e.g. `$x`).
        name: String,
    },
    /// A reference (method call / constant read / global variable read
    /// or write) whose backing RBS declaration carries a
    /// `%a{deprecated}` — optionally `%a{deprecated: <message>}` —
    /// annotation. Mirrors Steep's `Ruby::DeprecatedReference`
    /// (`lib/steep/diagnostic/ruby.rb` `class DeprecatedReference`),
    /// reported at `Severity::Warning` by default (Steep parity).
    ///
    /// `subject_kind` selects the message wording (`method` /
    /// `constant` / `global_variable`), matching Steep's
    /// `header_line` node-type dispatch. `subject_name` is the name
    /// as written at the reference site (bare identifier for method,
    /// constant name for constant, `$name` for gvar). `message`
    /// carries the trailer from `%a{deprecated: msg}`; `None` when
    /// the annotation was bare.
    ///
    /// For methods, both member-level (`Method.annotations`) and
    /// per-overload (`TypeDef.overload_annotations`) sites are
    /// considered — the overload-level check runs only on the
    /// narrowed overload that the call actually resolved to, so
    /// unmatched deprecated overloads stay silent (Steep parity).
    /// Emitted at most once per reference (member vs overload does
    /// not double-fire).
    DeprecatedReference {
        subject_kind: DeprecatedSubjectKind,
        subject_name: String,
        message: Option<String>,
    },
    InfusionProviderSkipped {
        provider: String,
        subject: String,
        reason: String,
    },

    /// A `[infusion.config]` YAML file declares the same mapping key more
    /// than once. crema follows Ruby's Psych (last-wins) instead of
    /// aborting, and reports this at `Severity::Warning` so the run still
    /// surfaces every other diagnostic. `key` is the duplicated key name;
    /// `file` is the config file it appeared in.
    ///
    /// The location is pinned to the file's start, not the duplicate key's
    /// line: `serde_yaml`'s generic `Visitor` does not expose byte offsets
    /// for parsed values, so a precise position is not available here.
    DuplicatedConfigKey {
        key: String,
        file: String,
    },

    /// Prism reported a `.rb` source as syntactically invalid. The CLI
    /// driver emits exactly one of these per file (using the first prism
    /// error's location and message) and then skips inline annotation
    /// collection and type checking for that file. `message` carries
    /// Prism's human-readable error string verbatim; the crema-side
    /// `Display` impl prefixes it with `Syntax error:` so the JSON
    /// `message` field documents that the file was rejected before any
    /// type analysis ran. Pinned to `Severity::Error` in every preset
    /// via `DiagnosticKind::all_ignore_severity_for_code` (the only
    /// preset that would otherwise demote it).
    SyntaxError {
        message: String,
    },

    /// Dev-only marker for type-checker silent fall-through points that
    /// have not been instrumented as a real diagnostic yet. Emitted at
    /// `_ => return` / `_ => None` arms where crema currently gives up on
    /// an unsupported variant. `site` names the instrument location and
    /// `subject_display` carries the unsupported subject so the crema
    /// developer can pinpoint the gap without re-running with
    /// `--verbose`.
    ///
    /// Defaults to `Severity::Ignore` in every preset (including
    /// `all_error`) — see `DiagnosticKind::is_dev_only_code` and the
    /// guard in `Preset::severity_for_code`. Surfaces only when the user
    /// explicitly sets `[diagnostic] "Crema::NotImplementedYet" = "..."`.
    /// Expected to be removed once the silent fall-through points are
    /// exhausted; this is not a permanent diagnostic kind.
    NotImplementedYet {
        /// Instrument location identifier (e.g. `"check_no_method"`,
        /// `"infer_type"`).
        site: String,
        /// Display string for the unsupported subject — usually the
        /// receiver type (`"RBS::Types::t"`) for `check_no_method`
        /// sites, or the Prism node variant name
        /// (`"LocalVariableWriteNode"`) for `infer_type` sites.
        subject_display: String,
    },
}

/// Mirrors Steep's `UnknownConstant#kind` symbol
/// (`:constant` / `:class` / `:module`, set via `.class!` / `.module!`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConstantKind {
    Constant,
    Class,
    Module,
}

impl ConstantKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ConstantKind::Constant => "constant",
            ConstantKind::Class => "class",
            ConstantKind::Module => "module",
        }
    }
}

/// Where to draw [`DiagnosticKind::UnknownConstant`]'s `did_you_mean`
/// candidates from, carried on the diagnostic itself so suggestion
/// computation can be deferred to output time instead of running during
/// check (ADR-0032 Decision 5a). A head segment / bare read looks in the
/// lexical `Context`; a path child segment looks in its parent module's
/// `Children`; an intermediate-value segment has no module to enumerate,
/// so `None`. `TypeName` is a process-stable id (ADR-0032 Decision 2), so
/// this stays meaningful against a `DefinitionBuilder` built later than
/// the one in scope when the diagnostic was first produced.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CandidateScope {
    Context(ConstantContext),
    Children(TypeName),
    None,
}

/// Compute `did_you_mean` suggestions for one unresolved name against
/// `env`'s current constant table. Free function (rather than a
/// `DefinitionBuilder` method) so it stays next to `CandidateScope` and
/// `join_did_you_mean`, its only caller.
fn candidates_for_scope(
    env: &DefinitionBuilder,
    input: &str,
    scope: &CandidateScope,
) -> Vec<String> {
    let resolver = env.constant_resolver();
    let dict: Vec<crate::name::Symbol> = match scope {
        CandidateScope::Context(ctx) => resolver.constants(ctx).keys().copied().collect(),
        CandidateScope::Children(tn) => resolver.children(tn).keys().copied().collect(),
        CandidateScope::None => return Vec::new(),
    };
    let names = env.env().names();
    let dictionary: Vec<String> = dict.into_iter().map(|s| names.resolve(s)).collect();
    crate::spell_checker::correct(input, &dictionary)
}

/// Join phase (ADR-0032 Decision 5a): fill in `did_you_mean` on every
/// `UnknownConstant` diagnostic from its `candidate_scope`, computed
/// fresh against `env`. The single function both the CLI check loop
/// (`main.rs`, right before `emitter.emit`) and the test helpers run, so
/// a diagnostic's suggestions are always true to the *current* env
/// regardless of whether it came from a fresh check or a cached replay.
/// A no-op on every other diagnostic kind.
pub fn join_did_you_mean(env: &DefinitionBuilder, diagnostics: &mut [Diagnostic]) {
    for diag in diagnostics.iter_mut() {
        if let DiagnosticKind::UnknownConstant {
            name,
            did_you_mean,
            candidate_scope,
            ..
        } = &mut diag.kind
        {
            *did_you_mean = candidates_for_scope(env, name, candidate_scope);
        }
    }
}

/// Subject discriminator for [`DiagnosticKind::DeprecatedReference`].
/// Selects the message wording (`method` / `constant` /
/// `global_variable`), mirroring Steep's `header_line` node-type
/// dispatch (`lib/steep/diagnostic/ruby.rb` `class DeprecatedReference`
/// lines 1063-1080).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeprecatedSubjectKind {
    Method,
    Constant,
    GlobalVariable,
}

impl DeprecatedSubjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeprecatedSubjectKind::Method => "method",
            DeprecatedSubjectKind::Constant => "constant",
            DeprecatedSubjectKind::GlobalVariable => "global_variable",
        }
    }

    /// Display noun used in the diagnostic message. Uses a space
    /// (`"global variable"`) rather than the JSON-friendly underscore
    /// form. Matches Steep's `header_line` wording.
    pub fn display_noun(self) -> &'static str {
        match self {
            DeprecatedSubjectKind::Method => "method",
            DeprecatedSubjectKind::Constant => "constant",
            DeprecatedSubjectKind::GlobalVariable => "global variable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BoundKind {
    Upper,
    Lower,
}

impl BoundKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BoundKind::Upper => "upper",
            BoundKind::Lower => "lower",
        }
    }
}

/// Whether `bar(...)` forwarding to a callee failed by arity (too many
/// or too few positional / keyword slots), by element type (a
/// caller-side slot is not a subtype of the callee-side slot), or by
/// block incompatibility (the caller's block, viewed as a proc-or-nil
/// type, is not a subtype of the callee's). Mirrors Steep's message
/// split in `lib/steep/diagnostic/ruby.rb` (`Cannot forward arguments`
/// vs `Cannot forward block`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ForwardingMismatchKind {
    Arity,
    Type,
    Block,
}

impl ForwardingMismatchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ForwardingMismatchKind::Arity => "arity",
            ForwardingMismatchKind::Type => "type",
            ForwardingMismatchKind::Block => "block",
        }
    }
}

impl Diagnostic {
    pub fn at(location: SourceLocation, kind: DiagnosticKind) -> Self {
        debug_assert!(
            location.range.start_byte <= location.range.end_byte,
            "diagnostic range must not be reversed: {:?}",
            location.range
        );
        Self {
            kind,
            location,
            scope: None,
        }
    }

    pub fn location_for_byte_range(
        file: PathBuf,
        source: &[u8],
        start_byte: usize,
        end_byte: usize,
    ) -> SourceLocation {
        let start_byte = start_byte.min(source.len());
        let end_byte = end_byte.min(source.len());
        let start_char = byte_to_char_offset(source, start_byte);
        let end_char = byte_to_char_offset(source, end_byte);
        SourceLocation {
            file,
            range: LocationRange::new(start_char, start_byte as u32, end_char, end_byte as u32),
        }
    }

    pub fn method_name(&self) -> &str {
        match &self.kind {
            DiagnosticKind::ArgumentTypeMismatch { method_name, .. }
            | DiagnosticKind::InsufficientPositionalArguments { method_name, .. }
            | DiagnosticKind::UnexpectedPositionalArgument { method_name, .. }
            | DiagnosticKind::UnexpectedKeywordArgument { method_name, .. }
            | DiagnosticKind::UnexpectedTypeArgument { method_name, .. }
            | DiagnosticKind::InsufficientTypeArguments { method_name, .. }
            | DiagnosticKind::MethodBodyTypeMismatch { method_name, .. }
            | DiagnosticKind::InsufficientKeywordArguments { method_name, .. }
            | DiagnosticKind::BlockBodyTypeMismatch { method_name, .. }
            | DiagnosticKind::RequiredBlockMissing { method_name, .. }
            | DiagnosticKind::NoMethod { method_name, .. }
            | DiagnosticKind::PrivateMethodCall { method_name, .. } => method_name,
            DiagnosticKind::ClassModuleMismatch { name, .. } => name,
            DiagnosticKind::MethodArityMismatch { method_name, .. }
            | DiagnosticKind::MethodParameterMismatch { method_name, .. }
            | DiagnosticKind::DifferentMethodParameterKind { method_name, .. }
            | DiagnosticKind::UnresolvedOverloading { method_name, .. }
            | DiagnosticKind::IncompatibleArgumentForwarding { method_name, .. }
            | DiagnosticKind::DuplicatedMethodDefinition { method_name, .. } => method_name,
            DiagnosticKind::RecursiveAliasDefinition { type_name, .. } => type_name,
            DiagnosticKind::RecursiveTypeAlias { alias_names, .. } => alias_names
                .first()
                .map(String::as_str)
                .unwrap_or("<recursive-type-alias>"),
            DiagnosticKind::RecursiveAncestor { type_name, .. } => type_name,
            DiagnosticKind::InstanceVariableDuplication { variable_name, .. }
            | DiagnosticKind::ClassInstanceVariableDuplication { variable_name, .. } => {
                variable_name
            }
            DiagnosticKind::UnknownTupleIndex { .. } | DiagnosticKind::UnknownRecordKey { .. } => {
                "[]"
            }
            DiagnosticKind::TypeArgumentBoundViolation { container_name, .. } => container_name,
            DiagnosticKind::AnnotationSyntaxError { .. } => "<inline-annotation>",
            DiagnosticKind::InlineClassAliasMissingTypeName { .. } => "<inline-annotation>",
            DiagnosticKind::MixinMultipleArguments => "<inline-annotation>",
            DiagnosticKind::NonConstantClassName => "<class-name>",
            DiagnosticKind::NonConstantModuleName => "<module-name>",
            DiagnosticKind::NonConstantSuperClassName => "<superclass>",
            DiagnosticKind::UnusedInlineAnnotation => "<inline-annotation>",
            DiagnosticKind::TopLevelMethodDefinition => "<top-level-method>",
            DiagnosticKind::NestedSingletonScope
            | DiagnosticKind::TopLevelSingletonScope
            | DiagnosticKind::NonSelfSingletonScope => "<singleton-scope>",
            DiagnosticKind::MixinTypeArgumentArityMismatch { target, .. } => target,
            DiagnosticKind::DuplicatedDeclaration { name, .. } => name,
            DiagnosticKind::UnknownTypeName { name } => name,
            DiagnosticKind::FalseAssertion { .. } => "<inline-annotation>",
            DiagnosticKind::FallbackAny { .. } => "<yield>",
            DiagnosticKind::UnexpectedYield { method_name } => method_name,
            DiagnosticKind::UnexpectedSuper { method_name } => method_name,
            DiagnosticKind::UnsatisfiableConstraint { type_param, .. } => type_param,
            DiagnosticKind::UnknownConstant { name, .. } => name,
            DiagnosticKind::NonExhaustiveCase { .. } => "<case>",
            DiagnosticKind::UnreachableValueBranch => "<else>",
            DiagnosticKind::IncompatibleAssignment { .. } => "<assignment>",
            DiagnosticKind::MultipleAssignmentConversion { .. } => "<multi-assign>",
            DiagnosticKind::UnknownInstanceVariable { name } => name,
            DiagnosticKind::UnknownClassVariable { name } => name,
            DiagnosticKind::UnknownGlobalVariable { name } => name,
            DiagnosticKind::DeprecatedReference { subject_name, .. } => subject_name,
            DiagnosticKind::InfusionProviderSkipped { subject, .. } => subject,
            DiagnosticKind::DuplicatedConfigKey { key, .. } => key,
            DiagnosticKind::NotImplementedYet { .. } => "<crema-internal>",
            DiagnosticKind::SyntaxError { .. } => "<syntax-error>",
        }
    }

    /// Produce a structured JSON value for machine consumption.
    /// `severity` is supplied by the emitter after resolving any
    /// `[diagnostic]` config override, since the diagnostic itself no
    /// longer carries one.
    pub fn to_json_value(&self, severity: Severity) -> Value {
        self.to_json_value_with_line(severity, self.fallback_line())
    }

    fn to_json_value_with_line(&self, severity: Severity, line: usize) -> Value {
        let mut obj = Map::new();
        obj.insert("file".into(), json!(self.location.file.to_string_lossy()));
        obj.insert("line".into(), json!(line));
        obj.insert("start_byte".into(), json!(self.location.range.start_byte));
        obj.insert("end_byte".into(), json!(self.location.range.end_byte));
        obj.insert("severity".into(), json!(severity.as_str()));
        obj.insert("code".into(), json!(self.kind.code()));
        obj.insert("message".into(), json!(self.kind.to_string()));

        // Add type-specific fields when available
        match &self.kind {
            DiagnosticKind::ArgumentTypeMismatch {
                method_name,
                expected,
                actual,
                defined_in,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("expected".into(), json!(expected));
                obj.insert("actual".into(), json!(actual));
                if let Some(defined_in) = defined_in {
                    obj.insert("defined_in".into(), json!(defined_in));
                }
            }
            DiagnosticKind::MethodBodyTypeMismatch {
                method_name,
                expected,
                actual,
                ..
            }
            | DiagnosticKind::BlockBodyTypeMismatch {
                method_name,
                expected,
                actual,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("expected".into(), json!(expected));
                obj.insert("actual".into(), json!(actual));
            }
            DiagnosticKind::InsufficientPositionalArguments {
                method_name,
                expected,
                actual,
                defined_in,
            }
            | DiagnosticKind::UnexpectedPositionalArgument {
                method_name,
                expected,
                actual,
                defined_in,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("expected".into(), json!(expected));
                obj.insert("actual".into(), json!(actual));
                if let Some(defined_in) = defined_in {
                    obj.insert("defined_in".into(), json!(defined_in));
                }
            }
            DiagnosticKind::UnexpectedKeywordArgument {
                method_name,
                keyword,
                defined_in,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("keyword".into(), json!(keyword));
                if let Some(defined_in) = defined_in {
                    obj.insert("defined_in".into(), json!(defined_in));
                }
            }
            DiagnosticKind::UnexpectedTypeArgument {
                method_name,
                method_type,
                type_arg,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("method_type".into(), json!(method_type));
                obj.insert("type_arg".into(), json!(type_arg));
            }
            DiagnosticKind::InsufficientTypeArguments {
                method_name,
                method_type,
                expected,
                actual,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("method_type".into(), json!(method_type));
                obj.insert("expected".into(), json!(expected));
                obj.insert("actual".into(), json!(actual));
            }
            DiagnosticKind::InsufficientKeywordArguments {
                method_name,
                defined_in,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                if let Some(defined_in) = defined_in {
                    obj.insert("defined_in".into(), json!(defined_in));
                }
            }
            DiagnosticKind::RequiredBlockMissing { method_name } => {
                obj.insert("method_name".into(), json!(method_name));
            }
            DiagnosticKind::NoMethod {
                method_name,
                receiver_type,
                missing_from,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("receiver_type".into(), json!(receiver_type));
                if !missing_from.is_empty() {
                    obj.insert("missing_from".into(), json!(missing_from));
                }
            }
            DiagnosticKind::ClassModuleMismatch {
                ruby_kind,
                rbs_kind,
                ..
            } => {
                obj.insert("ruby_kind".into(), json!(ruby_kind));
                obj.insert("rbs_kind".into(), json!(rbs_kind));
            }
            DiagnosticKind::MethodArityMismatch {
                method_name,
                ruby_arity,
                rbs_arity,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("ruby_arity".into(), json!(ruby_arity));
                obj.insert("rbs_arity".into(), json!(rbs_arity));
            }
            DiagnosticKind::IncompatibleArgumentForwarding {
                method_name,
                caller_signature,
                callee_signature,
                mismatch_kind,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("caller_signature".into(), json!(caller_signature));
                obj.insert("callee_signature".into(), json!(callee_signature));
                obj.insert("mismatch_kind".into(), json!(mismatch_kind.as_str()));
            }
            DiagnosticKind::MethodParameterMismatch {
                method_name,
                param_name,
                ruby_kind,
                rbs_kind,
                ..
            }
            | DiagnosticKind::DifferentMethodParameterKind {
                method_name,
                param_name,
                ruby_kind,
                rbs_kind,
                ..
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("param_name".into(), json!(param_name));
                obj.insert("ruby_kind".into(), json!(ruby_kind));
                obj.insert("rbs_kind".into(), json!(rbs_kind));
            }
            DiagnosticKind::UnresolvedOverloading {
                method_name,
                receiver_type,
                call_arguments,
                method_types,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("receiver_type".into(), json!(receiver_type));
                obj.insert("call_arguments".into(), json!(call_arguments));
                obj.insert("method_types".into(), json!(method_types));
            }
            DiagnosticKind::UnknownTupleIndex {
                index,
                tuple_length,
            } => {
                obj.insert("index".into(), json!(index));
                obj.insert("tuple_length".into(), json!(tuple_length));
            }
            DiagnosticKind::UnknownRecordKey { key, known_keys } => {
                obj.insert("key".into(), json!(key));
                obj.insert("known_keys".into(), json!(known_keys));
            }
            DiagnosticKind::TypeArgumentBoundViolation {
                param_name,
                bound_kind,
                bound,
                actual,
                ..
            } => {
                obj.insert("param_name".into(), json!(param_name));
                obj.insert("bound_kind".into(), json!(bound_kind.as_str()));
                obj.insert("bound".into(), json!(bound));
                obj.insert("actual".into(), json!(actual));
            }
            DiagnosticKind::AnnotationSyntaxError {
                annotation_text,
                parser_error,
            } => {
                obj.insert("annotation_text".into(), json!(annotation_text));
                if let Some(err) = parser_error {
                    obj.insert("parser_error".into(), json!(err));
                }
            }
            DiagnosticKind::PrivateMethodCall {
                method_name,
                receiver_type,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                obj.insert("receiver_type".into(), json!(receiver_type));
            }
            DiagnosticKind::MixinTypeArgumentArityMismatch {
                kind,
                target,
                class,
                expected,
                got,
            } => {
                obj.insert("kind".into(), json!(kind));
                obj.insert("target".into(), json!(target));
                obj.insert("class".into(), json!(class));
                obj.insert("expected".into(), json!(expected));
                obj.insert("got".into(), json!(got));
            }
            DiagnosticKind::InlineClassAliasMissingTypeName { kind } => {
                let kind_str = match kind {
                    InlineAliasKind::Class => "class",
                    InlineAliasKind::Module => "module",
                };
                obj.insert("alias_kind".into(), json!(kind_str));
            }
            DiagnosticKind::MixinMultipleArguments => {}
            DiagnosticKind::NonConstantClassName => {}
            DiagnosticKind::NonConstantModuleName => {}
            DiagnosticKind::NonConstantSuperClassName => {}
            DiagnosticKind::UnusedInlineAnnotation => {}
            DiagnosticKind::TopLevelMethodDefinition => {}
            DiagnosticKind::NestedSingletonScope => {}
            DiagnosticKind::TopLevelSingletonScope => {}
            DiagnosticKind::NonSelfSingletonScope => {}
            DiagnosticKind::DuplicatedMethodDefinition {
                method_name,
                duplicate_source,
            } => {
                obj.insert("method_name".into(), json!(method_name));
                if let Some(src) = duplicate_source {
                    obj.insert("duplicate_source".into(), duplicate_source_json(src));
                }
            }
            DiagnosticKind::DuplicatedDeclaration {
                name,
                detail,
                conflicting_source,
            } => {
                obj.insert("name".into(), json!(name));
                obj.insert("detail".into(), json!(detail));
                if let Some(loc) = conflicting_source {
                    obj.insert(
                        "conflicting_source".into(),
                        json!({
                            "file": loc.file.display().to_string(),
                            "start_byte": loc.range.start_byte,
                            "end_byte": loc.range.end_byte,
                        }),
                    );
                }
            }
            DiagnosticKind::RecursiveAliasDefinition {
                type_name,
                alias_names,
                primary_source,
            } => {
                obj.insert("type_name".into(), json!(type_name));
                obj.insert("alias_names".into(), json!(alias_names));
                if let Some(loc) = primary_source {
                    obj.insert(
                        "primary_source".into(),
                        json!({
                            "file": loc.file.display().to_string(),
                            "start_byte": loc.range.start_byte,
                            "end_byte": loc.range.end_byte,
                        }),
                    );
                }
            }
            DiagnosticKind::RecursiveTypeAlias {
                alias_names,
                primary_source,
            } => {
                obj.insert("alias_names".into(), json!(alias_names));
                if let Some(loc) = primary_source {
                    obj.insert(
                        "primary_source".into(),
                        json!({
                            "file": loc.file.display().to_string(),
                            "start_byte": loc.range.start_byte,
                            "end_byte": loc.range.end_byte,
                        }),
                    );
                }
            }
            DiagnosticKind::RecursiveAncestor {
                type_name,
                chain,
                primary_source,
            } => {
                obj.insert("type_name".into(), json!(type_name));
                obj.insert("chain".into(), json!(chain));
                if let Some(loc) = primary_source {
                    obj.insert(
                        "primary_source".into(),
                        json!({
                            "file": loc.file.display().to_string(),
                            "start_byte": loc.range.start_byte,
                            "end_byte": loc.range.end_byte,
                        }),
                    );
                }
            }
            DiagnosticKind::InstanceVariableDuplication {
                type_name,
                variable_name,
                duplicate_source,
            }
            | DiagnosticKind::ClassInstanceVariableDuplication {
                type_name,
                variable_name,
                duplicate_source,
            } => {
                obj.insert("type_name".into(), json!(type_name));
                obj.insert("variable_name".into(), json!(variable_name));
                if let Some(loc) = duplicate_source {
                    obj.insert(
                        "duplicate_source".into(),
                        json!({
                            "file": loc.file.display().to_string(),
                            "start_byte": loc.range.start_byte,
                            "end_byte": loc.range.end_byte,
                        }),
                    );
                }
            }
            DiagnosticKind::UnknownTypeName { name } => {
                obj.insert("name".into(), json!(name));
            }
            DiagnosticKind::FalseAssertion { natural, asserted } => {
                obj.insert("natural".into(), json!(natural));
                obj.insert("asserted".into(), json!(asserted));
            }
            DiagnosticKind::FallbackAny { reason } => {
                obj.insert("reason".into(), json!(reason));
            }
            DiagnosticKind::UnexpectedYield { method_name } => {
                obj.insert("method_name".into(), json!(method_name));
            }
            DiagnosticKind::UnexpectedSuper { method_name } => {
                obj.insert("method_name".into(), json!(method_name));
            }
            DiagnosticKind::UnsatisfiableConstraint {
                lower,
                upper,
                type_param,
                method_type,
            } => {
                obj.insert("lower".into(), json!(lower));
                obj.insert("upper".into(), json!(upper));
                obj.insert("type_param".into(), json!(type_param));
                obj.insert("method_type".into(), json!(method_type));
            }
            DiagnosticKind::UnknownConstant {
                name,
                path,
                did_you_mean,
                searched_namespaces,
                ..
            } => {
                obj.insert("name".into(), json!(name));
                obj.insert("path".into(), json!(path));
                obj.insert("searched_namespaces".into(), json!(searched_namespaces));
                if !did_you_mean.is_empty() {
                    obj.insert("did_you_mean".into(), json!(did_you_mean));
                }
            }
            DiagnosticKind::NonExhaustiveCase {
                scrutinee_type,
                residue_type,
            } => {
                obj.insert("scrutinee_type".into(), json!(scrutinee_type));
                obj.insert("residue_type".into(), json!(residue_type));
            }
            DiagnosticKind::UnreachableValueBranch => {}
            DiagnosticKind::IncompatibleAssignment { lhs_type, rhs_type } => {
                obj.insert("lhs_type".into(), json!(lhs_type));
                obj.insert("rhs_type".into(), json!(rhs_type));
            }
            DiagnosticKind::MultipleAssignmentConversion {
                original_type,
                returned_type,
            } => {
                obj.insert("original_type".into(), json!(original_type));
                obj.insert("returned_type".into(), json!(returned_type));
            }
            DiagnosticKind::UnknownInstanceVariable { name }
            | DiagnosticKind::UnknownClassVariable { name }
            | DiagnosticKind::UnknownGlobalVariable { name } => {
                obj.insert("name".into(), json!(name));
            }
            DiagnosticKind::DeprecatedReference {
                subject_kind,
                subject_name,
                message,
            } => {
                obj.insert("subject_kind".into(), json!(subject_kind.as_str()));
                obj.insert("subject_name".into(), json!(subject_name));
                if let Some(msg) = message {
                    obj.insert("deprecation_message".into(), json!(msg));
                }
            }
            DiagnosticKind::InfusionProviderSkipped {
                provider,
                subject,
                reason,
            } => {
                obj.insert("provider".into(), json!(provider));
                obj.insert("subject".into(), json!(subject));
                obj.insert("reason".into(), json!(reason));
            }
            DiagnosticKind::DuplicatedConfigKey { key, file } => {
                obj.insert("key".into(), json!(key));
                obj.insert("file".into(), json!(file));
            }
            DiagnosticKind::NotImplementedYet {
                site,
                subject_display,
            } => {
                obj.insert("site".into(), json!(site));
                obj.insert("subject_display".into(), json!(subject_display));
            }
            DiagnosticKind::SyntaxError { .. } => {}
        }

        Value::Object(obj)
    }

    fn fallback_line(&self) -> usize {
        std::fs::read(&self.location.file)
            .ok()
            .map(|source| byte_offset_to_line(&source, self.location.range.start_byte as usize))
            .unwrap_or(1)
    }
}

/// JSON shape for `DuplicatedMethodDefinition`'s `duplicate_source`:
/// discriminated on a `location`/`synthesized` key so consumers can tell
/// a real file apart from crema's own infusion synthesis without
/// guessing from a missing location.
fn duplicate_source_json(src: &DuplicateSource) -> Value {
    match src {
        DuplicateSource::Location(loc) => json!({
            "location": {
                "file": loc.file.display().to_string(),
                "start_byte": loc.range.start_byte,
                "end_byte": loc.range.end_byte,
            },
        }),
        DuplicateSource::Synthesized { infusion } => json!({
            "synthesized": { "infusion": infusion.as_str() },
        }),
    }
}

fn byte_to_char_offset(source: &[u8], offset: usize) -> u32 {
    let offset = offset.min(source.len());
    std::str::from_utf8(&source[..offset])
        .map(|s| s.chars().count() as u32)
        .unwrap_or(offset as u32)
}

macro_rules! diagnostic_codes {
    ($($pat:pat => $code:literal, $description:literal, $slug:literal),* $(,)?) => {
        /// Machine-readable error code (Steep-compatible).
        pub fn code(&self) -> &'static str {
            match self { $($pat => $code,)* }
        }

        /// Short human-readable summary of what this diagnostic detects.
        pub fn description(&self) -> &'static str {
            match self { $($pat => $description,)* }
        }

        /// Every `code()` string that any `DiagnosticKind` can produce.
        /// Used by `[diagnostic]` validation to detect typos in override
        /// keys, and by `crema doc diagnostic` to enumerate all codes.
        /// Generated from the same table as `code()` — a missing arm is a
        /// non-exhaustive compile error, so the two are always in sync.
        pub const ALL_CODES: &'static [&'static str] = &[$($code),*];

        /// Every `description()` string, index-aligned with [`Self::ALL_CODES`].
        pub const ALL_DESCRIPTIONS: &'static [&'static str] = &[$($description),*];

        /// Every diagnostic doc body, index-aligned with [`Self::ALL_CODES`].
        /// Missing markdown files fail at compile time via `include_str!`.
        pub const ALL_DOC_BODIES: &'static [&'static str] = &[
            $(include_str!(concat!("diagnostic_docs/", $slug, ".md"))),*
        ];
    };
}

impl DiagnosticKind {
    diagnostic_codes! {
        DiagnosticKind::ArgumentTypeMismatch { .. } => "Ruby::ArgumentTypeMismatch",
            "A call passes an argument whose inferred type is not accepted by the target method parameter.",
            "Ruby_ArgumentTypeMismatch",
        DiagnosticKind::InsufficientPositionalArguments { .. } => "Ruby::InsufficientPositionalArguments",
            "A call supplies fewer positional arguments than the target method requires.",
            "Ruby_InsufficientPositionalArguments",
        DiagnosticKind::UnexpectedPositionalArgument { .. } => "Ruby::UnexpectedPositionalArgument",
            "A call supplies more positional arguments than the target method accepts.",
            "Ruby_UnexpectedPositionalArgument",
        DiagnosticKind::UnexpectedKeywordArgument { .. } => "Ruby::UnexpectedKeywordArgument",
            "A call supplies a keyword argument that the target method does not accept.",
            "Ruby_UnexpectedKeywordArgument",
        DiagnosticKind::UnexpectedTypeArgument { .. } => "Ruby::UnexpectedTypeArgument",
            "A call-site type application supplies more type arguments than the target method accepts.",
            "Ruby_UnexpectedTypeArgument",
        DiagnosticKind::InsufficientTypeArguments { .. } => "Ruby::InsufficientTypeArguments",
            "A call-site type application supplies fewer type arguments than the target method requires.",
            "Ruby_InsufficientTypeArguments",
        DiagnosticKind::MethodBodyTypeMismatch { .. } => "Ruby::MethodBodyTypeMismatch",
            "A method body returns a value whose inferred type is not compatible with the declared return type.",
            "Ruby_MethodBodyTypeMismatch",
        DiagnosticKind::InsufficientKeywordArguments { .. } => "Ruby::InsufficientKeywordArguments",
            "A call omits a keyword argument required by the target method.",
            "Ruby_InsufficientKeywordArguments",
        DiagnosticKind::BlockBodyTypeMismatch { .. } => "Ruby::BlockBodyTypeMismatch",
            "A passed block returns a value whose inferred type is not compatible with the declared block return type.",
            "Ruby_BlockBodyTypeMismatch",
        DiagnosticKind::RequiredBlockMissing { .. } => "Ruby::RequiredBlockMissing",
            "A call omits the block required by the target method signature.",
            "Ruby_RequiredBlockMissing",
        DiagnosticKind::NoMethod { .. } => "Ruby::NoMethod",
            "A method call targets a receiver type that has no matching method definition.",
            "Ruby_NoMethod",
        DiagnosticKind::ClassModuleMismatch { .. } => "Ruby::ClassModuleMismatch",
            "A constant is implemented as a class or module in Ruby but declared as the other kind in RBS.",
            "Ruby_ClassModuleMismatch",
        DiagnosticKind::MethodArityMismatch { .. } => "Ruby::MethodArityMismatch",
            "A Ruby method definition has a parameter arity that differs from its RBS declaration.",
            "Ruby_MethodArityMismatch",
        DiagnosticKind::IncompatibleArgumentForwarding { .. } => "Ruby::IncompatibleArgumentForwarding",
            "A method forwards arguments or a block to a callee whose signature cannot accept the forwarded shape.",
            "Ruby_IncompatibleArgumentForwarding",
        DiagnosticKind::MethodParameterMismatch { .. } => "Ruby::MethodParameterMismatch",
            "A Ruby method parameter kind differs from the corresponding RBS parameter kind.",
            "Ruby_MethodParameterMismatch",
        DiagnosticKind::DifferentMethodParameterKind { .. } => "Ruby::DifferentMethodParameterKind",
            "The method parameter has different kind from the declaration: a Ruby looser parameter (optarg / restarg / kwoptarg / kwrestarg) does not match the RBS slot.",
            "Ruby_DifferentMethodParameterKind",
        DiagnosticKind::UnresolvedOverloading { .. } => "Ruby::UnresolvedOverloading",
            "A call targets an overloaded method, but none of the overloads accepts the supplied arguments.",
            "Ruby_UnresolvedOverloading",
        DiagnosticKind::UnknownTupleIndex { .. } => "Ruby::UnknownTupleIndex",
            "Tuple indexing uses an integer index outside the known tuple length.",
            "Ruby_UnknownTupleIndex",
        DiagnosticKind::UnknownRecordKey { .. } => "Ruby::UnknownRecordKey",
            "Record access uses a key that is not present in the known record type.",
            "Ruby_UnknownRecordKey",
        DiagnosticKind::TypeArgumentBoundViolation { .. } => "Crema::TypeArgumentBoundViolation",
            "A generic type argument does not satisfy the declared upper or lower bound.",
            "Crema_TypeArgumentBoundViolation",
        DiagnosticKind::AnnotationSyntaxError { .. } => "Ruby::AnnotationSyntaxError",
            "An inline RBS annotation contains syntax that the RBS parser rejects.",
            "Ruby_AnnotationSyntaxError",
        DiagnosticKind::InlineClassAliasMissingTypeName { .. } => "Ruby::ClassModuleAliasDeclarationMissingTypeName",
            "A class-alias or module-alias inline annotation cannot infer the aliased constant name.",
            "Ruby_ClassModuleAliasDeclarationMissingTypeName",
        DiagnosticKind::MixinMultipleArguments => "Ruby::MixinMultipleArguments",
            "An inline mixin call supplies multiple module arguments where only one can be converted to RBS.",
            "Ruby_MixinMultipleArguments",
        DiagnosticKind::NonConstantClassName => "Ruby::NonConstantClassName",
            "A class declaration name contains a dynamic expression instead of a constant path.",
            "Ruby_NonConstantClassName",
        DiagnosticKind::NonConstantModuleName => "Ruby::NonConstantModuleName",
            "A module declaration name contains a dynamic expression instead of a constant path.",
            "Ruby_NonConstantModuleName",
        DiagnosticKind::NonConstantSuperClassName => "Ruby::NonConstantSuperClassName",
            "A class declaration uses a dynamic expression as its superclass.",
            "Ruby_NonConstantSuperClassName",
        DiagnosticKind::UnusedInlineAnnotation => "Ruby::UnusedInlineAnnotation",
            "An inline RBS annotation is present but not consumed by the Ruby construct it was meant to describe.",
            "Ruby_UnusedInlineAnnotation",
        DiagnosticKind::TopLevelMethodDefinition => "Ruby::TopLevelMethodDefinition",
            "A top-level inline Ruby method definition has no RBS declaration target.",
            "Ruby_TopLevelMethodDefinition",
        DiagnosticKind::NestedSingletonScope => "Ruby::NestedSingletonScope",
            "An inline singleton scope targets the singleton class of a singleton class, which RBS cannot represent.",
            "Ruby_NestedSingletonScope",
        DiagnosticKind::TopLevelSingletonScope => "Ruby::TopLevelSingletonScope",
            "A top-level inline singleton scope opens main's singleton class, which RBS cannot represent.",
            "Ruby_TopLevelSingletonScope",
        DiagnosticKind::NonSelfSingletonScope => "Ruby::NonSelfSingletonScope",
            "An inline singleton scope targets a non-self expression, which RBS cannot represent.",
            "Ruby_NonSelfSingletonScope",
        DiagnosticKind::PrivateMethodCall { .. } => "Crema::PrivateMethodCall",
            "A private method is called with an explicit receiver.",
            "Crema_PrivateMethodCall",
        DiagnosticKind::MixinTypeArgumentArityMismatch { .. } => "Crema::MixinTypeArgumentArityMismatch",
            "A superclass or mixin reference supplies the wrong number of generic type arguments.",
            "Crema_MixinTypeArgumentArityMismatch",
        DiagnosticKind::DuplicatedMethodDefinition { .. } => "Ruby::DuplicatedMethodDefinitionError",
            "Inline RBS declares the same non-overload method more than once in one type.",
            "Ruby_DuplicatedMethodDefinitionError",
        DiagnosticKind::DuplicatedDeclaration { .. } => "RBS::DuplicatedDeclarationError",
            "Inline Ruby declarations define the same RBS name in conflicting declaration kinds.",
            "RBS_DuplicatedDeclarationError",
        DiagnosticKind::RecursiveAliasDefinition { .. } => "RBS::RecursiveAliasDefinitionError",
            "A method alias chain loops back to one of its own alias entries.",
            "RBS_RecursiveAliasDefinitionError",
        DiagnosticKind::RecursiveTypeAlias { .. } => "RBS::RecursiveTypeAliasError",
            "One or more type aliases form a cycle through transparent type constructors.",
            "RBS_RecursiveTypeAliasError",
        DiagnosticKind::RecursiveAncestor { .. } => "RBS::RecursiveAncestorError",
            "A class or module ancestor graph contains a cycle.",
            "RBS_RecursiveAncestorError",
        DiagnosticKind::InstanceVariableDuplication { .. } => "RBS::InstanceVariableDuplicationError",
            "A type declares the same instance variable name more than once.",
            "RBS_InstanceVariableDuplicationError",
        DiagnosticKind::ClassInstanceVariableDuplication { .. } => "RBS::ClassInstanceVariableDuplicationError",
            "A type declares the same class instance variable name more than once.",
            "RBS_ClassInstanceVariableDuplicationError",
        DiagnosticKind::UnknownTypeName { .. } => "RBS::UnknownTypeName",
            "A referenced RBS type name cannot be resolved in the loaded environment.",
            "RBS_UnknownTypeName",
        DiagnosticKind::FalseAssertion { .. } => "Ruby::FalseAssertion",
            "An inline assertion names a type incompatible with the expression's inferred type.",
            "Ruby_FalseAssertion",
        DiagnosticKind::FallbackAny { .. } => "Ruby::FallbackAny",
            "An expression could not be inferred precisely and fell back to untyped.",
            "Ruby_FallbackAny",
        DiagnosticKind::UnexpectedYield { .. } => "Ruby::UnexpectedYield",
            "A method body yields even though its RBS declaration has no block parameter.",
            "Ruby_UnexpectedYield",
        DiagnosticKind::UnexpectedSuper { .. } => "Ruby::UnexpectedSuper",
            "A super call cannot resolve a matching superclass method.",
            "Ruby_UnexpectedSuper",
        DiagnosticKind::UnsatisfiableConstraint { .. } => "Ruby::UnsatisfiableConstraint",
            "Method type parameter bounds generated from a call have no satisfiable subtype relation.",
            "Ruby_UnsatisfiableConstraint",
        DiagnosticKind::UnknownConstant { .. } => "Ruby::UnknownConstant",
            "A constant reference or declaration path cannot be resolved in the current scope.",
            "Ruby_UnknownConstant",
        DiagnosticKind::NonExhaustiveCase { .. } => "Crema::NonExhaustiveCase",
            "A case expression without else leaves part of the scrutinee type uncovered by its when clauses.",
            "Crema_NonExhaustiveCase",
        DiagnosticKind::UnreachableValueBranch => "Ruby::UnreachableValueBranch",
            "A case expression's else branch cannot run because previous when clauses cover every scrutinee value.",
            "Ruby_UnreachableValueBranch",
        DiagnosticKind::IncompatibleAssignment { .. } => "Ruby::IncompatibleAssignment",
            "A variable assignment writes a value whose inferred type is incompatible with the declared variable type.",
            "Ruby_IncompatibleAssignment",
        DiagnosticKind::MultipleAssignmentConversion { .. } => "Ruby::MultipleAssignmentConversionError",
            "A multiple assignment's RHS resolved to_ary but the return type is not a Tuple or Array[T], which Ruby itself rejects at runtime.",
            "Ruby_MultipleAssignmentConversionError",
        DiagnosticKind::UnknownInstanceVariable { .. } => "Ruby::UnknownInstanceVariable",
            "An instance variable write targets a variable with no declaration on the current self type.",
            "Ruby_UnknownInstanceVariable",
        DiagnosticKind::UnknownClassVariable { .. } => "Ruby::UnknownClassVariable",
            "A class variable write targets a variable with no declaration on the current class context.",
            "Ruby_UnknownClassVariable",
        DiagnosticKind::UnknownGlobalVariable { .. } => "Ruby::UnknownGlobalVariable",
            "A global variable write targets a variable with no declaration in loaded RBS signatures.",
            "Ruby_UnknownGlobalVariable",
        DiagnosticKind::DeprecatedReference { .. } => "Ruby::DeprecatedReference",
            "A reference targets a method, constant, or global variable whose RBS declaration is marked `%a{deprecated}`.",
            "Ruby_DeprecatedReference",
        DiagnosticKind::InfusionProviderSkipped { .. } => "Crema::InfusionProviderSkipped",
            "An infusion provider skipped generating type information for a subject and recorded the reason.",
            "Crema_InfusionProviderSkipped",
        DiagnosticKind::DuplicatedConfigKey { .. } => "Crema::DuplicatedConfigKey",
            "A [infusion.config] YAML file declares the same mapping key more than once; crema keeps the last value (Psych-compatible) and warns.",
            "Crema_DuplicatedConfigKey",
        DiagnosticKind::NotImplementedYet { .. } => "Crema::NotImplementedYet",
            "An unsupported internal type-checking path was reached and emitted its instrumentation marker.",
            "Crema_NotImplementedYet",
        DiagnosticKind::SyntaxError { .. } => "Ruby::SyntaxError",
            "Ruby source parsing failed before inline annotation collection or type checking could run.",
            "Ruby_SyntaxError",
    }

    pub fn doc_for_code(code: &str) -> Option<&'static str> {
        Self::ALL_CODES
            .iter()
            .position(|known_code| *known_code == code)
            .map(|index| Self::ALL_DOC_BODIES[index])
    }

    pub fn is_known_code(code: &str) -> bool {
        Self::ALL_CODES.contains(&code)
    }

    /// The built-in severity for this diagnostic kind, used as the
    /// fallback when `crema.toml` `[diagnostic]` does not override it.
    /// Delegates to [`Self::default_severity_for_code`] so the mapping
    /// has a single code-string-keyed source.
    pub fn default_severity(&self) -> Severity {
        Self::default_severity_for_code(self.code())
    }

    /// Code-string-keyed form of [`Self::default_severity`], for callers
    /// that hold a `code()` string rather than an instance (e.g. `crema
    /// diagnostic list`). The mapping depends only on the variant
    /// (fields are ignored in `default_severity`), so keying by `code()`
    /// is equivalent; the `default_severity_steep_*_kinds` tests pin the
    /// correspondence and turn red if `code()` and this match drift.
    ///
    /// Mirrors Steep's `default` preset
    /// (`steep/lib/steep/diagnostic/ruby.rb` self.default):
    ///
    /// - `FalseAssertion`, `FallbackAny`, `UnsatisfiableConstraint`,
    ///   `IncompatibleAssignment`, `UnexpectedTypeArgument`,
    ///   `InsufficientTypeArguments` (Steep: `InsufficientTypeArgument`),
    ///   `DifferentMethodParameterKind`,
    ///   `MultipleAssignmentConversionError` → `Hint`
    /// - `BlockBodyTypeMismatch`, `UnexpectedYield`, `UnknownGlobalVariable`
    ///   → `Warning`
    /// - `UnknownRecordKey`, `UnknownInstanceVariable` → `Information`
    /// - Crema-only diagnostics (`Crema::*` codes, `RBS::*` codes
    ///   without a Steep counterpart) → `Error` by default
    /// - Everything else → `Error`
    ///
    /// Deliberate deviation: Steep's `default` preset maps
    /// `UnknownConstant` to `:warning` (only `strict` / `all_error` make
    /// it an error). crema keeps it at `Error` because an unresolved
    /// constant is a high-value preflight signal worth failing on.
    /// `SyntaxError` is likewise pinned to `Error` in every preset (see
    /// its variant doc comment) even though Steep's `default` ships it as
    /// `:information` — Ruby itself treats a syntax error as fatal.
    /// These two are the *only* intentional deviations; every other code
    /// above tracks Steep's `default` value exactly.
    ///
    /// `UnknownClassVariable` has no Steep counterpart at all (Steep does
    /// not check class variable references), so there is nothing to
    /// mirror; it is set to `Information` here as crema's own choice.
    pub fn default_severity_for_code(code: &str) -> Severity {
        match code {
            "Ruby::FalseAssertion"
            | "Ruby::FallbackAny"
            | "Ruby::UnsatisfiableConstraint"
            | "Ruby::UnreachableValueBranch"
            | "Ruby::IncompatibleAssignment"
            | "Ruby::UnexpectedTypeArgument"
            | "Ruby::InsufficientTypeArguments"
            | "Ruby::DifferentMethodParameterKind"
            | "Crema::NonExhaustiveCase"
            | "Ruby::MultipleAssignmentConversionError" => Severity::Hint,
            "Ruby::BlockBodyTypeMismatch"
            | "Ruby::UnexpectedYield"
            | "Ruby::IncompatibleArgumentForwarding"
            | "Ruby::UnknownGlobalVariable"
            | "Ruby::DeprecatedReference" => Severity::Warning,
            "Ruby::UnexpectedSuper" => Severity::Information,
            "Ruby::UnknownRecordKey" => Severity::Information,
            "Ruby::UnknownInstanceVariable" => Severity::Information,
            "Ruby::UnknownClassVariable" => Severity::Information,
            "Crema::InfusionProviderSkipped" => Severity::Information,
            "Crema::DuplicatedConfigKey" => Severity::Warning,
            "Crema::NotImplementedYet" => Severity::Ignore,
            _ => Severity::Error,
        }
    }

    /// `all_error` preset's code → severity map. Mirrors the shape of
    /// [`Self::default_severity_for_code`]: every preset that has
    /// per-code exceptions gets its own table here so the policy lives
    /// next to the codes it describes. `all_error` currently has no
    /// exceptions — every diagnostic is promoted to `Severity::Error`.
    /// `Crema::NotImplementedYet` is held at `Ignore` by the
    /// [`is_dev_only_code`](Self::is_dev_only_code) short-circuit one
    /// level up in [`crate::config::Preset::severity_for_code`], not
    /// here.
    pub fn all_error_severity_for_code(_code: &str) -> Severity {
        Severity::Error
    }

    /// `all_ignore` preset's code → severity map. Mirrors the shape of
    /// [`Self::default_severity_for_code`]. The only exception is
    /// `Ruby::SyntaxError` — Ruby itself rejects parse-error input as a
    /// hard failure, so crema follows suit and keeps the diagnostic at
    /// `Error` even when the user opts every other code into `Ignore`.
    /// Per-code overrides in `crema.toml` `[diagnostic]` are still
    /// honored above this layer (`DiagnosticConfig::severity_for_code`
    /// consults the override map first), so a user who writes
    /// `"Ruby::SyntaxError" = "ignore"` explicitly does get `Ignore`.
    pub fn all_ignore_severity_for_code(code: &str) -> Severity {
        match code {
            "Ruby::SyntaxError" => Severity::Error,
            _ => Severity::Ignore,
        }
    }

    /// Dev-only diagnostic codes stay at `Severity::Ignore` in *every*
    /// preset, including `all_error`. Only an explicit per-code override
    /// in `crema.toml` `[diagnostic]` can surface them. Used as the
    /// short-circuit at the top of [`crate::config::Preset::severity_for_code`].
    ///
    /// Hard-coded by exact code string (not by `Crema::` prefix) so the
    /// existing `Crema::*` diagnostics retain their normal preset
    /// semantics — `is_dev_only_code` is a per-kind opt-in, not a
    /// namespace policy.
    pub fn is_dev_only_code(code: &str) -> bool {
        matches!(code, "Crema::NotImplementedYet")
    }

    /// Codes that the user cannot override in `crema.toml`
    /// `[diagnostic]`. The mirror image of
    /// [`is_dev_only_code`](Self::is_dev_only_code): dev-only codes
    /// stay at `Ignore` until the user opts in, always-error codes
    /// stay at `Error` and refuse to be opted out. Currently only
    /// `Ruby::SyntaxError`: Ruby itself rejects parse-error input as
    /// a hard failure (`ruby -e '{a!: }'` exits with `SyntaxError`),
    /// and Steep records it via
    /// `SourceFile.with_syntax_error` regardless of config
    /// (`steep/lib/steep/services/type_check_service.rb:335-347`), so
    /// crema follows suit and treats the code as a structural error
    /// the user cannot suppress. Attempting to set
    /// `[diagnostic] "Ruby::SyntaxError" = "..."` is rejected as
    /// `ConfigError::InvalidDiagnostic` at startup so the
    /// misconfiguration is visible rather than silently dropped.
    pub fn is_always_error_code(code: &str) -> bool {
        matches!(code, "Ruby::SyntaxError")
    }

    /// Extra detail rendered as a Steep-style box below the one-line
    /// `Display` message. Each entry becomes one continuation line in
    /// [`Diagnostic`]'s text output, prefixed with `│ `. Kinds without
    /// structured detail return an empty slice and skip the box entirely.
    ///
    /// Currently only [`DiagnosticKind::UnresolvedOverloading`] participates:
    /// the short message names just the method, while the box carries the
    /// call-site argument shape and the full overload list so AI agents
    /// don't have to grep RBS to act on the diagnostic.
    pub fn detail_lines(&self) -> Vec<String> {
        match self {
            DiagnosticKind::UnresolvedOverloading {
                method_name,
                call_arguments,
                method_types,
                ..
            } => {
                let mut lines = Vec::with_capacity(3 + method_types.len());
                lines.push("Arguments:".to_string());
                lines.push(format!("  {}", call_arguments));
                if let Some((head, rest)) = method_types.split_first() {
                    // The `Method types:` header is gated on a non-empty
                    // list — an empty `method_types` would otherwise
                    // leave a dangling header with nothing under it.
                    lines.push("Method types:".to_string());
                    let prefix = format!("  def {}:", method_name);
                    lines.push(format!("{} {}", prefix, head));
                    // Continuation lines align `|` with the `:` of the
                    // `def NAME:` head — drop one column so the bar lands
                    // directly under the colon, matching Steep's box.
                    let indent = " ".repeat(prefix.len() - 1);
                    for ty in rest {
                        lines.push(format!("{}| {}", indent, ty));
                    }
                }
                lines
            }
            _ => Vec::new(),
        }
    }
}

/// Debug-oriented rendering. The emitter resolves the real severity
/// (potentially overridden by `[diagnostic]` config) before emitting,
/// so this `Display` impl falls back to `default_severity()` for
/// `format!("{}", diag)` callers that have no emitter handy
/// (integration tests, ad-hoc inspection).
///
/// Kinds with [`DiagnosticKind::detail_lines`] populated get a Steep-style
/// box appended on their own lines, each prefixed with `│ `.
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}: [{}] {}",
            self.location.file.display(),
            self.fallback_line(),
            self.kind.default_severity().as_str(),
            self.kind,
        )?;
        for line in self.kind.detail_lines() {
            write!(f, "\n│ {}", line)?;
        }
        Ok(())
    }
}

fn format_method_arity_message_part(arity: &str) -> String {
    if arity.contains("positional") || arity.contains("keyword") {
        arity.to_string()
    } else {
        format!("{} parameters", arity)
    }
}

impl fmt::Display for DiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticKind::ArgumentTypeMismatch {
                method_name,
                param_index,
                keyword,
                expected,
                actual,
                ..
            } => {
                if let Some(kw) = keyword {
                    write!(
                        f,
                        "Cannot pass `{}` as keyword `{}` of `{}`, expected `{}`",
                        actual, kw, method_name, expected,
                    )
                } else if let Some(idx) = param_index {
                    write!(
                        f,
                        "Cannot pass `{}` as argument {} of `{}`, expected `{}`",
                        actual,
                        idx + 1,
                        method_name,
                        expected,
                    )
                } else {
                    write!(
                        f,
                        "Cannot pass `{}` to `{}`, expected `{}`",
                        actual, method_name, expected,
                    )
                }
            }
            DiagnosticKind::InsufficientPositionalArguments {
                method_name,
                expected,
                actual,
                ..
            } => {
                write!(
                    f,
                    "Method `{}` requires at least {} positional arguments, but got {}",
                    method_name, expected, actual,
                )
            }
            DiagnosticKind::UnexpectedPositionalArgument {
                method_name,
                expected,
                actual,
                ..
            } => {
                write!(
                    f,
                    "Method `{}` accepts at most {} positional arguments, but got {}",
                    method_name, expected, actual,
                )
            }
            DiagnosticKind::UnexpectedKeywordArgument {
                method_name,
                keyword,
                ..
            } => {
                write!(
                    f,
                    "Method `{}` does not accept keyword argument `{}`",
                    method_name, keyword,
                )
            }
            DiagnosticKind::UnexpectedTypeArgument {
                method_type,
                type_arg,
                ..
            } => {
                write!(
                    f,
                    "Unexpected type arg `{}` is given to method type `{}`",
                    type_arg, method_type,
                )
            }
            DiagnosticKind::InsufficientTypeArguments {
                method_type,
                expected,
                actual,
                ..
            } => {
                write!(
                    f,
                    "Requires {} types, but {} given: `{}`",
                    expected, actual, method_type,
                )
            }
            DiagnosticKind::MethodBodyTypeMismatch {
                method_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Return value `{}` of `{}` does not match expected type `{}`",
                    actual, method_name, expected,
                )
            }
            DiagnosticKind::InsufficientKeywordArguments {
                method_name,
                keyword,
                ..
            } => {
                write!(
                    f,
                    "Missing required keyword argument `{}` for `{}`",
                    keyword, method_name,
                )
            }
            DiagnosticKind::BlockBodyTypeMismatch {
                method_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Block of `{}` returns `{}`, expected `{}`",
                    method_name, actual, expected,
                )
            }
            DiagnosticKind::RequiredBlockMissing { method_name } => {
                write!(
                    f,
                    "Method `{}` requires a block but none was given",
                    method_name,
                )
            }
            DiagnosticKind::NoMethod { method_name, .. } => {
                write!(f, "Type does not have method `{}`", method_name)
            }
            DiagnosticKind::ClassModuleMismatch {
                name,
                ruby_kind,
                rbs_kind,
            } => {
                write!(
                    f,
                    "`{}` is a {} in Ruby but declared as {} in RBS",
                    name, ruby_kind, rbs_kind,
                )
            }
            DiagnosticKind::MethodArityMismatch {
                method_name,
                ruby_arity,
                rbs_arity,
            } => {
                let ruby_display = format_method_arity_message_part(ruby_arity);
                let rbs_display = format_method_arity_message_part(rbs_arity);
                write!(
                    f,
                    "Method `{}` has {} in Ruby but {} in RBS",
                    method_name, ruby_display, rbs_display,
                )
            }
            DiagnosticKind::IncompatibleArgumentForwarding {
                method_name,
                caller_signature,
                callee_signature,
                mismatch_kind,
            } => match mismatch_kind {
                ForwardingMismatchKind::Arity => write!(
                    f,
                    "Cannot forward arguments to `{}`: incompatible arity: {} and {}",
                    method_name, caller_signature, callee_signature,
                ),
                ForwardingMismatchKind::Type => write!(
                    f,
                    "Cannot forward arguments to `{}`: {} is not a subtype of {}",
                    method_name, caller_signature, callee_signature,
                ),
                ForwardingMismatchKind::Block => write!(
                    f,
                    "Cannot forward block to `{}`: {} is not a subtype of {}",
                    method_name, caller_signature, callee_signature,
                ),
            },
            DiagnosticKind::MethodParameterMismatch {
                method_name,
                param_name,
                ruby_kind,
                rbs_kind,
            }
            | DiagnosticKind::DifferentMethodParameterKind {
                method_name,
                param_name,
                ruby_kind,
                rbs_kind,
            } => {
                write!(
                    f,
                    "Parameter `{}` of `{}` is {} in Ruby but {} in RBS",
                    param_name, method_name, ruby_kind, rbs_kind,
                )
            }
            DiagnosticKind::UnresolvedOverloading { method_name, .. } => {
                write!(f, "Cannot find compatible overload for `{}`", method_name)
            }
            DiagnosticKind::UnknownTupleIndex {
                index,
                tuple_length,
            } => {
                write!(
                    f,
                    "Tuple index {} is out of range for tuple of length {}",
                    index, tuple_length,
                )
            }
            DiagnosticKind::UnknownRecordKey { key, known_keys } => {
                let known = if known_keys.is_empty() {
                    "(none)".to_string()
                } else {
                    known_keys
                        .iter()
                        .map(|k| format!(":{}", k))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                write!(f, "Unknown record key `:{}` (known keys: {})", key, known,)
            }
            DiagnosticKind::TypeArgumentBoundViolation {
                container_name,
                param_name,
                bound_kind,
                bound,
                actual,
            } => match bound_kind {
                BoundKind::Upper => write!(
                    f,
                    "Type argument `{}` violates upper bound of `{}` in `{}`: `{}` is not a subtype of `{}`",
                    actual, param_name, container_name, actual, bound,
                ),
                BoundKind::Lower => write!(
                    f,
                    "Type argument `{}` violates lower bound of `{}` in `{}`: `{}` is not a subtype of `{}`",
                    actual, param_name, container_name, bound, actual,
                ),
            },
            DiagnosticKind::AnnotationSyntaxError {
                annotation_text,
                parser_error,
            } => match parser_error {
                Some(err) => write!(
                    f,
                    "Syntax error in inline annotation `{}` ({})",
                    annotation_text, err,
                ),
                None => write!(f, "Syntax error in inline annotation `{}`", annotation_text,),
            },
            DiagnosticKind::InlineClassAliasMissingTypeName { kind } => match kind {
                InlineAliasKind::Class => {
                    write!(f, "Class name is missing in class alias declaration")
                }
                InlineAliasKind::Module => {
                    write!(f, "Module name is missing in module alias declaration")
                }
            },
            DiagnosticKind::MixinMultipleArguments => {
                write!(f, "Mixing multiple modules with one call is not supported")
            }
            DiagnosticKind::NonConstantClassName => {
                write!(f, "Class name must be a constant")
            }
            DiagnosticKind::NonConstantModuleName => {
                write!(f, "Module name must be a constant")
            }
            DiagnosticKind::NonConstantSuperClassName => {
                write!(f, "Super class name must be a constant")
            }
            DiagnosticKind::UnusedInlineAnnotation => {
                write!(f, "Unused inline rbs annotation")
            }
            DiagnosticKind::TopLevelMethodDefinition => {
                write!(f, "Top-level method definition is not supported")
            }
            DiagnosticKind::NestedSingletonScope => {
                write!(
                    f,
                    "Nested `class << self` opens the singleton class of a singleton class, which has no representation in RBS"
                )
            }
            DiagnosticKind::TopLevelSingletonScope => {
                write!(
                    f,
                    "Top-level `class << self` opens the singleton class of `main`, which has no representation in RBS"
                )
            }
            DiagnosticKind::NonSelfSingletonScope => {
                write!(
                    f,
                    "Non-self expression in `class << ...` opens an anonymous singleton class, which has no representation in RBS"
                )
            }
            DiagnosticKind::PrivateMethodCall { method_name, .. } => write!(
                f,
                "Private method `{}` cannot be called with an explicit receiver",
                method_name,
            ),
            DiagnosticKind::MixinTypeArgumentArityMismatch {
                kind,
                target,
                class,
                expected,
                got,
            } => write!(
                f,
                "{} {} for {} expects {} type argument(s), got {}",
                kind, target, class, expected, got,
            ),
            DiagnosticKind::DuplicatedMethodDefinition { method_name, .. } => {
                write!(f, "Duplicated method definition: `{}`", method_name,)
            }
            DiagnosticKind::DuplicatedDeclaration { name, detail, .. } => {
                write!(f, "Duplicated declaration for `{}` ({})", name, detail)
            }
            DiagnosticKind::RecursiveAliasDefinition {
                type_name,
                alias_names,
                ..
            } => write!(
                f,
                "Recursive aliases in {}: {}",
                type_name,
                alias_names.join(", "),
            ),
            DiagnosticKind::RecursiveTypeAlias { alias_names, .. } => {
                write!(
                    f,
                    "Recursive type alias definition found for: {}",
                    alias_names.join(", "),
                )
            }
            DiagnosticKind::RecursiveAncestor { chain, .. } => {
                write!(f, "Detected recursive ancestors: {}", chain.join(" < "),)
            }
            DiagnosticKind::InstanceVariableDuplication {
                type_name,
                variable_name,
                ..
            } => write!(
                f,
                "Duplicated instance variable name `{}` in `{}`",
                variable_name, type_name,
            ),
            DiagnosticKind::ClassInstanceVariableDuplication {
                type_name,
                variable_name,
                ..
            } => write!(
                f,
                "Duplicated class instance variable name `{}` in `{}`",
                variable_name, type_name,
            ),
            DiagnosticKind::UnknownTypeName { name } => {
                write!(f, "Cannot find type `{}`", name,)
            }
            DiagnosticKind::FalseAssertion { natural, asserted } => {
                write!(
                    f,
                    "Type `{}` is incompatible with asserted type `{}`",
                    natural, asserted,
                )
            }
            DiagnosticKind::FallbackAny { reason } => {
                write!(f, "Cannot detect the type of the expression ({})", reason,)
            }
            DiagnosticKind::UnexpectedYield { .. } => {
                write!(f, "No block given for `yield`")
            }
            DiagnosticKind::UnexpectedSuper { method_name } => {
                write!(f, "No superclass method `{}` defined", method_name)
            }
            DiagnosticKind::UnsatisfiableConstraint {
                lower,
                upper,
                type_param,
                method_type,
            } => {
                write!(
                    f,
                    "Unsatisfiable constraint `{} <: {} <: {}` is generated through {}",
                    lower, type_param, upper, method_type,
                )
            }
            DiagnosticKind::UnknownConstant { name, kind, .. } => {
                write!(
                    f,
                    "Cannot find the declaration of {}: `{}`",
                    kind.as_str(),
                    name
                )
            }
            DiagnosticKind::NonExhaustiveCase { residue_type, .. } => {
                write!(
                    f,
                    "Non-exhaustive `case`: residue `{}` is not covered by any `when` and no `else` is present",
                    residue_type,
                )
            }
            DiagnosticKind::UnreachableValueBranch => {
                write!(
                    f,
                    "The `else` branch is unreachable: every value of the scrutinee is covered by the preceding `when` clauses",
                )
            }
            DiagnosticKind::IncompatibleAssignment { lhs_type, rhs_type } => {
                write!(
                    f,
                    "Cannot assign a value of type `{}` to a variable of type `{}`",
                    rhs_type, lhs_type,
                )
            }
            DiagnosticKind::MultipleAssignmentConversion {
                original_type,
                returned_type,
            } => {
                write!(
                    f,
                    "Cannot convert `{}` to Array or tuple (`#to_ary` returns `{}`)",
                    original_type, returned_type,
                )
            }
            DiagnosticKind::UnknownInstanceVariable { name } => {
                write!(
                    f,
                    "Cannot find the declaration of instance variable: `{}`",
                    name
                )
            }
            DiagnosticKind::UnknownClassVariable { name } => {
                write!(
                    f,
                    "Cannot find the declaration of class variable: `{}`",
                    name
                )
            }
            DiagnosticKind::UnknownGlobalVariable { name } => {
                write!(
                    f,
                    "Cannot find the declaration of global variable: `{}`",
                    name
                )
            }
            DiagnosticKind::DeprecatedReference {
                subject_kind,
                subject_name,
                message,
            } => match message {
                Some(msg) => write!(
                    f,
                    "The {} `{}` is deprecated: {}",
                    subject_kind.display_noun(),
                    subject_name,
                    msg,
                ),
                None => write!(
                    f,
                    "The {} `{}` is deprecated",
                    subject_kind.display_noun(),
                    subject_name,
                ),
            },
            DiagnosticKind::InfusionProviderSkipped {
                provider,
                subject,
                reason,
            } => {
                write!(f, "{} infusion skipped `{}`: {}", provider, subject, reason)
            }
            DiagnosticKind::DuplicatedConfigKey { key, file } => {
                write!(
                    f,
                    "config key `{}` is duplicated in {}; keeping the last value",
                    key, file
                )
            }
            DiagnosticKind::NotImplementedYet {
                site,
                subject_display,
            } => {
                write!(
                    f,
                    "Not implemented yet at {}: subject {}",
                    site, subject_display
                )
            }
            DiagnosticKind::SyntaxError { message } => {
                write!(f, "Syntax error: {}", message)
            }
        }
    }
}

/// Stream-oriented writer that serializes each `Diagnostic` to a
/// `Write` sink as soon as the layer produces it. Crema does not buffer
/// diagnostics across layers — every layer (inline parse, build
/// validate, type check) calls `emit` directly and the bytes hit the
/// writer immediately.
///
/// `format` selects the wire format (currently only `Jsonl`); the field
/// keeps the door open for additional formats (Sarif, Csv, text) to be
/// added without touching the emit call sites.
///
/// Generic over `W: Write` so production code can wire it to a locked
/// stdout while tests can feed it a `Vec<u8>` for assertions. Use
/// `had_any_error()` to drive the exit-code decision after the run —
/// only `Severity::Error` flips the flag, matching the LSP-aligned
/// exit policy (warning / information / hint diagnostics still emit
/// to stdout but keep the process at exit 0).
///
/// `BrokenPipe` errors are swallowed silently: the README documents
/// `crema check | head -n 5` as the canonical capping pattern, and
/// `head` closes stdout once it has read its quota. Panicking there
/// would punish exactly the UNIX-composition use case crema is
/// supposed to encourage.
pub struct DiagnosticEmitter<W: Write> {
    format: Format,
    writer: W,
    config: DiagnosticConfig,
    /// ADR-0029 §3: CLI positional targets narrow the *output* to this
    /// set, leaving the environment scope (and hence `check_source`'s
    /// walk) untouched. `None` means bare `crema check` — no filtering.
    /// A diagnostic whose `location.file` misses this set is dropped
    /// before it can affect `had_any_error` or reach the writer, so a
    /// filtered-out `Error` never flips the exit code (ADR-0029 slice S3
    /// design point 5 — the CLI argument is a view, not a scope).
    filter: Option<HashSet<PathBuf>>,
    /// Base directory (the canonicalized process cwd) that every
    /// emitted `"file"` value is relativized against. Internal file
    /// identities stay canonical/absolute (filter matching, line-starts
    /// lookup) — this only affects the serialized output, keeping the
    /// pre-ADR-0029 cwd-relative display (compact, `jq`-filterable per
    /// ADR-0005) now that scope paths are canonicalized internally.
    /// `None` (unresolvable cwd, or unit tests via `new`'s default)
    /// leaves paths untouched.
    display_base: Option<PathBuf>,
    line_starts_by_file: HashMap<PathBuf, Vec<u32>>,
    // Cache of `line_starts` derived by reading the file off disk when
    // the location's path was not registered via `register_source`. The
    // outer Option wraps the read outcome so read failures (missing
    // file, unreadable) are memoized as `None` and stop re-triggering
    // disk I/O on later diagnostics pointing at the same path. Held on
    // top of `line_starts_by_file` — never fed into it — so the
    // "registered in-memory source beats disk content" invariant stays
    // observable in the lookup order.
    fallback_line_starts: HashMap<PathBuf, Option<Vec<u32>>>,
    /// Registered source bytes for fingerprint snippet extraction.
    /// Mirrors the "registered in-memory source beats disk content"
    /// invariant of `line_starts_by_file`: the snippet must come from
    /// the bytes the checker actually walked, not whatever is on disk
    /// at emit time (mid-run editor saves, `-e` input that never
    /// persists).
    sources_by_file: HashMap<PathBuf, Arc<[u8]>>,
    /// Source bytes read off disk on demand when the location's path
    /// was not registered. Populated only when a diagnostic is
    /// actually emitted, so a zero-diagnostic run pays no cost here.
    /// `None` memoizes read failures — the snippet material is then
    /// empty but the fingerprint is still computed and emitted.
    fingerprint_sources: HashMap<PathBuf, Option<Vec<u8>>>,
    had_any_error: bool,
    broken: bool,
    /// When `true`, `emit()` diverts the fully-serialized JSONL line to
    /// an internal buffer of (file, code, fingerprint) tuples instead of
    /// writing streaming output. `finish()` sorts that buffer by
    /// `(file, code, fingerprint)` byte-lex and writes each row as a
    /// hand-serialized JSON object with the keys in that same order —
    /// the sort key is the row's serialized byte prefix, so external
    /// `LC_ALL=C sort` yields the same order. Contract-level flag
    /// (`git status --porcelain` analogue for baseline / lockfile
    /// workflows), orthogonal to `Format` (which will grow serialization
    /// variants like xml/sarif). `finish()` is idempotent and safe to
    /// call on a non-tamped emitter (no-op).
    tamped: bool,
    tamp_buffer: Vec<TampedRecord>,
}

/// One row of a `--tamp` stream: the stable-projection triple that
/// survives edits and machine boundaries. Message text / line numbers /
/// byte ranges / severity are deliberately dropped — including any of
/// them would let unrelated edits churn `git diff` on the baseline file.
struct TampedRecord {
    file: String,
    code: String,
    fingerprint: String,
}

impl<W: Write> DiagnosticEmitter<W> {
    pub fn new(format: Format, writer: W, config: DiagnosticConfig) -> Self {
        DiagnosticEmitter {
            format,
            writer,
            config,
            filter: None,
            display_base: None,
            line_starts_by_file: HashMap::new(),
            fallback_line_starts: HashMap::new(),
            sources_by_file: HashMap::new(),
            fingerprint_sources: HashMap::new(),
            had_any_error: false,
            broken: false,
            tamped: false,
            tamp_buffer: Vec::new(),
        }
    }

    /// Opt into the `--tamp` stable-projection mode (see the `tamped`
    /// field doc). No-op when `enabled == false` — this is the CLI
    /// wiring point, called with the raw `--tamp` boolean.
    pub fn with_tamped(mut self, enabled: bool) -> Self {
        self.tamped = enabled;
        self
    }

    /// Set the base directory every emitted `"file"` value is displayed
    /// relative to (see the `display_base` field doc). The caller passes
    /// the canonicalized process cwd so it compares equal against the
    /// canonicalized paths diagnostics carry internally.
    pub fn with_display_base(mut self, base: Option<PathBuf>) -> Self {
        self.display_base = base;
        self
    }

    /// Set the ADR-0029 §3 output filter. `Some(set)` restricts every
    /// subsequent `emit` to diagnostics located in one of `set`'s files;
    /// `None` (the `new` default) emits everything. Paths must be in the
    /// same representation `emit`'s `Diagnostic::location.file` uses
    /// (the caller's job — `DiagnosticEmitter` does no normalization).
    pub fn with_filter(mut self, filter: Option<HashSet<PathBuf>>) -> Self {
        self.filter = filter;
        self
    }

    /// Takes a shared handle rather than copying: the caller keeps the
    /// same bytes alive for parsing and the build validator, so a
    /// second owned copy per check-scope file was pure duplication
    /// (~19MB on the gitlab workload).
    pub fn register_source(&mut self, file: PathBuf, source: Arc<[u8]>) {
        self.line_starts_by_file
            .insert(file.clone(), line_starts(&source));
        self.sources_by_file.insert(file, source);
    }

    pub fn emit(&mut self, diag: &Diagnostic) {
        if let Some(filter) = &self.filter
            && !filter.contains(&diag.location.file)
        {
            return;
        }
        let severity = self.config.severity_for(&diag.kind);
        if severity == Severity::Ignore {
            return;
        }
        if severity == Severity::Error {
            self.had_any_error = true;
        }
        if self.broken {
            return;
        }
        match self.format {
            Format::Jsonl => {
                let line_number = self.line_for(diag);
                let mut value = diag.to_json_value_with_line(severity, line_number);
                if let Some(base) = &self.display_base {
                    relativize_file_fields(&mut value, base);
                }
                let rel_file = value
                    .get("file")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let fingerprint = self.fingerprint_for(diag, &rel_file);
                if self.tamped {
                    // Divert to the internal buffer — `finish()` sorts
                    // and writes at end-of-run. Streaming would break
                    // the byte-lex sort contract the tamp mode owes to
                    // baseline consumers.
                    self.tamp_buffer.push(TampedRecord {
                        file: rel_file,
                        code: diag.kind.code().to_string(),
                        fingerprint,
                    });
                    return;
                }
                if let Some(obj) = value.as_object_mut() {
                    obj.insert("fingerprint".into(), json!(fingerprint));
                }
                let line =
                    serde_json::to_string(&value).expect("diagnostic must serialize to JSON");
                if let Err(err) = writeln!(self.writer, "{}", line) {
                    if err.kind() == std::io::ErrorKind::BrokenPipe {
                        self.broken = true;
                        return;
                    }
                    panic!("diagnostic emit failed: {}", err);
                }
            }
        }
    }

    /// Flush the tamp buffer as sorted stable-projection JSONL. No-op on
    /// a non-tamped emitter or a `broken` (BrokenPipe-seen) writer.
    /// Called at every process-exit path before `drop(emitter)` so all
    /// four `Commands::Check` exits (config-infusion error, warm-build
    /// failure, cold-build failure, normal end) flush the buffer before
    /// the writer is torn down. Idempotent (the sort + write drains the
    /// buffer). Per-line write errors are surfaced the same way the
    /// streaming path surfaces them: BrokenPipe silently flips `broken`
    /// so `crema check --tamp | head -n 5` stays clean, other IO errors
    /// panic.
    pub fn finish(&mut self) {
        if !self.tamped || self.broken || self.tamp_buffer.is_empty() {
            self.tamp_buffer.clear();
            return;
        }
        // Render each record to its full JSON line first, THEN sort
        // the rendered byte sequences. Sorting the structured
        // `(file, code, fingerprint)` tuple would diverge from
        // `LC_ALL=C sort` on the emitted file whenever a `file` value
        // contains bytes that JSON-escapes: filenames legal on Unix
        // may carry `"`, `\`, or control bytes, and the raw byte order
        // then differs from the post-escape byte order. Sorting the
        // rendered rows is what the README promises external tools
        // ("preserve the same ordering" under merged-baseline
        // re-sort).
        //
        // Hand-serialize with the specific key order (`file` → `code`
        // → `fingerprint`). serde_json's Value serializer is
        // alphabetical by default (`code` first), which would silently
        // break the sort-contract; `preserve_order` would affect the
        // streaming path too. `serde_json::to_string` on each scalar
        // handles JSON escapes and Unicode correctly.
        let mut lines: Vec<String> = self
            .tamp_buffer
            .drain(..)
            .map(|rec| {
                let file_json =
                    serde_json::to_string(&rec.file).expect("string must serialize to JSON");
                let code_json =
                    serde_json::to_string(&rec.code).expect("string must serialize to JSON");
                let fp_json =
                    serde_json::to_string(&rec.fingerprint).expect("string must serialize to JSON");
                format!(
                    "{{\"file\":{},\"code\":{},\"fingerprint\":{}}}",
                    file_json, code_json, fp_json
                )
            })
            .collect();
        lines.sort();
        for line in lines {
            if let Err(err) = writeln!(self.writer, "{}", line) {
                if err.kind() == std::io::ErrorKind::BrokenPipe {
                    self.broken = true;
                    return;
                }
                panic!("tamp emit failed: {}", err);
            }
        }
    }

    pub fn had_any_error(&self) -> bool {
        self.had_any_error
    }

    /// Stable identifier for a diagnostic across runs, unrelated edits
    /// (line shifts) and checkout locations. Material:
    /// `rel_path \0 code \0 scope \0 line_bytes \0 column_u32_le`
    /// hashed with seedless xxh3_64, rendered as 16 hex digits.
    /// `line_bytes` is the source bytes of the line(s) containing the
    /// diagnostic's byte range — from the start of the line containing
    /// `start_byte` to the end of the line containing `end_byte`,
    /// exclusive of the trailing `\n` (a preceding `\r` from CRLF is
    /// kept). `column_u32_le` is `start_byte - line_start` (0-based
    /// byte offset within the starting line) encoded little-endian.
    /// Widening the material from the diag's own byte range to the
    /// enclosing line lets `keys << x` and `vals << y` fingerprint
    /// differently even though the diagnostic itself only spans the
    /// `<<` operator. The trailing column offset keeps
    /// `a.nmae + b.wat` (two `NoMethod`s on the same physical line)
    /// distinct: their line bytes are identical, so line bytes alone
    /// would collapse them. Line numbers, message text and absolute
    /// paths are deliberately excluded — each would break stability
    /// across edits, crema versions, or machines. Identical
    /// `(path, code, scope, line_bytes, column)` yields identical
    /// fingerprints (no ordinal suffix); that collision is an accepted
    /// limit of the scheme. Line-internal reformatting (indent change,
    /// identifier rename, comment edit) changes the fingerprint by
    /// design — "editing a line makes it a different diagnostic" is
    /// the intended semantics; indentation shifts move the column
    /// offset in lockstep with `line_bytes`, so the semantics stay
    /// line-scoped, not column-scoped. Path stability assumes
    /// `display_base` resolved (`rel_file` already relativized); if
    /// cwd resolution failed the whole run degrades to absolute paths
    /// and fingerprints are only machine-local.
    ///
    /// The `line_starts_by_file` / `fallback_line_starts` caches are
    /// consulted for line boundaries via `partition_point` (O(log n))
    /// rather than scanning raw source bytes for `\n` on every diag —
    /// `emit()` calls `line_for()` before `fingerprint_for()`, so by
    /// the time we get here one of those two caches is populated for
    /// the file (either registered up-front or filled by `line_for`'s
    /// disk-fallback path).
    fn fingerprint_for(&mut self, diag: &Diagnostic, rel_file: &str) -> String {
        let mut material = Vec::new();
        material.extend_from_slice(rel_file.as_bytes());
        material.push(0);
        material.extend_from_slice(diag.kind.code().as_bytes());
        material.push(0);
        if let Some(scope) = &diag.scope {
            material.extend_from_slice(scope.as_bytes());
        }
        material.push(0);
        if !self.sources_by_file.contains_key(&diag.location.file) {
            self.fingerprint_sources
                .entry(diag.location.file.clone())
                .or_insert_with(|| std::fs::read(&diag.location.file).ok());
        }
        let source = self
            .sources_by_file
            .get(&diag.location.file)
            .map(|s| &s[..])
            .or_else(|| {
                self.fingerprint_sources
                    .get(&diag.location.file)?
                    .as_deref()
            });
        let starts = self
            .line_starts_by_file
            .get(&diag.location.file)
            .or_else(|| {
                self.fallback_line_starts
                    .get(&diag.location.file)
                    .and_then(Option::as_ref)
            });
        if let (Some(source), Some(starts)) = (source, starts) {
            let start = (diag.location.range.start_byte as usize).min(source.len());
            let end = (diag.location.range.end_byte as usize)
                .min(source.len())
                .max(start);
            // Line index containing byte `b` = last `starts[i]` with
            // `starts[i] <= b`. `starts` is monotonically increasing
            // and always begins with 0, so partition_point returns
            // >= 1 and the saturating_sub is a defensive no-op.
            let start_line_idx = starts
                .partition_point(|&s| (s as usize) <= start)
                .saturating_sub(1);
            let end_line_idx = starts
                .partition_point(|&s| (s as usize) <= end)
                .saturating_sub(1);
            let line_start = starts[start_line_idx] as usize;
            // End of last spanned line = start of next line - 1 (strip
            // `\n`), or source.len() when the diag lands on the final
            // line and there is no next entry.
            let line_end = starts
                .get(end_line_idx + 1)
                .map(|&s| (s as usize).saturating_sub(1))
                .unwrap_or(source.len())
                .max(line_start);
            if line_start < line_end {
                material.extend_from_slice(&source[line_start..line_end]);
            }
            material.push(0);
            let column = (start - line_start) as u32;
            material.extend_from_slice(&column.to_le_bytes());
        }
        format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&material))
    }

    fn line_for(&mut self, diag: &Diagnostic) -> usize {
        if let Some(starts) = self.line_starts_by_file.get(&diag.location.file) {
            return line_from_starts(starts, diag.location.range.start_byte);
        }
        if let Some(cached) = self.fallback_line_starts.get(&diag.location.file) {
            return match cached {
                Some(starts) => line_from_starts(starts, diag.location.range.start_byte),
                None => 1,
            };
        }
        // Cache miss: pay the disk read + PathBuf clone once per file.
        let starts = std::fs::read(&diag.location.file)
            .ok()
            .map(|source| line_starts(&source));
        let line = match &starts {
            Some(starts) => line_from_starts(starts, diag.location.range.start_byte),
            None => 1,
        };
        self.fallback_line_starts
            .insert(diag.location.file.clone(), starts);
        line
    }
}

/// Rewrite every `"file"` string field in a serialized diagnostic to be
/// relative to `base` when it lies under it (paths outside `base` — e.g.
/// a walk-up crema.toml whose scope sits above the cwd — stay absolute).
/// Applied recursively so the nested location shapes (`duplicate_source`,
/// `conflicting_source`, `primary_source`) get the same display rule as
/// the top-level `"file"`; `"file"` is the single path-carrying key in
/// the diagnostic JSON contract, so keying on the name is exact, not
/// heuristic.
fn relativize_file_fields(value: &mut Value, base: &Path) {
    match value {
        Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if key == "file"
                    && let Value::String(s) = v
                    && let Ok(rel) = Path::new(s.as_str()).strip_prefix(base)
                {
                    *v = Value::String(rel.to_string_lossy().into_owned());
                } else {
                    relativize_file_fields(v, base);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                relativize_file_fields(item, base);
            }
        }
        _ => {}
    }
}

/// Sort environment-level diagnostics (parser / inline / infusion /
/// validate — everything emitted before the per-file check loop) into a
/// canonical `(file, start_byte, code, message)` lexicographic order so
/// crema's stdout does not leak the FxHashMap iteration order or the
/// split between G-snapshot cold and warm walks. The per-file check
/// stream stays untouched (file-completion streaming is documented at
/// the emit site in `main.rs`).
///
/// The message tiebreak formats the `DiagnosticKind` alone, not the
/// whole `Diagnostic`: the containing `Diagnostic::Display` calls
/// `fallback_line()` which does an unmemoized `fs::read` of the
/// location's source, and pulling filesystem I/O into a sort
/// comparator would both violate the documented sort key
/// `(file, start_byte, code, message)` and clash with the fallback-
/// line memoize done by `DiagnosticEmitter::line_for`.
pub fn sort_by_canonical_order(diags: &mut [Diagnostic]) {
    diags.sort_by(|a, b| {
        a.location
            .file
            .cmp(&b.location.file)
            .then_with(|| {
                a.location
                    .range
                    .start_byte
                    .cmp(&b.location.range.start_byte)
            })
            .then_with(|| a.kind.code().cmp(b.kind.code()))
            .then_with(|| a.kind.to_string().cmp(&b.kind.to_string()))
    });
}

fn line_starts(source: &[u8]) -> Vec<u32> {
    let mut starts = vec![0];
    for (idx, byte) in source.iter().enumerate() {
        if *byte == b'\n' {
            starts.push((idx + 1) as u32);
        }
    }
    starts
}

fn line_from_starts(starts: &[u32], offset: u32) -> usize {
    starts.partition_point(|start| *start <= offset)
}

