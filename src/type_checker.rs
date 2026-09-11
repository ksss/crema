mod calls;
mod cond_env;
mod constraints;
mod inference;
pub mod interface;
mod method_resolver;
mod methods;
mod visitor;


use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::path::PathBuf;

use ruby_prism::Visit;

use crate::ast::ruby::LineIndex;
use crate::context::{Context, ScopeSnapshot};
use crate::definition::Method;
use crate::definition_builder::{ConsultationLog, ConsultationView, DefinitionBuilder};
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::inline_parser::CommentAssociation;
use crate::name::NameTable;
use crate::pure_call_env::PureKey;
use crate::type_name::TypeName;
use crate::types::{FunctionType, MethodType, Ty};

/// Mirrors Steep's `SPECIAL_LVAR_NAMES = Set[:_, :__any__, :__skip__]`
/// (`lib/steep/type_construction.rb:40`). Writes to these names produce
/// untyped without binding into the env, and reads short-circuit to
/// untyped, enabling the `(_ = nil) #: T` cast idiom.
pub(crate) fn is_special_lvar_name(name: &[u8]) -> bool {
    matches!(name, b"_" | b"__any__" | b"__skip__")
}

/// Raw `(start_byte, end_byte)` source span carried by the type-check
/// path in place of a `SourceLocation`. Materialization to a
/// `SourceLocation` costs an O(offset) `byte_to_char_offset` scan from
/// the source start (run twice — once per side) plus a `PathBuf` clone;
/// holding the raw byte pair lets the hot paths skip both costs when
/// no diagnostic actually fires. Used for argument spans
/// (`positional_spans`, keyword `_, _, ArgSpan, ArgSpan`), call-site
/// spans (`check_call_arguments::loc_span`), and any other site that
/// only needs the bytes until a diagnostic-emit site lifts them via
/// [`TypeChecker::span_to_location`].
type ArgSpan = (u32, u32);

/// Record an argument's source span as raw `(start_byte, end_byte)` without
/// building a `SourceLocation`. See [`ArgSpan`] for why materialization is
/// deferred to diagnostic-emit time.
fn arg_span(start_byte: usize, end_byte: usize) -> ArgSpan {
    (start_byte as u32, end_byte as u32)
}

#[derive(Debug, Clone)]
pub(super) struct CallArguments {
    positional: Vec<Ty>,
    positional_spans: Vec<ArgSpan>,
    keywords: Vec<(String, Ty, ArgSpan, ArgSpan)>,
    /// `**h` elements of the braceless keyword hash, as (value type,
    /// span). Only `fold_braceless_keywords_for` reads them: a
    /// keyword-less overload folds them (with `keywords`) into one
    /// positional `Hash[K, V]`. Keyword-bearing overloads ignore them
    /// (Steep's `KeywordArgs::SplatArg` side is a separate todo), so
    /// `f(**h)` against `(a: Integer)` still reports the missing `a`.
    kwsplats: Vec<(Ty, ArgSpan)>,
    has_block: bool,
    explicit_type_args: Option<Vec<Ty>>,
    /// Unexpandable trailing splat: `f(a, *xs)` where `xs: Array[E]` (or
    /// untyped) leaves `Some(SplatTail { element_ty: E })` here after the
    /// expandable forms (ArrayNode literal, `Type::Tuple` value) have
    /// already been folded into `positional`. The downstream arity / per-arg
    /// path treats the tail as "consume the rest slot" (Steep's `uniform_type`
    /// path) or — when the overload has no rest — as `UnexpectedPositionalArgument`.
    splat_tail: Option<SplatTail>,
}

#[derive(Debug, Clone)]
struct SplatTail {
    /// Element type to match against the overload's rest slot. `Ty::UNTYPED`
    /// for the conservative bucket (Union / Optional / opaque receiver
    /// values), which suppresses arity false positives without making the
    /// per-arg subtype check meaningful.
    element_ty: Ty,
    span: ArgSpan,
}

#[derive(Debug, Clone)]
struct MethodTarget {
    method_name: String,
    method_def: Method,
}

/// One resolved component of a `Type::Union` receiver. Carries everything
/// the return-type path needs to re-run inference for that member.
#[derive(Debug, Clone)]
pub(super) struct UnionComponent {
    method_def: Method,
    bindings: FxHashMap<crate::type_param::TypeVarKey, Ty>,
    /// The union member type, used to build the `self`/`instance`/`class`
    /// substitution for this component's return type.
    receiver_type: Ty,
}

#[derive(Debug, Clone)]
enum CallTarget {
    Method {
        method_name: String,
        /// Class of the receiver this call dispatches against. The
        /// CallNode path sets this to the resolved receiver class; the
        /// SuperNode synthetic-target path (calls.rs:check_super_node)
        /// sets it to the caller's own self class — sufficient for the
        /// only current consumer (`verbose_log_call_resolved`), but read
        /// with care from new consumers because the super case doesn't
        /// carry the super-target's defining class.
        receiver_class: TypeName,
        method_def: Method,
        /// Pre-built `{type_param -> concrete_ty}` bindings from walking the
        /// ancestor chain. Needed because the method may have been declared on
        /// a parent class whose type params differ from `receiver_class`, as
        /// in `class Child < Parent[String]`.
        bindings: FxHashMap<crate::type_param::TypeVarKey, Ty>,
    },
    /// A `Type::Union` receiver whose called name resolves on *every*
    /// component (ADR-0021 per-name dispatch). The return type unions each
    /// component's return; argument checking against the union
    /// (`MethodType.union`) is a separate slice and is intentionally not
    /// performed here, so `method_definition` returns `None` to short-circuit
    /// the arg-check path.
    UnionMethod {
        method_name: String,
        components: Vec<UnionComponent>,
    },
}

impl CallTarget {
    fn method_name(&self) -> &str {
        match self {
            CallTarget::Method { method_name, .. } => method_name,
            CallTarget::UnionMethod { method_name, .. } => method_name,
        }
    }

    fn method_definition(&self) -> Option<&Method> {
        match self {
            CallTarget::Method { method_def, .. } => Some(method_def),
            CallTarget::UnionMethod { .. } => None,
        }
    }

    /// True iff every dispatch target this call could resolve to is pure
    /// (ADR-0024). For `Method` this is the single resolved def; for
    /// `UnionMethod` (union receiver, ADR-0021 per-name dispatch) *every*
    /// component must be pure -- a mix would let a second call on the
    /// non-pure member return a different value, silently invalidating a
    /// cached narrow. Steep parity confirmed via steep-playground: a union
    /// with one pure and one non-pure member does NOT narrow (only the
    /// unrelated `Integer | nil` NoMethod on the un-narrowed call survives).
    fn is_pure(&self, name: crate::name::Symbol, names: &crate::name::NameTable) -> bool {
        match self {
            CallTarget::Method { method_def, .. } => method_def.is_pure(name, names),
            CallTarget::UnionMethod { components, .. } => {
                components.iter().all(|c| c.method_def.is_pure(name, names))
            }
        }
    }

    fn bindings(&self) -> FxHashMap<crate::type_param::TypeVarKey, Ty> {
        match self {
            CallTarget::Method { bindings, .. } => bindings.clone(),
            CallTarget::UnionMethod { .. } => FxHashMap::default(),
        }
    }
}

/// Options for `check_source()`.
///
/// New options can be added without changing call sites that use `Default`.
#[derive(Debug, Clone)]
pub struct CheckOptions {
    /// Print type checker decisions to stderr.
    pub verbose: bool,
    /// Honor inline declaration annotations. Expression assertions are
    /// consumed independently of this flag.
    pub inline: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions {
            verbose: false,
            inline: true,
        }
    }
}

/// Walks a Prism AST and emits type diagnostics.
pub struct TypeChecker<'env> {
    env: ConsultationView<'env>,
    ctx: Context,
    file: PathBuf,
    source: Vec<u8>,
    /// Precomputed newline-offset index over `source`. Lets per-call line
    /// lookups in hot paths like `lookup_callsite_type_args` avoid re-scanning
    /// the source on every Prism node.
    line_index: LineIndex,
    diagnostics: RefCell<Vec<Diagnostic>>,
    options: CheckOptions,
    /// Inline comment index for trailing `#: T` lookup. Built once per source
    /// in `check_source` and shared for the whole walk.
    comments: CommentAssociation,
    /// When true, the statement-position gate
    /// (`apply_statement_assertion_gate` driven by
    /// `check_statements_with_hint`) suppresses the last stmt's
    /// `FalseAssertion` emission because the def-body return-type
    /// path (`check_return_type` in methods.rs) owns that emission for
    /// the body's last expression. Flipped on entry to the body walk
    /// in `check_def_node_in_current_context` and cleared as
    /// `check_statements_with_hint` recurses, so nested
    /// StatementsNodes (Parens body, Begin body, etc.) inside the
    /// def's last stmt still fire their gate normally. OR'd with
    /// `suppress_parens_routed_last_assertion` at the consume site;
    /// see that field's doc for why the two stay as separate bools
    /// rather than a single enum.
    suppress_method_body_last_assertion: bool,
    /// Sibling of `suppress_method_body_last_assertion` for the
    /// parens-routed dup. Named after the prototypical trigger (an
    /// lvasgn-wrapped Parens), but the mechanism covers any RHS that
    /// re-enters `check_statements_with_hint` with the forwarded hint
    /// — Begin, If arms, etc. Set by `visit_local_variable_write_node`
    /// when the lvasgn has consumed a trailing `#: T` and is about to
    /// walk the RHS. Without the flag, the inner gate would re-emit a
    /// `FalseAssertion` for the same trailing comment that the lvasgn
    /// route already owns. Steep parity (`source.rb:527-535`): the
    /// outer Parens owns the assertion and inner statements are
    /// stripped of trailing comments before recursion. Cleared
    /// (`mem::replace(..., false)`) on entry to the outermost
    /// `check_statements_with_hint`, so nested bodies inside the
    /// suppressed stmt (e.g. `x = (begin; 1 #: T; end) #: U` — the
    /// inner Begin's body) still run their gate normally. OR'd with
    /// `suppress_method_body_last_assertion` at the consume site;
    /// both flags share the same "skip the last stmt's gate exactly
    /// once" semantics but originate from independent set-sites (def
    /// body vs lvasgn), so they stay as separate bools rather than a
    /// single enum — nested `def f; x = (...) #: T; end` can set both
    /// legitimately.
    suppress_parens_routed_last_assertion: bool,
    /// Transient local-variable overlays consulted by `infer_type` before the
    /// regular `ctx` lookup. Used to bind block parameter names during
    /// call-return inference without pushing a real scope (which would require
    /// `&mut self`). Lifetime is scoped by `with_overlay`; no persistence.
    overlay_stack: std::cell::RefCell<Vec<FxHashMap<crate::name::Name, Ty>>>,
    pure_overlay_stack: std::cell::RefCell<Vec<FxHashMap<PureKey, Ty>>>,
    /// Interner for session-local names: local variable names, block/keyword
    /// parameter names added during type checking. Kept separate from
    /// `env.names()` so the frozen environment's `NameTable` can eventually
    /// become a `RodeoReader`.
    checker_names: NameTable,
    /// Recursion guard for `unify_interface_into_bindings`: unlike every
    /// other `unify_into_bindings` arm, which only ever recurses into the
    /// already-finite structure of the same `Ty` (bounded by the type
    /// expression's own size), the interface arm reaches for a fresh `Ty`
    /// via method-return-type dispatch that can structurally echo the
    /// input forever for a self-referential interface (e.g. `interface
    /// _Node[T]; def next: () -> _Node[T]; end`). Mirrors
    /// `SubtypeChecker`'s coinductive-assumption abort limit
    /// (`subtyping.rs`) with a plain depth counter: unify only needs to
    /// stop, not stay sound under the cycle, so bailing to "no binding"
    /// past the cap is always safe.
    interface_unify_depth: std::cell::Cell<u32>,
    /// `Some` only under `check_source_extract`: the site sink for
    /// `crema extract`. `None` (every `crema check` run) makes the
    /// record hooks a single branch, mirroring `options.verbose`'s cost
    /// profile on the hot path.
    extract: Option<crate::extract::ExtractSitesCollector>,
    /// Extract-mode per-call state flags, keyed by the call node's
    /// byte span and consumed at the walk-entry record points. The
    /// check paths flag a span when the call's own diagnostics fired
    /// (a diagnostic-count delta over the call's own check segments —
    /// never span containment, so a block-body diagnostic cannot error
    /// the outer call) or when a no-target call was skipped (untyped
    /// receiver) / diagnosed (NoMethod). Empty unless `extract` is
    /// `Some`; recording is bookkeeping only (no env queries).
    extract_call_states: RefCell<FxHashMap<(u32, u32), ExtractCallState>>,
    /// Extract-mode hand-off of check-computed expression types to the
    /// Visit-descent record point (`visit_call_node`), keyed by the
    /// consuming call node's byte span. The check deposits a value only
    /// when its own frame already computed it — the receiver type held
    /// by `check_call`, the per-argument types built by
    /// `collect_arguments_from`, the block-body type computed by
    /// `check_block` — and never re-infers for the sake of this map
    /// (that would widen `consulted` beyond "what the check
    /// consulted"). Span keying makes the mismatch cases (paren
    /// wrappers, splat-folded elements) fall out as "no entry" — a
    /// site the check computed no value for stays `return_type: null`.
    /// Empty unless `extract` is `Some`.
    extract_carried_types: RefCell<FxHashMap<(u32, u32), Ty>>,
    /// Argument-channel deposit gate for [`Self::extract_carried_types`].
    /// The argument collectors run from several frames — the canonical
    /// check pass in `check_call_arguments` (whose collected values ARE
    /// consumed by `check_call`'s argument visit right after), but also
    /// auxiliary re-collections (`lookup_block_type`,
    /// `infer_call_return_type`, yield/super paths) whose deposits
    /// would be orphans (their spans were already consumed) or, worse,
    /// per-union-component trial-hint values that the check never
    /// settled on. Armed only around the canonical collection;
    /// explicitly disarmed across the union-dispatch loops (a
    /// last-write-wins trial value is not "what the check computed" —
    /// better null). Receiver and block-tail deposits bypass this gate:
    /// their deposit sites are already exact.
    extract_carry_armed: std::cell::Cell<bool>,
}

/// Pending `state` classification for one extract-mode call site; the
/// JSON vocabulary lives in [`crate::extract::method_call_state`].
/// Absence of a flag means `typed` for resolved sites and "not
/// recorded" for no-target sites (the silent boundary: bot receivers,
/// classifier-gated NoMethod).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtractCallState {
    Error,
    Untyped,
    NoMethodError,
}

impl<'env> TypeChecker<'env> {
    fn new(
        env: ConsultationView<'env>,
        file: PathBuf,
        source: Vec<u8>,
        line_index: LineIndex,
        comments: CommentAssociation,
        options: CheckOptions,
    ) -> Self {
        TypeChecker {
            env,
            ctx: Context::new(),
            file,
            source,
            line_index,
            diagnostics: RefCell::new(Vec::new()),
            comments,
            suppress_method_body_last_assertion: false,
            suppress_parens_routed_last_assertion: false,
            options,
            overlay_stack: RefCell::new(Vec::new()),
            pure_overlay_stack: RefCell::new(Vec::new()),
            checker_names: NameTable::new(),
            interface_unify_depth: std::cell::Cell::new(0),
            extract: None,
            extract_call_states: RefCell::new(FxHashMap::default()),
            extract_carried_types: RefCell::new(FxHashMap::default()),
            extract_carry_armed: std::cell::Cell::new(false),
        }
    }

    /// Flag the call at `span` with a pending extract state. No-op on
    /// the plain check path (`extract` is `None`).
    pub(super) fn extract_flag_call_state(&self, span: (u32, u32), state: ExtractCallState) {
        if self.extract.is_some() {
            self.extract_call_states.borrow_mut().insert(span, state);
        }
    }

    /// Consume the pending extract state for the call at `span`.
    fn extract_take_call_state(&self, span: (u32, u32)) -> Option<ExtractCallState> {
        self.extract_call_states.borrow_mut().remove(&span)
    }

    /// Deposit a check-computed expression type for the call at `span`,
    /// to be consumed by `visit_call_node`'s record point. No-op on the
    /// plain check path (`extract` is `None`). See
    /// [`Self::extract_carried_types`] for the deposit discipline.
    pub(super) fn extract_carry_type(&self, span: (u32, u32), ty: Ty) {
        if self.extract.is_some() {
            self.extract_carried_types.borrow_mut().insert(span, ty);
        }
    }

    /// Consume the carried expression type for the call at `span`.
    pub(super) fn extract_take_carried_type(&self, span: (u32, u32)) -> Option<Ty> {
        self.extract_carried_types.borrow_mut().remove(&span)
    }

    /// Current diagnostic count, for the extract-mode delta that
    /// attributes a diagnostic burst to the call segment that emitted
    /// it.
    pub(super) fn diagnostics_len(&self) -> usize {
        self.diagnostics.borrow().len()
    }

    /// Whether any diagnostic pushed since `before` (a prior
    /// `diagnostics_len()` snapshot) is a `NoMethod`. Used by the
    /// extract-mode `no_method_error` gate to distinguish an actual
    /// NoMethod diagnostic from other kinds (e.g. `NotImplementedYet`)
    /// that can also fire inside the same count window.
    pub(super) fn diagnostics_since_include_no_method(&self, before: usize) -> bool {
        self.diagnostics.borrow()[before..]
            .iter()
            .any(|d| matches!(d.kind, DiagnosticKind::NoMethod { .. }))
    }

    /// Landing symbol for one resolved method at a site: the distinct
    /// `implemented_in` owners across `defs` (the axis extract's
    /// `symbol` is defined on), rendered as `::Owner#name` /
    /// `::Owner.name`. Usually one owner; an overloading extension
    /// (`def bar: (...) | ...`) merges defs from several classes and a
    /// site through it really runs code from each, so each owner gets
    /// its own symbol.
    /// An interface-typed receiver has `implemented_in: None` (an
    /// interface declares without implementing); the declaring
    /// interface is the only nameable owner, so `defined_in` stands in.
    fn extract_landing_symbols(
        &self,
        method: &Method,
        method_name: &str,
        site_singleton: bool,
    ) -> Vec<String> {
        let mut owners: Vec<crate::type_name::TypeName> = Vec::new();
        for td in &method.defs {
            let owner = td.implemented_in.unwrap_or(td.defined_in);
            if !owners.contains(&owner) {
                owners.push(owner);
            }
        }
        let separator = if site_singleton { "." } else { "#" };
        owners
            .into_iter()
            .map(|owner| {
                format!(
                    "{}{}{}",
                    self.env.names().display_type_name(owner),
                    separator,
                    method_name
                )
            })
            .collect()
    }

    /// Records every resolved call site for `crema extract`. Called
    /// from the two walk entries for real `CallNode` sites (they know
    /// the site's expression type — see the note in
    /// `check_call_arguments`), plus the synthetic dispatch points for
    /// compound writes.
    ///
    /// `expr_type` is the site's expression type (the call's return
    /// type) when the check itself computed one, `None` otherwise.
    /// Extract never runs its own inference for this field: a synthesis
    /// the check does not perform would both widen `consulted` beyond
    /// "what the check consulted" and, worse, run outside the site's
    /// checked scope (block params unbound), producing wrong lookups.
    fn record_extract_call(
        &self,
        start_offset: usize,
        end_offset: usize,
        target: &CallTarget,
        receiver_type: Ty,
        expr_type: Option<Ty>,
    ) {
        let Some(collector) = &self.extract else {
            return;
        };
        let site_singleton = matches!(
            self.env.types().resolve(receiver_type),
            crate::types::Type::ClassSingleton { .. }
        );
        let mut symbols: Vec<String> = Vec::new();
        match target {
            CallTarget::Method {
                method_name,
                method_def,
                ..
            } => {
                symbols = self.extract_landing_symbols(method_def, method_name, site_singleton);
            }
            // Union receiver: the site dispatches into every component,
            // so each component's landing definition is "used" here.
            // Components sharing an implementation (a common ancestor's
            // method) dedupe to one symbol.
            CallTarget::UnionMethod {
                method_name,
                components,
            } => {
                for c in components {
                    for symbol in
                        self.extract_landing_symbols(&c.method_def, method_name, site_singleton)
                    {
                        if !symbols.contains(&symbol) {
                            symbols.push(symbol);
                        }
                    }
                }
            }
        }
        let span = (start_offset as u32, end_offset as u32);
        let state = match self.extract_take_call_state(span) {
            Some(ExtractCallState::Error) => crate::extract::method_call_state::ERROR,
            _ => crate::extract::method_call_state::TYPED,
        };
        let method_name = match target {
            CallTarget::Method { method_name, .. }
            | CallTarget::UnionMethod { method_name, .. } => method_name.clone(),
        };
        let receiver = self.display_type(receiver_type);
        let return_type = expr_type.map(|t| self.display_type(t));
        for symbol in symbols {
            collector.push_method_call(crate::extract::MethodCallRecord {
                state,
                kind: crate::extract::ReferenceKind::Call.as_str(),
                start_byte: span.0,
                end_byte: span.1,
                method_name: method_name.clone(),
                receiver_type: receiver.clone(),
                return_type: return_type.clone(),
                symbol: Some(symbol),
            });
        }
    }

    /// Super-side sibling of [`Self::record_extract_call`]: records a
    /// resolved `super` site. Called from both super resolution points
    /// — `check_super_node` (`super(args)`) and
    /// `emit_unexpected_super_if_unresolvable` (bare `super`) — on their
    /// lookup-success paths. `method` is the resolved super target; the
    /// site's singleton kind and the receiver column both come from the
    /// enclosing `self_ty`.
    fn record_extract_super(
        &self,
        start_offset: usize,
        end_offset: usize,
        method: &Method,
        method_name: &str,
        self_ty: Ty,
        expr_type: Option<Ty>,
    ) {
        let Some(collector) = &self.extract else {
            return;
        };
        let site_singleton = matches!(
            self.env.types().resolve(self_ty),
            crate::types::Type::ClassSingleton { .. }
        );
        let span = (start_offset as u32, end_offset as u32);
        let state = match self.extract_take_call_state(span) {
            Some(ExtractCallState::Error) => crate::extract::method_call_state::ERROR,
            _ => crate::extract::method_call_state::TYPED,
        };
        let receiver = self.display_type(self_ty);
        let return_type = expr_type.map(|t| self.display_type(t));
        for symbol in self.extract_landing_symbols(method, method_name, site_singleton) {
            collector.push_method_call(crate::extract::MethodCallRecord {
                state,
                kind: crate::extract::ReferenceKind::Super.as_str(),
                start_byte: span.0,
                end_byte: span.1,
                method_name: method_name.to_string(),
                receiver_type: receiver.clone(),
                return_type: return_type.clone(),
                symbol: Some(symbol),
            });
        }
    }

    /// Extract-mode record of every type a def's declared signature
    /// references. The consultation log alone cannot see these:
    /// annotation lowering resolved them during the env phase, and a
    /// body that never touches a parameter raises no env query for its
    /// type — yet for dependency-graph consumers the file plainly
    /// depends on them. Called from `enter_method_context` once the
    /// def's unified signature is known.
    fn record_extract_signature_types(&self, method_type: &MethodType) {
        let Some(collector) = &self.extract else {
            return;
        };
        let mut record = |name: crate::type_name::TypeName| collector.push_signature_type(name);
        self.walk_reference_type_names_in_fn(&method_type.type_, &mut record);
        if let Some(b) = &method_type.block {
            self.walk_reference_type_names_in_fn(&b.type_, &mut record);
            if let Some(t) = b.self_type {
                self.walk_reference_type_names(t, &mut record);
            }
        }
    }

    /// Record a `def` site as an implementation, on the lexical symbol
    /// axis (see [`crate::extract::ImplementsRecord`]): the enclosing
    /// class/module plus the def name, independent of signature
    /// resolution. Called from `visit_def_node` *before* the synthetic
    /// concern-target dispatch replaces the class stack, so a concern
    /// module's def records its lexical module once — not once per
    /// including class.
    pub(super) fn record_extract_implements<'pr>(&self, node: &ruby_prism::DefNode<'pr>) {
        let Some(collector) = &self.extract else {
            return;
        };
        let is_singleton = self.is_singleton_method_def(node);
        if let Some(receiver) = node.receiver() {
            // `def obj.foo` (and `def self.foo` inside `class << self`,
            // a singleton-of-singleton) has no statically-spellable
            // absolute rbs owner — skip, mirroring the checker's own
            // early-outs in `check_def_node_in_current_context`.
            if receiver.as_self_node().is_none() || self.ctx.in_singleton_class() {
                return;
            }
        }
        let owner = match self.ctx.current_class_typename() {
            Some(tn) => self.env.names().display_type_name(*tn),
            // Ruby semantics: a top-level `def` is a private instance
            // method on Object. (Inside any visited class body the
            // typename is always `Some` — `push_decl_class` skips the
            // body entirely when it cannot resolve a name.)
            None => self
                .env
                .names()
                .display_type_name(self.env.names().builtins().object),
        };
        let separator = if is_singleton { "." } else { "#" };
        let name = String::from_utf8_lossy(node.name().as_slice());
        collector.push_implements(crate::extract::ImplementsRecord {
            symbol: format!("{owner}{separator}{name}"),
            kind: if is_singleton {
                "singleton_method"
            } else {
                "instance_method"
            },
            start_byte: node.location().start_offset() as u32,
            end_byte: node.location().end_offset() as u32,
        });
    }

    /// Record a constant assignment (`FOO = 1` / `A::B = 2`) as an
    /// implementation of kind `constant`, on the same lexical axis as
    /// the def records. `path` is the LHS as written (`BAR`, `A::B`,
    /// `::A::B`); the symbol is the inline collector's
    /// `qualify_under(innermost class, path)` — rooted paths kept,
    /// everything else concatenated under the class stack's last entry
    /// (top-level is rooted `::X`, not `::Object::X`: rbs spells
    /// top-level constants that way and `definitions` agrees). Called
    /// from the head of `check_constant_write` /
    /// `check_constant_path_write` only — both the visitor arm and the
    /// `check_node` arm land there, and the `infer_type` oracle arms
    /// never do, so one site is one record. Callers pass `None` for a
    /// dynamic-parent path (`obj.foo::BAR = 1`), which has no absolute
    /// rbs spelling and is skipped like `def obj.foo`.
    pub(super) fn record_extract_constant_write(
        &self,
        start_offset: usize,
        end_offset: usize,
        path: Option<String>,
    ) {
        let Some(collector) = &self.extract else {
            return;
        };
        let Some(path) = path else {
            return;
        };
        let symbol = if path.starts_with("::") {
            path
        } else {
            match self.ctx.current_class_typename() {
                Some(tn) => format!("{}::{path}", self.env.names().display_type_name(*tn)),
                None => format!("::{path}"),
            }
        };
        collector.push_implements(crate::extract::ImplementsRecord {
            symbol,
            kind: "constant",
            start_byte: start_offset as u32,
            end_byte: end_offset as u32,
        });
    }

    /// Record a call site whose target did not resolve, from the same
    /// walk entries as [`Self::record_extract_call`]. The state comes
    /// from the flag `check_call_arguments`' no-target branch left for
    /// this span: `Untyped` (receiver untyped, resolution never
    /// attempted) or `NoMethodError` (receiver resolved, the NoMethod
    /// diagnostic actually fired). No flag means the site stayed
    /// silent (bot receiver, classifier-gated NoMethod) and is not
    /// recorded — the v1 silent boundary is preserved.
    fn record_extract_call_no_target<'pr>(
        &self,
        node: &ruby_prism::CallNode<'pr>,
        receiver_type: Ty,
        expr_type: Option<Ty>,
    ) {
        let Some(collector) = &self.extract else {
            return;
        };
        let span = (
            node.location().start_offset() as u32,
            node.location().end_offset() as u32,
        );
        let state = match self.extract_take_call_state(span) {
            Some(ExtractCallState::Untyped) => crate::extract::method_call_state::UNTYPED,
            Some(ExtractCallState::NoMethodError) => {
                crate::extract::method_call_state::NO_METHOD_ERROR
            }
            _ => return,
        };
        collector.push_method_call(crate::extract::MethodCallRecord {
            state,
            kind: crate::extract::ReferenceKind::Call.as_str(),
            start_byte: span.0,
            end_byte: span.1,
            method_name: String::from_utf8_lossy(node.name().as_slice()).into_owned(),
            receiver_type: self.display_type(receiver_type),
            return_type: expr_type.map(|t| self.display_type(t)),
            symbol: None,
        });
    }

    /// Record a constant read site for extract mode. `leaf` is the
    /// resolved constant when the walk produced one (`typed` / `error`
    /// states); `ty` is the type the check computed for the read.
    /// Called only from the side-effecting read paths
    /// (`check_constant_read` and the constant-path read finisher) —
    /// never from the silent `&self` resolvers, so `infer_receiver_type`
    /// re-peeks cannot duplicate a site, and never from write /
    /// superclass / declaration-head contexts (their emits don't route
    /// through the read paths). Bookkeeping only: no env queries.
    pub(super) fn record_extract_constant(
        &self,
        start_offset: usize,
        end_offset: usize,
        state: &'static str,
        path: String,
        leaf: Option<&crate::definition::ResolverConstant>,
        ty: Option<Ty>,
    ) {
        let Some(collector) = &self.extract else {
            return;
        };
        collector.push_constant(crate::extract::ConstantRecord {
            state,
            start_byte: start_offset as u32,
            end_byte: end_offset as u32,
            path,
            symbol: leaf.map(|c| self.env.names().display_type_name(c.name)),
            constant_type: ty.map(|t| self.display_type(t)),
        });
    }

    /// Run `f` with `overlay` pushed onto the overlay stack, popping
    /// unconditionally on return. Callers must use this helper rather than
    /// push/pop individually so the stack can never leak a frame.
    pub(super) fn with_overlay<R>(
        &self,
        overlay: FxHashMap<crate::name::Name, Ty>,
        f: impl FnOnce(&Self) -> R,
    ) -> R {
        self.overlay_stack.borrow_mut().push(overlay);
        let result = f(self);
        self.overlay_stack.borrow_mut().pop();
        result
    }

    pub(super) fn with_pure_overlay<R>(
        &self,
        overlay: FxHashMap<PureKey, Ty>,
        f: impl FnOnce(&Self) -> R,
    ) -> R {
        self.pure_overlay_stack.borrow_mut().push(overlay);
        let result = f(self);
        self.pure_overlay_stack.borrow_mut().pop();
        result
    }

    /// Look up `name` through the overlay stack (most-recently-pushed first).
    /// Returns `None` when no overlay binds it.
    pub(super) fn lookup_overlay(&self, name: crate::name::Name) -> Option<Ty> {
        self.overlay_stack
            .borrow()
            .iter()
            .rev()
            .find_map(|m| m.get(&name).copied())
    }

    pub(super) fn lookup_pure_overlay(&self, key: &PureKey) -> Option<Ty> {
        self.pure_overlay_stack
            .borrow()
            .iter()
            .rev()
            .find_map(|m| m.get(key).copied())
    }

    /// Run `f` under the narrowing described by `env`. An empty `env`
    /// is a transparent pass-through, matching the `(empty, empty)`
    /// shape `analyze_condition` returns for any predicate the current
    /// arm set cannot read.
    pub(super) fn with_cond_narrowing<R>(
        &mut self,
        env: &cond_env::CondEnv,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        use cond_env::Scrutinee;
        if env.is_empty() {
            return f(self);
        }
        let mut lvar_tokens = Vec::new();
        let mut pure_tokens = Vec::new();
        for (scrutinee, ty) in env.iter_lvars_then_pures() {
            match scrutinee {
                Scrutinee::Lvar(name) => {
                    lvar_tokens.push(self.ctx.enter_narrow(*name, ty));
                }
                Scrutinee::Pure(key) => {
                    pure_tokens.push(self.ctx.enter_pure_narrow(key.clone(), ty));
                }
            }
        }
        let r = f(self);
        for token in pure_tokens.into_iter().rev() {
            self.ctx.exit_pure_narrow(token);
        }
        for token in lvar_tokens.into_iter().rev() {
            self.ctx.exit_narrow(token);
        }
        r
    }

    /// Evaluate one arm of an if/unless in branch isolation: plant the
    /// narrowing on the innermost scope, run the body (whose
    /// assignments may land on any scope via Ruby's closure-writes-outer
    /// semantics), snapshot the arm-after env across all scopes, then
    /// rewind the entire scope chain to `base`. The returned
    /// `ScopeSnapshot` is what `Context::join_branches` consumes at the
    /// merge point.
    ///
    /// Unlike `with_cond_narrowing`, this path uses raw
    /// `set_local_variable` instead of `enter_narrow`/`exit_narrow` —
    /// the wholesale `restore_scopes(base)` at the end already undoes
    /// every write the arm made (narrow and assignment alike, inner and
    /// outer scope alike), so the `NarrowToken` bookkeeping that
    /// protects mid-narrow assignments in the substrate API is
    /// redundant here.
    pub(super) fn with_cond_branch<R>(
        &mut self,
        base: &ScopeSnapshot,
        narrowing: &cond_env::CondEnv,
        f: impl FnOnce(&mut Self) -> R,
    ) -> (ScopeSnapshot, R) {
        use cond_env::Scrutinee;
        for (scrutinee, ty) in narrowing.iter_lvars_then_pures() {
            match scrutinee {
                Scrutinee::Lvar(name) => self.ctx.set_local_variable(*name, ty),
                Scrutinee::Pure(key) => self.ctx.pure_call_env_mut().set(key.clone(), ty),
            }
        }
        let r = f(self);
        let after = self.ctx.snapshot_scopes();
        self.ctx.restore_scopes(base.clone());
        (after, r)
    }

    /// `with_cond_branch` that bundles the arm's after-snapshot and the
    /// body's last_ty into a [`BranchOutcome`] so the caller can hand it
    /// straight to `join_arms_with_divergence` without re-pairing the
    /// pieces. Used by if/unless/while/until — anywhere the divergence
    /// check matters.
    pub(in crate::type_checker) fn with_branch_outcome(
        &mut self,
        base: &ScopeSnapshot,
        narrowing: &cond_env::CondEnv,
        f: impl FnOnce(&mut Self) -> Ty,
    ) -> inference::BranchOutcome {
        let (after, last_ty) = self.with_cond_branch(base, narrowing, f);
        inference::BranchOutcome { after, last_ty }
    }

    pub fn diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics.into_inner()
    }

    /// Record a diagnostic to be returned from `check_source`.
    ///
    /// `NotImplementedYet` is deduplicated against `(line, column, site,
    /// subject_display)` to absorb call-path recursion: a single call
    /// site can reach the same `infer_type` catch-all twice (once from
    /// `ResolvedCall::resolve` -> `infer_receiver_type`, once from
    /// `check_no_method` -> `infer_receiver_type`). Without dedup an
    /// opt-in run would surface two identical lines for one source
    /// position, breaking `jq -r .subject_display` aggregation and
    /// reading as a phantom hot spot. The scan is O(N) but only runs
    /// for this dev-only kind, whose emissions are rare by definition.
    pub(super) fn push_diagnostic(&self, mut diagnostic: Diagnostic) {
        diagnostic.scope = self.current_scope();
        if let DiagnosticKind::NotImplementedYet {
            site,
            subject_display,
        } = &diagnostic.kind
        {
            let mut diags = self.diagnostics.borrow_mut();
            let is_dup = diags.iter().any(|d| {
                d.location == diagnostic.location
                    && matches!(
                        &d.kind,
                        DiagnosticKind::NotImplementedYet {
                            site: s,
                            subject_display: r,
                        } if s == site && r == subject_display
                    )
            });
            if !is_dup {
                diags.push(diagnostic);
            }
            return;
        }
        self.diagnostics.borrow_mut().push(diagnostic);
    }

    /// Fingerprint scope material from the current checking position:
    /// `::Foo::Bar#baz` / `::Foo::Bar.baz` inside a method body, the
    /// class path alone at class body level, `None` at top level.
    fn current_scope(&self) -> Option<String> {
        let class = self
            .ctx
            .current_class_typename()
            .map(|name| self.env.names().resolve(name));
        match (class, self.ctx.method_name()) {
            (Some(class), Some(method)) => {
                let sep = if self.ctx.is_singleton_method() {
                    "."
                } else {
                    "#"
                };
                Some(format!("{}{}{}", class, sep, method))
            }
            (Some(class), None) => Some(class),
            // Top-level defs land on Object (same fallback
            // `enter_method_context` uses for sig lookup).
            (None, Some(method)) => Some(format!("::Object#{}", method)),
            (None, None) => None,
        }
    }

    pub(super) fn offset_to_location(&self, offset: usize) -> crate::location::SourceLocation {
        self.byte_range_to_location(offset, offset)
    }

    pub(super) fn byte_range_to_location(
        &self,
        start_byte: usize,
        end_byte: usize,
    ) -> crate::location::SourceLocation {
        Diagnostic::location_for_byte_range(self.file.clone(), &self.source, start_byte, end_byte)
    }

    /// Lazily materialize an [`ArgSpan`] into a `SourceLocation`. Called only
    /// at diagnostic-emit time so the hot type-check path never pays the
    /// char-offset scan for arguments that check cleanly.
    pub(super) fn span_to_location(&self, span: ArgSpan) -> crate::location::SourceLocation {
        self.byte_range_to_location(span.0 as usize, span.1 as usize)
    }

    /// Interner for session-local names. Use instead of `env.names()` when
    /// interning local variable / block / keyword parameter names.
    pub(super) fn checker_names(&self) -> &NameTable {
        &self.checker_names
    }

    /// Format a type as a human-readable string.
    pub(super) fn display_type(&self, ty: Ty) -> String {
        ty.display(self.env.types(), self.env.names())
    }

    /// Render the call-site argument shape as an rbs-style signature
    /// fragment, used by `UnresolvedOverloading` diagnostics so users
    /// don't have to re-derive what they wrote.
    ///
    /// Layout mirrors rbs: positional types (each widened to its class
    /// so `"foo"` reads as `::String`, not `"foo"`), then a `*element`
    /// rest entry (matching how rbs spells `rest_positionals` — the
    /// element type, not the wrapping `Array[E]`, so users can compare
    /// against `method_types` letter-for-letter) when a splat tail is
    /// present, then `name: ty` keyword pairs in declaration order.
    /// `has_block` appends a trailing `{ ... }` — the block's element
    /// types aren't available at the call site, so the placeholder is
    /// intentionally opaque (over-engineering would invent type info
    /// that isn't there).
    ///
    /// `CallArguments` currently carries no kwsplat (`**opts`) tail, so
    /// the formatter has nothing to emit for it; once a kwsplat field is
    /// added (out of scope for this todo), extend this helper to mirror
    /// the positional `*tail` rendering.
    pub(super) fn display_call_arguments(&self, args: &CallArguments) -> String {
        let mut parts: Vec<String> = Vec::new();
        for ty in &args.positional {
            parts.push(self.display_type(self.widen_literal(*ty)));
        }
        if let Some(tail) = &args.splat_tail {
            if tail.element_ty == Ty::UNTYPED {
                parts.push("*untyped".to_string());
            } else {
                parts.push(format!(
                    "*{}",
                    self.display_type(self.widen_literal(tail.element_ty)),
                ));
            }
        }
        for (name, ty, _, _) in &args.keywords {
            parts.push(format!(
                "{}: {}",
                name,
                self.display_type(self.widen_literal(*ty)),
            ));
        }
        let block_suffix = if args.has_block { " { ... }" } else { "" };
        format!("({}){}", parts.join(", "), block_suffix)
    }

    /// Widen a Literal type to its class instance so call-argument
    /// display reads as a class (`::String`) rather than the literal
    /// value (`"foo"`). The user's RBS overloads are written in class
    /// types, so matching that vocabulary keeps the `Arguments:` /
    /// `Method types:` blocks visually aligned.
    ///
    /// Non-literal types pass through; widening Tuple / Record / Nil is
    /// out of scope (the values they represent are richer than a single
    /// class and would lose information if collapsed here).
    fn widen_literal(&self, ty: Ty) -> Ty {
        match self.env.types().resolve(ty) {
            crate::types::Type::Literal(lit) => self
                .env
                .class_instance_type(*lit.class_typename(self.env.names().builtins())),
            _ => ty,
        }
    }

    /// Render every overload of `method_def` as a bare `(...) -> R`
    /// string in declaration order. The `def NAME:` head is reattached
    /// once by the text-Display layer so JSON consumers stay free to
    /// reformat (no `def` prefix leaking into JSON values).
    pub(super) fn display_method_overloads(
        &self,
        method_def: &crate::definition::Method,
    ) -> Vec<String> {
        method_def
            .method_types()
            .map(|mt| self.display_method_type(mt))
            .collect()
    }

    /// Same as [`Self::display_method_overloads`] but flattens every
    /// component of a union receiver. Per-component overloads keep their
    /// declaration order; components are sorted by rendered receiver so
    /// the flattened list mirrors the canonical union display order (see
    /// `Type::Union` display). This avoids warm/cold divergence in
    /// `method_types` that would otherwise track the intern-order-based
    /// union member sequence.
    ///
    /// Components that inherit the same method unchanged from a common
    /// ancestor (e.g. two classes both `include`ing a module without
    /// overriding it) render identical signature strings; those are
    /// deduped to a single entry so the overload count reflects distinct
    /// signatures rather than how many components happen to expose them.
    /// Dedup is keyed on rendered signature *plus* each type param's bound
    /// (via [`Self::type_params_fingerprint`]), since `display_method_type`
    /// itself omits `[X < Bound]` — two overloads that render identically
    /// but constrain their type variable differently must not collapse.
    pub(super) fn display_union_method_overloads(
        &self,
        components: &[UnionComponent],
    ) -> Vec<String> {
        let mut per_component: Vec<(String, Vec<(String, String)>)> = components
            .iter()
            .map(|c| {
                let receiver = self.display_type(c.receiver_type);
                let overloads: Vec<(String, String)> = c
                    .method_def
                    .method_types()
                    .map(|mt| {
                        let rendered = self.display_method_type(mt);
                        let key = format!("{}{}", self.type_params_fingerprint(mt), rendered);
                        (rendered, key)
                    })
                    .collect();
                (receiver, overloads)
            })
            .collect();
        // Tie-break by the joined overload list so two components that
        // render the same receiver (e.g. two distinct type variables both
        // displaying as `T`) still sort deterministically. Without this,
        // the tie falls back to the input `components` order, which
        // ultimately derives from union member iteration — the very
        // intern-order dependency this fix targets.
        per_component.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut seen = std::collections::HashSet::new();
        per_component
            .into_iter()
            .flat_map(|(_, ols)| ols)
            .filter(|(_, key)| seen.insert(key.clone()))
            .map(|(rendered, _)| rendered)
            .collect()
    }

    /// Renders each type param's name and bounds for use as a dedup key
    /// fragment (never shown to the user — `display_method_type` is the
    /// user-facing rendering and intentionally stays bound-free).
    fn type_params_fingerprint(&self, method_type: &MethodType) -> String {
        method_type
            .type_params
            .iter()
            .map(|tp| {
                format!(
                    "{:?}<{}..{}>",
                    tp.name,
                    tp.upper_bound
                        .map(|t| self.display_type(t))
                        .unwrap_or_default(),
                    tp.lower_bound
                        .map(|t| self.display_type(t))
                        .unwrap_or_default(),
                )
            })
            .collect()
    }

    /// Format a method type as a human-readable signature string.
    /// e.g. `(String, Integer) -> void`, `(?) -> String`,
    /// `() { () -> ::Integer } -> void`
    pub(super) fn display_method_type(&self, method_type: &MethodType) -> String {
        display_method_type(method_type, self.env.types(), self.env.names())
    }

    /// Write a verbose log line to stderr when verbose mode is enabled.
    pub(super) fn verbose_log(&self, msg: impl std::fmt::Display) {
        if self.options.verbose {
            eprintln!("[verbose] {}", msg);
        }
    }
}

/// Free-function core of [`TypeChecker::display_method_type`], split
/// out so signature rendering is available without a checker instance
/// (`crema extract`'s definition records render from the environment
/// alone). One implementation keeps the extract and diagnostic
/// renderings from drifting apart.
pub fn display_method_type(
    method_type: &MethodType,
    types: &crate::types::TypeTable,
    names: &NameTable,
) -> String {
    let params = display_function_params(&method_type.type_, types, names);
    let block = match &method_type.block {
        Some(b) => {
            let self_binding = b
                .self_type
                .map(|st| format!(" [self: {}]", st.display(types, names)))
                .unwrap_or_default();
            format!(
                "{}{{ ({}){} -> {} }} ",
                if b.required { "" } else { "?" },
                display_function_params(&b.type_, types, names),
                self_binding,
                b.return_type().display(types, names),
            )
        }
        None => String::new(),
    };
    let ret = method_type.return_type().display(types, names);
    format!("({}) {}-> {}", params, block, ret)
}

/// Format a function's parameter list (the part between the
/// parentheses), `?` for an untyped function.
fn display_function_params(
    func: &FunctionType,
    types: &crate::types::TypeTable,
    names: &NameTable,
) -> String {
    match func {
        FunctionType::Untyped(_) => "?".to_string(),
        FunctionType::Typed(f) => {
            let mut parts = Vec::new();
            for ty in &f.required_positionals {
                parts.push(ty.display(types, names));
            }
            for ty in &f.optional_positionals {
                parts.push(format!("?{}", ty.display(types, names)));
            }
            if let Some(rest) = f.rest_positional {
                parts.push(format!("*{}", rest.display(types, names)));
            }
            for (name, ty) in &f.required_keywords {
                parts.push(format!("{}: {}", name, ty.display(types, names)));
            }
            for (name, ty) in &f.optional_keywords {
                parts.push(format!("?{}: {}", name, ty.display(types, names)));
            }
            if let Some(rest_kw) = f.rest_keyword {
                parts.push(format!("**{}", rest_kw.display(types, names)));
            }
            parts.join(", ")
        }
    }
}

/// Run type checking on a Ruby source file and return any diagnostics.
///
/// Files whose `parse_result` carries any Prism error are treated as
/// out-of-scope: walking the recovery AST would surface type errors on
/// input Ruby itself rejects with `SyntaxError`. The CLI driver
/// (`main.rs`) is expected to detect parse errors first and emit a
/// `Ruby::SyntaxError` diagnostic of its own; library callers that
/// reach this entry point with a broken `parse_result` get an empty
/// `Vec` back instead of a polluted diagnostic stream.
pub fn check_source(
    env: &DefinitionBuilder,
    file: PathBuf,
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    options: CheckOptions,
) -> Vec<Diagnostic> {
    check_source_with_log(env, file, source, parse_result, options, None)
}

/// Like [`check_source`], but routes checker reads through `log` when
/// given — the per-file consultation collection entry point
/// (ADR-0032 Decision 2, axis 3). `log: None` reproduces `check_source`
/// exactly (same discard-only view), so this is the only entry point
/// `cache_io`'s incremental recording needs to call.
pub fn check_source_with_log(
    env: &DefinitionBuilder,
    file: PathBuf,
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    options: CheckOptions,
    log: Option<&ConsultationLog>,
) -> Vec<Diagnostic> {
    if parse_result.errors().next().is_some() {
        return Vec::new();
    }
    let root = parse_result.node();

    let line_index = LineIndex::from_source(source);
    let comments = CommentAssociation::from_source(source, &line_index, parse_result);
    let view = ConsultationView::new(env, log);
    let mut checker = TypeChecker::new(view, file, source.to_vec(), line_index, comments, options);
    checker.visit(&root);
    checker.diagnostics()
}

/// Extract-mode sibling of [`check_source`]: runs the same walk with an
/// [`crate::extract::ExtractSitesCollector`] attached (and consultation
/// recording routed through `log`, like [`check_source_with_log`]) and
/// returns the collected sites instead of diagnostics (which are
/// discarded — extract's stdout carries only the summary line).
/// Parse-broken files return empty records, mirroring `check_source`'s
/// out-of-scope treatment.
pub fn check_source_extract(
    env: &DefinitionBuilder,
    file: PathBuf,
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    options: CheckOptions,
    log: Option<&ConsultationLog>,
) -> (
    Vec<crate::extract::MethodCallRecord>,
    Vec<crate::extract::ImplementsRecord>,
    Vec<crate::extract::ConstantRecord>,
    Vec<crate::type_name::TypeName>,
) {
    if parse_result.errors().next().is_some() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let root = parse_result.node();

    let line_index = LineIndex::from_source(source);
    let comments = CommentAssociation::from_source(source, &line_index, parse_result);
    let view = ConsultationView::new(env, log);
    let mut checker = TypeChecker::new(view, file, source.to_vec(), line_index, comments, options);
    checker.extract = Some(crate::extract::ExtractSitesCollector::new());
    checker.visit(&root);
    checker
        .extract
        .expect("set above, never cleared")
        .into_parts()
}
