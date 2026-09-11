use ruby_prism::{ArgumentsNode, CallNode, Location, Node, SuperNode, Visit};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::ruby::annotations::TypeApplicationAnnotation;
use crate::context::ScopeKind;
use crate::definition_builder;
use crate::diagnostic::{Diagnostic, DiagnosticKind, ForwardingMismatchKind};
use crate::inline_parser::TrailingAnnotation;
use crate::location::SourceLocation;
use crate::type_param::TypeParamScope;
use crate::types::{
    Block, Function, FunctionType, Literal, MethodType, RecordKey, Ty, Type, Visibility,
    intersection_of, union_of,
};

use super::method_resolver;
use super::{ArgSpan, CallArguments, CallTarget, SplatTail, TypeChecker, arg_span};

/// Abstraction over the two prism nodes that drive argument checking:
/// `CallNode` (regular method call) and `SuperNode` (explicit `super(args)`).
///
/// Shared helpers (`pick_hint_overload`, `setup_block_scope`,
/// `check_against_method_def`, etc.) were originally CallNode-only because
/// only that path needed them. Adding SuperNode parity for block-body
/// typecheck (Axis 1), Record hint propagation (Axis 2), and bound
/// diagnostics (Axis 3) requires running those helpers from the super
/// path too. `CallSite` exposes the small surface (`arguments`, `block`,
/// `location`) the helpers actually need, plus an `as_call_node()` escape
/// hatch for CallNode-only sub-features (selector annotations, message
/// location for diagnostics).
#[derive(Clone, Copy)]
pub(super) enum CallSite<'a, 'pr> {
    Call(&'a CallNode<'pr>),
    Super(&'a SuperNode<'pr>),
}

impl<'a, 'pr> CallSite<'a, 'pr> {
    pub(super) fn arguments(&self) -> Option<ArgumentsNode<'pr>> {
        match self {
            CallSite::Call(c) => c.arguments(),
            CallSite::Super(s) => s.arguments(),
        }
    }

    pub(super) fn block(&self) -> Option<Node<'pr>> {
        match self {
            CallSite::Call(c) => c.block(),
            CallSite::Super(s) => s.block(),
        }
    }

    pub(super) fn location(&self) -> Location<'pr> {
        match self {
            CallSite::Call(c) => c.location(),
            CallSite::Super(s) => s.location(),
        }
    }

    /// CallNode escape hatch. Returns `None` for SuperNode so helpers
    /// that need genuinely CallNode-specific state (selector annotations,
    /// `node.message_loc()`) can short-circuit without re-deriving the
    /// dispatch shape per call site. Adding a third escape route is a
    /// signal to dissolve the enum and keep dispatch at the caller — the
    /// abstraction only earns its keep while CallNode-only sub-features
    /// stay countable on one hand.
    pub(super) fn as_call_node(self) -> Option<&'a CallNode<'pr>> {
        match self {
            CallSite::Call(c) => Some(c),
            CallSite::Super(_) => None,
        }
    }
}

/// Outcome of the Tuple/Record literal-access specialization.
///
/// Both Tuple `[]` with a non-negative integer literal and Record `[]` with
/// a Symbol literal route through the same two outcomes: either the literal
/// resolves to a specific element/field (`Found`), or the literal is known
/// not to exist (`Missing`). For `Missing` we keep a usable fallback type
/// so downstream checks don't cascade into untyped.
pub(super) enum LiteralAccessResult {
    Found(Ty),
    Missing { fallback: Ty, diag: DiagnosticKind },
}

/// Result of classifying a NoMethod receiver. The single-receiver path
/// emits NoMethod on `Pass(widened)` (using the widened `Ty` so a
/// `Literal(1)` receiver reports `::Integer` instead of `1`), a dev-only
/// `Crema::NotImplementedYet` on `Unhandled` (to flag receiver kinds
/// without a NoMethod arm yet), and stays silent via `verbose_log` on
/// `Unnameable` (gate fail under partial RBS). The `Type::Union`
/// per-member path treats both non-`Pass` outcomes as "suppress the
/// whole union" conservatively.
pub(super) enum NoMethodReceiverGate {
    Pass(Ty),
    Unnameable,
    Unhandled,
}

/// Single-shot resolution of a call site — the target lookup result plus
/// the receiver's type, computed once per call.
///
/// `check_call` (the `&mut self` diagnostic-emission pass) and
/// `infer_call_return_type` (the `&self` return-type synth pass) both
/// need the same `(target, receiver_ty)` pair. Resolving in each pass
/// independently doubles the work and invites the two paths to drift
/// when call-resolution rules evolve. Callers should build a
/// `ResolvedCall` once and hand it to both passes.
#[derive(Debug, Clone)]
pub(super) struct ResolvedCall {
    pub target: Option<CallTarget>,
    pub receiver_ty: Ty,
}

impl ResolvedCall {
    pub(super) fn resolve<'env, 'pr>(checker: &TypeChecker<'env>, call: &CallNode<'pr>) -> Self {
        let receiver_ty = checker.infer_receiver_type(call);
        let target = checker.resolve_call_target(call, receiver_ty);
        Self {
            target,
            receiver_ty,
        }
    }
}

/// Classify the receiver expression for visibility purposes.
///
/// Returns `true` for bare calls (`foo()`) and calls on the literal `self`
/// (`self.foo`). Ruby ≥ 2.7 allows the latter even for private methods, and
/// crema matches that (what Ruby permits, crema permits). Any other
/// receiver — `obj.foo`, `Foo.new.foo`,
/// `self&.foo` with a non-self inner chain — counts as explicit and makes
/// a private method call illegal.
fn is_implicit_or_self_receiver<'pr>(node: &ruby_prism::CallNode<'pr>) -> bool {
    match node.receiver() {
        None => true,
        Some(r) => r.as_self_node().is_some(),
    }
}

/// Block-clause shape of one method overload, for the union-receiver
/// dispatch decision. Feeds the block case of Steep `MethodType#|`
/// (`method_type.rb:234-271`). The block-param arity is kept as a
/// `[min_arity, max_arity]` range (`max_arity == None` means a rest param,
/// i.e. unbounded) so optional/rest params don't read as a hard arity
/// mismatch — Steep combines block params via `b.params & ob.params`,
/// which succeeds whenever the ranges overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockShape {
    NoBlock,
    Block {
        required: bool,
        min_arity: usize,
        max_arity: Option<usize>,
    },
    /// Untyped method overload (`(...) -> untyped`): the method-param axis is
    /// the sibling `MethodType.union` slice, so on the block axis an untyped
    /// overload combines with anything and the union method survives —
    /// keeping the "don't emit a false NoMethod" stance.
    Untyped,
}

/// Overlap two arity ranges (`max == None` is unbounded). `None` when they
/// are disjoint — Steep's `return unless b.params & ob.params`.
fn overlap_arity(
    a: (usize, Option<usize>),
    b: (usize, Option<usize>),
) -> Option<(usize, Option<usize>)> {
    let lo = a.0.max(b.0);
    let hi = match (a.1, b.1) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    };
    match hi {
        Some(hi) if hi < lo => None,
        _ => Some((lo, hi)),
    }
}

/// Combine two block shapes per Steep `MethodType#|`'s block case. `None`
/// means the pair fails to combine — Steep's `else return` (one required
/// block + one blockless) or `return unless b.params & ob.params` (two
/// block-bearing shapes whose param arity ranges are disjoint).
fn combine_block_shapes(a: BlockShape, b: BlockShape) -> Option<BlockShape> {
    use BlockShape::*;
    match (a, b) {
        (Untyped, other) | (other, Untyped) => Some(other),
        (NoBlock, NoBlock) => Some(NoBlock),
        // One side blockless: keep an optional block (Steep `when b.optional?
        // → b`); a required block has no blockless counterpart so it fails.
        (
            NoBlock,
            block @ Block {
                required: false, ..
            },
        )
        | (
            block @ Block {
                required: false, ..
            },
            NoBlock,
        ) => Some(block),
        (NoBlock, Block { required: true, .. }) | (Block { required: true, .. }, NoBlock) => None,
        (
            Block {
                required: ar,
                min_arity: amin,
                max_arity: amax,
            },
            Block {
                required: br,
                min_arity: bmin,
                max_arity: bmax,
            },
        ) => {
            let (min_arity, max_arity) = overlap_arity((amin, amax), (bmin, bmax))?;
            Some(Block {
                required: ar || br,
                min_arity,
                max_arity,
            })
        }
    }
}

/// Block shapes of every overload of a method, in declaration order.
fn method_block_shapes(method: &crate::definition::Method) -> Vec<BlockShape> {
    method
        .method_types()
        .map(|mt| {
            if mt.is_untyped_function() {
                return BlockShape::Untyped;
            }
            match &mt.block {
                None => BlockShape::NoBlock,
                Some(block) => match &block.type_ {
                    crate::types::FunctionType::Typed(f) => BlockShape::Block {
                        required: block.required,
                        min_arity: f.min_arity(),
                        max_arity: f.max_arity(),
                    },
                    crate::types::FunctionType::Untyped(_) => BlockShape::Block {
                        required: block.required,
                        min_arity: 0,
                        max_arity: None,
                    },
                },
            }
        })
        .collect()
}

/// Whether a union method survives the block-axis combination across its
/// components (Steep `union_shape`, `builder.rb:332-390`). Folds the
/// components left to right, keeping every overload-pair combination that
/// succeeds; the method is dropped (`false`) once a component leaves no
/// surviving combination — Steep's `break nil if method_overloads.empty?`.
/// A component whose method has no overloads stays on the surviving side.
fn union_method_block_survives<'a>(
    methods: impl Iterator<Item = &'a crate::definition::Method>,
) -> bool {
    let mut survivors: Option<Vec<BlockShape>> = None;
    for method in methods {
        let shapes = method_block_shapes(method);
        if shapes.is_empty() {
            continue;
        }
        survivors = Some(match survivors {
            None => shapes,
            Some(prev) => {
                let mut next: Vec<BlockShape> = Vec::new();
                for &s in &prev {
                    for &t in &shapes {
                        if let Some(combined) = combine_block_shapes(s, t)
                            && !next.contains(&combined)
                        {
                            next.push(combined);
                        }
                    }
                }
                if next.is_empty() {
                    return false;
                }
                next
            }
        });
    }
    true
}

impl<'env> TypeChecker<'env> {
    /// Side-effecting evaluator for a single call expression: emits
    /// argument / block diagnostics, and pushes/pops the block scope
    /// around the block body walk. The caller pre-resolves the target
    /// via `ResolvedCall::resolve` so the same target lookup is shared
    /// with the read-only `infer_call_return_type` pass.
    pub(super) fn check_call<'pr>(
        &mut self,
        node: &CallNode<'pr>,
        hint: Option<Ty>,
        resolved: &ResolvedCall,
    ) {
        let target = resolved.target.as_ref();

        // Extract-mode `error` attribution: the call's own diagnostics
        // (arguments / visibility / overloading, and the block-clause
        // checks below) are emitted inside two segments of this
        // function, while every child diagnostic (receiver, argument
        // and block-body subtrees) is emitted by the `visit` calls
        // between them. A diagnostic-count delta over just the two
        // segments therefore attributes exactly the call's own
        // diagnostics to this site — no span containment, so a
        // diagnostic inside a block body never errors the outer call.
        // Gated so the default check path keeps its single-None-branch
        // cost profile (no RefCell borrows added per call).
        let track_error = self.extract.is_some() && target.is_some();
        let before = if track_error {
            self.diagnostics_len()
        } else {
            0
        };
        // Argument-channel carry is armed only across this canonical
        // collection — the one whose per-argument values the argument
        // visit below actually consumes. Auxiliary re-collections
        // (`lookup_block_type`, `infer_call_return_type`, yield/super
        // paths) stay unarmed so they can't deposit orphans or
        // overwrite the consumed values (see `extract_carry_armed`).
        let carry_prev = self.extract_carry_armed.replace(self.extract.is_some());
        self.check_call_arguments(node, target, resolved.receiver_ty);
        self.extract_carry_armed.set(carry_prev);
        let mut errored = track_error && self.diagnostics_len() > before;

        // Visit receiver and arguments BEFORE pushing block scope,
        // so block parameters don't leak into receiver/argument evaluation.
        if let Some(receiver) = node.receiver() {
            // Extract-mode hand-off: `resolved.receiver_ty` IS the inner
            // call's check-computed return type, so carry it to the
            // receiver's own record point (reached via the `visit`
            // below). Guarded where the resolved value diverges from
            // the receiver expression's checked value: safe navigation
            // strips nil (`unwrap_optional`) and an adjacent trailing
            // `#: T` overrides the natural type (`infer_receiver_type`)
            // — carrying either would record a value the check never
            // computed for the inner call. The adjacency probe is a
            // comment scan (no type lowering), so the visit path stays
            // env-query-free by construction.
            if self.extract.is_some()
                && receiver.as_call_node().is_some()
                && !node.is_safe_navigation()
                && self
                    .trailing_node_assertion_adjacent(receiver.location().end_offset())
                    .is_none()
            {
                self.extract_carry_type(
                    (
                        receiver.location().start_offset() as u32,
                        receiver.location().end_offset() as u32,
                    ),
                    resolved.receiver_ty,
                );
            }
            // Must run after `visit`: the assertion gate reads the
            // checker's live scope state, which only reflects a
            // preceding sibling statement inside the receiver's own
            // parens (`(\n a = 1\n a #: T\n).m`) once that statement has
            // actually executed via this visit.
            self.visit(&receiver);
            self.apply_receiver_assertion_gate(&receiver);
        }
        if let Some(arguments) = node.arguments() {
            self.visit_call_arguments(&arguments, target);
        }

        let site = CallSite::Call(node);
        let before_block = if track_error {
            self.diagnostics_len()
        } else {
            0
        };
        let block_scope_pushed = self.setup_block_scope(site, target, hint);
        self.check_block(site, target, hint);
        errored |= track_error && self.diagnostics_len() > before_block;
        if track_error && errored {
            self.extract_flag_call_state(
                (
                    node.location().start_offset() as u32,
                    node.location().end_offset() as u32,
                ),
                super::ExtractCallState::Error,
            );
        }

        if let Some(block) = node.block() {
            self.visit(&block);
        }
        if block_scope_pushed {
            self.ctx.pop_scope();
        }
    }

    /// Walk a call's arguments. Lambda literals are pre-dispatched into
    /// `check_node` with the parameter type as hint so their params bind
    /// typed (Steep parity); every other argument takes the plain
    /// default-walker path, which reaches `check_node` hintless.
    ///
    /// The hint is only resolved for a single-overload, non-generic
    /// target on a call with no splat / keyword arguments. With several
    /// candidates the winning slot type isn't decided until arg matching
    /// runs, and a generic slot would bind a free type variable — either
    /// would make the body walk a false-positive source. Failing to
    /// resolve a hint only costs UNTYPED params (silence), never a wrong
    /// binding.
    fn visit_call_arguments<'pr>(
        &mut self,
        arguments: &ruby_prism::ArgumentsNode<'pr>,
        target: Option<&CallTarget>,
    ) {
        let args: Vec<_> = arguments.arguments().iter().collect();
        if !args.iter().any(|arg| arg.as_lambda_node().is_some()) {
            self.visit_arguments_node(arguments);
            return;
        }
        let simple_shape = args.iter().all(|arg| {
            arg.as_splat_node().is_none()
                && arg.as_keyword_hash_node().is_none()
                && arg.as_forwarding_arguments_node().is_none()
        });
        let overload = target
            .and_then(|t| t.method_definition())
            .filter(|def| def.method_types().count() == 1)
            .and_then(|def| def.first_method_type())
            .filter(|mt| mt.type_params.is_empty());
        for (index, arg) in args.iter().enumerate() {
            if arg.as_lambda_node().is_some() {
                let hint = if simple_shape {
                    overload.and_then(|o| o.positional_param_for_call(index, args.len()))
                } else {
                    None
                };
                self.check_node(arg, hint);
            } else {
                self.visit(arg);
            }
        }
    }

    /// Separate positional and keyword arguments from a call node.
    /// Accepts both CallNode and SuperNode via CallSite; trailing
    /// type annotations (`#[T]` / `#$ T`) are CallNode-only and skip
    /// for SuperNode (no selector line to anchor on).
    pub(super) fn collect_call_arguments<'pr>(&self, node: CallSite<'_, 'pr>) -> CallArguments {
        let mut arguments = self.collect_arguments_from(node.arguments(), node.block().is_some());
        arguments.explicit_type_args = node
            .as_call_node()
            .and_then(|c| self.lookup_callsite_type_args(c));
        arguments
    }

    fn lookup_callsite_type_args<'pr>(&self, node: &CallNode<'pr>) -> Option<Vec<Ty>> {
        // Only `#[T]` / `#$ T` forms reach the type-application lookup below;
        // gate on their presence (not on any trailing comment) so a file with
        // only `#:` assertions pays no per-call `byte_offset_to_line` scan.
        if !self.comments.has_trailing_type_applications() {
            return None;
        }

        // Check end line for `#[T]` (standard rbs TypeApplication).
        let end_line = self
            .line_index
            .line(node.location().end_offset().saturating_sub(1));
        if let Some(TrailingAnnotation::TypeApplication { range, body }) =
            self.comments.trailing_annotation(&self.source, end_line)
        {
            let gap = self
                .source
                .get(node.location().end_offset()..range.0 as usize);
            if gap.is_some_and(|g| g.iter().all(u8::is_ascii_whitespace))
                && let Some(annotation) = crate::ast_builder::parse_trailing_type_application_text(
                    body,
                    range,
                    self.env.names(),
                )
            {
                return Some(self.lower_annotation_args(&annotation));
            }
            // TypeApplication found but gap check or parse failed — do not fall through to #$.
            return None;
        }

        // Check selector line for `#$ T` (Steep-style block type-arg annotation).
        // Only attaches to a call that actually carries a block; calls without a block
        // (e.g. the `each` in `arr.each.with_object({}) do ... #$ T`) are skipped so
        // that only the block-owning call (`with_object`) receives the annotation.
        let block = node.block()?;
        let selector_line = self.line_index.line(node.message_loc()?.start_offset());
        let TrailingAnnotation::DollarTypeApplication { range, body } = self
            .comments
            .trailing_annotation(&self.source, selector_line)?
        else {
            return None;
        };

        // For literal BlockNode with brace form (`{ }`), a gap check disambiguates
        // chains where multiple block-bearing calls share the selector line
        // (e.g. `foo { }.bar { } #$ T` — foo fails the gap check, bar passes).
        // The gap check applies only when the annotation sits AFTER the call's end
        // (range.0 > call_end): the "after-block" form `bar { } #$ T`.  When the
        // annotation is inside the block header (`inject(nil) {|acc, x| #$ Integer`,
        // range.0 < call_end), no gap check is needed — the block gate above suffices.
        // For `do...end` blocks `end` is on a different line (call_end > range.0 always),
        // so the condition never triggers.  For BlockArgumentNode (`&proc_var`, `&:sym`),
        // no gap check is applied — Steep-verified: those forms receive the annotation
        // just like do...end.
        if let Some(block_node) = block.as_block_node() {
            let is_brace = self
                .source
                .get(block_node.opening_loc().start_offset())
                .is_some_and(|&b| b == b'{');
            if is_brace {
                let call_end = node.location().end_offset();
                if range.0 as usize > call_end {
                    let gap = self.source.get(call_end..range.0 as usize);
                    if !gap.is_some_and(|g| g.iter().all(u8::is_ascii_whitespace)) {
                        return None;
                    }
                }
            }
        }

        let bracketed = format!("[{}]", body);
        let annotation = crate::ast_builder::parse_trailing_type_application_text(
            &bracketed,
            range,
            self.env.names(),
        )?;
        Some(self.lower_annotation_args(&annotation))
    }

    fn lower_annotation_args(&self, annotation: &TypeApplicationAnnotation) -> Vec<Ty> {
        let context = super::visitor::build_lowering_context_from_class_stack(
            self.ctx.class_stack(),
            self.env.names(),
        );
        annotation
            .type_args
            .iter()
            .map(|ast_ty| {
                self.env
                    .lower_ast_type(ast_ty, &context, &TypeParamScope::default())
            })
            .collect()
    }

    pub(super) fn explicit_type_bindings_for_overload(
        &self,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
    ) -> Option<FxHashMap<crate::type_param::TypeVarKey, Ty>> {
        let explicit = arguments.explicit_type_args.as_ref()?;
        if explicit.len() != overload.type_params.len() {
            return None;
        }
        let mut bindings = FxHashMap::default();
        for (param, &ty) in overload.type_params.iter().zip(explicit) {
            bindings.insert(param.name.clone(), ty);
        }
        Some(bindings)
    }

    fn emit_callsite_type_arg_arity_mismatch(
        &mut self,
        span: ArgSpan,
        method_name: &str,
        method_def: &crate::definition::Method,
        arguments: &CallArguments,
    ) -> bool {
        let Some(explicit) = arguments.explicit_type_args.as_ref() else {
            return false;
        };
        if method_def
            .method_types()
            .any(|overload| overload.type_params.len() == explicit.len())
        {
            return false;
        }
        let Some(overload) = method_def.method_types().next() else {
            return false;
        };
        let expected = overload.type_params.len();
        let actual = explicit.len();
        let method_type = self.display_method_type(overload);
        let kind = if actual < expected {
            DiagnosticKind::InsufficientTypeArguments {
                method_name: method_name.to_string(),
                method_type,
                expected,
                actual,
            }
        } else {
            let type_arg = explicit
                .get(expected)
                .or_else(|| explicit.last())
                .map(|&ty| self.display_type(ty))
                .unwrap_or_else(|| "untyped".to_string());
            DiagnosticKind::UnexpectedTypeArgument {
                method_name: method_name.to_string(),
                method_type,
                type_arg,
            }
        };
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: self.span_to_location(span),
            kind,
        });
        true
    }

    fn emit_callsite_type_arg_arity_mismatch_for_overload(
        &mut self,
        span: ArgSpan,
        method_name: &str,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
    ) -> bool {
        let Some(explicit) = arguments.explicit_type_args.as_ref() else {
            return false;
        };
        let expected = overload.type_params.len();
        let actual = explicit.len();
        if expected == actual {
            return false;
        }
        let method_type = self.display_method_type(overload);
        let kind = if actual < expected {
            DiagnosticKind::InsufficientTypeArguments {
                method_name: method_name.to_string(),
                method_type,
                expected,
                actual,
            }
        } else {
            let type_arg = explicit
                .get(expected)
                .or_else(|| explicit.last())
                .map(|&ty| self.display_type(ty))
                .unwrap_or_else(|| "untyped".to_string());
            DiagnosticKind::UnexpectedTypeArgument {
                method_name: method_name.to_string(),
                method_type,
                type_arg,
            }
        };
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: self.span_to_location(span),
            kind,
        });
        true
    }

    /// Extract-mode hand-off of a check-computed argument type to the
    /// argument's own record point (`visit_call_node`, reached by
    /// `check_call`'s argument visit AFTER the `collect_arguments_from`
    /// pass that computes these types). Keyed by the argument node's
    /// span, so only a bare call argument is ever consumed — the
    /// assoc/key spans stored for keywords and the splat-folded element
    /// positions never match a call node's span and simply stay
    /// `return_type: null`. No-op unless a call-node argument in
    /// extract mode.
    fn extract_carry_argument<'pr>(&self, arg: &Node<'pr>, ty: Ty) {
        if self.extract_carry_armed.get() && arg.as_call_node().is_some() {
            self.extract_carry_type(
                (
                    arg.location().start_offset() as u32,
                    arg.location().end_offset() as u32,
                ),
                ty,
            );
        }
    }

    /// Node-agnostic core of `collect_call_arguments`. Both `CallNode` and
    /// `SuperNode` expose `arguments() -> Option<ArgumentsNode>` and a block,
    /// so `super(args)` argument checking reuses this directly.
    pub(super) fn collect_arguments_from<'pr>(
        &self,
        arguments: Option<ruby_prism::ArgumentsNode<'pr>>,
        has_block: bool,
    ) -> CallArguments {
        let mut positional = vec![];
        let mut positional_spans = vec![];
        let mut keywords = vec![];
        let mut kwsplats = vec![];
        let mut splat_tail: Option<SplatTail> = None;

        let Some(arguments) = arguments else {
            return CallArguments {
                positional,
                positional_spans,
                keywords,
                kwsplats,
                has_block,
                explicit_type_args: None,
                splat_tail,
            };
        };

        // Tracks whether an unexpandable splat (Array[E] / opaque) has
        // already populated `splat_tail`. Once seen, a second unexpandable
        // splat or any normal positional arg breaks the
        // "tail = single trailing rest" invariant — fall back to an
        // untyped tail (safety net per Codex review 2026-06-07; the
        // alternative of overwriting silently lost the prior splat's
        // element type and produced false negatives).
        let mut unexpandable_seen = false;
        for arg in arguments.arguments().iter() {
            if let Some(keyword_hash) = arg.as_keyword_hash_node() {
                for elem in keyword_hash.elements().iter() {
                    if let Some(assoc) = elem.as_assoc_node() {
                        let key = assoc.key();
                        if let Some(sym) = key.as_symbol_node() {
                            let name = String::from_utf8_lossy(sym.unescaped()).to_string();
                            let value = assoc.value();
                            let value_type = self.infer_type(&value, None);
                            self.extract_carry_argument(&value, value_type);
                            let key_location = sym.value_loc().unwrap_or_else(|| sym.location());
                            keywords.push((
                                name,
                                value_type,
                                arg_span(
                                    assoc.location().start_offset(),
                                    assoc.location().end_offset(),
                                ),
                                arg_span(key_location.start_offset(), key_location.end_offset()),
                            ));
                        }
                    } else if let Some(splat) = elem.as_assoc_splat_node()
                        && let Some(value) = splat.value()
                    {
                        let value_type = self.infer_type(&value, None);
                        self.extract_carry_argument(&value, value_type);
                        kwsplats.push((
                            value_type,
                            arg_span(
                                splat.location().start_offset(),
                                splat.location().end_offset(),
                            ),
                        ));
                    }
                }
            } else if let Some(splat) = arg.as_splat_node() {
                let tail_was_none = splat_tail.is_none();
                self.absorb_splat_into_args(
                    &splat,
                    &mut positional,
                    &mut positional_spans,
                    &mut splat_tail,
                    |this, _idx, elem| this.infer_type(elem, None),
                );
                let tail_added_this_call = splat_tail.is_some() && tail_was_none;
                if tail_added_this_call && unexpandable_seen {
                    splat_tail = Some(SplatTail {
                        element_ty: Ty::UNTYPED,
                        span: arg_span(
                            splat.location().start_offset(),
                            splat.location().end_offset(),
                        ),
                    });
                }
                if splat_tail.is_some() {
                    unexpandable_seen = true;
                }
            } else if unexpandable_seen {
                splat_tail = Some(SplatTail {
                    element_ty: Ty::UNTYPED,
                    span: arg_span(arg.location().start_offset(), arg.location().end_offset()),
                });
                let arg_ty = self.infer_type(&arg, None);
                self.extract_carry_argument(&arg, arg_ty);
                positional.push(arg_ty);
                positional_spans.push(arg_span(
                    arg.location().start_offset(),
                    arg.location().end_offset(),
                ));
            } else {
                let arg_ty = self.infer_type(&arg, None);
                self.extract_carry_argument(&arg, arg_ty);
                positional.push(arg_ty);
                positional_spans.push(arg_span(
                    arg.location().start_offset(),
                    arg.location().end_offset(),
                ));
            }
        }

        CallArguments {
            positional,
            positional_spans,
            keywords,
            kwsplats,
            has_block,
            explicit_type_args: None,
            splat_tail,
        }
    }

    /// Classify a SplatNode argument and fold the result into
    /// `positional` / `splat_tail`:
    ///
    /// * `*[e1, e2, ...]` (ArrayNode literal, no inner splat) — expand each
    ///   element via `infer_elem`, pushing element types into `positional`.
    ///   Treats the literal as a Tuple even when no hint is in scope (Steep
    ///   widens to `Array[E]`; crema deliberately diverges here per todo
    ///   `high_call_site_splat_arg_distribution`).
    /// * Inner expression typed as `Type::Tuple(elems)` — push each element
    ///   type as a positional arg (Steep parity via `SendArgs#consume`).
    /// * Inner expression typed as `Array[E]` (`ClassInstance` with the
    ///   builtin Array name) — store `SplatTail { element_ty: E }`. The
    ///   downstream arity / per-arg path matches the tail against the
    ///   overload's rest slot (Steep parity via `uniform_type`).
    /// * Anything else (untyped, Union, Optional, bare `*` without an
    ///   expression) — `SplatTail { element_ty: Ty::UNTYPED }`. The
    ///   conservative bucket suppresses arity false positives.
    ///
    /// A nested splat inside an array literal (`*[1, *xs]`) bails out to
    /// the value-based classification (Tuple / Array[E] / opaque) — the
    /// element-wise expansion of nested-splat literals is the sibling
    /// todo `mid_array_literal_splat_tuple_inline`'s scope.
    fn absorb_splat_into_args<'pr, F>(
        &self,
        splat: &ruby_prism::SplatNode<'pr>,
        positional: &mut Vec<Ty>,
        positional_spans: &mut Vec<ArgSpan>,
        splat_tail: &mut Option<SplatTail>,
        mut infer_elem: F,
    ) where
        F: FnMut(&Self, usize, &Node<'pr>) -> Ty,
    {
        let splat_span = arg_span(
            splat.location().start_offset(),
            splat.location().end_offset(),
        );
        let Some(expr) = splat.expression() else {
            *splat_tail = Some(SplatTail {
                element_ty: Ty::UNTYPED,
                span: splat_span,
            });
            return;
        };

        if let Some(array) = expr.as_array_node() {
            let elements: Vec<Node<'pr>> = array.elements().iter().collect();
            let has_inner_splat = elements.iter().any(|el| el.as_splat_node().is_some());
            if !has_inner_splat {
                for (idx, el) in elements.iter().enumerate() {
                    let ty = infer_elem(self, idx, el);
                    positional.push(ty);
                    positional_spans.push(arg_span(
                        el.location().start_offset(),
                        el.location().end_offset(),
                    ));
                }
                return;
            }
        }

        let ty = self.infer_type(&expr, None);
        match self.env.types().resolve(ty) {
            Type::Tuple(elems) => {
                positional.extend(elems.iter().copied());
                positional_spans.extend((0..elems.len()).map(|_| splat_span));
            }
            Type::ClassInstance { name, args } if self.is_array_name(name) => {
                let elem = args.first().copied().unwrap_or(Ty::UNTYPED);
                *splat_tail = Some(SplatTail {
                    element_ty: elem,
                    span: splat_span,
                });
            }
            _ => {
                *splat_tail = Some(SplatTail {
                    element_ty: Ty::UNTYPED,
                    span: splat_span,
                });
            }
        }
    }

    /// Hint-aware variant of `collect_call_arguments` (ADR-0012).
    ///
    /// For each positional / keyword argument, the overload's param type
    /// at that slot becomes a bidirectional hint. Hash literals whose hint
    /// resolves to a `Type::Record` are synthesized as Records via
    /// `synthesize_hash_as_record`.
    ///
    /// This path is `&self` — extras (unknown hint keys) are computed into
    /// a throwaway Vec and dropped. Diagnostic emission for those extras
    /// is a separate `&mut self` pass (`emit_hash_literal_record_extras`)
    /// driven by `check_call_arguments`; return-type inference doesn't
    /// need to emit (the call-check path already did).
    ///
    /// `overload` is `None` when no target / no overloads are available;
    /// that reduces to hintless behavior.
    pub(super) fn collect_call_arguments_hinted<'pr>(
        &self,
        node: CallSite<'_, 'pr>,
        overload: Option<&crate::types::MethodType>,
    ) -> CallArguments {
        let mut positional: Vec<Ty> = vec![];
        let mut positional_spans = vec![];
        let mut keywords = vec![];
        let mut kwsplats = vec![];
        let mut splat_tail: Option<SplatTail> = None;
        let has_block = node.block().is_some();
        // `#[T]` / `#$ T` trailing annotations only attach to CallNode
        // (selector-line scan); super(args) has no selector line, so
        // the lookup is skipped for SuperNode.
        let explicit_type_args = node
            .as_call_node()
            .and_then(|c| self.lookup_callsite_type_args(c));

        let Some(arguments) = node.arguments() else {
            return CallArguments {
                positional,
                positional_spans,
                keywords,
                kwsplats,
                has_block,
                explicit_type_args,
                splat_tail,
            };
        };

        // `positional_count` here is the post-expansion total — needed by
        // `positional_param_for_call` to dispatch optional vs trailing
        // slots. Cheap pre-pass: ArrayNode literals reveal their length
        // statically, Tuple values reveal it through inference, Array[E]
        // / opaque splats are unknown (modeled by adding 1 so the per-arg
        // hint falls back to required / optional slots; the tail itself
        // is hinted against the rest type instead).
        let positional_count = self.estimate_post_expansion_positional_count(&arguments);

        let mut positional_index = 0usize;
        // Mirrors `collect_arguments_from` — see the safety-net comment
        // there. Once an unexpandable splat has populated `splat_tail`,
        // any subsequent splat / normal positional arg degrades the tail
        // to untyped so a second splat's element type doesn't silently
        // overwrite the first.
        let mut unexpandable_seen = false;
        for arg in arguments.arguments().iter() {
            if let Some(keyword_hash) = arg.as_keyword_hash_node() {
                for elem in keyword_hash.elements().iter() {
                    if let Some(assoc) = elem.as_assoc_node() {
                        let key = assoc.key();
                        if let Some(sym) = key.as_symbol_node() {
                            let name = String::from_utf8_lossy(sym.unescaped()).to_string();
                            let hint = overload.and_then(|o| o.keyword_param_for_call(&name));
                            let value = assoc.value();
                            let value_type = self.infer_type(&value, hint);
                            self.extract_carry_argument(&value, value_type);
                            let key_location = sym.value_loc().unwrap_or_else(|| sym.location());
                            keywords.push((
                                name,
                                value_type,
                                arg_span(
                                    assoc.location().start_offset(),
                                    assoc.location().end_offset(),
                                ),
                                arg_span(key_location.start_offset(), key_location.end_offset()),
                            ));
                        }
                    } else if let Some(splat) = elem.as_assoc_splat_node()
                        && let Some(value) = splat.value()
                    {
                        let value_type = self.infer_type(&value, None);
                        self.extract_carry_argument(&value, value_type);
                        kwsplats.push((
                            value_type,
                            arg_span(
                                splat.location().start_offset(),
                                splat.location().end_offset(),
                            ),
                        ));
                    }
                }
            } else if let Some(splat) = arg.as_splat_node() {
                let start_index = positional_index;
                let tail_was_none = splat_tail.is_none();
                self.absorb_splat_into_args(
                    &splat,
                    &mut positional,
                    &mut positional_spans,
                    &mut splat_tail,
                    |this, idx_within_splat, elem| {
                        let slot = start_index + idx_within_splat;
                        let hint = overload
                            .and_then(|o| o.positional_param_for_call(slot, positional_count));
                        this.infer_type(elem, hint)
                    },
                );
                let tail_added_this_call = splat_tail.is_some() && tail_was_none;
                if tail_added_this_call && unexpandable_seen {
                    splat_tail = Some(SplatTail {
                        element_ty: Ty::UNTYPED,
                        span: arg_span(
                            splat.location().start_offset(),
                            splat.location().end_offset(),
                        ),
                    });
                }
                if splat_tail.is_some() {
                    unexpandable_seen = true;
                }
                positional_index = positional.len();
            } else if unexpandable_seen {
                splat_tail = Some(SplatTail {
                    element_ty: Ty::UNTYPED,
                    span: arg_span(arg.location().start_offset(), arg.location().end_offset()),
                });
                let hint = overload
                    .and_then(|o| o.positional_param_for_call(positional_index, positional_count));
                let ty = self.infer_type(&arg, hint);
                self.extract_carry_argument(&arg, ty);
                positional.push(ty);
                positional_spans.push(arg_span(
                    arg.location().start_offset(),
                    arg.location().end_offset(),
                ));
                positional_index += 1;
            } else {
                let hint = overload
                    .and_then(|o| o.positional_param_for_call(positional_index, positional_count));
                let ty = self.infer_type(&arg, hint);
                self.extract_carry_argument(&arg, ty);
                positional.push(ty);
                positional_spans.push(arg_span(
                    arg.location().start_offset(),
                    arg.location().end_offset(),
                ));
                positional_index += 1;
            }
        }

        CallArguments {
            positional,
            positional_spans,
            keywords,
            kwsplats,
            has_block,
            explicit_type_args,
            splat_tail,
        }
    }

    pub(super) fn collect_forwarded_call_arguments<'pr>(
        &self,
        node: &CallNode<'pr>,
        caller_mt: &crate::types::MethodType,
    ) -> Option<CallArguments> {
        let arguments = node.arguments()?;
        let mut iter = arguments.arguments().iter();
        let first = iter.next()?;
        if first.as_forwarding_arguments_node().is_none() || iter.next().is_some() {
            return None;
        }

        let func = caller_mt.func()?;
        if !func.optional_positionals.is_empty()
            || func.rest_positional.is_some()
            || !func.trailing_positionals.is_empty()
            || !func.required_keywords.is_empty()
            || !func.optional_keywords.is_empty()
            || func.rest_keyword.is_some()
        {
            return None;
        }

        Some(CallArguments {
            positional: func.required_positionals.clone(),
            positional_spans: vec![
                arg_span(
                    first.location().start_offset(),
                    first.location().end_offset(),
                );
                func.required_positionals.len()
            ],
            keywords: vec![],
            kwsplats: vec![],
            has_block: caller_mt.block.is_some(),
            explicit_type_args: self.lookup_callsite_type_args(node),
            splat_tail: None,
        })
    }

    /// Pre-pass for the hinted collect path: estimate how many positional
    /// args remain after splat expansion. ArrayNode literals contribute
    /// their literal element count; opaque splats contribute 1 (their
    /// type is unknown). Used to drive `positional_param_for_call`'s
    /// optional/trailing dispatch.
    fn estimate_post_expansion_positional_count<'pr>(
        &self,
        arguments: &ruby_prism::ArgumentsNode<'pr>,
    ) -> usize {
        let mut count = 0usize;
        for arg in arguments.arguments().iter() {
            if arg.as_keyword_hash_node().is_some() {
                continue;
            }
            if let Some(splat) = arg.as_splat_node() {
                if let Some(expr) = splat.expression()
                    && let Some(array) = expr.as_array_node()
                {
                    let elements: Vec<Node<'pr>> = array.elements().iter().collect();
                    let has_inner_splat = elements.iter().any(|el| el.as_splat_node().is_some());
                    if !has_inner_splat {
                        count += elements.len();
                        continue;
                    }
                }
                count += 1;
            } else {
                count += 1;
            }
        }
        count
    }

    /// Walk a call's hash-literal args (positional and keyword) against the
    /// given overload and emit `UnknownRecordKey` for every key that is
    /// absent from the corresponding Record hint. Recurses into nested
    /// hash literals via `synthesize_hash_as_record`.
    ///
    /// Separate from `collect_call_arguments_hinted` because `infer_type`
    /// is `&self` and cannot push diagnostics; this is the `&mut self`
    /// counterpart that owns the diagnostic side of ADR-0012 at the call
    /// site.
    pub(super) fn emit_hash_literal_record_extras<'pr>(
        &mut self,
        node: CallSite<'_, 'pr>,
        overload: &crate::types::MethodType,
    ) {
        let Some(arguments) = node.arguments() else {
            return;
        };
        let positional_count = arguments
            .arguments()
            .iter()
            .filter(|a| a.as_keyword_hash_node().is_none())
            .count();
        let mut positional_index = 0usize;
        for arg in arguments.arguments().iter() {
            if let Some(keyword_hash) = arg.as_keyword_hash_node() {
                for elem in keyword_hash.elements().iter() {
                    if let Some(assoc) = elem.as_assoc_node() {
                        let key = assoc.key();
                        if let Some(sym) = key.as_symbol_node() {
                            let name = String::from_utf8_lossy(sym.unescaped());
                            let hint = overload.keyword_param_for_call(&name);
                            self.maybe_emit_record_extras(&assoc.value(), hint);
                        }
                    }
                }
            } else {
                let hint = overload.positional_param_for_call(positional_index, positional_count);
                self.maybe_emit_record_extras(&arg, hint);
                positional_index += 1;
            }
        }
    }

    fn maybe_emit_record_extras<'pr>(&mut self, arg: &Node<'pr>, hint: Option<Ty>) {
        let Some(hint_ty) = hint else { return };
        let Some(hash) = arg.as_hash_node() else {
            return;
        };
        let elements: Vec<_> = hash.elements().iter().collect();
        // `pick_record_hint` unwraps Optional/Union and picks the
        // committed Record candidate (or returns None for no-match in the
        // multi-candidate strict-pick case). Sharing this with the
        // `infer_type` HashNode arm keeps the two paths from drifting.
        let Some(record_hint) = self.pick_record_hint(&elements, hint_ty) else {
            return;
        };
        let mut extras = Vec::new();
        // Only emit extras when synthesis *succeeded*. If it returned
        // `None` (e.g. a `**splat` or non-literal key aborted the walk
        // mid-loop) the literal is a Hash, not a Record — surfacing the
        // partially-collected extras would flag keys from a literal we
        // are not interpreting as a Record at all.
        if self
            .synthesize_hash_as_record(&elements, record_hint, &mut extras)
            .is_none()
        {
            return;
        }
        for extra in extras {
            let position = self.offset_to_location(extra.offset);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownRecordKey {
                    key: extra.key.display(),
                    known_keys: extra.known_keys,
                },
            });
        }
    }

    /// Re-concretize a `self` receiver for method lookup. `self` receivers
    /// are the opaque `Ty::SELF_TYPE` (for identity tracking), but
    /// `method_resolver` is self-context-free and returns `None` for it, so
    /// lookup must run against the concrete `current_self_type()`. Non-self
    /// receivers pass through unchanged.
    fn concrete_self_for_lookup(&self, receiver_type: Ty) -> Ty {
        if receiver_type == Ty::SELF_TYPE {
            self.current_self_type()
        } else {
            receiver_type
        }
    }

    /// Build the call-site `Substitution`, keeping `-> self` return-type
    /// identity. For a `SELF_TYPE` receiver, the type-var / instance / class
    /// bindings are built from the concrete `current_self_type()` (so generic
    /// args and `instance`/`class` keywords resolve), but `self` itself is
    /// overwritten back to `SELF_TYPE` so a `-> self` return stays opaque and
    /// propagates. Non-self receivers go straight through.
    pub(super) fn substitution_for_call_receiver(
        &self,
        receiver_type: Ty,
        bindings: FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) -> crate::substitution::Substitution {
        if receiver_type == Ty::SELF_TYPE {
            definition_builder::substitution_for_receiver(
                self.env,
                self.current_self_type(),
                bindings,
            )
            .with_self_type(Ty::SELF_TYPE)
        } else {
            definition_builder::substitution_for_receiver(self.env, receiver_type, bindings)
        }
    }

    /// Look up the method that `node` resolves to under `receiver_type`,
    /// or `None` when the receiver is `untyped` or the method is not
    /// declared on the receiver. `receiver_type` is taken as an argument
    /// so callers that have already inferred it (`ResolvedCall::resolve`)
    /// don't re-run `infer_receiver_type` inside this function.
    pub(super) fn resolve_call_target<'pr>(
        &self,
        node: &CallNode<'pr>,
        receiver_type: Ty,
    ) -> Option<CallTarget> {
        let method_name = String::from_utf8_lossy(node.name().as_slice());
        self.resolve_call_target_at(receiver_type, &method_name)
    }

    /// CallNode-free core of `resolve_call_target`: dispatches on
    /// `(receiver_type, method_name)` alone, so synthetic call sites
    /// without a Prism `CallNode` (`x += rhs` ≡ `x = x.+(rhs)`) reuse the
    /// same target-resolution rules — alias peel, sugar widen, per-component
    /// union dispatch, block-shape survival.
    pub(super) fn resolve_call_target_at(
        &self,
        receiver_type: Ty,
        method_name: &str,
    ) -> Option<CallTarget> {
        let lookup_type = self.concrete_self_for_lookup(receiver_type);
        if lookup_type.is_untyped() {
            return None;
        }
        let method_name_n = self.env.names().intern_symbol(method_name);

        // Dispatch-boundary receiver normalization: alias expand → Optional
        // / Bool sugar widen → alias-of-union flatten, applied once with a
        // memoized result (`DefinitionBuilder::normalize_receiver_cache`).
        // Steep's per-call `raw_shape` Alias arm + sugar handling under
        // ADR-0021's no-Shape-layer constraint. Cyclic aliases bottom out
        // at `ALIAS_EXPANSION_LIMIT` and stay as `Type::Alias` so the
        // trailing arms can still classify them as Unhandled.
        let lookup_type = definition_builder::normalize_receiver(self.env, lookup_type);

        // Union receiver: per-name dispatch (ADR-0021). Resolve the called
        // name on every component; if any component lacks it, return `None`
        // so the caller reports NoMethod for the whole union. When all
        // components resolve, carry them as a `UnionMethod` so the return
        // type can union each component's return.
        //
        // This recursion lives here rather than inside `lookup_method`'s
        // `Type::Union` arm because that arm must keep returning `None`:
        // `nil_predicate_resolves_to_primitive` calls `lookup_method`
        // directly and relies on the union fail-open (pinned by
        // `test_if_nil_predicate_union_member_override_pins_known_limitation`).
        // A union needs many resolved methods anyway, which the single
        // `ResolvedMethod` return of `lookup_method` cannot express.
        if let Type::Union(members) = self.env.types().resolve(lookup_type) {
            let mut components = Vec::with_capacity(members.len());
            for &member in members {
                let resolved = method_resolver::lookup_method(self.env, member, method_name_n)?;
                components.push(super::UnionComponent {
                    method_def: resolved.method,
                    bindings: resolved.bindings,
                    receiver_type: member,
                });
            }
            // Steep `union_shape` drops a method whose components' block
            // clauses fail to combine (required block + blockless, or
            // arity-incompatible block params). Returning `None` routes the
            // call to `check_no_method`, which reports NoMethod for the
            // whole union — matching Steep for both `x.m` and `x.m { }`.
            if !union_method_block_survives(components.iter().map(|c| &c.method_def)) {
                return None;
            }
            return Some(CallTarget::UnionMethod {
                method_name: method_name.to_string(),
                components,
            });
        }

        let resolved = method_resolver::lookup_method(self.env, lookup_type, method_name_n)?;

        Some(CallTarget::Method {
            method_name: method_name.to_string(),
            receiver_class: resolved.receiver_class,
            method_def: resolved.method,
            bindings: resolved.bindings,
        })
    }

    /// Argument / block diagnostics for one resolved call. No `hint`
    /// parameter: the diagnostic logic never read one — it existed only
    /// for the removed extract-mode occurrence-type thunk (recording now
    /// happens at the walk entries; see `record_extract_call`).
    pub(super) fn check_call_arguments<'pr>(
        &mut self,
        node: &CallNode<'pr>,
        target: Option<&CallTarget>,
        receiver_type: Ty,
    ) {
        let method_name_str = String::from_utf8_lossy(node.name().as_slice()).to_string();
        // Hot path: hold the call's source span as raw byte offsets and
        // materialize a `SourceLocation` only at diagnostic-emit time.
        // The verbose-log paths below just need the byte numbers; the
        // helper chain (`check_against_method_def`,
        // `check_forwarding_call`, `check_union_call_arguments`)
        // forwards the span and lifts it lazily in its own emit sites.
        let loc_span: ArgSpan = (
            node.location().start_offset() as u32,
            node.location().end_offset() as u32,
        );

        // `receiver_type` is threaded in from `ResolvedCall::resolve`, which
        // already inferred it (`calls.rs` `ResolvedCall::resolve`). Inferring
        // it is `&self` and side-effect-free, so reusing the resolved value
        // here avoids a second `infer_receiver_type` pass per call; the single
        // value is then shared across every branch below (peek / no-target /
        // target / verbose-log).

        // Tuple/Record literal-access diagnostics. The shared specializer also
        // drives the return type in `infer_call_return_type`; here we only emit
        // the `Missing` diagnostic (UnknownTupleIndex / UnknownRecordKey). The
        // existing `Array#[]`/`Hash#[]` path below still runs, but won't
        // double-emit because literal ints and symbols validly match those
        // overloads.
        if let Some(LiteralAccessResult::Missing { diag, .. }) =
            self.element_access_specialization(receiver_type, &method_name_str, node)
        {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: self.span_to_location(loc_span),
                kind: diag,
            });
        }

        let Some(target) = target else {
            if receiver_type.is_untyped() {
                // Untyped receiver: resolution was never attempted.
                // Flag now, record at the walk entry (which knows the
                // site's expression type when the check computed one).
                self.extract_flag_call_state(loc_span, super::ExtractCallState::Untyped);
                self.verbose_log(format_args!(
                    "{}:{} skipped .{} — receiver is untyped",
                    loc_span.0, loc_span.1, method_name_str
                ));
            } else {
                let receiver_display = self.display_type(receiver_type);
                let separator = match self.env.types().resolve(receiver_type) {
                    Type::ClassSingleton { .. } => ".",
                    _ => "#",
                };
                self.verbose_log(format_args!(
                    "{}:{} not_found {}{}{}",
                    loc_span.0, loc_span.1, receiver_display, separator, method_name_str
                ));
            }
            // Extract-mode `no_method_error`: only a site whose
            // NoMethod diagnostic actually fired is recorded, so the
            // silent boundary (bot receivers, classifier-gated
            // receivers, untyped short-circuit inside
            // `check_no_method`, `NotImplementedYet`-only sites) stays
            // unrecorded exactly as before. Kind-checked rather than a
            // bare count delta, since `check_no_method` can also push
            // `NotImplementedYet` (severity `Ignore`, silent in
            // `check`) without a `NoMethod` alongside it.
            // Gated to keep the default check path borrow-free.
            let track_no_method = self.extract.is_some() && !receiver_type.is_untyped();
            let before = if track_no_method {
                self.diagnostics_len()
            } else {
                0
            };
            self.check_no_method(node);
            if track_no_method && self.diagnostics_since_include_no_method(before) {
                self.extract_flag_call_state(loc_span, super::ExtractCallState::NoMethodError);
            }
            return;
        };

        self.verbose_log_call_resolved(node, target, receiver_type);
        // The extract-mode record is NOT pushed here: its `return_type`
        // field is the check's own answer for the site, which the two
        // walk entries (`check_node`'s CallNode arm / `visit_call_node`)
        // know — this helper does not. See `record_extract_call`.

        if let CallTarget::UnionMethod {
            method_name,
            components,
        } = target
        {
            // Union dispatch collects once per component with that
            // component's trial hint; a last-write-wins deposit from
            // whichever component ran last is not a value the check
            // settled on. Disarm so union-target argument calls stay
            // `return_type: null` (better null than a plausible-but-
            // arbitrary value).
            let carry_prev = self.extract_carry_armed.replace(false);
            self.check_union_call_arguments(node, method_name, components, receiver_type, loc_span);
            self.extract_carry_armed.set(carry_prev);
            return;
        }

        let Some(method_def) = target.method_definition().cloned() else {
            return;
        };

        // Visibility gate. Must run before `check_against_method_def` so a
        // `Foo.new.secret(bad_arg)` emits only PrivateMethodCall, not an
        // additional ArgumentTypeMismatch cascade. Also runs before the
        // forwarding branch so `Foo.new.private_bar(...)` is caught (the
        // forwarding signature compat check would otherwise mask the
        // visibility violation).
        if method_def.accessibility == Visibility::Private && !is_implicit_or_self_receiver(node) {
            // Anchor to the method name token, not the whole CallNode span
            // (which for a chained call like `a.b(x).c(y)` covers from the
            // chain root through `c`'s closing paren). Mirrors
            // `check_no_method`'s `message_loc()` narrowing above.
            // For a dot-setter (`equal_loc()` is `Some`) extend through the
            // `=` so the highlighted text matches the `bar=` method_name
            // field. `equal_loc()` is also `Some` for index writes
            // (`foo[bar] = v`, method name `[]=`) but `opening_loc()`
            // (the `[`) distinguishes those — message_loc there already
            // covers the bracket expression, so extending through `=`
            // would not reconstruct `[]=` and only pulls in an unrelated
            // `=` token.
            let name_location = if let Some(message_loc) = node.message_loc() {
                let end_offset = if node.opening_loc().is_none() {
                    node.equal_loc()
                        .map_or(message_loc.end_offset(), |eq| eq.end_offset())
                } else {
                    message_loc.end_offset()
                };
                self.byte_range_to_location(message_loc.start_offset(), end_offset)
            } else {
                self.span_to_location(loc_span)
            };
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: name_location,
                kind: DiagnosticKind::PrivateMethodCall {
                    method_name: target.method_name().to_string(),
                    receiver_type: self.display_type(receiver_type),
                },
            });
            return;
        }

        // Argument forwarding (`bar(...)`) needs a sig-level compatibility
        // check against `ctx.forward_arg_type()`, not the per-arg type-check
        // loop. Steep parity (`Ruby::IncompatibleArgumentForwarding`):
        // arity / element-type mismatch fires here, then bail before the
        // ordinary positional / keyword check so we don't double-emit a
        // cascade of `InsufficientPositionalArguments` etc. on the
        // already-forwarded slot.
        if self.call_has_forwarding_args(node) {
            self.check_forwarding_call(node, &method_def, target, receiver_type, loc_span);
            return;
        }

        // ADR-0012 hint propagation. `pick_hint_overload` falls back to
        // `None` for multi-overload calls with no Record-shape match,
        // preserving `UnresolvedOverloading` from `narrow_overloads`.
        // bindings + receiver are passed through so the chosen overload
        // is substituted into the receiver's concrete type arguments
        // before its parameter types are surfaced as hints — keeping the
        // hint path aligned with the substitution state the subtype-check
        // path applies below.
        let bindings = target.bindings();
        let hint_overload =
            self.pick_hint_overload(&method_def, CallSite::Call(node), &bindings, receiver_type);
        if let Some(ov) = hint_overload.as_ref() {
            self.emit_hash_literal_record_extras(CallSite::Call(node), ov);
        }
        let arguments =
            self.collect_call_arguments_hinted(CallSite::Call(node), hint_overload.as_ref());
        let method_name = target.method_name().to_string();
        self.check_deprecated_send(
            &method_name,
            &method_def,
            &arguments,
            &bindings,
            receiver_type,
            loc_span,
        );
        self.check_against_method_def(
            loc_span,
            Some(CallSite::Call(node)),
            &method_name,
            &method_def,
            &arguments,
            &bindings,
            receiver_type,
        );
    }

    /// Emit `Ruby::DeprecatedReference` when the resolved method (or a
    /// specific matched overload of it) carries `%a{deprecated}`.
    /// Mirrors Steep's `deprecated_send?` + emit pair in
    /// `type_construction.rb:3268-3346`. Runs after the visibility and
    /// forwarding gates so `PrivateMethodCall` / forwarding diagnostics
    /// take precedence.
    ///
    /// Two-step check:
    /// 1. Member-level annotation on `Method.annotations` (shared across
    ///    all overloads) fires regardless of which overload matched.
    /// 2. Per-overload annotations fire only on the overload that
    ///    `narrow_overloads` — including its bound / block-presence /
    ///    hint filters — actually selects, so unmatched deprecated
    ///    overloads and bound-eliminated overloads stay silent.
    ///
    /// At most one diagnostic per call — the first match wins (member
    /// beats per-overload, first matched overload beats later).
    fn check_deprecated_send(
        &self,
        method_name: &str,
        method_def: &crate::definition::Method,
        arguments: &CallArguments,
        bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
        loc_span: ArgSpan,
    ) {
        if self.check_deprecated_member_only(method_name, &method_def.annotations, loc_span) {
            return;
        }

        // `narrow_overloads` (not `narrow_overloads_by_args`) is what
        // `check_against_method_def` uses to pick the actually-resolved
        // overload: it adds the bound filter and block-presence
        // preference on top of the structural arg match. Reusing it
        // here keeps the deprecated check from firing on overloads
        // that the real resolver would reject (e.g. a
        // structurally-matching but bound-violating overload marked
        // deprecated in an otherwise non-deprecated set). Hint
        // tiebreaker (`call_hint`) is not threaded here — the caller
        // does not carry it into the shared arg-check pipeline
        // either, and hint only breaks ties among already-viable
        // candidates so its absence never widens the deprecated set.
        let narrowed = self.narrow_overloads(
            method_def,
            arguments,
            bindings,
            receiver_type,
            None,
            false,
            None,
        );
        if narrowed.is_empty() {
            return;
        }
        let names = self.env.names();
        for td in &method_def.defs {
            let matched = narrowed
                .iter()
                .any(|n| std::ptr::eq(*n as *const _, &td.type_ as *const _));
            if !matched {
                continue;
            }
            if let Some(message) =
                crate::definition::method::deprecated_annotation(&td.overload_annotations, names)
            {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(loc_span),
                    kind: DiagnosticKind::DeprecatedReference {
                        subject_kind: crate::diagnostic::DeprecatedSubjectKind::Method,
                        subject_name: method_name.to_string(),
                        message,
                    },
                });
                return;
            }
        }
    }

    /// Member-level (`Method.annotations`) deprecated check. Shared
    /// between the plain call path, the forwarding-args path, and the
    /// union-receiver path — the latter two have no concrete argument
    /// types to narrow against, so per-overload matching is out of
    /// reach and only the member-level annotation gate remains. Returns
    /// `true` when a diagnostic was emitted so callers can skip the
    /// per-overload check below.
    fn check_deprecated_member_only(
        &self,
        method_name: &str,
        annotations: &[crate::ast::annotation::Annotation],
        loc_span: ArgSpan,
    ) -> bool {
        let names = self.env.names();
        let Some(message) = crate::definition::method::deprecated_annotation(annotations, names)
        else {
            return false;
        };
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: self.span_to_location(loc_span),
            kind: DiagnosticKind::DeprecatedReference {
                subject_kind: crate::diagnostic::DeprecatedSubjectKind::Method,
                subject_name: method_name.to_string(),
                message,
            },
        });
        true
    }

    /// Full super-call check for an explicit `super(args)` (`SuperNode`).
    ///
    /// Resolves the super-target method (the same `lookup_super_method` the
    /// return-type path uses), then mirrors the CallNode orchestration in
    /// `check_call`. The two paths run in parallel rather than sharing a
    /// "block phase" helper because their setup differs (receiver visit,
    /// hint source); a shared extraction makes sense once a third caller
    /// (e.g. yield with a block) joins.
    ///
    /// 1. Argument checking via the shared `check_against_method_def` —
    ///    with hint-aware overload pick + arg collection (ADR-0012) so
    ///    hash literals like `super({ name: 42 })` synthesize against the
    ///    target's Record param.
    /// 2. Recursive visit of the arguments subtree (nested diagnostics).
    /// 3. Block-body typecheck against the target's block signature, with
    ///    block scope push/pop bracketing a recursive visit of the block
    ///    body (`x + "wrong"` inside `super { |x| x + "wrong" }` reaches
    ///    `Integer#+` overload resolution this way).
    ///
    /// An unresolvable super (`lookup_super_method` is `None`) emits
    /// `Ruby::UnexpectedSuper` and bails before argument checks (Steep
    /// matches super(args) only when the target resolves; argument
    /// diagnostics on an unresolved target would be noise) — but still
    /// visits children so nested diagnostics inside the args/block aren't
    /// silently lost when the super target itself is missing.
    ///
    /// Bare `super` (`ForwardingSuperNode`) is intentionally NOT routed
    /// here: Steep emits no argument diagnostic for forwarded args
    /// (measured 2026-06-05). Its UnexpectedSuper emission for
    /// unresolvable bare super lives in `inference.rs::check_node`'s
    /// ForwardingSuperNode arm.
    pub(super) fn check_super_node<'pr>(&mut self, node: &ruby_prism::SuperNode<'pr>) {
        let Some(method_name) = self.ctx.method_name().map(|s| s.to_string()) else {
            self.visit_super_children(node);
            return;
        };
        // Same Steep `method_context.method` gate as the bare-super path;
        // see `emit_unexpected_super_if_unresolvable` for the rationale.
        let has_rbs_decl = self.ctx.method_type().is_some();
        let self_ty = self.current_self_type();
        let sym = self.env.names().intern_symbol(&method_name);
        let Some((method, bindings)) = self.env.lookup_super_method(self_ty, sym) else {
            // Mirror `emit_unexpected_super_if_unresolvable`: module methods
            // defer super resolution to the include site, so stay silent.
            if has_rbs_decl && !self.is_inside_module_definition() {
                let position = self.offset_to_location(node.location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::UnexpectedSuper { method_name },
                });
            }
            // Still descend into args / block so nested diagnostics
            // (e.g. NoMethod on an arg expression) aren't suppressed
            // along with the unresolved super.
            self.visit_super_children(node);
            return;
        };
        // Hint-aware overload pick + arg collection so Record literals fed
        // to super(args) get the same bidirectional synthesis the CallNode
        // path runs in `check_call_arguments` (ADR-0012). Falls back to
        // hint=None when no overload qualifies, preserving the prior
        // hintless behavior.
        let site = CallSite::Super(node);
        // Extract-mode `error` attribution: same two-segment
        // diagnostic-count delta as `check_call` (argument checks here,
        // block-clause checks below), excluding the child-subtree
        // visits in between.
        let track_error = self.extract.is_some();
        let super_span = (
            node.location().start_offset() as u32,
            node.location().end_offset() as u32,
        );
        let before = if track_error {
            self.diagnostics_len()
        } else {
            0
        };
        let hint_overload = self.pick_hint_overload(&method, site, &bindings, self_ty);
        if let Some(ov) = hint_overload.as_ref() {
            self.emit_hash_literal_record_extras(site, ov);
        }
        let arguments = self.collect_call_arguments_hinted(site, hint_overload.as_ref());
        // Match the prior `offset_to_location(start)` shape: a single-point
        // span anchored at the call's start. Preserves the diagnostic
        // location bytes (start_byte == end_byte) the super-arg path has
        // emitted historically.
        let start_byte = node.location().start_offset() as u32;
        let span = (start_byte, start_byte);
        self.check_against_method_def(
            span,
            Some(site),
            &method_name,
            &method,
            &arguments,
            &bindings,
            self_ty,
        );
        let mut errored = track_error && self.diagnostics_len() > before;

        // Visit args BEFORE pushing block scope so block params don't
        // leak into argument evaluation (mirrors the order in
        // `check_call`).
        if let Some(arguments_node) = node.arguments() {
            self.visit_arguments_node(&arguments_node);
        }

        // Build a synthetic CallTarget for the shared block-check path.
        // `receiver_class` is only consulted by `verbose_log_call_resolved`
        // (CallNode-only diagnostic path); for super we set it to the
        // current self class so it's well-formed without claiming the
        // super-target's defining class (which `lookup_super_method`
        // doesn't surface).
        let receiver_class = match self.env.types().resolve(self_ty) {
            Type::ClassInstance { name, .. } => *name,
            Type::ClassSingleton { name } => *name,
            // Invariant enforced by `lookup_super_method`: it returns
            // `Some` only when `self_ty` resolves to one of the two
            // arms above (see definition_builder.rs:1257). The earlier
            // `let Some(...) = lookup_super_method(...) else { ... }`
            // guarantees we don't reach here. `unreachable!` makes the
            // invariant explicit so a future widening of
            // `lookup_super_method` doesn't silent-drop the super
            // block check.
            _ => unreachable!("lookup_super_method returned Some on non-class self_ty"),
        };
        let target = CallTarget::Method {
            method_name: method_name.clone(),
            receiver_class,
            method_def: method,
            bindings: bindings.clone(),
        };
        let target_ref = Some(&target);
        let before_block = if track_error {
            self.diagnostics_len()
        } else {
            0
        };
        let block_scope_pushed = self.setup_block_scope(site, target_ref, None);
        self.check_block(site, target_ref, None);
        errored |= track_error && self.diagnostics_len() > before_block;
        if track_error && errored {
            self.extract_flag_call_state(super_span, super::ExtractCallState::Error);
        }
        if let Some(block) = node.block() {
            self.visit(&block);
        }
        if block_scope_pushed {
            self.ctx.pop_scope();
        }
    }

    fn visit_super_children<'pr>(&mut self, node: &ruby_prism::SuperNode<'pr>) {
        if let Some(arguments_node) = node.arguments() {
            self.visit_arguments_node(&arguments_node);
        }
        if let Some(block) = node.block() {
            self.visit(&block);
        }
    }

    /// Log verbose information about a resolved call target.
    ///
    /// When the method has a known source file, prefers a compact
    /// `` found `::Foo#bar` from (path) `` form over printing the full
    /// RBS signature — long signatures (e.g. `File.write`) are hard to
    /// read, and the file path is the more useful pointer.
    /// Without a source file, falls back to `found ::Foo#bar: (sig) -> ret`
    /// so tests and REPL-style use still see the type info.
    fn verbose_log_call_resolved<'pr>(
        &self,
        node: &CallNode<'pr>,
        target: &CallTarget,
        receiver_type: Ty,
    ) {
        if !self.options.verbose {
            return;
        }
        let loc = self.offset_to_location(node.location().start_offset());
        match target {
            CallTarget::Method {
                method_name,
                receiver_class,
                method_def,
                ..
            } => {
                let receiver_str = self.env.names().resolve(receiver_class);
                let separator = match self.env.types().resolve(receiver_type) {
                    Type::ClassSingleton { .. } => ".",
                    _ => "#",
                };
                let qualified = format!("{}{}{}", receiver_str, separator, method_name);
                match self.format_method_source(method_def) {
                    Some(from) => self.verbose_log(format_args!(
                        "{}:{} found `{}` from {}",
                        loc.range.start_byte, loc.range.end_byte, qualified, from
                    )),
                    None => {
                        let sig = method_def
                            .method_types()
                            .next()
                            .map(|o| self.display_method_type(o))
                            .unwrap_or_else(|| "?".to_string());
                        self.verbose_log(format_args!(
                            "{}:{} found {}: {}",
                            loc.range.start_byte, loc.range.end_byte, qualified, sig
                        ));
                    }
                }
            }
            CallTarget::UnionMethod {
                method_name,
                components,
            } => {
                let receiver_str = self.display_type(receiver_type);
                self.verbose_log(format_args!(
                    "{}:{} found `{}#{}` on all {} union components",
                    loc.range.start_byte,
                    loc.range.end_byte,
                    receiver_str,
                    method_name,
                    components.len()
                ));
            }
        }
    }

    /// Format a Method's primary source location as `(file)` for verbose
    /// logs, or `None` when no file-bearing source is available.
    fn format_method_source(&self, method_def: &crate::definition::Method) -> Option<String> {
        let source = method_def.primary_source_location(self.env.names())?;
        Some(format!("({})", source.file.display()))
    }

    /// Returns true when this overload's method-level bounds are satisfied
    /// by the argument-inferred and block-body-inferred bindings.
    ///
    /// Used internally by `narrow_overloads` plus the strict diagnostic
    /// path in `check_against_method_def`, which needs to distinguish
    /// "no overload arg-matched" from "arg-matched but bound violated" and
    /// therefore can't go through the fallback-returning narrower.
    fn overload_passes_bounds<'pr>(
        &self,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        call: Option<CallSite<'_, 'pr>>,
    ) -> bool {
        let has_any_bound = overload
            .type_params
            .iter()
            .any(|p| p.upper_bound.is_some() || p.lower_bound.is_some());
        if !has_any_bound {
            return true;
        }

        let mut local_bindings =
            self.collect_call_site_bindings(overload, arguments, receiver_bindings);
        if let Some(call) = call {
            self.augment_bindings_with_block_body(call, overload, &mut local_bindings);
        }

        let subtyper = self.subtyper();
        subtyper
            .check_type_arg_bounds(&overload.type_params, &local_bindings)
            .is_empty()
    }

    /// Narrow `candidates` by `predicate`. If the filter leaves nothing,
    /// fall back to the unfiltered input so downstream diagnostics can still
    /// pick a representative overload. This is the core "prefer-but-don't-
    /// require" idiom used by every overload filter step.
    fn narrow_with_fallback<'a, F>(
        candidates: Vec<&'a crate::types::MethodType>,
        predicate: F,
    ) -> Vec<&'a crate::types::MethodType>
    where
        F: Fn(&&'a crate::types::MethodType) -> bool,
    {
        let filtered: Vec<_> = candidates.iter().copied().filter(&predicate).collect();
        if filtered.is_empty() {
            candidates
        } else {
            filtered
        }
    }

    /// Port of Steep's `SendArgs#positional_arg` / `kwargs_node`
    /// (`type_inference/send_args.rb`): when the overload declares no
    /// keyword params (`keyword_params.empty?` — no required, optional,
    /// or rest keyword), Ruby passes a braceless keyword hash as one
    /// trailing positional Hash. Fold the collected keywords (and `**h`
    /// splats) into one positional so arity / subtype / binding checks
    /// see what the runtime sees. Per-overload: an overload that does
    /// declare keywords keeps the keyword routing, so
    /// `(Integer x) | (a: Integer)` still resolves `f(a: 1)` through the
    /// keyword overload.
    ///
    /// Literal keywords alone fold to a `Type::Record`. Once a `**h` is
    /// present the key set is not static, so the fold is a `Hash[K, V]`
    /// merging the literal keys (`Symbol` / widened values) with each
    /// splat's K / V (`absorb_kwsplat_ty`: Hash args, Record widened,
    /// untyped dropped) — Steep's `type_hash` no-hint path. `**h` alone
    /// therefore types as `h` itself.
    ///
    /// Returns `None` when nothing folds (no keywords / splats, overload
    /// takes keywords, or `(?) -> untyped`) and when a splat tail is
    /// present — `CallArguments` keeps the tail as the last positional,
    /// and the folded hash would have to land after it; that shape stays
    /// on the keyword path (residual false positive, tracked in the todo).
    pub(super) fn fold_braceless_keywords_for(
        &self,
        arguments: &CallArguments,
        overload: &crate::types::MethodType,
    ) -> Option<CallArguments> {
        if (arguments.keywords.is_empty() && arguments.kwsplats.is_empty())
            || arguments.splat_tail.is_some()
            || overload.is_untyped_function()
            || !overload.required_keywords().is_empty()
            || !overload.optional_keywords().is_empty()
            || overload.rest_keyword().is_some()
        {
            return None;
        }
        let mut fields: Vec<(crate::types::RecordKey, Ty, bool)> = Vec::new();
        for (name, ty, _, _) in &arguments.keywords {
            let key = crate::types::RecordKey::Symbol(name.clone());
            // Widen literal values (`a: 1` → Integer): the positional param
            // is typically `Hash[Symbol, Integer]`, invariant in V, and
            // Steep types the folded kwargs node with widened values too.
            let ty = self.widen_literal(*ty);
            match fields.iter_mut().find(|(k, _, _)| *k == key) {
                // Duplicate literal key: Ruby keeps the last value.
                Some(slot) => slot.1 = ty,
                None => fields.push((key, ty, true)),
            }
        }
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        let folded_ty = if arguments.kwsplats.is_empty() {
            self.env.types().intern(Type::Record { fields })
        } else {
            let mut key_members = Vec::new();
            let mut value_members = Vec::new();
            if !fields.is_empty() {
                let record = self.widen_record_to_hash(&fields);
                self.absorb_kwsplat_ty(record, &mut key_members, &mut value_members);
            }
            for (ty, _) in &arguments.kwsplats {
                self.absorb_kwsplat_ty(*ty, &mut key_members, &mut value_members);
            }
            // Every contributor was untyped (or non-Hash): `Hash[untyped,
            // untyped]`, which still occupies the positional slot.
            let key_ty = self.union_of_members(&key_members);
            let value_ty = self.union_of_members(&value_members);
            self.env.types().intern(Type::ClassInstance {
                name: self.env.names().builtins().hash,
                args: vec![key_ty, value_ty],
            })
        };
        let spans = arguments
            .keywords
            .iter()
            .map(|(_, _, span, _)| *span)
            .chain(arguments.kwsplats.iter().map(|(_, span)| *span));
        let first = spans.clone().map(|s| s.0).min()?;
        let last = spans.map(|s| s.1).max()?;
        let mut folded = arguments.clone();
        folded.positional.push(folded_ty);
        folded.positional_spans.push((first, last));
        folded.keywords.clear();
        folded.kwsplats.clear();
        Some(folded)
    }

    /// Union of an `absorb_kwsplat_ty` member pool; untyped when the pool
    /// is empty (every contributor dropped).
    fn union_of_members(&self, members: &[Ty]) -> Ty {
        match members {
            [] => Ty::UNTYPED,
            [single] => *single,
            _ => self.env.types().intern(Type::Union(members.to_vec())),
        }
    }

    /// Strict structural filtering shared by all three call-path consumers
    /// (`check_against_method_def`, `infer_return_type`, `lookup_block_type`).
    ///
    /// Applies arity, positional arg subtyping, keyword shape, and
    /// required-block presence as hard AND'd filters. Any step that
    /// empties the candidate set short-circuits to `vec![]`.
    ///
    /// The bound filter and the block-presence *preference* are soft
    /// steps applied in `narrow_overloads`, not here — keeping them out
    /// lets `check_against_method_def` reuse this helper for its strict
    /// arg-matching path without conflating structural failure with
    /// bound failure (which it needs to distinguish for
    /// `TypeArgumentBoundViolation`).
    ///
    /// `only_block_bearing` hard-filters to overloads with a `block`
    /// clause (for `lookup_block_type`, which has no meaningful answer
    /// without one).
    pub(super) fn narrow_overloads_by_args<'a>(
        &self,
        method_def: &'a crate::definition::Method,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
        only_block_bearing: bool,
    ) -> Vec<&'a crate::types::MethodType> {
        let subtyper = self.subtyper();
        let base_subst =
            self.substitution_for_call_receiver(receiver_type, receiver_bindings.clone());

        let candidates: Vec<_> = method_def.method_types().collect();

        // Block-bearing hard filter first: a method with no block-bearing
        // overload has no block body to type-check regardless of arity/args.
        let candidates: Vec<_> = if only_block_bearing {
            candidates
                .into_iter()
                .filter(|o| o.block.is_some())
                .collect()
        } else {
            candidates
        };
        if candidates.is_empty() {
            return vec![];
        }

        // Per-overload view of the arguments: keyword-less overloads see
        // the braceless keyword hash as a trailing positional Record.
        let candidates: Vec<(
            &crate::types::MethodType,
            std::borrow::Cow<'_, CallArguments>,
        )> = candidates
            .into_iter()
            .map(|o| {
                let args = match self.fold_braceless_keywords_for(arguments, o) {
                    Some(folded) => std::borrow::Cow::Owned(folded),
                    None => std::borrow::Cow::Borrowed(arguments),
                };
                (o, args)
            })
            .collect();

        // Arity — strict. With a splat tail in scope, an exact `arg_count`
        // can't be enforced (the tail's length is unknown statically), so
        // the arity filter is relaxed:
        //   * untyped tail — defer to the diagnostic path (which emits
        //     `UnexpectedPositionalArgument` for rest-less overloads,
        //     silent for rest-bearing); accept everything here so the
        //     diagnostic path can still pick a representative overload.
        //   * non-untyped tail + overload has a rest slot — accept on any
        //     prefix length. The rest slot absorbs every surplus arg, so
        //     bounding `arg_count` is wrong: `g(0, 1, 2, *xs)` against
        //     `(Integer, *Integer)` is a valid Ruby call.
        //   * non-untyped tail + no rest slot — reject (the tail has
        //     nowhere to land; matches Steep's `UnexpectedPositionalArgument`).
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(o, arguments)| {
                if o.is_untyped_function() {
                    return true;
                }
                let arg_count = arguments.positional.len();
                match &arguments.splat_tail {
                    None => o.arity_accepts(arg_count),
                    // Both untyped and non-untyped tails require a rest
                    // slot to land in. Untyped tails reject rest-less
                    // overloads here so `check_against_method_def`'s
                    // rest-less arm can emit `UnexpectedPositionalArgument`
                    // (Steep parity / user decision 2026-06-07: surface
                    // the gap rather than silently accept opaque splats).
                    Some(_) => o.rest_positional().is_some(),
                }
            })
            .collect();
        if candidates.is_empty() {
            return vec![];
        }

        // Positional arg subtyping — strict. `subtyper.check` already
        // wildcard-matches untyped actuals and `TypeVariable` on either
        // side, so no explicit skip is needed here.
        //
        // The splat tail (if any non-untyped) is matched against the
        // overload's rest type AND against any residual required slots
        // the expanded prefix didn't cover — Steep's `uniform_type`
        // intersects required and rest for the splat element, since at
        // runtime any tail element may end up in either slot.
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(o, arguments)| {
                if o.is_untyped_function() {
                    return true;
                }
                let arg_count = arguments.positional.len();
                let subst = |ty: Ty| {
                    let mut ty = base_subst.apply(ty, self.env.types());
                    if let Some(explicit) = self.explicit_type_bindings_for_overload(o, arguments) {
                        ty = crate::substitution::Substitution::from_mapping(explicit)
                            .apply(ty, self.env.types());
                    }
                    ty
                };
                for (index, &actual) in arguments.positional.iter().enumerate() {
                    let Some(expected) = o.positional_param_for_call(index, arg_count) else {
                        return false;
                    };
                    if !subtyper.check(actual, subst(expected)) {
                        return false;
                    }
                }
                if let Some(tail) = &arguments.splat_tail
                    && !tail.element_ty.is_untyped()
                {
                    let expanded_count = arguments.positional.len();
                    let required_len = o.min_arity();
                    // Residual required slots: the tail's element type
                    // must satisfy every required slot the prefix didn't
                    // fill (Steep's uniform_type intersection).
                    for slot in expanded_count..required_len {
                        if let Some(expected) = o.positional_param_for_call(slot, arg_count)
                            && !subtyper.check(tail.element_ty, subst(expected))
                        {
                            return false;
                        }
                    }
                    if let Some(rest_ty) = o.rest_positional()
                        && !subtyper.check(tail.element_ty, subst(rest_ty))
                    {
                        return false;
                    }
                }
                true
            })
            .collect();
        if candidates.is_empty() {
            return vec![];
        }

        // Keyword shape — strict. Required keywords present, each keyword
        // type-checks, unknown keywords rejected unless the overload
        // carries `**rest`.
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(o, arguments)| {
                if o.is_untyped_function() {
                    return true;
                }
                let subst = |ty: Ty| {
                    let mut ty = base_subst.apply(ty, self.env.types());
                    if let Some(explicit) = self.explicit_type_bindings_for_overload(o, arguments) {
                        ty = crate::substitution::Substitution::from_mapping(explicit)
                            .apply(ty, self.env.types());
                    }
                    ty
                };
                for (required_name, _) in o.required_keywords() {
                    if !arguments
                        .keywords
                        .iter()
                        .any(|(name, _, _, _)| name == required_name)
                    {
                        return false;
                    }
                }
                for (keyword_name, actual, _, _) in &arguments.keywords {
                    let expected = o
                        .required_keywords()
                        .iter()
                        .find(|(name, _)| name == keyword_name)
                        .map(|(_, ty)| *ty)
                        .or_else(|| {
                            o.optional_keywords()
                                .iter()
                                .find(|(name, _)| name == keyword_name)
                                .map(|(_, ty)| *ty)
                        })
                        .or(o.rest_keyword());
                    match expected {
                        Some(expected_type) => {
                            if !subtyper.check(*actual, subst(expected_type)) {
                                return false;
                            }
                        }
                        None => {
                            if o.rest_keyword().is_none() {
                                return false;
                            }
                        }
                    }
                }
                true
            })
            .collect();
        if candidates.is_empty() {
            return vec![];
        }

        // Required-block presence — strict. A call without a block cannot
        // reach an overload whose block is required.
        candidates
            .into_iter()
            .filter(|(o, _)| {
                if let Some(block) = &o.block
                    && block.required
                    && !arguments.has_block
                {
                    return false;
                }
                true
            })
            .map(|(o, _)| o)
            .collect()
    }

    /// Full overload narrowing: strict structural filtering
    /// (`narrow_overloads_by_args`) followed by the bound filter and the
    /// block-presence preference step, both with fallback semantics.
    /// Used by `infer_return_type` and `lookup_block_type` (via
    /// `select_block_overload`) — both want the final first-fit candidate
    /// without distinguishing structural vs bound vs block-shape failures.
    /// `check_against_method_def` calls `narrow_overloads_by_args` directly
    /// and inspects bounds itself because it needs to emit a
    /// `TypeArgumentBoundViolation` distinctly from generic per-arg errors.
    ///
    /// Order matters: bound runs before block preference so a
    /// bound-violating block-bearing overload does not lock the selection
    /// away from a bound-valid non-block alternative when the call passes
    /// a block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn narrow_overloads<'a, 'pr>(
        &self,
        method_def: &'a crate::definition::Method,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
        call: Option<CallSite<'_, 'pr>>,
        only_block_bearing: bool,
        call_hint: Option<Ty>,
    ) -> Vec<&'a crate::types::MethodType> {
        let candidates = self.narrow_overloads_by_args(
            method_def,
            arguments,
            receiver_bindings,
            receiver_type,
            only_block_bearing,
        );
        if candidates.is_empty() {
            return candidates;
        }

        // Bound filter (soft, with fallback): when every candidate
        // violates a bound, keep the structural set so the diagnostic
        // path can still point at a specific overload for
        // TypeArgumentBoundViolation (Phase C).
        let candidates = Self::narrow_with_fallback(candidates, |o| {
            self.overload_passes_bounds(o, arguments, receiver_bindings, call)
        });

        // Block-presence preference (soft, with fallback): tiebreaker
        // among already-viable overloads. If the call passes a block,
        // prefer block-bearing overloads; else prefer block-less /
        // optional-block. Placed AFTER bound so a bound-valid
        // block-less overload isn't shadowed by a bound-invalid
        // block-bearing one on block calls.
        let candidates = if only_block_bearing {
            candidates
        } else if arguments.has_block {
            Self::narrow_with_fallback(candidates, |o| o.block.is_some())
        } else {
            Self::narrow_with_fallback(candidates, |o| o.block.as_ref().is_none_or(|b| !b.required))
        };

        // Hint-aware tiebreaker (soft, with fallback): prefer overloads
        // whose return type is compatible with the call-site hint. Placed
        // last so structural filters (arity, arg types, bounds,
        // block-presence) run first and hint only breaks ties among
        // already-viable candidates.
        //
        // Without this, apply_hint_override emits a false-positive
        // UnsatisfiableConstraint whenever a hint-compatible overload
        // exists later in definition order (e.g. the second overload of
        // `def m: [U] (U) { () -> U } -> U | (Integer) { () -> String } -> String`
        // called as `m(1) { "s" } #: String`).
        if let Some(hint_ty) = call_hint {
            Self::narrow_with_fallback(candidates, |o| {
                self.overload_passes_hint(o, arguments, receiver_bindings, hint_ty)
            })
        } else {
            candidates
        }
    }

    /// Returns true when this overload's return type is compatible with
    /// `hint_ty` for all method-level type parameters it binds.
    ///
    /// Used as the hint-aware tiebreaker in `narrow_overloads`. Mirrors
    /// the compatibility gate in `apply_hint_override`: if seed and hint
    /// agree (one is a subtype of the other), the overload is a valid
    /// candidate; if they conflict, prefer a later overload whose return
    /// type naturally matches the hint.
    fn overload_passes_hint(
        &self,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        hint_ty: Ty,
    ) -> bool {
        if overload.type_params.is_empty() {
            return true;
        }
        let seed_bindings = self.collect_call_site_bindings(overload, arguments, receiver_bindings);
        let mut hint_bindings: FxHashMap<crate::type_param::TypeVarKey, Ty> = FxHashMap::default();
        self.unify_return_against_hint(overload.return_type(), hint_ty, &mut hint_bindings);
        if hint_bindings.is_empty() {
            return true;
        }
        let method_level: FxHashSet<crate::type_param::TypeVarKey> = overload
            .type_params
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let checker = self.subtyper();
        for (name, hint_val) in hint_bindings {
            if !method_level.contains(&name) {
                continue;
            }
            let hint_expanded = crate::definition_builder::expand_alias(self.env, hint_val);
            if hint_expanded.is_untyped() {
                continue;
            }
            if let Some(&seed_val) = seed_bindings.get(&name) {
                let compatible =
                    checker.check(seed_val, hint_val) || checker.check(hint_val, seed_val);
                if !compatible {
                    return false;
                }
            }
        }
        true
    }

    /// Validate that any method-level type parameter inferred by this call
    /// satisfies its declared upper/lower bound. Runs only after overload
    /// selection succeeds (so receiver/arg typing is sound) and only when
    /// the chosen overload actually carries bounds — skipping avoids
    /// recomputing call-site bindings for every bound-free call.
    ///
    /// Receiver-side class bindings are kept but not re-validated here:
    /// class-level bound enforcement is deferred until `#:` inline
    /// annotations are supported.
    fn check_method_level_bounds<'pr>(
        &mut self,
        site: CallSite<'_, 'pr>,
        method_name: &str,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) {
        let has_any_bound = overload
            .type_params
            .iter()
            .any(|p| p.upper_bound.is_some() || p.lower_bound.is_some());
        if !has_any_bound {
            return;
        }

        let mut local_bindings =
            self.collect_call_site_bindings(overload, arguments, receiver_bindings);
        self.augment_bindings_with_block_body(site, overload, &mut local_bindings);

        let subtyper = self.subtyper();
        let violations = subtyper.check_type_arg_bounds(&overload.type_params, &local_bindings);
        if violations.is_empty() {
            return;
        }

        let receiver_type = self.receiver_type_for_site(site);
        let container = {
            let resolved = self.env.types().resolve(receiver_type);
            match resolved {
                Type::ClassSingleton { name } => {
                    format!("{}.{}", self.env.names().resolve(name), method_name)
                }
                Type::ClassInstance { name, .. } => {
                    format!("{}#{}", self.env.names().resolve(name), method_name)
                }
                _ => method_name.to_string(),
            }
        };
        let location = self.offset_to_location(site.location().start_offset());

        for violation in violations {
            // BoundViolation carries the raw RBS-written name as a Symbol
            // (`subtyping.rs` extracts `param.name.raw` from the TypeVarKey),
            // so resolving it back to a string is just an interner lookup.
            let param_name = self.env.names().resolve(violation.param_name);
            let bound = self.display_type(violation.bound);
            let actual = self.display_type(violation.actual);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: location.clone(),
                kind: DiagnosticKind::TypeArgumentBoundViolation {
                    container_name: container.clone(),
                    param_name,
                    bound_kind: violation.bound_kind,
                    bound,
                    actual,
                },
            });
        }
    }

    /// Report NoMethod when receiver type is known but method is not found.
    fn check_no_method<'pr>(&mut self, node: &CallNode<'pr>) {
        let method_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let name_location = if let Some(message_loc) = node.message_loc() {
            // For a dot-setter (`equal_loc()` is `Some`) extend through the
            // `=` so the highlighted text matches the `bar=` method_name
            // field. `equal_loc()` is also `Some` for index writes
            // (`foo[bar] = v`, method name `[]=`) but `opening_loc()` (the
            // `[`) distinguishes those — message_loc there already covers
            // the bracket expression, so extending through `=` would not
            // reconstruct `[]=` and only pulls in an unrelated `=` token.
            let end_offset = if node.opening_loc().is_none() {
                node.equal_loc()
                    .map_or(message_loc.end_offset(), |eq| eq.end_offset())
            } else {
                message_loc.end_offset()
            };
            self.byte_range_to_location(message_loc.start_offset(), end_offset)
        } else {
            self.byte_range_to_location(
                node.location().start_offset(),
                node.location().end_offset(),
            )
        };
        // Re-concretize a `self` receiver inside `check_no_method_at`: NoMethod
        // resolution and its reported receiver type need the concrete self,
        // just like `resolve_call_target` lookup. The synthetic-call sibling
        // (`x += rhs`) hands in a raw lvar type that's never SELF_TYPE, so the
        // bot-self arm below never fires for those callers.
        let raw_receiver = self.infer_receiver_type(node);
        if raw_receiver == Ty::BOTTOM
            && let Some(receiver) = node.receiver()
            && let Some(lvar) = receiver.as_local_variable_read_node()
        {
            let name_str = String::from_utf8_lossy(lvar.name().as_slice());
            let name = self.checker_names().intern(&name_str);
            if self.ctx.is_bot_rhs_local_variable(name) {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: name_location,
                    kind: DiagnosticKind::NoMethod {
                        method_name,
                        receiver_type: self.display_type(Ty::BOTTOM),
                        missing_from: Vec::new(),
                    },
                });
                return;
            }
        }
        self.check_no_method_at(raw_receiver, &method_name, name_location);
    }

    /// CallNode-free core of `check_no_method`. Reused by the synthetic-call
    /// path (`visit_local_variable_operator_write_node`) so operator-write
    /// dispatch reports the same NoMethod diagnostics — including the
    /// union / intersection / Optional / Bool / Alias widenings — as a
    /// regular call.
    pub(super) fn check_no_method_at(
        &mut self,
        raw_receiver: Ty,
        method_name: &str,
        name_location: SourceLocation,
    ) {
        let receiver_type = self.concrete_self_for_lookup(raw_receiver);
        if receiver_type.is_untyped() {
            return;
        }

        // Bot `self` receiver: a one-Some union block folds block `self` to
        // bot (`union_blocks`), and `self` is a *value* whose type is bot — the
        // send is reachable, so Steep reports every self-send as NoMethod on
        // `self`. This is scoped to `raw_receiver == SELF_TYPE`: a bot reached
        // any other way (a diverging expression `(raise).foo`, a method
        // returning bot, a local narrowed to bot in an unreachable branch) is a
        // *control-flow* bot whose continuation Steep treats as unreachable and
        // leaves silent — crema matches by falling through to the plain return.
        if receiver_type == Ty::BOTTOM {
            if raw_receiver == Ty::SELF_TYPE {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: name_location,
                    kind: DiagnosticKind::NoMethod {
                        method_name: method_name.to_string(),
                        receiver_type: "self".to_string(),
                        missing_from: Vec::new(),
                    },
                });
            }
            return;
        }

        // Mirror `resolve_call_target_at`: dispatch-boundary receiver
        // normalization (alias expand → Optional/Bool sugar widen →
        // alias-of-union flatten) so NoMethod reporting routes through
        // `check_no_method_union` (whose gates handle partial RBS) for
        // these sugars too. Memoized via `DefinitionBuilder`.
        let receiver_type = definition_builder::normalize_receiver(self.env, receiver_type);

        // Union receiver: per-name dispatch (ADR-0021). NoMethod fires when
        // any component lacks the method, reported once for the whole union.
        if let Type::Union(members) = self.env.types().resolve(receiver_type) {
            self.check_no_method_union_at(members, method_name, receiver_type, name_location);
            return;
        }

        // Intersection receiver: dual of the union path. `lookup_method`
        // already returns `Some` whenever any member carries the method,
        // so reaching here means every member lacked it. Apply the same
        // classifier gates as the union case so partial RBS does not
        // produce false positives, then report one NoMethod for the
        // whole intersection.
        if let Type::Intersection(members) = self.env.types().resolve(receiver_type) {
            self.check_no_method_intersection_at(
                members,
                method_name,
                receiver_type,
                name_location,
            );
            return;
        }

        // Single classifier shared with the union per-member path: see
        // `classify_no_method_receiver` for the Class vs Interface gate
        // and the widening it folds in. The three arms below carry the
        // duties unique to the single-receiver path (NoMethod report on
        // pass, verbose_log on gate fail, dev-only NotImplementedYet on
        // unhandled variants).
        match self.classify_no_method_receiver(receiver_type) {
            NoMethodReceiverGate::Pass(widened) => {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: name_location,
                    kind: DiagnosticKind::NoMethod {
                        method_name: method_name.to_string(),
                        receiver_type: self.display_type(widened),
                        missing_from: Vec::new(),
                    },
                });
            }
            NoMethodReceiverGate::Unnameable => {
                self.verbose_log(format_args!(
                    "{}:{} .{} → NoMethod check skipped (receiver gate not passed for {})",
                    name_location.range.start_byte,
                    name_location.range.end_byte,
                    method_name,
                    self.display_type(receiver_type)
                ));
            }
            NoMethodReceiverGate::Unhandled => {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: name_location,
                    kind: DiagnosticKind::NotImplementedYet {
                        site: "check_no_method".into(),
                        subject_display: self.display_type(receiver_type),
                    },
                });
            }
        }
    }

    /// NoMethod for a `Type::Union` receiver. Emits once for the whole union
    /// in either dispatch-failure case that `resolve_call_target` returned
    /// `None` for: at least one component lacks the method, or every component
    /// has it but their block clauses fail to combine (Steep `union_shape`
    /// drop, via `union_method_block_survives`). Conservatively suppresses
    /// when any component cannot be named as a class or has an incomplete
    /// ancestor chain — mirroring the single-receiver gate, so partial RBS
    /// does not produce false positives.
    /// NoMethod for a `Type::Union` receiver dispatched per-component
    /// (ADR-0021). Position offset is taken as an argument so synthetic
    /// call sites (`x += rhs`) drive the same union dispatch as a regular
    /// `CallNode`.
    fn check_no_method_union_at(
        &mut self,
        members: &[Ty],
        method_name: &str,
        union_type: Ty,
        name_location: SourceLocation,
    ) {
        let method_sym = self.env.names().intern_symbol(method_name);
        // `missing_from` stays empty in the block-survives-drop sub-case
        // (every member has the method, only block compatibility fails),
        // and populates per missing member otherwise. The bool used here
        // historically (`any_missing`) is now the `!missing_from.is_empty()`
        // reading of this vec — same signal, plus the member identities
        // needed for the `missing_from` JSON key.
        let mut missing_from: Vec<String> = Vec::new();
        let mut methods: Vec<crate::definition::Method> = Vec::new();
        for &member in members {
            match self.classify_no_method_receiver(member) {
                NoMethodReceiverGate::Pass(_) => {}
                NoMethodReceiverGate::Unnameable => return,
                NoMethodReceiverGate::Unhandled => {
                    // Mirror the single-receiver `Unhandled` arm: a union
                    // containing an unsupported member (e.g.
                    // `(^() -> void) | Integer`) would otherwise be
                    // silently swallowed, hiding the same gap the
                    // dev-only diagnostic is supposed to surface. Report
                    // against the offending member's `subject_display`
                    // so aggregation (jq -r .subject_display) groups
                    // matching gaps regardless of caller site.
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: name_location,
                        kind: DiagnosticKind::NotImplementedYet {
                            site: "check_no_method_union".into(),
                            subject_display: self.display_type(member),
                        },
                    });
                    return;
                }
            }
            match method_resolver::lookup_method(self.env, member, method_sym) {
                None => missing_from.push(self.display_type(member)),
                Some(resolved) => methods.push(resolved.method),
            }
        }
        // Canonicalize by rendered string so the same union set produces
        // the same `missing_from` order regardless of the underlying
        // member vector order (source-declared order, or intern-order
        // divergence between cold fresh build and warm a-snapshot decode
        // for inferred unions) — same rationale as the `Type::Union`
        // display sort (9697352d).
        missing_from.sort();
        // Reached only when `resolve_call_target` returned `None` for the
        // union: either a component lacks the method, or every component has
        // it but their block clauses fail to combine (Steep `union_shape`
        // drop). Both surface as one NoMethod on the whole union.
        if missing_from.is_empty() && union_method_block_survives(methods.iter()) {
            return;
        }
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: name_location,
            kind: DiagnosticKind::NoMethod {
                method_name: method_name.to_string(),
                receiver_type: self.display_type(union_type),
                missing_from,
            },
        });
    }

    /// NoMethod for a `Type::Intersection` receiver. Symmetric to
    /// `check_no_method_union`: applies the same classifier gates so an
    /// incompletely-typed member suppresses the report (partial RBS
    /// safety), and an unhandled member surfaces the dev-only
    /// `NotImplementedYet`. Reaching this helper implies the
    /// `lookup_method` Intersection arm returned `None` for every
    /// member, so once all members pass the gate the whole intersection
    /// receives a single NoMethod.
    ///
    /// Unlike `check_no_method_union`, no `union_method_block_survives`
    /// check is needed: any-of dispatch means we already lost on the
    /// first member that could carry the method, with no per-member
    /// methods to combine.
    /// NoMethod for a `Type::Intersection` receiver. Reaching this helper
    /// means `lookup_method`'s Intersection arm returned `None` for every
    /// member. Same gates as `check_no_method_union_at` so partial RBS
    /// does not produce false positives.
    fn check_no_method_intersection_at(
        &mut self,
        members: &[Ty],
        method_name: &str,
        inter_type: Ty,
        name_location: SourceLocation,
    ) {
        for &member in members {
            match self.classify_no_method_receiver(member) {
                NoMethodReceiverGate::Pass(_) => {}
                NoMethodReceiverGate::Unnameable => return,
                NoMethodReceiverGate::Unhandled => {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: name_location,
                        kind: DiagnosticKind::NotImplementedYet {
                            site: "check_no_method_intersection".into(),
                            subject_display: self.display_type(member),
                        },
                    });
                    return;
                }
            }
        }
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: name_location,
            kind: DiagnosticKind::NoMethod {
                method_name: method_name.to_string(),
                receiver_type: self.display_type(inter_type),
                missing_from: Vec::new(),
            },
        });
    }

    /// Dispatch a synthetic method call — one without a Prism `CallNode`.
    ///
    /// Used by `visit_local_variable_operator_write_node` to model
    /// `x += rhs` as `x = x.<op>(rhs)` per Steep `type_construction.rb`
    /// `:op_asgn`. Runs the full single-call diagnostic battery — NoMethod,
    /// ArgumentTypeMismatch / KeywordTypeMismatch / UnresolvedOverloading
    /// via `check_against_method_def` — and returns the operator method's
    /// return type. Method-level bound diagnostics are skipped because the
    /// shared bound checker needs a `CallNode` to re-augment its bindings,
    /// matching the existing super-call (`call == None`) path.
    pub(super) fn check_synthetic_method_call(
        &mut self,
        receiver_type: Ty,
        method_name: &str,
        arg_types: Vec<Ty>,
        name_location: SourceLocation,
    ) -> Ty {
        let Some(target) = self.resolve_call_target_at(receiver_type, method_name) else {
            self.check_no_method_at(receiver_type, method_name, name_location);
            return Ty::UNTYPED;
        };

        // Synthetic operator calls (`x += rhs`) carry only a pre-built name
        // `SourceLocation`, whose byte fields are already `u32` — read them
        // directly rather than routing through `arg_span` (which takes the
        // `usize` Prism offsets the real-call paths start from).
        let name_span: ArgSpan = (name_location.range.start_byte, name_location.range.end_byte);

        // The synthetic dispatch is recorded for `crema extract`
        // occurrences — a `h[k] ||= v` really runs `[]` and `[]=`, so
        // reference consumers must see both. The speculative siblings
        // (`resolve_synthetic_*` / `infer_synthetic_method_return_type`)
        // never reach here, so no duplicate records from value-position
        // inference. Extract occurrence is recorded after the match below, with the
        // return type the check itself computes there — extract runs no
        // inference of its own. The clone is behind the extract gate so
        // the plain check path pays nothing.
        let extract_target = self.extract.as_ref().map(|_| target.clone());

        let positional_spans = vec![name_span; arg_types.len()];
        let arguments = CallArguments {
            positional: arg_types,
            positional_spans,
            keywords: vec![],
            kwsplats: vec![],
            has_block: false,
            explicit_type_args: None,
            splat_tail: None,
        };
        let loc_span = name_span;
        // Extract-mode `error` attribution: the match below emits only
        // this synthetic call's own diagnostics (no child subtree is
        // visited), so a single delta suffices. Gated to keep the
        // default check path borrow-free.
        let before = if extract_target.is_some() {
            self.diagnostics_len()
        } else {
            0
        };
        let ret = 'ret: {
            match target {
                CallTarget::Method {
                    method_def,
                    bindings,
                    ..
                } => {
                    // Visibility gate. Operator-write's receiver is always
                    // the lvar — an explicit receiver — so a private operator
                    // method is always illegal. Runs before arg checks so a
                    // private call surfaces as `PrivateMethodCall` only and
                    // does not cascade into `ArgumentTypeMismatch`.
                    if method_def.accessibility == Visibility::Private {
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: name_location,
                            kind: DiagnosticKind::PrivateMethodCall {
                                method_name: method_name.to_string(),
                                receiver_type: self.display_type(receiver_type),
                            },
                        });
                        break 'ret Ty::UNTYPED;
                    }
                    // Arg-type / overload-narrow / UnresolvedOverloading
                    // diagnostics. `call: None` skips bound checks — operator
                    // methods rarely carry type-level bounds and a synthetic
                    // assignment-operator desugaring has no AST node to bind
                    // a block body off, so the gap is acceptable for v1.
                    self.check_against_method_def(
                        loc_span,
                        None,
                        method_name,
                        &method_def,
                        &arguments,
                        &bindings,
                        receiver_type,
                    );
                    self.infer_return_type(
                        &method_def,
                        &bindings,
                        &arguments,
                        receiver_type,
                        None::<&CallNode<'_>>,
                        None,
                    )
                }
                CallTarget::UnionMethod { components, .. } => {
                    // Visibility gate, mirroring the single arm above and the
                    // regular union call path (`check_union_call_arguments`).
                    // OR rule per Steep `union_shape`: the union method is
                    // private if any component's is. Operator-write's receiver
                    // is always explicit (the lvar), so the gate fires
                    // unconditionally on any private component.
                    let any_private = components
                        .iter()
                        .any(|c| c.method_def.accessibility == Visibility::Private);
                    if any_private {
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: self.span_to_location(loc_span),
                            kind: DiagnosticKind::PrivateMethodCall {
                                method_name: method_name.to_string(),
                                receiver_type: self.display_type(receiver_type),
                            },
                        });
                        break 'ret Ty::UNTYPED;
                    }
                    // Union arg-check: any component whose overloads all fail
                    // structural arg-match surfaces one UnresolvedOverloading
                    // (Steep `MethodType.union` slice for arg shape lives
                    // elsewhere; this mirrors `check_union_call_arguments`'s
                    // per-component all-accept gate without the node-bound
                    // branch).
                    let all_accept = components.iter().all(|c| {
                        !self
                            .narrow_overloads_by_args(
                                &c.method_def,
                                &arguments,
                                &c.bindings,
                                c.receiver_type,
                                false,
                            )
                            .is_empty()
                    });
                    if !all_accept {
                        let call_arguments = self.display_call_arguments(&arguments);
                        let method_types = self.display_union_method_overloads(&components);
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: self.span_to_location(loc_span),
                            kind: DiagnosticKind::UnresolvedOverloading {
                                method_name: method_name.to_string(),
                                receiver_type: self.display_type(receiver_type),
                                call_arguments,
                                method_types,
                            },
                        });
                    }
                    // Per-component return types, unioned (ADR-0021). Mirrors
                    // the Union arm of `infer_call_return_type_core`.
                    let returns: Vec<Ty> = components
                        .iter()
                        .map(|c| {
                            self.infer_return_type(
                                &c.method_def,
                                &c.bindings,
                                &arguments,
                                c.receiver_type,
                                None::<&CallNode<'_>>,
                                None,
                            )
                        })
                        .collect();
                    crate::types::union_of_many(&returns, self.env.types())
                }
            }
        };
        if let Some(t) = extract_target {
            if self.diagnostics_len() > before {
                self.extract_flag_call_state(name_span, super::ExtractCallState::Error);
            }
            self.record_extract_call(
                name_span.0 as usize,
                name_span.1 as usize,
                &t,
                receiver_type,
                Some(ret),
            );
        }
        ret
    }

    /// Read-only sibling of `check_synthetic_method_call`: resolves the
    /// same `(receiver_type, method_name, arg_types)` target and returns
    /// its return type without emitting diagnostics, so it can be called
    /// from `&self` contexts (`infer_type`) that must stay side-effect
    /// free. Skips `check_against_method_def`'s arg-shape diagnostics —
    /// callers that need those must go through `check_synthetic_method_call`
    /// instead. The private-method gate is *not* skipped: it still returns
    /// `Ty::UNTYPED` on a private target, mirroring `check_synthetic_method_
    /// call`'s early return (crema-review spec-consistency finding
    /// 2026-07-18 — without this, a private `[]` would leak its narrowed
    /// return type through `infer_type` while the live `check_node` path
    /// stays `Ty::UNTYPED`, breaking the two paths' parity). The
    /// synthesized spans are never read (no diagnostic is built on this
    /// path).
    pub(super) fn infer_synthetic_method_return_type(
        &self,
        receiver_type: Ty,
        method_name: &str,
        arg_types: Vec<Ty>,
    ) -> Ty {
        let Some(target) = self.resolve_call_target_at(receiver_type, method_name) else {
            return Ty::UNTYPED;
        };
        let positional_spans = vec![arg_span(0, 0); arg_types.len()];
        let arguments = CallArguments {
            positional: arg_types,
            positional_spans,
            keywords: vec![],
            kwsplats: vec![],
            has_block: false,
            explicit_type_args: None,
            splat_tail: None,
        };
        match target {
            CallTarget::Method {
                method_def,
                bindings,
                ..
            } => {
                if method_def.accessibility == Visibility::Private {
                    return Ty::UNTYPED;
                }
                self.infer_return_type(
                    &method_def,
                    &bindings,
                    &arguments,
                    receiver_type,
                    None::<&CallNode<'_>>,
                    None,
                )
            }
            CallTarget::UnionMethod { components, .. } => {
                if components
                    .iter()
                    .any(|c| c.method_def.accessibility == Visibility::Private)
                {
                    return Ty::UNTYPED;
                }
                let returns: Vec<Ty> = components
                    .iter()
                    .map(|c| {
                        self.infer_return_type(
                            &c.method_def,
                            &c.bindings,
                            &arguments,
                            c.receiver_type,
                            None::<&CallNode<'_>>,
                            None,
                        )
                    })
                    .collect();
                crate::types::union_of_many(&returns, self.env.types())
            }
        }
    }

    pub(super) fn check_synthetic_method_call_with_single_arg_node<'pr>(
        &mut self,
        receiver_type: Ty,
        method_name: &str,
        fallback_arg_type: Ty,
        arg_node: &Node<'pr>,
        name_location: SourceLocation,
    ) -> Ty {
        let Some(target) = self.resolve_call_target_at(receiver_type, method_name) else {
            self.check_no_method_at(receiver_type, method_name, name_location);
            return Ty::UNTYPED;
        };

        let location = name_location;
        // Extract occurrence for the union arm is recorded after the
        // match below, with the return type the check computes there —
        // extract runs no inference of its own. The Method arm
        // delegates to `check_synthetic_method_call`, which records the
        // site itself. Clone behind the extract gate: plain check pays
        // nothing.
        let extract_target = match &target {
            CallTarget::UnionMethod { .. } => self.extract.as_ref().map(|_| target.clone()),
            _ => None,
        };
        let record_span = (location.range.start_byte, location.range.end_byte);
        // Extract-mode `error` attribution for the union arm (the
        // Method arm delegates to `check_synthetic_method_call`, which
        // flags and records itself under the same span). Gated to keep
        // the default check path borrow-free.
        let before = if extract_target.is_some() {
            self.diagnostics_len()
        } else {
            0
        };
        let ret = 'ret: {
            match target {
                CallTarget::Method { .. } => self.check_synthetic_method_call(
                    receiver_type,
                    method_name,
                    vec![fallback_arg_type],
                    location,
                ),
                CallTarget::UnionMethod { components, .. } => {
                    let any_private = components
                        .iter()
                        .any(|c| c.method_def.accessibility == Visibility::Private);
                    if any_private {
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location,
                            kind: DiagnosticKind::PrivateMethodCall {
                                method_name: method_name.to_string(),
                                receiver_type: self.display_type(receiver_type),
                            },
                        });
                        break 'ret Ty::UNTYPED;
                    }

                    let mut component_arguments = Vec::with_capacity(components.len());
                    let mut all_accept = true;
                    for c in &components {
                        let (accepts, arguments) = self.synthetic_single_arg_component_arguments(
                            &c.method_def,
                            &c.bindings,
                            c.receiver_type,
                            arg_node,
                            fallback_arg_type,
                        );
                        component_arguments.push(arguments);
                        all_accept &= accepts;
                    }
                    if !all_accept {
                        // Synthetic single-arg path: every per-component args
                        // shares the same call-site arity / keyword shape, so
                        // the first component's `CallArguments` is a faithful
                        // representative for display. `component_arguments` is
                        // non-empty whenever this branch fires (mirrors the
                        // non-empty `components` invariant of `UnionMethod`).
                        let representative = component_arguments
                            .first()
                            .expect("UnionMethod always has at least one component");
                        let call_arguments = self.display_call_arguments(representative);
                        let method_types = self.display_union_method_overloads(&components);
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location,
                            kind: DiagnosticKind::UnresolvedOverloading {
                                method_name: method_name.to_string(),
                                receiver_type: self.display_type(receiver_type),
                                call_arguments,
                                method_types,
                            },
                        });
                    }

                    self.synthetic_single_arg_union_return_type(&components, &component_arguments)
                }
            }
        };
        if let Some(t) = extract_target {
            if self.diagnostics_len() > before {
                self.extract_flag_call_state(record_span, super::ExtractCallState::Error);
            }
            self.record_extract_call(
                record_span.0 as usize,
                record_span.1 as usize,
                &t,
                receiver_type,
                Some(ret),
            );
        }
        ret
    }

    /// Union-receiver return for the synthetic single-arg dispatch:
    /// each component's return under its own hinted arguments, joined
    /// (ADR-0021). Shared with the extract-mode occurrence type so the
    /// recorded type is the one the checker actually yields.
    pub(super) fn synthetic_single_arg_union_return_type(
        &self,
        components: &[super::UnionComponent],
        component_arguments: &[CallArguments],
    ) -> Ty {
        let returns: Vec<Ty> = components
            .iter()
            .zip(component_arguments.iter())
            .map(|(c, arguments)| {
                self.infer_return_type(
                    &c.method_def,
                    &c.bindings,
                    arguments,
                    c.receiver_type,
                    None::<&CallNode<'_>>,
                    None,
                )
            })
            .collect();
        crate::types::union_of_many(&returns, self.env.types())
    }

    pub(super) fn synthetic_single_arg_component_arguments<'pr>(
        &self,
        method_def: &crate::definition::Method,
        bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
        arg_node: &Node<'pr>,
        fallback_arg_type: Ty,
    ) -> (bool, CallArguments) {
        for overload in method_def.method_types() {
            let hint = overload.positional_param_for_call(0, 1);
            let arguments = CallArguments {
                positional: vec![self.infer_type(arg_node, hint)],
                positional_spans: vec![arg_span(
                    arg_node.location().start_offset(),
                    arg_node.location().end_offset(),
                )],
                keywords: vec![],
                kwsplats: vec![],
                has_block: false,
                explicit_type_args: None,
                splat_tail: None,
            };
            if !self
                .narrow_overloads_by_args(method_def, &arguments, bindings, receiver_type, false)
                .is_empty()
            {
                return (true, arguments);
            }
        }

        let fallback = CallArguments {
            positional: vec![fallback_arg_type],
            positional_spans: vec![arg_span(
                arg_node.location().start_offset(),
                arg_node.location().end_offset(),
            )],
            keywords: vec![],
            kwsplats: vec![],
            has_block: false,
            explicit_type_args: None,
            splat_tail: None,
        };
        let accepts = !self
            .narrow_overloads_by_args(method_def, &fallback, bindings, receiver_type, false)
            .is_empty();
        (accepts, fallback)
    }

    /// Classify a receiver for NoMethod reporting. Widens `Literal`/`Tuple`/
    /// `Record`/`Nil` to a `ClassInstance` (the single-receiver path used to
    /// do this inline) and applies the per-kind gate: `ClassInstance`/
    /// `ClassSingleton` need a complete ancestor chain, `Interface` needs
    /// registration in `interface_decls()`. Centralizing both gates here
    /// keeps the single-receiver and `Type::Union` per-member paths in sync
    /// so a future receiver kind only adds one match arm.
    ///
    /// The single-receiver caller distinguishes `Unnameable` (gate fail,
    /// silent skip via `verbose_log`) from `Unhandled` (no NoMethod arm yet,
    /// surfaced as dev-only `Crema::NotImplementedYet`). The union caller
    /// treats both as "suppress the whole union" since partial RBS must not
    /// produce false positives.
    fn classify_no_method_receiver(&self, ty: Ty) -> NoMethodReceiverGate {
        // Void receiver: Steep emits NoMethod with display "void" for
        // any selector (its shape builder has no Void arm, so the send
        // path drops into the NoMethod branch). Match that — reuse the
        // existing NoMethod diagnostic, no Object/untyped widen, no new
        // diagnostic kind. `display_type(Ty::VOID)` already renders
        // "void", so Pass(ty) carries the right receiver display.
        if matches!(self.env.types().resolve(ty), Type::Void) {
            return NoMethodReceiverGate::Pass(ty);
        }
        let widened = match self.env.types().resolve(ty) {
            Type::Literal(lit) => self
                .env
                .class_instance_type(*lit.class_typename(self.env.names().builtins())),
            Type::Tuple(members) => self.widen_tuple_to_array(members),
            Type::Record { fields } => self.widen_record_to_hash(fields),
            Type::Nil => self
                .env
                .class_instance_type(self.env.names().builtins().nil_class),
            Type::Proc { .. } => {
                // Steep `proc_shape` builds its Shape with the proc type
                // itself as `Shape.type` while merging in `::Proc` methods.
                // `type_construction.rb` then reports NoMethod against
                // `interface&.type || receiver_type`, so the diagnostic
                // displays the proc signature (`^() -> bool`), not the
                // widened class. Mirror that: gate completeness against
                // `::Proc` (it needs a real ancestor chain for the dispatch
                // to land), but Pass the original `ty` so `display_type`
                // emits the signature.
                let widened = self
                    .env
                    .class_instance_type(self.env.names().builtins().proc);
                return match self.env.types().resolve(widened) {
                    Type::ClassInstance { name, .. } => {
                        if self.env.has_complete_ancestor_chain(name) {
                            NoMethodReceiverGate::Pass(ty)
                        } else {
                            NoMethodReceiverGate::Unnameable
                        }
                    }
                    _ => NoMethodReceiverGate::Unhandled,
                };
            }
            Type::ClassInstance { .. } | Type::ClassSingleton { .. } | Type::Interface { .. } => ty,
            Type::Alias { .. } => {
                // Steep `raw_shape` expands `Name::Alias` and recurses for
                // method dispatch. Reclassify on the expansion so the gate
                // (complete ancestor chain / interface registration) applies
                // to the underlying class. A cyclic alias (`type a = b;
                // type b = a`) stays as alias after `ALIAS_EXPANSION_LIMIT`
                // hops — keep it Unhandled rather than recursing forever.
                let expanded = definition_builder::expand_alias(self.env, ty);
                if matches!(self.env.types().resolve(expanded), Type::Alias { .. }) {
                    return NoMethodReceiverGate::Unhandled;
                }
                return self.classify_no_method_receiver(expanded);
            }
            // Untyped is the call-dispatch escape hatch: Steep silently
            // accepts every send on `untyped`. The single-receiver path
            // already early-returns at `check_no_method` entry
            // (`receiver_type.is_untyped()`); reaching here implies a
            // per-member iteration in `check_no_method_union` /
            // `_intersection`, where `Unnameable` suppresses the whole
            // composite — matching the single-path silence.
            Type::Untyped => return NoMethodReceiverGate::Unnameable,
            // `top` is the universal supertype with no declared methods.
            // Steep reports NoMethod with display "top" (measured
            // 2026-06-06). Pass `ty` straight through — `display_type`
            // already renders "top", and `lookup_method`'s `_ => None`
            // arm guarantees the union any_missing path fires.
            Type::Top => return NoMethodReceiverGate::Pass(ty),
            // Free / bounded TypeVariable both need bound resolution to
            // match Steep: Steep looks up `config.upper_bound(name)` and
            // builds the shape from the bound (`::Numeric` for
            // `[S < ::Numeric]`, untyped for unbounded). crema does not
            // plumb the bound into `Type::TypeVariable { raw, scope }`
            // yet, so we cannot distinguish bounded vs free here. Choose
            // silence (Unnameable) over false positives: dropping a
            // legitimate NoMethod on free `[S]` is less harmful than
            // wrongly flagging `[S < ::Numeric] (S x); x + 1` as
            // NoMethod. Lifting this to spec-correct NoMethod requires
            // a separate todo that ports the bound widen.
            Type::TypeVariable { .. } => return NoMethodReceiverGate::Unnameable,
            _ => return NoMethodReceiverGate::Unhandled,
        };
        match self.env.types().resolve(widened) {
            Type::ClassInstance { name, .. } | Type::ClassSingleton { name, .. } => {
                if self.env.has_complete_ancestor_chain(name) {
                    NoMethodReceiverGate::Pass(widened)
                } else {
                    NoMethodReceiverGate::Unnameable
                }
            }
            Type::Interface { name, .. } => {
                if self.env.is_declared_interface(name) {
                    NoMethodReceiverGate::Pass(widened)
                } else {
                    NoMethodReceiverGate::Unnameable
                }
            }
            _ => NoMethodReceiverGate::Unhandled,
        }
    }

    /// Argument and visibility checking for a `Type::Union` receiver
    /// (ADR-0021 per-name dispatch). Decomposes `MethodType.union` rather
    /// than synthesizing it: argument acceptance against the union is
    /// logically equivalent to every component accepting the args
    /// independently — `arg <: (A & B)` iff `arg <: A` and `arg <: B`
    /// (the sup-side Intersection rule in `subtyping.rs`), and
    /// `∃i∃j (arg<:A_i ∧ arg<:B_j)` iff `(∃i arg<:A_i) ∧ (∃j arg<:B_j)`.
    /// So each component is run through `narrow_overloads_by_args` and one
    /// union-level `UnresolvedOverloading` is emitted if any component has no
    /// matching overload. Visibility follows Steep's `union_shape` OR rule:
    /// the union method is private if any component's is.
    fn check_union_call_arguments<'pr>(
        &mut self,
        node: &CallNode<'pr>,
        method_name: &str,
        components: &[super::UnionComponent],
        receiver_type: Ty,
        loc_span: ArgSpan,
    ) {
        // Visibility gate before arg checking, mirroring the single-receiver
        // order so a private call emits only PrivateMethodCall, not an
        // additional UnresolvedOverloading cascade.
        let any_private = components
            .iter()
            .any(|c| c.method_def.accessibility == Visibility::Private);
        if any_private && !is_implicit_or_self_receiver(node) {
            // Same method-name narrowing as the single-receiver visibility
            // gate above (`check_call_arguments`). For a dot-setter
            // (`equal_loc()` is `Some`) extend through the `=` so the
            // highlighted text matches the `bar=` method_name field.
            // `equal_loc()` is also `Some` for index writes (`foo[bar] = v`,
            // method name `[]=`) but `opening_loc()` (the `[`) distinguishes
            // those — message_loc there already covers the bracket
            // expression, so extending through `=` would not reconstruct
            // `[]=` and only pulls in an unrelated `=` token.
            let name_location = if let Some(message_loc) = node.message_loc() {
                let end_offset = if node.opening_loc().is_none() {
                    node.equal_loc()
                        .map_or(message_loc.end_offset(), |eq| eq.end_offset())
                } else {
                    message_loc.end_offset()
                };
                self.byte_range_to_location(message_loc.start_offset(), end_offset)
            } else {
                self.span_to_location(loc_span)
            };
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: name_location,
                kind: DiagnosticKind::PrivateMethodCall {
                    method_name: method_name.to_string(),
                    receiver_type: self.display_type(receiver_type),
                },
            });
            return;
        }

        // Member-level deprecated check: fire once if any component's
        // resolved method carries `%a{deprecated}` on its whole `def`.
        // Per-overload narrowing across a union receiver is deferred
        // to the same follow-up that lets `narrow_overloads` operate
        // on a union — for now the member-level gate matches Steep
        // parity for the common case.
        for component in components {
            if self.check_deprecated_member_only(
                method_name,
                &component.method_def.annotations,
                loc_span,
            ) {
                break;
            }
        }

        // Argument forwarding (`bar(...)`) over a union receiver: per-component
        // `forwarding_compat` mirrors the single-receiver branch
        // (`check_forwarding_call`), wrapped in a component-level ∀ —
        // `caller <: meet(component_i.callee_sig)` is equivalent to "every
        // component is fwd-compatible with the caller". One diagnostic per
        // call site even when multiple components fail.
        if self.call_has_forwarding_args(node) {
            let Some(caller_mt) = self.ctx.forward_arg_type() else {
                return;
            };
            let mut first_failure: Option<(ForwardingMismatchKind, MethodType)> = None;
            for component in components {
                if let Some(failure) = self.try_forwarding_compat_method_def(
                    &component.method_def,
                    &caller_mt,
                    component.receiver_type,
                    component.bindings.clone(),
                ) && first_failure.is_none()
                {
                    first_failure = Some(failure);
                }
            }
            if let Some((mismatch_kind, callee_mt)) = first_failure {
                let callee_signature = self.display_method_type(&callee_mt);
                let caller_signature = self.display_method_type(&caller_mt);
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(loc_span),
                    kind: DiagnosticKind::IncompatibleArgumentForwarding {
                        method_name: method_name.to_string(),
                        caller_signature,
                        callee_signature,
                        mismatch_kind,
                    },
                });
            }
            return;
        }

        // Hintless args: hint-driven overload selection across the whole union
        // is `MethodType.union` territory, deferred. A component accepts when it
        // has an overload that both structurally arg-matches
        // (`narrow_overloads_by_args`) and satisfies its method-level bounds
        // (`overload_passes_bounds`) — the same two-stage gate the single-receiver
        // path applies in `check_against_method_def`. Without the bound stage a
        // structurally-matching but bound-violating generic overload would slip
        // through as a false success.
        for c in components {
            let hint_overload = self.pick_hint_overload(
                &c.method_def,
                CallSite::Call(node),
                &c.bindings,
                c.receiver_type,
            );
            let arguments =
                self.collect_call_arguments_hinted(CallSite::Call(node), hint_overload.as_ref());
            if self.emit_callsite_type_arg_arity_mismatch(
                loc_span,
                method_name,
                &c.method_def,
                &arguments,
            ) {
                return;
            }
        }
        // Retain the first component's hint-driven arguments while
        // walking the accept-check loop so the failure branch below can
        // reuse them for display without re-running `pick_hint_overload`
        // + `collect_call_arguments_hinted` (one AST walk + a hint
        // lookup avoided on the failure path).
        let mut representative_args: Option<CallArguments> = None;
        let all_accept = components.iter().all(|c| {
            let hint_overload = self.pick_hint_overload(
                &c.method_def,
                CallSite::Call(node),
                &c.bindings,
                c.receiver_type,
            );
            let arguments =
                self.collect_call_arguments_hinted(CallSite::Call(node), hint_overload.as_ref());
            let accepts = self
                .narrow_overloads_by_args(
                    &c.method_def,
                    &arguments,
                    &c.bindings,
                    c.receiver_type,
                    false,
                )
                .iter()
                .any(|overload| {
                    self.overload_passes_bounds(
                        overload,
                        &arguments,
                        &c.bindings,
                        Some(CallSite::Call(node)),
                    )
                });
            if representative_args.is_none() {
                representative_args = Some(arguments);
            }
            accepts
        });
        if !all_accept {
            // The first component's hint-driven arguments are a faithful
            // representative — hint nuances differ per component but the
            // call-site arity / keyword shape doesn't.
            let call_arguments = representative_args
                .as_ref()
                .map(|a| self.display_call_arguments(a))
                .unwrap_or_else(|| "()".to_string());
            let method_types = self.display_union_method_overloads(components);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: self.span_to_location(loc_span),
                kind: DiagnosticKind::UnresolvedOverloading {
                    method_name: method_name.to_string(),
                    receiver_type: self.display_type(receiver_type),
                    call_arguments,
                    method_types,
                },
            });
        }
    }

    /// Core overload matching and diagnostic reporting logic.
    /// Used by both regular method calls and `ClassName.new`.
    /// `call` carries the source `CallSite` so the bound-violation
    /// sub-checks (`overload_passes_bounds` / `check_method_level_bounds`)
    /// can augment bindings from a block body and re-infer the receiver
    /// display. `None` is reserved for synthetic call surfaces (no AST
    /// node available — e.g. the assignment-operator desugaring that
    /// re-issues `a[i] = a[i] + 1` as a method call without a CallNode of
    /// its own); both CallNode and SuperNode paths pass
    /// `Some(CallSite::...)` so super(args) reaches the bound checks too.
    #[allow(clippy::too_many_arguments)]
    fn check_against_method_def<'pr>(
        &mut self,
        // Raw call-site span; materialized lazily at each diagnostic-emit
        // site so successful calls (the hot path) pay no char-offset scan.
        loc_span: ArgSpan,
        call: Option<CallSite<'_, 'pr>>,
        method_name: &str,
        method_def: &crate::definition::Method,
        arguments: &CallArguments,
        bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
    ) {
        let subtyper = self.subtyper();
        let types_tbl = self.env.types();
        if self.emit_callsite_type_arg_arity_mismatch(loc_span, method_name, method_def, arguments)
        {
            return;
        }
        let check_bindings = method_def
            .method_types()
            .next()
            .and_then(|overload| self.explicit_type_bindings_for_overload(overload, arguments))
            .map(|explicit| {
                let mut bindings = bindings.clone();
                for (name, ty) in explicit {
                    bindings.insert(name, ty);
                }
                bindings
            })
            .unwrap_or_else(|| bindings.clone());
        // Helper closure: substitute type-variable expectations and `self`
        // against call-site bindings before subtype-checking. Still needed
        // for the per-arg / per-keyword diagnostic fallback below, even
        // though the strict arg-matching itself is delegated to
        // `narrow_overloads_by_args`.
        let substitution = self.substitution_for_call_receiver(receiver_type, check_bindings);
        let subst = |ty: Ty| substitution.apply(ty, types_tbl);

        // Strict arg matching now shared with inference / block selection.
        // `narrow_overloads_by_args` applies exactly the structural AND
        // this path used inline (arity, positional, keyword, required
        // block) and returns `vec![]` when nothing matches — exactly the
        // "no overload matched" signal the diagnostic path needs.
        let arg_matching =
            self.narrow_overloads_by_args(method_def, arguments, bindings, receiver_type, false);

        if !arg_matching.is_empty() {
            if self.emit_callsite_type_arg_arity_mismatch_for_overload(
                loc_span,
                method_name,
                arg_matching[0],
                arguments,
            ) {
                return;
            }
            // Prefer an overload that also satisfies its method-level bounds.
            // When multiple overloads match on arg types but disagree on bounds
            // (e.g. `[T < String] (T) -> _ | [T < Integer] (T) -> _`), the
            // arg-only `find` would commit to the first and then falsely flag
            // the caller. Filtering by bound lets the later overload win.
            let any_bound_passing = arg_matching
                .iter()
                .any(|overload| self.overload_passes_bounds(overload, arguments, bindings, call));
            if any_bound_passing {
                return;
            }
            // Every arg-matching overload violated a bound: emit one
            // diagnostic on the first arg-matching overload so the user
            // sees a concrete param/bound pair rather than a generic
            // UnresolvedOverloading. Both CallNode and SuperNode reach
            // this — only synthetic call surfaces (`call == None`) skip.
            if let Some(site) = call {
                self.check_method_level_bounds(
                    site,
                    method_name,
                    arg_matching[0],
                    arguments,
                    bindings,
                );
            }
            return;
        }
        if method_def.defs.is_empty() {
            return;
        }

        // When multiple overloads all fail, report only UnresolvedOverloading.
        // Individual argument errors against first_overload are unreliable
        // because they may not apply to the overload the caller intended.
        if method_def.defs.len() > 1 {
            let receiver_type = self.display_type(receiver_type);
            let call_arguments = self.display_call_arguments(arguments);
            let method_types = self.display_method_overloads(method_def);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: self.span_to_location(loc_span),
                kind: DiagnosticKind::UnresolvedOverloading {
                    method_name: method_name.to_string(),
                    receiver_type,
                    call_arguments,
                    method_types,
                },
            });
            return;
        }

        let first_overload = &method_def.defs[0].type_;
        // Keyword-less overload: match the braceless keyword hash as one
        // positional (see `fold_braceless_keywords_for`). `unfolded` is
        // kept for the surplus case below, which reports per-keyword.
        let unfolded = arguments;
        let folded = self.fold_braceless_keywords_for(arguments, first_overload);
        let arguments = folded.as_ref().unwrap_or(arguments);
        // `method_def.defs.len() > 1` already short-circuited to
        // `UnresolvedOverloading` above, so by this point there is
        // exactly one def and its `defined_in` unambiguously owns every
        // diagnostic emitted below.
        let defined_in = self
            .env
            .names()
            .display_type_name(method_def.defs[0].defined_in);
        let required_len = first_overload.min_arity();
        let optional_len = first_overload.optional_positionals().len();
        let max_len = required_len + optional_len;
        let rest_type = first_overload.rest_positional();

        // Track whether a positional arity error occurred.
        // When arity is wrong, positional type checking is unreliable
        // because arguments are shifted — skip it to avoid cascade errors.
        let mut positional_arity_error = false;
        let mut report_folded_keywords = false;

        // Splat tail handling. An unexpandable trailing splat (Array[E] or
        // opaque untyped) takes one of four paths:
        //
        //   * Untyped tail + overload has a rest slot — silent. The rest
        //     slot absorbs the opaque tail regardless of length.
        //   * Untyped tail + no rest slot — `UnexpectedPositionalArgument`
        //     (Steep parity / user decision 2026-06-07: surface the gap
        //     even though the tail's length is statically unknown).
        //   * Non-untyped tail + rest slot — subtype-check the tail's
        //     element type against the rest type AND against any residual
        //     required slot the expanded prefix didn't cover (Steep's
        //     `uniform_type` intersection: at runtime a tail element may
        //     land in either slot, so it must satisfy both).
        //   * Non-untyped tail + no rest slot — `UnexpectedPositionalArgument`.
        //
        // The expanded prefix's per-positional subtype check still runs
        // afterwards (even under an untyped tail) so determinate prefix
        // slots like `f("bad", *opaque)` flag the `"bad"` mismatch.
        if let Some(tail) = &arguments.splat_tail {
            if let Some(rest_ty) = rest_type {
                if !tail.element_ty.is_untyped() {
                    let rest_s = subst(rest_ty);
                    if !subtyper.check(tail.element_ty, rest_s) {
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: self.span_to_location(tail.span),
                            kind: DiagnosticKind::ArgumentTypeMismatch {
                                method_name: method_name.to_string(),
                                param_index: Some(arguments.positional.len()),
                                keyword: None,
                                expected: self.display_type(rest_s),
                                actual: self.display_type(tail.element_ty),
                                defined_in: Some(defined_in.clone()),
                            },
                        });
                    }
                    let expanded_count = arguments.positional.len();
                    let call_arg_count = expanded_count;
                    for slot in expanded_count..required_len {
                        if let Some(expected) =
                            first_overload.positional_param_for_call(slot, call_arg_count)
                        {
                            let expected_s = subst(expected);
                            if !subtyper.check(tail.element_ty, expected_s) {
                                self.push_diagnostic(Diagnostic {
                                    scope: None,
                                    location: self.span_to_location(tail.span),
                                    kind: DiagnosticKind::ArgumentTypeMismatch {
                                        method_name: method_name.to_string(),
                                        param_index: Some(slot),
                                        keyword: None,
                                        expected: self.display_type(expected_s),
                                        actual: self.display_type(tail.element_ty),
                                        defined_in: Some(defined_in.clone()),
                                    },
                                });
                            }
                        }
                    }
                }
            } else {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(loc_span),
                    kind: DiagnosticKind::UnexpectedPositionalArgument {
                        method_name: method_name.to_string(),
                        expected: max_len,
                        actual: arguments.positional.len() + 1,
                        defined_in: Some(defined_in.clone()),
                    },
                });
                // Don't set `positional_arity_error`: the expanded prefix
                // slots are determinate (it's only the *tail* with nowhere
                // to land), so per-positional subtype check on the prefix
                // should still fire — e.g. `f("bad", *opaque)` reports
                // both `UnexpectedPositionalArgument` for the opaque tail
                // and `ArgumentTypeMismatch` for the literal `"bad"`.
            }
        } else {
            if arguments.positional.len() < required_len {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(loc_span),
                    kind: DiagnosticKind::InsufficientPositionalArguments {
                        method_name: method_name.to_string(),
                        expected: required_len,
                        actual: arguments.positional.len(),
                        defined_in: Some(defined_in.clone()),
                    },
                });
                positional_arity_error = true;
            }

            if !positional_arity_error
                && rest_type.is_none()
                && arguments.positional.len() > max_len
            {
                // Steep (`send_args.rb` `PositionalArgs::UnexpectedArg` with
                // a `:kwargs` node): when the surplus positional is the
                // folded keyword hash, report each keyword as
                // `UnexpectedKeywordArgument` — the user wrote keywords,
                // so point at them rather than at an invisible Hash.
                // Emitted after the per-positional check below so
                // diagnostics stay in source order.
                let surplus_is_folded_hash = folded.is_some();
                report_folded_keywords = surplus_is_folded_hash;
                let plain_surplus =
                    arguments.positional.len() - usize::from(surplus_is_folded_hash);
                if plain_surplus > max_len {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: self.span_to_location(loc_span),
                        kind: DiagnosticKind::UnexpectedPositionalArgument {
                            method_name: method_name.to_string(),
                            expected: max_len,
                            actual: plain_surplus,
                            defined_in: Some(defined_in.clone()),
                        },
                    });
                    positional_arity_error = true;
                }
                // When only the folded hash overflows, the prefix slots
                // are determinate (same reasoning as the splat-tail arm
                // above), so `f("bad", color: "x")` against `(Integer)`
                // still reports the mismatch on `"bad"` alongside the
                // unexpected keyword.
            }
        }

        // Per-positional subtype check on the expanded prefix. Runs
        // even when the tail is untyped — the expanded slots are
        // determinate regardless of the tail's opaqueness, so
        // `f("bad", *opaque)` should still flag `"bad"`. The only
        // suppression is `positional_arity_error`: when the overload's
        // arity is wrong outright (no-tail path or rest-less untyped
        // tail), shifted arg/param pairs would cascade into noise.
        if !positional_arity_error {
            let arg_count = arguments.positional.len();
            for (index, &actual) in arguments.positional.iter().enumerate() {
                let Some(expected) = first_overload.positional_param_for_call(index, arg_count)
                else {
                    continue;
                };

                let expected_s = subst(expected);
                if !subtyper.check(actual, expected_s) {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: self.span_to_location(arguments.positional_spans[index]),
                        kind: DiagnosticKind::ArgumentTypeMismatch {
                            method_name: method_name.to_string(),
                            param_index: Some(index),
                            keyword: None,
                            expected: self.display_type(expected_s),
                            actual: self.display_type(actual),
                            defined_in: Some(defined_in.clone()),
                        },
                    });
                }
            }
        }

        if report_folded_keywords {
            for (keyword_name, _, _, keyword_name_span) in &unfolded.keywords {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(*keyword_name_span),
                    kind: DiagnosticKind::UnexpectedKeywordArgument {
                        method_name: method_name.to_string(),
                        keyword: keyword_name.clone(),
                        defined_in: Some(defined_in.clone()),
                    },
                });
            }
        }

        // Keyword checks are independent of positional checks
        for (required_name, _) in first_overload.required_keywords() {
            if !arguments
                .keywords
                .iter()
                .any(|(name, _, _, _)| name == required_name)
            {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(loc_span),
                    kind: DiagnosticKind::InsufficientKeywordArguments {
                        method_name: method_name.to_string(),
                        keyword: required_name.clone(),
                        defined_in: Some(defined_in.clone()),
                    },
                });
            }
        }

        for (keyword_name, actual, keyword_span, keyword_name_span) in &arguments.keywords {
            let expected = first_overload
                .required_keywords()
                .iter()
                .find(|(name, _)| name == keyword_name)
                .map(|(_, ty)| *ty)
                .or_else(|| {
                    first_overload
                        .optional_keywords()
                        .iter()
                        .find(|(name, _)| name == keyword_name)
                        .map(|(_, ty)| *ty)
                })
                .or(first_overload.rest_keyword());

            if let Some(expected_type) = expected {
                if !subtyper.check(*actual, expected_type) {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: self.span_to_location(*keyword_span),
                        kind: DiagnosticKind::ArgumentTypeMismatch {
                            method_name: method_name.to_string(),
                            param_index: None,
                            keyword: Some(keyword_name.clone()),
                            expected: self.display_type(expected_type),
                            actual: self.display_type(*actual),
                            defined_in: Some(defined_in.clone()),
                        },
                    });
                }
            } else {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: self.span_to_location(*keyword_name_span),
                    kind: DiagnosticKind::UnexpectedKeywordArgument {
                        method_name: method_name.to_string(),
                        keyword: keyword_name.clone(),
                        defined_in: Some(defined_in.clone()),
                    },
                });
            }
        }

        // Block check is independent of argument checks
        if let Some(block) = &first_overload.block
            && block.required
            && !arguments.has_block
        {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: self.span_to_location(loc_span),
                kind: DiagnosticKind::RequiredBlockMissing {
                    method_name: method_name.to_string(),
                },
            });
        }
    }

    pub(super) fn lookup_block_type<'pr>(
        &self,
        node: CallSite<'_, 'pr>,
        target: &CallTarget,
        call_hint: Option<Ty>,
    ) -> Option<Block> {
        // Pick the block-bearing overload that best matches the call's args.
        // Relying on `overloads.first().block` would miss block checks whenever
        // the first overload is block-less (e.g. `def m: () -> T | () { () -> T } -> T`
        // where the no-block form is listed first). When multiple block-bearing
        // overloads exist with differing block signatures, narrow by arity and
        // argument subtyping so the block's expected types match the overload
        // actually selected — otherwise generic block params would be taken
        // from an unrelated overload.
        if let CallTarget::UnionMethod { components, .. } = target {
            return self.lookup_union_block_type(node, components, call_hint);
        }
        let overload = self.select_block_overload(node, target, call_hint)?;
        let block = overload.block.clone()?;
        let arguments = self.collect_call_arguments(node);
        let recv = target.bindings();
        let receiver_type = self.receiver_type_for_site(node);
        Some(self.resolve_block_for_overload(
            overload,
            &block,
            &arguments,
            &recv,
            receiver_type,
            call_hint,
        ))
    }

    /// Concretize a selected overload's block against the call site, then
    /// substitute the receiver. Applies call-site bindings so generic block
    /// params/returns resolve (e.g. `include Container[V]` makes a block param
    /// `V` resolve to Integer; without it generic block types stay
    /// `TypeVariable`, which the subtyper wildcard-matches and misses real
    /// mismatches). For method-level type params (`[X]`) a trailing `#:` hint
    /// overrides the seed-derived bindings, mirroring `infer_return_type` so
    /// `xs: U` resolves through the hint rather than the seed's
    /// `Array[untyped]`. Shared by the single-receiver path and each union
    /// component so the two cannot drift.
    fn resolve_block_for_overload(
        &self,
        overload: &crate::types::MethodType,
        block: &Block,
        arguments: &CallArguments,
        recv_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
        call_hint: Option<Ty>,
    ) -> Block {
        let mut bindings = self.collect_call_site_bindings(overload, arguments, recv_bindings);
        // Pass `None` for the diagnostic location: the same hint conflict is
        // already reported by `infer_return_type` in `infer_call_return_type`.
        if let Some(hint_ty) = call_hint {
            self.apply_hint_override(overload, hint_ty, &mut bindings, None);
        }
        let substitution = self.substitution_for_call_receiver(receiver_type, bindings);
        substitution.apply_block(block, self.env.types())
    }

    /// Synthesize the block type for a `Type::Union` receiver by folding each
    /// component's selected block per Steep `MethodType#|`
    /// (`method_type.rb:234-271`): block params union, block return
    /// intersection, self_type union/Bot/None. Each component's block is
    /// resolved under its own `receiver_type`/`bindings` before folding so
    /// generic block params/returns concretize correctly.
    ///
    /// A component with no block-bearing overload is skipped rather than
    /// bailing the whole fold — Steep `MethodType#|` keeps the other side's
    /// optional block (`when b.optional? → b`). Dispatch (Axis A,
    /// `union_method_block_survives`) has already dropped required-block +
    /// blockless unions, so a blockless component reaching here means
    /// optional + blockless, whose optional block body must still be checked.
    /// Returns `None` (no block-body check) when no component contributes a
    /// block or the folded blocks are arity-incompatible — the conservative
    /// choice that avoids false positives. self_type folding is handled by
    /// `union_blocks` (both-Some → union, one-Some → Bot, none → None).
    fn lookup_union_block_type<'pr>(
        &self,
        node: CallSite<'_, 'pr>,
        components: &[super::UnionComponent],
        call_hint: Option<Ty>,
    ) -> Option<Block> {
        let arguments = self.collect_call_arguments(node);
        let mut acc: Option<Block> = None;
        for comp in components {
            // Skip a component with no block-bearing overload rather than
            // bailing the whole fold: Steep `MethodType#|` keeps the other
            // side's optional block (`when b.optional? → b`). Axis A has
            // already dropped required-block + blockless unions, so a
            // blockless component here means optional + blockless, whose
            // optional block body must still be checked.
            let Some(overload) = self.select_block_overload_for(
                &comp.method_def,
                comp.receiver_type,
                &comp.bindings,
                node,
                call_hint,
            ) else {
                continue;
            };
            let Some(block) = overload.block.clone() else {
                continue;
            };
            let resolved = self.resolve_block_for_overload(
                overload,
                &block,
                &arguments,
                &comp.bindings,
                comp.receiver_type,
                call_hint,
            );
            acc = Some(match acc {
                None => resolved,
                Some(prev) => self.union_blocks(&prev, &resolved)?,
            });
        }
        acc
    }

    /// Fold two resolved blocks per Steep `MethodType#|`: params union
    /// (position-wise; arity must match else `None`), return intersection,
    /// required OR (the dual of Steep's `optional && optional`).
    ///
    /// `self_type` folds per Steep `method_type.rb:236-279`: both-`Some` →
    /// `union_of` (block `self` becomes the union, so `self.method` dispatches
    /// over every component via the existing `Type::Union` path), one-`Some` →
    /// `Bot` (block `self` collapses to `BOTTOM`, so `check_no_method` reports
    /// every self-send — see its bot-receiver arm), none → `None` (block `self`
    /// stays the outer/lexical self).
    ///
    /// Using `union_of` for both-`Some` mirrors Steep's `Union.build`, which
    /// drops `Bot` as the union identity (`ast/types/union.rb:29-30`). So a
    /// left fold that has already produced `BOTTOM` (one earlier one-`Some`)
    /// and then meets another `Some` collapses back to that `Some` — Steep's
    /// fold-order-dependent quirk, reproduced for free rather than special-cased.
    fn union_blocks(&self, a: &Block, b: &Block) -> Option<Block> {
        let types = self.env.types();
        let a_params = a.params();
        let b_params = b.params();
        if a_params.len() != b_params.len() {
            return None;
        }
        let merged_params: Vec<Ty> = a_params
            .iter()
            .zip(b_params.iter())
            .map(|(&x, &y)| union_of(x, y, types))
            .collect();
        let merged_return = intersection_of(a.return_type(), b.return_type(), types);
        let self_type = match (a.self_type, b.self_type) {
            (Some(x), Some(y)) => Some(union_of(x, y, types)),
            (Some(_), None) | (None, Some(_)) => Some(Ty::BOTTOM),
            (None, None) => None,
        };
        let mut function = crate::types::Function::empty(merged_return);
        function.required_positionals = merged_params;
        Some(Block {
            required: a.required || b.required,
            type_: crate::types::FunctionType::Typed(function),
            self_type,
        })
    }

    /// Pick the block-bearing overload for a call. Must stay consistent
    /// with `infer_return_type`'s selection so the block body is validated
    /// against the same overload whose return type is reported.
    ///
    /// Resolution order:
    /// 1. Inference-side narrowing (not block-only) committed to an
    ///    overload → if it has a block clause use it; if it's block-less
    ///    the block is semantically ignored and we return `None` (no
    ///    block-body check fires). Querying inference first is load-
    ///    bearing: the block-only narrowing may return a bound-violating
    ///    block-bearing overload via its own bound fallback even when
    ///    inference commits to a bound-valid non-block overload, so
    ///    trusting the block-only result first would route block checks
    ///    to an unreachable signature.
    /// 2. Block-only narrowing returns a survivor → use it. Reached only
    ///    when step 1's inference result is empty, so no conflict with
    ///    `infer_return_type`'s selection is possible.
    /// 3. Both narrowings empty (structural mismatch, e.g. wrong
    ///    positional arg) → best-effort: prefer a block-bearing overload
    ///    whose arity accepts the call. Gives `m(:sym) { block }` on
    ///    `() { ... } | (Integer) { ... }` a target for
    ///    `BlockBodyTypeMismatch` even when positional matching fails.
    /// 4. If no arity-fit block-bearing overload exists but the method
    ///    has *any* block-bearing overload, fall back to the first one in
    ///    definition order so a block-body diagnostic still surfaces
    ///    alongside the arity error. Preserves the pre-refactor behavior
    ///    for methods whose every overload carries a
    ///    block clause.
    fn select_block_overload<'a, 'pr>(
        &self,
        node: CallSite<'_, 'pr>,
        target: &'a CallTarget,
        call_hint: Option<Ty>,
    ) -> Option<&'a crate::types::MethodType> {
        let def = target.method_definition()?;
        let recv_bindings = target.bindings();
        let receiver_type = self.receiver_type_for_site(node);
        self.select_block_overload_for(def, receiver_type, &recv_bindings, node, call_hint)
    }

    /// Receiver-explicit core of `select_block_overload`. Union dispatch
    /// resolves the block-bearing overload per component, so it passes the
    /// component's own `receiver_type`/`bindings` rather than the union-level
    /// `infer_receiver_type(node)` (which would be the whole union and narrow
    /// against the wrong receiver).
    fn select_block_overload_for<'a, 'pr>(
        &self,
        def: &'a crate::definition::Method,
        receiver_type: Ty,
        recv_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        node: CallSite<'_, 'pr>,
        call_hint: Option<Ty>,
    ) -> Option<&'a crate::types::MethodType> {
        let arguments = self.collect_call_arguments(node);

        let inferred = self.narrow_overloads(
            def,
            &arguments,
            recv_bindings,
            receiver_type,
            Some(node),
            false,
            call_hint,
        );
        if let Some(first) = inferred.first() {
            if first.block.is_some() {
                return Some(*first);
            }
            return None;
        }

        let block_only = self.narrow_overloads(
            def,
            &arguments,
            recv_bindings,
            receiver_type,
            Some(node),
            true,
            call_hint,
        );
        if let Some(first) = block_only.first() {
            return Some(*first);
        }

        let arg_count = arguments.positional.len();
        if let Some(arity_fit) = def
            .method_types()
            .find(|o| o.block.is_some() && (o.is_untyped_function() || o.arity_accepts(arg_count)))
        {
            return Some(arity_fit);
        }

        def.method_types().find(|o| o.block.is_some())
    }

    /// Resolve the return type of a `&:sym` block pass against the expected
    /// block parameter type — the Symbol#to_proc special case. Port of
    /// Steep's `:block_pass` symbol-literal branch (`type_construction.rb`):
    /// look up `method_name` on `param_ty` and take the return type of the
    /// first overload callable with zero arguments (same gate as
    /// `try_convert_to_ary_return_type`; Steep filters on `params.nil? ||
    /// params.optional?` and fetches the first survivor). Private methods
    /// are allowed (Steep passes `private: true`).
    ///
    /// Union receivers are resolved per member — all members must carry the
    /// method or the whole union reports `NoMethod` (Steep builds the union
    /// interface, where one missing member drops the method; measured
    /// 2026-08-30 on `[1, nil].map(&:succ)`). `Skip` covers everything that
    /// must stay silent and untyped: untyped / type-variable params (gradual
    /// escape hatch), receiver kinds `lookup_method` cannot answer for, and
    /// a method that exists but needs arguments (Steep falls through to its
    /// generic block-pass compatibility check there, which crema does not
    /// port — see the todo's intentional-divergence notes).
    pub(super) fn symbol_to_proc_return_type(
        &self,
        param_ty: Ty,
        method_name: crate::name::Symbol,
    ) -> SymbolToProcResolution {
        if param_ty.is_untyped() || crate::types::contains_type_variable(param_ty, self.env.types())
        {
            return SymbolToProcResolution::Skip;
        }
        if let Type::Union(members) = self.env.types().resolve(param_ty) {
            let members = members.clone();
            let mut returns = Vec::new();
            let mut missing = Vec::new();
            for member in members {
                match self.symbol_to_proc_return_type(member, method_name) {
                    SymbolToProcResolution::Resolved(ty) => returns.push(ty),
                    SymbolToProcResolution::NoMethod { missing: m } => missing.extend(m),
                    SymbolToProcResolution::Skip => return SymbolToProcResolution::Skip,
                }
            }
            if !missing.is_empty() {
                return SymbolToProcResolution::NoMethod { missing };
            }
            return SymbolToProcResolution::Resolved(crate::types::union_of_many(
                &returns,
                self.env.types(),
            ));
        }
        // `lookup_method` also returns `None` for receiver kinds it does not
        // handle (Var, Self, Bool, ...). Gate through the NoMethod receiver
        // classifier so only kinds the regular NoMethod path would report on
        // produce a NoMethod here; everything else stays silent.
        let Some(resolved) = method_resolver::lookup_method(self.env, param_ty, method_name) else {
            return match self.classify_no_method_receiver(param_ty) {
                NoMethodReceiverGate::Pass(_) => SymbolToProcResolution::NoMethod {
                    missing: vec![param_ty],
                },
                _ => SymbolToProcResolution::Skip,
            };
        };
        let Some(arg_free) = resolved.method.method_types().find(|mt| {
            mt.is_untyped_function()
                || (mt.required_positionals().is_empty()
                    && mt.required_keywords().is_empty()
                    && mt.trailing_positionals().is_empty())
        }) else {
            return SymbolToProcResolution::Skip;
        };
        let subst =
            definition_builder::substitution_for_receiver(self.env, param_ty, resolved.bindings);
        SymbolToProcResolution::Resolved(subst.apply(arg_free.return_type(), self.env.types()))
    }

    /// NoMethod check for a `&:sym` block pass: when the expected block takes
    /// exactly one argument and the symbol's method does not resolve on the
    /// (typed) parameter type, report `NoMethod` at the symbol. Diverges from
    /// Steep's opaque `Cannot pass a value of type ::Proc ...` wording by
    /// design — the diagnostic names the receiver type and method directly.
    fn check_symbol_to_proc_block_pass<'pr>(
        &mut self,
        node: CallSite<'_, 'pr>,
        target: &CallTarget,
        method_name: &str,
        sym_offset: usize,
        call_hint: Option<Ty>,
    ) {
        let Some(expected_block) = self.lookup_block_type(node, target, call_hint) else {
            return;
        };
        let Some(param_ty) = symbol_to_proc_one_arg_param(&expected_block) else {
            return;
        };
        let sym = self.env.names().intern_symbol(method_name);
        let SymbolToProcResolution::NoMethod { missing } =
            self.symbol_to_proc_return_type(param_ty, sym)
        else {
            return;
        };
        // Union members must all pass the classifier gate, mirroring
        // `check_no_method_union_at`: partial RBS suppresses the whole report.
        for &member in &missing {
            if !matches!(
                self.classify_no_method_receiver(member),
                NoMethodReceiverGate::Pass(_)
            ) {
                return;
            }
        }
        let missing_from = if matches!(self.env.types().resolve(param_ty), Type::Union(_)) {
            let mut displays: Vec<String> = missing.iter().map(|&m| self.display_type(m)).collect();
            displays.sort();
            displays
        } else {
            Vec::new()
        };
        let location = self.offset_to_location(sym_offset);
        self.push_diagnostic(Diagnostic {
            scope: None,
            location,
            kind: DiagnosticKind::NoMethod {
                method_name: method_name.to_string(),
                receiver_type: self.display_type(param_ty),
                missing_from,
            },
        });
    }

    /// Check block parameter types and return type against the RBS block type.
    pub(super) fn check_block<'pr>(
        &mut self,
        node: CallSite<'_, 'pr>,
        target: Option<&CallTarget>,
        call_hint: Option<Ty>,
    ) {
        let Some(block_node) = node.block() else {
            return;
        };
        let Some(target) = target else {
            return;
        };
        if let Some((name, sym_offset)) = block_pass_symbol(&block_node) {
            self.check_symbol_to_proc_block_pass(node, target, &name, sym_offset, call_hint);
            return;
        }
        let Some(block) = block_node.as_block_node() else {
            return;
        };
        let Some(expected_block) = self.lookup_block_type(node, target, call_hint) else {
            return;
        };

        let block_return_type = expected_block.return_type();
        if block_return_type.is_untyped() || block_return_type == Ty::VOID {
            return;
        }

        let Some(body) = block.body() else {
            let subtyper = self.subtyper();
            if !subtyper.check(Ty::NIL, block_return_type) {
                let method_name = self.method_name_for_site(node);
                let location = self.offset_to_location(block.location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location,
                    kind: DiagnosticKind::BlockBodyTypeMismatch {
                        method_name,
                        expected: self.display_type(block_return_type),
                        actual: "nil".to_string(),
                    },
                });
            }
            return;
        };

        // Propagate the substituted block return type as a hint so the
        // body's final expression elaborates tuple/array/record shapes
        // against it. `lookup_block_type` may leave unresolved
        // `TypeVariable`s when no call-site anchor solved them (e.g.
        // `def m: [X] () { () -> X } -> X` called without an LHS hint);
        // pass `None` in that case to keep bare variables out of literal
        // inference. `check_statements_with_hint` forwards the hint to
        // the body's final statement only.
        let body_hint = if crate::types::contains_type_variable(block_return_type, self.env.types())
        {
            None
        } else {
            Some(block_return_type)
        };
        let actual = self.infer_type(&body, body_hint);
        // Extract-mode hand-off: `actual` is the body's tail-expression
        // type (`infer_statements_with_hint` infers exactly the last
        // statement), so when the tail is itself a call, carry the value
        // to its record point (reached by `check_call`'s block visit
        // AFTER this). The read-only infer applies no statement
        // assertions, so the carried value is the call's own computed
        // type. Deposited before the untyped early-return below — an
        // untyped body type is still a value this check computed.
        if self.extract.is_some() {
            let tail = body
                .as_statements_node()
                .and_then(|stmts| stmts.body().iter().last());
            let tail = tail.as_ref().unwrap_or(&body);
            if tail.as_call_node().is_some() {
                self.extract_carry_type(
                    (
                        tail.location().start_offset() as u32,
                        tail.location().end_offset() as u32,
                    ),
                    actual,
                );
            }
        }
        if actual.is_untyped() {
            return;
        }

        let subtyper = self.subtyper();
        if !subtyper.check(actual, block_return_type) {
            let method_name = self.method_name_for_site(node);
            let location = self.offset_to_location(block.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location,
                kind: DiagnosticKind::BlockBodyTypeMismatch {
                    method_name,
                    expected: self.display_type(block_return_type),
                    actual: self.display_type(actual),
                },
            });
        }
    }

    /// Compute the type bound to each block parameter, by position.
    ///
    /// Returns `(requireds_types, rest_type)`:
    /// - `requireds_types[i]` — type for the i-th required param (`None` = unbound).
    /// - `rest_type` — type to bind to the `*rest` param, or `None` when there is
    ///   no rest param.
    ///
    /// Auto-splat fires when `expected_params` is a single aggregate and the block
    /// has 2+ required params **or** at least 1 required param plus a rest param.
    /// Pure rest-only `|*b|` does NOT splat: Ruby `yield([1,2]) { |*b| p b }` gives
    /// `[[1,2]]` (whole tuple collected into rest). Mirrors Steep's `BlockParams#zip`
    /// and `expandable?` (`block_params.rb:264-367`):
    ///
    /// - Tuple `[A, B]`: required `i` gets `members[i]` (excess → `nil`); rest
    ///   gets `Array[union of leftover members]`, or `Array[bot]` when there are
    ///   no leftover members (Ruby `*c = []`; Steep uses `nil` but Ruby is `[]`).
    /// - `Array[T]`: every required gets `T | nil`; rest gets `Array[T]`.
    /// - Non-aggregate single yield: required `0` gets the type, others get `nil`;
    ///   rest gets `Array[bot]` (no leftover elements; Steep uses `Array[untyped]`).
    ///
    /// Non-splat path (index mapping): required `i` gets `expected_params[i]` if
    /// present, otherwise `None` (unbound). Rest gets `Array[union of remaining
    /// expected_params]`, or `Array[bot]` when no params remain after requireds.
    /// This handles `|*b|` (rest-only) correctly: rest gets `Array[expected_params[0]]`.
    /// `block_param_types` when the call's block type resolved; every param
    /// UNTYPED otherwise, so an unresolvable block never invents a binding
    /// (e.g. `Array[bot]` for `*rest`) that could surface a new diagnostic.
    fn unresolved_or_block_param_types(
        &self,
        expected_block: Option<&Block>,
        requireds_count: usize,
        rest_present: bool,
    ) -> (Vec<Option<Ty>>, Option<Ty>) {
        match expected_block {
            Some(block) => self.block_param_types(block.params(), requireds_count, rest_present),
            None => (
                vec![Some(Ty::UNTYPED); requireds_count],
                rest_present.then_some(Ty::UNTYPED),
            ),
        }
    }

    pub(super) fn block_param_types(
        &self,
        expected_params: &[Ty],
        requireds_count: usize,
        rest_present: bool,
    ) -> (Vec<Option<Ty>>, Option<Ty>) {
        let splat = expected_params.len() == 1
            && (requireds_count >= 2 || (requireds_count >= 1 && rest_present));
        if splat {
            let expanded = definition_builder::expand_alias(self.env, expected_params[0]);
            match self.env.types().resolve(expanded) {
                Type::Tuple(members) => {
                    let requireds = (0..requireds_count)
                        .map(|i| Some(members.get(i).copied().unwrap_or(Ty::NIL)))
                        .collect();
                    let rest = rest_present.then(|| {
                        let start = requireds_count.min(members.len());
                        let leftover = &members[start..];
                        let elem = match leftover {
                            [] => Ty::BOTTOM,
                            [single] => *single,
                            many => self.env.types().intern(Type::Union(many.to_vec())),
                        };
                        self.make_array_type(elem)
                    });
                    return (requireds, rest);
                }
                Type::ClassInstance { name, args }
                    if args.len() == 1 && self.is_array_name(name) =>
                {
                    let elem = args[0];
                    let elem_or_nil = union_of(elem, Ty::NIL, self.env.types());
                    let requireds = (0..requireds_count).map(|_| Some(elem_or_nil)).collect();
                    let rest = rest_present.then(|| self.make_array_type(elem));
                    return (requireds, rest);
                }
                _ => {
                    // Non-aggregate: first required gets the yielded type, others get nil.
                    // Rest gets Array[bot] — all expected params are consumed by requireds.
                    let requireds = (0..requireds_count)
                        .map(|i| Some(if i == 0 { expected_params[0] } else { Ty::NIL }))
                        .collect();
                    let rest = rest_present.then(|| self.make_array_type(Ty::BOTTOM));
                    return (requireds, rest);
                }
            }
        }
        // Index mapping (no splat). Rest gets Array of the expected_params residual
        // after requireds have claimed their slots (mirrors Steep block_params.rb:310-321).
        let requireds = (0..requireds_count)
            .map(|i| expected_params.get(i).copied())
            .collect();
        let rest = rest_present.then(|| {
            let remaining = if requireds_count < expected_params.len() {
                &expected_params[requireds_count..]
            } else {
                &[][..]
            };
            let elem = match remaining {
                [] => Ty::BOTTOM,
                [single] => *single,
                many => self.env.types().intern(Type::Union(many.to_vec())),
            };
            self.make_array_type(elem)
        });
        (requireds, rest)
    }

    fn make_array_type(&self, elem: Ty) -> Ty {
        let name = self.env.names().builtins().array;
        self.env.types().intern(Type::ClassInstance {
            name,
            args: vec![elem],
        })
    }

    fn is_array_name(&self, name: &crate::type_name::TypeName) -> bool {
        name == &self.env.names().builtins().array
    }

    /// Push a block scope and bind block parameters. Returns true if scope was pushed.
    ///
    /// The scope is pushed for every `BlockNode`, even when the block type
    /// cannot be resolved (untyped receiver, sig without a block, unresolved
    /// method on self). The body is walked regardless, and prism's
    /// `LocalVariableWriteNode#depth` counts every enclosing block, so the
    /// scope stack must have a matching `ScopeKind::Block` or an outer-lvar
    /// write inside the block overshoots the stack (`set_local_variable_at_depth`)
    /// and is dropped. Params bind UNTYPED in that case, mirroring the
    /// hintless arm of `check_lambda_node`. Only a `&blk` argument
    /// (`BlockArgumentNode`, no body) skips the push.
    pub(super) fn setup_block_scope<'pr>(
        &mut self,
        node: CallSite<'_, 'pr>,
        target: Option<&CallTarget>,
        call_hint: Option<Ty>,
    ) -> bool {
        let Some(block_node) = node.block() else {
            return false;
        };
        let Some(block) = block_node.as_block_node() else {
            return false;
        };
        let expected_block =
            target.and_then(|target| self.lookup_block_type(node, target, call_hint));

        self.ctx.push_scope(ScopeKind::Block);
        if let Some(self_ty) = expected_block.as_ref().and_then(|b| b.self_type) {
            // A `[self: self]` block binding substitutes to SELF_TYPE via
            // `substitution_for_call_receiver`. Concretize it before storing as
            // the override: `current_self_type()` must never become the opaque
            // SELF_TYPE, or lookups inside the block (and NoMethod reporting)
            // fall through and silent-drop. `self_receiver_type()` still yields
            // SELF_TYPE for identity tracking within the block.
            self.ctx
                .set_self_type_override(self.concrete_self_for_lookup(self_ty));
        }

        if let Some(params_node) = block.parameters() {
            if let Some(block_params) = params_node.as_block_parameters_node()
                && let Some(params) = block_params.parameters()
            {
                let requireds = params.requireds();
                let rest_present = params.rest().is_some();
                let (param_types, rest_ty) = self.unresolved_or_block_param_types(
                    expected_block.as_ref(),
                    requireds.len(),
                    rest_present,
                );
                for (index, param) in requireds.iter().enumerate() {
                    if let Some(required) = param.as_required_parameter_node()
                        && let Some(Some(expected_type)) = param_types.get(index).copied()
                    {
                        let name_str = String::from_utf8_lossy(required.name().as_slice());
                        let name = self.checker_names().intern(&name_str);
                        let expected_type = self.freeze_self_type_for_lvar_binding(expected_type);
                        self.ctx.set_local_variable(name, expected_type);
                    }
                }
                if let Some(ty) = rest_ty
                    && let Some(rest_node) = params.rest()
                    && let Some(rest_param) = rest_node.as_rest_parameter_node()
                    && let Some(name_id) = rest_param.name()
                {
                    let name_str = String::from_utf8_lossy(name_id.as_slice());
                    let name = self.checker_names().intern(&name_str);
                    let ty = self.freeze_self_type_for_lvar_binding(ty);
                    self.ctx.set_local_variable(name, ty);
                }
            } else if params_node.as_it_parameters_node().is_some() {
                // Implicit `it` (Ruby 3.4+): bind it like a single named param
                // `|it|`. requireds_count=1 with no rest never auto-splats, so
                // `it` receives the whole yielded value — same as `|x|`. Always
                // bind (UNTYPED when the block yields nothing) so a nested
                // implicit `it` can't fall through `lookup_local_variable` to an
                // enclosing block's `it`.
                let (param_types, _) =
                    self.unresolved_or_block_param_types(expected_block.as_ref(), 1, false);
                let it_ty = param_types
                    .first()
                    .copied()
                    .flatten()
                    .unwrap_or(Ty::UNTYPED);
                let name = self.checker_names().intern("it");
                let it_ty = self.freeze_self_type_for_lvar_binding(it_ty);
                self.ctx.set_local_variable(name, it_ty);
            } else if let Some(numbered) = params_node.as_numbered_parameters_node() {
                // Numbered parameters `_1`..`_9` (structural twin of the `it`
                // arm above and of the numbered arm in
                // `augment_bindings_with_block_body`; keep all of them in
                // step). `maximum` is the highest index referenced, which is
                // exactly the required-param count Ruby uses: `maximum == 1`
                // behaves like `|x|` (no auto-splat, receives the whole
                // yielded value), `maximum >= 2` like `|a, b, ...|` (splats).
                // Feeding it to `block_param_types` as `requireds_count` with
                // no rest reproduces both cases without a new rule. Always
                // bind (UNTYPED when the block yields nothing) so a lookup
                // can't fall through to an enclosing binding.
                let count = numbered.maximum() as usize;
                let (param_types, _) =
                    self.unresolved_or_block_param_types(expected_block.as_ref(), count, false);
                for index in 0..count {
                    let ty = param_types
                        .get(index)
                        .copied()
                        .flatten()
                        .unwrap_or(Ty::UNTYPED);
                    let name = self.checker_names().intern(&format!("_{}", index + 1));
                    let ty = self.freeze_self_type_for_lvar_binding(ty);
                    self.ctx.set_local_variable(name, ty);
                }
            }
        }

        true
    }

    /// Run the Tuple then Record element-access specializers, returning the
    /// first match. These are the selectors Steep's `tuple_shape` /
    /// `record_shape` override after merging the widened Array / Hash shape
    /// (`lib/steep/interface/builder.rb`), where the widened `Elem` union has
    /// already lost the per-position / per-key type.
    ///
    /// Gated on the selector name with a typed receiver as a performance
    /// short-circuit: every specializer self-gates on the method name and
    /// receiver shape, but collecting the call arguments to feed them is
    /// wasted work for any other call (the common case). Shared by the
    /// `&mut self` diagnostic pass (`check_call_arguments`, which acts on
    /// `Missing`) and the `&self` return-type pass (`infer_call_return_type`,
    /// which maps `Found`/`Missing` to a type) so the entry condition and
    /// specializer order live in one place. The peek is hintless on purpose —
    /// the specializers inspect the argument's literal (Symbol / Integer), not
    /// its hint-driven widening.
    pub(super) fn element_access_specialization<'pr>(
        &self,
        receiver: Ty,
        method: &str,
        node: &CallNode<'pr>,
    ) -> Option<LiteralAccessResult> {
        // `fetch` joins `[]` for Record receivers only (see `record_key_specialization`).
        // Tuple `fetch` still falls through to the `Array#fetch` widening path
        // because `tuple_index_specialization` keeps its `method != "[]"` gate.
        // `first` / `last` are Tuple-only (see `tuple_end_specialization`).
        if !matches!(method, "[]" | "fetch" | "first" | "last") || receiver.is_untyped() {
            return None;
        }
        // Peel `type t = ...` so alias-wrapped Record/Tuple receivers reach
        // the structural match below. Mirrors `lookup_method`'s `Type::Alias`
        // arm (src/type_checker/method_resolver.rs) and the call dispatcher
        // (`resolve_call_target`); without this, `check_call_arguments`'s
        // raw `receiver_type` skips the specializers and falls back to the
        // `Hash#[]`/`Array#[]` widening path, returning the all-fields union.
        // Cyclic aliases bottom out as `Type::Alias` (`ALIAS_EXPANSION_LIMIT`)
        // and the specializers' `_ => None` arm bails safely.
        let receiver = definition_builder::expand_alias(self.env, receiver);
        // Selector name alone stopped being a useful filter once `first` /
        // `last` joined the gate: those two are common on plain `Array[E]`
        // receivers, which no specializer can serve. Check the receiver's
        // shape before paying for `collect_call_arguments`.
        if !self.has_element_access_shape(receiver) {
            return None;
        }
        let args = self.collect_call_arguments(CallSite::Call(node));

        // Union receiver: dispatch per member. `resolve_call_target` peels
        // Unions for the method-lookup path (ADR-0021), but this entry runs
        // from `check_call_arguments` / `infer_call_return_type` which see
        // the raw `receiver_type`. Without a per-member peel here,
        // `(rec_a | rec_b)[:key]` returns the union of *all* fields across
        // all members (e.g. `String | Git | Local`) rather than the union
        // of the same key's values from each member (`Git | Local`). One
        // unmatched member collapses the whole call to `None` so the
        // widening path handles it intact.
        if let Type::Union(members) = self.env.types().resolve(receiver) {
            let flat = definition_builder::flatten_alias_union_members(self.env, members);
            let mut founds: Vec<Ty> = Vec::with_capacity(flat.len());
            let mut first_missing: Option<(Ty, DiagnosticKind)> = None;
            for member in flat {
                match self
                    .tuple_index_specialization(member, method, &args)
                    .or_else(|| self.tuple_end_specialization(member, method, &args))
                    .or_else(|| self.record_key_specialization(member, method, &args))
                {
                    Some(LiteralAccessResult::Found(ty)) => founds.push(ty),
                    Some(LiteralAccessResult::Missing { fallback, diag }) => {
                        // Do NOT push fallback into founds. fallback is the generic
                        // Hash[] return for that member (union of all its values).
                        // Including it would produce unrelated field types in the result
                        // (e.g. "stdlib" from Stdlib::se polluting a "name" lookup on
                        // Git::se | Stdlib::se). Steep includes the fallback; crema
                        // intentionally diverges here to avoid false positives in
                        // multi-overload sig scenarios.
                        if first_missing.is_none() {
                            first_missing = Some((fallback, diag));
                        }
                    }
                    None => return None,
                }
            }
            if founds.is_empty() {
                let (fallback, diag) = first_missing?;
                return Some(LiteralAccessResult::Missing { fallback, diag });
            }
            return Some(LiteralAccessResult::Found(self.members_union(&founds)));
        }

        self.tuple_index_specialization(receiver, method, &args)
            .or_else(|| self.tuple_end_specialization(receiver, method, &args))
            .or_else(|| self.record_key_specialization(receiver, method, &args))
    }

    /// Whether any specializer could serve this receiver. A Union passes on
    /// one structural member — the per-member loop still bails to `None` for
    /// the whole call if a sibling member can't be served, so this only
    /// decides whether collecting the arguments is worth it.
    fn has_element_access_shape(&self, receiver: Ty) -> bool {
        let is_structural = |ty: Ty| {
            matches!(
                self.env.types().resolve(ty),
                Type::Tuple(_) | Type::Record { .. }
            )
        };
        match self.env.types().resolve(receiver) {
            Type::Union(members) => {
                definition_builder::flatten_alias_union_members(self.env, members)
                    .into_iter()
                    .any(is_structural)
            }
            _ => is_structural(receiver),
        }
    }

    /// Specialize `tuple.first` / `tuple.last` to the element type at that
    /// end. Widening to `Array[union]` for dispatch (`widen_tuple_to_array`)
    /// makes `Array#first`/`#last`'s `() -> Elem` yield the whole element
    /// union, which is what Steep's `tuple_shape` overrides these two
    /// selectors to avoid (builder.rb: `tuple.types[0]` / `tuple.types.last`,
    /// falling back to `nil` for the empty tuple).
    ///
    /// Steep replaces the Array overloads outright, so `last(1)` is an
    /// arity error there. crema deliberately diverges: `["a", 1.0].last(1)`
    /// is valid Ruby, and rejecting working code costs more than the missed
    /// diagnostic. Any call carrying arguments, keywords, a block, or a
    /// splat tail returns `None` and keeps the `Array#first`/`#last`
    /// widening path intact.
    pub(super) fn tuple_end_specialization(
        &self,
        receiver: Ty,
        method: &str,
        arguments: &CallArguments,
    ) -> Option<LiteralAccessResult> {
        let take_last = match method {
            "first" => false,
            "last" => true,
            _ => return None,
        };
        let members = match self.env.types().resolve(receiver) {
            Type::Tuple(members) => members,
            _ => return None,
        };
        if !arguments.positional.is_empty()
            || !arguments.keywords.is_empty()
            || arguments.has_block
            || arguments.splat_tail.is_some()
        {
            return None;
        }
        let element = if take_last {
            members.last().copied()
        } else {
            members.first().copied()
        };
        Some(LiteralAccessResult::Found(element.unwrap_or(Ty::NIL)))
    }

    /// Specialize `tuple[lit]` to the element type when `lit` is a non-negative
    /// integer literal. See `LiteralAccessResult` for the two outcomes. Returns
    /// `None` when the call shape doesn't match (non-tuple receiver, method
    /// other than `[]`, non-literal/negative index, extra args) so the caller
    /// falls through to the existing `Array#[]`-via-widening path.
    pub(super) fn tuple_index_specialization(
        &self,
        receiver: Ty,
        method: &str,
        arguments: &CallArguments,
    ) -> Option<LiteralAccessResult> {
        if method != "[]" {
            return None;
        }
        let members = match self.env.types().resolve(receiver) {
            Type::Tuple(members) => members,
            _ => return None,
        };
        if arguments.positional.len() != 1 || !arguments.keywords.is_empty() {
            return None;
        }
        let arg_ty = arguments.positional[0];
        // A tuple can never have anywhere near i64::MAX elements, so an
        // index literal too large to fit `i64` simply can't match — fall
        // through to the general `Integer#[]` call resolution instead of
        // treating it as a tuple-index specialization.
        let n: i64 = match self.env.types().resolve(arg_ty) {
            Type::Literal(Literal::Integer(s)) => s.parse().ok()?,
            _ => return None,
        };
        if n < 0 {
            return None;
        }
        let idx = n as usize;
        if idx < members.len() {
            Some(LiteralAccessResult::Found(members[idx]))
        } else {
            let fallback = self.members_union(members);
            let diag = DiagnosticKind::UnknownTupleIndex {
                index: n,
                tuple_length: members.len(),
            };
            Some(LiteralAccessResult::Missing { fallback, diag })
        }
    }

    /// Specialize record `[](literal_key)` / `fetch(literal_key, ...)` to the
    /// per-field value type. Without this, Hash widening (`Hash[K, V_union]`)
    /// loses per-key precision, and any record with a `top`-typed field
    /// (e.g. `boolish`) collapses the widened `V` to `top` via union
    /// absorption — killing downstream overload resolution.
    ///
    /// Method semantics (Steep parity — Steep's `Interface::Builder`
    /// `record_shape` generates `[]` and `fetch` overloads independently and
    /// does NOT special-case `record.optional?` for the fetch-with-default
    /// arm; crema matches that shape):
    /// - `[](k)` — required field → `V`; optional field → `V?` (Hash `[]`'s
    ///   `implicitly-returns-nil` shape).
    /// - `fetch(k)` no default — required OR optional → `V`. Static-permissive
    ///   for optional (runtime `KeyError` not surfaced statically). Unknown
    ///   key → `Missing { UnknownRecordKey }`.
    /// - `fetch(k, default)` — key present (required OR optional) →
    ///   `V | default_ty` (RBS overload contract `(K, X) -> (V | X)`; even for
    ///   required fields we honor the contract so a wrong `default` type
    ///   surfaces at the call site). Unknown key → `Missing { UnknownRecordKey }`
    ///   with `fallback = default_ty` (fetch-with-default never raises at
    ///   runtime, but `UnknownRecordKey` is a typo-detection diagnostic;
    ///   suppressing it just because `[]` and `fetch(k, default)` have
    ///   different runtime behavior would silence real typos).
    /// - `fetch(k) { block }` — return `None` so the caller falls back to the
    ///   `Hash#fetch` widening path (block return type is not accessible at
    ///   this layer; a follow-up todo can lift it).
    ///
    /// Non-literal keys, unexpected arity, keyword arguments, or a trailing
    /// splat (`fetch(k, *xs)`) return `None` so the widening path handles the
    /// call intact.
    pub(super) fn record_key_specialization(
        &self,
        receiver: Ty,
        method: &str,
        arguments: &CallArguments,
    ) -> Option<LiteralAccessResult> {
        let is_bracket = method == "[]";
        let is_fetch = method == "fetch";
        if !is_bracket && !is_fetch {
            return None;
        }
        let fields = match self.env.types().resolve(receiver) {
            Type::Record { fields } => fields,
            _ => return None,
        };
        if !arguments.keywords.is_empty() || arguments.splat_tail.is_some() {
            return None;
        }
        // Arity + block gates. `[]` accepts exactly 1 positional (block passes
        // through; `receiver.[](k) { blk }` is a rare form and `[]` semantics
        // ignore blocks). `fetch` accepts 1 or 2 positionals with no block —
        // block form falls back to the widening path.
        let arity_ok = if is_bracket {
            arguments.positional.len() == 1
        } else {
            matches!(arguments.positional.len(), 1 | 2) && !arguments.has_block
        };
        if !arity_ok {
            return None;
        }
        let arg_ty = arguments.positional[0];
        let key = match self.env.types().resolve(arg_ty) {
            Type::Literal(Literal::Symbol(s)) => RecordKey::Symbol(s.clone()),
            Type::Literal(Literal::String(s)) => RecordKey::String(s.clone()),
            Type::Literal(Literal::Integer(i)) => RecordKey::Integer(i.clone()),
            Type::Literal(Literal::Bool(b)) => RecordKey::Bool(*b),
            _ => return None,
        };
        let default_ty = if is_fetch && arguments.positional.len() == 2 {
            Some(arguments.positional[1])
        } else {
            None
        };
        for (field_key, ty, required) in fields {
            if field_key == &key {
                let result = match (is_bracket, *required, default_ty) {
                    // `[](k)` required → V; optional → V? (Hash `[]`
                    // `implicitly-returns-nil`).
                    (true, true, _) => *ty,
                    (true, false, _) => self.env.types().intern(Type::Optional(*ty)),
                    // `fetch(k)` no default (required or optional) → V.
                    (false, _, None) => *ty,
                    // `fetch(k, default)` (required or optional) → V | default_ty.
                    // Steep parity: RBS `[X](K, X) -> (V | X)` overload does not
                    // eliminate X even when the key is statically present.
                    (false, _, Some(dt)) => self.members_union(&[*ty, dt]),
                };
                return Some(LiteralAccessResult::Found(result));
            }
        }
        // Unknown key. Both `[]` and `fetch` surface `UnknownRecordKey` for
        // typo detection, regardless of whether runtime raises. For
        // `fetch(k, default)`, we ALSO emit the diagnostic (via `Missing`)
        // but set `fallback = default_ty` so the return type reflects that
        // the default fires (fetch-with-default never raises `KeyError`).
        let field_tys: Vec<Ty> = fields.iter().map(|(_, t, _)| *t).collect();
        let known_keys: Vec<String> = fields.iter().map(|(k, _, _)| k.display()).collect();
        let fallback = match default_ty {
            Some(dt) => dt,
            None => self.members_union(&field_tys),
        };
        let diag = DiagnosticKind::UnknownRecordKey {
            key: key.display(),
            known_keys,
        };
        Some(LiteralAccessResult::Missing { fallback, diag })
    }

    /// Intern a union from a slice of tys, collapsing trivial cases. Empty
    /// becomes `BOTTOM` — the only kind of "fallback" for an empty tuple/
    /// record (`[]` or `{}`), since there are no members to union.
    fn members_union(&self, members: &[Ty]) -> Ty {
        if members.is_empty() {
            Ty::BOTTOM
        } else if members.len() == 1 {
            members[0]
        } else {
            self.env.types().intern(Type::Union(members.to_vec()))
        }
    }

    pub(super) fn widen_tuple_to_array(&self, members: &[Ty]) -> Ty {
        method_resolver::widen_tuple_to_array(self.env, members)
    }

    pub(super) fn widen_record_to_hash(&self, fields: &[(RecordKey, Ty, bool)]) -> Ty {
        method_resolver::widen_record_to_hash(self.env, fields)
    }
}

impl<'env> TypeChecker<'env> {
    /// Choose the overload whose Record param hints drive bidirectional
    /// hash-literal synthesis at this call site.
    ///
    /// Multi-overload: trial each overload in RBS definition order; the
    /// first whose every hash-literal arg synthesizes + width-subtypes
    /// against its Record hint wins (Steep `pick_one_of` port,
    /// specific-first / general-last RBS convention). `None` means no
    /// overload qualifies — the literal stays `Hash[untyped, untyped]`
    /// and `narrow_overloads` surfaces `UnresolvedOverloading` rather
    /// than committing to a wrong hint.
    pub(super) fn pick_hint_overload<'pr>(
        &self,
        method_def: &crate::definition::Method,
        node: CallSite<'_, 'pr>,
        bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        receiver_type: Ty,
    ) -> Option<crate::types::MethodType> {
        // An empty-`defs` bucket has no `MethodType` to hint with. crema does
        // not validate RBS (ADR-0013), so a bare overloading-only `def foo: ...`
        // — which rbs itself rejects as `InvalidOverloadMethodError` — is
        // accepted and lowered to empty `defs`. Returning here also keeps the
        // `expect("checked above")` below honest: the only way past this guard
        // and the `== 1` return is `defs.len() >= 2`, which the `> 1` early-bail
        // guarantees has `node.arguments()` present.
        if method_def.defs.is_empty() {
            return None;
        }
        // Bail out before building the receiver-aware substitution for the
        // common multi-overload no-hash-literal call (`Array#[]`,
        // `Kernel#format`, every defaulted-arg method). The trial loop is
        // the only consumer that needs substitution; if we won't enter it
        // we keep the path allocation-free.
        if method_def.defs.len() > 1 {
            let arguments = node.arguments()?;
            if !arguments
                .arguments()
                .iter()
                .any(|a| a.as_hash_node().is_some() || a.as_keyword_hash_node().is_some())
            {
                return None;
            }
        }

        // Build the receiver-aware substitution once. `pick_hint_overload`
        // and its downstream consumers (`collect_call_arguments_hinted`,
        // `emit_hash_literal_record_extras`) all need the hint to align
        // with the subtype-check path's substituted expected type — without
        // this, a generic method's hint flows raw (e.g. `Elem` instead of
        // the receiver's concrete `[Integer, String]`) and the literal
        // synthesis layer (`array_tuple_hint` etc.) silently bails on the
        // unresolved TypeVar.
        let types_tbl = self.env.types();
        let substitution = self.substitution_for_call_receiver(receiver_type, bindings.clone());
        let subst_mt =
            |mt: &crate::types::MethodType| substitution.apply_method_type(mt, types_tbl);

        if method_def.defs.len() == 1 {
            return method_def.method_types().next().map(subst_mt);
        }
        // Already verified in the early-bail block above.
        let arguments = node.arguments().expect("checked above");
        let positional_count = arguments
            .arguments()
            .iter()
            .filter(|a| a.as_keyword_hash_node().is_none())
            .count();

        'next_overload: for raw_overload in method_def.method_types() {
            let overload = subst_mt(raw_overload);
            let mut positional_index = 0usize;
            for arg in arguments.arguments().iter() {
                if let Some(keyword_hash) = arg.as_keyword_hash_node() {
                    for elem in keyword_hash.elements().iter() {
                        let Some(assoc) = elem.as_assoc_node() else {
                            continue;
                        };
                        let Some(sym) = assoc.key().as_symbol_node() else {
                            continue;
                        };
                        let name = String::from_utf8_lossy(sym.unescaped());
                        // A keyword this overload doesn't declare (and has
                        // no `**rest` for) can never be the call's target,
                        // regardless of the value's node shape. Without
                        // this early reject, a non-literal value (variable
                        // / method call) falls through
                        // `argument_matches_hint`'s pass-through arm and
                        // the first-declared overload wins by declaration
                        // order even when it lacks this keyword entirely —
                        // handing a wrong (or absent) hint to downstream
                        // inference of the value expression.
                        let Some(hint) = overload.keyword_param_for_call(&name) else {
                            continue 'next_overload;
                        };
                        if !self.argument_matches_hint(&assoc.value(), Some(hint)) {
                            continue 'next_overload;
                        }
                    }
                } else {
                    let hint =
                        overload.positional_param_for_call(positional_index, positional_count);
                    if !self.argument_matches_hint(&arg, hint) {
                        continue 'next_overload;
                    }
                    positional_index += 1;
                }
            }
            return Some(overload);
        }
        None
    }

    /// Pick predicate. Dispatches on argument shape:
    ///
    /// * Hash literal — force the overload's hint to expose a Record
    ///   candidate (after Optional/Union unwrapping) AND the trial
    ///   synthesis to width-subtype that candidate. When no candidate is
    ///   Record-shaped, fall back to plain `Hash[K, V]` hint synthesis
    ///   (mirrors the `infer_type` `HashNode` arm's Record/Hash[K,V]
    ///   fallback pair) so a tuple/array-typed `V` still reaches nested
    ///   array literal values during the trial.
    /// * Simple scalar literal (Integer/Float/String/Symbol/Bool/Nil) —
    ///   compare the literal's own type to the hint via plain subtyping
    ///   so a mismatching positional slot rules out the overload before
    ///   the hash-arg trial commits.
    /// * Variable or expression — pass through. Their subtype check is
    ///   `narrow_overloads`'s job, matching the parent todo's
    ///   "no behavior change for variable hashes" invariant.
    fn argument_matches_hint<'pr>(&self, arg: &Node<'pr>, hint: Option<Ty>) -> bool {
        if let Some(hash) = arg.as_hash_node() {
            let Some(hint_ty) = hint else {
                return false;
            };
            let elements: Vec<_> = hash.elements().iter().collect();
            let candidates = self.expand_hint_candidates(hint_ty);
            if let Some(record_hint) = self.pick_record_hint_from_candidates(&elements, &candidates)
            {
                let mut extras = Vec::new();
                let Some(synth) =
                    self.synthesize_hash_as_record(&elements, record_hint, &mut extras)
                else {
                    return false;
                };
                return self.subtyper().check(synth, record_hint);
            }
            for c in candidates {
                if let Some(hash_ty) = self.synthesize_hash_with_hint(&elements, c)
                    && self.subtyper().check(hash_ty, c)
                {
                    return true;
                }
            }
            return false;
        }

        if !Self::is_simple_literal_arg(arg) {
            return true;
        }
        let Some(hint_ty) = hint else {
            return true;
        };
        let actual = self.infer_type(arg, None);
        self.subtyper().check(actual, hint_ty)
    }

    /// Cheap node-shape predicate for the literal subset whose
    /// `infer_type` arms stay shallow — at most one
    /// `from_utf8_lossy().to_string()` for String/Symbol on the way to
    /// `intern`, no walk and no diagnostic emission. Gates the
    /// `argument_matches_hint` subtype check so non-literal args
    /// (variables, calls, complex expressions) bypass the inference
    /// entry entirely — `narrow_overloads_by_args` retains sole
    /// authority for those.
    fn is_simple_literal_arg<'pr>(arg: &Node<'pr>) -> bool {
        matches!(
            arg,
            Node::IntegerNode { .. }
                | Node::FloatNode { .. }
                | Node::StringNode { .. }
                | Node::SymbolNode { .. }
                | Node::TrueNode { .. }
                | Node::FalseNode { .. }
                | Node::NilNode { .. }
        )
    }

    /// True when `node`'s argument list contains a `ForwardingArgumentsNode`
    /// (`bar(...)`). Used by `check_call_arguments` to route to the
    /// dedicated forwarding-compatibility check instead of the ordinary
    /// per-arg type-check loop.
    pub(super) fn call_has_forwarding_args<'pr>(&self, node: &CallNode<'pr>) -> bool {
        node.arguments()
            .map(|args| {
                args.arguments()
                    .iter()
                    .any(|a| a.as_forwarding_arguments_node().is_some())
            })
            .unwrap_or(false)
    }

    /// `bar(...)` site: compare the caller's `def foo(...)` signature
    /// (`ctx.forward_arg_type()`) against the callee's method definition.
    /// On mismatch, emit `Ruby::IncompatibleArgumentForwarding`. Mirrors
    /// Steep's `lib/steep/type_construction.rb` forwarded_args block.
    /// `None` caller (def has no RBS sig or no `(...)` parameter) is a
    /// silent skip — without a caller signature we cannot judge
    /// compatibility, and the def-side body walk is gated by the same
    /// `method_type().is_some()` check upstream.
    pub(super) fn check_forwarding_call<'pr>(
        &mut self,
        node: &CallNode<'pr>,
        method_def: &crate::definition::method::Method,
        target: &CallTarget,
        receiver_type: Ty,
        loc_span: ArgSpan,
    ) {
        let method_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        // Forwarding calls have no concrete argument types, so
        // per-overload narrowing is out of reach. Fire on the
        // member-level annotation only, matching Steep's parity for
        // the common case (`%a{deprecated}` above the whole `def`).
        self.check_deprecated_member_only(&method_name, &method_def.annotations, loc_span);
        let Some(caller_mt) = self.ctx.forward_arg_type() else {
            return;
        };
        let Some((mismatch_kind, callee_mt)) = self.try_forwarding_compat_method_def(
            method_def,
            &caller_mt,
            receiver_type,
            target.bindings(),
        ) else {
            return;
        };
        let callee_signature = self.display_method_type(&callee_mt);
        let caller_signature = self.display_method_type(&caller_mt);
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: self.span_to_location(loc_span),
            kind: DiagnosticKind::IncompatibleArgumentForwarding {
                method_name,
                caller_signature,
                callee_signature,
                mismatch_kind,
            },
        });
    }

    /// Single-method-def forwarding judgment shared by the single-receiver
    /// (`check_forwarding_call`) and union-receiver (`check_union_call_arguments`)
    /// paths. Receiver-side bindings substitution mirrors the regular call
    /// path (`check_against_method_def`): without it a `Box[Integer]#take`
    /// call forwarded from a `(String) -> String` def would compare caller
    /// `(String)` against raw `(T)` and silently pass.
    ///
    /// Overload-level union (Steep parity): returns `None` (= accepts) as
    /// soon as one overload accepts the caller's forwarded signature, and
    /// the first failing overload otherwise. An empty `method_types()` (e.g.
    /// an unresolved-alias placeholder that slipped through resolution) also
    /// returns `None` — there is no signature to refute, so treat the
    /// component as accepting rather than silently dropping the diagnostic
    /// when paired with a real failure in a sibling component.
    fn try_forwarding_compat_method_def(
        &self,
        method_def: &crate::definition::method::Method,
        caller_mt: &MethodType,
        receiver_type: Ty,
        bindings: FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) -> Option<(ForwardingMismatchKind, MethodType)> {
        let types_tbl = self.env.types();
        let substitution = self.substitution_for_call_receiver(receiver_type, bindings);
        let mut first_failure: Option<(ForwardingMismatchKind, MethodType)> = None;
        for callee_mt in method_def.method_types() {
            let callee_mt = substitution.apply_method_type(callee_mt, types_tbl);
            match self.forwarding_compat(caller_mt, &callee_mt) {
                None => return None,
                Some(kind) => {
                    if first_failure.is_none() {
                        first_failure = Some((kind, callee_mt));
                    }
                }
            }
        }
        first_failure
    }

    /// Subtype-compatibility check for forwarding. Returns `Some(kind)` on
    /// failure (the first failing dimension wins) and `None` when the
    /// caller can forward into the callee.
    fn forwarding_compat(
        &self,
        caller: &MethodType,
        callee: &MethodType,
    ) -> Option<ForwardingMismatchKind> {
        // A caller-side untyped function (`(?) -> untyped`) is a full
        // wildcard: Steep sets `forward_arg_type` to `true` and skips
        // the params and block checks alike.
        if caller.is_untyped_function() {
            return None;
        }
        // A callee-side untyped function only waives the params
        // comparison (Steep gates `check_method_params` on both sides
        // having params); its block lives on the MethodType and is
        // still checked below. RBS cannot currently express `(?)` with
        // a block, so the fall-through is defensive. Without this
        // guard, the empty delegation accessors would trip the
        // caller-rest-without-callee-rest arity gate.
        if let (Some(caller_fn), Some(callee_fn)) = (caller.func(), callee.func())
            && let Some(kind) = self.params_flow_compat(caller_fn, callee_fn)
        {
            return Some(kind);
        }

        // Block — Steep parity: both sides convert to proc-or-nil
        // (`Block#to_proc_type`, lib/steep/interface/block.rb:72-79)
        // and the forwarded (caller) type must be a subtype of the
        // callee's (type_construction.rb:4317-4342). Decomposed
        // structurally instead of building actual proc types. All
        // block-side failures report as `Block`, mirroring Steep's
        // separate `Cannot forward block` message.
        match (&caller.block, &callee.block) {
            (None, None) => {}
            // nil <: proc-or-nil holds only when the callee block is
            // optional.
            (None, Some(cb)) => {
                if cb.required {
                    return Some(ForwardingMismatchKind::Block);
                }
            }
            // proc-or-nil <: nil never holds — the forwarded block has
            // nowhere to go even when the caller's block is optional.
            (Some(_), None) => {
                return Some(ForwardingMismatchKind::Block);
            }
            (Some(caller_block), Some(callee_block)) => {
                if let Some(kind) = self.block_flow_compat(caller_block, callee_block) {
                    return Some(kind);
                }
            }
        }

        None
    }

    /// `caller_block <: callee_block` as proc types (Steep's Proc <:
    /// Proc, `lib/steep/subtyping/check.rb:465-492`): params are
    /// contravariant, the return type is covariant, and the `[self: T]`
    /// binding follows `check_self_type_binding` (check.rb:925-940).
    fn block_flow_compat(
        &self,
        caller_block: &Block,
        callee_block: &Block,
    ) -> Option<ForwardingMismatchKind> {
        // The caller's optional block contributes a nil component that
        // a required callee block cannot absorb.
        if !caller_block.required && callee_block.required {
            return Some(ForwardingMismatchKind::Block);
        }
        // Params are contravariant — the callee's yield values flow
        // into the caller's block — so the flow check runs flipped.
        // Steep skips the params comparison when either side is an
        // untyped function (params nil).
        if let (FunctionType::Typed(callee_fn), FunctionType::Typed(caller_fn)) =
            (&callee_block.type_, &caller_block.type_)
            && self.params_flow_compat(callee_fn, caller_fn).is_some()
        {
            return Some(ForwardingMismatchKind::Block);
        }
        // Return type is covariant: the caller's block return flows out
        // of the callee's yield expression.
        if !self
            .subtyper()
            .check(caller_block.return_type(), callee_block.return_type())
        {
            return Some(ForwardingMismatchKind::Block);
        }
        // `[self: T]` binding: contravariant when both sides bind. A
        // one-sided binding is a mismatch unless the caller side is
        // exactly `top` (Steep's check_self_type_binding else-branch).
        match (caller_block.self_type, callee_block.self_type) {
            (None, None) => None,
            (Some(caller_self), Some(callee_self)) => {
                (!self.subtyper().check(callee_self, caller_self))
                    .then_some(ForwardingMismatchKind::Block)
            }
            (Some(caller_self), None) => {
                (!matches!(self.env.types().resolve(caller_self), Type::Top))
                    .then_some(ForwardingMismatchKind::Block)
            }
            (None, Some(_)) => Some(ForwardingMismatchKind::Block),
        }
    }

    /// Structural params-flow check shared by the method-level
    /// forwarding comparison (`src` = caller, `dst` = callee) and the
    /// block comparison (flipped: `src` = callee block, `dst` = caller
    /// block — yield values flow from the callee into the caller's
    /// block). Answers "can an argument list shaped by `src`'s
    /// parameters always flow into `dst`'s slots?"; element types are
    /// covariant src → dst. Mirrors Steep's `match_params`
    /// (`lib/steep/subtyping/check.rb:994-1086`), which likewise runs
    /// caller→callee at the method level and flipped for proc params
    /// via the contravariant pair reversal.
    fn params_flow_compat(&self, src: &Function, dst: &Function) -> Option<ForwardingMismatchKind> {
        let src_req = &src.required_positionals;
        let src_opt = &src.optional_positionals;
        let dst_req = &dst.required_positionals;
        let dst_opt = &dst.optional_positionals;
        let src_rest = src.rest_positional;
        let dst_rest = dst.rest_positional;
        let dst_min = dst.required_positionals.len();
        let dst_max = if dst_rest.is_some() {
            None
        } else {
            Some(dst.required_positionals.len() + dst.optional_positionals.len())
        };

        // Arity: src's required count must fit within the dst's
        // [min, max]. Steep's `Incompatible arity` branch fires when the
        // required slots themselves can't be filled, or when even the
        // src's full positional (req + opt) overshoots the dst.
        // Exception: when both sides have a rest, the src's rest can
        // fill the dst's leftover required slots (Steep parity, e.g.
        // src `(*Integer)` + dst `(Integer, *Integer)`).
        if src_req.len() < dst_min && !(src_rest.is_some() && dst_rest.is_some()) {
            return Some(ForwardingMismatchKind::Arity);
        }
        if let Some(max) = dst_max
            && src_req.len() + src_opt.len() > max
        {
            return Some(ForwardingMismatchKind::Arity);
        }
        // Src's rest can forward an unbounded count. A dst without
        // a rest slot has a finite max it cannot absorb, so this is
        // always an arity mismatch regardless of element types.
        if src_rest.is_some() && dst_rest.is_none() {
            return Some(ForwardingMismatchKind::Arity);
        }

        // Element subtype: walk src's required+optional positionals
        // and resolve each to the dst's slot at the same index. The
        // dst's slot is its req, then opt, then rest (covariant —
        // src's value flows into the dst's parameter). Without
        // resolving to the dst's rest, a src `(Integer)` flowing
        // into dst `(*String)` would miss the element-type mismatch.
        let subtyper = self.subtyper();
        let src_positionals = src_req.iter().chain(src_opt.iter());
        for (i, src_ty) in src_positionals.enumerate() {
            let dst_slot = if i < dst_req.len() {
                dst_req[i]
            } else if i - dst_req.len() < dst_opt.len() {
                dst_opt[i - dst_req.len()]
            } else {
                // The src-rest-no-dst-rest case is already gated
                // above as Arity, so reaching here means dst has rest.
                dst_rest.expect("dst rest absent past req+opt despite arity gate")
            };
            if !subtyper.check(*src_ty, dst_slot) {
                return Some(ForwardingMismatchKind::Type);
            }
        }
        // Src's rest is the source for any dst req/opt slots that
        // outrun the src's positional list (Steep parity, e.g. src
        // `(*String)` + dst `(Integer, *Integer)` reports the rest
        // element type against the dst's required slot).
        if let Some(src_rest_ty) = src_rest {
            let src_positional_count = src_req.len() + src_opt.len();
            let dst_fixed = dst_req.len() + dst_opt.len();
            for i in src_positional_count..dst_fixed {
                let dst_slot = if i < dst_req.len() {
                    dst_req[i]
                } else {
                    dst_opt[i - dst_req.len()]
                };
                if !subtyper.check(src_rest_ty, dst_slot) {
                    return Some(ForwardingMismatchKind::Type);
                }
            }
        }
        // Both sides have rest: element types must be subtype-compatible
        // (covariant). The rest×no-rest combinations are already handled
        // — src-only-rest was rejected above as Arity, dst-only-rest
        // is folded into the slot-resolution loop's rest fallback.
        if let (Some(s_rest), Some(d_rest)) = (src_rest, dst_rest)
            && !subtyper.check(s_rest, d_rest)
        {
            return Some(ForwardingMismatchKind::Type);
        }

        // Keywords — Steep parity (match_params, subtyping/check.rb:
        // 1061-1084). Three rules:
        // 1. Every dst *required* keyword must be present among the
        //    src's *required* keywords: an optional src keyword or a
        //    `**rest` may simply not carry the key at runtime, so it
        //    guarantees nothing. Checked before the element types so a
        //    structural gap reports as Arity even when the same key
        //    also mismatches in type (Steep's match_params fails before
        //    any collected pair is type-checked).
        // 2. Every src keyword (required AND optional — Steep's
        //    `flat_keywords`) must be accepted by the dst, via its flat
        //    keywords or its `**rest`, with a covariant element check.
        //    A src `**rest` whose dst declares no `**rest` is accepted:
        //    Steep pairs the keyword rests only when both exist, with
        //    no counterpart to the positional caller-rest-without-
        //    callee-rest arity gate (asymmetry is Steep's, kept as is).
        // 3. When both sides declare `**rest`, the src's rest element
        //    must flow into the dst's.
        let src_rest_kw = src.rest_keyword;
        let dst_rest_kw = dst.rest_keyword;
        for (name, _) in &dst.required_keywords {
            if !src.required_keywords.iter().any(|(n, _)| n == name) {
                return Some(ForwardingMismatchKind::Arity);
            }
        }
        for (name, src_ty) in src
            .required_keywords
            .iter()
            .chain(src.optional_keywords.iter())
        {
            let dst_ty = dst
                .required_keywords
                .iter()
                .chain(dst.optional_keywords.iter())
                .find_map(|(n, t)| (n == name).then_some(*t))
                .or(dst_rest_kw);
            let Some(dst_ty) = dst_ty else {
                return Some(ForwardingMismatchKind::Arity);
            };
            if !subtyper.check(*src_ty, dst_ty) {
                return Some(ForwardingMismatchKind::Type);
            }
        }
        if let (Some(s_rest_kw), Some(d_rest_kw)) = (src_rest_kw, dst_rest_kw)
            && !subtyper.check(s_rest_kw, d_rest_kw)
        {
            return Some(ForwardingMismatchKind::Type);
        }

        None
    }
}

/// Outcome of resolving a `&:sym` block pass via Symbol#to_proc semantics.
/// See [`TypeChecker::symbol_to_proc_return_type`].
pub(super) enum SymbolToProcResolution {
    /// Method resolved; the substituted return type of its zero-arg overload.
    Resolved(Ty),
    /// Method missing on the parameter type (or on these union members).
    NoMethod { missing: Vec<Ty> },
    /// Stay silent and untyped: untyped/variable param, unhandled receiver
    /// kind, or a method that needs arguments.
    Skip,
}

/// If the call's block argument is a `&:sym` literal (a `BlockArgumentNode`
/// wrapping a `SymbolNode`), return the method name and the symbol's start
/// offset. `&var` / `&proc` and brace blocks return `None` — Steep's
/// Symbol#to_proc special case is literal-only.
pub(super) fn block_pass_symbol(block_arg: &Node<'_>) -> Option<(String, usize)> {
    let pass = block_arg.as_block_argument_node()?;
    let expr = pass.expression()?;
    let sym = expr.as_symbol_node()?;
    Some((
        String::from_utf8_lossy(sym.unescaped()).to_string(),
        expr.location().start_offset(),
    ))
}

/// Steep `one_arg?`: the expected block takes exactly one required
/// positional and nothing else. Multi-arg blocks (e.g. `{ (K, V) -> void }`)
/// are outside the Symbol#to_proc special case (Steep logs and skips).
pub(super) fn symbol_to_proc_one_arg_param(expected_block: &Block) -> Option<Ty> {
    match &expected_block.type_ {
        FunctionType::Typed(f)
            if f.required_positionals.len() == 1
                && f.optional_positionals.is_empty()
                && f.rest_positional.is_none()
                && f.trailing_positionals.is_empty()
                && f.required_keywords.is_empty()
                && f.optional_keywords.is_empty()
                && f.rest_keyword.is_none() =>
        {
            Some(f.required_positionals[0])
        }
        _ => None,
    }
}
