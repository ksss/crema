use ruby_prism::{
    CallNode, ConstantReadNode, LocalVariableTargetNode, MultiWriteNode, Node, Visit,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::definition::{ConstantContext, ConstantOrigin};
use crate::definition_builder::{self, type_params_as_variable_args};
use crate::diagnostic::CandidateScope;
use crate::environment::DeclKindLocal;
use crate::name::{Name, Symbol};
use crate::substitution::Substitution;
use crate::subtyping::SubtypeChecker;
use crate::types::{
    self, Function, FunctionType, Literal, RecordKey, Ty, Type, TypeTable, Visibility,
    partition_falsy, partition_truthy, union_of, union_of_many,
};

use super::calls::{LiteralAccessResult, ResolvedCall};
use super::cond_env::{CondEnv, Scrutinee};
use super::visitor::{and_write_value_ty, or_write_value_ty};
use super::{ArgSpan, CallArguments, CallTarget, TypeChecker};
use crate::context::{ScopeKind, ScopeSnapshot};
use crate::pure_call_env::PureKey;

/// One arm's after-state for [`TypeChecker::join_arms_with_divergence`]:
/// the scope snapshot taken at the join point and the body's last
/// expression type. The latter lets the join detect divergence
/// (`last_ty == Ty::BOTTOM`) and drop the arm from the per-name union.
pub(in crate::type_checker) struct BranchOutcome {
    pub after: ScopeSnapshot,
    pub last_ty: Ty,
}

/// Which side's shape is preserved by the shared unify traversal in
/// [`TypeChecker::unify_into_bindings`]. Arg-side unification widens
/// leaf literals/tuples/records to match the subtype check at the
/// call-site boundary; hint-side unification keeps the user-supplied
/// shape so tuple/record bindings survive into downstream inference.
#[derive(Clone, Copy)]
enum UnifyMode {
    WidenLeaves,
    KeepShape,
}

/// A hash-literal key that wasn't declared by the Record hint — surfaced
/// by `synthesize_hash_as_record` so the caller can turn it into an
/// `UnknownRecordKey` diagnostic at the correct source offset.
#[derive(Debug)]
pub(super) struct RecordExtraKey {
    pub key: RecordKey,
    pub known_keys: Vec<String>,
    pub offset: usize,
}

/// Walks each literal element with its positional Tuple hint and
/// interns the result as `Type::Tuple`. The caller is responsible for:
/// (a) deciding the literal is splat-free and arity-matching, (b)
/// resolving the Tuple hint into the element-hint vector. Free-
/// standing so the closure can hold its own (im)mutable borrow of the
/// `TypeChecker`. The Array[E] hint counterpart is inlined at each
/// call site (in `check_array_node` / `infer_type`'s ArrayNode arm) so
/// it can route splat elements through `splat_element_type` and reuse
/// the shared `synthesize_element_union` (widen + dedup + absorb).
fn fold_array_tuple_type<'pr, F>(
    types: &TypeTable,
    tuple_elem_hints: &[Ty],
    elements: &[ruby_prism::Node<'pr>],
    mut infer_elem: F,
) -> Ty
where
    F: FnMut(&ruby_prism::Node<'pr>, Option<Ty>) -> Ty,
{
    let elem_tys: Vec<Ty> = elements
        .iter()
        .zip(tuple_elem_hints.iter())
        .map(|(el, h)| infer_elem(el, Some(*h)))
        .collect();
    types.intern(Type::Tuple(elem_tys))
}

/// Return the index of the literal's sole `SplatNode` element, or `None`
/// when there are zero or more than one. The splat-tuple-expansion path
/// requires a single splat so that scalar positions before/after the
/// splat map deterministically onto hint positions. Multi-splat
/// (`[*xs, *ys]`) is scope outside and falls through to raw synthesis.
fn single_splat_index<'pr>(elements: &[Node<'pr>]) -> Option<usize> {
    let mut found = None;
    for (i, el) in elements.iter().enumerate() {
        if el.as_splat_node().is_some() {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found
}

/// Extract the `AssocNode` / `AssocSplatNode` element list from either a
/// `HashNode` (`{a: 1}`) or a `KeywordHashNode` (the trailing `key: v`
/// inside `[..., key: v]` or `foo(..., key: v)`). The two Prism variants
/// are structurally identical (Steep folds them under one `when :hash, :kwargs`
/// at `type_construction.rb:1466`); this helper lets the hash inference
/// arms accept both without duplicating the synth helpers.
fn hash_or_keyword_hash_elements<'pr>(node: &Node<'pr>) -> Option<Vec<Node<'pr>>> {
    if let Some(h) = node.as_hash_node() {
        Some(h.elements().iter().collect())
    } else {
        node.as_keyword_hash_node()
            .map(|kh| kh.elements().iter().collect())
    }
}

/// Fold a begin expression's normal-completion type and its rescue arm
/// types into the begin's value type. No rescue + no else is a
/// pass-through (preserve the body's exact intern slot rather than round
/// it through `union_of_many`); an `else` clause shadows the body in the
/// normal-completion slot (the body's last value is unreachable once
/// `else` runs); otherwise union the normal-completion type with every
/// rescue arm (`union_of_many` drops Bottom / absorbs untyped).
///
/// Free-standing so both `check_begin_node` (`&mut self`, walks arms via
/// `check_node` for side effects) and `infer_type`'s BeginNode arm
/// (`&self`, walks via `infer_type`) share the union shape. Mirrors the
/// `fold_array_tuple_type` split.
fn begin_value_type(types: &TypeTable, body_ty: Ty, rescue_tys: &[Ty], else_ty: Option<Ty>) -> Ty {
    if rescue_tys.is_empty() && else_ty.is_none() {
        return body_ty;
    }
    let normal_ty = else_ty.unwrap_or(body_ty);
    let mut members = Vec::with_capacity(1 + rescue_tys.len());
    members.push(normal_ty);
    members.extend_from_slice(rescue_tys);
    union_of_many(&members, types)
}

impl<'env> TypeChecker<'env> {
    /// Infer the type of a Prism AST node.
    /// Returns `Ty::UNTYPED` for any node whose type cannot be determined.
    /// Side-effecting expression evaluator that threads `hint` through
    /// the AST as an explicit argument and returns the inferred `Ty`.
    ///
    /// This is the new entry point that supersedes the Visit-trait
    /// `visit_X` callbacks for expression nodes — `Visit::visit_X` has
    /// no return value and no hint parameter, which forced Phase B to
    /// patch the gap with the `pending_call_hint` global slot.
    /// `check_node` carries the hint as an argument so wrappers like
    /// `Array[...]`, `begin; ...; end`, and `cond ? a : b` can forward
    /// it structurally to the value-yielding child(ren).
    ///
    /// Each branch is responsible for: (1) emitting any side-effecting
    /// diagnostics for the node, (2) recursing into children via
    /// `check_node` (NOT `infer_type` and NOT the Visit-trait default
    /// walker) so the hint can be split / shaped per wrapper kind, and
    /// (3) returning the inferred `Ty`. Wrappers that don't accept a
    /// hint (literal / read leaves) delegate to `infer_type`, which is
    /// `&self` and side-effect-free.
    pub(super) fn check_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        match node {
            Node::CallNode { .. } => {
                if let Some(call) = node.as_call_node() {
                    // Pure-call cache fast path. If a structurally
                    // equivalent send was previously typed in the same
                    // lvar scope (and its lvar dependencies have not
                    // been reassigned), reuse the cached / narrowed
                    // type without re-resolving overloads. The cache is
                    // populated by the writeback below; analyze_condition
                    // updates it with the narrowed branch type via
                    // `with_cond_narrowing` / `with_cond_branch`.
                    if let Some(key) = self.try_pure_key(&call.as_node())
                        && let Some(ty) = self
                            .lookup_pure_overlay(&key)
                            .or_else(|| self.ctx.pure_call_env().get(&key))
                    {
                        return ty;
                    }
                    let resolved = ResolvedCall::resolve(self, &call);
                    self.check_call(&call, hint, &resolved);
                    let ret = self.infer_call_return_type(&call, hint, &resolved);
                    self.maybe_cache_pure_call(&call, &resolved, ret);
                    // Extract occurrence, with the type this arm just
                    // computed. Recording happens at the walk entries —
                    // not inside `check_call` — because only the entry
                    // knows whether the check produced a value for the
                    // site (here: yes).
                    if let Some(target) = resolved.target.as_ref() {
                        self.record_extract_call(
                            call.location().start_offset(),
                            call.location().end_offset(),
                            target,
                            resolved.receiver_ty,
                            Some(ret),
                        );
                    } else {
                        self.record_extract_call_no_target(&call, resolved.receiver_ty, Some(ret));
                    }
                    ret
                } else {
                    Ty::UNTYPED
                }
            }
            Node::ArrayNode { .. } => self.check_array_node(node, hint),
            Node::HashNode { .. } | Node::KeywordHashNode { .. } => {
                self.check_hash_node(node, hint)
            }
            Node::RangeNode { .. } => self.check_range_node(node, hint),
            Node::InterpolatedStringNode { .. }
            | Node::InterpolatedSymbolNode { .. }
            | Node::InterpolatedRegularExpressionNode { .. }
            | Node::InterpolatedXStringNode { .. }
            | Node::InterpolatedMatchLastLineNode { .. } => {
                self.check_interpolated_node(node, hint)
            }
            Node::BeginNode { .. } => self.check_begin_node(node, hint),
            Node::StatementsNode { .. } => {
                if let Some(stmts) = node.as_statements_node() {
                    self.check_statements_with_hint(&stmts, hint)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::IfNode { .. } => self.check_if_node(node, hint),
            Node::ElseNode { .. } => self.check_else_node(node, hint),
            Node::OrNode { .. } => self.check_or_node(node, hint),
            Node::AndNode { .. } => self.check_and_node(node, hint),
            Node::UnlessNode { .. } => self.check_unless_node(node, hint),
            Node::CaseNode { .. } => self.check_case_node(node, hint),
            Node::CaseMatchNode { .. } => self.check_case_match_node(node, hint),
            Node::WhileNode { .. } => self.check_while_node(node),
            Node::UntilNode { .. } => self.check_until_node(node),
            Node::ForNode { .. } => self.check_for_node(node, hint),
            Node::RescueModifierNode { .. } => self.check_rescue_modifier_node(node, hint),
            Node::ParenthesesNode { .. } => {
                if let Some(par) = node.as_parentheses_node() {
                    if let Some(body) = par.body() {
                        self.check_node(&body, hint)
                    } else {
                        Ty::NIL
                    }
                } else {
                    Ty::UNTYPED
                }
            }
            // Shorthand hash value (`{a:}`): Prism wraps the desugared
            // read in `ImplicitNode`, whose only field is `value()` — the
            // actual `LocalVariableReadNode` or receiver-less `CallNode`
            // Ruby evaluates. The wrapper has no semantics of its own, so
            // descend with `check_node` (preserving hint) to fire the
            // value's side-effect diagnostics (NoMethod on the inner
            // call, etc.). Without this mirror arm the catch-all routes
            // to `infer_type`, which is read-only and would suppress
            // those diagnostics. `ImplicitRestNode` (`|x,|`) is a splat
            // marker — not transparent — and stays on its own arm.
            Node::ImplicitNode { .. } => node
                .as_implicit_node()
                .map(|i| self.check_node(&i.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `*expr` reached through the evaluator: a `when *EXPR` condition
            // (`check_case_node`) or an array element (`check_array_node`).
            // Descend into the inner expression so its calls / constant reads
            // fire their diagnostics — without this arm the splat falls to
            // `infer_type`'s `_ => UNTYPED` and the subtree is never walked.
            // Call-argument splats route through the Visit walker
            // (`check_call`'s `visit_arguments_node`) instead, so they don't
            // reach here and are not double-emitted. Bare `*` (anonymous rest)
            // has no expression and is a no-op.
            Node::SplatNode { .. } => {
                if let Some(splat) = node.as_splat_node()
                    && let Some(expr) = splat.expression()
                {
                    self.check_node(&expr, None);
                }
                Ty::UNTYPED
            }
            // One-line pattern matches. Without these arms both forms
            // fall to `infer_type`'s `_ => UNTYPED` and the value's
            // calls never fire (path-dependent silence: the Visit
            // walker reached them, `check_node` did not). The value is
            // a plain expression; the pattern side goes through
            // `check_pattern` so captures bind with derived types.
            // `x in pat` evaluates to true/false; `x => pat` is a
            // void-value statement (using it as a value is a
            // SyntaxError), so its type never reaches a value position
            // and UNTYPED is fine.
            Node::MatchPredicateNode { .. } => {
                if let Some(mp) = node.as_match_predicate_node() {
                    let value_ty = self.check_node(&mp.value(), None);
                    self.check_pattern(&mp.pattern(), Some(value_ty));
                }
                Ty::BOOL
            }
            Node::MatchRequiredNode { .. } => {
                if let Some(mr) = node.as_match_required_node() {
                    let value_ty = self.check_node(&mr.value(), None);
                    self.check_pattern(&mr.pattern(), Some(value_ty));
                }
                Ty::UNTYPED
            }
            // `/(?<x>...)/ =~ s` — prism wraps the `=~` CallNode when the
            // regexp literal is the LHS and Ruby binds each named capture
            // as a local. The captures bind UNTYPED (Steep parity: it
            // avoids the `String?` false positives the `if /re/ =~ s`
            // idiom would produce without narrowing); the expression's
            // value is the inner call's return type (`Integer?`).
            Node::MatchWriteNode { .. } => {
                if let Some(mw) = node.as_match_write_node() {
                    let call_ty = self.check_node(&mw.call().as_node(), hint);
                    for target in mw.targets().iter() {
                        if let Some(t) = target.as_local_variable_target_node() {
                            // Already-bound names keep their type (Steep
                            // parity, playground-measured 2026-08-20: after
                            // `x = "hello"; /(?<x>a)/ =~ "a"` Steep hovers
                            // `x: ::String` and still flags `x + 1`).
                            // Only fresh capture names bind UNTYPED.
                            let name_str = String::from_utf8_lossy(t.name().as_slice());
                            let name = self.checker_names().intern(&name_str);
                            if self.lookup_local_variable_for_read(name).is_none() {
                                self.bind_local_variable_target(&t, Ty::UNTYPED);
                            }
                        }
                    }
                    call_ty
                } else {
                    Ty::UNTYPED
                }
            }
            // `class << expr; body; end` — singleton class scope. The
            // supported slice is `class << self` inside a class/module and
            // outside another singleton-class body; other shapes stay silent
            // until their diagnostics todo owns them.
            Node::SingletonClassNode { .. } => {
                if let Some(scn) = node.as_singleton_class_node() {
                    self.check_node(&scn.expression(), None);
                    if !self.ctx.in_singleton_class()
                        && self.ctx.current_class_typename().is_some()
                        && scn.expression().as_self_node().is_some()
                        && let Some(body) = scn.body()
                    {
                        self.ctx.enter_singleton_class();
                        self.check_singleton_class_body(&body);
                        self.ctx.leave_singleton_class();
                    }
                }
                Ty::NIL
            }
            // Declaration / statement nodes — delegate to their
            // Visit-trait handlers so the declarative skeleton
            // (scope push/pop, environment registration, binding
            // updates) runs. The Visit handler's default walker then
            // descends into the RHS / value sub-tree so any call
            // there still fires side effects through `visit_call_node`
            // (the trivial-shim path). Hint is discarded: these nodes
            // don't produce a value the assertion gate cares about.
            Node::ConstantOperatorWriteNode { .. }
            | Node::ConstantPathOperatorWriteNode { .. }
            | Node::ClassVariableOrWriteNode { .. }
            | Node::ClassVariableAndWriteNode { .. } => {
                self.visit(node);
                Ty::UNTYPED
            }
            // `foo[idx] ||= v` / `&&= v` in value position: the
            // truthy/falsy-narrowing value type computed by
            // `check_index_compound_write` (via `check_index_or_write` /
            // `check_index_and_write`), not UNTYPED. Calls the live
            // dispatch directly (mirrors `IndexOperatorWriteNode` below)
            // instead of going through `self.visit(node)`, so the result
            // isn't thrown away.
            Node::IndexOrWriteNode { .. } => node
                .as_index_or_write_node()
                .map(|w| self.check_index_or_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::IndexAndWriteNode { .. } => node
                .as_index_and_write_node()
                .map(|w| self.check_index_and_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // `foo.attr ||= v` / `&&= v` in value position — call-attribute
            // siblings of the Index arms above, same live-dispatch plumbing.
            Node::CallOrWriteNode { .. } => node
                .as_call_or_write_node()
                .map(|w| self.check_call_or_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::CallAndWriteNode { .. } => node
                .as_call_and_write_node()
                .map(|w| self.check_call_and_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // Simple-write subfamily (F1 of the write-node value-type
            // migration): `x = expr` / `@x = expr` / `@@x = expr` /
            // `$x = expr` / `Const = expr` / `Const::Path = expr` in
            // value position. Each `check_*_write` helper is the same
            // side-effecting implementation `visit_*_write_node` already
            // ran (now thin wrappers around these) — this just stops
            // discarding the live-computed value as `Ty::UNTYPED`. Fixes
            // the rescue-tail nil-leak bug for these 6 kinds the same
            // way F2/F3/F5 already fixed it for their subfamilies: the
            // value is read from the *live* (pre-rescue-join) scope at
            // this dispatch point, instead of re-derived later via
            // `infer_type`'s post-widen fallback walk
            // (`check_def_node_in_current_context`, visitor.rs).
            Node::LocalVariableWriteNode { .. } => node
                .as_local_variable_write_node()
                .map(|w| self.check_local_variable_write(&w, hint))
                .unwrap_or(Ty::UNTYPED),
            Node::InstanceVariableWriteNode { .. } => node
                .as_instance_variable_write_node()
                .map(|w| self.check_instance_variable_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::ClassVariableWriteNode { .. } => node
                .as_class_variable_write_node()
                .map(|w| self.check_class_variable_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableWriteNode { .. } => node
                .as_global_variable_write_node()
                .map(|w| self.check_global_variable_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantWriteNode { .. } => node
                .as_constant_write_node()
                .map(|w| self.check_constant_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantPathWriteNode { .. } => node
                .as_constant_path_write_node()
                .map(|w| self.check_constant_path_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // `x ||= v` / `x &&= v` in value position: F2 of the
            // write-node value-type migration (or/and-write subfamily).
            // The value is the RHS type (Steep `:or_asgn`/`:and_asgn`,
            // `type_construction.rb:2395-2428` — no union narrowing with
            // the pre-truthy LHS type, confirmed by ask-steep 2026-07-17;
            // `rebind_compound_lvar`'s doc comment already documented
            // this), computed by side-effect implementations that
            // already exist per kind — plumbing pattern mirrors F3
            // above.
            //
            // `ClassVariableOrWriteNode`/`AndWriteNode` are excluded and
            // stay in the UNTYPED discard cluster above: Steep's
            // `:or_asgn`/`:and_asgn` `case asgn.type` (`type_construction.rb
            // :2399-2426`) only handles `:lvasgn`/`:ivasgn`/`:gvasgn`/
            // `:send` — `:cvasgn` isn't a case there and falls to
            // `fallback_to_any`, i.e. Steep types `@@x ||= v` as `any`,
            // unlike ivar/gvar which do get the RHS type. Verified via
            // `steep check` (2026-07-17): `x = (@@x ||= 1); "s" + x`
            // reports no `ArgumentTypeMismatch` (x is `any`) while the
            // ivar/gvar equivalents do error. `check_compound_cvar_write_
            // rhs` (visitor.rs) still exists for the plain `@@x = rhs`
            // path (F1, unaffected) but is no longer called from this
            // dispatch.
            //
            // `ConstantOrWriteNode`/`ConstantAndWriteNode` and their
            // ConstantPath counterparts stay on the read-only
            // `infer_type` oracle below (their side-effect visit is a
            // deliberate no-op, same rationale as the excluded
            // Constant/ConstantPath operator-write kinds in F3) rather
            // than getting a `check_*` helper of their own.
            Node::LocalVariableOrWriteNode { .. } => node
                .as_local_variable_or_write_node()
                .map(|w| {
                    self.rebind_compound_lvar(w.name().as_slice(), w.depth(), &w.value(), hint)
                })
                .unwrap_or(Ty::UNTYPED),
            Node::LocalVariableAndWriteNode { .. } => node
                .as_local_variable_and_write_node()
                .map(|w| {
                    self.rebind_compound_lvar(w.name().as_slice(), w.depth(), &w.value(), hint)
                })
                .unwrap_or(Ty::UNTYPED),
            Node::InstanceVariableOrWriteNode { .. } => node
                .as_instance_variable_or_write_node()
                .map(|w| {
                    self.check_compound_ivar_write_rhs(
                        w.name().as_slice(),
                        &w.value(),
                        hint,
                        w.location().start_offset(),
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            Node::InstanceVariableAndWriteNode { .. } => node
                .as_instance_variable_and_write_node()
                .map(|w| {
                    self.check_compound_ivar_write_rhs(
                        w.name().as_slice(),
                        &w.value(),
                        hint,
                        w.location().start_offset(),
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableOrWriteNode { .. } => node
                .as_global_variable_or_write_node()
                .map(|w| self.check_global_variable_or_write(&w, hint))
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableAndWriteNode { .. } => node
                .as_global_variable_and_write_node()
                .map(|w| self.check_global_variable_and_write(&w, hint))
                .unwrap_or(Ty::UNTYPED),
            // Read-only oracle path (see the doc comment above the lvar
            // arm for why): `infer_type` synthesizes the RHS with the
            // live (pre-rescue-join) scope still in effect at this
            // dispatch point, so the rescue-tail bug is fixed the same
            // way as the other kinds even though no diagnostic push
            // happens here (matches the existing no-op visit contract).
            // Hint is `None`, not the outer `hint` param: unlike lvar
            // (whose Steep `:lvasgn` arm genuinely forwards the outer
            // hint, since an lvar has no declaration of its own to hint
            // with instead), a bare/path constant's RHS has nothing to
            // do with whatever hint happens to be ambient at the
            // enclosing expression (e.g. the method's declared return
            // type) — forwarding it would leak an unrelated hint into
            // the constant's RHS synthesis (e.g. bidirectionally typing
            // an empty-array-literal RHS from the *caller's* hint).
            Node::ConstantOrWriteNode { .. } => node
                .as_constant_or_write_node()
                .map(|w| self.infer_type(&w.value(), None))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantAndWriteNode { .. } => node
                .as_constant_and_write_node()
                .map(|w| self.infer_type(&w.value(), None))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantPathOrWriteNode { .. } => node
                .as_constant_path_or_write_node()
                .map(|w| self.infer_type(&w.value(), None))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantPathAndWriteNode { .. } => node
                .as_constant_path_and_write_node()
                .map(|w| self.infer_type(&w.value(), None))
                .unwrap_or(Ty::UNTYPED),
            // `foo[idx] += value` in value position: unlike its `||=`/
            // `&&=` siblings above, the whole-expression type here is
            // unambiguous (no truthy-narrowing branch to resolve — see
            // `check_index_operator_write`), so it gets its own arm
            // instead of the discard-and-return-UNTYPED cluster. Calls
            // `check_index_operator_write` directly (mirrors `CallNode`'s
            // `check_call`) rather than going through `self.visit(node)`,
            // so the live dispatch result isn't thrown away.
            Node::IndexOperatorWriteNode { .. } => node
                .as_index_operator_write_node()
                .map(|w| self.check_index_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // `foo.attr += v` in value position — call-attribute sibling
            // of the Index arm above.
            Node::CallOperatorWriteNode { .. } => node
                .as_call_operator_write_node()
                .map(|w| self.check_call_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // `x += rhs` / `@x += rhs` / `@@x += rhs` / `$x += rhs` in
            // value position: F3 of the write-node value-type migration
            // (operator-write subfamily). The value is the operator
            // method's return type (Steep `:op_asgn`,
            // `type_construction.rb:858-908`), computed by side-effect
            // implementations that already exist per kind — plumbing
            // pattern mirrors `IndexOperatorWriteNode` /
            // `MultiWriteNode` above. `ConstantOperatorWriteNode` /
            // `ConstantPathOperatorWriteNode` are excluded: their
            // side-effect dispatch doesn't exist yet (visit is a
            // deliberate no-op, see visitor.rs), so they stay in the
            // UNTYPED discard cluster above (parked to a follow-up todo).
            Node::LocalVariableOperatorWriteNode { .. } => node
                .as_local_variable_operator_write_node()
                .map(|w| self.check_local_variable_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::InstanceVariableOperatorWriteNode { .. } => node
                .as_instance_variable_operator_write_node()
                .map(|w| self.check_instance_variable_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::ClassVariableOperatorWriteNode { .. } => node
                .as_class_variable_operator_write_node()
                .map(|w| self.check_class_variable_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableOperatorWriteNode { .. } => node
                .as_global_variable_operator_write_node()
                .map(|w| self.check_global_variable_operator_write(&w))
                .unwrap_or(Ty::UNTYPED),
            // `a, b = value` in value position: Phase 1 of the write-node
            // value-type migration (multi-write subfamily). Calls
            // `check_multi_write_node` directly (mirrors
            // `IndexOperatorWriteNode` above) instead of going through
            // `self.visit(node)`, so the live dispatch result isn't
            // discarded. Only the literal-array, no-splat, no-assertion,
            // arity-matching shape returns a real tuple type; every other
            // shape still returns `UNTYPED` (Phase 2 scope, see
            // `check_multi_write_node`'s doc comment).
            Node::MultiWriteNode { .. } => node
                .as_multi_write_node()
                .map(|w| self.check_multi_write_node(&w, hint))
                .unwrap_or(Ty::UNTYPED),
            // `def foo; end` / `def self.foo; end` — both forms evaluate
            // to a `Symbol` (the method name). Walk the body via
            // `self.visit(node)` so `visit_def_node` runs (method
            // registration, parameter binding, body diagnostics), and
            // return `Symbol` so `x = def foo; end` binds `x` correctly.
            // Steep parity: `type_construction.rb:1073` / `:1146`.
            Node::DefNode { .. } => {
                self.visit(node);
                self.env
                    .class_instance_type(self.env.names().builtins().symbol)
            }
            // `class C ... end` / `module M ... end` evaluate to `nil`
            // (Steep `type_construction.rb:1556` / `:1599`). Split from
            // the UNTYPED declaration group so the assertion gate
            // observes NIL; `self.visit` still drives the body walker.
            Node::ClassNode { .. } | Node::ModuleNode { .. } => {
                self.visit(node);
                Ty::NIL
            }
            // `alias new old` / `alias $new $old` — Ruby evaluates these
            // to `nil`. Walker descent (`self.visit`) lets a CallNode
            // embedded in an `InterpolatedSymbolNode` new_name fire its
            // own diagnostics; the value position itself yields
            // `Ty::NIL`. Steep mirrors this in `type_construction.rb`
            // (`when :alias → AST::Builtin.nil_type`). Without these
            // arms both variants fall through to `infer_type`'s
            // catch-all and emit `Crema::NotImplementedYet`, which is
            // misleading — these are known statement nodes, not
            // unimplemented surfaces.
            Node::AliasMethodNode { .. } | Node::AliasGlobalVariableNode { .. } => {
                self.visit(node);
                Ty::NIL
            }
            // Unlike the declaration nodes above, `yield` carries a value
            // (the block return type). Run its side-effect checks via
            // `visit` (UnexpectedYield, argument types), then delegate to
            // the read-only `infer_type` arm for the type so the value
            // position (`x = yield`) agrees with the receiver position
            // (`yield.foo`, reached through `infer_receiver_type`).
            Node::YieldNode { .. } => {
                self.visit(node);
                self.infer_type(node, hint)
            }
            // Divergent control-flow leaves: the statement carries no
            // value through to the surrounding expression. Returning
            // `Ty::BOTTOM` lets `check_if_node` / `check_unless_node`
            // (and the case/when path) detect a "this arm cannot reach
            // the join point" condition by checking the body's last_ty,
            // matching Steep's terminal model
            // (`type_construction.rb:1943`). The `visit` call still
            // fires the declared-return-type subtype check for return
            // and any side-effect diagnostics for break/next.
            Node::ReturnNode { .. } | Node::BreakNode { .. } | Node::NextNode { .. } => {
                self.visit(node);
                Ty::BOTTOM
            }
            // Bare constant read reached through the evaluator (a write
            // RHS like `x = Foo`, or a class / def body statement): emit
            // `UnknownConstant` on a miss. The receiver / top-level paths
            // are handled by `visit_constant_read_node` instead; the two
            // are disjoint so a single occurrence emits exactly once.
            Node::ConstantReadNode { .. } => {
                if let Some(constant) = node.as_constant_read_node() {
                    self.check_constant_read(&constant)
                } else {
                    Ty::UNTYPED
                }
            }
            // Constant path read via the evaluator (`x = Foo::Bar`,
            // class/def body): same disjoint-path story as
            // ConstantReadNode — the walker overrides
            // `visit_constant_path_node`, so the two paths never touch
            // the same occurrence.
            Node::ConstantPathNode { .. } => {
                if let Some(path) = node.as_constant_path_node() {
                    self.check_constant_path_node(&path)
                } else {
                    Ty::UNTYPED
                }
            }
            // `super(args)` (SuperNode): full orchestration in
            // `check_super_node` — argument diagnostics, arg subtree visit,
            // block-body typecheck, block subtree visit — mirroring the
            // CallNode arm's `check_call` entry. `infer_type` then projects
            // the overload-selected return type.
            // Bare `super` (ForwardingSuperNode) is not matched here — it
            // needs no argument diagnostic and reaches the read-only return
            // via the `_ => infer_type` delegation below.
            Node::SuperNode { .. } => {
                if let Some(super_node) = node.as_super_node() {
                    // `check_super_node` handles the full orchestration:
                    // arg check, arg subtree visit, block scope + body
                    // check, block subtree visit. Mirrors the CallNode
                    // entry point `check_call`.
                    self.check_super_node(&super_node);
                }
                let ret = self.infer_type(node, hint);
                // Extract occurrence with the type just computed; the
                // super target is re-resolved under the extract gate
                // (the same lookup the check made above — the consulted
                // map dedups, so no new key enters the log).
                self.record_extract_super_at(node, ret, false);
                ret
            }
            // Bare `super` (`ForwardingSuperNode`). The SuperNode arm above
            // emits `Ruby::UnexpectedSuper` via `check_super_node` when the
            // super target does not resolve; this arm is the mirror for the
            // forwarding form. Both paths share the `lookup_super_method`
            // failure point, but the SuperNode arm couples the emission with
            // argument and block checking, while here we only need the
            // diagnostic. Single-path entry guaranteed by
            // `visit_forwarding_super_node` routing through `check_node` (see
            // visitor.rs).
            Node::ForwardingSuperNode { .. } => {
                self.emit_unexpected_super_if_unresolvable(node);
                let ret = self.infer_type(node, hint);
                // `require_rbs_decl` mirrors the emission gate above: a
                // bare super inside an RBS-undeclared def was never
                // recorded (the check skips it), and stays unrecorded.
                self.record_extract_super_at(node, ret, true);
                ret
            }
            // Global variable read: side-effect emit of
            // `Ruby::DeprecatedReference` when the declared gvar carries
            // `%a{deprecated}`, then delegate to `infer_type` for the
            // value type. Write paths are hooked in `visitor.rs`
            // (`visit_global_variable_*_write_node`).
            Node::GlobalVariableReadNode { .. } => {
                if let Some(gvar) = node.as_global_variable_read_node() {
                    let name_str = String::from_utf8_lossy(gvar.name().as_slice()).into_owned();
                    self.check_deprecated_global_ref(
                        &name_str,
                        gvar.location().start_offset(),
                        gvar.location().end_offset(),
                    );
                }
                self.infer_type(node, hint)
            }
            // `-> { ... }` in a value position. Without this arm the
            // catch-all routes to the read-only `infer_type`, whose
            // `lambda_literal_type` only builds the Proc type — every
            // diagnostic inside the body is dropped (path-dependent
            // silence, same shape as the ImplicitNode / SplatNode arms
            // above: the Visit walker reached lambda bodies at statement
            // and argument positions, `check_node` did not).
            // `visit_lambda_node` routes the walker into this arm too, so
            // a lambda is walked exactly once whichever path reaches it.
            Node::LambdaNode { .. } => self.check_lambda_node(node, hint),
            // Literal / read leaves and other read-only nodes: delegate
            // to the side-effect-free `infer_type` synth path. These
            // nodes don't accept a hint structurally and don't recurse
            // into children that need block-diagnostic emission.
            _ => self.infer_type(node, hint),
        }
    }

    pub(super) fn check_singleton_class_body<'pr>(&mut self, body: &Node<'pr>) {
        self.ctx.push_scope(ScopeKind::Method);
        if let Some(stmts) = body.as_statements_node() {
            for stmt in stmts.body().iter() {
                if self.should_check_singleton_class_body_statement(&stmt) {
                    let last_ty = self.check_node(&stmt, None);
                    self.apply_statement_assertion_gate(&stmt, last_ty);
                }
            }
        } else if self.should_check_singleton_class_body_statement(body) {
            let last_ty = self.check_node(body, None);
            self.apply_statement_assertion_gate(body, last_ty);
        }
        self.ctx.pop_scope();
    }

    /// Statement-position trailing `#: T` gate shared by
    /// `check_statements_with_hint` (most body shapes) and
    /// `check_singleton_class_body` (which iterates stmts directly
    /// with its own filter). Eligibility is decided by
    /// `is_statement_assertion_eligible`: assignment forms and
    /// declarations have their own assertion pipelines and are skipped.
    pub(super) fn apply_statement_assertion_gate<'pr>(&mut self, stmt: &Node<'pr>, natural: Ty) {
        if !is_statement_assertion_eligible(stmt) {
            return;
        }
        // Skip receiverless calls that the inline-declaration parser
        // (`inline_parser::collect_attr_call` /
        // `collect_mixin_call`) has already claimed the trailing
        // `#: T` for at load phase. Their `#: T` is a declared method
        // / mixin type, not an expression assertion — Steep parses
        // them as declaration nodes, never as a CallNode + assertion.
        if is_inline_declaration_call_stmt(stmt) {
            return;
        }
        let Some(asserted) = self.lookup_trailing_assertion(
            stmt.location().start_offset(),
            stmt.location().end_offset(),
        ) else {
            return;
        };
        let position = self.offset_to_location(stmt.location().start_offset());
        self.emit_false_assertion_if_incompatible(natural, asserted, position);
    }

    fn should_check_singleton_class_body_statement<'pr>(&self, stmt: &Node<'pr>) -> bool {
        match stmt {
            Node::ClassNode { .. } | Node::ModuleNode { .. } => false,
            Node::DefNode { .. } => stmt
                .as_def_node()
                .is_some_and(|def| def.receiver().is_none()),
            _ => true,
        }
    }

    /// `check_node`'s ArrayNode branch: unwrap the hint via
    /// `array_element_hint` (matches the existing `infer_type` Array
    /// branch shape) and forward it to every element via `check_node`
    /// recursion so element calls fire their side effects.
    fn check_array_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let array = match node.as_array_node() {
            Some(a) => a,
            None => return Ty::UNTYPED,
        };
        let name = self.env.names().builtins().array;
        let elements: Vec<_> = array.elements().iter().collect();
        let has_splat = elements.iter().any(|el| el.as_splat_node().is_some());
        // Tuple hint path: when the literal is splat-free (element-wise
        // hints can't dispatch a splat) and the hint resolves
        // unambiguously to a single Tuple of matching arity, synthesize
        // a `Type::Tuple` so call-site subtype checks see the structural
        // element types instead of the widened `Array[untyped]`.
        if !has_splat
            && let Some(hint_ty) = hint
            && let Some(tuple_elem_hints) = self.array_tuple_hint(hint_ty, elements.len())
        {
            return fold_array_tuple_type(
                self.env.types(),
                &tuple_elem_hints,
                &elements,
                |el, h| self.check_node(el, h),
            );
        }
        // Splat + Tuple hint path: try `try_tuple_type!`-style expansion
        // (Steep parity). Tuple splat gets per-element subtype check
        // (bail on mismatch → `Case C`); Array[T] splat trusts the hint
        // (no check → `Case I`). If expansion succeeds, return a
        // `Type::Tuple`. Every other shape under this gate (arity
        // overflow, multi-splat, decide bail) lands on Array[union] so
        // the outer assertion fires `FalseAssertion` against the Tuple
        // hint instead of getting swallowed by Array[untyped]. The splat
        // expression is walked exactly once on whichever branch wins.
        if has_splat
            && let Some(hint_ty) = hint
            && let Some(tuple_elem_hints) = self.array_tuple_hint_loose(hint_ty)
        {
            if let Some(splat_idx) = single_splat_index(&elements) {
                let scalars_before = splat_idx;
                let scalars_after = elements.len() - splat_idx - 1;
                if scalars_before + scalars_after <= tuple_elem_hints.len()
                    && let Some(splat) = elements[splat_idx].as_splat_node()
                    && let Some(expr) = splat.expression()
                {
                    // Walk in Ruby's left-to-right evaluation order: every
                    // leading scalar fires its diagnostics and binds any
                    // lvar writes before the splat's expression looks
                    // them up. Walking the splat first would let a
                    // pre-write `*xs` lookup race a `(xs = ...)` write
                    // sitting in `leading[i]`.
                    let leading: Vec<Ty> = (0..scalars_before)
                        .map(|i| self.check_node(&elements[i], Some(tuple_elem_hints[i])))
                        .collect();
                    let splat_inner_ty = self.check_node(&expr, None);
                    let splat_slot_count = tuple_elem_hints.len() - scalars_before - scalars_after;
                    let slot_hints: Vec<Ty> = tuple_elem_hints
                        [scalars_before..scalars_before + splat_slot_count]
                        .to_vec();
                    if let Some(splat_slot_types) =
                        self.decide_splat_slot_types(splat_inner_ty, &slot_hints)
                    {
                        let mut out = Vec::with_capacity(tuple_elem_hints.len());
                        out.extend(leading);
                        out.extend(splat_slot_types);
                        for j in 0..scalars_after {
                            let hint_idx = tuple_elem_hints.len() - scalars_after + j;
                            out.push(self.check_node(
                                &elements[splat_idx + 1 + j],
                                Some(tuple_elem_hints[hint_idx]),
                            ));
                        }
                        return self.env.types().intern(Type::Tuple(out));
                    }
                    // Decide bail: leading + splat are already walked;
                    // synthesize Array[union] by walking trailing scalars
                    // (no hint) and reusing the walked types. The hint-
                    // walked leading element types still synthesize the
                    // same union via `widen_literal_to_base` (literal hint
                    // doesn't widen; literal types fold back to their base
                    // class through `synthesize_element_union`).
                    let mut raw = Vec::with_capacity(elements.len());
                    raw.extend(leading);
                    raw.push(self.splat_element_type(splat_inner_ty));
                    for j in 0..scalars_after {
                        raw.push(self.check_node(&elements[splat_idx + 1 + j], None));
                    }
                    let element_ty = self.synthesize_element_union(&raw);
                    return self.env.types().intern(Type::ClassInstance {
                        name,
                        args: vec![element_ty],
                    });
                }
            }
            // Arity-overflow or multi-splat under Tuple hint: walk all
            // elements once via the standard splat-aware raw collection
            // and surface Array[union] so the outer assertion can fire.
            let raw: Vec<Ty> = elements
                .iter()
                .map(|el| match el.as_splat_node() {
                    Some(splat) => match splat.expression() {
                        Some(expr) => {
                            let inner = self.check_node(&expr, None);
                            self.splat_element_type(inner)
                        }
                        None => Ty::UNTYPED,
                    },
                    None => self.check_node(el, None),
                })
                .collect();
            let element_ty = self.synthesize_element_union(&raw);
            return self.env.types().intern(Type::ClassInstance {
                name,
                args: vec![element_ty],
            });
        }
        if let Some(elem_hint) = hint.and_then(|h| self.array_element_hint(h)) {
            // Array[E] hint path. Element-wise walking mirrors the
            // no-hint raw path below — splat elements bypass
            // `check_node`'s SplatNode arm (which returns UNTYPED) and
            // walk their expression directly so `splat_element_type`
            // can extract the element contribution. Scalars carry
            // `Some(elem_hint)` so Stage 1 bidirectional propagation
            // (`[1] #: Array[Integer]` → Integer literal stays Integer)
            // still works through. The collected types pass through
            // `fold_hint_element_union` (widen literals + dedup +
            // flatten one level, **but keep untyped as a Union
            // member**) so the resulting `Array[E']` matches Steep's
            // hover shape (e.g. `Array[(Integer | String)]` for an
            // Integer scalar + Array[String] splat) while still letting
            // the outer subtype gate catch a concrete element under a
            // non-matching hint. Empty literal falls back to the hint
            // itself.
            let element_ty = if elements.is_empty() {
                elem_hint
            } else {
                let raw: Vec<Ty> = elements
                    .iter()
                    .map(|el| match el.as_splat_node() {
                        Some(splat) => match splat.expression() {
                            Some(expr) => {
                                let inner = self.check_node(&expr, None);
                                self.splat_element_type(inner)
                            }
                            None => Ty::UNTYPED,
                        },
                        None => self.check_node(el, Some(elem_hint)),
                    })
                    .collect();
                self.fold_hint_element_union(&raw)
            };
            return self.env.types().intern(Type::ClassInstance {
                name,
                args: vec![element_ty],
            });
        }
        // The hint paths above didn't apply. With genuinely no hint,
        // synthesize the element type from the literal's contents (Steep
        // `try_array_type` parity); a hint that existed but didn't resolve
        // as a tuple / element hint (ambiguous, arity mismatch) stays at
        // the conservative `Array[untyped]`. Either way every element is
        // walked first so element-call side effects (block diagnostics,
        // etc.) fire; `array_from_raw` makes the synthesize / empty
        // decision. Splat elements bypass `check_node`'s SplatNode arm
        // (which would return UNTYPED) and walk their expression directly
        // so `splat_element_type` can extract the element contribution
        // (Tuple → element union, Array[T] → T, other → untyped).
        let raw: Vec<Ty> = elements
            .iter()
            .map(|el| match el.as_splat_node() {
                Some(splat) => match splat.expression() {
                    Some(expr) => {
                        let inner = self.check_node(&expr, None);
                        self.splat_element_type(inner)
                    }
                    None => Ty::UNTYPED,
                },
                None => self.check_node(el, None),
            })
            .collect();
        self.array_from_raw(&raw, hint.is_none())
    }

    /// Widen a literal element type to its base class so a collection
    /// literal yields e.g. `Array[Integer]` rather than `Array[1]`
    /// (Steep parity: collection elements are class-typed; literal types
    /// survive only under a hint). Mirrors the `Type::Literal` arm of the
    /// receiver widening in `calls.rs`. Non-literal types pass through.
    pub(super) fn widen_literal_to_base(&self, ty: Ty) -> Ty {
        match self.env.types().resolve(ty) {
            // `true`/`false` widen to `bool`, not `TrueClass`/`FalseClass`, so
            // a no-hint `[true, false]` synthesizes `Array[bool]` (Steep
            // parity). `class_typename` stays TrueClass/FalseClass for method
            // lookup and subtype paths that need the concrete class.
            Type::Literal(Literal::Bool(_)) => Ty::BOOL,
            Type::Literal(lit) => self
                .env
                .class_instance_type(*lit.class_typename(self.env.names().builtins())),
            _ => ty,
        }
    }

    /// Read-only value type of `foo[idx] ||= v` / `&&=`, matching
    /// `check_index_compound_write`'s Or/And narrowing (`visitor.rs`)
    /// without live dispatch. Resolves the receiver and `[]`'s return type
    /// via `infer_type` + `infer_synthetic_method_return_type` (both
    /// `&self`), then narrows with the same `or_write_value_ty` /
    /// `and_write_value_ty` helpers the visitor path uses — keeping the
    /// two paths' value types in sync. `is_or` selects `||=` (`true`) vs
    /// `&&=` (`false`).
    fn infer_index_compound_write_value_ty<'pr>(
        &self,
        receiver: Option<Node<'pr>>,
        arguments: Option<ruby_prism::ArgumentsNode<'pr>>,
        has_block: bool,
        value: &Node<'pr>,
        is_or: bool,
    ) -> Ty {
        let Some(recv) = receiver else {
            return Ty::UNTYPED;
        };
        let recv_ty = self.infer_type(&recv, None);
        if recv_ty.is_untyped() {
            return Ty::UNTYPED;
        }
        let idx_args = self.collect_arguments_from(arguments, has_block);
        let read_ty = self.infer_synthetic_method_return_type(recv_ty, "[]", idx_args.positional);
        if read_ty.is_untyped() {
            return Ty::UNTYPED;
        }
        let value_ty = self.infer_type(value, None);
        let widened = self.widen_literal_to_base(value_ty);
        if is_or {
            or_write_value_ty(read_ty, widened, self.env.types())
        } else {
            and_write_value_ty(read_ty, widened, self.env.types())
        }
    }

    /// Read-only sibling of `check_call_compound_write` for the
    /// `||=`/`&&=` value-position oracle — mirrors
    /// `infer_index_compound_write_value_ty` minus the index arguments.
    fn infer_call_compound_write_value_ty<'pr>(
        &self,
        receiver: Option<Node<'pr>>,
        is_safe_navigation: bool,
        read_name: &[u8],
        value: &Node<'pr>,
        is_or: bool,
    ) -> Ty {
        let Some(recv) = receiver else {
            return Ty::UNTYPED;
        };
        let recv_ty = self.infer_type(&recv, None);
        if recv_ty.is_untyped() {
            return Ty::UNTYPED;
        }
        let recv_ty = if is_safe_navigation {
            self.unwrap_optional(recv_ty)
        } else {
            recv_ty
        };
        let read_name = String::from_utf8_lossy(read_name).to_string();
        let read_ty = self.infer_synthetic_method_return_type(recv_ty, &read_name, vec![]);
        if read_ty.is_untyped() {
            return Ty::UNTYPED;
        }
        let value_ty = self.infer_type(value, None);
        let widened = self.widen_literal_to_base(value_ty);
        if is_or {
            or_write_value_ty(read_ty, widened, self.env.types())
        } else {
            and_write_value_ty(read_ty, widened, self.env.types())
        }
    }

    /// Fold raw element types into a single collection element type:
    /// widen each literal to its base class, then `union_of_many`
    /// (flatten / absorb `untyped` / drop `Bottom` / dedupe). Shared by
    /// the Array and Hash no-hint synthesis paths.
    fn synthesize_element_union(&self, raw: &[Ty]) -> Ty {
        let widened: Vec<Ty> = raw.iter().map(|&t| self.widen_literal_to_base(t)).collect();
        union_of_many(&widened, self.env.types())
    }

    /// Hint-driven element fold for the Array[E] hint path. Widens
    /// literals, flattens nested Union one level, dedupes by intern
    /// id, sorts, and collapses a single member — but **keeps
    /// `untyped` as a Union member** rather than absorbing it. The
    /// untyped policy is the only difference from
    /// `synthesize_element_union` and the only reason the two paths
    /// cannot share a helper.
    ///
    /// Why the divergence: the no-hint raw path synthesizes the
    /// element type from raw evidence, where untyped absorption is
    /// the correct widening (`untyped | T = untyped` is the gradual-
    /// typing fallback). The hint-driven walk runs upstream of an
    /// outer subtype gate (assertion / kwarg type), and absorbing
    /// untyped would silently mask a concrete-vs-hint mismatch — a
    /// `String + untyped` mix under `Array[Integer]` hint would
    /// collapse to `Array[untyped]` and pass the gate, when the
    /// concrete `String` should still fire `FalseAssertion` /
    /// `ArgumentTypeMismatch`. Keeping untyped as a member preserves
    /// the OLD `fold_array_element_type` soundness (which interned
    /// `Type::Union(...)` directly without absorption) while adding
    /// widening + dedup that the OLD path lacked.
    fn fold_hint_element_union(&self, raw: &[Ty]) -> Ty {
        let mut members: Vec<Ty> = Vec::with_capacity(raw.len());
        let mut seen: FxHashSet<Ty> = FxHashSet::default();
        for &t in raw {
            let widened = self.widen_literal_to_base(t);
            match self.env.types().resolve(widened) {
                Type::Union(inner) => {
                    for &m in inner {
                        if seen.insert(m) {
                            members.push(m);
                        }
                    }
                }
                _ => {
                    if seen.insert(widened) {
                        members.push(widened);
                    }
                }
            }
        }
        members.sort();
        match members.len() {
            0 => Ty::BOTTOM,
            1 => members[0],
            _ => self.env.types().intern(Type::Union(members)),
        }
    }

    /// Build `Array[E]` from raw (already-walked) element types. When
    /// `no_hint` is set and the literal is non-empty, its widened element
    /// union becomes `Array[E]`; otherwise — a hint was present but
    /// unusable, or the literal is empty — it keeps `Array[untyped]`
    /// (the element type can't be read statically). The decision lives
    /// here so the `check_node` and `infer_type` no-hint Array paths
    /// can't drift apart.
    ///
    /// Splat entries in `raw` are already collapsed by the caller via
    /// [`splat_element_type`] — a `Tuple` splat contributes its element
    /// union, an `Array[T]` splat contributes `T`, a `Union` splat
    /// decomposes member-wise and `union_of_many` collapses the result
    /// (Steep parity with `flatten_array_elements`,
    /// `type_construction.rb:4916-4923`), and any other operand goes
    /// through `try_convert_to_a` to unfold `Range[E]` / `Enumerator[E]`
    /// / `Hash[K, V]` etc. into their `to_a` return-type element. When
    /// `to_a` is absent the operand passes through unchanged so a
    /// scalar `Integer` splat contributes `Integer`. This keeps
    /// `array_from_raw` agnostic to whether a splat was present.
    fn array_from_raw(&self, raw: &[Ty], no_hint: bool) -> Ty {
        let name = self.env.names().builtins().array;
        let args = if no_hint && !raw.is_empty() {
            vec![self.synthesize_element_union(raw)]
        } else {
            vec![Ty::UNTYPED]
        };
        self.env.types().intern(Type::ClassInstance { name, args })
    }

    /// Element-type contribution of a splat inside an array literal.
    /// Used by the no-hint synthesis path so `[scalar, *splat, scalar]`
    /// can fold into `Array[union]` instead of bailing to
    /// `Array[untyped]`. `Type::Tuple` flattens to the element union,
    /// `Type::ClassInstance { ::Array, [T] }` contributes `T`,
    /// `Type::Tuple([])` contributes `Bottom` (dropped by
    /// `union_of_many` so an empty splat doesn't poison the synthesis),
    /// `Type::Union(_)` recurses into each member and `union_of_many`
    /// collapses the results — Steep parity with `flatten_array_elements`
    /// (`type_construction.rb:4916-4923`): an `Array[T] | B` operand
    /// yields `T | B`. The recursion is defensive against nested
    /// Unions; in practice `push_flat` in `union_of` keeps live unions
    /// flat, so nested Unions don't surface here. Every other shape
    /// goes through `try_convert_to_a`: if the operand has a `to_a`
    /// method whose
    /// params are all optional (Steep parity with
    /// `type_construction.rb:5106-5108` and `try_convert` at
    /// `interface/function.rb:948`), its return type is unfolded one
    /// step — `Array[E] -> E`, `Tuple -> element union`, otherwise
    /// passed through as a scalar. Scalars without `to_a` (`Integer`,
    /// `untyped`, `Bottom`, user-defined types) keep the original
    /// pass-through (`splat_inner_ty`), matching Steep's
    /// `flatten_array_elements` fallback for non-Array operands.
    pub(super) fn splat_element_type(&self, splat_inner_ty: Ty) -> Ty {
        match self.env.types().resolve(splat_inner_ty) {
            Type::Tuple(elems) => {
                if elems.is_empty() {
                    Ty::BOTTOM
                } else {
                    union_of_many(elems, self.env.types())
                }
            }
            Type::ClassInstance { name, args }
                if *name == self.env.names().builtins().array && args.len() == 1 =>
            {
                args[0]
            }
            Type::Union(members) => {
                // Outer Union: each member is unfolded by a fresh
                // `splat_element_type` call (so an `Array[T]` member
                // contributes `T`, a `Range[T]` member goes through
                // `try_convert_to_a` once, a scalar passes through).
                // Results are folded by `splat_union_fold` which
                // **preserves `untyped` as a member** — `union_of_many`
                // would absorb it and silently mask a concrete-vs-hint
                // mismatch downstream in `fold_hint_element_union`.
                let inner: Vec<Ty> = members
                    .iter()
                    .map(|&m| self.splat_element_type(m))
                    .collect();
                self.splat_union_fold(&inner)
            }
            _ => {
                if let Some(converted) = self.try_convert_to_a(splat_inner_ty) {
                    // Inline unfold (Steep parity with `try_array_type`
                    // lines 5110-5118: Array -> args[0], Tuple ->
                    // element union, else -> converted as scalar). We
                    // do not recurse into `splat_element_type` to keep
                    // the to_a resolution depth bounded to one step.
                    self.splat_unfold_one_step(converted)
                } else {
                    splat_inner_ty
                }
            }
        }
    }

    /// One-step unfold of a `to_a` return type into a splat element type.
    /// Steep parity with `flatten_array_elements` (`type_construction.rb:
    /// 4916-4923`) running once after `try_convert`: Array/Tuple are
    /// opened, Union members are opened in-place (no re-entry into
    /// `try_convert_to_a` — that would risk unbounded recursion when
    /// `to_a` returns a Union containing the receiver type), and every
    /// other shape passes through as a scalar.
    fn splat_unfold_one_step(&self, ty: Ty) -> Ty {
        match self.env.types().resolve(ty) {
            Type::Tuple(elems) => {
                if elems.is_empty() {
                    Ty::BOTTOM
                } else {
                    union_of_many(elems, self.env.types())
                }
            }
            Type::ClassInstance { name, args }
                if *name == self.env.names().builtins().array && args.len() == 1 =>
            {
                args[0]
            }
            Type::Union(members) => {
                let inner: Vec<Ty> = members
                    .iter()
                    .map(|&m| match self.env.types().resolve(m) {
                        Type::Tuple(elems) => {
                            if elems.is_empty() {
                                Ty::BOTTOM
                            } else {
                                union_of_many(elems, self.env.types())
                            }
                        }
                        Type::ClassInstance { name, args }
                            if *name == self.env.names().builtins().array && args.len() == 1 =>
                        {
                            args[0]
                        }
                        _ => m,
                    })
                    .collect();
                self.splat_union_fold(&inner)
            }
            _ => ty,
        }
    }

    /// Fold splat-element members into a single Ty. 1-step flatten of
    /// any nested Union, dedupe by intern id, sort, and collapse — but
    /// **never absorb `untyped`**. The caller's outer fold
    /// (`synthesize_element_union` for the no-hint path,
    /// `fold_hint_element_union` for the hint-driven path) decides the
    /// `untyped` policy. If `union_of_many` ran here a single `untyped`
    /// member would erase concrete siblings, masking
    /// `FalseAssertion` / `ArgumentTypeMismatch` downstream.
    fn splat_union_fold(&self, members: &[Ty]) -> Ty {
        let mut out: Vec<Ty> = Vec::with_capacity(members.len());
        let mut seen: FxHashSet<Ty> = FxHashSet::default();
        for &m in members {
            match self.env.types().resolve(m) {
                Type::Union(inner) => {
                    for &t in inner {
                        if seen.insert(t) {
                            out.push(t);
                        }
                    }
                }
                _ => {
                    if seen.insert(m) {
                        out.push(m);
                    }
                }
            }
        }
        out.sort();
        match out.len() {
            0 => Ty::BOTTOM,
            1 => out[0],
            _ => self.env.types().intern(Type::Union(out)),
        }
    }

    /// Resolve the `to_a` method on `ty` and return its
    /// param-binding-applied return type, when an arity-zero overload
    /// exists. Mirrors Steep's `try_convert(type, :to_a)` at
    /// `type_construction.rb:5019-5029`, with the overload-select
    /// condition `params.nil? || params.optional?` (an UntypedFunction
    /// also passes, matching Steep's `params.nil?` arm).
    ///
    /// Returns `None` when the receiver has no `to_a`, when no overload
    /// is arity-zero, or when the receiver shape is not handled by
    /// `lookup_method` (e.g. `Type::Union`, `Type::Untyped`). Callers
    /// fall back to scalar pass-through in those cases.
    pub(super) fn try_convert_to_a(&self, ty: Ty) -> Option<Ty> {
        let to_a = self.env.names().intern_symbol("to_a");
        let resolved = super::method_resolver::lookup_method(self.env, ty, to_a)?;
        // Steep `try_convert` uses `calculate_interface(private: false)`
        // (`type_construction.rb:5019`); a private `to_a` must not act
        // as an implicit conversion. Symmetric with
        // `try_convert_to_ary_return_type` in visitor.rs.
        if resolved.method.accessibility == Visibility::Private {
            return None;
        }

        let arg_free = resolved.method.defs.iter().find(|td| {
            let mt = &td.type_;
            mt.is_untyped_function()
                || (mt.required_positionals().is_empty()
                    && mt.required_keywords().is_empty()
                    && mt.trailing_positionals().is_empty())
        })?;

        let return_ty = arg_free.type_.return_type();
        let subst =
            crate::definition_builder::substitution_for_receiver(self.env, ty, resolved.bindings);
        Some(subst.apply(return_ty, self.env.types()))
    }

    /// Build `Hash[K, V]`. Single source for the Hash `ClassInstance`
    /// shape so `hash_untyped_untyped` and the no-hint synthesis path
    /// agree on the name source (`builtins().hash`).
    fn hash_instance(&self, key_ty: Ty, value_ty: Ty) -> Ty {
        let name = self.env.names().builtins().hash;
        self.env.types().intern(Type::ClassInstance {
            name,
            args: vec![key_ty, value_ty],
        })
    }

    /// `check_node`'s HashNode branch: walk every key / value via
    /// `check_node` so calls nested inside the literal fire their
    /// block / argument diagnostics, then delegate to `infer_type` for
    /// the type-query half (record-hint synthesis is already there
    /// and is read-only). Avoids duplicating the record-pick logic.
    fn check_hash_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        if let Some(elements) = hash_or_keyword_hash_elements(node) {
            for elem in elements.iter() {
                if let Some(assoc) = elem.as_assoc_node() {
                    self.check_node(&assoc.key(), None);
                    self.check_node(&assoc.value(), None);
                } else if let Some(splat) = elem.as_assoc_splat_node()
                    && let Some(value) = splat.value()
                {
                    self.check_node(&value, None);
                }
            }
        }
        self.infer_type(node, hint)
    }

    /// `check_node`'s RangeNode branch: walk each present bound via
    /// `check_node` so a call in `(foo()..bar())` fires its diagnostics,
    /// then delegate to `infer_type` for the `Range[E]` type. Mirrors
    /// `check_hash_node` — without this arm RangeNode falls to
    /// `infer_type` (read-only), and a bound's side effects are walked
    /// only on the top-level Visit-walker path, not in `check_node`
    /// contexts like an assignment RHS (`r = foo()..4`).
    fn check_range_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        if let Some(range) = node.as_range_node() {
            if let Some(left) = range.left() {
                self.check_node(&left, None);
            }
            if let Some(right) = range.right() {
                self.check_node(&right, None);
            }
        }
        self.infer_type(node, hint)
    }

    /// `check_node`'s interpolated-literal branch (string / symbol /
    /// regexp / xstring / match-last-line): walk each `#{...}`
    /// EmbeddedStatementsNode so a call inside the interpolation fires
    /// its diagnostics, then delegate to `infer_type` for the literal's
    /// class type. Mirrors `check_range_node` — without this arm an
    /// interpolated node falls to `infer_type` (read-only) and the
    /// embedded statements are walked only on the top-level Visit-walker
    /// path, not in `check_node` contexts (assignment RHS, def / class
    /// body statement, `if`-condition). The EmbeddedVariableNode form
    /// (`#@x`) holds only a variable read (no call position), so it has
    /// no side effects to fire and is skipped.
    fn check_interpolated_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let parts = match node {
            Node::InterpolatedStringNode { .. } => {
                node.as_interpolated_string_node().map(|n| n.parts())
            }
            Node::InterpolatedSymbolNode { .. } => {
                node.as_interpolated_symbol_node().map(|n| n.parts())
            }
            Node::InterpolatedRegularExpressionNode { .. } => node
                .as_interpolated_regular_expression_node()
                .map(|n| n.parts()),
            Node::InterpolatedXStringNode { .. } => {
                node.as_interpolated_x_string_node().map(|n| n.parts())
            }
            Node::InterpolatedMatchLastLineNode { .. } => node
                .as_interpolated_match_last_line_node()
                .map(|n| n.parts()),
            _ => None,
        };
        if let Some(parts) = parts {
            for part in parts.iter() {
                if let Some(embedded) = part.as_embedded_statements_node()
                    && let Some(stmts) = embedded.statements()
                {
                    self.check_node(&stmts.as_node(), None);
                }
            }
        }
        self.infer_type(node, hint)
    }

    /// `check_node`'s BeginNode branch: forward the hint to the main
    /// statements block so the last expression sees it, then walk the
    /// `rescue` / `else` / `ensure` clauses so their bodies fire
    /// side-effect diagnostics. The body and the rescue / else bodies
    /// are value positions (the begin's result is whichever ran), so
    /// they receive the hint; `ensure` does not contribute the value
    /// and runs with `None`.
    ///
    /// Env join wiring follows Steep `type_construction.rb:2106-2226`:
    ///   1. body evaluates from `pre_base` in place
    ///   2. rescue arms start from `join(pre_base, post_body)` — an
    ///      exception can fire at any point inside the body, so neither
    ///      "nothing ran" nor "everything ran" is a sound starting env
    ///   3. else evaluates from `post_body` (only reached when the body
    ///      completed cleanly)
    ///   4. post-begin env = chain-join of (else-or-body) and every
    ///      rescue arm
    ///   5. ensure body is checked but its env writes are discarded
    ///
    /// Result type follows Steep `type_construction.rb:2185-2196`:
    ///   - else present: `union(else_ty, *rescue_arm_tys)` — body is
    ///     dropped (the body's last value is shadowed by else)
    ///   - else absent + rescue arms present: `union(body_ty, *rescue_arm_tys)`
    ///   - no rescue + no else: `body_ty` unchanged
    ///
    /// `union_of_many` auto-drops `Ty::BOTTOM`, so a `raise`-only body
    /// or arm contributes nothing to the union (Steep's "resbody Bot
    /// filter" is satisfied for free).
    fn check_begin_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let begin = match node.as_begin_node() {
            Some(b) => b,
            None => return Ty::UNTYPED,
        };

        let pre_base = self.ctx.snapshot_scopes();

        let body_ty = if let Some(stmts) = begin.statements() {
            self.check_statements_with_hint(&stmts, hint)
        } else {
            Ty::NIL
        };
        let post_body = self.ctx.snapshot_scopes();

        // rescue arms run against `join(pre_base, post_body)` to model
        // exceptions raised mid-body.
        self.ctx.join_branches(&pre_base, &post_body, self.env);
        let rescue_start = self.ctx.snapshot_scopes();

        let mut rescue_afters: Vec<crate::context::ScopeSnapshot> = Vec::new();
        let mut rescue_tys: Vec<Ty> = Vec::new();
        let mut rescue = begin.rescue_clause();
        while let Some(clause) = rescue {
            let (arm, arm_ty) = self.with_cond_branch(&rescue_start, &CondEnv::empty(), |c| {
                c.check_rescue_clause(&clause, hint)
            });
            // Mirror Steep's `resbody_pairs.select!` (type_construction.rb:2175):
            // a bot-typed arm cannot complete normally, so its env never
            // reaches post-begin. Steep filters type and env in one pass;
            // crema drops bot from the type side via `union_of_many`'s
            // auto-drop and from the env side via this guard.
            if arm_ty != Ty::BOTTOM {
                rescue_afters.push(arm);
            }
            rescue_tys.push(arm_ty);
            rescue = clause.subsequent();
        }

        // `with_cond_branch` evaluates the body against the live scope
        // (not against its `base` argument), so we rewind live to
        // `post_body` first. Otherwise else would run under the
        // rescue-start join env left behind by the last rescue arm.
        let (else_after, else_ty) = if let Some(else_clause) = begin.else_clause() {
            self.ctx.restore_scopes(post_body.clone());
            let (arm, ty) = self.with_cond_branch(&post_body, &CondEnv::empty(), |c| {
                c.check_node(&else_clause.as_node(), hint)
            });
            (Some(arm), Some(ty))
        } else {
            (None, None)
        };

        // Chain-join the normal-completion env (else-after if present,
        // otherwise post_body) with every rescue arm. `acc` walks the
        // live scope as we join each arm in turn.
        let mut acc = else_after.unwrap_or(post_body);
        for arm in rescue_afters {
            self.ctx.restore_scopes(acc.clone());
            self.ctx.join_branches(&acc, &arm, self.env);
            acc = self.ctx.snapshot_scopes();
        }
        self.ctx.restore_scopes(acc);

        // ensure body's check runs but its env writes are discarded
        // (Steep's `for_branch(node)` semantics, confirmed 2026-06-02).
        if let Some(ensure_clause) = begin.ensure_clause()
            && let Some(stmts) = ensure_clause.statements()
        {
            let env_before_ensure = self.ctx.snapshot_scopes();
            self.check_statements_with_hint(&stmts, None);
            self.ctx.restore_scopes(env_before_ensure);
        }

        begin_value_type(self.env.types(), body_ty, &rescue_tys, else_ty)
    }

    /// Walk a single `rescue` clause: the matched exception classes and
    /// the bound reference (`rescue E => err`) are non-value positions
    /// (`None`), while the handler body is a value position and takes
    /// the begin's hint. The `subsequent` chain is driven by the caller.
    /// Returns the arm's value type: the body's last expression type, or
    /// `Ty::NIL` when the rescue body is empty (Steep `:resbody` 2210).
    fn check_rescue_clause<'pr>(
        &mut self,
        clause: &ruby_prism::RescueNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        let mut exception_tys: Vec<Ty> = Vec::new();
        for exc in clause.exceptions().iter() {
            exception_tys.push(self.check_node(&exc, None));
        }
        if let Some(reference) = clause.reference() {
            // `rescue A => e` binds `e` to the instance type of the listed
            // exception classes (union for several), Steep parity
            // (`type_construction.rb:2144-2165`). Only lvar targets narrow;
            // ivar / gvar / etc. keep the generic visit (Steep treats
            // non-lvasgn as no assignment).
            if let Some(target) = reference.as_local_variable_target_node() {
                let bound = self.rescue_reference_type(&exception_tys);
                self.bind_local_variable_target(&target, bound);
            } else {
                self.visit(&reference);
            }
        }
        if let Some(stmts) = clause.statements() {
            self.check_statements_with_hint(&stmts, hint)
        } else {
            Ty::NIL
        }
    }

    /// The type bound to `e` in `rescue A, B => e`, from the already-
    /// checked exception-expression types. Each `ClassSingleton` maps to
    /// its instance type (Steep's `to_instance_type`); anything else —
    /// dynamic expressions, splats — contributes untyped, which the
    /// union then absorbs. Bare `rescue => e` diverges from Steep
    /// (untyped) on purpose: Ruby's bare rescue captures StandardError,
    /// so crema binds `::StandardError` — falling back to untyped only
    /// when the environment has no such declaration (minimal preludes).
    fn rescue_reference_type(&mut self, exception_tys: &[Ty]) -> Ty {
        if exception_tys.is_empty() {
            let tn = self.env.names().parse_type_name("::StandardError");
            return match self.env.declared_type_name_by_type_name(tn) {
                Some(tn) => self.env.class_instance_type(tn),
                None => Ty::UNTYPED,
            };
        }
        let instance_tys: Vec<Ty> = exception_tys
            .iter()
            .map(|&ty| match self.env.types().resolve(ty) {
                Type::ClassSingleton { name } => self.env.class_instance_type(*name),
                _ => Ty::UNTYPED,
            })
            .collect();
        crate::types::union_of_many(&instance_tys, self.env.types())
    }

    /// `check_node`'s UnlessNode branch: mirror `check_if_node`.
    /// Predicate is a non-value position; the body and the `else` clause
    /// are value positions and receive the caller's hint.
    pub(super) fn check_unless_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let unless = match node.as_unless_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let predicate = unless.predicate();
        let stmts = unless.statements();
        let else_clause = unless.else_clause();
        self.check_node(&predicate, None);
        // `unless` is `if` with branches swapped — feed falsy to the body
        // and truthy to the else clause.
        let (truthy, falsy) = self.analyze_condition(&predicate);

        let base = self.ctx.snapshot_scopes();
        let arm_body = self.with_branch_outcome(&base, &falsy, |checker| {
            if let Some(s) = &stmts {
                checker.check_statements_with_hint(s, hint)
            } else {
                Ty::NIL
            }
        });
        let arm_else = if let Some(else_clause) = else_clause {
            self.with_branch_outcome(&base, &truthy, |checker| {
                checker.check_node(&else_clause.as_node(), hint)
            })
        } else {
            // Mirror `check_if_node`'s no-else branch: even without an
            // explicit `else`, the silent arm still runs under truthy
            // narrowing so a divergent `unless` body (`unless x; raise;
            // end`) can adopt `x: String` into the post-unless env.
            self.with_branch_outcome(&base, &truthy, |_| Ty::NIL)
        };
        self.join_arms_with_divergence(base, arm_body, arm_else)
    }

    /// `check_node`'s CaseNode branch (`case x; when ...; end`).
    /// Predicate and every `when` condition are non-value positions;
    /// each `when` body and the `else` clause are value positions and
    /// receive the caller's hint.
    ///
    /// Narrows the scrutinee across `when` bodies when the predicate is a
    /// bare local variable and each `when` cond resolves to a Class/Module
    /// literal (ADR-0022). The narrow accumulates: each subsequent `when`
    /// sees the falsy residue of all prior whens; the `else` clause runs
    /// under the final residue. An unresolvable `when` cond forfeits the
    /// *residue* for the rest of the chain (we can't tell what it
    /// consumed), but every later arm still narrows to its own class /
    /// literal target: the frozen residue is a superset of what actually
    /// reaches that arm, so narrowing against it stays sound.
    pub(super) fn check_case_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let case = match node.as_case_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };

        // Predicate scrutinee: either a bare local variable (the
        // original ADR-0022 case path) or a pure-call expression. The
        // helper `scrutinee_and_current_ty` falls back to a fresh
        // infer_type on cache miss, so `case c.phone; when String`
        // works even before any cache entry has been written.
        let scrutinee: Option<(Scrutinee, Ty)> = case.predicate().and_then(|predicate| {
            self.check_node(&predicate, None);
            self.scrutinee_and_current_ty(&predicate)
        });

        let base = self.ctx.snapshot_scopes();

        let mut iter_base = base.clone();
        let scrutinee_full_ty: Option<Ty> = scrutinee.as_ref().map(|(_, t)| *t);
        let mut iter_falsy_ty: Option<Ty> = scrutinee_full_ty;
        // Only the residue tracking is forfeited by an underivable cond;
        // each arm still derives its own target (see the doc comment).
        let mut residue_alive = scrutinee.is_some();
        let mut arms: Vec<BranchOutcome> = Vec::new();

        for cond in case.conditions().iter() {
            let when_node = match cond.as_when_node() {
                Some(w) => w,
                None => continue,
            };
            let when_conds: Vec<Node<'pr>> = when_node.conditions().iter().collect();

            for c in &when_conds {
                self.check_node(c, None);
            }

            let target_union: Option<Ty> = if scrutinee.is_some() {
                let mut targets: Vec<Ty> = Vec::with_capacity(when_conds.len());
                let mut ok = true;
                for c in &when_conds {
                    if let Some(t) = self.case_when_literal_target(c) {
                        targets.push(t);
                        continue;
                    }
                    match self.case_when_class_target(c) {
                        Some(t) => targets.push(t),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && !targets.is_empty() {
                    Some(union_of_many(&targets, self.env.types()))
                } else {
                    None
                }
            } else {
                None
            };

            let predicate_less_condition_narrow = if scrutinee.is_none() {
                self.predicate_less_case_condition_narrow(&when_conds)
            } else {
                None
            };

            let truthy = match (
                scrutinee.as_ref(),
                target_union,
                &predicate_less_condition_narrow,
            ) {
                (Some((s, _)), Some(target), _) => {
                    let truthy_ty =
                        crate::narrowing::narrow(iter_falsy_ty.unwrap(), target, self.env);
                    CondEnv::single(s.clone(), truthy_ty)
                }
                (_, _, Some((truthy, _))) => truthy.clone(),
                _ => CondEnv::empty(),
            };

            let outcome = self.with_branch_outcome(&iter_base, &truthy, |c| {
                if let Some(stmts) = when_node.statements() {
                    c.check_statements_with_hint(&stmts, hint)
                } else {
                    Ty::NIL
                }
            });
            arms.push(outcome);

            if let (Some((s, _)), Some(target), true) =
                (scrutinee.as_ref(), target_union, residue_alive)
            {
                let next_falsy =
                    crate::narrowing::subtract(iter_falsy_ty.unwrap(), target, self.env);
                iter_falsy_ty = Some(next_falsy);
                self.ctx.restore_scopes(base.clone());
                match s {
                    Scrutinee::Lvar(name) => self.ctx.set_local_variable(*name, next_falsy),
                    Scrutinee::Pure(key) => {
                        self.ctx.pure_call_env_mut().set(key.clone(), next_falsy)
                    }
                }
                iter_base = self.ctx.snapshot_scopes();
            } else if let Some((_, falsy)) = predicate_less_condition_narrow {
                let (after, _) = self.with_cond_branch(&iter_base, &falsy, |_| Ty::NIL);
                iter_base = after;
                self.ctx.restore_scopes(iter_base.clone());
            } else if scrutinee.is_some() && target_union.is_none() {
                residue_alive = false;
            }
        }

        let else_clause_opt = case.else_clause();

        // Single source of truth for "the case has provably covered
        // its scrutinee's type". Requires `residue_alive` (an
        // unresolvable when cond forfeits the claim)
        // and a `BOTTOM` residue from the cumulative subtract.
        // Drives both the exhaustiveness diagnostics (ADR-0022) and
        // the implicit / explicit fallthrough skip in the join below.
        let case_exhausted =
            residue_alive && iter_falsy_ty.map(|t| t == Ty::BOTTOM).unwrap_or(false);

        // Exhaustiveness diagnostics (ADR-0022 § exhaustiveness). Only
        // emit when residue_alive — if any `when` cond was unresolvable,
        // we have no basis to claim what is or isn't covered.
        if residue_alive {
            use crate::diagnostic::{Diagnostic, DiagnosticKind};
            let residue = iter_falsy_ty.expect("residue_alive implies iter_falsy_ty is Some");
            if !case_exhausted && else_clause_opt.is_none() {
                let position = self.offset_to_location(node.location().start_offset());
                let scrutinee_str = self.display_type(
                    scrutinee_full_ty.expect("residue_alive implies scrutinee is Some"),
                );
                let residue_str = self.display_type(residue);
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::NonExhaustiveCase {
                        scrutinee_type: scrutinee_str,
                        residue_type: residue_str,
                    },
                });
            }
        }

        // Walk the `else` body when present for diagnostic side effects,
        // but skip the arm when the branch is unreachable (Steep /
        // mypy alignment — unreachable assignments must not pollute the
        // post-case env). Symmetrically, when the case is exhaustive and
        // there is no explicit `else`, the implicit `else; nil` arm is
        // also unreachable and must not be joined — otherwise a lvar
        // first-assigned in every reachable `when` would widen to
        // `T | nil` post-case (the "unrun assignment reads as nil" path
        // in `Context::join_branches`).
        if let Some(else_clause) = else_clause_opt {
            let outcome = self.with_branch_outcome(&iter_base, &CondEnv::empty(), |c| {
                c.check_node(&else_clause.as_node(), hint)
            });
            if case_exhausted && outcome.last_ty != Ty::BOTTOM {
                use crate::diagnostic::{Diagnostic, DiagnosticKind};
                let position =
                    self.offset_to_location(else_clause.as_node().location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::UnreachableValueBranch,
                });
            }
            if !case_exhausted {
                arms.push(outcome);
            }
        } else if !case_exhausted {
            // Implicit `else; nil` arm. Carries the post-narrow base so
            // the join still sees the iter_base env, with last_ty = NIL
            // so it never counts as divergent.
            arms.push(BranchOutcome {
                after: iter_base.clone(),
                last_ty: Ty::NIL,
            });
        }

        self.join_n_arms_with_divergence(base, arms)
    }

    /// Walk a pattern-position subtree (`in` patterns, one-line `in` /
    /// `=>` patterns). Two jobs: (1) reach every embedded expression
    /// (pinned expressions, pattern constants) so their diagnostics
    /// fire, and (2) bind captures. A capture binds to the type derived
    /// from its pattern (`pattern_target_ty`), a bare lvar target binds
    /// to the scrutinee's type when the caller knows it, and everything
    /// underivable binds to UNTYPED explicitly — bound but unknown, so
    /// later reads never trip a false NoMethod. Element decomposition
    /// (`deconstruct` / `deconstruct_keys` typing) is out of scope:
    /// captures nested in Array/Hash/Find patterns get no scrutinee
    /// type (`None`), only their own pattern-derived type.
    fn check_pattern<'pr>(&mut self, node: &Node<'pr>, scrutinee: Option<Ty>) {
        if let Some(cap) = node.as_capture_pattern_node() {
            let value = cap.value();
            self.check_pattern(&value, None);
            let ty = self.pattern_target_ty(&value).unwrap_or(Ty::UNTYPED);
            self.bind_local_variable_target(&cap.target(), ty);
        } else if let Some(target) = node.as_local_variable_target_node() {
            self.bind_local_variable_target(&target, scrutinee.unwrap_or(Ty::UNTYPED));
        } else if let Some(alt) = node.as_alternation_pattern_node() {
            self.check_pattern(&alt.left(), None);
            self.check_pattern(&alt.right(), None);
        } else if let Some(arr) = node.as_array_pattern_node() {
            if let Some(c) = arr.constant() {
                self.visit(&c);
            }
            for e in arr.requireds().iter() {
                self.check_pattern(&e, None);
            }
            if let Some(rest) = arr.rest() {
                self.check_pattern(&rest, None);
            }
            for e in arr.posts().iter() {
                self.check_pattern(&e, None);
            }
        } else if let Some(fp) = node.as_find_pattern_node() {
            if let Some(c) = fp.constant() {
                self.visit(&c);
            }
            self.check_pattern(&fp.left().as_node(), None);
            for e in fp.requireds().iter() {
                self.check_pattern(&e, None);
            }
            self.check_pattern(&fp.right(), None);
        } else if let Some(hp) = node.as_hash_pattern_node() {
            if let Some(c) = hp.constant() {
                self.visit(&c);
            }
            for e in hp.elements().iter() {
                if let Some(assoc) = e.as_assoc_node() {
                    self.check_pattern(&assoc.value(), None);
                } else {
                    self.check_pattern(&e, None);
                }
            }
            if let Some(rest) = hp.rest() {
                self.check_pattern(&rest, None);
            }
        } else if let Some(splat) = node.as_splat_node() {
            if let Some(expr) = splat.expression() {
                self.check_pattern(&expr, None);
            }
        } else if let Some(pin) = node.as_pinned_expression_node() {
            self.check_node(&pin.expression(), None);
        } else {
            // Constants, literals, pinned variables, ranges: the
            // default walker reaches their reads and fires diagnostics
            // (e.g. UnknownConstant) through the existing Visit hooks.
            self.visit(node);
        }
    }

    /// Derive the type a pattern guarantees for a capture bound through
    /// it. Reuses the case/when target helpers: class/module constants
    /// resolve to their instance type, static-equality literals to
    /// their literal type. An alternation derives the union of its
    /// branches, or nothing if any branch is underivable. `None` means
    /// the capture must stay UNTYPED (value constants, Array/Hash
    /// patterns, ranges, …) — guessing here would turn legitimate
    /// later calls into false NoMethods.
    fn pattern_target_ty<'pr>(&self, node: &Node<'pr>) -> Option<Ty> {
        // `in Integer => n` guarantees the same type as `in Integer`
        // for both the capture and the scrutinee.
        if let Some(cap) = node.as_capture_pattern_node() {
            return self.pattern_target_ty(&cap.value());
        }
        if let Some(alt) = node.as_alternation_pattern_node() {
            let l = self.pattern_target_ty(&alt.left())?;
            let r = self.pattern_target_ty(&alt.right())?;
            return Some(crate::types::union_of(l, r, self.env.types()));
        }
        self.case_when_class_target(node)
            .or_else(|| self.case_when_literal_target(node))
    }

    /// Split an `in` clause's pattern into the pattern proper and its
    /// guard. Prism wraps a guarded pattern in an `IfNode` /
    /// `UnlessNode` whose `statements` hold the real pattern and whose
    /// `predicate` is the guard expression, so every caller that wants
    /// the pattern must unwrap first.
    fn in_pattern_and_guard<'pr>(&self, pattern: Node<'pr>) -> (Node<'pr>, Option<Node<'pr>>) {
        if let Some(if_node) = pattern.as_if_node()
            && let Some(inner) = if_node.statements().and_then(|s| s.body().iter().last())
        {
            return (inner, Some(if_node.predicate()));
        }
        if let Some(unless) = pattern.as_unless_node()
            && let Some(inner) = unless.statements().and_then(|s| s.body().iter().last())
        {
            return (inner, Some(unless.predicate()));
        }
        (pattern, None)
    }

    /// `check_node`'s CaseMatchNode branch (`case x; in pat; end`).
    /// Patterns are walked via `check_pattern` so a constant used in a
    /// pattern still fires and captures bind with derived types; each
    /// `in` body and the `else` clause are value positions and receive
    /// the hint.
    ///
    /// Narrowing mirrors `check_case_node` (ADR-0022): the scrutinee is
    /// narrowed to each arm's pattern-derived type inside that arm, and
    /// the falsy residue accumulates into the following arms and the
    /// `else`. An underivable pattern (Array/Hash/Find, value constant)
    /// forfeits the residue for the rest of the chain, but later arms
    /// still narrow to their own pattern type against the frozen
    /// residue, same as `check_case_node`. A
    /// guarded arm narrows its own body but never subtracts — a failing
    /// guard falls through, so its pattern type still reaches later
    /// arms. Unlike case/when, no exhaustiveness diagnostics are emitted
    /// (2026-08-21 user decision): the residue exists only to drive
    /// narrowing and the value-type join.
    pub(super) fn check_case_match_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let case = match node.as_case_match_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        // Two reads of the predicate with different jobs: `predicate_ty`
        // is the value type a bare lvar pattern (`in n`) binds to and is
        // defined for any expression, while `scrutinee` is the narrow
        // target and only exists for lvar / pure-call shapes. An untyped
        // scrutinee narrows too: a derivable pattern is the same
        // `Module#===` dispatch as case/when, which already narrows
        // untyped (`narrow(untyped, target) = target`), and `subtract`
        // passes untyped through so later arms and `else` keep it.
        let predicate_ty = case.predicate().map(|p| self.check_node(&p, None));
        let scrutinee: Option<(Scrutinee, Ty)> = case
            .predicate()
            .and_then(|p| self.scrutinee_and_current_ty(&p));

        let base = self.ctx.snapshot_scopes();
        let mut iter_base = base.clone();
        let mut iter_falsy_ty: Option<Ty> = scrutinee.as_ref().map(|(_, t)| *t);
        let mut residue_alive = scrutinee.is_some();
        let mut arms: Vec<BranchOutcome> = Vec::new();

        for cond in case.conditions().iter() {
            let in_node = match cond.as_in_node() {
                Some(n) => n,
                None => continue,
            };
            let (pattern, guard) = self.in_pattern_and_guard(in_node.pattern());
            let target = if scrutinee.is_some() {
                self.pattern_target_ty(&pattern)
            } else {
                None
            };
            let truthy_ty = target.map(|t| {
                crate::narrowing::narrow(iter_falsy_ty.expect("target implies ty"), t, self.env)
            });
            let cond_env = match (scrutinee.as_ref(), truthy_ty) {
                (Some((s, _)), Some(ty)) => CondEnv::single(s.clone(), ty),
                _ => CondEnv::empty(),
            };
            let outcome = self.with_branch_outcome(&iter_base, &cond_env, |c| {
                c.check_pattern(&pattern, truthy_ty.or(predicate_ty));
                if let Some(g) = &guard {
                    c.check_node(g, None);
                }
                match in_node.statements() {
                    Some(stmts) => c.check_statements_with_hint(&stmts, hint),
                    None => Ty::NIL,
                }
            });
            arms.push(outcome);

            match (scrutinee.as_ref(), target) {
                // A guarded arm may fail its guard and fall through, so
                // its pattern type must stay in the residue.
                (Some(_), Some(_)) if guard.is_some() || !residue_alive => {}
                (Some((s, _)), Some(t)) => {
                    let next = crate::narrowing::subtract(
                        iter_falsy_ty.expect("target implies ty"),
                        t,
                        self.env,
                    );
                    iter_falsy_ty = Some(next);
                    self.ctx.restore_scopes(base.clone());
                    match s {
                        Scrutinee::Lvar(name) => self.ctx.set_local_variable(*name, next),
                        Scrutinee::Pure(key) => self.ctx.pure_call_env_mut().set(key.clone(), next),
                    }
                    iter_base = self.ctx.snapshot_scopes();
                }
                (Some(_), None) => residue_alive = false,
                (None, _) => {}
            }
        }
        if let Some(else_clause) = case.else_clause() {
            let outcome = self.with_branch_outcome(&iter_base, &CondEnv::empty(), |c| {
                c.check_node(&else_clause.as_node(), hint)
            });
            arms.push(outcome);
        }
        // No implicit base arm without `else`: an unmatched case/in
        // raises NoMatchingPatternError, so post-case code only runs
        // after some arm matched. The join makes a capture missing
        // from one arm read as `T | nil` (join_branches), matching
        // Ruby's "unrun binding reads as nil".
        self.join_n_arms_with_divergence(base, arms)
    }

    /// `check_node`'s WhileNode branch. A `while` loop evaluates to
    /// `nil`, so the body is a non-value position and runs with `None`.
    ///
    /// Post-loop env joins the body-after env with the exit env (the
    /// base scope with cond's falsy narrowing applied), mirroring
    /// Steep's `:while` arm in `type_construction.rb`. The body is
    /// evaluated once with no fixed-point, so writes from a second
    /// iteration onward are not propagated; this matches the precision
    /// Steep gets without `pin_local_variables`'s `enforced_type`,
    /// which crema lacks.
    ///
    /// The do-while form (`begin; ...; end while cond`) reaches the
    /// same WhileNode variant with `is_begin_modifier()` set. There the
    /// body runs once before cond is ever evaluated, so cond's
    /// narrowing must not flow into the body. That post-condition path
    /// is out of scope for this todo, so fall back to the pre-narrowing
    /// behavior (no narrowing, no join) which keeps the modifier form
    /// observationally unchanged from before this todo.
    fn check_while_node<'pr>(&mut self, node: &Node<'pr>) -> Ty {
        let while_node = match node.as_while_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let predicate = while_node.predicate();
        let stmts = while_node.statements();
        if while_node.is_begin_modifier() {
            self.check_node(&predicate, None);
            if let Some(s) = stmts {
                self.check_statements_with_hint(&s, None);
            }
            // do-while form (`begin; ...; end while cond`) also evaluates
            // to nil at runtime; align with the non-modifier path so
            // direct-receiver shapes like `(begin; ...; end while c).foo`
            // surface NilClass NoMethod uniformly.
            return Ty::NIL;
        }
        self.check_node(&predicate, None);
        let (truthy, falsy) = self.analyze_condition(&predicate);

        let base = self.ctx.snapshot_scopes();
        if let Some(s) = stmts {
            // Route through join_arms_with_divergence so a divergent
            // body (`while ...; return; end`) drops out of the post-loop
            // env join — same terminal-model treatment as if/unless.
            // The exit arm always reaches the join, so its last_ty stays
            // a concrete `Ty::NIL`; we never want (true, true) here.
            let body = self
                .with_branch_outcome(&base, &truthy, |c| c.check_statements_with_hint(&s, None));
            let exit = self.with_branch_outcome(&base, &falsy, |_| Ty::NIL);
            self.join_arms_with_divergence(base, body, exit);
        } else {
            // No body: pick the falsy-narrow env directly. Joining with
            // `base` would wash out the narrowing — e.g. `while x.nil?;
            // end` would leave `x` as `String | nil` instead of
            // `String`. Mirrors Steep's no-body branch on the same arm.
            let (exit_after, _) = self.with_cond_branch(&base, &falsy, |_| {});
            self.ctx.restore_scopes(exit_after);
        }
        // Ruby runtime: `while ... end` always evaluates to nil
        // (only `break value` can change this; tracked separately).
        Ty::NIL
    }

    /// `check_node`'s UntilNode branch. Symmetric to `check_while_node`
    /// with truthy/falsy swapped: `until` runs the body while the
    /// predicate is falsy and exits when it turns truthy. The do-until
    /// modifier (`begin; ...; end until cond`) is handled the same way
    /// as do-while: out of scope, fall back to the pre-narrowing path.
    fn check_until_node<'pr>(&mut self, node: &Node<'pr>) -> Ty {
        let until_node = match node.as_until_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let predicate = until_node.predicate();
        let stmts = until_node.statements();
        if until_node.is_begin_modifier() {
            self.check_node(&predicate, None);
            if let Some(s) = stmts {
                self.check_statements_with_hint(&s, None);
            }
            // do-until form: same nil-collapse as do-while above.
            return Ty::NIL;
        }
        self.check_node(&predicate, None);
        let (truthy, falsy) = self.analyze_condition(&predicate);

        let base = self.ctx.snapshot_scopes();
        if let Some(s) = stmts {
            // Mirror `check_while_node`'s divergence-aware join with
            // truthy/falsy swapped: `until` runs the body while falsy
            // and exits when truthy turns. A divergent body still drops
            // out of the post-loop env.
            let body =
                self.with_branch_outcome(&base, &falsy, |c| c.check_statements_with_hint(&s, None));
            let exit = self.with_branch_outcome(&base, &truthy, |_| Ty::NIL);
            self.join_arms_with_divergence(base, body, exit);
        } else {
            let (exit_after, _) = self.with_cond_branch(&base, &truthy, |_| {});
            self.ctx.restore_scopes(exit_after);
        }
        // Ruby runtime: `until ... end` always evaluates to nil.
        Ty::NIL
    }

    /// `check_node`'s ForNode branch. A `for` loop evaluates to its
    /// collection (`(for i in xs; end)` returns `xs`), so the collection
    /// is the value position and receives the caller's hint. The index
    /// is a write target walked via the default walker; the body is a
    /// non-value position and runs with `None`.
    fn check_for_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let for_node = match node.as_for_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        self.visit(&for_node.index());
        let collection_ty = self.check_node(&for_node.collection(), hint);
        if let Some(stmts) = for_node.statements() {
            self.check_statements_with_hint(&stmts, None);
        }
        collection_ty
    }

    /// `check_node`'s RescueModifierNode branch (`x rescue y`). Both the
    /// expression and the rescue expression are value positions (the
    /// result is whichever ran), so both receive the caller's hint.
    ///
    /// Value type is `T(body) | T(rescue_expression)`, Steep parity
    /// (`type_construction.rb:2195` `union_type(body_pair, *resbody_types)`,
    /// the `else`-less branch). Env handling mirrors the `:rescue` arm
    /// at `type_construction.rb:2106`:
    ///   1. Walk the body, capturing the post-body env.
    ///   2. Rescue runs against `join(pre_body, post_body)` — an
    ///      exception can raise at any point inside the body, so the
    ///      rescue arm must observe an env that's compatible with both
    ///      the pre-body state and any writes the body completed.
    ///   3. Post-modifier env is `join(post_body, rescue_after)` when
    ///      the rescue arm completes normally; a `BOTTOM` rescue arm
    ///      (e.g. `x rescue raise(...)`) drops out and the live env
    ///      reverts to `post_body` alone (Steep `:resbody` Bot filter,
    ///      `type_construction.rb:2175`). The shared `check_begin_node`
    ///      implements the same flow for the full begin/rescue form.
    fn check_rescue_modifier_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let rescue = match node.as_rescue_modifier_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };

        let pre_base = self.ctx.snapshot_scopes();
        let body_ty = self.check_node(&rescue.expression(), hint);
        let post_body = self.ctx.snapshot_scopes();

        // Rescue base = join(pre_body, post_body).
        self.ctx.join_branches(&pre_base, &post_body, self.env);
        let rescue_start = self.ctx.snapshot_scopes();

        let (rescue_after, rescue_ty) =
            self.with_cond_branch(&rescue_start, &CondEnv::empty(), |c| {
                c.check_node(&rescue.rescue_expression(), hint)
            });

        // Post-modifier env: post_body alone if rescue diverged,
        // otherwise join(post_body, rescue_after).
        if rescue_ty == Ty::BOTTOM {
            self.ctx.restore_scopes(post_body);
        } else {
            self.ctx.restore_scopes(post_body.clone());
            self.ctx.join_branches(&post_body, &rescue_after, self.env);
        }

        union_of(body_ty, rescue_ty, self.env.types())
    }

    /// Walk every statement: middle statements run with `hint=None`,
    /// only the last receives the caller's hint. This is the structural
    /// fix for the bug where `infer_type`'s StatementsNode arm passed
    /// `None` to the last expression, silently dropping the hint for
    /// any `begin ... end #: T` chain.
    fn check_statements_with_hint<'pr>(
        &mut self,
        stmts: &ruby_prism::StatementsNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        let body: Vec<_> = stmts.body().iter().collect();
        if body.is_empty() {
            return Ty::NIL;
        }
        // Consume the suppress flag (set by
        // `check_def_node_in_current_context` to hand off the def-body
        // last assertion to `check_return_type`). Nested
        // `check_statements_with_hint` calls inside this body's last
        // stmt see `false`, so they still apply the gate normally
        // (e.g. `def f; (begin; x #: T; end); end` — the inner Begin's
        // last stmt fires).
        let suppress_last = std::mem::replace(&mut self.suppress_method_body_last_assertion, false);
        // Sibling consume of the parens-routed flag (set by
        // `visit_local_variable_write_node` to hand off the lvasgn
        // trailing `#: T` so the Parens inner gate doesn't re-emit a
        // duplicate `FalseAssertion`). Cleared here so nested
        // StatementsNodes inside the suppressed last stmt (e.g. the
        // inner Begin's body) see `false` and run their gate normally.
        let suppress_parens_routed =
            std::mem::replace(&mut self.suppress_parens_routed_last_assertion, false);
        let last_idx = body.len() - 1;
        let mut last_ty = Ty::NIL;
        for (i, stmt) in body.iter().enumerate() {
            let is_last = i == last_idx;
            let stmt_hint = if is_last { hint } else { None };
            last_ty = self.check_node(stmt, stmt_hint);
            // Statement-position trailing `#: T` gate. Steep parity
            // (`type_construction.rb` `:assertion`): any value-position
            // expression at statement position is an assertion site, not
            // just lvar / multi assignment. Assignment forms and
            // declarations are handled by their own pipelines and are
            // filtered out via `is_statement_assertion_eligible`, so
            // `z = x #: String` fires exactly once (via
            // `visit_local_variable_write_node`) and `def foo #: T`
            // never reaches this gate.
            if !(is_last && (suppress_last || suppress_parens_routed)) {
                self.apply_statement_assertion_gate(stmt, last_ty);
            }
            // Divergence cutoff. Mirrors Steep's TypeConstruction#synthesize
            // (`type_construction.rb:1218` `unless break_type.is_a?(AST::Types::Bot)`
            // and parallel sites): once a stmt evaluates to `Ty::BOTTOM`
            // (return/raise/break/next, or any enclosing form whose arms
            // all diverge), the remaining stmts are dead code. Walking
            // them would emit false diagnostics under the pre-divergence
            // scope - the narrowing dropped by the divergent arm never
            // gets a chance to run if we keep evaluating dead stmts in
            // base env.
            if last_ty == Ty::BOTTOM {
                break;
            }
        }
        last_ty
    }

    /// `check_node`'s IfNode branch (covers both `if` statements and
    /// ternary `cond ? a : b`): predicate runs with `hint=None`, both
    /// branches receive the caller's hint so an annotation on the
    /// whole conditional reaches both alternatives.
    ///
    /// Overall value type is the union of the surviving arms' last_ty
    /// (BOTTOM arms drop via `join_arms_with_divergence` → `union_of`).
    /// else-less `if`/ternary contribute an implicit `Ty::NIL` arm so
    /// `if cond then 1 end` evaluates to `1 | nil`. Steep models the
    /// same shape but base-widens literals first; crema preserves
    /// literals (see `test_assertion_narrowing_from_optional_surfaces_
    /// with_literal_union` for the visible difference).
    pub(super) fn check_if_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let if_node = match node.as_if_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let predicate = if_node.predicate();
        let stmts = if_node.statements();
        let subsequent = if_node.subsequent();
        self.check_node(&predicate, None);
        let (truthy, falsy) = self.analyze_condition(&predicate);

        let base = self.ctx.snapshot_scopes();
        let arm_a = self.with_branch_outcome(&base, &truthy, |checker| {
            if let Some(s) = &stmts {
                checker.check_statements_with_hint(s, hint)
            } else {
                Ty::NIL
            }
        });
        let arm_b = if let Some(sub) = subsequent {
            self.with_branch_outcome(&base, &falsy, |checker| checker.check_node(&sub, hint))
        } else {
            // No `else` clause — the unwritten arm carries an implicit
            // `nil` value (Steep's unrun-arm join semantics). Crucially,
            // it still runs under the *falsy* narrowing so a divergent
            // truthy arm (`if x.nil?; return; end`) can adopt this
            // env's `x: String` into the post-if scope chain.
            self.with_branch_outcome(&base, &falsy, |_| Ty::NIL)
        };
        self.join_arms_with_divergence(base, arm_a, arm_b)
    }

    /// Two-arm join with divergence-aware fallthrough. Mirrors Steep's
    /// terminal model (`type_construction.rb:1943`): when an arm's
    /// body type is `Ty::BOTTOM`, that arm cannot reach the join point
    /// (it `return`s, `raise`s, `break`s, or `next`s), so the other
    /// arm's env is adopted wholesale into the post-if scope chain.
    /// When both arms diverge the if/unless itself has type `BOTTOM`
    /// and the base env is restored — outer statements walking past
    /// this expression observe `BOTTOM` as their last_ty and propagate
    /// the divergence one level up (this is how nested guard clauses
    /// are transparent without extra logic).
    ///
    /// Value type: the surviving arms' `last_ty` are unioned. A single
    /// live arm passes its type through unchanged; `union_of` handles
    /// the `untyped` and BOTTOM absorption (e.g. `untyped | T → untyped`).
    pub(super) fn join_arms_with_divergence(
        &mut self,
        base: ScopeSnapshot,
        arm_a: BranchOutcome,
        arm_b: BranchOutcome,
    ) -> Ty {
        match (arm_a.last_ty == Ty::BOTTOM, arm_b.last_ty == Ty::BOTTOM) {
            (true, true) => {
                self.ctx.restore_scopes(base);
                Ty::BOTTOM
            }
            (true, false) => {
                self.ctx.restore_scopes(arm_b.after);
                arm_b.last_ty
            }
            (false, true) => {
                self.ctx.restore_scopes(arm_a.after);
                arm_a.last_ty
            }
            (false, false) => {
                self.ctx.join_branches(&arm_a.after, &arm_b.after, self.env);
                union_of(arm_a.last_ty, arm_b.last_ty, self.env.types())
            }
        }
    }

    /// N-arm divergence join for `case/when` and any future arm-set
    /// form. Mirrors Steep's `branch_results.reject! { is_a?(Bot) }`
    /// from both `type_construction.rb:2091` (predicate-less case) and
    /// `case_when.rb:246` (predicate-bearing case): drop arms whose
    /// body diverged, then join the survivors' env.
    ///
    /// - All arms diverged: rewind to `base` and surface `Ty::BOTTOM`
    ///   so the surrounding statements-list cuts off at this expression.
    /// - One or more live arms: fold their `after` scopes through
    ///   `Context::join_branches`, restore the merged env, return the
    ///   union of the live arms' `last_ty`. `union_of_many` handles
    ///   the `untyped` and dedup normalization.
    ///
    /// The 2-arm `join_arms_with_divergence` is kept as a separate
    /// helper because its name advertises the truthy/falsy pair shape
    /// that `if`/`unless`/`while`/`until` rely on; migrating those
    /// call sites to this N-arm form is intentionally out of scope.
    pub(super) fn join_n_arms_with_divergence(
        &mut self,
        base: ScopeSnapshot,
        arms: Vec<BranchOutcome>,
    ) -> Ty {
        let live: Vec<BranchOutcome> = arms
            .into_iter()
            .filter(|a| a.last_ty != Ty::BOTTOM)
            .collect();
        if live.is_empty() {
            self.ctx.restore_scopes(base);
            return Ty::BOTTOM;
        }
        let mut iter = live.into_iter();
        let first = iter.next().expect("live is non-empty");
        let mut acc = first.after;
        let mut tys: Vec<Ty> = vec![first.last_ty];
        for arm in iter {
            self.ctx.restore_scopes(base.clone());
            self.ctx.join_branches(&acc, &arm.after, self.env);
            acc = self.ctx.snapshot_scopes();
            tys.push(arm.last_ty);
        }
        self.ctx.restore_scopes(acc);
        union_of_many(&tys, self.env.types())
    }

    /// Read the narrowing carried by `node` viewed as a boolean
    /// predicate. Returns `(truthy, falsy)` — each side is a `CondEnv`
    /// (possibly empty) that names the locals narrowed on that branch.
    ///
    /// Current scope:
    /// - bare `LocalVariableReadNode` — both sides via
    ///   `narrowing::{narrow, subtract}` against the `nil | false`
    ///   target. Bottom results collapse to `CondEnv::empty` so callers
    ///   never see "narrow to unreachable" when the SubtypeChecker
    ///   simply couldn't partition the current type.
    /// - `LocalVariableWriteNode` — `unless x = expr; ...` style. The
    ///   write is routed through `visit_local_variable_write_node` by
    ///   `check_node` before this function runs, so the lvar is already
    ///   bound to the RHS type and the narrowing is identical to the
    ///   bare-read case. RHS-recursive narrowing (`unless x = y.nil?`
    ///   narrowing `y`, per Steep's `logic_type_interpreter.rb:105`)
    ///   is intentionally not implemented; the scrutinee is `x` alone.
    /// - `x.nil?` — both sides via `narrowing::{narrow, subtract}`
    /// - `x.is_a?(C)` / `x.kind_of?(C)` — both sides against the
    ///   instance type derived from a Class/Module literal arg
    /// - `C === x` — same, with the literal on the receiver side
    /// - `x == <literal>` — same, with the literal on the argument
    ///   side; gated on `x`'s `==` resolving to a builtin value class
    ///   (`eq_resolves_to_builtin_value_class`)
    /// - `a && b` / `a || b` — `CondEnv::merge_sequential` and
    ///   `CondEnv::join_branch` over the operands' envs (short-circuit
    ///   evaluation: rhs runs under lhs's truthy env for `&&`, falsy for
    ///   `||`)
    /// - `!a` — receiver's `(truthy, falsy)` swapped
    ///
    /// All other predicates return `(empty, empty)` and behave
    /// identically to the pre-narrowing checker.
    pub(super) fn analyze_condition<'pr>(&mut self, node: &Node<'pr>) -> (CondEnv, CondEnv) {
        if let Some(par) = node.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    return self.analyze_condition(&last);
                }
                return (CondEnv::empty(), CondEnv::empty());
            }
            return self.analyze_condition(&body);
        }
        if let Some(lvar) = node.as_local_variable_read_node() {
            return self.lvar_bare_narrow_by_name(lvar.name().as_slice());
        }
        if let Some(write) = node.as_local_variable_write_node() {
            // Prism's lexical resolution guarantees the same scope handles
            // both this write (via `set_local_variable_at_depth`) and any
            // later read of `x`, so name-only lookup matches the binding
            // the surrounding statement will actually see — depth need not
            // be threaded through.
            return self.lvar_bare_narrow_by_name(write.name().as_slice());
        }
        if let Some(mp) = node.as_match_predicate_node() {
            // `x in pat` — same membership split as `is_a?` / `===`,
            // with the target coming from the pattern instead of an
            // argument. Underivable patterns fall through to no narrow.
            let value = mp.value();
            if let Some((scrutinee, current)) = self.scrutinee_and_current_ty(&value)
                && !current.is_untyped()
                && let Some(target) = self.pattern_target_ty(&mp.pattern())
            {
                return self.narrow_pair(scrutinee, current, target);
            }
            return (CondEnv::empty(), CondEnv::empty());
        }
        if let Some(write) = node.as_multi_write_node()
            && let Some(truthy) = self.multi_write_truthy_env(&write)
        {
            return (truthy, CondEnv::empty());
        }
        if let Some(or) = node.as_or_node() {
            // `a || b` — dual of &&. Truthy is lt joined with (lf merged
            // with rt-under-lf); falsy is sequential merge of (lf, rf
            // under lf). Right operand runs under lf because Ruby short-
            // circuits `||` — `b` only evaluates when `a` is falsy.
            let (lt, lf) = self.analyze_condition(&or.left());
            let (rt, rf) = self.with_cond_narrowing(&lf, |c| c.analyze_condition(&or.right()));
            let truthy = lt.join_branch(lf.clone().merge_sequential(rt), self.env);
            let falsy = lf.merge_sequential(rf);
            return (truthy, falsy);
        }
        if let Some(and) = node.as_and_node() {
            // `a && b` — truthy is sequential merge of (lt, rt under lt);
            // falsy is left-falsy joined with (lt merged with right-falsy
            // under lt). The right operand runs under lt because Ruby
            // short-circuits — `a && b` only evaluates `b` when `a` is
            // truthy, so `b`'s analysis sees `lt` already in scope.
            let (lt, lf) = self.analyze_condition(&and.left());
            let (rt, rf) = self.with_cond_narrowing(&lt, |c| c.analyze_condition(&and.right()));
            let truthy = lt.clone().merge_sequential(rt);
            let falsy = lf.join_branch(lt.merge_sequential(rf), self.env);
            return (truthy, falsy);
        }
        if let Some(call) = node.as_call_node() {
            // `!x` — flip the receiver's narrowing (swap truthy/falsy).
            // Shape gate (no args, no block, has receiver) intentionally
            // allows user-defined `!` to narrow; method-name + shape only,
            // matching the nil? / is_a? precedent.
            if call.name().as_slice() == b"!"
                && call.arguments().is_none()
                && call.block().is_none()
                && let Some(recv) = call.receiver()
            {
                let (t, f) = self.analyze_condition(&recv);
                return (f, t);
            }
            if let Some((scrutinee, current)) = self.nil_predicate_target(&call) {
                return self.narrow_pair(scrutinee, current, Ty::NIL);
            }
            if let Some((scrutinee, current, target_ty)) = self.is_a_predicate_target(&call) {
                return self.narrow_pair(scrutinee, current, target_ty);
            }
            if let Some((scrutinee, current, target_ty)) = self.case_eq_predicate_target(&call) {
                return self.narrow_pair(scrutinee, current, target_ty);
            }
            if let Some((scrutinee, current, target_ty)) = self.eq_literal_predicate_target(&call) {
                return self.narrow_pair(scrutinee, current, target_ty);
            }
            // Union-receiver predicate call (`if union_recv.pred?`) whose
            // per-member return type diverges into truthy/falsy. Runs after
            // the named predicates above so `nil?`/`is_a?`/`===`/`==` keep
            // first dispatch on a union receiver. This narrows the
            // *receiver*; it's independent of (and merged with, below) the
            // bare pure-call arm, which narrows the *call's own return
            // value* for a pure predicate on a union receiver (e.g. an
            // attr-reader-style `tag` whose members don't diverge in
            // truthiness but whose own nilable return still benefits from
            // narrowing on repeat reads). Returning `union_result` alone
            // whenever it hit anything used to silently drop the bare-call
            // narrow for the untouched side — see the 2026-07-16 crema-review
            // finding this comment refers to.
            let union_result = self.union_member_predicate_env(&call);
            // Bare pure-call scrutinee (`if c.phone; ...`). Falls back to
            // the truthy/falsy split against `nil | false`, mirroring the
            // bare-lvar arm above but routing the narrow through the
            // pure-call cache instead of an lvar scope binding.
            let bare_result = if let Some((scrutinee, current)) =
                self.scrutinee_and_current_ty(&call.as_node())
                && matches!(scrutinee, Scrutinee::Pure(_))
                && !current.is_untyped()
            {
                Some(self.bare_truthy_falsy_env(scrutinee, current))
            } else {
                None
            };
            match (union_result, bare_result) {
                (Some((ut, uf)), Some((bt, bf))) => {
                    return (ut.merge_sequential(bt), uf.merge_sequential(bf));
                }
                (Some(result), None) | (None, Some(result)) => return result,
                (None, None) => {}
            }
        }
        (CondEnv::empty(), CondEnv::empty())
    }

    fn predicate_less_case_condition_can_narrow<'pr>(&self, node: &Node<'pr>) -> bool {
        if let Some(par) = node.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    return self.predicate_less_case_condition_can_narrow(&last);
                }
                return false;
            }
            return self.predicate_less_case_condition_can_narrow(&body);
        }
        if node.as_local_variable_write_node().is_some() {
            return true;
        }
        if node.as_local_variable_read_node().is_some() {
            return true;
        }
        if node.as_multi_write_node().is_some() {
            return true;
        }
        if let Some(or) = node.as_or_node() {
            return self.predicate_less_case_condition_can_narrow(&or.left())
                || self.predicate_less_case_condition_can_narrow(&or.right());
        }
        if let Some(and) = node.as_and_node() {
            return self.predicate_less_case_condition_can_narrow(&and.left())
                || self.predicate_less_case_condition_can_narrow(&and.right());
        }
        let Some(call) = node.as_call_node() else {
            return false;
        };
        if call.name().as_slice() == b"!"
            && call.arguments().is_none()
            && call.block().is_none()
            && let Some(receiver) = call.receiver()
        {
            return self.predicate_less_case_condition_can_narrow(&receiver);
        }
        self.nil_predicate_target(&call).is_some()
            || self.is_a_predicate_target(&call).is_some()
            || self.case_eq_predicate_target(&call).is_some()
            // Bare pure-call scrutinee (`case; when x.pure_call`) --
            // mirrors `analyze_condition`'s CallNode arm so predicate-less
            // case/when narrows the same pure calls `if x.pure_call` does.
            || self
                .scrutinee_and_current_ty(&call.as_node())
                .is_some_and(|(s, t)| matches!(s, Scrutinee::Pure(_)) && !t.is_untyped())
    }

    fn predicate_less_case_condition_narrow<'pr>(
        &mut self,
        conditions: &[Node<'pr>],
    ) -> Option<(CondEnv, CondEnv)> {
        match conditions {
            [single] if self.predicate_less_case_condition_can_narrow(single) => {
                Some(self.analyze_condition(single))
            }
            [_, ..]
                if conditions.iter().all(|condition| {
                    self.predicate_less_case_condition_is_lvar_write(condition)
                }) =>
            {
                self.predicate_less_case_multi_write_narrow(conditions)
            }
            [_, ..]
                if conditions.iter().any(|condition| {
                    self.predicate_less_case_condition_contains_write(condition)
                }) =>
            {
                None
            }
            [_, ..]
                if conditions
                    .iter()
                    .any(|condition| self.predicate_less_case_condition_can_narrow(condition)) =>
            {
                Some(self.predicate_less_case_multi_condition_narrow(conditions))
            }
            [] => None,
            [_, ..] => None,
        }
    }

    fn predicate_less_case_multi_condition_narrow<'pr>(
        &mut self,
        conditions: &[Node<'pr>],
    ) -> (CondEnv, CondEnv) {
        let (mut truthy, mut falsy) = self.analyze_condition(&conditions[0]);
        for condition in conditions.iter().skip(1) {
            let (condition_truthy, condition_falsy) =
                self.with_cond_narrowing(&falsy, |c| c.analyze_condition(condition));
            truthy = truthy.join_branch(falsy.clone().merge_sequential(condition_truthy), self.env);
            falsy = falsy.merge_sequential(condition_falsy);
        }
        (truthy, falsy)
    }

    fn predicate_less_case_condition_is_lvar_write<'pr>(&self, node: &Node<'pr>) -> bool {
        if let Some(par) = node.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    return self.predicate_less_case_condition_is_lvar_write(&last);
                }
                return false;
            }
            return self.predicate_less_case_condition_is_lvar_write(&body);
        }
        node.as_local_variable_write_node().is_some()
    }

    fn predicate_less_case_condition_contains_write<'pr>(&self, node: &Node<'pr>) -> bool {
        if let Some(par) = node.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    return self.predicate_less_case_condition_contains_write(&last);
                }
                return false;
            }
            return self.predicate_less_case_condition_contains_write(&body);
        }
        if node.as_local_variable_write_node().is_some() || node.as_multi_write_node().is_some() {
            return true;
        }
        if let Some(or) = node.as_or_node() {
            return self.predicate_less_case_condition_contains_write(&or.left())
                || self.predicate_less_case_condition_contains_write(&or.right());
        }
        if let Some(and) = node.as_and_node() {
            return self.predicate_less_case_condition_contains_write(&and.left())
                || self.predicate_less_case_condition_contains_write(&and.right());
        }
        if let Some(call) = node.as_call_node()
            && call.name().as_slice() == b"!"
            && call.arguments().is_none()
            && call.block().is_none()
            && let Some(receiver) = call.receiver()
        {
            return self.predicate_less_case_condition_contains_write(&receiver);
        }
        false
    }

    fn predicate_less_case_multi_write_narrow<'pr>(
        &mut self,
        conditions: &[Node<'pr>],
    ) -> Option<(CondEnv, CondEnv)> {
        let (first_name, mut truthy, mut falsy) =
            self.predicate_less_case_lvar_write_env(&conditions[0])?;
        for condition in conditions.iter().skip(1) {
            let (name, condition_truthy, condition_falsy) = self
                .with_cond_narrowing(&falsy, |c| c.predicate_less_case_lvar_write_env(condition))?;
            if name != first_name {
                return None;
            }
            truthy = truthy.join_branch(falsy.clone().merge_sequential(condition_truthy), self.env);
            falsy = falsy.merge_sequential(condition_falsy);
        }
        Some((truthy, falsy))
    }

    fn predicate_less_case_lvar_write_env<'pr>(
        &mut self,
        node: &Node<'pr>,
    ) -> Option<(Name, CondEnv, CondEnv)> {
        if let Some(par) = node.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                return stmts
                    .body()
                    .iter()
                    .last()
                    .and_then(|last| self.predicate_less_case_lvar_write_env(&last));
            }
            return self.predicate_less_case_lvar_write_env(&body);
        }
        let write = node.as_local_variable_write_node()?;
        let name_str = String::from_utf8_lossy(write.name().as_slice());
        let name = self.checker_names().intern(&name_str);
        let value_ty = self.infer_type(&write.value(), None);
        let (truthy, falsy) = if value_ty.is_untyped() {
            (CondEnv::empty(), CondEnv::empty())
        } else {
            self.bare_truthy_falsy_env(Scrutinee::Lvar(name), value_ty)
        };
        Some((name, truthy, falsy))
    }

    /// Shared narrowing path for a bare-lvar scrutinee given the lvar's
    /// raw byte name. Bridges `LocalVariableReadNode` (`if x`) and
    /// `LocalVariableWriteNode` (`if x = expr`) — both end up splitting
    /// the lvar's currently-bound type against `nil | false`, the only
    /// difference between the two arms is where the name comes from in
    /// the Prism node.
    fn lvar_bare_narrow_by_name(&mut self, name_bytes: &[u8]) -> (CondEnv, CondEnv) {
        let name_str = String::from_utf8_lossy(name_bytes);
        let name = self.checker_names().intern(&name_str);
        let Some(current) = self.lookup_local_variable_for_read(name) else {
            return (CondEnv::empty(), CondEnv::empty());
        };
        if current.is_untyped() {
            return (CondEnv::empty(), CondEnv::empty());
        }
        self.bare_truthy_falsy_env(Scrutinee::Lvar(name), current)
    }

    fn multi_write_truthy_env<'pr>(&mut self, write: &MultiWriteNode<'pr>) -> Option<CondEnv> {
        if write.value().as_array_node().is_some() {
            return Some(CondEnv::empty());
        }
        if write.rest().is_some() {
            return None;
        }
        let lefts: Option<Vec<LocalVariableTargetNode<'pr>>> = write
            .lefts()
            .iter()
            .map(|node| node.as_local_variable_target_node())
            .collect();
        let rights: Option<Vec<LocalVariableTargetNode<'pr>>> = write
            .rights()
            .iter()
            .map(|node| node.as_local_variable_target_node())
            .collect();
        let (lefts, rights) = (lefts?, rights?);
        let truthy = partition_truthy(self.infer_type(&write.value(), None), self.env.types())?;
        if truthy.is_untyped() {
            return None;
        }
        let truthy = match self.union_of_tuple_to_tuple_of_union(truthy) {
            Some(per_position) => self.env.types().intern(Type::Tuple(per_position)),
            None => truthy,
        };
        let target_tys = self.multi_write_truthy_target_types(truthy, &lefts, &rights)?;
        let targets = lefts.iter().chain(rights.iter());
        let mut env = CondEnv::empty();
        for (target, ty) in targets.zip(target_tys) {
            let name_bytes = target.name().as_slice();
            if super::is_special_lvar_name(name_bytes) {
                continue;
            }
            let name_str = String::from_utf8_lossy(name_bytes);
            let name = self.checker_names().intern(&name_str);
            env = env.merge_sequential(CondEnv::single(Scrutinee::Lvar(name), ty));
        }
        Some(env)
    }

    fn multi_write_truthy_target_types(
        &self,
        truthy: Ty,
        lefts: &[LocalVariableTargetNode<'_>],
        rights: &[LocalVariableTargetNode<'_>],
    ) -> Option<Vec<Ty>> {
        match self.env.types().resolve(truthy) {
            Type::Tuple(elem_types) => {
                let nleading = lefts.len();
                let ntrailing = rights.len();
                let nelems = elem_types.len();
                let mut tys = Vec::with_capacity(nleading + ntrailing);
                for i in 0..nleading {
                    tys.push(*elem_types.get(i).unwrap_or(&Ty::NIL));
                }
                for i in 0..ntrailing {
                    let pos = nelems as isize - ntrailing as isize + i as isize;
                    let ty = if pos >= nleading as isize && pos >= 0 && (pos as usize) < nelems {
                        elem_types[pos as usize]
                    } else {
                        Ty::NIL
                    };
                    tys.push(ty);
                }
                Some(tys)
            }
            Type::ClassInstance { name, args }
                if *name == self.env.names().builtins().array && args.len() == 1 =>
            {
                let elem_ty = args[0];
                let elem_or_nil = self.env.types().intern(Type::Optional(elem_ty));
                Some(vec![elem_or_nil; lefts.len() + rights.len()])
            }
            _ if self.multi_assign_truthy_is_scalar(truthy) => {
                let target_count = lefts.len() + rights.len();
                if target_count == 0 {
                    return None;
                }
                let mut tys = Vec::with_capacity(target_count);
                tys.push(truthy);
                tys.extend(std::iter::repeat_n(Ty::NIL, target_count - 1));
                Some(tys)
            }
            _ => None,
        }
    }

    /// Truthy/falsy `CondEnv` pair for a bare-scrutinee predicate -- a
    /// predicate that IS the scrutinee (not a predicate method like
    /// `nil?`). The split runs against `nil | false`: truthy keeps the
    /// non-nil-non-false residue, falsy keeps the nil/false
    /// intersection. Bottom or unchanged results collapse to
    /// `CondEnv::empty` so callers never observe "narrow to
    /// unreachable" silencing real type errors on the dropped branch
    /// via the Bottom <: anything subtype rule.
    ///
    /// `scrutinee` selects the substrate: `Scrutinee::Lvar` plants on
    /// the lvar scope chain (matching the original ADR-0022 path),
    /// `Scrutinee::Pure` plants on the pure-call cache (ADR-0024).
    fn bare_truthy_falsy_env(&self, scrutinee: Scrutinee, current: Ty) -> (CondEnv, CondEnv) {
        let types = self.env.types();
        let false_ty = types.intern(Type::Literal(Literal::Bool(false)));
        let nil_or_false = union_of(Ty::NIL, false_ty, types);
        let truthy_ty = crate::narrowing::subtract(current, nil_or_false, self.env);
        let falsy_ty = crate::narrowing::narrow(current, nil_or_false, self.env);
        let truthy = if truthy_ty != current && truthy_ty != Ty::BOTTOM {
            CondEnv::single(scrutinee.clone(), truthy_ty)
        } else {
            CondEnv::empty()
        };
        let falsy = if falsy_ty != current && falsy_ty != Ty::BOTTOM {
            CondEnv::single(scrutinee, falsy_ty)
        } else {
            CondEnv::empty()
        };
        (truthy, falsy)
    }

    /// Build the `(truthy, falsy)` pair shared by every membership-style
    /// predicate (`nil?`, `is_a?`, `kind_of?`, `===`): `truthy` keeps
    /// the intersection with `target`, `falsy` keeps the complement.
    /// `scrutinee` selects whether the narrow plants on the lvar scope
    /// chain or on the pure-call cache (ADR-0024).
    fn narrow_pair(&self, scrutinee: Scrutinee, current: Ty, target: Ty) -> (CondEnv, CondEnv) {
        let truthy_ty = crate::narrowing::narrow(current, target, self.env);
        let falsy_ty = crate::narrowing::subtract(current, target, self.env);
        (
            CondEnv::single(scrutinee.clone(), truthy_ty),
            CondEnv::single(scrutinee, falsy_ty),
        )
    }

    /// Try to interpret a node as a pure-call cache key. Matches the
    /// shape rules of ADR-0024 Phase 1 plus constant singleton calls: bare
    /// local variable reads, `self`, constant paths, and chained
    /// `Receiver.method` sends with no arguments and no block. Implicit self
    /// (`(send nil :m)`) and explicit `self.m` both normalize to
    /// `Send(SelfRef, m)`. Ivar reads return `None`; they belong to
    /// `mid_pure_narrowing_extension`.
    pub(super) fn try_pure_key<'pr>(&self, node: &Node<'pr>) -> Option<PureKey> {
        if let Some(lvar) = node.as_local_variable_read_node() {
            let name_str = String::from_utf8_lossy(lvar.name().as_slice());
            let name = self.checker_names().intern(&name_str);
            return Some(PureKey::Lvar(name));
        }
        if node.as_self_node().is_some() {
            return Some(PureKey::SelfRef);
        }
        if let Some(constant) = node.as_constant_read_node() {
            let name = String::from_utf8_lossy(constant.name().as_slice());
            let ty = self.try_resolve_constant_read(&name, node)?.ty;
            return self.const_singleton_key(ty);
        }
        if let Some(path) = node.as_constant_path_node() {
            let ty = self.resolve_constant_path_outcome(&path).resolved_ty();
            return self.const_singleton_key(ty);
        }
        if let Some(call) = node.as_call_node() {
            if call.arguments().is_some() || call.block().is_some() {
                return None;
            }
            // Implicit self (`(send nil :path)`) has no receiver node; the
            // base is self. Explicit `self.path` recurses into the
            // SelfNode arm above. Both normalize to `Send(SelfRef, _)`.
            let recv_key = match call.receiver() {
                Some(receiver) => self.try_pure_key(&receiver)?,
                None => PureKey::SelfRef,
            };
            let method_str = String::from_utf8_lossy(call.name().as_slice());
            let method = self.env.names().intern_symbol(&method_str);
            return Some(PureKey::Send(Box::new(recv_key), method));
        }
        None
    }

    fn const_singleton_key(&self, ty: Ty) -> Option<PureKey> {
        match self.env.types().resolve(ty) {
            Type::ClassSingleton { name } => Some(PureKey::ConstPath(*name)),
            _ => None,
        }
    }

    /// Write the just-computed return type back into the pure-call
    /// cache, if the call shape and the resolved method qualify for
    /// caching. Called from `check_node`'s CallNode arm immediately
    /// after `infer_call_return_type` returns. Phase 1 qualification:
    ///
    /// 1. The call shape is PureKey-derivable -- no args, no block,
    ///    receiver chain rooted at an lvar, self, or a constant path.
    /// 2. The resolved method passes `Method::is_pure` (attr_reader /
    ///    attr_accessor reader; annotation-driven recognition is the
    ///    follow-up `mid_pure_narrowing_extension`).
    /// 3. The receiver chain is inductively pure: every cache entry
    ///    has been gated through this function, so a `Send` whose
    ///    receiver-key is a base case (`Lvar`, `SelfRef`, or
    ///    `ConstPath`) or already present in the cache (inductively pure)
    ///    is safe. A `Send`
    ///    whose receiver-key is an uncached `Send` would mean the
    ///    receiver was reached via a non-pure path -- caching the
    ///    outer narrow then would be unsound, because re-calling the
    ///    receiver could yield a different value and invalidate the
    ///    cached narrow at the outer link.
    fn maybe_cache_pure_call<'pr>(
        &mut self,
        call: &CallNode<'pr>,
        resolved: &ResolvedCall,
        ret: Ty,
    ) {
        let Some(key) = self.try_pure_key(&call.as_node()) else {
            return;
        };
        if let PureKey::Send(recv_key, _) = &key {
            let receiver_pure = match recv_key.as_ref() {
                PureKey::Lvar(_) | PureKey::SelfRef | PureKey::ConstPath(_) => true,
                PureKey::Send(_, _) => self.ctx.pure_call_env().get(recv_key.as_ref()).is_some(),
            };
            if !receiver_pure {
                return;
            }
        }
        let Some(target) = resolved.target.as_ref() else {
            return;
        };
        let method_name_str = String::from_utf8_lossy(call.name().as_slice());
        let method_name = self.env.names().intern_symbol(&method_name_str);
        if !target.is_pure(method_name, self.env.names()) {
            return;
        }
        self.ctx.pure_call_env_mut().set(key, ret);
    }

    /// Recursively verify that the receiver-chain rooted at `node` is
    /// pure under Phase 1 rules: every link must be a no-arg, no-block
    /// send whose resolved method passes `Method::is_pure`, with a
    /// bare local variable, `self`, or constant path at the base (implicit
    /// self is a receiverless send, so a no-receiver link is itself a pure
    /// base).
    /// Used by
    /// `scrutinee_and_current_ty` before installing a pure-call narrow
    /// on a key the cache has not yet seen -- the cache-write gate in
    /// `maybe_cache_pure_call` proves chain purity inductively, but
    /// receivers reached only via `infer_type` (e.g. the `c.phone` in
    /// `c.phone.nil?`) skip that gate, so the predicate path has to
    /// re-verify before planting a narrow.
    fn verify_pure_chain<'pr>(&self, node: &Node<'pr>) -> bool {
        if node.as_local_variable_read_node().is_some() {
            return true;
        }
        if node.as_self_node().is_some() {
            return true;
        }
        if node.as_constant_read_node().is_some() {
            return self.try_pure_key(node).is_some();
        }
        if let Some(path) = node.as_constant_path_node() {
            return self.try_pure_key(&path.as_node()).is_some();
        }
        let Some(call) = node.as_call_node() else {
            return false;
        };
        if call.arguments().is_some() || call.block().is_some() {
            return false;
        }
        // No receiver = implicit self, a pure base. Explicit receivers
        // recurse (SelfNode bottoms out at the arm above; lvar at the top).
        if let Some(receiver) = call.receiver()
            && !self.verify_pure_chain(&receiver)
        {
            return false;
        }
        let resolved = ResolvedCall::resolve(self, &call);
        let Some(target) = resolved.target.as_ref() else {
            return false;
        };
        let method_str = String::from_utf8_lossy(call.name().as_slice());
        let method = self.env.names().intern_symbol(&method_str);
        target.is_pure(method, self.env.names())
    }

    /// Read the scrutinee identity (lvar name or pure-call key) and the
    /// current type from the live envs. Used by every predicate-target
    /// helper that previously only handled bare lvars.
    ///
    /// For a pure-call receiver the cache is consulted first; on a miss
    /// the function falls back to a fresh `infer_type`. The miss path
    /// matters when the pure call appears as the receiver of a
    /// non-pure method (e.g. the `c.phone` in `c.phone.nil?`) -- those
    /// receivers are reached only via `infer_type` during enclosing
    /// resolution, so `check_node`'s cache writeback never fires for
    /// them. The narrow we plant via `with_cond_branch` then populates
    /// the cache, so subsequent body reads via `infer_type` hit and
    /// return the narrowed type.
    /// Build a `(Scrutinee::Lvar, current_ty)` pair from raw lvar bytes.
    /// `None` when the lvar isn't bound in any reachable scope, which
    /// suppresses narrowing the same way `check_node`'s read-without-
    /// assign fallback would. Shared by the bare-read and post-write
    /// arms of `scrutinee_and_current_ty`.
    fn lvar_scrutinee_from_name(&self, name_bytes: &[u8]) -> Option<(Scrutinee, Ty)> {
        let name_str = String::from_utf8_lossy(name_bytes);
        let name = self.checker_names().intern(&name_str);
        let current = self.lookup_local_variable_for_read(name)?;
        Some((Scrutinee::Lvar(name), current))
    }

    fn scrutinee_and_current_ty<'pr>(&self, receiver: &Node<'pr>) -> Option<(Scrutinee, Ty)> {
        if let Some(par) = receiver.as_parentheses_node()
            && let Some(body) = par.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    return self.scrutinee_and_current_ty(&last);
                }
                return None;
            }
            return self.scrutinee_and_current_ty(&body);
        }
        if let Some(lvar) = receiver.as_local_variable_read_node() {
            return self.lvar_scrutinee_from_name(lvar.name().as_slice());
        }
        // `case x = expr; when T` predicate. The caller (`check_case_node`
        // / `analyze_condition`) runs `check_node` on the predicate
        // first, which routes through `visit_local_variable_write_node`
        // and binds the lvar to the RHS type. Reading the lvar back here
        // is identical to the bare-read case. Mirrors the precedent in
        // `analyze_condition` (`if x = expr; ...`).
        if let Some(write) = receiver.as_local_variable_write_node() {
            return self.lvar_scrutinee_from_name(write.name().as_slice());
        }
        let key = self.try_pure_key(receiver)?;
        if let Some(current) = self.ctx.pure_call_env().get(&key) {
            return Some((Scrutinee::Pure(key), current));
        }
        // Cache miss -- the write gate in `maybe_cache_pure_call` did
        // not run for this receiver (typically because it appears only
        // as the receiver of a non-pure outer call, e.g. the `c.phone`
        // in `c.phone.nil?`). Re-verify chain purity here before
        // planting a narrow; otherwise a non-pure call could get a
        // soundness-violating cache entry installed by
        // `with_cond_branch`, since second-call values are not
        // guaranteed equal.
        if !self.verify_pure_chain(receiver) {
            return None;
        }
        let current = self.infer_type(receiver, None);
        if current.is_untyped() {
            return None;
        }
        Some((Scrutinee::Pure(key), current))
    }

    /// Recognise `<local_var>.nil?` with no arguments and no block,
    /// returning `(name, current_type)`. Anything else (different
    /// method, non-local receiver, args, block) is `None`.
    ///
    /// Additionally gates on method resolution: the narrow only fires
    /// when the receiver's `nil?` resolves to a primitive overload
    /// (`Object` / `NilClass` / `Kernel`). A user-defined `nil?` on
    /// the receiver's class blocks the narrow, matching Steep
    /// `Interface::Builder` (`builder.rb:761-772`). See
    /// [`Self::nil_predicate_resolves_to_primitive`] for the
    /// fail-open semantics on unresolved / method-absent receivers.
    fn nil_predicate_target<'pr>(&self, call: &CallNode<'pr>) -> Option<(Scrutinee, Ty)> {
        if call.name().as_slice() != b"nil?" {
            return None;
        }
        if call.arguments().is_some() || call.block().is_some() {
            return None;
        }
        let receiver = call.receiver()?;
        let (scrutinee, current) = self.scrutinee_and_current_ty(&receiver)?;
        if !self.nil_predicate_resolves_to_primitive(current) {
            return None;
        }
        Some((scrutinee, current))
    }

    /// Returns `true` when every overload of `receiver#nil?` is declared
    /// in `::Object`, `::NilClass`, or `::Kernel` — the three primitive
    /// owners Steep recognises.
    ///
    /// Returns `true` (fail-open) when `lookup_method` returns `None`,
    /// which happens in two distinct cases:
    ///   1. the receiver kind isn't yet handled by the per-call resolver
    ///      (untyped / Union / Interface / Var — `_ => None` in
    ///      `method_resolver`). Union member with a user-defined `nil?`
    ///      slips past the gate — pinned by
    ///      `test_if_nil_predicate_union_member_override_pins_known_limitation`
    ///      and tracked in the todo Known limitations.
    ///   2. `nil?` itself isn't declared on the receiver's ancestor
    ///      chain. Benign — no overload exists, so there's no override
    ///      to detect, and pre-gate behaviour matches.
    ///
    /// Empty `defs` is treated as "not primitive": vacuous truth would
    /// let a synthesised empty-method record bypass the gate, and the
    /// safer default with no overload data is to suppress narrowing.
    fn nil_predicate_resolves_to_primitive(&self, receiver: Ty) -> bool {
        let nil_q = self.env.names().intern_symbol("nil?");
        let Some(resolved) = super::method_resolver::lookup_method(self.env, receiver, nil_q)
        else {
            return true;
        };
        if resolved.method.defs.is_empty() {
            return false;
        }
        let builtins = self.env.names().builtins();
        resolved.method.defs.iter().all(|td| {
            td.defined_in == builtins.object
                || td.defined_in == builtins.nil_class
                || td.defined_in == builtins.kernel
        })
    }

    /// Recognise `<local_var>.is_a?(<class_literal>)` or
    /// `<local_var>.kind_of?(<class_literal>)`. Returns
    /// `(name, current_type, target_instance_type)` where `target_*`
    /// is the instance type built from the Class/Module literal
    /// (generics filled with `Ty::UNTYPED`). The user-override
    /// question is intentionally ignored — see the `nil?` precedent.
    fn is_a_predicate_target<'pr>(&self, call: &CallNode<'pr>) -> Option<(Scrutinee, Ty, Ty)> {
        let method = call.name();
        let method_bytes = method.as_slice();
        if method_bytes != b"is_a?" && method_bytes != b"kind_of?" {
            return None;
        }
        if call.block().is_some() {
            return None;
        }
        let args = call.arguments()?;
        let arg_nodes: Vec<_> = args.arguments().iter().collect();
        if arg_nodes.len() != 1 {
            return None;
        }
        let receiver = call.receiver()?;
        let (scrutinee, current) = self.scrutinee_and_current_ty(&receiver)?;
        let target_ty = self.class_literal_target(&arg_nodes[0])?;
        Some((scrutinee, current, target_ty))
    }

    /// Recognise `<class_literal> === <local_var>`. The Class/Module
    /// receiver restriction (ADR-0022 — `Range#===` / `Regexp#===` /
    /// user-defined `===` on instances are out of scope) falls out
    /// structurally: `class_literal_target` only succeeds when the
    /// receiver type is `ClassSingleton`, so a `Range` instance
    /// receiver (`(1..10) === x`) returns `None` here without any
    /// explicit method-name allowlist.
    fn case_eq_predicate_target<'pr>(&self, call: &CallNode<'pr>) -> Option<(Scrutinee, Ty, Ty)> {
        if call.name().as_slice() != b"===" {
            return None;
        }
        if call.block().is_some() {
            return None;
        }
        let args = call.arguments()?;
        let arg_nodes: Vec<_> = args.arguments().iter().collect();
        if arg_nodes.len() != 1 {
            return None;
        }
        let (scrutinee, current) = self.scrutinee_and_current_ty(&arg_nodes[0])?;
        let receiver = call.receiver()?;
        let target_ty = self.class_literal_target(&receiver)?;
        Some((scrutinee, current, target_ty))
    }

    /// Truthy/falsy `CondEnv` pair for a predicate call on a union
    /// receiver, splitting members by whether their own per-member return
    /// type can be truthy / can be falsy. Port of Steep's
    /// `evaluate_union_method_call` (logic_type_interpreter.rb:447).
    ///
    /// `resolve_call_target_at` already requires *every* union member to
    /// resolve the called name (ADR-0021 per-name dispatch) — if one
    /// member lacks it, dispatch fails for the whole union and the
    /// ordinary `check_no_method` path reports it, so this function just
    /// declines to narrow (`None`) rather than duplicating that gate.
    ///
    /// `-> bool` and `-> untyped` returns are placed on *both* sides
    /// directly instead of through `partition_truthy`/`partition_falsy`:
    /// those helpers' default arm treats an undecomposed `Type::Bool` as
    /// falsy-impossible, which is correct for the value-splitting
    /// `&&`/`||` call sites they were built for but wrong here — a
    /// `bool`-returning member truly can go either way, and Steep's
    /// measured behavior is to leave the union whole rather than guess.
    ///
    /// Returns `None` (instead of an all-empty pair) when neither side
    /// narrows anything, so the caller can fall through to the bare
    /// pure-call arm for a union receiver whose predicate is itself pure
    /// (ADR-0024) but whose members don't diverge in truthiness.
    fn union_member_predicate_env<'pr>(&self, call: &CallNode<'pr>) -> Option<(CondEnv, CondEnv)> {
        if call.block().is_some() {
            return None;
        }
        let receiver = call.receiver()?;
        let (scrutinee, current) = self.scrutinee_and_current_ty(&receiver)?;
        if !matches!(self.env.types().resolve(current), Type::Union(_)) {
            return None;
        }
        let method_name = String::from_utf8_lossy(call.name().as_slice()).to_string();
        let CallTarget::UnionMethod { components, .. } =
            self.resolve_call_target_at(current, &method_name)?
        else {
            return None;
        };
        let arguments = self.collect_call_arguments(super::calls::CallSite::Call(call));
        let types = self.env.types();
        let mut truthy_members = Vec::with_capacity(components.len());
        let mut falsy_members = Vec::with_capacity(components.len());
        for c in &components {
            let ret = self.infer_return_type(
                &c.method_def,
                &c.bindings,
                &arguments,
                c.receiver_type,
                Some(call),
                None,
            );
            let (truthy_possible, falsy_possible) =
                if ret.is_untyped() || matches!(types.resolve(ret), Type::Bool) {
                    (true, true)
                } else {
                    (
                        partition_truthy(ret, types).is_some(),
                        partition_falsy(ret, types).is_some(),
                    )
                };
            if truthy_possible {
                truthy_members.push(c.receiver_type);
            }
            if falsy_possible {
                falsy_members.push(c.receiver_type);
            }
        }
        let truthy_ty = union_of_many(&truthy_members, types);
        let falsy_ty = union_of_many(&falsy_members, types);
        let truthy_hit = truthy_ty != current && truthy_ty != Ty::BOTTOM;
        let falsy_hit = falsy_ty != current && falsy_ty != Ty::BOTTOM;
        if !truthy_hit && !falsy_hit {
            return None;
        }
        let truthy = if truthy_hit {
            CondEnv::single(scrutinee.clone(), truthy_ty)
        } else {
            CondEnv::empty()
        };
        let falsy = if falsy_hit {
            CondEnv::single(scrutinee, falsy_ty)
        } else {
            CondEnv::empty()
        };
        Some((truthy, falsy))
    }

    /// Recognise `<scrutinee> == <literal>` (`x == :foo`, `n == 1`,
    /// `s == "a"`). The literal must sit on the argument side — the
    /// receiver is resolved through `scrutinee_and_current_ty`, which
    /// only recognises lvar / pure-call shapes and structurally
    /// rejects a literal receiver, so `:foo == x` never reaches this
    /// arm (matches Steep: `ReceiverIsArg` narrows the receiver, not
    /// the argument). Literal recognition on the argument side shares
    /// `case_when_literal_target` with `when` conds so `n == 1` and
    /// `when 1` gain the same literal set together.
    ///
    /// Bails when the truthy narrow would collapse to `Ty::BOTTOM` —
    /// `current` and `target_ty` share no overlap (e.g. a plain
    /// `Widget == 1`, `defined_in` gate passes since `Widget` never
    /// overrides `==`). `narrow_pair` has no `BOTTOM` guard of its own,
    /// so installing a Bottom-narrowed receiver into the truthy branch
    /// would silently swallow that branch's own diagnostics (`NoMethod`
    /// etc. — mirrors `bare_truthy_falsy_env`'s guard, crema-review
    /// 2026-07-16 adversarial finding).
    fn eq_literal_predicate_target<'pr>(
        &self,
        call: &CallNode<'pr>,
    ) -> Option<(Scrutinee, Ty, Ty)> {
        if call.name().as_slice() != b"==" {
            return None;
        }
        if call.block().is_some() {
            return None;
        }
        let args = call.arguments()?;
        let arg_nodes: Vec<_> = args.arguments().iter().collect();
        if arg_nodes.len() != 1 {
            return None;
        }
        let receiver = call.receiver()?;
        let (scrutinee, current) = self.scrutinee_and_current_ty(&receiver)?;
        let target_ty = self.case_when_literal_target(&arg_nodes[0])?;
        if !self.eq_resolves_to_builtin_value_class(current) {
            return None;
        }
        if crate::narrowing::narrow(current, target_ty, self.env) == Ty::BOTTOM {
            return None;
        }
        Some((scrutinee, current, target_ty))
    }

    /// Returns `true` when every overload of `receiver#==` is declared
    /// in one of the builtin value classes Steep gates literal-equality
    /// narrowing on (`interface/builder.rb:812-829`): `BasicObject`,
    /// `Object`, `Kernel`, `String`, `Integer`, `Symbol`, `TrueClass`,
    /// `FalseClass`, `NilClass`. A user-defined `==` outside this set
    /// can implement any equivalence relation, so narrowing on it
    /// would be unsound. Fail-open semantics mirror
    /// [`Self::nil_predicate_resolves_to_primitive`] — see that doc
    /// comment for the two `None` cases this covers.
    fn eq_resolves_to_builtin_value_class(&self, receiver: Ty) -> bool {
        let eq = self.env.names().intern_symbol("==");
        let Some(resolved) = super::method_resolver::lookup_method(self.env, receiver, eq) else {
            return true;
        };
        if resolved.method.defs.is_empty() {
            return false;
        }
        let builtins = self.env.names().builtins();
        resolved.method.defs.iter().all(|td| {
            td.defined_in == builtins.basic_object
                || td.defined_in == builtins.object
                || td.defined_in == builtins.kernel
                || td.defined_in == builtins.string
                || td.defined_in == builtins.integer
                || td.defined_in == builtins.symbol
                || td.defined_in == builtins.true_class
                || td.defined_in == builtins.false_class
                || td.defined_in == builtins.nil_class
        })
    }

    /// Resolve a Class/Module literal expression to the instance type
    /// it stands for, ready to feed `narrow` / `subtract` as the target.
    ///
    /// Accepts only `ConstantReadNode` and `ConstantPathNode` —
    /// dynamic forms (`klass = Foo; x.is_a?(klass)`) return `None`.
    /// The resolved constant must be a `ClassSingleton`; anything else
    /// (instance values, aliases, interfaces, unresolved misses) is
    /// `None`. Generics are filled with `Ty::UNTYPED` so
    /// `is_a?(Array)` narrows to `Array[untyped]`, matching Steep.
    ///
    /// Resolution rides the silent `&self` constant lookup paths
    /// (`try_resolve_constant_read` / `resolve_constant_path_outcome`),
    /// so no diagnostics are pushed — the side-effecting walk has
    /// already emitted any `UnknownConstant` from
    /// `check_node(&predicate, None)`.
    fn class_literal_target<'pr>(&self, node: &Node<'pr>) -> Option<Ty> {
        let resolved_ty = if let Some(cr) = node.as_constant_read_node() {
            let name_str = String::from_utf8_lossy(cr.name().as_slice()).to_string();
            let generic = cr.as_node();
            self.try_resolve_constant_read(&name_str, &generic)?.ty
        } else if let Some(path) = node.as_constant_path_node() {
            match self.resolve_constant_path_outcome(&path) {
                ConstantPathOutcome::Walked {
                    kind: ConstantPathOutcomeKind::Resolved(ty, _),
                    ..
                } => ty,
                _ => return None,
            }
        } else {
            return None;
        };

        let types = self.env.types();
        let name = match types.resolve(resolved_ty) {
            Type::ClassSingleton { name } => name,
            _ => return None,
        };
        let arity = self
            .env
            .class_type_params_by_type_name(name)
            .map(|tp| tp.len())
            .unwrap_or(0);
        let args = vec![Ty::UNTYPED; arity];
        Some(types.intern(Type::ClassInstance { name: *name, args }))
    }

    fn case_when_literal_target<'pr>(&self, node: &Node<'pr>) -> Option<Ty> {
        // Case/when literals whose `===` semantics are static equality.
        // Class/module targets still flow through `case_when_class_target`.
        if node.as_nil_node().is_some() {
            return Some(Ty::NIL);
        }
        if node.as_false_node().is_some() {
            return Some(self.env.types().intern(Type::Literal(Literal::Bool(false))));
        }
        if node.as_true_node().is_some() {
            return Some(self.env.types().intern(Type::Literal(Literal::Bool(true))));
        }
        if let Some(sym) = node.as_symbol_node() {
            let s = String::from_utf8_lossy(sym.unescaped()).to_string();
            return Some(self.env.types().intern(Type::Literal(Literal::Symbol(s))));
        }
        if let Some(int_node) = node.as_integer_node() {
            let integer = int_node.value();
            let (negative, digits) = integer.to_u32_digits();
            let val = prism_digits_to_decimal_string(negative, digits);
            return Some(
                self.env
                    .types()
                    .intern(Type::Literal(Literal::Integer(val))),
            );
        }
        if let Some(str_node) = node.as_string_node() {
            let s = String::from_utf8_lossy(str_node.unescaped()).to_string();
            return Some(self.env.types().intern(Type::Literal(Literal::String(s))));
        }
        None
    }

    fn case_when_class_target<'pr>(&self, node: &Node<'pr>) -> Option<Ty> {
        self.class_literal_target(node).or_else(|| {
            let call = node.as_call_node()?;
            self.dynamic_case_when_class_target(&call)
        })
    }

    fn infer_type_with_scrutinee_overlay<'pr>(
        &self,
        node: &Node<'pr>,
        hint: Option<Ty>,
        scrutinee: &Scrutinee,
        ty: Ty,
    ) -> Ty {
        match scrutinee {
            Scrutinee::Lvar(name) => {
                let mut overlay = FxHashMap::default();
                overlay.insert(*name, ty);
                self.with_overlay(overlay, |checker| checker.infer_type(node, hint))
            }
            Scrutinee::Pure(key) => {
                let mut overlay = FxHashMap::default();
                overlay.insert(key.clone(), ty);
                self.with_pure_overlay(overlay, |checker| checker.infer_type(node, hint))
            }
        }
    }

    /// Read-only mirror of `bare_truthy_falsy_env` for the `infer_type`
    /// path. Returns `(scrutinee, truthy_overlay, falsy_overlay)` when
    /// the predicate is a shape `scrutinee_and_current_ty` recognizes
    /// (bare lvar read, `x = expr` write predicate, or a pure receiver
    /// chain). Each overlay is `Some(narrowed_ty)` when the narrow
    /// changes something and does not collapse the type, and `None`
    /// otherwise — mirroring `bare_truthy_falsy_env`'s
    /// `truthy_ty != current && truthy_ty != Ty::BOTTOM` gate. Without
    /// the `BOTTOM` guard a non-nilable scrutinee (`v: Integer`) plants
    /// `v = BOTTOM` into the else arm's overlay; reading `v` in that
    /// arm yields `BOTTOM` and `union_of` silently drops the arm's
    /// value, hiding a legitimate `MethodBodyTypeMismatch` on
    /// `if v; "s"; else v; end` where the declared return is `String`.
    ///
    /// Currently limited to what `scrutinee_and_current_ty` covers — a
    /// bare truthy check. Membership predicates (`x.nil?`, `is_a?`,
    /// `kind_of?`, `===`), boolean composites (`a && b`, `a || b`), and
    /// `!a` still fall through to unnarrowed inference in `infer_type`.
    /// Extending them here would mirror `analyze_condition` in the same
    /// way, but is left as follow-up scope — the reported false positive
    /// (`rbs/lib/rbs/repository.rb:87 find_best_version`) is a bare-lvar
    /// predicate.
    fn bare_truthy_falsy_pair_read_only<'pr>(
        &self,
        predicate: &Node<'pr>,
    ) -> Option<(Scrutinee, Option<Ty>, Option<Ty>)> {
        let (scrutinee, current) = self.scrutinee_and_current_ty(predicate)?;
        let types = self.env.types();
        let false_ty = types.intern(Type::Literal(Literal::Bool(false)));
        let nil_or_false = union_of(Ty::NIL, false_ty, types);
        let truthy_ty = crate::narrowing::subtract(current, nil_or_false, self.env);
        let falsy_ty = crate::narrowing::narrow(current, nil_or_false, self.env);
        let useful = |narrow_ty: Ty| {
            if narrow_ty != current && narrow_ty != Ty::BOTTOM {
                Some(narrow_ty)
            } else {
                None
            }
        };
        Some((scrutinee, useful(truthy_ty), useful(falsy_ty)))
    }

    /// `infer_type` an if/unless arm body, applying a scrutinee overlay
    /// when the caller has one. `None` overlay means "leave the scope
    /// alone" — either the predicate shape was not recognized or the
    /// narrow was a no-op / collapse (see `bare_truthy_falsy_pair_read_only`).
    fn infer_arm_with_optional_overlay<'pr>(
        &self,
        node: &Node<'pr>,
        hint: Option<Ty>,
        overlay: Option<(&Scrutinee, Ty)>,
    ) -> Ty {
        match overlay {
            Some((scrutinee, ty)) => {
                self.infer_type_with_scrutinee_overlay(node, hint, scrutinee, ty)
            }
            None => self.infer_type(node, hint),
        }
    }

    fn dynamic_case_when_class_target<'pr>(&self, call: &CallNode<'pr>) -> Option<Ty> {
        if call.name().as_slice() != b"class" {
            return None;
        }
        if call.arguments().is_some() || call.block().is_some() {
            return None;
        }
        let receiver = call.receiver()?;
        receiver.as_local_variable_read_node()?;
        let class_ty = self.infer_type(&call.as_node(), None);
        self.dynamic_class_target_from_class_object(class_ty)
    }

    fn dynamic_class_target_from_class_object(&self, class_ty: Ty) -> Option<Ty> {
        if class_ty.is_untyped() {
            return None;
        }
        match self.env.types().resolve(class_ty) {
            Type::ClassSingleton { name } => {
                let arity = self
                    .env
                    .class_type_params_by_type_name(name)
                    .map(|tp| tp.len())
                    .unwrap_or(0);
                let args = vec![Ty::UNTYPED; arity];
                Some(
                    self.env
                        .types()
                        .intern(Type::ClassInstance { name: *name, args }),
                )
            }
            Type::Alias { .. } => {
                let expanded = definition_builder::expand_alias(self.env, class_ty);
                if expanded == class_ty {
                    return None;
                }
                self.dynamic_class_target_from_class_object(expanded)
            }
            Type::Union(members) => {
                let targets: Option<Vec<Ty>> = members
                    .iter()
                    .map(|member| self.dynamic_class_target_from_class_object(*member))
                    .collect();
                let targets = targets?;
                if targets.is_empty() {
                    return None;
                }
                Some(union_of_many(&targets, self.env.types()))
            }
            _ => None,
        }
    }

    /// `check_node`'s ElseNode branch: forward the hint to the body so
    /// `cond ? a : b` (whose `else` arm is an ElseNode wrapper) and
    /// full `if ... else ... end` both reach their value-yielding
    /// statements with the caller's hint.
    fn check_else_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let else_node = match node.as_else_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        if let Some(stmts) = else_node.statements() {
            self.check_statements_with_hint(&stmts, hint)
        } else {
            Ty::NIL
        }
    }

    /// `check_node`'s OrNode branch: `a || b` — Steep `:or` aligned
    /// (lib/steep/type_construction.rb:1854-1910).
    ///
    /// `||` short-circuits on truthy, so each operand sits in a
    /// different branch of the result:
    ///
    /// * The **left** is the result only when truthy; when falsy the
    ///   value is statically `nil` or `false`. Widen the caller hint
    ///   to `hint | nil | false` so a nilable / boolean left does not
    ///   fight the assertion target during synthesis.
    /// * The **right** is the result only when the left was falsy. Hand
    ///   it the truthy partition of the left's synthesized type instead
    ///   of forwarding the caller hint blindly — this is the type the
    ///   right operand has to produce for the overall expression to
    ///   satisfy the outer assertion.
    ///
    /// Shape `a or b` ≡ `if a; a; else; b; end`: routing the two operands
    /// through `with_branch_outcome` + `join_arms_with_divergence` lets
    /// the same terminal model that `check_if_node` uses pin the narrow
    /// when the right operand diverges (`x or raise`, `x or return`).
    ///
    /// Value type vs env join are computed separately: the env join still
    /// threads the full `left_ty` (preserving the post-or env shape from
    /// before the value-type modeling), while the value type is
    /// `partition_truthy(left) | right_value` so a nilable left contributes
    /// only its truthy partition. Falls to `Ty::BOTTOM` when both arms
    /// diverge — matching how nested guard clauses propagate
    /// (`if x.nil?; raise "a" or raise "b"; end`).
    ///
    /// Note: the right operand is always synthesized for diagnostic side
    /// effects (constants, calls), not short-circuited at compile time.
    /// `1 || 2.foo` will still surface `2.foo`'s NoMethod — same as
    /// Steep's `:or` arm.
    pub(super) fn check_or_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let or = match node.as_or_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let left = or.left();
        let left_hint = hint.map(|h| {
            let types = self.env.types();
            let false_ty = types.intern(Type::Literal(Literal::Bool(false)));
            union_of_many(&[h, Ty::NIL, false_ty], types)
        });
        let left_ty = self.check_node(&left, left_hint);
        if left_ty == Ty::BOTTOM {
            // Left diverged (`raise "a" or 42`): the right operand is
            // unreachable. Surface `Ty::BOTTOM` so the surrounding stmt
            // list cuts off at this expression instead of evaluating the
            // right arm and joining a live env back into the post scope.
            return Ty::BOTTOM;
        }
        let right_hint = partition_truthy(left_ty, self.env.types());
        let (truthy, falsy) = self.analyze_condition(&left);

        // Env join still threads the full `left_ty` through arm_truthy
        // so the post-or env mirrors Steep's "both operands evaluated"
        // shape. The value-type union is computed separately so the
        // truthy-arm contribution can use `partition_truthy(left_ty)`
        // without affecting BOTTOM-driven env adoption in
        // `join_arms_with_divergence`.
        let base = self.ctx.snapshot_scopes();
        let arm_truthy = self.with_branch_outcome(&base, &truthy, |_| left_ty);
        let arm_falsy = self.with_branch_outcome(&base, &falsy, |checker| {
            checker.check_node(&or.right(), right_hint)
        });
        let right_value_ty = arm_falsy.last_ty;
        let left_diverged = arm_truthy.last_ty == Ty::BOTTOM;
        let right_diverged = arm_falsy.last_ty == Ty::BOTTOM;
        let env_join_ty = self.join_arms_with_divergence(base, arm_truthy, arm_falsy);
        if left_diverged && right_diverged {
            return env_join_ty; // BOTTOM
        }
        // Value type: truthy partition of left ∪ right's synthesized type.
        // Statically-falsy left collapses arm_truthy to BOTTOM (drops via
        // union_of); divergent right keeps the truthy partition alone.
        // Statically-truthy left (`partition_falsy` empty) makes the right
        // arm unreachable too — Steep folds to the left's truthy partition
        // alone (type_construction.rb:1820-1828) even though the right is
        // still walked above for its side-effect diagnostics.
        //
        // `narrowing::subtract` (not `partition_truthy`) computes the
        // truthy partition: `subtract` expands `Type::Alias` before
        // comparing (`partition_truthy` deliberately stays `TypeTable`-only
        // and never expands aliases) and decomposes an undecomposed
        // `Type::Bool` against a literal target, neither of which
        // `partition_truthy` does. `narrowing::subtract` is the same
        // substrate the ordinary bare-lvar narrow path
        // (`bare_truthy_falsy_env`) uses and handles both shapes correctly.
        let types = self.env.types();
        let false_ty = types.intern(Type::Literal(Literal::Bool(false)));
        let nil_or_false = union_of(Ty::NIL, false_ty, types);
        let truthy_value = crate::narrowing::subtract(left_ty, nil_or_false, self.env);
        // `x = y or raise`: when the right arm diverges, `x` is
        // confirmed non-{nil,false} for the remainder of this
        // statement's control flow. `analyze_condition`'s own narrow of
        // `left` plants that confirmation on the innermost scope only
        // when it's a real narrow (`y`'s raw type was still nilable) —
        // if `y` was already non-nilable (e.g. a chained call whose
        // static return type excludes nil), the CondEnv came back empty
        // and no shadow was planted. An *enclosing* conditional
        // (`unless x; x = y or raise; end`) still plants its own
        // predicate-narrow on the innermost scope for its other arm, so
        // `Context::join_branches`'s per-scope-index union then compares
        // that inner shadow against `x`'s un-narrowed real-home-scope
        // binding (the shadow's `shadows_outer` check drops it instead of
        // feeding it into the outer union) — silently reintroducing
        // `nil` at the join. Force the shadow here unconditionally so
        // the confirmation is always visible at the scrutinee's
        // innermost representation, matching Steep's parity for this
        // guard idiom regardless of the RHS's raw nilability.
        if right_diverged
            && truthy_value != Ty::BOTTOM
            && let Some(write) = left.as_local_variable_write_node()
        {
            let name_str = String::from_utf8_lossy(write.name().as_slice());
            let name = self.checker_names().intern(&name_str);
            self.ctx.set_local_variable(name, truthy_value);
        }
        if right_diverged || partition_falsy(left_ty, self.env.types()).is_none() {
            return truthy_value;
        }
        union_of(truthy_value, right_value_ty, self.env.types())
    }

    /// `check_node`'s AndNode branch: `a && b` — both operands receive
    /// the caller's hint. The right operand evaluates under the truthy
    /// env of the left so `b`'s analysis sees the narrowed local types.
    /// Walk left first, compute narrowing second, then walk right under
    /// the env — same order as visit_and_node so the two callbacks
    /// stay lockstep.
    ///
    /// Shape `a and b` ≡ `if a; b; else; a; end`. Same terminal model
    /// routing as `check_or_node` so a divergent right arm
    /// (`x.nil? and raise`) leaves the left-falsy narrow visible to the
    /// post-and statements.
    pub(super) fn check_and_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        let and = match node.as_and_node() {
            Some(n) => n,
            None => return Ty::UNTYPED,
        };
        let left = and.left();
        let left_ty = self.check_node(&left, hint);
        if left_ty == Ty::BOTTOM {
            // Left diverged (`raise "a" and foo`): same dead-code
            // shape as `check_or_node`. Skip the right arm and let the
            // stmt-list cutoff swallow whatever follows.
            return Ty::BOTTOM;
        }
        let (truthy, falsy) = self.analyze_condition(&left);

        // Env join still threads the full `left_ty` through arm_falsy
        // (mirroring the pre-change shape so narrowing flow into the
        // post-and env is unchanged — `if x.nil? && y.nil?; else; ...`
        // must keep the both-arms-evaluated env join). Value type is
        // computed separately as `right ∪ falsy_partition(left)`.
        let base = self.ctx.snapshot_scopes();
        let arm_truthy = self.with_branch_outcome(&base, &truthy, |checker| {
            checker.check_node(&and.right(), hint)
        });
        let arm_falsy = self.with_branch_outcome(&base, &falsy, |_| left_ty);
        let right_value_ty = arm_truthy.last_ty;
        let left_diverged = arm_falsy.last_ty == Ty::BOTTOM;
        let right_diverged = arm_truthy.last_ty == Ty::BOTTOM;
        let env_join_ty = self.join_arms_with_divergence(base, arm_truthy, arm_falsy);
        if left_diverged && right_diverged {
            return env_join_ty; // BOTTOM
        }
        // Statically-falsy left (`partition_truthy` empty) makes the right
        // arm unreachable too — Steep folds to the left's falsy partition
        // alone (type_construction.rb:1879-1886) even though the right is
        // still walked above for its side-effect diagnostics.
        let falsy_value = partition_falsy(left_ty, self.env.types()).unwrap_or(Ty::BOTTOM);
        if right_diverged || partition_truthy(left_ty, self.env.types()).is_none() {
            return falsy_value;
        }
        union_of(right_value_ty, falsy_value, self.env.types())
    }

    pub(super) fn infer_type<'pr>(&self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        match node {
            Node::IntegerNode { .. } => {
                if let Some(int_node) = node.as_integer_node() {
                    let integer = int_node.value();
                    let (negative, digits) = integer.to_u32_digits();
                    let val = prism_digits_to_decimal_string(negative, digits);
                    return self
                        .env
                        .types()
                        .intern(Type::Literal(Literal::Integer(val)));
                }
                self.env
                    .class_instance_type(self.env.names().builtins().integer)
            }
            Node::FloatNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().float),
            Node::StringNode { .. } => {
                if let Some(str_node) = node.as_string_node() {
                    let s = String::from_utf8_lossy(str_node.unescaped()).to_string();
                    return self.env.types().intern(Type::Literal(Literal::String(s)));
                }
                self.env
                    .class_instance_type(self.env.names().builtins().string)
            }
            Node::InterpolatedStringNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().string),
            Node::SymbolNode { .. } => {
                if let Some(sym_node) = node.as_symbol_node() {
                    let s = String::from_utf8_lossy(sym_node.unescaped()).to_string();
                    return self.env.types().intern(Type::Literal(Literal::Symbol(s)));
                }
                self.env
                    .class_instance_type(self.env.names().builtins().symbol)
            }
            Node::InterpolatedSymbolNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().symbol),
            // `def foo; end` and `def self.foo; end` both evaluate to a
            // `Symbol` (the method name). Steep types these at
            // `type_construction.rb:1073` (`:def`) and `:1146` (`:defs`)
            // as `AST::Builtin::Symbol.instance_type`. Prism represents
            // both forms as a single `DefNode` (distinguished by the
            // `receiver` field), so one arm covers both. This is the
            // read-only path; the side-effecting walker is driven by
            // `check_node`'s sibling arm.
            Node::DefNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().symbol),
            // `class C ... end` / `module M ... end` evaluate to `nil`
            // (Steep `type_construction.rb:1556` / `:1599`). Read-only
            // path; the side-effecting walker lives in `check_node`'s
            // sibling arm. Without these arms an unless/begin body
            // value position fell through to the catch-all and emitted
            // dev-only `Crema::NotImplementedYet`.
            Node::ClassNode { .. } | Node::ModuleNode { .. } => Ty::NIL,
            // `alias new old` / `alias $g_new $g_old` both evaluate to
            // `nil` (Steep `type_construction.rb:2608-2609` `:alias`
            // arm → `AST::Builtin.nil_type`). Read-only path; the
            // side-effecting walker fires from `check_node`'s sibling
            // arm (top-level) and `visit_program_node` / the class-body
            // walker (in-body). The arm is intentionally a no-`visit`
            // mirror of `check_node` so the statement-assertion gate
            // can read the value type without re-driving the side
            // effects already executed by the caller.
            Node::AliasMethodNode { .. } | Node::AliasGlobalVariableNode { .. } => Ty::NIL,
            // `class << expr; body; end` evaluates to `nil` (Steep
            // `type_construction.rb:1614, 1629` `:sclass` arm returns
            // `AST::Builtin.nil_type` on both the unsupported-shape and
            // the body-walked branches). Read-only path: the singleton
            // class scope push/pop and body check fire from
            // `check_node`'s sibling arm, so this arm just exposes the
            // value type to the statement-assertion gate without
            // re-driving the side effects.
            Node::SingletonClassNode { .. } => Ty::NIL,
            Node::NilNode { .. } => Ty::NIL,
            Node::TrueNode { .. } => self.env.types().intern(Type::Literal(Literal::Bool(true))),
            Node::FalseNode { .. } => self.env.types().intern(Type::Literal(Literal::Bool(false))),
            // `/(?<x>...)/ =~ s`: the expression's value is the inner
            // `=~` call's return type. Read-only projection — the call
            // diagnostics and the UNTYPED capture-local binding fire
            // from `check_node`'s sibling arm; this arm only exposes
            // the value type to the statement-assertion gate.
            Node::MatchWriteNode { .. } => node
                .as_match_write_node()
                .map(|mw| self.infer_type(&mw.call().as_node(), hint))
                .unwrap_or(Ty::UNTYPED),
            // Both regexp forms are `::Regexp` instances (non-generic).
            // Symmetric with the InterpolatedStringNode → String arm: the
            // interpolated parts don't change the literal's class.
            Node::RegularExpressionNode { .. } | Node::InterpolatedRegularExpressionNode { .. } => {
                self.env
                    .class_instance_type(self.env.names().builtins().regexp)
            }
            // Fixed-base-class literal/source nodes. Each returns a single
            // non-generic instance type so a receiver in this position
            // reports NoMethod instead of falling through to untyped.
            // `String` / `Integer` are pre-interned in `BuiltinNames`;
            // `Proc` / `Rational` / `Complex` / `Encoding` are not (rbs
            // `RBS::BuiltinNames` parity), so they're built inline via
            // `parse_absolute` like the Array arm's `::Array`.
            //
            // X-strings (`` `cmd` ``) and `__FILE__` are `::String`,
            // symmetric with the InterpolatedString → String arm.
            Node::XStringNode { .. }
            | Node::InterpolatedXStringNode { .. }
            | Node::SourceFileNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().string),
            Node::SourceLineNode { .. } => self
                .env
                .class_instance_type(self.env.names().builtins().integer),
            Node::LambdaNode { .. } => self.lambda_literal_type(node, hint),
            Node::RationalNode { .. } => self
                .env
                .class_instance_type(self.env.names().parse_type_name("::Rational")),
            Node::ImaginaryNode { .. } => self
                .env
                .class_instance_type(self.env.names().parse_type_name("::Complex")),
            Node::SourceEncodingNode { .. } => self
                .env
                .class_instance_type(self.env.names().parse_type_name("::Encoding")),
            // `defined?(x)`, `$1` (NumberedReferenceReadNode), and `$&`
            // (BackReferenceReadNode) all evaluate to `String?`
            // (`::String | nil`): `defined?` returns the kind string or
            // nil, and the regexp match globals are nil until a match
            // sets them (rbs `global_variables.rbs` `$&: String?` /
            // `$1: String?`). Read-only like the sibling literal arms;
            // `defined?`'s argument is not evaluated at runtime and the
            // receiver-position side-effect walk lives in `check_call`'s
            // `visit(&receiver)`, so no `check_node` arm is needed.
            Node::DefinedNode { .. }
            | Node::NumberedReferenceReadNode { .. }
            | Node::BackReferenceReadNode { .. } => {
                let string = self
                    .env
                    .class_instance_type(self.env.names().builtins().string);
                union_of(string, Ty::NIL, self.env.types())
            }
            // `(a..b)` is a `::Range[E]`. Synthesize `E` by widening each
            // bound's literal to its base class and unioning the two, so
            // `(1..4)` is `Range[Integer]`, not `Range[1 | 4]` (same
            // `synthesize_element_union` the Array / Hash no-hint paths
            // use). An absent bound (beginless `(..4)` / endless `(1..)`)
            // contributes `nil`, matching Steep: `(1..)` is
            // `Range[Integer | nil]`, `(..4)` is `Range[nil | Integer]`.
            Node::RangeNode { .. } => {
                let name = self.env.names().builtins().range;
                let elem = match node.as_range_node() {
                    Some(range) => {
                        let bounds = [range.left(), range.right()].map(|bound| match bound {
                            Some(bound) => self.infer_type(&bound, None),
                            None => Ty::NIL,
                        });
                        self.synthesize_element_union(&bounds)
                    }
                    None => Ty::UNTYPED,
                };
                self.env.types().intern(Type::ClassInstance {
                    name,
                    args: vec![elem],
                })
            }
            Node::ArrayNode { .. } => {
                let name = self.env.names().parse_type_name("::Array");
                // Bidirectional (ADR-0012 parity with the HashNode branch):
                // when the hint reduces to a Tuple of matching arity (no
                // splat, no ambiguity), synthesize a `Type::Tuple`;
                // otherwise, when the hint reduces to `Array[E]`,
                // propagate `E` into the literal. Without a usable hint
                // the original Array[untyped] shape stays in place so
                // existing call-site behavior is unchanged.
                if let Some(array_node) = node.as_array_node() {
                    let elements: Vec<_> = array_node.elements().iter().collect();
                    let has_splat = elements.iter().any(|el| el.as_splat_node().is_some());
                    if !has_splat
                        && let Some(hint_ty) = hint
                        && let Some(tuple_elem_hints) =
                            self.array_tuple_hint(hint_ty, elements.len())
                    {
                        return fold_array_tuple_type(
                            self.env.types(),
                            &tuple_elem_hints,
                            &elements,
                            |el, h| self.infer_type(el, h),
                        );
                    }
                    // Splat + Tuple hint path (mirror of `check_array_node`
                    // — see the prose there). `infer_type` doesn't emit
                    // diagnostics, but call-site receiver_type / hint
                    // propagation still reads the synthesized `Type::Tuple`
                    // when the literal is reached via `infer_type` (e.g.
                    // call-site arg synthesis with a Tuple param hint).
                    if has_splat
                        && let Some(hint_ty) = hint
                        && let Some(tuple_elem_hints) = self.array_tuple_hint_loose(hint_ty)
                    {
                        if let Some(splat_idx) = single_splat_index(&elements) {
                            let scalars_before = splat_idx;
                            let scalars_after = elements.len() - splat_idx - 1;
                            if scalars_before + scalars_after <= tuple_elem_hints.len()
                                && let Some(splat) = elements[splat_idx].as_splat_node()
                                && let Some(expr) = splat.expression()
                            {
                                // Same walk order as `check_array_node` — leading
                                // scalars first, splat next — so Ruby's
                                // left-to-right evaluation is preserved when
                                // `infer_type` reaches this literal as a value
                                // expression (e.g. nested in a call-site hint).
                                let leading: Vec<Ty> = (0..scalars_before)
                                    .map(|i| {
                                        self.infer_type(&elements[i], Some(tuple_elem_hints[i]))
                                    })
                                    .collect();
                                let splat_inner_ty = self.infer_type(&expr, None);
                                let splat_slot_count =
                                    tuple_elem_hints.len() - scalars_before - scalars_after;
                                let slot_hints: Vec<Ty> = tuple_elem_hints
                                    [scalars_before..scalars_before + splat_slot_count]
                                    .to_vec();
                                if let Some(splat_slot_types) =
                                    self.decide_splat_slot_types(splat_inner_ty, &slot_hints)
                                {
                                    let mut out = Vec::with_capacity(tuple_elem_hints.len());
                                    out.extend(leading);
                                    out.extend(splat_slot_types);
                                    for j in 0..scalars_after {
                                        let hint_idx = tuple_elem_hints.len() - scalars_after + j;
                                        out.push(self.infer_type(
                                            &elements[splat_idx + 1 + j],
                                            Some(tuple_elem_hints[hint_idx]),
                                        ));
                                    }
                                    return self.env.types().intern(Type::Tuple(out));
                                }
                                let mut raw = Vec::with_capacity(elements.len());
                                raw.extend(leading);
                                raw.push(self.splat_element_type(splat_inner_ty));
                                for j in 0..scalars_after {
                                    raw.push(self.infer_type(&elements[splat_idx + 1 + j], None));
                                }
                                let element_ty = self.synthesize_element_union(&raw);
                                return self.env.types().intern(Type::ClassInstance {
                                    name,
                                    args: vec![element_ty],
                                });
                            }
                        }
                        let raw: Vec<Ty> = elements
                            .iter()
                            .map(|el| match el.as_splat_node() {
                                Some(splat) => match splat.expression() {
                                    Some(expr) => {
                                        self.splat_element_type(self.infer_type(&expr, None))
                                    }
                                    None => Ty::UNTYPED,
                                },
                                None => self.infer_type(el, None),
                            })
                            .collect();
                        let element_ty = self.synthesize_element_union(&raw);
                        return self.env.types().intern(Type::ClassInstance {
                            name,
                            args: vec![element_ty],
                        });
                    }
                    if let Some(elem_hint) = hint.and_then(|h| self.array_element_hint(h)) {
                        // Array[E] hint path under `infer_type` — read-only
                        // mirror of `check_array_node`'s Array[E] arm.
                        // Splat elements bypass the SplatNode catch-all
                        // (would emit `NotImplementedYet` + return UNTYPED)
                        // and project their element contribution via
                        // `splat_element_type`. Scalars carry the elem hint
                        // for Stage 1 bidirectional propagation.
                        // `fold_hint_element_union` widens literals and
                        // dedupes so the resulting `Array[E']` matches
                        // Steep's hover shape (e.g. `Array[(Integer |
                        // String)]` for a mixed-element splat,
                        // `Array[Integer]` when every element collapses
                        // to Integer), but keeps untyped as a Union
                        // member so the outer subtype gate still catches
                        // concrete-vs-hint mismatches.
                        let element_ty = if elements.is_empty() {
                            elem_hint
                        } else {
                            let raw: Vec<Ty> = elements
                                .iter()
                                .map(|el| match el.as_splat_node() {
                                    Some(splat) => match splat.expression() {
                                        Some(expr) => {
                                            self.splat_element_type(self.infer_type(&expr, None))
                                        }
                                        None => Ty::UNTYPED,
                                    },
                                    None => self.infer_type(el, Some(elem_hint)),
                                })
                                .collect();
                            self.fold_hint_element_union(&raw)
                        };
                        return self.env.types().intern(Type::ClassInstance {
                            name,
                            args: vec![element_ty],
                        });
                    }
                    // Synthesize from the literal contents (read-only
                    // mirror of `check_array_node`). `array_from_raw` only
                    // synthesizes when `no_hint`; an unusable hint or
                    // empty literal stays at `Array[untyped]`. Splat
                    // elements bypass the SplatNode catch-all (returns
                    // UNTYPED) and project their element contribution
                    // via `splat_element_type`.
                    let raw: Vec<Ty> = elements
                        .iter()
                        .map(|el| match el.as_splat_node() {
                            Some(splat) => match splat.expression() {
                                Some(expr) => self.splat_element_type(self.infer_type(&expr, None)),
                                None => Ty::UNTYPED,
                            },
                            None => self.infer_type(el, None),
                        })
                        .collect();
                    return self.array_from_raw(&raw, hint.is_none());
                }
                self.env.types().intern(Type::ClassInstance {
                    name,
                    args: vec![Ty::UNTYPED],
                })
            }
            // `HashNode` (`{a: 1}`) and `KeywordHashNode` (trailing
            // `key: v` in array literals or call args) share an identical
            // `AssocNode`/`AssocSplatNode` element shape. Steep handles
            // both under one `when :hash, :kwargs` arm
            // (`type_construction.rb:1466`); we do the same so a
            // KeywordHashNode reached via the read-only path (e.g.
            // `[1, k: v]`'s last element under ArrayNode element-hint
            // propagation) synthesizes the same `Hash[K, V]` / Record
            // shape. The call-site (positional → kwargs) split lives in
            // `calls.rs` and is unaffected.
            Node::HashNode { .. } | Node::KeywordHashNode { .. } => {
                if let Some(elements) = hash_or_keyword_hash_elements(node) {
                    // Bidirectional (ADR-0012): hint may be wrapped in
                    // Optional/Union; pick the best Record candidate
                    // (Steep `pick_one_of` parity for multi-candidate
                    // disjunctions, relaxed for single-candidate cases).
                    // Extras are discarded here — only call-site synthesis
                    // emits UnknownRecordKey; other `infer_type` callers
                    // don't have a mutable diagnostic sink at hand.
                    if let Some(hint_ty) = hint {
                        let candidates = self.expand_hint_candidates(hint_ty);
                        if let Some(record_hint) =
                            self.pick_record_hint_from_candidates(&elements, &candidates)
                        {
                            let mut extras = Vec::new();
                            if let Some(ty) =
                                self.synthesize_hash_as_record(&elements, record_hint, &mut extras)
                            {
                                return ty;
                            }
                        }
                        for c in candidates {
                            if let Some(hash_ty) = self.synthesize_hash_with_hint(&elements, c) {
                                return hash_ty;
                            }
                        }
                    }
                    // With genuinely no hint, synthesize `Hash[K, V]` from
                    // the literal's key / value contents. A kwsplat
                    // (`**h`) member contributes its value's K/V args via
                    // `absorb_kwsplat_value` (Steep `type_hash` no-hint
                    // path uses `hint_hash = Hash[any, any]`, mirrored
                    // here as `Ty::UNTYPED`). Empty literal still
                    // collapses to `Hash[untyped, untyped]`.
                    if hint.is_none() && !elements.is_empty() {
                        let mut keys: Vec<Ty> = Vec::with_capacity(elements.len());
                        let mut vals: Vec<Ty> = Vec::with_capacity(elements.len());
                        for e in &elements {
                            if let Some(assoc) = e.as_assoc_node() {
                                keys.push(self.infer_type(&assoc.key(), None));
                                vals.push(self.infer_type(&assoc.value(), None));
                            } else if let Some(splat) = e.as_assoc_splat_node()
                                && let Some(value) = splat.value()
                            {
                                self.absorb_kwsplat_value(
                                    &value,
                                    Ty::UNTYPED,
                                    Ty::UNTYPED,
                                    &mut keys,
                                    &mut vals,
                                );
                            }
                        }
                        if keys.is_empty() && vals.is_empty() {
                            return self.hash_untyped_untyped();
                        }
                        let key_ty = self.synthesize_element_union(&keys);
                        let val_ty = self.synthesize_element_union(&vals);
                        return self.hash_instance(key_ty, val_ty);
                    }
                }
                self.hash_untyped_untyped()
            }
            Node::SelfNode { .. } => self.self_receiver_type(),
            Node::CallNode { .. } => {
                if let Some(call) = node.as_call_node() {
                    // Same cache short-circuit as check_node's CallNode
                    // arm. `infer_type` is `&self`, so it never writes to
                    // the cache; it only reads what an earlier check_node
                    // pass already stored.
                    if let Some(key) = self.try_pure_key(&call.as_node())
                        && let Some(ty) = self
                            .lookup_pure_overlay(&key)
                            .or_else(|| self.ctx.pure_call_env().get(&key))
                    {
                        return ty;
                    }
                    let resolved = ResolvedCall::resolve(self, &call);
                    self.infer_call_return_type(&call, hint, &resolved)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::LocalVariableReadNode { .. } => {
                if let Some(local) = node.as_local_variable_read_node() {
                    let name_bytes = local.name().as_slice();
                    // Steep parity (`lib/steep/type_construction.rb:818-831`):
                    // `SPECIAL_LVAR_NAMES` reads short-circuit to untyped, so
                    // an overlay binding can't override the cast idiom.
                    if super::is_special_lvar_name(name_bytes) {
                        return Ty::UNTYPED;
                    }
                    let name_str = String::from_utf8_lossy(name_bytes);
                    let name = self.checker_names().intern(&name_str);
                    // Overlay (transient, set by `with_overlay`) takes precedence.
                    // This lets `infer_call_return_type` bind block params for
                    // block body inference without pushing a real scope —
                    // `infer_type` remains `&self`.
                    self.lookup_overlay(name)
                        .or_else(|| self.lookup_local_variable_for_read(name))
                        .unwrap_or(Ty::UNTYPED)
                } else {
                    Ty::UNTYPED
                }
            }
            // The implicit `it` block param (Ruby 3.4+). Prism gives it no name,
            // so it binds under the fixed local name `it` in `setup_block_scope`
            // (real scope) and resolves here exactly like a named read.
            Node::ItLocalVariableReadNode { .. } => {
                let name = self.checker_names().intern("it");
                self.lookup_overlay(name)
                    .or_else(|| self.lookup_local_variable_for_read(name))
                    .unwrap_or(Ty::UNTYPED)
            }
            Node::GlobalVariableReadNode { .. } => {
                if let Some(global) = node.as_global_variable_read_node() {
                    let name_str = String::from_utf8_lossy(global.name().as_slice());
                    self.env.lookup_global(&name_str).unwrap_or(Ty::UNTYPED)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::InstanceVariableReadNode { .. } => {
                if let Some(ivar) = node.as_instance_variable_read_node() {
                    let name_str = String::from_utf8_lossy(ivar.name().as_slice());
                    let var_sym = self.env.names().intern_symbol(&name_str);
                    resolve_ivar_at_self(self, var_sym).unwrap_or(Ty::UNTYPED)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::ClassVariableReadNode { .. } => {
                if let Some(cvar) = node.as_class_variable_read_node() {
                    let name_str = String::from_utf8_lossy(cvar.name().as_slice());
                    let var_sym = self.env.names().intern_symbol(&name_str);
                    resolve_cvar_at_self(self, var_sym).unwrap_or(Ty::UNTYPED)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::ConstantReadNode { .. } => {
                if let Some(constant) = node.as_constant_read_node() {
                    let name_str = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                    self.resolve_constant_read(&name_str, node)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::ConstantPathNode { .. } => {
                if let Some(path) = node.as_constant_path_node() {
                    self.resolve_constant_path_node(&path)
                } else {
                    Ty::UNTYPED
                }
            }
            Node::StatementsNode { .. } => {
                if let Some(statements) = node.as_statements_node() {
                    self.infer_statements_with_hint(&statements, hint)
                } else {
                    Ty::UNTYPED
                }
            }
            // Read-only mirror of `check_begin_node`'s value computation: a
            // `begin ... end` receiver (`begin; 1; end.foo`) reaches here
            // through `infer_receiver_type`. Walk body / rescue arms / else
            // via `infer_type` (no narrowing, no scope snapshots, no
            // diagnostics — those fire on the `check_node` path) and fold
            // them through the shared `begin_value_type`, so the receiver
            // position and the assignment-RHS position agree on the type.
            Node::BeginNode { .. } => {
                let begin = match node.as_begin_node() {
                    Some(b) => b,
                    None => return Ty::UNTYPED,
                };
                let infer_stmts =
                    |this: &Self, stmts: Option<ruby_prism::StatementsNode<'pr>>| match stmts {
                        Some(s) => this.infer_type(&s.as_node(), hint),
                        None => Ty::NIL,
                    };
                let body_ty = infer_stmts(self, begin.statements());
                let mut rescue_tys: Vec<Ty> = Vec::new();
                let mut rescue = begin.rescue_clause();
                while let Some(clause) = rescue {
                    rescue_tys.push(infer_stmts(self, clause.statements()));
                    rescue = clause.subsequent();
                }
                let else_ty = begin
                    .else_clause()
                    .map(|e| infer_stmts(self, e.statements()));
                begin_value_type(self.env.types(), body_ty, &rescue_tys, else_ty)
            }
            // Read-only mirror of `check_node`'s ParenthesesNode arm: a
            // parenthesized receiver (`(1..4).foo`) reaches here through
            // `infer_receiver_type`, so unwrap to the body (a StatementsNode,
            // whose arm infers the last expression). Empty `()` is `nil`.
            Node::ParenthesesNode { .. } => {
                if let Some(par) = node.as_parentheses_node() {
                    if let Some(body) = par.body() {
                        self.infer_type(&body, hint)
                    } else {
                        Ty::NIL
                    }
                } else {
                    Ty::UNTYPED
                }
            }
            // Divergent control-flow leaves: read-only mirror of the
            // check_node arm. Without this, the surrounding control-flow
            // `infer_type` arms (IfNode / CaseNode / OrNode / AndNode)
            // fold `return` / `break` / `next` as `Ty::UNTYPED` via the
            // default arm and `union_of(untyped, T)` degenerates to
            // untyped — silencing NoMethod on direct-receiver shapes
            // like `(if cond then 1 else return end).foo`.
            Node::ReturnNode { .. } | Node::BreakNode { .. } | Node::NextNode { .. } => Ty::BOTTOM,
            // Read-only mirror of each control-flow `check_*_node` arm:
            // narrowing / scope snapshots / diagnostics fire on the
            // check_node path; the receiver path needs only the value
            // type so that `(if c then 1 else 2 end).foo` and
            // `r = if c then 1 else 2 end; r.foo` agree on the union.
            // Divergent arms (BOTTOM) are absorbed by `union_of` /
            // `union_of_many` (BOTTOM is the union identity).
            Node::IfNode { .. } => {
                let if_node = match node.as_if_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let predicate = if_node.predicate();
                let narrow_pair = self.bare_truthy_falsy_pair_read_only(&predicate);
                let truthy_overlay = narrow_pair
                    .as_ref()
                    .and_then(|(sc, t, _)| t.map(|ty| (sc, ty)));
                let falsy_overlay = narrow_pair
                    .as_ref()
                    .and_then(|(sc, _, f)| f.map(|ty| (sc, ty)));
                let arm_a = match if_node.statements() {
                    Some(s) => {
                        self.infer_arm_with_optional_overlay(&s.as_node(), hint, truthy_overlay)
                    }
                    None => Ty::NIL,
                };
                // `subsequent()` is either an ElseNode (final else) or
                // another IfNode (an elsif chain). Recurse on IfNode;
                // unwrap ElseNode to its inner statements — `infer_type`
                // has no ElseNode arm of its own (mirrors check_else_node
                // for the read-only path).
                let arm_b = match if_node.subsequent() {
                    Some(sub) => {
                        let body = if let Some(else_node) = sub.as_else_node() {
                            else_node.statements().map(|s| s.as_node())
                        } else {
                            Some(sub)
                        };
                        match body {
                            Some(n) => {
                                self.infer_arm_with_optional_overlay(&n, hint, falsy_overlay)
                            }
                            None => Ty::NIL,
                        }
                    }
                    None => Ty::NIL,
                };
                union_of(arm_a, arm_b, self.env.types())
            }
            Node::UnlessNode { .. } => {
                let unless_node = match node.as_unless_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let predicate = unless_node.predicate();
                // Unless flips the arm roles: body runs on the falsy side
                // of the predicate, else on the truthy side.
                let narrow_pair = self.bare_truthy_falsy_pair_read_only(&predicate);
                let truthy_overlay = narrow_pair
                    .as_ref()
                    .and_then(|(sc, t, _)| t.map(|ty| (sc, ty)));
                let falsy_overlay = narrow_pair
                    .as_ref()
                    .and_then(|(sc, _, f)| f.map(|ty| (sc, ty)));
                let arm_body = match unless_node.statements() {
                    Some(s) => {
                        self.infer_arm_with_optional_overlay(&s.as_node(), hint, falsy_overlay)
                    }
                    None => Ty::NIL,
                };
                // Unwrap the ElseNode wrapper directly here: `infer_type`
                // has no ElseNode arm of its own (mirrors the IfNode
                // subsequent handling above). Falling through to
                // `infer_type(&e.as_node())` would route to the default
                // arm and silently return `Ty::UNTYPED`, collapsing the
                // union to untyped for direct-receiver shapes.
                let arm_else = match unless_node.else_clause() {
                    Some(e) => match e.statements() {
                        Some(s) => {
                            self.infer_arm_with_optional_overlay(&s.as_node(), hint, truthy_overlay)
                        }
                        None => Ty::NIL,
                    },
                    None => Ty::NIL,
                };
                union_of(arm_body, arm_else, self.env.types())
            }
            Node::CaseNode { .. } => {
                let case = match node.as_case_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let mut tys: Vec<Ty> = Vec::new();
                // Read-only mirror of `check_case_node`'s lvar case
                // narrowing. Method return checks use `infer_type`
                // directly, so branch value synthesis must see the same
                // literal/class residue even though this path cannot
                // mutate the real scope chain.
                let scrutinee = case.predicate().and_then(|predicate| {
                    if let Some(write) = predicate.as_local_variable_write_node() {
                        if super::is_special_lvar_name(write.name().as_slice()) {
                            return None;
                        }
                        let name_str = String::from_utf8_lossy(write.name().as_slice());
                        let name = self.checker_names().intern(&name_str);
                        return Some((
                            Scrutinee::Lvar(name),
                            self.infer_type(&write.value(), None),
                        ));
                    }
                    self.scrutinee_and_current_ty(&predicate)
                });
                let mut iter_falsy_ty = scrutinee.as_ref().map(|(_, ty)| *ty);
                let mut residue_alive = scrutinee.is_some();
                for cond in case.conditions().iter() {
                    if let Some(when_node) = cond.as_when_node() {
                        let when_conds: Vec<Node<'pr>> = when_node.conditions().iter().collect();
                        let target_union = if scrutinee.is_some() {
                            let mut targets: Vec<Ty> = Vec::with_capacity(when_conds.len());
                            let mut ok = true;
                            for c in &when_conds {
                                if let Some(t) = self.case_when_literal_target(c) {
                                    targets.push(t);
                                    continue;
                                }
                                match self.case_when_class_target(c) {
                                    Some(t) => targets.push(t),
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if ok && !targets.is_empty() {
                                Some(union_of_many(&targets, self.env.types()))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let truthy_ty = target_union.map(|target| {
                            crate::narrowing::narrow(
                                iter_falsy_ty.expect("target implies scrutinee type"),
                                target,
                                self.env,
                            )
                        });
                        let arm_ty = match when_node.statements() {
                            Some(s) => match (scrutinee.as_ref(), truthy_ty) {
                                (Some((scrutinee, _)), Some(ty)) => self
                                    .infer_type_with_scrutinee_overlay(
                                        &s.as_node(),
                                        hint,
                                        scrutinee,
                                        ty,
                                    ),
                                _ => self.infer_type(&s.as_node(), hint),
                            },
                            None => Ty::NIL,
                        };
                        tys.push(arm_ty);
                        match target_union {
                            Some(target) if residue_alive => {
                                iter_falsy_ty = Some(crate::narrowing::subtract(
                                    iter_falsy_ty.expect("target implies scrutinee type"),
                                    target,
                                    self.env,
                                ));
                            }
                            Some(_) => {}
                            None if scrutinee.is_some() => residue_alive = false,
                            None => {}
                        }
                    }
                }
                let case_exhausted =
                    residue_alive && iter_falsy_ty.map(|t| t == Ty::BOTTOM).unwrap_or(false);

                // Match `check_case_node`'s exhaustiveness test for
                // implicit fallthrough: an exhaustive case without an
                // explicit else has no reachable `else; nil` arm. An
                // explicit else is still evaluated and joined, preserving
                // value synthesis and diagnostics inside that body.
                if let Some(e) = case.else_clause() {
                    let else_ty = match e.statements() {
                        Some(s) => match (scrutinee.as_ref(), residue_alive, iter_falsy_ty) {
                            (Some((scrutinee, _)), true, Some(ty)) => self
                                .infer_type_with_scrutinee_overlay(
                                    &s.as_node(),
                                    hint,
                                    scrutinee,
                                    ty,
                                ),
                            _ => self.infer_type(&s.as_node(), hint),
                        },
                        None => Ty::NIL,
                    };
                    if !case_exhausted {
                        tys.push(else_ty);
                    }
                } else if !case_exhausted {
                    tys.push(Ty::NIL);
                }
                union_of_many(&tys, self.env.types())
            }
            // Read-only mirror of `check_case_match_node`: same arm
            // narrowing and residue rules, value type only. Without an
            // `else` there is no implicit `nil` arm — an unmatched
            // case/in raises NoMatchingPatternError.
            Node::CaseMatchNode { .. } => {
                let case = match node.as_case_match_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let scrutinee = case
                    .predicate()
                    .and_then(|p| self.scrutinee_and_current_ty(&p));
                let mut iter_falsy_ty = scrutinee.as_ref().map(|(_, t)| *t);
                let mut residue_alive = scrutinee.is_some();
                let mut tys: Vec<Ty> = Vec::new();
                for cond in case.conditions().iter() {
                    let in_node = match cond.as_in_node() {
                        Some(n) => n,
                        None => continue,
                    };
                    let (pattern, guard) = self.in_pattern_and_guard(in_node.pattern());
                    let target = if scrutinee.is_some() {
                        self.pattern_target_ty(&pattern)
                    } else {
                        None
                    };
                    let truthy_ty = target.map(|t| {
                        crate::narrowing::narrow(
                            iter_falsy_ty.expect("target implies ty"),
                            t,
                            self.env,
                        )
                    });
                    let arm_ty = match in_node.statements() {
                        Some(stmts) => match (scrutinee.as_ref(), truthy_ty) {
                            (Some((s, _)), Some(ty)) => self.infer_type_with_scrutinee_overlay(
                                &stmts.as_node(),
                                hint,
                                s,
                                ty,
                            ),
                            _ => self.infer_type(&stmts.as_node(), hint),
                        },
                        None => Ty::NIL,
                    };
                    tys.push(arm_ty);
                    match target {
                        Some(_) if guard.is_some() || !residue_alive => {}
                        Some(t) => {
                            iter_falsy_ty = Some(crate::narrowing::subtract(
                                iter_falsy_ty.expect("target implies ty"),
                                t,
                                self.env,
                            ));
                        }
                        None if scrutinee.is_some() => residue_alive = false,
                        None => {}
                    }
                }
                if let Some(e) = case.else_clause() {
                    let else_ty = match e.statements() {
                        Some(stmts) => match (scrutinee.as_ref(), residue_alive, iter_falsy_ty) {
                            (Some((s, _)), true, Some(ty)) => self
                                .infer_type_with_scrutinee_overlay(&stmts.as_node(), hint, s, ty),
                            _ => self.infer_type(&stmts.as_node(), hint),
                        },
                        None => Ty::NIL,
                    };
                    tys.push(else_ty);
                }
                union_of_many(&tys, self.env.types())
            }
            Node::WhileNode { .. } => {
                // Both the regular and do-while modifier forms evaluate
                // to nil at runtime (and in Steep's
                // `type_construction.rb:2337` AST::Builtin.nil_type).
                // Narrowing differs between the two forms (irrelevant
                // here — infer_type has no env to narrow).
                if node.as_while_node().is_some() {
                    Ty::NIL
                } else {
                    Ty::UNTYPED
                }
            }
            Node::UntilNode { .. } => {
                if node.as_until_node().is_some() {
                    Ty::NIL
                } else {
                    Ty::UNTYPED
                }
            }
            Node::ForNode { .. } => {
                let for_node = match node.as_for_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                self.infer_type(&for_node.collection(), hint)
            }
            Node::OrNode { .. } => {
                let or = match node.as_or_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let left_ty = self.infer_type(&or.left(), None);
                let right_hint = partition_truthy(left_ty, self.env.types());
                let right_ty = self.infer_type(&or.right(), right_hint);
                let arm_truthy = partition_truthy(left_ty, self.env.types()).unwrap_or(Ty::BOTTOM);
                union_of(arm_truthy, right_ty, self.env.types())
            }
            Node::AndNode { .. } => {
                let and = match node.as_and_node() {
                    Some(n) => n,
                    None => return Ty::UNTYPED,
                };
                let left_ty = self.infer_type(&and.left(), hint);
                let right_ty = self.infer_type(&and.right(), hint);
                let arm_falsy = partition_falsy(left_ty, self.env.types()).unwrap_or(Ty::BOTTOM);
                union_of(arm_falsy, right_ty, self.env.types())
            }
            // `yield`'s value is the current method's block return type
            // (`ctx.method_type().block`, the same source
            // `visit_yield_node` checks arguments against). Read-only
            // synth path: the side-effecting yield checks
            // (UnexpectedYield, argument types) fire on the `check_node`
            // / `visit_yield_node` path. No declared block → untyped, so
            // a `yield` receiver in a block-less method stays untyped
            // (the `UnexpectedYield` flag is the actionable signal there).
            Node::YieldNode { .. } => self
                .ctx
                .method_type()
                .and_then(|mt| mt.block.as_ref())
                .map(|b| b.return_type())
                .unwrap_or(Ty::UNTYPED),
            // `super` / `super(...)` resolve the same-named method starting
            // after the defining class in the ancestor linearization. The
            // value position (`x = super`) reaches here through `check_node`'s
            // `_ => infer_type` delegation. `super(args)` (SuperNode) selects
            // the super-target overload by its args so the return type tracks
            // the chosen overload; bare `super` (ForwardingSuperNode) has no
            // args to inspect and keeps the first-overload projection.
            Node::SuperNode { .. } => {
                let args = node
                    .as_super_node()
                    .map(|s| self.collect_arguments_from(s.arguments(), s.block().is_some()));
                self.super_return_type(args.as_ref())
            }
            Node::ForwardingSuperNode { .. } => self.super_return_type(None),
            // ivar / cvar / gvar writes in value position (`x = (@x = 1)`,
            // method body whose last expression is a write). The
            // assignment value is the RHS expression's type — mirrors
            // Steep `type_construction.rb` `ivasgn` / `cvasgn` / `gvasgn`
            // returning `add_typing(node, type: rhs_type)`. Side-effect
            // checks (subtype gate against the declared type, Unknown*
            // emission) fire via `check_node` / the Visit override and
            // are not duplicated here.
            Node::InstanceVariableWriteNode { .. } => node
                .as_instance_variable_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // ivar `||=` / `&&=` / `+=` in value position. The Steep
            // `or_asgn` / `op_asgn` desugar (`type_construction.rb`
            // `:2395-2428` / `:858-908`) reduces to `ivasgn`, which
            // adds a typing of the rhs type. Side-effect diagnostics
            // (UnknownInstanceVariable / IncompatibleAssignment /
            // ArgumentTypeMismatch) fire via the visitor's
            // `visit_instance_variable_{or,and,operator}_write_node`
            // when reached through `check_node`; the value-position
            // path here is read-only.
            Node::InstanceVariableOrWriteNode { .. } => node
                .as_instance_variable_or_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::InstanceVariableAndWriteNode { .. } => node
                .as_instance_variable_and_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `@x += y` value-position type is the operator method's
            // return type (Steep desugars to `@x = @x.+(y)`, whose
            // `ivasgn` step adds a typing of the *send* return type —
            // `type_construction.rb:858-879` / `:2824-2840`). For the
            // common case `@x: T` with `T#+: ... -> T`, that's `T`;
            // the read-only `infer_type` cannot replay operator
            // dispatch, so divergent overloads (`T#+: (U) -> V`) are
            // smoothed to `T` here. The visitor path
            // (`visit_instance_variable_operator_write_node`) still
            // runs the real dispatch and emits `IncompatibleAssignment`
            // when the actual send return type doesn't subtype `T`.
            Node::InstanceVariableOperatorWriteNode { .. } => node
                .as_instance_variable_operator_write_node()
                .map(|w| {
                    // Walk the RHS for its side-effect diagnostics —
                    // mirrors plain `InstanceVariableWriteNode`'s
                    // value-position arm.
                    self.infer_type(&w.value(), None);
                    let name_str = String::from_utf8_lossy(w.name().as_slice()).into_owned();
                    let var_sym = self.env.names().intern_symbol(&name_str);
                    resolve_ivar_at_self(self, var_sym).unwrap_or(Ty::UNTYPED)
                })
                .unwrap_or(Ty::UNTYPED),
            // `foo[idx] ||= v` / `&&=` in value position. Matches
            // `check_index_compound_write`'s Or/And narrowing
            // (`or_write_value_ty` / `and_write_value_ty`) via the
            // read-only sibling `infer_index_compound_write_value_ty`, so
            // the value type agrees with the live `check_node` path
            // (`infer_type_index_or_and_write_narrowing_gap` todo — this
            // arm previously returned the bare rhs type, diverging from
            // `check_node`'s narrowed union). Side-effect diagnostics
            // ([] / []= dispatch, NoMethod, ArgumentTypeMismatch) still
            // fire only through the visitor path (`visit_index_or_write_
            // node` / `visit_index_and_write_node`) when reached via
            // `check_node`; this arm stays read-only.
            Node::IndexOrWriteNode { .. } => node
                .as_index_or_write_node()
                .map(|w| {
                    self.infer_index_compound_write_value_ty(
                        w.receiver(),
                        w.arguments(),
                        w.block().is_some(),
                        &w.value(),
                        true,
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            Node::IndexAndWriteNode { .. } => node
                .as_index_and_write_node()
                .map(|w| {
                    self.infer_index_compound_write_value_ty(
                        w.receiver(),
                        w.arguments(),
                        w.block().is_some(),
                        &w.value(),
                        false,
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            // `foo.attr ||= v` / `&&=` in value position — read-only
            // call-attribute siblings of the Index arms above, via
            // `infer_call_compound_write_value_ty`. Side-effect
            // diagnostics still fire only through the visitor path.
            Node::CallOrWriteNode { .. } => node
                .as_call_or_write_node()
                .map(|w| {
                    self.infer_call_compound_write_value_ty(
                        w.receiver(),
                        w.is_safe_navigation(),
                        w.read_name().as_slice(),
                        &w.value(),
                        true,
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            Node::CallAndWriteNode { .. } => node
                .as_call_and_write_node()
                .map(|w| {
                    self.infer_call_compound_write_value_ty(
                        w.receiver(),
                        w.is_safe_navigation(),
                        w.read_name().as_slice(),
                        &w.value(),
                        false,
                    )
                })
                .unwrap_or(Ty::UNTYPED),
            // `foo.attr += v` in value position — same read-only
            // contract as the Index `+=` arm below: the result would
            // require a reader dispatch, so B direction stays UNTYPED
            // and the rhs is walked for its own side effects.
            Node::CallOperatorWriteNode { .. } => node
                .as_call_operator_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            // `foo[idx] += v` in value position. Unlike the ivar `+=`
            // arm, the index family has no declared cell type — so the
            // value-position result type would have to come from
            // dispatching `[]` (its return type after the operator).
            // That would make `infer_type` non-read-only, which breaks
            // the contract the value-position arms rely on. B direction
            // returns `Ty::UNTYPED` here; the visitor path still runs
            // the full three-stage dispatch and surfaces every
            // diagnostic. The rhs is walked for its own side effects
            // (mirrors the ivar `+=` arm).
            Node::IndexOperatorWriteNode { .. } => node
                .as_index_operator_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            // cvar plain write value-position oracle. Unlike the
            // ivar/gvar arms directly below (unconditional RHS type),
            // this mirrors `check_class_variable_write`'s Steep
            // `:cvasgn` mismatch/undeclared-falls-back-to-untyped
            // nuance (`type_construction.rb:2580-2591`,
            // `fallback_to_any` fires on both the mismatch branch and
            // the `var_type` nil / undeclared branch) — without it,
            // this fallback re-walk (triggered whenever the live
            // path's `Ty::UNTYPED` result reaches
            // `check_def_node_in_current_context`'s `ty.is_untyped()`
            // re-derive) would silently overwrite the live path's
            // correct `untyped` with the RHS type, resurfacing the same
            // spurious `MethodBodyTypeMismatch` cascade
            // `check_class_variable_write`'s doc comment describes.
            Node::ClassVariableWriteNode { .. } => node
                .as_class_variable_write_node()
                .map(|w| {
                    let rhs_ty = self.infer_type(&w.value(), hint);
                    let name_str = String::from_utf8_lossy(w.name().as_slice()).into_owned();
                    let var_sym = self.env.names().intern_symbol(&name_str);
                    match resolve_cvar_at_self(self, var_sym) {
                        Some(lhs) if self.subtyper().check(rhs_ty, lhs) => rhs_ty,
                        _ => Ty::UNTYPED,
                    }
                })
                .unwrap_or(Ty::UNTYPED),
            // cvar `||=` / `&&=` in value position: unlike the ivar/gvar
            // compound-write arms above, this does NOT mirror them —
            // Steep's `:or_asgn`/`:and_asgn` `case asgn.type`
            // (`type_construction.rb:2399-2426`) has no `:cvasgn` case,
            // so `@@x ||= v` falls to `fallback_to_any` and types as
            // `any` (verified via `steep check`, 2026-07-17; see
            // `check_node`'s dispatch arm for the same finding on the
            // statement-position path). The RHS is still walked
            // read-only for its side effects (mirrors the ivar arm's
            // shape) but the result is discarded.
            Node::ClassVariableOrWriteNode { .. } => node
                .as_class_variable_or_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            Node::ClassVariableAndWriteNode { .. } => node
                .as_class_variable_and_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            Node::ClassVariableOperatorWriteNode { .. } => node
                .as_class_variable_operator_write_node()
                .map(|w| {
                    // Walk the RHS for its side-effect diagnostics —
                    // mirrors plain `ClassVariableWriteNode`'s
                    // value-position arm.
                    self.infer_type(&w.value(), None);
                    let name_str = String::from_utf8_lossy(w.name().as_slice()).into_owned();
                    let var_sym = self.env.names().intern_symbol(&name_str);
                    resolve_cvar_at_self(self, var_sym).unwrap_or(Ty::UNTYPED)
                })
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableWriteNode { .. } => node
                .as_global_variable_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // gvar `||=` / `&&=` / `+=` in value position. Mirrors the
            // ivar/cvar compound-write arms above; the visitor path runs
            // the real side-effect checks while `infer_type` stays
            // read-only.
            Node::GlobalVariableOrWriteNode { .. } => node
                .as_global_variable_or_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableAndWriteNode { .. } => node
                .as_global_variable_and_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::GlobalVariableOperatorWriteNode { .. } => node
                .as_global_variable_operator_write_node()
                .map(|w| {
                    // Walk the RHS for its side-effect diagnostics —
                    // mirrors plain `GlobalVariableWriteNode`'s
                    // value-position arm.
                    self.infer_type(&w.value(), None);
                    let name_str = String::from_utf8_lossy(w.name().as_slice()).into_owned();
                    self.env.lookup_global(&name_str).unwrap_or(Ty::UNTYPED)
                })
                .unwrap_or(Ty::UNTYPED),
            // `Const = expr` in value position — the read-only oracle
            // path for this kind's fallback re-walk
            // (`check_def_node_in_current_context`, when the live
            // `check_constant_write` value happens to be untyped, e.g.
            // the `Class.new { ... }` construction pattern, which the
            // live path special-cases as a side-effecting block-body
            // walk with no live-computed RHS value). Mirrors the
            // ivar/cvar/gvar oracle arms above — RHS type, ambient hint
            // forwarded. Before this arm existed, a method body ending
            // in a bare `Const = expr` write hit the catch-all's
            // `NotImplementedYet` on this fallback path (this kind had
            // no oracle arm at all).
            Node::ConstantWriteNode { .. } => node
                .as_constant_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `Foo ||= rhs` / `Foo &&= rhs` in value position. Sibling
            // of the ivar / cvar / gvar Or/And arms above. NOT Steep
            // parity — Steep 2.0.0 reports `or_asgn with casgn lhs is
            // not supported` / `and_asgn with casgn lhs is not
            // supported` (2026-06-09 measurement). crema covers the
            // Ruby-runtime-valid syntax by returning the RHS type and
            // forwarding the hint, matching the ivar/cvar/gvar
            // paradigm. UnknownConstant emit stays out of this arm —
            // the silent value-position read mirrors `try_resolve_
            // constant_read`'s `&self` story (`check_constant_read`'s
            // doc comment) so a receiver-position occurrence like
            // `(Foo ||= 1).bar` doesn't multiply-emit through the
            // `infer_receiver_type` peek path.
            Node::ConstantOrWriteNode { .. } => node
                .as_constant_or_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantAndWriteNode { .. } => node
                .as_constant_and_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `Foo += rhs` in value position. Steep's `:op_asgn` handler
            // (`type_construction.rb:858-913`) has no `:casgn` case, so
            // `Foo += rhs` falls to the `else` branch and types as `any`
            // — matched here with `Ty::UNTYPED`, symmetric with
            // `check_node`'s value-position handling for this kind
            // (sibling done `write_node_value_type_operator_write_constant`).
            // The RHS is still walked read-only for its bookkeeping
            // (`infer_type` is `&self`, so this does NOT push
            // diagnostics — the visitor-side side effects fire on the
            // statement path), mirroring the ivar/cvar/gvar Operator
            // arms above.
            Node::ConstantOperatorWriteNode { .. } => node
                .as_constant_operator_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            // `M::Bar ||= rhs` / `&&=` in value position. Path
            // counterparts of the bare-constant Or/And arms above —
            // return the RHS type (hint forwarded).
            // `Const::Path = expr` in value position — read-only oracle
            // fallback, symmetric with `ConstantWriteNode`'s arm above.
            // Before this arm existed, this kind (like its
            // bare-constant sibling) fell to the catch-all's
            // `NotImplementedYet` when the live
            // `check_constant_path_write` value was untyped.
            Node::ConstantPathWriteNode { .. } => node
                .as_constant_path_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantPathOrWriteNode { .. } => node
                .as_constant_path_or_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            Node::ConstantPathAndWriteNode { .. } => node
                .as_constant_path_and_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `M::Bar += rhs` in value position. Path counterpart of the
            // bare-constant Operator arm above — same Steep-parity
            // fallback (`:op_asgn` has no `:casgn` case), so this stays
            // `Ty::UNTYPED` rather than resolving the path's declared
            // type. The RHS is still walked read-only for bookkeeping.
            Node::ConstantPathOperatorWriteNode { .. } => node
                .as_constant_path_operator_write_node()
                .map(|w| {
                    self.infer_type(&w.value(), None);
                    Ty::UNTYPED
                })
                .unwrap_or(Ty::UNTYPED),
            // lvar write in value position (`(x = expr).method`,
            // method body whose last expression is a lvasgn). The
            // assignment value is the RHS expression's type — mirrors
            // Steep `type_construction.rb` `:lvasgn` returning
            // `add_typing(node, type: rhs_type)`. SPECIAL_LVAR names
            // (`_`, `__any__`, `__skip__`) short-circuit to untyped,
            // matching the `LocalVariableReadNode` arm above so the
            // cast idiom `(_ = nil) #: T` keeps working in receiver
            // position. Side-effects (env binding, subtype gate,
            // FalseAssertion) fire via `visit_local_variable_write_node`
            // and are not duplicated here.
            Node::LocalVariableWriteNode { .. } => node
                .as_local_variable_write_node()
                .map(|w| {
                    if super::is_special_lvar_name(w.name().as_slice()) {
                        Ty::UNTYPED
                    } else {
                        self.infer_type(&w.value(), hint)
                    }
                })
                .unwrap_or(Ty::UNTYPED),
            // `||=` / `&&=` in value position. Steep parity
            // (`type_construction.rb:2395-2402` → `lvasgn(node, type)`
            // at line 2798-2822): the expression value is the RHS type
            // (`add_typing(node, type: type)`). SPECIAL_LVAR names
            // collapse to `any_type` at line 2802, mirrored here by
            // `Ty::UNTYPED`. Side effects (env rebind) fire via the
            // visitor and are not duplicated.
            Node::LocalVariableOrWriteNode { .. } => node
                .as_local_variable_or_write_node()
                .map(|w| {
                    if super::is_special_lvar_name(w.name().as_slice()) {
                        Ty::UNTYPED
                    } else {
                        self.infer_type(&w.value(), hint)
                    }
                })
                .unwrap_or(Ty::UNTYPED),
            Node::LocalVariableAndWriteNode { .. } => node
                .as_local_variable_and_write_node()
                .map(|w| {
                    if super::is_special_lvar_name(w.name().as_slice()) {
                        Ty::UNTYPED
                    } else {
                        self.infer_type(&w.value(), hint)
                    }
                })
                .unwrap_or(Ty::UNTYPED),
            // `+=` / `-=` etc. in value position. Steep `:op_asgn`
            // (`type_construction.rb:858-868`) rewrites the node to
            // `x = x.<op>(rhs)` and sets the expression value to the
            // operator method's **return type**, not the RHS type.
            // Mirrors `visit_local_variable_operator_write_node` minus
            // side effects: the visitor handles binding rebind and
            // diagnostic emission via `check_synthetic_method_call`;
            // here we only compute the return type for upstream
            // (e.g. `(x += 1.0).bytesize`'s receiver inference).
            //
            // The receiver lookup reads the **pre-rebind** binding
            // because `infer_type` is `&self` and the visitor may not
            // have run yet when reached via `infer_receiver_type` on a
            // ParenthesesNode-wrapped value-position write. Steep has
            // the same property — `synthesize` cannot mutate env until
            // it dispatches through `lvasgn`. Side-effect-free target
            // resolution (`resolve_call_target_at` + `infer_return_type`)
            // matches the sibling-done refactor.
            Node::LocalVariableOperatorWriteNode { .. } => {
                let Some(w) = node.as_local_variable_operator_write_node() else {
                    return Ty::UNTYPED;
                };
                let name_bytes = w.name().as_slice();
                if super::is_special_lvar_name(name_bytes) {
                    return Ty::UNTYPED;
                }
                let name_str = String::from_utf8_lossy(name_bytes);
                let name = self.checker_names().intern(&name_str);
                let receiver_ty = self
                    .lookup_local_variable_for_read(name)
                    .unwrap_or(Ty::UNTYPED);
                if receiver_ty.is_untyped() {
                    return Ty::UNTYPED;
                }
                let rhs_ty = self.infer_type(&w.value(), None);
                let value_node = w.value();
                let op_bytes = w.binary_operator().as_slice();
                let op_name = String::from_utf8_lossy(op_bytes);
                let Some(target) = self.resolve_call_target_at(receiver_ty, &op_name) else {
                    return Ty::UNTYPED;
                };
                let arguments = CallArguments {
                    positional: vec![rhs_ty],
                    positional_spans: vec![super::arg_span(
                        value_node.location().start_offset(),
                        value_node.location().end_offset(),
                    )],
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
                        // Visibility gate mirrors
                        // `check_synthetic_method_call`: the visitor
                        // emits PrivateMethodCall + UNTYPED rebind for
                        // a private operator. Returning a concrete
                        // type here would let the next call dispatch
                        // off a method the visitor already refused —
                        // a non-symmetric leak.
                        if method_def.accessibility == Visibility::Private {
                            return Ty::UNTYPED;
                        }
                        self.infer_return_type(
                            &method_def,
                            &bindings,
                            &arguments,
                            receiver_ty,
                            None::<&CallNode<'_>>,
                            hint,
                        )
                    }
                    CallTarget::UnionMethod { components, .. } => {
                        // OR rule: union method is private if any
                        // component is, matching the visitor's union
                        // arm at `check_synthetic_method_call` and
                        // Steep `union_shape`.
                        let any_private = components
                            .iter()
                            .any(|c| c.method_def.accessibility == Visibility::Private);
                        if any_private {
                            return Ty::UNTYPED;
                        }
                        // Per-component union return (ADR-0021, mirrors
                        // `infer_call_return_type_core`'s UnionMethod arm).
                        let returns: Vec<Ty> = components
                            .iter()
                            .map(|c| {
                                let (_, component_arguments) = self
                                    .synthetic_single_arg_component_arguments(
                                        &c.method_def,
                                        &c.bindings,
                                        c.receiver_type,
                                        &value_node,
                                        rhs_ty,
                                    );
                                self.infer_return_type(
                                    &c.method_def,
                                    &c.bindings,
                                    &component_arguments,
                                    c.receiver_type,
                                    None::<&CallNode<'_>>,
                                    hint,
                                )
                            })
                            .collect();
                        union_of_many(&returns, self.env.types())
                    }
                }
            }
            // `(a, b = 1, 2)` in value position. Steep `type_masgn`
            // (`type_construction.rb:2903-2955`) sets the expression
            // value to `truthy_rhs_type` (the RHS's tuple/array shape).
            // For an ArrayNode literal RHS, delegating to `infer_type`
            // on the RHS rides the existing ArrayNode arm, which
            // produces the same tuple shape Steep's `try_tuple_type!`
            // does. For non-literal RHS (`a, b = expr`) the full
            // `try_convert(:to_ary)` parity is left to the visitor's
            // `multi_assign_fallback` (see Scope outside in the todo).
            Node::MultiWriteNode { .. } => node
                .as_multi_write_node()
                .map(|w| self.infer_type(&w.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // Shorthand hash value (`{a:}`): transparent wrapper around
            // the inner `value()`. Mirrors the `check_node` arm above so
            // read-only `infer_type` callers (e.g. `synthesize_hash_with_hint`
            // forwarding the Hash value hint to each AssocNode value)
            // see the inner read's type with the hint preserved —
            // shorthand `{a:}` and explicit `{a: a}` must agree.
            Node::ImplicitNode { .. } => node
                .as_implicit_node()
                .map(|i| self.infer_type(&i.value(), hint))
                .unwrap_or(Ty::UNTYPED),
            // `bar(...)` argument-forwarding marker. Steep parity
            // (`type_construction.rb:2702-2703`, `:forwarded_args ->
            // any_type`). Argument-side compatibility against the
            // caller's `forward_arg_type` is handled separately in
            // `check_forwarding_call` (`calls.rs:858`); this arm only
            // resolves the node as a value so the read-only path
            // (`collect_arguments_from` / `collect_call_arguments_hinted`)
            // doesn't fall through to the catch-all.
            Node::ForwardingArgumentsNode { .. } => Ty::UNTYPED,
            // `x rescue y` reached read-only: receiver-position
            // (`(x rescue y).foo`), call-arg synthesis, or any place
            // `infer_type` is called on the modifier as a value. Mirror
            // of `check_rescue_modifier_node`'s union return without
            // the side-effecting walk — `&self` precludes both
            // diagnostic push and the env snapshot/join flow the
            // mutating path runs. Steep parity
            // (`type_construction.rb:2195`
            // `union_type(body_pair, *resbody_types)`, else-less branch).
            Node::RescueModifierNode { .. } => node
                .as_rescue_modifier_node()
                .map(|rescue| {
                    let body_ty = self.infer_type(&rescue.expression(), hint);
                    let rescue_ty = self.infer_type(&rescue.rescue_expression(), hint);
                    union_of(body_ty, rescue_ty, self.env.types())
                })
                .unwrap_or(Ty::UNTYPED),
            // Silent fall-through visualizer (dev-only). Push
            // `Crema::NotImplementedYet` so a `crema.toml` opt-in
            // surfaces which Prism node variant reached the catch-all,
            // then keep returning `Ty::UNTYPED` so existing callers
            // (`infer_receiver_type` etc.) stay unaffected at the
            // default `Ignore` severity. `subject_display` carries the
            // variant name extracted from the Debug impl's leading
            // token (`VariantName { ... }` — ruby-prism 1.9.0).
            _ => {
                use crate::diagnostic::{Diagnostic, DiagnosticKind};
                let debug = format!("{:?}", node);
                let variant = debug
                    .split([' ', '{', '('])
                    .next()
                    .unwrap_or("UnknownNode")
                    .to_string();
                let location = self.offset_to_location(node.location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location,
                    kind: DiagnosticKind::NotImplementedYet {
                        site: "infer_type".into(),
                        subject_display: variant,
                    },
                });
                Ty::UNTYPED
            }
        }
    }

    /// Extract-mode occurrence for a `super` site, recorded from the
    /// walk arms after `infer_type` produced the site's type — extract
    /// itself runs no inference. Re-resolves the super target, which is
    /// a repeat of a query the check already logged (map dedup keeps
    /// `consulted` unchanged); the whole body is behind the extract
    /// gate so the plain check path pays nothing.
    fn record_extract_super_at<'pr>(&self, node: &Node<'pr>, ret: Ty, require_rbs_decl: bool) {
        if self.extract.is_none() {
            return;
        }
        if require_rbs_decl && self.ctx.method_type().is_none() {
            return;
        }
        let Some(method_name) = self.ctx.method_name().map(|s| s.to_string()) else {
            return;
        };
        let sym = self.env.names().intern_symbol(&method_name);
        let self_ty = self.current_self_type();
        let Some((method, _)) = self.env.lookup_super_method(self_ty, sym) else {
            return;
        };
        self.record_extract_super(
            node.location().start_offset(),
            node.location().end_offset(),
            &method,
            &method_name,
            self_ty,
            Some(ret),
        );
    }

    /// Emit `Ruby::UnexpectedSuper` for a bare `super` (`ForwardingSuperNode`)
    /// whose target cannot be resolved. Shares the `lookup_super_method`
    /// failure semantics with `super_return_type`; emission is split off so
    /// the `&self` return-type path stays side-effect-free. The SuperNode
    /// (`super(args)`) form is handled by `check_super_node` (calls.rs).
    fn emit_unexpected_super_if_unresolvable<'pr>(&mut self, node: &Node<'pr>) {
        use crate::diagnostic::{Diagnostic, DiagnosticKind};
        // Steep gates `UnexpectedSuper` on `method_context.method` being
        // present (`type_construction.rb` `synthesize_send`). crema mirrors
        // that by requiring `ctx.method_type()` to be `Some` — `None` means
        // the enclosing def has no RBS declaration, in which case Steep
        // doesn't walk the body at all and never reaches the super check.
        // Without this gate, RBS-undeclared defs that use `super` produce a
        // false positive (the super target is genuinely unresolvable, but
        // the call is unconstrained by any signature contract).
        if self.ctx.method_type().is_none() {
            return;
        }
        let Some(method_name) = self.ctx.method_name().map(|s| s.to_string()) else {
            return;
        };
        let sym = self.env.names().intern_symbol(&method_name);
        let self_ty = self.current_self_type();
        if self.env.lookup_super_method(self_ty, sym).is_some() {
            return;
        }
        // Module methods can call `super`; which ancestor provides the method
        // depends on the include site, which is not statically known. Emitting
        // here would be a false positive matching Steep's silent behavior.
        if self.is_inside_module_definition() {
            return;
        }
        let position = self.offset_to_location(node.location().start_offset());
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: position,
            kind: DiagnosticKind::UnexpectedSuper { method_name },
        });
    }

    /// Return type of a `super` call. Resolves the super-target method by
    /// walking the ancestor chain after the defining class
    /// ([`DefinitionBuilder::lookup_super_method`]).
    ///
    /// `arguments` is `Some` for `super(args)` (SuperNode): the args drive
    /// overload selection and method-level generic instantiation via the
    /// shared `infer_return_type`, so the return type matches the overload
    /// the args pick. `arguments` is `None` for bare `super`
    /// (ForwardingSuperNode): the args are not inspected, so the
    /// first-overload return type is projected with the receiver + ancestor
    /// bindings applied (the parent todo's behavior).
    ///
    /// An unresolvable `super` (no ancestor above the defining class defines
    /// the method, or a module definition whose host is not statically
    /// visible) stays `untyped`, matching Steep.
    pub(super) fn super_return_type(&self, arguments: Option<&CallArguments>) -> Ty {
        let Some(method_name) = self.ctx.method_name() else {
            return Ty::UNTYPED;
        };
        let sym = self.env.names().intern_symbol(method_name);
        let self_ty = self.current_self_type();
        let Some((method, bindings)) = self.env.lookup_super_method(self_ty, sym) else {
            return Ty::UNTYPED;
        };
        if let Some(arguments) = arguments {
            return self.infer_return_type(&method, &bindings, arguments, self_ty, None, None);
        }
        let Some(return_ty) = method.defs.first().map(|d| d.type_.return_type()) else {
            return Ty::UNTYPED;
        };
        let subst = definition_builder::substitution_for_receiver(self.env, self_ty, bindings);
        let result = subst.apply(return_ty, self.env.types());
        // Method-level type params (`def m: [T] (T) -> T`) are not bound here
        // because the bare-super arguments are not inspected; fall any
        // survivors back to `untyped` so downstream sees a concrete gradual
        // type instead of a leaked type variable.
        types::fallback_unbound_type_vars_to_untyped(result, self.env.types())
    }

    /// Infer the return type of a method call.
    ///
    /// Tuple/Record literal access is specialized first (see
    /// `tuple_index_specialization` / `record_key_specialization`). For
    /// `Found`, the element/field type wins over the receiver-widened
    /// `Array#[]`/`Hash#[]` return type. For `Missing`, we still give the
    /// fallback type so downstream checks don't cascade into untyped — the
    /// diagnostic is emitted from `check_call_arguments`, not here.
    pub(super) fn infer_call_return_type<'pr>(
        &self,
        call: &CallNode<'pr>,
        hint: Option<Ty>,
        resolved: &ResolvedCall,
    ) -> Ty {
        let ret = self.infer_call_return_type_core(call, hint, resolved);
        // Safe-navigation widens the call expression with nil: when the
        // receiver is nil at runtime, `&.` skips the call and yields nil.
        // `union_of` normalizes, so a method already returning `T | nil`
        // stays single-nil after the widen.
        if call.is_safe_navigation() {
            union_of(ret, Ty::NIL, self.env.types())
        } else {
            ret
        }
    }

    fn infer_call_return_type_core<'pr>(
        &self,
        call: &CallNode<'pr>,
        hint: Option<Ty>,
        resolved: &ResolvedCall,
    ) -> Ty {
        let method_name = String::from_utf8_lossy(call.name().as_slice()).to_string();
        if call.is_attribute_write()
            && let Some(arguments) = call.arguments()
            && let Some(value) = arguments.arguments().last()
        {
            return self.infer_type(&value, None);
        }

        let receiver = resolved.receiver_ty;
        // Tuple/Record literal-access specialization (shared with the
        // check-side diagnostic pass; see `element_access_specialization`).
        // Here we map the outcome to a return type rather than emitting a
        // diagnostic.
        if let Some(result) = self.element_access_specialization(receiver, &method_name, call) {
            return match result {
                LiteralAccessResult::Found(ty) => ty,
                LiteralAccessResult::Missing { fallback, .. } => fallback,
            };
        }

        let Some(target) = resolved.target.as_ref() else {
            return Ty::UNTYPED;
        };

        match target {
            CallTarget::Method {
                method_def,
                bindings,
                ..
            } => {
                // Hint propagation so generic Record params like
                // `[X] ({ foo: X })` unify field-wise in
                // `collect_call_site_bindings`. bindings + receiver are
                // forwarded so the chosen overload is substituted into
                // the receiver's concrete type arguments — without this,
                // a generic method's hint (e.g. `Elem`) reaches the
                // literal-synthesis layer unresolved and silently bails.
                let hint_overload = self.pick_hint_overload(
                    method_def,
                    super::calls::CallSite::Call(call),
                    bindings,
                    receiver,
                );
                let raw = if self.call_has_forwarding_args(call)
                    && let Some(caller_mt) = self.ctx.forward_arg_type()
                {
                    let Some(arguments) = self.collect_forwarded_call_arguments(call, &caller_mt)
                    else {
                        return Ty::UNTYPED;
                    };
                    self.infer_return_type(
                        method_def,
                        bindings,
                        &arguments,
                        receiver,
                        Some(call),
                        hint,
                    )
                } else {
                    let arguments = self.collect_call_arguments_hinted(
                        super::calls::CallSite::Call(call),
                        hint_overload.as_ref(),
                    );
                    self.infer_return_type(
                        method_def,
                        bindings,
                        &arguments,
                        receiver,
                        Some(call),
                        hint,
                    )
                };
                self.apply_compact_special_return(call, &method_name, method_def, raw)
            }
            CallTarget::UnionMethod { components, .. } => {
                // Per-component return types, unioned (ADR-0021). Arguments
                // are normally collected hintless: hint-driven overload
                // selection across the union belongs to the MethodType.union
                // slice (child todo `high_union_receiver_method_type_union`).
                // `bar(...)` is the exception: the forwarded caller sig is
                // the call site's argument shape, not a bidirectional hint.
                let arguments = if self.call_has_forwarding_args(call)
                    && let Some(caller_mt) = self.ctx.forward_arg_type()
                {
                    let Some(arguments) = self.collect_forwarded_call_arguments(call, &caller_mt)
                    else {
                        return Ty::UNTYPED;
                    };
                    arguments
                } else {
                    self.collect_call_arguments(super::calls::CallSite::Call(call))
                };
                let returns: Vec<Ty> = components
                    .iter()
                    .map(|c| {
                        let raw = self.infer_return_type(
                            &c.method_def,
                            &c.bindings,
                            &arguments,
                            c.receiver_type,
                            Some(call),
                            hint,
                        );
                        self.apply_compact_special_return(call, &method_name, &c.method_def, raw)
                    })
                    .collect();
                types::union_of_many(&returns, self.env.types())
            }
        }
    }

    /// Special-method hook for `Array#compact` / `Enumerable#compact` /
    /// `Hash#compact` — peel one layer of `nil` off the element / value
    /// type so `Array[T?].compact` yields `Array[T]`. Steep parity with
    /// the `array_compact` / `hash_compact` arms of
    /// `SPECIAL_METHOD_NAMES` in
    /// `lib/steep/type_construction.rb:3602-3667`.
    ///
    /// Gated on three conjunctive conditions, matching Steep:
    /// - method name is `compact`
    /// - the call site passes no positional/keyword args, no block, no
    ///   block-pass (`call.arguments().is_none() && call.block().is_none()`)
    /// - the method itself was declared on `::Array`, `::Enumerable`, or
    ///   `::Hash` (any overload's `defined_in` matches), so a
    ///   user-defined `MyBag#compact` is not hijacked
    ///
    /// Only the return type is rewritten; the receiver `Ty` is untouched
    /// (mutable narrowing is out of ADR scope).
    fn apply_compact_special_return(
        &self,
        call: &CallNode<'_>,
        method_name: &str,
        method_def: &crate::definition::Method,
        raw_return: Ty,
    ) -> Ty {
        if method_name != "compact" {
            return raw_return;
        }
        if call.arguments().is_some() || call.block().is_some() {
            return raw_return;
        }
        let builtins = self.env.names().builtins();
        let array_name = builtins.array;
        let enumerable_name = builtins.enumerable;
        let hash_name = builtins.hash;
        let defined_matches = method_def.defs.iter().any(|td| {
            td.defined_in == array_name
                || td.defined_in == enumerable_name
                || td.defined_in == hash_name
        });
        if !defined_matches {
            return raw_return;
        }
        let types = self.env.types();
        match types.resolve(raw_return) {
            // `Enumerable#compact` also returns `Array[E]` per
            // `rbs/core/enumerable.rbs:414`, so its return type lands in
            // this `array_name` arm too — `enumerable_name` is only
            // consulted in the `defined_in` gate above, never in the
            // return-type shape match.
            Type::ClassInstance { name, args } if *name == array_name && args.len() == 1 => {
                let inner = args[0];
                let unwrapped = self.unwrap_optional(inner);
                if unwrapped == inner {
                    raw_return
                } else {
                    let name = *name;
                    types.intern(Type::ClassInstance {
                        name,
                        args: vec![unwrapped],
                    })
                }
            }
            Type::ClassInstance { name, args } if *name == hash_name && args.len() == 2 => {
                let key = args[0];
                let value = args[1];
                let unwrapped = self.unwrap_optional(value);
                if unwrapped == value {
                    raw_return
                } else {
                    let name = *name;
                    types.intern(Type::ClassInstance {
                        name,
                        args: vec![key, unwrapped],
                    })
                }
            }
            _ => raw_return,
        }
    }

    pub(super) fn infer_return_type<'pr>(
        &self,
        method_def: &crate::definition::Method,
        bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
        arguments: &CallArguments,
        receiver_type: Ty,
        call: Option<&CallNode<'pr>>,
        hint: Option<Ty>,
    ) -> Ty {
        // Delegate overload narrowing (arity / block-presence / arg
        // subtyping / keyword shape / bound) to the shared selector so
        // filter logic doesn't diverge from the diagnostic and block-body
        // paths. Structural filters are strict — when no overload matches,
        // the narrower returns empty and this path falls back to `untyped`.
        // That keeps downstream checks from cascading off a return type
        // guessed from an unreachable overload while `UnresolvedOverloading`
        // is already being reported for the same call.
        let candidates = self.narrow_overloads(
            method_def,
            arguments,
            bindings,
            receiver_type,
            call.map(super::calls::CallSite::Call),
            false,
            hint,
        );

        // First-fit: once candidates are narrowed, commit to the first one in
        // RBS definition order. Merging all matches into a union was a past
        // choice that produced spurious unions whenever `TypeVariable`
        // wildcard matching left multiple overloads viable (see Phase C).
        // RBS stdlib is written specific-first / general-last, so definition
        // order already encodes the author's case analysis — Steep and Sorbet
        // both resolve this way.
        let result = match candidates.first() {
            None => Ty::UNTYPED,
            Some(overload) => {
                // Augment the receiver-side bindings by unifying call-site
                // args against this overload's params. Receiver bindings take
                // precedence: `collect_call_site_bindings` only fills slots
                // that are not already bound.
                let mut local_bindings =
                    self.collect_call_site_bindings(overload, arguments, bindings);

                // Phase B (ADR-0012, hint-driven instantiation): the context
                // hint (e.g. a trailing `#:` assertion on the call expression)
                // is the caller's stated intent. Unify it against the
                // overload's return type and let the resulting bindings
                // override the seed-derived ones for *method-level* type
                // params only — receiver-side bindings retain their
                // precedence over the call site.
                //
                // The override is direct (not via `Constraints::add_upper`)
                // because by this point `collect_call_site_bindings` has
                // already bound those params from the seed args, so the
                // Constraints unbound-only path would skip them.
                //
                // Subtype gate: only override when the hint and seed agree
                // (one is a subtype of the other). When they conflict
                // (e.g. `Animal` from arg vs `Dog` from hint with no
                // subtype relation), keep the seed binding so the existing
                // argument-vs-param check downstream still surfaces a
                // mismatch. A dedicated `Ruby::UnsatisfiableConstraint`
                // diagnostic lives in a separate todo and will replace this
                // silent fallback with a proper report.
                if let Some(hint_ty) = hint {
                    let call_offset = call.map(|c| c.location().start_offset());
                    self.apply_hint_override(overload, hint_ty, &mut local_bindings, call_offset);
                }

                // Method-level type params that appear only in the block's
                // return type (e.g. `[X] () { (T) -> X } -> Array[X]`) are
                // unbound after arg-based unification. Infer the block body
                // under a transient overlay and unify against the block's
                // return type to pick up X.
                if let Some(call) = call {
                    self.augment_bindings_with_block_body(
                        super::calls::CallSite::Call(call),
                        overload,
                        &mut local_bindings,
                    );
                }

                // Route through `substitution_for_call_receiver` (not the
                // raw `substitution_for_receiver`): a `SELF_TYPE` receiver
                // needs its `instance` / `class` bindings built from the
                // concrete `current_self_type()` while `self` itself stays
                // opaque so `-> self` returns keep propagating.
                let local_subst =
                    self.substitution_for_call_receiver(receiver_type, local_bindings);
                local_subst.apply(overload.return_type(), self.env.types())
            }
        };

        // Any `TypeVariable` left here was not bound by either the receiver
        // or call-site args. Fall back to `untyped` so callers see a concrete
        // gradual-typed result, matching RBS/Steep.
        types::fallback_unbound_type_vars_to_untyped(result, self.env.types())
    }

    /// Build a bindings map for a single overload by unifying each argument
    /// type against the corresponding parameter type. Used to bind generic
    /// class type vars (e.g. `Pair[K, V]`'s `K`) at call sites where the
    /// receiver alone cannot provide them — the archetype being `.new`.
    ///
    /// Two-stage to preserve receiver-beats-call-site precedence even when
    /// the same type var appears in multiple param positions with different
    /// arg types: unify into a fresh map (which union-merges repeats — see
    /// `unify_arg_against_param`), then fold into `receiver_bindings` with
    /// `or_insert` so existing receiver entries are never touched.
    pub(super) fn collect_call_site_bindings(
        &self,
        overload: &crate::types::MethodType,
        arguments: &CallArguments,
        receiver_bindings: &FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) -> FxHashMap<crate::type_param::TypeVarKey, Ty> {
        let mut call_site: FxHashMap<crate::type_param::TypeVarKey, Ty> = FxHashMap::default();
        let folded = self.fold_braceless_keywords_for(arguments, overload);
        let arguments = folded.as_ref().unwrap_or(arguments);
        let arg_count = arguments.positional.len();

        for (index, &actual) in arguments.positional.iter().enumerate() {
            let expected = overload.positional_param_for_call(index, arg_count);
            if let Some(expected) = expected {
                self.unify_arg_against_param(expected, actual, &mut call_site);
            }
        }

        // Splat tail (unexpandable `Array[E]`) unifies its element type
        // against the overload's rest slot, so a generic rest like
        // `def f: [T] (*T) -> Array[T]` can infer `T` from `f(*xs)` where
        // `xs: Array[Integer]`. Untyped tails are skipped — they leave
        // every generic unconstrained and the receiver bindings fill in.
        if let Some(tail) = &arguments.splat_tail
            && !tail.element_ty.is_untyped()
            && let Some(rest_ty) = overload.rest_positional()
        {
            self.unify_arg_against_param(rest_ty, tail.element_ty, &mut call_site);
        }

        for (kw_name, actual, _, _) in &arguments.keywords {
            let expected = overload
                .required_keywords()
                .iter()
                .find(|(name, _)| name == kw_name)
                .map(|(_, ty)| *ty)
                .or_else(|| {
                    overload
                        .optional_keywords()
                        .iter()
                        .find(|(name, _)| name == kw_name)
                        .map(|(_, ty)| *ty)
                })
                .or(overload.rest_keyword());
            if let Some(expected) = expected {
                self.unify_arg_against_param(expected, *actual, &mut call_site);
            }
        }

        if let Some(explicit) = self.explicit_type_bindings_for_overload(overload, arguments) {
            for (name, ty) in explicit {
                call_site.insert(name, ty);
            }
        }

        let method_level: FxHashSet<crate::type_param::TypeVarKey> = overload
            .type_params
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let mut bindings = receiver_bindings.clone();
        for (name, ty) in call_site {
            // Method-level type parameters (`[X]` at method scope) shadow
            // any same-named class-level binding, matching Steep's scoping.
            // Other names come from the receiver and must not be overwritten.
            if method_level.contains(&name) {
                bindings.insert(name, ty);
            } else {
                bindings.entry(name).or_insert(ty);
            }
        }
        bindings
    }

    /// Phase 3 (block-only) inference: if the overload has method-level type
    /// parameters still unbound after argument unification and the block's
    /// return type references them, infer the block body under a transient
    /// overlay and unify the result against the expected block return type.
    ///
    /// No-op when the call has no block argument, the overload has no block
    /// signature, or all method-level type params are already bound by arg
    /// unification (Phase 2 path wins — arg-first precedence).
    pub(super) fn augment_bindings_with_block_body<'pr>(
        &self,
        call: super::calls::CallSite<'_, 'pr>,
        overload: &crate::types::MethodType,
        local_bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) {
        let Some(block_arg) = call.block() else {
            return;
        };
        let Some(expected_block) = overload.block.as_ref() else {
            return;
        };

        // Only method-level type params still unbound are candidates.
        let unbound: Vec<crate::type_param::TypeVarKey> = overload
            .type_params
            .iter()
            .map(|p| p.name.clone())
            .filter(|n| !local_bindings.contains_key(n))
            .collect();
        if unbound.is_empty() {
            return;
        }

        // Substitute block param types with current bindings so the overlay
        // binds block names to concrete types (e.g. `T -> Integer`).
        let subst = Substitution::from_mapping(local_bindings.clone());

        // `&:sym` block pass: no block body to infer — Symbol#to_proc
        // semantics resolve the symbol's method on the substituted one-arg
        // block parameter and its return type plays the role of the body
        // type below (Steep's `:block_pass` symbol-literal branch). NoMethod
        // reporting stays in `check_block`'s pass; this path only infers.
        if block_arg.as_block_node().is_none() {
            let Some((name, _)) = super::calls::block_pass_symbol(&block_arg) else {
                return;
            };
            let Some(param_ty) = super::calls::symbol_to_proc_one_arg_param(expected_block) else {
                return;
            };
            let param_ty = subst.apply(param_ty, self.env.types());
            let sym = self.env.names().intern_symbol(&name);
            let super::calls::SymbolToProcResolution::Resolved(ret) =
                self.symbol_to_proc_return_type(param_ty, sym)
            else {
                return;
            };
            self.bind_unbound_from_block_return(expected_block, ret, unbound, local_bindings);
            return;
        }

        let Some(block) = block_arg.as_block_node() else {
            return;
        };
        let Some(body) = block.body() else { return };

        let expected_params: Vec<Ty> = expected_block
            .params()
            .iter()
            .map(|&p| subst.apply(p, self.env.types()))
            .collect();

        // Build overlay by walking the block's AST parameter list and pairing
        // each required param with the corresponding substituted expected type.
        // Mirrors the name extraction in `setup_block_scope` but yields a map
        // instead of pushing onto ctx.
        let mut overlay: FxHashMap<crate::name::Name, Ty> = FxHashMap::default();
        if let Some(params_node) = block.parameters() {
            if let Some(block_params) = params_node.as_block_parameters_node()
                && let Some(params) = block_params.parameters()
            {
                let requireds = params.requireds();
                let rest_present = params.rest().is_some();
                let (param_types, rest_ty) =
                    self.block_param_types(&expected_params, requireds.len(), rest_present);
                for (index, param) in requireds.iter().enumerate() {
                    if let Some(required) = param.as_required_parameter_node()
                        && let Some(Some(expected_type)) = param_types.get(index).copied()
                    {
                        let name_str = String::from_utf8_lossy(required.name().as_slice());
                        let name = self.checker_names().intern(&name_str);
                        overlay.insert(name, expected_type);
                    }
                }
                if let Some(ty) = rest_ty
                    && let Some(rest_node) = params.rest()
                    && let Some(rest_param) = rest_node.as_rest_parameter_node()
                    && let Some(name_id) = rest_param.name()
                {
                    let name_str = String::from_utf8_lossy(name_id.as_slice());
                    let name = self.checker_names().intern(&name_str);
                    overlay.insert(name, ty);
                }
            } else if params_node.as_it_parameters_node().is_some() {
                // Implicit `it` (Ruby 3.4+): bind it like a single named param
                // `|it|`. This branch is the structural twin of the `it` arm in
                // `setup_block_scope`; keeping the two identical is what stops
                // the block-param binding paths from drifting again (this bug
                // was exactly that drift — augment lacked the arm). UNTYPED when
                // the block yields nothing mirrors `setup_block_scope`.
                let (param_types, _) = self.block_param_types(&expected_params, 1, false);
                let it_ty = param_types
                    .first()
                    .copied()
                    .flatten()
                    .unwrap_or(Ty::UNTYPED);
                let name = self.checker_names().intern("it");
                overlay.insert(name, it_ty);
            } else if let Some(numbered) = params_node.as_numbered_parameters_node() {
                // Numbered parameters `_1`..`_9`: structural twin of the arm in
                // `setup_block_scope`. `maximum` doubles as the required-param
                // count, so `block_param_types` gives `|x|` semantics at 1 and
                // auto-splat at >= 2 with no new rule.
                let count = numbered.maximum() as usize;
                let (param_types, _) = self.block_param_types(&expected_params, count, false);
                for index in 0..count {
                    let ty = param_types
                        .get(index)
                        .copied()
                        .flatten()
                        .unwrap_or(Ty::UNTYPED);
                    let name = self.checker_names().intern(&format!("_{}", index + 1));
                    overlay.insert(name, ty);
                }
            }
        }

        // Infer the block body under the overlay. The overlay is popped
        // unconditionally by `with_overlay` — no state leaks.
        let body_ty = self.with_overlay(overlay, |checker| checker.infer_type(&body, None));
        self.bind_unbound_from_block_return(expected_block, body_ty, unbound, local_bindings);
    }

    /// Shared tail of Phase 3: unify the block's actual return type (an
    /// inferred body type, or a Symbol#to_proc resolved return type) against
    /// the expected block return type and solve the still-unbound
    /// method-level type params. Untyped actuals bind nothing.
    fn bind_unbound_from_block_return(
        &self,
        expected_block: &crate::types::Block,
        body_ty: Ty,
        unbound: Vec<crate::type_param::TypeVarKey>,
        local_bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) {
        if body_ty.is_untyped() {
            return;
        }

        // Unify the body type against the expected block return type
        // (kept un-substituted so `X` remains a `TypeVariable`). This produces
        // lower bounds for `X` by walking the structure — e.g. an expected
        // `Array[X]` against an actual `Array[String]` yields `X := String`.
        let mut block_return_bindings: FxHashMap<crate::type_param::TypeVarKey, Ty> =
            FxHashMap::default();
        self.unify_arg_against_param(
            expected_block.return_type(),
            body_ty,
            &mut block_return_bindings,
        );

        // Filter through `Constraints` so only unknowns (the still-unbound
        // method-level type params) contribute to `local_bindings`. This is
        // the Phase A use of the Constraints API; future variance / bounds /
        // default todos will expand the path without changing call sites.
        let mut constraints = super::constraints::Constraints::new(unbound.iter().cloned());
        for (name, ty) in block_return_bindings {
            constraints.add_lower(name, ty);
        }
        for (name, ty) in constraints.solution(self.env.types()) {
            local_bindings.insert(name, ty);
        }
    }

    /// Structural unification of a parameter type against an argument type,
    /// collecting bindings for unbound `TypeVariable`s found on the param
    /// side. Corresponds to the inverse of `substitute_type_vars`: the
    /// latter applies known bindings, the former produces them.
    ///
    /// Does not fail or mutate outside `bindings`; when structures do not
    /// line up (different class names, mismatched arities, Union on either
    /// side, etc.) the call is a no-op, leaving argument type-checking to
    /// the subtype checker.
    /// Phase B (ADR-0012): override `bindings` for `overload`'s method-level
    /// type params with the bindings produced by unifying its return type
    /// against `hint_ty`. Three-case gate per param, modelling Steep's
    /// `seed <: U <: hint` constraint solver:
    ///
    /// 1. `hint_val` is itself `untyped` → no-op. An `untyped` hint
    ///    carries no narrowing information, and the gradual `<: untyped`
    ///    rule would otherwise pass any seed through and silently erase
    ///    diagnostics that the seed-bound binding would have produced.
    /// 2. seed is unbound, or seed's structure transitively contains
    ///    `untyped` (e.g. `Array[untyped]` from an empty array literal) →
    ///    override. The seed has no concrete commitment; the hint is the
    ///    only source of intent, matching Steep's choice of `U = hint`
    ///    when the lower bound is `bot`/`untyped`.
    /// 3. both seed and hint are concrete → keep the seed. Steep's
    ///    solver picks the lower bound as `U`'s assigned type; widening
    ///    to the hint would silently mask specific-type errors. Genuine
    ///    conflicts (`Animal` seed vs `Dog` hint with no gradual hole)
    ///    fall through to the downstream argument-vs-param check.
    ///
    /// A dedicated `Ruby::UnsatisfiableConstraint` diagnostic (separate
    /// todo) will eventually report case-3 conflicts directly.
    /// Phase B (ADR-0012) hint override with conflict detection.
    ///
    /// `call_offset`: byte offset of the call expression start, used to
    /// position the `UnsatisfiableConstraint` diagnostic. `None` suppresses
    /// the diagnostic (block-scope binding path does not need to re-emit it).
    pub(super) fn apply_hint_override(
        &self,
        overload: &crate::types::MethodType,
        hint_ty: Ty,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
        call_offset: Option<usize>,
    ) {
        let mut hint_bindings: FxHashMap<crate::type_param::TypeVarKey, Ty> = FxHashMap::default();
        self.unify_return_against_hint(overload.return_type(), hint_ty, &mut hint_bindings);
        if hint_bindings.is_empty() {
            return;
        }
        let method_level: FxHashSet<crate::type_param::TypeVarKey> = overload
            .type_params
            .iter()
            .map(|p| p.name.clone())
            .collect();
        for (name, hint_val) in hint_bindings {
            if !method_level.contains(&name) {
                continue;
            }
            // Expand aliases before the gate so `type Any = untyped` (and
            // similar one-hop aliases) behave like raw `untyped`. Mirrors
            // `subtyping.rs::check`, which expands both sides before
            // comparing.
            let hint_expanded = crate::definition_builder::expand_alias(self.env, hint_val);
            if hint_expanded.is_untyped() {
                continue;
            }
            match bindings.get(&name).copied() {
                None => {
                    // seed unbound — safe override, no conflict possible
                    bindings.insert(name, hint_val);
                }
                Some(seed_val) => {
                    let checker = self.subtyper();
                    let compatible =
                        checker.check(seed_val, hint_val) || checker.check(hint_val, seed_val);
                    if !compatible {
                        // Lower (seed) and upper (hint) have no subtype
                        // relation in either direction: emit the diagnostic
                        // and override with the hint so downstream inference
                        // continues from the user-stated intent.
                        if let Some(offset) = call_offset {
                            use crate::diagnostic::{Diagnostic, DiagnosticKind};
                            let position = self.offset_to_location(offset);
                            let type_param_str = self.env.names().resolve(name.raw);
                            self.push_diagnostic(Diagnostic {
                                scope: None,
                                location: position,
                                kind: DiagnosticKind::UnsatisfiableConstraint {
                                    lower: self.display_type(seed_val),
                                    upper: self.display_type(hint_val),
                                    type_param: type_param_str,
                                    method_type: self.display_method_type(overload),
                                },
                            });
                        }
                        bindings.insert(name, hint_val);
                    } else if self.ty_contains_untyped(seed_val) {
                        // seed contains untyped but hint is compatible —
                        // override (same as before: hint carries more
                        // concrete information than an untyped-containing seed)
                        bindings.insert(name, hint_val);
                    }
                    // else: seed is concrete and compatible → keep seed.
                    // Widening a concrete seed to a looser hint would
                    // silently mask type errors in the body.
                }
            }
        }
    }

    /// Returns true if `ty` is `untyped` or transitively contains `untyped`
    /// at any nested position (e.g. `Array[untyped]`, `Hash[Symbol, untyped]`,
    /// `Tuple[Integer, untyped]`, `Optional[untyped]`, etc.).
    ///
    /// Used by `apply_hint_override` to decide whether the seed binding
    /// carries enough concrete information to be preferred over the hint.
    fn ty_contains_untyped(&self, ty: Ty) -> bool {
        let mut seen: FxHashSet<Ty> = FxHashSet::default();
        self.ty_contains_untyped_inner(ty, &mut seen)
    }

    /// Recursive walker behind [`ty_contains_untyped`]. The `seen` set
    /// keys on the post-expand `Ty` (intern id) — different from Steep's
    /// `Factory#deep_expand_alias`, which guards on alias *name* and only
    /// recurses through Union/Intersection. crema's walker descends into
    /// ClassInstance args, Tuple members, Record fields, and Optional
    /// inners too, so a recursive alias whose body re-references itself
    /// (`type ctx = [ctx, ...] | nil`) needs the broader, expansion-level
    /// guard: descending into the inner Tuple element re-expands `ctx` to
    /// the same Union, hits the seen set, and returns `false` (no untyped
    /// found on that path) instead of recursing forever.
    fn ty_contains_untyped_inner(&self, ty: Ty, seen: &mut FxHashSet<Ty>) -> bool {
        let ty = crate::definition_builder::expand_alias(self.env, ty);
        if !seen.insert(ty) {
            return false;
        }
        if ty.is_untyped() {
            return true;
        }
        let types = self.env.types();
        match types.resolve(ty) {
            crate::types::Type::ClassInstance { args, .. } => args
                .iter()
                .any(|&a| self.ty_contains_untyped_inner(a, seen)),
            crate::types::Type::Tuple(members) => members
                .iter()
                .any(|&m| self.ty_contains_untyped_inner(m, seen)),
            crate::types::Type::Union(members) => members
                .iter()
                .any(|&m| self.ty_contains_untyped_inner(m, seen)),
            crate::types::Type::Optional(inner) => self.ty_contains_untyped_inner(*inner, seen),
            crate::types::Type::Record { fields } => fields
                .iter()
                .any(|(_, t, _)| self.ty_contains_untyped_inner(*t, seen)),
            _ => false,
        }
    }

    pub(super) fn unify_arg_against_param(
        &self,
        param: Ty,
        arg: Ty,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) {
        self.unify_into_bindings(param, arg, bindings, UnifyMode::WidenLeaves);
    }

    /// Unify an overload's return type against a hint type, leaving the
    /// hint's user-supplied shape intact. The arg-side caller uses
    /// [`unify_arg_against_param`] which widens `Tuple → Array`,
    /// `Record → Hash`, and `Literal → class` at the boundary to match
    /// the call-site subtype check. Hint-side bindings must NOT widen:
    /// when the user writes `Hash[K, [String, bool]]` as a hint, the
    /// `V := [String, bool]` binding is the whole point — widening it to
    /// `V := Array[String | bool]` defeats hint-driven shape recovery
    /// (e.g. `transform_values { |v| [v, true] }` collapsing back to
    /// `Array[String | bool]` instead of staying a tuple).
    pub(super) fn unify_return_against_hint(
        &self,
        return_ty: Ty,
        hint_ty: Ty,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
    ) {
        self.unify_into_bindings(return_ty, hint_ty, bindings, UnifyMode::KeepShape);
    }

    /// Shared traversal behind [`unify_arg_against_param`] and
    /// [`unify_return_against_hint`]. `mode` toggles whether `Literal →
    /// class`, `Tuple → Array`, `Record → Hash` widening is applied to
    /// the actual side: arg unification widens (matching the subtype
    /// check at the call-site boundary), hint unification preserves
    /// the user-supplied shape so tuple/record bindings survive.
    fn unify_into_bindings(
        &self,
        param: Ty,
        arg: Ty,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
        mode: UnifyMode,
    ) {
        let types_tbl = self.env.types();
        let param_t = types_tbl.resolve(param);

        // Record-on-param: unify field-wise without widening arg → Hash,
        // so `[X] ({ foo: X })` against a synthesized `{ foo: ::Integer }`
        // binds X = Integer. The Record→Hash widening below would erase
        // the field-level structure before we could see it.
        if let Type::Record { fields: p_fields } = &param_t {
            if let Type::Record { fields: a_fields } = types_tbl.resolve(arg) {
                for (p_key, p_ty, _) in p_fields.iter() {
                    if let Some((_, a_ty, _)) = a_fields.iter().find(|(k, _, _)| k == p_key) {
                        self.unify_into_bindings(*p_ty, *a_ty, bindings, mode);
                    }
                }
            }
            return;
        }

        // Arg-side boundary widening (Literal → class, Tuple → Array,
        // Record → Hash). Applied only under `UnifyMode::WidenLeaves`
        // (call-site arg unification). `UnifyMode::KeepShape` preserves
        // the user-supplied shape so `V := [String, bool]` survives
        // instead of collapsing to `V := Array[String | bool]`.
        let arg_widened = if matches!(mode, UnifyMode::WidenLeaves) {
            match types_tbl.resolve(arg) {
                Type::Literal(lit) => self
                    .env
                    .class_instance_type(*lit.class_typename(self.env.names().builtins())),
                Type::Tuple(members) => self.widen_tuple_to_array(members),
                Type::Record { fields } => self.widen_record_to_hash(fields),
                _ => arg,
            }
        } else {
            arg
        };

        match param_t {
            Type::TypeVariable { raw, scope } => {
                let key = crate::type_param::TypeVarKey {
                    raw: *raw,
                    scope: scope.clone(),
                };
                let merged = match bindings.get(&key).copied() {
                    Some(existing) => self.union_merge_bindings(existing, arg_widened),
                    None => arg_widened,
                };
                bindings.insert(key, merged);
            }
            Type::ClassInstance {
                name: p_name,
                args: p_args,
            } => {
                if let Type::ClassInstance {
                    name: a_name,
                    args: a_args,
                } = types_tbl.resolve(arg_widened)
                    && p_name == a_name
                    && p_args.len() == a_args.len()
                {
                    for (&p, &a) in p_args.iter().zip(a_args.iter()) {
                        self.unify_into_bindings(p, a, bindings, mode);
                    }
                }
            }
            Type::Optional(p_inner) => match types_tbl.resolve(arg_widened) {
                Type::Optional(a_inner) => {
                    self.unify_into_bindings(*p_inner, *a_inner, bindings, mode);
                }
                Type::Nil => {}
                _ => {
                    // e.g. `String?` param receiving a `::String` arg.
                    self.unify_into_bindings(*p_inner, arg_widened, bindings, mode);
                }
            },
            Type::Tuple(p_members) => {
                if let Type::Tuple(a_members) = types_tbl.resolve(arg_widened)
                    && p_members.len() == a_members.len()
                {
                    for (&p, &a) in p_members.iter().zip(a_members.iter()) {
                        self.unify_into_bindings(p, a, bindings, mode);
                    }
                }
            }
            // T-binding for proc types: unify return type and each parameter
            // slot (required / optional / trailing positionals, rest_positional,
            // required / optional keywords keyed by name, rest_keyword) on
            // both the function and the optional block. Proc params are
            // contravariant, but this is a heuristic lower-bound extraction,
            // not a sound variance-aware check.
            Type::Proc {
                type_: p_ft,
                block: p_block,
                ..
            } => {
                if let Type::Proc {
                    type_: a_ft,
                    block: a_block,
                    ..
                } = types_tbl.resolve(arg_widened)
                {
                    self.unify_function_slots(p_ft, a_ft, bindings, mode);
                    if let (Some(p_blk), Some(a_blk)) = (p_block, a_block) {
                        self.unify_function_slots(&p_blk.type_, &a_blk.type_, bindings, mode);
                    }
                }
            }
            // T-binding for interface params (`_MyEntry[E]`): dispatch each
            // required method against the arg's concrete shape and unify
            // the interface's own signature (folded into the caller's type
            // vars, see `unify_interface_into_bindings`'s doc) against the
            // arg side's concretized signature. Best-effort like every
            // other arm here — an unknown interface or a missing method on
            // the arg side yields no binding rather than an error, mirroring
            // `SubtypeChecker::check_interface_conformance`'s gradual-typing
            // fallbacks (that function does the actual subtyping check;
            // this one only extracts bindings).
            Type::Interface {
                name: iface_name,
                args: iface_args,
            } => {
                self.unify_interface_into_bindings(
                    *iface_name,
                    iface_args,
                    arg_widened,
                    bindings,
                    mode,
                );
            }
            _ => {}
        }
    }

    /// Extract type-var bindings for an interface-typed param from a
    /// concrete arg, for the `Type::Interface` arm of
    /// [`Self::unify_into_bindings`]. Mirrors
    /// `SubtypeChecker::check_interface_conformance`'s method-dispatch walk
    /// (same receiver-shape dispatch, same required-method-name source),
    /// but unifies each required method's signature into `bindings`
    /// instead of checking subtype compatibility.
    ///
    /// `lookup_interface_method_with_args(iface_name, iface_args, ...)`
    /// folds the interface's own class-level params into `iface_args` the
    /// same way `lookup_instance_method_with_args` folds a class's own
    /// params into the receiver's args — so when `iface_args` is itself
    /// the caller's unresolved type var (e.g. `_MyEntry[E]`'s `E` is
    /// `D.check`'s own method-level `[E]`), the interface method's
    /// signature comes back with that type var already substituted in
    /// place of the interface's own parameter. Unifying it directly
    /// against the arg side's concretized signature therefore writes into
    /// `bindings` in the caller's scope with no separate propagation step.
    fn unify_interface_into_bindings(
        &self,
        iface_name: crate::type_name::TypeName,
        iface_args: &[Ty],
        arg: Ty,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
        mode: UnifyMode,
    ) {
        let types_tbl = self.env.types();
        let Some(required_methods) = self.env.interface_method_names_by_type_name(&iface_name)
        else {
            // Unknown interface -- no binding (gradual typing, same
            // early-out as `check_interface_conformance`).
            return;
        };

        // Same receiver-shape dispatch as `check_interface_conformance`,
        // plus one addition: `sub_is_interface` tracks whether the arg's
        // own shape was an interface (the "same shape" completion case,
        // e.g. `_MyEntry[Integer]`), because that side needs a different
        // lookup than `check_interface_conformance` uses.
        // `check_interface_conformance` calls `lookup_instance_method_with_args`
        // uniformly (even when `sub` is itself an interface) and gets away
        // with it because its `SubtypeChecker` treats an unsubstituted free
        // type var as a wildcard that matches anything -- fine for a
        // tolerant subtype *check*, but wrong for *unification*:
        // `lookup_instance_method_with_args` does not fold `sub_args` into
        // an interface's own class-level params (that folding is
        // `interface_ancestors`-specific), so it returns empty bindings for
        // an interface `sub_class_name` and unifying would bind the
        // interface's own unsubstituted class param straight into
        // `bindings` instead of the concrete type it actually carries.
        // `lookup_interface_method_with_args` is the "interface
        // counterpart" that does fold `sub_args` (see its doc), so use it
        // whenever the arg's own shape is an interface.
        let (sub_class_name, sub_args, is_singleton, sub_is_interface) =
            match types_tbl.resolve(arg) {
                Type::ClassInstance { name, args } => (*name, args.clone(), false, false),
                Type::Interface { name, args } => (*name, args.clone(), false, true),
                Type::ClassSingleton { name } => (*name, Vec::new(), true, false),
                Type::Nil => (
                    self.env.names().builtins().nil_class,
                    Vec::new(),
                    false,
                    false,
                ),
                // No method-dispatch shape to unify against -- best-effort,
                // no binding.
                _ => return,
            };

        // Depth cap: a self-referential interface (`interface _Node[T]; def
        // next: () -> _Node[T]; end`, satisfied by an equally
        // self-referential class) recurses back into this same arm through
        // `unify_function_slots` -> `unify_into_bindings` on the method's
        // own return type forever otherwise, because that return type is a
        // *fresh* `Ty` reached via method lookup rather than sub-structure
        // of the original `arg` (every other `unify_into_bindings` arm only
        // ever recurses into the latter, which is bounded by the type
        // expression's own finite size). See `interface_unify_depth`'s doc.
        const MAX_INTERFACE_UNIFY_DEPTH: u32 = 16;
        if self.interface_unify_depth.get() >= MAX_INTERFACE_UNIFY_DEPTH {
            return;
        }
        self.interface_unify_depth
            .set(self.interface_unify_depth.get() + 1);

        // `self`/`instance` inside the dispatched methods' own signatures
        // (`next: () -> self`, `self.build: () -> instance`) must resolve
        // to the concrete arg before unifying, mirroring
        // `check_interface_conformance`'s `sub_subst`/`singleton_instance_ty`
        // handling -- left unresolved, `Substitution::apply` passes
        // `Type::SelfType`/`Type::InstanceType` through untouched and this
        // arm's `TypeVariable` case would bind the caller's type var to
        // that raw sentinel instead of the real receiver type. Unlike
        // `check_interface_conformance`, which splits `self` into a
        // return-position (`self` -> the interface type) vs
        // param-position (`self` -> `sub`) substitution for coinductive-
        // soundness reasons specific to subtype checking, this arm applies
        // a single `self` -> `arg` substitution throughout: unification has
        // no soundness obligation to preserve, and resolving directly to
        // the concrete receiver is the intuitively correct binding for a
        // `next: () -> self`-shaped method.
        let singleton_instance_ty = is_singleton.then(|| {
            types_tbl.intern(Type::ClassInstance {
                name: sub_class_name,
                args: Vec::new(),
            })
        });

        for method_name in required_methods {
            let sub_lookup = if is_singleton {
                self.env
                    .lookup_singleton_method(&sub_class_name, method_name)
            } else if sub_is_interface {
                self.env
                    .lookup_interface_method_with_args(&sub_class_name, &sub_args, method_name)
            } else {
                self.env
                    .lookup_instance_method_with_args(&sub_class_name, &sub_args, method_name)
            };
            // Arg side doesn't implement this required method -- best
            // effort, skip just this method (subtyping's own
            // `check_interface_conformance` call still handles whether
            // that fires a diagnostic; unify never does).
            let Some((sub_method, sub_bindings)) = sub_lookup else {
                continue;
            };
            let Some((iface_method, iface_own_bindings)) = self
                .env
                .lookup_interface_method_with_args(&iface_name, iface_args, method_name)
            else {
                continue;
            };

            let mut sub_subst = Substitution::from_mapping(sub_bindings).with_self_type(arg);
            if let Some(instance_ty) = singleton_instance_ty {
                sub_subst = sub_subst.with_instance_type(instance_ty);
            }
            let substituted_sub_defs =
                definition_builder::substitute_method_defs(&sub_method.defs, &sub_subst, types_tbl);
            let sub_method_substituted = crate::definition::Method::from_defs(
                substituted_sub_defs,
                sub_method.accessibility,
            );

            let iface_subst = Substitution::from_mapping(iface_own_bindings).with_self_type(arg);
            let substituted_iface_defs = definition_builder::substitute_method_defs(
                &iface_method.defs,
                &iface_subst,
                types_tbl,
            );
            let iface_method_substituted = crate::definition::Method::from_defs(
                substituted_iface_defs,
                iface_method.accessibility,
            );

            // Interface and implementer overload *declaration order* is not
            // required to match (rbs core has both orders: `Array#each: ()
            // { (E) -> void } -> self | () -> Enumerator[E, self]` vs.
            // `Hash#select` / `Struct#each_pair`, which list the blockless
            // overload first) -- positional `.zip()` would silently unify
            // against the wrong overload whenever they don't line up, e.g.
            // pairing a required block-taking overload against the arg's
            // blockless one so the block-position unify below never fires.
            // `check_method_compatibility` (subtyping.rs) has the same
            // problem for subtype checking and solves it by searching for
            // *any* compatible sub overload; this uses arity + block-
            // presence as a lighter proxy (a full subtype-compatibility
            // search would need the same iface/param self-split
            // `check_interface_conformance` does) since unify only needs a
            // plausible candidate, not a soundness proof. Overloads with
            // the same shape but genuinely different concrete param types
            // (e.g. `put: (String) -> void | (Integer) -> void` satisfying
            // a single-overload `_Sink[E]`) stay inherently ambiguous by
            // structural dispatch alone -- this picks the first shape match
            // deterministically rather than trying to disambiguate further.
            for iface_ol in iface_method_substituted.method_types() {
                let Some(sub_ol) = sub_method_substituted
                    .method_types()
                    .find(|sub_ol| interface_overload_shape_matches(iface_ol, sub_ol))
                else {
                    continue;
                };
                self.unify_function_slots(&iface_ol.type_, &sub_ol.type_, bindings, mode);
                if let (Some(iface_blk), Some(sub_blk)) = (&iface_ol.block, &sub_ol.block) {
                    self.unify_function_slots(&iface_blk.type_, &sub_blk.type_, bindings, mode);
                }
            }
        }

        self.interface_unify_depth
            .set(self.interface_unify_depth.get() - 1);
    }

    /// Unify each parameter slot of two `FunctionType`s (used for proc-type
    /// function arms and block arms). Both sides must be `Typed`; mixed
    /// Typed/Untyped pairs yield no bindings since untyped carries no slot
    /// shape. Positional slots zip on min length; keywords match by name.
    fn unify_function_slots(
        &self,
        p_ft: &FunctionType,
        a_ft: &FunctionType,
        bindings: &mut FxHashMap<crate::type_param::TypeVarKey, Ty>,
        mode: UnifyMode,
    ) {
        let (FunctionType::Typed(p_f), FunctionType::Typed(a_f)) = (p_ft, a_ft) else {
            return;
        };
        self.unify_into_bindings(p_f.return_type, a_f.return_type, bindings, mode);
        for (&p, &a) in p_f
            .required_positionals
            .iter()
            .zip(a_f.required_positionals.iter())
        {
            self.unify_into_bindings(p, a, bindings, mode);
        }
        for (&p, &a) in p_f
            .optional_positionals
            .iter()
            .zip(a_f.optional_positionals.iter())
        {
            self.unify_into_bindings(p, a, bindings, mode);
        }
        for (&p, &a) in p_f
            .trailing_positionals
            .iter()
            .zip(a_f.trailing_positionals.iter())
        {
            self.unify_into_bindings(p, a, bindings, mode);
        }
        if let (Some(p_rest), Some(a_rest)) = (p_f.rest_positional, a_f.rest_positional) {
            self.unify_into_bindings(p_rest, a_rest, bindings, mode);
        }
        // For each p-side keyword name (flat across required+optional),
        // find a-side match across required+optional, else fall back to
        // a-side rest_keyword. Mirrors `subtyping::find_keyword` and
        // `collect_call_site_bindings`'s required → optional → rest chain
        // (Steep `check.rb:1061-1074`).
        for (p_name, p_ty) in p_f
            .required_keywords
            .iter()
            .chain(p_f.optional_keywords.iter())
        {
            let a_ty = a_f
                .required_keywords
                .iter()
                .chain(a_f.optional_keywords.iter())
                .find(|(n, _)| n == p_name)
                .map(|(_, ty)| *ty)
                .or(a_f.rest_keyword);
            if let Some(a_ty) = a_ty {
                self.unify_into_bindings(*p_ty, a_ty, bindings, mode);
            }
        }
        if let (Some(p_rk), Some(a_rk)) = (p_f.rest_keyword, a_f.rest_keyword) {
            self.unify_into_bindings(p_rk, a_rk, bindings, mode);
        }
    }

    /// Merge two candidate bindings for the same type var into a single
    /// `Ty`: dedup when structurally equal, otherwise produce a flattened
    /// `Type::Union`. Expects both inputs to be already widened (Literal →
    /// class, Tuple → Array, Record → Hash) by `unify_arg_against_param`.
    ///
    /// Fully flattens nested unions: if either input resolves to
    /// `Union[Union[A, B], C]` (possible when overload-return aggregation
    /// interns non-canonical unions), the result is `Union[A, B, C]`, not
    /// a re-nested one.
    fn union_merge_bindings(&self, existing: Ty, new_ty: Ty) -> Ty {
        if existing == new_ty {
            return existing;
        }
        let types_tbl = self.env.types();
        let mut members: Vec<Ty> = Vec::new();
        self.push_flat_union_members(existing, &mut members);
        self.push_flat_union_members(new_ty, &mut members);
        if members.len() == 1 {
            members[0]
        } else {
            types_tbl.intern(Type::Union(members))
        }
    }

    /// Append `ty` (or its members, recursively, if `ty` is a `Union`) to
    /// `out`, skipping structurally-equal duplicates. Guarantees `out`
    /// contains no `Type::Union` entries after the call returns.
    fn push_flat_union_members(&self, ty: Ty, out: &mut Vec<Ty>) {
        let types_tbl = self.env.types();
        match types_tbl.resolve(ty) {
            Type::Union(inner) => {
                for &m in inner {
                    self.push_flat_union_members(m, out);
                }
            }
            _ => {
                if !out.contains(&ty) {
                    out.push(ty);
                }
            }
        }
    }

    /// Synthesize a hash literal as a `Type::Record` when a Record hint
    /// exists — the first showcase of ADR-0012 bidirectional inference.
    ///
    /// Returns `None` when the literal cannot be interpreted as a Record:
    /// - the hint does not resolve to `Type::Record`
    /// - any entry is a `**splat` (not an `AssocNode`)
    /// - any key is not a `sym/str/int/true/false` literal
    ///   In those cases the caller falls back to `Hash[untyped, untyped]`.
    ///
    /// `extras` accumulates keys present in the literal but absent from the
    /// hint, across recursion into nested `{ … }` values. The synthesized
    /// Record drops those fields (so `check_record_structural` sees only
    /// hint-declared keys and does not double-fire `ArgumentTypeMismatch`).
    ///
    /// All synthesized fields are `required = true`: Steep inherits the
    /// hint's `required_keys` instead, but crema deliberately lets the
    /// literal state "this key exists" and leaves the optional-key
    /// bookkeeping to subtyping (`check_record_structural`).
    pub(super) fn synthesize_hash_as_record<'pr>(
        &self,
        elements: &[Node<'pr>],
        hint: Ty,
        extras: &mut Vec<RecordExtraKey>,
    ) -> Option<Ty> {
        let hint_fields = match self
            .env
            .types()
            .resolve(definition_builder::expand_alias(self.env, hint))
        {
            Type::Record { fields } => fields,
            _ => return None,
        };
        let known_keys: Vec<String> = hint_fields.iter().map(|(k, _, _)| k.display()).collect();
        let mut fields: Vec<(RecordKey, Ty, bool)> = Vec::new();
        for elem in elements.iter() {
            let assoc = elem.as_assoc_node()?;
            let key_node = assoc.key();
            let record_key = record_key_from_literal_node(&key_node)?;
            let value_node = assoc.value();
            let hint_field_ty = hint_fields
                .iter()
                .find(|(k, _, _)| k == &record_key)
                .map(|(_, ty, _)| *ty);
            match hint_field_ty {
                Some(field_hint) => {
                    // Recurse directly for nested Hash literals so inner
                    // extras flow into the caller's `extras` Vec — but
                    // only on success. If the nested call returns `None`
                    // (splat / non-literal key aborted it) the inner
                    // literal is a Hash, not a Record, and any extras it
                    // accumulated describe a shape we are no longer
                    // interpreting. Stage them in a local Vec and merge
                    // only when the nested synthesis commits.
                    let value_ty =
                        if let Some(nested_elems) = hash_or_keyword_hash_elements(&value_node) {
                            if matches!(
                                self.env.types().resolve(definition_builder::expand_alias(
                                    self.env, field_hint
                                )),
                                Type::Record { .. }
                            ) {
                                let mut nested_extras = Vec::new();
                                match self.synthesize_hash_as_record(
                                    &nested_elems,
                                    field_hint,
                                    &mut nested_extras,
                                ) {
                                    Some(ty) => {
                                        extras.extend(nested_extras);
                                        ty
                                    }
                                    None => self.hash_untyped_untyped(),
                                }
                            } else {
                                self.infer_type(&value_node, Some(field_hint))
                            }
                        } else {
                            self.infer_type(&value_node, Some(field_hint))
                        };
                    fields.push((record_key, value_ty, true));
                }
                None => {
                    extras.push(RecordExtraKey {
                        key: record_key,
                        known_keys: known_keys.clone(),
                        offset: key_node.location().start_offset(),
                    });
                }
            }
        }
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        Some(self.env.types().intern(Type::Record { fields }))
    }

    /// Decompose a hint into the candidate types a hash literal could
    /// match. Strips `Type::Alias`, peels `Type::Optional`, and flattens
    /// `Type::Union`; drops `Nil` and `Literal(Bool(false))` members
    /// (semantically equivalent to `partition_truthy` but scoped to hint
    /// expansion so the two responsibilities stay separate). Atoms become
    /// a one-element vec.
    pub(super) fn expand_hint_candidates(&self, hint: Ty) -> Vec<Ty> {
        let mut out = Vec::new();
        self.collect_hint_candidates(hint, &mut out);
        out
    }

    fn collect_hint_candidates(&self, hint: Ty, out: &mut Vec<Ty>) {
        let resolved = self
            .env
            .types()
            .resolve(definition_builder::expand_alias(self.env, hint));
        match resolved {
            Type::Nil => {}
            Type::Literal(Literal::Bool(false)) => {}
            Type::Optional(inner) => self.collect_hint_candidates(*inner, out),
            Type::Union(members) => {
                for &m in members {
                    self.collect_hint_candidates(m, out);
                }
            }
            other => {
                let canonical = self.env.types().intern(other.clone());
                if !out.contains(&canonical) {
                    out.push(canonical);
                }
            }
        }
    }

    /// Pick a single `Tuple` candidate from `hint` whose arity matches
    /// `expected_len`, returning its element types. Returns `None` when
    /// no Tuple candidate fits, when every candidate has a different
    /// arity (size mismatch falls back to `Array[E]` / `Array[untyped]`),
    /// or when multiple distinct Tuple shapes match (ambiguous). Parity
    /// with [`array_element_hint`]'s conservative "ambiguous = None"
    /// stance, not Steep's first-success-or-fallback flow.
    pub(super) fn array_tuple_hint(&self, hint: Ty, expected_len: usize) -> Option<Vec<Ty>> {
        let candidates = self.expand_hint_candidates(hint);
        let mut picked: Option<Vec<Ty>> = None;
        for c in candidates {
            if let Type::Tuple(members) = self.env.types().resolve(c)
                && members.len() == expected_len
            {
                match picked {
                    None => picked = Some(members.to_vec()),
                    Some(ref prev) if *prev == *members => {}
                    Some(_) => return None,
                }
            }
        }
        picked
    }

    /// Sibling of [`array_tuple_hint`] for splat-bearing literals: pick
    /// the single Tuple candidate without imposing an arity match.
    /// `[scalar, *splat, scalar]` can fit any tuple arity whose length
    /// ≥ scalar count, so the splat-tuple-expansion path validates
    /// arity itself after walking the splat. Returns `None` on no Tuple
    /// candidate or ambiguous (multiple distinct Tuple shapes).
    fn array_tuple_hint_loose(&self, hint: Ty) -> Option<Vec<Ty>> {
        let candidates = self.expand_hint_candidates(hint);
        let mut picked: Option<Vec<Ty>> = None;
        for c in candidates {
            if let Type::Tuple(members) = self.env.types().resolve(c) {
                match picked {
                    None => picked = Some(members.to_vec()),
                    Some(ref prev) if *prev == *members => {}
                    Some(_) => return None,
                }
            }
        }
        picked
    }

    /// Decide the element types that fill the splat slot in a hint-driven
    /// tuple-with-splat expansion. `splat_inner_ty` is the splat
    /// expression's inferred type; `slot_hints` is the hint positions the
    /// splat is expected to cover (after taking out leading/trailing
    /// scalar positions).
    ///
    /// - `Type::Tuple(elems)`: arity must match `slot_hints.len()` and
    ///   every elem must subtype its slot hint (Steep parity — `Case C`
    ///   bails here on element mismatch, the outer assertion then fires
    ///   `FalseAssertion`).
    /// - `Type::ClassInstance { ::Array, [T] }`: trust the hint — no
    ///   element-type check (Steep parity, `Case I`). The slot returns
    ///   the hint positions verbatim.
    /// - Anything else: bail.
    fn decide_splat_slot_types(&self, splat_inner_ty: Ty, slot_hints: &[Ty]) -> Option<Vec<Ty>> {
        match self.env.types().resolve(splat_inner_ty) {
            Type::Tuple(elems) => {
                if elems.len() != slot_hints.len() {
                    return None;
                }
                let subtyper = self.subtyper();
                for (sub, sup) in elems.iter().zip(slot_hints.iter()) {
                    if !subtyper.check(*sub, *sup) {
                        return None;
                    }
                }
                Some(elems.to_vec())
            }
            Type::ClassInstance { name, args }
                if *name == self.env.names().builtins().array && args.len() == 1 =>
            {
                let _ = args;
                Some(slot_hints.to_vec())
            }
            _ => None,
        }
    }

    /// Pick the single `Array[E]` element type carried by `hint`, or
    /// `None` when the hint is unrelated, ambiguous (multiple `Array[E]`
    /// candidates with differing `E`), or shaped differently. Used by
    /// the `ArrayNode` branch of [`infer_type`] to drive Stage 1
    /// bidirectional propagation for `[] #: Array[T]` and friends.
    fn array_element_hint(&self, hint: Ty) -> Option<Ty> {
        let candidates = self.expand_hint_candidates(hint);
        let array_name = &self.env.names().builtins().array;
        let mut element: Option<Ty> = None;
        for c in candidates {
            if let Type::ClassInstance { name, args } = self.env.types().resolve(c)
                && name == array_name
                && args.len() == 1
            {
                match element {
                    None => element = Some(args[0]),
                    Some(prev) if prev == args[0] => {}
                    Some(_) => return None,
                }
            }
        }
        element
    }

    /// Choose the Record-shaped hint candidate the literal should be
    /// synthesized against, or `None` to fall through to Hash[K,V] /
    /// untyped paths.
    ///
    /// Single Record candidate (plain Record, `Optional(Record)`,
    /// `Record | nil`): commit unconditionally so the caller can still
    /// emit `UnknownRecordKey` for extras — the interpretation is
    /// unambiguous, so a typo is the actionable signal.
    ///
    /// Multiple Record candidates (`Record_A | Record_B`): Steep
    /// `pick_one_of` parity. Trial-synthesize each candidate; commit on
    /// the first that produces no extras AND width-subtypes the hint.
    /// No-match returns `None` rather than committing to the first — the
    /// user wrote a disjunction, so "no branch matches" is the actionable
    /// signal, not per-branch typo guesses (same philosophy as the
    /// `pick_hint_overload` no-match fallback).
    pub(super) fn pick_record_hint<'pr>(&self, elements: &[Node<'pr>], hint: Ty) -> Option<Ty> {
        let candidates = self.expand_hint_candidates(hint);
        self.pick_record_hint_from_candidates(elements, &candidates)
    }

    /// Caller-supplied candidates variant — lets `infer_type` reuse one
    /// `expand_hint_candidates` allocation across the Record and
    /// Hash[K,V] passes without re-walking the hint.
    pub(super) fn pick_record_hint_from_candidates<'pr>(
        &self,
        elements: &[Node<'pr>],
        candidates: &[Ty],
    ) -> Option<Ty> {
        let record_candidates: Vec<Ty> = candidates
            .iter()
            .copied()
            .filter(|c| definition_builder::resolves_to_record(self.env, *c))
            .collect();
        match record_candidates.len() {
            0 => None,
            1 => Some(record_candidates[0]),
            _ => {
                let subtyper = self.subtyper();
                for c in &record_candidates {
                    let mut extras = Vec::new();
                    if let Some(ty) = self.synthesize_hash_as_record(elements, *c, &mut extras)
                        && extras.is_empty()
                        && subtyper.check(ty, *c)
                    {
                        return Some(*c);
                    }
                }
                None
            }
        }
    }

    fn hash_untyped_untyped(&self) -> Ty {
        self.hash_instance(Ty::UNTYPED, Ty::UNTYPED)
    }

    /// Propagate a `Hash[K, V]` hint into each AssocNode element so the
    /// inferred result is `Hash[Union(inferred_K), Union(inferred_V)]`
    /// (Steep `type_hash` style, `lib/steep/type_construction.rb:5190-5268`).
    ///
    /// Returns `None` when the hint is not `::Hash[_, _]` after alias
    /// expansion, or when any element is not an `AssocNode` (splat or
    /// other non-pair form). The caller then falls back to
    /// `Hash[untyped, untyped]`.
    ///
    /// The result must NOT be the hint itself: returning `hint` directly
    /// would let the call-site subtype check trivially pass against the
    /// param type, silently hiding literal/hint mismatches. By returning
    /// inferred unions we preserve the literal types so that
    /// `check_against_method_def` can surface `ArgumentTypeMismatch`.
    /// Empty hashes are the one principled exception: with no elements
    /// to infer from, we seed each union with the hint arg directly
    /// (matching Steep's `key_types << key_hint if key_hint`).
    pub(super) fn synthesize_hash_with_hint<'pr>(
        &self,
        elements: &[Node<'pr>],
        hint: Ty,
    ) -> Option<Ty> {
        let resolved = self
            .env
            .types()
            .resolve(definition_builder::expand_alias(self.env, hint));
        let (hash_name, key_hint, value_hint) = match resolved {
            Type::ClassInstance { name, args }
                if args.len() == 2 && *name == self.env.names().builtins().hash =>
            {
                (name, args[0], args[1])
            }
            _ => return None,
        };

        let mut key_members: Vec<Ty> = Vec::new();
        let mut value_members: Vec<Ty> = Vec::new();

        if elements.is_empty() {
            key_members.push(key_hint);
            value_members.push(value_hint);
        } else {
            for elem in elements.iter() {
                if let Some(assoc) = elem.as_assoc_node() {
                    let k_ty = self.infer_type(&assoc.key(), Some(key_hint));
                    let v_ty = self.infer_type(&assoc.value(), Some(value_hint));
                    self.push_flat_union_members(k_ty, &mut key_members);
                    self.push_flat_union_members(v_ty, &mut value_members);
                } else if let Some(splat) = elem.as_assoc_splat_node() {
                    // Steep `type_construction.rb:5241-5250` parity:
                    // synthesize the kwsplat value with a `Hash[K, V]`
                    // hint (Steep's `hint_hash`), then if the result is
                    // a Hash class instance, push its args into the
                    // pool. Record values widen via
                    // `widen_record_to_hash`; untyped / other shapes
                    // are dropped (Steep's `reject! { Any }`).
                    if let Some(value) = splat.value() {
                        self.absorb_kwsplat_value(
                            &value,
                            key_hint,
                            value_hint,
                            &mut key_members,
                            &mut value_members,
                        );
                    }
                } else {
                    return None;
                }
            }
        }

        // Steep `type_construction.rb:5260-5261`: empty pools fall back
        // to `Hash[untyped, untyped]`. Honor that, otherwise a literal
        // whose only element is an untyped kwsplat would synthesize as
        // `Hash[Bottom, Bottom]` via the empty-Union case.
        if key_members.is_empty() && value_members.is_empty() {
            return Some(self.hash_untyped_untyped());
        }

        let key_ty = if key_members.len() == 1 {
            key_members[0]
        } else {
            self.env.types().intern(Type::Union(key_members))
        };
        let value_ty = if value_members.len() == 1 {
            value_members[0]
        } else {
            self.env.types().intern(Type::Union(value_members))
        };

        Some(self.env.types().intern(Type::ClassInstance {
            name: *hash_name,
            args: vec![key_ty, value_ty],
        }))
    }

    /// Synthesize the value of a `**kwsplat` element under a `Hash[K, V]`
    /// hint, then route the result into `key_members` / `value_members`
    /// via [`absorb_kwsplat_ty`].
    fn absorb_kwsplat_value<'pr>(
        &self,
        value: &Node<'pr>,
        key_hint: Ty,
        value_hint: Ty,
        key_members: &mut Vec<Ty>,
        value_members: &mut Vec<Ty>,
    ) {
        let splat_hint = self.hash_instance(key_hint, value_hint);
        let inferred = self.infer_type(value, Some(splat_hint));
        self.absorb_kwsplat_ty(inferred, key_members, value_members);
    }

    /// Push the K/V contribution of an already-inferred kwsplat value
    /// type into the merge pool. `Hash[K_, V_]` instance contributes its
    /// args, `Record` widens through `widen_record_to_hash`, `Union` and
    /// `Optional` recurse into each member so a value typed as
    /// `Hash[K1,V1] | Hash[K2,V2]` still merges both branches. Anything
    /// else (untyped, mismatched class) drops silently — Steep
    /// `type_construction.rb:5241-5258` (`reject! { Any }`) parity.
    pub(super) fn absorb_kwsplat_ty(
        &self,
        ty: Ty,
        key_members: &mut Vec<Ty>,
        value_members: &mut Vec<Ty>,
    ) {
        let resolved = self
            .env
            .types()
            .resolve(definition_builder::expand_alias(self.env, ty));
        match resolved {
            Type::ClassInstance { name, args }
                if args.len() == 2 && *name == self.env.names().builtins().hash =>
            {
                self.push_flat_union_members(args[0], key_members);
                self.push_flat_union_members(args[1], value_members);
            }
            Type::Record { fields } => {
                let widened = self.widen_record_to_hash(fields);
                if let Type::ClassInstance { args, .. } = self.env.types().resolve(widened)
                    && args.len() == 2
                {
                    self.push_flat_union_members(args[0], key_members);
                    self.push_flat_union_members(args[1], value_members);
                }
            }
            Type::Union(members) => {
                for &m in members {
                    self.absorb_kwsplat_ty(m, key_members, value_members);
                }
            }
            Type::Optional(inner) => {
                self.absorb_kwsplat_ty(*inner, key_members, value_members);
            }
            _ => {}
        }
    }

    /// Compute the type of `self` in the current context.
    ///
    /// - class body (no method): `singleton(ClassName)`
    /// - singleton method (`def self.foo`): `singleton(ClassName)`
    /// - instance method: `ClassInstance(ClassName, type_params)`
    /// - top-level: instance of `::RBS::Unnamed::TopLevelSelfClass` (the main
    ///   object). Falls back to `UNTYPED` if that class is not loaded (e.g.,
    ///   tests without core RBS).
    pub(super) fn current_self_type(&self) -> Ty {
        if let Some(override_ty) = self.ctx.current_self_type_override() {
            return override_ty;
        }
        self.lexical_self_type()
    }

    /// Returns `true` when the lexical class context is a module (not a class)
    /// **and** the current method is an instance method (not `def self.m`).
    ///
    /// Used to suppress `UnexpectedSuper` in module instance methods: the
    /// include site is not statically known, so `super` resolution is deferred
    /// to runtime. Singleton methods (`def self.m`) are NOT suppressed because
    /// their super chain (module singleton → Module → Object → …) is resolved
    /// through `singleton_ancestors`, independent of any include site.
    pub(super) fn is_inside_module_definition(&self) -> bool {
        if self.ctx.is_singleton_method() {
            return false;
        }
        let Some(tn) = self.ctx.current_class_typename() else {
            return false;
        };
        self.env.class_or_module_kind(tn) == Some(DeclKindLocal::Module)
    }

    pub(super) fn lexical_self_type(&self) -> Ty {
        let Some(name) = self.ctx.current_class_typename() else {
            return self.env.class_instance_type(
                self.env
                    .names()
                    .parse_type_name("::RBS::Unnamed::TopLevelSelfClass"),
            );
        };
        // The stack now holds the parsed `TypeName`, so `self` no longer
        // re-parses a path string per reference. The legacy guard is kept:
        // a name RBS never declared in any category still yields `untyped`
        // self. The old guard checked this via a `Name` (String) intern
        // lookup, which happened to hit whenever `build_lowering_maps`
        // (definition/type_lowering.rs) had pre-interned the qualified path
        // — that walk covers `class_decls`, `interface_decls`,
        // `type_alias_decls`, `constant_decls` and `class_alias_decls`.
        // Checking those five id-keyed maps directly reproduces the same
        // "declared somewhere" semantics without the String round-trip; a
        // narrower `class_decls`-only check (matching
        // `is_inside_module_definition` above) diverges when a Ruby class
        // name collides with e.g. an RBS class alias of the same name.
        let declared = self.env.is_declared_class(name)
            || self.env.is_declared_interface(name)
            || self.env.is_declared_type_alias(name)
            || self.env.is_declared_constant(name)
            || self.env.is_declared_class_alias(name);
        if !declared {
            return Ty::UNTYPED;
        }
        let name = *name;
        if self.ctx.method_name().is_none() || self.ctx.is_singleton_method() {
            self.env.types().class_singleton(name)
        } else {
            let args = type_params_as_variable_args(
                self.env.class_type_params_by_type_name(&name),
                &name,
                self.env.types(),
            );
            self.env.types().intern(Type::ClassInstance { name, args })
        }
    }

    pub(super) fn lookup_local_variable_for_read(&self, name: crate::name::Name) -> Option<Ty> {
        let (ty, binding_self_override) = self.ctx.self_type_override_at_binding(name)?;
        let current_self_override = self.ctx.current_self_type_override();
        if current_self_override == binding_self_override {
            return Some(ty);
        }
        let binding_self_type = binding_self_override.unwrap_or_else(|| self.lexical_self_type());
        let current_self_type = current_self_override.unwrap_or_else(|| self.lexical_self_type());
        if current_self_type == binding_self_type {
            return Some(ty);
        }
        Some(
            Substitution::new()
                .with_self_type(binding_self_type)
                .apply(ty, self.env.types()),
        )
    }

    /// Type of a `self` receiver (an explicit `self` node or an implicit
    /// receiver). A declared class yields the opaque `Ty::SELF_TYPE` so
    /// `-> self` identity survives inference; an undeclared class (untyped
    /// self) stays untyped so `self.foo` is suppressed under gradual typing.
    /// Method lookup re-concretizes `SELF_TYPE` via `current_self_type()` at
    /// the call boundary (`resolve_call_target`).
    pub(super) fn self_receiver_type(&self) -> Ty {
        if self.current_self_type().is_untyped() {
            Ty::UNTYPED
        } else {
            Ty::SELF_TYPE
        }
    }

    pub(super) fn freeze_self_type_for_lvar_binding(&self, ty: Ty) -> Ty {
        if !self.ty_contains_self_type(ty) {
            return ty;
        }
        Substitution::new()
            .with_self_type(self.current_self_type())
            .apply(ty, self.env.types())
    }

    fn ty_contains_self_type(&self, ty: Ty) -> bool {
        match self.env.types().resolve(ty) {
            Type::SelfType => true,
            Type::ClassInstance { args, .. }
            | Type::Interface { args, .. }
            | Type::Alias { args, .. } => args.iter().any(|&a| self.ty_contains_self_type(a)),
            Type::Union(members) | Type::Intersection(members) | Type::Tuple(members) => {
                members.iter().any(|&m| self.ty_contains_self_type(m))
            }
            Type::Optional(inner) => self.ty_contains_self_type(*inner),
            Type::Proc {
                type_,
                self_type,
                block,
            } => {
                self.function_type_contains_self_type(type_)
                    || self_type.is_some_and(|ty| self.ty_contains_self_type(ty))
                    || block
                        .as_ref()
                        .is_some_and(|b| self.block_contains_self_type(b))
            }
            Type::Record { fields } => fields
                .iter()
                .any(|(_, ty, _)| self.ty_contains_self_type(*ty)),
            _ => false,
        }
    }

    fn block_contains_self_type(&self, block: &crate::types::Block) -> bool {
        self.function_type_contains_self_type(&block.type_)
            || block
                .self_type
                .is_some_and(|ty| self.ty_contains_self_type(ty))
    }

    fn function_type_contains_self_type(&self, type_: &FunctionType) -> bool {
        match type_ {
            FunctionType::Typed(f) => {
                f.required_positionals
                    .iter()
                    .chain(f.optional_positionals.iter())
                    .chain(f.trailing_positionals.iter())
                    .any(|&ty| self.ty_contains_self_type(ty))
                    || f.rest_positional
                        .is_some_and(|ty| self.ty_contains_self_type(ty))
                    || f.required_keywords
                        .iter()
                        .chain(f.optional_keywords.iter())
                        .any(|(_, ty)| self.ty_contains_self_type(*ty))
                    || f.rest_keyword
                        .is_some_and(|ty| self.ty_contains_self_type(ty))
                    || self.ty_contains_self_type(f.return_type)
            }
            FunctionType::Untyped(f) => self.ty_contains_self_type(f.return_type),
        }
    }

    /// Build a `SubtypeChecker` bound to the current `self` for SUB-side
    /// `self <: X` widening. Every in-TypeChecker subtype check goes through
    /// this (or [`subtyper_preserving`](Self::subtyper_preserving)) so the
    /// self binding is wired in one place rather than repeated at each
    /// construction site.
    pub(super) fn subtyper(&self) -> SubtypeChecker<'env> {
        SubtypeChecker::new(self.env).with_self_bound(self.current_self_type())
    }

    /// Like [`subtyper`](Self::subtyper) but preserves free type variables
    /// instead of treating them as wildcards (method-body / return checks).
    pub(super) fn subtyper_preserving(&self) -> SubtypeChecker<'env> {
        SubtypeChecker::preserving_type_variables(self.env)
            .with_self_bound(self.current_self_type())
    }

    /// Side-effecting walk of a lambda literal's body, then the read-only
    /// literal type from `infer_type`.
    ///
    /// Params bind in a pushed `ScopeKind::Block` scope — the same shape
    /// `setup_block_scope` uses for blocks, so body-local writes stay
    /// inside the lambda while outer locals remain visible through the
    /// scope chain. The binding table is `lambda_param_overlay` (shared
    /// with the read-only `lambda_literal_type` path, `it` / numbered
    /// params included).
    ///
    /// Param types come from the hint when it resolves to a typed Proc
    /// (Steep parity: a hinted lambda checks its body against the
    /// expected param types); unhinted params bind UNTYPED so an
    /// undeclared-shape lambda can't produce false positives.
    fn check_lambda_node<'pr>(&mut self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        if let Some(lambda) = node.as_lambda_node() {
            let param_tys = self.lambda_walk_param_types(&lambda, hint);
            let overlay = self.lambda_param_overlay(lambda.parameters(), &param_tys);
            self.ctx.push_scope(ScopeKind::Block);
            for (name, ty) in overlay {
                self.ctx.set_local_variable(name, ty);
            }
            // Default-value expressions (`->(x = 1.upcase) { }`) live in
            // the parameters subtree, which the default walker used to
            // reach and this arm now owns. They evaluate at call time in
            // the lambda's own scope, so the walk belongs inside the
            // pushed scope and after the param bindings — a later
            // default can read an earlier param (`->(a, b = a.foo) {}`).
            if let Some(params_node) = lambda.parameters() {
                self.visit(&params_node);
            }
            if let Some(body) = lambda.body() {
                self.check_node(&body, None);
            }
            self.ctx.pop_scope();
        }
        self.infer_type(node, hint)
    }

    /// Param types for the body walk's binding table: the hint's declared
    /// positionals when the hint is a typed Proc, otherwise UNTYPED for
    /// every declared param slot (required / optional / rest). The
    /// all-UNTYPED vector is built explicitly rather than left empty so
    /// each name is actually bound — an unbound name would fall through
    /// `lookup_local_variable` to an enclosing lambda's same-named param.
    fn lambda_walk_param_types<'pr>(
        &self,
        lambda: &ruby_prism::LambdaNode<'pr>,
        hint: Option<Ty>,
    ) -> Vec<Ty> {
        if let Some(hint) = hint
            && let Type::Proc {
                type_: FunctionType::Typed(function),
                ..
            } = self.env.types().resolve(hint)
        {
            return function.required_positionals.clone();
        }
        let Some(params) = self.lambda_parameters(lambda.parameters()) else {
            return vec![Ty::UNTYPED];
        };
        let count = params.requireds().len()
            + params.optionals().len()
            + usize::from(params.rest().is_some());
        vec![Ty::UNTYPED; count]
    }

    fn lambda_literal_type<'pr>(&self, node: &Node<'pr>, hint: Option<Ty>) -> Ty {
        if let Some(hint) = hint
            && let Type::Proc {
                type_: FunctionType::Typed(function),
                ..
            } = self.env.types().resolve(hint)
            && let Some(lambda) = node.as_lambda_node()
        {
            let overlay =
                self.lambda_param_overlay(lambda.parameters(), &function.required_positionals);
            let actual_return = match lambda.body() {
                Some(body) => self.with_overlay(overlay, |checker| checker.infer_type(&body, None)),
                None => Ty::NIL,
            };
            if actual_return.is_untyped()
                || function.return_type == Ty::VOID
                || self.subtyper().check(actual_return, function.return_type)
            {
                return hint;
            }

            let mut function = function.clone();
            function.return_type = actual_return;
            return self.env.types().intern(Type::Proc {
                type_: FunctionType::Typed(function),
                self_type: None,
                block: None,
            });
        }

        if let Some(lambda) = node.as_lambda_node() {
            let function = self.unhinted_lambda_function(lambda);
            return self.env.types().intern(Type::Proc {
                type_: FunctionType::Typed(function),
                self_type: None,
                block: None,
            });
        }

        self.env
            .class_instance_type(self.env.names().builtins().proc)
    }

    fn unhinted_lambda_function<'pr>(&self, lambda: ruby_prism::LambdaNode<'pr>) -> Function {
        let mut function = Function::empty(Ty::UNTYPED);
        let mut param_tys = Vec::new();
        if let Some(params) = self.lambda_parameters(lambda.parameters()) {
            function.required_positionals = vec![Ty::UNTYPED; params.requireds().len()];
            function.optional_positionals = vec![Ty::UNTYPED; params.optionals().len()];
            function.rest_positional = params.rest().map(|_| Ty::UNTYPED);
            param_tys.extend(
                function
                    .required_positionals
                    .iter()
                    .chain(function.optional_positionals.iter())
                    .copied(),
            );
            if let Some(rest) = function.rest_positional {
                param_tys.push(rest);
            }
        } else if let Some(params_node) = lambda.parameters() {
            // Implicit `it` / numbered params are valid in lambdas too and
            // determine arity (`-> { it }.arity == 1`, `-> { _2 }.arity == 2`).
            // Unlike blocks, lambdas never auto-splat, so `maximum` maps to
            // plain required positionals.
            if params_node.as_it_parameters_node().is_some() {
                function.required_positionals = vec![Ty::UNTYPED];
            } else if let Some(numbered) = params_node.as_numbered_parameters_node() {
                function.required_positionals = vec![Ty::UNTYPED; numbered.maximum() as usize];
            }
            param_tys.extend(function.required_positionals.iter().copied());
        }
        let overlay = self.lambda_param_overlay(lambda.parameters(), &param_tys);
        function.return_type = match lambda.body() {
            Some(body) => self.with_overlay(overlay, |checker| checker.infer_type(&body, None)),
            None => Ty::NIL,
        };
        function
    }

    fn lambda_param_overlay<'pr>(
        &self,
        params: Option<Node<'pr>>,
        param_tys: &[Ty],
    ) -> FxHashMap<Name, Ty> {
        let mut overlay: FxHashMap<Name, Ty> = FxHashMap::default();
        if let Some(params_node) = &params {
            // Implicit `it` / numbered params (structural twin of the arms in
            // `setup_block_scope` / `augment_bindings_with_block_body`, minus
            // auto-splat: lambdas pass positionals 1:1, so each `_n` takes
            // `param_tys[n-1]` directly).
            if params_node.as_it_parameters_node().is_some() {
                let it_ty = param_tys.first().copied().unwrap_or(Ty::UNTYPED);
                overlay.insert(self.checker_names().intern("it"), it_ty);
                return overlay;
            }
            if let Some(numbered) = params_node.as_numbered_parameters_node() {
                for index in 0..numbered.maximum() as usize {
                    let ty = param_tys.get(index).copied().unwrap_or(Ty::UNTYPED);
                    overlay.insert(self.checker_names().intern(&format!("_{}", index + 1)), ty);
                }
                return overlay;
            }
        }
        let Some(params) = self.lambda_parameters(params) else {
            return overlay;
        };

        for (index, param) in params.requireds().iter().enumerate() {
            if let Some(required) = param.as_required_parameter_node()
                && let Some(expected_type) = param_tys.get(index)
            {
                let name_str = String::from_utf8_lossy(required.name().as_slice());
                let name = self.checker_names().intern(&name_str);
                overlay.insert(name, *expected_type);
            }
        }

        let optional_offset = params.requireds().len();
        for (index, param) in params.optionals().iter().enumerate() {
            if let Some(optional) = param.as_optional_parameter_node()
                && let Some(expected_type) = param_tys.get(optional_offset + index)
            {
                let name_str = String::from_utf8_lossy(optional.name().as_slice());
                let name = self.checker_names().intern(&name_str);
                overlay.insert(name, *expected_type);
            }
        }

        let rest_offset = optional_offset + params.optionals().len();
        if let Some(rest_node) = params.rest()
            && let Some(rest_param) = rest_node.as_rest_parameter_node()
            && let Some(name_id) = rest_param.name()
            && let Some(expected_type) = param_tys.get(rest_offset)
        {
            let name_str = String::from_utf8_lossy(name_id.as_slice());
            let name = self.checker_names().intern(&name_str);
            overlay.insert(name, *expected_type);
        }

        overlay
    }

    fn lambda_parameters<'pr>(
        &self,
        params: Option<Node<'pr>>,
    ) -> Option<ruby_prism::ParametersNode<'pr>> {
        let params_node = params?;
        if let Some(params) = params_node.as_parameters_node() {
            return Some(params);
        }
        params_node
            .as_block_parameters_node()
            .and_then(|block_params| block_params.parameters())
    }

    /// Infer the type of a call's receiver (explicit or implicit self).
    ///
    /// Safe-navigation (`x&.foo`) drops nil from the receiver so dispatch
    /// runs on the non-nil components only — Steep `(T | nil)&.foo` reports
    /// `Type ::T does not have method foo`, not the union. The strip is
    /// no-op for receivers without nil and preserves the input when the
    /// receiver is pure nil so the existing `::NilClass` diagnostic stays
    /// (bot-label parity with Steep is a separate todo).
    pub(super) fn infer_receiver_type<'pr>(&self, call: &CallNode<'pr>) -> Ty {
        if let Some(receiver) = call.receiver() {
            let ty = if receiver.as_lambda_node().is_some() {
                self.env
                    .class_instance_type(self.env.names().builtins().proc)
            } else if let Some(asserted) = self.lookup_receiver_trailing_assertion(&receiver) {
                // `(expr #: T).method_call` — the receiver carries its own
                // trailing type assertion. `infer_type` below has no
                // knowledge of assertions (that's `apply_statement_assertion_gate`'s
                // job, and it only fires at statement position), so
                // without this the receiver falls back to its natural
                // inferred type and the assertion is silently dropped at
                // the very position it exists to override.
                asserted
            } else {
                self.infer_type(&receiver, None)
            };
            if call.is_safe_navigation() {
                self.unwrap_optional(ty)
            } else {
                ty
            }
        } else {
            self.self_receiver_type()
        }
    }

    /// CallSite-aware variant. Super calls have no receiver node in prism;
    /// their receiver is always the enclosing method's `self`, which
    /// `check_super_node` represents by passing `current_self_type()` down
    /// to `check_against_method_def`. Re-derive the same concrete self here
    /// so bound diagnostics ( `TypeArgumentBoundViolation` 's
    /// `container_name` ) resolve to the actual class instead of the
    /// opaque `SELF_TYPE`.
    pub(super) fn receiver_type_for_site<'pr>(&self, site: super::calls::CallSite<'_, 'pr>) -> Ty {
        match site {
            super::calls::CallSite::Call(c) => self.infer_receiver_type(c),
            super::calls::CallSite::Super(_) => self.current_self_type(),
        }
    }

    /// Method name for a CallSite. CallNode carries it inline; SuperNode
    /// reuses the enclosing method context's name (super calls the same
    /// method in an ancestor). Used by `check_block` /
    /// `BlockBodyTypeMismatch` to label the call in diagnostics.
    pub(super) fn method_name_for_site<'pr>(
        &self,
        site: super::calls::CallSite<'_, 'pr>,
    ) -> String {
        match site {
            super::calls::CallSite::Call(c) => {
                String::from_utf8_lossy(c.name().as_slice()).to_string()
            }
            super::calls::CallSite::Super(_) => self
                .ctx
                .method_name()
                .map(|s| s.to_string())
                .unwrap_or_default(),
        }
    }

    /// Drop nil components from a type, mirroring Steep's
    /// `unwrap_optional` (`lib/steep/ast/types/factory.rb:428`). Walks
    /// through `Type::Alias`, `Type::Optional`, and `Type::Union` so
    /// alias-of-Optional (`type ot = ::String?`) and `T1 | T2 | nil`
    /// both reduce to their non-nil portion. Returns the input
    /// unchanged when no non-nil component remains so safe-navigation
    /// callers keep the `::NilClass` diagnostic for pure-nil receivers
    /// (and so `Array[nil].compact` stays `Array[nil]` — Steep collapses
    /// to `Array[Bot]` via an explicit `|| Bot.instance` fallback; we
    /// intentionally diverge per todo
    /// `mid_array_hash_compact_narrow_optional_steep_parity` which scopes
    /// the Bot fallback out).
    ///
    /// Shared by safe-navigation receiver narrowing
    /// (`infer_receiver_type`) and the `compact` special-method
    /// return-type rewrite (`apply_compact_special_return`).
    pub(super) fn unwrap_optional(&self, ty: Ty) -> Ty {
        let mut members: Vec<Ty> = Vec::new();
        let mut visited: FxHashSet<Ty> = FxHashSet::default();
        self.collect_non_nil(ty, &mut members, &mut visited);
        if members.is_empty() {
            ty
        } else {
            union_of_many(&members, self.env.types())
        }
    }

    /// `visited` guards against cyclic aliases (`type a = ::String | a`)
    /// reaching this helper through unvalidated RBS. The pattern matches
    /// `flatten_alias_union_into` in `calls.rs`, which solves the same
    /// recursion-via-Union problem on the dispatch side.
    fn collect_non_nil(&self, ty: Ty, out: &mut Vec<Ty>, visited: &mut FxHashSet<Ty>) {
        if !visited.insert(ty) {
            return;
        }
        let resolved = definition_builder::expand_alias(self.env, ty);
        match self.env.types().resolve(resolved) {
            Type::Nil => {}
            Type::Optional(inner) => self.collect_non_nil(*inner, out, visited),
            Type::Union(members) => {
                for &m in members {
                    self.collect_non_nil(m, out, visited);
                }
            }
            _ => out.push(resolved),
        }
    }

    /// Resolve `Foo` (a `ConstantReadNode`) through `ConstantResolver`
    /// against the current lexical scope. `node` is kept for the
    /// `--verbose` location string.
    ///
    /// rbs `Resolver::ConstantResolver::resolve(name, context:)`
    /// handles the lexical scope + ancestor chain + Object/toplevel
    /// splice in one shot, so the type checker only needs to lift the
    /// `Context::class_stack` into a [`ConstantContext`].
    fn resolve_constant_read<'pr>(&self, name_str: &str, node: &Node<'pr>) -> Ty {
        self.try_resolve_constant_read(name_str, node)
            .map(|c| c.ty)
            .unwrap_or(Ty::UNTYPED)
    }

    /// Core constant resolution shared by the silent `&self` type-query
    /// path (`resolve_constant_read`) and the side-effecting `&mut self`
    /// path (`check_constant_read`). Resolves `name_str` against the
    /// lexical scope and returns the full [`ResolverConstant`] on a hit
    /// (extract's `constant` record needs the absolute name, not just
    /// the type), `None` on a miss.
    /// Emits the `--verbose` location string either way; it never pushes
    /// a diagnostic, so callers that need `UnknownConstant` must do so
    /// themselves. Bare reads emit in `check_constant_read` (exactly once
    /// per occurrence — see its doc for the duplicate-emit story); a class
    /// superclass emits at its `visit_class_node` site.
    pub(super) fn try_resolve_constant_read<'pr>(
        &self,
        name_str: &str,
        node: &Node<'pr>,
    ) -> Option<crate::definition::ResolverConstant> {
        // Hot path: skip the SourceLocation construction (which scans
        // `source[..offset]` to count chars) and read raw byte offsets
        // straight from `prism`. The verbose log only needs the byte
        // numbers; a SourceLocation here would be discarded on every
        // resolved constant read.
        let start_byte = node.location().start_offset() as u32;
        let names = self.env.names();
        let sym = names.intern_symbol(name_str);
        let context = constant_context_from_class_stack(self.ctx.class_stack());
        match self.env.resolve_constant(sym, &context) {
            Some(constant) => {
                self.verbose_log(format_args!(
                    "{}:{} constant {} → {}",
                    start_byte,
                    start_byte,
                    name_str,
                    self.display_type(constant.ty)
                ));
                Some(constant)
            }
            None => {
                self.verbose_log(format_args!(
                    "{}:{} constant {} → not found",
                    start_byte, start_byte, name_str,
                ));
                None
            }
        }
    }

    /// Side-effecting constant read: resolve `name_str` and, on a miss,
    /// push one `UnknownConstant` diagnostic at the node's location.
    /// Returns the resolved type (`untyped` on a miss, so downstream
    /// type flow is unchanged).
    ///
    /// Emit lives here — a `&mut self` path that runs exactly once per
    /// syntactic occurrence — rather than in `resolve_constant_read`,
    /// because `infer_receiver_type` re-infers a call's receiver through
    /// the silent `&self` path several times per call (peek paths in
    /// `calls.rs`); emitting there multiplied the same `Foo` into 6 / 16
    /// duplicates for `Foo.new` / `Foo.new.bar`.
    pub(super) fn check_constant_read<'pr>(&mut self, node: &ConstantReadNode<'pr>) -> Ty {
        use crate::diagnostic::{ConstantKind, Diagnostic, DiagnosticKind};
        let name_str = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let generic = node.as_node();
        let start_offset = generic.location().start_offset();
        let end_offset = generic.location().end_offset();
        match self.try_resolve_constant_read(&name_str, &generic) {
            Some(constant) => {
                let deprecated =
                    self.check_deprecated_constant_read(&name_str, start_offset, end_offset);
                let state = if deprecated {
                    crate::extract::constant_state::ERROR
                } else {
                    crate::extract::constant_state::TYPED
                };
                self.record_extract_constant(
                    start_offset,
                    end_offset,
                    state,
                    name_str,
                    Some(&constant),
                    Some(constant.ty),
                );
                constant.ty
            }
            None => {
                let position = self.byte_range_to_location(start_offset, end_offset);
                let context = constant_context_from_class_stack(self.ctx.class_stack());
                let candidate_scope = CandidateScope::Context(context);
                let searched_namespaces = self.searched_namespaces_for_scope(&candidate_scope);
                self.record_extract_constant(
                    start_offset,
                    end_offset,
                    crate::extract::constant_state::UNKNOWN_CONSTANT,
                    name_str.clone(),
                    None,
                    None,
                );
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::UnknownConstant {
                        name: name_str.clone(),
                        path: name_str,
                        kind: ConstantKind::Constant,
                        did_you_mean: Vec::new(),
                        searched_namespaces,
                        candidate_scope,
                    },
                });
                Ty::UNTYPED
            }
        }
    }

    /// Emit `Ruby::DeprecatedReference` when a global variable
    /// reference (read or write) resolves to a declaration marked
    /// `%a{deprecated}`. Shared entry point for the gvar read arm and
    /// the four gvar write visitors so all five sites emit at the
    /// exact same node span. Runs only on resolve-hit so
    /// `UnknownGlobalVariable` and `DeprecatedReference` never
    /// co-fire (Steep parity).
    pub(super) fn check_deprecated_global_ref(
        &self,
        name_str: &str,
        start_offset: usize,
        end_offset: usize,
    ) {
        let Some(sym) = self.env.names().lookup_symbol(name_str) else {
            return;
        };
        let Some(message) = self.env.global_deprecated_message(sym) else {
            return;
        };
        let location = self.byte_range_to_location(start_offset, end_offset);
        self.push_diagnostic(crate::diagnostic::Diagnostic {
            scope: None,
            location,
            kind: crate::diagnostic::DiagnosticKind::DeprecatedReference {
                subject_kind: crate::diagnostic::DeprecatedSubjectKind::GlobalVariable,
                subject_name: name_str.to_string(),
                message,
            },
        });
    }

    /// Emit `Ruby::DeprecatedReference` when a bare constant read
    /// resolves to a declaration marked `%a{deprecated}`. Runs only on
    /// the resolve-hit path so `UnknownConstant` and
    /// `DeprecatedReference` never co-fire (Steep parity — Steep's
    /// `check_deprecation_constant` is only called from resolved paths
    /// in `type_construction.rb:1644`, `1664`).
    /// Returns whether the diagnostic fired, so the extract-mode read
    /// record can classify the site `error` without a span-keyed flag
    /// (resolve, diagnose, and record all live in one function here,
    /// unlike the method-call paths).
    fn check_deprecated_constant_read(
        &self,
        name_str: &str,
        start_offset: usize,
        end_offset: usize,
    ) -> bool {
        let names = self.env.names();
        let sym = names.intern_symbol(name_str);
        let context = constant_context_from_class_stack(self.ctx.class_stack());
        let Some(constant) = self.env.resolve_constant(sym, &context) else {
            return false;
        };
        let Some(message) = self.deprecated_message_for_type_name(&constant) else {
            return false;
        };
        let location = self.byte_range_to_location(start_offset, end_offset);
        self.push_diagnostic(crate::diagnostic::Diagnostic {
            scope: None,
            location,
            kind: crate::diagnostic::DiagnosticKind::DeprecatedReference {
                subject_kind: crate::diagnostic::DeprecatedSubjectKind::Constant,
                subject_name: name_str.to_string(),
                message,
            },
        });
        true
    }

    /// Constant-path counterpart. Steep flags the **leaf** of a path
    /// only (`type_construction.rb:1664` — one call per path), so
    /// intermediate deprecated segments stay silent. Bails on non-Walked
    /// or non-Resolved outcomes so `UnknownConstant` / intermediate-value
    /// paths don't co-fire.
    /// Returns whether the diagnostic fired — see
    /// `check_deprecated_constant_read` for the extract-record rationale.
    pub(super) fn check_deprecated_constant_path<'pr>(
        &self,
        path: &ruby_prism::ConstantPathNode<'pr>,
        outcome: &ConstantPathOutcome,
    ) -> bool {
        let ConstantPathOutcome::Walked { segments, kind, .. } = outcome else {
            return false;
        };
        let ConstantPathOutcomeKind::Resolved(_, leaf_opt) = kind else {
            return false;
        };
        let Some(leaf) = leaf_opt.as_ref() else {
            return false;
        };
        let Some(message) = self.deprecated_message_for_type_name(leaf) else {
            return false;
        };
        let joined = segments
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join("::");
        // Reproduce the leading `::` when the path was absolute so JSON
        // consumers can round-trip the identifier.
        let subject_name = if matches!(
            outcome,
            ConstantPathOutcome::Walked {
                is_absolute: true,
                ..
            }
        ) {
            format!("::{}", joined)
        } else {
            joined
        };
        let location = self
            .byte_range_to_location(path.location().start_offset(), path.location().end_offset());
        self.push_diagnostic(crate::diagnostic::Diagnostic {
            scope: None,
            location,
            kind: crate::diagnostic::DiagnosticKind::DeprecatedReference {
                subject_kind: crate::diagnostic::DeprecatedSubjectKind::Constant,
                subject_name,
                message,
            },
        });
        true
    }

    /// Aggregate annotations for a resolved constant / class / module /
    /// alias and run the deprecation match. Mirrors Steep's
    /// `check_deprecation_constant` (`type_construction.rb:5355-5387`):
    /// class / module reopens contribute every Signature-side decl's
    /// annotations flat-mapped, aliases and plain `FOO: T` decls
    /// contribute their single decl's annotations. Ruby-side inline
    /// class/module declarations have no `%a{}` syntax so their decls
    /// are skipped (Steep does the same via `is_a?(RBS::AST::Declarations::Base)`).
    fn deprecated_message_for_type_name(
        &self,
        constant: &crate::definition::ResolverConstant,
    ) -> Option<Option<String>> {
        let type_name = &constant.name;
        match &constant.origin {
            ConstantOrigin::Module { .. } => self
                .env
                .class_or_module_deprecated_message(type_name)
                .or_else(|| self.env.class_alias_deprecated_message(type_name)),
            ConstantOrigin::Constant => self.env.constant_deprecated_message(type_name),
        }
    }

    pub(super) fn searched_namespaces_for_scope(&self, scope: &CandidateScope) -> Vec<String> {
        match scope {
            CandidateScope::Context(ctx) => {
                let scopes = ctx.scopes();
                if scopes.is_empty() {
                    return vec!["::".to_string()];
                }
                scopes
                    .iter()
                    .map(|tn| self.env.names().display_type_name(*tn))
                    .collect()
            }
            CandidateScope::Children(tn) => vec![self.env.names().display_type_name(*tn)],
            CandidateScope::None => Vec::new(),
        }
    }

    /// `did_you_mean` candidate scope drawn from the constants visible in
    /// the current lexical scope (the full class stack). Used by the
    /// superclass miss, which is resolved before the class is pushed.
    /// Building-only — actual suggestion computation is deferred to
    /// `crate::diagnostic::join_did_you_mean` at output time (ADR-0032
    /// Decision 5a), so this never touches the constant table.
    pub(super) fn candidate_scope_in_current_scope(&self) -> CandidateScope {
        let context = constant_context_from_class_stack(self.ctx.class_stack());
        CandidateScope::Context(context)
    }

    pub(super) fn searched_namespaces_in_current_scope(&self) -> Vec<String> {
        self.searched_namespaces_for_scope(&self.candidate_scope_in_current_scope())
    }

    /// `did_you_mean` candidate scope from the *enclosing* scope — the
    /// class stack minus its innermost entry. Used by the class / module
    /// declaration-name miss, which runs after the (undeclared) name has
    /// already been pushed, so the name being declared is dropped to look
    /// at its siblings. Building-only, see `candidate_scope_in_current_scope`.
    pub(super) fn candidate_scope_in_enclosing_scope(&self) -> CandidateScope {
        let stack = self.ctx.class_stack();
        let outer = &stack[..stack.len().saturating_sub(1)];
        CandidateScope::Context(constant_context_from_class_stack(outer))
    }

    pub(super) fn searched_namespaces_in_enclosing_scope(&self) -> Vec<String> {
        self.searched_namespaces_for_scope(&self.candidate_scope_in_enclosing_scope())
    }

    /// `did_you_mean` candidate scope for a constant *write*, mirroring
    /// `ConstantResolver::resolve_in_namespace`: the innermost scope's
    /// children, or top-level when there is no enclosing scope. Keeps the
    /// `CandidateScope` mapping next to its siblings so callers never
    /// build the enum themselves. Building-only, see
    /// `candidate_scope_in_current_scope`.
    pub(super) fn candidate_scope_in_namespace(
        &self,
        scope: Option<&crate::type_name::TypeName>,
    ) -> CandidateScope {
        match scope {
            Some(tn) => CandidateScope::Children(*tn),
            None => CandidateScope::Context(ConstantContext::toplevel()),
        }
    }

    pub(super) fn searched_namespaces_in_namespace(
        &self,
        scope: Option<&crate::type_name::TypeName>,
    ) -> Vec<String> {
        self.searched_namespaces_for_scope(&self.candidate_scope_in_namespace(scope))
    }

    /// Side-effecting constant-path read: resolve `path` and, on the
    /// first unresolved segment, push one `UnknownConstant`
    /// (`kind=Constant`) at that segment's location. Returns `untyped`
    /// on any miss so downstream type flow is unchanged.
    ///
    /// Mirrors `check_constant_read` for the path case. Emit lives on
    /// the `&mut self` side-effect path because `infer_receiver_type`
    /// peeks a call's receiver through the silent `&self` resolver
    /// several times per call (see the doc on `check_constant_read`).
    pub(super) fn check_constant_path_node<'pr>(
        &mut self,
        path: &ruby_prism::ConstantPathNode<'pr>,
    ) -> Ty {
        use crate::diagnostic::ConstantKind;
        self.check_constant_path_inner(path, Some(ConstantKind::Constant))
    }

    fn check_constant_path_inner<'pr>(
        &mut self,
        path: &ruby_prism::ConstantPathNode<'pr>,
        leaf_kind: Option<crate::diagnostic::ConstantKind>,
    ) -> Ty {
        let outcome = self.resolve_constant_path_outcome(path);
        self.finish_constant_path_read(path, &outcome, leaf_kind);
        outcome.resolved_ty()
    }

    /// Side-effect finisher shared by the two constant-path *read*
    /// entries (`check_constant_path_inner` and
    /// `visit_constant_path_node`): emit the outcome's diagnostics and
    /// push the extract-mode `constant` record. Keeping the record here
    /// — not inside `emit_constant_path_outcome` — is deliberate: the
    /// emit helper also serves superclass, declaration-head, and
    /// write-target paths, which must never produce a read record.
    pub(super) fn finish_constant_path_read<'pr>(
        &mut self,
        path: &ruby_prism::ConstantPathNode<'pr>,
        outcome: &ConstantPathOutcome,
        leaf_kind: Option<crate::diagnostic::ConstantKind>,
    ) {
        use crate::extract::constant_state;
        self.emit_constant_path_outcome(outcome, leaf_kind);
        let deprecated = self.check_deprecated_constant_path(path, outcome);
        if self.extract.is_none() {
            return;
        }
        let start = path.location().start_offset();
        let end = path.location().end_offset();
        match outcome {
            // Dynamic-parent path (`expr::CONST`): resolution was never
            // attempted, so the whole path is one `untyped` site. The
            // segment walker collected nothing, so the source text
            // itself is the only faithful `path` rendering.
            ConstantPathOutcome::Malformed => {
                let text = String::from_utf8_lossy(&self.source[start..end]).into_owned();
                self.record_extract_constant(start, end, constant_state::UNTYPED, text, None, None);
            }
            ConstantPathOutcome::Walked {
                path_str,
                is_absolute,
                kind,
                ..
            } => {
                let display_path = if *is_absolute {
                    format!("::{path_str}")
                } else {
                    path_str.clone()
                };
                match kind {
                    ConstantPathOutcomeKind::Resolved(ty, Some(leaf)) => {
                        let state = if deprecated {
                            constant_state::ERROR
                        } else {
                            constant_state::TYPED
                        };
                        self.record_extract_constant(
                            start,
                            end,
                            state,
                            display_path,
                            Some(leaf),
                            Some(*ty),
                        );
                    }
                    // Untyped-intermediate short-circuit (`A::B` with
                    // `A: untyped`): the leaf was never looked up, so
                    // this is the honesty column, not `typed`.
                    ConstantPathOutcomeKind::Resolved(_, None) => {
                        self.record_extract_constant(
                            start,
                            end,
                            constant_state::UNTYPED,
                            display_path,
                            None,
                            None,
                        );
                    }
                    ConstantPathOutcomeKind::Miss { .. }
                    | ConstantPathOutcomeKind::IntermediateValue { .. } => {
                        self.record_extract_constant(
                            start,
                            end,
                            constant_state::UNKNOWN_CONSTANT,
                            display_path,
                            None,
                            None,
                        );
                    }
                }
            }
        }
    }

    /// Walk a `ConstantPathNode` against the current lexical scope and
    /// return what was found (`Resolved` / `Miss` / `IntermediateValue`)
    /// without emitting anything. Used by `visit_class_node` and
    /// `visit_module_node` to keep resolution in the *outer* lexical
    /// scope (before `push_class`) while deferring the emit until after
    /// the declaration-name diagnostic — preserving the
    /// "decl-name, then superclass" emit order pinned by
    /// `test_unknown_constant_superclass_undeclared`.
    pub(super) fn resolve_constant_path_outcome<'pr>(
        &self,
        path: &ruby_prism::ConstantPathNode<'pr>,
    ) -> ConstantPathOutcome {
        let mut segments: Vec<(String, usize)> = Vec::new();
        let mut is_absolute = false;
        if !collect_constant_path_segments_with_offsets(path, &mut segments, &mut is_absolute)
            || segments.is_empty()
        {
            return ConstantPathOutcome::Malformed;
        }
        let path_str = segments
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join("::");
        let path_start_byte = path.location().start_offset() as u32;
        // Single-point span (matches the prior `offset_to_location(start)`
        // shape that produced `start_byte == end_byte`).
        let path_loc = (path_start_byte, path_start_byte);

        let context = if is_absolute {
            ConstantContext::toplevel()
        } else {
            constant_context_from_class_stack(self.ctx.class_stack())
        };
        let names = self.env.names();
        let head_sym = names.intern_symbol(&segments[0].0);
        let kind = match self.env.resolve_constant(head_sym, &context) {
            None => ConstantPathOutcomeKind::Miss {
                idx: 0,
                scope: CandidateScope::Context(context.clone()),
            },
            Some(mut current) => {
                let mut kind = ConstantPathOutcomeKind::Resolved(current.ty, Some(current.clone()));
                for (idx, (seg, _)) in segments.iter().enumerate().skip(1) {
                    let target = match &current.origin {
                        ConstantOrigin::Module { target } => *target,
                        ConstantOrigin::Constant if current.ty.is_untyped() => {
                            // Steep parity: a constant declared `FOO:
                            // untyped` used as a namespace prefix
                            // (`FOO::Bar::Baz`) evaluates the whole
                            // remaining path to untyped rather than
                            // reporting the next segment unresolvable —
                            // Steep's constant typing treats `untyped`
                            // as "no further information", not a value
                            // that blocks path continuation.
                            //
                            // `leaf = None` — the synthesised `untyped`
                            // has no backing decl to attach annotations
                            // to, so downstream deprecation checks stay
                            // silent for the whole path.
                            kind = ConstantPathOutcomeKind::Resolved(Ty::UNTYPED, None);
                            break;
                        }
                        ConstantOrigin::Constant => {
                            kind = ConstantPathOutcomeKind::IntermediateValue {
                                idx,
                                scope: CandidateScope::None,
                            };
                            break;
                        }
                    };
                    match self
                        .env
                        .resolve_constant_child(&target, names.intern_symbol(seg))
                    {
                        Some(c) => {
                            current = c;
                            kind = ConstantPathOutcomeKind::Resolved(
                                current.ty,
                                Some(current.clone()),
                            );
                        }
                        None => {
                            kind = ConstantPathOutcomeKind::Miss {
                                idx,
                                scope: CandidateScope::Children(target),
                            };
                            break;
                        }
                    }
                }
                kind
            }
        };
        ConstantPathOutcome::Walked {
            segments,
            path_str,
            is_absolute,
            path_loc,
            kind,
        }
    }

    /// Emit the diagnostic derived from an [`ConstantPathOutcome`]. A
    /// `Miss` and an `IntermediateValue` both name an unresolvable
    /// segment at the same index, so they share one emit path.
    ///
    /// `leaf_kind` is the [`ConstantKind`] to report when the
    /// unresolvable segment is the path's leaf, or `None` to suppress
    /// the leaf emit (declaration contexts where the leaf is the name
    /// being declared, e.g. `class Foo::Bar`). Head and intermediate
    /// segments are always `Constant`; only a superclass leaf differs
    /// (`Class`), matching Steep. Verbose logging fires regardless via
    /// the shared `log_constant_path_outcome`.
    pub(super) fn emit_constant_path_outcome(
        &mut self,
        outcome: &ConstantPathOutcome,
        leaf_kind: Option<crate::diagnostic::ConstantKind>,
    ) {
        use crate::diagnostic::{ConstantKind, Diagnostic, DiagnosticKind};
        self.log_constant_path_outcome(outcome);
        let ConstantPathOutcome::Walked {
            segments,
            path_str,
            is_absolute,
            kind,
            ..
        } = outcome
        else {
            return;
        };
        let (idx, scope) = match kind {
            ConstantPathOutcomeKind::Miss { idx, scope }
            | ConstantPathOutcomeKind::IntermediateValue { idx, scope, .. } => (*idx, scope),
            ConstantPathOutcomeKind::Resolved(..) => return,
        };
        let is_leaf = idx == segments.len() - 1;
        let kind = if is_leaf {
            match leaf_kind {
                Some(k) => k,
                None => return,
            }
        } else {
            ConstantKind::Constant
        };
        let (seg_name, seg_offset) = &segments[idx];
        let position = self.byte_range_to_location(*seg_offset, *seg_offset + seg_name.len());
        let searched_namespaces = self.searched_namespaces_for_scope(scope);
        let path = if *is_absolute {
            format!("::{path_str}")
        } else {
            path_str.clone()
        };
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: position,
            kind: DiagnosticKind::UnknownConstant {
                name: seg_name.clone(),
                path,
                kind,
                did_you_mean: Vec::new(),
                searched_namespaces,
                candidate_scope: scope.clone(),
            },
        });
    }

    /// `--verbose` log for a `ConstantPathOutcome` — silent on the
    /// diagnostic axis. Shared by `resolve_constant_path_node` (type
    /// query) and `emit_constant_path_outcome` (diagnostic path) so the
    /// log text stays in lock-step with the resolver state.
    fn log_constant_path_outcome(&self, outcome: &ConstantPathOutcome) {
        let ConstantPathOutcome::Walked {
            path_str,
            path_loc,
            kind,
            ..
        } = outcome
        else {
            return;
        };
        match kind {
            ConstantPathOutcomeKind::Resolved(ty, _) => {
                self.verbose_log(format_args!(
                    "{}:{} constant {} → {}",
                    path_loc.0,
                    path_loc.1,
                    path_str,
                    self.display_type(*ty),
                ));
            }
            ConstantPathOutcomeKind::IntermediateValue { .. } => {
                self.log_constant_not_found(*path_loc, path_str, " (intermediate is a value)");
            }
            ConstantPathOutcomeKind::Miss { .. } => {
                self.log_constant_not_found(*path_loc, path_str, "");
            }
        }
    }

    /// Silent type-query path: walk `Outer::Inner` and return the
    /// resolved type, or `untyped` on any miss / dynamic-parent /
    /// intermediate-value. Never pushes a diagnostic — emit lives in
    /// `check_constant_path_node`. The walk itself is shared via
    /// `resolve_constant_path_outcome` so the two paths can never
    /// drift apart.
    fn resolve_constant_path_node<'pr>(&self, path: &ruby_prism::ConstantPathNode<'pr>) -> Ty {
        let outcome = self.resolve_constant_path_outcome(path);
        self.log_constant_path_outcome(&outcome);
        outcome.resolved_ty()
    }

    fn log_constant_not_found(&self, span: ArgSpan, path_str: &str, suffix: &str) {
        self.verbose_log(format_args!(
            "{}:{} constant {} → not found{}",
            span.0, span.1, path_str, suffix
        ));
    }

    fn infer_statements_with_hint<'pr>(
        &self,
        stmts: &ruby_prism::StatementsNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        match stmts.body().iter().last() {
            Some(last) => self.infer_type(&last, hint),
            None => Ty::NIL,
        }
    }
}

/// Whether `sub_ol` is a plausible unify candidate for `iface_ol`, for
/// [`TypeChecker::unify_interface_into_bindings`]'s per-overload search.
/// Block presence must match exactly (the caller only unifies block slots
/// when both sides have one); positional arity must match for `Typed`
/// functions, since a mismatched arity can't share the same slot
/// structure. This is a lighter proxy than `check_overload_compatibility`
/// (subtyping.rs) -- unify only needs a plausible candidate to extract
/// bindings from, not a soundness proof.
fn interface_overload_shape_matches(
    a: &crate::types::MethodType,
    b: &crate::types::MethodType,
) -> bool {
    if a.block.is_some() != b.block.is_some() {
        return false;
    }
    match (&a.type_, &b.type_) {
        (FunctionType::Typed(af), FunctionType::Typed(bf)) => {
            af.required_positionals.len() == bf.required_positionals.len()
        }
        (FunctionType::Untyped(_), FunctionType::Untyped(_)) => true,
        _ => false,
    }
}

/// Lift the type-checker's class stack into a [`ConstantContext`]. The
/// stack already holds parsed [`TypeName`]s, so each scope is pushed
/// without re-parsing a path string.
fn constant_context_from_class_stack(stack: &[crate::type_name::TypeName]) -> ConstantContext {
    let mut ctx = ConstantContext::toplevel();
    for tn in stack {
        ctx = ConstantContext::with_scope(&ctx, *tn);
    }
    ctx
}

/// Resolution outcome of a `ConstantPathNode` walk — the data
/// `check_constant_path_inner` needs to emit a diagnostic. Split out so
/// declaration contexts (`visit_class_node` / `visit_module_node`) can
/// resolve in the *outer* lexical scope and emit later, after the
/// declaration-name diagnostic, without rewalking the path.
pub(super) enum ConstantPathOutcome {
    /// `prism` tree was malformed or empty — no resolution attempted.
    Malformed,
    Walked {
        segments: Vec<(String, usize)>,
        path_str: String,
        is_absolute: bool,
        /// Raw byte span of the constant path. Kept as an [`ArgSpan`]
        /// so successful path walks (the hot case — every `Foo::Bar`
        /// reference) skip the char-offset scan that a `SourceLocation`
        /// would cost. Materialized only when a diagnostic actually
        /// fires for the path; see `emit_constant_path_outcome` and
        /// `log_constant_path_outcome`.
        path_loc: ArgSpan,
        kind: ConstantPathOutcomeKind,
    },
}

pub(super) enum ConstantPathOutcomeKind {
    /// Path fully resolved. `ty` is the type the whole path evaluates
    /// to; `leaf` is the leaf `ResolverConstant` when the resolver's
    /// walk produced one directly. `None` when the outcome comes from
    /// the untyped-intermediate short-circuit (`Constant when
    /// current.ty.is_untyped()`), which synthesises `Ty::UNTYPED`
    /// without a backing decl to attach annotations to.
    /// `check_deprecated_constant_path` treats `None` as "no
    /// declaration to inspect" and stays silent.
    Resolved(Ty, Option<crate::definition::ResolverConstant>),
    Miss {
        idx: usize,
        scope: CandidateScope,
    },
    /// A non-final segment resolved to a value constant, so its child
    /// (`idx`) cannot be looked up. Steep reports that child as an
    /// unknown constant, identical to a `Miss` at the same index.
    /// Exception: an `untyped`-typed value constant short-circuits to
    /// `Resolved(Ty::UNTYPED)` instead (see the `resolve` loop in
    /// `resolve_constant_path_outcome`) — it never reaches this variant.
    IntermediateValue {
        idx: usize,
        scope: CandidateScope,
    },
}

impl ConstantPathOutcome {
    /// `Resolved` → that type; anything else (miss / intermediate value
    /// / malformed) → `untyped`. Matches `resolve_constant_path_node`'s
    /// downstream type flow.
    pub(super) fn resolved_ty(&self) -> Ty {
        match self {
            ConstantPathOutcome::Walked {
                kind: ConstantPathOutcomeKind::Resolved(ty, _),
                ..
            } => *ty,
            _ => Ty::UNTYPED,
        }
    }
}

/// Flatten a `ConstantPathNode` into an outer-first list of
/// `(segment_name, byte_offset)` pairs plus an "absolute path" flag.
/// The flag fires when the path's leftmost segment has a `nil` parent
/// (bare `::Foo::Bar`); otherwise the head segment is a
/// `ConstantReadNode` and the path is relative to the lexical scope.
/// Returns `false` for malformed prism trees or for dynamic parents
/// (`expr::CONST`) so the caller can short-circuit / fall back to the
/// default walker. Offsets are paired so the side-effecting walker can
/// attribute a diagnostic to the exact unresolved segment.
fn collect_constant_path_segments_with_offsets<'pr>(
    path: &ruby_prism::ConstantPathNode<'pr>,
    out: &mut Vec<(String, usize)>,
    is_absolute: &mut bool,
) -> bool {
    let Some(name) = path.name() else {
        return false;
    };
    let child_name = String::from_utf8_lossy(name.as_slice()).to_string();
    let child_offset = path.name_loc().start_offset();

    match path.parent() {
        Some(parent) => {
            if let Some(constant) = parent.as_constant_read_node() {
                let parent_name = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                let parent_offset = constant.location().start_offset();
                out.push((parent_name, parent_offset));
                out.push((child_name, child_offset));
                true
            } else if let Some(inner_path) = parent.as_constant_path_node() {
                if !collect_constant_path_segments_with_offsets(&inner_path, out, is_absolute) {
                    return false;
                }
                out.push((child_name, child_offset));
                true
            } else {
                false
            }
        }
        None => {
            *is_absolute = true;
            out.push((child_name, child_offset));
            true
        }
    }
}

/// Converts a prism `Integer` node's `to_u32_digits()` output (magnitude as
/// little-endian base-2^32 words, plus sign) into the canonical decimal
/// string used as `Literal::Integer`/`RecordKey::Integer`'s payload.
/// Arbitrary precision — unlike an `i64` accumulator, the magnitude has no
/// upper bound, matching Ruby's own Integer semantics.
fn prism_digits_to_decimal_string(negative: bool, digits: &[u32]) -> String {
    fn is_zero(digits: &[u32]) -> bool {
        digits.iter().all(|&d| d == 0)
    }
    let mut digits = digits.to_vec();
    if is_zero(&digits) {
        return "0".to_string();
    }
    // Base conversion via repeated long division: divide the base-2^32
    // bignum by 10^9, taking the remainder as the next (base-10^9) decimal
    // chunk, least significant first, until the quotient is zero.
    let mut chunks: Vec<u32> = Vec::new();
    while !is_zero(&digits) {
        let mut remainder: u64 = 0;
        for d in digits.iter_mut().rev() {
            let cur = (remainder << 32) | (*d as u64);
            *d = (cur / 1_000_000_000) as u32;
            remainder = cur % 1_000_000_000;
        }
        chunks.push(remainder as u32);
    }
    let mut result = String::new();
    for (idx, chunk) in chunks.iter().enumerate().rev() {
        if idx == chunks.len() - 1 {
            result.push_str(&chunk.to_string());
        } else {
            result.push_str(&format!("{:09}", chunk));
        }
    }
    if negative {
        format!("-{result}")
    } else {
        result
    }
}

/// Extract a `RecordKey` from a hash-literal key node that is itself a
/// literal (`sym`, `str`, `int`, `true`, `false`). Returns `None` for
/// non-literal keys — the caller uses that to fall through to `Hash`
/// synthesis.
fn record_key_from_literal_node<'pr>(node: &Node<'pr>) -> Option<RecordKey> {
    if let Some(sym) = node.as_symbol_node() {
        let s = String::from_utf8_lossy(sym.unescaped()).to_string();
        return Some(RecordKey::Symbol(s));
    }
    if let Some(str_node) = node.as_string_node() {
        let s = String::from_utf8_lossy(str_node.unescaped()).to_string();
        return Some(RecordKey::String(s));
    }
    if let Some(int_node) = node.as_integer_node() {
        let integer = int_node.value();
        let (negative, digits) = integer.to_u32_digits();
        return Some(RecordKey::Integer(prism_digits_to_decimal_string(
            negative, digits,
        )));
    }
    if node.as_true_node().is_some() {
        return Some(RecordKey::Bool(true));
    }
    if node.as_false_node().is_some() {
        return Some(RecordKey::Bool(false));
    }
    None
}

/// Resolve `@foo` at the current `self` to its declared type, or
/// `None` when no declaration exists on the linearized ancestor chain.
///
/// Instance-side `self` (in instance methods, top-level under a class
/// body) reads from `build_instance(name).instance_variables`. The
/// receiver's type args are threaded through the ancestor walk so
/// `include M[String]`'s `@value: X` resolves to `::String`.
///
/// Singleton-side `self` (class-method bodies; class instance variables /
/// `self.@foo: T`) walks the singleton-side variable lookup path, starting
/// from `build_singleton(name).instance_variables`.
///
/// Read-side callers fold `None` into `Ty::UNTYPED` to keep the silent
/// fallback (see `Node::InstanceVariableReadNode` arm). Write-side
/// callers distinguish `None` to emit `UnknownInstanceVariable`.
pub(super) fn resolve_ivar_at_self(checker: &TypeChecker<'_>, var_sym: Symbol) -> Option<Ty> {
    let self_ty = checker.current_self_type();
    let found = match checker.env.types().resolve(self_ty) {
        Type::ClassInstance { name, args } => checker
            .env
            .lookup_instance_variable_with_args(name, args, var_sym),
        Type::ClassSingleton { name } => checker
            .env
            .lookup_class_instance_variable_with_args(name, var_sym),
        _ => None,
    };
    found.map(|(variable, bindings)| {
        let subst = Substitution::from_mapping(bindings);
        subst.apply(variable.ty, checker.env.types())
    })
}

/// Resolve `@@foo` at the current `self` to its declared type, or
/// `None` when no declaration exists. Both instance-side and
/// singleton-side `self` look up the same `class_variables` map on the
/// instance-side Definition (rbs's `build_singleton0` copies them from
/// there; crema keeps them only on the instance side and lets the walk
/// pick them up regardless of which face of `self` we're on).
///
/// Read-side callers fold `None` into `Ty::UNTYPED`; write-side callers
/// distinguish `None` to emit `UnknownClassVariable`.
pub(super) fn resolve_cvar_at_self(checker: &TypeChecker<'_>, var_sym: Symbol) -> Option<Ty> {
    let self_ty = checker.current_self_type();
    let (name, args) = match checker.env.types().resolve(self_ty) {
        Type::ClassInstance { name, args } => (*name, args.to_vec()),
        Type::ClassSingleton { name } => (*name, vec![]),
        _ => return None,
    };
    checker
        .env
        .lookup_class_variable_with_args(&name, &args, var_sym)
        .map(|(variable, bindings)| {
            let subst = Substitution::from_mapping(bindings);
            subst.apply(variable.ty, checker.env.types())
        })
}

/// Statement-position trailing `#: T` eligibility for the
/// `check_statements_with_hint` gate. Skip nodes whose own pipelines
/// already evaluate the trailing annotation: assignment forms (gated
/// in `visit_local_variable_write_node` / `visit_multi_write_node` and
/// the constant / ivar / cvar / gvar declaration paths), `DefNode`
/// headers (declarations, not assertion sites; Steep skips `:def` /
/// `:defs` explicitly, `source.rb:512`), divergent control-flow leaves
/// (`return EXPR #: T` is owned by `check_explicit_return`; `break` /
/// `next` carry no value through to a statement-position assertion
/// either), and `ParenthesesNode` — its inner StatementsNode propagates
/// the gate to whatever expression is inside via the recursive
/// `check_statements_with_hint` walk, so firing both inner and outer
/// would double-emit on single-line `(expr) #: T`. The top-level walker
/// (`visit_program_node`) handles its own Parens unwrapping since the
/// Visit-trait descent doesn't reach `check_statements_with_hint`.
/// `ClassNode` / `ModuleNode` / `AliasMethodNode` are *not* on this
/// list: Steep has no explicit skip for `:class` / `:module` / `:alias`
/// (`source.rb:510-516`), so `class C; end #: T` / `module M; end #: T`
/// / `alias old new #: T` all reach the assertion evaluator. All three
/// evaluate to `nil` in crema (`check_node` / `infer_type` NIL arms),
/// so the gate distinguishes a real nil match from an untyped-widened
/// stealth pass. `AliasGlobalVariableNode` is *not* on this list either:
/// top-level `alias $new $old #: T` evaluates to
/// `nil` and must reach the gate (Steep `type_construction.rb:2608-2609`
/// `:alias` arm → `AST::Builtin.nil_type`); the `infer_type` arm pairs
/// with it so the gate sees a non-UNTYPED natural and can surface
/// `FalseAssertion` instead of stealth-passing.
/// `SingletonClassNode` is also *not* on this list: `class << self; end #: T`
/// evaluates to `nil` (Steep `type_construction.rb:1614, 1629` `:sclass`
/// arm returns `AST::Builtin.nil_type` on both the unsupported-shape and
/// the body-walked branches); the `infer_type` arm pairs with it so the
/// gate distinguishes a real nil match from an untyped-widened stealth
/// pass.
/// Receiverless `attr_reader` / `attr_writer` / `attr_accessor` /
/// `include` / `extend` / `prepend` are inline declaration sites — the
/// inline_parser consumes their trailing `#: T` at load phase to
/// register a declared method or mixin. The statement-position
/// assertion gate must not treat the CallNode's return type as a
/// natural for `FalseAssertion`, since Steep parses these as
/// declaration nodes rather than calls.
fn is_inline_declaration_call_stmt<'pr>(stmt: &Node<'pr>) -> bool {
    let Some(call) = stmt.as_call_node() else {
        return false;
    };
    if call.receiver().is_some() {
        return false;
    }
    matches!(
        call.name().as_slice(),
        b"attr_reader" | b"attr_writer" | b"attr_accessor" | b"include" | b"extend" | b"prepend"
    )
}

pub(super) fn is_statement_assertion_eligible<'pr>(stmt: &Node<'pr>) -> bool {
    !matches!(
        stmt,
        Node::LocalVariableWriteNode { .. }
            | Node::LocalVariableOrWriteNode { .. }
            | Node::LocalVariableAndWriteNode { .. }
            | Node::LocalVariableOperatorWriteNode { .. }
            | Node::MultiWriteNode { .. }
            | Node::InstanceVariableWriteNode { .. }
            | Node::InstanceVariableOrWriteNode { .. }
            | Node::InstanceVariableAndWriteNode { .. }
            | Node::InstanceVariableOperatorWriteNode { .. }
            | Node::ClassVariableWriteNode { .. }
            | Node::ClassVariableOrWriteNode { .. }
            | Node::ClassVariableAndWriteNode { .. }
            | Node::ClassVariableOperatorWriteNode { .. }
            | Node::GlobalVariableWriteNode { .. }
            | Node::GlobalVariableOrWriteNode { .. }
            | Node::GlobalVariableAndWriteNode { .. }
            | Node::GlobalVariableOperatorWriteNode { .. }
            | Node::ConstantWriteNode { .. }
            | Node::ConstantOrWriteNode { .. }
            | Node::ConstantAndWriteNode { .. }
            | Node::ConstantOperatorWriteNode { .. }
            | Node::ConstantPathWriteNode { .. }
            | Node::ConstantPathOrWriteNode { .. }
            | Node::ConstantPathAndWriteNode { .. }
            | Node::ConstantPathOperatorWriteNode { .. }
            | Node::IndexOrWriteNode { .. }
            | Node::IndexAndWriteNode { .. }
            | Node::IndexOperatorWriteNode { .. }
            | Node::CallOrWriteNode { .. }
            | Node::CallAndWriteNode { .. }
            | Node::CallOperatorWriteNode { .. }
            | Node::DefNode { .. }
            | Node::ReturnNode { .. }
            | Node::BreakNode { .. }
            | Node::NextNode { .. }
            | Node::ParenthesesNode { .. }
    )
}
