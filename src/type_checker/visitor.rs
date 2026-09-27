use ruby_prism::{
    AndNode, CallAndWriteNode, CallNode, CallOperatorWriteNode, CallOrWriteNode, CaseMatchNode,
    CaseNode, ClassNode, ClassVariableAndWriteNode, ClassVariableOperatorWriteNode,
    ClassVariableOrWriteNode, ClassVariableWriteNode, ConstantAndWriteNode,
    ConstantOperatorWriteNode, ConstantOrWriteNode, ConstantPathAndWriteNode, ConstantPathNode,
    ConstantPathOperatorWriteNode, ConstantPathOrWriteNode, ConstantPathWriteNode,
    ConstantReadNode, ConstantWriteNode, DefNode, DefinedNode, ForwardingSuperNode,
    GlobalVariableAndWriteNode, GlobalVariableOperatorWriteNode, GlobalVariableOrWriteNode,
    GlobalVariableReadNode, GlobalVariableWriteNode, IfNode, IndexAndWriteNode,
    IndexOperatorWriteNode, IndexOrWriteNode, InstanceVariableAndWriteNode,
    InstanceVariableOperatorWriteNode, InstanceVariableOrWriteNode, InstanceVariableWriteNode,
    LocalVariableAndWriteNode, LocalVariableOperatorWriteNode, LocalVariableOrWriteNode,
    LocalVariableTargetNode, LocalVariableWriteNode, MatchPredicateNode, MatchRequiredNode,
    MatchWriteNode, ModuleNode, MultiWriteNode, Node, OrNode, ProgramNode, ReturnNode, SuperNode,
    UnlessNode, Visit, YieldNode, visit_constant_path_node, visit_yield_node,
};
use rustc_hash::FxHashMap;

use crate::ast::types as ast_types;
use crate::ast_builder;
use crate::class_new_recognizer::is_class_dot_new;
use crate::data_struct_recognizer::data_struct_construction_kind;
use crate::definition::ConstantOrigin;
use crate::definition_builder;
use crate::diagnostic::{CandidateScope, ConstantKind, Diagnostic, DiagnosticKind};
use crate::environment::DeclKindLocal;
use crate::inline_parser::TrailingAnnotation;
use crate::location::SourceLocation;
use crate::name::Name;
use crate::type_name::{Kind, TypeName};
use crate::type_param::TypeParamScope;
use crate::types::{
    FunctionType, Ty, Type, TypeTable, Visibility, partition_falsy, partition_truthy, union_of,
    union_of_many,
};

use super::TypeChecker;
use super::calls::ResolvedCall;
use super::inference::{ConstantPathOutcome, resolve_cvar_at_self, resolve_ivar_at_self};
use super::is_special_lvar_name;

struct AssertionTypeLocator<'a> {
    text: &'a str,
    base_offset: usize,
    cursor: usize,
}

impl<'a> AssertionTypeLocator<'a> {
    fn new(text: &'a str, base_offset: usize) -> Self {
        Self {
            text,
            base_offset,
            cursor: 0,
        }
    }

    fn find_type_application(&mut self, container_name: &str) -> usize {
        let needle = container_name.strip_prefix("::").unwrap_or(container_name);
        if let Some(index) = self.text[self.cursor..].find(needle) {
            let local = self.cursor + index;
            self.cursor = local + needle.len();
            self.base_offset + local
        } else if let Some(index) = self.text.find(needle) {
            self.cursor = index + needle.len();
            self.base_offset + index
        } else {
            self.base_offset
        }
    }
}

/// Descend through `(expr)` / `((expr))` / etc. to the innermost
/// value-position expression. The caller transfers ownership of
/// `current`; on each iteration the function reassigns `current` to
/// the parens body (or the body's last stmt when it's a
/// StatementsNode), mirroring how the parens evaluates to its inner
/// expression's value. Used at top-level statement position where
/// the gate sits in `visit_program_node` (the
/// `check_statements_with_hint`-driven gate doesn't reach inside the
/// inner StatementsNode through Visit-trait descent).
fn unwrap_top_level_parens<'pr>(mut current: Node<'pr>) -> Node<'pr> {
    loop {
        let next = if let Some(parens) = current.as_parentheses_node()
            && let Some(body) = parens.body()
        {
            if let Some(stmts) = body.as_statements_node() {
                if let Some(last) = stmts.body().iter().last() {
                    last
                } else {
                    break;
                }
            } else {
                body
            }
        } else {
            break;
        };
        current = next;
    }
    current
}

impl<'pr, 'env> Visit<'pr> for TypeChecker<'env> {
    fn visit_program_node(&mut self, node: &ProgramNode<'pr>) {
        // Top-level statements: walk via the Visit trait so existing
        // visitor handlers (visit_constant_path_node's Malformed
        // fallback, etc.) run unchanged, then apply the
        // statement-position trailing `#: T` gate per stmt. class /
        // module / def bodies route through `check_node` →
        // `check_statements_with_hint`, which has its own gate; this
        // covers top-level without changing that funnel.
        //
        // `ParenthesesNode` is in the gate's skip list because
        // `check_statements_with_hint` already fires on the inner
        // expression via its recursive descent. At top level there is
        // no recursive descent into the inner StatementsNode through
        // the gate, so unwrap parens here and gate the innermost
        // expression directly.
        //
        // `infer_type` is only safe to call on stmts that are gate-
        // eligible. Declaration / assignment forms are excluded because
        // their value position is not the surface a trailing assertion
        // targets — the side-effecting `check_node` path routes the
        // hint where it belongs (e.g. into the RHS of a write).
        for stmt in node.statements().body().iter() {
            self.visit(&stmt);
            let effective = unwrap_top_level_parens(stmt);
            if super::inference::is_statement_assertion_eligible(&effective) {
                let natural = self.infer_type(&effective, None);
                self.apply_statement_assertion_gate(&effective, natural);
            }
        }
    }

    fn visit_class_node(&mut self, node: &ClassNode<'pr>) {
        // The superclass is resolved in the enclosing lexical scope,
        // before the class itself is opened (Ruby semantics), so the
        // resolution runs ahead of `push_class`. The emit, however, is
        // deferred until after the declaration-name diagnostic so the two
        // come out in source order (decl name at col 6, then superclass) —
        // crema does not position-sort diagnostics. A single-segment
        // superclass (`class Foo < Bar`) that does not resolve is an
        // unknown constant; Steep tags it `.class!`. ConstantPath
        // superclasses (`class Foo < A::B`) belong to the sibling
        // ConstantPath todo.
        let superclass_miss = if let Some(super_node) = node.superclass()
            && let Some(cr) = super_node.as_constant_read_node()
        {
            let super_name = String::from_utf8_lossy(cr.name().as_slice()).to_string();
            let start = super_node.location().start_offset();
            let end = super_node.location().end_offset();
            match self.try_resolve_constant_read(&super_name, &super_node) {
                Some(constant) => {
                    // extract v7: the superclass position records like
                    // a read, with the check's own resolution.
                    self.record_extract_constant(
                        start,
                        end,
                        crate::extract::constant_state::TYPED,
                        super_name,
                        Some(&constant),
                        Some(constant.ty),
                    );
                    None
                }
                None => {
                    self.record_extract_constant(
                        start,
                        end,
                        crate::extract::constant_state::UNKNOWN_CONSTANT,
                        super_name.clone(),
                        None,
                        None,
                    );
                    let position = self.byte_range_to_location(start, end);
                    let candidate_scope = self.candidate_scope_in_current_scope();
                    let searched_namespaces = self.searched_namespaces_in_current_scope();
                    Some((super_name, position, candidate_scope, searched_namespaces))
                }
            }
        } else {
            None
        };

        // ConstantPath forms of superclass (`class Foo < A::B`) and
        // compact declaration name (`class Foo::Bar`) are resolved
        // here, in the *outer* lexical scope (before `push_class`).
        // Emission is deferred until after the declaration-name
        // diagnostic so the order matches the bare-ConstantReadNode
        // path (decl name first, then superclass) — see
        // `test_unknown_constant_constantpath_superclass_emits_head`.
        let super_path = node.superclass().and_then(|s| s.as_constant_path_node());
        let super_path_outcome = super_path
            .as_ref()
            .map(|p| self.resolve_constant_path_outcome(p));
        let decl_path_outcome = node
            .constant_path()
            .as_constant_path_node()
            .map(|p| self.resolve_constant_path_outcome(&p));

        // Walk a dynamic superclass expression — one that is neither a bare
        // ConstantRead nor a ConstantPath — so calls nested inside it (e.g.
        // `class Sub < make_super()`) fire diagnostics. The ConstantRead and
        // ConstantPath forms are already resolved above via `superclass_miss`
        // and `super_path_outcome`; walking them here would double-emit.
        if let Some(super_node) = node.superclass()
            && super_node.as_constant_read_node().is_none()
            && super_node.as_constant_path_node().is_none()
        {
            self.check_node(&super_node, None);
        }

        let leaf = String::from_utf8_lossy(node.name().as_slice());
        if self.push_decl_class(&node.constant_path(), &leaf).is_none() {
            return;
        }

        // The declaration-name lookup drives two diagnostics off the same
        // `declared_kind_by_type_name` result: a missing declaration is an
        // unknown constant (`.class!`), a declaration that says `module` is
        // a class/module mismatch. `declared_kind_by_type_name` is an id
        // lookup; the qualified-path String is only built inside the
        // mismatch arm below (once per actual mismatch, not once per
        // class/module declaration) so this stays cheap on the common path.
        let decl_info = self
            .ctx
            .current_class_typename()
            .copied()
            .map(|tn| (tn, self.env.declared_kind_by_type_name(&tn)));
        if let Some((tn, declared_kind)) = decl_info {
            match declared_kind {
                // Only a bare-constant declaration name (`class Foo`,
                // including proper lexical nesting) is one unknown constant.
                // A compact path name (`class Foo::Bar`) has a
                // ConstantPathNode whose head resolves through the
                // constant-path resolver — the sibling ConstantPath todo's
                // domain — and `push_class` builds the qualified name from
                // the leaf alone, so emitting here would misreport the leaf
                // (`Bar`) instead of the unresolved head (`Foo`). Defer it.
                None if node.constant_path().as_constant_read_node().is_some() => {
                    let name = String::from_utf8_lossy(node.name().as_slice()).to_string();
                    let location = self.byte_range_to_location(
                        node.constant_path().location().start_offset(),
                        node.constant_path().location().end_offset(),
                    );
                    let candidate_scope = self.candidate_scope_in_enclosing_scope();
                    let searched_namespaces = self.searched_namespaces_in_enclosing_scope();
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location,
                        kind: DiagnosticKind::UnknownConstant {
                            name: name.clone(),
                            path: name,
                            kind: ConstantKind::Class,
                            did_you_mean: Vec::new(),
                            searched_namespaces,
                            candidate_scope,
                        },
                    });
                }
                Some(DeclKindLocal::Module) => {
                    let qualified = self.env.names().resolve(tn);
                    let location = self.offset_to_location(node.location().start_offset());
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location,
                        kind: DiagnosticKind::ClassModuleMismatch {
                            name: qualified,
                            ruby_kind: "class".to_string(),
                            rbs_kind: "module".to_string(),
                        },
                    });
                }
                _ => {}
            }
        }

        if let Some((super_name, position, candidate_scope, searched_namespaces)) = superclass_miss
        {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownConstant {
                    name: super_name.clone(),
                    path: super_name,
                    kind: ConstantKind::Class,
                    did_you_mean: Vec::new(),
                    searched_namespaces,
                    candidate_scope,
                },
            });
        }

        if let (Some(path), Some(outcome)) = (super_path.as_ref(), super_path_outcome.as_ref()) {
            self.emit_constant_path_outcome(outcome, Some(ConstantKind::Class));
            // Diagnostics only above (no deprecation check runs here), so
            // the record can never be `error`.
            self.record_extract_constant_path_outcome(path, outcome, false);
        }
        if let Some(outcome) = decl_path_outcome {
            self.emit_constant_path_outcome(&outcome, None);
        }

        // Walk the class body via `check_node` so any expression inside
        // (bare calls, constant writes with side-effecting RHS) flows
        // through the side-effecting evaluator instead of the Visit-
        // trait default walker. Declaration descendants (`def`, nested
        // `class` / `module`, `return`, `yield`, local-variable writes)
        // are dispatched back to their Visit-trait handlers from inside
        // `check_node`'s declarative arms.
        let saved_singleton_class_depth = self.ctx.replace_singleton_class_depth(0);
        if let Some(body) = node.body() {
            self.check_node(&body, None);
        }
        self.ctx
            .replace_singleton_class_depth(saved_singleton_class_depth);
        self.ctx.pop_class();
    }

    fn visit_module_node(&mut self, node: &ModuleNode<'pr>) {
        // See `visit_class_node`: resolve the compact-path module name
        // (`module Foo::Bar`) in the *outer* lexical scope before
        // pushing, and defer the emit until after the declaration-name
        // diagnostic.
        let decl_path_outcome = node
            .constant_path()
            .as_constant_path_node()
            .map(|p| self.resolve_constant_path_outcome(&p));

        let leaf = String::from_utf8_lossy(node.name().as_slice());
        if self.push_decl_class(&node.constant_path(), &leaf).is_none() {
            return;
        }

        // Same declaration-name lookup as `visit_class_node`: a missing
        // declaration is an unknown constant (`.module!`), a declaration
        // that says `class` is a class/module mismatch. See `visit_class_node`
        // for why the qualified-path String is deferred into the mismatch arm.
        let decl_info = self
            .ctx
            .current_class_typename()
            .copied()
            .map(|tn| (tn, self.env.declared_kind_by_type_name(&tn)));
        if let Some((tn, declared_kind)) = decl_info {
            match declared_kind {
                // See `visit_class_node`: a compact path name
                // (`module Foo::Bar`) is deferred to the sibling
                // ConstantPath todo; only a bare-constant name emits here.
                None if node.constant_path().as_constant_read_node().is_some() => {
                    let name = String::from_utf8_lossy(node.name().as_slice()).to_string();
                    let location = self.byte_range_to_location(
                        node.constant_path().location().start_offset(),
                        node.constant_path().location().end_offset(),
                    );
                    let candidate_scope = self.candidate_scope_in_enclosing_scope();
                    let searched_namespaces = self.searched_namespaces_in_enclosing_scope();
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location,
                        kind: DiagnosticKind::UnknownConstant {
                            name: name.clone(),
                            path: name,
                            kind: ConstantKind::Module,
                            did_you_mean: Vec::new(),
                            searched_namespaces,
                            candidate_scope,
                        },
                    });
                }
                Some(DeclKindLocal::Class) => {
                    let qualified = self.env.names().resolve(tn);
                    let location = self.offset_to_location(node.location().start_offset());
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location,
                        kind: DiagnosticKind::ClassModuleMismatch {
                            name: qualified,
                            ruby_kind: "module".to_string(),
                            rbs_kind: "class".to_string(),
                        },
                    });
                }
                _ => {}
            }
        }

        if let Some(outcome) = decl_path_outcome {
            self.emit_constant_path_outcome(&outcome, None);
        }

        // Same shape as `visit_class_node`: walk the module body via
        // `check_node` so expression statements route through the
        // side-effecting evaluator.
        let saved_singleton_class_depth = self.ctx.replace_singleton_class_depth(0);
        if let Some(body) = node.body() {
            self.check_node(&body, None);
        }
        self.ctx
            .replace_singleton_class_depth(saved_singleton_class_depth);
        self.ctx.pop_class();
    }

    fn visit_def_node(&mut self, node: &DefNode<'pr>) {
        self.record_extract_implements(node);
        // Inside a retargeted concern block walk the class stack already
        // names the one target this pass is for — see the field doc.
        let synthetic_targets = if self.concern_block_target_walk {
            Vec::new()
        } else {
            self.synthetic_method_context_targets(node)
        };

        if !synthetic_targets.is_empty() {
            for target in synthetic_targets {
                let saved_class_stack = self.ctx.replace_class_stack(vec![target]);
                self.check_def_node_in_current_context(node);
                self.ctx.restore_class_stack(saved_class_stack);
            }
            return;
        }

        self.check_def_node_in_current_context(node);
    }

    fn visit_singleton_class_node(&mut self, node: &ruby_prism::SingletonClassNode<'pr>) {
        self.check_node(&node.expression(), None);
        if !self.ctx.in_singleton_class()
            && self.ctx.current_class_typename().is_some()
            && node.expression().as_self_node().is_some()
            && let Some(body) = node.body()
        {
            self.ctx.enter_singleton_class();
            self.check_singleton_class_body(&body);
            self.ctx.leave_singleton_class();
        }
    }

    fn visit_return_node(&mut self, node: &ReturnNode<'pr>) {
        // Walk the returned expression before the declared-return check.
        // `check_explicit_return` only *infers* the value (read-only), so
        // without this descent nothing in a return value ever fires its
        // diagnostics — the default walker doesn't descend either once
        // this callback is overridden. `check_node` runs the
        // side-effecting pass exactly once; the subsequent `infer_type`
        // inside `check_explicit_return` is read-only and can't
        // double-emit. The declared return type rides along as the hint
        // (Steep `type_construction.rb:1161` synthesizes the return value
        // with `hint: method_return_type`), so a returned lambda literal
        // binds its params from the declared Proc type instead of UNTYPED.
        if let Some(arguments) = node.arguments()
            && let Some(first) = arguments.arguments().iter().next()
        {
            let hint = self
                .ctx
                .method_type()
                .and_then(|mt| super::methods::return_type_hint(mt.return_type()));
            self.check_node(&first, hint);
        }
        self.check_explicit_return(node);
    }

    fn visit_local_variable_write_node(&mut self, node: &LocalVariableWriteNode<'pr>) {
        // Default-walker glue, mirroring `visit_index_operator_write_node`'s
        // relationship to `check_index_operator_write`: the value-position
        // type is driven by `check_node`'s `LocalVariableWriteNode` arm,
        // which calls `check_local_variable_write` directly with the
        // ambient hint. This callback handles paths reached via the
        // Visit-trait's automatic descent instead, where no ambient hint
        // is available.
        self.check_local_variable_write(node, None);
    }

    fn visit_local_variable_or_write_node(&mut self, node: &LocalVariableOrWriteNode<'pr>) {
        self.rebind_compound_lvar(node.name().as_slice(), node.depth(), &node.value(), None);
    }

    fn visit_local_variable_and_write_node(&mut self, node: &LocalVariableAndWriteNode<'pr>) {
        self.rebind_compound_lvar(node.name().as_slice(), node.depth(), &node.value(), None);
    }

    fn visit_local_variable_operator_write_node(
        &mut self,
        node: &LocalVariableOperatorWriteNode<'pr>,
    ) {
        // Default-walker glue, mirroring `visit_index_operator_write_node`'s
        // relationship to `check_index_operator_write`: the value-position
        // type is driven by `check_node`'s dispatch arm, which calls
        // `check_local_variable_operator_write` directly. This callback
        // handles paths reached via the Visit-trait's automatic descent
        // instead, where the result type is unused.
        self.check_local_variable_operator_write(node);
    }

    fn visit_multi_write_node(&mut self, node: &MultiWriteNode<'pr>) {
        // Default-walker glue, mirroring `visit_index_operator_write_node`'s
        // relationship to `check_index_operator_write`: the value-position
        // type is driven by `check_node`'s `MultiWriteNode` arm, which calls
        // `check_multi_write_node` directly. This callback handles paths
        // reached via the Visit-trait's automatic descent instead (e.g. as
        // a top-level statement), where the result type is unused — no
        // hint reaches this path either way.
        self.check_multi_write_node(node, None);
    }

    fn visit_local_variable_target_node(&mut self, node: &LocalVariableTargetNode<'pr>) {
        // Reached only when the default `Visit` walker descends into a
        // Target node outside `visit_multi_write_node` (e.g. nested
        // inside a `MultiTargetNode`, walked by the scope-outside arm
        // above). Bind to UNTYPED so the lvar is at least registered
        // and a later read doesn't trip the "undeclared lvar → untyped"
        // fallback unconditionally.
        self.bind_local_variable_target(node, Ty::UNTYPED);
    }

    fn visit_if_node(&mut self, node: &IfNode<'pr>) {
        // Default-walker descent into `if` arms would write each arm's
        // assignments straight onto the live scope chain with no
        // snapshot / restore, so the outer-scope writes from one arm
        // leak into the next and past the merge point. Route through
        // `check_if_node` instead so the branch-isolation + join path
        // runs uniformly in every position the visitor reaches —
        // notably block bodies, which are visited via
        // `check_call`'s `self.visit(&block)` rather than `check_node`.
        // Method / class / module bodies don't pass through here: they
        // go through `check_node(&body, None)` already and dispatch to
        // `check_if_node` from `check_node`'s `IfNode` arm directly.
        let node_ref = node.as_node();
        self.check_if_node(&node_ref, None);
    }

    fn visit_unless_node(&mut self, node: &UnlessNode<'pr>) {
        // Same routing-through-check_node story as `visit_if_node`.
        let node_ref = node.as_node();
        self.check_unless_node(&node_ref, None);
    }

    fn visit_case_node(&mut self, node: &CaseNode<'pr>) {
        // Same routing-through-check_node story as `visit_if_node`.
        // case/when bodies in a block would otherwise hit the default
        // walker and leak each `when` arm's writes to the outer scope.
        let node_ref = node.as_node();
        self.check_case_node(&node_ref, None);
    }

    fn visit_case_match_node(&mut self, node: &CaseMatchNode<'pr>) {
        // Same routing-through-check_node story as `visit_case_node`:
        // without this, case/in reached via the Visit walker (block
        // bodies) would bind captures UNTYPED through the default
        // walker instead of `check_pattern`'s derived types.
        let node_ref = node.as_node();
        self.check_case_match_node(&node_ref, None);
    }

    fn visit_match_predicate_node(&mut self, node: &MatchPredicateNode<'pr>) {
        // Route through `check_node`'s MatchPredicateNode arm so the
        // Visit-walker path (statement position, block bodies) binds
        // captures with the same derived types as the check_node path
        // — otherwise the default walker's descent would bind them
        // UNTYPED via `visit_local_variable_target_node`.
        let node_ref = node.as_node();
        self.check_node(&node_ref, None);
    }

    fn visit_match_required_node(&mut self, node: &MatchRequiredNode<'pr>) {
        // Same routing story as `visit_match_predicate_node`.
        let node_ref = node.as_node();
        self.check_node(&node_ref, None);
    }

    fn visit_match_write_node(&mut self, node: &MatchWriteNode<'pr>) {
        // Same routing story as `visit_match_required_node`: without
        // this, the default walker's descent would fire
        // `visit_call_node` on the inner `=~` a second time and skip
        // the capture-local binding.
        let node_ref = node.as_node();
        self.check_node(&node_ref, None);
    }

    fn visit_or_node(&mut self, node: &OrNode<'pr>) {
        let node_ref = node.as_node();
        self.check_or_node(&node_ref, None);
    }

    fn visit_and_node(&mut self, node: &AndNode<'pr>) {
        let node_ref = node.as_node();
        self.check_and_node(&node_ref, None);
    }

    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        // Default-walker glue. Hint-bearing flow is driven by
        // `check_node`'s CallNode arm, where the hint is passed as an
        // explicit argument. This callback handles the remaining paths
        // where a call is reached via the Visit-trait's automatic
        // descent (yield args, block bodies via `check_call`'s
        // `self.visit(&block)`, write-node RHSs whose visitor we
        // haven't taken over). The hint is `None` here by design:
        // those paths have no enclosing assertion to forward.
        let resolved = ResolvedCall::resolve(self, node);
        self.check_call(node, None, &resolved);
        // Extract record. `return_type` is the value the enclosing check
        // frame already computed for this site and handed down through
        // `extract_carried_types` (receiver / argument / block-tail
        // positions), or None when no frame consumed the call's value
        // (statement position, span mismatch). Extract reports exactly
        // what the check computed — it does not run its own inference
        // (ADR-lite: `consulted` is "what the check consulted").
        let carried = self.extract_take_carried_type((
            node.location().start_offset() as u32,
            node.location().end_offset() as u32,
        ));
        if let Some(target) = resolved.target.as_ref() {
            self.record_extract_call(
                node.location().start_offset(),
                node.location().end_offset(),
                target,
                resolved.receiver_ty,
                carried,
            );
        } else {
            self.record_extract_call_no_target(node, resolved.receiver_ty, carried);
        }
    }

    fn visit_lambda_node(&mut self, node: &ruby_prism::LambdaNode<'pr>) {
        // Default-walker glue. The body walk (param binding + scope
        // push/pop) lives in `check_node`'s LambdaNode arm, where the
        // hint is an explicit argument. Routing the descent through it
        // — instead of letting the default walker descend into the body
        // — keeps the walk single-path: statement / argument positions
        // reach the lambda here, value positions reach `check_node`
        // directly, and neither double-emits. The hint is `None` here by
        // design; hinted argument positions are pre-dispatched by
        // `check_call`'s `visit_call_arguments`.
        self.check_node(&node.as_node(), None);
    }

    fn visit_super_node(&mut self, node: &SuperNode<'pr>) {
        // Default-walker glue for `super(args)` reached via the Visit-trait
        // descent — notably as a call receiver (`super(x).foo`), where
        // `check_call`'s `self.visit(&receiver)` lands here. Route through
        // `check_node`'s SuperNode arm so the argument diagnostics fire
        // exactly once. Statement / value positions don't pass through here:
        // they reach `check_node` directly via the method-body walk, so the
        // two paths are disjoint and a single `super(args)` emits once.
        self.check_node(&node.as_node(), None);
    }

    fn visit_forwarding_super_node(&mut self, node: &ForwardingSuperNode<'pr>) {
        // Mirror of `visit_super_node` for bare `super`. Routes to
        // `check_node`'s ForwardingSuperNode arm so `Ruby::UnexpectedSuper`
        // fires exactly once when `super.foo` reaches here as a CallNode
        // receiver. Statement / value positions land on the arm directly via
        // the method-body walk, so the two paths stay disjoint.
        self.check_node(&node.as_node(), None);
    }

    fn visit_defined_node(&mut self, _node: &DefinedNode<'pr>) {
        // `defined?(x)` does not evaluate its operand at runtime — the
        // result is the kind string or nil, never an error from `x`. The
        // default walker would descend into the operand and type-check it
        // (firing spurious NoMethod / UnknownConstant), so we override it
        // to a no-op. The result type (`String?`) comes from `infer_type`'s
        // DefinedNode arm. Mirrors Steep's `type_any_rec(only_children:)`,
        // which types the operand as `any` without synthesizing it.
    }

    fn visit_constant_read_node(&mut self, node: &ConstantReadNode<'pr>) {
        // Constant reads reached via the Visit-trait default walker: a
        // top-level statement (`crema check -e 'Foo'`) or a call receiver
        // (`check_call`'s `self.visit(&receiver)`). Body / write-RHS reads
        // route through `check_node`'s ConstantReadNode arm instead, so
        // the two paths are disjoint and a single occurrence emits once.
        self.check_constant_read(node);
    }

    fn visit_constant_path_node(&mut self, node: &ConstantPathNode<'pr>) {
        // Symmetric to `visit_constant_read_node`. Static paths (whose
        // parent is `nil`, ConstantReadNode, or ConstantPathNode) are
        // emitted through `check_constant_path_node`; we deliberately
        // skip the default walker to avoid double-emitting the head
        // ConstantReadNode at the same source location.
        //
        // Dynamic-parent paths (`Foo.bar::Const`, `obj::C`) are handled
        // inside `check_constant_path_outcome`: it type-checks the parent
        // expression once (firing its own diagnostics, e.g. `Foo.bar`'s
        // NoMethod) and resolves the leaf against the parent's type.
        // Only a truly malformed prism tree stays `Malformed`, and that
        // has nothing left to walk.
        let outcome = self.check_constant_path_outcome(node);
        self.finish_constant_path_read(node, &outcome, Some(ConstantKind::Constant));
    }

    fn visit_global_variable_read_node(&mut self, node: &GlobalVariableReadNode<'pr>) {
        // Fires `Ruby::DeprecatedReference` when a top-level or
        // Visit-trait-reached bare gvar read targets a declared gvar
        // marked `%a{deprecated}`. `check_node`'s
        // `GlobalVariableReadNode` arm covers the same ground for
        // gvars reached inside a method / block body (where the type
        // checker calls `check_node` on statements). The two paths are
        // disjoint so a single occurrence emits once.
        let name_str = String::from_utf8_lossy(node.name().as_slice()).into_owned();
        self.check_deprecated_global_ref(
            &name_str,
            node.location().start_offset(),
            node.location().end_offset(),
        );
    }

    fn visit_instance_variable_write_node(&mut self, node: &InstanceVariableWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_write_node`.
        // `check_node`'s `InstanceVariableWriteNode` arm calls
        // `check_instance_variable_write` directly for the value-position
        // dispatch; this callback covers the remaining Visit-trait
        // descent paths, where the result is unused.
        self.check_instance_variable_write(node);
    }

    fn visit_instance_variable_or_write_node(&mut self, node: &InstanceVariableOrWriteNode<'pr>) {
        // Steep `or_asgn`/`and_asgn` with an `ivasgn` LHS dispatches to
        // the same `constr.ivasgn(asgn, type)` as plain `@x = rhs`
        // (`type_construction.rb:2395-2428`). Declared ivar type is
        // immutable here — narrowing is a `LocalVariableOrWriteNode`-only
        // concept (lvars have no RBS declaration to anchor to).
        self.check_compound_ivar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            None,
            node.location().start_offset(),
        );
    }

    fn visit_instance_variable_and_write_node(&mut self, node: &InstanceVariableAndWriteNode<'pr>) {
        // See `visit_instance_variable_or_write_node`; `&&=` shares the
        // same `or_asgn`/`and_asgn` dispatch in Steep.
        self.check_compound_ivar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            None,
            node.location().start_offset(),
        );
    }

    fn visit_instance_variable_operator_write_node(
        &mut self,
        node: &InstanceVariableOperatorWriteNode<'pr>,
    ) {
        // Default-walker glue — see `visit_local_variable_operator_write_node`.
        self.check_instance_variable_operator_write(node);
    }

    fn visit_index_or_write_node(&mut self, node: &IndexOrWriteNode<'pr>) {
        // Default-walker glue, mirroring `visit_index_operator_write_node`'s
        // relationship to `check_index_operator_write`: the value-position
        // type is driven by `check_node`'s `IndexOrWriteNode` arm, which
        // calls `check_index_or_write` directly. This callback handles
        // paths reached via the Visit-trait's automatic descent instead,
        // where the result type is unused.
        self.check_index_or_write(node);
    }

    fn visit_index_and_write_node(&mut self, node: &IndexAndWriteNode<'pr>) {
        // Default-walker glue — see `visit_index_or_write_node`.
        self.check_index_and_write(node);
    }

    fn visit_index_operator_write_node(&mut self, node: &IndexOperatorWriteNode<'pr>) {
        // Default-walker glue, mirroring `visit_call_node`'s relationship
        // to `check_call`: the value-position type is driven by
        // `check_node`'s `IndexOperatorWriteNode` arm, which calls
        // `check_index_operator_write` directly. This callback handles
        // paths reached via the Visit-trait's automatic descent instead
        // (e.g. as a top-level statement), where the result type is
        // unused.
        self.check_index_operator_write(node);
    }

    fn visit_call_or_write_node(&mut self, node: &CallOrWriteNode<'pr>) {
        // Default-walker glue — see `visit_index_or_write_node`.
        self.check_call_or_write(node);
    }

    fn visit_call_and_write_node(&mut self, node: &CallAndWriteNode<'pr>) {
        // Default-walker glue — see `visit_index_or_write_node`.
        self.check_call_and_write(node);
    }

    fn visit_call_operator_write_node(&mut self, node: &CallOperatorWriteNode<'pr>) {
        // Default-walker glue — see `visit_index_operator_write_node`.
        self.check_call_operator_write(node);
    }

    fn visit_class_variable_write_node(&mut self, node: &ClassVariableWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_write_node`.
        self.check_class_variable_write(node);
    }

    fn visit_class_variable_or_write_node(&mut self, node: &ClassVariableOrWriteNode<'pr>) {
        // Steep `or_asgn`/`and_asgn` with a `cvasgn` LHS dispatches to
        // the same `constr.cvasgn(asgn, type)` as plain `@@x = rhs`
        // (`type_construction.rb:2395-2428`). Declared cvar type is
        // immutable here, mirroring the ivar paradigm.
        self.check_compound_cvar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            node.location().start_offset(),
        );
    }

    fn visit_class_variable_and_write_node(&mut self, node: &ClassVariableAndWriteNode<'pr>) {
        // See `visit_class_variable_or_write_node`; `&&=` shares the
        // same `or_asgn`/`and_asgn` dispatch in Steep.
        self.check_compound_cvar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            node.location().start_offset(),
        );
    }

    fn visit_class_variable_operator_write_node(
        &mut self,
        node: &ClassVariableOperatorWriteNode<'pr>,
    ) {
        // Default-walker glue — see `visit_local_variable_operator_write_node`.
        self.check_class_variable_operator_write(node);
    }

    fn visit_global_variable_write_node(&mut self, node: &GlobalVariableWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_write_node`.
        self.check_global_variable_write(node);
    }

    fn visit_global_variable_or_write_node(&mut self, node: &GlobalVariableOrWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_operator_write_node`.
        self.check_global_variable_or_write(node, None);
    }

    fn visit_global_variable_and_write_node(&mut self, node: &GlobalVariableAndWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_operator_write_node`.
        self.check_global_variable_and_write(node, None);
    }

    fn visit_global_variable_operator_write_node(
        &mut self,
        node: &GlobalVariableOperatorWriteNode<'pr>,
    ) {
        // Default-walker glue — see `visit_local_variable_operator_write_node`.
        self.check_global_variable_operator_write(node);
    }

    fn visit_constant_write_node(&mut self, node: &ConstantWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_write_node`.
        self.check_constant_write(node);
    }

    fn visit_constant_path_write_node(&mut self, node: &ConstantPathWriteNode<'pr>) {
        // Default-walker glue — see `visit_local_variable_write_node`.
        // Unlike its Or/And/Operator siblings below (deliberate no-ops:
        // their value-position read is a silent oracle peek, and the
        // side-effecting existence check for those is a separate todo —
        // see the comment above `visit_constant_or_write_node`), the
        // plain write had no override at all before this migration;
        // `check_constant_path_write` is this kind's first
        // side-effecting implementation.
        self.check_constant_path_write(node);
    }

    // Constant / ConstantPath compound writes — silent LHS-descent guard.
    // The value-position `infer_type` arms (inference.rs) intentionally
    // use the silent constant resolvers (`try_resolve_constant_read` /
    // `resolve_constant_path_outcome`) so a receiver-position occurrence
    // like `(N::Undeclared ||= 1).inspect` doesn't push `UnknownConstant`.
    // But `check_call` walks the receiver via `self.visit(&receiver)`,
    // which means without an override on these six nodes the default
    // walker descends into the LHS `ConstantPathNode` / `ConstantReadNode`
    // and emits `UnknownConstant` through `visit_constant_path_node` /
    // `visit_constant_read_node`. All six override the LHS descent, but
    // they now split into two groups on the RHS: the Or/And four below
    // stay complete no-ops (RHS untouched too — separate todo, Or/And
    // is out of this migration's scope), while the Operator two further
    // down (`visit_constant_operator_write_node` /
    // `visit_constant_path_operator_write_node`) additionally evaluate
    // the RHS for its own side effects (`write_node_value_type_operator_
    // write_constant`, F3 follow-up) — see their own doc comment for why.
    //
    // The Or/And siblings' statement-position side-effect (declared-type
    // subtype gate, `UnknownConstant` emit on `Foo ||= 1` standalone)
    // remains a separate todo.
    fn visit_constant_or_write_node(&mut self, _node: &ConstantOrWriteNode<'pr>) {}

    fn visit_constant_and_write_node(&mut self, _node: &ConstantAndWriteNode<'pr>) {}

    // `Foo += v`: Steep's `:op_asgn` handler has no `:casgn` branch — it
    // falls to the generic `else` (`type_construction.rb:907-911`),
    // which synthesizes the rhs for its side effects (so e.g. a
    // NoMethod inside the rhs still surfaces) but assigns the whole
    // expression `any` (untyped) with no diagnostic of its own — Steep
    // doesn't implement a dedicated constant-operator-write dispatch at
    // all (confirmed: `Foo += 1` parses to an `:op_asgn`/`:casgn` node,
    // `Steep.logger.error`-only fallback, verified via Ruby's `parser`
    // gem and reading `type_construction.rb` directly). So unlike the
    // lvar/ivar/cvar/gvar F3 siblings (real operator dispatch, value =
    // operator's return type), this kind stays at "evaluate the rhs,
    // discard its type" — `check_node`'s discard-cluster arm
    // (`inference.rs`) still returns `Ty::UNTYPED` for it, matching
    // Steep's `any`.
    fn visit_constant_operator_write_node(&mut self, node: &ConstantOperatorWriteNode<'pr>) {
        self.check_node(&node.value(), None);
    }

    fn visit_constant_path_or_write_node(&mut self, _node: &ConstantPathOrWriteNode<'pr>) {}

    fn visit_constant_path_and_write_node(&mut self, _node: &ConstantPathAndWriteNode<'pr>) {}

    // See `visit_constant_operator_write_node`.
    fn visit_constant_path_operator_write_node(
        &mut self,
        node: &ConstantPathOperatorWriteNode<'pr>,
    ) {
        self.check_node(&node.value(), None);
    }

    fn visit_yield_node(&mut self, node: &YieldNode<'pr>) {
        let info = self.ctx.method_type().map(|mt| {
            let block_is_none = mt.block.is_none();
            // Required + optional slots (Steep `flat_unnamed_params`): a
            // `yield 1, "s"` into `(Integer, ?Integer)` must check the
            // second argument against the optional slot.
            let req_positionals: Vec<Ty> = mt
                .block
                .as_ref()
                .map(|b| b.flat_positionals())
                .unwrap_or_default();
            (block_is_none, req_positionals)
        });

        if let Some((block_is_none, req_positionals)) = info {
            if block_is_none {
                let position = self.offset_to_location(node.location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position.clone(),
                    kind: DiagnosticKind::FallbackAny {
                        reason: "yield outside of a block-declared method".into(),
                    },
                });
                let method_name = self.ctx.method_name().unwrap_or("<unknown>").to_string();
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::UnexpectedYield { method_name },
                });
            } else if let Some(arguments) = node.arguments() {
                let checker = self.subtyper();
                let defined_in = self
                    .ctx
                    .defined_in()
                    .map(|tn| self.env.names().display_type_name(tn));
                for (index, (arg, &expected)) in arguments
                    .arguments()
                    .iter()
                    .zip(req_positionals.iter())
                    .enumerate()
                {
                    let actual = self.infer_type(&arg, Some(expected));
                    if !checker.check(actual, expected) {
                        let position = self.offset_to_location(arg.location().start_offset());
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: position,
                            kind: DiagnosticKind::ArgumentTypeMismatch {
                                method_name: "yield".to_string(),
                                param_index: Some(index),
                                keyword: None,
                                expected: self.display_type(expected),
                                actual: self.display_type(actual),
                                defined_in: defined_in.clone(),
                            },
                        });
                    }
                }
            }
        }

        visit_yield_node(self, node);
    }
}

/// A write-node's own expression value never legitimately diverges
/// control flow (the assignment always completes) — only the RHS's
/// *type* can degenerate to `Ty::BOTTOM` (e.g. `@x = a_method_declared
/// _to_never_return`). `check_statements_with_hint`'s divergence
/// cutoff (`if last_ty == Ty::BOTTOM { break }`) treats a `Ty::BOTTOM`
/// statement value as "unreachable code follows" (return / raise /
/// diverging if), so surfacing a real `Ty::BOTTOM` from a write kind's
/// `check_node` dispatch arm would silently skip live, still-reachable
/// statements after it. Caps to `Ty::UNTYPED` here.
/// `check_local_variable_write` has its own inline variant of this
/// same cap instead of calling this, because it also needs to bind the
/// lvar to the real (uncapped) `Ty::BOTTOM` in the env first (`y =
/// self.boom` — `y.foo` must still see `y: bot` and report NoMethod).
fn cap_write_value_bottom(ty: Ty) -> Ty {
    if ty == Ty::BOTTOM { Ty::UNTYPED } else { ty }
}

impl<'env> TypeChecker<'env> {
    fn freeze_lvar_bound_type(&self, ty: Ty, depth: u32) -> Ty {
        if depth == 0 {
            return ty;
        }
        self.freeze_self_type_for_lvar_binding(ty)
    }

    /// Enter a closure (block / lambda / proc body): push the `Block`
    /// scope and pin every visible outer lvar to its current type.
    /// Steep's `type_env.pin_local_variables(nil)` at the block-call
    /// and lambda sites (`type_construction.rb`).
    ///
    /// The pin is the visible type itself: literal expressions are
    /// already class-typed at synthesis unless a hint asked for the
    /// literal (`infer_type`, Steep `test_literal_type`), so `r = 1`
    /// binds `::Integer` and a pinned `r = 2` inside the closure is a
    /// same-type write. `true` / `false` are the exception (their
    /// synthesis is not widened) and pin as `bool`, so `x = false; each
    /// { x = true }` is not flagged.
    ///
    /// The pin is additionally `self`-frozen the way
    /// `freeze_lvar_bound_type` freezes every closure-crossing write:
    /// `spy = self` binds the opaque `self`, the write inside the block
    /// arrives as the concrete class, and the subtype check must compare
    /// like with like. The freeze is not written back to the outer
    /// binding — outside the closure `self` identity stays tracked.
    pub(super) fn push_block_scope(&mut self) {
        let visible = self.ctx.visible_local_variables();
        let mut pins: FxHashMap<Name, Ty> = FxHashMap::default();
        for (name, ty) in visible {
            let pin = match self.env.types().resolve(ty) {
                Type::Literal(crate::types::Literal::Bool(_)) => Ty::BOOL,
                _ => ty,
            };
            pins.insert(name, self.freeze_self_type_for_lvar_binding(pin));
        }
        self.ctx.push_block_scope_with_pins(pins);
    }

    /// Closure-crossing lvar write (prism `depth >= 1`) against a pinned
    /// outer variable. Returns `Some(pinned)` when a pin applies: the
    /// write is rebound to the pinned type (Steep's
    /// `assign_local_variable`: `enforced_type || var_type`) and an
    /// `IncompatibleAssignment` is reported on the write node's byte
    /// range when `ty` is not a subtype of it (`type_construction.rb`
    /// lvasgn arm). Untyped on either side is silent. `None` means no
    /// pin — the caller binds `ty` as usual.
    ///
    /// The range is taken as bytes and only turned into a
    /// `SourceLocation` on the diagnostic path: `byte_range_to_location`
    /// scans the source for char offsets and clones the file path, and
    /// this helper runs on every lvar write (measured 2026-09-11: doing
    /// that eagerly cost ~8% User time on `crema check lib` in steep).
    fn bind_pinned_lvar_write(
        &mut self,
        name: Name,
        ty: Ty,
        depth: u32,
        byte_range: (usize, usize),
    ) -> Option<Ty> {
        let pinned = self.ctx.pinned_local_variable_type(name, depth)?;
        if !ty.is_untyped() && !pinned.is_untyped() && !self.subtyper().check(ty, pinned) {
            let location = self.byte_range_to_location(byte_range.0, byte_range.1);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location,
                kind: DiagnosticKind::IncompatibleAssignment {
                    lhs_type: self.display_type(pinned),
                    // A mismatching literal RHS is already class-typed
                    // at synthesis (the pin is its hint); `true` /
                    // `false` are not, and print as `bool` like the pin
                    // does.
                    rhs_type: self.display_type(match self.env.types().resolve(ty) {
                        Type::Literal(crate::types::Literal::Bool(_)) => Ty::BOOL,
                        _ => ty,
                    }),
                },
            });
        }
        self.ctx.set_local_variable_at_depth(name, pinned, depth);
        Some(pinned)
    }

    fn check_def_node_in_current_context<'pr>(&mut self, node: &DefNode<'pr>) {
        if let Some(receiver) = node.receiver() {
            self.check_node(&receiver, None);
            if self.ctx.in_singleton_class() {
                return;
            }
        }

        let entered_method = self.enter_method_context(node);
        self.check_method_params(node);

        if let Some(params) = node.parameters() {
            for p in params.optionals().iter() {
                if let Some(opt) = p.as_optional_parameter_node() {
                    self.check_node(&opt.value(), None);
                }
            }
            for p in params.keywords().iter() {
                if let Some(opt) = p.as_optional_keyword_parameter_node() {
                    self.check_node(&opt.value(), None);
                }
            }
        }

        // Body visit must run before `check_return_type` so the latter can
        // reuse the value this walk computes (Steep `type_construction.rb:
        // 971-1075` `:def` is a single-pass design — `check(body_node,
        // return_type) { |_, actual_type, ...| }` uses the type synthesized
        // during the walk itself). Running `check_return_type` first made
        // the prediction read stale bindings (declared param types survive
        // as-is, fresh locals fall back to `Ty::UNTYPED` and silently
        // pass), which surfaced as the param-shadow false positive *and*
        // the fresh-local false negative.
        //
        // The walk is given the method's own return hint and its result is
        // captured (`body_value`) for `check_return_type` to use directly,
        // rather than always re-deriving it via a second `infer_type` pass:
        // a rescue-bearing body's live scope is already widened back to
        // its post-join state by the time a second pass would run, so
        // re-reading a local variable there would pick up the widened type
        // instead of the narrow type live at the original evaluation
        // point.
        let body_value = if let Some(body) = node.body() {
            // `check_return_type` (below) owns the def-body return
            // assertion (`#: T` on the last expression) — its
            // `emit_false_assertion_if_incompatible` fires once for
            // that line. Flip the suppression flag so the
            // statement-position gate inside `check_statements_with_hint`
            // doesn't re-emit on the same last stmt. The flag is
            // consumed (replaced with `false`) on entry to the
            // outermost `check_statements_with_hint`, so nested
            // statement bodies inside the last stmt (e.g. `def f;
            // (begin; x; end); end`) still apply their own gate.
            let saved = std::mem::replace(&mut self.suppress_method_body_last_assertion, true);
            let return_hint = self.current_method_return_hint();
            let ty = self.check_node(&body, return_hint);
            self.suppress_method_body_last_assertion = saved;
            // Two shapes where `check_node`'s side-effecting walk yields a
            // value the return-type check can't use directly, so fall
            // back to the read-only `infer_type` path for this specific
            // (uncommon) body shape:
            //
            // - `Ty::UNTYPED`: `check_node`'s dispatch still shortcuts a
            //   few "declaration" node kinds (`ConstantOperatorWriteNode`
            //   et al. — the simple/or-and/operator/multi write
            //   subfamilies were split out one at a time, F1-F5 of the
            //   write-node value-type migration, though `IndexOrWriteNode`/
            //   `IndexAndWriteNode` graduated out of this cluster
            //   entirely) straight to `Ty::UNTYPED` when they're the
            //   body's last expression — their side effects still fire
            //   via `visit`, but the assignment's own RHS type isn't
            //   surfaced there (it's not needed by that dispatch's other
            //   callers). `infer_type` *does* return the RHS type for
            //   these, so e.g. `def f; @@x ||= 1; end` still reports a
            //   return-type mismatch against a non-Integer declared type.
            // - `Ty::BOTTOM`: `check_statements_with_hint` stops walking
            //   a statement list as soon as a stmt evaluates to BOTTOM
            //   (divergent control flow — the rest is dead code under a
            //   stale env), so a trailing unreachable statement after an
            //   explicit `return`/`raise` is never walked and the body's
            //   value collapses to that BOTTOM. `infer_type`'s statement
            //   arm has no such cutoff — it evaluates whichever node is
            //   textually last regardless of reachability — matching
            //   this method's existing contract that trailing dead code
            //   is still checked against the declared return type
            //   (`test_explicit_return_does_not_double_report_with_last_expression`).
            let ty = if ty.is_untyped() || ty == Ty::BOTTOM {
                self.infer_type(&body, return_hint)
            } else {
                ty
            };
            Some(ty)
        } else {
            None
        };

        self.check_return_type(node, body_value);

        if entered_method {
            self.ctx.leave_method();
        }
    }

    /// Shared body for `x ||= rhs` / `x &&= rhs`. Steep parity
    /// (`type_construction.rb` `:or_asgn` / `:and_asgn`): evaluate the RHS
    /// once for diagnostics, then overwrite the lvar binding with that
    /// type. No union narrowing — Steep does not preserve the LHS type
    /// even when `||=` could observably keep it. The `#: T` assertion is
    /// intentionally not consulted here (scope outside, pinned by test).
    /// `x += rhs` is NOT routed here: it needs the operator method's
    /// return type, handled separately.
    ///
    /// `hint` is the enclosing expression's type hint, forwarded to the
    /// RHS synthesis — Steep's `:lvasgn` arm inside `:or_asgn`/`:and_asgn`
    /// does the same (`synthesize(rhs, hint: hint)`, `type_construction.rb
    /// :2401`; unlike ivar/cvar/gvar, an lvar has no RBS declaration to
    /// hint with instead, so the outer hint is what Steep uses).
    ///
    /// `SPECIAL_LVAR_NAMES` (`_`, `__any__`, `__skip__`) get the same
    /// short-circuit as plain writes: the RHS is still checked for its
    /// side effects but the binding is left untouched, preserving the
    /// `_ = nil` cast idiom even on the compound path.
    pub(super) fn rebind_compound_lvar(
        &mut self,
        name_bytes: &[u8],
        depth: u32,
        value_node: &Node<'_>,
        hint: Option<Ty>,
    ) -> Ty {
        let value_type = self.check_node(value_node, hint);
        if is_special_lvar_name(name_bytes) {
            return Ty::UNTYPED;
        }
        let name_str = String::from_utf8_lossy(name_bytes);
        let name = self.checker_names().intern(&name_str);
        let value_type = self.freeze_lvar_bound_type(value_type, depth);
        self.ctx
            .set_local_variable_at_depth(name, value_type, depth);
        value_type
    }

    /// Bind a `LocalVariableTargetNode` to `ty` in the current scope chain
    /// at the target's declared depth. Mirrors the final
    /// `set_local_variable_at_depth` call in `visit_local_variable_write_node`
    /// for the multi-assignment target paths.
    pub(super) fn bind_local_variable_target(
        &mut self,
        node: &ruby_prism::LocalVariableTargetNode<'_>,
        ty: Ty,
    ) {
        let name_str = String::from_utf8_lossy(node.name().as_slice());
        let name = self.checker_names().intern(&name_str);
        let ty = self.freeze_lvar_bound_type(ty, node.depth());
        let byte_range = (node.location().start_offset(), node.location().end_offset());
        if self
            .bind_pinned_lvar_write(name, ty, node.depth(), byte_range)
            .is_some()
        {
            return;
        }
        self.ctx.set_local_variable_at_depth(name, ty, node.depth());
    }

    /// Multi-assignment fallback helper: bind the target's lvar to UNTYPED
    /// when it's a `LocalVariableTargetNode`, otherwise hand off to the
    /// default `Visit` walker (which recurses into nested `MultiTargetNode`
    /// / `SplatNode` and fires `visit_local_variable_target_node` on any
    /// inner lvar targets). Non-lvar targets (ivar / cvar / gvar / const /
    /// call / index) are scope-outside; the walker still descends so any
    /// side-effects on their inner expressions fire.
    fn visit_multi_assign_target<'pr>(&mut self, target_node: &Node<'pr>) {
        if let Some(target) = target_node.as_local_variable_target_node() {
            self.bind_local_variable_target(&target, Ty::UNTYPED);
        } else {
            self.visit(target_node);
        }
    }

    /// Multi-assignment UNTYPED fallback: drive the RHS once (so its
    /// inner diagnostics still fire), then register every LHS target
    /// (lefts / rest / rights) with the lvar-or-walker helper above.
    fn multi_assign_fallback<'pr>(
        &mut self,
        value_node: &Node<'pr>,
        lefts: &[Node<'pr>],
        rest: Option<&Node<'pr>>,
        rights: &[Node<'pr>],
    ) {
        self.check_node(value_node, None);
        self.multi_assign_fallback_targets(lefts, rest, rights);
    }

    fn multi_assign_fallback_targets<'pr>(
        &mut self,
        lefts: &[Node<'pr>],
        rest: Option<&Node<'pr>>,
        rights: &[Node<'pr>],
    ) {
        for target_node in lefts {
            self.visit_multi_assign_target(target_node);
        }
        if let Some(rest_node) = rest {
            self.visit_multi_assign_target(rest_node);
        }
        for target_node in rights {
            self.visit_multi_assign_target(target_node);
        }
    }

    /// Steep `expand_tuple` parity: distribute `elem_types` across
    /// `lefts` (from the front) and `rights` (from the back), with the
    /// remaining middle slice given to `splat` as `Type::Tuple`. Each
    /// binding (leading / splat / trailing) is then passed through
    /// [`wrap_optional_if`], so a single call covers both the
    /// `optional=false` ArrayNode-literal path and the `optional=true`
    /// Union-RHS path.
    ///
    /// Missing positions bind `Ty::NIL`. `splat = None` is the
    /// anonymous-splat case and skips its own bind while still
    /// consuming the middle slice.
    fn distribute_tuple_then_wrap(
        &mut self,
        elem_types: &[Ty],
        optional: bool,
        lefts: &[LocalVariableTargetNode<'_>],
        splat: Option<&LocalVariableTargetNode<'_>>,
        rights: &[LocalVariableTargetNode<'_>],
    ) {
        let nleading = lefts.len();
        let ntrailing = rights.len();
        let nelems = elem_types.len();

        for (i, target) in lefts.iter().enumerate() {
            let ty = if i < nelems { elem_types[i] } else { Ty::NIL };
            let wrapped = self.wrap_optional_if(ty, optional);
            self.bind_local_variable_target(target, wrapped);
        }

        if let Some(splat_target) = splat {
            let middle_start = nleading.min(nelems);
            let middle_end = nelems.saturating_sub(ntrailing).max(middle_start);
            let middle: Vec<Ty> = elem_types[middle_start..middle_end].to_vec();
            let tuple_ty = self.env.types().intern(Type::Tuple(middle));
            let wrapped = self.wrap_optional_if(tuple_ty, optional);
            self.bind_local_variable_target(splat_target, wrapped);
        }

        for (i, target) in rights.iter().enumerate() {
            // rights[i] takes elem_types[nelems - ntrailing + i] when
            // that position falls strictly after the leading region;
            // otherwise the trailing slot is missing and binds nil.
            let pos = nelems as isize - ntrailing as isize + i as isize;
            let ty = if pos >= nleading as isize && pos >= 0 && (pos as usize) < nelems {
                elem_types[pos as usize]
            } else {
                Ty::NIL
            };
            let wrapped = self.wrap_optional_if(ty, optional);
            self.bind_local_variable_target(target, wrapped);
        }
    }

    /// Steep `try_convert(type, :to_ary)` port
    /// (`type_construction.rb:2926` + `try_convert` at 5019).
    /// Returns the substituted return type of a `to_ary` overload on
    /// `receiver_ty` when one is publicly declared and truly arity-
    /// zero (no required positional / keyword / trailing param).
    /// Returns `None` when the method is absent, private, all
    /// overloads take required parameters, or the receiver falls
    /// outside `lookup_method`'s handled kinds (Union, Var, etc.).
    ///
    /// Overload selection mirrors [`Self::try_convert_to_a`] on the
    /// `to_a` side (`inference.rs`): same `required_*` gate,
    /// `substitution_for_receiver` for parameter binding. `block`
    /// shape is intentionally not inspected — Steep's `try_convert`
    /// doesn't look at it, so neither does this port. A pathological
    /// `to_ary` with a mandatory block would be picked up here just
    /// as Steep does.
    ///
    /// Consumed by [`Self::visit_multi_write_node`] to destructure a
    /// non-Tuple / non-Array[T] RHS via the receiver's `to_ary` — the
    /// core steep-repo idiom `type, constr = synthesize(node)` where
    /// the RHS returns a class carrying `def to_ary: () -> [...]`.
    fn try_convert_to_ary_return_type(&self, receiver_ty: Ty) -> Option<Ty> {
        let to_ary = self.env.names().intern_symbol("to_ary");
        let resolved = super::method_resolver::lookup_method(self.env, receiver_ty, to_ary)?;
        if resolved.method.accessibility == Visibility::Private {
            return None;
        }
        let arg_free = resolved.method.method_types().find(|mt| {
            mt.is_untyped_function()
                || (mt.required_positionals().is_empty()
                    && mt.required_keywords().is_empty()
                    && mt.trailing_positionals().is_empty())
        })?;
        let subst =
            definition_builder::substitution_for_receiver(self.env, receiver_ty, resolved.bindings);
        Some(subst.apply(arg_free.return_type(), self.env.types()))
    }

    fn bind_scalar_multi_assign_rhs(
        &mut self,
        value_ty: Ty,
        lefts: &[LocalVariableTargetNode<'_>],
        rights: &[LocalVariableTargetNode<'_>],
    ) {
        let mut targets = lefts.iter().chain(rights.iter());
        if let Some(first) = targets.next() {
            self.bind_local_variable_target(first, value_ty);
        }
        for target in targets {
            self.bind_local_variable_target(target, Ty::NIL);
        }
    }

    pub(in crate::type_checker) fn multi_assign_truthy_is_expandable(&self, truthy: Ty) -> bool {
        let truthy = definition_builder::expand_alias(self.env, truthy);
        match self.env.types().resolve(truthy) {
            Type::Tuple(_) => true,
            Type::ClassInstance { name, args } => {
                *name == self.env.names().builtins().array && args.len() == 1
            }
            _ => false,
        }
    }

    pub(in crate::type_checker) fn multi_assign_truthy_is_scalar(&self, truthy: Ty) -> bool {
        let truthy = definition_builder::expand_alias(self.env, truthy);
        if truthy.is_untyped() {
            return false;
        }
        match self.env.types().resolve(truthy) {
            Type::Tuple(_) => false,
            Type::ClassInstance { name, args } => {
                !(*name == self.env.names().builtins().array && args.len() == 1)
            }
            Type::Union(members) => members
                .iter()
                .all(|member| self.multi_assign_truthy_is_scalar(*member)),
            Type::Optional(inner) => self.multi_assign_truthy_is_scalar(*inner),
            _ => true,
        }
    }

    /// Steep `union_of_tuple_to_tuple_of_union` port
    /// (`type_construction.rb:4743`). Resolve `ty` and, when it is a
    /// `Type::Union` whose every member is a `Type::Tuple`, return the
    /// per-position element types: short tuples are padded with
    /// `Ty::NIL` to the max arity, columns are transposed, and each
    /// column is unified via `union_of_many`. Returns `None` when the
    /// gate fails (non-Union, or any member is not a Tuple).
    pub(in crate::type_checker) fn union_of_tuple_to_tuple_of_union(
        &self,
        ty: Ty,
    ) -> Option<Vec<Ty>> {
        let types = self.env.types();
        let ty = definition_builder::expand_alias(self.env, ty);
        let members = match types.resolve(ty) {
            Type::Union(members) => members,
            _ => return None,
        };
        let tuple_rows: Option<Vec<Vec<Ty>>> = members
            .iter()
            .copied()
            .map(|m| {
                let member = definition_builder::expand_alias(self.env, m);
                match types.resolve(member) {
                    Type::Tuple(elems) => Some(elems.to_vec()),
                    _ => None,
                }
            })
            .collect();
        let tuple_rows = tuple_rows?;
        let max = tuple_rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut columns: Vec<Vec<Ty>> = (0..max)
            .map(|_| Vec::with_capacity(tuple_rows.len()))
            .collect();
        for row in &tuple_rows {
            for (i, col) in columns.iter_mut().enumerate() {
                col.push(*row.get(i).unwrap_or(&Ty::NIL));
            }
        }
        Some(
            columns
                .into_iter()
                .map(|col| union_of_many(&col, types))
                .collect(),
        )
    }

    /// Wrap `ty` with `nil` when `optional` is set, mirroring Steep's
    /// `AST::Builtin.optional` (`type_construction.rb:2869`).
    ///
    /// Uses `Type::Optional(ty)` so the receiver-display layer can apply
    /// its existing unfold (`Optional(T) → T | nil`) and any inner widen
    /// (`Tuple → Array[union]`) in sequence — building `Union(ty, Nil)`
    /// directly would bypass the Tuple widen on the splat slice.
    ///
    /// Dedup:
    /// - already `Type::Nil` → no-op (Steep's `Optional(nil) = nil`)
    /// - already `Type::Optional(_)` → no-op (avoid `Optional(Optional(T))`)
    fn wrap_optional_if(&self, ty: Ty, optional: bool) -> Ty {
        if !optional {
            return ty;
        }
        let types = self.env.types();
        match types.resolve(ty) {
            Type::Nil => Ty::NIL,
            Type::Optional(_) => ty,
            _ => types.intern(Type::Optional(ty)),
        }
    }

    /// Steep `type_masgn_type` port (`type_construction.rb:2860`). Takes
    /// the truthy partition `truthy` and the `optional` flag, dispatches
    /// to the matching `expand_*` shape, and wraps each binding with
    /// `Optional` when `optional` is set. The splat target receives the
    /// same `Optional` wrap as leading/trailing.
    ///
    /// - `Type::Tuple(elems)` → `expand_tuple` parity (front-shift /
    ///   back-pop, middle becomes `Type::Tuple(middle)`)
    /// - `Type::ClassInstance{::Array, [T]}` → `expand_array` parity
    ///   (leading/trailing = `Optional(T)`, splat = `Array[T]`)
    /// - Anything else → UNTYPED fallback bind (the RHS side-effect has
    ///   already fired by the caller)
    fn multi_assign_distribute_with_optional(
        &mut self,
        truthy: Ty,
        optional: bool,
        lefts: &[LocalVariableTargetNode<'_>],
        splat: Option<&LocalVariableTargetNode<'_>>,
        rights: &[LocalVariableTargetNode<'_>],
    ) {
        let truthy = definition_builder::expand_alias(self.env, truthy);
        let truthy_resolved = self.env.types().resolve(truthy);
        match truthy_resolved {
            Type::Tuple(elem_types) => {
                self.distribute_tuple_then_wrap(elem_types, optional, lefts, splat, rights);
            }
            Type::ClassInstance { name, args }
                if *name == self.env.names().builtins().array && args.len() == 1 =>
            {
                let elem_ty = args[0];
                let leading_ty = self.env.types().intern(Type::Optional(elem_ty));
                let leading_wrapped = self.wrap_optional_if(leading_ty, optional);
                let array_name = self.env.names().builtins().array;
                let array_ty = self.env.types().intern(Type::ClassInstance {
                    name: array_name,
                    args: vec![elem_ty],
                });
                let splat_wrapped = self.wrap_optional_if(array_ty, optional);
                for target in lefts {
                    self.bind_local_variable_target(target, leading_wrapped);
                }
                if let Some(t) = splat {
                    self.bind_local_variable_target(t, splat_wrapped);
                }
                for target in rights {
                    self.bind_local_variable_target(target, leading_wrapped);
                }
            }
            _ => {
                for target in lefts {
                    self.bind_local_variable_target(target, Ty::UNTYPED);
                }
                if let Some(t) = splat {
                    self.bind_local_variable_target(t, Ty::UNTYPED);
                }
                for target in rights {
                    self.bind_local_variable_target(target, Ty::UNTYPED);
                }
            }
        }
    }

    /// Walk the body of a `Const = <ctor>(<args?>) do ... end` class
    /// construction (`Class.new`, `Struct.new`, or `Data.define`) as
    /// the body of `class Const`. The dispatch site has already
    /// confirmed the call shape and that the LHS is declared in RBS as
    /// a class.
    ///
    /// Arguments are walked first (in the outer lexical scope) so a
    /// missing parent constant / member symbol still emits its usual
    /// outer-scope diagnostics. The class is then pushed onto the
    /// lexical stack and the block body is walked via `check_node` —
    /// the same entry point `visit_class_node` uses for a real `class`
    /// declaration body. `singleton_class_depth` is reset around the
    /// body, mirroring `visit_class_node`, so a stray `class << self`
    /// arm inside the block starts at depth 0.
    ///
    /// The push is `push_class_outside_cref`: Ruby's `class_eval` on
    /// the block changes `self` and the `def` target but not the cref,
    /// so `INNER = 1` inside the block defines the constant in the
    /// enclosing scope and `INNER` reads resolve from there (rbs's
    /// `InlineParser` drops such writes entirely; Steep with sig-only
    /// resolves them at the outer scope as well).
    fn walk_class_construction_block_body<'pr>(&mut self, call: &CallNode<'pr>, lhs_name: &str) {
        if let Some(args) = call.arguments() {
            for arg in args.arguments().iter() {
                self.check_node(&arg, None);
            }
        }

        self.ctx.push_class_outside_cref(lhs_name, self.env.names());
        let saved_singleton_class_depth = self.ctx.replace_singleton_class_depth(0);

        if let Some(block_arg) = call.block()
            && let Some(block_node) = block_arg.as_block_node()
            && let Some(body) = block_node.body()
        {
            self.check_node(&body, None);
        }

        self.ctx
            .replace_singleton_class_depth(saved_singleton_class_depth);
        self.ctx.pop_class();
    }

    /// `included do` / `prepended do` on an ActiveSupport::Concern
    /// module. Rails `class_eval`s the block on every class that
    /// (transitively) includes the concern, so the body is walked once
    /// per final target with the class stack replaced by `[target]` —
    /// `self` is `singleton(target)`, the same retargeting
    /// `visit_def_node` applies to a synthetic concern `def`, and the
    /// same body entry (`check_node`) `walk_class_construction_block_body`
    /// uses. A `|base|` block param is bound to that singleton too. The
    /// block scope itself was already pushed by `check_call`'s
    /// `setup_block_scope`, so the rebinding lands in it.
    ///
    /// Returns `false` when the call is not a block the infusion pipeline
    /// expanded (receiver present, not `included` / `prepended`, a
    /// non-Concern module, a second `included do` in the same module),
    /// and the caller keeps the default walk under the module's own
    /// singleton. A collected block with zero targets returns `true`
    /// without walking anything: Rails never runs it, and a
    /// module-singleton walk would only report `NoMethod` on DSL calls
    /// (`has_many`) the concern module itself never answers.
    pub(super) fn walk_concern_block_body<'pr>(
        &mut self,
        call: &CallNode<'pr>,
        block: &Node<'pr>,
    ) -> bool {
        if call.receiver().is_some() || self.ctx.method_name().is_some() {
            return false;
        }
        let name = call.name().as_slice();
        if name != b"included" && name != b"prepended" {
            return false;
        }
        let Some(block_node) = block.as_block_node() else {
            return false;
        };
        let Some(concern) = self.ctx.current_class_typename().copied() else {
            return false;
        };
        let location = crate::inline_parser::prism_location_range(block_node.location());
        let source_file = self.env.names().intern(&self.file.to_string_lossy());
        let Some(targets) = self
            .env
            .concern_block_targets(concern, location, Some(source_file))
        else {
            return false;
        };
        let Some(body) = block_node.body() else {
            return true;
        };
        let base_param = block_node
            .parameters()
            .and_then(|p| p.as_block_parameters_node())
            .and_then(|bp| bp.parameters())
            .and_then(|params| {
                let requireds = params.requireds();
                if requireds.len() != 1 {
                    return None;
                }
                requireds.iter().next()
            })
            .and_then(|p| p.as_required_parameter_node())
            .map(|p| String::from_utf8_lossy(p.name().as_slice()).to_string());

        let saved_walk = std::mem::replace(&mut self.concern_block_target_walk, true);
        // Each target is a separate `class_eval` at runtime, so each walk
        // starts from the same lvar state: without the reset, a binding
        // (or narrowing) from the first target's walk would leak into the
        // next one through the single block scope `check_call` pushed
        // (crema-review adversarial finding).
        let scopes_before = self.ctx.snapshot_scopes();
        for target in targets {
            self.ctx.restore_scopes(scopes_before.clone());
            let saved_class_stack = self.ctx.replace_class_stack(vec![target]);

            let saved_singleton_class_depth = self.ctx.replace_singleton_class_depth(0);
            if let Some(base) = &base_param {
                let base = self.checker_names().intern(base);
                let ty = self.env.types().class_singleton(target);
                self.ctx.set_local_variable(base, ty);
            }
            self.check_node(&body, None);
            self.ctx
                .replace_singleton_class_depth(saved_singleton_class_depth);
            self.ctx.restore_class_stack(saved_class_stack);
        }
        self.concern_block_target_walk = saved_walk;
        true
    }

    /// Push a `class` / `module` declaration onto the context stack keyed by
    /// its FULL lexical name: `class A::B` is `::A::B`, not the Prism leaf
    /// `::B`. A rooted path (`class ::A::B`) is already absolute and ignores
    /// nesting. Returns `None` for a dynamic path (`class (expr)::Foo`) so
    /// callers can skip the body — mirrors the inline collector's
    /// `push_class_abs_path`.
    fn push_decl_class(&mut self, constant_path: &Node<'_>, _leaf: &str) -> Option<()> {
        match crate::inline_parser::class_decl_path_string(constant_path) {
            Some(path) if path.starts_with("::") => {
                self.ctx.push_class_absolute(&path, self.env.names())
            }
            Some(path) => self.ctx.push_class(&path, self.env.names()),
            None => return None,
        }
        Some(())
    }

    /// Resolve any trailing `#: T` on the source line ending at
    /// `end_offset`, returning the asserted `Ty` when the annotation
    /// parses and every referenced type name resolves against the
    /// environment. `start_offset` positions the `UnknownTypeName`
    /// diagnostic. Shared by the assignment path (`#: T` on the
    /// assignment's line) and the return paths (`#: T` on a method
    /// body's last expression / an explicit `return`).
    ///
    /// When the annotation parses but mentions an unknown class /
    /// interface / type alias, emits one `UnknownTypeName` per missing
    /// name and returns `None` so the caller falls back to the natural
    /// inferred type (matches Steep's `RBSError` short-circuit shape).
    /// Parse failures and other build-time errors still fall back
    /// silently for now; the existing inline-parse pass already reports
    /// `AnnotationSyntaxError` for the syntactic-typo case.
    pub(super) fn lookup_trailing_assertion(
        &mut self,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<Ty> {
        if !self.comments.has_trailing_annotations() {
            return None;
        }
        // `end_offset` from Prism is exclusive; step one byte back to land
        // on the last byte of the node so we look up the comment on its
        // source line.
        let end_line = self.line_index.line(end_offset.saturating_sub(1));
        let trailing = self.comments.trailing_annotation(&self.source, end_line)?;
        let (range, type_text) = match trailing {
            TrailingAnnotation::NodeTypeAssertion { range, type_text } => (range, type_text),
            _ => return None,
        };
        let type_text = type_text.to_string();

        let ast_type = match ast_builder::parse_trailing_type_text(&type_text, self.env.names()) {
            Some(ast_type) => ast_type,
            None => {
                // Two-axis design for expression assertions:
                //   - empty `#:` (zero-length / whitespace-only body) →
                //     silent. Steep parity: `ast/node/type_assertion.rb`
                //     requires `\A:\s*(.+)` so empty bodies never become
                //     assertions.
                //   - non-empty broken type text (e.g. `#: NotAType<<`) →
                //     emit `AnnotationSyntaxError`. Intentional divergence
                //     from Steep, which silently drops the assertion via
                //     `type_syntax?` rescuing `RBS::ParsingError`. crema
                //     surfaces type-annotation mistakes as a preflight
                //     checker.
                // Declaration paths (constant / attr / def trailing
                // return) emit on both axes through their own pipelines;
                // this branch governs both the `lvar = expr #:`
                // assignment form and the bare statement-position form
                // (`expr #:` at top-level / class body / def body),
                // gated in `check_statements_with_hint` via
                // `is_statement_assertion_eligible`.
                if !type_text.trim_ascii().is_empty() {
                    self.push_annotation_syntax_error(range);
                }
                return None;
            }
        };
        let context =
            build_lowering_context_from_cref_stack(self.ctx.cref_stack(), self.env.names());
        let ty = self
            .env
            .lower_ast_type(&ast_type, &context, &TypeParamScope::default());

        let mut unknown_names = Vec::new();
        self.collect_unknown_type_names(ty, &mut unknown_names);
        if !unknown_names.is_empty() {
            let position = self.offset_to_location(start_offset);
            for name in unknown_names {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position.clone(),
                    kind: DiagnosticKind::UnknownTypeName { name },
                });
            }
            return None;
        }

        let type_text_offset = self.trailing_type_text_offset(range, &type_text);
        let mut locator = AssertionTypeLocator::new(&type_text, type_text_offset);
        self.check_assertion_class_type_arg_bounds(&ast_type, &context, &mut locator);

        Some(ty)
    }

    /// Resolve a method call receiver's own trailing `#: T` assertion,
    /// trying each parenthesization layer outside-in. A bare receiver's
    /// `end_offset` is tried first; if no assertion is adjacent there,
    /// one paren layer is unwrapped and retried, and so on. This handles
    /// both `(expr #: T).method_call` (assertion adjacent to the
    /// innermost expression, one layer in) and `((expr) #: T).method_call`
    /// (assertion adjacent to an intermediate parenthesized wrapper, not
    /// the innermost expression) — jumping straight to the innermost
    /// expression (an earlier version of this lookup did) skips the
    /// outer layers entirely and misses the second shape, silently
    /// reproducing the exact bug this lookup exists to fix (caught by
    /// crema-review adversarial pass on `assertion_type_lost_map_block_sig_defined_call`).
    pub(super) fn lookup_receiver_trailing_assertion<'pr>(&self, node: &Node<'pr>) -> Option<Ty> {
        if let Some(ty) = self.lookup_trailing_assertion_ty_readonly(node.location().end_offset()) {
            return Some(ty);
        }
        let body = node.as_parentheses_node()?.body()?;
        match body.as_statements_node() {
            Some(stmts) => {
                let last = stmts.body().iter().last()?;
                self.lookup_receiver_trailing_assertion(&last)
            }
            None => self.lookup_receiver_trailing_assertion(&body),
        }
    }

    /// Diagnostic-free core of `lookup_trailing_assertion`: same parse +
    /// lower + unknown-name check, but returns `None` (rather than
    /// emitting `AnnotationSyntaxError` / `UnknownTypeName`) on any
    /// failure instead of pushing to the diagnostic sink. For `&self`
    /// contexts that cannot call the `&mut self` original — currently
    /// `infer_receiver_type`, which resolves a method call's receiver
    /// type and is shared by both the `&mut self` diagnostic-emission
    /// pass and the `&self` return-type synthesis pass (see
    /// `ResolvedCall::resolve`'s doc comment in `calls.rs`). Relies on
    /// `apply_statement_assertion_gate` (or another statement-level
    /// caller of `lookup_trailing_assertion`) already covering
    /// diagnostic emission for the same source range when the asserted
    /// expression also sits in statement position; a bare `#: T` that
    /// only ever appears at a receiver position (never a statement) does
    /// not get syntax/unknown-name diagnostics from this path, matching
    /// how type argument bounds are checked only by the `&mut self`
    /// path today.
    pub(super) fn lookup_trailing_assertion_ty_readonly(&self, end_offset: usize) -> Option<Ty> {
        let type_text = self.trailing_node_assertion_adjacent(end_offset)?;
        let type_text = type_text.to_string();
        let ast_type = ast_builder::parse_trailing_type_text(&type_text, self.env.names())?;
        let context =
            build_lowering_context_from_cref_stack(self.ctx.cref_stack(), self.env.names());
        let ty = self
            .env
            .lower_ast_type(&ast_type, &context, &TypeParamScope::default());

        let mut unknown_names = Vec::new();
        self.collect_unknown_type_names(ty, &mut unknown_names);
        if !unknown_names.is_empty() {
            return None;
        }
        Some(ty)
    }

    /// Whether a trailing `#: T` annotation comment sits immediately
    /// after `end_offset` (nothing but ASCII whitespace between them),
    /// returning its raw type text when so. Shared by
    /// `lookup_trailing_assertion_ty_readonly` and the receiver-position
    /// diagnostic gate (`apply_receiver_assertion_gate`).
    ///
    /// `trailing_annotation` keys only by line number, which is fine for
    /// statement-level callers (a statement's own end_offset is always
    /// its line's last real token, immediately before any trailing
    /// comment). Both callers of this helper run at a sub-expression
    /// position, where `end_offset` can be *mid-line*, sharing a line
    /// with an unrelated trailing assertion on a later sub-expression
    /// (e.g. `x = a.foo(y).bar #: T` — `.bar`'s receiver `a.foo(y)` ends
    /// mid-line, `#: T` belongs to the whole statement, not to
    /// `a.foo(y)`). Requiring every byte between `end_offset` and the
    /// comment's own start to be ASCII whitespace ensures only an
    /// assertion truly adjacent to this node is used.
    pub(super) fn trailing_node_assertion_adjacent(&self, end_offset: usize) -> Option<&str> {
        if !self.comments.has_trailing_annotations() {
            return None;
        }
        let end_line = self.line_index.line(end_offset.saturating_sub(1));
        let trailing = self.comments.trailing_annotation(&self.source, end_line)?;
        let (range, type_text) = match trailing {
            TrailingAnnotation::NodeTypeAssertion { range, type_text } => (range, type_text),
            _ => return None,
        };
        let comment_start = range.0 as usize;
        if end_offset > comment_start
            || !self
                .source
                .get(end_offset..comment_start)
                .is_some_and(|between| between.iter().all(u8::is_ascii_whitespace))
        {
            return None;
        }
        Some(type_text)
    }

    /// Diagnostic-emitting counterpart of `lookup_receiver_trailing_assertion`,
    /// for `check_call`'s `&mut self` pass (`calls.rs::check_call`). Same
    /// paren-unwrap traversal, but gated by `trailing_node_assertion_adjacent`
    /// before delegating to `lookup_trailing_assertion` — a receiver's
    /// end_offset can share a line with an unrelated later sub-expression's
    /// trailing assertion (`a.foo(y).bar #: T`), and `lookup_trailing_assertion`
    /// itself has no adjacency check (statement-position callers don't need
    /// one).
    ///
    /// Callers must run `self.visit(&receiver)` (or otherwise execute the
    /// receiver's own subtree) *before* calling this — the natural type is
    /// computed lazily, only once an adjacent assertion is actually found,
    /// by reading the checker's live scope state. If a preceding sibling
    /// statement inside the receiver's parens hasn't run yet (e.g. `(\n a
    /// = 1\n a #: T\n).m` — `a`'s binding happens in that first
    /// statement), reading `a`'s type before that statement executes
    /// either finds no binding at all (a fresh local falls back to
    /// `UNTYPED`, silently swallowing a real `FalseAssertion`) or an
    /// outer scope's now-stale binding (misreporting a valid assertion as
    /// false) — caught by crema-review adversarial pass.
    pub(super) fn apply_receiver_assertion_gate<'pr>(&mut self, node: &Node<'pr>) {
        let end_offset = node.location().end_offset();
        if self.trailing_node_assertion_adjacent(end_offset).is_some() {
            // `LocalVariableWriteNode` / `MultiWriteNode` / `ReturnNode`
            // each have their own dedicated pipeline that independently
            // calls `lookup_trailing_assertion` on this same comment
            // (`check_local_variable_write`, `check_multi_write_node`,
            // `check_explicit_return`), reached via the `self.visit` walk
            // callers run alongside this gate. Consuming it here too
            // would double-fire. Deliberately narrower than
            // `is_statement_assertion_eligible` (statement position's
            // exclusion list): that list also excludes `ParenthesesNode`,
            // but this gate's own unwrap below is exactly what must
            // consume an assertion adjacent to an *intermediate* paren
            // layer (`((expr)) #: T`), so `ParenthesesNode` has to stay
            // eligible here — caught by crema-review spec-consistency
            // pass.
            if matches!(
                node,
                Node::LocalVariableWriteNode { .. }
                    | Node::MultiWriteNode { .. }
                    | Node::ReturnNode { .. }
            ) {
                return;
            }
            let start_offset = node.location().start_offset();
            let natural = self.infer_type(node, None);
            if let Some(asserted) = self.lookup_trailing_assertion(start_offset, end_offset) {
                let position = self.offset_to_location(start_offset);
                self.emit_false_assertion_if_incompatible(natural, asserted, position);
            }
            return;
        }
        let Some(body) = node.as_parentheses_node().and_then(|p| p.body()) else {
            return;
        };
        match body.as_statements_node() {
            Some(stmts) => {
                if let Some(last) = stmts.body().iter().last() {
                    self.apply_receiver_assertion_gate(&last);
                }
            }
            None => self.apply_receiver_assertion_gate(&body),
        }
    }

    fn trailing_type_text_offset(
        &self,
        range: crate::ast::ruby::PrismByteRange,
        type_text: &str,
    ) -> usize {
        let comment = self
            .source
            .get(range.0 as usize..range.1 as usize)
            .unwrap_or_default();
        let local = std::str::from_utf8(comment)
            .ok()
            .and_then(|text| text.find(type_text))
            .unwrap_or(0);
        range.0 as usize + local
    }

    fn check_assertion_class_type_arg_bounds(
        &mut self,
        ast_ty: &ast_types::Type,
        context: &[Option<Name>],
        locator: &mut AssertionTypeLocator<'_>,
    ) {
        match ast_ty {
            ast_types::Type::ClassInstance(ast_types::ClassInstanceType { args, .. }) => {
                let ty = self
                    .env
                    .lower_ast_type(ast_ty, context, &TypeParamScope::default());
                if let Type::ClassInstance { name, args } = self.env.types().resolve(ty) {
                    let container_name = self.env.names().resolve(name).to_string();
                    let position_offset = locator.find_type_application(&container_name);
                    self.emit_assertion_class_type_arg_bound_violations(
                        *name,
                        args,
                        position_offset,
                    );
                }
                for ast_arg in args {
                    self.check_assertion_class_type_arg_bounds(ast_arg, context, locator);
                }
            }
            ast_types::Type::Interface(ast_types::InterfaceType { args, .. })
            | ast_types::Type::Alias(ast_types::AliasType { args, .. })
            | ast_types::Type::ClassSingleton(ast_types::ClassSingletonType { args, .. }) => {
                for ast_arg in args {
                    self.check_assertion_class_type_arg_bounds(ast_arg, context, locator);
                }
            }
            ast_types::Type::Union(ast_types::UnionType { types, .. })
            | ast_types::Type::Intersection(ast_types::IntersectionType { types, .. })
            | ast_types::Type::Tuple(ast_types::TupleType { types, .. }) => {
                for member in types {
                    self.check_assertion_class_type_arg_bounds(member, context, locator);
                }
            }
            ast_types::Type::Optional(ast_types::OptionalType { ty, .. }) => {
                self.check_assertion_class_type_arg_bounds(ty, context, locator);
            }
            ast_types::Type::Record(ast_types::RecordType { fields, .. }) => {
                for field in fields {
                    self.check_assertion_class_type_arg_bounds(&field.ty, context, locator);
                }
            }
            ast_types::Type::Proc(proc_type) => {
                self.check_assertion_class_type_arg_bounds_in_fn(
                    &proc_type.function,
                    context,
                    locator,
                );
                if let Some(self_type) = &proc_type.self_type {
                    self.check_assertion_class_type_arg_bounds(self_type, context, locator);
                }
                if let Some(block) = &proc_type.block {
                    self.check_assertion_class_type_arg_bounds_in_fn(
                        &block.function,
                        context,
                        locator,
                    );
                    if let Some(self_type) = &block.self_type {
                        self.check_assertion_class_type_arg_bounds(self_type, context, locator);
                    }
                }
            }
            _ => {}
        }
    }

    fn check_assertion_class_type_arg_bounds_in_fn(
        &mut self,
        ast_fn: &ast_types::Function,
        context: &[Option<Name>],
        locator: &mut AssertionTypeLocator<'_>,
    ) {
        let ast_fn = match ast_fn {
            ast_types::Function::Typed(f) => {
                for param in &f.required_positionals {
                    self.check_assertion_class_type_arg_bounds(&param.ty, context, locator);
                }
                for param in &f.optional_positionals {
                    self.check_assertion_class_type_arg_bounds(&param.ty, context, locator);
                }
                if let Some(param) = &f.rest_positionals {
                    self.check_assertion_class_type_arg_bounds(&param.ty, context, locator);
                }
                for param in &f.trailing_positionals {
                    self.check_assertion_class_type_arg_bounds(&param.ty, context, locator);
                }
                for param in &f.required_keywords {
                    self.check_assertion_class_type_arg_bounds(&param.param.ty, context, locator);
                }
                for param in &f.optional_keywords {
                    self.check_assertion_class_type_arg_bounds(&param.param.ty, context, locator);
                }
                if let Some(param) = &f.rest_keywords {
                    self.check_assertion_class_type_arg_bounds(&param.ty, context, locator);
                }
                f
            }
            ast_types::Function::Untyped(f) => {
                self.check_assertion_class_type_arg_bounds(&f.return_type, context, locator);
                return;
            }
        };
        self.check_assertion_class_type_arg_bounds(&ast_fn.return_type, context, locator);
    }

    fn emit_assertion_class_type_arg_bound_violations(
        &mut self,
        name: TypeName,
        args: &[Ty],
        position_offset: usize,
    ) {
        let Some(params) = self.env.class_type_params_by_type_name(&name) else {
            return;
        };
        if params.is_empty() || args.is_empty() {
            return;
        }

        let mut bindings = FxHashMap::default();
        for (param, &arg) in params.iter().zip(args) {
            bindings.insert(param.name.clone(), arg);
        }

        let violations = self.subtyper().check_type_arg_bounds(params, &bindings);
        if violations.is_empty() {
            return;
        }

        let container_name = self.env.names().resolve(name).to_string();
        let position = self.offset_to_location(position_offset);
        for violation in violations {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position.clone(),
                kind: DiagnosticKind::TypeArgumentBoundViolation {
                    container_name: container_name.clone(),
                    param_name: self.env.names().resolve(violation.param_name),
                    bound_kind: violation.bound_kind,
                    bound: self.display_type(violation.bound),
                    actual: self.display_type(violation.actual),
                },
            });
        }
    }

    /// Emit a `FalseAssertion` (Hint severity) when a `#: T` assertion
    /// and the expression's natural type are mutually incompatible —
    /// neither widens nor narrows into the other. Shared by the
    /// assignment path and the return paths so `expr #: T` carries the
    /// same meaning regardless of position. Uses `subtyper()` (free type
    /// variables as wildcards) to match the assignment gate.
    pub(super) fn emit_false_assertion_if_incompatible(
        &mut self,
        natural: Ty,
        asserted: Ty,
        location: crate::location::SourceLocation,
    ) {
        let checker = self.subtyper();
        if !checker.check(natural, asserted) && !checker.check(asserted, natural) {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location,
                kind: DiagnosticKind::FalseAssertion {
                    natural: self.display_type(natural),
                    asserted: self.display_type(asserted),
                },
            });
        }
    }

    /// Shared body for ivar `||=` / `&&=`. `hint` is the *ambient* hint
    /// this whole `or_asgn`/`and_asgn` expression received from
    /// `check_node`'s own hint parameter — a method's declared return
    /// type when the compound write sits in tail position, but equally
    /// an array/tuple/hash element hint, a call argument hint, etc., for
    /// any other value position `check_node` is reached from. Steep's
    /// `:ivasgn` arm of the `or_asgn`/`and_asgn` case forwards that same
    /// ambient `hint` to the RHS unchanged (`type_construction.rb:2403-2405`,
    /// `synthesize(rhs, hint: hint)`), never the ivar's own declared
    /// type. Using the declared type as the RHS hint instead (as plain
    /// `@x = v` correctly does) leaks it into generic `T` unification
    /// for a block-returning RHS (`[T] () { () -> T } -> T`):
    /// `apply_hint_override` binds `T` from that hint before
    /// `augment_bindings_with_block_body` gets a chance to unify it
    /// against the block's actual return type, and a bound `T` is
    /// skipped there (`infer_return_type`, `inference.rs`) — widening
    /// `T` to whatever the declared type happens to be instead of what
    /// the block actually returns. `None` at statement position (no
    /// ambient hint to give), matching `rebind_compound_lvar`'s own
    /// statement-position callers. `+=` has different semantics
    /// (operator dispatch on the declared type) and does not share
    /// this helper.
    pub(super) fn check_compound_ivar_write_rhs(
        &mut self,
        name_bytes: &[u8],
        value: &Node<'_>,
        hint: Option<Ty>,
        location_offset: usize,
    ) -> Ty {
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_ivar_at_self(self, var_sym);
        self.check_var_write_rhs(
            value,
            &name_str,
            hint,
            declared,
            |name| DiagnosticKind::UnknownInstanceVariable { name },
            location_offset,
        )
    }

    /// Sibling of `check_compound_ivar_write_rhs` for cvar `||=` / `&&=`.
    /// Steep's `or_asgn`/`and_asgn` `case asgn.type` has no `:cvasgn`
    /// arm at all (only `:lvasgn`/`:ivasgn`/`:gvasgn`/`:send`) — a
    /// `@@x ||= v` falls to `fallback_to_any`, so there's no ambient
    /// hint to forward in the first place (verified 2026-07-17, see the
    /// `ClassVariableOrWriteNode` dispatch comment in `inference.rs`).
    /// No `hint` param: this is only reached from the statement-position
    /// visitor callback, which has no ambient hint to give either.
    pub(super) fn check_compound_cvar_write_rhs(
        &mut self,
        name_bytes: &[u8],
        value: &Node<'_>,
        location_offset: usize,
    ) -> Ty {
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_cvar_at_self(self, var_sym);
        self.check_var_write_rhs(
            value,
            &name_str,
            declared,
            declared,
            |name| DiagnosticKind::UnknownClassVariable { name },
            location_offset,
        )
    }

    /// Sibling of `check_compound_ivar_write_rhs` for gvar `||=` / `&&=`.
    /// Globals are namespace-free so the lookup is a single
    /// `env.lookup_global` (`&str`) — no Symbol intern, no ancestor
    /// walk. Steep's `:gvasgn` arm forwards the ambient `hint` the same
    /// way `:ivasgn` does (`type_construction.rb:2406-2408`) — see
    /// `check_compound_ivar_write_rhs`'s doc comment for the full
    /// rationale.
    pub(super) fn check_compound_gvar_write_rhs(
        &mut self,
        name_bytes: &[u8],
        value: &Node<'_>,
        hint: Option<Ty>,
        location_offset: usize,
    ) -> Ty {
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let declared = self.env.lookup_global(&name_str);
        self.check_var_write_rhs(
            value,
            &name_str,
            hint,
            declared,
            |name| DiagnosticKind::UnknownGlobalVariable { name },
            location_offset,
        )
    }

    /// `$x ||= y`: Steep `or_asgn`/`and_asgn` with a `gvasgn` LHS
    /// dispatches to the same `constr.gvasgn(asgn, type)` as plain
    /// `$x = rhs` (`type_construction.rb:2406-2408`). Declared gvar
    /// type is immutable here, mirroring the ivar/cvar paradigm.
    /// Bundles `check_deprecated_global_ref` (gvar-specific side
    /// effect the ivar/cvar siblings don't have) alongside the RHS
    /// check so `check_node`'s dispatch doesn't need to duplicate it.
    /// `$x = expr` in value position. Mirrors Steep
    /// `type_construction.rb` `gvasgn` (`:2433-2445`, helper at
    /// `:2842-2858`) — globals are namespace-free, so no ancestor walk
    /// is needed. `check_var_write_rhs` returns the RHS type
    /// unconditionally on a mismatch too, matching `gvasgn`'s helper
    /// (unlike `cvasgn`, see `check_class_variable_write`).
    pub(super) fn check_global_variable_write<'pr>(
        &mut self,
        node: &GlobalVariableWriteNode<'pr>,
    ) -> Ty {
        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let declared = self.env.lookup_global(&name_str);
        let rhs_ty = self.check_var_write_rhs(
            &node.value(),
            &name_str,
            declared,
            declared,
            |name| DiagnosticKind::UnknownGlobalVariable { name },
            node.location().start_offset(),
        );
        let name_loc = node.name_loc();
        self.check_deprecated_global_ref(&name_str, name_loc.start_offset(), name_loc.end_offset());
        cap_write_value_bottom(rhs_ty)
    }

    /// `Const = expr` in value position (`(Const = expr).method`,
    /// method body whose last expression is a plain casgn). Mirrors
    /// Steep `type_construction.rb` `:casgn` (`:1650-1667`): resolve
    /// the constant's own declared type and hint the RHS synthesis with
    /// it (bidirectional inference, e.g. an empty-array-literal RHS),
    /// then return the RHS's live-computed value type — replaces the
    /// old default-walker RHS descent (`self.visit(node)`, no value)
    /// with `check_node` (mirrors the ivar/cvar/gvar plumbing above).
    ///
    /// Unlike Steep's `casgn`, this does **not** yet subtype-check the
    /// RHS against the declared type / emit `IncompatibleAssignment` on
    /// a mismatch — the `visit_constant_or_write_node` /
    /// `visit_constant_and_write_node` doc comment already calls out
    /// that declared-type subtype gate as a separate todo
    /// (statement-position side effect, out of scope for the
    /// value-type plumbing this migration covers).
    ///
    /// `Const = Class.new(Parent?) { ... }` / `Struct.new` /
    /// `Data.define` stays untyped — that branch walks the block body
    /// directly instead of synthesizing the RHS via `check_node`, so
    /// there is no live-computed RHS value to return (unchanged scope
    /// from before this migration; the `infer_type` oracle arm added
    /// alongside this function covers the fallback re-walk so that
    /// branch doesn't fall to the catch-all's `NotImplementedYet`).
    pub(super) fn check_constant_write<'pr>(&mut self, node: &ConstantWriteNode<'pr>) -> Ty {
        self.ctx.invalidate_const_pure_calls();
        let name_str = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let position = self
            .byte_range_to_location(node.name_loc().start_offset(), node.name_loc().end_offset());
        // `class << self; S = 1` defines `S` on the singleton class
        // (Ruby cref), which no RBS declaration can spell — so it is
        // unknown by construction, with nothing searched and no extract
        // record (no absolute symbol exists for it, same as `||=`).
        // Reads inside `class << self` are untouched: their cref chain
        // continues to the enclosing class, so a declared `Foo::S`
        // still resolves. Path writes (`Foo::T = 1`) name their target
        // and go through `check_constant_path_write` instead.
        if self.ctx.in_singleton_class() {
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownConstant {
                    name: name_str.clone(),
                    path: name_str,
                    kind: ConstantKind::Constant,
                    did_you_mean: Vec::new(),
                    searched_namespaces: Vec::new(),
                    candidate_scope: CandidateScope::None,
                },
            });
            let ty = self.check_node(&node.value(), None);
            return cap_write_value_bottom(ty);
        }
        // Extract bookkeeping first, before the `Class.new do` fast
        // path below returns early, so that route records once too.
        self.record_extract_constant_write(
            node.location().start_offset(),
            node.location().end_offset(),
            Some(name_str.clone()),
        );
        // A constant write defines a new binding in the *current* cref
        // scope; the read-side resolver walks parent scopes and would
        // treat a same-named constant in an outer namespace as "already
        // declared". Restrict the declaration check to the innermost
        // scope (or `::Object` at top-level) so writes get diagnosed
        // even when a parent namespace happens to declare the name.
        let names = self.env.names();
        let sym = names.intern_symbol(&name_str);
        let scope = self.ctx.cref_stack().last();
        let resolved = self.env.resolve_constant_in_namespace(scope, sym);
        if resolved.is_none() {
            let candidate_scope = self.candidate_scope_in_namespace(scope);
            let searched_namespaces = self.searched_namespaces_in_namespace(scope);
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownConstant {
                    name: name_str.clone(),
                    path: name_str.clone(),
                    kind: ConstantKind::Constant,
                    did_you_mean: Vec::new(),
                    searched_namespaces,
                    candidate_scope,
                },
            });
        }

        // `Const = Class.new(Parent?) do ... end` / `Const = Struct.new(...) do ... end`
        // / `Const = Data.define(...) do ... end` — when the LHS is
        // declared in RBS as a class, walk the block body as that
        // class's body so `self` resolves to `singleton(::Const)` and
        // `def m` registers as an instance method on `::Const`.
        // Mirrors Ruby's `class_eval` semantics on the block. Falls
        // back to the default walker when the LHS is undeclared /
        // non-class, the RHS does not match a recognized class
        // construction pattern with a block, or any other mismatch —
        // those routes keep the legacy "block self stays as the
        // singleton receiver" behavior that anonymous calls rely on.
        if let Some(target) = resolved.as_ref().and_then(|rc| match &rc.origin {
            ConstantOrigin::Module { target } => Some(*target),
            ConstantOrigin::Constant => None,
        }) && self.env.declared_kind_by_type_name(&target) == Some(DeclKindLocal::Class)
            && let Some(unwrapped) = unwrap_single_cast_call(&node.value())
            && (is_class_new_named_lhs_pattern(&unwrapped.call)
                || is_data_struct_named_lhs_pattern(&unwrapped.call))
        {
            // The default walker's cast-form path (`visit_local_variable_write_node`
            // line 400-421) resolves the trailing `#: T` on the inner
            // `_ =` line via `lookup_trailing_assertion`. Returning
            // from the fast path bypasses that walker, so surface the
            // same assertion side effects (`UnknownTypeName` /
            // `AnnotationSyntaxError`) here. The outer
            // `ConstantWriteNode` is excluded from
            // `is_statement_assertion_eligible`, so no other site would.
            if let Some(cast) = &unwrapped.cast {
                self.lookup_trailing_assertion(
                    cast.location().start_offset(),
                    cast.location().end_offset(),
                );
            }
            self.walk_class_construction_block_body(&unwrapped.call, &name_str);
            return Ty::UNTYPED;
        }

        let declared = resolved.map(|rc| rc.ty);
        let ty = self.check_node(&node.value(), declared);
        cap_write_value_bottom(ty)
    }

    /// `Const::Path = expr` in value position — the constant-path
    /// counterpart of `check_constant_write`. No prior implementation
    /// existed for this kind (`grep -rn "constant_path_write"` returned
    /// zero hits in visitor.rs/inference.rs before this migration): the
    /// default Visit walker silently descended into the RHS with no
    /// existence check and no live value.
    ///
    /// Reuses the same silent resolver the read side
    /// (`visit_constant_path_node`) uses: Prism's `ConstantPathWriteNode`
    /// carries the LHS as a `ConstantPathNode` via `.target()`, so
    /// `resolve_constant_path_outcome` walks it exactly like a read.
    /// `emit_constant_path_outcome` surfaces `UnknownConstant` on a
    /// miss/intermediate-value, mirroring the existence check
    /// `check_constant_write` performs for the bare-constant sibling
    /// (`leaf_kind = Some(ConstantKind::Constant)`, matching a plain
    /// write's `ConstantKind::Constant`).
    ///
    /// Like `check_constant_write`, this does **not** yet subtype-check
    /// the RHS against the resolved declared type — the declared-type
    /// subtype gate is the same separate todo noted there.
    pub(super) fn check_constant_path_write<'pr>(
        &mut self,
        node: &ConstantPathWriteNode<'pr>,
    ) -> Ty {
        let target = node.target();
        let outcome = self.resolve_constant_path_outcome(&target);
        // Lexical spelling of the LHS; `None` on a dynamic parent, which
        // is exactly the `Malformed` case below.
        self.record_extract_constant_write(
            node.location().start_offset(),
            node.location().end_offset(),
            static_constant_path_string(&target),
        );
        let ty = if matches!(outcome, ConstantPathOutcome::Malformed) {
            // Dynamic parent (`obj.foo::BAR = v`) — `target` isn't a
            // constant path at all, so there's no existence check to
            // run, but the parent sub-expression may still have its
            // own diagnostics worth firing (`obj.foo`'s `NoMethod`).
            // Mirrors `visit_constant_path_node`'s own `Malformed`
            // fallback: default-walk `target` directly instead of
            // silently dropping it (this kind had no override at all
            // before this migration, so the generic default walker
            // used to cover this path unconditionally).
            visit_constant_path_node(self, &target);
            self.check_node(&node.value(), None)
        } else {
            self.emit_constant_path_outcome(&outcome, Some(ConstantKind::Constant));
            let declared_ty = outcome.resolved_ty();
            let hint = (!declared_ty.is_untyped()).then_some(declared_ty);
            self.check_node(&node.value(), hint)
        };
        cap_write_value_bottom(ty)
    }

    pub(super) fn check_global_variable_or_write<'pr>(
        &mut self,
        node: &GlobalVariableOrWriteNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        let ty = self.check_compound_gvar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            hint,
            node.location().start_offset(),
        );
        let name_str = String::from_utf8_lossy(node.name().as_slice()).into_owned();
        let name_loc = node.name_loc();
        self.check_deprecated_global_ref(&name_str, name_loc.start_offset(), name_loc.end_offset());
        ty
    }

    /// See `check_global_variable_or_write`; `&&=` shares the same
    /// `or_asgn`/`and_asgn` dispatch in Steep.
    pub(super) fn check_global_variable_and_write<'pr>(
        &mut self,
        node: &GlobalVariableAndWriteNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        let ty = self.check_compound_gvar_write_rhs(
            node.name().as_slice(),
            &node.value(),
            hint,
            node.location().start_offset(),
        );
        let name_str = String::from_utf8_lossy(node.name().as_slice()).into_owned();
        let name_loc = node.name_loc();
        self.check_deprecated_global_ref(&name_str, name_loc.start_offset(), name_loc.end_offset());
        ty
    }

    /// `a, b = value`: value-position type is the RHS's array/tuple type
    /// (Steep `type_masgn`, `type_construction.rb:2228`). Phase 1 covers
    /// only the literal-array, no-splat, no-assertion, arity-matching
    /// shape (`a, b = [x, y]`) — the common case and the multi-write
    /// analogue of the sibling bare-literal rescue-tail fix. Every other
    /// shape (splat, trailing assertion, arity mismatch, non-literal RHS)
    /// keeps returning `UNTYPED`; extending them is Phase 2 scope.
    ///
    /// `hint` is the enclosing type hint (e.g. a declared method return
    /// type), threaded through so a literal RBS type in tuple position
    /// (`-> [1, String]`) resolves a per-position hint and skips the
    /// widen — mirrors `check_array_node`'s tuple-hint path, and matters
    /// because `widen_literal_to_base` (see its doc comment) only holds
    /// "literal types survive under a hint" when the hint actually
    /// reaches the element check.
    pub(super) fn check_multi_write_node<'pr>(
        &mut self,
        node: &MultiWriteNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        // Splat-shape classification for the LHS. Each variant maps
        // 1:1 to a path below, so missing arms are a compile error
        // instead of a comment to keep in sync.
        enum SplatShape<'a> {
            /// `rest` is `None` — no splat in the LHS.
            None,
            /// `*` with no captured target (anonymous splat).
            Anonymous,
            /// `*x` where `x` is a plain lvar target.
            Lvar(LocalVariableTargetNode<'a>),
            /// `*expr` where `expr` is a nested target / non-lvar
            /// (scope-outside; falls through to the UNTYPED fallback).
            Unsupported,
        }

        let value_node = node.value();
        let lefts_vec: Vec<Node<'pr>> = node.lefts().iter().collect();
        let rights_vec: Vec<Node<'pr>> = node.rights().iter().collect();
        let rest_node = node.rest();

        let lefts_lvars: Option<Vec<LocalVariableTargetNode<'pr>>> = lefts_vec
            .iter()
            .map(|n| n.as_local_variable_target_node())
            .collect();
        let rights_lvars: Option<Vec<LocalVariableTargetNode<'pr>>> = rights_vec
            .iter()
            .map(|n| n.as_local_variable_target_node())
            .collect();

        let splat_shape: SplatShape<'pr> = match &rest_node {
            None => SplatShape::None,
            Some(rn) => match rn.as_splat_node() {
                None => SplatShape::Unsupported,
                Some(splat) => match splat.expression() {
                    None => SplatShape::Anonymous,
                    Some(expr) => match expr.as_local_variable_target_node() {
                        Some(t) => SplatShape::Lvar(t),
                        None => SplatShape::Unsupported,
                    },
                },
            },
        };

        let (Some(lefts_lvars), Some(rights_lvars)) = (lefts_lvars, rights_lvars) else {
            self.multi_assign_fallback(&value_node, &lefts_vec, rest_node.as_ref(), &rights_vec);
            return Ty::UNTYPED;
        };

        // Anonymous and Lvar share the splat-distribution path; bridge
        // them through `Option<&LocalVariableTargetNode>` so the helpers
        // see a single shape (None = skip splat bind, Some = bind).
        let splat_target_opt: Option<Option<&LocalVariableTargetNode<'pr>>> = match &splat_shape {
            SplatShape::None | SplatShape::Unsupported => None,
            SplatShape::Anonymous => Some(None),
            SplatShape::Lvar(t) => Some(Some(t)),
        };

        if matches!(splat_shape, SplatShape::Unsupported) {
            self.multi_assign_fallback(&value_node, &lefts_vec, rest_node.as_ref(), &rights_vec);
            return Ty::UNTYPED;
        }
        let no_splat = matches!(splat_shape, SplatShape::None);
        let assertion_ty = self.lookup_trailing_assertion(
            node.location().start_offset(),
            node.location().end_offset(),
        );
        let assertion_tuple: Option<Vec<Ty>> = assertion_ty.and_then(|ty| {
            let ty = definition_builder::expand_alias(self.env, ty);
            if let Type::Tuple(elem_types) = self.env.types().resolve(ty) {
                Some(elem_types.to_vec())
            } else {
                None
            }
        });

        // ArrayNode literal RHS: keep the predecessor path. Literals
        // never carry a Union shape (the parser produces a single
        // ArrayNode regardless of element types), so they bypass the
        // 4-stage Steep `type_masgn` pipeline below.
        if let Some(array) = value_node.as_array_node() {
            let elements: Vec<Node<'pr>> = array.elements().iter().collect();
            let nested_splat = elements.iter().any(|e| e.as_splat_node().is_some());
            if let Some(asserted) = assertion_ty {
                let value_ty = self.check_node(&value_node, Some(asserted));
                let position = self.offset_to_location(node.location().start_offset());
                self.emit_false_assertion_if_incompatible(value_ty, asserted, position);
                if let Some(elem_types) = assertion_tuple.as_ref() {
                    let splat_for_distribute = splat_target_opt.unwrap_or(None);
                    self.distribute_tuple_then_wrap(
                        elem_types,
                        false,
                        &lefts_lvars,
                        splat_for_distribute,
                        &rights_lvars,
                    );
                    // Phase 1 scope excludes the trailing-assertion shape
                    // (`a, b = [x, y] #: [T, T]`) — bound above, not
                    // returned as a value.
                    return Ty::UNTYPED;
                }
            }

            match splat_shape {
                SplatShape::None => {
                    if !nested_splat && elements.len() == lefts_lvars.len() {
                        // Resolve the outer hint (e.g. a declared method
                        // return type) into a per-position tuple hint when
                        // possible, mirroring `check_array_node`'s
                        // tuple-hint path. Without this, a literal RBS
                        // type in tuple position (`-> [1, String]`) would
                        // get force-widened below and false-positive
                        // against the very hint it should satisfy.
                        let tuple_elem_hints: Option<Vec<Ty>> = if assertion_ty.is_none() {
                            hint.and_then(|h| self.array_tuple_hint(h, &elements))
                        } else {
                            None
                        };
                        let mut value_elem_types: Vec<Ty> = Vec::with_capacity(elements.len());
                        for (i, (target, element)) in
                            lefts_lvars.iter().zip(elements.iter()).enumerate()
                        {
                            let per_position_hint = tuple_elem_hints.as_ref().map(|hints| hints[i]);
                            let element_ty = if assertion_ty.is_some() {
                                self.infer_type(element, None)
                            } else {
                                self.check_node(element, per_position_hint)
                            };
                            self.bind_local_variable_target(target, element_ty);
                            // The bound lvar type stays narrow (unwidened);
                            // only the value position widens below. The
                            // "Steep parity" claim is about the *value*
                            // widen matching `type_masgn`'s no-hint
                            // literal-to-class behavior (mirrors the
                            // Anonymous/Lvar splat branch) — bind-side
                            // narrowing for a *plain literal* (`c = 1`)
                            // is a separate, pre-existing crema-vs-Steep
                            // gap unrelated to this change (Steep widens
                            // lvasgn targets too; crema doesn't, here or
                            // elsewhere). A resolved per-position hint
                            // skips the value widen (`widen_literal_to_
                            // base`'s own contract: literal types survive
                            // only under a hint).
                            value_elem_types.push(if per_position_hint.is_some() {
                                element_ty
                            } else {
                                self.widen_literal_to_base(element_ty)
                            });
                        }
                        // Phase 1 core case: no trailing assertion, so the
                        // value position is the live (rescue-join-free)
                        // tuple of the just-bound element types — the
                        // multi-write analogue of the bare-literal
                        // rescue-tail fix.
                        return if assertion_ty.is_none() {
                            self.env.types().intern(Type::Tuple(value_elem_types))
                        } else {
                            Ty::UNTYPED
                        };
                    }
                    if assertion_ty.is_some() {
                        self.multi_assign_fallback_targets(
                            &lefts_vec,
                            rest_node.as_ref(),
                            &rights_vec,
                        );
                        return Ty::UNTYPED;
                    }
                    self.multi_assign_fallback(
                        &value_node,
                        &lefts_vec,
                        rest_node.as_ref(),
                        &rights_vec,
                    );
                    return Ty::UNTYPED;
                }
                SplatShape::Anonymous | SplatShape::Lvar(_) => {
                    let splat_target =
                        splat_target_opt.expect("Anonymous / Lvar yields Some above");
                    if nested_splat {
                        // Nested splat in RHS literal is the
                        // mid_array_literal_splat_tuple_inline territory.
                        if assertion_ty.is_some() {
                            self.multi_assign_fallback_targets(
                                &lefts_vec,
                                rest_node.as_ref(),
                                &rights_vec,
                            );
                            return Ty::UNTYPED;
                        }
                        self.multi_assign_fallback(
                            &value_node,
                            &lefts_vec,
                            rest_node.as_ref(),
                            &rights_vec,
                        );
                        return Ty::UNTYPED;
                    }
                    // Widen literal elements to their base class (Steep
                    // parity for `expand_tuple` on a literal array RHS).
                    let elem_types: Vec<Ty> = elements
                        .iter()
                        .map(|el| {
                            let raw = if assertion_ty.is_some() {
                                self.infer_type(el, None)
                            } else {
                                self.check_node(el, None)
                            };
                            self.widen_literal_to_base(raw)
                        })
                        .collect();
                    self.distribute_tuple_then_wrap(
                        &elem_types,
                        false,
                        &lefts_lvars,
                        splat_target,
                        &rights_lvars,
                    );
                    // Phase 1 scope excludes the splat shape (`a, *rest =
                    // [x, y, z]`) — bound above, not returned as a value.
                    return Ty::UNTYPED;
                }
                SplatShape::Unsupported => unreachable!("handled above"),
            }
        }

        // `@ivar ||= [a, b]` / `@ivar &&= [a, b]`: Steep peeks through the
        // `or_asgn`/`and_asgn` wrapper to a literal array RHS for masgn
        // destructuring purposes, synthesizing a per-position Tuple hint
        // from the LHS shape and threading it down to the literal
        // (`hint_for_mlhs` → `synthesize(rhs, hint: hint)` in
        // `type_construction.rb`, crema-review spec-consistency finding
        // 2026-07-21, verified against a live `steep check`: a
        // heterogeneous literal `[1, "s"]` binds position 0 to `Integer`
        // and position 1 to `String`, not a merged `Integer | String`
        // union broadcast to both). Read this branch as the masgn
        // analogue of the bare-literal fast path below (`value_node.
        // as_array_node()`) — same per-position walk, same widening,
        // just sourced from the array nested one level inside a
        // compound-write wrapper instead of `value_node` itself. Walking
        // `checked_value_ty` back apart (an earlier version of this fix)
        // is wrong: `check_node`'s no-hint `ArrayNode` path folds every
        // element into one unioned `Array[T]`, so reading `T` back and
        // broadcasting it to every position collapses exactly the
        // distinction Steep preserves (adversarial finding, same date).
        // `assertion_ty.is_none()` keeps a trailing `#: T` on the whole
        // masgn (`a, b = (@x ||= [..]) #: T`) on the existing path below
        // instead of this one. Scoped to ivar only — cvar / gvar / lvar
        // share the same Steep behavior (verified) but are deliberately
        // left to a follow-up (one change per unit of work).
        struct CompoundIvarWriteLiteral<'x> {
            name_bytes: Vec<u8>,
            elements: Vec<Node<'x>>,
            location_offset: usize,
        }
        fn compound_ivar_write_literal_array<'x>(
            node: Node<'x>,
        ) -> Option<CompoundIvarWriteLiteral<'x>> {
            // `(@ivar ||= [a, b])` parses the parens as a `ParenthesesNode`
            // wrapping the `InstanceVariableOrWriteNode` — unwrap first
            // (`unwrap_top_level_parens` already exists for this exact
            // shape elsewhere in this file).
            let node = unwrap_top_level_parens(node);
            let (name_bytes, value, location_offset) =
                if let Some(w) = node.as_instance_variable_or_write_node() {
                    (
                        w.name().as_slice().to_vec(),
                        w.value(),
                        w.location().start_offset(),
                    )
                } else if let Some(w) = node.as_instance_variable_and_write_node() {
                    (
                        w.name().as_slice().to_vec(),
                        w.value(),
                        w.location().start_offset(),
                    )
                } else {
                    return None;
                };
            let array = value.as_array_node()?;
            Some(CompoundIvarWriteLiteral {
                name_bytes,
                elements: array.elements().iter().collect(),
                location_offset,
            })
        }
        if assertion_ty.is_none()
            && no_splat
            && let Some(literal) = compound_ivar_write_literal_array(node.value())
            && literal.elements.len() == lefts_lvars.len()
            && !literal.elements.iter().any(|e| e.as_splat_node().is_some())
        {
            // Walk each element exactly once (mirrors the bare-literal
            // fast path's no-hint loop below) — the LHS lvars carry no
            // declared type to hint with, so every position widens like
            // an un-hinted literal does.
            let value_elem_types: Vec<Ty> = literal
                .elements
                .iter()
                .map(|el| {
                    let ty = self.check_node(el, None);
                    self.widen_literal_to_base(ty)
                })
                .collect();
            for (target, ty) in lefts_lvars.iter().zip(value_elem_types.iter()) {
                self.bind_local_variable_target(target, *ty);
            }
            // Diagnostics mirror `check_var_write_rhs`
            // (IncompatibleAssignment / UnknownInstanceVariable), checked
            // against the *same* Tuple this branch returns in value
            // position. Steep gets that Tuple for free: `type_masgn`
            // hands `hint_for_mlhs`'s per-position hint to the literal
            // (`multiple_assignment.rb:171-191` →
            // `type_construction.rb:2403-2405`), so `ivasgn` compares a
            // Tuple, not an `Array[T]`. Folding back to `Array[union]`
            // here (as this branch did until 2026-07-26) rejects exactly
            // the `keys, vals = (@x ||= [a, b])` idiom Steep accepts —
            // `rbs/lib/rbs/buffer.rb:94` was the live false positive.
            // The `&&=` mismatch case pins that Steep really reports
            // `[::Integer, ::Integer]` and is not merely silent.
            let name_str = String::from_utf8_lossy(&literal.name_bytes).into_owned();
            let var_sym = self.env.names().intern_symbol(&name_str);
            let declared = resolve_ivar_at_self(self, var_sym);
            let rhs_ty = self.env.types().intern(Type::Tuple(value_elem_types));
            let position = self.offset_to_location(literal.location_offset);
            match declared {
                Some(lhs) => {
                    if !self.subtyper().check(rhs_ty, lhs) {
                        self.push_diagnostic(Diagnostic {
                            scope: None,
                            location: position,
                            kind: DiagnosticKind::IncompatibleAssignment {
                                lhs_type: self.display_type(lhs),
                                rhs_type: self.display_type(rhs_ty),
                            },
                        });
                    }
                }
                None => {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: position,
                        kind: DiagnosticKind::UnknownInstanceVariable { name: name_str },
                    });
                }
            }
            return rhs_ty;
        }

        // Non-literal RHS: drive side-effects once, then run Steep's
        // 4-stage `type_masgn` (partition → truthy unify → per-position
        // normalize on a pure-Tuple union → expand + optional wrap).
        let checked_value_ty = self.check_node(&value_node, assertion_ty);
        if let Some(asserted) = assertion_ty {
            let position = self.offset_to_location(node.location().start_offset());
            self.emit_false_assertion_if_incompatible(checked_value_ty, asserted, position);
        }

        let value_ty = if let Some(elem_types) = assertion_tuple.as_ref() {
            self.env.types().intern(Type::Tuple(elem_types.clone()))
        } else if assertion_ty.is_some() {
            self.infer_type(&value_node, None)
        } else {
            checked_value_ty
        };
        let partition_ty = definition_builder::expand_alias(self.env, value_ty);
        let optional = partition_falsy(partition_ty, self.env.types()).is_some();
        let splat_for_distribute = splat_target_opt.unwrap_or(None);
        let truthy = partition_truthy(partition_ty, self.env.types());
        let truthy = truthy.and_then(|ty| {
            self.union_of_tuple_to_tuple_of_union(ty)
                .map(|per_position| self.env.types().intern(Type::Tuple(per_position)))
                .or(Some(ty))
        });
        // Steep types the whole masgn expression as `truthy_rhs_type`
        // (`type_construction.rb:2955`, `constr.add_typing(node, type:
        // truthy_rhs_type)`) unconditionally — the to_ary/to_a
        // conversion, the scalar 1-tuple wrap, and even the
        // MultipleAssignmentConversion error path all bind the *lhs*
        // from a converted/wrapped shape, but the *expression value* is
        // always the pre-conversion truthy type, never re-wrapped with
        // `optional` (ask-steep verified: to_ary success, to_ary
        // returning a non-expandable type, and a nilable receiver all
        // yield the pre-conversion type). A trailing assertion
        // (`a, b = expr #: T`) keeps the Phase 1 scope-out (bound above,
        // not returned as a value). `cap_write_value_bottom` guards a
        // statically-all-falsy RHS (`x, y = nil`): the assignment always
        // completes, so a raw `Ty::BOTTOM` here would trip
        // `check_statements_with_hint`'s divergence cutoff and silently
        // skip live statements after it (crema-review adversarial
        // finding, 2026-07-18 — reproduced via a mid-body `x, y = nil`
        // swallowing a later `NoMethod`).
        let value_position = |ty: Ty| -> Ty {
            if assertion_ty.is_none() {
                cap_write_value_bottom(ty)
            } else {
                Ty::UNTYPED
            }
        };
        if let Some(truthy) = truthy
            && !self.multi_assign_truthy_is_expandable(truthy)
        {
            // Steep `try_convert(truthy_rhs_type, :to_ary)`
            // (`type_construction.rb:2926`). Resolve `to_ary` on the
            // non-Tuple / non-Array[T] receiver and, when the return
            // type is itself expandable (Tuple / Array[T]), route
            // through the same `type_masgn_type` shape dispatch used
            // by native tuple / array RHS. `to_a` fallback and Union
            // receivers are deferred (see the todo scope note).
            if let Some(ary_ty) = self.try_convert_to_ary_return_type(truthy) {
                let normalized = definition_builder::expand_alias(self.env, ary_ty);
                let normalized = self
                    .union_of_tuple_to_tuple_of_union(normalized)
                    .map(|per_position| self.env.types().intern(Type::Tuple(per_position)))
                    .unwrap_or(normalized);
                if self.multi_assign_truthy_is_expandable(normalized) {
                    self.multi_assign_distribute_with_optional(
                        normalized,
                        optional,
                        &lefts_lvars,
                        splat_for_distribute,
                        &rights_lvars,
                    );
                    return value_position(truthy);
                }
                // Steep's `try_convert(:to_ary)` succeeded but returned
                // a non-expandable type — `rb_check_array_type` rejects
                // this at runtime with TypeError. Emit
                // MultipleAssignmentConversion (Steep parity with
                // `Ruby::MultipleAssignmentConversionError` at
                // `type_construction.rb:2930-2952`) and bind every lhs
                // to UNTYPED so subsequent uses don't cascade a
                // spurious `NilClass` NoMethod. Steep's short-circuit
                // (`try_convert(:to_ary) || try_convert(:to_a)`) also
                // means we do NOT advance to `to_a` here.
                //
                // `original_type` uses the pre-partition RHS type
                // (`partition_ty`) so a nilable receiver still shows
                // the `nil` variant, matching Steep's `rhs_type` at
                // `type_construction.rb:2932`. Location anchors on
                // the RHS node (Steep passes `node: rhs`).
                let position = self.offset_to_location(value_node.location().start_offset());
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: DiagnosticKind::MultipleAssignmentConversion {
                        original_type: self.display_type(partition_ty),
                        returned_type: self.display_type(ary_ty),
                    },
                });
                for target in &lefts_lvars {
                    self.bind_local_variable_target(target, Ty::UNTYPED);
                }
                if let Some(splat_target) = splat_for_distribute {
                    self.bind_local_variable_target(splat_target, Ty::UNTYPED);
                }
                for target in &rights_lvars {
                    self.bind_local_variable_target(target, Ty::UNTYPED);
                }
                return value_position(truthy);
            } else if let Some(ary_ty) = self.try_convert_to_a(truthy) {
                let normalized = definition_builder::expand_alias(self.env, ary_ty);
                let normalized = self
                    .union_of_tuple_to_tuple_of_union(normalized)
                    .map(|per_position| self.env.types().intern(Type::Tuple(per_position)))
                    .unwrap_or(normalized);
                if self.multi_assign_truthy_is_expandable(normalized) {
                    self.multi_assign_distribute_with_optional(
                        normalized,
                        optional,
                        &lefts_lvars,
                        splat_for_distribute,
                        &rights_lvars,
                    );
                    return value_position(truthy);
                }
            }
            if no_splat && self.multi_assign_truthy_is_scalar(truthy) {
                self.bind_scalar_multi_assign_rhs(value_ty, &lefts_lvars, &rights_lvars);
            } else {
                self.multi_assign_distribute_with_optional(
                    truthy,
                    optional,
                    &lefts_lvars,
                    splat_for_distribute,
                    &rights_lvars,
                );
            }
            return value_position(truthy);
        }
        if truthy.is_none() && no_splat && self.multi_assign_truthy_is_scalar(value_ty) {
            self.bind_scalar_multi_assign_rhs(value_ty, &lefts_lvars, &rights_lvars);
            return value_position(Ty::BOTTOM);
        }
        let distribute_ty = truthy.unwrap_or_else(|| self.env.types().intern(Type::Tuple(vec![])));
        self.multi_assign_distribute_with_optional(
            distribute_ty,
            optional,
            &lefts_lvars,
            splat_for_distribute,
            &rights_lvars,
        );
        // `truthy` (not `distribute_ty`) drives the value: Steep's
        // `truthy_rhs_type` is `bot` when the RHS is statically all-falsy
        // (empty union), while the empty-Tuple fallback above is purely
        // for the lhs distribute shape.
        value_position(truthy.unwrap_or(Ty::BOTTOM))
    }

    /// `x += rhs` is sugar for `x = x.+(rhs)` (Steep
    /// `type_construction.rb` `:op_asgn`): the lvar is rebound to the
    /// operator method's **return type**, not the RHS type. Paradigm
    /// differs from `||=` / `&&=` (RHS-type overwrite, handled by
    /// `rebind_compound_lvar`), so the helper is not shared. Returns the
    /// bound type (the value-position type of the whole expression) so
    /// `check_node` doesn't have to re-run the dispatch.
    /// `x = expr` in value position (`(x = expr).method`, method body
    /// whose last expression is a plain lvasgn). Side effects (env
    /// binding, `#: T` assertion gate) are unchanged from before this
    /// kind was split out of `check_node`'s UNTYPED-discard cluster —
    /// only the return value is new.
    ///
    /// `hint` is the *ambient* hint from the enclosing expression (e.g.
    /// the method's declared return type when this write is the body's
    /// last statement) — forwarded into the RHS synthesis when no
    /// trailing `#: T` assertion claims the hint instead. Steep parity:
    /// `:lvasgn` (`type_construction.rb:772-784`) prefers an
    /// `enforced_type` (crema's assertion) over the ambient hint but
    /// falls back to the ambient hint when no assertion is present — an
    /// lvar has no RBS declaration of its own to hint with instead,
    /// unlike ivar/cvar/gvar/constant (`check_var_write_rhs` /
    /// `check_constant_write` hint from their own declaration). Mirrors
    /// the same fix already applied to `rebind_compound_lvar` (F2, `x
    /// ||= v` / `x &&= v`). The Visit-trait wrapper
    /// (`visit_local_variable_write_node`) has no ambient hint to give
    /// and passes `None`, matching `rebind_compound_lvar`'s
    /// statement-position callers.
    pub(super) fn check_local_variable_write<'pr>(
        &mut self,
        node: &LocalVariableWriteNode<'pr>,
        hint: Option<Ty>,
    ) -> Ty {
        // Steep parity (`SPECIAL_LVAR_NAMES = Set[:_, :__any__, :__skip__]`,
        // `lib/steep/type_construction.rb:760-770`): writes to these names
        // produce untyped and never bind into the env, enabling the
        // `(_ = nil) #: T` cast idiom. `__skip__` additionally skips RHS
        // synthesis so the RHS can hold deliberately ill-typed expressions
        // without surfacing diagnostics. The trailing `#: T` is still
        // resolved so an unknown type name on the assertion surfaces
        // (`_ = nil #: MissingType` ⇒ UnknownTypeName), matching the
        // assertion-side validation Steep performs in its `:assertion` arm
        // before delegating to lvasgn.
        let name_bytes = node.name().as_slice();
        if is_special_lvar_name(name_bytes) {
            self.lookup_trailing_assertion(
                node.location().start_offset(),
                node.location().end_offset(),
            );
            if name_bytes != b"__skip__" {
                self.check_node(&node.value(), Some(Ty::UNTYPED));
            }
            return Ty::UNTYPED;
        }

        // Stage 1: an inline `#: T` on the assignment line, when present
        // and `--inline=true`, becomes a bidirectional hint for the RHS
        // and overrides the bound variable type (Steep parity, see
        // `lib/steep/type_construction.rb` `when :assertion`). Without a
        // trailing assertion, the ambient `hint` param flows through
        // instead (see this function's doc comment).
        //
        // `check_node` is the side-effecting expression evaluator that
        // threads the hint through the RHS as an argument and fires
        // block / argument diagnostics along the way — no Visit-trait
        // default walker is invoked here, and no `pending_call_hint`
        // global slot is set. The hint flows structurally: Array
        // literal unwraps it for elements, `begin ... end` forwards it
        // to the last statement, `cond ? a : b` forwards it to both
        // branches.
        let assertion_ty = self.lookup_trailing_assertion(
            node.location().start_offset(),
            node.location().end_offset(),
        );
        let name_str = String::from_utf8_lossy(name_bytes);
        let name = self.checker_names().intern(&name_str);
        let depth = node.depth();
        // A pinned outer lvar lends its pinned type to the RHS as the
        // hint (Steep `type_construction.rb` lvasgn: `hint =
        // enforced_type` when there is no hint, or when the enforced
        // type is the more specific of the two). This is what types
        // `ctx = [ctx, name]` inside a block as the tuple the pinned
        // alias expects instead of `Array[...]`.
        let pinned = self.ctx.pinned_local_variable_type(name, depth);
        let hint = match (hint, pinned) {
            (None, Some(pinned)) => Some(pinned),
            (Some(h), Some(pinned)) if self.subtyper().check(pinned, h) => Some(pinned),
            (hint, _) => hint,
        };
        // When the lvasgn has claimed a trailing `#: T`, flip the
        // parens-routed suppression flag so a Parens (or any node that
        // recurses through `check_statements_with_hint` with the
        // forwarded hint) won't re-emit a `FalseAssertion` for the same
        // trailing comment. The flag is cleared on entry to the
        // outermost `check_statements_with_hint`, so nested
        // StatementsNodes inside the last stmt still apply their gate
        // (see `test_lvasgn_parens_nested_begin_inner_assertion_
        // independent`). No trailing (`assertion_ty.is_none()`) leaves
        // the flag untouched — the inner gate is the only route in
        // that case and must keep working. The SPECIAL_LVAR early
        // return above (`_`, `__any__`, `__skip__`) takes a different
        // code path and never reaches this set; pinned by
        // `test_special_lvar_cast_idiom_unaffected_by_parens_routed_flag`.
        let value_type = if assertion_ty.is_some() {
            let saved = std::mem::replace(&mut self.suppress_parens_routed_last_assertion, true);
            let ty = self.check_node(&node.value(), assertion_ty);
            self.suppress_parens_routed_last_assertion = saved;
            ty
        } else {
            self.check_node(&node.value(), hint)
        };

        // Stage 2: double-direction subtype gate. Either direction matches
        // (widening or narrowing) is accepted; only mutual incompatibility
        // surfaces as `FalseAssertion`.
        if let Some(asserted) = assertion_ty {
            let position = self.offset_to_location(node.location().start_offset());
            self.emit_false_assertion_if_incompatible(value_type, asserted, position);
        }

        // Stage 3: assertion type wins over the natural inferred type when
        // present, even when Stage 2 emitted a FalseAssertion — matches
        // Steep `constr.add_typing(node, type: type)`.
        let bound_ty = self.freeze_lvar_bound_type(assertion_ty.unwrap_or(value_type), depth);
        // Closure-crossing write against a pinned outer lvar: the pin
        // check runs on the bound (asserted-or-inferred) type, so a
        // `#: T` write inside a block reports IncompatibleAssignment
        // against the pin and never a second FalseAssertion for the
        // same mismatch (Steep, measured 2026-09-11).
        let byte_range = (node.location().start_offset(), node.location().end_offset());
        if let Some(pinned) = self.bind_pinned_lvar_write(name, bound_ty, depth, byte_range) {
            return pinned;
        }
        if value_type == Ty::BOTTOM && bound_ty == Ty::BOTTOM {
            self.ctx
                .set_local_variable_at_depth_from_bot_rhs(name, bound_ty, depth);
            // Return UNTYPED, not `bound_ty` (BOTTOM), for this branch
            // specifically: `check_statements_with_hint`'s divergence
            // cutoff (`if last_ty == Ty::BOTTOM { break }`) treats a
            // BOTTOM statement value as unreachable-code-follows (return
            // / raise / an if-all-arms-diverge). A bot-*valued* RHS
            // bound to an lvar (`y = self.boom` where `boom: () -> bot`)
            // is a different thing — the assignment itself completes
            // normally and later statements are still live code (`y.foo`
            // must still be walked and diagnosed, pinned by
            // `test_bottom_lvar_receiver_reports_no_method`). Before
            // this kind was split out of the UNTYPED-discard cluster,
            // `check_node` always returned `Ty::UNTYPED` here regardless
            // of the RHS, so the cutoff never saw this case; UNTYPED
            // preserves that pre-migration behavior for this one branch
            // while the non-bot branch below returns the real value.
            Ty::UNTYPED
        } else {
            self.ctx.set_local_variable_at_depth(name, bound_ty, depth);
            bound_ty
        }
    }

    pub(super) fn check_local_variable_operator_write<'pr>(
        &mut self,
        node: &LocalVariableOperatorWriteNode<'pr>,
    ) -> Ty {
        let name_bytes = node.name().as_slice();
        let value_node = node.value();

        // SPECIAL_LVAR (`_` / `__any__` / `__skip__`): the binding is
        // never overwritten — the cast idiom (`_ = nil #: T`) relies on
        // it. `__skip__` additionally suppresses RHS synthesis so the
        // RHS can hold deliberately ill-typed expressions, matching the
        // plain-write path in `visit_local_variable_write_node`. Steep
        // is silent here for the same reason.
        if is_special_lvar_name(name_bytes) {
            if name_bytes != b"__skip__" {
                self.check_node(&value_node, None);
            }
            return Ty::UNTYPED;
        }

        // Non-special path: synthesize the RHS for its side-effect
        // diagnostics, then dispatch the synthetic operator call.
        let rhs_type = self.check_node(&value_node, None);

        let name_str = String::from_utf8_lossy(name_bytes);
        let name = self.checker_names().intern(&name_str);
        let depth = node.depth();

        // Uninitialized lvars read as untyped (Steep parity for
        // `y += rhs` with no prior assignment). The synthetic dispatch
        // would short-circuit to NoMethod-free silent on untyped
        // receivers anyway; collapsing to a direct UNTYPED rebind keeps
        // the binding consistent with the read.
        let receiver_ty = self
            .lookup_local_variable_for_read(name)
            .unwrap_or(Ty::UNTYPED);
        if receiver_ty.is_untyped() {
            self.ctx
                .set_local_variable_at_depth(name, Ty::UNTYPED, depth);
            return Ty::UNTYPED;
        }

        // `node.binary_operator()` is the operator without `=` (Prism
        // gives `:+` for `+=`, `:-` for `-=`, etc.), so it doubles as
        // the dispatch method name. Position is anchored at the operator
        // token (`binary_operator_loc`) so NoMethod points at the
        // operator, not the value or the lvar.
        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        let name_location = self
            .byte_range_to_location(op_loc.start_offset(), op_loc.start_offset() + op_name.len());

        let bound_ty = self.check_synthetic_method_call_with_single_arg_node(
            receiver_ty,
            &op_name,
            rhs_type,
            &value_node,
            name_location,
        );
        let bound_ty = self.freeze_lvar_bound_type(bound_ty, depth);
        let byte_range = (node.location().start_offset(), node.location().end_offset());
        if let Some(pinned) = self.bind_pinned_lvar_write(name, bound_ty, depth, byte_range) {
            return pinned;
        }
        self.ctx.set_local_variable_at_depth(name, bound_ty, depth);
        bound_ty
    }

    /// `@x += y`: Steep desugars to `@x = @x.+(y)` (`:858-908`): the
    /// operator method is dispatched against the *declared* ivar type,
    /// and the return type is subtype-checked against the declared ivar
    /// type via `ivasgn`. Unlike
    /// `check_local_variable_operator_write`, the binding is not
    /// rebound — declared ivar type is immutable. Returns the operator
    /// dispatch's result type (the value-position type of the whole
    /// expression).
    /// `@x = expr` in value position (`(@x = expr).method`, method body
    /// whose last expression is a plain ivasgn). Mirrors Steep
    /// `type_construction.rb` `ivasgn` (`:2824-2840`): resolve the
    /// declared ivar type on the current `self` (walking the linearized
    /// ancestor chain and applying receiver-side subst), then either
    /// subtype-check the RHS against it or surface
    /// `UnknownInstanceVariable` — via `check_var_write_rhs`, shared
    /// with the plain cvar/gvar write paths. The `#: T` assertion on the
    /// assignment line does *not* override the declaration — that's a
    /// deliberate divergence from `LocalVariableWriteNode`, where `#: T`
    /// cast wins (Steep 2026-06-06 measurement).
    ///
    /// `check_var_write_rhs` returns the RHS type unconditionally,
    /// matching Steep's `ivasgn` helper (`add_typing(node, type:
    /// rhs_type)` regardless of subtype match/mismatch — unlike
    /// `cvasgn`, see `check_class_variable_write`).
    pub(super) fn check_instance_variable_write<'pr>(
        &mut self,
        node: &InstanceVariableWriteNode<'pr>,
    ) -> Ty {
        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_ivar_at_self(self, var_sym);
        let rhs_ty = self.check_var_write_rhs(
            &node.value(),
            &name_str,
            declared,
            declared,
            |name| DiagnosticKind::UnknownInstanceVariable { name },
            node.location().start_offset(),
        );
        cap_write_value_bottom(rhs_ty)
    }

    pub(super) fn check_instance_variable_operator_write<'pr>(
        &mut self,
        node: &InstanceVariableOperatorWriteNode<'pr>,
    ) -> Ty {
        let value_node = node.value();
        let rhs_type = self.check_node(&value_node, None);

        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_ivar_at_self(self, var_sym);

        let Some(declared_ty) = declared else {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownInstanceVariable { name: name_str },
            });
            return Ty::UNTYPED;
        };

        // Declared `@x: untyped` escapes operator dispatch — mirrors
        // the lvar paradigm where an untyped receiver short-circuits.
        if declared_ty.is_untyped() {
            return Ty::UNTYPED;
        }

        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        let op_location = self
            .byte_range_to_location(op_loc.start_offset(), op_loc.start_offset() + op_name.len());
        let result_ty =
            self.check_synthetic_method_call(declared_ty, &op_name, vec![rhs_type], op_location);

        // Guard against `IncompatibleAssignment` cascading on top of an
        // operator failure (NoMethod / PrivateMethodCall): when the
        // dispatch errored, `check_synthetic_method_call` returns
        // `Ty::UNTYPED`, which would trivially subtype the declared ivar
        // type and emit a spurious second diagnostic.
        if !result_ty.is_untyped() && !self.subtyper().check(result_ty, declared_ty) {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::IncompatibleAssignment {
                    lhs_type: self.display_type(declared_ty),
                    rhs_type: self.display_type(result_ty),
                },
            });
        }
        result_ty
    }

    /// `@@x += y`: Steep desugars to `@@x = @@x.+(y)` (`:858-908`).
    /// Mostly mirrors `check_instance_variable_operator_write`, but the
    /// `IncompatibleAssignment` branch does not: Steep's `:cvasgn` has no
    /// dedicated method (unlike `ivasgn`/`gvasgn`, `type_construction.rb
    /// :2824,:2842`) — it's inlined in the `synthesize` case statement
    /// (`:2562-2589`) and calls `fallback_to_any(node)` on a failed
    /// subtype check, so the *value* of a mismatched `@@x += y` is
    /// `untyped`, not the operator's return type. `ivasgn`/`gvasgn`
    /// return `rhs_type` unconditionally, mismatch or not — this
    /// asymmetry is Steep's own, not a crema simplification.
    /// `@@x = expr` in value position. Mirrors Steep
    /// `type_construction.rb` `cvasgn` (`:2562-2591`); see
    /// `check_instance_variable_write` for the shared
    /// `check_var_write_rhs` structure — the only difference is the
    /// lookup helper.
    ///
    /// Unlike `ivasgn`/`gvasgn`, Steep's `cvasgn` does **not**
    /// unconditionally return the RHS type on a declared-type mismatch
    /// — the mismatch branch calls `fallback_to_any` (`:2580`), i.e.
    /// the *value* of a mismatched `@@x = wrong_type` is `untyped`, not
    /// the RHS type. **The undeclared branch does too**
    /// (`type_construction.rb:2589-2591`: `var_type` nil also falls to
    /// `fallback_to_any(node)`, no diagnostic block) — an undeclared
    /// `@@x = v` is `untyped`, not the RHS type, same as the mismatch
    /// case. This is Steep's own asymmetry with `ivasgn`/`gvasgn`
    /// (verified 2026-07-17 against `type_construction.rb`), not a
    /// crema simplification — mirrors the same divergence already
    /// ported for `check_class_variable_operator_write`'s `let Some
    /// (declared_ty) = declared else { ...; return Ty::UNTYPED }`.
    /// `check_var_write_rhs` still emits `UnknownClassVariable` /
    /// `IncompatibleAssignment` (side effects unchanged); only the
    /// returned value is overridden here.
    pub(super) fn check_class_variable_write<'pr>(
        &mut self,
        node: &ClassVariableWriteNode<'pr>,
    ) -> Ty {
        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_cvar_at_self(self, var_sym);
        let rhs_ty = self.check_var_write_rhs(
            &node.value(),
            &name_str,
            declared,
            declared,
            |name| DiagnosticKind::UnknownClassVariable { name },
            node.location().start_offset(),
        );
        let ty = match declared {
            Some(lhs) if self.subtyper().check(rhs_ty, lhs) => rhs_ty,
            _ => Ty::UNTYPED,
        };
        cap_write_value_bottom(ty)
    }

    pub(super) fn check_class_variable_operator_write<'pr>(
        &mut self,
        node: &ClassVariableOperatorWriteNode<'pr>,
    ) -> Ty {
        let value_node = node.value();
        let rhs_type = self.check_node(&value_node, None);

        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let var_sym = self.env.names().intern_symbol(&name_str);
        let declared = resolve_cvar_at_self(self, var_sym);

        let Some(declared_ty) = declared else {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownClassVariable { name: name_str },
            });
            return Ty::UNTYPED;
        };

        // Declared `@@x: untyped` escapes operator dispatch — mirrors
        // the ivar/lvar paradigm.
        if declared_ty.is_untyped() {
            return Ty::UNTYPED;
        }

        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        let op_location = self
            .byte_range_to_location(op_loc.start_offset(), op_loc.start_offset() + op_name.len());
        let result_ty =
            self.check_synthetic_method_call(declared_ty, &op_name, vec![rhs_type], op_location);

        // Guard against `IncompatibleAssignment` cascading on top of an
        // operator failure (NoMethod / PrivateMethodCall): when the
        // dispatch errored, `check_synthetic_method_call` returns
        // `Ty::UNTYPED`, which would trivially subtype the declared cvar
        // type and emit a spurious second diagnostic.
        if !result_ty.is_untyped() && !self.subtyper().check(result_ty, declared_ty) {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::IncompatibleAssignment {
                    lhs_type: self.display_type(declared_ty),
                    rhs_type: self.display_type(result_ty),
                },
            });
            // Steep's `fallback_to_any` on this exact path (see doc
            // comment above) — the mismatched value doesn't propagate.
            return Ty::UNTYPED;
        }
        result_ty
    }

    /// `$x += y`: Steep desugars to `$x = $x.+(y)` (`:858-908`,
    /// `:890-893` for the `:gvasgn` LHS path). Mirrors
    /// `check_instance_variable_operator_write` /
    /// `check_class_variable_operator_write`.
    pub(super) fn check_global_variable_operator_write<'pr>(
        &mut self,
        node: &GlobalVariableOperatorWriteNode<'pr>,
    ) -> Ty {
        let value_node = node.value();
        let rhs_type = self.check_node(&value_node, None);

        let name_bytes = node.name().as_slice();
        let name_str = String::from_utf8_lossy(name_bytes).into_owned();
        let declared = self.env.lookup_global(&name_str);
        let name_loc = node.name_loc();
        self.check_deprecated_global_ref(&name_str, name_loc.start_offset(), name_loc.end_offset());

        let Some(declared_ty) = declared else {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::UnknownGlobalVariable { name: name_str },
            });
            return Ty::UNTYPED;
        };

        // Declared `$x: untyped` escapes operator dispatch — mirrors
        // the ivar/cvar/lvar paradigm.
        if declared_ty.is_untyped() {
            return Ty::UNTYPED;
        }

        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        let op_location = self
            .byte_range_to_location(op_loc.start_offset(), op_loc.start_offset() + op_name.len());
        let result_ty =
            self.check_synthetic_method_call(declared_ty, &op_name, vec![rhs_type], op_location);

        // Guard against `IncompatibleAssignment` cascading on top of an
        // operator failure (NoMethod / PrivateMethodCall): when the
        // dispatch errored, `check_synthetic_method_call` returns
        // `Ty::UNTYPED`, which would trivially subtype the declared gvar
        // type and emit a spurious second diagnostic.
        if !result_ty.is_untyped() && !self.subtyper().check(result_ty, declared_ty) {
            let position = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: position,
                kind: DiagnosticKind::IncompatibleAssignment {
                    lhs_type: self.display_type(declared_ty),
                    rhs_type: self.display_type(result_ty),
                },
            });
        }
        result_ty
    }

    /// `foo[idx] += value`: Steep desugars to `foo.[]=(idx,
    /// foo.[](idx).+(value))` (`type_construction.rb:858-908`). The
    /// three-stage dispatch ([], op, []=) is folded into one helper call
    /// by passing the operator name; the helper handles the op-failure
    /// cascade so `[]=` does not surface a second ArgumentTypeMismatch on
    /// top of an operator error. Returns the operator dispatch's result
    /// type (the value-position type of the whole expression) so callers
    /// that need it (`check_node`) don't have to re-run the dispatch.
    pub(super) fn check_index_operator_write<'pr>(
        &mut self,
        node: &IndexOperatorWriteNode<'pr>,
    ) -> Ty {
        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        self.check_index_compound_write(
            node.receiver(),
            node.arguments(),
            node.block().is_some(),
            IndexCompoundWriteOp::Operator(
                &op_name,
                self.byte_range_to_location(
                    op_loc.start_offset(),
                    op_loc.start_offset() + op_name.len(),
                ),
            ),
            &node.value(),
            self.byte_range_to_location(
                node.opening_loc().start_offset(),
                node.closing_loc().end_offset(),
            ),
            self.byte_range_to_location(
                op_loc.start_offset(),
                op_loc.start_offset() + op_name.len(),
            ),
        )
    }

    /// `foo[idx] ||= value`: value type is the truthy-narrowing union of
    /// the read type and the rhs (`or_write_value_ty`) — Steep parity
    /// (`type_construction.rb:2395-2428` desugars to `foo.[](idx) ||
    /// foo.[]=(idx, value)` and runs the ordinary `:or` narrowing).
    /// Mirrors `check_index_operator_write`'s split from its visitor
    /// callback so `check_node` can consume the live dispatch result
    /// without re-running it.
    pub(super) fn check_index_or_write<'pr>(&mut self, node: &IndexOrWriteNode<'pr>) -> Ty {
        self.check_index_compound_write(
            node.receiver(),
            node.arguments(),
            node.block().is_some(),
            IndexCompoundWriteOp::Or,
            &node.value(),
            self.byte_range_to_location(
                node.opening_loc().start_offset(),
                node.closing_loc().end_offset(),
            ),
            self.byte_range_to_location(
                node.operator_loc().start_offset(),
                node.operator_loc().end_offset(),
            ),
        )
    }

    /// `foo[idx] &&= value`: value type is the falsy-narrowing union of
    /// the rhs and the read type (`and_write_value_ty`). Shares the
    /// dispatch shape with `||=` — see `check_index_or_write`.
    pub(super) fn check_index_and_write<'pr>(&mut self, node: &IndexAndWriteNode<'pr>) -> Ty {
        self.check_index_compound_write(
            node.receiver(),
            node.arguments(),
            node.block().is_some(),
            IndexCompoundWriteOp::And,
            &node.value(),
            self.byte_range_to_location(
                node.opening_loc().start_offset(),
                node.closing_loc().end_offset(),
            ),
            self.byte_range_to_location(
                node.operator_loc().start_offset(),
                node.operator_loc().end_offset(),
            ),
        )
    }

    /// `foo.attr += value`: Steep desugars to `foo.attr=(foo.attr.+(value))`
    /// (`type_construction.rb:858-908` `:op_asgn` with a `:send` LHS) —
    /// the same three-stage dispatch as `check_index_operator_write`
    /// minus the index arguments. Returns the operator dispatch's result
    /// type (the value-position type of the whole expression).
    pub(super) fn check_call_operator_write<'pr>(
        &mut self,
        node: &CallOperatorWriteNode<'pr>,
    ) -> Ty {
        let op_bytes = node.binary_operator().as_slice();
        let op_name = String::from_utf8_lossy(op_bytes).to_string();
        let op_loc = node.binary_operator_loc();
        let op_location = self
            .byte_range_to_location(op_loc.start_offset(), op_loc.start_offset() + op_name.len());
        let read_location = self.call_write_read_location(node.message_loc(), node.location());
        self.check_call_compound_write(
            node.receiver(),
            node.is_safe_navigation(),
            node.read_name().as_slice(),
            node.write_name().as_slice(),
            IndexCompoundWriteOp::Operator(&op_name, op_location.clone()),
            &node.value(),
            read_location,
            op_location,
        )
    }

    /// `foo.attr ||= value`: reader/writer synthetic dispatch plus the
    /// truthy-narrowing value type (`or_write_value_ty`) — Steep parity
    /// (`type_construction.rb:2395-2428` `:or_asgn` with a `:send` LHS
    /// desugars to `foo.attr || foo.attr=(value)`). Mirrors
    /// `check_index_or_write`'s split from its visitor callback.
    pub(super) fn check_call_or_write<'pr>(&mut self, node: &CallOrWriteNode<'pr>) -> Ty {
        let read_location = self.call_write_read_location(node.message_loc(), node.location());
        let op_loc = node.operator_loc();
        self.check_call_compound_write(
            node.receiver(),
            node.is_safe_navigation(),
            node.read_name().as_slice(),
            node.write_name().as_slice(),
            IndexCompoundWriteOp::Or,
            &node.value(),
            read_location,
            self.byte_range_to_location(op_loc.start_offset(), op_loc.end_offset()),
        )
    }

    /// `foo.attr &&= value`: falsy-narrowing variant of
    /// `check_call_or_write` (`and_write_value_ty`).
    pub(super) fn check_call_and_write<'pr>(&mut self, node: &CallAndWriteNode<'pr>) -> Ty {
        let read_location = self.call_write_read_location(node.message_loc(), node.location());
        let op_loc = node.operator_loc();
        self.check_call_compound_write(
            node.receiver(),
            node.is_safe_navigation(),
            node.read_name().as_slice(),
            node.write_name().as_slice(),
            IndexCompoundWriteOp::And,
            &node.value(),
            read_location,
            self.byte_range_to_location(op_loc.start_offset(), op_loc.end_offset()),
        )
    }

    /// Diagnostic anchor for the reader dispatch of a call-attribute
    /// compound write: the message span when present, the whole node
    /// otherwise (message_loc is structurally always present for
    /// `Call*WriteNode` in Prism's grammar; the fallback only guards
    /// the Option shape of the generated accessor).
    fn call_write_read_location(
        &self,
        message_loc: Option<ruby_prism::Location<'_>>,
        node_loc: ruby_prism::Location<'_>,
    ) -> SourceLocation {
        match message_loc {
            Some(l) => self.byte_range_to_location(l.start_offset(), l.end_offset()),
            None => self.byte_range_to_location(node_loc.start_offset(), node_loc.end_offset()),
        }
    }

    /// Shared body for the three call-attribute compound writes
    /// (`foo.attr ||= v`, `&&= v`, `+= v`). Same dispatch shape as
    /// `check_index_compound_write` minus the index arguments: the
    /// reader takes no args and the writer takes exactly the written
    /// value, so the index-arg cascade guard has no counterpart here.
    /// The op-failure cascade guard is shared: a failed reader or
    /// operator dispatch propagates `UNTYPED` into the writer's value
    /// arg, which subtypes any param, so no spurious
    /// `ArgumentTypeMismatch` piles on top of the original failure
    /// (Steep parity: `foo.num += "str"` reports only the `+` error).
    ///
    /// Safe navigation (`a&.attr ||= v`) unwraps the receiver's nil
    /// before dispatch, mirroring `check_call`'s receiver handling. The
    /// nil-widening of the whole expression's value type is not modeled
    /// (treated as the plain-call variant — Steep unmeasured, noted in
    /// the todo).
    #[allow(clippy::too_many_arguments)]
    fn check_call_compound_write<'pr>(
        &mut self,
        receiver_node: Option<Node<'pr>>,
        is_safe_navigation: bool,
        read_name: &[u8],
        write_name: &[u8],
        op: IndexCompoundWriteOp<'_>,
        value: &Node<'pr>,
        read_location: SourceLocation,
        write_location: SourceLocation,
    ) -> Ty {
        let Some(recv) = receiver_node else {
            // Receiver-less `Call*WriteNode` is structurally unreachable
            // from Prism's Ruby grammar (a bare `x ||= v` parses as a
            // LocalVariable*WriteNode). Defensive descent mirrors
            // `check_index_compound_write`.
            self.check_node(value, None);
            return Ty::UNTYPED;
        };

        let recv_ty = self.check_node(&recv, None);
        let value_ty = self.check_node(value, None);

        if recv_ty.is_untyped() {
            return Ty::UNTYPED;
        }
        let recv_ty = if is_safe_navigation {
            self.unwrap_optional(recv_ty)
        } else {
            recv_ty
        };

        let read_name = String::from_utf8_lossy(read_name).to_string();
        let write_name = String::from_utf8_lossy(write_name).to_string();

        let read_ty = self.check_synthetic_method_call(recv_ty, &read_name, vec![], read_location);

        let write_arg_ty = match &op {
            IndexCompoundWriteOp::Operator(op_name, op_location) => {
                if read_ty.is_untyped() {
                    Ty::UNTYPED
                } else {
                    self.check_synthetic_method_call(
                        read_ty,
                        op_name,
                        vec![value_ty],
                        op_location.clone(),
                    )
                }
            }
            IndexCompoundWriteOp::Or | IndexCompoundWriteOp::And => value_ty,
        };

        // Writer dispatch always runs so `foo.undefined ||= 1` surfaces
        // both NoMethod diagnostics (reader and writer) — Steep parity.
        // The writer's own return type is discarded: Ruby's assignment
        // expression evaluates to the written value.
        let _ = self.check_synthetic_method_call(
            recv_ty,
            &write_name,
            vec![write_arg_ty],
            write_location,
        );

        if read_ty.is_untyped() {
            Ty::UNTYPED
        } else {
            match &op {
                IndexCompoundWriteOp::Operator(..) => write_arg_ty,
                // Widen the rhs literal before the narrowing union —
                // same Steep parity as the index siblings (see
                // `check_index_compound_write` step 4).
                IndexCompoundWriteOp::Or => {
                    let widened = self.widen_literal_to_base(value_ty);
                    or_write_value_ty(read_ty, widened, self.env.types())
                }
                IndexCompoundWriteOp::And => {
                    let widened = self.widen_literal_to_base(value_ty);
                    and_write_value_ty(read_ty, widened, self.env.types())
                }
            }
        }
    }

    /// Shared body for the three index compound writes (`foo[idx] ||= v`,
    /// `&&= v`, `+= v`). The dispatch sequence is:
    ///
    /// 1. synthesize the receiver, the index arguments, and the rhs value
    /// 2. if the receiver is untyped, escape (matches the general
    ///    method-call paradigm and the ivar `+=` arm)
    /// 3. dispatch `[]` for its read-side diagnostics (NoMethod /
    ///    PrivateMethodCall etc.). The return type feeds step 4 for all
    ///    three ops (`+=`'s operator dispatch, `||=`/`&&=`'s narrowing)
    /// 4. compute the value passed to `[]=`: `+=` dispatches the named
    ///    operator method on the read type with the rhs as its single
    ///    argument; `||=`/`&&=` compute the truthy/falsy-narrowing union
    ///    of the read type and the rhs (`or_write_value_ty` /
    ///    `and_write_value_ty`) — see those functions' doc comments for
    ///    why this isn't shared with `check_or_node`/`check_and_node`
    /// 5. dispatch `[]=` with `[idx_tys..., write_value_ty]`
    ///
    /// Cascade guard: when a prior dispatch returns `UNTYPED` (NoMethod
    /// / private), the failed type is propagated forward so the
    /// downstream `check_synthetic_method_call` sees an UNTYPED arg.
    /// UNTYPED subtypes everything, so no spurious
    /// `ArgumentTypeMismatch` piles up on top of the original failure —
    /// mirrors the ivar `+=` cascade guard (`visit_instance_variable_
    /// operator_write_node`).
    ///
    /// Returns the value passed to `[]=` (`write_value_ty`), matching
    /// Ruby's assignment-expression semantics (the whole expression
    /// evaluates to the value written, not `[]=`'s declared return
    /// type).
    #[allow(clippy::too_many_arguments)]
    fn check_index_compound_write<'pr>(
        &mut self,
        receiver_node: Option<Node<'pr>>,
        arguments_node: Option<ruby_prism::ArgumentsNode<'pr>>,
        has_block: bool,
        op: IndexCompoundWriteOp<'_>,
        value: &Node<'pr>,
        read_location: SourceLocation,
        write_location: SourceLocation,
    ) -> Ty {
        let Some(recv) = receiver_node else {
            // Receiver-less `Index*WriteNode` is structurally unreachable
            // from Prism's Ruby grammar (`[idx] ||= v` at parse time
            // splits into a freestanding `ArrayNode` and a separate
            // expression — verified 2026-06-08). The defensive descent
            // below preserves diagnostic surface on rhs / index args in
            // the unlikely event a future Prism release lifts that
            // restriction; without it, the rhs subtree would silently
            // drop NoMethod / UnknownConstant emissions.
            self.check_node(value, None);
            if let Some(args_node) = arguments_node {
                self.visit_arguments_node(&args_node);
            }
            return Ty::UNTYPED;
        };

        let recv_ty = self.check_node(&recv, None);
        let value_ty = self.check_node(value, None);

        // Walk the index-argument subtree so calls / constants inside
        // `foo[bad_call()] ||= v` surface their own diagnostics. The
        // `collect_arguments_from` pass below is `infer_type`-based and
        // read-only (`calls.rs:274-345`); without this explicit visit,
        // inner CallNode / ConstantReadNode emissions are lost. Mirrors
        // `check_call`'s `visit_arguments_node` call at `calls.rs:251-253`.
        if let Some(args_node) = arguments_node.as_ref() {
            self.visit_arguments_node(args_node);
        }

        if recv_ty.is_untyped() {
            return Ty::UNTYPED;
        }

        let args = self.collect_arguments_from(arguments_node, has_block);
        let idx_tys = args.positional.clone();

        // 1. `[]` dispatch — emits NoMethod / PrivateMethodCall / arg
        //    diagnostics for the read side. The diagnostic-count delta
        //    is the failure signal for the cascade guard below; relying
        //    on `Ty::UNTYPED` alone would miss `ArgumentTypeMismatch`
        //    cases (the overload resolves a non-untyped return type
        //    while ATM emits separately).
        let diag_count_before = self.diagnostics.borrow().len();
        let read_ty =
            self.check_synthetic_method_call(recv_ty, "[]", idx_tys.clone(), read_location);
        let read_emitted_diag = self.diagnostics.borrow().len() != diag_count_before;

        // Cascade guard for the index args. If `[]` already complained
        // about the index-arg shape (wrong key type, NoMethod, private),
        // pass UNTYPED idx args to `[]=` so the same key-mismatch isn't
        // reported twice. Method-presence and value-param diagnostics
        // on `[]=` still fire (UNTYPED arg subtypes any param).
        let write_idx_tys: Vec<Ty> = if read_emitted_diag {
            vec![Ty::UNTYPED; idx_tys.len()]
        } else {
            idx_tys
        };

        // 2. Compute the value passed to `[]=`. `+=` dispatches the named
        //    operator on the read type (skipped when `[]` already failed,
        //    to avoid dispatching on an untyped receiver). `||=`/`&&=`
        //    always pass the rhs itself — Ruby's runtime desugar (`a[i]
        //    || (a[i] = v)` / `a[i] && (a[i] = v)`) writes `v` verbatim,
        //    never the narrowed union computed in step 4. Passing that
        //    union here instead would spuriously mismatch a `[]=` param
        //    declared narrower than the read type (e.g. `[]=`'s value
        //    param typed `String` rejecting a `read | rhs` union).
        let write_arg_ty = match &op {
            IndexCompoundWriteOp::Operator(op_name, op_location) => {
                if read_ty.is_untyped() {
                    Ty::UNTYPED
                } else {
                    self.check_synthetic_method_call(
                        read_ty,
                        op_name,
                        vec![value_ty],
                        op_location.clone(),
                    )
                }
            }
            IndexCompoundWriteOp::Or | IndexCompoundWriteOp::And => value_ty,
        };

        // 3. `[]=` dispatch. When `write_arg_ty` is UNTYPED (from a
        //    failed operator dispatch) the value-param subtype check
        //    passes silently — the original failure already surfaced.
        //    `[]=`'s own return type is discarded here: Ruby's assignment
        //    expression semantics ignore it and evaluate to the written
        //    value instead (verified 2026-07-17 via ask-steep).
        let mut write_args = write_idx_tys;
        write_args.push(write_arg_ty);
        let _ = self.check_synthetic_method_call(recv_ty, "[]=", write_args, write_location);

        // 4. The whole expression's value type. For `+=` this is the
        //    same as the `[]=` argument (the operator result — Ruby's
        //    assignment-expression semantics). For `||=`/`&&=` it's the
        //    truthy/falsy-narrowing union of the read type and the rhs
        //    (Steep parity, `or_write_value_ty` / `and_write_value_ty`),
        //    which is *not* what was passed to `[]=` in step 2.
        if read_ty.is_untyped() {
            Ty::UNTYPED
        } else {
            match &op {
                IndexCompoundWriteOp::Operator(..) => write_arg_ty,
                // Widen the rhs's literal type before folding it into
                // the narrowing union — Steep parity (steep-playground
                // 2026-07-17: `NilableBox#[]: Integer?` + `self[0] ||=
                // "hi"` types as `Integer | String`, not `Integer |
                // "hi"`). Collection literals widen the same way
                // (`widen_literal_to_base`'s doc comment); `[]=`'s
                // dispatch argument above stays unwidened since that's
                // the actual runtime value passed, which may be a
                // narrower literal than the declared param type.
                IndexCompoundWriteOp::Or => {
                    let widened = self.widen_literal_to_base(value_ty);
                    or_write_value_ty(read_ty, widened, self.env.types())
                }
                IndexCompoundWriteOp::And => {
                    let widened = self.widen_literal_to_base(value_ty);
                    and_write_value_ty(read_ty, widened, self.env.types())
                }
            }
        }
    }

    /// Shared RHS-handling for ivar / cvar / gvar writes. `declared` is
    /// the variable's declared type (or `None` when no declaration was
    /// found), used for the subtype check / `IncompatibleAssignment`
    /// diagnostic. `hint` is what's actually threaded into `check_node`
    /// as the bidirectional inference hint. For plain writes the two
    /// coincide (Steep `ivasgn`/`gvasgn` hint from `context.type_env`,
    /// `type_construction.rb:2436-2437`); for compound (`||=`/`&&=`)
    /// writes they diverge — see `check_compound_ivar_write_rhs`'s doc
    /// comment for why. When `declared` is `Some` and the RHS does not
    /// subtype it, an `IncompatibleAssignment` is emitted at the
    /// assignment's start offset; otherwise the `make_unknown_kind`
    /// closure decides which `Unknown*Variable` kind to surface.
    pub(super) fn check_var_write_rhs(
        &mut self,
        value: &Node<'_>,
        name: &str,
        hint: Option<Ty>,
        declared: Option<Ty>,
        make_unknown_kind: impl FnOnce(String) -> DiagnosticKind,
        location_offset: usize,
    ) -> Ty {
        let position = self.offset_to_location(location_offset);
        let rhs_ty = self.check_node(value, hint);
        match declared {
            Some(lhs) => {
                if !self.subtyper().check(rhs_ty, lhs) {
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location: position.clone(),
                        kind: DiagnosticKind::IncompatibleAssignment {
                            lhs_type: self.display_type(lhs),
                            rhs_type: self.display_type(rhs_ty),
                        },
                    });
                }
            }
            None => {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: position,
                    kind: make_unknown_kind(name.to_string()),
                });
            }
        }
        rhs_ty
    }

    /// Walk the resolved `Type` shape of `ty` and append every
    /// reference-position `TypeName` that is not declared anywhere in
    /// the environment to `out`. Mirrors what the build-layer validator
    /// does for `.rbs` declarations (`UnknownTypeName`), so inline
    /// assertions inside method bodies surface the same kind of error.
    fn collect_unknown_type_names(&self, ty: Ty, out: &mut Vec<String>) {
        self.walk_reference_type_names(ty, &mut |name| self.check_type_name(&name, out));
    }

    /// Walk the resolved `Type` shape of `ty`, invoking `record` on
    /// every reference-position `TypeName` (class / singleton /
    /// interface / alias heads, recursing through args, unions,
    /// records, and proc signatures). Shared by the unknown-type-name
    /// check above and extract's signature-dependency record
    /// (`record_extract_signature_types`) so the two never drift on
    /// which positions count as references.
    pub(super) fn walk_reference_type_names(&self, ty: Ty, record: &mut dyn FnMut(TypeName)) {
        let resolved = self.env.types().resolve(ty);
        match resolved {
            Type::ClassInstance { name, args } => {
                record(*name);
                for &arg in args {
                    self.walk_reference_type_names(arg, record);
                }
            }
            Type::ClassSingleton { name } => record(*name),
            Type::Interface { name, args } => {
                record(*name);
                for &arg in args {
                    self.walk_reference_type_names(arg, record);
                }
            }
            Type::Alias { name, args } => {
                record(*name);
                for &arg in args {
                    self.walk_reference_type_names(arg, record);
                }
            }
            Type::Union(members) | Type::Intersection(members) | Type::Tuple(members) => {
                for &m in members {
                    self.walk_reference_type_names(m, record);
                }
            }
            Type::Optional(inner) => self.walk_reference_type_names(*inner, record),
            Type::Record { fields } => {
                for (_, v, _) in fields {
                    self.walk_reference_type_names(*v, record);
                }
            }
            Type::Proc {
                type_,
                self_type,
                block,
            } => {
                self.walk_reference_type_names_in_fn(type_, record);
                if let Some(t) = *self_type {
                    self.walk_reference_type_names(t, record);
                }
                if let Some(b) = block {
                    self.walk_reference_type_names_in_fn(&b.type_, record);
                    if let Some(t) = b.self_type {
                        self.walk_reference_type_names(t, record);
                    }
                }
            }
            _ => {}
        }
    }

    /// Walk a function-type signature (parameters + return), invoking
    /// `record` on every reference-position `TypeName`. Shared by the
    /// `Type::Proc` arm and def-signature walks.
    pub(super) fn walk_reference_type_names_in_fn(
        &self,
        fn_ty: &FunctionType,
        record: &mut dyn FnMut(TypeName),
    ) {
        let f = match fn_ty {
            FunctionType::Typed(f) => f,
            FunctionType::Untyped(u) => {
                self.walk_reference_type_names(u.return_type, record);
                return;
            }
        };
        for ty in &f.required_positionals {
            self.walk_reference_type_names(*ty, record);
        }
        for ty in &f.optional_positionals {
            self.walk_reference_type_names(*ty, record);
        }
        if let Some(ty) = f.rest_positional {
            self.walk_reference_type_names(ty, record);
        }
        for ty in &f.trailing_positionals {
            self.walk_reference_type_names(*ty, record);
        }
        for (_, ty) in &f.required_keywords {
            self.walk_reference_type_names(*ty, record);
        }
        for (_, ty) in &f.optional_keywords {
            self.walk_reference_type_names(*ty, record);
        }
        if let Some(ty) = f.rest_keyword {
            self.walk_reference_type_names(ty, record);
        }
        self.walk_reference_type_names(f.return_type, record);
    }

    /// Surface a `#: T` whose body did not parse as a Steep / RBS type
    /// (`#: <>>`, `#: 1+2`, etc.) as `AnnotationSyntaxError`. Without
    /// this the user would see no signal that the assertion they wrote
    /// is silently discarded.
    fn push_annotation_syntax_error(&mut self, range: crate::ast::ruby::PrismByteRange) {
        let position = self.offset_to_location(range.0 as usize);
        let annotation_text =
            String::from_utf8_lossy(&self.source[range.0 as usize..range.1 as usize]).to_string();
        self.push_diagnostic(Diagnostic {
            scope: None,
            location: position,
            kind: DiagnosticKind::AnnotationSyntaxError {
                annotation_text,
                parser_error: None,
            },
        });
    }

    fn check_type_name(&self, name: &TypeName, out: &mut Vec<String>) {
        let exists = match self.env.names().type_name_kind(*name) {
            Some(Kind::Class) => {
                self.env.is_declared_class(name) || self.env.is_declared_class_alias(name)
            }
            Some(Kind::Interface) => self.env.is_declared_interface(name),
            Some(Kind::Alias) => self.env.is_declared_type_alias(name),
            // A namespace root cannot be declared.
            None => false,
        };
        if !exists {
            out.push(self.env.names().resolve(name));
        }
    }
}

/// Which compound-write family `check_index_compound_write` is
/// servicing. The `+=`-family dispatches a named operator method on the
/// read type; `||=`/`&&=` compute their value type via truthy/falsy
/// narrowing instead (`or_write_value_ty` / `and_write_value_ty`).
enum IndexCompoundWriteOp<'a> {
    Operator(&'a str, SourceLocation),
    Or,
    And,
}

/// Value type of `foo[idx] ||= v`: Steep desugars this to `foo.[](idx)
/// || foo.[]=(idx, v)` (`type_construction.rb:2395-2428`) and runs the
/// ordinary `:or` narrowing on it, which folds the right operand away
/// entirely when the left is statically always-truthy (steep-playground
/// 2026-07-17: `x: Integer` / `x || "hi"` types as `Integer`, not a
/// union). `check_or_node` (`inference.rs:3246`) does not implement
/// this fold — it always unions both arms regardless of reachability
/// (a known gap, tracked separately) — so this is a standalone
/// reimplementation rather than a
/// shared extraction, scoped to what index compound writes need.
///
/// `pub(super)`: also called from `infer_type`'s read-only `IndexOrWriteNode`
/// arm (`inference.rs`) so the two paths' value types stay in sync.
pub(super) fn or_write_value_ty(read_ty: Ty, write_value_ty: Ty, types: &TypeTable) -> Ty {
    match partition_falsy(read_ty, types) {
        // No falsy member: `[]`'s result is statically always-truthy,
        // so the `[]=` branch never executes at runtime. Steep folds
        // the whole expression to the read type alone.
        None => read_ty,
        Some(_) => {
            let truthy = partition_truthy(read_ty, types).unwrap_or(Ty::BOTTOM);
            union_of(truthy, write_value_ty, types)
        }
    }
}

/// Value type of `foo[idx] &&= v`: dual of `or_write_value_ty` (see its
/// doc comment for the Steep desugar / reachability-fold rationale).
/// Two reachability folds, mirroring `type_construction.rb:1820-1828`'s
/// `case` on `left_truthy.unreachable` / `left_falsy.unreachable`:
///
/// - read is statically always-truthy (no falsy member): `&&=` always
///   executes the `[]=` branch, so the value is the rhs alone
/// - read is statically always-falsy (no truthy member, e.g. `[]`
///   declared to return exactly `nil`): the `[]=` branch is
///   unreachable (`&&` short-circuits on the falsy left), so the value
///   is the read type's falsy partition alone — this can't ride on
///   `union_of`'s `Bottom` absorption the way `or_write_value_ty`'s
///   symmetric case does, because here the "empty" side is the left
///   argument, so it needs its own branch.
///
/// `pub(super)`: also called from `infer_type`'s read-only `IndexAndWriteNode`
/// arm (`inference.rs`) so the two paths' value types stay in sync.
pub(super) fn and_write_value_ty(read_ty: Ty, write_value_ty: Ty, types: &TypeTable) -> Ty {
    match partition_falsy(read_ty, types) {
        None => write_value_ty,
        Some(falsy) => match partition_truthy(read_ty, types) {
            None => falsy,
            Some(_) => union_of(write_value_ty, falsy, types),
        },
    }
}

/// See `cast_unwrap::UnwrappedCastRhs`. Re-exported here for source
/// stability; the type-checker visitor is the sole `cast:`-consuming
/// caller so keeping the name reachable here matches its old surface.
use crate::cast_unwrap::UnwrappedCastRhs;

/// Peel a single `_ = expr` cast wrapper off a constant write's RHS so
/// `Foo = _ = Struct.new(...) do ... end` recognizes the same
/// `CallNode` that the uncast form does. Gate semantics
/// (`SPECIAL_LVAR_NAMES` idiom, `__skip__` contract, single-level) are
/// documented at the `cast_unwrap` module's [`CastUnwrapConfig::VISITOR`].
fn unwrap_single_cast_call<'pr>(value: &Node<'pr>) -> Option<UnwrappedCastRhs<'pr>> {
    crate::cast_unwrap::unwrap_cast(value, &crate::cast_unwrap::CastUnwrapConfig::VISITOR)
}

/// Recognize the `Const = Class.new(Parent?) do ... end` shape: a call
/// whose receiver is bare or rooted `Class`, whose name is `new`, that
/// carries a `do ... end` / `{}` block, and whose argument list is
/// empty or one bare/path constant (no splat, keyword, or dynamic
/// expression). Reject every other shape so the default walker keeps
/// running for them — including `Const = Class.new(make_super) do ...`
/// where the parent is dynamic.
/// Spell a constant-path LHS as written (`A::B`, `::A::B`), or `None`
/// when any parent in the chain is not a constant (`obj.foo::BAR`).
/// Same shape as the inline collector's `static_constant_path_string`
/// so extract's constant implements symbols join its definitions.
fn static_constant_path_string(node: &ConstantPathNode<'_>) -> Option<String> {
    let child_name = String::from_utf8_lossy(node.name()?.as_slice()).to_string();
    match node.parent() {
        Some(parent) => {
            if let Some(constant) = parent.as_constant_read_node() {
                let parent_name = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                Some(format!("{parent_name}::{child_name}"))
            } else if let Some(path) = parent.as_constant_path_node() {
                Some(format!(
                    "{}::{child_name}",
                    static_constant_path_string(&path)?
                ))
            } else {
                None
            }
        }
        None => Some(format!("::{child_name}")),
    }
}

fn is_class_new_named_lhs_pattern(call: &CallNode<'_>) -> bool {
    if !is_class_dot_new(call) {
        return false;
    }
    let Some(block_arg) = call.block() else {
        return false;
    };
    if block_arg.as_block_node().is_none() {
        return false;
    }
    let Some(args) = call.arguments() else {
        return true;
    };
    let mut iter = args.arguments().iter();
    let Some(first) = iter.next() else {
        return true;
    };
    if iter.next().is_some() {
        return false;
    }
    first.as_constant_read_node().is_some() || first.as_constant_path_node().is_some()
}

/// Recognize the `Const = Struct.new(...) do ... end` /
/// `Const = Data.define(...) do ... end` shape: a call whose
/// receiver+method matches one of the Struct/Data constructors and
/// that carries a `do ... end` / `{}` block. The argument list (member
/// symbols, optional keyword_init hash, etc.) is intentionally not
/// gated here — the recognizer mirrors `data_struct_class_kind_and_name`
/// in the inline parser, which also only inspects receiver+method, and
/// the dispatch only fires when the LHS is already declared as a class
/// in RBS, so the argument shape cannot accidentally promote an
/// unrelated call.
fn is_data_struct_named_lhs_pattern(call: &CallNode<'_>) -> bool {
    if data_struct_construction_kind(call).is_none() {
        return false;
    }
    let Some(block_arg) = call.block() else {
        return false;
    };
    block_arg.as_block_node().is_some()
}

/// Convert the type checker's `cref_stack` (parsed `TypeName`s like
/// `::A`, `::A::B`) into the `&[Option<Name>]` shape that
/// `type_builder::build_type` walks for relative-name resolution. Each
/// entry's path is interned against the environment `NameTable` so it
/// participates in the resolver's `all_names`-driven lookup.
///
/// The input is the cref, not `class_stack`: a bare type name written
/// in an annotation resolves through Ruby's lexical nesting
/// (`Module.nesting`), which a `Const = Class.new do ... end` block
/// does not extend even though `self` / `def` inside it belong to the
/// new class. rbs's `InlineParser` and Steep resolve such names at the
/// enclosing scope; `::Ctor::Widget` must not shadow `::Widget` there.
pub(super) fn build_lowering_context_from_cref_stack(
    stack: &[TypeName],
    names: &crate::name::NameTable,
) -> Vec<Option<Name>> {
    stack
        .iter()
        .map(|tn| Some(names.intern(&names.resolve(tn))))
        .collect()
}
