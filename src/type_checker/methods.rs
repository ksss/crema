use ruby_prism::{DefNode, Node, ReturnNode};

use crate::ast::MethodKind;
use crate::definition_builder::{self, type_params_as_variable_args};
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::DeclKindLocal;
use crate::substitution::Substitution;
use crate::type_name::TypeName;
use crate::types::{MethodType, Ty, Type, intersection_of, union_of, union_of_many};

use super::{MethodTarget, TypeChecker};

impl<'env> TypeChecker<'env> {
    pub(super) fn enter_method_context<'pr>(&mut self, node: &DefNode<'pr>) -> bool {
        let method_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let is_singleton = self.is_singleton_method_def(node);
        let is_def_on_instance = node.receiver().is_some() && !is_singleton;
        let target = if is_def_on_instance {
            None
        } else {
            self.lookup_method_target(node)
        };
        let method_type = target
            .as_ref()
            .and_then(|target| target.method_def.unified_method_type(self.env.types()));
        // `method_type` above merges every overload's block signature via
        // `unified_method_type` (`MethodType::unify_overload` /
        // `unify_blocks`), so a `yield` mismatch checked against it may
        // stem from any of `defs`, not just the leaf declaration. Unlike
        // `calls.rs::check_against_method_def` (which bails to
        // `UnresolvedOverloading` before ever reading `defs[0].defined_in`
        // when `defs.len() > 1`), this path has no such short-circuit —
        // `method_type` is still produced and checked. Report `defined_in`
        // only when every overload agrees on the owner; a mixed-owner
        // chain (e.g. an `include`d module's overload merged with the
        // includer's own `... `-continued overload) would otherwise
        // attribute the violation to the wrong class.
        let defined_in = target.as_ref().and_then(|target| {
            let defs = &target.method_def.defs;
            let first = defs.first()?.defined_in;
            defs.iter()
                .all(|def| def.defined_in == first)
                .then_some(first)
        });

        let loc = self.offset_to_location(node.location().start_offset());
        let class_name = self
            .ctx
            .current_class_typename()
            .map(|tn| self.env.names().resolve(tn))
            .unwrap_or_else(|| "(top)".to_string());
        let separator = if is_singleton { "." } else { "#" };
        if let Some(ref mt) = method_type {
            self.record_extract_signature_types(mt);
            let sig = self.display_method_type(mt);
            self.verbose_log(format_args!(
                "{}:{} def {}{}{}: {}",
                loc.range.start_byte, loc.range.end_byte, class_name, separator, method_name, sig
            ));
        } else {
            self.verbose_log(format_args!(
                "{}:{} def {}{}{} — RBS: not found",
                loc.range.start_byte, loc.range.end_byte, class_name, separator, method_name
            ));
        }

        // `def self.x` inside a `[self: T]` block defines a singleton method
        // on the runtime self (the `T` instance), so its body keeps `T`
        // (Steep parity). Capture the enclosing override before the `Method`
        // scope is pushed — `current_self_type_override` stops at that
        // boundary — and re-pin it on the method scope. An instance `def`
        // takes the lexical class instead and gets no override.
        let singleton_self_override = if is_singleton {
            self.ctx.current_self_type_override()
        } else {
            None
        };

        let has_forwarding_param = params_have_forwarding(&node.parameters());
        let forward_leading_positional = node
            .parameters()
            .as_ref()
            .map(|p| p.requireds().len() + p.optionals().len())
            .unwrap_or(0);

        self.ctx.enter_method(
            method_name,
            method_type,
            defined_in,
            is_singleton,
            has_forwarding_param,
            forward_leading_positional,
        );
        if is_def_on_instance {
            self.ctx.set_self_type_override(Ty::UNTYPED);
        } else if let Some(ty) = singleton_self_override {
            self.ctx.set_self_type_override(ty);
        }
        self.bind_parameters(node);
        true
    }

    pub(super) fn lookup_method_target<'pr>(&self, node: &DefNode<'pr>) -> Option<MethodTarget> {
        let method_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let is_singleton = self.is_singleton_method_def(node);

        if node.receiver().is_some() && !is_singleton {
            return None;
        }

        // Top-level `def name` outside any class stack is Ruby-semantically
        // a private instance method on `Object` — fall back to `::Object`
        // so a matching `class Object; def name: ...` sig binds the body.
        // Sig mode only: in `--inline=true` mode the inline collector
        // has already emitted `Ruby::TopLevelMethodDefinition` (parity
        // with rbs `RBS::InlineParser`), and adding a body check on top
        // of that would surface a second, redundant diagnostic. Keep the
        // pre-existing early-return so inline mode's behavior is
        // preserved verbatim.
        // Singleton receivers (`def self.foo`) at the top level are
        // out of scope for this fallback in either mode.
        let class_name = match self.ctx.current_class_typename() {
            Some(name) => *name,
            None if !self.options.inline && !is_singleton => self.env.names().builtins().object,
            None => return None,
        };
        let method_def = if is_singleton {
            definition_builder::lookup_singleton_method_by_type_name(
                self.env,
                class_name,
                self.env.names().intern_symbol(&method_name),
            )?
        } else {
            let method_symbol = self.env.names().lookup_symbol(&method_name)?;
            definition_builder::lookup_instance_method_by_type_name(
                self.env,
                class_name,
                method_symbol,
            )?
        };

        // BasicObject#initialize non-inheritance (soutaro-approved rule):
        // an inline `def initialize` without annotation whose ancestor walk
        // lands on `BasicObject#initialize: () -> void` must NOT inherit
        // that signature — else every user-written `initialize(a, b:)` gets
        // checked against `() -> void` and reports `MethodParameterMismatch`.
        // Returning `None` folds into the same code path as "no method type
        // found": `enter_method_context` stashes no signature and both
        // `check_method_params` / `check_return_type` early-return. A typed
        // intermediate class (`class Base < Object; def initialize: (Integer)
        // -> void`) resolves to `defined_in == Base`, not BasicObject, so
        // the annotated inheritance chain is untouched.
        if !is_singleton
            && method_name == "initialize"
            && definition_builder::defs_all_from_basic_object(self.env, &method_def)
        {
            return None;
        }

        Some(MethodTarget {
            method_name,
            method_def: method_def.clone(),
        })
    }

    pub(super) fn synthetic_method_context_targets<'pr>(
        &self,
        node: &DefNode<'pr>,
    ) -> Vec<TypeName> {
        let method_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let method_kind = if self.is_singleton_method_def(node) {
            MethodKind::Singleton
        } else {
            MethodKind::Instance
        };
        let location = crate::inline_parser::prism_location_range(node.location());
        let source_file = self.env.names().intern(&self.file.to_string_lossy());
        let current = self.ctx.current_class_typename().copied();
        let method = self.env.names().intern_symbol(&method_name);

        // ADR-0032 Decision 5b: index lookup replaces the old per-`def`-node
        // full A-layer walk (cfp-app warm: 8925 entries × N defs). The index
        // is built once at `DefinitionBuilder` construction
        // (`scan_synthetic_concern_defs`).
        self.env.synthetic_concern_targets(
            method,
            method_kind,
            location,
            Some(source_file),
            current,
        )
    }

    pub(super) fn is_singleton_method_def<'pr>(&self, node: &DefNode<'pr>) -> bool {
        match node.receiver() {
            Some(receiver) => receiver.as_self_node().is_some(),
            None => self.ctx.in_singleton_class(),
        }
    }

    pub(super) fn bind_parameters<'pr>(&mut self, node: &DefNode<'pr>) {
        let method_type = match self.ctx.method_type() {
            Some(method_type) => method_type.clone(),
            None => return,
        };

        let params = match node.parameters() {
            Some(params) => params,
            None => return,
        };

        for (index, param) in params.requireds().iter().enumerate() {
            if let Some(required) = param.as_required_parameter_node()
                && let Some(&ty) = method_type.required_positionals().get(index)
            {
                let name_str = String::from_utf8_lossy(required.name().as_slice());
                let name = self.checker_names().intern(&name_str);
                let ty = self.freeze_self_type_for_lvar_binding(ty);
                self.ctx.set_local_variable(name, ty);
            }
        }

        for (index, param) in params.optionals().iter().enumerate() {
            if let Some(optional) = param.as_optional_parameter_node()
                && let Some(&ty) = method_type.optional_positionals().get(index)
            {
                let name_str = String::from_utf8_lossy(optional.name().as_slice());
                let name = self.checker_names().intern(&name_str);
                let ty = self.freeze_self_type_for_lvar_binding(ty);
                self.ctx.set_local_variable(name, ty);
            }
        }

        if let Some(rest_node) = params.rest()
            && let Some(rest) = rest_node.as_rest_parameter_node()
            && let Some(rest_type) = method_type.rest_positional()
            && let Some(name_id) = rest.name()
        {
            let name_str = String::from_utf8_lossy(name_id.as_slice());
            let name = self.checker_names().intern(&name_str);
            let array_name = self.env.names().builtins().array;
            let array_type = self.env.types().intern(Type::ClassInstance {
                name: array_name,
                args: vec![rest_type],
            });
            let array_type = self.freeze_self_type_for_lvar_binding(array_type);
            self.ctx.set_local_variable(name, array_type);
        }

        // Prism `posts()` are the trailing required slots after `*rest`
        // (e.g. `tail` in `def m(*xs, tail)`). They mirror RBS
        // `trailing_positionals`. On the multi-overload path the unified
        // `method_type` has these dropped (Steep-parity, see
        // `Function::merge_for_overload`), so the loop is a no-op there.
        for (index, param) in params.posts().iter().enumerate() {
            if let Some(post) = param.as_required_parameter_node()
                && let Some(&ty) = method_type.trailing_positionals().get(index)
            {
                let name_str = String::from_utf8_lossy(post.name().as_slice());
                let name = self.checker_names().intern(&name_str);
                let ty = self.freeze_self_type_for_lvar_binding(ty);
                self.ctx.set_local_variable(name, ty);
            }
        }

        // `merge_for_overload` widens a keyword that appears in only some
        // overloads' `required_keywords` to `optional(T | nil)`. For Ruby
        // `:kwoptarg`, that `nil` is a phantom — the missing-from-overload
        // case is filled by Ruby's default value at runtime. We resolve the
        // body type from per-overload sigs to bypass the merge widening; the
        // target lookup is shared across all keywords in this def to avoid
        // re-walking the ancestor chain per keyword.
        let keyword_target = params
            .keywords()
            .iter()
            .any(|k| k.as_optional_keyword_parameter_node().is_some())
            .then(|| self.lookup_method_target(node))
            .flatten();

        for keyword_node in params.keywords().iter() {
            if let Some(required) = keyword_node.as_required_keyword_parameter_node() {
                let keyword_str = String::from_utf8_lossy(required.name().as_slice()).to_string();
                if let Some(&(_, ty)) = method_type
                    .required_keywords()
                    .iter()
                    .find(|(keyword_name, _)| keyword_name == &keyword_str)
                {
                    let name = self.checker_names().intern(&keyword_str);
                    let ty = self.freeze_self_type_for_lvar_binding(ty);
                    self.ctx.set_local_variable(name, ty);
                }
            } else if let Some(optional) = keyword_node.as_optional_keyword_parameter_node() {
                let keyword_str = String::from_utf8_lossy(optional.name().as_slice()).to_string();
                let body_ty = keyword_target
                    .as_ref()
                    .and_then(|t| self.keyword_body_type_from_overloads(t, &keyword_str));
                let ty = body_ty.or_else(|| {
                    method_type
                        .optional_keywords()
                        .iter()
                        .find(|(keyword_name, _)| keyword_name == &keyword_str)
                        .map(|&(_, t)| t)
                });
                if let Some(ty) = ty {
                    let name = self.checker_names().intern(&keyword_str);
                    let ty = self.freeze_self_type_for_lvar_binding(ty);
                    self.ctx.set_local_variable(name, ty);
                }
            }
        }

        if let Some(block_param) = params.block()
            && let Some(block) = method_type.block.as_ref()
            && let Some(name_id) = block_param.name()
        {
            let proc_ty = self.env.types().intern(Type::Proc {
                type_: block.type_.clone(),
                self_type: block.self_type,
                block: None,
            });
            let ty = if block.required {
                proc_ty
            } else {
                union_of(proc_ty, Ty::NIL, self.env.types())
            };
            let name_str = String::from_utf8_lossy(name_id.as_slice());
            let name = self.checker_names().intern(&name_str);
            let ty = self.freeze_self_type_for_lvar_binding(ty);
            self.ctx.set_local_variable(name, ty);
        }
    }

    /// Body-side type for a `:kwoptarg` keyword: union of the per-overload
    /// sig types (from `required_keywords` or `optional_keywords`). Overloads
    /// that omit the keyword contribute nothing — bypassing the merge-induced
    /// `nil` that the unified `method_type` carries. Returns `None` when no
    /// overload declares the keyword.
    fn keyword_body_type_from_overloads(
        &self,
        target: &MethodTarget,
        keyword_name: &str,
    ) -> Option<Ty> {
        let mut contributions = Vec::new();
        for overload in target.method_def.method_types() {
            if let Some(&(_, ty)) = overload
                .required_keywords()
                .iter()
                .find(|(n, _)| n == keyword_name)
            {
                contributions.push(ty);
            } else if let Some(&(_, ty)) = overload
                .optional_keywords()
                .iter()
                .find(|(n, _)| n == keyword_name)
            {
                contributions.push(ty);
            }
        }
        if contributions.is_empty() {
            None
        } else {
            Some(union_of_many(&contributions, self.env.types()))
        }
    }

    /// Check an explicit `return expr` against the method's declared return type.
    pub(super) fn check_explicit_return<'pr>(&mut self, node: &ReturnNode<'pr>) {
        let Some(method_type) = self.ctx.method_type() else {
            return;
        };

        let return_type = method_type.return_type();
        if return_type == Ty::VOID || return_type.is_untyped() {
            return;
        }

        let natural = if let Some(arguments) = node.arguments() {
            let args = arguments.arguments();
            if let Some(first) = args.iter().next() {
                self.infer_type(&first, return_type_hint(return_type))
            } else {
                Ty::NIL
            }
        } else {
            Ty::NIL
        };

        // A trailing `#: T` on the `return` line overrides the natural
        // type (Steep `:assertion` parity), exactly as the assignment
        // path treats `x = expr #: T`.
        let actual = if let Some(asserted) = self
            .lookup_trailing_assertion(node.location().start_offset(), node.location().end_offset())
        {
            let position = self.offset_to_location(node.location().start_offset());
            self.emit_false_assertion_if_incompatible(natural, asserted, position);
            asserted
        } else {
            natural
        };

        if actual.is_untyped() {
            return;
        }

        let subtyper = self.subtyper_preserving();
        let instance_subst = self.enclosing_instance_substitution();
        let types = self.env.types();
        let gate_expected = match &instance_subst {
            Some(s) => s.apply(return_type, types),
            None => return_type,
        };
        // Steep parity for `() -> instance`: flip the relation when the
        // declared return is `instance` so a subclass instance is refused.
        // See `check.rb:293-301` ("`T <: instance` doesn't hold generally,
        // but ... for compatibility").
        let bidirectional =
            instance_subst.is_some() && matches!(types.resolve(return_type), Type::InstanceType);
        let gate_ok = subtyper.check(actual, gate_expected)
            && (!bidirectional || subtyper.check(gate_expected, actual));
        if !gate_ok {
            let method_name = self.ctx.method_name().unwrap_or("(unknown)").to_string();
            let location = self.offset_to_location(node.location().start_offset());
            self.push_diagnostic(Diagnostic {
                scope: None,
                location,
                kind: DiagnosticKind::MethodBodyTypeMismatch {
                    method_name,
                    expected: self.display_type(return_type),
                    actual: self.display_type(actual),
                },
            });
        }
    }

    /// Check Ruby def parameters against RBS method type parameters.
    /// Reports MethodArityMismatch and MethodParameterMismatch.
    ///
    /// Steep parity: the def is checked against the single composite
    /// signature that `enter_method_context` already folded from the
    /// overload set (`TypeConstruction#for_new_method`'s
    /// `inject {|t1, t2| t1 + t2}`). A def must accept every caller shape
    /// any overload admits; the merge encodes that by melting one-sided
    /// slots into optionals (see `Function::merge_for_overload`), so
    /// role-split overload sets like
    /// `(Buffer, Integer, Integer) | (buffer:, start_pos:, end_pos:)`
    /// compare cleanly against Ruby's all-optarg/kwoptarg def.
    pub(super) fn check_method_params<'pr>(&mut self, node: &DefNode<'pr>) {
        // The visitor calls `enter_method_context` immediately before this,
        // and it stashes `Method::unified_method_type` for exactly the defs
        // `lookup_method_target` resolves (both bail on def-on-instance), so
        // the ctx clone is the same merged signature without re-folding.
        let Some(merged) = self.ctx.method_type().cloned() else {
            return;
        };
        let Some(method_name) = self.ctx.method_name().map(str::to_string) else {
            return;
        };
        // `(?) -> T` accepts any def shape; one untyped overload melts the
        // whole merged signature into untyped, so bail before the per-param
        // detectors read its empty accessors as "no parameter" slots.
        if merged.is_untyped_function() {
            return;
        }

        let params = node.parameters();
        let ruby_params = RubyParams::from_node(&params);

        let positional_mismatches = ruby_params.detect_positional_mismatches(&merged, &params);
        if ruby_params.matches_overload_with(&merged, &params, &positional_mismatches) {
            return;
        }

        let method_location = self.offset_to_location(node.location().start_offset());

        // The three axes (positional per-param, keyword, kwrestarg) are
        // detected independently — a positional-kind divergence must not
        // silence a keyword mismatch (Steep runs the keyword walk
        // regardless). When any per-parameter diagnostic fires we suppress
        // `MethodArityMismatch` — the more specific diagnostic already
        // explains the gap.
        let arity_matches =
            ruby_params.positional_kinds_compatible(&merged) && ruby_params.arity_matches(&merged);
        let keyword_mismatch = ruby_params.first_keyword_kind_mismatch(&merged);
        let kwrestarg_mismatch = ruby_params.detect_kwrestarg_mismatch(&merged, &params);
        let has_per_param = !positional_mismatches.is_empty()
            || keyword_mismatch.is_some()
            || kwrestarg_mismatch.is_some();

        if has_per_param {
            for mismatch in positional_mismatches {
                let position = self.byte_range_to_location(mismatch.range.0, mismatch.range.1);
                self.push_diagnostic(Diagnostic::at(
                    position,
                    param_mismatch_diagnostic(
                        mismatch.class,
                        method_name.clone(),
                        mismatch.param_name,
                        mismatch.ruby_kind,
                        mismatch.rbs_kind,
                    ),
                ));
            }
            if let Some(kw) = keyword_mismatch {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: method_location.clone(),
                    kind: param_mismatch_diagnostic(
                        kw.class,
                        method_name.clone(),
                        kw.param_name,
                        kw.ruby_kind,
                        kw.rbs_kind,
                    ),
                });
            }
            if let Some(kw) = kwrestarg_mismatch {
                self.push_diagnostic(Diagnostic {
                    scope: None,
                    location: method_location,
                    kind: param_mismatch_diagnostic(
                        kw.class,
                        method_name,
                        kw.param_name,
                        kw.ruby_kind,
                        kw.rbs_kind,
                    ),
                });
            }
        } else if !arity_matches {
            let rbs_required = merged.min_arity();
            let rbs_optional = merged.optional_positionals().len();
            let rbs_has_rest = merged.rest_positional().is_some();
            let rbs_total_kw = merged.required_keywords().len() + merged.optional_keywords().len();
            let rbs_has_rest_kw = merged.rest_keyword().is_some();
            self.push_diagnostic(Diagnostic {
                scope: None,
                location: method_location,
                kind: DiagnosticKind::MethodArityMismatch {
                    method_name,
                    ruby_arity: format_arity(
                        ruby_params.required(),
                        ruby_params.optional,
                        ruby_params.has_rest,
                        ruby_params.total_keywords(),
                        ruby_params.has_rest_kw,
                    ),
                    rbs_arity: format_arity(
                        rbs_required,
                        rbs_optional,
                        rbs_has_rest,
                        rbs_total_kw,
                        rbs_has_rest_kw,
                    ),
                },
            });
        }
    }

    /// Return the current method's return-type hint, or `None` when
    /// there is no method type to check against or the merge collapsed
    /// to an untyped function (`(?) -> T` accepts any body, so no hint
    /// is useful). Exposed so `check_def_node_in_current_context` can
    /// hand the body's own `check_node` walk the same hint
    /// `check_return_type` would otherwise need to re-derive.
    pub(super) fn current_method_return_hint(&self) -> Option<Ty> {
        let merged = self.ctx.method_type()?;
        if merged.is_untyped_function() {
            return None;
        }
        return_type_hint(merged.return_type())
    }

    pub(super) fn check_return_type<'pr>(&mut self, node: &DefNode<'pr>, body_value: Option<Ty>) {
        let Some(target) = self.lookup_method_target(node) else {
            return;
        };
        // Mirrors Steep's `for_new_method` (`type_construction.rb:158-163`):
        // the def body is checked once against the overload set folded
        // into a single composite signature, not once per overload
        // (`ctx.method_type()` is the same unified signature
        // `check_method_params` already consumes — see its doc comment).
        // `None` here means the same "no method type to check against"
        // cases `enter_method_context` bails on (def-on-instance, empty
        // defs, unresolved alias) — `check_return_type` simply has
        // nothing to compare the body's type against.
        let Some(merged) = self.ctx.method_type().cloned() else {
            return;
        };
        // A merge with any untyped-side overload collapses the whole
        // composite to an untyped function (`check_method_params`'s
        // `is_untyped_function` bail, methods.rs:443) — `(?) -> T` accepts
        // any body, so bail before comparing against it.
        if merged.is_untyped_function() {
            return;
        }

        let instance_subst = self.enclosing_instance_substitution();
        let body = match node.body() {
            Some(body) => body,
            None => {
                if !self.unified_return_accepts(&merged, Ty::NIL, &instance_subst) {
                    let expected = merged.return_type();
                    let location = self.offset_to_location(node.location().start_offset());
                    self.push_diagnostic(Diagnostic {
                        scope: None,
                        location,
                        kind: DiagnosticKind::MethodBodyTypeMismatch {
                            method_name: target.method_name,
                            expected: self.display_type(expected),
                            actual: "nil".to_string(),
                        },
                    });
                }
                return;
            }
        };

        // `body_value` is the Ty the body's own `check_node` walk already
        // computed (`check_def_node_in_current_context` gives that walk
        // the return hint and captures its result). Reusing it instead of
        // re-inferring here avoids a second, position-blind pass: a
        // rescue-bearing body's live scope is already widened back
        // (`check_begin_node`'s post-join env, e.g. `T | nil` for a
        // rescue arm that may never touch a body-local) by the time a
        // second pass would run, so re-reading a local variable there
        // would pick up the widened type instead of the narrow type live
        // at the tuple/array literal's original evaluation point.
        let natural = body_value.expect("body_value is Some whenever node.body() is Some");
        // A trailing `#: T` on the body's last expression overrides the
        // natural type (Steep `:assertion` parity), exactly as the
        // assignment path treats `x = expr #: T`. Without this, an
        // idiom like `self.class.new(...) #: self` satisfying a
        // `-> self` signature is a false MethodBodyTypeMismatch.
        // `emit_false_assertion_if_incompatible` here owns the
        // `FalseAssertion` for the def-body return — the statement-
        // position gate in `check_statements_with_hint` suppresses its
        // last-stmt emission via the
        // `suppress_method_body_last_assertion` flag flipped by
        // `check_def_node_in_current_context`, so no double-emit.
        let (assertion_start, assertion_end) = last_stmt_offsets(&body);
        let actual_return_type = if let Some(asserted) =
            self.lookup_trailing_assertion(assertion_start, assertion_end)
        {
            let position = self.offset_to_location(assertion_start);
            self.emit_false_assertion_if_incompatible(natural, asserted, position);
            asserted
        } else {
            natural
        };
        if actual_return_type.is_untyped() {
            return;
        }

        if self.unified_return_accepts(&merged, actual_return_type, &instance_subst) {
            return;
        }

        let expected_return = merged.return_type();
        let location = self.offset_to_location(node.location().start_offset());
        self.push_diagnostic(Diagnostic {
            scope: None,
            location,
            kind: DiagnosticKind::MethodBodyTypeMismatch {
                method_name: target.method_name,
                expected: self.display_type(expected_return),
                actual: self.display_type(actual_return_type),
            },
        });
    }

    /// Steep parity for `() -> instance`: when the declared return is
    /// `instance`, flip the relation so a subclass instance is refused
    /// even though the one-way direction holds. See `check.rb:293-301`
    /// ("`T <: instance` doesn't hold generally, but ... for
    /// compatibility").
    ///
    /// The bidirectional flip is keyed on a *specific* return being
    /// `instance` pre-substitution, so it must survive `unify_overload`
    /// folding an `instance`-returning overload's return into a `Union`
    /// with another overload's return (`() -> instance | () -> Integer`
    /// unifies to `instance | Integer`). Testing `original` for
    /// `Type::InstanceType` as a whole (as this used to) never matches
    /// once it is a `Union` member, silently disabling the flip for any
    /// method with more than one overload — decompose `original` and
    /// test each member independently instead.
    ///
    /// `actual` is decomposed the same way. Verified against a live
    /// Steep (steep-playground, sig-mode target, fresh `steep check`):
    /// a *single* `() -> instance` overload whose body branches between
    /// `self` and a `Q < P` subclass instance is itself rejected by
    /// Steep (`Cannot allow method body have type (self | ::Q) because
    /// declared as type instance`, with the trace showing `::P <: self`
    /// failing inside the decomposed check) — so per-member
    /// decomposition on the `actual` side is correct Steep behavior, not
    /// a shortcut crema is introducing. See the sibling regression test
    /// `test_method_body_singleton_instance_return_rejects_branch_mixing_new_and_subclass`
    /// for the pinned repro (this is a genuine, if perhaps surprising,
    /// Steep restriction: `() -> instance` bodies may not branch between
    /// `self`/`new` and an explicit subclass literal, even though each
    /// branch alone would satisfy the signature).
    fn unified_return_accepts(
        &self,
        merged: &MethodType,
        actual: Ty,
        instance_subst: &Option<Substitution>,
    ) -> bool {
        let subtyper = self.subtyper_preserving();
        let types = self.env.types();
        let original = merged.return_type();
        let expected_members = union_members(original, types);
        let actual_members = union_members(actual, types);

        actual_members.iter().all(|&actual_member| {
            expected_members.iter().any(|&original_member| {
                let expected_member = match instance_subst {
                    Some(s) => s.apply(original_member, types),
                    None => original_member,
                };
                let bidirectional = instance_subst.is_some()
                    && matches!(types.resolve(original_member), Type::InstanceType);
                expected_member == Ty::VOID
                    || expected_member.is_untyped()
                    || (subtyper.check(actual_member, expected_member)
                        && (!bidirectional || subtyper.check(expected_member, actual_member)))
            })
        })
    }

    /// Build a `Substitution` that rewrites `Type::InstanceType` to the
    /// enclosing scope's instance type. Returns `None` only when there is
    /// no enclosing class/module (top-level, or an unresolved name).
    ///
    /// For a class, the target is `ClassInstance(name, [type_params...])`.
    /// For a module, Steep parity (`type_construction.rb:388`) demands an
    /// Intersection `(Object & ...self_types & M)`: a value typed as
    /// `instance` inside `module M` is "an Object that mixes in M plus
    /// every declared self-type".
    ///
    /// The subtyper itself has no arm for `Type::InstanceType`, so any
    /// `instance` left in the expected return type would fall through to
    /// `false` and produce a spurious `MethodBodyTypeMismatch`. Mirrors the
    /// pre-substitution pattern `subtyping.rs::satisfies_interface` already
    /// applies to singleton-side method signatures.
    fn enclosing_instance_substitution(&self) -> Option<Substitution> {
        let class_name = *self.ctx.current_class_typename()?;
        let kind = self.env.class_or_module_kind(&class_name)?;
        let types = self.env.types();
        let args = type_params_as_variable_args(
            self.env.class_type_params_by_type_name(&class_name),
            &class_name,
            types,
        );
        let enclosing_instance_ty = types.intern(Type::ClassInstance {
            name: class_name,
            args,
        });
        let instance_ty = if kind == DeclKindLocal::Module {
            self.module_instance_intersection(&class_name, enclosing_instance_ty)
        } else {
            enclosing_instance_ty
        };
        Some(Substitution::new().with_instance_type(instance_ty))
    }

    /// Steep `type_construction.rb:388` parity: build a module body's
    /// `instance` as `(Object & ...self_types & M)`. The Object base is
    /// unconditional because rbs's `module_self_types_or_default`
    /// substitutes `[Object]` when no self-types were declared and omits
    /// it whenever the user wrote any explicit self-type — adding it
    /// here covers both cases, and `intersection_of` collapses the
    /// duplicate that arises from the implicit default (or from a user-
    /// written `: Object`, since Object is monomorphic and indistinct).
    fn module_instance_intersection(&self, module_name: &TypeName, module_self_ty: Ty) -> Ty {
        let types = self.env.types();
        let object_ty = types.intern(Type::ClassInstance {
            name: self.env.names().builtins().object,
            args: Vec::new(),
        });
        let one = self.env.one_instance_ancestors_arc(module_name);
        let mut acc = object_ty;
        for mixin in &one.self_types {
            // Steep distinguishes `name.interface?` / `name.class?` purely
            // by spelling (`_Foo` vs `Foo`); use the TypeName-kind probe so
            // an unresolved interface name still lands as Type::Interface
            // rather than getting silently demoted to ClassInstance.
            let mixin_ty = if self.env.names().is_interface(mixin.name) {
                types.intern(Type::Interface {
                    name: mixin.name,
                    args: mixin.args.clone(),
                })
            } else {
                types.intern(Type::ClassInstance {
                    name: mixin.name,
                    args: mixin.args.clone(),
                })
            };
            acc = intersection_of(acc, mixin_ty, types);
        }
        intersection_of(acc, module_self_ty, types)
    }
}

/// Full union decomposition: `Union(members)` recurses into each member,
/// `Optional(inner)` decomposes to `inner`'s members plus `NIL`, anything
/// else yields itself as a singleton. Used by `unified_return_accepts` so
/// per-member checks (the `Instance` bidirectional flip) still fire
/// correctly after `unify_overload` folds several overloads' returns into
/// one `Union`.
///
/// The recursion into both variants is required: `union_of`/
/// `normalize_union_members` (types.rs) flatten nested `Type::Union`s but
/// never peel `Type::Optional` (RBS's `T?` stays a distinct node, not
/// sugar expanded into `T | nil` at construction), so `Optional` can
/// surface as a `Union` member (`Union[String, Optional(Y)]`) or as the
/// whole type (`Optional(String)`) alike. Treating an unexpanded
/// `Optional` as a single opaque member made `String?` fail a member-wise
/// subtype check against `String | false | nil` even though `String? <:
/// (String | false | nil)` holds (rbs `resolve_namespace` false positive).
fn union_members(ty: Ty, types: &crate::types::TypeTable) -> Vec<Ty> {
    match types.resolve(ty) {
        Type::Union(members) => members
            .iter()
            .flat_map(|&m| union_members(m, types))
            .collect(),
        Type::Optional(inner) => {
            let mut members = union_members(*inner, types);
            members.push(Ty::NIL);
            members
        }
        _ => vec![ty],
    }
}

/// Byte offsets (start, end) of the source span whose trailing line a
/// method body's return assertion lives on: the last statement of a
/// `StatementsNode` body, or the body node itself otherwise.
fn last_stmt_offsets(body: &Node<'_>) -> (usize, usize) {
    if let Some(stmts) = body.as_statements_node()
        && let Some(last) = stmts.body().iter().last()
    {
        let loc = last.location();
        return (loc.start_offset(), loc.end_offset());
    }
    let loc = body.location();
    (loc.start_offset(), loc.end_offset())
}

pub(super) fn return_type_hint(return_type: Ty) -> Option<Ty> {
    if return_type == Ty::VOID || return_type.is_untyped() {
        None
    } else {
        Some(return_type)
    }
}

/// Format an arity description for diagnostic messages.
fn format_arity(
    required: usize,
    optional: usize,
    has_rest: bool,
    total_kw: usize,
    has_rest_kw: bool,
) -> String {
    let total = required + optional;
    let positional = if has_rest {
        format!("{}+", required)
    } else if optional > 0 {
        format!("{}..{}", required, total)
    } else {
        format!("{}", required)
    };
    if total_kw == 0 && !has_rest_kw {
        return positional;
    }
    let keyword = if has_rest_kw && total_kw == 0 {
        "keyword rest parameter".to_string()
    } else if has_rest_kw {
        format!("{} keyword parameters plus keyword rest", total_kw)
    } else {
        format!("{} keyword parameters", total_kw)
    };
    format!("{} positional, {}", positional, keyword)
}

/// Build the right `DiagnosticKind` for a per-parameter mismatch based on
/// the Steep classification. `Strict` (Ruby `:arg` / `:kwarg`) maps to
/// `MethodParameterMismatch`; `Loose` (Ruby `:optarg` / `:restarg` /
/// `:kwoptarg` / `:kwrestarg`) maps to `DifferentMethodParameterKind`.
fn param_mismatch_diagnostic(
    class: ParamMismatchClass,
    method_name: String,
    param_name: String,
    ruby_kind: String,
    rbs_kind: String,
) -> DiagnosticKind {
    match class {
        ParamMismatchClass::Strict => DiagnosticKind::MethodParameterMismatch {
            method_name,
            param_name,
            ruby_kind,
            rbs_kind,
        },
        ParamMismatchClass::Loose => DiagnosticKind::DifferentMethodParameterKind {
            method_name,
            param_name,
            ruby_kind,
            rbs_kind,
        },
    }
}

/// `true` when a Prism ParametersNode declares `(...)` (the Ruby
/// argument-forwarding shorthand). Centralised so the def-side
/// `enter_method_context` and the RubyParams sig-arity check stay in
/// lockstep — Codex quality review (2026-06-09) flagged the original
/// inline duplication.
fn params_have_forwarding(params: &Option<ruby_prism::ParametersNode<'_>>) -> bool {
    params
        .as_ref()
        .and_then(|p| p.keyword_rest())
        .map(|kr| kr.as_forwarding_parameter_node().is_some())
        .unwrap_or(false)
}

/// A single positional kind mismatch detected when Ruby parameters do not
/// align with an RBS overload. Reported per-parameter (Steep parity).
#[derive(Clone)]
struct PositionalKindMismatch {
    param_name: String,
    ruby_kind: String,
    rbs_kind: String,
    /// `(start_byte, end_byte)` of the Ruby parameter node — the full
    /// Prism location (e.g. `b = nil` for an optarg, `*rest` for a
    /// restarg), not just the name. Lets the diagnostic highlight the
    /// whole parameter instead of a single point.
    range: (usize, usize),
    class: ParamMismatchClass,
}

/// A single keyword kind mismatch. Carries the Steep classification so the
/// emitter picks between `MethodParameterMismatch` (Strict) and
/// `DifferentMethodParameterKind` (Loose).
struct KeywordKindMismatch {
    param_name: String,
    ruby_kind: String,
    rbs_kind: String,
    class: ParamMismatchClass,
}

/// Which Steep diagnostic to emit for a parameter mismatch.
/// `Strict` → `Ruby::MethodParameterMismatch` (Ruby side is required positional / required keyword).
/// `Loose`  → `Ruby::DifferentMethodParameterKind` (Ruby side is optarg / restarg / kwoptarg / kwrestarg).
#[derive(Clone, Copy)]
enum ParamMismatchClass {
    Strict,
    Loose,
}

/// Ruby-side parameter information extracted from a DefNode.
struct RubyParams {
    leading_required: usize,
    trailing_required: usize,
    optional: usize,
    has_rest: bool,
    required_kw: Vec<String>,
    optional_kw: Vec<String>,
    has_rest_kw: bool,
    /// `def foo(...)` — Ruby's argument-forwarding shorthand. The
    /// caller-side `bar(...)` is verified by Steep's
    /// `Ruby::IncompatibleArgumentForwarding` path (see
    /// `type_checker/calls.rs`); on the def side we simply suppress the
    /// arity / per-parameter kind diagnostics that would otherwise fire
    /// against the RBS signature, since `...` accepts any caller shape.
    has_forwarding: bool,
}

impl RubyParams {
    /// Total required count (leading + trailing). Kept for arity comparisons
    /// where the leading/trailing distinction doesn't matter.
    fn required(&self) -> usize {
        self.leading_required + self.trailing_required
    }

    fn total_keywords(&self) -> usize {
        self.required_kw.len() + self.optional_kw.len()
    }

    fn from_node(params: &Option<ruby_prism::ParametersNode<'_>>) -> Self {
        RubyParams {
            leading_required: params.as_ref().map(|p| p.requireds().len()).unwrap_or(0),
            trailing_required: params.as_ref().map(|p| p.posts().len()).unwrap_or(0),
            optional: params.as_ref().map(|p| p.optionals().len()).unwrap_or(0),
            has_rest: params.as_ref().map(|p| p.rest().is_some()).unwrap_or(false),
            required_kw: params
                .as_ref()
                .map(|p| {
                    p.keywords()
                        .iter()
                        .filter_map(|k| {
                            k.as_required_keyword_parameter_node()
                                .map(|rk| String::from_utf8_lossy(rk.name().as_slice()).to_string())
                        })
                        .collect()
                })
                .unwrap_or_default(),
            optional_kw: params
                .as_ref()
                .map(|p| {
                    p.keywords()
                        .iter()
                        .filter_map(|k| {
                            k.as_optional_keyword_parameter_node()
                                .map(|ok| String::from_utf8_lossy(ok.name().as_slice()).to_string())
                        })
                        .collect()
                })
                .unwrap_or_default(),
            has_rest_kw: params
                .as_ref()
                .map(|p| p.keyword_rest().is_some())
                .unwrap_or(false),
            has_forwarding: params_have_forwarding(params),
        }
    }

    /// Check if the arity (counts) matches an overload.
    /// Keyword arity compares total count (required + optional), not individual counts.
    /// The distinction between required and optional keywords is a kind mismatch, not arity.
    fn arity_matches(&self, overload: &MethodType) -> bool {
        if self.has_forwarding {
            return true;
        }
        if overload.is_untyped_function() {
            return true;
        }
        let rbs_total_kw = overload.required_keywords().len() + overload.optional_keywords().len();
        let rbs_has_rest_kw = overload.rest_keyword().is_some();

        self.positional_arity_matches(overload)
            && self.total_keywords() == rbs_total_kw
            && self.has_rest_kw == rbs_has_rest_kw
    }

    fn positional_arity_matches(&self, overload: &MethodType) -> bool {
        // `def foo(...)` accepts any caller shape — defer all sig
        // compatibility to the call-side forwarding diagnostic
        // (`Ruby::IncompatibleArgumentForwarding`).
        if self.has_forwarding {
            return true;
        }
        if overload.is_untyped_function() {
            return true;
        }
        let rbs_required = overload.min_arity();
        let rbs_has_rest = overload.rest_positional().is_some();

        // Ruby's required count must not exceed RBS required: if Ruby needs
        // more args than the minimum RBS callers will provide, calls fail.
        if self.required() > rbs_required {
            return false;
        }

        // Ruby must be able to accept at least rbs_required positionals (the
        // minimum every RBS caller will pass).
        let ruby_max = if self.has_rest {
            None // unlimited
        } else {
            Some(self.required() + self.optional)
        };
        if ruby_max.is_some_and(|m| m < rbs_required) {
            return false;
        }

        // If RBS has rest, callers can pass unlimited positionals; Ruby must
        // also have rest to absorb them. A has_rest mismatch is additionally
        // reported as a kind mismatch by positional_kinds_compatible, but we
        // gate on it here too so arity_matches reliably returns false.
        if rbs_has_rest && !self.has_rest {
            return false;
        }

        // If RBS has no rest, callers pass at most max_arity() positionals.
        // Ruby must be able to receive that many. max_arity() returns None
        // when rest is present (already handled above), so Some(_) here means
        // RBS is bounded.
        if overload
            .max_arity()
            .is_some_and(|rbs_max| ruby_max.is_some_and(|m| m < rbs_max))
        {
            return false;
        }

        true
    }

    fn is_leading_rest_pattern(&self) -> bool {
        self.has_rest && self.leading_required == 0 && self.trailing_required > 0
    }

    /// Check if arity, positional kinds, and keyword kinds all match the
    /// merged overload signature, reusing the positional-mismatch list the
    /// caller also needs for emission. Used by [`check_method_params`] as
    /// the fast-path early exit before falling back to per-parameter
    /// diagnostic emission.
    fn matches_overload_with(
        &self,
        overload: &MethodType,
        params_node: &Option<ruby_prism::ParametersNode<'_>>,
        positional_mismatches: &[PositionalKindMismatch],
    ) -> bool {
        if !self.arity_matches(overload) {
            return false;
        }
        if !positional_mismatches.is_empty() {
            return false;
        }
        if self.first_keyword_kind_mismatch(overload).is_some() {
            return false;
        }
        self.detect_kwrestarg_mismatch(overload, params_node)
            .is_none()
    }

    /// Returns true when Ruby's and RBS's positional structure differ in
    /// kind (rest presence / trailing presence). Used to route between
    /// arity-only mismatch (`MethodArityMismatch`) and per-parameter kind
    /// mismatch (`MethodParameterMismatch`).
    fn positional_kinds_compatible(&self, overload: &MethodType) -> bool {
        let rbs_has_rest = overload.rest_positional().is_some();
        if self.has_rest != rbs_has_rest {
            return false;
        }
        // When Ruby has a leading-rest (`*xs, tail` style: leading=0, rest, trailing>0)
        // the RBS overload must also be leading=0/rest. Otherwise we'd
        // bind Ruby's `*xs` against an RBS slot whose semantics differ.
        if self.is_leading_rest_pattern() && !overload.required_positionals().is_empty() {
            return false;
        }
        // Ruby trailing required count must match RBS trailing slot count
        // when both sides have rest. If they don't, Ruby's `tail` lands
        // on different RBS slots than declared.
        if self.has_rest && self.trailing_required != overload.trailing_positionals().len() {
            return false;
        }
        // When neither side has rest, Ruby's trailing should be 0 (it is,
        // because posts only exist with a rest). RBS trailing may still
        // be nonempty without rest in malformed input, but that's outside
        // the scope of crema's validation (rbs rejects such sigs).
        true
    }

    /// Per-parameter positional kind mismatches. Mirrors Steep's
    /// `type_inference/method_params.rb` left-to-right slot simulation,
    /// emitting one diagnostic per Ruby parameter that lands on an
    /// incompatible RBS slot. Steep's classification (`MethodParameterMismatch`
    /// vs `DifferentMethodParameterKind`) is preserved via
    /// [`ParamMismatchClass`]: Ruby strict side (`:arg`) → `Strict`, Ruby
    /// loose side (`:optarg` / `:restarg`) → `Loose`.
    fn detect_positional_mismatches(
        &self,
        overload: &MethodType,
        params_node: &Option<ruby_prism::ParametersNode<'_>>,
    ) -> Vec<PositionalKindMismatch> {
        let mut out = Vec::new();
        let Some(params) = params_node.as_ref() else {
            return out;
        };
        let rbs_required_count = overload.required_positionals().len();
        let rbs_optional_count = overload.optional_positionals().len();
        let rbs_has_rest = overload.rest_positional().is_some();
        let rbs_trailing_count = overload.trailing_positionals().len();

        // (a) Ruby leading required that overflows into RBS rest. When
        //     RBS has no rest, the count mismatch is reported as
        //     `MethodArityMismatch` via `positional_kinds_compatible`,
        //     so we only emit the kind-mismatch diagnostic for the
        //     rest case here. Steep classifies this as `MethodParameterMismatch`
        //     (Ruby `:arg` side is strict, see method_params.rb:283).
        //
        // (a2) Ruby leading required that overflows into an RBS Optional
        //      slot (rather than Rest). Distinct position range from (a)
        //      — `[rbs_required_count, rbs_required_count +
        //      rbs_optional_count)` vs (a)'s rest-only overflow — so a
        //      param can match at most one of the two arms. Steep
        //      classifies `:arg` vs `Optional` as `MethodParameterMismatch`
        //      too (`:arg` is strict either way, method_params.rb:273-279).
        //      Without this arm, crema fell back to a whole-def
        //      `MethodArityMismatch` for this shape instead of pointing at
        //      the specific overflowing param.
        //
        // (a3) Ruby leading required that overflows past BOTH the Required
        //      and Optional runs, with no RBS Rest slot to catch it either
        //      (`i >= rbs_required_count + rbs_optional_count && !rbs_has_rest`).
        //      Steep's `:arg` walk (method_params.rb's `positional_params`
        //      iterator) has no "ran out of RBS slots entirely" special
        //      case distinct from (a)/(a2) — once the iterator is
        //      exhausted it hits the same `when nil` arm those two also
        //      route through, which unconditionally emits
        //      `MethodParameterMismatch`. Verified live: `def f(a, b)` vs
        //      `(Integer) -> void` (no optional, no rest) reports
        //      `Ruby::MethodParameterMismatch` on `b`, not an arity
        //      diagnostic (steep-playground, sig-mode target, fresh
        //      `steep check`, 2026-07-11). The original design note
        //      assumed this shape stays on the arity fallback — that
        //      assumption didn't hold against live Steep and is corrected
        //      here.
        for (i, param) in params.requireds().iter().enumerate() {
            // (a) and (a2) must stay mutually exclusive per position: when
            // RBS has both an Optional run and a Rest slot (e.g.
            // `(Integer, ?Integer, *Integer)`), a naive `i >=
            // rbs_required_count && rbs_has_rest` guard for (a) overlaps
            // (a2)'s `[rbs_required_count, rbs_required_count +
            // rbs_optional_count)` range — a param inside the Optional
            // range would satisfy both and get pushed twice with
            // contradictory `rbs_kind`s (crema-review adversarial finding,
            // reproduced with `def f(a, b, c)` against exactly that RBS
            // shape). Gate (a) on having fully passed the Optional range
            // too, so (a) only fires for positions genuinely landing on
            // Rest.
            if i >= rbs_required_count + rbs_optional_count
                && rbs_has_rest
                && let Some(req) = param.as_required_parameter_node()
            {
                let name = String::from_utf8_lossy(req.name().as_slice()).to_string();
                out.push(PositionalKindMismatch {
                    param_name: name,
                    ruby_kind: "required positional".to_string(),
                    rbs_kind: "rest parameter".to_string(),
                    range: (
                        param.location().start_offset(),
                        param.location().end_offset(),
                    ),
                    class: ParamMismatchClass::Strict,
                });
            }
            if i >= rbs_required_count
                && i < rbs_required_count + rbs_optional_count
                && let Some(req) = param.as_required_parameter_node()
            {
                let name = String::from_utf8_lossy(req.name().as_slice()).to_string();
                out.push(PositionalKindMismatch {
                    param_name: name,
                    ruby_kind: "required positional".to_string(),
                    rbs_kind: "optional positional".to_string(),
                    range: (
                        param.location().start_offset(),
                        param.location().end_offset(),
                    ),
                    class: ParamMismatchClass::Strict,
                });
            }
            if i >= rbs_required_count + rbs_optional_count
                && !rbs_has_rest
                && let Some(req) = param.as_required_parameter_node()
            {
                let name = String::from_utf8_lossy(req.name().as_slice()).to_string();
                out.push(PositionalKindMismatch {
                    param_name: name,
                    ruby_kind: "required positional".to_string(),
                    rbs_kind: "no parameter".to_string(),
                    range: (
                        param.location().start_offset(),
                        param.location().end_offset(),
                    ),
                    class: ParamMismatchClass::Strict,
                });
            }
        }

        // (b) Ruby restarg. Two failure modes, both classified as
        //     `DifferentMethodParameterKind` (Ruby `:restarg` side is loose,
        //     see method_params.rb:378):
        //       (b1) Ruby leading-rest pattern (`*xs, tail`) vs an RBS shape
        //            that doesn't have leading=0/rest with matching trailing
        //            counts — the trailing args land on wrong RBS slots.
        //       (b2) Ruby restarg lands at a position where RBS still has
        //            Required/Optional slots remaining before the Rest slot,
        //            or RBS has no Rest at all. Mirrors Steep's restarg
        //            simulation that walks unconsumed RBS slots.
        let mut rest_mismatched = false;
        if let Some(rest_node) = params.rest() {
            let leading_consumed = self.leading_required + self.optional;
            let required_unconsumed = rbs_required_count.saturating_sub(leading_consumed);
            let optional_offset = leading_consumed.saturating_sub(rbs_required_count);
            let optional_unconsumed = rbs_optional_count.saturating_sub(optional_offset);
            let is_leading_rest = self.is_leading_rest_pattern();
            let leading_rest_incompatible = is_leading_rest
                && (!rbs_has_rest
                    || rbs_required_count != 0
                    || self.trailing_required != rbs_trailing_count);
            let consumes_unmatched_slot = !is_leading_rest
                && (required_unconsumed > 0 || optional_unconsumed > 0 || !rbs_has_rest);
            if leading_rest_incompatible || consumes_unmatched_slot {
                // Always suppress trailing diagnostics once we know the rest
                // slot itself is incompatible, even if the rest node shape
                // (e.g. anonymous `*`) doesn't yield a RestParameterNode we
                // can name.
                rest_mismatched = true;
                if let Some(rest) = rest_node.as_rest_parameter_node() {
                    let name = rest
                        .name()
                        .map(|n| String::from_utf8_lossy(n.as_slice()).to_string())
                        .unwrap_or_default();
                    let rbs_kind = if !rbs_has_rest {
                        "no rest parameter".to_string()
                    } else if leading_rest_incompatible {
                        "rest parameter at a different position".to_string()
                    } else {
                        "rest parameter consuming required/optional slots".to_string()
                    };
                    out.push(PositionalKindMismatch {
                        param_name: name,
                        ruby_kind: "rest parameter".to_string(),
                        rbs_kind,
                        range: (
                            rest_node.location().start_offset(),
                            rest_node.location().end_offset(),
                        ),
                        class: ParamMismatchClass::Loose,
                    });
                }
            }
        }

        // (c) Ruby trailing required vs RBS trailing slots. Suppress if
        //     rest already flagged (Steep groups the trailing into the
        //     rest mismatch). Steep classifies this as `MethodParameterMismatch`
        //     (Ruby trailing post-rest `:arg` is strict).
        if !rest_mismatched {
            for (i, param) in params.posts().iter().enumerate() {
                if i >= rbs_trailing_count
                    && let Some(post) = param.as_required_parameter_node()
                {
                    let name = String::from_utf8_lossy(post.name().as_slice()).to_string();
                    let rbs_kind = if rbs_trailing_count == 0 {
                        "no trailing parameter".to_string()
                    } else {
                        format!("only {} trailing parameter(s) declared", rbs_trailing_count)
                    };
                    out.push(PositionalKindMismatch {
                        param_name: name,
                        ruby_kind: "trailing positional".to_string(),
                        rbs_kind,
                        range: (
                            param.location().start_offset(),
                            param.location().end_offset(),
                        ),
                        class: ParamMismatchClass::Strict,
                    });
                }
            }
        }

        // (d) Ruby optarg landings. For each `:optarg` at Ruby position
        //     `leading_required + j`, classify against the RBS slot at the
        //     same position: Required / Rest / missing → `DifferentMethodParameterKind`
        //     (Steep method_params.rb:306/315/322), Optional → no emit.
        for (j, param) in params.optionals().iter().enumerate() {
            let Some(opt) = param.as_optional_parameter_node() else {
                continue;
            };
            let name = String::from_utf8_lossy(opt.name().as_slice()).to_string();
            let pos = self.leading_required + j;
            let rbs_slot_kind = if pos < rbs_required_count {
                Some("required positional")
            } else if pos < rbs_required_count + rbs_optional_count {
                None // optarg vs optional slot → OK
            } else if rbs_has_rest {
                Some("rest parameter")
            } else {
                Some("no parameter")
            };
            if let Some(rbs_kind) = rbs_slot_kind {
                out.push(PositionalKindMismatch {
                    param_name: name,
                    ruby_kind: "optional positional".to_string(),
                    rbs_kind: rbs_kind.to_string(),
                    range: (
                        param.location().start_offset(),
                        param.location().end_offset(),
                    ),
                    class: ParamMismatchClass::Loose,
                });
            }
        }

        out
    }

    /// Find the first keyword parameter kind mismatch against a single
    /// overload (post-merge: the composite signature). Steep classification
    /// (method_params.rb:396-458):
    /// - Ruby required kw vs RBS optional kw → `Strict` (line 406)
    /// - Ruby required kw vs RBS rest kw    → `Strict` (line 414)
    /// - Ruby required kw not in RBS        → `Strict` (line 422)
    /// - Ruby optional kw vs RBS required kw → `Loose`  (line 434)
    /// - Ruby optional kw vs RBS rest kw    → `Loose`  (line 446)
    /// - Ruby optional kw not in RBS        → `Loose`  (line 454)
    fn first_keyword_kind_mismatch(&self, overload: &MethodType) -> Option<KeywordKindMismatch> {
        if overload.is_untyped_function() {
            return None;
        }
        let rbs_has_rest_kw = overload.rest_keyword().is_some();

        let ruby_kws = self
            .required_kw
            .iter()
            .map(|name| (name, true))
            .chain(self.optional_kw.iter().map(|name| (name, false)));
        for (ruby_name, ruby_required) in ruby_kws {
            let in_rbs_required = overload
                .required_keywords()
                .iter()
                .any(|(name, _)| name == ruby_name);
            let in_rbs_optional = overload
                .optional_keywords()
                .iter()
                .any(|(name, _)| name == ruby_name);
            if in_rbs_required == ruby_required && (in_rbs_required || in_rbs_optional) {
                continue;
            }
            let rbs_kind = if in_rbs_required {
                "required keyword"
            } else if in_rbs_optional {
                "optional keyword"
            } else if rbs_has_rest_kw {
                "rest keyword"
            } else {
                "no keyword"
            };
            let (ruby_kind, class) = if ruby_required {
                ("required keyword", ParamMismatchClass::Strict)
            } else {
                ("optional keyword", ParamMismatchClass::Loose)
            };
            return Some(KeywordKindMismatch {
                param_name: ruby_name.clone(),
                ruby_kind: ruby_kind.to_string(),
                rbs_kind: rbs_kind.to_string(),
                class,
            });
        }

        None
    }

    /// Detect Ruby `**kwrestarg` against an RBS overload. Emits when there
    /// are RBS keywords the kwrestarg would absorb (named keys consumed by
    /// `**`) or when RBS has no rest keyword. Mirrors Steep
    /// method_params.rb:466-503: any keyword left after the explicit kw/kwopt
    /// loop is consumed by `**` with `has_error = true`, and absence of an
    /// RBS rest keyword also flips `has_error`. Always Loose.
    fn detect_kwrestarg_mismatch(
        &self,
        overload: &MethodType,
        params_node: &Option<ruby_prism::ParametersNode<'_>>,
    ) -> Option<KeywordKindMismatch> {
        if !self.has_rest_kw || self.has_forwarding {
            // `def f(...)` is the argument-forwarding shorthand; treat it as
            // accepting any sig and defer compatibility to the call-side
            // `Ruby::IncompatibleArgumentForwarding` path. Without this
            // guard, prism reports forwarding as a `keyword_rest` and the
            // kwrestarg detector would fire on every `(...)` def.
            return None;
        }
        let rbs_has_rest_kw = overload.rest_keyword().is_some();
        // Linear scan beats a BTreeSet here: this runs once per def against
        // the merged signature and Ruby keyword lists are typically < 10,
        // so allocating a set per call would dominate the comparison cost.
        let is_in_ruby_explicit = |rbs_name: &str| {
            self.required_kw.iter().any(|n| n == rbs_name)
                || self.optional_kw.iter().any(|n| n == rbs_name)
        };
        let unconsumed_required = overload
            .required_keywords()
            .iter()
            .filter(|(name, _)| !is_in_ruby_explicit(name.as_str()))
            .count();
        let unconsumed_optional = overload
            .optional_keywords()
            .iter()
            .filter(|(name, _)| !is_in_ruby_explicit(name.as_str()))
            .count();
        let has_error = unconsumed_required > 0 || unconsumed_optional > 0 || !rbs_has_rest_kw;
        if !has_error {
            return None;
        }
        let name = params_node
            .as_ref()
            .and_then(|p| p.keyword_rest())
            .and_then(|kr| kr.as_keyword_rest_parameter_node())
            .and_then(|krn| krn.name())
            .map(|n| String::from_utf8_lossy(n.as_slice()).to_string())
            .unwrap_or_default();
        let rbs_kind = if !rbs_has_rest_kw {
            if unconsumed_required > 0 || unconsumed_optional > 0 {
                "no rest keyword (unconsumed keywords remain)".to_string()
            } else {
                "no rest keyword".to_string()
            }
        } else {
            "rest keyword consuming required/optional keywords".to_string()
        };
        Some(KeywordKindMismatch {
            param_name: name,
            ruby_kind: "rest keyword".to_string(),
            rbs_kind,
            class: ParamMismatchClass::Loose,
        })
    }
}
