use rustc_hash::{FxHashMap, FxHashSet};

use crate::name::{Name, NameTable};
use crate::pure_call_env::{PureCallEnv, PureKey};
use crate::type_name::TypeName;
use crate::types::{Function, FunctionType, MethodType, Ty};

/// The kind of scope in the scope chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Top-level or module scope
    Top,
    /// Method body scope
    Method,
    /// Block scope (e.g., `each { |x| ... }`)
    Block,
}

/// A single scope in the chain, holding variable bindings.
#[derive(Debug)]
struct Scope {
    kind: ScopeKind,
    bindings: FxHashMap<Name, Ty>,
    bot_rhs_bindings: FxHashSet<Name>,
    self_type_override: Option<Ty>,
    /// Outer lvars pinned on entry to this `Block` scope: the type each
    /// visible lvar had when the closure was entered. Steep's
    /// `enforced_type` slot (`type_env.rb` `local_variable_types`
    /// `name => [type, enforced_type]`, planted by
    /// `pin_local_variables`). A write from inside the closure that
    /// targets a scope outside this one is checked against, and
    /// rebinds to, the pinned type. Lives and dies with the scope, so
    /// `snapshot_scopes` / `join_branches` (which only rewrite
    /// `bindings`) leave it alone. Always empty for non-`Block` scopes.
    pinned: FxHashMap<Name, Ty>,
}

#[derive(Debug)]
struct MethodFrame {
    method_name: Option<String>,
    method_type: Option<MethodType>,
    defined_in: Option<TypeName>,
    has_forwarding_param: bool,
    forward_leading_positional: usize,
    is_singleton_method: bool,
}

/// Saved class nesting returned by [`Context::replace_class_stack`] and
/// handed back to [`Context::restore_class_stack`].
pub struct ClassStackFrames {
    class_stack: Vec<TypeName>,
    cref_stack: Vec<TypeName>,
    class_stack_in_cref: Vec<bool>,
}

/// Tracks class nesting and a scope chain during AST traversal.
pub struct Context {
    class_stack: Vec<TypeName>,
    /// Ruby's cref — the lexical scope constants are defined in and
    /// looked up from. Usually identical to `class_stack`, but a
    /// `Const = Class.new do ... end` block body is walked as `class
    /// Const` for `self` / `def` ownership while its cref stays the
    /// *outer* scope (Ruby: `INNER = 1` inside the block defines
    /// `::INNER`, not `::Const::INNER`). `class_stack_in_cref[i]` says
    /// whether `class_stack[i]` was mirrored here, so `pop_class` can
    /// keep the two aligned.
    cref_stack: Vec<TypeName>,
    class_stack_in_cref: Vec<bool>,
    method_name: Option<String>,
    method_type: Option<MethodType>,
    /// rbs `TypeDef#defined_in` of the enclosing method's first overload.
    /// Threaded alongside `method_type` so a `yield` argument mismatch
    /// (checked against the enclosing method's block signature, not a
    /// separate call target) can still report which class/module owns
    /// the signature being violated.
    defined_in: Option<TypeName>,
    /// `true` when the enclosing `def` declares `(...)`, the
    /// argument-forwarding shorthand. Together with [`method_type`] this
    /// gives the call-side `bar(...)` check the caller's signature to
    /// compare against the callee — Steep's `method_context.forward_arg_type`
    /// (`lib/steep/type_construction.rb` `synthesize_method_call`).
    has_forwarding_param: bool,
    /// Leading positional count on the enclosing `def`, summing
    /// `requireds()` and `optionals()` (`x` and `y = 1` in
    /// `def foo(x, y = 1, ...)`). Forward-arg-type normalisation drops
    /// this many positional slots from the sig head before exposing the
    /// residual to the callee check. Steep's `MethodParams.build`
    /// (`lib/steep/type_inference/method_params.rb:298-339`) consumes
    /// both `:arg` (required) and `:optarg` (optional) into per-param
    /// entries before recording the residual as `forward_arg_type`, so
    /// crema must mirror the combined count. Only meaningful when
    /// `has_forwarding_param` is `true`.
    forward_leading_positional: usize,
    is_singleton_method: bool,
    singleton_class_depth: usize,
    scopes: Vec<Scope>,
    /// Sequential type-env axis for pure-method-call returns. Mirrors
    /// Steep's `TypeEnv#pure_method_calls`: a flat
    /// `structural-call → narrowed Ty` store that lives independently of
    /// the lvar scope chain. Within a method body a pure entry survives
    /// inner scope push/pop and is dropped only by lvar reassignment via
    /// [`PureCallEnv::invalidate_by_lvar`]. It is, however, method-scoped:
    /// [`Context::enter_method`] saves and clears it, [`Context::leave_method`]
    /// restores it, matching Steep's per-method-body `TypeEnv`. Without this
    /// a self-rooted key (`Send(SelfRef, _)`) — which carries no class
    /// identity and is never invalidated by an lvar write — would leak a
    /// narrow into a later method's body. The owner is `crate::type_checker`
    /// — only the type-checking layer reads or writes here; build / lowering
    /// paths leave it empty.
    pure_call_env: PureCallEnv,
    saved_method_frames: Vec<MethodFrame>,
    /// Save stack for the method-scoped `pure_call_env`. One entry is
    /// pushed per [`Context::enter_method`] and popped per
    /// [`Context::leave_method`], so nested `def`s restore correctly.
    saved_pure_call_envs: Vec<PureCallEnv>,
}

impl Context {
    pub fn new() -> Self {
        Context {
            class_stack: vec![],
            cref_stack: vec![],
            class_stack_in_cref: vec![],
            method_name: None,
            method_type: None,
            defined_in: None,
            has_forwarding_param: false,
            forward_leading_positional: 0,
            is_singleton_method: false,
            singleton_class_depth: 0,
            scopes: vec![Scope {
                kind: ScopeKind::Top,
                bindings: FxHashMap::default(),
                bot_rhs_bindings: FxHashSet::default(),
                self_type_override: None,
                pinned: FxHashMap::default(),
            }],
            pure_call_env: PureCallEnv::new(),
            saved_method_frames: Vec::new(),
            saved_pure_call_envs: Vec::new(),
        }
    }

    /// Read access to the pure-call cache. Type-checker code reads this
    /// to decide whether a send has a previously-narrowed return type
    /// stored under a structural key.
    pub fn pure_call_env(&self) -> &PureCallEnv {
        &self.pure_call_env
    }

    /// Mutable access to the pure-call cache. Used by the send-typing
    /// path to write back a pure call's return type and by
    /// `analyze_condition` to install branch-local narrows on pure keys.
    pub fn pure_call_env_mut(&mut self) -> &mut PureCallEnv {
        &mut self.pure_call_env
    }

    pub fn invalidate_const_pure_calls(&mut self) {
        self.pure_call_env.invalidate_const_paths();
    }

    /// The innermost enclosing class/module as a [`TypeName`]. The stack
    /// stores the structured name (parsed once at [`push_class`]) so hot
    /// callers — `current_self_type`, constant resolution — read it without
    /// re-parsing a path string on every reference.
    pub fn current_class_typename(&self) -> Option<&TypeName> {
        self.class_stack.last()
    }

    /// Full lexical class stack, outer-first. Mirrors rbs's
    /// `Resolver::context = nil | [parent, last]` linked list flattened
    /// into a slice — the trailing element is rbs's `last`, the rest
    /// is `parent`, and an empty slice is rbs's `nil`. Used by
    /// `type_checker::inference` to feed `ConstantResolver::resolve`.
    pub fn class_stack(&self) -> &[TypeName] {
        &self.class_stack
    }

    /// Ruby's cref, outer-first — the stack constant reads resolve
    /// against and bare constant writes define into. See the field doc
    /// for where it diverges from [`class_stack`](Self::class_stack).
    pub fn cref_stack(&self) -> &[TypeName] {
        &self.cref_stack
    }

    /// Replace the whole class nesting (synthetic method contexts). The
    /// cref follows the replacement — a synthesized target is a real
    /// class body for every purpose — and is restored with it.
    pub fn replace_class_stack(&mut self, class_stack: Vec<TypeName>) -> ClassStackFrames {
        let in_cref = vec![true; class_stack.len()];
        ClassStackFrames {
            class_stack: std::mem::replace(&mut self.class_stack, class_stack.clone()),
            cref_stack: std::mem::replace(&mut self.cref_stack, class_stack),
            class_stack_in_cref: std::mem::replace(&mut self.class_stack_in_cref, in_cref),
        }
    }

    pub fn restore_class_stack(&mut self, saved: ClassStackFrames) {
        self.class_stack = saved.class_stack;
        self.cref_stack = saved.cref_stack;
        self.class_stack_in_cref = saved.class_stack_in_cref;
    }

    pub fn push_class(&mut self, name: &str, names: &NameTable) {
        let type_name = self.qualified_class_name(name, names);
        self.class_stack.push(type_name);
        self.cref_stack.push(type_name);
        self.class_stack_in_cref.push(true);
    }

    /// Push a class that owns `self` / `def` but is *not* a cref frame:
    /// the `Const = Class.new do ... end` block body. Constant reads and
    /// bare writes inside keep resolving against the enclosing cref.
    pub fn push_class_outside_cref(&mut self, name: &str, names: &NameTable) {
        let type_name = self.qualified_class_name(name, names);
        self.class_stack.push(type_name);
        self.class_stack_in_cref.push(false);
    }

    /// Qualify a declaration name under the enclosing *cref*, not the
    /// enclosing `self`-owner: `class Widget` inside a `Const = Class.new
    /// do ... end` block defines `::Widget` (Ruby's `class` keyword, like
    /// a bare constant write, defines into the cref).
    fn qualified_class_name(&self, name: &str, names: &NameTable) -> TypeName {
        let qualified = match self.cref_stack.last() {
            Some(parent) => format!("{}::{}", names.resolve(parent), name),
            None => format!("::{}", name),
        };
        names.parse_type_name(&qualified)
    }

    /// Push a class whose absolute path is already resolved (leading `::`),
    /// e.g. a rooted declaration name `class ::A::B`. Unlike [`push_class`],
    /// the path is not qualified under the enclosing class.
    pub fn push_class_absolute(&mut self, abs_path: &str, names: &NameTable) {
        let type_name = names.parse_type_name(abs_path);
        self.class_stack.push(type_name);
        self.cref_stack.push(type_name);
        self.class_stack_in_cref.push(true);
    }

    pub fn pop_class(&mut self) {
        self.class_stack.pop();
        if self.class_stack_in_cref.pop() == Some(true) {
            self.cref_stack.pop();
        }
    }

    pub fn enter_method(
        &mut self,
        name: String,
        method_type: Option<MethodType>,
        defined_in: Option<TypeName>,
        is_singleton: bool,
        has_forwarding_param: bool,
        forward_leading_positional: usize,
    ) {
        self.saved_method_frames.push(MethodFrame {
            method_name: self.method_name.clone(),
            method_type: self.method_type.clone(),
            defined_in: self.defined_in,
            has_forwarding_param: self.has_forwarding_param,
            forward_leading_positional: self.forward_leading_positional,
            is_singleton_method: self.is_singleton_method,
        });
        self.method_name = Some(name);
        self.method_type = method_type;
        self.defined_in = defined_in;
        self.has_forwarding_param = has_forwarding_param;
        self.forward_leading_positional = forward_leading_positional;
        self.is_singleton_method = is_singleton;
        self.saved_pure_call_envs
            .push(std::mem::take(&mut self.pure_call_env));
        self.push_scope(ScopeKind::Method);
    }

    pub fn leave_method(&mut self) {
        self.pop_scope();
        if let Some(saved) = self.saved_pure_call_envs.pop() {
            self.pure_call_env = saved;
        }
        if let Some(saved) = self.saved_method_frames.pop() {
            self.method_name = saved.method_name;
            self.method_type = saved.method_type;
            self.defined_in = saved.defined_in;
            self.has_forwarding_param = saved.has_forwarding_param;
            self.forward_leading_positional = saved.forward_leading_positional;
            self.is_singleton_method = saved.is_singleton_method;
        } else {
            self.method_name = None;
            self.method_type = None;
            self.defined_in = None;
            self.has_forwarding_param = false;
            self.forward_leading_positional = 0;
            self.is_singleton_method = false;
        }
    }

    pub fn is_singleton_method(&self) -> bool {
        self.is_singleton_method
    }

    pub fn in_singleton_class(&self) -> bool {
        self.singleton_class_depth > 0
    }

    pub fn enter_singleton_class(&mut self) {
        self.singleton_class_depth += 1;
    }

    pub fn leave_singleton_class(&mut self) {
        debug_assert!(
            self.singleton_class_depth > 0,
            "leave_singleton_class called outside singleton class"
        );
        self.singleton_class_depth -= 1;
    }

    pub fn replace_singleton_class_depth(&mut self, depth: usize) -> usize {
        std::mem::replace(&mut self.singleton_class_depth, depth)
    }

    pub fn method_name(&self) -> Option<&str> {
        self.method_name.as_deref()
    }

    pub fn method_type(&self) -> Option<&MethodType> {
        self.method_type.as_ref()
    }

    pub fn defined_in(&self) -> Option<TypeName> {
        self.defined_in
    }

    /// Returns the caller-side `MethodType` to compare against on a
    /// `bar(...)` call when the enclosing `def` declared `(...)`.
    /// `None` if the def does not use forwarding, or has no RBS sig — in
    /// either case there is no caller signature for the callee check.
    ///
    /// When the def declares leading positionals (`def foo(x, y = 1, ...)`),
    /// the sig's `required_positionals` head is consumed first and the
    /// `optional_positionals` head absorbs the remainder — Steep parity
    /// with `MethodParams.build` (`lib/steep/type_inference/method_params.rb:298-339`),
    /// which advances `positional_params.tail` through both `:arg` and
    /// `:optarg` before recording the residual.
    /// Returning `None` when the total leading count exceeds
    /// `required_positionals.len() + optional_positionals.len()` is a
    /// silent skip (sig-under edge): forwarding compatibility cannot be
    /// judged when the def consumes more leading positionals than the
    /// sig provides, but no diagnostic is emitted from this path.
    pub fn forward_arg_type(&self) -> Option<MethodType> {
        if !self.has_forwarding_param {
            return None;
        }
        let mt = self.method_type.as_ref()?;
        if self.forward_leading_positional == 0 {
            return Some(mt.clone());
        }
        match &mt.type_ {
            FunctionType::Untyped(_) => Some(mt.clone()),
            FunctionType::Typed(f) => {
                let sig_req = f.required_positionals.len();
                let sig_opt = f.optional_positionals.len();
                if self.forward_leading_positional > sig_req + sig_opt {
                    return None;
                }
                let (req_drop, opt_drop) = if self.forward_leading_positional <= sig_req {
                    (self.forward_leading_positional, 0)
                } else {
                    (sig_req, self.forward_leading_positional - sig_req)
                };
                let residual = Function {
                    required_positionals: f.required_positionals[req_drop..].to_vec(),
                    optional_positionals: f.optional_positionals[opt_drop..].to_vec(),
                    rest_positional: f.rest_positional,
                    trailing_positionals: f.trailing_positionals.clone(),
                    required_keywords: f.required_keywords.clone(),
                    optional_keywords: f.optional_keywords.clone(),
                    rest_keyword: f.rest_keyword,
                    return_type: f.return_type,
                };
                Some(MethodType {
                    type_params: mt.type_params.clone(),
                    type_: FunctionType::Typed(residual),
                    block: mt.block.clone(),
                })
            }
        }
    }

    /// Push a new scope onto the chain.
    pub fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope {
            kind,
            bindings: FxHashMap::default(),
            bot_rhs_bindings: FxHashSet::default(),
            self_type_override: None,
            pinned: FxHashMap::default(),
        });
    }

    /// Every lvar visible from the innermost scope, with the binding a
    /// read would see (innermost wins, `Method` is a hard boundary —
    /// same walk as [`Self::lookup_local_variable`]). Feeds
    /// [`Self::pin_local_variables`] on closure entry.
    pub fn visible_local_variables(&self) -> Vec<(Name, Ty)> {
        let mut seen: FxHashSet<Name> = FxHashSet::default();
        let mut out = Vec::new();
        for scope in self.scopes.iter().rev() {
            for (&name, &ty) in &scope.bindings {
                if seen.insert(name) {
                    out.push((name, ty));
                }
            }
            if scope.kind == ScopeKind::Method {
                break;
            }
        }
        out
    }

    /// Push the `Block` scope for a closure being entered, carrying
    /// `pins` (name → pinned type, computed by the caller from
    /// [`Self::visible_local_variables`]).
    pub fn push_block_scope_with_pins(&mut self, pins: FxHashMap<Name, Ty>) {
        self.scopes.push(Scope {
            kind: ScopeKind::Block,
            bindings: FxHashMap::default(),
            bot_rhs_bindings: FxHashSet::default(),
            self_type_override: None,
            pinned: pins,
        });
    }

    /// The pinned type a write to `name` at `depth` must respect, if the
    /// write crosses a closure boundary that pinned it. `depth` follows
    /// prism (`LocalVariableWriteNode#depth`, the number of enclosing
    /// blocks between the write and the owning scope); depth 0 never
    /// pins. Among the `Block` scopes the write crosses, the one nearest
    /// the owning scope wins — that is the first closure entered after
    /// the variable came into existence, matching Steep's
    /// `pin_local_variables` leaving an existing `enforced_type` alone.
    pub fn pinned_local_variable_type(&self, name: Name, depth: u32) -> Option<Ty> {
        if depth == 0 {
            return None;
        }
        let mut remaining = depth;
        let mut target = None;
        for i in (0..self.scopes.len()).rev() {
            if remaining == 0 {
                target = Some(i);
                break;
            }
            if self.scopes[i].kind == ScopeKind::Block {
                remaining -= 1;
            }
        }
        let target = target?;
        self.scopes[target + 1..]
            .iter()
            .find_map(|scope| scope.pinned.get(&name).copied())
    }

    pub fn set_self_type_override(&mut self, ty: Ty) {
        debug_assert!(
            !self.scopes.is_empty(),
            "set_self_type_override called with no scopes"
        );
        if let Some(scope) = self.scopes.last_mut() {
            scope.self_type_override = Some(ty);
        }
    }

    /// Returns the innermost `[self: T]` override, if any scope carries one.
    /// `Method` is a hard boundary (same walk as
    /// [`Self::visible_local_variables`]): a `def` body does not inherit the
    /// self of an enclosing `[self: T]` block. Ruby defines the method on the
    /// lexical class (cref), so the body's self is that class's instance
    /// regardless of the block it was written in. A `Method` scope that
    /// carries its own override (set at `def` entry) is still honoured.
    pub fn current_self_type_override(&self) -> Option<Ty> {
        Self::self_type_override_within(&self.scopes)
    }

    /// The `[self: T]` override in effect at the innermost of `scopes`,
    /// walking outward and stopping at the first `Method` scope.
    fn self_type_override_within(scopes: &[Scope]) -> Option<Ty> {
        for scope in scopes.iter().rev() {
            if scope.self_type_override.is_some() {
                return scope.self_type_override;
            }
            if scope.kind == ScopeKind::Method {
                return None;
            }
        }
        None
    }

    pub fn self_type_override_at_binding(&self, name: Name) -> Option<(Ty, Option<Ty>)> {
        for (index, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(&ty) = scope.bindings.get(&name) {
                let self_type_override = Self::self_type_override_within(&self.scopes[..=index]);
                return Some((ty, self_type_override));
            }
            if scope.kind == ScopeKind::Method {
                return None;
            }
        }
        None
    }

    /// Pop the current scope. Panics if only the top scope remains.
    pub fn pop_scope(&mut self) {
        debug_assert!(self.scopes.len() > 1, "Cannot pop the top scope");
        self.scopes.pop();
    }

    /// Set a variable in the current (innermost) scope.
    ///
    /// Reassigning `name` invalidates every pure-call cache entry whose
    /// key references it — `pure_call_env.invalidate_by_lvar(name)` —
    /// mirroring Steep's lvar-side hook into `pure_node_invalidation`
    /// (`type_env.rb:117-118`). Narrowing-only writes go through
    /// [`Self::enter_narrow`] and intentionally skip this invalidation.
    pub fn set_local_variable(&mut self, name: Name, ty: Ty) {
        self.pure_call_env.invalidate_by_lvar(name);
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(name, ty);
            scope.bot_rhs_bindings.remove(&name);
        }
    }

    /// Set a variable at a specific depth from the innermost scope.
    /// depth=0 is the current scope, depth=1 is one scope up, etc.
    /// Skips past Method boundaries — depth refers to block nesting depth,
    /// not the raw scope stack index.
    pub fn set_local_variable_at_depth(&mut self, name: Name, ty: Ty, depth: u32) {
        if depth == 0 {
            self.set_local_variable(name, ty);
            return;
        }

        // Block writes to an outer-scope lvar invalidate the same pure
        // cache as a top-level reassignment of `name`. Skipping this
        // would let a stale narrow survive a closure write of the same
        // identifier.
        self.pure_call_env.invalidate_by_lvar(name);
        let len = self.scopes.len();
        let mut remaining = depth;
        for i in (0..len).rev() {
            if remaining == 0 {
                self.scopes[i].bindings.insert(name, ty);
                self.scopes[i].bot_rhs_bindings.remove(&name);
                return;
            }
            // An enclosing conditional's narrow (`with_cond_branch`)
            // plants its shadow on the innermost scope regardless of
            // which scope actually owns the name — if this write's
            // target is further out, that shadow is now stale and must
            // not linger, or a same-arm read would find it first
            // (`lookup_local_variable` walks innermost-first) and see
            // the pre-write narrow instead of this write's value.
            self.scopes[i].bindings.remove(&name);
            self.scopes[i].bot_rhs_bindings.remove(&name);
            if self.scopes[i].kind == ScopeKind::Block {
                remaining -= 1;
            }
        }
        debug_assert!(
            false,
            "set_local_variable_at_depth: depth {depth} exceeds the live block-scope nesting; \
             every scope walked past had its stale `name` entry cleared with no target left to \
             write into"
        );
    }

    pub fn set_local_variable_at_depth_from_bot_rhs(&mut self, name: Name, ty: Ty, depth: u32) {
        self.set_local_variable_at_depth(name, ty, depth);
        if ty != Ty::BOTTOM {
            return;
        }
        let len = self.scopes.len();
        let mut remaining = depth;
        for i in (0..len).rev() {
            if remaining == 0 {
                self.scopes[i].bot_rhs_bindings.insert(name);
                return;
            }
            if self.scopes[i].kind == ScopeKind::Block {
                remaining -= 1;
            }
        }
    }

    /// Look up a variable by walking the scope chain from inner to outer.
    /// A `Method` scope is a hard boundary — the walk stops without
    /// crossing into the enclosing top-level / module scope, matching
    /// Ruby's lexical visibility rules for method bodies.
    pub fn lookup_local_variable(&self, name: Name) -> Option<Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(&ty) = scope.bindings.get(&name) {
                return Some(ty);
            }
            if scope.kind == ScopeKind::Method {
                return None;
            }
        }
        None
    }

    pub fn is_bot_rhs_local_variable(&self, name: Name) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.bindings.contains_key(&name) {
                return scope.bot_rhs_bindings.contains(&name);
            }
            if scope.kind == ScopeKind::Method {
                return false;
            }
        }
        false
    }

    /// Sequential-environment substrate, low half of the narrow pair.
    /// Swap the innermost scope's binding for `name` to `ty` and return
    /// an opaque token. The token must be handed back to [`exit_narrow`]
    /// to undo the swap.
    ///
    /// The write lands in the same place an assignment would, so reads
    /// inside the narrow see `ty` via the normal `lookup_local_variable`
    /// path — no overlay layer. Outer-scope bindings are not touched and
    /// reappear naturally when the narrow is unwound.
    ///
    /// Callers in this crate should prefer the TypeChecker-level closure
    /// wrapper (`with_truthy_narrowing` and successors) so the pair is
    /// statically guaranteed to balance; this raw pair exists because the
    /// closure form would force the caller to borrow `&mut Context`
    /// across the body, blocking concurrent access to the surrounding
    /// `TypeChecker`.
    pub fn enter_narrow(&mut self, name: Name, ty: Ty) -> NarrowToken {
        let prev_in_innermost = self
            .scopes
            .last()
            .and_then(|s| s.bindings.get(&name).copied());
        let prev_bot_rhs_in_innermost = self
            .scopes
            .last()
            .map(|s| s.bot_rhs_bindings.contains(&name))
            .unwrap_or(false);
        self.set_local_variable(name, ty);
        NarrowToken {
            name,
            narrowed_ty: ty,
            prev_in_innermost,
            prev_bot_rhs_in_innermost,
        }
    }

    /// Undo an [`enter_narrow`] by restoring the prior innermost binding,
    /// **unless an assignment has overwritten the narrow value during the
    /// narrow's lifetime**. If `name` no longer holds the narrowed type
    /// in the innermost scope (an assignment landed there), the prior
    /// binding is left alone so the assignment's effect survives — this
    /// preserves the long-standing invariant that writes through a narrow
    /// are durable, captured by `test_narrow_assignment_persists_past_exit`.
    pub fn exit_narrow(&mut self, token: NarrowToken) {
        let NarrowToken {
            name,
            narrowed_ty,
            prev_in_innermost,
            prev_bot_rhs_in_innermost,
        } = token;
        if let Some(scope) = self.scopes.last_mut() {
            if scope.bindings.get(&name).copied() != Some(narrowed_ty) {
                return;
            }
            match prev_in_innermost {
                Some(prev) => {
                    scope.bindings.insert(name, prev);
                    if prev_bot_rhs_in_innermost {
                        scope.bot_rhs_bindings.insert(name);
                    } else {
                        scope.bot_rhs_bindings.remove(&name);
                    }
                }
                None => {
                    scope.bindings.remove(&name);
                    scope.bot_rhs_bindings.remove(&name);
                }
            }
        }
    }
}

/// Snapshot of every active scope's bindings, taken by
/// [`Context::snapshot_scopes`] and restored by
/// [`Context::restore_scopes`].
///
/// Used by the if/unless branch-isolation path: the caller snapshots
/// the pre-if env once, evaluates each arm against it, then joins the
/// two arm-after snapshots back into the live scope chain. The snapshot
/// covers all scopes (outer-first order) so that block-level
/// `set_local_variable_at_depth` writes — Ruby's "closure writes the
/// enclosing lvar" semantics — are captured and rolled back per arm,
/// keeping arms order-independent.
#[derive(Clone, Debug)]
pub struct ScopeSnapshot {
    /// Per-scope bindings, outer-first. `per_scope[0]` is the outermost
    /// (Top) frame; `per_scope.last()` is the innermost. The length
    /// equals the live `Context::scopes` length at snapshot time, and
    /// must still match at `restore_scopes` / `join_branches` time
    /// (debug-asserted) — arms are expected to leave the scope stack
    /// balanced.
    per_scope: Vec<FxHashMap<Name, Ty>>,
    per_scope_bot_rhs: Vec<FxHashSet<Name>>,
    /// Pure-call cache at snapshot time. The cache lives on the
    /// `Context` itself (not in `scopes`) so it carries across scope
    /// push/pop, but per-arm snapshots still need to capture it so
    /// branch arms see their pre-arm view and the join can intersect
    /// the per-arm post-state. Mirrors Steep's flat `pure_method_calls`
    /// (`type_env.rb:42-310`).
    pure_call_env: PureCallEnv,
}

impl Context {
    /// Clone every scope's bindings (outer-first) and the pure-call
    /// cache into a [`ScopeSnapshot`].
    pub fn snapshot_scopes(&self) -> ScopeSnapshot {
        ScopeSnapshot {
            per_scope: self.scopes.iter().map(|s| s.bindings.clone()).collect(),
            per_scope_bot_rhs: self
                .scopes
                .iter()
                .map(|s| s.bot_rhs_bindings.clone())
                .collect(),
            pure_call_env: self.pure_call_env.clone(),
        }
    }

    /// Overwrite every scope's bindings with `snapshot`. Any binding
    /// present in the current scope chain but absent from the snapshot
    /// is dropped; any binding in the snapshot replaces the current
    /// one. Requires that the scope depth has not changed since the
    /// snapshot was taken — a mismatch here would silently leave
    /// `zip`'s shorter side in charge, so we panic instead of trusting
    /// `debug_assert` to catch it only in dev builds.
    pub fn restore_scopes(&mut self, snapshot: ScopeSnapshot) {
        assert_eq!(
            self.scopes.len(),
            snapshot.per_scope.len(),
            "restore_scopes called with mismatched scope depth"
        );
        for ((scope, bindings), bot_rhs_bindings) in self
            .scopes
            .iter_mut()
            .zip(snapshot.per_scope)
            .zip(snapshot.per_scope_bot_rhs)
        {
            scope.bindings = bindings;
            scope.bot_rhs_bindings = bot_rhs_bindings;
        }
        self.pure_call_env = snapshot.pure_call_env;
    }

    /// Write the 2-way branch join into every active scope.
    ///
    /// `arm_a` and `arm_b` are each arm's after-env, taken by
    /// `with_cond_branch` — so each already carries every binding the
    /// arm did not touch. Both snapshots must have the same scope depth
    /// as the live scope chain.
    ///
    /// Semantics (mirrors Steep's `TypeEnv#join`,
    /// `lib/steep/type_inference/type_env.rb`):
    ///   - For each scope index, the post-env's name set is the union
    ///     of `arm_a` and `arm_b` names at that scope.
    ///   - When only one arm has the name, two sub-cases apply:
    ///     - If an outer scope (at a smaller index in either snapshot)
    ///       holds the same name, the entry is a transient inner-scope
    ///       binding — typically a narrowing planted on the innermost
    ///       scope by `with_cond_branch`. Dropping it from the joined
    ///       inner scope avoids shadowing the live outer binding with
    ///       `T | nil`.
    ///     - Otherwise the name is truly arm-introduced, and the
    ///       silent arm contributes `Ty::NIL` (Steep's "the unrun
    ///       assignment reads as nil").
    pub fn join_branches(
        &mut self,
        arm_a: &ScopeSnapshot,
        arm_b: &ScopeSnapshot,
        env: crate::definition_builder::ConsultationView,
    ) {
        assert_eq!(
            arm_a.per_scope.len(),
            arm_b.per_scope.len(),
            "join_branches called with mismatched arm scope depths"
        );
        assert_eq!(
            self.scopes.len(),
            arm_a.per_scope.len(),
            "join_branches called with arm scope depth differing from live chain"
        );

        for (i, scope) in self.scopes.iter_mut().enumerate() {
            let a = &arm_a.per_scope[i];
            let b = &arm_b.per_scope[i];
            let mut all_names: FxHashSet<Name> = FxHashSet::default();
            all_names.extend(a.keys().copied());
            all_names.extend(b.keys().copied());

            let mut joined: FxHashMap<Name, Ty> = FxHashMap::default();
            let mut joined_bot_rhs: FxHashSet<Name> = FxHashSet::default();
            for name in all_names {
                let ty_a = a.get(&name).copied();
                let ty_b = b.get(&name).copied();
                match (ty_a, ty_b) {
                    (Some(ta), Some(tb)) => {
                        let ty = crate::types::union_of(ta, tb, env.types());
                        joined.insert(name, ty);
                        if ty == Ty::BOTTOM
                            && arm_a.per_scope_bot_rhs[i].contains(&name)
                            && arm_b.per_scope_bot_rhs[i].contains(&name)
                        {
                            joined_bot_rhs.insert(name);
                        }
                    }
                    (Some(t), None) | (None, Some(t)) => {
                        let shadows_outer =
                            arm_a.per_scope[..i].iter().any(|s| s.contains_key(&name))
                                || arm_b.per_scope[..i].iter().any(|s| s.contains_key(&name));
                        if shadows_outer {
                            // Inner-scope shadow of an outer binding —
                            // most commonly a narrowing planted into
                            // the innermost scope. Drop it so the
                            // outer binding remains visible through
                            // `lookup_local_variable`'s scope walk.
                            continue;
                        }
                        joined.insert(name, crate::types::union_of(t, Ty::NIL, env.types()));
                    }
                    (None, None) => unreachable!(),
                }
            }
            scope.bindings = joined;
            scope.bot_rhs_bindings = joined_bot_rhs;
        }

        // Pure-call cache joins on the same "both-arms keep, one-arm
        // drops" rule but lives outside the scope chain. Each arm's
        // post-arm pure_call_env is in its snapshot; intersect them and
        // union the surviving types via
        // [`PureCallEnv::join_at_branch`]. The receiver's pre-join
        // pure_call_env is discarded — every arm already started from
        // the same pre-arm snapshot, so the joined state replaces it.
        let env_a = arm_a.pure_call_env.clone();
        let env_b = arm_b.pure_call_env.clone();
        self.pure_call_env = env_a.join_at_branch(env_b, env);
    }
}

/// Opaque undo handle produced by [`Context::enter_narrow`]. Carries
/// the name + the narrowed type + the prior innermost binding so
/// [`Context::exit_narrow`] can tell a still-present narrow apart from
/// an assignment that overwrote it.
#[must_use = "drop a NarrowToken without exit_narrow and the narrow leaks"]
pub struct NarrowToken {
    name: Name,
    narrowed_ty: Ty,
    prev_in_innermost: Option<Ty>,
    prev_bot_rhs_in_innermost: bool,
}

/// Opaque undo handle produced by [`Context::enter_pure_narrow`]. The
/// pure-call axis carries no scope hierarchy, so the prior state is
/// just `Option<Ty>` (None = key absent before narrow). Mirrors
/// [`NarrowToken`] in spirit but cannot share an enum without exposing
/// the discriminator at every call site.
#[must_use = "drop a PureNarrowToken without exit_pure_narrow and the narrow leaks"]
pub struct PureNarrowToken {
    key: PureKey,
    narrowed_ty: Ty,
    prev: Option<Ty>,
}

impl Context {
    /// Plant a narrowed type on a pure-call cache entry. Returns a token
    /// that, when handed to [`Self::exit_pure_narrow`], restores the
    /// previous cache entry (or removes the planted one when none
    /// existed). Lvar-side counterpart is [`Self::enter_narrow`].
    pub fn enter_pure_narrow(&mut self, key: PureKey, ty: Ty) -> PureNarrowToken {
        let prev = self.pure_call_env.get(&key);
        self.pure_call_env.set(key.clone(), ty);
        PureNarrowToken {
            key,
            narrowed_ty: ty,
            prev,
        }
    }

    /// Undo an [`Self::enter_pure_narrow`], unless an intervening write
    /// has replaced the narrowed value (in which case the replacement
    /// stays — same durability rule as [`Self::exit_narrow`]).
    pub fn exit_pure_narrow(&mut self, token: PureNarrowToken) {
        let PureNarrowToken {
            key,
            narrowed_ty,
            prev,
        } = token;
        if self.pure_call_env.get(&key) != Some(narrowed_ty) {
            return;
        }
        match prev {
            Some(prev_ty) => self.pure_call_env.set(key, prev_ty),
            None => {
                self.pure_call_env.remove(&key);
            }
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

