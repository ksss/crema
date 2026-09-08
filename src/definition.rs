//! Definition layer — lazy-on-demand memoized `Definition`s on top of
//! the frozen [`Environment`].
//!
//! Mirrors `RBS::DefinitionBuilder` and `RBS::Definition` (see
//! ADR-0017 and ADR-0019). Each class /
//! module declaration produces two `Definition`s (instance side and
//! singleton side); each interface declaration produces one. Populated
//! lazily on the first `build_instance` / `build_singleton` /
//! `build_interface` call and memoized in `Arc`-wrapped caches (ADR-0020).
//! Method resolution walks ancestors via
//! [`AncestorBuilder::instance_ancestors`] / `singleton_ancestors` /
//! `interface_ancestors` and binds the root's type params with
//! [`InstanceAncestors::apply`].
//! ADR-0009 is preserved at the data-shape level: `Definition.methods`
//! stores own-class entries only; no ancestor-merged method tables are
//! pre-built.

use rustc_hash::FxHashMap;
use std::cell::OnceCell;
use std::sync::Arc;

use crate::name::Symbol;
use crate::type_name::TypeName;
use crate::types::{Ty, Type};

pub mod ancestor_builder;
pub mod ancestor_graph;
pub mod constant_resolver;
pub mod lowering_maps;
pub mod method;
pub mod method_builder;
pub mod type_lowering;


pub use ancestor_builder::{
    Ancestor, AncestorBuilder, AncestorSource, InstanceAncestors, OneAncestors, SingletonAncestors,
};
pub use ancestor_graph::{AncestorGraph, Node as AncestorNode};
pub use constant_resolver::{
    ConstantContext, ConstantOrigin, ConstantResolver, ConstantTable, ResolverConstant,
};
pub use lowering_maps::LoweringMaps;
pub use method::{MemberRef, Method, TypeDef};
pub use type_lowering::LoweringEnv;

/// Re-exported from `crate::environment` so the Phase 5a definition
/// layer can present a single import path for its public types.
pub use crate::environment::MixinRef;

/// Type-unit view of a class / module / interface, mirroring
/// `RBS::Definition` shape. One class declaration produces two
/// `Definition` instances (one for the instance type, one for the
/// singleton type); one interface declaration produces one.
///
/// Populate divergence from rbs: `methods` holds own-class entries
/// only. ADR-0009 keeps method lookup as a lazy walk over the
/// linearized `ancestors`; the ancestor-merged map rbs precomputes for
/// LSP is not built here.
#[derive(Debug, Clone)]
pub struct Definition {
    pub type_name: TypeName,
    /// `Ty::ClassInstance` / `Ty::ClassSingleton` / `Ty::Interface`
    /// — the type this `Definition` is the view of. Mirrors
    /// `RBS::Definition#self_type`.
    pub self_type: Ty,
    /// Linearized ancestor view: `Instance` for class & interface
    /// instance sides, `Singleton` for class & module singleton sides.
    /// Mirrors `RBS::Definition#ancestors`; rbs uses
    /// `interface_ancestors` for the interface entry point, so its
    /// `InstanceAncestors` variant covers both class instance and
    /// interface views (the variant name follows rbs).
    ///
    /// Lazy-populated via [`Definition::ancestors`]: full recursive
    /// linearization is the heaviest ancestor op (3-10% of build phase
    /// on rbs-scale fixtures), and the field has no in-tree caller
    /// outside `Definition::ancestors` itself — populating eagerly
    /// pays for an LSP / `crema ancestors` future use that hasn't
    /// landed. `OnceCell` keeps the rbs port shape (field present,
    /// same `Ancestors` enum) while making the cost zero when nothing
    /// reads it.
    ///
    /// `pub(crate)` so the raw cell is not part of the crate's public
    /// API surface — external callers go through the accessor, which
    /// guarantees variant dispatch via `self_type`. Same-crate tests
    /// can still observe init state directly for invariant pinning.
    pub(crate) ancestors: OnceCell<Arc<Ancestors>>,
    /// Own methods on this type-view. Singleton-side `Definition`s
    /// carry the class's singleton methods, instance-side carries
    /// instance methods, interface carries declared methods.
    pub methods: FxHashMap<Symbol, Method>,
    /// Own instance variables of this type-view. On instance-side
    /// `Definition`s, holds `@foo: T` declarations and instance-side
    /// `attr_*` ivar backings. On singleton-side `Definition`s, holds
    /// `self.@foo: T` declarations and singleton-side `attr_*` ivar
    /// backings (rbs `build_singleton0` parks ClassInstanceVariable
    /// here, not in `class_variables`). Own-only, see [`Variable`].
    pub instance_variables: FxHashMap<Symbol, Variable>,
    /// Own class variables (`@@foo: T`). Populated on instance-side
    /// `Definition`s only — singleton-side `Definition`s carry an empty
    /// map, matching rbs `build_singleton0`. Own-only, see [`Variable`].
    pub class_variables: FxHashMap<Symbol, Variable>,
}

/// Linearized ancestor view for a `Definition`. Wraps the existing
/// `InstanceAncestors` / `SingletonAncestors`, mirroring rbs's
/// `InstanceAncestors | SingletonAncestors` union — interfaces use
/// the `Instance` variant because `RBS::DefinitionBuilder#build_interface`
/// stores `ancestor_builder.interface_ancestors` (an `InstanceAncestors`).
#[derive(Debug, Clone)]
pub enum Ancestors {
    Instance(InstanceAncestors),
    Singleton(SingletonAncestors),
}

impl Definition {
    /// Lazily resolve the linearized `ancestors`. The first call picks
    /// `instance_ancestors` / `singleton_ancestors` / `interface_ancestors`
    /// from `builder` based on `self.self_type`'s variant and stores the
    /// result in the `OnceCell`; subsequent calls return the cached
    /// `Arc<Ancestors>` without recomputing.
    ///
    /// # Contract
    ///
    /// `builder` **must** be derived from the same [`Environment`] that
    /// produced this `Definition`. Crossing builders is undefined:
    /// `self.self_type` is a [`TypeTable`]-local index, so resolving it
    /// against a foreign builder's `TypeTable` may return an unrelated
    /// `Type` or panic on out-of-range. The first such call writes the
    /// wrong `Arc<Ancestors>` into the [`OnceCell`], which then poisons
    /// every subsequent `ancestors` lookup on this `Definition` for the
    /// program's lifetime — `OnceCell::get_or_init` never re-runs.
    ///
    /// The compiler cannot enforce this binding: `Definition` deliberately
    /// does not carry an env reference, mirroring `RBS::Definition` shape
    /// (ADR-0019). Caller discipline is the only line of defense. In
    /// practice, always pass `definition_builder.ancestor_builder()`
    /// where `definition_builder` is the [`DefinitionBuilder`] that
    /// produced this `Definition`.
    ///
    /// [`Environment`]: crate::environment::Environment
    /// [`TypeTable`]: crate::types::TypeTable
    /// [`DefinitionBuilder`]: crate::definition_builder::DefinitionBuilder
    pub fn ancestors(&self, builder: &AncestorBuilder) -> &Arc<Ancestors> {
        self.ancestors.get_or_init(|| {
            let resolved = builder.types().resolve(self.self_type);
            let ancestors = match resolved {
                Type::ClassInstance { .. } => {
                    Ancestors::Instance(builder.instance_ancestors(&self.type_name).as_ref().clone())
                }
                Type::ClassSingleton { .. } => {
                    Ancestors::Singleton(
                        builder.singleton_ancestors(&self.type_name).as_ref().clone(),
                    )
                }
                Type::Interface { .. } => {
                    Ancestors::Instance(
                        builder.interface_ancestors(&self.type_name).as_ref().clone(),
                    )
                }
                _ => unreachable!(
                    "Definition.self_type must be ClassInstance / ClassSingleton / Interface, got {:?}",
                    resolved
                ),
            };
            Arc::new(ancestors)
        })
    }
}

/// Type-view variable record, port of `RBS::Definition::Variable`
/// (rbs/lib/rbs/definition.rb:5-28). Mirrors the rbs shape
/// (`parent_variable` / `type` / `declared_in` / `source`) so call-site
/// type-checking can ask "what's the type of `@foo` on this class?"
/// with the same field structure rbs / Steep use.
///
/// **Populate divergence (own-only, ADR-0009 / ADR-0019 extension).**
/// rbs eagerly merges ancestor `instance_variables` / `class_variables`
/// into each subclass's `Definition` and pre-applies `crate::substitution::Substitution` at
/// merge time. crema does not. The per-`Definition` `instance_variables`
/// / `class_variables` holds **own-class declarations only**; the
/// linearized ancestor chain is walked at lookup time
/// (`definition_builder::lookup_instance_variable_with_args`) and the
/// ancestor.args -> target.type_params substitution is baked along the
/// way, just like the method-side `lookup_instance_method_with_args`.
/// This preserves rbs's struct shape while keeping populate cost flat
/// at preflight scale, matching the method-layer divergence ADR-0009
/// set up.
///
/// `parent_variable` retains the rbs semantics for **same-class
/// duplicate detection**: when `insert_variable` is called twice for
/// the same name within one Definition (e.g. two `@foo: T` lines in
/// the same class body), the second call's `parent_variable` points at
/// the first. The actual diagnostic raise is deferred to a follow-up
/// todo (`mid_variable_duplicate_validation.md`); this struct only
/// carries the chain.
#[derive(Debug, Clone)]
pub struct Variable {
    pub parent_variable: Option<Box<Variable>>,
    pub ty: Ty,
    pub declared_in: TypeName,
    pub source: VariableSource,
}

/// Source-AST origin for a [`Variable`], port of rbs
/// `Definition::Variable#source` union (rbs/sig/definition.rbs:9-19).
/// Keeps the rbs split between signature-side declarations
/// (`AST::Members::*`) and inline-side declarations
/// (`AST::Ruby::Members::*`) so downstream diagnostics (and the
/// follow-up duplicate-validation todo) can branch on the exact AST
/// shape — mirroring rbs's `case ... when AST::Members::InstanceVariable`
/// / `when AST::Ruby::Members::InstanceVariableMember` branching in
/// `definition_builder.rb#validate_variable`.
#[derive(Debug, Clone)]
pub enum VariableSource {
    AttrReader(Arc<crate::ast::members::AttrReaderMember>),
    AttrWriter(Arc<crate::ast::members::AttrWriterMember>),
    AttrAccessor(Arc<crate::ast::members::AttrAccessorMember>),
    InstanceVariable(Arc<crate::ast::members::InstanceVariableMember>),
    ClassInstanceVariable(Arc<crate::ast::members::ClassInstanceVariableMember>),
    ClassVariable(Arc<crate::ast::members::ClassVariableMember>),
    RubyAttrReader(Arc<crate::ast::ruby::members::AttrReaderMember>),
    RubyAttrWriter(Arc<crate::ast::ruby::members::AttrWriterMember>),
    RubyAttrAccessor(Arc<crate::ast::ruby::members::AttrAccessorMember>),
    RubyInstanceVariable(Arc<crate::ast::ruby::members::InstanceVariableMember>),
    /// crema extension, absent from the rbs `Definition::Variable#source`
    /// union: a declaration synthesized from an unconditional top-level
    /// `@ivar = param` assignment in `initialize` (Sorbet-style instance
    /// variable inference). Carries the originating Ruby `initialize`
    /// `DefMember`; the per-assignment fact lives in its
    /// `ivar_param_pairs`.
    InferredInitializeParam(Arc<crate::ast::ruby::members::DefMember>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableDuplicationKind {
    Instance,
    ClassInstance,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDuplication {
    pub kind: VariableDuplicationKind,
    pub type_name: TypeName,
    pub variable_name: Symbol,
    pub location: Option<crate::location::SourceLocation>,
    /// File-bearing source location for inline-Ruby variable sources.
    /// `None` for sig-side sources where file is not attached to the AST node.
    pub ruby_source_location: Option<crate::location::RubyLocation>,
}
