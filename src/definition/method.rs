//! `Method` — port of `RBS::Definition::Method` (`lib/rbs/definition.rb:30-207`).
//!
//! The per-class resolved method definition produced by the build phase.


use std::sync::Arc;

use crate::ast::annotation::Annotation;
use crate::ast::members::{
    AliasMember, AttrAccessorMember as AstAttrAccessor, AttrReaderMember as AstAttrReader,
    AttrWriterMember as AstAttrWriter, MethodDefinitionMember as MethodDefinition,
};
use crate::ast::ruby::members::{
    AttrAccessorMember as RubyAttrAccessorMember, AttrReaderMember as RubyAttrReaderMember,
    AttrWriterMember as RubyAttrWriterMember, DefMember as RubyDefMember,
};
use crate::type_name::TypeName;
use crate::types::{MethodType, TypeTable, Visibility};

/// Mirrors the duck union `RBS::Definition::Method::method_member`
/// (`sig/definition.rbs:27-28`). One of the 7 AST member types that can
/// originate a method type definition.
///
/// rbs's `TypeDef#member` is the source AST node from which the resolved
/// overload was lowered. Each variant carries an `Arc` so the AST owner
/// (the enclosing class declaration) is not required to outlive the
/// resolved `TypeDef`.
#[derive(Debug, Clone)]
pub enum MemberRef {
    Method(Arc<MethodDefinition>),
    AttrReader(Arc<AstAttrReader>),
    AttrWriter(Arc<AstAttrWriter>),
    AttrAccessor(Arc<AstAttrAccessor>),
    RubyDef(Arc<RubyDefMember>),
    RubyAttrReader(Arc<RubyAttrReaderMember>),
    RubyAttrWriter(Arc<RubyAttrWriterMember>),
    RubyAttrAccessor(Arc<RubyAttrAccessorMember>),
    /// `RBS::AST::Members::Alias`. Carried in a bucket's `originals` so
    /// the bucket-iter path (`Sorter::each_strongly_connected_component`)
    /// can resolve `alias new_name old_name` against the same bucket.
    /// Aliases never reach `lower_member_to_method` — they short-circuit
    /// to a clone of the target `Method` at bucket-iter time.
    Alias(Arc<AliasMember>),
    /// crema-only: methods crema synthesises rather than reads from an
    /// AST member. Today `synthesize_new_from_initialize` /
    /// `synthesize_untyped_new` produce `Class#new` overloads from
    /// `#initialize` or fall back to an untyped accept-all signature.
    /// rbs reaches the same shape through `Class#new` declared in
    /// `core/builtin.rbs` (a real `MethodDefinition` AST node); the
    /// synthesised variant exists because crema's build phase does not
    /// load the core.rbs `Class#new`. A follow-up todo will drop this
    /// path in favour of the rbs declaration.
    Synthesized,
}

/// Mirrors `RBS::Definition::Method::TypeDef` (`lib/rbs/definition.rb:31-97`).
///
/// One overload after build-phase resolution: the `MethodType`, the AST
/// member it was lowered from, and the class identity needed for
/// `super_method` resolution.
#[derive(Debug, Clone)]
pub struct TypeDef {
    /// rbs field `type`. Renamed to `type_` because `type` is a Rust keyword.
    pub type_: MethodType,
    /// rbs field `member`. The source AST node.
    pub member: MemberRef,
    /// rbs field `defined_in`. The class that declared this overload.
    pub defined_in: TypeName,
    /// rbs field `implemented_in`. The class that implements this
    /// overload, or `None` for declarations on interfaces (which only
    /// declare). For class / module members this is the same as
    /// `defined_in`.
    pub implemented_in: Option<TypeName>,
    /// rbs field `member_annotations`. Uniformly set to the enclosing
    /// `Method::annotations` on every TypeDef of that Method (rbs
    /// `define_method` lines 966-968). Reading per-TypeDef is therefore
    /// equivalent to reading `Method::annotations` — the slot exists so
    /// `each_annotation` (rbs `definition.rb:89-94`) can yield both
    /// `member_annotations` and `overload_annotations` from a single
    /// `TypeDef` handle without dragging the parent `Method` along.
    ///
    /// The uniform invariant is re-applied at lookup time: when the
    /// lookup walk merges defs across an ancestor boundary (e.g. an
    /// overloading-only child + canonical parent), every entry is
    /// rewritten with the merged `Method.annotations`. See
    /// `definition_builder::merge_inherited_annotations`.
    pub member_annotations: Vec<Annotation>,
    /// rbs field `overload_annotations`. Annotations attached to the
    /// individual overload (the per-`|` rhs in a method signature).
    /// Carried per-`TypeDef` because each `|` arm can have its own
    /// annotation set; `member_annotations` is the same for every
    /// TypeDef in a `Method`, this slot is what makes them differ.
    pub overload_annotations: Vec<Annotation>,
}

impl TypeDef {
    /// Convenience constructor with empty annotation slots. Build-phase
    /// callers (`lower_member_to_method` /
    /// `lower_bucket_defn_to_method`) fill `overload_annotations` and
    /// `member_annotations` after construction, mirroring rbs's
    /// `TypeDef.new(...).tap { |t| t.overload_annotations.replace(...) }`
    /// pattern (`definition_builder.rb:733-742`).
    pub fn new(
        type_: MethodType,
        member: MemberRef,
        defined_in: TypeName,
        implemented_in: Option<TypeName>,
    ) -> Self {
        TypeDef {
            type_,
            member,
            defined_in,
            implemented_in,
            member_annotations: Vec::new(),
            overload_annotations: Vec::new(),
        }
    }
}

impl MemberRef {
    /// Source location of the originating AST member, when available.
    /// Returns `None` only for `Synthesized` methods (no AST anchor).
    /// `RubyDef`'s `DefMember.location` is a `PrismByteRange` (always
    /// present, never optional), widened the same way `attr_*` members
    /// are. Ruby-side members expose their `PrismByteRange` as a
    /// byte-only `LocationRange` (char offsets zeroed — JSONL diagnostic
    /// emit reads `start_byte`/`end_byte` and derives `line` from the
    /// file source, so char offsets are not consumed on this path).
    pub fn location(&self) -> Option<crate::location::LocationRange> {
        match self {
            MemberRef::Method(m) => m.location.map(|l| l.range),
            MemberRef::AttrReader(r) => r.location.map(|l| l.range),
            MemberRef::AttrWriter(w) => w.location.map(|l| l.range),
            MemberRef::AttrAccessor(a) => a.location.map(|l| l.range),
            MemberRef::Alias(a) => a.location.map(|l| l.range),
            MemberRef::RubyAttrReader(r) => Some(ruby_attribute_location(&r.attribute)),
            MemberRef::RubyAttrWriter(w) => Some(ruby_attribute_location(&w.attribute)),
            MemberRef::RubyAttrAccessor(a) => Some(ruby_attribute_location(&a.attribute)),
            MemberRef::RubyDef(d) => Some(ruby_byte_range_location(d.location)),
            MemberRef::Synthesized => None,
        }
    }

    pub fn source_file(&self) -> Option<crate::name::Name> {
        match self {
            MemberRef::Method(m) => m.source_file,
            MemberRef::AttrReader(r) => r.source_file,
            MemberRef::AttrWriter(w) => w.source_file,
            MemberRef::AttrAccessor(a) => a.source_file,
            MemberRef::Alias(a) => a.source_file,
            MemberRef::RubyAttrReader(r) => r.attribute.source_file,
            MemberRef::RubyAttrWriter(w) => w.attribute.source_file,
            MemberRef::RubyAttrAccessor(a) => a.attribute.source_file,
            MemberRef::RubyDef(d) => d.source_file,
            MemberRef::Synthesized => None,
        }
    }
}

/// Widen the byte-only `PrismByteRange` carried by an inline
/// `AttributeMember` to a `LocationRange`. Char offsets are zeroed —
/// diagnostic emit (`Diagnostic::to_json_value_with_line`) derives the
/// line number from `start_byte` and the file source, and the JSONL
/// payload writes `start_byte` / `end_byte` directly.
pub(crate) fn ruby_attribute_location(
    attribute: &crate::ast::ruby::members::AttributeMember,
) -> crate::location::LocationRange {
    ruby_byte_range_location(attribute.location)
}

/// Same widening as [`ruby_attribute_location`], for callers (like
/// `MemberRef::RubyDef`) that already have a bare `PrismByteRange`
/// rather than an `AttributeMember` to borrow it from.
pub(crate) fn ruby_byte_range_location(
    range: crate::ast::ruby::PrismByteRange,
) -> crate::location::LocationRange {
    let (start_byte, end_byte) = range;
    // start_char / end_char are placeholders — callers on this path
    // (dup-diagnostic emit) read only bytes and file. New consumers wanting char offsets must derive
    // them from the source buffer via `line_index`, not read these zeros.
    crate::location::LocationRange::new(0, start_byte, 0, end_byte)
}

/// Mirrors `RBS::Definition::Method`.
///
/// The method name is **not** stored here — the surrounding
/// `HashMap<Symbol, Method>` (analogous to rbs's `Hash[Symbol, Method]`)
/// carries identity at its key.
#[derive(Debug, Clone)]
pub struct Method {
    /// rbs field `defs`. Each entry wraps one `MethodType` overload with
    /// its source AST member and the class identity it was declared in.
    pub defs: Vec<TypeDef>,
    /// rbs field `accessibility`. Effective visibility after resolving
    /// member-level modifiers and surrounding `private` / `public`
    /// visibility-members. Always a concrete `Public` / `Private` — the
    /// RBS `Unspecified` state is resolved against the enclosing scope
    /// at load time.
    pub accessibility: Visibility,
    /// rbs field `super_method`. The same-name `Method` on the parent
    /// class's instance side, used for `super` call resolution. Slot
    /// reserved as `None` for now; the build-phase wiring (rbs's
    /// `define_method` super_method chain) is a follow-up.
    pub super_method: Option<Arc<Method>>,
    /// rbs field `alias_of`. When this method was introduced via the
    /// `alias new_name old_name` syntax, this points to the resolved
    /// `Method` of `old_name`. Slot reserved as `None` for now; the
    /// alias expansion currently produces a separate `Method` and is
    /// rewired by a follow-up.
    pub alias_of: Option<Arc<Method>>,
    /// rbs field `alias_member`. The originating `AST::Members::Alias`
    /// node when this method was introduced via the `alias` syntax.
    /// Slot reserved as `None` for now; see `alias_of`.
    pub alias_member: Option<Arc<AliasMember>>,
    /// rbs field `annotations`. Union of annotations given to `def`s
    /// and `alias` at the member level (the `%a{...}` lines that
    /// precede a `def` / `alias` / `attr_*` declaration). Does NOT
    /// include per-overload `%a{...}` (those live in
    /// `TypeDef::overload_annotations`). For overloading-extras, rbs
    /// concats each extras member's annotations onto this list
    /// (`definition_builder.rb:963`).
    ///
    /// At lookup-walk time, when the walker merges defs across an
    /// ancestor boundary, the running `annotations` is extended by each
    /// parent ancestor's `annotations` in child→parent order. This
    /// differs from rbs's `[parent, child]` ordering used in
    /// `define_method`'s `when nil` branch (lines 891 + 963), but set
    /// semantics are equivalent and the only consumer to date
    /// (`Method::is_pure`) tests `contains` rather than position. See
    /// `definition_builder::merge_inherited_annotations`.
    pub annotations: Vec<Annotation>,
}

impl Method {
    /// Build-phase constructor for a freshly resolved method. Populates
    /// the rbs-aligned `super_method` / `alias_of` / `alias_member` /
    /// `annotations` slots with their default-empty values; callers fill
    /// them after construction (mirrors rbs's
    /// `Method.new(...).tap { |m| m.annotations.replace(...) }` pattern,
    /// `definition_builder.rb:713-723`). `super_method` and `alias_of`
    /// wiring is still a follow-up todo.
    ///
    /// Centralising the defaults keeps the 6+ build sites in sync when
    /// future port work widens the struct shape.
    pub fn from_defs(defs: Vec<TypeDef>, accessibility: Visibility) -> Self {
        Method {
            defs,
            accessibility,
            super_method: None,
            alias_of: None,
            alias_member: None,
            annotations: Vec::new(),
        }
    }

    /// Build-phase placeholder for an alias whose target is resolved
    /// lazily at lookup time. The lookup walk detects this shape via
    /// [`Self::is_unresolved_alias`], re-walks the same ancestor chain
    /// for the alias `old_name`, and constructs a resolved Method whose
    /// `defs` are the target's overloads with `defined_in` /
    /// `implemented_in` rewritten to the alias-declaring class (mirrors
    /// rbs `define_method` lines 715-717).
    ///
    /// The accessibility field carries the alias-side accessibility
    /// (special_accessibility for names like `initialize`, otherwise
    /// `Visibility::Public` as a placeholder); the lookup-time path
    /// overrides it with the target's accessibility when no special
    /// accessibility applies.
    pub fn unresolved_alias(alias_member: Arc<AliasMember>, accessibility: Visibility) -> Self {
        Method {
            defs: Vec::new(),
            accessibility,
            super_method: None,
            alias_of: None,
            alias_member: Some(alias_member),
            annotations: Vec::new(),
        }
    }

    /// True when this Method is the build-phase unresolved-alias
    /// placeholder produced by [`Self::unresolved_alias`]. Centralises
    /// the sentinel shape so lookup-side resolvers don't repeat the
    /// `alias_member.is_some() && alias_of.is_none() && defs.is_empty()`
    /// triplet at every call site.
    pub fn is_unresolved_alias(&self) -> bool {
        self.alias_member.is_some() && self.alias_of.is_none() && self.defs.is_empty()
    }

    /// Record `includer` as the implementer of every def, for a method
    /// picked up from an interface entry during the ancestor walk.
    ///
    /// An interface declares without implementing, so `build_interface`
    /// leaves `implemented_in: None`. rbs resolves this when the
    /// interface is mixed in: `import_methods` passes
    /// `defined_in: interface.name, implemented_in: module_name`
    /// (`definition_builder.rb:671-678`), naming the class or module
    /// that included it. crema walks ancestors at lookup time instead of
    /// merging definitions at build time, so the same stamp happens here
    /// — `includer` is the chain entry the interface was reached
    /// through.
    ///
    /// A module's own methods are *not* restamped this way: rbs gives
    /// them `implemented_in: module_name` (their own module) and the
    /// ancestor merge never rewrites it, so including a module leaves
    /// the implementation with the module.
    pub fn stamp_interface_implementer(&mut self, includer: TypeName) {
        for td in self.defs.iter_mut() {
            td.implemented_in = Some(includer);
        }
    }

    /// Convenience: iterate over the `MethodType` payloads of `defs` in
    /// declaration order. Many callers (subtyping, calls, lookup) only
    /// care about the type signatures and not about the `TypeDef`
    /// metadata; this keeps those call sites concise without forcing
    /// every reader to unpack `td.type_` inline.
    pub fn method_types(&self) -> impl Iterator<Item = &MethodType> {
        self.defs.iter().map(|td| &td.type_)
    }

    /// First `MethodType` overload, when one exists. Equivalent to
    /// `method_types().next()`; the dedicated name reads better at the
    /// call sites that pull a representative arity / return-type signature
    /// for diagnostics or arity-mismatch checks.
    pub fn first_method_type(&self) -> Option<&MethodType> {
        self.defs.first().map(|td| &td.type_)
    }

    /// Compose every overload into a single `MethodType` so a method body can
    /// be type-checked against the union of all declared signatures, matching
    /// Steep's `for_new_method` (`lib/steep/type_construction.rb:158-163`).
    /// Returns `None` when `defs` is empty (`unresolved_alias` placeholder).
    ///
    /// Reduces left-to-right via `MethodType::unify_overload`, so the result
    /// keeps the declaration order of overloads as the left-folding seed.
    pub fn unified_method_type(&self, types: &TypeTable) -> Option<MethodType> {
        let mut iter = self.method_types();
        let first = iter.next()?.clone();
        Some(iter.fold(first, |acc, next| acc.unify_overload(next, types)))
    }

    /// True when every def site originated from an `AST::Members::
    /// MethodDefinition` whose source carried the `...` trailer (the
    /// member-level `overloading?` flag). Matches rbs's `case nil`
    /// branch of `define_method` (`lib/rbs/definition_builder.rb:916-934`)
    /// where the absence of any canonical `def` triggers ancestor-chain
    /// inheritance at lookup time.
    ///
    /// Aliases short-circuit to `false`: an `alias` syntax has no `...`
    /// trailer of its own. Even when the aliased target's defs all came
    /// from overloading members (which the cloned defs preserve via the
    /// `member` back-ref), the alias is itself a fresh canonical
    /// declaration.
    ///
    /// Empty `defs` is the `def foo: ...` form with zero `|` overloads
    /// — every contribution to this method is an extras-only def. There
    /// is no canonical typed overload in this class, so the lookup walk
    /// must continue to ancestors. Treat as overloading.
    ///
    /// Ruby (inline) members follow `RBS::AST::Ruby::Members::DefMember
    /// #overloading?` (`lib/rbs/ast/ruby/members.rb:524-533, 581-583`):
    /// the def is "overloading" when its annotation carries the `| ...`
    /// trailer (`dot3_location`) or when it is fully empty — the latter
    /// mirrors `define_method:881`'s "inherit from parent if any" path.
    /// Attribute members have no `...` form so they short-circuit to
    /// `false`.
    pub fn is_overloading(&self) -> bool {
        if self.alias_member.is_some() {
            return false;
        }
        if self.defs.is_empty() {
            return true;
        }
        self.defs.iter().all(|td| match &td.member {
            MemberRef::Method(md) => md.overloading,
            MemberRef::RubyDef(rd) => rd.method_type.overloading() || rd.method_type.is_empty(),
            _ => false,
        })
    }

    /// Drop the `(?) -> untyped` defs that `lower_member_to_method`
    /// gives an unannotated Ruby `def`, but only when a typed def is
    /// present to take their place.
    ///
    /// rbs `definition_builder.rb:884` (`original.method_type.empty? &&
    /// existing_method`) *replaces* an unannotated def's defs with the
    /// declared ones rather than appending to them. crema assembles
    /// defs from two directions, so both need the same replacement:
    ///
    /// - within one bucket, where a `sig/` declaration and an inline
    ///   `def` for the same method both contribute (the inline def
    ///   arrives as an overloading extras member, splices to the
    ///   *front*, and would otherwise shadow the declared signature)
    /// - across the ancestor walk, where an unannotated override
    ///   inherits its parent's signature
    ///
    /// Call it after the defs are joined. Leaving the placeholder in
    /// place would both swallow argument errors (the `(?)` arm matches
    /// any call) and risk `UnresolvedOverloading` from the extra arm.
    ///
    /// When *every* def is a placeholder the replacement still applies,
    /// it just has nothing typed to land on: a chain of unannotated
    /// overrides keeps the last one, which the joins order as the
    /// furthest ancestor — the class that introduced the name and, in
    /// rbs, the one every override's `defined_in` ends up naming.
    /// Keeping all of them would report a single call site as a
    /// reference to every class in the chain.
    ///
    /// The dropped placeholder is not forgotten: rbs pairs the same
    /// replacement with `defn.update(implemented_in: implemented_in)`
    /// (`definition_builder.rb:888`), keeping `defined_in` on whoever
    /// declared the type while moving `implemented_in` to the class
    /// whose unannotated `def` is what actually runs. The surviving
    /// defs therefore inherit the dropped placeholder's owner as their
    /// implementer.
    pub fn drop_unannotated_placeholder_defs(&mut self) {
        fn is_placeholder(td: &TypeDef) -> bool {
            matches!(&td.member, MemberRef::RubyDef(rd) if rd.method_type.is_empty())
        }
        // The joins append ancestors, so `defs[0]` is the most derived
        // entry — the override whose body a call actually reaches. Read
        // its `implemented_in` first: a walk down a chain of unannotated
        // overrides calls this once per join, and every join past the
        // first sees a placeholder that already carries the implementer
        // from the join before it.
        let implementer = self
            .defs
            .first()
            .filter(|td| is_placeholder(td))
            .map(|td| td.implemented_in.unwrap_or(td.defined_in));
        if self.defs.iter().any(|td| !is_placeholder(td)) {
            self.defs.retain(|td| !is_placeholder(td));
        } else if self.defs.len() > 1 {
            self.defs.drain(..self.defs.len() - 1);
        }
        if let Some(implementer) = implementer {
            for td in self.defs.iter_mut() {
                td.implemented_in = Some(implementer);
            }
        }
    }

    /// True iff this method is recognised as **pure** by ADR-0024 — the
    /// narrowed return value can be cached and reused for a structurally
    /// identical call. Two of the three pure forms are recognised here:
    ///
    /// - every def site is an attribute reader (`AttrReader`,
    ///   `AttrAccessor`, or their inline-Ruby counterparts); or
    /// - every other def site carries a `%a{pure}` annotation.
    ///
    /// The bucket name must not end with `=`: the writer-half of an
    /// `AttrAccessor` lives in a separate `name=` bucket whose `member`
    /// is also `AttrAccessor`, so the name check is what excludes it.
    ///
    /// `all` over `defs` mirrors Steep's `method_decls.all?`
    /// (`lib/steep/type_inference/method_call.rb:129-141`): an attribute
    /// is reader-pure by variant, anything else is pure only when
    /// `each_annotation` yields a `"pure"`. crema's `each_annotation`
    /// equivalent is `member_annotations` followed by
    /// `overload_annotations`, both wired by `mid_pure_annotation_wiring`.
    pub fn is_pure(&self, name: crate::name::Symbol, names: &crate::name::NameTable) -> bool {
        if self.defs.is_empty() {
            return false;
        }
        if names.resolve(name).ends_with('=') {
            return false;
        }
        self.defs.iter().all(|td| match td.member {
            MemberRef::AttrReader(_)
            | MemberRef::AttrAccessor(_)
            | MemberRef::RubyAttrReader(_)
            | MemberRef::RubyAttrAccessor(_) => true,
            _ => td
                .member_annotations
                .iter()
                .chain(td.overload_annotations.iter())
                .any(|a| a.string == names.intern_symbol("pure")),
        })
    }
}

/// `%a{crema:method_missing}` — stamped by infusion synthesis on a
/// singleton method that only exists at runtime through
/// `method_missing` (ActionMailer actions). Ruby resolves such a call
/// *after* the whole singleton ancestry, so the definition builder
/// drops the stamped member whenever an ancestor defines the name for
/// real (`DefinitionBuilder::drop_method_missing_members_shadowed_by_ancestors`).
/// An rbs annotation is used rather than a new `TypeDef` field so the
/// rbs mirror shapes stay untouched; the marker rides
/// `Method::annotations` like `%a{deprecated}` does.
pub const METHOD_MISSING_ANNOTATION: &str = "crema:method_missing";

pub fn has_method_missing_annotation(
    annotations: &[Annotation],
    names: &crate::name::NameTable,
) -> bool {
    annotations
        .iter()
        .any(|a| names.resolve(a.string) == METHOD_MISSING_ANNOTATION)
}

/// Scan `annotations` for the first entry matching `%a{deprecated}`
/// or `%a{deprecated: <message>}` and return its optional message
/// trailer. `None` when nothing matches; `Some(None)` for the bare
/// form; `Some(Some(msg))` when a colon-separated message is
/// present.
///
/// Anchored match on `"deprecated"` (Steep uses a non-anchored regex
/// but crema pins the prefix so identifiers like `not_deprecated`
/// stay silent). Whitespace immediately following the colon is
/// trimmed, matching Steep's `deprecated(:\s*(?<message>.+))?`
/// (`lib/steep/annotations_helper.rb:7`).
///
/// Mirrors Steep's `AnnotationsHelper.deprecated_annotation?`
/// (`lib/steep/annotations_helper.rb:5-16`). The `steep:deprecated`
/// alias (line 10 in Steep) is intentionally omitted — crema has no
/// `steep:` namespacing story yet, so recognising it would surface
/// annotations that Steep alone would flag.
pub fn deprecated_annotation(
    annotations: &[Annotation],
    names: &crate::name::NameTable,
) -> Option<Option<String>> {
    for a in annotations {
        let resolved = names.resolve(a.string);
        if let Some(message) = match_deprecated(resolved) {
            return Some(message);
        }
    }
    None
}

fn match_deprecated(s: &str) -> Option<Option<String>> {
    let rest = s.strip_prefix("deprecated")?;
    if rest.is_empty() {
        return Some(None);
    }
    let after_colon = rest.strip_prefix(':')?;
    Some(Some(after_colon.trim_start().to_string()))
}
