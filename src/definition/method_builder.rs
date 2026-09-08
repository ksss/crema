//! `MethodBuilder` family — port of `RBS::DefinitionBuilder::MethodBuilder`
//! (`lib/rbs/definition_builder/method_builder.rb`).
//!
//! Build-phase intermediate that aggregates same-name method declarations
//! into per-class buckets before they collapse into `Definition.methods`.
//! See ADR-0019 (sub-operation A) for the role of this layer in the
//! lazy-walk dispatch architecture.
//!
//! 1 type 1 bucket: [`MethodBuilder::build_instance`] /
//! [`MethodBuilder::build_singleton`] walk every reopen decl of a
//! class / module and feed all members — including aliases — into a
//! single [`Methods`] per type kind. An AST-level
//! [`Substitution`](crate::ast::types::substitution::Substitution)
//! pre-rewrites each reopen decl's type-param uses onto the primary
//! decl's names so the bucket holds primary-scope members directly,
//! mirroring rbs's `member.update(overloads: member.overloads.map { |o| o.sub(subst) })`.
//!
//! Interface bucket build mirrors rbs's `build_interface` (single-decl
//! walk, no reopen, no subst); the same Sorter machinery resolves alias
//! buckets at flush time.

use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::ast::MethodKind;
use crate::ast::declarations::{AsMember, Member};
use crate::ast::members::{
    AliasKind, AttrAccessorMember as AstAttrAccessor, AttrReaderMember as AstAttrReader,
    AttrWriterMember as AstAttrWriter, AttributeKind, MethodDefinitionMember as MethodDefinition,
};
use crate::ast::ruby::members::Member as RubyMember;
use crate::ast::types::substitution::Substitution as AstSubst;

/// Instance vs. singleton discriminator for the `build_class_or_module` pass.
///
/// Internal build-phase enum, distinct from the AST-level [`AttributeKind`] /
/// [`AliasKind`] that live on member nodes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BuildSide {
    Instance,
    Singleton,
}
use crate::environment::DeclOrigin;
use crate::environment::Environment;
use crate::environment::frozen::{ClassDeclaration, ClassOrModule, ModuleDeclaration};
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;
use crate::type_param::TypeVarScope;
use crate::types::{Ty, Type, TypeTable, Visibility};

use super::MemberRef;
use crate::definition_builder::{attribute_visibility, special_instance_visibility};

/// Mirrors `RBS::DefinitionBuilder::MethodBuilder`.
///
/// Owns one cache per type kind (instance / singleton / interface). Each
/// cache entry holds a [`Methods`] (one type's bucket collection) and is
/// populated lazily on `build_*` request.
///
/// `errors` is a crema-only collection slot — rbs raises on the first
/// dup via `Methods#validate!`, crema collects so the dup diagnostic
/// surface stays consistent with `MethodDups` (ADR-0010).
///
/// rbs's `update(env:, except:)` is not ported: crema's environment is
/// eagerly populated (ADR-0017) and has no mutation path that would
/// invalidate the cache.
pub struct MethodBuilder<'env> {
    env: &'env Environment,
    types: &'env TypeTable,
    instance_methods: FxHashMap<TypeName, Methods>,
    singleton_methods: FxHashMap<TypeName, Methods>,
    interface_methods: FxHashMap<TypeName, Methods>,
    errors: Vec<DuplicatedMethodDefinitionError>,
}

impl<'env> MethodBuilder<'env> {
    pub fn new(env: &'env Environment, types: &'env TypeTable) -> Self {
        MethodBuilder {
            env,
            types,
            instance_methods: FxHashMap::default(),
            singleton_methods: FxHashMap::default(),
            interface_methods: FxHashMap::default(),
            errors: Vec::new(),
        }
    }

    pub fn env(&self) -> &'env Environment {
        self.env
    }

    pub fn instance_methods(&self) -> &FxHashMap<TypeName, Methods> {
        &self.instance_methods
    }

    pub fn singleton_methods(&self) -> &FxHashMap<TypeName, Methods> {
        &self.singleton_methods
    }

    pub fn interface_methods(&self) -> &FxHashMap<TypeName, Methods> {
        &self.interface_methods
    }

    pub fn errors(&self) -> &[DuplicatedMethodDefinitionError] {
        &self.errors
    }

    /// Mirrors rbs `build_instance(type_name)`. Builds the instance-side
    /// bucket for `type_name` from every reopen decl, using
    /// [`AstSubst`] to rewrite each decl's type-param uses onto the
    /// primary decl's names. Alias members are pushed via
    /// [`MethodBuilder::build_alias`] so the bucket-iter path can
    /// resolve them in a single pass.
    pub fn build_instance(&mut self, type_name: &TypeName) -> &Methods {
        if !self.instance_methods.contains_key(type_name) {
            let methods = self.build_class_or_module(type_name, BuildSide::Instance);
            self.errors.extend(methods.validate());
            self.instance_methods.insert(*type_name, methods);
        }
        self.instance_methods.get(type_name).unwrap()
    }

    pub fn build_singleton(&mut self, type_name: &TypeName) -> &Methods {
        if !self.singleton_methods.contains_key(type_name) {
            let methods = self.build_class_or_module(type_name, BuildSide::Singleton);
            self.errors.extend(methods.validate());
            self.singleton_methods.insert(*type_name, methods);
        }
        self.singleton_methods.get(type_name).unwrap()
    }

    /// Mirrors rbs `build_interface(type_name)`. Interfaces cannot be
    /// reopened, so the bucket holds the single decl's members directly
    /// (no AST-level subst). Method and alias members both flow through
    /// the same `Methods` bucket; alias resolution happens at flush time
    /// via [`Sorter::each_strongly_connected_component`].
    pub fn build_interface(&mut self, type_name: &TypeName) -> &Methods {
        if !self.interface_methods.contains_key(type_name) {
            let methods = self.build_interface_inner(type_name);
            self.errors.extend(methods.validate());
            self.interface_methods.insert(*type_name, methods);
        }
        self.interface_methods.get(type_name).unwrap()
    }

    fn build_interface_inner(&self, type_name: &TypeName) -> Methods {
        let entry = self
            .env
            .interface_decls()
            .get(type_name)
            .expect("MethodBuilder::build_interface called with a type_name not in env");
        let names = self.env.names();
        let decl = entry.decl();
        let bucket_type = self.interface_bucket_type(type_name, &decl.type_params, names);
        let mut methods = Methods::new(bucket_type);
        for member in &decl.members {
            match member {
                Member::MethodDefinition(md) => {
                    // rbs `build_interface` passes `:public` as the
                    // bucket-level default; `define_method` then folds
                    // `special_accessibility` on top of it
                    // (`lib/rbs/definition_builder.rb:746`). Mirror that:
                    // fold special at push time so the bucket's
                    // `accessibilities` already carries Private for the
                    // 5 special names. The overload-only path
                    // (`md.overloading == true` skips this push) is
                    // covered downstream by
                    // `lower_bucket_defn_to_method`'s special fallback.
                    // Any AST-side `visibility` value is ignored to match
                    // `build_interface`'s contract.
                    let name_str = names.resolve(md.name);
                    let visibility =
                        special_instance_visibility(&name_str).unwrap_or(Visibility::Public);
                    let md_arc: Arc<MethodDefinition> = Arc::new(md.clone());
                    MethodBuilder::build_method(
                        &mut methods,
                        md.name,
                        MemberRef::Method(md_arc),
                        visibility,
                        md.overloading,
                        entry.file,
                    );
                }
                Member::Alias(alias) => {
                    MethodBuilder::build_alias(
                        &mut methods,
                        alias.new_name,
                        MemberRef::Alias(Arc::new(alias.clone())),
                        entry.file,
                    );
                }
                // `include` does not contribute an own-method bucket: its
                // reach flows through `interface_ancestors`
                // (`AncestorBuilder::build_interface_ancestors`).
                Member::Include(_) => {}
                _ => {}
            }
        }
        methods
    }

    fn interface_bucket_type(
        &self,
        type_name: &TypeName,
        type_params: &[crate::ast::TypeParam],
        _names: &NameTable,
    ) -> Ty {
        let args: Vec<Ty> = type_params
            .iter()
            .map(|tp| {
                self.types.intern(Type::TypeVariable {
                    raw: tp.name,
                    scope: TypeVarScope::Interface(*type_name),
                })
            })
            .collect();
        self.types.intern(Type::Interface {
            name: *type_name,
            args,
        })
    }

    fn build_class_or_module(&self, type_name: &TypeName, kind: BuildSide) -> Methods {
        let entry =
            self.env.class_decls().get(type_name).expect(
                "MethodBuilder::build_{instance,singleton} called with a type_name not in env",
            );
        let names = self.env.names();
        let primary_params: &[crate::ast::TypeParam] = match entry {
            ClassOrModule::Class(c) => {
                super::ancestor_builder::primary_signature_class_type_params(c)
            }
            ClassOrModule::Module(m) => {
                super::ancestor_builder::primary_signature_module_type_params(m)
            }
        };
        let bucket_type = self.bucket_type(type_name, kind, primary_params, names);
        let mut methods = Methods::new(bucket_type);
        match entry {
            ClassOrModule::Class(c) => {
                for (origin, _, decl) in c.context_decls() {
                    match decl {
                        ClassDeclaration::Signature(sd) => {
                            let subst = AstSubst::build(&sd.type_params, primary_params);
                            push_signature_members(
                                &mut methods,
                                &sd.members,
                                &subst,
                                kind,
                                names,
                                *origin,
                            );
                        }
                        ClassDeclaration::Ruby(rd) => {
                            push_ruby_members(&mut methods, &rd.members, kind, names, *origin);
                        }
                    }
                }
            }
            ClassOrModule::Module(m) => {
                for (origin, _, decl) in m.context_decls() {
                    match decl {
                        ModuleDeclaration::Signature(sd) => {
                            let subst = AstSubst::build(&sd.type_params, primary_params);
                            push_signature_members(
                                &mut methods,
                                &sd.members,
                                &subst,
                                kind,
                                names,
                                *origin,
                            );
                        }
                        ModuleDeclaration::Ruby(rd) => {
                            push_ruby_members(&mut methods, &rd.members, kind, names, *origin);
                        }
                    }
                }
            }
        }
        methods
    }

    fn bucket_type(
        &self,
        type_name: &TypeName,
        kind: BuildSide,
        primary_params: &[crate::ast::TypeParam],
        names: &NameTable,
    ) -> Ty {
        match kind {
            BuildSide::Instance => {
                let _ = names;
                let args: Vec<Ty> = primary_params
                    .iter()
                    .map(|tp| {
                        self.types.intern(Type::TypeVariable {
                            raw: tp.name,
                            scope: TypeVarScope::Class(*type_name),
                        })
                    })
                    .collect();
                self.types.intern(Type::ClassInstance {
                    name: *type_name,
                    args,
                })
            }
            BuildSide::Singleton => self.types.class_singleton(*type_name),
        }
    }

    /// Mirrors rbs `build_method`. Pushes a method member into its
    /// bucket: extras-only (`...` trailer) members are appended to
    /// `overloads`; canonical members are appended to `originals`, with
    /// the effective `accessibility` recorded alongside.
    pub fn build_method(
        methods: &mut Methods,
        name: Symbol,
        member: MemberRef,
        accessibility: Visibility,
        overloading: bool,
        origin: DeclOrigin,
    ) {
        let type_ = methods.type_;
        let defn = methods
            .methods
            .entry(name)
            .or_insert_with(|| methods::Definition::empty(name, type_));
        if overloading {
            defn.overloads.push(member);
        } else {
            defn.accessibilities.push(accessibility);
            defn.originals.push(member);
            defn.origins.push(origin);
        }
    }

    /// Mirrors rbs `build_attribute`. AttrReader/AttrAccessor contribute
    /// the reader-name bucket; AttrWriter/AttrAccessor contribute the
    /// writer-name bucket (`#{name}=`). `writer_name` is pre-interned by
    /// the caller because `Symbol` construction needs a `NameTable`.
    pub fn build_attribute(
        methods: &mut Methods,
        reader_name: Symbol,
        writer_name: Option<Symbol>,
        member: MemberRef,
        accessibility: Visibility,
        origin: DeclOrigin,
    ) {
        Self::push_attribute_buckets(
            methods,
            reader_name,
            writer_name,
            member,
            accessibility,
            origin,
        );
    }

    /// Mirrors rbs `build_ruby_attribute`. Same bucket logic as
    /// `build_attribute`; the rbs split exists because Ruby-side attr
    /// declarations can list multiple names per declaration. crema's
    /// caller iterates the names and calls this once per name, so the
    /// per-name body is identical to `build_attribute`. Both wrappers
    /// stay separate to preserve the rbs-side shared vocabulary.
    pub fn build_ruby_attribute(
        methods: &mut Methods,
        reader_name: Symbol,
        writer_name: Option<Symbol>,
        member: MemberRef,
        accessibility: Visibility,
        origin: DeclOrigin,
    ) {
        Self::push_attribute_buckets(
            methods,
            reader_name,
            writer_name,
            member,
            accessibility,
            origin,
        );
    }

    /// Mirrors rbs `build_alias`. An alias contributes the alias member
    /// itself to `originals` (no visibility — `accessibility()` on an
    /// alias bucket panics, matching rbs's raise).
    pub fn build_alias(methods: &mut Methods, name: Symbol, member: MemberRef, origin: DeclOrigin) {
        let type_ = methods.type_;
        let defn = methods
            .methods
            .entry(name)
            .or_insert_with(|| methods::Definition::empty(name, type_));
        defn.originals.push(member);
        defn.origins.push(origin);
    }

    fn push_attribute_buckets(
        methods: &mut Methods,
        reader_name: Symbol,
        writer_name: Option<Symbol>,
        member: MemberRef,
        accessibility: Visibility,
        origin: DeclOrigin,
    ) {
        let type_ = methods.type_;
        let is_reader = matches!(
            member,
            MemberRef::AttrReader(_)
                | MemberRef::AttrAccessor(_)
                | MemberRef::RubyAttrReader(_)
                | MemberRef::RubyAttrAccessor(_)
        );
        let is_writer = matches!(
            member,
            MemberRef::AttrWriter(_)
                | MemberRef::AttrAccessor(_)
                | MemberRef::RubyAttrWriter(_)
                | MemberRef::RubyAttrAccessor(_)
        );
        if is_reader {
            let defn = methods
                .methods
                .entry(reader_name)
                .or_insert_with(|| methods::Definition::empty(reader_name, type_));
            defn.accessibilities.push(accessibility);
            defn.originals.push(member.clone());
            defn.origins.push(origin);
        }
        if let Some(w) = writer_name
            && is_writer
        {
            let defn = methods
                .methods
                .entry(w)
                .or_insert_with(|| methods::Definition::empty(w, type_));
            defn.accessibilities.push(accessibility);
            defn.originals.push(member);
            defn.origins.push(origin);
        }
    }
}

/// Push a signature decl's members into the bucket, mirroring the
/// per-member dispatch in rbs `build_instance` /
/// `build_singleton`'s `case decl when AST::Declarations::Base` branch.
/// Each method / attribute is type-rewritten through `subst` before
/// being pushed so the bucket holds primary-scope members directly.
fn push_signature_members<M: AsMember>(
    methods: &mut Methods,
    members: &[M],
    subst: &AstSubst,
    kind: BuildSide,
    names: &NameTable,
    origin: DeclOrigin,
) {
    let mut current_visibility = Visibility::Public;
    for wrapper in members {
        let Some(member) = wrapper.as_member() else {
            continue;
        };
        match member {
            Member::Public(_) => {
                current_visibility = Visibility::Public;
            }
            Member::Private(_) => {
                current_visibility = Visibility::Private;
            }
            Member::MethodDefinition(md) => {
                if !method_kind_matches_receiver(md.kind, kind) {
                    continue;
                }
                let raw_visibility = Visibility::from_ast_or_default(md.visibility);
                let visibility = match kind {
                    BuildSide::Instance => {
                        if md.kind == MethodKind::SingletonInstance {
                            Visibility::Private
                        } else {
                            let name_str = names.resolve(md.name);
                            special_instance_visibility(&name_str).unwrap_or(raw_visibility)
                        }
                    }
                    BuildSide::Singleton => raw_visibility,
                };
                let rewritten = MethodDefinition {
                    name: md.name,
                    kind: md.kind,
                    overloads: subst.apply_overloads(&md.overloads, names),
                    annotations: md.annotations.clone(),
                    overloading: md.overloading,
                    visibility: md.visibility,
                    location: md.location,
                    source_file: md.source_file,
                    comment: md.comment.clone(),
                };
                let md_arc: Arc<MethodDefinition> = Arc::new(rewritten);
                MethodBuilder::build_method(
                    methods,
                    md.name,
                    MemberRef::Method(md_arc),
                    visibility,
                    md.overloading,
                    origin,
                );
            }
            Member::AttrReader(r) => {
                if !attribute_kind_matches_receiver(r.kind, kind) {
                    continue;
                }
                let vis = attribute_visibility(r.kind, r.visibility, current_visibility);
                let rewritten = AstAttrReader {
                    name: r.name,
                    ty: subst.apply_type(&r.ty),
                    kind: r.kind,
                    ivar_name: r.ivar_name.clone(),
                    annotations: r.annotations.clone(),
                    location: r.location,
                    source_file: r.source_file,
                    comment: r.comment.clone(),
                    visibility: r.visibility,
                };
                MethodBuilder::build_attribute(
                    methods,
                    r.name,
                    None,
                    MemberRef::AttrReader(Arc::new(rewritten)),
                    vis,
                    origin,
                );
            }
            Member::AttrAccessor(a) => {
                if !attribute_kind_matches_receiver(a.kind, kind) {
                    continue;
                }
                let vis = attribute_visibility(a.kind, a.visibility, current_visibility);
                let writer_str = format!("{}=", names.resolve(a.name));
                let writer_name = names.intern_symbol(&writer_str);
                let rewritten = AstAttrAccessor {
                    name: a.name,
                    ty: subst.apply_type(&a.ty),
                    kind: a.kind,
                    ivar_name: a.ivar_name.clone(),
                    annotations: a.annotations.clone(),
                    location: a.location,
                    source_file: a.source_file,
                    comment: a.comment.clone(),
                    visibility: a.visibility,
                };
                MethodBuilder::build_attribute(
                    methods,
                    a.name,
                    Some(writer_name),
                    MemberRef::AttrAccessor(Arc::new(rewritten)),
                    vis,
                    origin,
                );
            }
            Member::AttrWriter(w) => {
                if !attribute_kind_matches_receiver(w.kind, kind) {
                    continue;
                }
                let vis = attribute_visibility(w.kind, w.visibility, current_visibility);
                let writer_str = format!("{}=", names.resolve(w.name));
                let writer_name = names.intern_symbol(&writer_str);
                let rewritten = AstAttrWriter {
                    name: w.name,
                    ty: subst.apply_type(&w.ty),
                    kind: w.kind,
                    ivar_name: w.ivar_name.clone(),
                    annotations: w.annotations.clone(),
                    location: w.location,
                    source_file: w.source_file,
                    comment: w.comment.clone(),
                    visibility: w.visibility,
                };
                MethodBuilder::build_attribute(
                    methods,
                    w.name,
                    Some(writer_name),
                    MemberRef::AttrWriter(Arc::new(rewritten)),
                    vis,
                    origin,
                );
            }
            Member::Alias(alias) => {
                if !alias_kind_matches_receiver(alias.kind, kind) {
                    continue;
                }
                MethodBuilder::build_alias(
                    methods,
                    alias.new_name,
                    MemberRef::Alias(Arc::new((*alias).clone())),
                    origin,
                );
            }
            _ => {}
        }
    }
}

/// Push a Ruby decl's members into the bucket. Ruby decls carry no
/// class-level type-param subst (the AST has no rbs-style
/// `Substitution.build` driver on the inline path today), so no
/// rewrite is applied — `MemberRef::RubyDef` references the original
/// AST `Arc` directly.
fn push_ruby_members(
    methods: &mut Methods,
    members: &[RubyMember],
    kind: BuildSide,
    names: &NameTable,
    origin: DeclOrigin,
) {
    // Inline attr is instance-only: `class << self` is not supported on
    // the inline side (rbs/docs/inline.md §"Method definitions",
    // "class << self syntax is not supported"). Skip when building the
    // singleton bucket.
    let inline_attr_kind = BuildSide::Instance;
    for member in members {
        match member {
            RubyMember::Def(def) => {
                if !method_kind_matches_receiver(def.kind, kind) {
                    continue;
                }
                let bucket_name = names.intern_symbol(&def.name);
                let bucket_visibility = match kind {
                    BuildSide::Instance => {
                        special_instance_visibility(&def.name).unwrap_or(Visibility::Public)
                    }
                    BuildSide::Singleton => Visibility::Public,
                };
                let is_overloading = def.method_type.overloading() || def.method_type.is_empty();
                MethodBuilder::build_method(
                    methods,
                    bucket_name,
                    MemberRef::RubyDef(Arc::new(def.clone())),
                    bucket_visibility,
                    is_overloading,
                    origin,
                );
            }
            RubyMember::AttrReader(r) if kind == inline_attr_kind => {
                let arc = Arc::new(r.clone());
                for name in r.attribute.names() {
                    let reader_name = names.intern_symbol(name);
                    MethodBuilder::build_ruby_attribute(
                        methods,
                        reader_name,
                        None,
                        MemberRef::RubyAttrReader(Arc::clone(&arc)),
                        Visibility::Public,
                        origin,
                    );
                }
            }
            RubyMember::AttrWriter(w) if kind == inline_attr_kind => {
                let arc = Arc::new(w.clone());
                for name in w.attribute.names() {
                    let writer_str = format!("{name}=");
                    let writer_name = names.intern_symbol(&writer_str);
                    MethodBuilder::build_ruby_attribute(
                        methods,
                        names.intern_symbol(name),
                        Some(writer_name),
                        MemberRef::RubyAttrWriter(Arc::clone(&arc)),
                        Visibility::Public,
                        origin,
                    );
                }
            }
            RubyMember::AttrAccessor(a) if kind == inline_attr_kind => {
                let arc = Arc::new(a.clone());
                for name in a.attribute.names() {
                    let reader_name = names.intern_symbol(name);
                    let writer_str = format!("{name}=");
                    let writer_name = names.intern_symbol(&writer_str);
                    MethodBuilder::build_ruby_attribute(
                        methods,
                        reader_name,
                        Some(writer_name),
                        MemberRef::RubyAttrAccessor(Arc::clone(&arc)),
                        Visibility::Public,
                        origin,
                    );
                }
            }
            _ => {}
        }
    }
}

fn method_kind_matches_receiver(method_kind: MethodKind, receiver: BuildSide) -> bool {
    matches!(
        (method_kind, receiver),
        (MethodKind::Instance, BuildSide::Instance)
            | (MethodKind::Singleton, BuildSide::Singleton)
            | (
                MethodKind::SingletonInstance,
                BuildSide::Instance | BuildSide::Singleton
            )
    )
}

fn attribute_kind_matches_receiver(attr_kind: AttributeKind, receiver: BuildSide) -> bool {
    matches!(
        (attr_kind, receiver),
        (AttributeKind::Instance, BuildSide::Instance)
            | (AttributeKind::Singleton, BuildSide::Singleton)
    )
}

fn alias_kind_matches_receiver(alias_kind: AliasKind, receiver: BuildSide) -> bool {
    matches!(
        (alias_kind, receiver),
        (AliasKind::Instance, BuildSide::Instance) | (AliasKind::Singleton, BuildSide::Singleton)
    )
}

/// Mirrors `RBS::DefinitionBuilder::MethodBuilder::Methods`.
///
/// Per-type bucket collection. The `methods` map carries one bucket
/// (`methods::Definition`) per method name. `validate` rejects buckets
/// with more than one canonical original, mirroring rbs's `validate!`.
#[derive(Debug, Clone)]
pub struct Methods {
    pub type_: Ty,
    pub methods: FxHashMap<Symbol, methods::Definition>,
}

impl Methods {
    pub fn new(type_: Ty) -> Self {
        Methods {
            type_,
            methods: FxHashMap::default(),
        }
    }

    /// Mirrors rbs `validate!`. Returns one error per bucket that holds
    /// more than one canonical `original` (rbs raises on the first; crema
    /// collects so the dup diagnostic surface stays consistent with the
    /// existing `MethodDups` collection model).
    pub fn validate(&self) -> Vec<DuplicatedMethodDefinitionError> {
        let mut errs = Vec::new();
        for defn in self.methods.values() {
            if defn.originals.len() > 1 {
                errs.push(DuplicatedMethodDefinitionError {
                    type_: self.type_,
                    method_name: defn.name,
                    members: defn.originals.clone(),
                    origins: defn.origins.clone(),
                });
            }
        }
        errs
    }
}

/// Inner namespace mirroring rbs's nested `Methods::Definition` class.
/// Kept as a lowercase module so the qualified path
/// `methods::Definition` reads the same as rbs's `Methods::Definition`
/// at every call site — the port = shared-vocabulary principle
/// (MEMORY `feedback_rbs_naming_as_shared_language`).
pub mod methods {
    use super::{MemberRef, Symbol, Ty, Visibility};

    /// Mirrors `RBS::DefinitionBuilder::MethodBuilder::Methods::Definition`.
    ///
    /// Per-method-name bucket built up over the per-class members pass.
    /// `originals` collects canonical defs and alias members;
    /// `overloads` collects extras-only defs (the `...` form);
    /// `accessibilities` collects per-def visibility values. The three
    /// `Vec`s mirror rbs's
    /// `Struct.new(:name, :type, :originals, :overloads, :accessibilities)`
    /// directly so the bucket-build logic maps 1:1 to
    /// `method_builder.rb`.
    #[derive(Debug, Clone)]
    pub struct Definition {
        pub name: Symbol,
        pub type_: Ty,
        pub originals: Vec<MemberRef>,
        /// Parallel to `originals` (same index, same push site — see
        /// `MethodBuilder::build_method` / `build_alias` /
        /// `push_attribute_buckets`): the `DeclOrigin` of the reopen decl
        /// each original member came from. Threaded through so a
        /// `DuplicatedMethodDefinitionError`'s members can be told apart
        /// as "real file" vs. "infusion synthesis" without guessing from
        /// a `None` location (`mid_dup_method_infusion_provenance`).
        pub origins: Vec<crate::environment::DeclOrigin>,
        pub overloads: Vec<MemberRef>,
        pub accessibilities: Vec<Visibility>,
    }

    impl Definition {
        pub fn empty(name: Symbol, type_: Ty) -> Self {
            Definition {
                name,
                type_,
                originals: Vec::new(),
                origins: Vec::new(),
                overloads: Vec::new(),
                accessibilities: Vec::new(),
            }
        }

        /// Mirrors rbs `original` (`originals.first`).
        pub fn original(&self) -> Option<&MemberRef> {
            self.originals.first()
        }

        /// Mirrors rbs `accessibility` (`accessibilities[0]`). rbs raises
        /// when called on an alias member; the crema bucket-iter path
        /// must not call this on alias-only buckets.
        pub fn accessibility(&self) -> Visibility {
            *self.accessibilities.first().expect(
                "Methods::Definition::accessibility called on a bucket without accessibilities",
            )
        }

        /// True when the bucket's canonical original is an alias
        /// member. The bucket-iter path uses this to dispatch alias
        /// resolution (clone the target Method) rather than the
        /// per-overload `lower_member_to_method` path.
        pub fn is_alias(&self) -> bool {
            matches!(self.originals.first(), Some(MemberRef::Alias(_)))
        }
    }
}

/// Mirrors `RBS::DefinitionBuilder::MethodBuilder::Sorter`.
///
/// Walks the alias DAG in strongly-connected-component order so the
/// alias rewrite in `Definition.methods` runs after each target is
/// resolved. SCCs of size > 1 are cyclic aliases and reported through
/// the `f` callback so the caller can decide whether to skip them
/// silently (current crema behaviour, see ADR-0010) or emit a
/// diagnostic (`low_alias_cycle_diagnostic`).
pub struct Sorter<'methods> {
    methods: &'methods FxHashMap<Symbol, methods::Definition>,
}

impl<'methods> Sorter<'methods> {
    pub fn new(methods: &'methods FxHashMap<Symbol, methods::Definition>) -> Self {
        Sorter { methods }
    }

    pub fn methods(&self) -> &FxHashMap<Symbol, methods::Definition> {
        self.methods
    }

    /// Mirrors `RBS::DefinitionBuilder::MethodBuilder::Methods::Sorter#each_strongly_connected_component`.
    /// Visits every SCC of the alias DAG, where the DAG's nodes are
    /// the bucket entries and edges go from an alias bucket to the
    /// bucket of its `old_name` (when that bucket exists in this
    /// `Methods`). SCCs of size 1 are normal methods or terminal
    /// aliases; SCCs of size > 1 are cyclic alias chains.
    ///
    /// Implementation uses a simplified single-edge traversal: each
    /// bucket has at most one outgoing edge (its alias's old_name),
    /// so the alias DAG is a functional graph. Walking forward from
    /// each unvisited bucket builds a path; the path terminates when
    /// (a) the current bucket is non-alias, (b) the alias points
    /// outside the bucket, (c) we hit an already-finished bucket,
    /// or (d) we revisit a bucket on the current path (cycle).
    pub fn each_strongly_connected_component<F>(&self, mut f: F)
    where
        F: FnMut(&[&methods::Definition]),
    {
        #[derive(Clone, Copy)]
        enum NodeState {
            Unvisited,
            OnPath(u32),
            Done,
        }

        // FxHashMap iteration order is non-deterministic for the root
        // pick, but path-internal order is deterministic (chain a→b→c
        // always emits c, b, a from the path-pop in reverse).
        let names: Vec<Symbol> = self.methods.keys().copied().collect();

        let mut state: FxHashMap<Symbol, NodeState> =
            names.iter().map(|n| (*n, NodeState::Unvisited)).collect();

        let emit_singletons_in_reverse =
            |path: &mut Vec<Symbol>, state: &mut FxHashMap<Symbol, NodeState>, f: &mut F| {
                while let Some(n) = path.pop() {
                    state.insert(n, NodeState::Done);
                    let defn = self.methods.get(&n).expect("path node in methods");
                    f(&[defn]);
                }
            };
        let emit = emit_singletons_in_reverse;

        for &start in &names {
            if !matches!(state.get(&start), Some(NodeState::Unvisited)) {
                continue;
            }
            let mut path: Vec<Symbol> = Vec::new();
            let mut cur = start;
            loop {
                match state.get(&cur).copied().unwrap_or(NodeState::Unvisited) {
                    NodeState::Unvisited => {
                        state.insert(cur, NodeState::OnPath(path.len() as u32));
                        path.push(cur);
                        let defn = self.methods.get(&cur).expect("cur in methods");
                        if let Some(MemberRef::Alias(alias)) = defn.originals.first()
                            && self.methods.contains_key(&alias.old_name)
                        {
                            cur = alias.old_name;
                            continue;
                        }
                        // Non-alias or alias pointing outside the
                        // bucket — emit the path as size-1 SCCs in
                        // reverse (target first, alias chain after).
                        emit(&mut path, &mut state, &mut f);
                        break;
                    }
                    NodeState::OnPath(idx) => {
                        // Cycle: path[idx..] forms the SCC.
                        let idx = idx as usize;
                        let cycle_names: Vec<Symbol> = path[idx..].to_vec();
                        let cycle_defs: Vec<&methods::Definition> = cycle_names
                            .iter()
                            .map(|n| self.methods.get(n).expect("cycle node in methods"))
                            .collect();
                        for n in &cycle_names {
                            state.insert(*n, NodeState::Done);
                        }
                        f(&cycle_defs);
                        path.truncate(idx);
                        emit(&mut path, &mut state, &mut f);
                        break;
                    }
                    NodeState::Done => {
                        // Edge into already-finished node — emit
                        // remaining path as size-1 SCCs.
                        emit(&mut path, &mut state, &mut f);
                        break;
                    }
                }
            }
        }
    }
}

/// Mirrors rbs `RBS::DuplicatedMethodDefinitionError`. Produced by
/// `Methods::validate` when a bucket holds more than one canonical
/// `original`. Consumers project this into the `MethodDups`
/// diagnostic stream.
#[derive(Debug, Clone)]
pub struct DuplicatedMethodDefinitionError {
    pub type_: Ty,
    pub method_name: Symbol,
    pub members: Vec<MemberRef>,
    /// Parallel to `members` — see [`methods::Definition::origins`].
    pub origins: Vec<crate::environment::DeclOrigin>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::members::AliasMember;
    use crate::name::NameTable;
    use rustc_hash::FxHashSet;

    #[test]
    fn methods_new_is_empty() {
        let m = Methods::new(Ty::UNTYPED);
        assert_eq!(m.type_, Ty::UNTYPED);
        assert!(m.methods.is_empty());
    }

    #[test]
    fn methods_definition_empty_starts_empty() {
        let names = NameTable::new();
        let sym = names.intern_symbol("foo");
        let defn = methods::Definition::empty(sym, Ty::UNTYPED);
        assert_eq!(defn.name, sym);
        assert_eq!(defn.type_, Ty::UNTYPED);
        assert!(defn.originals.is_empty());
        assert!(defn.overloads.is_empty());
        assert!(defn.accessibilities.is_empty());
    }

    #[test]
    fn methods_definition_original_is_none_when_empty() {
        let names = NameTable::new();
        let defn = methods::Definition::empty(names.intern_symbol("bar"), Ty::UNTYPED);
        assert!(defn.original().is_none());
    }

    #[test]
    fn build_method_routes_canonical_to_originals_and_extras_to_overloads() {
        let names = NameTable::new();
        let name = names.intern_symbol("foo");
        let mut methods = Methods::new(Ty::UNTYPED);

        MethodBuilder::build_method(
            &mut methods,
            name,
            MemberRef::Synthesized,
            Visibility::Public,
            /* overloading */ false,
            DeclOrigin::Unspecified,
        );
        MethodBuilder::build_method(
            &mut methods,
            name,
            MemberRef::Synthesized,
            Visibility::Private,
            /* overloading */ true,
            DeclOrigin::Unspecified,
        );

        let defn = methods.methods.get(&name).expect("bucket created");
        assert_eq!(defn.originals.len(), 1, "canonical → originals");
        assert_eq!(defn.overloads.len(), 1, "`...` form → overloads");
        assert_eq!(
            defn.accessibilities,
            vec![Visibility::Public],
            "only canonicals contribute accessibility, mirroring rbs",
        );
    }

    fn make_alias(new_name: Symbol, old_name: Symbol) -> AliasMember {
        AliasMember {
            new_name,
            old_name,
            kind: AliasKind::Instance,
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        }
    }

    #[test]
    fn build_alias_appends_alias_member_to_originals() {
        let names = NameTable::new();
        let mut methods = Methods::new(Ty::UNTYPED);
        let new_name = names.intern_symbol("baz");
        let old_name = names.intern_symbol("bar");

        MethodBuilder::build_alias(
            &mut methods,
            new_name,
            MemberRef::Alias(Arc::new(make_alias(new_name, old_name))),
            DeclOrigin::Unspecified,
        );
        let defn = methods.methods.get(&new_name).expect("alias bucket");
        assert_eq!(defn.originals.len(), 1);
        assert!(defn.is_alias());
        assert!(defn.accessibilities.is_empty());
    }

    #[test]
    fn member_ref_alias_location_passes_through_alias_location() {
        use crate::location::{AliasMemberLocation, LocationRange};
        let names = NameTable::new();
        let new_name = names.intern_symbol("baz");
        let old_name = names.intern_symbol("bar");
        let mut alias = make_alias(new_name, old_name);
        let range = LocationRange::new(0, 0, 0, 10);
        let loc = AliasMemberLocation {
            range,
            keyword_range: range,
            new_name_range: range,
            old_name_range: range,
            new_kind_range: None,
            old_kind_range: None,
        };
        alias.location = Some(loc);

        let member = MemberRef::Alias(Arc::new(alias));
        assert_eq!(member.location(), Some(range));
    }

    /// Build a bucket from a list of (name, original) pairs. Aliases
    /// are wired by passing `Some(old_name)` to point at the target
    /// bucket; methods pass `None`.
    fn make_bucket(
        names: &NameTable,
        entries: &[(Symbol, Option<Symbol>)],
    ) -> FxHashMap<Symbol, methods::Definition> {
        let mut map = FxHashMap::default();
        for (name, old) in entries {
            let mut defn = methods::Definition::empty(*name, Ty::UNTYPED);
            match old {
                Some(o) => {
                    defn.originals
                        .push(MemberRef::Alias(Arc::new(make_alias(*name, *o))));
                }
                None => {
                    defn.originals.push(MemberRef::Synthesized);
                    defn.accessibilities.push(Visibility::Public);
                }
            }
            defn.origins.push(DeclOrigin::Unspecified);
            map.insert(*name, defn);
            let _ = names;
        }
        map
    }

    fn collect_sccs<'a>(sorter: &Sorter<'a>) -> Vec<Vec<Symbol>> {
        let mut out: Vec<Vec<Symbol>> = Vec::new();
        sorter.each_strongly_connected_component(|scc| {
            out.push(scc.iter().map(|d| d.name).collect());
        });
        out
    }

    #[test]
    fn sorter_orders_alias_target_before_alias() {
        // alias a -> b -> c (c is a real method)
        let names = NameTable::new();
        let a = names.intern_symbol("a");
        let b = names.intern_symbol("b");
        let c = names.intern_symbol("c");
        let map = make_bucket(&names, &[(a, Some(b)), (b, Some(c)), (c, None)]);
        let sorter = Sorter::new(&map);
        let sccs = collect_sccs(&sorter);

        let order: Vec<Symbol> = sccs.iter().flatten().copied().collect();
        let pos_a = order.iter().position(|&s| s == a).unwrap();
        let pos_b = order.iter().position(|&s| s == b).unwrap();
        let pos_c = order.iter().position(|&s| s == c).unwrap();
        assert!(
            pos_c < pos_b && pos_b < pos_a,
            "target should be visited before alias; got {:?}",
            order
        );
        assert!(sccs.iter().all(|s| s.len() == 1), "no cycles");
    }

    #[test]
    fn sorter_detects_self_loop() {
        let names = NameTable::new();
        let a = names.intern_symbol("a");
        let map = make_bucket(&names, &[(a, Some(a))]);
        let sorter = Sorter::new(&map);
        let sccs = collect_sccs(&sorter);

        // Functional-graph SCC: a self-loop emits as a singleton SCC
        // (size 1), not as size 2 the way Tarjan would in general
        // graphs. The `scc.len() > 1` filter the caller uses to detect
        // cyclic chains therefore CANNOT see a self-loop — the caller
        // (flush_bucket_via_sorter) instead silently skips by failing
        // the `dest.get(&alias.old_name)` lookup because the target
        // bucket is the alias itself and was never inserted into
        // `dest`. Future `low_alias_cycle_diagnostic` work needs an
        // additional `new_name == old_name` check on the singleton
        // branch to recover the self-loop signal.
        let cycles: Vec<&Vec<Symbol>> = sccs.iter().filter(|s| s.len() > 1).collect();
        assert!(
            cycles.is_empty(),
            "self-loop must NOT appear as a size>1 SCC; got {:?}",
            cycles
        );
        assert_eq!(sccs.iter().flatten().count(), 1);
        assert_eq!(sccs[0], vec![a]);
    }

    #[test]
    fn sorter_detects_two_node_cycle() {
        // alias a -> b ; alias b -> a (mutual cycle, SCC of size 2)
        let names = NameTable::new();
        let a = names.intern_symbol("a");
        let b = names.intern_symbol("b");
        let map = make_bucket(&names, &[(a, Some(b)), (b, Some(a))]);
        let sorter = Sorter::new(&map);
        let sccs = collect_sccs(&sorter);

        let cycles: Vec<&Vec<Symbol>> = sccs.iter().filter(|s| s.len() > 1).collect();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
        let cycle_set: FxHashSet<Symbol> = cycles[0].iter().copied().collect();
        assert_eq!(cycle_set, [a, b].into_iter().collect());
    }
}
