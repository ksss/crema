use std::path::Path;
use std::sync::Arc;


use crate::ast::Source;
use crate::ast::declarations::{
    ClassAliasDeclaration as ClassAlias, ClassDeclaration as Class, ClassMember,
    ClassSuper as Super, ConstantDeclaration as Constant, Declaration, GlobalDeclaration as Global,
    InterfaceDeclaration as Interface, Member, ModuleAliasDeclaration as ModuleAlias,
    ModuleDeclaration as Module, ModuleMember, ModuleSelf as SelfType,
    TypeAliasDeclaration as TypeAlias,
};
use crate::ast::directives::{
    Directive, ResolveTypeNamesDirective, UseClause, UseDirective, UseSingleClause,
    UseWildcardClause,
};
use crate::ast::members::{
    AliasKind, AliasMember, AttrAccessorMember, AttrReaderMember, AttrWriterMember,
    AttributeKind as AstAttributeKind, ClassInstanceVariableMember, ClassVariableMember,
    ExtendMember, IncludeMember, InstanceVariableMember, IvarName, MethodDefinitionMember,
    PrependMember, PrivateMember, PublicMember,
};
use crate::ast::types::{BaseType, BaseTypeKind, Type as AstType};
use crate::ast::{MethodKind, TypeParam, Variance, Visibility as AstVisibility};
use crate::ast_builder;
use crate::environment::draft::{Context as DraftContext, EnvironmentDraft};
use crate::name::Name;
use crate::name_resolver::qualified_name;
use crate::rbs_raw::{
    AliasKind as RawAliasKind, AttributeKind as RawAttributeKind, AttributeVisibility, DeclKind,
    MemberKind, MethodDefinitionKind, MethodDefinitionVisibility, Parser, RawNodeList,
    StructuralKind, classify_decl, classify_member, classify_structural,
};

use crate::location::{
    AliasDeclarationLocation, AliasMemberLocation, AttributeMemberLocation,
    ClassDeclarationLocation, ClassSuperLocation, ConstantDeclarationLocation,
    GlobalDeclarationLocation, InterfaceDeclarationLocation, MethodDefinitionLocation,
    MixinMemberLocation, ModuleDeclarationLocation, ModuleSelfLocation,
    TypeAliasDeclarationLocation, VariableMemberLocation,
};
use crate::rbs_raw::{
    ffi_range_to_location_range, ffi_range_to_optional_location_range, node_location_range,
};
use crate::type_name::TypeName;
use crate::types::Visibility;
use ruby_rbs_sys::bindings::{
    rbs_ast_declarations_class_alias_t, rbs_ast_declarations_class_super_t,
    rbs_ast_declarations_class_t, rbs_ast_declarations_constant_t, rbs_ast_declarations_global_t,
    rbs_ast_declarations_interface_t, rbs_ast_declarations_module_alias_t,
    rbs_ast_declarations_module_self_t, rbs_ast_declarations_module_t,
    rbs_ast_declarations_type_alias_t, rbs_ast_members_alias_t, rbs_ast_members_attr_accessor_t,
    rbs_ast_members_attr_reader_t, rbs_ast_members_attr_writer_t,
    rbs_ast_members_class_instance_variable_t, rbs_ast_members_class_variable_t,
    rbs_ast_members_extend_t, rbs_ast_members_include_t, rbs_ast_members_instance_variable_t,
    rbs_ast_members_method_definition_t, rbs_ast_members_prepend_t, rbs_attr_ivar_name,
    rbs_attr_ivar_name_tag,
};

/// File identity passed through the RBS load pipeline.
type LoadFile = Option<Name>;

/// rbs_raw skips every `rbs_ast_comment` field today, for both member and
/// declaration nodes. Keep the call sites explicit until the binding
/// exposes it.
fn rbs_raw_comment_unavailable() -> Option<crate::ast::comment::Comment> {
    None
}

/// Resolve the effective visibility for an **instance** method or attribute
/// declaration. Member-level modifier wins; `Unspecified` falls back to the
/// enclosing scope's current default (which is flipped by bare
/// `private` / `public` visibility-members).
fn resolve_method_instance_visibility(
    modifier: MethodDefinitionVisibility,
    current_default: Visibility,
) -> Visibility {
    match modifier {
        MethodDefinitionVisibility::Public => Visibility::Public,
        MethodDefinitionVisibility::Private => Visibility::Private,
        MethodDefinitionVisibility::Unspecified => current_default,
    }
}

/// Resolve the effective visibility for a **singleton** method declaration.
/// Bare `private` / `public` visibility-members do **not** affect singleton
/// methods (matches RBS gem's `build_singleton`, which has no accessibility
/// tracking, and Ruby's own semantics where `private` alone covers instance
/// methods only — `private_class_method` is needed for singletons).
fn resolve_method_singleton_visibility(modifier: MethodDefinitionVisibility) -> Visibility {
    match modifier {
        MethodDefinitionVisibility::Public => Visibility::Public,
        MethodDefinitionVisibility::Private => Visibility::Private,
        MethodDefinitionVisibility::Unspecified => Visibility::Public,
    }
}

/// Map a raw `AttributeVisibility` to `Option<AstVisibility>`, preserving
/// `Unspecified` as `None` so the definition layer can fold it against
/// the surrounding marker context. Mirrors RBS gem's approach of keeping
/// raw `nil` in AST nodes until `DefinitionBuilder` resolves them.
fn to_attr_ast_visibility(modifier: AttributeVisibility) -> Option<AstVisibility> {
    match modifier {
        AttributeVisibility::Public => Some(AstVisibility::Public),
        AttributeVisibility::Private => Some(AstVisibility::Private),
        AttributeVisibility::Unspecified => None,
    }
}

fn to_decl_visibility(v: Visibility) -> AstVisibility {
    match v {
        Visibility::Public => AstVisibility::Public,
        Visibility::Private => AstVisibility::Private,
    }
}

fn to_decl_variance(v: crate::rbs_raw::TypeParamVariance) -> Variance {
    match v {
        crate::rbs_raw::TypeParamVariance::Invariant => Variance::Invariant,
        crate::rbs_raw::TypeParamVariance::Covariant => Variance::Covariant,
        crate::rbs_raw::TypeParamVariance::Contravariant => Variance::Contravariant,
    }
}

fn to_decl_method_kind(k: MethodDefinitionKind) -> MethodKind {
    match k {
        MethodDefinitionKind::Instance => MethodKind::Instance,
        MethodDefinitionKind::Singleton => MethodKind::Singleton,
        MethodDefinitionKind::SingletonInstance => MethodKind::SingletonInstance,
    }
}

/// Detect the `# resolve-type-names: true|false` magic comment on the
/// very first line of an RBS source.
///
/// Mirrors `RBS::Parser.magic_comment` in `rbs/lib/rbs/parser_aux.rb:46-68`:
///
/// ```ruby
/// /\A#\s*(?<keyword>resolve-type-names)\s*(?<colon>:)\s+(?<value>true|false)$/
/// ```
///
/// The librbs C parser does not surface this comment, so crema reads
/// the raw source bytes itself before handing them to
/// `Parser::parse_signature`.
fn parse_magic_comment(source: &[u8]) -> Option<ResolveTypeNamesDirective> {
    let first_line_end = source
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(source.len());
    let line = std::str::from_utf8(&source[..first_line_end]).ok()?;
    // Normalize CRLF.
    let line = line.trim_end_matches('\r');

    let after_hash = line.strip_prefix('#')?;
    let after_ws1 = after_hash.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let after_keyword = after_ws1.strip_prefix("resolve-type-names")?;
    let after_ws2 = after_keyword.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let after_colon = after_ws2.strip_prefix(':')?;
    // rbs regex requires `\s+` after the colon (at least one whitespace).
    let after_ws3 = after_colon.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if after_ws3.len() == after_colon.len() {
        return None;
    }
    match after_ws3 {
        "true" => Some(ResolveTypeNamesDirective {
            value: true,
            location: None,
        }),
        "false" => Some(ResolveTypeNamesDirective {
            value: false,
            location: None,
        }),
        _ => None,
    }
}

impl EnvironmentDraft {
    /// Load all .rbs files from a directory recursively into this draft,
    /// walking with [`crate::file_finder::each_file`] (rbs `FileFinder`):
    /// `skip_hidden` prunes `_` dirs and is `true` for core / library /
    /// collection sources, `false` for user sig dirs (rbs `-I`).
    /// ADR-0017 Phase 5b: legacy buffer is still
    /// updated in lockstep so existing internal helpers
    /// (`process_class_decl_ast` etc.) keep compiling; the buffer is no
    /// longer consumed downstream and Phase E will remove the dead push
    /// path.
    pub fn load_dir(&mut self, dir: &Path, skip_hidden: bool) -> Result<(), String> {
        for path in crate::file_finder::each_file(dir, skip_hidden)? {
            self.load_file(&path)?;
        }
        Ok(())
    }

    /// Load and parse a single .rbs file into this draft.
    pub fn load_file(&mut self, path: &Path) -> Result<(), String> {
        let source =
            std::fs::read(path).map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;

        let file = self.names.intern(&path.to_string_lossy());
        self.load_rbs_source_with_file(&source, Some(file))
    }

    /// Parse RBS source bytes without a file identity. Used by tests
    /// that feed RBS strings inline; resulting method definitions have
    /// `source: None` and the per-decl origin slot is
    /// [`crate::environment::DeclOrigin::Unspecified`].
    pub fn load_rbs_source(&mut self, source: &[u8]) -> Result<(), String> {
        self.load_rbs_source_with_file(source, None)
    }

    /// Parse RBS source bytes with an optional file identity for location
    /// tracking, mirroring decls and directives into the draft.
    fn load_rbs_source_with_file(&mut self, source: &[u8], file: LoadFile) -> Result<(), String> {
        let source_str =
            std::str::from_utf8(source).map_err(|e| format!("Invalid UTF-8: {}", e))?;
        let magic = parse_magic_comment(source);
        let (parser, signature) = Parser::parse_signature(source_str.as_bytes())?;
        let mut source_ast = self.build_source_ast(&parser, signature, file);
        // Mirror rbs `parse_signature`: prepend the magic-comment
        // directive to the directives list (`dirs.unshift(resolved)`).
        if let Some(rtn) = magic {
            source_ast
                .directives
                .insert(0, Directive::ResolveTypeNames(rtn));
        }
        self.add_directives(file, &source_ast.directives);
        self.process_source(&source_ast, file)
    }

    /// Lower the parsed RBS signature into an [`ast::Source`] (directives
    /// + declarations) without touching the draft yet.
    fn build_source_ast(
        &mut self,
        parser: &Parser,
        signature: *mut ruby_rbs_sys::bindings::rbs_signature_t,
        file: LoadFile,
    ) -> Source {
        let directives: Vec<Directive> = parser
            .signature_directives(signature)
            .iter()
            .filter_map(|dir_ptr| self.lower_directive(parser, dir_ptr))
            .collect();

        let declarations: Vec<Declaration> = parser
            .signature_declarations(signature)
            .iter()
            .filter_map(|decl_ptr| self.build_declaration_ast(parser, decl_ptr, "", file))
            .collect();

        Source {
            directives,
            declarations,
        }
    }

    /// Lower one top-level directive raw node emitted by the librbs
    /// signature parser. Today this only sees `use` clauses; the
    /// `# resolve-type-names` magic comment lives outside the parser's
    /// output and is injected by [`parse_magic_comment`] in
    /// [`Self::load_rbs_source_with_file`] instead.
    fn lower_directive(
        &self,
        parser: &Parser,
        raw: *const ruby_rbs_sys::bindings::rbs_node,
    ) -> Option<Directive> {
        match classify_structural(parser, raw) {
            StructuralKind::Use { clauses } => {
                let lowered: Vec<UseClause> = clauses
                    .iter()
                    .filter_map(|clause_ptr| self.lower_use_clause(parser, clause_ptr))
                    .collect();
                Some(Directive::Use(UseDirective {
                    clauses: lowered,
                    location: None,
                }))
            }
            _ => None,
        }
    }

    /// Lower a single clause inside a `use` directive.
    fn lower_use_clause(
        &self,
        parser: &Parser,
        raw: *const ruby_rbs_sys::bindings::rbs_node,
    ) -> Option<UseClause> {
        match classify_structural(parser, raw) {
            StructuralKind::UseSingleClause {
                type_name,
                new_name,
            } => {
                let qualified = parser.type_name_to_string(type_name);

                let tn = self.names.parse_type_name(&qualified);
                let new_name_sym = new_name.map(|sym_ptr| {
                    let s = parser.resolve_constant(sym_ptr).to_string();
                    self.names.intern_symbol(&s)
                });
                Some(UseClause::Single(UseSingleClause {
                    type_name: tn,
                    new_name: new_name_sym,
                    location: None,
                }))
            }
            StructuralKind::UseWildcardClause { rbs_namespace } => {
                let qualified = parser.namespace_to_string(rbs_namespace);
                let ns = self.names.parse_type_name(&qualified);
                Some(UseClause::Wildcard(UseWildcardClause {
                    namespace: ns,
                    location: None,
                }))
            }
            _ => None,
        }
    }

    /// Mirror a parsed file's declarations into the draft.
    /// Directives carried alongside in [`Source`] are intentionally
    /// not consumed here yet — they flow through to the resolver in
    /// a later phase. Today the draft only needs the declarations.
    fn process_source(&mut self, source: &Source, file: LoadFile) -> Result<(), String> {
        let top_decls = &source.declarations;

        // `process_source` is only called at top level so the
        // namespace stack is just `[root]`; nested decls are surfaced
        // through `Class.members` and the draft `build()` walk
        // recovers their nesting itself.
        let draft_ctx: DraftContext = std::sync::Arc::from([self.names.absolute_root()]);
        // `Path` for real files, `Unspecified` for inline test sources
        // (`load_rbs_source`) — the `From` impl encodes exactly that split.
        let file: crate::environment::DeclOrigin = file.into();

        for decl in top_decls {
            // Each arm runs the insert and, on error, resolves the
            // interned `Symbol`s embedded in `BuildError` against
            // `self.names` so the user sees `::Foo::Bar` rather than
            // `Symbol(2685)`. Collecting `(label, result)` first releases
            // the `&mut self` borrow before the error formatter takes an
            // immutable borrow of `self.names`.
            let (label, res) = match decl {
                Declaration::TypeAlias(ta) => (
                    "insert_type_alias",
                    self.insert_type_alias(file, draft_ctx.clone(), ta.clone()),
                ),
                Declaration::ClassAlias(ca) => (
                    "insert_class_alias",
                    self.insert_class_alias(file, draft_ctx.clone(), ca.clone()),
                ),
                Declaration::ModuleAlias(ma) => (
                    "insert_module_alias",
                    self.insert_module_alias(file, draft_ctx.clone(), ma.clone()),
                ),
                Declaration::Class(d) => (
                    "insert_class_decl",
                    self.insert_class_decl(file, draft_ctx.clone(), d.clone()),
                ),
                Declaration::Module(d) => (
                    "insert_module_decl",
                    self.insert_module_decl(file, draft_ctx.clone(), d.clone()),
                ),
                Declaration::Interface(d) => (
                    "insert_interface_decl",
                    self.insert_interface_decl(file, draft_ctx.clone(), d.clone()),
                ),
                Declaration::Constant(d) => (
                    "insert_constant",
                    self.insert_constant(file, draft_ctx.clone(), d.clone()),
                ),
                Declaration::Global(d) => (
                    "insert_global",
                    self.insert_global(file, draft_ctx.clone(), d.clone()),
                ),
            };
            res.map_err(|e| format!("draft {}: {}", label, e.format_with(&self.names)))?;
        }

        Ok(())
    }

    fn decl_name(
        &self,
        parser: &Parser,
        name: *const ruby_rbs_sys::bindings::rbs_type_name,
        ns_prefix: &str,
    ) -> TypeName {
        let raw = qualified_name(ns_prefix, &parser.type_name_to_string(name));
        self.names.parse_type_name(&raw)
    }

    fn ref_name(
        &self,
        parser: &Parser,
        name: *const ruby_rbs_sys::bindings::rbs_type_name,
    ) -> TypeName {
        let raw = parser.type_name_to_string(name);
        self.names.parse_type_name(&raw)
    }

    fn convert_decl_type_params(
        &self,
        parser: &Parser,
        type_params: RawNodeList,
    ) -> Vec<TypeParam> {
        type_params
            .iter()
            .filter_map(|tp| {
                let StructuralKind::TypeParam {
                    name,
                    variance,
                    upper_bound,
                    lower_bound,
                    default_type,
                    unchecked,
                } = classify_structural(parser, tp)
                else {
                    return None;
                };
                Some(TypeParam {
                    name: self.names.intern_symbol(parser.resolve_constant(name)),
                    variance: to_decl_variance(variance),
                    upper_bound: upper_bound.map(|n| {
                        ast_builder::build_type(parser, n, &self.names).unwrap_or(AstType::Base(
                            BaseType {
                                kind: BaseTypeKind::Any { todo: false },
                                location: None,
                            },
                        ))
                    }),
                    lower_bound: lower_bound.map(|n| {
                        ast_builder::build_type(parser, n, &self.names).unwrap_or(AstType::Base(
                            BaseType {
                                kind: BaseTypeKind::Any { todo: false },
                                location: None,
                            },
                        ))
                    }),
                    default_type: default_type.map(|n| {
                        ast_builder::build_type(parser, n, &self.names).unwrap_or(AstType::Base(
                            BaseType {
                                kind: BaseTypeKind::Any { todo: false },
                                location: None,
                            },
                        ))
                    }),
                    unchecked,
                    location: None,
                })
            })
            .collect()
    }

    fn attr_ivar_from_raw(&self, parser: &Parser, ivar_name: rbs_attr_ivar_name) -> IvarName {
        match ivar_name.tag {
            rbs_attr_ivar_name_tag::RBS_ATTR_IVAR_NAME_TAG_UNSPECIFIED => IvarName::Unspecified,
            rbs_attr_ivar_name_tag::RBS_ATTR_IVAR_NAME_TAG_EMPTY => IvarName::Empty,
            rbs_attr_ivar_name_tag::RBS_ATTR_IVAR_NAME_TAG_NAME => IvarName::Name(
                self.names
                    .intern_symbol(parser.resolve_constant_id(ivar_name.name)),
            ),
            _ => IvarName::Unspecified,
        }
    }

    /// Build a `MethodDefinition` AST node from a `classify_member`
    /// payload. Shared between `build_class_or_module_member_ast`
    /// (class / module decls) and the interface arm of
    /// `build_declaration_ast` so the two call sites stay in lock-step
    /// when `MethodDefinition` gains new fields. Interface callers pass
    /// `current_default = Visibility::Public`; class / module callers
    /// thread `*current_visibility` so the surrounding `public` /
    /// `private` marker is honoured.
    #[expect(clippy::too_many_arguments)]
    fn build_method_definition_from_kind(
        &self,
        parser: &Parser,
        name: *const ruby_rbs_sys::bindings::rbs_ast_symbol,
        raw_kind: MethodDefinitionKind,
        overloads: RawNodeList<'_>,
        annotations: RawNodeList<'_>,
        overloading: bool,
        visibility_marker: MethodDefinitionVisibility,
        current_default: Visibility,
        location: Option<MethodDefinitionLocation>,
        file: LoadFile,
    ) -> MethodDefinitionMember {
        let method_name_str = parser.resolve_constant(name).to_string();
        let kind = to_decl_method_kind(raw_kind);
        let visibility = match kind {
            MethodKind::Instance => {
                resolve_method_instance_visibility(visibility_marker, current_default)
            }
            MethodKind::Singleton => resolve_method_singleton_visibility(visibility_marker),
            MethodKind::SingletonInstance => Visibility::Public,
        };
        MethodDefinitionMember {
            name: self.names.intern_symbol(&method_name_str),
            kind,
            overloads: ast_builder::build_overloads(parser, overloads, file, &self.names),
            annotations: ast_builder::build_annotations(parser, annotations, file, &self.names),
            overloading,
            visibility: Some(to_decl_visibility(visibility)),
            location,
            source_file: file,
            comment: rbs_raw_comment_unavailable(),
        }
    }

    /// Build an `Alias` AST node from a `classify_member` payload.
    /// Shared between the class / module and interface arms; `Alias`
    /// has no visibility / accessibility distinction so the helper
    /// takes no `current_default` argument.
    #[expect(clippy::too_many_arguments)]
    fn build_alias_from_kind(
        &self,
        parser: &Parser,
        raw_kind: RawAliasKind,
        old_name: *const ruby_rbs_sys::bindings::rbs_ast_symbol,
        new_name: *const ruby_rbs_sys::bindings::rbs_ast_symbol,
        annotations: RawNodeList<'_>,
        location: Option<AliasMemberLocation>,
        file: LoadFile,
    ) -> AliasMember {
        AliasMember {
            new_name: self.names.intern_symbol(parser.resolve_constant(new_name)),
            old_name: self.names.intern_symbol(parser.resolve_constant(old_name)),
            kind: match raw_kind {
                RawAliasKind::Instance => AliasKind::Instance,
                RawAliasKind::Singleton => AliasKind::Singleton,
            },
            annotations: ast_builder::build_annotations(parser, annotations, file, &self.names),
            location,
            source_file: file,
            // rbs_raw bindings do not yet expose `comment` for member
            // nodes (`scripts/generate_rbs_raw.rb` skips `rbs_ast_comment`
            // for every variant).
            comment: rbs_raw_comment_unavailable(),
        }
    }

    fn build_class_or_module_member_ast(
        &mut self,
        parser: &Parser,
        node: *const ruby_rbs_sys::bindings::rbs_node,
        ns_prefix: &str,
        file: LoadFile,
        current_visibility: &mut Visibility,
    ) -> Option<ClassMember> {
        match classify_member(parser, node) {
            MemberKind::MethodDefinition {
                name,
                kind,
                overloads,
                annotations,
                overloading,
                visibility: mod_vis,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_method_definition_t) };
                    MethodDefinitionLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                        overloading_range: ffi_range_to_optional_location_range(
                            typed.overloading_range,
                        ),
                        visibility_range: ffi_range_to_optional_location_range(
                            typed.visibility_range,
                        ),
                    }
                });
                Some(ClassMember::Member(Member::MethodDefinition(
                    self.build_method_definition_from_kind(
                        parser,
                        name,
                        kind,
                        overloads,
                        annotations,
                        overloading,
                        mod_vis,
                        *current_visibility,
                        location,
                        file,
                    ),
                )))
            }
            MemberKind::Include {
                name,
                args,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_include_t) };
                    MixinMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        args_range: ffi_range_to_optional_location_range(typed.args_range),
                    }
                });
                Some(ClassMember::Member(Member::Include(IncludeMember {
                    name: self.ref_name(parser, name),
                    args: self.convert_type_args(parser, args),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            MemberKind::Extend {
                name,
                args,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_extend_t) };
                    MixinMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        args_range: ffi_range_to_optional_location_range(typed.args_range),
                    }
                });
                Some(ClassMember::Member(Member::Extend(ExtendMember {
                    name: self.ref_name(parser, name),
                    args: self.convert_type_args(parser, args),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            MemberKind::Prepend {
                name,
                args,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_prepend_t) };
                    MixinMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        args_range: ffi_range_to_optional_location_range(typed.args_range),
                    }
                });
                Some(ClassMember::Member(Member::Prepend(PrependMember {
                    name: self.ref_name(parser, name),
                    args: self.convert_type_args(parser, args),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            MemberKind::Alias {
                kind,
                old_name,
                new_name,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_alias_t) };
                    AliasMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        new_name_range: ffi_range_to_location_range(typed.new_name_range),
                        old_name_range: ffi_range_to_location_range(typed.old_name_range),
                        new_kind_range: ffi_range_to_optional_location_range(typed.new_kind_range),
                        old_kind_range: ffi_range_to_optional_location_range(typed.old_kind_range),
                    }
                });
                Some(ClassMember::Member(Member::Alias(
                    self.build_alias_from_kind(
                        parser,
                        kind,
                        old_name,
                        new_name,
                        annotations,
                        location,
                        file,
                    ),
                )))
            }
            MemberKind::NestedDecl(decl_ptr) => self
                .build_declaration_ast(parser, decl_ptr, ns_prefix, file)
                .map(ClassMember::Declaration),
            MemberKind::AttrReader {
                name,
                type_,
                ivar_name,
                kind,
                annotations,
                visibility,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_attr_reader_t) };
                    AttributeMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                        ivar_range: ffi_range_to_optional_location_range(typed.ivar_range),
                        ivar_name_range: ffi_range_to_optional_location_range(
                            typed.ivar_name_range,
                        ),
                        visibility_range: ffi_range_to_optional_location_range(
                            typed.visibility_range,
                        ),
                    }
                });
                Some(ClassMember::Member(Member::AttrReader(AttrReaderMember {
                    name: self.names.intern_symbol(parser.resolve_constant(name)),
                    ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                        AstType::Base(BaseType {
                            kind: BaseTypeKind::Any { todo: false },
                            location: None,
                        }),
                    ),
                    kind: if matches!(kind, RawAttributeKind::Singleton) {
                        AstAttributeKind::Singleton
                    } else {
                        AstAttributeKind::Instance
                    },
                    ivar_name: self.attr_ivar_from_raw(parser, ivar_name),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                    visibility: to_attr_ast_visibility(visibility),
                })))
            }
            MemberKind::AttrWriter {
                name,
                type_,
                ivar_name,
                kind,
                annotations,
                visibility,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_attr_writer_t) };
                    AttributeMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                        ivar_range: ffi_range_to_optional_location_range(typed.ivar_range),
                        ivar_name_range: ffi_range_to_optional_location_range(
                            typed.ivar_name_range,
                        ),
                        visibility_range: ffi_range_to_optional_location_range(
                            typed.visibility_range,
                        ),
                    }
                });
                Some(ClassMember::Member(Member::AttrWriter(AttrWriterMember {
                    name: self.names.intern_symbol(parser.resolve_constant(name)),
                    ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                        AstType::Base(BaseType {
                            kind: BaseTypeKind::Any { todo: false },
                            location: None,
                        }),
                    ),
                    kind: if matches!(kind, RawAttributeKind::Singleton) {
                        AstAttributeKind::Singleton
                    } else {
                        AstAttributeKind::Instance
                    },
                    ivar_name: self.attr_ivar_from_raw(parser, ivar_name),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                    visibility: to_attr_ast_visibility(visibility),
                })))
            }
            MemberKind::AttrAccessor {
                name,
                type_,
                ivar_name,
                kind,
                annotations,
                visibility,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_attr_accessor_t) };
                    AttributeMemberLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                        ivar_range: ffi_range_to_optional_location_range(typed.ivar_range),
                        ivar_name_range: ffi_range_to_optional_location_range(
                            typed.ivar_name_range,
                        ),
                        visibility_range: ffi_range_to_optional_location_range(
                            typed.visibility_range,
                        ),
                    }
                });
                Some(ClassMember::Member(Member::AttrAccessor(
                    AttrAccessorMember {
                        name: self.names.intern_symbol(parser.resolve_constant(name)),
                        ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                            AstType::Base(BaseType {
                                kind: BaseTypeKind::Any { todo: false },
                                location: None,
                            }),
                        ),
                        kind: if matches!(kind, RawAttributeKind::Singleton) {
                            AstAttributeKind::Singleton
                        } else {
                            AstAttributeKind::Instance
                        },
                        ivar_name: self.attr_ivar_from_raw(parser, ivar_name),
                        annotations: ast_builder::build_annotations(
                            parser,
                            annotations,
                            file,
                            &self.names,
                        ),
                        location,
                        source_file: file,
                        comment: rbs_raw_comment_unavailable(),
                        visibility: to_attr_ast_visibility(visibility),
                    },
                )))
            }
            MemberKind::Public => {
                *current_visibility = Visibility::Public;
                Some(ClassMember::Member(Member::Public(PublicMember {
                    location: None,
                })))
            }
            MemberKind::Private => {
                *current_visibility = Visibility::Private;
                Some(ClassMember::Member(Member::Private(PrivateMember {
                    location: None,
                })))
            }
            MemberKind::InstanceVariable { name, type_ } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_instance_variable_t) };
                    VariableMemberLocation {
                        range,
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                    }
                });
                let (name, ty, location, comment) =
                    self.build_variable_fields(parser, name, type_, location);
                Some(ClassMember::Member(Member::InstanceVariable(
                    InstanceVariableMember {
                        name,
                        ty,
                        location,
                        source_file: file,
                        comment,
                    },
                )))
            }
            MemberKind::ClassInstanceVariable { name, type_ } => {
                let location = node_location_range(node).map(|range| {
                    let typed =
                        unsafe { &*(node as *const rbs_ast_members_class_instance_variable_t) };
                    VariableMemberLocation {
                        range,
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                    }
                });
                let (name, ty, location, comment) =
                    self.build_variable_fields(parser, name, type_, location);
                Some(ClassMember::Member(Member::ClassInstanceVariable(
                    ClassInstanceVariableMember {
                        name,
                        ty,
                        location,
                        source_file: file,
                        comment,
                    },
                )))
            }
            MemberKind::ClassVariable { name, type_ } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_members_class_variable_t) };
                    VariableMemberLocation {
                        range,
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                        kind_range: ffi_range_to_optional_location_range(typed.kind_range),
                    }
                });
                let (name, ty, location, comment) =
                    self.build_variable_fields(parser, name, type_, location);
                Some(ClassMember::Member(Member::ClassVariable(
                    ClassVariableMember {
                        name,
                        ty,
                        location,
                        source_file: file,
                        comment,
                    },
                )))
            }
            _ => None,
        }
    }

    /// Shared core for the three `MemberKind::{InstanceVariable,
    /// ClassVariable, ClassInstanceVariable}` builders. All three AST
    /// structs share the same 4-field shape, so the loader extracts the
    /// fields once and the arm constructs the appropriate
    /// `ClassMember::Member(Member::*)` variant. Mirrors the `build_attribute_inner`
    /// / `build_mixin_inner` pattern; the AST stays as three distinct
    /// structs (rbs port shape) rather than being collapsed.
    ///
    /// `comment` is intentionally unavailable until rbs_raw exposes comment
    /// fields.
    fn build_variable_fields(
        &self,
        parser: &Parser,
        name: *const ruby_rbs_sys::bindings::rbs_ast_symbol,
        type_: *const ruby_rbs_sys::bindings::rbs_node,
        location: Option<VariableMemberLocation>,
    ) -> (
        crate::name::Symbol,
        AstType,
        Option<VariableMemberLocation>,
        Option<crate::ast::comment::Comment>,
    ) {
        let name = self.names.intern_symbol(parser.resolve_constant(name));
        let ty = ast_builder::build_type(parser, type_, &self.names).unwrap_or(AstType::Base(
            BaseType {
                kind: BaseTypeKind::Any { todo: false },
                location: None,
            },
        ));
        (name, ty, location, rbs_raw_comment_unavailable())
    }

    fn build_declaration_ast(
        &mut self,
        parser: &Parser,
        node: *const ruby_rbs_sys::bindings::rbs_node,
        ns_prefix: &str,
        file: LoadFile,
    ) -> Option<Declaration> {
        match classify_decl(parser, node) {
            DeclKind::Class {
                name,
                type_params,
                super_class,
                members,
                annotations,
            } => {
                let name_str = qualified_name(ns_prefix, &parser.type_name_to_string(name));
                let super_class = super_class.map(|super_class_ptr| {
                    let super_node = super_class_ptr as *const ruby_rbs_sys::bindings::rbs_node;
                    let args = if let DeclKind::ClassSuper { args, .. } =
                        classify_decl(parser, super_node)
                    {
                        self.convert_type_args(parser, args)
                    } else {
                        vec![]
                    };
                    let super_raw = parser.class_super_type_name(super_class_ptr);
                    let location = node_location_range(super_node).map(|range| {
                        let typed = unsafe {
                            &*(super_class_ptr as *const rbs_ast_declarations_class_super_t)
                        };
                        ClassSuperLocation {
                            range,
                            name_range: ffi_range_to_location_range(typed.name_range),
                            args_range: ffi_range_to_optional_location_range(typed.args_range),
                        }
                    });
                    Super {
                        name: self.names.parse_type_name(&super_raw),
                        args,
                        location,
                        source_file: file,
                    }
                });
                let mut visibility = Visibility::Public;
                let members = members
                    .iter()
                    .filter_map(|member| {
                        self.build_class_or_module_member_ast(
                            parser,
                            member,
                            &name_str,
                            file,
                            &mut visibility,
                        )
                    })
                    .collect();
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_class_t) };
                    ClassDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        end_range: ffi_range_to_location_range(typed.end_range),
                        type_params_range: ffi_range_to_optional_location_range(
                            typed.type_params_range,
                        ),
                        lt_range: ffi_range_to_optional_location_range(typed.lt_range),
                    }
                });
                Some(Declaration::Class(Arc::new(Class {
                    name: self.names.parse_type_name(&name_str),
                    type_params: self.convert_decl_type_params(parser, type_params),
                    super_class,
                    members,
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::Module {
                name,
                type_params,
                self_types,
                members,
                annotations,
            } => {
                let name_str = qualified_name(ns_prefix, &parser.type_name_to_string(name));
                let self_types = self_types
                    .iter()
                    .filter_map(|self_type| {
                        if let DeclKind::ModuleSelf { name, args } =
                            classify_decl(parser, self_type)
                        {
                            let location = node_location_range(self_type).map(|range| {
                                let typed = unsafe {
                                    &*(self_type as *const rbs_ast_declarations_module_self_t)
                                };
                                ModuleSelfLocation {
                                    range,
                                    name_range: ffi_range_to_location_range(typed.name_range),
                                    args_range: ffi_range_to_optional_location_range(
                                        typed.args_range,
                                    ),
                                }
                            });
                            Some(SelfType {
                                name: self.ref_name(parser, name),
                                args: self.convert_type_args(parser, args),
                                location,
                                source_file: file,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut visibility = Visibility::Public;
                let members: Vec<ModuleMember> = members
                    .iter()
                    .filter_map(|member| {
                        self.build_class_or_module_member_ast(
                            parser,
                            member,
                            &name_str,
                            file,
                            &mut visibility,
                        )
                        .map(ModuleMember::from)
                    })
                    .collect();
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_module_t) };
                    ModuleDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        end_range: ffi_range_to_location_range(typed.end_range),
                        type_params_range: ffi_range_to_optional_location_range(
                            typed.type_params_range,
                        ),
                        colon_range: ffi_range_to_optional_location_range(typed.colon_range),
                        self_types_range: ffi_range_to_optional_location_range(
                            typed.self_types_range,
                        ),
                    }
                });
                Some(Declaration::Module(Arc::new(Module {
                    name: self.names.parse_type_name(&name_str),
                    type_params: self.convert_decl_type_params(parser, type_params),
                    self_types,
                    members,
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::Interface {
                name,
                type_params,
                members,
                annotations,
            } => {
                let name_str = qualified_name(ns_prefix, &parser.type_name_to_string(name));
                let members = members
                    .iter()
                    .filter_map(|member| match classify_member(parser, member) {
                        MemberKind::MethodDefinition {
                            name,
                            kind,
                            overloads,
                            annotations,
                            overloading,
                            visibility,
                        } => {
                            let location = node_location_range(member).map(|range| {
                                let typed = unsafe {
                                    &*(member as *const rbs_ast_members_method_definition_t)
                                };
                                MethodDefinitionLocation {
                                    range,
                                    keyword_range: ffi_range_to_location_range(typed.keyword_range),
                                    name_range: ffi_range_to_location_range(typed.name_range),
                                    kind_range: ffi_range_to_optional_location_range(
                                        typed.kind_range,
                                    ),
                                    overloading_range: ffi_range_to_optional_location_range(
                                        typed.overloading_range,
                                    ),
                                    visibility_range: ffi_range_to_optional_location_range(
                                        typed.visibility_range,
                                    ),
                                }
                            });
                            Some(Member::MethodDefinition(
                                self.build_method_definition_from_kind(
                                    parser,
                                    name,
                                    kind,
                                    overloads,
                                    annotations,
                                    overloading,
                                    visibility,
                                    // rbs `build_interface` forces accessibility :public.
                                    // No surrounding `public` / `private` marker exists
                                    // inside interfaces, so Visibility::Public is the
                                    // canonical current_default.
                                    Visibility::Public,
                                    location,
                                    file,
                                ),
                            ))
                        }
                        MemberKind::Alias {
                            kind,
                            old_name,
                            new_name,
                            annotations,
                        } => {
                            let location = node_location_range(member).map(|range| {
                                let typed = unsafe { &*(member as *const rbs_ast_members_alias_t) };
                                AliasMemberLocation {
                                    range,
                                    keyword_range: ffi_range_to_location_range(typed.keyword_range),
                                    new_name_range: ffi_range_to_location_range(
                                        typed.new_name_range,
                                    ),
                                    old_name_range: ffi_range_to_location_range(
                                        typed.old_name_range,
                                    ),
                                    new_kind_range: ffi_range_to_optional_location_range(
                                        typed.new_kind_range,
                                    ),
                                    old_kind_range: ffi_range_to_optional_location_range(
                                        typed.old_kind_range,
                                    ),
                                }
                            });
                            Some(Member::Alias(self.build_alias_from_kind(
                                parser,
                                kind,
                                old_name,
                                new_name,
                                annotations,
                                location,
                                file,
                            )))
                        }
                        MemberKind::Include {
                            name,
                            args,
                            annotations,
                        } => {
                            let location = node_location_range(member).map(|range| {
                                let typed =
                                    unsafe { &*(member as *const rbs_ast_members_include_t) };
                                MixinMemberLocation {
                                    range,
                                    keyword_range: ffi_range_to_location_range(typed.keyword_range),
                                    name_range: ffi_range_to_location_range(typed.name_range),
                                    args_range: ffi_range_to_optional_location_range(
                                        typed.args_range,
                                    ),
                                }
                            });
                            Some(Member::Include(IncludeMember {
                                name: self.ref_name(parser, name),
                                args: self.convert_type_args(parser, args),
                                annotations: ast_builder::build_annotations(
                                    parser,
                                    annotations,
                                    file,
                                    &self.names,
                                ),
                                location,
                                source_file: file,
                                comment: rbs_raw_comment_unavailable(),
                            }))
                        }
                        // rbs's C parser rejects `extend` / `prepend` inside
                        // an interface declaration at parse time
                        // (`rbs/src/parser.c:2126-2131`), so `MemberKind::Extend`
                        // and `MemberKind::Prepend` are unreachable here in
                        // practice. The default `_ => None` arm covers the
                        // theoretical case along with non-mixin members rbs
                        // already rejects (Constant, Attr, Visibility marker, etc.).
                        _ => None,
                    })
                    .collect();
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_interface_t) };
                    InterfaceDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        end_range: ffi_range_to_location_range(typed.end_range),
                        type_params_range: ffi_range_to_optional_location_range(
                            typed.type_params_range,
                        ),
                    }
                });
                Some(Declaration::Interface(Arc::new(Interface {
                    name: self.names.parse_type_name(&name_str),
                    type_params: self.convert_decl_type_params(parser, type_params),
                    members,
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::TypeAlias {
                name,
                type_params,
                type_,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_type_alias_t) };
                    TypeAliasDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        name_range: ffi_range_to_location_range(typed.name_range),
                        eq_range: ffi_range_to_location_range(typed.eq_range),
                        type_params_range: ffi_range_to_optional_location_range(
                            typed.type_params_range,
                        ),
                    }
                });
                Some(Declaration::TypeAlias(Arc::new(TypeAlias {
                    name: self.decl_name(parser, name, ns_prefix),
                    type_params: self.convert_decl_type_params(parser, type_params),
                    ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                        AstType::Base(BaseType {
                            kind: BaseTypeKind::Any { todo: false },
                            location: None,
                        }),
                    ),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::Constant {
                name,
                type_,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_constant_t) };
                    ConstantDeclarationLocation {
                        range,
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                    }
                });
                Some(Declaration::Constant(Arc::new(Constant {
                    name: self.decl_name(parser, name, ns_prefix),
                    ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                        AstType::Base(BaseType {
                            kind: BaseTypeKind::Any { todo: false },
                            location: None,
                        }),
                    ),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::Global {
                name,
                type_,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_global_t) };
                    GlobalDeclarationLocation {
                        range,
                        name_range: ffi_range_to_location_range(typed.name_range),
                        colon_range: ffi_range_to_location_range(typed.colon_range),
                    }
                });
                Some(Declaration::Global(Arc::new(Global {
                    name: self.names.intern_symbol(parser.resolve_constant(name)),
                    ty: ast_builder::build_type(parser, type_, &self.names).unwrap_or(
                        AstType::Base(BaseType {
                            kind: BaseTypeKind::Any { todo: false },
                            location: None,
                        }),
                    ),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::ClassAlias {
                new_name,
                old_name,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_class_alias_t) };
                    AliasDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        new_name_range: ffi_range_to_location_range(typed.new_name_range),
                        eq_range: ffi_range_to_location_range(typed.eq_range),
                        old_name_range: ffi_range_to_location_range(typed.old_name_range),
                    }
                });
                Some(Declaration::ClassAlias(Arc::new(ClassAlias {
                    new_name: self.decl_name(parser, new_name, ns_prefix),
                    old_name: self.ref_name(parser, old_name),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            DeclKind::ModuleAlias {
                new_name,
                old_name,
                annotations,
            } => {
                let location = node_location_range(node).map(|range| {
                    let typed = unsafe { &*(node as *const rbs_ast_declarations_module_alias_t) };
                    AliasDeclarationLocation {
                        range,
                        keyword_range: ffi_range_to_location_range(typed.keyword_range),
                        new_name_range: ffi_range_to_location_range(typed.new_name_range),
                        eq_range: ffi_range_to_location_range(typed.eq_range),
                        old_name_range: ffi_range_to_location_range(typed.old_name_range),
                    }
                });
                Some(Declaration::ModuleAlias(Arc::new(ModuleAlias {
                    new_name: self.decl_name(parser, new_name, ns_prefix),
                    old_name: self.ref_name(parser, old_name),
                    annotations: ast_builder::build_annotations(
                        parser,
                        annotations,
                        file,
                        &self.names,
                    ),
                    location,
                    source_file: file,
                    comment: rbs_raw_comment_unavailable(),
                })))
            }
            _ => None,
        }
    }

    /// Convert the `args: RawNodeList` of an include/extend/prepend/class-super
    /// into a `Vec<AstType>` of type-argument ast nodes. Resolution to `Ty`
    /// happens in `LegacyEnvironmentBuffer::resolve` together with the
    /// captured `context` and `type_param_scope` of the enclosing mixin.
    fn convert_type_args(&self, parser: &Parser, args: RawNodeList) -> Vec<AstType> {
        args.iter()
            .map(|arg_ptr| {
                ast_builder::build_type(parser, arg_ptr, &self.names).unwrap_or(AstType::Base(
                    BaseType {
                        kind: BaseTypeKind::Any { todo: false },
                        location: None,
                    },
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod magic_comment_tests {
    use super::*;

    #[test]
    fn test_magic_comment_false() {
        let src = b"# resolve-type-names: false\nmodule Foo\nend\n";
        let got = parse_magic_comment(src).expect("should match");
        assert!(!got.value);
    }

    #[test]
    fn test_magic_comment_true() {
        let src = b"# resolve-type-names: true\nmodule Foo\nend\n";
        let got = parse_magic_comment(src).expect("should match");
        assert!(got.value);
    }

    #[test]
    fn test_magic_comment_absent() {
        let src = b"module Foo\nend\n";
        assert!(parse_magic_comment(src).is_none());
    }

    #[test]
    fn test_magic_comment_not_first_line() {
        let src = b"module Foo\nend\n# resolve-type-names: false\n";
        assert!(parse_magic_comment(src).is_none());
    }

    #[test]
    fn test_magic_comment_unrelated_comment_first_line() {
        let src = b"# some other comment\n# resolve-type-names: false\n";
        assert!(parse_magic_comment(src).is_none());
    }

    #[test]
    fn test_magic_comment_no_space_after_colon() {
        // rbs regex requires `\s+` after the colon.
        let src = b"# resolve-type-names:false\nmodule Foo\nend\n";
        assert!(parse_magic_comment(src).is_none());
    }

    #[test]
    fn test_magic_comment_invalid_value() {
        let src = b"# resolve-type-names: maybe\nmodule Foo\nend\n";
        assert!(parse_magic_comment(src).is_none());
    }

    #[test]
    fn test_magic_comment_crlf() {
        let src = b"# resolve-type-names: false\r\nmodule Foo\r\nend\r\n";
        let got = parse_magic_comment(src).expect("should match across CRLF");
        assert!(!got.value);
    }
}
