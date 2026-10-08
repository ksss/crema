//! Insertion path for Ruby declaration AST into the
//! [`EnvironmentDraft`].
//!
//! The inline parser collects a tree of `ast::ruby::*` declarations and
//! hands them to this layer, which forwards each top-level decl to the
//! draft and flattens nested constants. Phase E (c) of ADR-0017 reduced
//! this file to the draft path only; the legacy push API is gone.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ast::declarations::ConstantDeclaration as Constant;
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::declarations::{ConstantDecl, ConstantValueKind, Declaration};
use crate::ast::ruby::members::{AttributeMember, Member};
use crate::ast::types::{BaseType, BaseTypeKind, ClassInstanceType, Type as AstType};
use crate::ast_builder;
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::DeclOrigin;
use crate::environment::draft::{
    BuildError, ClassAliasDraft, ClassDeclarationDraft, ClassOrModuleDraft,
    Context as DraftContext, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::location::{LocationRange, SourceLocation};
use crate::name::NameTable;
use crate::rbs_raw::Parser as RbsParser;
use crate::source_ref::SourceRef;
use crate::type_name::TypeName;

/// Per-source bookkeeping threaded through the recursion: the original
/// Ruby bytes (for echoing annotation text in `AnnotationSyntaxError`),
/// the on-disk path (for the `Diagnostic.file` field), and the
/// diagnostic sink.
struct LoadCtx<'a> {
    source: SourceRef<'a>,
    file: Option<&'a Path>,
    diagnostics: &'a mut Vec<Diagnostic>,
}

impl LoadCtx<'_> {
    fn file_name(&self, names: &NameTable) -> Option<crate::name::Name> {
        self.file.map(|p| names.intern(&p.to_string_lossy()))
    }
}

impl EnvironmentDraft {
    /// Insert a single Ruby declaration tree into this draft. Top-level
    /// callers iterate over the inline collector's `Vec<Declaration>` and
    /// call this once per top-level declaration.
    pub fn insert_ruby_decl(
        &mut self,
        decl: &Declaration,
        source: SourceRef<'_>,
        file: Option<&Path>,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        let mut ctx = LoadCtx {
            source,
            file,
            diagnostics,
        };

        let draft_ctx: DraftContext = Arc::from([self.names.absolute_root()]);
        let file_name: DeclOrigin = ctx
            .file
            .map(|p| self.names.intern(&p.to_string_lossy()))
            .into();
        match decl {
            Declaration::Class(class_decl) => {
                let name_raw = class_decl.class_name;
                let conflicting_source = self.conflicting_constant_source(name_raw);
                if let Err(err) = self.insert_ruby_class_decl(
                    file_name,
                    Arc::clone(&draft_ctx),
                    name_raw,
                    Arc::clone(class_decl),
                ) {
                    push_build_error_diagnostic(
                        err,
                        &self.names,
                        &mut ctx,
                        Some(class_decl.name_location),
                        conflicting_source,
                    );
                    return;
                }
                self.flatten_nested_constants_into_draft(&class_decl.members, &mut ctx);
                check_attr_annotations(&class_decl.members, &mut ctx, &self.names);
            }
            Declaration::Module(module_decl) => {
                let name_raw = module_decl.module_name;
                let conflicting_source = self.conflicting_constant_source(name_raw);
                if let Err(err) = self.insert_ruby_module_decl(
                    file_name,
                    Arc::clone(&draft_ctx),
                    name_raw,
                    Arc::clone(module_decl),
                ) {
                    push_build_error_diagnostic(
                        err,
                        &self.names,
                        &mut ctx,
                        Some(module_decl.name_location),
                        conflicting_source,
                    );
                    return;
                }
                self.flatten_nested_constants_into_draft(&module_decl.members, &mut ctx);
                check_attr_annotations(&module_decl.members, &mut ctx, &self.names);
            }
            Declaration::Constant(constant_decl) => {
                let name_raw = constant_decl.constant_name;
                let conflicting_source = self.conflicting_constant_source(name_raw);
                let ty = constant_decl_ast_type(constant_decl, &mut ctx, &self.names);
                let rbs_constant = Arc::new(Constant {
                    name: name_raw,
                    ty,
                    location: Some(constant_decl_location(constant_decl.name_location)),
                    annotations: Vec::new(),
                    comment: None,
                });
                if let Err(err) = self.insert_constant(file_name, draft_ctx, rbs_constant) {
                    push_build_error_diagnostic(
                        err,
                        &self.names,
                        &mut ctx,
                        Some(constant_decl.name_location),
                        conflicting_source,
                    );
                }
            }
            Declaration::ClassModuleAlias(alias_decl) => {
                let name_raw = alias_decl.new_name;
                let conflicting_source = self.conflicting_constant_source(name_raw);
                let arc = Arc::new(alias_decl.clone());
                if let Err(err) = self.insert_ruby_alias(file_name, draft_ctx, arc) {
                    push_build_error_diagnostic(
                        err,
                        &self.names,
                        &mut ctx,
                        Some(alias_decl.name_location),
                        conflicting_source,
                    );
                }
            }
        }
    }

    /// Walk the nested member tree of a Ruby class/module and push every
    /// `Declaration::Constant` (qualified-name absolute) into the draft's
    /// top-level `constant_decls`. Mirrors the legacy path's recursive
    /// flattening (`load_class_or_module` → `load_member` →
    /// `Declaration::Constant` arm in [`load_declaration`]). Top-level
    /// constants are handled by the caller before this walker runs;
    /// only `Class` / `Module` arms recurse here.
    ///
    /// `qualified_name` on every Ruby decl is already absolute (the
    /// inline collector qualifies against the enclosing scope), so the
    /// flattened entries can be inserted at the draft's root context.
    fn flatten_nested_constants_into_draft(&mut self, members: &[Member], ctx: &mut LoadCtx<'_>) {
        let draft_ctx: DraftContext = Arc::from([self.names.absolute_root()]);
        for member in members {
            let Member::Declaration(decl) = member else {
                continue;
            };
            match decl {
                Declaration::Constant(constant_decl) => {
                    let ty = constant_decl_ast_type(constant_decl, ctx, &self.names);
                    let rbs_constant = Arc::new(Constant {
                        name: constant_decl.constant_name,
                        ty,
                        location: Some(constant_decl_location(constant_decl.name_location)),
                        annotations: Vec::new(),
                        comment: None,
                    });
                    if let Err(err) = self.insert_constant(
                        ctx.file_name(&self.names).into(),
                        Arc::clone(&draft_ctx),
                        rbs_constant,
                    ) {
                        push_build_error_diagnostic(err, &self.names, ctx, None, None);
                    }
                }
                Declaration::Class(class_decl) => {
                    self.flatten_nested_constants_into_draft(&class_decl.members, ctx);
                }
                Declaration::Module(module_decl) => {
                    self.flatten_nested_constants_into_draft(&module_decl.members, ctx);
                }
                // Nested aliases are flattened during draft.build() via
                // `resolve_ruby_member`, which pushes them into
                // `FlattenedDecls.ruby_{class,module}_aliases`. Nothing to do here.
                Declaration::ClassModuleAlias(_) => {}
            }
        }
    }

    fn conflicting_constant_source(&self, name: TypeName) -> Option<SourceLocation> {
        if let Some(entry) = self.class_decls.get(&name) {
            return class_or_module_source_location(entry, &self.names);
        }
        if let Some(entry) = self.constant_decls.get(&name) {
            return entry
                .file
                .file()
                .zip(entry.decl.location)
                .map(|(file, location)| source_location(file, location.name_range, &self.names));
        }
        if let Some(entry) = self.class_alias_decls.get(&name) {
            return class_alias_source_location(entry, &self.names);
        }
        None
    }
}

fn push_build_error_diagnostic(
    err: BuildError,
    names: &NameTable,
    ctx: &mut LoadCtx<'_>,
    range: Option<PrismByteRange>,
    conflicting_source: Option<SourceLocation>,
) {
    let name = match &err {
        BuildError::DuplicatedDeclaration { name }
        | BuildError::GenericParameterMismatch { name }
        | BuildError::SuperclassConflict { name } => names.display_type_name(*name),
        BuildError::DuplicatedGlobal { name } => names.resolve(*name).to_string(),
    };
    let range = match (&err, range) {
        (BuildError::DuplicatedDeclaration { .. }, Some(range)) => range,
        _ => (0, 0),
    };
    ctx.diagnostics.push(Diagnostic {
        scope: None,
        kind: DiagnosticKind::DuplicatedDeclaration {
            name,
            detail: err.format_with(names),
            conflicting_source,
        },
        location: Diagnostic::location_for_byte_range(
            ctx.file.map(|p| p.to_path_buf()).unwrap_or_default(),
            ctx.source.bytes(),
            range.0 as usize,
            range.1 as usize,
        ),
    });
}

fn class_or_module_source_location(
    entry: &ClassOrModuleDraft,
    names: &NameTable,
) -> Option<SourceLocation> {
    match entry {
        ClassOrModuleDraft::Class(entry) => entry
            .context_decls
            .iter()
            .rev()
            .find_map(|(file, _, decl)| class_source_location(file.file(), decl, names)),
        ClassOrModuleDraft::Module(entry) => entry
            .context_decls
            .iter()
            .rev()
            .find_map(|(file, _, decl)| module_source_location(file.file(), decl, names)),
    }
}

fn class_source_location(
    file: Option<crate::name::Name>,
    decl: &ClassDeclarationDraft,
    names: &NameTable,
) -> Option<SourceLocation> {
    match decl {
        ClassDeclarationDraft::Signature(decl) => decl
            .source_file
            .or(file)
            .zip(decl.location)
            .map(|(file, location)| source_location(file, location.name_range, names)),
        ClassDeclarationDraft::Ruby(decl) => {
            file.map(|file| ruby_source_location(file, decl.name_location, names))
        }
    }
}

fn module_source_location(
    file: Option<crate::name::Name>,
    decl: &ModuleDeclarationDraft,
    names: &NameTable,
) -> Option<SourceLocation> {
    match decl {
        ModuleDeclarationDraft::Signature(decl) => decl
            .source_file
            .or(file)
            .zip(decl.location)
            .map(|(file, location)| source_location(file, location.name_range, names)),
        ModuleDeclarationDraft::Ruby(decl) => {
            file.map(|file| ruby_source_location(file, decl.name_location, names))
        }
    }
}

fn class_alias_source_location(
    entry: &crate::environment::draft::ClassAliasEntry,
    names: &NameTable,
) -> Option<SourceLocation> {
    match &entry.decl {
        ClassAliasDraft::Class(decl) => decl
            .source_file
            .or(entry.file.file())
            .zip(decl.location)
            .map(|(file, location)| source_location(file, location.new_name_range, names)),
        ClassAliasDraft::Module(decl) => decl
            .source_file
            .or(entry.file.file())
            .zip(decl.location)
            .map(|(file, location)| source_location(file, location.new_name_range, names)),
        ClassAliasDraft::Ruby(decl) => entry
            .file
            .file()
            .map(|file| ruby_source_location(file, decl.name_location, names)),
    }
}

fn source_location(
    file: crate::name::Name,
    range: LocationRange,
    names: &NameTable,
) -> SourceLocation {
    SourceLocation {
        file: PathBuf::from(names.resolve(file)),
        range,
    }
}

fn ruby_source_location(
    file: crate::name::Name,
    range: PrismByteRange,
    names: &NameTable,
) -> SourceLocation {
    source_location(file, prism_range_to_location_range(range), names)
}

fn prism_range_to_location_range(range: PrismByteRange) -> LocationRange {
    LocationRange::new(range.0, range.0, range.1, range.1)
}

fn constant_decl_location(range: PrismByteRange) -> crate::location::ConstantDeclarationLocation {
    let range = prism_range_to_location_range(range);
    crate::location::ConstantDeclarationLocation {
        range,
        name_range: range,
        colon_range: range,
    }
}

/// Eager-parse every `attr_reader` / `attr_writer` / `attr_accessor`
/// trailing `#: T` annotation under this member tree and push an
/// `AnnotationSyntaxError` per attribute (not per name) when the parse
/// fails. Mirrors rbs's `parse_attribute_call` (`rbs/lib/rbs/inline_parser.rb`
/// L374-L378): one diagnostic per attribute call, no multiplication by
/// the number of names. The build-side fallback in `definition_builder`
/// continues to silently re-parse and collapse to `Ty::UNTYPED`,
/// suppressing downstream cascades.
fn check_attr_annotations(members: &[Member], ctx: &mut LoadCtx<'_>, names: &NameTable) {
    for member in members {
        match member {
            Member::AttrReader(r) => check_one_attr_annotation(&r.attribute, ctx, names),
            Member::AttrWriter(w) => check_one_attr_annotation(&w.attribute, ctx, names),
            Member::AttrAccessor(a) => check_one_attr_annotation(&a.attribute, ctx, names),
            Member::Declaration(Declaration::Class(c)) => {
                check_attr_annotations(&c.members, ctx, names);
            }
            Member::Declaration(Declaration::Module(m)) => {
                check_attr_annotations(&m.members, ctx, names);
            }
            _ => {}
        }
    }
}

fn check_one_attr_annotation(
    attribute: &AttributeMember,
    ctx: &mut LoadCtx<'_>,
    names: &NameTable,
) {
    let (Some(text), Some(range)) = (attribute.type_text.as_ref(), attribute.annotation_range)
    else {
        return;
    };
    if let Err(err) = parse_rbs_type(text.as_bytes(), names) {
        ctx.diagnostics.push(build_annotation_syntax_error(
            ctx.source.bytes(),
            ctx.file,
            range,
            err,
        ));
    }
}

/// A broken `#: T` collapses to literal-based inference (not
/// `Untyped`) so the right-hand side keeps its precision.
fn constant_decl_ast_type(cd: &ConstantDecl, ctx: &mut LoadCtx<'_>, names: &NameTable) -> AstType {
    match &cd.type_text {
        Some(text) => match parse_rbs_type(text.as_bytes(), names) {
            Ok(t) => t,
            Err(err) => {
                if let Some(range) = cd.annotation_range {
                    ctx.diagnostics.push(build_annotation_syntax_error(
                        ctx.source.bytes(),
                        ctx.file,
                        range,
                        err,
                    ));
                }
                constant_value_kind_to_ast_type(cd.value_kind, names)
            }
        },
        None => constant_value_kind_to_ast_type(cd.value_kind, names),
    }
}

fn constant_value_kind_to_ast_type(kind: ConstantValueKind, names: &NameTable) -> AstType {
    let class = |name: &TypeName| {
        AstType::ClassInstance(ClassInstanceType {
            name: *name,
            args: vec![],
            location: None,
        })
    };
    let builtins = names.builtins();
    match kind {
        ConstantValueKind::Integer => class(&builtins.integer),
        ConstantValueKind::Float => class(&builtins.float),
        ConstantValueKind::String => class(&builtins.string),
        ConstantValueKind::True | ConstantValueKind::False => AstType::Base(BaseType {
            kind: BaseTypeKind::Bool,
            location: None,
        }),
        ConstantValueKind::Symbol => class(&builtins.symbol),
        ConstantValueKind::Nil => AstType::Base(BaseType {
            kind: BaseTypeKind::Nil,
            location: None,
        }),
        ConstantValueKind::Other => AstType::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        }),
    }
}

/// Returns `None` for sources with no file identity (eval / `-e`
/// without a path).
/// echoed back in the message.
pub(crate) fn build_annotation_syntax_error(
    source: &[u8],
    file: Option<&Path>,
    range: PrismByteRange,
    parser_error: String,
) -> Diagnostic {
    let annotation_text = source
        .get(range.0 as usize..range.1 as usize)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .unwrap_or("")
        .to_string();
    let parser_error = if parser_error.is_empty() {
        None
    } else {
        Some(parser_error)
    };
    Diagnostic {
        scope: None,
        kind: DiagnosticKind::AnnotationSyntaxError {
            annotation_text,
            parser_error,
        },
        location: Diagnostic::location_for_byte_range(
            file.map(|p| p.to_path_buf()).unwrap_or_default(),
            source,
            range.0 as usize,
            range.1 as usize,
        ),
    }
}

pub(crate) fn parse_rbs_type(type_text: &[u8], names: &NameTable) -> Result<AstType, String> {
    let (parser, type_node) = RbsParser::parse_type(type_text)?;
    Ok(
        ast_builder::build_type(&parser, type_node, names).unwrap_or(AstType::Base(BaseType {
            kind: BaseTypeKind::Any { todo: false },
            location: None,
        })),
    )
}
