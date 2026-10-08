// Ported from ADR-0028 spike 1 verbatim; refactor forbidden by slice 1a scope.
#![allow(clippy::too_many_arguments)]

//! Frozen `Environment` → mirror conversion, collecting scale stats.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;

use crate::ast::TypeParam;
use crate::ast::declarations as cd;
use crate::ast::members as cm;
use crate::ast::method_type::MethodType;
use crate::ast::types as ct;
use crate::environment::frozen::{
    ClassAliasDeclaration, ClassDeclaration, ClassOrModule, ClassOrModuleAliasEntry, Environment,
    ModuleAliasDeclaration, ModuleDeclaration, NormalizeModuleNameResult,
};
use crate::location as cl;
use crate::name::{Name, NameTable, Symbol};
use crate::type_name::TypeName;

use crate::snapshot::mirror::*;

const ABSOLUTE_ROOT_HASH: u64 = 0xA850_1075_0001_0001;
const RELATIVE_ROOT_HASH: u64 = 0x8E1A_71FE_0001_0001;

pub fn fold(h: u64) -> u64 {
    if h == 0 { 1 } else { h }
}

pub fn sym_id_of(s: &str) -> u64 {
    fold(xxh3_64(s.as_bytes()))
}

pub fn child_hash(parent: u64, segment: u64) -> u64 {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&parent.to_le_bytes());
    buf[8..].copy_from_slice(&segment.to_le_bytes());
    fold(xxh3_64(&buf))
}

#[derive(Default)]
pub struct Stats {
    pub decls: u64,
    pub members: u64,
    pub method_defs: u64,
    pub overloads: u64,
    pub type_nodes: u64,
    pub max_type_depth: u64,
    pub annotations: u64,
    pub comments: u64,
    pub nested_decl_refs: u64,
}

pub struct Cx<'a> {
    pub names: &'a NameTable,
    pub symbols: FxMap<Box<str>>,
    pub type_names: FxMap<MTnEntry>,
    pub name_ids: FxHashMap<String, u32>,
    pub names_vec: Vec<String>,
    pub stats: Stats,
}

impl<'a> Cx<'a> {
    pub fn new(names: &'a NameTable) -> Self {
        Cx {
            names,
            symbols: FxMap::default(),
            type_names: FxMap::default(),
            name_ids: FxHashMap::default(),
            names_vec: Vec::new(),
            stats: Stats::default(),
        }
    }

    pub(crate) fn sym(&mut self, s: Symbol) -> Sym {
        let string = self.names.resolve(s);
        let id = sym_id_of(string);
        self.symbols.entry(id).or_insert_with(|| string.into());
        id
    }

    /// Register `t` (and its parent chain) into the mirror type-name table,
    /// verifying crema's content-addressed recipe against a local
    /// re-computation — a drift check on the replicated hash recipe.
    pub(crate) fn tn(&mut self, t: TypeName) -> Tn {
        let id = t.get();
        if self.type_names.contains_key(&id) {
            return id;
        }
        match self.names.type_name_parent(t) {
            None => {
                assert!(
                    id == ABSOLUTE_ROOT_HASH || id == RELATIVE_ROOT_HASH,
                    "root TypeName id mismatch: {:#x}",
                    id
                );
                let absolute = id == ABSOLUTE_ROOT_HASH;
                self.type_names.insert(id, (0, 0, absolute));
            }
            Some(parent) => {
                let parent_id = self.tn(parent);
                let seg = self
                    .names
                    .last_segment(t)
                    .expect("non-root TypeName has a segment");
                let seg_id = self.sym(seg);
                assert_eq!(
                    child_hash(parent_id, seg_id),
                    id,
                    "TypeName recipe drift for {}",
                    self.names.display_type_name(t)
                );
                let absolute = self.names.type_name_is_absolute(t);
                self.type_names.insert(id, (parent_id, seg_id, absolute));
            }
        }
        id
    }

    pub(crate) fn name_id(&mut self, n: Name) -> NameId {
        let s = self.names.resolve(n);
        if let Some(&id) = self.name_ids.get(s) {
            return id;
        }
        let id = self.names_vec.len() as u32;
        self.names_vec.push(s.to_string());
        self.name_ids.insert(s.to_string(), id);
        id
    }

    pub(crate) fn opt_name(&mut self, n: Option<Name>) -> Option<NameId> {
        n.map(|n| self.name_id(n))
    }

    pub(crate) fn context(&mut self, ctx: &[TypeName]) -> MContext {
        ctx.iter().map(|t| self.tn(*t)).collect()
    }
}

pub(crate) fn range(r: cl::LocationRange) -> MRange {
    MRange(r.start_char, r.start_byte, r.end_char, r.end_byte)
}

fn opt_range(r: Option<cl::LocationRange>) -> Option<MRange> {
    r.map(range)
}

fn name_args_loc(
    r: cl::LocationRange,
    name_range: cl::LocationRange,
    args_range: Option<cl::LocationRange>,
) -> MNameArgsLoc {
    MNameArgsLoc {
        range: range(r),
        name_range: range(name_range),
        args_range: opt_range(args_range),
    }
}

fn visibility(v: cm::Visibility) -> u8 {
    match v {
        cm::Visibility::Public => 0,
        cm::Visibility::Private => 1,
    }
}

fn ast_visibility(v: crate::ast::Visibility) -> u8 {
    match v {
        crate::ast::Visibility::Public => 0,
        crate::ast::Visibility::Private => 1,
    }
}

pub(crate) fn method_kind(k: cm::MethodKind) -> u8 {
    match k {
        cm::MethodKind::Instance => 0,
        cm::MethodKind::Singleton => 1,
        cm::MethodKind::SingletonInstance => 2,
    }
}

impl<'a> Cx<'a> {
    pub(crate) fn ty(&mut self, t: &ct::Type, depth: u64) -> MType {
        self.stats.type_nodes += 1;
        if depth > self.stats.max_type_depth {
            self.stats.max_type_depth = depth;
        }
        match t {
            ct::Type::Base(b) => MType::Base(MBaseType {
                kind: match &b.kind {
                    ct::BaseTypeKind::Bool => MBaseTypeKind::Bool,
                    ct::BaseTypeKind::Void => MBaseTypeKind::Void,
                    ct::BaseTypeKind::Any { todo } => MBaseTypeKind::Any { todo: *todo },
                    ct::BaseTypeKind::Nil => MBaseTypeKind::Nil,
                    ct::BaseTypeKind::Top => MBaseTypeKind::Top,
                    ct::BaseTypeKind::Bottom => MBaseTypeKind::Bottom,
                    ct::BaseTypeKind::SelfType => MBaseTypeKind::SelfType,
                    ct::BaseTypeKind::Instance => MBaseTypeKind::Instance,
                    ct::BaseTypeKind::Class => MBaseTypeKind::Class,
                },
                location: opt_range(b.location),
            }),
            ct::Type::Variable(v) => MType::Variable(MVariableType {
                name: self.sym(v.name),
                location: opt_range(v.location),
            }),
            ct::Type::ClassSingleton(n) => MType::ClassSingleton(MNamedType {
                name: self.tn(n.name),
                args: n.args.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: n
                    .location
                    .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            }),
            ct::Type::Interface(n) => MType::Interface(MNamedType {
                name: self.tn(n.name),
                args: n.args.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: n
                    .location
                    .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            }),
            ct::Type::ClassInstance(n) => MType::ClassInstance(MNamedType {
                name: self.tn(n.name),
                args: n.args.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: n
                    .location
                    .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            }),
            ct::Type::Alias(n) => MType::Alias(MNamedType {
                name: self.tn(n.name),
                args: n.args.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: n
                    .location
                    .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            }),
            ct::Type::Tuple(t) => MType::Tuple(MTypes {
                types: t.types.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: opt_range(t.location),
            }),
            ct::Type::Record(r) => MType::Record(MRecordType {
                fields: r
                    .fields
                    .iter()
                    .map(|f| MRecordField {
                        key: match &f.key {
                            ct::RecordKey::Symbol(s) => MRecordKey::Symbol(self.sym(*s)),
                            ct::RecordKey::String(s) => MRecordKey::String(s.clone()),
                            ct::RecordKey::Integer(s) => MRecordKey::Integer(s.clone()),
                            ct::RecordKey::Bool(b) => MRecordKey::Bool(*b),
                        },
                        ty: self.ty(&f.ty, depth + 1),
                        required: f.required,
                    })
                    .collect(),
                location: opt_range(r.location),
            }),
            ct::Type::Optional(o) => MType::Optional(MOptionalType {
                ty: Box::new(self.ty(&o.ty, depth + 1)),
                location: opt_range(o.location),
            }),
            ct::Type::Union(u) => MType::Union(MTypes {
                types: u.types.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: opt_range(u.location),
            }),
            ct::Type::Intersection(i) => MType::Intersection(MTypes {
                types: i.types.iter().map(|a| self.ty(a, depth + 1)).collect(),
                location: opt_range(i.location),
            }),
            ct::Type::Proc(p) => MType::Proc(Box::new(MProcType {
                function: self.function(&p.function, depth + 1),
                block: p.block.as_ref().map(|b| self.block(b, depth + 1)),
                self_type: p
                    .self_type
                    .as_ref()
                    .map(|t| Box::new(self.ty(t, depth + 1))),
                location: opt_range(p.location),
            })),
            ct::Type::Literal(l) => MType::Literal(MLiteralType {
                literal: match &l.literal {
                    ct::Literal::Integer(s) => MLiteral::Integer(s.clone()),
                    ct::Literal::String(s) => MLiteral::String(s.clone()),
                    ct::Literal::Symbol(s) => MLiteral::Symbol(self.sym(*s)),
                    ct::Literal::Bool(b) => MLiteral::Bool(*b),
                },
                location: opt_range(l.location),
            }),
        }
    }

    fn fn_param(&mut self, p: &ct::FunctionParam, depth: u64) -> MFunctionParam {
        MFunctionParam {
            ty: Box::new(self.ty(&p.ty, depth)),
            name: p.name.map(|s| self.sym(s)),
            location: p.location.map(|l| MFunctionParamLoc {
                range: range(l.range),
                name_range: opt_range(l.name_range),
            }),
        }
    }

    pub(crate) fn function(&mut self, f: &ct::Function, depth: u64) -> MFunction {
        match f {
            ct::Function::Typed(t) => MFunction::Typed(MFunctionType {
                required_positionals: t
                    .required_positionals
                    .iter()
                    .map(|p| self.fn_param(p, depth))
                    .collect(),
                optional_positionals: t
                    .optional_positionals
                    .iter()
                    .map(|p| self.fn_param(p, depth))
                    .collect(),
                rest_positionals: t
                    .rest_positionals
                    .as_ref()
                    .map(|p| Box::new(self.fn_param(p, depth))),
                trailing_positionals: t
                    .trailing_positionals
                    .iter()
                    .map(|p| self.fn_param(p, depth))
                    .collect(),
                required_keywords: t
                    .required_keywords
                    .iter()
                    .map(|k| MKeywordParam {
                        name: self.sym(k.name),
                        param: self.fn_param(&k.param, depth),
                    })
                    .collect(),
                optional_keywords: t
                    .optional_keywords
                    .iter()
                    .map(|k| MKeywordParam {
                        name: self.sym(k.name),
                        param: self.fn_param(&k.param, depth),
                    })
                    .collect(),
                rest_keywords: t
                    .rest_keywords
                    .as_ref()
                    .map(|p| Box::new(self.fn_param(p, depth))),
                return_type: Box::new(self.ty(&t.return_type, depth)),
            }),
            ct::Function::Untyped(u) => MFunction::Untyped(MUntypedFunctionType {
                return_type: Box::new(self.ty(&u.return_type, depth)),
            }),
        }
    }

    fn block(&mut self, b: &ct::BlockType, depth: u64) -> MBlockType {
        MBlockType {
            required: b.required,
            function: self.function(&b.function, depth),
            self_type: b.self_type.as_ref().map(|t| Box::new(self.ty(t, depth))),
        }
    }

    fn type_param(&mut self, p: &TypeParam) -> MTypeParam {
        MTypeParam {
            name: self.sym(p.name),
            variance: match p.variance {
                crate::ast::Variance::Invariant => 0,
                crate::ast::Variance::Covariant => 1,
                crate::ast::Variance::Contravariant => 2,
            },
            upper_bound: p.upper_bound.as_ref().map(|t| self.ty(t, 0)),
            lower_bound: p.lower_bound.as_ref().map(|t| self.ty(t, 0)),
            default_type: p.default_type.as_ref().map(|t| self.ty(t, 0)),
            unchecked: p.unchecked,
            location: p.location.map(|l| MTypeParamLoc {
                range: range(l.range),
                name_range: range(l.name_range),
                variance_range: opt_range(l.variance_range),
                unchecked_range: opt_range(l.unchecked_range),
                upper_bound_range: opt_range(l.upper_bound_range),
                lower_bound_range: opt_range(l.lower_bound_range),
                default_range: opt_range(l.default_range),
            }),
        }
    }

    pub(crate) fn method_type(&mut self, m: &MethodType) -> MMethodType {
        self.stats.method_defs += 1;
        MMethodType {
            type_params: m.type_params.iter().map(|p| self.type_param(p)).collect(),
            function: self.function(&m.function, 0),
            block: m.block.as_ref().map(|b| self.block(b, 0)),
            location: m.location.map(|l| MMethodTypeLoc {
                range: range(l.range),
                type_range: range(l.type_range),
                type_params_range: opt_range(l.type_params_range),
            }),
        }
    }

    pub(crate) fn annotations(
        &mut self,
        a: &[crate::ast::annotation::Annotation],
    ) -> Vec<MAnnotation> {
        self.stats.annotations += a.len() as u64;
        a.iter()
            .map(|a| MAnnotation {
                string: self.sym(a.string),
                location: opt_range(a.location),
            })
            .collect()
    }

    fn comment(&mut self, c: &Option<crate::ast::comment::Comment>) -> Option<MComment> {
        c.as_ref().map(|c| {
            self.stats.comments += 1;
            MComment {
                string: self.sym(c.string),
                location: opt_range(c.location),
            }
        })
    }

    fn mixin(
        &mut self,
        name: TypeName,
        args: &[ct::Type],
        annotations: &[crate::ast::annotation::Annotation],
        location: &Option<cl::MixinMemberLocation>,
        source_file: Option<Name>,
        comment: &Option<crate::ast::comment::Comment>,
    ) -> MMixinMember {
        MMixinMember {
            name: self.tn(name),
            args: args.iter().map(|a| self.ty(a, 0)).collect(),
            annotations: self.annotations(annotations),
            location: location.map(|l| MMixinMemberLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                args_range: opt_range(l.args_range),
            }),
            source_file: self.opt_name(source_file),
            comment: self.comment(comment),
        }
    }

    fn attr(
        &mut self,
        name: Symbol,
        ty: &ct::Type,
        ivar_name: &cm::IvarName,
        kind: cm::AttributeKind,
        annotations: &[crate::ast::annotation::Annotation],
        location: &Option<cl::AttributeMemberLocation>,
        source_file: Option<Name>,
        comment: &Option<crate::ast::comment::Comment>,
        vis: Option<crate::ast::Visibility>,
    ) -> MAttrMember {
        MAttrMember {
            name: self.sym(name),
            ty: self.ty(ty, 0),
            ivar_name: match ivar_name {
                cm::IvarName::Unspecified => MIvarName::Unspecified,
                cm::IvarName::Empty => MIvarName::Empty,
                cm::IvarName::Name(s) => MIvarName::Name(self.sym(*s)),
            },
            kind: match kind {
                cm::AttributeKind::Instance => 0,
                cm::AttributeKind::Singleton => 1,
            },
            annotations: self.annotations(annotations),
            location: location.map(|l| MAttributeMemberLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                colon_range: range(l.colon_range),
                kind_range: opt_range(l.kind_range),
                ivar_range: opt_range(l.ivar_range),
                ivar_name_range: opt_range(l.ivar_name_range),
                visibility_range: opt_range(l.visibility_range),
            }),
            source_file: self.opt_name(source_file),
            comment: self.comment(comment),
            visibility: vis.map(ast_visibility),
        }
    }

    fn var_member(
        &mut self,
        name: Symbol,
        ty: &ct::Type,
        location: &Option<cl::VariableMemberLocation>,
        source_file: Option<Name>,
        comment: &Option<crate::ast::comment::Comment>,
    ) -> MVarMember {
        MVarMember {
            name: self.sym(name),
            ty: self.ty(ty, 0),
            location: location.map(|l| MVariableMemberLoc {
                range: range(l.range),
                name_range: range(l.name_range),
                colon_range: range(l.colon_range),
                kind_range: opt_range(l.kind_range),
            }),
            source_file: self.opt_name(source_file),
            comment: self.comment(comment),
        }
    }

    fn member(&mut self, m: &cd::Member) -> MMember {
        self.stats.members += 1;
        match m {
            cd::Member::MethodDefinition(d) => MMember::MethodDefinition(MMethodDefinitionMember {
                name: self.sym(d.name),
                kind: method_kind(d.kind),
                overloads: {
                    self.stats.overloads += d.overloads.len() as u64;
                    d.overloads
                        .iter()
                        .map(|o| MMethodDefinitionOverload {
                            method_type: self.method_type(&o.method_type),
                            annotations: self.annotations(&o.annotations),
                        })
                        .collect()
                },
                annotations: self.annotations(&d.annotations),
                overloading: d.overloading,
                visibility: d.visibility.map(visibility),
                location: d.location.map(|l| MMethodDefLoc {
                    range: range(l.range),
                    keyword_range: range(l.keyword_range),
                    name_range: range(l.name_range),
                    kind_range: opt_range(l.kind_range),
                    overloading_range: opt_range(l.overloading_range),
                    visibility_range: opt_range(l.visibility_range),
                }),
                source_file: self.opt_name(d.source_file),
                comment: self.comment(&d.comment),
            }),
            cd::Member::Include(d) => MMember::Include(self.mixin(
                d.name,
                &d.args,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
            )),
            cd::Member::Extend(d) => MMember::Extend(self.mixin(
                d.name,
                &d.args,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
            )),
            cd::Member::Prepend(d) => MMember::Prepend(self.mixin(
                d.name,
                &d.args,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
            )),
            cd::Member::AttrReader(d) => MMember::AttrReader(self.attr(
                d.name,
                &d.ty,
                &d.ivar_name,
                d.kind,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
                d.visibility,
            )),
            cd::Member::AttrWriter(d) => MMember::AttrWriter(self.attr(
                d.name,
                &d.ty,
                &d.ivar_name,
                d.kind,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
                d.visibility,
            )),
            cd::Member::AttrAccessor(d) => MMember::AttrAccessor(self.attr(
                d.name,
                &d.ty,
                &d.ivar_name,
                d.kind,
                &d.annotations,
                &d.location,
                d.source_file,
                &d.comment,
                d.visibility,
            )),
            cd::Member::Public(d) => MMember::Public(MVisibilityMarker {
                location: opt_range(d.location),
            }),
            cd::Member::Private(d) => MMember::Private(MVisibilityMarker {
                location: opt_range(d.location),
            }),
            cd::Member::Alias(d) => MMember::Alias(MAliasMember {
                new_name: self.sym(d.new_name),
                old_name: self.sym(d.old_name),
                kind: match d.kind {
                    cm::AliasKind::Instance => 0,
                    cm::AliasKind::Singleton => 1,
                },
                annotations: self.annotations(&d.annotations),
                location: d.location.map(|l| MAliasMemberLoc {
                    range: range(l.range),
                    keyword_range: range(l.keyword_range),
                    new_name_range: range(l.new_name_range),
                    old_name_range: range(l.old_name_range),
                    new_kind_range: opt_range(l.new_kind_range),
                    old_kind_range: opt_range(l.old_kind_range),
                }),
                source_file: self.opt_name(d.source_file),
                comment: self.comment(&d.comment),
            }),
            cd::Member::InstanceVariable(d) => MMember::InstanceVariable(self.var_member(
                d.name,
                &d.ty,
                &d.location,
                d.source_file,
                &d.comment,
            )),
            cd::Member::ClassInstanceVariable(d) => MMember::ClassInstanceVariable(
                self.var_member(d.name, &d.ty, &d.location, d.source_file, &d.comment),
            ),
            cd::Member::ClassVariable(d) => MMember::ClassVariable(self.var_member(
                d.name,
                &d.ty,
                &d.location,
                d.source_file,
                &d.comment,
            )),
        }
    }

    fn decl_ref(&mut self, d: &cd::Declaration) -> MDeclRef {
        self.stats.nested_decl_refs += 1;
        match d {
            cd::Declaration::Class(x) => MDeclRef {
                kind: 0,
                id: self.tn(x.name),
            },
            cd::Declaration::Module(x) => MDeclRef {
                kind: 1,
                id: self.tn(x.name),
            },
            cd::Declaration::Interface(x) => MDeclRef {
                kind: 2,
                id: self.tn(x.name),
            },
            cd::Declaration::Constant(x) => MDeclRef {
                kind: 3,
                id: self.tn(x.name),
            },
            cd::Declaration::Global(x) => MDeclRef {
                kind: 4,
                id: self.sym(x.name),
            },
            cd::Declaration::TypeAlias(x) => MDeclRef {
                kind: 5,
                id: self.tn(x.name),
            },
            cd::Declaration::ClassAlias(x) => MDeclRef {
                kind: 6,
                id: self.tn(x.new_name),
            },
            cd::Declaration::ModuleAlias(x) => MDeclRef {
                kind: 7,
                id: self.tn(x.new_name),
            },
        }
    }

    fn class_super(&mut self, s: &cd::ClassSuper) -> MClassSuper {
        MClassSuper {
            name: self.tn(s.name),
            args: s.args.iter().map(|a| self.ty(a, 0)).collect(),
            location: s
                .location
                .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            source_file: self.opt_name(s.source_file),
        }
    }

    fn module_self(&mut self, s: &cd::ModuleSelf) -> MClassSuper {
        MClassSuper {
            name: self.tn(s.name),
            args: s.args.iter().map(|a| self.ty(a, 0)).collect(),
            location: s
                .location
                .map(|l| name_args_loc(l.range, l.name_range, l.args_range)),
            source_file: self.opt_name(s.source_file),
        }
    }

    pub fn class_decl(&mut self, d: &cd::ClassDeclaration) -> MClassDeclaration {
        self.stats.decls += 1;
        MClassDeclaration {
            name: self.tn(d.name),
            type_params: d.type_params.iter().map(|p| self.type_param(p)).collect(),
            super_class: d.super_class.as_ref().map(|s| self.class_super(s)),
            members: d
                .members
                .iter()
                .map(|m| match m {
                    cd::ClassMember::Member(m) => MBodyMember::Member(self.member(m)),
                    cd::ClassMember::Declaration(d) => MBodyMember::Declaration(self.decl_ref(d)),
                })
                .collect(),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MClassDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                end_range: range(l.end_range),
                type_params_range: opt_range(l.type_params_range),
                lt_range: opt_range(l.lt_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }

    pub fn module_decl(&mut self, d: &cd::ModuleDeclaration) -> MModuleDeclaration {
        self.stats.decls += 1;
        MModuleDeclaration {
            name: self.tn(d.name),
            type_params: d.type_params.iter().map(|p| self.type_param(p)).collect(),
            self_types: d.self_types.iter().map(|s| self.module_self(s)).collect(),
            members: d
                .members
                .iter()
                .map(|m| match m {
                    cd::ModuleMember::Member(m) => MBodyMember::Member(self.member(m)),
                    cd::ModuleMember::Declaration(d) => MBodyMember::Declaration(self.decl_ref(d)),
                })
                .collect(),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MModuleDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                end_range: range(l.end_range),
                type_params_range: opt_range(l.type_params_range),
                colon_range: opt_range(l.colon_range),
                self_types_range: opt_range(l.self_types_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }

    pub fn interface_decl(&mut self, d: &cd::InterfaceDeclaration) -> MInterfaceDeclaration {
        self.stats.decls += 1;
        MInterfaceDeclaration {
            name: self.tn(d.name),
            type_params: d.type_params.iter().map(|p| self.type_param(p)).collect(),
            members: d.members.iter().map(|m| self.member(m)).collect(),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MInterfaceDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                end_range: range(l.end_range),
                type_params_range: opt_range(l.type_params_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }

    pub fn type_alias_decl(&mut self, d: &cd::TypeAliasDeclaration) -> MTypeAliasDeclaration {
        self.stats.decls += 1;
        MTypeAliasDeclaration {
            name: self.tn(d.name),
            type_params: d.type_params.iter().map(|p| self.type_param(p)).collect(),
            ty: self.ty(&d.ty, 0),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MTypeAliasDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                name_range: range(l.name_range),
                eq_range: range(l.eq_range),
                type_params_range: opt_range(l.type_params_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }

    pub fn constant_decl(&mut self, d: &cd::ConstantDeclaration) -> MConstantDeclaration {
        self.stats.decls += 1;
        MConstantDeclaration {
            name: self.tn(d.name),
            ty: self.ty(&d.ty, 0),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MConstGlobalLoc {
                range: range(l.range),
                name_range: range(l.name_range),
                colon_range: range(l.colon_range),
            }),
            comment: self.comment(&d.comment),
        }
    }

    pub fn global_decl(&mut self, d: &cd::GlobalDeclaration) -> MGlobalDeclaration {
        self.stats.decls += 1;
        MGlobalDeclaration {
            name: self.sym(d.name),
            ty: self.ty(&d.ty, 0),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MConstGlobalLoc {
                range: range(l.range),
                name_range: range(l.name_range),
                colon_range: range(l.colon_range),
            }),
            comment: self.comment(&d.comment),
        }
    }

    pub fn class_alias_decl(&mut self, d: &cd::ClassAliasDeclaration) -> MClassAliasDeclaration {
        self.stats.decls += 1;
        MClassAliasDeclaration {
            new_name: self.tn(d.new_name),
            old_name: self.tn(d.old_name),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MAliasDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                new_name_range: range(l.new_name_range),
                eq_range: range(l.eq_range),
                old_name_range: range(l.old_name_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }

    pub fn module_alias_decl(&mut self, d: &cd::ModuleAliasDeclaration) -> MClassAliasDeclaration {
        self.stats.decls += 1;
        MClassAliasDeclaration {
            new_name: self.tn(d.new_name),
            old_name: self.tn(d.old_name),
            annotations: self.annotations(&d.annotations),
            location: d.location.map(|l| MAliasDeclLoc {
                range: range(l.range),
                keyword_range: range(l.keyword_range),
                new_name_range: range(l.new_name_range),
                eq_range: range(l.eq_range),
                old_name_range: range(l.old_name_range),
            }),
            source_file: self.opt_name(d.source_file),
            comment: self.comment(&d.comment),
        }
    }
}

fn expect_signature_class(d: &ClassDeclaration) -> &cd::ClassDeclaration {
    match d {
        ClassDeclaration::Signature(a) => a,
        ClassDeclaration::Ruby(_) => panic!("unexpected Ruby class decl in gem environment"),
    }
}

fn expect_signature_module(d: &ModuleDeclaration) -> &cd::ModuleDeclaration {
    match d {
        ModuleDeclaration::Signature(a) => a,
        ModuleDeclaration::Ruby(_) => panic!("unexpected Ruby module decl in gem environment"),
    }
}

/// Convert the frozen environment into the full mirror snapshot.
///
/// Assumes a G-layer (gem) environment: `expect_signature_class`/`_module`
/// panic on `ClassDeclaration::Ruby(_)` / `ModuleDeclaration::Ruby(_)`
/// (inline decls). A-layer support is scope 2 of ADR-0028.
pub fn convert(env: &Environment) -> (SnapshotA, Stats) {
    let mut cx = Cx::new(env.names());

    let mut class_decls: FxMap<MClassOrModule> = FxMap::default();
    for (name, com) in env.class_decls() {
        let id = cx.tn(*name);
        let m = match com {
            ClassOrModule::Class(e) => {
                let primary = e
                    .context_decls()
                    .iter()
                    .position(|(_, _, d)| match (d, e.primary_decl()) {
                        (ClassDeclaration::Signature(a), ClassDeclaration::Signature(b)) => {
                            Arc::ptr_eq(a, b)
                        }
                        _ => false,
                    })
                    .unwrap_or(0) as u32;
                MClassOrModule::Class(MClassEntry {
                    name: id,
                    context_decls: e
                        .context_decls()
                        .iter()
                        .map(|(_file, ctx, d)| {
                            (
                                cx.context(ctx),
                                Arc::new(cx.class_decl(expect_signature_class(d))),
                            )
                        })
                        .collect(),
                    primary,
                })
            }
            ClassOrModule::Module(e) => {
                let primary = e
                    .context_decls()
                    .iter()
                    .position(|(_, _, d)| match (d, e.primary_decl()) {
                        (ModuleDeclaration::Signature(a), ModuleDeclaration::Signature(b)) => {
                            Arc::ptr_eq(a, b)
                        }
                        _ => false,
                    })
                    .unwrap_or(0) as u32;
                MClassOrModule::Module(MModuleEntry {
                    name: id,
                    context_decls: e
                        .context_decls()
                        .iter()
                        .map(|(_file, ctx, d)| {
                            (
                                cx.context(ctx),
                                Arc::new(cx.module_decl(expect_signature_module(d))),
                            )
                        })
                        .collect(),
                    primary,
                })
            }
        };
        class_decls.insert(id, m);
    }

    let mut interface_decls: FxMap<MInterfaceEntry> = FxMap::default();
    for (name, e) in env.interface_decls() {
        let id = cx.tn(*name);
        interface_decls.insert(
            id,
            MInterfaceEntry {
                name: id,
                context: cx.context(e.context()),
                decl: Arc::new(cx.interface_decl(e.decl())),
            },
        );
    }

    let mut class_alias_decls: FxMap<MClassOrModuleAliasEntry> = FxMap::default();
    for (name, e) in env.class_alias_decls() {
        let id = cx.tn(*name);
        let m = match e {
            ClassOrModuleAliasEntry::Class(a) => MClassOrModuleAliasEntry {
                kind: 0,
                name: id,
                context: cx.context(a.context()),
                decl: Arc::new(match a.decl() {
                    ClassAliasDeclaration::Signature(d) => cx.class_alias_decl(d),
                    ClassAliasDeclaration::Ruby(_) => {
                        panic!("unexpected Ruby class alias in gem environment")
                    }
                }),
            },
            ClassOrModuleAliasEntry::Module(a) => MClassOrModuleAliasEntry {
                kind: 1,
                name: id,
                context: cx.context(a.context()),
                decl: Arc::new(match a.decl() {
                    ModuleAliasDeclaration::Signature(d) => cx.module_alias_decl(d),
                    ModuleAliasDeclaration::Ruby(_) => {
                        panic!("unexpected Ruby module alias in gem environment")
                    }
                }),
            },
        };
        class_alias_decls.insert(id, m);
    }

    let mut type_alias_decls: FxMap<MSingleEntry<MTypeAliasDeclaration>> = FxMap::default();
    for (name, e) in env.type_alias_decls() {
        let id = cx.tn(*name);
        type_alias_decls.insert(
            id,
            MSingleEntry {
                name: id,
                context: cx.context(&e.context),
                decl: Arc::new(cx.type_alias_decl(&e.decl)),
            },
        );
    }

    let mut constant_decls: FxMap<MSingleEntry<MConstantDeclaration>> = FxMap::default();
    for (name, e) in env.constant_decls() {
        let id = cx.tn(*name);
        constant_decls.insert(
            id,
            MSingleEntry {
                name: id,
                context: cx.context(&e.context),
                decl: Arc::new(cx.constant_decl(&e.decl)),
            },
        );
    }

    let mut global_decls: FxMap<MGlobalEntry> = FxMap::default();
    for (name, e) in env.global_decls() {
        let id = cx.sym(*name);
        global_decls.insert(
            id,
            MGlobalEntry {
                name: id,
                file: cx.opt_name(e.file.file()),
                context: cx.context(&e.context),
                decl: Arc::new(cx.global_decl(&e.decl)),
            },
        );
    }

    // Precomputed alias-chain folds: keys are the class_alias_decls keys.
    let mut normalized: FxMap<MNormResult> = FxMap::default();
    for name in env.class_alias_decls().keys() {
        let id = cx.tn(*name);
        let r = match env.normalize_module_name_result(name) {
            NormalizeModuleNameResult::Normalized(t) => MNormResult::Normalized(cx.tn(t)),
            NormalizeModuleNameResult::UnknownTarget { original, target } => {
                MNormResult::UnknownTarget {
                    original: cx.tn(original),
                    target: cx.tn(target),
                }
            }
            NormalizeModuleNameResult::Cycle { original } => MNormResult::Cycle {
                original: cx.tn(original),
            },
            NormalizeModuleNameResult::NotClassOrModule { original } => {
                MNormResult::NotClassOrModule {
                    original: cx.tn(original),
                }
            }
        };
        normalized.insert(id, r);
    }

    let sources: Vec<Option<(NameId, u32, u32)>> = env
        .sources()
        .iter()
        .map(|s| s.map(|l| (cx.name_id(l.file), l.start_byte, l.end_byte)))
        .collect();

    let snap = SnapshotA {
        names: cx.names_vec.clone(),
        symbols: cx.symbols.clone(),
        type_names: cx.type_names.clone(),
        class_decls,
        interface_decls,
        class_alias_decls,
        type_alias_decls,
        constant_decls,
        global_decls,
        normalized,
        sources,
    };
    (snap, cx.stats)
}

/// Frozen-env display name for a mirror TypeName id (via the mirror tables).
pub fn display_tn(snap: &SnapshotA, id: u64) -> String {
    let mut segs: Vec<u64> = Vec::new();
    let mut cur = id;
    let absolute = loop {
        let &(parent, seg, abs) = snap.type_names.get(&cur).expect("tn registered");
        if parent == 0 {
            break abs;
        }
        segs.push(seg);
        cur = parent;
    };
    segs.reverse();
    let mut s = String::new();
    if absolute {
        s.push_str("::");
    }
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            s.push_str("::");
        }
        s.push_str(&snap.symbols[seg]);
    }
    s
}
