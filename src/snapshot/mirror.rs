// Ported from ADR-0028 spike 1 verbatim; refactor forbidden by slice 1a scope.
#![allow(clippy::large_enum_variant)]

//! Serializable mirror of crema's frozen `Environment`.
//!
//! Fidelity policy:
//! - Symbol / TypeName are content-addressed u64 ids in crema; stored raw.
//! - `Name` (insertion-order id, file paths) is remapped to a dense u32 into `names`.
//! - Every AST struct mirrors its crema counterpart field-for-field so that
//!   node counts, allocation counts (Vec/Box/String/Arc) and byte sizes match.
//!   Structurally identical location structs are shared (same byte layout).
//! - `Context` (`Arc<[TypeName]>` in crema, shared across nested decls) is
//!   stored owned per entry (`Vec<u64>`); slight alloc overcount, noted in
//!   the report.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type Sym = u64;
pub type Tn = u64;
pub type NameId = u32;

pub type FxMap<V> = std::collections::HashMap<u64, V, rustc_hash::FxBuildHasher>;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MRange(pub u32, pub u32, pub u32, pub u32);

// ---- locations ----

/// Alias / ClassInstance / ClassSingleton / Interface / ClassSuper / ModuleSelf
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MNameArgsLoc {
    pub range: MRange,
    pub name_range: MRange,
    pub args_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MFunctionParamLoc {
    pub range: MRange,
    pub name_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MMethodTypeLoc {
    pub range: MRange,
    pub type_range: MRange,
    pub type_params_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MTypeParamLoc {
    pub range: MRange,
    pub name_range: MRange,
    pub variance_range: Option<MRange>,
    pub unchecked_range: Option<MRange>,
    pub upper_bound_range: Option<MRange>,
    pub lower_bound_range: Option<MRange>,
    pub default_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MClassDeclLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub end_range: MRange,
    pub type_params_range: Option<MRange>,
    pub lt_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MModuleDeclLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub end_range: MRange,
    pub type_params_range: Option<MRange>,
    pub colon_range: Option<MRange>,
    pub self_types_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MInterfaceDeclLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub end_range: MRange,
    pub type_params_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MTypeAliasDeclLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub eq_range: MRange,
    pub type_params_range: Option<MRange>,
}

/// Constant / Global declaration location.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MConstGlobalLoc {
    pub range: MRange,
    pub name_range: MRange,
    pub colon_range: MRange,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MAliasDeclLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub new_name_range: MRange,
    pub eq_range: MRange,
    pub old_name_range: MRange,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MMethodDefLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub kind_range: Option<MRange>,
    pub overloading_range: Option<MRange>,
    pub visibility_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MVariableMemberLoc {
    pub range: MRange,
    pub name_range: MRange,
    pub colon_range: MRange,
    pub kind_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MMixinMemberLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub args_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MAttributeMemberLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub name_range: MRange,
    pub colon_range: MRange,
    pub kind_range: Option<MRange>,
    pub ivar_range: Option<MRange>,
    pub ivar_name_range: Option<MRange>,
    pub visibility_range: Option<MRange>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MAliasMemberLoc {
    pub range: MRange,
    pub keyword_range: MRange,
    pub new_name_range: MRange,
    pub old_name_range: MRange,
    pub new_kind_range: Option<MRange>,
    pub old_kind_range: Option<MRange>,
}

// ---- types ----

#[derive(Serialize, Deserialize)]
pub enum MType {
    Base(MBaseType),
    Variable(MVariableType),
    ClassSingleton(MNamedType),
    Interface(MNamedType),
    ClassInstance(MNamedType),
    Alias(MNamedType),
    Tuple(MTypes),
    Record(MRecordType),
    Optional(MOptionalType),
    Union(MTypes),
    Intersection(MTypes),
    Proc(Box<MProcType>),
    Literal(MLiteralType),
}

#[derive(Serialize, Deserialize)]
pub struct MBaseType {
    pub kind: MBaseTypeKind,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub enum MBaseTypeKind {
    Bool,
    Void,
    Any { todo: bool },
    Nil,
    Top,
    Bottom,
    SelfType,
    Instance,
    Class,
}

#[derive(Serialize, Deserialize)]
pub struct MVariableType {
    pub name: Sym,
    pub location: Option<MRange>,
}

/// ClassSingleton / Interface / ClassInstance / Alias type node.
#[derive(Serialize, Deserialize)]
pub struct MNamedType {
    pub name: Tn,
    pub args: Vec<MType>,
    pub location: Option<MNameArgsLoc>,
}

/// Tuple / Union / Intersection.
#[derive(Serialize, Deserialize)]
pub struct MTypes {
    pub types: Vec<MType>,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub struct MRecordType {
    pub fields: Vec<MRecordField>,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub enum MRecordKey {
    Symbol(Sym),
    String(String),
    Integer(String),
    Bool(bool),
}

#[derive(Serialize, Deserialize)]
pub struct MRecordField {
    pub key: MRecordKey,
    pub ty: MType,
    pub required: bool,
}

#[derive(Serialize, Deserialize)]
pub struct MOptionalType {
    pub ty: Box<MType>,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub struct MProcType {
    pub function: MFunction,
    pub block: Option<MBlockType>,
    pub self_type: Option<Box<MType>>,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub struct MLiteralType {
    pub literal: MLiteral,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub enum MLiteral {
    Integer(String),
    String(String),
    Symbol(Sym),
    Bool(bool),
}

#[derive(Serialize, Deserialize)]
pub enum MFunction {
    Typed(MFunctionType),
    Untyped(MUntypedFunctionType),
}

#[derive(Serialize, Deserialize)]
pub struct MFunctionType {
    pub required_positionals: Vec<MFunctionParam>,
    pub optional_positionals: Vec<MFunctionParam>,
    pub rest_positionals: Option<Box<MFunctionParam>>,
    pub trailing_positionals: Vec<MFunctionParam>,
    pub required_keywords: Vec<MKeywordParam>,
    pub optional_keywords: Vec<MKeywordParam>,
    pub rest_keywords: Option<Box<MFunctionParam>>,
    pub return_type: Box<MType>,
}

#[derive(Serialize, Deserialize)]
pub struct MFunctionParam {
    pub ty: Box<MType>,
    pub name: Option<Sym>,
    pub location: Option<MFunctionParamLoc>,
}

#[derive(Serialize, Deserialize)]
pub struct MKeywordParam {
    pub name: Sym,
    pub param: MFunctionParam,
}

#[derive(Serialize, Deserialize)]
pub struct MUntypedFunctionType {
    pub return_type: Box<MType>,
}

#[derive(Serialize, Deserialize)]
pub struct MBlockType {
    pub required: bool,
    pub function: MFunction,
    pub self_type: Option<Box<MType>>,
}

// ---- type params / method types / annotations ----

#[derive(Serialize, Deserialize)]
pub struct MTypeParam {
    pub name: Sym,
    pub variance: u8,
    pub upper_bound: Option<MType>,
    pub lower_bound: Option<MType>,
    pub default_type: Option<MType>,
    pub unchecked: bool,
    pub location: Option<MTypeParamLoc>,
}

#[derive(Serialize, Deserialize)]
pub struct MMethodType {
    pub type_params: Vec<MTypeParam>,
    pub function: MFunction,
    pub block: Option<MBlockType>,
    pub location: Option<MMethodTypeLoc>,
}

#[derive(Serialize, Deserialize)]
pub struct MAnnotation {
    pub string: Sym,
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub struct MComment {
    pub string: Sym,
    pub location: Option<MRange>,
}

// ---- members ----

#[derive(Serialize, Deserialize)]
pub enum MMember {
    MethodDefinition(MMethodDefinitionMember),
    Include(MMixinMember),
    Extend(MMixinMember),
    Prepend(MMixinMember),
    AttrReader(MAttrMember),
    AttrWriter(MAttrMember),
    AttrAccessor(MAttrMember),
    Public(MVisibilityMarker),
    Private(MVisibilityMarker),
    Alias(MAliasMember),
    InstanceVariable(MVarMember),
    ClassInstanceVariable(MVarMember),
    ClassVariable(MVarMember),
}

#[derive(Serialize, Deserialize)]
pub struct MMethodDefinitionOverload {
    pub method_type: MMethodType,
    pub annotations: Vec<MAnnotation>,
}

#[derive(Serialize, Deserialize)]
pub struct MMethodDefinitionMember {
    pub name: Sym,
    pub kind: u8,
    pub overloads: Vec<MMethodDefinitionOverload>,
    pub annotations: Vec<MAnnotation>,
    pub overloading: bool,
    pub visibility: Option<u8>,
    pub location: Option<MMethodDefLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

/// Include / Extend / Prepend.
#[derive(Serialize, Deserialize)]
pub struct MMixinMember {
    pub name: Tn,
    pub args: Vec<MType>,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MMixinMemberLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

#[derive(Serialize, Deserialize)]
pub enum MIvarName {
    Unspecified,
    Empty,
    Name(Sym),
}

/// AttrReader / AttrWriter / AttrAccessor.
#[derive(Serialize, Deserialize)]
pub struct MAttrMember {
    pub name: Sym,
    pub ty: MType,
    pub ivar_name: MIvarName,
    pub kind: u8,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MAttributeMemberLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
    pub visibility: Option<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct MVisibilityMarker {
    pub location: Option<MRange>,
}

#[derive(Serialize, Deserialize)]
pub struct MAliasMember {
    pub new_name: Sym,
    pub old_name: Sym,
    pub kind: u8,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MAliasMemberLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

/// InstanceVariable / ClassInstanceVariable / ClassVariable.
#[derive(Serialize, Deserialize)]
pub struct MVarMember {
    pub name: Sym,
    pub ty: MType,
    pub location: Option<MVariableMemberLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

// ---- declarations ----

#[derive(Serialize, Deserialize)]
pub struct MClassDeclaration {
    pub name: Tn,
    pub type_params: Vec<MTypeParam>,
    pub super_class: Option<MClassSuper>,
    pub members: Vec<MBodyMember>,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MClassDeclLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

/// ClassSuper / ModuleSelf.
#[derive(Serialize, Deserialize)]
pub struct MClassSuper {
    pub name: Tn,
    pub args: Vec<MType>,
    pub location: Option<MNameArgsLoc>,
    pub source_file: Option<NameId>,
}

#[derive(Serialize, Deserialize)]
pub struct MModuleDeclaration {
    pub name: Tn,
    pub type_params: Vec<MTypeParam>,
    pub self_types: Vec<MClassSuper>,
    pub members: Vec<MBodyMember>,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MModuleDeclLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

#[derive(Serialize, Deserialize)]
pub struct MInterfaceDeclaration {
    pub name: Tn,
    pub type_params: Vec<MTypeParam>,
    pub members: Vec<MMember>,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MInterfaceDeclLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

#[derive(Serialize, Deserialize)]
pub struct MTypeAliasDeclaration {
    pub name: Tn,
    pub type_params: Vec<MTypeParam>,
    pub ty: MType,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MTypeAliasDeclLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

#[derive(Serialize, Deserialize)]
pub struct MConstantDeclaration {
    pub name: Tn,
    pub ty: MType,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MConstGlobalLoc>,
    pub comment: Option<MComment>,
}

#[derive(Serialize, Deserialize)]
pub struct MGlobalDeclaration {
    pub name: Sym,
    pub ty: MType,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MConstGlobalLoc>,
    pub comment: Option<MComment>,
}

/// ClassAlias / ModuleAlias declaration.
#[derive(Serialize, Deserialize)]
pub struct MClassAliasDeclaration {
    pub new_name: Tn,
    pub old_name: Tn,
    pub annotations: Vec<MAnnotation>,
    pub location: Option<MAliasDeclLoc>,
    pub source_file: Option<NameId>,
    pub comment: Option<MComment>,
}

/// ClassMember / ModuleMember (member-or-nested-declaration).
///
/// crema shares one `Arc` between a nested declaration node in the parent's
/// members and that declaration's own env entry. A snapshot serializing both
/// by value would double-store every nested decl, so the mirror stores the
/// member-side occurrence as a reference to the entry (kind + id) — the
/// layout a real snapshot would use.
#[derive(Serialize, Deserialize)]
pub enum MBodyMember {
    Member(MMember),
    Declaration(MDeclRef),
}

/// kind: 0=Class 1=Module 2=Interface 3=Constant 4=Global 5=TypeAlias
/// 6=ClassAlias 7=ModuleAlias. id is the entry key (TypeName id; Symbol id
/// for globals).
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MDeclRef {
    pub kind: u8,
    pub id: u64,
}

// ---- frozen entries ----

pub type MContext = Vec<Tn>;

#[derive(Serialize, Deserialize)]
pub struct MClassEntry {
    pub name: Tn,
    pub context_decls: Vec<(MContext, Arc<MClassDeclaration>)>,
    pub primary: u32,
}

#[derive(Serialize, Deserialize)]
pub struct MModuleEntry {
    pub name: Tn,
    pub context_decls: Vec<(MContext, Arc<MModuleDeclaration>)>,
    pub primary: u32,
}

#[derive(Serialize, Deserialize)]
pub enum MClassOrModule {
    Class(MClassEntry),
    Module(MModuleEntry),
}

#[derive(Serialize, Deserialize)]
pub struct MInterfaceEntry {
    pub name: Tn,
    pub context: MContext,
    pub decl: Arc<MInterfaceDeclaration>,
}

/// kind: 0 = class alias, 1 = module alias.
#[derive(Serialize, Deserialize)]
pub struct MClassOrModuleAliasEntry {
    pub kind: u8,
    pub name: Tn,
    pub context: MContext,
    pub decl: Arc<MClassAliasDeclaration>,
}

#[derive(Serialize, Deserialize)]
pub struct MSingleEntry<D> {
    pub name: Tn,
    pub context: MContext,
    pub decl: Arc<D>,
}

#[derive(Serialize, Deserialize)]
pub struct MGlobalEntry {
    pub name: Sym,
    pub file: Option<NameId>,
    pub context: MContext,
    pub decl: Arc<MGlobalDeclaration>,
}

#[derive(Serialize, Deserialize)]
pub enum MNormResult {
    Normalized(Tn),
    UnknownTarget { original: Tn, target: Tn },
    Cycle { original: Tn },
    NotClassOrModule { original: Tn },
}

/// Per-entry payload for the hybrid (B) entries blob.
#[derive(Serialize, Deserialize)]
pub enum MEntry {
    ClassOrModule(MClassOrModule),
    Interface(MInterfaceEntry),
    ClassAlias(MClassOrModuleAliasEntry),
    TypeAlias(MSingleEntry<MTypeAliasDeclaration>),
    Constant(MSingleEntry<MConstantDeclaration>),
    Global(MGlobalEntry),
}

/// (parent, segment, absolute) — parent/segment are 0 for roots.
pub type MTnEntry = (u64, u64, bool);

/// Full-blob snapshot for layout A (bincode all, hashmaps rebuilt on decode).
#[derive(Serialize, Deserialize)]
pub struct SnapshotA {
    pub names: Vec<String>,
    pub symbols: FxMap<Box<str>>,
    pub type_names: FxMap<MTnEntry>,
    pub class_decls: FxMap<MClassOrModule>,
    pub interface_decls: FxMap<MInterfaceEntry>,
    pub class_alias_decls: FxMap<MClassOrModuleAliasEntry>,
    pub type_alias_decls: FxMap<MSingleEntry<MTypeAliasDeclaration>>,
    pub constant_decls: FxMap<MSingleEntry<MConstantDeclaration>>,
    pub global_decls: FxMap<MGlobalEntry>,
    pub normalized: FxMap<MNormResult>,
    pub sources: Vec<Option<(NameId, u32, u32)>>,
}

// ---- baked G-scan diagnostics (ADR-0028 slice 2b) ----

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedLoc {
    pub file: std::path::PathBuf,
    pub range: MRange,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedAliasCycle {
    pub type_name: String,
    pub alias_names: Vec<String>,
    pub primary_location: Option<MBakedLoc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedMethodGroup {
    pub owner: Tn,
    /// `(method_name, duplicate_location, original_location)` — mirrors
    /// [`crate::definition_builder::MethodDups`]. The redefining site
    /// anchors the diagnostic; the original site backs the surfaced
    /// `duplicate_source` field.
    pub dups: Vec<(String, Option<MBakedLoc>, Option<MBakedLoc>)>,
    pub cycles: Vec<MBakedAliasCycle>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedVariableDup {
    /// 0 = Instance, 1 = ClassInstance (`VariableDuplicationKind`).
    pub kind: u8,
    pub variable_name: String,
    pub location: Option<MBakedLoc>,
    /// `(file, start_byte, end_byte)` of a `RubyLocation`. Always `None`
    /// for the G layer today (gem input is `.rbs` only); carried so the
    /// mirror stays field-for-field with `VariableDuplication`.
    pub ruby_source_location: Option<(String, u32, u32)>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedVariableGroup {
    pub owner: Tn,
    pub dups: Vec<MBakedVariableDup>,
}

/// One collapsed `RecursiveAncestor` finding from the cold G-only walk
/// (ADR-0028 slice 2b-2). `participants` keys the warm-side exclusion.
#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedAncestorCycle {
    pub participants: Vec<Tn>,
    pub type_name: String,
    pub chain: Vec<String>,
    pub primary_source: Option<MBakedLoc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedArityViolation {
    /// 0 = superclass, 1 = include, 2 = extend, 3 = prepend.
    pub kind: u8,
    pub target: String,
    pub class: String,
    pub expected: String,
    pub got: u64,
    pub location: Option<MBakedLoc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedArityGroup {
    pub owner: Tn,
    pub violations: Vec<MBakedArityViolation>,
}

/// G-only artifacts baked at cold time (`encode_g_snapshot`) so warm
/// runs consume them instead of re-scanning the G layer. Two shapes
/// coexist here:
///
/// - **diagnostic groups** (`classes`, `interfaces`, `variables`,
///   `ancestor_cycles`, `arity`): preserve the cold walk's emission
///   order so the warm splice reproduces it; owners (or cycle
///   participants) the A layer redeclares are skipped at splice time
///   (the merged A-map entry covers them).
/// - **super-edge graph** ([`Self::super_edges`], a two-field
///   [`MSuperEdges`]): infusion's AR model detection input. Cold
///   resolves each gem class's super against G-only names; hits go to
///   `super_edges.resolved` as `(class, resolved_super)`, misses (gem
///   supers referencing an A-only name — Rails-engine
///   `class GemModel < ApplicationRecord` idiom) go to
///   `super_edges.unresolved` with the declaration context stack needed
///   to retry against the live A ∪ G resolver. Both arms feed the same
///   set fixpoint on the warm side, so ordering carries no contract and
///   A-redeclared owners are NOT skipped (the union with live A edges
///   converges regardless).
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MBakedDiagnostics {
    pub classes: Vec<MBakedMethodGroup>,
    pub interfaces: Vec<MBakedMethodGroup>,
    pub variables: Vec<MBakedVariableGroup>,
    pub ancestor_cycles: Vec<MBakedAncestorCycle>,
    pub arity: Vec<MBakedArityGroup>,
    /// G-only class/module super-edges (ADR-0028 slice 2b-4). Two arms
    /// bundled so schema evolution touches one field instead of two.
    pub super_edges: MSuperEdges,
    /// G-construction stderr warnings (stale pin, missing library, load
    /// failure, ...) baked in verbatim so a warm hit can replay them
    /// (v7, `gem_dirs_cache_removal_warning_replay`).
    pub warnings: Vec<String>,
}

/// Mirror-side twin of `definition_builder::SuperEdges`. Both arms
/// share a single struct so `baked_to_mirror` / `baked_from_mirror`
/// stay symmetric and a new super-edge shape (skip counters, etc.)
/// evolves through one field rename instead of two parallel ones.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MSuperEdges {
    /// `(class, resolved_super)` edges, both endpoints
    /// content-addressed `TypeName`s present in the type-name table.
    pub resolved: Vec<(Tn, Tn)>,
    /// Gem super references cold's G-only resolver could not bind, kept
    /// for warm-side retry against the merged A ∪ G resolver
    /// (`high_infusion_ar_a_layer_super_bridging`).
    pub unresolved: Vec<MBakedUnresolvedSuperEdge>,
}

/// Serialized shape of a gem super reference cold could not resolve.
/// Warm retries [`Self::raw_super`] against a live A ∪ G
/// `TypeNameResolver`, with [`Self::context`] as the walker's stack —
/// mirroring the declaration context that produced the reference.
#[derive(Serialize, Deserialize, Clone)]
pub struct MBakedUnresolvedSuperEdge {
    pub class: Tn,
    pub raw_super: Tn,
    pub context: Vec<Tn>,
}
