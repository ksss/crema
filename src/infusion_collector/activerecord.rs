//! ActiveRecord DSL rules.

use std::path::Path;
use std::sync::Arc;

use ruby_prism::{CallNode, Node, Visit};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::MethodKind;
use crate::ast::method_type::MethodType;
use crate::ast::ruby::PrismByteRange;
use crate::ast::ruby::annotations::ColonMethodTypeAnnotation;
use crate::ast::ruby::declarations::{ClassDecl, Declaration, ModuleDecl};
use crate::ast::ruby::members::{
    DefMember, DefMemberOrigin, ExplicitAnnotation, ExtendMember, Member, MethodTypeAnnotation,
    MixinMember, TypeAnnotations,
};
use crate::ast::types::{
    BaseType, BaseTypeKind, BlockType, ClassInstanceType, Function, FunctionParam, FunctionType,
    KeywordParam, Literal, LiteralType, OptionalType, Type, UnionType, UntypedFunctionType,
};
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::draft::EnvironmentDraft;
use crate::infusion_collector::inflector::Inflector;
use crate::infusion_collector::pipeline::{
    EnumLiteral, InfusionCall, InfusionScopeParams, SourceUnit, push_def,
};
use crate::inline_parser::{prism_location_range, push_class_abs_path};
use crate::name::NameTable;
use crate::type_name::TypeName;

const PROVIDER: &str = "activerecord";


#[derive(Clone)]
pub(crate) struct ActiveRecordAssociation {
    pub(crate) owner: TypeName,
    pub(crate) kind: ActiveRecordAssociationKind,
    pub(crate) name: String,
    pub(crate) target_name: Option<ActiveRecordAssociationTargetName>,
    pub(crate) polymorphic: bool,
}

#[derive(Clone)]
pub(crate) enum ActiveRecordAssociationTargetName {
    Absolute(TypeName),
    Relative(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveRecordAssociationKind {
    BelongsTo,
    HasOne,
    HasMany,
}

/// A `scope :name, body` declaration, carried from collection to the AR
/// synthesis pass so `<Model>::GeneratedRelationMethods` mirrors the scope
/// with the same typed signature as the model's singleton def (orthoses
/// scope.rb writes the identical definition string on both sides).
#[derive(Clone)]
pub(crate) struct ActiveRecordScope {
    pub(crate) owner: TypeName,
    pub(crate) name: String,
    pub(crate) params: Option<InfusionScopeParams>,
}

pub(crate) fn scope_from_call(owner: TypeName, call: &InfusionCall) -> Option<ActiveRecordScope> {
    if call.name != "scope" {
        return None;
    }
    let arg = call.symbol_args.first()?;
    Some(ActiveRecordScope {
        owner,
        name: arg.name.clone(),
        params: call.scope_lambda_params.clone(),
    })
}

/// Enum scopes ride the same typed channel as `scope`: Rails defines each
/// of them via `klass.scope name, -> { where(...) }` (enum.rb:317,321), a
/// 0-arg lambda, so `params: Some(default)` yields the same
/// `() -> <Model>::ActiveRecord_Relation` on both the model and the
/// GeneratedRelationMethods mirror.
pub(crate) fn enum_scopes_from_call(
    owner: TypeName,
    call: &InfusionCall,
) -> Vec<ActiveRecordScope> {
    if call.name != "enum" || keyword_bool(call, "scopes") == Some(false) {
        return Vec::new();
    }
    let Some(arg) = call.symbol_args.first() else {
        return Vec::new();
    };
    let mut scopes = Vec::new();
    for value in &call.enum_values {
        for label in enum_label_and_alias(&value.name) {
            let value_method_name = enum_value_method_name(&arg.name, &label, call);
            for name in [
                value_method_name.clone(),
                format!("not_{value_method_name}"),
            ] {
                scopes.push(ActiveRecordScope {
                    owner,
                    name,
                    params: Some(InfusionScopeParams::default()),
                });
            }
        }
    }
    scopes
}

/// The `def self.<names>` enum mapping, carried to AR synthesis so
/// `GeneratedRelationMethods` mirrors it typed (the untyped sweep only
/// collects `TypeAnnotations::None` defs and no longer sees it).
#[derive(Clone)]
pub(crate) struct ActiveRecordEnumMapping {
    pub(crate) owner: TypeName,
    /// Pluralized mapping-method name (`statuses` for `enum :status`).
    pub(crate) name: String,
    /// Raw enum attribute name (`status`) — the schema-column suppression
    /// key in [`emit_schema`], matched against column names verbatim.
    pub(crate) attr_name: String,
    pub(crate) value: EnumMappingValue,
}

/// The mapping's value-type verdict, computed at collection where the
/// literals live (synthesis only converts it to a `Type`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnumMappingValue {
    Integer,
    String,
    Untyped,
}

pub(crate) fn enum_mapping_from_call(
    owner: TypeName,
    call: &InfusionCall,
    inflector: &Inflector,
) -> Option<ActiveRecordEnumMapping> {
    if call.name != "enum" {
        return None;
    }
    let arg = call.symbol_args.first()?;
    Some(ActiveRecordEnumMapping {
        owner,
        name: inflector.pluralize(&arg.name),
        attr_name: arg.name.clone(),
        value: mapping_value_kind(call),
    })
}

/// `() -> ::ActiveSupport::HashWithIndifferentAccess[::String, <value>]`
/// for the Relation-side mirror of an enum mapping.
pub(crate) fn enum_mapping_mirror_method_type(
    names: &NameTable,
    mapping: &ActiveRecordEnumMapping,
) -> MethodType {
    let value_type = match mapping.value {
        EnumMappingValue::Integer => class_instance("::Integer", names),
        EnumMappingValue::String => class_instance("::String", names),
        EnumMappingValue::Untyped => untyped_type(),
    };
    enum_mapping_method_type(names, value_type)
}

/// Orthoses scope.rb `parameters_to_type` ported 1:1: every param slot is
/// `untyped`, only the structure (arity/kind) is carried, and the return
/// type is always `<model>::ActiveRecord_Relation` (AR falls back to `all`
/// even when the scope body returns nil). `params: None` is the static
/// stand-in for orthoses's `(...)` case: `(?) -> Relation`.
pub(crate) fn scope_method_type(
    names: &NameTable,
    model: TypeName,
    params: Option<&InfusionScopeParams>,
) -> MethodType {
    let relation = Type::ClassInstance(ClassInstanceType {
        name: names.append_type_name(model, names.intern_symbol("ActiveRecord_Relation")),
        args: Vec::new(),
        location: None,
    });
    let untyped_param = |name: &String| FunctionParam {
        ty: Box::new(untyped_type()),
        name: Some(names.intern_symbol(name)),
        location: None,
    };
    let untyped_keyword = |name: &String| KeywordParam {
        name: names.intern_symbol(name),
        param: FunctionParam {
            ty: Box::new(untyped_type()),
            name: None,
            location: None,
        },
    };
    let (function, block) = match params {
        None => (
            Function::Untyped(UntypedFunctionType {
                return_type: Box::new(relation),
            }),
            None,
        ),
        Some(params) => (
            Function::Typed(FunctionType {
                required_positionals: params.requireds.iter().map(untyped_param).collect(),
                optional_positionals: params.optionals.iter().map(untyped_param).collect(),
                rest_positionals: params.rest.then(|| {
                    Box::new(FunctionParam {
                        ty: Box::new(untyped_type()),
                        name: None,
                        location: None,
                    })
                }),
                trailing_positionals: params.trailings.iter().map(untyped_param).collect(),
                required_keywords: params
                    .required_keywords
                    .iter()
                    .map(untyped_keyword)
                    .collect(),
                optional_keywords: params
                    .optional_keywords
                    .iter()
                    .map(untyped_keyword)
                    .collect(),
                rest_keywords: params.keyrest.then(|| {
                    Box::new(FunctionParam {
                        ty: Box::new(untyped_type()),
                        name: None,
                        location: None,
                    })
                }),
                return_type: Box::new(relation),
            }),
            // orthoses: `block = " { (*untyped) -> untyped }"` — required.
            params.block.then(|| BlockType {
                required: true,
                function: Function::Typed(FunctionType {
                    required_positionals: Vec::new(),
                    optional_positionals: Vec::new(),
                    rest_positionals: Some(Box::new(FunctionParam {
                        ty: Box::new(untyped_type()),
                        name: None,
                        location: None,
                    })),
                    trailing_positionals: Vec::new(),
                    required_keywords: Vec::new(),
                    optional_keywords: Vec::new(),
                    rest_keywords: None,
                    return_type: Box::new(untyped_type()),
                }),
                self_type: None,
            }),
        ),
    };
    MethodType {
        type_params: Vec::new(),
        function,
        block,
        location: None,
    }
}

/// Schema dump files under `<project_root>/db`, in a deterministic order.
///
/// Covers the dump targets Rails names under `schema_format: :ruby`
/// (`ActiveRecord::DatabaseConfigurations::HashConfig#schema_dump`): the
/// primary database dumps to `schema.rb` and every other database in a
/// multi-database setup dumps to `<name>_schema.rb`. The `:sql` format
/// (`structure.sql`) carries no `create_table` calls to read, and a
/// `database.yml` that overrides `schema_dump` with a name outside the
/// convention is not covered.
fn schema_paths(project_root: &Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(project_root.join("db")) else {
        return Vec::new();
    };
    let mut paths: Vec<std::path::PathBuf> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name == "schema.rb" || name.ends_with("_schema.rb")).then_some(path)
        })
        .collect();
    // read_dir order is filesystem-dependent; declaration insertion order
    // must not vary between machines.
    paths.sort();
    paths
}

/// `self.table_name = "..."` declarations collected from model class bodies,
/// keyed by table name. Values are absolute model paths (`::Admin::User`) in
/// source order — several models may legally share one table in Rails, and
/// each of them gets the table's columns.
#[derive(Default)]
pub struct TableNameOverrides {
    by_table: FxHashMap<String, Vec<String>>,
}

/// Model-file facts the schema pass needs before it runs, gathered by
/// [`collect_table_name_overrides`]'s walk: the table-name overrides.
/// Enum names are NOT collected here — a concern's `included do enum ...`
/// only surfaces after concern expansion, so the same-named-column
/// suppression set is built from the post-expansion
/// [`ActiveRecordEnumMapping`]s instead (see [`emit_schema`]).
#[derive(Default)]
pub struct ArModelPrepass {
    pub(crate) table_names: TableNameOverrides,
}

/// Pre-pass over the parsed Ruby sources, run inside [`prepare_schema`]:
/// collect `self.table_name = <string literal>` from class bodies. Only
/// statements directly in a class body count — a `table_name=` inside a
/// method or a conditional only takes effect at runtime, which is outside
/// the deterministic-input boundary infusion stays within. A non-literal
/// value is reported as `InfusionProviderSkipped` instead of guessed at.
pub fn collect_table_name_overrides(sources: &[SourceUnit]) -> (ArModelPrepass, Vec<Diagnostic>) {
    let mut assignments: Vec<(String, String)> = Vec::new();
    let mut diagnostics = Vec::new();
    for unit in sources {
        let mut stack = Vec::new();
        collect_overrides_in_body(
            &unit.parse_result.node(),
            &mut stack,
            false,
            unit.file,
            unit.source,
            &mut assignments,
            &mut diagnostics,
        );
    }
    // Ruby reassignment semantics: only a model's last `table_name =` is in
    // effect at runtime, so earlier assignments must not also claim their
    // tables. Resolve last-write-wins per model first, then group by table
    // in assignment order to keep declaration order deterministic.
    let mut final_table: FxHashMap<&str, &str> = FxHashMap::default();
    for (model, table) in &assignments {
        final_table.insert(model, table);
    }
    let mut overrides = TableNameOverrides::default();
    for (model, table) in &assignments {
        if final_table.get(model.as_str()) != Some(&table.as_str()) {
            continue;
        }
        let models = overrides.by_table.entry(table.clone()).or_default();
        if !models.contains(model) {
            models.push(model.clone());
        }
    }
    (
        ArModelPrepass {
            table_names: overrides,
        },
        diagnostics,
    )
}

fn collect_overrides_in_body(
    body: &Node<'_>,
    stack: &mut Vec<String>,
    in_class: bool,
    file: Option<&Path>,
    source: &[u8],
    assignments: &mut Vec<(String, String)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let statements = if let Some(program) = body.as_program_node() {
        program.statements().body()
    } else if let Some(statements) = body.as_statements_node() {
        statements.body()
    } else if let Some(begin) = body.as_begin_node() {
        let Some(statements) = begin.statements() else {
            return;
        };
        statements.body()
    } else {
        return;
    };
    for stmt in statements.iter() {
        if let Some(class_node) = stmt.as_class_node() {
            if push_class_abs_path(stack, &class_node.constant_path()).is_some() {
                if let Some(body) = class_node.body() {
                    collect_overrides_in_body(
                        &body,
                        stack,
                        true,
                        file,
                        source,
                        assignments,
                        diagnostics,
                    );
                }
                stack.pop();
            }
        } else if let Some(module_node) = stmt.as_module_node() {
            if push_class_abs_path(stack, &module_node.constant_path()).is_some() {
                if let Some(body) = module_node.body() {
                    collect_overrides_in_body(
                        &body,
                        stack,
                        false,
                        file,
                        source,
                        assignments,
                        diagnostics,
                    );
                }
                stack.pop();
            }
        } else if let Some(call) = stmt.as_call_node() {
            if !in_class
                || call.name().as_slice() != b"table_name="
                || call
                    .receiver()
                    .is_none_or(|receiver| receiver.as_self_node().is_none())
            {
                continue;
            }
            let Some(model) = stack.last() else {
                continue;
            };
            match first_string_arg(&call) {
                Some(table) => {
                    assignments.push((model.clone(), table));
                }
                None => {
                    let location = prism_location_range(call.location());
                    diagnostics.push(Diagnostic {
                        scope: None,
                        kind: DiagnosticKind::InfusionProviderSkipped {
                            provider: PROVIDER.to_string(),
                            subject: format!("{}.table_name", model),
                            reason: "table_name assignment is not a string literal".to_string(),
                        },
                        location: Diagnostic::location_for_byte_range(
                            file.map(|p| p.to_path_buf()).unwrap_or_default(),
                            source,
                            location.0 as usize,
                            location.1 as usize,
                        ),
                    });
                }
            }
        }
    }
}

/// One parsed schema dump: the tables plus the file identity needed later
/// to attribute the synthesized declarations and their diagnostics.
pub struct ParsedSchemaFile {
    path: std::path::PathBuf,
    source: Vec<u8>,
    tables: Vec<SchemaTable>,
}

/// The schema facts collected before `load_all` runs, emitted into the
/// draft after concern expansion by [`emit_schema`]: parsing needs nothing
/// from the Ruby sources, but the enum suppression set that emission
/// consumes only exists post-expansion.
pub struct PreparsedSchema {
    pub(crate) prepass: ArModelPrepass,
    pub(crate) files: Vec<ParsedSchemaFile>,
}

/// Parse the schema dumps and the model-file prepass facts. Runs before
/// `load_all`; the result is handed to `load_all_with_schema`, which emits
/// the declarations once the post-expansion enum set is known.
pub fn prepare_schema(
    project_root: &Path,
    sources: &[SourceUnit],
) -> (PreparsedSchema, Vec<Diagnostic>) {
    let (prepass, mut diagnostics) = collect_table_name_overrides(sources);
    let mut files = Vec::new();
    let mut seen_tables = FxHashSet::default();
    for schema_path in schema_paths(project_root) {
        if let Some(file) = parse_schema_file(
            project_root,
            &schema_path,
            &mut seen_tables,
            &mut diagnostics,
        ) {
            files.push(file);
        }
    }
    (PreparsedSchema { prepass, files }, diagnostics)
}

/// Synthesize and insert the schema-derived declarations. `enums_by_attr`
/// is the post-expansion enum attribute set per model (direct class-body
/// enums and concern-expanded ones alike): a same-named column's typed
/// reader/writer is dropped in its favor, the `name?` query method stays.
/// Rails overrides the column accessors at runtime (label strings, not raw
/// values), so the enum-side typed defs are the truth and keeping both
/// would raise DuplicatedMethodDefinitionError on every real enum-backed
/// column.
pub(crate) fn emit_schema(
    schema: &PreparsedSchema,
    draft: &mut EnvironmentDraft,
    inflector: &Inflector,
    enums_by_attr: &FxHashMap<TypeName, FxHashSet<String>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for file in &schema.files {
        let declarations = schema_declarations(
            draft.names(),
            &file.path,
            &file.source,
            &file.tables,
            inflector,
            &schema.prepass,
            enums_by_attr,
            diagnostics,
        );
        for declaration in &declarations {
            draft.insert_ruby_decl(declaration, &file.source, Some(&file.path), diagnostics);
        }
    }
}

/// Parse one schema dump. A parse failure skips this file alone — the
/// other databases' dumps still contribute their tables.
///
/// A table already claimed by an earlier dump (`seen_tables`) is skipped:
/// horizontal sharding dumps the same schema once per shard, and those files
/// describe one logical table backing one model, not several.
fn parse_schema_file(
    project_root: &Path,
    schema_path: &Path,
    seen_tables: &mut FxHashSet<String>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ParsedSchemaFile> {
    let source = std::fs::read(schema_path).ok()?;
    // Scoped: `ParseResult` borrows `source`, which moves into the return
    // value after the parse artifacts are consumed.
    let tables = {
        let parse_result = ruby_prism::parse(&source);
        if let Some(first_err) = parse_result.errors().next() {
            let err_location = first_err.location();
            let subject = schema_path
                .strip_prefix(project_root)
                .unwrap_or(schema_path)
                .display()
                .to_string();
            diagnostics.push(Diagnostic {
                scope: None,
                kind: DiagnosticKind::InfusionProviderSkipped {
                    provider: PROVIDER.to_string(),
                    subject,
                    reason: format!("schema.rb parse error: {}", first_err.message()),
                },
                location: Diagnostic::location_for_byte_range(
                    schema_path.to_path_buf(),
                    &source,
                    err_location.start_offset(),
                    err_location.end_offset(),
                ),
            });
            return None;
        }

        let mut collector = SchemaCollector::default();
        collector.visit(&parse_result.node());
        collector
            .tables
            .into_iter()
            .filter(|table| seen_tables.insert(table.name.clone()))
            .collect()
    };
    Some(ParsedSchemaFile {
        path: schema_path.to_path_buf(),
        source,
        tables,
    })
}

pub(crate) fn collect_call(
    members: &mut Vec<Member>,
    call: &InfusionCall,
    inflector: &Inflector,
    names: &NameTable,
    owner: TypeName,
) {
    collect_belongs_to_call(members, call);
    collect_has_one_call(members, call);
    collect_has_many_call(members, call, inflector);
    collect_enum_call(members, call, inflector, names, owner);
    collect_scope_call(members, call, names, owner);
}

pub(crate) fn association_from_call(
    names: &NameTable,
    owner: TypeName,
    call: &InfusionCall,
) -> Option<ActiveRecordAssociation> {
    let kind = match call.name.as_str() {
        "belongs_to" => ActiveRecordAssociationKind::BelongsTo,
        "has_one" => ActiveRecordAssociationKind::HasOne,
        "has_many" => ActiveRecordAssociationKind::HasMany,
        _ => return None,
    };
    let arg = call.symbol_args.first()?;
    let target_name = keyword_string(call, "class_name").map(|raw| {
        if raw.starts_with("::") {
            ActiveRecordAssociationTargetName::Absolute(names.parse_type_name(raw))
        } else {
            ActiveRecordAssociationTargetName::Relative(raw.to_string())
        }
    });
    Some(ActiveRecordAssociation {
        owner,
        kind,
        name: arg.name.clone(),
        target_name,
        polymorphic: keyword_bool(call, "polymorphic") == Some(true),
    })
}

#[derive(Default)]
struct SchemaCollector {
    tables: Vec<SchemaTable>,
}

struct SchemaTable {
    name: String,
    location: PrismByteRange,
    columns: Vec<SchemaColumn>,
}

struct SchemaColumn {
    name: String,
    sql_type: String,
    nullable: bool,
    location: PrismByteRange,
}

struct ColumnCollector {
    receiver_name: String,
    columns: Vec<SchemaColumn>,
}

impl<'pr> Visit<'pr> for SchemaCollector {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if node.receiver().is_some() || node.name().as_slice() != b"create_table" {
            ruby_prism::visit_call_node(self, node);
            return;
        }
        let Some(table_name) = first_string_arg(node) else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        let Some(block) = node.block().and_then(|block| block.as_block_node()) else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        let Some(body) = block.body() else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        let receiver_name = first_block_param_name(&block).unwrap_or_else(|| "t".to_string());
        let mut collector = ColumnCollector {
            receiver_name,
            columns: Vec::new(),
        };
        collector.visit(&body);
        self.tables.push(SchemaTable {
            name: table_name,
            location: prism_location_range(node.location()),
            columns: collector.columns,
        });
    }
}

impl<'pr> Visit<'pr> for ColumnCollector {
    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        let Some(receiver) = node.receiver() else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        if !receiver_is_lvar(&receiver, &self.receiver_name) {
            ruby_prism::visit_call_node(self, node);
            return;
        }
        let call_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        if NON_COLUMN_METHODS.contains(&call_name.as_str()) {
            ruby_prism::visit_call_node(self, node);
            return;
        }
        let Some(column_name) = string_or_symbol_arg(node, 0) else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        let Some(sql_type) = schema_column_type(node, &call_name) else {
            ruby_prism::visit_call_node(self, node);
            return;
        };
        self.columns.push(SchemaColumn {
            name: column_name,
            sql_type,
            nullable: keyword_bool_node(node, "null").unwrap_or(true),
            location: prism_location_range(node.location()),
        });
    }
}

#[allow(clippy::too_many_arguments)] // one synthesis site, mirrors emit_schema's inputs
fn schema_declarations(
    names: &NameTable,
    schema_path: &Path,
    source: &[u8],
    tables: &[SchemaTable],
    inflector: &Inflector,
    prepass: &ArModelPrepass,
    enums_by_attr: &FxHashMap<TypeName, FxHashSet<String>>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    for table in tables {
        // A `self.table_name =` override consumes the table: the columns go
        // to the declaring model(s) and the inflection-derived class (which
        // no real model backs) is not synthesized at all.
        let model_absolutes: Vec<String> =
            if let Some(models) = prepass.table_names.by_table.get(&table.name) {
                models.clone()
            } else if let Some(model_name) = model_name_for_table(&table.name, inflector) {
                vec![format!("::{}", model_name)]
            } else {
                diagnostics.push(skip_diagnostic(
                    schema_path,
                    source,
                    table.location,
                    &table.name,
                    "irregular table name is outside the naive inflector",
                ));
                continue;
            };
        let mut column_members = Vec::new();
        let mut column_kwargs: Vec<(String, Type)> = Vec::new();
        for column in &table.columns {
            let Some(column_type) = type_for_sql(&column.sql_type, names) else {
                diagnostics.push(skip_diagnostic(
                    schema_path,
                    source,
                    column.location,
                    &format!("{}.{}", table.name, column.name),
                    &format!("unknown SQL type `{}`", column.sql_type),
                ));
                continue;
            };
            let ty = if column.nullable {
                optional(column_type)
            } else {
                column_type
            };
            push_typed_def(
                &mut column_members,
                column.name.clone(),
                vec![],
                ty.clone(),
                column.location,
            );
            push_typed_def(
                &mut column_members,
                format!("{}=", column.name),
                vec![ty.clone()],
                ty.clone(),
                column.location,
            );
            push_typed_def(
                &mut column_members,
                format!("{}?", column.name),
                vec![],
                bool_type(),
                column.location,
            );
            column_kwargs.push((column.name.clone(), ty));
        }
        if column_members.is_empty() {
            continue;
        }

        // Each schema-derived AR model gets two dedicated `*_ClassMethods`
        // modules and extends both. `ActiveRecord_Persistence_ClassMethods`
        // (create / create! / build) is a faithful port of
        // orthoses-rails' `Orthoses::ActiveRecord::Persistence` — the module
        // name mirrors orthoses's own `"#{base_name}::ActiveRecord_Persistence_ClassMethods"`.
        // Each persistence method carries two overloads (kwarg-column form
        // and bulk `Array[Hash]` form), mirroring orthoses persistence.rb's
        // own `|`-joined signature; `new` keeps a single overload since Rails
        // never treats it as bulk-capable.
        // `ActiveRecord_Inheritance_ClassMethods` (new) has no orthoses
        // counterpart; the name and shape are ported directly from Rails
        // source (`activerecord/lib/active_record/inheritance.rb`'s
        // `ClassMethods#new`), matching the sibling paradigm used by
        // `src/infusion_collector/active_record_synthesis.rs` (`ActiveRecord_Relation`).
        // Modules — not interfaces — so users can reopen them from their own
        // sig files. Return type is the `instance` base type so STI
        // subclasses (`class Sub < Speaker`) get their own type back from
        // `Sub.new`, not the base model's.
        for model_absolute in &model_absolutes {
            let mut members = column_members.clone();
            // A same-named enum owns the reader/writer surface (Rails
            // redefines the column accessors); only the `name?` query
            // method is collision-free and stays. See [`emit_schema`].
            if let Some(enum_names) = enums_by_attr.get(&names.parse_type_name(model_absolute)) {
                members.retain(|member| match member {
                    Member::Def(def) => {
                        let base = def.name.strip_suffix('=').unwrap_or(&def.name);
                        def.name.ends_with('?') || !enum_names.contains(base)
                    }
                    _ => true,
                });
            }
            let persistence_module =
                format!("{}::ActiveRecord_Persistence_ClassMethods", model_absolute);
            let inheritance_module =
                format!("{}::ActiveRecord_Inheritance_ClassMethods", model_absolute);

            push_extend(&mut members, &persistence_module, table.location);
            push_extend(&mut members, &inheritance_module, table.location);

            let mut persistence_members = Vec::new();
            for method_name in ["create", "create!", "build"] {
                push_ar_persistence_class_method(
                    &mut persistence_members,
                    method_name,
                    &column_kwargs,
                    names,
                    table.location,
                );
            }
            let mut inheritance_members = Vec::new();
            push_ar_kwarg_class_method(
                &mut inheritance_members,
                "new",
                &column_kwargs,
                names,
                table.location,
            );

            declarations.push(Declaration::Module(Arc::new(ModuleDecl {
                module_name: names.parse_type_name(&persistence_module),
                name_location: table.location,
                members: persistence_members,
            })));
            declarations.push(Declaration::Module(Arc::new(ModuleDecl {
                module_name: names.parse_type_name(&inheritance_module),
                name_location: table.location,
                members: inheritance_members,
            })));
            declarations.push(Declaration::Class(Arc::new(ClassDecl {
                class_name: names.parse_type_name(model_absolute),
                name_location: table.location,
                super_class: None,
                members,
                block_body: false,
            })));
        }
    }
    declarations
}

fn push_extend(members: &mut Vec<Member>, module_name: &str, location: PrismByteRange) {
    members.push(Member::Extend(ExtendMember {
        mixin: MixinMember {
            module_name: module_name.to_string(),
            location,
            name_location: location,
            annotation: None,
        },
    }));
}

/// Build one of the AR ctor-family class methods (`new` / `create` / `create!` /
/// `build`) as an instance method of a `*_ClassMethods` module (`extend M` on
/// the model surfaces it as a singleton method). The kwarg-column overload
/// shape mirrors orthoses persistence.rb, with `instance` (not the base model
/// type) as the return so STI subclasses stay well-typed:
///
/// ```text
/// def <name>: (?col1: T1?, ?col2: T2, ..., **untyped) ?{ (instance) -> void } -> instance
/// ```
///
/// `**untyped` is load-bearing: Rails accepts any `=`-suffixed setter (from
/// attr_accessor / user-defined attribute) as a kwarg, and the static schema
/// cannot know those. `?{ (instance) -> void }` matches the block form
/// `Speaker.new { |s| s.foo = ... }` — the block param is typed as `instance`
/// so `Sub.new { |sub| sub.only_on_sub = ... }` sees `sub` as the subclass.
///
/// `new` never gets a bulk-array overload (Rails' `Inheritance#new` doesn't
/// branch on `is_a?(Array)`), so it stays single-overload via this function;
/// see `push_ar_persistence_class_method` for the two-overload sibling used
/// by `create` / `create!` / `build`.
fn push_ar_kwarg_class_method(
    members: &mut Vec<Member>,
    method_name: &str,
    columns: &[(String, Type)],
    names: &NameTable,
    location: PrismByteRange,
) {
    push_overloaded_ctor_def(
        members,
        method_name,
        vec![kwarg_overload_method_type(columns, names)],
        location,
    );
}

/// Like `push_ar_kwarg_class_method`, but for `create` / `create!` / `build`,
/// which Rails (and orthoses persistence.rb) additionally accept a single
/// `Array[Hash]` positional for bulk creation:
///
/// ```text
/// def <name>: (?col1: T1?, ..., **untyped) ?{ (instance) -> void } -> instance
///            | (::Array[::Hash[::Symbol, untyped]]) ?{ (instance) -> void } -> ::Array[instance]
/// ```
fn push_ar_persistence_class_method(
    members: &mut Vec<Member>,
    method_name: &str,
    columns: &[(String, Type)],
    names: &NameTable,
    location: PrismByteRange,
) {
    push_overloaded_ctor_def(
        members,
        method_name,
        vec![
            kwarg_overload_method_type(columns, names),
            bulk_array_overload_method_type(names),
        ],
        location,
    );
}

fn push_overloaded_ctor_def(
    members: &mut Vec<Member>,
    method_name: &str,
    overloads: Vec<MethodType>,
    location: PrismByteRange,
) {
    let annotations = overloads
        .into_iter()
        .map(|method_type| {
            ExplicitAnnotation::Colon(ColonMethodTypeAnnotation {
                location,
                prefix_location: location,
                annotations: Vec::new(),
                method_type,
            })
        })
        .collect();
    members.push(Member::Def(DefMember {
        ivar_param_pairs: Vec::new(),
        name: method_name.to_string(),
        kind: MethodKind::Instance,
        location,
        name_location: location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(annotations),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    }));
}

fn kwarg_overload_method_type(columns: &[(String, Type)], names: &NameTable) -> MethodType {
    let optional_keywords: Vec<KeywordParam> = columns
        .iter()
        .map(|(col_name, col_ty)| KeywordParam {
            name: names.intern_symbol(col_name),
            param: FunctionParam {
                ty: Box::new(col_ty.clone()),
                name: None,
                location: None,
            },
        })
        .collect();
    let rest_keywords = Some(Box::new(FunctionParam {
        ty: Box::new(untyped_type()),
        name: None,
        location: None,
    }));
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals: Vec::new(),
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords,
            rest_keywords,
            return_type: Box::new(instance_type()),
        }),
        block: instance_yielding_block(),
        location: None,
    }
}

/// The bulk-array overload's shape, ported verbatim from orthoses
/// persistence.rb (`::Array[Hash[Symbol, untyped]]` in, `::Array[#{base_name}]`
/// out — using `instance` per the STI note above instead of orthoses's fixed
/// `base_name`).
fn bulk_array_overload_method_type(names: &NameTable) -> MethodType {
    let hash_of_symbol_untyped = class_instance_generic(
        "::Hash",
        vec![class_instance("::Symbol", names), untyped_type()],
        names,
    );
    let array_of_hash = class_instance_generic("::Array", vec![hash_of_symbol_untyped], names);
    let array_of_instance = class_instance_generic("::Array", vec![instance_type()], names);
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals: vec![FunctionParam {
                ty: Box::new(array_of_hash),
                name: None,
                location: None,
            }],
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(array_of_instance),
        }),
        block: instance_yielding_block(),
        location: None,
    }
}

/// `?{ (instance) -> void }` — shared by both overloads above.
fn instance_yielding_block() -> Option<BlockType> {
    Some(BlockType {
        required: false,
        function: Function::Typed(FunctionType {
            required_positionals: vec![FunctionParam {
                ty: Box::new(instance_type()),
                name: None,
                location: None,
            }],
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(void_type()),
        }),
        self_type: None,
    })
}

fn untyped_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Any { todo: false },
        location: None,
    })
}

fn void_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Void,
        location: None,
    })
}

/// `instance` — rbs `Types::Bases::Instance`. Method dispatch substitutes it
/// with the receiver's actual class, so a `def new: (...) -> instance`
/// declared on the base model resolves to the subclass when a subclass calls
/// `Sub.new`.
fn instance_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Instance,
        location: None,
    })
}

fn push_typed_def(
    members: &mut Vec<Member>,
    name: String,
    required_positionals: Vec<Type>,
    return_type: Type,
    location: PrismByteRange,
) {
    let method_type = MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals: required_positionals
                .into_iter()
                .map(|ty| FunctionParam {
                    ty: Box::new(ty),
                    name: None,
                    location: None,
                })
                .collect(),
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(return_type),
        }),
        block: None,
        location: None,
    };
    members.push(Member::Def(DefMember {
        ivar_param_pairs: Vec::new(),
        name,
        kind: MethodKind::Instance,
        location,
        name_location: location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(vec![ExplicitAnnotation::Colon(
                ColonMethodTypeAnnotation {
                    location,
                    prefix_location: location,
                    annotations: Vec::new(),
                    method_type,
                },
            )]),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    }));
}

fn first_string_arg(node: &CallNode<'_>) -> Option<String> {
    let arguments = node.arguments()?;
    let first = arguments.arguments().iter().next()?;
    first
        .as_string_node()
        .map(|string| String::from_utf8_lossy(string.unescaped()).to_string())
}

fn string_or_symbol_arg(node: &CallNode<'_>, index: usize) -> Option<String> {
    let arguments = node.arguments()?;
    let arg = arguments.arguments().iter().nth(index)?;
    if let Some(symbol) = arg.as_symbol_node() {
        Some(String::from_utf8_lossy(symbol.unescaped()).to_string())
    } else {
        arg.as_string_node()
            .map(|string| String::from_utf8_lossy(string.unescaped()).to_string())
    }
}

/// `t.check_constraint`/`t.index`/`t.foreign_key`/`t.exclusion_constraint`/
/// `t.unique_constraint` are table-level constraint declarations, not column
/// definitions — `ColumnCollector` must ignore them rather than treat their
/// first positional arg as a column name. The PG-only pair
/// (`exclusion_constraint`/`unique_constraint`, `TableDefinition` methods in
/// `postgresql/schema_definitions.rb`) is dumped inside the `create_table`
/// block by `PostgreSQL::SchemaDumper` alongside the other three, same as
/// `activerecord-8.1.3/lib/active_record/schema_dumper.rb`'s `#table`.
const NON_COLUMN_METHODS: &[&str] = &[
    "check_constraint",
    "index",
    "foreign_key",
    "exclusion_constraint",
    "unique_constraint",
];

fn schema_column_type(node: &CallNode<'_>, call_name: &str) -> Option<String> {
    if call_name == "column" {
        string_or_symbol_arg(node, 1)
    } else {
        Some(call_name.to_string())
    }
}

fn first_block_param_name(block: &ruby_prism::BlockNode<'_>) -> Option<String> {
    let params_node = block.parameters()?;
    let block_params = params_node.as_block_parameters_node()?;
    let params = block_params.parameters()?;
    let first = params.requireds().iter().next()?;
    let required = first.as_required_parameter_node()?;
    Some(String::from_utf8_lossy(required.name().as_slice()).to_string())
}

fn receiver_is_lvar(node: &Node<'_>, name: &str) -> bool {
    node.as_local_variable_read_node()
        .is_some_and(|lvar| lvar.name().as_slice() == name.as_bytes())
}

fn keyword_bool_node(node: &CallNode<'_>, name: &str) -> Option<bool> {
    let arguments = node.arguments()?;
    let args: Vec<_> = arguments.arguments().iter().collect();
    for arg in args.into_iter().rev() {
        let Some(kw_hash) = arg.as_keyword_hash_node() else {
            continue;
        };
        let elems: Vec<_> = kw_hash.elements().iter().collect();
        for elem in elems.into_iter().rev() {
            let Some(assoc) = elem.as_assoc_node() else {
                continue;
            };
            let Some(sym) = assoc.key().as_symbol_node() else {
                continue;
            };
            if sym.unescaped() != name.as_bytes() {
                continue;
            }
            let value = assoc.value();
            if value.as_true_node().is_some() {
                return Some(true);
            }
            if value.as_false_node().is_some() {
                return Some(false);
            }
        }
    }
    None
}

fn type_for_sql(sql_type: &str, names: &NameTable) -> Option<Type> {
    let raw = match sql_type {
        "integer" | "bigint" | "big_integer" | "serial" | "bigserial" | "oid" => "::Integer",
        "float" => "::Float",
        "decimal" | "money" => "::BigDecimal",
        "string" | "text" | "citext" | "uuid" | "binary" | "immutable_string" | "xml" | "enum"
        | "bit" | "bit_varying" | "ltree" | "macaddr" | "tsvector" => "::String",
        "datetime" | "timestamp" | "timestamptz" => "::ActiveSupport::TimeWithZone",
        "boolean" => return Some(bool_type()),
        "date" => "::Date",
        "time" => "::Time",
        "cidr" | "inet" => "::IPAddr",
        "interval" => "::ActiveSupport::Duration",
        // `PostgreSQL::OID::Hstore#deserialize` maps the `NULL` marker to a
        // Ruby `nil` value (`hstore.rb`'s `scanner.scan(/NULL/)` branch), so
        // the value side is nilable even though the key side never is.
        "hstore" => {
            return Some(class_instance_generic(
                "::Hash",
                vec![
                    class_instance("::String", names),
                    optional(class_instance("::String", names)),
                ],
                names,
            ));
        }
        // Structurally unstable/complex shapes (json's schema-free payload,
        // PG geometric/range generics) map to `untyped` rather than being
        // modeled precisely.
        "json" | "jsonb" | "point" | "line" | "lseg" | "box" | "path" | "polygon" | "circle"
        | "daterange" | "numrange" | "int4range" | "int8range" | "tsrange" | "tstzrange" => {
            return Some(untyped_type());
        }
        _ => return None,
    };
    Some(class_instance(raw, names))
}

fn class_instance(raw: &str, names: &NameTable) -> Type {
    Type::ClassInstance(ClassInstanceType {
        name: names.parse_type_name(raw),
        args: Vec::new(),
        location: None,
    })
}

fn class_instance_generic(raw: &str, args: Vec<Type>, names: &NameTable) -> Type {
    Type::ClassInstance(ClassInstanceType {
        name: names.parse_type_name(raw),
        args,
        location: None,
    })
}

fn optional(ty: Type) -> Type {
    Type::Optional(OptionalType {
        ty: Box::new(ty),
        location: None,
    })
}

fn bool_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Bool,
        location: None,
    })
}

fn model_name_for_table(table: &str, inflector: &Inflector) -> Option<String> {
    let singular = inflector.singularize(table);
    let class_name = inflector.camelize(&singular);
    (!class_name.is_empty()).then_some(class_name)
}

fn skip_diagnostic(
    file: &Path,
    source: &[u8],
    location: PrismByteRange,
    subject: &str,
    reason: &str,
) -> Diagnostic {
    let offset = location.0 as usize;
    Diagnostic {
        scope: None,
        kind: DiagnosticKind::InfusionProviderSkipped {
            provider: PROVIDER.to_string(),
            subject: subject.to_string(),
            reason: reason.to_string(),
        },
        location: Diagnostic::location_for_byte_range(
            file.to_path_buf(),
            source,
            offset,
            location.1 as usize,
        ),
    }
}

fn collect_belongs_to_call(members: &mut Vec<Member>, call: &InfusionCall) {
    if call.name != "belongs_to" {
        return;
    }
    let Some(arg) = call.symbol_args.first() else {
        return;
    };
    let writer = format!("{}=", arg.name);
    let reloader = format!("reload_{}", arg.name);

    for name in [arg.name.clone(), writer, reloader] {
        push_def(
            members,
            name,
            MethodKind::Instance,
            call.location,
            arg.location,
        );
    }
    if keyword_bool(call, "polymorphic") == Some(true) {
        return;
    }
    push_association_builders(members, &arg.name, call.location, arg.location);
}

fn collect_has_one_call(members: &mut Vec<Member>, call: &InfusionCall) {
    if call.name != "has_one" {
        return;
    }
    let Some(arg) = call.symbol_args.first() else {
        return;
    };
    let writer = format!("{}=", arg.name);
    let reloader = format!("reload_{}", arg.name);

    for name in [arg.name.clone(), writer, reloader] {
        push_def(
            members,
            name,
            MethodKind::Instance,
            call.location,
            arg.location,
        );
    }
    push_association_builders(members, &arg.name, call.location, arg.location);
}

fn collect_has_many_call(members: &mut Vec<Member>, call: &InfusionCall, inflector: &Inflector) {
    if call.name != "has_many" {
        return;
    }
    let Some(arg) = call.symbol_args.first() else {
        return;
    };
    let setter = format!("{}=", arg.name);

    push_def(
        members,
        arg.name.clone(),
        MethodKind::Instance,
        call.location,
        arg.location,
    );
    push_def(
        members,
        setter,
        MethodKind::Instance,
        call.location,
        arg.location,
    );
    let singular = inflector.singularize(&arg.name);
    if !singular.is_empty() {
        let ids_reader = format!("{singular}_ids");
        let ids_writer = format!("{ids_reader}=");
        push_def(
            members,
            ids_reader,
            MethodKind::Instance,
            call.location,
            arg.location,
        );
        push_def(
            members,
            ids_writer,
            MethodKind::Instance,
            call.location,
            arg.location,
        );
    }
}

/// Synthesizes the model-side `def self.<name>: (<params>) -> <Model>::ActiveRecord_Relation`
/// (orthoses scope.rb line 24). The explicit annotation also keeps the def
/// out of `collect_ruby_class_methods`' untyped sweep — the Relation-side
/// mirror is supplied typed via [`ActiveRecordScope`] instead.
fn collect_scope_call(
    members: &mut Vec<Member>,
    call: &InfusionCall,
    names: &NameTable,
    owner: TypeName,
) {
    if call.name != "scope" {
        return;
    }
    let Some(arg) = call.symbol_args.first() else {
        return;
    };
    let method_type = scope_method_type(names, owner, call.scope_lambda_params.as_ref());
    // Redeclaring a scope replaces the singleton method at runtime
    // (`define_singleton_method`), so the last declaration wins. Keeping
    // both defs would arity-check against the first one and, being typed,
    // trip the duplicate-definition validator. Mirrors the last-wins dedup
    // on the GeneratedRelationMethods side.
    members.retain(|member| {
        !matches!(
            member,
            Member::Def(def)
                if def.kind == MethodKind::Singleton && def.name == arg.name
        )
    });
    members.push(Member::Def(DefMember {
        ivar_param_pairs: Vec::new(),
        name: arg.name.clone(),
        kind: MethodKind::Singleton,
        location: call.location,
        name_location: arg.location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(vec![ExplicitAnnotation::Colon(
                ColonMethodTypeAnnotation {
                    location: call.location,
                    prefix_location: call.location,
                    annotations: Vec::new(),
                    method_type,
                },
            )]),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    }));
}

/// Synthesizes the typed enum surface, ported from orthoses enum.rb
/// (`_enum`) with the type-alias pair inlined as literal unions (Ruby-decl
/// AST has no TypeAlias variant; user decision 2026-08-26):
///
/// ```text
/// def <name>: () -> ("a" | "b")
/// def <name>=: (("a" | "b")) -> void | ((:a | :b)) -> void | ((0 | 1)) -> void
/// def <label>?: () -> bool          (also sanitized alias, Rails enum.rb:271-278)
/// def <label>!: () -> bool
/// def self.<names>: () -> ::ActiveSupport::HashWithIndifferentAccess[::String, ::Integer]
/// def self.<label>: () -> <Model>::ActiveRecord_Relation   (0-arg lambda, Rails enum.rb:317)
/// def self.not_<label>: () -> <Model>::ActiveRecord_Relation
/// ```
///
/// Non-literal values (constant refs) drop only the value-typed surface to
/// `untyped` (setter value overload, mapping value type); the label-derived
/// string/symbol surface stays typed. Being typed also keeps every def out
/// of `collect_ruby_class_methods`' untyped sweep — the Relation-side
/// mirror arrives via [`ActiveRecordScope`] / [`ActiveRecordEnumMapping`].
fn collect_enum_call(
    members: &mut Vec<Member>,
    call: &InfusionCall,
    inflector: &Inflector,
    names: &NameTable,
    owner: TypeName,
) {
    if call.name != "enum" {
        return;
    }
    let Some(arg) = call.symbol_args.first() else {
        return;
    };

    let labels: Vec<&str> = call.enum_values.iter().map(|v| v.name.as_str()).collect();
    let string_union = literal_union(
        labels
            .iter()
            .map(|label| Literal::String((*label).to_string())),
    );
    // orthoses `name_symbol`: any label outside /[a-zA-Z_]/ widens the
    // whole symbol overload to ::Symbol instead of a literal union.
    let symbol_union = if labels.iter().any(|label| {
        label
            .chars()
            .any(|c| !(c.is_ascii_alphabetic() || c == '_'))
    }) {
        class_instance("::Symbol", names)
    } else {
        literal_union(
            labels
                .iter()
                .map(|label| Literal::Symbol(names.intern_symbol(label))),
        )
    };
    let value_union = call
        .enum_values
        .iter()
        .map(|v| v.value.as_ref())
        .collect::<Option<Vec<_>>>()
        .map(|values| {
            literal_union(values.into_iter().map(|value| match value {
                EnumLiteral::Int(raw) => Literal::Integer(raw.clone()),
                EnumLiteral::Str(raw) => Literal::String(raw.clone()),
            }))
        })
        .unwrap_or_else(untyped_type);

    push_typed_def(
        members,
        arg.name.clone(),
        vec![],
        string_union.clone(),
        arg.location,
    );
    push_instance_overloaded_def(
        members,
        format!("{}=", arg.name),
        [string_union, symbol_union, value_union]
            .into_iter()
            .map(unary_void_method_type)
            .collect(),
        call.location,
    );

    push_singleton_typed_def(
        members,
        inflector.pluralize(&arg.name),
        enum_mapping_method_type(names, mapping_value_type(call, names)),
        call.location,
        arg.location,
    );
    let scopes_enabled = keyword_bool(call, "scopes") != Some(false);
    for value in &call.enum_values {
        for label in enum_label_and_alias(&value.name) {
            let value_method_name = enum_value_method_name(&arg.name, &label, call);
            if scopes_enabled {
                push_singleton_typed_def(
                    members,
                    value_method_name.clone(),
                    scope_method_type(names, owner, Some(&InfusionScopeParams::default())),
                    call.location,
                    value.location,
                );
                push_singleton_typed_def(
                    members,
                    format!("not_{value_method_name}"),
                    scope_method_type(names, owner, Some(&InfusionScopeParams::default())),
                    call.location,
                    value.location,
                );
            }
            for method in [
                format!("{value_method_name}?"),
                format!("{value_method_name}!"),
            ] {
                push_typed_def(members, method, vec![], bool_type(), value.location);
            }
        }
    }
}

/// The raw label plus its sanitized alias when they differ — Rails
/// enum.rb:271-278 runs `define_enum_methods` for both spellings
/// (`:"on-hold"` also gets `on_hold?` etc.).
fn enum_label_and_alias(label: &str) -> Vec<String> {
    let alias = sanitize_enum_label(label);
    if alias == label {
        vec![label.to_string()]
    } else {
        vec![label.to_string(), alias]
    }
}

/// Rails `label.gsub(/[\W&&[:ascii:]]+/, "_")`: each run of ASCII
/// non-word characters collapses to one underscore; non-ASCII passes
/// through untouched.
fn sanitize_enum_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut in_run = false;
    for c in label.chars() {
        if c.is_ascii() && !(c.is_ascii_alphanumeric() || c == '_') {
            if !in_run {
                out.push('_');
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

/// `A | B | ...` from literal members; a single member stays a bare
/// literal type (rbs prints and parses it the same way).
fn literal_union(literals: impl Iterator<Item = Literal>) -> Type {
    let mut types: Vec<Type> = literals
        .map(|literal| {
            Type::Literal(LiteralType {
                literal,
                location: None,
            })
        })
        .collect();
    match types.len() {
        0 => untyped_type(),
        1 => types.remove(0),
        _ => Type::Union(UnionType {
            types,
            location: None,
        }),
    }
}

/// `(<param>) -> void` — one setter overload.
fn unary_void_method_type(param: Type) -> MethodType {
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals: vec![FunctionParam {
                ty: Box::new(param),
                name: None,
                location: None,
            }],
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(void_type()),
        }),
        block: None,
        location: None,
    }
}

/// `() -> ::ActiveSupport::HashWithIndifferentAccess[::String, <value>]`
/// — the `self.<names>` mapping. The external type is emitted
/// unconditionally, same stance as `type_for_sql`'s TimeWithZone.
fn enum_mapping_method_type(names: &NameTable, value_type: Type) -> MethodType {
    let hash = class_instance_generic(
        "::ActiveSupport::HashWithIndifferentAccess",
        vec![class_instance("::String", names), value_type],
        names,
    );
    MethodType {
        type_params: Vec::new(),
        function: Function::Typed(FunctionType {
            required_positionals: Vec::new(),
            optional_positionals: Vec::new(),
            rest_positionals: None,
            trailing_positionals: Vec::new(),
            required_keywords: Vec::new(),
            optional_keywords: Vec::new(),
            rest_keywords: None,
            return_type: Box::new(hash),
        }),
        block: None,
        location: None,
    }
}

/// The mapping's value type from the collected value literals: a single
/// literal kind maps to its class, anything else (mixed kinds, any
/// non-literal) falls back to `untyped`. Deliberate deviation from
/// orthoses' `Hash === values ? [String, String]` — crema holds the
/// static literals and can be exact (todo enum_typed_signature).
fn mapping_value_kind(call: &InfusionCall) -> EnumMappingValue {
    let mut kinds = call.enum_values.iter().map(|v| {
        v.value.as_ref().map(|value| match value {
            EnumLiteral::Int(_) => EnumMappingValue::Integer,
            EnumLiteral::Str(_) => EnumMappingValue::String,
        })
    });
    let Some(Some(first)) = kinds.next() else {
        return EnumMappingValue::Untyped;
    };
    if kinds.all(|kind| kind == Some(first)) {
        first
    } else {
        EnumMappingValue::Untyped
    }
}

fn mapping_value_type(call: &InfusionCall, names: &NameTable) -> Type {
    match mapping_value_kind(call) {
        EnumMappingValue::Integer => class_instance("::Integer", names),
        EnumMappingValue::String => class_instance("::String", names),
        EnumMappingValue::Untyped => untyped_type(),
    }
}

/// Singleton typed def with last-wins dedup: redeclaring the same enum
/// (or a scope of the same name) replaces the runtime singleton method,
/// mirroring `collect_scope_call`'s retain.
fn push_singleton_typed_def(
    members: &mut Vec<Member>,
    name: String,
    method_type: MethodType,
    location: PrismByteRange,
    name_location: PrismByteRange,
) {
    members.retain(|member| {
        !matches!(
            member,
            Member::Def(def)
                if def.kind == MethodKind::Singleton && def.name == name
        )
    });
    members.push(Member::Def(DefMember {
        ivar_param_pairs: Vec::new(),
        name,
        kind: MethodKind::Singleton,
        location,
        name_location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(vec![ExplicitAnnotation::Colon(
                ColonMethodTypeAnnotation {
                    location,
                    prefix_location: location,
                    annotations: Vec::new(),
                    method_type,
                },
            )]),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    }));
}

/// Instance def carrying several `#:` overloads (the enum setter's
/// three-way accept surface).
fn push_instance_overloaded_def(
    members: &mut Vec<Member>,
    name: String,
    overloads: Vec<MethodType>,
    location: PrismByteRange,
) {
    let annotations = overloads
        .into_iter()
        .map(|method_type| {
            ExplicitAnnotation::Colon(ColonMethodTypeAnnotation {
                location,
                prefix_location: location,
                annotations: Vec::new(),
                method_type,
            })
        })
        .collect();
    members.push(Member::Def(DefMember {
        ivar_param_pairs: Vec::new(),
        name,
        kind: MethodKind::Instance,
        location,
        name_location: location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(annotations),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    }));
}

fn push_association_builders(
    members: &mut Vec<Member>,
    name: &str,
    location: PrismByteRange,
    name_location: PrismByteRange,
) {
    for method in [
        format!("build_{name}"),
        format!("create_{name}"),
        format!("create_{name}!"),
    ] {
        push_def(
            members,
            method,
            MethodKind::Instance,
            location,
            name_location,
        );
    }
}

fn keyword_bool(call: &InfusionCall, name: &str) -> Option<bool> {
    call.keyword_bools
        .iter()
        .rev()
        .find(|kw| kw.name == name)
        .and_then(|kw| kw.value)
}

fn keyword_string<'a>(call: &'a InfusionCall, name: &str) -> Option<&'a str> {
    call.keyword_strings
        .iter()
        .rev()
        .find(|kw| kw.name == name)
        .map(|kw| kw.value.as_str())
}

fn enum_value_method_name(enum_name: &str, value: &str, call: &InfusionCall) -> String {
    if let Some(prefix) = keyword_string(call, "prefix") {
        format!("{prefix}_{value}")
    } else if let Some(suffix) = keyword_string(call, "suffix") {
        format!("{value}_{suffix}")
    } else if keyword_bool(call, "prefix") == Some(true) {
        format!("{enum_name}_{value}")
    } else if keyword_bool(call, "suffix") == Some(true) {
        format!("{value}_{enum_name}")
    } else {
        value.to_string()
    }
}
