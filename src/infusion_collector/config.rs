use std::cell::RefCell;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_yaml::{Mapping, Number, Value};

use crate::ast::declarations::{ConstantDeclaration, InterfaceDeclaration, Member};
use crate::ast::members::{
    MethodDefinitionMember, MethodDefinitionOverload, MethodKind, Visibility,
};
use crate::ast::method_type::MethodType;
use crate::ast::types::{
    BaseType, BaseTypeKind, ClassInstanceType, Function, FunctionType, InterfaceType, Type,
    UnionType,
};
use crate::config::ConfigInfusionTable;
use crate::diagnostic::{Diagnostic, DiagnosticKind};
use crate::environment::draft::{Context, EnvironmentDraft};
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::type_name::TypeName;

#[derive(Debug)]
pub enum ConfigInfusionError {
    EmptyFiles,
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    RootNotMapping {
        path: PathBuf,
    },
    InvalidMerge {
        key: String,
    },
    InvalidConstName {
        name: String,
    },
    InvalidMethodName {
        key: String,
    },
    Build {
        message: String,
    },
}

impl fmt::Display for ConfigInfusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigInfusionError::EmptyFiles => {
                write!(f, "files must contain at least one YAML file")
            }
            ConfigInfusionError::Read { path, source } => {
                write!(f, "cannot read {}: {}", path.display(), source)
            }
            ConfigInfusionError::Parse { path, source } => {
                write!(f, "cannot parse {}: {}", path.display(), source)
            }
            ConfigInfusionError::RootNotMapping { path } => {
                write!(
                    f,
                    "{} must contain a YAML mapping at the root",
                    path.display()
                )
            }
            ConfigInfusionError::InvalidMerge { key } => {
                write!(f, "YAML merge key {:?} must point to mapping values", key)
            }
            ConfigInfusionError::InvalidConstName { name } => {
                write!(f, "const_name {:?} is not a valid constant path", name)
            }
            ConfigInfusionError::InvalidMethodName { key } => {
                write!(f, "YAML key {:?} is not a valid Ruby method name", key)
            }
            ConfigInfusionError::Build { message } => write!(f, "{}", message),
        }
    }
}

/// Load `[infusion.config]` YAML files into `draft`, returning warning
/// diagnostics for recoverable issues (currently duplicate mapping keys,
/// which are resolved last-wins à la Psych). Genuinely unrecoverable
/// problems (unreadable file, non-mapping root, invalid names) still
/// return `Err` and abort the infusion.
pub fn load(
    table: &ConfigInfusionTable,
    draft: &mut EnvironmentDraft,
) -> Result<Vec<Diagnostic>, ConfigInfusionError> {
    if table.files.is_empty() {
        return Err(ConfigInfusionError::EmptyFiles);
    }
    if !valid_const_path(&table.const_name) {
        return Err(ConfigInfusionError::InvalidConstName {
            name: table.const_name.clone(),
        });
    }

    let mut diagnostics = Vec::new();
    let mut merged = Mapping::new();
    for path in &table.files {
        let content =
            std::fs::read_to_string(path).map_err(|source| ConfigInfusionError::Read {
                path: path.clone(),
                source,
            })?;
        let (value, dup_keys) =
            parse_yaml_last_wins(&content).map_err(|source| ConfigInfusionError::Parse {
                path: path.clone(),
                source,
            })?;
        for key in dup_keys {
            diagnostics.push(Diagnostic {
                scope: None,
                kind: DiagnosticKind::DuplicatedConfigKey {
                    key,
                    file: path.display().to_string(),
                },
                location: Diagnostic::location_for_byte_range(
                    path.clone(),
                    content.as_bytes(),
                    0,
                    0,
                ),
            });
        }
        let Value::Mapping(mapping) = value else {
            return Err(ConfigInfusionError::RootNotMapping { path: path.clone() });
        };
        merge_mapping(&mut merged, expand_mapping_merges(mapping)?);
    }

    let root_constant_name = draft
        .names()
        .parse_type_name(&format!("::{}", table.const_name));
    let root_interface_name = interface_name_for_const(draft, &table.const_name);
    let mut interfaces = Vec::new();
    let root_members =
        synthesize_members(table, draft, root_interface_name, &merged, &mut interfaces)?;
    interfaces.insert(0, interface_decl(root_interface_name, root_members));

    let context: Context = Arc::from([]);
    let root_constant = ConstantDeclaration {
        name: root_constant_name,
        ty: interface_type(root_interface_name),
        annotations: Vec::new(),
        location: None,
        comment: None,
    };
    draft
        .insert_constant(
            DeclOrigin::Synthesized(InfusionUnit::Config, None),
            Arc::clone(&context),
            Arc::new(root_constant),
        )
        .map_err(|err| ConfigInfusionError::Build {
            message: err.format_with(draft.names()),
        })?;
    for interface in interfaces {
        draft
            .insert_interface_decl(
                DeclOrigin::Synthesized(InfusionUnit::Config, None),
                Arc::clone(&context),
                Arc::new(interface),
            )
            .map_err(|err| ConfigInfusionError::Build {
                message: err.format_with(draft.names()),
            })?;
    }
    Ok(diagnostics)
}

/// Parse a single-document YAML string into a `Value`, tolerating duplicate
/// mapping keys instead of erroring like `serde_yaml`'s stock `Value`
/// deserializer. Duplicates resolve last-wins (Psych semantics); every
/// duplicated key name, at any nesting depth, is collected into the returned
/// `Vec`. Only genuine parse errors surface as `Err`.
fn parse_yaml_last_wins(content: &str) -> Result<(Value, Vec<String>), serde_yaml::Error> {
    let dups = RefCell::new(Vec::new());
    let mut docs = serde_yaml::Deserializer::from_str(content);
    let value = match docs.next() {
        Some(doc) => TolerantSeed { dups: &dups }.deserialize(doc)?,
        None => Value::Null,
    };
    Ok((value, dups.into_inner()))
}

/// Recursively deserializes any YAML value while sharing a duplicate-key
/// collector, so nested mappings are checked too. `next_value::<Value>()`
/// would route nested maps back through `serde_yaml`'s stock visitor and
/// reject their duplicates before we could see them — hence the seed.
struct TolerantSeed<'a> {
    dups: &'a RefCell<Vec<String>>,
}

impl<'de> DeserializeSeed<'de> for TolerantSeed<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(TolerantVisitor { dups: self.dups })
    }
}

struct TolerantVisitor<'a> {
    dups: &'a RefCell<Vec<String>>,
}

impl<'de> Visitor<'de> for TolerantVisitor<'_> {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any YAML value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(Number::from(v)))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(Number::from(v)))
    }
    fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
        Ok(Value::Number(Number::from(v)))
    }
    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }
    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }
    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        TolerantSeed { dups: self.dups }.deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(TolerantSeed { dups: self.dups })? {
            items.push(item);
        }
        Ok(Value::Sequence(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut mapping = Mapping::new();
        while let Some(key) = map.next_key::<Value>()? {
            let value = map.next_value_seed(TolerantSeed { dups: self.dups })?;
            if mapping.contains_key(&key) {
                // A `<<` merge key is not an ordinary config key: Psych applies
                // *every* `<<` in a mapping, later ones taking precedence over
                // earlier ones on overlapping keys (verified against Psych
                // `aliases: true`). Normalize repeated `<<` into the same
                // `Value::Sequence` shape `expand_mapping_merges` /
                // `merge_yaml_merge_value` already handle for `<<: [a, b]`,
                // prepending so the later occurrence keeps its precedence
                // (that downstream sequence merge is leftmost-wins).
                if key.as_str() == Some("<<") {
                    let existing = mapping.remove(&key).unwrap();
                    mapping.insert(key, combine_merge_values(value, existing));
                    continue;
                }
                self.dups.borrow_mut().push(display_key(&key));
            }
            mapping.insert(key, value);
        }
        Ok(Value::Mapping(mapping))
    }
}

/// Flattens a `<<` value into its constituent merge sources, so repeated `<<`
/// keys and explicit `<<: [a, b]` sequences normalize to the same shape.
fn merge_key_items(value: Value) -> Vec<Value> {
    match value {
        Value::Sequence(items) => items,
        other => vec![other],
    }
}

fn combine_merge_values(new: Value, existing: Value) -> Value {
    let mut items = merge_key_items(new);
    items.extend(merge_key_items(existing));
    Value::Sequence(items)
}

/// Duplicate keys only ever surface a diagnostic for string keys: a
/// non-string mapping key fails `synthesize_members` with `InvalidMethodName`
/// before `load` returns, so the `Debug` fallback here is never user-visible.
fn display_key(key: &Value) -> String {
    key.as_str()
        .map(|s| s.to_owned())
        .unwrap_or_else(|| format!("{:?}", key))
}

fn merge_mapping(base: &mut Mapping, next: Mapping) {
    for (key, value) in next {
        match (base.get_mut(&key), value) {
            (Some(Value::Mapping(base_mapping)), Value::Mapping(next_mapping)) => {
                merge_mapping(base_mapping, next_mapping);
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

fn expand_value_merges(value: Value) -> Result<Value, ConfigInfusionError> {
    match value {
        Value::Mapping(mapping) => Ok(Value::Mapping(expand_mapping_merges(mapping)?)),
        Value::Sequence(values) => values
            .into_iter()
            .map(expand_value_merges)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Sequence),
        value => Ok(value),
    }
}

fn expand_mapping_merges(mapping: Mapping) -> Result<Mapping, ConfigInfusionError> {
    let mut expanded = Mapping::new();
    let mut explicit = Vec::new();
    for (key, value) in mapping {
        if key.as_str() == Some("<<") {
            merge_yaml_merge_value(&mut expanded, value)?;
        } else {
            explicit.push((key, value));
        }
    }
    for (key, value) in explicit {
        expanded.insert(key, expand_value_merges(value)?);
    }
    Ok(expanded)
}

fn merge_yaml_merge_value(base: &mut Mapping, value: Value) -> Result<(), ConfigInfusionError> {
    match value {
        Value::Mapping(mapping) => {
            merge_mapping(base, expand_mapping_merges(mapping)?);
            Ok(())
        }
        Value::Sequence(values) => {
            for value in values.into_iter().rev() {
                let Value::Mapping(mapping) = value else {
                    return Err(ConfigInfusionError::InvalidMerge {
                        key: "<<".to_string(),
                    });
                };
                merge_mapping(base, expand_mapping_merges(mapping)?);
            }
            Ok(())
        }
        _ => Err(ConfigInfusionError::InvalidMerge {
            key: "<<".to_string(),
        }),
    }
}

fn synthesize_members(
    table: &ConfigInfusionTable,
    draft: &EnvironmentDraft,
    owner: TypeName,
    mapping: &Mapping,
    interfaces: &mut Vec<InterfaceDeclaration>,
) -> Result<Vec<Member>, ConfigInfusionError> {
    let mut members = Vec::new();
    for (key, value) in mapping {
        let Some(key) = key.as_str() else {
            return Err(ConfigInfusionError::InvalidMethodName {
                key: format!("{:?}", key),
            });
        };
        if table.except_keys.iter().any(|except| except == key) {
            continue;
        }
        if !valid_method_name(key) {
            return Err(ConfigInfusionError::InvalidMethodName {
                key: key.to_string(),
            });
        }
        let ty = match value {
            Value::Mapping(nested) => {
                let interface_name = nested_interface_name(draft, owner, key)?;
                let nested_members =
                    synthesize_members(table, draft, interface_name, nested, interfaces)?;
                interfaces.push(interface_decl(interface_name, nested_members));
                interface_type(interface_name)
            }
            _ => value_type(draft, value),
        };
        members.push(method_returning(draft, key, MethodKind::Instance, ty));
    }
    Ok(members)
}

fn interface_decl(name: TypeName, members: Vec<Member>) -> InterfaceDeclaration {
    InterfaceDeclaration {
        name,
        type_params: Vec::new(),
        members,
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    }
}

fn method_returning(
    draft: &EnvironmentDraft,
    name: &str,
    kind: MethodKind,
    return_type: Type,
) -> Member {
    Member::MethodDefinition(MethodDefinitionMember {
        name: draft.names().intern_symbol(name),
        kind,
        overloads: vec![MethodDefinitionOverload {
            method_type: MethodType {
                type_params: Vec::new(),
                function: Function::Typed(FunctionType {
                    required_positionals: Vec::new(),
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
            },
            annotations: Vec::new(),
        }],
        annotations: Vec::new(),
        overloading: false,
        visibility: Some(Visibility::Public),
        location: None,
        source_file: None,
        comment: None,
    })
}

fn value_type(draft: &EnvironmentDraft, value: &Value) -> Type {
    match value {
        Value::Bool(_) => base(BaseTypeKind::Bool),
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            class_instance(draft.names().parse_type_name("::Integer"))
        }
        Value::Number(_) => class_instance(draft.names().parse_type_name("::Float")),
        Value::String(_) => class_instance(draft.names().parse_type_name("::String")),
        Value::Sequence(values) => array_type(draft, values),
        Value::Null => base(BaseTypeKind::Nil),
        _ => untyped_type(),
    }
}

fn array_type(draft: &EnvironmentDraft, values: &[Value]) -> Type {
    let element = if values.is_empty() {
        untyped_type()
    } else {
        union_type(
            values
                .iter()
                .map(|value| value_type(draft, value))
                .collect(),
        )
    };
    Type::ClassInstance(ClassInstanceType {
        name: draft.names().parse_type_name("::Array"),
        args: vec![element],
        location: None,
    })
}

fn union_type(mut types: Vec<Type>) -> Type {
    types.sort_by_key(|ty| format!("{:?}", ty));
    types.dedup();
    if types.len() == 1 {
        types.pop().unwrap()
    } else {
        Type::Union(UnionType {
            types,
            location: None,
        })
    }
}

fn class_instance(name: TypeName) -> Type {
    Type::ClassInstance(ClassInstanceType {
        name,
        args: Vec::new(),
        location: None,
    })
}

fn interface_type(name: TypeName) -> Type {
    Type::Interface(InterfaceType {
        name,
        args: Vec::new(),
        location: None,
    })
}

fn base(kind: BaseTypeKind) -> Type {
    Type::Base(BaseType {
        kind,
        location: None,
    })
}

fn untyped_type() -> Type {
    base(BaseTypeKind::Any { todo: false })
}

fn interface_name_for_const(draft: &EnvironmentDraft, const_name: &str) -> TypeName {
    let segment = draft
        .names()
        .intern_symbol(&format!("_CremaConfig{}", const_name));
    draft
        .names()
        .append_type_name(draft.names().absolute_root(), segment)
}

fn nested_interface_name(
    draft: &EnvironmentDraft,
    owner: TypeName,
    key: &str,
) -> Result<TypeName, ConfigInfusionError> {
    let segment = camelize(key);
    if !valid_const_segment(&segment) {
        return Err(ConfigInfusionError::InvalidMethodName {
            key: key.to_string(),
        });
    }
    let symbol = draft.names().intern_symbol(&segment);
    Ok(draft.names().append_type_name(owner, symbol))
}

fn camelize(key: &str) -> String {
    let mut out = String::new();
    for part in key.split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

fn valid_const_path(name: &str) -> bool {
    !name.contains("::") && valid_const_segment(name)
}

fn valid_const_segment(segment: &str) -> bool {
    let mut chars = segment.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_uppercase())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn valid_method_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(ch) if ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '?' || ch == '!' || ch.is_ascii_alphanumeric())
        && name
            .chars()
            .enumerate()
            .all(|(index, ch)| (ch != '?' && ch != '!') || index == name.len() - 1)
}
