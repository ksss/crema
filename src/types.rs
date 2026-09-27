use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::{Cell, UnsafeCell};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;
use crate::type_param::TypeVarScope;

/// An interned type, represented as an index into a `TypeTable`.
/// Copy-cheap, comparison-cheap (u32 == u32). The Ord / PartialOrd
/// derives expose the raw intern id ordering — semantically meaningless
/// but enough for callers that need a deterministic Vec<Ty> order (e.g.
/// CondEnv::join_branch deduping members for interning).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ty(u32);

impl Ty {
    // Well-known type constants. Order must match TypeTable::new() interning order.
    pub const VOID: Ty = Ty(0);
    pub const NIL: Ty = Ty(1);
    pub const BOOL: Ty = Ty(2);
    pub const UNTYPED: Ty = Ty(3);
    pub const TOP: Ty = Ty(4);
    pub const BOTTOM: Ty = Ty(5);
    pub const SELF_TYPE: Ty = Ty(6);
    pub const INSTANCE_TYPE: Ty = Ty(7);
    pub const CLASS_TYPE: Ty = Ty(8);

    pub fn is_untyped(self) -> bool {
        self == Ty::UNTYPED
    }

    /// Format the type as a human-readable string using the type and name tables.
    pub fn display(self, types: &TypeTable, names: &NameTable) -> String {
        match types.resolve(self) {
            Type::ClassInstance { name, args } => {
                // Internal repr keeps TrueClass/FalseClass distinct for
                // method lookup; display folds to `bool`.
                let builtins = names.builtins();
                if args.is_empty() && builtins.is_bool_class(*name) {
                    return "bool".to_string();
                }
                let name_str = names.resolve(name);
                if args.is_empty() {
                    name_str
                } else {
                    let args_str: Vec<String> =
                        args.iter().map(|a| a.display(types, names)).collect();
                    format!("{}[{}]", name_str, args_str.join(", "))
                }
            }
            Type::ClassSingleton { name } => {
                let name_str = names.resolve(name);
                format!("singleton({})", name_str)
            }
            Type::Union(members) => {
                // Drop only the second `bool` so the bool fold does not
                // surface as `bool | bool`; other display duplicates stay
                // visible as construction-layer signals.
                let mut parts: Vec<String> = Vec::with_capacity(members.len());
                let mut bool_emitted = false;
                for m in members.iter() {
                    let s = m.display(types, names);
                    if s == "bool" {
                        if bool_emitted {
                            continue;
                        }
                        bool_emitted = true;
                    }
                    parts.push(s);
                }
                // Canonicalize by rendered string so the same union set
                // renders identically across intern orders (the a-snapshot
                // warm/cold divergence root cause). `union_of` sorts by
                // Ty(u32) which is insertion-order dependent — cold fresh
                // build and warm snapshot decode intern types in different
                // orders, so the same union collapses to different member
                // sequences at the construction layer. The display sort
                // absorbs that.
                parts.sort();
                parts.join(" | ")
            }
            Type::Intersection(members) => {
                let mut parts: Vec<String> =
                    members.iter().map(|m| m.display(types, names)).collect();
                parts.sort();
                parts.join(" & ")
            }
            Type::Optional(inner) => format!("{}?", inner.display(types, names)),
            Type::Void => "void".to_string(),
            Type::Nil => "nil".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Untyped => "untyped".to_string(),
            Type::Top => "top".to_string(),
            Type::Bottom => "bot".to_string(),
            Type::SelfType => "self".to_string(),
            Type::InstanceType => "instance".to_string(),
            Type::ClassType => "class".to_string(),
            Type::Literal(lit) => match lit {
                Literal::Integer(n) => n.clone(),
                Literal::String(s) => format!("\"{}\"", s),
                Literal::Symbol(s) => format!(":{}", s),
                Literal::Bool(b) => b.to_string(),
            },
            Type::TypeVariable { raw, .. } => names.resolve(*raw),
            Type::Interface { name, args } => {
                let name_str = names.resolve(name);
                if args.is_empty() {
                    name_str
                } else {
                    let args_str: Vec<String> =
                        args.iter().map(|a| a.display(types, names)).collect();
                    format!("{}[{}]", name_str, args_str.join(", "))
                }
            }
            Type::Alias { name, args } => {
                let name_str = names.resolve(name);
                let short = name_str.strip_prefix("::").unwrap_or(&name_str);
                if args.is_empty() {
                    short.to_string()
                } else {
                    let args_str: Vec<String> =
                        args.iter().map(|a| a.display(types, names)).collect();
                    format!("{}[{}]", short, args_str.join(", "))
                }
            }
            Type::Tuple(members) => {
                let parts: Vec<String> = members.iter().map(|m| m.display(types, names)).collect();
                format!("[{}]", parts.join(", "))
            }
            Type::Record { fields } => {
                if fields.is_empty() {
                    "{}".to_string()
                } else {
                    let parts: Vec<String> = fields
                        .iter()
                        .map(|(key, ty, required)| {
                            let value = ty.display(types, names);
                            match key {
                                RecordKey::Symbol(s) => {
                                    let prefix = if *required { "" } else { "?" };
                                    format!("{}{}: {}", prefix, s, value)
                                }
                                RecordKey::String(s) => {
                                    format!("\"{}\" => {}", s, value)
                                }
                                RecordKey::Integer(i) => {
                                    format!("{} => {}", i, value)
                                }
                                RecordKey::Bool(b) => {
                                    format!("{} => {}", b, value)
                                }
                            }
                        })
                        .collect();
                    format!("{{ {} }}", parts.join(", "))
                }
            }
            Type::Proc {
                type_,
                self_type,
                block,
            } => {
                let mut out = String::from("^");
                out.push_str(&format_function_type(type_, types, names));
                append_self_type_binding(&mut out, *self_type, types, names);
                if let Some(b) = &block {
                    out.push(' ');
                    if !b.required {
                        out.push('?');
                    }
                    out.push_str("{ ");
                    out.push_str(&format_function_type(&b.type_, types, names));
                    append_self_type_binding(&mut out, b.self_type, types, names);
                    out.push_str(&format!(" -> {} }}", b.return_type().display(types, names)));
                }
                out.push_str(&format!(
                    " -> {}",
                    type_.return_type().display(types, names)
                ));
                out
            }
        }
    }
}

fn append_self_type_binding(
    out: &mut String,
    self_type: Option<Ty>,
    types: &TypeTable,
    names: &NameTable,
) {
    if let Some(st) = self_type {
        out.push_str(&format!(" [self: {}]", st.display(types, names)));
    }
}

/// Format a `FunctionType` parameter list into the RBS `(...)` form.
/// Does not include the return type or the leading `^` — just the parenthesized
/// parameter list (including `(?)` for untyped functions).
fn format_function_type(ft: &FunctionType, types: &TypeTable, names: &NameTable) -> String {
    match ft {
        FunctionType::Untyped(_) => "(?)".to_string(),
        FunctionType::Typed(f) => {
            let mut parts: Vec<String> = Vec::new();
            for t in &f.required_positionals {
                parts.push(t.display(types, names));
            }
            for t in &f.optional_positionals {
                parts.push(format!("?{}", t.display(types, names)));
            }
            if let Some(t) = f.rest_positional {
                parts.push(format!("*{}", t.display(types, names)));
            }
            for t in &f.trailing_positionals {
                parts.push(t.display(types, names));
            }
            for (name, t) in &f.required_keywords {
                parts.push(format!("{}: {}", name, t.display(types, names)));
            }
            for (name, t) in &f.optional_keywords {
                parts.push(format!("?{}: {}", name, t.display(types, names)));
            }
            if let Some(t) = f.rest_keyword {
                parts.push(format!("**{}", t.display(types, names)));
            }
            format!("({})", parts.join(", "))
        }
    }
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ty({})", self.0)
    }
}

/// Internal type representation for type checking.
/// Stored inside a `TypeTable`; external code uses `Ty` handles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(clippy::large_enum_variant)]
pub enum Type {
    /// A class instance type, e.g. `Integer`, `Array[String]`.
    /// Mirrors `RBS::Types::ClassInstance`.
    ClassInstance { name: TypeName, args: Vec<Ty> },
    /// A class singleton type, e.g. `singleton(Foo)`.
    /// Mirrors `RBS::Types::ClassSingleton`.
    ///
    /// Represents the class object itself, not its instances.
    /// Singleton types have no generic arguments in RBS semantics
    /// (generics apply to instances); Sorbet's `singleton(X)[T]` form is
    /// accepted by the RBS parser but its type arguments are ignored here.
    ClassSingleton { name: TypeName },
    /// A union type, e.g. `Integer | String`. Mirrors `RBS::Types::Union`.
    Union(Vec<Ty>),
    /// An intersection type, e.g. `_Reader & _Writer`.
    /// Mirrors `RBS::Types::Intersection`. Dual of `Union`: a value has
    /// **all** listed types simultaneously.
    ///
    /// Order is preserved for interning stability; structurally identical
    /// vectors intern to the same `Ty`. No normalization is performed —
    /// uninhabited intersections like `Integer & String` are kept as-is and
    /// are naturally unreachable under the subtyping rules.
    Intersection(Vec<Ty>),
    /// An optional type, e.g. `Integer?` (sugar for `Integer | nil`).
    /// Mirrors `RBS::Types::Optional`.
    Optional(Ty),
    /// Mirrors `RBS::Types::Bases::Void`.
    Void,
    /// Mirrors `RBS::Types::Bases::Nil`.
    Nil,
    /// Mirrors `RBS::Types::Bases::Bool`.
    Bool,
    /// Gradual typing escape hatch — compatible with everything.
    /// Mirrors `RBS::Types::Bases::Any`; rbs's user-visible keyword is
    /// `untyped` (with `any` accepted as a deprecated synonym), so the
    /// variant is named after the keyword rather than the rbs class name.
    Untyped,
    /// Mirrors `RBS::Types::Bases::Top`.
    Top,
    /// Mirrors `RBS::Types::Bases::Bottom`.
    Bottom,
    /// Mirrors `RBS::Types::Bases::Self`. Suffixed `Type` because `Self` is
    /// a Rust keyword; the build-layer port (`crate::ast::types::Type`)
    /// spells the same variant `Self_`.
    SelfType,
    /// The `instance` base type. Mirrors `RBS::Types::Bases::Instance`.
    /// Suffixed `Type` to disambiguate from `Type::ClassInstance`; the
    /// build-layer port spells it `Instance`.
    /// Always refers to the instance type of the enclosing class regardless
    /// of method kind (instance vs singleton). Substituted at call sites via
    /// `crate::substitution::Substitution::instance_type`.
    InstanceType,
    /// The `class` base type. Mirrors `RBS::Types::Bases::Class`.
    /// Suffixed `Type` for parity with `InstanceType`; the build-layer port
    /// spells it `Class`.
    /// Always refers to the singleton type of the enclosing class regardless
    /// of method kind. Substituted at call sites via `crate::substitution::Substitution::class_type`.
    ClassType,
    /// A literal type, e.g. `1`, `"hello"`, `:sym`. Mirrors `RBS::Types::Literal`.
    Literal(Literal),
    /// A type variable from a generic class, e.g. `Elem` in `class Array[Elem]`.
    /// Mirrors `RBS::Types::Variable`. Renamed `TypeVariable` in both crema
    /// layers to keep "variable" available for unrelated concepts (locals,
    /// type params).
    ///
    /// Per ADR-0023, the scope identifying the declaration site is carried
    /// as a struct field instead of encoded into the raw `Symbol`. The pair
    /// is the full identity of a type variable: two `TypeVariable`s compare
    /// equal iff they came from the same declaration site.
    TypeVariable { raw: Symbol, scope: TypeVarScope },
    /// An interface type, e.g. `_ToStr`, `_Each[String, void]`.
    /// Mirrors `RBS::Types::Interface`.
    Interface { name: TypeName, args: Vec<Ty> },
    /// A type alias reference, e.g. `int`, `array[String]`.
    /// Mirrors `RBS::Types::Alias`. Preserved lazily; expanded on demand
    /// during subtyping.
    Alias { name: TypeName, args: Vec<Ty> },
    /// A Proc type, e.g. `^(Integer) -> String`.
    /// Mirrors `RBS::Types::Proc`. The `type` field on rbs becomes `type_`
    /// here to escape the Rust keyword; the build-layer port escapes the
    /// same field as `function` instead (see ADR-0014 layer-drift notes).
    ///
    /// `self_type` is the `[self: T]` binding when present.
    /// `block` is a block argument attached to the proc (RBS allows procs to
    /// declare blocks, though it is rarely used in practice).
    Proc {
        type_: FunctionType,
        self_type: Option<Ty>,
        block: Option<Block>,
    },
    /// A Tuple type, e.g. `[Integer, String]`.
    /// Mirrors `RBS::Types::Tuple`. Fixed-length heterogeneous array.
    /// Element count is part of the type identity: `[A]` and `[A, A]` are distinct.
    Tuple(Vec<Ty>),
    /// A Record type, e.g. `{ name: String, ?email: String, "tag" => Symbol }`.
    /// Mirrors `RBS::Types::Record`. Fixed set of keyed fields; each field
    /// is required or optional.
    ///
    /// Keys may be Symbol / String / Integer / Bool literals (see `RecordKey`).
    /// Only Symbol keys can be optional — the RBS parser rejects `?"foo" =>`
    /// and `?3 =>` syntactically, so non-Symbol keys always have `required = true`.
    ///
    /// Invariant: `fields` is sorted ascending by `RecordKey` so that
    /// structurally equal records intern to the same `Ty`.
    Record { fields: Vec<(RecordKey, Ty, bool)> },
}

/// A key used inside a `Type::Record`. crema-only fold: rbs stores
/// `RBS::Types::Record` keys as plain Ruby literal values; this enum
/// enumerates the same domain at the Rust type level.
///
/// RBS record keys may be any literal: Symbol (`foo:` shortcut or `:foo =>`),
/// String (`"foo" =>` / `'foo' =>`), Integer (`3 =>`), or Bool (`true =>`).
/// `Symbol(s)` and `String(s)` are distinct keys even for the same inner
/// string — the RBS parser rejects a record that mixes them, but as types
/// they represent different runtime identities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordKey {
    Symbol(String),
    String(String),
    Integer(String),
    Bool(bool),
}

/// Discriminant used to order `RecordKey` variants against each other —
/// mirrors the derive-macro convention of ordering by declaration order.
fn record_key_variant_rank(key: &RecordKey) -> u8 {
    match key {
        RecordKey::Symbol(_) => 0,
        RecordKey::String(_) => 1,
        RecordKey::Integer(_) => 2,
        RecordKey::Bool(_) => 3,
    }
}

/// Numeric ordering of two canonical decimal strings (optional `-` sign, no
/// leading zeros, no upper bound). A derived `Ord` would compare the raw
/// strings lexicographically — correct for a fixed-width type, but wrong
/// once the same-length invariant of `i64` is gone (`"10" < "2"` as
/// strings). Same-sign magnitudes of equal length can't differ in leading
/// zeros (canonical form), so length-then-lexicographic order equals
/// numeric order.
fn decimal_str_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (a_neg, a_mag) = match a.strip_prefix('-') {
        Some(mag) => (true, mag),
        None => (false, a),
    };
    let (b_neg, b_mag) = match b.strip_prefix('-') {
        Some(mag) => (true, mag),
        None => (false, b),
    };
    match (a_neg, b_neg) {
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        (false, false) => a_mag.len().cmp(&b_mag.len()).then_with(|| a_mag.cmp(b_mag)),
        (true, true) => a_mag
            .len()
            .cmp(&b_mag.len())
            .then_with(|| a_mag.cmp(b_mag))
            .reverse(),
    }
}

impl PartialOrd for RecordKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecordKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (RecordKey::Symbol(a), RecordKey::Symbol(b)) => a.cmp(b),
            (RecordKey::String(a), RecordKey::String(b)) => a.cmp(b),
            (RecordKey::Integer(a), RecordKey::Integer(b)) => decimal_str_cmp(a, b),
            (RecordKey::Bool(a), RecordKey::Bool(b)) => a.cmp(b),
            _ => record_key_variant_rank(self).cmp(&record_key_variant_rank(other)),
        }
    }
}

impl RecordKey {
    /// The key rendered as it would appear in a diagnostic message —
    /// standalone, no field-context prefix (`:` / `=>`) or `?` marker.
    /// Examples: `:foo`, `"foo"`, `3`, `true`.
    pub fn display(&self) -> String {
        match self {
            RecordKey::Symbol(s) => format!(":{}", s),
            RecordKey::String(s) => format!("\"{}\"", s),
            RecordKey::Integer(i) => i.clone(),
            RecordKey::Bool(b) => b.to_string(),
        }
    }

    /// The fully-qualified class that a value of this key kind belongs to at
    /// runtime. Used by record→hash widening to build the key-type union.
    pub fn widen_class_name(&self) -> &'static str {
        match self {
            RecordKey::Symbol(_) => "::Symbol",
            RecordKey::String(_) => "::String",
            RecordKey::Integer(_) => "::Integer",
            RecordKey::Bool(true) => "::TrueClass",
            RecordKey::Bool(false) => "::FalseClass",
        }
    }
}

/// The literal value carried by a `Type::Literal`. crema-only enum that
/// enumerates the cases rbs stores as a plain Ruby value in
/// `RBS::Types::Literal#literal` (Symbol / String / Integer / TrueClass /
/// FalseClass).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Integer(String),
    String(String),
    Symbol(String),
    Bool(bool),
}

impl Literal {
    /// The pre-interned builtin [`TypeName`] this literal widens to.
    /// Single source of truth for the `Literal` variant → builtin class
    /// mapping inside crema; not an rbs port (`RBS::Types::Literal` has no
    /// equivalent helper).
    pub fn class_typename<'a>(&self, builtins: &'a crate::name::BuiltinNames) -> &'a TypeName {
        match self {
            Literal::Integer(_) => &builtins.integer,
            Literal::String(_) => &builtins.string,
            Literal::Symbol(_) => &builtins.symbol,
            Literal::Bool(true) => &builtins.true_class,
            Literal::Bool(false) => &builtins.false_class,
        }
    }
}

/// Fully typed function: all parameter fields + return_type.
/// Mirrors `RBS::Types::Function`.
///
/// rbs wraps each parameter in a `RBS::Types::Function::Param` struct that
/// carries `type`, `name`, and `location`. crema currently flattens parameters
/// to bare `Ty` values and discards the parameter name; ADR-0014 reserves
/// restoring a `Function::Param` shape as a later phase.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub required_positionals: Vec<Ty>,
    pub optional_positionals: Vec<Ty>,
    pub rest_positional: Option<Ty>,
    pub trailing_positionals: Vec<Ty>,
    pub required_keywords: Vec<(String, Ty)>,
    pub optional_keywords: Vec<(String, Ty)>,
    pub rest_keyword: Option<Ty>,
    pub return_type: Ty,
}

impl Function {
    /// Create a `Function` with no parameters.
    pub fn empty(return_type: Ty) -> Self {
        Function {
            required_positionals: vec![],
            optional_positionals: vec![],
            rest_positional: None,
            trailing_positionals: vec![],
            required_keywords: vec![],
            optional_keywords: vec![],
            rest_keyword: None,
            return_type,
        }
    }

    pub fn min_arity(&self) -> usize {
        self.required_positionals.len() + self.trailing_positionals.len()
    }

    pub fn max_arity(&self) -> Option<usize> {
        if self.rest_positional.is_some() {
            None
        } else {
            Some(self.min_arity() + self.optional_positionals.len())
        }
    }

    pub fn arity_accepts(&self, count: usize) -> bool {
        if count < self.min_arity() {
            return false;
        }
        match self.max_arity() {
            Some(max) => count <= max,
            None => true,
        }
    }

    /// Resolve the parameter type expected at call-site positional `index`,
    /// given total `arg_count`. Trailing positionals are anchored to the END
    /// of the actual argument list (Ruby semantics): the last `t` args bind
    /// to trailing slots from-the-end; the middle is filled by optional then
    /// rest. Caller must ensure `arity_accepts(arg_count)` first.
    pub fn positional_param_for_call(&self, index: usize, arg_count: usize) -> Option<Ty> {
        let r = self.required_positionals.len();
        let t = self.trailing_positionals.len();
        let o = self.optional_positionals.len();

        if index < r {
            return Some(self.required_positionals[index]);
        }
        if index >= arg_count.saturating_sub(t) {
            return Some(self.trailing_positionals[index - (arg_count - t)]);
        }
        let middle_idx = index - r;
        if middle_idx < o {
            return Some(self.optional_positionals[middle_idx]);
        }
        self.rest_positional
    }

    /// Merge two function signatures for overload unification. Ports Steep's
    /// `PositionalParams.merge_for_overload` + `KeywordParams#+` semantics so
    /// a body type-checker can use a single composite signature in place of
    /// the per-overload list.
    ///
    /// Positionals walk both sides head-to-head with the 16-case lattice
    /// (Required/Optional/Rest/none x same). Keywords are merged by name
    /// (required only if both sides require). `rest_keyword` is unioned when
    /// both sides have one; `return_type` is always unioned.
    ///
    /// `trailing_positionals` have no Steep reference behavior (Steep's
    /// `ast/types/factory.rb:265` omits RBS trailing slots when lowering to
    /// its `Interface::Function::Params`; the axis does not exist there).
    /// crema models trailing faithfully to rbs, so the merge preserves the
    /// slots with a pairwise type union when both sides declare the same
    /// trailing count — otherwise the merged trailing is empty, which only
    /// loosens the composite (an end-anchored required slot melts into the
    /// rest/optional region) and never invents a caller shape the overloads
    /// don't admit. The body-context consumer (`bind_parameters`) binds
    /// Ruby's post-rest `tail` from the preserved slot, mirroring the
    /// single-overload path.
    ///
    /// rbs parser rejects same-name keyword duplicates within a single
    /// signature, so `required_keywords` / `optional_keywords` are treated
    /// as `name`-unique maps; only the first match per name is consulted.
    pub fn merge_for_overload(&self, other: &Function, types: &TypeTable) -> Function {
        let (req, opt, rest) = merge_positionals(self, other, types);

        let (req_kw, opt_kw, rest_kw) = merge_keywords(self, other, types);

        let trailing = if self.trailing_positionals.len() == other.trailing_positionals.len() {
            self.trailing_positionals
                .iter()
                .zip(&other.trailing_positionals)
                .map(|(x, y)| union_of(*x, *y, types))
                .collect()
        } else {
            vec![]
        };

        Function {
            required_positionals: req,
            optional_positionals: opt,
            rest_positional: rest,
            trailing_positionals: trailing,
            required_keywords: req_kw,
            optional_keywords: opt_kw,
            rest_keyword: rest_kw,
            return_type: union_of(self.return_type, other.return_type, types),
        }
    }
}

/// Position kind during overload positional merge. Mirrors Steep's
/// `PositionalParams::{Required, Optional, Rest}` head shapes.
#[derive(Clone, Copy)]
enum PosKind {
    Required,
    Optional,
    Rest,
}

/// Presence of a keyword on one side of the keyword merge. Keywords have no
/// `Rest` axis (the `**rest` slot is a separate field on `Function`), so
/// keeping this distinct from `PosKind` lets the merge `match` stay
/// exhaustive without `unreachable!` arms.
#[derive(Clone, Copy)]
enum KeywordPresence {
    Required,
    Optional,
}

/// Iterator-like state over a `Function`'s positionals. Yields `(PosKind, Ty)`
/// items in declaration order. `Rest` is yielded once and persists (does not
/// advance the cursor) so callers can pair `Rest` against successive items on
/// the other side, matching Steep's recursive head/tail walk.
struct PosCursor<'a> {
    func: &'a Function,
    idx: usize,
}

impl<'a> PosCursor<'a> {
    fn new(func: &'a Function) -> Self {
        PosCursor { func, idx: 0 }
    }

    fn peek(&self) -> Option<(PosKind, Ty)> {
        let r = self.func.required_positionals.len();
        let o = self.func.optional_positionals.len();
        if self.idx < r {
            Some((PosKind::Required, self.func.required_positionals[self.idx]))
        } else if self.idx < r + o {
            Some((
                PosKind::Optional,
                self.func.optional_positionals[self.idx - r],
            ))
        } else if self.idx == r + o {
            self.func.rest_positional.map(|t| (PosKind::Rest, t))
        } else {
            None
        }
    }

    /// Advance past the current item. For `Rest` this moves the cursor past
    /// the rest slot so a subsequent `peek` returns `None`. The caller decides
    /// when to call this based on the 16-case lattice (Steep keeps `Rest`
    /// against multiple opponents before consuming it).
    fn advance(&mut self) {
        self.idx += 1;
    }
}

fn merge_positionals(
    a: &Function,
    b: &Function,
    types: &TypeTable,
) -> (Vec<Ty>, Vec<Ty>, Option<Ty>) {
    let mut req: Vec<Ty> = vec![];
    let mut opt: Vec<Ty> = vec![];
    let mut rest: Option<Ty> = None;

    let mut ca = PosCursor::new(a);
    let mut cb = PosCursor::new(b);

    let union_with_nil = |t1: Ty, t2: Ty, types: &TypeTable| {
        let u = union_of(t1, t2, types);
        union_of(u, Ty::NIL, types)
    };

    loop {
        match (ca.peek(), cb.peek()) {
            (Some((PosKind::Required, x)), Some((PosKind::Required, y))) => {
                req.push(union_of(x, y, types));
                ca.advance();
                cb.advance();
            }
            (Some((PosKind::Required, x)), Some((PosKind::Optional, y))) => {
                opt.push(union_with_nil(x, y, types));
                ca.advance();
                cb.advance();
            }
            (Some((PosKind::Optional, x)), Some((PosKind::Required, y))) => {
                opt.push(union_with_nil(x, y, types));
                ca.advance();
                cb.advance();
            }
            (Some((PosKind::Required, x)), Some((PosKind::Rest, y))) => {
                opt.push(union_with_nil(x, y, types));
                ca.advance();
            }
            (Some((PosKind::Rest, x)), Some((PosKind::Required, y))) => {
                opt.push(union_with_nil(x, y, types));
                cb.advance();
            }
            (Some((PosKind::Required, x)), None) => {
                opt.push(union_of(x, Ty::NIL, types));
                ca.advance();
            }
            (None, Some((PosKind::Required, y))) => {
                opt.push(union_of(y, Ty::NIL, types));
                cb.advance();
            }
            (Some((PosKind::Optional, x)), Some((PosKind::Optional, y))) => {
                opt.push(union_of(x, y, types));
                ca.advance();
                cb.advance();
            }
            (Some((PosKind::Optional, x)), Some((PosKind::Rest, y))) => {
                opt.push(union_of(x, y, types));
                ca.advance();
            }
            (Some((PosKind::Rest, x)), Some((PosKind::Optional, y))) => {
                opt.push(union_of(x, y, types));
                cb.advance();
            }
            (Some((PosKind::Optional, x)), None) => {
                opt.push(x);
                ca.advance();
            }
            (None, Some((PosKind::Optional, y))) => {
                opt.push(y);
                cb.advance();
            }
            (Some((PosKind::Rest, x)), Some((PosKind::Rest, y))) => {
                rest = Some(union_of(x, y, types));
                break;
            }
            (Some((PosKind::Rest, x)), None) => {
                rest = Some(x);
                break;
            }
            (None, Some((PosKind::Rest, y))) => {
                rest = Some(y);
                break;
            }
            (None, None) => break,
        }
    }

    (req, opt, rest)
}

#[allow(clippy::type_complexity)]
fn merge_keywords(
    a: &Function,
    b: &Function,
    types: &TypeTable,
) -> (Vec<(String, Ty)>, Vec<(String, Ty)>, Option<Ty>) {
    use std::collections::BTreeSet;

    let lookup = |kws_req: &[(String, Ty)],
                  kws_opt: &[(String, Ty)],
                  key: &str|
     -> Option<(KeywordPresence, Ty)> {
        if let Some((_, t)) = kws_req.iter().find(|(n, _)| n == key) {
            return Some((KeywordPresence::Required, *t));
        }
        if let Some((_, t)) = kws_opt.iter().find(|(n, _)| n == key) {
            return Some((KeywordPresence::Optional, *t));
        }
        None
    };

    let mut all_keys: BTreeSet<&str> = BTreeSet::new();
    for (n, _) in &a.required_keywords {
        all_keys.insert(n);
    }
    for (n, _) in &a.optional_keywords {
        all_keys.insert(n);
    }
    for (n, _) in &b.required_keywords {
        all_keys.insert(n);
    }
    for (n, _) in &b.optional_keywords {
        all_keys.insert(n);
    }

    let mut req: Vec<(String, Ty)> = vec![];
    let mut opt: Vec<(String, Ty)> = vec![];

    for key in all_keys {
        let lhs = lookup(&a.required_keywords, &a.optional_keywords, key);
        let rhs = lookup(&b.required_keywords, &b.optional_keywords, key);
        let merged: (KeywordPresence, Ty) = match (lhs, rhs, a.rest_keyword, b.rest_keyword) {
            // Both required → required(union)
            (Some((KeywordPresence::Required, t)), Some((KeywordPresence::Required, s)), _, _) => {
                (KeywordPresence::Required, union_of(t, s, types))
            }
            // Required + Optional → optional(union, +nil)
            (Some((KeywordPresence::Required, t)), Some((KeywordPresence::Optional, s)), _, _)
            | (Some((KeywordPresence::Optional, s)), Some((KeywordPresence::Required, t)), _, _) => {
                let u = union_of(t, s, types);
                (KeywordPresence::Optional, union_of(u, Ty::NIL, types))
            }
            // Both optional → optional(union)
            (Some((KeywordPresence::Optional, t)), Some((KeywordPresence::Optional, s)), _, _) => {
                (KeywordPresence::Optional, union_of(t, s, types))
            }
            // One side required, other absent → opponent rest absorbs or +nil
            (Some((KeywordPresence::Required, t)), None, _, Some(r)) => {
                let u = union_of(t, r, types);
                (KeywordPresence::Optional, union_of(u, Ty::NIL, types))
            }
            (None, Some((KeywordPresence::Required, t)), Some(r), _) => {
                let u = union_of(t, r, types);
                (KeywordPresence::Optional, union_of(u, Ty::NIL, types))
            }
            (Some((KeywordPresence::Required, t)), None, _, None) => {
                (KeywordPresence::Optional, union_of(t, Ty::NIL, types))
            }
            (None, Some((KeywordPresence::Required, t)), None, _) => {
                (KeywordPresence::Optional, union_of(t, Ty::NIL, types))
            }
            // One side optional, other absent → opponent rest unions, no +nil
            (Some((KeywordPresence::Optional, t)), None, _, Some(r)) => {
                (KeywordPresence::Optional, union_of(t, r, types))
            }
            (None, Some((KeywordPresence::Optional, t)), Some(r), _) => {
                (KeywordPresence::Optional, union_of(t, r, types))
            }
            (Some((KeywordPresence::Optional, t)), None, _, None) => (KeywordPresence::Optional, t),
            (None, Some((KeywordPresence::Optional, t)), None, _) => (KeywordPresence::Optional, t),
            // Neither side has the key (unreachable: key came from a or b)
            (None, None, _, _) => continue,
        };

        match merged.0 {
            KeywordPresence::Required => req.push((key.to_string(), merged.1)),
            KeywordPresence::Optional => opt.push((key.to_string(), merged.1)),
        }
    }

    let rest_kw = match (a.rest_keyword, b.rest_keyword) {
        (Some(x), Some(y)) => Some(union_of(x, y, types)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    };

    (req, opt, rest_kw)
}

/// Untyped function type `(?) -> T` — accepts any arguments without checking.
/// Mirrors `RBS::Types::UntypedFunction`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UntypedFunction {
    pub return_type: Ty,
}

/// Union of typed and untyped function types. crema-only fold:
/// rbs encodes `RBS::Types::Function | RBS::Types::UntypedFunction` as a
/// duck-typed union; this enum closes it into a Rust sum type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionType {
    Typed(Function),
    Untyped(UntypedFunction),
}

impl FunctionType {
    pub fn return_type(&self) -> Ty {
        match self {
            FunctionType::Typed(f) => f.return_type,
            FunctionType::Untyped(u) => u.return_type,
        }
    }

    /// Compose two `FunctionType`s for overload unification:
    /// Typed/Typed delegates to `Function::merge_for_overload`; any untyped
    /// side collapses to `Untyped` with the return type unioned across both.
    /// Used by `MethodType::unify_overload` and the block-composition path.
    fn merge_for_overload(&self, other: &FunctionType, types: &TypeTable) -> FunctionType {
        match (self, other) {
            (FunctionType::Typed(f1), FunctionType::Typed(f2)) => {
                FunctionType::Typed(f1.merge_for_overload(f2, types))
            }
            _ => FunctionType::Untyped(UntypedFunction {
                return_type: union_of(self.return_type(), other.return_type(), types),
            }),
        }
    }
}

/// A block argument type, e.g. `{ (Integer) -> String }` or
/// `{ (Integer) [self: Foo] -> String }`.
/// Mirrors `RBS::Types::Block`. The rbs field `type` is renamed `type_`
/// here to escape the Rust keyword.
///
/// `self_type` is the `[self: T]` binding declared on the block argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Block {
    pub required: bool,
    pub type_: FunctionType,
    pub self_type: Option<Ty>,
}

impl Block {
    pub fn return_type(&self) -> Ty {
        self.type_.return_type()
    }

    /// Returns the required positional parameter types of the block.
    pub fn params(&self) -> &[Ty] {
        match &self.type_ {
            FunctionType::Typed(f) => &f.required_positionals,
            FunctionType::Untyped(_) => &[],
        }
    }

    /// Required then optional positional types, flattened in declaration
    /// order — the slots a block's leading params zip against (Steep
    /// `Function#flat_unnamed_params`). Trailing positionals are not
    /// included, matching Steep.
    pub fn flat_positionals(&self) -> Vec<Ty> {
        match &self.type_ {
            FunctionType::Typed(f) => f
                .required_positionals
                .iter()
                .chain(f.optional_positionals.iter())
                .copied()
                .collect(),
            FunctionType::Untyped(_) => vec![],
        }
    }

    /// The `*rest` positional type of the block, if declared.
    pub fn rest_positional(&self) -> Option<Ty> {
        match &self.type_ {
            FunctionType::Typed(f) => f.rest_positional,
            FunctionType::Untyped(_) => None,
        }
    }
}

/// A method overload type. Mirrors `RBS::MethodType` (which holds
/// `type: function` + `block: Block?`).
///
/// `type_params` are method-level type parameters (the `X` in
/// `def map: [X] () { (T) -> X } -> Array[X]`). They are inferred at each call
/// site; class-level parameters (e.g. `Array[T]`'s `T`) are bound from the
/// receiver and live outside this struct.
///
/// Each entry carries an optional upper/lower bound. Method-level variance is
/// forbidden by RBS syntax (see rbs docs/syntax.md §822) so `variance` is
/// always `Invariant` here.
#[derive(Debug, Clone)]
pub struct MethodType {
    pub type_params: Vec<crate::type_param::TypeParam>,
    pub type_: FunctionType,
    pub block: Option<Block>,
}

impl MethodType {
    /// Synthesize a getter method type: `() -> T`
    pub fn getter(return_type: Ty) -> Self {
        MethodType {
            type_params: vec![],
            type_: FunctionType::Typed(Function::empty(return_type)),
            block: None,
        }
    }

    /// Synthesize a setter method type: `(T) -> T`
    pub fn setter(ty: Ty) -> Self {
        MethodType {
            type_params: vec![],
            type_: FunctionType::Typed(Function {
                required_positionals: vec![ty],
                return_type: ty,
                ..Function::empty(ty)
            }),
            block: None,
        }
    }

    pub fn return_type(&self) -> Ty {
        self.type_.return_type()
    }

    /// Returns `true` if this is an `UntypedFunction` (`(?) -> T`).
    /// Callers should skip argument checking for untyped methods.
    pub fn is_untyped_function(&self) -> bool {
        matches!(self.type_, FunctionType::Untyped(_))
    }

    /// Returns the underlying `Function` for typed overloads, or `None` for UntypedFunction.
    pub fn func(&self) -> Option<&Function> {
        match &self.type_ {
            FunctionType::Typed(f) => Some(f),
            FunctionType::Untyped(_) => None,
        }
    }

    // Delegation methods — return empty slices / None for UntypedFunction.

    pub fn required_positionals(&self) -> &[Ty] {
        self.func().map_or(&[], |f| &f.required_positionals)
    }

    pub fn optional_positionals(&self) -> &[Ty] {
        self.func().map_or(&[], |f| &f.optional_positionals)
    }

    pub fn rest_positional(&self) -> Option<Ty> {
        self.func().and_then(|f| f.rest_positional)
    }

    pub fn trailing_positionals(&self) -> &[Ty] {
        self.func().map_or(&[], |f| &f.trailing_positionals)
    }

    pub fn required_keywords(&self) -> &[(String, Ty)] {
        self.func().map_or(&[], |f| &f.required_keywords)
    }

    pub fn optional_keywords(&self) -> &[(String, Ty)] {
        self.func().map_or(&[], |f| &f.optional_keywords)
    }

    pub fn rest_keyword(&self) -> Option<Ty> {
        self.func().and_then(|f| f.rest_keyword)
    }

    /// Expected type for the keyword argument `name`: a required
    /// keyword, then an optional one, then `**rest`. `None` means the
    /// overload has no home for it (Steep `KeywordArgs#keyword_type`
    /// followed by `rest_type`).
    pub fn keyword_param_type(&self, name: &str) -> Option<Ty> {
        self.required_keywords()
            .iter()
            .chain(self.optional_keywords())
            .find(|(param, _)| param == name)
            .map(|(_, ty)| *ty)
            .or(self.rest_keyword())
    }

    /// Minimum positional arguments required. Always 0 for UntypedFunction.
    pub fn min_arity(&self) -> usize {
        self.func().map_or(0, |f| f.min_arity())
    }

    /// Maximum positional arguments accepted. `None` means unlimited.
    /// UntypedFunction always returns `None` (accepts any count).
    pub fn max_arity(&self) -> Option<usize> {
        match &self.type_ {
            FunctionType::Typed(f) => f.max_arity(),
            FunctionType::Untyped(_) => None,
        }
    }

    /// Whether this overload accepts `count` positional arguments.
    /// UntypedFunction always returns `true`.
    pub fn arity_accepts(&self, count: usize) -> bool {
        match &self.type_ {
            FunctionType::Typed(f) => f.arity_accepts(count),
            FunctionType::Untyped(_) => true,
        }
    }

    /// Resolve the parameter type at call-site positional `index` given total
    /// `arg_count`. Honors RBS layout `required → optional → rest → trailing`
    /// with trailing anchored to the END of the actual argument list (Ruby
    /// semantics). Returns `None` for UntypedFunction or when no slot matches
    /// (e.g. extra args without a rest).
    pub fn positional_param_for_call(&self, index: usize, arg_count: usize) -> Option<Ty> {
        self.func()
            .and_then(|f| f.positional_param_for_call(index, arg_count))
    }

    /// Resolve the parameter type expected for a call-site keyword `name`:
    /// required → optional → `**rest`. Returns `None` when the overload
    /// has neither a matching declared keyword nor a `**rest`.
    pub fn keyword_param_for_call(&self, name: &str) -> Option<Ty> {
        self.required_keywords()
            .iter()
            .chain(self.optional_keywords().iter())
            .find(|(n, _)| n == name)
            .map(|(_, ty)| *ty)
            .or_else(|| self.rest_keyword())
    }

    /// Combine two overload signatures into one composite `MethodType` that a
    /// method body can be type-checked against. Ports Steep's
    /// `MethodType#unify_overload` (function merge + return union + block
    /// composition).
    ///
    /// - `type_`: typed+typed delegates to `Function::merge_for_overload`;
    ///   any untyped side collapses to `Untyped` with the return type unioned.
    /// - `block`: both `Some` composes via `Block#+` (params merged, return
    ///   unioned, `required` ANDed, `self_type` unioned across the non-None
    ///   sides). One-sided block becomes `Some` with `required = false`
    ///   (Steep `to_optional`). Both `None` stays `None`.
    /// - `type_params`: simple concat. Name collision avoidance (Steep
    ///   `TypeParam.rename`) is out of scope for this port.
    pub fn unify_overload(&self, other: &MethodType, types: &TypeTable) -> MethodType {
        let mut type_params = self.type_params.clone();
        type_params.extend(other.type_params.iter().cloned());

        let type_ = self.type_.merge_for_overload(&other.type_, types);
        let block = unify_blocks(self.block.as_ref(), other.block.as_ref(), types);

        MethodType {
            type_params,
            type_,
            block,
        }
    }
}

/// Compose two optional `Block` arguments under overload unification, mirroring
/// Steep's `Block#+` together with `MethodType#unify_overload`'s `to_optional`
/// fallback for the one-sided case.
fn unify_blocks(a: Option<&Block>, b: Option<&Block>, types: &TypeTable) -> Option<Block> {
    match (a, b) {
        (Some(b1), Some(b2)) => {
            let type_ = b1.type_.merge_for_overload(&b2.type_, types);
            let self_type = match (b1.self_type, b2.self_type) {
                (None, None) => None,
                (Some(t), None) | (None, Some(t)) => Some(t),
                (Some(t), Some(s)) => Some(union_of(t, s, types)),
            };
            Some(Block {
                required: b1.required && b2.required,
                type_,
                self_type,
            })
        }
        (Some(b), None) | (None, Some(b)) => Some(Block {
            required: false,
            type_: b.type_.clone(),
            self_type: b.self_type,
        }),
        (None, None) => None,
    }
}

/// Method / attribute visibility as it appears on the declared class.
///
/// RBS (`syntax.md` `_visibility_`) recognises only `public` and `private`.
/// `protected` is deliberately not represented: adding it would exceed the
/// RBS grammar crema is meant to mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Public,
    Private,
}

impl Visibility {
    pub fn from_ast(v: crate::ast::Visibility) -> Self {
        match v {
            crate::ast::Visibility::Public => Visibility::Public,
            crate::ast::Visibility::Private => Visibility::Private,
        }
    }

    /// Resolve an `Option<AstVisibility>` (rbs port shape — `None` is
    /// the source-level "unspecified" state) into a concrete `Visibility`.
    /// Today the build phase eager-folds visibility at the rbs_loader
    /// boundary, so most callers see `Some(_)` and the `None` arm is the
    /// safety net for paths that haven't been folded yet (or that may
    /// preserve raw nil when the unspecified-state migration lands).
    pub fn from_ast_or_default(v: Option<crate::ast::Visibility>) -> Self {
        v.map(Self::from_ast).unwrap_or_default()
    }
}

const TYPE_STORE_CHUNK: usize = 1024;

/// Append-only storage for interned types.
///
/// Each chunk is a fixed-capacity `Vec<Type>` allocated in its own `Box`. The
/// invariants that make `&Type` references stable across subsequent `push`
/// calls are:
///
/// - The outer `Vec<Box<Vec<Type>>>` may reallocate when a new chunk is added,
///   but `Box` does not move the heap data it owns when its slot in the outer
///   vec is relocated. Existing chunk contents stay at the same heap address.
/// - Each inner `Vec<Type>` is allocated with `Vec::with_capacity(CHUNK)` and
///   never pushed past `CHUNK` items, so it never reallocates and the slot
///   memory inside it never moves either.
///
/// Combined: once a `Type` is pushed and a `&Type` is handed out, subsequent
/// `push` calls cannot invalidate that reference. This is the foundation
/// `TypeTable::resolve` relies on for borrowed reads.
///
/// `push` and `get` both take `&self` to avoid materialising a `&mut
/// ChunkedTypeStore` at the call site — that mutable borrow would alias the
/// `&Type` references already lent out, which is what makes a naive
/// `&mut self` design unsound under stacked borrows.
#[allow(clippy::vec_box)]
struct ChunkedTypeStore {
    chunks: UnsafeCell<Vec<Box<Vec<Type>>>>,
    len: Cell<u32>,
}

impl ChunkedTypeStore {
    fn new() -> Self {
        Self {
            chunks: UnsafeCell::new(Vec::new()),
            len: Cell::new(0),
        }
    }

    fn len(&self) -> u32 {
        self.len.get()
    }

    /// Append a `Type` and return its `u32` index.
    ///
    /// # Safety / soundness rationale
    ///
    /// The `&mut Vec<Box<Vec<Type>>>` materialised inside this function is
    /// confined to a single statement and the contents it touches (the outer
    /// vec's tail, or the last chunk's tail via `Box::deref_mut`) do not
    /// overlap with any `&Type` previously returned by [`Self::get`]:
    ///
    /// - Outer-vec growth (adding a new chunk box) writes to a slot in the
    ///   outer vec, not to any existing chunk's heap data.
    /// - Inner-vec growth (pushing into the last chunk box) only touches the
    ///   inner vec's `len` field and the freshly-claimed slot — both at the
    ///   tail. Previously-pushed slots are untouched.
    ///
    /// So while the mutable borrow conceptually covers the entire `Vec<Box<...>>`,
    /// the bytes it actually mutates are disjoint from the slot bytes any
    /// outstanding `&Type` is reading. This is the same pattern `elsa`'s
    /// `FrozenVec<Box<T>>` relies on for `&self` push.
    fn push(&self, ty: Type) -> u32 {
        let id = self.len.get();
        debug_assert!(id < u32::MAX, "TypeTable exceeded u32::MAX entries");

        // SAFETY: this is the only `&mut` materialised against `self.chunks`;
        // it is released at the end of this statement. The mutation writes
        // only to the outer vec's tail or the last chunk's tail, which are
        // disjoint from any `&Type` previously handed out by `get` (see the
        // soundness rationale above).
        let chunks = unsafe { &mut *self.chunks.get() };
        if chunks
            .last()
            .is_none_or(|chunk| chunk.len() == TYPE_STORE_CHUNK)
        {
            chunks.push(Box::new(Vec::with_capacity(TYPE_STORE_CHUNK)));
        }
        let chunk = chunks.last_mut().expect("chunk exists after allocation");
        debug_assert!(chunk.len() < TYPE_STORE_CHUNK);
        chunk.push(ty);

        self.len.set(id + 1);
        id
    }

    /// Borrow the `Type` at index `idx`. The returned reference is valid for
    /// the lifetime of `&self`, including across subsequent `push` calls,
    /// because pushes do not move existing slot memory (see struct doc).
    fn get(&self, idx: u32) -> &Type {
        debug_assert!(idx < self.len.get());
        let idx = idx as usize;
        let chunk_idx = idx / TYPE_STORE_CHUNK;
        let slot_idx = idx % TYPE_STORE_CHUNK;
        // SAFETY: only shared borrows of `self.chunks` are materialised here.
        // `push` is the only mutator and it takes `&self` plus this single
        // statement's `&mut` is released before any `&Type` escapes — so the
        // mutable side and the shared side cannot be live at the same time
        // for a given thread. `TypeTable` is `!Sync` by construction
        // (`UnsafeCell` is not `Sync`), so we do not need to reason across
        // threads.
        let chunks = unsafe { &*self.chunks.get() };
        &chunks[chunk_idx][slot_idx]
    }

    #[cfg(test)]
    fn chunk_count(&self) -> usize {
        // SAFETY: shared borrow only; see `get` for the rationale.
        let chunks = unsafe { &*self.chunks.get() };
        chunks.len()
    }
}

fn hash_type(ty: &Type) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    ty.hash(&mut hasher);
    hasher.finish()
}

/// A type interner that maps structural `Type` values to `Ty` handles.
///
/// `intern` is the only mutation path. It appends new entries without moving
/// existing storage slots, so `resolve` can lend references to stored types
/// even across subsequent `intern` calls — including the
/// `match types.resolve(t) { ... types.intern(...) ... }` pattern that recurs
/// throughout the type checker.
///
/// # Soundness sketch
///
/// `chunks` and `index` are held in **separate** `UnsafeCell`s. `resolve`
/// only touches `chunks`; `intern` mutates `index` and then (separately)
/// `chunks`. The two mutable borrows never overlap with each other and they
/// live in different allocations from any `&Type` previously handed out by
/// `resolve`, so the references `resolve` lends remain valid across `intern`
/// calls. Internal mutability for `chunks` is encapsulated inside
/// [`ChunkedTypeStore`] (see its doc for the chunk-stability argument).
pub struct TypeTable {
    chunks: ChunkedTypeStore,
    index: UnsafeCell<FxHashMap<u64, Vec<Ty>>>,
}

impl fmt::Debug for TypeTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeTable({} types)", self.chunks.len())
    }
}

impl TypeTable {
    pub fn new() -> Self {
        let table = TypeTable {
            chunks: ChunkedTypeStore::new(),
            index: UnsafeCell::new(FxHashMap::default()),
        };
        // Pre-intern well-known types. Order must match Ty constants.
        table.intern(Type::Void); // Ty(0) = Ty::VOID
        table.intern(Type::Nil); // Ty(1) = Ty::NIL
        table.intern(Type::Bool); // Ty(2) = Ty::BOOL
        table.intern(Type::Untyped); // Ty(3) = Ty::UNTYPED
        table.intern(Type::Top); // Ty(4) = Ty::TOP
        table.intern(Type::Bottom); // Ty(5) = Ty::BOTTOM
        table.intern(Type::SelfType); // Ty(6) = Ty::SELF_TYPE
        table.intern(Type::InstanceType); // Ty(7) = Ty::INSTANCE_TYPE
        table.intern(Type::ClassType); // Ty(8) = Ty::CLASS_TYPE
        table
    }

    /// Intern a structural type, returning the same `Ty` for structurally equal types.
    pub fn intern(&self, ty: Type) -> Ty {
        let hash = hash_type(&ty);
        self.intern_with_hash_internal(ty, hash)
    }

    #[cfg(test)]
    pub(crate) fn intern_with_hash(&self, ty: Type, hash: u64) -> Ty {
        self.intern_with_hash_internal(ty, hash)
    }

    fn intern_with_hash_internal(&self, ty: Type, hash: u64) -> Ty {
        // Phase A: look up the hash bucket. We snapshot the bucket's Ty list
        // and immediately release the borrow on `index` — this lets us touch
        // `chunks` next without keeping a live `&mut`/`&` into `index`.
        let candidates: Vec<Ty> = {
            // SAFETY: shared borrow of `index`, released at the end of this
            // block. No other code in this function holds an `index` borrow
            // concurrently. The borrow does not escape this block.
            let index = unsafe { &*self.index.get() };
            match index.get(&hash) {
                Some(bucket) => bucket.clone(),
                None => Vec::new(),
            }
        };

        // Phase B: probe candidates via `chunks` (shared borrows only).
        for existing in &candidates {
            if self.chunks.get(existing.0) == &ty {
                return *existing;
            }
        }

        // Phase C: append the new type to `chunks`. `ChunkedTypeStore::push`
        // takes `&self`; its internal `&mut` is confined to the call and does
        // not alias any `&Type` we have lent out (see ChunkedTypeStore doc).
        let id = Ty(self.chunks.push(ty));

        // Phase D: record the new id in `index`. The mutable borrow is
        // confined to this block and does not overlap with `chunks` borrows.
        // SAFETY: `index` and `chunks` live in disjoint allocations, so this
        // `&mut` cannot alias any `&Type` returned by `resolve`. No other
        // code in this function holds an `index` borrow concurrently —
        // Phase A's borrow was already released.
        unsafe {
            (*self.index.get()).entry(hash).or_default().push(id);
        }
        id
    }

    /// Resolve a `Ty` handle back to its structural `Type`.
    ///
    /// The returned `&Type` is valid for the lifetime of `&self`, including
    /// across subsequent `intern` calls. This is what makes the
    /// `match types.resolve(t) { ... types.intern(...) ... }` idiom safe.
    pub fn resolve(&self, ty: Ty) -> &Type {
        self.chunks.get(ty.0)
    }

    #[cfg(test)]
    pub(crate) fn chunk_count(&self) -> usize {
        self.chunks.chunk_count()
    }

    /// Convenience: intern a class instance type with no type arguments.
    pub fn class_instance(&self, name: TypeName) -> Ty {
        self.intern(Type::ClassInstance { name, args: vec![] })
    }

    /// Convenience: intern a class singleton type.
    pub fn class_singleton(&self, name: TypeName) -> Ty {
        self.intern(Type::ClassSingleton { name })
    }
}

impl Default for TypeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Replace every remaining `Type::TypeVariable { .. }` inside `ty` with
/// `Ty::UNTYPED`, recursing into composite types.
///
/// Used at the tail of call-site return-type computation: type parameters
/// that never got bound (e.g. `class Pair[K, V]` whose `initialize` only
/// references `K`) would otherwise leak out as bare type variables. RBS/
/// Steep surface them as `untyped` instead, and crema follows the same
/// gradual-typing escape hatch.
///
/// Does **not** touch `SelfType` or perform any binding lookup — pair this
/// with `crate::substitution::Substitution::apply` (which does the binding lookup and preserves
/// unbound variables) to get the full behavior.
pub fn fallback_unbound_type_vars_to_untyped(ty: Ty, types: &TypeTable) -> Ty {
    match types.resolve(ty) {
        Type::TypeVariable { .. } => Ty::UNTYPED,
        Type::ClassInstance { name, args } => {
            let new_args: Vec<Ty> = args
                .iter()
                .map(|&a| fallback_unbound_type_vars_to_untyped(a, types))
                .collect();
            types.intern(Type::ClassInstance {
                name: *name,
                args: new_args,
            })
        }
        Type::Union(members) => {
            let new_members: Vec<Ty> = members
                .iter()
                .map(|&m| fallback_unbound_type_vars_to_untyped(m, types))
                .collect();
            types.intern(Type::Union(new_members))
        }
        Type::Intersection(members) => {
            let new_members: Vec<Ty> = members
                .iter()
                .map(|&m| fallback_unbound_type_vars_to_untyped(m, types))
                .collect();
            types.intern(Type::Intersection(new_members))
        }
        Type::Optional(inner) => {
            let new_inner = fallback_unbound_type_vars_to_untyped(*inner, types);
            types.intern(Type::Optional(new_inner))
        }
        Type::Interface { name, args } => {
            let new_args: Vec<Ty> = args
                .iter()
                .map(|&a| fallback_unbound_type_vars_to_untyped(a, types))
                .collect();
            types.intern(Type::Interface {
                name: *name,
                args: new_args,
            })
        }
        Type::Alias { name, args } => {
            let new_args: Vec<Ty> = args
                .iter()
                .map(|&a| fallback_unbound_type_vars_to_untyped(a, types))
                .collect();
            types.intern(Type::Alias {
                name: *name,
                args: new_args,
            })
        }
        Type::Tuple(members) => {
            let new_members: Vec<Ty> = members
                .iter()
                .map(|&m| fallback_unbound_type_vars_to_untyped(m, types))
                .collect();
            types.intern(Type::Tuple(new_members))
        }
        Type::Record { fields } => {
            let new_fields: Vec<(RecordKey, Ty, bool)> = fields
                .iter()
                .map(|(key, ty, required)| {
                    (
                        key.clone(),
                        fallback_unbound_type_vars_to_untyped(*ty, types),
                        *required,
                    )
                })
                .collect();
            types.intern(Type::Record { fields: new_fields })
        }
        Type::Proc {
            type_,
            self_type,
            block,
        } => {
            let new_type = fallback_unbound_in_function_type(type_, types);
            let new_self = self_type.map(|st| fallback_unbound_type_vars_to_untyped(st, types));
            let new_block = block.as_ref().map(|b| Block {
                required: b.required,
                type_: fallback_unbound_in_function_type(&b.type_, types),
                self_type: b
                    .self_type
                    .map(|st| fallback_unbound_type_vars_to_untyped(st, types)),
            });
            types.intern(Type::Proc {
                type_: new_type,
                self_type: new_self,
                block: new_block,
            })
        }
        _ => ty,
    }
}

fn fallback_unbound_in_function_type(ft: &FunctionType, types: &TypeTable) -> FunctionType {
    let walk = |t: Ty| fallback_unbound_type_vars_to_untyped(t, types);
    match ft {
        FunctionType::Typed(f) => FunctionType::Typed(Function {
            required_positionals: f.required_positionals.iter().copied().map(walk).collect(),
            optional_positionals: f.optional_positionals.iter().copied().map(walk).collect(),
            rest_positional: f.rest_positional.map(walk),
            trailing_positionals: f.trailing_positionals.iter().copied().map(walk).collect(),
            required_keywords: f
                .required_keywords
                .iter()
                .map(|(k, t)| (k.clone(), walk(*t)))
                .collect(),
            optional_keywords: f
                .optional_keywords
                .iter()
                .map(|(k, t)| (k.clone(), walk(*t)))
                .collect(),
            rest_keyword: f.rest_keyword.map(walk),
            return_type: walk(f.return_type),
        }),
        FunctionType::Untyped(u) => FunctionType::Untyped(UntypedFunction {
            return_type: walk(u.return_type),
        }),
    }
}

/// Returns `true` when `ty` (or any nested element) still contains an
/// unresolved `Type::TypeVariable`. Used as a guard for hint propagation:
/// passing a hint that still mentions a generic parameter would poison
/// downstream inference (literal nodes would latch onto the bare variable).
/// Mirrors the traversal shape of [`fallback_unbound_type_vars_to_untyped`].
pub fn contains_type_variable(ty: Ty, types: &TypeTable) -> bool {
    match types.resolve(ty) {
        Type::TypeVariable { .. } => true,
        Type::ClassInstance { args, .. }
        | Type::Interface { args, .. }
        | Type::Alias { args, .. } => args.iter().any(|&a| contains_type_variable(a, types)),
        Type::Union(members) | Type::Intersection(members) | Type::Tuple(members) => {
            members.iter().any(|&m| contains_type_variable(m, types))
        }
        Type::Optional(inner) => contains_type_variable(*inner, types),
        Type::Record { fields } => fields
            .iter()
            .any(|(_, t, _)| contains_type_variable(*t, types)),
        Type::Proc {
            type_,
            self_type,
            block,
        } => {
            function_type_contains_type_variable(type_, types)
                || self_type.is_some_and(|st| contains_type_variable(st, types))
                || block.as_ref().is_some_and(|b| {
                    function_type_contains_type_variable(&b.type_, types)
                        || b.self_type
                            .is_some_and(|st| contains_type_variable(st, types))
                })
        }
        _ => false,
    }
}

fn function_type_contains_type_variable(ft: &FunctionType, types: &TypeTable) -> bool {
    let walk = |t: Ty| contains_type_variable(t, types);
    match ft {
        FunctionType::Typed(f) => {
            f.required_positionals.iter().copied().any(walk)
                || f.optional_positionals.iter().copied().any(walk)
                || f.rest_positional.is_some_and(walk)
                || f.trailing_positionals.iter().copied().any(walk)
                || f.required_keywords.iter().any(|(_, t)| walk(*t))
                || f.optional_keywords.iter().any(|(_, t)| walk(*t))
                || f.rest_keyword.is_some_and(walk)
                || walk(f.return_type)
        }
        FunctionType::Untyped(u) => walk(u.return_type),
    }
}

/// Recursive walk backing [`partition_truthy`]: pushes every truthy leaf
/// reachable by peeling nested `Optional`/`Union` layers into `out`,
/// skipping `nil` and the `false` literal wherever they appear. Each
/// call strictly peels one layer, so this terminates on any finite type
/// tree — a defensively-handled `Optional(Optional(T))` bottoms out
/// after two steps rather than looping (this shape shouldn't arise given
/// rbs's invariant that `T?` sugar never nests, but non-canonical unions
/// from intern paths like overload-return aggregation are the reason
/// this recurses instead of matching one layer).
fn collect_truthy_members(ty: Ty, types: &TypeTable, out: &mut Vec<Ty>) {
    match types.resolve(ty) {
        Type::Optional(inner) => collect_truthy_members(*inner, types, out),
        Type::Nil | Type::Literal(Literal::Bool(false)) => {}
        Type::Union(members) => {
            for &m in members.iter() {
                collect_truthy_members(m, types, out);
            }
        }
        _ => out.push(ty),
    }
}

/// Partition a type into its truthy side for flow-sensitive narrowing.
///
/// Returns `Some(t)` where `t` is `ty` with `nil` and `false` literal
/// components removed; `None` if the truthy set is empty (the branch is
/// statically unreachable).
///
/// Recurses through nested `Optional`/`Union` layers, so a member that is
/// itself `Optional` (e.g. `Integer? | String`, the shape a union of two
/// nilable-returning methods takes) still has its own falsy component
/// stripped instead of passing through unexamined.
pub fn partition_truthy(ty: Ty, types: &TypeTable) -> Option<Ty> {
    let mut truthy = Vec::new();
    collect_truthy_members(ty, types, &mut truthy);
    if truthy.is_empty() {
        None
    } else {
        // `union_of_many` (not a bare `intern(Type::Union(..))`) so a
        // duplicate leaf — e.g. `Integer? | Integer` unwraps to two
        // `Integer` members — collapses through the same dedup pass
        // `union_of` itself relies on, instead of surviving verbatim
        // into callers (`multi_write_truthy_env` et al.) that don't
        // happen to re-normalize the result themselves.
        Some(union_of_many(&truthy, types))
    }
}

/// Recursive walk backing [`partition_falsy`]: pushes every `false`
/// literal leaf into `out` and records whether any `nil` was seen along
/// the way via `has_nil`, so the caller can push a single `nil` at the
/// end instead of once per `Optional` layer peeled — `Integer? | String?`
/// (two independently-nilable union members) must yield one `nil`, not a
/// duplicate pair. See [`collect_truthy_members`] for the termination
/// argument, which applies identically here.
fn collect_falsy_members(ty: Ty, types: &TypeTable, has_nil: &mut bool, out: &mut Vec<Ty>) {
    match types.resolve(ty) {
        Type::Optional(inner) => {
            *has_nil = true;
            collect_falsy_members(*inner, types, has_nil, out);
        }
        Type::Nil => *has_nil = true,
        Type::Literal(Literal::Bool(false)) => out.push(ty),
        Type::Union(members) => {
            for &m in members.iter() {
                collect_falsy_members(m, types, has_nil, out);
            }
        }
        _ => {}
    }
}

/// Partition a type into its falsy side — the dual of [`partition_truthy`].
///
/// Returns `Some(t)` where `t` contains only `nil` and `false` literal
/// components of `ty`; `None` if `ty` has no falsy member (the branch is
/// statically unreachable).
///
/// Used by `&&` to type the left's falsy passthrough: `a && b` keeps
/// `a`'s falsy partition when `a` was falsy, otherwise yields `b`.
/// Recurses through nested `Optional`/`Union` layers for the same reason
/// as [`partition_truthy`] — see its doc comment.
pub fn partition_falsy(ty: Ty, types: &TypeTable) -> Option<Ty> {
    let mut has_nil = false;
    let mut falsy = Vec::new();
    collect_falsy_members(ty, types, &mut has_nil, &mut falsy);
    if has_nil {
        falsy.push(Ty::NIL);
    }
    if falsy.is_empty() {
        None
    } else {
        // See `partition_truthy`'s matching comment: `union_of_many`
        // dedupes a repeated `false` literal the same way it dedupes a
        // repeated non-nil leaf on the truthy side.
        Some(union_of_many(&falsy, types))
    }
}

/// Build a normalized union of two types.
///
/// Pipeline (mirrors `Steep::AST::Types::Union.build` in
/// `lib/steep/ast/types/union.rb:11-44`):
///
/// 1. flatten one level of `Type::Union`
/// 2. absorb `Untyped` — `untyped | T = untyped` (Steep's Any absorption)
/// 3. absorb `Top` — `Top | T = Top`
/// 4. drop `Bottom` (union identity element)
/// 5. dedupe by intern id
/// 6. sort by intern id so `union_of(a, b)` and `union_of(b, a)` collapse
///    to the same `Ty` at the intern table
/// 7. collapse: 0 → `Bottom`, 1 → the member, n → `Type::Union`
///
/// Two deliberate divergences from Steep:
///
/// - **`Untyped` wins over `Top`**. Steep's `case` walk returns whichever
///   absorber appears first in the flattened member list; crema pins
///   `Untyped` as the winner so a `union_of(untyped, T)` call always
///   degrades to `untyped`. This keeps the gradual-typing fallback
///   target stable across refactors and across operand reordering.
/// - **Member sort**. Steep / rbs preserve insertion order (visible in
///   `to_s`); crema sorts by intern id so commutativity holds at the
///   intern table. Diagnostic display order will not match Steep's, but
///   `union_of(a, b)` and `union_of(b, a)` reuse the same intern slot.
///   The `Type::Union` / `Type::Intersection` display then re-sorts
///   members by rendered string so the same union renders identically
///   across intern orders (cold fresh build vs warm a-snapshot decode
///   produce different intern sequences). This canonicalization applies
///   to every `Type::Union` display path — including source-declared
///   unions constructed via `intern(Type::Union(..))` in
///   `type_builder` / `substitution` / `type_param` that would otherwise
///   preserve the RBS-source member order. The trade-off is intentional:
///   a single canonical rendering per union set is more valuable for AI
///   consumers than mirroring the surface-syntax member order across
///   both source-declared and inferred unions.
pub fn union_of(a: Ty, b: Ty, types: &TypeTable) -> Ty {
    if a == b {
        return a;
    }
    let mut members: Vec<Ty> = Vec::with_capacity(2);
    push_flat(a, &mut members, types);
    push_flat(b, &mut members, types);
    normalize_union_members(members, types)
}

/// Build a normalized union from a slice of types. Returns `Bottom`
/// for an empty slice (the identity element of union). Runs the same
/// pipeline as [`union_of`].
pub fn union_of_many(tys: &[Ty], types: &TypeTable) -> Ty {
    let mut members: Vec<Ty> = Vec::with_capacity(tys.len());
    for &t in tys {
        push_flat(t, &mut members, types);
    }
    normalize_union_members(members, types)
}

fn push_flat(ty: Ty, out: &mut Vec<Ty>, types: &TypeTable) {
    match types.resolve(ty) {
        Type::Union(members) => {
            for &m in members {
                out.push(m);
            }
        }
        _ => out.push(ty),
    }
}

/// Build a normalized intersection — the dual of [`union_of`]. Absorption is
/// mirrored: `Bottom` annihilates (the union's `Top`), `Top` members drop
/// (the union's `Bottom`), and an empty result is `Top` (the identity of
/// intersection). `untyped` degrades to `untyped` as in [`union_of`], which
/// lets `check_block` skip a block whose synthesized return is gradual.
pub fn intersection_of(a: Ty, b: Ty, types: &TypeTable) -> Ty {
    if a == b {
        return a;
    }
    let mut members: Vec<Ty> = Vec::with_capacity(2);
    push_flat_intersection(a, &mut members, types);
    push_flat_intersection(b, &mut members, types);
    if members.iter().any(|t| t.is_untyped()) {
        return Ty::UNTYPED;
    }
    if members.contains(&Ty::BOTTOM) {
        return Ty::BOTTOM;
    }
    members.retain(|&t| t != Ty::TOP);
    let mut seen: FxHashSet<Ty> = FxHashSet::default();
    members.retain(|t| seen.insert(*t));
    members.sort();
    match members.len() {
        0 => Ty::TOP,
        1 => members[0],
        _ => types.intern(Type::Intersection(members)),
    }
}

fn push_flat_intersection(ty: Ty, out: &mut Vec<Ty>, types: &TypeTable) {
    match types.resolve(ty) {
        Type::Intersection(members) => {
            for &m in members {
                out.push(m);
            }
        }
        _ => out.push(ty),
    }
}

fn normalize_union_members(mut members: Vec<Ty>, types: &TypeTable) -> Ty {
    // Absorption pass — covers both surface operands and members hoisted
    // out of nested unions by `push_flat`.
    if members.iter().any(|t| t.is_untyped()) {
        return Ty::UNTYPED;
    }
    if members.contains(&Ty::TOP) {
        return Ty::TOP;
    }
    members.retain(|&t| t != Ty::BOTTOM);
    let mut seen: FxHashSet<Ty> = FxHashSet::default();
    members.retain(|t| seen.insert(*t));
    members.sort();
    match members.len() {
        0 => Ty::BOTTOM,
        1 => members[0],
        _ => types.intern(Type::Union(members)),
    }
}

