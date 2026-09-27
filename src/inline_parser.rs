use rustc_hash::FxHashSet;
use std::path::Path;
use std::sync::Arc;

use ruby_prism::{
    CallNode, ClassNode, ConstantPathWriteNode, ConstantWriteNode, DefNode, ModuleNode, Node,
    SingletonClassNode, Visit,
};

use crate::ast::method_type::MethodType as AstMethodType;
use crate::ast::ruby::annotations::{
    ColonMethodTypeAnnotation, LeadingAnnotation, TypeApplicationAnnotation,
};
use crate::ast::ruby::comment_block::{CommentBlock, CommentLine};
use crate::ast::ruby::declarations::{
    ClassDecl, ClassModuleAliasDecl, ConstantDecl, ConstantValueKind, Declaration, ModuleDecl,
    SuperClass,
};
use crate::ast::ruby::members::{
    AttrAccessorMember, AttrReaderMember, AttrWriterMember, AttributeMember, AttributeNameNode,
    DefMember, DefMemberOrigin, ExplicitAnnotation, ExtendMember, IncludeMember,
    InitializeIvarParamPair, InitializeParamRef, InstanceVariableMember, Member,
    MethodTypeAnnotation, MixinMember, ModuleSelfMember, PrependMember, TrailingResolution,
    TypeAnnotations,
};
use crate::ast::ruby::{LineIndex, PrismByteRange};
use crate::ast::types::{
    BaseType, BaseTypeKind, Function, FunctionParam, FunctionType, KeywordParam, Literal,
    LiteralType, RecordField, RecordKey, RecordType, TupleType, Type, UnionType,
};
use crate::ast::{MethodKind, Visibility};
use crate::ast_builder;
use crate::class_new_recognizer::class_dot_new_call;
use crate::data_struct_recognizer::{DataStructConstructionKind, data_struct_construction_kind};
use crate::diagnostic::{Diagnostic, DiagnosticKind, InlineAliasKind};
use crate::environment::draft::EnvironmentDraft;
use crate::environment::ruby_decl::{build_annotation_syntax_error, parse_rbs_type};
use crate::name::{Name, NameTable};
use crate::rbs_raw::Parser as RbsParser;
use crate::type_name::TypeName;

mod comment_association;
pub(crate) use comment_association::{
    CommentAssociation, TrailingAnnotation, classify_trailing_range,
};

/// AST walker that builds a tree of [`Declaration`]s and their nested
/// [`Member`]s from a Ruby source file, mirroring rbs's
/// `RBS::AST::Ruby::*` shape.
///
/// The walker keeps two stacks in sync:
///
/// - `class_stack` is the lexical nesting of class / module names
///   (raw segment strings), used to derive each decl's absolute
///   `TypeName` and to gate top-level-only checks (`is_empty()`).
/// - `cref_stack` is Ruby's cref for bare constant writes. It mirrors
///   `class_stack` except inside a `Const = Class.new do ... end` block
///   body, which owns `def`s (pushed on `class_stack`) but whose
///   constants are defined in the *outer* scope (not pushed here).
/// - `scope_stack` is the **current open class/module's member list**.
///   When the walker enters a class/module it pushes an empty
///   `Vec<Member>`; when it leaves it pops, wraps the result in a
///   [`Declaration`], and attaches it either to the outer scope (as
///   `Member::Declaration`) or — if there is no outer scope — to the
///   top-level list.
///
/// Top-level `def` / `attr_*` / `include`-`extend`-`prepend` calls are
/// intentionally dropped (as they already were with the flat
/// `RawInlineMember` enum) because there is no class on which to hang
/// them. Top-level `CONST =` becomes a top-level [`Declaration::Constant`].
/// A check target's on-disk identity as the inline collector needs it:
/// the path (for `Diagnostic.file`) and that same path already interned
/// as a [`Name`] by the caller (for `Location.source_file` on the
/// collected members). Pre-interning keeps the collector from writing
/// into the positional `Name` overlay of whichever `NameTable` it runs
/// against — a parallel-ingest worker's table (ADR-0033) would otherwise
/// mint `Name` ids main cannot resolve.
#[derive(Clone, Copy)]
pub struct SourceFile<'a> {
    pub path: &'a Path,
    pub name: Name,
}

impl<'a> SourceFile<'a> {
    /// Intern `path` into `names` (the table the collected declarations
    /// will be inserted into) and pair it with the path.
    pub fn intern(path: &'a Path, names: &NameTable) -> Self {
        SourceFile {
            path,
            name: names.intern(&path.to_string_lossy()),
        }
    }
}

struct InlineCollector<'a> {
    source: &'a [u8],
    line_index: LineIndex,
    file: Option<SourceFile<'a>>,
    comments: &'a CommentAssociation,
    names: &'a NameTable,
    class_stack: Vec<String>,
    cref_stack: Vec<String>,
    /// For each `cref_stack` entry, the index of its member list in
    /// `scope_stack`. A `Class.new do` block pushes a `scope_stack`
    /// frame without a cref frame, so declarations written inside it
    /// ([`attach_declaration`]) go to the cref's list, not the block's.
    cref_scope_frames: Vec<usize>,
    /// Declarations emitted at the file's top level (outside any class
    /// or module). Populated by [`attach_declaration`] when
    /// `scope_stack` is empty.
    top_level: Vec<Declaration>,
    /// Top-level `def name` (no receiver, no enclosing class / module)
    /// collected in inline mode. Ruby defines these as private instance
    /// methods of `Object`, so [`finish_object_reopen`] folds them into
    /// one synthetic `class Object` reopen appended to `top_level`.
    /// Intentional divergence from rbs `RBS::InlineParser`, which
    /// rejects them with `TopLevelMethodDefinition` (see
    /// specs/inline.md "既知の意図的乖離").
    object_members: Vec<Member>,
    /// One entry per currently-open class / module, storing the
    /// in-progress `members: Vec<Member>` of that declaration. Empty
    /// between top-level siblings.
    scope_stack: Vec<Vec<Member>>,
    /// Diagnostics raised during collection (e.g. alias declarations
    /// missing both an explicit and an inferable old name).
    diagnostics: Vec<Diagnostic>,
    associated_leading_lines: FxHashSet<usize>,
    singleton_class_depth: usize,
    /// One entry per `scope_stack` frame (pushed / popped at the same
    /// three sites). Tracks the Ruby-side `private` / `public` state of
    /// that class body — a crema extension over the rbs inline contract,
    /// see [`DefMember::visibility`].
    visibility_frames: Vec<VisibilityFrame>,
    /// Explicit modifier being applied to the argument currently under
    /// visit (`private def x`, `private attr_reader :x`). Set by
    /// [`collect_visibility_call`] around the argument visit only, and
    /// consumed by `visit_def_node` / `collect_attr_call`.
    pending_explicit_visibility: Option<Visibility>,
    /// `--inline=true` scan. Gates diagnostics that mirror rbs's
    /// `RBS::InlineParser` restrictions (e.g. `TopLevelMethodDefinition`,
    /// `rbs/lib/rbs/inline_parser.rb:236`) — those are inline-mode
    /// restrictions upstream and must not surface when the collector is
    /// only harvesting skeleton for the sig-mode pipeline.
    inline_mode: bool,
}

/// Ruby-side visibility state of one open class / module body.
///
/// `ambient` is the default set by a bare `private` / `public` /
/// `protected` statement (`None` until one appears; a fresh frame per
/// `class` keyword, so reopening resets it like Ruby does).
/// `direct_statement_starts` holds the start offsets of the body's
/// top-level statements: a visibility call only counts when it is one
/// of them, so `if cond; private; end` (not statically decidable) is
/// ignored and stays on the public side.
struct VisibilityFrame {
    ambient: Option<Visibility>,
    direct_statement_starts: FxHashSet<u32>,
}

impl VisibilityFrame {
    fn new(body: Option<Node<'_>>) -> Self {
        // A body with `rescue` / `ensure` clauses parses as a BeginNode
        // wrapping the statements; its main statements are still direct
        // children of the class body in Ruby's eyes.
        let statements = body.and_then(|body| {
            body.as_statements_node()
                .or_else(|| body.as_begin_node().and_then(|b| b.statements()))
        });
        let direct_statement_starts = statements
            .map(|s| {
                s.body()
                    .iter()
                    .map(|stmt| stmt.location().start_offset() as u32)
                    .collect()
            })
            .unwrap_or_default();
        VisibilityFrame {
            ambient: None,
            direct_statement_starts,
        }
    }
}

#[derive(Clone, Copy)]
struct ConstantAssignmentName {
    name: TypeName,
    location: PrismByteRange,
}

#[derive(Clone, Copy)]
enum AttributeCallSiteKind {
    Reader,
    Writer,
    Accessor,
}

#[derive(Clone, Copy)]
enum MixinCallSiteKind {
    Include,
    Extend,
    Prepend,
}

struct DataStructClassCall<'pr> {
    kind: DataStructConstructionKind,
    super_class_name: TypeName,
    call: CallNode<'pr>,
}

struct DataStructAttribute {
    name: String,
    ty: Type,
    member: Member,
}

#[derive(Default)]
struct DataStructOptions {
    keyword_init: Option<bool>,
    required_new_args: bool,
    readonly_attributes: bool,
}

/// Recognized `Const = Class.new(Parent?) do ... end` shape. The
/// recognizer (`class_new_call`) accepts the call with or without a
/// block and with zero or one bare/path constant parent argument; the
/// no-parent case substitutes `::Object` (Ruby runtime default for
/// `Class.new` — see `Class#initialize`'s `super_class=Object`).
/// `super_class_byte_range` points to the parent argument source
/// range, or `None` when the argument list was empty.
///
/// rbs's own `InlineParser` does not recognize `Const = Class.new`
/// shapes — this is a crema-internal extension, not a port. The
/// closest analogue in rbs is the type checker bridging `Class.new`
/// blocks via `# @implements`, which Steep also requires; crema
/// replaces the annotation with syntactic LHS recognition.
struct ClassNewCall<'pr> {
    super_class_name: TypeName,
    super_class_byte_range: Option<PrismByteRange>,
    call: CallNode<'pr>,
}

impl<'a> InlineCollector<'a> {
    fn new(
        source: &'a [u8],
        line_index: LineIndex,
        file: Option<SourceFile<'a>>,
        comments: &'a CommentAssociation,
        names: &'a NameTable,
        inline_mode: bool,
    ) -> Self {
        InlineCollector {
            source,
            line_index,
            file,
            comments,
            names,
            class_stack: vec![],
            cref_stack: vec![],
            cref_scope_frames: vec![],
            top_level: vec![],
            object_members: vec![],
            scope_stack: vec![],
            diagnostics: vec![],
            associated_leading_lines: FxHashSet::default(),
            singleton_class_depth: 0,
            visibility_frames: vec![],
            pending_explicit_visibility: None,
            inline_mode,
        }
    }

    fn path(&self) -> Option<&'a Path> {
        self.file.map(|f| f.path)
    }

    fn file_name(&self) -> Option<Name> {
        self.file.map(|f| f.name)
    }

    /// Push a leaf member (def / attr / mixin) into the currently-open
    /// class or module scope.
    ///
    /// Callers must verify an enclosing scope exists before invoking —
    /// top-level leaves are skipped by the visitor itself via the
    /// `class_stack.is_empty()` guard, so in practice this is never
    /// reached without a scope.
    fn push_member(&mut self, m: Member) {
        self.scope_stack
            .last_mut()
            .expect("push_member called without an enclosing class/module scope")
            .push(m);
    }

    /// Attach a finished declaration to its parent: the member list of
    /// the enclosing *cref* (Ruby defines classes, modules and constants
    /// into the cref), or the top-level list when there is none. Inside
    /// a `Const = Class.new do ... end` block the innermost `scope_stack`
    /// frame is the block's class, which is not a cref frame, so the
    /// declaration is hoisted past it — the environment builder derives
    /// each declaration's resolution context from this tree, and a
    /// `::Widget` nested under `::Ctor` would resolve its body's names
    /// through `::Ctor` first.
    fn attach_declaration(&mut self, decl: Declaration) {
        match self.cref_scope_frames.last() {
            Some(&frame) => self.scope_stack[frame].push(Member::Declaration(decl)),
            None => self.top_level.push(decl),
        }
    }

    /// Wrap the collected top-level defs ([`object_members`]) into a
    /// synthetic `::Object` reopen and attach it to `top_level`. The
    /// decl is `block_body: true` because a top-level def's cref is the
    /// top level, not `Object` — its annotations must resolve in the
    /// enclosing (root) context exactly like a `Class.new do` body.
    /// `name_location` points at the first def's name: the reopen has
    /// no `class` keyword of its own to anchor to.
    fn finish_object_reopen(&mut self) {
        if self.object_members.is_empty() {
            return;
        }
        let mut members = std::mem::take(&mut self.object_members);
        members.sort_by_key(Member::location_start);
        let name_location = match &members[0] {
            Member::Def(def) => def.name_location,
            _ => unreachable!("object_members only holds Member::Def"),
        };
        self.top_level.push(Declaration::Class(Arc::new(ClassDecl {
            class_name: self.names.parse_type_name("::Object"),
            name_location,
            super_class: None,
            members,
            block_body: true,
        })));
    }

    /// Build an absolute class-kind [`TypeName`] for the currently-open
    /// declaration. Returns `None` at the top level. Mirrors the rbs
    /// `class_name` / `module_name` / `constant_name` shape required by
    /// `ast::ruby::declarations::*` — the inline collector qualifies
    /// every defining name against the lexical scope, so the result is
    /// always absolute (`namespace.is_absolute() == true`).
    fn current_class_typename(&self) -> Option<TypeName> {
        let abs_path = self.class_stack.last()?;
        Some(self.names.parse_type_name(abs_path))
    }

    fn diagnostic_location(&self, range: PrismByteRange) -> crate::location::SourceLocation {
        Diagnostic::location_for_byte_range(
            self.path().map(|p| p.to_path_buf()).unwrap_or_default(),
            self.source,
            range.0 as usize,
            range.1 as usize,
        )
    }

    fn prism_diagnostic_location(
        &self,
        location: ruby_prism::Location<'_>,
    ) -> crate::location::SourceLocation {
        self.diagnostic_location(prism_location_range(location))
    }

    fn collect_attr_call(&mut self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }

        let name = String::from_utf8_lossy(node.name().as_slice());
        let call_kind = match name.as_ref() {
            "attr_reader" => AttributeCallSiteKind::Reader,
            "attr_writer" => AttributeCallSiteKind::Writer,
            "attr_accessor" => AttributeCallSiteKind::Accessor,
            _ => return false,
        };

        // Top-level attr is skipped — the tree shape encodes the
        // enclosing scope via `attach_declaration` / `push_member`, so
        // we only need the empty-stack guard here.
        if self.class_stack.is_empty() {
            return true;
        }

        let byte_range = (
            node.location().start_offset() as u32,
            node.location().end_offset() as u32,
        );

        // Extract attr names from arguments (symbol nodes)
        let arguments = match node.arguments() {
            Some(args) => args,
            None => return true,
        };

        let mut name_nodes = Vec::new();
        for arg in arguments.arguments().iter() {
            if let Node::SymbolNode { .. } = &arg {
                let sym_text = arg.as_symbol_node().unwrap();
                let name = String::from_utf8_lossy(sym_text.unescaped()).to_string();
                let location = prism_location_range(sym_text.location());
                name_nodes.push(AttributeNameNode { name, location });
            }
        }

        if name_nodes.is_empty() {
            return true;
        }

        let end_line = self
            .line_index
            .line(node.location().end_offset().saturating_sub(1));
        let (type_text, annotation_range) =
            match self.comments.trailing_annotation(self.source, end_line) {
                Some(TrailingAnnotation::NodeTypeAssertion { range, type_text }) => {
                    (Some(type_text.to_string()), Some(range))
                }
                Some(TrailingAnnotation::TypeApplication { range, body }) => {
                    if let Err(err) = RbsParser::parse_inline_trailing(body.as_bytes()) {
                        self.diagnostics.push(build_annotation_syntax_error(
                            self.source,
                            self.path(),
                            range,
                            err,
                        ));
                    }
                    (None, None)
                }
                _ => (None, None),
            };

        let attribute = AttributeMember {
            visibility: self.effective_instance_visibility(),
            location: byte_range,
            name_nodes,
            type_text,
            annotation_range,
            source_file: self.file_name(),
        };
        if self.in_singleton_class() {
            self.push_singleton_attr_methods(&attribute, call_kind);
            return true;
        }

        let member = match call_kind {
            AttributeCallSiteKind::Reader => Member::AttrReader(AttrReaderMember { attribute }),
            AttributeCallSiteKind::Writer => Member::AttrWriter(AttrWriterMember { attribute }),
            AttributeCallSiteKind::Accessor => {
                Member::AttrAccessor(AttrAccessorMember { attribute })
            }
        };
        self.push_member(member);
        true
    }

    /// Visibility an instance-side def / attr pushed right now should
    /// carry: an explicit modifier under visit wins over the frame's
    /// ambient marker; `None` when neither applies.
    fn effective_instance_visibility(&self) -> Option<Visibility> {
        self.pending_explicit_visibility
            .or_else(|| self.visibility_frames.last().and_then(|f| f.ambient))
    }

    /// Recognize a receiver-less `private` / `public` / `protected`
    /// statement written directly in the open class body and fold it
    /// into Ruby-side member visibility (crema extension; rbs inline
    /// does not model visibility). Returns `false` to let the default
    /// visitor handle anything else (including the same calls nested in
    /// `if` / `begin`, inside `class << self`, or at the top level).
    ///
    /// `protected` is folded to `Public`: crema's `Visibility` has no
    /// protected state, and reporting a protected method as private
    /// would flag legitimate same-class explicit-receiver calls.
    ///
    /// Forms handled: bare (sets the ambient default for the rest of
    /// the body), `X def name` / `X attr_* :name` (modifier on the
    /// argument), and `X :a, :b` (retroactive on already-collected
    /// instance defs / single-name attrs). Anything else — `module_function`,
    /// `private_class_method`, array or string arguments, unknown names —
    /// is left alone, i.e. stays public.
    fn collect_visibility_call(&mut self, node: &CallNode<'_>) -> bool {
        let marker = match node.name().as_slice() {
            b"private" => Visibility::Private,
            b"public" | b"protected" => Visibility::Public,
            _ => return false,
        };
        if node.receiver().is_some() || self.class_stack.is_empty() || self.in_singleton_class() {
            return false;
        }
        let Some(frame) = self.visibility_frames.last() else {
            return false;
        };
        if !frame
            .direct_statement_starts
            .contains(&(node.location().start_offset() as u32))
        {
            return false;
        }

        let Some(arguments) = node.arguments() else {
            self.visibility_frames.last_mut().unwrap().ambient = Some(marker);
            self.mark_enclosed_leading_lines(node.location());
            return true;
        };
        for arg in arguments.arguments().iter() {
            if let Some(sym) = arg.as_symbol_node() {
                let name = String::from_utf8_lossy(sym.unescaped()).to_string();
                self.apply_retroactive_visibility(&name, marker);
            } else if arg.as_def_node().is_some() || is_attr_call(&arg) {
                let saved = self.pending_explicit_visibility.replace(marker);
                self.visit(&arg);
                self.pending_explicit_visibility = saved;
            } else {
                self.visit(&arg);
            }
        }
        true
    }

    /// `private :name` after the fact: rewrite the visibility of the
    /// matching instance def / attr already collected in the open body.
    /// Attr writers match on `name=`; multi-name attrs (`attr_reader :a,
    /// :b` then `private :a`) and accessors (`private :x` privatizes only
    /// the reader in Ruby, but the member carries one visibility for
    /// both) are left public rather than over-privatized. Unknown names
    /// are ignored.
    fn apply_retroactive_visibility(&mut self, name: &str, marker: Visibility) {
        let Some(members) = self.scope_stack.last_mut() else {
            return;
        };
        for member in members.iter_mut() {
            match member {
                Member::Def(def) if def.kind == MethodKind::Instance && def.name == name => {
                    def.visibility = Some(marker);
                }
                Member::AttrReader(r) if single_attr_name(&r.attribute) == Some(name) => {
                    r.attribute.visibility = Some(marker);
                }
                Member::AttrWriter(w)
                    if name
                        .strip_suffix('=')
                        .is_some_and(|base| single_attr_name(&w.attribute) == Some(base)) =>
                {
                    w.attribute.visibility = Some(marker);
                }
                _ => {}
            }
        }
    }

    fn collect_mixin_call(&mut self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }

        let name = String::from_utf8_lossy(node.name().as_slice());
        let call_kind = match name.as_ref() {
            "include" => MixinCallSiteKind::Include,
            "extend" => MixinCallSiteKind::Extend,
            "prepend" => MixinCallSiteKind::Prepend,
            _ => return false,
        };

        if self.class_stack.is_empty() {
            return true;
        }

        let arguments = match node.arguments() {
            Some(args) => args,
            None => return true,
        };

        if arguments.arguments().len() > 1 {
            self.diagnostics
                .push(self.mixin_multiple_arguments_diagnostic(node));
            return true;
        }

        // A trailing `#[T]` is only meaningful for single-argument mixin
        // calls (`include M #[T]`). For `include M1, M2 #[T]` the target
        // is ambiguous, so skip.
        let annotation = if arguments.arguments().len() == 1 {
            let call_end_line = self
                .line_index
                .line(node.location().end_offset().saturating_sub(1));
            match self
                .comments
                .trailing_annotation(self.source, call_end_line)
            {
                Some(TrailingAnnotation::TypeApplication { range, body }) => {
                    build_type_application_annotation(body, range, self.names)
                }
                _ => None,
            }
        } else {
            None
        };

        for arg in arguments.arguments().iter() {
            let module_name = if let Some(constant) = arg.as_constant_read_node() {
                String::from_utf8_lossy(constant.name().as_slice()).to_string()
            } else if let Some(path) = arg.as_constant_path_node() {
                constant_path_to_string(&path)
            } else {
                // Dynamic argument (variable, call, etc.) — skip
                continue;
            };

            let arg_loc = arg.location();
            let name_location = prism_location_range(arg_loc);
            let mixin = MixinMember {
                module_name,
                location: prism_location_range(node.location()),
                name_location,
                annotation: annotation.clone(),
            };
            let member = match (self.in_singleton_class(), call_kind) {
                (true, MixinCallSiteKind::Include | MixinCallSiteKind::Prepend) => {
                    Member::Extend(ExtendMember { mixin })
                }
                (true, MixinCallSiteKind::Extend) => continue,
                (false, MixinCallSiteKind::Include) => Member::Include(IncludeMember { mixin }),
                (false, MixinCallSiteKind::Extend) => Member::Extend(ExtendMember { mixin }),
                (false, MixinCallSiteKind::Prepend) => Member::Prepend(PrependMember { mixin }),
            };
            self.push_member(member);
        }
        true
    }

    fn is_infusion_owned_block_call(&self, node: &CallNode<'_>) -> bool {
        if node.receiver().is_some() {
            return false;
        }
        let name = String::from_utf8_lossy(node.name().as_slice());
        match name.as_ref() {
            "included" | "prepended" | "class_methods" => true,
            // Only claimed when the infusion collector can actually
            // synthesize the `<owner>::<Topic>` module — an enclosing
            // class/module plus a literal symbol/string topic (mirrors
            // `pipeline::collect_concerning_call`'s bail-outs; keep in
            // sync). Unclaimed calls fall through to the general block
            // descent so their `def`s still land somewhere instead of
            // vanishing.
            "concerning" => {
                !self.class_stack.is_empty()
                    && node
                        .arguments()
                        .and_then(|args| args.arguments().iter().next())
                        .is_some_and(|arg| {
                            arg.as_symbol_node().is_some() || arg.as_string_node().is_some()
                        })
            }
            _ => false,
        }
    }

    fn in_singleton_class(&self) -> bool {
        self.singleton_class_depth > 0
    }

    fn push_singleton_attr_methods(
        &mut self,
        attribute: &AttributeMember,
        call_kind: AttributeCallSiteKind,
    ) {
        for name in attribute.names() {
            let ty = self.singleton_attr_type(attribute);
            let location = attribute.location;
            match call_kind {
                AttributeCallSiteKind::Reader => self.push_member(generated_def(
                    name,
                    MethodKind::Singleton,
                    location,
                    vec![method_type_returning(ty)],
                )),
                AttributeCallSiteKind::Writer => self.push_member(generated_def(
                    &format!("{name}="),
                    MethodKind::Singleton,
                    location,
                    vec![method_type_with_required_positional_returning(
                        ty,
                        void_ast_type(),
                    )],
                )),
                AttributeCallSiteKind::Accessor => {
                    self.push_member(generated_def(
                        name,
                        MethodKind::Singleton,
                        location,
                        vec![method_type_returning(ty.clone())],
                    ));
                    self.push_member(generated_def(
                        &format!("{name}="),
                        MethodKind::Singleton,
                        location,
                        vec![method_type_with_required_positional_returning(
                            ty,
                            void_ast_type(),
                        )],
                    ));
                }
            }
        }
    }

    fn singleton_attr_type(&mut self, attribute: &AttributeMember) -> Type {
        let (Some(text), Some(range)) = (attribute.type_text.as_ref(), attribute.annotation_range)
        else {
            return untyped_ast_type();
        };
        match parse_rbs_type(text.as_bytes(), self.names) {
            Ok(ty) => ty,
            Err(err) => {
                self.diagnostics.push(build_annotation_syntax_error(
                    self.source,
                    self.path(),
                    range,
                    err,
                ));
                untyped_ast_type()
            }
        }
    }

    fn classify_value_kind(value: &Node<'_>) -> ConstantValueKind {
        match value {
            Node::IntegerNode { .. } => ConstantValueKind::Integer,
            Node::FloatNode { .. } => ConstantValueKind::Float,
            Node::StringNode { .. } => ConstantValueKind::String,
            Node::TrueNode { .. } => ConstantValueKind::True,
            Node::FalseNode { .. } => ConstantValueKind::False,
            Node::SymbolNode { .. } => ConstantValueKind::Symbol,
            Node::NilNode { .. } => ConstantValueKind::Nil,
            _ => ConstantValueKind::Other,
        }
    }

    fn data_struct_class_kind_and_name(
        &self,
        call: &CallNode<'_>,
    ) -> Option<(DataStructConstructionKind, TypeName)> {
        let kind = data_struct_construction_kind(call)?;
        let receiver = call.receiver()?;
        let receiver_name = receiver_node_static_name(&receiver)?;
        Some((kind, self.names.parse_type_name(&receiver_name)))
    }

    fn data_struct_class_call<'n>(&self, value: &Node<'n>) -> Option<DataStructClassCall<'n>> {
        // Wrapper peeling (single/multi-level, gate semantics) is centralized
        // in `crate::cast_unwrap`; the parser-side registration path takes
        // any lvar name because no RHS type-check is at stake.
        let unwrapped =
            crate::cast_unwrap::unwrap_cast(value, &crate::cast_unwrap::CastUnwrapConfig::PARSER)?;
        let (kind, super_class_name) = self.data_struct_class_kind_and_name(&unwrapped.call)?;
        Some(DataStructClassCall {
            kind,
            super_class_name,
            call: unwrapped.call,
        })
    }

    fn collect_data_struct_attributes(
        &self,
        call: &CallNode<'_>,
        kind: DataStructConstructionKind,
        options: &DataStructOptions,
    ) -> Vec<DataStructAttribute> {
        let Some(arguments) = call.arguments() else {
            return Vec::new();
        };
        let mut attributes = Vec::new();
        for arg in arguments.arguments().iter() {
            if matches!(kind, DataStructConstructionKind::Struct)
                && (arg.as_string_node().is_some() || arg.as_keyword_hash_node().is_some())
            {
                continue;
            }

            let Some(sym) = arg.as_symbol_node() else {
                continue;
            };
            let name = String::from_utf8_lossy(sym.unescaped()).to_string();
            let name_location = prism_location_range(sym.location());
            let arg_end_line = self
                .line_index
                .line(arg.location().end_offset().saturating_sub(1));
            let (type_text, annotation_range) =
                match self.comments.trailing_annotation(self.source, arg_end_line) {
                    Some(TrailingAnnotation::NodeTypeAssertion { range, type_text }) => {
                        (Some(type_text.to_string()), Some(range))
                    }
                    _ => (None, None),
                };
            let ty = type_text
                .as_deref()
                .and_then(|text| ast_builder::parse_trailing_type_text(text, self.names))
                .unwrap_or_else(untyped_ast_type);
            let attribute = AttributeMember {
                visibility: None,
                location: prism_location_range(arg.location()),
                name_nodes: vec![AttributeNameNode {
                    name: name.clone(),
                    location: name_location,
                }],
                type_text,
                annotation_range,
                source_file: self.file_name(),
            };
            let member = match kind {
                DataStructConstructionKind::Struct if options.readonly_attributes => {
                    Member::AttrReader(AttrReaderMember { attribute })
                }
                DataStructConstructionKind::Struct => {
                    Member::AttrAccessor(AttrAccessorMember { attribute })
                }
                DataStructConstructionKind::Data => {
                    Member::AttrReader(AttrReaderMember { attribute })
                }
            };
            attributes.push(DataStructAttribute { name, ty, member });
        }
        attributes
    }

    fn collect_data_struct_options(
        &self,
        call: &CallNode<'_>,
        decl_location: ruby_prism::Location<'_>,
    ) -> DataStructOptions {
        let mut options = DataStructOptions::default();
        if let Some(arguments) = call.arguments() {
            for arg in arguments.arguments().iter() {
                let Some(kw_hash) = arg.as_keyword_hash_node() else {
                    continue;
                };
                for elem in kw_hash.elements().iter() {
                    let Some(assoc) = elem.as_assoc_node() else {
                        continue;
                    };
                    let Some(sym) = assoc.key().as_symbol_node() else {
                        continue;
                    };
                    if String::from_utf8_lossy(sym.unescaped()) != "keyword_init" {
                        continue;
                    }
                    let value = assoc.value();
                    if value.as_true_node().is_some() {
                        options.keyword_init = Some(true);
                    } else if value.as_false_node().is_some() {
                        options.keyword_init = Some(false);
                    }
                }
            }
        }

        let start_line = self.line_index.line(decl_location.start_offset());
        if let Some(block) = self.comments.leading_block_for(start_line) {
            for annotation in rbs_inline_annotation_strings(block) {
                match annotation.as_str() {
                    "rbs-inline:new-args=required" => options.required_new_args = true,
                    "rbs-inline:readonly-attributes=true" => options.readonly_attributes = true,
                    _ => {}
                }
            }
        }
        options
    }

    fn data_struct_super_class(
        &self,
        super_class_name: TypeName,
        call: &CallNode<'_>,
        kind: DataStructConstructionKind,
        attributes: &[DataStructAttribute],
    ) -> SuperClass {
        let byte_range = prism_location_range(call.location());
        // Mirrors rbs-inline writer.rb:436-444: for `Const = Struct.new(...)`,
        // synthesize a single `Struct[E]` type argument by collecting each
        // attribute's annotated type and wrapping multiple types in a Union.
        // Skip the synthesis for Data (rbs core makes Data non-generic — any
        // arg would itself produce an arity error) and for empty Struct.new()
        // (no member types to feed E; leaving the annotation None lets the
        // validator's MixinTypeArgumentArityMismatch surface the mistake).
        // No `.uniq` dedup here — that belongs to the Ty-layer normalizer.
        //
        // The synthesized annotation has no `#[T]` source range to point at
        // (rbs-inline's writer.rb sets `location: nil`); we reuse `byte_range`
        // — same value as `SuperClass.byte_range` — so downstream diagnostics
        // that peek at the location still land on the `Struct.new(...)` call.
        let type_annotation = match kind {
            DataStructConstructionKind::Struct if !attributes.is_empty() => {
                let mut member_types: Vec<Type> =
                    attributes.iter().map(|attr| attr.ty.clone()).collect();
                let arg = if member_types.len() == 1 {
                    member_types.pop().unwrap()
                } else {
                    Type::Union(UnionType {
                        types: member_types,
                        location: None,
                    })
                };
                Some(TypeApplicationAnnotation {
                    type_args: vec![arg],
                    location: byte_range,
                })
            }
            _ => None,
        };
        SuperClass {
            type_name: super_class_name,
            type_annotation,
            byte_range: Some(byte_range),
        }
    }

    /// Recognize the `Const = Class.new(Parent?) do ... end` shape: a
    /// call whose receiver is bare or rooted `Class`, whose name is
    /// `new`, with zero or one bare/path constant parent argument. The
    /// block is optional — `Const = Class.new` and
    /// `Const = Class.new(Parent)` are accepted too. Reject every other
    /// shape (splat, keyword args, dynamic parent expression) so the
    /// constant falls through to the `ConstantDecl` default.
    ///
    /// Mirrors the type_checker-side `is_class_new_named_lhs_pattern` in
    /// `visitor.rs`, but relaxes the block requirement: the inline
    /// collector registers `Const` as a class for the no-block form too
    /// (with an empty body), so anonymous classes assigned to a constant
    /// gain the class kind regardless of whether a class body is
    /// supplied.
    fn class_new_call<'n>(&self, value: &Node<'n>) -> Option<ClassNewCall<'n>> {
        let call = class_dot_new_call(value)?;
        let (super_class_name, super_class_byte_range) = match call.arguments() {
            None => (self.names.parse_type_name("::Object"), None),
            Some(args) => {
                let mut iter = args.arguments().iter();
                match iter.next() {
                    None => (self.names.parse_type_name("::Object"), None),
                    Some(first) => {
                        if iter.next().is_some() {
                            return None;
                        }
                        let parent_name = if let Some(c) = first.as_constant_read_node() {
                            String::from_utf8_lossy(c.name().as_slice()).to_string()
                        } else if let Some(p) = first.as_constant_path_node() {
                            static_constant_path_string(&p)?
                        } else {
                            return None;
                        };
                        (
                            self.names.parse_type_name(&parent_name),
                            Some(prism_location_range(first.location())),
                        )
                    }
                }
            }
        };
        Some(ClassNewCall {
            super_class_name,
            super_class_byte_range,
            call,
        })
    }

    /// Walk a block body that should be treated as a class body — used
    /// by both `Const = Class.new(...) do ... end` and
    /// `Const = Struct.new(...)/Data.define(...) do ... end`. Pushes the
    /// class onto `class_stack` only — not `cref_stack`, since Ruby's
    /// `class_eval` on the block leaves the cref at the outer scope, so
    /// a bare `INNER = 1` inside registers as the outer scope's
    /// constant — resets `singleton_class_depth`, and
    /// dispatches the body through the default `Visit` traversal so
    /// `def`, `attr_*`, and `include`/`extend`/`prepend` calls register
    /// against the class. Also sweeps enclosed `@ivar: T` annotations
    /// over the outer call's location range, mirroring what
    /// `visit_class_node` does for real `class Const ... end` bodies.
    fn collect_class_body_block_members(
        &mut self,
        class_name: TypeName,
        call: &CallNode<'_>,
    ) -> Vec<Member> {
        // Skip the class_stack / scope_stack push entirely when there is
        // no block — callers (the `Vec::new()`-passing data_struct site
        // before this helper was introduced, and the bare
        // `Const = Class.new` shape) would otherwise pay push/pop +
        // `collect_enclosed_instance_variable_members` walk for nothing.
        let Some(block_arg) = call.block() else {
            return Vec::new();
        };
        let Some(block_node) = block_arg.as_block_node() else {
            return Vec::new();
        };

        let abs_path = self.names.resolve(class_name);
        self.class_stack.push(abs_path);
        let saved_singleton_class_depth = self.singleton_class_depth;
        self.singleton_class_depth = 0;
        self.scope_stack.push(Vec::new());
        self.visibility_frames
            .push(VisibilityFrame::new(block_node.body()));

        if let Some(body) = block_node.body() {
            self.visit(&body);
        }
        self.collect_enclosed_instance_variable_members(call.location());

        let members = self.scope_stack.pop().unwrap();
        self.visibility_frames.pop();
        self.singleton_class_depth = saved_singleton_class_depth;
        self.class_stack.pop();
        members
    }

    /// Build a `ClassDecl` for `Const = Class.new(Parent?) do ... end`.
    /// Block body extraction is delegated to
    /// [`collect_class_body_block_members`], shared with the Struct.new
    /// / Data.define paths. Members are sorted here (not in the helper)
    /// so the data_struct caller's own `sort_by_key` over the merged
    /// `attributes + generated + block` slice owns ordering and the
    /// helper's output stays unsorted (no double-sort).
    fn class_new_class_decl(
        &mut self,
        class_name: TypeName,
        name_location: PrismByteRange,
        class_new: ClassNewCall<'_>,
    ) -> ClassDecl {
        let mut members = self.collect_class_body_block_members(class_name, &class_new.call);
        members.sort_by_key(Member::location_start);
        ClassDecl {
            class_name,
            name_location,
            super_class: Some(SuperClass {
                type_name: class_new.super_class_name,
                type_annotation: None,
                byte_range: class_new.super_class_byte_range,
            }),
            members,
            block_body: true,
        }
    }

    fn data_struct_class_decl(
        &self,
        class_name: TypeName,
        name_location: PrismByteRange,
        decl_location: ruby_prism::Location<'_>,
        data_struct: DataStructClassCall<'_>,
        members: Vec<Member>,
    ) -> ClassDecl {
        let options = self.collect_data_struct_options(&data_struct.call, decl_location);
        let attributes =
            self.collect_data_struct_attributes(&data_struct.call, data_struct.kind, &options);
        let mut data_struct_members: Vec<Member> =
            attributes.iter().map(|attr| attr.member.clone()).collect();
        data_struct_members.extend(self.data_struct_generated_methods(
            data_struct.kind,
            &attributes,
            &options,
            data_struct.call.location(),
        ));
        data_struct_members.extend(members);
        let mut members = data_struct_members;
        members.sort_by_key(Member::location_start);
        ClassDecl {
            class_name,
            name_location,
            super_class: Some(self.data_struct_super_class(
                data_struct.super_class_name,
                &data_struct.call,
                data_struct.kind,
                &attributes,
            )),
            members,
            block_body: true,
        }
    }

    fn data_struct_generated_methods(
        &self,
        kind: DataStructConstructionKind,
        attributes: &[DataStructAttribute],
        options: &DataStructOptions,
        location: ruby_prism::Location<'_>,
    ) -> Vec<Member> {
        let mut members = Vec::new();
        let location = prism_location_range(location);
        match kind {
            DataStructConstructionKind::Struct => {
                let mut overloads = Vec::new();
                if options.keyword_init != Some(true) {
                    overloads.push(method_type_with_positionals(
                        attributes,
                        options.required_new_args,
                    ));
                }
                if options.keyword_init != Some(false) {
                    overloads.push(method_type_with_keywords(
                        attributes,
                        options.required_new_args,
                        self.names,
                    ));
                }
                if options.keyword_init == Some(true) {
                    overloads.push(method_type_with_required_record(attributes, self.names));
                }
                members.push(generated_def(
                    "new",
                    MethodKind::Singleton,
                    location,
                    overloads,
                ));
            }
            DataStructConstructionKind::Data => {
                members.push(generated_def(
                    "new",
                    MethodKind::Singleton,
                    location,
                    vec![
                        method_type_with_required_positionals(attributes),
                        method_type_with_required_keywords(attributes, self.names),
                    ],
                ));
                let members_method =
                    method_type_returning(symbol_tuple_type(attributes, self.names));
                members.push(generated_def(
                    "members",
                    MethodKind::Singleton,
                    location,
                    vec![members_method.clone()],
                ));
                members.push(generated_def(
                    "members",
                    MethodKind::Instance,
                    location,
                    vec![members_method],
                ));
            }
        }
        members
    }
}

impl<'pr, 'a> Visit<'pr> for InlineCollector<'a> {
    fn visit_class_node(&mut self, node: &ClassNode<'pr>) {
        if push_class_abs_path_under(
            &mut self.class_stack,
            self.cref_stack.last().map(String::as_str),
            &node.constant_path(),
        )
        .is_none()
        {
            let loc = node.constant_path().location();
            self.diagnostics.push(Diagnostic {
                scope: None,
                kind: DiagnosticKind::NonConstantClassName,
                location: self.prism_diagnostic_location(loc),
            });
            return;
        }
        self.cref_stack
            .push(self.class_stack.last().unwrap().clone());

        let class_name = self.current_class_typename().unwrap();
        let name_location = prism_location_range(node.constant_path().location());

        let data_struct_super = node
            .superclass()
            .and_then(|super_node| self.data_struct_class_call(&super_node));
        let mut type_annotation: Option<crate::ast::ruby::annotations::TypeApplicationAnnotation> =
            None;
        let mut byte_range: Option<PrismByteRange> = None;
        let raw_name = if let Some(super_node) = node.superclass() {
            let name = if let Some(data_struct) = data_struct_super.as_ref() {
                Some(self.names.resolve(data_struct.super_class_name))
            } else if let Some(constant) = super_node.as_constant_read_node() {
                Some(String::from_utf8_lossy(constant.name().as_slice()).to_string())
            } else {
                let static_path = super_node
                    .as_constant_path_node()
                    .and_then(|path| static_constant_path_string(&path));
                // Inline-mode-only (rbs `RBS::InlineParser` parity): in
                // sig mode the RBS declaration decides the superclass, so
                // a dynamic Ruby-side expression (`ActiveType::Record[User]`)
                // is not an error and drops silently.
                if static_path.is_none() && self.inline_mode {
                    self.diagnostics
                        .push(self.non_constant_super_class_diagnostic(&super_node));
                }
                static_path
            };
            if name.is_some() && data_struct_super.is_none() {
                let loc = super_node.location();
                byte_range = Some((loc.start_offset() as u32, loc.end_offset() as u32));
                let super_end_line = self
                    .line_index
                    .line(super_node.location().end_offset().saturating_sub(1));
                if let Some(TrailingAnnotation::TypeApplication { range, body }) = self
                    .comments
                    .trailing_annotation(self.source, super_end_line)
                {
                    type_annotation = build_type_application_annotation(body, range, self.names);
                }
            }
            name
        } else {
            None
        };
        // SuperClass.type_name preserves the source-form absoluteness so
        // `resolve_ruby_class_recursive` can distinguish "Bar" (relative
        // lookup against the enclosing context) from "::Foo" (already
        // absolute). `TypeName::parse` reads the leading `::` to set
        // `namespace.is_absolute()` accordingly.
        let super_class = raw_name.map(|raw_name| SuperClass {
            type_name: self.names.parse_type_name(&raw_name),
            type_annotation,
            byte_range,
        });

        // Open a fresh member list, walk the class body into it, then
        // wrap the list in a ClassDecl and attach to the parent.
        let saved_singleton_class_depth = self.singleton_class_depth;
        self.singleton_class_depth = 0;
        self.scope_stack.push(Vec::new());
        self.visibility_frames
            .push(VisibilityFrame::new(node.body()));
        self.cref_scope_frames.push(self.scope_stack.len() - 1);
        ruby_prism::visit_class_node(self, node);
        self.collect_enclosed_instance_variable_members(node.location());
        let mut members = self.scope_stack.pop().unwrap();
        self.visibility_frames.pop();
        self.cref_scope_frames.pop();
        self.singleton_class_depth = saved_singleton_class_depth;
        members.sort_by_key(Member::location_start);
        self.class_stack.pop();
        self.cref_stack.pop();

        let class_decl = if let Some(data_struct) = data_struct_super {
            self.data_struct_class_decl(
                class_name,
                name_location,
                node.location(),
                data_struct,
                members,
            )
        } else {
            ClassDecl {
                class_name,
                name_location,
                super_class,
                members,
                block_body: false,
            }
        };

        self.attach_declaration(Declaration::Class(Arc::new(class_decl)));
    }

    fn visit_module_node(&mut self, node: &ModuleNode<'pr>) {
        if push_class_abs_path_under(
            &mut self.class_stack,
            self.cref_stack.last().map(String::as_str),
            &node.constant_path(),
        )
        .is_none()
        {
            let loc = node.constant_path().location();
            self.diagnostics.push(Diagnostic {
                scope: None,
                kind: DiagnosticKind::NonConstantModuleName,
                location: self.prism_diagnostic_location(loc),
            });
            return;
        }
        self.cref_stack
            .push(self.class_stack.last().unwrap().clone());

        let module_name = self.current_class_typename().unwrap();
        let name_location = prism_location_range(node.constant_path().location());
        let module_start_line = self.line_index.line(node.location().start_offset());

        let saved_singleton_class_depth = self.singleton_class_depth;
        self.singleton_class_depth = 0;
        self.scope_stack.push(Vec::new());
        self.visibility_frames
            .push(VisibilityFrame::new(node.body()));
        self.cref_scope_frames.push(self.scope_stack.len() - 1);
        self.collect_leading_module_self_members(module_start_line);
        ruby_prism::visit_module_node(self, node);
        self.collect_enclosed_instance_variable_members(node.location());
        let mut members = self.scope_stack.pop().unwrap();
        self.visibility_frames.pop();
        self.cref_scope_frames.pop();
        self.singleton_class_depth = saved_singleton_class_depth;
        members.sort_by_key(Member::location_start);
        self.class_stack.pop();
        self.cref_stack.pop();

        self.attach_declaration(Declaration::Module(Arc::new(ModuleDecl {
            module_name,
            name_location,
            members,
        })));
    }

    fn visit_def_node(&mut self, node: &DefNode<'pr>) {
        if self.is_node_skipped(node.location()) {
            return;
        }

        let kind = match (self.in_singleton_class(), node.receiver()) {
            (true, None) => MethodKind::Singleton,
            (true, Some(receiver)) if receiver.as_self_node().is_some() => {
                if self.inline_mode {
                    let loc = receiver.location();
                    self.diagnostics.push(
                        self.singleton_scope_diagnostic(loc, DiagnosticKind::NestedSingletonScope),
                    );
                }
                let def_start_line = self.line_index.line(node.location().start_offset());
                let leading_comment = collect_consecutive_leading(self.comments, def_start_line);
                self.mark_comment_block_associated(leading_comment.as_ref());
                self.mark_enclosed_leading_lines(node.location());
                return;
            }
            (true, Some(_)) => return,
            (false, Some(receiver)) if receiver.as_self_node().is_some() => MethodKind::Singleton,
            (false, Some(_)) => return,
            (false, None) => MethodKind::Instance,
        };

        // Top-level `def name` in inline mode is collected as a private
        // `::Object` instance method (Ruby semantics; see
        // `object_members`). Everything else at the top level is
        // dropped: `def self.name` is a singleton method of `main` that
        // RBS cannot represent, so inline mode reports
        // `TopLevelMethodDefinition` (rbs `RBS::InlineParser` parity);
        // sig mode drops silently and the type checker binds the def
        // against the `::Object` sig in `lookup_method_target`.
        let top_level = self.class_stack.is_empty();
        if top_level && !(self.inline_mode && kind == MethodKind::Instance) {
            let def_start_line = self.line_index.line(node.location().start_offset());
            let leading_comment = collect_consecutive_leading(self.comments, def_start_line);
            self.report_unused_comment_block(leading_comment.as_ref());
            if self.inline_mode {
                self.diagnostics
                    .push(self.top_level_method_definition_diagnostic(node));
            }
            return;
        }

        let name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let location = prism_location_range(node.location());
        let name_location = prism_location_range(node.name_loc());

        let def_start_line = self.line_index.line(node.location().start_offset());
        let leading_comment = collect_consecutive_leading(self.comments, def_start_line);
        self.mark_comment_block_associated(leading_comment.as_ref());
        self.mark_enclosed_leading_lines(node.location());

        // Build a single-line `CommentBlock` for the def's header line
        // (the line carrying the `def` keyword and the signature) so
        // `MethodTypeAnnotation::build` can classify the trailing
        // annotation (`def foo(...) #: T`) the same way it consumes
        // the leading block. Mirrors rbs's `trailing_block` argument.
        //
        // `node.location().end_offset()` points at the `end` keyword on
        // multi-line defs, so using it here would miss header-line
        // trailing comments. The start-line approximation handles the
        // common single-line-header case; multi-line signatures whose
        // closing `)` lands on a later line are a follow-up case.
        let trailing_block = collect_trailing_block(self.source, self.comments, def_start_line);

        let (method_type, unused_leading, trailing_resolution) = MethodTypeAnnotation::build(
            leading_comment.as_ref(),
            trailing_block.as_ref(),
            &[],
            node,
            self.source,
            self.names,
        );
        self.report_unused_leading(unused_leading);
        match trailing_resolution {
            TrailingResolution::ParseFailed {
                range,
                parser_error,
            } => {
                // Empty `#:` on a def trailing return surfaces as
                // `AnnotationSyntaxError` (Steep parity with
                // `RBS::InlineDiagnostic`), not the generic
                // `UnusedInlineAnnotation` that would fire for an
                // annotation that happened to be skipped.
                let diag =
                    build_annotation_syntax_error(self.source, self.path(), range, parser_error);
                self.diagnostics.push(diag);
            }
            TrailingResolution::Unused(trailing) => {
                self.report_unused_trailing(trailing);
            }
            TrailingResolution::None => {}
        }

        // Instance-side `initialize` bodies get one extra shallow scan for
        // `@ivar = param` facts (Sorbet-style instance variable inference,
        // crema extension over the rbs skeleton contract). Every other def
        // keeps the body un-walked. A top-level `def initialize` is never
        // called as a constructor, and synthesizing ivars onto `::Object`
        // would leak them into every class, so it stays empty.
        let ivar_param_pairs = if kind == MethodKind::Instance && name == "initialize" && !top_level
        {
            collect_initialize_ivar_param_pairs(node)
        } else {
            Vec::new()
        };

        let visibility = if top_level {
            // Top-level defs are always private on `Object`. A bare
            // `public` at the top level is not interpreted.
            Some(Visibility::Private)
        } else if kind == MethodKind::Instance {
            self.effective_instance_visibility()
        } else {
            // Ruby's `private` never reaches `def self.x` (the modifier
            // form receives the *instance* method name `:x`), so the
            // singleton side carries no Ruby-derived visibility.
            None
        };

        let member = Member::Def(DefMember {
            visibility,
            name,
            kind,
            location,
            name_location,
            method_type,
            leading_comment,
            origin: DefMemberOrigin::Real,
            source_file: self.file_name(),
            ivar_param_pairs,
        });
        if top_level {
            self.object_members.push(member);
        } else {
            self.push_member(member);
        }

        // Don't recurse into def body — not needed for skeleton extraction
    }

    fn visit_call_node(&mut self, node: &CallNode<'pr>) {
        if node.receiver().is_some() {
            self.mark_enclosed_leading_lines(node.location());
            return;
        }
        if self.is_infusion_owned_block_call(node) {
            self.mark_enclosed_leading_lines(node.location());
            return;
        }
        if self.collect_attr_call(node) || self.collect_mixin_call(node) {
            self.mark_enclosed_leading_lines(node.location());
            return;
        }
        if self.collect_visibility_call(node) {
            return;
        }
        ruby_prism::visit_call_node(self, node);
    }

    fn visit_singleton_class_node(&mut self, node: &SingletonClassNode<'pr>) {
        if self.is_node_skipped(node.location()) {
            return;
        }
        if self.in_singleton_class() {
            self.mark_enclosed_leading_lines(node.location());
            if self.inline_mode {
                self.diagnostics.push(self.singleton_scope_diagnostic(
                    node.location(),
                    DiagnosticKind::NestedSingletonScope,
                ));
            }
            return;
        }
        if node.expression().as_self_node().is_none() {
            self.mark_enclosed_leading_lines(node.location());
            if self.inline_mode {
                self.diagnostics.push(self.singleton_scope_diagnostic(
                    node.expression().location(),
                    DiagnosticKind::NonSelfSingletonScope,
                ));
            }
            return;
        }
        if self.class_stack.is_empty() {
            self.mark_enclosed_leading_lines(node.location());
            if self.inline_mode {
                self.diagnostics.push(self.singleton_scope_diagnostic(
                    node.location(),
                    DiagnosticKind::TopLevelSingletonScope,
                ));
            }
            return;
        }

        self.singleton_class_depth += 1;
        ruby_prism::visit_singleton_class_node(self, node);
        self.singleton_class_depth -= 1;
    }

    fn visit_constant_write_node(&mut self, node: &ConstantWriteNode<'pr>) {
        if self.is_node_skipped(node.location()) {
            return;
        }
        // `class << self; S = 1` defines `S` on the singleton class
        // (Ruby cref), which RBS cannot declare — drop it like the
        // other singleton-scope shapes, with the inline-only diagnostic.
        // rbs's `InlineParser` registers it as `Foo::S` only because it
        // does not track `class << self` at all (docs/inline.md); crema
        // already diverges there for `def` (singleton method kind).
        if self.in_singleton_class() {
            self.mark_enclosed_leading_lines(node.location());
            if self.inline_mode {
                self.diagnostics.push(self.singleton_scope_diagnostic(
                    node.location(),
                    DiagnosticKind::SingletonScopeConstantDefinition,
                ));
            }
            return;
        }
        let const_name = String::from_utf8_lossy(node.name().as_slice()).to_string();
        let qualified_path = qualify_under(self.cref_stack.last().map(String::as_str), &const_name);
        let qualified_name = self.names.parse_type_name(&qualified_path);
        self.process_constant_assignment(
            ConstantAssignmentName {
                name: qualified_name,
                location: prism_location_range(node.name_loc()),
            },
            node.value(),
            node.location(),
        );
    }

    fn visit_constant_path_write_node(&mut self, node: &ConstantPathWriteNode<'pr>) {
        if self.is_node_skipped(node.location()) {
            return;
        }
        let target = node.target();
        let path = constant_path_to_string(&target);
        let qualified_path = if path.starts_with("::") {
            path
        } else {
            qualify_under(self.class_stack.last().map(String::as_str), &path)
        };
        let qualified_name = self.names.parse_type_name(&qualified_path);
        self.process_constant_assignment(
            ConstantAssignmentName {
                name: qualified_name,
                location: prism_location_range(target.location()),
            },
            node.value(),
            node.location(),
        );
    }
}

impl<'a> InlineCollector<'a> {
    /// Returns true when the node's leading block contains a `# @rbs skip`
    /// paragraph. Mirrors rbs's `skip_node?` behavior for constant
    /// assignments — skipped nodes drop out of the AST entirely (no
    /// decl, no diagnostic). Plain comments and other annotation kinds
    /// in the same leading block are silently ignored.
    fn is_node_skipped(&self, node_location: ruby_prism::Location<'_>) -> bool {
        let start_line = self.line_index.line(node_location.start_offset());
        let Some(block) = self.comments.leading_block_for(start_line) else {
            return false;
        };
        block
            .each_paragraph(&[], self.names)
            .iter()
            .any(|a| matches!(a, LeadingAnnotation::Skip(_)))
    }

    /// Common entry for constant write nodes — dispatches between the
    /// regular `ConstantDecl` path and the class/module alias path based
    /// on the trailing inline annotation, mirroring rbs's
    /// `RBS::InlineParser#parse_constant_declaration`.
    fn process_constant_assignment(
        &mut self,
        constant_name: ConstantAssignmentName,
        value: Node<'_>,
        node_location: ruby_prism::Location<'_>,
    ) {
        let end_line = self
            .line_index
            .line(node_location.end_offset().saturating_sub(1));
        let trailing = self.comments.trailing_annotation(self.source, end_line);

        let byte_range = (
            node_location.start_offset() as u32,
            node_location.end_offset() as u32,
        );

        match trailing {
            Some(TrailingAnnotation::ClassAlias {
                range,
                type_name_text,
            }) => self.collect_alias(
                InlineAliasKind::Class,
                constant_name,
                &value,
                range,
                type_name_text,
                byte_range,
            ),
            Some(TrailingAnnotation::ModuleAlias {
                range,
                type_name_text,
            }) => self.collect_alias(
                InlineAliasKind::Module,
                constant_name,
                &value,
                range,
                type_name_text,
                byte_range,
            ),
            Some(TrailingAnnotation::NodeTypeAssertion { range, type_text }) => {
                self.attach_declaration(Declaration::Constant(ConstantDecl {
                    constant_name: constant_name.name,
                    name_location: constant_name.location,
                    type_text: Some(type_text.to_string()),
                    annotation_range: Some(range),
                    value_kind: Self::classify_value_kind(&value),
                    leading_comment: None,
                }));
            }
            _ => {
                if let Some(data_struct) = self.data_struct_class_call(&value) {
                    let block_members = self
                        .collect_class_body_block_members(constant_name.name, &data_struct.call);
                    self.attach_declaration(Declaration::Class(Arc::new(
                        self.data_struct_class_decl(
                            constant_name.name,
                            constant_name.location,
                            node_location,
                            data_struct,
                            block_members,
                        ),
                    )));
                } else if let Some(class_new) = self.class_new_call(&value) {
                    let class_decl = self.class_new_class_decl(
                        constant_name.name,
                        constant_name.location,
                        class_new,
                    );
                    self.attach_declaration(Declaration::Class(Arc::new(class_decl)));
                } else {
                    self.attach_declaration(Declaration::Constant(ConstantDecl {
                        constant_name: constant_name.name,
                        name_location: constant_name.location,
                        type_text: None,
                        annotation_range: None,
                        value_kind: Self::classify_value_kind(&value),
                        leading_comment: None,
                    }));
                }
            }
        }
    }

    fn collect_alias(
        &mut self,
        kind: InlineAliasKind,
        new_name: ConstantAssignmentName,
        value: &Node<'_>,
        annotation_range: PrismByteRange,
        type_name_text: Option<&str>,
        byte_range: PrismByteRange,
    ) {
        // `infered_old_name` preserves the source-form absoluteness from
        // the right-hand-side constant path (e.g. `::Foo` stays absolute,
        // `Bar` stays relative). `resolve_ruby_alias_decl` absolutizes it
        // against the enclosing context later.
        let infered_old_name =
            value_node_as_constant_path(value).map(|raw| self.names.parse_type_name(&raw));
        let explicit_type_name_text = type_name_text.map(|s| s.to_string());

        if explicit_type_name_text.is_none() && infered_old_name.is_none() {
            self.diagnostics
                .push(self.alias_missing_type_name_diagnostic(kind, annotation_range));
            return;
        }

        let old_name_location = self
            .file_name()
            .map(|file_name| crate::location::RubyLocation {
                file: file_name,
                start_byte: byte_range.0,
                end_byte: byte_range.1,
            });
        let annotation = crate::ast::ruby::annotations::AliasAnnotation::new(
            kind,
            crate::ast::ruby::annotations::AliasAnnotationFields {
                location: annotation_range,
                type_name_text: explicit_type_name_text,
            },
        );
        let alias_decl = ClassModuleAliasDecl {
            new_name: new_name.name,
            name_location: new_name.location,
            infered_old_name,
            annotation,
            byte_range: Some(byte_range),
            old_name_location,
            leading_comment: None,
        };
        self.attach_declaration(Declaration::ClassModuleAlias(alias_decl));
    }

    fn alias_missing_type_name_diagnostic(
        &self,
        kind: InlineAliasKind,
        annotation_range: PrismByteRange,
    ) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::InlineClassAliasMissingTypeName { kind },
            location: self.diagnostic_location(annotation_range),
        }
    }

    fn mixin_multiple_arguments_diagnostic(&self, node: &CallNode<'_>) -> Diagnostic {
        let range = prism_location_range(node.location());
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::MixinMultipleArguments,
            location: self.diagnostic_location(range),
        }
    }

    fn top_level_method_definition_diagnostic(&self, node: &DefNode<'_>) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::TopLevelMethodDefinition,
            location: self.prism_diagnostic_location(node.name_loc()),
        }
    }

    fn non_constant_super_class_diagnostic(&self, super_node: &Node<'_>) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::NonConstantSuperClassName,
            location: self.prism_diagnostic_location(super_node.location()),
        }
    }

    fn singleton_scope_diagnostic(
        &self,
        location: ruby_prism::Location<'_>,
        kind: DiagnosticKind,
    ) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind,
            location: self.prism_diagnostic_location(location),
        }
    }

    fn unused_annotation_diagnostic(&self, range: PrismByteRange) -> Diagnostic {
        Diagnostic {
            scope: None,
            kind: DiagnosticKind::UnusedInlineAnnotation,
            location: self.diagnostic_location(range),
        }
    }

    fn report_unused_leading(&mut self, unused: Vec<LeadingAnnotation>) {
        for a in unused {
            let diag = self.unused_annotation_diagnostic(a.location());
            self.diagnostics.push(diag);
        }
    }

    fn report_unused_trailing(&mut self, trailing: TrailingAnnotation<'_>) {
        let diag = self.unused_annotation_diagnostic(trailing.range());
        self.diagnostics.push(diag);
    }

    fn report_unused_comment_block(&mut self, block: Option<&CommentBlock>) {
        let Some(block) = block else {
            return;
        };
        for annotation in block.each_paragraph(&[], self.names) {
            let diag = self.unused_annotation_diagnostic(annotation.location());
            self.diagnostics.push(diag);
        }
    }
}

/// Extract the constant-path text from an assignment's right-hand side
/// when the value is a `ConstantReadNode` (`Foo`) or a fully-static
/// `ConstantPathNode` (`Foo::Bar`, `::Foo`). Returns `None` for any
/// dynamic path (e.g. `factory.const`) or non-constant expression —
/// these defeat alias inference and require the explicit
/// `#: class-alias Foo` form. Mirrors rbs's
/// `Helpers::ConstantHelper#constant_as_type_name`, which yields nil
/// for any non-constant receiver.
fn value_node_as_constant_path(value: &Node<'_>) -> Option<String> {
    if let Some(constant) = value.as_constant_read_node() {
        return Some(String::from_utf8_lossy(constant.name().as_slice()).to_string());
    }
    if let Some(path) = value.as_constant_path_node() {
        return static_constant_path_string(&path);
    }
    None
}

fn receiver_node_static_name(value: &Node<'_>) -> Option<String> {
    if let Some(constant) = value.as_constant_read_node() {
        return Some(String::from_utf8_lossy(constant.name().as_slice()).to_string());
    }
    if let Some(path) = value.as_constant_path_node() {
        return static_constant_path_string(&path);
    }
    None
}

/// The source-form name of a class / module declaration as a path string:
/// the leaf for a `ConstantReadNode` (`class Foo` → "Foo"), the full dotted
/// path for a static `ConstantPathNode` (`class A::B` → "A::B", rooted
/// `class ::A::B` → "::A::B"). Returns `None` for a dynamic path
/// (`class factory.const::Bar`); mirrors rbs `constant_as_type_name`
/// returning nil on `DynamicPartsInConstantPathError`. Shared with the
/// type checker so both layers derive the class-stack key identically.
pub(crate) fn class_decl_path_string(constant_path: &Node<'_>) -> Option<String> {
    if let Some(c) = constant_path.as_constant_read_node() {
        Some(String::from_utf8_lossy(c.name().as_slice()).to_string())
    } else if let Some(p) = constant_path.as_constant_path_node() {
        static_constant_path_string(&p)
    } else {
        None
    }
}

/// Returns `None` for dynamic constant paths (call node, etc.); mirrors rbs
/// `constant_as_type_name` → nil on `DynamicPartsInConstantPathError`.
/// After a successful push every element of `stack` is an absolute path
/// string with a leading `::`.  Rooted paths (`class ::A::B`) ignore nesting.
pub(crate) fn push_class_abs_path(stack: &mut Vec<String>, constant_path: &Node<'_>) -> Option<()> {
    let enclosing = stack.last().cloned();
    push_class_abs_path_under(stack, enclosing.as_deref(), constant_path)
}

/// [`push_class_abs_path`] with an explicit `enclosing` cref: usually
/// `stack.last()`, but inside a `Const = Class.new do ... end` block the
/// inline collector passes the block's *outer* scope (Ruby's `class`
/// keyword defines into the cref, which `class_eval` leaves alone).
fn push_class_abs_path_under(
    stack: &mut Vec<String>,
    enclosing: Option<&str>,
    constant_path: &Node<'_>,
) -> Option<()> {
    let path_str = class_decl_path_string(constant_path)?;
    let abs = if path_str.starts_with("::") {
        path_str
    } else {
        qualify_under(enclosing, &path_str)
    };
    stack.push(abs);
    Some(())
}

/// Qualify `name` under `enclosing` (the current absolute class path, or
/// `None` at top level), returning an absolute `::…` path string.
fn qualify_under(enclosing: Option<&str>, name: &str) -> String {
    match enclosing {
        None => format!("::{}", name),
        Some(parent) => format!("{}::{}", parent, name),
    }
}

/// Match the call's receiver against `Class` (bare) or `::Class`
/// (rooted). A nested path like `Foo::Class` is rejected — only the
/// top-level `Class` constant counts as the syntactic anchor for the
/// `Class.new` pattern. Mirrors the type_checker-side helper of the
/// same name (the two passes need to agree on what shape qualifies as
/// "an anonymous class assigned to a constant").
/// Same shape as [`constant_path_to_string`] but rejects any path whose
/// chain contains a non-constant receiver (e.g. `factory.const`).
fn static_constant_path_string(node: &ruby_prism::ConstantPathNode<'_>) -> Option<String> {
    let child_name = String::from_utf8_lossy(node.name()?.as_slice()).to_string();
    match node.parent() {
        Some(parent) => {
            if let Some(constant) = parent.as_constant_read_node() {
                let parent_name = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                Some(format!("{}::{}", parent_name, child_name))
            } else if let Some(path) = parent.as_constant_path_node() {
                Some(format!(
                    "{}::{}",
                    static_constant_path_string(&path)?,
                    child_name
                ))
            } else {
                None
            }
        }
        None => Some(format!("::{}", child_name)),
    }
}

/// Build a string from a ConstantPathNode (e.g. "Foo::Bar::Baz").
fn constant_path_to_string(node: &ruby_prism::ConstantPathNode<'_>) -> String {
    let child_name = String::from_utf8_lossy(node.name().unwrap().as_slice()).to_string();
    match node.parent() {
        Some(parent) => {
            if let Some(constant) = parent.as_constant_read_node() {
                let parent_name = String::from_utf8_lossy(constant.name().as_slice()).to_string();
                format!("{}::{}", parent_name, child_name)
            } else if let Some(path) = parent.as_constant_path_node() {
                format!("{}::{}", constant_path_to_string(&path), child_name)
            } else {
                child_name
            }
        }
        None => {
            // Rooted path (::Foo) — starts with ::
            format!("::{}", child_name)
        }
    }
}

/// Pass 1: Extract inline annotations and untyped skeleton from Ruby
/// source and push them into the unresolved [`LegacyEnvironmentBuffer`].
///
/// `file` identifies the Ruby source file for `Location` tracking. Pass
/// `None` for tests or when the source does not originate from a file
/// on disk.
///
/// This entry point is responsible for the Prism-side collection only —
/// it walks the source into a tree of [`Declaration`]s and then hands
/// each top-level declaration to [`EnvironmentDraft::insert_ruby_decl`].
pub fn load_inline_annotations(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<&Path>,
    draft: &mut EnvironmentDraft,
) -> Vec<Diagnostic> {
    load_inline_annotations_impl(source, parse_result, file, draft, true)
}

pub fn parse_inline_annotations(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<&Path>,
    draft: &mut EnvironmentDraft,
) -> Vec<Diagnostic> {
    load_inline_annotations_impl(source, parse_result, file, draft, false)
}

/// Same diagnostics [`load_inline_annotations`] would emit for `source` —
/// including inline-mode-only diagnostics like `TopLevelMethodDefinition`
/// (see `load_inline_annotations_impl`'s `inline_mode` doc) — without
/// inserting anything into a draft (`names` is read-only, for interning
/// lookups during collection). ADR-0028 S8: a warm `crema check` run needs
/// this to reproduce an *unchanged* file's diagnostics, whose declarations
/// already live in the decoded A-snapshot state and must not be re-inserted.
pub fn inline_diagnostics_only(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<&Path>,
    names: &crate::name::NameTable,
) -> Vec<Diagnostic> {
    if parse_result.errors().next().is_some() {
        return Vec::new();
    }
    let file = file.map(|path| SourceFile::intern(path, names));
    collect_inline_declarations(source, parse_result, file, names, true).1
}

fn load_inline_annotations_impl(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<&Path>,
    draft: &mut EnvironmentDraft,
    // `true` for `load_inline_annotations` (inline mode = `--inline=true`),
    // `false` for `parse_inline_annotations` (sig mode). This one flag
    // carries two semantically-linked decisions: (1) whether to insert
    // the collected declarations into the draft (inline mode owns them),
    // and (2) whether to emit inline-mode-only restriction diagnostics
    // such as `TopLevelMethodDefinition` (rbs `RBS::InlineParser`
    // parity). Both live on the same axis in the CLI (`Config::inline`),
    // so passing a single bool is intentional — the doc-comment pins
    // that assumption so a future callsite that wants one behavior
    // without the other has to split the flag rather than silently drift.
    inline_mode: bool,
) -> Vec<Diagnostic> {
    // Files whose `parse_result` carries any Prism error are treated
    // as out-of-scope at the library boundary too: walking the
    // recovery AST would push half-formed declarations into the
    // draft, polluting the merged Environment that other files
    // resolve against. The CLI driver detects parse errors first and
    // emits `Ruby::SyntaxError` itself; library callers get an empty
    // `Vec` here and the draft is left untouched. This intentionally
    // diverges from rbs gem's `InlineParser`, which extracts
    // declarations from prism's recovery AST without consulting
    // `prism.errors` — crema's stance is "Ruby rejects -> crema also
    // rejects".
    if parse_result.errors().next().is_some() {
        return Vec::new();
    }
    let source_file = file.map(|path| SourceFile::intern(path, draft.names()));
    let (top_level, mut diagnostics) = collect_inline_declarations(
        source,
        parse_result,
        source_file,
        draft.names(),
        inline_mode,
    );

    if inline_mode {
        for decl in &top_level {
            draft.insert_ruby_decl(decl, source, file, &mut diagnostics);
        }
    }
    diagnostics
}

/// Run only the inline collector and hand back the raw top-level
/// declaration tree plus any collection-time diagnostics. Used by
/// [`load_inline_annotations`] above and by tests that need to observe
/// the pre-resolution form (e.g. that `SuperClass.type_name` is
/// relative before `EnvironmentDraft::build` runs).
pub fn collect_inline_declarations(
    source: &[u8],
    parse_result: &ruby_prism::ParseResult<'_>,
    file: Option<SourceFile<'_>>,
    names: &NameTable,
    inline_mode: bool,
) -> (Vec<Declaration>, Vec<Diagnostic>) {
    let line_index = LineIndex::from_source(source);
    let comments = CommentAssociation::from_source(source, &line_index, parse_result);
    let root = parse_result.node();
    let mut collector =
        InlineCollector::new(source, line_index, file, &comments, names, inline_mode);
    collector.visit(&root);
    collector.finish_object_reopen();
    (collector.top_level, collector.diagnostics)
}

/// Leading `CommentBlock` immediately above the line that starts a def.
/// Delegates to the rbs-port block pre-built by `CommentAssociation`:
/// the block contains every paragraph in declaration order (plain
/// comments and annotation paragraphs alike), so a plain comment
/// between the annotation and the `def` no longer truncates the block.
/// Mirrors rbs's `comments.leading_block(def_node)` query.
pub(crate) fn collect_consecutive_leading(
    comments: &CommentAssociation,
    def_start_line: usize,
) -> Option<CommentBlock> {
    comments.leading_block_for(def_start_line).cloned()
}

/// Wrap a def's trailing comment line as a single-entry `CommentBlock`.
///
/// `MethodTypeAnnotation::build` takes `Option<&CommentBlock>` for the
/// trailing slot to mirror rbs; the def's own line either has one
/// trailing comment (`#: T` / `#[T]`) or none.
pub(crate) fn collect_trailing_block(
    source: &[u8],
    comments: &CommentAssociation,
    def_end_line: usize,
) -> Option<CommentBlock> {
    let (start, end) = comments.trailing_range(def_end_line)?;
    let raw = source.get(start as usize..end as usize)?;
    let text = std::str::from_utf8(raw).ok()?.to_string();
    CommentBlock::new(vec![CommentLine {
        location: (start, end),
        text,
    }])
}

fn untyped_ast_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Any { todo: false },
        location: None,
    })
}

fn void_ast_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Void,
        location: None,
    })
}

fn self_instance_ast_type() -> Type {
    Type::Base(BaseType {
        kind: BaseTypeKind::Instance,
        location: None,
    })
}

fn ast_param(ty: Type, name: Option<crate::name::Symbol>) -> FunctionParam {
    FunctionParam {
        ty: Box::new(ty),
        name,
        location: None,
    }
}

fn method_type_returning(return_type: Type) -> AstMethodType {
    AstMethodType {
        type_params: vec![],
        function: Function::Typed(FunctionType {
            required_positionals: vec![],
            optional_positionals: vec![],
            rest_positionals: None,
            trailing_positionals: vec![],
            required_keywords: vec![],
            optional_keywords: vec![],
            rest_keywords: None,
            return_type: Box::new(return_type),
        }),
        block: None,
        location: None,
    }
}

fn method_type_with_required_positionals(attributes: &[DataStructAttribute]) -> AstMethodType {
    let mut mt = method_type_returning(self_instance_ast_type());
    let Function::Typed(function) = &mut mt.function else {
        unreachable!("generated method types are typed")
    };
    function.required_positionals = attributes
        .iter()
        .map(|attr| ast_param(attr.ty.clone(), None))
        .collect();
    mt
}

fn method_type_with_required_positional_returning(param: Type, return_type: Type) -> AstMethodType {
    let mut mt = method_type_returning(return_type);
    let Function::Typed(function) = &mut mt.function else {
        unreachable!("generated method types are typed")
    };
    function.required_positionals = vec![ast_param(param, None)];
    mt
}

fn method_type_with_positionals(
    attributes: &[DataStructAttribute],
    required: bool,
) -> AstMethodType {
    let mut mt = method_type_returning(self_instance_ast_type());
    let Function::Typed(function) = &mut mt.function else {
        unreachable!("generated method types are typed")
    };
    let params: Vec<_> = attributes
        .iter()
        .map(|attr| ast_param(attr.ty.clone(), None))
        .collect();
    if required {
        function.required_positionals = params;
    } else {
        function.optional_positionals = params;
    }
    mt
}

fn method_type_with_required_keywords(
    attributes: &[DataStructAttribute],
    names: &NameTable,
) -> AstMethodType {
    method_type_with_keywords(attributes, true, names)
}

fn method_type_with_keywords(
    attributes: &[DataStructAttribute],
    required: bool,
    names: &NameTable,
) -> AstMethodType {
    let mut mt = method_type_returning(self_instance_ast_type());
    let Function::Typed(function) = &mut mt.function else {
        unreachable!("generated method types are typed")
    };
    let keywords: Vec<_> = attributes
        .iter()
        .map(|attr| KeywordParam {
            name: names.intern_symbol(&attr.name),
            param: ast_param(attr.ty.clone(), None),
        })
        .collect();
    if required {
        function.required_keywords = keywords;
    } else {
        function.optional_keywords = keywords;
    }
    mt
}

fn method_type_with_required_record(
    attributes: &[DataStructAttribute],
    names: &NameTable,
) -> AstMethodType {
    let mut mt = method_type_returning(self_instance_ast_type());
    let Function::Typed(function) = &mut mt.function else {
        unreachable!("generated method types are typed")
    };
    function.required_positionals = vec![ast_param(record_type(attributes, names), None)];
    mt
}

fn symbol_tuple_type(attributes: &[DataStructAttribute], names: &NameTable) -> Type {
    Type::Tuple(TupleType {
        types: attributes
            .iter()
            .map(|attr| {
                Type::Literal(LiteralType {
                    literal: Literal::Symbol(names.intern_symbol(&attr.name)),
                    location: None,
                })
            })
            .collect(),
        location: None,
    })
}

fn record_type(attributes: &[DataStructAttribute], names: &NameTable) -> Type {
    Type::Record(RecordType {
        fields: attributes
            .iter()
            .map(|attr| RecordField {
                key: RecordKey::Symbol(names.intern_symbol(&attr.name)),
                ty: attr.ty.clone(),
                required: true,
            })
            .collect(),
        location: None,
    })
}

fn generated_def(
    name: &str,
    kind: MethodKind,
    location: PrismByteRange,
    overloads: Vec<AstMethodType>,
) -> Member {
    let annotations = overloads
        .into_iter()
        .map(|method_type| {
            ExplicitAnnotation::Colon(ColonMethodTypeAnnotation {
                location,
                prefix_location: location,
                annotations: vec![],
                method_type,
            })
        })
        .collect();
    Member::Def(DefMember {
        visibility: None,
        ivar_param_pairs: Vec::new(),
        name: name.to_string(),
        kind,
        location,
        name_location: location,
        method_type: MethodTypeAnnotation {
            type_annotations: TypeAnnotations::Array(annotations),
        },
        leading_comment: None,
        origin: DefMemberOrigin::Real,
        source_file: None,
    })
}

fn rbs_inline_annotation_strings(block: &CommentBlock) -> Vec<String> {
    block
        .comments
        .iter()
        .filter_map(|line| {
            let text = line.text.trim_start();
            if !text.starts_with("@rbs") {
                return None;
            }
            let marker = "%a{";
            let start = text.find(marker)? + marker.len();
            let end = text[start..].find('}')? + start;
            Some(text[start..end].to_string())
        })
        .collect()
}

fn build_type_application_annotation(
    body: &str,
    location: PrismByteRange,
    names: &NameTable,
) -> Option<crate::ast::ruby::annotations::TypeApplicationAnnotation> {
    let (parser, node) = RbsParser::parse_inline_trailing(body.as_bytes()).ok()?;
    ast_builder::build_type_application_annotation(&parser, node, location, names)
}

impl<'a> InlineCollector<'a> {
    fn with_source_location(
        &self,
        mut annotation: crate::ast::ruby::annotations::InstanceVariableAnnotation,
    ) -> crate::ast::ruby::annotations::InstanceVariableAnnotation {
        annotation.source_location =
            self.file_name()
                .map(|file_name| crate::location::RubyLocation {
                    file: file_name,
                    start_byte: annotation.location.0,
                    end_byte: annotation.location.1,
                });
        annotation
    }

    fn mark_comment_block_associated(
        &mut self,
        block: Option<&crate::ast::ruby::comment_block::CommentBlock>,
    ) {
        let Some(block) = block else {
            return;
        };
        for line in &block.comments {
            let line_no = self.line_index.line(line.location.0 as usize);
            self.associated_leading_lines.insert(line_no);
        }
    }

    fn mark_enclosed_leading_lines(&mut self, location: ruby_prism::Location<'_>) {
        let start_line = self.line_index.line(location.start_offset());
        let end_line = self
            .line_index
            .line(location.end_offset().saturating_sub(1));
        if start_line + 1 >= end_line {
            return;
        }
        for line in (start_line + 1)..end_line {
            if self.comments.leading_annotation(line).is_some() {
                self.associated_leading_lines.insert(line);
            }
        }
    }

    /// Push `# @rbs module-self: T` annotations from the module's
    /// leading block as `Member::ModuleSelf` into the currently-open
    /// scope. Non-module-self paragraphs (other annotations, plain
    /// comments) are silently skipped — there is no module-level
    /// `unused_annotation` reporter in crema, matching the current
    /// (intentional) gap.
    ///
    /// Mirrors rbs `inline_parser.rb#visit_module_node`'s `leading_block`
    /// loop (`AST::Ruby::Annotations::ModuleSelfAnnotation` →
    /// `Members::ModuleSelfMember`).
    fn collect_leading_module_self_members(&mut self, module_start_line: usize) {
        let Some(block) = self.comments.leading_block_for(module_start_line) else {
            return;
        };
        // Clone to release the immutable borrow on `self.comments` before
        // mutating `self` (push_member / associated_leading_lines insert).
        let block = block.clone();
        for paragraph in block.each_paragraph(&[], self.names) {
            if let LeadingAnnotation::ModuleSelf(ms) = paragraph {
                let line_no = self.line_index.line(ms.location.0 as usize);
                self.push_member(Member::ModuleSelf(ModuleSelfMember { annotation: ms }));
                self.associated_leading_lines.insert(line_no);
            }
        }
    }

    fn collect_enclosed_instance_variable_members(&mut self, location: ruby_prism::Location<'_>) {
        let start_line = self.line_index.line(location.start_offset());
        let end_line = self
            .line_index
            .line(location.end_offset().saturating_sub(1));
        if start_line + 1 >= end_line {
            return;
        }
        for line in (start_line + 1)..end_line {
            if self.associated_leading_lines.contains(&line) {
                continue;
            }
            let Some((range, text)) = self.comments.leading_annotation(line) else {
                continue;
            };
            let Ok((parser, node)) = RbsParser::parse_inline_leading(text.as_bytes()) else {
                continue;
            };
            let Some(mut annotation) =
                ast_builder::build_leading_annotation(&parser, node, text.as_bytes(), self.names)
            else {
                continue;
            };
            let crate::ast::ruby::annotations::LeadingAnnotation::InstanceVariable(ivar) =
                &mut annotation
            else {
                continue;
            };
            ivar.location = offset_range(ivar.location, range.0);
            ivar.name_location = offset_range(ivar.name_location, range.0);
            let ivar = self.with_source_location(ivar.clone());
            self.push_member(Member::InstanceVariable(InstanceVariableMember {
                annotation: ivar,
            }));
            self.associated_leading_lines.insert(line);
        }
    }
}

/// Shallow scan of an instance-side `initialize` body for the
/// Sorbet-inferable `@ivar = param` shape: an unconditional top-level
/// statement whose RHS is a bare read of a required/optional positional
/// or keyword parameter. Anything else (conditionals, expressions,
/// local-variable indirection, rest/block/post params) is skipped —
/// not inferring is always the safe direction. The first eligible
/// assignment per ivar wins; later ones are left to the write check.
fn collect_initialize_ivar_param_pairs(node: &DefNode<'_>) -> Vec<InitializeIvarParamPair> {
    let Some(params) = node.parameters() else {
        return Vec::new();
    };

    let mut param_refs: Vec<(String, InitializeParamRef)> = Vec::new();
    for (index, param) in params.requireds().iter().enumerate() {
        if let Some(p) = param.as_required_parameter_node() {
            param_refs.push((
                String::from_utf8_lossy(p.name().as_slice()).into_owned(),
                InitializeParamRef::RequiredPositional(index),
            ));
        }
    }
    for (index, param) in params.optionals().iter().enumerate() {
        if let Some(p) = param.as_optional_parameter_node() {
            param_refs.push((
                String::from_utf8_lossy(p.name().as_slice()).into_owned(),
                InitializeParamRef::OptionalPositional(index),
            ));
        }
    }
    for keyword in params.keywords().iter() {
        let name = if let Some(p) = keyword.as_required_keyword_parameter_node() {
            String::from_utf8_lossy(p.name().as_slice()).into_owned()
        } else if let Some(p) = keyword.as_optional_keyword_parameter_node() {
            String::from_utf8_lossy(p.name().as_slice()).into_owned()
        } else {
            continue;
        };
        param_refs.push((name.clone(), InitializeParamRef::Keyword(name)));
    }
    if param_refs.is_empty() {
        return Vec::new();
    }

    let Some(body) = node.body() else {
        return Vec::new();
    };
    let Some(stmts) = body.as_statements_node() else {
        return Vec::new();
    };

    let mut pairs: Vec<InitializeIvarParamPair> = Vec::new();
    for stmt in stmts.body().iter() {
        let Some(write) = stmt.as_instance_variable_write_node() else {
            continue;
        };
        let Some(read) = write.value().as_local_variable_read_node() else {
            continue;
        };
        if read.depth() != 0 {
            continue;
        }
        let read_name = String::from_utf8_lossy(read.name().as_slice());
        let Some((_, param_ref)) = param_refs.iter().find(|(n, _)| *n == read_name) else {
            continue;
        };
        let ivar_name = String::from_utf8_lossy(write.name().as_slice()).into_owned();
        if pairs.iter().any(|p| p.ivar_name == ivar_name) {
            continue;
        }
        pairs.push(InitializeIvarParamPair {
            ivar_name,
            param: param_ref.clone(),
        });
    }
    pairs
}

fn is_attr_call(node: &Node<'_>) -> bool {
    node.as_call_node().is_some_and(|call| {
        call.receiver().is_none()
            && matches!(
                call.name().as_slice(),
                b"attr_reader" | b"attr_writer" | b"attr_accessor"
            )
    })
}

fn single_attr_name(attr: &AttributeMember) -> Option<&str> {
    match attr.name_nodes.as_slice() {
        [only] => Some(only.name.as_str()),
        _ => None,
    }
}

pub(crate) fn prism_location_range(location: ruby_prism::Location<'_>) -> PrismByteRange {
    (location.start_offset() as u32, location.end_offset() as u32)
}

fn offset_range(range: PrismByteRange, offset: u32) -> PrismByteRange {
    (range.0 + offset, range.1 + offset)
}

