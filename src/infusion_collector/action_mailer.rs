//! ActionMailer action → class-method synthesis.
//!
//! Rails mailers write actions as instance methods and call them on the
//! class:
//!
//! ```ruby
//! class FooMailer < ApplicationMailer
//!   def hello(user) = mail(to: user.email)
//! end
//! FooMailer.hello(user).deliver_later
//! ```
//!
//! `ActionMailer::Base.method_missing` turns `FooMailer.hello` into
//! `MessageDelivery.new(self, :hello, ...)` whenever `hello` is one of
//! `action_methods` (`actionmailer/lib/action_mailer/base.rb`). No
//! singleton declaration exists statically, so every such call is a
//! `Ruby::NoMethod` false positive.
//!
//! Ported from orthoses-rails' `Orthoses::ActionMailer::Base#call`
//! (`lib/orthoses/action_mailer/base.rb`), which emits
//! `def self.#{action}: #{params} -> ::ActionMailer::MessageDelivery`
//! per action. orthoses copies the runtime `parameters`; crema's
//! unannotated `def` is `(?) -> untyped` on the instance side, so the
//! synthesized singleton is `(?)` too — there is nothing to copy.
//! Annotated actions (`#: (User) -> untyped`) keep the `(?)` shape for
//! now; mirroring their parameters is a deliberate follow-up.
//!
//! The "a real `def self.x` beats `method_missing`" rule has two halves.
//! Here, an action already declared on the mailer's own singleton (Ruby
//! or RBS) is skipped. Ruby's lookup walks the whole singleton ancestry
//! before `method_missing`, though, so an action named after a class
//! method inherited from `ApplicationMailer`, `ActionMailer::Base`
//! (`default`, `layout`, ...) or `Module` (`name`) never reaches
//! `method_missing` at runtime either. That half needs the resolved
//! ancestor chain, which does not exist at draft time: each synthesized
//! member is stamped `%a{crema:method_missing}` and the definition
//! builder drops it when an ancestor defines the name for real
//! (`DefinitionBuilder::drop_method_missing_members_shadowed_by_ancestors`).
//!
//! Known gap: Rails' `action_methods` is *public* instance methods, but
//! crema does not model Ruby-side `private` yet (rbs inline: "Method
//! visibility declaration is not supported yet"), so private helpers on a
//! mailer are synthesized too. `initialize` is skipped explicitly; every
//! other Base-internal method lives on `ActionMailer::Base` itself and
//! never appears as a descendant's Ruby `def`.
//!
//! `::ActionMailer::MessageDelivery` is *not* declared here — it comes
//! from the user's rbs collection (gem_rbs_collection's actionmailer).
//! When it is absent the synthesized return type is an unknown name and
//! the call degrades to whatever diagnostic that yields, never a crash.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::ast::MethodKind;
use crate::ast::annotation::Annotation;
use crate::ast::declarations::{
    ClassDeclaration as Class, ClassMember, Declaration as SigDeclaration, Member as SigMember,
    ModuleMember,
};
use crate::ast::ruby::declarations::Declaration as RubyDeclaration;
use crate::ast::ruby::members::Member as RubyMember;
use crate::definition::method::METHOD_MISSING_ANNOTATION;
use crate::environment::draft::{
    ClassDeclarationDraft, ClassOrModuleDraft, EnvironmentDraft, ModuleDeclarationDraft,
};
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    descendant_class_names, method_returning,
};
use crate::name::{NameTable, Symbol};
use crate::type_name::TypeName;

const ACTION_MAILER_BASE: &str = "::ActionMailer::Base";
const MESSAGE_DELIVERY: &str = "::ActionMailer::MessageDelivery";

/// Per mailer, the Ruby instance `def` names in source order (the
/// action candidates) and every name already declared on the singleton,
/// in Ruby (`def self.x`) or RBS (`def self.x: ...`).
#[derive(Default)]
struct MailerMethods {
    instance: Vec<Symbol>,
    singleton: Vec<Symbol>,
}

/// Declare `def self.<action>: (?) -> ::ActionMailer::MessageDelivery`
/// on every `ActionMailer::Base` descendant for each of its Ruby
/// instance `def`s that the class does not already define on its
/// singleton (a real `def self.x` beats `method_missing` at runtime).
pub(crate) fn synthesize(draft: &mut EnvironmentDraft) {
    let mailers = descendant_class_names(draft, ACTION_MAILER_BASE);
    if mailers.is_empty() {
        return;
    }
    let methods = collect_mailer_methods(draft, &mailers);
    let names = draft.names();
    let message_delivery = names.parse_type_name(MESSAGE_DELIVERY);
    let method_missing = names.intern_symbol(METHOD_MISSING_ANNOTATION);
    let context = Arc::from([names.absolute_root()]);

    let mut decls = Vec::new();
    for mailer in mailers {
        let Some(m) = methods.get(&mailer) else {
            continue;
        };
        let mut seen = m.singleton.clone();
        let members: Vec<ClassMember> = m
            .instance
            .iter()
            .copied()
            .filter(|name| {
                if seen.contains(name) {
                    return false;
                }
                seen.push(*name);
                true
            })
            .map(|name| {
                let mut member = method_returning(name, MethodKind::Singleton, message_delivery);
                if let SigMember::MethodDefinition(def) = &mut member {
                    def.annotations.push(Annotation {
                        string: method_missing,
                        location: None,
                    });
                }
                ClassMember::Member(member)
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        decls.push(Arc::new(Class {
            name: mailer,
            type_params: Vec::new(),
            super_class: None,
            members,
            annotations: Vec::new(),
            location: None,
            source_file: None,
            comment: None,
        }));
    }
    for decl in decls {
        let _ = draft.insert_class_decl(
            DeclOrigin::Synthesized(InfusionUnit::ActionMailer, None),
            Arc::clone(&context),
            decl,
        );
    }
}

/// One read-only pass over the draft's declaration trees. Nested Ruby
/// decls are not flattened into `class_decls` at draft time, so the walk
/// descends into `Member::Declaration` the way AR synthesis'
/// `collect_nested_class_methods` does.
fn collect_mailer_methods(
    draft: &EnvironmentDraft,
    mailers: &[TypeName],
) -> FxHashMap<TypeName, MailerMethods> {
    let names = draft.names();
    let mut out: FxHashMap<TypeName, MailerMethods> = FxHashMap::default();
    for (name, entry) in &draft.class_decls {
        match entry {
            ClassOrModuleDraft::Class(e) => {
                for (_, _, decl) in &e.context_decls {
                    match decl {
                        ClassDeclarationDraft::Ruby(r) => {
                            if mailers.contains(name) {
                                collect_ruby_members(
                                    names,
                                    &r.members,
                                    out.entry(*name).or_default(),
                                );
                            }
                            collect_ruby_nested(names, &r.members, mailers, &mut out);
                        }
                        ClassDeclarationDraft::Signature(c) => {
                            if mailers.contains(name) {
                                collect_signature_members(
                                    &c.members,
                                    out.entry(*name).or_default(),
                                );
                            }
                            collect_signature_nested(&c.members, mailers, &mut out);
                        }
                    }
                }
            }
            ClassOrModuleDraft::Module(e) => {
                for (_, _, decl) in &e.context_decls {
                    match decl {
                        ModuleDeclarationDraft::Ruby(r) => {
                            collect_ruby_nested(names, &r.members, mailers, &mut out);
                        }
                        ModuleDeclarationDraft::Signature(m) => {
                            collect_signature_module_nested(&m.members, mailers, &mut out);
                        }
                    }
                }
            }
        }
    }
    out
}

fn collect_ruby_members(names: &NameTable, members: &[RubyMember], out: &mut MailerMethods) {
    for member in members {
        let RubyMember::Def(def) = member else {
            continue;
        };
        let name = names.intern_symbol(&def.name);
        match def.kind {
            MethodKind::Instance => {
                if def.name != "initialize" {
                    out.instance.push(name);
                }
            }
            MethodKind::Singleton | MethodKind::SingletonInstance => out.singleton.push(name),
        }
    }
}

fn collect_ruby_nested(
    names: &NameTable,
    members: &[RubyMember],
    mailers: &[TypeName],
    out: &mut FxHashMap<TypeName, MailerMethods>,
) {
    for member in members {
        let RubyMember::Declaration(decl) = member else {
            continue;
        };
        match decl {
            RubyDeclaration::Class(c) => {
                if mailers.contains(&c.class_name) {
                    collect_ruby_members(names, &c.members, out.entry(c.class_name).or_default());
                }
                collect_ruby_nested(names, &c.members, mailers, out);
            }
            RubyDeclaration::Module(m) => collect_ruby_nested(names, &m.members, mailers, out),
            RubyDeclaration::Constant(_) | RubyDeclaration::ClassModuleAlias(_) => {}
        }
    }
}

/// RBS-side declarations contribute singleton names only: an RBS
/// `def self.hello` is an explicit declaration that must not be
/// shadowed, while RBS instance `def`s are not Ruby actions.
fn collect_signature_members(members: &[ClassMember], out: &mut MailerMethods) {
    for member in members {
        if let ClassMember::Member(SigMember::MethodDefinition(method)) = member
            && matches!(
                method.kind,
                MethodKind::Singleton | MethodKind::SingletonInstance
            )
        {
            out.singleton.push(method.name);
        }
    }
}

fn collect_signature_nested(
    members: &[ClassMember],
    mailers: &[TypeName],
    out: &mut FxHashMap<TypeName, MailerMethods>,
) {
    for member in members {
        if let ClassMember::Declaration(decl) = member {
            collect_signature_decl(decl, mailers, out);
        }
    }
}

fn collect_signature_module_nested(
    members: &[ModuleMember],
    mailers: &[TypeName],
    out: &mut FxHashMap<TypeName, MailerMethods>,
) {
    for member in members {
        if let ModuleMember::Declaration(decl) = member {
            collect_signature_decl(decl, mailers, out);
        }
    }
}

fn collect_signature_decl(
    decl: &SigDeclaration,
    mailers: &[TypeName],
    out: &mut FxHashMap<TypeName, MailerMethods>,
) {
    match decl {
        SigDeclaration::Class(c) => {
            if mailers.contains(&c.name) {
                collect_signature_members(&c.members, out.entry(c.name).or_default());
            }
            collect_signature_nested(&c.members, mailers, out);
        }
        SigDeclaration::Module(m) => collect_signature_module_nested(&m.members, mailers, out),
        SigDeclaration::Interface(_)
        | SigDeclaration::ClassAlias(_)
        | SigDeclaration::ModuleAlias(_)
        | SigDeclaration::TypeAlias(_)
        | SigDeclaration::Constant(_)
        | SigDeclaration::Global(_) => {}
    }
}
