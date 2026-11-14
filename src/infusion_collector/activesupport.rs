//! ActiveSupport DSL rules.

use crate::ast::MethodKind;
use crate::ast::members::Visibility;
use crate::ast::ruby::members::Member;
use crate::infusion_collector::pipeline::{InfusionCall, push_def, push_def_with_visibility};

pub(crate) fn collect_call(members: &mut Vec<Member>, call: &InfusionCall) {
    let (reads, writes) = match call.name.as_str() {
        "cattr_reader" | "mattr_reader" => (true, false),
        "cattr_writer" | "mattr_writer" => (false, true),
        "cattr_accessor" | "mattr_accessor" => (true, true),
        "class_attribute" => return collect_class_attribute(members, call),
        "delegate" => return collect_delegate(members, call),
        _ => return,
    };

    let mut instance_reader = true;
    let mut instance_writer = true;
    let mut instance_accessor = true;
    for kw in &call.keyword_bools {
        let Some(b) = kw.value else { continue };
        match kw.name.as_str() {
            "instance_reader" => instance_reader = b,
            "instance_writer" => instance_writer = b,
            "instance_accessor" => instance_accessor = b,
            _ => {}
        }
    }

    for arg in &call.symbol_args {
        if reads {
            push_def(
                members,
                arg.name.clone(),
                MethodKind::Singleton,
                call.location,
                arg.location,
            );
            if instance_reader && instance_accessor {
                push_def(
                    members,
                    arg.name.clone(),
                    MethodKind::Instance,
                    call.location,
                    arg.location,
                );
            }
        }
        if writes {
            let setter = format!("{}=", arg.name);
            push_def(
                members,
                setter.clone(),
                MethodKind::Singleton,
                call.location,
                arg.location,
            );
            if instance_writer && instance_accessor {
                push_def(
                    members,
                    setter,
                    MethodKind::Instance,
                    call.location,
                    arg.location,
                );
            }
        }
    }
}

/// Rails keyword defaults chain (`activesupport/core_ext/class/attribute.rb`):
/// `instance_reader`/`instance_writer` default to `instance_accessor`, but an
/// explicit value wins over it. `instance_predicate` defaults to true
/// independently. Non-literal keyword values fall back to the default.
fn collect_class_attribute(members: &mut Vec<Member>, call: &InfusionCall) {
    let mut instance_reader = None;
    let mut instance_writer = None;
    let mut instance_accessor = true;
    let mut instance_predicate = true;
    for kw in &call.keyword_bools {
        let Some(b) = kw.value else { continue };
        match kw.name.as_str() {
            "instance_reader" => instance_reader = Some(b),
            "instance_writer" => instance_writer = Some(b),
            "instance_accessor" => instance_accessor = b,
            "instance_predicate" => instance_predicate = b,
            _ => {}
        }
    }
    let instance_reader = instance_reader.unwrap_or(instance_accessor);
    let instance_writer = instance_writer.unwrap_or(instance_accessor);

    for arg in &call.symbol_args {
        let mut synthesize = |name: String, kind: MethodKind| {
            push_def(members, name, kind, call.location, arg.location);
        };
        synthesize(arg.name.clone(), MethodKind::Singleton);
        synthesize(format!("{}=", arg.name), MethodKind::Singleton);
        if instance_predicate {
            synthesize(format!("{}?", arg.name), MethodKind::Singleton);
        }
        if instance_reader {
            synthesize(arg.name.clone(), MethodKind::Instance);
            if instance_predicate {
                synthesize(format!("{}?", arg.name), MethodKind::Instance);
            }
        }
        if instance_writer {
            synthesize(format!("{}=", arg.name), MethodKind::Instance);
        }
    }
}

/// Tier 0 of `Module#delegate` (`activesupport/lib/active_support/delegation.rb`
/// `generate`): only the names are synthesized, as `(?) -> untyped`
/// instance methods; the target's type is not consulted. Visibility is
/// public unless `private: true` — Rails defines the methods through a
/// `module_eval` string, a fresh frame that an enclosing bare `private`
/// does not reach. `prefix: true` requires a method-name target (Rails
/// raises for `:@ivar` / constants and turns `:"self.class"` into a
/// singleton method), so those shapes synthesize nothing rather than an
/// unprefixed stand-in.
fn collect_delegate(members: &mut Vec<Member>, call: &InfusionCall) {
    // Every `key:` lands in `keyword_bools` (value `None` for non-bool
    // literals), which is how a `to: Const` is told apart from no `to:`.
    if !call.keyword_bools.iter().any(|kw| kw.name == "to") {
        return;
    }
    let keyword_string = |name: &str| {
        call.keyword_strings
            .iter()
            .find(|kw| kw.name == name)
            .map(|kw| kw.value.as_str())
    };
    let keyword_bool = |name: &str| {
        call.keyword_bools
            .iter()
            .find(|kw| kw.name == name)
            .and_then(|kw| kw.value)
    };
    let prefix = if let Some(custom) = keyword_string("prefix") {
        Some(format!("{custom}_"))
    } else if keyword_bool("prefix") == Some(true) {
        match keyword_string("to") {
            Some(to)
                if to.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
                    && !to.contains('.') =>
            {
                Some(format!("{to}_"))
            }
            _ => return,
        }
    } else {
        None
    };
    let visibility = (keyword_bool("private") == Some(true)).then_some(Visibility::Private);
    for arg in &call.symbol_args {
        let name = match &prefix {
            Some(prefix) => format!("{prefix}{}", arg.name),
            None => arg.name.clone(),
        };
        push_def_with_visibility(
            members,
            name,
            MethodKind::Instance,
            call.location,
            arg.location,
            visibility,
        );
    }
}
