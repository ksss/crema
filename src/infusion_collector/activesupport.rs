//! ActiveSupport DSL rules.

use crate::ast::MethodKind;
use crate::ast::ruby::members::Member;
use crate::infusion_collector::pipeline::{InfusionCall, push_def};

pub(crate) fn collect_call(members: &mut Vec<Member>, call: &InfusionCall) {
    let (reads, writes) = match call.name.as_str() {
        "cattr_reader" | "mattr_reader" => (true, false),
        "cattr_writer" | "mattr_writer" => (false, true),
        "cattr_accessor" | "mattr_accessor" => (true, true),
        "class_attribute" => return collect_class_attribute(members, call),
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
