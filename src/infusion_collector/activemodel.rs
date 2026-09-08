//! ActiveModel DSL rules.

use crate::ast::MethodKind;
use crate::ast::ruby::members::Member;
use crate::infusion_collector::pipeline::{InfusionCall, push_def};

pub(crate) fn collect_call(members: &mut Vec<Member>, call: &InfusionCall) {
    collect_has_secure_password_call(members, call);
}

fn collect_has_secure_password_call(members: &mut Vec<Member>, call: &InfusionCall) {
    if call.name != "has_secure_password" {
        return;
    }
    let attribute = call
        .symbol_args
        .first()
        .map(|arg| (arg.name.as_str(), arg.location))
        .unwrap_or(("password", call.location));
    let reader = attribute.0.to_string();
    let writer = format!("{}=", attribute.0);
    let confirmation_writer = format!("{}_confirmation=", attribute.0);
    let authenticator = format!("authenticate_{}", attribute.0);

    for name in [reader, writer, confirmation_writer, authenticator] {
        push_def(
            members,
            name,
            MethodKind::Instance,
            call.location,
            attribute.1,
        );
    }
    if attribute.0 == "password" {
        push_def(
            members,
            "authenticate".to_string(),
            MethodKind::Instance,
            call.location,
            attribute.1,
        );
    }
}
