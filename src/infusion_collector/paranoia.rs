//! paranoia acts_as_paranoid mixin synthesis.
//!
//! The [paranoia](https://github.com/rubysherpas/paranoia) gem adds
//! soft-delete methods (`restore!`, `paranoia_destroyed?`,
//! `with_deleted`, `only_deleted`, ...) to a model at runtime when its
//! class body calls `acts_as_paranoid`. Statically none of those
//! declarations exist, so every call is a `Ruby::NoMethod` false
//! positive.
//!
//! Ported from orthoses-paranoia's `Orthoses::Paranoia#call`
//! (`~40 lines`, <https://github.com/ksss/orthoses-paranoia>). Its
//! `CallTracer::Lazy.new.trace("ActiveRecord::Base.acts_as_paranoid")`
//! becomes the collector's receiver-less `acts_as_paranoid` DSL-call
//! detection (`pipeline::InfusionBody::paranoia_model`), and each of
//! its four store lines becomes one synthesized mixin member:
//!
//! | orthoses (runtime trace)                                  | this pass (static)          |
//! |-----------------------------------------------------------|-----------------------------|
//! | `base << "include ::Paranoia::InstanceMethods[Base]"`     | model class include         |
//! | `base << "extend ::Paranoia::ClassMethods[Base, Rel]"`    | model class extend          |
//! | `Rel << "include ::Paranoia::ClassMethods[Base, Rel]"`    | `ActiveRecord_Relation` include |
//! | `Proxy << "include ::Paranoia::ClassMethods[Base, Rel]"`  | `..._CollectionProxy` include   |
//!
//! Do not add members beyond these four lines (per-method `def`
//! synthesis included) without a corresponding change on the orthoses
//! side. The `::Paranoia` module types themselves are *not* synthesized
//! — they come from the user's rbs collection (gem_rbs_collection's
//! paranoia), and when they are absent the mixins silently resolve to
//! nothing, leaving the original `Ruby::NoMethod` diagnostics in place
//! (infusion design goal: crema does not ship type definitions).
//!
//! `acts_as_paranoid` options (`column:`, `sentinel_value:`, ...) are
//! deliberately not interpreted: none of them change any signature in
//! the gem's collection types.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::ast::declarations::{ClassDeclaration as Class, ClassMember, Member};
use crate::ast::members::ExtendMember as Extend;
use crate::environment::draft::EnvironmentDraft;
use crate::environment::{DeclOrigin, InfusionUnit};
use crate::infusion_collector::active_record_synthesis::{
    active_record_model_names, class_instance, include_member, nested_name,
};
use crate::type_name::TypeName;

const PARANOIA_INSTANCE_METHODS: &str = "::Paranoia::InstanceMethods";
const PARANOIA_CLASS_METHODS: &str = "::Paranoia::ClassMethods";

/// Attach the four orthoses-paranoia mixin members for every collected
/// `acts_as_paranoid` model that is an `ActiveRecord::Base` descendant.
///
/// The descendant filter is the static counterpart of orthoses tracing
/// `ActiveRecord::Base.acts_as_paranoid` — a receiver-less
/// `acts_as_paranoid` in a non-AR class never reaches that method at
/// runtime, and synthesizing here would fabricate a
/// `<Model>::ActiveRecord_Relation` class that AR synthesis never
/// declared.
///
/// Runs directly after `active_record_synthesis::synthesize`, which
/// declares the `ActiveRecord_Relation` / `..._CollectionProxy` classes
/// these mixins reopen.
pub(crate) fn synthesize(draft: &mut EnvironmentDraft, paranoia_models: &[TypeName]) {
    if paranoia_models.is_empty() {
        return;
    }
    let ar_models: FxHashSet<TypeName> = active_record_model_names(draft).into_iter().collect();
    // Dedup: the same model may carry `acts_as_paranoid` in several
    // reopen sites; one synthesis per model suffices (mixin members
    // aggregate per class, duplicates would only bloat the draft).
    let mut seen = FxHashSet::default();
    let models: Vec<TypeName> = paranoia_models
        .iter()
        .copied()
        .filter(|model| ar_models.contains(model) && seen.insert(*model))
        .collect();
    if models.is_empty() {
        return;
    }
    let context = Arc::from([draft.names().absolute_root()]);
    for model in models {
        synthesize_model(draft, Arc::clone(&context), model);
    }
}

fn synthesize_model(
    draft: &mut EnvironmentDraft,
    context: crate::environment::draft::Context,
    model: TypeName,
) {
    let names = draft.names();
    let relation = nested_name(names, model, "ActiveRecord_Relation");
    let collection_proxy = nested_name(names, model, "ActiveRecord_Associations_CollectionProxy");
    let instance_methods = names.parse_type_name(PARANOIA_INSTANCE_METHODS);
    let class_methods = names.parse_type_name(PARANOIA_CLASS_METHODS);
    let class_methods_args = || vec![class_instance(model), class_instance(relation)];

    let model_decl = Arc::new(Class {
        name: model,
        type_params: Vec::new(),
        super_class: None,
        members: vec![
            include_member(instance_methods, vec![class_instance(model)]),
            ClassMember::Member(Member::Extend(Extend {
                name: class_methods,
                args: class_methods_args(),
                annotations: Vec::new(),
                location: None,
                source_file: None,
                comment: None,
            })),
        ],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let relation_decl = Arc::new(Class {
        name: relation,
        type_params: Vec::new(),
        super_class: None,
        members: vec![include_member(class_methods, class_methods_args())],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });
    let collection_proxy_decl = Arc::new(Class {
        name: collection_proxy,
        type_params: Vec::new(),
        super_class: None,
        members: vec![include_member(class_methods, class_methods_args())],
        annotations: Vec::new(),
        location: None,
        source_file: None,
        comment: None,
    });

    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::Paranoia, None),
        Arc::clone(&context),
        model_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::Paranoia, None),
        Arc::clone(&context),
        relation_decl,
    );
    let _ = draft.insert_class_decl(
        DeclOrigin::Synthesized(InfusionUnit::Paranoia, None),
        context,
        collection_proxy_decl,
    );
}
