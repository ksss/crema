# crema doc config

Sniff the project yourself, compare the current file with these keys, and ask the user before changing behavior — this is the configuration vocabulary, not a script to run unattended.

## check

`check` — `Array[String]`, default `none (required)`. Example: `check = ["app", "lib"]`

The directories and `.rb` files that make up the type-check scope. Directories are expanded recursively; entries resolve relative to the directory containing `crema.toml` (or the file passed with `--config <PATH>` — the same rule applies to every path inside it). CLI positional arguments only narrow this scope: a file argument must already be inside `check`, a directory argument is intersected with it.

`crema.toml` is found by walk-up search from the current directory toward the repository boundary. `crema check` exits `2` with a copy-pasteable suggestion when the file or `check` is missing; other subcommands (e.g. `crema doc diagnostic`) run with defaults.

Propose adding `check` for any project that doesn't have one yet.

## sig

`sig` — `Array[String]`, default `[]`. Example: `sig = ["sig"]`

RBS signature directories or files loaded before checking Ruby code.

Propose it when the project keeps app-specific or generated `.rbs` files outside the command-line `--sig` flow.

## ignore

`ignore` — `Array[String]`, default `[]`. Example: `ignore = ["db/schema.rb", "vendor"]`

Glob patterns subtracted from the `.rb` files `check` walks, relative to the directory holding `crema.toml`. An entry without glob metacharacters also matches as a path prefix, so `"vendor"` drops the whole directory.

This is a scope subtraction, not a diagnostic filter: an excluded file supplies no declarations, so its classes and inline annotations are invisible to every other file. Naming an excluded file as a positional argument to `crema check` exits `2`.

Propose it when `check` names a directory that also holds generated or vendored `.rb` files.

## sig_ignore

`sig_ignore` — `Array[String]`, default `[]`. Example: `sig_ignore = ["sig/generated/**/*.rbs"]`

The same subtraction for signatures. Patterns apply to every entry from `sig`, `--sig` and `--add-sig`, and each dropped file is reported as a warning. `sig` entries match relative to the project root; `--sig` and `--add-sig` entries relative to the current directory.

Propose it when the signature tree carries generated or stale `.rbs` files that should not load.

## libraries

`libraries` — `Array[String]`, default `[]`. Example: `libraries = ["json", "prism"]`

Bundled or gem-provided RBS libraries to load by name, including manifest dependencies.

Propose it when checked code calls stdlib or gem APIs whose RBS should be available project-wide.

## inline

`inline` — `Boolean`, default `true`. Example: `inline = false`

Whether `# @rbs` inline annotations in Ruby files participate in type checking.

Propose `false` only for sig-only checking; inline annotation syntax errors are then hidden from normal checks.

## incremental

`incremental` — `any value`, ignored. Example: none — remove the key

Has no effect. While the key is present, every run prints a warning on stderr; the next release rejects it as an unknown key. Propose deleting the line; `.crema/cache/incremental_v1.bin`, if present, is unused and can be deleted with it.

## did_you_mean

`did_you_mean` — `Boolean`, default `false`. Example: `did_you_mean = true`

Fills the `did_you_mean` field of `Ruby::UnknownConstant` with spelling suggestions from the constants in scope; off, the field is absent. The spell-check runs per diagnostic, so on a large tree with many unresolved constants it can cost as much as the type check itself.

Propose `true` only when unresolved-constant typos recur and the extra time is acceptable.

## diagnostic

`diagnostic` — `Table`, default `{ preset = "default" }`. Example: `diagnostic = { preset = "default" }`

`preset` selects a baked-in severity table:

- `default` — crema's built-in severities.
- `all_error` — promotes normal configurable diagnostics to `error` (dev-only diagnostics stay ignored unless named explicitly).
- `all_ignore` — demotes every configurable diagnostic to `ignore`.

Per-code overrides in the same table (e.g. `"Ruby::NoMethod" = "warning"`) win over the preset. Accepted severities:

- `error`
- `warning`
- `information`
- `hint`
- `ignore`

Always-error diagnostics such as `Ruby::SyntaxError` cannot be overridden.

Propose `preset = "default"` unless the user wants a migration policy such as strict CI (`all_error`) or broad temporary suppression (`all_ignore`).

## baseline

`baseline` — `Boolean` or a path string, default `false`. Example: `baseline = true`

Grandfathers existing diagnostics, like `rubocop_todo.yml`. Accepted values:

- `false` — disabled; `crema check` reports everything.
- `true` — use `crema_baseline.jsonl` next to `crema.toml`.
- a path string — use that file, relative to the `crema.toml` directory.

`crema check --update-baseline` writes one row per current diagnostic: `file`, `code` and a `fingerprint` that survives line shifts and edits elsewhere in the file (the rows `crema check --tamp` prints). Every later `crema check` drops each diagnostic matching a row, so only new diagnostics are printed and decide the exit code. A row listed twice absorbs two diagnostics. Rows that matched nothing are counted in one stderr line suggesting `--update-baseline`; they never fail the run. `--no-baseline` reports everything for one run.

Once the key is on, the file must exist (`crema check` exits `2` until `--update-baseline` creates it, even if empty), and `file` is relative to the `crema.toml` directory, so the same rows match whichever subdirectory crema runs from. Severity `ignore` in `diagnostic` removes a code before the baseline sees it, so those codes never enter the file.

Propose `baseline = true` for an existing project adopting crema with more diagnostics than it will fix at once, and commit the file. Refresh it with `--update-baseline` after each cleanup rather than editing rows by hand.

## collection_config

`collection_config` — `String`, default `none (auto-discovered)`. Example: `collection_config = "rbs_collection.yaml"`

Explicit path to an `rbs_collection.yaml`; the lockfile path is derived from it. Without it, crema walks up from the project root (the `crema.toml` directory) to discover the file, stopping at `.git`, so every subdirectory of the project reads the same collection.

Propose it when the project uses rbs collection and auto-discovery would pick the wrong file.

## infusion

`infusion` — `Table`, default `absent`. Example: `infusion = { rails = { enabled = true } }`

Synthesizes, in memory and without writing RBS, the declarations that metaprogramming or static data files would produce at runtime. Each sub-table below is an independent opt-in; the feature is inert while `[infusion]` is absent. Unknown names under `[infusion]` are a parse error, and a value crema cannot act on (an empty `decorator_suffix`, an empty inflection entry) is rejected at config load even when its provider is disabled.

### [infusion.rails]

- `enabled` — `Boolean`, default `false`. Turns on the ActiveSupport, ActiveModel and ActiveRecord rule families together, plus zeitwerk-style constant synthesis. Also synthesizes each ActionMailer action (an instance `def` on an `ActionMailer::Base` descendant) as a class method returning `ActionMailer::MessageDelivery`. The unit of choice is the framework, not the individual rule.
- `inflections` — `Table`, default `absent`. Locale sub-tables; only `en` exists.

Propose this table when the project uses Rails.

### [infusion.rails.inflections.en]

Mirrors the `inflect.` directives in `config/initializers/inflections.rb`, which crema does not parse — transcribe them by hand.

- `irregular` — `Array[[String, String]]`, default `[]`. Example: `irregular = [["person", "people"]]`. `[singular, plural]` pairs, like `inflect.irregular`. Neither string may be empty.
- `acronym` — `Array[String]`, default `[]`. Example: `acronym = ["API", "HTTP"]`. Like `inflect.acronym`. Entries may not be empty.

`plural` / `singular` regex pairs and `uncountable` are not supported; any locale key other than `en` is a parse error. The table is validated even when `infusion.rails.enabled` is `false`, but has nothing to apply to until the ActiveRecord pipeline runs.

Propose it when that initializer declares `inflect.irregular` or `inflect.acronym`.

### [infusion.config]

Synthesizes a constant whose methods mirror the keys of YAML files, so `Settings.database.host` type-checks against the real file.

- `files` — `Array[String]`, default `[]`, at least one entry required once the table exists. Example: `files = ["config/settings.yml", "config/settings/production.yml"]`. Files merge in order, later entries winning. Each must be pure YAML (ERB fails to parse) with a mapping at the root. Duplicate keys in one file resolve last-wins like Psych and report a warning.
- `const_name` — `String`, default `"Settings"`. Example: `const_name = "Config"`. A single constant name: an ASCII uppercase letter, then letters, digits or `_`; no `::`.
- `except_keys` — `Array[String]`, default `[]`. Example: `except_keys = ["defaults"]`. Drops every key of that name at any depth. Use it for YAML anchors or scaffolding.

Every remaining key becomes a method, so it must read as a Ruby method name (leading letter or `_`, then letters, digits, `_`, optional trailing `?` or `!`); nested mappings become interfaces. A key that cannot be a method name is a hard error, the signal that the file needs `except_keys` or a rename.

Propose it when the project reads settings from a pure-YAML file.

### [infusion.active_decorator]

Covers the [active_decorator](https://github.com/amatsuda/active_decorator) gem, whose decorator modules are written as if the model's methods were their own.

- `enabled` — `Boolean`, default `false`. A module under `app/decorators` whose name ends with `decorator_suffix` gains the same-named model as its self type, so `UserDecorator` resolves `User`'s methods.
- `decorator_suffix` — `String`, default `"Decorator"`. Example: `decorator_suffix = "Presenter"`. Mirrors `ActiveDecorator.config.decorator_suffix`. An empty string is rejected.

The directory is fixed at `app/decorators`. A decorator whose model is not declared is skipped silently, and an explicitly written self type is never overwritten.

Propose it when the Gemfile depends on active_decorator or the project has `app/decorators`; match `decorator_suffix` to the initializer.

### [infusion.paranoia]

Covers the [paranoia](https://github.com/rubysherpas/paranoia) gem, whose `acts_as_paranoid` adds soft-delete methods (`restore!`, `paranoia_destroyed?`, `with_deleted`, `only_deleted`, ...).

- `enabled` — `Boolean`, default `false`. Every ActiveRecord model calling `acts_as_paranoid` gains `include Paranoia::InstanceMethods[Model]` and `extend Paranoia::ClassMethods[Model, Model::ActiveRecord_Relation]`; its synthesized relation and collection-proxy classes gain the same `ClassMethods` include.

Requires `[infusion.rails] enabled = true`; enabling paranoia alone is a config error. The `Paranoia` module types come from the rbs collection (the `paranoia` gem's entry); when absent, the mixins quietly resolve to nothing. `acts_as_paranoid` options such as `column:` are not interpreted.

Propose it when the Gemfile depends on paranoia and models call `acts_as_paranoid`.

### [infusion.sidekiq]

Covers the [sidekiq](https://github.com/sidekiq/sidekiq) gem, whose `include Sidekiq::Job` wires the class-level API (`perform_async`, `sidekiq_options`, ...) onto the includer.

- `enabled` — `Boolean`, default `false`. Every class whose body calls `include Sidekiq::Job` gains `include ::Sidekiq::Job::Options`, `extend ::Sidekiq::Job::Options::ClassMethods` and `extend ::Sidekiq::Job::ClassMethods`. `include Sidekiq::Worker` is covered when the collection declares `Worker` as an alias of `Job`.

Independent of `[infusion.rails]`. Only classes are synthesized; a module that includes `Sidekiq::Job` is skipped, because the `extend` would land on that module's singleton, not on its includers. The `Sidekiq::Job` types come from the rbs collection (the `sidekiq` gem's entry); when absent, the original `Ruby::NoMethod` diagnostics stay.

Propose it when the Gemfile depends on sidekiq and job classes include `Sidekiq::Job` (or `Sidekiq::Worker`).
