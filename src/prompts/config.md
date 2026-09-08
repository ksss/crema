# crema doc config

Sniff the project yourself, compare the current file with these keys, and ask the user before changing behavior — this is the configuration vocabulary, not a script to run unattended.

## check

`check` — `Array[String]`, default `none (required)`. Example: `check = ["app", "lib"]`

Declares the directories and `.rb` files that make up the project's type-check scope — the same "project source" role `sig` plays for signatures. Directories are expanded recursively for `.rb` files; entries resolve relative to the directory containing `crema.toml` (cwd-relative when `--config <PATH>` is used instead). CLI positional arguments do not add files to this scope — they narrow it to a subset (a file argument must already be inside `check`; a directory argument is expanded and intersected with it). Without `--config`, crema discovers `crema.toml` via walk-up search from the current directory toward the repository boundary; `crema check` exits `2` with a copy-pasteable suggestion when no `crema.toml` is found or `check` is absent, while other subcommands (e.g. `crema doc diagnostic`) still run with sensible defaults when no config file is found.

Propose adding `check` for any project that doesn't have one yet.

## sig

`sig` — `Array[String]`, default `[]`. Example: `sig = ["sig"]`

RBS signature directories or files crema loads before checking Ruby code.

Propose it when the project keeps app-specific or generated `.rbs` files outside the command-line `--sig` flow.

## ignore

`ignore` — `Array[String]`, default `[]`. Example: `ignore = ["db/schema.rb", "vendor"]`

Glob patterns subtracted from the `.rb` files `check` walks, matched against paths relative to the directory holding `crema.toml`. An entry without glob metacharacters also matches as a plain path prefix, so `"vendor"` drops that whole directory. This is a set subtraction, not a diagnostic filter: an excluded file supplies no declarations at all, so its classes and its inline annotations are invisible to every other file. Files leave the scope quietly, but naming one as a positional argument to `crema check` exits `2` saying so rather than silently checking nothing.

Propose it when `check` has to name a directory that also holds generated or vendored `.rb` files the project does not want in scope.

## sig_ignore

`sig_ignore` — `Array[String]`, default `[]`. Example: `sig_ignore = ["sig/generated/**/*.rbs"]`

The same subtraction on the signature side: patterns are removed from the final signature file set. It applies to every entry regardless of origin — `sig`, `--sig`, or `--add-sig` — and each dropped file is reported as a warning, so an explicit `--sig <file>` that a pattern swallows cannot be mistaken for a typo. The matching basis does inherit the asymmetry of those origins: `sig` entries are matched relative to the project root, `--sig` and `--add-sig` entries relative to the current directory.

Propose it when the signature tree carries generated or stale `.rbs` files that should not load.

## libraries

`libraries` — `Array[String]`, default `[]`. Example: `libraries = ["json", "prism"]`

Bundled or gem-provided RBS libraries to load by name, including manifest dependencies.

Propose it when checked code calls APIs from stdlib or gems and those RBS libraries should be available project-wide.

## inline

`inline` — `Boolean`, default `true`. Example: `inline = false`

Controls whether `# @rbs` inline annotations in Ruby files participate in type checking.

Propose `false` only when the project wants sig-only checking and accepts that inline annotation syntax errors will be hidden from normal checks.

## incremental

`incremental` — `Boolean` or the string `"verify"`, default `false`. Example: `incremental = true`

Opts into the incremental check cache: `crema check` records what each file read from the type environment under `.crema/cache/`, and the next run rechecks only the files a change can reach, replaying stored diagnostics for the rest. The output stays identical to a full check — the cache decides which files to skip, never what the checker sees — and anything that could change the environment (crema version, lockfile, `crema.toml`, sig paths), or a cache it cannot read, falls back to a full check. It never activates for `crema check -e`, and a run narrowed by path arguments reads the cache without writing it back.

Whether `true` pays off is a question of repetition before size. Only the checking of Ruby files is skipped — signatures load and the environment is rebuilt every run — so the saving lands where one working tree is checked over and over (an edit-check loop, a watch loop, a pre-commit hook), while a one-shot CI run has no cache to begin with. On a small scope a full check already sits near that floor, so the win narrows, but it does not invert.

`"verify"` does everything `true` does and additionally rechecks each replayed file behind the scenes, comparing it against the stored result. A mismatch means the cache recorded too little — the one failure that can break the guarantee above, and one that shows up as a *missing* error rather than a crash. It reports on stderr and fails the run with exit 1 even when the check is clean. Slower than a full check by construction, it is a watch to run deliberately, not a mode to leave on.

Propose `incremental = true` for a project large enough that checking is the slow part and checked repeatedly from the same tree, and add `.crema/` to `.gitignore` alongside it. Propose `"verify"` only for a deliberate correctness watch.

## did_you_mean

`did_you_mean` — `Boolean`, default `false`. Example: `did_you_mean = true`

Fills the `did_you_mean` field of `Ruby::UnknownConstant` with spelling suggestions drawn from the constants in scope. Off, the field is absent from the line, as it is whenever there are no suggestions. The suggestion is a spell-check of every constant visible at the failure site, repeated per diagnostic, so on a large tree with many unresolved constants it can cost as much as the type check itself.

Propose `true` only when unresolved-constant typos are a recurring failure mode and the extra time per check is acceptable.

## diagnostic

`diagnostic` — `Table`, default `{ preset = "default" }`. Example: `diagnostic = { preset = "default" }`

Configures diagnostic severities. `preset` selects a baked-in table: `default` keeps crema's built-in severities, `all_error` promotes normal configurable diagnostics to `error` (dev-only diagnostics stay ignored unless named explicitly), and `all_ignore` demotes every configurable diagnostic to `ignore`. Per-code overrides in the same table accept `error`, `warning`, `information`, `hint`, or `ignore`, and win over the preset for configurable codes; always-error diagnostics such as `Ruby::SyntaxError` cannot be overridden.

Propose `preset = "default"` unless the user explicitly wants a migration policy such as strict CI (`all_error`) or broad temporary suppression (`all_ignore`).

## collection_config

`collection_config` — `String`, default `none (auto-discovered)`. Example: `collection_config = "rbs_collection.yaml"`

Points crema at an explicit `rbs_collection.yaml`; the lockfile path is derived from that config path. By default, crema uses automatic walk-up discovery for `rbs_collection.yaml`.

Propose this key when the project uses rbs collection but the file automatic discovery would choose is not the intended one.

## infusion

`infusion` — `Table`, default `absent`. Example: `infusion = { rails = { enabled = true } }`

Synthesizes declarations for methods and constants that metaprogramming or static data files would produce at runtime, in memory, without writing any RBS. Each sub-table below is an independent opt-in and the whole feature is inert while `[infusion]` is absent. Unknown names anywhere under `[infusion]` are a parse error rather than a silent no-op, and a value crema cannot act on (an empty `decorator_suffix`, an empty inflection entry) is rejected at config load even when the provider it belongs to is disabled.

### [infusion.rails]

- `enabled` — `Boolean`, default `false`. Turns on the ActiveSupport, ActiveModel, and ActiveRecord rule families together, plus the zeitwerk-style constant synthesis that runs after them. The unit of choice is the framework, not the individual DSL rule.
- `inflections` — `Table`, default `absent`. Holds locale sub-tables; only `en` exists (see below).

Propose this table when the project uses Rails.

### [infusion.rails.inflections.en]

Mirrors the `inflect.` directives in `config/initializers/inflections.rb`, which crema does not parse itself — transcribe them by hand.

- `irregular` — `Array[[String, String]]`, default `[]`. Example: `irregular = [["person", "people"]]`. Each pair is `[singular, plural]` and overrides the default inflection for that word, the way `inflect.irregular` does. Neither string may be empty.
- `acronym` — `Array[String]`, default `[]`. Example: `acronym = ["API", "HTTP"]`. Mirrors `inflect.acronym`, reshaping the camelize result for table names containing these words. Entries may not be empty strings.

`plural` / `singular` regex pairs and `uncountable` are not supported, and any locale key other than `en` is a parse error. The table is accepted and validated even when `infusion.rails.enabled` is `false`; it simply has nothing to apply to until the ActiveRecord pipeline runs.

Propose it when that initializer declares `inflect.irregular` or `inflect.acronym`.

### [infusion.config]

Synthesizes a constant whose methods mirror the keys of one or more YAML files, so `Settings.database.host` type-checks against the real file.

- `files` — `Array[String]`, default `[]`, but at least one entry is required once the table exists. Example: `files = ["config/settings.yml", "config/settings/production.yml"]`. Files merge in order with later entries winning. Each must be pure YAML — ERB or any other preprocessing fails to parse — with a mapping at the root. Duplicate keys inside one file resolve last-wins like Psych and report a warning diagnostic rather than failing.
- `const_name` — `String`, default `"Settings"`. Example: `const_name = "Config"`. Must be a single constant name: it starts with an ASCII uppercase letter, continues with letters, digits, or `_`, and may not contain `::`.
- `except_keys` — `Array[String]`, default `[]`. Example: `except_keys = ["defaults"]`. Drops every key of that name at any nesting depth, not just at the root. Use it for keys that are YAML anchors or scaffolding rather than settings.

Every remaining key becomes a method, so each one must read as a Ruby method name (leading letter or `_`, then letters, digits, `_`, with a trailing `?` or `!` allowed); nested mappings become interfaces. A key that cannot be a method name is a hard error, which is the intended signal that the file needs `except_keys` or a rename.

Propose it when the project reads settings from a pure-YAML file.

### [infusion.active_decorator]

Covers the [active_decorator](https://github.com/amatsuda/active_decorator) gem, whose decorator modules are written as if the decorated model's methods were their own.

- `enabled` — `Boolean`, default `false`. When true, a module declared under `app/decorators` whose name ends with `decorator_suffix` gains the same-named model as its self type, so `UserDecorator` resolves `User`'s methods.
- `decorator_suffix` — `String`, default `"Decorator"`. Example: `decorator_suffix = "Presenter"`. Mirrors the gem's own `ActiveDecorator.config.decorator_suffix`. An empty string is rejected: every module name ends with it, so each module would be handed itself as its model.

The search directory is fixed at `app/decorators` and is not configurable — that is what keeps identically named modules from gem code out of scope. A decorator whose model is not declared anywhere is skipped silently, and an explicitly written self type is never overwritten.

Propose it when the Gemfile depends on active_decorator or the project has an `app/decorators` directory, and set `decorator_suffix` to match the initializer when one changes it.

### [infusion.paranoia]

Covers the [paranoia](https://github.com/rubysherpas/paranoia) gem, whose `acts_as_paranoid` class-body call adds soft-delete methods (`restore!`, `paranoia_destroyed?`, `with_deleted`, `only_deleted`, ...) at runtime.

- `enabled` — `Boolean`, default `false`. When true, every ActiveRecord model whose class body calls `acts_as_paranoid` gains `include Paranoia::InstanceMethods[Model]` and `extend Paranoia::ClassMethods[Model, Model::ActiveRecord_Relation]`, and its synthesized relation and collection-proxy classes gain the same `ClassMethods` include — the mixins the gem's RBS documentation says to write by hand.

Requires `[infusion.rails] enabled = true`; enabling paranoia alone is a config error, because the synthesized mixins reference the relation classes only the Rails preset declares. The `Paranoia` module types themselves are not synthesized — they come from the rbs collection (the `paranoia` gem's entry), and when they are absent the mixins quietly resolve to nothing. `acts_as_paranoid` options (`column:` and friends) do not change any signature and are not interpreted.

Propose it when the Gemfile depends on paranoia and models call `acts_as_paranoid`.
