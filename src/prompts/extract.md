# crema doc extract

`crema extract` runs the same pipeline as `crema check` and exports the per-file facts the check computed — definitions, implements, state-tagged method-call and constant-read sites, and consulted symbols — as a single JSON document on stdout. Pipe it into jq or redirect it to a file (`crema extract > extract.json`) and build tools (references, dead-code detection, call coverage, dependency graphs) on top, without crema growing a subcommand per use case. The document is large — tens of MB on a big project, one record per call site — so do not read it straight into a context window; always redirect or pipe. stdout is one JSON value and nothing else (warnings go to stderr).

## Scratch queries (`-e`)

`crema extract -e '<ruby code>'` answers "what type does this expression have?" for code that is not in a file yet. It builds the same environment as a bare `crema extract` (so project types are in scope), and checks only the snippet — `files` holds exactly one entry, keyed `"-e"`. Nothing under `.crema/` is read, written, or created, so the command works in a directory with no `crema.toml` at all.

```
crema extract -e '[1, 2].map { _1.to_s }' | jq -r '.files["-e"].method_call[].return_type'
```

Type errors in the snippet are not reported here — extract discards diagnostics in every mode; run `crema check -e '<ruby code>'` for those. A snippet that fails to parse is not an error either: its entry comes back with every record array empty, indistinguishable from code that simply had nothing to record, so use `crema check -e` when an empty answer is surprising.

## Document shape

```json
{
  "version": 6,
  "root": "/absolute/path/to/project",
  "files": {
    "lib/user.rb": { "content_hash": "…", "definitions": [], "implements": [], "method_call": [], "constant": [], "consulted": [] }
  }
}
```

- `version` — Integer. Schema version of the document; this page describes version 6.
- `root` — String. Base directory the `files` keys are relative to.
- `files` — Object. One entry per file in the check scope, keyed by relative path. Files that fail to parse still get an entry (with whatever facts were recoverable), so the key set is the full scope.

Every symbol string in the document is absolute RBS syntax: `::User` (type), `::User#save` (instance method), `::User.new` (singleton method), `::Billing::Invoice` (namespaced type). Global variables (`$foo`) have no absolute RBS spelling and are absent.

All `start_byte`/`end_byte` pairs are byte offsets into the file; `end_byte` is exclusive.

## File record

- `content_hash` — String. Hash of the file bytes as fixed-width lowercase hex. Compare against a fresh hash of the working-tree file to detect staleness before trusting byte offsets.
- `definitions` — Array. Symbols this file declares.
- `implements` — Array. Ruby `def` sites and constant assignments in this file.
- `method_call` — Array. Every call/super site in this file, tagged with a `state`.
- `constant` — Array. Every constant read site in this file, tagged with a `state`.
- `consulted` — Array of String. Every symbol the check of this file touched.

### definitions

Symbols declared by the project's own files only — declarations from gems and core never appear. Type-level records span the declaration name; method records span the defining member.

- `symbol` — String.
- `kind` — String: `class` / `module` / `interface` / `class_alias` / `module_alias` / `type_alias` / `constant` / `instance_method` / `singleton_method`.
- `start_byte`, `end_byte` — position of the declaration.
- `method_type` — String, only on method records (absent otherwise). The declared signature in RBS syntax; overloads are joined with ` | `.

A method defined in one file but re-declared or aliased in another appears under the file that contributed each declaration, so one symbol can have records in several files.

### implements

One record per Ruby `def` / `def self.` site and per constant assignment (`FOO = 1`, `A::B = 2`) — the implementation counterpart of `definitions`, which only sees the declaration space. In a project whose signatures live in separate `.rbs` files, a `.rb` file has empty `definitions`; `implements` is how "which file implements this symbol" stays answerable. To map an implementation to its declaration, join `implements[].symbol` against `definitions[].symbol` across files.

- `symbol` — String. The enclosing class/module plus the name (the lexical position of the site, not the declaration it was checked against): a `def bar` in `class Sub` records `::Sub#bar` even when only a superclass declares `bar`. A site with no declaration anywhere is still recorded. Top-level defs record as `::Object` instance methods, which is what Ruby makes them; top-level constants record rooted (`X = 1` is `::X`), which is how RBS spells them. A constant path is concatenated as written, never resolved: `A::B = 2` inside `class Foo` is `::Foo::A::B`, and `::A::B = 2` stays `::A::B`. A constant assigned inside `class << self` or a `Class.new do … end` block is placed under the enclosing class (`::Foo::S`, `::Ctor::INNER`), matching the check's own scope and inline `definitions`.
- `kind` — String: `instance_method` / `singleton_method` / `constant` (same vocabulary as `definitions` records). `def self.foo` and defs inside `class << self` are `singleton_method`.
- `start_byte`, `end_byte` — span of the whole `def` node (keyword through `end`) or the whole assignment (name through the end of the right-hand side).

Only literal `def` sites appear: methods synthesized by `attr_accessor`, `define_method`, or `alias` have no record, and a `def` on a singleton object (`def obj.foo`) is skipped — its owner has no absolute RBS spelling. Likewise only plain `=` assignments to a static constant path appear: `FOO ||= x`, `FOO += 1`, and a write through a dynamic parent (`obj.foo::BAR = 1`) have no record. As with `definitions`, one symbol can have records in several files (a method redefined in a reopened class, a constant reassigned), and each site is its own record.

### method_call

One record per call/super site the check touched, tagged with a `state`. A union-typed receiver dispatches into every component, so one source site can have several records (one per landing definition), sharing the same span and `return_type`.

- `state` — String. One of:
  - `typed` — the call resolved and the check emitted no diagnostic for it.
  - `error` — the call resolved (so `symbol` is present) but the check diagnosed the call itself: argument mismatch, unresolved overloading, visibility, block misuse. One record per site no matter how many diagnostics fired; the details live in `crema check` output. A diagnostic inside a block body belongs to the inner site, never to the outer call.
  - `no_method_error` — the receiver type resolved but the method was not found (the NoMethod diagnostic fired). No `symbol` — there is no landing definition. Like every state, `return_type` follows the null-means-uncomputed rule: `"untyped"` when the call sits in value position, `null` in statement position.
  - `untyped` — the receiver was untyped, so no resolution was attempted. This is the honesty column: report "N confirmed + M unknown" instead of silently dropping these sites.
- `kind` — String: `call` / `super`.
- `start_byte`, `end_byte` — span of the call node (never a diagnostic's narrowed span).
- `method_name` — String. Bare method name, present in every state.
- `receiver_type` — String. Static type of the receiver at the site, RBS syntax (e.g. `::User`, `singleton(::User)`, `self`, a union, a generic instantiation). For `kind: "super"` the enclosing self class; `"untyped"` for the `untyped` state.
- `return_type` — String or null. The expression type the check computed for the site (the call's return type; safe-navigation nil widen included, union receivers joined). `null` when the check never computed a value for the site (e.g. a statement-position call whose value is discarded) — extract reports what the check computed and runs no inference of its own. A diagnosed site follows the same rule.
- `symbol` — String, `typed` / `error` states only (key absent otherwise). The definition the call lands on: the owner that implements the method (which may be an ancestor of the receiver's class). The axis is where the implementation lives, not who declared the type: if `Sub < Base` has a `def bar` with no declaration of its own, a call on a `Sub` value records `::Sub#bar` even though `bar`'s type comes from `Base`; a `Sub` that does not override records `::Base#bar`. For a call through an interface-typed receiver, the declaring interface (`::_Speak#speak`) — the interface is the only nameable owner, so the site is kept rather than dropped.

`method_call` records every dispatched call, including the synthetic dispatches a compound write implies: `h[k] ||= v` records both `[]` and `[]=` on the write line, and an operator write (`x += 1`, `@x += 1`) records the operator method (`+`). One exception remains: `case`/`when` narrows without recording an implied `===` (crema performs no dispatch there). A compound write on an untyped receiver has no record (only real call sites feed the `untyped` state), and a site the check itself stays silent about (e.g. a control-flow-bottom receiver) has no record either.

### constant

One record per constant *read* site the check touched, tagged with a `state` — the constant sibling of `method_call`'s taxonomy. Reads only: the target of an assignment (`X = 1`, `A::B = 2`) belongs to `implements`, and the superclass position of `class Sub < Base` records nothing here (an extension point, not an omission to work around).

- `state` — String. One of:
  - `typed` — the constant resolved and the check emitted no diagnostic for the read.
  - `error` — the read resolved (so `symbol` is present) but the check diagnosed the site itself; today that is a read of a constant marked deprecated.
  - `unknown_constant` — resolution was attempted and found nothing (the UnknownConstant diagnostic fired). No `symbol` — there is nothing to name.
  - `untyped` — resolution was never completed: a dynamic parent (`Billing.name::Dyn`) that cannot be walked statically, or an untyped value partway down the path. This is the honesty column, as in `method_call`.
- `start_byte`, `end_byte` — span of the constant node. A path node spans the whole `A::B::C`, not just the leaf.
- `path` — String. The constant as written in source (`Billing::MAX`), keeping the leading `::` of an absolute path. Present in every state — it is all a failed resolution has to offer.
- `symbol` — String, `typed` / `error` states only (key absent otherwise). The constant the read resolved to, in absolute RBS syntax (`::Billing::MAX`).
- `type` — String. The type the check computed for the read (`::Integer`, `singleton(::Billing)`). The key is absent in the `unknown_constant` and `untyped` states — unlike `method_call`'s `return_type`, which spells an uncomputed value `null`.

A constant used as a call receiver records on its own span, so `Billing.name::Dyn` yields two records: `typed` for `Billing`, and `untyped` for the whole path. Only the read itself records, once — the check's internal peeks at a receiver do not each add a record.

### consulted

Deduped, sorted, positionless symbol strings: every type and method the check consulted while checking this file, including implicit dependencies that never appear as source occurrences — ancestor chains, alias expansion, and the types referenced by the file's own `def` signatures. Symbols whose lookup found nothing are included too; a miss is still a dependency. Dependency-graph and invalidation consumers must read this array, not `method_call`.

## Recipes

Each recipe reads a saved document (`crema extract > extract.json`); replace `extract.json` with `<(crema extract)` or pipe directly for one-shot queries.

Per-file call sites by state:

```bash
jq -r '.files | to_entries[] | [.key, ([.value.method_call[] | select(.state == "typed")] | length), ([.value.method_call[] | select(.state != "typed")] | length)] | @tsv' extract.json
```

All call sites landing on one method:

```bash
jq -r '.files | to_entries[] | .key as $f | .value.method_call[] | select(.symbol == "::User#save") | "\($f):\(.start_byte)"' extract.json
```

Dead-definition candidates — project-declared methods no file's call sites or consulted set mention (verify candidates manually; reflective and external entry points have no call sites):

```bash
jq -r '
  ([.files[].method_call[].symbol // empty] + [.files[].consulted[]] | unique) as $used
  | .files | to_entries[] | .key as $f
  | .value.definitions[]
  | select(.kind | endswith("_method"))
  | select(.symbol as $s | $used | index($s) | not)
  | "\($f):\(.start_byte)\t\(.symbol)"
' extract.json
```
