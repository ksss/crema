---
name: rbs-from-diagnostics
license: MIT
description: Turn `crema check` diagnostics into RBS signatures for gems that ship none. Ranks unresolved constants by how many NoMethod diagnostics cascade from them, classifies each as stdlib / gem / app, and for one gem per cycle reads the gem source and writes `sig/gem-patch/<gem>/*.rbs` for a human to review. Use when a Ruby project's crema output is dominated by UnknownConstant / NoMethod on gem constants.
---

# rbs-from-diagnostics

One cycle = one gem. crema reports, the agent reads and writes, a human
approves. Nothing in this skill infers a type from a name; every concrete
type traces to a line of the gem's source, and anything not traced is
`untyped`.

## Why this shape

- **A wrong signature costs more than `untyped`.** crema is a preflight
  checker: a wrong signature produces false positives on every call site,
  `untyped` only produces silence. When unsure, write `untyped`.
- **A cycle is judged on accuracy, not on how many counts moved.** One
  wrong declaration outweighs a thousand cleared NoMethod records. The
  reviewer reads every line against the gem source; a line they cannot
  confirm is a failed cycle even when the totals look good.
- **The cascade is the target.** `class Base < SomeGem::Base` is one
  UnknownConstant, but it leaves every method called on `Base` and its
  subclasses unresolvable, so it owns thousands of NoMethod records. The
  worklist ranks by that weight, not by raw count.
- **Files are the contract.** The signatures live in the project, have an
  author (the agent) and an approver (the human), and can be reviewed
  line by line. `rbs prototype` is not used.

## Inputs

- A `crema check` JSONL file (run from the project root:
  `crema check > /tmp/crema.jsonl`).
- A runner for fact lookups, default `bundle exec ruby -e '...'`. Only if
  the gem cannot be required without the application, use
  `bin/rails runner '...'` instead. The runner is for facts
  (`Object.const_source_location`, `.ancestors`,
  `instance_methods(false)`), never for generating types.
- The RBS syntax reference ships with the rbs gem. When unsure how to
  write a construct (`[self: T]`, `?{ }`, `**untyped`, `attr_accessor`,
  a constant, an alias), grep it there instead of guessing:
  ```sh
  RBS=$(bundle exec ruby -e 'puts Gem.loaded_specs["rbs"].full_gem_path')
  grep -n "self:" "$RBS/docs/syntax.md"
  ```
  `rbs validate` (step 4) is the judge of what you wrote.

## Cycle

### 1. Rank: build the worklist

```sh
ruby <skill dir>/scripts/worklist.rb /tmp/crema.jsonl --top 30
```

Output is TSV: `path` (the constant as written), `unknown_constant` (raw
count), `cascade_no_method` (NoMethod records whose receiver inherits
from or mixes in that path), `total`, `roots` (up to three
`file:line` sites where the path is a superclass / include / extend /
prepend), and `smoke` (the file with the most cascading NoMethod
records, the one to re-check first in step 5). The script parses the project's `.rb` files under the
`check` entries of `crema.toml` with Prism, so a class chain is followed
transitively. Work from the top.

An app-defined name with a large cascade (for example a helper module
outside the `check` scope) is a scope gap in `crema.toml`, not a gem to
patch. Report it and move on.

### 2. Classify each path

Take the first segment of `path` and decide, in this order:

1. **stdlib** — the rbs gem ships it:
   ```sh
   RBS=$(bundle exec ruby -e 'puts Gem.loaded_specs["rbs"].full_gem_path')
   ls "$RBS/stdlib/<name>"      # e.g. yaml, ipaddr, open3
   ```
   Directory names are the require names, not a downcased constant:
   `OptionParser` is `optparse`, `Net::HTTP` is `net-http`. When the
   obvious name misses, `ls "$RBS/stdlib"` and look before deciding.
   Fix is config, not a signature: add the name to `libraries = [...]` in
   `crema.toml`.
2. **app** — the project defines it:
   ```sh
   grep -rn "^\s*\(class\|module\) <Name>\b" <check dirs>
   ```
   Out of scope for this skill. Report it (unresolved relative names such
   as `CreateService` inside `lib/api/*` usually mean a lexical-lookup or
   scope issue on the project side, not a missing gem).
3. **gem** — a gem defines it:
   ```sh
   bundle exec ruby -e 'require "<gem>"; p Object.const_source_location("<Path>")'
   ```
   A path under the gems directory confirms it.

Classify every row of the worklist, not only the first. Then pick one
gem for this cycle: the gem of the highest `total` row. Every other row
that classifies to the same gem is in scope this cycle too, whatever its
rank. A bare name with no roots (`Boolean`, `Types`) is often a constant
the gem sets on its base class (`Base::Boolean = SomeGem::Boolean`) and
reaches the project lexically through the superclass: one line in the
same patch, not a next cycle.

### 3. Check for existing signatures

Before writing anything:

```sh
GEM=$(bundle exec ruby -e 'puts Gem.loaded_specs["<gem>"].full_gem_path')
ls "$GEM/sig"                          # gem-bundled signatures
ls .gem_rbs_collection/<gem>           # rbs collection (if the project uses one)
```

If either exists, do not write a patch. Load the bundled `sig/` via
`libraries = ["<gem>"]`, or add the gem to `rbs_collection.yaml`. Only a
gem with no signatures anywhere gets a patch.

### 4. Write `sig/gem-patch/<gem>/*.rbs`

One directory per gem, named after the gem (`Gem.loaded_specs` name,
not the constant): a constant's gem is where `const_source_location`
put it in step 2, and a shared namespace does not mean a shared gem —
`Foo::Base` may be `foo/` while `Foo::Entity` is `foo-entity/`. Never add
a second gem's constants to a directory to keep a namespace together,
and never merge two gems into one file.

Scope: only the constants and methods the project actually uses. Calls
with an explicit receiver are a grep; the cascade of a root is not,
because the calls sit in subclasses with an implicit receiver, so ask the
worklist for them:

```sh
grep -rhoE "<Path>(::[A-Z][A-Za-z0-9_]*)*\.[a-z_]+[!?]?" <check dirs> | sort | uniq -c | sort -rn
ruby <skill dir>/scripts/worklist.rb /tmp/crema.jsonl --methods <Path>
```

The cascade list mixes three kinds of name. Sort them with the runner
before reading any source:

```sh
bundle exec ruby -e 'require "<gem>"
  %w[get params present current_user].each { |m|
    um = (<Path>.singleton_class.instance_method(m) rescue nil)
    puts [m, um&.owner, um&.parameters&.inspect, um&.source_location&.join(":") || "-"].join("\t") }'
```

- a gem path: declare it on `owner` (this cycle)
- an app path (`lib/api/helpers.rb`): app-side, report and skip
- `-`: the name is not a method of `<Path>` at all. Either it lives on
  the object a DSL block runs against (`present`, `requires`, `tags` —
  see *Blocks that switch self* below) or it is dynamic. Never declare it
  on `<Path>` to make the count drop.

The mixin edges come from the runner too. This prints, for every module
on the chain, the modules it includes directly — one line per RBS
`include` / `extend` edge, the singleton chain for class-level calls
(`.singleton_class.ancestors`), `.ancestors` for instance calls:

```sh
bundle exec ruby -e 'require "<gem>"
  chain = <Path>.singleton_class.ancestors.take_while { |m| m != Class }
  chain.each { |m|
    tr = m.ancestors.drop(1)
    direct = tr - tr.flat_map { |a| a.ancestors.drop(1) }
    puts [m.inspect, *direct.map(&:inspect)].join("\t") unless direct.empty? }'
```

Transcribe that list and nothing else. `ActiveSupport::Concern` makes a
Ruby `include` extend the class with `ClassMethods`; the runner output
already shows the result (`#<Class:Instance>` → `Routing::ClassMethods`),
so write `extend` on the class and do not reason about the hook.

Rules for every declaration:

- A method is declared on its `owner`, never on the class the project
  calls it through. When the runner says `owner` is `Foo::DSL::Settings`
  for a method the project calls on `Foo::Base`, the patch declares the
  module with an instance method and has `Base` `extend` it. The
  chain to reproduce is `.singleton_class.ancestors` for class-level
  calls and `.ancestors` for instance calls; mirror the `include` /
  `extend` edges, no more. A flattened patch is wrong twice: the
  method's `defined_in` lies, and the next gem class that mixes in the
  same module gets nothing.
- Arity and parameter kinds are transcribed from `.parameters`:
  `[:req, :key], [:req, :val]` is `(untyped key, untyped val)`, `:opt`
  is `?`, `:rest` is `*`, `:key` / `:keyreq` are keywords, `:keyrest` is
  `**untyped`, `:block` is `?{ ... }`. A parameter never becomes
  optional, and a `**` is never dropped, to make a diagnostic go away.
- Parameters are `untyped`. A return is `untyped` unless every path
  of the body returns something whose class is on the line: a literal
  (`true`, `nil`, `"..."`, `[]`), `self`, or `Foo.new`. That is the
  whole list. A name, a doc comment, a `to_sym`, a guard, a default
  value, or the way the project calls it pins nothing; `untyped` is
  the answer, not a weaker guess. The declaration itself is the
  citation: owner module plus method name finds the `def` by grep.
- The file has no comments. None: no header naming the gem version, no
  `# file.rb:NN`, no reason for an `untyped` or a `[self: T]`. The
  owner module plus the method name is the citation; everything else
  (why something is untyped, how a hook works, what a future cycle should
  do) goes in the report. A comment in the file is a sign the declaration
  is not certain enough to be in the file.
- Anything you cannot cite, or whose behaviour is dynamic
  (`method_missing`, `define_method` over a runtime list, a proxy that
  forwards to a value of unknown type) is `untyped`. For a method whose
  parameters you cannot read, write `(?) -> untyped`.
- Do not turn `foo?` into `bool` because of the name. Ruby returns
  non-boolean truthy values from `?` methods all the time; if the source
  returns `@parent`, the type is `untyped`.
- A Ruby `&block` parameter is optional; an RBS `{ ... }` block is
  required. Write `?{ ... }` unless the source raises or yields
  unconditionally, or every call site will get `RequiredBlockMissing`.
- Declarations are nested (`module Foo` / `class API` / `class Base`),
  never the compact `class Foo::API::Base`, unless every outer namespace
  is declared elsewhere. rbs rejects the compact form (`rbs validate`:
  `Could not find ::Foo::API`); crema loads the file
  without a word and the constant stays unknown, so the re-check shows
  no change at all.
- Never write into the gem directory or `.gem_rbs_collection/`.

#### Blocks that switch `self`

A DSL block the gem runs on another object (`instance_eval`,
`instance_exec`, `class_eval`, `define_method(name, &block)` on another
class, `Scope.new(self, &block)`) does not see the class it is written
in. Cite that line and put the receiver on the block. Only when the
receiver is a named class or module: a block the gem runs on an anonymous
object (`Module.new.tap { |mod| mod.class_eval(&block) }`) gets no
`[self: T]`, and its calls stay NoMethod in the report (see *Unknown
beats wrong* below).

```rbs
module Foo
  module DSL
    module ClassMethods
      def route: (*untyped args) ?{ () [self: Foo::Context] -> untyped } -> void
    end
  end
end
```

Then declare, in the same patch, the methods the project calls inside
those blocks on that class (`Foo::Context#render`, `#status`). crema
resolves calls inside the block against the `self:` type. Without it
every call inside the block resolves against the outer class: a NoMethod
at best, and a wrong-arity report when the outer class has a method of
the same name (`content_type(key, val)` on the class, `content_type(val)`
on the context). Two same-named methods on two
receivers are two declarations; never merge them into one signature on
the outer class.

The file name ends in `.rbs`. Both rbs and crema scan a directory for
`.rbs` only and skip everything else without a word, so a `.rb` passes
`validate` and changes nothing. Validate the file with rbs before
touching crema; it catches a compact class name, a bad `include` target
and syntax slips in a second. Then confirm the constants are actually
loaded from the directory:

```sh
bundle exec rbs -I sig/gem-patch validate
bundle exec rbs -I sig/gem-patch list | grep '^::<Path>'
```

An empty `grep` means the file was not read (wrong extension, wrong
directory); fix that before running crema at all.

Then make crema load the directory. `crema.toml` does not read `sig/`
implicitly. `sig` is a top-level key, so it must come before any
`[table]` section, not at the end of the file:

```toml
check = ["app", "lib"]
sig = ["sig/gem-patch"]

[infusion.rails]
enabled = true
```

### 5. Re-check and report

Re-check the `smoke` file from the worklist before the whole project
(seconds, not minutes). It is the heaviest consumer of the path, so a
missing method, a wrong block `self` or a same-named method on two
receivers shows up there; the `roots` file is often a near-empty base
class that proves nothing beyond "the constant resolved". If its counts
do not move, the patch did not load — a `.rb` extension, a compact class
name, or a `sig` key placed after a table — and the full run would only
confirm that:

```sh
crema check <smoke file> | jq -r .code | sort | uniq -c
```

Iterate on single files; the whole project runs at most twice per cycle
(after the patch, after the gap fixes).

`.crema/` appears in the project root after the first run. It is
crema's incremental cache (`incremental = true` in `crema.toml`): the
parsed files and their diagnostics, keyed on the content of the `.rb`
and `.rbs` files, so an edit to the patch is picked up on the next run
by itself. It belongs in `.gitignore`, and it is what makes a re-run
take seconds instead of minutes. A result that looks stale is a patch
problem, not a cache problem.

Then the full run:

```sh
crema check > /tmp/crema.after.jsonl
jq -r .code /tmp/crema.jsonl | sort | uniq -c | sort -rn > /tmp/before.txt
jq -r .code /tmp/crema.after.jsonl | sort | uniq -c | sort -rn > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
# still unresolved under this gem's path
jq -c 'select(.code=="Ruby::UnknownConstant" and (.path|startswith("<Path>")))' /tmp/crema.after.jsonl
# newly surfaced now that the receiver has a type: NoMethod carries
# receiver_type, the argument diagnostics carry defined_in
jq -c 'select((.receiver_type? // .defined_in? // "") | contains("<Path>"))' /tmp/crema.after.jsonl
# NoMethod still cascading from the root (the receiver is a subclass, not <Path>)
ruby <skill dir>/scripts/worklist.rb /tmp/crema.after.jsonl --methods <Path>
```

Newly surfaced NoMethod / ArgumentTypeMismatch records are the point of
the exercise: they were silent while the receiver was untyped. Give each
one of four verdicts:

- **signature gap** — the gem defines it, the patch does not yet (a
  missing method, a block declared required, a nil the source can
  return). Fix it in the same cycle when the citation is at hand.
- **true positive** — the project calls something the gem does not
  define, or with the wrong arguments. Note that a call repeated after a
  nil guard (`return if X.current.nil?; X.current.foo`) is a true
  positive under a `() -> X?` signature: the second call is not narrowed.
- **block self** — the call sits inside a DSL block that the gem runs on
  another object, and the patch has no `[self: T]` on that block (or `T`
  lacks the method). A signature gap of the block kind, not a checker
  limitation: fix it in the same cycle.
- **checker bug** — the signature is right, the project code is right by
  Ruby semantics, and crema still reports. Confirm against Ruby (run the
  call) before deciding, and report it to crema rather than bending the
  signature to silence it. Example seen: braceless keyword arguments
  passed to a single positional parameter (`Foo.for(id: 1)` against
  `(untyped item)`).

#### Unknown beats wrong

Diagnostics are not equal. NoMethod and UnknownConstant on an untyped or
unknown receiver say "not known yet"; an argument diagnostic
(InsufficientPositionalArguments, UnexpectedPositionalArgument,
ArgumentTypeMismatch, RequiredBlockMissing) under a signature you wrote
says "wrong", and is read as a bug by whoever runs crema next. When a
choice between two declarations trades one for the other, keep the
NoMethod. The case that forces the choice: a DSL block that is
`class_eval`'d on an anonymous module and contains `def`s. The block's
own statements run with the module as `self`; the `def` bodies run later
with an instance of whatever class includes that module as `self`. One
`[self: T]` cannot name both, and either choice turns every correct call
site on the other side into an argument error. No `[self: T]` there, and
the NoMethod records it leaves are reported as block self with that
reason.

A diagnostic whose receiver is a module the project prepends or includes
into the gem class (`module Ext; def x; __sync!; end; end;
Gem.prepend(Ext)`) needs a self-type constraint on the project side
(`module Ext : Gem`), not a line in the patch. Report it as app-side.

The report a human needs to approve the cycle:

- Gem and version, which constants and methods were declared, and how
  many declarations are `untyped` and why.
- Before / after counts per diagnostic code.
- Every remaining diagnostic on the gem's constants with its verdict
  (signature gap / true positive / block self / checker bug / app-side /
  deliberately untyped), grouped by receiver and method, down to the
  last group — a top-N of the biggest groups is not a report.
- What the next cycle's top of worklist is.

Zero remaining diagnostics is not the goal. A remaining diagnostic with a
reason is a finished cycle; a `bool` written from a method name is not.
