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
- **The cascade is the target.** `class Base < Grape::API::Instance` is one
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

## Cycle

### 1. Rank: build the worklist

```sh
ruby <skill dir>/scripts/worklist.rb /tmp/crema.jsonl --top 30
```

Output is TSV: `path` (the constant as written), `unknown_constant` (raw
count), `cascade_no_method` (NoMethod records whose receiver inherits
from or mixes in that path), `total`, and `roots` (up to three
`file:line` sites where the path is a superclass / include / extend /
prepend). The script parses the project's `.rb` files under the
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
   A path under the gems directory confirms it. Pick one gem for this
   cycle: the highest `total` whose class is gem.

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

Scope: only the constants and methods the project actually uses.

```sh
grep -rhoE "<Path>(::[A-Z][A-Za-z0-9_]*)*\.[a-z_]+[!?]?" <check dirs> | sort | uniq -c | sort -rn
```

Get the structural facts from the runner, then read the gem source for
each method:

```sh
bundle exec ruby -e 'require "<gem>"
  p <Path>.ancestors.first(3)
  p <Path>.singleton_class.instance_methods(false).sort
  p <Path>.instance_methods(false).sort'
```

Rules for every declaration:

- Superclass and mixins come from `.ancestors`, never from guessing.
- A method gets a concrete type only when you can cite the defining line.
  Put the citation on the line above as a comment:
  `# batch_loader.rb:17  def self.for(item)`.
- Anything you cannot cite, or whose behaviour is dynamic
  (`method_missing`, `define_method` over a runtime list, a proxy that
  forwards to a value of unknown type) is `untyped`. For a method whose
  parameters you cannot read, write `(?) -> untyped`.
- Do not turn `foo?` into `bool` because of the name. Ruby returns
  non-boolean truthy values from `?` methods all the time; if the source
  returns `@parent`, the type is that ivar's type or `untyped`. The same
  goes for parameters: a `cache: true` default that is only tested for
  truthiness is `boolish`, not `bool`.
- A Ruby `&block` parameter is optional; an RBS `{ ... }` block is
  required. Write `?{ ... }` unless the source raises or yields
  unconditionally, or every call site will get `RequiredBlockMissing`.
- Never write into the gem directory or `.gem_rbs_collection/`.

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

```sh
crema check > /tmp/crema.after.jsonl
jq -r .code /tmp/crema.jsonl | sort | uniq -c | sort -rn > /tmp/before.txt
jq -r .code /tmp/crema.after.jsonl | sort | uniq -c | sort -rn > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
# still unresolved under this gem's path
jq -c 'select(.code=="Ruby::UnknownConstant" and (.path|startswith("<Path>")))' /tmp/crema.after.jsonl
# newly surfaced now that the receiver has a type
jq -c 'select(.receiver_type? // "" | contains("<Path>"))' /tmp/crema.after.jsonl
```

Newly surfaced NoMethod / ArgumentTypeMismatch records are the point of
the exercise: they were silent while the receiver was untyped. Give each
one of three verdicts:

- **signature gap** — the gem defines it, the patch does not yet (a
  missing method, a block declared required, a nil the source can
  return). Fix it in the same cycle when the citation is at hand.
- **true positive** — the project calls something the gem does not
  define, or with the wrong arguments. Note that a call repeated after a
  nil guard (`return if X.current.nil?; X.current.foo`) is a true
  positive under a `() -> X?` signature: the second call is not narrowed.
- **checker bug** — the signature is right, the project code is right by
  Ruby semantics, and crema still reports. Confirm against Ruby (run the
  call) before deciding, and report it to crema rather than bending the
  signature to silence it. Example seen: braceless keyword arguments
  passed to a single positional parameter (`Foo.for(id: 1)` against
  `(untyped item)`).

A diagnostic whose receiver is a module the project prepends or includes
into the gem class (`module Ext; def x; __sync!; end; end;
Gem.prepend(Ext)`) needs a self-type constraint on the project side
(`module Ext : Gem`), not a line in the patch. Report it as app-side.

The report a human needs to approve the cycle:

- Gem and version, which constants and methods were declared, and how
  many declarations are `untyped` and why.
- Before / after counts per diagnostic code.
- Every remaining diagnostic on the gem's constants with its verdict
  (signature gap / true positive / checker bug / app-side / deliberately
  untyped).
- What the next cycle's top of worklist is.

Zero remaining diagnostics is not the goal. A remaining diagnostic with a
reason is a finished cycle; a `bool` written from a method name is not.
