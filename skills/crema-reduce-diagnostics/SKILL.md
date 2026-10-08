---
description: Reduce `crema check` diagnostics in a Ruby project against the baseline file named by `baseline` in its `crema.toml`, one diagnostic code and a few files per cycle. Reads crema's own baseline matching after every edit so the agent sees exactly what it removed, what it newly introduced, and what surfaced from the same root cause. Use when a project keeps a crema baseline (rubocop_todo.yml style) and wants it to shrink without regressions.
license: MIT
metadata:
    github-path: skills/crema-reduce-diagnostics
    github-repo: https://github.com/ksss/crema
name: crema-reduce-diagnostics
---
# crema-reduce-diagnostics

One cycle = one diagnostic code, a few files. The baseline says what is
known, crema's matching says what changed, the human approves the
diff. Counts never decide anything; the matching does.

## Why this shape

- **The baseline is the contract.** `baseline` in `crema.toml` names a
  file of `file`, `code`, `fingerprint` rows, stable under line shifts
  and diffable in git. `crema check` drops every record that matches a
  row, so its stdout is exactly what the baseline does not know, and one
  stderr line counts the rows that no longer match anything. Every cycle
  starts by matching the baseline and ends by regenerating it. A cycle
  that skips either step cannot say what it changed.
- **A code is too wide, a file is too narrow.** One code across a whole
  project is hundreds of records with several root causes; one file
  mixes codes with unrelated fixes. A code plus the files where it
  clusters is the unit a reviewer can read in one sitting.
- **Introduced beats removed.** Fixing a NoMethod on `x.foo` often makes
  crema see further: an ArgumentTypeMismatch or UnresolvedOverloading
  that was silent behind the error now appears, sometimes in a file you
  did not touch. crema prints those as records the baseline does not
  know. A cycle reports every new record with a verdict; a cycle that
  only reports the count going down is not finished.
- **The fix for a code lives in `crema doc`.** Run
  `crema doc diagnostic <code>` before the first edit; its *Typical fix*
  is the catalogue. This skill holds the loop, not the fixes.

## Inputs

- A project whose `crema.toml` has `baseline = true` (the file is
  `crema_baseline.jsonl` next to `crema.toml`) or `baseline = "<path>"`
  (relative to the `crema.toml` directory), and that file committed.
  `<baseline>` below is that file. `crema doc config` describes the key.
- `jq`, `comm`, `grep`.
- Three ways to run the check, used for different questions:
  - `crema check` — only the records the baseline does not know, every
    field; exit `1` if any of them is an error.
  - `crema check --no-baseline` — every record, as if there were no
    baseline.
  - `crema check --update-baseline` — rewrite `<baseline>` from the
    current records; exit `0`, prints nothing.

### No `baseline` key

Stop before step 0 and ask the human. Turning the key on changes what a
bare `crema check` prints and its exit code (a project with 100 known
records goes from exit `1` to exit `0`, in CI too), and that is the
project's decision, not the agent's. Propose the one line and say what
it changes:

- No baseline of any kind: `baseline = true` at the top level of
  `crema.toml` (before any `[table]`), then
  `crema check --update-baseline` and commit the new file on its own.
- A hand-kept `crema check --tamp > <file>` baseline (often behind a Rake
  task): `baseline = "<file>"`. The rows are the same format, so the
  existing file works as it is; no regeneration. Mention that the Rake
  task can then be replaced by `crema check` and
  `crema check --update-baseline`, and leave that to the project.

When the key is on but the file does not exist (`crema check` exits `2`
with `baseline file not found`), the decision is already made: run
`crema check --update-baseline` and commit the file on its own, nothing
else in that commit. A baseline born in the same commit as a fix cannot
show what the fix changed.

## Cycle

### 0. Match the baseline

Never read `crema check` output through `head`; the run is JSONL and a
cut stream is a wrong count with no warning. Redirect to a file, then
query the file.

```sh
crema check > /tmp/new.jsonl 2> /tmp/check.err
grep '^baseline:' /tmp/check.err  # must print nothing
wc -l < /tmp/new.jsonl            # must print 0
```

Both together mean the current records and the baseline rows are the
same set. A record in `/tmp/new.jsonl` or a `baseline: N entries no
longer reported` line means the baseline is stale (someone edited
without regenerating, a different crema version, a gem update). Stop and
regenerate it with `crema check --update-baseline` as its own commit
before any fix; otherwise the diff at step 4 mixes your work with the
drift. Other stderr lines (warnings) are not drift.

Also note `git status` and `crema --version` now. Files already modified
before the cycle are not yours: leave them alone and name them in the
report. The version goes in the report too, because a baseline is only
comparable across runs of the same crema.

`.crema/` is the gem-environment snapshot. It does not hold per-file
diagnostics, so a record you did not expect is never a cache problem;
do not delete it to "retry".

### 1. Pick one code and its files

The baseline hides every known record from a bare `crema check`, so
picking reads the full set:

```sh
crema check --no-baseline > /tmp/crema.jsonl
jq -r .code /tmp/crema.jsonl | sort | uniq -c | sort -rn
jq -r 'select(.code=="<Code>") | .file' /tmp/crema.jsonl | sort | uniq -c | sort -rn
```

Take the code the human named, or the top error-severity code. Then
take the files at the top of the second list until the record count is
what one review can read (10-20 records is a cycle; 50 is two). Note
`severity`: `hint` and `information` records are advice, `error` records
are the ones a CI gate reads. Do not spend a cycle on hints while errors
remain in the same file.

Read the doc for the code, then the records for the chosen files with
every field:

```sh
crema doc diagnostic <Code>
jq -c 'select(.code=="<Code>" and (.file|test("<pattern>")))' /tmp/crema.jsonl
```

`receiver_type`, `missing_from`, `defined_in`, `actual` / `expected`
name the root. Group the records by root before editing: five NoMethod
records with `missing_from: ["nil"]` on `Location` receivers are one
fix pattern applied five times, not five fixes.

### 2. Sort the roots

For each group decide, in this order:

1. **A gem with no signatures at all** — the receiver is an
   `UnknownConstant` from a gem, or `missing_from` names a gem constant
   and neither the gem's own `sig/` nor `.gem_rbs_collection/<gem>`
   exists. Out of scope: typing a whole gem is its own job with its own
   rules. Leave the group in the baseline and name it in the report
   (a `crema-rbs-from-diagnostics` skill exists for that job; this skill
   does not depend on it).
2. **A gap in a signature that exists** — the gem's `sig/` or the
   collection declares the class but not this method, and the method is
   in the gem's source:
   ```sh
   GEM=$(bundle exec ruby -e 'puts Gem.loaded_specs["<gem>"].full_gem_path')
   grep -rn "def <m>\b" "$GEM/lib"
   bundle exec ruby -e 'require "<gem>"; p <Path>.instance_method(:<m>).parameters'
   ```
   Declare that one method in `sig/gem-patch/<gem>/<file>.rbs` under its
   owner module (the class or module the `def` sits in, not the class
   the project calls it through), with arity transcribed from
   `.parameters` (`:req` → positional, `:opt` → `?`, `:rest` → `*`,
   `:key`/`:keyreq` → keyword, `:keyrest` → `**untyped`, `:block` →
   `?{ ... }`) and every type `untyped` unless the return is a literal,
   `self` or `Foo.new` on the line. The patch loads through `sig` in
   `crema.toml`, a top-level key, so before any `[table]`:
   ```toml
   check = ["lib"]
   sig = ["sig/gem-patch"]
   ```
   Validate with `bundle exec rbs -I sig/gem-patch validate` before
   re-checking. A gap in the collection is also an upstream contribution;
   say so in the report, the patch is the local fix.
3. **Project code** — the receiver type is right and the code does not
   handle it (`Location?` used without a guard, a `nil` initializer pinned
   by a block, a block parameter declared `?{ }` and called without
   `&.`). Fix in the project, one pattern at a time. The *Typical fix*
   in the doc is the guide; `case`/`when` narrowing is a fix, and it can
   itself raise `Crema::NonExhaustiveCase` (a hint) when the `case` has no
   `else`, which step 4 will show.
4. **Wrong signature in the project's own RBS or inline annotation** —
   fix the annotation, not the call site.
5. **Checker bug** — the code is right by Ruby semantics, the signature
   is right, crema still reports. Confirm by running the Ruby (a
   one-liner or the project's test), then leave the record in place and
   report it to crema. Never bend code or a signature to silence a
   record you believe is wrong.

### 3. Edit, re-check the touched files

crema takes file arguments, so iterate on the files of this cycle
(milliseconds) and run the whole project once at the end. Two runs
answer two questions:

```sh
# what is left to fix in these files, known or not
crema check --no-baseline <file1> <file2> | jq -c '{file, line, code, severity, message}'
# what in these files the baseline does not know: introduced, surfaced, moved
crema check <file1> <file2> | jq -c '{file, line, code, severity, message}'
```

The second list should stay empty or hold only records you can already
explain; step 4 gives each one a verdict. A run with file arguments
never prints the `baseline:` stale line (it cannot see the rest of the
project), so removals are counted at step 4.

Every edit is the smallest change that handles the type. An `or return`
/ `or next` after a nilable read, a `&.` on an optional block, an inline
`#: T?` on a variable the block reassigns. No refactoring around the
fix, no renaming, no reformatting of neighbouring lines: the fingerprint
hashes the line, so an unrelated edit on a line with a baseline record
turns that record into a "removed + added" pair in step 4 and costs the
reviewer a look.

### 4. Read the match

After the last edit of the cycle, the whole project:

```sh
crema check > /tmp/added.jsonl 2> /tmp/check.err  # records the baseline does not know
grep '^baseline:' /tmp/check.err                   # "baseline: N entries no longer reported"
crema check --tamp > /tmp/after.tamp               # every record as a baseline row
LC_ALL=C comm -23 <baseline> /tmp/after.tamp > /tmp/removed.jsonl
wc -l < /tmp/removed.jsonl                         # must equal N (0 when no line)
jq -r .code /tmp/removed.jsonl | sort | uniq -c
jq -c '{file, line, code, severity, message}' /tmp/added.jsonl
```

`/tmp/added.jsonl` is crema's own answer, with every field. The
`removed` rows come from the baseline itself, because crema only counts
them; `--tamp` prints the same rows the baseline holds, in the same
byte order (`LC_ALL=C` keeps `comm` on that order). If the removed count
is not N, the baseline file was edited by hand or is not crema's: stop
and say so instead of reading the lists.

Read `added` to the last record and give each one a verdict:

- **surfaced** — same file as a removed record, a different code, the
  same root (the `Location?` you guarded is now passed to a method that
  wants `Location`). It was always there; the fix let crema see it. Fix
  it in this cycle if it is the same pattern, otherwise leave it and say
  so.
- **introduced** — a record your edit created: a `case` without `else`
  (NonExhaustiveCase), a guard that made a later expression `Integer?`
  (UnresolvedOverloading on `+`). Fix or justify; an unexplained
  introduced record fails the cycle.
- **moved** — a baseline record on a line you edited for another reason;
  same code, same file, same message as one in `removed`. Fine, name it.

Then run the project's tests. A type fix that changes behaviour
(`or return` where the code used to raise) is a behaviour change, and
only the tests say whether the project accepts it.

### 5. Regenerate the baseline and report

```sh
crema check --update-baseline
```

Then step 0's three lines once more: nothing printed, `0`. That is the
proof the new baseline matches the tree you hand over.

The report for the reviewer:

- Code and files of the cycle; records removed, by root pattern
  (`Location?` unguarded ×12, `each_cons` element ×6, ...).
- Every `added` record with its verdict, down to the last one.
- Gem-patch declarations written (owner, method, why any type is not
  `untyped`); groups left for a whole-gem signature job; checker bugs
  left in place with the Ruby evidence.
- Test result, verbatim summary line; the crema version.
- The next cycle's code and its top files, from step 1 run again on the
  regenerated tree.

The baseline shrinking is the visible progress; the `added` list with
verdicts is what the reviewer reads.
