# Ruby::DifferentMethodParameterKind

## Overview
A Ruby method parameter on the looser side (optional positional, rest, optional keyword, keyword rest) does not match the corresponding RBS slot.

## Trigger
Steep parity: emitted when the Ruby side is `:optarg`, `:restarg`, `:kwoptarg`, or `:kwrestarg` and the RBS slot the parameter lands on has a different kind (required, missing, rest at another position, etc.). When the Ruby side is the strict one (required positional / required keyword), [`Ruby::MethodParameterMismatch`](Ruby_MethodParameterMismatch.md) is emitted instead.

## Example
```ruby
def save(name, opts = {}); end
# RBS: def save: (String) -> void
```
Here `opts` is `:optarg` on the Ruby side but has no corresponding RBS slot.

## Typical fix
Align the Ruby parameter shape and the RBS method type. Either add the slot to the RBS declaration (`def save: (String, ?untyped) -> void`) or drop the looser Ruby parameter.

## Overloads are checked as one merged signature
Following Steep, the RBS overload set is folded into a single composite signature before the def-side check (`MethodType#+` / `merge_for_overload`): a slot required by only some overloads becomes optional in the merge. So `def rbs_location: (Prism::Location) | (Prism::Location, Prism::Location)` merges to `(Prism::Location, ?Prism::Location?)` and Ruby `def rbs_location(location, loc2 = nil)` matches silently. Role-split overloads (one positional-heavy, one keyword-heavy) melt the same way: `(Buffer, Integer, Integer) | (buffer: Buffer, start_pos: Integer, end_pos: Integer)` merges to all-optional slots on both axes, matching Ruby `def new(buffer_ = nil, start_pos_ = nil, end_pos_ = nil, buffer: nil, start_pos: nil, end_pos: nil)`. A Ruby parameter that still cannot receive some merged slot — e.g. `(a, b = nil)` against `(Int) | (Int, Int, Int)`, whose merge `(Int, ?Int?, ?Int?)` admits an arity-3 call — keeps raising a positional shape diagnostic.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::MethodParameterMismatch, Ruby::MethodArityMismatch.
