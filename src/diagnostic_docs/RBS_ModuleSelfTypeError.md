# RBS::ModuleSelfTypeError

## Overview
A class or module includes / extends a module whose self type constraint it does not satisfy.

## Trigger
`module M : _K` constrains every type that mixes `M` in: an `include M` requires the includer's instance type to be a subtype of `_K`, an `extend M` requires its singleton type to be. A module without an explicit self type is constrained to `Object`. The check runs once per `include` / `extend` member of a declared class or module and reports each violating mixin at that member.

## Example
```rbs
interface _Each[A]
  def each: () { (A) -> void } -> void
end

module Collection[T] : _Each[T]
  def to_a: () -> Array[T]
end

class Ints
  include Collection[Integer]   # error: ::Ints <: ::_Each[::Integer]
  def each: () { (String) -> void } -> void
end
```

The JSONL diagnostic carries the relation: `type_name` is the includer, `ancestor` the mixin with its arguments, `self_type` the constraint after substituting the module's parameters (`::_Each[::Integer]`), and `actual` the sub type that was compared (`::Ints`, `singleton(::Ints)` for `extend`, an intersection with the includer's own self types for a module includer).

## Typical fix
Declare the methods the self type requires on the includer (or on its singleton for `extend`), fix their signatures, or add the constraint to the includer's own self type when it is a module (`module Q : _K`).

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::RecursiveAncestorError, Crema::TypeArgumentBoundViolation.
