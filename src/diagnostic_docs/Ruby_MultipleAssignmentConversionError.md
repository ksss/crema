# Ruby::MultipleAssignmentConversionError

## Overview
A multiple assignment (`a, b = rhs`) drove implicit conversion via `#to_ary`, but the method returned a value that is neither a Tuple nor `Array[T]`. Ruby itself rejects this at runtime with `TypeError` from `rb_check_array_type`, so the pattern is a real bug rather than a benign scalar fallback.

## Trigger
`try_convert(rhs, :to_ary)` succeeds (the receiver has a public arity-zero `to_ary`), but the substituted return type cannot be destructured into positional slots.

## Example
```ruby
class BadAry
  def to_ary: () -> Integer
end

a, b = BadAry.new  # to_ary returns Integer, not a Tuple / Array[T]
```

## Typical fix
Change `to_ary` to return a Tuple or `Array[T]`, or drop the `to_ary` declaration entirely so the assignment falls back to the scalar path.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::NoMethod, Ruby::IncompatibleAssignment.
