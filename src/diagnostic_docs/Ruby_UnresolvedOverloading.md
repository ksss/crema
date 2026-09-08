# Ruby::UnresolvedOverloading

## Overview
A call targets an overloaded method, but no overload accepts the supplied arguments.

## Trigger
crema tries every overload and rejects each candidate by arity, keyword, block, or argument type.

## Example
```ruby
parse(:id) # RBS overloads accept String or Integer, not Symbol
```

## Typical fix
Pass arguments matching one overload, add an overload for the intended shape, or narrow values before the call so overload selection can succeed.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::ArgumentTypeMismatch, Ruby::MethodArityMismatch, Ruby::NoMethod.
