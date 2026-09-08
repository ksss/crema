# Ruby::ArgumentTypeMismatch

## Overview
A method call passes a value whose inferred type is not accepted by the corresponding RBS parameter.

## Trigger
The receiver method is found, arity is compatible enough to inspect the argument, and a positional or keyword argument is not a subtype of the declared parameter type.

## Example
```ruby
greet(123) # RBS: def greet: (String name) -> void
```

## Typical fix
Pass a value of the declared type, convert before the call, or update the RBS parameter when the implementation intentionally accepts the wider type.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnresolvedOverloading, Ruby::UnexpectedKeywordArgument, Ruby::InsufficientPositionalArguments.
