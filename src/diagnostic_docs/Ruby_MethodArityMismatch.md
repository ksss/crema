# Ruby::MethodArityMismatch

## Overview
A Ruby method definition and its RBS declaration have different parameter arity.

## Trigger
Method body validation compares Ruby parameters with the RBS method type and finds too few or too many positional or keyword slots.

## Example
```ruby
def resize(width); end
# RBS: def resize: (Integer width, Integer height) -> void
```

## Typical fix
Align the Ruby parameter list and the RBS method type, including optional, rest, keyword, and block parameters.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::MethodParameterMismatch, Ruby::InsufficientPositionalArguments, Ruby::UnexpectedPositionalArgument.
