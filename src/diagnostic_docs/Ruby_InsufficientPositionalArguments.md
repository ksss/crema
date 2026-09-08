# Ruby::InsufficientPositionalArguments

## Overview
A call omits one or more required positional arguments.

## Trigger
The selected method signature requires more positional arguments than the call supplies.

## Example
```ruby
move(10) # RBS: def move: (Integer x, Integer y) -> void
```

## Typical fix
Add the missing arguments, make the RBS parameters optional, or provide defaults in Ruby and RBS consistently.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnexpectedPositionalArgument, Ruby::MethodArityMismatch, Ruby::UnresolvedOverloading.
