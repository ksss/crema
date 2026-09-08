# Ruby::UnexpectedPositionalArgument

## Overview
A call supplies more positional arguments than the target method accepts.

## Trigger
The selected method signature has no positional slot for one or more supplied arguments.

## Example
```ruby
name("Ada", "Lovelace") # RBS: def name: (String first) -> String
```

## Typical fix
Remove the extra argument, add a rest parameter, or update the RBS signature when the implementation accepts that argument.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::InsufficientPositionalArguments, Ruby::MethodArityMismatch, Ruby::UnresolvedOverloading.
