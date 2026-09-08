# Ruby::DuplicatedMethodDefinitionError

## Overview
Inline RBS declares the same non-overload method more than once in one type.

## Trigger
RBS method collection sees duplicate method definitions that are not marked as overload entries.

## Example
```ruby
# @rbs def name: () -> String
# @rbs def name: () -> String
```

## Typical fix
Remove the duplicate declaration or express intentional alternatives as overloads.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::DuplicatedDeclarationError, Ruby::MethodArityMismatch.
