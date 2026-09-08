# Ruby::IncompatibleAssignment

## Overview
A variable assignment writes a value incompatible with the declared variable type.

## Trigger
An instance, class, or global variable has an RBS declaration and the assigned RHS is not a subtype of that declaration.

## Example
```ruby
@count = "one" # RBS: @count: Integer
```

## Typical fix
Assign the declared type, convert before assignment, or update the variable declaration to the real value type.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::UnknownInstanceVariable, Ruby::UnknownClassVariable, Ruby::UnknownGlobalVariable.
