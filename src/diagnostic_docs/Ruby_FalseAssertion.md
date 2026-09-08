# Ruby::FalseAssertion

## Overview
An inline assertion names a type incompatible with the expression's inferred type.

## Trigger
Neither the inferred expression type nor the asserted type is a subtype of the other.

## Example
```ruby
value = 1 #: String
```

## Typical fix
Correct the asserted type, convert the expression before asserting, or remove an assertion that was only documenting an old expectation.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::AnnotationSyntaxError, Ruby::MethodBodyTypeMismatch, Ruby::IncompatibleAssignment.
