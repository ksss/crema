# Ruby::MethodBodyTypeMismatch

## Overview
A method body returns a value incompatible with its declared RBS return type.

## Trigger
The inferred body result is not a subtype of the method return type.

## Example
```ruby
def count
  "one"
end
# RBS: def count: () -> Integer
```

## Typical fix
Return the declared type, narrow or convert the body value, or correct the RBS return type when the implementation contract is different.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::BlockBodyTypeMismatch, Ruby::FalseAssertion, Ruby::IncompatibleAssignment.
