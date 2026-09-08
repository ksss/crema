# Ruby::UnsatisfiableConstraint

## Overview
Method type parameter bounds generated from a call cannot be satisfied.

## Trigger
Argument-derived lower bounds and hint-derived upper bounds for one method type parameter have no valid subtype relation.

## Example
```ruby
xs.with_object([]) { |i, acc| acc << i } #: Array[String]
```

## Typical fix
Make the trailing assertion match the values produced by the call, add conversions inside the block, or choose an overload with compatible type parameters.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::ArgumentTypeMismatch, Ruby::FalseAssertion.
