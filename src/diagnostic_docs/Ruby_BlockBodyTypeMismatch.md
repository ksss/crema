# Ruby::BlockBodyTypeMismatch

## Overview
A passed block returns a value incompatible with the block return type expected by the method.

## Trigger
The method signature accepts a block and the inferred block body type is not a subtype of the declared block return.

## Example
```ruby
each_name { 123 } # RBS block: { (String) -> String }
```

## Typical fix
Return the expected type from the block, update the block type in RBS, or choose an overload whose block contract matches the call.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::RequiredBlockMissing, Ruby::ArgumentTypeMismatch, Ruby::MethodBodyTypeMismatch.
