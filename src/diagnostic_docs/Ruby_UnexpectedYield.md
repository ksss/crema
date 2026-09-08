# Ruby::UnexpectedYield

## Overview
A method body yields even though its RBS declaration has no block parameter.

## Trigger
The checker sees `yield` inside a method whose selected method type does not accept a block.

## Example
```ruby
def around
  yield
end
# RBS: def around: () -> void
```

## Typical fix
Add a block parameter to the RBS method type, remove the yield, or guard the yield behind an API whose contract includes a block.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::RequiredBlockMissing, Ruby::BlockBodyTypeMismatch.
