# Ruby::RequiredBlockMissing

## Overview
A method call omits a block required by the selected RBS signature.

## Trigger
The method type requires a block and the call site does not pass one.

## Example
```ruby
with_lock() # RBS: def with_lock: () { () -> void } -> void
```

## Typical fix
Pass the required block, make the block optional in RBS if nil is accepted, or call a different API that does not require a block.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::BlockBodyTypeMismatch, Ruby::UnexpectedYield.
