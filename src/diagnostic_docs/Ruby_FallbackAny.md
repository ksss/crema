# Ruby::FallbackAny

## Overview
An expression could not be inferred precisely and fell back to untyped.

## Trigger
Type construction reaches a supported but insufficiently typed path and records why it returned untyped.

## Example
```ruby
value = dynamic_send(name)
```

## Typical fix
Add a local inline assertion, improve the receiver or method RBS, or replace highly dynamic code with a statically visible call where practical.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Crema::NotImplementedYet.
