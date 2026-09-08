# Ruby::IncompatibleArgumentForwarding

## Overview
A forwarding call such as `target(...)` cannot safely pass the caller's arguments or block to the callee.

## Trigger
The caller method has a forwarded argument shape that is not accepted by the resolved callee signature.

## Example
```ruby
def wrapper(...)
  strict(...)
end
# strict has narrower positional, keyword, or block requirements.
```

## Typical fix
Make the wrapper signature match the callee, forward explicit arguments after validation, or widen the callee signature when it intentionally accepts the forwarded shape.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::MethodArityMismatch, Ruby::ArgumentTypeMismatch, Ruby::UnresolvedOverloading.
