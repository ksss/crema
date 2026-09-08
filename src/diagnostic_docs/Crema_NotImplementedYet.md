# Crema::NotImplementedYet

## Overview
An unsupported internal type-checking path was reached and emitted its instrumentation marker.

## Trigger
Development-only instrumentation is explicitly enabled through `[diagnostic]` and a fallback arm records the unsupported site and subject.

## Example
```toml
[diagnostic]
"Crema::NotImplementedYet" = "information"
```

## Typical fix
For crema development, implement or split a todo for the reported site. For normal project checking, keep this code ignored.

## Recommended severity
Recommended severity: ignore.

## Related diagnostics
Ruby::FallbackAny.
