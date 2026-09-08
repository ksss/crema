# Crema::MixinTypeArgumentArityMismatch

## Overview
A superclass or mixin reference supplies the wrong number of generic type arguments.

## Trigger
The build-layer validator checks a superclass, include, extend, or prepend reference against the target type parameters and the supplied argument count is outside the accepted range.

## Example
```rbs
class Bag
  include Enumerable
end
```

## Typical fix
Add the missing type arguments, remove extra type arguments, or update the target declaration's type parameters if the generic contract changed.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Crema::TypeArgumentBoundViolation, RBS::UnknownTypeName.
