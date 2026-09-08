# Crema::TypeArgumentBoundViolation

## Overview
A generic type argument violates a declared upper or lower bound.

## Trigger
Type argument application checks a generic parameter bound and the actual argument is outside the accepted subtype relation.

## Example
```rbs
class Box[T < Numeric]
end

type bad = Box[String]
```

## Typical fix
Use a type argument within the bound, loosen the bound, or introduce a new generic parameter when the wider type is valid.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Crema::MixinTypeArgumentArityMismatch, RBS::UnknownTypeName.
