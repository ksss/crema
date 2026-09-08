# RBS::UnknownTypeName

## Overview
A referenced RBS type name cannot be resolved in the loaded environment.

## Trigger
The build layer resolves a type name from an alias, superclass, mixin, or other RBS reference and finds no declaration.

## Example
```rbs
class User < ApplicationRecord
end
```

## Typical fix
Load the missing signature, correct the type name or namespace, or add a declaration for the referenced type.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnknownConstant, Crema::MixinTypeArgumentArityMismatch.
