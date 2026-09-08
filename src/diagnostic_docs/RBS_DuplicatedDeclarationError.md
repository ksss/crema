# RBS::DuplicatedDeclarationError

## Overview
Inline Ruby declarations define the same RBS name in conflicting declaration kinds.

## Trigger
The environment builder finds two declarations for the same name that cannot be merged.

## Example
```ruby
class Account; end
module Account; end
```

## Typical fix
Keep one declaration kind for the name, rename one constant, or split compatible partial declarations without changing class/module kind.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::ClassModuleMismatch, Ruby::DuplicatedMethodDefinitionError.
