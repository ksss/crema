# RBS::RecursiveAncestorError

## Overview
A class or module ancestor graph contains a cycle.

## Trigger
Superclass, include, prepend, or self-type expansion reaches a type already on the active ancestor path.

## Example
```rbs
module A
  include B
end
module B
  include A
end
```

## Typical fix
Remove one ancestor edge, extract shared behavior into a third module, or correct the mistaken include/prepend target.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::RecursiveAliasDefinitionError, Crema::MixinTypeArgumentArityMismatch.
