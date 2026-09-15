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

A single declaration can also close the cycle on itself. Superclass and mixin names resolve in the enclosing namespace, so a nested class named after its intended parent picks itself:

```rbs
class UserMention
end
module Reports
  class UserMention < UserMention   # resolves to ::Reports::UserMention
  end
end
```

## Typical fix
Remove one ancestor edge, extract shared behavior into a third module, or correct the mistaken include/prepend target. For a self-referencing superclass, qualify the parent (`< ::UserMention`).

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::RecursiveAliasDefinitionError, Crema::MixinTypeArgumentArityMismatch.
