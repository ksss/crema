# RBS::RecursiveAliasDefinitionError

## Overview
A method alias chain loops back to one of its own aliases.

## Trigger
Method alias graph sorting detects a strongly connected component.

## Example
```rbs
class User
  alias name display_name
  alias display_name name
end
```

## Typical fix
Break the alias cycle by pointing at a real method implementation or removing one alias.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::RecursiveTypeAliasError, RBS::RecursiveAncestorError.
