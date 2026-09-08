# RBS::RecursiveTypeAliasError

## Overview
One or more type aliases form a cycle through transparent type constructors.

## Trigger
Alias dependency analysis finds a cycle such as `a = b` and `b = a`.

## Example
```rbs
type a = b
type b = a
```

## Typical fix
Introduce an opaque container, remove one alias, or replace the alias chain with a concrete recursive class/interface model.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::RecursiveAliasDefinitionError, RBS::UnknownTypeName.
