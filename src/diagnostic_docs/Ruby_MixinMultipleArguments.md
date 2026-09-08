# Ruby::MixinMultipleArguments

## Overview
An inline mixin call supplies multiple module arguments where crema can only convert one.

## Trigger
`include`, `extend`, or `prepend` appears with more than one argument in inline Ruby declaration collection.

## Example
```ruby
include A, B
```

## Typical fix
Split the mixins into separate calls so each one can be represented independently in RBS.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Crema::MixinTypeArgumentArityMismatch, Ruby::NonConstantModuleName.
