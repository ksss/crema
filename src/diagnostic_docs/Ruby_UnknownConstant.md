# Ruby::UnknownConstant

## Overview
A constant reference or declaration path cannot be resolved in the current scope.

## Trigger
Lexical lookup and ancestor lookup cannot find the named constant in loaded Ruby inline declarations or RBS signatures.

## Example
```ruby
AdminUser.new
```

## Typical fix
Load the missing RBS, correct the namespace or spelling, add an inline declaration, or require the library whose signature defines the constant.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::UnknownTypeName, Ruby::NonConstantClassName.
