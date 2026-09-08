# Ruby::ClassModuleMismatch

## Overview
A constant is a class in one source and a module in the other.

## Trigger
Inline Ruby declarations and loaded RBS disagree about whether the same constant is a class or module.

## Example
```ruby
module Account; end
# RBS: class Account
```

## Typical fix
Change either the Ruby declaration or the RBS declaration so both describe the same kind of constant.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::DuplicatedDeclarationError, Ruby::NonConstantClassName, Ruby::NonConstantModuleName.
