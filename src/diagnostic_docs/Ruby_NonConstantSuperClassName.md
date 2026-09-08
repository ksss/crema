# Ruby::NonConstantSuperClassName

## Overview
A class declaration uses a dynamic expression as its superclass.

## Trigger
The superclass expression in `class A < expr` is not a static constant path.

## Example
```ruby
class User < base_class
end
```

## Typical fix
Name the superclass as a constant path or write the intended inheritance in RBS when runtime metaprogramming hides it from static analysis.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::NonConstantClassName, Ruby::UnknownConstant, RBS::UnknownTypeName.
