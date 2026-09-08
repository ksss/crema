# Ruby::UnexpectedSuper

## Overview
A `super` call cannot resolve a matching superclass method.

## Trigger
The method checker looks above the defining class in the ancestor chain and finds no compatible method for `super`.

## Example
```ruby
def save
  super
end
```

## Typical fix
Declare the superclass method in RBS, remove the `super` call, or check that the class inherits from the intended parent.

## Recommended severity
Recommended severity: information.

## Related diagnostics
Ruby::NoMethod, Ruby::MethodArityMismatch.
