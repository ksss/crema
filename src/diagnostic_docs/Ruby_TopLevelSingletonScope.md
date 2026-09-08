# Ruby::TopLevelSingletonScope

## Description

A top-level inline Ruby `class << self` opens the singleton class of `main`. RBS has no declaration shape for methods or members on that object.

## Recommended severity

Recommended severity: error.

## Example

```ruby
class << self
  def foo; end
end
```

## Fix

Move the declaration under a class or module that RBS can name.

## Related Diagnostics

Ruby::NestedSingletonScope, Ruby::NonSelfSingletonScope.
