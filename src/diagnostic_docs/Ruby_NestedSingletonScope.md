# Ruby::NestedSingletonScope

## Description

An inline Ruby `class << self` scope nested inside another `class << self`, or a `def self.foo` inside `class << self`, targets the singleton class of a singleton class. RBS has no declaration shape for that target.

## Recommended severity

Recommended severity: error.

## Example

```ruby
class A
  class << self
    def self.foo; end
  end
end
```

## Fix

Move the method to the representable singleton side:

```ruby
class A
  def self.foo; end
end
```

## Related Diagnostics

Ruby::TopLevelSingletonScope, Ruby::NonSelfSingletonScope.
