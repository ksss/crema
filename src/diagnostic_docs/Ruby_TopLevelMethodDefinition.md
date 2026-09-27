# Ruby::TopLevelMethodDefinition

## Description

A top-level inline Ruby `def self.name` defines a singleton method on the `main` object, which has no named class or module declaration target. RBS cannot represent that method definition as a member.

A top-level `def name` (no receiver) is not reported: crema collects it as a private instance method of `::Object`, matching Ruby's semantics.

## Recommended severity

Recommended severity: error.

## Example

```ruby
# @rbs () -> void
def self.foo; end
```

## Fix

Move the method under a class or module that RBS can name, or drop the `self.` receiver so it becomes an `Object` instance method.

## Related Diagnostics

Ruby::UnusedInlineAnnotation, Ruby::TopLevelSingletonScope.
