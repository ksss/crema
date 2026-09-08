# Ruby::TopLevelMethodDefinition

## Description

A top-level inline Ruby `def` has no named class or module declaration target. RBS cannot represent that method definition as a member.

## Recommended severity

Recommended severity: error.

## Example

```ruby
# @rbs () -> void
def self.foo; end
```

## Fix

Move the method under a class or module that RBS can name.

## Related Diagnostics

Ruby::UnusedInlineAnnotation, Ruby::TopLevelSingletonScope.
