# Ruby::NonSelfSingletonScope

## Description

An inline Ruby `class << expr` where `expr` is not `self` opens an anonymous singleton class. RBS has no declaration shape for that target.

Reported only in inline mode (`inline = true`). Sig mode reads no declarations from Ruby files, so it does not report this.

## Recommended severity

Recommended severity: error.

## Example

```ruby
class A
  obj = Object.new
  class << obj
    def foo; end
  end
end
```

## Fix

Move the method to a named class or module, or use a regular `class << self` scope inside the named class when the method belongs to that class object.

## Related Diagnostics

Ruby::NestedSingletonScope, Ruby::TopLevelSingletonScope.
