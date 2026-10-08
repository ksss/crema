# Ruby::SingletonScopeConstantDefinition

## Description

An inline Ruby constant assignment (`S = 1`) inside `class << self` defines the constant on the singleton class of the enclosing class (`Foo.singleton_class::S`), not on the class itself (`Foo::S`). RBS has no declaration shape for a singleton class's constants, so the assignment is dropped from the inline declarations. The check side reports the same assignment as `Ruby::UnknownConstant`, because no RBS declaration can ever cover it. Reads of `S` inside `class << self` are unaffected: Ruby's lookup continues to the enclosing class, so a `Foo::S` declared in RBS resolves as usual.

Reported only in inline mode (`inline = true`). Sig mode reads no declarations from Ruby files, so it does not report this; the check side's `Ruby::UnknownConstant` is reported in both modes.

## Recommended severity

Recommended severity: error.

## Example

```ruby
class Foo
  class << self
    S = 1
  end
end
```

## Fix

Move the assignment to the class body, where RBS can declare it as `Foo::S`:

```ruby
class Foo
  S = 1
  class << self
  end
end
```

A path write (`Foo::S = 1`) inside `class << self` already targets `Foo::S` and is not reported.

## Related Diagnostics

Ruby::NestedSingletonScope, Ruby::TopLevelSingletonScope, Ruby::NonSelfSingletonScope, Ruby::UnknownConstant.
