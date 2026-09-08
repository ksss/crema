# Ruby::DeprecatedReference

## Overview
A reference targets a method, constant, or global variable whose RBS declaration is marked with `%a{deprecated}`.

## Trigger
The checker resolves a call, constant read, or global variable read/write to a declaration whose annotations contain `deprecated` (optionally `deprecated: <message>`). For methods, both member-level (`%a{deprecated}` above the whole `def`) and per-overload annotations (on the specific overload arm the call resolved to) are considered.

## Example
```ruby
class Foo
  %a{deprecated: use `bar` instead}
  def foo: () -> Integer
end
Foo.new.foo  # Ruby::DeprecatedReference: The method `foo` is deprecated: use `bar` instead
```

## Typical fix
Migrate to the replacement API named in the deprecation message, or update the RBS declaration to remove `%a{deprecated}` if the deprecation is no longer intended.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::UnknownConstant, Ruby::UnknownGlobalVariable.
