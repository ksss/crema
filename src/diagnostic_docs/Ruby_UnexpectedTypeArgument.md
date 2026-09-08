# Ruby::UnexpectedTypeArgument

## Overview
A call-site type application supplies more type arguments than the selected method type accepts.

## Example
```ruby
c.foo("x") #[Integer, String]
```

```rbs
class C
  def foo: [T] (T) -> T
end
```

## Typical fix
Remove the extra type argument or update the method type parameters.

## Recommended severity
Recommended severity: hint.
