# Ruby::InsufficientTypeArguments

## Overview
A call-site type application supplies fewer type arguments than the selected method type requires.

## Example
```ruby
c.pair(1, "x") #[Integer]
```

```rbs
class C
  def pair: [T, U] (T, U) -> [T, U]
end
```

## Typical fix
Supply every method type argument or remove the explicit type application.

## Recommended severity
Recommended severity: hint.
