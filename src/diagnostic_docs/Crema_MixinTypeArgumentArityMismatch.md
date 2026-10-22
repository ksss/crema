# Crema::MixinTypeArgumentArityMismatch

## Overview
A superclass or mixin reference supplies the wrong number of generic type arguments.

## Trigger
The build-layer validator checks a superclass, include, extend, or prepend reference against the target type parameters and the supplied argument count is outside the accepted range.

## Example
```rbs
class Bag
  include Enumerable
end
```

## Typical fix
Add the missing type arguments, remove extra type arguments, or update the target declaration's type parameters if the generic contract changed.

Type arguments on a mixin or superclass are spelled differently in the two file kinds:

```rbs
class Bag
  include Enumerable[String]
end
class Names < Array[String]
end
```

```ruby
class Bag
  include Enumerable #[String]
end
class Names < Array #[String]
end
```

When too few arguments are supplied, the diagnostic carries a `help` key saying how many type arguments to add and where, in the syntax of the reference's own file, for example ``add a type argument after this reference, e.g. `#[untyped]` `` (inline) or `` `[untyped]` `` (`.rbs`). The reference is the diagnostic's byte range. `untyped` is only an example that makes the reference well-formed: write the real element type when you know it.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Crema::TypeArgumentBoundViolation, RBS::UnknownTypeName.
