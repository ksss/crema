# Ruby::IncompatibleAssignment

## Overview
A variable assignment writes a value incompatible with the type the variable is held to.

## Trigger
Two shapes:

- An instance, class, or global variable has an RBS declaration and the assigned RHS is not a subtype of that declaration.
- A local variable is assigned from inside a block, lambda, or proc, and the RHS is not a subtype of the type the variable had when the closure was entered. Entering a closure pins every visible local variable to its current type, because the closure body may run zero times or later than the surrounding code. The variable keeps the pinned type inside and after the closure; a plain sequential reassignment outside any closure is not pinned and never triggers this.

## Example
```ruby
@count = "one" # RBS: @count: Integer

r = 1
[1].each { |m| r = "s" } # ::String is not a subtype of ::Integer (pinned on block entry)
r + 1                    # r is still ::Integer here
```

## Typical fix
For a declared variable: assign the declared type, convert before assignment, or update the declaration to the real value type.

For a local variable written from a closure: declare the wider type up front with an inline annotation (`r = nil #: Integer?`), initialize it with a value of the type the closure writes, or restructure so the closure returns the value instead of assigning it (`r = arr.map { |m| ... }`).

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::UnknownInstanceVariable, Ruby::UnknownClassVariable, Ruby::UnknownGlobalVariable, Ruby::FalseAssertion.
