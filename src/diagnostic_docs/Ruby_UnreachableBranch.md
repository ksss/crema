# Ruby::UnreachableBranch

## Overview
An `if`, `unless`, `elsif` or ternary branch cannot run because its condition narrows a local variable or a pure method call to no possible value.

## Trigger
A type test such as `is_a?`, `nil?` or `===` leaves no value of the tested variable on the branch's side. Inside that branch the variable reads as `untyped`, and the branch does not contribute to the types after the conditional.

## Example
```ruby
def run(n) # n: Integer
  if n.is_a?(String)
    n.upcase
  end
end
```

## Typical fix
Remove the dead branch, or widen the variable's declared type if the branch is expected to run.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::UnreachableValueBranch.
