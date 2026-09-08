# Ruby::UnreachableValueBranch

## Overview
A `case` expression's `else` branch cannot run because previous `when` clauses cover every scrutinee value.

## Trigger
Case narrowing proves the scrutinee residue is empty before the `else` branch.

## Example
```ruby
case value # value: Integer | String
when Integer
when String
else
  unreachable
end
```

## Typical fix
Remove the unreachable `else`, widen the scrutinee type if the else is expected, or revise the `when` branches.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Crema::NonExhaustiveCase.
