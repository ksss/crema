# Ruby::UnreachableValueBranch

## Overview
A `case` branch that yields a value cannot run: the `else` branch after `when` clauses that cover every scrutinee value, or a `when` clause that matches no value left in the scrutinee.

## Trigger
Case narrowing proves the scrutinee residue is empty before the `else` branch, or narrows the scrutinee to nothing for every test of a `when` clause. A branch whose value is `bot` (such as `raise`) is not reported.

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
Crema::NonExhaustiveCase, Ruby::UnreachableBranch.
