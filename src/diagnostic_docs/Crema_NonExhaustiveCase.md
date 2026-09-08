# Crema::NonExhaustiveCase

## Overview
A `case` expression without `else` leaves part of the scrutinee type uncovered.

## Trigger
Every `when` branch can be interpreted as a class or module narrowing target, and subtracting those branches leaves a non-empty residue.

## Example
```ruby
case value # value: Integer | String | Symbol
when Integer
when String
end
```

## Typical fix
Add a `when` for the missing type, add an `else`, or narrow the scrutinee type before the case.

## Recommended severity
Recommended severity: hint.

## Related diagnostics
Ruby::UnreachableValueBranch.
