# RBS::InstanceVariableDuplicationError

## Overview
A type declares the same instance variable more than once.

## Trigger
RBS variable collection sees duplicate `@ivar` declarations for one type.

## Example
```rbs
class User
  @name: String
  @name: Integer
end
```

## Typical fix
Keep one declaration, merge the intended type into a union if both values are valid, or move separate state to different variable names.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::ClassInstanceVariableDuplicationError, Ruby::IncompatibleAssignment.
