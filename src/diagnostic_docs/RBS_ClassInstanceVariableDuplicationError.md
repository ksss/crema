# RBS::ClassInstanceVariableDuplicationError

## Overview
A type declares the same class instance variable more than once.

## Trigger
RBS variable collection sees duplicate singleton-side `@ivar` declarations for one type.

## Example
```rbs
class User
  self.@registry: Hash[Symbol, User]
  self.@registry: Array[User]
end
```

## Typical fix
Keep one declaration, merge the valid alternatives into one type, or rename separate variables.

## Recommended severity
Recommended severity: error.

## Related diagnostics
RBS::InstanceVariableDuplicationError, Ruby::IncompatibleAssignment.
