# Ruby::UnknownInstanceVariable

## Overview
An instance variable write targets a variable with no declaration on the current self type.

## Trigger
The checker sees an `@ivar =` write and cannot find an instance-variable declaration on the receiver's ancestor chain.

## Example
```ruby
@token = token
```

## Typical fix
Declare the instance variable in RBS, fix a misspelled variable name, or write through an accessor with an existing method contract.

## Recommended severity
Recommended severity: information.

## Related diagnostics
Ruby::IncompatibleAssignment, RBS::InstanceVariableDuplicationError.
