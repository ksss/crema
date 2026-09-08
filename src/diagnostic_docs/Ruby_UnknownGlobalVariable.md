# Ruby::UnknownGlobalVariable

## Overview
A global variable write targets a variable with no declaration in loaded RBS signatures.

## Trigger
The checker sees a `$global =` write and the environment has no matching global variable declaration.

## Example
```ruby
$feature_flag = true
```

## Typical fix
Declare the global variable in RBS, avoid global state, or route the value through a typed object.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::IncompatibleAssignment.
