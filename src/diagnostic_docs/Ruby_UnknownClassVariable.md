# Ruby::UnknownClassVariable

## Overview
A class variable write targets a variable with no declaration on the current class context.

## Trigger
The checker sees a `@@var =` write and cannot find a class-variable declaration in loaded RBS for the current class context.

## Example
```ruby
@@cache = {}
```

## Typical fix
Declare the class variable in RBS, correct the variable name, or replace class variables with class instance variables when that is the intended storage.

## Recommended severity
Recommended severity: information.

## Related diagnostics
Ruby::IncompatibleAssignment, Ruby::UnknownInstanceVariable.
