# Ruby::NoMethod

## Overview
A method call targets a receiver type that has no matching method definition.

## Trigger
Method lookup fails for the receiver after considering its known class, module, interface, or union members.

## Example
```ruby
name.upcase_all # RBS: name is String
```

## Typical fix
Fix the method name, narrow the receiver before calling, add the missing method to the RBS declaration, or change the receiver type to the object that actually owns the method.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnresolvedOverloading, Crema::PrivateMethodCall.
