# Crema::PrivateMethodCall

## Overview
A private method is called with an explicit receiver.

## Trigger
Method lookup finds the target method but marks it private, and the call uses an explicit non-self receiver.

## Example
```ruby
obj.secret
```

## Typical fix
Call the private method without an explicit receiver from inside the owning type, expose a public wrapper, or change the method visibility if external calls are intended.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::NoMethod.
