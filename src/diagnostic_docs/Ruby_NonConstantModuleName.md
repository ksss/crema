# Ruby::NonConstantModuleName

## Overview
A module declaration name contains a dynamic expression instead of a constant path.

## Trigger
The inline parser cannot convert the Ruby module name node into a static RBS type name.

Reported only in inline mode (`inline = true`). Sig mode reads no declarations from Ruby files, so it does not report this.

## Example
```ruby
module namespace.call::Helpers
end
```

## Typical fix
Use a static constant path for the module declaration, or declare the dynamic module in RBS manually if crema cannot infer it.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::NonConstantClassName, Ruby::UnknownConstant.
