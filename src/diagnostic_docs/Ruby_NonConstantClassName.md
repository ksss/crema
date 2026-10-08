# Ruby::NonConstantClassName

## Overview
A class declaration name contains a dynamic expression instead of a constant path.

## Trigger
The inline parser cannot convert the Ruby class name node into a static RBS type name.

Reported only in inline mode (`inline = true`). Sig mode reads no declarations from Ruby files, so it does not report this.

## Example
```ruby
class factory.call::User
end
```

## Typical fix
Use a literal constant path for the class declaration and move dynamic class creation behind a separately declared constant.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::NonConstantModuleName, Ruby::NonConstantSuperClassName, Ruby::UnknownConstant.
