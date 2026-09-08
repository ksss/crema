# Ruby::ClassModuleAliasDeclarationMissingTypeName

## Overview
A class-alias or module-alias inline annotation cannot infer the aliased constant name.

## Trigger
The inline alias annotation omits an explicit type name and the right-hand side is not a constant path.

## Example
```ruby
Foo = factory.call #: class-alias
```

## Typical fix
Write the target name explicitly in the annotation or assign from a constant path that can be inferred.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::AnnotationSyntaxError, Ruby::NonConstantClassName, Ruby::NonConstantModuleName.
