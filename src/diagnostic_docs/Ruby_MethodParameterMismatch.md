# Ruby::MethodParameterMismatch

## Overview
A Ruby parameter kind differs from the corresponding RBS parameter kind.

## Trigger
The method name matches but a parameter is positional, keyword, rest, or block in Ruby while RBS declares a different kind.

## Example
```ruby
def save(options); end
# RBS: def save: (**String options) -> void
```

## Typical fix
Make the Ruby parameter shape and RBS method type use the same parameter kind and name.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::MethodArityMismatch, Ruby::UnexpectedKeywordArgument.
