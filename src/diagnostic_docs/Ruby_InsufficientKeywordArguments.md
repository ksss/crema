# Ruby::InsufficientKeywordArguments

## Overview
A call omits a required keyword argument.

## Trigger
The selected method signature declares a required keyword that is absent from the call.

## Example
```ruby
create(name: "Ada") # RBS: def create: (name: String, email: String) -> User
```

## Typical fix
Pass the missing keyword, make the keyword optional in RBS and Ruby, or remove the requirement from the signature if it is stale.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnexpectedKeywordArgument, Ruby::ArgumentTypeMismatch, Ruby::UnresolvedOverloading.
