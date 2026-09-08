# Ruby::UnexpectedKeywordArgument

## Overview
A call passes a keyword that the selected method signature does not declare.

## Trigger
Keyword checking finds a supplied keyword outside the accepted required, optional, or rest keywords.

## Example
```ruby
create(name: "Ada", admin: true) # RBS: def create: (name: String) -> User
```

## Typical fix
Remove or rename the keyword, add it to the RBS signature, or add a keyword rest parameter when arbitrary keywords are intended.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::InsufficientKeywordArguments, Ruby::ArgumentTypeMismatch, Ruby::UnresolvedOverloading.
