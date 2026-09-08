# Ruby::SyntaxError

## Overview
Ruby source parsing failed before inline annotation collection or type checking could run.

## Trigger
Prism reports a syntax error for a checked Ruby file.

## Example
```ruby
def broken(
```

## Typical fix
Fix the Ruby syntax first, then rerun crema so inline declarations and type checking can proceed.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::AnnotationSyntaxError.
