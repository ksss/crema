# Ruby::AnnotationSyntaxError

## Overview
An inline RBS annotation cannot be parsed.

## Trigger
The text after `#:` or `# @rbs` is sent to the RBS parser and rejected as invalid syntax.

## Example
```ruby
x = value #: Array[
```

## Typical fix
Fix the inline RBS syntax, reduce the annotation to a known-good type, or move complex declarations into a `.rbs` file.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnusedInlineAnnotation, Ruby::FalseAssertion.
