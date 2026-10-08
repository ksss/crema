# Ruby::UnusedInlineAnnotation

## Overview
An inline RBS annotation is present but not attached to the Ruby construct it was meant to describe.

## Trigger
The parser sees an annotation but the following or preceding Ruby node cannot consume it.

Reported only in inline mode (`inline = true`), where annotations declare the surrounding Ruby code. Sig mode reads no declarations from Ruby files, so a declaration annotation there is neither read nor reported.

## Example
```ruby
#: Integer
puts value
```

## Typical fix
Move the annotation next to a supported construct, correct parameter names in method annotations, or delete stale annotations.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::AnnotationSyntaxError, Ruby::FalseAssertion.
