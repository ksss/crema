# Ruby::AnnotationSyntaxError

## Overview
An inline RBS annotation cannot be parsed.

## Trigger
The text after `#:` or `# @rbs` is sent to the RBS parser and rejected as invalid syntax.

Which annotations are parsed depends on the mode:

- Type assertions (`expr #: T`) and type applications (`foo #: [T]`) are read by the type checker and reported in both modes.
- Declaration annotations (method types such as `def foo #: () -> void`, attributes, constants, class/module aliases, `# @rbs`) are read only in inline mode (`inline = true`). Sig mode reads no declarations from Ruby files, so it does not report syntax errors in them.

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
