# Crema::InfusionProviderSkipped

## Overview
An infusion provider skipped generating type information for a subject and recorded the reason.

## Trigger
A framework-specific infusion rule recognizes a target but declines to synthesize RBS because required metadata is missing or unsupported.

## Example
```ruby
class User < ApplicationRecord
  scope dynamic_name, -> { all }
end
```

## Typical fix
Inspect the provider reason, simplify the DSL usage, add explicit RBS for the skipped subject, or leave it as information when the provider intentionally does not cover the pattern.

## Recommended severity
Recommended severity: information.

## Related diagnostics
Ruby::FallbackAny.
