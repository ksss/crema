# Crema::DuplicatedConfigKey

## Overview
A `[infusion.config]` YAML file declares the same mapping key more than once; crema keeps the last value (Psych-compatible) and warns.

## Trigger
A config YAML merged by `[infusion.config]` contains a duplicate key at the same nesting level. Ruby's Psych silently resolves this last-wins, so crema does the same instead of aborting, and records a warning so the run still surfaces every other diagnostic.

## Example
```yaml
foo: 1
foo: "a"
```

## Typical fix
Remove the earlier duplicate key, or rename it, so the mapping declares each key once. If the last-wins value is the intended one, no code change is required.

## Recommended severity
Recommended severity: warning.

## Related diagnostics
Ruby::DuplicatedMethodDefinitionError, RBS::InstanceVariableDuplicationError.
