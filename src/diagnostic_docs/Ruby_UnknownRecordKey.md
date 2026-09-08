# Ruby::UnknownRecordKey

## Overview
Record access uses a key that is not present in the known record type.

## Trigger
The receiver is a record type and the requested symbol key is not one of its declared keys.

## Example
```ruby
user[:email] # user: { name: String }
```

## Typical fix
Use a declared key, add the key to the record type, or use a Hash type when open-ended keys are intended.

## Recommended severity
Recommended severity: information.

## Related diagnostics
Ruby::UnknownTupleIndex, Ruby::ArgumentTypeMismatch.
