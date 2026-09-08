# Ruby::UnknownTupleIndex

## Overview
Tuple indexing uses an integer index outside the known tuple length.

## Trigger
The receiver is a tuple type and the literal index is negative or greater than the last tuple slot.

## Example
```ruby
pair[2] # pair: [String, Integer]
```

## Typical fix
Use an in-range index, change the tuple type to include the slot, or model the value as an Array when the length is not fixed.

## Recommended severity
Recommended severity: error.

## Related diagnostics
Ruby::UnknownRecordKey, Ruby::ArgumentTypeMismatch.
