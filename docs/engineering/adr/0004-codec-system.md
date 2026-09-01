# ADR-0004: Two-Level Codec System for Type Conversion

## Status

Accepted

## Context

The V2 Inference Protocol represents all data as named tensors with explicit
datatypes.  This is great for interoperability but inconvenient for Python
model authors who work with NumPy arrays, Pandas DataFrames, string lists,
and other high-level types.

The server needs a type-conversion layer that:

- Encodes high-level Python objects into V2 tensors on the response path.
- Decodes V2 tensors back into high-level Python objects on the request path.
- Supports both per-tensor (e.g., a single NumPy array) and per-request
  (e.g., a DataFrame spanning multiple tensors) conversions.
- Is extensible so that runtime plugins can register custom codecs.

## Decision

Implement a two-level codec system:

### InputCodec (per-tensor)

Operates on individual `RequestInput` / `ResponseOutput` objects.  Each codec
declares a `ContentType` string and a `TypeHint` for auto-detection.
Methods: `encode_input`, `decode_input`, `encode_output`, `decode_output`.

### RequestCodec (per-request)

Operates on entire `InferenceRequest` / `InferenceResponse` objects.  Used
when the encoding spans multiple tensors (e.g., DataFrame columns).
Methods: `encode_request`, `decode_request`, `encode_response`,
`decode_response`.

### CodecRegistry

A singleton `_CodecRegistry` holds all registered codecs, keyed by
`ContentType`.  Codecs are auto-registered via `@register_input_codec` and
`@register_request_codec` decorators at import time.

### Resolution Priority

1. **Content-type** — explicit match from request/model metadata.
2. **Payload type** — `can_encode(payload)` probing.
3. **Type hint** — `TypeHint` class variable comparison.

## Consequences

**Benefits:**
- Model authors can work with familiar Python types without manually
  constructing V2 tensors.
- The `MLModel` base class exposes `decode()`, `decode_request()`,
  `encode()`, and `encode_response()` convenience methods.
- Runtime plugins can register custom codecs for specialised types.

**Trade-offs:**
- Multiple matching codecs trigger a warning and use the first match — there
  is no priority system beyond registration order.
- The two-level split (Input vs. Request) can be confusing for new
  contributors.
- Deprecated `encode()` / `decode()` methods on the codec classes add API
  surface that must be maintained.
