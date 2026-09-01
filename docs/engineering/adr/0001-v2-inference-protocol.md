# ADR-0001: V2 Inference Protocol as the Wire Format

## Status

Accepted

## Context

MLServer needs a standardised wire format for inference requests and responses
that works across REST and gRPC transports.  The landscape in 2020 offered
several options:

- **Custom API** — maximum flexibility but zero ecosystem interoperability.
- **TensorFlow Serving API** — widely adopted but coupled to TF model types.
- **NVIDIA Triton API / KFServing V2 Protocol** — vendor-neutral, designed
  for multi-framework serving, backed by the KServe community.

The V2 protocol (now called the Open Inference Protocol) provides:

- Framework-agnostic tensor representation with named inputs/outputs.
- Explicit datatype annotations matching NumPy dtypes.
- Model versioning and metadata introspection.
- Liveness, readiness, and model-readiness health checks.
- Model repository management (load / unload / index).
- Both unary and streaming inference RPCs.

## Decision

Adopt the V2 Inference Protocol as MLServer's sole wire format for both
REST (JSON over HTTP) and gRPC (protobuf) transports.

All internal types (`InferenceRequest`, `InferenceResponse`, `RequestInput`,
`ResponseOutput`, etc.) map 1:1 to V2 message schemas.  The `DataPlane` class
implements the protocol's required RPCs; transport servers (REST, gRPC) are
thin adapters that convert between HTTP/protobuf representations and the
shared `DataPlane` interface.

## Consequences

**Benefits:**
- Interoperability with KServe, Triton, TorchServe (V2 mode), and any
  client implementing the protocol.
- Single `DataPlane` implementation shared by REST and gRPC — no duplicated
  business logic.
- Model repository operations (load/unload) are part of the protocol, enabling
  dynamic model management without custom APIs.

**Trade-offs:**
- The V2 tensor format is not the most ergonomic for complex Python types.
  This is mitigated by the codec system (see ADR-0004).
- Streaming inference (`ModelStreamInfer`) is gRPC-native; the REST path uses
  Server-Sent Events as an approximation.
- Extensions beyond the V2 spec (e.g., `runtime_security`) require custom
  endpoints/RPCs that clients must opt into.
