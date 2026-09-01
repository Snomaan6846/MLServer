# ADR-0003: Adaptive Batching for Inference Requests

## Status

Accepted

## Context

Many ML frameworks achieve significantly higher throughput when processing
batched inputs (vectorised operations on arrays/tensors) compared to
processing individual requests sequentially.  However, clients typically send
one request at a time.

The server needs to transparently batch incoming requests without requiring
client-side changes, while keeping latency bounded.

Options considered:

1. **Client-side batching** — requires all clients to implement batching
   logic; impractical for heterogeneous client ecosystems.
2. **Fixed-interval batching** — collects requests for a fixed time window.
   Simple but adds unnecessary latency when load is high (batch fills before
   the timer expires).
3. **Adaptive batching** — collects up to N requests OR waits up to T
   seconds, whichever comes first.  Balances throughput and latency.

## Decision

Implement adaptive batching via the `AdaptiveBatcher` class, configured
per-model through `max_batch_size` and `max_batch_time` in `ModelSettings`:

- Each incoming request is placed in an `asyncio.Queue` and the caller
  receives a `Future` for its individual response.
- A background `_batcher()` coroutine collects requests until the batch is
  full or the time window expires.
- Collected requests are merged into a single `InferenceRequest` via
  `BatchedRequests.merge()`.
- The merged request is passed to the model's original `predict()`.
- The merged response is split back into individual responses via
  `BatchedRequests.split_response()`.

Batching is injected as a model load hook (`load_batching`) that wraps the
model's `predict` method — the model itself is unaware of batching.

## Consequences

**Benefits:**
- Transparent to both clients and model implementations.
- Adapts to load: under high load, batches fill quickly (low latency);
  under low load, the timer prevents indefinite waiting.
- Multiple batches can be in-flight concurrently.

**Trade-offs:**
- Adds `max_batch_time` latency in the worst case (single request, low load).
- Merge/split logic assumes homogeneous tensor shapes across requests in a
  batch — models with variable-shape inputs may not benefit.
- Streaming inference (`predict_stream`) is not batched.
