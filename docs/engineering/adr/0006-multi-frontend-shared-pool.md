# ADR-0006: Multi-Frontend Shared-Pool Serving and Worker-Side Batching

## Status

Accepted

## Context

Two scaling limits share one root cause: the MLServer frontend (REST and
gRPC parsing, validation, response serialisation) and the adaptive
batcher both run in the main process.

1. **Frontend saturation.** A single Python process cannot use more than
   one core for request parsing and response serialisation. At high
   request rates the frontend, not the model, becomes the bottleneck.
2. **Batcher fragmentation.** The adaptive batcher wrapped the
   `ParallelModel` in the main process. Scaling the frontend across
   processes would give each frontend batcher only a fraction of the
   traffic, shrinking batches and stalling windows (the same failure
   mode that historically motivated keeping batching in the main
   process when workers were introduced, and that recurred whenever
   batching moved away from the full traffic stream).

Options considered:

1. **N independent MLServer replicas behind a load balancer.** Rejected
   as the in-process solution: every replica loads its own copy of
   every model (N x memory, critical for GPU), and the model repository
   API gains no single-facade semantics.
2. **A central broker process between frontends and workers.** Rejected:
   every inference payload would transit one extra process (one extra
   serialisation hop and one extra event loop in the hot path),
   recreating the single-process bottleneck the design removes.
3. **N frontends sharing one inference pool**, with the supervisor
   process owning the pool and coordinating model lifecycle over
   control pipes. Keeps one copy of every model, keeps the data plane
   peer-to-peer (ZeroMQ), and moves the batcher next to the model in
   the workers so batch formation always sees the full traffic rate.

## Decision

1. Add a `frontend_workers` setting (default `0`). At `N >= 1` a
   `FrontendSupervisor` process spawns `N` frontend processes serving
   gRPC on the shared port via `SO_REUSEPORT` and REST through a
   single listening socket the supervisor binds once and shares by
   file descriptor with every frontend (one accept queue; each
   frontend publishes its open-connection count into a shared array
   and accepts cooperatively, least-loaded first — see the amendment
   below), owns the single
   inference pool, and coordinates model loads/unloads: operations
   arriving at any frontend are applied on the pool and broadcast so
   every frontend serves the model; the origin's call returns only
   after every live frontend has acknowledged.
2. Require `parallel_inference_transport = "zmq"` and
   `parallel_workers >= 1` at `N >= 1` (startup validation): the queue
   transport is parent-child, and without a pool each frontend would
   load its own model copies.
3. Move the adaptive batcher to the model's point of execution: the
   `load_batching` hook is installed in the worker's model registry
   when a pool exists (unchanged in the main process at
   `parallel_workers = 0`). Requests for a batched model use
   model-affine routing (all requests to one worker per rotation
   window, `parallel_routing_rotation`) so the batcher sees the full
   rate; unbatched models keep round-robin execution spread.
4. Wire cancellation end-to-end: a client disconnect during a pending
   dispatch evicts the queued request from its worker-side batcher
   (fire-and-forget; a lost eviction merely wastes one execution).
5. Children arm `PR_SET_PDEATHSIG` (Linux) so a killed supervisor
   leaves no orphans; shutdown drains frontends first, then stops
   workers through their control queues. Crashed frontends restart with
   exponential backoff (1s doubling, capped at 16s) up to
   `frontend_max_restarts` (default 5) per frontend.

## Consequences

- Frontend-bound throughput scales with `frontend_workers` (measured
  ~1.9x at N=2 with a single worker owning the remaining bottleneck).
- Batched single-request latency is unchanged at any frontend count,
  and relocating the batcher measurably improves pooled batching
  throughput on its own.
- Model operations gain cross-process acknowledgement semantics;
  frontend crashes auto-ack their pending slots so operations resolve
  immediately.
- The supervisor's control plane uses inherited multiprocessing pipes
  (small, reliable messages); the ZeroMQ data plane stays peer-to-peer
  with per-frontend reply routing.
- Metrics rely on the existing Prometheus multiprocess mode: every
  frontend's `/metrics` reports server-wide totals.

## Amendment (2026-11-24): REST moved to a shared listener with
least-loaded accepting

The original decision placed REST on `SO_REUSEPORT` alongside gRPC.
That is replaced for REST: the supervisor binds ONE listening socket
and shares it by file descriptor with every frontend, and each
frontend accepts cooperatively — publishing its open-connection count
into a shared array and deferring (up to 5 ms) whenever a less loaded
peer can take the connection.

- **Motivation**: the kernel's `SO_REUSEPORT` hash distributes
  connections blindly; under real client patterns (few source
  tuples, keep-alive reuse) a small subset of frontends receives
  nearly all connections while others stay idle. The shared queue
  makes the initial assignment least-loaded by construction.
- **Additional properties**: the port is bound once (conflicts fail
  fast instead of crash-looping through the restart budget) and is
  held by the supervisor across frontend restarts (pending
  connections queue instead of being refused).
- **Rejected**: keeping `SO_REUSEPORT` for REST (the measured skew);
  a supervisor-side accept proxy (an extra hop on every request).
- **Scope**: gRPC and the metrics endpoint keep `SO_REUSEPORT`
  deliberately — gRPC connections are long-lived and multiplexed
  (HTTP/2), so initial-assignment skew matters far less than for
  REST's per-connection model, and the grpcio Python API exposes no
  shared-socket mechanism (verified against the official wheels).
