# ADR-0002: Parallel Inference via Multiprocessing Worker Pools

## Status

Accepted

## Context

Python's Global Interpreter Lock (GIL) prevents true CPU parallelism within
a single process.  For CPU-bound inference workloads (e.g., scikit-learn,
XGBoost), this means that multiple concurrent requests are effectively
serialised, limiting throughput.

Options considered:

1. **Threading** — blocked by the GIL for CPU-bound work.
2. **`concurrent.futures.ProcessPoolExecutor`** — simple but lacks model
   lifecycle management (load/unload per worker).
3. **Custom multiprocessing workers with message queues** — full control over
   worker lifecycle, model loading, and environment isolation.
4. **External process orchestration (e.g., Ray)** — powerful but introduces a
   heavy dependency.

## Decision

Implement parallel inference using custom `multiprocessing.Process` workers
coordinated by a `Dispatcher` and `InferencePool`:

- Each `Worker` runs its own asyncio event loop and `MultiModelRegistry`.
- The main process sends `ModelRequestMessage` (inference) and
  `ModelUpdateMessage` (load/unload) via per-worker `multiprocessing.Queue`s.
- Workers return `ModelResponseMessage` on a shared response queue.
- The `Dispatcher` maps response messages back to `asyncio.Future`s in the
  main process.
- Crashed workers are automatically detected via SIGCHLD, replaced, and all
  models are reloaded into the replacement worker.

The `InferencePoolRegistry` extends this with environment isolation: models
that declare a custom `environment_tarball` or `environment_path` are loaded
in a dedicated pool with its own set of workers running inside that Python
environment.

## Consequences

**Benefits:**
- True CPU parallelism for inference, bypassing the GIL.
- Models are isolated per-worker — a crash in one model's predict does not
  affect the main process.
- Environment isolation allows different models to use different Python
  dependencies without conflicts.
- Transparent to the model author — the `MLModel` API is unchanged.

**Trade-offs:**
- Each worker loads its own copy of the model, multiplying memory usage by
  `parallel_workers`.
- IPC overhead for serialising requests/responses through `Queue`s.
- Workers ignore SIGINT/SIGTERM (the main process coordinates shutdown),
  which can complicate debugging.
- The `select()` loop on queue file descriptors relies on CPython
  implementation details of `multiprocessing.Queue`.
