# MLServer serving-stack enhancement roadmap (phase 3+)

Ticket: RHOAIENG-102092 (phase 2 delivered; this is the forward plan)

Every item is grounded in a phase-2 measurement (ledger entries P2-B01 through P2-V03) or
the reference-engine survey (Triton, vLLM V1 — sources at the end). The
stages are ordered by measured impact and dependency; each is
self-contained enough to be picked up, freshly researched, and executed
independently.

## Starting point (after phase 2, Tesla T4, RF-DETR Small)

| Path | RPS | GPU util | vs Triton |
|---|---|---|---|
| REST binary (recommended cfg) | 31.5 | 83% | 90% of 35.1 |
| REST JSON (unchanged clients) | 7.9 | 23% | 30% of Triton's 26.2 JSON peak |
| gRPC (opt-in + converter fix) | 31.2 (C1 22.8) | ~82% | 89% of 35.1 |
| Batched (any variant, C8+) | 1.7 | 0–9% | collapsed — see Stage 1 |

Triton references (same node/model/harness): binary/gRPC 35.1 @ 97% GPU;
JSON peak 26.2 @ C8.

---

## Stage 1 — Adaptive batcher: overlap scheduling + persistent buffers

**Priority: 1 (measured #1 throughput bottleneck). Source: vLLM V1
overlap scheduler + persistent batch; our P2-V02 measurement.**

**The problem (measured).** The final campaign (60 batched trials + 9
mixed, all bit-exact gated) proved that adaptive batching works at C1–C4
but collapses at C8/C16 for **all variants including stock**: RPS 8.3 → 1.7
(−79%), p50 240 ms → 4,600–9,200 ms, GPU 0–9% (idle while requests queue),
round-to-round spreads 80–375%. The GPU is not the bottleneck — the
batcher's serial design is.

**The two root causes (from profiling + code inspection):**

1. *Serial batch loop* — today: collect requests → build merged input →
   execute → split → respond → collect next batch. The GPU idles during
   every non-execution phase. vLLM V1's overlap scheduler eliminated this
   CPU-GPU serialization and credits it as a primary driver of their 1.7×
   gain from architecture alone (same kernels).

2. *Per-batch rebuild cost* — `_merge_data` concatenates Python lists,
   `_merge_input_parameters` builds fresh dicts, and a new
   `InferenceRequest` is constructed from scratch every batch. For a
   4-request batch of 786K-float inputs, that is ~3M Python float
   references walked per batch construction. vLLM V1 uses persistent
   pre-allocated buffers with NumPy slice-assignment for diffs only.

**The fix (design outline).**
- Maintain a persistent pre-allocated NumPy input buffer sized to
  `max_batch_size × tensor_shape` (allocated at model load; freed at
  unload)
- Fill/evict slots as requests enter/leave the batch (NumPy
  slice-assignment, not list concatenation)
- Pre-stage batch N+1 while the GPU executes batch N (asyncio task
  per batch; result write-back as a separate completion callback)
- The batch window (`max_batch_time`) becomes a deadline for the
  *current* batch's dispatch, not a blocking wait

**Expected outcome.** The GPU should never wait for the batcher; C8/C16
throughput should recover to the C1–C4 baseline (7–8 RPS) or better
(batching should *increase* throughput, not decrease it). The +50–100%
batching expectation from the original analysis becomes reachable.

**Effort:** medium-high (batcher redesign in `mlserver/batching/`).
**Risk:** correctness under concurrent fill/evict (the existing 22 batching
tests + the mixed-BYTES tests cover the contract; extend for concurrency).
**Dependencies:** Stage 2 (ndarray-native) is a natural prerequisite — the
persistent buffers need the codec to keep arrays, not lists.

**Research questions (fresh-start checklist):**
- [ ] Read vLLM V1's scheduler design (sources below) — focus on the
      overlap pattern and the persistent-batch diff application
- [ ] Profile the current batcher at C8 to confirm the serial-loop and
      rebuild-cost attribution (py-spy on the batcher coroutine)
- [ ] Prototype: single-input persistent buffer with slot fill/evict,
      measure batch-construction CPU cost vs the current `_merge_data`
- [ ] Design: asyncio task structure for overlap (who owns the buffer,
      how are results routed back, cancellation semantics)
- [ ] Prototype: two-batch overlap (prepare N+1 while N executes) with
      a mock model, measure the GPU-idle gap
- [ ] Verification plan: re-run the batched campaign (the harness, model
      and references are persisted — `bash benchmarking/campaign/run-final-campaign.sh`
      with the batched sessions)

**Parity note:** through-batcher batch=1 responses were bit-exact
self-consistent in the final campaign. The earlier tolerance-gate
prerequisite applies to multi-request batches (accumulation order), not
to the mechanism itself — the verification campaign must address this.

---

## Stage 2 — Ndarray-native response pipeline

**Priority: 2 (prerequisite for Stage 1's persistent buffers; measured
payoff per line of change). Source: our profiling (P2-X05).**

**The problem (measured).** `NumpyCodec._encode_data` flattens every
output to a Python float list (`data.flatten().tolist()`), and every
downstream layer pays for it: the worker's response leg (~6 CPU-ms/req —
the measured explanation for the 17% GPU idle), IPC pickling of
28.5K-float lists, the binary REST encoder's list→array re-conversion,
and ~42 MB/request of allocation churn (memray). This is also the
identified lever for the remaining ~10% Triton gap on the binary path
(31.5 vs 35.1 RPS).

**The fix.** Keep tensors as ndarrays from the model to the wire; convert
to lists only at the JSON REST boundary. Specific touch points:
- `NumpyCodec`: return the array itself (or a view) from `_encode_data`
- gRPC raw packing: `serialize_bytes` / `deserialize_bytes` already work
  on arrays; remove the list round-trip
- Binary REST encoder (`mlserver/rest/binary.py`): `_encode_output_bytes`
  already handles ndarray data — remove the list re-conversion path
- IPC: pickle serializes ndarrays more compactly than equal-length float
  lists (contiguous buffer vs per-object pointers)

**Expected outcome.** +5–10% RPS toward 35 on the binary/gRPC paths;
large allocation-churn reduction (memray should show <10 MB/request,
down from 42); IPC payload reduction.

**Effort:** medium-large (codecs, REST encoders, gRPC pack — all
bit-exactness gates apply). **Risk:** JSON response bytes must stay
identical (floats round-trip exactly — proven by the msgpack framing
work, P2-X03).

**Research questions:**
- [ ] Read `mlserver/codecs/numpy.py` `_encode_data` and map every caller
- [ ] Read `mlserver/grpc/converters.py` and `mlserver/rest/binary.py`
      `_encode_output_bytes` — identify the list→array→list round-trips
- [ ] Prototype: return `data` (the ndarray) from `_encode_data` for
      non-BYPES datatypes; run the full test suite to find breakage
- [ ] Measure: allocation churn per request (memray before/after)
- [ ] Verify: bit-exact REST / byte-exact gRPC on the full campaign

---

## Stage 3 — Fused JSON→tensor fast path

**Priority: 3. Source: our profiling (P2-B03); vLLM V1's NumPy-native
input preparation.**

**The problem (measured).** Today a JSON request crosses three passes
over 786K Python floats: orjson parse (to Python objects) → Pydantic
`list[Any]` re-walk (validation) → NumPy convert (`np.array(list)`, 18
CPU-ms). Measured headroom for the conversion alone: `np.array(list)`
17.97 ms, `np.fromiter` 14.15 ms, a native typed loop 1.06 ms — ~17
ms/request on the table.

**The fix.** Validate-and-convert in one step at the `TensorData`
boundary. When the request's input is a flat list of numbers matching
the declared shape and datatype, skip the intermediate representation
and write directly into a pre-allocated NumPy array. Fast-path contract
(from the phase-2 gate): exact FP32 bits including signed zeros and
subnormals; strict fallback to the current decoder for mixed types,
nesting, overflow and anything unsupported — never a silent truncate.

**Expected outcome.** 1.3–1.5× JSON frontend throughput (the ~99 CPU-ms
frontend chain drops by the conversion's share).

**Effort:** medium. **Risk:** validation-parity surface (the fallback
must be provably identical to the current path for all non-fast-path
inputs).

**Research questions:**
- [ ] Read `mlserver/types.py` `TensorData` and `mlserver/codecs/numpy.py`
      `decode_input` — identify the exact boundary where the fast path
      can intercept
- [ ] Read Pydantic v2's `model_validate` fast paths — can we leverage
      Pydantic's own NumPy support for the validation step?
- [ ] Prototype: `np.fromiter(iter, dtype, count)` with a count check
      against the declared shape — measure vs `np.array(list)`
- [ ] Prototype: Cython/Rust typed loop (if `fromiter` is insufficient)
      — the 1.06 ms diagnostic sets the bar; note the Konflux multi-arch
      build-matrix requirement (the Cython verdict, P2-X05)
- [ ] Verify: differential test — fast path vs fallback, every edge case
      from the phase-2 gate (signed zero, subnormals, mixed types,
      overflow, nesting)

---

## Stage 4 — Response caching (opt-in, per-model)

**Priority: 4. Source: Triton response cache; vLLM V1 prefix caching
(the zero-overhead design pattern).**

**The problem.** MLServer has no response cache. For workloads with
repeated inputs (common in CV: the same test image, the same batch of
preprocessed crops), identical inference is re-executed.

**The fix.** A local LRU cache keyed on SHA-256 of (model name, version,
all input tensor metadata + data). Opt-in via model-settings
`parameters.extra` (`response_cache: {enable: true, size: N}`); off by
default. Triton's semantics: cache hits return the stored result without
executing the model; only successful inferences are cached; only
CPU-resident tensors are hashable.

**Expected outcome.** Workload-dependent (up to cache-hit-rate ×
inference-cost elimination). For the benchmarking workload itself
(deterministic corpus), the cache would make repeated trials
near-instant.

**Effort:** small-medium. **Risk:** cache-key correctness (hash
collision → wrong result; use SHA-256, not a short hash); memory bound
(LRU eviction; make the size configurable).

**Research questions:**
- [ ] Read Triton's response-cache implementation (local + Redis) —
      focus on the hash construction and the eviction policy
- [ ] Read vLLM V1's prefix caching — the zero-overhead design (constant-
      time LRU, minimized Python object creation) applies to any cache
- [ ] Design: where in the inference path to intercept (before the
      batcher? after it? before ORT but after validation?)
- [ ] Verify: the frozen-hash parity discipline we already enforce makes
      correctness verification straightforward (same input → same
      cached output, bit-exact)

---

## Stage 5 — Multi-process REST frontend

**Priority: 5. Source: our profiling (P2-B03: single frontend process
is the serial ceiling); vLLM V1 EngineCore process separation validates
the design.**

**The problem (measured).** A single asyncio frontend process (GIL)
caps JSON throughput at ~10 RPS (~99 CPU-ms of Python per JSON request)
while the GPU sits at 23%. Worker count is irrelevant (RPS flat
7.88–7.93 across W1–W8). Triton's C++ thread pool parallelizes the same
work; vLLM V1 separates the API server from the engine entirely.

**The fix.** Run N frontend processes over one shared inference pool.
Two viable architectures:
- **SO_REUSEPORT:** N uvicorn workers binding the same port; the kernel
  load-balances connections. Each process has its own event loop.
- **Upstream acceptor:** a lightweight acceptor process distributes
  connections to N frontend processes via Unix sockets.

Either way, the model registry and inference pool must be shared or
replicated across processes. This is the architectural item — design
doc first.

**Expected outcome.** ~N× JSON until GPU-bound — 3–4 frontends ≈ 25–30
RPS (Triton JSON parity). Benefits ALL protocols, not just JSON.

**Effort:** large (architecture). **Risk:** model lifecycle consistency
across processes; metrics aggregation; graceful shutdown coordination.

**Research questions:**
- [ ] Read vLLM V1's EngineCore design — how they separate the API
      server from the engine (ZMQ IPC, async input/output queues)
- [ ] Design doc: process ownership (who owns the model registry?),
      lifecycle (load/unload coordination), metrics aggregation
- [ ] Prototype: 2 frontend processes with SO_REUSEPORT over the
      existing worker pool; measure JSON throughput at C8
- [ ] Investigate: Python 3.14 free-threaded CPython (the queueing
      initiative RHOAIENG-99489 names it) — if the GIL constraint
      relaxes, a single multi-threaded frontend may suffice without
      process separation; monitor the 3.10–3.12 support matrix

---

## Stage 6 — Concurrent ORT execution (multi-instance)

**Priority: 6. Source: Triton instance groups; our observation that
workers are heavyweight.**

**The problem.** MLServer's parallel workers each load a separate model
copy (full Python process + ORT session, ~100+ MB overhead each).
Triton's model instances are lightweight threads sharing one process,
and ORT itself supports concurrent `Run()` calls on a single session.

**The fix.** Two levels to try, in order:
1. *Cheap experiment:* concurrent `Run()` calls on a single ORT session
   from one worker (ORT supports this; our current path is serial per
   worker). If ORT's internal threading handles this well, it is a
   configuration change, not a code change.
2. *Medium effort:* multiple ORT sessions in one worker process (shared
   Python interpreter, separate execution contexts). This overlaps
   H2D/D2H transfers with inference compute, as Triton's instance
   groups do.

**Expected outcome.** Overlap of H2D/D2H with compute; better
small-model GPU utilization. Directly relevant after Stage 1 makes
batching viable.

**Effort:** small (experiment) to medium (multi-session worker).
**Risk:** ORT session thread safety; GPU memory multiplication.

**Research questions:**
- [ ] Read ORT documentation on concurrent `Run()` calls and
      `RunAsync` — thread-safety guarantees, per-session thread pools
- [ ] Experiment: two concurrent `Run()` calls from one Python process
      on the RF-DETR model; measure throughput vs serial
- [ ] Read Triton's instance-group scheduling — how they distribute
      requests across instances and manage GPU memory

---

## Stage 7 — H2D/compute overlap in CudaGraphSession

**Priority: 7. Source: our profiling (P2-X04: kernel launch is NOT
the cost — graphs gave only +0.5%).**

**The problem.** Today upload+replay+download serialize under one lock
in `CudaGraphSession`. The H2D copy (~3 MB per request) and the D2H
download (~114 KB) are on the critical path.

**The fix.** Pipeline input upload of request N+1 with graph execution
of N (double-buffered device inputs, separate CUDA streams). This is the
classic CUDA async pattern: issue the H2D for the next input on a
copy stream while the compute stream replays the current graph.

**Expected outcome.** Recovers the H2D portion of the remaining GPU
idle. **Effort:** large (CudaGraphSession redesign, stream lifecycle).
**Risk:** correctness under concurrent workers sharing the device.
**Dependencies:** Stage 2 (ndarray-native) should land first — the
device buffers should be ndarray-backed.

**Research questions:**
- [ ] Read the current `CudaGraphSession` implementation
      (`mlserver/codecs/onnx/cuda.py`) — map the lock and the
      upload/replay/download sequence
- [ ] Read CUDA stream documentation — copy streams vs compute streams,
      event-based synchronization
- [ ] Prototype: double-buffered input with a separate copy stream;
      measure GPU idle gap (DCGM) before/after

---

## Stage 8 — Piecewise CUDA graphs for the batched path

**Priority: 8 (research). Source: vLLM V1 piecewise CUDA graphs.**

**The problem.** Our current CUDA graph support (phase-2, config
recommendation) captures the full model and requires fixed shapes. The
batched serving path has varying batch sizes (1–4), so full-graph replay
cannot apply.

**The fix.** vLLM V1 captures CUDA graphs piecewise — segment by
segment — excluding operations that need dynamic shapes. This gives
graph-replay speed for the static portions while accommodating dynamic
batch sizes. ONNX Runtime has similar capabilities via its graph
optimization levels. This is an ORT-level investigation more than a
MLServer code change.

**Expected outcome.** Extends CUDA graph benefit to the batched path.
**Effort:** research. **Dependencies:** Stage 1 (batcher fix) should
land first — piecewise graphs on a collapsed batcher are pointless.

**Research questions:**
- [ ] Read vLLM V1's piecewise CUDA graph design (sources below)
- [ ] Read ORT's graph optimization documentation — which levels
      support dynamic shapes, what is the capture/replay boundary
- [ ] Experiment: capture a partial graph (everything except the
      batch-dim-dependent ops) on the RF-DETR model

---

## Stage 9 — FP16 model export (policy-gated)

**Priority: 9 (blocked on precision-policy decision). Source: our
analysis (T4 tensor cores ≈ 2× FP32).**

**The problem.** The Tesla T4's tensor cores deliver ~2× FP32 throughput
for FP16 workloads. Exporting the model in FP16 would increase GPU-bound
throughput beyond Triton's FP32 reference.

**The fix.** Re-export the model with FP16 precision (half-precision
weights and activations). This is a workload-level change, not an
MLServer code change.

**Expected outcome.** ~1.5–1.8× GPU throughput (T4 tensor-core rate);
would exceed Triton's FP32 reference (35.1 RPS).

**Policy prerequisite:** a precision contract change — the current
acceptance gates require bit-exact FP32. FP16 introduces different
numerics (still deterministic, but not bit-identical to FP32). This
requires a user/workload-level decision, not an engine-level one.

---

## Closed with evidence — do not re-open

| Item | Evidence | Why closed |
| --- | --- | --- |
| httptools HTTP parser | P2-X01 | No measurable gain — HTTP framing is negligible vs 15 MB body parsing |
| msgpack IPC serialization | P2-X03 (3-session arc) | Reached parity — IPC is ~3% of CPU budget; no serializer can win meaningfully |
| `rest_worker_json` on slow-CPU GPU nodes | P2-B03 (factorial) | 0.67× regression — destroys frontend/worker overlap; helps only fast-core machines (hardware-class guidance shipped) |
| J1 server-side shrink | Phase-2 evaluation | Client-side serialization change; does not remove per-number parsing |
| Metadata-only protobuf copy | Phase-1 study | Reduced a local cost but regressed serving |
| Direct buffer injection | Phase-1 study | Unstable and usually slower |
| NHWC layout | Phase-1 study | Failed strict raw-output qualification |

## Watch items (no action now)

- **Free-threaded CPython (3.14+)** — the queueing initiative
  (RHOAIENG-99489) names it. Would relax the single-frontend GIL
  constraint that motivates Stage 5; potentially a cheaper path once
  the 3.10–3.12 support matrix moves.
- **ORT version upgrades** — kernel efficiency is part of the residual
  Triton gap; re-measure the reference on each upgrade.
- **Shared-memory tensor transport (Triton extension)** — zero-
  serialization for co-located clients (handle-passing instead of
  bytes). Strictly faster than our binary extension for same-pod use.
  Narrow benefit; revisit if co-located inference becomes a primary
  deployment pattern.

## Techniques evaluated and NOT adopted

| Technique | Source | Why not |
| --- | --- | --- |
| PagedAttention / KV cache | vLLM | LLM-specific; our workloads are single-pass CNN/detector models |
| Continuous batching (iteration-level) | vLLM | Requires token-by-token model execution; our models complete in one forward pass |
| Speculative decoding | vLLM / TRT-LLM | LLM-only (draft model + verification); no equivalent for non-autoregressive models |
| Tensor parallelism across processes | vLLM | Multi-GPU scaling; outside single-GPU scope |
| TensorRT / OpenVINO accelerators | Triton | Different inference engines; deployment alternatives, not MLServer code changes |

## Reference sources

- Triton: [Optimization guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) (dynamic batcher, instance groups, NUMA) · [Response Cache](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/response_cache.html) (local + Redis) · [Shared Memory extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html) · [Rate Limiter](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/rate_limiter.html) · [Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html)
- vLLM V1: [Architecture blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (EngineCore, overlap scheduler, persistent batch, piecewise CUDA graphs, prefix caching, 1.7x from architecture) · [Red Hat article](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture) · [Prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [CUDA Graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- Internal: phase-2 study ledger (`BENCHMARKING_HISTORY.md` entries P2-B01 through P2-V03) · phase-3 research notes (`benchmark-results/phase3-research/notes.md`) · profiling attribution (P2-X05: py-spy + memray)
