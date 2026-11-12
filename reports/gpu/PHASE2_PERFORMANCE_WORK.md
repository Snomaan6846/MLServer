# MLServer phase 2 performance work: RF-DETR on Tesla T4

Ticket: [RHOAIENG-102092](https://redhat.atlassian.net/browse/RHOAIENG-102092)
Branch: `rhoaieng-102092` (opendatahub-io/MLServer master; developed on master @ `9f4cb29a`, rebased onto current master @ `c39ea879` before PR)
Status: **FINAL** — all work complete; final campaign (P2-V02) validated
the ship candidate at 159/159 gated trials. Only PR creation remains.

## 1. Purpose and scope

Improve MLServer ONNX inference performance for REST and gRPC, using RF-DETR
Small to identify and validate prediction hot-path improvements, building on
the PR #356 opt-in options and the phase-1 study. Scope is steady-state
prediction performance; startup, model loading and shutdown are excluded.

Hardware: one NVIDIA Tesla T4 (Turing, CC 7.5) on a GCP `n1-highmem-32` node,
28 vCPU Guaranteed QoS. Reference engine: Triton on the same node, same model,
same harness. All numbers are reproducible from the archived session data
(see §10).

## 2. Methodology

Every change follows one loop: research-backed (targeting a cost attributed by
profiling or engine research) → implemented with unit tests (`make lint` +
full suite green) → validated on a live server (port ownership + config proof
from logs) → validated again in the exact cluster image (podman) → benchmarked
as a paired sweep against the frozen baseline → gated.

Gates for acceptance: RPS or CPU-ms/request improvement with bootstrap CI
(20k resamples) excluding no-effect; bit-exact REST / byte-exact gRPC
responses in **every** trial; no P99 or memory regressions. Failed experiments
are documented in the ledger only and never merged.

Sweep discipline: 3 trials per configuration, alternating control/candidate
order per round, 60 s warm-up + 60 s measured, per-trial fresh pod with UID
proof, providers verified from server logs, DCGM/cgroup utilization sampled at
3 s. Variance target <3%.

## 3. Baseline (P2-B02)

Stock MLServer (official CUDA image, W1/T2 defaults) on Tesla T4:

| Protocol | C1 | C8 | C1 p50 | GPU util |
|---|---|---|---|---|
| REST (15.26 MB JSON) | 3.99 RPS | 7.85 RPS | 250 ms | 19–23% |
| gRPC | 4.88 RPS | 7.94 RPS | 205 ms | 19–24% |

The serving path, not hardware, was the limit: ~7.9 RPS with CPU ~7% and GPU
~23%. Profiling (P2-B03) attributed the stock REST cost to two overlapped
serial legs — frontend (~99 CPU-ms: h11 parse + orjson decode + Pydantic
validation + dispatch) and worker (~120 CPU-ms: NumPy conversion + ORT +
response construction) — with `rest_worker_json` destroying that overlap
(0.67×, hardware-class dependent: phase-1's +86% on a fast desktop CPU became
−25–33% on this node's slower cores).

gRPC opt-in config (PR #356: W2 + raw input buffers + zmq + CUDA
graph/blocking) already reached 31.1 RPS (3.92×, GPU 70–83%, ~26 ms device
time per request).

## 4. Accepted change: KServe v2 binary tensor extension for REST (P2-X02)

Commit `0dd67502`. REST inference gains the KServe v2 Binary Tensor Data
Extension: zero-copy `np.frombuffer` views over the request body, binary
response encoding, JSON fallback with zero behavior change.

**Result: 4.0× REST throughput** — 31.4 vs 7.8 RPS stock at C2–C16
(6.24× at C1), GPU 24% → 80%, CPU 178 → 42 ms/request, request payload
15.26 MB → 3.15 MB (4.8×), response 542 KB → 114 KB. Bit-exact parity vs
the JSON path; fallback compliance verified by a dedicated sweep (final
campaign: JSON fallback 1.05×, parity at every concurrency). The
recommended REST config: binary tensor + zmq transport + W1
(`rest_worker_json` OFF on this hardware class).

## 5. Evaluated and closed (with evidence)

- **httptools HTTP parser** (P2-X01): no measurable gain — HTTP framing is
  a tiny fraction of the body pass-through cost. Closed.
- **Smaller FP32 JSON payloads** (ticket item 3): the reference payload's
  786,432 floats are all already FP32-exact; FP32-trimmed re-serialization
  saves 0.0% (orjson emits shortest-roundtrip repr). Within the JSON + FP32
  contract no shrink exists; binary transport is the only smaller FP32
  representation. Closed.
- **Zero-copy IPC serialization, msgpack** (P2-X03): three-session arc —
  v1 broken (two ZMQ multipart bugs, caught only on cluster after a
  false-positive local validation), v2 correct but 0.45× (response lists
  walked element-wise: 21 ms + 550 KB metadata per response), v3 enhanced
  with numeric-list framing to 1.00–1.02× (C1 +2.3% CI-significant, parity at
  load). **Archived, not shipped**: pickle's total IPC cost is ~1.2 CPU-ms per
  request (~3% of budget) — no serializer can win meaningfully at this
  workload. Full lineage on branch `rhoaieng-102092-msgpack-archive`.
- **CUDA graph replay on the binary REST path** (P2-X04): +0.3–0.6%
  (CI-significant at 4/5 concurrencies), zero regressions. **Config
  recommendation only** — `cuda_graph=true` for fixed-shape FP32
  single-input models; no code change. Key negative result: the remaining
  Triton gap is NOT kernel-launch overhead.

## 6. Phase-3 profiling attribution (P2-X05)

First frame-level attribution of the optimized paths (py-spy 100 Hz,
`--subprocesses`; memray allocation audit; C8):

- **Binary REST (W1):** worker 64% / parent 36% of samples. ORT execution
  (GPU + H2D/D2H + sync) 51.2% ≈ 24 CPU-ms/request; **the worker's
  post-inference response leg (~6 CPU-ms: codec `tolist` + pickling the
  28,500-float output list) explains the measured 18% GPU idle**. Parent:
  h11/ASGI ~8 ms, binary endpoints ~7 ms, IPC ~6 ms.
- **gRPC opt-in (W2):** parent 72% / workers 22%. **`model_copy(deep=True)`
  in the raw-output converter = 33.2% of all samples (~11 CPU-ms/request)**
  — a deep copy of every output tensor's Python list before a field-clearing
  that `include_contents=False` never reads. Pure waste.
- **Memray:** ~42 MB allocated per request (vs 3.15 MB payload + 114 KB
  response) — churn, not retention (peak 191 MB). IPC pickling 26% of
  allocation volume; the request body is materialized ~4× through the receive
  path.

## 7. gRPC converter fix (P2-X06, ACCEPTED — commit `8424bf32`)

Eliminates the redundant deep copy in both request and response converters
(pack from the originals; `include_contents=False` already skips the data
field). Zero copies, zero mutations, byte-identical wire output.

**Result (paired, 30/30 valid, byte-exact every trial): C1 +19.5%
[CI95 1.180–1.209]** — 19.08 → 22.81 RPS, p50 52.3 → 43.7 ms. C2–C16
parity: the GPU ceiling absorbs the saving. No p99 or memory regressions.
The second accepted change on the ship branch.

## 8. Final campaign — independent validation of the ship candidate (P2-V02)

Fully automated campaign at `8424bf32` (ship image `sha256:be187c53…`,
digest verified in every session's receipts): **159 trials, 4 sessions,
all valid, zero failed requests, every trial bit-exact REST / byte-exact
gRPC gated**. Hermetic test gate: `tox -e mlserver-venv` from
`poetry.lock` — 1427 passed / 37 skipped / 0 failed (excluding
docker-daemon-dependent `tests/cli/test_build.py`; environmental on the
local podman socket, no code-path overlap).

| Comparison | Result |
|---|---|
| REST binary vs stock (C2–C16) | **4.00×** [CI95 tight]; 6.24× at C1 |
| REST JSON fallback vs stock | 1.05× — parity (one marginal cell: C4 p99 +5.4%) |
| CUDA graph on binary | +0.5% — parity (config recommendation) |
| gRPC opt-in vs stock | **3.94×** (4.51× at C1); zero p99/memory regressions |
| Batched binary vs batched stock (C1–C4) | 1.08–1.23× |
| Mixed binary/JSON co-batching | no cost vs pure JSON |

Round-to-round spreads 0.1–2.8% on all unbatched cells (target <3%).

**Route qualification (Gate-8 completion session):** the binary tensor
extension is fully qualified through the actual OpenShift route (edge
termination): 4.07× / 4.01× binary-vs-JSON at C8/C16 through the route —
statistically identical to the direct endpoint (0.997× / 0.999×
route-vs-direct at load). The before-control comparison on the official
baseline image confirms zero regression for unchanged JSON clients
(parity at all concurrencies, both J0 and J1 tracks). gRPC over routes is
not a supported topology on the platform (plaintext h2c backend);
gRPC results remain direct-predictor controls.

**Batched-model finding (Phase-3 input):** with `max_batch_size=4` /
`max_batch_time=5ms`, all batched variants — stock included — collapse at
C8/C16: RPS 8.3 → 1.7, p50 up to 9.2 s, GPU 0–9% (idle while requests
queue), round spreads 80–375%. Pre-existing adaptive-batcher queue
behavior, independent of the phase-2 changes; measured and documented as
Phase-3 Track C1 with the full harness persisted for the fix→verify loop.

## 9. Remaining gap and recommendations

Triton reference on this node/model: 35.1 RPS at 97% GPU (both protocols).
Our REST binary path: 31.4 RPS at 82% GPU (89.5% of Triton). The remaining
gap is the worker's serial non-GPU response leg; the identified lever is an
ndarray-native response pipeline (codec keeps arrays; list conversion only at
the JSON boundary), which would also shrink IPC pickling and the memray
allocation churn. Recommended config for this hardware class: binary tensor
REST (or raw-buffer gRPC) + zmq + W1–W2 + `cuda_graph=true`; keep
`rest_worker_json` OFF on slow-CPU GPU nodes.

## 10. Raw data and durability

- Ledger with full per-experiment evidence: `BENCHMARKING_HISTORY.md`
  (entries P2-B01 through P2-V03).
- Final campaign (P2-V02): `benchmark-results/final-campaign/` —
  RESULTS.md, 159 trials across 4 sessions with receipts (env, pod UID,
  image digest, parity hashes), telemetry, paired analyses, archive
  `rhoaieng-102092-final-campaign-20261112T025530Z.tar.gz` (sha256
  `45312280…`; also uploaded to the `rfdetr-models` PVC).
- Earlier session data, plans, variants, samples, profiles:
  `benchmark-results/sessions/*`.
- Flamegraphs + memray capture: `benchmark-results/sessions/gpu-prof3/profiles/`.
- Scripts: `benchmarking/campaign/` (orchestrator + resume),
  `benchmarking/cluster/*` (hook, sampler, summarizer, validator,
  infra manifests), `benchmarking/study_tools/*` (sweep, client, analyze),
  `benchmarking/overlay/` (image-build Dockerfiles).
- Models: original `rfdetr-small` (SHA-verified `a018914e…`) and the
  dynamic-batch export + settings + parity references, all local under
  `model-preparation/` and `final-campaign/`.
- Archival to durable storage: **DONE** (sub-task 5.3).
- Soak test (60-min sustained): **DONE** — PASS (P2-V01).
- Final independent benchmark of the ship candidate: **DONE** (P2-V02).
