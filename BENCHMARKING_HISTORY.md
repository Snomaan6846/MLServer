# RHOAIENG-102092: Benchmarking History & State Ledger

> Complete benchmarking record for
> [RHOAIENG-102092](https://redhat.atlassian.net/browse/RHOAIENG-102092). This
> document is self-contained — all results, provenance, and failure analysis are
> inline, with no external file dependencies. The companion engineering report
> (FINAL_REPORT.md) is attached to the same ticket. Raw per-trial data (receipts,
> telemetry, parity hashes) is preserved in the benchmarking repository and
> archived to durable storage under Sub-task 5.3; it is not attached due to size.

## Abbreviations & Conventions

### Variant Naming Convention

Every benchmark variant name follows a structured pattern:

```
[campaign]-[wire-track]-[processing-location]-[modifiers]@[payload-spec]
```

**Campaign prefixes** (identify which measurement campaign produced the data):

| Prefix | Campaign | Purpose |
|---|---|---|
| `wb-` | Worker-binary campaign | `rest_worker_binary` feature benchmark |
| `mlp-` | MLP (phase-1 replication) | Frontend-bound regime measurement |
| `cpu-` | CPU thread sweep | CPU thread/worker configuration |
| `gpu-` | GPU thread sweep | GPU thread/cuDNN configuration |
| `head-` | Phase-2 final campaign | Binary tensor + gRPC validation |
| (none) | Intermediate experiments | Profiling, msgpack, httptools, etc. |

**Wire tracks** (what format the requests/responses use):

| Track name | Meaning | Server setting |
|---|---|---|
| `stock-rest` / `json-frontend` | Plain JSON REST (default) | — (all opt-ins off) |
| `binary-rest` / `bin-frontend` | KServe v2 binary tensor REST | `rest_binary_tensor=true` |
| `binary-worker` / `bin-worker` | Binary REST, worker-side processing | + `rest_worker_binary=true` |
| `json-worker` | JSON REST, worker-side processing | `rest_worker_json=true` |
| `grpc-stock` / `stock-grpc` | Stock gRPC (protobuf) | — |
| `grpc-raw` / `grpc-optin` | gRPC with raw FP32 input buffers | `grpc_raw_input_buffers=true` |

**Processing location** (where parsing and encoding happen):

| Suffix | Meaning |
|---|---|
| `-frontend` (or no suffix) | Parsing/encoding in the frontend asyncio process |
| `-worker` | Parsing/encoding in parallel inference workers |

**Configuration modifiers** (appended after the track name):

| Suffix | Meaning | Example |
|---|---|---|
| `-wN` | N parallel workers | `mlp-bin-worker-w8` = binary, worker, 8 workers |
| `-tN` | N intra-op threads | `cpu-w4t8` = 4 workers, 8 threads |
| `-queue` | Queue transport (vs default zmq) | `mlp-json-worker-queue` |
| `-graph` | CUDA graph replay enabled | `wb-binary-worker-graph` |
| `-spin-on`/`-spin-off` | Thread spinning on/off | `cpu-spin-on` |
| `-binoff`/`-binon` | Binary tensor off/on (A2 regression test) | `wb-json-worker-binon` |
| `-wbinon` | Worker-binary on (A3 orthogonality) | `wb-json-worker-wbinon` |
| `@rN` | N rows per request (MLP payload spec) | `mlp-bin-worker@r100` = 100 rows |

**Example decomposition:**

```
mlp-bin-worker-w8-queue@r100
 │   │   │     │   │     │
 │   │   │     │   │     └── 100 rows per request
 │   │   │     │   └──────── queue transport (not zmq)
 │   │   │     └──────────── 8 parallel workers
 │   │   └────────────────── processing in workers
 │   └────────────────────── binary tensor wire format
 └────────────────────────── MLP campaign (phase-1 replication)
```

### Other Abbreviations

| Abbreviation | Meaning |
|---|---|
| **P2-** | Phase 2 (the RHOAIENG-102092 core campaign) |
| **P3-** | Phase 3 (worker-binary feature + thread config + phase-1 replication) |
| **B** (category) | Baseline & Environment |
| **X** (category) | Optimization Experiment |
| **V** (category) | Validation & Delivery |
| **A** (category) | Administrative |
| **J0** | Original JSON wire track (15.26 MB request body for RF-DETR) |
| **J1** | Shorter round-trip FP32 JSON wire track (~8.4 MB) |
| **B** (wire track) | Binary tensor transport (3.15 MB request body) |
| **B-W** | Binary tensor with worker-side processing |
| **G** | Raw gRPC wire track (protobuf with `raw_input_contents`) |
| **C1/C2/C4/C8/C16/C64/C128** | Client concurrency level |
| **RPS** | Requests per second (throughput) |
| **p50/p90/p95/p99** | Latency percentile (milliseconds) |
| **CPU-ms/req** | Whole-server CPU milliseconds consumed per request |
| **GPU util** | GPU utilization percentage (DCGM `GPU_UTIL`, pod-attributed) |
| **CI95** | 95% confidence interval (paired bootstrap, 20,000 resamples) |
| **W1/W2/W4/W8/W12** | MLServer parallel worker count |
| **T1/T2/T4/T8** | ONNX Runtime `intra_op_num_threads` value |
| **ORT** | ONNX Runtime |
| **MLP-54** | Synthetic dense network 54→1024→512→256→128→7 (phase-1 model) |
| **RF-DETR** | RF-DETR Small COCO object detector (main benchmark model) |
| **parity gate** | Per-trial: bit-exact REST / byte-exact gRPC vs frozen references |
| **binary gate** | Per-trial: tensor-payload byte-identity across processing configs |
| **receipts** | Per-trial provenance (image digest, pod UID, env, providers, hashes) |
| **fresh pod** | Server restarted per trial (old pod proven gone via UID check) |
| **stock** | Default MLServer configuration (all opt-ins disabled) |
| **opt-in** | Performance options enabled (raw buffers, zmq, binary tensor, etc.) |
| **zmq** | ZeroMQ inference transport |
| **KSERVE_QUALIFIED** | Binary extension verified through the route path |
| **ROUTED_G_UNAVAILABLE** | gRPC over routes unsupported (h2c backend) |

## System & Baseline Specs
- Cluster: Red Hat OpenShift Container Platform (OCP) 4.21 / CRI-O 1.34.14 — context `default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user`
- Node: 1x NVIDIA Tesla T4 (`syedali-cluster-pool-j6qh5-gpu-t4-a-8kpk4`, GCP `n1-highmem-32`, us-central1-c) — Compute Capability 7.5 (turing), 15360 MiB VRAM, driver 595.91.07 (CUDA 13.2 labels), GPU Operator v9.6.20260929-0 healthy, MIG incapable, sharing=none (GPU exclusive), no taints. Allocatable: 31500m CPU / ~204 GiB RAM / 1 GPU.
- Target Models: RF-DETR Small COCO (ONNX FP32, opset 17, batch 1) — input `input` [1,3,512,512] FP32, outputs `dets` [1,300,4] + `labels` [1,300,91]; model SHA-256 `a018914e11d589b8b51154f00ed1103f148a320a3960f768bd44f57d55d0ab51`. MLP-54 (ONNX FP32, opset 17, dynamic batch) — input `input` [N,54] FP32, output `output` [N,7]; deterministic seeded weights.
- Base Branch: `opendatahub-io/MLServer` master @ `9f4cb29ae70880e34a47c8fccfe0d402e2762fb9` (includes PR #356 phase-1 opt-in perf options; clean tree)
- Client environment: Python 3.12.12, ONNX Runtime 1.23.2, MLServer client built from the PR branch (`rhoaieng-102092`)
- Server images: baseline `quay.io/opendatahub/odh-mlserver-cuda@sha256:4c8168e9955daad321eedcd65bbcd27c9f210cfd56f8317f3fa12b10324de3fe`; ship (phase-2) `quay.io/syedali/mlserver-cuda@sha256:2bcd2f47ddb1347d823649ecf3b3d5b9b4cb4e417cd2aa6eafd5a11b19125b34` @ `8424bf32e46b268bd02ecf1570c5dfe4a4f3f9f7`; ship (worker-binary) `quay.io/syedali/mlserver-cuda@sha256:b6094ab1ae67ef09287e61e70e26ab412731551dd89c5790cc4898843800030c` @ `a772f6312bf8d961222f4f800170b794283ebbff` (branch tip; merged as PR #378)

## Benchmark Summary Matrix

All values are medians from the machine-generated session summaries
(pooled percentiles). Host CPU is whole-pod CPU utilization; CPU-ms/req is
whole-server CPU milliseconds per request.

| Milestone | Protocol | Concurrency | RPS | P50 (ms) | P90 (ms) | P99 (ms) | Host CPU (%) | Tesla T4 GPU (%) | CPU-ms/req |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline stock (T4) | REST (JSON) | 8 | 7.84 | 1,017 | 1,042 | 1,196 | 6.6 | 23.7 | 178 |
| Baseline stock (T4) | gRPC | 8 | 7.92 | 1,012 | 1,028 | 1,070 | 5.5 | 24.2 | 194 |
| Binary tensor (final campaign) | REST (Binary) | 8 | 31.47 | 254 | 258 | 260 | 4.8 | 79.7 | 42 |
| gRPC opt-in (final campaign) | gRPC | 8 | 31.12 | 257 | 261 | 263 | 2.2 | 81.8 | 16 |
| Binary + worker processing | REST (Binary) | 1 | 25.89 | 39 | 39 | 40 | 3.4 | 59.7 | 34 |
| Binary + worker processing | REST (Binary) | 8 | 31.04 | 258 | 261 | 263 | 4.3 | 79.9 | 35 |
| CPU best (W1×T8, spin on) | REST (Binary) | 8 | 6.00 | 1,329 | 1,351 | 1,382 | 28.2 | — | 1,157 |
| MLP binary worker (1-row) | REST (Binary) | 64 | 1,064 | 60 | 64 | 68 | 7.2 | — | 1.9 |
| MLP gRPC raw (1-row) | gRPC | 64 | 1,075 | 58 | 64 | 69 | 6.9 | — | 1.7 |
| Soak (60-min, binary worker) | REST (Binary) | 8 | 30.81 | 260 | 262 | 264 | 4.4 | 80.1 | 40 |

---

## Incremental Execution Log

### Phase 2 — Core Performance Changes

#### P2-B01: GPU & Execution Provider Validation
- Validated CUDA EP on Tesla T4, driver compatibility, single-GPU exclusivity. Established the frozen baseline environment.

#### P2-B02: Master Branch Steady-State Baseline
- Measured stock MLServer (official image, W1/T2): REST JSON 7.84 RPS, gRPC 7.92 RPS at C8. GPU only 24% utilized — frontend-bound, not device-bound.

#### P2-B03: REST Regression Root Cause + Hot-Path Cost Attribution
- py-spy profiling attributed 200+ CPU-ms/request to JSON parsing, float-list construction, and pickling. Identified the binary tensor extension as the only path to significant REST improvement.

#### P2-B04: Triton Reference Measurement
- Triton (C++ serving): binary/gRPC 35.1 RPS @ 96-97% GPU, JSON peak 26.2 RPS. Established the theoretical REST ceiling (~35 RPS) and proved the binary extension could close the gap.

#### P2-X01: httptools Parser Evaluation — RULED OUT
- 30 paired trials: all CIs include 1.0. HTTP protocol framing (~12ms/req) too small relative to JSON body decode (~200ms/req). Not a lever.

#### P2-X02: Binary Tensor Extension — 4.0× REST Improvement — **ACCEPTED**
- KServe v2 binary tensor framing: 3.15MB body instead of 15.26MB JSON. 4.0× throughput at C8 (7.84 → 31.47 RPS), bit-exact outputs, JSON fallback parity.

#### P2-X03: Zero-Copy IPC (msgpack) — **ARCHIVED**
- v1 failed (serialization bugs; 15 of 30 trials failed the parity gate). A second revision was functionally correct but 2.2× slower than the pickle control (13-14 vs 31 RPS at C8). The final revision achieved parity (+2.7% at C1 only by median ratio; C2+ parity). The ndarray-native response pipeline (phase-3 roadmap) supersedes this approach.

#### P2-X04: CUDA Graph Replay — Config Recommendation
- 30 trials: parity (1.002-1.007×). Not required for throughput; optional for fixed-shape FP32 single-input models.

#### P2-X05: Phase-3 Profiling Attribution
- py-spy + memray: gRPC deep copy = 33% of serving CPU; worker response leg = ~6 CPU-ms/req (tolist + list pickling). Identified the ndarray-native pipeline as the phase-3 lever.

#### P2-X06: gRPC Converter Deep-Copy Elimination — **ACCEPTED**
- Eliminates per-response deep copy in raw converters. +19.5% at C1 (19.08 → 22.81 RPS). C2+ parity (device-bound).

#### P2-V01: Soak Test — PASS
- 60 minutes, 113,483/113,483 requests, 31.52 RPS sustained, zero failures.

#### P2-V02: Final Benchmark Campaign — 159 Trials
- 4 sessions (REST, gRPC, batched, mixed): REST binary 4.02×, gRPC 3.93×, JSON fallback parity, batched parity, mixed wire co-batching verified. All 159 trials gated and valid.

#### P2-V03: Gate-8 Completion
- Route qualification (KSERVE_QUALIFIED), before-control parity (ship vs official on J0), J1 track. 84 trials.

#### P2-A01: Binary-Extension Correctness Iteration
- Nine review rounds, adaptive-batching fix (transport-independent list restoration), per-output binary negotiation, true zero-copy input.

---

### Phase 3 — Worker Processing & Thread Configuration

#### P3-B01: Thread-Sweep Infrastructure & Matrix Design
- 14 phase plans, 39+ variants, sampler/summarize/validator reuse. Hook hardened (configurable timeouts).

#### P3-X01: GPU Intra-Op Thread Sweep — T-Insensitive
- 24 trials: T1-T8 flat (31.43-31.52 RPS) on both protocols. Device-bound; thread tuning has no effect on GPU.

#### P3-X02: GPU Thread Spinning — No Effect
- 12 trials: spinning ON/OFF identical at C1 and C8.

#### P3-X03: cuDNN Conv Algorithm Search — EXHAUSTIVE No-Op
- 4 gated EXHAUSTIVE trials (31.15-31.59 RPS, median 31.51): EXHAUSTIVE = HEURISTIC plateau, bit-exact. DEFAULT is numerically divergent (9.0e-05) — gate correctly rejects. No cuDNN change recommended.

#### P3-X04: CPU W×T Full Factorial — T Dominant (2.71×), W Irrelevant
- 36 trials: T1→T8 = 5.06× improvement; T2→T8 = 2.71× (frozen baseline → best). W1/W2/W4 medians within ~3% at every T (no systematic worker effect; the T-lever moves throughput 5×).

#### P3-X05: CPU gRPC Thread Scaling — Confirmed
- Stock gRPC: 1.76 → 3.82 RPS (T2→T8). Opt-in gRPC: 1.98 → 5.35 RPS (+40% from raw buffers at T8).

#### P3-X06: CPU Thread Spinning — +10.9%
- 6 trials: 5.41 → 6.00 RPS at W4×T8. ORT defaults already optimal (tuned parameters add nothing).

#### P3-X07: CPU Inter-Op Execution Mode — SEQUENTIAL Optimal
- SEQUENTIAL = plateau. PARALLEL is unconfigurable via MLServer's JSON settings (ORT enum limitation) and pointless for single-graph models.

#### P3-X08: CPU Thread Affinity — OS-Managed Optimal
- Pinned (corrected format): 5.42 vs OS-managed control: 5.37 RPS (dedicated affinity-session control; an independent W1×T8 control in the follow-up session measured 5.34, i.e. control-level variance ~1-2%). Parity. Keep OS scheduling.

#### P3-X09: CPU Spin-Parameter Tuning — Defaults Already Optimal
- 6 trials: tuned (5.975) vs defaults (5.982). No effect.

#### P3-X10: Cross-Protocol + Latency
- Thread scaling holds on all wire tracks. C1 latency: T8 = 2.64× lower p50 than T2 (194 vs 513ms).

---

### Phase 3 continued — Worker-Binary Feature & Phase-1 Replication

#### P3-V01: Worker-Binary Feature Implementation
- `rest_worker_binary` setting (default off, requires `rest_binary_tensor`, validated). Unit tests for worker processing, settings validation, and server metadata; the merged PR adds 148 test functions in total across 8 test files (binary tensor parsing, worker processing, per-output negotiation, batching normalization, settings validation, server metadata). CI green, hermetic tox green.

#### P3-V02: Worker-Binary Campaign — 76 Trials, All Valid

**A1-core (48 trials) — the feature benchmark:**

| Config | C1 RPS | C1 p50 | C8 RPS | C8 CPU-ms/req |
|---|---|---|---|---|
| Stock JSON (queue) | 3.96 | 252ms | 7.88 | 218.8 |
| Stock JSON (zmq) | 3.97 | 251ms | 8.24 | 240.6 |
| Binary frontend | 24.15 | 41ms | 31.59 | 46.0 |
| **Binary worker** | **25.89** | **39ms** | 31.04 | **34.7** |

**A2-regression (12 trials) — Steve's verification:**
- Worker-JSON + binary ON vs OFF: C8 ratio 1.005× — **parity**. The delegation-scan regression identified in review (≈16% on the worker-JSON path) is eliminated by the single-byte prefilter fix.

**A3-orthogonality (6 trials):** JSON requests unaffected by `rest_worker_binary` (5.885 vs 5.882 RPS at C4, 0.999× — parity).

**A4-batched (6 trials):** Batched model fallback parity (8.25 vs 8.29 RPS, 0.995×).

**A5-CUDA graph (3 trials):** Binary worker + graph = 31.19 RPS (parity).

**A6-soak (1 trial):** 110,917/110,917 requests, 60 min, 30.81 RPS, zero failures.

#### P3-V03: Phase-1 MLP Replication — 148 Trials, All Valid

**B2-core (54 trials) — 6 tracks × 3 row counts at C64, W4, zmq:**

| Track | 1-row | 100-row | 1000-row |
|---|---|---|---|
| Stock JSON | 807 | 411 | 56 |
| Worker JSON | 1,057 | 354 | 51 |
| Binary frontend | 775 | 522 | 83 |
| Binary worker | 1,064 | 466 | 84 |
| Stock gRPC | 1,178 | 402 | 57 |
| gRPC raw | 1,075 | 544 | 85 |

**B3-wsweep (46 trials; one additional attempt aborted without a
measurement):** All tracks flat across W1-W12. Frontend dispatch is the
ceiling.

**B4-csweep (36 trials):** All tracks saturate by C16.

**B5-transport (9 trials):** zmq +10-11% over queue, consistent.

**B6-threads (3 trials):** T1 ≈ T2 (1057 vs 1064 RPS, 0.994×). Threads irrelevant for fast models.

---

### Administrative

#### P2-A01: Binary-Extension Correctness Iteration and Commit Restructure
- (As documented previously)

#### P3-A01: Review Fixes (Steve's findings + CodeRabbit)
- Nested BYTES outputs: recursion fix + framing flatten (3 unit tests)
- `/v2` discovery: `binary_tensor_data` advertisement (2 tests)
- Delegation prefilter: single-byte gate restoring JSON delegation parity (verified by the A2 benchmark: 1.005× at C8)
- Binary detection gating: smuggled IHCL header rejected (2 tests)
- All folded into the binary tensor commit; CI green; hermetic tox green.

#### P3-A02: Documentation
- `docs/user-guide/performance.md` (shipped in the PR): `rest_worker_binary` section added, `/v2` discovery documented.
- FINAL_REPORT.md (attached to this ticket): comprehensive engineering report with all campaign data.

---

## Failure Register

**Total: 1,214 trials, 1,184 valid, 30 invalid.** Every invalid trial is root-caused:

| # | Category | Count | Root Cause | Attribution |
|---|---|---|---|---|
| 1 | cuDNN DEFAULT numerical drift | 3 | Provider-level algorithm difference (9.0e-05); bit-exact gate correctly rejected | Pre-existing ORT behavior |
| 2 | Operator interference | 2 | Deployment reset and phase-3 start during orphaned sweep | [OP] operator |
| 3 | Inter-op PARALLEL unconfigurable | 3 | ORT enum fields cannot be expressed in JSON; TypeError at session creation | [HARNESS] variant bug |
| 4 | Pinned affinity format errors | 6 | Two successive format bugs (0-based IDs; count≠T−1; dual-model collision) | [HARNESS] variant bugs |
| 5 | Experimental msgpack v1 | 15 | Serialization bugs in never-shipped experimental code | [HARNESS]/experimental |
| 6 | Environment/rollout transient | 1 | One-off rollout stall | [ENV] |

**Zero failures attributed to the PR's code changes.**

---

## Data Provenance Map

Every trial produced a provenance receipt (image digest, pod UID, applied
environment, execution providers verified from server logs, parity hashes,
model-ready timing) and aligned telemetry (pod CPU, memory working set, PSI,
GPU utilization at 3-second intervals). Raw session data and distilled
evidence snapshots are preserved in the benchmarking repository and archived
to durable storage (Sub-task 5.3). All headline results are inline in this
ledger and in the attached FINAL_REPORT.md.

| Campaign | Evidence Preserved | Ledger Entries |
|---|---|---|
| Phase-2 final | Per-trial receipts + session summaries | P2-V02 |
| Gate-8 | Route-qualification + before-control results | P2-V03 |
| Soak (frontend binary) | 60-min run log (113,483/113,483 requests) | P2-V01 |
| gRPC converter fix | Paired isolation benchmark results | P2-X06 |
| Thread sweep | 232 distilled evidence files | P3-X01 through P3-X10 |
| Worker-binary campaign | 109 distilled evidence files | P3-V02 |
| MLP campaign | 171 distilled evidence files | P3-V03 |
| Failure register | This document, "Failure Register" section | — |

---

## Scope Changes

- **User mandate (phase-3 workflow).** All serving-engine changes on branch `rhoaieng-102092`. One commit per accepted change; gates: research-backed, unit-tested, paired-benchmarked, CI-passing, bit-exact parity, no p99/memory regression.
- **Worker-binary feature (P3-V01).** `rest_worker_binary` opt-in extending the phase-1 worker-processing pattern to binary tensor frames. Requires `rest_binary_tensor`. Pattern-matched to `rest_worker_json` (same dispatch, gating, accounting, fallbacks).
- **Phase-1 replication (P3-V03).** 150-trial campaign using the phase-1 study's MLP-54 model to validate the frontend-bound regime with current optimizations.
