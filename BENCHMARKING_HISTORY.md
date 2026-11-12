# RHOAIENG-102092: Benchmarking History & State Ledger

> Canonical location: `MLServer-resources` repo root — the benchmarking repo for
> [RHOAIENG-102092](https://redhat.atlassian.net/browse/RHOAIENG-102092). The sibling
> `MLServer` checkout is reserved for server code changes only. Raw result files are
> stored under `benchmark-results/` (gitignored by this repo's own convention)
> and archived to durable storage in Sub-task 5.3.

## Abbreviations & Conventions

| Abbreviation | Meaning |
|---|---|
| **P2-** | Phase 2 (the RHOAIENG-102092 campaign; phase 3+ will use P3-, P4-, …) |
| **B** (category) | Baseline & Environment — setup, validation, baseline measurement, reference engines |
| **X** (category) | Optimization Experiment — an evaluated change (accepted, closed, or archived) |
| **V** (category) | Validation & Delivery — soak tests, benchmark campaigns, evidence completion |
| **A** (category) | Administrative — commit restructures, tooling changes, non-measurement work |
| **J0** | Original JSON wire track (unchanged-client baseline; 15.26 MB request body) |
| **J1** | Shorter round-trip FP32 JSON wire track (cooperating client; ~8.4 MB — same tensor values, fewer decimal digits) |
| **B** (wire track) | Binary tensor transport (KServe v2 extension; 3.15 MB request body) |
| **G** | Raw gRPC wire track (protobuf with `raw_input_contents`) |
| **C1/C2/C4/C8/C16** | Client concurrency level (total outstanding requests) |
| **RPS** | Requests per second (throughput) |
| **p50/p90/p95/p99** | Latency percentile (milliseconds) |
| **CPU-ms/req** | Whole-server CPU milliseconds consumed per request |
| **GPU util** | GPU utilization percentage (DCGM `GPU_UTIL`, pod-attributed) |
| **CI95** | 95% confidence interval (paired bootstrap, 20,000 resamples) |
| **CI** | Confidence interval (general) |
| **LGT** | Latency guardrail test |
| **W1/W2/W4/W8** | MLServer parallel worker count |
| **T1/T2/T4** | MLServer intra-op thread count for ONNX Runtime |
| **ORT** | ONNX Runtime |
| **IHCL** | `Inference-Header-Content-Length` (the binary tensor framing header) |
| **zmq** | ZeroMQ inference transport (opt-in, replaces the default multiprocessing queue) |
| **stock** | Default MLServer configuration (all opt-in features disabled) |
| **opt-in** | PR #356 performance options enabled (raw buffers, zmq, CUDA graph, etc.) |
| **ship** | The implementation branch's image (contains the phase-2 accepted changes) |
| **official** | The baseline image built from `opendatahub-io/MLServer` master (no phase-2 changes) |
| **before/after** | Paired comparison: official image (before) vs ship image (after) |
| **direct** | Request path via the in-cluster service endpoint (pod network) |
| **route** | Request path via the OpenShift Route (edge-terminated, external DNS) |
| **routed gRPC** | gRPC via OpenShift Route — not supported on the platform (h2c backend, see P2-V03) |
| **KSERVE_QUALIFIED** | The binary tensor extension verified through the actual route path |
| **ROUTED_G_UNAVAILABLE** | gRPC over routes tested and documented as unsupported (see P2-V03) |
| **parity gate** | Per-trial correctness check: bit-exact REST / byte-exact gRPC vs frozen references |
| **receipts** | Per-trial provenance record (image digest, pod UID, applied env, providers, parity hashes) |
| **fresh pod** | Server restarted per trial (old pod proven gone via UID check) |
| **GPU idle** | GPU utilization 0–9% while requests are queued (the batcher-collapse signature) |
| **h2c** | HTTP/2 over cleartext (prior-knowledge, no TLS) — MLServer's gRPC server binding |
| **SLO** | Service-level objective (open-loop latency target; not tested in this campaign) |

## System & Baseline Specs
- Cluster: Red Hat OpenShift Container Platform (OCP) 4.21 / CRI-O 1.34.14 — context `default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user`
- Node: 1x NVIDIA Tesla T4 (`syedali-cluster-pool-j6qh5-gpu-t4-a-8kpk4`, GCP `n1-highmem-32`, us-central1-c) — Compute Capability 7.5 (turing), 15360 MiB VRAM, driver 595.91.07 (CUDA 13.2 labels), GPU Operator v9.6.20260929-0 healthy, MIG incapable, sharing=none (GPU exclusive), no taints. Allocatable: 31500m CPU / ~204 GiB RAM / 1 GPU.
- Target Model: RF-DETR Small COCO (ONNX FP32, opset 17, batch 1) — input `input` [1,3,512,512] FP32, outputs `dets` [1,300,4] + `labels` [1,300,91]; model SHA-256 `a018914e11d589b8b51154f00ed1103f148a320a3960f768bd44f57d55d0ab51` (byte-identical to phase-1 reference); local bundle: `~/rfdetr-small-bundle`
- Base Branch: `opendatahub-io/MLServer` master @ `9f4cb29a` (includes PR #356 phase-1 opt-in perf options; clean tree)
- Benchmarking repo: `~/Workspace/mlserver-performance/MLServer-resources` (branch `resources`, clean; client venv `.venv` = Python 3.12.12, ORT 1.23.2, mlserver editable from sibling checkout)
- Server image (baseline): `quay.io/opendatahub/odh-mlserver-cuda@sha256:4c8168e9955daad321eedcd65bbcd27c9f210cfd56f8317f3fa12b10324de3fe` (tag `odh-master`, built 2026-11-03T21:13:52Z, Dockerfile.cuda: CUDA 12-9 libs, `mlserver_onnx[cuda]`, ORT 1.23.2, trusted runtimes = OnnxModel only)
- Built images (published to `quay.io/syedali/`, local-commit workflow): `mlserver-cuda:odh-master-zmq` @ `sha256:4da0f8c94b8fbc2550d7689ac5ec5acd0bf8070a12f1a5d832fecf5bfabd0ce3` (official CUDA + pyzmq 27.2.0 only); `mlserver:odh-master-zmq` @ `sha256:64efd7b8eeeac45300cd50f38cb18809e9b1786217c6475c7f84ba4551c575a0` (official CPU + pyzmq); `mlserver-cuda:odh-master-prof` @ `sha256:752744d365ded8ede882a13d765a98f632c2f98039f08bfd92436d97ac732c7c` (zmq overlay + py-spy 0.4.3 / viztracer 1.1.1 / memray 1.20.0 for Phase 2; inert when not invoked)
- Cluster objects (namespace `mlserver-perf`): Deployment `mlserver-rfdetr` (1 replica, Recreate, Guaranteed QoS 28 CPU / 32Gi / 1 GPU, nodeSelector Tesla-Tesla T4, no explicit UID — restricted-v2 SCC assigns namespace range); Service `mlserver-rfdetr` (ClusterIP 8080 rest / 8081 grpc / 8082 metrics); PVC `rfdetr-models` (2Gi standard-csi, RWO, bound on Tesla T4 node zone); model repo on PVC at `/mnt/models` (settings.json: 0.0.0.0, ports 8080/8081/8082, parallel_workers 1, grpc_max_message_length 32MiB, access_log off; `rfdetr-small/model-settings.json`: CUDA+CPU providers, device_id 0, intra_op 2 / inter_op 1, spinning disabled, no adaptive batching, all PR#356 perf options at defaults/OFF)
- Helper pod `model-loader` (ubi9, no GPU request, Tesla T4-node pinned for PVC zone affinity) holds PVC access for file updates

## Benchmark Summary Matrix

| Milestone | Protocol | Concurrency | RPS | P50 (ms) | P90 (ms) | P99 (ms) | Host CPU (%) | Tesla T4 GPU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Master, stock, Tesla T4) | REST (JSON) | 1 | 3.995 | 250.4 | 256.5 | 264.9 | 3.5 | 18.6 |
| Baseline (Master, stock, Tesla T4) | REST (JSON) | 8 | 7.853 | 1012.0 | 1034.3 | 1100.6 | 6.7 | 22.9 |
| Baseline (Master, stock, Tesla T4) | gRPC | 1 | 4.882 | 204.8 | 210.5 | 217.1 | 3.5 | 18.6 |
| Baseline (Master, stock, Tesla T4) | gRPC | 8 | 7.943 | 1004.9 | 1024.9 | 1127.4 | 5.7 | 23.8 |
| Baseline (Master, stock, CPU) | REST (JSON) | 1 | 1.422 | 702.0 | 719.9 | 738.4 | 5.8 | — |
| Baseline (Master, stock, CPU) | REST (JSON) | 8 | 1.738 | 4581.5 | 4724.7 | 4760.5 | 7.0 | — |
| Baseline (Master, stock, CPU) | gRPC | 1 | 1.500 | 666.4 | 688.3 | 716.2 | 5.9 | — |
| Baseline (Master, stock, CPU) | gRPC | 8 | 1.723 | 4633.5 | 4807.4 | 4924.2 | 6.7 | — |
| Opt-in phase-1 cfg (Tesla T4) | REST (JSON) | 8 | 5.903 | 1353.1 | 1375.5 | 1393.5 | 4.2 | 19.2 |
| Opt-in phase-1 cfg (Tesla T4) | gRPC | 8 | 31.121 | 257.2 | 260.9 | 262.8 | 3.1 | 81.2 |
| Opt-in phase-1 cfg (CPU) | REST (JSON) | 8 | 1.609 | 4963.5 | 5021.3 | 5097.1 | 6.3 | — |
| Opt-in phase-1 cfg (CPU) | gRPC | 8 | 2.027 | 3950.1 | 4141.7 | 4216.9 | 6.9 | — |
| Fast JSON (orjson) | REST (JSON) | 8 | | | | | | |
| Binary Tensor Ext (final campaign, P2-V02) | REST (Binary) | 8 | 31.47 | 254.4 | 257.9 | 260.0 | 4.8 | 79.7 |
| Binary Tensor Ext (final campaign, P2-V02) | REST (Binary) | 16 | 31.44 | 509.4 | 515.9 | 518.9 | 4.8 | 79.6 |
| Zero-Copy IPC (msgpack, enhanced) — parity; +2.3% C1 only; **archived** | REST (Binary) | 8 | 31.481 | 253.7 | 257.6 | 260.4 | 4.7 | 82.1 |
| Cython / IOBinding | gRPC | 8 | | | | | | |

## Scope Changes

- **User mandate (phase-3 workflow).** All serving-engine changes
  happen on a SINGLE branch `rhoaieng-102092` in the MLServer repo (base:
  master @ 9f4cb29a). One commit per accepted change; a commit lands ONLY
  after the change passes the full gate set:
  1. research-backed (targets a cost attributed by profiling or the 3.0
     engine research),
  2. implemented with unit tests; repo `make lint` + `tox -e mlserver-venv`
     pass,
  3. benchmarked via paired sweep against the frozen baseline (and stacked
     comparisons against the branch HEAD as changes accumulate),
  4. gates: RPS or CPU-ms/request improvement with bootstrap CI excluding
     no-effect; bit-exact REST / byte-exact gRPC parity; no P99 or memory
     regression; fallback compliance for protocol-facing changes,
  5. documented: ledger entry with data; commit message carries the
     evidence summary and ledger reference.
  Failed experiments are documented in this ledger only (no commits).
  Build discipline: fast wheel-overlay candidate images for the loop;
  full Dockerfile.cuda recipe build to confirm any final/ship claim.
  PRs to opendatahub-io/MLServer remain ask-first.

- **User decision:** Phase 3 gains a research workstream — comparative
  analysis of industry inference engines (Triton, vLLM, OVMS, and related)
  to identify adoptable architectural patterns and quantify the gap to
  best-in-class on identical hardware/workload. Planned deliverables: (a)
  architecture research mapped to our measured bottlenecks, (b) live Triton
  reference measurement on the Tesla T4 with the same RF-DETR model and harness
  (mirrors the phase-1 OVMS methodology; vLLM serves LLMs only, so it is an
  architecture reference, not a benchmark target). **Sequencing constraint
  (user): the live Triton benchmark must NOT run concurrently with the
  MLServer sweep campaign — it is scheduled strictly after the stock, CPU,
  and paired sessions complete, since the Tesla T4 node is shared and concurrent
  load would contaminate both measurements.** Desktop research (docs/source
  study) is not resource-contending and may proceed in parallel. Findings
  feed Phase 3/4 design decisions.

## Incremental Execution Log
| ID | Title | Outcome |
|---|---|---|
| P2-B01 | GPU & execution provider validation | Baseline established |
| P2-B02 | Master branch steady-state baseline | Stock: 7.9 RPS (REST/gRPC), GPU 23% |
| P2-B03 | REST regression root cause + cost attribution | `rest_worker_json` 0.67×; single frontend = ceiling |
| P2-B04 | Triton reference measurement | 35.1 RPS (binary/gRPC), 26.2 (JSON peak) |
| P2-X01 | httptools parser evaluation | Closed — no gain |
| P2-X02 | Binary tensor extension | **Accepted — 4.0× REST** |
| P2-X03 | msgpack IPC (3-session arc) | Archived — parity |
| P2-X04 | CUDA graph replay | Config recommendation (+0.5%) |
| P2-X05 | Phase-3 profiling attribution | gRPC deepcopy 33%, response leg 17% GPU idle |
| P2-X06 | gRPC converter deep-copy elimination | **Accepted — +19.5% C1** |
| P2-V01 | Soak test (60 min) | **Pass** — 113,483/113,483 |
| P2-V02 | Final benchmark campaign | **159/159 gated — REST 4.0×, gRPC 3.9×** |
| P2-V03 | Gate-8 completion (route, before-control, J1) | Route qualified; before-control parity |
| P2-A01 | Commit restructure | Administrative |


Entries use structured IDs: `P<phase>-<category><nn>`, where the
category is `B` (baseline), `X` (experiment), `V` (validation/delivery),
or `A` (administrative). Within a category, numbers are sequential.
References from other documents use the full ID (e.g. `P2-X02`).

### Baseline & Environment

### P2-B01: GPU & Execution Provider Validation (Sub-task 1.1)
- **Sub-task ID:** 1.1
- **Git Commit / Working Tree State:** MLServer master @ `9f4cb29a` (clean); MLServer-resources @ `resources` (clean); no code changes made.
- **Action Taken:**
  1. Located live Tesla T4 cluster (default kubeconfig context; the two `mlserver-testing/*` contexts are dead/DNS).
  2. Verified Tesla T4 node allocatable limits, taints (none), conditions (all healthy), GPU Operator stack (driver DS 9.6.20260929-0, device plugin, DCGM, container toolkit, GFD — all Running; CUDA + device-plugin validators Completed).
  3. Prepared RF-DETR Small bundle locally (`model-preparation/prepare.py`; PyTorch CPU export; onnx checker + PyTorch detection parity passed).
  4. Created namespace `mlserver-perf`, PVC (zone-pinned to Tesla T4 node via loader pod nodeSelector), copied model repo onto PVC, SHA-256 verified byte-identical.
  5. Deployed baseline server from official CUDA image pinned by digest; model loaded with `['CUDAExecutionProvider', 'CPUExecutionProvider']` active.
  6. Container GPU access proven: pod with `nvidia.com/gpu: 1` on Tesla T4 node saw `/dev/nvidia*`, `nvidia-smi` OK (Tesla Tesla T4, 595.91.07, 15360 MiB), ORT 1.23.2 CUDA EP **activated** (tiny ONNX Add model executed on GPU, correct output).
  7. REST correctness via port-forward: ready 200, metadata shapes exact, two identical requests byte-equal, parity vs CPU reference: dets max-abs-diff 1.462e-04 / labels 3.166e-04 (expected CUDA-vs-CPU FP32 divergence; same order as export-time PyTorch-vs-ONNX diffs). gRPC smoke via `benchmarking/inference.py` (2 requests) succeeded.
- **Commands Executed:** (key)
  - `oc get nodes -l nvidia.com/gpu.present=true` (55-cluster scan; only default ctx alive with Tesla T4)
  - `oc -n mlserver-perf apply -f gpu-provider-check.yaml` (GPU/EP proof pod)
  - `poetry -C model-preparation/export-env run python $RESOURCES_ROOT/model-preparation/prepare.py --output ~/rfdetr-small-bundle`
  - `oc -n mlserver-perf cp ... model.onnx model-loader:/mnt/models/rfdetr-small/` + `sha256sum` verify
  - `oc -n mlserver-perf apply -f mlserver-baseline.yaml` (Deployment+Service, image by digest)
  - `oc port-forward` + `rest_check.py` (correctness), `inference.py --protocol grpc` (smoke)
- **Observed Metrics / Profiling Findings:**
  - Throughput: N/A (validation only; port-forward latencies are not benchmark data)
  - Latency (P50 / P99): N/A this sub-task
  - Memory / CPU / GPU: node idle-allocation 469m CPU; GPU idle at 0 MiB used after checks; server pod 1/1 Running, model loaded in ~35 s from pod start
- **Numerical Parity Status:** Pass (CUDA vs CPU-reference within FP32 divergence: max 3.17e-04; exact determinism across identical requests; strict Δ=0 / MSE≤1e-6 gates reserved for optimized-vs-baseline CUDA-to-CUDA comparisons in Sub-task 4.3)
- **Decisions & Next Steps:**
  - Baseline image frozen at digest `sha256:4c8168e...`; model frozen at SHA `a018914e...`; resource envelope frozen (28 CPU / 32Gi / 1 GPU, Guaranteed QoS).
  - Known gap: `pyzmq` is NOT in the official CUDA image (optional `zmq` extra; not in `odh-runtimes-cuda` group) → the phase-1 "selected config" (zmq transport) baseline requires a minimal overlay image (`FROM official digest` + `pip install pyzmq==27.2.0`) to be built via in-cluster BuildConfig (Sub-task 1.2).
  - OpenShift learnings recorded: namespace PSA `restricted` + SCC `restricted-v2` (deployment pods get namespace-range UIDs; do not set runAsUser/fsGroup explicitly); loader/pods created directly by admin user pass via `anyuid` SCC.
  - Ledger relocated to `MLServer-resources` root (this file) per ticket intent — the MLServer checkout stays code-only.
  - Raw phase-2 results will be written to `benchmark-results/` (already gitignored in this repo); curated numbers stay in this ledger; final report target `reports/` (phase-1 convention).
  - **CPU-vs-GPU coverage:** every run records Host CPU % + Tesla T4 GPU % (matrix columns). **CONFIRMED (user): full CPU-device sweep added to Sub-task 1.2** — same RF-DETR model/harness/sweeps (c ∈ {1,2,4,8,16} × REST+gRPC, 3 reps) on CPUExecutionProvider, at both stock and opt-in configs, to quantify the Tesla T4 benefit and isolate CPU-side serving overheads. Controlled comparison: same image, only the provider list differs (second PVC + CPU deployment required; current PVC is RWO/attached to the GPU pod).
  - Next: Sub-task 1.2 — deploy in-cluster benchmark client pod (cross-node, same zone, direct Service endpoint), then execute concurrency sweeps c ∈ {1,2,4,8,16} REST + gRPC on stock config (and the PR#356 opt-in configs) with ≥60 s warm-up, 3 repetitions, <3% variance target; collect CPU (cgroup/metrics) and GPU (DCGM) utilization.

### P2-B02: Master Branch Steady-State Baseline (Sub-task 1.2 complete)
- **Sub-task ID:** 1.2 (incl. user-added CPU-device leg and opt-in paired comparisons)
- **Git Commit / Working Tree State:** MLServer master @ `9f4cb29a` (clean, no code changes); MLServer-resources @ `resources` + local commit `ee8f457` (cluster tooling + ledger, not pushed).
- **Action Taken:**
  1. Built the full in-cluster benchmarking harness: client pod (direct Service endpoint, cross-node CPU isolation on worker-c, Guaranteed 4 CPU), sweep prepare hook (variant → env/image patch → fresh pod with UID proof → readiness → provider verification → REST bit-exact + gRPC byte-exact parity → receipts), RBAC (sweep-hook SA), utilization sampler (kubelet Summary cumulative CPU + DCGM GPU/VRAM/power/throttle, pod-attributed, 3 s), summarizer (pooled p50/p90/p95/p99, CPU-ms/request integration), 9 frozen sweep plans.
  2. Executed 8 sessions / 180 trials / 0 invalid: 4 stock (GPU/CPU × REST/gRPC, 15 trials each) + 4 paired (stock ↔ opt-in interleaved, alternating order, 3 rounds, c ∈ {1,2,4,8,16}, 60 s warm-up + 60 s measured, per-trial fresh-process restart; 30 trials each).
  3. Ran `analyze.py` paired bootstrap (20k resamples) for all four comparisons.
  4. Phase 3.0 desktop research completed (`benchmark-results/phase3-research/notes.md`).
- **Observed Metrics / Profiling Findings:** (full per-cell CSVs in `benchmark-results/sessions/*/metrics-*.csv`)
  - **Stock W1/T2 ceiling on Tesla T4: ~7.9 RPS both protocols** (REST 3.99→7.90, gRPC 4.88→8.00 across C1→C16); latency scales linearly with c beyond saturation; CPU ~7% of 28 cores, GPU 19–37% — the serving path, not hardware, is the limit.
  - **gRPC opt-in config (W2 + raw buffers + zmq + graph/blocking): 3.92× [CI95 3.90–3.98] decisive win** — 31.1 RPS at C2–C16, C1 p50 205→52 ms, CPU-ms/request 170–200 → 26–30, GPU util → 70–83% (Tesla T4 becomes device-bound at ~26 ms kernel time/request; further gains need faster inference or more device concurrency, not serving-path work).
  - **REST opt-in config (W8 + worker-JSON + zmq): 0.74× [CI95 0.73–0.75] REGRESSION at C2+** (5.9 vs 7.9 RPS) despite lower CPU/request (164–213 vs 191–235 ms). C1 only: 1.14×. **Phase-1's REST gains (+86% on RTX 4070/fast host CPU) do NOT transfer to this slow-CPU Tesla T4 node.** Candidate mechanisms for Phase 2 profiling: ZeroMQ low-concurrency caveat (phase-1's own warning: C1 regression 10.44→8.89), W8 CUDA context thrashing on a single Tesla T4, per-core speed changing the distribute-vs-serialize balance. Escalated per ticket guidance ("escalate conflicting results or compatibility tradeoffs").
  - **CPU-device leg:** Tesla T4 = 4.6× CPU-device at stock (7.9 vs 1.72 RPS); gRPC opt-in wins on CPU device too (1.15–1.26×, 1.71→~2.0 RPS); REST opt-in a wash (0.93×). Device-independent serving overhead isolated at ~200–250 CPU-ms/request (GPU-device total minus CPU-device ORT compute delta).
  - **Stock reproducibility:** paired-session stock rows reproduce solo stock sessions within noise (7.87–7.94 vs 7.85–7.97 RPS).
  - **httptools MISSING confirmed live** (official CPU + CUDA images): stock REST parses 15 MB bodies with pure-Python h11; uvloop IS active. Phase 3.1 candidate.
  - **Variance gate:** GPU cells pass (<2% typical; two cells 3.4–3.7%). CPU-device cells at ~1.7–2.0 RPS show 4–7.5% spread on a few cells (absolute ±0.05–0.10 RPS); medians stable across concurrencies. Flagged, not re-run (absolute jitter immaterial to conclusions).
- **Numerical Parity Status:** Pass — every one of the 180 campaign trials gated on REST bit-exact + gRPC byte-exact responses vs per-device references, including CUDA graph replay on Tesla T4 (Turing) and all opt-in paths. Providers verified from server logs per trial.
- **Decisions & Next Steps:**
  1. Baseline FROZEN: matrix above + session artifacts are the reference for all Phase 3/4 paired comparisons.
  2. Phase 2 profiling starts now on `mlserver-cuda:odh-master-prof` @ `sha256:752744d3…` (py-spy/viztracer/memray): (a) attribute the ~250 CPU-ms/req stock REST cost across stages; (b) **diagnose the REST opt-in regression mechanism** (zmq-vs-queue A/B, W-sweep on Tesla T4, worker-stack sampling) — this is now a first-class profiling target; (c) memray copy/allocation audit.
  3. Phase 3 priorities (from baseline + research): binary tensor extension (largest attributed cost), httptools parser, IPC serialization (vLLM msgspec framing), then re-tuned worker/transport configs per hardware class.
  4. Housekeeping: check why the CPU server pod was recreated overnight (~03:00 UTC Nov 10; events/restart reason) for the stability record; variance-flagged CPU cells may be re-run if Phase 3 needs tighter CIs on them.
  5. Triton live reference benchmark (Phase 3.0) queued after profiling, per user sequencing constraint.

### P2-B03: REST Regression Root Cause + Hot-Path Cost Attribution (Sub-task 2.1/2.2)
- **Sub-task ID:** 2.1 (CPU cost isolation) + 2.2 partial (py-spy; memray pending) + REST-regression root cause
- **Git Commit / Working Tree State:** MLServer master @ `9f4cb29a` (clean, no code changes); MLServer-resources @ `2d826bf` (local commits: tooling, ledger, data-driven variants).
- **Action Taken:**
  1. REST-regression factorial (24 trials, C8, 3 rounds, all variants on the SAME overlay image so transport/W are the only variables): `{queue,zmq} × {W1,W2,W4,W8}` with worker-JSON on, plus in-session stock control.
  2. py-spy profiling (100 Hz, 150 s window per trial, wrapper-as-ancestor for ptrace safety, profiles written to model PVC): stock vs W1+worker-JSON minimal reproducer. 16,021 / 16,021-sample flamegraphs, 0 errors.
  3. Stock worker-count sweep (12 trials): stock env at W1/W2/W4/W8 (the missing factorial control).
  4. Flamegraph attribution table (per-request, normalized by measured request counts).
- **Observed Metrics / Profiling Findings:**
  - **Factorial (C8, CI95 via paired bootstrap):** stock control 7.92 RPS; worker-JSON at ANY worker count with queue = 0.67–0.69× (5.33–5.38 RPS); with zmq = 0.74× (5.84–5.87). Worker count is IRRELEVANT to the regression; zmq recovers ~10% over queue. Overlay image ≡ official image (stock-ov 7.92 vs official 7.85–7.94).
  - **Stock W-sweep (C8):** 7.876 / 7.859 / 7.887 / 7.930 RPS at W1/W2/W4/W8 — **flat**. Stock REST is frontend-serial-bound on this node; more workers do not raise the ceiling (GPU util 22→32% merely absorbs the limited stream).
  - **py-spy attribution (per request, profiler-inflated ~20% vs cgroup):**
    - Stock: frontend request chain ≈ 99 CPU-ms (h11 HTTP parse + orjson decode + Pydantic validate + dispatch); worker ≈ 120 CPU-ms (NumPy + ORT + response objects); the two OVERLAP → ~126 ms effective serial → 7.9 RPS.
    - worker-JSON W1: all stages consolidate into the worker ≈ 274 CPU-ms serial; frontend ≈ 47 (relay) → ~232 ms effective → 4.3 RPS. **Pipeline-overlap destruction is the regression mechanism.**
    - h11 (pure-Python HTTP parsing) is the hottest single frontend frame — stock images lack httptools (uvicorn `auto` → h11).
    - Response chain (objects → pickle → queue → unpickle → validate → orjson encode) is comparable in magnitude to the request chain.
  - **Phase-1 reconciliation:** on the fast-CPU 4070 host the frontend serial cost was ~2× smaller, so distributing parse work to W4/W8 workers genuinely raised the ceiling (14.72→27.34 RPS). On this node the frontend remains the cap at any W, so redistribution only adds relay cost → net −25–33%.
- **Numerical Parity Status:** Pass — all 38 profiling-era trials gated on bit-exact REST / byte-exact gRPC responses (incl. fresh restarts under the py-spy wrapper).
- **Decisions & Next Steps:**
  1. REST optimization must attack the FRONTEND serial path, not worker distribution: (a) KServe binary tensor extension (Phase 3.2 — removes JSON parse, float conversion, and orjson response encode; 15.26 MB → 3.15 MB), (b) httptools parser (Phase 3.1 — hottest frontend frame), (c) IPC serialization (pickle → msgspec-style framing, Phase 3.3).
  2. `rest_worker_json` needs a hardware-class caveat in docs until Phase 3 lands (it regresses −25–33% on slow-CPU nodes at every W; helps only where per-core CPU is fast).
  3. gRPC path: already device-bound at 31 RPS with opt-in config; remaining serving-path work there is second-order.
  4. memray allocation audit (Sub-task 2.2) remains open — scheduled with Phase 3 candidate testing.
  5. Jira baseline update POSTED to RHOAIENG-102092 as comment 19449251 (user-approved). Posted as proper ADF via `acli jira workitem comment update --body-adf` after the MCP markdown converter mangled tildes/plus-signs/underscores; the exact ADF is archived at `benchmark-results/jira-comment-19449251-adf.json`.

### P2-B04: Triton Reference Measurement (Phase 3.0 research)
- **Sub-task ID:** 3.0 (competitive research — Triton live reference)
- **Git Commit / Working Tree State:** MLServer branch `rhoaieng-102092` @ `9f4cb29a` (no changes yet); MLServer-resources @ `4f2cf5c4` (validator + sampler fix).
- **Action Taken:**
  1. Deployed Triton Inference Server 26.10-py3 (nvcr.io) on the same Tesla T4 node, same envelope (28 CPU / 32 Gi / 1 GPU Guaranteed), same RF-DETR model (SHA-verified), same payloads.
  2. Built triton_hook.py with the same rigor as the MLServer hook: exclusivity enforcement (MLServer scaled to zero), fresh Triton process per trial, readiness + model-ready checks, ONNX Runtime backend verification from logs, value-parity gate vs MLServer CUDA reference (tolerance 1e-3 for cross-ORT-build variance; observed 1.90e-04).
  3. Built the KServe binary-tensor REST payload (JSON header + 3.15 MB tensor, `Inference-Header-Content-Length: 225`) — doubles as the Phase 3.2 fixture.
  4. Ran 3 sessions (REST JSON, REST binary, gRPC) with full utilization sampling; data-completeness validated per session.
- **Observed Metrics / Profiling Findings:**
  - **Triton gRPC (validated COMPLETE, 15/15 trials, <0.5% spread):**
    | c | RPS | P50 ms | CPU-ms/req | GPU% |
    |---|---|---|---|---|
    | 1 | 31.5 | 32 | 30 | 77% |
    | 2 | 35.1 | 57 | 30 | 97% |
    | 8 | 35.0 | 228 | 29 | 96% |
    | 16 | 34.9 | 456 | 25 | 97% |
    **This is the hardware ceiling** — the Tesla T4 is fully saturated at ~35 RPS for RF-DETR FP32 batch-1.
  - **Triton REST JSON (validated COMPLETE, 15/15 trials, high variance 22-74% at C4+):**
    C8: ~26.2 RPS, 153 CPU-ms/req, 58% GPU. High variance is a Triton dynamic-batching characteristic (settles differently per fresh process).
  - **Triton REST binary (validated COMPLETE, 15/15 trials, <1% spread):**
    | c | RPS | P50 ms | CPU-ms/req | GPU% |
    |---|---|---|---|---|
    | 1 | 31.9 | 31 | 29 | 80% |
    | 2 | 35.1 | 57 | 26 | 97% |
    | 8 | 35.1 | 228 | 33 | 97% |
    | 16 | 35.1 | 455 | 32 | 97% |
    **Triton REST binary ≈ Triton gRPC** — the KServe binary tensor extension completely eliminates the REST penalty and saturates the GPU at the same ~35 RPS ceiling.
  - **Comparison summary (at C8):**
    | Engine | Protocol | RPS | CPU-ms/req | GPU% | Gap vs MLServer stock |
    |---|---|---|---|---|---|
    | Triton | REST binary | 35.1 | 33 | 97% | 4.5x |
    | Triton | gRPC | 35.0 | 29 | 96% | 4.4x |
    | Triton | REST JSON | 26.2 | 153 | 58% | 3.3x |
    | MLServer opt-in | gRPC | 31.1 | 26 | 81% | 3.9x |
    | MLServer stock | REST/gRPC | 7.9 | 215 | 23% | 1.0x |
  - **Critical finding for Phase 3.2:** Triton REST binary (35.1 RPS) = Triton gRPC (35.0 RPS). The KServe binary tensor extension completely eliminates the REST-vs-gRPC gap. If MLServer implements the same extension, the theoretical REST ceiling is the GPU limit (~35 RPS on Tesla T4) — a potential **4.5× improvement** over the current 7.9 RPS stock REST.
  - **Key insight:** MLServer's gRPC opt-in config (31.1 RPS / 81% GPU) is already within 12% of Triton's gRPC ceiling (35.0 / 96.5%). The remaining gRPC gap is small. The REST gap (7.9 vs 26.2) is where Phase 3 work matters most.
  - **Value parity:** Triton outputs match MLServer CUDA reference within 1.90e-04 max-abs-diff (cross-ORT-build variance, well within the 1e-3 tolerance).
- **Numerical Parity Status:** Pass — value parity within tolerance on all trials.
- **Decisions & Next Steps:**
  1. Triton REST binary re-run in progress; will complete the three-dimensional picture.
  2. httptools paired sweep queued immediately after (image built, plan ready).
  3. Binary tensor extension design notes next — Triton's binary REST number will directly inform the expected gain for MLServer's Phase 3.2 implementation.
  4. MLServer gRPC opt-in is close to the hardware ceiling — remaining gRPC work is second-order compared to REST.


### Optimization Experiments

### P2-X01: httptools Parser Evaluation (Phase 3 — ruled out)
- **Sub-task ID:** Phase 3 candidate evaluation
- **Git Commit / Working Tree State:** MLServer branch `rhoaieng-102092` @ `9f4cb29a` (no code changes — candidate rejected); MLServer-resources @ `82c39e0a`.
- **Action Taken:**
  1. Built httptools overlay image (`quay.io/syedali/mlserver-cuda:odh-master-httptools` @ `sha256:a558cc25...`): zmq overlay + httptools (uvicorn 0.32.0 `http=auto` picks it up when importable).
  2. Ran paired sweep: `gpu-httptools` vs `gpu-stock-ov` (identical env, same overlay base, only httptools presence differs). 30 trials, 0 invalid, data-completeness validated COMPLETE.
  3. Paired bootstrap analysis (20k resamples).
- **Observed Metrics / Profiling Findings:**
  - RPS ratios (httptools/stock): 0.999 @C1, 0.976 @C2, 1.014 @C4, 0.999 @C8, 0.999 @C16 — all CIs include 1.0.
  - CPU-ms/request: httptools 198-222 vs stock 196-245 — no consistent reduction.
  - **Conclusion: httptools is NOT a meaningful lever for this workload.** The HTTP protocol framing (~12 ms/req from py-spy) is too small relative to the JSON body decode + float conversion (~200 ms/req total). The parser swap doesn't touch the actual bottleneck.
- **Numerical Parity Status:** Pass (all trials bit-exact/byte-exact as usual).
- **Decisions & Next Steps:**
  1. **httptools ruled out** — no commit on `rhoaieng-102092`. Documented here as a negative result with full evidence.
  2. This sharpens the Phase 3 focus: the binary tensor extension (which eliminates JSON parsing entirely) is the only remaining path to significant REST improvement. The Triton reference shows the ceiling is ~35 RPS with binary transport.
  3. Next: binary tensor extension design notes.

### P2-X02: Binary Tensor Extension — 3.8× REST Improvement (Phase 3.2)
- **Sub-task ID:** 3.2 (KServe binary tensor extension)
- **Git Commit / Working Tree State:** MLServer branch `rhoaieng-102092` @ `c596dde6` (binary tensor extension, 832 lines, 5 files); MLServer-resources @ `1288067f`.
- **Action Taken:**
  1. Implemented the KServe v2 Binary Tensor Data Extension for REST inference
     (zero-copy NumPy views via np.frombuffer, binary response encoding,
     JSON fallback with zero behavior change).
  2. Code review round: simplified header lookup (Starlette normalizes to
     lowercase), reused parse_body/encode_response from worker.py, added
     _infer_metrics accounting, removed duplicate logic.
  3. Local verification: 31/31 unit tests, 161/161 REST suite (zero
     regressions), bit-exact parity, flake8/black/mypy clean.
  4. Built candidate image: zmq overlay + branch wheel (force-reinstall)
     → quay.io/syedali/mlserver-cuda:rhoaieng-102092-binary @ sha256:883f1dbe...
  5. Ran two paired sweeps (60 trials total, 56 valid — 4 lost to a
     transient network blip): binary gain and fallback check.
- **Observed Metrics / Profiling Findings:**
  - **BINARY GAIN (gpu-binary-rest vs gpu-stock-ov, paired bootstrap):**
    | C | Ratio | 95% CI | Binary RPS | Stock RPS | Binary GPU% | Stock GPU% |
    |---|---|---|---|---|---|---|
    | 1 | 5.91× | [5.89, 6.09] | 23.6 | 4.0 | 54% | 13% |
    | 2 | 3.81× | [3.77, 3.83] | 30.2 | 7.9 | 78% | 28% |
    | 8 | 3.82× | [3.78, 3.86] | 30.2 | 7.9 | 78% | 32% |
    | 16 | 3.85× | [3.84, 3.87] | 30.2 | 7.8 | 78% | 34% |
  - **FALLBACK (gpu-binary-rest with JSON payload vs gpu-stock-ov):**
    All ratios 0.99-1.01 with CIs including 1.0 — zero regression on the
    JSON fallback path.
  - **CPU-ms/request:** binary 39-44 vs stock 199-242 (5.5× reduction).
  - **P50 @C1:** binary 42 ms vs stock 252 ms (6× improvement).
  - **Memory:** binary 885-998 MB vs stock 1010-1954 MB.
  - **Triton comparison:** binary REST 30.2 RPS @ 78% GPU = 86% of Triton's
    35.1 RPS @ 97% GPU. The remaining gap is device-side (more workers or
    further serving-path optimization could close it).
- **Numerical Parity Status:** Pass — bit-exact outputs verified locally
  (JSON and binary paths produce identical values for the same input).
- **Decisions & Next Steps:**
  1. **Binary tensor extension PASSES all gates** — commit c596dde6 on
     branch rhoaieng-102092 stands as the first accepted optimization.
  2. Next candidates: IPC serialization (pickle → msgspec framing),
     per-hardware-class guidance for rest_worker_json.
  3. The 3.8× REST improvement + zero-regression fallback + bit-exact
     parity makes this the highest-impact change of the engagement.

### P2-X03: Zero-Copy IPC (msgpack) — v1 Benchmark Failure, Root Cause, and Fix (Phase 3.3)
- **Sub-task ID:** 3.3 (zero-copy IPC serialization)
- **Git Commit / Working Tree State:** MLServer branch `rhoaieng-102092` @ `f5e19443` (amended from `24ee4788`; 9 files, +821/−14); candidate image `quay.io/syedali/mlserver-cuda:rhoaieng-102092-full` @ `sha256:1b03ff83f161aabbe6e1422c890d63df7d95c7c53c89c19cc83462e7934b17f1`.
- **v1 Benchmark Failure (session `gpu-msgpack-ipc`, archived):** all 15 `gpu-binary-msgpack` trials INVALID — the hook's REST correctness probe returned HTTP 500 on every request (15/15 `gpu-binary-zmq-pickle` control trials were valid). Server traceback: `orjson.JSONDecodeError: Input must be bytes, bytearray, memoryview, or str` at `zmq.py:_receive` → the msgpack parent-receive path fed `zmq.Frame` objects straight into `orjson.loads`.
- **Root-Cause Investigation (three independent failure layers):**
  1. **The "local validation passed" claim in `24ee4788` was FALSE — the msgpack server never ran locally.** The local poetry env lacked the optional `zmq` extra (pyzmq), so `mlserver start` crashed at pool init (`ModuleNotFoundError: No module named 'zmq'` — see `/tmp/opencode/mlserver-msgpack.log`). The health check and all 5 "validation" tests were answered by one of 15 stale mlserver processes left from earlier runs (port 18080 held by a Nov10 binary-tensor validation server). The pre-commit cleanup used `grep "mlserver start"`, which never matched the actual cmdline (`...bin/mlserver', 'start', ...`) — a silent no-op. The 1299-test suite passed because all five `tests/parallel/test_zmq*.py` files `importorskip("zmq")` — they were silently skipped.
  2. **Bug 1 (parent receive):** `recv_multipart(copy=False)` returns `zmq.Frame` objects; decode must use their `.buffer` memoryviews. Hit on every response → HTTP 500 for every request.
  3. **Bug 2 (worker receive):** the worker used single-frame `recv()` for multipart messages — with the benchmark payload (3.15 MB tensor → separate frame) the worker would only read the metadata frame: `IndexError` in decode plus permanent ZMQ stream corruption from unread frames. Latent in v1 only because the hook's correctness probe sends a JSON request whose tensors stay as Python lists until the worker converts them (single metadata frame).
  4. **Serialization audit found three more gaps** (all would have corrupted specific paths): `bytes` payloads (raw REST worker `method_args`/`RawReply.body`) fell through to `str()`; `ModelResponseMessage.exception` (`WorkerError` instances) fell through to `str()` → pydantic validation failure on decode → transport death on any model error; NumPy scalars fell through to `str()`.
- **Fix (commit `f5e19443`, amends the unpushed `24ee4788`):**
  - `zmq.py`: `_frame_buffers()` converts received Frames to zero-copy memoryviews; parent receive decodes from them; worker receive uses `recv_multipart()` when msgpack; `ParentTransport` carries msgpack as instance state plumbed from `InferencePool` through the response queue (replacing a module-level global — no cross-test contamination, no reliance on pytest file order); worker-side flag compared with `is True` so `Mock(spec=Worker)` test workers keep the pickle path.
  - `serialization.py`: `bytes`/`bytearray` payloads share the inline (<256 B hex)/frame split; `BaseException` values are rebuilt as real exception objects (original class via import path, `str()` preserved exactly — `WorkerError` messages round-trip byte-identically, `RuntimeError` fallback for unknown classes); dataclasses (e.g. `RawReply`) rebuilt via the `__class__` marker mechanism; NumPy scalars via `.item()`; decode accepts `memoryview` frames (probe-verified: `orjson.loads` and `np.frombuffer` accept them; Frame buffers decode to WRITABLE arrays — full parity with the pickle path).
  - `pool.py`: serialization choice propagated through the response queue + `logger.info("ZeroMQ inference IPC enabled (serialization: %s)")` for operational visibility (this log line is what makes msgpack mode positively verifiable in E2E).
- **Re-validation (every layer, this time with proof the msgpack path was exercised):**
  - pyzmq 27.2.0 installed from the already-declared `zmq` extra (no dependency change; env-only).
  - `tests/parallel/test_zmq.py::test_real_ipc_preserves_request_response_arrays_bytes_and_errors` parametrized over pickle/msgpack (real sockets, arrays + bytes + WorkerError round-trips) — the unit test that would have caught bug 1.
  - `tests/parallel/test_zmq_integration.py` — the full spawned-worker integration suite parametrized over both serializations + new `test_large_input_tensors_across_concurrent_requests` (8 × 16 KiB inputs, above the inline threshold, concurrent across 2 workers) — the integration test that would have caught bug 2.
  - `tests/parallel/test_serialization.py` extended: bytes round-trips, memoryview frames, WorkerError/CancelledError/unknown-class exceptions, RawReply dataclass, NumPy scalars (28 tests total).
  - Full suite: **1351 passed / 11 skipped** (zmq suites now actually run; was 1299/17). flake8/black/mypy clean on all changed files. `tests/cli/test_build.py` 6 failures and repo-wide black notebook drift verified PRE-EXISTING via `git stash` re-run on the clean tree (docker daemon + missing `GITHUB_*` credential + `docs/examples/*.ipynb` formatting; untouched by this change).
  - Local server E2E (source tree, fresh port ownership verified via `ss -ltnp` pid match, msgpack confirmed via the new log line): JSON + binary requests, bit-exact parity, 10 sequential + 8 concurrent 3.15 MB binary requests all correct. (Note: the bundle's own `settings.json` pins 127.0.0.1:18080 + `parallel_workers: 1` and overrides `MLSERVER_HTTP_PORT`/`MLSERVER_PARALLEL_WORKERS` env — local E2E effectively ran W1; W2 concurrency is covered by the integration tests.)
  - Container E2E on the EXACT cluster image digest `1b03ff83...` (podman, `--network=host --security-opt label=disable`, whole-bundle mount so the model's relative `uri: ../../../model/model.onnx` resolves): model ready in 6 s in both workers, all 5 tests passed, response sizes byte-identical to the source-tree run (542157 B JSON / 114679 B binary). Local-container learnings: SELinux Enforcing requires `label=disable` (or `:Z`) for host mounts; the image `Cmd` is `sh -c mlserver start $MLSERVER_MODELS_DIR`.
  - Environment repairs (local only, no repo changes): `poetry install --extras zmq` synced the env to the lockfile which downgraded protobuf 6.x → 5.28.3 and broke `onnx`'s 6.31.1 gencode (collection error in `tests/test_log_propagation.py`); restored `protobuf==6.31.1` via pip (mlserver's own 5.27.2 gencode is one-major-older = supported; this was the pre-install working state).
- **Numerical Parity Status:** Pass at every layer — unit round-trips, integration suites (both serializations), source-tree E2E, container E2E on the cluster image (bit-exact JSON↔binary parity; byte-identical response sizes vs the pickle path). In the v2 cluster session: identical REST response SHA-256 in all 30 trials.
- **v2 Benchmark Result (session `gpu-msgpack-ipc-v2`, 30/30 valid, control-first interleaved rounds, both variants on digest `1b03ff83...`, msgpack activation proven from trial receipts `env_applied`):**
  - **msgpack = 0.45× [CI95 0.45–0.46] vs pickle at every concurrency — REGRESSION, REJECTED.** pickle (zmq): 24.3 RPS C1 / 31.4 RPS C2+, GPU 82–83%, ~42 CPU-ms/req, C1 p50 41 ms. msgpack: 9.9 RPS C1 / 14.3 RPS C2+, GPU 38%, ~87 CPU-ms/req, C1 p50 101 ms. Spreads <2.2%; paired bootstrap on 15 pairs per concurrency.
  - **Root cause (micro-benchmark on this node, request/response replicas of the exact message shapes):**
    - REQUEST side is a msgpack WIN: 0.14 ms encode + 0.02 ms decode vs pickle 0.31 + 0.07 — the 3.15 MB binary-tensor input is a single ndarray frame with tiny metadata.
    - RESPONSE side is the disaster: `NumpyCodec.encode_output` converts output arrays to **Python lists** (dets 1,200 + labels 27,300 floats), and `_to_serializable` walks them element-by-element — **21.0 ms of pure Python recursion** producing a **550 KB JSON metadata frame** (vs pickle's 0.24 ms dumps + 0.58 ms loads). Decode adds 3.3 ms (another element-wise walk + pydantic re-validation). Cluster delta ≈ 45–60 CPU-ms/req including metadata allocation churn. The JSON request path is similarly hurt (hook correctness probe: 1.95 s vs pickle's 0.67 s — the 786K-float request list walks the same way).
  - **Why the experiment cannot reach a win even if fixed:** a list-framing fix (frame homogeneous numeric lists as array frames, restore via `tolist()`) would cut the response cost to ~1.5–2 ms — still ≥ pickle's 0.8 ms — while the request-side saving is only ~0.2 ms. Net expected ≈ 0.95–1.0×. **The IPC serialization is not a bottleneck for the binary-tensor workload: pickle's total IPC cost is ~1.2 CPU-ms/request.** The vLLM EngineCore pattern pays off for flat primitive messages; MLServer's pydantic object graphs already pickle at C speed.
  - **Stacking interaction (key insight):** the IPC optimization targeted the big-Python-list payloads of the JSON serving path — a cost the binary-tensor extension (P2-X02) had already eliminated for the optimized path. Optimization B's target was optimized away by optimization A.
- **v3 Benchmark Result (session `gpu-msgpack-ipc-v3`, 30/30 valid, enhanced build with numeric-list framing, digest `4f2d64d5...`, activation proven from receipts):**
  - **Enhanced msgpack = 1.00–1.02× vs pickle — parity at load, small CI-significant win at C1.** Ratios: C1 **1.023× [CI95 1.008–1.042]** (p50 40.2 vs 41.1 ms — the request-side zero-copy advantage on the serial path); C2–C16 1.001–1.004× with CIs including 1.0 (GPU-bound regime absorbs the difference). No p99 or memory regressions (933 vs 923 MB). GPU util 81–82% both variants at C2+ — same operating point.
  - Trade-off visible in the auxiliary metric: CPU-ms/request at C1 is 39.0 vs 36.1 (+2.9) — the framing's purity check/asarray add total CPU while the critical-path latency still improves; at C2+ CPU-ms is a wash (36.8–43.0 vs 37.2–41.3).
  - JSON path: correctness-probe latency 0.69 s vs pickle 0.65 s — parity (v2's 1.95 s penalty eliminated by framing, but no win).
  - **Gate reading:** "RPS or CPU-ms improvement with CI excluding no-effect" is met at C1 only (RPS), not at operating concurrency (C2+), and CPU-ms slightly worsens at C1. Strictly a pass by the letter; practically marginal (~900 lines of opt-in serializer for +2.3% at C1, zero at load).
  - Net arc across the three sessions: v1 broken (HTTP 500, two ZMQ bugs), v2 correct-but-2.2×-slower (list walking), v3 parity-with-C1-edge (framing). The IPC-ceiling analysis held: pickle's total IPC cost (~1.2 CPU-ms/req) bounds what any serializer can win here.
- **Enhancement iteration (user-directed, "enhance once, then decide"):**
  - Implemented **flat numeric-list framing** in `serialization.py`: homogeneous Python float/int/bool lists ≥64 elements are sent as raw buffer frames (wire dtypes `<f8`/`<i8`/`|b1`) and restored with `tolist()` — values and types reproduce exactly. Mixed-type, NaN-as-None, string, nested-outer and small lists keep the element-wise walk (parity guard: framing a mixed `[1, 2.0]` as float64 would change JSON bytes). Nested lists frame row-by-row (innermost flat level). Bonus: Python ints beyond int64 now round-trip via a `__bigint__` marker (orjson rejects >64-bit ints — a pre-existing msgpack-mode divergence from pickle, now fixed).
  - **Micro-benchmark after the fix (same replicas as above):** response encode **21.3 → 1.12 ms**, decode **3.3 → 0.63 ms**, response metadata **550 KB → 851 B** (frames: 9,600 + 218,400 B); JSON-request encode ~550 → **34.6 ms** (1,536 row frames). Round-trip exactness verified (values + `type(x) is float`).
  - **Net vs pickle per request: +0.77 CPU-ms** (request −0.12 ms, response +0.89 ms). Expected cluster outcome ≈ 0.97–1.00× — parity, no win, matching the IPC-ceiling analysis. 11 new unit tests (39 total in the serialization suite).
- **Decisions & Next Steps:**
  1. **DISPOSITION EXECUTED (user-approved): enhanced msgpack archived, not shipped.** Ship branch `rhoaieng-102092` reset to `c596dde6` (binary tensor only). Full lineage preserved on local branch `rhoaieng-102092-msgpack-archive` (`f5e19443` msgpack core + `f252cae6` numeric-list framing), with the complete v1→v3 evidence trail in this entry. Revival path: the ndarray-native response pipeline.
  2. **Item 3 closure — "smaller FP32 JSON payloads" evaluated (local evidence, no cluster run needed):** the reference 15.26 MB JSON payload's 786,432 floats are **all already FP32-exact** (0 non-exact), and FP32-trimmed re-serialization saves **0.0%** — orjson emits shortest-roundtrip repr (~18.5 chars/float). Within the JSON + FP32 contract no payload shrink exists; text encoding of FP32 values costs ~4.6× the 4 bytes/value binary form. The only smaller FP32 representation is binary tensor transport (3.15 MB, 4.8× smaller) — already evaluated and shipped (P2-X02, 3.8×). Cluster mini-experiment cancelled; queue reduced ~1.5 h.
  3. Note on running sessions: the gpu-binary-graph experiment (launched as a separate session uses digest `4f2d64d5` (contains archived msgpack code, inert under `serialization=pickle`); both arms run the same image so the paired single-variable design (graph flag) remains valid. For the soak test and final independent benchmark, the ship candidate is the P2-X02 image `rhoaieng-102092-binary` @ `sha256:883f1dbe` (built from exactly `c596dde6`) or a fresh rebuild from the reset branch.
  2. Process lesson recorded (from the v1 false-validation): a local server "validation" requires (a) startup verified from the server log, (b) port ownership verified against the launched PID, (c) the config positively confirmed in logs (the IPC log line added in `f5e19443`), and (d) relevant optional extras installed so the code paths actually run — the `importorskip`-silently-skipped zmq suites are now part of the local gate. The same discipline applied to the v2 session: msgpack activation proven from trial receipts before interpreting numbers.
  3. Soak test (60-min sustained load) queued next, on the binary-tensor candidate config.
  4. PR draft for the binary tensor extension remains ask-first; Jira milestone correction (comment 19449251 described IPC as validated — now a rejected experiment) queued, ask-first.

### P2-X04: CUDA Graph Replay on the Binary REST Path (Phase 3.4)
- **Sub-task ID:** 3.4 (buffer reuse / ticket item 4)
- **Git Commit / Working Tree State:** no code change — existing `cuda_graph` config from PR #356 validated on the binary path; ship branch `rhoaieng-102092` @ `c596dde6`; both arms on digest `4f2d64d5` (msgpack code inert under `serialization=pickle`).
- **Action Taken:** paired sweep `gpu-binary-graph` (30 trials, single variable `MLSERVER_MODEL_ONNX_CUDA_GRAPH=true`), replicating the P2-B02 gRPC-opt-in finding for the REST binary path. `CudaGraphSession` constraints verified compatible with RF-DETR (single fixed-shape FP32 input, distinct outputs).
- **Observed Metrics / Profiling Findings:**
  - **CUDA graphs = +0.3–0.6%** — C1 1.006× [CI95 1.004–1.008], C2 1.003× [0.999–1.007], C4 1.004× [1.003–1.006], C8 1.004× [1.004–1.005], C16 1.005× [1.005–1.005]. Absolute: 31.62–31.65 RPS (C2+) vs 31.49–31.53 control; C1 p50 41.0 vs 41.4 ms. No p99 regressions.
  - **Gate:** passes by the letter (CI excludes no-effect at 4/5 concurrencies) — but the effect is a free config flag, not code. **Disposition: recommend `cuda_graph=true` in the per-model config for fixed-shape FP32 single-input models** (zero cost, no regressions, small consistent gain). No branch change.
  - **KEY FINDING: the remaining Triton gap (31.5 vs 35.1 RPS, ~10%) is NOT kernel-launch overhead** — launch cost on this model+GPU is ≤0.6%. The ~15% GPU idle under our stack (vs Triton's ~3%) must live in the feeding pipeline (H2D/D2H transfers, sync patterns, response path) — attribution handed to the py-spy/memray session.
  - Data-quality note: 14/30 trials lost `receipts.json` to the recurring network blip (metrics complete, paired analysis clean — 0 invalid / 0 missing pairs; graph activation proven from surviving receipts: `CUDA_GRAPH=true` applied, model loaded CUDA-first). Tooling follow-up: `summarize.py load_cases` crashes on missing receipts — needs tolerance.
- **Numerical Parity Status:** Pass — identical REST response SHA-256 across all trials with intact receipts, including CUDA graph replay (consistent with P2-B02's Turing replay validation).
- **Decisions & Next Steps:** profiling session next (binary REST + gRPC opt-in py-spy, parent-side memray audit); graphs enter the recommended-config guidance for the final report.

### P2-X05: Phase-3 Profiling Attribution — gRPC deepcopy + Worker Response Path (ticket items 2 & 4)
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` @ `c596dde6`; prof image `rhoaieng-102092-prof` @ `sha256:310dcc50` (branch wheel + py-spy/viztracer/memray). Phase-2 profiles (gpu-prof-stock/w1json) recollected from PVC alongside today's.
- **Observed Metrics / Profiling Findings (per-request normalization: ~4,500 requests/150 s window; totals reconcile with the measured 38–42 CPU-ms/req envelope):**
  - **PROCESS SPLIT — binary REST (W1):** worker 64.2% / parent 35.8%. **gRPC opt-in (W2):** parent 71.9% / workers 22.1%.
  - **BINARY REST worker subtree:** `OnnxModel._get_model_outputs` 51.3% → `Session.run` (ORT, incl. H2D/GPU/D2H + sync) 51.2% ≈ 24 CPU-ms/req (matches the ~26 ms device time). **Worker non-ORT response work ≈ 6 CPU-ms/req (codecs `tolist` + `ForkingPickler.dumps` of the 28.5K-float list ≈ 3.6%+3.3% + encode). THE GPU-IDLE ARITHMETIC CLOSES: ~6 ms serial non-GPU work per ~30 ms worker cycle ≈ the measured 18% GPU idle (82% util). The path to Triton parity is shrinking the worker's post-inference serial leg.**
  - **BINARY REST parent (35.8% ≈ 16.7 CPU-ms/req):** h11/ASGI chain ~8 ms (framework body pass-through — httptools already ruled out), binary endpoints + `encode_binary_response` ~7 ms (incl. list→array re-conversion), IPC send+receive ~5–6 ms.
  - **gRPC PARENT — THE HEADLINE: `ModelInferResponseConverter.from_types` = 33.4% of ALL samples, of which `BaseModel.model_copy(deep=True)` → `deepcopy` chain = 33.2%** (converters.py:453: `outputs = [output.model_copy(deep=True) ...]` before `extract_raw`). Pydantic's deep copy walks all 28,500 output floats in Python per response — **~46% of the parent's gRPC work is a redundant copy** (the original response object is discarded after conversion). Plus `extract_raw`/`pack` 8.5%, servicer/`_InsertHeaders` ~6%, `ForkingPickler.dumps` ~9%.
  - **gRPC workers (22.1%):** `CudaGraphSession.run` visible with `update_inplace` + `run_with_iobinding` — graph path working as designed; `OnnxModel.predict` 10.4%.
- **Memray allocation audit (parent-side, trial valid at 31.3 RPS — overhead negligible):** **1.81M allocations / 193.7 GB cumulative / 191.7 MB peak over ~4,600 requests ≈ 42 MB allocated per request** (vs a 3.15 MB payload + 114 KB response — pure churn, not retention). Top allocators by size: `ForkingPickler.dumps` **50.5 GB (26%)** — IPC pickle churn (~11 MB/req transient, incl. buffer-doubling REALLOCs of the 28.5K-float response list); **four ~16.09 GB entries** ≈ the 3.15 MB request body materialized repeatedly through the receive path (h11 `handle_events` reception, starlette `Request.body` concatenation copy, dispatch-side references) — the body crosses ~4 allocations before inference. By count: 394 allocations/request — `ParentTransport._receive` 347K, h11 `_util` 181K, pydantic `model_validate` 112K. Confirms the "fewer copies" targets: IPC pickling churn and body-path copies; peak memory is healthy.
- **Conclusions:**
  1. **New gate-candidate: eliminate the redundant `model_copy(deep=True)` in the gRPC raw-output converter** — a measured 33% of gRPC serving samples; small, targeted fix (restructure `from_types`/`extract_raw` to avoid copying before packing). Expected: gRPC parent CPU −~⅓; RPS gain at the 70–83% GPU operating point to be determined by paired benchmark. *(→ executed as P2-X06, ACCEPTED.)*
  2. REST binary's remaining Triton gap is the worker's ~6 CPU-ms/req response leg (tolist + list pickling) — the ndarray-native response pipeline is the identified lever (kills the tolist AND shrinks the pickle); deferred as a larger change pending the deepcopy fix's outcome.
  3. Ticket item 2 now fully closed: gRPC path has frame-level attribution (the prior soft spot).
  4. **Cython/native-conversion verdict (ticket item 4, closed with evidence):** the binary-path attribution shows no MLServer-owned hot Python loop above ~2 CPU-ms/request — the remaining Python costs are (a) third-party framework code (h11/uvicorn body plumbing, ~8 CPU-ms), (b) data-structure costs solvable without compilation (tolist → ndarray-native pipeline; the msgpack list-walk was fixed 14× with numpy framing, not Cython), and (c) thin orchestration around C libraries (ORT, orjson, pydantic-core are already native). Compiling an extension would additionally impose the multi-arch (x86_64/ppc64le/aarch64) × py3.10–3.12 build matrix on Konflux. **Not justified at this workload; revisit only for a JSON-tensor-span parser if Track B of the phase-3 roadmap (fused JSON→tensor) proves insufficient.**
- **Tooling notes:** memray capture crashed on first attempt (memray exec needs absolute binary path — `/usr/local/bin/mlserver`); rerun in flight (parent-side only — workers are spawn-started). `summarize.py load_cases` needs tolerance for missing receipts. `oc cp` requires tar in-container — use `oc exec cat` for collection. py-spy SVG counts are comma-formatted ≥1,000 — parse accordingly.
- **Numerical Parity Status:** Pass — both py-spy trials valid with correct responses (profiler overhead ~5%: 29.9/29.7 RPS vs 31.5 unprofiled).

### P2-X06: gRPC Converter Deep-Copy Elimination — ACCEPTED (Phase 3.5)
- **Sub-task ID:** 3.5 (ticket items 2 & 4: profiling-attributed "fewer copies")
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` @ `2cbc6db2` (on `c596dde6`); fix image `rhoaieng-102092-grpcfix` @ `sha256:51967581…`; control image `rhoaieng-102092-binary` @ `sha256:883f1dbe…` (built from exactly `c596dde6`).
- **Action Taken:** removed the redundant `model_copy(deep=True)` + `extract_raw` from both `ModelInferRequestConverter.from_types` and `ModelInferResponseConverter.from_types` — the raw payload is now packed straight from the original objects (`include_contents=not use_raw` never reads the data field, so neither the copy nor the clearing was required). Zero copies, zero mutations, byte-identical wire output. New large-output regression test added; the existing FP16 double-conversion/caller-mutation guards pass unchanged.
- **Observed Metrics (paired sweep `gpu-grpc-deepcopy-fix`, 30/30 valid, gRPC opt-in config, byte-exact responses in every trial):**
  - **C1: 1.195× [CI95 1.180–1.209]** — 19.08 → 22.81 RPS, p50 52.3 → 43.7 ms (−8.6 ms ≈ the profiled ~11 CPU-ms deep-copy cost, profiler-adjusted). Matches the prediction band (+20–27% projected).
  - **C2–C16: parity** (0.999–1.003) — the GPU ceiling (31.2 RPS at ~82% util) fully absorbs the saving; the gRPC thread pool already overlaps conversion across concurrent requests.
  - No p99 regressions (262.8 → 261.9 ms at C8; 523.9 → 522.1 at C16).
- **Gate verdict: PASS** — CI-excluding RPS improvement at C1, parity elsewhere, no regressions. **This is the second accepted change on the ship branch** (binary tensor + gRPC converter fix), plus the `cuda_graph=true` config recommendation.
- **Tooling notes:** `summarize.py` crashed mid-session (JSON decode error — second robustness issue after `load_cases`); the paired analysis and validation were unaffected (COMPLETE, 0 invalid). The CPU-ms envelope was not joined into SUMMARY for this session — derivable from the sampler file if needed for the report. **Sweep overwrite guard:** `sweep.py` uses `mkdir(exist_ok=False)` — reusing a plan's session name against an existing pod-side output directory fails instantly (rc=1), and a run script that then copies the stale directory analyzes old data silently. Run scripts must clear or uniquely name output paths (hit once: the first final-benchmark attempt re-analyzed the round-1 session's data before detection).
- **Numerical Parity Status:** Pass — byte-exact gRPC responses vs the CUDA reference in all 30 trials; local cross-protocol parity (gRPC decoded outputs bit-exact vs REST) and byte-identical response repeats.


### Validation & Delivery

### P2-V01: Soak Test — 60-Minute Sustained Load, PASS (ticket item 7)
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` @ `2cbc6db2`; ship image `sha256:51967581…` (binary tensor + gRPC fix) with the recommended config (zmq, W1, binary tensor, `cuda_graph=true`).
- **Action Taken:** single 60-minute sustained trial at C8 (session `gpu-soak`), following the sweep discipline (fresh pod, readiness + parity gates, DCGM/cgroup sampling).
- **Observed Metrics:**
  - **113,483 / 113,483 requests — zero failures over the hour.**
  - **31.52 RPS sustained** — matches the short-trial sessions (31.4–31.6), no degradation.
  - Latency rock-stable: p50 253.8 ms, p90 255.9, p95 257.0, p99 258.0 (1.7% p50→p99 spread).
  - Memory flat: steady state ~1040 MB working set, +2.4 MB/hour trend (noise), PSI memory-pressure 0.00 for the entire run, trial peak 1109 MB. GPU 83.6% sustained, CPU 5.0%.
  - No pod restarts (single deployment generation; zero failed requests proves continuity).
- **Numerical Parity Status:** Pass — trial validity included response-size and correctness gates.
- **Tooling note:** the sampler wrote ~693 rows over the hour (~5.3 s effective interval vs the 3 s target — scrape skips); the drift analysis remains valid. GPU util field name differs from the ad-hoc reader's guess; trial-level value (83.6%) came from the summarizer.
- **Decisions & Next Steps:** final independent REST sweep of the ship candidate (C1–C16 × 3) next; gRPC final numbers already exist as the fix arm of `gpu-grpc-deepcopy-fix` (same image). Then archival (5.3), report finalization, PR draft (ask-first).

### P2-V02: Final Benchmark Campaign — 159 trials, 4 sessions, all gated (ticket final sweep)
- **Sub-task ID:** final campaign (the independent REST sweep of the ship candidate + batched/mixed coverage)
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` @ `8424bf32` (two commits: `0dd67502` binary tensor REST + `8424bf32` gRPC converter fix), tree clean. Post-campaign history edits (no code changes): commit messages reworded for concision, user-guide/reference docs for `rest_binary_tensor` added to the feature commit, and the branch rebased onto current midstream master `c39ea879` (RHOAIENG-89413, metrics/registry only — no overlap with our files). Final commits `0dd67502`/`8424bf32`; our changes' diffs are bit-identical to the benchmarked `e54e7a7a`/`ce586a7d`, and the full hermetic suite was re-verified green on the combined tree. Ship image `sha256:be187c53…` (tag `rhoaieng-102092-final`), digest verified in every session's receipts. Test gate: hermetic `tox -e mlserver-venv` (fresh venv from `poetry.lock` — pydantic 2.9.2, numpy 1.26.4), **100% green** excluding `tests/cli/test_build.py` (requires a real docker daemon; local socket is podman's shim which prefixes built tags with `localhost/` so docker-py inspects 404 — environmental, no code-path overlap with our commits).
- **Action Taken:** fully automated campaign (`benchmarking/campaign/run-final-campaign.sh`): 159 trials across 4 sessions — A: unbatched REST (60), B: unbatched gRPC (30), C: dynamic-batch model `max_batch_size=4` (60), D: mixed binary/JSON co-batching (9). Dynamic-batch RF-DETR exported (`dynamic_batch=True`, batch=2 inference verified), deployed to PVC between Sessions B and C (A/B kept single-model conditions), parity references captured once via `hook.py --capture-refs` and then compared bit-exact/byte-exact on every trial. Clean env per variant (strategic-merge deletes stale entries — no residue); every trial restarts a fresh pod, proves old pod gone, verifies CUDA providers, image digest, and response parity.
- **Observed Metrics (medians; full data in `final-campaign/RESULTS.md`):**
  - **REST binary vs stock: 4.00× (C2–C16; 6.24× at C1)** — 31.4 vs 7.8 RPS, p50 509 vs 2,013 ms @ C16, CPU 46 vs 229 ms/req, GPU 79.6% vs 32.4%. Paired CI95 tight; round spreads 0.1–2.8%.
  - **JSON fallback (binary ON, JSON requests): 1.05× parity** — no regression; one marginal flag: C4 p99 592.5 vs 562.1 ms (+5.4%, just over the 5% tolerance; all other concurrencies equal or better).
  - **CUDA graph on binary: +0.5%** — parity; remains a config recommendation only.
  - **gRPC opt-in vs stock: 3.94× (4.51× at C1)** — 31.1 vs 7.9 RPS; zero p99/memory regressions; CPU 16–20 vs 170–194 ms/req.
  - **Batched (C):** C1–C4 binary 1.08–1.23× over stock; mixed binary/JSON co-batching (D) at no cost vs pure JSON (8.26 vs 7.3–8.3 RPS @ C8). **BUT all batched variants — stock included — collapse at C8/C16:** RPS 8.3 → 1.7 (−79%), p50 240 ms → 4,600–9,200 ms, GPU 0–9% (idle while queued), round-to-round spreads 80–375%. Pre-existing adaptive-batcher queue behavior, independent of our changes (stock degrades identically) — see Phase-3 roadmap Track C.
- **Numerical Parity Status:** Pass — all 159 trials bit-exact REST / byte-exact gRPC against frozen references (original model references + newly captured batched-model references); zero failed requests.
- **Tooling notes (root-caused across 5 launches):** (1) `set -euo pipefail` killed the success path — `grep "^FAILED"` with no matches exits 1; (2) the client pod's `/tools` is root-owned/read-only for the non-root container user — tooling relocated to `/store/tools` (writable PVC); (3) this `oc` version requires `--` before exec commands; (4) `summarize.py` dropped genuine `gpu_util_mean == 0.0` measurements via a truthiness test — fixed with `is not None` (7 cells' means were silently skewed, e.g. `batched-grpc-raw@c16` 28.1 → true 9.4).
- **Decisions & Next Steps:** these are the final numbers for the report and PR. Everything persisted under `MLServer-resources` (data, scripts, models SHA-verified, references, payloads, infra manifests, image overlay) — the PVC/cluster hold nothing unique. Open: branch `rhoaieng-102092` still unpushed (push pending user approval); Phase-2 report + PR draft need these numbers filled in.

### P2-V03: Gate-8 completion — route qualification, before-control, J1 track
- **Sub-task ID:** Gate-8 evidence completion (roadmap comparisons 1, 3, 4 + route qualification)
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` @ `8424bf32` (implementation PR head); before-control on the official baseline image `quay.io/opendatahub/odh-mlserver-cuda@sha256:4c8168e9…`.
- **Action Taken:** 84 trials in two sessions on the frozen environment: E1 (gate8-before-j1, 48 trials — official image vs ship image, J0 vs J1) and E2 (gate8-route, 36 trials — OpenShift route vs direct endpoint, J0 vs binary). Every trial gated on bit-exact REST responses with per-trial receipts. gRPC route qualification attempted across passthrough, edge-TLS and edge-insecure configurations.
- **Observed Metrics (paired bootstrap ratios, CI95):**
  - **J0 before vs after (unchanged JSON clients):** parity at all concurrencies — 0.992–1.002× (no regression confirmed with fresh before-control).
  - **J1 before vs after:** parity — 0.991–1.015×.
  - **J0 vs J1 (client-serialization gain):** ~8% at C1 only (1.083–1.088×); parity at C2+ (frontend saturated regardless of body size — bottleneck is per-number parsing, not byte transfer).
  - **Route vs direct (binary):** parity at load — 0.997× C8, 0.999× C16. Route adds no measurable overhead.
  - **Binary vs JSON through the route:** 5.77× C1, 4.07× C8, 4.01× C16 — the binary extension is fully route-qualified (KSERVE_QUALIFIED).
  - **Routed gRPC:** unavailable — plaintext h2c backend; OpenShift routes require TLS passthrough or HTTP/2 backend forwarding. All configurations tested and documented as ROUTED_G_UNAVAILABLE with the failing-hop reproducer preserved.
- **Numerical Parity Status:** Pass — all 84 trials bit-exact REST, zero failed requests.
- **Decisions & Next Steps:** Gate-8 evidence matrix complete. Route qualification closes the deployment-claim gap for REST binary; gRPC documented as direct-only. Phase-2 report updated. Roadmap gate criteria fully satisfied.


### Administrative

### P2-A01: Binary-Extension Correctness Iteration and Commit Restructure
- **Sub-task ID:** 3.2 (binary tensor extension, correctness iteration)
- **Git Commit / Working Tree State:** ship branch `rhoaieng-102092` restructured to two commits: `7c192c1b` (KServe v2 binary tensor data extension for REST — supersedes `c596dde6`/`051d739a`; contains all nine review rounds' fixes) and `ca6f5087` (gRPC raw converter deep-copy elimination — supersedes `2cbc6db2`/`34aae368`). Tree-identity verified against the pre-restructure backup (`backup-pre-squash-2`, empty diff). Correctness-iteration details are documented in the PR description (`reports/gpu/PR_DRAFT_rhoaieng-102092.md`), not here.
- **Benchmarking-relevant behavioral changes from the iteration:**
  1. `DataPlane._prepare_payload` restores array-backed inputs to the public list representation before list-based batching **transport-independently** (previously gated on `grpc_raw_input_buffers` only) — eliminates the adaptive-batching breakage for binary REST requests; unbatched models keep compact arrays. The gRPC measured paths are unaffected (the settings gate was a subset of the new condition).
  2. Binary responses now resolve format **per output** (KServe negotiation) instead of all-or-nothing, and preserve model/CloudEvents headers.
  3. True zero-copy input: tensor arrays are memoryview views of the request body (`np.shares_memory` verified).
- **Validation state:** full suite **1350 passed / 11 skipped** (two gRPC batching tests updated to the transport-independent contract — reasoning documented in the test); real-model E2E: 7 negotiation paths bit-exact through actual RF-DETR inference; live parity gates unchanged. Post-fix image `sha256:063de227…` (tag `rhoaieng-102092-binary-v2`).
- **Re-benchmark (running):** session `gpu-binary-review-fixes` — pre-fix image (`51967581`) vs post-fix (`063de227`), single variable, every trial gated on bit-exact REST parity vs the frozen CUDA reference. Verifies the correctness iteration cost nothing in performance; the pre-fix numbers (P2-X02 + final sweep: 31.5 RPS @ C8) remain the record for the earlier implementation.

