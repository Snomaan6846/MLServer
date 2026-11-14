# MLServer Performance Engineering Report

## RHOAIENG-102092: Comprehensive Benchmark Analysis & Configuration Guide

**Classification:** Engineering Report
**Status:** Final
**Total Trials:** 1,214 (1,184 valid, 97.5%)
**Sessions:** 63 across 6 campaigns
**Models:** RF-DETR Small (compute-bound, ~250ms GPU) and MLP-54 (frontend-bound, ~1.2ms CPU)
**Hardware:** NVIDIA Tesla T4 (16GB, CC 7.5) / 28 vCPU (GCP n1-highmem-32)
**Platform:** Red Hat OpenShift 4.21, MLServer (branch `rhoaieng-102092` @ `a772f6312bf8d961222f4f800170b794283ebbff`, merged as PR #378)

> **Trial accounting:** counts are from the per-session SUMMARY.json records
> (the authoritative log of every trial that produced a measurement).
> One harness-validation run and two aborted attempts (no measurement) are
> excluded. Case directories pruned from disk during storage cleanup remain
> counted through their SUMMARY.json entries.

---

## 1. Executive Summary

This report presents the consolidated findings from 1,214 gated benchmark trials
conducted across six measurement campaigns on a production-grade OpenShift
cluster. The work evaluates MLServer's serving performance across three
dimensions: **wire-format efficiency**, **processing architecture**, and
**runtime configuration**, using two complementary workload profiles that
bracket the practical serving spectrum.

### Headline Results

| Metric | Improvement | Context |
|---|---|---|
| REST throughput (binary tensor vs JSON) | **4.0×** | 7.9 → 31.5 RPS on Tesla T4 |
| gRPC throughput (opt-in vs stock) | **3.9×** | 7.9 → 31.1 RPS on Tesla T4 |
| CPU throughput (intra-op threads T2→T8) | **2.7×** | 1.98 → 5.36 RPS, config-only |
| CPU cost per request (worker-binary) | **−24.5%** | 46.0 → 34.7 CPU-ms/request |
| Single-request latency (worker-binary) | **−6.3% p50** | 41.2 → 38.6 ms at C1 |
| Frontend-bound throughput (MLP, worker-binary) | **+37%** | 775 → 1,064 RPS at C64 |
| Stability (60-min soak) | **100%** | 110,917/110,917 requests, zero failures |

### Shipped Changes (PR #378)

Three shipped changes across two commits on branch `rhoaieng-102092`,
each independently gated:

1. **REST Binary Tensor Extension** (`mlserver/rest/binary.py`): KServe v2
   binary tensor framing for REST — 3.15 MB request body instead of 15.26 MB
   JSON. 4.0× throughput at C8 with bit-exact outputs.

2. **gRPC Converter Deep-Copy Elimination** (`mlserver/grpc/converters.py`):
   Removes a per-response deep copy on the gRPC response path. +19.5% at C1
   in isolation (19.08 → 22.81 RPS).

3. **Worker-Side Binary Processing** (`rest_worker_binary` setting): Extends
   the existing `rest_worker_json` dispatch pattern to binary tensor frames.
   Both IPC crossings carry raw bytes; response encoding runs in parallel
   workers. −24.5% CPU-ms/request; +37% throughput for small-model serving.

### Top Configuration Recommendations

| Serving Profile | Recommended Configuration |
|---|---|
| GPU (compute-bound) | `rest_binary_tensor=true`, `parallel_inference_transport=zmq`, T2/W1 — **keep threads at default** |
| CPU (compute-bound) | `intra_op_num_threads=8`, `allow_spinning=1`, `rest_binary_tensor=true` — **3.0× over baseline** |
| CPU (frontend-bound, small model) | `rest_binary_tensor=true`, `rest_worker_binary=true`, `parallel_inference_transport=zmq` — **+37% over frontend parsing** |
| CPU (frontend-bound, large payload) | `rest_binary_tensor=true` (frontend parse), `parallel_inference_transport=zmq` — frontend wins above ~100 rows/request |

---

## 2. Methodology

### 2.1 Measurement Framework

Every trial in every campaign follows the same disciplined protocol:

- **Fresh pod per trial**: the serving deployment is patched and restarted;
  the previous pod's UID is proven gone before measurement begins.
- **Correctness gate**: every trial passes bit-exact REST (SHA-256 of parsed
  output arrays) and byte-exact gRPC (SHA-256 of response bytes) parity
  against frozen references captured from the official baseline image.
- **Binary wire gate**: binary tensor responses are verified
  tensor-payload-byte-identical (SHA-256 of the raw output bytes after the
  JSON header, with the per-request generated ID exempted) across all
  processing configurations.
- **Provenance receipts**: each trial records the image digest, pod UID,
  applied environment, execution providers verified from server logs, parity
  hashes, and model-ready timing.
- **Cross-image stability proof**: frozen references are re-captured on each
  new image and byte-compared against the originals; any difference halts
  the campaign. All frozen references (11 response files: 8 RF-DETR and
  3 MLP-54, covering REST, gRPC, and binary-REST paths) have been
  identical across every image build — transitively proven by every trial
  on every image passing parity against the same frozen references.
- **Paired design**: control and candidate variants alternate within each
  round (even rounds forward, odd rounds reversed) to control for temporal
  drift. 3 rounds per cell; medians reported.
- **Synchronized measurement**: 2 load clients with a ≤20ms start-gate;
  60-second warm-up followed by 60-second measured window.
- **Continuous telemetry**: pod CPU (cumulative core-ns), memory working set,
  PSI, GPU utilization, framebuffer, temperature, and power sampled at
  3-second intervals, aligned to each trial's measurement window.

### 2.2 Workload Models

| Property | RF-DETR Small | MLP-54 |
|---|---|---|
| Purpose | Compute-bound regime | Frontend-bound regime |
| Architecture | Object detector (DINO-DETR) | Dense network 54→1024→512→256→128→7 |
| Model size | 114 MB ONNX FP32 | 3.0 MB ONNX FP32 |
| Input shape | [1,3,512,512] FP32 | [N,54] FP32 (N=1/100/1000) |
| Output | dets [1,300,4] + labels [1,300,91] | output [N,7] |
| Inference time | ~250 ms (GPU) / ~195 ms (CPU T8) | ~1.2 ms (CPU) |
| Request body (JSON) | 15.26 MB | 1.1 KB / 104 KB / 1.04 MB |
| Request body (binary) | 3.15 MB | 216 B / 21.2 KB / 212 KB |
| Execution provider | CUDA (Tesla T4) / CPU | CPU only |

### 2.3 Wire-Format Nomenclature

Every variant name in this report follows a structured convention:

```
[campaign]-[wire-track]-[processing-location]-[modifiers]@[payload-spec]
```

**Example:** `mlp-bin-worker-w8-queue@r100` = MLP campaign, binary tensor
wire format, worker-side processing, 8 workers, queue transport, 100 rows
per request.

| Track | Description | Server Setting |
|---|---|---|
| J0 | Legacy JSON (15.26 MB for RF-DETR) | — |
| B-REST | KServe v2 binary tensor REST | `rest_binary_tensor=true` |
| B-REST-W | Binary REST with worker processing | + `rest_worker_binary=true` |
| J-W | JSON with worker processing | `rest_worker_json=true` |
| G | Stock gRPC (protobuf) | — |
| G-R | gRPC with raw input buffers | `grpc_raw_input_buffers=true` |

**Campaign prefixes:** `wb-` (worker-binary), `mlp-` (phase-1 replication),
`cpu-`/`gpu-` (thread sweeps), `head-` (phase-2 final), (none) = older.

**Modifiers:** `-wN` (workers), `-tN` (threads), `-queue` (transport),
`-graph` (CUDA graph), `-spin-on/off`, `-binoff/binon` (binary toggle),
`@rN` (rows per request).

---

## 3. Findings

### 3.1 Wire-Format Efficiency (GPU, Compute-Bound)

**Context:** On a Tesla T4, RF-DETR inference takes ~250ms, making the GPU
the bottleneck at concurrency ≥4. Wire-format improvements remove the
frontend processing overhead, allowing the serving stack to reach the
device ceiling.

#### 3.1.1 REST: Binary Tensor vs Stock JSON

| Configuration | C1 | C2 | C4 | C8 | C16 | CPU-ms/req (C1–C16) | GPU% (C8) | Mem (C8) |
|---|---|---|---|---|---|---|---|---|
| Stock JSON | 3.97 | 7.74 | 7.89 | 7.84 | 7.84 | 178-257 | 23.7 | 1,397 |
| **Binary tensor** | **24.44** | **31.46** | **31.46** | **31.47** | **31.44** | **31-46** | **79.7** | **953** |
| Improvement | 6.2× | 4.1× | 4.0× | **4.0×** | 4.0× | **−75%** | +237% | −32% |

**Key observations:**
- At C1, binary tensor provides 6.2× improvement — the single-request
  latency drops from 251ms to 41ms because the 15.26MB JSON body is
  replaced by a 3.15MB binary frame, eliminating Python float-list
  construction on the input side.
- At C4-C16, both tracks reach a steady state: stock JSON is
  frontend-bound (GPU only 24-32% utilized), binary tensor is
  device-bound (GPU 80-84%).
- Memory drops from ~1.4 GB to ~950 MB — the binary representation
  avoids materializing Python float lists.

#### 3.1.2 gRPC: Opt-in vs Stock

| Configuration | C1 | C8 | C16 | CPU-ms/req (C8) |
|---|---|---|---|---|
| Stock gRPC | 5.02 | 7.92 | 7.90 | 194.4 |
| **Opt-in (raw buffers + converter fix)** | **22.86** | **31.12** | **31.11** | **16.1** |
| Improvement | 4.5× | **3.9×** | 3.9× | **−92%** |

The gRPC opt-in path (raw FP32 input buffers + deep-copy elimination in
the response converters) achieves the same device-bound plateau as binary
REST. The converter fix alone contributed +19.5% at C1 (measured in
isolation: 19.08 → 22.81 RPS).

#### 3.1.3 Cross-Format Comparison at Device Saturation (C8)

| Track | RPS | p50 | p99 | CPU-ms/req | GPU% |
|---|---|---|---|---|---|
| Binary REST | 31.47 | 254ms | 260ms | 42.1 | 79.7 |
| gRPC opt-in | 31.12 | 257ms | 263ms | 16.1 | 81.8 |
| Stock JSON | 7.84 | 1,017ms | 1,196ms | 178.4 | 23.7 |
| Stock gRPC | 7.92 | 1,012ms | 1,070ms | 194.4 | 24.2 |

Both optimized tracks reach the Tesla T4's device ceiling (~31.5 RPS,
~82% GPU utilization). The remaining difference is within measurement
noise.

#### 3.1.4 JSON Fallback Compliance

With binary tensor enabled but clients still sending JSON (the fallback
path), throughput is 1.006-1.061× vs stock — no regression. The only
latency delta was p99 at C4 (562 → 593ms medians, +5.4%), which lies
within the round-to-round variation of that cell (per-round p99s spanned
548-598ms for the fallback and 548-568ms for stock). This proves
the feature is safe to enable unilaterally; JSON clients are unaffected.

---

### 3.2 Processing Architecture (Frontend vs Worker)

**Context:** MLServer can process requests in the frontend (single asyncio
process) or in parallel inference workers. The `rest_worker_json` and
`rest_worker_binary` settings control where parsing and response encoding
happen.

#### 3.2.1 GPU Serving (RF-DETR, Compute-Bound)

| Configuration | C1 RPS | C1 p50 | C8 RPS | C8 CPU-ms/req |
|---|---|---|---|---|
| Binary, frontend parse | 24.15 | 41.2ms | 31.59 | 46.0 |
| **Binary, worker parse** | **25.89** | **38.6ms** | 31.04 | **34.7** |
| Delta | **+7.2%** | **−6.3%** | −1.7% (device-bound plateau) | **−24.5%** |

**Finding:** Worker processing provides a **7.2% single-request latency
improvement** and a **24.5% CPU-cost reduction** on the compute-bound GPU
workload. At C4+ the GPU is the bottleneck (device-bound), so throughput
is unchanged (both tracks sit on the ~31.5 RPS device plateau). The CPU
reduction is the primary benefit: both IPC crossings
carry raw bytes instead of Python float lists, and response encoding runs
in parallel workers.

#### 3.2.2 CPU Serving (MLP, Frontend-Bound)

| Track | 1-row | 100-row | 1000-row |
|---|---|---|---|
| Stock JSON (frontend) | 807 | 411 | 56 |
| Worker JSON | 1,057 | 354 | 51 |
| Binary frontend | 775 | 522 | 83 |
| **Binary worker** | **1,064** | 466 | 84 |
| Stock gRPC | 1,178 | 402 | 57 |
| gRPC raw buffers | 1,075 | **544** | **85** |

**Key findings:**

1. **Small requests (1-row)**: Worker processing wins decisively —
   binary-worker (1,064) and worker-JSON (1,057) both achieve **+30-37%**
   over their frontend counterparts. The parse/encode offload from the
   frontend's single asyncio event loop is the mechanism.

2. **Medium requests (100-row)**: Frontend processing wins —
   binary-frontend (522) beats binary-worker (466) by 12%. At this
   payload size, the IPC cost of forwarding the full body to a worker
   outweighs the parallelism gain.

3. **Large requests (1000-row)**: All tracks converge to the compute
   bound (~51-85 RPS). Binary formats (83-85) consistently outperform
   JSON (51-57) by ~50%.

4. **Crossover point**: The worker-vs-frontend advantage flips between
   1-row and 100-row payloads. The crossover depends on the ratio of
   parse/encode cost to IPC cost.

#### 3.2.3 W-Scaling (Worker Count)

| Track | W1 | W2 | W4 | W8 | W12 |
|---|---|---|---|---|---|
| Stock JSON | 797 | 803 | 807 | 817 | 810 |
| Worker JSON | 1,069 | 1,052 | 1,057 | 1,062 | 1,029 |
| Binary worker | 1,051 | 1,063 | 1,064 | 1,058 | 1,078 |
| gRPC raw | 1,087 | 1,086 | 1,075 | 1,052 | 1,075 |

*Note: gRPC-raw W8 and W12 medians are from 2 rounds (the third attempt
aborted without producing a measurement); all other cells have 3 rounds.*

**Finding: All tracks are flat across W1-W12.** For sub-2ms models, even
a single worker keeps up with the frontend dispatch rate. The bottleneck
is the **frontend asyncio event loop's routing throughput**, not worker
processing capacity. Adding workers does not help for fast models.

This confirms the phase-1 study's core insight: *"adding inference
workers can improve the first [worker execution] without fixing the
second [frontend preparation]"* — with the nuance that on this hardware,
the dispatch itself saturates before worker count matters.

#### 3.2.4 Concurrency Saturation

| Track | C2 | C16 | C64 | C128 |
|---|---|---|---|---|
| Stock JSON | 735 | 822 | 807 | 816 |
| Worker JSON | 708 | 1,119 | 1,057 | 1,055 |
| Binary worker | 724 | 1,071 | 1,064 | 1,043 |
| gRPC raw | 818 | 1,084 | 1,075 | 1,075 |

All tracks saturate by C16. Below C16, throughput is concurrency-limited;
above it, the frontend dispatch ceiling applies. Practical implication:
**more than ~16 concurrent requests adds queueing without adding
throughput** for fast models.

---

### 3.3 Runtime Configuration (Threads, Spinning, cuDNN)

#### 3.3.1 GPU Thread Configuration

| Knob | Values Tested | Result |
|---|---|---|
| `intra_op_num_threads` | 1, 2, 4, 8 | **No effect** (31.43-31.53 RPS flat, both protocols) |
| Thread spinning | ON, OFF | **No effect** (identical at C1 and C8) |
| `cudnn_conv_algo_search` | HEURISTIC, EXHAUSTIVE | **No effect** (median 31.51 RPS over 4 gated EXHAUSTIVE trials, bit-exact) |
| CUDA graph replay | ON, OFF | **Parity** (1.002-1.007×; optional config, not required) |

**Conclusion:** GPU serving is entirely device-bound at ~82% utilization.
No thread-level or cuDNN tuning changes throughput. The frozen baseline
(T2, W1, spinning off) is optimal.

#### 3.3.2 CPU Thread Configuration

| Configuration | RPS (C8, binary REST) | vs T2 baseline |
|---|---|---|
| T1 | 1.06 | 0.54× |
| T2 (frozen baseline) | 1.98 | 1.00× |
| T4 | 3.51 | 1.77× |
| **T8** | **5.36** | **2.71×** |
| T8 + spinning ON | **6.00** | **3.03×** |

**W×T factorial (36 trials):** Worker count (W1/W2/W4) has no systematic
effect at any thread level (medians within ~3% at every T; the T-lever
moves throughput 5×). The T-dominant finding is unambiguous:
`intra_op_num_threads` is the only CPU thread lever. Scaling curve:
T1→T2 +87%, T2→T4 +77%, T4→T8 +53% (saturating).

**Thread spinning:** +10.9% (5.41 → 6.00 RPS) at W4×T8. Spin parameter
tuning (`spin_duration_us=1000`, `spin_backoff_max=8`) adds nothing
over ORT defaults. **Recommendation: enable spinning, keep defaults.**

#### 3.3.3 Inter-Op Execution Mode

- SEQUENTIAL (default): 5.36 RPS — identical to the W4×T8 plateau.
- PARALLEL: **unconfigurable** via MLServer's JSON settings (ORT enum
  fields cannot be expressed in JSON; the attempt raises TypeError at
  session creation). Also pointless for single-graph models like RF-DETR.

#### 3.3.4 Thread Affinity

- OS-managed (default): 5.37 RPS.
- Pinned (1-based, count=T−1): 5.42 RPS. **Parity** — within noise.
- **Recommendation:** keep OS-managed scheduling.

#### 3.3.5 Thread Impact on Latency (C1)

| Config | RPS at C1 | p50 |
|---|---|---|
| CPU W1×T2 | 1.96 | 513ms |
| CPU W1×T8 | 5.06 | 194ms |

Even at concurrency 1, thread count matters: **T8 gives 2.64× lower p50**
because a single inference parallelizes across the intra-op pool.

---

### 3.4 Infrastructure

#### 3.4.1 IPC Transport (Queue vs ZeroMQ)

| Track | zmq | queue | zmq advantage |
|---|---|---|---|
| Stock JSON (MLP) | 807 | 733 | +10% |
| Worker JSON (MLP) | 1,057 | 956 | +11% |
| Binary worker (MLP) | 1,064 | 967 | +10% |

ZeroMQ consistently provides **10-11%** throughput improvement over the
default multiprocessing queue across all processing locations.

#### 3.4.2 Route Qualification

| Path | C1 | C8 | C16 |
|---|---|---|---|
| Direct (cluster-internal) | 24.60 | 31.33 | 31.33 |
| OpenShift Route (edge TLS) | 20.83 | 31.41 | 31.37 |

Route overhead is visible only at C1 (20.83 vs 24.60, −15%); at C8+ the
route reaches the same device-bound plateau. The binary tensor extension
is fully route-qualified (KSERVE_QUALIFIED).

Routed gRPC is **unavailable** on this platform (h2c cleartext backend;
OpenShift routes require TLS or HTTP/2 backend forwarding). Documented
as ROUTED_G_UNAVAILABLE — not a binary-tensor issue.

#### 3.4.3 Cross-Image Output Stability

All frozen references (11 response files across RF-DETR and MLP-54) are
byte-identical across all image builds — the code changes do not alter any
model output. This is verified before every campaign (the campaign halts
if any reference changes).

---

### 3.5 Stability & Reliability

#### 3.5.1 Soak Test (Binary REST, Worker Processing)

| Metric | Value |
|---|---|
| Duration | 60 minutes |
| Requests attempted | 110,917 |
| Requests successful | 110,917 |
| Failure rate | 0% |
| Sustained throughput | 30.81 RPS |
| Memory (max working set) | 1,063 MB |

#### 3.5.2 Failure Register

Of 1,214 total trials, 30 were invalid. Every one has been root-caused:

| Category | Count | Attribution |
|---|---|---|
| cuDNN DEFAULT numerical drift | 3 | Provider-level algorithm difference; gate correctly rejected |
| Operator interference | 2 | Deployment reset/phase-3 start during orphaned sweep |
| Inter-op PARALLEL unconfigurable | 3 | ORT enum limitation; not a server defect |
| Pinned affinity format errors | 6 | Harness variant-definition bugs (fixed) |
| Experimental msgpack v1 bugs | 15 | Never shipped; code archived |
| Environment/rollout transients | 1 | One-off rollout stall |

**Zero failures attributed to the PR's code changes.**

---

## 4. Configuration Decision Matrix

### 4.1 GPU Serving (Compute-Bound Models)

```json
{
    "parallel_workers": 1,
    "rest_binary_tensor": true,
    "parallel_inference_transport": "zmq",
    "session_options": {
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1
    }
}
```

**Rationale:** The GPU is the bottleneck (~82% utilization at C8+).
Thread tuning, spinning, and cuDNN search have no effect. The wire-format
changes (binary REST 4.0×, gRPC opt-in 3.9×) are the only levers that
matter. Keep threads at default (T2); workers at 1.

### 4.2 CPU Serving (Compute-Bound Models)

```json
{
    "parallel_workers": 1,
    "rest_binary_tensor": true,
    "parallel_inference_transport": "zmq",
    "session_options": {
        "intra_op_num_threads": 8,
        "inter_op_num_threads": 1
    },
    "session_config_entries": {
        "session.intra_op.allow_spinning": "1"
    }
}
```

**Rationale:** `intra_op_num_threads=8` provides 2.71× over the T2
baseline. Spinning adds another 10.9%. Workers are irrelevant (proven
by the W×T factorial). Total improvement: **3.03×** (1.98 → 6.00 RPS).

### 4.3 CPU Serving (Frontend-Bound, Small Models)

```json
{
    "parallel_workers": 4,
    "rest_binary_tensor": true,
    "rest_worker_binary": true,
    "parallel_inference_transport": "zmq",
    "session_options": {
        "intra_op_num_threads": 2
    }
}
```

**Rationale:** For sub-2ms models with small requests, worker processing
provides +37% over frontend parsing. Binary wire format is essential.
Worker count beyond 1 adds nothing (W-flat), but 4 provides redundancy.
zmq adds +10% over queue.

### 4.4 CPU Serving (Frontend-Bound, Large Payloads)

```json
{
    "parallel_workers": 1,
    "rest_binary_tensor": true,
    "rest_worker_binary": false,
    "parallel_inference_transport": "zmq"
}
```

**Rationale:** For payloads above ~100 rows per request, frontend
processing wins (IPC cost of forwarding bodies to workers outweighs the
parallelism gain). Keep `rest_worker_binary=false`; the binary wire
format still provides the serialization advantage.

---

## 5. Correctness Verification Summary

| Gate | Method | Result |
|---|---|---|
| REST bit-exact | SHA-256 of parsed output arrays vs frozen references | ✅ every trial |
| gRPC byte-exact | SHA-256 of response bytes vs frozen references | ✅ every trial |
| Binary tensor payload | SHA-256 of raw output bytes (ID-exempt) | ✅ every binary trial |
| Cross-image stability | Frozen references re-captured and byte-compared | ✅ all 11 identical |
| JSON fallback | Binary ON, JSON requests → no regression vs stock | ✅ 1.006-1.061× |
| Adaptive batching | Binary requests to batched models → list restoration | ✅ verified with real AdaptiveBatcher |
| Worker reply guard | worker_pid check, malformed envelope rejection | ✅ unit-tested |
| Error priority | Malformed frame → 400 even when model lookup fails | ✅ unit-tested |
| Extension gating | Binary parsing gated on `rest_binary_tensor` | ✅ unit-tested |
| Nested BYTES outputs | Mixed binary/JSON negotiation with nested element form | ✅ unit-tested |
| `/v2` discovery | `binary_tensor_data` advertised when enabled | ✅ unit-tested |
| Hermetic test suite | `tox -e mlserver-venv` from `poetry.lock` | ✅ green — 1,509 passed (docker-dependent CLI build tests excluded locally; green in CI) |
| CI | GitHub Actions full matrix (3 Python versions × 3 tox envs) | ✅ green |

---

## 6. Phase-1 Study Alignment

The phase-1 CPU redesign study established the architectural principle:
*distribute expensive conversion work first, then tune communication and
execution around it.* This campaign validates and extends that principle:

| Phase-1 Finding | This Campaign's Confirmation |
|---|---|
| Frontend is the bottleneck for fast models | ✅ W-flat across all tracks; frontend dispatch is the ceiling |
| Worker-side JSON distributes parse/encode | ✅ +30-37% for small requests on MLP |
| ZeroMQ improves IPC | ✅ +10-11% consistently across all tracks |
| Worker count is a tuning choice, not a universal default | ✅ W1 sufficient for fast models; W4 for compute-bound |
| ORT thread tuning matters on CPU | ✅ T8 = 2.71×; threads are the dominant CPU lever |

**Extension beyond phase-1:** The `rest_worker_binary` feature extends
the worker-processing pattern to the binary wire format, providing the
same architectural benefit (parse/encode in parallel workers) with the
additional advantage of raw-byte IPC crossings (no Python float lists).

---

## Appendix A: Complete Campaign Inventory

| Campaign | Sessions | Trials | Valid | Purpose |
|---|---|---|---|---|
| Phase-2 Final | 4 | 159 | 159 | Binary tensor + gRPC validation |
| Gate-8 | 2 | 84 | 84 | Route qualification, before-control |
| Soak | 1 | 1 | 1 | 60-min stability |
| Thread Sweep | 18 | 153 | 139 | GPU/CPU thread configuration |
| Worker-Binary | 6 | 76 | 76 | Worker-processing feature |
| MLP (Phase-1 Replication) | 5 | 148 | 148 | Frontend-bound regime |
| Intermediate experiments | 27 | 593 | 577 | Profiling, msgpack, httptools, etc. |
| **Total** | **63** | **1,214** | **1,184** | |

*Excluded from the inventory: one harness-validation session (1 run) and
two aborted case attempts that produced no measurement. Intermediate
debug images used during review-fix verification and msgpack experiments
are recorded in the per-trial receipts.*

## Appendix B: Image Provenance

| Image | Digest | Built From | Used In |
|---|---|---|---|
| Official baseline | `sha256:4c8168e9955daad321eedcd65bbcd27c9f210cfd56f8317f3fa12b10324de3fe` | master @ `9f4cb29ae70880e34a47c8fccfe0d402e2762fb9` | Gate-8 before-control |
| Ship (phase-2) | `sha256:2bcd2f47ddb1347d823649ecf3b3d5b9b4cb4e417cd2aa6eafd5a11b19125b34` | `rhoaieng-102092` @ `8424bf32e46b268bd02ecf1570c5dfe4a4f3f9f7` | Final campaign, thread sweep |
| Ship (worker-binary) | `sha256:b6094ab1ae67ef09287e61e70e26ab412731551dd89c5790cc4898843800030c` | `rhoaieng-102092` @ `a772f6312bf8d961222f4f800170b794283ebbff` (branch tip; merged as PR #378) | Worker-binary campaign, MLP campaign |

*Additional intermediate build digests (review-fix verification, msgpack
experiments, CPU-image runs) are recorded in the per-trial provenance
receipts; every trial's image is pinned by digest.*

## Appendix C: Session-to-Finding Mapping

| Finding | Primary Sessions | Total Trials |
|---|---|---|
| Binary tensor 4.0× | final-unbatched-rest | 60 |
| gRPC opt-in 3.9× | final-unbatched-grpc, gpu-grpc-deepcopy-fix | 60 |
| JSON fallback safety | final-unbatched-rest (head-binary-json) | 15 |
| Worker-binary CPU reduction | wb-A1-core | 48 |
| Worker-binary C1 latency | wb-A1-core | 12 |
| Regression fix verification | wb-A2-jsonworker | 12 |
| Orthogonality (JSON unaffected) | wb-A3-orthogonality | 6 |
| Batched model fallback | wb-A4-batched | 6 |
| CUDA graph parity | wb-A5-graph, gpu-binary-graph | 33 |
| Soak stability | wb-A6-soak, gpu-soak | 2 |
| GPU T-insensitivity | phase1a, phase1b | 24 |
| GPU spinning no-effect | phase2a | 12 |
| cuDNN EXHAUSTIVE no-op | phase2b, phase2b2 | 9 |
| CPU T-dominance | phase3a | 36 |
| CPU gRPC thread scaling | phase3b, phase3b2 | 15 |
| CPU spinning +10.9% | phase4a | 6 |
| Inter-op unconfigurable | phase4b | 6 |
| Affinity parity | phase4c, phase4c2, phase4c3 | 15 |
| Spin tuning no-effect | phase4d | 6 |
| Cross-protocol | phase5a, phase5b | 12 |
| Latency at C1 | phase6a, phase6b | 12 |
| MLP frontend-bound patterns | mlp-B2-core | 54 |
| MLP W-scaling flat | mlp-B3-wsweep | 46 |
| MLP C-saturation | mlp-B4-csweep | 36 |
| MLP transport comparison | mlp-B5-transport | 9 |
| MLP thread check | mlp-B6-threads | 3 |
| Route qualification | gate8-route | 36 |
| Before-control parity | gate8-before-j1 | 48 |
