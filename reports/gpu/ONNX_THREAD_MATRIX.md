# ONNX Thread Configuration Benchmark Matrix — Complete Plan

**Research goal:** determine the optimal ONNX Runtime session configuration
(intra-op threads, inter-op threads, spinning, affinity, cuDNN algo search)
and its interaction with MLServer worker count, across both REST and gRPC
protocols, on both GPU and CPU devices. This fills the thread-tuning gap in
the phase-2 campaign and directly addresses RHOAIENG-99489's objective:
"Investigate the interaction between MLServer concurrency and ONNX Runtime
intra-op/inter-op threading."

**Base image:** ship image (`rhoaieng-102092` @ `8424bf32`) — includes the
binary tensor extension and gRPC converter fix. All phase-2 changes in place.

**Hardware:** 28 vCPU (14 physical cores, HT), 1 Tesla T4 GPU, Guaranteed QoS.

**Baseline (current frozen config):** `intra_op=2, inter_op=1, spinning=off,
execution_mode=SEQUENTIAL` — every previous measurement used this.

**Wire tracks tested:**
- **B-REST**: binary REST (`MLSERVER_REST_BINARY_TENSOR=true`, zmq transport)
- **B-gRPC**: raw-buffer gRPC (`MLSERVER_GRPC_RAW_INPUT_BUFFERS=true`, zmq transport)
- **J-REST**: JSON REST (stock env — control for frontend-bound path)

---

## Complete ORT session-tuning coverage

| # | ORT knob | Values in this matrix | Block |
|---|---|---|---|
| 1 | `intra_op_num_threads` | 1, 2, 4, 8 | A, C |
| 2 | `inter_op_num_threads` | 1, 2 | E |
| 3 | `execution_mode` | SEQUENTIAL, PARALLEL | E |
| 4 | `allow_spinning` | off, on | B, F |
| 5 | `spin_duration_us` | unset (legacy), 1000 | G |
| 6 | `spin_backoff_max` | 1 (none), 8 | G |
| 7 | Thread affinity | OS-managed, pinned | H |
| 8 | `cudnn_conv_algo_search` | DEFAULT, EXHAUSTIVE | I |
| 9 | `graph_optimization_level` | ORT_ENABLE_ALL (fixed) | — already optimal |
| 10 | `enable_cpu_mem_arena` | True (fixed) | — already optimal |
| 11 | `enable_mem_pattern` | True (fixed) | — already optimal |
| 12 | `enable_mem_reuse` | True (fixed) | — already optimal |
| 13 | `use_per_session_threads` | True (fixed) | — correct for our use case |
| 14 | MLServer `parallel_workers` | 1, 2, 4 | A, C |
| 15 | Device (providers) | CUDA, CPU | all blocks |
| 16 | `cuda_graph` | false (fixed for sweep) | tested in phase 2 |
| 17 | `cuda_blocking_sync` | false (fixed for sweep) | tested in phase 2 |
| 18 | `run_options` | unset | — operational, not perf |

---

## Phase 1 — GPU thread sweep (both protocols)

**Question (Q1):** Does intra-op thread count affect the GPU path on either
protocol? Expected: no — device-bound at 80%+ GPU on both.

**Base config:** ship image, zmq transport, spinning=off, inter_op=1.
REST uses binary tensor; gRPC uses raw input buffers.

### Block A1 — GPU REST binary, T sweep

| Parameter | Value |
|---|---|
| Device | CUDA |
| Wire | Binary REST |
| Workers (W) | 1 |
| Concurrency | C8 |
| Rounds | 3 |
| Spinning | off |

| Variant | T | Trials |
|---|---|---|
| gpu-rest-t1 | 1 | 3 |
| gpu-rest-t2 | 2 (baseline) | 3 |
| gpu-rest-t4 | 4 | 3 |
| gpu-rest-t8 | 8 | 3 |

**12 trials.**

### Block A2 — GPU gRPC, T sweep

| Parameter | Value |
|---|---|
| Device | CUDA |
| Wire | gRPC (raw buffers) |
| Workers (W) | 2 (gRPC standard) |
| Concurrency | C8 |
| Rounds | 3 |
| Spinning | off |

| Variant | T | Trials |
|---|---|---|
| gpu-grpc-t1 | 1 | 3 |
| gpu-grpc-t2 | 2 (baseline) | 3 |
| gpu-grpc-t4 | 4 | 3 |
| gpu-grpc-t8 | 8 | 3 |

**12 trials.**

**Phase 1 total: 24 trials (~1 h).**

---

## Phase 2 — GPU spinning + cuDNN (REST binary)

**Question (Q2):** Does thread spinning or cuDNN exhaustive algo search
improve the GPU path?

### Block B1 — GPU spinning

| Parameter | Value |
|---|---|
| Device | CUDA |
| Wire | Binary REST |
| Workers (W) | 1 |
| T | 2 (baseline) |
| Concurrency | C1 and C8 |
| Rounds | 3 |

| Variant | Spinning | Concurrency | Trials |
|---|---|---|---|
| gpu-spin-off-c1 | off | C1 | 3 |
| gpu-spin-on-c1 | on | C1 | 3 |
| gpu-spin-off-c8 | off | C8 | 3 |
| gpu-spin-on-c8 | on | C8 | 3 |

**12 trials.**

### Block I — cuDNN algo search

| Parameter | Value |
|---|---|
| Device | CUDA |
| Wire | Binary REST |
| Workers (W) | 1 |
| T | 2 |
| Concurrency | C8 |
| Rounds | 3 |

| Variant | cudnn_conv_algo_search | Trials |
|---|---|---|
| gpu-cudnn-default | DEFAULT (baseline) | 3 |
| gpu-cudnn-exhaustive | EXHAUSTIVE | 3 |

**6 trials.** Note: EXHAUSTIVE benchmarks all conv algorithms at model
load time — expect slower startup (~30–60 s extra) but potentially faster
inference for conv-heavy models like RF-DETR.

**Phase 2 total: 18 trials (~45 min).**

---

## Phase 3 — CPU full factorial W×T (the main event)

**Question (Q3, Q4):** What is the optimal W×T for CPU inference? Is there
a W×T interaction (same total threads, different distribution)?

**Question (Q3b):** Does the optimal W×T differ between REST and gRPC?

### Block C1 — CPU REST binary, W×T factorial

| Parameter | Value |
|---|---|
| Device | CPU only |
| Wire | Binary REST |
| Concurrency | C8 |
| Rounds | 3 |
| Spinning | off |
| inter_op | 1 |

Full factorial:

| W \ T | T1 | T2 | T4 | T8 |
|---|---|---|---|---|
| **W1** | ✓ | ✓ | ✓ | ✓ |
| **W2** | ✓ | ✓ | ✓ | ✓ |
| **W4** | ✓ | ✓ | ✓ | ✓ |

**12 variants × 3 rounds = 36 trials.**

### Block C2 — CPU gRPC, W×T (reduced — key points only)

Run gRPC at the same W×T combinations as REST for cross-protocol
comparison, but only at the critical points:

| W \ T | T1 | T2 (baseline) | T-best | T8 |
|---|---|---|---|---|
| **W1** | — | ✓ | ✓ | ✓ |
| **W2** | — | ✓ | ✓ | — |
| **W4** | — | — | — | — |

Note: T-best is determined from Block C1 results (filled in after Phase 3
Block C1 completes). Assuming T4 or T8 wins.

**~6 variants × 3 rounds = 18 trials.**

**Phase 3 total: 54 trials (~2.5 h).**

**Equal-total-thread analysis cells (Q4):**

| Total threads | W1 | W2 | W4 |
|---|---|---|---|
| 2 | W1×T2 ✓ | W2×T1 ✓ | — |
| 4 | W1×T4 ✓ | W2×T2 ✓ | W4×T1 ✓ |
| 8 | W1×T8 ✓ | W2×T4 ✓ | W4×T2 ✓ |
| 16 | — | W2×T8 ✓ | W4×T4 ✓ |
| 32 | — | — | W4×T8 (oversubscribed) ✓ |

---

## Phase 4 — CPU refinements (REST binary at best W×T)

**Question (Q5):** Do spinning, inter-op parallelism, thread affinity, or
tuned spin parameters further improve the CPU optimal config?

**Prerequisite:** best W×T from Phase 3 Block C1.

### Block F — CPU spinning

| Variant | Spinning | Trials |
|---|---|---|
| cpu-spin-off | off (baseline) | 3 |
| cpu-spin-on | on | 3 |

**6 trials.**

### Block E — CPU inter-op parallelism

| Variant | execution_mode | inter_op | Trials |
|---|---|---|---|
| cpu-seq | SEQUENTIAL (baseline) | 1 | 3 |
| cpu-par | PARALLEL | 2 | 3 |

**6 trials.**

### Block G — CPU spin tuning (conditional: only if Block F shows spinning ON helps)

| Variant | spin_duration_us | spin_backoff_max | Trials |
|---|---|---|---|
| cpu-spin-default | unset (legacy) | 1 | 3 |
| cpu-spin-tuned | 1000 | 8 | 3 |

**6 trials.** ORT benchmarks: `spin_duration_us=1000` + `spin_backoff_max=8`
was "the most consistent best performer across models and thread counts."

### Block H — CPU thread affinity

| Variant | Affinity | Trials |
|---|---|---|
| cpu-affinity-os | OS-managed (baseline) | 3 |
| cpu-affinity-pinned | Pinned to cores 0–N | 3 |

**6 trials.** Pinning avoids thread migration overhead. On our
single-socket VM, expect 5–10% (less than the 20% ORT reports for
multi-NUMA, but still measurable).

**Phase 4 total: 24 trials (~1 h).**

---

## Phase 5 — Cross-protocol confirmation

**Question:** Do the findings from the REST sweep transfer to gRPC and JSON?

**Prerequisite:** optimal config from Phases 3–4.

### Block X1 — gRPC confirmation (CPU)

| Variant | Config | Trials |
|---|---|---|
| grpc-cpu-baseline | T2/W1 (current frozen) | 3 |
| grpc-cpu-optimal | Best W×T from Phase 3 | 3 |

**6 trials.**

### Block X2 — JSON REST control (CPU)

| Variant | Config | Trials |
|---|---|---|
| json-cpu-baseline | T2/W1 (current frozen) | 3 |
| json-cpu-optimal | Best W×T from Phase 3 | 3 |

**6 trials.** Expected: JSON REST is frontend-bound; thread tuning should
NOT help (the bottleneck is the single asyncio frontend, not ORT).

**Phase 5 total: 12 trials (~30 min).**

---

## Phase 6 — Latency check

**Question (Q6):** Does the throughput-optimal config hurt single-request
latency?

**Prerequisite:** optimal config from Phases 3–4.

### Block L — GPU latency

| Variant | Wire | Config | Trials |
|---|---|---|---|
| gpu-rest-c1-baseline | Binary REST | T2/W1 | 3 |
| gpu-rest-c1-optimal | Binary REST | Best from Phase 2 | 3 |
| gpu-grpc-c1-baseline | gRPC | T2/W2 | 3 |
| gpu-grpc-c1-optimal | gRPC | Best from Phase 2 | 3 |

**12 trials.**

### Block L2 — CPU latency

| Variant | Wire | Config | Trials |
|---|---|---|---|
| cpu-rest-c1-baseline | Binary REST | T2/W1 | 3 |
| cpu-rest-c1-optimal | Binary REST | Best from Phase 3 | 3 |

**6 trials.**

**Phase 6 total: 18 trials (~45 min).**

---

## Grand summary

| Phase | Blocks | Question | Trials | Est. time |
|---|---|---|---|---|
| 1 | A1, A2 | GPU T sweep (both protocols) | 24 | 1 h |
| 2 | B1, I | GPU spinning + cuDNN | 18 | 45 min |
| 3 | C1, C2 | CPU W×T factorial (both protocols) | 54 | 2.5 h |
| 4 | F, E, G, H | CPU refinements | 24 | 1 h |
| 5 | X1, X2 | Cross-protocol confirmation | 12 | 30 min |
| 6 | L, L2 | Latency check | 18 | 45 min |
| **Total** | **14 blocks** | | **~150** | **~6.5 h** |

---

## Expected outcome table (hypotheses to verify)

| Path | Wire | Current (T2/W1/spin off) | Hypothesized optimal | Expected gain |
|---|---|---|---|---|
| GPU binary REST | B-REST | 31.4 RPS | Same (T-insensitive) | 0% (device-bound) |
| GPU gRPC | B-gRPC | 31.1 RPS | Same (T-insensitive) | 0% (device-bound) |
| GPU + cuDNN EXHAUSTIVE | B-REST | 31.4 RPS | 33–35 RPS | +5–10% (conv algo choice) |
| CPU binary REST | B-REST | 1.7 RPS | 3–6 RPS (T8/W1/spin on) | +75–250% |
| CPU gRPC | B-gRPC | ~2.0 RPS | 4–7 RPS (same optimal) | +100–250% |
| CPU JSON REST | J-REST | ~1.4 RPS | ~1.4 RPS (frontend-bound) | 0% (bottleneck is frontend) |

## Decision tree

```
Phase 1 (GPU T sweep)
├── Flat (expected) → T2 confirmed for GPU; proceed to Phase 2
└── T-dependent → investigate H2D/D2H copy parallelism; revise GPU model

Phase 2 (GPU spinning + cuDNN)
├── Spinning helps → include in optimal GPU config
├── cuDNN EXHAUSTIVE helps → config recommendation for conv models
└── Neither helps → GPU optimal = current T2/W1/spin-off

Phase 3 (CPU factorial)
├── W1×T_N wins → single-process recommendation (no IPC overhead)
├── W2×T_N/2 wins → balanced recommendation
└── W4×T_N/4 wins → multi-process recommendation (overlapping I/O)

Phase 4 (refinements)
├── Spinning helps → test tuned spin (Block G)
├── Inter-op helps → CNN has parallel branches (unexpected for RF-DETR)
└── Affinity helps → include core-pinning guidance

Phase 5 (cross-protocol)
├── gRPC confirms → recommendation is protocol-independent
└── JSON flat → confirms frontend is the JSON bottleneck (not ORT)

Phase 6 (latency)
├── No regression → ship the optimal config
└── Latency regression → tradeoff documented; per-use-case guidance
```

---

## Implementation details

### Thread settings via environment variables

MLServer's OnnxSettings reads env vars with the `MLSERVER_MODEL_ONNX_` prefix.
Thread parameters go in `session_options` (JSON dict):

```bash
MLSERVER_MODEL_ONNX_SESSION_OPTIONS='{"intra_op_num_threads": 4}'
```

Spinning goes in `session_config_entries`:

```bash
MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES='{"session.intra_op.allow_spinning": "1"}'
```

For spin tuning:

```bash
MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES='{"session.intra_op.allow_spinning": "1", "session.intra_op.spin_duration_us": "1000", "session.intra_op.spin_backoff_max": "8"}'
```

For inter-op parallelism:

```bash
MLSERVER_MODEL_ONNX_SESSION_OPTIONS='{"intra_op_num_threads": 4, "inter_op_num_threads": 2, "execution_mode": 1}'
```

(execution_mode: 0=SEQUENTIAL, 1=PARALLEL)

For cuDNN algo search (provider_options):

```bash
MLSERVER_MODEL_ONNX_PROVIDER_OPTIONS='{"cudnn_conv_algo_search": "EXHAUSTIVE"}'
```

### Thread affinity

Set via session config entries:

```bash
MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES='{"session.intra_op_thread_affinities": "1;2;3;4;5;6;7;8"}'
```

(Semicolon-separated core IDs; each thread pinned to one core.)

### CPU deployment

Use the existing `mlserver-rfdetr-cpu` deployment. The variant env must
include:
- `MLSERVER_MODEL_ONNX_PROVIDERS='["CPUExecutionProvider"]'`
- `MLSERVER_REST_BINARY_TENSOR=true` (for the binary REST path)
- The thread/spinning parameters above

### Model-settings note

The current model-settings.json on the PVC has `intra_op=2, inter_op=1,
spinning=off` baked in via `parameters.extra`. Env vars override
model-settings (the merge is env-first, then file). Verify per trial that
the env-var override took effect by checking the ORT session logs.
