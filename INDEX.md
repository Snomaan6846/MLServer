# Index — MLServer performance resources

This file is the source of truth for every artifact in this repository and
its local data tree. It documents what each file is, where it fits, and what
it references. If something is not listed here, it does not belong.

## Conventions

- **Tracked content** (`git ls-files`): tooling, scripts, manifests, reports
  and documentation. Version-controlled on the fork's `resources` branch.
- **Local data** (gitignored): benchmark outputs, session data, payloads,
  references, model binaries and archives. Never committed; durable copies
  of campaign archives also live on the `rfdetr-models` PVC.
- The MLServer implementation itself is a sibling checkout (`MLServer/`,
  branch `rhoaieng-102092`); this repository contains no server code.

## 1. Repository map

| Path | What it is |
| --- | --- |
| `INDEX.md` | This file — master index |
| `BENCHMARKING_HISTORY.md` | Study ledger: system specs, summary matrix, scope changes, execution log (entries P2-B01 through P3-V03), failure register, naming conventions |
| `BENCHMARKING_GUIDE.md` | Ramp-up guide: how to run benchmarking with this harness |
| `README.md` | Orientation and entry points |
| `benchmarking/` | All benchmarking tooling (tracked) |
| `assets/` | Everything needed to bootstrap a fresh cluster: model SHA contracts + settings, wire payloads, parity references, cluster state (tracked) |
| `results/phase2/` | Version-controlled evidence snapshot: distilled results and per-trial provenance receipts for every phase-2 session (tracked) |
| `results/onnx-thread-sweep/` | Version-controlled evidence snapshot for the ONNX thread-configuration campaign (tracked) |
| `results/worker-binary/` | Version-controlled evidence snapshot for the worker-binary feature campaign (tracked) |
| `results/mlp-campaign/` | Version-controlled evidence snapshot for the phase-1 MLP replication campaign (tracked) |
| `benchmark-results/` | Raw working data: per-request samples, telemetry streams, models, payloads, references, archives (gitignored, local) |
| `model-preparation/` | Model fixture preparation and export (tooling tracked, artifacts local) |
| `reports/` | Study reports and the final engineering report |
| `Makefile`, `pyproject.toml`, `poetry.lock`, `.flake8`, `LICENSE` | Repo hygiene: lint/test targets, dependency pins for the resource scripts, license |

## 2. Tracked content

### 2.1 Root

| File | Description |
| --- | --- |
| `README.md` | Orientation; points here, the guide, the ledger and the reports |
| `BENCHMARKING_HISTORY.md` | Ledger. Sections: *Abbreviations & Conventions* (naming patterns, campaign prefixes, wire tracks, modifiers), *System & Baseline Specs*, *Benchmark Summary Matrix*, *Incremental Execution Log* (P2-B01 through P3-V03), *Failure Register*, *Data Provenance Map* |
| `BENCHMARKING_GUIDE.md` | Ramp-up and operations guide for the benchmarking harness |
| `INDEX.md` | This file |

### 2.2 `benchmarking/campaign/` — campaign orchestration

| File | Description |
| --- | --- |
| `bootstrap-cluster.sh` | Fresh-cluster setup: namespace, PVCs, RBAC, client image build + deploy, model regeneration + upload, serving deployment, payload/reference/tooling upload |
| `export-models.py` | Exports both benchmark models in the pinned export environment and verifies them against `assets/MANIFEST.json` |
| `run-final-campaign.sh` | Phase-2 campaign orchestrator: pre-flight, hermetic test gate, model export, ship-image build/push, tooling sync, sessions A–D, consolidated results, PVC archival |
| `resume-campaign.sh` | Resume driver: re-checks prerequisites and runs a missing session plus results/archival |
| `write-variants.py` | Writes `benchmark-results/variants.json`: every phase-2 campaign variant consumed by the prepare hook |
| `write-plans.py` | Writes the four phase-2 session plans and paired-analysis specs |
| `build-gbatched.py` | Builds the batched-model gRPC wire file by swapping the protobuf `model_name` |
| `generate-results.py` | Consolidates all phase-2 sessions into `RESULTS.md` |
| `batched-model-settings.json` | Model settings for the dynamic-batch deployment |
| `write-worker-binary-plans.py` | Writes the worker-binary campaign variants and session plans (A1–A6) |
| `run-worker-binary-campaign.sh` | Worker-binary campaign orchestrator: reference refresh with cross-image stability proof, stale-sweep guard, model-state control, sessions A1–A6 |
| `write-mlp-plans.py` | Writes the phase-1 MLP replication campaign variants and session plans (B2–B6) |
| `run-mlp-campaign.sh` | MLP campaign orchestrator: single-model deployment, reference capture, sessions B2–B6, PVC restore |
| `distill-thread-sweep.py` | Distills the ONNX thread-sweep campaign into tracked evidence under `results/onnx-thread-sweep/` |

### 2.3 `benchmarking/cluster/` — in-cluster measurement tooling

| File | Description |
| --- | --- |
| `hook.py` | Sweep prepare hook: maps variant → deployment config, scales conflicting deployments to zero, patches image/env, forces a fresh pod (UID proof), waits model readiness, verifies providers, enforces REST bit-exact / gRPC byte-exact / binary tensor-payload parity. Configurable timeouts via `BENCHMARK_MODEL_READY_TIMEOUT` and `BENCHMARK_PARITY_TIMEOUT` |
| `triton_hook.py` | Same contract for the Triton reference deployment |
| `sampler.py` | Workstation-side utilization sampler: kubelet Summary API + DCGM, one JSON line per interval |
| `summarize.py` | Session post-processing: pooled latency percentiles, CPU-ms/request integration, GPU/memory window alignment |
| `validate_session.py` | Session completeness gate |
| `manifests/` | Creation manifests (apply-ready) |
| `captures/` | Live-state captures from the working cluster |

### 2.4 `benchmarking/study_tools/` — load generation and analysis

| File | Description |
| --- | --- |
| `sweep.py` | Session driver: reads a frozen JSON plan, runs prepare-hook + timed load per trial |
| `client.py` | Timed full-response load client (REST and raw gRPC) |
| `analyze.py` | Paired bootstrap analysis (20k resamples) |
| `payloads.py` | Builds the wire-payload corpus |
| `launch_server.py` | Starts a local serving instance |
| `preflight.py` | Pre-flight checks |

### 2.5 `benchmarking/overlay/` — image build recipes

| File | Description |
| --- | --- |
| `Dockerfile.cuda-binary` | Ship-image recipe: official CUDA base + MLServer wheel overlay |
| `Dockerfile.cuda-zmq`, `Dockerfile.cpu-zmq` | ZMQ-extra overlays |
| `Dockerfile.cuda-httptools` | httptools evaluation overlay (experiment, closed) |
| `Dockerfile.cuda-prof`, `Dockerfile.cuda-prof-branch` | Profiling overlays |

### 2.6 `reports/` — study reports

| File | Description |
| --- | --- |
| `reports/README.md` | Report index |
| `reports/FINAL_REPORT.md` | **The comprehensive engineering report** — all campaigns, all findings, configuration decision matrix, correctness verification, phase-1 alignment, appendices |
| `reports/gpu/PHASE2_PERFORMANCE_WORK.md` | Phase-2 report: baseline, accepted changes, closed experiments, profiling attribution, final campaign |
| `reports/gpu/PHASE3_ROADMAP.md` | Phase-3 enhancement roadmap |
| `reports/gpu/ONNX_THREAD_MATRIX.md` | Thread-configuration benchmark matrix (executed) |
| `reports/gpu/PR_DRAFT_rhoaieng-102092.md` | **Historical record** — pre-submission draft for PR #378 (merged) |
| `reports/gpu/GPU_PERFORMANCE_STUDY.md` | Predecessor GPU study (RTX 4070) |
| `reports/cpu/PERFORMANCE_REDESIGN.md` | Predecessor CPU study (REST/JSON, CPU ONNX, OVMS comparison) |

### 2.7 `results/` — versioned evidence snapshots

| Path | Description |
| --- | --- |
| `results/phase2/` | Phase-2 final campaign: distilled results, per-trial receipts, plans, paired-specs |
| `results/onnx-thread-sweep/` | ONNX thread-configuration campaign: per-session SUMMARY/metrics/receipts, variants, session log, cudnn/interop/affinity investigations |
| `results/worker-binary/` | Worker-binary feature campaign: per-session SUMMARY/metrics/receipts (A1–A6), variants, session log |
| `results/mlp-campaign/` | Phase-1 MLP replication campaign: per-session SUMMARY/metrics/receipts (B2–B6), variants, session log |

### 2.8 `assets/` — fresh-start kit

| Path | Description |
| --- | --- |
| `MANIFEST.json` | SHA-256 of every asset |
| `models/` | Model SHA contracts + model settings |
| `payloads/` | Essential wire payloads + corpus `MANIFEST.json` |
| `references/` | Every parity reference: original + batched, REST + gRPC, CUDA + CPU |
| `cluster-state/` | Cluster deployment/service captures, Triton config |

### 2.9 `model-preparation/`

| Path | Description |
| --- | --- |
| `prepare.py`, `serve.py`, `validate.py` | Model preparation, local serving, validation |
| `export-env/` | Pinned export environment |
| `REFERENCE_MANIFEST.json` | Reference SHA contracts |

## 3. Local data (gitignored)

| Path | Size | Description |
| --- | --- | --- |
| `benchmark-results/` | ~3.0 GB | Raw working data: per-request samples, telemetry, all campaign sessions |
| `.venv/` | ~3.6 GB | Python virtual environment for benchmarking scripts |
| `benchmarking/overlay/dist/` | ~6 MB | Built wheel artifacts (not tracked) |

## 4. External references

- PR #378: https://github.com/opendatahub-io/MLServer/pull/378 (merged)
- Jira: https://redhat.atlassian.net/browse/RHOAIENG-102092
- Predecessor studies: `reports/gpu/GPU_PERFORMANCE_STUDY.md`, `reports/cpu/PERFORMANCE_REDESIGN.md`
