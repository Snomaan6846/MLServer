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
| `BENCHMARKING_HISTORY.md` | Study ledger: system specs, summary matrix, scope changes, execution log (entries P2-B01 through P2-V03) |
| `BENCHMARKING_GUIDE.md` | Ramp-up guide: how to run benchmarking with this harness |
| `README.md` | Orientation and entry points |
| `benchmarking/` | All benchmarking tooling (tracked) |
| `assets/` | Everything needed to bootstrap a fresh cluster: model SHA contracts + settings, wire payloads, parity references, cluster state (tracked) |
| `results/phase2/` | Version-controlled evidence snapshot: distilled results and per-trial provenance receipts for every phase-2 session (tracked) |
| `benchmark-results/` | Raw working data: per-request samples, telemetry streams, models, payloads, references, archives (gitignored, local) |
| `model-preparation/` | Model fixture preparation and export (tooling tracked, artifacts local) |
| `reports/` | Study reports and the PR draft |
| `Makefile`, `pyproject.toml`, `poetry.lock`, `.flake8`, `LICENSE` | Repo hygiene: lint/test targets, dependency pins for the resource scripts, license |

## 2. Tracked content

### 2.1 Root

| File | Description |
| --- | --- |
| `README.md` | Orientation; points here, the guide, the ledger and the reports |
| `BENCHMARKING_HISTORY.md` | Ledger. Sections: *System & Baseline Specs* (cluster, node, images, model), *Benchmark Summary Matrix* (headline numbers), *Scope Changes* (user decisions), *Incremental Execution Log* (entries P2-B01 through P2-V03, one per experiment with evidence) |
| `BENCHMARKING_GUIDE.md` | Ramp-up and operations guide for the benchmarking harness |
| `INDEX.md` | This file |

### 2.2 `benchmarking/campaign/` — campaign orchestration

| File | Description |
| --- | --- |
| `bootstrap-cluster.sh` | Fresh-cluster setup: namespace, PVCs, RBAC, client image build + deploy, model regeneration (if absent) + upload (SHA-verified), serving deployment, payload/reference/tooling upload — ends with both models ready |
| `export-models.py` | Exports both benchmark models in the pinned export environment and verifies them against `assets/MANIFEST.json` (GitHub LFS is unavailable on public forks, so models ship as SHA contracts + reproduction scripts) |
| `run-final-campaign.sh` | Full campaign orchestrator: pre-flight, hermetic test gate (`tox -e mlserver-venv` from the MLServer `poetry.lock`), model export, ship-image build/push, tooling sync to the client pod, sessions A–D, batched-model deployment + reference capture, consolidated results, PVC archival. Retry with exponential backoff and automatic cluster re-login throughout |
| `resume-campaign.sh` | Resume driver: re-checks prerequisites and runs a missing session plus results/archival (used when a campaign aborts after completed sessions) |
| `write-variants.py` | Writes `benchmark-results/variants.json`: every campaign variant (deployment, env, image, model) consumed by the prepare hook |
| `write-plans.py` | Writes the four session plans and paired-analysis specs into `benchmark-results/final-campaign/` |
| `build-gbatched.py` | Builds the batched-model gRPC wire file (`g-batched-0.bin`) from the reference `g-0.bin` by swapping the protobuf `model_name` |
| `generate-results.py` | Consolidates all sessions into `RESULTS.md` (tables, paired ratios, regression checks, computed summary) |
| `batched-model-settings.json` | Model settings for the dynamic-batch deployment (`max_batch_size=4`, `max_batch_time=5ms`); minimal delta from the original model's settings |

### 2.3 `benchmarking/cluster/` — in-cluster measurement tooling

| File | Description |
| --- | --- |
| `hook.py` | Sweep prepare hook, invoked once per trial: maps variant → deployment config, scales conflicting deployments to zero, patches image/env (with stale-env deletion — no residue across variants), forces a fresh pod (UID proof), waits model readiness, verifies execution providers from server logs, and enforces REST bit-exact / gRPC byte-exact parity against captured references. Supports per-variant model targeting and a deliberate one-time `--capture-refs` mode. Writes `receipts.json` per trial |
| `triton_hook.py` | Same contract for the Triton reference deployment |
| `sampler.py` | Workstation-side read-only utilization sampler: kubelet Summary API (per-pod CPU, memory working set, PSI) + DCGM (GPU util, framebuffer, temperature, power), one JSON line per interval, pod-attributed |
| `summarize.py` | Session post-processing: pooled latency percentiles, CPU-ms/request integration from cumulative counters, GPU/memory window alignment, per-case CSV and per-cell aggregates. Treats `0.0` GPU utilization as a real measurement (never as missing) |
| `validate_session.py` | Session completeness gate: every trial must have RPS, latency percentiles, CPU-ms/request, GPU% and memory; exits nonzero on gaps |
| `manifests/` | Creation manifests (apply-ready): `pvc-rfdetr-models.yaml`, `sweep-rbac.yaml`, `client-build.yaml` (ImageStream + BuildConfig + client-store PVC), `client-deploy.yaml`, `mlserver-baseline.yaml` (GPU serving deployment + service), `mlserver-cpu.yaml`, `model-loader.yaml` (PVC upload pod), `gpu-provider-check.yaml` |
| `captures/` | Live-state captures from the working cluster (deployments, services, triton deployment) — reference for what ran, not for creation |
| `README.md` | Tooling notes and in-cluster requirements (ServiceAccount, `/store` layout) |

### 2.4 `benchmarking/study_tools/` — load generation and analysis

| File | Description |
| --- | --- |
| `sweep.py` | Session driver: reads a frozen JSON plan, runs prepare-hook + timed load per trial, enforces a synchronized start gate, writes per-case results and `SUMMARY.json` |
| `client.py` | Timed full-response load client (REST and raw gRPC), per-payload header overrides for mixed wire formats, per-request validation (status + minimum response size), warmup/measured phases |
| `analyze.py` | Paired bootstrap analysis (20k resamples) of control vs candidate RPS/latency/CPU ratios with CI95; one comparison per invocation; never converts missing trials into wins |
| `payloads.py` | Builds the wire-payload corpus (legacy JSON, short FP32 JSON, raw gRPC protobuf × 8 deterministic inputs) with SHA-256 manifest |
| `launch_server.py` | Starts a local serving instance with prepared CUDA library paths (workstation-side checks) |
| `preflight.py` | Pre-flight checks for workstation-side runs |

### 2.5 `benchmarking/overlay/` — image build recipes

| File | Description |
| --- | --- |
| `Dockerfile.cuda-binary` | Ship-image recipe: official CUDA base + MLServer wheel overlay (used for all campaign variants) |
| `Dockerfile.cuda-zmq`, `Dockerfile.cpu-zmq` | ZMQ-extra overlays |
| `Dockerfile.cuda-httptools` | httptools evaluation overlay (experiment, closed) |
| `Dockerfile.cuda-prof`, `Dockerfile.cuda-prof-branch` | Profiling overlays (py-spy/memray instrumented) |

### 2.6 `benchmarking/` — top level and legacy

| File | Description |
| --- | --- |
| `README.md` | Benchmarking entry point (points to the guide) |
| `inference.py`, `__init__.py` | Single-request benchmark client (predecessor of `study_tools/`; simple serving check) |
| `STUDY_GUIDE.rst` | Baseline study guide: timed sweeps, payloads, accounting, worker/thread tuning (predecessor workflows) |
| `PROFILING.rst` | Root/CPU/GIL/host-stage and CUDA/NVTX profiling instructions |
| `OPENSHIFT.rst` | OpenShift predictor and route measurements |
| `templates/sweep.example.json` | Example sweep plan |
| `tests/` | Unit tests for the resource scripts (`test_study_tools.py`, `test_inference.py`, `test_grpc.py`) — run via `make test` |
| `profiling/legacy-queueing-sources.tar.gz`, `profiling/SOURCE_MANIFEST.json` | Archived queueing-model sources with manifest |

### 2.7 `model-preparation/`

| File | Description |
| --- | --- |
| `README.md` | Preparation guide: downloads, export, fixture normalization |
| `prepare.py` | Downloads public RF-DETR Small assets, exports the pinned ONNX model, builds the normalized input fixture |
| `validate.py` | Correctness checks on the prepared fixture |
| `serve.py` | CUDA-environment helper for local serving |
| `REFERENCE_MANIFEST.json` | SHA-256 manifest of the prepared bundle (model, sample, references) |
| `tests/test_prepare.py` | Unit tests |
| `export-env/poetry.toml`, `export-env/poetry.lock` | Pinned export environment (rfdetr 1.11.2, CPU torch, numpy 1.26.4 matching MLServer's lock) |

### 2.8 `reports/`

| File | Description |
| --- | --- |
| `reports/README.md` | Report index with scope/status per report |
| `gpu/PHASE2_PERFORMANCE_WORK.md` | Phase-2 final report: baseline, accepted changes (binary tensor 4.0×, gRPC converter +19.5%), closed experiments, profiling attribution, final campaign (159/159 gated trials), remaining gap, durability map |
| `gpu/PHASE3_ROADMAP.md` | Enhancement roadmap; Track C1 documents the measured adaptive-batcher collapse at high concurrency |
| `gpu/PR_DRAFT_rhoaieng-102092.md` | PR description for `rhoaieng-102092` — ready to submit |
| `gpu/ONNX_THREAD_MATRIX.md` | Planned benchmark matrix for ONNX thread configuration (W×T×spinning×inter_op sweep on GPU and CPU paths) — fills the thread-tuning gap; feeds RHOAIENG-99489 |
| `gpu/GPU_PERFORMANCE_STUDY.md` | Predecessor study (RTX 4070; input representation, worker JSON, ZeroMQ, CUDA execution) |
| `cpu/PERFORMANCE_REDESIGN.md` | Predecessor CPU study (REST/JSON, CPU ONNX, OVMS comparison) |

### 2.9 `results/phase2/` — versioned evidence snapshot

| Path | Description |
| --- | --- |
| `README.md` | What the snapshot contains and what deliberately stays out |
| `final-campaign/RESULTS.md` | Consolidated campaign results (tables, paired ratios, regression checks) |
| `final-campaign/sessions/<name>/` | Per session: `SUMMARY.json` (every trial), `metrics-*.csv/json`, `paired-*.json`, and `case-*/receipts.json` (per-trial provenance: image digest, pod UID, env, providers, parity hashes) |
| `final-campaign/plans/`, `paired-specs/` | The frozen inputs that produced the sessions |
| `final-campaign/ship-image.txt` | The exact image digest benchmarked |
| `final-campaign/capture-batched-receipts.json` | Receipts of the one-time batched-reference capture |
| `sessions/<name>/` | The intermediate experiment sessions (ledger entries 001–012): `plan.json`, `SUMMARY.json`, `metrics-*`, per-trial `receipts.json` |

The raw captures behind this snapshot (per-request latency samples,
sampler streams, client logs, models, payloads, reference binaries,
archives) stay in `benchmark-results/` and the PVC archive.

### 2.10 `assets/` — fresh-start kit

| Path | Description |
| --- | --- |
| `MANIFEST.json` | SHA-256 of every asset — the bootstrap verifies uploads against it |
| `models/rfdetr-small/` | Original model SHA contract (`a018914e…`) + model settings; the binary regenerates via `campaign/export-models.py` and is gitignored locally |
| `models/rfdetr-small-batched/` | Dynamic-batch model SHA contract (`6d4e2e3b…`, `max_batch_size=4`) + model settings; regenerates the same way |
| `payloads/` | Essential wire payloads: `j0-0.bin` (legacy JSON), `j1-0.bin` (short FP32 JSON), `g-0.bin` (raw gRPC), `g-batched-0.bin` (batched-model gRPC), `triton-binary-rest.bin` (binary REST), `input-0.npy` (corpus source) + the corpus `MANIFEST.json`. The full 8-input corpus regenerates via `study_tools/payloads.py` |
| `references/` | Every parity reference: original + batched, REST + gRPC, CUDA + CPU |
| `cluster-state/` | Server-wide PVC `pvc-settings.json`, `images.json` (variant→image map), `triton-repo/` (Triton reference config; serves the same model bytes) |

## 3. Local data (gitignored)

Raw working data. The distilled, version-controlled evidence extracted
from these trees lives in `results/phase2/`; everything here stays local
(durable campaign copies also live on the `rfdetr-models` PVC).

### 3.1 `benchmark-results/` root

| Path | Description |
| --- | --- |
| `variants.json` | Current variant registry consumed by the prepare hook (written by `write-variants.py`) |
| `images.json` | Variant → container image mapping (official digests + overlay tags) for hooks without an explicit image |
| `payloads/` | Wire-payload corpus: `j0-*/j1-*/g-*` tracks × 8 inputs, `input-*.npy`, `triton-binary-rest.bin`, `g-batched-0.bin` source `g-0.bin`, `MANIFEST.json` with SHA-256 of every body |
| `references/` | Canonical per-device parity references: `cuda-rest-response.bin`, `cuda-grpc-response.bin`, `cpu-*` equivalents, plus SHA-256 sidecars. The batched-model references live in `final-campaign/references/` |
| `plans/` | Frozen sweep plans for the pre-campaign sessions (29 files) |
| `samples/` | Utilization sampler output per session (28 JSONL files) |
| `phase3-binary-tensor-design.md` | Design notes for the binary tensor extension (pre-implementation) |
| `jira-draft.md`, `jira-comment-19449251-adf.json` | Jira milestone comment draft and the posted ADF payload (baseline/profiling milestone) |
| `infra/scripts/` | Historical session runners and utilities (`run-gpu-sessions.sh`, `run-cpu-sessions.sh`, `run-paired-sessions.sh`, `capture_refs.py`, `rest_check.py`, `find_t4.sh`, `build_adf.py`) — superseded by `benchmarking/campaign/` |
| `infra/logs/` | Setup logs (model preparation, payload generation) |
| `profiles/` | Early flamegraph captures (`gpu-prof-stock.svg`, `gpu-prof-w1json.svg`) |
| `phase1-context/` | Source discussion threads that motivated the work |
| `phase3-research/notes.md` | Industry serving-engine comparison research notes |

### 3.2 `benchmark-results/sessions/` — experiment sessions

Every directory holds `SUMMARY.json`, `plan.json`, per-case `case-*/`
(`result.json`, `receipts.json`, `samples.jsonl`, `start.json`,
`prepare.log`, client logs) and the summarizer's `metrics-*.csv/json`.

| Session | Cases | Variants | Purpose |
| --- | --- | --- | --- |
| `validate-harness` | 1 | gpu-stock | Harness validation trial |
| `gpu-rest-stock` / `gpu-grpc-stock` | 15 | gpu-stock | GPU baseline (stock) |
| `cpu-rest-stock` / `cpu-grpc-stock` | 15 | cpu-stock | CPU baseline (stock) |
| `gpu-rest-paired` / `gpu-grpc-paired` | 30 | stock ↔ opt-in | Opt-in config vs stock, GPU |
| `cpu-rest-paired` / `cpu-grpc-paired` | 30 | stock ↔ opt-in | Opt-in config vs stock, CPU |
| `triton-rest` / `triton-grpc` / `triton-rest-binary` | 15 | triton-stock | Triton reference engine measurements |
| `gpu-rest-factorial` | 24 | stock + 8 fc-* | REST regression factorial: transport × worker count |
| `gpu-rest-stock-wsweep` | 12 | stock + sw-* | Stock worker-count sweep |
| `gpu-binary-wsweep` | 38 | binary w1/w2/w4 | Binary extension worker-count sweep |
| `gpu-binary-gain` | 28 | stock-ov ↔ binary-rest | Binary extension gain measurement |
| `gpu-binary-fallback` | 30 | stock-ov ↔ binary-rest | JSON fallback no-regression sweep |
| `gpu-httptools-paired` | 30 | stock-ov ↔ httptools | httptools evaluation (closed: no gain) |
| `gpu-msgpack-ipc` / `-v2` / `-v3` | 30 | pickle ↔ msgpack | Zero-copy IPC serialization arc (archived) |
| `gpu-binary-graph` | 16 | binary + graph | CUDA graph replay evaluation |
| `gpu-prof` / `gpu-prof3` / `gpu-prof3-memray` | 2/2/1 | prof variants | py-spy and memray profiling captures (flamegraphs under `gpu-prof3/profiles/`) |
| `gpu-grpc-deepcopy-fix` | 30 | optin-control ↔ optin-fix | gRPC converter deep-copy elimination measurement |
| `gpu-binary-review-fixes` | 30 | binary-prefix ↔ binary-ship | Review-fix re-benchmark (no performance cost) |
| `gpu-final-rest` | 15 | binary-ship | Final independent REST sweep of the ship candidate |
| `gpu-soak` | 1 | binary-ship | 60-minute sustained-load soak (PASS) |

### 3.3 `benchmark-results/final-campaign/` — the final campaign

| Path | Description |
| --- | --- |
| `RESULTS.md` | Consolidated results: per-session tables, paired bootstrap ratios with CI95, regression checks, computed summary |
| `sessions/final-unbatched-rest` | 60 trials — stock / binary / JSON-fallback / CUDA-graph (REST) |
| `sessions/final-unbatched-grpc` | 30 trials — stock / opt-in (gRPC) |
| `sessions/final-batched` | 60 trials — dynamic-batch model, four variants |
| `sessions/final-mixed` | 9 trials — alternating binary/JSON co-batching |
| `plans/` | The four session plans |
| `paired-specs/` | Paired-analysis comparison specs per session |
| `samples/` | Utilization sampler output per session |
| `references/` | Batched-model parity references (captured once, compared every trial) |
| `model-dynamic/` | Exported dynamic-batch ONNX model (batch=2 inference verified) |
| `logs/` | Campaign log (all launch attempts preserved: `campaign.log*`), per-step and per-session logs |
| `ship-image.txt` | Ship image digest (`quay.io/syedali/mlserver-cuda@sha256:…`) — verified in every session's receipts |
| `capture-batched-receipts.json` | Receipts of the deliberate one-time reference capture |
| `g-batched-0.bin` | Batched-model gRPC wire file (built by `build-gbatched.py`) |
| `rhoaieng-102092-final-campaign-*.tar.gz` + `.sha256` | Archive of the campaign (also uploaded to the PVC) |

### 3.4 `model-preparation/` local artifacts

| Path | Description |
| --- | --- |
| `rfdetr-small-original/` | Original fixed-batch model + settings, pulled from the PVC — SHA-256 verified `a018914e…` |
| `rfdetr-small-bundle/` | Prepared bundle (model, sample, references, provenance) per `REFERENCE_MANIFEST.json` |
| `export-env/.venv/` | Local export virtualenv (not tracked) |

## 4. External references

| What | Where |
| --- | --- |
| Implementation branch | `MLServer` sibling checkout, branch `rhoaieng-102092` (fork `Snomaan6846/MLServer`; commits `1aa92105` + `a87caa21` on master `c39ea879`) |
| Ship image | `quay.io/syedali/mlserver-cuda` tag `rhoaieng-102092-final`, digest in `final-campaign/ship-image.txt` |
| Official baseline image | `quay.io/opendatahub/odh-mlserver-cuda@sha256:4c8168e9…` |
| Original model SHA-256 | `a018914e11d589b8b51154f00ed1103f148a320a3960f768bd44f57d55d0ab51` |
| Cluster | OpenShift, context `default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user`, namespace `mlserver-perf`, GPU node `syedali-cluster-pool-j6qh5-gpu-t4-a-8kpk4` (Tesla T4) |
| Durable archives | `rfdetr-models` PVC, `/mnt/models/archives/` |
| Archived experiment lineage | Fork branch `rhoaieng-102092-msgpack-archive` |
| RF-DETR weights cache | `~/.roboflow/models/rf-detr-small.pth` (re-downloadable; rfdetr 1.11.2 pinned) |
