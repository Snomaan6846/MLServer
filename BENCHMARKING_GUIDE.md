# Benchmarking guide — ramp-up and operations

This guide gets a new contributor from zero to running trustworthy
benchmarks with the harness in this repository. It covers the concepts,
the moving parts, the standard runs, and the traps we already hit so you
do not have to hit them again.

The complete file inventory lives in [`INDEX.md`](INDEX.md); the evidence
for every past experiment lives in
[`BENCHMARKING_HISTORY.md`](BENCHMARKING_HISTORY.md).

## 0. Fresh start — one command

On an empty OpenShift project, everything needed is checked in. After
cloning this repository (plain clone — no LFS required; the models are
SHA-pinned and regenerate automatically on the first run):

```bash
bash benchmarking/campaign/bootstrap-cluster.sh
```

This creates the namespace, PVCs, RBAC, builds and deploys the client
pod, regenerates the models if absent (pinned export environment,
SHA-verified against `assets/MANIFEST.json` — GitHub LFS is unavailable
on public forks, so the models ship as pinned SHAs plus the exact
scripts that reproduce them), uploads them plus the Triton reference
repo onto the model PVC, starts the serving
deployment, waits for both models to be ready, and uploads the wire
payloads, parity references and tooling to the client pod. When it
prints `BOOTSTRAP COMPLETE`, the environment is identical to the one the
phase-2 campaign ran on. What remains is benchmark-specific: build the
image you want to measure, write variants/plans, and run the campaign
(sections 6–7).

To verify the assets on an already-bootstrapped cluster, compare
`/mnt/models` and `/store` contents against `assets/MANIFEST.json`.

## 1. What this harness measures

Steady-state inference performance of a model server (MLServer or Triton)
on an OpenShift cluster with a GPU node: throughput (RPS), latency
percentiles, CPU per request, GPU utilization and memory — with
**correctness enforced on every single trial** (bit-exact REST responses,
byte-exact gRPC responses against captured references).

Scope is prediction performance only; startup, model loading and shutdown
are excluded.

## 2. Core concepts

| Concept | What it is |
| --- | --- |
| **Variant** | A named serving configuration: container image + environment + target model. Defined in `variants.json`, applied by the prepare hook. All comparisons are variant-vs-variant with everything else held constant |
| **Plan** | A frozen JSON description of a session: ordered variants, concurrency levels, rounds, timing, payload spec. Written before the run, never edited during it |
| **Trial** | One (variant, concurrency, round) cell: fresh pod, warmup, 60 s measured load, per-request samples |
| **Prepare hook** | `benchmarking/cluster/hook.py` — runs before every trial: applies the variant to the deployment, forces a fresh pod (proves the old one is gone), waits readiness, verifies execution providers, and **gates on response parity**. Writes `receipts.json` recording everything it did |
| **Receipts** | Per-trial proof: image digest, pod UID, applied env, provider line, parity hashes. If a receipt is missing or wrong, the trial is invalid |
| **Sampler** | Workstation-side, read-only: kubelet CPU/memory + DCGM GPU numbers every few seconds, aligned to trial windows afterwards |
| **Gates** | A session is valid only if: every trial passed parity, zero failed requests, metrics complete (RPS/latency/CPU/GPU/memory), and the paired analysis CI excludes no-effect where a claim is made |

## 3. Components and data flow

```
write-variants.py ──▶ variants.json ──┐
write-plans.py ────▶ plans/*.json ────┤
                                      ▼
run-final-campaign.sh (orchestrator)  │  retry + auto re-login
  ├─ hermetic test gate (tox)         │
  ├─ model export + image build       │
  ├─ tooling sync → client pod        │
  └─ per session:                     │
       sweep.py (on client pod)       │
         └─ hook.py per trial ────────┘ (patches deployment, parity gate)
         └─ client.py (load, per-request validation)
       sampler.py (workstation, utilization)
       summarize.py → metrics per trial
       validate_session.py → completeness gate
       analyze.py → paired bootstrap ratios
generate-results.py → RESULTS.md
archive (tar + SHA-256) → PVC
```

## 4. Prerequisites

- **Checkouts**: sibling directories `MLServer` (implementation branch)
  and `MLServer-resources` (this repo). The relative paths in the poetry
  setup require this layout.
- **Environments**: root poetry env for the resource scripts
  (`poetry install`); the separate `model-preparation/export-env` for
  model export (kept isolated on purpose — its numpy pin matches
  MLServer's `poetry.lock`).
- **Cluster access**: `oc` logged into the OpenShift cluster with admin
  on the `mlserver-perf` namespace. Credentials for automated re-login
  (`CLUSTER_USER`/`CLUSTER_PASS`/`CLUSTER_URL`) live in `~/.bashrc`; the
  orchestrator sources them and never logs their values.
- **Tools**: podman (image builds), skopeo (digest inspection), a
  `python -m tox`-capable Python 3.12 (the MLServer test gate).
- **In-cluster**: the `mlserver-client` pod (with `/store`), the serving
  deployment, the `rfdetr-models` PVC and the `sweep-hook`
  ServiceAccount. Current-state manifests are in
  `benchmarking/cluster/manifests/`; original setup objects in
  `benchmark-results/infra/manifests/`.

## 5. One-time setup (fresh cluster)

Run the bootstrap script — it is the one-time setup:

```bash
bash benchmarking/campaign/bootstrap-cluster.sh
```

It applies the creation manifests in
`benchmarking/cluster/manifests/` (PVC, RBAC, client build/deploy,
serving deployment, model-loader pod), builds the client image from the
tracked Dockerfile and tooling, and populates both PVCs from the
checked-in `assets/` (models SHA-verified, wire payloads, parity
references, image map). The full payload corpus beyond the essential
set is regenerable with `study_tools/payloads.py`.

What the assets deliberately contain: both ONNX models' SHA contracts
and settings (the model binaries regenerate via
`campaign/export-models.py` in the pinned export env — byte-identical,
verified against `assets/MANIFEST.json`), the
Triton reference configuration, the server-wide PVC settings, the
essential wire payloads (`j0-0`, `j1-0`, `g-0`, `g-batched-0`,
`triton-binary-rest`, `input-0`), every parity reference, and
`images.json`. Raw per-request data and campaign archives are not
assets — they are outputs (see [`INDEX.md`](INDEX.md)).

## 6. Running a full campaign

```bash
bash benchmarking/campaign/run-final-campaign.sh
```

What it does, in order: pre-flight checks → hermetic test gate on the
implementation HEAD → model export → ship-image build/push → variants +
plans + tooling sync to the client pod → sessions (unbatched REST,
unbatched gRPC, batched-model deployment + reference capture, batched,
mixed) → consolidated `RESULTS.md` → archive to the PVC.

Expect several hours. Everything is logged under
`benchmark-results/final-campaign/logs/` (every launch attempt is
preserved). If a campaign aborts after some sessions completed, use
`resume-campaign.sh` — it re-checks prerequisites and finishes the
remaining work without re-running completed sessions.

## 7. Running individual sweeps

For targeted experiments, drive the pieces directly:

1. Write or adapt a plan (see `benchmarking/templates/sweep.example.json`
   and the plans in `benchmark-results/plans/`).
2. Ensure the variants you need are in `variants.json` on the client pod.
3. Run the sweep on the client pod:
   ```bash
   oc exec <client-pod> -- python3 /store/tools/sweep.py \
       --plan /store/plans/<plan>.json --output /store/sessions/<name>
   ```
4. Collect, summarize, validate:
   ```bash
   oc cp <client-pod>:/store/sessions/<name> benchmark-results/sessions/<name>
   python3 benchmarking/cluster/summarize.py \
       --session benchmark-results/sessions/<name> \
       --sampler benchmark-results/samples/<name>.jsonl
   python3 benchmarking/cluster/validate_session.py \
       --session benchmark-results/sessions/<name>
   ```
5. Paired analysis (one control/candidate pair per invocation):
   ```bash
   .venv/bin/python benchmarking/study_tools/analyze.py \
       --summary <session>/SUMMARY.json \
       --control <control-variant> --candidate <candidate-variant> \
       --output <session>/paired-<control>-vs-<candidate>.json
   ```

## 8. Acceptance gates

A performance claim requires all of:

- **Correctness**: bit-exact REST / byte-exact gRPC in every trial
  (enforced by the hook; check `receipts.json`).
- **Statistics**: paired bootstrap CI95 (20k resamples) excluding
  no-effect in the claimed direction; ≥3 trials per cell; round-to-round
  spread <3% (except known-unstable regimes — document them).
- **No regressions**: candidate p99 within 5% and memory within 10% of
  control at every concurrency; JSON-fallback parity when a feature is
  enabled but unused.
- **Reproducibility**: image digest recorded in receipts; a fresh pod per
  trial with UID proof; providers verified from server logs.

## 9. Extending the harness

- **New variant**: add an entry to `write-variants.py` (deployment,
  env, image, optional `model` key for model-scoped parity), re-upload
  `variants.json`, reference the variant in a plan.
- **New model**: export it (pin the environment like `export-env`),
  upload model + settings to the PVC through a server pod (the client
  pod does not mount `/mnt/models`), build a gRPC wire file with the
  right `model_name` if using gRPC, capture references once with
  `hook.py <variant> --capture-refs`, then benchmark.
- **New comparison**: add it to the session's paired spec and re-run
  `analyze.py` per pair; `generate-results.py` picks up the per-pair
  files.

## 10. Known traps (all of these bit us)

| Trap | Rule |
| --- | --- |
| The client pod's `/tools` is root-owned read-only | Sync tooling to `/store/tools/` (writable PVC) and invoke it from there |
| This `oc` version requires `--` before exec commands | Always `oc exec <pod> -- <command>` |
| The local docker socket is podman's shim | Image builds work, but docker-py cannot find results (tags get a `localhost/` prefix) — docker-dependent tests need a real daemon; use the tox gate for the suite |
| `set -euo pipefail` + `grep` with no matches | A no-match `grep` exits 1 and kills the script — guard with `\|\| true` inside pipelines |
| GPU utilization of `0.0` is a real measurement | Truthiness checks drop it; always compare `is not None` |
| The PVC mounts on the *server* deployment only | Model uploads go through a server pod via `oc exec … -- cat >` |
| Strategic-merge env patches keep old entries | The hook deletes stale env vars per patch; variants must set every knob explicitly |
| Sampler writes lag the kill signal | Stop the sampler, then collect; the summarizer tolerates sparse samples but validate completeness |

## 11. Reading the results

- `results/phase2/final-campaign/RESULTS.md` is the consolidated,
  version-controlled view; the per-session `SUMMARY.json`, metrics and
  per-trial `receipts.json` sit alongside it.
- The ledger (`BENCHMARKING_HISTORY.md`) is the narrative: every
  experiment, its evidence and its decision.
- Per-trial raw data: `benchmark-results/sessions/<name>/case-*/samples.jsonl`
  (per-request latencies) and the working session directories.
- Reports in `reports/` interpret the campaigns; the phase-2 report is
  the current summary of accepted work.
