# Benchmarking

Two levels of measurement live here.

## Quick serving check

[`inference.py`](inference.py) sends a prepared V2 JSON request to an
already-running model and reports RPS, p50/p95/p99 and payload hashes.
Use the root Poetry environment:

```bash
poetry run python benchmarking/inference.py \
  --model rfdetr-small --request "$HOME/rfdetr-small-bundle/sample/request.json" \
  --port 18080 --requests 1000 --concurrency 16 --output rest-result.json
```

Add `--protocol grpc --port 18081` for raw-tensor gRPC. The client
performs 32 warm-up requests, then measures completed requests at the
chosen concurrency. `--host` defaults to `127.0.0.1`; the prepared
RF-DETR repository uses ports 18080/18081. Request errors fail the
command; validate numerical results separately before benchmarking.

When comparing configurations, hold the model, request, worker/thread
counts, client concurrency and hardware fixed; alternate configurations
for at least three rounds; check concurrency 1 as well; test REST and
gRPC separately; use representative payload sizes (the RF-DETR input is
3 MiB as FP32, about 15 MB as JSON).

This client measures serving throughput and client latency only — not
server CPU, GPU utilization or GIL waiting. Reserve CPU capacity for the
client, or run it on a separate machine, when checking a fast server.

## Measured study harness

Everything the ledger's numbers rest on:

- [`study_tools/`](study_tools/) — sweep driver, timed load clients with
  per-request validation, paired bootstrap analysis, payload generation
- [`cluster/`](cluster/README.md) — prepare hook (per-trial deployment,
  freshness and parity gates), utilization sampler, summarizer, session
  validator, cluster manifests
- [`campaign/`](campaign/) — full-campaign orchestrators and result
  generators

Start with [`../BENCHMARKING_GUIDE.md`](../BENCHMARKING_GUIDE.md); the
complete file inventory is in [`../INDEX.md`](../INDEX.md).

## Predecessor workflow guides

- [`STUDY_GUIDE.rst`](STUDY_GUIDE.rst) — timed sweeps, payloads,
  accounting, worker/thread tuning
- [`PROFILING.rst`](PROFILING.rst) — root, CPU/GIL, host-stage and
  CUDA/NVTX profiling
- [`OPENSHIFT.rst`](OPENSHIFT.rst) — OpenShift predictor and route
  measurements
