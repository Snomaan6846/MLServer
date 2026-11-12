# Phase 2 results — versioned evidence

Version-controlled evidence snapshot for the RHOAIENG-102092 phase-2 work
(RF-DETR Small on a Tesla T4). Everything a reviewer needs to verify the
claims in [`../../reports/gpu/PHASE2_PERFORMANCE_WORK.md`](../../reports/gpu/PHASE2_PERFORMANCE_WORK.md)
and [`../../BENCHMARKING_HISTORY.md`](../../BENCHMARKING_HISTORY.md):

- `final-campaign/` — the definitive 159-trial campaign:
  - `RESULTS.md` — consolidated tables, paired bootstrap ratios with CI95,
    regression checks
  - `sessions/<name>/SUMMARY.json` — every trial's result (RPS, latency
    percentiles, validity)
  - `sessions/<name>/metrics-*.csv|json` — per-trial and per-cell metrics
    (CPU-ms/request, GPU%, memory)
  - `sessions/<name>/paired-*.json` — paired bootstrap analyses
  - `sessions/<name>/case-*/receipts.json` — per-trial provenance: image
    digest, pod UID, applied env, execution providers, parity hashes
  - `plans/`, `paired-specs/` — the frozen inputs that produced the session
  - `ship-image.txt` — the exact image digest benchmarked
  - `capture-batched-receipts.json` — the deliberate one-time capture of
    the batched-model parity references
- `sessions/<name>/` — the intermediate experiment sessions behind ledger
  entries 001–012, with the same per-session summary/metrics and
  per-trial receipts.

## What is deliberately not here

Raw per-request latency samples, utilization sampler streams, client
logs, model binaries, wire payloads, reference responses and campaign
archives. Those live in the local `benchmark-results/` working tree
(see [`../../INDEX.md`](../../INDEX.md)) and, for the final campaign, in
the `rhoaieng-102092-final-campaign-*.tar.gz` archive on the
`rfdetr-models` PVC. This snapshot contains the distilled evidence, not
the raw captures; the archive contains everything.

Future phases add sibling directories (`phase3/`, …) with the same
structure.
