# Cluster benchmarking tooling (RHOAIENG-102092)

Environment-specific tooling for measuring MLServer serving performance on
the OpenShift Tesla T4 cluster. These files are the versioned source of
record; the campaign orchestrator syncs them to the client pod's
`/store/tools/` (a writable PVC path — the image's `/tools` is root-owned
and read-only for the pod's non-root user).

- `hook.py` — sweep prepare hook (invoked by `study_tools/sweep.py` per
  trial): maps a variant name to a deployment configuration, restarts the
  serving instance with a fresh process, proves the old pod is gone, waits
  for model readiness, verifies active execution providers from server
  logs, and enforces REST (bit-exact) / raw-gRPC (byte-exact) response
  parity against captured references. Writes `receipts.json` per trial.
  Requires the `sweep-hook` ServiceAccount in the `mlserver-perf`
  namespace and `/store` (payloads, references, images.json).
- `sampler.py` — workstation-side, read-only utilization sampler: kubelet
  Summary API per-pod cumulative CPU / working set / PSI plus DCGM GPU
  util / FB memory / temperature / power / throttle reasons, all
  pod-attributed. Run alongside every sweep session.
- `summarize.py` — session post-processing: pooled percentiles
  (p50/p90/p95/p99) from raw client samples, CPU-ms/request via
  cumulative-counter integration across each trial window, GPU%/memory
  alignment, and per-variant/concurrency aggregation with RPS spread.

Environment constants baked into these tools (namespace `mlserver-perf`,
node `syedali-cluster-pool-j6qh5-gpu-t4-a-8kpk4`, services
`mlserver-rfdetr{,-cpu}`) match the current-state manifests stored in
[`manifests/`](manifests/); the original setup objects are preserved in
`benchmark-results/infra/manifests/`.
