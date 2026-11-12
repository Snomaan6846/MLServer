# MLServer performance resources

Benchmarking harness, model preparation and study reports for MLServer
serving-performance work (RHOAIENG-102092 and its predecessor studies).
This is the orphan `resources` branch: it contains no MLServer server code
and is independent of the implementation PR's history.

| Start here | For |
| --- | --- |
| [`INDEX.md`](INDEX.md) | Master index — every file in this repository and the local data tree, with descriptions and references |
| [`BENCHMARKING_GUIDE.md`](BENCHMARKING_GUIDE.md) | Ramp-up — concepts, setup, running campaigns and sweeps, acceptance gates, known traps |
| [`BENCHMARKING_HISTORY.md`](BENCHMARKING_HISTORY.md) | The study ledger — evidence and decisions for every experiment |
| [`reports/`](reports/README.md) | Phase-2 final report, phase-3 roadmap, PR draft, predecessor studies |

## Layout

| Directory | Contents |
| --- | --- |
| [`benchmarking/`](benchmarking/README.md) | All tooling: bootstrap + campaign orchestrators, cluster prepare-hook / sampler / summarizer / validator, load clients, paired analysis, image-build overlays |
| [`assets/`](assets/MANIFEST.json) | Fresh-start kit: model SHA contracts + settings, wire payloads, parity references, cluster state — one command rebuilds the environment (models regenerate from the pinned export env, SHA-verified) |
| [`results/phase2/`](results/phase2/README.md) | Version-controlled evidence snapshot: distilled results and per-trial provenance receipts for every phase-2 session |
| `benchmark-results/` | Raw working data: per-request samples, telemetry, models, payloads, references, archives (gitignored; durable campaign copies on the `rfdetr-models` PVC) |
| [`model-preparation/`](model-preparation/README.md) | RF-DETR Small fixture: public-asset download, pinned ONNX export, input normalization, correctness checks |
| [`reports/`](reports/README.md) | Measured studies and the implementation PR draft |

## Checkouts and environments

Use two sibling checkouts: the implementation in `MLServer` (branch
`rhoaieng-102092`) and this repository in `MLServer-resources`. The
directory named `MLServer` is required by the relative Poetry dependencies.

```bash
git clone --branch rhoaieng-102092 \
  git@github.com:Snomaan6846/MLServer.git MLServer
git clone --single-branch --branch resources \
  git@github.com:Snomaan6846/MLServer.git MLServer-resources
cd MLServer-resources

poetry env use python3.12
poetry install
```

The root Poetry environment installs the sibling MLServer and ONNX runtime
as editable dependencies, with PyZMQ, ONNX Runtime GPU and CUDA/cuDNN
packages. A compatible NVIDIA driver is needed to run CUDA inference; the
load client itself does not need a GPU. Keep the separate
[`model-preparation/export-env/`](model-preparation/export-env/) for model
export — its pins (including numpy, matched to MLServer's `poetry.lock`)
must not leak into other environments.

Cluster benchmarking additionally needs `oc` access to the `mlserver-perf`
namespace, podman and skopeo; see the guide's prerequisites section.

## Quick check versus measured study

- **Quick serving check**: `benchmarking/inference.py` sends a prepared
  request to an already-running model — see
  [benchmarking/README.md](benchmarking/README.md).
- **Measured study**: timed sweeps with per-trial correctness gates,
  utilization sampling and paired statistics — see
  [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md). This is how every
  number in the ledger and reports was produced.

## Utility checks

```bash
make lint
make test
```

These check the resource scripts and their loopback REST/gRPC clients.
They do not download the model, run a performance study or launch GPU
inference.
