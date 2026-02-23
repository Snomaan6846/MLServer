# Requirements Generation

This directory contains various helper scripts; this document focuses on the tooling that generates pinned requirement files with SHA256 hashes for MLServer runtime variants.

The flow is driven by:

- `.github/workflows/requirements.yml`
- `hack/generate-pinned-requirements.py`
- `hack/requirements-config.json`

## What This Does

The process generates `requirements/requirements-<variant-name>.txt` files that:

- resolve the latest dependency graph for a set of root packages
- pin every resolved package to an exact version
- attach `--hash=sha256:...` entries for reproducible installs
- include artifacts compatible with both `x86_64` and `aarch64` platforms

## Configuration

Configuration lives in `hack/requirements-config.json`.

Current shape:

```json
{
  "root_packages": [
    "mlserver",
    "mlserver-lightgbm",
    "mlserver-sklearn",
    "mlserver-xgboost"
  ],
  "variants": [
    { "name": "cpu", "dockerfile": "Dockerfile.konflux" }
  ]
}
```

- `root_packages`: top-level packages to resolve from the variant's configured index.
- `variants`: list of output targets (for example `cpu`, `cuda`, `rocm`).
  - `name`: suffix used in output file name (`requirements-<name>.txt`).
  - `dockerfile`: path from repo root used to discover the base image.

## How the Script Works

`hack/generate-pinned-requirements.py` runs in two phases:

1. **Resolve dependencies**  
   Uses `pip download` on root packages to discover exact `(name, version)` pairs.
2. **Collect platform artifacts + hashes**  
   Downloads artifacts for both `x86_64` and `aarch64` platforms, then computes SHA256 for every artifact and writes hash-pinned output.

Important behavior:

- Package names are normalized per PEP 503 rules for matching.
- The script keeps root packages first in output order, then appends remaining resolved packages.
- If an explicit index URL is not provided, it uses system pip config/env (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or `pip config get global.index-url`).

## CI / GitHub Workflow

`.github/workflows/requirements.yml` runs:

- on manual trigger (`workflow_dispatch`)
- every 12 hours (`0 */12 * * *`)

Per variant in config, the workflow:

1. checks out branch `rhoai-staging`
2. sets up Python 3.12 and Podman
3. extracts the base image from the configured Dockerfile using:
   - `python hack/generate-pinned-requirements.py --print-base-image <dockerfile>`
4. runs the generator inside that base image container:
   - `python hack/generate-pinned-requirements.py -o requirements/requirements-<name>.txt`
5. creates a PR if files under `requirements/` change
6. requests reviewers from the repository `OWNERS` file (`reviewers` list)

Optional registry login is supported with secrets:

- `REGISTRY_USERNAME` / `REGISTRY_PASSWORD`
- or `QUAY_USERNAME` / `QUAY_PASSWORD`
- optional `REGISTRY` (defaults to `quay.io`)

## Local Usage

### Print base image from Dockerfile

```bash
python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
```

### Generate pinned requirements in current environment

```bash
python hack/generate-pinned-requirements.py -o requirements/requirements-cpu.txt
```

### Dry run (show pip commands only)

```bash
python hack/generate-pinned-requirements.py -o requirements/requirements-cpu.txt --dry-run
```

### Custom platform tags

`--platform` can be repeated. When used, each provided platform is treated as its own download group.

```bash
python hack/generate-pinned-requirements.py \
  -o requirements/requirements-cpu.txt \
  --platform manylinux2014_x86_64 \
  --platform manylinux2014_aarch64
```

## Operational Notes

- Run generation inside the target runtime base image for each variant so pip resolves against the intended index and environment.
- Keep `requirements-config.json` and workflow behavior aligned when adding new variants.
- Generated files are expected under `requirements/` and are the only artifacts committed by the workflow.
