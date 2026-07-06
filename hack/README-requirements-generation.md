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
- include artifacts compatible with `x86_64`, `aarch64` and `ppc64le` platforms

## Configuration

Configuration lives in `hack/requirements-config.json`.

Current shape:

```json
{
  "variants": [
    {
      "name": "cpu",
      "dockerfile": "Dockerfile.konflux",
      "root_packages": [
        "mlserver",
        "mlserver-lightgbm",
        "mlserver-onnx",
        "mlserver-sklearn",
        "mlserver-xgboost"
      ]
    },
    {
      "name": "cuda",
      "dockerfile": "Dockerfile.cuda.konflux",
      "root_packages": [
        "mlserver",
        "mlserver-onnx[cuda]"
      ]
    }
  ]
}
```

- `variants`: list of output targets. Each variant defines:
  - `name`: suffix used in output file name (`requirements-<name>.txt`).
  - `dockerfile`: path from repo root used to discover the base image.
  - `root_packages`: packages to resolve from the variant's configured index.

## How the Script Works

`hack/generate-pinned-requirements.py` runs in two phases:

1. **Resolve dependencies**  
   Uses `pip install --dry-run --report ...` on root packages to discover exact `(name, version)` pairs from pip's JSON report.
2. **Collect platform artifacts + hashes**  
   Uses `pip download` for `x86_64`, `aarch64` and `ppc64le` platform groups in parallel, then computes SHA256 for downloaded artifacts and writes hash-pinned output.

Important behavior:

- Package names are normalized per PEP 503 rules for matching.
- The script keeps root packages first in output order, then appends remaining resolved packages.
- If an explicit index URL is not provided, it uses system pip config/env (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or `pip config get global.index-url`).
- For compatibility with base images, only Phase 1 relies on pip's JSON report; Phase 2 does not use `pip download --report`.
- Live pip output is streamed during execution, so long downloads are visible in real time.

## CI / GitHub Workflow

`.github/workflows/requirements.yml` (`Requirements Regeneration`) runs:

- on manual trigger (`workflow_dispatch`)
- every 12 hours (`0 */12 * * *`)

Execution rules:

- Manual runs are allowed for any branch via the required `branch` input.
- Scheduled runs execute only for `opendatahub-io/MLServer` and process `rhoai-staging` as the base branch.

Per variant in config, the workflow:

1. validates that the selected `BASE_BRANCH` exists in the target repository
2. checks out the selected `BASE_BRANCH` (`workflow_dispatch` input branch, or `rhoai-staging` for schedule)
3. sets up Python 3.12 and installs `podman`, `yq`, and `jq`
4. extracts the base image from the configured Dockerfile using:
   - `python hack/generate-pinned-requirements.py --print-base-image <dockerfile>`
5. runs the generator inside that base image container:
   - `python hack/generate-pinned-requirements.py --variant <name> -o requirements/requirements-<name>.txt`
6. fixes workspace ownership (`sudo chown -R "$USER:$USER" .`) after container execution
7. creates or updates a PR if files under `requirements/` change using branch `requirements/regenerate-<BASE_BRANCH>`
8. requests reviewers from the repository `OWNERS` file (`reviewers` list) only for `opendatahub-io/MLServer`

Registry login is required and uses secrets:

- `AIPCC_QUAY_USERNAME` / `AIPCC_QUAY_PASSWORD`
- `quay.io` registry

The workflow fails early if credentials are missing.

## Local Usage

### Print base image from Dockerfile

```bash
python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
python hack/generate-pinned-requirements.py --print-base-image Dockerfile.cuda.konflux
```

### Generate pinned requirements in current environment

```bash
python hack/generate-pinned-requirements.py --variant cpu -o requirements/requirements-cpu.txt
python hack/generate-pinned-requirements.py --variant cuda -o requirements/requirements-cuda.txt
```

### Generate pinned requirements with explicit index override

```bash
python hack/generate-pinned-requirements.py --variant cpu \
  -o requirements/requirements-cpu.txt \
  --index-url https://example.com/simple/
```

### Dry run (show pip commands only)

```bash
python hack/generate-pinned-requirements.py --variant cpu -o requirements/requirements-cpu.txt --dry-run
```

### Custom platform tags

`--platform` can be repeated. When used, each provided platform is treated as its own download group.

```bash
python hack/generate-pinned-requirements.py --variant cpu \
  -o requirements/requirements-cpu.txt \
  --platform manylinux2014_x86_64 \
  --platform manylinux2014_aarch64 \
  --platform manylinux2014_ppc64le
```

## Operational Notes

- Run generation inside the target runtime base image for each variant so pip resolves against the intended index and environment.
- Keep `requirements-config.json` and workflow behavior aligned when adding new variants.
- Generated files are expected under `requirements/` and are the only artifacts committed by the workflow.
