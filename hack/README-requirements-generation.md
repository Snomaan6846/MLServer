# Requirements Generation

This directory contains various helper scripts; this document focuses on the tooling that generates pinned requirement files with SHA256 hashes for MLServer runtime variants.

The flow is driven by:

- `.github/workflows/requirements.yml`
- `hack/generate-pinned-requirements.py`
- `hack/requirements-config.json`

## What This Does

The process generates `requirements/requirements-<variant-name>.txt` files that:

- resolve the dependency graph for a set of version-pinned root packages
- pin every resolved package to an exact version
- attach `--hash=sha256:...` entries for reproducible installs
- include artifacts compatible with configured target architectures per variant

## Configuration

Configuration lives in `hack/requirements-config.json`.

Illustrative configuration (each branch defines its own values in
`hack/requirements-config.json`):

```json
{
  "variants": [
    {
      "name": "cpu",
      "dockerfile": "Dockerfile.konflux",
      "root_packages": [
        {"name": "mlserver", "version": "1.7.0.dev0"},
        {"name": "mlserver-lightgbm", "version": "1.7.0.dev0"},
        {"name": "mlserver-onnx", "version": "1.7.0.dev0", "extras": ["cpu"]},
        {"name": "mlserver-sklearn", "version": "1.7.0.dev0"},
        {"name": "mlserver-xgboost", "version": "1.7.0.dev0"}
      ],
      "platforms": ["x86_64", "aarch64", "ppc64le"],
      "download_timeout": 480
    },
    {
      "name": "cuda",
      "dockerfile": "Dockerfile.konflux.cuda",
      "root_packages": [
        {"name": "mlserver", "version": "1.7.0.dev0"},
        {"name": "mlserver-onnx", "version": "1.7.0.dev0", "extras": ["cuda"]}
      ],
      "platforms": ["x86_64", "aarch64"],
      "download_timeout": 720
    }
  ]
}
```

- `variants`: list of output targets. Each variant defines:
  - `name`: suffix used in output file name (`requirements-<name>.txt`).
  - `dockerfile`: path from repo root used to discover the base image.
  - `root_packages`: packages to resolve from the variant's configured index.
    Each entry is an object with:
    - `name` (required): bare package name without extras.
    - `version` (required): exact version to resolve (no `==` prefix).
    - `extras` (optional): list of pip extras (e.g. `["cpu"]`).
    Legacy bare string entries (e.g. `"mlserver"`) are also accepted for
    backward compatibility but will resolve to the latest version on the index.
  - `platforms`: (optional) list of target architectures for Phase 2 downloads.
    Accepted values: `x86_64`, `amd64`, `aarch64`, `arm64`, `ppc64le`, `s390x`.
    Aliases `amd64` and `arm64` resolve to `x86_64` and `aarch64` respectively.
    Defaults to `["x86_64"]` if omitted.
  - `download_timeout`: (optional) per-platform-group download timeout in seconds.
    Defaults to 480. CUDA variants should use 720 due to large wheel sizes.
    Overridable via the `--timeout` CLI flag.

### Version Sync

Root package versions in `requirements-config.json` are automatically updated
by `hack/update-version.sh` alongside `mlserver/version.py`,
`runtimes/*/pyproject.toml`, and `docs/conf.py`. On branches where the config
file does not exist (e.g. upstream forks), the version sync step is silently
skipped.

## How the Script Works

`hack/generate-pinned-requirements.py` runs in two phases:

1. **Resolve dependencies**
   Uses `pip install --dry-run --report ...` on root packages (with exact version
   pins from config) to discover exact `(name, version)` pairs from pip's JSON
   report.
2. **Collect platform artifacts + hashes**
   Uses `pip download` for each platform group (derived from the variant's `platforms` config) in parallel, then computes SHA256 for downloaded artifacts and writes hash-pinned output. Each platform group uses the configured `download_timeout`.

Important behavior:

- Package names are normalized per PEP 503 rules for matching.
- The script keeps root packages first in output order, then appends remaining resolved packages.
- If an explicit index URL is not provided, it uses system pip config/env (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or `pip config get global.index-url`).
- For compatibility with base images, only Phase 1 relies on pip's JSON report; Phase 2 does not use `pip download --report`.
- Live pip output is streamed during execution, so long downloads are visible in real time.
- If one platform group fails, all remaining groups still complete before the script reports failure (prevents race-condition temp-dir cleanup).

### Skip-Gracefully Behavior

After a version bump (e.g. `1.7.1+rhaiv.11` → `rhaiv.12`), the AIPCC wheel
build may not have completed yet. When Phase 1 fails because a root package
version is not found on the index, the script:

1. Detects the "No matching distribution found" error for a root package.
2. Writes a skip marker file (`requirements/.skip-<variant>.json`) with details.
3. Exits with code 0 (so CI stays green).

The existing `requirements-*.txt` on disk remains unchanged. The workflow
sends a Slack notification and the next scheduled run (daily) will
retry automatically.

## CI / GitHub Workflow

`.github/workflows/requirements.yml` (`Requirements Regeneration`) runs:

- on manual trigger (`workflow_dispatch`)
- daily (`0 6 * * *`)

Execution rules:

- Manual runs are allowed for any branch via the required `branch` input.
- Scheduled runs execute only for `opendatahub-io/MLServer` and process `rhoai-staging` as the base branch.

The workflow uses a four-job fanout pattern:

1. **gate** — validates that the selected `BASE_BRANCH` exists in the repository
2. **generate** (matrix: cpu, cuda) — runs the requirements generation per variant in parallel:
   - checks out the `BASE_BRANCH`
   - sets up Python 3.12 and installs `podman`, `yq`, and `jq`
   - extracts the base image from the configured Dockerfile
   - runs the generator inside that base image container
   - checks for skip markers (root package version not on index)
   - uploads the generated requirements file OR skip marker as an artifact
3. **create-pr** — downloads all variant artifacts and creates/updates a single PR:
   - skips if no requirements artifacts were generated (all variants skipped)
   - requests reviewers from `OWNERS` file (only for `opendatahub-io/MLServer`)
   - uses branch `requirements/regenerate-<BASE_BRANCH>`
4. **notify-slack** — sends a Slack notification when any variant was skipped:
   - downloads skip marker artifacts
   - builds an informational message with expected wheels and status
   - sends via the `ALERTS_MLSERVER_CHANNEL_SLACK_WEBHOOK_URL` webhook

Registry login is required and uses secrets:

- `AIPCC_QUAY_USERNAME` / `AIPCC_QUAY_PASSWORD`
- `quay.io` registry

The workflow fails early if credentials are missing.

## Local Usage

### Print base image from Dockerfile

```bash
python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux.cuda
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

### Custom platform selection

`--platform` can be repeated. When used, it overrides the variant's configured `platforms` value. Only short architecture names are accepted.

```bash
python hack/generate-pinned-requirements.py --variant cpu \
  -o requirements/requirements-cpu.txt \
  --platform x86_64 \
  --platform aarch64 \
  --platform ppc64le
```

Unsupported architecture names cause an immediate error listing all supported values.

### Custom download timeout

`--timeout` overrides the variant's `download_timeout` config value:

```bash
python hack/generate-pinned-requirements.py --variant cuda \
  -o requirements/requirements-cuda.txt \
  --timeout 900
```

## Operational Notes

- Run generation inside the target runtime base image for each variant so pip resolves against the intended index and environment.
- Keep `requirements-config.json` and workflow behavior aligned when adding new variants.
- Generated files are expected under `requirements/` and are the only artifacts committed by the workflow.
- CUDA variants must NOT include `ppc64le` in their platforms — the AIPCC CUDA index does not provide ppc64le wheels for several packages. The script emits a warning but does not hard-fail if this misconfiguration is detected.
- After a version bump, the requirements regeneration may skip until the new wheels are published to the AIPCC index. A Slack notification is sent when this happens.
