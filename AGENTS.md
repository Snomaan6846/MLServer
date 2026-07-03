# AGENTS.md — MLServer (opendatahub-io/MLServer)

V2 Inference Protocol (KFServing) server for multi-model ML serving over
REST (FastAPI) and gRPC. ODH midstream fork of `SeldonIO/MLServer` with
Konflux builds, runtime security hardening, and release automation.
Poetry-only monorepo: core `mlserver` + 11 runtime packages under `runtimes/`.

- **Python:** 3.10–3.12 (all branches)
- **Package manager:** Poetry (always use Poetry, never pip install directly)

## Constraints

- **Generated files are read-only.** Never hand-edit files produced by
  `make generate`. Edit the source, then run `make generate`:

  | Generated | Source |
  |---|---|
  | `mlserver/grpc/*_pb2*.py`, `*_pb2*.pyi` | `proto/*.proto` |
  | `mlserver/types/dataplane.py`, `model_repository.py` | `openapi/*.yaml` |

- **`runtimes/*/tox.ini` are based on `tox.runtime.ini`.** Most are exact
  copies; a few (e.g. alibi-detect) add runtime-specific env vars. For shared
  test config, edit the root template only. Exception: `runtimes/onnx-cuda/tox.ini`
  has a custom layout (default env excludes `cuda`-marked tests; separate
  `testenv:cuda` runs them serially). The `bootstrap-test` Makefile target
  skips `onnx-cuda` to preserve its custom `tox.ini` layout.
- **Do not modify without explicit request:** `.tekton/`, `.github/workflows/`,
  `OWNERS`, release/sync automation. These control the supply chain — unauthorized
  edits risk pipeline injection, secrets exposure, privilege escalation, or release
  tampering. Exceptions require an approver sign-off (see `OWNERS`) via normal PR
  review; changes to these files are also covered by the `Ask First` boundary below.
- **Branch syncs must use merge commits**, never squash. Add the
  `tide/merge-method-merge` label.
- **`pyproject.toml` is not auto-synced** from `release-*` to `rhoai-staging`.
  Manual sync PR required, then `make lock`.

## Development

- **Formatter:** black (line length 88). Run `make fmt`.
- **Linter:** flake8 (line length 88, ignore E203).
- **Type checker:** mypy with `ignore_missing_imports = true`. Do not add
  `# type: ignore` to suppress errors — fix them or update stubs.
- **Type annotations** expected on new public functions and methods.
- **Tests:** `async def test_*` with `asyncio_mode = auto`; parallel via `-n auto`.
  CI matrix: Python 3.10/3.11/3.12; all-runtimes suite runs on push only (not PR).
- **Tox envs:** `mlserver-{conda,venv}` (core), `all-runtimes-{conda,venv}` (everything), `licenses`.
- **CUDA tests** are marked `@pytest.mark.cuda` and auto-skip without GPU
  hardware. Run via `tox -c ./runtimes/onnx-cuda -e cuda` on a GPU machine.

```bash
make install-dev                     # Install all deps (all-runtimes + dev)
make install-dev-odh                 # ODH CPU runtimes + dev tools
make install-dev-odh-cuda            # ODH CUDA runtime + dev tools
make lint                            # black --check, flake8, mypy
make fmt                             # black .
make generate                        # Protobuf/OpenAPI codegen
make test                            # Full suite (root tox + each runtime)
make test-cuda                       # Run GPU tests (requires GPU hardware)
make lock                            # Regenerate poetry.lock (root + runtimes)
poetry run tox -e mlserver-venv      # Core tests with venv isolation
poetry run tox -c ./runtimes/<name>  # Single runtime tests
```

## Gotchas

1. **New test fixtures must be registered — tests run in PRODUCTION mode.**
   New `MLModel` subclasses must be in `tests/fixtures.py` and added to
   `TEST_ONLY_EXTRA_IMPLEMENTATIONS` in root `conftest.py`, or the
   trusted-runtimes allowlist rejects them. Two-tier security model:
   - **PRODUCTION** (`/etc/mlserver/trusted-runtimes.json` present): only
     runtimes in `ALLOWED_MODEL_IMPLEMENTATIONS` (`mlserver/settings.py`) load;
     custom envs and wildcard CORS blocked.
   - **DEVELOPMENT** (no artifact file): any runtime loads freely.

2. **Serial test suites: kafka, parallel, grpc, env, cli.** Run after the
   parallel bulk to avoid port conflicts. Do not introduce shared-port
   usage or global state mutations in these.

3. **`pyproject.toml` changes require `make lock`.** Always regenerate
   lockfiles for root and affected runtimes after dependency changes.

4. **Version must stay in sync.** `mlserver/version.py` is the source of
   truth. All `runtimes/*/pyproject.toml` and `docs/conf.py` must match.
   Use `hack/update-version.sh <version>` for bumps — never hand-edit.

5. **`Dockerfile.konflux` and `Dockerfile.cuda.konflux` exist only on
   `rhoai-staging`.** Not on `master` or `release-*`. Renovate auto-updates
   their base images.

6. **Adding a built-in runtime** requires updating
   `ALLOWED_MODEL_IMPLEMENTATIONS` in `mlserver/settings.py` and the
   `TRUSTED_RUNTIMES` build arg in the Dockerfile.

7. **`.tekton/` varies across branches.** On `master`: PR, push, and
   early-gate pipelines. On `release-*`: push pipeline only (produces
   versioned `odh-vX.Y` tags). On `rhoai-staging`: no Tekton pipelines.

8. **Early-gate CI** is triggered by commenting `/early-gate` on a PR. It
   runs a Konflux build+test pipeline for pre-merge validation. Do not
   remove or rename `.tekton/early-gate-ci-{build,test}.yaml` without
   coordinating with the Konflux team.

9. **`mlserver-onnx-cuda` is a metapackage** that depends on `mlserver-onnx`
   and adds `onnxruntime-gpu`. It shares the `mlserver_onnx.OnnxModel`
   runtime class — no separate `ALLOWED_MODEL_IMPLEMENTATIONS` entry needed.
   Because `onnxruntime` (CPU) and `onnxruntime-gpu` share the same Python
   namespace, CUDA Dockerfiles run `pip uninstall onnxruntime` then
   `pip install --force-reinstall --no-deps onnxruntime-gpu` after installing
   the wheels.

10. **`onnxruntime` and `onnxruntime-gpu` version bounds must stay in sync.**
    When bumping `onnxruntime` bounds in `runtimes/onnx/pyproject.toml`, the
    mirrored `onnxruntime-gpu` bounds in `runtimes/onnx-cuda/pyproject.toml`
    must be updated in the same PR. Run `make lock` for both runtimes.

## Boundaries

### Always
- Run `make lint && poetry run tox -e mlserver-venv` before proposing changes
- Run `make generate` after editing `.proto` or OpenAPI YAML files
- Commit generated file changes together with the source edit that triggered them
- Use `hack/update-version.sh` for version bumps
- For runtime changes, also run `poetry run tox -c ./runtimes/<name>`

### Ask First
- Changes to CI workflows or Tekton pipelines
- Dependency version bumps in `pyproject.toml`
- Changes to `conftest.py` trusted-runtime configuration
- Modifications to `Dockerfile`, `Dockerfile.cuda`, `Dockerfile.konflux`, or `Dockerfile.cuda.konflux`

### Never
- Hand-edit generated files under `mlserver/grpc/` or `mlserver/types/`
- Edit `runtimes/*/tox.ini` for shared config — update `tox.runtime.ini` instead
- Squash-merge branch sync PRs
- Modify release automation or tagging workflows without explicit request
- Edit `OWNERS` / `OWNERS_ALIASES` without team agreement

## Branch Strategy

### ODH Branches (`opendatahub-io/MLServer`)

| Branch | Purpose | Version | Build Source | Nightly Gate |
|---|---|---|---|---|
| `master` | Development — features and bug fixes land here first | `1.7.0.dev0` | `Dockerfile` | No |
| `release-1.7.x` | Stable ODH release line; ODH release tags are cut here | `1.7.1` | `Dockerfile` | ODH nightly |
| `rhoai-staging` | RHOAI staging; `Dockerfile.konflux` and pinned `requirements/` live here | `1.7.1+rhaiv.8` | `Dockerfile` | No |

**ODH image tags** (`quay.io/opendatahub/mlserver`):

- `master` push: `odh-stable` (floating)
- `master` PR: `odh-pr` (floating) + `odh-pr-<PR#>` (e.g. `odh-pr-197`) + `odh-pr-<sha>` (pinned commit SHA)
- `release-*` push: `odh-vX.Y[-EAN]` (e.g. `odh-v3.4` for GA, `odh-v3.5-EA2` for EA)
- `rhoai-staging`: no Konflux pipelines; image built from `Dockerfile.konflux` in RHDS

### RHDS Branches (`red-hat-data-services/MLServer`)

| Branch | Purpose | Build Source | Nightly Gate |
|---|---|---|---|
| `main` | Synced from ODH `rhoai-staging` via automated devops workflow (no image builds) | `Dockerfile.konflux` | No |
| `rhoai-*` | RHOAI release branches cut from `main` | `Dockerfile.konflux` | RHOAI nightly |

**RHDS image tags** (`quay.io/rhoai/odh-mlserver-rhel9`):

- `rhoai-<ver>-<sha>` — pinned per-build (every push)
- `rhoai-<ver>` — floating latest GA build
- From 3.4+: `rhoai-<ver>-ea.<N>` / `rhoai-<ver>-ea.<N>-<sha>` for EA milestones
- Release progression: `ea.1` → `ea.2` → … → GA

### Flow

```
upstream (SeldonIO/MLServer)
    │  pull-bot weekly sync
    ▼
  master  ──cherry-pick──▶  release-1.7.x  ──Prow Merge workflow──▶  rhoai-staging
                               (ODH releases                    (Dockerfile.konflux lives here;
                                tagged here)                     RHOAI staging/validation)
                                                                            │
                                                              devops auto-merge every 4h (check freeze dates!)
                                                                            ▼
                                                         RHDS main (red-hat-data-services/MLServer)
                                                                            │
                                                              devops auto-sync until code freeze
                                                                            ▼
                                                              RHDS rhoai-* release branch
```

- New ODH major release branches (`release-1.8.x`, etc.) are cut from `master`.
- Past RHOAI release fixes: cherry-pick from RHDS `main` to the relevant `rhoai-*` branch.

## Release Process

### ODH Release

> **Ordering constraint:** The MLServer release must be completed before the
> `odh-model-controller` release so the expected MLServer image tag exists.

1. **Validate release branch** — confirm all required changes have been cherry-picked from
   `master` to the latest `release-*` branch.

2. **Create tag and bump** — run the `Create Tag and Bump for Next Release` workflow from
   the release branch. Provide the current release tag and the next development tag.
   Example flow: `odh-v3.5-EA1` → `odh-v3.5-EA2` → `odh-v3.5` (GA). This creates the
   release tag and bumps Konflux workflow references to the next tag.

3. **Validate `odh-model-controller`** — confirm the mlserver image tag in `params.env`
   reflects the new release tag once the odh-model-controller release is done.

### RHOAI Release

1. **Sync release branch → rhoai-staging** — run the
   `Prow Merge release-* Branch to rhoai-staging Branch` workflow from the release branch.
   On conflict, perform a manual merge. Always use `tide/merge-method-merge` to prevent squash.

2. **Create AIPCC tag** from `rhoai-staging` if any changes have landed since last release:
   - Git tag format: `v<mlserver-version>+rhaiv.<N>` — e.g. `v1.7.1+rhaiv.9`
   - Before tagging, run `hack/update-version.sh <version>` where `<version>` is the tag
     **without the `v` prefix** (e.g. `1.7.1+rhaiv.9`) to update `mlserver/version.py`,
     all `runtimes/*/pyproject.toml`, and `docs/conf.py`.
   - AIPCC team uses this tag to build the mlserver wheel bundle required for RHOAI image builds.

3. **Sync rhoai-staging → RHDS main** — automated every 4 hours; not gated by code freeze.
   On failure, sync manually with `tide/merge-method-merge`.

4. **Hermetic build dependencies** — `requirements/*.txt` on `rhoai-staging` are
   regenerated by the `Requirements Regeneration` workflow (`.github/workflows/requirements.yml`),
   which runs every 12 hours and opens a PR against `rhoai-staging` on branch
   `requirements/regenerate-rhoai-staging`. The PR requires review before merging — files
   are not auto-committed. Once merged, the updated files flow to RHDS `main` via step 3.
   To trigger manually: **Actions → Requirements Regeneration → Run workflow**, set
   `branch` to `rhoai-staging`.

5. **Sync RHDS main → RHOAI release branch** — automated until code freeze; stops when
   devops empties `releases.yaml`. For post-freeze manual syncs, use `tide/merge-method-merge`.

6. **RHOAI release tagging** — Konflux pipeline tags on Quay:
   `rhoai-<ver>-ea.<N>` (EA) → `rhoai-<ver>` (GA); each build also gets a pinned `*-<sha>` tag.

## Code Ownership

Approvers and reviewers are defined in the `OWNERS` file at the repo root.
