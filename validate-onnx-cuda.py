#!/usr/bin/env python3
"""ONNX / ONNX-CUDA Validation Script for MLServer.

Portable script that validates every documented command, structural invariant,
and cross-file consistency rule for mlserver-onnx and mlserver-onnx-cuda.

Three independent phases:
  Phase 1 – Local repo checks (static file checks + optional Make target validation)
  Phase 2 – Container image validation (run per image via podman)
  Phase 3 – CUDA node validation (GPU hardware required)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Result types and display helpers
# ---------------------------------------------------------------------------

# (name, passed, detail)  —  passed: True=PASS, False=FAIL, None=SKIP
Result = tuple[str, Optional[bool], str]

_IS_TTY = sys.stdout.isatty()
_STATE_FILE = Path(".validate-onnx-cuda-state.json")
_FAIL_FAST = False


class FailFastExit(Exception):
    """Raised when --fail-fast is set and a check fails."""

    def __init__(self, result: "Result"):
        self.result = result


def _load_state(state_file: Path) -> dict:
    """Load resume state from JSON file."""
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed": {}, "results": {}}


def _save_state(state_file: Path, state: dict) -> None:
    """Save resume state to JSON file."""
    try:
        state_file.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


def _mark_done(
    state: dict, step: str, results: list[Result], force: bool = False
) -> None:
    """Mark a step as completed and store its results.

    By default, only caches if all checks passed or were skipped (no FAILs).
    Use force=True to cache regardless (e.g. for steps that are expected to
    have failures you don't want to re-run).
    """
    has_failures = any(p is False for _, p, _ in results)
    if has_failures and not force:
        return
    state["completed"][step] = True
    state["results"][step] = [
        {"name": n, "passed": p, "detail": d} for n, p, d in results
    ]


def _is_done(state: dict, step: str) -> bool:
    """Check if a step was already completed."""
    return state.get("completed", {}).get(step, False)


def _replay_results(state: dict, step: str) -> list[Result]:
    """Replay stored results from a previous run."""
    stored = state.get("results", {}).get(step, [])
    return [(r["name"], r["passed"], r["detail"]) for r in stored]


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _IS_TTY else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _IS_TTY else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _IS_TTY else s


def _tag(passed: Optional[bool]) -> str:
    if passed is True:
        return _green("[PASS]")
    elif passed is False:
        return _red("[FAIL]")
    return _yellow("[SKIP]")


def _print_result(r: Result) -> None:
    name, passed, detail = r
    tag = _tag(passed)
    print(f"  {tag} {name}: {detail}")
    if _FAIL_FAST and passed is False:
        raise FailFastExit(r)


def _print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def _summarize(results: list[Result], label: str) -> bool:
    passed = sum(1 for _, p, _ in results if p is True)
    failed = sum(1 for _, p, _ in results if p is False)
    skipped = sum(1 for _, p, _ in results if p is None)
    total = passed + failed + skipped
    line = f"  {label}: {passed} passed, {failed} failed, {skipped} skipped (total {total})"
    if failed > 0:
        print(_red(line))
    else:
        print(_green(line))
    return failed == 0


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _read_file(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _run(
    cmd: list[str], cwd: Optional[Path] = None, **kw
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, **kw)
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            cmd,
            returncode=127,
            stdout="",
            stderr=f"command not found: {cmd[0]}",
        )


def _run_live(cmd: list[str], cwd: Optional[Path] = None) -> int:
    """Run a command with output streamed to the terminal in real-time."""
    try:
        proc = subprocess.run(cmd, cwd=cwd)
        return proc.returncode
    except FileNotFoundError:
        print(f"  command not found: {cmd[0]}")
        return 127


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError(
            "Python 3.11+ tomllib or pip-installed tomli required for TOML parsing"
        )
    return tomllib.loads(path.read_text())


def _extract_version_bounds(dep_value) -> dict[str, str]:
    """Build {marker -> version_spec} dict from a TOML dependency value.

    Handles both a single dict and list-of-dicts (marker-keyed entries).
    """
    if isinstance(dep_value, str):
        return {"": dep_value}
    if isinstance(dep_value, dict):
        return {dep_value.get("markers", ""): dep_value.get("version", "*")}
    if isinstance(dep_value, list):
        result: dict[str, str] = {}
        for entry in dep_value:
            if isinstance(entry, dict):
                result[entry.get("markers", "")] = entry.get("version", "*")
        return result
    return {}


# =========================================================================
# PHASE 1 — Local Repo Checks
# =========================================================================


def _cat1_package_structure(repo: Path) -> list[Result]:
    """Category 1: Package Structure."""
    results: list[Result] = []

    onnx_dir = repo / "runtimes" / "onnx"
    results.append(("runtimes/onnx/ exists", onnx_dir.is_dir(), str(onnx_dir)))

    cuda_dir = repo / "runtimes" / "onnx-cuda"
    results.append(("runtimes/onnx-cuda/ exists", cuda_dir.is_dir(), str(cuda_dir)))

    init_py = cuda_dir / "mlserver_onnx_cuda" / "__init__.py"
    init_content = _read_file(init_py)
    has_reexport = "from mlserver_onnx import *" in init_content
    results.append(
        (
            "onnx-cuda re-exports mlserver_onnx",
            has_reexport,
            "from mlserver_onnx import *" if has_reexport else "not found",
        )
    )

    cuda_pyproject = cuda_dir / "pyproject.toml"
    if cuda_pyproject.exists() and tomllib is not None:
        data = _load_toml(cuda_pyproject)
        deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        has_ort_gpu = "onnxruntime-gpu" in deps
        results.append(
            (
                "onnx-cuda depends on mlserver-onnx",
                "mlserver-onnx" in deps,
                str(list(deps.keys())),
            )
        )
        results.append(
            (
                "onnx-cuda depends on onnxruntime-gpu",
                has_ort_gpu,
                str(list(deps.keys())),
            )
        )
    else:
        results.append(
            ("onnx-cuda pyproject.toml readable", False, "missing or no TOML parser")
        )

    onnx_pyproject = onnx_dir / "pyproject.toml"
    if onnx_pyproject.exists() and tomllib is not None:
        data = _load_toml(onnx_pyproject)
        deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        results.append(
            (
                "onnx depends on onnxruntime",
                "onnxruntime" in deps,
                str(list(deps.keys())),
            )
        )
    else:
        results.append(
            ("onnx pyproject.toml readable", False, "missing or no TOML parser")
        )

    cuda_tests = cuda_dir / "tests"
    results.append(
        (
            "onnx-cuda has NO tests/ dir (shared via onnx/tests)",
            not cuda_tests.exists(),
            "absent (correct)" if not cuda_tests.exists() else "exists (unexpected)",
        )
    )

    return results


def _cat2_version_sync(repo: Path) -> list[Result]:
    """Category 2: Version Sync."""
    results: list[Result] = []
    if tomllib is None:
        results.append(
            ("TOML parser available", False, "install tomli for Python <3.11")
        )
        return results

    onnx_pyproject = repo / "runtimes" / "onnx" / "pyproject.toml"
    cuda_pyproject = repo / "runtimes" / "onnx-cuda" / "pyproject.toml"

    if not onnx_pyproject.exists() or not cuda_pyproject.exists():
        results.append(("pyproject.toml files exist", False, "one or both missing"))
        return results

    onnx_data = _load_toml(onnx_pyproject)
    cuda_data = _load_toml(cuda_pyproject)

    onnx_deps = onnx_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    cuda_deps = cuda_data.get("tool", {}).get("poetry", {}).get("dependencies", {})

    ort_bounds = _extract_version_bounds(onnx_deps.get("onnxruntime", "*"))
    ort_gpu_bounds = _extract_version_bounds(cuda_deps.get("onnxruntime-gpu", "*"))

    bounds_match = ort_bounds == ort_gpu_bounds
    results.append(
        (
            "onnxruntime / onnxruntime-gpu bounds match",
            bounds_match,
            f"ort={ort_bounds} gpu={ort_gpu_bounds}",
        )
    )

    version_py = repo / "mlserver" / "version.py"
    version_content = _read_file(version_py)
    m = re.search(r'__version__\s*=\s*"([^"]+)"', version_content)
    source_version = m.group(1) if m else ""

    for name, pyproject_path in [
        ("onnx", onnx_pyproject),
        ("onnx-cuda", cuda_pyproject),
    ]:
        data = _load_toml(pyproject_path)
        pkg_version = data.get("tool", {}).get("poetry", {}).get("version", "")
        match = pkg_version == source_version
        results.append(
            (
                f"runtimes/{name} version matches version.py",
                match,
                f"{pkg_version} vs {source_version}",
            )
        )

    for name in ["onnx", "onnx-cuda"]:
        lock_path = repo / "runtimes" / name / "poetry.lock"
        results.append(
            (
                f"runtimes/{name}/poetry.lock exists",
                lock_path.exists(),
                (
                    "exists"
                    if lock_path.exists()
                    else "MISSING (staleness checked in Category 11)"
                ),
            )
        )

    return results


def _cat3_makefile_targets(repo: Path) -> list[Result]:
    """Category 3: Makefile Targets."""
    results: list[Result] = []
    makefile = repo / "Makefile"
    content = _read_file(makefile)

    if not content:
        results.append(("Makefile exists", False, "not found"))
        return results

    all_phony_targets: set[str] = set()
    for m in re.finditer(r"^\.PHONY:\s*(.+?)$", content, re.MULTILINE):
        line_targets = m.group(1)
        line_targets = re.sub(r"\\\s*\n\s*", " ", line_targets)
        all_phony_targets.update(line_targets.split())

    for target in ["install-dev-odh", "install-dev-odh-cuda", "test-cuda"]:
        results.append(
            (
                f".PHONY includes {target}",
                target in all_phony_targets,
                (
                    "present"
                    if target in all_phony_targets
                    else f"missing from: {sorted(all_phony_targets)[:10]}..."
                ),
            )
        )

    patterns = {
        "install-dev-odh target": (
            r"^install-dev-odh:",
            r"--with odh-runtimes --with dev",
        ),
        "install-dev-odh-cuda target": (
            r"^install-dev-odh-cuda:",
            r"--with odh-runtimes-cuda --with odh-runtimes-cuda-dev --with dev",
        ),
        "test-cuda target": (r"^test-cuda:", r"\./runtimes/onnx-cuda.*-e cuda"),
    }
    for name, (header_re, body_re) in patterns.items():
        header_match = re.search(header_re, content, re.MULTILINE)
        if header_match:
            pos = header_match.end()
            next_target = re.search(r"^\S+:", content[pos:], re.MULTILINE)
            block = (
                content[pos : pos + next_target.start()]
                if next_target
                else content[pos:]
            )
            body_ok = re.search(body_re, block) is not None
            results.append(
                (
                    name,
                    body_ok,
                    "content matches" if body_ok else f"body mismatch in block",
                )
            )
        else:
            results.append((name, False, "target header not found"))

    bt_header = re.search(r"^bootstrap-test:", content, re.MULTILINE)
    if bt_header:
        pos = bt_header.end()
        next_target = re.search(r"^\S+:", content[pos:], re.MULTILINE)
        bt_block = (
            content[pos : pos + next_target.start()] if next_target else content[pos:]
        )
        skip_guard = "onnx-cuda" in bt_block and "continue" in bt_block
    else:
        skip_guard = False
    results.append(
        (
            "bootstrap-test skips onnx-cuda",
            skip_guard,
            "skip guard found" if skip_guard else "not found",
        )
    )

    actionable_globs = ["Makefile", "*.py", "*.ini"]
    self_name = Path(__file__).resolve().name
    stale_found: list[str] = []
    for g in actionable_globs:
        for f in repo.rglob(g):
            if ".tox" in f.parts or ".git" in f.parts or "node_modules" in f.parts:
                continue
            if f.name == self_name:
                continue
            try:
                text = f.read_text()
            except Exception:
                continue
            if re.search(r"\binstall-dev-cuda\b", text):
                stale_found.append(str(f.relative_to(repo)))
    for f in repo.rglob("pyproject.toml"):
        if ".tox" in f.parts or ".git" in f.parts:
            continue
        try:
            text = f.read_text()
        except Exception:
            continue
        if re.search(r"\binstall-dev-cuda\b", text):
            stale_found.append(str(f.relative_to(repo)))

    results.append(
        (
            "no stale install-dev-cuda in actionable files",
            len(stale_found) == 0,
            "clean" if not stale_found else f"found in: {stale_found}",
        )
    )

    return results


def _cat4_tox_config(repo: Path) -> list[Result]:
    """Category 4: Tox Configuration."""
    results: list[Result] = []
    tox_path = repo / "runtimes" / "onnx-cuda" / "tox.ini"
    tox_runtime = repo / "tox.runtime.ini"
    tox_content = _read_file(tox_path)
    runtime_content = _read_file(tox_runtime)

    if not tox_content:
        results.append(("onnx-cuda tox.ini exists", False, str(tox_path)))
        return results

    results.append(
        (
            "onnx-cuda tox.ini is NOT a copy of tox.runtime.ini",
            tox_content.strip() != runtime_content.strip(),
            (
                "custom content"
                if tox_content.strip() != runtime_content.strip()
                else "identical (bad)"
            ),
        )
    )

    results.append(
        (
            "[testenv] uses -m 'not cuda' -n auto",
            '-m "not cuda"' in tox_content and "-n auto" in tox_content,
            "found" if '-m "not cuda"' in tox_content else "missing",
        )
    )

    cuda_section = re.search(r"\[testenv:cuda\](.*?)(?=\[|\Z)", tox_content, re.DOTALL)
    if cuda_section:
        cuda_block = cuda_section.group(1)
        results.append(
            (
                "[testenv:cuda] uses -m cuda (serial)",
                "-m cuda" in cuda_block and "-n auto" not in cuda_block,
                (
                    "serial cuda tests"
                    if "-n auto" not in cuda_block
                    else "-n auto found (should be serial)"
                ),
            )
        )
        results.append(
            (
                "[testenv:cuda] passenv includes CUDA vars",
                all(
                    v in cuda_block
                    for v in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME"]
                ),
                (
                    "all present"
                    if all(
                        v in cuda_block
                        for v in [
                            "CUDA_VISIBLE_DEVICES",
                            "LD_LIBRARY_PATH",
                            "CUDA_HOME",
                        ]
                    )
                    else "missing some"
                ),
            )
        )
    else:
        results.append(("[testenv:cuda] section exists", False, "not found"))

    results.append(
        (
            "test paths point to ../onnx/tests",
            "../onnx/tests" in tox_content,
            "found" if "../onnx/tests" in tox_content else "missing",
        )
    )

    return results


def _cat5_dependency_groups(repo: Path) -> list[Result]:
    """Category 5: Dependency Groups (root pyproject.toml)."""
    results: list[Result] = []
    root_pyproject = repo / "pyproject.toml"
    if not root_pyproject.exists() or tomllib is None:
        results.append(
            ("root pyproject.toml readable", False, "missing or no TOML parser")
        )
        return results

    data = _load_toml(root_pyproject)
    groups = data.get("tool", {}).get("poetry", {}).get("group", {})

    has_cuda_group = "odh-runtimes-cuda" in groups
    results.append(
        (
            "odh-runtimes-cuda group exists",
            has_cuda_group,
            "present" if has_cuda_group else "missing",
        )
    )

    if has_cuda_group:
        cuda_deps = groups["odh-runtimes-cuda"].get("dependencies", {})
        results.append(
            (
                "odh-runtimes-cuda includes mlserver-onnx-cuda",
                "mlserver-onnx-cuda" in cuda_deps,
                str(list(cuda_deps.keys())),
            )
        )

    all_rt_deps = groups.get("all-runtimes", {}).get("dependencies", {})
    results.append(
        (
            "all-runtimes excludes mlserver-onnx-cuda",
            "mlserver-onnx-cuda" not in all_rt_deps,
            (
                "excluded (correct)"
                if "mlserver-onnx-cuda" not in all_rt_deps
                else "INCLUDED (namespace collision risk)"
            ),
        )
    )

    return results


def _cat6_dockerfile_consistency(repo: Path) -> list[Result]:
    """Category 6: Dockerfile Consistency + 6a/6b subsections."""
    results: list[Result] = []

    # -- Dockerfile.cuda.konflux (only on rhoai-staging branch) --
    dcf = _read_file(repo / "Dockerfile.cuda.konflux")
    exists = bool(dcf)
    results.append(
        (
            "Dockerfile.cuda.konflux exists",
            True if exists else None,
            "found" if exists else "not present (only on rhoai-staging branch)",
        )
    )
    if exists:
        results.append(
            (
                "  pip freeze > /tmp/constraints.txt",
                "pip freeze > /tmp/constraints.txt" in dcf,
                "",
            )
        )
        results.append(
            (
                "  --constraint /tmp/constraints.txt",
                "--constraint /tmp/constraints.txt" in dcf,
                "",
            )
        )
        results.append(
            ("  rm -rf /root/.cache/pip", "rm -rf /root/.cache/pip" in dcf, "")
        )
        results.append(
            ("  TRUSTED_RUNTIMES validation", "is_valid_runtime_import_path" in dcf, "")
        )
        results.append(("  USER 1000", "USER 1000" in dcf, ""))
        results.append(
            (
                "  LD_LIBRARY_PATH comment (AIPCC)",
                "LD_LIBRARY_PATH" in dcf and "AIPCC" in dcf,
                "",
            )
        )

    # -- Dockerfile.konflux (only on rhoai-staging branch) --
    dkf = _read_file(repo / "Dockerfile.konflux")
    exists_k = bool(dkf)
    results.append(
        (
            "Dockerfile.konflux exists",
            True if exists_k else None,
            "found" if exists_k else "not present (only on rhoai-staging branch)",
        )
    )
    if exists_k:
        results.append(
            (
                "  Dockerfile.konflux has rm -rf /root/.cache/pip",
                "rm -rf /root/.cache/pip" in dkf,
                "",
            )
        )

    # -- Dockerfile.cuda --
    dc = _read_file(repo / "Dockerfile.cuda")
    if dc:
        results.append(
            (
                "Dockerfile.cuda CUDA_VERSION sync comment",
                "Dockerfile.cuda.konflux" in dc,
                "",
            )
        )
        results.append(
            ("Dockerfile.cuda wheel loop hyphen fix", "${_runtime//-/_}" in dc, "")
        )
        results.append(("Dockerfile.cuda --constraint", "--constraint" in dc, ""))
        results.append(
            (
                "Dockerfile.cuda TRUSTED_RUNTIMES validation",
                "is_valid_runtime_import_path" in dc,
                "",
            )
        )
        results.append(("Dockerfile.cuda USER 1000", "USER 1000" in dc, ""))

    # -- Dockerfile (CPU) --
    d = _read_file(repo / "Dockerfile")
    if d:
        results.append(
            ("Dockerfile wheel loop hyphen fix", "${_runtime//-/_}" in d, "")
        )

    # -- 6a: Requirements Generation Config --
    cfg_path = repo / "hack" / "requirements-config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        variants = cfg.get("variants", [])
        cuda_variant = [v for v in variants if v.get("name") == "cuda"]
        results.append(
            ("requirements-config.json has cuda variant", len(cuda_variant) == 1, "")
        )
        if cuda_variant:
            cv = cuda_variant[0]
            results.append(
                (
                    "  cuda variant dockerfile",
                    cv.get("dockerfile") == "Dockerfile.cuda.konflux",
                    cv.get("dockerfile", "?"),
                )
            )
            rp = cv.get("root_packages", [])
            results.append(
                (
                    "  cuda variant root_packages",
                    "mlserver" in rp and "mlserver-onnx-cuda" in rp,
                    str(rp),
                )
            )
    else:
        results.append(("requirements-config.json exists", False, "missing"))

    gen_script = repo / "hack" / "generate-pinned-requirements.py"
    if gen_script.exists():
        gen_content = gen_script.read_text()
        results.append(
            (
                "generate-pinned-requirements.py supports --variant",
                "--variant" in gen_content,
                "",
            )
        )
    else:
        results.append(("generate-pinned-requirements.py exists", False, "missing"))

    # -- 6b: Renovate Config --
    renovate_path = repo / ".github" / "renovate.json"
    if renovate_path.exists():
        ren = json.loads(renovate_path.read_text())
        rules = ren.get("packageRules", [])
        cuda_in_filenames = any(
            "Dockerfile.cuda.konflux" in r.get("matchFileNames", []) for r in rules
        )
        cuda_in_depnames = any(
            any("cuda" in d for d in r.get("matchDepNames", [])) for r in rules
        )
        results.append(
            (
                "renovate matchFileNames includes Dockerfile.cuda.konflux",
                cuda_in_filenames,
                "",
            )
        )
        results.append(
            ("renovate matchDepNames includes CUDA base image", cuda_in_depnames, "")
        )
    else:
        results.append(
            (
                "renovate.json exists",
                None,
                "file not present (may be on different branch)",
            )
        )

    return results


def _cat7_test_infrastructure(repo: Path) -> list[Result]:
    """Category 7: Test Infrastructure."""
    results: list[Result] = []
    test_cuda = repo / "runtimes" / "onnx" / "tests" / "test_cuda.py"
    content = _read_file(test_cuda)

    results.append(("test_cuda.py exists", bool(content), str(test_cuda)))
    if not content:
        return results

    results.append(
        (
            "from typing import AsyncGenerator NOT present",
            "from typing import AsyncGenerator" not in content,
            (
                "absent (correct)"
                if "from typing import AsyncGenerator" not in content
                else "PRESENT (anti-pattern)"
            ),
        )
    )

    results.append(("_has_cuda() function exists", "def _has_cuda" in content, ""))

    has_marks = "pytest.mark.cuda" in content and "requires_cuda" in content
    results.append(("pytestmark applies cuda + requires_cuda", has_marks, ""))

    results.append(
        (
            "test_invalid_device_id_falls_back_to_cpu exists",
            "test_invalid_device_id_falls_back_to_cpu" in content,
            "",
        )
    )

    inline_import = any(
        re.match(r"[ \t]+import asyncio", line) for line in content.split("\n")
    )

    results.append(
        (
            "no inline import asyncio in test functions",
            not inline_import,
            "clean" if not inline_import else "found indented import asyncio",
        )
    )

    return results


def _cat8_documentation_consistency(repo: Path) -> list[Result]:
    """Category 8: Documentation Consistency."""
    results: list[Result] = []
    agents = _read_file(repo / "AGENTS.md")

    bash_block = re.search(r"```bash\n(.*?)```", agents, re.DOTALL)
    if bash_block:
        block = bash_block.group(1)
        for cmd in ["install-dev-odh", "install-dev-odh-cuda", "test-cuda"]:
            results.append(
                (
                    f"AGENTS.md bash block lists {cmd}",
                    cmd in block,
                    "present" if cmd in block else "missing",
                )
            )
    else:
        results.append(("AGENTS.md bash block found", False, "no ```bash block"))

    results.append(
        (
            "AGENTS.md documents bootstrap-test skipping onnx-cuda",
            "bootstrap-test" in agents and "onnx-cuda" in agents,
            (
                "found"
                if "bootstrap-test" in agents and "onnx-cuda" in agents
                else "missing"
            ),
        )
    )

    has_mirrored = "mirrored" in agents.lower()
    has_lock = "lock" in agents.lower()
    results.append(
        (
            "AGENTS.md has lock file sync gotcha (mirrored + lock)",
            has_mirrored and has_lock,
            "found" if has_mirrored and has_lock else "missing content",
        )
    )

    testing_envs = _read_file(repo / "docs" / "testing" / "TESTING_ENVIRONMENTS.md")
    results.append(
        (
            "TESTING_ENVIRONMENTS.md has odh-runtimes-cuda",
            "odh-runtimes-cuda" in testing_envs,
            "found" if "odh-runtimes-cuda" in testing_envs else "missing",
        )
    )

    onnx_readme = _read_file(repo / "runtimes" / "onnx" / "README.md")
    results.append(
        (
            "onnx README references install-dev-odh-cuda",
            "install-dev-odh-cuda" in onnx_readme,
            "found" if "install-dev-odh-cuda" in onnx_readme else "missing",
        )
    )

    gitbook = _read_file(repo / "docs-gb" / "runtimes" / "onnx.md")
    if gitbook:
        results.append(
            (
                "gitbook onnx.md references install-dev-odh-cuda",
                "install-dev-odh-cuda" in gitbook,
                "found" if "install-dev-odh-cuda" in gitbook else "missing",
            )
        )
    else:
        results.append(
            ("gitbook onnx.md exists", None, "not found (may not be on this branch)")
        )

    self_name = Path(__file__).resolve().name
    stale_found: list[str] = []
    for g in ["Makefile", "*.py", "*.ini"]:
        for f in repo.rglob(g):
            if ".tox" in f.parts or ".git" in f.parts:
                continue
            if f.name == self_name:
                continue
            try:
                text = f.read_text()
            except Exception:
                continue
            if re.search(r"\binstall-dev-cuda\b", text):
                stale_found.append(str(f.relative_to(repo)))
    for f in repo.rglob("pyproject.toml"):
        if ".tox" in f.parts or ".git" in f.parts:
            continue
        try:
            text = f.read_text()
        except Exception:
            continue
        if re.search(r"\binstall-dev-cuda\b", text):
            stale_found.append(str(f.relative_to(repo)))

    results.append(
        (
            "no install-dev-cuda in actionable files",
            len(stale_found) == 0,
            "clean" if not stale_found else f"found in: {stale_found}",
        )
    )

    return results


def _cat9_import_verification(repo: Path) -> list[Result]:
    """Category 9: Import Verification (runtime checks)."""
    results: list[Result] = []

    for module, import_stmt in [
        ("mlserver_onnx", "from mlserver_onnx import OnnxModel"),
        ("mlserver_onnx_cuda", "from mlserver_onnx_cuda import OnnxModel"),
        ("onnxruntime", "import onnxruntime"),
    ]:
        proc = _run(["poetry", "run", "python", "-c", import_stmt])
        if proc.returncode == 0:
            results.append((f"import {module}", True, "ok"))
        else:
            if "ModuleNotFoundError" in proc.stderr or "No module named" in proc.stderr:
                results.append((f"import {module}", None, "not installed"))
            else:
                results.append((f"import {module}", False, proc.stderr.strip()[:120]))

    identity_check = _run(
        [
            "poetry",
            "run",
            "python",
            "-c",
            "from mlserver_onnx import OnnxModel as A; "
            "from mlserver_onnx_cuda import OnnxModel as B; "
            "assert A is B, 'class identity mismatch'",
        ]
    )
    if identity_check.returncode == 0:
        results.append(
            ("OnnxModel class identity (onnx == onnx-cuda)", True, "same class")
        )
    elif (
        "ModuleNotFoundError" in identity_check.stderr
        or "No module named" in identity_check.stderr
    ):
        results.append(
            ("OnnxModel class identity (onnx == onnx-cuda)", None, "not installed")
        )
    else:
        results.append(
            (
                "OnnxModel class identity (onnx == onnx-cuda)",
                False,
                identity_check.stderr.strip()[:120],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Category 11: Live Make Target Validation
# ---------------------------------------------------------------------------

_INSTALL_TARGETS = {
    "install-dev": {
        "expected": [
            "mlserver",
            "mlserver-sklearn",
            "mlserver-xgboost",
            "mlserver-lightgbm",
            "mlserver-onnx",
            "mlserver-mlflow",
            "mlserver-huggingface",
            "mlserver-alibi-explain",
            "mlserver-alibi-detect",
            "mlserver-catboost",
        ],
        "expected_ort": "onnxruntime",
        "not_expected": [],
        "imports": [
            "mlserver",
            "mlserver_onnx.OnnxModel",
            "mlserver_sklearn.SKLearnModel",
        ],
    },
    "install-dev-odh": {
        "expected": [
            "mlserver",
            "mlserver-sklearn",
            "mlserver-xgboost",
            "mlserver-lightgbm",
            "mlserver-onnx",
            "pytest",
            "flake8",
            "mypy",
            "black",
        ],
        "expected_ort": "onnxruntime",
        "not_expected": ["mlserver-mlflow", "mlserver-huggingface"],
        "imports": ["mlserver", "mlserver_onnx.OnnxModel"],
    },
    "install-dev-odh-cuda": {
        "expected": [
            "mlserver",
            "mlserver-onnx",
            "mlserver-onnx-cuda",
            "onnxruntime-gpu",
            "pytest",
            "flake8",
            "mypy",
            "black",
        ],
        "expected_ort": "onnxruntime-gpu",
        "not_expected": ["mlserver-sklearn", "mlserver-xgboost", "mlserver-lightgbm"],
        "imports": [
            "mlserver",
            "mlserver_onnx.OnnxModel",
            "mlserver_onnx_cuda.OnnxModel",
        ],
    },
}


def _get_installed_packages(repo: Path) -> dict[str, str]:
    """Return {lowered_name: version} for all installed packages.

    Uses ``importlib.metadata`` inside the Poetry venv (standard library,
    no pip dependency needed).  Falls back to ``poetry show`` then
    ``poetry run pip list``.
    """
    methods = [
        (
            "importlib.metadata",
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import importlib.metadata, json; "
                "print(json.dumps({d.metadata['Name'].lower(): d.metadata['Version'] "
                "for d in importlib.metadata.distributions()}))",
            ],
            "json",
        ),
        (
            "poetry show",
            ["poetry", "show", "--no-ansi"],
            "table",
        ),
        (
            "poetry run pip list",
            ["poetry", "run", "pip", "list", "--format=json"],
            "json",
        ),
    ]

    for name, cmd, fmt in methods:
        proc = _run(cmd, cwd=repo)
        if proc.returncode != 0:
            print(
                f"    [debug] {name}: exit {proc.returncode} — {proc.stderr.strip()[:120]}"
            )
            continue
        if not proc.stdout.strip():
            print(f"    [debug] {name}: empty output")
            continue

        if fmt == "json":
            try:
                data = json.loads(proc.stdout)
                if isinstance(data, dict) and data:
                    return data
                if isinstance(data, list) and data:
                    return {p["name"].lower(): p.get("version", "") for p in data}
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"    [debug] {name}: parse error — {e}")
                continue
        else:
            result: dict[str, str] = {}
            for line in proc.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    result[parts[0].lower()] = parts[1]
            if result:
                return result

        print(f"    [debug] {name}: returned empty result")

    print(f"    [debug] All methods failed to list packages from {repo}")
    return {}


def _ort_gpu_namespace_check(installed: dict[str, str], repo: Path) -> list[Result]:
    """Validate onnxruntime-gpu / onnxruntime coexistence.

    Both packages install into the same `onnxruntime` Python namespace.
    In this project, both appear in pip list because mlserver-onnx-cuda
    depends on mlserver-onnx which depends on onnxruntime (CPU), while
    mlserver-onnx-cuda itself depends on onnxruntime-gpu.  The GPU wheel
    overwrites the CPU files, so what matters is the runtime behavior:
    the onnxruntime module must expose CUDAExecutionProvider.
    """
    results: list[Result] = []
    cpu_present = "onnxruntime" in installed
    gpu_present = "onnxruntime-gpu" in installed

    if cpu_present and gpu_present:
        results.append(
            (
                "onnxruntime namespace: both packages in pip list",
                True,
                "expected (mlserver-onnx→onnxruntime, mlserver-onnx-cuda→onnxruntime-gpu)",
            )
        )
        prov_proc = _run(
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import onnxruntime; provs = onnxruntime.get_available_providers(); "
                "print(','.join(provs))",
            ],
            cwd=repo,
        )
        if prov_proc.returncode == 0:
            has_cuda = "CUDAExecutionProvider" in prov_proc.stdout
            results.append(
                (
                    "onnxruntime namespace: GPU provider wins",
                    None if not has_cuda else True,
                    (
                        prov_proc.stdout.strip()[:80]
                        if has_cuda
                        else "CUDAExecutionProvider not available (expected on CPU-only host)"
                    ),
                )
            )
        else:
            results.append(
                (
                    "onnxruntime namespace: import check",
                    False,
                    f"import failed: {prov_proc.stderr.strip()[:80]}",
                )
            )
    elif gpu_present and not cpu_present:
        results.append(
            (
                "onnxruntime namespace: only onnxruntime-gpu",
                True,
                "clean (no CPU package)",
            )
        )

    return results


def _run_live_targets(repo: Path, state: dict, state_file: Path) -> list[Result]:
    """Category 11: Live Make Target Validation."""
    results: list[Result] = []

    if not _confirm(
        "Run live Make target validation? (will DELETE and recreate Poetry venv per target)"
    ):
        results.append(("Live Make target validation", None, "skipped by user"))
        return results

    ran_any = False

    for target, spec in _INSTALL_TARGETS.items():
        step_id = f"live_{target}"
        if _is_done(state, step_id):
            target_results = _replay_results(state, step_id)
            print(f"  {target}: (resumed from previous run)")
            for r in target_results:
                _print_result(r)
            results.extend(target_results)
            continue

        if not _confirm(
            f"  Run {target}? This will DELETE and recreate your Poetry venv."
        ):
            results.append((f"make {target}", None, "skipped by user"))
            continue

        ran_any = True
        target_results: list[Result] = []

        print(f"  Removing venv...")
        _run_live(["poetry", "env", "remove", "--all"], cwd=repo)

        print(f"  Running make {target}...")
        rc = _run_live(["make", target], cwd=repo)
        if rc != 0:
            target_results.append((f"make {target} succeeds", False, f"exit code {rc}"))
            _print_result(target_results[-1])
            _mark_done(state, step_id, target_results)
            _save_state(state_file, state)
            results.extend(target_results)
            continue
        target_results.append((f"make {target} succeeds", True, "exit 0"))
        _print_result(target_results[-1])

        installed = _get_installed_packages(repo)

        for pkg in spec["expected"]:
            present = pkg.lower() in installed
            target_results.append(
                (
                    f"  {target}: {pkg} installed",
                    present,
                    "present" if present else "MISSING",
                )
            )

        ort = spec["expected_ort"]
        target_results.append(
            (
                f"  {target}: {ort} installed",
                ort.lower() in installed,
                "present" if ort.lower() in installed else "MISSING",
            )
        )

        if ort == "onnxruntime-gpu":
            ns_results = _ort_gpu_namespace_check(installed, repo)
            for r in ns_results:
                name, passed, detail = r
                target_results.append((f"  {target}: {name}", passed, detail))

        for pkg in spec["not_expected"]:
            present = pkg.lower() in installed
            target_results.append(
                (
                    f"  {target}: {pkg} NOT installed",
                    not present,
                    "absent (correct)" if not present else "PRESENT (unexpected)",
                )
            )

        for imp in spec["imports"]:
            parts = imp.split(".")
            if len(parts) == 1:
                stmt = f"import {imp}"
            else:
                stmt = f"from {parts[0]} import {parts[1]}"
            proc = _run(["poetry", "run", "python", "-c", stmt], cwd=repo)
            target_results.append(
                (
                    f"  {target}: import {imp}",
                    proc.returncode == 0,
                    "ok" if proc.returncode == 0 else proc.stderr.strip()[:80],
                )
            )

        if target == "install-dev-odh-cuda":
            proc = _run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-c",
                    "from mlserver_onnx import OnnxModel as A; "
                    "from mlserver_onnx_cuda import OnnxModel as B; "
                    "assert A is B",
                ],
                cwd=repo,
            )
            target_results.append(
                (
                    f"  {target}: OnnxModel class identity",
                    proc.returncode == 0,
                    "same class" if proc.returncode == 0 else "MISMATCH",
                )
            )

        for r in target_results:
            _print_result(r)
        _mark_done(state, step_id, target_results)
        _save_state(state_file, state)
        results.extend(target_results)

    # bootstrap-test
    if _is_done(state, "live_bootstrap"):
        bt_results = _replay_results(state, "live_bootstrap")
        print(f"  bootstrap-test: (resumed from previous run)")
        for r in bt_results:
            _print_result(r)
        results.extend(bt_results)
    elif _confirm("  Run make bootstrap-test?"):
        bt_results: list[Result] = []
        tox_before = _read_file(repo / "runtimes" / "onnx-cuda" / "tox.ini")
        rc = _run_live(["make", "bootstrap-test"], cwd=repo)
        bt_results.append(
            (
                "make bootstrap-test succeeds",
                rc == 0,
                f"exit {rc}",
            )
        )
        tox_after = _read_file(repo / "runtimes" / "onnx-cuda" / "tox.ini")
        bt_results.append(
            (
                "onnx-cuda tox.ini preserved after bootstrap-test",
                tox_before == tox_after and "[testenv:cuda]" in tox_after,
                "preserved" if tox_before == tox_after else "OVERWRITTEN",
            )
        )
        for r in bt_results:
            _print_result(r)
        _mark_done(state, "live_bootstrap", bt_results)
        _save_state(state_file, state)
        results.extend(bt_results)

    # poetry lock freshness check
    if _is_done(state, "live_lock_check"):
        lc_results = _replay_results(state, "live_lock_check")
        print(f"  poetry lock check: (resumed from previous run)")
        for r in lc_results:
            _print_result(r)
        results.extend(lc_results)
    elif _confirm("  Run poetry lock freshness check for onnx and onnx-cuda?"):
        lc_results: list[Result] = []
        # Poetry 2.x replaced `poetry lock --check` with `poetry check --lock`
        probe = _run(["poetry", "lock", "--check"])
        if "does not exist" in probe.stderr:
            lock_cmd = ["poetry", "check", "--lock"]
        else:
            lock_cmd = ["poetry", "lock", "--check"]
        for name in ["onnx", "onnx-cuda"]:
            runtime_dir = repo / "runtimes" / name
            proc = _run(lock_cmd, cwd=runtime_dir)
            lc_results.append(
                (
                    f"poetry lock check ({name})",
                    proc.returncode == 0,
                    (
                        "in sync"
                        if proc.returncode == 0
                        else f"STALE: {proc.stderr.strip()[:80]}"
                    ),
                )
            )
        for r in lc_results:
            _print_result(r)
        _mark_done(state, "live_lock_check", lc_results)
        _save_state(state_file, state)
        results.extend(lc_results)

    # make lint
    if _is_done(state, "live_lint"):
        lint_results = _replay_results(state, "live_lint")
        print(f"  make lint: (resumed from previous run)")
        for r in lint_results:
            _print_result(r)
        results.extend(lint_results)
    elif _confirm("  Run lint (black, flake8, mypy)?"):
        lint_results: list[Result] = []
        for lint_cmd, lint_name in [
            (["poetry", "run", "black", "--check", "."], "black --check"),
            (["poetry", "run", "flake8", "."], "flake8"),
            (["poetry", "run", "mypy", "./mlserver"], "mypy mlserver"),
            (["poetry", "run", "mypy", "./runtimes/onnx"], "mypy runtimes/onnx"),
            (
                ["poetry", "run", "mypy", "./runtimes/onnx-cuda"],
                "mypy runtimes/onnx-cuda",
            ),
        ]:
            rc = _run_live(lint_cmd, cwd=repo)
            lint_results.append((f"lint: {lint_name}", rc == 0, f"exit {rc}"))
            _print_result(lint_results[-1])
        _mark_done(state, "live_lint", lint_results)
        _save_state(state_file, state)
        results.extend(lint_results)

    # Restore environment
    if ran_any:
        print()
        choice = input(
            "  Restore your dev environment? [install-dev / install-dev-odh / install-dev-odh-cuda / skip]: "
        ).strip()
        if choice in ("install-dev", "install-dev-odh", "install-dev-odh-cuda"):
            print(f"  Restoring with make {choice}...")
            _run_live(["poetry", "env", "remove", "--all"], cwd=repo)
            _run_live(["make", choice], cwd=repo)
            print(f"  Restored.")

    return results


def run_phase1(repo: Path, state: dict, state_file: Path) -> list[Result]:
    """Phase 1: All local repo checks."""
    results: list[Result] = []

    static_categories = [
        ("cat1", "Category 1: Package Structure", _cat1_package_structure),
        ("cat2", "Category 2: Version Sync", _cat2_version_sync),
        ("cat3", "Category 3: Makefile Targets", _cat3_makefile_targets),
        ("cat4", "Category 4: Tox Configuration", _cat4_tox_config),
        ("cat5", "Category 5: Dependency Groups", _cat5_dependency_groups),
        ("cat6", "Category 6: Dockerfile Consistency", _cat6_dockerfile_consistency),
        ("cat7", "Category 7: Test Infrastructure", _cat7_test_infrastructure),
        (
            "cat8",
            "Category 8: Documentation Consistency",
            _cat8_documentation_consistency,
        ),
        ("cat9", "Category 9: Import Verification", _cat9_import_verification),
    ]

    for step_id, title, func in static_categories:
        _print_section(title)
        if _is_done(state, step_id):
            cat_results = _replay_results(state, step_id)
            print(f"  (resumed from previous run)")
        else:
            cat_results = func(repo)
            _mark_done(state, step_id, cat_results)
            _save_state(state_file, state)
        for r in cat_results:
            _print_result(r)
        results.extend(cat_results)

    _print_section("Static Summary")
    _summarize(results, "Static checks")

    _print_section("Category 11: Live Make Target Validation")
    if _is_done(state, "cat11"):
        live = _replay_results(state, "cat11")
        print(f"  (resumed from previous run)")
    else:
        live = _run_live_targets(repo, state, state_file)
        _mark_done(state, "cat11", live)
        _save_state(state_file, state)
    for r in live:
        _print_result(r)
    results.extend(live)

    if any(p is not None for _, p, _ in live):
        _print_section("Live Summary")
        _summarize(live, "Live checks")

    return results


# =========================================================================
# PHASE 2 — Container Image Validation
# =========================================================================

_CONTAINER_NAME = "mlserver-validate"
_CONTAINER_IMAGE: Optional[str] = None
_USE_RUN_MODE = False

_DOCKERFILE_EXPECTATIONS = {
    "Dockerfile": {
        "is_cuda": False,
        "expected_runtimes": ["lightgbm", "onnx", "sklearn", "xgboost"],
        "ort_package": "onnxruntime",
        "label_component": None,
    },
    "Dockerfile.cuda": {
        "is_cuda": True,
        "expected_runtimes": ["onnx", "onnx-cuda"],
        "ort_package": "onnxruntime-gpu",
        "label_component": None,
    },
    "Dockerfile.konflux": {
        "is_cuda": False,
        "expected_runtimes": ["lightgbm", "onnx", "sklearn", "xgboost"],
        "ort_package": "onnxruntime",
        "label_component": "odh-mlserver-rhel9",
    },
    "Dockerfile.cuda.konflux": {
        "is_cuda": True,
        "expected_runtimes": ["onnx-cuda"],
        "ort_package": "onnxruntime-gpu",
        "label_component": "odh-mlserver-cuda-rhel9",
    },
}


def _podman_exec(cmd: str) -> subprocess.CompletedProcess:
    global _USE_RUN_MODE
    if _USE_RUN_MODE:
        return _podman_run_cmd(cmd)
    base = ["podman", "exec", "--workdir", "/", _CONTAINER_NAME]
    proc = _run(base + ["bash", "-c", cmd])
    if proc.returncode in (125, 126, 127) or (
        proc.returncode != 0
        and "not found" in (proc.stderr + proc.stdout).lower()
        and any(w in (proc.stderr + proc.stdout).lower() for w in ("bash", "exec"))
    ):
        proc = _run(base + ["sh", "-c", cmd])
    if proc.returncode in (137, 139):
        print(
            f"    (podman exec crashed with signal {proc.returncode - 128}, "
            "switching to podman-run mode)"
        )
        _USE_RUN_MODE = True
        return _podman_run_cmd(cmd)
    return proc


def _podman_run_cmd(cmd: str) -> subprocess.CompletedProcess:
    """Run a command in a fresh disposable container (fallback for exec)."""
    return _run(
        ["podman", "run", "--rm", _CONTAINER_IMAGE or "", "sh", "-c", cmd]
    )


def _container_cleanup() -> None:
    _run(["podman", "rm", "-f", _CONTAINER_NAME])


def _container_start(image: str) -> bool:
    global _CONTAINER_IMAGE, _USE_RUN_MODE
    _CONTAINER_IMAGE = image
    _USE_RUN_MODE = False
    _container_cleanup()
    proc = _run(
        [
            "podman",
            "run",
            "-d",
            "--name",
            _CONTAINER_NAME,
            "--entrypoint",
            "sleep",
            image,
            "infinity",
        ]
    )
    if proc.returncode != 0:
        print(
            f"  {_red('[FAIL]')} Failed to start container: {proc.stderr.strip()[:200]}"
        )
        return False

    import time

    time.sleep(1)
    status = _run(
        [
            "podman",
            "inspect",
            "--format",
            "{{.State.Status}}",
            _CONTAINER_NAME,
        ]
    )
    state = status.stdout.strip() if status.returncode == 0 else "unknown"
    if state != "running":
        print(
            f"  {_yellow('[WARN]')} Container not running (state={state}), "
            "using podman-run fallback for each check."
        )
        _USE_RUN_MODE = True
    return True


def run_phase2(image: str, dockerfile: str) -> list[Result]:
    """Phase 2: Container Image Validation."""
    results: list[Result] = []

    if dockerfile not in _DOCKERFILE_EXPECTATIONS:
        results.append(
            (
                "Dockerfile recognized",
                False,
                f"{dockerfile} not in {list(_DOCKERFILE_EXPECTATIONS.keys())}",
            )
        )
        return results

    spec = _DOCKERFILE_EXPECTATIONS[dockerfile]
    is_cuda = spec["is_cuda"]

    print(f"\n  Starting container from {image}...")
    if not _container_start(image):
        return results

    try:
        # Sanity-check: confirm bash and pip are reachable inside the container
        sanity = _podman_exec("pip --version")
        if sanity.returncode != 0:
            detail = sanity.stderr.strip()[:120] or sanity.stdout.strip()[:120]
            results.append(
                ("pip reachable in container", False, f"rc={sanity.returncode}: {detail}")
            )
            for r in results:
                _print_result(*r)
            return results

        # Package presence
        for pkg in ["mlserver", "mlserver-onnx"]:
            proc = _podman_exec(f"pip show {pkg}")
            if proc.returncode == 0:
                detail = "installed"
            else:
                err = proc.stderr.strip()[:80]
                detail = f"MISSING{' — ' + err if err else ''}"
            results.append((f"pip show {pkg}", proc.returncode == 0, detail))

        ort = spec["ort_package"]
        proc = _podman_exec(f"pip show {ort}")
        if proc.returncode == 0:
            detail = "installed"
        else:
            err = proc.stderr.strip()[:80]
            detail = f"MISSING{' — ' + err if err else ''}"
        results.append((f"pip show {ort}", proc.returncode == 0, detail))

        # Import verification
        proc = _podman_exec('python3 -c "from mlserver_onnx import OnnxModel"')
        results.append(
            (
                "import mlserver_onnx.OnnxModel",
                proc.returncode == 0,
                "ok" if proc.returncode == 0 else proc.stderr.strip()[:80],
            )
        )

        # CUDA provider
        if is_cuda:
            proc = _podman_exec(
                'python3 -c "import onnxruntime; providers = onnxruntime.get_available_providers(); assert \\"CUDAExecutionProvider\\" in providers, providers"'
            )
            results.append(
                (
                    "CUDAExecutionProvider available",
                    proc.returncode == 0,
                    (
                        "available"
                        if proc.returncode == 0
                        else f"not available: {proc.stderr.strip()[:80]}"
                    ),
                )
            )

        # Trusted runtimes
        proc = _podman_exec(
            "test -f /etc/mlserver/trusted-runtimes.json && cat /etc/mlserver/trusted-runtimes.json"
        )
        if proc.returncode == 0:
            results.append(
                ("trusted-runtimes.json exists", True, proc.stdout.strip()[:80])
            )
            perm_proc = _podman_exec(
                'python3 -c "import os; print(oct(os.stat(\\"/etc/mlserver/trusted-runtimes.json\\").st_mode & 0o777))"'
            )
            if perm_proc.returncode == 0:
                perm = perm_proc.stdout.strip()
                results.append(
                    (
                        "trusted-runtimes.json permissions",
                        perm == "0o444",
                        f"mode={perm}",
                    )
                )
        else:
            results.append(("trusted-runtimes.json exists", False, "not found"))

        # Non-root user (image metadata)
        inspect_target = _CONTAINER_NAME if not _USE_RUN_MODE else image
        inspect_proc = _run(
            ["podman", "inspect", "--format", "{{.Config.User}}", inspect_target]
        )
        if inspect_proc.returncode == 0:
            user_val = inspect_proc.stdout.strip()
            results.append(
                ("image USER (podman inspect)", user_val == "1000", f"User={user_val}")
            )

        # Non-root user (runtime)
        id_proc = _podman_exec("id -u")
        if id_proc.returncode == 0:
            uid = id_proc.stdout.strip()
            results.append(("runtime UID (id -u)", uid == "1000", f"UID={uid}"))

        # Pip cache
        proc = _podman_exec("test -d /root/.cache/pip")
        results.append(
            (
                "pip cache cleaned",
                proc.returncode != 0,
                (
                    "absent (correct)"
                    if proc.returncode != 0
                    else "EXISTS (should be cleaned)"
                ),
            )
        )

        # Environment variables
        for var, expected in [
            ("MLSERVER_MODELS_DIR", "/mnt/models"),
            ("MLSERVER_PATH", "/opt/mlserver"),
        ]:
            proc = _podman_exec(f"printenv {var}")
            val = proc.stdout.strip() if proc.returncode == 0 else ""
            results.append(
                (f"ENV {var}", val == expected, f"{val}" if val else "not set")
            )

        if is_cuda:
            proc = _podman_exec("printenv MLSERVER_MODEL_ONNX_PROVIDERS")
            val = proc.stdout.strip() if proc.returncode == 0 else ""
            results.append(
                (
                    "ENV MLSERVER_MODEL_ONNX_PROVIDERS",
                    "CUDAExecutionProvider" in val,
                    val[:80] if val else "not set",
                )
            )

        # Labels
        if spec["label_component"]:
            label_proc = _run(
                [
                    "podman",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "com.redhat.component"}}',
                    inspect_target,
                ]
            )
            label_val = label_proc.stdout.strip() if label_proc.returncode == 0 else ""
            results.append(
                (
                    "label com.redhat.component",
                    label_val == spec["label_component"],
                    f"{label_val} (expected {spec['label_component']})",
                )
            )

        # License files
        for lpath in ["/opt/mlserver/license.txt", "/licenses/license.txt"]:
            proc = _podman_exec(f"test -f {lpath}")
            results.append(
                (
                    f"license file {lpath}",
                    proc.returncode == 0,
                    "exists" if proc.returncode == 0 else "MISSING",
                )
            )

        # CUDA-specific: no onnxruntime (CPU) installed
        if is_cuda:
            proc = _podman_exec(
                "pip show onnxruntime 2>/dev/null && echo FOUND || echo NOTFOUND"
            )
            output = proc.stdout.strip()
            results.append(
                (
                    "no onnxruntime (CPU) in CUDA image",
                    "NOTFOUND" in output,
                    (
                        "absent (correct)"
                        if "NOTFOUND" in output
                        else "PRESENT (namespace conflict)"
                    ),
                )
            )

    finally:
        if not _USE_RUN_MODE:
            _container_cleanup()

    return results


# =========================================================================
# PHASE 3 — CUDA Node Validation
# =========================================================================


def _setup_nvidia_ld_path(repo: Path) -> None:
    """Discover pip-installed NVIDIA CUDA libs and add to LD_LIBRARY_PATH.

    Same logic as the pytest_configure hook in conftest.py, but for
    non-pytest contexts (validation script probes, smoke tests).
    """
    proc = _run([
        "poetry", "run", "python", "-c",
        "import importlib\n"
        "pkgs = ['nvidia.cublas.lib','nvidia.cudnn.lib','nvidia.cuda_runtime.lib',"
        "'nvidia.nvjitlink.lib','nvidia.cufft.lib','nvidia.curand.lib',"
        "'nvidia.cusolver.lib','nvidia.cusparse.lib','nvidia.cuda_nvrtc.lib']\n"
        "paths = []\n"
        "for p in pkgs:\n"
        "    try: paths.extend(importlib.import_module(p).__path__)\n"
        "    except Exception: pass\n"
        "print(':'.join(paths))"
    ], cwd=repo)
    if proc.returncode == 0 and proc.stdout.strip():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new = proc.stdout.strip()
        os.environ["LD_LIBRARY_PATH"] = f"{new}:{existing}" if existing else new
        print(f"  Set LD_LIBRARY_PATH for NVIDIA CUDA libs")


def run_phase3(repo: Path, state: dict, state_file: Path) -> list[Result]:
    """Phase 3: CUDA Node Validation."""
    results: list[Result] = []

    _print_section("Phase 3: CUDA Node Validation")

    _setup_nvidia_ld_path(repo)

    # Prerequisites (always re-checked — these are fast hardware queries)
    print("  Checking prerequisites...\n")

    nvidia_proc = _run(["nvidia-smi", "-L"])
    if nvidia_proc.returncode != 0:
        results.append(
            ("nvidia-smi available", False, "not found — Phase 3 requires GPU hardware")
        )
        print(f"  {_red('[FAIL]')} nvidia-smi not available. Phase 3 cannot continue.")
        return results

    gpu_lines = [
        l for l in nvidia_proc.stdout.strip().split("\n") if l.strip().startswith("GPU")
    ]
    gpu_count = len(gpu_lines)
    results.append(("nvidia-smi available", True, gpu_lines[0] if gpu_lines else ""))
    results.append(("CUDA GPUs detected", gpu_count > 0, f"{gpu_count} GPU(s)"))

    if gpu_count == 0:
        print(f"  {_red('[FAIL]')} No GPUs detected. Phase 3 cannot continue.")
        return results

    driver_proc = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    driver = (
        driver_proc.stdout.strip().split("\n")[0]
        if driver_proc.returncode == 0
        else "unknown"
    )
    results.append(("CUDA driver version", True, driver))

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    has_cuda_libs = "cuda" in ld_path.lower()
    if has_cuda_libs:
        results.append(("LD_LIBRARY_PATH includes CUDA", True, ld_path[:80]))
    else:
        results.append(
            (
                "LD_LIBRARY_PATH includes CUDA",
                None,
                f"WARN: {ld_path[:80]} — make install-dev-odh-cuda will set this up",
            )
        )

    ort_check = _run(
        [
            "poetry",
            "run",
            "python",
            "-c",
            "import onnxruntime; print(onnxruntime.get_available_providers())",
        ],
        cwd=repo,
    )
    if ort_check.returncode == 0 and "CUDAExecutionProvider" in ort_check.stdout:
        results.append(
            (
                "CUDAExecutionProvider available (pre-install)",
                True,
                ort_check.stdout.strip()[:80],
            )
        )
    else:
        results.append(
            (
                "CUDAExecutionProvider available (pre-install)",
                None,
                "WARN: not yet available — install step below will provide it",
            )
        )

    for r in results:
        _print_result(r)

    # Dev environment setup
    step_id = "p3_dev_install"
    if _is_done(state, step_id):
        dev_results = _replay_results(state, step_id)
        print(f"\n  CUDA dev install: (resumed from previous run)")
        for r in dev_results:
            _print_result(r)
        results.extend(dev_results)
    elif _confirm("\n  Run CUDA dev install (make install-dev-odh-cuda)?"):
        dev_results: list[Result] = []
        print("  Removing venv...")
        _run_live(["poetry", "env", "remove", "--all"], cwd=repo)
        print("  Running make install-dev-odh-cuda...")
        rc = _run_live(["make", "install-dev-odh-cuda"], cwd=repo)
        dev_results.append(("make install-dev-odh-cuda", rc == 0, f"exit {rc}"))

        if rc == 0:
            _setup_nvidia_ld_path(repo)
            installed = _get_installed_packages(repo)

            ort_gpu = "onnxruntime-gpu" in installed
            dev_results.append(
                (
                    "onnxruntime-gpu installed",
                    ort_gpu,
                    "present" if ort_gpu else "MISSING",
                )
            )

            ns_results = _ort_gpu_namespace_check(installed, repo)
            dev_results.extend(ns_results)

            imp_proc = _run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-c",
                    "from mlserver_onnx_cuda import OnnxModel",
                ],
                cwd=repo,
            )
            dev_results.append(
                (
                    "import mlserver_onnx_cuda",
                    imp_proc.returncode == 0,
                    "ok" if imp_proc.returncode == 0 else imp_proc.stderr.strip()[:80],
                )
            )

            prov_proc = _run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-c",
                    "import onnxruntime; print(onnxruntime.get_available_providers())",
                ],
                cwd=repo,
            )
            has_cuda_prov = (
                prov_proc.returncode == 0
                and "CUDAExecutionProvider" in prov_proc.stdout
            )
            dev_results.append(
                (
                    "CUDAExecutionProvider in providers",
                    has_cuda_prov,
                    (
                        prov_proc.stdout.strip()[:80]
                        if prov_proc.returncode == 0
                        else "failed"
                    ),
                )
            )

            dev_proc = _run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-c",
                    "import onnxruntime; print(onnxruntime.get_device())",
                ],
                cwd=repo,
            )
            is_gpu = dev_proc.returncode == 0 and dev_proc.stdout.strip() == "GPU"
            dev_results.append(
                (
                    "onnxruntime.get_device() == GPU",
                    is_gpu,
                    dev_proc.stdout.strip() if dev_proc.returncode == 0 else "failed",
                )
            )

        for r in dev_results:
            _print_result(r)
        _mark_done(state, step_id, dev_results)
        _save_state(state_file, state)
        results.extend(dev_results)

    # Hardware probe
    step_id = "p3_hw_probe"
    if _is_done(state, step_id):
        probe_results = _replay_results(state, step_id)
        print(f"\n  Hardware probe: (resumed from previous run)")
        for r in probe_results:
            _print_result(r)
        results.extend(probe_results)
    elif _confirm("\n  Run _has_cuda() hardware probe?"):
        probe_results: list[Result] = []
        probe_code = """
try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("NO_PROVIDER")
        raise SystemExit(0)
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "cuda_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 7
    model_bytes = model_proto.SerializeToString()
    sess = ort.InferenceSession(model_bytes, providers=["CUDAExecutionProvider"])
    active = sess.get_providers()
    if "CUDAExecutionProvider" in active:
        print("CUDA_OK")
    else:
        print(f"CUDA_FAIL: session used {active} — CUDA libs (libcublasLt.so.12 etc.) not loadable")
except Exception as e:
    print(f"CUDA_FAIL: {e}")
"""
        proc = _run(["poetry", "run", "python", "-c", probe_code], cwd=repo)
        output = proc.stdout.strip()
        if "CUDA_OK" in output:
            probe_results.append(("_has_cuda() probe", True, "GPU hardware confirmed"))
        elif "NO_PROVIDER" in output:
            probe_results.append(
                ("_has_cuda() probe", False, "CUDAExecutionProvider not in providers")
            )
        else:
            probe_results.append(("_has_cuda() probe", False, output[:120]))
        for r in probe_results:
            _print_result(r)
        _mark_done(state, step_id, probe_results)
        _save_state(state_file, state)
        results.extend(probe_results)

    # Test execution — CPU subset
    step_id = "p3_cpu_tests"
    if _is_done(state, step_id):
        cpu_results = _replay_results(state, step_id)
        print(f"\n  CPU test subset: (resumed from previous run)")
        for r in cpu_results:
            _print_result(r)
        results.extend(cpu_results)
    elif _confirm("\n  Run CPU test subset (tox -c ./runtimes/onnx-cuda)?"):
        cpu_results: list[Result] = []
        rc = _run_live(["poetry", "run", "tox", "-c", "./runtimes/onnx-cuda"], cwd=repo)
        cpu_results.append(
            ("CPU test subset (tox onnx-cuda default)", rc == 0, f"exit {rc}")
        )
        for r in cpu_results:
            _print_result(r)
        _mark_done(state, step_id, cpu_results)
        _save_state(state_file, state)
        results.extend(cpu_results)

    # Test execution — GPU suite
    step_id = "p3_gpu_tests"
    if _is_done(state, step_id):
        gpu_results = _replay_results(state, step_id)
        print(f"\n  GPU test suite: (resumed from previous run)")
        for r in gpu_results:
            _print_result(r)
        results.extend(gpu_results)
    elif _confirm("\n  Run GPU test suite (make test-cuda)?"):
        gpu_results: list[Result] = []
        rc = _run_live(["make", "test-cuda"], cwd=repo)
        gpu_results.append(("make test-cuda", rc == 0, f"exit {rc}"))
        for r in gpu_results:
            _print_result(r)
        _mark_done(state, step_id, gpu_results)
        _save_state(state_file, state)
        results.extend(gpu_results)

    # Smoke test
    step_id = "p3_smoke"
    if _is_done(state, step_id):
        smoke_results = _replay_results(state, step_id)
        print(f"\n  Smoke test: (resumed from previous run)")
        for r in smoke_results:
            _print_result(r)
        results.extend(smoke_results)
    elif _confirm("\n  Run ONNX CUDA inference smoke test?"):
        smoke_results: list[Result] = []
        smoke_code = """
import sys
try:
    import onnx
    v = tuple(int(x) for x in onnx.__version__.split(".")[:2])
    if v < (1, 21):
        print(f"SKIP: onnx {onnx.__version__} < 1.21.0")
        sys.exit(0)
except ImportError:
    print("SKIP: onnx not installed")
    sys.exit(0)

import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
ones = numpy_helper.from_array(np.ones((2, 4), dtype=np.float32), name="ones")
add_node = helper.make_node("Add", ["X", "ones"], ["Y"])
graph = helper.make_graph([add_node], "smoke", [X], [Y], [ones])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7

sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
providers = sess.get_providers()
inp = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)
result = sess.run(None, {"X": inp})[0]
expected = inp + 1

if not np.allclose(result, expected):
    print(f"FAIL: result mismatch {result} vs {expected}")
    sys.exit(1)

if "CUDAExecutionProvider" not in providers:
    print(f"FAIL: providers={providers}")
    sys.exit(1)

print("SMOKE_OK")
"""
        proc = _run(["poetry", "run", "python", "-c", smoke_code], cwd=repo)
        output = proc.stdout.strip()
        if "SMOKE_OK" in output:
            smoke_results.append(
                ("CUDA inference smoke test", True, "add-one model passed")
            )
        elif "SKIP" in output:
            smoke_results.append(("CUDA inference smoke test", None, output))
        else:
            smoke_results.append(
                (
                    "CUDA inference smoke test",
                    False,
                    (output + " " + proc.stderr.strip())[:120],
                )
            )
        for r in smoke_results:
            _print_result(r)
        _mark_done(state, step_id, smoke_results)
        _save_state(state_file, state)
        results.extend(smoke_results)

    # Multi-GPU check
    step_id = "p3_multi_gpu"
    if _is_done(state, step_id):
        mg_results = _replay_results(state, step_id)
        print(f"\n  Multi-GPU: (resumed from previous run)")
        for r in mg_results:
            _print_result(r)
        results.extend(mg_results)
    elif gpu_count > 1:
        _print_section("Multi-GPU Checks")
        mg_results: list[Result] = []
        restrict_proc = _run(
            ["nvidia-smi", "-L"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        if restrict_proc.returncode == 0:
            restricted_lines = [
                l
                for l in restrict_proc.stdout.strip().split("\n")
                if l.strip().startswith("GPU")
            ]
            mg_results.append(
                (
                    "CUDA_VISIBLE_DEVICES=0 restricts to 1 GPU",
                    len(restricted_lines) == 1,
                    f"{len(restricted_lines)} GPU(s) visible",
                )
            )

        device_code = """
import onnxruntime as ort, onnx
from onnx import helper, TensorProto
node = helper.make_node("Identity", ["x"], ["y"])
graph = helper.make_graph([node], "dev_test",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()
sess = ort.InferenceSession(model_bytes,
    providers=[("CUDAExecutionProvider", {"device_id": "DEVICE_ID"})],
)
print("DEV_OK")
"""
        for dev_id in ["0", "1"]:
            code = device_code.replace("DEVICE_ID", dev_id)
            proc = _run(["poetry", "run", "python", "-c", code], cwd=repo)
            ok = proc.returncode == 0 and "DEV_OK" in proc.stdout
            mg_results.append(
                (
                    f"device_id={dev_id} InferenceSession",
                    ok,
                    "ok" if ok else proc.stderr.strip()[:80],
                )
            )
        for r in mg_results:
            _print_result(r)
        _mark_done(state, step_id, mg_results)
        _save_state(state_file, state)
        results.extend(mg_results)
    elif gpu_count == 1:
        results.append(("Multi-GPU checks", None, "only 1 GPU — skipped"))
        _print_result(results[-1])

    return results


# =========================================================================
# Main entry point
# =========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ONNX / ONNX-CUDA Validation Script for MLServer",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        default=None,
        help="Skip phase selection prompt",
    )
    parser.add_argument("--repo", default=None, help="Repository root path (Phase 1/3)")
    parser.add_argument("--image", default=None, help="Container image for Phase 2")
    parser.add_argument(
        "--dockerfile", default=None, help="Dockerfile name for Phase 2"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping completed steps",
    )
    parser.add_argument(
        "--clear-state", action="store_true", help="Clear saved state and start fresh"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Stop on first failure (default: enabled)",
    )
    parser.add_argument(
        "--no-fail-fast", action="store_true", help="Continue running after failures"
    )
    args = parser.parse_args()

    global _FAIL_FAST
    _FAIL_FAST = not args.no_fail_fast

    state_file = _STATE_FILE
    if args.clear_state:
        if state_file.exists():
            state_file.unlink()
            print("State cleared.\n")

    state = _load_state(state_file) if args.resume else {"completed": {}, "results": {}}
    if args.resume and state["completed"]:
        print(
            f"=== ONNX / ONNX-CUDA Validation (resuming — {len(state['completed'])} steps cached) ===\n"
        )
    else:
        print("=== ONNX / ONNX-CUDA Validation ===\n")

    if args.phase is None:
        print(
            "Phase 1: Local repo checks (static file checks + optional Make target validation)"
        )
        print("Phase 2: Container image validation (run per image via podman)")
        print("Phase 3: CUDA node validation (GPU hardware required)")
        print()
        try:
            phase = input("Select phase [1/2/3/all]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
    else:
        phase = args.phase

    all_results: list[Result] = []
    run_phases = set()
    if phase in ("1", "all"):
        run_phases.add(1)
    if phase in ("2", "all"):
        run_phases.add(2)
    if phase in ("3", "all"):
        run_phases.add(3)

    if not run_phases:
        print(f"Unknown phase: {phase}")
        return 1

    try:
        # Phase 1
        if 1 in run_phases:
            repo_path = args.repo
            if repo_path is None:
                try:
                    repo_path = input("Repo path [.]: ").strip() or "."
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
            repo = Path(repo_path).resolve()
            if not (repo / "Makefile").exists():
                print(
                    f"Error: {repo} does not look like an MLServer repo (no Makefile)"
                )
                return 1

            print(f"\nRunning Phase 1 against {repo}...")
            results = run_phase1(repo, state, state_file)
            _save_state(state_file, state)
            all_results.extend(results)

        # Phase 2
        if 2 in run_phases:
            while True:
                image = args.image
                dockerfile = args.dockerfile

                if image is None:
                    try:
                        image = input(
                            "\nContainer image (e.g. quay.io/opendatahub/mlserver:odh-stable): "
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                if not image:
                    break

                if dockerfile is None:
                    print(
                        "Dockerfile options: Dockerfile, Dockerfile.cuda, Dockerfile.konflux, Dockerfile.cuda.konflux"
                    )
                    try:
                        dockerfile = input("Dockerfile name: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break

                if not dockerfile:
                    break

                p2_key = f"p2_{dockerfile}_{image}"
                if _is_done(state, p2_key):
                    _print_section(f"Phase 2: Container Validation ({image})")
                    results = _replay_results(state, p2_key)
                    print(f"  (resumed from previous run)")
                else:
                    _print_section(f"Phase 2: Container Validation ({image})")
                    results = run_phase2(image, dockerfile)
                    _mark_done(state, p2_key, results)
                    _save_state(state_file, state)
                for r in results:
                    _print_result(r)
                all_results.extend(results)

                _print_section("Container Summary")
                _summarize(results, f"Container ({dockerfile})")

                # Reset for next iteration
                args.image = None
                args.dockerfile = None

                if not _confirm("\nValidate another image?"):
                    break

        # Phase 3
        if 3 in run_phases:
            repo_path = args.repo
            if repo_path is None:
                try:
                    repo_path = input("\nRepo path [.]: ").strip() or "."
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
            repo = Path(repo_path).resolve()

            results = run_phase3(repo, state, state_file)
            _save_state(state_file, state)
            all_results.extend(results)

            _print_section("Phase 3 Summary")
            _summarize(results, "CUDA Node")

    except FailFastExit as e:
        name, _, detail = e.result
        print(f"\n  {_red('STOPPED')}: --fail-fast triggered on: {name}")
        print(f"  Fix the issue above, then re-run with --resume to continue.\n")
        _save_state(state_file, state)
        return 1

    # Final summary
    if len(run_phases) > 1 or (all_results and len(run_phases) == 1):
        _print_section("Overall Summary")
        ok = _summarize(all_results, "All phases")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
