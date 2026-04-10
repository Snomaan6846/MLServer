"""
Root test configuration shared across both `tests/` and `runtimes/*` suites.

This file must live at repository root so pytest applies the trusted-runtimes
fixture to runtime-specific tests invoked from tox (e.g. `runtimes/sklearn`),
which do not inherit from `tests/conftest.py`.

## Test Mode Architecture

**Global Default (PRODUCTION mode):**
The pytest_configure hook sets up PRODUCTION mode globally for all tests by
creating a trusted-runtimes.json artifact that includes:
- All built-in runtimes (mlserver_sklearn, mlserver_xgboost, etc.)
- Test-only fixtures (tests.fixtures.SumModel, etc.)

This means tests run in PRODUCTION mode by default without needing any fixture.

**Fixture Overrides:**
Individual tests can override the global default using fixtures:
- `development_mode` - Override to DEVELOPMENT mode (no runtime restrictions)
- `empty_allowlist_mode` - Override to PRODUCTION mode with empty allowlist

Tests that don't specify a fixture use the global PRODUCTION mode default.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

import mlserver.settings as mlserver_settings

TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV = "MLSERVER_TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH"
REPO_ROOT = str(Path(__file__).resolve().parent)

_TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH = None
_ORIGINAL_GET_TRUSTED_RUNTIMES_ARTIFACT_PATH = (
    mlserver_settings._get_trusted_runtimes_artifact_path
)
_TEST_BOOTSTRAP_DIR = None
_ORIGINAL_PYTHONPATH = None
_ORIGINAL_PYTHONHOME = None


TEST_ONLY_EXTRA_IMPLEMENTATIONS = {
    # Core repo test fixtures.
    "tests.fixtures.SumModel",
    "tests.fixtures.SlowModel",
    "tests.fixtures.SimpleModel",
    "tests.fixtures.ErrorModel",
    "tests.fixtures.EnvModel",
    "tests.fixtures.EchoModel",
    "tests.fixtures.TextModel",
    "tests.fixtures.TextStreamModel",
    "tests.metrics.test_custom.CustomMetricsModel",
    "fixtures.SumModel",
    "env_models.DummySKLearnModel",
}


def _clear_trusted_runtimes_caches() -> None:
    mlserver_settings.clear_trusted_runtime_caches()


def _apply_trusted_runtimes_override(artifact_path: str) -> None:
    mlserver_settings._get_trusted_runtimes_artifact_path = lambda: artifact_path
    _clear_trusted_runtimes_caches()


@pytest.fixture(autouse=True)
def clear_trusted_runtimes_caches_between_tests():
    _clear_trusted_runtimes_caches()
    yield
    _clear_trusted_runtimes_caches()


@pytest.fixture
def development_mode(monkeypatch, tmp_path):
    """Override global PRODUCTION mode default to use DEVELOPMENT mode.

    Points to a non-existent trusted-runtimes.json file, simulating DEVELOPMENT mode
    where all runtimes are allowed and dynamic loading is permitted.

    Use this fixture when you need to test development-mode-specific behavior.
    """
    non_existent = tmp_path / "does-not-exist.json"
    # Set env var for spawned processes (read by sitecustomize.py)
    monkeypatch.setenv(
        TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
        str(non_existent),
    )
    # Set in-process override for current test
    monkeypatch.setattr(
        mlserver_settings,
        "_get_trusted_runtimes_artifact_path",
        lambda: str(non_existent),
    )
    mlserver_settings.clear_trusted_runtime_caches()
    yield
    mlserver_settings.clear_trusted_runtime_caches()


@pytest.fixture
def empty_allowlist_mode(monkeypatch, tmp_path):
    """Override global PRODUCTION mode default to use an empty allowlist.

    Creates a trusted-runtimes.json file with an empty list, simulating PRODUCTION
    mode where no runtimes are explicitly allowed (edge case for testing).

    Use this fixture to test edge cases where the allowlist is empty.
    """
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text("[]", encoding="utf-8")
    # Set env var for spawned processes (read by sitecustomize.py)
    monkeypatch.setenv(
        TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
        str(artifact_path),
    )
    # Set in-process override for current test
    monkeypatch.setattr(
        mlserver_settings,
        "_get_trusted_runtimes_artifact_path",
        lambda: str(artifact_path),
    )
    mlserver_settings.clear_trusted_runtime_caches()
    yield
    mlserver_settings.clear_trusted_runtime_caches()


def _configure_spawned_python_bootstrap() -> None:
    """Configure sitecustomize.py to inject test configuration into spawned workers.

    Problem: Spawned worker processes (multiprocessing.Process) don't inherit
    in-memory monkeypatch changes from the parent process.

    Solution: Create sitecustomize.py (Python's automatic startup hook) that:
    1. Reads TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV environment variable
    2. Overrides _get_trusted_runtimes_artifact_path() to return that path
    3. Runs automatically in every spawned Python process (via PYTHONPATH)

    This allows tests to control which trusted-runtimes.json file workers use
    by setting the environment variable. Workers inherit environment variables
    but not in-memory state, so this bridges the gap.
    """
    global _TEST_BOOTSTRAP_DIR
    global _ORIGINAL_PYTHONPATH
    global _ORIGINAL_PYTHONHOME
    _ORIGINAL_PYTHONPATH = os.environ.get("PYTHONPATH")
    _ORIGINAL_PYTHONHOME = os.environ.get("PYTHONHOME")
    _TEST_BOOTSTRAP_DIR = tempfile.mkdtemp(prefix="mlserver-test-bootstrap-")
    bootstrap_file = os.path.join(_TEST_BOOTSTRAP_DIR, "sitecustomize.py")
    # sitecustomize.py is a Python startup hook imported automatically when
    # present on PYTHONPATH. We use it to inject the trusted-runtime artifact
    # override into spawned worker processes before MLServer modules are used.
    bootstrap_code = (
        "import os\n"
        f"artifact = os.environ.get('{TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV}')\n"
        "if artifact:\n"
        "    import mlserver.settings as settings\n"
        "    settings._get_trusted_runtimes_artifact_path = lambda: artifact\n"
        "    settings.clear_trusted_runtime_caches()\n"
    )
    with open(bootstrap_file, "w", encoding="utf-8") as f:
        f.write(bootstrap_code)

    # This intentionally mutates process-wide env for the full test session so
    # all spawned Python subprocesses (e.g. multiprocessing spawn workers) load
    # the same trusted-runtime bootstrap deterministically.
    # Placing _TEST_BOOTSTRAP_DIR first on PYTHONPATH intentionally makes this
    # test-only sitecustomize.py take precedence over any pre-existing
    # sitecustomize.py in the environment; we restore PYTHONPATH in teardown.
    # Also scrub ambient import state to avoid non-hermetic PYTHONHOME/PYTHONPATH
    # leakage from developer shells or CI hosts.
    os.environ.pop("PYTHONHOME", None)
    os.environ["PYTHONPATH"] = _TEST_BOOTSTRAP_DIR + os.pathsep + REPO_ROOT


def _cleanup_spawned_python_bootstrap() -> None:
    global _TEST_BOOTSTRAP_DIR
    global _ORIGINAL_PYTHONPATH
    global _ORIGINAL_PYTHONHOME
    if _ORIGINAL_PYTHONPATH is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = _ORIGINAL_PYTHONPATH
    if _ORIGINAL_PYTHONHOME is None:
        os.environ.pop("PYTHONHOME", None)
    else:
        os.environ["PYTHONHOME"] = _ORIGINAL_PYTHONHOME
    if _TEST_BOOTSTRAP_DIR and os.path.isdir(_TEST_BOOTSTRAP_DIR):
        shutil.rmtree(_TEST_BOOTSTRAP_DIR, ignore_errors=True)
    _TEST_BOOTSTRAP_DIR = None
    _ORIGINAL_PYTHONPATH = None
    _ORIGINAL_PYTHONHOME = None


def _configure_test_trusted_runtimes_artifact() -> None:
    """Create global trusted-runtimes.json artifact for PRODUCTION mode default.

    This creates the artifact that all tests use by default (PRODUCTION mode with
    comprehensive allowlist including built-in runtimes and test fixtures).
    """
    global _TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH
    test_allowed_model_implementations = (
        mlserver_settings.ALLOWED_MODEL_IMPLEMENTATIONS.union(
            TEST_ONLY_EXTRA_IMPLEMENTATIONS
        )
    )

    fd, artifact_path = tempfile.mkstemp(
        prefix="trusted-runtimes-",
        suffix=".json",
    )
    os.close(fd)
    with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        artifact_file.write(
            json.dumps(sorted(test_allowed_model_implementations)),
        )

    _TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH = artifact_path
    os.environ[TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV] = artifact_path
    _apply_trusted_runtimes_override(artifact_path)
    _configure_spawned_python_bootstrap()


def _cleanup_test_trusted_runtimes_artifact() -> None:
    global _TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH
    artifact_path = _TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH
    mlserver_settings._get_trusted_runtimes_artifact_path = (
        _ORIGINAL_GET_TRUSTED_RUNTIMES_ARTIFACT_PATH
    )
    if artifact_path and os.path.isfile(artifact_path):
        os.remove(artifact_path)
    os.environ.pop(TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV, None)
    _TEST_TRUSTED_RUNTIMES_ARTIFACT_PATH = None
    _clear_trusted_runtimes_caches()
    _cleanup_spawned_python_bootstrap()


def pytest_configure(config: pytest.Config) -> None:
    """Set up global PRODUCTION mode default for all tests.

    This hook runs before test collection and configures PRODUCTION mode globally
    by creating a trusted-runtimes.json artifact with all built-in runtimes and
    test fixtures. This means:

    - All tests run in PRODUCTION mode by default (no fixture needed)
    - Individual tests can override using `development_mode` or `empty_allowlist_mode`
    - Import-time ModelSettings validations work correctly
    - Works for both `tests/` and `runtimes/*` test suites
    """
    _configure_test_trusted_runtimes_artifact()


def pytest_unconfigure(config: pytest.Config) -> None:
    _cleanup_test_trusted_runtimes_artifact()
