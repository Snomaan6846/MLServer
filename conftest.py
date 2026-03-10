"""
Root test configuration shared across both `tests/` and `runtimes/*` suites.

This file must live at repository root so pytest applies the trusted-runtimes
fixture to runtime-specific tests invoked from tox (e.g. `runtimes/sklearn`),
which do not inherit from `tests/conftest.py`.
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


def _configure_spawned_python_bootstrap() -> None:
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
    # Run before test collection so import-time ModelSettings validations can use
    # the test trusted-runtimes artifact.
    # Ownership note: this root hook configures trusted-runtime state globally
    # for both `tests/` and `runtimes/*` suites.
    _configure_test_trusted_runtimes_artifact()


def pytest_unconfigure(config: pytest.Config) -> None:
    _cleanup_test_trusted_runtimes_artifact()
