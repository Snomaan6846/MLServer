"""Tests for the pytest_configure CUDA LD_LIBRARY_PATH hook in conftest.py.

These tests run on both CPU and GPU environments to verify the hook
is resilient and correct regardless of whether NVIDIA pip packages
are installed.

The hook runs before test collection (via conftest.py pytest_configure),
so by the time these tests execute LD_LIBRARY_PATH has already been set.
"""

import importlib
import os
from pathlib import Path

from conftest import NVIDIA_LIB_MODULES


def _importable_nvidia_modules() -> dict[str, list[str]]:
    """Return {module_name: [lib_paths]} for importable NVIDIA packages."""
    result: dict[str, list[str]] = {}
    for pkg in NVIDIA_LIB_MODULES:
        try:
            mod = importlib.import_module(pkg)
            result[pkg] = list(mod.__path__)
        except (ImportError, ModuleNotFoundError):
            pass
    return result


class TestCudaLdLibraryPathHook:
    """Validate that conftest.py pytest_configure correctly sets LD_LIBRARY_PATH."""

    def test_hook_ran_without_error(self):
        """The hook completed (we would not reach this test otherwise)."""

    def test_nvidia_paths_present_when_packages_installed(self):
        """Every importable NVIDIA lib path is in LD_LIBRARY_PATH."""
        nvidia_modules = _importable_nvidia_modules()
        if not nvidia_modules:
            return

        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        assert ld_path, (
            "LD_LIBRARY_PATH is empty but NVIDIA packages are installed: "
            f"{list(nvidia_modules.keys())}"
        )

        for pkg_name, paths in nvidia_modules.items():
            for p in paths:
                assert p in ld_path, (
                    f"Expected {p} (from {pkg_name}) in LD_LIBRARY_PATH.\n"
                    f"Actual: {ld_path}"
                )

    def test_nvidia_lib_paths_contain_shared_objects(self):
        """Every NVIDIA lib path contains .so files.

        Catches regressions where a module's __path__ changes to a
        directory that no longer holds shared libraries, making the
        LD_LIBRARY_PATH entry useless.
        """
        nvidia_modules = _importable_nvidia_modules()
        if not nvidia_modules:
            return

        for pkg_name, paths in nvidia_modules.items():
            for p in paths:
                so_files = list(Path(p).glob("*.so*"))
                assert so_files, (
                    f"{pkg_name} path {p} contains no .so files — "
                    f"LD_LIBRARY_PATH entry would be useless"
                )

    def test_no_nvidia_paths_when_packages_absent(self):
        """On CPU-only envs, the hook should not pollute LD_LIBRARY_PATH."""
        nvidia_modules = _importable_nvidia_modules()
        if nvidia_modules:
            return

        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        for pkg in NVIDIA_LIB_MODULES:
            pkg_dir = pkg.replace(".", "/")
            assert pkg_dir not in ld_path, (
                f"Found {pkg_dir} in LD_LIBRARY_PATH but {pkg} is not importable"
            )
