"""Tests for shared runtime validation utilities."""

import pytest

from mlserver.cli._runtime_utils import (
    calculate_runtime_requirements,
    check_destination_collision,
    validate_module_coverage,
    validate_module_name,
    validate_nested_runtime_package_coverage,
)
from mlserver.settings import ALLOWED_MODEL_IMPLEMENTATIONS


def test_validate_module_name_accepts_valid():
    validate_module_name("custom", "custom.py")
    validate_module_name("my_module", "my_module.py")
    validate_module_name("Custom123", "Custom123.py")


def test_validate_module_name_rejects_invalid():
    with pytest.raises(ValueError, match="valid Python module name"):
        validate_module_name("custom-module", "custom-module.py")

    with pytest.raises(ValueError, match="valid Python module name"):
        validate_module_name("123custom", "123custom.py")

    with pytest.raises(ValueError, match="valid Python module name"):
        validate_module_name("", "")


@pytest.mark.parametrize("keyword_name", ["class", "from", "for"])
def test_validate_module_name_rejects_python_keywords(keyword_name: str):
    with pytest.raises(ValueError, match="valid Python module name"):
        validate_module_name(keyword_name, f"{keyword_name}.py")


def test_validate_module_name_rejects_dunder():
    with pytest.raises(ValueError, match="reserved dunder"):
        validate_module_name("__main__", "__main__.py")

    with pytest.raises(ValueError, match="reserved dunder"):
        validate_module_name("__future__", "__future__.py")


def test_check_destination_collision_accepts_unique():
    destination_names: dict[str, str] = {}
    check_destination_collision(destination_names, "custom.py", "custom.py")
    check_destination_collision(destination_names, "other.py", "other.py")
    assert len(destination_names) == 2


def test_check_destination_collision_accepts_same_source():
    destination_names: dict[str, str] = {}
    check_destination_collision(destination_names, "custom.py", "custom.py")
    check_destination_collision(destination_names, "custom.py", "custom.py")


def test_check_destination_collision_rejects_collision():
    destination_names: dict[str, str] = {}
    check_destination_collision(destination_names, "runtime.py", "a/runtime.py")

    with pytest.raises(ValueError, match="same in-image destination"):
        check_destination_collision(destination_names, "runtime.py", "b/runtime.py")


def test_calculate_runtime_requirements_required_modules():
    custom_runtimes = [
        "custom.RuntimeA",
        "custom.RuntimeB",
        "other.Runtime",
        "mlserver_sklearn.SKLearnModel",
    ]

    required_modules, _ = calculate_runtime_requirements(
        custom_runtimes,
        ALLOWED_MODEL_IMPLEMENTATIONS,
    )

    assert required_modules == {"custom", "other"}


def test_calculate_runtime_requirements_nested_packages():
    custom_runtimes = [
        "acme.runtime.CustomRuntime",
        "acme.nested.OtherRuntime",
        "flat.Runtime",
        "mlserver_sklearn.SKLearnModel",
    ]

    _, required_nested_packages = calculate_runtime_requirements(
        custom_runtimes,
        ALLOWED_MODEL_IMPLEMENTATIONS,
    )

    assert required_nested_packages == {"acme"}


def test_validate_module_coverage_success():
    required = {"custom", "other"}
    discovered = {"custom", "other"}

    validate_module_coverage(required, discovered)


def test_validate_module_coverage_missing():
    required = {"custom", "other"}
    discovered = {"custom"}

    with pytest.raises(ValueError, match="Missing runtime source paths"):
        validate_module_coverage(required, discovered)


def test_validate_module_coverage_undeclared():
    required = {"custom"}
    discovered = {"custom", "orphan"}

    with pytest.raises(ValueError, match="undeclared runtime module"):
        validate_module_coverage(required, discovered)


def test_validate_nested_runtime_package_coverage_success():
    required_packages = {"acme"}
    discovered_packages = {"acme", "other"}

    validate_nested_runtime_package_coverage(required_packages, discovered_packages)


def test_validate_nested_runtime_package_coverage_missing():
    required_packages = {"acme"}
    discovered_packages = set()

    with pytest.raises(ValueError, match="require package-directory runtime paths"):
        validate_nested_runtime_package_coverage(required_packages, discovered_packages)
