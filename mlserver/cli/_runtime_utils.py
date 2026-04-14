"""Shared validation utilities for runtime path normalization."""

import json
import keyword
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Tuple

from ..repository import DEFAULT_MODEL_SETTINGS_FILENAME
from ..settings import canonicalize_runtime_import_path, is_valid_runtime_import_path

DOCKER_GLOB_METACHARACTERS = frozenset("*?[]")
RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_CLASSNAME = "Expected format: module.ClassName"
RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_DOTTED = "Expected a dotted Python import path."


class RuntimePathValidationError(ValueError):
    """Base exception for runtime path validation failures."""

    @property
    def cli_suggestion(self) -> str:
        return ""

    def cli_message(self) -> str:
        """Format CLI-friendly error message including optional suggestion."""
        message = str(self)
        if self.cli_suggestion:
            return f"{message.rstrip('.')}. {self.cli_suggestion}"
        return message


class MissingRuntimeSourcePathsError(RuntimePathValidationError):
    """Raised when required runtime modules have no matching source paths."""

    def __init__(self, modules: set[str]):
        self.modules = modules
        super().__init__(
            "Missing runtime source paths for custom allowlisted runtime module(s): "
            f"{', '.join(sorted(modules))}."
        )

    @property
    def cli_suggestion(self) -> str:
        return "Pass one or more --runtime-path values pointing to those modules."


class UndeclaredRuntimeSourcePathsError(RuntimePathValidationError):
    """Raised when discovered runtime modules are not allowlisted."""

    def __init__(self, modules: set[str]):
        self.modules = modules
        super().__init__(
            "Runtime source paths include undeclared runtime module(s): "
            f"{', '.join(sorted(modules))}."
        )

    @property
    def cli_suggestion(self) -> str:
        return "Declare matching `--allow-runtime module.ClassName` values."


class MissingNestedRuntimePackagesError(RuntimePathValidationError):
    """Raised when nested runtimes are missing package-directory sources."""

    def __init__(self, packages: set[str]):
        self.packages = packages
        super().__init__(
            "Nested allowlisted runtime module(s) require package-directory "
            "runtime paths for top-level module(s): "
            f"{', '.join(sorted(packages))}."
        )

    @property
    def cli_suggestion(self) -> str:
        return "Use `--runtime-path <module>/` for those top-level modules."


class RuntimePathsRequireAllowlistError(RuntimePathValidationError):
    """Raised when runtime paths are provided without custom allowlisted runtimes."""

    def __init__(self, message: str):
        super().__init__(message)

    @property
    def cli_suggestion(self) -> str:
        return "Declare matching `--allow-runtime module.ClassName` values."


def canonicalize_runtime_import_paths(
    allow_runtime_import_paths: Sequence[str] = (),
) -> list[str]:
    """Canonicalize and deduplicate runtime import paths preserving order."""
    canonical = []
    seen = set()
    for allow_runtime_import_path in allow_runtime_import_paths:
        allow_runtime_import_path = canonicalize_runtime_import_path(
            allow_runtime_import_path.strip()
        )
        if allow_runtime_import_path not in seen:
            seen.add(allow_runtime_import_path)
            canonical.append(allow_runtime_import_path)
    return canonical


def normalise_runtime_import_paths(
    allow_runtime_import_paths: Sequence[str],
    *,
    invalid_label: str = "Invalid runtime import path(s)",
    expected_format: str = RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_DOTTED,
) -> list[str]:
    """Validate, canonicalize, and deduplicate runtime import paths."""
    invalid_runtimes = sorted(
        {
            runtime.strip()
            for runtime in allow_runtime_import_paths
            if not is_valid_runtime_import_path(runtime.strip())
        }
    )
    if invalid_runtimes:
        invalid_values = ", ".join(invalid_runtimes)
        raise ValueError(f"{invalid_label}: {invalid_values}. {expected_format}")
    return canonicalize_runtime_import_paths(allow_runtime_import_paths)


def load_model_settings_json(model_settings_path: Path) -> dict:
    """Load and validate raw model-settings JSON as an object."""
    try:
        with open(model_settings_path, "r", encoding="utf-8") as f:
            model_settings = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON in "
            f"{model_settings_path}:{exc.lineno}:{exc.colno} - {exc.msg}"
        ) from exc
    except UnicodeError as exc:
        raise ValueError(
            f"Invalid encoding in {model_settings_path}: expected UTF-8."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Could not read {model_settings_path}: {exc.strerror or str(exc)}"
        ) from exc

    if not isinstance(model_settings, dict):
        raise ValueError(
            f"Invalid JSON schema in {model_settings_path}: " "expected a JSON object."
        )
    return model_settings


def collect_runtime_declaration_issues(
    folder: str,
    effective_allowlist: set[str],
) -> tuple[list[tuple[Path, object]], list[tuple[Path, str]]]:
    """Scan model settings and collect invalid and missing runtime declarations."""
    invalid_runtime_implementations: list[tuple[Path, object]] = []
    missing_runtime_declarations: list[tuple[Path, str]] = []
    model_settings_paths = Path(folder).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    for model_settings_path in model_settings_paths:
        model_settings = load_model_settings_json(model_settings_path)
        implementation = model_settings.get("implementation")
        if implementation is None:
            continue
        if not is_valid_runtime_import_path(implementation):
            invalid_runtime_implementations.append(
                (model_settings_path, implementation)
            )
            continue

        implementation = canonicalize_runtime_import_path(implementation)
        if implementation not in effective_allowlist:
            missing_runtime_declarations.append((model_settings_path, implementation))

    return invalid_runtime_implementations, missing_runtime_declarations


def format_invalid_runtime_implementations_error(
    invalid_runtime_implementations: list[tuple[Path, object]],
) -> str:
    """Format user-facing message for invalid implementation values."""
    invalid_details = "\n".join(
        f"- {path}: {implementation!r}"
        for path, implementation in invalid_runtime_implementations
    )
    return (
        "Invalid implementation value(s) found in model settings:\n"
        f"{invalid_details}\n\n"
        f"{RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_CLASSNAME}"
    )


def format_missing_runtime_declarations_error(
    missing_runtime_declarations: list[tuple[Path, str]],
) -> str:
    """Format user-facing message for missing --allow-runtime declarations."""
    missing_details = "\n".join(
        f"- {path}: {implementation}"
        for path, implementation in missing_runtime_declarations
    )
    missing_implementations = sorted(
        {implementation for _, implementation in missing_runtime_declarations}
    )
    suggested_flags = " ".join(
        f"--allow-runtime {implementation}"
        for implementation in missing_implementations
    )
    return (
        "Found non-built-in model implementations not declared with "
        "`--allow-runtime`:\n"
        f"{missing_details}\n\n"
        "Suggested flags:\n"
        f"{suggested_flags}"
    )


def validate_module_name(module_name: str, runtime_path: str) -> None:
    """Validate that module_name is a valid Python identifier."""
    if not module_name.isidentifier() or keyword.iskeyword(module_name):
        raise ValueError(
            f"Runtime path '{runtime_path}' must map to a valid Python module name."
        )
    if module_name.startswith("__") and module_name.endswith("__"):
        raise ValueError(
            f"Runtime path '{runtime_path}' must not use reserved dunder "
            "module names."
        )


def validate_runtime_path_syntax(runtime_path: str) -> Path:
    """Validate runtime path syntax and return parsed relative Path."""
    runtime_path = runtime_path.strip()
    if not runtime_path:
        raise ValueError("Runtime path must not be empty.")
    if any(char.isspace() for char in runtime_path):
        raise ValueError(f"Runtime path '{runtime_path}' must not contain whitespace.")
    if "\\" in runtime_path:
        raise ValueError(f"Runtime path '{runtime_path}' must use POSIX separators.")
    if any(char in runtime_path for char in DOCKER_GLOB_METACHARACTERS):
        raise ValueError(
            f"Runtime path '{runtime_path}' must not contain Docker glob "
            "metacharacters."
        )

    candidate = Path(runtime_path)
    if candidate.is_absolute():
        raise ValueError(f"Runtime path '{runtime_path}' must be relative.")

    parts = candidate.parts
    if any(part in (".", "..") for part in parts):
        raise ValueError(
            f"Runtime path '{runtime_path}' must not contain '.' or '..' segments."
        )
    if any(part.startswith("-") for part in parts):
        raise ValueError(f"Runtime path '{runtime_path}' must not start with '-'.")
    return candidate


def check_destination_collision(
    destination_names: dict[str, str],
    destination_name: str,
    runtime_path: str,
) -> None:
    """Check for destination name collisions in runtime paths.

    Mutates destination_names by adding the mapping if no collision is found.
    """
    previous = destination_names.get(destination_name)
    if previous and previous != runtime_path:
        raise ValueError(
            "Runtime paths resolve to the same in-image destination: "
            f"{previous}, {runtime_path}. Rename one of them."
        )
    destination_names[destination_name] = runtime_path


def validate_module_coverage(
    required_modules: set[str],
    discovered_modules: set[str],
) -> None:
    """Validate that all required modules have runtime paths and vice versa."""
    missing_modules = required_modules.difference(discovered_modules)
    if missing_modules:
        raise MissingRuntimeSourcePathsError(missing_modules)

    undeclared_modules = discovered_modules.difference(required_modules)
    if undeclared_modules:
        raise UndeclaredRuntimeSourcePathsError(undeclared_modules)


def validate_nested_runtime_package_coverage(
    required_packages: set[str],
    discovered_packages: set[str],
) -> None:
    """Validate nested runtime imports are sourced from package directories."""
    missing_packages = required_packages.difference(discovered_packages)
    if missing_packages:
        raise MissingNestedRuntimePackagesError(missing_packages)


def calculate_runtime_requirements(
    custom_runtimes: list[str],
    allowed_implementations: set[str],
) -> tuple[set[str], set[str]]:
    """Compute required top-level modules and nested packages in one pass."""
    required_modules: set[str] = set()
    required_nested_packages: set[str] = set()
    for runtime in custom_runtimes:
        if runtime in allowed_implementations:
            continue
        module_path = runtime.rpartition(".")[0]
        top_level = module_path.split(".", 1)[0]
        required_modules.add(top_level)
        if "." in module_path:
            required_nested_packages.add(top_level)
    return required_modules, required_nested_packages


def validate_runtime_path_preconditions(
    runtime_source_paths: Optional[Sequence[str]],
    required_modules: set[str],
    *,
    runtime_paths_without_allowlist_message: str,
) -> bool:
    """Validate runtime-path presence and allowlist preconditions.

    Returns:
        True when runtime-path normalization should continue, False when no
        runtime paths are needed.
    """
    if not runtime_source_paths:
        if required_modules:
            raise MissingRuntimeSourcePathsError(required_modules)
        return False
    if not required_modules:
        raise RuntimePathsRequireAllowlistError(runtime_paths_without_allowlist_message)
    return True


def _resolve_and_compute_output_path(
    candidate: Path,
    runtime_path: str,
    folder_path: Optional[Path],
    build_folder: Optional[str],
    *,
    use_relative_paths_in_build_folder: bool,
    reject_build_folder_root: bool,
) -> Tuple[Optional[Path], str, str]:
    """Resolve runtime path and compute output/destination paths."""
    if folder_path is None:
        resolved = None
    else:
        resolved = (folder_path / candidate).resolve()
        try:
            resolved.relative_to(folder_path)
        except ValueError as exc:
            raise ValueError(
                f"Runtime path '{runtime_path}' must be inside build folder "
                f"'{build_folder}'."
            ) from exc
        if not resolved.exists():
            raise ValueError(f"Runtime path '{runtime_path}' does not exist.")

    output_runtime_path = runtime_path
    destination_source = runtime_path
    if resolved is not None and use_relative_paths_in_build_folder:
        assert folder_path is not None
        relative = resolved.relative_to(folder_path).as_posix()
        if reject_build_folder_root and relative in {".", ""}:
            raise ValueError(
                "Runtime path must point to a Python module or package inside "
                "the build folder, not the build folder itself."
            )
        if any(part.startswith("-") for part in relative.split("/")):
            raise ValueError(f"Runtime path '{runtime_path}' must not start with '-'.")
        output_runtime_path = relative
        destination_source = relative

    return resolved, output_runtime_path, destination_source


def _classify_and_validate_runtime_source(
    candidate: Path,
    runtime_path: str,
    resolved: Optional[Path],
    *,
    source_specific_module_name_errors: bool,
) -> tuple[str, bool]:
    """Classify runtime source and validate module naming constraints."""
    if candidate.suffix == ".py":
        module_name = candidate.stem
        if resolved is not None and not resolved.is_file():
            raise ValueError(
                f"Runtime file '{runtime_path}' must be a Python file (*.py)."
            )
        is_package = False
        error_noun = "Runtime file"
    else:
        module_name = candidate.name
        if resolved is None:
            raise ValueError(
                "Directory runtime paths require build_folder context for package "
                f"validation: '{runtime_path}'."
            )
        init_file = resolved / "__init__.py"
        if not init_file.is_file():
            raise ValueError(
                f"Runtime directory '{runtime_path}' must be a Python package "
                "containing '__init__.py'."
            )
        is_package = True
        error_noun = "Runtime directory"

    try:
        validate_module_name(module_name, runtime_path)
    except ValueError as exc:
        if source_specific_module_name_errors:
            raise ValueError(str(exc).replace("Runtime path", error_noun)) from exc
        raise

    return module_name, is_package


def normalise_runtime_source_paths(
    runtime_source_paths: Sequence[str],
    required_modules: set[str],
    required_nested_packages: set[str],
    *,
    build_folder: Optional[str] = None,
    use_relative_paths_in_build_folder: bool = False,
    reject_build_folder_root: bool = False,
    source_specific_module_name_errors: bool = False,
) -> list[str]:
    """Normalise runtime source paths and validate coverage constraints.

    Returns normalized paths suitable for Docker COPY handling.

    Raises:
        MissingRuntimeSourcePathsError: Required modules were not covered.
        UndeclaredRuntimeSourcePathsError: Discovered modules were undeclared.
        MissingNestedRuntimePackagesError: Required package directories missing.
        ValueError: Path syntax/resolution/classification validation failed.
    """
    normalised: list[str] = []
    discovered_modules = set()
    discovered_packages = set()
    destination_names: dict[str, str] = {}
    folder_path = Path(build_folder).resolve() if build_folder is not None else None

    for raw_runtime_path in runtime_source_paths:
        candidate = validate_runtime_path_syntax(raw_runtime_path)
        runtime_path = candidate.as_posix()
        resolved, output_runtime_path, destination_source = (
            _resolve_and_compute_output_path(
                candidate,
                runtime_path,
                folder_path,
                build_folder,
                use_relative_paths_in_build_folder=use_relative_paths_in_build_folder,
                reject_build_folder_root=reject_build_folder_root,
            )
        )
        module_name, is_package = _classify_and_validate_runtime_source(
            candidate,
            runtime_path,
            resolved,
            source_specific_module_name_errors=source_specific_module_name_errors,
        )

        if is_package:
            discovered_packages.add(module_name)
        discovered_modules.add(module_name)

        check_destination_collision(
            destination_names,
            Path(destination_source).name,
            destination_source,
        )
        normalised.append(output_runtime_path)

    validate_module_coverage(required_modules, discovered_modules)
    validate_nested_runtime_package_coverage(
        required_nested_packages,
        discovered_packages,
    )
    return normalised
