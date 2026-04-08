import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

DEFAULT_IMAGE_TAG = "my-custom-image:0.1.0"


@pytest.fixture
def cli_main():
    return importlib.import_module("mlserver.cli.main")


@pytest.fixture
def cli_build():
    return importlib.import_module("mlserver.cli.build")


@pytest.fixture
def runner():
    return CliRunner()


def _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured=None):
    def _fake_generate_dockerfile(*args, **kwargs):
        if captured is not None:
            captured["custom_runtimes"] = kwargs.get("custom_runtimes")
            captured["runtime_paths"] = kwargs.get("runtime_paths")
            captured["dev"] = kwargs.get("dev")
        return "FROM test"

    def _fake_build_image(folder, dockerfile, image_tag, no_cache):
        if captured is not None:
            captured["folder"] = folder
            captured["dockerfile"] = dockerfile
            captured["image_tag"] = image_tag
            captured["no_cache"] = no_cache
        return image_tag

    monkeypatch.setattr(cli_build, "generate_dockerfile", _fake_generate_dockerfile)
    monkeypatch.setattr(cli_main, "build_image", _fake_build_image)


def _write_model_settings_batch(models):
    for rel_path, payload in models.items():
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _write_runtime_py(path: str, class_names=("Runtime",)):
    runtime_file = Path(path)
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"class {class_name}: pass\n" for class_name in class_names)
    runtime_file.write_text(body, encoding="utf-8")


def _write_runtime_package(path: str, init_content: str = ""):
    package_dir = Path(path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(init_content, encoding="utf-8")


def _invoke_build(
    runner: CliRunner, cli_main, allow_runtimes=(), runtime_paths=(), dev=False
):
    args = ["build", ".", "-t", DEFAULT_IMAGE_TAG]
    if dev:
        args.append("--dev")
    for runtime in allow_runtimes:
        args.extend(["--allow-runtime", runtime])
    for runtime_path in runtime_paths:
        args.extend(["--runtime-path", runtime_path])

    return runner.invoke(cli_main.root, args)


def _invoke_dockerfile(
    runner: CliRunner, cli_main, allow_runtimes=(), runtime_paths=(), dev=False
):
    args = ["dockerfile", "."]
    if dev:
        args.append("--dev")
    for runtime in allow_runtimes:
        args.extend(["--allow-runtime", runtime])
    for runtime_path in runtime_paths:
        args.extend(["--runtime-path", runtime_path])

    return runner.invoke(cli_main.root, args)


def test_build_passes_allow_runtime_to_dockerfile_generator(
    monkeypatch, cli_main, cli_build, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        _write_runtime_py("another.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime", "another.Runtime"),
            runtime_paths=("custom.py", "another.py"),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.Runtime", "another.Runtime"]
    assert captured["image_tag"] == DEFAULT_IMAGE_TAG
    assert captured["runtime_paths"] == ["custom.py", "another.py"]


def test_build_fails_when_custom_implementation_is_not_allowlisted(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/custom/model-settings.json": {
                    "name": "custom-model",
                    "implementation": "custom.MyRuntime",
                }
            }
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Found non-built-in model implementations" in result.output
    assert "custom.MyRuntime" in result.output
    assert "--allow-runtime custom.MyRuntime" in result.output


def test_build_accepts_legacy_builtin_implementation_without_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/sk/model-settings.json": {
                    "name": "sk-model",
                    "implementation": "mlserver_sklearn.sklearn.SKLearnModel",
                }
            }
        )
        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == []


def test_build_allows_multiple_model_settings_when_all_custom_are_allowlisted(
    monkeypatch, cli_main, cli_build, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/a/model-settings.json": {
                    "name": "model-a",
                    "implementation": "custom.RuntimeA",
                },
                "models/b/v2/model-settings.json": {
                    "name": "model-b",
                    "implementation": "custom.RuntimeB",
                },
            }
        )
        _write_runtime_py("custom.py", class_names=("RuntimeA", "RuntimeB"))

        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.RuntimeA", "custom.RuntimeB"),
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.RuntimeA", "custom.RuntimeB"]


def test_build_fails_when_custom_runtime_source_path_missing(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
        )

    assert result.exit_code == 2
    assert "--allow-runtime requires --runtime-path" in result.output


def test_build_fails_when_runtime_path_contains_whitespace(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom runtime.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom runtime.py",),
        )

    assert result.exit_code == 2
    assert "must not contain whitespace" in result.output


def test_build_fails_when_runtime_directory_is_not_package(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom/runtime.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom",),
        )

    assert result.exit_code == 2
    assert "containing '__init__.py'" in result.output


def test_build_accepts_runtime_package_directory(
    monkeypatch, cli_main, cli_build, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured)

    with runner.isolated_filesystem():
        _write_runtime_package("custom", init_content="class Runtime: pass\n")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom",),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["runtime_paths"] == ["custom"]


def test_build_fails_when_nested_runtime_uses_module_file(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("acme.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("acme.runtime.CustomRuntime",),
            runtime_paths=("acme.py",),
        )

    assert result.exit_code == 2
    assert "require package-directory runtime paths" in result.output


def test_build_fails_when_runtime_path_is_given_without_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_build(
            runner,
            cli_main,
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 2
    assert "--runtime-path requires --allow-runtime" in result.output


@pytest.mark.parametrize("bad_path", ["-custom.py", "-custom"])
def test_build_fails_when_runtime_path_segment_starts_with_dash(
    monkeypatch, cli_main, cli_build, runner, bad_path
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        if bad_path.endswith(".py"):
            _write_runtime_py(bad_path)
        else:
            _write_runtime_package(bad_path)
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(bad_path,),
        )

    assert result.exit_code == 2
    assert "must not start with '-'" in result.output


def test_build_fails_when_runtime_path_uses_backslash_separator(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(r"custom\module.py",),
        )

    assert result.exit_code == 2
    assert "must use POSIX separators" in result.output


@pytest.mark.parametrize("runtime_path", ["custom*.py", "custom?.py", "custom[ab].py"])
def test_build_fails_when_runtime_path_contains_docker_glob_metacharacters(
    monkeypatch, cli_main, cli_build, runner, runtime_path
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(runtime_path,),
        )

    assert result.exit_code == 2
    assert "must not contain Docker glob metacharacters" in result.output


@pytest.mark.parametrize("runtime_path", ["custom*.py", "custom?.py", "custom[ab].py"])
def test_dockerfile_fails_when_runtime_path_contains_docker_glob_metacharacters(
    monkeypatch, cli_main, cli_build, runner, runtime_path
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(runtime_path,),
        )

    assert result.exit_code == 2
    assert "must not contain Docker glob metacharacters" in result.output


def test_build_fails_when_runtime_paths_share_same_destination_name(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("a/runtime.py", class_names=("RuntimeA",))
        _write_runtime_py("b/runtime.py", class_names=("RuntimeB",))
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("runtime.Runtime",),
            runtime_paths=("a/runtime.py", "b/runtime.py"),
        )

    assert result.exit_code == 2
    assert "same in-image destination" in result.output


@pytest.mark.parametrize("reserved_name", ["__future__", "__main__", "__builtin__"])
def test_build_fails_when_runtime_path_is_reserved_module_name(
    monkeypatch, cli_main, cli_build, runner, reserved_name
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py(f"{reserved_name}.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(f"{reserved_name}.py",),
        )

    assert result.exit_code == 2
    assert "must not use reserved dunder module names" in result.output


@pytest.mark.parametrize("keyword_name", ["class", "from"])
def test_build_fails_when_runtime_path_is_python_keyword(
    monkeypatch, cli_main, cli_build, runner, keyword_name
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py(f"{keyword_name}.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=(f"{keyword_name}.py",),
        )

    assert result.exit_code == 2
    assert "must map to a valid Python module name" in result.output


def test_build_fails_when_runtime_path_module_is_not_allowlisted(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        _write_runtime_py("extra.py", class_names=("Extra",))
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom.py", "extra.py"),
        )

    assert result.exit_code == 2
    assert "undeclared runtime module(s): extra" in result.output


@pytest.mark.parametrize(
    "invalid_runtime", ["invalid-runtime", "_private.Runtime", "custom.runtime"]
)
def test_build_fails_when_allow_runtime_format_is_invalid(
    monkeypatch, cli_main, cli_build, runner, invalid_runtime
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=(invalid_runtime,),
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 2
    assert "Invalid --allow-runtime value(s)" in result.output
    assert "Expected format: module.ClassName" in result.output


def test_build_fails_when_model_implementation_format_is_invalid(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/custom/model-settings.json": {
                    "name": "custom-model",
                    "implementation": "invalid-runtime",
                }
            }
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid implementation" in result.output
    assert "Expected format: module.ClassName" in result.output


def test_build_fails_when_model_settings_json_is_not_an_object(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {"models/custom/model-settings.json": ["not-an-object"]}
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid JSON schema" in result.output
    assert "expected a JSON object" in result.output


def test_build_fails_when_model_settings_json_is_malformed(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        path = Path("models/custom/model-settings.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ invalid json }", encoding="utf-8")

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid JSON in" in result.output


def test_build_fails_when_runtime_path_is_build_folder_root(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        package_root = Path("custom")
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text(
            "class Runtime: pass\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            cli_main.root,
            [
                "build",
                "custom",
                "-t",
                DEFAULT_IMAGE_TAG,
                "--allow-runtime",
                "custom.Runtime",
                "--runtime-path",
                ".",
            ],
        )

    assert result.exit_code == 2
    assert "not the build folder itself" in result.output


def test_dockerfile_fails_when_runtime_path_is_build_folder_root(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        package_root = Path("custom")
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text(
            "class Runtime: pass\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            cli_main.root,
            [
                "dockerfile",
                "custom",
                "--allow-runtime",
                "custom.Runtime",
                "--runtime-path",
                ".",
            ],
        )

    assert result.exit_code == 2
    assert "not the build folder itself" in result.output


def test_dockerfile_passes_allow_runtime_to_dockerfile_generator(
    monkeypatch, cli_main, cli_build, runner
):
    captured = {}

    def _fake_generate_dockerfile(*args, **kwargs):
        captured["custom_runtimes"] = kwargs.get("custom_runtimes")
        captured["runtime_paths"] = kwargs.get("runtime_paths")
        return "FROM test"

    def _fake_write_dockerfile(folder, dockerfile, include_dockerignore):
        captured["folder"] = folder
        captured["dockerfile"] = dockerfile
        captured["include_dockerignore"] = include_dockerignore
        return "Dockerfile"

    monkeypatch.setattr(cli_build, "generate_dockerfile", _fake_generate_dockerfile)
    monkeypatch.setattr(cli_main, "write_dockerfile", _fake_write_dockerfile)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        _write_runtime_py("another.py")
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime", "another.Runtime"),
            runtime_paths=("custom.py", "another.py"),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.Runtime", "another.Runtime"]
    assert captured["runtime_paths"] == ["custom.py", "another.py"]
    assert captured["folder"] == "."
    assert captured["dockerfile"] == "FROM test"


def test_build_wraps_generate_dockerfile_value_error_as_usage_error(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)
    monkeypatch.setattr(
        cli_build,
        "generate_dockerfile",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("sink boom")),
    )

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 2
    assert "sink boom" in result.output


def test_dockerfile_wraps_generate_dockerfile_value_error_as_usage_error(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)
    monkeypatch.setattr(
        cli_build,
        "generate_dockerfile",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("sink boom")),
    )

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 2
    assert "sink boom" in result.output


def test_build_dev_succeeds(monkeypatch, cli_main, cli_build, runner):
    """Test --dev builds successfully without runtime flags."""
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, cli_build, captured)

    with runner.isolated_filesystem():
        result = _invoke_build(runner, cli_main, dev=True)

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["dev"] is True
    assert captured.get("custom_runtimes") is None
    assert captured["image_tag"] == DEFAULT_IMAGE_TAG


def test_build_dev_rejects_allow_runtime(monkeypatch, cli_main, cli_build, runner):
    """Test --dev cannot be combined with --allow-runtime."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_build(
            runner, cli_main, dev=True, allow_runtimes=("custom.Runtime",)
        )

    assert result.exit_code == 2
    assert (
        "--dev cannot be combined with --allow-runtime or --runtime-path"
        in result.output
    )


def test_build_dev_rejects_runtime_path(monkeypatch, cli_main, cli_build, runner):
    """Test --dev cannot be combined with --runtime-path."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_build(runner, cli_main, dev=True, runtime_paths=("custom.py",))

    assert result.exit_code == 2
    assert (
        "--dev cannot be combined with --allow-runtime or --runtime-path"
        in result.output
    )


def test_dockerfile_dev_succeeds(monkeypatch, cli_main, cli_build, runner):
    """Test dockerfile --dev generates successfully without runtime flags."""
    captured = {}

    def _fake_generate_dockerfile(*args, **kwargs):
        captured["custom_runtimes"] = kwargs.get("custom_runtimes")
        captured["dev"] = kwargs.get("dev")
        return "FROM test"

    def _fake_write_dockerfile(folder, dockerfile, include_dockerignore):
        captured["folder"] = folder
        captured["dockerfile"] = dockerfile
        return "Dockerfile"

    monkeypatch.setattr(cli_build, "generate_dockerfile", _fake_generate_dockerfile)
    monkeypatch.setattr(cli_main, "write_dockerfile", _fake_write_dockerfile)

    with runner.isolated_filesystem():
        result = _invoke_dockerfile(runner, cli_main, dev=True)

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["dev"] is True
    assert captured.get("custom_runtimes") is None
    assert captured["folder"] == "."
    assert captured["dockerfile"] == "FROM test"


def test_dockerfile_dev_rejects_allow_runtime(monkeypatch, cli_main, cli_build, runner):
    """Test dockerfile --dev cannot be combined with --allow-runtime."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_dockerfile(
            runner, cli_main, dev=True, allow_runtimes=("custom.Runtime",)
        )

    assert result.exit_code == 2
    assert (
        "--dev cannot be combined with --allow-runtime or --runtime-path"
        in result.output
    )


def test_dockerfile_dev_rejects_runtime_path(monkeypatch, cli_main, cli_build, runner):
    """Test dockerfile --dev cannot be combined with --runtime-path."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_dockerfile(
            runner, cli_main, dev=True, runtime_paths=("custom.py",)
        )

    assert result.exit_code == 2
    assert (
        "--dev cannot be combined with --allow-runtime or --runtime-path"
        in result.output
    )


def test_build_rejects_builtin_runtime_with_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("mlserver_sklearn.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("mlserver_sklearn.SKLearnModel",),
            runtime_paths=("mlserver_sklearn.py",),
        )

    assert result.exit_code == 2
    assert (
        "Built-in runtime(s) 'mlserver_sklearn.SKLearnModel' cannot be specified"
        in result.output
    )
    assert "use a different module name" in result.output


def test_build_rejects_legacy_builtin_runtime_with_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("mlserver_sklearn.py")
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("mlserver_sklearn.sklearn.SKLearnModel",),
            runtime_paths=("mlserver_sklearn.py",),
        )

    assert result.exit_code == 2
    assert (
        "Built-in runtime(s) 'mlserver_sklearn.sklearn.SKLearnModel' "
        "cannot be specified" in result.output
    )
    assert "use a different module name" in result.output


def test_dockerfile_rejects_builtin_runtime_with_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    """Test dockerfile rejects built-in runtimes specified with --allow-runtime."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("mlserver_sklearn.py")
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("mlserver_sklearn.SKLearnModel",),
            runtime_paths=("mlserver_sklearn.py",),
        )

    assert result.exit_code == 2
    assert (
        "Built-in runtime(s) 'mlserver_sklearn.SKLearnModel' cannot be specified"
        in result.output
    )
    assert "use a different module name" in result.output


def test_dockerfile_rejects_legacy_builtin_runtime_with_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    """Test dockerfile rejects legacy built-in runtime paths with --allow-runtime."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("mlserver_sklearn.py")
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("mlserver_sklearn.sklearn.SKLearnModel",),
            runtime_paths=("mlserver_sklearn.py",),
        )

    assert result.exit_code == 2
    assert (
        "Built-in runtime(s) 'mlserver_sklearn.sklearn.SKLearnModel' "
        "cannot be specified" in result.output
    )
    assert "use a different module name" in result.output


def test_dockerfile_fails_when_custom_runtime_source_path_missing(
    monkeypatch, cli_main, cli_build, runner
):
    """Test dockerfile requires --runtime-path when --allow-runtime is specified."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime",),
        )

    assert result.exit_code == 2
    assert "--allow-runtime requires --runtime-path" in result.output


def test_dockerfile_fails_when_runtime_path_is_given_without_allow_runtime(
    monkeypatch, cli_main, cli_build, runner
):
    """Test dockerfile requires --allow-runtime when --runtime-path is specified."""
    _patch_build_pipeline(monkeypatch, cli_main, cli_build)

    with runner.isolated_filesystem():
        _write_runtime_py("custom.py")
        result = _invoke_dockerfile(
            runner,
            cli_main,
            runtime_paths=("custom.py",),
        )

    assert result.exit_code == 2
    assert "--runtime-path requires --allow-runtime" in result.output
