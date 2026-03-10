import os
import sys
import pytest
import json
from unittest.mock import patch

from mlserver.settings import CORSSettings, Settings, ModelSettings, ModelParameters
from mlserver.repository import DEFAULT_MODEL_SETTINGS_FILENAME
import mlserver.settings as mlserver_settings

from .conftest import TESTDATA_PATH, TESTS_PATH


def test_settings_from_env(monkeypatch):
    http_port = 5000
    monkeypatch.setenv("mlserver_http_port", str(http_port))

    settings = Settings()

    assert settings.http_port == http_port


def test_settings_debug_default_is_disabled(monkeypatch):
    monkeypatch.delenv("MLSERVER_DEBUG", raising=False)
    monkeypatch.delenv("mlserver_debug", raising=False)
    settings = Settings(_env_file=None)
    assert settings.debug is False


def test_settings_from_env_file(monkeypatch):
    env_file = f"{TESTDATA_PATH}/.test.env"

    settings = Settings(_env_file=env_file)
    cors_settings = CORSSettings(_env_file=env_file)
    model_settings = ModelSettings(_env_file=env_file)
    model_settings.parameters = ModelParameters(_env_file=env_file)

    assert settings.http_port == 9999
    assert settings.debug is True

    assert cors_settings.allow_origin_regex == ".*"
    assert cors_settings.max_age == 999

    assert model_settings.name == "dummy-name"
    assert model_settings.parameters.uri == "dummy-uri"


def test_model_settings_from_env(monkeypatch):
    model_name = "foo-model"
    model_version = "v0.1.0"
    model_uri = "/mnt/models/my-model"

    monkeypatch.setenv("mlserver_model_name", model_name)
    monkeypatch.setenv("mlserver_model_version", model_version)
    monkeypatch.setenv("mlserver_model_uri", model_uri)
    monkeypatch.setenv("mlserver_model_implementation", "tests.fixtures.SumModel")

    model_settings = ModelSettings()
    model_settings.parameters = ModelParameters()

    assert model_settings.name == model_name
    assert model_settings.parameters.version == model_version
    assert model_settings.parameters.uri == model_uri


@pytest.mark.parametrize(
    "obj",
    [
        ({"name": "foo", "implementation": "tests.fixtures.SumModel"}),
        (
            {
                "_source": os.path.join(TESTS_PATH, DEFAULT_MODEL_SETTINGS_FILENAME),
                "name": "foo",
                "implementation": "fixtures.SumModel",
            }
        ),
    ],
)
def test_model_settings_model_validate(obj: dict):
    pre_sys_path = sys.path[:]
    model_settings = ModelSettings.model_validate(obj)
    post_sys_path = sys.path[:]

    assert pre_sys_path == post_sys_path
    assert model_settings.implementation.__name__ == "SumModel"


def _build_model_settings(implementation=None) -> ModelSettings:
    payload = {
        "_source": os.path.join(TESTS_PATH, DEFAULT_MODEL_SETTINGS_FILENAME),
        "name": "foo",
    }
    if implementation is not None:
        payload["implementation"] = implementation
    source = payload.pop("_source")
    model_settings = ModelSettings(**payload)
    model_settings._source = source
    return model_settings


def _assert_implementation_resolves_to_mocked_runtime(
    model_settings: ModelSettings, expected_import_path: str, mocked_runtime_name: str
) -> None:
    mocked_runtime = type(mocked_runtime_name, (), {})
    with patch(
        "mlserver.settings.import_string", return_value=mocked_runtime
    ) as mock_import:
        implementation = model_settings.implementation

    assert implementation is mocked_runtime
    mock_import.assert_called_once_with(expected_import_path)


def _clear_internal_test_runtime_overrides(_monkeypatch):
    _monkeypatch.setattr(
        mlserver_settings,
        "_get_trusted_runtimes_artifact_path",
        lambda: mlserver_settings.TRUSTED_RUNTIMES_ARTIFACT_PATH,
    )
    mlserver_settings.clear_trusted_runtime_caches()


def test_model_settings_allowlisted_implementation():
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_builtin_runtime_class_is_canonicalized():
    built_in_runtime = type(
        "SKLearnModel", (), {"__module__": "mlserver_sklearn.sklearn"}
    )
    model_settings = ModelSettings(name="foo", implementation=built_in_runtime)

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_model_settings_builtin_runtime_setter_is_canonicalized():
    built_in_runtime = type(
        "SKLearnModel", (), {"__module__": "mlserver_sklearn.sklearn"}
    )
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )

    model_settings.implementation = built_in_runtime

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_model_settings_builtin_submodule_import_path_is_canonicalized():
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.sklearn.SKLearnModel"
    )

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_model_settings_access_time_canonicalizes_legacy_builtin_alias():
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    # Simulate direct attribute mutation after validation.
    model_settings.implementation_ = "mlserver_sklearn.sklearn.SKLearnModel"

    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )
    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_model_settings_untrusted_implementation_rejected():
    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        _build_model_settings(implementation="malicious.CustomModel")


def test_model_settings_invalid_import_path_rejected():
    with pytest.raises(ValueError, match="invalid import path"):
        _build_model_settings(implementation="custom.Runtime-Model")

    with pytest.raises(ValueError, match="invalid import path"):
        _build_model_settings(implementation="custom.runtime")


@pytest.mark.parametrize("invalid_mutation", [[], 123])
def test_model_settings_access_time_invalid_mutation_rejected(invalid_mutation):
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    # Simulate direct mutation after validation (defense-in-depth check).
    model_settings.implementation_ = invalid_mutation  # type: ignore[assignment]

    with pytest.raises(ValueError, match="invalid import path"):
        _ = model_settings.implementation


def test_model_settings_untrusted_env_implementation_rejected(monkeypatch):
    monkeypatch.setenv("mlserver_model_name", "foo")
    monkeypatch.setenv("mlserver_model_implementation", "malicious.CustomModel")
    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        ModelSettings()


def test_model_settings_file_implementation_overrides_untrusted_env(monkeypatch):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "malicious.CustomModel")
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_missing_file_implementation_falls_back_to_env_rejected(
    monkeypatch,
):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "malicious.CustomModel")
    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        _build_model_settings()


def test_model_settings_missing_file_implementation_falls_back_to_allowlisted_env(
    monkeypatch,
):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "mlserver_sklearn.SKLearnModel")
    model_settings = _build_model_settings()
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_empty_allowlist_rejected(monkeypatch):
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(mlserver_settings, "ALLOWED_MODEL_IMPLEMENTATIONS", set())

    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_malformed_allowlist_entry_rejected(monkeypatch):
    # Whitespace-padded entries are treated as malformed and fail closed.
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {" mlserver_sklearn.SKLearnModel "},
    )

    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_image_baked_custom_runtime_allowed(monkeypatch, tmp_path):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    model_settings = _build_model_settings(implementation="custom.RuntimeModel")
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="custom.RuntimeModel",
        mocked_runtime_name="MockedCustomRuntime",
    )


def test_model_settings_image_baked_builtin_alias_is_canonicalized(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text(
        '["mlserver_sklearn.sklearn.SKLearnModel"]', encoding="utf-8"
    )
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(mlserver_settings, "ALLOWED_MODEL_IMPLEMENTATIONS", set())
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_invalid_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('{"runtime": "custom.RuntimeModel"}', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError, match="Trusted runtimes artifact must be a JSON list"
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_unparseable_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text("{invalid-json", encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError, match="Trusted runtimes artifact .* could not be loaded"
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_unreadable_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    real_open = open

    def _failing_open(path, *args, **kwargs):
        if path == str(artifact_path):
            raise OSError("permission denied")
        return real_open(path, *args, **kwargs)

    with patch("mlserver.settings.open", side_effect=_failing_open):
        with pytest.raises(
            ValueError, match="Trusted runtimes artifact .* could not be loaded"
        ):
            _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_invalid_runtime_import_path_in_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom-runtime"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError,
        match="Trusted runtimes artifact contains an invalid runtime import path",
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


@pytest.mark.parametrize(
    "invalid_runtime_path",
    [
        "RuntimeOnly",
        "custom.runtime",
        "_private.RuntimeModel",
        "custom._RuntimeModel",
        "custöm.RuntimeModel",
        "custom.Runtime-Model",
        "custom.runtime$Model",
    ],
)
def test_model_settings_unicode_or_special_runtime_in_artifact_rejected(
    monkeypatch, tmp_path, invalid_runtime_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text(
        json.dumps([invalid_runtime_path]),
        encoding="utf-8",
    )
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError,
        match="Trusted runtimes artifact contains an invalid runtime import path",
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_custom_runtime_not_in_image_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.AllowedRuntime"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    with pytest.raises(ValueError, match="allowlist of trusted runtimes"):
        _build_model_settings(implementation="custom.NotAllowedRuntime")


def test_model_settings_trusted_runtime_artifact_is_cached(monkeypatch, tmp_path):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    mocked_runtime = type("MockedCustomRuntime", (), {})
    with patch("mlserver.settings.open", wraps=open) as mock_open:
        model_settings = _build_model_settings(implementation="custom.RuntimeModel")
        with patch("mlserver.settings.import_string", return_value=mocked_runtime):
            # Access twice to confirm the trusted-runtimes artifact is read once.
            _ = model_settings.implementation
            _ = model_settings.implementation

    read_calls = [
        call for call in mock_open.call_args_list if call.args[0] == str(artifact_path)
    ]
    assert len(read_calls) == 1


def test_model_settings_serialisation():
    # Module may have been reloaded in a diff test, so let's re-import it
    from .fixtures import SumModel

    expected = "tests.fixtures.SumModel"
    model_settings = ModelSettings(name="foo", implementation=SumModel)

    assert model_settings.implementation == SumModel
    assert model_settings.implementation_ == expected

    # Dump `by_alias` to ensure that our alias overrides [1] are used
    # [2][3].
    #
    # > Whether to serialize using field aliases. [2][3]
    #
    # [1] https://github.com/jesse-c/MLServer/blob/4ac2da1d0dd7aa4b3796c047013b841fffa60e58/mlserver/settings.py#L373-L376  # noqa: E501
    # [2]  https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump  # noqa: E501
    # [3]  https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump_json  # noqa: E501

    as_dict = model_settings.model_dump(by_alias=True)
    as_dict["implementation"] == expected

    as_json = model_settings.model_dump_json(by_alias=True)
    as_dict = json.loads(as_json)
    as_dict["implementation"] == expected
