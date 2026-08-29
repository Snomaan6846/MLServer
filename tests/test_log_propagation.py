"""
Unit tests for log-level propagation via _configure_framework_logger() hook.

Each test class is independently skipped when its runtime package is not
installed, so the suite works in core-only CI as well as full-runtimes CI.

Tests cover:
  - Base class hook contract (no-op, called in __init__, polymorphic dispatch)
  - Level-mapping dicts (XGBoost, ONNX, CatBoost)
  - _configure_framework_logger() side-effects for each runtime
  - Deferred application in load() where required (CatBoost, ONNX)
  - Level captured once as ``self._mlserver_log_level`` on ``MLModel.__init__``
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlserver.model import MLModel
from mlserver.settings import ModelSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _set_mlserver_level(level: int) -> None:
    """Force the mlserver logger to a specific level for testing."""
    logging.getLogger("mlserver").setLevel(level)


def _make_settings(name: str = "test-model", **kwargs) -> ModelSettings:
    # Use a test-fixture implementation that passes the production-mode
    # trusted-runtimes allowlist (TEST_ONLY_EXTRA_IMPLEMENTATIONS).
    kwargs.setdefault("implementation", "tests.fixtures.SimpleModel")
    return ModelSettings(name=name, **kwargs)


@pytest.fixture(autouse=True)
def _restore_mlserver_level():
    """Reset the mlserver logger level after each test."""
    original = logging.getLogger("mlserver").level
    yield
    logging.getLogger("mlserver").setLevel(original)


# ---------------------------------------------------------------------------
# mlserver.logging.get_log_level
# ---------------------------------------------------------------------------


class TestGetLogLevel:
    def test_returns_int(self):
        from mlserver.logging import get_log_level

        assert isinstance(get_log_level(), int)

    @pytest.mark.parametrize(
        "level",
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
    )
    def test_reflects_logger_level(self, level: int):
        from mlserver.logging import get_log_level

        _set_mlserver_level(level)
        assert get_log_level() == level


# ---------------------------------------------------------------------------
# Base class hook contract
# ---------------------------------------------------------------------------


class TestConfigureFrameworkLogger:
    def test_base_is_noop(self):
        settings = _make_settings()
        model = MLModel(settings)
        assert model._configure_framework_logger() is None

    def test_hook_called_during_init(self):
        settings = _make_settings()
        with patch.object(MLModel, "_configure_framework_logger") as mock_hook:
            MLModel(settings)
            mock_hook.assert_called_once()

    def test_subclass_override_invoked(self):
        called = []

        class FakeRuntime(MLModel):
            def _configure_framework_logger(self) -> None:
                called.append(True)

        settings = _make_settings()
        FakeRuntime(settings)
        assert called == [True]

    def test_init_sets_log_level_from_mlserver(self):
        _set_mlserver_level(logging.ERROR)
        model = MLModel(_make_settings())
        assert model._mlserver_log_level == logging.ERROR


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_lightgbm"),
    reason="mlserver-lightgbm not installed",
)
class TestLGBMConfigureFrameworkLogger:
    def test_registers_mlserver_lightgbm_logger(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        settings = _make_settings()
        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings)
            mock_lgb.register_logger.assert_called_once()
            registered = mock_lgb.register_logger.call_args[0][0]
            assert registered.name == "mlserver.lightgbm"

    def test_registered_logger_inherits_mlserver_level(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        _set_mlserver_level(logging.WARNING)
        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings=_make_settings())
            lgb_logger = mock_lgb.register_logger.call_args[0][0]
        assert lgb_logger.level == logging.WARNING
        assert lgb_logger.getEffectiveLevel() == logging.WARNING

    def test_registered_logger_has_info_and_warning(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings=_make_settings())
            registered = mock_lgb.register_logger.call_args[0][0]
        for method in ("info", "warning"):
            assert callable(getattr(registered, method, None))


# ---------------------------------------------------------------------------
# XGBoost verbosity mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_xgboost"),
    reason="mlserver-xgboost not installed",
)
class TestXGBoostVerbosityMapping:
    @pytest.fixture(autouse=True)
    def _mapping(self):
        from mlserver_xgboost.xgboost import _XGB_VERBOSITY

        self.mapping = _XGB_VERBOSITY

    @pytest.fixture(autouse=True)
    def _reset_xgboost_logger(self):
        yield
        logging.getLogger("xgboost").setLevel(logging.NOTSET)

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, 3),
            (logging.INFO, 2),
            (logging.WARNING, 1),
            (logging.ERROR, 0),
            (logging.CRITICAL, 0),
        ],
    )
    def test_mapping_values(self, level: int, expected: int):
        assert self.mapping[level] == expected

    def test_configure_calls_set_config(self):
        from mlserver_xgboost.xgboost import XGBoostModel

        with patch("mlserver_xgboost.xgboost.xgb") as mock_xgb:
            _set_mlserver_level(logging.WARNING)
            XGBoostModel(settings=_make_settings())
            mock_xgb.set_config.assert_called_once_with(verbosity=1)

    def test_configure_sets_python_xgboost_logger(self):
        from mlserver_xgboost.xgboost import XGBoostModel

        with patch("mlserver_xgboost.xgboost.xgb"):
            _set_mlserver_level(logging.ERROR)
            XGBoostModel(settings=_make_settings())
            assert logging.getLogger("xgboost").level == logging.ERROR


# ---------------------------------------------------------------------------
# ONNX Runtime severity mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxLogSeverityMapping:
    @pytest.fixture(autouse=True)
    def _mapping(self):
        from mlserver_onnx.utils import _ORT_LOG_SEVERITY

        self.mapping = _ORT_LOG_SEVERITY

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, 0),
            (logging.INFO, 1),
            (logging.WARNING, 2),
            (logging.ERROR, 3),
            (logging.CRITICAL, 4),
        ],
    )
    def test_mapping_values(self, level: int, expected: int):
        assert self.mapping[level] == expected

    def test_unknown_level_falls_back_to_info(self):
        from mlserver_onnx.utils import _ORT_LOG_SEVERITY

        assert _ORT_LOG_SEVERITY.get(99, 1) == 1


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxBuildSessionOptions:
    """Tests for _build_session_options auto-set and user-override semantics."""

    def test_returns_session_options_when_settings_none(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options
        import onnxruntime as ort

        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings)
        assert isinstance(result, ort.SessionOptions)

    def test_auto_sets_log_severity_from_mlserver_level(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        _set_mlserver_level(logging.DEBUG)
        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings)
        assert result.log_severity_level == 0  # DEBUG → 0

    def test_auto_sets_warning_severity(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        _set_mlserver_level(logging.WARNING)
        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings)
        assert result.log_severity_level == 2  # WARNING → 2

    def test_respects_user_supplied_log_severity(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        _set_mlserver_level(logging.DEBUG)
        settings = OnnxSettings(session_options={"log_severity_level": 4})
        result = _build_session_options(settings)
        assert result.log_severity_level == 4  # user override wins

    def test_auto_sets_when_other_session_options_present(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        _set_mlserver_level(logging.ERROR)
        settings = OnnxSettings(session_options={"inter_op_num_threads": 2})
        result = _build_session_options(settings)
        assert result.log_severity_level == 3  # ERROR → 3
        assert result.inter_op_num_threads == 2


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxApplySessionConfigEntries:
    """Tests for _apply_session_config_entries."""

    def test_noop_when_entries_none(self):
        import onnxruntime as ort
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _apply_session_config_entries

        options = ort.SessionOptions()
        settings = OnnxSettings(session_config_entries=None)
        result = _apply_session_config_entries(options, settings)
        assert result is options

    def test_applies_config_entries(self):
        import onnxruntime as ort
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _apply_session_config_entries

        options = ort.SessionOptions()
        settings = OnnxSettings(
            session_config_entries={
                "session.use_env_allocators": "1",
            }
        )
        result = _apply_session_config_entries(options, settings)
        assert result is options
        assert result.get_session_config_entry("session.use_env_allocators") == "1"


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxConfigureFrameworkLogger:
    """Verify OnnxModel._configure_framework_logger() stores per-session severity."""

    def test_does_not_call_global_api(self):
        from mlserver_onnx.onnx import OnnxModel

        with patch("mlserver_onnx.onnx.ort") as mock_ort:
            settings = _make_settings()
            OnnxModel(settings)
            assert not mock_ort.set_default_logger_severity.called


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxProviderFallbackLogging:
    """Tests for provider mismatch warning in OnnxModel.load()."""

    @staticmethod
    def _make_mock_session(active_providers, input_names=None):
        if input_names is None:
            input_names = ["input_0"]
        mock_session = MagicMock()
        mock_session.get_providers.return_value = active_providers
        mock_inputs = []
        for name in input_names:
            inp = MagicMock()
            inp.name = name
            mock_inputs.append(inp)
        mock_session.get_inputs.return_value = mock_inputs
        out = MagicMock()
        out.name = "output_0"
        mock_session.get_outputs.return_value = [out]
        return mock_session

    @pytest.fixture()
    def _patch_onnx_load(self):
        """Patch all externals so OnnxModel.load() runs without a model file."""
        patches = [
            patch(
                "mlserver_onnx.onnx.get_model_uri",
                new_callable=AsyncMock,
                return_value="/fake/model.onnx",
            ),
            patch("mlserver_onnx.onnx.onnx.load", return_value=MagicMock()),
            patch(
                "mlserver_onnx.onnx._extract_metadata",
                return_value={"inputs": [], "outputs": []},
            ),
        ]
        mocks = [p.start() for p in patches]
        yield mocks
        for p in patches:
            p.stop()

    async def test_logs_warning_on_provider_mismatch(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(
                extra={"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
            ),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" in caplog.text
        assert "CUDAExecutionProvider" in caplog.text

    async def test_no_warning_when_providers_match(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(extra={"providers": ["CPUExecutionProvider"]}),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" not in caplog.text

    async def test_no_warning_when_ort_adds_cpu_fallback(
        self, _patch_onnx_load, caplog
    ):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(extra={"providers": ["CUDAExecutionProvider"]}),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" not in caplog.text

    async def test_logs_info_with_active_providers(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.INFO, logger="mlserver"):
                await model.load()

        assert "loaded with execution providers" in caplog.text
        assert "CPUExecutionProvider" in caplog.text


# ---------------------------------------------------------------------------
# CatBoost log-level mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_catboost"),
    reason="mlserver-catboost not installed",
)
class TestCatBoostLogLevelMapping:
    @pytest.fixture(autouse=True)
    def _mapping(self):
        from mlserver_catboost.catboost import _CB_LOG_LEVEL

        self.mapping = _CB_LOG_LEVEL

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, "Debug"),
            (logging.INFO, "Info"),
            (logging.WARNING, "Verbose"),
            (logging.ERROR, "Silent"),
            (logging.CRITICAL, "Silent"),
        ],
    )
    def test_mapping_values(self, level: int, expected: str):
        assert self.mapping[level] == expected

    def test_verbosity_decreases_as_level_rises(self):
        scale = ["Silent", "Verbose", "Info", "Debug"]
        info_idx = scale.index(self.mapping[logging.INFO])
        warn_idx = scale.index(self.mapping[logging.WARNING])
        assert info_idx > warn_idx

    def test_configure_stores_correct_level_at_warning(self):
        from mlserver_catboost.catboost import CatboostModel

        _set_mlserver_level(logging.WARNING)
        model = CatboostModel(settings=_make_settings())
        assert model._catboost_log_level == "Verbose"

    def test_configure_stores_correct_level_at_debug(self):
        from mlserver_catboost.catboost import CatboostModel

        _set_mlserver_level(logging.DEBUG)
        model = CatboostModel(settings=_make_settings())
        assert model._catboost_log_level == "Debug"

    @pytest.mark.parametrize("unmapped_level", [25, 15, 5])
    def test_configure_stores_info_for_unmapped_level(self, unmapped_level):
        from mlserver_catboost.catboost import CatboostModel

        _set_mlserver_level(unmapped_level)
        model = CatboostModel(settings=_make_settings())
        assert model._catboost_log_level == "Info"

    @pytest.mark.asyncio
    async def test_load_passes_stored_logging_level_to_classifier(self):
        from mlserver_catboost.catboost import CatboostModel
        from mlserver.settings import ModelParameters

        _set_mlserver_level(logging.WARNING)
        settings = _make_settings(
            parameters=ModelParameters(uri="/fake/model.cbm"),
        )
        model = CatboostModel(settings)

        with patch(
            "mlserver_catboost.catboost.get_model_uri",
            new_callable=AsyncMock,
            return_value="/fake/model.cbm",
        ), patch("mlserver_catboost.catboost.CatBoostClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            await model.load()

        mock_cls.assert_called_once_with(logging_level="Verbose")
        mock_instance.load_model.assert_called_once_with("/fake/model.cbm")


# ---------------------------------------------------------------------------
# sklearn
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_sklearn"),
    reason="mlserver-sklearn not installed",
)
class TestSklearnConfigureFrameworkLogger:
    @pytest.fixture(autouse=True)
    def _reset_sklearn_loggers(self):
        yield
        logging.getLogger("sklearn").setLevel(logging.NOTSET)
        logging.getLogger("joblib").setLevel(logging.NOTSET)

    def test_sets_sklearn_and_joblib_levels(self):
        from mlserver_sklearn.sklearn import SKLearnModel

        _set_mlserver_level(logging.WARNING)
        SKLearnModel(settings=_make_settings())
        assert logging.getLogger("sklearn").level == logging.WARNING
        assert logging.getLogger("joblib").level == logging.WARNING


# ---------------------------------------------------------------------------
# alibi-detect
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_alibi_detect"),
    reason="mlserver-alibi-detect not installed",
)
class TestAlibiDetectConfigureFrameworkLogger:
    @pytest.fixture(autouse=True)
    def _reset_alibi_detect_logger(self):
        yield
        logging.getLogger("alibi_detect").setLevel(logging.NOTSET)

    def test_sets_alibi_detect_logger_level(self):
        from mlserver_alibi_detect.runtime import AlibiDetectRuntime

        _set_mlserver_level(logging.INFO)
        AlibiDetectRuntime(settings=_make_settings())
        assert logging.getLogger("alibi_detect").level == logging.INFO


# ---------------------------------------------------------------------------
# alibi-explain
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_alibi_explain"),
    reason="mlserver-alibi-explain not installed",
)
class TestAlibiExplainConfigureFrameworkLogger:
    @pytest.fixture(autouse=True)
    def _reset_alibi_logger(self):
        yield
        logging.getLogger("alibi").setLevel(logging.NOTSET)

    def test_sets_alibi_logger_level(self):
        from mlserver_alibi_explain.runtime import AlibiExplainRuntimeBase
        from mlserver_alibi_explain.common import AlibiExplainSettings

        _set_mlserver_level(logging.WARNING)
        settings = _make_settings()
        explainer_settings = AlibiExplainSettings()
        AlibiExplainRuntimeBase(settings, explainer_settings)
        assert logging.getLogger("alibi").level == logging.WARNING


@pytest.mark.skipif(
    not _can_import("mlserver_alibi_explain"),
    reason="mlserver-alibi-explain not installed",
)
class TestAlibiExplainBlackBoxLoad:
    @pytest.fixture(autouse=True)
    def _reset_alibi_logger(self):
        yield
        logging.getLogger("alibi").setLevel(logging.NOTSET)

    @pytest.mark.asyncio
    async def test_load_preserves_alibi_logger_level(self):
        from mlserver.settings import ModelParameters
        from mlserver_alibi_explain.explainers.black_box_runtime import (
            AlibiExplainBlackBoxRuntime,
        )

        _set_mlserver_level(logging.ERROR)
        settings = _make_settings(
            parameters=ModelParameters(
                extra={
                    "infer_uri": "http://localhost:8080/v2/models/foo/infer",
                    "explainer_type": "anchor_tabular",
                    "init_parameters": {},
                }
            ),
        )
        mock_explainer_cls = MagicMock()
        runtime = AlibiExplainBlackBoxRuntime(settings, mock_explainer_cls)
        await runtime.load()

        assert logging.getLogger("alibi").level == logging.ERROR
        mock_explainer_cls.assert_called_once()

