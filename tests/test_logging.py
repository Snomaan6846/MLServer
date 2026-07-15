import logging
import pytest
import json


from mlserver import ModelSettings
from mlserver.context import model_context
from mlserver.logging import (
    ModelLoggerFormatter,
    configure_logger,
    get_log_level,
    logger,
    _STREAM_HANDLER_NAME,
)
from mlserver.settings import ModelParameters, Settings
from tests.fixtures import SumModel
from logging import INFO


@pytest.fixture(autouse=True)
def _restore_mlserver_level():
    """Reset the mlserver logger level after each test."""
    original = logger.level
    yield
    logger.setLevel(original)


@pytest.mark.parametrize(
    "name, version, expected_model_fmt, fmt_present_in_all",
    [
        (
            "foo",
            "v1.0",
            "[foo:v1.0]",
            False,
        ),
        (
            "foo",
            "",
            "[foo]",
            False,
        ),
        (
            "",
            "v1.0",
            "",
            True,
        ),
        (
            "",
            "",
            "",
            True,
        ),
    ],
)
def test_model_logging_formatter_unstructured(
    name: str,
    version: str,
    expected_model_fmt: str,
    fmt_present_in_all: bool,
    settings: Settings,
    caplog,
):
    settings.use_structured_logging = False
    caplog.handler.setFormatter(ModelLoggerFormatter(settings))
    caplog.set_level(INFO)

    model_settings = ModelSettings(
        name=name, implementation=SumModel, parameters=ModelParameters(version=version)
    )

    logger.info("Before model context")
    with model_context(model_settings):
        logger.info("Inside model context")
    logger.info("After model context")

    log_records = caplog.get_records("call")
    assert len(log_records) == 3

    assert all(hasattr(lr, "model") for lr in log_records)

    if fmt_present_in_all:
        assert all(lr.model == expected_model_fmt for lr in log_records)
    else:
        assert expected_model_fmt != log_records[0].model
        assert expected_model_fmt == log_records[1].model
        assert expected_model_fmt != log_records[2].model


@pytest.mark.parametrize(
    "name, version, expected_model_fmt, fmt_present_in_all",
    [
        (
            "foo",
            "v1.0",
            ', "model_name": "foo", "model_version": "v1.0"',
            False,
        ),
        (
            "foo",
            "",
            ', "model_name": "foo"',
            False,
        ),
        (
            "",
            "v1.0",
            "",
            True,
        ),
        ("", "", "", True),
    ],
)
def test_model_logging_formatter_structured(
    name: str,
    version: str,
    expected_model_fmt: str,
    fmt_present_in_all: bool,
    settings: Settings,
    caplog,
):
    settings.use_structured_logging = True
    caplog.handler.setFormatter(ModelLoggerFormatter(settings))
    caplog.set_level(INFO)

    model_settings = ModelSettings(
        name=name, implementation=SumModel, parameters=ModelParameters(version=version)
    )

    logger.info("Before model context")
    with model_context(model_settings):
        logger.info("Inside model context")
    logger.info("After model context")

    _ = [json.loads(lr) for lr in caplog.text.strip().split("\n")]
    log_records = caplog.get_records("call")
    assert len(log_records) == 3

    assert all(hasattr(lr, "model") for lr in log_records)

    if fmt_present_in_all:
        assert all(lr.model == expected_model_fmt for lr in log_records)
    else:
        assert expected_model_fmt != log_records[0].model
        assert expected_model_fmt == log_records[1].model
        assert expected_model_fmt != log_records[2].model


def test_get_log_level_returns_info_by_default(settings: Settings):
    settings.debug = False
    settings.log_level = "INFO"
    configure_logger(settings)
    assert get_log_level() == logging.INFO


@pytest.mark.parametrize(
    "debug, expected_level",
    [(True, logging.DEBUG), (False, logging.INFO)],
)
def test_get_log_level_reflects_debug_setting(
    debug: bool, expected_level: int, settings: Settings
):
    settings.debug = debug
    settings.log_level = "INFO"
    configure_logger(settings)
    assert get_log_level() == expected_level


@pytest.mark.parametrize(
    "log_level, expected",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_configure_logger_respects_log_level(
    log_level: str, expected: int, settings: Settings
):
    settings.debug = False
    settings.log_level = log_level
    configure_logger(settings)
    assert logger.level == expected


def test_configure_logger_log_level_case_insensitive(settings: Settings):
    settings.debug = False
    settings.log_level = "warning"
    configure_logger(settings)
    assert logger.level == logging.WARNING


def test_configure_logger_notset_level_falls_back_to_info():
    with pytest.warns(UserWarning, match="Unrecognised log_level"):
        s = Settings(log_level="NOTSET", _env_file=None)
    assert s.log_level == "INFO"
    configure_logger(s)
    assert logger.level == logging.INFO


def test_configure_logger_direct_mutation_invalid_falls_back_to_info(
    settings: Settings,
):
    settings.debug = False
    settings.log_level = "BOGUS_DIRECT"
    configure_logger(settings)
    assert logger.level == logging.INFO


def test_configure_logger_debug_overrides_log_level(settings: Settings):
    settings.debug = True
    settings.log_level = "ERROR"
    configure_logger(settings)
    assert logger.level == logging.DEBUG


def test_apply_logging_file_defaults_disable_existing_loggers_false(tmp_path):
    from mlserver.logging import apply_logging_file

    sentinel_name = "__test_sentinel__"
    sentinel = logging.getLogger(sentinel_name)
    sentinel.disabled = False

    config = {"version": 1, "loggers": {"mlserver": {"level": "WARNING"}}}
    config_file = tmp_path / "log.json"
    config_file.write_text(json.dumps(config))

    try:
        apply_logging_file(str(config_file))
        assert not sentinel.disabled
    finally:
        logging.Logger.manager.loggerDict.pop(sentinel_name, None)


def test_apply_logging_file_dict_defaults_disable_existing_loggers_false():
    from mlserver.logging import apply_logging_file

    sentinel_name = "__test_sentinel_dict__"
    sentinel = logging.getLogger(sentinel_name)
    sentinel.disabled = False

    try:
        apply_logging_file({"version": 1, "loggers": {"mlserver": {"level": "ERROR"}}})
        assert not sentinel.disabled
    finally:
        logging.Logger.manager.loggerDict.pop(sentinel_name, None)


def test_logging_settings_overrides_debug(settings: Settings, tmp_path):
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {"mlserver": {"level": "WARNING"}},
    }
    config_file = tmp_path / "log_config.json"
    config_file.write_text(json.dumps(log_config))

    settings.debug = True
    settings.logging_settings = str(config_file)
    configure_logger(settings)
    assert logger.level == logging.WARNING


@pytest.mark.parametrize("debug", [True, False])
def test_log_level_gets_persisted(debug: bool, settings: Settings, caplog):
    settings.debug = debug
    configure_logger(settings)

    test_log_message = "foo - bar - this is a test"
    logger.debug(test_log_message)

    if debug:
        assert test_log_message in caplog.text
    else:
        assert test_log_message not in caplog.text


def test_configure_logger_when_called_multiple_times_with_same_logger(settings):
    logger = configure_logger()

    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert handler.name == _STREAM_HANDLER_NAME
    assert (
        hasattr(handler.formatter, "use_structured_logging")
        and handler.formatter.use_structured_logging is False
    )

    logger = configure_logger(settings)

    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert handler.name == _STREAM_HANDLER_NAME
    assert (
        hasattr(handler.formatter, "use_structured_logging")
        and handler.formatter.use_structured_logging is False
    )

    settings.use_structured_logging = True
    logger = configure_logger(settings)

    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert handler.name == _STREAM_HANDLER_NAME
    assert (
        hasattr(handler.formatter, "use_structured_logging")
        and handler.formatter.use_structured_logging is True
    )
