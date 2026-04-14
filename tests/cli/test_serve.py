import os
import sys

from mlserver.cli.serve import DEFAULT_SETTINGS_FILENAME, load_settings
from mlserver.settings import Settings, ModelSettings


async def _load_settings_without_sys_path_mutation(model_folder: str):
    original_sys_path_obj = sys.path
    original_sys_path_entries = original_sys_path_obj[:]
    observed_sys_path_obj = original_sys_path_obj
    observed_sys_path_entries = original_sys_path_entries
    try:
        result = await load_settings(model_folder)
    finally:
        observed_sys_path_obj = sys.path
        observed_sys_path_entries = sys.path[:]
        sys.path = original_sys_path_obj
        original_sys_path_obj[:] = original_sys_path_entries
    assert observed_sys_path_obj is original_sys_path_obj
    assert observed_sys_path_entries == original_sys_path_entries
    return result


async def test_load_models(sum_model_settings: ModelSettings, model_folder: str):
    _, models_settings = await _load_settings_without_sys_path_mutation(model_folder)

    assert len(models_settings) == 1

    model_settings = models_settings[0]
    parameters = models_settings[0].parameters
    assert model_settings.name == sum_model_settings.name
    assert parameters.version == sum_model_settings.parameters.version  # type: ignore


async def test_disable_load_models(settings: Settings, model_folder: str):
    settings.load_models_at_startup = False

    settings_path = os.path.join(model_folder, DEFAULT_SETTINGS_FILENAME)
    with open(settings_path, "w") as settings_file:
        settings_file.write(settings.model_dump_json())

    _, models_settings = await _load_settings_without_sys_path_mutation(model_folder)

    assert len(models_settings) == 0
