import builtins

from mlserver.repository.repository import ModelRepository
from mlserver.settings import ModelSettings
from mlserver.errors import ModelNotFound


class DummyModelRepository(ModelRepository):
    def __init__(self, root: str, files: builtins.list[str]) -> None:
        self._model_settings = []

        if files:
            model_settings_files = files
            for model_settings_file in model_settings_files:
                model_settings_path = model_settings_file
                model_settings = ModelSettings.parse_file(model_settings_path)
                self._model_settings.append(model_settings)

    async def list(self) -> builtins.list[ModelSettings]:
        return self._model_settings

    async def find(self, name: str) -> builtins.list[ModelSettings]:
        all_settings = await self.list()
        result: builtins.list[ModelSettings] = []
        for model_settings in all_settings:
            if model_settings.name == name:
                result.append(model_settings)

        if len(result) == 0:
            raise ModelNotFound(name)

        return result
