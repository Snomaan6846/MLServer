import logging

import lightgbm as lgb

from mlserver import types
from mlserver.model import MLModel
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.codecs import NumpyCodec, NumpyRequestCodec

WELLKNOWN_MODEL_FILENAMES = ["model.bst"]


class LightGBMModel(MLModel):
    """
    Implementation of the MLModel interface to load and serve `lightgbm` models.
    """

    def _configure_framework_logger(self) -> None:
        """Register a child logger with LightGBM for clear source identification."""
        level = self._mlserver_log_level
        lgb_logger = logging.getLogger("mlserver.lightgbm")
        lgb_logger.setLevel(level)
        lgb.register_logger(lgb_logger)
        logger.debug(
            "Configured %s framework logger to %s",
            "lightgbm",
            logging.getLevelName(level),
        )

    async def load(self) -> bool:
        model_uri = await get_model_uri(
            self._settings, wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
        )

        self._model = lgb.Booster(model_file=model_uri)

        return True

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        decoded = self.decode_request(payload, default_codec=NumpyRequestCodec)
        prediction = self._model.predict(decoded)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output(
                    name="predict", payload=prediction  # type: ignore
                )
            ],
        )
