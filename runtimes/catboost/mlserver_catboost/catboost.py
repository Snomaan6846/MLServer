import logging

from catboost import CatBoostClassifier

from mlserver import types
from mlserver.model import MLModel
from mlserver.utils import get_model_uri
from mlserver.codecs import NumpyCodec, NumpyRequestCodec
from mlserver.logging import logger


WELLKNOWN_MODEL_FILENAMES = ["model.cbm", "model.bin"]

# CatBoost verbosity (least → most): Silent < Verbose < Info < Debug.
_CB_LOG_LEVEL = {
    logging.DEBUG: "Debug",
    logging.INFO: "Info",
    logging.WARNING: "Verbose",
    logging.ERROR: "Silent",
    logging.CRITICAL: "Silent",
}


class CatboostModel(MLModel):
    """
    Implementation of the MLModel interface to load and serve `catboost` models.
    """

    def _configure_framework_logger(self) -> None:
        """Store CatBoost log level for use in load().

        CatBoost accepts ``logging_level`` as a constructor parameter rather
        than via a global API, so the mapped value is applied when the model
        is created.
        """
        level = self._mlserver_log_level
        self._catboost_log_level = _CB_LOG_LEVEL.get(level, "Info")
        logger.debug(
            "Configured %s framework logger to %s",
            "catboost",
            logging.getLevelName(level),
        )

    async def load(self) -> bool:
        model_uri = await get_model_uri(
            self._settings, wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
        )

        self._model = CatBoostClassifier(logging_level=self._catboost_log_level)
        self._model.load_model(model_uri)
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        decoded = self.decode_request(payload, default_codec=NumpyRequestCodec)
        prediction = self._model.predict(decoded)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[NumpyCodec.encode(name="predict", payload=prediction)],
        )
