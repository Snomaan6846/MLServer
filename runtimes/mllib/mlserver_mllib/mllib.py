import logging

from mlserver import MLModel, types
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.errors import InferenceError
from pyspark import SparkContext, SparkConf

from .utils import get_mllib_load

_SPARK_LOG_LEVELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
}


class MLlibModel(MLModel):
    def _configure_framework_logger(self) -> None:
        """Configure py4j logger and store Spark level for load().

        ``SparkContext`` is created in ``load()``, so ``sc.setLogLevel()`` is
        applied there.
        """
        level = self._mlserver_log_level
        logging.getLogger("py4j").setLevel(level)
        self._spark_log_level = _SPARK_LOG_LEVELS.get(level, "WARN")
        logger.debug(
            "Configured %s framework logger to %s",
            "mllib",
            logging.getLevelName(level),
        )

    async def load(self) -> bool:
        # TODO: To be more configurable
        # Ref https://spark.apache.org/docs/latest/configuration.html
        conf = SparkConf().set("spark.driver.host", "127.0.0.1")
        sc = SparkContext(appName="MLlibModel", conf=conf)
        sc.setLogLevel(self._spark_log_level)

        model_uri = await get_model_uri(self._settings)
        model_load = await get_mllib_load(self._settings)

        self._model = model_load(sc, model_uri)

        return True

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)
        prediction = self._model.predict(payload.inputs[0].data)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="predict",
                    shape=[1],
                    datatype="FP32",
                    data=prediction,
                )
            ],
        )

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        if len(payload.inputs) != 1:
            raise InferenceError(
                "MLlibModel only supports a single input tensor "
                f"({len(payload.inputs)} were received)"
            )

        return payload
