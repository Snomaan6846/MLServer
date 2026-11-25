from typing import Any
from collections.abc import Callable
from enum import Enum
from functools import wraps

from ..batching.adaptive import _bypass_aiter, is_streaming_bypass_active
from ..errors import InferenceError, MLServerError
from ..handlers.custom import get_custom_handlers, register_custom_handler
from ..model import MLModel, PREDICT_NOT_IMPLEMENTED
from ..types import MetadataModelResponse, InferenceRequest, InferenceResponse

from .errors import WorkerError
from .messages import ModelRequestMessage
from .dispatcher import Dispatcher


class ModelMethods(Enum):
    Predict = "predict"
    Metadata = "metadata"


class ParallelModel(MLModel):
    def __init__(self, model: MLModel, dispatcher: Dispatcher):
        super().__init__(model.settings)
        self._on_load_complete: Callable[[], None] = lambda: None
        self._on_load_failed: Callable[[], None] = lambda: None
        self._model = model
        self._dispatcher = dispatcher
        self._metadata: MetadataModelResponse | None = None
        self._add_custom_handlers()

    def _load_complete(self) -> None:
        self._on_load_complete()

    def _load_failed(self) -> None:
        self._on_load_failed()

    def _add_custom_handlers(self):
        custom_handlers = get_custom_handlers(self._model)
        for handler, method in custom_handlers:
            parallelised = self._parallelise(method)
            register_custom_handler(handler, parallelised)
            setattr(self, method.__name__, parallelised)

    def _parallelise(self, method: Callable):
        @wraps(method)
        async def _inner(*args, **kwargs):
            return await self._send(method.__name__, *args, **kwargs)

        return _inner

    async def metadata(self) -> MetadataModelResponse:
        # NOTE: We cache metadata to avoid expensive worker calls for static
        # data
        if self._metadata is None:
            metadata_response = await self._send(ModelMethods.Metadata.value)
            if not isinstance(metadata_response, MetadataModelResponse):
                raise InferenceError(f"Model '{self.name}' returned no metadata")

            self._metadata = metadata_response

        return self._metadata

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        inference_response = await self._send(ModelMethods.Predict.value, payload)
        if not isinstance(inference_response, InferenceResponse):
            raise InferenceError(
                f"Model '{self.name}' returned no predictions after inference"
            )

        return inference_response

    async def _send(self, method_name: str, *args, **kwargs) -> Any | None:
        req_message = ModelRequestMessage(
            model_name=self.name,
            model_version=self.version,
            method_name=method_name,
            method_args=args,
            method_kwargs=kwargs,
            # A single-shot stream fallback holds the batcher-bypass
            # mark in this task; the worker-side batcher cannot see a
            # frontend task-local, so carry the meaning explicitly.
            skip_batching=is_streaming_bypass_active(),
        )

        response_message = await self._dispatcher.dispatch_request(req_message)
        return response_message.return_value

    async def predict_stream(self, payloads):
        """Fail fast when streaming cannot be served through the pool.

        The pool's request/response dispatch cannot carry an async
        generator back from a worker. A model that only implements
        ``predict_stream`` would otherwise reach the worker's
        unimplemented ``predict`` MID-STREAM — after the SSE response
        headers are already sent, the client just sees a truncated
        (empty 200) stream with no error. Reject the request before
        any payload is consumed, with an actionable message.

        Models that also implement ``predict`` keep the base class'
        single-shot fallback (one response per stream, dispatched
        through the regular pool path).
        """
        params = self.settings.parameters
        is_env_model = bool(
            params and (params.environment_tarball or params.environment_path)
        )
        # Environment models are only instantiated inside workers, so
        # their real class is unknown here: their failure is translated
        # from the worker error below instead of pre-checked.
        if not is_env_model and type(self._model).predict is MLModel.predict:
            raise self._streaming_unsupported_error()
        try:
            # The fallback's predict dispatch must bypass the
            # worker-side batcher: hold the bypass mark across each
            # advance so _send carries skip_batching on the dispatch.
            async for response in _bypass_aiter(MLModel.predict_stream(self, payloads)):
                yield response
        except WorkerError as error:
            if PREDICT_NOT_IMPLEMENTED in str(error):
                raise self._streaming_unsupported_error() from error
            raise

    def stream_is_single_shot(self) -> bool:
        """Delegate to the wrapped model; env models are unknown here.

        Environment models are only instantiated inside workers, so
        their real class cannot be checked in this process: report
        multi-shot (the safe producer/queue path applies).
        """
        params = self.settings.parameters
        is_env_model = bool(
            params and (params.environment_tarball or params.environment_path)
        )
        if is_env_model:
            return False
        return self._model.stream_is_single_shot()

    def _streaming_unsupported_error(self) -> MLServerError:
        return MLServerError(
            f"Model '{self.name}' only implements streaming inference, "
            "which the parallel inference pool cannot serve: the pool's "
            "request/response dispatch cannot carry a response stream "
            "back from a worker. Serve this model in-process by setting "
            "the model-level `parallel_workers = 0` (deprecated, and only "
            "supported at `frontend_workers = 0`), or deploy it in a "
            "separate MLServer instance with `parallel_workers = 0`."
        )
