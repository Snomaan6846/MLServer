from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from prometheus_client import (
    Counter,
    Summary,
)
from collections.abc import AsyncIterator, Iterator

from ..model import MLModel
from ..errors import ModelNotReady
from ..context import model_context
from ..settings import Settings
from ..registry import MultiModelRegistry
from ..types import (
    MetadataModelResponse,
    MetadataServerResponse,
    InferenceRequest,
    InferenceResponse,
    RuntimeSecurityResponse,
    RuntimeSecurityMode,
)
from ..middleware import InferenceMiddlewares
from ..cloudevents import CloudEventsMiddleware
from ..utils import generate_uuid


@dataclass
class _InferenceMetrics:
    enabled: bool = True
    excluded_duration: float = 0.0
    inference_succeeded: bool = False


class DataPlane:
    """
    Internal implementation of handlers, used by both the gRPC and REST
    servers.

    It also tracks startup completion. Until `startup_complete()` is called,
    the readiness check reports not ready.
    """

    def __init__(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        *,
        inference_metrics: bool = True,
    ):
        self._settings = settings
        self._model_registry = model_registry
        self._inference_metrics_enabled = inference_metrics
        self._startup_complete = False
        self._inference_middleware = InferenceMiddlewares(
            CloudEventsMiddleware(settings)
        )

        # TODO: Update to standardised set of labels
        self._ModelInferRequestSuccess = Counter(
            "model_infer_request_success",
            "Model infer request success count",
            ["model", "version"],
        )
        self._ModelInferRequestFailure = Counter(
            "model_infer_request_failure",
            "Model infer request failure count",
            ["model", "version"],
        )
        self._ModelInferRequestDuration = Summary(
            "model_infer_request_duration",
            "Model infer request duration",
            ["model", "version"],
        )

    @property
    def is_startup_complete(self) -> bool:
        """
        Whether the initial server startup has completed.

        The readiness check reports not ready until startup is marked complete,
        regardless of the readiness settings. This prevents traffic from being
        routed to the server before the initial startup phase actions have
        completed.
        """
        return self._startup_complete

    def startup_complete(self) -> None:
        """
        Mark that the initial server startup has completed.

        This should only be called by the server's main startup process after all
        initial models have been loaded. From then on, the readiness check
        follows `strict_readiness` and `empty_registry_readiness`.
        """
        self._startup_complete = True

    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        """
        Server readiness.

        Never ready before startup completes. After that,
        `empty_registry_readiness` decides what an empty registry means, and
        `strict_readiness` decides whether the state of the loaded models is
        taken into account.
        """
        # Return false if startup has not completed
        if not self._startup_complete:
            return False

        # Fetch models from the registry
        models = list(await self._model_registry.get_models())

        # Apply empty readiness mode based on setting
        if not models:
            return self._settings.empty_registry_readiness

        # Apply strict or lenient readiness mode based on setting
        if self._settings.strict_readiness:
            # Strict mode (V2 protocol compliant): ALL models must be ready
            return all(model.ready for model in models)
        else:
            # Lenient mode: model state is ignored so that a model that is still
            # loading doesn't block traffic to any other loaded models or new
            # model lifecycle requests.
            return True

    async def model_ready(self, name: str, version: str | None = None) -> bool:
        model = await self._model_registry.get_model(name, version)
        return model.ready

    async def metadata(self) -> MetadataServerResponse:
        # Built-in extensions that are always available
        builtin_extensions = ["model_repository", "runtime_security"]
        if self._settings.rest_binary_tensor:
            # KServe v2 Binary Tensor Data Extension
            builtin_extensions.append("binary_tensor_data")
        # Combine with user-configured extensions
        all_extensions = builtin_extensions + self._settings.extensions
        return MetadataServerResponse(
            name=self._settings.server_name,
            version=self._settings.server_version,
            extensions=all_extensions,
        )

    @staticmethod
    async def runtimes() -> RuntimeSecurityResponse:
        from ..settings import (
            _load_image_baked_allowed_model_implementations,
            _get_trusted_runtimes_artifact_path,
        )

        allowed = _load_image_baked_allowed_model_implementations(
            _get_trusted_runtimes_artifact_path()
        )

        if allowed is not None:
            # Production mode: file exists
            return RuntimeSecurityResponse(
                mode=RuntimeSecurityMode.PRODUCTION,
                allowed_model_implementations=sorted(allowed),
            )
        else:
            # Development mode: no file
            return RuntimeSecurityResponse(
                mode=RuntimeSecurityMode.DEVELOPMENT,
                allowed_model_implementations=None,
            )

    async def model_metadata(
        self, name: str, version: str | None = None
    ) -> MetadataModelResponse:
        model = await self._model_registry.get_model(name, version)
        # TODO: Make await optional for sync methods
        with model_context(model.settings):
            return await model.metadata()

    async def infer(
        self,
        payload: InferenceRequest,
        name: str,
        version: str | None = None,
        model: MLModel | None = None,
    ) -> InferenceResponse:
        # ``model``: a pre-resolved instance (the /infer_stream
        # branch decision and its execution must use the SAME model —
        # a reload can swap the registration between the two).
        async with self._infer_contextmanager(name, version, model) as resolved:
            payload = self._prepare_payload(payload, resolved)

            prediction = await resolved.predict(payload)

            # Ensure ID matches
            prediction.id = payload.id
            self._inference_middleware.response_middleware(
                prediction, resolved.settings
            )
            return prediction

    async def infer_stream(
        self,
        payloads: AsyncIterator[InferenceRequest],
        name: str,
        version: str | None = None,
        model: MLModel | None = None,
    ) -> AsyncIterator[InferenceResponse]:
        # ``model``: see infer() — the pre-resolved instance.
        async with self._infer_contextmanager(name, version, model) as resolved:
            # we need to get the first payload to get the ID
            async for payload in payloads:
                break

            payloads_decorated = self._prepare_payloads_generator(
                payload, payloads, resolved
            )
            stream = resolved.predict_stream(payloads_decorated)
            try:
                async for prediction in stream:
                    prediction.id = payload.id  # Ensure ID matches
                    self._inference_middleware.response_middleware(
                        prediction, resolved.settings
                    )
                    yield prediction
            finally:
                # Closing THIS generator (the SSE chain's aclose)
                # must also close the model's iterator: suspended at
                # a yield, it holds model-side resources nothing else
                # unwinds. Iterators without aclose (a plain
                # async-for source) are left to their own lifecycle.
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    with suppress(Exception):
                        await aclose()

    @staticmethod
    def _has_array_inputs(payload: InferenceRequest) -> bool:
        """Whether any input carries an ndarray payload.

        Binary REST requests (like gRPC raw input buffers) hand compact
        NumPy views to the model path.  List-based adaptive batching
        cannot merge them, so they are restored to the public list
        representation before batching.
        """
        return any(
            isinstance(getattr(tensor.data, "root", tensor.data), np.ndarray)
            for tensor in payload.inputs
        )

    async def resolve_model(self, name: str, version: str | None = None) -> MLModel:
        """The active model for a request-branch decision."""
        return await self._model_registry.get_model(name, version)

    def _prepare_payload(
        self, payload: InferenceRequest, model: MLModel
    ) -> InferenceRequest:
        if (model.settings.max_batch_size or model.settings.max_batch_time) and (
            self._settings.grpc_raw_input_buffers or self._has_array_inputs(payload)
        ):
            from ..grpc.buffers import restore_list_inputs

            # Batching merges the public list representation. Keep compact
            # NumPy buffers on the measured unbatched path (gRPC raw input
            # buffers and REST binary tensors alike), while preserving the
            # existing batching contract without changing the batcher.
            restore_list_inputs(payload)
        if payload.id is None:
            payload.id = generate_uuid()

        self._inference_middleware.request_middleware(payload, model.settings)
        return payload

    async def _prepare_payloads_generator(
        self,
        first_payload: InferenceRequest,
        subsequent_payloads: AsyncIterator[InferenceRequest],
        model: MLModel,
    ) -> AsyncIterator[InferenceRequest]:
        # yield the first payload after preparing it
        first_payload = self._prepare_payload(first_payload, model)
        yield first_payload

        # Yield the rest of the payloads after preparing them
        # and set the ID to match the first payload. Note that
        # we don't make any assumptions about how many inputs and
        # outputs there are. Thus, everything gets the same ID, cause
        # otherwise we could have one to many, many to one, or many to
        # many id mappings.
        async for payload in subsequent_payloads:
            payload.id = first_payload.id
            payload = self._prepare_payload(payload, model)
            yield payload

    @contextmanager
    def _infer_metrics(
        self, name: str, version: str | None
    ) -> Iterator[_InferenceMetrics]:
        # Raw REST learns whether the body is valid from the worker's reply.
        # Defer recording until that is known; schema errors never entered the
        # standard DataPlane and must not become inference failures here.
        observation = _InferenceMetrics(enabled=self._inference_metrics_enabled)
        started = perf_counter()
        try:
            yield observation
        except BaseException as error:
            # Match count_exceptions(): caller cancellation is not an inference
            # failure, but a completed timing observation still belongs to it.
            if observation.enabled:
                if observation.inference_succeeded:
                    self._ModelInferRequestSuccess.labels(
                        model=name, version=version
                    ).inc()
                elif isinstance(error, Exception):
                    self._ModelInferRequestFailure.labels(
                        model=name, version=version
                    ).inc()
            raise
        else:
            if observation.enabled:
                self._ModelInferRequestSuccess.labels(model=name, version=version).inc()
        finally:
            if observation.enabled:
                elapsed = max(
                    0.0, perf_counter() - started - observation.excluded_duration
                )
                self._ModelInferRequestDuration.labels(
                    model=name, version=version
                ).observe(elapsed)

    @asynccontextmanager
    async def _infer_contextmanager(
        self,
        name: str,
        version: str | None = None,
        model: MLModel | None = None,
    ) -> AsyncIterator[MLModel]:
        with self._infer_metrics(name, version):
            # A pre-resolved instance skips the registry lookup: the
            # caller already decided on it (see infer's ``model``).
            if model is None:
                model = await self._model_registry.get_model(name, version)
            if not model.ready:
                raise ModelNotReady(name, version)

            with model_context(model.settings):
                yield model
