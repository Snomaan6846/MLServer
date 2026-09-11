import time
import asyncio

from asyncio import Future, Queue, wait_for, Task
from functools import partial, wraps
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import status

from ..errors import MLServerError
from ..logging import logger
from ..model import MLModel
from ..types import (
    InferenceRequest,
    InferenceResponse,
)
from ..utils import generate_uuid, schedule_with_callback, get_wrapped_method
from .. import metrics

from .requests import BatchedRequests

_AdaptiveBatchingAttr = "__adaptive_batching__"


class InvalidBatchingMethod(MLServerError):
    def __init__(self, method_name: str, reason: str | None = None):
        msg = f"Method {method_name} can't be used for adaptive batching"
        if reason:
            msg += f": {reason}"

        super().__init__(msg)


class AdaptiveBatcher:
    def __init__(self, model: MLModel):
        self._model = model

        self._max_batch_size = model.settings.max_batch_size
        self._max_batch_time = model.settings.max_batch_time

        # Save predict function before it gets decorated
        self._predict_fn = model.predict
        self._predict_stream_fn = model.predict_stream
        self.__requests: Queue[tuple[str, InferenceRequest]] | None = None
        self._async_responses: dict[str, Future[InferenceResponse]] = {}
        self._batching_task = None
        metrics.register("batch_request_queue", "counter of request queue batch size")

    @classmethod
    def get_batcher(cls, model: MLModel) -> "AdaptiveBatcher | None":
        return getattr(model, _AdaptiveBatchingAttr, None)

    def setup(self) -> None:
        setattr(self._model, _AdaptiveBatchingAttr, self)

        # Decorate predict methods
        setattr(self._model, "predict", adaptive_batching(self._model.predict))
        setattr(
            self._model,
            "predict_stream",
            not_implemented_warning(self._model.predict_stream),
        )

    async def teardown(self) -> None:
        # Cancel the background batching task
        if self._batching_task and not self._batching_task.done():
            self._batching_task.cancel()
            try:
                await self._batching_task
            except asyncio.CancelledError:
                pass

        # Remove instance overrides so the model's class methods are visible again
        delattr(self._model, "predict")
        delattr(self._model, "predict_stream")

        # Remove batcher from model to break reference cycle
        delattr(self._model, _AdaptiveBatchingAttr)

    async def predict(self, req: InferenceRequest) -> InferenceResponse:
        internal_id, _ = await self._queue_request(req)
        self._start_batcher_if_needed()
        return await self._wait_response(internal_id)

    @property
    def _requests(self) -> Queue[tuple[str, InferenceRequest]]:
        # NOTE: We need to create Queue within the async request path (and not
        # during __init__!!) to ensure that it shares the same AsyncIO loop.
        if self.__requests is None:
            self.__requests = Queue(maxsize=self._max_batch_size)

        return self.__requests

    async def _queue_request(
        self,
        req: InferenceRequest,
    ) -> tuple[str, Awaitable[InferenceResponse]]:
        internal_id = generate_uuid()
        self._batch_queue_monitor()
        await self._requests.put((internal_id, req))

        loop = asyncio.get_running_loop()
        async_response = loop.create_future()
        self._async_responses[internal_id] = async_response

        return internal_id, async_response

    def _batch_queue_monitor(self):
        """Monitorize batch queue size"""
        batch_queue_size = self._requests.qsize()
        metrics.log(batch_request_queue=batch_queue_size)

    async def _wait_response(self, internal_id: str) -> InferenceResponse:
        async_response = self._async_responses[internal_id]

        try:
            response = await async_response
            return response
        finally:
            del self._async_responses[internal_id]

    def _start_batcher_if_needed(self):
        if self._batching_task is not None:
            if not self._batching_task.done():
                # If task hasn't finished yet, let it keep running
                return

        self._batching_task = schedule_with_callback(
            self._batcher(), self._batching_task_callback
        )

    def _batching_task_callback(self, batching_task: Task):
        if batching_task.cancelled():
            self._clear_queue(
                MLServerError(
                    "Batching task cancelled", status.HTTP_503_SERVICE_UNAVAILABLE
                )
            )
            return
        err = batching_task.exception()
        if err:
            # Clear queue
            self._clear_queue(err)

    def _clear_queue(self, err: BaseException):
        # Cancel all pending async responses
        for async_response in self._async_responses.values():
            if not async_response.done():
                async_response.set_exception(err)

        # Empty queue
        for _ in range(self._requests.qsize()):
            self._requests.get_nowait()

    async def _batcher(self):
        async for batched in self._batch_requests():
            # We run prediction as a Task to ensure it gets scheduled
            # immediately.
            # That way, we can process multiple batches concurrently.
            schedule_with_callback(
                self._predict_fn(batched.merged_request),
                partial(self._predict_callback, batched),
            )

    def _predict_callback(self, batched: BatchedRequests, predict_task: Task):
        try:
            batched_response = predict_task.result()
            responses = batched.split_response(batched_response)
            for internal_id, response in responses.items():
                self._async_responses[internal_id].set_result(response)
        except Exception as err:
            for internal_id in batched.inference_requests.keys():
                self._async_responses[internal_id].set_exception(err)

    async def _batch_requests(self) -> AsyncIterator[BatchedRequests]:
        while not self._requests.empty():
            to_batch: dict[str, InferenceRequest] = {}
            start = time.time()
            timeout = self._max_batch_time

            try:
                while len(to_batch) < self._max_batch_size:
                    internal_id, inference_request = await self._get_request(
                        timeout=timeout
                    )
                    to_batch[internal_id] = inference_request

                    # Update remaining timeout
                    current = time.time()
                    timeout = self._max_batch_time - (current - start)
            except asyncio.TimeoutError:
                # NOTE: Hit timeout, continue
                pass

            yield BatchedRequests(to_batch)

    async def _get_request(self, timeout: float) -> tuple[str, InferenceRequest]:
        if not self._requests.empty():
            return await self._requests.get()

        read_op = self._requests.get()
        return await wait_for(read_op, timeout=timeout)


def adaptive_batching(f: Callable[[InferenceRequest], Awaitable[InferenceResponse]]):
    """
    Decorator for the `predict()` method which will ensure it uses the
    underlying adaptive batcher instance.
    """

    @wraps(f)
    async def _inner(payload: InferenceRequest) -> InferenceResponse:
        batcher = _get_batcher(f)
        return await batcher.predict(payload)

    return _inner


def not_implemented_warning(
    f: Callable[[AsyncIterator[InferenceRequest]], AsyncIterator[InferenceResponse]],
):
    """
    Decorator to lets users know that adaptive batching is not required on
    method `f`.
    """
    model = _get_model(f)
    logger.warning(
        f"Adaptive Batching is enabled for model '{model.name}'"
        " but not supported for inference streaming."
        " Falling back to non-batched inference streaming."
    )

    @wraps(f)
    async def _inner_stream(
        payload: AsyncIterator[InferenceRequest],
    ) -> AsyncIterator[InferenceResponse]:
        async for response in f(payload):
            yield response

    return _inner_stream


def _get_batcher(f: Callable) -> AdaptiveBatcher:
    wrapped_f = get_wrapped_method(f)
    model = _get_model(f)

    if not hasattr(model, _AdaptiveBatchingAttr):
        raise InvalidBatchingMethod(
            wrapped_f.__name__, reason="adaptive batching has not been loaded"
        )

    return getattr(model, _AdaptiveBatchingAttr)


def _get_model(f: Callable) -> MLModel:
    wrapped_f = get_wrapped_method(f)
    if not hasattr(wrapped_f, "__self__"):
        raise InvalidBatchingMethod(wrapped_f.__name__, reason="method is not bound")

    return getattr(wrapped_f, "__self__")
