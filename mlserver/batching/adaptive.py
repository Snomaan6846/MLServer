import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

from asyncio import Future, Queue, wait_for, Task
from functools import partial, wraps
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

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
_AdaptiveBatcherStreamingTask: ContextVar[Task[Any] | None] = ContextVar(
    "adaptive_batcher_streaming_task", default=None
)
# Dispatch (wire) id of the request currently being queued, set by the
# worker's request processing so remote cancellations can target it.
_CurrentDispatchId: ContextVar[str | None] = ContextVar(
    "adaptive_batcher_dispatch_id", default=None
)


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
        # Wire dispatch id -> batcher internal id, for remote cancellation.
        self._external_ids: dict[str, str] = {}
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
        setattr(self._model, "predict", self._predict_fn)
        setattr(self._model, "predict_stream", self._predict_stream_fn)

        # Remove batcher from model to break reference cycle
        delattr(self._model, _AdaptiveBatchingAttr)

    async def predict(self, req: InferenceRequest) -> InferenceResponse:
        external_id = _CurrentDispatchId.get()
        internal_id, _ = await self._queue_request(req)
        self._start_batcher_if_needed()
        try:
            return await self._wait_response(internal_id)
        finally:
            if external_id is not None:
                self._external_ids.pop(external_id, None)

    def evict(self, external_id: str) -> bool:
        """Cancel a still-queued request by its wire dispatch id.

        Called from the worker's control path when the dispatching
        frontend cancels the request. Cancelling the response future is
        enough: the collection loop skips requests whose future is done,
        so an evicted request never enters a batch. A request already
        merged into an executing batch completes and its (ignored) reply
        is discarded by the frontend.
        """
        internal_id = self._external_ids.pop(external_id, None)
        if internal_id is None:
            return False
        future = self._async_responses.get(internal_id)
        if future is not None and not future.done():
            future.cancel()
        return True

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
        external_id = _CurrentDispatchId.get()
        # An evict racing this map write can miss — fire-and-forget by design.
        if external_id is not None:
            self._external_ids[external_id] = internal_id

        # Create the response future before admission can block: a
        # cancel that arrives while the queue is full must find a
        # cancellable future, not just an ID mapping that gets removed
        # without effect while the request still runs once admitted.
        loop = asyncio.get_running_loop()
        async_response = loop.create_future()
        self._async_responses[internal_id] = async_response

        try:
            # The monitor runs inside the admission try: it touches
            # metric context that a direct (non-server) caller may not
            # have set, and its failure must clean up the registrations
            # above instead of leaking them.
            self._batch_queue_monitor()
            await self._requests.put((internal_id, req))
        except BaseException:
            # Admission failed or was cancelled: the request will never
            # run, so the future and the ID mapping must not leak.
            self._async_responses.pop(internal_id, None)
            if external_id is not None:
                self._external_ids.pop(external_id, None)
            raise

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
            self._async_responses.pop(internal_id, None)

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
                future = self._async_responses.get(internal_id)
                if future is not None and not future.done():
                    future.set_result(response)
        except asyncio.CancelledError:
            for internal_id in batched.inference_requests.keys():
                future = self._async_responses.get(internal_id)
                if future is not None and not future.done():
                    future.cancel()
        except Exception as err:
            for internal_id in batched.inference_requests.keys():
                future = self._async_responses.get(internal_id)
                if future is not None and not future.done():
                    future.set_exception(err)

    async def _batch_requests(self) -> AsyncIterator[BatchedRequests]:
        while not self._requests.empty():
            to_batch: dict[str, InferenceRequest] = {}
            deadline = monotonic() + self._max_batch_time

            try:
                while len(to_batch) < self._max_batch_size:
                    timeout = deadline - monotonic()
                    if timeout <= 0 and (to_batch or self._requests.empty()):
                        break

                    internal_id, inference_request = await self._get_request(
                        timeout=timeout
                    )

                    future = self._async_responses.get(internal_id)
                    if future is None or future.done():
                        continue

                    to_batch[internal_id] = inference_request
            except asyncio.TimeoutError:
                # NOTE: Hit timeout, continue
                pass

            if to_batch:
                # Drop requests cancelled while queued (remote eviction):
                # a done future means nobody is waiting for the result.
                live = {}
                for internal_id, queued_request in to_batch.items():
                    future = self._async_responses.get(internal_id)
                    if future is not None and not future.done():
                        live[internal_id] = queued_request
                if live:
                    yield BatchedRequests(live)

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
        if _AdaptiveBatcherStreamingTask.get() is asyncio.current_task():
            return await f(payload)

        batcher = _get_batcher(f)
        return await batcher.predict(payload)

    return _inner


@contextmanager
def streaming_bypass() -> Iterator[None]:
    """Hold the adaptive-batching bypass for the current task.

    Marks the current asyncio task as serving a single-shot stream so
    the batcher-wrapped ``predict`` runs the model directly instead of
    joining a batch. Used by the single-shot ``/infer_stream``
    shortcut; the streaming wrapper holds the same mark per event.
    """
    token = _AdaptiveBatcherStreamingTask.set(asyncio.current_task())
    try:
        yield
    finally:
        _AdaptiveBatcherStreamingTask.reset(token)


def is_streaming_bypass_active() -> bool:
    """Whether the current task holds the batcher bypass mark."""
    return _AdaptiveBatcherStreamingTask.get() is asyncio.current_task()


async def _bypass_aiter(
    stream: AsyncIterator[InferenceResponse],
) -> AsyncIterator[InferenceResponse]:
    """Iterate a stream with the batcher bypass held across each
    ``anext`` — the fallback's ``predict`` dispatch sees the mark.

    Async generators resume in the consumer's context, so the mark
    must be (re)applied around each advance, not held across yields.
    """
    while True:
        token = _AdaptiveBatcherStreamingTask.set(asyncio.current_task())
        try:
            response = await anext(stream)
        except StopAsyncIteration:
            return
        finally:
            _AdaptiveBatcherStreamingTask.reset(token)

        yield response


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
        stream = aiter(f(payload))
        try:
            async for response in _bypass_aiter(stream):
                yield response
        finally:
            token = _AdaptiveBatcherStreamingTask.set(asyncio.current_task())
            try:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
            finally:
                _AdaptiveBatcherStreamingTask.reset(token)

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
    # Walk the __wrapped__ chain but keep any binding found on the way:
    # a bound method of a decorated function (e.g. @decode_args) still
    # identifies its model, while parallelised custom handlers are
    # unbound closures wrapping the model's bound methods.
    current = f
    name = getattr(current, "__name__", "method")
    while current is not None:
        if hasattr(current, "__self__"):
            return current.__self__
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            break
        current = wrapped
        name = getattr(current, "__name__", name)
    raise InvalidBatchingMethod(name, reason="method is not bound")
