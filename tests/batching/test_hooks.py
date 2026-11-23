import asyncio
import json
import os
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock

import pytest

from mlserver.model import MLModel
from mlserver.parallel.model import ParallelModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.batching.hooks import load_batching, unload_batching
from mlserver.batching.adaptive import AdaptiveBatcher

from tests.fixtures import SumModel


@pytest.fixture(autouse=True)
def _batch_record_env(tmp_path, monkeypatch):
    """Expose a batch-record file to worker-side models.

    Runs before the ``mlserver`` fixture so worker processes (forked or
    spawned during its setup) inherit the path.
    """
    monkeypatch.setenv("BATCH_RECORD_PATH", str(tmp_path / "batch-records.jsonl"))


class RecordingBatchModel(SumModel):
    """Worker-side model recording the row count of every merged batch.

    Batch formation is observed where the model executes: the merged
    request arrives either from the worker-side batcher (supervised
    topology, frontend_workers >= 1) or from the main-process batcher
    (direct server, frontend_workers == 0) — one record per execution
    either way.
    """

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        record_path = os.environ.get("BATCH_RECORD_PATH")
        if record_path:
            with open(record_path, "a") as records:
                records.write(
                    json.dumps({"model": self.name, "rows": payload.inputs[0].shape[0]})
                    + "\n"
                )
        return await super().predict(payload)


async def test_batching_predict(
    sum_model: MLModel, inference_request: InferenceRequest
):
    await load_batching(sum_model)
    response = await sum_model.predict(inference_request)

    assert response is not None
    assert isinstance(response, InferenceResponse)
    assert len(response.outputs) == 1


async def test_batching_predict_stream(
    text_stream_model: MLModel, generate_request: InferenceRequest, caplog
):
    # Force batching to be enabled
    text_stream_model.settings.max_batch_size = 10
    text_stream_model.settings.max_batch_time = 0.4
    await load_batching(text_stream_model)

    async def get_stream_request(request):
        yield request

    stream = text_stream_model.predict_stream(get_stream_request(generate_request))
    responses = [r async for r in stream]

    assert len(responses) > 0
    assert isinstance(responses[0], InferenceResponse)
    assert len(responses[0].outputs) == 1
    assert "not supported for inference streaming" in caplog.records[0].message


async def test_batching_default_predict_stream_bypasses_batcher(
    text_model: MLModel, generate_request: InferenceRequest, mocker
):
    text_model.settings.max_batch_size = 10
    text_model.settings.max_batch_time = 0.4
    await load_batching(text_model)

    batcher = AdaptiveBatcher.get_batcher(text_model)
    assert batcher is not None
    predict_mock = mocker.patch.object(
        batcher,
        "predict",
        wraps=batcher.predict,
    )

    async def get_stream_request(request):
        yield request

    stream = text_model.predict_stream(get_stream_request(generate_request))
    try:
        response = await anext(stream)

        assert len(response.outputs) == 1
        predict_mock.assert_not_awaited()

        await text_model.predict(generate_request)
        predict_mock.assert_awaited_once_with(generate_request)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    assert not batcher._async_responses


async def test_streaming_bypass_is_limited_to_stream_task(
    text_model: MLModel, generate_request: InferenceRequest, mocker
):
    """Streaming cleanup bypasses batching, but inherited child contexts do not."""
    text_model.settings.max_batch_size = 10
    text_model.settings.max_batch_time = 0.05

    child_released = asyncio.Event()

    async def custom_stream(
        self, payload: AsyncIterator[InferenceRequest]
    ) -> AsyncIterator[InferenceResponse]:
        async def child_predict():
            await child_released.wait()
            await self.predict(generate_request)

        child_task = asyncio.create_task(child_predict())
        try:
            yield await self.predict(generate_request)
        finally:
            # This call is made from aclose() and remains in the streaming
            # bypass scope.
            await self.predict(generate_request)
            child_released.set()
            await child_task

    setattr(text_model, "predict_stream", MethodType(custom_stream, text_model))
    await load_batching(text_model)

    batcher = AdaptiveBatcher.get_batcher(text_model)
    assert batcher is not None
    predict_mock = mocker.patch.object(batcher, "predict", wraps=batcher.predict)

    async def stream_requests() -> AsyncIterator[InferenceRequest]:
        yield generate_request

    stream = text_model.predict_stream(stream_requests())
    try:
        await anext(stream)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    # Only the child task should enter the batcher. The generator's cleanup
    # runs in the stream task and is still covered by the bypass.
    predict_mock.assert_awaited_once_with(generate_request)


@pytest.mark.parametrize(
    "max_batch_size, max_batch_time",
    [
        (0, 10),
        (-1, 10),
        (1, 10),
        (10, 0),
        (10, -1),
        (0, 0),
    ],
)
async def test_load_batching_disabled(
    max_batch_size: int, max_batch_time: float, sum_model: MLModel
):
    sum_model.settings.max_batch_size = max_batch_size
    sum_model.settings.max_batch_time = max_batch_time

    expected = sum_model.predict
    await load_batching(sum_model)

    assert expected == sum_model.predict  # type: ignore


async def test_unload_batching_removes_batcher(sum_model: MLModel):
    """Test that unload_batching removes the batcher attribute"""
    # Enable and load batching
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4
    await load_batching(sum_model)

    # Batcher should exist
    assert AdaptiveBatcher.get_batcher(sum_model) is not None

    # Unload batching
    await unload_batching(sum_model)

    # Batcher should be removed
    assert AdaptiveBatcher.get_batcher(sum_model) is None


async def test_unload_batching_idempotent(sum_model: MLModel):
    """Test that unload_batching is safe to call on model without batching"""
    # Don't load batching
    assert AdaptiveBatcher.get_batcher(sum_model) is None

    # Should not raise
    result = await unload_batching(sum_model)
    assert result is sum_model


async def test_unload_batching_restores_predict_and_predict_stream_methods(
    sum_model: MLModel,
):
    """Test that unload_batching restores original predict and predict_stream methods"""
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4

    original_predict = sum_model.predict
    original_predict_stream = sum_model.predict_stream

    await load_batching(sum_model)
    assert AdaptiveBatcher.get_batcher(sum_model) is not None

    await unload_batching(sum_model)

    assert sum_model.predict == original_predict
    assert sum_model.predict_stream == original_predict_stream
    assert AdaptiveBatcher.get_batcher(sum_model) is None


async def test_load_batching_for_parallel_model(sum_model: MLModel):
    """The load_batching hook wraps any model it is given (hook unit test)."""
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4

    parallel_model = ParallelModel(sum_model, MagicMock())
    await load_batching(parallel_model)

    assert AdaptiveBatcher.get_batcher(parallel_model) is not None


async def test_parallel_registry_batches_per_model_and_maps_responses(
    mlserver, sum_model_settings, inference_request: InferenceRequest
):
    """Worker-side batchers merge requests per model and map responses."""
    settings_by_model = []
    for name in ("batched-model-a", "batched-model-b"):
        settings = deepcopy(sum_model_settings)
        settings.name = name
        settings.max_batch_size = 2
        settings.max_batch_time = 0.2
        settings.implementation = RecordingBatchModel
        settings_by_model.append(settings)

    record_path = Path(os.environ["BATCH_RECORD_PATH"])
    record_path.unlink(missing_ok=True)

    models = []

    try:
        for settings in settings_by_model:
            models.append(await mlserver._model_registry.load(settings))

        # At frontend_workers=0 (this fixture's default) the batcher
        # stays in the MAIN process, wrapping the ParallelModel — the
        # previous behaviour, restored. The workers do not install
        # their own batcher in this topology.
        assert all(AdaptiveBatcher.get_batcher(model) is not None for model in models)

        # Use different values and minibatch sizes so that incorrect merging,
        # response slicing, or request-to-response association is observable.
        request_data_by_model = (
            (
                ([[1, 2, 3]], [6]),
                ([[10, 11, 12], [20, 21, 22]], [33, 63]),
            ),
            (
                ([[100, 101, 102]], [303]),
                ([[1000, 1001, 1002], [2000, 2001, 2002]], [3003, 6003]),
            ),
        )
        requests = [
            [
                inference_request.copy(
                    deep=True,
                    update={
                        "id": f"{model.name}-{index}",
                        "inputs": [
                            inference_request.inputs[0].copy(
                                update={
                                    "shape": [len(values), 3],
                                    "data": values,
                                }
                            )
                        ],
                    },
                )
                for index, (values, _) in enumerate(model_request_data)
            ]
            for model, model_request_data in zip(models, request_data_by_model)
        ]
        expected_outputs = [
            expected
            for model_request_data in request_data_by_model
            for _, expected in model_request_data
        ]

        responses = await asyncio.gather(
            *[
                model.predict(request)
                for model, model_requests in zip(models, requests)
                for request in model_requests
            ]
        )

        assert [response.id for response in responses] == [
            request.id for model_requests in requests for request in model_requests
        ]
        assert [response.model_name for response in responses] == [
            model.name
            for model_requests, model in zip(requests, models)
            for _ in model_requests
        ]
        assert [list(response.outputs[0].data) for response in responses] == [
            expected for expected in expected_outputs
        ]
        # Exactly one merged execution per model, each observing all
        # three rows: at frontend_workers=0 the MAIN-process batcher
        # merges, and the worker's model sees the merged request.
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        assert sorted((r["model"], r["rows"]) for r in records) == [
            ("batched-model-a", 3),
            ("batched-model-b", 3),
        ]
    finally:
        for model in models:
            await mlserver._model_registry.unload(model.name)


async def test_evict_cancels_queued_request(
    sum_model: MLModel, inference_request: InferenceRequest, mocker
):
    """A remote cancellation evicts a queued request before it merges."""
    from mlserver.batching.adaptive import _CurrentDispatchId

    sum_model.settings.max_batch_size = 5
    sum_model.settings.max_batch_time = 0.3
    await load_batching(sum_model)
    try:
        batcher = AdaptiveBatcher.get_batcher(sum_model)
        assert batcher is not None

        # Signal when the request is admitted to the batch queue — a
        # fixed sleep cannot prove the collector queued it.
        admitted = asyncio.Event()
        original_queue = batcher._queue_request

        async def signalling_queue(req):
            result = await original_queue(req)
            admitted.set()
            return result

        mocker.patch.object(batcher, "_queue_request", signalling_queue)

        token = _CurrentDispatchId.set("dispatch-evict")
        try:
            task = asyncio.create_task(sum_model.predict(inference_request))
            await asyncio.wait_for(admitted.wait(), timeout=5)
            assert batcher.evict("dispatch-evict") is True
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            _CurrentDispatchId.reset(token)
    finally:
        await unload_batching(sum_model)


async def test_evicted_request_never_executes(
    sum_model: MLModel, inference_request: InferenceRequest, mocker
):
    """The evicted request does not reach the merged execution."""
    from mlserver.batching.adaptive import _CurrentDispatchId

    sum_model.settings.max_batch_size = 5
    sum_model.settings.max_batch_time = 0.25
    await load_batching(sum_model)
    try:
        batcher = AdaptiveBatcher.get_batcher(sum_model)
        assert batcher is not None
        spy = mocker.patch.object(batcher, "_predict_fn", wraps=batcher._predict_fn)

        admitted = asyncio.Event()
        original_queue = batcher._queue_request

        async def signalling_queue(req):
            result = await original_queue(req)
            admitted.set()
            return result

        mocker.patch.object(batcher, "_queue_request", signalling_queue)

        token = _CurrentDispatchId.set("dispatch-ghost")
        try:
            task = asyncio.create_task(sum_model.predict(inference_request))
            await asyncio.wait_for(admitted.wait(), timeout=5)
            assert batcher.evict("dispatch-ghost") is True
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            _CurrentDispatchId.reset(token)

        # Wait for the batch window to close (the collector task
        # drains the empty batch) before checking the spy.
        await asyncio.sleep(sum_model.settings.max_batch_time + 0.1)
        spy.assert_not_awaited()
    finally:
        await unload_batching(sum_model)


async def test_evict_unknown_id_is_false(sum_model: MLModel):
    sum_model.settings.max_batch_size = 5
    sum_model.settings.max_batch_time = 0.1
    await load_batching(sum_model)
    batcher = AdaptiveBatcher.get_batcher(sum_model)
    assert batcher is not None
    assert batcher.evict("never-queued") is False


async def test_direct_server_zmq_keeps_batcher_in_main_process(
    prometheus_registry,
    sum_model_settings,
    inference_request: InferenceRequest,
):
    """End-to-end at frontend_workers=0 (the direct server) with the
    zmq transport: the batcher stays in the MAIN process (the previous
    behaviour) and still merges concurrent requests into one execution
    per model."""
    from mlserver import MLServer
    from mlserver.settings import Settings
    from mlserver.utils import generate_uuid
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    batched = deepcopy(sum_model_settings)
    batched.name = "zmq-batched"
    batched.max_batch_size = 2
    batched.max_batch_time = 0.2
    batched.implementation = RecordingBatchModel

    record_path = Path(os.environ["BATCH_RECORD_PATH"])
    record_path.unlink(missing_ok=True)

    server = MLServer(settings)
    task = asyncio.create_task(server.start())
    try:
        model = await server._model_registry.load(batched)
        # The batcher wraps the parallel wrapper in the MAIN process.
        assert AdaptiveBatcher.get_batcher(model) is not None

        requests = [
            inference_request.copy(
                deep=True,
                update={
                    "id": generate_uuid(),
                    "inputs": [
                        inference_request.inputs[0].copy(
                            update={"shape": [1, 3], "data": values}
                        )
                    ],
                },
            )
            for values in ([[1, 2, 3]], [[10, 20, 30]])
        ]
        responses = await asyncio.gather(
            *[model.predict(request) for request in requests]
        )
        assert [list(response.outputs[0].data) for response in responses] == [
            [6],
            [60],
        ]
        # One merged execution observed at the model: the main-process
        # batcher coalesced both requests before dispatch.
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        assert [(r["model"], r["rows"]) for r in records] == [("zmq-batched", 2)]
    finally:
        await server.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_no_pool_batches_in_process_unchanged(
    prometheus_registry,
    sum_model_settings,
    inference_request: InferenceRequest,
):
    """At parallel_workers=0 (no inference pool) the batcher wraps the
    in-process model in the main registry — the branch untouched by
    the frontend work, locked here as the no-change contract."""
    from mlserver import MLServer
    from mlserver.settings import Settings
    from mlserver.utils import generate_uuid
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        parallel_workers=0,
    )
    batched = deepcopy(sum_model_settings)
    batched.name = "inproc-batched"
    batched.max_batch_size = 2
    batched.max_batch_time = 0.2
    batched.implementation = RecordingBatchModel

    record_path = Path(os.environ["BATCH_RECORD_PATH"])
    record_path.unlink(missing_ok=True)

    server = MLServer(settings)
    task = asyncio.create_task(server.start())
    try:
        model = await server._model_registry.load(batched)
        # In-process batcher attached by the main registry.
        assert AdaptiveBatcher.get_batcher(model) is not None

        requests = [
            inference_request.copy(
                deep=True,
                update={
                    "id": generate_uuid(),
                    "inputs": [
                        inference_request.inputs[0].copy(
                            update={"shape": [1, 3], "data": values}
                        )
                    ],
                },
            )
            for values in ([[1, 2, 3]], [[10, 20, 30]])
        ]
        responses = await asyncio.gather(
            *[model.predict(request) for request in requests]
        )
        assert [list(response.outputs[0].data) for response in responses] == [
            [6],
            [60],
        ]
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        assert [(r["model"], r["rows"]) for r in records] == [("inproc-batched", 2)]
    finally:
        await server.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_load_batching_idempotent_per_instance(sum_model_settings):
    """Wrapping the same model instance twice must not double-batch:
    a superseded origin's hook chain re-runs on the wrapper an apply
    already batched (the stale-publication swap). Without the guard,
    the second wrap nests the predict decorators and orphans the
    first batcher."""
    from mlserver.context import model_context

    with model_context(sum_model_settings):
        model = SumModel(sum_model_settings)
        sum_model_settings.max_batch_size = 2
        sum_model_settings.max_batch_time = 0.1

        await load_batching(model)
        first = AdaptiveBatcher.get_batcher(model)
        assert first is not None

        await load_batching(model)
        assert AdaptiveBatcher.get_batcher(model) is first
