import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from types import MethodType
from unittest.mock import MagicMock

import pytest

from mlserver.model import MLModel
from mlserver.parallel.model import ParallelModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.batching.hooks import load_batching, unload_batching
from mlserver.batching.adaptive import AdaptiveBatcher


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
    """load_batching is installed on the main-process ParallelModel."""
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4

    parallel_model = ParallelModel(sum_model, MagicMock())
    await load_batching(parallel_model)

    assert AdaptiveBatcher.get_batcher(parallel_model) is not None


async def test_parallel_registry_batches_per_model_and_maps_responses(
    mlserver, sum_model_settings, inference_request: InferenceRequest, mocker
):
    """The server registry batches requests before dispatching them to workers."""
    settings_by_model = []
    for name in ("batched-model-a", "batched-model-b"):
        settings = deepcopy(sum_model_settings)
        settings.name = name
        settings.max_batch_size = 2
        settings.max_batch_time = 0.2
        settings_by_model.append(settings)

    models = []

    try:
        for settings in settings_by_model:
            models.append(await mlserver._model_registry.load(settings))

        batchers = [AdaptiveBatcher.get_batcher(model) for model in models]
        assert all(batchers)
        dispatch_spies = [
            mocker.patch.object(batcher, "_predict_fn", wraps=batcher._predict_fn)
            for batcher in batchers
            if batcher is not None
        ]

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
        for dispatch_spy in dispatch_spies:
            dispatch_spy.assert_awaited_once()
            merged_request = dispatch_spy.await_args.args[0]
            assert merged_request.inputs[0].shape == [3, 3]
    finally:
        for model in models:
            await mlserver._model_registry.unload(model.name)
