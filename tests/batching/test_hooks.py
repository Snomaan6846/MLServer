import pytest

from unittest.mock import MagicMock

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


async def test_load_batching_noop_for_parallel_model(sum_model: MLModel):
    """load_batching must be a no-op for ParallelModel — batching is
    installed on workers, not the main process."""
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4

    parallel_model = ParallelModel(sum_model, MagicMock())
    original_predict = parallel_model.predict

    await load_batching(parallel_model)

    assert parallel_model.predict == original_predict
    assert AdaptiveBatcher.get_batcher(parallel_model) is None
