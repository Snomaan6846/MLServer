import pytest

from collections.abc import AsyncGenerator

from mlserver.errors import MLServerError
from mlserver.handlers.custom import get_custom_handlers
from mlserver.types import InferenceRequest, MetadataModelResponse
from mlserver.model import MLModel
from mlserver.settings import ModelSettings
from mlserver.parallel.pool import InferencePool

from ..fixtures import ErrorModel


@pytest.fixture
async def sum_model(
    inference_pool: InferencePool, sum_model: MLModel
) -> AsyncGenerator[MLModel, None]:
    parallel_model = await inference_pool.load_model(sum_model)

    yield parallel_model

    await inference_pool.unload_model(sum_model)


async def test_predict(
    sum_model: MLModel,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    inference_response = await sum_model.predict(inference_request)

    assert inference_response.id == inference_request.id
    assert inference_response.model_name == sum_model_settings.name
    assert len(inference_response.outputs) == 1


async def test_predict_error(
    error_model: MLModel,
    inference_request: InferenceRequest,
):
    with pytest.raises(MLServerError) as excinfo:
        await error_model.predict(inference_request)

    expected_msg = f"mlserver.errors.MLServerError: {ErrorModel.error_message}"
    assert str(excinfo.value) == expected_msg


async def test_metadata(
    sum_model: MLModel,
    sum_model_settings: ModelSettings,
):
    metadata = await sum_model.metadata()

    assert isinstance(metadata, MetadataModelResponse)
    assert metadata.name == sum_model_settings.name


async def test_metadata_cached(
    sum_model: MLModel, sum_model_settings: ModelSettings, mocker
):
    expected_metadata = MetadataModelResponse(name="foo", platform="bar")

    async def _send(*args, **kwargs) -> MetadataModelResponse:
        return expected_metadata

    send_stub = mocker.stub("_send")
    send_stub.side_effect = _send
    sum_model._send = send_stub  # type: ignore[attr-defined]

    metadata_1 = await sum_model.metadata()
    metadata_2 = await sum_model.metadata()

    assert metadata_1 == expected_metadata
    assert metadata_2 == expected_metadata
    send_stub.assert_called_once()


async def test_custom_handlers(sum_model: MLModel):
    handlers = get_custom_handlers(sum_model)
    assert len(handlers) == 2

    response = await sum_model.my_payload([1, 2, 3])  # type: ignore[attr-defined]
    assert response == 6


@pytest.fixture
async def pooled_stream_model(
    inference_pool: InferencePool,
) -> AsyncGenerator[MLModel, None]:
    from ..fixtures import TextStreamModel

    settings = ModelSettings(name="pooled-stream-model", implementation=TextStreamModel)
    model = TextStreamModel(settings)
    parallel_model = await inference_pool.load_model(model)

    yield parallel_model

    await inference_pool.unload_model(model)


async def test_predict_stream_streaming_only_model_fails_fast(
    pooled_stream_model: MLModel,
    inference_request: InferenceRequest,
):
    """A pooled model that only implements predict_stream must fail
    BEFORE any payload is consumed: the inherited single-shot default
    would dispatch predict to the worker, whose mid-stream
    NotImplementedError reaches the client as a truncated (empty 200)
    SSE response instead of an error."""
    consumed = []

    async def payloads():
        consumed.append(inference_request)
        yield inference_request

    with pytest.raises(MLServerError) as excinfo:
        async for _ in pooled_stream_model.predict_stream(payloads()):
            pass

    # The rejection is immediate and actionable, not a worker round
    # trip that dies mid-stream.
    assert not consumed
    assert "parallel inference pool cannot serve" in str(excinfo.value)
    assert "parallel_workers = 0" in str(excinfo.value)


async def test_predict_stream_predict_model_streams_single_shot(
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    """A pooled model that implements predict keeps the base class'
    single-shot streaming fallback: one response per stream, dispatched
    through the regular pool path."""

    async def payloads():
        yield inference_request

    responses = [response async for response in sum_model.predict_stream(payloads())]

    assert len(responses) == 1
    assert responses[0].id == inference_request.id


async def test_predict_stream_env_model_translates_worker_error(
    inference_request: InferenceRequest,
):
    """A model whose real class is only known inside the worker (custom
    environments) cannot be pre-checked in the parent: the worker's
    unimplemented-predict error is translated into the same actionable
    rejection. The branch is exercised with a predict-defining wrapper
    whose dispatch fails exactly as an environment model's would."""
    from mlserver.parallel.errors import WorkerError
    from mlserver.parallel.model import ParallelModel

    class FailingDispatcher:
        async def dispatch_request(self, message):
            raise WorkerError(NotImplementedError("predict() method not implemented"))

    from ..fixtures import SumModel

    parallel = ParallelModel(
        SumModel(ModelSettings(name="env-stream-model", implementation=SumModel)),
        FailingDispatcher(),  # type: ignore[arg-type]
    )

    async def payloads():
        yield inference_request

    with pytest.raises(MLServerError) as excinfo:
        async for _ in parallel.predict_stream(payloads()):
            pass

    assert "parallel inference pool cannot serve" in str(excinfo.value)
