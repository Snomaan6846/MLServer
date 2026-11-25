import asyncio
import contextlib
from typing import Any, cast

import pytest
from pytest_lazyfixture import lazy_fixture

from httpx import AsyncClient
from httpx_sse import aconnect_sse

from mlserver import __version__
from mlserver.batching.adaptive import AdaptiveBatcher
from mlserver.handlers import DataPlane
from mlserver.settings import ModelSettings, Settings
from mlserver.model import MLModel
from mlserver.registry import MultiModelRegistry
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    MetadataServerResponse,
    MetadataModelResponse,
    Parameters,
    RequestInput,
    TensorData,
    RepositoryIndexRequest,
)
from mlserver.cloudevents import (
    CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT,
    CLOUDEVENTS_HEADER_SPECVERSION,
)

from ..fixtures import EchoModel


async def test_live(rest_client: AsyncClient):
    endpoint = "/v2/health/live"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_ready(rest_client: AsyncClient):
    endpoint = "/v2/health/ready"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_model_ready(rest_client: AsyncClient, sum_model: MLModel):
    endpoint = f"/v2/models/{sum_model.name}/versions/{sum_model.version}/ready"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_metadata(rest_client: AsyncClient):
    endpoint = "/v2"
    response = await rest_client.get(endpoint)

    metadata = MetadataServerResponse.model_validate(response.json())

    assert metadata.name == "mlserver"
    assert metadata.version == __version__
    assert metadata.extensions == ["model_repository", "runtime_security"]


async def test_openapi(rest_client: AsyncClient):
    endpoint = "/v2/docs"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


async def test_docs(rest_client: AsyncClient):
    endpoint = "/v2/docs/dataplane.json"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "openapi" in response.json()


async def test_model_metadata(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = f"v2/models/{sum_model_settings.name}"
    response = await rest_client.get(endpoint)

    metadata = MetadataModelResponse.model_validate(response.json())

    assert metadata.name == sum_model_settings.name
    assert metadata.platform == sum_model_settings.platform
    assert metadata.versions == sum_model_settings.versions
    assert metadata.inputs == sum_model_settings.inputs


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_model_openapi(
    rest_client: AsyncClient, model_name: str, model_version: str | None
):
    endpoint = f"/v2/models/{model_name}/docs/dataplane.json"
    if model_version is not None:
        endpoint = (
            f"/v2/models/{model_name}/versions/{model_version}/docs/dataplane.json"
        )
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "openapi" in response.json()


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_model_docs(
    rest_client: AsyncClient, model_name: str, model_version: str | None
):
    endpoint = f"/v2/models/{model_name}/docs"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/docs"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_infer(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    model_name: str,
    model_version: str | None,
):
    endpoint = f"/v2/models/{model_name}/infer"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/infer"

    response = await rest_client.post(endpoint, json=inference_request.model_dump())
    assert response.status_code == 200

    prediction = InferenceResponse.model_validate(response.json())
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0].data == TensorData(root=[6])


async def test_infer_single_adaptive_batch_preserves_output_parameters(
    rest_client: AsyncClient, model_registry: MultiModelRegistry
):
    model_settings = ModelSettings(
        name="echo-model",
        implementation=EchoModel,
        parallel_workers=0,
        max_batch_size=32,
        max_batch_time=0.01,
    )
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="docs",
                datatype="INT32",
                shape=[2],
                data=[10, 11],
                parameters=Parameters(id="123", tags=["123", "456"], count=123),
            )
        ]
    )

    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model(model_settings.name)
        assert AdaptiveBatcher.get_batcher(model) is not None

        response = await rest_client.post(
            "/v2/models/echo-model/infer", json=request.model_dump()
        )

        assert response.status_code == 200
        prediction = InferenceResponse.model_validate(response.json())
        output = prediction.outputs[0]
        assert output.data == TensorData(root=[10, 11])
        assert output.parameters is not None
        assert output.parameters.model_dump(exclude_none=True) == {
            "id": "123",
            "tags": ["123", "456"],
            "count": 123,
        }
    finally:
        await model_registry.unload(model_settings.name)


@pytest.mark.parametrize("sum_model", [lazy_fixture("text_model")])
@pytest.mark.parametrize(
    "model_name,model_version", [("text-model", "v1.2.3"), ("text-model", None)]
)
async def test_generate(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    model_name: str,
    model_version: str | None,
):
    endpoint = f"/v2/models/{model_name}/generate"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/generate"

    response = await rest_client.post(endpoint, json=generate_request.model_dump())
    assert response.status_code == 200

    prediction = InferenceResponse.model_validate(response.json())
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0].data == TensorData(
        root=["What is the capital of France?"]
    )


@pytest.mark.parametrize("settings", [lazy_fixture("settings_stream")])
@pytest.mark.parametrize("sum_model", [lazy_fixture("text_stream_model")])
@pytest.mark.parametrize("endpoint", ["generate_stream", "infer_stream"])
async def test_generate_stream(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    text_stream_model: MLModel,
    endpoint: str,
):
    endpoint = f"/v2/models/{text_stream_model.name}/{endpoint}"
    conn = aconnect_sse(
        rest_client, "POST", endpoint, json=generate_request.model_dump()
    )
    ref_text = ["What", " is", " the", " capital", " of", " France?"]

    async with conn as stream:
        i = 0
        async for response in stream.aiter_sse():
            prediction = InferenceResponse.model_validate(response.json())
            assert len(prediction.outputs) == 1
            assert prediction.outputs[0].data == TensorData(root=[ref_text[i]])
            i += 1


async def test_infer_headers(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
):
    endpoint = f"/v2/models/{sum_model_settings.name}/infer"
    response = await rest_client.post(
        endpoint, json=inference_request.model_dump(), headers={"x-foo": "bar"}
    )

    assert response.status_code == 200
    assert "x-foo" in response.headers
    assert response.headers["x-foo"] == "bar"

    assert CLOUDEVENTS_HEADER_SPECVERSION in response.headers
    assert (
        response.headers[CLOUDEVENTS_HEADER_SPECVERSION]
        == CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT
    )

    # Transport headers must NOT leak into the response body: the
    # pre-encoded endpoint serializes the model itself, so
    # extract_headers must clear parameters.headers BEFORE
    # serialization (the ordering the response_model path guaranteed
    # implicitly). The sum model echoes x-* headers into
    # response.parameters, so their absence here pins the ordering.
    body = response.json()
    assert "parameters" not in body or "headers" not in (
        body.get("parameters") or {}
    ), f"transport headers leaked into body: {body.get('parameters')}"


@pytest.mark.parametrize("endpoint", ["infer", "generate"])
async def test_infer_error(
    rest_client: AsyncClient, inference_request: InferenceRequest, endpoint: str
):
    endpoint = f"/v2/models/my-model/versions/v0/{endpoint}"
    response = await rest_client.post(endpoint, json=inference_request.model_dump())

    assert response.status_code == 404
    assert response.json()["error"] == "Model my-model with version v0 not found"


async def test_model_repository_index_excludes_registry_models_by_default(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(
                exclude={"include_registry_models"}
            ),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    assert [model["name"] for model in response.json()] == [sum_model_settings.name]


async def test_model_repository_index_includes_registry_models(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)
    repository_index_request.include_registry_models = True

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    assert {model["name"] for model in response.json()} == {
        sum_model_settings.name,
        registry_only_settings.name,
    }


@pytest.mark.parametrize("ready", [False, True])
async def test_model_repository_index_ready_filters_selected_models(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
    ready: bool,
):
    await model_registry.unload(sum_model_settings.name)

    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)
    repository_index_request.include_registry_models = True
    repository_index_request.ready = ready

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    expected_names = {registry_only_settings.name}
    if not ready:
        expected_names.add(sum_model_settings.name)

    assert {model["name"] for model in response.json()} == expected_names


async def test_model_repository_unload(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = f"/v2/repository/models/{sum_model_settings.name}/unload"
    response = await rest_client.post(endpoint)

    assert response.status_code == 200

    model_metadata = await rest_client.get(f"/v2/models/{sum_model_settings.name}")
    assert model_metadata.status_code == 404


async def test_model_repository_load(
    rest_client: AsyncClient,
    sum_model_settings: ModelSettings,
):
    await rest_client.post(f"/v2/repository/models/{sum_model_settings.name}/unload")

    endpoint = f"/v2/repository/models/{sum_model_settings.name}/load"
    response = await rest_client.post(endpoint)

    assert response.status_code == 200

    model_metadata = await rest_client.get(f"/v2/models/{sum_model_settings.name}")
    assert model_metadata.status_code == 200


async def test_model_repository_load_error(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = "/v2/repository/models/my-model/load"
    response = await rest_client.post(endpoint)

    assert response.status_code == 404
    assert response.json()["error"] == "Model my-model not found"


async def test_infer_invalid_datatype_error(
    rest_client, inference_request_invalid_datatype, datatype_error_message
):
    endpoint = "/v2/models/sum-model/infer"
    response = await rest_client.post(endpoint, json=inference_request_invalid_datatype)

    assert response.status_code == 422

    assert response.json()["detail"][0]["msg"] == datatype_error_message


async def test_runtimes_endpoint_production_mode(rest_client):
    """Test GET /v2/runtimes in PRODUCTION mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "PRODUCTION"
    assert "allowed_model_implementations" in data
    assert isinstance(data["allowed_model_implementations"], list)
    assert len(data["allowed_model_implementations"]) > 0
    # Should include builtin runtimes
    assert "mlserver_sklearn.SKLearnModel" in data["allowed_model_implementations"]


async def test_runtimes_endpoint_development_mode(development_mode, rest_client):
    """Test GET /v2/runtimes in DEVELOPMENT mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "DEVELOPMENT"
    # In DEVELOPMENT mode, allowed_model_implementations field should not be in response
    assert "allowed_model_implementations" not in data


async def test_runtimes_endpoint_schema_validation(rest_client):
    """Test that /v2/runtimes response matches RuntimeSecurityResponse schema."""
    from mlserver.types import RuntimeSecurityResponse

    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    # In PRODUCTION mode, field should be present in JSON response
    assert "allowed_model_implementations" in data

    # Should be able to parse as RuntimeSecurityResponse
    runtime_response = RuntimeSecurityResponse.model_validate(data)
    assert runtime_response.mode == "PRODUCTION"
    assert runtime_response.allowed_model_implementations is not None


async def test_runtimes_endpoint_includes_all_implementations(rest_client):
    """Test that /v2/runtimes includes all allowlisted implementations."""
    import mlserver.settings as mlserver_settings
    from conftest import TEST_ONLY_EXTRA_IMPLEMENTATIONS

    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()
    implementations_set = set(data["allowed_model_implementations"])

    # Should include all builtin runtimes
    for builtin in mlserver_settings.ALLOWED_MODEL_IMPLEMENTATIONS:
        assert builtin in implementations_set

    # Should include test-only implementations
    for test_impl in TEST_ONLY_EXTRA_IMPLEMENTATIONS:
        assert test_impl in implementations_set


async def test_runtimes_endpoint_sorted_list(rest_client):
    """Test that /v2/runtimes returns a sorted list in PRODUCTION mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    implementations = data["allowed_model_implementations"]
    # Verify list is sorted
    sorted_list = sorted(implementations)
    assert implementations == sorted_list


async def test_metadata_advertises_binary_tensor_extension(
    data_plane: DataPlane, rest_client: AsyncClient
):
    """The KServe v2 Binary Tensor Data Extension is discoverable."""
    data_plane._settings.rest_binary_tensor = True
    response = await rest_client.get("/v2")

    metadata = MetadataServerResponse.model_validate(response.json())
    assert "binary_tensor_data" in metadata.extensions


async def test_metadata_without_binary_tensor_omits_extension(
    rest_client: AsyncClient,
):
    """The extension stays absent unless the opt-in is enabled."""
    response = await rest_client.get("/v2")

    metadata = MetadataServerResponse.model_validate(response.json())
    assert "binary_tensor_data" not in metadata.extensions


async def test_infer_validation_error_renders_orjson(rest_client: AsyncClient):
    # The RequestValidationError handler renders through the engine's orjson
    # Response: the echoed input's scientific-notation floats use orjson's
    # single-digit exponent (1e-7), not stdlib json's (1e-07).
    endpoint = "/v2/models/sum-model/infer"
    response = await rest_client.post(endpoint, json={"inputs": {"bad": 1e-7}})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    detail = response.json()["detail"]
    assert detail[0]["type"] == "list_type"
    assert detail[0]["loc"] == ["body", "inputs"]
    assert b"1e-7" in response.content
    assert b"1e-07" not in response.content


async def test_infer_stream_single_shot_fast_path(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model: MLModel,
):
    """Single-shot models take the fast path: one SSE event whose
    payload equals the /infer response — byte-identical body, clean
    HTTP semantics, no streaming machinery."""
    request_body = inference_request.model_dump()
    request_body["id"] = "fast-path-equivalence-test"
    endpoint = f"/v2/models/{sum_model.name}/infer_stream"
    response = await rest_client.post(endpoint, json=request_body)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Exactly one SSE event, framing intact
    body = response.content
    assert body.startswith(b"data: ") and body.endswith(b"\n\n")
    assert body.count(b"data: ") == 1

    # The event payload equals the /infer response for the same request
    infer = await rest_client.post(
        f"/v2/models/{sum_model.name}/infer",
        json=request_body,
    )
    fast = InferenceResponse.model_validate_json(body[len(b"data: ") : -len(b"\n\n")])
    slow = InferenceResponse.model_validate_json(infer.content)
    # Compare the semantic payload (id, model, outputs): response
    # parameters differ by design — /infer moves CloudEvents headers
    # to HTTP headers while the SSE body embeds them in the response
    # parameters. Both correct, just different placement.
    assert fast.id == slow.id
    assert fast.model_name == slow.model_name
    assert len(fast.outputs) == len(slow.outputs)
    for fast_out, slow_out in zip(fast.outputs, slow.outputs):
        assert fast_out.name == slow_out.name
        assert fast_out.shape == slow_out.shape
        assert fast_out.datatype == slow_out.datatype
        assert fast_out.data.root == slow_out.data.root


async def test_infer_stream_single_shot_error_surfaces_cleanly(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    error_model: MLModel,
):
    """A failing single-shot model returns a clean HTTP error from
    /infer_stream — never a truncated 200."""
    endpoint = f"/v2/models/{error_model.name}/infer_stream"
    response = await rest_client.post(endpoint, json=inference_request.model_dump())

    # MLServerError defaults to 400 (mlserver/errors.py); the point is
    # a clean HTTP error with a JSON body — not a 200 SSE stream that
    # truncates after the headers.
    assert response.status_code == 400
    assert not response.headers.get("content-type", "").startswith("text/event-stream")
    # error_model is the ParallelModel wrapper, so read the message
    # from the underlying fixture class rather than the instance.
    from tests.fixtures import ErrorModel

    assert ErrorModel.error_message in response.text


async def test_infer_stream_single_shot_batched_model_fast_path(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
    model_registry: MultiModelRegistry,
):
    """A batched single-shot model through /infer_stream: the class-
    level check reports single-shot (SumModel doesn't override
    predict_stream), so the fast path applies — but the AdaptiveBatcher
    wraps predict at instance level, meaning the response goes through
    the same batched inference the stream fallback would have run.
    The SSE payload must be equivalent to the /infer response.

    This pins the instance-shadowing interplay against future batcher
    refactors: if the batcher's wrapping ever changes what the fast
    path produces vs the stream fallback, this test catches it.
    """

    batched_settings = sum_model_settings.model_copy(deep=True)
    batched_settings.max_batch_size = 4
    batched_settings.max_batch_time = 0.001
    batched_settings.name = "sum-model-batched-test"
    await model_registry.load(batched_settings)

    # Use a fixed request id so both responses carry the same id
    # (auto-generated UUIDs would differ between the two requests).
    request_body = inference_request.model_dump()
    request_body["id"] = "batched-fast-path-test"

    endpoint = f"/v2/models/{batched_settings.name}/infer_stream"
    response = await rest_client.post(endpoint, json=request_body)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.count(b"data: ") == 1

    # The SSE payload equals the /infer response for the same request
    infer = await rest_client.post(
        f"/v2/models/{batched_settings.name}/infer",
        json=request_body,
    )
    assert infer.status_code == 200
    fast = InferenceResponse.model_validate_json(
        response.content[len(b"data: ") : -len(b"\n\n")]
    )
    slow = InferenceResponse.model_validate_json(infer.content)
    # Compare the inference outputs (the semantic payload): response
    # parameters differ because /infer moves CloudEvents headers to
    # HTTP headers while the SSE body embeds them in the response
    # parameters — both correct, just different placement.
    assert fast.id == slow.id
    assert fast.model_name == slow.model_name
    assert len(fast.outputs) == len(slow.outputs)
    for fast_out, slow_out in zip(fast.outputs, slow.outputs):
        assert fast_out.name == slow_out.name
        assert fast_out.shape == slow_out.shape
        assert fast_out.datatype == slow_out.datatype
        assert fast_out.data.root == slow_out.data.root

    await model_registry.unload(batched_settings.name)


async def test_infer_stream_multi_stream_producer_path(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    text_stream_model: MLModel,
):
    """Multi-stream models (predict_stream overridden) take the producer
    path: multiple SSE events, in order, each correctly framed.

    This test is the multi-stream counterpart to the fast-path tests:
    it proves the producer/queue/cancel-settle machinery still works
    correctly when the fast path is present in the code but NOT taken.
    """
    endpoint = f"/v2/models/{text_stream_model.name}/infer_stream"
    conn = aconnect_sse(
        rest_client, "POST", endpoint, json=generate_request.model_dump()
    )
    ref_text = ["What", " is", " the", " capital", " of", " France?"]

    events = []
    async with conn as stream:
        async for response in stream.aiter_sse():
            prediction = InferenceResponse.model_validate(response.json())
            events.append(prediction)

    # ALL events arrived
    assert len(events) == len(
        ref_text
    ), f"expected {len(ref_text)} events, got {len(events)}"
    # In order, correct content
    for i, prediction in enumerate(events):
        assert len(prediction.outputs) == 1
        assert prediction.outputs[0].data == TensorData(root=[ref_text[i]])
    # Same model
    assert all(e.model_name == text_stream_model.name for e in events)


async def test_infer_stream_multi_stream_sse_framing(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    text_stream_model: MLModel,
):
    """The raw SSE body for a multi-stream model has correct framing:
    every event is 'data: <json>\\n\\n', no extra bytes."""
    endpoint = f"/v2/models/{text_stream_model.name}/infer_stream"
    response = await rest_client.post(endpoint, json=generate_request.model_dump())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.content
    # Count events by the 'data: ' prefix
    events = body.split(b"data: ")
    # First element is empty (body starts with 'data: ')
    assert events[0] == b""
    # Every event ends with \n\n
    for event in events[1:]:
        assert event.endswith(
            b"\n\n"
        ), f"event does not end with \\n\\n: {event[-20:]!r}"
    # 6 events for the text stream model
    assert len(events) - 1 == 6


async def test_infer_stream_multi_stream_mid_stream_error(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
    model_registry: MultiModelRegistry,
    development_mode,
):
    """A multi-stream model that errors MID-STREAM (after the first
    event): the late error must end the response ABNORMALLY — the
    exception propagates to the client, never masked as a normal
    end-of-stream (a 200 that quietly truncates after the partial
    output is indistinguishable from success)."""

    from mlserver.errors import MLServerError
    from tests.fixtures import FailingStreamModel

    fail_settings = sum_model_settings.model_copy(deep=True)
    fail_settings.name = "failing-stream-model"
    fail_settings.implementation = FailingStreamModel
    # pw=0 keeps the model unwrapped: the REAL predict_stream runs on
    # the producer path. Through the pool, the single-shot fallback
    # would predict once and complete normally — the mid-stream raise
    # would never fire and the test would pass vacuously.
    fail_settings.parallel_workers = 0
    await model_registry.load(fail_settings)

    endpoint = f"/v2/models/{fail_settings.name}/infer_stream"
    # The headers (HTTP 200) were already sent when the error hit, so
    # the status cannot change and Starlette cannot run the error
    # handler — the only honest signal left is an abnormal
    # completion: the request itself raises, with the model's error
    # in the chain (a normal completion returns a response instead).
    with pytest.raises(Exception) as excinfo:
        await rest_client.post(endpoint, json=inference_request.model_dump())
    cause = excinfo.value.__cause__
    assert isinstance(cause, MLServerError), (
        f"the late error must propagate (got {excinfo.value!r} " f"caused by {cause!r})"
    )
    assert "mid-stream failure" in str(cause)

    await model_registry.unload(fail_settings.name)


async def test_infer_stream_multi_stream_error_before_first_event(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
    model_registry: MultiModelRegistry,
    development_mode,
):
    """A multi-stream model that errors BEFORE the first event: the
    error surfaces as a clean HTTP error — the SSE headers were never
    sent, so the client gets a JSON error body, not an empty 200 SSE
    stream it cannot distinguish from a legitimate one."""

    from tests.fixtures import FailingStreamModel

    fail_settings = sum_model_settings.model_copy(deep=True)
    fail_settings.name = "failing-stream-model"
    fail_settings.implementation = FailingStreamModel
    fail_settings.parallel_workers = 0
    await model_registry.load(fail_settings)

    model = await model_registry.get_model(fail_settings.name)
    setattr(model, "fail_before_first", True)

    endpoint = f"/v2/models/{fail_settings.name}/infer_stream"
    response = await rest_client.post(endpoint, json=inference_request.model_dump())

    # MLServerError defaults to 400 (mlserver/errors.py): a clean
    # HTTP error with a JSON body — the stream never started.
    assert response.status_code == 400
    assert not response.headers.get("content-type", "").startswith("text/event-stream")
    assert "mid-stream failure" in response.text

    await model_registry.unload(fail_settings.name)


async def test_infer_stream_mid_stream_error_settles_model_stream(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
    model_registry: MultiModelRegistry,
    development_mode,
):
    """A late stream error must not orphan the producer: the SSE
    settlement (the consumer's shielded finally) closes the model's
    iterator on the error path — the same once-only cleanup a normal
    completion runs. The stream_closed hook also proves the REAL
    predict_stream ran (the hook only fires inside it)."""

    from mlserver.errors import MLServerError
    from tests.fixtures import FailingStreamModel

    fail_settings = sum_model_settings.model_copy(deep=True)
    fail_settings.name = "failing-stream-model"
    fail_settings.implementation = FailingStreamModel
    fail_settings.parallel_workers = 0
    await model_registry.load(fail_settings)

    model = await model_registry.get_model(fail_settings.name)
    stream_closed = asyncio.Event()
    setattr(model, "stream_closed", stream_closed)

    endpoint = f"/v2/models/{fail_settings.name}/infer_stream"
    with pytest.raises(Exception) as excinfo:
        await rest_client.post(endpoint, json=inference_request.model_dump())
    # The error propagated (abnormal completion — see the mid-stream
    # error test) with the model's error in the chain...
    assert isinstance(excinfo.value.__cause__, MLServerError)

    # ...AND the model's iterator closed: neither an orphaned
    # producer nor a stranded generator.
    await asyncio.wait_for(stream_closed.wait(), timeout=5)
    assert (
        stream_closed.is_set()
    ), "the model iterator must close when a late error ends the stream"

    await model_registry.unload(fail_settings.name)


async def test_infer_stream_single_vs_multi_path_selection(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    generate_request: InferenceRequest,
    sum_model: MLModel,
    text_stream_model: MLModel,
):
    """Single-shot and multi-stream models on the SAME deployment:
    each takes its correct path. Single-shot gets one event (fast
    path), multi-stream gets six (producer path). Proves the routing
    is per-model, not per-deployment."""
    # Single-shot: exactly 1 event (sum_model needs numeric input)
    infer_req = inference_request.model_copy(deep=True)
    infer_req.id = "path-selection-test"
    single_endpoint = f"/v2/models/{sum_model.name}/infer_stream"
    single_response = await rest_client.post(
        single_endpoint, json=infer_req.model_dump()
    )
    assert single_response.status_code == 200
    assert single_response.content.count(b"data: ") == 1

    # Multi-stream: exactly 6 events (text_stream_model needs text input)
    multi_endpoint = f"/v2/models/{text_stream_model.name}/infer_stream"
    multi_response = await rest_client.post(
        multi_endpoint, json=generate_request.model_dump()
    )
    assert multi_response.status_code == 200
    assert multi_response.content.count(b"data: ") == 6


async def test_infer_endpoint_takes_python_mode(rest_client, sum_model):
    """The mode pin the buffer-native unary path rests on: /infer must
    deliver the python-mode dump to the JSON encoder — the tensor data
    slot contains an NDARRAY (which orjson renders straight from the
    buffer). The FastAPI response_model path would deliver mode="json"
    output — a LIST, materialising every element on the serving loop.
    Records what the encoder actually receives through a real request.
    """
    import mlserver.codecs.json as json_codec

    observed = []
    original = json_codec.encode_to_json_bytes

    def recording_encode(v):
        try:
            observed.append(type(v["outputs"][0]["data"]).__name__)
        except Exception:
            observed.append("other")
        return original(v)

    json_codec.encode_to_json_bytes = recording_encode
    try:
        payload = {
            "inputs": [
                {
                    "name": "input-0",
                    "shape": [1, 2],
                    "datatype": "FP32",
                    "data": [1.5, 2.5],
                }
            ]
        }
        response = await rest_client.post(
            f"/v2/models/{sum_model.name}/infer", json=payload
        )
        assert response.status_code == 200
    finally:
        json_codec.encode_to_json_bytes = original

    # The ndarray reached the encoder: the python-mode path ran.
    assert "ndarray" in observed, observed
    assert "list" not in observed, observed


async def test_infer_response_metadata_json_scalars(
    rest_client: AsyncClient, model_registry: MultiModelRegistry
):
    """Response parameters with Decimal / timedelta values render as
    JSON scalars (the response-model path's JSON-mode conversion):
    Decimal -> "3.14", timedelta -> ISO duration. The pre-encoded
    endpoint must keep that conversion for metadata while rendering
    tensor data natively."""
    from tests.fixtures import MetadataEchoModel

    model_settings = ModelSettings(
        name="metadata-model",
        implementation=MetadataEchoModel,
        parallel_workers=0,
    )
    request = InferenceRequest(
        inputs=[
            RequestInput(name="input-0", datatype="INT32", shape=[1, 3], data=[1, 2, 3])
        ]
    )

    await model_registry.load(model_settings)
    try:
        response = await rest_client.post(
            "/v2/models/metadata-model/infer", json=request.model_dump()
        )
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["parameters"] == {
            "decimal": "3.14",
            "delta": "PT2S",
        }, "the base's JSON-mode scalar conversion"
        assert body["outputs"][0]["parameters"] == {"decimal": "2.72"}
        # Tensor data still renders natively (the echoed input).
        assert body["outputs"][0]["data"] == [1, 2, 3]
    finally:
        await model_registry.unload(model_settings.name)


async def test_infer_stream_disconnect_before_first_event_settles_model():
    """A client that disconnects during the first-event prefetch must
    not leave the model stream running: until the StreamingResponse
    starts, nothing else listens on the ASGI receive channel, so the
    endpoint itself races the disconnect, then cancels and settles
    the producer (the base closed the model generator via
    Starlette's streaming disconnect watch)."""
    from mlserver.rest.endpoints import Endpoints

    # A streaming model whose first event is slow (the prefetch
    # window), tracking generator closure.
    closed = asyncio.Event()
    started = asyncio.Event()

    async def slow_stream(payloads, name, version=None):
        started.set()
        try:
            await asyncio.sleep(5)
            yield InferenceResponse(model_name=name, outputs=[])
        finally:
            closed.set()

    class SlowModel:
        name = "slow-model"

        def stream_is_single_shot(self) -> bool:
            return False

    slow_model = SlowModel()

    class SlowStreamDataPlane:
        async def resolve_model(self, name, version=None):
            return slow_model

        def infer_stream(self, payloads, name, version=None, model=None):
            return slow_stream(payloads, name, version)

    # A request whose client disconnects immediately.
    class DisconnectingRequest:
        headers = {"content-type": "application/json"}

        async def receive(self):
            return {"type": "http.disconnect"}

    endpoints = Endpoints.__new__(Endpoints)
    endpoints._data_plane = SlowStreamDataPlane()

    payload = InferenceRequest(
        inputs=[RequestInput(name="input-0", datatype="INT32", shape=[1], data=[1])]
    )
    response = await asyncio.wait_for(
        endpoints.infer_stream(DisconnectingRequest(), payload, "slow-model", None),
        timeout=2,
    )
    # The endpoint completed (no hang on the first event) and the
    # model generator was closed, not left running.
    assert response.status_code == 499
    assert closed.is_set(), "the producer settled the model stream"


async def test_infer_stream_reload_uses_the_resolved_instance() -> None:
    """The streaming-mode decision and the execution use the SAME
    model instance: a reload can swap the registration after the
    single-shot cache was populated from the old model — the stale
    answer must not route the NEW model's request through the unary
    predict() fast path."""
    from mlserver.rest.endpoints import Endpoints

    used: list = []

    class OldModel:
        name = "reload-model"

        def stream_is_single_shot(self) -> bool:
            return True  # the stale cached answer's source

    class NewModel:
        name = "reload-model"

        def stream_is_single_shot(self) -> bool:
            return False  # the replacement overrides predict_stream

        async def predict(self, payload):
            used.append(("unary-predict", self))
            raise AssertionError("the wrong inference method ran")

        async def predict_stream(self, payloads):
            used.append(("predict_stream", self))
            yield InferenceResponse(model_name=self.name, outputs=[])
            yield InferenceResponse(model_name=self.name, outputs=[])

    new_model = NewModel()

    class ReloadedDataPlane:
        async def resolve_model(self, name, version=None):
            return new_model

        async def infer(self, payload, name, version=None, model=None):
            used.append(("infer", model))
            raise AssertionError("the unary fast path ran")

        def infer_stream(self, payloads, name, version=None, model=None):
            assert model is new_model, "execution uses the resolved instance"
            return model.predict_stream(payloads)

    endpoints = Endpoints.__new__(Endpoints)
    setattr(endpoints, "_data_plane", ReloadedDataPlane())

    payload = InferenceRequest(
        inputs=[RequestInput(name="input-0", datatype="INT32", shape=[1], data=[1])]
    )

    class StubRequest:
        headers: dict = {}

        async def receive(self):
            # A connected client: no disconnect arrives during the
            # prefetch.
            await asyncio.sleep(3600)

    response = await endpoints.infer_stream(
        cast(Any, StubRequest()), payload, "reload-model", None
    )
    # The producer path ran with the resolved instance — two events,
    # not the stale-cache single-shot branch.
    assert used == [("predict_stream", new_model)]
    assert response.status_code == 200
    # Drive the body so the producer settles (no pending tasks at
    # teardown).
    async for _ in cast(Any, response).body_iterator:
        pass


async def test_infer_stream_single_shot_disconnect_settles_inference() -> None:
    """A single-shot model's /infer_stream with a client that
    disconnects mid-inference: the inference task must be cancelled
    and settled, not left running (the base's StreamingResponse
    watched for http.disconnect; the pre-encoded Response has no
    such watch without the race)."""
    from mlserver.rest.endpoints import Endpoints

    infer_started = asyncio.Event()
    infer_settled = asyncio.Event()

    async def slow_infer(payload, name, version=None, model=None):
        infer_started.set()
        try:
            await asyncio.sleep(5)
            return InferenceResponse(model_name=name, outputs=[])
        finally:
            infer_settled.set()

    class SingleShotDataPlane:
        async def resolve_model(self, name, version=None):
            class M:
                def stream_is_single_shot(self):
                    return True

            return M()

        async def infer(self, payload, name, version=None, model=None):
            return await slow_infer(payload, name, version, model)

    class DisconnectingRequest:
        headers: dict = {}

        async def receive(self):
            return {"type": "http.disconnect"}

    endpoints = Endpoints.__new__(Endpoints)
    setattr(endpoints, "_data_plane", SingleShotDataPlane())

    payload = InferenceRequest(
        inputs=[RequestInput(name="input-0", datatype="INT32", shape=[1], data=[1])]
    )
    response = await asyncio.wait_for(
        endpoints.infer_stream(
            cast(Any, DisconnectingRequest()), payload, "single-shot-model", None
        ),
        timeout=2,
    )
    assert response.status_code == 499
    assert infer_started.is_set(), "the inference ran"
    assert infer_settled.is_set(), "the inference was settled, not left running"


async def test_infer_stream_backpressure_disconnect_settles_producer() -> None:
    """A client that stops reading (SSE backpressure: the queue is
    full and the producer is blocked on put) and then goes away must
    not strand the producer: on disconnect the ASGI send raises and
    Starlette's async-for exits WITHOUT closing the body iterator,
    so the generator's finally may never run. The producer must be
    settled from the request task's own completion, and the model's
    generator closed with it."""
    from mlserver.rest.endpoints import Endpoints

    closed = asyncio.Event()
    yielded: list[int] = []

    async def stream(payloads, name, version=None):
        try:
            for i in range(6):
                yielded.append(i)
                yield InferenceResponse(model_name=name, outputs=[])
        finally:
            closed.set()

    class BackpressureModel:
        name = "bp-model"

        def stream_is_single_shot(self) -> bool:
            return False

    bp_model = BackpressureModel()

    class BackpressureDataPlane:
        def __init__(self) -> None:
            # Keep the model generator referenced: the settlement
            # under test must not lean on GC finalisation.
            self.captured: list = []

        async def resolve_model(self, name, version=None):
            return bp_model

        def infer_stream(self, payloads, name, version=None, model=None):
            gen = stream(payloads, name, version)
            self.captured.append(gen)
            return gen

    class ConnectedRequest:
        headers: dict = {}

        async def receive(self):
            # A connected client during the prefetch: no disconnect
            # arrives there — this test exercises the mid-stream path.
            await asyncio.sleep(3600)

    data_plane = BackpressureDataPlane()
    endpoints = Endpoints.__new__(Endpoints)
    setattr(endpoints, "_data_plane", data_plane)

    payload = InferenceRequest(
        inputs=[RequestInput(name="input-0", datatype="INT32", shape=[1], data=[1])]
    )

    # Hold references to the response and its body iterator so the
    # stranded generator cannot be settled by GC finalisation
    # instead of the fix under test.
    holder: dict = {}
    read_first = asyncio.Event()

    async def request_flow() -> None:
        response = await endpoints.infer_stream(
            cast(Any, ConnectedRequest()), payload, "bp-model", None
        )
        holder["response"] = response
        agen = cast(Any, response).body_iterator
        holder["agen"] = agen
        holder["chunk"] = await asyncio.wait_for(agen.__anext__(), timeout=2)
        read_first.set()
        # The client stalls: it never reads another chunk, so the
        # queue fills and the producer blocks on put.
        await asyncio.Event().wait()

    request_task = asyncio.create_task(request_flow())
    await asyncio.wait_for(read_first.wait(), timeout=2)
    # Let the pipeline reach the blocked state: one event consumed
    # by the prefetch, one in the queue, one in the producer's hand.
    await asyncio.sleep(0.05)
    assert len(yielded) == 3, "the producer is blocked on a full queue"

    # The request task dies mid-stream — exactly what happens when
    # the ASGI send raises on a disconnected client (or the response
    # task is cancelled on older ASGI servers).
    request_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await request_task

    await asyncio.wait_for(closed.wait(), timeout=1)
    assert closed.is_set(), "the producer was settled, not left blocked"
    assert holder["chunk"]


async def test_infer_response_datetime_data_wire_format(
    rest_client: AsyncClient, model_registry: MultiModelRegistry
):
    """Datetime objects in BYTES output data render per-endpoint
    exactly as the base did: /infer through the response-model
    serialization (UTC -> Z, offsets unchanged), /infer_stream
    through ServerSentEvent's python-mode dump + orjson (UTC ->
    +00:00 — the base's own SSE behaviour, unchanged by this PR)."""
    from tests.fixtures import DatetimeEchoModel

    model_settings = ModelSettings(
        name="datetime-model",
        implementation=DatetimeEchoModel,
        parallel_workers=0,
    )
    request = InferenceRequest(
        inputs=[
            RequestInput(name="input-0", datatype="INT32", shape=[1, 3], data=[1, 2, 3])
        ]
    )

    await model_registry.load(model_settings)
    try:
        response = await rest_client.post(
            "/v2/models/datetime-model/infer", json=request.model_dump()
        )
        assert response.status_code == 200, response.text

        data = response.json()["outputs"][0]["data"]
        assert data[0] == "2023-01-01T12:00:00Z", "UTC renders Z, not +00:00"
        assert data[1] == "2023-01-01T12:00:00+02:00"

        stream = await rest_client.post(
            "/v2/models/datetime-model/infer_stream", json=request.model_dump()
        )
        assert stream.status_code == 200, stream.text
        assert "2023-01-01T12:00:00+00:00" in stream.text, stream.text
        assert "2023-01-01T12:00:00Z" not in stream.text, stream.text
    finally:
        await model_registry.unload(model_settings.name)


async def test_infer_response_wire_identical_across_serving_paths(
    rest_client: AsyncClient, model_registry: MultiModelRegistry
):
    """The SAME response renders byte-identically whether the
    frontend serializes it (parallel_workers=0) or the worker does
    (the shared pool): one serializer, one conversion, no variations
    between serving paths. The per-request id is the only allowed
    difference."""
    import orjson

    from tests.fixtures import ExoticEchoModel

    request = InferenceRequest(
        inputs=[
            RequestInput(name="input-0", datatype="INT32", shape=[1, 3], data=[1, 2, 3])
        ]
    )

    bodies = []
    for workers in (0, 2):
        model_settings = ModelSettings(
            name="wire-parity-model",
            implementation=ExoticEchoModel,
            parallel_workers=workers,
        )
        await model_registry.load(model_settings)
        try:
            response = await rest_client.post(
                "/v2/models/wire-parity-model/infer", json=request.model_dump()
            )
            assert response.status_code == 200, response.text
            body = orjson.loads(response.content)
            body.pop("id", None)
            bodies.append(orjson.dumps(body))
        finally:
            await model_registry.unload(model_settings.name)

    assert bodies[0] == bodies[1], "frontend and worker wire bytes diverged"
    data = orjson.loads(bodies[0])["outputs"][0]["data"]
    assert data[0] == "2023-01-01T12:00:00Z"
    assert data[1] == "2023-01-01T12:00:00+02:00"
    assert data[2] == "3.14"
    assert data[3] == "PT2S"
    assert data[4] == 0.10000000149011612, "the widened float32 form"


async def test_infer_stream_asgi_response_owns_cleanup(
    settings: Settings, model_registry: MultiModelRegistry
):
    """The streaming response OWNS its producer's settlement: the
    ASGI call exits on send failure (the disconnected client raises
    through Starlette's sender) while the CALLER task stays alive —
    neither the body generator's finally (the iterator is stranded
    at a yield) nor the request task's completion callback (the task
    lives on) can be the cleanup owner. The response's own __call__
    cleanup settles the producer, and the settlement chain closes
    the model's iterator."""
    from starlette.requests import ClientDisconnect

    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="tracked-stream-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("tracked-stream-model")
        stream_closed = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        # Hold the model's generator referenced: the cleanup under
        # test must not lean on GC finalisation (an unreferenced
        # suspended generator gets closed by the event loop's
        # async-gen finaliser, which would rescue a broken chain).
        captured: list = []
        original_predict_stream = model.predict_stream

        def capturing_predict_stream(payloads):
            gen = original_predict_stream(payloads)
            captured.append(gen)
            return gen

        setattr(model, "predict_stream", capturing_predict_stream)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        response = await endpoints.infer_stream(
            cast(Any, ConnectedRequest()), payload, "tracked-stream-model", None
        )

        # Drive the real ASGI call: the send blocks after the first
        # body chunk (the queue fills, the producer blocks on put),
        # then fails the way a disconnected client does.
        sent: list[str] = []
        release_send = asyncio.Event()

        async def send(message):
            sent.append(message["type"])
            if message["type"] == "http.response.body":
                await release_send.wait()
                raise OSError("client disconnected")

        async def receive():
            await asyncio.sleep(3600)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        }

        asgi_call = asyncio.create_task(response(scope, receive, send))
        await asyncio.sleep(0.05)  # the send blocks; the queue fills
        assert "http.response.body" in sent

        release_send.set()
        # The caller task stays ALIVE throughout (the assertion runs
        # in it): only the response's own cleanup can settle.
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(asgi_call, timeout=5)

        await asyncio.wait_for(stream_closed.wait(), timeout=5)
        assert (
            stream_closed.is_set()
        ), "the model iterator must close when the ASGI response exits"
    finally:
        await model_registry.unload(model_settings.name)


async def test_infer_stream_asgi23_disconnect_completes_model_cleanup(
    settings: Settings, model_registry: MultiModelRegistry
):
    """On the ASGI 2.3 task-group path, a disconnect after the first
    event cancels the body iterator while it waits for the next one:
    the iterator's cleanup settles the producer (the model enters
    its ASYNC finally), and the response's own cleanup must AWAIT the
    same settlement — a second cancellation would interrupt the
    model's resource release mid-flight. Cleanup must be complete
    when the response returns."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="tracked23-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("tracked23-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)
        captured: list = []
        original_predict_stream = model.predict_stream

        def capturing_predict_stream(payloads):
            gen = original_predict_stream(payloads)
            captured.append(gen)
            return gen

        setattr(model, "predict_stream", capturing_predict_stream)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        response = await endpoints.infer_stream(
            cast(Any, ConnectedRequest()), payload, "tracked23-model", None
        )

        first_body_sent = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                first_body_sent.set()
            return None

        async def receive():
            # The client disconnects after the first event, while
            # the response waits for the next one.
            await first_body_sent.wait()
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        }

        asgi_call = asyncio.create_task(response(scope, receive, send))
        # The model's async cleanup starts (the producer settles, the
        # aclose chain reaches the model's finally) and waits on the
        # release gate.
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)

        # Release the gate shortly. With the once-only settlement,
        # the response's cleanup AWAILS the same settlement (no
        # second cancellation): the model's async release runs to
        # completion and the response returns with the cleanup done.
        # A second-cancel bug interrupts the in-flight release
        # instead — the cleanup never finishes.
        async def release_later() -> None:
            await asyncio.sleep(0.05)
            slow_release.set()

        releaser = asyncio.create_task(release_later())
        await asyncio.wait_for(asgi_call, timeout=5)
        await releaser

        assert stream_closed.is_set(), (
            "the model's async cleanup must COMPLETE when the "
            "response returns, not be interrupted by a second "
            "cancellation"
        )
    finally:
        await model_registry.unload(model_settings.name)


async def test_prefetch_disconnect_cancellation_preserves_model_cleanup(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A disconnect during the prefetch settles the producer (the
    model enters its async cleanup); a cancellation of the REQUEST
    task while that cleanup is in flight must not propagate into the
    producer a second time (the prefetch helper settles through the
    shared once-only settlement, and gathers its watcher tasks
    separately)."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="prefetch-cancel-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("prefetch-cancel-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class DisconnectingRequest:
            headers: dict = {}

            async def receive(self):
                return {"type": "http.disconnect"}

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        request_task = asyncio.create_task(
            endpoints.infer_stream(
                cast(Any, DisconnectingRequest()),
                payload,
                "prefetch-cancel-model",
                None,
            )
        )
        # The disconnect fires during the prefetch; the producer
        # settles and the model's async cleanup starts on the gate.
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)

        # Cancel the REQUEST task while the model's cleanup is in
        # flight: the settlement is shielded, so the cancellation
        # must not reach the producer a second time — and the
        # CANCELLATION WINS over the disconnect's 499 (the request
        # task must end cancelled, not return a response that goes
        # nowhere).
        request_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await request_task
        assert (
            request_task.cancelled()
        ), "a cancelled request must end cancelled, not return 499"

        slow_release.set()
        await asyncio.wait_for(stream_closed.wait(), timeout=5)
        assert stream_closed.is_set(), (
            "the model's async cleanup must survive the request " "task's cancellation"
        )
    finally:
        await model_registry.unload(model_settings.name)


async def test_repeated_sse_streams_do_not_accumulate_callbacks(
    settings: Settings, model_registry: MultiModelRegistry
):
    """Completed streams must not leave settlement backstop callbacks
    on a REUSED (still-alive) ASGI caller task: each callback retains
    its producer and settlement state until the caller exits. The
    settlement removes the backstop when it completes."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="repeat-stream-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )

        async def send(message):
            return None

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        }

        caller = asyncio.current_task()
        assert caller is not None
        baseline = len(caller._callbacks)

        for _ in range(5):
            response = await endpoints.infer_stream(
                cast(Any, ConnectedRequest()), payload, "repeat-stream-model", None
            )
            await response(scope, ConnectedRequest().receive, send)

        assert len(caller._callbacks) == baseline, (
            "completed streams must remove their settlement backstop "
            "callbacks from a reused caller task"
        )
    finally:
        await model_registry.unload(model_settings.name)


async def test_prefetch_wait_cancellation_settles_producer(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A request cancelled while WAITING for the first event (before
    any disconnect): the settlement must START before the watcher
    cleanup so an interruption during that cleanup cannot skip it —
    the producer settles and the request ends cancelled."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="wait-cancel-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("wait-cancel-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)
        # A gated predict holds the first event back: the request is
        # cancelled while the prefetch WAIT is pending (before any
        # disconnect, before the first event).
        predict_gate = asyncio.Event()
        original_predict = model.predict

        async def slow_predict(payload):
            await predict_gate.wait()
            return await original_predict(payload)

        setattr(model, "predict", slow_predict)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        request_task = asyncio.create_task(
            endpoints.infer_stream(
                cast(Any, ConnectedRequest()), payload, "wait-cancel-model", None
            )
        )
        # The producer is mid-inference (the prefetch wait, the
        # first event gated); cancel the request: the wait is
        # interrupted, the settlement must start regardless of what
        # the cleanup awaits.
        await asyncio.sleep(0.05)
        request_task.cancel()
        predict_gate.set()  # moot for the cancelled call; hygiene

        # The settlement runs while the request task waits for it
        # (shielded): the producer was cancelled and the model's
        # async cleanup starts on the gate — observed and released
        # BEFORE awaiting the task, whose unwind completes with the
        # settlement.
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
        slow_release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(request_task, timeout=5)
        assert request_task.cancelled()

        await asyncio.wait_for(stream_closed.wait(), timeout=5)
        assert stream_closed.is_set()
    finally:
        await model_registry.unload(model_settings.name)


async def test_prefetch_handoff_joins_disconnect_watcher(
    settings: Settings, model_registry: MultiModelRegistry
):
    """The first-event success path must JOIN the cancelled
    disconnect watcher before handing the receive channel to the
    StreamingResponse: a pending first read and Starlette's own read
    would overlap (an ASGI server rejects a concurrent read of the
    channel). A receive wrapper that takes an async step to release
    detects the overlap."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="handoff-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class TrackedReceive:
            """A receive channel that tracks concurrent reads and
            takes an async step to release each one."""

            def __init__(self) -> None:
                self.in_flight = 0
                self.max_in_flight = 0

            async def __call__(self):
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                try:
                    await asyncio.sleep(0.01)
                    return {"type": "http.disconnect"}
                finally:
                    self.in_flight -= 1

        class ConnectedRequest:
            headers: dict = {}

            def __init__(self) -> None:
                self.receive_channel = TrackedReceive()

            async def receive(self):
                return await self.receive_channel()

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        request = ConnectedRequest()
        response = await endpoints.infer_stream(
            cast(Any, request), payload, "handoff-model", None
        )

        sent: list[str] = []

        async def send(message):
            sent.append(message["type"])

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        }
        await asyncio.wait_for(
            response(scope, request.receive_channel, send), timeout=5
        )

        assert "http.response.body" in sent
        assert request.receive_channel.max_in_flight == 1, (
            "the cancelled watcher's receive must unwind before the "
            "StreamingResponse reads the channel"
        )
    finally:
        await model_registry.unload(model_settings.name)


async def test_blocked_send_cancellation_propagates_through_response(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A request cancelled while the response's own cleanup awaits the
    model's async release must end CANCELLED — the response boundary
    awaits the settlement without absorbing the cancellation (the
    settlement continues, shielded, in the background)."""
    from starlette.requests import ClientDisconnect

    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="blocked-cancel-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("blocked-cancel-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        response = await endpoints.infer_stream(
            cast(Any, ConnectedRequest()), payload, "blocked-cancel-model", None
        )

        send_release = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                await send_release.wait()
                raise OSError("client disconnected")

        async def receive():
            await asyncio.sleep(3600)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        }
        asgi_call = asyncio.create_task(response(scope, receive, send))
        await asyncio.sleep(0.05)

        # The send fails (the disconnected client): the response's
        # cleanup starts the settlement and awaits it — the model's
        # async cleanup is waiting on the release gate.
        send_release.set()
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)

        # Cancel the REQUEST while the cleanup is in flight: the
        # cancellation must propagate (the task ends cancelled), not
        # be absorbed into a normal return with cleanup pending.
        asgi_call.cancel()
        with contextlib.suppress(asyncio.CancelledError, ClientDisconnect):
            await asyncio.wait_for(asgi_call, timeout=5)
        assert asgi_call.cancelled(), (
            "a request cancelled during the response's cleanup must "
            "end cancelled, not return normally"
        )

        # The settlement continued in the background: the cleanup
        # completes on the gate.
        slow_release.set()
        await asyncio.wait_for(stream_closed.wait(), timeout=5)
        assert stream_closed.is_set()
    finally:
        await model_registry.unload(model_settings.name)


async def test_receive_failure_is_not_a_client_disconnect_stream(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A receive wrapper that RAISES during the prefetch is a
    transport failure, not a client disconnect: the endpoint settles
    the producer and propagates the error — 499 is reserved for an
    actual http.disconnect."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="recvfail-stream-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class FailingReceiveRequest:
            headers: dict = {}

            async def receive(self):
                raise OSError("broken transport")

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        with pytest.raises(OSError, match="broken transport"):
            await asyncio.wait_for(
                endpoints.infer_stream(
                    cast(Any, FailingReceiveRequest()),
                    payload,
                    "recvfail-stream-model",
                    None,
                ),
                timeout=5,
            )
    finally:
        await model_registry.unload(model_settings.name)


async def test_receive_failure_is_not_a_client_disconnect_single_shot(
    settings: Settings, model_registry: MultiModelRegistry
):
    """The single-shot SSE branch: a receive wrapper that RAISES is a
    transport failure, not a client disconnect — the error propagates
    after settling the inference."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import SumModel

    model_settings = ModelSettings(
        name="recvfail-single-model",
        implementation=SumModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class FailingReceiveRequest:
            headers: dict = {}

            async def receive(self):
                raise OSError("broken transport")

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        with pytest.raises(OSError, match="broken transport"):
            await asyncio.wait_for(
                endpoints.infer_stream(
                    cast(Any, FailingReceiveRequest()),
                    payload,
                    "recvfail-single-model",
                    None,
                ),
                timeout=5,
            )
    finally:
        await model_registry.unload(model_settings.name)


async def test_cancelled_receive_watcher_still_settles_producer(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A receive watcher that completes by CANCELLATION (not a
    disconnect) must not strand the producer: the settlement starts
    BEFORE the watcher is inspected (.exception() raises on a
    cancelled task), the caller stays alive, and the model's async
    cleanup completes."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="watcher-cancel-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("watcher-cancel-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class SelfCancellingReceive:
            """A receive channel that cancels its own calling task
            (the watcher): the watcher completes cancelled — not a
            disconnect, and the caller task stays alive."""

            headers: dict = {}

            async def receive(self):
                task = asyncio.current_task()
                if task is not None:
                    task.cancel()
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        # The CALLER task stays ALIVE throughout: the endpoint's
        # cancellation outcome is contained by the caller (the way a
        # server's error handling contains it), so the request-task
        # backstop never fires — only the settlement itself can clean
        # the producer.
        caller_done = asyncio.Event()

        async def alive_caller() -> None:
            with contextlib.suppress(asyncio.CancelledError):
                await endpoints.infer_stream(
                    cast(Any, SelfCancellingReceive()),
                    payload,
                    "watcher-cancel-model",
                    None,
                )
            # Stay alive after the endpoint call: the request-task
            # backstop must not fire while the test asserts (the
            # reviewer's probe: the caller lived on).
            await caller_done.wait()

        request_task = asyncio.create_task(alive_caller())
        try:
            await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
            slow_release.set()
            await asyncio.wait_for(stream_closed.wait(), timeout=5)
            assert stream_closed.is_set(), (
                "the producer must settle even when the watcher was "
                "cancelled and the caller stayed alive"
            )
        finally:
            caller_done.set()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(request_task, timeout=5)
    finally:
        await model_registry.unload(model_settings.name)


async def test_single_shot_disconnect_cancellation_preserves_cleanup(
    settings: Settings, model_registry: MultiModelRegistry
):
    """The single-shot counterpart of the streaming cleanup test: a
    disconnect settles the inference (the model enters its async
    finally); cancelling the REQUEST while that cleanup is in flight
    must not reach the inference task a second time — the request
    ends cancelled and the model's cleanup completes."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedSingleShotModel

    model_settings = ModelSettings(
        name="single-cancel-model",
        implementation=TrackedSingleShotModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("single-cancel-model")
        stream_closed = asyncio.Event()
        cleanup_entered = asyncio.Event()
        slow_release = asyncio.Event()
        setattr(model, "stream_closed", stream_closed)
        setattr(model, "cleanup_entered", cleanup_entered)
        setattr(model, "slow_release", slow_release)
        # A gated predict (inside the model's try): the settlement's
        # cancel lands mid-predict, so the model's async finally runs
        # (uninterrupted by the settlement) and waits on the release
        # gate.
        predict_gate = asyncio.Event()
        setattr(model, "predict_gate", predict_gate)

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class DisconnectingRequest:
            headers: dict = {}

            async def receive(self):
                return {"type": "http.disconnect"}

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        request_task = asyncio.create_task(
            endpoints.infer_stream(
                cast(Any, DisconnectingRequest()),
                payload,
                "single-cancel-model",
                None,
            )
        )
        # The disconnect settles the inference (the cancel lands on
        # the gated predict); the model's async finally then runs and
        # waits on the release gate.
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)

        # Cancel the REQUEST while the cleanup is in flight: the
        # settlement is shielded — the cancellation propagates without
        # a second cancel into the inference task.
        request_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(request_task, timeout=5)
        assert request_task.cancelled()

        slow_release.set()
        await asyncio.wait_for(stream_closed.wait(), timeout=5)
        assert stream_closed.is_set(), (
            "the model's async cleanup must survive the request's " "cancellation"
        )
    finally:
        await model_registry.unload(model_settings.name)


async def test_normal_completion_cancellation_not_swallowed(
    settings: Settings, model_registry: MultiModelRegistry
):
    """A request cancelled while the body iterator's cleanup awaits
    the settlement (the end-of-stream marker ALREADY QUEUED — the
    producer finished naturally) must end CANCELLED: the iterator
    awaits the settlement WITHOUT absorbing the cancellation — the
    terminal more_body=False frame must not follow a recorded
    cancellation.

    The deterministic shape: a one-event model with no cleanup gate.
    The first body send holds the iterator suspended AT ITS YIELD
    while the producer finishes (the marker queues), then schedules
    the cancellation via call_soon — it lands on the iterator's
    settlement await after the send returns and the marker is
    consumed."""
    from mlserver.handlers import DataPlane
    from mlserver.rest.endpoints import Endpoints
    from tests.fixtures import TrackedStreamModel

    model_settings = ModelSettings(
        name="terminal-cancel-model",
        implementation=TrackedStreamModel,
        parallel_workers=0,
    )
    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model("terminal-cancel-model")
        setattr(model, "stream_length", 1)  # one event, no cleanup gate

        data_plane = DataPlane(settings=settings, model_registry=model_registry)
        endpoints = Endpoints(data_plane)

        class ConnectedRequest:
            headers: dict = {}

            async def receive(self):
                await asyncio.sleep(3600)

        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                )
            ]
        )
        response = await endpoints.infer_stream(
            cast(Any, ConnectedRequest()), payload, "terminal-cancel-model", None
        )

        sent: list[dict] = []

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                # The iterator is suspended AT ITS YIELD (inside this
                # send): let the producer finish naturally so the
                # end-of-stream marker is queued, then schedule the
                # request cancellation — it fires after this send
                # returns, once the iterator has consumed the marker
                # and entered its settlement await.
                await asyncio.sleep(0.05)
                asyncio.get_running_loop().call_soon(asgi_call.cancel)

        async def receive():
            await asyncio.sleep(3600)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        }
        asgi_call = asyncio.create_task(response(scope, receive, send))

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(asgi_call, timeout=5)
        assert asgi_call.cancelled(), (
            "a request cancelled during the iterator's cleanup must "
            "end cancelled, not return normally"
        )

        # No terminal frame after the recorded cancellation.
        terminal = [
            m
            for m in sent
            if m["type"] == "http.response.body" and m.get("more_body") is False
        ]
        assert not terminal, "no terminal frame may follow a cancellation"
    finally:
        await model_registry.unload(model_settings.name)
