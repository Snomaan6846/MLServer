"""Tests for opt-in worker-side processing of binary tensor REST requests.

Exercises the same dispatch boundary as ``test_worker.py`` for JSON: the
worker parses the binary frame, runs inference and encodes the negotiated
response, while the frontend keeps inference accounting.  Frontend
routing, fallbacks and error mapping mirror the raw JSON contract, the
accounting tests mirror ``test_worker_metrics.py``, the app-level tests
mirror ``test_binary_tensor_endpoints.py`` (settings matrix), and the
parity tests pin the worker reply to the bytes the local frontend path
would produce.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from httpx import AsyncClient
import numpy as np
import orjson
import pytest
from pytest_mock import MockerFixture
from starlette.requests import Request

from mlserver import ModelSettings, Settings
from mlserver.errors import ModelNotFound, ModelNotReady
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.parallel.errors import WorkerError
from mlserver.parallel.messages import ModelRequestMessage, ModelResponseMessage
from mlserver.parallel.model import ParallelModel
from mlserver.parallel.worker import Worker
from mlserver.registry import MultiModelRegistry
from mlserver.rest import worker as raw
from mlserver.rest.app import create_app
from mlserver.rest.binary import parse_binary_request
from mlserver.rest.binary_endpoints import (
    BinaryTensorEndpoints,
    encode_worker_binary_reply,
)
from mlserver.rest.worker import RawReply
from mlserver.types import InferenceRequest, InferenceResponse

# The sum model sums along axis 1: inputs are [batch, features].
INPUT_SHAPE = [1, 3]
INPUT_VALUES = [1.0, 2.0, 3.0]  # sum = 6.0

BINARY_HEADERS = {
    "content-type": "application/octet-stream",
    "inference-header-content-length": "0",  # replaced per test
    "x-test": "echo-me",
}


@pytest.fixture
def settings(settings: Settings) -> Settings:
    # These tests exercise the dispatch boundary without spawning subprocesses.
    settings.parallel_workers = 0
    settings.rest_binary_tensor = True
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


def _framed_body(header: dict, payload: bytes = b"") -> tuple[bytes, int]:
    header_bytes = orjson.dumps(header)
    return header_bytes + payload, len(header_bytes)


def binary_frame(
    outputs: list | None = None,
    parameters: dict | None = None,
    request_id: str | None = None,
) -> tuple[bytes, int]:
    """A valid binary tensor request for the sum model."""
    tensor = np.array(INPUT_VALUES, dtype="<f4")
    header: dict = {
        "inputs": [
            {
                "name": "input-0",
                "datatype": "FP32",
                "shape": INPUT_SHAPE,
                "parameters": {"binary_data_size": tensor.nbytes},
            }
        ]
    }
    if request_id is not None:
        header["id"] = request_id
    if outputs is not None:
        header["outputs"] = outputs
    if parameters is not None:
        header["parameters"] = parameters
    return _framed_body(header, tensor.tobytes())


def frame_headers(header_length: int) -> dict[str, str]:
    headers = dict(BINARY_HEADERS)
    headers["inference-header-content-length"] = str(header_length)
    return headers


def binary_request(body: bytes, header_length: int) -> Request:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/infer",
        "headers": [
            (key.encode(), value.encode())
            for key, value in frame_headers(header_length).items()
        ],
    }
    return Request(scope, receive)


def binary_message(
    body: bytes, model: MLModel, header_length: int
) -> ModelRequestMessage:
    return ModelRequestMessage(
        id="dispatch-id",
        model_name=model.name,
        model_version=model.version,
        method_name=raw.RAW_INFER_METHOD,
        method_args=[body, frame_headers(header_length)],
    )


def worker_stub(
    settings: Settings, model_registry: MultiModelRegistry
) -> SimpleNamespace:
    return SimpleNamespace(_settings=settings, _model_registry=model_registry)


def parallel(model: MLModel, reply: RawReply | None = None) -> ParallelModel:
    if reply is None:
        reply = RawReply(b'{"id":"from-worker"}', {"x-reply": "yes"}, worker_pid=101)
    dispatcher = Mock()
    dispatcher.dispatch_request = AsyncMock(
        return_value=ModelResponseMessage(return_value=reply)
    )
    wrapped = ParallelModel(model, dispatcher)
    wrapped.ready = True
    return wrapped


def parallel_local(model: MLModel) -> ParallelModel:
    """A parallel model whose ordinary methods run the real model.

    The dispatcher answers ``RAW_INFER_METHOD`` with a canned reply (so
    tests can prove it was never called) and forwards every other method
    — e.g. ``predict`` — to the wrapped model, matching what a real
    worker pool would return for the local-parsing fallback path.
    """

    async def dispatch(message: ModelRequestMessage) -> ModelResponseMessage:
        if message.method_name == raw.RAW_INFER_METHOD:
            return ModelResponseMessage(
                return_value=RawReply(
                    b'{"id":"from-worker"}', {"x-reply": "yes"}, worker_pid=101
                )
            )
        result = await model.predict(*message.method_args, **message.method_kwargs)
        return ModelResponseMessage(return_value=result)

    dispatcher = Mock()
    dispatcher.dispatch_request = AsyncMock(side_effect=dispatch)
    wrapped = ParallelModel(model, dispatcher)
    wrapped.ready = True
    return wrapped


def dispatch_mock(model: ParallelModel) -> AsyncMock:
    method = model._dispatcher.dispatch_request
    assert isinstance(method, AsyncMock)
    return method


def endpoints(data_plane: DataPlane, worker_binary: bool) -> BinaryTensorEndpoints:
    return BinaryTensorEndpoints(data_plane, worker_binary=worker_binary)


class TestWorkerBinaryProcessing:
    """The worker-side half: parse, infer, encode the negotiated reply."""

    async def test_binary_request_gets_binary_reply(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """Binary outputs negotiate a framed reply with the right headers."""
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)),
            binary_message(body, sum_model, header_length),
        )

        assert response.id == "dispatch-id"
        assert response.exception is None
        reply = response.return_value
        assert isinstance(reply, RawReply)
        assert reply.status == 200
        assert reply.worker_pid == os.getpid()
        assert reply.conversion_duration > 0.0
        assert reply.headers["content-type"] == "application/octet-stream"
        header_length_out = int(reply.headers["Inference-Header-Content-Length"])
        header = orjson.loads(reply.body[:header_length_out])
        payload = reply.body[header_length_out:]
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4
        assert np.frombuffer(payload, dtype="<f4") == 6.0

    async def test_worker_reply_matches_local_frontend_path(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        data_plane: DataPlane,
        sum_model: MLModel,
    ):
        """Parity: the worker reply is byte-identical to the local path."""
        outputs = [{"name": "total", "parameters": {"binary_data": True}}]
        body, header_length = binary_frame(outputs=outputs, request_id="parity-id")

        # Both paths share one DataPlane so only the processing location
        # differs — the encoded reply must be byte-identical either way.
        worker = worker_stub(settings, model_registry)
        worker._rest_raw_data_plane = data_plane
        worker_response = await raw.process_raw_request(
            cast(Any, worker), binary_message(body, sum_model, header_length)
        )
        worker_reply = worker_response.return_value
        assert isinstance(worker_reply, RawReply)

        payload, _ = parse_binary_request(body, header_length)
        raw.insert_headers(payload, frame_headers(header_length))
        inference = await data_plane.infer(payload, sum_model.name, None)
        local_reply = encode_worker_binary_reply(inference, payload)

        assert worker_reply.body == local_reply.body
        assert worker_reply.status == local_reply.status
        assert worker_reply.headers == local_reply.headers
        reply_header_length = int(
            worker_reply.headers["Inference-Header-Content-Length"]
        )
        assert (
            orjson.loads(worker_reply.body[:reply_header_length])["id"] == "parity-id"
        )

    async def test_worker_reply_matches_local_path_json_negotiation(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        data_plane: DataPlane,
        sum_model: MLModel,
    ):
        """Parity also holds when outputs are negotiated as JSON."""
        body, header_length = binary_frame(request_id="parity-id")

        worker = worker_stub(settings, model_registry)
        worker._rest_raw_data_plane = data_plane
        worker_response = await raw.process_raw_request(
            cast(Any, worker), binary_message(body, sum_model, header_length)
        )
        worker_reply = worker_response.return_value
        assert isinstance(worker_reply, RawReply)

        payload, _ = parse_binary_request(body, header_length)
        raw.insert_headers(payload, frame_headers(header_length))
        inference = await data_plane.infer(payload, sum_model.name, None)
        local_reply = encode_worker_binary_reply(inference, payload)

        assert worker_reply.body == local_reply.body
        assert worker_reply.headers == local_reply.headers
        assert "content-type" not in worker_reply.headers
        assert orjson.loads(worker_reply.body)["outputs"][0]["data"] == [6.0]

    async def test_mixed_outputs_resolved_per_output(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """One binary + one JSON output in a single worker reply."""

        async def predict(payload: InferenceRequest) -> InferenceResponse:
            from mlserver.types import ResponseOutput, TensorData

            return InferenceResponse(
                model_name="sum-model",
                outputs=[
                    ResponseOutput(
                        name="a",
                        datatype="FP32",
                        shape=[1],
                        data=TensorData(root=[1.0]),
                    ),
                    ResponseOutput(
                        name="b",
                        datatype="FP32",
                        shape=[1],
                        data=TensorData(root=[2.0]),
                    ),
                ],
            )

        original = sum_model.predict
        sum_model.predict = predict  # type: ignore[method-assign]
        try:
            body, header_length = binary_frame(
                outputs=[
                    {"name": "a", "parameters": {"binary_data": True}},
                    {"name": "b", "parameters": {"binary_data": False}},
                ]
            )
            response = await raw.process_raw_request(
                cast(Any, worker_stub(settings, model_registry)),
                binary_message(body, sum_model, header_length),
            )
        finally:
            sum_model.predict = original  # type: ignore[method-assign]

        reply = response.return_value
        assert isinstance(reply, RawReply)
        assert reply.status == 200
        assert reply.headers["content-type"] == "application/octet-stream"
        header_length_out = int(reply.headers["Inference-Header-Content-Length"])
        header = orjson.loads(reply.body[:header_length_out])
        payload = reply.body[header_length_out:]
        by_name = {output["name"]: output for output in header["outputs"]}
        assert by_name["a"]["parameters"]["binary_data_size"] == 4
        assert "binary_data_size" not in by_name["b"].get("parameters", {})
        assert by_name["b"]["data"] == [2.0]
        assert len(payload) == 4  # only the binary output's bytes
        assert np.frombuffer(payload, dtype="<f4") == 1.0

    async def test_request_wide_binary_default_without_explicit_outputs(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """binary_data_output=true with no outputs listed applies to all."""
        body, header_length = binary_frame(parameters={"binary_data_output": True})
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)),
            binary_message(body, sum_model, header_length),
        )

        reply = response.return_value
        assert isinstance(reply, RawReply)
        assert reply.status == 200
        assert reply.headers["content-type"] == "application/octet-stream"
        header_length_out = int(reply.headers["Inference-Header-Content-Length"])
        header = orjson.loads(reply.body[:header_length_out])
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    async def test_model_and_cloudevents_headers_propagate(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """Echoed request headers survive the worker round trip."""
        body, header_length = binary_frame(request_id="header-test-id")
        message = binary_message(body, sum_model, header_length)
        message.method_args[1]["ce-id"] = "cloudevent-id"  # type: ignore[index]
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)), message
        )

        reply = response.return_value
        assert isinstance(reply, RawReply)
        header = orjson.loads(reply.body)
        assert header["id"] == "header-test-id"
        assert "parameters" not in header or "headers" not in header.get(
            "parameters", {}
        )
        lower = {key.lower(): value for key, value in reply.headers.items()}
        # CloudEvents semantics rewrite ce-id to the request id (matching
        # the raw JSON worker contract); ordinary headers echo untouched.
        assert lower["ce-id"] == "header-test-id"
        assert lower["x-test"] == "echo-me"

    async def test_worker_preserves_model_context(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        """The worker's DataPlane keeps the model context vars."""
        from mlserver.context import model_name_var, model_version_var

        original = sum_model.predict
        contexts = []

        async def predict(payload: InferenceRequest) -> InferenceResponse:
            contexts.append((model_name_var.get(), model_version_var.get()))
            return await original(payload)

        mocker.patch.object(sum_model, "predict", side_effect=predict)
        body, header_length = binary_frame()
        await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)),
            binary_message(body, sum_model, header_length),
        )

        assert contexts == [(sum_model.name, sum_model.version)]

    async def test_worker_uses_one_dataplane_across_requests(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        """The worker's DataPlane is built lazily exactly once."""
        factory = mocker.patch.object(raw, "DataPlane")
        body, header_length = binary_frame()
        stub = worker_stub(settings, model_registry)
        for _ in range(2):
            await raw.process_raw_request(
                cast(Any, stub), binary_message(body, sum_model, header_length)
            )
        factory.assert_called_once_with(
            settings, model_registry, inference_metrics=False
        )

    @pytest.mark.parametrize(
        "failure", [RuntimeError("prediction failed"), asyncio.CancelledError()]
    )
    async def test_prediction_failures_wrap_as_worker_errors(
        self,
        failure: BaseException,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        """Inference failures keep the raw JSON error contract."""
        mocker.patch.object(sum_model, "predict", side_effect=failure)
        body, header_length = binary_frame()
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)),
            binary_message(body, sum_model, header_length),
        )

        assert response.id == "dispatch-id"
        assert response.return_value is None
        assert isinstance(response.exception, WorkerError)
        assert failure.__class__.__name__ in str(response.exception)

    @pytest.mark.parametrize(
        "bad_args", [["not-bytes", {}], [b"{}", "not-headers"], []]
    )
    async def test_worker_rejects_malformed_internal_envelopes(self, bad_args: list):
        envelope = ModelRequestMessage(
            model_name="unused", method_name=raw.RAW_INFER_METHOD, method_args=bad_args
        )
        result = await raw.process_raw_request(cast(Any, SimpleNamespace()), envelope)
        assert result.id == envelope.id
        assert isinstance(result.exception, WorkerError)


class TestWorkerBinaryParseErrors:
    """Malformed frames map to 400 replies — never worker errors."""

    async def _post_frame(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        body: bytes,
        header_length: int,
        mocker: MockerFixture,
    ) -> RawReply:
        factory = mocker.patch.object(raw, "DataPlane")
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)),
            binary_message(body, sum_model, header_length),
        )
        factory.assert_not_called()
        assert response.exception is None
        reply = response.return_value
        assert isinstance(reply, RawReply)
        return reply

    async def test_header_length_beyond_body(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        body, header_length = binary_frame()
        reply = await self._post_frame(
            settings,
            model_registry,
            sum_model,
            body,
            header_length + len(body) + 1,
            mocker,
        )
        assert reply.status == 400
        assert "exceeds" in orjson.loads(reply.body)["detail"]

    @pytest.mark.parametrize("raw_length", ["-1", "not-a-number", ""])
    async def test_invalid_header_length_values(
        self,
        raw_length: str,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        body, _ = binary_frame()
        message = binary_message(body, sum_model, 0)
        headers_arg = message.method_args[1]
        headers_arg["inference-header-content-length"] = raw_length
        factory = mocker.patch.object(raw, "DataPlane")
        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)), message
        )
        factory.assert_not_called()
        reply = response.return_value
        assert isinstance(reply, RawReply)
        assert reply.status == 400
        assert orjson.loads(reply.body)["detail"]

    async def test_invalid_json_header(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        reply = await self._post_frame(
            settings, model_registry, sum_model, b"{invalid", 1, mocker
        )
        assert reply.status == 400
        assert orjson.loads(reply.body)["detail"]

    async def test_header_missing_inputs(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        header = orjson.dumps({"outputs": []})
        reply = await self._post_frame(
            settings, model_registry, sum_model, header, len(header), mocker
        )
        assert reply.status == 400
        assert "inputs" in orjson.loads(reply.body)["detail"]

    async def test_binary_data_size_mismatch(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        tensor = np.array(INPUT_VALUES, dtype="<f4")
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "input-0",
                        "datatype": "FP32",
                        "shape": INPUT_SHAPE,
                        "parameters": {"binary_data_size": tensor.nbytes + 4},
                    }
                ]
            }
        )
        reply = await self._post_frame(
            settings,
            model_registry,
            sum_model,
            header + tensor.tobytes(),
            len(header),
            mocker,
        )
        assert reply.status == 400
        assert orjson.loads(reply.body)["detail"]


class TestFrontendBinaryDispatch:
    """The frontend half: routing, fallbacks and the raw-bytes handoff."""

    async def test_parallel_model_forwards_raw_frame(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """Binary frames cross the dispatch untouched, like raw JSON."""
        model = parallel(sum_model)
        get_model = mocker.patch.object(
            data_plane._model_registry, "get_model", return_value=model
        )
        local = mocker.patch.object(data_plane, "infer")
        body, header_length = binary_frame()

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), model.name
        )

        get_model.assert_awaited_once_with(model.name, None)
        (envelope,) = dispatch_mock(model).call_args.args
        assert envelope.method_name == raw.RAW_INFER_METHOD
        assert envelope.method_args == [body, frame_headers(header_length)]
        # The frontend served the worker's reply without touching it.
        assert response.status_code == 200
        assert response.body == b'{"id":"from-worker"}'
        assert response.headers["x-reply"] == "yes"
        local.assert_not_called()

    async def test_disabled_keeps_local_parsing(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """The opt-in off (default) preserves today's frontend parsing."""
        model = parallel_local(sum_model)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}],
            request_id="local-id",
        )

        response = await endpoints(data_plane, False).infer(
            binary_request(body, header_length), model.name
        )

        # No RAW_INFER dispatch: only the ordinary predict method crossed.
        raw_calls = [
            call.args[0].method_name for call in dispatch_mock(model).call_args_list
        ]
        assert raw_calls == ["predict"]
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert b"local-id" in response.body

    async def test_batched_model_falls_back_to_local_parsing(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """Adaptive batching keeps local parsing, mirroring raw JSON."""
        sum_model.settings.max_batch_size = 4
        model = parallel_local(sum_model)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}],
            request_id="local-id",
        )

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), model.name
        )

        raw_calls = [
            call.args[0].method_name for call in dispatch_mock(model).call_args_list
        ]
        assert raw_calls == ["predict"]
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert b"local-id" in response.body

    async def test_nonparallel_model_falls_back_to_local_parsing(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """Non-parallel models keep local parsing."""
        mocker.patch.object(
            data_plane._model_registry, "get_model", return_value=sum_model
        )
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}],
            request_id="local-id",
        )

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), sum_model.name
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header = orjson.loads(
            response.body[: int(response.headers["inference-header-content-length"])]
        )
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    async def test_worker_reply_error_surfaces(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """A 400 reply from the worker reaches the client unchanged."""
        reply = RawReply(
            orjson.dumps({"detail": "Invalid Inference-Header-Content-Length"}),
            {},
            status=400,
            worker_pid=101,
        )
        model = parallel(sum_model, reply=reply)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), model.name
        )

        assert response.status_code == 400
        assert orjson.loads(response.body)["detail"].startswith("Invalid")

    async def test_not_ready_model_raises(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """A not-ready parallel model surfaces ModelNotReady before dispatch."""
        model = parallel(sum_model)
        model.ready = False
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()

        with pytest.raises(ModelNotReady):
            await endpoints(data_plane, True).infer(
                binary_request(body, header_length), model.name
            )
        dispatch_mock(model).assert_not_called()

    async def test_unknown_model_raises(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """An unknown model surfaces ModelNotFound before any parsing."""
        mocker.patch.object(
            data_plane._model_registry,
            "get_model",
            side_effect=ModelNotFound("missing-model"),
        )
        body, header_length = binary_frame()

        with pytest.raises(ModelNotFound):
            await endpoints(data_plane, True).infer(
                binary_request(body, header_length), "missing-model"
            )

    @pytest.mark.parametrize(
        "reply_value",
        [
            b"not-a-reply",
            RawReply("not-bytes", {}, worker_pid=101),  # type: ignore[arg-type]
        ],
        ids=["non-reply", "non-bytes-body"],
    )
    async def test_worker_returning_invalid_reply_raises(
        self,
        reply_value: Any,
        data_plane: DataPlane,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        """The dispatch guards mirror the raw JSON contract."""
        dispatcher = Mock()
        dispatcher.dispatch_request = AsyncMock(
            return_value=ModelResponseMessage(return_value=reply_value)
        )
        model = ParallelModel(sum_model, dispatcher)
        model.ready = True
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()

        with pytest.raises(TypeError):
            await endpoints(data_plane, True).infer(
                binary_request(body, header_length), model.name
            )

    async def test_worker_reply_from_frontend_pid_raises(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """A reply the frontend produced itself is rejected, not trusted."""
        reply = RawReply(b"{}", {}, worker_pid=os.getpid())
        model = parallel(sum_model, reply=reply)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()

        with pytest.raises(TypeError):
            await endpoints(data_plane, True).infer(
                binary_request(body, header_length), model.name
            )


class TestBinaryBatchingEndToEnd:
    """Binary REST through a real AdaptiveBatcher.

    The batched-model fallback must keep list-based merging intact with
    the opt-in enabled: concurrent binary requests restore their arrays
    (``_prepare_payload``), merge into one batch, and split back into
    per-request responses — identical to the opt-in-off behaviour.
    """

    @pytest.fixture
    def sum_model_settings(self, sum_model_settings: ModelSettings) -> ModelSettings:
        sum_model_settings.max_batch_size = 2
        sum_model_settings.max_batch_time = 0.5
        return sum_model_settings

    async def test_concurrent_binary_requests_batch_and_split(
        self,
        data_plane: DataPlane,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        from mlserver.batching.adaptive import AdaptiveBatcher

        model = await model_registry.get_model("sum-model")
        batcher = AdaptiveBatcher.get_batcher(model)
        assert batcher is not None, "registry must wrap the model with a batcher"

        # Spy on the batcher's inner predict to prove the merge happened.
        merged_shapes: list[list[int]] = []
        original_predict = batcher._predict_fn

        async def spy(payload: InferenceRequest) -> InferenceResponse:
            merged_shapes.append(list(payload.inputs[0].shape))
            return await original_predict(payload)

        batcher._predict_fn = spy

        # Two concurrent requests with distinct values and ids.
        tensor_a = np.array([1.0, 2.0, 3.0], dtype="<f4")  # sum = 6.0
        tensor_b = np.array([4.0, 5.0, 6.0], dtype="<f4")  # sum = 15.0
        frames = []
        for request_id, tensor in (("req-a", tensor_a), ("req-b", tensor_b)):
            header = orjson.dumps(
                {
                    "id": request_id,
                    "inputs": [
                        {
                            "name": "input-0",
                            "datatype": "FP32",
                            "shape": INPUT_SHAPE,
                            "parameters": {"binary_data_size": tensor.nbytes},
                        }
                    ],
                    "outputs": [{"name": "total", "parameters": {"binary_data": True}}],
                }
            )
            frames.append((header + tensor.tobytes(), len(header)))

        endpoint = BinaryTensorEndpoints(data_plane, worker_binary=True)
        response_a, response_b = await asyncio.gather(
            *(
                endpoint.infer(binary_request(body, header_length), "sum-model")
                for body, header_length in frames
            )
        )

        # One merged predict of batch size 2 proves the batcher engaged.
        assert merged_shapes == [[2, 3]]

        for response, request_id, expected in (
            (response_a, "req-a", 6.0),
            (response_b, "req-b", 15.0),
        ):
            assert response.status_code == 200
            header_length = int(response.headers["inference-header-content-length"])
            response_header = orjson.loads(response.body[:header_length])
            payload = response.body[header_length:]
            assert response_header["id"] == request_id
            assert np.frombuffer(payload, dtype="<f4") == expected

    async def test_worker_binary_opt_in_matches_disabled_results(
        self,
        data_plane: DataPlane,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """Batched-model responses are identical with the opt-in on/off."""
        tensor = np.array(INPUT_VALUES, dtype="<f4")
        header = orjson.dumps(
            {
                "id": "parity-batch",
                "inputs": [
                    {
                        "name": "input-0",
                        "datatype": "FP32",
                        "shape": INPUT_SHAPE,
                        "parameters": {"binary_data_size": tensor.nbytes},
                    }
                ],
                "outputs": [{"name": "total", "parameters": {"binary_data": True}}],
            }
        )
        body, header_length = header + tensor.tobytes(), len(header)

        responses = {}
        for enabled in (False, True):
            endpoint = BinaryTensorEndpoints(data_plane, worker_binary=enabled)
            response = await endpoint.infer(
                binary_request(body, header_length), "sum-model"
            )
            responses[enabled] = (
                response.body,
                response.status_code,
                response.headers["inference-header-content-length"],
            )

        assert responses[False] == responses[True]


class TestFrontendBinaryMetrics:
    """Inference accounting belongs to the frontend, mirroring raw JSON."""

    def snapshot(self, plane: DataPlane, name: str) -> tuple[float, ...]:
        labels = {"model": name, "version": None}
        duration = plane._ModelInferRequestDuration.labels(**labels)
        return (
            plane._ModelInferRequestSuccess.labels(**labels)._value.get(),
            plane._ModelInferRequestFailure.labels(**labels)._value.get(),
            duration._count.get(),
            duration._sum.get(),
        )

    async def test_success_recorded_with_excluded_conversion(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """Successes count once; the worker's conversion time is excluded."""
        model = parallel(sum_model)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()
        before = self.snapshot(data_plane, model.name)

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), model.name
        )

        assert response.status_code == 200
        after = self.snapshot(data_plane, model.name)
        assert after[0] == before[0] + 1  # success
        assert after[1] == before[1]  # no failure
        assert after[2] == before[2] + 1  # one duration observation

    async def test_parse_error_reply_not_accounted_as_inference(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """A worker 400 reply is a rejected request, not a failure."""
        reply = RawReply(
            orjson.dumps({"detail": "bad frame"}),
            {},
            status=400,
            worker_pid=101,
        )
        model = parallel(sum_model, reply=reply)
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()
        before = self.snapshot(data_plane, model.name)

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), model.name
        )

        assert response.status_code == 400
        after = self.snapshot(data_plane, model.name)
        assert after == before

    async def test_not_ready_failure_counted(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """A valid request rejected before dispatch counts as a failure."""
        model = parallel(sum_model)
        model.ready = False
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        body, header_length = binary_frame()
        before = self.snapshot(data_plane, model.name)

        with pytest.raises(ModelNotReady):
            await endpoints(data_plane, True).infer(
                binary_request(body, header_length), model.name
            )

        after = self.snapshot(data_plane, model.name)
        assert after[1] == before[1] + 1  # failure

    @pytest.mark.parametrize(
        "lookup_failure", [ModelNotReady("sum-model", None), ModelNotFound("gone")]
    )
    async def test_malformed_frame_takes_priority_over_model_error(
        self,
        lookup_failure: Exception,
        data_plane: DataPlane,
        sum_model: MLModel,
        mocker: MockerFixture,
    ):
        """A malformed frame returns 400 even when the model lookup fails.

        Mirrors RawEndpoints' short-circuit for invalid JSON (and the
        local binary path's parse-before-lookup order): the client sees
        its actual mistake, not an unrelated 404/503.
        """
        mocker.patch.object(
            data_plane._model_registry,
            "get_model",
            side_effect=lookup_failure,
        )
        body, header_length = binary_frame()
        # A frame whose header length exceeds the body is invalid.
        request = binary_request(body, header_length + len(body) + 1)
        before = self.snapshot(data_plane, "sum-model")

        response = await endpoints(data_plane, True).infer(request, "sum-model")

        assert response.status_code == 400
        assert "exceeds" in orjson.loads(response.body)["detail"]
        # An invalid request rejected before dispatch is not a failure.
        after = self.snapshot(data_plane, "sum-model")
        assert after == before

    async def test_fallback_records_exactly_once_through_dataplane(
        self, data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
    ):
        """Non-parallel fallback accounting is identical to the opt-in-off
        behaviour: the dispatch's outer context stays inert and the
        DataPlane's own context records exactly one success."""
        mocker.patch.object(
            data_plane._model_registry, "get_model", return_value=sum_model
        )
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        before = self.snapshot(data_plane, sum_model.name)

        response = await endpoints(data_plane, True).infer(
            binary_request(body, header_length), sum_model.name
        )

        assert response.status_code == 200
        after = self.snapshot(data_plane, sum_model.name)
        # Exactly one success and one duration observation — not zero
        # (dispatch context inert) and not two (no double counting).
        assert after[0] == before[0] + 1
        assert after[1] == before[1]
        assert after[2] == before[2] + 1


class TestSettings:
    def test_default_off(self, settings: Settings):
        assert settings.rest_worker_binary is False

    async def test_worker_gate_routes_binary_method(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """The worker routes RAW_INFER_METHOD when only the binary opt-in is on."""
        settings.rest_worker_json = False
        settings.rest_worker_binary = True
        stub = worker_stub(settings, model_registry)
        body, header_length = binary_frame()
        envelope = binary_message(body, sum_model, header_length)

        response = await Worker._process_request(cast(Any, stub), envelope)

        assert response.id == "dispatch-id"
        assert response.exception is None
        assert isinstance(response.return_value, RawReply)

    async def test_worker_gate_ignores_raw_method_when_both_off(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """With both opt-ins off, RAW_INFER_METHOD follows the normal path."""
        from mlserver.parallel.errors import WorkerError

        settings.rest_worker_json = False
        settings.rest_worker_binary = False
        stub = worker_stub(settings, model_registry)
        body, header_length = binary_frame()
        envelope = binary_message(body, sum_model, header_length)

        # The normal path cannot find a RAW_INFER_METHOD method on the
        # model, so the message wraps as a worker error.
        response = await Worker._process_request(cast(Any, stub), envelope)

        assert isinstance(response.exception, WorkerError)


class TestAppWiring:
    """The settings matrix through the full HTTP application.

    Mirrors the app-factory pattern of ``test_binary_tensor_endpoints.py``.
    With a non-parallel model every combination falls back to local
    parsing, proving the wiring never breaks the local path; the parallel
    dispatch itself is covered at the endpoint level above.
    """

    @pytest.fixture
    def app_factory(
        self,
        data_plane: DataPlane,
        model_repository_handlers: ModelRepositoryHandlers,
        sum_model: MLModel,
    ):
        def factory(worker_json: bool, worker_binary: bool) -> FastAPI:
            data_plane._settings.rest_binary_tensor = True
            data_plane._settings.rest_worker_json = worker_json
            data_plane._settings.rest_worker_binary = worker_binary
            return create_app(
                data_plane._settings, data_plane, model_repository_handlers
            )

        lookup = AsyncMock(return_value=sum_model)
        with patch.object(data_plane._model_registry, "get_model", lookup):
            yield factory

    @pytest.mark.parametrize(
        "worker_json,worker_binary",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
        ids=["local", "json-worker", "binary-worker", "both-workers"],
    )
    async def test_binary_request_served_locally_for_nonparallel_models(
        self, app_factory: Any, worker_json: bool, worker_binary: bool
    ):
        """Every combination keeps the local path for non-parallel models."""
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        app = app_factory(worker_json, worker_binary)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/v2/models/sum-model/infer",
                content=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Inference-Header-Content-Length": str(header_length),
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header = orjson.loads(
            response.content[: int(response.headers["inference-header-content-length"])]
        )
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    @pytest.mark.parametrize(
        "worker_json,worker_binary",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
        ids=["local", "json-worker", "binary-worker", "both-workers"],
    )
    async def test_json_request_unaffected_by_binary_opt_in(
        self, app_factory: Any, worker_json: bool, worker_binary: bool
    ):
        """JSON requests keep their behaviour regardless of binary settings."""
        payload = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "input-0",
                        "datatype": "FP32",
                        "shape": INPUT_SHAPE,
                        "data": INPUT_VALUES,
                    }
                ]
            }
        )
        app = app_factory(worker_json, worker_binary)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/v2/models/sum-model/infer",
                content=payload,
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert orjson.loads(response.content)["outputs"][0]["data"] == [6.0]


class TestBinaryDetectionGating:
    """Binary frame parsing honors the rest_binary_tensor opt-in."""

    async def test_extension_disabled_rejects_smuggled_binary_frame(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """A binary frame under a JSON content type parses as JSON when
        the extension is off — it must not produce a binary reply."""
        settings.rest_worker_json = True
        settings.rest_binary_tensor = False
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        message = binary_message(body, sum_model, header_length)
        # The smuggled request claims to be JSON while carrying the
        # binary framing header.
        headers_arg = message.method_args[1]
        headers_arg["content-type"] = "application/json"

        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)), message
        )

        assert response.exception is None
        reply = response.return_value
        assert isinstance(reply, RawReply)
        # Parsed as JSON: the binary frame body is invalid JSON.
        assert reply.status == 422

    async def test_extension_enabled_still_parses_binary_frames(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        sum_model: MLModel,
    ):
        """The same request parses as binary when the extension is on."""
        settings.rest_worker_json = True
        settings.rest_binary_tensor = True
        body, header_length = binary_frame(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        message = binary_message(body, sum_model, header_length)
        headers_arg = message.method_args[1]
        headers_arg["content-type"] = "application/json"

        response = await raw.process_raw_request(
            cast(Any, worker_stub(settings, model_registry)), message
        )

        assert response.exception is None
        reply = response.return_value
        assert isinstance(reply, RawReply)
        assert reply.status == 200
        assert reply.headers["content-type"] == "application/octet-stream"
