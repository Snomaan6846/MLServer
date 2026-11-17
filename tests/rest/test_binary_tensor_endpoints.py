"""Endpoint-level tests for the binary tensor extension.

Exercises the full HTTP application with ``rest_binary_tensor``
enabled — route composition, per-output negotiation, error mapping,
metrics accounting and header propagation.  Mirrors the httpx/ASGI
pattern of ``test_worker_body_errors.py``.
"""

from collections.abc import Callable, Generator
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from httpx import AsyncClient
import numpy as np
import orjson
import pytest

from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.parallel.model import ParallelModel
from mlserver.rest.app import create_app
from mlserver.types import (
    InferenceResponse,
    Parameters,
    ResponseOutput,
    TensorData,
)

AppFactory = Callable[[bool], FastAPI]

# The sum model sums along axis 1: inputs are [batch, features].
INPUT_SHAPE = [1, 3]
INPUT_VALUES = [1.0, 2.0, 3.0]  # sum = 6.0


def _json_payload(outputs: list | None = None, parameters: dict | None = None):
    request: dict = {
        "inputs": [
            {
                "name": "input-0",
                "datatype": "FP32",
                "shape": INPUT_SHAPE,
                "data": INPUT_VALUES,
            }
        ]
    }
    if outputs is not None:
        request["outputs"] = outputs
    if parameters is not None:
        request["parameters"] = parameters
    return orjson.dumps(request)


def _framed_body(header: dict, payload: bytes = b"") -> tuple[bytes, int]:
    header_bytes = orjson.dumps(header)
    return header_bytes + payload, len(header_bytes)


def _binary_input_header(outputs: list | None = None, parameters: dict | None = None):
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
    if outputs is not None:
        header["outputs"] = outputs
    if parameters is not None:
        header["parameters"] = parameters
    return _framed_body(header, tensor.tobytes())


async def _post_framed(client: AsyncClient, body: bytes, header_length: int):
    return await client.post(
        "/v2/models/sum-model/infer",
        content=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(header_length),
        },
    )


async def _post_json(client: AsyncClient, payload: bytes):
    return await client.post(
        "/v2/models/sum-model/infer",
        content=payload,
        headers={"Content-Type": "application/json"},
    )


def _parse_framed(response) -> tuple[dict, bytes]:
    header_length = int(response.headers["inference-header-content-length"])
    return (
        orjson.loads(response.content[:header_length]),
        response.content[header_length:],
    )


@pytest.fixture
def binary_app(
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
) -> Generator[AppFactory, None, None]:
    """Non-parallel app for success-path tests (real inference happens)."""

    def factory(worker_json: bool) -> FastAPI:
        data_plane._settings.rest_worker_json = worker_json
        data_plane._settings.rest_binary_tensor = True
        return create_app(data_plane._settings, data_plane, model_repository_handlers)

    lookup = AsyncMock(return_value=sum_model)
    with patch.object(data_plane._model_registry, "get_model", lookup):
        yield factory


@pytest.fixture(params=[False, True], ids=["nonparallel", "parallel"])
def binary_error_app(
    request: pytest.FixtureRequest,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
) -> Generator[AppFactory, None, None]:
    """App for error-path tests on both model forms (no inference happens)."""
    model = ParallelModel(sum_model, Mock()) if request.param else sum_model
    lookup = AsyncMock(return_value=model)

    def factory(worker_json: bool) -> FastAPI:
        data_plane._settings.rest_worker_json = worker_json
        data_plane._settings.rest_binary_tensor = True
        return create_app(data_plane._settings, data_plane, model_repository_handlers)

    with patch.object(data_plane._model_registry, "get_model", lookup):
        yield factory


class TestOutputNegotiation:
    """KServe per-output binary resolution through the HTTP app."""

    async def test_json_request_can_request_binary_outputs(self, binary_app):
        """KServe allows binary responses to JSON requests."""
        payload = _json_payload(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header, payload_bytes = _parse_framed(response)
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4
        assert np.frombuffer(payload_bytes, dtype="<f4") == 6.0

    async def test_framed_global_binary_without_explicit_outputs(self, binary_app):
        """binary_data_output=true with no outputs listed applies to all."""
        body, header_length = _binary_input_header(
            parameters={"binary_data_output": True}
        )
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header, _ = _parse_framed(response)
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    async def test_mixed_outputs_resolved_per_output(
        self, data_plane, model_repository_handlers, sum_model
    ):
        """One binary + one JSON output in a single framed response."""

        async def predict(payload):
            return InferenceResponse(
                model_name="sum-model",
                outputs=[
                    ResponseOutput(
                        name="a", datatype="FP32", shape=[1], data=TensorData([1.0])
                    ),
                    ResponseOutput(
                        name="b", datatype="FP32", shape=[1], data=TensorData([2.0])
                    ),
                ],
            )

        lookup = AsyncMock(return_value=sum_model)
        with patch.object(sum_model, "predict", AsyncMock(side_effect=predict)):
            with patch.object(data_plane._model_registry, "get_model", lookup):
                data_plane._settings.rest_worker_json = False
                data_plane._settings.rest_binary_tensor = True
                app = create_app(
                    data_plane._settings, data_plane, model_repository_handlers
                )
                payload = _json_payload(
                    outputs=[
                        {"name": "a", "parameters": {"binary_data": True}},
                        {"name": "b", "parameters": {"binary_data": False}},
                    ]
                )
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await _post_json(client, payload)

        assert response.status_code == 200
        header, payload_bytes = _parse_framed(response)
        outputs = {o["name"]: o for o in header["outputs"]}
        assert outputs["a"]["parameters"]["binary_data_size"] == 4
        assert "binary_data_size" not in outputs["b"].get("parameters", {})
        assert outputs["b"]["data"] == [2.0]
        assert len(payload_bytes) == 4  # only the binary output's bytes

    async def test_all_json_explicit_outputs_return_plain_json(self, binary_app):
        """Explicitly all-JSON outputs must not produce an empty framed response."""
        payload = _json_payload(
            outputs=[{"name": "total", "parameters": {"binary_data": False}}]
        )
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "inference-header-content-length" not in response.headers
        assert response.json()["outputs"][0]["data"] == [6.0]

    async def test_default_json_response(self, binary_app):
        body, header_length = _binary_input_header()
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/json"
        assert "inference-header-content-length" not in response.headers


class TestErrorMapping:
    """Malformed binary metadata gets 4xx, never 500."""

    @pytest.mark.parametrize(
        "header",
        [
            {"inputs": None},
            {"inputs": [42]},
            {"inputs": [{"shape": [1], "datatype": "FP32"}]},  # missing name
            {"inputs": [{"name": "x", "shape": ["a"], "datatype": "FP32"}]},
            {"inputs": [{"name": "x", "shape": [1], "datatype": [1]}]},
        ],
        ids=["null-inputs", "scalar-item", "missing-name", "bad-shape", "bad-datatype"],
    )
    async def test_malformed_metadata_is_400(self, binary_error_app, header):
        body, header_length = _framed_body(header, b"\x00" * 4)
        async with AsyncClient(
            app=binary_error_app(False), base_url="http://test"
        ) as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 400
        assert response.headers["content-type"] == "application/json"
        assert "detail" in response.json()

    async def test_malformed_gzip_is_400(self, binary_error_app):
        """Body-read errors keep the JSON 400 on the binary route too."""
        from gzip import compress

        async with AsyncClient(
            app=binary_error_app(False), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v2/models/sum-model/infer",
                content=compress(b"{}", mtime=0)[:-4],
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )

        assert response.status_code == 400
        assert response.json() == {"detail": "There was an error parsing the body"}

    async def test_text_plain_json_body_rejected(self, binary_error_app):
        """JSON-shaped text/plain bodies get the validation response."""
        async with AsyncClient(
            app=binary_error_app(False), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v2/models/sum-model/infer",
                content=_json_payload(),
                headers={"Content-Type": "text/plain"},
            )

        assert response.status_code == 422


class TestMetricsAccounting:
    async def test_single_accounting_per_request(self, binary_app, data_plane, mocker):
        spy = mocker.spy(DataPlane, "_infer_metrics")

        body, header_length = _binary_input_header(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200
        assert spy.call_count == 1


@pytest.fixture
def header_app(
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
) -> Generator[FastAPI, None, None]:
    """Non-parallel app whose model sets response headers (CloudEvents-style)."""

    async def predict(payload):
        return InferenceResponse(
            model_name="sum-model",
            model_version="v1.2.3",
            outputs=[
                ResponseOutput(
                    name="total", datatype="FP32", shape=[1, 1], data=TensorData([6.0])
                )
            ],
            parameters=Parameters(
                headers={"x-model-result": "kept", "ce-id": "abc-123"}
            ),
        )

    lookup = AsyncMock(return_value=sum_model)
    with patch.object(sum_model, "predict", AsyncMock(side_effect=predict)):
        with patch.object(data_plane._model_registry, "get_model", lookup):
            data_plane._settings.rest_worker_json = False
            data_plane._settings.rest_binary_tensor = True
            yield create_app(
                data_plane._settings, data_plane, model_repository_handlers
            )


@pytest.fixture
def spoofing_header_app(
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
) -> Generator[FastAPI, None, None]:
    """Non-parallel app whose model echoes transport-owned headers."""

    async def predict(payload):
        return InferenceResponse(
            model_name="sum-model",
            model_version="v1.2.3",
            outputs=[
                ResponseOutput(
                    name="total", datatype="FP32", shape=[1, 1], data=TensorData([6.0])
                )
            ],
            parameters=Parameters(
                headers={
                    # Must not relabel the binary frame as JSON:
                    "content-type": "application/json",
                    # Must not fork the framing header into two values:
                    "inference-header-content-length": "bogus",
                    "content-length": "999",
                    # Model metadata survives:
                    "x-model-result": "kept",
                }
            ),
        )

    lookup = AsyncMock(return_value=sum_model)
    with patch.object(sum_model, "predict", AsyncMock(side_effect=predict)):
        with patch.object(data_plane._model_registry, "get_model", lookup):
            data_plane._settings.rest_worker_json = False
            data_plane._settings.rest_binary_tensor = True
            yield create_app(
                data_plane._settings, data_plane, model_repository_handlers
            )


class TestResponseHeaders:
    async def test_binary_response_preserves_model_headers(self, header_app):
        """Model/CloudEvents headers survive on binary responses."""
        body, header_length = _binary_input_header(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        async with AsyncClient(app=header_app, base_url="http://test") as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200
        assert response.headers["x-model-result"] == "kept"
        assert "abc-123" in response.headers["ce-id"]
        assert "inference-header-content-length" in response.headers
        # The internal headers representation must not leak into the
        # binary JSON header.
        header, _ = _parse_framed(response)
        assert "headers" not in (header.get("parameters") or {})

    async def test_model_cannot_override_transport_headers(self, spoofing_header_app):
        """Model-supplied transport headers (content-type, an echoed
        lowercase inference-header-content-length, content-length) are
        dropped: the binary frame keeps its octet-stream label, and the
        framing header has exactly one authoritative value."""
        body, header_length = _binary_input_header(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        async with AsyncClient(
            app=spoofing_header_app, base_url="http://test"
        ) as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["x-model-result"] == "kept"
        # Exactly one framing header, carrying the real header length
        ihcl_values = response.headers.get_list("inference-header-content-length")
        assert len(ihcl_values) == 1
        header, binary = _parse_framed(response)
        assert ihcl_values == [str(len(response.content) - len(binary))]
        # Content-length describes the actual body, not the model's value
        assert response.headers["content-length"] == str(len(response.content))


class TestRouteComposition:
    async def test_all_json_overrides_under_true_default_return_plain_json(
        self, binary_app
    ):
        """Every output overriding binary_data=False returns plain JSON
        even when the request-wide default is True."""
        payload = _json_payload(
            outputs=[
                {"name": "total", "parameters": {"binary_data": False}},
                {"name": "extra", "parameters": {"binary_data": False}},
            ],
            parameters={"binary_data_output": True},
        )
        async with AsyncClient(app=binary_app(False), base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "inference-header-content-length" not in response.headers

    async def test_worker_json_path_preserved_for_json_bodies(self, binary_app, mocker):
        """With both settings on, JSON bodies keep the worker JSON path."""
        from mlserver.rest.worker import RawEndpoints

        spy = mocker.spy(RawEndpoints, "infer")
        app = binary_app(worker_json=True)

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_json(client, _json_payload())

        assert response.status_code == 200
        assert spy.call_count == 1

    async def test_worker_json_path_preserves_binary_negotiation(self, binary_app):
        """Binary-output negotiation survives the worker JSON composition."""
        app = binary_app(worker_json=True)
        payload = _json_payload(
            outputs=[{"name": "total", "parameters": {"binary_data": True}}]
        )
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header, payload_bytes = _parse_framed(response)
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4
        assert np.frombuffer(payload_bytes, dtype="<f4") == 6.0

    async def test_worker_json_global_binary_output_negotiation(self, binary_app):
        """Request-wide binary_data_output also survives the composition."""
        app = binary_app(worker_json=True)
        payload = _json_payload(parameters={"binary_data_output": True})
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header, _ = _parse_framed(response)
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    async def test_escaped_negotiation_key_composes_correctly(self, binary_app):
        """An escaped JSON key spelling still negotiates binary output."""
        app = binary_app(worker_json=True)
        payload = (
            b'{"inputs":[{"name":"input-0","datatype":"FP32","shape":[1,3],'
            b'"data":[1,2,3]}],"parameters":{"binary_\\u0064ata_output":true}}'
        )
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_json(client, payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        header, _ = _parse_framed(response)
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 4

    async def test_worker_json_plain_requests_still_delegate(self, binary_app, mocker):
        """Requests without binary markers keep the worker JSON path."""
        from mlserver.rest.worker import RawEndpoints

        spy = mocker.spy(RawEndpoints, "infer")
        app = binary_app(worker_json=True)

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_json(client, _json_payload())

        assert response.status_code == 200
        assert spy.call_count == 1

    async def test_binary_frames_take_binary_path_with_worker_json(
        self, binary_app, mocker
    ):
        from mlserver.rest.worker import RawEndpoints

        spy = mocker.spy(RawEndpoints, "infer")
        app = binary_app(worker_json=True)

        body, header_length = _binary_input_header()
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await _post_framed(client, body, header_length)

        assert response.status_code == 200
        # The binary frame bypasses the worker JSON path entirely.
        assert spy.call_count == 0


class TestBatchingCompatibility:
    def test_array_inputs_restored_before_list_batching(
        self, data_plane: DataPlane, sum_model: MLModel
    ):
        """A batched model must never see ndarray-backed inputs."""
        from mlserver.types import InferenceRequest, RequestInput

        tensor = np.arange(4, dtype="<f4").reshape(2, 2)
        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="FP32",
                    shape=[2, 2],
                    data=TensorData(tensor.reshape(-1)),
                )
            ]
        )

        model = Mock(spec=sum_model)
        model.settings.max_batch_size = 2
        model.settings.max_batch_time = 0

        data_plane._settings.grpc_raw_input_buffers = False
        data_plane._prepare_payload(payload, model)

        restored = getattr(payload.inputs[0].data, "root", payload.inputs[0].data)
        assert not isinstance(restored, np.ndarray)
        assert list(restored) == list(tensor.reshape(-1))

    def test_unbatched_models_keep_array_inputs(
        self, data_plane: DataPlane, sum_model: MLModel
    ):
        """Without batching, the compact representation is preserved."""
        from mlserver.types import InferenceRequest, RequestInput

        tensor = np.arange(4, dtype="<f4")
        payload = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input-0",
                    datatype="FP32",
                    shape=[4],
                    data=TensorData(tensor),
                )
            ]
        )

        model = Mock(spec=sum_model)
        model.settings.max_batch_size = 0
        model.settings.max_batch_time = 0

        data_plane._settings.grpc_raw_input_buffers = False
        data_plane._prepare_payload(payload, model)

        restored = getattr(payload.inputs[0].data, "root", payload.inputs[0].data)
        assert isinstance(restored, np.ndarray)


def test_binary_output_itemsize_mismatch_falls_back_to_json():
    """A response array whose byte width differs from the declared
    datatype cannot pack losslessly: the output falls back to inline
    JSON (protocol-legal) instead of silently truncating or padding —
    the same check the gRPC raw path raises on."""
    import numpy as np

    from mlserver.rest.binary import _encode_output_bytes

    # int32 array declared INT64: widths differ (4 vs 8 bytes).
    output = Mock()
    output.datatype = "INT64"
    output.data = np.array([1, 2], dtype=np.int32)
    assert _encode_output_bytes(output) is None

    # Same-width kind conversion still packs (values convert exactly).
    output2 = Mock()
    output2.datatype = "FP64"
    output2.data = np.array([1, 2], dtype=np.int64)
    packed = _encode_output_bytes(output2)
    assert packed == np.array([1, 2], dtype=np.float64).tobytes()


def test_binary_output_big_endian_normalises_to_little_endian():
    """A big-endian response array packs as little-endian bytes (the
    wire format's requirement), matching the little-endian array's
    output exactly."""
    import numpy as np

    from mlserver.rest.binary import _encode_output_bytes

    big_endian = np.array([1.5, -2.5], dtype=">f4")
    little_endian = np.array([1.5, -2.5], dtype="<f4")

    output_be = Mock()
    output_be.datatype = "FP32"
    output_be.data = big_endian
    output_le = Mock()
    output_le.datatype = "FP32"
    output_le.data = little_endian

    assert _encode_output_bytes(output_be) == _encode_output_bytes(output_le)
    assert _encode_output_bytes(output_be) == little_endian.tobytes()
