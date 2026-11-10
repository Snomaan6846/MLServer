"""Tests for the KServe v2 binary tensor data extension.

Unit tests verify parsing and encoding correctness.  Integration tests
use the same httpx/ASGI pattern as the existing REST endpoint tests.
"""

import numpy as np
import orjson
import pytest

from mlserver.rest.binary import (
    BinaryTensorError,
    DATATYPE_MAP,
    encode_binary_response,
    get_header_length,
    is_binary_request,
    parse_binary_request,
)
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
)


@pytest.fixture
def fp32_tensor():
    """A [1, 3, 512, 512] FP32 tensor matching RF-DETR's input."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 3, 512, 512)).astype(np.float32)


@pytest.fixture
def binary_body(fp32_tensor):
    """A valid binary tensor request body for RF-DETR."""
    header = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3, 512, 512],
                "datatype": "FP32",
                "parameters": {"binary_data_size": fp32_tensor.nbytes},
            }
        ],
        "outputs": [
            {"name": "dets", "parameters": {"binary_data": True}},
            {"name": "labels", "parameters": {"binary_data": True}},
        ],
    }
    header_bytes = orjson.dumps(header)
    return header_bytes + fp32_tensor.tobytes(), len(header_bytes)


def _unwrap(data):
    """Unwrap TensorData to the underlying payload."""
    return getattr(data, "root", data)


class TestIsBinaryRequest:
    def test_with_header(self):
        assert is_binary_request({"inference-header-content-length": "100"})

    def test_without_header(self):
        assert not is_binary_request({"content-type": "application/json"})

    def test_starlette_normalised(self):
        """Starlette normalises header names to lowercase."""
        assert is_binary_request({"inference-header-content-length": "50"})

    def test_empty_headers(self):
        assert not is_binary_request({})


class TestGetHeaderLength:
    def test_valid(self):
        assert get_header_length({"inference-header-content-length": "225"}) == 225

    def test_zero(self):
        assert get_header_length({"inference-header-content-length": "0"}) == 0

    def test_missing(self):
        with pytest.raises(BinaryTensorError, match="Missing"):
            get_header_length({})

    def test_non_integer(self):
        with pytest.raises(BinaryTensorError, match="not an integer"):
            get_header_length({"inference-header-content-length": "abc"})

    def test_negative(self):
        with pytest.raises(BinaryTensorError, match="negative"):
            get_header_length({"inference-header-content-length": "-1"})


class TestParseBinaryRequest:
    def test_valid_fp32(self, binary_body):
        body, header_length = binary_body
        request = parse_binary_request(body, header_length)

        assert isinstance(request, InferenceRequest)
        assert len(request.inputs) == 1
        assert request.inputs[0].name == "input"
        assert request.inputs[0].datatype == "FP32"
        assert request.inputs[0].shape == [1, 3, 512, 512]

        data = _unwrap(request.inputs[0].data)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32
        assert data.shape == (1, 3, 512, 512)

    def test_zero_copy(self, binary_body):
        """Verify np.frombuffer shares memory with the body."""
        body, header_length = binary_body
        request = parse_binary_request(body, header_length)
        data = _unwrap(request.inputs[0].data)
        assert isinstance(data, np.ndarray)
        assert not data.flags["WRITEABLE"]  # frombuffer is read-only

    def test_header_exceeds_body(self):
        with pytest.raises(BinaryTensorError, match="exceeds body size"):
            parse_binary_request(b"short", 100)

    def test_invalid_json_header(self):
        with pytest.raises(BinaryTensorError, match="Invalid JSON"):
            parse_binary_request(b"not json{{", 8)

    def test_missing_inputs(self):
        header = orjson.dumps({"outputs": []})
        with pytest.raises(BinaryTensorError, match="missing 'inputs'"):
            parse_binary_request(header, len(header))

    def test_both_data_and_binary_size(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [1.0, 2.0],
                        "parameters": {"binary_data_size": 8},
                    }
                ]
            }
        )
        body = header + b"\x00" * 8
        with pytest.raises(BinaryTensorError, match="mutually exclusive"):
            parse_binary_request(body, len(header))

    def test_binary_size_exceeds_remaining(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [4],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 1000},
                    }
                ]
            }
        )
        body = header + b"\x00" * 16
        with pytest.raises(BinaryTensorError, match="exceeds remaining"):
            parse_binary_request(body, len(header))

    def test_size_shape_mismatch(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [4],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 32},
                    }
                ]
            }
        )
        body = header + b"\x00" * 32
        with pytest.raises(BinaryTensorError, match="doesn't match shape"):
            parse_binary_request(body, len(header))

    def test_trailing_bytes(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [2],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 8},
                    }
                ]
            }
        )
        body = header + b"\x00" * 8 + b"extra"
        with pytest.raises(BinaryTensorError, match="unaccounted"):
            parse_binary_request(body, len(header))

    def test_unsupported_datatype(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [1],
                        "datatype": "STRING",
                        "parameters": {"binary_data_size": 4},
                    }
                ]
            }
        )
        body = header + b"\x00" * 4
        with pytest.raises(BinaryTensorError, match="unsupported datatype"):
            parse_binary_request(body, len(header))

    def test_multiple_inputs(self):
        tensor1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "a",
                        "shape": [3],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor1.nbytes},
                    },
                    {
                        "name": "b",
                        "shape": [2, 2],
                        "datatype": "INT64",
                        "parameters": {"binary_data_size": tensor2.nbytes},
                    },
                ]
            }
        )
        body = header + tensor1.tobytes() + tensor2.tobytes()
        request = parse_binary_request(body, len(header))

        assert np.array_equal(_unwrap(request.inputs[0].data), tensor1)
        assert np.array_equal(_unwrap(request.inputs[1].data), tensor2)

    def test_mixed_binary_and_json_inputs(self):
        tensor = np.array([1.0, 2.0], dtype=np.float32)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "binary_input",
                        "shape": [2],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor.nbytes},
                    },
                    {
                        "name": "json_input",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [3.0, 4.0],
                    },
                ]
            }
        )
        body = header + tensor.tobytes()
        request = parse_binary_request(body, len(header))

        assert np.array_equal(_unwrap(request.inputs[0].data), tensor)
        assert list(_unwrap(request.inputs[1].data)) == [3.0, 4.0]


class TestDatatypeMap:
    def test_all_kserve_types_covered(self):
        expected = {
            "BOOL",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "INT8",
            "INT16",
            "INT32",
            "INT64",
            "FP16",
            "FP32",
            "FP64",
        }
        assert expected == set(DATATYPE_MAP.keys())

    def test_multi_byte_little_endian(self):
        for name, dtype in DATATYPE_MAP.items():
            if dtype.itemsize > 1:
                assert dtype.byteorder in (
                    "<",
                    "=",
                ), f"{name} not little-endian compatible: {dtype.byteorder}"


class TestEncodeBinaryResponse:
    def test_roundtrip(self):
        dets = np.random.default_rng(42).uniform(0, 1, (1, 300, 4)).astype(np.float32)
        labels = (
            np.random.default_rng(43).uniform(0, 1, (1, 300, 91)).astype(np.float32)
        )

        response = InferenceResponse(
            model_name="test",
            model_version="1",
            id="req-1",
            outputs=[
                ResponseOutput(
                    name="dets", datatype="FP32", shape=[1, 300, 4], data=dets
                ),
                ResponseOutput(
                    name="labels", datatype="FP32", shape=[1, 300, 91], data=labels
                ),
            ],
        )

        body, header_length = encode_binary_response(response)

        header = orjson.loads(body[:header_length])
        assert header["model_name"] == "test"
        assert len(header["outputs"]) == 2
        assert header["outputs"][0]["parameters"]["binary_data_size"] == dets.nbytes
        assert header["outputs"][1]["parameters"]["binary_data_size"] == labels.nbytes

        offset = header_length
        dets_restored = np.frombuffer(
            body[offset : offset + dets.nbytes], dtype="<f4"
        ).reshape(1, 300, 4)
        offset += dets.nbytes
        labels_restored = np.frombuffer(
            body[offset : offset + labels.nbytes], dtype="<f4"
        ).reshape(1, 300, 91)

        np.testing.assert_array_equal(dets_restored, dets)
        np.testing.assert_array_equal(labels_restored, labels)

    def test_header_length_positive(self):
        response = InferenceResponse(
            model_name="test",
            outputs=[
                ResponseOutput(
                    name="x",
                    datatype="FP32",
                    shape=[1],
                    data=np.array([1.0], dtype=np.float32),
                ),
            ],
        )
        body, header_length = encode_binary_response(response)
        assert header_length > 0
        assert len(body) > header_length

    def test_no_parameters_field_when_empty(self):
        response = InferenceResponse(
            model_name="test",
            outputs=[
                ResponseOutput(
                    name="x",
                    datatype="FP32",
                    shape=[1],
                    data=np.array([1.0], dtype=np.float32),
                ),
            ],
        )
        body, header_length = encode_binary_response(response)
        header = orjson.loads(body[:header_length])
        assert "parameters" not in header


class TestClientWantsBinary:
    def test_no_outputs(self):
        assert not orjson.loads(orjson.dumps({})).get("outputs")

    def test_binary_data_true(self):
        header = {"outputs": [{"name": "x", "parameters": {"binary_data": True}}]}
        from mlserver.rest.binary import client_wants_binary

        assert client_wants_binary(header)

    def test_binary_data_output_true(self):
        header = {
            "outputs": [{"name": "x"}],
            "parameters": {"binary_data_output": True},
        }
        from mlserver.rest.binary import client_wants_binary

        assert client_wants_binary(header)

    def test_binary_data_false(self):
        header = {"outputs": [{"name": "x", "parameters": {"binary_data": False}}]}
        from mlserver.rest.binary import client_wants_binary

        assert not client_wants_binary(header)

    def test_no_parameters(self):
        header = {"outputs": [{"name": "x"}]}
        from mlserver.rest.binary import client_wants_binary

        assert not client_wants_binary(header)
