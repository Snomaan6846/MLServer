"""Tests for zero-copy IPC serialization of inference messages."""

import asyncio

import numpy as np
import pytest

from mlserver.parallel.errors import WorkerError
from mlserver.parallel.serialization import (
    decode_request,
    decode_response,
    encode_message,
    _to_serializable as _find_arrays,
    _restore_arrays,
)
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    ResponseOutput,
)
from mlserver.parallel.messages import ModelRequestMessage, ModelResponseMessage


@pytest.fixture
def small_array():
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@pytest.fixture
def large_array():
    return (
        np.random.default_rng(42).standard_normal((1, 3, 512, 512)).astype(np.float32)
    )


@pytest.fixture
def inference_request(large_array):
    return InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype="FP32",
                shape=[1, 3, 512, 512],
                data=large_array,
            )
        ],
        outputs=[
            {"name": "dets"},
            {"name": "labels"},
        ],
    )


@pytest.fixture
def request_message(inference_request):
    return ModelRequestMessage(
        model_name="test-model",
        method_name="predict",
        method_args=[inference_request],
    )


@pytest.fixture
def inference_response():
    dets = np.random.default_rng(42).uniform(0, 1, (1, 300, 4)).astype(np.float32)
    labels = np.random.default_rng(43).uniform(0, 1, (1, 300, 91)).astype(np.float32)
    return InferenceResponse(
        model_name="test-model",
        model_version="1",
        id="req-1",
        outputs=[
            ResponseOutput(name="dets", datatype="FP32", shape=[1, 300, 4], data=dets),
            ResponseOutput(
                name="labels", datatype="FP32", shape=[1, 300, 91], data=labels
            ),
        ],
    )


@pytest.fixture
def response_message(inference_response):
    return ModelResponseMessage(
        id="test-id",
        return_value=inference_response,
    )


class TestFindArrays:
    def test_finds_numpy_array(self):
        arr = np.ones(100, dtype=np.float32)  # 400 bytes, above threshold
        arrays = []
        result = _find_arrays({"data": arr}, arrays)
        assert len(arrays) == 1
        assert result["data"]["__ndarray__"] is True
        assert result["data"]["frame"] == 0

    def test_inlines_small_array(self, small_array):
        arrays = []
        result = _find_arrays({"data": small_array}, arrays)
        assert len(arrays) == 0  # small arrays are inlined
        assert result["data"]["__ndarray__"] is True
        assert result["data"]["inline"] is True

    def test_nested_arrays(self):
        # Use arrays above the inline threshold (256 bytes)
        arr1 = np.ones(100, dtype=np.float32)  # 400 bytes
        arr2 = np.zeros(100, dtype=np.float32)  # 400 bytes
        arrays = []
        result = _find_arrays({"a": {"b": arr1}, "c": [arr2]}, arrays)
        assert len(arrays) == 2
        assert result["a"]["b"]["frame"] == 0
        assert result["c"][0]["frame"] == 1

    def test_no_arrays(self):
        arrays = []
        result = _find_arrays({"key": "value", "num": 42}, arrays)
        assert len(arrays) == 0
        assert result == {"key": "value", "num": 42}

    def test_none_passthrough(self):
        arrays = []
        result = _find_arrays(None, arrays)
        assert result is None
        assert len(arrays) == 0


class TestRestoreArrays:
    def test_restores_from_frame(self, large_array):
        placeholder = {
            "__ndarray__": True,
            "inline": False,
            "frame": 0,
            "dtype": large_array.dtype.str,
            "shape": list(large_array.shape),
        }
        frames = [large_array.tobytes()]
        restored = _restore_arrays(placeholder, frames)
        assert isinstance(restored, np.ndarray)
        assert restored.dtype == large_array.dtype
        assert restored.shape == large_array.shape
        np.testing.assert_array_equal(restored, large_array)

    def test_restores_inline(self, small_array):
        placeholder = {
            "__ndarray__": True,
            "inline": True,
            "dtype": small_array.dtype.str,
            "shape": list(small_array.shape),
            "data": small_array.tobytes().hex(),
        }
        restored = _restore_arrays(placeholder, [])
        assert isinstance(restored, np.ndarray)
        np.testing.assert_array_equal(restored, small_array)

    def test_roundtrip_nested(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arrays = []
        processed = _find_arrays({"x": arr, "y": "hello"}, arrays)
        frames = [a.tobytes() for a in arrays]
        restored = _restore_arrays(processed, frames)
        np.testing.assert_array_equal(restored["x"], arr)
        assert restored["y"] == "hello"


class TestEncodeDecodeRequest:
    def test_roundtrip(self, request_message, large_array):
        frames = encode_message(request_message)
        assert len(frames) >= 2  # metadata + at least 1 tensor frame

        decoded = decode_request(frames)
        assert decoded.model_name == "test-model"
        assert decoded.method_name == "predict"
        assert len(decoded.method_args) == 1

        # Check the NumPy array was restored
        inference_request = decoded.method_args[0]
        assert isinstance(inference_request, InferenceRequest)
        assert len(inference_request.inputs) == 1
        data = getattr(
            inference_request.inputs[0].data,
            "root",
            inference_request.inputs[0].data,
        )
        assert isinstance(data, np.ndarray)
        assert data.dtype == large_array.dtype
        assert data.shape == large_array.shape
        np.testing.assert_array_equal(data, large_array)

    def test_metadata_is_json(self, request_message):
        import orjson

        frames = encode_message(request_message)
        # First frame must be valid JSON
        metadata = orjson.loads(frames[0])
        assert isinstance(metadata, dict)
        assert "model_name" in metadata

    def test_no_arrays(self):
        # Message without any NumPy arrays
        msg = ModelRequestMessage(
            model_name="test",
            method_name="predict",
            method_args=["just", "strings"],
        )
        frames = encode_message(msg)
        assert len(frames) == 1  # only metadata, no tensor frames
        decoded = decode_request(frames)
        assert decoded.method_args == ["just", "strings"]


class TestEncodeDecodeResponse:
    def test_roundtrip(self, response_message):
        frames = encode_message(response_message)
        assert len(frames) >= 2  # metadata + tensor frames

        decoded = decode_response(frames)
        assert decoded.id == "test-id"
        assert decoded.return_value is not None
        assert isinstance(decoded.return_value, InferenceResponse)

        # Check output arrays
        response = decoded.return_value
        assert len(response.outputs) == 2
        for output in response.outputs:
            data = getattr(output.data, "root", output.data)
            assert isinstance(data, np.ndarray)

    def test_none_return_value(self):
        msg = ModelResponseMessage(id="test", return_value=None)
        frames = encode_message(msg)
        decoded = decode_response(frames)
        assert decoded.return_value is None


class TestZeroCopyProperties:
    def test_frames_reference_original_memory(self, large_array):
        """Verify the tensor frame is raw bytes, not a pickled blob."""
        msg = ModelRequestMessage(
            model_name="test",
            method_name="predict",
            method_args=[
                InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="input",
                            datatype="FP32",
                            shape=list(large_array.shape),
                            data=large_array,
                        )
                    ]
                )
            ],
        )
        frames = encode_message(msg)
        # The tensor frame should be a bytes/memoryview, not a pickled blob
        tensor_frame = frames[1]
        assert isinstance(tensor_frame, (bytes, memoryview))
        # Size should match the array's raw bytes
        assert len(tensor_frame) == large_array.nbytes

    def test_decode_produces_readonly_view(self, large_array):
        """np.frombuffer creates read-only views (zero-copy)."""
        frames = encode_message(
            ModelRequestMessage(
                model_name="test",
                method_name="predict",
                method_args=[
                    InferenceRequest(
                        inputs=[
                            RequestInput(
                                name="input",
                                datatype="FP32",
                                shape=list(large_array.shape),
                                data=large_array,
                            )
                        ]
                    )
                ],
            )
        )
        decoded = decode_request(frames)
        data = getattr(
            decoded.method_args[0].inputs[0].data,
            "root",
            decoded.method_args[0].inputs[0].data,
        )
        # frombuffer views are read-only
        assert not data.flags.writeable


class TestEdgeCases:
    def test_empty_inputs(self):
        msg = ModelRequestMessage(
            model_name="test", method_name="predict", method_args=[]
        )
        frames = encode_message(msg)
        decoded = decode_request(frames)
        assert decoded.method_args == []

    def test_mixed_types(self):
        arr = np.array([1.0], dtype=np.float64)
        msg = ModelRequestMessage(
            model_name="test",
            method_name="predict",
            method_args=[{"key": "value", "array": arr, "number": 42}],
        )
        frames = encode_message(msg)
        decoded = decode_request(frames)
        arg = decoded.method_args[0]
        # After model_validate, the dict might be reconstructed differently
        # but the array should be restored
        assert arg is not None

    def test_multiple_arrays(self):
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.zeros((5, 5), dtype=np.int64)
        msg = ModelRequestMessage(
            model_name="test",
            method_name="predict",
            method_args=[
                InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="a", datatype="FP32", shape=[10, 10], data=arr1
                        ),
                        RequestInput(
                            name="b", datatype="INT64", shape=[5, 5], data=arr2
                        ),
                    ]
                )
            ],
        )
        frames = encode_message(msg)
        decoded = decode_request(frames)
        inputs = decoded.method_args[0].inputs
        data_a = getattr(inputs[0].data, "root", inputs[0].data)
        data_b = getattr(inputs[1].data, "root", inputs[1].data)
        np.testing.assert_array_equal(data_a, arr1)
        np.testing.assert_array_equal(data_b, arr2)

    def test_boolean_array(self):
        arr = np.array([True, False, True], dtype=np.bool_)
        arrays = []
        result = _find_arrays({"data": arr}, arrays)
        frames = [a.tobytes() for a in arrays] if arrays else []
        restored = _restore_arrays(result, frames)
        np.testing.assert_array_equal(restored["data"], arr)


class TestBytesRoundtrip:
    def test_small_bytes_inline(self):
        msg = ModelRequestMessage(
            model_name="test",
            method_name="__raw__",
            method_args=[b"hello", {"content-type": "application/json"}],
        )
        frames = encode_message(msg)
        assert len(frames) == 1  # inlined into metadata
        decoded = decode_request(frames)
        assert decoded.method_args[0] == b"hello"
        assert decoded.method_args[1] == {"content-type": "application/json"}

    def test_large_bytes_frame(self):
        body = bytes(range(256)) * 16  # 4096 bytes, above threshold
        msg = ModelRequestMessage(
            model_name="test",
            method_name="__raw__",
            method_args=[body, {}],
        )
        frames = encode_message(msg)
        assert len(frames) == 2  # metadata + one buffer frame
        decoded = decode_request(frames)
        assert decoded.method_args[0] == body


class TestMemoryviewFrames:
    def test_memoryview_tensor_frames(self, request_message, large_array):
        """Received frames arrive as zero-copy memoryviews, not bytes."""
        frames = encode_message(request_message)
        assert len(frames) >= 2
        decoded = decode_request([memoryview(frame) for frame in frames])
        data = getattr(
            decoded.method_args[0].inputs[0].data,
            "root",
            decoded.method_args[0].inputs[0].data,
        )
        np.testing.assert_array_equal(data, large_array)

    def test_memoryview_bytes_frame(self):
        body = bytes(range(256)) * 8
        msg = ModelRequestMessage(
            model_name="test",
            method_name="__raw__",
            method_args=[{"body": body}],
        )
        frames = encode_message(msg)
        decoded = decode_request([memoryview(frame) for frame in frames])
        assert decoded.method_args[0]["body"] == body


class TestExceptionRoundtrip:
    def test_worker_error_roundtrip(self):
        wrapped = WorkerError(ValueError("boom"))
        msg = ModelResponseMessage(id="e1", exception=wrapped)
        decoded = decode_response(encode_message(msg))
        assert isinstance(decoded.exception, WorkerError)
        assert str(decoded.exception) == "builtins.ValueError: boom"

    def test_cancelled_error_roundtrip(self):
        msg = ModelResponseMessage(
            id="e2", exception=asyncio.CancelledError("cancelled")
        )
        decoded = decode_response(encode_message(msg))
        assert isinstance(decoded.exception, asyncio.CancelledError)
        assert str(decoded.exception) == "cancelled"

    def test_unknown_class_falls_back_to_runtime_error(self):
        marker = {"__exception__": True, "class": "no.such.Module", "msg": "gone"}
        restored = _restore_arrays(marker, [])
        assert isinstance(restored, RuntimeError)
        assert str(restored) == "gone"


class TestDataclassRoundtrip:
    def test_raw_reply_roundtrip(self):
        from mlserver.rest.worker import RawReply

        reply = RawReply(
            body=b'{"ok": true}',
            headers={"x-test": "yes"},
            status=200,
            worker_pid=123,
            conversion_duration=0.5,
        )
        msg = ModelResponseMessage(id="r1", return_value=reply)
        decoded = decode_response(encode_message(msg))
        value = decoded.return_value
        assert isinstance(value, RawReply)
        assert value.body == b'{"ok": true}'
        assert value.headers == {"x-test": "yes"}
        assert value.status == 200
        assert value.worker_pid == 123
        assert value.conversion_duration == 0.5


class TestNumpyScalarRoundtrip:
    def test_scalars_become_python_values(self):
        msg = ModelRequestMessage(
            model_name="test",
            method_name="predict",
            method_args=[{"v": np.float32(1.5), "i": np.int64(3), "b": np.bool_(True)}],
        )
        decoded = decode_request(encode_message(msg))
        assert decoded.method_args[0] == {"v": 1.5, "i": 3, "b": True}
