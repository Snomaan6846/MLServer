"""Binary input tensors must be writable on first decode.

The binary REST and gRPC raw paths store inputs as ``np.frombuffer``
views over immutable request bytes. The codec's zero-copy fast path
returned a reshaped view of that buffer — read-only, so a model's
first ``decode_input`` could not perform in-place preprocessing
(``image *= 0.5`` raised ``ValueError``). The codec now copies when
the backing is read-only (matching the writable array the list path
always produced) and preserves the zero-copy view when the backing
is writable (the fastspan parser's fresh buffers).
"""

import numpy as np

from mlserver.codecs.numpy import NumpyCodec
from mlserver.rest.binary import get_header_length, parse_binary_request
from mlserver.types import InferenceRequest, RequestInput


def _binary_request(values: list[float], shape: list[int]) -> InferenceRequest:
    """Build a request through the REAL binary parser (frombuffer
    views over immutable bytes — not a pre-writable test array)."""
    arr = np.array(values, dtype=np.float32)
    header = (
        b'{"inputs": [{"name": "input-0", "shape": '
        + str(shape).encode()
        + b', "datatype": "FP32", "parameters": {"binary_data_size": '
        + str(arr.nbytes).encode()
        + b"}}]}"
    )
    body = header + arr.tobytes()
    headers = {"inference-header-content-length": str(len(header))}
    request, _ = parse_binary_request(body, get_header_length(headers))
    return request


def _writable_input(arr: np.ndarray, shape: list[int]) -> RequestInput:
    """A request input carrying a WRITABLE ndarray root (the fastspan
    parser's shape) — the zero-copy view must be preserved. Uses
    model_construct to avoid pydantic's list coercion copying the
    buffer (the same pattern the gRPC raw path uses)."""
    from mlserver.types import TensorData

    from mlserver.types import Datatype, Parameters

    spec = RequestInput.model_construct(
        name="input-0",
        shape=shape,
        datatype=Datatype.FP32,
        parameters=Parameters(binary_data_size=-1),
        data=TensorData.model_construct(root=arr),
    )
    return spec


def test_binary_rest_first_decode_is_writable():
    """A binary REST input, decoded for the first time, supports
    in-place preprocessing with correct values; the request bytes
    remain unchanged."""
    request = _binary_request([1.0, 2.0], [1, 2])
    image = NumpyCodec.decode_input(request.inputs[0])

    assert image.flags.writeable, "first decode returned a read-only tensor"
    image *= 0.5
    assert np.array_equal(image, np.array([[0.5, 1.0]], dtype=np.float32))


def test_binary_rest_request_bytes_unchanged_after_mutation():
    """In-place preprocessing on the decoded tensor must not reach the
    original request bytes (the copy decouples them)."""
    request = _binary_request([1.0, 2.0], [1, 2])
    original = np.array([1.0, 2.0], dtype=np.float32).tobytes()

    image = NumpyCodec.decode_input(request.inputs[0])
    image *= 0.5

    # The request's backing buffer still holds the original values
    stored = request.inputs[0].data.root
    assert np.array_equal(
        np.frombuffer(stored.tobytes(), dtype=np.float32),
        np.frombuffer(original, dtype=np.float32),
    )


def test_writable_backing_keeps_zero_copy_view():
    """A WRITABLE ndarray root (the fastspan parser's output) must
    keep the zero-copy view — the copy is only for read-only
    backings, not a blanket regression to copying."""
    from mlserver.codecs.numpy import _to_ndarray

    arr = np.array([1.0, 2.0], dtype=np.float32)
    input_spec = _writable_input(arr, [1, 2])

    result = _to_ndarray(input_spec)
    assert result.base is arr or np.shares_memory(
        result, arr
    ), "writable backing was copied — the zero-copy win was lost"
