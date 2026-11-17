import pytest
import numpy as np

from typing import Any

from mlserver.codecs.numpy import (
    NumpyCodec,
    convert_nan,
    set_ndarray_response_pipeline,
    to_datatype,
    to_dtype,
)
from mlserver.types import RequestInput, ResponseOutput, Parameters, Datatype


@pytest.mark.parametrize(
    "payload, expected",
    [(np.array([1, 2, 3]), True), (np.array(["foo", "bar"]), True), ([1, 2, 3], False)],
)
def test_can_encode(payload: Any, expected: bool):
    assert NumpyCodec.can_encode(payload) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(float("nan"), None), (np.float32(np.nan), None), (1.5, 1.5), (None, None)],
)
def test_convert_nan_handles_scalar_values(value, expected):
    result = convert_nan(value)
    if isinstance(value, float) and np.isnan(value):
        assert result is None
    else:
        assert result == expected


def test_convert_nan_preserves_non_scalar_values():
    value = [1.0, np.nan]

    assert convert_nan(value) is value


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            np.array([1, 2, 3]),
            ResponseOutput(
                name="foo",
                shape=[3, 1],
                data=np.array([1, 2, 3]),
                datatype="INT64",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ResponseOutput(
                name="foo",
                shape=[2, 2],
                data=np.array([1, 2, 3, 4]),
                datatype="INT64",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            ResponseOutput(
                name="foo",
                shape=[2, 2],
                data=np.array([1, 2, 3, 4], dtype=np.int32),
                datatype="INT32",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([1.0, 2.0]),
            ResponseOutput(
                name="foo",
                shape=[2, 1],
                data=np.array([1.0, 2.0]),
                datatype="FP64",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
            ResponseOutput(
                name="foo",
                shape=[2, 1],
                data=[b"\x01\x02"],
                datatype="BYTES",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array(["foo", "bar"], dtype=str),
            ResponseOutput(
                name="foo",
                shape=[2, 1],
                data=[b"foo", b"bar"],
                datatype="BYTES",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([None, "bar"]),
            ResponseOutput(
                name="foo",
                shape=[2, 1],
                data=[None, "bar"],
                datatype="BYTES",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
        (
            np.array([2.3, 3.4, np.NaN]),
            ResponseOutput(
                name="foo",
                shape=[3, 1],
                data=np.array([2.3, 3.4, np.nan]),
                datatype="FP64",
                parameters=Parameters(content_type=NumpyCodec.ContentType),
            ),
        ),
    ],
)
def test_encode_output(payload: np.ndarray, expected: ResponseOutput):
    set_ndarray_response_pipeline(True)
    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    assert response_output.name == expected.name
    assert response_output.shape == expected.shape
    assert response_output.datatype == expected.datatype
    assert response_output.parameters == expected.parameters
    if isinstance(expected.data.root, np.ndarray):
        # Non-BYTES tensors keep their NumPy representation (flat view)
        assert isinstance(response_output.data.root, np.ndarray)
        np.testing.assert_array_equal(response_output.data.root, expected.data.root)
        assert response_output.data.root.dtype == expected.data.root.dtype
    else:
        assert response_output.data.root == expected.data.root


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_encode_output_keeps_flat_ndarray(dtype):
    """Non-BYTES payloads stay ndarrays: flat views, no element churn."""
    set_ndarray_response_pipeline(True)

    payload = np.zeros((2, 3), dtype=dtype)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    root = response_output.data.root
    assert isinstance(root, np.ndarray)
    assert root.ndim == 1
    assert len(root) == payload.size
    assert root.dtype == payload.dtype
    # The encoded root is a detached snapshot of the payload: it owns
    # its buffer and does not alias the model's output.
    assert root.base is None
    assert not np.shares_memory(root, payload)


def test_encode_output_masked_array_keeps_legacy_list_representation():
    """Masked entries must serialise as None (the legacy wire format).

    An ndarray root would carry the MaskedArray subclass through to
    the wire encoders, whose tobytes() substitutes the array's
    fill_value (7.0 here) instead of a missing-value marker.
    """
    payload = np.ma.array(
        [42.0, 2.0], dtype="float32", mask=[True, False], fill_value=7.0
    )

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    assert response_output.data.root == [None, 2.0]
    assert not isinstance(response_output.data.root, np.ndarray)


def test_encode_output_masked_array_without_masked_entries():
    """A MaskedArray with no masked entries still routes through the
    list path (type-based): its values pass through unchanged."""
    payload = np.ma.array(
        [1.0, 2.0], dtype="float32", mask=[False, False], fill_value=7.0
    )

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    assert response_output.data.root == [1.0, 2.0]


@pytest.mark.parametrize(
    "dtype, canonical_nan_bits",
    [
        (np.float16, 0x7E00),
        (np.float32, 0x7FC00000),
        (np.float64, 0x7FF8000000000000),
    ],
)
def test_encode_output_canonicalises_nan_payloads(dtype, canonical_nan_bits):
    """NaN payloads canonicalise to the standard quiet NaN (like the
    historic list pipeline's tolist -> None -> float("nan") round-trip),
    and the payload's own buffer is never mutated."""
    set_ndarray_response_pipeline(True)

    payload = np.array([1.5, 2.5], dtype=dtype)
    uint_dtype = np.dtype(f"u{payload.itemsize}")
    raw = payload.view(uint_dtype)
    raw[1] = canonical_nan_bits | 0x1  # quiet NaN with a payload bit set
    original_bytes = payload.tobytes()

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    root = response_output.data.root
    assert isinstance(root, np.ndarray)
    assert root.view(uint_dtype)[1] == canonical_nan_bits
    assert root[0] == 1.5
    # The model's own buffer keeps its original bits
    assert payload.tobytes() == original_bytes


def test_encode_output_nan_free_root_is_snapshot():
    """NaN-free payloads still get a detached root: encoding always
    returns a stable snapshot, whether or not normalisation is needed."""
    set_ndarray_response_pipeline(True)

    payload = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    root = response_output.data.root
    assert root.base is None
    # Mutating the payload after encoding must not change the response
    payload[0, 0] = 99.0
    assert root[0] == 1.5


@pytest.mark.parametrize(
    "request_input, expected",
    [
        (
            RequestInput(name="foo", shape=[3], data=[1, 2, 3], datatype="INT32"),
            np.array([1, 2, 3]),
        ),
        (
            RequestInput(name="foo", shape=[2, 2], data=[1, 2, 3, 4], datatype="INT32"),
            np.array([[1, 2], [3, 4]]),
        ),
        (
            RequestInput(name="foo", shape=[2], data=[1, 2], datatype="FP32"),
            np.array([1.0, 2.0]),
        ),
        (
            RequestInput(
                name="foo", shape=[2, 1], data=[b"\x01", b"\x02"], datatype="BYTES"
            ),
            np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
        ),
        (
            RequestInput(name="foo", shape=[2], data=["foo", "bar"], datatype="BYTES"),
            np.array(["foo", "bar"], dtype=str),
        ),
        (
            RequestInput(name="foo", shape=[3], data=[1, 2, None], datatype="FP16"),
            np.array([1, 2, np.NaN], dtype="float16"),
        ),
    ],
)
def test_decode_input(request_input: RequestInput, expected: np.ndarray):
    decoded = NumpyCodec.decode_input(request_input)
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize(
    "request_input",
    [
        RequestInput(name="foo", shape=[3], data=[1, 2, 3], datatype="INT32"),
        RequestInput(name="foo", shape=[2, 2], data=[1, 2, 3, 4], datatype="INT32"),
        RequestInput(name="foo", shape=[2], data=[1, 2], datatype="FP32"),
        RequestInput(name="foo", shape=[2, 1], data=[b"\x01\x02"], datatype="BYTES"),
        RequestInput(name="foo", shape=[2], data=["foo", "bar"], datatype="BYTES"),
    ],
)
def test_codec_idempotent(request_input: RequestInput):
    decoded = NumpyCodec.decode_input(request_input)
    response_output = NumpyCodec.encode_output(name="foo", payload=decoded)

    request_input_result = NumpyCodec.encode_input(name="foo", payload=decoded)
    assert response_output.datatype == request_input_result.datatype
    assert response_output.shape == request_input_result.shape
    if isinstance(response_output.data.root, np.ndarray) or isinstance(
        request_input_result.data.root, np.ndarray
    ):
        # Non-BYTES tensors keep their NumPy representation on both sides
        np.testing.assert_array_equal(
            response_output.data.root, request_input_result.data.root
        )
    else:
        assert response_output.data == request_input_result.data
    assert request_input_result.parameters is not None
    assert request_input_result.parameters.content_type == NumpyCodec.ContentType


@pytest.mark.parametrize(
    "payload, expected",
    [
        (np.array([1, 2, 3]), np.array([[1], [2], [3]])),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
        (np.array([1.0, 2.0]), np.array([[1.0], [2.0]])),
        (
            np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
            np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
        ),
    ],
)
def test_decode_output(payload, expected):
    # Note that `decode(encode(x)) != x`, because of shape changes to make
    # the `[N] == [N, 1]` explicit
    response_output = NumpyCodec.encode_output(name="foo", payload=payload)
    output_response_decoded = NumpyCodec.decode_output(response_output)
    np.testing.assert_array_equal(output_response_decoded, expected)


@pytest.mark.parametrize(
    "dtype, datatype",
    [
        (np.bool_, Datatype.BOOL),
        (np.uint8, Datatype.UINT8),
        (np.uint16, Datatype.UINT16),
        (np.uint32, Datatype.UINT32),
        (np.uint64, Datatype.UINT64),
        (np.int8, Datatype.INT8),
        (np.int16, Datatype.INT16),
        (np.int32, Datatype.INT32),
        (np.int64, Datatype.INT64),
        (np.float16, Datatype.FP16),
        (np.float32, Datatype.FP32),
        (np.float64, Datatype.FP64),
        (np.byte, Datatype.INT8),
        (bytes, Datatype.BYTES),
        (str, Datatype.BYTES),
    ],
)
def test_to_datatype(dtype, datatype):
    dtype = np.dtype(dtype)

    obtained_datatype = to_datatype(dtype)
    assert datatype == obtained_datatype


@pytest.mark.parametrize(
    "data, datatype, shape",
    [
        # Fast path with a None element (NaN, matching np.array)
        ([1.5, None, 2.0], "FP32", [3]),
        # Nested list whose outer length matches the declared count: fromiter
        # cannot consume it and must fall back
        ([[1.0, 2.0], [3.0, 4.0]], "FP32", [2]),
        # Nested BOOL whose outer length matches the declared count:
        # fromiter would silently convert the inner list to True
        # (bool([False]) is True) — must fall back to preserve [[False]]
        ([[False]], "BOOL", [1]),
        # Mixed nested BOOL — the nested item hides behind a leading
        # scalar: only a full scan catches it (fromiter would convert
        # [2] to True, silently changing the wire values).
        ([1, [2]], "BOOL", [2]),
        # Flat numeric lists whose length matches the declared shape:
        # these ENTER the fromiter fast path (the cases above all fall
        # through to the generic path).
        ([1.0, 2.0, 3.0, 4.0], "FP32", [4]),
        ([1, 2, 3], "INT32", [3]),
        ([True, False], "BOOL", [2]),
        # Declared shape does not match the flat length: skipped, generic path
        ([1.0, 2.0, 3.0], "FP32", [4]),
        # Variable-size dtype (BYTES string fallthrough): fromiter raises,
        # falls back to np.array's auto-width conversion
        (["hello", "world"], "BYTES", [2]),
        # Empty tensor
        ([], "FP32", [0]),
    ],
)
def test_decode_input_flat_list_fast_path_fallbacks(
    data: list, datatype: str, shape: list[int]
):
    """The fromiter fast path must be bit-identical to the generic np.array
    path for everything it accepts, and fall back to it (same result or
    same error) for everything it cannot consume."""
    request_input = RequestInput(name="foo", shape=shape, datatype=datatype, data=data)

    try:
        decoded = NumpyCodec.decode_input(request_input)
        result: tuple[str, Any, str] = ("ok", decoded.tobytes(), str(decoded.dtype))
    except Exception as error:
        result = ("raise", type(error).__name__, "")

    dtype = to_dtype(request_input)
    try:
        reference = np.array(data, dtype=dtype).reshape(shape)
        expected: tuple[str, Any, str] = (
            "ok",
            reference.tobytes(),
            str(reference.dtype),
        )
    except Exception as error:
        expected = ("raise", type(error).__name__, "")

    assert result == expected


def test_encode_output_does_not_mutate_source_buffer():
    """The model's own buffer is never modified by encoding — the
    encoded root is a detached copy, so normalisation cannot write
    through to the model's output."""
    set_ndarray_response_pipeline(True)
    # Use a non-canonical NaN (payload bit set) so the test verifies
    # normalisation actually happens — a canonical NaN in the source
    # would pass even if normalisation were removed or wrote the same
    # bits through the source buffer.
    payload = np.array([1.5, np.nan, 3.0], dtype=np.float32)
    # Set a payload bit on the NaN to make it non-canonical
    payload.view(np.uint32)[1] = 0x7FC00001
    original_bits = payload.tobytes()

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    # Source bits unchanged (the non-canonical NaN is preserved)
    assert payload.tobytes() == original_bits
    assert payload.view(np.uint32)[1] == 0x7FC00001
    # Encoded root has the canonical quiet NaN (payload bit cleared)
    assert response_output.data.root.view(np.uint32)[1] == 0x7FC00000


def test_encode_output_root_detached_from_source():
    """The encoded root never shares memory with the model's output —
    the response is an immutable snapshot taken at encode time."""
    set_ndarray_response_pipeline(True)
    payload = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    # The root does not share memory with the source (detached snapshot)
    root = response_output.data.root
    assert not np.shares_memory(root, payload)
    assert root.base is None


def test_encode_output_reused_buffer_isolated():
    """A model that reuses its output buffer across predict() calls must
    not corrupt in-flight responses: each encoding captures the state of
    the buffer at encode time."""
    buffer = np.zeros(4, dtype=np.float32)

    # First call
    output_1 = NumpyCodec.encode_output(name="foo", payload=buffer)
    # Model mutates the buffer between calls (the reuse pattern)
    buffer[0] = 99.0
    # Second call
    output_2 = NumpyCodec.encode_output(name="foo", payload=buffer)

    # The second encoding sees the mutated value
    assert output_2.data.root[0] == 99.0
    # The first response is unaffected — it captured the original state
    assert output_1.data.root[0] == 0.0


def test_decode_input_invalid_datatype_raises():
    """An unknown datatype string raises a pydantic ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestInput(name="foo", shape=[2], datatype="NOT_A_TYPE", data=[1, 2])


def test_encode_output_zero_dimension_array():
    """A 0-d array produces a single-element output (shape [1, 1])."""
    payload = np.float32(3.14)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    assert response_output.datatype == "FP32"
    assert response_output.shape == [1]
    assert response_output.data.root[0] == np.float32(3.14)


def test_encode_output_default_is_list_payload():
    """Default (single-frontend) topology: master's data flow — the
    codec materialises the Python element list, IPC carries lists, and
    every wire boundary takes its legacy path."""
    payload = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    root = response_output.data.root
    assert isinstance(root, list)
    assert root == [1.5, 2.5, 3.5, 4.5]


def test_encode_output_multi_frontend_keeps_ndarray():
    """Multi-frontend opt-in: the ndarray-native pipeline stays on."""
    set_ndarray_response_pipeline(True)
    payload = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    response_output = NumpyCodec.encode_output(name="foo", payload=payload)

    root = response_output.data.root
    assert isinstance(root, np.ndarray)
    np.testing.assert_array_equal(
        root, np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int64,
        np.int32,
        np.bool_,
    ],
)
def test_encode_output_gate_values_identical(dtype):
    """Both gate states produce identical element values — the gate
    changes payload representation and placement, never values."""
    payload = np.array([[1, 2], [3, 4]], dtype=dtype)

    set_ndarray_response_pipeline(True)
    nd_root = NumpyCodec.encode_output(name="foo", payload=payload).data.root
    set_ndarray_response_pipeline(False)
    list_root = NumpyCodec.encode_output(name="foo", payload=payload).data.root

    assert isinstance(nd_root, np.ndarray)
    assert isinstance(list_root, list)
    assert nd_root.tolist() == list_root


def test_encode_output_default_path_nan_passes_through():
    """Default (fw=1) path: NaN stays a float in the materialised list.

    orjson renders it as null on REST JSON (byte-identical to master's
    convert_nan -> None -> null wire form) and gRPC typed contents
    serialise it natively — both pinned by wire tests elsewhere.
    """
    payload = np.array([1.5, np.nan, 3.0], dtype=np.float32)

    root = NumpyCodec.encode_output(name="foo", payload=payload).data.root

    assert isinstance(root, list)
    assert root[0] == 1.5
    assert root[2] == 3.0
    assert isinstance(root[1], float) and np.isnan(root[1])


def test_encode_output_default_path_nan_free_values_untouched():
    """Negative: NaN-free payloads carry plain floats, no mapping."""
    payload = np.array([1.5, 2.5], dtype=np.float32)

    root = NumpyCodec.encode_output(name="foo", payload=payload).data.root

    assert root == [1.5, 2.5]


def test_encode_output_default_path_all_nan():
    """Edge: every entry NaN."""
    payload = np.array([np.nan, np.nan], dtype=np.float64)

    root = NumpyCodec.encode_output(name="foo", payload=payload).data.root

    assert len(root) == 2
    assert all(np.isnan(value) for value in root)


def test_encode_output_default_path_nan_rest_json_renders_null():
    """Wire pin: the default path's NaN list renders as null —
    byte-identical to master's wire form."""
    from mlserver.codecs.json import encode_to_json_bytes

    payload = np.array([1.5, np.nan], dtype=np.float32)
    root = NumpyCodec.encode_output(name="foo", payload=payload).data.root

    assert encode_to_json_bytes({"data": root}) == b'{"data":[1.5,null]}'
