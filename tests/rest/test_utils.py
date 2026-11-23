from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from fastapi import status

from mlserver.codecs.json import encode_to_json_bytes
from mlserver.rest.responses import ServerSentEvent
from mlserver.rest.utils import encode_inference_response_bytes, to_status_code
from mlserver.types import InferenceResponse, ResponseOutput, TensorData


@pytest.mark.parametrize(
    "flag,expected", [(True, status.HTTP_200_OK), (False, status.HTTP_400_BAD_REQUEST)]
)
def test_to_status_code(flag: bool, expected: int):
    status_code = to_status_code(flag)
    assert status_code == expected


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)], id="utc"),
        pytest.param(
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))],
            id="offset",
        ),
        pytest.param([datetime(2023, 1, 1, 12, 0, 0)], id="naive"),
        pytest.param([Decimal("3.14")], id="decimal"),
        pytest.param([timedelta(seconds=2)], id="timedelta"),
        pytest.param([b"hello"], id="bytes"),
        pytest.param([uuid4()], id="uuid"),
        pytest.param([1, 2.5, "x", True, None], id="plain-scalars"),
        pytest.param(
            {"nested": [datetime(2023, 1, 1, tzinfo=timezone.utc)]}, id="nested"
        ),
    ],
)
def test_encode_inference_response_bytes_matches_schema_json_mode(data):
    """Non-ndarray tensor data renders byte-identically to the
    schema's JSON mode: orjson renders some natively-serialisable
    types differently (a UTC datetime becomes +00:00 instead of Z)
    with no exception to trip the fallback, so the conversion is
    fused into the dump itself."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o", datatype="BYTES", shape=[1], data=TensorData(root=data)
            )
        ],
    )
    fast = encode_inference_response_bytes(response)
    ground_truth = encode_to_json_bytes(
        response.model_dump(mode="json", exclude_unset=True)
    )
    assert fast == ground_truth


def test_encode_inference_response_bytes_utc_datetime_renders_z():
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o",
                datatype="BYTES",
                shape=[1],
                data=TensorData(
                    root=[datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)]
                ),
            )
        ],
    )
    assert b'"2023-01-01T12:00:00Z"' in encode_inference_response_bytes(response)


def test_encode_inference_response_bytes_mixed_ndarray_and_datetime():
    """A response with BOTH an ndarray output (the native buffer
    path) and a datetime BYTES output: the ndarray keeps the fast
    path, the datetime still renders with Z — byte-identical to the
    schema's JSON mode."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="arr",
                datatype="FP32",
                shape=[4],
                data=TensorData(root=np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")),
            ),
            ResponseOutput(
                name="ts",
                datatype="BYTES",
                shape=[1],
                data=TensorData(
                    root=[datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)]
                ),
            ),
        ],
    )
    fast = encode_inference_response_bytes(response)
    ground_truth = encode_to_json_bytes(
        response.model_dump(mode="json", exclude_unset=True)
    )
    assert fast == ground_truth
    assert b'"2023-01-01T12:00:00Z"' in fast


@pytest.mark.parametrize(
    "data",
    [pytest.param([bytes([0xFF, 0xFE])], id="binary-bytes")],
)
def test_encode_inference_response_bytes_unserialisable_data_raises(data):
    """Values neither the schema nor orjson can render raise (a 500,
    as on the base): parity, not a regression to paper over."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o", datatype="BYTES", shape=[1], data=TensorData(root=data)
            )
        ],
    )
    with pytest.raises(Exception):
        encode_inference_response_bytes(response)


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param([np.float32(0.1)], b"[0.10000000149011612]", id="np-float32"),
        pytest.param([np.float64(0.1)], b"[0.1]", id="np-float64"),
        pytest.param([np.int32(7)], b"[7]", id="np-int32"),
        pytest.param([np.uint8(3)], b"[3]", id="np-uint8"),
        pytest.param([np.bool_(True)], b"[true]", id="np-bool"),
        pytest.param(np.float32(0.5), b"0.5", id="np-scalar-root"),
    ],
)
def test_encode_inference_response_bytes_numpy_scalars_coerce(data, expected):
    """Loose numpy scalars coerce to python scalars at the wire
    boundary — the KServe reference implementation's own semantics
    (set_data_from_numpy does ``val.item()`` per element), and the
    KServe v2 protocol explicitly sanctions server-side coercion
    ("The server can decide to coerce each element to the required
    type or return an error"). The base's response-model path 500s
    on these (pydantic cannot serialize a loose numpy scalar) — a
    documented deliberate deviation: spec-sanctioned coercion,
    matching the reference implementation. The widened python float
    renders exactly what the tolist'd ndarray path renders."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o", datatype="BYTES", shape=[1], data=TensorData(root=data)
            )
        ],
    )
    encoded = encode_inference_response_bytes(response)
    assert expected in encoded


def test_encode_inference_response_bytes_numpy_scalar_matches_ndarray():
    """The SAME float32 value renders identically whether it travels
    as a loose scalar in a list or as an ndarray root: the coerced
    python float is the float64-widened form (the byte-preserving
    bridge), never orjson's short float32 repr — that is the
    wire-format change the rest_json_compact_floats opt-in gates,
    and it must not leak through the loose-scalar path."""
    loose = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o",
                datatype="BYTES",
                shape=[1],
                data=TensorData(root=[np.float32(0.1)]),
            )
        ],
    )
    ndarray_resp = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o",
                datatype="FP32",
                shape=[1],
                data=TensorData(root=np.array([0.1], dtype="float32")),
            )
        ],
    )
    loose_data = encode_inference_response_bytes(loose).split(b'"data":')[1]
    ndarray_data = encode_inference_response_bytes(ndarray_resp).split(b'"data":')[1]
    assert loose_data == ndarray_data


def test_encode_inference_response_bytes_numpy_scalar_does_not_abort_conversion():
    """A loose numpy scalar must not strand the exotics next to it:
    pydantic's whole-tree conversion aborts on the first
    unknown-type element, which would leave a UTC datetime rendering
    orjson's +00:00 (the exact divergence the conversion exists to
    prevent). The numpy-coercion retry keeps the conversion alive."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o",
                datatype="BYTES",
                shape=[3],
                data=TensorData(
                    root=[
                        np.float32(0.1),
                        datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        Decimal("3.14"),
                    ]
                ),
            )
        ],
    )
    encoded = encode_inference_response_bytes(response)
    assert b'"2023-01-01T12:00:00Z"' in encoded
    assert b'"3.14"' in encoded
    assert b"0.10000000149011612" in encoded


def test_encode_inference_response_bytes_sse_dump_keeps_python_mode():
    """The SSE encoding (ServerSentEvent) dumps WITHOUT the wire
    context: python-mode objects reach orjson untouched, keeping the
    base's per-endpoint rendering (a UTC datetime stays +00:00 on
    /infer_stream — the base's own SSE behaviour)."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="o",
                datatype="BYTES",
                shape=[1],
                data=TensorData(
                    root=[datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)]
                ),
            )
        ],
    )
    sse = ServerSentEvent(response).encode()
    assert b'"2023-01-01T12:00:00+00:00"' in sse


def test_encode_inference_response_bytes_object_array_exotics():
    """An OBJECT-dtype ndarray serializes to a LIST (the native
    buffer path does not apply): its exotic elements must take the
    JSON-mode conversion — orjson renders a UTC datetime as +00:00
    where the schema's JSON mode renders Z. Eligibility is decided
    from the SERIALIZED value, not the original root."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="obj",
                datatype="BYTES",
                shape=[1],
                data=TensorData(
                    root=np.array(
                        [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
                        dtype=object,
                    )
                ),
            )
        ],
    )
    encoded = encode_inference_response_bytes(response)
    assert b'"2023-01-01T12:00:00Z"' in encoded


def test_encode_inference_response_bytes_mixed_native_and_object_array():
    """A response with BOTH a native FP32 ndarray output (the buffer
    fast path — untouched) and an object-dtype array carrying a UTC
    datetime (converted): each takes its own path, both render
    byte-identically to the schema's JSON mode."""
    response = InferenceResponse(
        model_name="m",
        outputs=[
            ResponseOutput(
                name="arr",
                datatype="FP32",
                shape=[4],
                data=TensorData(root=np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")),
            ),
            ResponseOutput(
                name="ts",
                datatype="BYTES",
                shape=[1],
                data=TensorData(
                    root=np.array(
                        [
                            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                            Decimal("2.72"),
                        ],
                        dtype=object,
                    )
                ),
            ),
        ],
    )
    fast = encode_inference_response_bytes(response)
    ground_truth = encode_to_json_bytes(
        response.model_dump(mode="json", exclude_unset=True)
    )
    assert fast == ground_truth
    assert b'"2023-01-01T12:00:00Z"' in fast
    assert b'"2.72"' in fast
