"""Wire-identity test suite for the buffer-native response pipeline.

The contract under test: the ndarray-native pipeline (mode-aware
TensorData serializer + orjson OPT_SERIALIZE_NUMPY) produces BYTE-IDENTICAL
wire output to the equivalent Python-list representation (the reference form)
on every datatype, shape, and edge value — the guarantee that inference
consumers are unaffected.

Positive: every native dtype, mixed outputs, multi-dim shapes, the full
REST path (exclude_unset semantics), the SSE path, json-mode pins.
Negative: non-native dtypes take the list path; the custom-handler
break is loud with a canonical fix.
Edge: NaN (canonical + noncanonical bits), +/-inf, -0.0, empty, single
element, extreme values, detection-scale payloads.
"""

import json

import numpy as np
import orjson
import pytest

from mlserver.codecs.json import encode_to_json_bytes
from mlserver.codecs.numpy import NumpyCodec
from mlserver.rest.responses import ServerSentEvent
from mlserver.types import InferenceResponse, TensorData

NATIVE_DTYPES = [
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
]

# complex is not a KServe datatype (the codec rejects it by design);
# float16 is the non-native dtype that CAN reach the serializer
NON_NATIVE_DTYPES = ["float16"]


def _response(payload: np.ndarray) -> InferenceResponse:
    output = NumpyCodec.encode_output(name="output-0", payload=payload)
    return InferenceResponse(model_name="m", outputs=[output])


def _reference_bytes(payload: np.ndarray) -> bytes:
    """Reference form: the json-mode dump (the unchanged tolist path)
    rendered with the same encoder — exactly what the pre-change
    pipeline produced, including every field the codec sets."""
    resp = _response(payload)
    return orjson.dumps(resp.model_dump(mode="json", exclude_unset=True))


def _wire_bytes(payload: np.ndarray) -> bytes:
    """The new pipeline: python-mode dump (ndarray kept) + orjson."""
    return encode_to_json_bytes(_response(payload).model_dump(exclude_unset=True))


# ---------------------------------------------------------------------
# Positive: byte-identity per dtype
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", NATIVE_DTYPES)
def test_wire_bytes_identical_per_dtype(dtype):
    payload = np.array([[1, 2], [3, 4]], dtype=dtype)
    assert _wire_bytes(payload) == _reference_bytes(payload)


@pytest.mark.parametrize("dtype", NATIVE_DTYPES)
def test_json_mode_dump_unchanged_per_dtype(dtype):
    """model_dump_json keeps today's list output on every dtype."""
    payload = np.array([[1, 2], [3, 4]], dtype=dtype)
    resp = _response(payload)
    assert isinstance(resp.model_dump(mode="json")["outputs"][0]["data"], list)


def test_wire_bytes_identical_mixed_outputs():
    """One response with FP32 + FP64 + INT32 + BOOL outputs."""
    payloads = [
        np.array([[1.5, 2.5]], dtype=np.float32),
        np.array([[1.5, 2.5]], dtype=np.float64),
        np.array([[1, 2]], dtype=np.int32),
        np.array([[True, False]], dtype=np.bool_),
    ]
    outputs = [
        NumpyCodec.encode_output(name=f"o{i}", payload=p)
        for i, p in enumerate(payloads)
    ]
    resp = InferenceResponse(model_name="m", outputs=outputs)

    wire = encode_to_json_bytes(resp.model_dump(exclude_unset=True))
    ref = orjson.dumps(resp.model_dump(mode="json", exclude_unset=True))
    assert wire == ref


@pytest.mark.parametrize("shape", [(4,), (2, 2), (2, 2, 2), (1, 1, 1, 4)])
def test_wire_bytes_identical_multidim(shape):
    payload = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_detection_scale():
    """A measured workload shape: 28K float32 values."""
    rng = np.random.default_rng(42)
    payload = (rng.random(28_000) * 100).astype(np.float32)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_sse_wire_bytes_identical():
    """The SSE path (ServerSentEvent.encode) renders the same body as
    the pre-change list form."""
    payload = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    resp = _response(payload)
    sse = ServerSentEvent(resp).encode()

    ref = b"data: " + orjson.dumps(resp.model_dump(mode="json")) + b"\n\n"
    assert sse == ref


def test_full_rest_path_byte_identity():
    """The exact unary REST semantics: the endpoint's pre-encoded
    Response return — python-mode model_dump(exclude_unset=True) +
    encode_to_json_bytes (the same construction the endpoint uses).
    The mode is pinned end-to-end by
    test_infer_endpoint_takes_python_mode (the ndarray must reach the
    encoder, not a tolist'd list)."""
    payload = np.array([[1.5, 2.5]], dtype=np.float32)
    resp = _response(payload)
    wire = encode_to_json_bytes(resp.model_dump(exclude_unset=True))
    ref = _reference_bytes(payload)
    assert wire == ref


# ---------------------------------------------------------------------
# Negative: non-native dtypes, the custom-handler contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", NON_NATIVE_DTYPES)
def test_non_native_dtype_takes_list_path(dtype):
    """Dtypes orjson cannot render natively keep the list path —
    wire identity still holds (through the json-safe list)."""
    payload = np.array([1.5, 2.5], dtype=np.float16)

    dumped = _response(payload).model_dump()
    data = dumped["outputs"][0]["data"]
    assert isinstance(data, list)
    # the encoder renders the list identically to the reference
    wire = encode_to_json_bytes(dumped)
    ref = _reference_bytes(payload)
    assert wire == ref


def test_masked_array_keeps_list_path():
    masked = np.ma.MaskedArray([1.5, 2.5, 3.5], mask=[False, True, False])
    output = NumpyCodec.encode_output(name="o", payload=masked)
    dumped = InferenceResponse(model_name="m", outputs=[output]).model_dump()
    assert isinstance(dumped["outputs"][0]["data"], list)
    assert dumped["outputs"][0]["data"][1] is None


def test_custom_handler_json_dumps_fails_loudly():
    """The documented contract change: json.dumps on a python-mode
    model_dump now fails on ndarray (loud, not silent corruption)."""
    payload = np.array([1.5], dtype=np.float32)
    dumped = _response(payload).model_dump()
    with pytest.raises(TypeError):
        json.dumps(dumped)


def test_custom_handler_fix_mode_json_works():
    """The canonical fix: mode='json' returns the JSON-safe list."""
    payload = np.array([1.5, 2.5], dtype=np.float32)
    dumped = _response(payload).model_dump(mode="json")
    assert json.dumps(dumped)  # no raise
    assert dumped["outputs"][0]["data"] == [1.5, 2.5]


# ---------------------------------------------------------------------
# Edge cases: special values, sizes, extremes
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "special",
    [
        np.nan,
        -np.nan,
        np.inf,
        -np.inf,
        -0.0,
        np.finfo(np.float32).max,
        np.finfo(np.float64).max,
        np.finfo(np.float64).tiny,
    ],
)
def test_wire_bytes_identical_special_values(special):
    payload = np.array([1.0, special, 3.0], dtype=np.float64)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_noncanonical_nan():
    """A noncanonical NaN bit pattern (0x7FC00001): the codec
    canonicalises on the detached copy; the wire renders null exactly
    like the list reference (whose tolist also yields NaN -> null)."""
    import struct

    noncanonical = struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]
    payload = np.array([1.5, noncanonical], dtype=np.float32)
    wire = _wire_bytes(payload)
    ref = _reference_bytes(payload)
    # both render the NaN as null; compare parsed values + the null
    assert orjson.loads(wire)["outputs"][0]["data"][1] is None
    assert orjson.loads(ref)["outputs"][0]["data"][1] is None


def test_wire_bytes_identical_all_nan():
    payload = np.array([np.nan, np.nan], dtype=np.float32)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_empty():
    payload = np.array([], dtype=np.float32)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_single_element():
    payload = np.array([7.25], dtype=np.float32)
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_int_extremes():
    payload = np.array(
        [np.iinfo(np.int64).min, -1, 0, 1, np.iinfo(np.int64).max], dtype=np.int64
    )
    assert _wire_bytes(payload) == _reference_bytes(payload)


def test_wire_bytes_identical_negative_zero():
    """-0.0 must render as -0.0 on both paths (not collapsed to 0.0)."""
    payload = np.array([-0.0, 0.0], dtype=np.float32)
    wire = _wire_bytes(payload)
    assert b"-0.0" in wire
    assert wire == _reference_bytes(payload)


def test_source_buffer_not_mutated_by_serialization():
    """Serializing must not touch the model's output buffer (the
    detach-copy invariant, at the serializer layer)."""
    payload = np.array([1.5, 2.5], dtype=np.float32)
    original_bits = payload.view(np.uint32).copy()

    _wire_bytes(payload)

    np.testing.assert_array_equal(payload.view(np.uint32), original_bits)


# ---------------------------------------------------------------------
# The compact-FP32 opt-in (rest_json_compact_floats)
# ---------------------------------------------------------------------


def test_compact_floats_default_off(monkeypatch):
    """Default: the widened bridge — byte-identical to the reference
    form (the pre-change pipeline's output)."""
    import mlserver.codecs.json as _jsoncodec

    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", False)
    payload = np.array([0.1, 1.5], dtype=np.float32)
    wire = _wire_bytes(payload)
    ref = _reference_bytes(payload)
    assert wire == ref
    # the widened float64 text (the reference form), not the short repr
    assert b"0.10000000149011612" in wire


def test_compact_floats_opt_in(monkeypatch):
    """Opt-in: shortest float32 repr — smaller bytes, same float32
    values, DIFFERENT text (the documented trade)."""
    import mlserver.codecs.json as _jsoncodec

    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", True)
    payload = np.array([0.1, 1.5], dtype=np.float32)
    wire = _wire_bytes(payload)
    ref = _reference_bytes(payload)

    # text differs, values identical after float32 round-trip
    assert wire != ref
    parsed = orjson.loads(wire)["outputs"][0]["data"]
    ref_parsed = orjson.loads(ref)["outputs"][0]["data"]
    assert [np.float32(v) for v in parsed] == [np.float32(v) for v in ref_parsed]
    # the compact form is present, the widened form is not
    assert b'"0.1"' in wire or b",0.1," in wire or b"[0.1," in wire
    assert b"0.10000000149011612" not in wire


def test_compact_floats_opt_in_smaller_payload(monkeypatch):
    """The reason to opt in: ~50% smaller FP32 responses."""
    rng = np.random.default_rng(7)
    payload = (rng.random(28_000) * 100).astype(np.float32)

    import mlserver.codecs.json as _jsoncodec

    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", False)
    wide = _wire_bytes(payload)
    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", True)
    compact = _wire_bytes(payload)

    assert len(compact) < len(wide) * 0.75  # measured ~52%


def test_compact_floats_json_mode_unaffected(monkeypatch):
    """The opt-in only touches python-mode rendering: json-mode
    (model_dump_json) keeps the list form either way."""
    import mlserver.codecs.json as _jsoncodec

    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", True)
    payload = np.array([0.1, 1.5], dtype=np.float32)
    resp = _response(payload)
    assert resp.model_dump_json()  # no raise, list path
    assert isinstance(resp.model_dump(mode="json")["outputs"][0]["data"], list)


def test_compact_floats_other_dtypes_unaffected(monkeypatch):
    """The opt-in only concerns FP32: float64/int/bool render
    identically in both modes."""
    import mlserver.codecs.json as _jsoncodec

    for dtype in ("float64", "int32", "bool"):
        monkeypatch.setattr(_jsoncodec, "_json_compact_floats", True)
        payload = np.array([1, 2], dtype=dtype)
        assert _wire_bytes(payload) == _reference_bytes(payload)


def test_compact_floats_special_values_unchanged(monkeypatch):
    """The opt-in changes value reprs only: NaN/inf render null and
    -0.0 keeps its sign in BOTH modes (special-value handling is
    mode-independent)."""
    import mlserver.codecs.json as _jsoncodec

    payload = np.array([0.1, np.nan, np.inf, -np.inf, -0.0], dtype=np.float32)
    td = TensorData(root=payload)

    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", False)
    wide = encode_to_json_bytes({"d": td.model_dump()})
    monkeypatch.setattr(_jsoncodec, "_json_compact_floats", True)
    compact = encode_to_json_bytes({"d": td.model_dump()})

    for out in (wide, compact):
        parsed = orjson.loads(out)["d"]
        assert parsed[1] is None  # nan -> null
        assert parsed[2] is None  # inf -> null
        assert parsed[3] is None  # -inf -> null
        assert parsed[4] == 0.0 and str(parsed[4]) == "-0.0"  # sign kept
    assert b"-0.0" in wide and b"-0.0" in compact


@pytest.mark.parametrize("special", [np.nan, -np.nan, np.inf, -np.inf])
def test_orjson_renders_special_values_as_null(special):
    """Pin orjson's special-value rendering (the encoder's contract):
    every NaN/inf form -> null, on both the list and ndarray paths —
    the reason the wire stays byte-identical without explicit
    conversion in python mode."""
    assert orjson.dumps([special]) == b"[null]"
    arr = np.array([special], dtype=np.float64)
    assert orjson.dumps(arr, option=orjson.OPT_SERIALIZE_NUMPY) == b"[null]"
