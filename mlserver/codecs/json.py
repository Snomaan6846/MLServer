# seperate file to side step circular dependency on the decode_str function

import json
import numpy as np
from typing import Any

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

from .string import decode_str
from .lists import as_list
from .utils import InputOrOutput

from ..logging import logger


# originally taken from: mlserver/rest/responses.py
class _BytesJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # If we get a bytes payload, try to decode it back to a string on a
            # "best effort" basis
            return decode_str(obj)

        return super().default(self, obj)


def _encode_object_to_bytes(obj: Any) -> str:
    """
    Add compatibility with `bytes` payloads to `orjson`
    """
    if isinstance(obj, bytes):
        # If we get a bytes payload, try to decode it back to a string on a
        # "best effort" basis
        return decode_str(obj)

    raise TypeError


# Opt-in compact FP32 JSON rendering (rest_json_compact_floats):
# set once at bootstrap from Settings (server + frontend children +
# workers). Default False = the byte-preserving widened form.
_json_compact_floats = False

# Response-metadata JSON-mode conversion (see json_mode_convert):
# built once — it runs on the response hot path.
_JSON_MODE_ADAPTER: "Any | None" = None


def set_json_compact_floats(enabled: bool) -> None:
    """Set the compact-FP32 JSON rendering opt-in (see Settings)."""
    global _json_compact_floats
    _json_compact_floats = enabled


def json_compact_floats_enabled() -> bool:
    """Read the compact-FP32 JSON rendering opt-in's current value.

    Reads at call time so every process sees its own bootstrap value
    (set wherever Settings load: server, frontends, workers).
    """
    return _json_compact_floats


def encode_to_json_bytes(v: Any) -> bytes:
    """encodes a dict into json bytes, can deal with byte like values gracefully"""
    if orjson is None:
        # Dead path in practice (orjson is a hard dependency), kept for
        # environments without it: NOTE it cannot render python-mode
        # ndarray roots (raises loudly) — only the orjson path can.
        # Original implementation of starlette's JSONResponse, using our
        # custom encoder (capable of "encoding" bytes).
        # Original implementation can be seen here:
        # https://github.com/encode/starlette/blob/
        # f53faba229e3fa2844bc3753e233d9c1f54cca52/starlette/responses.py#L173-L180
        return json.dumps(
            v,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=_BytesJSONEncoder,
        ).encode("utf-8")

    return orjson.dumps(
        v,
        default=_encode_object_to_bytes,
        # Ndarray data renders straight from the buffer (no Python
        # element objects): ~3.5x faster than tolist + orjson on a
        # 28K-float response, byte-identical via the float64 bridge
        # in the TensorData serializer.
        option=orjson.OPT_SERIALIZE_NUMPY,
    )


def _coerce_numpy_scalars(value: Any) -> Any:
    """Recursively replace loose numpy scalars with python scalars.

    The KServe reference implementation normalises numpy values to
    python scalars at the wire boundary (``[val.item() for val in
    tensor.flatten()]``); this mirrors it for values that reach the
    conversion as loose scalars. The widened python float renders
    exactly what the tolist'd ndarray path renders — orjson's own
    loose-scalar rendering would emit the SHORT float32 repr, the
    wire-format change the rest_json_compact_floats opt-in gates.
    """
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, list):
        return [_coerce_numpy_scalars(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_coerce_numpy_scalars(v) for v in value)
    if isinstance(value, dict):
        return {k: _coerce_numpy_scalars(v) for k, v in value.items()}
    return value


def json_mode_convert(value: Any) -> Any:
    """Pydantic's JSON-mode conversion for the REST wire: response
    metadata AND non-ndarray tensor data."""
    # The pre-encoded response path renders ndarray tensor data
    # natively; everything else keeps the scalar conversion the
    # response-model path performed (Decimal -> str, timedelta ->
    # ISO duration, UTC datetime -> Z). Values the conversion cannot
    # handle return unchanged — orjson's default adapter (bytes)
    # still applies to them.
    global _JSON_MODE_ADAPTER
    if _JSON_MODE_ADAPTER is None:
        from pydantic import TypeAdapter

        _JSON_MODE_ADAPTER = TypeAdapter(Any)
    try:
        return _JSON_MODE_ADAPTER.dump_python(value, mode="json")
    except Exception:
        pass
    # Retry with loose numpy scalars coerced to python scalars: a
    # single unknown-type element (e.g. a loose np.float32) aborts
    # pydantic's whole-tree walk, which would strand any exotics
    # next to it (a UTC datetime rendering +00:00 instead of Z) and
    # leak orjson's short float32 repr into the default
    # (byte-preserving) mode. The retry only runs on the failure
    # path — plain data never pays for it.
    try:
        return _JSON_MODE_ADAPTER.dump_python(_coerce_numpy_scalars(value), mode="json")
    except Exception:
        # A genuine pydantic failure degrades to the orjson default
        # adapter — leave a trace so it is diagnosable, not silent.
        logger.debug(
            "JSON-mode metadata conversion failed; using the default" " adapter",
            exc_info=True,
        )
        return value


def decode_from_bytelike_json_to_dict(v: bytes | str) -> dict:
    if orjson is None:
        return json.loads(v)
    return orjson.loads(v)


class JSONEncoderWithArray(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def encode_to_json(v: Any, use_bytes: bool = True) -> str | bytes:
    enc_v = json.dumps(
        v,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        cls=JSONEncoderWithArray,
    )
    if use_bytes:
        enc_v = enc_v.encode("utf-8")  # type: ignore[assignment]
    return enc_v


def decode_json_input_or_output(input_or_output: InputOrOutput) -> list[Any]:
    packed = input_or_output.data.root
    unpacked = map(json.loads, as_list(packed))
    return list(unpacked)
