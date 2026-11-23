"""KServe v2 Binary Tensor Data Extension for REST inference.

When the ``rest_binary_tensor`` setting is enabled, REST inference
requests carrying the ``Inference-Header-Content-Length`` header are
parsed as a JSON header followed by raw binary tensor data.  Numeric
tensor bytes are viewed in place (``np.frombuffer`` over a memoryview
of the request body); BYTES tensors use the KServe length-prefixed
element framing.

Responses follow the client's per-output negotiation: the request-wide
``parameters.binary_data_output`` is the default and each requested
output's ``parameters.binary_data`` overrides it.  Binary outputs are
appended after the JSON header with ``binary_data_size`` parameters;
JSON outputs keep their inline data in the header.  Standard JSON
requests are unaffected, and may themselves request binary outputs.
"""

from __future__ import annotations

import numpy as np
import orjson
import struct
from typing import Any

from pydantic import ValidationError

from ..codecs.numpy import LOGICAL_BYTES_ELEMENTS, LOGICAL_BYTES_SENTINEL
from ..types import InferenceRequest, InferenceResponse, TensorData

HEADER_NAME = "inference-header-content-length"

# KServe datatype string -> NumPy dtype with explicit little-endian byte
# order, as required by the binary tensor wire format.  This parallels
# ``mlserver.codecs.numpy._DatatypeToNumpy`` but serves a different
# purpose: that map handles in-memory type conversion, while this one
# maps the wire format's mandated little-endian representation.
# Single-byte types (BOOL, UINT8, INT8) have no byte-order concern.
# BYTES is framed separately (length-prefixed elements) and is absent
# from this map by design.
DATATYPE_MAP: dict[str, np.dtype] = {
    "BOOL": np.dtype("i1"),
    "UINT8": np.dtype("u1"),
    "UINT16": np.dtype("<u2"),
    "UINT32": np.dtype("<u4"),
    "UINT64": np.dtype("<u8"),
    "INT8": np.dtype("i1"),
    "INT16": np.dtype("<i2"),
    "INT32": np.dtype("<i4"),
    "INT64": np.dtype("<i8"),
    "FP16": np.dtype("<f2"),
    "FP32": np.dtype("<f4"),
    "FP64": np.dtype("<f8"),
}

_BYTES_DATATYPE = "BYTES"
_BYTES_LENGTH_FORMAT = "<I"  # 4-byte little-endian element length prefix


def _damages_unicode(elements: list[bytes]) -> bool:
    """Whether the Unicode (JSON-equivalent) path would alter any element.

    numpy's fixed-width string arrays strip trailing NULs, and lone
    surrogates cannot round-trip strict UTF-8 encoding.
    """
    for elem in elements:
        if elem.endswith(b"\x00"):
            return True
        try:
            elem.decode("utf-8")
        except UnicodeDecodeError:
            return True
    return False


class BinaryTensorError(Exception):
    """Raised when a binary tensor request or response is malformed."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def is_binary_request(headers: dict[str, str]) -> bool:
    """Check whether the request carries the binary tensor header.

    Starlette normalises header names to lowercase, so a direct
    membership test suffices.
    """
    return HEADER_NAME in headers


def get_header_length(headers: dict[str, str]) -> int:
    """Extract and validate ``Inference-Header-Content-Length``."""
    raw = headers.get(HEADER_NAME)
    if raw is None:
        raise BinaryTensorError(f"Missing {HEADER_NAME} header")
    try:
        length = int(raw)
    except (ValueError, TypeError):
        raise BinaryTensorError(f"Invalid {HEADER_NAME}: not an integer ({raw!r})")
    if length < 0:
        raise BinaryTensorError(f"Invalid {HEADER_NAME}: negative ({length})")
    return length


def _unpack_bytes_region(buf: Any, offset: int, size: int) -> list[bytes]:
    """Unpack length-prefixed BYTES elements, bounds-checked.

    Each element is a 4-byte little-endian length prefix followed by
    that many payload bytes (the KServe binary data representation for
    BYTES tensors).
    """
    end = offset + size
    pos = offset
    elems: list[bytes] = []
    while pos < end:
        if pos + 4 > end:
            raise BinaryTensorError("Truncated BYTES element length prefix")
        (elem_size,) = struct.unpack_from(_BYTES_LENGTH_FORMAT, buf, pos)
        pos += 4
        if pos + elem_size > end:
            raise BinaryTensorError("Truncated BYTES element payload")
        elems.append(bytes(buf[pos : pos + elem_size]))
        pos += elem_size
    return elems


def _pack_bytes_elements(elements: Any) -> bytes:
    """Pack BYTES elements with the length-prefixed framing."""
    if elements is None:
        elements = []
    if not isinstance(elements, (list, tuple)):
        elements = [elements]
    chunks: list[bytes] = []
    for elem in elements:
        if isinstance(elem, str):
            # surrogateescape restores the exact wire bytes for
            # non-UTF-8 elements parsed on the input side
            elem = elem.encode("utf-8", "surrogateescape")
        elif isinstance(elem, (bytearray, memoryview)):
            elem = bytes(elem)
        elif not isinstance(elem, bytes):
            raise BinaryTensorError(
                f"Unsupported BYTES element type ({type(elem).__name__})"
            )
        chunks.append(struct.pack(_BYTES_LENGTH_FORMAT, len(elem)))
        chunks.append(elem)
    return b"".join(chunks)


def _check_header_structure(header: dict[str, Any]) -> None:
    """Validate header structure before any field is dereferenced.

    Keeps malformed metadata (null inputs, scalar input entries, bad
    shapes or datatypes) on the 4xx path instead of escaping as an
    unhandled HTTP 500.
    """
    inputs = header.get("inputs")
    if not isinstance(inputs, list):
        raise BinaryTensorError(
            f"'inputs' must be a list (got {type(inputs).__name__})"
        )
    for input_spec in inputs:
        name = input_spec.get("name", "?") if isinstance(input_spec, dict) else "?"
        if not isinstance(input_spec, dict):
            raise BinaryTensorError(
                f"Each input must be an object (got {type(input_spec).__name__})"
            )
        params = input_spec.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise BinaryTensorError(f"Input '{name}': 'parameters' must be an object")
        shape = input_spec.get("shape", [])
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in shape
        ):
            raise BinaryTensorError(f"Input '{name}': invalid shape ({shape!r})")
        datatype = input_spec.get("datatype")
        if not isinstance(datatype, str):
            raise BinaryTensorError(f"Input '{name}': 'datatype' must be a string")


def parse_binary_request(
    body: bytes, header_length: int
) -> tuple[InferenceRequest, dict[str, Any]]:
    """Parse a binary tensor request body into an InferenceRequest.

    The body is a JSON header followed by binary tensor data for each
    input carrying a ``binary_data_size`` parameter.  Numeric tensors
    are stored as flat read-only NumPy views over the request body
    (mirroring the gRPC raw-input representation: ``TensorData`` is flat
    even when the logical tensor has shape).  BYTES tensors are stored
    as lists of bytes decoded from the length-prefixed framing.

    Returns ``(request, header_dict)`` — the raw header dict is reused
    for output negotiation without re-parsing the body.
    """
    if header_length > len(body):
        raise BinaryTensorError(
            f"Header length ({header_length}) exceeds body size ({len(body)})"
        )

    body_view = memoryview(body)
    header_bytes = body_view[:header_length]
    binary_bytes = body_view[header_length:]

    try:
        header = orjson.loads(header_bytes)
    except orjson.JSONDecodeError as error:
        raise BinaryTensorError(f"Invalid JSON header: {error}")

    if not isinstance(header, dict) or "inputs" not in header:
        raise BinaryTensorError("JSON header missing 'inputs' field")

    _check_header_structure(header)

    offset = 0
    for input_spec in header["inputs"]:
        params = input_spec.get("parameters") or {}
        binary_size = params.get("binary_data_size")
        if binary_size is None:
            if "data" not in input_spec:
                raise BinaryTensorError(
                    f"Input '{input_spec.get('name', '?')}' has neither "
                    "'data' nor 'parameters.binary_data_size'"
                )
            continue

        if "data" in input_spec:
            raise BinaryTensorError(
                f"Input '{input_spec.get('name', '?')}' has both 'data' "
                "and 'parameters.binary_data_size' (mutually exclusive)"
            )

        if (
            not isinstance(binary_size, int)
            or isinstance(binary_size, bool)
            or binary_size < 0
        ):
            raise BinaryTensorError(
                f"Input '{input_spec.get('name', '?')}': invalid "
                f"binary_data_size ({binary_size!r})"
            )

        if offset + binary_size > len(binary_bytes):
            raise BinaryTensorError(
                f"Input '{input_spec.get('name', '?')}': binary_data_size "
                f"({binary_size}) exceeds remaining data "
                f"({len(binary_bytes) - offset} bytes)"
            )

        name = input_spec.get("name", "?")
        datatype = input_spec["datatype"]
        shape = input_spec.get("shape", [])
        expected_elements = 1
        for dim in shape:
            expected_elements *= dim

        if datatype == _BYTES_DATATYPE:
            elements = _unpack_bytes_region(binary_bytes, offset, binary_size)
            if len(elements) != expected_elements:
                raise BinaryTensorError(
                    f"Input '{name}': binary_data_size ({binary_size}) frames "
                    f"{len(elements)} BYTES elements, expected {expected_elements}"
                )
            # Bridge to the codec's representation.  UTF-8-decodable
            # elements decode exactly like the equivalent JSON string
            # input (NumpyCodec's str path).  Anything else is opaque
            # binary: an object-dtype array preserves each element's
            # exact value and length through the NumPy codec round-trip
            # (no fixed-width truncation, no trailing-NUL stripping),
            # and NumpyCodec.encode_output returns the exact bytes.
            # Values that the Unicode path would damage — non-UTF-8
            # bytes or trailing NULs, which numpy's fixed-width string
            # arrays strip — are kept as opaque byte elements, flagged
            # so the codec preserves logical-element semantics through
            # container changes (batching converts the object array to
            # a plain list).  Untouched values decode exactly like the
            # equivalent JSON string input.
            if _damages_unicode(elements):
                input_spec["data"] = np.array(elements, dtype=object)
                params = input_spec.setdefault("parameters", {})
                params[LOGICAL_BYTES_ELEMENTS] = LOGICAL_BYTES_SENTINEL
            else:
                input_spec["data"] = [elem.decode("utf-8") for elem in elements]
        else:
            if datatype not in DATATYPE_MAP:
                raise BinaryTensorError(
                    f"Input '{name}': unsupported datatype for binary "
                    f"tensor ({datatype})"
                )
            dtype = DATATYPE_MAP[datatype]
            expected_bytes = expected_elements * dtype.itemsize
            if binary_size != expected_bytes:
                raise BinaryTensorError(
                    f"Input '{name}': binary_data_size ({binary_size}) "
                    f"doesn't match shape {shape} x {datatype} "
                    f"({expected_bytes} bytes)"
                )
            tensor = np.frombuffer(
                binary_bytes, dtype=dtype, count=expected_elements, offset=offset
            )
            input_spec["data"] = tensor.reshape(-1)
        offset += binary_size

    if offset < len(binary_bytes):
        raise BinaryTensorError(
            f"{len(binary_bytes) - offset} unaccounted binary bytes after all inputs"
        )

    # Pydantic's ``list[Any]`` coercion iterates array payloads element
    # by element — 786K iterations for an image tensor.  Strip the
    # bulky payloads, validate the request structure, then attach the
    # zero-copy views without re-validation.
    array_data: dict[int, Any] = {}
    for input_spec in header["inputs"]:
        value = input_spec.get("data")
        if isinstance(value, np.ndarray):
            array_data[id(input_spec)] = value
            input_spec["data"] = None

    try:
        request = InferenceRequest.model_validate(header)
    except ValidationError as error:
        raise BinaryTensorError(f"Invalid inference request header: {error}")

    for input_spec, request_input in zip(header["inputs"], request.inputs):
        saved = array_data.get(id(input_spec))
        if saved is not None:
            request_input.data = TensorData.model_construct(root=saved)

    return request, header


def resolve_binary_outputs(
    request: InferenceRequest,
) -> tuple[bool, dict[str, bool] | None]:
    """Resolve the response format per output, per KServe semantics.

    ``parameters.binary_data_output`` is the request-wide default and
    each requested output's ``parameters.binary_data`` overrides it.
    Returns ``(default, per_output)`` where ``per_output`` maps the
    requested output names to their resolved format, or is ``None``
    when the request listed no outputs.
    """
    default = False
    if request.parameters is not None:
        default = getattr(request.parameters, "binary_data_output", None) is True

    if not request.outputs:
        return default, None

    per_output: dict[str, bool] = {}
    for output in request.outputs:
        wants = default
        if output.parameters is not None:
            override = getattr(output.parameters, "binary_data", None)
            if override is not None:
                wants = override is True
        per_output[output.name] = wants
    return default, per_output


def _flatten_elements(data: Any) -> list[Any]:
    """Flatten nested element form to the wire's flat element list.

    The KServe framing carries ``prod(shape)`` flat elements, so nested
    arrays (a model's data matching a multi-dimensional shape) frame
    identically to their flat equivalent. Non-list elements pass
    through, preserving the packed single-buffer form.
    """
    if not isinstance(data, (list, tuple)):
        return [data]
    flat: list[Any] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten_elements(item))
        else:
            flat.append(item)
    return flat


def _bytes_wire_elements(output: Any, data: Any) -> list[Any] | None:
    """Resolve the wire elements for a BYTES output.

    Bridges the codec's representations to the KServe wire framing:

    - ``'S'``-dtype arrays (a model's byte arrays): packed via
      ``tobytes()`` and split into ``prod(shape)`` equal chunks — the
      raw buffer is preserved exactly, with no ``.tolist()`` trailing-
      NUL stripping.
    - Element form (``len(data) == prod(shape)``, e.g. the parser's
      surrogate-escaped strings or string arrays encoded as UTF-8
      elements): framed as-is. Nested element form (a model's data
      matching a multi-dimensional shape) flattens first, so it frames
      identically to its flat equivalent.
    - Packed form (the NumPy codec's ``[data.tobytes()]`` for ``'S'``
      arrays): the buffer is split into ``prod(shape)`` equal chunks —
      the codec's ``frombuffer`` decode normalises both forms the same
      way, so the split round-trips.
    - Zero-element tensors frame nothing (``binary_data_size == 0``).

    Returns None when the output cannot be framed (mixed element
    types, or a buffer that does not divide evenly).
    """
    if isinstance(data, np.ndarray) and data.dtype == object:
        # Opaque byte elements: exact values and lengths, no stripping
        data = data.reshape(-1).tolist()

    if isinstance(data, np.ndarray) and data.dtype.kind in ("S", "a"):
        # Pack the raw buffer (exact bytes, no .tolist() NUL stripping)
        # and let the equal-chunk split below restore the elements.
        data = [data.tobytes()]

    if isinstance(data, (list, tuple)):
        data = _flatten_elements(data)

    elements = data if isinstance(data, (list, tuple)) else [data]
    shape = output.shape or [len(elements)]
    expected = 1
    for dim in shape:
        expected *= dim

    if expected == 0:
        # Zero-element tensor: no frames.  The codec's empty packed
        # form ([b""]) and an empty element list both mean "no data".
        if not elements or all(elem in (b"", "") for elem in elements):
            return []
        return None

    if len(elements) == expected:
        return list(elements)

    bytes_elements = [
        bytes(elem) if isinstance(elem, (bytes, bytearray)) else elem
        for elem in elements
    ]
    if all(isinstance(elem, bytes) for elem in bytes_elements):
        joined = b"".join(bytes_elements)
        if len(joined) > 0 and len(joined) % expected == 0:
            itemsize = len(joined) // expected
            return [joined[i * itemsize : (i + 1) * itemsize] for i in range(expected)]
    return None


def _encode_output_bytes(output: Any) -> bytes | None:
    """Encode one output's tensor bytes, or None when unframable.

    Unframable outputs (unknown datatypes, unsupported BYTES elements,
    shape/data mismatches, oversized elements) fall back to inline JSON
    in the header, which is protocol-legal — matching the plain-JSON
    path's tolerance of the same conditions.
    """
    data = getattr(output.data, "root", output.data)
    datatype = str(output.datatype or "")
    try:
        if datatype == _BYTES_DATATYPE:
            elements = _bytes_wire_elements(output, data)
            if elements is None:
                return None
            return _pack_bytes_elements(elements)
        dtype = DATATYPE_MAP.get(datatype)
        if dtype is None:
            return None
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=dtype)
        elif data.dtype != dtype:
            if data.dtype.itemsize > dtype.itemsize:
                # A narrowing conversion cannot be lossless: the packed
                # bytes would silently truncate or wrap values the
                # plain-JSON path preserves exactly (an INT64 array
                # under an INT32 declaration). Refuse binary packing so
                # the output falls back to inline JSON — the same
                # never-corrupt discipline as the gRPC raw path's
                # checked semantics.
                raise BinaryTensorError(
                    f"binary output declared {datatype} "
                    f"({dtype.itemsize} bytes) but the array dtype is "
                    f"{data.dtype} ({data.dtype.itemsize} bytes)"
                )
            # Same-width or widening conversion to the declared
            # datatype. Only same-kind widening is provably lossless
            # (int32->int64, float32->float64). Same-width cross-kind
            # conversions follow numpy's astype semantics and CAN alter
            # values (float 1.5 under an INT32 declaration packs as 1;
            # int32 under FP32 rounds above 2^24; uint32 under INT32
            # wraps at 2^31) — a documented divergence from the
            # plain-JSON path, matching the base REST behavior this
            # path historically kept while the gRPC raw path stayed
            # struct-checked.
            data = data.astype(dtype)
        # NOTE: no reshape — tobytes(order="C") on a flat contiguous
        # array does not depend on .shape, and a declared-shape/data
        # mismatch passes through the same way the JSON path tolerates
        # it (binary_data_size remains authoritative for framing).
        # Ensure little-endian contiguous, as required by the wire format
        if data.dtype.byteorder == ">" or (
            data.dtype.byteorder == "=" and not np.little_endian
        ):
            data = data.astype(data.dtype.newbyteorder("<"))
        return data.tobytes(order="C")
    except (BinaryTensorError, ValueError, struct.error):
        # ValueError: unframable data/shape combinations;
        # struct.error: BYTES elements >= 4 GiB overflow the
        # 4-byte length prefix
        return None


def _inline_json_element(elem: Any) -> Any:
    """Inline header representation for one data element.

    Bytes elements decode as UTF-8 strings; strings carrying surrogate
    escapes (the parser's non-UTF-8 bridge) map losslessly through
    latin-1, since JSON cannot carry lone surrogates. Nested lists (a
    model's data matching a multi-dimensional shape) recurse, matching
    the plain-JSON path's structure.
    """
    if isinstance(elem, (bytes, bytearray)):
        return elem.decode("utf-8", "replace")
    if isinstance(elem, str):
        try:
            elem.encode("utf-8")
            return elem
        except UnicodeEncodeError:
            return elem.encode("utf-8", "surrogateescape").decode("latin-1")
    if isinstance(elem, (list, tuple)):
        return [_inline_json_element(item) for item in elem]
    return elem


def _inline_json_data(data: Any) -> Any:
    """Inline header representation for JSON-resolved outputs.

    Both the element form (lists, including nested element form
    matching a multi-dimensional shape) and array form (object- or
    'S'-dtype ndarrays from the parser or the model) go through the
    same element loop — orjson cannot serialize raw bytes, so an
    undecoded element would turn a protocol-legal JSON fallback into a
    500.

    FP32 arrays honour the rest_json_compact_floats opt-in: with the
    flag on, the flat array is returned for orjson's native ndarray
    rendering (shortest round-trip text) so the fallback renders
    byte-identically to the plain-JSON path — the fallback's contract
    is "what the JSON path would have returned". Default (flag off):
    tolist's widened Python floats, unchanged.
    """
    if isinstance(data, np.ndarray):
        if data.dtype.name == "float32":
            # Deferred import: reads the CURRENT flag value (set at
            # bootstrap from Settings in every process).
            from ..codecs.json import json_compact_floats_enabled

            if json_compact_floats_enabled():
                return data.reshape(-1)
        data = data.reshape(-1).tolist()
    if isinstance(data, (list, tuple)):
        return [_inline_json_element(elem) for elem in data]
    return data


def encode_binary_response(
    response: InferenceResponse,
    default_binary: bool = False,
    per_output: dict[str, bool] | None = None,
) -> tuple[bytes, int]:
    """Encode an InferenceResponse with per-output format resolution.

    Outputs resolved as binary contribute their raw bytes after the
    JSON header (with ``binary_data_size`` parameters); outputs
    resolved as JSON keep their inline data in the header.

    Returns ``(body, header_length)`` where ``header_length`` is the
    byte length of the JSON header (for the response's
    ``Inference-Header-Content-Length`` header).
    """
    header: dict[str, Any] = {
        "model_name": response.model_name,
        "model_version": response.model_version,
        "id": response.id,
        "outputs": [],
    }
    if response.parameters:
        header["parameters"] = response.parameters.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )

    chunks: list[bytes] = []
    for output in response.outputs:
        wants_binary = default_binary
        if per_output is not None and output.name in per_output:
            wants_binary = per_output[output.name]

        params: dict[str, Any] = {}
        if output.parameters:
            params = output.parameters.model_dump(
                mode="json", exclude_unset=True, exclude_none=True
            )

        out_header: dict[str, Any] = {
            "name": output.name,
            "datatype": str(output.datatype or ""),
            "shape": list(output.shape or []),
        }

        raw = _encode_output_bytes(output) if wants_binary else None
        if wants_binary and raw is not None:
            params["binary_data_size"] = len(raw)
            out_header["parameters"] = params
            chunks.append(raw)
        else:
            data = getattr(output.data, "root", output.data)
            out_header["data"] = _inline_json_data(data)
            if params:
                out_header["parameters"] = params
        header["outputs"].append(out_header)

    # OPT_SERIALIZE_NUMPY: renders the compact-floats FP32 arrays
    # natively (the fallback's contract: byte-identical to the
    # plain-JSON path). Lists/strings/numbers render identically with
    # or without the option — every other header value is unaffected.
    header_bytes = orjson.dumps(header, option=orjson.OPT_SERIALIZE_NUMPY)
    return b"".join([header_bytes, *chunks]), len(header_bytes)
