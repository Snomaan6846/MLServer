"""KServe v2 Binary Tensor Data Extension for REST inference.

When the ``rest_binary_tensor`` setting is enabled, unary REST inference
requests carrying the ``Inference-Header-Content-Length`` header are
parsed as a JSON header followed by raw binary tensor data.  Zero-copy
NumPy views are created via ``np.frombuffer``.  Responses use the same
format when the client requests binary output.  Standard JSON requests
are unaffected.
"""

from __future__ import annotations

import numpy as np
import orjson
from typing import Any

from ..types import InferenceRequest, InferenceResponse

HEADER_NAME = "inference-header-content-length"

# KServe datatype string -> NumPy dtype with explicit little-endian byte
# order, as required by the binary tensor wire format.  This parallels
# ``mlserver.codecs.numpy._DatatypeToNumpy`` but serves a different
# purpose: that map handles in-memory type conversion, while this one
# maps the wire format's mandated little-endian representation.
# Single-byte types (BOOL, UINT8, INT8) have no byte-order concern.
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


class BinaryTensorError(Exception):
    """Raised when a binary tensor request is malformed."""

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


def parse_binary_request(body: bytes, header_length: int) -> InferenceRequest:
    """Parse a binary tensor request body into an InferenceRequest.

    The body is a JSON header followed by binary tensor data for each
    input carrying a ``binary_data_size`` parameter.  NumPy arrays are
    created as zero-copy views and stored in the ``data`` field so the
    existing codec pipeline can decode them without modification.
    """
    if header_length > len(body):
        raise BinaryTensorError(
            f"Header length ({header_length}) exceeds body size " f"({len(body)})"
        )

    header_bytes = body[:header_length]
    binary_bytes = body[header_length:]

    try:
        header = orjson.loads(header_bytes)
    except orjson.JSONDecodeError as error:
        raise BinaryTensorError(f"Invalid JSON header: {error}")

    if not isinstance(header, dict) or "inputs" not in header:
        raise BinaryTensorError("JSON header missing 'inputs' field")

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

        if not isinstance(binary_size, int) or binary_size < 0:
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

        datatype = input_spec.get("datatype", "")
        if datatype not in DATATYPE_MAP:
            raise BinaryTensorError(
                f"Input '{input_spec.get('name', '?')}: unsupported "
                f"datatype for binary tensor ({datatype})"
            )

        dtype = DATATYPE_MAP[datatype]
        shape = input_spec.get("shape", [])
        expected_elements = 1
        for dim in shape:
            expected_elements *= dim
        expected_bytes = expected_elements * dtype.itemsize

        if binary_size != expected_bytes:
            raise BinaryTensorError(
                f"Input '{input_spec.get('name', '?')}': binary_data_size "
                f"({binary_size}) doesn't match shape {shape} x "
                f"{datatype} ({expected_bytes} bytes)"
            )

        tensor = np.frombuffer(
            binary_bytes, dtype=dtype, count=expected_elements, offset=offset
        ).reshape(shape)
        input_spec["data"] = tensor
        offset += binary_size

    if offset < len(binary_bytes):
        raise BinaryTensorError(
            f"{len(binary_bytes) - offset} unaccounted binary bytes " "after all inputs"
        )

    return InferenceRequest.model_validate(header)


def client_wants_binary(header_dict: dict[str, Any]) -> bool:
    """Check whether the client requested binary output."""
    outputs = header_dict.get("outputs") or []
    if not outputs:
        return False
    params = header_dict.get("parameters") or {}
    if params.get("binary_data_output") is True:
        return True
    for out in outputs:
        out_params = out.get("parameters") or {}
        if out_params.get("binary_data") is True:
            return True
    return False


def encode_binary_response(
    response: InferenceResponse,
) -> tuple[bytes, int]:
    """Encode an InferenceResponse as binary tensor data.

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

    binary_chunks: list[bytes] = []
    for output in response.outputs:
        # Unwrap TensorData to get the underlying payload
        data = getattr(output.data, "root", output.data)

        # Convert to NumPy array if needed (worker IPC may flatten to
        # lists before the response reaches the frontend)
        if not isinstance(data, np.ndarray):
            dtype = DATATYPE_MAP.get(str(output.datatype), np.dtype("<f4"))
            if hasattr(data, "__len__"):
                data = np.asarray(data, dtype=dtype)
            else:
                data = np.array([data], dtype=dtype)

        # Use the output's declared shape (may be multi-dimensional)
        shape = output.shape or list(data.shape)
        expected = 1
        for dim in shape:
            expected *= dim
        if data.size != expected:
            data = data.reshape(shape)
        datatype = str(output.datatype or "FP32")

        # Ensure little-endian contiguous, as required by the wire format
        if data.dtype.byteorder == ">" or (
            data.dtype.byteorder == "=" and not np.little_endian
        ):
            data = data.astype(data.dtype.newbyteorder("<"))
        raw = data.tobytes(order="C")

        if output.parameters:
            params = output.parameters.model_dump(
                mode="json", exclude_unset=True, exclude_none=True
            )
        else:
            params = {}
        params["binary_data_size"] = len(raw)
        header["outputs"].append(
            {
                "name": output.name,
                "datatype": datatype,
                "shape": shape,
                "parameters": params,
            }
        )
        binary_chunks.append(raw)

    header_bytes = orjson.dumps(header)
    return header_bytes + b"".join(binary_chunks), len(header_bytes)
