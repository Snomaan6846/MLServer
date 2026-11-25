import numpy as np

from typing import Any

from ..types import RequestInput, ResponseOutput, Parameters, Datatype

from .base import InputCodec, register_input_codec, register_request_codec
from .utils import SingleInputRequestCodec, InputOrOutput, inject_batch_dimension
from .lists import is_list_of
from .string import encode_str

# Parameter marking logical BYTES elements (the binary tensor
# transport's length-prefixed wire elements).  Flagged data preserves
# element semantics through container changes — the NumPy codec decodes
# it as an object array of exact byte elements — while unflagged bytes
# lists keep the legacy packed-buffer interpretation.
#
# The value is a TUPLE sentinel: JSON parameters can only produce
# lists, so a client cannot forge the flag through an ordinary request
# (Parameters uses extra="allow", making any string key wire-settable).
# Pickle (IPC) preserves tuples, so the flag survives the worker hop.
LOGICAL_BYTES_ELEMENTS = "_mlserver_logical_bytes_elements"
LOGICAL_BYTES_SENTINEL = ("mlserver-internal", "logical-bytes-elements")


def set_logical_elements(request_input: InputOrOutput) -> None:
    """Mark an input as carrying logical (opaque) byte elements.

    Used by test fixtures and any internal caller that needs to flag
    programmatic inputs; the binary tensor transport's parser sets the
    sentinel directly on the raw header dict (pre-validation, before
    the RequestInput object exists).
    """
    if request_input.parameters is None:
        request_input.parameters = Parameters()
    setattr(request_input.parameters, LOGICAL_BYTES_ELEMENTS, LOGICAL_BYTES_SENTINEL)


def _is_sentinel(value: Any) -> bool:
    return isinstance(value, tuple) and value == LOGICAL_BYTES_SENTINEL


def _has_logical_elements(input_or_output: InputOrOutput) -> bool:
    params = input_or_output.parameters
    if params is None:
        return False
    value = getattr(params, LOGICAL_BYTES_ELEMENTS, None)
    if _is_sentinel(value):
        return True
    # Adaptive batching aggregates duplicate parameters into lists;
    # every-sentinel means every merged entry carried logical elements
    return (
        isinstance(value, list) and bool(value) and all(_is_sentinel(v) for v in value)
    )


_DatatypeToNumpy = {
    Datatype.BOOL: "bool",
    Datatype.UINT8: "uint8",
    Datatype.UINT16: "uint16",
    Datatype.UINT32: "uint32",
    Datatype.UINT64: "uint64",
    Datatype.INT8: "int8",
    Datatype.INT16: "int16",
    Datatype.INT32: "int32",
    Datatype.INT64: "int64",
    Datatype.FP16: "float16",
    Datatype.FP32: "float32",
    Datatype.FP64: "float64",
    Datatype.BYTES: "bytes",
}

_NumpyToDatatype = {value: key for key, value in _DatatypeToNumpy.items()}

# NOTE: numpy has more types than v2 protocol
_NumpyToDatatype["object"] = Datatype.BYTES
_NumpyToDatatype["S"] = Datatype.BYTES
_NumpyToDatatype["U"] = Datatype.BYTES


def to_dtype(input_or_output: InputOrOutput) -> "np.dtype":
    dtype = _DatatypeToNumpy[Datatype(input_or_output.datatype)]

    if input_or_output.datatype == "BYTES":
        data = getattr(input_or_output.data, "root", input_or_output.data)
        if is_list_of(data, str):
            # Handle special case of strings being treated as Numpy arrays
            return np.dtype(str)

        if _has_logical_elements(input_or_output) or (
            isinstance(data, np.ndarray) and data.dtype == object
        ):
            # Opaque byte elements (e.g. from the binary tensor
            # transport): each element keeps its exact value and length
            # — no fixed-width truncation, no trailing-NUL stripping.
            # The flag survives container changes (batching converts
            # the object array to a plain list).
            return np.dtype(object)

        # bytes have variable size, so need to specify as part of type
        # TODO: Make elem size variable (and not just the last dimension)
        elem_size = input_or_output.shape[-1]
        return np.dtype((dtype, elem_size))

    return np.dtype(dtype)


def to_datatype(dtype: np.dtype) -> Datatype:
    as_str = str(dtype)

    if as_str not in _NumpyToDatatype:
        # If not present, try with kind
        as_str = getattr(dtype, "kind")

    datatype = _NumpyToDatatype[as_str]

    return datatype


def _to_ndarray(input_or_output: InputOrOutput) -> np.ndarray:
    data = getattr(input_or_output.data, "root", input_or_output.data)
    dtype = to_dtype(input_or_output)

    if Datatype(input_or_output.datatype) == Datatype.BYTES:
        if _has_logical_elements(input_or_output) or (
            isinstance(data, np.ndarray) and data.dtype == object
        ):
            # Logical byte elements: exact values and lengths, with no
            # packed-buffer interpretation.
            return np.array(data, dtype=object)

        if is_list_of(data, bytes):
            # If the inputs is of type `BYTES`, there could be multiple "lists"
            # serialised into multiple buffers.
            # We will deserialise all of them and concatenate them together.
            decoded = [np.frombuffer(buffer, dtype) for buffer in data]
            return np.concatenate(decoded)

    # Zero-copy fast path: an ndarray of the target dtype (from the
    # binary REST path or the native JSON span parser) is reshaped
    # into a view — no element walk, no copy. Safety: the request
    # is consumed once by the model (the response pipeline copies
    # independently, see _encode_data); the batcher's
    # restore_list_inputs() converts ndarrays back to lists before
    # list-based merging; and cross-process dispatch (ZMQ pickle)
    # copies anyway, so the view is only exposed in the direct
    # single-process topology where the request is short-lived.
    # Boundary: repeated decode() calls within one request return
    # views of the same buffer when the backing is WRITABLE (the
    # span parser's fresh buffers), so an in-place mutation is
    # visible to later decodes. Read-only backings (np.frombuffer
    # over request bytes) copy per decode, like list roots —
    # mutations never reach the immutable request bytes.
    # Models that decode, mutate and re-decode the same input are
    # outside the consumed-once contract.
    if isinstance(data, np.ndarray) and data.dtype == dtype:
        result = data.reshape(input_or_output.shape)
        if not result.flags.writeable:
            # A read-only backing (np.frombuffer over immutable
            # request bytes — the binary REST and gRPC raw paths)
            # cannot support in-place preprocessing; copy to match
            # the writable array the list path always produced.
            result = result.copy()
        return result

    if isinstance(data, list):
        # Fast path: a flat list whose length matches the declared shape
        # converts element-wise through fromiter. Anything fromiter cannot
        # consume (nested lists, strings, overflow, unsupported dtypes)
        # falls back to the generic np.array path, so behaviour stays
        # bit-identical for every input the generic path accepts.
        expected = 1
        for dim in input_or_output.shape:
            expected *= dim
        try:
            if expected == len(data):
                # Reject nested data: np.fromiter would silently convert
                # a non-empty inner list to True (bool([False]) is True),
                # changing the wire format for BOOL tensors. Other
                # dtypes self-error on nested items, so only the first
                # element needs the fast check; BOOL must scan every
                # element because fromiter accepts a nested item hiding
                # behind leading scalars ([1, [2]] -> [True, True]).
                if data and isinstance(data[0], (list, tuple)):
                    raise TypeError("nested data not supported by fromiter")
                if dtype == np.bool_ and any(
                    isinstance(value, (list, tuple)) for value in data
                ):
                    raise TypeError("nested data not supported by fromiter")
                return np.fromiter(data, dtype, count=expected)
        except (TypeError, ValueError, OverflowError):
            pass

    return np.array(data, dtype)


# The ndarray-native response pipeline runs at EVERY topology: the
# mode-aware TensorData serializer (python mode keeps the array,
# float32 widened to float64) plus orjson's OPT_SERIALIZE_NUMPY render
# eliminates the Python element list entirely, so there is no
# materialisation work left to relocate — the reason the old topology
# gate existed is gone.
# Byte-identity with the list path is verified by the wire fixtures.


def _encode_data(data: np.ndarray, datatype: Datatype) -> Any:
    # Non-BYTES tensors keep their NumPy representation (flat view) to the wire.
    if datatype == Datatype.BYTES:
        if np.issubdtype(data.dtype, str):
            # Handle special case of a string Numpy array, where the diff elems
            # need to be encoded as well
            as_list = data.flatten().tolist()
            return list(map(encode_str, as_list))

        if np.issubdtype(data.dtype, bytes):
            # `tobytes` is way faster than tolist, although it's harder to serialise
            # and only makes sense for actual bytes inputs (#253).
            # Note that `.tobytes()` will return a single `bytes` payload, thus we
            # need to encapsulate it into a list so that it's compatible.
            return [data.tobytes()]

        # Opaque byte elements (e.g. object arrays): exact values and
        # lengths as an element list.
        return data.flatten().tolist()

    if isinstance(data, np.ma.MaskedArray):
        # Masked arrays keep the legacy list representation: tolist
        # converts masked entries to None, matching the pre-ndarray
        # wire format. An ndarray root would carry the MaskedArray
        # subclass through to the wire encoders, whose tobytes()
        # substitutes the array's fill_value instead of a missing
        # value marker.
        return data.flatten().tolist()

    flat = data.ravel()
    if flat is data or flat.base is not None:
        # ravel returned a view sharing the caller's buffer; copy to
        # detach so post-encode mutation of the model's output (e.g. a
        # model reusing its buffer for the next request) cannot change
        # a response that is still in flight. This fires for every
        # contiguous output (ravel returns a view of itself) — one
        # memcpy per output tensor, the accepted cost of buffer safety.
        flat = flat.copy()
    if np.issubdtype(flat.dtype, np.floating):
        nan_mask = np.isnan(flat)
        if nan_mask.any():
            # Canonicalise noncanonical NaN bit patterns (e.g. 0x7FC00001)
            # to the canonical quiet NaN: mask-assignment with the np.nan
            # scalar stores that constant's bits (a scalar broadcast, not
            # an FPU arithmetic op), so every NaN payload collapses to one
            # bit pattern and bit-exact parity checks hold. In-place is
            # safe: flat is private here (explicit copy above, or ravel's
            # own copy for non-contiguous inputs). Verified by
            # TestBufferMutation.test_nan_payload_does_not_mutate_source.
            flat[nan_mask] = np.nan
    return flat


def convert_nan(val):
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return None

    return val


@register_input_codec
class NumpyCodec(InputCodec):
    """
    Decodes an request input (response output) as a NumPy array.
    """

    ContentType = "np"
    TypeHint = np.ndarray

    @classmethod
    def can_encode(csl, payload: Any) -> bool:
        return isinstance(payload, np.ndarray)

    @classmethod
    def encode_output(cls, name: str, payload: np.ndarray, **kwargs) -> ResponseOutput:
        datatype = to_datatype(payload.dtype)

        shape = inject_batch_dimension(list(payload.shape))

        return ResponseOutput(
            name=name,
            datatype=datatype,
            shape=shape,
            data=_encode_data(payload, datatype),
            parameters=Parameters(content_type=cls.ContentType),
        )

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> np.ndarray:
        return cls.decode_input(response_output)  # type: ignore

    @classmethod
    def encode_input(cls, name: str, payload: np.ndarray, **kwargs) -> RequestInput:
        output = cls.encode_output(name=name, payload=payload)

        return RequestInput(
            name=output.name,
            datatype=output.datatype,
            shape=output.shape,
            data=output.data,
            parameters=Parameters(content_type=cls.ContentType),
        )

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> np.ndarray:
        model_data = _to_ndarray(request_input)

        # TODO: Check if reshape not valid
        return model_data.reshape(request_input.shape)


@register_request_codec
class NumpyRequestCodec(SingleInputRequestCodec):
    """
    Decodes the first input (output) of request (response) as a NumPy array.
    This codec can be useful for cases where the whole payload is a single
    NumPy tensor.
    """

    InputCodec = NumpyCodec
    ContentType = NumpyCodec.ContentType
