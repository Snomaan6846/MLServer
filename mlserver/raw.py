import struct

from functools import reduce
from operator import mul

import numpy as np

from .types import Datatype, TensorData
from .codecs.string import encode_str
from .codecs.utils import InputOrOutput
from .codecs.lists import ListElement

_DatatypeToCtype = {
    Datatype.BOOL: "?",
    Datatype.UINT8: "B",
    Datatype.UINT16: "H",
    Datatype.UINT32: "I",
    # 'q'/'Q' (explicitly 8 bytes), not 'l'/'L': struct's long is
    # 4 bytes on some platforms, which would break the pack/unpack
    # symmetry against the declared 8-byte itemsize.
    Datatype.UINT64: "Q",
    Datatype.INT8: "b",
    Datatype.INT16: "h",
    Datatype.INT32: "i",
    Datatype.INT64: "q",
    Datatype.FP16: "e",
    Datatype.FP32: "f",
    Datatype.FP64: "d",
}

_SizeFormat = "<I"


def _tensor_length(shape: list[int]) -> int:
    return reduce(mul, shape, 1)


def _tensor_format(elem: InputOrOutput) -> str:
    size = _tensor_length(elem.shape)
    ctype = _DatatypeToCtype[Datatype(elem.datatype)]
    return f"{size}{ctype}"


def _unpack_bytes(raw: bytes) -> list[bytes]:
    """
    From Triton's implementation:
        https://github.com/triton-inference-server/client/blob/6cc412c50ca4282cec6e9f62b3c2781be433dcc6/src/python/library/tritonclient/utils/__init__.py#L246-L273
    """
    offset = 0
    length = len(raw)
    elems = []
    while offset < length:
        [size] = struct.unpack_from(_SizeFormat, raw, offset)
        offset += 4
        [elem] = struct.unpack_from(f"<{size}s", raw, offset)
        offset += size

        elems.append(elem)

    return elems


def _pack_bytes(unpacked: list[ListElement]) -> bytes:
    packed = []
    for elem in unpacked:
        as_bytes = _ensure_bytes(elem)
        size = len(as_bytes)

        packed_size = struct.pack(_SizeFormat, size)
        packed.append(packed_size)

        packed_elem = struct.pack(f"<{size}s", as_bytes)
        packed.append(packed_elem)

    return b"".join(packed)


def _ensure_bytes(elem: ListElement) -> bytes:
    if isinstance(elem, str):
        return encode_str(elem)

    return elem


def _unpack_tensor(elem: InputOrOutput, raw: bytes) -> list:
    tensor_format = _tensor_format(elem)
    return list(struct.unpack(tensor_format, raw))


def _pack_tensor(elem: InputOrOutput) -> bytes:
    data = getattr(elem.data, "root", elem.data)
    if isinstance(data, np.ndarray):
        packed = _pack_ndarray(elem, data)
        if packed is not None:
            return packed
        # Dtype does not match the declaration: fall through to the
        # legacy checked conversion (struct semantics — numeric
        # coercion for representable values, rejection otherwise),
        # exactly as the list path always behaved.

    tensor_format = _tensor_format(elem)
    if Datatype(elem.datatype) in (Datatype.FP16, Datatype.FP32, Datatype.FP64):
        # Codecs represent NaN as None for JSON; restore it for binary transport.
        values = [float("nan") if value is None else value for value in elem.data]
        return struct.pack(tensor_format, *values)
    return struct.pack(tensor_format, *elem.data)


_DATATYPE_DTYPES = {
    "BOOL": np.bool_,
    "UINT8": np.uint8,
    "INT8": np.int8,
    "UINT16": np.uint16,
    "INT16": np.int16,
    "UINT32": np.uint32,
    "INT32": np.int32,
    "UINT64": np.uint64,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
}


def _pack_ndarray(elem: InputOrOutput, data: np.ndarray) -> "bytes | None":
    """Pack an ndarray root; None when the dtype does not EXACTLY match
    the declared datatype (the caller then uses the legacy checked
    conversion — struct semantics: numeric coercion, rejection of
    non-numerics).

    A width check alone is insufficient: an int32 array under a
    declared FP32 output has the same itemsize but different storage
    semantics (struct coerces 1 -> 1.0f), an object or
    structured dtype can match a numeric width while its bytes are
    pointers/records, and a float array under an integer declaration
    must reject exactly as the struct path does.
    """
    expected_size = _tensor_length(elem.shape)
    if data.size != expected_size:
        # Fail with the same error the list path raises on mismatches.
        raise struct.error(
            f"pack expected {expected_size} items for packing (got {data.size})"
        )
    expected_dtype = _DATATYPE_DTYPES.get(elem.datatype)
    if expected_dtype is None or data.dtype != np.dtype(expected_dtype):
        # Mismatched (or object/structured/foreign) dtype: the legacy
        # conversion decides — never emit raw storage for a
        # declaration it does not exactly match.
        return None
    if isinstance(data, np.ma.MaskedArray):
        # Masked arrays carry a fill value in their underlying buffer:
        # tobytes would emit it (not the NaN the FP conversion
        # produces for masked entries). The legacy path decides.
        return None
    if elem.datatype == "BOOL":
        # A bool array viewed from uint8 can carry noncanonical true
        # bytes (e.g. 02 where the wire requires 01): normalize
        # through astype, which maps any nonzero to 1.
        return data.reshape(-1).astype(np.uint8).tobytes(order="C")
    # Exact match: native byte order (no copy when already native);
    # NaN normalised upstream.
    data = data.reshape(-1).astype(data.dtype.newbyteorder("="), copy=False)
    return data.tobytes(order="C")


def unpack(elem: InputOrOutput, raw: bytes) -> list:
    if elem.datatype == "BYTES":
        return _unpack_bytes(raw)

    return _unpack_tensor(elem, raw)


def pack(elem: InputOrOutput) -> bytes:
    # Unwrap the TensorData root on both paths, the same way
    # _pack_tensor does: the BYTES branch iterates the raw elements
    # (list, object-dtype or bytes-dtype ndarray roots) directly
    # instead of relying on the wrapper's __iter__.
    data = getattr(elem.data, "root", elem.data)
    if elem.datatype == "BYTES":
        return _pack_bytes(data)  # type: ignore

    return _pack_tensor(elem)


def inject_raw(
    elems: list[InputOrOutput], raw_contents: list[bytes]
) -> list[InputOrOutput]:
    raw_idx = 0
    for elem in elems:
        if not elem.data:
            # Only unpack raw entry, if input / output is empty
            # This is to allow for mixed reqs / resp, where some entries have
            # data, some entries are raw
            raw = raw_contents[raw_idx]
            elem.data = TensorData(unpack(elem, raw))  # type: ignore
            raw_idx += 1

    return elems


def extract_raw(elems: list[InputOrOutput]) -> tuple[list[InputOrOutput], list[bytes]]:
    raw_contents = []
    for elem in elems:
        raw = pack(elem)
        raw_contents.append(raw)
        elem.data = []  # type: ignore

    return elems, raw_contents
