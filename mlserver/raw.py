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
        return _pack_ndarray(elem, data)

    tensor_format = _tensor_format(elem)
    if Datatype(elem.datatype) in (Datatype.FP16, Datatype.FP32, Datatype.FP64):
        # Codecs represent NaN as None for JSON; restore it for binary transport.
        values = [float("nan") if value is None else value for value in elem.data]
        return struct.pack(tensor_format, *values)
    return struct.pack(tensor_format, *elem.data)


def _pack_ndarray(elem: InputOrOutput, data: np.ndarray) -> bytes:
    # Contiguous-buffer fast path (native byte order, matching struct exactly).
    expected = _tensor_length(elem.shape)
    if data.size != expected:
        # Fail with the same error the list path raises on mismatches.
        raise struct.error(
            f"pack expected {expected} items for packing (got {data.size})"
        )
    # The array's itemsize must match the declared datatype's width;
    # a mismatch (e.g. int64 array with INT32 datatype) would write
    # the wrong number of bytes per element.
    expected_itemsize = _datatype_to_itemsize(elem.datatype)
    if expected_itemsize is not None and data.dtype.itemsize != expected_itemsize:
        raise struct.error(
            f"pack expected {elem.datatype} ({expected_itemsize} bytes) "
            f"but array dtype is {data.dtype} ({data.dtype.itemsize} bytes)"
        )
    # Native byte order (no copy when already native); NaN normalised upstream.
    data = data.reshape(-1).astype(data.dtype.newbyteorder("="), copy=False)
    return data.tobytes(order="C")


_DATATYPE_ITEMSIZES = {
    "BOOL": 1,
    "UINT8": 1,
    "INT8": 1,
    "UINT16": 2,
    "INT16": 2,
    "FP16": 2,
    "UINT32": 4,
    "INT32": 4,
    "FP32": 4,
    "UINT64": 8,
    "INT64": 8,
    "FP64": 8,
}


def _datatype_to_itemsize(datatype: str) -> int | None:
    # Returns the expected byte width for a datatype, or None for
    # variable-width types (BYTES).
    return _DATATYPE_ITEMSIZES.get(datatype)


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
