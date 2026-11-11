"""Zero-copy IPC serialization for inference messages.

Replaces pickle-based serialization when the ``msgpack`` IPC serialization
is enabled.  Messages are encoded as a JSON metadata frame followed by
raw buffer frames.  Large NumPy arrays and byte strings travel as their
own frames (one contiguous copy on encode, zero-copy views on decode),
following the pattern established by vLLM's EngineCore IPC.

Beyond NumPy arrays, the encoder handles the full shape of inference
messages:

- Pydantic models (e.g. ``InferenceRequest``) are rebuilt through an
  import path plus their fields.
- Dataclasses (e.g. the raw REST worker's ``RawReply``) use the same
  ``__class__`` marker mechanism.
- Exceptions (``ModelResponseMessage.exception``) are rebuilt as real
  exception objects so worker failures keep reaching the frontend.
- ``bytes`` payloads (e.g. raw REST bodies) share the inline/frame split
  used for arrays.
- Flat lists of homogeneous Python numbers (tensor payloads reach the IPC
  layer as flattened lists, e.g. from ``NumpyCodec._encode_data``) are
  framed as raw buffers and restored with ``tolist()``, reproducing the
  original values and types exactly.
"""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import orjson
from typing import Any

from pydantic import BaseModel, RootModel

# Markers used in serialized metadata to flag special values
_NDARRAY_MARKER = "__ndarray__"
_BYTES_MARKER = "__bytes__"
_EXCEPTION_MARKER = "__exception__"
_BIGINT_MARKER = "__bigint__"

# Python ints beyond the JSON-metadata int64 range travel as strings
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Buffers below this size (bytes) are inlined into the metadata frame
_INLINE_THRESHOLD = 256

# Flat numeric lists at or above this many elements are framed as buffers
_LIST_FRAME_MIN_ELEMENTS = 64

# Wire dtypes for the Python numeric types that can be framed losslessly
_PYTHON_NUMERIC_DTYPES = {
    float: "<f8",
    int: "<i8",
    bool: "|b1",
}


def _try_frame_numeric_list(obj: list, buffers: list[bytes]) -> dict[str, Any] | None:
    """Frame a flat list of homogeneous Python numbers as a raw buffer.

    Large tensor payloads reach the IPC layer as flat Python lists (e.g.
    ``NumpyCodec._encode_data`` flattens outputs), and walking them
    element-by-element dominates the serialization cost.  When every
    element shares one Python numeric type, the list is sent as a single
    frame and rebuilt with ``tolist()`` on decode, which reproduces the
    original values and types exactly.  Anything else (mixed types,
    strings, nested lists, very small lists) returns None and keeps the
    element-wise walk.
    """
    if len(obj) < _LIST_FRAME_MIN_ELEMENTS:
        return None
    types = set(map(type, obj))
    if len(types) != 1:
        return None
    dtype = _PYTHON_NUMERIC_DTYPES.get(types.pop())
    if dtype is None:
        return None
    try:
        arr = np.asarray(obj, dtype=dtype)
    except (OverflowError, ValueError, TypeError):
        # e.g. ints beyond the int64 range — fall back to the walk
        return None
    buffers.append(arr.tobytes())
    return {
        _NDARRAY_MARKER: True,
        "inline": False,
        "frame": len(buffers) - 1,
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "as_list": True,
    }


def _to_serializable(obj: Any, buffers: list[bytes]) -> Any:
    """Recursively convert an object to a JSON-serializable structure.

    NumPy arrays and byte strings are either inlined (small) or extracted
    into the ``buffers`` list (large) and replaced with frame references.
    Pydantic models and dataclasses are converted to dicts with a
    ``__class__`` marker so they can be rebuilt on decode.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.nbytes < _INLINE_THRESHOLD:
            return {
                _NDARRAY_MARKER: True,
                "inline": True,
                "dtype": obj.dtype.str,
                "shape": list(obj.shape),
                "data": obj.tobytes().hex(),
            }
        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        buffers.append(obj.tobytes())
        return {
            _NDARRAY_MARKER: True,
            "inline": False,
            "frame": len(buffers) - 1,
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
        }
    if isinstance(obj, (bytes, bytearray)):
        raw = bytes(obj)
        if len(raw) < _INLINE_THRESHOLD:
            return {_BYTES_MARKER: True, "inline": True, "data": raw.hex()}
        buffers.append(raw)
        return {_BYTES_MARKER: True, "inline": False, "frame": len(buffers) - 1}
    if isinstance(obj, np.generic):
        # NumPy scalars round-trip as plain Python scalars.
        return obj.item()
    if isinstance(obj, BaseException):
        return {
            _EXCEPTION_MARKER: True,
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "msg": str(obj),
        }
    if isinstance(obj, RootModel):
        # RootModel subclasses (like TensorData) wrap a single value.
        # Serialize the root value directly — the field's type annotation
        # will handle reconstruction on decode.
        return _to_serializable(obj.root, buffers)
    if isinstance(obj, BaseModel):
        result: dict[str, Any] = {
            "__class__": f"{type(obj).__module__}.{type(obj).__qualname__}"
        }
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            result[name] = _to_serializable(value, buffers)
        return result
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {"__class__": f"{type(obj).__module__}.{type(obj).__qualname__}"}
        for field in dataclasses.fields(obj):
            result[field.name] = _to_serializable(getattr(obj, field.name), buffers)
        return result
    if isinstance(obj, dict):
        return {k: _to_serializable(v, buffers) for k, v in obj.items()}
    if isinstance(obj, list):
        framed = _try_frame_numeric_list(obj, buffers)
        if framed is not None:
            return framed
        return [_to_serializable(item, buffers) for item in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(item, buffers) for item in obj]
    if isinstance(obj, int) and not isinstance(obj, bool):
        if not _INT64_MIN <= obj <= _INT64_MAX:
            # orjson rejects ints beyond 64-bit; keep pickle-path parity
            return {_BIGINT_MARKER: True, "value": str(obj)}
        return obj
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # Best-effort for anything else
    return str(obj)


def _import_class(class_path: str) -> type | None:
    """Resolve a fully-qualified class path, returning None on failure."""
    if not class_path:
        return None
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)
    except (ValueError, AttributeError, ImportError):
        return None


def _validate_model(cls: Any, fields: dict[str, Any]) -> Any:
    """Rebuild a Pydantic model or dataclass from restored fields."""
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return cls.model_validate(fields)
    if isinstance(cls, type) and dataclasses.is_dataclass(cls):
        return cls(**fields)
    return fields


def _restore_exception(obj: dict[str, Any]) -> BaseException:
    """Rebuild an exception, falling back to RuntimeError with its message."""
    message = obj.get("msg", "")
    cls = _import_class(obj.get("class", ""))
    if isinstance(cls, type) and issubclass(cls, BaseException):
        try:
            return cls(message)
        except Exception:
            pass
    return RuntimeError(message)


def _restore_arrays(obj: Any, frames: list[bytes] | list[memoryview]) -> Any:
    """Recursively restore NumPy arrays, byte strings, models and exceptions."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if obj.get(_NDARRAY_MARKER):
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            if obj.get("inline"):
                raw = bytes.fromhex(obj["data"])
                arr = np.frombuffer(raw, dtype=dtype)
            else:
                frame = frames[obj["frame"]]
                arr = np.frombuffer(frame, dtype=dtype)
            arr = arr.reshape(shape)
            if obj.get("as_list"):
                # Framed numeric lists must come back as lists (not arrays)
                # so downstream JSON encoding and type semantics are
                # unchanged compared to the element-wise walk.
                return arr.tolist()
            return arr
        if obj.get(_BYTES_MARKER):
            if obj.get("inline"):
                return bytes.fromhex(obj["data"])
            return bytes(frames[obj["frame"]])
        if obj.get(_EXCEPTION_MARKER):
            return _restore_exception(obj)
        if obj.get(_BIGINT_MARKER):
            return int(obj["value"])
        # Check for Pydantic model / dataclass marker
        class_path = obj.get("__class__")
        if class_path:
            cls = _import_class(class_path)
            # Restore fields, excluding the marker
            fields = {
                k: _restore_arrays(v, frames)
                for k, v in obj.items()
                if k != "__class__"
            }
            return _validate_model(cls, fields)
        return {k: _restore_arrays(v, frames) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(item, frames) for item in obj]
    return obj


def encode_message(message: Any) -> list[bytes]:
    """Encode a message into multipart frames for zero-copy transport.

    Returns a list where:
    - Frame 0: JSON metadata (message structure with buffer placeholders)
    - Frames 1+: raw buffer data (one per large NumPy array or byte string)
    """
    buffers: list[bytes] = []
    metadata = _to_serializable(message, buffers)
    return [orjson.dumps(metadata), *buffers]


def decode_message(
    frames: list[bytes] | list[memoryview], message_cls: type[BaseModel]
) -> Any:
    """Decode multipart frames back into a message.

    Args:
        frames: ZMQ multipart frames (metadata + buffer data).  Received
            frames may be ``bytes`` or zero-copy ``memoryview`` objects.
        message_cls: The class to reconstruct (e.g. ModelRequestMessage)

    Returns:
        The reconstructed message with NumPy arrays and nested models
        restored.
    """
    if not frames:
        raise ValueError("Empty frames list")

    metadata = orjson.loads(frames[0])
    tensor_frames = list(frames[1:])
    restored = _restore_arrays(metadata, tensor_frames)

    # If _restore_arrays already reconstructed a message (via __class__
    # markers), return it directly. Otherwise validate against message_cls.
    if isinstance(restored, message_cls):
        return restored
    return message_cls.model_validate(restored)


def decode_request(frames: list[bytes] | list[memoryview]):
    """Decode multipart frames into a ModelRequestMessage."""
    from .messages import ModelRequestMessage

    return decode_message(frames, ModelRequestMessage)


def decode_response(frames: list[bytes] | list[memoryview]):
    """Decode multipart frames into a ModelResponseMessage."""
    from .messages import ModelResponseMessage

    return decode_message(frames, ModelResponseMessage)
