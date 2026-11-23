"""ctypes loader for the fastspan JSON span parser.

The C parser (``parser.c``, alongside this module) turns a flat
numeric JSON array span directly into a preallocated NumPy buffer —
no intermediate Python float objects exist, and the parsed values
are bit-identical to the standard path's, enforced by differential
tests against it.

Library state: the compiled ``libfastspan.so`` lives next to this
module — development builds compile it in place, and the image build
(Dockerfile gcc step) compiles it before the wheel is built so the
wheel ships it. A plain pip install from an sdist without the compile
step gets graceful absence: every accessor returns None and the
caller takes today's parsing path unchanged.
"""

import ctypes
from pathlib import Path

import numpy as np

_SONAME = "libfastspan.so"
_lib = None
_loaded = False


def _load() -> "ctypes.CDLL | None":
    """Locate and bind the shared library (idempotent, failure-tolerant)."""
    global _lib, _loaded
    if _loaded:
        return _lib
    lib = None
    for path in [Path(__file__).parent / _SONAME]:
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.fastspan_parse_f32.restype = ctypes.c_long
            lib.fastspan_parse_f32.argtypes = [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            lib.fastspan_parse_f64.restype = ctypes.c_long
            lib.fastspan_parse_f64.argtypes = [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
            ]
            break
        except OSError:
            continue
    # Publish the result BEFORE the loaded flag: a concurrent first
    # call must never see _loaded set with _lib still None (which
    # would cache a spurious absence for the process lifetime). The
    # benign worst case without a lock is a double CDLL open, which
    # resolves to the same handle.
    _lib = lib
    _loaded = True
    return _lib


def parse_span_f32(span: bytes, count: int) -> "np.ndarray | None":
    """Parse a flat FP32 span; None when the native parser is absent
    or the span deviates from the strict grammar (the caller falls
    back to the standard path)."""
    lib = _load()
    if lib is None:
        return None
    out = np.empty(count, dtype=np.float32)
    # The pointer is pinned in a named variable for the duration of
    # the call: on a moving-GC platform (PyPy, a future compacting
    # CPython) an inline temporary could theoretically let the buffer
    # relocate between pointer calculation and the first C write.
    ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    parsed = lib.fastspan_parse_f32(span, len(span), ptr, count)
    if parsed != count:
        return None
    return out


def parse_span_f64(span: bytes, count: int) -> "np.ndarray | None":
    """Parse a flat FP64 span; same contract as parse_span_f32."""
    lib = _load()
    if lib is None:
        return None
    out = np.empty(count, dtype=np.float64)
    ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    parsed = lib.fastspan_parse_f64(span, len(span), ptr, count)
    if parsed != count:
        return None
    return out


_INT_FUNCS = {
    "INT8": ("i8", ctypes.c_int8, np.int8),
    "INT16": ("i16", ctypes.c_int16, np.int16),
    "INT32": ("i32", ctypes.c_int32, np.int32),
    "INT64": ("i64", ctypes.c_int64, np.int64),
    "UINT8": ("u8", ctypes.c_uint8, np.uint8),
    "UINT16": ("u16", ctypes.c_uint16, np.uint16),
    "UINT32": ("u32", ctypes.c_uint32, np.uint32),
    "UINT64": ("u64", ctypes.c_uint64, np.uint64),
}


def parse_span_int(span: bytes, count: int, datatype: str) -> "np.ndarray | None":
    """Parse a flat integer span; None when the native parser is absent
    or the span deviates from the strict grammar (the caller falls
    back to the standard path)."""
    lib = _load()
    if lib is None:
        return None
    entry = _INT_FUNCS.get(datatype)
    if entry is None:
        return None
    suffix, ctype, dtype = entry
    fn = getattr(lib, f"fastspan_parse_{suffix}", None)
    if fn is None:
        return None
    if fn.argtypes is None:
        # Bind once; every call would otherwise repeat the (identical)
        # restype/argtypes mutation on the shared CDLL function.
        fn.restype = ctypes.c_long
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctype),
            ctypes.c_size_t,
        ]
    out = np.empty(count, dtype=dtype)
    ptr = out.ctypes.data_as(ctypes.POINTER(ctype))
    parsed = fn(span, len(span), ptr, count)
    if parsed != count:
        return None
    return out


def parse_span_bool(span: bytes, count: int) -> "np.ndarray | None":
    """Parse a flat BOOL span (true/false literals + numerics)."""
    lib = _load()
    if lib is None:
        return None
    fn = getattr(lib, "fastspan_parse_bool", None)
    if fn is None:
        return None
    if fn.argtypes is None:
        # Bind once; every call would otherwise repeat the (identical)
        # restype/argtypes mutation on the shared CDLL function.
        fn.restype = ctypes.c_long
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
    out = np.empty(count, dtype=np.bool_)
    ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    parsed = fn(span, len(span), ptr, count)
    if parsed != count:
        return None
    return out
