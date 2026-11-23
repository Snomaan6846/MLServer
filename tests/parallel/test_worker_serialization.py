"""Worker-side serialization guards for the shared data plane.

Buffer detachment for in-flight responses, codec fast-path fallbacks
for nested BOOL inputs, raw packing dtype validation, and exact-length
byte elements.
"""

import struct

import numpy as np
import pytest


class TestNestedBoolFromiter:
    """Nested BOOL data bypasses the fromiter fast path."""

    def test_nested_bool_falls_back(self):
        """[[False]] with shape [1,1] preserves False, not True."""
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        request_input = RequestInput(
            name="test",
            shape=[1, 1],
            datatype="BOOL",
            data=[[False]],
        )
        # The fast path (fromiter) would convert [False] to True
        # (bool([False]) is True). The guard must reject nested data
        # so the generic path produces the correct [[False]].
        decoded = NumpyCodec.decode_input(request_input)
        result = decoded.tolist()
        assert result == [[False]], (
            f"Expected [[False]] but got {result} — "
            "nested BOOL fell through to fromiter"
        )

    def test_flat_bool_still_fast(self):
        """Flat [False] with shape [1] uses fromiter correctly."""
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        request_input = RequestInput(
            name="test",
            shape=[1],
            datatype="BOOL",
            data=[False],
        )
        decoded = NumpyCodec.decode_input(request_input)
        assert decoded.tolist() == [False]

    def test_nested_int_falls_back(self):
        """[[1,2]] with shape [1,2] uses the generic path."""
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        request_input = RequestInput(
            name="test",
            shape=[1, 2],
            datatype="INT32",
            data=[[1, 2]],
        )
        decoded = NumpyCodec.decode_input(request_input)
        assert decoded.tolist() == [[1, 2]]


class TestBufferMutation:
    """Encoded data is detached from caller-owned buffers."""

    def test_ravel_returns_copy(self):
        """Encoded output does not share memory with the input array."""
        from mlserver.codecs.numpy import NumpyCodec

        original = np.array([[1.0, 2.0]], dtype=np.float32)
        # Encode through the public codec — internally uses _encode_data
        # which detaches the ravel view from the caller's buffer
        output = NumpyCodec.encode_output(name="test", payload=original)

        # Mutate the original after encoding
        original[0, 0] = 999.0

        # The encoded data should NOT reflect the mutation
        root = output.data.root
        assert root[0] == 1.0, (
            f"Buffer mutation leaked: got {root.tolist()} — " "ravel() view not copied"
        )

    def test_non_contiguous_still_works(self):
        """Non-contiguous arrays (already copied by ravel) still work."""
        from mlserver.codecs.numpy import NumpyCodec

        arr = np.array([[1, 2], [3, 4]], dtype=np.float32).T  # non-contiguous
        output = NumpyCodec.encode_output(name="test", payload=arr)
        assert output.data is not None
        assert len(output.shape) == 2

    def test_sliced_view_input_is_detached(self):
        """A 1-D slice of a larger buffer (ravel view whose base is the
        grandparent array) is detached at encode time."""
        from mlserver.codecs.numpy import NumpyCodec

        pool = np.zeros((4, 4), dtype=np.float32)
        original = pool[1]  # row slice: view of pool
        output = NumpyCodec.encode_output(name="test", payload=original)
        root = output.data.root

        # Mutating the backing pool after encoding must not leak in
        pool[1, 0] = 999.0
        assert root[0] == 0.0, "mutation of the backing buffer leaked"

    def test_flat_1d_input_is_detached(self):
        """A 1-D contiguous array (ravel returns a view of itself) is
        also detached: 1-D is the common shape for model outputs."""
        from mlserver.codecs.numpy import NumpyCodec

        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = NumpyCodec.encode_output(name="test", payload=original)
        root = output.data.root

        original[0] = 999.0
        assert root[0] == 1.0, "mutation of a flat 1-D output leaked"
        assert not np.shares_memory(root, original)

    def test_nan_payload_does_not_mutate_source(self):
        """NaN canonicalisation writes to the detached copy, never to
        the model's output buffer — proven with a noncanonical NaN: an
        encoder that canonicalised the input in place (then copied it)
        would flip the input's payload bits and fail here. (The old
        default-path test pair collapsed into this single-mode
        successor: the ndarray path's detach-copy proves the
        invariant; the wire renders null either way.)"""
        from mlserver.codecs.numpy import NumpyCodec

        # 0x7FC00001: a quiet NaN with a noncanonical payload (np.nan
        # itself is the canonical 0x7FC00000 and would prove nothing).
        noncanonical = struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]
        original = np.array([1.5, noncanonical], dtype=np.float32)

        output = NumpyCodec.encode_output(name="test", payload=original)

        # The input keeps its noncanonical payload bits.
        assert original.view(np.uint32)[1] == 0x7FC00001
        # The detached output carries the canonical quiet NaN.
        root = output.data.root
        assert root.view(np.uint32)[1] == 0x7FC00000


class TestRawDtypeMismatch:
    """Raw packing routes dtype mismatches to the legacy conversion.

    An ndarray whose dtype does not exactly match the declared
    datatype returns None from _pack_ndarray (the fast path declines)
    and the public pack() applies struct semantics: numeric coercion
    for representable values, rejection otherwise.
    """

    def test_int64_array_with_int32_datatype_converts(self):
        """An int64 array with INT32 datatype takes the legacy
        conversion: struct coerces (or raises on range) — never raw
        int64 storage under an INT32 declaration."""
        import struct as struct_module

        from mlserver.raw import _pack_ndarray, pack
        from mlserver.types import ResponseOutput

        data = np.array([7], dtype=np.int64)  # representable in int32
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="INT32",
            data=data,
        )
        raw_data = np.asarray(output.data.root)
        # The fast path declines (dtype mismatch).
        assert _pack_ndarray(output, raw_data) is None
        # The public pack applies the legacy struct conversion.
        assert pack(output) == struct_module.pack("i", 7)

    def test_matching_dtype_succeeds(self):
        """Correct dtype works normally."""
        from mlserver.raw import _pack_ndarray
        from mlserver.types import ResponseOutput

        data = np.array([7], dtype=np.int32)
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="INT32",
            data=data,
        )
        raw_data = np.asarray(output.data.root)
        result = _pack_ndarray(output, raw_data)
        assert len(result) == 4  # 1 element * 4 bytes

    def test_fp32_array_with_fp64_datatype_converts(self):
        """An FP32 array with FP64 datatype takes the legacy
        conversion: struct coerces to double — never raw float32
        storage under an FP64 declaration."""
        import struct as struct_module

        from mlserver.raw import _pack_ndarray, pack
        from mlserver.types import ResponseOutput

        data = np.array([1.5], dtype=np.float32)  # exactly representable
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="FP64",
            data=data,
        )
        raw_data = np.asarray(output.data.root)
        assert _pack_ndarray(output, raw_data) is None
        assert pack(output) == struct_module.pack("d", 1.5)

    def test_out_of_range_rejection(self):
        """A value outside the declared type's range rejects (struct
        semantics — the reference behavior)."""
        import struct as struct_module

        from mlserver.raw import pack
        from mlserver.types import ResponseOutput

        data = np.array([2**63], dtype=np.float64)  # not representable in int64
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="INT64",
            data=data,
        )
        with pytest.raises(struct_module.error):
            pack(output)


class TestRawBytesSkipsItemsize:
    """BYTES is variable-width: width validation must skip it."""

    def test_bytes_datatype_has_no_fixed_width(self):
        """The production width authorities exclude BYTES from every
        fixed-width path: the numpy dtype table (pack-side) has no
        entry, and the struct ctype mapping (unpack-side) cannot
        format it — which is why both routes branch BYTES out first.
        The numeric widths those tables imply come from numpy itself
        (the single source of truth), not a parallel table."""
        import numpy as np

        from mlserver.raw import _DATATYPE_DTYPES, _DatatypeToCtype
        from mlserver.types import Datatype

        assert _DATATYPE_DTYPES.get("BYTES") is None
        assert Datatype.BYTES not in _DatatypeToCtype
        assert np.dtype(_DATATYPE_DTYPES["FP32"]).itemsize == 4

    def test_bytes_packing_not_rejected_by_itemsize(self):
        """A BYTES output with any numpy dtype is not rejected by the
        itemsize check (it has no expected width), and the public pack
        produces the BYTES wire format: one little-endian length
        prefix per element."""
        import numpy as np

        from mlserver.raw import pack
        from mlserver.types import ResponseOutput

        # An object-dtype array (what BYTES typically produces) must
        # not hit the itemsize validation.
        output = ResponseOutput(
            name="test",
            shape=[2],
            datatype="BYTES",
            data=np.array([b"hello", b"world"], dtype=object),
        )
        # The public entry point routes BYTES to the element-wise
        # packing: 4-byte length prefixes around each payload.
        assert pack(output) == b"\x05\x00\x00\x00hello\x05\x00\x00\x00world"
