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
        would flip the input's payload bits and fail here."""
        from mlserver.codecs.numpy import NumpyCodec, set_ndarray_response_pipeline

        set_ndarray_response_pipeline(True)

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

    def test_nan_payload_default_path_never_mutates_source(self):
        """Default (single-frontend) path: tolist() snapshots without
        touching the source buffer, and the noncanonical NaN passes
        through as a plain Python float (the wire renders null either
        way)."""
        from mlserver.codecs.numpy import NumpyCodec

        noncanonical = struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]
        original = np.array([1.5, noncanonical], dtype=np.float32)

        output = NumpyCodec.encode_output(name="test", payload=original)

        assert original.view(np.uint32)[1] == 0x7FC00001
        root = output.data.root
        assert isinstance(root, list)
        assert root[0] == 1.5
        assert np.isnan(root[1])


class TestRawDtypeMismatch:
    """Raw packing validates dtype matches the declared datatype."""

    def test_int64_array_with_int32_datatype_raises(self):
        """An int64 array with INT32 datatype raises an error."""
        from mlserver.raw import _pack_ndarray
        from mlserver.types import ResponseOutput

        data = np.array([7], dtype=np.int64)  # mismatched
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="INT32",
            data=data,
        )
        # Access the underlying ndarray from TensorData
        raw_data = np.asarray(output.data.root)
        with pytest.raises(struct.error, match="expected INT32"):
            _pack_ndarray(output, raw_data)

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

    def test_fp32_array_with_fp64_datatype_raises(self):
        """An FP32 array with FP64 datatype raises an error."""
        from mlserver.raw import _pack_ndarray
        from mlserver.types import ResponseOutput

        data = np.array([1.5], dtype=np.float32)  # mismatched
        output = ResponseOutput(
            name="test",
            shape=[1],
            datatype="FP64",
            data=data,
        )
        raw_data = np.asarray(output.data.root)
        with pytest.raises(struct.error, match="expected FP64"):
            _pack_ndarray(output, raw_data)


class TestRawBytesSkipsItemsize:
    """BYTES is variable-width: itemsize validation must skip it."""

    def test_bytes_datatype_returns_none_itemsize(self):
        from mlserver.raw import _datatype_to_itemsize

        assert _datatype_to_itemsize("BYTES") is None
        assert _datatype_to_itemsize("FP32") == 4

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
