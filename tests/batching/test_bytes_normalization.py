"""Direct unit tests for the BYTES batching normalization logic.

These test _bytes_mode, _standalone_elements, and
_normalize_bytes_entries in mlserver/batching/requests.py directly —
the least-covered, most subtle code in the binary tensor feature
(mixed-representation BYTES batching).
"""

from mlserver.batching.requests import (
    _bytes_mode,
    _normalize_bytes_entries,
    _standalone_elements,
)
from mlserver.codecs.numpy import (
    LOGICAL_BYTES_ELEMENTS,
    LOGICAL_BYTES_SENTINEL,
    set_logical_elements,
)
from mlserver.types import Parameters, RequestInput, TensorData


def _input(data, shape=None, datatype="BYTES", parameters=None):
    return RequestInput(
        name="s",
        datatype=datatype,
        shape=shape or [len(data) if isinstance(data, list) else 1, 1],
        data=TensorData(data),
        parameters=parameters,
    )


def _logical_input(elements, shape=None):
    inp = _input(list(elements), shape=shape)
    set_logical_elements(inp)
    return inp


def _string_codec_input(strings, shape=None):
    return _input(
        [s.encode("utf-8") for s in strings],
        shape=shape,
        parameters=Parameters(content_type="str"),
    )


def _legacy_input(buffer, shape):
    return _input([buffer], shape=shape)


class TestBytesMode:
    def test_logical(self):
        inp = _logical_input([b"\xff", b"abc"])
        assert _bytes_mode(list(inp.data.root), inp) == "logical"

    def test_string_codec(self):
        inp = _string_codec_input(["hello"])
        assert _bytes_mode(list(inp.data.root), inp) == "string-codec"

    def test_string(self):
        inp = _input(["hello", "world"])
        assert _bytes_mode(list(inp.data.root), inp) == "string"

    def test_legacy_bytes_list(self):
        inp = _legacy_input(b"\x01\x02", [2, 1])
        assert _bytes_mode(list(inp.data.root), inp) == "legacy"

    def test_legacy_raw_bytes(self):
        inp = _input(b"\x01\x02")
        assert _bytes_mode(inp.data.root, inp) == "legacy"

    def test_other(self):
        inp = _input([1, 2, 3], datatype="FP32")
        assert _bytes_mode(list(inp.data.root), inp) == "other"

    def test_forged_flag_not_logical(self):
        """A JSON-forged list sentinel does not classify as logical."""
        inp = _input(
            [b"\x01\x02"],
            parameters=Parameters(
                **{LOGICAL_BYTES_ELEMENTS: list(LOGICAL_BYTES_SENTINEL)}
            ),
        )
        assert _bytes_mode(list(inp.data.root), inp) == "legacy"


class TestStandaloneElements:
    def test_logical_passthrough(self):
        elements = [b"\xff", b"abc\x00"]
        result = _standalone_elements(
            list(elements), _logical_input(elements), "logical"
        )
        assert result == elements

    def test_string_codec_decodes_to_str(self):
        result = _standalone_elements(
            [b"hello"], _string_codec_input(["hello"]), "string-codec"
        )
        assert result == ["hello"]

    def test_string_passthrough(self):
        result = _standalone_elements(
            ["hello", "world"], _input(["hello", "world"]), "string"
        )
        assert result == ["hello", "world"]

    def test_legacy_splits_by_itemsize_with_nul_stripping(self):
        # b"a\x00bc" with shape [1,2] → itemsize 2 → [b"a", b"bc"]
        result = _standalone_elements(
            [b"a\x00bc"], _legacy_input(b"a\x00bc", [1, 2]), "legacy"
        )
        assert result == [b"a", b"bc"]

    def test_legacy_indivisible_returns_as_is(self):
        # 3 bytes, itemsize 2 → not evenly divisible → keep as-is
        result = _standalone_elements(
            [b"\x01\x02\x03"], _legacy_input(b"\x01\x02\x03", [1, 2]), "legacy"
        )
        assert result == [b"\x01\x02\x03"]

    def test_other_returns_none(self):
        result = _standalone_elements([1, 2], _input([1, 2], datatype="FP32"), "other")
        assert result is None


class TestNormalizeBytesEntries:
    def test_no_logical_flag_returns_unchanged(self):
        """O(1) pre-check: no per-element scan without the flag."""
        entries = [
            (list(inp.data.root), inp)
            for inp in (
                _legacy_input(b"\x01\x02", [2, 1]),
                _legacy_input(b"\x03\x04", [2, 1]),
            )
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert not normalized
        assert data == [e[0] for e in entries]

    def test_all_same_mode_returns_unchanged(self):
        """All-logical (same mode) also takes the direct path."""
        elements_a = [b"\xff", b"abc"]
        elements_b = [b"\x01\x00"]
        entries = [
            (list(elements_a), _logical_input(elements_a)),
            (list(elements_b), _logical_input(elements_b)),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert not normalized
        assert data == [elements_a, elements_b]

    def test_mixed_logical_string_codec(self):
        logical = [b"x\x00"]
        entries = [
            (list(logical), _logical_input(logical)),
            ([b"hello"], _string_codec_input(["hello"])),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert normalized
        assert data == [[b"x\x00"], ["hello"]]

    def test_mixed_logical_legacy_splits(self):
        logical = [b"\xff"]
        entries = [
            (list(logical), _logical_input(logical)),
            ([b"\x01\x02"], _legacy_input(b"\x01\x02", [2, 1])),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert normalized
        assert data == [[b"\xff"], [b"\x01", b"\x02"]]

    def test_mixed_logical_json_string(self):
        logical = [b"x\x00"]
        entries = [
            (list(logical), _logical_input(logical)),
            (["hello"], _input(["hello"])),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert normalized
        assert data == [[b"x\x00"], ["hello"]]

    def test_mixed_logical_other_keeps_raw(self):
        """Unknown representations pass through unmodified."""
        logical = [b"\xff"]
        entries = [
            (list(logical), _logical_input(logical)),
            ([1, 2], _input([1, 2], datatype="FP32")),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert normalized
        assert data == [[b"\xff"], [1, 2]]

    def test_forged_flag_does_not_trigger_normalization(self):
        """A JSON-forged flag doesn't cause the O(n) path or normalization."""
        forged = _input(
            [b"\x01\x02"],
            parameters=Parameters(
                **{LOGICAL_BYTES_ELEMENTS: list(LOGICAL_BYTES_SENTINEL)}
            ),
        )
        legacy = _legacy_input(b"\x03\x04", [2, 1])
        entries = [
            (list(forged.data.root), forged),
            (list(legacy.data.root), legacy),
        ]
        data, normalized = _normalize_bytes_entries(entries)
        assert not normalized  # forged flag not honored → direct path
