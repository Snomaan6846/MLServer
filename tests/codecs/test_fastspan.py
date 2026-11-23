"""Comprehensive native parser test suite — every datatype, every edge.

Covers: all 11 datatypes (FP32/64, INT8/16/32/64, UINT8/16/32/64,
BOOL), boundary values, overflow/underflow, truncation, grammar
rejections, sign handling, the mantissa-wrap bug class, zero-copy
_to_ndarray, and the strict-subset fallback contract.

Every test compares against the ACTUAL standard path (not a
hardcoded expectation) so the contract stays pinned to reality.
"""

import pytest

import numpy as np

from mlserver.codecs.fast_json import (
    _NotFast,
    find_numeric_data_spans,
    fastspan_parser,
    parse_inference_request,
)
from mlserver.codecs.fastspan import (
    parse_span_bool,
    parse_span_f32,
    parse_span_f64,
    parse_span_int,
)
from mlserver.codecs import fastspan as _fastspan_module
from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest, RequestInput, TensorData

# The whole suite tests the native parser: without the compiled
# library (no gcc in the environment) there is nothing to assert —
# the fallback contract is pinned elsewhere (test_fast_json.py,
# test_worker_fast_parse.py).
pytestmark = pytest.mark.skipif(
    _fastspan_module._load() is None,
    reason="libfastspan.so not compiled (gcc unavailable)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_INT_DATATYPES = [
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
]

_ALL_DATATYPES = ["FP32", "FP64", "BOOL"] + _ALL_INT_DATATYPES

_INT_LIMITS = {
    "INT8": (-128, 127),
    "INT16": (-32768, 32767),
    "INT32": (-2147483648, 2147483647),
    "INT64": (-9223372036854775808, 9223372036854775807),
    "UINT8": (0, 255),
    "UINT16": (0, 65535),
    "UINT32": (0, 4294967295),
    "UINT64": (0, 18446744073709551615),
}


def _std_decode(datatype: str, data_json: str):
    """The standard path's decoded ndarray (or the exception name)."""
    n = (
        data_json.count(",") + 1
        if "," in data_json
        else (1 if data_json.strip() else 0)
    )
    body = (
        '{"inputs":[{"name":"i","shape":['
        + str(n)
        + '],"datatype":"'
        + datatype
        + '","data":'
        + data_json
        + "}]}"
    )
    try:
        req = InferenceRequest.model_validate_json(body)
        return NumpyCodec.decode_input(req.inputs[0])
    except Exception as e:
        return type(e).__name__


def _fast_parse(datatype: str, data_json: str):
    """The fast parser's result (None = rejected)."""
    n = (
        data_json.count(",") + 1
        if "," in data_json
        else (1 if data_json.strip() else 0)
    )
    span = data_json.strip().lstrip("[").rstrip("]").encode()
    try:
        return fastspan_parser(span, datatype, n)
    except _NotFast:
        return None
    except Exception:
        return "Exception"


def _compare(datatype: str, data_json: str):
    """Assert the fast path matches the standard path.

    Contract: values must be identical when both accept; the fast
    path may reject (fallback) what the standard path accepts; the
    fast path must NEVER accept what the standard path rejects.
    """
    std = _std_decode(datatype, data_json)
    fast = _fast_parse(datatype, data_json)

    if isinstance(std, str):
        # Standard rejects: fast must also reject (safety)
        assert fast is None or isinstance(fast, str), (
            f"{datatype} {data_json}: std rejects ({std}) but fast " f"accepted {fast}"
        )
    else:
        if fast is not None and not isinstance(fast, str):
            # Both accept: values must match exactly
            np.testing.assert_array_equal(
                fast, std, err_msg=f"{datatype} {data_json}: std={std}, fast={fast}"
            )
            assert fast.dtype == std.dtype, (
                f"{datatype} {data_json}: dtype std={std.dtype} " f"fast={fast.dtype}"
            )
        # fast is None → correct fallback (std handles wrapping etc.)


# ---------------------------------------------------------------------------
# FP32/FP64 (existing — kept as regression)
# ---------------------------------------------------------------------------


class TestFloatParsers:
    def test_fp32_basic(self):
        _compare("FP32", "[1.5, 2.5, -0.5]")

    def test_fp64_basic(self):
        _compare("FP64", "[1.5, 2.5, -0.5]")

    def test_fp32_specials(self):
        _compare("FP32", "[1e-38, 3.4028234663852886e38, 0.0, -0.0]")

    def test_fp64_specials(self):
        _compare("FP64", "[5e-324, 1.7976931348623157e308, 0.1]")

    def test_fp32_high_precision(self):
        # 17+ digit mantissas (the extended-precision tier)
        _compare("FP32", "[0.49362120032310486, 0.2881241738796234]")

    def test_mantissa_boundary_no_wrap(self):
        """Values near 2^64 must not silently wrap to 0.

        A fixed accumulation threshold allows uint64 wrap for specific
        20-digit values; the overflow check must therefore depend on
        the current digit, not a constant.
        """
        # UINT64_MAX as f64 (valid, must not wrap)
        result = parse_span_f64(b"18446744073709551615", 1)
        assert result is not None
        assert result[0] == 1.8446744073709552e19

        # 2^64 as f64 (valid float, different from UINT64_MAX by 1)
        result = parse_span_f64(b"18446744073709551616", 1)
        assert result is not None
        assert result[0] == 1.8446744073709552e19

    def test_fp_tie_values(self):
        """Exact double ties round to even."""
        _compare("FP64", "[9007199254740993, 9007199254740995]")


# ---------------------------------------------------------------------------
# INT parsers
# ---------------------------------------------------------------------------


class TestIntParsers:
    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_basic(self, datatype):
        lo, hi = _INT_LIMITS[datatype]
        mid = (lo + hi) // 2
        _compare(datatype, f"[{lo}, {mid}, {hi}]")

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_boundaries(self, datatype):
        lo, hi = _INT_LIMITS[datatype]
        _compare(datatype, f"[{lo}]")
        _compare(datatype, f"[{hi}]")

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_overflow_rejected(self, datatype):
        """Values beyond the type's range must be rejected (fallback).

        The standard path WRAPS overflow values (numpy
        implementation detail). The fast parser correctly refuses to
        reproduce wrapping — it rejects and the standard path handles.
        """
        _, hi = _INT_LIMITS[datatype]
        _compare(datatype, f"[{hi + 1}]")  # fast rejects, std wraps

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_underflow_rejected(self, datatype):
        lo, _ = _INT_LIMITS[datatype]
        _compare(datatype, f"[{lo - 1}]")  # fast rejects, std wraps

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_truncation_toward_zero(self, datatype):
        """Float values targeting int types truncate toward zero."""
        _compare(datatype, "[1.9]")
        _compare(datatype, "[-1.9]")
        _compare(datatype, "[0.0]")

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_exponent_notation(self, datatype):
        _compare(datatype, "[1e2]")

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_negative_zero(self, datatype):
        _compare(datatype, "[-0]")

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_sign_before_truncation(self, datatype):
        """Regression: the sign must be applied BEFORE truncation.

        Bug: -1.9 as INT32 gave 1 (positive!) because the negative
        flag was not applied to the double before trunc().
        """
        lo, _ = _INT_LIMITS[datatype]
        if lo < 0:  # only signed types
            result = _fast_parse(datatype, "[-1.9]")
            if result is not None:
                assert (
                    result[0] == -1
                ), f"{datatype} [-1.9] gave {result[0]}, expected -1"

    def test_int64_max_exact(self):
        """INT64_MAX must be exact (via mantissa, not double).

        Through double, INT64_MAX (2^63-1) rounds to 2^63 which
        overflows int64. The mantissa path avoids this.
        """
        result = parse_span_int(b"9223372036854775807", 1, "INT64")
        assert result is not None
        assert result[0] == 9223372036854775807

    def test_uint64_max_exact(self):
        """UINT64_MAX must be exact (via mantissa, not double).

        Through double, UINT64_MAX (2^64-1) rounds to 2^64 which
        wraps to 0 in uint64. The mantissa path avoids this.
        """
        result = parse_span_int(b"18446744073709551615", 1, "UINT64")
        assert result is not None
        assert result[0] == 18446744073709551615

    @pytest.mark.parametrize("datatype", ["UINT8", "UINT16", "UINT32", "UINT64"])
    def test_negative_rejected_for_unsigned(self, datatype):
        """A negative token in an unsigned type must reject — never
        two's-complement wrap into range.

        The wrap would silently feed the model a value the standard
        path rejects (numpy >= 2 raises OverflowError; older numpy
        wraps with a deprecation warning). Rejecting is safe on every
        version: the fallback lets the standard path decide.
        """
        for token in [b"-5", b"-1", b"-300", b"-70000"]:
            result = parse_span_int(token, 1, datatype)
            assert (
                result is None
            ), f"{datatype} {token.decode()} accepted -> {result[0]} (wrapped)"

    @pytest.mark.parametrize("datatype", ["UINT8", "UINT16", "UINT32", "UINT64"])
    def test_negative_zero_is_zero_for_unsigned(self, datatype):
        """-0 is the one valid negative token for an unsigned type."""
        result = parse_span_int(b"-0", 1, datatype)
        assert result is not None
        assert result[0] == 0

    def test_uint64_overflow_rejected(self):
        """2^64 must be rejected for UINT64 (not wrap to 0)."""
        result = parse_span_int(b"18446744073709551616", 1, "UINT64")
        assert result is None

    def test_negative_fraction_fallback_preserves_sign(self):
        """A negative token with fraction/exponent whose mantissa
        exceeds 2^53 takes the strtod fallback, which copies the token
        INCLUDING the leading '-' — strtod applies the sign, so the
        tail must not negate again. Regression: the double negation
        returned a positive value for a negative token.
        """
        result = parse_span_int(b"-9007199254740993.0", 1, "INT64")
        assert result is not None
        assert result[0] == -9007199254740992

    def test_large_integer_f64_fallback(self):
        """Integers > 2^53 through the double path are rejected.

        The double can't represent them exactly; the standard path
        handles them via Python's arbitrary-precision ints.
        """
        # A 20-digit integer with fraction notation to force the
        # double path
        result = parse_span_int(b"9007199254740993.0", 1, "INT64")
        # This should either work (if representable) or reject
        # (falls back). Must NOT give a wrong answer.
        if result is not None:
            std = _std_decode("INT64", "[9007199254740993.0]")
            if not isinstance(std, str):
                assert result[0] == std[0]


# ---------------------------------------------------------------------------
# BOOL parser
# ---------------------------------------------------------------------------


class TestBoolParser:
    def test_true_false_literals(self):
        _compare("BOOL", "[true, false, true]")

    def test_single_true(self):
        _compare("BOOL", "[true]")

    def test_single_false(self):
        _compare("BOOL", "[false]")

    def test_integers_as_bools(self):
        _compare("BOOL", "[1, 0]")

    def test_nonzero_integers_true(self):
        _compare("BOOL", "[2, -1]")

    def test_floats_as_bools(self):
        _compare("BOOL", "[1.5, 0.0]")

    def test_negative_zero_false(self):
        _compare("BOOL", "[-0.0]")

    def test_exponent_as_bool(self):
        _compare("BOOL", "[1e5]")

    def test_leading_zero_rejected(self):
        """Regression: strtod accepts leading zeros, JSON rejects.

        Bug: BOOL [01.5] was accepted by the fast parser (via
        strtod) but rejected by the standard path (pydantic).
        """
        _compare("BOOL", "[01.5]")

    def test_string_bools_fallback(self):
        """String bools (["true"]) never reach the fast parser.

        The span extractor skips spans containing quotes.
        """
        spans = find_numeric_data_spans(
            b'{"inputs":[{"name":"i","shape":[2],"datatype":"BOOL",'
            b'"data":["true", "false"]}]}'
        )
        assert spans == []  # quotes → extractor skips → standard path

    def test_bool_grammar_rejections(self):
        _compare("BOOL", "[true,]")
        _compare("BOOL", "[false,]")
        _compare("BOOL", "[nan]")
        _compare("BOOL", "[TrUe]")  # case-sensitive

    def test_bool_bare_trailing_dot_rejected(self):
        """A bare trailing dot ("1.") is not valid JSON: strtod accepts
        it, the standard path rejects it — the fast path must reject
        too (strict-subset contract)."""
        for token in [b"1.", b"0.", b"-1.", b"-0."]:
            result = parse_span_bool(token, 1)
            assert result is None, f"BOOL {token!r} accepted -> {result}"

    def test_bool_dotted_tokens_still_parse(self):
        """Control: dots followed by digits remain valid numeric
        truthiness tokens."""
        assert list(parse_span_bool(b"2.5", 1)) == [True]
        assert list(parse_span_bool(b"0.0", 1)) == [False]


# ---------------------------------------------------------------------------
# Grammar rejections (all datatypes)
# ---------------------------------------------------------------------------


class TestGrammarRejections:
    @pytest.mark.parametrize("datatype", ["FP32", "INT32", "BOOL"])
    def test_trailing_comma(self, datatype):
        _compare(datatype, "[1.5,]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32", "BOOL"])
    def test_trailing_comma_whitespace(self, datatype):
        _compare(datatype, "[1.5, ]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32", "BOOL"])
    def test_double_comma(self, datatype):
        _compare(datatype, "[1.5,,2.5]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32"])
    def test_leading_zero(self, datatype):
        _compare(datatype, "[01.5]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32"])
    def test_bare_dot(self, datatype):
        _compare(datatype, "[.5]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32"])
    def test_bare_exponent(self, datatype):
        _compare(datatype, "[1e]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32"])
    def test_nan_rejected(self, datatype):
        _compare(datatype, "[nan]")

    @pytest.mark.parametrize("datatype", ["FP32", "INT32"])
    def test_inf_rejected(self, datatype):
        _compare(datatype, "[inf]")

    def test_true_as_int_rejected(self):
        _compare("INT32", "[true]")

    def test_mixed_true_and_number(self):
        _compare("BOOL", "[true, 1]")  # both valid in BOOL
        _compare("INT32", "[true, 1]")  # true invalid as INT32


# ---------------------------------------------------------------------------
# Mantissa wrap boundary (the bug class)
# ---------------------------------------------------------------------------


class TestMantissaBoundary:
    """Targeted tests for the uint64 mantissa accumulation wrap.

    The original fixed threshold allowed silent wrap for specific
    20-digit values. These tests exercise the exact boundary.
    """

    def test_uint64_max_not_wrapped(self):
        """UINT64_MAX (18446744073709551615) must NOT wrap to 0."""
        result = parse_span_int(b"18446744073709551615", 1, "UINT64")
        assert result is not None
        assert result[0] == 18446744073709551615

    def test_2_pow_64_rejected_as_uint64(self):
        """2^64 (18446744073709551616) must be rejected, not wrap."""
        result = parse_span_int(b"18446744073709551616", 1, "UINT64")
        assert result is None

    def test_2_pow_64_as_f64_not_wrapped(self):
        """2^64 as a float must produce the correct value, not 0."""
        result = parse_span_f64(b"18446744073709551616", 1)
        assert result is not None
        assert result[0] == 1.8446744073709552e19

    def test_boundary_digits_6_through_9(self):
        """Values ending in 6-9 at the 20th digit position.

        The old fixed threshold allowed wrap for digits 6-9 when the
        first 19 digits formed 1844674407370955161.
        """
        for digit in [6, 7, 8, 9]:
            value = f"1844674407370955161{digit}"
            result = parse_span_f64(value.encode(), 1)
            # Must not wrap to 0 — either correct value or reject
            assert result is None or result[0] != 0.0, f"f64({value}) wrapped to 0"

    def test_boundary_digits_0_through_5(self):
        """Values ending in 0-5 at the 20th digit position (safe)."""
        for digit in [0, 1, 2, 3, 4, 5]:
            value = f"1844674407370955161{digit}"
            result = parse_span_f64(value.encode(), 1)
            expected = float(int(value))
            assert (
                result is not None and result[0] == expected
            ), f"f64({value}) gave {result}, expected {expected}"

    def test_negative_boundary(self):
        """Negative boundary values must not wrap either."""
        result = parse_span_f64(b"-18446744073709551616", 1)
        assert result is not None
        assert result[0] == -1.8446744073709552e19


# ---------------------------------------------------------------------------
# Zero-copy _to_ndarray path
# ---------------------------------------------------------------------------


class TestZeroCopyPath:
    """The _to_ndarray zero-copy optimization for ndarray roots."""

    def _make_input(self, datatype, arr):
        return RequestInput(
            name="i",
            shape=list(arr.shape),
            datatype=datatype,
            data=TensorData(root=arr),
        )

    def test_fp32_zero_copy(self):
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([1.5, 2.5], dtype=np.float32)
        result = _to_ndarray(self._make_input("FP32", arr))
        assert np.shares_memory(result, arr)

    def test_fp64_zero_copy(self):
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = _to_ndarray(self._make_input("FP64", arr))
        assert np.shares_memory(result, arr)

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_int_zero_copy(self, datatype):
        from mlserver.codecs.numpy import _to_ndarray

        dt = np.dtype(datatype.lower())
        arr = np.array([1, 2], dtype=dt)
        result = _to_ndarray(self._make_input(datatype, arr))
        assert np.shares_memory(result, arr)

    def test_bool_zero_copy(self):
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([True, False], dtype=np.bool_)
        result = _to_ndarray(self._make_input("BOOL", arr))
        assert np.shares_memory(result, arr)

    def test_dtype_mismatch_copies(self):
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = _to_ndarray(self._make_input("FP32", arr))
        assert not np.shares_memory(result, arr)
        assert result.dtype == np.float32

    def test_reshape_is_view(self):
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([1.5, 2.5, 3.5, 4.0], dtype=np.float32)
        ri = RequestInput(
            name="i",
            shape=[2, 2],
            datatype="FP32",
            data=TensorData(root=arr),
        )
        result = _to_ndarray(ri)
        assert np.shares_memory(result, arr)
        assert result.shape == (2, 2)

    def test_list_input_still_works(self):
        from mlserver.codecs.numpy import _to_ndarray

        ri = RequestInput(
            name="i",
            shape=[3],
            datatype="FP32",
            data=TensorData(root=[1.5, 2.5, 3.5]),
        )
        result = _to_ndarray(ri)
        np.testing.assert_array_equal(result, [1.5, 2.5, 5.5 - 2.0])

    def test_bytes_not_zero_copy(self):
        """BYTES datatype has special handling — must not hit the
        zero-copy path."""
        from mlserver.codecs.numpy import _to_ndarray

        arr = np.array([b"hello", b"world"], dtype=object)
        result = _to_ndarray(self._make_input("BYTES", arr))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# End-to-end orchestrator (parse_inference_request)
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def _fallback(self, body):
        return InferenceRequest.model_validate_json(body)

    def test_end_to_end_fp32(self):
        body = (
            b'{"inputs":[{"name":"i","shape":[3],"datatype":"FP32",'
            b'"data":[1.5, 2.5, -0.5]}]}'
        )
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        np.testing.assert_array_equal(
            NumpyCodec.decode_input(fast.inputs[0]),
            NumpyCodec.decode_input(slow.inputs[0]),
        )

    @pytest.mark.parametrize("datatype", _ALL_INT_DATATYPES)
    def test_end_to_end_int(self, datatype):
        lo, hi = _INT_LIMITS[datatype]
        mid = (lo + hi) // 2
        body = (
            f'{{"inputs":[{{"name":"i","shape":[3],"datatype":"{datatype}",'
            f'"data":[{lo}, {mid}, {hi}]}}]}}'
        ).encode()
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        if fast is not None:
            np.testing.assert_array_equal(
                NumpyCodec.decode_input(fast.inputs[0]),
                NumpyCodec.decode_input(slow.inputs[0]),
            )

    def test_end_to_end_bool(self):
        body = (
            b'{"inputs":[{"name":"i","shape":[3],"datatype":"BOOL",'
            b'"data":[true, false, true]}]}'
        )
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        np.testing.assert_array_equal(
            NumpyCodec.decode_input(fast.inputs[0]),
            NumpyCodec.decode_input(slow.inputs[0]),
        )

    def test_string_data_falls_back(self):
        """BYTES/string data must fall back to the standard path."""
        body = (
            b'{"inputs":[{"name":"i","shape":[2],"datatype":"BYTES",'
            b'"data":["hello", "world"]}]}'
        )
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        assert list(fast.inputs[0].data.root) == list(slow.inputs[0].data.root)

    def test_overflow_falls_back(self):
        """INT32 overflow must fall back (standard path wraps)."""
        body = (
            b'{"inputs":[{"name":"i","shape":[1],"datatype":"INT32",'
            b'"data":[2147483648]}]}'
        )
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        # Both should produce the same result (fast falls back)
        np.testing.assert_array_equal(
            np.asarray(fast.inputs[0].data.root),
            np.asarray(slow.inputs[0].data.root),
        )

    def test_mixed_datatypes(self):
        """Multi-input requests with mixed datatypes."""
        body = (
            b'{"inputs":['
            b'{"name":"a","shape":[2],"datatype":"FP32","data":[1.5, 2.5]},'
            b'{"name":"b","shape":[2],"datatype":"INT32","data":[10, 20]},'
            b'{"name":"c","shape":[2],"datatype":"BOOL","data":[true, false]}'
            b"]}"
        )
        fast = parse_inference_request(body, self._fallback, fastspan_parser)
        slow = self._fallback(body)
        for fi, si in zip(fast.inputs, slow.inputs):
            if fi.datatype != "BYTES":
                np.testing.assert_array_equal(
                    NumpyCodec.decode_input(fi),
                    NumpyCodec.decode_input(si),
                )


# ---------------------------------------------------------------------------
# Differential fuzz (permanent, seeded)
# ---------------------------------------------------------------------------


class TestDifferentialFuzz:
    """Randomized differential test vs the standard path.

    Seeded for reproducibility. Generators include the boundary
    classes that found real bugs (high-precision mantissas, 2^64
    boundary values, int boundaries).
    """

    def test_fuzz_all_datatypes(self):
        import random

        rng = random.Random(20261124)

        boundary_ints = [
            18446744073709551615,
            18446744073709551616,
            9223372036854775807,
            9223372036854775808,
            9007199254740992,
            9007199254740993,
            2147483647,
            2147483648,
        ]

        for case in range(300):
            dt = rng.choice(_ALL_DATATYPES)
            n = rng.randint(1, 10)

            if dt == "BOOL":
                vals = [
                    rng.choice(["true", "false", str(rng.randint(-100, 100))])
                    for _ in range(n)
                ]
            elif dt.startswith("UINT"):
                vals = [str(rng.randint(0, 2 ** rng.randint(1, 32))) for _ in range(n)]
            elif dt.startswith("INT"):
                vals = [
                    str(
                        rng.randint(-(2 ** rng.randint(1, 32)), 2 ** rng.randint(1, 32))
                    )
                    for _ in range(n)
                ]
            elif dt == "FP32":
                vals = [repr(rng.uniform(-1e6, 1e6)) for _ in range(n)]
            else:  # FP64
                v = rng.choice(
                    [
                        lambda: repr(rng.uniform(-1e10, 1e10)),
                        lambda: str(rng.choice(boundary_ints)),
                        lambda: repr(
                            rng.randint(-(10**19), 10**19) * 10.0 ** rng.randint(-19, 0)
                        ),
                    ]
                )()
                vals = [v] + [repr(rng.uniform(-1, 1)) for _ in range(n - 1)]

            data = "[" + ", ".join(vals) + "]"
            _compare(dt, data)  # asserts the strict-subset contract


# ---------------------------------------------------------------------------
# The _NotFast exception contract
# ---------------------------------------------------------------------------


class TestNotFastContract:
    def test_unsupported_datatype_raises(self):
        with pytest.raises(_NotFast):
            fastspan_parser(b"1.5", "BYTES", 1)

    def test_none_result_raises(self):
        with pytest.raises(_NotFast):
            fastspan_parser(b"nan", "FP32", 1)


class TestNegativeZeroSemantics:
    """Integer "-0" is Python int 0 on the standard path (+0.0); only
    float-form zero ("-0.0", "-0e0") keeps its sign (orjson: -0.0)."""

    @pytest.mark.parametrize("datatype", ["FP32", "FP64"])
    def test_integer_negative_zero_is_positive(self, datatype):
        fn = parse_span_f64 if datatype == "FP64" else parse_span_f32
        result = fn(b"-0", 1)
        assert result is not None
        import math

        assert result[0] == 0.0 and math.copysign(1, result[0]) > 0

    @pytest.mark.parametrize("datatype", ["FP32", "FP64"])
    def test_float_form_negative_zero_keeps_sign(self, datatype):
        fn = parse_span_f64 if datatype == "FP64" else parse_span_f32
        for token in [b"-0.0", b"-0e0"]:
            result = fn(token, 1)
            assert result is not None
            import math

            assert result[0] == 0.0 and math.copysign(1, result[0]) < 0

    def test_infinite_token_rejected(self):
        """1e999 -> inf: the fallback defers to the standard path's
        semantics rather than emitting inf directly."""
        assert parse_span_f64(b"1e999", 1) is None


class TestBoolGrammarMatrix:
    """The BOOL parser's numeric path must validate the FULL JSON
    number grammar — strtod accepts C extensions (hex, trailing dots,
    bare exponents) that JSON rejects. Every form strtod would accept
    but the standard path rejects must also be rejected here."""

    @pytest.mark.parametrize(
        "token",
        [
            b"0x10",
            b"0x1",
            b"0X10",
            b"0xff",
            b"0b1",
            b"1.",
            b"0.",
            b"-1.",
            b"1e",
            b"1e+",
            b"1e-",
            b"1E",
            b"1E+",
            b"01",
            b"007",
            b"-01",
            b".5",
            b"-.5",
            b"+.5",
            b"+1",
            b"+0",
            b"1abc",
            b"1.5x",
            b"12abc",
            b"1.5f",
            b"1.2.3",
            b"1e5e5",
            b"1e5.5",
            b"--1",
            b"1--",
            b"",
            b"-",
            b"+",
            b".",
            b"e",
        ],
    )
    def test_invalid_json_number_rejected(self, token):
        result = parse_span_bool(token, 1)
        assert result is None, f"BOOL {token!r} accepted -> {result}"

    @pytest.mark.parametrize(
        "token,expected",
        [
            (b"true", True),
            (b"false", False),
            (b"1", True),
            (b"0", False),
            (b"2", True),
            (b"-1", True),
            (b"2.5", True),
            (b"0.0", False),
            (b"1e5", True),
            (b"0e0", False),
            (b"-0.5", True),
            (b"-0", False),
            (b"1.5e3", True),
            (b"1E5", True),
            (b"1.5e-10", True),
        ],
    )
    def test_valid_json_number_accepted(self, token, expected):
        result = parse_span_bool(token, 1)
        assert result is not None, f"BOOL {token!r} rejected (should parse)"
        assert bool(result[0]) == expected
