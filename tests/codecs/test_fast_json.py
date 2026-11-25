"""Fast two-phase JSON parse: acceptance, rejection and fallback.

The fast path must be a strict subset of the standard path: every
payload it takes must parse to the same values, and every payload it
rejects (or mishandles) must fall back to the standard parse — never
a wrong answer, never a different error.
"""

import orjson
import pytest

from mlserver.codecs.fast_json import (
    build_slim_body,
    fastspan_parser,
    find_numeric_data_spans,
    parse_inference_request,
    verify_envelope,
)
from mlserver.types import InferenceRequest


def _body(data_json: str, datatype: str = "FP32", extra: str = "") -> bytes:
    return (
        b'{"id":"x","inputs":[{"name":"input-0","shape":[3],'
        b'"datatype":"'
        + datatype.encode()
        + b'","data":'
        + data_json.encode()
        + b"}]"
        + extra.encode()
        + b"}"
    )


def _fallback(body: bytes) -> InferenceRequest:
    return InferenceRequest.model_validate_json(body)


def _span_parser(span: bytes, datatype: str, count: int):
    """A reference span parser: parse the JSON array, check the count."""
    import numpy as np

    dtype = {"FP32": np.float32, "FP64": np.float64, "INT64": np.int64}.get(datatype)
    if dtype is None:
        raise ValueError(datatype)
    values = orjson.loads(b"[" + span + b"]")
    if len(values) != count:
        raise ValueError(f"count mismatch: {len(values)} != {count}")
    return np.array(values, dtype=dtype)


class TestSpanExtraction:
    def test_flat_numeric_span(self):
        body = _body("[1.5, 2.5, -0.5]")
        spans = find_numeric_data_spans(body)
        assert spans is not None and len(spans) == 1
        start, end = spans[0]
        assert body[start:end] == b"1.5, 2.5, -0.5"

    def test_whitespace_and_negatives(self):
        body = _body("[ 1.5 ,\n -2.5 ,\t 0.0 ]")
        spans = find_numeric_data_spans(body)
        assert body[spans[0][0] : spans[0][1]] == b" 1.5 ,\n -2.5 ,\t 0.0 "

    def test_string_data_is_skipped(self):
        body = _body('["NaN", "Infinity", "-Infinity"]')
        assert find_numeric_data_spans(body) == []

    def test_nested_data_is_skipped(self):
        body = _body("[[1, 2], [3, 4]]")
        assert find_numeric_data_spans(body) == []

    def test_bool_data_is_a_span(self):
        # true/false contain no quotes or brackets: a parseable span
        body = _body("[true, false]", datatype="BOOL")
        spans = find_numeric_data_spans(body)
        assert len(spans) == 1

    def test_string_containing_bracket_rejected_not_misparsed(self):
        # A string element with ']' inside must not truncate a span
        # into a false numeric span: the truncated span contains its
        # opening quote and is skipped.
        body = _body('["a]b", 1.5]')
        assert find_numeric_data_spans(body) == []

    def test_unbalanced_brackets_reach_standard_error(self):
        # No closing bracket anywhere after the data: the extractor
        # reports the anomaly (None), and even when a later bracket
        # yields a bogus span, the slim body fails to parse and the
        # standard path raises the proper JSON error.
        body = (
            b'{"id":"x","inputs":[{"name":"i","shape":[3],'
            b'"datatype":"FP32","data":[1.5, 2.5'
        )
        assert find_numeric_data_spans(body) is None
        with pytest.raises(Exception):
            parse_inference_request(body, _fallback, _span_parser)

    def test_bogus_span_from_malformed_json_falls_back(self):
        # An unbalanced data array whose walker latches onto a later
        # bracket: the slim body is invalid JSON, so the fast path
        # defers and the standard path raises.
        body = (
            b'{"id":"x","inputs":[{"name":"i","shape":[3],'
            b'"datatype":"FP32","data":[1.5, 2.5}]}'
        )
        with pytest.raises(Exception):
            parse_inference_request(body, _fallback, _span_parser)

    def test_empty_array_is_a_span(self):
        body = _body("[]")
        spans = find_numeric_data_spans(body)
        assert len(spans) == 1
        assert body[spans[0][0] : spans[0][0]] == b""

    def test_data_key_inside_a_string_value_never_matches(self):
        # A parameter value containing the literal text `"data": [1]`
        # is escaped in the raw body (\"data\") — the closing quote of
        # the pattern can never match an in-string quote — so the
        # regex only ever matches real JSON keys. The fast path
        # proceeds correctly with just the tensor span.
        body = (
            b'{"id":"x","inputs":[{"name":"i","shape":[1],"datatype":"FP32",'
            b'"parameters":{"note":"{\\"data\\": [9, 9]}"},"data":[1.5]}]}'
        )
        spans = find_numeric_data_spans(body)
        assert spans is not None and len(spans) == 1
        slim = build_slim_body(body, spans)
        parsed = orjson.loads(slim)
        # The parameter string survived the slimming untouched
        assert parsed["inputs"][0]["parameters"]["note"] == '{"data": [9, 9]}'
        assert verify_envelope(parsed, len(spans)) is True

    def test_data_key_in_parameters_extra_rejected(self):
        # A REAL "data" key nested inside parameters.extra matches the
        # regex and gets zeroed — the path-aware verification must
        # reject the fast path so the standard parse sees the
        # original, uncorrupted value.
        body = (
            b'{"inputs":[{"name":"i","shape":[1],"datatype":"FP32",'
            b'"parameters":{"extra":{"data":[1, 2]}},'
            b'"data":[1.5]}]}'
        )
        spans = find_numeric_data_spans(body)
        assert len(spans) == 2  # both matched: the extra one and the tensor one
        parsed = orjson.loads(build_slim_body(body, spans))
        assert verify_envelope(parsed, len(spans)) is False

    def test_nested_inputs_inside_parameters_rejected(self):
        # parameters.extra containing a whole fake input object with a
        # data field: paths deeper than inputs[i] must be rejected.
        body = (
            b'{"inputs":[{"name":"i","shape":[1],"datatype":"FP32",'
            b'"parameters":{"extra":{"inputs":[{"data":[3, 3]}]}},'
            b'"data":[1.5]}]}'
        )
        spans = find_numeric_data_spans(body)
        parsed = orjson.loads(build_slim_body(body, spans))
        assert verify_envelope(parsed, len(spans)) is False

    def test_multiple_inputs(self):
        body = (
            b'{"inputs":['
            b'{"name":"a","shape":[2],"datatype":"FP32","data":[1.5, 2.5]},'
            b'{"name":"b","shape":[2],"datatype":"INT64","data":[7, 8]}]}'
        )
        spans = find_numeric_data_spans(body)
        assert len(spans) == 2
        assert body[spans[0][0] : spans[0][1]] == b"1.5, 2.5"
        assert body[spans[1][0] : spans[1][1]] == b"7, 8"


class TestEnvelopeVerification:
    def test_all_numeric_passes(self):
        body = _body("[1.5, 2.5, 3.5]")
        spans = find_numeric_data_spans(body)
        parsed = orjson.loads(build_slim_body(body, spans))
        assert verify_envelope(parsed, 1) is True

    def test_extra_data_key_in_parameters_rejected(self):
        body = (
            b'{"inputs":[{"name":"i","shape":[1],"datatype":"FP32",'
            b'"parameters":{"extra":{"data":[1, 2]}},'
            b'"data":[1.5]}]}'
        )
        spans = find_numeric_data_spans(body)
        # Both data arrays matched: the extra one and the tensor one
        parsed = orjson.loads(build_slim_body(body, spans))
        assert verify_envelope(parsed, len(spans)) is False

    def test_mixed_string_and_numeric_falls_back(self):
        body = (
            b'{"inputs":['
            b'{"name":"a","shape":[1],"datatype":"BYTES","data":["x"]},'
            b'{"name":"b","shape":[1],"datatype":"FP32","data":[1.5]}]}'
        )
        spans = find_numeric_data_spans(body)  # only the numeric one
        parsed = orjson.loads(build_slim_body(body, spans))
        # The string input's data is not the [] marker -> reject
        assert verify_envelope(parsed, len(spans)) is False


class TestParseInferenceRequest:
    def test_fast_path_matches_standard_values(self):
        import numpy as np

        body = _body("[1.5, 2.5, -0.5]")
        fast = parse_inference_request(body, _fallback, _span_parser)
        slow = _fallback(body)
        assert np.array_equal(
            np.asarray(fast.inputs[0].data.root, dtype=np.float32),
            np.asarray(slow.inputs[0].data.root, dtype=np.float32),
        )
        assert fast.id == slow.id
        assert fast.inputs[0].name == slow.inputs[0].name

    def test_fast_path_produces_ndarray(self):
        import numpy as np

        body = _body("[1.5, 2.5, -0.5]")
        fast = parse_inference_request(body, _fallback, _span_parser)
        assert isinstance(fast.inputs[0].data.root, np.ndarray)

    def test_fast_path_attach_does_not_walk_array_elements(self):
        """The span-parsed ndarray the fast path attaches must reach
        the validated request WITHOUT an element-by-element walk:
        validation of an ndarray root used to build (and discard) a
        temporary scalar list — O(N) per tensor per request."""

        import numpy as np

        class Observed(np.ndarray):
            elements = 0

            def __iter__(self):
                for value in super().__iter__():
                    type(self).elements += 1
                    yield value

        def counting_span_parser(span: bytes, datatype: str, count: int):
            return _span_parser(span, datatype, count).view(Observed)

        body = _body("[1.5, 2.5, -0.5]")
        fast = parse_inference_request(body, _fallback, counting_span_parser)

        assert Observed.elements == 0
        root = fast.inputs[0].data.root
        assert isinstance(root, np.ndarray)
        assert np.asarray(root, dtype=np.float32).tolist() == [1.5, 2.5, -0.5]

    def test_string_data_falls_back_cleanly(self):
        body = _body('["NaN", "Infinity"]')
        fast = parse_inference_request(body, _fallback, _span_parser)
        slow = _fallback(body)
        assert list(fast.inputs[0].data.root) == list(slow.inputs[0].data.root)

    def test_invalid_payload_error_is_the_standard_one(self):
        # A payload that pydantic rejects must produce the exact
        # standard-path error (the fast path never invents errors).
        body = (
            b'{"id":"x","inputs":[{"name":"i","shape":"bad",'
            b'"datatype":"FP32","data":[1.5]}]}'
        )
        with pytest.raises(Exception) as fast_exc:
            parse_inference_request(body, _fallback, _span_parser)
        with pytest.raises(Exception) as slow_exc:
            _fallback(body)
        assert type(fast_exc.value) is type(slow_exc.value)

    def test_fallback_called_on_span_parser_failure(self):
        body = _body("[1.5, 2.5]")

        def broken(span, datatype):
            raise RuntimeError("parser backend unavailable")

        fast = parse_inference_request(body, _fallback, broken)
        slow = _fallback(body)
        import numpy as np

        assert np.array_equal(
            np.asarray(fast.inputs[0].data.root), np.asarray(slow.inputs[0].data.root)
        )

    def test_datatype_outside_fast_set_falls_back(self):
        # BOOL spans parse fine by extraction, but the parser backend
        # in this test does not handle BOOL: fallback with intact data.
        body = _body("[true, false]", datatype="BOOL")
        fast = parse_inference_request(body, _fallback, _span_parser)
        slow = _fallback(body)
        assert list(fast.inputs[0].data.root) == list(slow.inputs[0].data.root)

    def test_parameters_extra_data_survives_via_fallback(self):
        # The corrupted slim body must never escape: the fallback
        # re-parses the ORIGINAL body, keeping parameters.extra intact.
        body = (
            b'{"inputs":[{"name":"i","shape":[1],"datatype":"FP32",'
            b'"parameters":{"extra":{"data":[1, 2]}},'
            b'"data":[1.5]}]}'
        )
        fast = parse_inference_request(body, _fallback, _span_parser)
        slow = _fallback(body)
        assert fast.inputs[0].parameters.extra == slow.inputs[0].parameters.extra
        import numpy as np

        assert np.array_equal(
            np.asarray(fast.inputs[0].data.root), np.asarray(slow.inputs[0].data.root)
        )

    def test_malformed_json_reaches_standard_error(self):
        body = b'{"inputs": [{'  # truncated
        with pytest.raises(Exception):
            parse_inference_request(body, _fallback, _span_parser)


class TestNativeBackend:
    """The C span parser behind the fast path (skipped when the
    compiled library is absent — a plain pip install)."""

    def _native_or_skip(self):
        from mlserver.codecs import fastspan

        if fastspan._load() is None:
            pytest.skip("native libfastspan.so not compiled")
        return fastspan

    def test_value_equality_on_specials(self):
        import numpy as np

        nat = self._native_or_skip()
        # Numeric tokens only: the C grammar is strict JSON numbers —
        # NaN/Infinity STRING spellings are the fallback's job (pinned
        # by the extraction tests).
        values = [
            0.0,
            -0.0,
            1e-38,
            3.4028234663852886e38,
            1.1754943508222875e-38,
            1.5,
            -2.5,
            0.1,
            6.02e23,
            -1.6e-19,
        ]
        span = ", ".join(repr(v) for v in values).encode()
        out = nat.parse_span_f32(span, len(values))
        assert out is not None
        ref = np.array(values, dtype=np.float32)
        assert np.array_equal(out, ref, equal_nan=True)

    def test_long_negative_tokens_keep_sign(self):
        # 20+-digit negative tokens take the strtod fallback, whose
        # token includes the sign: the historical double-negation bug.
        nat = self._native_or_skip()
        span = b"-0.00012038055865559727, -0.00045484330621547997"
        out = nat.parse_span_f32(span, 2)
        assert out is not None
        assert out[0] < 0 and out[1] < 0

    def test_grammar_rejections(self):
        nat = self._native_or_skip()
        # NOTE: counts MATCH the valid-value count so the grammar (not a
        # count mismatch) is what rejects — a trailing-comma bug once
        # hid behind a count-mismatch assertion.
        for bad, count in [
            (b"1.5, nan", 2),
            (b"01.5", 1),
            (b"1.5,", 1),  # trailing comma — with the MATCHING count
            (b"1.5,,2.5", 2),
            (b"[1", 1),
            (b'"x"', 1),
            (b"1.5 2.5", 2),
            (b"1.5, ", 1),  # dangling separator after whitespace
        ]:
            assert nat.parse_span_f32(bad, count) is None, (bad, count)

    def test_differential_fuzz_against_orjson(self):
        """Randomized differential test vs the standard path's own
        parse (orjson -> Python float). Found both the trailing-comma
        acceptance and the FP64 double-rounding in review; kept as the
        permanent proof that the C grammar and values stay a strict
        subset of the reference."""
        import random

        import numpy as np

        nat = self._native_or_skip()
        rng = random.Random(20261124)

        for _ in range(300):
            n = rng.randint(1, 40)
            values = []
            for _ in range(n):
                kind = rng.randint(0, 5)
                if kind == 0:
                    values.append(rng.uniform(-1e6, 1e6))
                elif kind == 1:
                    values.append(rng.uniform(-1, 1) * 10 ** rng.randint(-30, 30))
                elif kind == 2:
                    # high-precision mantissas: the double-rounding class
                    values.append(
                        rng.randint(-(10**19), 10**19) * 10.0 ** rng.randint(-19, 0)
                    )
                elif kind == 3:
                    values.append(float(rng.randint(-(2**63), 2**63 - 1)))
                elif kind == 4:
                    values.append(rng.choice([0.0, -0.0, 1.0, -1.0]))
                else:
                    # round-trip through repr: the JSON text form
                    values.append(float(repr(rng.uniform(-1e10, 1e10))))

            # The exact wire form: JSON number text via repr (what
            # json.dumps/orjson produce for these floats).
            span = ", ".join(repr(v) for v in values).encode()

            # FP64: must match orjson's correctly-rounded doubles EXACTLY
            out64 = nat.parse_span_f64(span, n)
            assert out64 is not None, values
            ref64 = orjson.loads(b"[" + span + b"]")
            for i, (a, b) in enumerate(zip(out64, ref64)):
                assert a == b, (i, values[i], repr(values[i]), a, b)

            # FP32: matches the reference path's f64->f32 narrowing
            out32 = nat.parse_span_f32(span, n)
            ref32 = np.asarray(ref64, dtype=np.float32)
            assert np.array_equal(out32, ref32, equal_nan=True)

    def test_fast_path_end_to_end_with_native(self):
        import numpy as np

        from mlserver.codecs import NumpyCodec
        from mlserver.codecs.fast_json import fastspan_parser

        self._native_or_skip()
        body = _body("[1.5, 2.5, -0.5, 0.0, 1e-8, 123.456]").replace(
            b'"shape":[3]', b'"shape":[6]'
        )
        fast = parse_inference_request(body, _fallback, fastspan_parser)
        slow = _fallback(body)
        assert isinstance(fast.inputs[0].data.root, np.ndarray)
        # The equality contract is at the DECODED level: the standard
        # path yields f64 Python floats, the fast path the declared
        # f32 — both decode to the same ndarray.
        fast_arr = NumpyCodec.decode_input(fast.inputs[0])
        slow_arr = NumpyCodec.decode_input(slow.inputs[0])
        assert np.array_equal(fast_arr, slow_arr)

    def test_count_mismatch_falls_back(self):
        # The shape declares 4 but the span carries 3: the standard
        # path does not validate counts at parse time (the reshape
        # happens at decode), so the fast path must defer and produce
        # the SAME result as the standard path — a list, unchanged.
        body = (
            b'{"inputs":[{"name":"i","shape":[4],"datatype":"FP32",'
            b'"data":[1.5, 2.5, 3.5]}]}'
        )
        from mlserver.codecs.fast_json import fastspan_parser

        self._native_or_skip()
        fast = parse_inference_request(body, _fallback, fastspan_parser)
        slow = _fallback(body)
        assert list(fast.inputs[0].data.root) == list(slow.inputs[0].data.root)


class TestDuplicateKeySpanAssociation:
    """Duplicate JSON keys can collapse dict positions while preserving
    the data field count — the marker-identity scheme must either pair
    spans correctly or fall back, never mis-associate."""

    def _run(self, body):
        return parse_inference_request(body, self._fallback, fastspan_parser)

    def _fallback(self, b):
        return InferenceRequest.model_validate_json(b)

    def _compare(self, body):
        fast = self._run(body)
        std = self._fallback(body)
        if fast is None:
            return  # fallback is always safe
        fast_vals = sorted(
            (
                i.name,
                (
                    tuple(i.data.root)
                    if not hasattr(i.data.root, "tobytes")
                    else tuple(i.data.root.tolist())
                ),
            )
            for i in fast.inputs
        )
        std_vals = sorted(
            (
                i.name,
                (
                    tuple(i.data.root)
                    if not hasattr(i.data.root, "tobytes")
                    else tuple(i.data.root.tolist())
                ),
            )
            for i in std.inputs
        )
        assert fast_vals == std_vals, f"fast={fast_vals} std={std_vals}"

    def test_dup_inputs_with_intervening_outputs(self):
        """The count-matching construction: first inputs has STRING data
        (no numeric span), so span count == parent count. Without
        marker identity, the output's value would attach to the input.
        """
        body = (
            b'{"inputs":[{"name":"a","shape":[1],"datatype":"FP32","data":["x"]}],'
            b'"outputs":[{"name":"o","shape":[1],"datatype":"FP32","data":[9]}],'
            b'"inputs":[{"name":"b","shape":[1],"datatype":"FP32","data":[11]}]}'
        )
        self._compare(body)

    def test_dup_data_key_within_item(self):
        """A duplicate data key within one item: orjson keeps the last,
        the span count exceeds the parent count → fallback."""
        body = (
            b'{"inputs":[{"name":"i","shape":[2],"datatype":"FP32",'
            b'"data":[1,2],"data":"x"}]}'
        )
        self._compare(body)

    def test_dup_datatype_key(self):
        """A duplicate datatype key: the last declaration wins on both
        paths."""
        body = (
            b'{"inputs":[{"name":"i","shape":[2],"datatype":"INT32",'
            b'"data":[1,2],"datatype":"FP32"}]}'
        )
        self._compare(body)

    def test_dup_inputs_array(self):
        """Duplicate top-level inputs arrays: orjson keeps the last."""
        body = (
            b'{"inputs":[{"name":"a","shape":[1],"datatype":"FP32","data":[7]}],'
            b'"inputs":[{"name":"b","shape":[1],"datatype":"FP32","data":[11]}]}'
        )
        self._compare(body)

    def _assert_fallback_ran(self, body):
        """Run the fast parse with a spying fallback and assert the
        fallback was invoked (parse_inference_request always returns
        a result — fast-parsed or fallback — so only the spy can
        distinguish the two)."""
        calls = []

        def spying_fallback(b):
            calls.append(b)
            return self._fallback(b)

        result = parse_inference_request(body, spying_fallback, fastspan_parser)
        assert calls, (
            "the fast path ran on a marker-collision body — the "
            "pre-check must force the fallback"
        )
        return result

    def test_marker_collision_falls_back(self):
        """A plain data value that equals the marker base must not be
        mis-associated: the pre-check detects the collision and the
        whole request falls back to the standard path."""
        from mlserver.codecs.fast_json import _SPAN_MARKER_BASE

        marker_val = _SPAN_MARKER_BASE  # -918273645
        body = (
            b'{"inputs":[{"name":"a","shape":[1],"datatype":"INT64",'
            b'"data":[' + str(marker_val).encode() + b"]}]}"
        )
        result = self._assert_fallback_ran(body)
        std = self._fallback(body)
        assert result.inputs[0].data.root[0] == std.inputs[0].data.root[0]

    def test_escaped_dup_key_marker_collision_falls_back(self):
        """An escaped duplicate data key can forge a marker.

        The scanner matches the literal ``"data"`` bytes, so an
        escaped ``dat\\u0061`` key is invisible to it — its value is
        never replaced with a marker. When that value equals the
        span's marker, the slim-body verifier accepts it as the
        marker's own slot and the fast path attaches the scanned
        span's value to the wrong tensor: the standard path keeps
        the escaped key's value (-918273645) while the fast path
        would return the literal key's value (11). The full-body
        marker pre-check must catch this and force the fallback.
        """
        from mlserver.codecs.fast_json import _SPAN_MARKER_BASE

        body = (
            b'{"inputs":[{"name":"a","shape":[1],"datatype":"INT64",'
            b'"dat\\u0061":[' + str(_SPAN_MARKER_BASE).encode() + b'],"data":[11]}]}'
        )
        # The contract is version-independent: the pre-check forces
        # the fallback and the fast result equals the standard
        # result. WHICH duplicate key the standard parser keeps is
        # pydantic-version dependent (2.9 keeps the first, 2.14 the
        # last) — the fast path delegates, so both agree regardless.
        std = self._fallback(body)
        result = self._assert_fallback_ran(body)
        assert result.inputs[0].data.root[0] == std.inputs[0].data.root[0]

    def test_marker_collision_in_skipped_span_falls_back(self):
        """A marker value inside a span the scanner skipped (mixed
        string+number data) must also force the fallback — the
        skipped content stays in the slim body and can forge a
        marker slot."""
        from mlserver.codecs.fast_json import _SPAN_MARKER_BASE

        body = (
            b'{"inputs":[{"name":"a","shape":[2],"datatype":"INT64",'
            b'"data":["skip", ' + str(_SPAN_MARKER_BASE).encode() + b"]}],"
            b'"inputs":[{"name":"b","shape":[1],"datatype":"FP32","data":[11]}]}'
        )
        result = self._assert_fallback_ran(body)
        std = self._fallback(body)
        std_vals = sorted(i.name for i in std.inputs)
        fast_vals = sorted(i.name for i in result.inputs)
        assert fast_vals == std_vals
