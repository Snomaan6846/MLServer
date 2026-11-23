"""Fast two-phase JSON request parsing for numeric tensor data.

The request-side floor on detection-scale payloads is Python float
object creation: the full-body parse materialises hundreds of
thousands of float objects (orjson ~51 ms, pydantic ~111 ms on a
15 MB body) that exist only to be poured into a NumPy buffer.

This module keeps the envelope on the standard path but parses the
numeric ``data`` arrays directly from their raw byte spans into
pre-allocated buffers:

  1. locate every array-valued ``data`` field's byte span (regex +
     balanced-bracket walk, all C-speed byte ops)
  2. build a slim body with those spans replaced by ``[]`` and parse
     + pydantic-validate it (no floats exist)
  3. verify the zeroed positions are exactly the tensor data fields
     (count, marker value, and no other ``data`` key anywhere in the
     tree) — any anomaly falls back to the standard path
  4. parse each span natively into the datatype's buffer and attach it

The fast path is a strict subset: anything unusual (strings in data,
nested arrays, extra ``data`` keys in parameters, structural
anomalies) falls back to today's exact parsing, so acceptance and
rejection behaviour is unchanged by construction.
"""

import re

from typing import Any, Callable

import orjson

from ..logging import logger
from ..types import InferenceRequest

# Array-valued "data" keys: `"data" <ws>* : <ws>* [`
_DATA_KEY_RE = re.compile(rb'"data"\s*:\s*\[')

# Datatypes the native span parser handles. Anything else takes the
# standard path. INT* and BOOL included (the int parsers use the
# mantissa directly for exact boundary values; BOOL accepts
# true/false literals and numeric truthiness).
_FAST_DATATYPES = frozenset(
    {
        "FP32",
        "FP64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "BOOL",
    }
)


def find_numeric_data_spans(body: bytes) -> "list[tuple[int, int]] | None":
    """Byte spans (start, end) of every array-valued ``data`` field
    whose content contains no strings.

    Returns None on structural anomaly (unbalanced brackets). Spans
    containing ``[`` (nested arrays) or ``"`` (string elements) are
    skipped — their payloads take the standard path.

    Limitation (safe by contract): the bracket walk is not
    string-aware — a ``]`` inside a JSON string value outside the
    data arrays can desynchronize depth tracking. A mis-located span
    then fails envelope verification (``_verified_parents``) and the
    whole request falls back to the standard parse, so the cost is
    fast-path opportunity, never correctness.
    """
    spans: "list[tuple[int, int]]" = []
    for match in _DATA_KEY_RE.finditer(body):
        start = match.end()
        depth = 1
        pos = start
        while depth:
            nxt = body.find(b"]", pos)
            if nxt < 0:
                return None
            depth += body.count(b"[", pos, nxt)
            depth -= 1
            pos = nxt + 1
        end = pos - 1
        span = body[start:end]
        if b'"' in span or b"[" in span:
            # String elements or nested arrays: not the flat numeric
            # subset (a string containing ']' can only make the span
            # SHORTER, and any such span contains its opening quote —
            # the check above rejects it).
            continue
        spans.append((start, end))
    return spans


# Unique span markers: large negative integers that cannot appear as
# real tensor data. Each span gets a distinct marker in the slim body;
# the parsed structure is verified BY MARKER IDENTITY, not by
# position — duplicate JSON keys can collapse positions while keeping
# the count, which would silently mis-pair spans with tensors.
_SPAN_MARKER_BASE = -918273645


def build_slim_body(body: bytes, spans: "list[tuple[int, int]]") -> bytes:
    """The body with every numeric data span replaced by a unique marker."""
    if not spans:
        return body
    parts: "list[bytes]" = []
    prev = 0
    for i, (start, end) in enumerate(spans):
        parts.append(body[prev:start])
        parts.append(str(_SPAN_MARKER_BASE + i).encode())
        prev = end
    parts.append(body[prev:])
    return b"".join(parts)


def _data_parents(parsed: Any) -> "list[tuple[dict, tuple]]":
    """Every dict carrying a ``data`` key, in document order, with its
    path from the parsed root."""
    parents: "list[tuple[dict, tuple]]" = []

    def walk(node: Any, path: tuple) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "data":
                    parents.append((node, path))
                else:
                    walk(value, path + (key,))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, path + (index,))

    walk(parsed, ())
    return parents


def _is_tensor_data_path(path: tuple) -> bool:
    """The path of a dict that is an element of the root's inputs or
    outputs list — the only valid positions for a tensor data field.

    Anything deeper (``parameters.extra``, nested user objects) or
    elsewhere is NOT a tensor data field: zeroing it would corrupt the
    request, so the fast path must be rejected.
    """
    return len(path) == 2 and path[0] in ("inputs", "outputs")


def _verified_parents(parsed: Any, n_spans: int) -> "list | None":
    """The parents of matched data fields, ordered by span marker.

    Each span's marker must appear at exactly one tensor data path,
    and the markers must form a complete set (0..n_spans-1). Duplicate
    JSON keys can collapse dict positions while preserving the data
    field count — pairing by position would silently attach the wrong
    span to the wrong tensor. Marker identity is unforgeable: a
    collapsed key either drops a marker (missing → fallback) or moves
    it to a non-tensor path (path check → fallback).

    Returns the parents in span order (parents[i] corresponds to
    spans[i]), or None when the envelope does not verify.
    """
    parents = _data_parents(parsed)
    if len(parents) != n_spans:
        return None
    result: "list[tuple[dict, tuple] | None]" = [None] * n_spans
    for parent, path in parents:
        if not _is_tensor_data_path(path):
            return None
        data = parent.get("data")
        if not isinstance(data, list) or len(data) != 1:
            return None
        marker = data[0]
        if (
            not isinstance(marker, int)
            or isinstance(marker, bool)
            or not (_SPAN_MARKER_BASE <= marker < _SPAN_MARKER_BASE + n_spans)
        ):
            return None
        idx = marker - _SPAN_MARKER_BASE
        if result[idx] is not None:
            return None  # duplicate marker
        result[idx] = (parent, path)
    if any(r is None for r in result):
        return None  # missing marker (collapsed by duplicate keys)
    return result


def verify_envelope(parsed: Any, n_spans: int) -> bool:
    """Public assertion API for the envelope verification step: the
    zeroed positions are exactly the tensor data fields. The
    production parse calls ``_verified_parents`` directly; this
    boolean-shaped wrapper is the stable entry point for the
    differential tests and any external verification tooling."""
    return _verified_parents(parsed, n_spans) is not None


def parse_inference_request(
    body: bytes,
    fallback: Callable[[bytes], InferenceRequest],
    span_parser: Callable[[bytes, str, int], Any],
) -> InferenceRequest:
    """Two-phase parse with a strict-subset fast path.

    ``fallback`` is the deployment's standard parse (today's exact
    behaviour, including its error semantics). ``span_parser`` parses
    one flat numeric span into the datatype's buffer, given the
    element count the envelope's shape declares. The datatype is read
    from the slim-parsed envelope (the sibling ``datatype`` field), so
    the parser sees the wire's own declaration.
    """
    try:
        spans = find_numeric_data_spans(body)
        if spans:
            # Pre-check: no marker value may appear anywhere in the
            # original body. A data field the scanner skipped (string
            # elements, nested arrays, duplicate-key remnants) could
            # carry a value that equals a marker, and the slim-body
            # verifier would accept it as the marker's parent —
            # associating the wrong span with the wrong tensor.
            # Checking the full body (not just spans) eliminates the
            # class entirely; the cost is that requests containing
            # values near the marker base fall back (an extremely
            # rare and acceptable trade).
            for i in range(len(spans)):
                marker_bytes = str(_SPAN_MARKER_BASE + i).encode()
                if marker_bytes in body:
                    raise _NotFast()  # marker collision → fallback
            slim = build_slim_body(body, spans)
            parsed = orjson.loads(slim)
            parents = _verified_parents(parsed, len(spans))
            if parents is not None:
                # Ordering invariant: spans are located in body byte
                # order and _data_parents walks the parsed dict in
                # insertion order — orjson.loads preserves the key
                # order of the input, so the two sequences pair
                # element-for-element. If the JSON library is ever
                # swapped, this zip (and the whole span pairing)
                # silently breaks — hence the assert.
                assert len(spans) == len(parents)
                for (start, end), (parent, _path) in zip(spans, parents):
                    datatype = _datatype_of(parent)
                    count = _declared_count(parent)
                    if datatype not in _FAST_DATATYPES or count is None:
                        raise _NotFast()
                    parent["data"] = span_parser(body[start:end], datatype, count)
                return InferenceRequest.model_validate(parsed)
    except _NotFast:
        pass
    except Exception:
        # An unexpected anomaly (not a designed rejection): the
        # standard path decides, but leave a diagnostic trace — a
        # silently-broken fast path is otherwise indistinguishable
        # from an absent library.
        logger.debug(
            "fast JSON parse failed unexpectedly; using the standard path",
            exc_info=True,
        )
    return fallback(body)


class _NotFast(Exception):
    """A data field the native parser does not handle — fall back."""


def _datatype_of(parent: dict) -> Any:
    """The datatype declared next to a tensor data field."""
    datatype = parent.get("datatype")
    return datatype if isinstance(datatype, str) else None


def _declared_count(parent: dict) -> "int | None":
    """The element count a tensor's shape declares, or None when the
    shape is absent or not a list of positive integers (the standard
    path then produces the proper validation error)."""
    shape = parent.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            return None
        count *= dim
    return count


def fastspan_parser(span: bytes, datatype: str, count: int) -> Any:
    """The native (C) span backend; _NotFast for anything it does not
    handle so the orchestrator falls back to the standard path."""
    from .fastspan import (
        parse_span_f32,
        parse_span_f64,
        parse_span_int,
        parse_span_bool,
    )

    # Every element needs at least one digit plus a separator (except
    # the last), so N elements require >= 2N-1 span bytes. A declared
    # count beyond that cannot fit: reject before the backend
    # allocates the output buffer, so a huge declared shape cannot
    # trigger the allocation ahead of the count-mismatch rejection.
    if count > (len(span) + 1) // 2:
        raise _NotFast(datatype)

    if datatype == "FP32":
        result = parse_span_f32(span, count)
    elif datatype == "FP64":
        result = parse_span_f64(span, count)
    elif datatype == "BOOL":
        result = parse_span_bool(span, count)
    elif datatype in (
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    ):
        result = parse_span_int(span, count, datatype)
    else:
        raise _NotFast(datatype)
    if result is None:
        # The native parser is absent or rejected the span's grammar.
        raise _NotFast(datatype)
    return result
