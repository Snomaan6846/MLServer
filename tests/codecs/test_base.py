import typing

import numpy as np
import pytest

from datetime import datetime

from mlserver.codecs.numpy import NumpyRequestCodec, NumpyCodec
from mlserver.codecs.string import StringRequestCodec, StringCodec
from mlserver.codecs.datetime import DatetimeCodec
from mlserver.codecs.base import (
    _CodecRegistry,
    _type_hints_equivalent,
    InputCodec,
    RequestCodec,
)
from mlserver.types import RequestInput


@pytest.fixture
def codec_registry():
    registry = _CodecRegistry()
    registry.register_request_codec(NumpyRequestCodec.ContentType, NumpyRequestCodec)
    registry.register_request_codec(StringRequestCodec.ContentType, StringRequestCodec)
    registry.register_input_codec(NumpyCodec.ContentType, NumpyCodec)
    registry.register_input_codec(StringCodec.ContentType, StringCodec)
    registry.register_input_codec(DatetimeCodec.ContentType, DatetimeCodec)

    return registry


def test_deprecated_methods(caplog):
    request_input = RequestInput(
        name="foo", shape=[3], data=[1, 2, 3], datatype="INT32"
    )
    expected = np.array([1, 2, 3])

    decoded = NumpyCodec.decode(request_input)

    assert any(["DEPRECATED" in rec.message for rec in caplog.records])
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"content_type": NumpyCodec.ContentType}, NumpyCodec),
        ({"content_type": StringCodec.ContentType}, StringCodec),
        ({"content_type": "application/octet-stream"}, None),
        ({"type_hint": list[str]}, StringCodec),
        ({"type_hint": dict}, None),
        ({"payload": [datetime.now()]}, DatetimeCodec),
    ],
)
def test_find_input_codec(
    codec_registry: _CodecRegistry, kwargs: dict, expected: InputCodec | None
):
    input_codec = codec_registry.find_input_codec(**kwargs)
    assert input_codec == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"content_type": NumpyRequestCodec.ContentType}, NumpyRequestCodec),
        ({"content_type": StringRequestCodec.ContentType}, StringRequestCodec),
        ({"content_type": "application/octet-stream"}, None),
        ({"type_hint": list[str]}, StringRequestCodec),
        ({"type_hint": dict}, None),
        ({"payload": ["foo"]}, StringRequestCodec),
    ],
)
def test_find_request_codec(
    codec_registry: _CodecRegistry, kwargs: dict, expected: RequestCodec | None
):
    request_codec = codec_registry.find_request_codec(**kwargs)
    assert request_codec == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Same style
        (list[str], list[str], True),
        (str, str, True),
        # Cross style: typing.X vs builtin
        (typing.List[str], list[str], True),
        (list[str], typing.List[str], True),
        (typing.Dict[str, int], dict[str, int], True),
        (typing.Tuple[str, int], tuple[str, int], True),
        (typing.Set[str], set[str], True),
        (typing.FrozenSet[str], frozenset[str], True),
        # Union: typing.Union vs pipe syntax
        (typing.Union[str, int], str | int, True),
        (typing.Union[str, int, float], str | int | float, True),
        (typing.Union[list[str], None], list[str] | None, True),
        (typing.Optional[str], str | None, True),
        (typing.Optional[int], int | None, True),
        (typing.Optional[list[str]], list[str] | None, True),
        # Union mismatches
        (typing.Union[str, int], str | float, False),
        (typing.Optional[str], int | None, False),
        (typing.Union[str, int], str | int | float, False),
        # Mismatched args
        (typing.List[str], list[int], False),
        (list[str], list[int], False),
        # Mismatched origins
        (list[str], dict[str, int], False),
        (str, int, False),
        (str, list[str], False),
    ],
)
def test_type_hints_equivalent(a, b, expected):
    assert _type_hints_equivalent(a, b) == expected


def test_find_input_codec_cross_style_type_hint(codec_registry: _CodecRegistry):
    """Lookup with typing.List[str] should find a codec registered with list[str]."""
    input_codec = codec_registry.find_input_codec(type_hint=typing.List[str])
    assert input_codec == StringCodec


def test_find_request_codec_cross_style_type_hint(codec_registry: _CodecRegistry):
    """Lookup with typing.List[str] should find a codec registered with list[str]."""
    request_codec = codec_registry.find_request_codec(type_hint=typing.List[str])
    assert request_codec == StringRequestCodec
