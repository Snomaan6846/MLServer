import pytest

from mlserver.grpc.utils import to_metadata


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"foo": "bar"}, (("foo", "bar"),)),
        ({"foo": "bar", "foo2": "bar2"}, (("foo", "bar"), ("foo2", "bar2"))),
        ({"foo": "bar", "X-Foo": "bar2"}, (("foo", "bar"), ("x-foo", "bar2"))),
    ],
)
def test_to_metadata(headers: dict[str, str], expected: tuple[tuple[str, str], ...]):
    metadata = to_metadata(headers)

    assert metadata == expected
