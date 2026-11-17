import pytest
import json
import numpy as np
from mlserver.types import InferenceRequest
from pydantic import ValidationError

from mlserver import types


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        [[1.0, 2.0, 3.0]],
        [34.5, 8.4],
        [True, False, True],
        ["one", "two", "three"],
    ],
)
def test_tensor_data(data):
    raw = json.dumps(data)
    tensor_data = types.TensorData.parse_raw(raw)

    assert tensor_data == types.TensorData(root=data)
    for tensor_elem, elem in zip(tensor_data, data):
        assert isinstance(tensor_elem, type(elem))


def test_tensor_data_keeps_ndarray_root_in_memory():
    arr = np.arange(3, dtype=np.float32)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.root is arr


def test_tensor_data_serialises_ndarray_root_as_flat_list():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump() == [1.0, 2.0, 3.0, 4.0]
    assert tensor_data.model_dump_json() == "[1.0,2.0,3.0,4.0]"


def test_tensor_data_serialises_ndarray_nan_as_null():
    arr = np.array([1.5, np.nan], dtype=np.float32)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump() == [1.5, None]
    assert tensor_data.model_dump_json() == "[1.5,null]"


def test_tensor_data_serialises_complex_nan_as_null():
    # numpy's isnan flags complex NaN (nan+0j, 0+nanj), but math.isnan
    # raises TypeError on complex values — the serializer must check
    # the real and imaginary parts separately or the dump crashes.
    arr = np.array([1.5 + 2j, np.nan + 0j, complex(0, np.nan)], dtype=np.complex64)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump() == [1.5 + 2j, None, None]


def test_tensor_data_preserves_masked_entries_with_unmasked_nan():
    # Masked entries are already None from tolist(); an unmasked NaN
    # enables the conversion loop, which must keep them instead of
    # feeding None to math.isnan (TypeError).
    arr = np.ma.array([1.0, np.nan], dtype=np.float32, mask=[True, False])

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump() == [None, None]
    assert tensor_data.model_dump_json() == "[null,null]"


def test_tensor_data_applies_serialisation_filters_to_ndarray_root():
    # The ndarray path must honour include/exclude like the list-backed
    # path: returning the converted list directly bypasses the filters.
    arr = np.array([10.0, 20.0], dtype=np.float32)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump(exclude={0}) == [20.0]
    assert tensor_data.model_dump_json(exclude={0}) == "[20.0]"
    assert tensor_data.model_dump_json(include={1}) == "[20.0]"
    # Unfiltered serialisation is unchanged by the delegation.
    assert tensor_data.model_dump() == [10.0, 20.0]
    assert tensor_data.model_dump_json() == "[10.0,20.0]"


def test_tensor_data_serialises_all_nan_complex_as_json_nulls():
    arr = np.array([np.nan + 0j, complex(0, np.nan)], dtype=np.complex64)

    tensor_data = types.TensorData(root=arr)

    # All values collapse to null, so the JSON path is serialisable
    # (non-NaN complex values are not JSON-serialisable by pydantic).
    assert tensor_data.model_dump() == [None, None]
    assert tensor_data.model_dump_json() == "[null,null]"


def test_tensor_data_serialises_complex_without_nan_unchanged():
    arr = np.array([1.5 + 2j, 3.0 - 1j], dtype=np.complex64)

    tensor_data = types.TensorData(root=arr)

    assert tensor_data.model_dump() == [1.5 + 2j, 3.0 - 1j]


@pytest.mark.parametrize(
    "root",
    [
        [b"foo", b"bar"],
        [[1, 2], [3, 4]],
        ["one", "two"],
        [1.5, None],
    ],
)
def test_tensor_data_non_ndarray_roots_serialise_unchanged(root):
    tensor_data = types.TensorData(root=root)

    assert tensor_data.model_dump() == root


def test_tensor_data_serialises_within_response():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    response = types.InferenceResponse(
        model_name="model",
        outputs=[
            types.ResponseOutput(
                name="output-0", datatype="FP32", shape=[2], data=types.TensorData(arr)
            )
        ],
    )

    as_dict = response.model_dump(mode="json")
    assert as_dict["outputs"][0]["data"] == [1.0, 2.0]


async def test_request_invalid_datatype(
    inference_request_invalid_datatype, datatype_error_message
):
    with pytest.raises(ValidationError) as excinfo:
        InferenceRequest.model_validate(inference_request_invalid_datatype)

    assert excinfo.value.errors()[0]["msg"] == datatype_error_message


def test_tensor_data_equality_with_ndarray_roots():
    """Downstream assertions like ``output.data == [1, 2]`` must keep
    working now that the response pipeline carries ndarray roots: the
    default field comparison raises (an element-wise array's truth
    value is ambiguous) or silently mis-compares for length-1 roots."""
    from mlserver.types import TensorData

    assert TensorData(root=np.array([1.0, 2.0])) == TensorData(root=[1.0, 2.0])
    assert TensorData(root=np.array([1, 2])) == [1, 2]
    assert [1, 2] == TensorData(root=np.array([1, 2]))
    assert not (TensorData(root=np.array([1.0, 2.0])) == [1.0, 3.0])
    assert not (TensorData(root=np.array([[5.0]])) == TensorData(root=[5.0]))
    assert TensorData(root=np.array([5])) == TensorData(root=[5])

    # Non-array roots keep standard equality.
    assert TensorData(root=[1, 2]) == TensorData(root=[1, 2])
    assert TensorData(root="abc") == "abc"
    assert TensorData(root=b"x") == b"x"
    assert not (TensorData(root=[1, 2]) == TensorData(root=[1, 3]))


def test_root_model_equality_shared_with_repository_types():
    """The generated RootModel equality is shared: repository index
    responses (list roots) keep comparing by their root values."""
    from mlserver.types import RepositoryIndexResponse, RepositoryIndexResponseItem

    item = RepositoryIndexResponseItem(name="m", version="v", state="READY", reason="")
    response = RepositoryIndexResponse(root=[item])
    assert response == [item]
    assert response == RepositoryIndexResponse(root=[item])
    assert not (
        response
        == [
            RepositoryIndexResponseItem(
                name="other", version="v", state="READY", reason=""
            )
        ]
    )
