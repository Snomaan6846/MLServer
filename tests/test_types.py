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
