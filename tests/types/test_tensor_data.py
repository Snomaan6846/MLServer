import numpy as np

from mlserver.types import (
    InferenceRequest,
    RepositoryIndexResponse,
    RepositoryIndexResponseItem,
    TensorData,
)


class Observed(np.ndarray):
    """Counts Python-level element iteration: nothing in the ndarray
    pipeline may walk a tensor buffer element-by-element (pydantic's
    list[Any] | Any union would build — and discard — a temporary
    scalar list on every validation)."""

    elements = 0

    def __iter__(self):
        for value in super().__iter__():
            type(self).elements += 1
            yield value


def _observed() -> np.ndarray:
    Observed.elements = 0
    return np.array([1.0, 2.0, 3.0], dtype=np.float32).view(Observed)


def test_tensor_data_ndarray_root_is_not_walked():
    """An ndarray root passes validation without element iteration and
    keeps its identity: the walk built a temporary scalar list and
    threw it away, so removing it changes no result."""
    data = _observed()
    wrapped = TensorData(root=data)

    assert Observed.elements == 0
    assert wrapped.root is data
    assert np.asarray(wrapped.root).tolist() == [1.0, 2.0, 3.0]


def test_inference_request_ndarray_data_is_not_walked():
    """The request-attachment shape (the fast JSON parse attaches the
    span-parsed array before model_validate): no walk, identity
    kept."""
    data = _observed()
    request = InferenceRequest.model_validate(
        {"inputs": [{"name": "i", "datatype": "FP32", "shape": [1, 3], "data": data}]}
    )

    assert Observed.elements == 0
    assert request.inputs[0].data.root is data


def test_tensor_data_list_root_still_validates():
    """Non-ndarray roots take the original validation unchanged: a
    JSON-parsed list root behaves exactly as before the ndarray
    short-circuit."""
    wrapped = TensorData(root=[1, 2, 3])
    assert list(wrapped.root) == [1, 2, 3]

    request = InferenceRequest.model_validate(
        {"inputs": [{"name": "i", "datatype": "FP32", "shape": [3], "data": [1, 2, 3]}]}
    )
    assert request.inputs[0].data.root == [1, 2, 3]


def test_repository_index_response_still_validates_items():
    """The other generated RootModel: its root validation (dicts into
    model items) must still run through the original handler — the
    ndarray short-circuit must not bypass validation for every root
    type."""
    response = RepositoryIndexResponse(
        root=[{"name": "m", "version": "1", "state": "READY", "reason": ""}]
    )

    item = response.root[0]
    assert isinstance(item, RepositoryIndexResponseItem)
    assert item.name == "m"
    assert item.version == "1"
