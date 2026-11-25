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


def _render_response(payload: np.ndarray) -> bytes:
    """The production render chain for a model output tensor."""
    from mlserver.codecs.json import encode_to_json_bytes
    from mlserver.codecs.numpy import NumpyCodec
    from mlserver.types import InferenceResponse

    output = NumpyCodec.encode_output(name="scores", payload=payload)
    response = InferenceResponse(model_name="m", outputs=[output])
    return encode_to_json_bytes(response.model_dump())


def _test_subclass_renders(dtype) -> None:
    class SubArray(np.ndarray):
        """A custom ndarray subclass, as model code may return."""

    subclass_output = np.array([1.0, 2.0, 3.0], dtype=dtype).view(SubArray)
    blob = _render_response(subclass_output)

    # Byte-identical to the equivalent plain ndarray output: the
    # subclass strips to a zero-copy base-class view and takes the
    # same native rendering path.
    plain_blob = _render_response(np.array([1.0, 2.0, 3.0], dtype=dtype))
    assert blob == plain_blob


def test_subclass_root_renders_through_native_path():
    """A custom ndarray subclass as a model output must render (the
    tolist boundary served these as plain lists; orjson's native
    path accepts only the base ndarray class) — never a TypeError
    from the JSON renderer. Covers both serializer branches: the
    native dtype return and the float32 widened bridge."""
    _test_subclass_renders(np.float64)
    _test_subclass_renders(np.float32)


def test_subclass_root_json_mode_still_renders():
    """JSON-mode dumps take the tolist path (subclass-safe): a
    subclass root renders the same values as a plain one."""

    class SubArray(np.ndarray):
        pass

    data = np.array([1.0, 2.0], dtype=np.float64).view(SubArray)
    wrapped = TensorData(root=data)
    assert wrapped.model_dump_json() == TensorData(root=[1.0, 2.0]).model_dump_json()


def test_masked_array_root_keeps_none_entries():
    """A directly-constructed masked root keeps the legacy list path
    in BOTH dump modes: masked entries are None on the wire, never
    the underlying buffer value. The subclass strip must not apply
    to this path (a base-class view would drop the mask)."""
    masked = np.ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])
    wrapped = TensorData(root=masked)

    assert wrapped.model_dump() == [1.0, None, 3.0]
    assert (
        wrapped.model_dump_json() == TensorData(root=[1.0, None, 3.0]).model_dump_json()
    )
