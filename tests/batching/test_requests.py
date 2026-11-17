import pytest

import numpy as np

from mlserver.types import (
    InferenceRequest,
    RequestInput,
    ResponseOutput,
    InferenceResponse,
    Parameters,
    TensorData,
)
from mlserver.batching.requests import BatchedRequests, _get_parameters


@pytest.mark.parametrize(
    "value, expected",
    [
        ("123", {"foo": ["123"]}),
        (["first", "second"], {"foo": ["first", "second"]}),
        (("first", "second"), {"foo": ["first", "second"]}),
    ],
)
def test_get_parameters_preserves_scalar_values(value, expected):
    response_output = ResponseOutput(
        name="foo",
        datatype="INT32",
        shape=[1],
        data=[1],
        parameters=Parameters(foo=value),
    )

    assert dict(_get_parameters(response_output)) == expected


@pytest.mark.parametrize("value", ["123", 123, False])
@pytest.mark.parametrize("batch_sizes", [(1, 1), (2, 1), (1, 2)])
def test_split_response_preserves_scalar_parameter_for_each_request(value, batch_sizes):
    first_data = list(range(batch_sizes[0]))
    second_data = list(range(batch_sizes[0], sum(batch_sizes)))
    requests = {
        "request-1": InferenceRequest(
            id="request-1",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[len(first_data)],
                    data=first_data,
                )
            ],
        ),
        "request-2": InferenceRequest(
            id="request-2",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[len(second_data)],
                    data=second_data,
                )
            ],
        ),
    }
    response = InferenceResponse(
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="output",
                datatype="INT32",
                shape=[sum(batch_sizes)],
                data=first_data + second_data,
                parameters=Parameters(foo=value),
            )
        ],
    )

    split = BatchedRequests(requests).split_response(response)

    assert split["request-1"].outputs[0].parameters.foo == value
    assert split["request-2"].outputs[0].parameters.foo == value
    assert type(split["request-1"].outputs[0].parameters.foo) is type(value)
    assert type(split["request-2"].outputs[0].parameters.foo) is type(value)
    assert split["request-1"].outputs[0].data.root == first_data
    assert split["request-2"].outputs[0].data.root == second_data
    assert split["request-1"].outputs[0].shape == [len(first_data)]
    assert split["request-2"].outputs[0].shape == [len(second_data)]


@pytest.mark.parametrize("batch_sizes", [(2, 1, 1), (1, 2, 1), (2, 3, 2)])
def test_split_response_matches_parameters_to_requests(
    batch_sizes: tuple[int, int, int],
) -> None:
    requests = {
        f"request-{idx}": InferenceRequest(
            id=f"external-{idx}",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[batch_size, 2],
                    data=list(range(idx * 10, idx * 10 + batch_size * 2)),
                    parameters=Parameters(
                        tag=f"tag-{idx}",
                        content_type="np",
                        headers={"x-test": "batch"},
                    ),
                )
            ],
        )
        for idx, batch_size in enumerate(batch_sizes)
    }
    batched = BatchedRequests(requests)
    response = InferenceResponse(
        model_name="test-model",
        outputs=[ResponseOutput(**batched.merged_request.inputs[0].model_dump())],
    )

    split = batched.split_response(response)

    for internal_id, request in requests.items():
        output = split[internal_id].outputs[0]
        request_input = request.inputs[0]
        assert split[internal_id].id == request.id
        assert output.parameters == request_input.parameters
        assert output.data == request_input.data
        assert output.shape == request_input.shape


@pytest.mark.parametrize(
    "value",
    [
        "123",
        123,
        False,
        ["123", "456"],
        [123, 456],
        [],
        {},
        {"key": 123},
        {"nested": {"values": []}},
    ],
)
def test_split_response_preserves_parameter_for_single_request(value):
    requests = {
        "request-1": InferenceRequest(
            id="request-1",
            inputs=[
                RequestInput(name="input", datatype="INT32", shape=[2], data=[1, 2])
            ],
        )
    }
    response = InferenceResponse(
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="output",
                datatype="INT32",
                shape=[2],
                data=[1, 2],
                parameters=Parameters(foo=value),
            )
        ],
    )

    split = BatchedRequests(requests).split_response(response)

    assert split["request-1"].outputs[0].parameters.foo == value


@pytest.mark.parametrize(
    "request_inputs, expected_request_input, expected_minibatch_sizes",
    [
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                ),
                "req-2": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                ),
                "req-3": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(foo="bar"),
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(content_type="np", foo="bar"),
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np", foo="bar"),
                ),
                "req-2": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                ),
                "req-3": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(foo="bar", time="13:00"),
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(
                    content_type="np", foo=["bar", "bar"], time="13:00"
                ),
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[2, 3],
                    data=[1, 2, 3, 10, 11, 12],
                ),
                "req-2": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[4, 5, 6]
                ),
                "req-3": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[7, 8, 9]
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[4, 3],
                data=[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8, 9],
            ),
            {"req-1": 2, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[1, 2, 3]]
                ),
                "req-2": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[4, 5, 6]]
                ),
                "req-3": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[7, 8, 9]]
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"
                ),
                "req-2": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"def"
                ),
                "req-3": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"
                ),
            },
            RequestInput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data=b"abcdefghi",
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="abc"
                ),
                "req-2": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="def"
                ),
                "req-3": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="ghi"
                ),
            },
            RequestInput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data="abcdefghi",
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
    ],
)
def test_merge_request_inputs(
    request_inputs: dict[str, RequestInput],
    expected_request_input: RequestInput,
    expected_minibatch_sizes: dict[str, int],
):
    batched = BatchedRequests()
    merged = batched._merge_request_inputs(request_inputs)

    assert merged == expected_request_input
    assert batched._minibatch_sizes == expected_minibatch_sizes


@pytest.mark.parametrize(
    "inference_requests, expected",
    [
        (
            {
                "req-1": InferenceRequest(
                    parameters=Parameters(content_type="np"),
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[1, 2, 3], shape=[1, 3]
                        )
                    ],
                ),
                "req-2": InferenceRequest(
                    parameters=Parameters(foo="bar"),
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[4, 5, 6], shape=[1, 3]
                        )
                    ],
                ),
            },
            InferenceRequest(
                parameters=Parameters(content_type="np", foo="bar"),
                inputs=[
                    RequestInput(
                        name="foo",
                        datatype="INT32",
                        data=[1, 2, 3, 4, 5, 6],
                        shape=[2, 3],
                    )
                ],
            ),
        ),
        (
            {
                "req-1": InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[1, 2, 3], shape=[1, 3]
                        ),
                        RequestInput(
                            name="bar", datatype="BYTES", data=b"abc", shape=[1, 3]
                        ),
                    ]
                ),
                "req-2": InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[4, 5, 6], shape=[1, 3]
                        ),
                        RequestInput(
                            name="bar", datatype="BYTES", data=b"def", shape=[1, 3]
                        ),
                    ]
                ),
            },
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="foo",
                        datatype="INT32",
                        data=[1, 2, 3, 4, 5, 6],
                        shape=[2, 3],
                    ),
                    RequestInput(
                        name="bar",
                        datatype="BYTES",
                        data=b"abcdef",
                        shape=[2, 3],
                    ),
                ]
            ),
        ),
    ],
)
def test_merged_request(
    inference_requests: dict[str, InferenceRequest],
    expected: InferenceRequest,
):
    batched = BatchedRequests(inference_requests)
    merged_request = batched.merged_request

    assert merged_request == expected


@pytest.mark.parametrize(
    "minibatch_sizes, response_output, expected",
    [
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(
                    content_type="np",
                    foo=["foo_1", "foo_2"],
                    bar=["bar_1", "bar_2", "bar_3"],
                ),
            ),
            [
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np", foo="foo_1", bar="bar_1"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                    parameters=Parameters(content_type="np", foo="foo_2", bar="bar_2"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(content_type="np", bar="bar_3"),
                ),
            ],
        ),
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(content_type="np"),
            ),
            [
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                    parameters=Parameters(content_type="np"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(content_type="np"),
                ),
            ],
        ),
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data=b"abcdefghi",
            ),
            [
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"def"),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"),
            ],
        ),
        (
            {"req-1": 1, "req-2": 2, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="BYTES",
                shape=[4, 3],
                data=b"abcdefjklghi",
            ),
            [
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"),
                ResponseOutput(
                    name="foo", datatype="BYTES", shape=[2, 3], data=b"defjkl"
                ),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"),
            ],
        ),
    ],
)
def test_split_response_output(
    minibatch_sizes: dict[str, int],
    response_output: ResponseOutput,
    expected: list[ResponseOutput],
):
    batched = BatchedRequests()
    batched._minibatch_sizes = minibatch_sizes
    split = batched._split_response_output(response_output)

    assert list(split.values()) == expected


@pytest.mark.parametrize(
    "inference_requests, inference_response, expected",
    [
        (
            {
                "query-1": InferenceRequest(
                    id="query-1",
                    inputs=[
                        RequestInput(
                            name="bli", datatype="FP32", shape=[1, 1], data=[0]
                        )
                    ],
                ),
                "query-2": InferenceRequest(
                    id="query-2",
                    inputs=[
                        RequestInput(
                            name="bli", datatype="FP32", shape=[1, 1], data=[0]
                        )
                    ],
                ),
            },
            InferenceResponse(
                model_name="sum-model",
                parameters=Parameters(foo="bar"),
                outputs=[
                    ResponseOutput(
                        name="foo",
                        datatype="INT32",
                        shape=[2, 2],
                        data=[1, 2, 3, 4],
                    ),
                    ResponseOutput(
                        name="bar",
                        datatype="BYTES",
                        shape=[2, 3],
                        data=b"abcdef",
                    ),
                ],
            ),
            [
                InferenceResponse(
                    id="query-1",
                    model_name="sum-model",
                    parameters=Parameters(foo="bar"),
                    outputs=[
                        ResponseOutput(
                            name="foo",
                            datatype="INT32",
                            shape=[1, 2],
                            data=[1, 2],
                        ),
                        ResponseOutput(
                            name="bar",
                            datatype="BYTES",
                            shape=[1, 3],
                            data=b"abc",
                        ),
                    ],
                ),
                InferenceResponse(
                    id="query-2",
                    model_name="sum-model",
                    parameters=Parameters(foo="bar"),
                    outputs=[
                        ResponseOutput(
                            name="foo",
                            datatype="INT32",
                            shape=[1, 2],
                            data=[3, 4],
                        ),
                        ResponseOutput(
                            name="bar",
                            datatype="BYTES",
                            shape=[1, 3],
                            data=b"def",
                        ),
                    ],
                ),
            ],
        )
    ],
)
def test_split_response(
    inference_requests: dict[str, InferenceRequest],
    inference_response: InferenceResponse,
    expected: list[InferenceResponse],
):
    batched = BatchedRequests(inference_requests)

    responses = batched.split_response(inference_response)

    assert inference_requests.keys() == responses.keys()
    assert list(responses.values()) == expected


def test_split_response_ndarray_output_data():
    """Ndarray output roots split into per-request views, matching the
    equivalent list-based splits."""
    requests = {
        "request-1": InferenceRequest(
            id="external-1",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="FP32",
                    shape=[3, 2],
                    data=[float(i) for i in range(6)],
                )
            ],
        ),
        "request-2": InferenceRequest(
            id="external-2",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="FP32",
                    shape=[5, 2],
                    data=[float(i) for i in range(6, 16)],
                )
            ],
        ),
    }
    batched = BatchedRequests(requests)
    response = InferenceResponse(
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="input",
                datatype="FP32",
                shape=[8, 2],
                data=TensorData(np.arange(16, dtype=np.float32)),
            )
        ],
    )

    split = batched.split_response(response)

    np.testing.assert_array_equal(
        split["request-1"].outputs[0].data.root, np.arange(6, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        split["request-2"].outputs[0].data.root, np.arange(6, 16, dtype=np.float32)
    )
    assert split["request-1"].outputs[0].shape == [3, 2]
    assert split["request-2"].outputs[0].shape == [5, 2]

    # The split views serialise to the same JSON as equivalent lists
    list_response = InferenceResponse(
        id="external-1",
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="input",
                datatype="FP32",
                shape=[3, 2],
                data=TensorData([float(i) for i in range(6)]),
            )
        ],
    )
    assert split["request-1"].model_dump_json() == list_response.model_dump_json()


def test_merge_request_inputs_ndarray_unequal_minibatches():
    """Ndarray-backed inputs with different minibatch sizes must merge
    by concatenating their elements (the list path's sum()
    equivalent): the merged shape declares the flattened total, and a
    list of unequal-length arrays cannot be converted by the decoder."""
    from mlserver.codecs.numpy import NumpyCodec

    req1 = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.arange(2).reshape(2, 1))]
    )
    req2 = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.arange(1).reshape(1, 1))]
    )

    batched = BatchedRequests({"req-1": req1, "req-2": req2})
    merged = batched.merged_request.inputs[0]

    assert merged.shape == [3, 1]
    np.testing.assert_array_equal(merged.data.root, np.array([0, 1, 0]))

    # The merged request decodes back to the declared shape.
    decoded = NumpyCodec.decode_input(merged)
    assert decoded.shape == (3, 1)


def test_merge_request_inputs_ndarray_equal_minibatches():
    """Equal-size ndarray minibatches concatenate the same way."""
    from mlserver.codecs.numpy import NumpyCodec

    req1 = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.array([1, 2]).reshape(2, 1))]
    )
    req2 = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.array([3, 4]).reshape(2, 1))]
    )

    batched = BatchedRequests({"req-1": req1, "req-2": req2})
    merged = batched.merged_request.inputs[0]

    assert merged.shape == [4, 1]
    np.testing.assert_array_equal(merged.data.root, np.array([1, 2, 3, 4]))


def test_merge_request_inputs_list_backed_still_flattens():
    """The list path keeps its sum() flattening for list-backed data
    (the pre-ndarray behaviour, unchanged)."""
    requests = {
        "req-1": InferenceRequest(
            inputs=[
                RequestInput(name="foo", datatype="INT32", data=[1, 2], shape=[2, 1])
            ]
        ),
        "req-2": InferenceRequest(
            inputs=[RequestInput(name="foo", datatype="INT32", data=[3], shape=[1, 1])]
        ),
    }

    batched = BatchedRequests(requests)
    merged = batched.merged_request.inputs[0]

    assert merged.shape == [3, 1]
    assert merged.data.root == [1, 2, 3]


def test_merge_request_inputs_mixed_list_and_ndarray_batches():
    """A list-backed request batched with an ndarray-backed one (a JSON
    request batched with a binary one) must concatenate in either order:
    the list path's sum() broadcasts a list against an ndarray
    ([1,2] + array([3,4]) == [4,6]), silently producing the wrong
    element count for the declared merged shape."""
    from mlserver.codecs.numpy import NumpyCodec

    ndarray_req = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.array([[3], [4]], dtype=np.int32))]
    )
    list_req = InferenceRequest(
        inputs=[RequestInput(name="foo", datatype="INT32", data=[1, 2], shape=[2, 1])]
    )

    # ndarray-first: elements follow the batch's request order.
    batched = BatchedRequests({"a": ndarray_req, "b": list_req})
    merged = batched.merged_request.inputs[0]
    assert merged.shape == [4, 1]
    np.testing.assert_array_equal(merged.data.root, np.array([3, 4, 1, 2]))

    # list-first: the order that used to broadcast instead of
    # concatenating.
    batched = BatchedRequests({"a": list_req, "b": ndarray_req})
    merged = batched.merged_request.inputs[0]
    assert merged.shape == [4, 1]
    np.testing.assert_array_equal(merged.data.root, np.array([1, 2, 3, 4]))

    # The merged request decodes back to the declared shape.
    decoded = NumpyCodec.decode_input(merged)
    assert decoded.shape == (4, 1)


def test_merge_request_inputs_uint64_precision_preserved():
    """The merge must convert entries to the tensor's declared dtype:
    numpy's uint64 + int64 promotion produces float64, which cannot
    represent 2**63 + 1 — the decoder would silently cast the rounded
    value back to uint64."""
    from mlserver.codecs.numpy import set_ndarray_response_pipeline

    set_ndarray_response_pipeline(True)
    from mlserver.codecs.numpy import NumpyCodec

    ndarray_req = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input("foo", np.array([[2**63 + 1]], dtype=np.uint64))
        ]
    )
    list_req = InferenceRequest(
        inputs=[RequestInput(name="foo", datatype="UINT64", data=[1], shape=[1, 1])]
    )

    batched = BatchedRequests({"a": ndarray_req, "b": list_req})
    merged = batched.merged_request.inputs[0]

    assert merged.shape == [2, 1]
    assert merged.data.root.dtype == np.uint64
    # The exact integer survives the merge and the decode.
    assert int(merged.data.root[0]) == 2**63 + 1
    decoded = NumpyCodec.decode_input(merged)
    assert decoded.dtype == np.uint64
    assert int(decoded[0][0]) == 2**63 + 1


def test_merge_request_inputs_nested_json_and_ndarray_batches():
    """A JSON request may carry nested data (shape [1,2] -> [[1,2]])
    while ndarray-native ones are flat. concatenate() rejects the
    dimension mismatch instead of merging, so every numeric entry
    flattens before concatenating."""
    from mlserver.codecs.numpy import set_ndarray_response_pipeline

    set_ndarray_response_pipeline(True)
    from mlserver.codecs.numpy import NumpyCodec

    ndarray_req = InferenceRequest(
        inputs=[NumpyCodec.encode_input("foo", np.array([[3, 4]], dtype=np.int32))]
    )
    nested_req = InferenceRequest(
        inputs=[RequestInput(name="foo", datatype="INT32", data=[[1, 2]], shape=[1, 2])]
    )

    # nested-first: the order that raised a ValueError from
    # concatenate() instead of merging.
    batched = BatchedRequests({"a": nested_req, "b": ndarray_req})
    merged = batched.merged_request.inputs[0]
    assert merged.shape == [2, 2]
    np.testing.assert_array_equal(merged.data.root, np.array([1, 2, 3, 4]))
    decoded = NumpyCodec.decode_input(merged)
    assert decoded.shape == (2, 2)

    # ndarray-first.
    batched = BatchedRequests({"a": ndarray_req, "b": nested_req})
    merged = batched.merged_request.inputs[0]
    assert merged.shape == [2, 2]
    np.testing.assert_array_equal(merged.data.root, np.array([3, 4, 1, 2]))


def test_merge_request_inputs_logical_bytes_not_truncated():
    """Logical byte elements merged with a fixed-width 'S' ndarray must
    keep their exact values: to_dtype() selects the sampled entry's 'S'
    dtype, and a fixed-width conversion truncates longer elements
    (b"hello" -> b"h") before the logical-elements flag is applied. The
    normalized merge uses object dtype instead."""
    from mlserver.codecs.numpy import NumpyCodec, set_logical_elements

    fixed_req = InferenceRequest(
        inputs=[
            RequestInput(
                name="foo",
                datatype="BYTES",
                data=np.array([b"a"], dtype="S1"),
                shape=[1, 1],
            )
        ]
    )
    logical_req = InferenceRequest(
        inputs=[
            RequestInput(name="foo", datatype="BYTES", data=[b"hello"], shape=[1, 1])
        ]
    )
    set_logical_elements(logical_req.inputs[0])

    # fixed-width-first: the order that truncated b"hello" to b"h".
    batched = BatchedRequests({"a": fixed_req, "b": logical_req})
    merged = batched.merged_request.inputs[0]
    assert sorted(merged.data.root) == [b"a", b"hello"]

    # logical-first.
    batched = BatchedRequests({"a": logical_req, "b": fixed_req})
    merged = batched.merged_request.inputs[0]
    assert sorted(merged.data.root) == [b"a", b"hello"]

    # The merged input decodes to the exact logical elements.
    decoded = NumpyCodec.decode_input(merged)
    assert sorted(decoded.reshape(-1).tolist()) == [b"a", b"hello"]
