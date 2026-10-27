import asyncio
import pytest

from mlserver.batching import adaptive as adaptive_module
from mlserver.batching.adaptive import AdaptiveBatcher
from mlserver.batching.requests import BatchedRequests
from mlserver.batching.shape import Shape
from mlserver.types import InferenceRequest, InferenceResponse, RequestInput
from mlserver.model import MLModel
from mlserver.utils import generate_uuid

from .conftest import RequestSender


async def test_batch_requests(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
):
    max_batch_size = adaptive_batcher._max_batch_size
    sent_requests = dict(
        await asyncio.gather(*[send_request() for _ in range(max_batch_size)])
    )

    batched_requests = [
        batched_req async for batched_req in adaptive_batcher._batch_requests()
    ]

    assert len(batched_requests) == 1
    assert batched_requests[0].inference_requests == sent_requests


async def test_batch_requests_timeout(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
):
    """
    Test that a batch size smaller than the max batch size, the timeout is hit
    and the request gets processed.
    """
    for _ in range(2):
        sent_request = dict([await send_request()])
        batched_requests = [
            batched_req async for batched_req in adaptive_batcher._batch_requests()
        ]

        assert len(batched_requests) == 1
        assert batched_requests[0].inference_requests == sent_request


async def test_batcher(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
    sum_model: MLModel,
):
    max_batch_size = adaptive_batcher._max_batch_size
    sent_requests = dict(
        await asyncio.gather(*[send_request() for _ in range(max_batch_size)])
    )

    await adaptive_batcher._batcher()

    assert sent_requests.keys() == adaptive_batcher._async_responses.keys()

    for internal_id, sent_request in sent_requests.items():
        async_response = adaptive_batcher._async_responses[internal_id]

        response = await async_response
        assert sent_request.id == response.id

        expected = await sum_model.predict(sent_request)
        assert expected == response


async def test_batcher_propagates_errors(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
    mocker,
):
    message = "This is an error"

    async def _async_exception():
        raise Exception(message)

    max_batch_size = adaptive_batcher._max_batch_size
    sent_requests = dict(
        await asyncio.gather(*[send_request() for _ in range(max_batch_size)])
    )

    adaptive_batcher._predict_fn = mocker.stub(  # type: ignore[assignment]
        "_predict_fn"
    )
    adaptive_batcher._predict_fn.return_value = (  # type: ignore[attr-defined]
        _async_exception()
    )
    await adaptive_batcher._batcher()

    for internal_id, _ in sent_requests.items():
        with pytest.raises(Exception) as err:
            await adaptive_batcher._async_responses[internal_id]

        assert str(err.value) == message


async def test_batcher_cancels_pending_responses(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
):
    requests = [await send_request() for _ in range(2)]
    unrelated_id, _ = await send_request()
    batched = BatchedRequests(dict(requests))

    prediction = asyncio.create_task(asyncio.sleep(60))
    prediction.cancel()
    await asyncio.gather(prediction, return_exceptions=True)

    adaptive_batcher._predict_callback(batched, prediction)

    for internal_id, _ in requests:
        future = adaptive_batcher._async_responses[internal_id]
        assert future.cancelled()
        with pytest.raises(asyncio.CancelledError):
            await future

    assert not adaptive_batcher._async_responses[unrelated_id].done()


async def test_batcher_ignores_missing_or_completed_response_on_cancellation(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
):
    completed_id, completed_request = await send_request()
    missing_id, missing_request = await send_request()
    batched = BatchedRequests(
        {completed_id: completed_request, missing_id: missing_request}
    )

    completed_response = InferenceResponse(
        model_name=adaptive_batcher._model.name,
        outputs=[],
    )
    completed_future = adaptive_batcher._async_responses[completed_id]
    completed_future.set_result(completed_response)
    adaptive_batcher._async_responses.pop(missing_id)

    prediction = asyncio.create_task(asyncio.sleep(60))
    prediction.cancel()
    await asyncio.gather(prediction, return_exceptions=True)

    adaptive_batcher._predict_callback(batched, prediction)

    assert completed_future.result() is completed_response
    assert missing_id not in adaptive_batcher._async_responses


async def test_batcher_ignores_cancelled_response(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
    sum_model: MLModel,
):
    cancelled_id, cancelled_request = await send_request()
    active_id, active_request = await send_request()

    # Model the state left by a caller cancelling predict() while waiting.
    adaptive_batcher._async_responses.pop(cancelled_id)
    batched = BatchedRequests(
        {
            cancelled_id: cancelled_request,
            active_id: active_request,
        }
    )
    prediction = asyncio.create_task(sum_model.predict(batched.merged_request))
    await prediction

    adaptive_batcher._predict_callback(batched, prediction)

    response = await adaptive_batcher._async_responses[active_id]
    assert response.id == active_request.id


async def test_batch_requests_skips_cancelled_request(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
):
    internal_id, _ = await send_request()
    adaptive_batcher._async_responses[internal_id].cancel()

    batched_requests = [
        batched_req async for batched_req in adaptive_batcher._batch_requests()
    ]

    assert batched_requests == []


async def test_batch_requests_timeout_includes_cancelled_requests(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
    mocker,
):
    """Cancelled requests must not extend the deadline for live requests."""
    adaptive_batcher._max_batch_size = 10
    adaptive_batcher._max_batch_time = 0.1

    requests = [await send_request() for _ in range(4)]
    for internal_id, _ in requests[1:]:
        adaptive_batcher._async_responses[internal_id].cancel()

    observed_timeouts = []

    async def get_request(timeout):
        observed_timeouts.append(timeout)
        return adaptive_batcher._requests.get_nowait()

    # Control the queue reads and monotonic clock so this test does not depend
    # on event-loop scheduling or wall-clock timing.
    mocker.patch.object(adaptive_batcher, "_get_request", side_effect=get_request)
    mocker.patch.object(
        adaptive_module,
        "monotonic",
        side_effect=[0.0, 0.0, 0.08, 0.08, 0.08, 0.1],
    )

    batch = await anext(adaptive_batcher._batch_requests())

    assert len(batch.inference_requests) == 1
    assert batch.inference_requests[requests[0][0]] == requests[0][1]
    assert observed_timeouts == pytest.approx([0.1, 0.02, 0.02, 0.02])


async def test_batch_requests_drains_available_request_at_deadline(
    adaptive_batcher: AdaptiveBatcher,
    send_request: RequestSender,
    mocker,
):
    """An expired deadline must not leave available requests undrained."""
    adaptive_batcher._max_batch_size = 10
    adaptive_batcher._max_batch_time = 0.1
    internal_id, request = await send_request()
    observed_timeouts = []

    async def get_request(timeout):
        observed_timeouts.append(timeout)
        return adaptive_batcher._requests.get_nowait()

    mocker.patch.object(adaptive_batcher, "_get_request", side_effect=get_request)
    mocker.patch.object(
        adaptive_module,
        "monotonic",
        side_effect=[0.0, 0.1, 0.1],
    )

    batch = await anext(adaptive_batcher._batch_requests())

    assert batch.inference_requests == {internal_id: request}
    assert observed_timeouts == pytest.approx([0.0])


async def test_batcher_cancels_responses(
    adaptive_batcher: AdaptiveBatcher,
    mocker,
):
    message = "This is an error"

    async def _async_exception():
        raise Exception(message)

    num_requests = adaptive_batcher._max_batch_size * 2 + 2

    adaptive_batcher._batcher = mocker.stub("_batcher")  # type: ignore[method-assign]
    adaptive_batcher._batcher.side_effect = iter(  # type: ignore[attr-defined]
        _async_exception, None
    )

    requests = [
        InferenceRequest(
            id=generate_uuid(),
            inputs=[
                RequestInput(
                    name="input-0",
                    shape=[1, 3],
                    datatype="INT32",
                    data=[idx, idx + 1, idx + 2],
                )
            ],
        )
        for idx in range(num_requests)
    ]

    responses = await asyncio.gather(
        *[adaptive_batcher.predict(request) for request in requests],
        return_exceptions=True,
    )

    for response in responses:
        assert isinstance(response, Exception)
        assert str(response) == message


@pytest.mark.parametrize(
    "requests",
    [
        [
            InferenceRequest(
                id=f"request-{idx}",
                inputs=[
                    RequestInput(
                        name="input-0",
                        shape=[1, 3],
                        datatype="INT32",
                        data=[idx, idx + 1, idx + 2],
                    )
                ],
            )
            # 10 is the max_batch_size for sum_model
            # Make sure one batch is only half-full
            for idx in range(10 * 2 + 2)
        ],
        [
            InferenceRequest(
                id="large-request",
                inputs=[
                    # 10 is the max batch size, so we send a minibatch with
                    # 20 entries
                    RequestInput(
                        name="input-0",
                        shape=[10 * 2, 3],
                        datatype="INT32",
                        data=[n for n in range(10 * 2 * 3)],
                    )
                ],
            ),
            InferenceRequest(
                id="regular-request",
                inputs=[
                    RequestInput(
                        name="input-0",
                        shape=[1, 3],
                        datatype="INT32",
                        data=[1000, 1001, 1002],
                    )
                ],
            ),
        ],
    ],
)
async def test_predict(
    requests: list[InferenceRequest],
    adaptive_batcher: AdaptiveBatcher,
    sum_model: MLModel,
):
    responses = await asyncio.gather(
        *[adaptive_batcher.predict(request) for request in requests]
    )

    assert len(requests) == len(responses)
    for req, res in zip(requests, responses):
        assert req.id == res.id

        req_shape = Shape(req.inputs[0].shape)
        res_shape = Shape(res.outputs[0].shape)
        assert req_shape.batch_size == res_shape.batch_size

        expected = await sum_model.predict(req)
        assert res == expected
