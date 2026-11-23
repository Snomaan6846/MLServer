from unittest.mock import AsyncMock, Mock

import grpc

from mlserver.grpc.utils import handle_mlserver_error, handle_mlserver_stream_error


def _raising_handler(exc: BaseException):
    async def handler(self, request, context):
        raise exc

    return handler


async def test_inference_timeout_maps_to_deadline_exceeded():
    """An expired inference deadline (the dispatch wait's dedicated
    InferenceTimeoutError) aborts with DEADLINE_EXCEEDED, not INTERNAL
    — gRPC clients must be able to tell a timing failure from a server
    fault apart."""
    from mlserver.errors import InferenceTimeoutError

    context = Mock()
    context.abort = AsyncMock()

    decorated = handle_mlserver_error(_raising_handler(InferenceTimeoutError()))
    await decorated(Mock(), object(), context)

    context.abort.assert_awaited_once()
    assert context.abort.await_args.kwargs["code"] == grpc.StatusCode.DEADLINE_EXCEEDED


async def test_foreign_timeout_keeps_internal():
    """A bare TimeoutError from any other await inside request handling
    is NOT an inference deadline: it keeps INTERNAL, as it always
    was."""
    context = Mock()
    context.abort = AsyncMock()

    decorated = handle_mlserver_error(_raising_handler(TimeoutError()))
    await decorated(Mock(), object(), context)

    context.abort.assert_awaited_once()
    assert context.abort.await_args.kwargs["code"] == grpc.StatusCode.INTERNAL


async def test_stream_inference_timeout_maps_to_deadline_exceeded():
    """The streaming variant maps inference deadlines identically."""
    from mlserver.errors import InferenceTimeoutError

    @handle_mlserver_stream_error
    async def stream_handler(self, request_stream, context):
        raise InferenceTimeoutError()
        yield  # pragma: no cover - makes this an async generator

    context = Mock()
    context.abort = AsyncMock()

    async for _ in stream_handler(Mock(), Mock(), context):
        pass

    context.abort.assert_awaited_once()
    assert context.abort.await_args.kwargs["code"] == grpc.StatusCode.DEADLINE_EXCEEDED


async def test_mlserver_error_still_maps_by_status():
    """The pre-existing MLServerError mapping is unchanged by the
    timeout branch."""
    from fastapi import status

    from mlserver.errors import MLServerError

    context = Mock()
    context.abort = AsyncMock()

    decorated = handle_mlserver_error(
        _raising_handler(MLServerError("boom", status_code=status.HTTP_404_NOT_FOUND))
    )
    await decorated(Mock(), object(), context)

    assert context.abort.await_args.kwargs["code"] == grpc.StatusCode.NOT_FOUND
