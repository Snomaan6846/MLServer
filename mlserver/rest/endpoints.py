from fastapi.requests import Request
from fastapi.responses import Response, HTMLResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html

from collections.abc import AsyncIterator, Callable
from contextlib import suppress
import asyncio

from ..types import (
    MetadataModelResponse,
    MetadataServerResponse,
    InferenceRequest,
    InferenceResponse,
    RepositoryIndexRequest,
    RepositoryIndexResponse,
    RuntimeSecurityResponse,
)
from ..handlers import DataPlane, ModelRepositoryHandlers
from ..batching.adaptive import streaming_bypass
from ..utils import insert_headers, extract_headers

from .responses import ServerSentEvent
from .openapi import get_openapi_schema, get_model_schema_uri, get_model_schema
from .utils import encode_inference_response_bytes, to_status_code


class Endpoints:
    """
    Implementation of REST endpoints.
    These take care of the REST/HTTP-specific things and then delegate the
    business logic to the internal handlers.
    """

    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane

    async def live(self) -> Response:
        is_live = await self._data_plane.live()
        return Response(status_code=to_status_code(is_live))

    async def ready(self) -> Response:
        is_ready = await self._data_plane.ready()
        return Response(status_code=to_status_code(is_ready))

    async def openapi(self) -> dict:
        return get_openapi_schema()

    async def docs(self) -> HTMLResponse:
        openapi_url = "/v2/docs/dataplane.json"
        title = "MLServer API Docs"
        return get_swagger_ui_html(openapi_url=openapi_url, title=title)

    async def model_openapi(
        self, model_name: str, model_version: str | None = None
    ) -> dict:
        # NOTE: Right now, we use the `model_metadata` method to check that the
        # model exists.
        # In the future, we will use this metadata to fill in more model
        # details in the schema (e.g. expected inputs, etc.).
        await self._data_plane.model_metadata(model_name, model_version)
        return get_model_schema(model_name, model_version)

    async def model_docs(
        self, model_name: str, model_version: str | None = None
    ) -> HTMLResponse:
        # NOTE: Right now, we use the `model_metadata` method to check that the
        # model exists.
        # In the future, we will use this metadata to fill in more model
        # details in the schema (e.g. expected inputs, etc.).
        await self._data_plane.model_metadata(model_name, model_version)
        openapi_url = get_model_schema_uri(model_name, model_version)

        title = f"MLServer API Docs - {model_name}"
        if model_version:
            title = f"{title} ({model_version})"

        return get_swagger_ui_html(openapi_url=openapi_url, title=title)

    async def model_ready(
        self, model_name: str, model_version: str | None = None
    ) -> Response:
        is_ready = await self._data_plane.model_ready(model_name, model_version)
        return Response(status_code=to_status_code(is_ready))

    async def metadata(self) -> MetadataServerResponse:
        return await self._data_plane.metadata()

    async def runtimes(self) -> RuntimeSecurityResponse:
        return await self._data_plane.runtimes()

    async def model_metadata(
        self, model_name: str, model_version: str | None = None
    ) -> MetadataModelResponse:
        return await self._data_plane.model_metadata(model_name, model_version)

    async def infer(
        self,
        raw_request: Request,
        payload: InferenceRequest,
        model_name: str,
        model_version: str | None = None,
    ) -> Response:
        # Pre-encoded response (the unary fast path): FastAPI skips
        # response_model serialization for pre-built Response returns,
        # so the python-mode model_dump runs — the ndarray root flows
        # to orjson's OPT_SERIALIZE_NUMPY and renders straight from
        # the buffer (~3.5x faster than the tolist + object-tree walk).
        # Byte-identical output to the list path is
        # pinned by the wire-identity suite. CloudEvents response
        # headers (extract_headers) move onto the returned Response —
        # the injected raw_response is ignored when a Response is
        # returned, so its headers would be lost otherwise.
        request_headers = dict(raw_request.headers)
        insert_headers(payload, request_headers)

        inference_response = await self._data_plane.infer(
            payload, model_name, model_version
        )

        # Extract (and clear) the transport headers BEFORE
        # serialization: extract_headers mutates
        # parameters.headers to None, so serializing first would leak
        # every request header (x-*, Ce-*) into the response body —
        # the ordering the response-model validation path guaranteed
        # implicitly by extracting before returning the model.
        response_headers = extract_headers(inference_response)

        # One serializer for the frontend and the worker (see
        # utils.encode_inference_response_bytes): native tensor
        # encoding with the metadata's JSON-mode scalar conversion.
        content = encode_inference_response_bytes(inference_response)
        response = Response(content=content, media_type="application/json")
        if response_headers:
            response.headers.update(response_headers)

        return response

    @staticmethod
    async def _first_event_or_disconnect(
        queue: "asyncio.Queue[bytes | Exception | None]",
        raw_request: Request,
        start_settlement: "Callable[[], asyncio.Task[None]]",
    ) -> "bytes | Exception | None":
        """The first SSE event, racing the client disconnect."""
        # A disconnected client must not leave the model stream
        # running: settle the producer (the ONCE-ONLY settlement — a
        # direct cancel here would race the response's own cleanup
        # into a second cancellation of a producer mid-cleanup), then
        # let the endpoint end (the response goes nowhere). On the
        # first event, the disconnect watcher is cancelled — its
        # pending receive consumes nothing, and the StreamingResponse
        # reads the channel fresh.
        get_task = asyncio.create_task(queue.get())
        disconnect_task = asyncio.create_task(_wait_for_disconnect(raw_request))
        try:
            done, _pending = await asyncio.wait(
                {get_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            # The request was cancelled while waiting: START the
            # settlement first (synchronous — it cannot be
            # interrupted), then clean the watchers, then let the
            # cancellation propagate (the raise). A cancellation
            # arriving during the watcher gather leaves the
            # settlement already running, not skipped.
            settlement_task = start_settlement()
            get_task.cancel()
            disconnect_task.cancel()
            await asyncio.gather(get_task, disconnect_task, return_exceptions=True)
            await asyncio.shield(settlement_task)
            raise
        disconnect_task.cancel()
        if disconnect_task in done:
            # The client is gone: settle the model stream now, not
            # when the first event eventually arrives.
            get_task.cancel()
            # If the first event landed in the same instant, retrieve
            # its exception (if any) so a completed task is never GC'd
            # with an unretrieved one.
            if get_task.done() and not get_task.cancelled():
                get_task.exception()
            # The settlement starts BEFORE the watcher is inspected:
            # a watcher that completed by CANCELLATION raises from
            # .exception(), and the producer must already be settling
            # by then.
            settlement_task = start_settlement()
            await asyncio.gather(get_task, return_exceptions=True)
            await asyncio.shield(settlement_task)
            if disconnect_task.cancelled():
                # The watcher itself was cancelled — not a client
                # disconnect: propagate the cancellation (499 is
                # reserved for an actual http.disconnect).
                raise asyncio.CancelledError()
            # A watcher that completed by EXCEPTION is a receive
            # FAILURE (a broken transport), not a client disconnect:
            # surface it after settling.
            receive_error = disconnect_task.exception()
            if receive_error is not None:
                raise receive_error
            raise _ClientDisconnected()
        # First event: JOIN the cancelled disconnect watcher before
        # handing the receive channel to the StreamingResponse — its
        # pending receive must unwind first (an ASGI server rejects a
        # second concurrent read of the channel).
        try:
            await asyncio.gather(disconnect_task, return_exceptions=True)
        except BaseException:
            # Cancelled during the handoff: same
            # cancellation-safe cleanup.
            settlement_task = start_settlement()
            await asyncio.shield(settlement_task)
            raise
        return get_task.result()

    async def infer_stream(
        self,
        raw_request: Request,
        payload: InferenceRequest,
        model_name: str,
        model_version: str | None = None,
    ) -> Response:
        # Response (not StreamingResponse): the single-shot fast path
        # below returns a pre-encoded plain Response; the producer path
        # returns a StreamingResponse — both are Response subclasses.

        request_headers = dict(raw_request.headers)
        insert_headers(payload, request_headers)

        # Single-shot fast path: when the model's predict_stream is the
        # base-class fallback, every /infer_stream request produces
        # exactly one response — semantically identical to the regular
        # infer path (base fallback: `yield await self.predict(...)` on
        # the first payload, prepared identically by the data plane).
        # Running the full inference in the endpoint task surfaces
        # errors as clean HTTP failures, keeps the data plane's context
        # managers in one asyncio context, needs no disconnect cleanup,
        # and skips the producer/queue coordination entirely — the
        # measured tail cost of the SSE hardening. Byte-identical SSE
        # body: the same ServerSentEvent encoding of the same response.
        # Resolve the active model ONCE: the streaming-mode decision
        # and the execution must use the same instance. A reload can
        # swap the registration between a cached decision and the
        # inference call — the single-shot cache may hold the OLD
        # model's answer while the registry now serves the new one
        # (whose predict_stream override differs).
        model = await self._data_plane.resolve_model(model_name, model_version)
        if model.stream_is_single_shot():
            # Race the inference against the client disconnect: the
            # base's StreamingResponse watched for http.disconnect
            # while streaming; a pre-encoded Response has no such
            # watch, so a client leaving mid-inference would leave
            # the model running with nobody reading the result.
            async def _infer_single_shot() -> InferenceResponse:
                # Hold the batcher-bypass mark for this task: a
                # single-shot stream is semantically a plain infer and
                # must not wait for a batch window (the streaming
                # wrapper's bypass, which this shortcut replaces).
                with streaming_bypass():
                    return await self._data_plane.infer(
                        payload, model_name, model_version, model=model
                    )

            infer_task = asyncio.create_task(_infer_single_shot())
            disconnect_task = asyncio.create_task(_wait_for_disconnect(raw_request))
            # The same once-only settlement pattern as the streaming
            # path, scoped to this inference: the first caller
            # cancels and settles it; a request cancellation is NOT
            # routed into the inference task a second time (its
            # model's async finally may be mid-release), and the
            # settlement continues shielded in the background.
            infer_settlement: "asyncio.Task[None] | None" = None

            def _start_infer_settlement() -> "asyncio.Task[None]":
                nonlocal infer_settlement
                if infer_settlement is None:

                    async def _settle_infer() -> None:
                        infer_task.cancel()
                        await asyncio.gather(infer_task, return_exceptions=True)

                    infer_settlement = asyncio.ensure_future(_settle_infer())
                return infer_settlement

            try:
                done, _pending = await asyncio.wait(
                    {infer_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                # The request was cancelled while waiting: settle the
                # inference FIRST (synchronous start), then clean the
                # watcher, then let the cancellation propagate.
                infer_settle = _start_infer_settlement()
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
                await asyncio.shield(infer_settle)
                raise
            disconnect_task.cancel()
            if disconnect_task in done:
                # The client is gone: settle the inference now, not
                # when it eventually completes. The settlement starts
                # BEFORE the watcher is inspected (a cancelled watcher
                # raises from .exception()); the request cancellation
                # is not absorbed.
                infer_settle = _start_infer_settlement()
                await asyncio.shield(infer_settle)
                if disconnect_task.cancelled():
                    # The watcher itself was cancelled — not a client
                    # disconnect: propagate the cancellation (499 is
                    # reserved for an actual http.disconnect).
                    raise asyncio.CancelledError()
                # Same as the prefetch: a receive FAILURE is not a
                # client disconnect — surface it after settling; 499
                # is reserved for an actual http.disconnect.
                receive_error = disconnect_task.exception()
                if receive_error is not None:
                    raise receive_error
                return Response(status_code=499)
            prediction = infer_task.result()
            return Response(
                content=ServerSentEvent(prediction).encode(),
                media_type="text/event-stream",
            )

        async def payloads_generator(
            payload: InferenceRequest,
        ) -> AsyncIterator[InferenceRequest]:
            yield payload

        payloads = payloads_generator(payload)
        infer_stream = self._data_plane.infer_stream(
            payloads, model_name, model_version, model=model
        )
        sse_stream = _as_sse(infer_stream)

        # Surface stream errors BEFORE the response starts: once the SSE
        # headers are sent, a failing generator can only truncate the
        # stream — an empty 200 the client cannot distinguish from a
        # legitimate one. The stream runs in a producer task so its
        # context managers (ContextVar set/reset in the data plane)
        # enter AND exit within one asyncio context; the endpoint and
        # starlette read from the queue without touching them.
        # maxsize=1 restores the natural backpressure of direct
        # iteration: a fast model with a slow client buffers at most
        # one encoded event, not the entire stream.
        queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(maxsize=1)

        async def producer():
            try:
                async for event in sse_stream:
                    await queue.put(event)
                await queue.put(None)  # end-of-stream sentinel
            except Exception as exc:  # noqa: BLE001 - forwarded to the reader
                await queue.put(exc)
            finally:
                # Explicitly close the inference iterator from the
                # PRODUCER's context: the data plane's context
                # managers (ContextVar set/reset) unwind here, in the
                # task that entered them. Relying on the generator's
                # finally alone is insufficient — a generator stranded
                # at a yield by a cancelled ASGI sender may never run
                # its finally, and the context reset error the probe
                # observed is exactly that.
                with suppress(asyncio.CancelledError, Exception):
                    await sse_stream.aclose()

        task = asyncio.create_task(producer())

        # The request task outlives this endpoint call: it runs the
        # whole ASGI response, Starlette streaming the body within
        # it. When it ends — normally, via ClientDisconnect (the
        # ASGI send raises on a disconnected client), or cancelled —
        # an async-for exited by an exception or cancellation does
        # NOT close its iterator, so the body generator above can be
        # stranded at a yield with its finally unrun. The producer
        # would then stay blocked on the full queue forever. Settle
        # it from the request task's own completion instead: the
        # done-callback fires on every completion path, on every
        # ASGI server.
        request_task = asyncio.current_task()

        # ONE settlement operation per producer: the first caller
        # cancels and settles it; every later caller (the response's
        # ASGI cleanup, the body iterator's cleanup, the request-task
        # backstop) awaits the SAME settlement. A second cancel would
        # interrupt the model's async cleanup (its finally may await
        # resource release) mid-release.
        settlement: "asyncio.Task[None] | None" = None

        async def _settle_once() -> None:
            try:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            finally:
                # The settlement completed: the backstop callback on
                # the (possibly long-lived) ASGI caller task has
                # nothing left to settle. Removing it releases this
                # closure chain — a reused caller task would
                # otherwise accumulate one callback (and its producer
                # and settlement state) per completed stream.
                if request_task is not None:
                    try:
                        request_task.remove_done_callback(settle_if_pending)
                    except ValueError:
                        pass  # already removed (or never registered)

        def _start_settlement() -> "asyncio.Task[None]":
            """Ensure the once-only settlement exists and return it.

            Synchronous by design: STARTING the settlement cannot be
            interrupted by a request cancellation, so a cancellation
            arriving anywhere in the cleanup below still leaves the
            producer settled (the task runs to completion shielded).
            """
            nonlocal settlement
            if settlement is None:
                settlement = asyncio.ensure_future(_settle_once())
            return settlement

        async def settle_producer() -> None:
            try:
                await asyncio.shield(_start_settlement())
            except asyncio.CancelledError:
                # The caller itself was cancelled while awaiting:
                # the settlement continues in the background (the
                # shield); a later caller can still await it.
                pass

        def settle_if_pending(done: asyncio.Task) -> None:
            if task.done():
                return  # already settled (normal end or prefetch path)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - loop shutting down
                return
            loop.create_task(settle_producer())

        if request_task is not None:
            request_task.add_done_callback(settle_if_pending)

        # Await the first event (or the error) before returning: a
        # rejection surfaces as a clean HTTP error here. Until the
        # StreamingResponse starts, though, NOTHING listens on the
        # ASGI receive channel — a client that disconnects during the
        # prefetch would otherwise leave the endpoint and the model
        # stream pending until the first event arrives. Race the
        # first-event wait against the disconnect.
        try:
            first = await self._first_event_or_disconnect(
                queue, raw_request, _start_settlement
            )
        except _ClientDisconnected:
            # The client is gone; the response goes nowhere, but the
            # endpoint completes normally (a routine disconnect is
            # not an error) and the model stream is settled.
            return Response(status_code=499)
        if isinstance(first, Exception):
            # Raw shield: the settlement continues while a request
            # cancellation propagates.
            await asyncio.shield(_start_settlement())
            raise first
        if first is None:
            # Empty stream: return an empty SSE response.
            await asyncio.shield(_start_settlement())

            async def _empty_sse() -> AsyncIterator[bytes]:
                return
                yield  # pragma: no cover - makes this an async generator

            return StreamingResponse(
                content=_empty_sse(), media_type="text/event-stream"
            )

        async def sse_from_queue() -> AsyncIterator[bytes]:
            try:
                yield first
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        # A late producer error must not look like a
                        # normal completion: propagate it (the ASGI
                        # sender surfaces it as an abnormal response
                        # end, matching the direct-iteration behavior
                        # the producer/queue path replaced). The
                        # finally shields the settlement on both
                        # paths.
                        raise item
                    yield item
            finally:
                # The once-only settlement, awaited WITHOUT absorbing
                # a request cancellation: the settlement continues
                # (shielded) while the cancellation propagates — a
                # cancelled request must not complete normally (the
                # terminal frame would follow a recorded
                # cancellation). Any concurrent caller (the response's
                # ASGI cleanup) awaits the SAME settlement instead of
                # issuing a second cancel.
                await asyncio.shield(_start_settlement())

        return _SSEStreamingResponse(
            content=sse_from_queue(),
            start_settlement=_start_settlement,
            media_type="text/event-stream",
        )


class _SSEStreamingResponse(StreamingResponse):
    """A streaming response that OWNS its producer's settlement.

    The ASGI call exits on completion, on send failure (a
    disconnected client raises through Starlette's sender), or on
    cancellation — and an async-for exited by an exception or
    cancellation does NOT close its iterator, so the body generator
    can be stranded at a yield with its finally unrun. The
    cancel-and-settle therefore also runs from the response's own
    ASGI cleanup, not only from the body generator's finally or the
    request task's completion callback (those remain as backstops:
    the settle is idempotent).
    """

    def __init__(
        self,
        content: AsyncIterator[bytes],
        start_settlement: "Callable[[], asyncio.Task[None]]",
        **kwargs,
    ) -> None:
        super().__init__(content=content, **kwargs)
        self._start_settlement = start_settlement

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # The once-only settlement, awaited WITHOUT absorbing a
            # request cancellation (the same pattern as the prefetch
            # boundary): a cancelled request must end cancelled, not
            # return normally with the cleanup still pending. The
            # shield keeps the settlement running in the background
            # either way.
            await asyncio.shield(self._start_settlement())


async def _wait_for_disconnect(request: Request) -> None:
    """Return when the client disconnects (blocks until then)."""
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


class _ClientDisconnected(Exception):
    """The client went away before the first SSE event."""


async def _as_sse(
    infer_stream: AsyncIterator[InferenceResponse],
) -> AsyncIterator[bytes]:
    """
    Helper to convert all the responses coming out of a generator to a
    Server-Sent Event object.
    """
    try:
        async for inference_response in infer_stream:
            # TODO: How should we send headers back?
            # response_headers = extract_headers(inference_response)
            yield ServerSentEvent(inference_response).encode()
    finally:
        # Closing the wrapper must settle the stream it wraps: when
        # the producer is cancelled mid-iteration (client disconnect
        # under backpressure), this generator is suspended at a yield
        # and the model's generator with it — without this, the
        # model's generator is left to the event loop's async-gen
        # finalisation (non-deterministic, possibly never).
        aclose = getattr(infer_stream, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()


class ModelRepositoryEndpoints:
    def __init__(self, handlers: ModelRepositoryHandlers):
        self._handlers = handlers

    async def index(self, payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
        return await self._handlers.index(payload)

    async def load(self, model_name: str) -> Response:
        loaded = await self._handlers.load(name=model_name)
        return Response(status_code=to_status_code(loaded))

    async def unload(self, model_name: str) -> Response:
        unloaded = await self._handlers.unload(name=model_name)
        return Response(status_code=to_status_code(unloaded))
