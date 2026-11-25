"""REST inference endpoints with KServe v2 binary tensor support.

When ``rest_binary_tensor`` is enabled, requests carrying the
``Inference-Header-Content-Length`` header are parsed as binary tensor
data (JSON header + raw bytes).  All other requests are parsed as
standard JSON — delegating to the ``rest_worker_json`` endpoint when
that setting is also enabled, so worker-side JSON processing is
preserved.  Both paths use the existing DataPlane for inference and
follow the client's per-output binary/JSON response negotiation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import os
from time import perf_counter
from typing import Optional

from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import Response as FastAPIResponse

from ..errors import ModelNotReady
from ..types import InferenceRequest, InferenceResponse
from ..handlers import DataPlane
from ..utils import insert_headers, extract_headers

from .binary import (
    BinaryTensorError,
    encode_binary_response,
    get_header_length,
    is_binary_request,
    parse_binary_request,
    resolve_binary_outputs,
)
from .worker import (
    RAW_INFER_METHOD,
    RawReply,
    _CONVERSION_DURATION_ATTR,
    _INFERENCE_SUCCEEDED_ATTR,
    _http_response,
    _json_content,
    _parse_worker_binary,
    encode_response,
    parse_body,
)

# Response headers the transport owns.  Model-supplied values for these
# are dropped before the response is constructed: a model's
# ``content-type`` must not relabel a binary frame as JSON, and an echoed
# ``inference-header-content-length`` must not fork into two values.
_TRANSPORT_OWNED_HEADERS = frozenset(
    {"content-type", "content-length", "inference-header-content-length"}
)


def wants_binary_reply(request: InferenceRequest) -> bool:
    """Whether the request's negotiation asks for any binary output.

    Per KServe semantics: ``parameters.binary_data_output`` is the
    request-wide default and each requested output's
    ``parameters.binary_data`` overrides it. When outputs were
    requested explicitly, the per-output resolution alone decides
    (an all-JSON negotiation returns plain JSON even under a True
    default); the request-wide default only applies when no outputs
    were requested.
    """
    default_binary, per_output = resolve_binary_outputs(request)
    if per_output is not None:
        return any(per_output.values())
    return default_binary


def encode_worker_binary_reply(
    response: InferenceResponse, request: InferenceRequest
) -> RawReply:
    """Encode a worker-side reply for a binary tensor request.

    Mirrors ``BinaryTensorEndpoints._respond``: each output's format is
    resolved from the request's negotiation and the reply carries the
    framing headers the frontend response would set. Binary replies
    include their own ``content-type`` because the shared raw-response
    wrapper defaults to JSON.
    """
    default_binary, per_output = resolve_binary_outputs(request)
    # The single predicate definition (also used by the worker's JSON
    # dispatch); the double resolve is a few dict lookups.
    wants_binary = wants_binary_reply(request)

    # extract_headers also clears the internal parameters.headers
    # representation so it does not leak into the encoded response.
    extracted = extract_headers(response) or {}
    response_headers = {
        key: value
        for key, value in extracted.items()
        if str(key).lower() not in _TRANSPORT_OWNED_HEADERS
    }

    if wants_binary:
        body, header_length = encode_binary_response(
            response, default_binary, per_output
        )
        response_headers["Inference-Header-Content-Length"] = str(header_length)
        response_headers["content-type"] = "application/octet-stream"
        return RawReply(body, response_headers)

    return RawReply(encode_response(response), response_headers)


# Signature of the JSON inference handler this endpoint can delegate to
# (``RawEndpoints.infer`` when ``rest_worker_json`` is enabled).
JsonInfer = Callable[..., Awaitable[FastAPIResponse]]


class BinaryTensorEndpoints:
    """Handles binary tensor and standard JSON inference requests.

    Follows the same pattern as ``RawEndpoints`` (rest_worker_json):
    takes a raw ``Request`` and handles body parsing for both binary
    and JSON formats, bypassing FastAPI's automatic body parsing.
    JSON requests are delegated to ``json_infer`` when provided, so the
    ``rest_worker_json`` optimization keeps applying to ordinary JSON
    traffic while both settings are enabled.
    """

    def __init__(
        self,
        data_plane: DataPlane,
        json_infer: Optional[JsonInfer] = None,
        worker_binary: bool = False,
    ):
        self._data_plane = data_plane
        self._json_infer = json_infer
        self._worker_binary = worker_binary

    async def infer(
        self,
        request: Request,
        model_name: str,
        model_version: str | None = None,
    ) -> FastAPIResponse:
        headers = dict(request.headers)

        if is_binary_request(headers):
            return await self._infer_binary(request, headers, model_name, model_version)

        if self._json_infer is not None:
            # Every JSON request delegates: the worker resolves
            # binary-output negotiation from the PARSED request (exact,
            # post-parse — the same KServe semantics the local path
            # applies) and encodes the reply with the shared functions.
            # The pre-parse marker scan this replaces cost ~5.5 ms of
            # frontend CPU on a 15 MB body whenever any underscore
            # appeared (request IDs, content_type parameter keys).
            return await self._json_infer(request, model_name, model_version)

        return await self._infer_json(request, headers, model_name, model_version)

    async def _read_body(self, request: Request) -> bytes:
        """Read the request body, mapping read errors to a JSON 400.

        A Request-only endpoint bypasses FastAPI's body parser,
        including its conversion of read/decompression errors into a
        JSON 400 — mirror that behaviour here (gzip framing errors,
        truncated bodies, etc.).
        """
        try:
            return await request.body()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=400, detail="There was an error parsing the body"
            ) from error

    async def _infer_json(
        self,
        request: Request,
        headers: dict[str, str],
        model_name: str,
        model_version: str | None = None,
        body: bytes | None = None,
    ) -> FastAPIResponse:
        """Standard JSON path — parse, infer, respond per negotiation.

        Reuses ``parse_body`` from the worker module so error locations
        and validation semantics match the rest of the codebase.  JSON
        requests may still request binary outputs (KServe allows it).
        """
        if body is None:
            body = await self._read_body(request)
        payload = parse_body(body, _json_content(headers))
        if isinstance(payload, RawReply):
            return self._raw_reply_response(payload)

        insert_headers(payload, headers)
        inference_response = await self._data_plane.infer(
            payload, model_name, model_version
        )
        return self._respond(inference_response, payload)

    async def _infer_binary(
        self,
        request: Request,
        headers: dict[str, str],
        model_name: str,
        model_version: str | None = None,
    ) -> FastAPIResponse:
        """Binary tensor path — parse binary, infer, respond per negotiation."""
        body = await self._read_body(request)
        if self._worker_binary:
            reply = await self._dispatch_worker_binary(
                body, headers, model_name, model_version
            )
            if reply is not None:
                return _http_response(reply)
        try:
            header_length = get_header_length(headers)
            payload, _ = parse_binary_request(body, header_length)
        except BinaryTensorError as error:
            raise HTTPException(status_code=400, detail=error.detail)

        insert_headers(payload, headers)
        inference_response = await self._data_plane.infer(
            payload, model_name, model_version
        )
        return self._respond(inference_response, payload)

    async def _dispatch_worker_binary(
        self,
        body: bytes,
        headers: dict[str, str],
        model_name: str,
        model_version: str | None = None,
    ) -> Optional[RawReply]:
        """Forward a binary tensor frame to the model's inference worker.

        Mirrors ``RawEndpoints.infer``'s dispatch: the raw bytes and HTTP
        headers cross the pool untouched, the worker parses the binary
        frame, runs inference and encodes the negotiated response, and
        inference accounting stays in the frontend across worker
        restarts. Returns ``None`` when the model cannot take the raw
        worker path (not parallel, or adaptive batching enabled) so the
        caller keeps the local frontend parsing unchanged.
        """
        from ..parallel.messages import ModelRequestMessage
        from ..parallel.model import ParallelModel

        with self._data_plane._infer_metrics(model_name, model_version) as metrics:
            try:
                model = await self._data_plane._model_registry.get_model(
                    model_name, model_version
                )
                if not model.ready:
                    raise ModelNotReady(model_name, model_version)
            except Exception:
                # FastAPI validates before model lookup. Only valid inference
                # requests belong to accounting; exclude this exceptional-path
                # validation just as the standard route excludes JSON parsing.
                # A malformed frame takes priority over the model error — the
                # same short-circuit RawEndpoints applies for invalid JSON —
                # so the client sees its actual mistake (400) rather than an
                # unrelated 404/503.
                started = perf_counter()
                invalid = _parse_worker_binary(body, headers)
                metrics.excluded_duration = perf_counter() - started
                metrics.enabled = not isinstance(invalid, RawReply)
                if isinstance(invalid, RawReply):
                    return invalid
                raise
            if (
                isinstance(model, ParallelModel)
                and not model.settings.max_batch_size
                and not model.settings.max_batch_time
            ):
                message = ModelRequestMessage(
                    model_name=model.name,
                    model_version=model.version,
                    method_name=RAW_INFER_METHOD,
                    method_args=[body, headers],
                )
                try:
                    response = await model._dispatcher.dispatch_request(message)
                    reply = response.return_value
                    if (
                        not isinstance(reply, RawReply)
                        or not isinstance(reply.body, bytes)
                        or reply.worker_pid == os.getpid()
                    ):
                        raise TypeError(
                            "Raw REST inference did not return a worker reply"
                        )
                except (Exception, asyncio.CancelledError) as error:
                    started = perf_counter()
                    invalid = _parse_worker_binary(body, headers)
                    metrics.enabled = not isinstance(invalid, RawReply)
                    metrics.excluded_duration = (
                        perf_counter()
                        - started
                        + getattr(error, _CONVERSION_DURATION_ATTR, 0.0)
                    )
                    # A failure formatting an already-completed prediction is
                    # outside the standard inference success/failure boundary.
                    metrics.inference_succeeded = getattr(
                        error, _INFERENCE_SUCCEEDED_ATTR, False
                    )
                    raise
                # Binary parse errors surface as 400 replies; JSON validation
                # errors as 422. Neither belongs to inference accounting.
                metrics.enabled = reply.status not in (400, 422)
                metrics.excluded_duration = reply.conversion_duration
                return reply
            # The raw worker path does not apply: keep today's local behavior.
            metrics.enabled = False
            return None

    def _respond(
        self,
        inference_response: InferenceResponse,
        request: InferenceRequest,
    ) -> FastAPIResponse:
        """Encode the response per the client's output negotiation.

        Binary is used when any output resolves to binary (or the
        request-wide default is set); otherwise the standard JSON
        encoding applies.  Model/CloudEvents headers extracted from the
        response parameters are preserved on both formats — minus the
        transport-owned framing headers, which the response itself
        controls (a model-supplied ``content-type`` must not relabel a
        binary frame, and an echoed ``inference-header-content-length``
        must not fork into two values).
        """
        default_binary, per_output = resolve_binary_outputs(request)
        # Per-output resolution already folds the request-wide default
        # into each entry — when outputs were requested explicitly,
        # per_output alone determines the format (an all-JSON
        # negotiation returns plain JSON even under a True default).
        # The request-wide default only applies when no outputs were
        # requested.
        if per_output is not None:
            wants_binary = any(per_output.values())
        else:
            wants_binary = default_binary

        # extract_headers also clears the internal parameters.headers
        # representation so it does not leak into the encoded response.
        extracted = extract_headers(inference_response) or {}
        response_headers = {
            key: value
            for key, value in extracted.items()
            if str(key).lower() not in _TRANSPORT_OWNED_HEADERS
        }

        if wants_binary:
            body, header_length = encode_binary_response(
                inference_response, default_binary, per_output
            )
            response_headers["Inference-Header-Content-Length"] = str(header_length)
            return FastAPIResponse(
                content=body,
                status_code=200,
                media_type="application/octet-stream",
                headers=response_headers,
            )

        encoded = encode_response(inference_response)
        return FastAPIResponse(
            content=encoded,
            status_code=200,
            media_type="application/json",
            headers=response_headers,
        )

    def _raw_reply_response(self, reply: RawReply) -> FastAPIResponse:
        """Convert a ``RawReply`` validation error to a HTTP response."""
        return FastAPIResponse(
            content=reply.body,
            status_code=reply.status,
            media_type="application/json",
            headers=reply.headers,
        )
