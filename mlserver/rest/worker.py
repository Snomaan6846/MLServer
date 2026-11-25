"""Opt-in REST JSON processing in the selected model's inference worker.

The frontend forwards the original JSON bytes, keeping the large Python input
object graph inside one worker. Parsing, validation and response encoding retain
the normal REST algorithms; inference still passes through the DataPlane.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from time import perf_counter
from typing import Any, TYPE_CHECKING

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.responses import Response

from ..codecs.json import decode_from_bytelike_json_to_dict, encode_to_json_bytes
from .utils import encode_inference_response_bytes
from ..errors import InferenceTimeoutError, ModelNotReady
from ..handlers import DataPlane
from ..parallel.logging import logger
from ..types import InferenceRequest, InferenceResponse
from ..utils import extract_headers, insert_headers

from .binary import is_binary_request

if TYPE_CHECKING:
    from ..parallel.messages import ModelRequestMessage, ModelResponseMessage
    from ..parallel.worker import Worker

RAW_INFER_METHOD = "__mlserver_rest_worker_infer__"
_RESPONSE_ADAPTER: TypeAdapter[InferenceResponse] = TypeAdapter(InferenceResponse)
_CONVERSION_DURATION_ATTR = "_rest_conversion_duration"
_INFERENCE_SUCCEEDED_ATTR = "_rest_inference_succeeded"


@dataclass
class RawReply:
    body: bytes
    headers: dict[str, str]
    status: int = 200
    worker_pid: int = field(default_factory=os.getpid)
    conversion_duration: float = 0.0


def _http_response(reply: RawReply) -> Response:
    return Response(
        reply.body,
        status_code=reply.status,
        headers=reply.headers,
        media_type="application/json",
    )


def _validation_reply(errors: list[dict[str, Any]]) -> RawReply:
    return RawReply(
        encode_to_json_bytes({"detail": jsonable_encoder(errors)}), {}, status=422
    )


def _missing_body() -> RawReply:
    return _validation_reply(
        [{"type": "missing", "loc": ("body",), "msg": "Field required", "input": None}]
    )


def _json_content(headers: dict[str, str]) -> bool:
    content_type = headers.get("content-type")
    if content_type is None:
        return True
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


class _FastPathUnavailable(Exception):
    """Internal signal: the orchestrator chose the standard path."""


def _reject_fast_path(_body: bytes) -> InferenceRequest:
    raise _FastPathUnavailable()


def _try_fast_parse(body: bytes) -> InferenceRequest | None:
    """Attempt the native span fast path; ``None`` means use the standard path.

    The orchestrator's strict-subset contract: any payload the native
    parser cannot reproduce exactly (unsupported datatype, grammar it
    rejects, envelope mismatch, native library absent) falls back to
    the standard parse — whose validation errors become the usual HTTP
    422 reply. The fast path never accepts what the standard path
    rejects, and never changes a value it does accept.
    """
    from ..codecs.fast_json import fastspan_parser, parse_inference_request

    try:
        return parse_inference_request(body, _reject_fast_path, fastspan_parser)
    except _FastPathUnavailable:
        return None


def parse_body(
    body: bytes,
    json_content: bool = True,
    fast_parse: bool = False,
) -> InferenceRequest | RawReply:
    """Use the existing JSON-object/schema path and FastAPI error locations."""
    if not body:
        return _missing_body()
    if fast_parse and json_content:
        fast = _try_fast_parse(body)
        if fast is not None:
            return fast
    decoded: Any = body
    if json_content:
        try:
            decoded = decode_from_bytelike_json_to_dict(body)
        except json.JSONDecodeError as error:
            return _validation_reply(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body", error.pos),
                        "msg": "JSON decode error",
                        "input": {},
                        "ctx": {"error": error.msg},
                    }
                ]
            )
    if decoded is None:
        # FastAPI treats JSON null as an absent required request body.
        return _missing_body()
    try:
        return InferenceRequest.model_validate(decoded, from_attributes=True)
    except ValidationError as error:
        errors: list[dict[str, Any]] = [
            dict(item) for item in error.errors(include_url=False)
        ]
        for item in errors:
            item["loc"] = ("body", *tuple(item["loc"]))
        return _validation_reply(errors)


def encode_response(response: InferenceResponse) -> bytes:
    """Apply the same validated schema and serialization flags as REST."""
    checked = _RESPONSE_ADAPTER.validate_python(response, from_attributes=True)
    # The SAME serializer as the frontend endpoint: python-mode dump
    # (ndarray roots to orjson's OPT_SERIALIZE_NUMPY — the
    # compact-floats opt-in applies) with the metadata's JSON-mode
    # scalar conversion. A JSON-mode dump here widened FP32 tensors
    # to Python floats before encoding, making the flag inert.
    return encode_inference_response_bytes(checked)


def _parse_worker_binary(
    body: bytes, headers: dict[str, str]
) -> InferenceRequest | RawReply:
    """Parse a binary tensor frame, mapping errors to a HTTP 400 reply.

    Mirrors the frontend endpoint's ``BinaryTensorError`` handling so both
    processing locations return the same status and body.
    """
    from .binary import BinaryTensorError, get_header_length, parse_binary_request

    try:
        payload, _ = parse_binary_request(body, get_header_length(headers))
    except BinaryTensorError as error:
        return RawReply(encode_to_json_bytes({"detail": error.detail}), {}, status=400)
    return payload


async def process_raw_request(
    worker: Worker, request: ModelRequestMessage
) -> ModelResponseMessage:
    """Handle the dedicated raw REST message inside its ordinary model worker."""
    # Import at the dispatch boundary: parallel startup imports REST app setup.
    from ..parallel.errors import WorkerError
    from ..parallel.messages import ModelResponseMessage

    conversion_duration = 0.0
    inference_succeeded = False
    try:
        if request.method_name != RAW_INFER_METHOD:
            raise ValueError("Unexpected method for raw REST inference")
        body, headers = request.method_args
        if not isinstance(body, bytes) or not isinstance(headers, dict):
            raise TypeError("Raw REST inference requires bytes and HTTP headers")
        # Binary frames are only honored when the extension is enabled: a
        # stray Inference-Header-Content-Length on a JSON request must not
        # bypass the rest_binary_tensor opt-in (a smuggled frame parses as
        # JSON and fails validation instead of producing a binary reply).
        binary_request = worker._settings.rest_binary_tensor and is_binary_request(
            headers
        )
        started = perf_counter()
        try:
            if binary_request:
                payload = _parse_worker_binary(body, headers)
            else:
                payload = parse_body(
                    body,
                    _json_content(headers),
                    fast_parse=worker._settings.rest_fast_parse,
                )
            if isinstance(payload, RawReply):
                return ModelResponseMessage(id=request.id, return_value=payload)
            insert_headers(payload, headers)
            data_plane = getattr(worker, "_rest_raw_data_plane", None)
            if data_plane is None:
                # Preserve middleware/context in the worker, while inference
                # accounting remains in the frontend across worker restarts.
                data_plane = DataPlane(
                    worker._settings, worker._model_registry, inference_metrics=False
                )
                setattr(worker, "_rest_raw_data_plane", data_plane)
        finally:
            conversion_duration += perf_counter() - started
        response = await data_plane.infer(
            payload, request.model_name, request.model_version
        )
        inference_succeeded = True
        started = perf_counter()
        try:
            if binary_request:
                # Import at the dispatch boundary: binary_endpoints imports
                # this module, so the dependency is deferred to call time.
                from .binary_endpoints import encode_worker_binary_reply

                reply = encode_worker_binary_reply(response, payload)
            else:
                # A JSON request may negotiate binary outputs (KServe
                # allows it): resolve from the PARSED request — exact,
                # including escaped key spellings the standard parse
                # decodes — and encode with the same shared functions
                # the frontend's local path uses.
                from .binary_endpoints import (
                    encode_worker_binary_reply,
                    wants_binary_reply,
                )

                if wants_binary_reply(payload):
                    reply = encode_worker_binary_reply(response, payload)
                else:
                    response_headers = extract_headers(response) or {}
                    encoded = encode_response(response)
                    reply = RawReply(encoded, response_headers)
        finally:
            conversion_duration += perf_counter() - started
        reply.conversion_duration = conversion_duration
        return ModelResponseMessage(id=request.id, return_value=reply)
    except (Exception, asyncio.CancelledError) as error:
        # Returning an error envelope bypasses Worker._process_request's usual
        # exception handler. Log here once while the original traceback exists.
        logger.exception(
            "An error occurred calling method 'predict' from model '%s'.",
            request.model_name,
        )
        failure = WorkerError(error)
        # Exception attributes survive Queue/ZeroMQ pickling. Error replies
        # must exclude worker JSON conversion from the public timer too.
        setattr(failure, _CONVERSION_DURATION_ATTR, conversion_duration)
        setattr(failure, _INFERENCE_SUCCEEDED_ATTR, inference_succeeded)
        return ModelResponseMessage(id=request.id, exception=failure)


class RawEndpoints:
    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane

    async def infer(
        self, request: Request, model_name: str, model_version: str | None = None
    ) -> Response:
        from ..parallel.messages import ModelRequestMessage
        from ..parallel.model import ParallelModel

        try:
            body = await request.body()
        except HTTPException:
            raise
        except Exception as error:
            # A Request-only endpoint bypasses FastAPI's body parser, including
            # its conversion of read/decompression errors into a JSON 400.
            raise HTTPException(
                status_code=400, detail="There was an error parsing the body"
            ) from error
        headers = dict(request.headers)
        # The content-type partition is request-static: computed once,
        # reused by every parse_body site below (three of the four are
        # error paths that deliberately re-parse; the partition is not
        # among the things they need to recompute).
        json_content = _json_content(headers)
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
                started = perf_counter()
                invalid = parse_body(
                    body,
                    json_content,
                    fast_parse=self._data_plane._settings.rest_fast_parse,
                )
                metrics.excluded_duration = perf_counter() - started
                metrics.enabled = not isinstance(invalid, RawReply)
                if isinstance(invalid, RawReply):
                    return _http_response(invalid)
                raise
            if (
                isinstance(model, ParallelModel)
                and not model.settings.max_batch_size
                and not model.settings.max_batch_time
                and json_content
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
                except InferenceTimeoutError as error:
                    # The dispatch wait expired. A parse failure in the
                    # worker produces a 422 reply (not a timeout), so a
                    # timeout means the request was valid and slow:
                    # count it without re-parsing the body — a 15 MB
                    # re-parse on the error path costs ~50 ms for a
                    # metrics flag we can already determine.
                    metrics.enabled = True
                    metrics.excluded_duration = getattr(
                        error, _CONVERSION_DURATION_ATTR, 0.0
                    )
                    metrics.inference_succeeded = getattr(
                        error, _INFERENCE_SUCCEEDED_ATTR, False
                    )
                    raise
                except (Exception, asyncio.CancelledError) as error:
                    started = perf_counter()
                    invalid = parse_body(
                        body,
                        json_content,
                        fast_parse=self._data_plane._settings.rest_fast_parse,
                    )
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
                metrics.enabled = reply.status != 422
                metrics.excluded_duration = reply.conversion_duration
            else:
                # The ordinary DataPlane owns accounting on every fallback.
                metrics.enabled = False
                payload = parse_body(
                    body,
                    json_content,
                    fast_parse=self._data_plane._settings.rest_fast_parse,
                )
                if isinstance(payload, RawReply):
                    reply = payload
                else:
                    insert_headers(payload, headers)
                    prediction = await self._data_plane.infer(
                        payload, model_name, model_version
                    )
                    # A JSON request may negotiate binary outputs even on
                    # the local fallback (nonparallel or batched models):
                    # resolve from the parsed request, exactly as the
                    # worker dispatch and the binary endpoint's local
                    # path do, so delegation never drops a negotiation.
                    from .binary_endpoints import (
                        encode_worker_binary_reply,
                        wants_binary_reply,
                    )

                    if wants_binary_reply(payload):
                        reply = encode_worker_binary_reply(prediction, payload)
                    else:
                        response_headers = extract_headers(prediction) or {}
                        reply = RawReply(encode_response(prediction), response_headers)
        return _http_response(reply)
