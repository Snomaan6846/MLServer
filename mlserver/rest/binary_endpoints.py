"""REST inference endpoints with KServe v2 binary tensor support.

When ``rest_binary_tensor`` is enabled, requests carrying the
``Inference-Header-Content-Length`` header are parsed as binary tensor
data (JSON header + raw bytes).  All other requests are parsed as
standard JSON.  Both paths use the existing DataPlane for inference.
"""

from __future__ import annotations

import orjson
from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import Response as FastAPIResponse

from ..types import InferenceResponse
from ..handlers import DataPlane
from ..utils import insert_headers, extract_headers

from .binary import (
    BinaryTensorError,
    client_wants_binary,
    encode_binary_response,
    get_header_length,
    is_binary_request,
    parse_binary_request,
)
from .worker import parse_body, encode_response, RawReply


class BinaryTensorEndpoints:
    """Handles both binary tensor and standard JSON inference requests.

    Follows the same pattern as ``RawEndpoints`` (rest_worker_json):
    takes a raw ``Request`` and handles body parsing for both binary
    and JSON formats, bypassing FastAPI's automatic body parsing.
    Reuses ``parse_body`` and ``encode_response`` from the worker
    module for the JSON fallback path.
    """

    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane

    async def infer(
        self,
        request: Request,
        model_name: str,
        model_version: str | None = None,
    ) -> FastAPIResponse:
        headers = dict(request.headers)

        if is_binary_request(headers):
            return await self._infer_binary(request, headers, model_name, model_version)

        return await self._infer_json(request, model_name, model_version)

    async def _infer_json(
        self,
        request: Request,
        model_name: str,
        model_version: str | None = None,
    ) -> FastAPIResponse:
        """Standard JSON path — parse, infer, respond as JSON.

        Reuses ``parse_body`` from the worker module so error locations
        and validation semantics match the rest of the codebase.
        """
        body = await request.body()
        payload = parse_body(body)
        if isinstance(payload, RawReply):
            return self._raw_reply_response(payload)

        insert_headers(payload, dict(request.headers))
        with self._data_plane._infer_metrics(model_name, model_version):
            inference_response = await self._data_plane.infer(
                payload, model_name, model_version
            )

        return self._json_response(inference_response, dict(request.headers))

    async def _infer_binary(
        self,
        request: Request,
        headers: dict[str, str],
        model_name: str,
        model_version: str | None = None,
    ) -> FastAPIResponse:
        """Binary tensor path — parse binary, infer, respond as binary."""
        try:
            header_length = get_header_length(headers)
            body = await request.body()
            payload = parse_binary_request(body, header_length)
        except BinaryTensorError as error:
            raise HTTPException(status_code=400, detail=error.detail)

        insert_headers(payload, dict(request.headers))
        with self._data_plane._infer_metrics(model_name, model_version):
            inference_response = await self._data_plane.infer(
                payload, model_name, model_version
            )

        # Determine output format from the original request header
        header_dict = orjson.loads(body[:header_length])
        if client_wants_binary(header_dict):
            return self._binary_response(inference_response)

        # Client did not request binary output — respond as JSON
        return self._json_response(inference_response, dict(request.headers))

    def _binary_response(
        self, inference_response: InferenceResponse
    ) -> FastAPIResponse:
        """Encode the response as binary tensor data."""
        body, header_length = encode_binary_response(inference_response)
        return FastAPIResponse(
            content=body,
            status_code=200,
            media_type="application/octet-stream",
            headers={"Inference-Header-Content-Length": str(header_length)},
        )

    def _json_response(
        self,
        inference_response: InferenceResponse,
        request_headers: dict[str, str],
    ) -> FastAPIResponse:
        """Encode the response as JSON, reusing the worker encoder."""
        response_headers = extract_headers(inference_response) or {}
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
