from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response as StarletteResponse

from .responses import Response
from ..errors import MLServerError


class APIErrorResponse(BaseModel):
    error: str | None = None


async def handle_mlserver_error(request: Request, exc: MLServerError) -> Response:
    err_res = APIErrorResponse(error=str(exc))
    return Response(status_code=exc.status_code, content=err_res.model_dump())


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> StarletteResponse:
    # Render body-validation errors with the same orjson Response as the
    # rest of the engine (FastAPI's default renders via stdlib json).
    try:
        detail = {"detail": jsonable_encoder(exc.errors())}
    except Exception:
        # The encoder itself can raise (a pydantic error context with a
        # type FastAPI has no encoder for): the handler must not fail,
        # or a client-triggerable validation error becomes a 500.
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    try:
        return Response(status_code=422, content=detail)
    except Exception:
        # orjson rejects values stdlib json accepts — pydantic's
        # validation errors echo the offending input, and an integer
        # above the 64-bit range makes the orjson render raise. The
        # fallback keeps the same body structure and status instead of
        # failing the handler itself (a 500 for a client-triggerable
        # validation error).
        return JSONResponse(status_code=422, content=detail)


_EXCEPTION_HANDLERS = {
    MLServerError: handle_mlserver_error,
    RequestValidationError: handle_request_validation_error,
}
# NOTE: only InferenceTimeoutError (an MLServerError subclass raised
# solely by the dispatch wait) maps to 504. A bare TimeoutError from
# any other await inside request handling keeps the generic 500 it has
# always had — the handler cannot know it was an inference deadline.
