from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from .responses import Response
from ..errors import MLServerError


class APIErrorResponse(BaseModel):
    error: str | None = None


async def handle_mlserver_error(request: Request, exc: MLServerError) -> Response:
    err_res = APIErrorResponse(error=str(exc))
    return Response(status_code=exc.status_code, content=err_res.model_dump())


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> Response:
    # Render body-validation errors with the same orjson Response as the
    # rest of the engine (FastAPI's default renders via stdlib json).
    return Response(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


_EXCEPTION_HANDLERS = {
    MLServerError: handle_mlserver_error,
    RequestValidationError: handle_request_validation_error,
}
