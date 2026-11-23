import json
from unittest.mock import Mock


async def test_validation_error_big_int_falls_back_to_stdlib_json():
    """A validation error echoing an integer above orjson's 64-bit
    range must still render as a clean 422: the orjson Response raises
    on such values, and the handler falls back to stdlib JSON instead
    of failing itself (a 500 for a client-triggerable error)."""
    from fastapi.exceptions import RequestValidationError

    from mlserver.rest.errors import handle_request_validation_error

    exc = RequestValidationError(
        [
            {
                "type": "string_type",
                "loc": ("body", "id"),
                "msg": "Input should be a valid string",
                "input": 10**30,
            }
        ]
    )
    response = await handle_request_validation_error(Mock(), exc)

    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["detail"][0]["input"] == 10**30


async def test_validation_error_unencodable_context_stays_422():
    """A pydantic error context carrying an object FastAPI's encoder
    cannot handle must not fail the handler — that would turn a
    client-triggerable validation error into a 500."""
    from fastapi.exceptions import RequestValidationError

    from mlserver.rest.errors import handle_request_validation_error

    class Unencodable:
        # No __dict__: jsonable_encoder's fallbacks (dict(), vars())
        # both fail and it raises ValueError.
        __slots__ = ()

    exc = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "inputs"),
                "msg": "bad value",
                "input": Unencodable(),
            }
        ]
    )
    response = await handle_request_validation_error(Mock(), exc)

    assert response.status_code == 422
    body = json.loads(response.body)
    assert isinstance(body["detail"], str) and body["detail"]


async def test_inference_timeout_maps_to_504():
    """An expired inference deadline renders as 504 Gateway Timeout
    with the same body the dedicated handler produced — clients and
    retries must be able to tell a timing failure from a server fault
    apart."""
    from mlserver.errors import InferenceTimeoutError
    from mlserver.rest.errors import handle_mlserver_error

    response = await handle_mlserver_error(Mock(), InferenceTimeoutError())

    assert response.status_code == 504
    body = json.loads(response.body)
    assert body["error"] == "Inference request timed out"


def test_foreign_timeout_keeps_generic_500():
    """A bare TimeoutError from any other await inside request handling
    (a custom handler's or model's own deadline) is NOT remapped: no
    handler is registered for it, so it keeps the generic 500 it has
    always had — only the dispatch wait's InferenceTimeoutError maps
    to 504."""
    from mlserver.errors import InferenceTimeoutError, MLServerError
    from mlserver.rest.errors import _EXCEPTION_HANDLERS

    assert TimeoutError not in _EXCEPTION_HANDLERS
    # InferenceTimeoutError is handled through the MLServerError
    # registration (it is a subclass with a 504 status code).
    assert issubclass(InferenceTimeoutError, MLServerError)
    assert MLServerError in _EXCEPTION_HANDLERS
