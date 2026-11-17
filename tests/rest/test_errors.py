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


async def test_timeout_error_maps_to_504():
    """An expired inference deadline (or any await timeout while
    serving) renders as 504 Gateway Timeout, not a generic 500 —
    clients and retries must be able to tell a timing failure from a
    server fault apart."""
    from mlserver.rest.errors import handle_timeout_error

    response = await handle_timeout_error(Mock(), TimeoutError())

    assert response.status_code == 504
    body = json.loads(response.body)
    assert body["error"] == "Inference request timed out"


def test_timeout_error_registered_on_app():
    """The 504 handler must be wired into the app's exception handlers:
    a handler that exists but is not registered still surfaces as a
    generic 500."""
    from mlserver.rest.errors import _EXCEPTION_HANDLERS

    assert TimeoutError in _EXCEPTION_HANDLERS
