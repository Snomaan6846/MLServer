from collections.abc import Callable

from fastapi import status
from fastapi.routing import APIRoute
from numpy import ndarray
from starlette.types import Scope

from ..codecs.json import json_mode_convert
from ..handlers.custom import CustomHandler
from ..types import InferenceResponse


def encode_inference_response_bytes(response: InferenceResponse) -> bytes:
    """Render an inference response to JSON bytes (the REST wire)."""
    # Python-mode dump: ndarray roots flow to orjson's
    # OPT_SERIALIZE_NUMPY (the compact-floats opt-in applies); the
    # METADATA (parameters) keeps the JSON-mode scalar conversion
    # (Decimal -> str, timedelta -> ISO duration) the
    # response-model path performed. Used by BOTH the frontend
    # endpoint and the worker's JSON rendering — one serializer,
    # identical bytes.
    # Late-bound: tests patch the codec module's encoder in place
    # to observe the serving path's inputs.
    from ..codecs.json import encode_to_json_bytes

    # Skeleton dump: the outputs' tensor data is excluded — pydantic's
    # walk over a list root is pure identity for plain scalars, and
    # the data needs exactly ONE pass (below), not a walk plus a
    # pass. Each root then re-attaches as the LAST key of its output
    # (the schema's declaration order): ndarray roots go through the
    # TensorData serializer (the native buffer path, identical to the
    # full dump's treatment); every other root takes the schema's
    # JSON-mode conversion — orjson renders some
    # natively-serialisable types DIFFERENTLY than the schema's JSON
    # mode (a UTC datetime becomes +00:00 instead of Z) with no
    # exception to trip the fallback below.
    dumped = response.model_dump(
        exclude_unset=True,
        exclude={"outputs": {"__all__": {"data"}}},
    )
    outputs = dumped.get("outputs") or []
    for output, output_model in zip(outputs, response.outputs or []):
        root = output_model.data.root
        if isinstance(root, ndarray):
            # The SERIALIZED value decides eligibility, not the
            # original root: an object-dtype array (or a masked one)
            # serializes to a LIST whose exotic elements need the
            # JSON-mode conversion — orjson renders a UTC datetime as
            # +00:00 where the schema's JSON mode renders Z. Native
            # dtypes keep the ndarray fast path (the buffer render).
            serialized = output_model.data.model_dump(exclude_unset=True)
            output["data"] = (
                serialized
                if isinstance(serialized, ndarray)
                else json_mode_convert(serialized)
            )
        else:
            output["data"] = json_mode_convert(root)
    if isinstance(dumped.get("parameters"), dict):
        dumped["parameters"] = json_mode_convert(dumped["parameters"])
    for output in outputs:
        if isinstance(output.get("parameters"), dict):
            output["parameters"] = json_mode_convert(output["parameters"])
    try:
        return encode_to_json_bytes(dumped)
    except Exception:
        # Not eligible for native tensor encoding: the schema's full
        # JSON-mode conversion (the adapter the response-model path
        # used — same bytes, tolist'd data).
        return encode_to_json_bytes(
            response.model_dump(mode="json", exclude_unset=True)
        )


def to_status_code(flag: bool, error_code: int = status.HTTP_400_BAD_REQUEST) -> int:
    """
    Convert a boolean flag into a HTTP status code.
    """
    if flag:
        return status.HTTP_200_OK

    return error_code


def matches(
    route: APIRoute, custom_handler: CustomHandler, handler_method: Callable
) -> bool:
    if route.endpoint != handler_method:
        return False

    scope = to_scope(custom_handler)
    match, _ = route.matches(scope)
    if match == match.NONE:
        return False

    return True


def to_scope(custom_handler: CustomHandler) -> Scope:
    return {
        "type": "http",
        "method": custom_handler.rest_method,
        "path": custom_handler.rest_path,
    }
