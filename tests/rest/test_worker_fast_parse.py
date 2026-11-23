"""Interaction tests: ``rest_worker_json`` + ``rest_fast_parse``.

The native span parser only sees raw request bytes on the worker JSON
path, so these tests exercise the full interaction at the dispatch
boundary:

- reply byte-identity (status + body) with the flag on and off, for
  every fast datatype and for every fallback class,
- the fast path being genuinely taken (ndarray data root reaching the
  model, not a silent fallback to the standard list),
- exact 422 error semantics preserved through fallback,
- the flag staying inert when the native library is absent,
- the settings fail-fast (``rest_fast_parse`` needs
  ``rest_worker_json``).
"""

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from prometheus_client import CollectorRegistry
from pytest_mock import MockerFixture
from starlette.requests import Request

from mlserver import ModelSettings, Settings
from mlserver.codecs import fastspan
from mlserver.handlers import DataPlane
from mlserver.model import MLModel
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.worker import Worker
from mlserver.registry import MultiModelRegistry
from mlserver.rest import worker as raw
from mlserver.types import InferenceRequest, InferenceResponse


# Tests that assert the fast path is genuinely taken (ndarray roots)
# require the compiled parser; the reply-identity tests run either way
# (without the library the fast path falls back and identity holds).
requires_native = pytest.mark.skipif(
    fastspan._load() is None,
    reason="libfastspan.so not compiled (gcc unavailable)",
)


@pytest.fixture
def settings(settings: Settings) -> Settings:
    # These tests exercise the dispatch boundary without spawning subprocesses.
    settings.parallel_workers = 0
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    # Adaptive batching's restore_list_inputs converts ndarray roots back
    # to lists before predict; disabling it lets these tests observe the
    # buffer the fast parser produced.
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


@pytest.fixture
def worker_data_plane(
    settings: Settings,
    model_registry: MultiModelRegistry,
    prometheus_registry: CollectorRegistry,
) -> DataPlane:
    # Built before the model loads (fixture order), so its Prometheus
    # collectors register on a clean registry; pre-seeded onto the
    # worker below so the lazy factory in process_raw_request never
    # constructs a duplicate.
    return DataPlane(settings, model_registry, inference_metrics=False)


def request(body: bytes, headers: dict[str, str] | None = None) -> Request:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/infer",
        "headers": [
            (key.encode(), value.encode()) for key, value in (headers or {}).items()
        ],
    }
    return Request(scope, receive)


def message(body: bytes, model: MLModel) -> ModelRequestMessage:
    return ModelRequestMessage(
        id="dispatch-id",
        model_name=model.name,
        model_version=model.version,
        method_name=raw.RAW_INFER_METHOD,
        method_args=[body, {"content-type": "application/json", "x-test": "echo-me"}],
    )


def body(datatype: str, data: str, shape: str = "[1, 3]") -> bytes:
    return (
        f'{{"id":"test-id","inputs":[{{"name":"input-0","shape":{shape},'
        f'"datatype":"{datatype}","data":{data}}}]}}'
    ).encode()


FAST_PAYLOADS = {
    "FP32": body("FP32", "[1.5, 2.5, -3.0]"),
    "FP64": body("FP64", "[0.1, 0.2, 1e-8]"),
    "BOOL": body("BOOL", "[true, false, true]"),
    "INT8": body("INT8", "[1, -2, 3]"),
    "INT16": body("INT16", "[1000, -2000, 3000]"),
    "INT32": body("INT32", "[1, 2, 3]"),
    "INT64": body("INT64", "[9223372036854775807, -1, 1]"),
    "UINT8": body("UINT8", "[0, 128, 255]"),
    "UINT16": body("UINT16", "[0, 32768, 65535]"),
    "UINT32": body("UINT32", "[0, 2147483648, 4294967295]"),
    "UINT64": body("UINT64", "[18446744073709551615, 0, 1]"),
}

# Payloads whose fast-path outcome must be "reject and fall back": the
# standard path decides, so the reply must stay byte-identical. (The
# string-data and count-mismatch fallback classes are verified at the
# parse level in tests/codecs/test_fastspan.py — here every case must
# also be servable by the sum model.)
FALLBACK_PAYLOADS = {
    "grammar-trailing-comma": body("FP32", "[1.5, 2.5, 3.0,]"),
    "grammar-leading-zero": body("FP32", "[01.5, 2.5, 3.0]"),
    "grammar-nan": body("FP32", "[nan, 1.0, 2.0]"),
    "int32-overflow-wraps": body("INT32", "[2147483648, 1, 1]"),
    # Negative in an unsigned type: the fast parser must reject and
    # fall back — the standard path then does whatever the numpy
    # version does (wrap on 1.x, OverflowError on 2.x); either way the
    # reply must be identical to the standard path's.
    "uint8-negative-fallback": body("UINT8", "[-5, 1, 1]"),
    "malformed-json": b"{",
    "empty-data": body("FP32", "[]", shape="[1, 0]"),
    "invalid-datatype": body("FP16", "[1.5, 2.5, 3.0]"),
}


async def _worker_reply(
    settings: Settings,
    model_registry: MultiModelRegistry,
    data_plane: DataPlane,
    payload: bytes,
    model: MLModel,
) -> tuple[int, bytes]:
    worker = cast(
        Worker,
        SimpleNamespace(
            _settings=settings,
            _model_registry=model_registry,
            _rest_raw_data_plane=data_plane,
        ),
    )
    response = await raw.process_raw_request(worker, message(payload, model))
    assert response.exception is None
    reply = response.return_value
    assert isinstance(reply, raw.RawReply)
    return reply.status, reply.body


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


def test_fast_parse_requires_worker_json():
    with pytest.raises(ValueError, match="rest_fast_parse requires rest_worker_json"):
        Settings(rest_fast_parse=True)


@requires_native
def test_fast_parse_valid_with_worker_json():
    settings = Settings(rest_fast_parse=True, rest_worker_json=True)
    assert settings.rest_fast_parse
    assert settings.rest_worker_json


def test_fast_parse_fails_loudly_without_library(mocker: MockerFixture):
    """Enabling the flag without the compiled parser must be a startup
    error — the deployment asked for the fast path and would otherwise
    silently not get it."""
    from mlserver.codecs import fastspan as fastspan_module

    mocker.patch.object(fastspan_module, "_load", lambda: None)
    with pytest.raises(ValueError, match="native parser.*not available"):
        Settings(rest_fast_parse=True, rest_worker_json=True)


# ---------------------------------------------------------------------------
# Worker dispatch: reply byte-identity across every fast datatype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("datatype", sorted(FAST_PAYLOADS))
async def test_worker_reply_identical_for_every_fast_datatype(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    datatype: str,
):
    payload = FAST_PAYLOADS[datatype]
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    fast_reply = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    std_reply = await _worker_reply(
        std, model_registry, worker_data_plane, payload, sum_model
    )
    assert fast_reply == std_reply
    assert fast_reply[0] == 200


# ---------------------------------------------------------------------------
# Worker dispatch: fallback classes keep exact standard-path replies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(FALLBACK_PAYLOADS), ids=lambda case: f"{case}")
async def test_worker_fallback_replies_identical(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    case: str,
):
    payload = FALLBACK_PAYLOADS[case]
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    fast_reply = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    std_reply = await _worker_reply(
        std, model_registry, worker_data_plane, payload, sum_model
    )
    assert fast_reply == std_reply, f"{case}: fast={fast_reply!r} vs std={std_reply!r}"


async def test_worker_validation_error_matches_standard_422(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
):
    """A grammar rejection must produce the standard path's 422 body —
    not a fast-path-specific error."""
    payload = FALLBACK_PAYLOADS["grammar-trailing-comma"]
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    status, reply_body = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    assert status == 422
    errors = json.loads(reply_body)
    # FastAPI-style error locations, produced by the standard parse.
    assert errors["detail"][0]["loc"][0] == "body"


# ---------------------------------------------------------------------------
# The fast path is genuinely taken
# ---------------------------------------------------------------------------


@requires_native
async def test_fast_path_reaches_model_as_ndarray(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    """When eligible, the parsed tensor must reach the model as an
    ndarray root (the native parser's buffer) — proving the fast path
    ran instead of a silent fallback to the standard list."""
    captured: list[InferenceRequest] = []
    original_predict = sum_model.predict

    async def spy(payload: InferenceRequest) -> InferenceResponse:
        captured.append(payload)
        return await original_predict(payload)

    mocker.patch.object(sum_model, "predict", side_effect=spy)
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    status, _ = await _worker_reply(
        fast, model_registry, worker_data_plane, FAST_PAYLOADS["FP32"], sum_model
    )
    assert status == 200
    assert len(captured) == 1
    root = captured[0].inputs[0].data.root
    assert isinstance(root, np.ndarray), f"fast path not taken, root={root!r}"
    assert root.dtype == np.float32
    np.testing.assert_array_equal(root, [1.5, 2.5, -3.0])


async def test_standard_path_reaches_model_as_list(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    """Control: the same request without the flag produces the standard
    list root (the contrast that proves the test above)."""
    captured: list[InferenceRequest] = []
    original_predict = sum_model.predict

    async def spy(payload: InferenceRequest) -> InferenceResponse:
        captured.append(payload)
        return await original_predict(payload)

    mocker.patch.object(sum_model, "predict", side_effect=spy)
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    await _worker_reply(
        std, model_registry, worker_data_plane, FAST_PAYLOADS["FP32"], sum_model
    )
    root = captured[0].inputs[0].data.root
    assert isinstance(root, list)


# ---------------------------------------------------------------------------
# Missing native library: the flag is inert
# ---------------------------------------------------------------------------


async def test_missing_library_leaves_flag_inert(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    """Without the compiled parser every accessor returns None: the
    flag must be a no-op and the standard path must serve the request
    (reply identical, list root reaching the model)."""
    from mlserver.codecs import fastspan

    mocker.patch.object(fastspan, "parse_span_f32", lambda *args: None)
    captured: list[InferenceRequest] = []
    original_predict = sum_model.predict

    async def spy(payload: InferenceRequest) -> InferenceResponse:
        captured.append(payload)
        return await original_predict(payload)

    mocker.patch.object(sum_model, "predict", side_effect=spy)
    inert = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    payload = FAST_PAYLOADS["FP32"]
    inert_reply = await _worker_reply(
        inert, model_registry, worker_data_plane, payload, sum_model
    )
    std_reply = await _worker_reply(
        std, model_registry, worker_data_plane, payload, sum_model
    )
    assert inert_reply == std_reply
    assert isinstance(captured[0].inputs[0].data.root, list)


# ---------------------------------------------------------------------------
# Zero-copy safety: in-place mutation cannot leak across requests
# ---------------------------------------------------------------------------


@requires_native
async def test_inplace_mutation_cannot_corrupt_subsequent_requests(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    """Pin the zero-copy contract end-to-end: the model receives a view
    of the parser's buffer, so a model that mutates its decoded input
    in place must not corrupt any OTHER request — the silent
    cross-request corruption class.

    The mutation is additive (``+= 1000``) so a reused or shared parse
    buffer would compound across requests and be caught: request two
    would start from request one's mutated values (sum 5001.0 instead
    of 3001.0).

    Equivalence-class boundary (documented in ``_to_ndarray``): within
    ONE request, repeated ``decode`` calls return views of the same
    buffer, so the mutation is visible to later decodes — where list
    roots hand out fresh copies. A model that decodes an input,
    mutates it in place and decodes it again therefore sees its own
    mutation with ``rest_fast_parse`` on; that pattern is outside the
    request-consumed-once contract.
    """
    original_predict = sum_model.predict

    async def mutating_predict(payload: InferenceRequest) -> InferenceResponse:
        sum_model.decode(payload.inputs[0])[:] += 1000.0
        return await original_predict(payload)

    mocker.patch.object(sum_model, "predict", side_effect=mutating_predict)
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    payload = FAST_PAYLOADS["FP32"]

    # Two consecutive requests through the mutating model: each must
    # see fresh input values — no cross-request buffer reuse.
    reply_a = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    reply_b = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    assert reply_a == reply_b
    assert reply_a[0] == 200
    # [1.5, 2.5, -3.0] + 1000 each -> sum 3001.0: the mutation is
    # visible in the model's own output (as it must be), and request
    # two produced the identical reply — its input was NOT request
    # one's mutated buffer (which would sum to 5001.0).
    assert json.loads(reply_a[1])["outputs"][0]["data"] == [3001.0]


# ---------------------------------------------------------------------------
# Nonparallel frontend fallback path (RawEndpoints with a local model)
# ---------------------------------------------------------------------------


async def test_nonparallel_model_serves_fast_parse(
    data_plane: DataPlane,
    sum_model: MLModel,
    settings: Settings,
):
    """A nonparallel model parses locally (no dispatch): the flag must
    apply there too, with byte-identical output."""
    settings.rest_worker_json = True
    data_plane._settings.rest_worker_json = True

    payload = FAST_PAYLOADS["INT32"]
    data_plane._settings.rest_fast_parse = False
    std_response = await raw.RawEndpoints(data_plane).infer(
        request(payload), sum_model.name, sum_model.version
    )
    data_plane._settings.rest_fast_parse = True
    fast_response = await raw.RawEndpoints(data_plane).infer(
        request(payload), sum_model.name, sum_model.version
    )
    assert fast_response.status_code == 200
    assert fast_response.body == std_response.body
    assert json.loads(fast_response.body)["outputs"][0]["data"] == [6]


# ---------------------------------------------------------------------------
# Mixed conditions on the dispatch path
# ---------------------------------------------------------------------------


async def test_mixed_datatype_inputs_serve_identically(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
):
    payload = (
        b'{"id":"test-id","inputs":['
        b'{"name":"input-0","shape":[1, 2],"datatype":"FP32","data":[1.5, 2.5]},'
        b'{"name":"b","shape":[1, 2],"datatype":"INT32","data":[10, 20]},'
        b'{"name":"c","shape":[1, 2],"datatype":"BOOL","data":[true, false]}'
        b"]}"
    )
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    fast_reply = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    std_reply = await _worker_reply(
        std, model_registry, worker_data_plane, payload, sum_model
    )
    assert fast_reply == std_reply


async def test_high_precision_fp64_serves_identically(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
):
    """≥17-digit mantissas exercise the tiered arithmetic path."""
    payload = body(
        "FP64",
        "[0.49362120032310486, 0.2881241738796234, 1.7976931348623157e308]",
    )
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    std = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": False}
    )
    fast_reply = await _worker_reply(
        fast, model_registry, worker_data_plane, payload, sum_model
    )
    std_reply = await _worker_reply(
        std, model_registry, worker_data_plane, payload, sum_model
    )
    assert fast_reply == std_reply


async def test_repeated_requests_stay_identical(
    settings: Settings,
    model_registry: MultiModelRegistry,
    worker_data_plane: DataPlane,
    sum_model: MLModel,
):
    """Library state (the cached CDLL handle) must not drift across
    requests: the same payload parsed repeatedly stays byte-identical."""
    fast = settings.model_copy(
        update={"rest_worker_json": True, "rest_fast_parse": True}
    )
    payload = FAST_PAYLOADS["FP32"]
    replies = [
        await _worker_reply(fast, model_registry, worker_data_plane, payload, sum_model)
        for _ in range(5)
    ]
    assert all(reply == replies[0] for reply in replies)
