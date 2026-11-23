"""End-to-end inference-timeout test: the full REST stack.

A slow model behind a parallel worker, a short ``inference_timeout``:
the request must fail with HTTP 504 and the "Inference request timed
out" body at the deadline — through the real endpoint, dispatcher
wait, exception mapping and handler, not the handler-level unit test.
"""

import aiohttp
import pytest

from mlserver import ModelSettings, Settings


@pytest.fixture
def settings(settings: Settings, tmp_path) -> Settings:
    # The dispatch must go through the parallel pool (the inference
    # deadline is enforced by the dispatcher's reply wait), and the
    # deadline must be far below the model's 2 s sleep. The metrics
    # dir switches the pool's counters to multiprocess mode (the
    # in-process REGISTRY would otherwise collide with the server's
    # DataPlane counters).
    settings.parallel_workers = 1
    settings.parallel_inference_transport = "zmq"
    settings.inference_timeout = 0.5
    settings.metrics_dir = str(tmp_path / "metrics")
    settings.metrics_endpoint = "/metrics"
    return settings


@pytest.fixture
def slow_model_settings() -> ModelSettings:
    return ModelSettings(
        name="slow-model",
        implementation="tests.fixtures.DeadlineModel",
    )


async def test_inference_timeout_returns_504_through_full_stack(
    mlserver, slow_model_settings: ModelSettings, settings: Settings
):
    await mlserver._model_registry.load(slow_model_settings)
    url = f"http://{settings.host}:{settings.http_port}/v2/models/slow-model/infer"
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [3],
                "datatype": "FP32",
                "data": [1.0, 2.0, 3.0],
            }
        ]
    }
    # RESTClient raises on non-200; use a plain session so the status
    # AND the body can both be asserted.
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            body = await response.text()
            assert response.status == 504, f"status {response.status}, body: {body!r}"

    assert "Inference request timed out" in body
