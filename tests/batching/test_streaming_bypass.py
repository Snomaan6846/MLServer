"""Streaming requests must bypass adaptive batching.

A model implementing only ``predict`` inherits the single-shot
``predict_stream`` fallback. The batching machinery deliberately
bypasses the adaptive batcher for that fallback (a stream is a plain
infer semantically) — both on the direct server (the streaming
wrapper's task-local mark) and across worker dispatch (the
``skip_batching`` message flag). Ordinary ``/infer`` must keep
batching.
"""

import pathlib

from collections.abc import AsyncGenerator

import pytest
from httpx import AsyncClient

from mlserver import ModelSettings, Settings
from mlserver.batching import load_batching, unload_batching
from mlserver.batching.adaptive import AdaptiveBatcher
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.registry import MultiModelRegistry
from mlserver.rest import RESTServer


def _batched_settings() -> ModelSettings:
    return ModelSettings(
        name="batched-echo",
        implementation="tests.fixtures.SumModel",
        max_batch_size=8,
        max_batch_time=0.05,
        inputs=[
            {
                "name": "input-0",
                "datatype": "FP32",
                "shape": [1, 2],
                "parameters": {"content_type": "np"},
            }
        ],
    )


def _request_body() -> dict:
    return {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, 2],
                "datatype": "FP32",
                "data": [1.0, 2.0],
            }
        ]
    }


@pytest.fixture
def batcher_spy(monkeypatch) -> dict:
    """Count AdaptiveBatcher.predict invocations in this process."""
    calls = {"count": 0}
    original = AdaptiveBatcher.predict

    async def counting(self, payload):
        calls["count"] += 1
        return await original(self, payload)

    monkeypatch.setattr(AdaptiveBatcher, "predict", counting)
    return calls


@pytest.fixture
async def batched_rest_client(
    settings: Settings,
    inference_pool_registry,
    prometheus_registry,
    batcher_spy: dict,
) -> AsyncGenerator[AsyncClient, None]:
    """A REST app whose registry holds ONE batched predict-only model
    (the direct-server shape: batching installed in this process)."""
    registry = MultiModelRegistry(
        on_model_load=[inference_pool_registry.load_model, load_batching],
        on_model_unload=[unload_batching, inference_pool_registry.unload_model],
        model_initialiser=inference_pool_registry.model_initialiser,
    )
    await registry.load(_batched_settings())
    try:
        data_plane = DataPlane(settings=settings, model_registry=registry)
        data_plane.startup_complete()
        from mlserver.repository import SchemalessModelRepository

        repo_handlers = ModelRepositoryHandlers(
            repository=SchemalessModelRepository(str(pathlib.Path(__file__).parent)),
            model_registry=registry,
        )
        server = RESTServer(
            settings,
            data_plane=data_plane,
            model_repository_handlers=repo_handlers,
        )
        async with AsyncClient(app=server._app, base_url="http://test") as ac:
            yield ac
    finally:
        try:
            await registry.unload("batched-echo")
        except Exception:
            pass


async def test_infer_stream_never_invokes_batcher(
    batched_rest_client: AsyncClient, batcher_spy: dict
):
    """The single-shot stream fallback must bypass the batcher.

    Regression: the endpoint shortcut called data_plane.infer()
    directly, skipping the streaming wrapper's task-local bypass mark
    — the stream request joined a batch and waited for the window.
    """
    response = await batched_rest_client.post(
        "/v2/models/batched-echo/infer_stream", json=_request_body()
    )
    assert response.status_code == 200
    assert b"data: " in response.content
    assert (
        batcher_spy["count"] == 0
    ), f"streaming entered the batcher ({batcher_spy['count']} calls)"


async def test_infer_still_batches(batched_rest_client: AsyncClient, batcher_spy: dict):
    """Ordinary /infer must keep using the adaptive batcher."""
    response = await batched_rest_client.post(
        "/v2/models/batched-echo/infer", json=_request_body()
    )
    assert response.status_code == 200
    assert batcher_spy["count"] >= 1, "ordinary infer stopped batching"


async def test_mixed_stream_and_infer_no_bypass_leak(
    batched_rest_client: AsyncClient, batcher_spy: dict
):
    """Alternating stream/infer requests: every stream bypasses, every
    infer batches — the task-local mark never leaks between requests."""
    for _ in range(3):
        before_stream = batcher_spy["count"]
        stream = await batched_rest_client.post(
            "/v2/models/batched-echo/infer_stream", json=_request_body()
        )
        assert stream.status_code == 200
        assert batcher_spy["count"] == before_stream, "stream request batched"

        before_infer = batcher_spy["count"]
        infer = await batched_rest_client.post(
            "/v2/models/batched-echo/infer", json=_request_body()
        )
        assert infer.status_code == 200
        assert batcher_spy["count"] > before_infer, "infer request bypassed"
