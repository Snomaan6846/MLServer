import asyncio

import pytest

from mlserver import MLServer
from mlserver.settings import Settings


async def test_server_start_is_single_flight(mocker, prometheus_registry):
    server = MLServer(Settings())
    transports_started = asyncio.Event()
    transports_released = asyncio.Event()
    started_count = 0
    transport_count = 2 + int(server._metrics_server is not None)

    async def transport():
        nonlocal started_count
        started_count += 1
        if started_count == transport_count:
            transports_started.set()
        await transports_released.wait()

    async def stop_transport(*args, **kwargs):
        transports_released.set()

    for transport_server in (
        server._rest_server,
        server._grpc_server,
        server._metrics_server,
    ):
        if transport_server is not None:
            mocker.patch.object(transport_server, "start", new=transport)
            mocker.patch.object(transport_server, "stop", new=stop_transport)

    startup_task = asyncio.create_task(server.start())
    await transports_started.wait()

    assert len(server._server_tasks) == transport_count
    assert not server._startup_model_tasks
    assert server._stop_task is None
    assert server._live

    await server.start()
    assert len(server._server_tasks) == transport_count
    assert not startup_task.done()

    await server.stop()
    await startup_task

    assert not server._server_tasks
    assert not server._startup_model_tasks
    assert server._stop_task is None
    assert not server._live


async def test_start_returns_while_shutdown_is_in_progress(mocker, prometheus_registry):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def stop_resources(_sig=None):
        cleanup_started.set()
        await release_cleanup.wait()

    mocker.patch.object(server, "_stop_resources", new=stop_resources)

    try:
        stop_task = asyncio.create_task(server.stop())
        await cleanup_started.wait()

        await server.start()
        assert not server._live
    finally:
        release_cleanup.set()
        await stop_task
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_stop_cancels_server_tasks_when_resource_cleanup_fails(
    mocker, prometheus_registry
):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    task_started = asyncio.Event()

    async def server_task_body():
        task_started.set()
        await asyncio.Future()

    server_task = asyncio.create_task(server_task_body())
    server._server_tasks.append(server_task)
    await task_started.wait()

    async def failing_stop_resources(_sig=None):
        raise RuntimeError("resource cleanup failed")

    mocker.patch.object(server, "_stop_resources", new=failing_stop_resources)

    try:
        await server.stop()

        assert server_task.cancelled()
        assert server._stop_task is None
    finally:
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_stop_cancels_startup_model_tasks(mocker, prometheus_registry):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    task_started = asyncio.Event()

    async def model_task_body():
        task_started.set()
        await asyncio.Future()

    model_task = asyncio.create_task(model_task_body())
    server._startup_model_tasks.append(model_task)
    await task_started.wait()
    mocker.patch.object(server, "_stop_resources", new=mocker.AsyncMock())

    try:
        await server.stop()

        assert model_task.cancelled()
        assert server._stop_task is None
    finally:
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_start_validation_failure_resets_live(mocker, prometheus_registry):
    server = MLServer(Settings())
    mocker.patch(
        "mlserver.server.log_runtime_security_mode",
        side_effect=RuntimeError("invalid trusted runtimes"),
    )

    try:
        with pytest.raises(RuntimeError, match="invalid trusted runtimes allowlist"):
            await server.start()

        assert not server._live
        assert server._stop_task is None
    finally:
        await server.stop()


async def test_server_propagates_model_operation_timeout(prometheus_registry):
    model_operation_timeout = 37
    server = MLServer(Settings(model_operation_timeout=model_operation_timeout))

    try:
        assert (
            server._model_registry._model_operation_timeout == model_operation_timeout
        )
        assert (
            server._model_repository_handlers._model_operation_timeout
            == model_operation_timeout
        )
    finally:
        await server.stop()


async def test_repeated_stop_does_not_start_duplicate_cleanup(
    mocker, prometheus_registry
):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_calls = 0

    async def stop_resources(_sig=None):
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        await release_cleanup.wait()

    mocker.patch.object(server, "_stop_resources", new=stop_resources)

    try:
        first_stop = asyncio.create_task(server.stop())
        await cleanup_started.wait()
        await server.stop()

        assert cleanup_calls == 1
    finally:
        release_cleanup.set()
        await first_stop
        server._stop_resources = original_stop_resources
        await server.stop()


@pytest.mark.parametrize(
    "frontend_workers,expect_ndarray", [(0, False), (1, False), (2, True)]
)
async def test_bootstrap_payload_mode_per_topology(
    prometheus_registry, frontend_workers, expect_ndarray
):
    """Bootstrap pin: MLServer construction sets the process-global
    payload mode from the topology — list payloads (master's data
    flow) at zero or ONE frontend process, ndarray payloads only at
    2+. The single supervised frontend (fw=1) must keep list payloads:
    it has exactly one frontend loop, the saturation profile where
    frontend-side materialisation measured as a p99 regression. The
    autouse conftest fixture resets the global around this test."""

    def _materialises_lists() -> bool:
        from mlserver.codecs.numpy import _MATERIALISE_LIST_PAYLOADS

        return _MATERIALISE_LIST_PAYLOADS

    server, _ = _placement_server(frontend_workers=frontend_workers)
    assert _materialises_lists() is (not expect_ndarray)


def _placement_server(frontend_workers: int):
    """An MLServer with a stubbed pool registry, for testing where the
    adaptive batcher is placed by _pool_load."""
    from unittest.mock import AsyncMock, Mock

    from mlserver import MLServer
    from mlserver.settings import Settings
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        parallel_workers=1,
        parallel_inference_transport="zmq",
        frontend_workers=frontend_workers,
    )
    pool = Mock()
    pool.load_model = AsyncMock()
    pool.unload_model = AsyncMock()
    pool.model_initialiser = Mock()
    server = MLServer(settings, inference_pool_registry=pool)
    # on_model_load[0] is the _pool_load closure.
    return server, pool


def _batched_settings(name: str = "m"):
    from mlserver.settings import ModelSettings

    return ModelSettings(
        name=name,
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.05,
    )


def _unbatched_settings(name: str = "m"):
    from mlserver.settings import ModelSettings

    return ModelSettings(
        name=name,
        implementation="tests.fixtures.SumModel",
    )


def _parallel_stub(settings):
    from unittest.mock import MagicMock

    from mlserver.parallel.model import ParallelModel

    model = MagicMock()
    model.settings = settings
    return ParallelModel(model, MagicMock())


async def test_pool_load_wraps_batched_wrapper_at_direct_server(prometheus_registry):
    """At frontend_workers=0 (the direct server) a batched model's parallel
    wrapper gets the MAIN-process batcher — the previous behaviour, restored."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=0)
    settings = _batched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    assert AdaptiveBatcher.get_batcher(wrapper) is not None


async def test_pool_load_leaves_wrapper_unwrapped_at_multiple_frontends(
    prometheus_registry,
):
    """In the supervised topology (frontend_workers>=1) the batcher lives
    in the worker: the main process must NOT wrap the parallel wrapper
    (one placement only)."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=2)
    settings = _batched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    assert AdaptiveBatcher.get_batcher(wrapper) is None


async def test_pool_load_never_wraps_unbatched_model_direct_server(
    prometheus_registry,
):
    """An unbatched model gets no batcher at frontend_workers=0."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=0)
    settings = _unbatched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    assert AdaptiveBatcher.get_batcher(wrapper) is None


async def test_pool_load_never_wraps_unbatched_model_multiple_frontends(
    prometheus_registry,
):
    """An unbatched model gets no batcher in the supervised topology."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=2)
    settings = _unbatched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    assert AdaptiveBatcher.get_batcher(wrapper) is None


async def test_pool_load_inprocess_optout_batches_in_main_direct_server(
    prometheus_registry,
):
    """A model the pool returns unchanged (model-level parallel
    inference opt-out) runs in-process and gets the main-process
    batcher at frontend_workers=0."""
    from mlserver.batching import AdaptiveBatcher
    from tests.fixtures import SumModel

    server, pool = _placement_server(frontend_workers=0)
    settings = _batched_settings()
    model = SumModel(settings)
    pool.load_model.return_value = model

    loaded = await server._model_registry._on_model_load[0](model)

    assert loaded is model
    assert AdaptiveBatcher.get_batcher(model) is not None


async def test_pool_load_inprocess_optout_batches_in_main_multiple_frontends(
    prometheus_registry,
):
    """Same in-process opt-out in the supervised topology: the model runs
    in the main process, so the main-process batcher applies there too."""
    from mlserver.batching import AdaptiveBatcher
    from tests.fixtures import SumModel

    server, pool = _placement_server(frontend_workers=2)
    settings = _batched_settings()
    model = SumModel(settings)
    pool.load_model.return_value = model

    loaded = await server._model_registry._on_model_load[0](model)

    assert loaded is model
    assert AdaptiveBatcher.get_batcher(model) is not None


async def test_direct_server_unload_chain_tears_batcher_down(prometheus_registry):
    """At frontend_workers=0 the main unload chain includes
    unload_batching (first, while the wrapped instance is in hand)."""
    from mlserver.batching import unload_batching

    server, _ = _placement_server(frontend_workers=0)
    assert server._model_registry._on_model_unload[0] is unload_batching


async def test_multiple_frontend_unload_chain_has_no_batcher_teardown(
    prometheus_registry,
):
    """In the supervised topology the worker owns batcher teardown: the
    main unload chain has no unload_batching."""
    from mlserver.batching import unload_batching

    server, _ = _placement_server(frontend_workers=2)
    assert unload_batching not in server._model_registry._on_model_unload


async def test_dataplane_stream_cache_wired_through_real_registry(
    prometheus_registry,
):
    """Pin the by-reference hook wiring: the DataPlane's
    stream-single-shot cache must invalidate through a REAL
    registry.load, not just direct hook invocation. If a registry
    refactor starts copying the hook lists, the append in
    MLServer.__init__ silently disconnects invalidation and this test
    is the only thing that notices (the direct-hook tests keep
    passing)."""
    from unittest.mock import AsyncMock

    from mlserver.settings import ModelSettings
    from tests.fixtures import SumModel

    server, pool = _placement_server(frontend_workers=0)
    data_plane = server._data_plane
    registry = server._model_registry

    # Real models through the stubbed pool: the initialiser builds a
    # SumModel and the pool returns it unchanged (in-process path).
    # The registry captured the pool's initialiser Mock at
    # construction, so patch the registry's own initialiser (before
    # the first load creates the per-model registry); load_model is a
    # live attribute lookup on the pool, so replacing it works.
    pool.load_model = AsyncMock(side_effect=lambda model: model)
    pool.unload_model = AsyncMock(return_value=None)
    registry._model_initialiser = lambda settings: SumModel(settings)

    # The stubbed pool registry returns the model unchanged.
    real_get_model = data_plane._model_registry.get_model

    async def counting_get_model(name, version=None):
        return await real_get_model(name, version)

    data_plane._model_registry.get_model = AsyncMock(side_effect=counting_get_model)

    settings = ModelSettings(
        name="wired-cache-model",
        implementation=SumModel,
    )
    await registry.load(settings)

    # First query computes and caches.
    assert await data_plane.model_stream_is_single_shot("wired-cache-model") is True
    assert data_plane._model_registry.get_model.await_count == 1
    # Second query is served from the cache.
    assert await data_plane.model_stream_is_single_shot("wired-cache-model") is True
    assert data_plane._model_registry.get_model.await_count == 1

    # A REAL reload must invalidate: the load hooks fire through the
    # registry's own machinery, hitting the DataPlane hook that was
    # appended by reference in MLServer.__init__.
    await registry.load(settings)
    assert await data_plane.model_stream_is_single_shot("wired-cache-model") is True
    assert data_plane._model_registry.get_model.await_count == 2

    await registry.unload("wired-cache-model")
