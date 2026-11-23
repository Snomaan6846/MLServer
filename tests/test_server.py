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


@pytest.mark.parametrize("frontend_workers", [0, 1, 2])
async def test_bootstrap_payload_mode_per_topology(
    prometheus_registry, frontend_workers
):
    """Bootstrap pin after the gate removal: MLServer construction
    leaves the ndarray pipeline active at EVERY topology — the codec
    output root stays an ndarray (no process-global list
    materialisation exists anymore; the mode-aware TensorData
    serializer + orjson OPT_SERIALIZE_NUMPY render it natively)."""

    import numpy as np

    from mlserver.codecs.numpy import NumpyCodec

    server, _ = _placement_server(frontend_workers=frontend_workers)
    payload = np.array([[1.5, 2.5]], dtype=np.float32)
    response_output = NumpyCodec.encode_output(name="foo", payload=payload)
    assert isinstance(response_output.data.root, np.ndarray)


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


@pytest.mark.parametrize("frontend_workers", [0, 2], ids=["direct", "multiple"])
async def test_pool_load_batched_wrapper_placement(
    prometheus_registry, frontend_workers
):
    """A batched model's parallel wrapper gets the MAIN-process
    batcher at frontend_workers=0 (the direct server — the previous
    behaviour, restored); in the supervised topology
    (frontend_workers>=1) the batcher lives in the worker, so the
    main process must NOT wrap the parallel wrapper (one placement
    only)."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=frontend_workers)
    settings = _batched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    if frontend_workers == 0:
        assert AdaptiveBatcher.get_batcher(wrapper) is not None
    else:
        assert AdaptiveBatcher.get_batcher(wrapper) is None


@pytest.mark.parametrize("frontend_workers", [0, 2], ids=["direct", "multiple"])
async def test_pool_load_never_wraps_unbatched_model(
    prometheus_registry, frontend_workers
):
    """An unbatched model gets no batcher in either topology."""
    from mlserver.batching import AdaptiveBatcher

    server, pool = _placement_server(frontend_workers=frontend_workers)
    settings = _unbatched_settings()
    wrapper = _parallel_stub(settings)
    pool.load_model.return_value = wrapper

    loaded = await server._model_registry._on_model_load[0](
        type("M", (), {"settings": settings})()
    )

    assert loaded is wrapper
    assert AdaptiveBatcher.get_batcher(wrapper) is None


@pytest.mark.parametrize("frontend_workers", [0, 2], ids=["direct", "multiple"])
async def test_pool_load_inprocess_optout_batches_in_main(
    prometheus_registry, frontend_workers
):
    """A model the pool returns unchanged (model-level parallel
    inference opt-out) runs in-process and gets the main-process
    batcher in either topology: the model runs in the main process,
    so the main-process batcher applies."""
    from mlserver.batching import AdaptiveBatcher
    from tests.fixtures import SumModel

    server, pool = _placement_server(frontend_workers=frontend_workers)
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


def test_freeze_startup_heap_moves_objects_to_permanent_generation():
    """collect-then-freeze: the permanent generation is non-empty after
    the call, and the operation is idempotent."""
    import gc

    from mlserver.server import _freeze_startup_heap

    before = gc.get_freeze_count()
    _freeze_startup_heap()
    after = gc.get_freeze_count()
    assert after >= before
    assert after > 0
