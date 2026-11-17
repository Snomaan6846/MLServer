"""Tests for the frontend_workers setting and supervisor."""

import pytest

from mlserver.settings import Settings


def test_frontend_workers_defaults_to_one():
    """Default is 1 — the single-process behaviour, no supervisor."""
    settings = Settings()
    assert settings.frontend_workers == 1


def test_frontend_workers_accepts_positive():
    settings = Settings(frontend_workers=4, parallel_inference_transport="zmq")
    assert settings.frontend_workers == 4


def test_frontend_workers_rejects_zero():
    with pytest.raises(Exception):
        Settings(frontend_workers=0)


def test_frontend_workers_rejects_negative():
    with pytest.raises(Exception):
        Settings(frontend_workers=-1)


def test_frontend_workers_gt1_with_queue_raises():
    """queue + frontend_workers>1 is a startup validation error."""
    with pytest.raises(ValueError, match="frontend_workers=4.*requires.*zmq"):
        Settings(
            frontend_workers=4,
            parallel_inference_transport="queue",
        )


def test_frontend_workers_gt1_with_zmq_valid():
    """zmq + frontend_workers>1 is a valid configuration."""
    settings = Settings(
        frontend_workers=4,
        parallel_inference_transport="zmq",
    )
    assert settings.frontend_workers == 4
    assert settings.parallel_inference_transport == "zmq"


def test_frontend_workers_1_with_queue_valid():
    """frontend_workers=1 works with any transport (backward compat)."""
    settings = Settings(
        frontend_workers=1,
        parallel_inference_transport="queue",
    )
    assert settings.frontend_workers == 1


def test_frontend_workers_env_var():
    """MLSERVER_FRONTEND_WORKERS env var works (with zmq transport)."""
    import os

    os.environ["MLSERVER_FRONTEND_WORKERS"] = "3"
    os.environ["MLSERVER_PARALLEL_INFERENCE_TRANSPORT"] = "zmq"
    try:
        settings = Settings()
        assert settings.frontend_workers == 3
    finally:
        del os.environ["MLSERVER_FRONTEND_WORKERS"]
        del os.environ["MLSERVER_PARALLEL_INFERENCE_TRANSPORT"]


def test_frontend_workers_gt1_without_pool_raises():
    """frontend_workers>1 with parallel_workers=0 is a startup error.

    Without a shared pool every frontend would load its own in-process
    copy of each model, multiplying memory and fragmenting batching.
    """
    with pytest.raises(Exception, match="parallel_workers >= 1"):
        Settings(
            frontend_workers=2,
            parallel_workers=0,
            parallel_inference_transport="zmq",
        )


def test_frontend_workers_gt1_with_pool_valid():
    settings = Settings(
        frontend_workers=3,
        parallel_workers=2,
        parallel_inference_transport="zmq",
    )
    assert settings.frontend_workers == 3


def test_frontend_workers_1_without_pool_valid():
    """Single frontend with in-process models keeps working as before."""
    settings = Settings(frontend_workers=1, parallel_workers=0)
    assert settings.parallel_workers == 0


def test_frontend_workers_gt1_without_pool_and_queue_raises_both():
    """Both validators fire; the queue one is checked first."""
    with pytest.raises(Exception):
        Settings(frontend_workers=2, parallel_workers=0)


def test_parallel_routing_rotation_defaults_and_validation():
    assert Settings().parallel_routing_rotation == 1.0
    assert Settings(parallel_routing_rotation=0).parallel_routing_rotation == 0
    with pytest.raises(Exception):
        Settings(parallel_routing_rotation=-0.5)


async def test_dead_frontend_wakes_pending_ack_collection():
    """A frontend crash during an in-flight model operation must not
    stall the operation for the full timeout: the auto-acked slot wakes
    the supervisor's ack collection immediately."""
    import asyncio
    from unittest.mock import Mock

    from mlserver.frontend import FrontendSupervisor, _PendingOp
    from mlserver.settings import Settings

    settings = Settings(
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    supervisor = FrontendSupervisor(settings)
    supervisor._frontends = {0: Mock(pid=101), 1: Mock(pid=102)}
    supervisor._restart_counts = {0: 0, 1: 0}

    pending = _PendingOp(origin=0, ok=True, error=None)
    supervisor._pending_ops["op-dead"] = pending

    collect_task = asyncio.create_task(
        supervisor._collect_acks("op-dead", 0, True, None)
    )
    await asyncio.sleep(0.05)
    assert not collect_task.done()

    # The origin acknowledges the broadcast apply (as it does on receipt).
    from mlserver.parallel.shared import ApplyAck

    await supervisor._handle_ack(0, ApplyAck(op_id="op-dead"))
    await asyncio.sleep(0.05)
    assert not collect_task.done()

    # Frontend 1 dies mid-operation: its auto-acked slot must wake the
    # collector immediately instead of stalling until the timeout.
    supervisor._handle_dead_frontend(1, supervisor._frontends[1])

    # The collector must resolve promptly (not wait the 30s timeout).
    await asyncio.wait_for(collect_task, timeout=2.0)

    # Cleanup: close out the supervisor without spawning restarts.
    supervisor._closing = True
    pending_tasks = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    for task in pending_tasks:
        task.cancel()
    await asyncio.gather(*pending_tasks, return_exceptions=True)


def test_frontend_max_restarts_defaults_and_validation():
    assert Settings().frontend_max_restarts == 5
    assert Settings(frontend_max_restarts=0).frontend_max_restarts == 0
    with pytest.raises(Exception):
        Settings(frontend_max_restarts=-1)


def test_frontend_workers_upper_bound():
    with pytest.raises(Exception):
        Settings(frontend_workers=65, parallel_inference_transport="zmq")
    assert (
        Settings(
            frontend_workers=64, parallel_inference_transport="zmq"
        ).frontend_workers
        == 64
    )
