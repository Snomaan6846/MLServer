from tests.utils import handle_and_settle

"""Unit tests for the frontend-side shared-pool registry and protocol."""

import asyncio
import itertools

from multiprocessing import Pipe
from unittest.mock import AsyncMock, Mock

import pytest

from mlserver.model import MLModel
from mlserver.parallel.shared import (
    deserialise_model_settings,
    serialise_model_settings,
    ApplyAck,
    ApplyLoad,
    ApplyUnload,
    CancelRequest,
    OpRequest,
    OpResult,
    SharedPoolRegistry,
    StartupApplyFailed,
    StartupComplete,
    WorkerReadyEvent,
    WorkerStoppedEvent,
)
from mlserver.parallel.model import ParallelModel
from mlserver.settings import ModelSettings, Settings
from tests.fixtures import SumModel


@pytest.fixture
def shared_settings() -> Settings:
    return Settings(
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=5,
    )


@pytest.fixture
def pipes():
    parent, child = Pipe(duplex=True)
    yield parent, child
    parent.close()
    child.close()


@pytest.fixture
async def shared(shared_settings: Settings, pipes):
    _parent, child = pipes
    registry = SharedPoolRegistry(
        shared_settings,
        frontend_id=0,
        reply_endpoint=f"ipc:///tmp/mlserver-test-shared-{id(object()):x}",
        control=child,
    )
    registry.attach_registry(Mock())
    await registry.start()
    yield registry
    await registry.close()


def _sum_settings(name: str = "sum-model") -> ModelSettings:
    return ModelSettings(
        name=name, implementation="tests.fixtures.SumModel", max_batch_size=1
    )


def test_serialise_round_trip_preserves_source(sum_model_settings):
    """The control-plane round trip preserves the private `_source`
    attribute: pydantic's model_validate_json drops it, so the
    deserialiser must take the dict path. Dev-mode models resolve their
    custom implementation relative to `_source`; losing it silently
    falls back to the base MLModel."""
    sum_model_settings._source = "/some/dev/dir/settings.json"

    serialised = serialise_model_settings(sum_model_settings)
    restored = deserialise_model_settings(serialised)

    assert restored._source == "/some/dev/dir/settings.json"


def test_serialise_rejects_non_json_values(sum_model_settings):
    """A value the JSON encoder cannot express (e.g. a Path in
    parameters.extra from a custom repository) is rejected with a
    clear error naming the model — before anything reaches the control
    plane. Silent string conversion would change the type the worker's
    runtime sees."""
    from pathlib import Path

    import pytest as pytest_module

    from mlserver.errors import MLServerError

    sum_model_settings.parameters = (
        sum_model_settings.parameters or type(sum_model_settings.parameters)()
    )
    sum_model_settings.parameters.extra = {"model_dir": Path("/some/dir")}

    with pytest_module.raises(MLServerError, match="sum-model") as excinfo:
        serialise_model_settings(sum_model_settings)
    assert "JSON-serialisable" in str(excinfo.value)


async def test_load_model_rejects_non_json_settings(shared, sum_model_settings):
    """The frontend load path surfaces the rejection: the caller's
    load fails with the clear error, no OpRequest is forwarded."""
    from pathlib import Path

    import pytest as pytest_module

    from mlserver.errors import MLServerError
    from mlserver.model import MLModel

    sum_model_settings.parameters = (
        sum_model_settings.parameters or type(sum_model_settings.parameters)()
    )
    sum_model_settings.parameters.extra = {"model_dir": Path("/some/dir")}
    model = MLModel(sum_model_settings)

    with pytest_module.raises(MLServerError, match="JSON-serialisable"):
        await shared.load_model(model)


async def _recv(parent, timeout: float = 5.0) -> object:
    # Poll before reading: run_in_executor work cannot be cancelled, so
    # a plain recv would leave the executor thread blocked inside
    # parent.recv() past this helper's timeout, until the fixture
    # closes the pipe at teardown. Reading only when a message is
    # pending keeps the executor call immediate.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if parent.poll():
            return await loop.run_in_executor(None, parent.recv)
        if loop.time() >= deadline:
            raise asyncio.TimeoutError()
        await asyncio.sleep(0.01)


async def test_load_model_forwards_and_resolves(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """load_model sends an OpRequest and resolves when the supervisor
    answers with OpResult after all frontends acknowledged."""
    parent, _child = pipes
    model = SumModel(sum_model_settings)
    load_task = asyncio.create_task(shared.load_model(model))
    request = await _recv(parent)
    assert isinstance(request, OpRequest)
    assert request.kind == "load"
    assert request.serialised_model_settings is not None
    assert (
        deserialise_model_settings(request.serialised_model_settings).name
        == sum_model_settings.name
    )
    # Supervisor completes the op everywhere.
    parent.send(OpResult(op_id=request.op_id, ok=True))
    wrapped = await asyncio.wait_for(load_task, timeout=5)
    assert isinstance(wrapped, ParallelModel)


async def test_load_model_failure_propagates(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    parent, _child = pipes
    model = SumModel(sum_model_settings)
    load_task = asyncio.create_task(shared.load_model(model))
    request = await _recv(parent)
    assert isinstance(request, OpRequest)
    parent.send(OpResult(op_id=request.op_id, ok=False, error="boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(load_task, timeout=5)


async def test_apply_load_on_non_origin_frontend(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """A broadcast apply registers the model locally and acknowledges."""
    parent, _child = pipes
    registry = Mock(spec=["load", "unload"])
    registry.load = AsyncMock()
    shared.attach_registry(registry)
    await handle_and_settle(
        shared,
        ApplyLoad(
            op_id="op-1",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        ),
    )
    registry.load.assert_awaited_once()
    assert registry.load.await_args is not None
    assert isinstance(registry.load.await_args.args[0], ModelSettings)
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)
    assert ack.op_id == "op-1"  # type: ignore[union-attr]


async def test_apply_load_on_origin_skips_local_apply(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """The origin's own suspended load call completes the registration;
    the listener only acknowledges receipt."""
    parent, _child = pipes
    registry = Mock(spec=["load", "unload"])
    registry.load = AsyncMock()
    shared.attach_registry(registry)
    # Simulate a suspended origin operation.
    waiter = asyncio.get_running_loop().create_future()
    shared._op_waiters["op-2"] = waiter
    await handle_and_settle(
        shared,
        ApplyLoad(
            op_id="op-2",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        ),
    )
    registry.load.assert_not_awaited()
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)
    assert ack.op_id == "op-2"


async def test_apply_load_failure_never_registers(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    parent, _child = pipes
    registry = Mock(spec=["load", "unload"])
    registry.load = AsyncMock()
    shared.attach_registry(registry)
    # No waiter: this runs the PEER path, where ignoring ok=False would
    # call the registry load and fail the assertion below. With a
    # waiter (the origin path), the handler skips the load regardless.
    await handle_and_settle(
        shared,
        ApplyLoad(
            op_id="op-3",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=False,
            error="pool load failed",
        ),
    )
    registry.load.assert_not_awaited()
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)


async def test_apply_unload_on_non_origin_frontend(shared: SharedPoolRegistry, pipes):
    parent, _child = pipes
    registry = Mock(spec=["load", "unload", "unload_version"])
    registry.unload = AsyncMock()
    registry.unload_exact_version = AsyncMock()
    shared.attach_registry(registry)
    await handle_and_settle(
        shared, ApplyUnload(op_id="op-4", name="sum-model", version=None, ok=True)
    )
    registry.unload_exact_version.assert_awaited_once_with("sum-model", None, False)
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)


async def test_apply_unload_missing_local_model_is_tolerated(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """A versionless unload for a model the frontend never registered
    resolves as a no-op through the real registry (which returns
    normally for an absent exact version), not by exercising the
    exception handler."""
    parent, _child = pipes
    from mlserver.registry import MultiModelRegistry

    # A real registry containing only ghost/v2: the versionless
    # unload targets the unversioned key, which is absent. Version
    # lives in parameters (the settings-level version is a property).
    real_registry = MultiModelRegistry()
    versioned = sum_model_settings.model_copy(deep=True)
    versioned.name = "ghost"
    versioned.parameters.version = "v2"
    await real_registry.load(versioned)
    assert "ghost" in real_registry._models

    shared.attach_registry(real_registry)

    await handle_and_settle(
        shared, ApplyUnload(op_id="op-5", name="ghost", version=None, ok=True)
    )
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)

    # The exact ghost/v2 instance survives the versionless miss (the
    # exact unversioned key was absent — a no-op, not a removal).
    assert "ghost" in real_registry._models
    model = await real_registry.get_model("ghost", "v2")
    assert model is not None
    assert model.version == "v2"


async def test_worker_events_manage_proxies(shared: SharedPoolRegistry):
    """Ready events create routable proxies; stop events revoke them."""
    await handle_and_settle(
        shared, WorkerReadyEvent(pid=101, request_endpoint="ipc:///q1")
    )
    assert 101 in shared._dispatcher._ready_workers
    assert shared._dispatcher._ready_workers[101].pid == 101
    assert shared._dispatcher._ready_workers[101]._zmq_request_endpoint == "ipc:///q1"
    await handle_and_settle(shared, WorkerStoppedEvent(pid=101, exit_code=-15))
    assert 101 not in shared._dispatcher._ready_workers


async def test_worker_reincarnation_replaces_proxy(shared: SharedPoolRegistry):
    """A new incarnation on the same PID replaces the old proxy."""
    await handle_and_settle(
        shared, WorkerReadyEvent(pid=202, request_endpoint="ipc:///old")
    )
    first = shared._dispatcher._ready_workers[202]
    await handle_and_settle(shared, WorkerStoppedEvent(pid=202, exit_code=-9))
    await handle_and_settle(
        shared, WorkerReadyEvent(pid=202, request_endpoint="ipc:///new")
    )
    second = shared._dispatcher._ready_workers[202]
    assert second is not first
    assert second._zmq_request_endpoint == "ipc:///new"


async def test_cancel_request_reaches_control_pipe(shared: SharedPoolRegistry, pipes):
    parent, _child = pipes
    shared.cancel_request("sum-model", "req-42", worker_pid=101, model_version="v2")
    message = await _recv(parent)
    assert isinstance(message, CancelRequest)
    assert message.model_name == "sum-model"
    assert message.request_id == "req-42"
    assert message.worker_pid == 101
    assert message.model_version == "v2"
    # A cancel without a version (the legacy shape) still works.
    shared.cancel_request("sum-model", "req-43", worker_pid=101)
    legacy = await _recv(parent)
    assert isinstance(legacy, CancelRequest)
    assert legacy.model_version is None


async def test_close_cancels_waiters(shared: SharedPoolRegistry, pipes):
    _parent, child = pipes
    waiter = asyncio.get_running_loop().create_future()
    shared._op_waiters["op-6"] = waiter
    await shared.close()
    assert waiter.cancelled()
    # Closing twice is a no-op.
    await shared.close()
    assert child.closed


async def test_model_initialiser_env_models_stay_dummies(
    shared: SharedPoolRegistry, env_model_settings
):
    """Environment-tarball models are not instantiated in the frontend."""
    model = shared.model_initialiser(env_model_settings)
    assert type(model) is MLModel


async def test_slow_apply_does_not_block_the_listener(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """The listener stays receptive while a slow peer apply runs: a
    worker event arriving mid-apply is processed immediately instead
    of freezing every control message (acks, results, worker events)
    behind the apply — the freeze that let two concurrent same-model
    loads from different frontends out-wait each other's ack
    collection."""
    _parent, _child = pipes
    registry = Mock(spec=["load", "unload"])
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def slow_load(settings):
        apply_started.set()
        await release_apply.wait()
        return Mock()

    registry.load = slow_load
    shared.attach_registry(registry)

    apply_task = await asyncio.wait_for(
        shared._handle(
            ApplyLoad(
                op_id="op-slow",
                serialised_model_settings=serialise_model_settings(sum_model_settings),
                ok=True,
            )
        ),
        timeout=5,
    )
    assert apply_task is not None
    await asyncio.wait_for(apply_started.wait(), timeout=5)

    # A worker event while the apply is still running: the listener
    # was free to receive and process it. The wait_for guards the
    # regression's failure mode: an inline apply blocks the listener
    # (and this scheduling call) indefinitely.
    await asyncio.wait_for(
        shared._handle(WorkerReadyEvent(pid=101, request_endpoint="ipc:///q1")),
        timeout=5,
    )
    assert 101 in shared._proxies

    release_apply.set()
    await apply_task


async def test_same_key_applies_sequence_in_arrival_order(
    shared: SharedPoolRegistry, sum_model_settings
):
    """Two applies for one model key apply in arrival order: routing
    tags and registry state stay consistent across concurrent
    operations on the same key."""
    registry = Mock(spec=["load", "unload"])
    order: list[str] = []
    gate = asyncio.Event()

    async def load_a(settings):
        order.append("a-start")
        await gate.wait()
        order.append("a-end")
        return Mock()

    async def load_b(settings):
        order.append("b-start")
        order.append("b-end")
        return Mock()

    loader_calls = itertools.count()

    async def registry_load(settings):
        # Direct assignment (not an AsyncMock side effect): the loader
        # must be awaited, not returned as a coroutine object.
        if next(loader_calls) == 0:
            await load_a(settings)
        else:
            await load_b(settings)

    registry.load = registry_load
    shared.attach_registry(registry)

    message_a = ApplyLoad(
        op_id="op-a",
        serialised_model_settings=serialise_model_settings(sum_model_settings),
        ok=True,
    )
    message_b = ApplyLoad(
        op_id="op-b",
        serialised_model_settings=serialise_model_settings(sum_model_settings),
        ok=True,
    )
    task_a = await shared._handle(message_a)
    await asyncio.sleep(0.05)
    task_b = await shared._handle(message_b)
    await asyncio.sleep(0.05)

    # The second apply is chained behind the first: it has not
    # started while the first is mid-load.
    assert order == ["a-start"]

    gate.set()
    assert task_a is not None and task_b is not None
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)
    assert order == ["a-start", "a-end", "b-start", "b-end"]


async def test_startup_complete_waits_for_pending_applies(
    shared: SharedPoolRegistry, sum_model_settings
):
    """The readiness gate releases only after every pending startup
    apply settled: StartupComplete arriving behind a slow apply must
    not open the gate early."""
    registry = Mock(spec=["load", "unload"])
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def slow_load(settings):
        apply_started.set()
        await release_apply.wait()
        return Mock()

    registry.load = slow_load
    shared.attach_registry(registry)
    shared._data_plane = Mock()

    apply_task = await shared._handle(
        ApplyLoad(
            op_id="op-start",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    assert apply_task is not None
    await asyncio.wait_for(apply_started.wait(), timeout=5)

    startup_task = asyncio.create_task(shared._handle(StartupComplete(model_count=1)))
    await asyncio.sleep(0.05)
    # The gate has not released while the apply is pending.
    assert not startup_task.done()
    shared._data_plane.startup_complete.assert_not_called()

    release_apply.set()
    await asyncio.wait_for(startup_task, timeout=5)
    shared._data_plane.startup_complete.assert_called_once()


async def test_failed_startup_apply_reports_to_supervisor(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """A frontend whose startup apply failed reports it at
    StartupComplete: the supervisor decides its fate (termination and
    restart) instead of the frontend lingering alive-but-never-ready,
    still receiving SO_REUSEPORT traffic it cannot serve."""
    parent, _child = pipes
    registry = Mock(spec=["load", "unload"])

    async def failing_load(settings):
        raise RuntimeError("apply failed")

    registry.load = failing_load
    shared.attach_registry(registry)
    shared._data_plane = Mock()

    await handle_and_settle(
        shared,
        ApplyLoad(
            op_id="op-bad",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        ),
    )
    assert shared._startup_apply_failed

    await shared._handle(StartupComplete(model_count=1))

    # The gate stays closed and the failure was reported upstream
    # (behind the failed apply's own acknowledgement).
    shared._data_plane.startup_complete.assert_not_called()
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)
    report = await _recv(parent)
    assert isinstance(report, StartupApplyFailed)
    assert report.model_name == sum_model_settings.name


async def test_chained_apply_cancellation_propagates(
    shared: SharedPoolRegistry, sum_model_settings
):
    """A chain task cancelled while waiting for its predecessor must
    end cancelled without running its own apply: swallowing the
    cancellation (the failure handler around a direct await) would
    make close()'s cancel-and-settle wait out a full model load
    during shutdown."""
    registry = Mock(spec=["load", "unload"])
    a_started = asyncio.Event()
    release_a = asyncio.Event()
    b_ran = asyncio.Event()
    loaders = iter(
        [
            # The first apply blocks; the second chains behind it.
            _blocking_loader(a_started, release_a),
            _recording_loader(b_ran),
        ]
    )

    async def registry_load(settings):
        await next(loaders)(settings)

    registry.load = registry_load
    shared.attach_registry(registry)

    task_a = await shared._handle(
        ApplyLoad(
            op_id="op-a",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    await asyncio.wait_for(a_started.wait(), timeout=5)
    task_b = await shared._handle(
        ApplyLoad(
            op_id="op-b",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    # Let B reach its wait on A.
    await asyncio.sleep(0.05)

    assert task_b is not None
    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_b
    # B's own apply never ran, and B reports cancelled (not
    # completed-with-swallowed-cancellation).
    assert task_b.cancelled()
    assert not b_ran.is_set()

    release_a.set()
    assert task_a is not None
    await asyncio.wait_for(task_a, timeout=5)


def _blocking_loader(started: asyncio.Event, release: asyncio.Event):
    async def _load(settings):
        started.set()
        await release.wait()
        return Mock()

    return _load


def _recording_loader(ran: asyncio.Event):
    async def _load(settings):
        ran.set()
        return Mock()

    return _load


async def test_close_cancels_every_chained_apply(
    shared: SharedPoolRegistry, sum_model_settings
):
    """close() must cancel and settle every link in an apply chain,
    not just the latest: cancelling a chain tip does not cancel the
    links it waits on, and mid-chain applies would otherwise keep
    running as orphan tasks past the teardown."""
    registry = Mock(spec=["load", "unload"])
    a_started = asyncio.Event()
    release_a = asyncio.Event()
    b_ran = asyncio.Event()
    loaders = iter(
        [
            _blocking_loader(a_started, release_a),
            _recording_loader(b_ran),
        ]
    )

    async def registry_load(settings):
        await next(loaders)(settings)

    registry.load = registry_load
    shared.attach_registry(registry)

    task_a = await shared._handle(
        ApplyLoad(
            op_id="op-a",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    await asyncio.wait_for(a_started.wait(), timeout=5)
    # B chains behind the blocked A: it is a mid-chain link, and the
    # chain map only keeps B (the tip).
    task_b = await shared._handle(
        ApplyLoad(
            op_id="op-b",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    await asyncio.sleep(0.05)

    await shared.close()

    # Both links settled as cancelled; neither apply ran to completion
    # past the teardown.
    assert task_a is not None and task_b is not None
    assert task_a.cancelled()
    assert task_b.cancelled()
    assert not b_ran.is_set()
