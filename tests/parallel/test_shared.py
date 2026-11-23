from tests.utils import handle_and_settle

"""Unit tests for the frontend-side shared-pool registry and protocol."""

import asyncio
import itertools

from multiprocessing import Pipe
from unittest.mock import AsyncMock, Mock

import pytest

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
from typing import TYPE_CHECKING

from mlserver.model import MLModel
from mlserver.settings import ModelSettings, Settings
from tests.fixtures import SumModel

if TYPE_CHECKING:
    from mlserver.frontend import FrontendSupervisor


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
    """A broadcast apply registers the model locally and acknowledges
    through the LOCK-FREE entry point — the locked load() would wait
    on the operation lock an origin operation holds across its
    supervisor round (the ACK-collection deadlock)."""
    parent, _child = pipes
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
    registry.load = AsyncMock()
    registry._load_for_apply = AsyncMock()
    shared.attach_registry(registry)
    await handle_and_settle(
        shared,
        ApplyLoad(
            op_id="op-1",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        ),
    )
    registry._load_for_apply.assert_awaited_once()
    assert registry._load_for_apply.await_args is not None
    assert isinstance(registry._load_for_apply.await_args.args[0], ModelSettings)
    # The locked origin path must NOT be used by a peer apply.
    registry.load.assert_not_awaited()
    ack = await _recv(parent)
    assert isinstance(ack, ApplyAck)
    assert ack.op_id == "op-1"  # type: ignore[union-attr]


async def test_apply_load_on_origin_skips_local_apply(
    shared: SharedPoolRegistry, pipes, sum_model_settings
):
    """The origin's own suspended load call completes the registration;
    the listener only acknowledges receipt."""
    parent, _child = pipes
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
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
    registry = Mock(
        spec=[
            "load",
            "unload",
            "unload_version",
            "unload_exact_version",
            "_unload_for_apply",
        ]
    )
    registry.unload = AsyncMock()
    registry.unload_exact_version = AsyncMock()
    registry._unload_for_apply = AsyncMock()
    shared.attach_registry(registry)
    await handle_and_settle(
        shared, ApplyUnload(op_id="op-4", name="sum-model", version=None, ok=True)
    )
    registry._unload_for_apply.assert_awaited_once_with("sum-model", None, False)
    # The locked origin path must NOT be used by a peer apply.
    registry.unload_exact_version.assert_not_awaited()
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def slow_load(settings, stale_check=None):
        apply_started.set()
        await release_apply.wait()
        return Mock()

    registry._load_for_apply = slow_load
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
    order: list[str] = []
    gate = asyncio.Event()

    async def load_a(settings):
        order.append("a-start")
        await gate.wait()
        order.append("a-end")
        return Mock()

    async def load_b(settings, stale_check=None):
        order.append("b-start")
        order.append("b-end")
        return Mock()

    loader_calls = itertools.count()

    async def registry_load(settings, stale_check=None):
        # Direct assignment (not an AsyncMock side effect): the loader
        # must be awaited, not returned as a coroutine object.
        if next(loader_calls) == 0:
            await load_a(settings)
        else:
            await load_b(settings)

    registry._load_for_apply = registry_load
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def slow_load(settings, stale_check=None):
        apply_started.set()
        await release_apply.wait()
        return Mock()

    registry._load_for_apply = slow_load
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])

    async def failing_load(settings, stale_check=None):
        raise RuntimeError("apply failed")

    registry._load_for_apply = failing_load
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
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

    async def registry_load(settings, stale_check=None):
        await next(loaders)(settings)

    registry._load_for_apply = registry_load
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
    registry = Mock(spec=["load", "unload", "_load_for_apply"])
    a_started = asyncio.Event()
    release_a = asyncio.Event()
    b_ran = asyncio.Event()
    loaders = iter(
        [
            _blocking_loader(a_started, release_a),
            _recording_loader(b_ran),
        ]
    )

    async def registry_load(settings, stale_check=None):
        await next(loaders)(settings)

    registry._load_for_apply = registry_load
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


# ------------------------------------------------------------------
# Peer-apply lock freedom (the ACK-collection deadlock)
# ------------------------------------------------------------------


class TestPeerApplyLockFreedom:
    """Peer applies must not wait on registry operation locks.

    An origin operation holds its frontend's registry locks (the
    multi-registry per-key lock AND the single-registry operation
    lock) across its supervisor round: the on-model-load hook waits
    for the pool's acknowledgements while the locks stay held. A peer
    apply that waited on those locks would stall the very
    acknowledgement collection the origin is waiting on — with two
    frontends originating same-key operations concurrently, each
    origin's suspended operation blocks the other's apply, and both
    report failure for operations that commit. Peer applies run
    lock-free and are sequenced per key by the apply chain instead.
    """

    @pytest.fixture
    async def real_shared(self, shared_settings: Settings, pipes):
        """A shared registry over a REAL MultiModelRegistry (bare: no
        hooks — the load/unload bodies are exercised, not the wiring
        above them)."""
        from mlserver.registry import MultiModelRegistry

        _parent, child = pipes
        registry = SharedPoolRegistry(
            shared_settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-lockfree-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry()
        registry.attach_registry(model_registry)
        await registry.start()
        yield registry, model_registry
        await registry.close()

    async def _held_locks(self, model_registry, name: str):
        """The (multi, single) operation locks an origin operation on
        `name` would hold across its supervisor round."""
        single = model_registry._models[name]
        return (
            model_registry._get_operation_lock(name),
            single._operation_lock,
        )

    async def test_load_apply_does_not_wait_on_origin_locks(
        self, real_shared, pipes, sum_model_settings
    ):
        """A load apply must complete while an origin operation on the
        same key holds BOTH registry locks — the reviewer's ACK-cycle
        construction: without lock freedom the apply stalls the
        acknowledgement the suspended origin is waiting on."""
        parent, _child = pipes
        shared, model_registry = real_shared
        settings = _sum_settings("lock-model")

        # Create the model's registries, then suspend like an origin:
        # both locks held across an await.
        await model_registry.load(settings)
        multi_lock, single_lock = await self._held_locks(model_registry, "lock-model")

        async with multi_lock, single_lock:
            task = await shared._handle(
                ApplyLoad(
                    op_id="op-held",
                    serialised_model_settings=serialise_model_settings(settings),
                    ok=True,
                    generation=2,
                )
            )
            assert task is not None
            # The discriminator: the apply must not queue on the locks.
            # (A locked apply hangs here until the origin's
            # model_operation_timeout releases them.)
            await asyncio.wait_for(task, timeout=5)

        ack = await _recv(parent)
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-held"  # type: ignore[union-attr]
        model = await model_registry.get_model("lock-model")
        assert model is not None

    async def test_unload_apply_does_not_wait_on_origin_locks(
        self, real_shared, pipes, sum_model_settings
    ):
        """An unload apply must complete while an origin operation on
        the same key holds both registry locks."""
        parent, _child = pipes
        shared, model_registry = real_shared
        settings = _sum_settings("lock-model")

        await model_registry.load(settings)
        multi_lock, single_lock = await self._held_locks(model_registry, "lock-model")

        async with multi_lock, single_lock:
            task = await shared._handle(
                ApplyUnload(
                    op_id="op-held",
                    name="lock-model",
                    version=None,
                    ok=True,
                    generation=2,
                )
            )
            assert task is not None
            await asyncio.wait_for(task, timeout=5)

        ack = await _recv(parent)
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-held"  # type: ignore[union-attr]
        assert "lock-model" not in model_registry._models

    async def test_peer_apply_leaves_locks_released(
        self, real_shared, pipes, sum_model_settings
    ):
        """The lock-free apply must not LEAVE a lock held either: the
        next origin operation on the key must acquire both locks
        without waiting."""
        _parent, _child = pipes
        shared, model_registry = real_shared
        settings = _sum_settings("lock-model")

        await model_registry.load(settings)
        multi_lock, single_lock = await self._held_locks(model_registry, "lock-model")

        task = await shared._handle(
            ApplyLoad(
                op_id="op-free",
                serialised_model_settings=serialise_model_settings(settings),
                ok=True,
                generation=2,
            )
        )
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

        assert not multi_lock.locked()
        assert not single_lock.locked()
        # Acquirable immediately (not just unlocked at the sample
        # instant).
        async with multi_lock, single_lock:
            pass

    async def test_failed_peer_apply_still_acks(
        self, shared: SharedPoolRegistry, pipes, sum_model_settings
    ):
        """The always-ACK contract: a peer apply whose local load
        fails must still acknowledge — without the ack the
        supervisor's collector waits the full model_operation_timeout
        for this frontend."""
        parent, _child = pipes
        registry = Mock(spec=["load", "unload", "_load_for_apply"])
        registry.load = AsyncMock()
        registry._load_for_apply = AsyncMock(side_effect=RuntimeError("boom"))
        shared.attach_registry(registry)

        task = await shared._handle(
            ApplyLoad(
                op_id="op-fail",
                serialised_model_settings=serialise_model_settings(sum_model_settings),
                ok=True,
            )
        )
        assert task is not None
        # Completes (the failure is contained in the apply), and acks.
        await asyncio.wait_for(task, timeout=5)
        ack = await _recv(parent)
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-fail"  # type: ignore[union-attr]

    async def test_origin_timeout_releases_locks_and_converges(
        self, shared_settings: Settings, pipes, sum_model_settings
    ):
        """The bounded recovery of the origin's critical section: a
        load whose pool round never resolves times out within
        model_operation_timeout, the registry locks are RELEASED (an
        operation that kept them would wedge every later operation on
        the key), and a subsequent peer apply converges the frontend."""
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=0.5,
        )
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-timeout-{id(object()):x}",
            control=child,
        )
        # The registry's deferral window must exceed the shared pool's
        # operation timeout: with equal windows the deferral's wait_for
        # cancels the load before its own timeout error can propagate
        # (the two deadlines race) and the task ends cancelled instead
        # of raising the operation timeout deterministically.
        model_registry = MultiModelRegistry(
            on_model_load=[registry.load_model],
            model_operation_timeout=5,
        )
        registry.attach_registry(model_registry)
        await registry.start()
        try:
            # The origin load: the local model loads, the hook forwards
            # the OpRequest to the pipe (no supervisor answers it) and
            # suspends holding both locks.
            origin = asyncio.create_task(
                model_registry.load(_sum_settings("stuck-model"))
            )
            with pytest.raises(RuntimeError, match="timed out"):
                await asyncio.wait_for(origin, timeout=5)

            # The critical section was released by the failure.
            multi_lock = model_registry._get_operation_lock("stuck-model")
            single = model_registry._models.get("stuck-model")
            assert not multi_lock.locked()
            if single is not None:
                assert not single._operation_lock.locked()

            # A peer apply of the same key converges the frontend.
            task = await registry._handle(
                ApplyLoad(
                    op_id="op-late",
                    serialised_model_settings=serialise_model_settings(
                        _sum_settings("stuck-model")
                    ),
                    ok=True,
                    generation=2,
                )
            )
            assert task is not None
            await asyncio.wait_for(task, timeout=5)
            model = await model_registry.get_model("stuck-model")
            assert model is not None
        finally:
            await registry.close()


# ------------------------------------------------------------------
# Overlapping origin operations (the full ACK sequence)
# ------------------------------------------------------------------


class TestOverlappingOriginOperations:
    """Two frontends originating same-key operations concurrently.

    The sequence that deadlocked when peer applies took the registry
    operation locks: frontend 0's origin load holds its locks across
    the supervisor round; frontend 1 concurrently originates an
    operation on the SAME key (its own locks held); the supervisor
    broadcasts frontend 0's apply to frontend 1, whose apply would
    wait on frontend 1's own suspended origin — stalling frontend 0's
    acknowledgement collection while both operations commit. Both
    origins must resolve within the operation timeout.
    """

    @staticmethod
    def _settings() -> Settings:
        return Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            # Generous for the healthy path (everything is in-process
            # and completes in milliseconds); a regression deadlocks
            # for this full window before the origin's own wait —
            # which starts earlier — expires.
            model_operation_timeout=2,
        )

    @staticmethod
    async def _build(supervisor, settings: Settings, gate: asyncio.Event):
        """Two real frontends (SharedPoolRegistry + MultiModelRegistry
        with the production hook wiring) and an in-memory supervisor:
        control messages route directly between the components, the
        pool is a stub whose load gates on both origins being in
        flight. The gate arms only for the overlap phase (call the
        returned ``arm`` closure): earlier operations must pass
        through ungated."""
        from types import SimpleNamespace

        from mlserver.registry import MultiModelRegistry

        shareds = {}
        pipes = []
        seen_op_requests: list[int] = []
        gate_state = {"armed": False, "baseline": 0}

        def arm() -> None:
            """Gate the next pool load on TWO in-flight OpRequests."""
            gate_state["armed"] = True
            gate_state["baseline"] = len(seen_op_requests)

        for fid in (0, 1):
            parent, child = Pipe(duplex=True)
            pipes.append((parent, child))
            shared = SharedPoolRegistry(
                settings,
                frontend_id=fid,
                reply_endpoint=f"ipc:///tmp/mlserver-test-ov-{fid}-{id(object()):x}",
                control=child,
            )
            # Production wiring shape: the registry's on-model-load
            # hook forwards to the shared pool (server.py's _pool_load
            # chain minus batching/handler decoration). The registry's
            # deferral window exceeds the pool's operation timeout so
            # an operation whose round is slow raises its timeout
            # error instead of racing the deferral's cancellation.
            model_registry = MultiModelRegistry(
                on_model_load=[shared.load_model],
                on_model_unload=[shared.unload_model],
                model_operation_timeout=settings.model_operation_timeout * 5,
            )
            shared.attach_registry(model_registry)
            await shared.start()

            async def _send(message, _fid=fid):
                if isinstance(message, OpRequest):
                    seen_op_requests.append(_fid)
                    if gate_state["armed"] and (
                        len(seen_op_requests) >= gate_state["baseline"] + 2
                    ):
                        gate.set()
                await supervisor._handle_control(_fid, message)

            # Route this frontend's control sends directly to the
            # in-memory supervisor (setattr: an instance-level method
            # override, matching the monkeypatch discipline mypy
            # enforces on direct assignment).
            setattr(shared, "_send", _send)
            shareds[fid] = shared

        async def pool_load_model(model):
            if gate_state["armed"]:
                # Hold the operation's pool load until both origins
                # are in flight — the discriminator of the deadlock:
                # both frontends' locks are held by suspended origins
                # when the first broadcast goes out.
                await gate.wait()
            return model

        supervisor._pool_registry = SimpleNamespace(
            should_load_model=lambda s: True,
            model_initialiser=lambda s: SimpleNamespace(settings=s),
            load_model=pool_load_model,
            unload_model=AsyncMock(return_value=None),
        )
        supervisor._frontends = {0: Mock(pid=101), 1: Mock(pid=102)}
        supervisor._controls = {0: Mock(), 1: Mock()}

        async def broadcast(message):
            for fid, sh in shareds.items():
                # The listener shape: the apply runs as a sequenced
                # background task; the broadcast does not wait for it.
                await sh._handle(message)
            return set(shareds)

        async def send_to(fid, message):
            await shareds[fid]._handle(message)

        supervisor._broadcast = broadcast
        supervisor._send_to = send_to
        return shareds, pipes, arm

    async def test_concurrent_same_key_loads_both_resolve(self):
        from mlserver.frontend import FrontendSupervisor

        settings = self._settings()
        supervisor = FrontendSupervisor(settings, [])
        gate = asyncio.Event()
        shareds, pipes, arm = await self._build(supervisor, settings, gate)
        try:
            # Gate the overlap: both origins must be in flight before
            # the first pool load proceeds.
            arm()
            load_settings = _sum_settings("overlap-model")
            origin_a = asyncio.create_task(
                shareds[0]._model_registry.load(load_settings)
            )
            origin_b = asyncio.create_task(
                shareds[1]._model_registry.load(load_settings)
            )

            results = await asyncio.wait_for(
                asyncio.gather(origin_a, origin_b, return_exceptions=True),
                timeout=15,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            assert not failures, f"overlapping origins failed: {failures!r}"

            # Both frontends converged on the committed state.
            for fid, shared in shareds.items():
                model = await shared._model_registry.get_model("overlap-model")
                assert model is not None, f"frontend {fid} missing the model"
                assert shared._key_generations[("overlap-model", "")] == 2
            # No operation was left waiting for an acknowledgement.
            assert not supervisor._pending_ops
        finally:
            for shared in shareds.values():
                await shared.close()
            for parent, child in pipes:
                parent.close()
                child.close()

    async def test_concurrent_load_and_unload_both_resolve(self):
        from mlserver.frontend import FrontendSupervisor

        settings = self._settings()
        supervisor = FrontendSupervisor(settings, [])
        gate = asyncio.Event()
        shareds, pipes, arm = await self._build(supervisor, settings, gate)
        try:
            load_settings = _sum_settings("overlap-model")
            # A settled first load: both frontends hold the model with
            # a committed generation, so the concurrent unload has a
            # real instance to retire. The gate stays disarmed — the
            # settled load must pass through ungated.
            await asyncio.wait_for(
                shareds[0]._model_registry.load(load_settings), timeout=10
            )
            model = await shareds[1]._model_registry.get_model("overlap-model")
            assert model is not None

            # Frontend 0 reloads while frontend 1 unloads — the same
            # key, overlapping, from opposite directions. Arm the gate
            # only now: both OpRequests must be in flight before the
            # reload's pool load proceeds.
            arm()
            origin_reload = asyncio.create_task(
                shareds[0]._model_registry.load(load_settings)
            )
            origin_unload = asyncio.create_task(
                shareds[1]._model_registry.unload("overlap-model")
            )

            results = await asyncio.wait_for(
                asyncio.gather(origin_reload, origin_unload, return_exceptions=True),
                timeout=15,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            assert not failures, f"overlapping load/unload failed: {failures!r}"
            assert not supervisor._pending_ops
        finally:
            for shared in shareds.values():
                await shared.close()
            for parent, child in pipes:
                parent.close()
                child.close()

    async def test_concurrent_unloads_both_resolve(self):
        """The unload-side ACK cycle: two frontends concurrently
        originating unloads of the SAME key. The first unload commits
        and advances the key's generation; the second resolves as
        superseded; displaced wrappers' retirements stay local. Both
        callers must see success — a committed removal reporting
        ModelUnloadError is the failure this test pins."""
        from mlserver.frontend import FrontendSupervisor

        settings = self._settings()
        supervisor = FrontendSupervisor(settings, [])
        gate = asyncio.Event()
        shareds, pipes, arm = await self._build(supervisor, settings, gate)
        try:
            load_settings = _sum_settings("overlap-model")
            await asyncio.wait_for(
                shareds[0]._model_registry.load(load_settings), timeout=10
            )
            assert await shareds[1]._model_registry.get_model("overlap-model")

            # Gate the POOL unload so both origin unloads are in
            # flight before the first commits.
            async def gated_pool_unload(instance: object) -> None:
                await gate.wait()

            supervisor._pool_registry.unload_model = gated_pool_unload
            arm()

            unload_a = asyncio.create_task(
                shareds[0]._model_registry.unload("overlap-model")
            )
            unload_b = asyncio.create_task(
                shareds[1]._model_registry.unload("overlap-model")
            )

            results = await asyncio.wait_for(
                asyncio.gather(unload_a, unload_b, return_exceptions=True),
                timeout=15,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            assert not failures, f"concurrent unloads failed: {failures!r}"

            for fid, shared in shareds.items():
                with pytest.raises(Exception):
                    await shared._model_registry.get_model("overlap-model")
            assert not supervisor._pending_ops
        finally:
            for shared in shareds.values():
                await shared.close()
            for parent, child in pipes:
                parent.close()
                child.close()

    async def test_delayed_reload_result_after_unload_does_not_resurrect(self):
        """The resurrection guard: a reload commits, an unload then
        removes the model pool-wide, and only then does the reload's
        delayed result reach its origin. The origin must NOT
        republish — a phantom registration would serve a model the
        pool no longer holds."""
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import OpResult

        settings = self._settings()
        supervisor = FrontendSupervisor(settings, [])
        gate = asyncio.Event()
        shareds, pipes, arm = await self._build(supervisor, settings, gate)
        try:
            load_settings = _sum_settings("overlap-model")
            await asyncio.wait_for(
                shareds[0]._model_registry.load(load_settings), timeout=10
            )
            assert await shareds[1]._model_registry.get_model("overlap-model")

            # Gate frontend 0's reload RESULT (generation 2) until the
            # unload has fully settled.
            release = asyncio.Event()
            result_arrived = asyncio.Event()
            inner_send_to = supervisor._send_to

            async def gated_send_to(fid: int, message: object) -> None:
                if (
                    fid == 0
                    and isinstance(message, OpResult)
                    and message.generation == 2
                ):
                    result_arrived.set()
                    await release.wait()
                await inner_send_to(fid, message)

            supervisor._send_to = gated_send_to

            reload_task = asyncio.create_task(
                shareds[0]._model_registry.load(load_settings)
            )
            # The reload commits pool-wide (its collector completed —
            # the result send is what the gate holds).
            await asyncio.wait_for(result_arrived.wait(), timeout=10)

            # The unload removes the model everywhere and settles.
            await asyncio.wait_for(
                shareds[1]._model_registry.unload("overlap-model"), timeout=10
            )
            with pytest.raises(Exception):
                await shareds[1]._model_registry.get_model("overlap-model")

            # NOW the delayed reload result reaches its origin: the
            # load must fail with the superseded-by-removal error, not
            # restore a ready model.
            release.set()
            with pytest.raises(RuntimeError, match="unloaded while"):
                await asyncio.wait_for(reload_task, timeout=10)
            with pytest.raises(Exception):
                await shareds[0]._model_registry.get_model("overlap-model")
            assert not supervisor._pending_ops
        finally:
            for shared in shareds.values():
                await shared.close()
            for parent, child in pipes:
                parent.close()
                child.close()


# ------------------------------------------------------------------
# Divergence reporting, resync, and removal convergence (round 8)
# ------------------------------------------------------------------


class TestDivergenceAndRemoval:
    """The round-8 fixes: the unload-side ACK cycle (generation
    advance + local-only retirements), the delayed-reload
    resurrection guard, the publish-time stale gate, and the
    failed-apply divergence report with supervisor resync."""

    async def test_stale_check_aborts_publication(self):
        """The publish-time gate: an apply whose generation was
        superseded while its model loaded must abort BEFORE
        publishing — the newer registration stands untouched."""
        from mlserver.registry import MultiModelRegistry, StalePublication

        registry = MultiModelRegistry()
        settings = _sum_settings("stale-model")
        first = await registry.load(settings)
        with pytest.raises(StalePublication):
            await registry._load_for_apply(settings, stale_check=lambda: True)
        # The pre-existing registration survived the aborted reload.
        model = await registry.get_model("stale-model")
        assert model is first

    async def test_retirement_of_superseded_wrapper_stays_local(
        self, shared_settings: Settings, pipes, sum_model_settings
    ):
        """A retirement (displaced-wrapper cleanup) whose generation is
        below the frontend's watermark stays LOCAL: the commit that
        displaced it already settled the pool state. Forwarding it is
        the unload-side ACK cycle — the retirement of a
        current-looking generation takes a live model offline."""
        from types import SimpleNamespace

        from mlserver.parallel.shared import _attach_generation, get_generation
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        registry = SharedPoolRegistry(
            shared_settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-retire-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry()
        registry.attach_registry(model_registry)
        await registry.start()
        try:
            settings = _sum_settings("retire-model")
            registered = await model_registry.load(settings)
            # Attach a pool generation like a committed apply would,
            # then advance the watermark past it (a newer commit).
            _attach_generation(registered, SimpleNamespace(generation=1))
            assert get_generation(registered) == 1
            registry._mark_key_generation("retire-model", None, 2)

            result = await registry.unload_model(registered)
            assert result is registered
            # Local-only: no OpRequest was forwarded to the supervisor.
            await asyncio.sleep(0.05)
            assert not parent.poll()
        finally:
            await registry.close()

    async def test_failed_peer_apply_reports_divergence_for_retirement(
        self,
    ) -> None:
        """A failed peer apply acks ok=False, and the collector
        retires the diverged LIVE frontend through the restart
        machinery — the same discipline as a startup-apply failure:
        control closed, process terminated, the death monitor owns
        the replacement."""
        from mlserver.frontend import FrontendSupervisor

        overlap = TestOverlappingOriginOperations()
        settings = overlap._settings()
        supervisor = FrontendSupervisor(settings, [])
        gate = asyncio.Event()
        shareds, pipes, arm = await overlap._build(supervisor, settings, gate)
        try:
            calls: list[str] = []

            async def flaky_load(
                load_settings: ModelSettings, stale_check: object = None
            ) -> object:
                calls.append(load_settings.name)
                raise RuntimeError("boom")

            mock_registry = Mock(
                spec=["load", "unload", "_load_for_apply", "_unload_for_apply"]
            )
            mock_registry._load_for_apply = flaky_load
            mock_registry.load = AsyncMock()
            shareds[1].attach_registry(mock_registry)

            # The collector sees the live process of frontend 1: the
            # retirement must go through the restart machinery.
            terminated: list = []

            def fake_terminate(fid, proc, control=None):
                terminated.append(fid)

            setattr(supervisor, "_terminate_frontend_for_restart", fake_terminate)

            # A load from frontend 0 commits pool-wide; frontend 1's
            # apply fails and acks ok=False.
            load_settings = _sum_settings("divergence-model")
            origin = asyncio.create_task(shareds[0]._model_registry.load(load_settings))
            wrapped = await asyncio.wait_for(origin, timeout=10)
            assert wrapped is not None

            assert calls == ["divergence-model"], "the apply ran and failed"
            assert 1 in terminated, (
                "the diverged LIVE frontend was retired through the "
                "restart machinery"
            )
            assert not supervisor._pending_ops
        finally:
            for shared in shareds.values():
                await shared.close()
            for parent, child in pipes:
                parent.close()
                child.close()

    async def test_stale_publication_rolls_back_hook_side_effects(
        self,
    ) -> None:
        """The stale abort must roll back the unpublished instance's
        hook side effects: the load hooks already ran (in production:
        REST routes registered, batchers attached), and nothing else
        will ever match them for removal — custom-route matching is
        bound-method identity, so no registered instance is the
        abandoned one. Without the rollback the routes leak
        permanently and shadow the newer generation's routes."""
        from mlserver.registry import MultiModelRegistry, StalePublication

        added: list[int] = []
        removed: list[int] = []

        async def load_hook(model: MLModel) -> MLModel:
            added.append(id(model))
            return model

        async def unload_hook(model: MLModel) -> MLModel:
            removed.append(id(model))
            return model

        registry = MultiModelRegistry(
            on_model_load=[load_hook], on_model_unload=[unload_hook]
        )
        settings = _sum_settings("rollback-model")
        first = await registry.load(settings)
        assert len(added) == 1

        with pytest.raises(StalePublication):
            await registry._load_for_apply(settings, stale_check=lambda: True)

        # The aborted apply's hook chain ran on the new instance...
        assert len(added) == 2
        # ...and the rollback cleaned exactly that instance up.
        assert removed == [added[1]]
        # The surviving registration is untouched.
        assert added[0] == id(first)
        model = await registry.get_model("rollback-model")
        assert model is first


class TestDivergenceRecoveryRound9:
    """Round-9 fixes: the late-reconcile publication gate (a removal
    landing mid-load must abort the publication) and the resync's
    bound committed-state replay (settings + operation generation +
    pool identity, removals included)."""

    async def test_late_reconcile_aborts_when_removed_mid_load(self, pipes):
        """Reviewer probe: a late reconcile whose local apply spans a
        local REMOVAL must not publish. The early watermark check
        alone raced — the removal applied while the reconcile loaded;
        the publication-time gate is what aborts it."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=0.5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-late-rec-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry(
            on_model_load=[shared.load_model],
            model_operation_timeout=5,
        )
        shared.attach_registry(model_registry)
        await shared.start()
        # Gate the reconcile's registry apply: it enters, blocks, and
        # the REAL _load_for_apply (with the publication-time stale
        # gate) runs only after the release.
        apply_entered = asyncio.Event()
        release_apply = asyncio.Event()
        real_apply = model_registry._load_for_apply

        async def gated_apply(
            load_settings: ModelSettings, stale_check: object = None
        ) -> MLModel:
            apply_entered.set()
            await release_apply.wait()
            return await real_apply(load_settings, stale_check=stale_check)

        setattr(model_registry, "_load_for_apply", gated_apply)
        try:
            model_settings = ModelSettings(
                name="late-reconcile-model",
                implementation="tests.fixtures.SumModel",
            )
            origin = asyncio.create_task(model_registry.load(model_settings))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)

            # The broadcast's origin branch: retention + ack.
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id=request.op_id,
                    serialised_model_settings=serialise_model_settings(model_settings),
                    ok=True,
                    generation=1,
                ),
            )
            await _recv(parent)  # the origin's ApplyAck

            # The origin's wait expires; the retention survives.
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(origin, timeout=5)

            # The committed result arrives late -> the reconcile
            # enters its (gated) local apply.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=1))
            await asyncio.wait_for(apply_entered.wait(), timeout=5)

            # A removal applies locally WHILE the reconcile loads.
            await handle_and_settle(
                shared,
                ApplyUnload(
                    op_id="op-remove",
                    name="late-reconcile-model",
                    version=None,
                    ok=True,
                    generation=3,
                ),
            )

            # Release: the reconcile's publication must abort against
            # the removal — no resurrection.
            release_apply.set()
            await asyncio.wait_for(
                asyncio.gather(*list(shared._reconcile_tasks), return_exceptions=True),
                timeout=5,
            )
            with pytest.raises(Exception):
                await model_registry.get_model("late-reconcile-model")
        finally:
            await shared.close()


class TestObsoleteApplyReject:
    """Round-10 blocker: an obsolete message (a stale replay or a
    duplicate broadcast) must be rejected BEFORE any side effect —
    routing tags and the registry may only change under the
    generation check. A stale replay would otherwise redirect live
    inference to the old pool and restore the old batching flag
    while the newer wrapper stays registered."""

    async def test_stale_replay_leaves_routing_untouched(self, pipes) -> None:
        from mlserver.parallel.shared import ApplyLoad, serialise_model_settings
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-stale-reject-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry(on_model_load=[shared.load_model])
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            batched = ModelSettings(
                name="m",
                implementation="tests.fixtures.SumModel",
                max_batch_size=4,
                max_batch_time=0.1,
            )
            unbatched = ModelSettings(
                name="m", implementation="tests.fixtures.SumModel"
            )
            key = ("m", "")

            task = await shared._handle(
                ApplyLoad(
                    op_id="op-gen2",
                    serialised_model_settings=serialise_model_settings(batched),
                    ok=True,
                    env_hash="env-2",
                    generation=2,
                )
            )
            assert task is not None
            await asyncio.wait_for(task, timeout=5)
            assert shared._dispatcher._model_pools[key] == "env-2"
            assert key in shared._dispatcher._batched_models

            # The stale generation-1 replay (older generation, old
            # pool, old batching): rejected before any side effect.
            task = await shared._handle(
                ApplyLoad(
                    op_id="op-replay-stale",
                    serialised_model_settings=serialise_model_settings(unbatched),
                    ok=True,
                    env_hash="env-1",
                    generation=1,
                )
            )
            assert task is not None
            await asyncio.wait_for(task, timeout=5)

            assert shared._dispatcher._model_pools[key] == "env-2"
            assert key in shared._dispatcher._batched_models
            wrapper = await model_registry.get_model("m")
            assert getattr(wrapper, "__pool_generation__", None) == 2
        finally:
            await shared.close()


class TestRetirementIncarnationBinding:
    """Round 15: ACK collection is bound to the control connection
    captured at BROADCAST time — a diverged frontend that dies and is
    replaced before the collector finishes must not take its healthy
    replacement down, and the first retirement invalidates subsequent
    collectors' requests for the same connection."""

    @staticmethod
    def _supervisor() -> "tuple[FrontendSupervisor, ModelSettings]":
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.registry import ENV_HASH_ATTR

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        supervisor = FrontendSupervisor(settings, [])
        setattr(supervisor, "_broadcast", AsyncMock(return_value={0, 1}))
        setattr(supervisor, "_send_to", AsyncMock())
        pool_registry = Mock()
        model_settings = ModelSettings(
            name="m", implementation="tests.fixtures.SumModel"
        )
        instance = Mock(settings=model_settings)
        setattr(instance, ENV_HASH_ATTR, "env-a")
        pool_registry.model_initialiser = Mock(return_value=instance)
        pool_registry.load_model = AsyncMock(return_value=instance)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry
        supervisor._frontends = {0: Mock(pid=101), 1: Mock(pid=102)}
        supervisor._controls[0] = Mock()
        supervisor._controls[1] = Mock()
        return supervisor, model_settings

    async def test_replacement_not_retired_for_dead_divergence(
        self,
    ) -> None:
        """The reviewer's sequence: frontend 1 diverges, dies, and is
        replaced before the collector finishes. The collector must
        NOT retire the healthy replacement."""
        from mlserver.parallel.shared import ApplyAck

        supervisor, model_settings = self._supervisor()
        import mlserver.frontend as fe_module

        await supervisor._load_settings(model_settings)
        confirms: list = []
        real_process_exited = fe_module._process_exited
        fe_module._process_exited = lambda proc: False
        setattr(
            supervisor,
            "_confirm_frontend_exit",
            AsyncMock(side_effect=lambda fid, proc: confirms.append(fid)),
        )
        try:
            unload = asyncio.create_task(
                supervisor._unload_settings(
                    "m", None, origin=0, op_id="op-u", generation=1
                )
            )
            await asyncio.sleep(0.05)
            # Frontend 1 diverges... then dies and is replaced.
            await supervisor._handle_ack(1, ApplyAck(op_id="op-u", ok=False))
            supervisor._frontends[1] = Mock(pid=202)
            replacement_control = Mock()
            supervisor._controls[1] = replacement_control
            await supervisor._handle_ack(0, ApplyAck(op_id="op-u", ok=True))
            await asyncio.wait_for(unload, timeout=5)

            # The replacement survived: its control was not closed and
            # no exit confirmation was queued for it.
            replacement_control.close.assert_not_called()
            assert not confirms, "no termination was queued for the replacement"
        finally:
            fe_module._process_exited = real_process_exited

    async def test_first_retirement_invalidates_second_collector(
        self,
    ) -> None:
        """Two collectors retire the same diverged frontend: the
        first closes the connection and queues the exit confirmation;
        the second's request for the same connection is invalidated —
        exactly one confirmation, not two."""
        from mlserver.parallel.shared import ApplyAck

        supervisor, model_settings = self._supervisor()
        await supervisor._load_settings(model_settings)
        other_settings = model_settings.model_copy(deep=True)
        other_settings.name = "m2"
        await supervisor._load_settings(other_settings)
        control = supervisor._controls[1]
        import mlserver.frontend as fe_module

        confirms: list = []
        real_process_exited = fe_module._process_exited
        fe_module._process_exited = lambda proc: False
        setattr(
            supervisor,
            "_confirm_frontend_exit",
            AsyncMock(side_effect=lambda fid, proc: confirms.append(fid)),
        )
        try:
            first = asyncio.create_task(
                supervisor._unload_settings(
                    "m", None, origin=0, op_id="op-u1", generation=1
                )
            )
            second = asyncio.create_task(
                supervisor._unload_settings(
                    "m2", None, origin=0, op_id="op-u2", generation=1
                )
            )
            # Both broadcasts capture the same live connection before either
            # collector can retire it. Distinct keys keep both unloads current.
            await asyncio.sleep(0)
            assert supervisor._pending_ops["op-u1"].broadcast_controls[1] is control
            assert supervisor._pending_ops["op-u2"].broadcast_controls[1] is control
            await supervisor._handle_ack(1, ApplyAck(op_id="op-u1", ok=False))
            await supervisor._handle_ack(1, ApplyAck(op_id="op-u2", ok=False))
            await supervisor._handle_ack(0, ApplyAck(op_id="op-u1", ok=True))
            await asyncio.wait_for(first, timeout=5)
            await asyncio.gather(*list(supervisor._retire_tasks))
            assert len(confirms) == 1, "the first collector retired it once"

            await supervisor._handle_ack(0, ApplyAck(op_id="op-u2", ok=True))
            await asyncio.wait_for(second, timeout=5)
            await asyncio.gather(*list(supervisor._retire_tasks))

            # Frontend 1's slot still holds its (now terminated)
            # process: the second collector's retirement was
            # invalidated by the first closing the connection — no
            # double confirmation.
            assert (
                len(confirms) == 1
            ), f"the second collector's request was invalidated: {confirms}"
        finally:
            fe_module._process_exited = real_process_exited


class TestOriginPublicationGate:
    """The origin load's publication window: the post-wait checks run
    inside the load hook, but remaining hooks and model preparation
    still await before the registry publishes. A lock-free unload
    apply committing a removal in that window must abort the
    publication (the registry's existing stale gate, armed by the
    hook) instead of resurrecting the removed model."""

    async def test_removal_mid_late_hooks_aborts_publication(self, pipes) -> None:
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import MultiModelRegistry, StalePublication

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-pubgate-{id(object()):x}",
            control=child,
        )
        # A hook AFTER the shared-pool load hook blocks past the
        # origin's generation check — the publication window.
        release = asyncio.Event()
        hook_entered = asyncio.Event()

        async def blocking_hook(model: MLModel) -> MLModel:
            hook_entered.set()
            await release.wait()
            return model

        model_registry = MultiModelRegistry(
            on_model_load=[shared.load_model, blocking_hook],
        )
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            model_settings = _sum_settings("pubgate-model")
            origin = asyncio.create_task(model_registry.load(model_settings))

            request = await _recv(parent)
            assert isinstance(request, OpRequest)
            # The origin's own broadcast: retention + ack (the origin
            # branch skips the local apply).
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id=request.op_id,
                    serialised_model_settings=serialise_model_settings(model_settings),
                    ok=True,
                    generation=2,
                ),
            )
            await _recv(parent)  # the origin's ApplyAck
            # The committed result: the origin's checks pass (no
            # removal yet) and the load hook returns; the blocking
            # hook holds the publication window open.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=2))
            await asyncio.wait_for(hook_entered.wait(), timeout=5)

            # A removal commits in the window — the peer apply is
            # lock-free, so it lands while the origin holds the
            # registry locks.
            await handle_and_settle(
                shared,
                ApplyUnload(
                    op_id="op-remove",
                    name="pubgate-model",
                    version=None,
                    ok=True,
                    generation=3,
                ),
            )

            release.set()
            # The publication must abort through the stale gate (the
            # same rollback as any stale publication), not resurrect.
            with pytest.raises(StalePublication):
                await asyncio.wait_for(origin, timeout=5)
            with pytest.raises(Exception):
                await model_registry.get_model("pubgate-model")
        finally:
            await shared.close()

    async def test_origin_load_completes_without_removal(self, pipes) -> None:
        """Control: the gate stays silent when nothing supersedes."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-pubgate-ctl-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry(on_model_load=[shared.load_model])
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            model_settings = _sum_settings("pubgate-ctl")
            origin = asyncio.create_task(model_registry.load(model_settings))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id=request.op_id,
                    serialised_model_settings=serialise_model_settings(model_settings),
                    ok=True,
                    generation=2,
                ),
            )
            await _recv(parent)
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=2))
            wrapper = await asyncio.wait_for(origin, timeout=5)
            model = await model_registry.get_model("pubgate-ctl")
            assert model is wrapper
        finally:
            await shared.close()


async def test_cancelled_queued_apply_leaves_no_unawaited_coroutine(
    shared, sum_model_settings, caplog
):
    """Cancelling an apply queued behind a slower one must not leave
    its coroutine unawaited (the runtime warning): the sequencer
    takes a factory, creating the coroutine only when its turn
    begins."""
    import warnings as warnings_module

    from mlserver.parallel.shared import ApplyLoad, serialise_model_settings

    started = asyncio.Event()
    release = asyncio.Event()
    load_calls = 0

    async def tracked_load(settings, stale_check=None):
        # The first call is the slow apply the second queues behind.
        # A second call means the CANCELLED apply started its turn —
        # the exact regression this test pins.
        nonlocal load_calls
        load_calls += 1
        if load_calls > 1:
            raise AssertionError("the cancelled apply ran its load")
        started.set()
        await release.wait()
        return Mock()

    registry = Mock(spec=["load", "unload", "_load_for_apply", "_unload_for_apply"])
    registry._load_for_apply = tracked_load
    shared.attach_registry(registry)

    first = await shared._handle(
        ApplyLoad(
            op_id="op-first",
            serialised_model_settings=serialise_model_settings(sum_model_settings),
            ok=True,
        )
    )
    assert first is not None
    await asyncio.wait_for(_wait_state(started), timeout=5)

    # The second apply chains behind the first; cancel it while
    # queued. Its factory must never run.
    factory_calls = []
    original_sequence = shared._sequence_apply

    def tracking_sequence(key, apply_factory):
        factory_calls.append(apply_factory)
        return original_sequence(key, apply_factory)

    shared._sequence_apply = tracking_sequence
    try:
        second = await shared._handle(
            ApplyLoad(
                op_id="op-second",
                serialised_model_settings=serialise_model_settings(sum_model_settings),
                ok=True,
            )
        )
    finally:
        shared._sequence_apply = original_sequence
    assert second is not None
    second.cancel()
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        release.set()
        await asyncio.wait_for(first, timeout=5)
        await asyncio.sleep(0.05)

    unawaited = [w for w in caught if "was never awaited" in str(w.message)]
    assert not unawaited, f"unawaited coroutine warnings: {unawaited}"
    # Retrieve the cancelled task's outcome explicitly: correct
    # behaviour is the cancellation; a regression that ran the queued
    # apply would surface the load-side tripwire here instead.
    with pytest.raises(asyncio.CancelledError):
        await second
    # The queued apply never ran: the sequencer saw only the first
    # apply's factory, and the load-side tripwire never fired.
    assert len(factory_calls) == 1
    assert load_calls == 1


async def _wait_state(event: asyncio.Event, timeout: float = 5.0) -> None:
    await asyncio.wait_for(event.wait(), timeout=timeout)


class TestGenerationBoundBeforePublication:
    """The peer apply's wrapper must carry its generation BEFORE the
    registry publishes it: _load_for_apply returns only after the
    retirement cleanup of the displaced wrapper awaits, and an origin
    resuming inside that window reads the registered wrapper through
    the displacement guard — a wrapper with no generation passes the
    guard (registered_generation is None) and the origin publishes
    over the newer state."""

    async def test_origin_displacement_guard_sees_peer_generation(self, pipes) -> None:
        """The reviewer's interleaving: the peer apply publishes gen 3
        and blocks INSIDE the retirement cleanup (the displaced
        wrapper's unload hooks await); the origin's result arrives in
        that window; the displacement guard must see the gen-3
        wrapper's generation and abort — a wrapper published without
        its generation would pass the guard (registered_generation
        is None) and the origin would publish over the newer state."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            OpResult,
            serialise_model_settings,
            get_generation,
        )
        from mlserver.registry import MultiModelRegistry, StalePublication

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-genbind-{id(object()):x}",
            control=child,
        )
        # A blocking unload hook: the peer apply's publication retires
        # the previously-registered instance and PAUSES inside the
        # cleanup — the apply's _load_for_apply has not returned, so
        # the post-await _attach_generation has NOT run. The wrapper
        # is registered (visible) but only the hook-side binding can
        # carry the generation.
        retire_blocked = asyncio.Event()
        retire_release = asyncio.Event()

        async def blocking_unload(model: MLModel) -> MLModel:
            retire_blocked.set()
            await retire_release.wait()
            return model

        model_registry = MultiModelRegistry(
            on_model_load=[shared.load_model],
            on_model_unload=[blocking_unload],
        )
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            model_settings = _sum_settings("genbind-model")

            # Seed a gen-1 registration (the instance the peer apply
            # will displace): apply it through the peer path.
            seed = await shared._handle(
                ApplyLoad(
                    op_id="op-seed",
                    serialised_model_settings=serialise_model_settings(model_settings),
                    ok=True,
                    generation=1,
                )
            )
            assert seed is not None
            await asyncio.wait_for(seed, timeout=5)
            assert model_registry.find_registered("genbind-model", None) is not None
            await _recv(parent)  # the seed's ApplyAck

            # The origin's load: sends the OpRequest and suspends.
            origin = asyncio.create_task(model_registry.load(model_settings))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)

            # A PEER apply of generation 3: publishes the gen-3
            # wrapper (displacing gen-1) then blocks in the
            # retirement cleanup (the blocking_unload hook).
            apply_task = await shared._handle(
                ApplyLoad(
                    op_id="op-peer-gen3",
                    serialised_model_settings=serialise_model_settings(model_settings),
                    ok=True,
                    generation=3,
                )
            )
            assert apply_task is not None
            # Wait until the retirement is blocked (the wrapper is
            # registered, the apply has not returned).
            await asyncio.wait_for(retire_blocked.wait(), timeout=5)
            registered = model_registry.find_registered("genbind-model", None)
            assert registered is not None, "the peer apply published"
            assert get_generation(registered) == 3, (
                "the generation is bound BEFORE publication (the fix): "
                f"got {get_generation(registered)}"
            )

            # THE CRITICAL ASSERTION: while the retirement is still
            # blocked (the apply's _load_for_apply has not returned,
            # the post-await _attach_generation has NOT run), the
            # registered wrapper ALREADY carries its generation —
            # the hook bound it before publication. Without the fix,
            # get_generation would be None here and the origin's
            # displacement guard would pass (registered_generation
            # is None), publishing gen 2 over gen 3.
            # (Asserted above: get_generation(registered) == 3.)

            # Release the retirement BEFORE the origin resumes: the
            # origin's _publish_model awaits the displaced wrapper's
            # cleanup (the same blocked task).
            retire_release.set()
            await asyncio.wait_for(apply_task, timeout=5)

            # The origin's result arrives: the guard sees the gen-3
            # wrapper (its generation was visible throughout the
            # window) and converges — never publishing gen 2.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=2))
            try:
                result = await asyncio.wait_for(origin, timeout=5)
                assert get_generation(result) == 3, (
                    "converged to the newer generation: " f"{get_generation(result)}"
                )
            except StalePublication:
                pass  # equally correct: the gate aborted the stale load

            final = model_registry.find_registered("genbind-model", None)
            assert final is not None
            assert (
                get_generation(final) == 3
            ), "the newer generation survived the origin's resume"
        finally:
            retire_release.set()
            await shared.close()


class TestUnversionedLoadSpareVersionedSibling:
    """Rejecting a stale unversioned load must not unload a healthy
    versioned sibling: the default-version fallback (get_model) would
    hand the v1 wrapper to the origin's swap, and the stale rollback
    would treat that live sibling as the aborted load's unpublished
    instance — unregistering it and emitting a real unload for its
    pool generation. The swap and the convergence must use EXACT-key
    lookup, and only a ready, strictly newer generation converges."""

    async def test_stale_unversioned_load_spares_versioned_sibling(self, pipes) -> None:
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
            get_generation,
        )
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-sibling-{id(object()):x}",
            control=child,
        )
        model_registry = MultiModelRegistry(on_model_load=[shared.load_model])
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            # A versioned sibling (m/v1) at generation 1.
            from mlserver.settings import ModelParameters

            v1_settings = _sum_settings("sibling-model")
            v1_settings.parameters = ModelParameters(version="v1")
            v1_apply = await shared._handle(
                ApplyLoad(
                    op_id="op-v1",
                    serialised_model_settings=serialise_model_settings(v1_settings),
                    ok=True,
                    generation=3,
                )
            )
            assert v1_apply is not None
            await asyncio.wait_for(v1_apply, timeout=5)
            await _recv(parent)  # v1's ApplyAck
            v1_wrapper = model_registry.find_registered("sibling-model", "v1")
            assert v1_wrapper is not None
            assert get_generation(v1_wrapper) == 3

            # An unversioned instance (m/) at generation 2.
            unversioned_settings = _sum_settings("sibling-model")
            unv_apply = await shared._handle(
                ApplyLoad(
                    op_id="op-unv",
                    serialised_model_settings=serialise_model_settings(
                        unversioned_settings
                    ),
                    ok=True,
                    generation=2,
                )
            )
            assert unv_apply is not None
            await asyncio.wait_for(unv_apply, timeout=5)
            await _recv(parent)  # unversioned's ApplyAck

            # The origin's UNVERSIONED load: suspends waiting for its
            # result.
            origin = asyncio.create_task(model_registry.load(unversioned_settings))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)

            # A peer removes the UNVERSIONED key (generation 3).
            removal = await shared._handle(
                ApplyUnload(
                    op_id="op-remove-unv",
                    name="sibling-model",
                    version=None,
                    ok=True,
                    generation=3,
                )
            )
            assert removal is not None
            await asyncio.wait_for(removal, timeout=5)
            await _recv(parent)  # removal's ApplyAck

            # The origin's result arrives: the removal won. The
            # origin must fail (the truth) and MUST NOT touch v1.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=2))
            with pytest.raises(Exception):
                await asyncio.wait_for(origin, timeout=5)

            # The v1 sibling survives: still registered, still ready,
            # and NO unload request was forwarded for it.
            v1_after = model_registry.find_registered("sibling-model", "v1")
            assert v1_after is not None, "the versioned sibling survives"
            assert v1_after is v1_wrapper
            assert getattr(v1_after, "ready", True), "the sibling stays ready"
            # No unload OpRequest was sent for v1: drain the pipe and
            # verify only the origin's OpRequest was ever sent.
            # (The pipe should be empty of unload requests: the only
            # message we haven't consumed is nothing — all acks were
            # consumed above.)
            await asyncio.sleep(0.1)
            assert not parent.poll(), (
                f"no unload request for the sibling: "
                f"{parent.recv() if parent.poll() else 'pipe empty'}"
            )
        finally:
            await shared.close()


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Poll a white-box condition until it holds (control-plane state
    lands inside sequenced background tasks)."""
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(0)


class TestReconcileRemovalWatermark:
    """The reconcile's load-side removal clear must only erase marks
    STRICTLY OLDER than the generation it published: a mark reserved
    by a NEWER unload while the reconcile was paused mid-retirement is
    live state. Erasing it lets a delayed older origin publish through
    the resurrection guard and re-register a model the pool removed
    at that newer generation."""

    def _build(self, pipes, blocking_generation: int):
        from mlserver.parallel.shared import SharedPoolRegistry
        from mlserver.registry import MultiModelRegistry

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-rrw-{id(object()):x}",
            control=child,
        )

        # The delayed origin's publication window: the load hook
        # chain holds the gen-4 wrapper (identified by its attached
        # generation) before the registry's gate.
        window_entered = asyncio.Event()
        window_release = asyncio.Event()

        async def origin_window_hook(model: MLModel) -> MLModel:
            if getattr(model, "__pool_generation__", None) == 4:
                window_entered.set()
                await window_release.wait()
            return model

        # The retirement pause: the blocked generation's cleanup
        # holds the reconcile (and the gen-5 unload's drain) open.
        unload_entered = asyncio.Event()
        unload_release = asyncio.Event()

        async def blocking_unload_hook(model: MLModel) -> MLModel:
            if getattr(model, "__pool_generation__", None) == blocking_generation:
                unload_entered.set()
                await unload_release.wait()
            return model

        model_registry = MultiModelRegistry(
            on_model_load=[shared.load_model, origin_window_hook],
            on_model_unload=[blocking_unload_hook],
        )
        shared.attach_registry(model_registry)
        return (
            shared,
            model_registry,
            {
                "window_entered": window_entered,
                "window_release": window_release,
                "unload_entered": unload_entered,
                "unload_release": unload_release,
            },
        )

    async def _stage(self, shared, model_registry, parent, name: str):
        from mlserver.parallel.shared import (
            ApplyLoad,
            serialise_model_settings,
        )
        from mlserver.settings import ModelParameters

        sibling = ModelSettings(
            name=name,
            implementation="tests.fixtures.SumModel",
            parameters=ModelParameters(version="v1"),
            max_batch_size=1,
        )
        plain = _sum_settings(name)

        # A versioned sibling keeps the per-name registry attached
        # through the unversioned key's removal.
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-sibling",
                serialised_model_settings=serialise_model_settings(sibling),
                ok=True,
                generation=1,
            ),
        )
        await _recv(parent)  # op-sibling ApplyAck
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-g1",
                serialised_model_settings=serialise_model_settings(plain),
                ok=True,
                generation=1,
            ),
        )
        await _recv(parent)  # op-g1 ApplyAck
        assert model_registry.find_registered(name, None) is not None
        return plain

    async def test_reconcile_does_not_erase_newer_removal_mark(self, pipes) -> None:
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import StalePublication

        shared, model_registry, events = self._build(pipes, blocking_generation=1)
        parent = pipes[0]
        await shared.start()
        try:
            name = "resurrect-model"
            plain = await self._stage(shared, model_registry, parent, name)

            # The delayed gen-4 origin: its request is out, its hook
            # waits for the supervisor's result.
            origin = asyncio.create_task(model_registry.load(plain))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)

            # The gen-3 reconcile (a committed load whose origin
            # stopped waiting) publishes gen 3 and pauses retiring
            # the gen-1 wrapper.
            shared._origin_applies["op-rec"] = ApplyLoad(
                op_id="op-rec",
                serialised_model_settings=serialise_model_settings(plain),
                ok=True,
                generation=3,
            )
            await shared._handle(OpResult(op_id="op-rec", ok=True, generation=3))
            await asyncio.wait_for(events["unload_entered"].wait(), timeout=5)

            # The delayed origin's result arrives: the post-wait
            # checks pass (nothing newer than 4 yet), the wrapper is
            # armed, and the publication window holds it.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=4))
            await asyncio.wait_for(events["window_entered"].wait(), timeout=5)

            # The gen-5 unload reserves its removal mark and stalls
            # draining the blocked gen-1 cleanup.
            unload_task = await shared._handle(
                ApplyUnload(
                    op_id="op-rm", name=name, version=None, ok=True, generation=5
                )
            )
            await _wait_until(lambda: shared._key_removals.get((name, "")) == 5)

            # Release the retirement: the reconcile finishes and runs
            # its load-side removal clear; the unload drains, retires
            # gen 3 and acks.
            events["unload_release"].set()
            ack = await _recv(parent)
            assert isinstance(ack, ApplyAck) and ack.op_id == "op-rm" and ack.ok
            if unload_task is not None:
                await unload_task

            # The watermark survived the reconcile's clear.
            assert shared._key_removals.get((name, "")) == 5

            # Release the delayed origin: the gate must abort it
            # against the surviving removal — no resurrection.
            events["window_release"].set()
            with pytest.raises(StalePublication):
                await asyncio.wait_for(origin, timeout=5)
            assert model_registry.find_registered(name, None) is None
            assert model_registry.find_registered(name, "v1") is not None
        finally:
            await shared.close()

    async def test_stale_load_does_not_return_older_wrapper(self, pipes) -> None:
        """The stale-publication convergence must report success only
        for a wrapper STRICTLY NEWER than the attempted load: a ready
        older wrapper still registered while its removal drains is
        not this load's success (the caller would hold a model the
        pool has already removed)."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import StalePublication

        shared, model_registry, events = self._build(pipes, blocking_generation=1)
        parent = pipes[0]
        await shared.start()
        try:
            name = "stale-conv-model"
            plain = await self._stage(shared, model_registry, parent, name)

            origin = asyncio.create_task(model_registry.load(plain))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)

            # The gen-3 reconcile publishes gen 3 and pauses retiring
            # the gen-1 wrapper.
            shared._origin_applies["op-rec"] = ApplyLoad(
                op_id="op-rec",
                serialised_model_settings=serialise_model_settings(plain),
                ok=True,
                generation=3,
            )
            await shared._handle(OpResult(op_id="op-rec", ok=True, generation=3))
            await asyncio.wait_for(events["unload_entered"].wait(), timeout=5)

            # The delayed gen-4 origin's result arms and holds its
            # wrapper in the publication window.
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=4))
            await asyncio.wait_for(events["window_entered"].wait(), timeout=5)

            # The gen-5 unload reserves its mark and stalls draining
            # the blocked cleanup — gen 3 stays registered.
            unload_task = await shared._handle(
                ApplyUnload(
                    op_id="op-rm", name=name, version=None, ok=True, generation=5
                )
            )
            await _wait_until(lambda: shared._key_removals.get((name, "")) == 5)

            # Release the origin FIRST (the removal still draining,
            # gen 3 still registered): the gate aborts against the
            # removal and the convergence must REJECT the older gen-3
            # wrapper — the load fails with the truth instead of
            # reporting success with a model already due for removal.
            events["window_release"].set()
            with pytest.raises(StalePublication):
                await asyncio.wait_for(origin, timeout=5)
            registered = model_registry.find_registered(name, None)
            assert (
                registered is None
                or getattr(registered, "__pool_generation__", None) != 4
            ), "the aborted gen-4 load must not be registered"

            # Settle: the retirement releases, the reconcile finishes,
            # the unload retires gen 3 and acks.
            events["unload_release"].set()
            ack = await _recv(parent)
            assert isinstance(ack, ApplyAck) and ack.op_id == "op-rm" and ack.ok
            if unload_task is not None:
                await unload_task
            assert model_registry.find_registered(name, None) is None
            assert model_registry.find_registered(name, "v1") is not None
        finally:
            await shared.close()


class TestLiveRemovalInvalidatesConvergence:
    """The convergence check must re-evaluate the LIVE exact-key
    removal watermark after the rollback: a removal reserved while
    the rollback cleanup ran retires the winner even though the
    abort-time snapshot still endorsed it."""

    async def test_removal_during_rollback_invalidates_convergence(self, pipes) -> None:
        from mlserver.parallel.shared import (
            ApplyLoad,
            ApplyUnload,
            OpResult,
            serialise_model_settings,
        )
        from mlserver.registry import MultiModelRegistry, StalePublication

        parent, child = pipes
        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        shared = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint=f"ipc:///tmp/mlserver-test-liverm-{id(object()):x}",
            control=child,
        )

        window_entered = asyncio.Event()
        window_release = asyncio.Event()

        async def origin_window_hook(model: MLModel) -> MLModel:
            if getattr(model, "__pool_generation__", None) == 4:
                window_entered.set()
                await window_release.wait()
            return model

        rollback_entered = asyncio.Event()
        rollback_release = asyncio.Event()

        async def blocking_unload_hook(model: MLModel) -> MLModel:
            if getattr(model, "__pool_generation__", None) == 4:
                rollback_entered.set()
                await rollback_release.wait()
            return model

        model_registry = MultiModelRegistry(
            on_model_load=[shared.load_model, origin_window_hook],
            on_model_unload=[blocking_unload_hook],
        )
        shared.attach_registry(model_registry)
        await shared.start()
        try:
            from mlserver.settings import ModelParameters

            name = "live-removal-model"
            sibling = ModelSettings(
                name=name,
                implementation="tests.fixtures.SumModel",
                parameters=ModelParameters(version="v1"),
                max_batch_size=1,
            )
            plain = _sum_settings(name)

            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id="op-sibling",
                    serialised_model_settings=serialise_model_settings(sibling),
                    ok=True,
                    generation=1,
                ),
            )
            await _recv(parent)
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id="op-g3",
                    serialised_model_settings=serialise_model_settings(plain),
                    ok=True,
                    generation=3,
                ),
            )
            await _recv(parent)

            # The gen-4 origin holds its publication window (between
            # the post-wait checks and the registry's gate).
            origin = asyncio.create_task(model_registry.load(plain))
            request = await _recv(parent)
            assert isinstance(request, OpRequest)
            parent.send(OpResult(op_id=request.op_id, ok=True, generation=4))
            await asyncio.wait_for(window_entered.wait(), timeout=5)

            # Generation 5 publishes fully while the origin waits in
            # its window.
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id="op-g5",
                    serialised_model_settings=serialise_model_settings(plain),
                    ok=True,
                    generation=5,
                ),
            )
            await _recv(parent)

            # Release the window: the gate aborts the gen-4 load
            # (superseded by the gen-5 publication) and the rollback
            # cleanup blocks.
            window_release.set()
            await asyncio.wait_for(rollback_entered.wait(), timeout=5)

            # The gen-6 removal is reserved while the rollback runs;
            # its registry unload stalls draining the blocked
            # rollback cleanup, so gen 5 stays registered.
            unload_task = await shared._handle(
                ApplyUnload(
                    op_id="op-rm", name=name, version=None, ok=True, generation=6
                )
            )
            await _wait_until(lambda: shared._key_removals.get((name, "")) == 6)

            # The rollback completes: the convergence must reject the
            # gen-5 winner against the LIVE watermark (the abort-time
            # snapshot carried no removal).
            rollback_release.set()
            with pytest.raises(StalePublication):
                await asyncio.wait_for(origin, timeout=5)

            # Settle: the drain completes, the unload retires gen 5
            # (its cleanup passes the gen-4-keyed hook).
            ack = await _recv(parent)
            assert isinstance(ack, ApplyAck) and ack.op_id == "op-rm" and ack.ok
            if unload_task is not None:
                await unload_task
            assert model_registry.find_registered(name, None) is None
            assert model_registry.find_registered(name, "v1") is not None
        finally:
            await shared.close()
