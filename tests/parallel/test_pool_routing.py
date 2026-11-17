from tests.utils import handle_and_settle

"""Pool-partitioned routing across shared inference pools.

Pool identity and displaced-instance retirement, quarantine routing
pauses, sticky batch affinity for worker-side batchers, and
per-version routing isolation.
"""

import threading

from multiprocessing import Pipe
from unittest.mock import AsyncMock, Mock

import pytest

from mlserver.settings import Settings, ModelSettings


class TestPoolPartitionRouting:
    """Worker and model pool tagging through the shared control plane."""

    @pytest.fixture
    async def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint="ipc:///tmp/test-reply",
            control=child,
        )
        registry.attach_registry(AsyncMock())
        registry.attach_data_plane(Mock())
        yield registry
        parent.close()
        child.close()

    async def test_worker_ready_event_tags_partition(self, shared):
        """A WorkerReadyEvent carrying an env hash routes its worker only
        within that pool's partition."""
        from mlserver.parallel.shared import WorkerReadyEvent

        await shared._handle_worker_ready(
            WorkerReadyEvent(
                pid=201, request_endpoint="ipc:///tmp/w201", env_hash="abc"
            )
        )

        assert shared._dispatcher._worker_pools[201] == "abc"
        assert shared._dispatcher._pool_pids.get("abc") == [201]
        assert 201 in shared._dispatcher._ready_workers

    async def test_worker_ready_event_default_pool(self, shared):
        """A WorkerReadyEvent without an env hash joins the default pool."""
        from mlserver.parallel.shared import WorkerReadyEvent

        await shared._handle_worker_ready(
            WorkerReadyEvent(pid=101, request_endpoint="ipc:///tmp/w101")
        )

        assert shared._dispatcher._worker_pools[101] is None
        assert shared._dispatcher._pool_pids.get(None) == [101]

    async def test_worker_stopped_removes_partition_entry(self, shared):
        """A stopped worker leaves its pool partition entirely."""
        from mlserver.parallel.shared import WorkerStoppedEvent, WorkerReadyEvent

        await shared._handle_worker_ready(
            WorkerReadyEvent(
                pid=201, request_endpoint="ipc:///tmp/w201", env_hash="abc"
            )
        )
        shared._handle_worker_stopped(
            WorkerStoppedEvent(pid=201, exit_code=0, env_hash="abc")
        )

        assert 201 not in shared._dispatcher._ready_workers
        assert 201 not in shared._dispatcher._worker_pools
        assert "abc" not in shared._dispatcher._pool_pids

    async def test_apply_load_tags_model_pool(self, shared):
        """An applied load tags the model with the loading pool's hash."""
        from mlserver.parallel.shared import ApplyLoad, serialise_model_settings

        model_settings = ModelSettings(
            name="sum-model", implementation="tests.fixtures.SumModel"
        )
        task = await shared._handle_apply_load(
            ApplyLoad(
                op_id="op-1",
                serialised_model_settings=serialise_model_settings(model_settings),
                ok=True,
                env_hash="abc",
            )
        )
        assert task is not None
        await task

        assert shared._dispatcher._model_pools[("sum-model", "")] == "abc"

    async def test_apply_load_failure_does_not_tag(self, shared):
        """A failed load never tags a routing partition for the model."""
        from mlserver.parallel.shared import ApplyLoad

        await shared._handle_apply_load(
            ApplyLoad(
                op_id="op-1",
                serialised_model_settings="{}",
                ok=False,
                error="boom",
            )
        )

        assert not shared._dispatcher._model_pools

    async def test_apply_unload_clears_model_pool(self, shared):
        """An applied unload resets the model name to the default pool."""
        from mlserver.parallel.shared import ApplyUnload

        shared._dispatcher.set_model_pool("sum-model", "abc")
        task = await shared._handle_apply_unload(
            ApplyUnload(op_id="op-2", name="sum-model", version=None, ok=True)
        )
        assert task is not None
        await task

        assert ("sum-model", "") not in shared._dispatcher._model_pools


class TestSupervisorPoolIdentity:
    """Supervisor broadcasts carry the pool identity of workers and models."""

    @pytest.fixture
    def supervisor(self):
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        return FrontendSupervisor(settings, [])

    async def test_worker_ready_broadcast_carries_env_hash(self, supervisor):
        """The worker-ready broadcast includes the worker's pool tag."""
        from mlserver.parallel.shared import WorkerReadyEvent

        supervisor._broadcast = AsyncMock()
        worker = Mock()
        worker.pid = 1234
        worker._zmq_request_endpoint = "ipc:///tmp/w1234"
        worker.env_hash = "abc"

        await supervisor._broadcast_worker_ready(worker)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, WorkerReadyEvent)
        assert event.env_hash == "abc"

    async def test_worker_stopped_broadcast_carries_env_hash(self, supervisor):
        """The worker-stopped broadcast includes the worker's pool tag."""
        from mlserver.parallel.shared import WorkerStoppedEvent

        supervisor._broadcast = AsyncMock()
        worker = Mock()
        worker.pid = 1234
        worker.exitcode = -9
        worker.env_hash = "abc"

        await supervisor._broadcast_worker_stopped(worker)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, WorkerStoppedEvent)
        assert event.env_hash == "abc"

    async def test_load_settings_broadcasts_env_hash(self, supervisor):
        """The apply-load broadcast carries the pool that committed the load."""
        from mlserver.parallel.registry import ENV_HASH_ATTR
        from mlserver.parallel.shared import ApplyLoad

        instance = Mock()
        setattr(instance, ENV_HASH_ATTR, "deadbeef")
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.model_initialiser.return_value = instance
        supervisor._pool_registry.load_model = AsyncMock()
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="pool-model", implementation="tests.fixtures.SumModel"
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert event.ok
        assert event.env_hash == "deadbeef"

    async def test_load_settings_failure_broadcasts_no_env_hash(self, supervisor):
        """A failed load broadcasts ok=False without a pool tag."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor._pool_registry = Mock()
        supervisor._pool_registry.model_initialiser.side_effect = RuntimeError("boom")
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="pool-model", implementation="tests.fixtures.SumModel"
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert not event.ok
        assert event.env_hash is None


class TestPoolQuarantineRouting:
    """A quarantined pool's workers stop routing on every frontend;
    workers stay registered (in-flight responses may still arrive) and
    recovery re-enables routing through the normal WorkerReadyEvents."""

    @pytest.fixture
    async def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint="ipc:///tmp/test-quarantine",
            control=child,
        )
        registry.attach_registry(AsyncMock())
        registry.attach_data_plane(Mock())
        yield registry
        parent.close()
        child.close()

    @staticmethod
    async def _ready(shared, pid: int, env_hash: str | None = None) -> None:
        from mlserver.parallel.shared import WorkerReadyEvent

        await shared._handle_worker_ready(
            WorkerReadyEvent(
                pid=pid,
                request_endpoint=f"ipc:///tmp/w{pid}",
                env_hash=env_hash,
            )
        )

    async def test_pool_quarantined_drops_pool_routing(self, shared):
        """The event removes the pool's workers from routing; the other
        pools keep serving."""
        from mlserver.parallel.errors import NoWorkersAvailable
        from mlserver.parallel.shared import PoolQuarantinedEvent

        await self._ready(shared, 101)
        await self._ready(shared, 201, "abc")
        shared._dispatcher.set_model_pool("env-model", "abc")

        await handle_and_settle(shared, PoolQuarantinedEvent(env_hash="abc"))

        assert 201 not in shared._dispatcher._ready_workers
        assert shared._dispatcher._pool_pids.get("abc") is None
        assert shared._dispatcher._pool_pids.get(None) == [101]
        with pytest.raises(NoWorkersAvailable):
            shared._dispatcher._get_worker("env-model")
        _, pid = shared._dispatcher._get_worker("plain-model")
        assert pid == 101

    async def test_pool_quarantined_default_pool(self, shared):
        """env_hash=None quarantines the default pool's partition."""
        from mlserver.parallel.shared import PoolQuarantinedEvent

        await self._ready(shared, 101)
        await self._ready(shared, 201, "abc")

        await handle_and_settle(shared, PoolQuarantinedEvent(env_hash=None))

        assert 101 not in shared._dispatcher._ready_workers
        assert shared._dispatcher._pool_pids.get("abc") == [201]

    async def test_worker_ready_after_quarantine_restores_routing(self, shared):
        """Recovery re-announces workers via WorkerReadyEvents, which
        re-enable routing without an explicit un-quarantine message."""
        from mlserver.parallel.errors import NoWorkersAvailable
        from mlserver.parallel.shared import PoolQuarantinedEvent

        await self._ready(shared, 201, "abc")
        shared._dispatcher.set_model_pool("env-model", "abc")
        await handle_and_settle(shared, PoolQuarantinedEvent(env_hash="abc"))
        with pytest.raises(NoWorkersAvailable):
            shared._dispatcher._get_worker("env-model")

        await self._ready(shared, 201, "abc")

        _, pid = shared._dispatcher._get_worker("env-model")
        assert pid == 201

    async def test_inflight_state_preserved(self, shared):
        """Quarantine pauses routing only: the worker proxy stays
        registered so in-flight responses can still resolve."""
        from mlserver.parallel.shared import PoolQuarantinedEvent

        await self._ready(shared, 201, "abc")
        await handle_and_settle(shared, PoolQuarantinedEvent(env_hash="abc"))

        assert 201 in shared._dispatcher._workers
        assert 201 in shared._proxies

    async def test_begin_recovery_fires_quarantine_hook(
        self, inference_pool, sum_model_settings
    ) -> None:
        """The pool fires its quarantine hooks (with its env hash) as
        soon as recovery begins — before the recovery work itself."""
        import asyncio as asyncio_module

        from mlserver.parallel.messages import (
            ModelUpdateMessage,
            ModelUpdateType,
        )

        fired: list = []
        recovery_ran = threading.Event()

        async def hook(env_hash: str | None) -> None:
            fired.append(env_hash)

        inference_pool._on_pool_quarantined.append(hook)

        async def recover_update(dispatch, message):
            # Ordering check INSIDE the recovery: the quarantine hook
            # must have recorded the pool id before any recovery work
            # runs. A failure here reaches pytest through the awaited
            # recovery task below.
            assert fired == [
                inference_pool.env_hash
            ], "recovery ran before the quarantine hook fired"
            recovery_ran.set()

        # Isolate the notification from the real recovery work: only
        # the hook ordering is under test here.
        inference_pool._recover_update = AsyncMock(side_effect=recover_update)

        loop = asyncio_module.get_running_loop()
        failed = loop.create_future()
        failed.set_exception(RuntimeError("dispatch failed"))
        message = ModelUpdateMessage(
            update_type=ModelUpdateType.Load, model_settings=sum_model_settings
        )
        inference_pool._begin_recovery(failed, message)

        # Await the retained recovery task: a failed order check must
        # reach pytest, not die in an unobserved task.
        recovery_task = inference_pool._recovery_task
        assert recovery_task is not None
        await asyncio_module.wait_for(recovery_task, timeout=5)

        assert fired == [inference_pool.env_hash]
        assert recovery_ran.is_set()
        # Retrieve the settled future's exception so it is not
        # reported as never-retrieved.
        assert isinstance(failed.exception(), RuntimeError)


class TestSupervisorQuarantineBroadcast:
    """The supervisor broadcasts PoolQuarantinedEvent to all frontends."""

    async def test_broadcast_carries_env_hash(self):
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import PoolQuarantinedEvent

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        supervisor = FrontendSupervisor(settings, [])
        supervisor._broadcast = AsyncMock()

        await supervisor._broadcast_pool_quarantined("abc")

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, PoolQuarantinedEvent)
        assert event.env_hash == "abc"


class TestPoolRoutingIdentity:
    """Every pool has a distinct routing id: environment pools use
    their env hash, GID-only pools use their GID — never None, which
    would collapse them into the default pool's routing partition."""

    def test_routing_id_env_pool(self):
        from mlserver.env import Environment
        from mlserver.parallel.pool import InferencePool
        from mlserver.settings import Settings

        env = Environment(env_path="/tmp/x", env_hash="abc123", delete_env=False)
        pool = InferencePool(Settings(), env=env)
        assert pool.routing_id == "abc123"

    def test_routing_id_gid_pool(self):
        from mlserver.parallel.pool import InferencePool
        from mlserver.settings import Settings

        pool = InferencePool(Settings(), pool_gid="gpu-a")
        # A GID-only pool has no env: without a distinct identity its
        # workers and models would tag None (the default partition).
        assert pool.env_hash is None
        assert pool.routing_id == "gpu-a"

    def test_routing_id_default_pool(self):
        from mlserver.parallel.pool import InferencePool
        from mlserver.settings import Settings

        pool = InferencePool(Settings())
        assert pool.routing_id is None

    async def test_find_resolves_gid_pool_via_routing_id(self):
        """A model tagged with a GID pool's routing id resolves to that
        pool — the routing id is the registry's _pools key."""
        from mlserver.parallel.pool import InferencePool
        from mlserver.parallel.registry import (
            InferencePoolRegistry,
            _set_environment_hash,
        )
        from mlserver.settings import ModelParameters, ModelSettings, Settings

        registry = InferencePoolRegistry(Settings())
        gid_pool = InferencePool(Settings(), pool_gid="gpu-a")
        registry._pools["gpu-a"] = gid_pool
        try:
            model = Mock(
                settings=ModelSettings(
                    name="m",
                    implementation="tests.fixtures.SumModel",
                    parameters=ModelParameters(inference_pool_gid="gpu-a"),
                )
            )
            # What load_model now does for a GID-only pool.
            _set_environment_hash(model, gid_pool.routing_id)
            assert await registry._find(model) is gid_pool
        finally:
            await gid_pool.close()


class TestVersionKeyedAffinityIsolation:
    """Unloading m/v1 must not clear m/v2's sticky routing."""

    async def test_unload_v1_preserves_v2_affinity(self):
        from mlserver.parallel.dispatcher import Dispatcher

        dispatcher = Dispatcher({}, __import__("multiprocessing").Queue())
        from tests.parallel.test_dispatcher import _register_pool_worker

        await _register_pool_worker(dispatcher, 101)
        await _register_pool_worker(dispatcher, 102)
        dispatcher.set_model_batched("m", True, "v1")
        dispatcher.set_model_batched("m", True, "v2")

        # Unload v1: only v1's affinity is cleared
        dispatcher.set_model_batched("m", False, "v1")
        assert ("m", "v1") not in dispatcher._batched_models
        assert ("m", "v2") in dispatcher._batched_models

        # v2 still gets sticky routing after v1's affinity was cleared:
        # with rotation disabled, repeated selection pins to one pid.
        selected_v2 = {dispatcher._get_worker("m", "v2")[1] for _ in range(5)}
        assert (
            len(selected_v2) == 1
        ), f"v2 lost its sticky routing after v1's unload: {selected_v2}"
        # v1 itself now round-robins (its stickiness is really gone,
        # not just missing from the tracking set).
        selected_v1 = {dispatcher._get_worker("m", "v1")[1] for _ in range(5)}
        assert len(selected_v1) > 1


class TestBatchedFlagLifecycle:
    """The sticky-routing flag is set explicitly on every load — a
    reload that disables batching must clear the flag the previous
    generation set."""

    @pytest.fixture
    def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint="ipc:///tmp/test-batched-flag",
            control=child,
        )
        yield registry
        parent.close()
        child.close()

    @staticmethod
    def _batched():
        return ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
            max_batch_size=4,
            max_batch_time=0.005,
        )

    @staticmethod
    def _unbatched():
        return ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
            max_batch_size=1,
        )

    def test_reload_disabling_batching_clears_sticky_flag(self, shared):
        shared._track_batched(self._batched(), "v1")
        assert ("m", "v1") in shared._dispatcher._batched_models

        # Same-version reload with batching disabled: the previous
        # generation's flag must not survive.
        shared._track_batched(self._unbatched(), "v1")
        assert ("m", "v1") not in shared._dispatcher._batched_models

    def test_unbatched_first_load_never_sets_flag(self, shared):
        shared._track_batched(self._unbatched(), "v1")
        assert ("m", "v1") not in shared._dispatcher._batched_models

    def test_batched_reload_keeps_flag(self, shared):
        shared._track_batched(self._batched(), "v1")
        shared._track_batched(self._batched(), "v1")
        assert ("m", "v1") in shared._dispatcher._batched_models


class TestPoolChangeRetiresDisplacedInstance:
    """A reload into a different inference pool must unload the
    displaced instance from its original pool — not orphan it."""

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        return FrontendSupervisor(settings, [])

    async def test_reload_into_different_pool_unloads_old_instance(self, mocker):
        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        # The model CAN be loaded into the pool (does not opt out).
        pool_registry.should_load_model = Mock(return_value=True)
        instance_a = Mock(
            settings=ModelSettings(
                name="m", version="v1", implementation="tests.fixtures.SumModel"
            )
        )
        setattr(instance_a, ENV_HASH_ATTR, "pool-a")
        instance_b = Mock(
            settings=ModelSettings(
                name="m", version="v1", implementation="tests.fixtures.SumModel"
            )
        )
        setattr(instance_b, ENV_HASH_ATTR, "pool-b")
        pool_registry.model_initialiser = Mock(return_value=instance_b)
        pool_registry.load_model = AsyncMock(return_value=instance_b)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        # First load: into pool A
        supervisor._loaded_models[("m", "v1")] = instance_a
        supervisor._generations[("m", "v1")] = 1

        # Reload with a different GID (pool B); version lives in
        # parameters (the settings-level version is a property).
        settings_b = ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
            parameters={"version": "v1", "inference_pool_gid": "pool-b"},
        )
        ok, error, gen = await supervisor._load_settings(settings_b)
        assert ok

        # The displaced pool-A instance was unloaded before the new
        # reference overwrote the inventory.
        pool_registry.unload_model.assert_awaited_once_with(instance_a)
        # The new instance is the one tracked.
        assert supervisor._loaded_models[("m", "v1")] is instance_b

    async def test_concurrent_same_key_moves_are_serialised(self):
        """Two frontends concurrently moving the same model key to
        different pools must serialise: without the per-key lock both
        read the SAME displaced instance, and the second inventory
        write strands the first replacement in its pool without an
        unload (a later unload of the key can never reach it)."""
        import asyncio as asyncio_module

        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.should_load_model = Mock(return_value=True)

        def instance(pool):
            inst = Mock(
                settings=ModelSettings(
                    name="m", version="v1", implementation="tests.fixtures.SumModel"
                )
            )
            setattr(inst, ENV_HASH_ATTR, pool)
            return inst

        instance_a, instance_b, instance_c = (
            instance("pool-a"),
            instance("pool-b"),
            instance("pool-c"),
        )
        pool_registry.model_initialiser = Mock(side_effect=[instance_b, instance_c])

        first_load_started = asyncio_module.Event()
        first_load_released = asyncio_module.Event()

        async def load_model(model):
            if model is instance_b:
                first_load_started.set()
                # Hold the first operation mid-load so the second can
                # demonstrably contend for the same key.
                await first_load_released.wait()
            return model

        pool_registry.load_model = AsyncMock(side_effect=load_model)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        supervisor._loaded_models[("m", "v1")] = instance_a
        supervisor._generations[("m", "v1")] = 1

        def settings_for(pool):
            return ModelSettings(
                name="m",
                implementation="tests.fixtures.SumModel",
                parameters={"version": "v1", "inference_pool_gid": pool},
            )

        op1 = asyncio_module.create_task(
            supervisor._load_settings(settings_for("pool-b"))
        )
        await asyncio_module.wait_for(first_load_started.wait(), timeout=5)
        # The second operation contends for the same key while the
        # first is mid-load: it must wait for the key lock, not start
        # its own pool load against the stale displaced instance.
        op2 = asyncio_module.create_task(
            supervisor._load_settings(settings_for("pool-c"))
        )
        await asyncio_module.sleep(0.05)
        assert pool_registry.load_model.await_count == 1

        first_load_released.set()
        ok1, _, _ = await asyncio_module.wait_for(op1, timeout=5)
        ok2, _, _ = await asyncio_module.wait_for(op2, timeout=5)
        assert ok1 and ok2

        # Both displaced instances unloaded exactly once, in commit
        # order: A by op1, then op1's committed replacement B by op2 —
        # never A twice with B stranded in pool B.
        assert [c.args[0] for c in pool_registry.unload_model.await_args_list] == [
            instance_a,
            instance_b,
        ]
        assert supervisor._loaded_models[("m", "v1")] is instance_c

    async def test_displaced_unload_failure_log_excludes_exception_text(
        self, mocker, caplog
    ):
        """A failed displaced-instance unload must not echo the worker's
        exception text into supervisor logs (CWE-532): a storage-backed
        runtime's unload error can carry credentials. The log records
        the class name only, and the reload still succeeds."""
        import logging

        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.should_load_model = Mock(return_value=True)
        instance_a = Mock(
            settings=ModelSettings(
                name="m", version="v1", implementation="tests.fixtures.SumModel"
            )
        )
        setattr(instance_a, ENV_HASH_ATTR, "pool-a")
        instance_b = Mock(
            settings=ModelSettings(
                name="m", version="v1", implementation="tests.fixtures.SumModel"
            )
        )
        setattr(instance_b, ENV_HASH_ATTR, "pool-b")
        pool_registry.model_initialiser = Mock(return_value=instance_b)
        pool_registry.load_model = AsyncMock(return_value=instance_b)
        pool_registry.unload_model = AsyncMock(
            side_effect=RuntimeError("token=super-secret-credential")
        )
        supervisor._pool_registry = pool_registry

        supervisor._loaded_models[("m", "v1")] = instance_a
        supervisor._generations[("m", "v1")] = 1

        settings_b = ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
            parameters={"version": "v1", "inference_pool_gid": "pool-b"},
        )
        with caplog.at_level(logging.ERROR, logger="mlserver.frontend"):
            ok, error, gen = await supervisor._load_settings(settings_b)

        # The displacement failure is tolerated: the reload succeeds
        # and the new instance is tracked.
        assert ok
        assert supervisor._loaded_models[("m", "v1")] is instance_b
        # The ERROR record names the failure and the class, but never
        # the exception text or traceback.
        assert any(
            "Failed unloading displaced pool instance" in rec.message
            for rec in caplog.records
        )
        assert any("RuntimeError" in rec.message for rec in caplog.records)
        assert "super-secret-credential" not in caplog.text


class TestDefaultPoolDisplacement:
    """A reload from the default pool to a named GID pool must unload
    the displaced default-pool instance (the != check handles every
    pool transition, not just named-to-named)."""

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        return FrontendSupervisor(settings, [])

    async def test_default_to_gid_reload_unloads_displaced(self, mocker):
        """A model loaded into the default pool (hash=None), then
        reloaded with a GID, must unload the default-pool instance."""
        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.should_load_model = Mock(return_value=True)

        default_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        # Default pool: hash is None
        setattr(default_instance, ENV_HASH_ATTR, None)

        gid_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        setattr(gid_instance, ENV_HASH_ATTR, "pool-b")

        pool_registry.model_initialiser = Mock(return_value=gid_instance)
        pool_registry.load_model = AsyncMock(return_value=gid_instance)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        # First load: into the default pool
        supervisor._loaded_models[("m", "")] = default_instance
        supervisor._generations[("m", "")] = 1

        # Reload with a GID (pool-b)
        settings_b = ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
            parameters={"inference_pool_gid": "pool-b"},
        )
        ok, error, gen = await supervisor._load_settings(settings_b)
        assert ok

        # The default-pool instance was unloaded.
        pool_registry.unload_model.assert_awaited_once_with(default_instance)
        assert supervisor._loaded_models[("m", "")] is gid_instance

    async def test_same_pool_reload_commits_replacement(self, mocker):
        """A reload within the same pool must COMMIT the replacement:
        the pool's load only stages the new version, and the workers
        keep the old one serving until the retiring unload arrives.
        Skipping it let a reload return success while the old code
        still served."""
        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.should_load_model = Mock(return_value=True)

        old_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        setattr(old_instance, ENV_HASH_ATTR, None)
        new_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        setattr(new_instance, ENV_HASH_ATTR, None)

        pool_registry.model_initialiser = Mock(return_value=new_instance)
        pool_registry.load_model = AsyncMock(return_value=new_instance)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        supervisor._loaded_models[("m", "")] = old_instance
        supervisor._generations[("m", "")] = 1

        # Reload without a GID (stays in default pool)
        settings_b = ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
        )
        ok, error, gen = await supervisor._load_settings(settings_b)
        assert ok

        # The retiring unload of the displaced instance committed the
        # replacement on the workers.
        pool_registry.unload_model.assert_awaited_once_with(old_instance)
        assert supervisor._loaded_models[("m", "")] is new_instance

    async def test_same_pool_commit_failure_fails_the_reload(self, mocker):
        """A same-pool commit failure must fail the reload: reporting
        success while the old version still serves is the divergence
        the commit exists to prevent. The old instance stays tracked
        for the retry."""
        from mlserver.parallel.registry import ENV_HASH_ATTR

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.should_load_model = Mock(return_value=True)

        old_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        setattr(old_instance, ENV_HASH_ATTR, None)
        new_instance = Mock(
            settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
        )
        setattr(new_instance, ENV_HASH_ATTR, None)

        pool_registry.model_initialiser = Mock(return_value=new_instance)
        pool_registry.load_model = AsyncMock(return_value=new_instance)
        pool_registry.unload_model = AsyncMock(
            side_effect=RuntimeError("commit dispatch failed")
        )
        supervisor._pool_registry = pool_registry

        supervisor._loaded_models[("m", "")] = old_instance
        supervisor._generations[("m", "")] = 1

        settings_b = ModelSettings(
            name="m",
            implementation="tests.fixtures.SumModel",
        )
        ok, error, gen = await supervisor._load_settings(settings_b)

        assert not ok
        assert "commit dispatch failed" in error
        # The OLD instance stays tracked: the retry re-stages and
        # re-commits against it.
        assert supervisor._loaded_models[("m", "")] is old_instance
