"""Model operations on the shared control plane (SharedPoolRegistry).

Load, unload and reload semantics with generation tracking,
acknowledgement completion, origin timeout reconciliation,
version-scoped retirement, log sanitisation, and the guards for
models that opt out of parallel inference.
"""

from tests.utils import handle_and_settle

import asyncio
from multiprocessing import Pipe
from unittest.mock import AsyncMock, Mock

import pytest

from mlserver.settings import Settings, ModelSettings


class TestPerOperationApplying:
    """While a broadcast apply is in flight, the apply's own load wraps
    locally; a concurrent user load for a different model still
    forwards to the supervisor (registering it everywhere with a worker
    behind it). Both scoping properties are verified on the real
    routing path, through the control pipe."""

    @pytest.fixture
    async def shared(self):
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
            reply_endpoint=f"ipc:///tmp/mlserver-test-guards-{id(object()):x}",
            control=child,
        )
        registry._test_parent = parent
        registry.attach_registry(AsyncMock())
        registry.attach_data_plane(Mock())
        await registry.start()
        try:
            yield registry
        finally:
            await registry.close()
            parent.close()
            child.close()

    @staticmethod
    def _model(name: str):
        model = Mock()
        model.settings = ModelSettings(
            name=name, implementation="tests.fixtures.SumModel"
        )
        return model

    async def _recv_request(self, shared, timeout: float = 5.0):
        import asyncio as asyncio_module

        # Poll before reading: run_in_executor work cannot be cancelled,
        # so a plain recv would leave the executor thread blocked inside
        # parent.recv() past this helper's timeout (until the fixture
        # closes the pipe). Reading only when a message is pending keeps
        # the executor call immediate.
        loop = asyncio_module.get_running_loop()
        ready = await loop.run_in_executor(None, shared._test_parent.poll, timeout)
        if not ready:
            raise asyncio_module.TimeoutError(f"no control message within {timeout}s")
        return await loop.run_in_executor(None, shared._test_parent.recv)

    async def test_overlapping_user_load_forwards_during_apply(self, shared):
        """A user load for model B, arriving while model A's broadcast
        apply is mid-flight, forwards to the supervisor as an
        OpRequest — it must not silently register B on this frontend
        alone (no worker would have it)."""
        import asyncio as asyncio_module

        from mlserver.parallel.model import ParallelModel
        from mlserver.parallel.shared import ApplyLoad, OpRequest, OpResult

        # The apply's local registration blocks on an event, holding
        # the apply "in flight" while B's load arrives.
        release = asyncio_module.Event()
        # Signals entry into the blocked registry load: B's load must
        # not start before the apply holds _applying_load.
        load_entered = asyncio_module.Event()

        async def blocking_load(settings, stale_check=None):
            load_entered.set()
            await release.wait()

        shared._model_registry._load_for_apply = AsyncMock(side_effect=blocking_load)

        serialised = ModelSettings(
            name="a-model", implementation="tests.fixtures.SumModel"
        )
        from mlserver.parallel.shared import serialise_model_settings

        apply_a = asyncio_module.create_task(
            shared._handle(
                ApplyLoad(
                    op_id="op-a",
                    serialised_model_settings=serialise_model_settings(serialised),
                    ok=True,
                )
            )
        )
        # Deterministic: the apply is inside its registry load (and
        # holds the applying-load context) before B's load starts.
        await asyncio_module.wait_for(load_entered.wait(), timeout=5)

        load_b = None
        try:
            # The concurrent user load for B (a different task, like a
            # REST handler) must forward to the supervisor.
            load_b = asyncio_module.create_task(
                shared.load_model(self._model("b-model"))
            )
            request = await self._recv_request(shared)
            assert isinstance(request, OpRequest)
            assert request.kind == "load"
            assert "b-model" in (request.serialised_model_settings or "")

            # Complete B's operation like the supervisor would.
            shared._test_parent.send(OpResult(op_id=request.op_id, ok=True))
            wrapped = await asyncio_module.wait_for(load_b, timeout=5)
            assert isinstance(wrapped, ParallelModel)
        finally:
            # Every exit path drains the tasks this test created: the
            # apply is released; an unresolved load is cancelled.
            release.set()
            tasks = [apply_a]
            if load_b is not None:
                if not load_b.done():
                    load_b.cancel()
                tasks.append(load_b)
            await asyncio_module.gather(*tasks, return_exceptions=True)
        # The apply's own registration completed locally.
        shared._model_registry._load_for_apply.assert_awaited_once()

    async def test_replayed_generation_reaches_unload_request(self, shared):
        """End-to-end replay generation flow: a replayed ApplyLoad
        attaches its generation to the registered wrapper, and that
        wrapper's later unload sends the generation back to the
        supervisor (a replacement that dropped it would send 0 and its
        unloads would be skipped as superseded)."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyLoad,
            OpRequest,
            OpResult,
            get_generation,
            serialise_model_settings,
        )

        # A real wrapper the frontend registry hands back from the
        # replayed apply.
        registered = self._model("m")
        registered.name = "m"
        registered.version = None
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        # A replayed load (unicast, no op waiter on this frontend):
        # the non-origin apply path registers and attaches.
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-replay",
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                generation=1,
                env_hash=None,
            ),
        )
        assert get_generation(registered) == 1

        # The wrapper's unload carries the replayed generation. The
        # replay's own ApplyAck comes through the pipe first (the
        # replacement acknowledges the unicast replay).
        unload_task = asyncio_module.create_task(shared.unload_model(registered))
        from mlserver.parallel.shared import ApplyAck

        first = await self._recv_request(shared)
        assert isinstance(first, ApplyAck) and first.op_id == "op-replay"
        request = await self._recv_request(shared)
        assert isinstance(request, OpRequest)
        assert request.kind == "unload"
        assert request.generation == 1

        # Resolve the operation like the supervisor would.
        shared._test_parent.send(OpResult(op_id=request.op_id, ok=True))
        await asyncio_module.wait_for(unload_task, timeout=5)

    async def test_apply_load_does_not_forward(self, shared):
        """The apply's own registry load never sends an OpRequest (it
        would loop the operation back to the supervisor) — exercised
        through the real load hook, not a mock that bypasses it."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad, serialise_model_settings

        # A registry whose load callback goes through the shared
        # registry's load_model (the real hook chain), proving the
        # task-local bypass prevents the loop.
        from mlserver.registry import MultiModelRegistry

        real_registry = MultiModelRegistry(
            on_model_load=[shared.load_model],
            model_operation_timeout=5,
        )
        shared.attach_registry(real_registry)

        settings = ModelSettings(
            name="a-model", implementation="tests.fixtures.SumModel"
        )
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-a",
                serialised_model_settings=serialise_model_settings(settings),
                ok=True,
            ),
        )
        # The model registered locally through the bypass...
        assert await real_registry.get_model("a-model") is not None
        # The apply's acknowledgement is expected — but no OpRequest
        # (which would loop the operation back to the supervisor).
        message = await self._recv_request(shared)
        from mlserver.parallel.shared import ApplyAck

        assert isinstance(message, ApplyAck)
        await asyncio_module.sleep(0.01)
        assert not shared._test_parent.poll()


class TestVersionScopedUnload:
    """Peer unloads target the exact version, not all versions."""

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
            settings, frontend_id=0, reply_endpoint="ipc:///tmp/test", control=child
        )
        registry._test_parent = parent
        registry.attach_registry(Mock())
        yield registry
        parent.close()
        child.close()

    async def test_unload_targets_exact_version(self, shared):
        """The exact-version unload path is used, so an unversioned
        unload never falls through to a versioned default."""
        from mlserver.parallel.shared import ApplyUnload

        shared._model_registry._unload_for_apply = AsyncMock()
        await handle_and_settle(
            shared, ApplyUnload(op_id="op-u", name="model-x", version="v2", ok=True)
        )
        shared._model_registry._unload_for_apply.assert_awaited_once_with(
            "model-x", "v2", False
        )

    async def test_unload_failure_still_acks(self, shared, caplog):
        """A failed unload still sends an acknowledgement — without it
        the supervisor's ack collection would stall the origin — and
        the failure is reported at error level (a miss is the normal
        no-op and is never logged as an error)."""
        import asyncio as asyncio_module
        import logging as logging_module

        from mlserver.parallel.shared import ApplyAck, ApplyUnload

        shared._model_registry._unload_for_apply = AsyncMock(
            side_effect=RuntimeError("retained cleanup keeps failing")
        )
        caplog.set_level(logging_module.ERROR)
        # Should not raise (the exception is caught)
        await handle_and_settle(
            shared, ApplyUnload(op_id="op-u", name="ghost", version=None, ok=True)
        )
        # The failure is visible as an apply failure — not the old
        # misleading "no local match" debug message.
        assert "failed to apply locally" in caplog.text
        assert "no local match" not in caplog.text
        # The ack must actually reach the pipe.
        ack = await asyncio_module.wait_for(
            asyncio_module.get_running_loop().run_in_executor(
                None, shared._test_parent.recv
            ),
            5,
        )
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-u"

    async def test_unload_miss_is_not_logged_as_failure(self, shared, caplog):
        """A local miss (the model was never registered here) returns
        normally from unload_exact_version: no error is logged, the ack
        still goes out."""
        import asyncio as asyncio_module
        import logging as logging_module

        from mlserver.parallel.shared import ApplyAck, ApplyUnload

        shared._model_registry._unload_for_apply = AsyncMock(return_value=None)
        caplog.set_level(logging_module.ERROR)
        await handle_and_settle(
            shared,
            ApplyUnload(op_id="op-miss", name="never-seen", version=None, ok=True),
        )
        assert "failed to apply locally" not in caplog.text
        ack = await asyncio_module.wait_for(
            asyncio_module.get_running_loop().run_in_executor(
                None, shared._test_parent.recv
            ),
            5,
        )
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-miss"

    async def test_unversioned_unload_spares_versioned_default(self, shared):
        """An unversioned unload must not remove a versioned default.

        A peer that registered m/v2 (e.g. it missed the earlier
        unversioned apply) receives an unversioned unload: the exact
        path must be a no-op, never falling through to the versioned
        default via the non-exact unload_version."""
        from mlserver.parallel.shared import ApplyUnload

        # Mock both paths: the exact path must be used; the non-exact
        # path (which would remove the versioned default) must not.
        shared._model_registry._unload_for_apply = AsyncMock()
        shared._model_registry.unload_version = AsyncMock()

        await handle_and_settle(
            shared, ApplyUnload(op_id="op-u", name="model-x", version=None, ok=True)
        )

        shared._model_registry._unload_for_apply.assert_awaited_once_with(
            "model-x", None, False
        )
        shared._model_registry.unload_version.assert_not_awaited()


class TestModelLevelParallelWorkersZero:
    """Models that opt out of parallel inference keep main-process
    batching: the _pool_load hook (wired as the first on_model_load
    callback) attaches the adaptive batcher in-process when the pool
    returns the model unchanged."""

    @staticmethod
    def _server(mock_pool):
        from mlserver.server import MLServer

        settings = Settings(parallel_workers=1)
        return MLServer(settings, inference_pool_registry=mock_pool)

    async def test_pool_load_applies_batching_when_model_opts_out(
        self, prometheus_registry
    ):
        """Pool returns the model unchanged (opt-out) with batched
        settings: load_batching wraps it in-process."""
        from mlserver.parallel.shared import SharedPoolRegistry

        mock_pool = Mock(spec=SharedPoolRegistry)
        mock_pool.load_model = AsyncMock(side_effect=lambda model: model)
        server = self._server(mock_pool)
        pool_load = server._model_registry._on_model_load[0]

        from mlserver.batching.adaptive import AdaptiveBatcher
        from tests.fixtures import SumModel

        batched = SumModel(
            ModelSettings(
                name="batched-model",
                implementation="tests.fixtures.SumModel",
                max_batch_size=4,
                max_batch_time=0.05,
            )
        )
        result = await pool_load(batched)

        mock_pool.load_model.assert_awaited_once_with(batched)
        # load_batching attaches the adaptive batcher to the model for
        # in-process execution (the registered model IS the model, with
        # its predict decorated and the batcher reachable via
        # AdaptiveBatcher.get_batcher).
        assert result is batched
        assert AdaptiveBatcher.get_batcher(result) is not None

    async def test_pool_load_skips_batching_when_unbatched(self, prometheus_registry):
        """Opt-out with unbatched settings: the model passes through."""
        from mlserver.parallel.shared import SharedPoolRegistry

        mock_pool = Mock(spec=SharedPoolRegistry)
        mock_pool.load_model = AsyncMock(side_effect=lambda model: model)
        server = self._server(mock_pool)
        pool_load = server._model_registry._on_model_load[0]

        from mlserver.batching.adaptive import AdaptiveBatcher
        from tests.fixtures import SumModel

        unbatched = SumModel(
            ModelSettings(
                name="unbatched-model",
                implementation="tests.fixtures.SumModel",
                max_batch_size=0,
                max_batch_time=0,
            )
        )
        result = await pool_load(unbatched)

        assert result is unbatched
        assert AdaptiveBatcher.get_batcher(result) is None

    async def test_pool_load_returns_pool_wrapper_unchanged(self, prometheus_registry):
        """When the pool wraps the model (parallel inference) in the
        direct server (frontend_workers=0, the default), the wrapper gets
        the MAIN-process batcher — the previous behaviour. The raw model
        carries no batcher."""
        from mlserver.parallel.model import ParallelModel
        from mlserver.parallel.shared import SharedPoolRegistry

        mock_pool = Mock(spec=SharedPoolRegistry)
        settings = ModelSettings(
            name="pooled-model",
            implementation="tests.fixtures.SumModel",
            max_batch_size=4,
            max_batch_time=0.05,
        )
        from unittest.mock import MagicMock

        inner = MagicMock()
        inner.settings = settings
        parallel_model = ParallelModel(inner, MagicMock())
        mock_pool.load_model = AsyncMock(return_value=parallel_model)
        server = self._server(mock_pool)
        pool_load = server._model_registry._on_model_load[0]

        from tests.fixtures import SumModel

        model = SumModel(
            ModelSettings(
                name="pooled-model",
                implementation="tests.fixtures.SumModel",
                max_batch_size=4,
                max_batch_time=0.05,
            )
        )
        result = await pool_load(model)

        assert result is parallel_model
        # At frontend_workers=0 (the direct server) the batcher wraps
        # the parallel wrapper in the MAIN process (the previous behaviour).
        from mlserver.batching.adaptive import AdaptiveBatcher

        assert AdaptiveBatcher.get_batcher(parallel_model) is not None
        # The raw model carries no in-process batcher.
        assert AdaptiveBatcher.get_batcher(model) is None


class TestModelConstructorFailure:
    """Model construction errors are caught and returned to the origin."""

    async def test_constructor_failure_sets_error(self):
        """A model constructor exception sets ok=False and the error message."""
        from mlserver.frontend import FrontendSupervisor
        from mlserver.settings import Settings

        settings = Settings(
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        supervisor = FrontendSupervisor(settings)

        # Mock the pool registry to raise on model_initialiser
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.model_initialiser = Mock(
            side_effect=RuntimeError("constructor failed")
        )
        supervisor._loaded_models = {}
        supervisor._pending_ops = {}
        supervisor._broadcast = AsyncMock()
        supervisor._collect_acks = AsyncMock()

        model_settings = ModelSettings(
            name="bad-model", implementation="tests.fixtures.SumModel"
        )

        # Should not raise; should set ok=False
        await supervisor._load_settings(model_settings, origin=0, op_id="op-1")

        # Verify the broadcast was called with ok=False
        call_args = supervisor._broadcast.call_args[0][0]
        assert call_args.ok is False
        assert "constructor failed" in call_args.error

    async def test_load_failure_log_excludes_exception_text(self, mocker, caplog):
        """The supervisor's error-level log carries the exception class
        and op id, never the exception text (which can echo
        credential-bearing settings — CWE-532); the traceback goes to
        debug only. The origin's result still carries the reason."""
        import logging

        from mlserver.frontend import FrontendSupervisor
        from mlserver.settings import Settings

        settings = Settings(
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        supervisor = FrontendSupervisor(settings)

        secret = "s3-model-store-token-xyz"
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.model_initialiser = Mock(
            side_effect=RuntimeError(f"auth failed for {secret}")
        )
        supervisor._loaded_models = {}
        supervisor._pending_ops = {}
        supervisor._broadcast = AsyncMock()
        supervisor._collect_acks = AsyncMock()

        caplog.set_level(logging.ERROR)
        model_settings = ModelSettings(
            name="bad-model", implementation="tests.fixtures.SumModel"
        )
        await supervisor._load_settings(model_settings, origin=0, op_id="op-leak")

        logged = caplog.text
        # The failure is visible: model name, exception class, op id.
        assert "bad-model" in logged
        assert "RuntimeError" in logged
        assert "op-leak" in logged
        # The exception text (with the credential) is absent at error
        # level.
        assert secret not in logged
        # The origin-facing broadcast still carries the reason (the
        # single-process server surfaces it to the same caller).
        call_args = supervisor._broadcast.call_args[0][0]
        assert call_args.ok is False
        assert secret in call_args.error


class TestCancelRequestPipeRace:
    """cancel_request goes through the send executor to avoid pipe races."""

    def test_cancel_uses_send_executor(self):
        """cancel_request submits to _send_executor, not direct send."""
        from mlserver.parallel.shared import CancelRequest, SharedPoolRegistry

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings, frontend_id=0, reply_endpoint="ipc:///tmp/test", control=child
        )
        # Spy on the connection: a regression that writes directly to
        # the pipe (with or without the executor submit) must fail.
        control = Mock()
        registry._control = control
        registry._send_executor = Mock()

        registry.cancel_request("model", "req-1", 42)

        # No inline write to the pipe.
        control.send.assert_not_called()
        # The executor received the send callable and the message.
        registry._send_executor.submit.assert_called_once()
        send_callable, message = registry._send_executor.submit.call_args[0]
        assert send_callable is control.send
        assert isinstance(message, CancelRequest)
        assert message.model_name == "model"
        assert message.request_id == "req-1"
        assert message.worker_pid == 42
        parent.close()
        child.close()


class TestNotFoundUnloadSingleBroadcast:
    """An unload for a key the pool does not hold broadcasts exactly one
    ApplyUnload resolving ok — the desired post-unload state. A second
    duplicate with the same op_id would make every peer ack twice and
    attempt a spurious unload."""

    async def test_not_found_unload_broadcasts_once(self, mocker):
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import ApplyUnload

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
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        supervisor._pool_registry = Mock()

        await supervisor._unload_settings("ghost-model", None, origin=0, op_id="op-u")

        broadcasts = [
            c.args[0]
            for c in supervisor._broadcast.call_args_list
            if isinstance(c.args[0], ApplyUnload)
        ]
        assert (
            len(broadcasts) == 1
        ), f"not-found unload must broadcast exactly once, got {len(broadcasts)}"
        assert broadcasts[0].ok


class TestControlMessageLogSanitization:
    """Control-message failure logs must not echo the message payload:
    ApplyLoad carries the serialised settings, whose parameters.extra
    can hold model-store credentials, and exception text can echo its
    input (CWE-532)."""

    @pytest.fixture
    def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry
        from tests.utils import get_available_ports

        http_port = get_available_ports(1)[0]
        settings = Settings(
            http_port=http_port,
            parallel_workers=1,
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
        registry._test_parent = parent
        yield registry
        parent.close()
        child.close()

    @staticmethod
    def _secret_settings(secret: str) -> str:
        import json

        return json.dumps(
            {
                "name": "m",
                "implementation": "tests.fixtures.SumModel",
                "parameters": {"extra": {"model_store_token": secret}},
            }
        )

    async def test_apply_load_failure_log_excludes_payload(self, shared, caplog):
        """A failing ApplyLoad handling logs type and op id at error
        level; neither the serialised settings nor the exception text
        (which echoes the secret) may appear."""
        import logging

        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad

        caplog.set_level(logging.ERROR)
        secret = "super-secret-model-store-token"
        # The secret leaks via both vectors: the message payload and
        # the exception text.
        shared._handle = AsyncMock(
            side_effect=RuntimeError(f"deserialise failed near {secret}")
        )

        listener = asyncio_module.create_task(shared._listen())
        shared._test_parent.send(
            ApplyLoad(
                op_id="op-x",
                serialised_model_settings=self._secret_settings(secret),
                ok=True,
            )
        )
        await asyncio_module.sleep(0.2)
        listener.cancel()
        try:
            await listener
        except asyncio_module.CancelledError:
            pass

        logged = caplog.text
        # The failure is visible: type, op id, exception class.
        assert "ApplyLoad" in logged
        assert "op-x" in logged
        assert "RuntimeError" in logged
        # The payload and its echo are absent.
        assert secret not in logged

    async def test_unknown_message_warning_excludes_payload(self, shared, caplog):
        """An unknown control message logs its type, not its repr."""
        import logging

        import asyncio as asyncio_module

        caplog.set_level(logging.WARNING)

        listener = asyncio_module.create_task(shared._listen())
        shared._test_parent.send({"unexpected": "payload-with-token"})
        await asyncio_module.sleep(0.2)
        listener.cancel()
        try:
            await listener
        except asyncio_module.CancelledError:
            pass

        logged = caplog.text
        assert "dict" in logged
        assert "payload-with-token" not in logged


class TestUnversionedRetirementScoping:
    """A versionless unload must distinguish retirement from the public
    all-versions unload by the unversioned key's generation."""

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

    async def test_unversioned_retirement_is_a_noop(self, mocker):
        """An unversioned reload's retirement (version=None, stale
        generation) must not expand to every version: neither the
        replacement committed by that reload nor a sibling version
        (m/2) may be unloaded."""
        from mlserver.parallel.shared import OpResult

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        replacement = Mock()
        sibling = Mock()
        supervisor._loaded_models[("m", "")] = replacement
        supervisor._loaded_models[("m", "2")] = sibling
        supervisor._generations[("m", "")] = 2
        supervisor._generations[("m", "2")] = 1

        # The displaced unversioned wrapper (generation 1) retires
        # after the reload committed generation 2.
        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u", generation=1
        )

        pool_registry.unload_model.assert_not_called()
        supervisor._broadcast.assert_not_called()
        assert ("m", "") in supervisor._loaded_models
        assert ("m", "2") in supervisor._loaded_models
        result = supervisor._send_to.call_args[0][1]
        assert isinstance(result, OpResult) and result.ok

    async def test_unversioned_unload_targets_only_its_own_key(self, mocker):
        """An unversioned wrapper's unload targets exactly the
        unversioned key: a sibling version survives (a repository sync
        that replaces unversioned m with m/v2 cleans up only the stale
        unversioned instance — it must not remove the version it just
        loaded). The public name-wide unload enumerates wrappers at the
        frontend, one exact operation per version."""
        from mlserver.parallel.shared import ApplyUnload

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry
        unversioned = Mock()
        sibling = Mock()
        supervisor._loaded_models[("m", "")] = unversioned
        supervisor._loaded_models[("m", "2")] = sibling
        supervisor._generations[("m", "")] = 2
        supervisor._generations[("m", "2")] = 1

        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u", generation=2
        )

        unloaded = {c.args[0] for c in pool_registry.unload_model.await_args_list}
        assert unloaded == {unversioned}
        assert ("m", "") not in supervisor._loaded_models
        assert ("m", "2") in supervisor._loaded_models
        versions = [
            c.args[0].version
            for c in supervisor._broadcast.call_args_list
            if isinstance(c.args[0], ApplyUnload)
        ]
        assert versions == [""]

    async def test_stale_generation_after_reload_cycle_does_not_unload(self, mocker):
        """A retirement whose generation differs from the committed one
        in EITHER direction is a no-op: generation counters are now
        monotonic per key (never reset on unload), so a numerically
        higher stale generation should not occur — this guards the
        inequality check defensively."""
        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        fresh = Mock()
        supervisor._loaded_models[("m", "v1")] = fresh
        # A divergent current value (e.g. state rebuilt after a
        # supervisor restart): any mismatch means stale.
        supervisor._generations[("m", "v1")] = 1

        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=3
        )

        pool_registry.unload_model.assert_not_called()
        assert ("m", "v1") in supervisor._loaded_models

    async def test_unversioned_retirement_with_only_sibling_loaded(self, mocker):
        """An unversioned retirement arriving when the unversioned key
        is no longer tracked must not fall through to the all-versions
        expansion and unload a surviving sibling version."""
        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        sibling = Mock()
        supervisor._loaded_models[("m", "2")] = sibling
        supervisor._generations[("m", "2")] = 1
        # Unversioned key absent: generations.get(("", ...)) is 0, and
        # the stale generation 1 differs from 0 — retirement no-op.

        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u", generation=1
        )

        pool_registry.unload_model.assert_not_called()
        assert ("m", "2") in supervisor._loaded_models


class TestUnloadAckCompletion:
    """An unload operation resolves only after every live frontend
    acknowledged its broadcast; a frontend dying mid-operation must not
    hang the collection."""

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

    @staticmethod
    def _seed(supervisor, versions):
        for v in versions:
            supervisor._loaded_models[("m", v)] = Mock()
            supervisor._generations[("m", v)] = 1

    async def _run_unload(self, supervisor, version):
        from mlserver.parallel.shared import ApplyUnload

        task = asyncio.create_task(
            supervisor._unload_settings(
                "m", version, origin=0, op_id="op-u", generation=1
            )
        )
        # Let the unload reach its broadcast and the ack collection.
        for _ in range(50):
            if (
                supervisor._broadcast.await_count
                and all(
                    isinstance(c.args[0], ApplyUnload)
                    for c in supervisor._broadcast.call_args_list
                )
                and supervisor._pending_ops.get("op-u") is not None
            ):
                break
            await asyncio.sleep(0.001)
        return task

    async def test_unload_collects_ack_from_every_frontend(self, mocker):
        """The operation stays pending after only one frontend acked;
        the second frontend's ack completes it."""
        from mlserver.parallel.shared import ApplyAck

        supervisor = self._supervisor()
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.unload_model = AsyncMock()
        supervisor._broadcast = AsyncMock(
            # The ack collection waits on the broadcast's captured
            # recipient set.
            return_value={0, 1}
        )
        supervisor._send_to = AsyncMock()
        supervisor._frontends = {0: Mock(), 1: Mock()}
        self._seed(supervisor, ["v1"])

        task = await self._run_unload(supervisor, "v1")
        assert supervisor._broadcast.await_count == 1

        # Only frontend 0 acked: still pending.
        await supervisor._handle_ack(0, ApplyAck(op_id="op-u"))
        await asyncio.sleep(0.05)
        assert not task.done()
        assert supervisor._send_to.await_count == 0

        # Frontend 1's ack completes the operation.
        await supervisor._handle_ack(1, ApplyAck(op_id="op-u"))
        await asyncio.wait_for(task, timeout=1)
        assert supervisor._send_to.await_count == 1

    async def test_dead_frontend_satisfies_remaining_acks(self, mocker):
        """A frontend dying mid-operation with zero acks must not hang
        the collection: the exit path marks it complete and the
        surviving frontend's ack resolves the origin."""
        from mlserver.parallel.shared import ApplyAck

        supervisor = self._supervisor()
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.unload_model = AsyncMock()
        supervisor._broadcast = AsyncMock(
            # The ack collection waits on the broadcast's captured
            # recipient set.
            return_value={0, 1}
        )
        supervisor._send_to = AsyncMock()
        supervisor._frontends = {0: Mock(), 1: Mock()}
        supervisor._schedule_restart = Mock()
        self._seed(supervisor, ["v1"])

        task = await self._run_unload(supervisor, "v1")

        # The origin fully acks; frontend 1 has not acked.
        await supervisor._handle_ack(0, ApplyAck(op_id="op-u"))
        await asyncio.sleep(0.02)
        assert not task.done()

        # Frontend 1 dies with zero acks.
        supervisor._handle_dead_frontend(1, Mock(pid=111, exitcode=-9))

        # The collection completes via the dead-frontend marking.
        await asyncio.wait_for(task, timeout=1)
        assert supervisor._send_to.await_count == 1


class TestGenerationAwareReloadRetirement:
    """A same-version reload retires the old wrapper on the frontend;
    that retirement must not unload the pool's current generation."""

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

    @staticmethod
    def _model_settings(name: str) -> "ModelSettings":
        return ModelSettings(name=name, implementation="tests.fixtures.SumModel")

    async def test_superseded_generation_unload_is_a_noop(self, mocker):
        """An unload carrying an older generation than the committed
        one resolves ok without touching the pool or broadcasting."""
        from mlserver.parallel.shared import OpResult

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        instance = Mock()
        supervisor._loaded_models[("m", "v1")] = instance
        supervisor._generations[("m", "v1")] = 3

        # A retirement carries the wrapper's exact version.
        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=2
        )

        pool_registry.unload_model.assert_not_called()
        supervisor._broadcast.assert_not_called()
        # The origin's operation resolves ok without an ack collection.
        result = supervisor._send_to.call_args[0][1]
        assert isinstance(result, OpResult) and result.ok

    async def test_current_generation_unload_proceeds(self, mocker):
        """An unload carrying the current (or no) generation is the
        real thing: pool unload, inventory and generation retired,
        broadcast sent."""
        from mlserver.parallel.shared import ApplyUnload

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry
        instance = Mock()
        supervisor._loaded_models[("m", "")] = instance
        supervisor._generations[("m", "")] = 3

        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u", generation=3
        )

        pool_registry.unload_model.assert_awaited_once_with(instance)
        assert ("m", "") not in supervisor._loaded_models
        # The generation counter ADVANCES past the unload (the removal
        # commits a value above every earlier one): retirements and
        # duplicate unloads of the removed wrappers resolve as
        # superseded instead of looking current.
        assert supervisor._generations[("m", "")] == 4
        broadcast = supervisor._broadcast.call_args[0][0]
        assert isinstance(broadcast, ApplyUnload) and broadcast.ok
        assert broadcast.generation == 4

    async def test_load_advances_generation_per_key(self, mocker):
        """Each committed (re)load advances its model key's generation,
        and the broadcast carries it. A same-key reload also COMMITS:
        the displaced instance's retiring unload publishes the staged
        replacement on the workers."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        first = Mock()
        second = Mock()
        pool_registry.model_initialiser = Mock(side_effect=[first, second])
        pool_registry.load_model = AsyncMock()
        pool_registry.unload_model = AsyncMock()

        await supervisor._load_settings(self._model_settings("m"))
        await supervisor._load_settings(self._model_settings("m"))

        applies = [
            c.args[0]
            for c in supervisor._broadcast.call_args_list
            if isinstance(c.args[0], ApplyLoad)
        ]
        assert [a.generation for a in applies] == [1, 2]
        # The reload committed the replacement: the first instance's
        # retiring unload was sent to the pool.
        pool_registry.unload_model.assert_awaited_once_with(first)

    async def test_frontend_forwards_wrapper_generation(self):
        """unload_model sends the wrapper's generation so the supervisor
        can distinguish retirement from a real unload."""
        from mlserver.parallel.model import ParallelModel
        from mlserver.parallel.shared import (
            OpRequest,
            OpResult,
            SharedPoolRegistry,
            _attach_generation,
        )

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings, frontend_id=0, reply_endpoint="ipc:///tmp/test-gen", control=child
        )
        await registry.start()
        try:
            model = Mock()
            model.name = "m"
            model.version = None
            model.settings = self._model_settings("m")
            wrapper = ParallelModel(model, registry._dispatcher)
            _attach_generation(wrapper, Mock(generation=2))

            import asyncio as asyncio_module

            task = asyncio_module.create_task(registry.unload_model(wrapper))
            request = await asyncio_module.wait_for(
                asyncio_module.get_running_loop().run_in_executor(None, parent.recv),
                5,
            )
            assert isinstance(request, OpRequest)
            assert request.kind == "unload"
            assert request.generation == 2
            parent.send(OpResult(op_id=request.op_id, ok=True))
            await asyncio_module.wait_for(task, timeout=5)
        finally:
            await registry.close()
            parent.close()
            child.close()

    async def test_unowned_model_cleanup_stays_local(self):
        """A model that never acquired a pool generation (the initial
        model of a failed load) cleans up locally: no OpRequest may be
        forwarded — the supervisor treats a generation-less unload as
        permission to unload the current serving instance."""
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
            reply_endpoint="ipc:///tmp/test-unowned",
            control=child,
        )
        await registry.start()
        try:
            model = Mock()
            model.name = "m"
            model.version = None
            model.settings = self._model_settings("m")
            # No generation attached: never committed via the pool.

            returned = await registry.unload_model(model)

            assert returned is model
            # Nothing was forwarded: no control message arrives on the
            # supervisor's end within a short window.
            assert not parent.poll(0.3)
        finally:
            await registry.close()
            parent.close()
            child.close()


class TestReplayGenerationPreventsFalseSupersede:
    """The restart replay must carry the committed generation. Without
    it, the replacement holds generation 0, and its later unloads look
    superseded (0 < current) and are silently skipped — leaving the
    model on workers and peers after the origin removed it."""

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
        )
        return FrontendSupervisor(settings, [])

    async def test_replacement_unload_not_falsely_superseded(self, mocker):
        """End-to-end: a replacement that received generation=1 from the
        replay sends an unload with generation=1 (not 0), and the
        supervisor processes it as a real unload, not a retirement.

        The wrapper→OpRequest side of this flow is covered through the
        real receive path by
        test_replayed_generation_reaches_unload_request (the unload
        request carries the replayed generation over the pipe); this
        test covers the supervisor's supersede decision on it."""
        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.unload_model = AsyncMock()

        # Model committed at generation 1
        instance = Mock()
        supervisor._loaded_models[("m", "v1")] = instance
        supervisor._generations[("m", "v1")] = 1

        # The replacement sends an unload with the generation it
        # received from the replay (generation=1, not 0)
        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=1
        )

        # The pool MUST unload — this is a real unload, not a retirement
        supervisor._pool_registry.unload_model.assert_awaited_once_with(instance)
        assert ("m", "v1") not in supervisor._loaded_models

    async def test_replacement_with_generation_zero_would_be_skipped(self):
        """Documents the failure mode: if the replay had sent generation
        0, an unload from the replacement would be silently skipped."""
        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.unload_model = AsyncMock()

        instance = Mock()
        supervisor._loaded_models[("m", "v1")] = instance
        supervisor._generations[("m", "v1")] = 1

        # A replacement with generation=0 sends an unload
        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=0
        )

        # The pool does NOT unload — the superseded check catches it.
        # This is why the replay MUST carry the committed generation.
        supervisor._pool_registry.unload_model.assert_not_called()
        assert ("m", "v1") in supervisor._loaded_models


class TestUnloadOutcomeAndGenerationInvariants:
    """Unload outcomes are captured when the pool call settles, and
    generation counters stay monotonic across unload/reload cycles."""

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

    @staticmethod
    def _settings():
        return ModelSettings(name="m", implementation="tests.fixtures.SumModel")

    async def test_superseded_unload_with_empty_keys_resolves_ok(self, mocker):
        """A versionless retirement arriving when no version of the
        name is loaded any more resolves ok — the no-op log path must
        not index into the empty key list."""
        from mlserver.parallel.shared import OpResult

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        # Nothing loaded; the unversioned key's generation counter was
        # advanced past the retiring wrapper's.
        supervisor._generations[("m", "")] = 4

        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u", generation=1
        )

        pool_registry.unload_model.assert_not_called()
        supervisor._broadcast.assert_not_called()
        result = supervisor._send_to.call_args[0][1]
        assert isinstance(result, OpResult) and result.ok

    async def test_generation_is_monotonic_across_unload_reload(self, mocker):
        """A real unload keeps the key's generation counter, so the
        next load commits a higher value — a stale wrapper from a peer
        whose unload failed can never collide with the fresh
        generation."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        instance = Mock(settings=self._settings())
        pool_registry.model_initialiser = Mock(return_value=instance)
        pool_registry.load_model = AsyncMock(return_value=instance)
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        await supervisor._load_settings(self._settings())  # generation 1
        await supervisor._unload_settings("m", None, origin=0, op_id="op-u")
        # the unload commits generation 2 (the counter advances)
        await supervisor._load_settings(self._settings())  # generation 3

        applies = [
            c.args[0]
            for c in supervisor._broadcast.call_args_list
            if isinstance(c.args[0], ApplyLoad)
        ]
        assert [a.generation for a in applies] == [1, 3], (
            "generation restarted after unload — a stale wrapper could "
            "collide with the fresh commit"
        )

    async def test_stale_peer_wrapper_cannot_unload_reloaded_key(self, mocker):
        """End-to-end collision scenario: a peer retains a generation-1
        wrapper (its local unload failed but was acked); the key is
        really unloaded and reloaded. The retained wrapper's retirement
        must be a no-op — with reset numbering it would match the fresh
        generation and unload it."""
        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        pool_registry.load_model = AsyncMock(return_value=Mock())
        pool_registry.unload_model = AsyncMock()
        supervisor._pool_registry = pool_registry

        await supervisor._load_settings(self._settings())  # generation 1
        await supervisor._unload_settings("m", None, origin=0, op_id="op-u")
        # the unload commits generation 2 (the counter advances)
        await supervisor._load_settings(self._settings())  # generation 3
        pool_registry.unload_model.reset_mock()

        # The peer's stale generation-1 wrapper retires now.
        await supervisor._unload_settings(
            "m", None, origin=0, op_id="op-u2", generation=1
        )

        pool_registry.unload_model.assert_not_called()
        assert ("m", "") in supervisor._loaded_models

    async def test_concurrent_load_during_unload_keeps_captured_outcome(self, mocker):
        """A load committing a replacement while the key's pool unload
        is in flight must not flip that unload's broadcast to failed,
        and must not be popped by the unload."""
        from mlserver.parallel.shared import ApplyUnload

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        original = Mock()
        replacement = Mock()
        supervisor._loaded_models[("m", "v1")] = original
        supervisor._generations[("m", "v1")] = 1

        async def unload_with_interleaved_load(instance):
            # A concurrent load commits a replacement while the pool
            # call is in flight.
            supervisor._loaded_models[("m", "v1")] = replacement
            supervisor._generations[("m", "v1")] = 2

        pool_registry.unload_model = AsyncMock(side_effect=unload_with_interleaved_load)

        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=1
        )

        # The pool call settled ok: the broadcast carries the captured
        # outcome, not inventory state observed after the interleaved
        # load.
        broadcast = supervisor._broadcast.call_args[0][0]
        assert isinstance(broadcast, ApplyUnload) and broadcast.ok
        # The identity-guarded removal spared the replacement.
        assert supervisor._loaded_models.get(("m", "v1")) is replacement

    async def test_missing_instance_broadcasts_desired_state(self, mocker):
        """A key resolved for unload whose instance is already absent
        broadcasts ok — the desired post-unload state."""
        from mlserver.parallel.shared import ApplyUnload

        supervisor = self._supervisor()
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        pool_registry = Mock()
        supervisor._pool_registry = pool_registry
        supervisor._loaded_models[("m", "v1")] = Mock()
        supervisor._loaded_models.pop(("m", "v1"))
        supervisor._generations[("m", "v1")] = 1

        await supervisor._unload_settings(
            "m", "v1", origin=0, op_id="op-u", generation=1
        )

        pool_registry.unload_model.assert_not_called()
        broadcast = supervisor._broadcast.call_args[0][0]
        assert isinstance(broadcast, ApplyUnload) and broadcast.ok


class TestOriginTimeoutReconciliation:
    """A committed load whose origin stopped waiting is late-applied:
    the origin frontend converges instead of missing the model the pool
    now serves."""

    @pytest.fixture
    async def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            # Short: the caller's expired wait is the trigger.
            model_operation_timeout=0.2,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint="ipc:///tmp/test-reconcile",
            control=child,
        )
        registry.attach_registry(AsyncMock())
        registry.attach_data_plane(Mock())
        registry._test_parent = parent
        await registry.start()
        try:
            yield registry
        finally:
            await registry.close()
            parent.close()
            child.close()

    @staticmethod
    def _model():
        model = Mock()
        model.settings = ModelSettings(
            name="m", implementation="tests.fixtures.SumModel"
        )
        return model

    async def _recv(self, shared, timeout: float = 5.0):
        import asyncio as asyncio_module

        loop = asyncio_module.get_running_loop()
        ready = await loop.run_in_executor(None, shared._test_parent.poll, timeout)
        if not ready:
            raise asyncio_module.TimeoutError("no control message")
        return await loop.run_in_executor(None, shared._test_parent.recv)

    async def test_committed_load_late_applies_after_origin_timeout(self, shared):
        """The race: the pool commits, the origin's wait expires before
        the result arrives, the late result applies the model on the
        origin with its committed generation."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyAck,
            ApplyLoad,
            OpRequest,
            OpResult,
            get_generation,
            serialise_model_settings,
        )

        registered = self._model()
        # The reconcile late-applies through the gated lock-free
        # entry point (the publication-time stale gate lives there).
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        load_task = asyncio_module.create_task(shared.load_model(self._model()))
        request = await self._recv(shared)
        assert isinstance(request, OpRequest) and request.kind == "load"

        # The supervisor broadcasts the apply (origin path: retain +
        # ack, no local registration yet).
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id=request.op_id,
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=1,
            ),
        )
        ack = await self._recv(shared)
        assert isinstance(ack, ApplyAck)

        # The origin's wait expires first: the load fails locally and
        # the origin has not applied anything.
        with pytest.raises(RuntimeError):
            await asyncio_module.wait_for(load_task, timeout=5)
        shared._model_registry.load.assert_not_awaited()

        # The committed result arrives late — through the listener, as
        # it would in production.
        shared._test_parent.send(OpResult(op_id=request.op_id, ok=True, generation=1))

        # The reconcile late-applies the load with the committed
        # generation attached.
        for _ in range(50):
            if shared._model_registry._load_for_apply.await_count:
                break
            await asyncio_module.sleep(0.02)
        shared._model_registry._load_for_apply.assert_awaited_once()
        assert get_generation(registered) == 1

    async def test_failed_late_result_does_not_apply(self, shared):
        """A late FAILURE result reconciles nothing: nothing was
        committed at the pool."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyLoad,
            OpResult,
            serialise_model_settings,
        )

        registered = self._model()
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        load_task = asyncio_module.create_task(shared.load_model(self._model()))
        request = await self._recv(shared)
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id=request.op_id,
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=1,
            ),
        )
        await self._recv(shared)  # the origin's ApplyAck

        with pytest.raises(RuntimeError):
            await asyncio_module.wait_for(load_task, timeout=5)

        shared._test_parent.send(
            OpResult(op_id=request.op_id, ok=False, error="pool load failed")
        )
        # Wait for the listener to actually handle the result: the
        # handler pops the retained apply, giving a deterministic
        # synchronization point (a fixed sleep could assert before the
        # message was read, passing vacuously).
        for _ in range(100):
            if request.op_id not in shared._origin_applies:
                break
            await asyncio_module.sleep(0.02)
        assert (
            request.op_id not in shared._origin_applies
        ), "the listener never processed the failed result"
        shared._model_registry.load.assert_not_awaited()

    async def test_successful_load_cleans_retention(self, shared):
        """The normal path leaves no retained apply behind (the result
        was consumed and the retention is dropped on success)."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyLoad,
            OpResult,
            serialise_model_settings,
        )

        registered = self._model()
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        load_task = asyncio_module.create_task(shared.load_model(self._model()))
        request = await self._recv(shared)
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id=request.op_id,
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=1,
            ),
        )
        await self._recv(shared)  # the origin's ApplyAck
        shared._test_parent.send(OpResult(op_id=request.op_id, ok=True, generation=1))
        await asyncio_module.wait_for(load_task, timeout=5)

        assert request.op_id not in shared._origin_applies
        # The origin never applies locally on the normal path.
        shared._model_registry.load.assert_not_awaited()

    async def test_late_result_superseded_by_newer_apply(self, shared):
        """Race: generation 1 times out on its origin, a
        retry's generation-2 apply publishes there, and only then does
        the generation-1 result arrive. The late apply must be skipped
        — re-registering generation 1 would replace the newer wrapper
        while routing keeps generation 2's tags."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyLoad,
            OpRequest,
            OpResult,
            serialise_model_settings,
        )

        registered = self._model()
        # The reconcile late-applies through the gated lock-free
        # entry point (the publication-time stale gate lives there).
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        load_task = asyncio_module.create_task(shared.load_model(self._model()))
        request = await self._recv(shared)
        assert isinstance(request, OpRequest)

        # The supervisor broadcasts the generation-1 apply (origin
        # path: retention + ack, no local registration).
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id=request.op_id,
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=1,
            ),
        )
        await self._recv(shared)  # the origin's ApplyAck

        # The origin's wait expires.
        with pytest.raises(RuntimeError):
            await asyncio_module.wait_for(load_task, timeout=5)

        # A retry's generation-2 apply publishes on this frontend (the
        # peer path: registers + bumps the key's revision).
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-retry",
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=2,
            ),
        )
        await self._recv(shared)  # the peer's ApplyAck
        assert shared._model_registry._load_for_apply.await_count == 1

        # The stale generation-1 result arrives late: the reconcile
        # must NOT re-register over the newer generation. Sync on the
        # retained apply's removal (the handler popped it), not a sleep.
        shared._test_parent.send(OpResult(op_id=request.op_id, ok=True, generation=1))
        for _ in range(100):
            if request.op_id not in shared._origin_applies:
                break
            await asyncio_module.sleep(0.02)
        assert request.op_id not in shared._origin_applies
        # The handler pops the retention BEFORE it schedules the
        # reconciliation task: drain the task so the load-count
        # assertion runs after the decision, not in that gap.
        if shared._reconcile_tasks:
            await asyncio_module.wait_for(
                asyncio_module.gather(
                    *list(shared._reconcile_tasks), return_exceptions=True
                ),
                timeout=5,
            )
        assert (
            shared._model_registry._load_for_apply.await_count == 1
        ), "a superseded late result must not re-apply an older generation"

    async def test_late_result_superseded_by_unload(self, shared):
        """An unload forwarded after the retention (but before the
        late result) must suppress the late apply: the model was
        removed after the load that timed out."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyLoad,
            OpResult,
            serialise_model_settings,
        )

        registered = self._model()
        registered.name = "m"
        registered.version = None
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        load_task = asyncio_module.create_task(shared.load_model(self._model()))
        request = await self._recv(shared)
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id=request.op_id,
                serialised_model_settings=serialise_model_settings(registered.settings),
                ok=True,
                env_hash=None,
                generation=1,
            ),
        )
        await self._recv(shared)  # the origin's ApplyAck
        with pytest.raises(RuntimeError):
            await asyncio_module.wait_for(load_task, timeout=5)

        # An unload of the same key is forwarded: resolve it like the
        # supervisor would. In production the supervisor broadcasts
        # ApplyUnload for real removals BEFORE resolving the origin —
        # simulate that broadcast (the origin path marks the key).
        from mlserver.parallel.shared import ApplyUnload, _attach_generation

        unload_wrapper = self._model()
        unload_wrapper.name = "m"
        unload_wrapper.version = None
        _attach_generation(unload_wrapper, Mock(generation=1))
        unload_task = asyncio_module.create_task(shared.unload_model(unload_wrapper))
        unload_request = await self._recv(shared)
        assert unload_request.kind == "unload"
        # The supervisor's broadcast (real removal only — superseded
        # retirements produce no broadcast).
        await handle_and_settle(
            shared,
            ApplyUnload(
                op_id=unload_request.op_id,
                name="m",
                version=None,
                ok=True,
                generation=1,
            ),
        )
        await self._recv(shared)  # the origin's ApplyAck for the broadcast
        shared._test_parent.send(OpResult(op_id=unload_request.op_id, ok=True))
        await asyncio_module.wait_for(unload_task, timeout=5)

        # The stale load result arrives after the unload: suppressed.
        # Sync on the retained apply's removal, not a sleep.
        shared._test_parent.send(OpResult(op_id=request.op_id, ok=True, generation=1))
        for _ in range(100):
            if request.op_id not in shared._origin_applies:
                break
            await asyncio_module.sleep(0.02)
        assert request.op_id not in shared._origin_applies
        # Wait for the reconciliation task to finish (or not) before
        # asserting: the retention map is cleared before the task runs.
        if shared._reconcile_tasks:
            await asyncio_module.gather(
                *shared._reconcile_tasks, return_exceptions=True
            )
        shared._model_registry.load.assert_not_awaited()


class TestPerKeyPartialSuccessUnload:
    """Each exact-key unload operation broadcasts its own key's
    outcome, not the aggregate of a name-wide sweep (the frontend
    enumerates wrappers, one operation per version)."""

    async def test_partial_failure_broadcasts_per_key(self, mocker):
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import ApplyUnload

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
        supervisor._broadcast = AsyncMock()
        supervisor._send_to = AsyncMock()
        supervisor._pool_registry = Mock()

        # Two versions loaded; v1 will unload successfully, v2 fails
        good_instance = Mock()
        bad_instance = Mock()
        supervisor._loaded_models[("m", "v1")] = good_instance
        supervisor._loaded_models[("m", "v2")] = bad_instance

        async def selective_unload(instance):
            if instance is bad_instance:
                raise RuntimeError("unload failed")

        supervisor._pool_registry.unload_model = AsyncMock(side_effect=selective_unload)

        # The frontend's name-wide unload enumerates its wrappers: one
        # exact operation per version.
        await supervisor._unload_settings("m", "v1", origin=0, op_id="op-u1")
        await supervisor._unload_settings("m", "v2", origin=0, op_id="op-u2")

        broadcasts = [
            c.args[0]
            for c in supervisor._broadcast.call_args_list
            if isinstance(c.args[0], ApplyUnload)
        ]
        # One broadcast per operation, each carrying its own key's
        # outcome: v1 ok, v2 not ok.
        assert len(broadcasts) == 2
        by_version = {b.version: b.ok for b in broadcasts}
        assert by_version == {"v1": True, "v2": False}


class TestInProcessModelOptOutInSharedTopology:
    """A model disabling parallel inference cannot run in the
    multi-frontend topology: in-process execution would load one copy
    per frontend, and its requests could never reach pool workers that
    did not load it. The load fails loudly instead of failing at
    inference time."""

    @staticmethod
    def _supervisor(frontend_workers: int):
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=frontend_workers,
            parallel_workers=1,
            parallel_inference_transport="zmq",
        )
        return FrontendSupervisor(settings, [])

    @staticmethod
    def _pool_registry():
        registry = Mock()
        registry.should_load_model.return_value = True
        instance = Mock()
        registry.model_initialiser.return_value = instance
        registry.load_model = AsyncMock(return_value=instance)
        return registry

    async def test_opted_out_model_rejected_with_multiple_frontends(self):
        """Supervised topology (frontend_workers>=1) + model-level
        parallel_workers=0 fails the load with an actionable error;
        nothing reaches the pool."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor(frontend_workers=2)
        pool = self._pool_registry()
        pool.should_load_model.return_value = False
        supervisor._pool_registry = pool
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="opted-out",
            implementation="tests.fixtures.SumModel",
            parallel_workers=0,
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert not event.ok
        # The error must tell the operator what to do about it.
        assert "frontend_workers=2" in event.error
        assert "in-process" in event.error
        assert "deprecated" in event.error
        pool.load_model.assert_not_called()
        assert ("opted-out", "") not in supervisor._loaded_models

    async def test_opted_out_model_rejected_at_single_supervised_frontend(self):
        """frontend_workers=1 is the supervised topology, so model-level
        parallel_workers=0 is rejected there exactly as at any higher
        count — supervised frontends never host models in-process."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor(frontend_workers=1)
        pool = self._pool_registry()
        pool.should_load_model.return_value = False
        supervisor._pool_registry = pool
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="opted-out-fw1",
            implementation="tests.fixtures.SumModel",
            parallel_workers=0,
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert not event.ok
        assert "frontend_workers=1" in event.error
        pool.load_model.assert_not_called()
        assert ("opted-out-fw1", "") not in supervisor._loaded_models

    async def test_opted_out_model_loads_in_process_at_direct_server(self):
        """frontend_workers=0 keeps the in-process execution path: the
        REAL pool registry returns the opted-out model untouched and
        never acquires a pool for it. (At frontend_workers=0 the CLI runs the direct
        server in production; the test drives the supervisor's load
        path directly to pin the opt-out branch.)"""
        from mlserver.parallel.registry import InferencePoolRegistry
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor(frontend_workers=0)
        # The real registry: only pool acquisition is mocked (as a
        # tripwire), so the opt-out branch itself is exercised. A fully
        # mocked registry would make this test vacuous — the supervisor
        # skips should_load_model entirely at frontend_workers=0.
        pool = InferencePoolRegistry(Settings(parallel_workers=1))
        pool._get_or_create = AsyncMock()
        supervisor._pool_registry = pool
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="opted-out",
            implementation="tests.fixtures.SumModel",
            parallel_workers=0,
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert event.ok
        # The opted-out model loaded in-process: no pool was acquired.
        pool._get_or_create.assert_not_awaited()
        # The instance is tracked in the supervisor's inventory.
        assert ("opted-out", "") in supervisor._loaded_models

    async def test_pooled_model_unaffected_in_shared_topology(self):
        """Models without the opt-out load normally with frontends>1."""
        from mlserver.parallel.shared import ApplyLoad

        supervisor = self._supervisor(frontend_workers=2)
        pool = self._pool_registry()
        supervisor._pool_registry = pool
        supervisor._broadcast = AsyncMock()

        model_settings = ModelSettings(
            name="pooled-model", implementation="tests.fixtures.SumModel"
        )
        await supervisor._load_settings(model_settings)

        event = supervisor._broadcast.call_args[0][0]
        assert isinstance(event, ApplyLoad)
        assert event.ok
        pool.load_model.assert_called_once()


class TestGenerationAwareUnloadTracking:
    """Unload records the removed generation (not infinity) so a newer
    load can still reconcile; a stale unload must not remove a newer
    generation on a peer."""

    @pytest.fixture
    async def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=5,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=1,  # peer (not origin)
            reply_endpoint="ipc:///tmp/test-gen-unload",
            control=child,
        )
        registry.attach_registry(AsyncMock())
        registry.attach_data_plane(Mock())
        registry._test_parent = parent
        await registry.start()
        try:
            yield registry
        finally:
            await registry.close()
            parent.close()
            child.close()

    async def _recv(self, shared, timeout=5.0):
        import asyncio as asyncio_module

        loop = asyncio_module.get_running_loop()
        ready = await loop.run_in_executor(None, shared._test_parent.poll, timeout)
        if not ready:
            raise asyncio_module.TimeoutError("no control message")
        return await loop.run_in_executor(None, shared._test_parent.recv)

    async def test_unload_records_generation_not_infinity(self, shared):
        """After unloading gen 1, a late gen-2 result can still
        reconcile (the pool serves gen 2; the origin should register
        it). The old infinity marking blocked this."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyAck,
            ApplyLoad,
            ApplyUnload,
            OpResult,
        )

        registered = Mock()
        registered.settings = ModelSettings(
            name="m", implementation="tests.fixtures.SumModel"
        )
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        # The peer starts BELOW the unload's generation: the tracked
        # value must be ADVANCED by the unload's own recording — a
        # pre-seeded equal value would make that assertion vacuous.
        shared._key_generations[("m", "")] = 0

        # An unload of gen 1 arrives (peer path).
        await handle_and_settle(
            shared,
            ApplyUnload(op_id="op-u", name="m", version=None, ok=True, generation=1),
        )
        ack = await self._recv(shared)
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-u"

        # The unload recorded its generation: 1 (not infinity, and not
        # the pre-seeded 0).
        assert shared._key_generations[("m", "")] == 1

        # A late gen-2 result can reconcile: 2 > 1.
        load_msg = ApplyLoad(
            op_id="op-late",
            serialised_model_settings=(
                '{"name": "m", "implementation": "tests.fixtures.SumModel"}'
            ),
            ok=True,
            env_hash=None,
            generation=2,
        )
        shared._origin_applies["op-late"] = load_msg
        shared._test_parent.send(OpResult(op_id="op-late", ok=True, generation=2))
        for _ in range(100):
            if shared._model_registry._load_for_apply.await_count > 0:
                break
            await asyncio_module.sleep(0.02)
        assert (
            shared._model_registry._load_for_apply.await_count == 1
        ), "gen-2 result should reconcile after gen-1 unload (2 > 1)"

    async def test_stale_unload_skipped_on_peer(self, shared):
        """A peer holding gen 2 must skip an unload targeting gen 1 —
        the newer load committed during the supervisor's pool await.
        Routing tags and batch affinity must survive the skip."""
        from mlserver.parallel.shared import ApplyUnload

        # Simulate: gen 2 was applied on this peer with routing tags.
        shared._key_generations[("m", "")] = 2
        shared._dispatcher.set_model_pool("m", "pool-b", None)
        shared._dispatcher.set_model_batched("m", True, None)

        # A stale unload for gen 1 arrives.
        await handle_and_settle(
            shared,
            ApplyUnload(
                op_id="op-stale", name="m", version=None, ok=True, generation=1
            ),
        )
        await self._recv(shared)  # the skip ack

        # The peer still holds gen 2 (the unload was skipped).
        assert shared._key_generations[("m", "")] == 2
        # Routing tags survived: the model still routes to pool-b with
        # batch affinity (a stale unload must not clear them).
        assert ("m", "") in shared._dispatcher._batched_models
        assert shared._dispatcher._model_pools.get(("m", "")) == "pool-b"

    async def test_same_generation_unload_skips_late_result(self, shared):
        """After unloading gen 2, a late gen-2 result must NOT
        reconcile (the model was removed at that generation)."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import (
            ApplyAck,
            ApplyLoad,
            ApplyUnload,
            OpResult,
        )

        registered = Mock()
        registered.settings = ModelSettings(
            name="m", implementation="tests.fixtures.SumModel"
        )
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        # The peer starts BELOW the unload's generation so the
        # recording is observable: the unload itself must advance the
        # tracked generation to 2.
        shared._key_generations[("m", "")] = 0
        await handle_and_settle(
            shared,
            ApplyUnload(op_id="op-u2", name="m", version=None, ok=True, generation=2),
        )
        ack = await self._recv(shared)
        assert isinstance(ack, ApplyAck)
        assert ack.op_id == "op-u2"
        assert shared._key_generations[("m", "")] == 2

        # A late gen-2 result arrives: 2 <= 2 → skip.
        load_msg = ApplyLoad(
            op_id="op-late2",
            serialised_model_settings=(
                '{"name": "m", "implementation": "tests.fixtures.SumModel"}'
            ),
            ok=True,
            env_hash=None,
            generation=2,
        )
        shared._origin_applies["op-late2"] = load_msg
        shared._test_parent.send(OpResult(op_id="op-late2", ok=True, generation=2))
        # Sync on the retention pop, then drain the reconciliation
        # task — a fixed sleep could assert before the decision runs.
        for _ in range(100):
            if "op-late2" not in shared._origin_applies:
                break
            await asyncio_module.sleep(0.02)
        if shared._reconcile_tasks:
            await asyncio_module.wait_for(
                asyncio_module.gather(
                    *list(shared._reconcile_tasks), return_exceptions=True
                ),
                timeout=5,
            )
        shared._model_registry.load.assert_not_awaited()


async def test_cancelled_op_task_does_not_leak_pending_entry(mocker):
    """An op task cancelled mid-broadcast (after the ack collector was
    registered, before the ack collection runs) must not leak the
    _pending_ops entry: the cleanup spans registration to completion,
    not just the ack call."""
    from mlserver.frontend import FrontendSupervisor
    from mlserver.settings import ModelSettings

    settings = Settings(
        http_port=0,
        grpc_port=0,
        metrics_port=0,
        frontend_workers=2,
        parallel_workers=1,
        parallel_inference_transport="zmq",
    )
    supervisor = FrontendSupervisor(settings, [])
    supervisor._broadcast = AsyncMock()  # returns instantly; the
    # cancellation below lands at an await inside the locked body.
    pool_registry = Mock()
    pool_registry.should_load_model = Mock(return_value=True)
    instance = Mock(
        settings=ModelSettings(name="m", implementation="tests.fixtures.SumModel")
    )
    pool_registry.model_initialiser = Mock(return_value=instance)
    pool_registry.load_model = AsyncMock(return_value=instance)
    supervisor._pool_registry = pool_registry

    registered = asyncio.Event()

    async def broadcast_then_block(message):
        registered.set()
        await asyncio.Event().wait()  # cancelled here, mid-operation

    supervisor._broadcast = broadcast_then_block

    model_settings = ModelSettings(name="m", implementation="tests.fixtures.SumModel")
    task = asyncio.create_task(
        supervisor._load_settings(model_settings, origin=0, op_id="op-leak")
    )
    # The entry is registered before the broadcast; the task is then
    # cancelled while the broadcast is still in flight.
    for _ in range(100):
        if "op-leak" in supervisor._pending_ops:
            break
        await asyncio.sleep(0.001)
    assert "op-leak" in supervisor._pending_ops

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The registered entry did not outlive the cancelled operation.
    assert "op-leak" not in supervisor._pending_ops


class TestReconcileClearsStaleRemovalMark:
    """A successful late reconcile clears a strictly-older removal
    mark, matching _apply_load — harmless by construction (every
    guard compares >= against strictly-later generations), but a
    successful publication should not leave removal state behind."""

    @pytest.fixture
    async def shared(self):
        from mlserver.parallel.shared import SharedPoolRegistry

        settings = Settings(
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_operation_timeout=0.2,
        )
        parent, child = Pipe(duplex=True)
        registry = SharedPoolRegistry(
            settings,
            frontend_id=0,
            reply_endpoint="ipc:///tmp/test-reconcile-pop",
            control=child,
        )
        registry._test_parent = parent
        registry.attach_registry(Mock())
        await registry.start()
        try:
            yield registry
        finally:
            await registry.close()
            parent.close()
            child.close()

    async def test_successful_reconcile_pops_older_removal_mark(self, shared) -> None:
        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad, OpResult

        registered = Mock()
        registered.settings = ModelSettings(
            name="m", implementation="tests.fixtures.SumModel"
        )
        shared._model_registry._load_for_apply = AsyncMock(return_value=registered)

        # A removal mark strictly OLDER than the incoming generation:
        # the reconcile's gate passes (1 < 2) and its publication
        # must clear the mark.
        shared._key_removals[("m", "")] = 1

        load_msg = ApplyLoad(
            op_id="op-late",
            serialised_model_settings=(
                '{"name": "m", "implementation": "tests.fixtures.SumModel"}'
            ),
            ok=True,
            env_hash=None,
            generation=2,
        )
        shared._origin_applies["op-late"] = load_msg
        shared._test_parent.send(OpResult(op_id="op-late", ok=True, generation=2))

        for _ in range(100):
            if shared._model_registry._load_for_apply.await_count:
                break
            await asyncio_module.sleep(0.02)
        # Let the reconcile finish past its publication.
        await asyncio_module.sleep(0.1)
        shared._model_registry._load_for_apply.assert_awaited_once()
        assert ("m", "") not in shared._key_removals
