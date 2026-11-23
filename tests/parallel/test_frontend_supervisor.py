"""Frontend supervisor lifecycle in the multi-frontend topology.

Spawn and readiness gating, restart retries and replay ordering
against live model updates, partial-frontend retirement tracked
through shutdown, control-plane send/recv executor isolation, death
detection that survives foreign reaping, and reuse-port binding.
"""

from tests.utils import handle_and_settle

import asyncio
import json
import socket as socket_module
from multiprocessing import Pipe
from unittest.mock import AsyncMock, Mock

import pytest

from mlserver.settings import Settings, ModelSettings


class TestStartupModelGating:
    """Startup readiness deferred until all startup models are applied."""

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
        yield registry
        parent.close()
        child.close()

    async def test_startup_pending_initially(self, shared):
        """startup_models_pending is True before StartupComplete arrives."""
        assert shared.startup_models_pending is True

    async def test_startup_complete_releases_gate(self, shared):
        """StartupComplete message releases the readiness gate and calls
        startup_complete on the data plane."""
        from mlserver.parallel.shared import StartupComplete

        await handle_and_settle(shared, StartupComplete(model_count=3))
        assert shared.startup_models_pending is False
        assert shared._startup_done is True
        shared._data_plane.startup_complete.assert_called_once()

    async def test_gate_survives_applies_before_complete(self, shared):
        """Broadcast applies landing before StartupComplete do not
        release the gate; only the StartupComplete message does."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            StartupComplete,
            serialise_model_settings,
        )

        model_settings = ModelSettings(
            name="sum-model", implementation="tests.fixtures.SumModel"
        )
        serialised = serialise_model_settings(model_settings)

        # Two startup-model applies land; the gate stays closed.
        for op_id in ("op-1", "op-2"):
            await handle_and_settle(
                shared,
                ApplyLoad(op_id=op_id, serialised_model_settings=serialised, ok=True),
            )
        assert shared.startup_models_pending is True
        shared._data_plane.startup_complete.assert_not_called()

        # StartupComplete releases the gate.
        await handle_and_settle(shared, StartupComplete(model_count=2))
        assert shared.startup_models_pending is False
        shared._data_plane.startup_complete.assert_called_once()

    @staticmethod
    def _started_server(mock_pool, mocker):
        """An MLServer wired to a stubbed-out serving surface, with its
        start() running — so the startup-gate decision is exercised
        through the real code path, not by inspecting injected values."""
        import asyncio as asyncio_module

        from mlserver.server import MLServer

        settings = Settings(parallel_workers=0)
        server = MLServer(settings, inference_pool_registry=mock_pool)
        server._data_plane.startup_complete = Mock()
        # Stub the metrics server too: it is incidental to the gate
        # decision, and a real bind on the shared metrics port races
        # the test's stop() (leaking the socket into the next test).
        for attr in ("_rest_server", "_grpc_server", "_metrics_server"):
            stub = Mock()
            stub.start = AsyncMock()
            stub.stop = AsyncMock()
            mocker.patch.object(server, attr, stub)
        task = asyncio_module.create_task(server.start([]))
        return server, task

    async def _stop_started_server(self, server, task):
        import asyncio as asyncio_module

        await server.stop()
        task.cancel()
        await asyncio_module.gather(task, return_exceptions=True)

    async def test_start_defers_startup_complete_with_pending(
        self, prometheus_registry, mocker, caplog
    ):
        """MLServer.start defers startup_complete while the shared pool
        has pending startup models."""
        import asyncio as asyncio_module
        import logging as logging_module

        from mlserver.parallel.shared import SharedPoolRegistry

        mock_pool = Mock(spec=SharedPoolRegistry)
        mock_pool.startup_models_pending = True
        mock_pool.close = AsyncMock()
        mock_pool.start = AsyncMock()

        server, task = self._started_server(mock_pool, mocker)
        try:
            # Milestone: start() must reach the readiness decision and
            # log the deferral. An early crash also leaves
            # startup_complete uncalled, which would otherwise satisfy
            # assert_not_called() vacuously.
            with caplog.at_level(logging_module.INFO):
                for _ in range(200):
                    if "deferring startup_complete" in caplog.text:
                        break
                    if task.done():
                        # Surface a startup exception (or a normal early
                        # return) instead of waiting out the poll.
                        task.result()
                        break
                    await asyncio_module.sleep(0.01)
            assert (
                "deferring startup_complete" in caplog.text
            ), "server.start() never reached the deferred-readiness decision"
            server._data_plane.startup_complete.assert_not_called()
        finally:
            await self._stop_started_server(server, task)

    async def test_start_calls_startup_complete_without_pending(
        self, prometheus_registry, mocker
    ):
        """MLServer.start calls startup_complete immediately when no
        startup models are pending (standard single-process path)."""
        from mlserver.parallel.shared import SharedPoolRegistry

        mock_pool = Mock(spec=SharedPoolRegistry)
        mock_pool.startup_models_pending = False
        mock_pool.close = AsyncMock()
        mock_pool.start = AsyncMock()

        server, task = self._started_server(mock_pool, mocker)
        import asyncio as asyncio_module

        await asyncio_module.sleep(0.2)
        try:
            server._data_plane.startup_complete.assert_called_once()
        finally:
            await self._stop_started_server(server, task)


class TestEmptyRegistryReadinessPreservation:
    """The user's empty_registry_readiness setting is not overwritten."""

    @pytest.mark.parametrize("user_value", [True, False])
    def test_user_setting_not_overwritten_with_models(self, user_value):
        """When startup models exist, the user's
        empty_registry_readiness setting is preserved (True catches
        the historical overwrite to False; False catches any flip
        back). Constructing the supervisor matters: reading the
        setting back without it cannot catch an overwrite."""
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            empty_registry_readiness=user_value,
        )
        models = [ModelSettings(name="m", implementation="tests.fixtures.SumModel")]
        FrontendSupervisor(settings, models)

        assert settings.empty_registry_readiness is user_value

    @pytest.mark.parametrize("user_value", [True, False])
    def test_frontend_spawn_settings_preserve_user_value(self, mocker, user_value):
        """The settings dict handed to a frontend process keeps the
        user's empty_registry_readiness even when startup models exist
        (True catches the historical overwrite to False; False catches
        any flip back). The startup gate is the pending flag, not a
        settings rewrite — and mutating only the per-frontend dict
        would bypass the constructor-level assertions above."""
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            empty_registry_readiness=user_value,
        )
        models = [ModelSettings(name="m", implementation="tests.fixtures.SumModel")]
        supervisor = FrontendSupervisor(settings, models)
        supervisor._reply_endpoints = [
            "ipc:///tmp/test-reply-0",
            "ipc:///tmp/test-reply-1",
        ]

        captured = {}

        class FakeSpawnCtx:
            def Pipe(self, duplex=True):
                return Mock(), Mock()

            def Process(self, **kwargs):
                captured.update(kwargs)
                return Mock()

        mocker.patch(
            "mlserver.frontend.multiprocessing.get_context",
            return_value=FakeSpawnCtx(),
        )

        supervisor._make_frontend_process(0)

        _frontend_id, settings_dict, _reply, _conn, _custom, _socks, _loads = captured[
            "args"
        ]
        assert settings_dict["empty_registry_readiness"] is user_value


class TestIPv6ReusePort:
    """_reuse_port_sockets binds every usable resolved address."""

    def test_ipv6_host_creates_ipv6_socket(self):
        """An IPv6 host creates an AF_INET6 socket."""
        from mlserver.utils import _reuse_port_sockets

        socks = _reuse_port_sockets("::1", 0)
        try:
            assert socks[0].family == socket_module.AF_INET6
        finally:
            for sock in socks:
                sock.close()

    def test_ipv4_host_creates_ipv4_socket(self):
        """An IPv4 host creates an AF_INET4 socket."""
        from mlserver.utils import _reuse_port_sockets

        socks = _reuse_port_sockets("0.0.0.0", 0)
        try:
            assert socks[0].family == socket_module.AF_INET
        finally:
            for sock in socks:
                sock.close()

    def test_proto_is_tcp(self):
        """The sockets have IPPROTO_TCP proto (required for nodelay)."""
        from mlserver.utils import _reuse_port_sockets

        socks = _reuse_port_sockets("0.0.0.0", 0)
        try:
            assert all(sock.proto == socket_module.IPPROTO_TCP for sock in socks)
        finally:
            for sock in socks:
                sock.close()

    def test_socket_creation_failure_falls_through(self, mocker):
        """A first candidate whose socket cannot even be created (e.g.
        an IPv6 address on a host without IPv6 support) falls through
        to the next resolved address instead of aborting the helper."""
        from mlserver.utils import _reuse_port_sockets

        # Port 0 lets the helper take an OS-assigned port with no
        # probe-then-bind gap for a parallel test to race.
        port = 0
        real_socket = socket_module.socket

        def selective_socket(family, *args, **kwargs):
            if family == socket_module.AF_INET6:
                raise OSError("IPv6 unavailable (EAFNOSUPPORT)")
            return real_socket(family, *args, **kwargs)

        mocker.patch(
            "socket.getaddrinfo",
            return_value=[
                (
                    socket_module.AF_INET6,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("::1", port),
                ),
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", port),
                ),
            ],
        )
        mocker.patch("socket.socket", side_effect=selective_socket)

        socks = _reuse_port_sockets("unused-hostname", port)
        try:
            assert len(socks) == 1
            assert socks[0].family == socket_module.AF_INET
            assert socks[0].getsockname()[0] == "127.0.0.1"
        finally:
            for sock in socks:
                sock.close()

    def test_falls_through_to_next_resolved_address(self, mocker):
        """A hostname whose first resolved address cannot bind (but
        whose second can) must still start — matching the
        direct-server listener, which tries every candidate."""
        from mlserver.utils import _reuse_port_sockets

        # Bind the blocker to port 0 and read the assigned port: the
        # port stays owned from assignment to use, with no probe-close
        # gap for a parallel test to race into. The blocker occupies
        # the first candidate WITHOUT SO_REUSEPORT: a later
        # SO_REUSEPORT bind of the same address fails (reuseport
        # requires every socket on the port to opt in).
        blocker = socket_module.socket(socket_module.AF_INET)
        blocker.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", port),
                ),
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("127.0.0.2", port),
                ),
            ],
        )
        try:
            socks = _reuse_port_sockets("unused-hostname", port)
            try:
                assert [s.getsockname()[0] for s in socks] == ["127.0.0.2"]
            finally:
                for sock in socks:
                    sock.close()
        finally:
            blocker.close()

    def test_raises_when_no_address_binds(self, mocker):
        """Every candidate unbindable: the last OSError propagates."""
        from mlserver.utils import _reuse_port_sockets

        blockers = []
        port = None
        for addr in ("127.0.0.1", "127.0.0.2"):
            # The first blocker's port-0 bind assigns the port; the
            # second reuses it — the port stays owned throughout, with
            # no probe-close gap for a parallel test to race into.
            blocker = socket_module.socket(socket_module.AF_INET)
            blocker.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            blocker.bind((addr, 0 if port is None else port))
            blocker.listen(1)
            if port is None:
                port = blocker.getsockname()[1]
            blockers.append(blocker)
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    (addr, port),
                )
                for addr in ("127.0.0.1", "127.0.0.2")
            ],
        )
        try:
            with pytest.raises(OSError):
                _reuse_port_sockets("unused-hostname", port)
        finally:
            for blocker in blockers:
                blocker.close()

    def test_all_bindable_candidates_are_returned(self, mocker):
        """Every bindable resolved address gets a listener — matching
        the direct-server path, which serves all of a hostname's
        addresses (not just the first)."""
        from mlserver.utils import _reuse_port_sockets

        # Port 0 lets each listener take an OS-assigned port with no
        # probe-then-bind gap; this test asserts the addresses served.
        port = 0
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", port),
                ),
                (
                    socket_module.AF_INET,
                    socket_module.SOCK_STREAM,
                    socket_module.IPPROTO_TCP,
                    "",
                    ("127.0.0.2", port),
                ),
            ],
        )
        socks = _reuse_port_sockets("unused-hostname", port)
        try:
            assert sorted(s.getsockname()[0] for s in socks) == [
                "127.0.0.1",
                "127.0.0.2",
            ]
        finally:
            for sock in socks:
                sock.close()


class TestProcessExitedDetection:
    """Frontend death detection survives a corpse reaped by another
    reaper: the pool's SIGCHLD handler waitpids -1 (consuming ANY
    child's corpse, frontends included) before filtering, and
    is_alive() then reports True forever (poll() maps ECHILD to
    None). The sentinel stays a reliable exit signal."""

    def test_process_exited_after_foreign_reap(self):
        import multiprocessing as mp_module
        import os as os_module
        import time as time_module

        from mlserver.frontend import _process_exited

        ctx = mp_module.get_context("spawn")
        proc = ctx.Process(target=time_module.sleep, args=(30,))
        proc.start()
        try:
            proc.kill()
            # Reap the corpse like the pool's SIGCHLD reaper would.
            deadline = time_module.time() + 10
            while time_module.time() < deadline:
                try:
                    pid, _ = os_module.waitpid(proc.pid, os_module.WNOHANG)
                    if pid == proc.pid:
                        break
                except ChildProcessError:
                    break
                time_module.sleep(0.05)
            # is_alive() is now permanently blind...
            assert proc.is_alive(), "expected the ECHILD blindness"
            # ...but the sentinel still reports the exit.
            assert _process_exited(proc)
        finally:
            proc.join(timeout=5)

    def test_process_exited_false_while_running(self):
        import multiprocessing as mp_module
        import time as time_module

        from mlserver.frontend import _process_exited

        ctx = mp_module.get_context("spawn")
        proc = ctx.Process(target=time_module.sleep, args=(30,))
        proc.start()
        try:
            assert not _process_exited(proc)
        finally:
            proc.terminate()
            proc.join(timeout=5)


class TestCustomServerSettingsPropagation:
    """The _custom_* server settings are plain class attributes
    excluded from model_dump(); the spawn must carry them alongside so
    programmatic embedders keep their custom server configuration in
    every frontend process."""

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        return FrontendSupervisor(settings, [])

    def test_factory_passes_custom_settings_in_spawn_args(self, mocker):
        from mlserver.frontend import _run_frontend

        supervisor = self._supervisor()
        supervisor._settings._custom_rest_server_settings = {"timeout_keep_alive": 5}
        supervisor._settings._custom_grpc_server_settings = {"options": [("grpc.x", 1)]}
        supervisor._reply_endpoints = ["ipc:///tmp/test-custom-r0"]

        ctx = mocker.patch("multiprocessing.get_context").return_value
        ctx.Pipe.return_value = (Mock(), Mock())
        supervisor._make_frontend_process(0)

        kwargs = ctx.Process.call_args.kwargs
        assert kwargs["target"] is _run_frontend
        # The settings dict itself cannot carry them (model_dump drops
        # plain class attributes) — they travel as a side channel.
        assert "_custom_rest_server_settings" not in kwargs["args"][1]
        assert kwargs["args"][4] == {
            "rest": {"timeout_keep_alive": 5},
            "metrics": None,
            "grpc": {"options": [("grpc.x", 1)]},
        }

    def test_factory_retires_superseded_slot_resources(self, mocker):
        """A factory call must retire the slot's previous connection
        and send executor before registering the new one. A connection
        left in the slot after a failed start() leaks its fds, and a
        send writer still blocked on the old connection would occupy
        the slot's single-thread executor — every send to the
        replacement would queue behind it, time out, and terminate the
        replacement (a restart storm until the budget is exhausted)."""
        supervisor = self._supervisor()
        supervisor._reply_endpoints = ["ipc:///tmp/test-retire-r0"]

        old_control = Mock()
        supervisor._controls = {0: old_control}
        old_executor = Mock()
        supervisor._send_executors = {0: old_executor}

        ctx = mocker.patch("multiprocessing.get_context").return_value
        ctx.Pipe.return_value = (Mock(), Mock())
        supervisor._make_frontend_process(0)

        old_control.close.assert_called_once()
        old_executor.shutdown.assert_called_once_with(wait=False)
        assert supervisor._controls[0] is not old_control
        assert 0 not in supervisor._send_executors

    def test_run_frontend_reattaches_custom_settings(self, mocker):
        """The child reconstructs Settings from the custom-free dump
        and re-attaches the passed dicts before building the server."""
        from mlserver import server as server_module
        from mlserver.frontend import _run_frontend

        supervisor = self._supervisor()
        settings_dict = supervisor._settings.model_dump()
        captured = {}

        class FakeServer:
            def __init__(self, settings, **kwargs):
                captured["settings"] = settings
                self._model_registry = Mock()
                self._data_plane = Mock()
                self._live = False

            async def start(self, models):
                return None

            async def stop(self):
                captured["stop"] = captured.get("stop", 0) + 1

        mocker.patch.object(server_module, "MLServer", FakeServer)
        # The entry point re-registers SIGTERM handlers: keep the test
        # process's handlers untouched.
        mocker.patch("mlserver.frontend.signal.signal")

        _run_frontend(
            0,
            settings_dict,
            "ipc:///tmp/test-custom-r0",
            Mock(),
            custom_server_settings={
                "rest": {"timeout_keep_alive": 5},
                "metrics": None,
                "grpc": None,
            },
        )

        child_settings = captured["settings"]
        assert child_settings._custom_rest_server_settings == {"timeout_keep_alive": 5}
        assert child_settings._custom_grpc_server_settings is None

    @pytest.mark.parametrize("live_at_finally", [True, False])
    def test_frontend_stops_only_when_server_live(self, mocker, live_at_finally):
        """A SIGTERM-initiated stop completes while the server tasks
        drain (_live False): the finally must not re-run the full
        shutdown path. An unexpected task exit keeps _live set, so the
        cleanup stop still runs."""
        from mlserver import server as server_module
        from mlserver.frontend import _run_frontend

        supervisor = self._supervisor()
        settings_dict = supervisor._settings.model_dump()
        counted = {"stop": 0}

        class FakeServer:
            def __init__(self, settings, **kwargs):
                self._model_registry = Mock()
                self._data_plane = Mock()
                self._live = live_at_finally

            async def start(self, models):
                return None

            async def stop(self):
                counted["stop"] += 1

        mocker.patch.object(server_module, "MLServer", FakeServer)
        mocker.patch("mlserver.frontend.signal.signal")

        _run_frontend(0, settings_dict, "ipc:///tmp/test-custom-r0", Mock())

        assert counted["stop"] == (1 if live_at_finally else 0)


class TestSIGCHLDHandlerScope:
    """The SIGCHLD handler only processes pool workers, not frontends —
    and never raises into interrupted code when its loop has closed."""

    # The source-string reaper filter tests were removed: the
    # behavioral test_reaper_skips_foreign_pids_and_keeps_failed_pools
    # (below) covers the same contract through the real handler —
    # foreign PIDs notify nobody, failed-pool PIDs notify their owner.

    def test_pool_tracks_pid_across_lifecycle_states(self):
        """tracks_pid recognises workers in every lifecycle structure,
        matching what on_worker_stop processes."""
        from unittest.mock import Mock
        from mlserver.parallel.pool import InferencePool
        from mlserver.settings import Settings

        pool = InferencePool(Settings())

        started = Mock()
        started.pid = 101
        pool._workers[101] = started
        assert pool.tracks_pid(101)

        starting = Mock()
        starting.pid = 102
        pool._starting_workers[102] = starting
        assert pool.tracks_pid(102)

        unstarted = Mock()
        unstarted.pid = 103
        pool._unstarted_workers.add(unstarted)
        assert pool.tracks_pid(103)

        closing = Mock()
        closing.pid = 104
        pool._closing_workers[id(closing)] = closing
        assert pool.tracks_pid(104)

        # A frontend child (or any foreign pid) is not tracked.
        assert not pool.tracks_pid(999)

    def test_wake_on_closed_loop_is_dropped(self):
        """A SIGCHLD arriving after the installing loop closed must not
        raise into whatever code the signal interrupted."""
        import asyncio as asyncio_module
        import signal as signal_module
        from mlserver.parallel.registry import InferencePoolRegistry
        from mlserver.settings import Settings

        registry = InferencePoolRegistry(Settings())
        dead_loop = asyncio_module.new_event_loop()
        dead_loop.close()

        # Must not raise RuntimeError("Event loop is closed")
        registry._wake_loop(dead_loop, signal_module.SIGCHLD, None)

    async def test_wake_on_live_loop_schedules_the_reaper(self):
        """On a live loop the wake schedules the worker-stop task —
        observed by the reaper actually starting, then draining."""
        import asyncio as asyncio_module
        import signal as signal_module
        from mlserver.parallel.registry import InferencePoolRegistry
        from mlserver.settings import Settings

        registry = InferencePoolRegistry(Settings())
        original = signal_module.getsignal(signal_module.SIGCHLD)
        # Signal from inside the reaper: without it, an empty task set
        # could also mean the handler never scheduled anything.
        reaper_started = asyncio_module.Event()
        real_handle = registry._handle_worker_stop

        async def observed_handle(signum, frame):
            reaper_started.set()
            await real_handle(signum, frame)

        registry._handle_worker_stop = observed_handle
        try:
            registry._install_signal_handler()
            handler = signal_module.getsignal(signal_module.SIGCHLD)
            assert handler is not original

            # Fire the installed handler as the OS would (from the main
            # thread): the wake lands on this test's live loop.
            handler(signal_module.SIGCHLD, None)
            # The reaper must actually start (else the test fails here
            # instead of passing vacuously on an untouched task set).
            await asyncio_module.wait_for(reaper_started.wait(), timeout=2)
            # It completes and drains its task set (no children).
            for _ in range(100):
                if not registry._worker_stop_tasks:
                    break
                await asyncio_module.sleep(0.01)
            assert not registry._worker_stop_tasks
        finally:
            registry._restore_signal_handler()
            assert signal_module.getsignal(signal_module.SIGCHLD) is original

    async def test_reaper_skips_foreign_pids_and_keeps_failed_pools(self, mocker):
        """Behavioral reap-filter check (not source strings): a foreign
        PID notifies nobody; a worker retained by a failed pool is
        still reaped and its owner notified."""
        from mlserver.parallel.registry import InferencePoolRegistry
        from mlserver.settings import Settings

        registry = InferencePoolRegistry(Settings())
        live_pool = Mock()
        live_pool.tracks_pid = Mock(side_effect=lambda pid: pid == 101)
        live_pool.on_worker_stop = AsyncMock()
        failed_pool = Mock()
        failed_pool.tracks_pid = Mock(side_effect=lambda pid: pid == 102)
        failed_pool.on_worker_stop = AsyncMock()
        registry._pools = {"live": live_pool}
        registry._failed_pools = {"failed": failed_pool}
        registry._default_pool = None

        # 999: foreign child (e.g. a frontend); 102: failed pool's
        # retained worker; 0: no more children.
        reaped = iter([(999, 0), (102, 0), (0, 0)])
        mocker.patch(
            "mlserver.parallel.registry.os.waitpid",
            side_effect=lambda *args: next(reaped),
        )

        await registry._handle_worker_stop(None, None)

        # The foreign PID notified nobody (no call carries it).
        for pool in (live_pool, failed_pool):
            notified = [call.args[0] for call in pool.on_worker_stop.call_args_list]
            assert 999 not in notified
        # The failed pool's retained worker reached its owner.
        failed_pool.on_worker_stop.assert_any_call(102, 0)


class TestWorkerReplayOnRestart:
    """Frontend restart replays both worker incarnations and models to
    the replacement frontend — workers first, so its empty dispatcher
    can route before any model re-applies."""

    async def test_restart_replays_workers_before_models(self, mocker) -> None:
        """_restart_frontend sends a WorkerReadyEvent per READY worker
        (skipping endpoint-less and quarantined ones, tagged with the
        pool env hash), followed by the committed model re-applies and
        the startup-gate release, in that order."""
        import asyncio as asyncio_module
        from unittest.mock import AsyncMock

        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.registry import ENV_HASH_ATTR
        from mlserver.parallel.shared import (
            ApplyLoad,
            StartupComplete,
            WorkerReadyEvent,
        )

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
        )
        supervisor = FrontendSupervisor(settings, [])

        # A pool with one ready worker, one not-yet-routable worker and
        # one quarantined worker (mid-recovery: still in _workers with a
        # live endpoint, but not in the ready set).
        routable = Mock(
            pid=101, _zmq_request_endpoint="ipc:///tmp/w101", env_hash="abc"
        )
        endpointless = Mock(pid=102, _zmq_request_endpoint=None, env_hash="abc")
        quarantined = Mock(
            pid=103, _zmq_request_endpoint="ipc:///tmp/w103", env_hash="abc"
        )
        pool = Mock()
        pool._workers = {101: routable, 102: endpointless, 103: quarantined}
        pool._dispatcher._ready_workers = {101: routable}
        pool.get_ready_worker_endpoints = Mock(return_value={101: routable})
        supervisor._pool_registry = Mock()
        supervisor._pool_registry._pools = {"abc": pool}
        supervisor._pool_registry._default_pool = None

        # One committed model, resolved into the "abc" pool. The version
        # lives in parameters.version (ModelSettings has no top-level
        # version field), and the loaded-models key must match it.
        model_settings = ModelSettings(
            name="sum-model",
            implementation="tests.fixtures.SumModel",
            parameters={"version": "v1"},
        )
        instance = Mock(settings=model_settings)
        setattr(instance, ENV_HASH_ATTR, "abc")
        supervisor._loaded_models[("sum-model", "v1")] = instance
        # Nonzero committed generation: the replay must carry it (a
        # replacement holding 0 would have its real unloads skipped as
        # superseded and models would leak on workers).
        supervisor._generations[("sum-model", "v1")] = 7

        # Frontend spawn machinery.
        def make(frontend_id):
            control = Mock()
            control.recv = Mock(side_effect=EOFError)
            supervisor._controls[frontend_id] = control
            proc = Mock()
            proc.pid = 4242
            return proc

        supervisor._make_frontend_process = make  # type: ignore[method-assign]
        # One shared history: both send paths append as the calls
        # occur, so the wire order assertions cannot pass vacuously
        # through concatenation.
        sent: list[object] = []

        async def record_send_to(frontend_id, message):
            sent.append(message)

        async def record_direct(frontend_id, message, strict=False):
            sent.append(message)

        supervisor._send_to = AsyncMock(  # type: ignore[method-assign]
            side_effect=record_send_to
        )
        supervisor._send_direct = AsyncMock(  # type: ignore[method-assign]
            side_effect=record_direct
        )
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._restart_counts[0] = 0
        supervisor._schedule_restart(0)
        for _ in range(4):
            tasks = list(supervisor._restart_tasks)
            if not tasks:
                break
            await asyncio_module.wait_for(
                asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
            )

        calls = sent
        ready_events = [c for c in calls if isinstance(c, WorkerReadyEvent)]
        apply_loads = [c for c in calls if isinstance(c, ApplyLoad)]
        completions = [c for c in calls if isinstance(c, StartupComplete)]

        # Only the ready worker is announced (the quarantined worker
        # with its live endpoint is NOT — recovery re-announces it).
        assert len(ready_events) == 1
        assert ready_events[0].pid == 101
        assert ready_events[0].request_endpoint == "ipc:///tmp/w101"
        assert ready_events[0].env_hash == "abc"
        # The committed model is re-applied with its pool identity...
        assert len(apply_loads) == 1
        assert apply_loads[0].env_hash == "abc"
        # ...with its committed generation, from the real restart
        # path (not a hand-built message)...
        assert apply_loads[0].generation == 7
        # ...and with the committed version in the replayed settings
        # (the versioned-key case, exercising parameters.version).
        replayed_settings = json.loads(apply_loads[0].serialised_model_settings)
        assert replayed_settings["parameters"]["version"] == "v1"
        # ...the replacement's startup gate is released after replay...
        assert len(completions) == 1
        assert completions[0].model_count == 1
        # ...and workers come first so the fresh dispatcher can route,
        # with the gate release last.
        assert calls.index(ready_events[0]) < calls.index(apply_loads[0])
        assert calls.index(apply_loads[0]) < calls.index(completions[0])


class TestControlSendExecutorIsolation:
    """Control sends must not share the listener executor with recv.

    Every listener thread blocks indefinitely on control.recv, so once
    the listener pool is fully occupied (frontend_workers >= 8, or any
    saturated pool) a send queued behind a blocked recv livelocks:
    frontends only send their ack after receiving the message that the
    queued send is trying to deliver. Sends therefore run on the
    loop's default executor instead.
    """

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        return FrontendSupervisor(settings, [])

    async def test_send_completes_when_listener_executor_is_saturated(
        self,
    ) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        supervisor = self._supervisor()
        # Shrink the listener pool to one thread and block it, mirroring
        # a listener blocked on control.recv with every thread occupied.
        blocked = threading.Event()
        release = threading.Event()
        recv_thread: dict[str, int] = {}
        executor = ThreadPoolExecutor(max_workers=1)

        def _blocked_recv() -> None:
            recv_thread["ident"] = threading.get_ident()
            blocked.set()
            release.wait()

        supervisor._listener_executor = executor
        executor.submit(_blocked_recv)
        assert blocked.wait(2)

        sent_from: list[threading.Thread] = []

        def _record_send(message: object) -> None:
            sent_from.append(threading.current_thread())

        control = Mock()
        control.send.side_effect = _record_send
        supervisor._controls[0] = control

        try:
            # Both would time out if the send queued behind the blocked
            # listener thread instead of using the send executor.
            await asyncio.wait_for(supervisor._broadcast(object()), timeout=2)
            await asyncio.wait_for(supervisor._send_to(0, object()), timeout=2)

            assert len(sent_from) == 2
            # The sends ran off the blocked listener thread.
            assert all(t.ident != recv_thread["ident"] for t in sent_from)
            # Every send ran on the send executor's single thread.
            assert sent_from[0].ident == sent_from[1].ident
            # A missing control is a no-op, not an error.
            await supervisor._send_to(99, object())
        finally:
            release.set()
            executor.shutdown(wait=True)

    async def test_concurrent_sends_on_one_connection_are_serialized(
        self,
    ) -> None:
        """Connection.send is single-writer, not thread-safe.

        On POSIX the length header and payload are separate writes for
        larger frames (and a partial write's retry loop can interleave
        with another writer), so two threads sending on one control
        pipe corrupt its framing. Concurrent model operations (e.g.
        two loads) issue overlapping broadcasts, so the supervisor must
        serialize every send through its single send-executor thread.
        """
        import threading
        import time

        supervisor = self._supervisor()
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def slow_send(message):
            # Hold the connection long enough that a second concurrent
            # sender would overlap if sends were not serialized.
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1

        control = Mock()
        control.send.side_effect = slow_send
        supervisor._controls[0] = control

        # Three in-flight operations sharing one control connection.
        await asyncio.gather(
            supervisor._broadcast(object()),
            supervisor._send_to(0, object()),
            supervisor._broadcast(object()),
        )

        assert control.send.call_count == 3
        assert state["peak"] == 1

    async def test_stalled_connection_does_not_starve_other_frontends(self) -> None:
        """One live frontend whose control pipe stays full must block
        only its own writer: the broadcast still reaches every other
        frontend PROMPTLY (not after the stalled one's delivery
        deadline), and the stalled frontend is terminated once its
        deadline expires — the monitor's restart machinery replaces it
        (a frontend that cannot receive control traffic can never
        acknowledge an operation again)."""
        import threading
        import time
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            model_operation_timeout=1.0,
        )
        supervisor = FrontendSupervisor(settings, [])

        stalled_release = threading.Event()
        healthy = Mock()
        stalled = Mock()
        delivered_at: dict[str, float] = {}
        # Record the discipline order: the supervisor's end of the
        # stalled control connection must close BEFORE the terminate —
        # a bare terminate leaves the frontend hung on its blocked
        # (non-daemon) control reader and the slot is never restarted.
        discipline: list[str] = []

        def healthy_send(message):
            delivered_at["t"] = time.monotonic()

        def blocking_send(message):
            stalled_release.wait(timeout=10)

        healthy.send.side_effect = healthy_send
        stalled.send.side_effect = blocking_send
        stalled.close.side_effect = lambda: discipline.append("close")
        # The stalled frontend comes FIRST: a broadcast that delivered
        # sequentially would hold the healthy frontend's message behind
        # the stalled one's deadline.
        supervisor._controls = {1: stalled, 0: healthy}

        # The stalled frontend's process is alive (a real sentinel pipe
        # reads as not-exited for _process_exited).
        sentinel, _ = Pipe(duplex=False)
        proc = Mock()
        proc.is_alive.return_value = True
        proc.sentinel = sentinel
        proc.terminate = Mock(side_effect=lambda: discipline.append("terminate"))
        supervisor._frontends = {1: proc}

        try:
            message = object()
            start = time.monotonic()
            # The broadcast completes: the healthy frontend got the
            # message immediately, the stalled one hit its deadline.
            await asyncio.wait_for(supervisor._broadcast(message), timeout=5)

            healthy.send.assert_called_once_with(message)
            # Prompt delivery: well before the stalled frontend's 1s
            # delivery deadline.
            assert delivered_at["t"] - start < 0.6, (
                "healthy frontend's delivery waited behind the stalled "
                "frontend's deadline"
            )
            # The termination runs in a tracked confirmation task:
            # drain it before asserting on the calls.
            if supervisor._retire_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*supervisor._retire_tasks, return_exceptions=True),
                    timeout=5,
                )
            proc.terminate.assert_called_once()
            assert discipline == [
                "close",
                "terminate",
            ], "the stalled connection must close before terminating"
        finally:
            stalled_release.set()
            sentinel.close()

    async def test_strict_send_timeout_propagates_and_terminates(self) -> None:
        """A strict (replay) send that exceeds the delivery deadline
        raises, so the restart handler rolls its replacement back —
        and the stalled frontend is still terminated for restart."""
        import threading
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            model_operation_timeout=0.2,
        )
        supervisor = FrontendSupervisor(settings, [])

        stalled_release = threading.Event()
        stalled = Mock()
        discipline: list[str] = []

        def blocking_send(message):
            stalled_release.wait(timeout=10)

        stalled.send.side_effect = blocking_send
        stalled.close.side_effect = lambda: discipline.append("close")
        supervisor._controls = {1: stalled}

        sentinel, _ = Pipe(duplex=False)
        proc = Mock()
        proc.is_alive.return_value = True
        proc.sentinel = sentinel
        proc.terminate = Mock(side_effect=lambda: discipline.append("terminate"))
        supervisor._frontends = {1: proc}

        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    supervisor._send_direct(1, object(), strict=True), timeout=5
                )
            # The termination runs in a tracked confirmation task:
            # drain it before asserting on the calls.
            if supervisor._retire_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*supervisor._retire_tasks, return_exceptions=True),
                    timeout=5,
                )
            proc.terminate.assert_called_once()
            assert discipline == [
                "close",
                "terminate",
            ], "the stalled connection must close before terminating"
        finally:
            stalled_release.set()
            sentinel.close()


class TestFrontendRestartRetry:
    """A failed restart launch is retried for real while budget remains.

    The dead frontend's slot is absent from _frontends, so the monitor
    loop cannot retrigger it — the retry must be scheduled explicitly.
    """

    async def test_restart_scheduling_has_one_owner_per_slot(self) -> None:
        """The monitor and a failing attempt's replay handler can both
        observe the same death and schedule a retry. Without a single
        owner, both tasks read the same restart count and spawn a
        process, and the later registration overwrites (and orphans)
        the earlier. A second scheduler is a no-op; only the incumbent
        hands over to its successor."""
        supervisor = self._supervisor(max_restarts=5)
        release = asyncio.Event()

        async def blocked_restart(frontend_id: int) -> None:
            await release.wait()

        supervisor._restart_frontend = blocked_restart  # type: ignore[method-assign]
        supervisor._restart_counts[0] = 0

        # The monitor schedules the first attempt.
        supervisor._schedule_restart(0)
        first = supervisor._restart_tasks_by_frontend[0]
        assert first is not None and not first.done()

        # The replay-failure path schedules for the same slot: it must
        # not create a second owner.
        supervisor._schedule_restart(0)
        assert supervisor._restart_tasks_by_frontend[0] is first
        assert len(supervisor._restart_tasks) == 1

        # The incumbent (failing attempt) hands over to a successor.
        supervisor._schedule_restart(0, exclude=first)
        successor = supervisor._restart_tasks_by_frontend[0]
        assert successor is not first
        assert len(supervisor._restart_tasks) == 2

        # The incumbent's completion clears only its own ownership.
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert supervisor._restart_tasks_by_frontend[0] is successor

        release.set()
        await asyncio.gather(successor, return_exceptions=True)
        assert 0 not in supervisor._restart_tasks_by_frontend

    async def test_failed_attempt_does_not_retire_newer_owner(self) -> None:
        """A failing restart attempt whose failure handling is slow
        (e.g. a delivery-deadline-blocked strict send) must not retire
        a replacement a newer attempt already registered in the slot:
        the rollback is identity-guarded on the attempt's own process
        and connection."""
        supervisor = self._supervisor(max_restarts=5)

        # A newer owner has claimed slot 0.
        new_proc = Mock()
        new_proc.is_alive.return_value = False
        new_control = Mock()
        supervisor._frontends[0] = new_proc
        supervisor._controls[0] = new_control

        # The older attempt's own registrations.
        old_proc = Mock()
        old_proc.is_alive.return_value = False
        old_control = Mock()
        old_listener = asyncio.create_task(asyncio.sleep(10))
        supervisor._listener_tasks.append(old_listener)

        await supervisor._retire_partial_frontend(
            0, old_listener, proc=old_proc, control=old_control
        )

        # The newer owner's registration survives untouched.
        assert supervisor._frontends[0] is new_proc
        assert supervisor._controls[0] is new_control
        new_control.close.assert_not_called()
        # This attempt's own resources were rolled back: its listener
        # cancelled, its connection closed, its process confirmed.
        assert old_listener.cancelled()
        old_control.close.assert_called_once()
        old_proc.join.assert_called()

    @staticmethod
    def _supervisor(max_restarts: int):
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            frontend_max_restarts=max_restarts,
        )
        return FrontendSupervisor(settings, [])

    async def test_failed_restart_launch_is_retried(self, mocker):
        """A spawn failure schedules another attempt that succeeds."""
        import asyncio as asyncio_module
        from unittest.mock import AsyncMock

        supervisor = self._supervisor(max_restarts=3)
        supervisor._restart_counts[0] = 0
        attempts = []

        def failing_make(frontend_id):
            attempts.append(frontend_id)
            if len(attempts) == 1:
                raise RuntimeError("spawn failed")
            proc = Mock()
            proc.pid = 4242
            control = Mock()
            # Listener exits immediately on the closed control pipe.
            control.recv = Mock(side_effect=EOFError)
            supervisor._controls[frontend_id] = control
            return proc

        supervisor._make_frontend_process = failing_make
        # Remove the real backoff delays so the retries run promptly.
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._schedule_restart(0)
        # First attempt fails, its retry is scheduled and runs.
        for _ in range(4):
            tasks = list(supervisor._restart_tasks)
            if not tasks:
                break
            await asyncio_module.wait_for(
                asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
            )

        assert len(attempts) == 2, f"expected a real retry, attempts={attempts}"
        assert 0 in supervisor._frontends
        assert supervisor._restart_counts[0] == 2

    async def test_restart_retry_stops_at_budget(self, mocker):
        """Persistent spawn failures stop after frontend_max_restarts."""
        import asyncio as asyncio_module
        from unittest.mock import AsyncMock

        supervisor = self._supervisor(max_restarts=2)
        supervisor._restart_counts[0] = 0
        attempts = []

        def always_failing(frontend_id):
            attempts.append(frontend_id)
            raise RuntimeError("spawn failed")

        supervisor._make_frontend_process = always_failing
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._schedule_restart(0)
        for _ in range(6):
            tasks = list(supervisor._restart_tasks)
            if not tasks:
                break
            await asyncio_module.wait_for(
                asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
            )

        assert len(attempts) == 2, (
            f"budget not enforced, attempts={attempts} " f"(max_restarts=2)"
        )
        assert 0 not in supervisor._frontends


class TestPartialFrontendRetirementTracking:
    """A partially started frontend must stay owned until its exit is
    confirmed, even when the restart task is cancelled mid-retirement —
    stop() cancels restart tasks exactly then, and the popped child is
    invisible to the _frontends sweep."""

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        return FrontendSupervisor(settings, [])

    @staticmethod
    def _make_proc():
        proc = Mock(pid=4242)
        proc.is_alive.return_value = True
        return proc

    async def test_cancellation_mid_retirement_still_confirms_exit(self, mocker):
        import threading

        supervisor = self._supervisor()
        proc = self._make_proc()
        supervisor._frontends[0] = proc
        control = Mock()
        supervisor._controls[0] = control
        listener = asyncio.create_task(asyncio.sleep(100))
        supervisor._listener_tasks.append(listener)

        exited = threading.Event()
        release_join = threading.Event()
        # The child looks alive until its join completes.
        mocker.patch(
            "mlserver.frontend._process_exited",
            side_effect=lambda p: exited.is_set(),
        )
        joins = []

        def fake_join(timeout=None):
            joins.append(timeout)
            # Block the confirmation mid-join, mirroring a child that
            # takes time to exit — the cancellation lands here.
            release_join.wait(5)
            exited.set()

        proc.join = fake_join

        retire = asyncio.create_task(supervisor._retire_partial_frontend(0, listener))
        # Let the retire reach the shielded confirmation (blocked in
        # its join), then cancel it like stop() would.
        await asyncio.sleep(0.05)
        retire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retire

        # The confirmation survived the cancellation and is tracked for
        # stop()'s drain: without tracking, the child would never be
        # confirmed and could hold the shared ports past shutdown.
        tracked = list(supervisor._retire_tasks)
        assert len(tracked) == 1
        release_join.set()
        await asyncio.wait_for(asyncio.gather(*tracked), timeout=5)

        proc.terminate.assert_called_once()
        assert joins == [5]
        # The quick cleanups ran before the process handling.
        assert listener.cancelled()
        control.close.assert_called_once()
        assert 0 not in supervisor._frontends
        assert 0 not in supervisor._controls

    async def test_normal_retirement_confirms_exit(self, mocker):
        import threading

        supervisor = self._supervisor()
        proc = self._make_proc()
        supervisor._frontends[0] = proc

        exited = threading.Event()

        def fake_join(timeout=None):
            exited.set()

        proc.join = fake_join
        mocker.patch(
            "mlserver.frontend._process_exited",
            side_effect=lambda p: exited.is_set(),
        )

        await supervisor._retire_partial_frontend(0, None)

        proc.terminate.assert_called_once()
        assert not supervisor._retire_tasks

    async def test_already_exited_child_skips_terminate(self, mocker):
        supervisor = self._supervisor()
        proc = self._make_proc()
        supervisor._frontends[0] = proc
        # Already dead: the exit check short-circuits.
        proc.is_alive.return_value = False
        mocker.patch("mlserver.frontend._process_exited", return_value=True)
        joins = []
        proc.join = lambda timeout=None: joins.append(timeout)

        await supervisor._retire_partial_frontend(0, None)

        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()
        assert joins == [5]


class TestReplayOrderingWithLiveUpdates:
    """Live model updates committing during a restart replay must queue
    behind the replay and apply in commit order: a live unload must not
    be overtaken by the replayed load of the model it removed, and a
    live reload must not pair the snapshot's old instance with the new
    generation read at send time."""

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

    def _spawn_machinery(self, supervisor):
        def make(frontend_id):
            control = Mock()
            control.recv = Mock(side_effect=EOFError)
            supervisor._controls[frontend_id] = control
            proc = Mock()
            proc.pid = 4242
            # Already-exited: a failure-path retirement must not reach
            # the sentinel wait with a Mock handle.
            proc.is_alive.return_value = False
            return proc

        supervisor._make_frontend_process = make

    @staticmethod
    def _committed(supervisor, name, generation, marker):
        from mlserver.settings import ModelSettings as _ModelSettings

        instance = Mock(
            settings=_ModelSettings(
                name=name,
                implementation="tests.fixtures.SumModel",
                parameters={"extra": {"marker": marker}},
            )
        )
        supervisor._loaded_models[(name, "")] = instance
        supervisor._generations[(name, "")] = generation
        return instance

    @staticmethod
    def _load_name(message) -> str:
        return json.loads(message.serialised_model_settings)["name"]

    async def test_live_unload_during_replay_applies_after_replayed_load(
        self, mocker
    ) -> None:
        """A live unload committing mid-replay must land AFTER the
        replayed load of the same model — sending it directly would let
        the stale replayed load resurrect a model the pool removed."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad, ApplyUnload, StartupComplete

        supervisor = self._supervisor()
        self._spawn_machinery(supervisor)
        supervisor._pool_registry = None  # skip the worker replay
        self._committed(supervisor, "ma", 1, "old")
        self._committed(supervisor, "mb", 1, "old")

        sent: list[object] = []
        live_fired = asyncio.Event()

        async def fake_direct(
            frontend_id: int, message: object, strict: bool = False
        ) -> None:
            sent.append(message)
            if isinstance(message, ApplyLoad) and not live_fired.is_set():
                live_fired.set()
                # Mid-replay: ma's load is in the send executor and the
                # loop is yielded — a live unload of mb commits now.
                await supervisor._broadcast(
                    ApplyUnload(
                        op_id="live",
                        name="mb",
                        version="",
                        ok=True,
                        generation=2,
                    )
                )

        supervisor._send_direct = AsyncMock(side_effect=fake_direct)
        supervisor._send_to = AsyncMock()
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._restart_counts[0] = 0
        supervisor._schedule_restart(0)
        tasks = list(supervisor._restart_tasks)
        await asyncio_module.wait_for(
            asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
        )

        load_mb = next(
            m for m in sent if isinstance(m, ApplyLoad) and self._load_name(m) == "mb"
        )
        unload_mb = next(
            m for m in sent if isinstance(m, ApplyUnload) and m.name == "mb"
        )
        # The replayed load precedes the live unload: the frontend ends
        # up with mb unloaded, matching the pool.
        assert sent.index(unload_mb) > sent.index(load_mb)
        # The queue drained and the replay state cleared.
        assert not supervisor._replay_pending
        assert 0 not in supervisor._replaying
        # The gate release was delivered after everything else.
        assert any(isinstance(m, StartupComplete) for m in sent)
        assert sent.index(next(m for m in sent if isinstance(m, StartupComplete))) > (
            sent.index(unload_mb)
        )

    async def test_live_reload_pairs_snapshot_instance_with_snapshot_generation(
        self, mocker
    ) -> None:
        """The snapshot captures instance and generation together: a
        reload committing mid-replay must not leave the replay sending
        the OLD instance stamped with the NEW generation."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad
        from mlserver.parallel.shared import serialise_model_settings

        supervisor = self._supervisor()
        self._spawn_machinery(supervisor)
        supervisor._pool_registry = None
        # ma seeds BEFORE mb so the live reload fires during ma's send
        # — BEFORE mb's replay message is built. A regression that
        # reads mb's generation at construction time (instead of the
        # snapshot) would then pick up the reloaded generation 2.
        self._committed(supervisor, "ma", 1, "old")
        self._committed(supervisor, "mb", 1, "old")

        sent: list[object] = []
        fired = asyncio.Event()

        async def fake_direct(
            frontend_id: int, message: object, strict: bool = False
        ) -> None:
            sent.append(message)
            if isinstance(message, ApplyLoad) and not fired.is_set():
                fired.set()
                # During ma's send: a live reload of mb commits a new
                # instance at generation 2.
                new_instance = self._committed(supervisor, "mb", 2, "new")
                await supervisor._broadcast(
                    ApplyLoad(
                        op_id="live",
                        serialised_model_settings=serialise_model_settings(
                            new_instance.settings
                        ),
                        ok=True,
                        env_hash=None,
                        generation=2,
                    )
                )

        supervisor._send_direct = AsyncMock(side_effect=fake_direct)
        supervisor._send_to = AsyncMock()
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._restart_counts[0] = 0
        supervisor._schedule_restart(0)
        tasks = list(supervisor._restart_tasks)
        await asyncio_module.wait_for(
            asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
        )

        loads = [m for m in sent if isinstance(m, ApplyLoad)]
        # ma's replayed load is the trigger point.
        assert self._load_name(loads[0]) == "ma"
        # mb's replayed load is built AFTER the reload committed, yet
        # must carry the SNAPSHOT pair: old marker, old generation.
        assert self._load_name(loads[1]) == "mb"
        assert loads[1].generation == 1
        assert '"old"' in loads[1].serialised_model_settings or "old" in str(
            loads[1].serialised_model_settings
        )
        # The queued live reload applied after it: new generation, new
        # marker.
        assert loads[2].generation == 2
        assert "new" in str(loads[2].serialised_model_settings)
        assert sent.index(loads[2]) > sent.index(loads[1])

    async def test_failed_replay_drops_queued_live_updates(self, mocker) -> None:
        """A replay failure retires the replacement: queued live updates
        are dropped (they were destined for a dying pipe), and the
        replay state is cleared for the next attempt."""
        import asyncio as asyncio_module

        from mlserver.parallel.shared import ApplyLoad, ApplyUnload

        supervisor = self._supervisor()
        # One attempt, no retry loop: budget 1 admits the first try
        # (0 < 1) and blocks the post-failure retry (1 < 1 is false).
        # A budget of 0 would make _schedule_restart return without
        # spawning anything, leaving every assertion below vacuous.
        supervisor._settings.frontend_max_restarts = 1
        self._spawn_machinery(supervisor)
        supervisor._pool_registry = None
        self._committed(supervisor, "mb", 1, "old")

        sent: list[object] = []

        async def failing_direct(
            frontend_id: int, message: object, strict: bool = False
        ) -> None:
            if isinstance(message, ApplyLoad):
                # A live update queues mid-replay, then the replay dies.
                await supervisor._broadcast(
                    ApplyUnload(
                        op_id="live",
                        name="mb",
                        version="",
                        ok=True,
                        generation=2,
                    )
                )
                raise RuntimeError("replay send failed")
            sent.append(message)

        supervisor._send_direct = AsyncMock(side_effect=failing_direct)
        supervisor._send_to = AsyncMock()
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._restart_counts[0] = 0
        supervisor._schedule_restart(0)
        tasks = list(supervisor._restart_tasks)
        assert tasks, "a restart attempt must actually run"
        results = await asyncio_module.wait_for(
            asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
        )
        # The replay error is handled inside the restart (logged, rolled
        # back, no retry under the exhausted budget): the task completes.
        assert all(not isinstance(r, BaseException) for r in results)
        # The failing send path actually executed.
        assert supervisor._send_direct.await_count >= 1

        # The queued live update was dropped, not sent.
        assert not any(isinstance(m, ApplyUnload) for m in sent)
        assert not supervisor._replay_pending
        assert 0 not in supervisor._replaying
        # The failed replacement was rolled back.
        assert 0 not in supervisor._frontends

    async def test_replay_write_failure_rolls_back_replacement(self, mocker) -> None:
        """A control write that fails during the replay must propagate
        and roll the replacement back: a live-but-unreadable child would
        otherwise keep its slot with a partial inventory and a readiness
        gate that nothing opens (the monitor only detects death)."""
        import asyncio as asyncio_module

        supervisor = self._supervisor()
        supervisor._settings.frontend_max_restarts = 1  # one attempt, no retry
        supervisor._pool_registry = None
        self._committed(supervisor, "mb", 1, "old")

        # Real _send_direct: the CONTROL's write fails, so only the
        # strict propagation can surface it. The control is created
        # during the restart's spawn, so the failure is installed in
        # the spawn machinery itself.
        def make(frontend_id):
            control = Mock()
            control.recv = Mock(side_effect=EOFError)
            control.send = Mock(side_effect=OSError("broken pipe"))
            supervisor._controls[frontend_id] = control
            proc = Mock()
            proc.pid = 4242
            proc.is_alive.return_value = False
            return proc

        supervisor._make_frontend_process = make
        supervisor._send_to = AsyncMock()

        retire = AsyncMock()
        supervisor._retire_partial_frontend = retire
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        supervisor._restart_counts[0] = 0
        supervisor._schedule_restart(0)
        tasks = list(supervisor._restart_tasks)
        assert tasks, "a restart attempt must actually run"
        results = await asyncio_module.wait_for(
            asyncio_module.gather(*tasks, return_exceptions=True), timeout=5
        )
        assert all(not isinstance(r, BaseException) for r in results)

        # The write failure propagated to the restart handler and the
        # replacement was rolled back instead of retained.
        retire.assert_awaited_once()
        assert 0 not in supervisor._replaying
        assert not supervisor._replay_pending

    async def test_stale_broadcast_not_queued_for_replacement(self) -> None:
        """A slot replaced mid-broadcast never receives the in-flight
        (older-generation) message on its NEW connection: the capture
        loop is synchronous (no await between capturing a connection
        and deciding its branch), so the message goes to the captured
        OLD connection — where a dead pipe fails harmlessly — and the
        replacement's replay snapshot remains the only thing that
        reaches the new connection. The slot-ownership guard keeps
        this true even if an await is ever reintroduced into the
        capture loop."""
        supervisor = self._supervisor()

        old_conn = Mock()
        new_conn = Mock()
        supervisor._controls = {0: Mock(), 1: old_conn}
        supervisor._replaying = set()
        supervisor._replay_pending = {}

        def send_on_first(message):
            # Frontend 0's send runs on its writer thread; while the
            # broadcast's sends are in flight, slot 1 dies and its
            # replacement (a new connection) starts replaying.
            supervisor._controls[1] = new_conn
            supervisor._replaying.add(1)

        supervisor._controls[0].send = send_on_first
        old_conn.send = Mock()
        new_conn.send = Mock()

        message = object()
        await supervisor._broadcast(message)

        # The in-flight message reached only the captured (old)
        # connection's writer; the replacement's connection never saw
        # it, and nothing was queued behind its replay.
        new_conn.send.assert_not_called()
        assert old_conn.send.call_count == 1
        assert 1 not in supervisor._replay_pending

        # Positive case: a replaying frontend whose connection still
        # owns the slot still gets live updates queued behind its
        # replay (post-snapshot traffic must apply after the snapshot).
        old_conn.send.reset_mock()
        supervisor._replaying = {1}
        supervisor._controls = {1: old_conn}
        supervisor._replay_pending = {}
        await supervisor._broadcast(message)
        assert supervisor._replay_pending[1] == [message]
        old_conn.send.assert_not_called()


class TestStartupApplyFailureKeepsGateClosed:
    """A frontend whose startup model apply fails must not report
    ready: the readiness gate stays closed even after StartupComplete
    (with the default empty_registry_readiness=True, releasing it would
    serve traffic without the model)."""

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
            reply_endpoint="ipc:///tmp/test-startup-fail",
            control=child,
        )
        registry.attach_registry(Mock())
        registry.attach_data_plane(Mock())
        yield registry
        parent.close()
        child.close()

    async def test_failed_apply_keeps_gate_closed(self, shared):
        """A startup apply failure prevents the readiness release."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            StartupComplete,
            serialise_model_settings,
        )

        settings = ModelSettings(
            name="startup-model", implementation="tests.fixtures.SumModel"
        )

        # The local registry load fails (e.g. the model artefact is
        # corrupt on this frontend). The peer apply entry point is
        # _load_for_apply — mock THAT, and assert it was awaited so
        # the failure path is genuinely exercised.
        shared._model_registry._load_for_apply = AsyncMock(
            side_effect=RuntimeError("model artefact corrupt")
        )
        try:
            await handle_and_settle(
                shared,
                ApplyLoad(
                    op_id="op-startup",
                    serialised_model_settings=serialise_model_settings(settings),
                    ok=True,
                ),
            )
        except RuntimeError:
            pass  # _listen would log and continue

        # StartupComplete arrives; the gate must NOT release.
        await handle_and_settle(shared, StartupComplete(model_count=1))
        shared._data_plane.startup_complete.assert_not_called()
        shared._model_registry._load_for_apply.assert_awaited_once()
        assert shared.startup_models_pending is False  # startup is done
        assert shared._startup_apply_failed is True

    async def test_successful_apply_releases_gate(self, shared):
        """A successful startup apply releases the gate normally."""
        from mlserver.parallel.shared import (
            ApplyLoad,
            StartupComplete,
            serialise_model_settings,
        )

        settings = ModelSettings(
            name="startup-model", implementation="tests.fixtures.SumModel"
        )
        shared._model_registry._load_for_apply = AsyncMock()
        await handle_and_settle(
            shared,
            ApplyLoad(
                op_id="op-startup",
                serialised_model_settings=serialise_model_settings(settings),
                ok=True,
            ),
        )
        await handle_and_settle(shared, StartupComplete(model_count=1))
        shared._model_registry._load_for_apply.assert_awaited_once()
        shared._data_plane.startup_complete.assert_called_once()


class TestStopClosesControlsBeforeJoin:
    """The supervisor closes its control ends before terminating and
    joining frontends: a frontend's control reader only observes EOF
    when the peer closes, and that is what lets its executor thread
    finish and the process exit within the grace period."""

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

    async def test_controls_close_before_terminate(self) -> None:
        import os as os_module

        supervisor = self._supervisor()

        calls: list[str] = []
        proc = Mock()
        proc.is_alive.return_value = True
        # A real (never-readable) pipe fd as the sentinel: stop() checks
        # _process_exited, which consults the sentinel after is_alive() —
        # a Mock attribute would break the selector registration.
        sentinel_r, sentinel_w = os_module.pipe()
        proc.sentinel = sentinel_r
        proc.terminate.side_effect = lambda: calls.append("terminate")
        proc.join.side_effect = lambda timeout=None: calls.append("join")
        control = Mock()
        control.close.side_effect = lambda: calls.append("close")
        supervisor._frontends = {0: proc}
        supervisor._controls = {0: control}

        try:
            await supervisor.stop()
        finally:
            os_module.close(sentinel_r)
            os_module.close(sentinel_w)

        assert "close" in calls and "terminate" in calls
        assert calls.index("close") < calls.index(
            "terminate"
        ), f"controls must close before terminate/join, got {calls}"

    async def test_pool_stop_failure_retains_owners_and_raises(self) -> None:
        """A failed pool close keeps the registry and the IPC directory
        (the registry still owns the workers), and the error surfaces
        to the caller instead of reporting a clean shutdown. The
        failure is terminal: a second stop() short-circuits on
        _closing (no retry promise)."""
        import os as os_module
        import tempfile

        supervisor = self._supervisor()
        supervisor._frontends = {}
        supervisor._controls = {}
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.close = AsyncMock(
            side_effect=RuntimeError("worker did not confirm exit")
        )
        transport = tempfile.TemporaryDirectory(prefix="mlserver-test-ipc-")
        supervisor._transport_directory = transport
        transport_path = transport.name

        with pytest.raises(RuntimeError, match="worker did not confirm exit"):
            await supervisor.stop()

        # Both owners retained for operator inspection.
        assert supervisor._pool_registry is not None
        assert supervisor._transport_directory is not None
        assert os_module.path.isdir(transport_path)

        # A second stop() short-circuits on _closing: the failure is
        # terminal, not a retry promise (the registry's internal close
        # task is in a failed state).
        supervisor._pool_registry.close = AsyncMock(return_value=None)
        await supervisor.stop()
        # The mock was NOT called — _closing prevented re-entry.
        supervisor._pool_registry.close.assert_not_awaited()

    async def test_pool_stop_success_cleans_everything(self) -> None:
        """A clean pool close releases the registry and the IPC
        directory and stop() returns normally."""
        import os as os_module
        import tempfile

        supervisor = self._supervisor()
        supervisor._frontends = {}
        supervisor._controls = {}
        supervisor._pool_registry = Mock()
        supervisor._pool_registry.close = AsyncMock(return_value=None)
        transport = tempfile.TemporaryDirectory(prefix="mlserver-test-ipc-")
        supervisor._transport_directory = transport
        transport_path = transport.name

        await supervisor.stop()

        assert supervisor._pool_registry is None
        assert supervisor._transport_directory is None
        assert not os_module.path.isdir(transport_path)


class TestStartupFailurePropagation:
    """A failed startup model fails the supervisor start, like the
    single-process server whose startup gather raises — frontends must
    not report ready with the default empty_registry_readiness while a
    startup model failed to load."""

    @staticmethod
    def _supervisor(models: list):
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
        supervisor = FrontendSupervisor(settings, models)
        # Normally created by _prepare_shared_resources during start().
        supervisor._reply_endpoints = []
        return supervisor

    @staticmethod
    def _model_settings(name: str) -> "ModelSettings":
        return ModelSettings(name=name, implementation="tests.fixtures.SumModel")

    async def test_startup_model_failure_raises(self, mocker):
        """_start_shared_pool raises when any startup model fails, and
        no StartupComplete is broadcast (frontends never get gated
        ready over a failed inventory)."""
        from mlserver.parallel.shared import ApplyLoad, StartupComplete

        supervisor = self._supervisor(
            [self._model_settings("good-model"), self._model_settings("bad-model")]
        )
        supervisor._broadcast = AsyncMock()

        good = Mock()
        pool_registry = Mock()
        pool_registry.start = AsyncMock()
        pool_registry.model_initialiser.side_effect = lambda settings: (
            good if settings.name == "good-model" else Mock()
        )

        async def load_model(instance):
            if instance is good:
                return instance
            raise RuntimeError("model artefact missing")

        pool_registry.load_model = AsyncMock(side_effect=load_model)
        mocker.patch(
            "mlserver.frontend.InferencePoolRegistry", return_value=pool_registry
        )

        with pytest.raises(RuntimeError, match="Startup model loads failed"):
            await supervisor._start_shared_pool()

        # Every load was attempted (diagnostics over fail-fast), and no
        # readiness release was broadcast.
        assert supervisor._pool_registry.load_model.await_count == 2
        broadcasts = [c.args[0] for c in supervisor._broadcast.call_args_list]
        assert not any(isinstance(b, StartupComplete) for b in broadcasts)
        # The failed load still broadcast its apply (peers ack without
        # registering anything for a failed load).
        assert any(isinstance(b, ApplyLoad) and not b.ok for b in broadcasts)

    async def test_startup_failure_error_sanitised(self, mocker):
        """The startup RuntimeError must not echo exception text: it
        escapes `mlserver start` to container stderr, and a failing
        runtime can embed model-store credentials in its message
        (CWE-532). The model name and exception class stay; the full
        detail is only logged at debug level."""
        supervisor = self._supervisor([self._model_settings("leaky-model")])
        supervisor._broadcast = AsyncMock()

        pool_registry = Mock()
        pool_registry.start = AsyncMock()
        pool_registry.model_initialiser.return_value = Mock()
        credential = "AKIAIOSFODNN7EXAMPLE"
        pool_registry.load_model = AsyncMock(
            side_effect=RuntimeError(
                f"failed fetching s3://{credential}:secret@bucket/model.onnx"
            )
        )
        mocker.patch(
            "mlserver.frontend.InferencePoolRegistry", return_value=pool_registry
        )

        with pytest.raises(RuntimeError) as exc_info:
            await supervisor._start_shared_pool()

        message = str(exc_info.value)
        assert "leaky-model" in message
        assert "RuntimeError" in message
        # The credential never reaches the escaping error text.
        assert credential not in message
        assert "secret@bucket" not in message

    async def test_startup_complete_only_after_all_models_load(self, mocker):
        """All startup models loading broadcasts StartupComplete once,
        after the applies."""
        from mlserver.parallel.shared import ApplyLoad, StartupComplete

        supervisor = self._supervisor([self._model_settings("only-model")])
        supervisor._broadcast = AsyncMock()
        pool_registry = Mock()
        pool_registry.start = AsyncMock()
        instance = Mock()
        pool_registry.model_initialiser.return_value = instance
        pool_registry.load_model = AsyncMock(return_value=instance)
        mocker.patch(
            "mlserver.frontend.InferencePoolRegistry", return_value=pool_registry
        )

        await supervisor._start_shared_pool()

        broadcasts = [c.args[0] for c in supervisor._broadcast.call_args_list]
        applies = [b for b in broadcasts if isinstance(b, ApplyLoad)]
        completions = [b for b in broadcasts if isinstance(b, StartupComplete)]
        assert len(applies) == 1 and applies[0].ok
        assert len(completions) == 1
        assert broadcasts.index(applies[0]) < broadcasts.index(completions[0])


class TestRestartCancellationRollback:
    """Shutdown cancels restart tasks mid-flight: after registration
    the partial replacement must roll back in the cancellation handler
    itself, not wait for stop()'s sweep over _frontends."""

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

    async def test_cancelled_restart_rolls_back_partial_replacement(
        self, mocker
    ) -> None:
        from mlserver.frontend import FrontendSupervisor  # noqa: F401

        supervisor = self._supervisor()
        made: dict[str, Mock] = {}
        replay_blocked = asyncio.Event()
        replay_reached = asyncio.Event()

        def make(frontend_id):
            control = Mock()
            control.recv = Mock(side_effect=EOFError)
            supervisor._controls[frontend_id] = control
            proc = Mock()
            proc.pid = 4242
            proc.is_alive.return_value = False
            made["proc"] = proc
            return proc

        supervisor._make_frontend_process = make  # type: ignore[method-assign]
        supervisor._pool_registry = None  # skip the worker replay
        mocker.patch("mlserver.frontend.asyncio.sleep", new=AsyncMock())

        async def blocked_replay_send(frontend_id, message, strict=False):
            replay_reached.set()
            await replay_blocked.wait()

        supervisor._send_direct = blocked_replay_send  # type: ignore[method-assign]

        task = asyncio.create_task(supervisor._restart_frontend(0))
        await asyncio.wait_for(replay_reached.wait(), timeout=5)
        assert 0 in supervisor._frontends

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The partial replacement was rolled back in the cancellation
        # handler: the slot and its control are empty, the listener is
        # gone, and the child's exit was confirmed.
        assert 0 not in supervisor._frontends
        assert 0 not in supervisor._controls
        assert not supervisor._listener_tasks
        assert made["proc"].join.called
        # The replay state was cleared by the finally block.
        assert 0 not in supervisor._replaying


class TestStopSettlesOpTasks:
    """Model operations spawned from control messages are tracked and
    settled by stop(): an op blocked on a pool call must not outlive
    the control pipes."""

    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            model_operation_timeout=5,
        )
        return FrontendSupervisor(settings, [])

    async def test_op_task_tracked_and_settled_by_stop(self) -> None:
        from mlserver.parallel.shared import OpRequest

        supervisor = self._supervisor()
        op_started = asyncio.Event()

        async def blocked_op(frontend_id, message):
            op_started.set()
            await asyncio.Event().wait()

        supervisor._handle_op = blocked_op  # type: ignore[method-assign]

        await supervisor._handle_control(0, OpRequest(op_id="op-1", kind="load"))
        assert len(supervisor._op_tasks) == 1
        await asyncio.wait_for(op_started.wait(), timeout=5)

        await supervisor.stop()

        # The blocked op was cancelled and awaited, not abandoned.
        assert not supervisor._op_tasks

    async def test_completed_op_task_discarded(self) -> None:
        from mlserver.parallel.shared import OpRequest

        supervisor = self._supervisor()

        async def quick_op(frontend_id, message):
            return None

        supervisor._handle_op = quick_op  # type: ignore[method-assign]

        await supervisor._handle_control(0, OpRequest(op_id="op-2", kind="load"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not supervisor._op_tasks


class TestStartupApplyFailedHandling:
    """A frontend reporting a failed startup apply is terminated for
    restart: it would otherwise linger alive-but-never-ready, still
    receiving SO_REUSEPORT traffic it cannot serve."""

    async def test_report_terminates_the_frontend(self) -> None:
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import StartupApplyFailed

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        supervisor = FrontendSupervisor(settings, [])

        sentinel, _ = Pipe(duplex=False)
        proc = Mock()
        proc.is_alive.return_value = True
        proc.sentinel = sentinel
        proc.terminate = Mock()
        supervisor._frontends = {0: proc}
        # The slot's control connection: the termination discipline
        # must close the supervisor's end before terminating, or the
        # SIGTERMed frontend hangs on its blocked control reader and
        # the slot is never restarted.
        control = Mock()
        supervisor._controls = {0: control}

        await supervisor._handle_control(0, StartupApplyFailed(model_name="m"))

        # The termination runs in a tracked confirmation task: drain
        # it before asserting on the calls.
        if supervisor._retire_tasks:
            await asyncio.wait_for(
                asyncio.gather(*supervisor._retire_tasks, return_exceptions=True),
                timeout=5,
            )
        proc.terminate.assert_called_once()
        control.close.assert_called_once()
        assert 0 not in supervisor._controls
        sentinel.close()

    async def test_report_from_dead_frontend_is_a_no_op(self) -> None:
        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import StartupApplyFailed

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        supervisor = FrontendSupervisor(settings, [])

        await supervisor._handle_control(99, StartupApplyFailed(model_name="m"))

    async def test_stale_report_cannot_terminate_replacement(self) -> None:
        """A StartupApplyFailed arriving on a replaced slot's old
        connection must not terminate the replacement: the frontend
        died right after reporting, the monitor restarted the slot,
        and the listener resumed with the already-received message —
        the frontend_id lookup alone would return the new process."""
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor
        from mlserver.parallel.shared import StartupApplyFailed

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        supervisor = FrontendSupervisor(settings, [])

        # The replacement (current slot occupant).
        new_sentinel, _ = Pipe(duplex=False)
        new_proc = Mock()
        new_proc.is_alive.return_value = True
        new_proc.sentinel = new_sentinel
        new_proc.terminate = Mock()
        supervisor._frontends = {0: new_proc}
        new_control = Mock()
        supervisor._controls = {0: new_control}

        # The report arrives on the OLD frontend's connection (the one
        # its listener captured before the slot was restarted).
        old_control = Mock()
        await supervisor._handle_control(
            0, StartupApplyFailed(model_name="m"), control=old_control
        )

        # The replacement is untouched: not terminated, its control
        # not closed. The stale report is a no-op.
        new_proc.terminate.assert_not_called()
        new_control.close.assert_not_called()
        assert supervisor._controls[0] is new_control
        new_sentinel.close()

    async def test_stalled_send_to_replaced_slot_spares_replacement(self) -> None:
        """A send stalled on a superseded connection must terminate
        nothing: the slot's replacement owns it now, and the stalled
        target's exit is the replacer's responsibility."""
        import threading
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor

        settings = Settings(
            http_port=0,
            grpc_port=0,
            metrics_port=0,
            model_operation_timeout=0.2,
        )
        supervisor = FrontendSupervisor(settings, [])

        stalled_release = threading.Event()

        def blocking_send(message):
            stalled_release.wait(timeout=10)

        old_control = Mock()
        old_control.send.side_effect = blocking_send

        # The slot was replaced after the send was queued: the current
        # occupant is a new process with a new connection.
        new_control = Mock()
        supervisor._controls = {1: new_control}
        new_sentinel, _ = Pipe(duplex=False)
        new_proc = Mock()
        new_proc.is_alive.return_value = True
        new_proc.sentinel = new_sentinel
        new_proc.terminate = Mock()
        supervisor._frontends = {1: new_proc}

        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    supervisor._send_on_connection(
                        1, old_control, object(), strict=True
                    ),
                    timeout=5,
                )
            if supervisor._retire_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*supervisor._retire_tasks, return_exceptions=True),
                    timeout=5,
                )
            new_proc.terminate.assert_not_called()
            new_control.close.assert_not_called()
            assert supervisor._controls[1] is new_control
        finally:
            stalled_release.set()
            new_sentinel.close()

    def test_close_unstarted_control_pops_and_closes(self) -> None:
        """A factory-registered connection whose start() failed is
        popped and closed immediately: sends to the slot fail fast
        instead of buffering into the unread pipe, and the fds are
        released deterministically."""
        from mlserver.frontend import FrontendSupervisor

        supervisor = FrontendSupervisor(
            Settings(http_port=0, grpc_port=0, metrics_port=0), []
        )
        control = Mock()
        supervisor._controls = {0: control}

        supervisor._close_unstarted_control(0)

        control.close.assert_called_once()
        assert 0 not in supervisor._controls
        # Idempotent on an absent slot.
        supervisor._close_unstarted_control(0)

    async def test_dead_frontend_ack_notify_is_tracked(self) -> None:
        """The pending-ack notification scheduled by death handling is
        tracked and settles: an untracked fire-and-forget task is how
        shutdown leaks start (a notify lost to a cancelled task leaves
        ack collectors waiting their full timeout)."""
        from mlserver.frontend import FrontendSupervisor

        supervisor = FrontendSupervisor(
            Settings(http_port=0, grpc_port=0, metrics_port=0), []
        )
        supervisor._schedule_restart = Mock()  # type: ignore[method-assign]
        supervisor._frontends = {1: Mock()}

        supervisor._handle_dead_frontend(1, Mock(pid=111, exitcode=-9))

        assert supervisor._ack_notify_tasks
        await asyncio.wait_for(
            asyncio.gather(*supervisor._ack_notify_tasks, return_exceptions=True),
            timeout=5,
        )
        assert not supervisor._ack_notify_tasks

    async def test_confirm_exit_tolerates_reaped_process(self) -> None:
        """A child reaped between the exit check and the signal raises
        ProcessLookupError from send_signal: the confirmation must
        complete (the child is gone) instead of crashing the retire
        task."""
        from multiprocessing import Pipe

        from mlserver.frontend import FrontendSupervisor

        supervisor = FrontendSupervisor(
            Settings(http_port=0, grpc_port=0, metrics_port=0), []
        )
        sentinel, _ = Pipe(duplex=False)
        proc = Mock()
        proc.sentinel = sentinel
        proc.terminate = Mock(side_effect=ProcessLookupError())
        proc.kill = Mock(side_effect=ProcessLookupError())
        proc.join = Mock()

        await supervisor._confirm_frontend_exit(0, proc)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        sentinel.close()


class TestReplayPendingWatermark:
    @staticmethod
    def _supervisor():
        from mlserver.frontend import FrontendSupervisor

        settings = Settings(http_port=0, grpc_port=0, metrics_port=0)
        return FrontendSupervisor(settings, [])

    async def test_watermark_warns_on_crossing(self, mocker, caplog) -> None:
        """The replay queue only grows during an in-flight replay, but
        unbounded growth means a stuck replay — crossing the watermark
        (each multiple) must log a warning so the condition is visible
        instead of silently absorbing memory."""
        import logging

        supervisor = self._supervisor()
        # Small watermark so the test crosses it twice.
        mocker.patch("mlserver.frontend._REPLAY_PENDING_WATERMARK", 4)
        supervisor._replaying = {0}
        supervisor._controls[0] = Mock()

        with caplog.at_level(logging.WARNING, logger="mlserver.frontend"):
            for i in range(8):
                await supervisor._send_to(0, f"msg-{i}")

        # All messages queued, in order.
        assert supervisor._replay_pending[0] == [f"msg-{i}" for i in range(8)]
        warnings = [
            r for r in caplog.records if "Replay queue for frontend 0" in r.message
        ]
        # Crossings at 4 and 8 pending messages: two warnings.
        assert len(warnings) == 2, [r.message for r in caplog.records]
        assert "reached 8 pending messages" in warnings[-1].getMessage()

    async def test_no_warning_below_watermark(self, mocker, caplog) -> None:
        """Below the watermark the queue stays silent (normal replay
        traffic must not log)."""
        import logging

        supervisor = self._supervisor()
        mocker.patch("mlserver.frontend._REPLAY_PENDING_WATERMARK", 100)
        supervisor._replaying = {0}
        supervisor._controls[0] = Mock()

        with caplog.at_level(logging.WARNING, logger="mlserver.frontend"):
            for i in range(10):
                await supervisor._send_to(0, f"msg-{i}")

        assert len(supervisor._replay_pending[0]) == 10
        assert not [r for r in caplog.records if "Replay queue" in r.message]


class TestCollectAcksMidCollectionDeath:
    """The ack-collection wait with a frontend dying mid-collection.

    One ack already landed (in flight), the second frontend dies while
    the collector waits: the death path must synthetic-ack every
    pending op and notify, completing the collection immediately —
    not via the model-operation deadline."""

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
            # Generous: completion must come from the synthetic ack,
            # well before this deadline (the discriminator between
            # the two outcomes).
            model_operation_timeout=10,
        )
        return FrontendSupervisor(settings, [])

    async def test_death_mid_collection_completes_via_synthetic_ack(
        self,
    ) -> None:
        import asyncio as asyncio_module
        from unittest.mock import Mock

        from mlserver.frontend import _PendingOp

        supervisor = self._supervisor()
        supervisor._frontends[0] = Mock(pid=101, exitcode=-9)
        supervisor._frontends[1] = Mock(pid=102, exitcode=-9)
        supervisor._controls[0] = Mock()
        supervisor._controls[1] = Mock()
        supervisor._pool_registry = None
        # The death path schedules a replacement: neutralize it — this
        # test targets the ack-collection path, not the restart (which
        # would spawn a real frontend process).
        supervisor._schedule_restart = Mock()

        # The pending op, registered before the broadcast (as both
        # callers do) so early acks are captured.
        pending = _PendingOp(origin=0, ok=True, error=None)
        supervisor._pending_ops["op-mid"] = pending

        collector = asyncio_module.create_task(
            supervisor._collect_acks(
                "op-mid", origin=0, ok=True, error=None, expected={0, 1}
            )
        )
        # Let the collector enter the condition wait.
        await asyncio_module.sleep(0)

        # Frontend 0's ACK lands mid-collection (the listener path's
        # effect on the shared pending op).
        async with supervisor._ack_condition:
            pending.acks.add(0)
            supervisor._ack_condition.notify_all()

        # Still waiting on frontend 1 — the collection must not have
        # completed with only one of two expected acks.
        await asyncio_module.sleep(0.05)
        assert not collector.done()

        # Frontend 1 dies mid-collection: the death path synthetic-acks
        # every pending op and notifies the condition.
        supervisor._handle_dead_frontend(1, supervisor._frontends[1])

        # Completes via the synthetic ack — far before the 10 s
        # deadline (a timeout completion would take the full window).
        await asyncio_module.wait_for(collector, timeout=2)
        assert pending.acks == {0, 1}
        assert "op-mid" not in supervisor._pending_ops
