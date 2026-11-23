"""The shared REST listener: one socket, load-aware cooperative accept.

Unit coverage for the socket helpers and the accept coordination
(publish, defer, not-accepting sentinel), plus full-topology coverage
for distribution fairness, frontend-death survival and slot resets
through the real supervisor.
"""

import asyncio
import signal
import socket as socket_module
import time
from pathlib import Path

import pytest

from mlserver.settings import Settings, ModelSettings
from mlserver.utils import _shared_listening_sockets

from tests.utils import get_available_ports

_NOT_ACCEPTING = 2**31 - 1
READY_TIMEOUT = 120.0
INFER_TIMEOUT = 30.0


async def _wait_ready(base_url: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = asyncio.get_event_loop().time() + READY_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await client.get(f"{base_url}/v2/health/ready")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError("frontends did not become ready in time")


async def _infer(base_url: str, data: list[int]) -> int:
    import httpx

    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, len(data)],
                "datatype": "INT32",
                "data": data,
            }
        ]
    }
    async with httpx.AsyncClient(timeout=INFER_TIMEOUT) as client:
        response = await client.post(
            f"{base_url}/v2/models/sum-model/infer", json=payload
        )
        response.raise_for_status()
        return response.json()["outputs"][0]["data"][0]


class TestSharedListeningSockets:
    """_shared_listening_sockets: one socket set, no SO_REUSEPORT."""

    def test_sockets_listen(self):
        """Each returned socket is bound and listening."""
        socks = _shared_listening_sockets("127.0.0.1", 0)
        try:
            for sock in socks:
                addr = sock.getsockname()
                assert addr[1] > 0
                # A connect succeeds against a listening socket.
                client = socket_module.create_connection(addr, timeout=5)
                client.close()
        finally:
            for sock in socks:
                sock.close()

    def test_no_reuse_port(self):
        """SO_REUSEPORT is NOT set: a second bind of the same port must
        fail (the whole point — exactly one accept queue)."""
        socks = _shared_listening_sockets("127.0.0.1", 0)
        try:
            port = socks[0].getsockname()[1]
            second = socket_module.socket(
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                proto=socket_module.IPPROTO_TCP,
            )
            second.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            second.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEPORT, 1)
            try:
                with pytest.raises(OSError):
                    second.bind(("127.0.0.1", port))
            finally:
                second.close()
        finally:
            for sock in socks:
                sock.close()

    def test_proto_is_tcp(self):
        """The sockets carry IPPROTO_TCP (accepted sockets inherit the
        listener's proto; asyncio's nodelay only fires for TCP-proto)."""
        socks = _shared_listening_sockets("127.0.0.1", 0)
        try:
            assert all(sock.proto == socket_module.IPPROTO_TCP for sock in socks)
        finally:
            for sock in socks:
                sock.close()

    def test_shared_socket_one_accept_queue(self):
        """Two holders of the same socket (as the frontends hold their
        fd dups) see ONE accept queue: a connection made to the
        listener can be accepted through an fd dup."""
        import os

        socks = _shared_listening_sockets("127.0.0.1", 0)
        try:
            addr = socks[0].getsockname()
            client = socket_module.create_connection(addr, timeout=5)
            dup = socket_module.socket(
                fileno=os.dup(socks[0].fileno()),
                family=socks[0].family,
                type=socks[0].type,
                proto=socks[0].proto,
            )
            try:
                dup.settimeout(5)
                conn, _ = dup.accept()
                conn.close()
            finally:
                dup.close()
            client.close()
        finally:
            for sock in socks:
                sock.close()


class TestNoSignalServerCoordination:
    """The accept coordination of _NoSignalServer, in one process."""

    @staticmethod
    def _bare_server(loads, index):
        """A _NoSignalServer with just the accept-coordination state."""
        from mlserver.rest.server import _NoSignalServer

        server = _NoSignalServer.__new__(_NoSignalServer)
        server._peer_loads = (loads, index)
        server._accept_tasks = set()
        server._connecting = 0
        return server

    def test_publish_load_counts_connections(self):
        """_publish_load writes connections + connecting into the slot."""
        from unittest.mock import Mock

        loads = [_NOT_ACCEPTING, _NOT_ACCEPTING]
        server = self._bare_server(loads, 1)
        server.server_state = Mock()
        server.server_state.connections = {1, 2, 3}
        server._connecting = 2
        server._publish_load()
        assert loads[1] == 5

    def test_least_loaded_tie_breaks_to_self(self):
        """Ties count as least loaded (the accept race resolves it)."""
        loads = [3, 3, 7]
        server = self._bare_server(loads, 1)
        assert server._least_loaded() is True
        loads[1] = 4
        assert server._least_loaded() is False

    def test_not_accepting_never_least_loaded(self):
        """_NOT_ACCEPTING (dead or starting frontend) never wins."""
        loads = [_NOT_ACCEPTING, 100]
        server = self._bare_server(loads, 0)
        assert server._least_loaded() is False

    @staticmethod
    async def _measure_accept_delay(loads, index, defer_checks, connections):
        """Start _accept on a listening socket, connect once, and
        return how long the server took to accept.

        ``connections`` seeds server_state.connections: the publish on
        readability overwrites the slot with the REAL load, so a busy
        server must be busy through real connection objects (the
        algorithm's own bookkeeping, not a hand-set slot value)."""
        from unittest.mock import Mock

        import mlserver.rest.server as rest_server_module
        from mlserver.rest.server import _NoSignalServer

        rest_server_module._ACCEPT_DEFER_CHECKS = defer_checks
        server = _NoSignalServer.__new__(_NoSignalServer)
        server._peer_loads = (loads, index)
        server._accept_tasks = set()
        server._connecting = 0
        server.server_state = Mock()
        server.server_state.connections = set(connections)
        accepted_at = {}

        async def fake_connect(loop, conn):
            accepted_at["t"] = time.monotonic()
            conn.close()

        server._connect = fake_connect

        sock = _shared_listening_sockets("127.0.0.1", 0)[0]
        try:
            sock.setblocking(False)
            task = asyncio.create_task(server._accept(sock))
            try:
                await asyncio.sleep(0.1)  # let _accept reach _readable
                start = time.monotonic()
                client = socket_module.create_connection(sock.getsockname(), timeout=5)
                deadline = time.monotonic() + 10
                while "t" not in accepted_at and time.monotonic() < deadline:
                    await asyncio.sleep(0.001)
                assert "t" in accepted_at, "server never accepted"
                client.close()
                return accepted_at["t"] - start
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        finally:
            sock.close()
            rest_server_module._ACCEPT_DEFER_CHECKS = 5

    async def test_defer_when_peer_less_loaded(self):
        """A busy server accepts a pending connection only after the
        defer budget expires (no peer took it)."""
        # This server really holds 3 connections; its peer holds 1.
        elapsed = await self._measure_accept_delay(
            [_NOT_ACCEPTING, 1], 0, defer_checks=50, connections={1, 2, 3}
        )
        # 50 checks x 1ms: the accept must have deferred for a large
        # fraction of the budget.
        assert elapsed >= 0.03, f"accepted after only {elapsed * 1000:.1f} ms"

    async def test_no_defer_when_least_loaded(self):
        """A least-loaded server accepts immediately (no defer)."""
        # This server holds no connections and its peer is not
        # accepting. Even a huge defer budget must not delay the accept.
        elapsed = await self._measure_accept_delay(
            [0, _NOT_ACCEPTING], 0, defer_checks=500, connections=set()
        )
        assert elapsed < 0.05, f"took {elapsed * 1000:.1f} ms to accept"


class TestSharedListenerTopology:
    """Full supervised topology on the shared listener."""

    @pytest.fixture
    def model_folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "models"
        (folder / "sum-model").mkdir(parents=True)
        (folder / "sum-model" / "model-settings.json").write_text(
            """
            {
              "name": "sum-model",
              "implementation": "tests.fixtures.SumModel",
              "max_batch_size": 1,
              "versions": ["v1"],
              "inputs": [
                {
                  "datatype": "INT32",
                  "name": "input-0",
                  "shape": [1, 3],
                  "parameters": {"content_type": "np"}
                }
              ],
              "parameters": {"version": "v1"}
            }
            """
        )
        return folder

    @pytest.fixture
    def supervisor_settings(self, model_folder: Path) -> Settings:
        http_port, grpc_port, metrics_port = get_available_ports(3)
        return Settings(
            http_port=http_port,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
            metrics_endpoint="",
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_repository_root=str(model_folder),
            model_operation_timeout=30,
            parallel_workers_timeout=10,
        )

    @pytest.fixture
    async def supervisor(self, supervisor_settings: Settings, model_folder: Path):
        from mlserver.frontend import FrontendSupervisor

        models = [
            ModelSettings.parse_file(
                str(model_folder / "sum-model" / "model-settings.json")
            )
        ]
        sup = FrontendSupervisor(supervisor_settings, models)
        task = asyncio.create_task(sup.start())
        setattr(sup, "_test_startup_task", task)
        try:
            yield sup
        finally:
            await sup.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _wait_all_accepting(self, sup, count=2, timeout=90.0):
        """Wait until every frontend slot publishes a real load."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            loads = [sup._peer_loads[i] for i in range(count)]
            if all(load != _NOT_ACCEPTING for load in loads):
                return loads
            await asyncio.sleep(0.2)
        raise TimeoutError(f"frontends not accepting: loads={list(sup._peer_loads)}")

    async def test_distribution_is_even(self, supervisor, supervisor_settings):
        """Held connections spread evenly across frontends (the
        measured failure mode of blind SO_REUSEPORT hashing)."""
        base_url = f"http://127.0.0.1:{supervisor_settings.http_port}"
        await _wait_ready(base_url)
        await self._wait_all_accepting(supervisor)

        conns = [
            socket_module.create_connection(
                ("127.0.0.1", supervisor_settings.http_port), timeout=5
            )
            for _ in range(8)
        ]
        try:
            await asyncio.sleep(1.0)
            loads = [supervisor._peer_loads[i] for i in range(2)]
            assert all(load != _NOT_ACCEPTING for load in loads)
            ratio = max(loads) / max(1, min(loads))
            assert ratio <= 1.5, f"uneven distribution: {loads}"
        finally:
            for conn in conns:
                conn.close()

        # Inference is unaffected by the held connections.
        result = await _infer(base_url, [1, 2, 3])
        assert result == 6

    async def test_kill_frontend_keeps_port_and_resets_slot(
        self, supervisor, supervisor_settings
    ):
        """SIGKILL of a frontend: the shared socket survives (the
        supervisor holds it), the survivor serves, the dead slot reads
        not-accepting, and the restart recovers."""
        import os

        base_url = f"http://127.0.0.1:{supervisor_settings.http_port}"
        await _wait_ready(base_url)
        await self._wait_all_accepting(supervisor)

        victim = supervisor._frontends[0]
        old_pid = victim.pid
        os.kill(old_pid, signal.SIGKILL)

        # The slot resets to not-accepting once the monitor sees the
        # death (its 1 s scan).
        deadline = asyncio.get_event_loop().time() + 10
        while asyncio.get_event_loop().time() < deadline:
            if supervisor._peer_loads[0] == _NOT_ACCEPTING:
                break
            await asyncio.sleep(0.2)
        assert supervisor._peer_loads[0] == _NOT_ACCEPTING

        # Traffic continues through the survivor.
        result = await _infer(base_url, [4, 5, 6])
        assert result == 15

        # The replacement is a different process that accepts again.
        deadline = asyncio.get_event_loop().time() + 60
        replacement = None
        while asyncio.get_event_loop().time() < deadline:
            proc = supervisor._frontends.get(0)
            if proc is not None and proc.pid != old_pid:
                replacement = proc
                break
            await asyncio.sleep(0.3)
        assert replacement is not None
        await self._wait_all_accepting(supervisor)

        result = await _infer(base_url, [7, 8, 9])
        assert result == 24

    async def test_stop_releases_port(self, supervisor, supervisor_settings):
        """After the supervisor stops, the port is released (no fd
        leak anywhere in the process tree)."""
        base_url = f"http://127.0.0.1:{supervisor_settings.http_port}"
        await _wait_ready(base_url)
        port = supervisor_settings.http_port

        await supervisor.stop()

        with pytest.raises((ConnectionRefusedError, OSError)):
            client = socket_module.create_connection(("127.0.0.1", port), timeout=2)
            client.close()

    async def test_supervisor_bound_the_shared_listener(
        self, supervisor, supervisor_settings
    ):
        """The supervisor owns the listener and the load array — the
        shared topology's precondition."""
        await _wait_ready(f"http://127.0.0.1:{supervisor_settings.http_port}")
        assert supervisor._rest_sockets is not None
        assert len(supervisor._rest_sockets) >= 1
        assert supervisor._peer_loads is not None
