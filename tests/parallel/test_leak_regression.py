"""Leak-regression tests for the supervisor and its reaper interplay.

Each test pins one resource-leak class: the restart fd slope,
listener-task accumulation, dead-process metric files, stale worker
IPC endpoint files, and the foreign-reaped exit status stash that
keeps death logs informative.
"""

import asyncio
import multiprocessing
import os
import signal
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlserver.settings import Settings, ModelSettings

from tests.utils import get_available_ports


def _fd_count(pid: int) -> int:
    return len(os.listdir(f"/proc/{pid}/fd"))


def _leak_test_child():
    """Module-level spawn target (nested functions cannot be pickled)."""
    time.sleep(60)


async def _wait_replacement(sup, frontend_id: int, old_pid: int, timeout: float = 60):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        proc = sup._frontends.get(frontend_id)
        if proc is not None and proc.pid != old_pid:
            return proc
        await asyncio.sleep(0.2)
    return None


class TestReleaseProcess:
    """_release_frontend_process: the restart-storm fd leak fix."""

    @pytest.fixture
    def sum_model_settings(self) -> ModelSettings:
        return ModelSettings(
            name="sum-model",
            implementation="tests.fixtures.SumModel",
            max_batch_size=1,
            inputs=[{"datatype": "INT32", "name": "input-0", "shape": [1, 3]}],
        )

    @pytest.fixture
    async def supervisor(self, sum_model_settings, tmp_path: Path):
        from mlserver.frontend import FrontendSupervisor

        http_port, grpc_port, metrics_port = get_available_ports(3)
        settings = Settings(
            http_port=http_port,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
            metrics_endpoint="",
            frontend_workers=2,
            parallel_workers=1,
            parallel_inference_transport="zmq",
            model_repository_root=str(tmp_path),
            model_operation_timeout=30,
            frontend_max_restarts=50,
        )
        sup = FrontendSupervisor(settings, [])
        task = asyncio.create_task(sup.start())
        try:
            yield sup
        finally:
            await sup.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_frontend_restarts_release_process_fds(
        self, supervisor, sum_model_settings, tmp_path: Path
    ):
        """Three frontend restarts must not grow the supervisor's fd
        count: the Process objects release their sentinel + bootstrap
        pipes (2 fds per restart before the fix)."""
        import httpx

        base_url = f"http://127.0.0.1:{supervisor._settings.http_port}"
        async with httpx.AsyncClient(timeout=30) as client:
            deadline = asyncio.get_event_loop().time() + 120
            while asyncio.get_event_loop().time() < deadline:
                try:
                    if (await client.get(f"{base_url}/v2/health/ready")).status_code:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)

        me = os.getpid()
        before = _fd_count(me)
        for _ in range(3):
            victim = supervisor._frontends[0]
            old_pid = victim.pid
            os.kill(old_pid, signal.SIGKILL)
            assert await _wait_replacement(supervisor, 0, old_pid) is not None
            await asyncio.sleep(1.5)

        # Allow the release path (called from the monitor's death
        # handling) to settle.
        await asyncio.sleep(1.0)
        after = _fd_count(me)
        # The fix targets the 2-fds-per-restart slope; a small constant
        # wiggle from unrelated machinery stays tolerated.
        assert after - before <= 2, (
            f"fd count grew {before} -> {after} across 3 restarts "
            f"({(after - before) / 3:.1f}/restart)"
        )

    async def test_listener_tasks_do_not_accumulate(self, supervisor):
        """A completed listener (its frontend died) must be discarded:
        the task set returns to its baseline size after a restart."""
        import httpx

        base_url = f"http://127.0.0.1:{supervisor._settings.http_port}"
        async with httpx.AsyncClient(timeout=30) as client:
            deadline = asyncio.get_event_loop().time() + 120
            while asyncio.get_event_loop().time() < deadline:
                try:
                    if (await client.get(f"{base_url}/v2/health/ready")).status_code:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)

        baseline = len(supervisor._listener_tasks)
        for _ in range(3):
            victim = supervisor._frontends[1]
            old_pid = victim.pid
            os.kill(old_pid, signal.SIGKILL)
            assert await _wait_replacement(supervisor, 1, old_pid) is not None
            await asyncio.sleep(1.5)

        after = len(supervisor._listener_tasks)
        assert (
            after <= baseline + 1
        ), f"listener tasks accumulated {baseline} -> {after} across 3 restarts"

    def test_release_helper_closes_spawned_process_fds(self):
        """Unit proof on a bare spawn: kill, confirm, release — the
        supervisor's fd count returns to its baseline."""
        ctx = multiprocessing.get_context("spawn")
        before = _fd_count(os.getpid())
        procs = [ctx.Process(target=_leak_test_child) for _ in range(2)]
        for proc in procs:
            proc.start()
        time.sleep(1.0)
        for proc in procs:
            os.kill(proc.pid, signal.SIGKILL)
        for proc in procs:
            proc.join()
        time.sleep(0.3)
        leaked = _fd_count(os.getpid())
        assert leaked - before >= 3, "expected the pre-fix leak to reproduce here"

        from mlserver.frontend import _release_frontend_process, _process_exited

        for proc in procs:
            _release_frontend_process(proc)
        released = _fd_count(os.getpid())
        assert (
            released <= before + 1
        ), f"release did not close the pipes: {before} -> {leaked} -> {released}"
        # The release is safe to repeat (stop() may sweep again).
        for proc in procs:
            _release_frontend_process(proc)

        # A released process must stay observably DEAD, never raise:
        # the monitor's death scan observes procs that a
        # terminate-for-restart confirmation already released, and an
        # AttributeError there killed the monitor and orphaned the
        # slot's replacement (the regression this pins).
        for proc in procs:
            assert _process_exited(proc) is True
            assert proc.is_alive() is False
            # exitcode stays readable through the stand-in: the
            # best-known code when the releaser had one, a plain-exit
            # 0 reading when it did not (an unknown status reads as
            # exited, never as running).
            assert proc.exitcode in (None, 0, -9, 1)
        # And a second release of an already-released proc is a no-op.
        for proc in procs:
            _release_frontend_process(proc, -9)
            assert proc.exitcode == -9


class TestForeignExitStash:
    """The reaper's exit-status stash for foreign-reaped children."""

    def test_record_and_pop(self):
        from mlserver.parallel.registry import InferencePoolRegistry

        registry = InferencePoolRegistry(Settings(parallel_workers=1))
        registry.record_foreign_exit(4242, -9)
        assert registry.pop_foreign_exit(4242) == -9
        # Consumed exactly once.
        assert registry.pop_foreign_exit(4242) is None

    def test_stash_is_bounded(self):
        from mlserver.parallel.registry import InferencePoolRegistry

        registry = InferencePoolRegistry(Settings(parallel_workers=1))
        for i in range(300):
            registry.record_foreign_exit(1000 + i, -15)
        assert len(registry._foreign_exit_codes) <= 256
        # The oldest entries were dropped; the newest survive.
        assert registry.pop_foreign_exit(1299) == -15


class TestDeadProcessMetricFiles:
    """A5: every *_{pid}.db file of a dead frontend is removed."""

    def test_mark_metrics_dead_removes_all_files(self, tmp_path: Path):
        from mlserver.frontend import FrontendSupervisor

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
        pid = os.getpid() + 100000  # a pid that owns nothing real
        for prefix in ("counter", "histogram", "summary", "gauge_all", "gauge_livesum"):
            (metrics_dir / f"{prefix}_{pid}.db").write_bytes(b"x")
        # A different pid's file must survive.
        other = metrics_dir / f"counter_{pid + 1}.db"
        other.write_bytes(b"x")

        settings = Settings(metrics_endpoint="/metrics", metrics_dir=str(metrics_dir))
        supervisor = FrontendSupervisor.__new__(FrontendSupervisor)
        supervisor._settings = settings
        proc = Mock()
        proc.pid = pid

        supervisor._mark_metrics_process_dead(proc)

        remaining = sorted(p.name for p in metrics_dir.iterdir())
        assert remaining == [
            other.name
        ], f"dead frontend's metric files survived: {remaining}"

    def test_remove_process_metric_files_removes_all(self, tmp_path: Path):
        from mlserver.metrics.prometheus import remove_process_metric_files

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
        pid = os.getpid() + 200000
        for prefix in ("counter", "histogram", "gauge_all"):
            (metrics_dir / f"{prefix}_{pid}.db").write_bytes(b"x")

        remove_process_metric_files(Settings(metrics_dir=str(metrics_dir)), pid)

        assert list(metrics_dir.iterdir()) == []

    def test_remove_process_metric_files_tolerates_missing(self, tmp_path: Path):
        from mlserver.metrics.prometheus import remove_process_metric_files

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        # No files at all: the cancellation-during-init case.
        remove_process_metric_files(Settings(metrics_dir=str(metrics_dir)), 12345)


class TestWorkerEndpointCleanup:
    """A6: a dead worker's ipc endpoint file is unlinked."""

    async def test_remove_worker_endpoint_unlinks(self, tmp_path: Path):
        from mlserver.frontend import FrontendSupervisor

        endpoint_file = tmp_path / "req-abc"
        endpoint_file.write_bytes(b"")
        worker = Mock()
        worker._zmq_request_endpoint = f"ipc://{endpoint_file}"

        supervisor = FrontendSupervisor.__new__(FrontendSupervisor)
        await supervisor._remove_worker_endpoint(worker)

        assert not endpoint_file.exists()

    async def test_remove_worker_endpoint_tolerates_missing(self, tmp_path: Path):
        from mlserver.frontend import FrontendSupervisor

        worker = Mock()
        worker._zmq_request_endpoint = f"ipc://{tmp_path / 'req-gone'}"

        supervisor = FrontendSupervisor.__new__(FrontendSupervisor)
        # A graceful stop already unlinked it: no error.
        await supervisor._remove_worker_endpoint(worker)

    async def test_remove_worker_endpoint_skips_non_ipc(self):
        from mlserver.frontend import FrontendSupervisor

        worker = Mock()
        worker._zmq_request_endpoint = "tcp://127.0.0.1:1234"

        supervisor = FrontendSupervisor.__new__(FrontendSupervisor)
        await supervisor._remove_worker_endpoint(worker)  # no-op, no error
