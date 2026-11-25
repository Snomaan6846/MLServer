"""Black-box integration tests for the multi-frontend shared-pool topology.

These spawn the real supervisor, frontend processes and inference
workers, and drive them over HTTP on the shared ports — exercising the
full control plane (startup loads, op forwarding, apply broadcasts,
worker-lifecycle mirroring) and the data plane (ZeroMQ reply routing by
frontend id, worker-side adaptive batching).
"""

import asyncio
import os
import json

import httpx

import numpy as np
import pytest

from collections.abc import AsyncGenerator
from pathlib import Path

from mlserver.frontend import FrontendSupervisor
from mlserver.settings import Settings, ModelSettings

from tests.utils import get_available_ports

INFER_TIMEOUT = 60.0
# CI runners are shared/slower than dev machines; frontends with real
# process spawning need generous timeouts for readiness and recovery.
# Parse CI explicitly: unset and "0" (both falsy intents) select the
# local timeout — a bare truthiness check treats "0" as enabled.
_CI_FLAG = os.environ.get("CI", "").strip().lower()
READY_TIMEOUT = 90.0 if _CI_FLAG in ("", "0", "false", "no") else 300.0


@pytest.fixture
def model_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "sum-model").mkdir()
    (folder / "sum-model" / "model-settings.json").write_text(
        """
        {
          "name": "sum-model",
          "implementation": "tests.fixtures.SumModel",
          "max_batch_size": 10,
          "max_batch_time": 0.05,
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
    # A second model loaded dynamically through the repository API.
    (folder / "late-model").mkdir()
    (folder / "late-model" / "model-settings.json").write_text(
        """
        {
          "name": "late-model",
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
    # A batched model with a LARGE batch window: a batched request
    # waits the window out, a bypassed one returns immediately — the
    # timing distinction the streaming-bypass test asserts on.
    (folder / "slow-batched-model").mkdir()
    (folder / "slow-batched-model" / "model-settings.json").write_text(
        """
        {
          "name": "slow-batched-model",
          "implementation": "tests.fixtures.SumModel",
          "max_batch_size": 8,
          "max_batch_time": 1.5,
          "inputs": [
            {
              "datatype": "INT32",
              "name": "input-0",
              "shape": [1, 3],
              "parameters": {"content_type": "np"}
            }
          ]
        }
        """
    )
    # A streaming-only model (predict_stream only, no predict): loaded
    # POOLED so the infer_stream endpoint's rejection path is
    # exercised (the SSE producer-queue surfaces it as a clean 400).
    (folder / "text-stream-model").mkdir()
    (folder / "text-stream-model" / "model-settings.json").write_text(
        """
        {
          "name": "text-stream-model",
          "implementation": "tests.fixtures.TextStreamModel"
        }
        """
    )
    return folder


@pytest.fixture
def supervisor_settings(model_folder: Path) -> Settings:
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
        parallel_routing_rotation=0,
        model_operation_timeout=30,
        parallel_workers_timeout=10,
    )


@pytest.fixture
async def supervisor(
    supervisor_settings: Settings, model_folder: Path
) -> AsyncGenerator[FrontendSupervisor, None]:
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        ),
        ModelSettings.parse_file(
            str(model_folder / "slow-batched-model" / "model-settings.json")
        ),
    ]
    supervisor = FrontendSupervisor(supervisor_settings, models)
    task = asyncio.create_task(supervisor.start())
    # Test-only handle (setattr keeps the typed surface clean).
    setattr(supervisor, "_test_startup_task", task)
    try:
        yield supervisor
    finally:
        await supervisor.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _wait_ready(base_url: str, startup_task: asyncio.Task | None = None) -> None:
    """Wait for readiness, propagating a supervisor startup failure.

    If the startup task is given and fails while polling, its exception
    is raised immediately — otherwise a broken server would only
    surface as a readiness timeout minutes later, hiding the cause.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            if startup_task is not None and startup_task.done():
                # Surface the startup failure (cancel is teardown, not
                # a failure — keep polling only for that case).
                if not startup_task.cancelled():
                    startup_task.result()
            try:
                response = await client.get(f"{base_url}/v2/health/ready")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError("frontends did not become ready in time")


async def _infer(base_url: str, model: str, data: list[int]) -> int:
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
            f"{base_url}/v2/models/{model}/infer", json=payload
        )
        response.raise_for_status()
        return response.json()["outputs"][0]["data"][0]


async def test_frontends_serve_startup_model_and_share_workers(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """Startup loads reach every frontend through the shared pool."""
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    # Many requests through the shared port hit both frontends; the
    # worker-side batcher must merge and split correctly for each.
    # Distinct payloads with distinct sums, compared in request order:
    # a splitter that swapped or duplicated responses cannot pass with
    # all-same expectations. The serving frontend's id is captured per
    # prediction (the x-mlserver-frontend-id header) and BOTH frontends
    # must serve — all requests landing on one frontend would mean the
    # other never received the startup model.
    served_by = set()

    async def _infer_capturing_frontend(data: list[int]) -> int:
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
            result = response.json()["outputs"][0]["data"][0]
            # Count the frontend only after its prediction is
            # validated: a wrong sum must not satisfy the check.
            if result == sum(data):
                served_by.add(response.headers.get("x-mlserver-frontend-id", ""))
            return result

    batch = 0
    results: list[int] = []
    while len(served_by) < 2 and batch < 10:
        # Fresh connections each batch: a persistent connection pins to
        # one frontend via SO_REUSEPORT.
        payloads = [
            [batch * 100 + i, batch * 100 + i + 1, batch * 100 + i + 2]
            for i in range(20)
        ]
        results = await asyncio.gather(
            *[_infer_capturing_frontend(payload) for payload in payloads]
        )
        assert results == [sum(payload) for payload in payloads]
        batch += 1
    assert served_by == {"0", "1"}, (
        f"requests must reach both frontends for the shared-worker "
        f"assertion to be meaningful, got {served_by}"
    )

    # The repository index sees the model (registry sync on both frontends).
    async with httpx.AsyncClient(timeout=10.0) as client:
        index = await client.post(f"{base_url}/v2/repository/index", json={})
        index.raise_for_status()
        names = [entry["name"] for entry in index.json()]
        assert "sum-model" in names


async def test_repository_load_reaches_all_frontends(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """A load on one frontend forwards to the supervisor's pool and
    broadcasts back so every frontend can serve the model."""
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    async with httpx.AsyncClient(timeout=INFER_TIMEOUT) as client:
        response = await client.post(f"{base_url}/v2/repository/models/late-model/load")
        response.raise_for_status()

    result = await _infer(base_url, "late-model", [4, 5, 6])
    assert result == 15

    # Verify through distinct connections (not one pooled client): a
    # single frontend answering is not proof that the broadcast apply
    # reached every registry. Fresh connections spread across the
    # SO_REUSEPORT frontends.
    async def _infer_on_fresh_connection(model: str, data: list[int]) -> int:
        async with httpx.AsyncClient(timeout=INFER_TIMEOUT) as client:
            response = await client.post(
                f"{base_url}/v2/models/{model}/infer",
                json={
                    "inputs": [
                        {
                            "name": "input-0",
                            "shape": [1, 3],
                            "datatype": "INT32",
                            "data": data,
                        }
                    ]
                },
            )
            return response.status_code

    for _ in range(10):
        if await _infer_on_fresh_connection("late-model", [1, 1, 1]) == 200:
            break
        await asyncio.sleep(1.0)
    else:
        raise AssertionError("late-model never became servable")

    # Unload through the API: every frontend's registry must drop the
    # model — verified by fresh connections, so a single stale registry
    # would surface as a 200 among the 404s.
    async with httpx.AsyncClient(timeout=INFER_TIMEOUT) as client:
        response = await client.post(
            f"{base_url}/v2/repository/models/late-model/unload"
        )
        response.raise_for_status()
    for _ in range(10):
        statuses = [
            await _infer_on_fresh_connection("late-model", [1, 1, 1]) for _ in range(6)
        ]
        if set(statuses) == {404}:
            break
        await asyncio.sleep(1.0)
    else:
        raise AssertionError(
            f"late-model still servable after unload (statuses {statuses}) — "
            "a frontend registry was not updated"
        )


async def test_infer_stream_bypasses_worker_side_batcher(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """A single-shot stream must not wait for the batch window.

    The worker-side batcher (supervised topology) cannot see the
    frontend's task-local bypass mark, so the dispatch carries the
    meaning explicitly (skip_batching). A bypassed stream returns
    well inside the 1.5s batch window; an ordinary /infer waits it
    out — the timing pair proves both directions.
    """
    import time as _time

    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, 3],
                "datatype": "INT32",
                "data": [1, 2, 3],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=INFER_TIMEOUT) as client:
        # Warmup: worker replay + batcher spin-up
        await client.post(
            f"{base_url}/v2/models/slow-batched-model/infer", json=payload
        )

        # The stream: bypassed -> well inside the window
        t0 = _time.monotonic()
        stream = await client.post(
            f"{base_url}/v2/models/slow-batched-model/infer_stream",
            json=payload,
        )
        stream_elapsed = _time.monotonic() - t0
        assert stream.status_code == 200
        assert b"data: " in stream.content

        # Ordinary infer: batched -> waits the window
        t0 = _time.monotonic()
        infer = await client.post(
            f"{base_url}/v2/models/slow-batched-model/infer", json=payload
        )
        infer_elapsed = _time.monotonic() - t0
        assert infer.status_code == 200

    assert (
        stream_elapsed < 1.0
    ), f"stream waited {stream_elapsed:.2f}s — entered the batcher"
    assert (
        infer_elapsed >= 1.2
    ), f"infer returned in {infer_elapsed:.2f}s — batching broken"


async def test_worker_death_is_recovered(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    # CI runners are slower; use generous timeouts for worker restart.
    """Killing the shared worker triggers the pool's replacement and the
    frontends route to the new incarnation."""
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))
    assert await _infer(base_url, "sum-model", [1, 2, 3]) == 6

    assert supervisor._pool_registry is not None
    pool = supervisor._pool_registry._default_pool
    assert pool is not None
    workers = list(pool._workers.values())
    assert len(workers) == 1
    # Kill without joining: the registry's installed SIGCHLD reaper must
    # own the waitpid, or the pool never learns the worker died.
    workers[0].kill()

    # The upstream pool replaces the worker and replays committed state;
    # the supervisor broadcasts the new incarnation to every frontend.
    # Recovery must reach BOTH frontends: the first passing prediction
    # could come from either one while the other still misses the new
    # worker's routing update — capture the serving frontend's id and
    # require a correct prediction from each.
    served_by: set[str] = set()

    async def _infer_capturing_frontend(data: list[int]) -> int:
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
            result = response.json()["outputs"][0]["data"][0]
            # Count the frontend only after its prediction is
            # validated: a wrong sum must not satisfy the check.
            if result == sum(data):
                served_by.add(response.headers.get("x-mlserver-frontend-id", ""))
            return result

    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT * 2
    last_error = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            assert await _infer_capturing_frontend([2, 3, 4]) == 9
            if served_by == {"0", "1"}:
                break
        except Exception as error:  # noqa: PERF203
            last_error = error
        await asyncio.sleep(1.0)
    else:
        raise AssertionError(f"inference did not recover: {last_error}")
    assert served_by == {
        "0",
        "1",
    }, f"worker replacement must reach both frontends, served by {served_by}"


async def test_frontend_death_is_restarted_and_reapplied(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """A crashed frontend restarts (capped) and re-receives every model."""
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    dead_slot = next(
        slot for slot, proc in supervisor._frontends.items() if proc.is_alive()
    )
    killed_pid = supervisor._frontends[dead_slot].pid
    supervisor._frontends[dead_slot].kill()

    # SO_REUSEPORT keeps the port served by the surviving frontend while
    # the supervisor restarts the dead one and re-applies the registry.
    # The replacement's own readiness gate is only released by the
    # post-replay StartupComplete, so readiness (not just liveness)
    # proves the replay completed. The slot's own process must be a
    # NEW live one: between the death handling (removing the dead
    # process) and the replacement's insertion, the survivor alone is
    # alive — waiting on "all alive" alone would exit in that window.
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT * 2
    while asyncio.get_running_loop().time() < deadline:
        try:
            if await _infer(base_url, "sum-model", [1, 1, 1]) == 3:
                replacement = supervisor._frontends.get(dead_slot)
                if (
                    replacement is not None
                    and replacement.is_alive()
                    and replacement.pid != killed_pid
                    and all(proc.is_alive() for proc in supervisor._frontends.values())
                ):
                    break
        except Exception:
            pass
        await asyncio.sleep(1.0)
    else:
        raise AssertionError("frontend did not restart in time")

    assert len(supervisor._frontends) == 2
    assert await _infer(base_url, "sum-model", [5, 5, 5]) == 15

    # The replacement frontend itself must be ready and serving: its
    # process being alive only proves spawn, not that its startup gate
    # was released. Probe readiness until it reports ready, then confirm
    # a correct prediction through the restarted slot.
    readiness_deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        while asyncio.get_running_loop().time() < readiness_deadline:
            try:
                response = await client.get("/v2/health/ready")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        else:
            raise AssertionError(
                "readiness never reported 200 after frontend restart — "
                "the replacement's startup gate was not released"
            )
    assert await _infer(base_url, "sum-model", [7, 7, 7]) == 21


async def test_startup_apply_failure_exits_and_recovers_slot(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """A frontend terminated for a failed startup apply must actually
    EXIT, and its slot must recover through the monitor's restart.

    The termination closes the supervisor's control end before the
    signal: a bare terminate() leaves the SIGTERMed frontend hung at
    interpreter exit on its blocked (non-daemon) control reader — no
    EOF arrives while the supervisor's end stays open — so the process
    never dies, the monitor (which only restarts on death) never
    recovers the slot, and the PID leaks. The exit is checked on the
    process sentinel, which is readable regardless of which reaper
    collects the corpse."""
    from multiprocessing.connection import wait as connection_wait

    from mlserver.parallel.shared import StartupApplyFailed

    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    slot = next(s for s, proc in supervisor._frontends.items() if proc.is_alive())
    proc = supervisor._frontends[slot]
    control = supervisor._controls[slot]
    old_pid = proc.pid

    # The listener delivers the frontend's own failure report on the
    # connection it arrived on (as _listen_frontend does).
    await supervisor._handle_control(
        slot, StartupApplyFailed(model_name="sum-model"), control=control
    )

    # The slot's control was retired with the termination.
    assert supervisor._controls.get(slot) is not control

    # The terminated frontend must actually exit within the confirm
    # discipline's budget — pre-fix it hung forever. The wait runs off
    # the event loop: the termination itself is an async confirmation
    # task that needs loop iterations to deliver the signal.
    exited = await asyncio.to_thread(connection_wait, [proc.sentinel], 30)
    assert exited, "frontend did not exit after startup-apply-failure termination"

    # The monitor restarts the dead slot: wait for a NEW live process
    # in it, then for readiness and a correct prediction (the
    # replacement's replay must complete before its gate opens).
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT * 2
    while asyncio.get_running_loop().time() < deadline:
        try:
            if await _infer(base_url, "sum-model", [1, 1, 1]) == 3:
                replacement = supervisor._frontends.get(slot)
                if (
                    replacement is not None
                    and replacement.pid != old_pid
                    and all(
                        p is not None and p.is_alive()
                        for p in supervisor._frontends.values()
                    )
                ):
                    break
        except Exception:
            pass
        await asyncio.sleep(1.0)
    else:
        raise AssertionError("frontend slot did not recover in time")

    assert await _infer(base_url, "sum-model", [5, 5, 5]) == 15


async def test_supervisor_stop_terminates_children(
    supervisor_settings: Settings, model_folder: Path
):
    """Ordered shutdown leaves no frontend or worker processes behind."""
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    supervisor = FrontendSupervisor(supervisor_settings, models)
    task = asyncio.create_task(supervisor.start())
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    try:
        await _wait_ready(base_url, task)
    finally:
        await supervisor.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    for proc in supervisor._frontends.values():
        assert not proc.is_alive()


# The exact series the metrics aggregation test increments: requests go
# to the versioned infer endpoint, so the counter carries version="v1".
INFER_COUNTER_LABELS = {"model": "sum-model", "version": "v1"}


async def test_metrics_aggregate_across_frontends(
    model_folder: Path,
):
    # Prometheus multiprocess mode: every frontend reports server-wide totals.
    from mlserver.frontend import FrontendSupervisor
    from mlserver.settings import Settings, ModelSettings
    from tests.utils import (
        get_available_ports,
        metrics_series_snapshot,
        query_metrics_series,
    )

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="/metrics",
        frontend_workers=2,
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    # The supervisor's configure_metrics mutates this process's
    # prometheus environment; restore it so subsequent tests are not
    # affected (a leaked multiprocess dir or value class changes where
    # every later test's counters land).
    import prometheus_client.values as _values

    _saved_multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    _saved_value_class = _values.ValueClass
    supervisor = FrontendSupervisor(settings, models)
    task = asyncio.create_task(supervisor.start())
    base_url = f"http://127.0.0.1:{settings.http_port}"
    metrics_url = f"http://127.0.0.1:{settings.metrics_port}"
    try:
        async with httpx.AsyncClient(base_url=metrics_url, timeout=10) as client:
            # Poll the metrics endpoint itself, to the startup
            # deadline: a readiness probe on another path does not
            # prove the metrics listener accepts connections, and
            # scraping unconditionally after a fixed probe count fails
            # when the listener needs longer to start. A failed
            # supervisor startup surfaces here instead of as a scrape
            # error.
            metrics_deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
            while asyncio.get_running_loop().time() < metrics_deadline:
                if task.done():
                    startup_error = task.exception()
                    if startup_error is not None:
                        raise startup_error
                try:
                    probe = await client.get("/metrics")
                    if probe.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.3)
            else:
                raise AssertionError(
                    f"metrics endpoint never became ready on {metrics_url}"
                )
            # Baseline before sending: the multiprocess metrics directory
            # is shared across supervisor instances in the same test run.
            # Baseline of the exact series the versioned infer requests
            # below increment, queried by label matchers (the PromQL
            # equivalent). Selection is never positional: the shared
            # multiprocess metrics directory accumulates series from
            # earlier tests in the same run, and exposition order is
            # unspecified.
            baseline_values = await query_metrics_series(
                client,
                "model_infer_request_success_total",
                INFER_COUNTER_LABELS,
            )
            baseline = baseline_values[0] if baseline_values else 0.0

        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            await _wait_ready(base_url, task)
            # Distinct connections spread across the SO_REUSEPORT
            # frontends; each connection's requests may land on a
            # different frontend's counter.
            total = 0
            served_by: set[str] = set()
            # Fresh connections until both frontends have served: four
            # pooled connections can all reach one frontend (SO_REUSEPORT
            # hashes by connection), which would make the aggregation
            # assertion vacuous. Every request still counts toward the
            # expected delta.
            while total < 20 or served_by != {"0", "1"}:
                async with httpx.AsyncClient(base_url=base_url, timeout=60) as sender:
                    for _ in range(5):
                        r = await sender.post(
                            "/v2/models/sum-model/versions/v1/infer",
                            content=json.dumps(
                                {
                                    "inputs": [
                                        {
                                            "name": "input-0",
                                            "shape": [1, 3],
                                            "datatype": "INT32",
                                            "data": [1, 2, 3],
                                        }
                                    ]
                                }
                            ).encode(),
                            headers={"content-type": "application/json"},
                        )
                        assert r.status_code == 200
                        # The header proves which frontend served; the
                        # aggregate assertion below only means something
                        # if both frontends actually handled traffic.
                        served_by.add(r.headers.get("x-mlserver-frontend-id", ""))
                        total += 1
                if total > 200:
                    break  # bounded: the spread assertion below reports
        assert served_by == {"0", "1"}, (
            f"requests must spread across both frontends for the "
            f"aggregation assertion to be meaningful, got {served_by}"
        )

        # Scrape repeatedly: SO_REUSEPORT may hit different frontends,
        # and every scrape must report the same server-wide total — the
        # guard against frontends regressing to per-process counters.
        # Retry the full scrape set so a transiently inconsistent view
        # (e.g. a scrape racing an increment on a slow runner) is not
        # mistaken for the regression.
        async with httpx.AsyncClient(base_url=metrics_url, timeout=30) as client:
            for attempt in range(3):
                await asyncio.sleep(1.0 if attempt > 0 else 0.2)
                scrapes = set()
                for _ in range(3):
                    values = await query_metrics_series(
                        client,
                        "model_infer_request_success_total",
                        INFER_COUNTER_LABELS,
                    )
                    if len(values) != 1:
                        snapshot = await metrics_series_snapshot(
                            client, "model_infer_request_success"
                        )
                        raise AssertionError(
                            f"expected exactly one infer counter series for "
                            f"{INFER_COUNTER_LABELS}, got {values}; "
                            f"exposition: {snapshot}"
                        )
                    scrapes.add(values[0] - baseline)
                if scrapes == {total}:
                    break
            if scrapes != {total}:
                snapshot = await metrics_series_snapshot(
                    client, "model_infer_request_success"
                )
                raise AssertionError(
                    f"metrics deltas {scrapes} != requests sent {total} "
                    f"(after {attempt + 1} attempts); baseline was {baseline}; "
                    f"exposition: {snapshot}"
                )
    finally:
        try:
            await supervisor.stop()
        finally:
            # Nested so the startup task is cancelled and joined and
            # the prometheus globals restored even when the shutdown
            # itself raised (without suppressing its error).
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Restore the prometheus environment this test mutated.
            if _saved_multiproc_dir is None:
                os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            else:
                os.environ["PROMETHEUS_MULTIPROC_DIR"] = _saved_multiproc_dir
            _values.ValueClass = _saved_value_class


async def test_pool_startup_failure_leaves_no_orphans(
    supervisor_settings: Settings, mocker, model_folder: Path
):
    """A failure while starting the shared pool must tear down the
    already-spawned frontends instead of leaving orphans on the ports."""
    from mlserver.parallel.registry import InferencePoolRegistry

    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    supervisor = FrontendSupervisor(supervisor_settings, models)
    mocker.patch.object(
        InferencePoolRegistry,
        "start",
        side_effect=RuntimeError("pool startup exploded"),
    )
    # The failed start's cleanup clears _frontends while terminating the
    # processes; spy on stop() to capture the handles before the clear,
    # so the no-orphan assertion runs on what was actually spawned.
    spawned: list = []
    original_stop = supervisor.stop

    async def spying_stop():
        spawned.extend(supervisor._frontends.values())
        await original_stop()

    mocker.patch.object(supervisor, "stop", side_effect=spying_stop)
    task = asyncio.create_task(supervisor.start())
    with pytest.raises(RuntimeError, match="pool startup exploded"):
        await asyncio.wait_for(task, timeout=30)

    assert spawned, "no frontend processes were spawned before the failure"

    # Every frontend process must be gone (no orphans on the ports).
    for proc in spawned:
        proc.join(timeout=5)
        assert not proc.is_alive(), f"frontend {proc.pid} orphaned"


async def test_grpc_serves_through_shared_frontends(model_folder: Path):
    """gRPC requests through the shared SO_REUSEPORT port reach a
    frontend and return correct predictions."""
    import grpc

    from mlserver.grpc import dataplane_pb2 as pb
    from mlserver.grpc import dataplane_pb2_grpc as pb_grpc

    from mlserver.frontend import FrontendSupervisor
    from mlserver.settings import Settings, ModelSettings
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        frontend_workers=2,
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    supervisor = FrontendSupervisor(settings, models)
    task = asyncio.create_task(supervisor.start())
    base_url = f"http://127.0.0.1:{settings.http_port}"
    try:
        await _wait_ready(base_url, task)
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{settings.grpc_port}")
        try:
            stub = pb_grpc.GRPCInferenceServiceStub(channel)
            request = pb.ModelInferRequest(
                model_name="sum-model",
                inputs=[
                    pb.ModelInferRequest.InferInputTensor(
                        name="input-0",
                        datatype="INT32",
                        shape=[1, 3],
                        parameters={
                            "content_type": pb.InferParameter(string_param="np")
                        },
                    )
                ],
                raw_input_contents=[np.array([[1, 2, 3]], dtype=np.int32).tobytes()],
            )
            # Retry: the gRPC request may hit a different frontend than
            # the one that answered the REST readiness check; on slower
            # machines not all frontends have the model registered yet.
            for attempt in range(5):
                try:
                    prediction = await asyncio.wait_for(
                        stub.ModelInfer(request), timeout=30
                    )
                    break
                except grpc.aio.AioRpcError as e:
                    # NOT_FOUND: a frontend without the model
                    # registration yet. UNAVAILABLE: the startup race
                    # before every frontend has bound the shared port.
                    retryable = (
                        e.code() == grpc.StatusCode.NOT_FOUND
                        or e.code() == grpc.StatusCode.UNAVAILABLE
                    )
                    if retryable and attempt < 4:
                        await asyncio.sleep(1.0)
                        continue
                    raise
            # The model's np content-type packs the output as raw
            # binary, carried at the response level (raw_output_contents).
            assert len(prediction.outputs) == 1
            assert len(prediction.raw_output_contents) == 1
            total = np.frombuffer(prediction.raw_output_contents[0], dtype=np.int64)
            assert total.tolist() == [6]
        finally:
            # Closed on every exit path: an inference failure or a
            # failed assertion must not leak the channel.
            await channel.close()
    finally:
        await supervisor.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_single_frontend_supervisor_serves_inference(
    model_folder: Path,
):
    """The supervisor topology at frontend_workers=1 serves inference.

    Regression test for the pool's frontend_workers=1 reply-endpoint double-bind: the
    pool's dispatcher must never bind a frontend-owned reply path (the
    worker's replies then land on the pool's socket and every inference
    request hangs). No other test runs the supervisor at frontend_workers=1 — the
    integration fixtures all use frontend_workers=2."""
    from mlserver.frontend import FrontendSupervisor
    from mlserver.settings import Settings, ModelSettings
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        frontend_workers=1,
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    supervisor = FrontendSupervisor(settings, models)
    task = asyncio.create_task(supervisor.start())
    base_url = f"http://{settings.host}:{settings.http_port}"
    try:
        await _wait_ready(base_url, task)
        assert await _infer(base_url, "sum-model", [4, 5, 6]) == 15
    finally:
        await supervisor.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_multi_frontend_batches_in_worker_and_maps_responses(
    model_folder: Path,
):
    """In the supervised topology (frontend_workers >= 1) the batcher
    lives in the WORKER: the
    frontends serve plain wrappers, and concurrent requests for a
    batched model produce correct responses across the shared
    frontends. Distinct payloads with distinct sums prove the batcher
    merges and splits without swapping or duplicating responses.
    Batching mechanics (window, queue, eviction) are covered by the
    unit tests in tests/batching/."""
    from mlserver.frontend import FrontendSupervisor
    from mlserver.settings import Settings, ModelSettings
    from tests.utils import get_available_ports

    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint="",
        frontend_workers=2,
        parallel_workers=1,
        parallel_inference_transport="zmq",
        model_operation_timeout=30,
    )
    models = [
        ModelSettings.parse_file(
            str(model_folder / "sum-model" / "model-settings.json")
        )
    ]
    models[0].max_batch_size = 4
    models[0].max_batch_time = 0.2

    supervisor = FrontendSupervisor(settings, models)
    task = asyncio.create_task(supervisor.start())
    base_url = f"http://{settings.host}:{settings.http_port}"
    try:
        await _wait_ready(base_url, task)

        # Distinct sums through the shared port: a batcher that
        # swapped or duplicated responses cannot pass.
        payloads = [[i, i + 1, i + 2] for i in range(8)]
        results = await asyncio.gather(
            *[_infer(base_url, "sum-model", payload) for payload in payloads]
        )
        assert results == [sum(payload) for payload in payloads]
    finally:
        await supervisor.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_infer_stream_pooled_streaming_only_rejected(
    supervisor: FrontendSupervisor, supervisor_settings: Settings
):
    """A pooled streaming-only model must be rejected with a clean HTTP
    400 BEFORE the SSE response starts — not a truncated empty-200
    stream the client can't distinguish from a legitimate one.

    This is the regression test for the SSE producer-queue mechanism:
    it would have caught the original ContextVar cross-context bug, a
    bare revert, and the bounded-queue deadlock.
    """
    base_url = f"http://{supervisor_settings.host}:{supervisor_settings.http_port}"
    await _wait_ready(base_url, getattr(supervisor, "_test_startup_task", None))

    async with httpx.AsyncClient(timeout=10) as client:
        # Load dynamically (the late-model pattern): the load itself
        # must succeed — only streaming is unsupported pooled.
        load_response = await client.post(
            f"{base_url}/v2/repository/models/text-stream-model/load"
        )
        assert load_response.status_code == 200

        response = await client.post(
            f"{base_url}/v2/models/text-stream-model/infer_stream",
            json={
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["hello"],
                    }
                ]
            },
        )

    assert response.status_code == 400, (
        f"expected 400 (actionable rejection), got {response.status_code} "
        f"with body: {response.text[:200]}"
    )
    assert "parallel inference pool cannot serve" in response.text
    assert "parallel_workers = 0" in response.text
