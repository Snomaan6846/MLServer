import os
import pytest
import asyncio
import numpy as np
from copy import deepcopy
from unittest.mock import Mock

from mlserver.errors import MLServerError
from mlserver.model import MLModel
from mlserver.settings import Settings, ModelSettings
from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec, StringCodec
from mlserver.parallel.pool import InferencePool, WorkerRegistry
from mlserver.parallel.worker import Worker
from mlserver.parallel.errors import InferencePoolUnavailable

from ..fixtures import ErrorModel, SumModel


def check_pid(pid):
    """
    Check For the existence of a unix pid.

    From https://stackoverflow.com/a/568285/5015573
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


async def _wait_for_workers(pool: InferencePool, expected: int):
    while len(pool._workers) < expected:
        await asyncio.sleep(0.1)


@pytest.fixture
def startup_worker(mocker):
    # These unit tests control replay without spawning a process. Represent a
    # live child accurately: exit status zero means confirmed exit, not liveness.
    worker = Mock(spec=Worker, pid=987654, exitcode=None, queues_retired=False)

    async def stop():
        worker.exitcode = 0

    async def retire(_timeout):
        worker.queues_retired = True

    worker.stop = stop
    worker.join = lambda _timeout: None
    worker.kill = mocker.Mock(side_effect=lambda: worker.record_exit(-9))
    worker.record_exit = lambda code: setattr(worker, "exitcode", code)
    worker.retire_queues = retire
    return worker


def test_workers_start(inference_pool: InferencePool, settings: Settings):
    assert len(inference_pool._workers) == settings.parallel_workers
    assert (
        inference_pool._dispatcher._model_operation_timeout
        == settings.model_operation_timeout
    )
    assert (
        inference_pool._dispatcher._worker_start_timeout
        == settings.worker_start_timeout
    )

    for worker_pid in inference_pool._workers:
        assert check_pid(worker_pid)


async def test_on_worker_stop(
    settings: Settings, inference_pool: InferencePool, sum_model: MLModel, mocker
):
    # Ensure pool has some loaded models
    await inference_pool.load_model(sum_model)

    prev_workers = list(inference_pool._workers.values())
    stopped_worker = prev_workers[0]
    close_queues = mocker.spy(stopped_worker, "close_queues")

    assert stopped_worker.pid is not None
    stopped_worker.kill()
    stopped_worker.join(settings.parallel_workers_timeout)
    await inference_pool.on_worker_stop(stopped_worker.pid, 23)
    close_queues.assert_called_once_with()

    # Wait for replacement worker (started via create_task in on_worker_stop)
    await asyncio.wait_for(
        _wait_for_workers(inference_pool, settings.parallel_workers),
        timeout=5,
    )

    # Make sure worker is taken out of the rota and a new worker is started
    new_workers = list(inference_pool._workers.values())
    assert len(new_workers) == settings.parallel_workers
    for worker in new_workers:
        assert worker.pid != stopped_worker.pid


async def test_replacement_worker_stays_pool_owned_until_close(
    settings: Settings, inference_pool: InferencePool, sum_model: MLModel
):
    """A replacement worker must remain in the pool's tracking from
    spawn until close: the dispatcher's worker table is private, so
    without the pool-side registration the replacement is invisible to
    the SIGCHLD reaper (no restart chain on its death) and to close()'s
    sweep — a live child that reaches multiprocessing's atexit join and
    hangs the process at exit."""
    await inference_pool.load_model(sum_model)

    original_pids = set(inference_pool._workers)
    stopped = list(inference_pool._workers.values())[0]
    assert stopped.pid is not None
    stopped.kill()
    stopped.join(settings.parallel_workers_timeout)
    await inference_pool.on_worker_stop(stopped.pid, 23)

    await asyncio.wait_for(
        _wait_for_workers(inference_pool, settings.parallel_workers), timeout=5
    )
    replacement_pids = [
        pid for pid in inference_pool._workers if pid not in original_pids
    ]
    assert len(replacement_pids) == 1
    replacement_pid = replacement_pids[0]

    # The replacement is owned: tracked for reaping and reap-able by
    # close() (registered under the worker-starting lock when its
    # replay begins, not only in the dispatcher's private routing
    # table).
    assert inference_pool.tracks_pid(replacement_pid)
    replacement = inference_pool._workers[replacement_pid]
    assert replacement.is_alive()

    await inference_pool.close()

    # close() must have confirmed the replacement's exit — a live
    # child here is the atexit-hang leak.
    assert replacement.exitcode is not None
    assert not check_pid(replacement_pid)


async def test_start_worker(
    settings: Settings,
    inference_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    # Ensure pool has some loaded models
    model = await inference_pool.load_model(sum_model)

    # Assert no traffic errors while new worker is starting
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    while not start_worker_task.done():
        inference_response = await model.predict(inference_request)
        assert len(inference_response.outputs) == 1

    await start_worker_task

    # Make a last pass through all workers
    for _ in range(settings.parallel_workers + 2):
        inference_response = await model.predict(inference_request)
        assert len(inference_response.outputs) == 1


async def test_start_worker_new_model(
    settings: Settings,
    inference_pool: InferencePool,
    sum_model: MLModel,
    simple_model: MLModel,
):
    # Ensure pool has some loaded models
    await inference_pool.load_model(sum_model)

    # Assert new models make their way through to the new worker
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    new_model = await inference_pool.load_model(simple_model)
    inference_request = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input("foo", np.array([[1, 2]], dtype=np.int32)),
            StringCodec.encode_input("bar", ["asd", "qwe"]),
        ]
    )
    while not start_worker_task.done():
        inference_response = await new_model.predict(inference_request)
        assert len(inference_response.outputs) == 1

    await start_worker_task

    # Make a last pass through all workers
    for _ in range(settings.parallel_workers + 2):
        inference_response = await new_model.predict(inference_request)
        assert len(inference_response.outputs) == 1


async def test_close_is_terminal(inference_pool: InferencePool, sum_model: MLModel):
    worker_pids = [pid for pid in inference_pool._workers]
    inference_pool._worker_registry.add(sum_model.settings)
    inference_pool._pending_reload[inference_pool._model_key(sum_model.settings)] = (
        sum_model
    )

    await inference_pool.close()

    assert len(inference_pool._workers) == 0
    assert inference_pool.empty()
    assert not inference_pool._pending_reload
    for worker_pid in worker_pids:
        assert not check_pid(worker_pid)

    with pytest.raises(InferencePoolUnavailable):
        await inference_pool.load_model(sum_model)
    with pytest.raises(InferencePoolUnavailable):
        await inference_pool.unload_model(sum_model)
    assert await inference_pool._start_worker() is None

    # Test idempotency
    await inference_pool.close()


async def test_close_clears_tracking_when_worker_cleanup_fails(
    settings: Settings, sum_model: MLModel, mocker
):
    # Purpose: Exercise an error raised after real worker cleanup has already finished.
    # Expect: Terminal close clears replay/replacement state while preserving its
    # recorded error for repeated callers.
    # Method: Populate both inventories, wrap successful native cleanup to raise
    # afterward and check empty model tracking, retired workers and the second close
    # result.
    inference_pool = InferencePool(settings)
    await inference_pool.start()
    inference_pool._worker_registry.add(sum_model.settings)
    inference_pool._pending_reload[inference_pool._model_key(sum_model.settings)] = (
        sum_model
    )
    close_workers = inference_pool._close_workers

    async def cleanup_then_fail():
        await close_workers()
        raise RuntimeError("worker cleanup failed")

    mocker.patch.object(inference_pool, "_close_workers", new=cleanup_then_fail)

    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        await inference_pool.close()

    assert inference_pool.empty()
    assert not inference_pool._pending_reload
    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        await inference_pool.close()
    assert not inference_pool._closing_workers


def test_worker_registry_keeps_model_name_and_version_keys_distinct(
    sum_model_settings: ModelSettings,
):
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.name = "foo-bar"
    first_settings.parameters.version = "v1"
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "foo"
    second_settings.parameters.version = "bar-v1"

    registry = WorkerRegistry()
    registry.add(first_settings)
    registry.add(second_settings)

    assert len(registry) == 2
    assert registry.has_model(first_settings)
    assert registry.has_model(second_settings)

    registry.remove(first_settings)

    assert len(registry) == 1
    assert not registry.has_model(first_settings)
    assert registry.has_model(second_settings)


async def test_pending_reload_keeps_model_name_and_version_keys_distinct(
    inference_pool: InferencePool,
    sum_model_settings: ModelSettings,
):
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.name = "foo-bar"
    first_settings.parameters.version = "v1"
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "foo"
    second_settings.parameters.version = "bar-v1"
    first_model = SumModel(first_settings)
    second_model = SumModel(second_settings)

    inference_pool._pending_reload[inference_pool._model_key(first_settings)] = (
        first_model
    )
    inference_pool._pending_reload[inference_pool._model_key(second_settings)] = (
        second_model
    )

    assert len(inference_pool._pending_reload) == 2
    assert (
        inference_pool._pending_reload[inference_pool._model_key(first_settings)]
        is first_model
    )
    assert (
        inference_pool._pending_reload[inference_pool._model_key(second_settings)]
        is second_model
    )

    del inference_pool._pending_reload[inference_pool._model_key(first_settings)]

    assert len(inference_pool._pending_reload) == 1
    assert (
        inference_pool._pending_reload[inference_pool._model_key(second_settings)]
        is second_model
    )


async def test_load(
    inference_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    sum_model.settings.name = "foo"
    assert len(inference_pool._worker_registry) == 0
    model = await inference_pool.load_model(sum_model)
    assert len(inference_pool._worker_registry) == 1

    # NOTE: This should leverage the worker inference_pool, after wrapping the
    # model
    inference_response = await model.predict(inference_request)

    assert inference_response.id == inference_request.id
    assert inference_response.model_name == sum_model.settings.name
    assert len(inference_response.outputs) == 1

    await inference_pool.unload_model(sum_model)
    assert len(inference_pool._worker_registry) == 0


async def test_load_error(
    inference_pool: InferencePool,
    load_error_model: MLModel,
):
    assert len(inference_pool._worker_registry) == 0
    with pytest.raises(MLServerError) as excinfo:
        await inference_pool.load_model(load_error_model)

    assert len(inference_pool._worker_registry) == 0
    expected_msg = f"mlserver.errors.MLServerError: {ErrorModel.error_message}"
    assert str(excinfo.value) == expected_msg


async def test_start_worker_waits_for_load_lock(
    inference_pool: InferencePool, sum_model: MLModel, mocker, startup_worker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    startup_started = asyncio.Event()
    worker = startup_worker

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)

    async def replay_worker(_worker):
        return None

    mocker.patch.object(inference_pool, "_replay_worker", new=replay_worker)

    original_on_worker_start = inference_pool._dispatcher.on_worker_start

    async def controlled_start(worker, init_coro):
        startup_started.set()
        await original_on_worker_start(worker, init_coro)

    mocker.patch.object(
        inference_pool._dispatcher, "on_worker_start", side_effect=controlled_start
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    await asyncio.sleep(0)

    assert not start_worker_task.done()
    assert not startup_started.is_set()
    release_dispatch.set()
    await asyncio.gather(load_task, start_worker_task)

    assert startup_started.is_set()


async def test_load_model_serializes(
    inference_pool: InferencePool, sum_model_settings: ModelSettings, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    dispatch_count = 0

    async def controlled_dispatch(_message):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    first_settings = deepcopy(sum_model_settings)
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "second-model"
    first = SumModel(first_settings)
    second = SumModel(second_settings)

    first_load = asyncio.create_task(inference_pool.load_model(first))
    await dispatch_started.wait()
    second_load = asyncio.create_task(inference_pool.load_model(second))
    await asyncio.sleep(0)

    assert not second_load.done()
    release_dispatch.set()
    await asyncio.gather(first_load, second_load)

    assert len(inference_pool._worker_registry) == 2


async def test_load_and_unload_serialize(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    dispatch_count = 0

    async def controlled_dispatch(_message):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    unload_task = asyncio.create_task(inference_pool.unload_model(sum_model))
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_dispatch.set()
    await asyncio.gather(load_task, unload_task)

    assert inference_pool.empty()


async def test_load_and_close_serialize(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    close_task = asyncio.create_task(inference_pool.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    assert inference_pool._closing

    release_dispatch.set()
    results = await asyncio.gather(load_task, close_task, return_exceptions=True)
    assert isinstance(results[0], InferencePoolUnavailable)
    assert results[1] is None

    assert inference_pool._closing
    assert inference_pool.empty()


async def test_start_worker_cancellation_settles_startup(
    inference_pool: InferencePool, mocker, startup_worker
):
    worker = startup_worker
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def controlled_replay(_worker):
        replay_started.set()
        await release_replay.wait()

    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)
    mocker.patch.object(inference_pool, "_replay_worker", side_effect=controlled_replay)

    start_task = asyncio.create_task(inference_pool._start_worker())
    await replay_started.wait()
    start_task.cancel()
    await asyncio.sleep(0)

    assert not start_task.done()
    release_replay.set()
    await start_task

    assert worker.pid in inference_pool._dispatcher._ready_workers


async def test_start_worker_replay_failure_kills_replacement(
    inference_pool: InferencePool, mocker, startup_worker
):
    worker = startup_worker

    async def replay_worker(_worker):
        raise RuntimeError("replay failed")

    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)
    mocker.patch.object(inference_pool, "_replay_worker", new=replay_worker)

    with pytest.raises(RuntimeError, match="replay failed"):
        await inference_pool._start_worker()

    worker.kill.assert_called_once_with()
    assert worker.pid not in inference_pool._dispatcher._ready_workers


async def test_load_model_cancellation_settles_and_preserves_tracking(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()

    load_task.cancel()
    await asyncio.sleep(0)
    assert not load_task.done()
    release_dispatch.set()

    await load_task

    assert inference_pool.has_model(sum_model.settings)


async def test_reload_cancellation_settles_and_preserves_pending_state(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    inference_pool._worker_registry.add(sum_model.settings)
    replacement = SumModel(deepcopy(sum_model.settings))

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    reload_task = asyncio.create_task(inference_pool.load_model(replacement))
    await dispatch_started.wait()

    reload_task.cancel()
    await asyncio.sleep(0)
    assert not reload_task.done()
    release_dispatch.set()

    await reload_task

    model_key = inference_pool._model_key(replacement.settings)
    assert inference_pool.has_model(sum_model.settings)
    assert inference_pool._pending_reload[model_key] is not None

    await inference_pool.unload_model(replacement)
    assert inference_pool.has_model(sum_model.settings)
    assert model_key not in inference_pool._pending_reload


async def test_unload_model_cancellation_settles_and_clears_tracking(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    inference_pool._worker_registry.add(sum_model.settings)

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    unload_task = asyncio.create_task(inference_pool.unload_model(sum_model))
    await dispatch_started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    release_dispatch.set()

    await unload_task

    assert inference_pool.empty()


async def test_failed_reload_preserves_committed_model_for_rollback(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    inference_pool._worker_registry.add(sum_model.settings)
    replacement = SumModel(deepcopy(sum_model.settings))
    dispatch = mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=[RuntimeError("replacement load failed"), []],
    )

    with pytest.raises(RuntimeError, match="replacement load failed"):
        await inference_pool.load_model(replacement)

    model_key = inference_pool._model_key(replacement.settings)
    assert inference_pool.has_model(sum_model.settings)
    assert model_key not in inference_pool._pending_reload

    dispatch.side_effect = None
    dispatch.return_value = []
    await inference_pool.unload_model(replacement)

    assert inference_pool.has_model(sum_model.settings)
    assert model_key not in inference_pool._pending_reload


async def test_shared_pool_stays_transport_free_at_one_frontend():
    """The pool's dispatcher must stay transport-free in the shared
    topology at ANY frontend count. Advertising the singular reply
    endpoint (the frontend_workers=1 shape) makes the pool's dispatcher bind a
    frontend-owned reply path: the double bind lets the worker's reply
    connection land on the pool's socket, swallowing the frontend's
    inference responses (every request then hangs). Update responses
    travel the multiprocessing queue, so the pool needs no transport."""
    import tempfile
    from pathlib import Path

    from mlserver.parallel.pool import InferencePool
    from mlserver.parallel.zmq import create_endpoint
    from mlserver.settings import Settings

    settings = Settings(
        parallel_workers=1,
        parallel_inference_transport="zmq",
        frontend_workers=0,
    )
    # A short private directory: pytest's tmp_path can exceed the
    # 100-byte IPC pathname limit on long --basetemp paths. The
    # directory is owned before the pool starts so a failed start
    # still cleans up.
    ipc_dir = Path(tempfile.mkdtemp(prefix="mlserver-tf-", dir="/tmp"))
    pool = None
    try:
        endpoints = [str(create_endpoint(ipc_dir, "r"))]
        pool = InferencePool(settings, reply_endpoints=endpoints)
        await pool.start()
        responses = pool._responses
        # The worker-facing list is intact...
        assert getattr(responses, "_zmq_reply_endpoints", None) == endpoints
        # ...but the dispatcher-creating singular attribute must NEVER be
        # advertised in the shared topology.
        assert getattr(responses, "_zmq_reply_endpoint", None) is None
        # The pool's own dispatcher must not have created a zmq transport.
        assert pool._owned_dispatcher._transport is None
    finally:
        # Nested finally: rmtree runs even if pool.close() raises,
        # so the IPC directory never leaks.
        try:
            if pool is not None:
                await pool.close()
        finally:
            import shutil

            shutil.rmtree(ipc_dir, ignore_errors=True)


async def test_shared_pool_stays_transport_free_at_two_frontends():
    """Same invariant at frontend_workers=2 (the previously-working shape): the list
    is advertised, the singular is not, no pool transport."""
    import tempfile
    from pathlib import Path

    from mlserver.parallel.pool import InferencePool
    from mlserver.parallel.zmq import create_endpoint
    from mlserver.settings import Settings

    settings = Settings(
        parallel_workers=1,
        parallel_inference_transport="zmq",
        frontend_workers=2,
    )
    ipc_dir = Path(tempfile.mkdtemp(prefix="mlserver-tf-", dir="/tmp"))
    pool = None
    try:
        endpoints = [str(create_endpoint(ipc_dir, "r")) for _ in range(2)]
        pool = InferencePool(settings, reply_endpoints=endpoints)
        await pool.start()
        responses = pool._responses
        assert getattr(responses, "_zmq_reply_endpoints", None) == endpoints
        assert getattr(responses, "_zmq_reply_endpoint", None) is None
        assert pool._owned_dispatcher._transport is None
    finally:
        # Nested finally: rmtree runs even if pool.close() raises,
        # so the IPC directory never leaks.
        try:
            if pool is not None:
                await pool.close()
        finally:
            import shutil

            shutil.rmtree(ipc_dir, ignore_errors=True)


async def test_track_batched_skips_affinity_at_direct_server():
    """At frontend_workers=0 (the direct server) the main-process batcher
    merges before dispatch: the dispatcher must NOT sticky-route
    (round-robin is correct for merged batches across multiple workers)."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=0,
    )
    pool = InferencePool(settings)
    from unittest.mock import Mock

    pool._owned_dispatcher = Mock()
    batched = ModelSettings(
        name="bstub",
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.05,
    )
    pool._track_batched(batched, True, None)
    # No affinity mark: the main-process batcher handles merging.
    pool._owned_dispatcher.set_model_batched.assert_not_called()


async def test_track_batched_sets_affinity_at_multiple_frontends():
    """In the supervised topology the batcher runs in the worker: sticky
    routing must be set so one worker's batcher sees full traffic."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
    )
    pool = InferencePool(settings)
    from unittest.mock import Mock

    pool._owned_dispatcher = Mock()
    batched = ModelSettings(
        name="bstub",
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.05,
    )
    pool._track_batched(batched, True, None)
    pool._owned_dispatcher.set_model_batched.assert_called_once_with("bstub", True, "")


async def test_get_ready_worker_endpoints_empty_before_activation():
    """A pool retained by the registry but not yet started has no
    dispatcher: the supervisor's restart replay must see an empty set,
    not an assertion failure (which would abort the replay and consume
    a restart attempt)."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import Settings

    settings = Settings(parallel_workers=1)
    pool = InferencePool(settings)

    assert pool.get_ready_worker_endpoints() == {}


async def test_get_ready_worker_endpoints_snapshots_ready_set():
    """After activation the accessor returns the dispatcher's ready
    workers as a snapshot copy."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import Settings

    settings = Settings(parallel_workers=1)
    pool = InferencePool(settings)
    worker = Mock(pid=101)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {101: worker}

    ready = pool.get_ready_worker_endpoints()

    assert ready == {101: worker}
    # A copy: later dispatcher mutations do not leak into the snapshot.
    pool._owned_dispatcher._ready_workers[202] = Mock(pid=202)
    assert 202 not in ready


async def test_unload_releases_batch_affinity_when_dispatch_fails():
    """A dispatch failure (deadline) arms recovery and skips anything
    after the dispatch, and a retry returns early through
    _retired_models: the affinity release must happen at the retirement
    commit point or the entry leaks for every version retired this
    way."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings
    from unittest.mock import AsyncMock, call

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
    )
    pool = InferencePool(settings)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {}
    pool._owned_dispatcher.dispatch_update = AsyncMock(side_effect=asyncio.TimeoutError)

    batched = ModelSettings(
        name="bto",
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.05,
    )
    model = Mock(spec=MLModel)
    model.settings = batched

    # The model was loaded previously and holds affinity.
    pool._worker_registry.add(batched)
    pool._track_batched(batched, True, None)
    assert ("bto", "") in pool._batched_names

    with pytest.raises(asyncio.TimeoutError):
        await pool.unload_model(model)

    # Affinity released despite the dispatch failure.
    assert ("bto", "") not in pool._batched_names
    pool._owned_dispatcher.set_model_batched.assert_called_with("bto", False, "")

    # A retry after the failed recovery: the settle re-raises the
    # recovery's error before any code could run — and must not
    # decrement the affinity a second time.
    with pytest.raises(asyncio.TimeoutError):
        await pool.unload_model(model)
    assert pool._owned_dispatcher.set_model_batched.call_args_list == [
        call("bto", True, ""),  # setup: the loaded model's affinity
        call("bto", False, ""),  # the unload's single release
    ]


async def test_unload_releases_batch_affinity_when_dispatch_outlives_deadline():
    """The dispatch can outlive the unload deadline: recovery later
    confirms it and a retry returns early through _retired_models. The
    affinity must be released exactly once, at the retirement commit
    point — not after the dispatch (skipped by the deadline) and not by
    the retry (early return)."""
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings
    from unittest.mock import call

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
        model_operation_timeout=0.1,
    )
    pool = InferencePool(settings)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {}
    release = asyncio.Event()

    async def slow_dispatch(message):
        # Outlives the unload deadline, then confirms.
        await release.wait()
        return []

    pool._owned_dispatcher.dispatch_update = slow_dispatch

    batched = ModelSettings(
        name="bto",
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.05,
    )
    model = Mock(spec=MLModel)
    model.settings = batched

    pool._worker_registry.add(batched)
    pool._track_batched(batched, True, None)
    assert ("bto", "") in pool._batched_names

    # The deadline expires after the local retirement committed.
    with pytest.raises(TimeoutError):
        await pool.unload_model(model)

    # The affinity was already released at the commit point, before
    # the dispatch settled.
    assert ("bto", "") not in pool._batched_names
    pool._owned_dispatcher.set_model_batched.assert_called_with("bto", False, "")

    # The dispatch confirms; recovery restores without a second release.
    release.set()
    await pool._settle_recovery()

    # The retry returns early through _retired_models.
    await pool.unload_model(model)
    assert pool._owned_dispatcher.set_model_batched.call_args_list == [
        call("bto", True, ""),  # setup: the loaded model's affinity
        call("bto", False, ""),  # the unload's single release
    ]


async def test_supervisor_path_reload_commits_ready_replacement():
    """A supervisor-driven reload (no registry hook chain) must commit
    the replacement as ready, so the retiring unload PUBLISHES the
    staged version instead of rolling it back."""
    from unittest.mock import AsyncMock

    from mlserver.parallel.messages import ModelUpdateType
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings
    from tests.fixtures import SumModel

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
    )
    pool = InferencePool(settings)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {}
    pool._owned_dispatcher.dispatch_update = AsyncMock(return_value=[])

    model_settings = ModelSettings(name="spr", implementation="tests.fixtures.SumModel")
    first = SumModel(model_settings)
    second = SumModel(model_settings.model_copy(deep=True))

    # Fresh load (no hooks active — the supervisor path).
    await pool.load_model(first)
    assert pool._worker_registry.has_model(model_settings)

    # The reload stages the replacement as READY: the no-hooks commit
    # is final, exactly like the hook-driven one.
    await pool.load_model(second)
    key = pool._model_key(model_settings)
    pending = pool._pending_reload[key]
    assert pending.ready, "the no-hooks commit must mark the replacement ready"

    # The retiring unload publishes the replacement: dispatched as a
    # commit (rollback=False), never a rollback of the staged version.
    await pool.unload_model(first)
    assert key not in pool._pending_reload
    assert pool._worker_registry.has_model(model_settings)
    unload_call = pool._owned_dispatcher.dispatch_update.call_args_list[-1]
    assert unload_call.args[0].update_type == ModelUpdateType.Unload
    assert unload_call.args[0].rollback is False


async def test_retry_after_deadline_outlived_unload_publishes_new_replacement():
    """A retiring unload whose cleanup outlives the caller's deadline
    completes in the background: the replacement is published and the
    old instance retired, but the caller saw a timeout and retries
    with a NEW staged replacement. The retry must publish and promote
    the new replacement — previously it returned early through
    _retired_models and reported success while the PREVIOUS
    replacement still served."""
    from mlserver.parallel.messages import ModelUpdateType
    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings
    from tests.fixtures import SumModel

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
        model_operation_timeout=0.1,
    )
    pool = InferencePool(settings)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {}
    release = asyncio.Event()
    dispatched = []

    async def slow_then_fast_dispatch(message):
        # The retiring unload's dispatch outlives the deadline; later
        # dispatches (the retry's promotion) confirm immediately.
        dispatched.append(message)
        if message.update_type == ModelUpdateType.Unload and not release.is_set():
            await release.wait()
        return []

    pool._owned_dispatcher.dispatch_update = slow_then_fast_dispatch

    def tagged(tag):
        return ModelSettings(
            name="spr2",
            implementation="tests.fixtures.SumModel",
            parameters={"extra": {"gen": tag}},
        )

    settings_a, settings_b, settings_c = tagged("A"), tagged("B"), tagged("C")
    a = SumModel(settings_a)
    b = SumModel(settings_b)
    c = SumModel(settings_c)

    # Fresh load of A, then a reload staging B as the ready
    # replacement (the supervisor path commits immediately).
    await pool.load_model(a)
    await pool.load_model(b)
    key = pool._model_key(settings_a)
    assert pool._pending_reload[key].ready

    # The retiring unload of A outlives the deadline: the caller
    # (supervisor) sees the timeout, but publication and retirement
    # already committed locally, and the shielded dispatch finishes in
    # the background.
    with pytest.raises(TimeoutError):
        await pool.unload_model(a)
    assert pool._worker_registry._models[key].parameters.extra["gen"] == "B"
    release.set()
    await pool._settle_recovery()

    # The retry reloads C; the supervisor still holds the stale
    # displaced instance A.
    await pool.load_model(c)
    assert pool._pending_reload[key].settings.parameters.extra["gen"] == "C"

    # The retry's unload of the already-retired A publishes and
    # promotes C instead of silently returning.
    await pool.unload_model(a)

    assert key not in pool._pending_reload
    assert pool._worker_registry._models[key].parameters.extra["gen"] == "C"
    # The promotion is an Unload dispatched as a commit (no rollback).
    promotion_calls = [
        message
        for message in dispatched
        if message.update_type == ModelUpdateType.Unload
    ]
    assert promotion_calls[-1].rollback is False
    assert promotion_calls[-1].model_settings.name == "spr2"


async def test_retry_with_unready_pending_leaves_it_staged():
    """A retired-instance retry whose staged replacement is not yet
    ready must not publish or dispatch: the pending stays staged for
    the operation that can settle it."""
    from unittest.mock import AsyncMock

    from mlserver.parallel.pool import InferencePool
    from mlserver.settings import ModelSettings, Settings
    from tests.fixtures import SumModel

    settings = Settings(
        parallel_workers=2,
        parallel_inference_transport="zmq",
        frontend_workers=2,
    )
    pool = InferencePool(settings)
    pool._owned_dispatcher = Mock()
    pool._owned_dispatcher._ready_workers = {}
    pool._owned_dispatcher.dispatch_update = AsyncMock(return_value=[])

    model_settings = ModelSettings(
        name="spr3", implementation="tests.fixtures.SumModel"
    )
    model = SumModel(model_settings)

    await pool.load_model(model)
    await pool.unload_model(model)  # retired, no pending

    # A staged-but-not-ready pending (a hook-driven load mid-flight).
    pending = Mock(ready=False)
    pool._pending_reload[pool._model_key(model_settings)] = pending

    dispatch_count = pool._owned_dispatcher.dispatch_update.call_count
    await pool.unload_model(model)

    # No publication, no dispatch — the pending survives untouched.
    assert pool._pending_reload[pool._model_key(model_settings)] is pending
    assert pool._owned_dispatcher.dispatch_update.call_count == dispatch_count

    # And with NO pending at all, the retry stays a silent no-op.
    pool._pending_reload.clear()
    await pool.unload_model(model)
    assert pool._owned_dispatcher.dispatch_update.call_count == dispatch_count


async def test_inference_timeout_reaches_owned_dispatcher(tmp_path):
    """The inference_timeout setting must reach the pool's owned
    dispatcher: a wiring break would silently leave every request
    unbounded (or wrongly bounded) with no test noticing."""
    settings = Settings(
        parallel_workers=1,
        inference_timeout=7.5,
        environments_dir=str(tmp_path),
    )
    pool = InferencePool(settings)
    try:
        await pool.start()
        assert pool._dispatcher._inference_timeout == 7.5
    finally:
        await pool.close()


async def test_inference_timeout_defaults_to_unbounded(tmp_path):
    """Without the setting, the owned dispatcher keeps the historical
    unbounded reply wait (None), not an implicit default."""
    settings = Settings(parallel_workers=1, environments_dir=str(tmp_path))
    pool = InferencePool(settings)
    try:
        await pool.start()
        assert pool._dispatcher._inference_timeout is None
    finally:
        await pool.close()
