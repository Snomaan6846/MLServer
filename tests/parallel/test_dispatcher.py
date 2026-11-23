import pytest
import asyncio
import threading
from unittest.mock import AsyncMock, Mock
from typing import cast

from mlserver.types import InferenceResponse
from mlserver.errors import InferenceTimeoutError
from mlserver.parallel.errors import WorkerStop, NoWorkersAvailable
from mlserver.parallel.dispatcher import Dispatcher
from mlserver.parallel.messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelRequestMessage,
)
from mlserver.parallel.worker import Worker
from mlserver.parallel.utils import terminate_queue, END_OF_QUEUE


async def test_on_worker_stop(dispatcher: Dispatcher):
    worker = list(dispatcher._workers.values())[0]
    await worker.stop()
    dispatcher.on_worker_stop(worker, 255)

    assert worker.pid not in dispatcher._workers
    # Ensure worker is no longer in round robin rotation
    workers_count = len(dispatcher._workers)
    for worker_pid in dispatcher._ready_workers:
        assert worker_pid != worker.pid
    assert workers_count >= 1


async def test_on_worker_stop_preserves_pool_ownership(
    inference_pool, dispatcher: Dispatcher
):
    """Revoking a worker's routing must not remove it from the pool's
    tracking table. The dispatcher keeps a private worker table: the
    pool remains the child's owner, so its close() still stops and
    reaps a worker whose routing was revoked first. Deleting the live
    child from the pool's table instead would make close() skip it —
    leaking the process, which then blocks multiprocessing's atexit
    join forever (the CI "tests pass then hang" failure)."""
    worker = list(dispatcher._workers.values())[0]

    dispatcher.on_worker_stop(worker, -9)

    # Routing is revoked...
    assert worker.pid not in dispatcher._ready_workers
    assert worker.pid not in dispatcher._workers
    # ...but the pool still owns the live child.
    assert inference_pool._workers.get(worker.pid) is worker
    assert worker.exitcode is None


async def test_dispatch(
    dispatcher: Dispatcher,
    load_message: ModelUpdateMessage,
    inference_request_message: ModelRequestMessage,
):
    await dispatcher.dispatch_update(load_message)
    response_message = await dispatcher.dispatch_request(inference_request_message)

    assert response_message.exception is None
    inference_response = response_message.return_value
    assert isinstance(inference_response, InferenceResponse)
    assert len(inference_response.outputs) > 0


async def test_inference_timeout_bounds_dropped_reply(
    responses,
    inference_request_message: ModelRequestMessage,
):
    """A configured inference deadline turns a dropped reply into a
    TimeoutError and clears the pending entry — a reply lost in flight
    (e.g. a frontend that stopped receiving) would otherwise leave the
    waiter, and its HTTP connection, pending forever with no
    server-side signal."""
    dispatcher = Dispatcher({}, responses, inference_timeout=0.05)
    dispatcher.start()
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher.on_worker_ready(worker)
    # The cancel handler is the worker-side batcher eviction: a timed
    # out request must be evicted instead of computing a response
    # nobody waits for.
    evictions: list[tuple] = []

    def cancel_handler(model_name, request_id, pid, model_version):
        evictions.append((model_name, request_id, pid, model_version))

    dispatcher.set_cancel_handler(cancel_handler)
    try:
        with pytest.raises(InferenceTimeoutError):
            await dispatcher.dispatch_request(inference_request_message)
        # The pending entry is cleared: a late reply resolves to no
        # waiter instead of accumulating leaked futures.
        assert inference_request_message.id not in dispatcher._async_responses._futures
        # The queued request was evicted from its worker-side batcher.
        assert evictions == [
            (
                inference_request_message.model_name,
                inference_request_message.id,
                worker_pid,
                inference_request_message.model_version,
            )
        ]
        # A late reply (the worker computed it anyway, or the response
        # was only delayed) resolves to no waiter: no crash, no
        # unhandled-error warning, no leaked future.
        dispatcher._async_responses.resolve(
            ModelResponseMessage(id=inference_request_message.id)
        )
        assert inference_request_message.id not in dispatcher._async_responses._futures
    finally:
        await terminate_queue(responses)
        await dispatcher.stop()


async def test_inference_reply_within_deadline_resolves(
    responses,
    inference_request_message: ModelRequestMessage,
):
    """The deadline only bounds the wait: a reply arriving within it
    resolves normally (the default of None keeps the historical
    unbounded wait, exercised by every other dispatch test)."""
    dispatcher = Dispatcher({}, responses, inference_timeout=5)
    dispatcher.start()
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher.on_worker_ready(worker)

    async def delayed_reply():
        await asyncio.sleep(0.05)
        dispatcher._async_responses.resolve(
            ModelResponseMessage(id=inference_request_message.id)
        )

    reply = asyncio.create_task(delayed_reply())
    try:
        response = await asyncio.wait_for(
            dispatcher.dispatch_request(inference_request_message), timeout=5
        )
        assert response.id == inference_request_message.id
        await reply
    finally:
        await terminate_queue(responses)
        await dispatcher.stop()


async def test_dispatch_request_no_ready_workers(
    dispatcher: Dispatcher,
    inference_request_message: ModelRequestMessage,
):
    dispatcher._ready_workers.clear()
    dispatcher._reset_round_robin()

    with pytest.raises(NoWorkersAvailable):
        await dispatcher.dispatch_request(inference_request_message)


async def test_dispatch_update_load_no_workers(
    responses,
    load_message: ModelUpdateMessage,
):
    dispatcher = Dispatcher({}, responses)
    dispatcher.start()
    try:
        with pytest.raises(NoWorkersAvailable):
            await dispatcher.dispatch_update(load_message)
    finally:
        await terminate_queue(responses)
        await dispatcher.stop()


async def test_cancel(dispatcher: Dispatcher, inference_request_message):
    worker = list(dispatcher._workers.values())[0]
    async_responses = dispatcher._async_responses
    async_responses._schedule(inference_request_message, worker)

    exit_code = 234
    async_responses.cancel(worker, exit_code)

    with pytest.raises(WorkerStop) as err:
        await async_responses._wait(inference_request_message.id)

    assert str(exit_code) in str(err)


async def test_response_loop_restarts_after_unexpected_error(dispatcher, mocker):
    process_responses = asyncio.get_running_loop().create_future()
    process_responses.set_exception(RuntimeError("response loop failed"))
    start = mocker.patch.object(dispatcher, "start")

    dispatcher._process_responses_cb(process_responses)

    start.assert_called_once_with()


async def test_response_loop_does_not_restart_after_cancellation(dispatcher, mocker):
    process_responses = asyncio.get_running_loop().create_future()
    process_responses.cancel()
    start = mocker.patch.object(dispatcher, "start")

    dispatcher._process_responses_cb(process_responses)

    start.assert_not_called()


async def test_on_worker_ready_adds_worker_to_routing(
    responses,
):
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher = Dispatcher({}, responses)

    dispatcher.on_worker_ready(worker)

    selected_worker, selected_pid = dispatcher._get_worker()
    assert selected_worker is worker
    assert selected_pid == worker_pid


async def test_dispatch_update_waits_for_worker_start(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    dispatcher = Dispatcher({}, responses)
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher._workers[worker_pid] = worker
    init_started = asyncio.Event()
    release_init = asyncio.Event()

    async def init_worker():
        init_started.set()
        await release_init.wait()

    async def dispatch_to_worker(_worker, _message):
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    start_task = asyncio.create_task(dispatcher.on_worker_start(worker, init_worker()))
    await init_started.wait()

    update_task = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await asyncio.sleep(0)
    assert not update_task.done()

    release_init.set()
    await asyncio.gather(start_task, update_task)


async def test_dispatch_update_serializes_updates(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher = Dispatcher({worker_pid: worker}, responses)
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch_to_worker(_worker, _message):
        dispatch_started.set()
        await release_dispatch.wait()
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    first_update = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await dispatch_started.wait()
    second_update = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await asyncio.sleep(0)

    assert not second_update.done()
    release_dispatch.set()
    await asyncio.gather(first_update, second_update)


async def test_dispatch_update_raises_non_worker_error(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, Mock(spec=Worker, pid=first_pid))
    second_worker = cast(Worker, Mock(spec=Worker, pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[RuntimeError("update failed"), ModelResponseMessage()],
    )

    with pytest.raises(RuntimeError, match="update failed"):
        await dispatcher.dispatch_update(load_message)


async def test_dispatch_update_raises_when_all_workers_stop_for_load(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, Mock(spec=Worker, pid=first_pid))
    second_worker = cast(Worker, Mock(spec=Worker, pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), WorkerStop(2)],
    )

    with pytest.raises(NoWorkersAvailable):
        await dispatcher.dispatch_update(load_message)


async def test_dispatch_update_returns_successes_when_some_workers_stop(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, Mock(spec=Worker, pid=first_pid))
    second_worker = cast(Worker, Mock(spec=Worker, pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    success = ModelResponseMessage()
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), success],
    )

    assert await dispatcher.dispatch_update(load_message) == [success]


async def test_dispatch_update_allows_all_workers_to_stop_during_unload(
    responses,
    unload_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, Mock(spec=Worker, pid=first_pid))
    second_worker = cast(Worker, Mock(spec=Worker, pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), WorkerStop(2)],
    )

    assert await dispatcher.dispatch_update(unload_message) == []


async def test_on_worker_start_defers_cancellation_until_initialization_finishes(
    responses,
):
    dispatcher = Dispatcher({}, responses)
    worker = cast(Worker, Mock(spec=Worker, pid=1234))
    init_started = asyncio.Event()
    release_init = asyncio.Event()

    async def init_worker():
        init_started.set()
        await release_init.wait()

    start_task = asyncio.create_task(dispatcher.on_worker_start(worker, init_worker()))
    await init_started.wait()

    start_task.cancel()
    await asyncio.sleep(0)
    assert not start_task.done()

    release_init.set()
    await start_task

    assert dispatcher._workers[worker.pid] is worker


async def test_dispatch_update_defers_cancellation_until_broadcast_finishes(
    responses, load_message, mocker
):
    worker = cast(Worker, Mock(spec=Worker, pid=1234))
    dispatcher = Dispatcher({worker.pid: worker}, responses)
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch_to_worker(_worker, _message):
        dispatch_started.set()
        await release_dispatch.wait()
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    update_task = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await dispatch_started.wait()

    update_task.cancel()
    await asyncio.sleep(0)
    assert not update_task.done()

    release_dispatch.set()
    await update_task


async def test_stop_cleans_up_response_processing(responses, mocker):
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    workers = {worker_pid: worker}
    dispatcher = Dispatcher(workers, responses)
    shutdown_executor = mocker.patch.object(
        dispatcher._executor, "shutdown", wraps=dispatcher._executor.shutdown
    )
    dispatcher.start()
    response_task = dispatcher._process_responses_task
    await terminate_queue(responses)

    await dispatcher.stop()

    assert not dispatcher._ready_workers
    assert dispatcher._workers == workers
    assert workers == {worker_pid: worker}
    assert response_task is not None
    assert response_task.done()
    # The early shutdown must never block the loop (a worker killed
    # mid-write leaves a broken queue), and the final joining shutdown
    # confirms thread exit through the public API, off-loop and bounded.
    shutdown_executor.assert_any_call(wait=False)
    shutdown_executor.assert_called_with(wait=True, cancel_futures=True)
    with pytest.raises(NoWorkersAvailable):
        dispatcher._get_worker()


async def test_stop_reports_stuck_response_reader(responses, mocker):
    """A reader thread stuck on a broken queue must not hang stop().

    The bounded join reports the failure instead of blocking the event
    loop indefinitely waiting for the reader to exit.
    """
    worker_pid = 1234
    worker = cast(Worker, Mock(spec=Worker, pid=worker_pid))
    dispatcher = Dispatcher({worker_pid: worker}, responses)
    release = threading.Event()
    entered = threading.Event()
    reader_self_released = threading.Event()

    def _blocked_get(*args, **kwargs):
        # Simulate a broken queue: the reader's blocking get never
        # returns, even after stop() queues END_OF_QUEUE to wake it.
        entered.set()
        # Independent deadline: if stop() ever regresses to a
        # synchronous executor join on the event loop, the finally
        # below cannot run and nothing would release this thread —
        # self-release so the regressed join completes and the test
        # FAILS instead of hanging the runner.
        if not release.wait(timeout=10):
            reader_self_released.set()
        return END_OF_QUEUE

    mocker.patch.object(dispatcher._responses, "get", side_effect=_blocked_get)
    dispatcher.start()
    assert dispatcher._process_responses_task is not None
    try:
        # Wait until the reader thread is inside the blocking get —
        # otherwise stop() flips _active before the loop's first
        # iteration and no thread ever blocks.
        assert await asyncio.to_thread(entered.wait, 5)

        with pytest.raises(RuntimeError, match="Response reader did not exit"):
            await dispatcher.stop(timeout=0.2)
    finally:
        # Release the stuck reader even when an assertion fails: its
        # blocked thread would keep the executor alive and prevent the
        # test process from exiting.
        release.set()
        await terminate_queue(responses)

    # The reader must have been released by the finally above, not by
    # its own deadline: a self-release means stop() blocked the event
    # loop on a synchronous join (the regression this guards against).
    assert not reader_self_released.is_set()


async def test_late_response_cannot_resolve_newer_request(event_loop, load_message):
    # Purpose: Exercise a late reply for a cancelled operation alongside a newer request
    # to the same worker.
    # Expect: The old reply cannot resolve the newer UUID or raise an unhandled callback
    # error.
    # Method: Schedule two controlled futures, cancel the old one, deliver each response
    # separately and inspect future completion plus the loop exception handler.
    from mlserver.parallel.dispatcher import AsyncResponses

    responses = AsyncResponses()
    worker = cast(Worker, Mock(spec=Worker, pid=1234))
    older = responses._schedule(load_message, worker)
    newer_message = load_message.model_copy(update={"id": "newer-operation"})
    newer = responses._schedule(newer_message, worker)
    errors = []
    original_handler = event_loop.get_exception_handler()
    event_loop.set_exception_handler(lambda loop, context: errors.append(context))
    try:
        older.cancel()
        responses.resolve(ModelResponseMessage(id=load_message.id))
        await asyncio.sleep(0)
        assert older.cancelled()
        assert not newer.done()
        responses.resolve(ModelResponseMessage(id=newer_message.id))
        await asyncio.sleep(0)
        assert newer.done()
        assert not errors
    finally:
        newer.cancel()
        event_loop.set_exception_handler(original_handler)


async def test_stopped_dispatcher_does_not_restart_failed_reader(responses, mocker):
    # Purpose: Exercise a response-reader failure callback arriving after dispatcher
    # shutdown begins.
    # Expect: The callback must not restart a reader against a stopped
    # dispatcher/executor.
    # Method: Supply a settled failing task to the callback on an inactive dispatcher
    # and assert its mocked start method was not called.
    dispatcher = Dispatcher({}, responses)
    start = mocker.patch.object(dispatcher, "start")

    async def fail():
        raise RuntimeError("reader failed as shutdown started")

    task = asyncio.create_task(fail())
    await asyncio.gather(task, return_exceptions=True)
    dispatcher._process_responses_cb(task)
    start.assert_not_called()
    await dispatcher.stop()


async def test_cancel_after_response_resolution_is_queued():
    # Purpose: Exercise caller cancellation after reply handling queues completion onto
    # the future's loop.
    # Expect: The callback rechecks completion state and avoids InvalidStateError on the
    # cancelled future.
    # Method: Call resolve before cancel, yield to run the queued callback and inspect
    # the loop's captured exception contexts.
    from mlserver.parallel.dispatcher import AsyncResponses

    responses = AsyncResponses()
    worker = cast(Worker, Mock(spec=Worker, pid=12345))
    message = ModelResponseMessage()
    future = responses._schedule(message, worker)
    loop = asyncio.get_running_loop()
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: errors.append(context))
    try:
        responses.resolve(message)
        # Exercise the handoff race, after resolve queued its loop callback.
        future.cancel()
        await asyncio.sleep(0)
        assert not errors
    finally:
        responses._clear_message(message.id)
        loop.set_exception_handler(previous)


def test_sticky_routing_batches_model_to_one_worker(dispatcher: Dispatcher):
    """A batched model routes to a single worker so its worker-side
    batcher sees the full traffic rate."""
    # Rotation off: the fixture's dispatcher carries the default
    # parallel_routing_rotation, and ten selections spanning a rotation
    # boundary would legitimately return two workers. Time-based
    # rotation has its own dedicated test.
    dispatcher._routing_rotation = 0
    dispatcher.set_model_batched("batched-model", True)
    workers = {dispatcher._get_worker("batched-model") for _ in range(10)}
    assert len(workers) == 1


def test_non_batched_model_round_robins(dispatcher: Dispatcher):
    """Models without batching keep execution spread across workers."""
    dispatcher.set_model_batched("batched-model", True)
    seen = {dispatcher._get_worker("plain-model")[1] for _ in range(10)}
    assert len(seen) > 1


def test_unknown_models_round_robin(dispatcher: Dispatcher):
    seen = {dispatcher._get_worker("unknown-model")[1] for _ in range(10)}
    assert len(seen) > 1


def test_sticky_routing_static_when_rotation_disabled(dispatcher: Dispatcher):
    """rotation=0 pins the assignment permanently."""
    dispatcher._routing_rotation = 0
    dispatcher.set_model_batched("batched-model", True)
    first = dispatcher._get_worker("batched-model")
    assert dispatcher._get_worker("batched-model") == first


def test_sticky_routing_rotates_over_time(mocker):
    """The assignment advances across rotation windows."""
    import multiprocessing

    dispatcher = Dispatcher({}, multiprocessing.Queue(), routing_rotation=1.0)
    for pid in (11, 22, 33):
        worker = Mock(spec=Worker)
        worker.pid = pid
        dispatcher.on_worker_ready(worker)
    dispatcher.set_model_batched("batched-model", True)
    mocker.patch(
        "mlserver.parallel.dispatcher.time.time",
        side_effect=[1000.0, 2000.0, 3000.0],
    )
    seen = {dispatcher._get_worker("batched-model")[1] for _ in range(3)}
    assert len(seen) > 1


def test_sticky_routing_tiny_rotation_degrades_to_static(mocker):
    """A rotation period so small the time quotient overflows the float
    range (1e-310) must degrade to static routing, not raise
    OverflowError on every request."""
    import multiprocessing

    dispatcher = Dispatcher({}, multiprocessing.Queue(), routing_rotation=1e-310)
    for pid in (11, 22, 33):
        worker = Mock(spec=Worker)
        worker.pid = pid
        dispatcher.on_worker_ready(worker)
    dispatcher.set_model_batched("batched-model", True)

    # Must not raise; and the degraded assignment is stable.
    first = dispatcher._get_worker("batched-model")
    assert dispatcher._get_worker("batched-model") == first


def test_sticky_routing_precision_boundary_is_static(mocker):
    """A quotient just beyond 2**53 cannot advance (float integer
    precision is exhausted): it behaves as static routing. Proven
    against the rotation-disabled selection — at this timestamp an
    unguarded int(quotient) conversion would select a DIFFERENT pid
    (digest%3 selects 33; digest+int(quotient) selects 22), so the
    test catches the guard's removal."""
    import multiprocessing

    dispatcher = Dispatcher({}, multiprocessing.Queue(), routing_rotation=1e-6)
    for pid in (11, 22, 33):
        worker = Mock(spec=Worker)
        worker.pid = pid
        dispatcher.on_worker_ready(worker)
    dispatcher.set_model_batched("batched-model", True)
    # time.time() mocked so time/period lands past 2**53.
    mocker.patch(
        "mlserver.parallel.dispatcher.time.time",
        return_value=2**54,
    )
    guarded = dispatcher._get_worker("batched-model")[1]
    # The same selection a rotation-disabled dispatcher makes.
    dispatcher._routing_rotation = 0
    static_pid = dispatcher._get_worker("batched-model")[1]
    assert guarded == static_pid == 33


def test_sticky_routing_single_worker_always_wins(dispatcher: Dispatcher):
    """W=1: the only worker is chosen regardless of batching."""
    import multiprocessing

    single = Dispatcher({}, multiprocessing.Queue())
    worker = Mock(spec=Worker)
    worker.pid = 4242
    single.on_worker_ready(worker)
    single.set_model_batched("batched-model", True)
    assert single._get_worker("batched-model")[1] == 4242


def test_set_model_batched_is_idempotent(dispatcher: Dispatcher):
    dispatcher.set_model_batched("m", True)
    dispatcher.set_model_batched("m", True)
    assert ("m", "") in dispatcher._batched_models
    dispatcher.set_model_batched("m", False)
    assert ("m", "") not in dispatcher._batched_models


async def test_frontend_id_stamped_on_dispatch(dispatcher: Dispatcher, mocker):
    """The dispatcher stamps its frontend id so replies route back to
    the dispatching frontend's reply endpoint."""
    import asyncio

    dispatcher._frontend_id = 7

    # Capture the message at the transport boundary: the stamp must be
    # on the outgoing request, not just on dispatcher state.
    # Record the frontend_id VALUE at the send boundary, not the
    # mutable message object: a stamp applied after send would
    # otherwise pass the assertion.
    sent_ids: list[int] = []

    async def capturing_send(worker, message):
        sent_ids.append(message.frontend_id)
        # No response: the dispatch stays pending so the stamp can be
        # asserted on the captured outgoing message.

    transport = Mock()
    transport.send = capturing_send
    transport.close = AsyncMock()
    transport.forget_worker = Mock()
    dispatcher._transport = transport

    request = ModelRequestMessage(model_name="sum-model", method_name="predict")
    task = asyncio.create_task(dispatcher.dispatch_request(request))
    # Let the dispatch reach the transport (a response never arrives in
    # this test; the stamp is asserted on the captured message).
    for _ in range(100):
        if sent_ids:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # The dispatch was pending on its response — the expected
        # outcome. Any other exception (a dispatch that raised after
        # sending) propagates and fails the test.
        pass

    assert sent_ids, "no message reached the transport boundary"
    assert sent_ids[0] == 7


async def test_cancel_handler_fires_on_caller_cancellation(mocker):
    """A cancelled dispatch notifies the topology's cancel hook so the
    queued request can be evicted from its worker-side batcher."""
    import multiprocessing

    dispatcher = Dispatcher({}, multiprocessing.Queue())
    worker = Mock(spec=Worker)
    worker.pid = 77
    dispatcher.on_worker_ready(worker)

    cancelled = mocker.MagicMock()
    dispatcher.set_cancel_handler(cancelled)

    # A response that never arrives keeps the dispatch pending; the
    # entry event proves the dispatch reached the response wait before
    # the cancellation (separating that from a cancel landing during
    # the send, which exercises a different notify path).
    entered_wait = asyncio.Event()

    async def hang(*args, **kwargs):
        entered_wait.set()
        await asyncio.Event().wait()

    mocker.patch.object(dispatcher._async_responses, "_wait", side_effect=hang)
    mocker.patch.object(
        dispatcher._async_responses, "_schedule", return_value=asyncio.Future()
    )
    dispatcher._transport = Mock()
    dispatcher._transport.send = AsyncMock()

    request = ModelRequestMessage(model_name="cancel-model", method_name="predict")
    task = asyncio.create_task(dispatcher.dispatch_request(request))
    await asyncio.wait_for(entered_wait.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cancelled.assert_called_once_with("cancel-model", request.id, 77, None)


# ============================================================================
# Pool-partitioned routing: workers of one pool never serve another pool's
# models (multi-pool shared topology).
# ============================================================================


def _pool_worker(pid: int) -> Mock:
    worker = Mock(spec=Worker)
    worker.pid = pid
    worker._zmq_request_endpoint = f"ipc:///tmp/mlserver-test-worker-{pid}"
    return worker


async def _register_pool_worker(
    dispatcher: Dispatcher, pid: int, env_hash: str | None = None
) -> Mock:
    worker = _pool_worker(pid)
    await dispatcher.on_worker_start(worker)
    dispatcher.on_worker_ready(worker, env_hash=env_hash)
    return worker


async def test_on_worker_ready_tags_worker_pool(responses):
    dispatcher = Dispatcher({}, responses)

    await _register_pool_worker(dispatcher, 201, "abc")

    assert dispatcher._worker_pools[201] == "abc"
    assert dispatcher._pool_pids == {"abc": [201]}


async def test_routing_stays_within_model_pool(responses):
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 102)
    await _register_pool_worker(dispatcher, 201, "abc")
    await _register_pool_worker(dispatcher, 202, "abc")
    dispatcher.set_model_pool("env-model", "abc")

    selected = set()
    for _ in range(8):
        _, pid = dispatcher._get_worker("env-model")
        selected.add(pid)

    # Round-robin visits every worker of the model's pool and no other.
    assert selected == {201, 202}


async def test_model_pool_without_ready_workers_raises(responses):
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    dispatcher.set_model_pool("env-model", "abc")

    with pytest.raises(NoWorkersAvailable):
        dispatcher._get_worker("env-model")


async def test_untagged_model_routes_to_untagged_workers(responses):
    """Backward compatibility: no tags on either side keeps routing whole."""
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 102)

    # Round-robin must alternate across BOTH workers — a selector
    # pinned to one passes a simple membership check.
    selected = {dispatcher._get_worker("plain-model")[1] for _ in range(10)}
    assert selected == {101, 102}


async def test_sticky_routing_partitions_by_pool(responses):
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 102)
    await _register_pool_worker(dispatcher, 201, "abc")
    await _register_pool_worker(dispatcher, 202, "abc")
    dispatcher.set_model_batched("batched-env-model", True)
    dispatcher.set_model_pool("batched-env-model", "abc")

    selected = set()
    for _ in range(5):
        _, pid = dispatcher._get_worker("batched-env-model")
        selected.add(pid)

    # Sticky selection is stable and confined to the model's pool, even
    # though default-pool workers are also ready.
    assert len(selected) == 1
    assert selected <= {201, 202}


async def test_worker_stop_keeps_other_pools_routing(responses):
    dispatcher = Dispatcher({}, responses)
    stopped = await _register_pool_worker(dispatcher, 201, "abc")
    await _register_pool_worker(dispatcher, 202, "abc")
    await _register_pool_worker(dispatcher, 101)
    dispatcher.set_model_pool("env-model", "abc")

    dispatcher.on_worker_stop(stopped, 0)

    # The surviving same-pool worker serves the model...
    _, pid = dispatcher._get_worker("env-model")
    assert pid == 202
    # ...and the default pool is unaffected.
    _, default_pid = dispatcher._get_worker("plain-model")
    assert default_pid == 101
    assert 201 not in dispatcher._worker_pools
    assert dispatcher._pool_pids["abc"] == [202]


async def test_set_model_pool_retag_moves_partition(responses):
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 201, "abc")
    dispatcher.set_model_pool("migrating-model", "abc")
    _, pid = dispatcher._get_worker("migrating-model")
    assert pid == 201

    # A reload into the default pool retags the name's partition.
    dispatcher.set_model_pool("migrating-model", None)
    _, pid = dispatcher._get_worker("migrating-model")
    assert pid == 101


async def test_version_partitions_route_independently(responses):
    """Different versions of the same name may live in different pools:
    the request's version selects the partition."""
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 201, "poolB")
    dispatcher.set_model_pool("m", None, "v1")
    dispatcher.set_model_pool("m", "poolB", "v2")

    seen_v1 = set()
    for _ in range(4):
        _, pid = dispatcher._get_worker("m", "v1")
        seen_v1.add(pid)
    assert seen_v1 == {101}

    seen_v2 = set()
    for _ in range(4):
        _, pid = dispatcher._get_worker("m", "v2")
        seen_v2.add(pid)
    assert seen_v2 == {201}

    # A versionless request uses its own (name, "") partition key.
    dispatcher.set_model_pool("m", None, None)
    _, pid = dispatcher._get_worker("m", None)
    assert pid == 101


async def test_version_partition_sticky_scoping(responses):
    """Sticky (batched) selection is confined to the version's pool."""
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 102)
    await _register_pool_worker(dispatcher, 201, "poolB")
    await _register_pool_worker(dispatcher, 202, "poolB")
    dispatcher.set_model_batched("m", True, "v2")
    dispatcher.set_model_pool("m", "poolB", "v2")

    selected = set()
    for _ in range(5):
        _, pid = dispatcher._get_worker("m", "v2")
        selected.add(pid)
    assert len(selected) == 1
    assert selected <= {201, 202}


# ============================================================================
# Pool quarantine: routing to a recovering pool pauses without dropping
# worker registration (in-flight responses may still arrive).
# ============================================================================


async def test_quarantine_pool_pauses_only_that_pool(responses):
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101)
    await _register_pool_worker(dispatcher, 201, "abc")
    dispatcher.set_model_pool("env-model", "abc")

    dispatcher.quarantine_pool("abc")

    # The quarantined pool has no routable workers...
    with pytest.raises(NoWorkersAvailable):
        dispatcher._get_worker("env-model")
    # ...while the default pool keeps serving.
    _, default_pid = dispatcher._get_worker("plain-model")
    assert default_pid == 101


async def test_quarantine_default_pool_pauses_untagged_ready_worker(responses):
    """A ready worker with no pool tag is default-pool by routing's
    derivation; quarantining the default pool must pause it too. The
    candidate scan derives from the ready set with .get(pid) — the same
    derivation the routing rebuild uses — so the two can never
    disagree."""
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 101, None)
    # A worker that became routable without any tag entry (no current
    # code path does this; the invariant must not depend on it).
    untagged = _pool_worker(102)
    await dispatcher.on_worker_start(untagged)
    dispatcher._ready_workers[102] = untagged
    dispatcher._reset_round_robin()

    dispatcher.quarantine_pool(None)

    assert 101 not in dispatcher._ready_workers
    assert 102 not in dispatcher._ready_workers
    # Still registered: quarantine pauses routing only.
    assert 101 in dispatcher._workers
    assert 102 in dispatcher._workers


async def test_quarantine_pool_keeps_workers_registered(responses):
    """Quarantine pauses routing only: workers stay registered so
    in-flight responses resolve and a later stop still cancels them."""
    dispatcher = Dispatcher({}, responses)
    quarantined = await _register_pool_worker(dispatcher, 201, "abc")
    await _register_pool_worker(dispatcher, 202, "abc")
    dispatcher.quarantine_pool("abc")

    assert 201 in dispatcher._workers
    assert 202 in dispatcher._workers

    # A real death during quarantine still processes fully.
    dispatcher.on_worker_stop(quarantined, 23)
    assert 201 not in dispatcher._workers
    assert 201 not in dispatcher._worker_pools


async def test_worker_ready_after_quarantine_restores_routing(responses):
    """Recovery re-announces workers with WorkerReadyEvents, which
    re-enable routing without any explicit un-quarantine message."""
    dispatcher = Dispatcher({}, responses)
    worker = await _register_pool_worker(dispatcher, 201, "abc")
    dispatcher.set_model_pool("env-model", "abc")
    dispatcher.quarantine_pool("abc")
    with pytest.raises(NoWorkersAvailable):
        dispatcher._get_worker("env-model")

    dispatcher.on_worker_ready(worker, env_hash="abc")

    _, pid = dispatcher._get_worker("env-model")
    assert pid == 201


async def test_quarantine_pool_is_idempotent(responses):
    """Chained quarantines (a second failure while recovering) re-fire
    the pause; the routing state stays consistent."""
    dispatcher = Dispatcher({}, responses)
    await _register_pool_worker(dispatcher, 201, "abc")
    await _register_pool_worker(dispatcher, 202, "abc")
    await _register_pool_worker(dispatcher, 101)

    dispatcher.quarantine_pool("abc")
    dispatcher.quarantine_pool("abc")

    assert dispatcher._pool_pids.get("abc") is None
    assert dispatcher._pool_pids.get(None) == [101]


def test_sticky_routing_moves_off_dead_worker(dispatcher: Dispatcher):
    """A batched model's sticky worker dying must move its routing to
    a live worker: a dead sticky target would send every subsequent
    batch to a worker that no longer exists."""

    dispatcher._routing_rotation = 0
    dispatcher.set_model_batched("batched-model", True)

    first, first_pid = dispatcher._get_worker("batched-model")
    # The sticky target dies (crash or pool stop).
    dispatcher.on_worker_stop(first, -9)

    for _ in range(10):
        worker, pid = dispatcher._get_worker("batched-model")
        assert pid != first_pid
