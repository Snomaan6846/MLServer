import asyncio

import time
import zlib

from typing import cast, TYPE_CHECKING
from weakref import WeakKeyDictionary
from collections.abc import Awaitable, Callable, Iterator
from itertools import cycle
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from asyncio import Future

from ..utils import (
    defer_cancellation,
    generate_uuid,
    schedule_with_callback,
    with_operation_lock,
)
from ..metrics import REGISTRY

from .errors import WorkerStop, NoWorkersAvailable
from .worker import Worker
from .logging import logger
from .utils import END_OF_QUEUE, cancel_task
from ..settings import DEFAULT_MODEL_OPERATION_TIMEOUT, DEFAULT_WORKER_START_TIMEOUT
from .messages import (
    Message,
    ModelUpdateMessage,
    ModelUpdateType,
    ModelRequestMessage,
    ModelResponseMessage,
)
from prometheus_client import Histogram

QUEUE_METRIC_NAME = "parallel_request_queue"

if TYPE_CHECKING:
    from .zmq import ParentTransport


class AsyncResponses:
    def __init__(self) -> None:
        self._futures: dict[str, Future[ModelResponseMessage]] = {}

        # _workers_map keeps track of which in-flight requests are being served
        # by each worker
        # Empty groups stay weakly keyed. Deleting after observing an empty
        # group can erase a request concurrently registered by a foreign loop.
        self._workers_map: WeakKeyDictionary[Worker, set[str]] = WeakKeyDictionary()
        # _futures_map keeps track of which worker is serving each in-flight
        # request
        self._futures_map: dict[str, Worker] = {}

        self.parallel_request_queue_size = self._get_or_create_metric()

    def _get_or_create_metric(self) -> Histogram:
        if QUEUE_METRIC_NAME in REGISTRY:
            return REGISTRY[QUEUE_METRIC_NAME]  # type: ignore

        return Histogram(
            QUEUE_METRIC_NAME,
            "counter of request queue size for workers",
            registry=REGISTRY,
        )

    async def schedule_and_wait(
        self, message: Message, worker: Worker, timeout: float | None = None
    ) -> ModelResponseMessage:
        """
        Schedule a response and wait until it gets resolved.
        """
        message_id = message.id
        self._schedule(message, worker)
        if timeout is None:
            return await self._wait(message_id)
        # A reply dropped in flight would otherwise leave this waiter
        # pending forever; the deadline raises TimeoutError and clears
        # the pending entry (a late reply resolves to no waiter).
        return await asyncio.wait_for(self._wait(message_id), timeout)

    def _schedule(self, message: Message, worker: Worker) -> Future:
        """
        Schedule a response to be resolved in the future.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        message_id = message.id
        self._futures[message_id] = future

        # Keep track of allocation for in-flight requests
        self._track_message(message, worker)

        # Monitor current in-flight requests
        in_flight_count = len(self._futures)
        self.parallel_request_queue_size.observe(in_flight_count)

        return future

    def _track_message(self, message: Message, worker: Worker) -> None:
        self._futures_map[message.id] = worker
        messages = self._workers_map.get(worker)
        if messages is None:
            messages = self._workers_map.setdefault(worker, set())
        messages.add(message.id)

    async def _wait(self, message_id: str) -> ModelResponseMessage:
        future = self._futures[message_id]

        try:
            response_message = await future
            return response_message
        finally:
            self._clear_message(message_id)

    def _clear_message(self, message_id: str) -> None:
        del self._futures[message_id]
        worker = self._futures_map.pop(message_id)
        messages = self._workers_map[worker]
        messages.remove(message_id)

    def resolve(self, response: ModelResponseMessage):
        """
        Resolve a previously scheduled response future.
        """
        message_id = response.id
        future = self._futures.get(message_id)
        if future is None or future.done():
            # Future already cancelled and cleared (e.g. worker crashed after
            # sending its response) — nothing left to resolve
            return

        # NOTE: Use call_soon_threadsafe to cover cases where `model.predict()`
        # (or other methods) get called from a separate thread (and a separate
        # AsyncIO loop)
        loop = future.get_loop()
        loop.call_soon_threadsafe(self._complete, future, response)

    @staticmethod
    def _complete(
        future: Future[ModelResponseMessage], response: ModelResponseMessage
    ) -> None:
        # Cancellation can run after resolve() schedules this callback. Check
        # at completion time, not only when the response first arrives.
        if future.done():
            return
        if response.exception:
            future.set_exception(response.exception)
        else:
            future.set_result(response)

    @staticmethod
    def _fail(future: Future[ModelResponseMessage], error: BaseException) -> None:
        if not future.done():
            future.set_exception(error)

    def fail_all(self, error: BaseException) -> None:
        """Wake inference waiters if the optional response transport fails."""
        for future in list(self._futures.values()):
            future.get_loop().call_soon_threadsafe(self._fail, future, error)

    def cancel(self, worker: Worker, exit_code: int):
        """
        Cancel in-flight requests for worker (e.g. because it died
        unexpectedly).
        """
        in_flight = self._workers_map.get(worker, set())
        if in_flight:
            logger.info(
                f"Cancelling {len(in_flight)} in-flight requests for "
                f"worker {worker.pid} which died unexpectedly with "
                f"exit code {exit_code}..."
            )
        # Foreign-loop waiters can clear entries as callbacks are delivered.
        for message_id in list(in_flight):
            err = WorkerStop(exit_code)
            future = self._futures.get(message_id)
            if future is None:
                continue
            loop = future.get_loop()
            loop.call_soon_threadsafe(self._fail, future, err)


class Dispatcher:
    def __init__(
        self,
        workers: dict[int, Worker],
        responses: Queue,
        model_operation_timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT,
        worker_start_timeout: float = DEFAULT_WORKER_START_TIMEOUT,
        frontend_id: int = 0,
        routing_rotation: float = 0.0,
        inference_timeout: float | None = None,
    ):
        self._responses = responses
        self._inference_timeout = inference_timeout
        # Private worker table: never alias the caller's dict. The pool
        # shares its tracking registry at construction, but a routing
        # revocation here (on_worker_stop) must not delete from the
        # pool's table: the pool — the child's owner — learns of the
        # death separately and is responsible for stopping, reaping and
        # retiring the worker. Deleting a live child from the pool's
        # tracking would make pool.close() skip it and leak the process
        # (which then blocks multiprocessing's atexit join forever).
        self._workers = dict(workers)
        self._model_operation_timeout = model_operation_timeout
        self._worker_start_timeout = worker_start_timeout
        self._frontend_id = frontend_id
        self._routing_rotation = routing_rotation
        # Model names whose adaptive batcher lives in the inference
        # workers. They route model-affinely so one batcher sees the full
        # traffic rate; everything else keeps round-robin execution spread.
        self._batched_models: set[tuple[str, str]] = set()
        # Pool partitioning: every worker is tagged with the env hash of
        # the pool that owns it, and every model with the pool that
        # loaded it. Requests only route within the model's pool, so a
        # worker never serves a model it did not load. None is the
        # default pool; single-pool dispatchers leave everything untagged
        # (which partitions identically).
        self._worker_pools: dict[int, str | None] = {}
        self._model_pools: dict[tuple[str, str], str | None] = {}
        self._pool_pids: dict[str | None, list[int]] = {}
        self._pool_round_robins: dict[str | None, Iterator[int]] = {}
        self._ready_workers = dict(workers)
        self._sorted_ready_pids: list[int] = sorted(self._ready_workers)
        self._workers_round_robin = self._reset_round_robin()
        self._worker_starting_lock = asyncio.Lock()
        self._active = False
        self._process_responses_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor()
        self._async_responses = AsyncResponses()
        self._transport: ParentTransport | None = None
        # Optional cancellation hook (model_name, request_id, worker_pid),
        # set by topologies that can evict queued requests remotely.
        self._cancel_handler: Callable[[str, str, int, "str | None"], None] | None = (
            None
        )

    def set_cancel_handler(
        self, handler: Callable[[str, str, int, "str | None"], None]
    ) -> None:
        self._cancel_handler = handler

    def set_model_batched(
        self, model_name: str, batched: bool, model_version: str | None = None
    ) -> None:
        """Track whether a model version's batcher runs in the workers."""
        key = (model_name, model_version or "")
        if batched:
            self._batched_models.add(key)
        else:
            self._batched_models.discard(key)

    def set_model_pool(
        self,
        model_name: str,
        env_hash: str | None,
        model_version: str | None = None,
    ) -> None:
        """Tag a model version with the pool whose workers must serve it.

        Routing is per (model name, version): different versions of the
        same name may live in different pools (e.g. a versioned reload
        into another environment), so the name alone cannot select the
        pool. An unload resets the version's tag to the default pool.
        """
        if env_hash is None:
            # An unload removes the tag entirely: a long-running
            # frontend with model churn must not retain every name and
            # version from its history. Both selectors already resolve
            # an absent tag to the default pool.
            self._model_pools.pop((model_name, model_version or ""), None)
        else:
            self._model_pools[(model_name, model_version or "")] = env_hash

    def quarantine_pool(self, env_hash: str | None) -> None:
        """Pause routing to a pool's workers while it recovers."""
        # Only the ready (routable) set is touched: workers stay
        # registered (alive, may resolve in-flight responses; a later
        # on_worker_stop must still cancel their pending requests).
        # Worker-ready events re-enable the pool after recovery; the
        # pause is idempotent.
        # Candidates derive from the ready set with .get(pid), the same
        # derivation the routing rebuild uses — so anything routable in
        # this pool is paused, including a worker whose tag is absent
        # rather than explicitly None.
        quarantined = [
            pid
            for pid in self._ready_workers
            if self._worker_pools.get(pid) == env_hash
        ]
        for pid in quarantined:
            self._ready_workers.pop(pid, None)
        if quarantined:
            self._reset_round_robin()

    def _sticky_worker(
        self, model_name: str, model_version: str | None = None
    ) -> tuple[Worker, int] | None:
        """Model-affine worker selection for batched models, or None.

        The assignment is derived from a stable hash of the model name and
        the wall clock, so every frontend (and the single-process server)
        computes the same worker without coordination. The per-model
        rotation period jitter prevents permanently colliding models at
        small worker counts. Only workers of the model's own pool are
        candidates: a batcher must not be split across pools.
        """
        pool = self._model_pools.get((model_name, model_version or ""))
        pids = self._pool_pids.get(pool, [])
        # A single worker is trivially sticky: parallel_routing_rotation
        # has no effect until a second worker exists.
        if len(pids) < 2 or (model_name, model_version or "") not in (
            self._batched_models
        ):
            return None
        rotation = self._routing_rotation
        digest = zlib.crc32(model_name.encode())
        if rotation > 0:
            period = rotation * (1 + (digest >> 8) % 4)
            quotient = time.time() / period
            # A period so small that the quotient overflows the float
            # range (e.g. rotation=1e-310) or exceeds 2**53 loses all
            # integer precision: the assignment could not advance
            # anyway, so degrade to static routing instead of raising
            # OverflowError on every request.
            generation = int(quotient) if quotient <= 2**53 else 0
        else:
            generation = 0
        pid = pids[(digest + generation) % len(pids)]
        worker = self._ready_workers.get(pid)
        if worker is None:
            # The snapshot went stale between rotation windows; fall
            # through to round-robin rather than raising.
            return None
        return worker, pid

    def _reset_round_robin(self) -> Iterator[int]:
        worker_pids = list(self._ready_workers.keys())
        self._sorted_ready_pids = sorted(self._ready_workers)
        self._workers_round_robin = cycle(worker_pids)
        # Partitions are derived from the ready set, so tags of workers
        # removed outside on_worker_stop (e.g. a pool quarantine clearing
        # the ready table directly) cannot leak into routing.
        by_pool: dict[str | None, list[int]] = {}
        for pid in self._sorted_ready_pids:
            by_pool.setdefault(self._worker_pools.get(pid), []).append(pid)
        self._pool_pids = by_pool
        self._pool_round_robins = {pool: cycle(pids) for pool, pids in by_pool.items()}
        return self._workers_round_robin

    @with_operation_lock(lambda self, *args, **kwargs: self._worker_starting_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._worker_start_timeout)
    async def on_worker_start(
        self,
        worker: Worker,
        init_coro: Awaitable[None] | None = None,
    ):
        """
        Handler for workers who have just started but are still not ready to
        receive traffic. Holds the lock throughout init_coro (Phase 1/2 model
        replay) so no concurrent dispatch_update can reach the worker in an
        inconsistent state during initialization.
        """
        self._workers[worker.pid] = worker  # type: ignore
        if init_coro is not None:
            await init_coro

    def on_worker_ready(self, worker: Worker, env_hash: str | None = None):
        """
        Handler for workers who are now ready to receive traffic. The
        optional env hash tags the worker with the pool that owns it; it
        partitions routing so a worker only serves its own pool's models
        (None, the default, is the default pool).
        """
        # A ready worker is by definition started: register it in the
        # private table too. The pool seeds its initial workers without
        # a separate on_worker_start, so this is where the dispatcher's
        # table first learns about them (dispatch_update broadcasts over
        # this table).
        self._workers[worker.pid] = worker  # type: ignore
        self._ready_workers[worker.pid] = worker  # type: ignore
        self._worker_pools[worker.pid] = env_hash  # type: ignore
        self._reset_round_robin()

    def on_worker_stop(self, worker: Worker, exit_code: int):
        """
        Handler for workers that stopped unexpectedly. Removes the worker from the
        active pool, resets the round-robin iterator, and cancels all
        in-flight requests assigned to that worker.
        """
        pid = worker.pid
        removed = False
        if pid is not None and self._workers.get(pid) is worker:
            del self._workers[pid]
            removed = True

        if pid is not None and self._ready_workers.get(pid) is worker:
            del self._ready_workers[pid]
            removed = True

        if removed and pid is not None:
            # Drop this incarnation's pool tag. A quarantined worker is
            # only in _workers (its ready entry was paused), so the tag
            # must not depend on the ready-branch above. A replacement
            # worker sharing the pid re-tags on its own worker-ready.
            self._worker_pools.pop(pid, None)

        self._reset_round_robin()
        try:
            if self._transport is not None and pid is not None:
                self._transport.forget_worker(worker, exit_code)
        finally:
            # Socket retirement can fail after routing has been revoked. The
            # response waiters still need this incarnation's terminal result.
            self._async_responses.cancel(worker, exit_code)

    def start(self):
        self._active = True
        reply = getattr(self._responses, "_zmq_reply_endpoint", None)
        if reply is not None and self._transport is None:
            from .zmq import ParentTransport

            self._transport = ParentTransport(reply)
        if self._transport is not None:
            self._transport.start(
                self._async_responses.resolve, self._async_responses.fail_all
            )
        self._process_responses_task = schedule_with_callback(
            self._process_responses(), self._process_responses_cb
        )

    def _process_responses_cb(self, process_responses):
        try:
            process_responses.result()
        except asyncio.CancelledError:
            # NOTE: The response loop was cancelled from the outside, so don't
            # restart
            return
        except Exception:
            if not self._active:
                # A failure already queued when close started must not create
                # a new reader after its executor has been shut down.
                return
            logger.exception("Response processing loop crashed. Restarting the loop...")
            # If process loop crashed, restart it
            self.start()

    async def _wait_request(self, message_id: str) -> ModelResponseMessage:
        """Wait for an inference reply, bounded by the configured deadline.

        A reply dropped in flight (e.g. a frontend that stopped
        receiving) would otherwise leave the waiter pending forever
        with no server-side signal. ``None`` preserves the historical
        unbounded wait; a configured deadline raises TimeoutError to
        the caller and clears the pending entry (a late reply resolves
        to no waiter).
        """
        if self._inference_timeout is None:
            return await self._async_responses._wait(message_id)
        return await asyncio.wait_for(
            self._async_responses._wait(message_id), self._inference_timeout
        )

    async def _process_responses(self):
        logger.debug("Starting response processing loop...")
        loop = asyncio.get_running_loop()
        while self._active:
            response = await loop.run_in_executor(self._executor, self._responses.get)

            # If the queue gets terminated, detect the "sentinel value" and
            # stop reading
            if response is END_OF_QUEUE:
                return

            self._async_responses.resolve(response)

    async def dispatch_request(
        self, request_message: ModelRequestMessage
    ) -> ModelResponseMessage:
        # Workers route each reply to the frontend that dispatched it.
        request_message.frontend_id = self._frontend_id
        worker, wpid = self._get_worker(
            request_message.model_name, request_message.model_version
        )
        if self._transport is not None:
            # Register before send: ZeroMQ can suspend and a response may
            # arrive before send returns, unlike the old Queue.put boundary.
            future = self._async_responses._schedule(request_message, worker)
            try:
                await self._transport.send(worker, request_message)
            except asyncio.CancelledError:
                # Caller cancellation racing send completion: the request
                # may already be in the worker's batcher. Notify so the
                # batcher evicts it instead of computing a discarded
                # response. Clean up the waiter before re-raising.
                self._notify_cancelled(request_message, wpid)
                if not future.cancel():
                    if not future.cancelled():
                        future.exception()
                self._async_responses._clear_message(request_message.id)
                raise
            except BaseException:
                if not future.cancel():
                    if not future.cancelled():
                        future.exception()
                self._async_responses._clear_message(request_message.id)
                raise
            try:
                return await self._wait_request(request_message.id)
            except asyncio.CancelledError:
                self._notify_cancelled(request_message, wpid)
                raise
            except TimeoutError:
                # Inference deadline expiry: evict the request from the
                # worker-side batcher instead of computing a response
                # nobody waits for.
                self._notify_cancelled(request_message, wpid)
                raise
        worker.send_request(request_message)

        try:
            return await self._async_responses.schedule_and_wait(
                request_message, worker, timeout=self._inference_timeout
            )
        except asyncio.CancelledError:
            self._notify_cancelled(request_message, wpid)
            raise
        except TimeoutError:
            # Inference deadline expiry: evict the request from the
            # worker-side batcher instead of computing a response
            # nobody waits for.
            self._notify_cancelled(request_message, wpid)
            raise

    def _notify_cancelled(
        self, request_message: ModelRequestMessage, worker_pid: int
    ) -> None:
        # Fire-and-forget on caller cancellation (e.g. an HTTP client
        # disconnect): lets the owning topology evict the queued request
        # from its worker-side batcher instead of computing it anyway.
        if self._cancel_handler is not None:
            try:
                self._cancel_handler(
                    request_message.model_name,
                    request_message.id,
                    worker_pid,
                    request_message.model_version,
                )
            except Exception:
                logger.debug("Cancel notification failed", exc_info=True)

    def _get_worker(
        self, model_name: str | None = None, model_version: str | None = None
    ) -> tuple[Worker, int]:
        """
        Get next available worker.
        Batched models use model-affine routing so their worker-side
        batcher sees the full traffic rate; everything else round-robins.
        Selection is confined to the model version's pool partition, so
        requests never cross into a pool that did not load the model.
        """
        if model_name is not None:
            sticky = self._sticky_worker(model_name, model_version)
            if sticky is not None:
                return sticky
        # An untagged model routes to the default pool (None), matching
        # workers registered without a pool tag.
        pool = (
            self._model_pools.get((model_name, model_version or ""))
            if model_name
            else None
        )
        try:
            worker_pid = next(self._pool_round_robins[pool])
        except (KeyError, StopIteration):
            raise NoWorkersAvailable() from None
        # A foreign loop (worker replacement on another event loop) can
        # retire the worker between the cycle's snapshot and this
        # lookup: report no workers rather than surfacing a KeyError.
        worker = self._ready_workers.get(worker_pid)
        if worker is None:
            raise NoWorkersAvailable() from None
        return worker, worker_pid

    @with_operation_lock(lambda self, *args, **kwargs: self._worker_starting_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def dispatch_update(
        self, model_update: ModelUpdateMessage
    ) -> list[ModelResponseMessage]:
        # Dispatch to all workers concurrently and wait for results
        # to ensure state is properly maintained
        results = await asyncio.gather(
            *[
                self.dispatch_update_to_worker(worker, model_update)
                for worker in self._workers.values()
            ],
            return_exceptions=True,
        )
        non_stop = None
        successes = []
        for r in results:
            if isinstance(r, ModelResponseMessage):
                successes.append(r)
            elif not isinstance(r, WorkerStop):
                non_stop = non_stop or r

        # If any error other than a worker stop error is returned raise immediately
        if non_stop:
            raise non_stop
        # If every worker stopped without returning a successful response, a
        # load cannot establish model state and must fail so the caller can
        # roll back. Unload remains successful because the desired state is
        # already removal.
        if not successes and model_update.update_type == ModelUpdateType.Load:
            raise NoWorkersAvailable()

        # If at least one result is a success, this means any workers that crashed
        # will properly recover on replay
        return cast(list[ModelResponseMessage], successes)

    async def dispatch_update_to_worker(
        self, worker: Worker, model_update: ModelUpdateMessage
    ) -> ModelResponseMessage:
        # NOTE: Need to rewrite the UUID to ensure each worker sends back a
        # unique result
        worker_update = model_update.copy()
        worker_update.id = generate_uuid()
        worker.send_update(worker_update)
        return await self._async_responses.schedule_and_wait(worker_update, worker)

    async def stop(self, timeout: float = 5):
        self._active = False
        self._ready_workers.clear()
        self._reset_round_robin()
        # A worker killed while writing can leave a broken response queue.
        # Never block the event loop indefinitely in executor.shutdown().
        self._executor.shutdown(wait=False)
        failures: list[BaseException] = []
        if self._transport is not None:
            try:
                await self._transport.close()
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
        if self._process_responses_task is not None:
            # Cancelling an executor Future does not stop its Queue.get().
            # Wake the reader before joining it, while the queue is still open.
            if not self._process_responses_task.done():
                self._responses.put(END_OF_QUEUE)
            try:
                await cancel_task(self._process_responses_task)
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
        # Join the executor's threads through its public shutdown API —
        # the earlier shutdown(wait=False) left them running, and the
        # _threads list is a CPython implementation detail we must not
        # read. wait_for bounds the wait: a reader thread stuck on a
        # broken queue must not hang stop(), and the loop stays free
        # because the join itself runs in a thread.
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._executor.shutdown, wait=True, cancel_futures=True
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            failures.append(
                RuntimeError(
                    "Response reader did not exit; " "dispatcher resources remain owned"
                )
            )
        except Exception as error:
            failures.append(error)
        if failures:
            raise failures[0]
