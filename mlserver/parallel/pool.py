import asyncio
import functools
import random

from contextlib import nullcontext
from weakref import WeakSet
from multiprocessing import Queue
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Awaitable, Callable, Iterable, Sequence
from prometheus_client import Counter

from ..model import MLModel
from ..registry import _model_load_hooks_active
from ..utils import defer_cancellation, with_operation_lock, create_task_independent
from ..types import InferenceRequest, InferenceResponse
from ..settings import Settings, ModelSettings
from ..env import Environment

from .model import ParallelModel
from .errors import InferencePoolUnavailable, WorkerError, WorkerStop
from .worker import Worker, WorkerModelHook
from .logging import logger
from .utils import configure_inference_pool, terminate_queue
from .messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)
from .dispatcher import Dispatcher

PredictMethod = Callable[[InferenceRequest], Awaitable[InferenceResponse]]
InferencePoolHook = Callable[[Worker], Awaitable[None]]
# Receives the quarantined pool's env hash (None = the default pool).
PoolQuarantineHook = Callable[[str | None], Awaitable[None]]


def _spawn_worker(pool: "InferencePool") -> Worker:
    worker = Worker(
        pool._settings,
        pool._responses,
        pool._env,
        pool._on_worker_load,
        pool._on_worker_unload,
    )
    # Tag the worker with its pool's routing identity before any
    # broadcast can observe it, so frontends route only same-pool
    # models to it.
    worker.env_hash = pool.routing_id
    # Worker construction is lightweight. Own it before queue acquisition and
    # start(), which can raise even after exposing a native child.
    pool._unstarted_workers.add(worker)
    with pool._env or nullcontext():
        worker.start()
    assert worker.pid is not None
    pool._starting_workers[worker.pid] = worker
    pool._unstarted_workers.discard(worker)
    return worker


class WorkerRegistry:
    """
    Simple registry to keep track of which models have been loaded.
    This can be used to re-load all models when a worker stops unexpectedly.
    """

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelSettings] = {}

    def _key(self, model_settings: ModelSettings) -> tuple[str, str]:
        return (model_settings.name, model_settings.version or "")

    def add(self, model_settings: ModelSettings):
        model_key = self._key(model_settings)
        self._models[model_key] = model_settings.model_copy(deep=True)

    def remove(self, model_settings: ModelSettings):
        model_key = self._key(model_settings)
        if model_key in self._models:
            del self._models[model_key]

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._key(model_settings) in self._models

    def clear(self) -> None:
        self._models.clear()

    def __len__(self) -> int:
        return len(self._models)

    @property
    def models(self) -> Iterable[ModelSettings]:
        return self._models.values()


class InferencePool:
    """
    The InferencePool class represents a pool of workers where we can run
    inference on.

    Under the hood, it's responsible for managing a pool of multiprocessing
    workers, where the model is loaded.
    This approach lets MLServer work around the GIL to make sure that inference
    can occur in parallel across multiple models or instances of a model.
    """

    # Shared counter across all instances - created on first use (lazy init
    # required because PROMETHEUS_MULTIPROC_DIR must be set before any metric
    # objects are created, and that happens at server start, not import time)
    _PoolCleanupFailuresTotal = None

    @classmethod
    def _increment_cleanup_failure_metric(cls, pool_id: str | None):
        if cls._PoolCleanupFailuresTotal is None:
            cls._PoolCleanupFailuresTotal = Counter(
                "pool_cleanup_failures_total",
                "Total number of inference pool cleanup failures",
                ["pool_id"],
            )
        cls._PoolCleanupFailuresTotal.labels(pool_id=pool_id or "").inc()

    def __init__(
        self,
        settings: Settings,
        env: Environment | None = None,
        pool_gid: str | None = None,
        on_worker_stop: Sequence[InferencePoolHook] = [],
        on_worker_load: Sequence[WorkerModelHook] = [],
        on_worker_unload: Sequence[WorkerModelHook] = [],
        on_worker_ready: Sequence[InferencePoolHook] = [],
        on_pool_quarantined: Sequence[PoolQuarantineHook] = [],
        reply_endpoints: Sequence[str] | None = None,
    ):
        # Copy the hook sequences: the defaults are shared mutable
        # lists, and storing them by reference would let one pool's
        # appends (e.g. a test hook) fire on every other pool.
        self._on_worker_stop = list(on_worker_stop)
        self._on_worker_load = list(on_worker_load)
        self._on_worker_unload = list(on_worker_unload)
        self._on_worker_ready = list(on_worker_ready)
        self._on_pool_quarantined = list(on_pool_quarantined)
        # Pre-created reply endpoints (one per frontend process). When
        # provided, workers reply to every frontend and the pool's own
        # dispatcher runs transport-free: data-plane requests come from
        # the frontends, model-control acks still use the response queue.
        self._reply_endpoints = (
            list(reply_endpoints) if reply_endpoints is not None else None
        )
        self._env = env
        self._pool_gid = pool_gid
        self._workers: dict[int, Worker] = {}
        # Refcount of loaded model versions whose batcher runs in workers.
        self._batched_names: dict[tuple[str, str], int] = {}
        # Acquisition owners are distinct from registered/routable workers.
        # Close and SIGCHLD must see them before dispatch can safely use them.
        self._starting_workers: dict[int, Worker] = {}
        self._unstarted_workers: set[Worker] = set()
        self._start_task: asyncio.Task | None = None
        self._started = False
        self._initialization_error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        # Replay holds committed configuration. Staged/pending replacements
        # also own resources, so this inventory alone cannot decide pool lifetime.
        self._worker_registry = WorkerRegistry()
        self._pending_reload: dict[tuple[str, str], MLModel] = {}
        self._closing = False
        self._close_task: asyncio.Task | None = None
        # Confirmed-exit workers can still own unretired queues while a later
        # incarnation reuses their PID. Keep both resource owners by identity.
        self._closing_workers: dict[int, Worker] = {}
        self._restart_tasks: set[asyncio.Task] = set()
        self._recovery_task: asyncio.Task | None = None
        self._staged: dict[ParallelModel, asyncio.Future] = {}
        self._quarantined_workers: dict[int, Worker] = {}
        self._retired_models: WeakSet[MLModel] = WeakSet()
        self._operation_lock = asyncio.Lock()
        self._settings = settings
        self._response_queue: Queue[ModelResponseMessage] | None = None
        self._owned_dispatcher: Dispatcher | None = None
        self._transport_directory: TemporaryDirectory[str] | None = None

    @property
    def _responses(self) -> "Queue[ModelResponseMessage]":
        assert self._response_queue is not None
        return self._response_queue

    @property
    def _dispatcher(self) -> Dispatcher:
        assert self._owned_dispatcher is not None
        return self._owned_dispatcher

    @defer_cancellation(lambda self: self._settings.model_operation_timeout)
    async def start(self) -> None:
        """Activate only after a manager has retained this lightweight owner."""
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        if self._start_task is None:
            self._start_task = create_task_independent(self._start())
        try:
            await asyncio.shield(self._start_task)
        except (Exception, asyncio.CancelledError) as primary:
            self._initialization_error = primary
            try:
                await self.close()
            except (Exception, asyncio.CancelledError) as cleanup:
                self._cleanup_error = cleanup
                setattr(primary, "cleanup_error", cleanup)
            raise

    @with_operation_lock(lambda self: self._operation_lock)
    @defer_cancellation(lambda self: self._settings.model_operation_timeout)
    async def _start(self) -> None:
        await self._activate()
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        self._started = True

    async def _activate(self) -> None:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        configure_inference_pool(self._settings)
        self._response_queue = Queue()
        if self._settings.parallel_inference_transport == "zmq":
            from .zmq import create_endpoint

            if self._reply_endpoints is not None:
                # Shared-pool topology: the supervisor pre-created one reply
                # endpoint per frontend. Workers connect to all of them and
                # route each reply by the request's frontend_id. The pool's
                # own dispatcher stays transport-free (frontend-owned
                # endpoints must not be bound twice): advertising the
                # singular attribute at ANY frontend count makes the pool's
                # dispatcher bind a frontend's reply path, and the worker's
                # reply connection then lands on whichever socket owned the
                # path at connect time — the frontend's inference responses
                # can be swallowed by the pool's socket. Update responses
                # already travel the multiprocessing response queue, so the
                # pool needs no zmq transport here.
                endpoints = list(self._reply_endpoints)
                setattr(self._responses, "_zmq_reply_endpoints", endpoints)
                setattr(self._responses, "_zmq_reply_endpoint", None)
            else:
                # Only endpoint strings cross spawn. The parent owns the
                # private directory until workers and the response transport
                # have stopped.
                self._transport_directory = TemporaryDirectory(prefix="mlserver-ipc-")
                reply = create_endpoint(Path(self._transport_directory.name), "r")
                # The list form supports one endpoint per frontend process;
                # the single-endpoint attribute stays for backward
                # compatibility.
                setattr(self._responses, "_zmq_reply_endpoints", [reply])
                setattr(self._responses, "_zmq_reply_endpoint", reply)
        # The supervisor's own dispatcher defaults to frontend_id=0.
        # It serves model OPERATIONS (load/unload broadcasts) only —
        # never route live inference through it in multi-frontend
        # mode: dispatch_request stamps frontend_id on the message,
        # and every reply would land on frontend 0's endpoint alone,
        # leaving frontends 1..N-1 with permanently pending waiters.
        self._owned_dispatcher = Dispatcher(
            self._workers,
            self._responses,
            model_operation_timeout=self._settings.model_operation_timeout,
            worker_start_timeout=self._settings.worker_start_timeout,
            routing_rotation=self._settings.parallel_routing_rotation,
            inference_timeout=self._settings.inference_timeout,
        )
        self._dispatcher.start()
        for _ in range(self._settings.parallel_workers):
            if self._closing:
                raise InferencePoolUnavailable(self.name)
            worker = _spawn_worker(self)
            assert worker.pid is not None
            self._workers[worker.pid] = worker
            self._starting_workers.pop(worker.pid, None)
            self._dispatcher.on_worker_ready(worker)
            await self._notify_worker_ready(worker)

    @property
    def env_hash(self) -> str | None:
        if not self._env:
            return None

        return self._env.env_hash

    @property
    def routing_id(self) -> str | None:
        """Identity used to partition frontend routing by pool.

        Every pool must have a distinct routing id: environment pools
        use their env hash, GID-only pools use their GID — otherwise a
        GID-only pool's workers and models would all be tagged None and
        collapse into the default pool's routing partition. The value
        is also the registry's ``_pools`` key for the pool.
        """
        return self.env_hash if self._env else self._pool_gid

    async def _notify_worker_ready(self, worker: Worker) -> None:
        """Notify observer hooks (e.g. the supervisor broadcasting to
        frontends) that a worker incarnation became routable."""
        for hook in self._on_worker_ready:
            try:
                await hook(worker)
            except Exception:
                logger.exception("Worker-ready hook failed; continuing")

    def _track_batched(
        self,
        model_settings: ModelSettings,
        loaded: bool,
        version: str | None = None,
    ) -> None:
        """Keep the dispatcher's batched-model set in sync with loads.

        Routing is per (name, version) — the same key the dispatcher's
        sticky routing uses. A refcount keeps mixed-version loads (one
        batched, one not) correct within each version key.

        At frontend_workers == 0 (the direct server) the batcher runs
        in the main process (the previous behaviour): requests are
        already merged before dispatch, and sticky routing would send
        every batch to one worker — round-robin is correct there. Only
        the supervised topology (batcher in the worker) needs the
        affinity mark.
        """
        batched = (
            model_settings.max_batch_size > 1 and model_settings.max_batch_time > 0
        )
        if not batched:
            return
        if self._settings.frontend_workers == 0:
            # Direct server: the main-process batcher merges before
            # dispatch; the dispatcher should round-robin the merged
            # batches across all workers.
            return
        resolved_version = version or model_settings.version or ""
        key = (model_settings.name, resolved_version)
        count = self._batched_names.get(key, 0) + (1 if loaded else -1)
        if count > 0:
            self._batched_names[key] = count
            self._dispatcher.set_model_batched(
                model_settings.name, True, resolved_version
            )
        else:
            self._batched_names.pop(key, None)
            self._dispatcher.set_model_batched(
                model_settings.name, False, resolved_version
            )

    @property
    def pool_gid(self) -> str | None:
        return self._pool_gid

    @property
    def name(self) -> str:
        if self.env_hash:
            return f"inference pool with hash '{self.env_hash}'"

        if self._pool_gid:
            return f"inference pool with GID '{self._pool_gid}'"

        return "default inference pool"

    def tracks_pid(self, pid: int) -> bool:
        """Whether pid is a worker of this pool in any lifecycle state.

        Covers started, starting, unstarted and closing workers — the
        full set whose exits ``on_worker_stop`` processes. The SIGCHLD
        reaper uses this to tell pool workers from other children of
        the server process (e.g. frontends in the multi-frontend
        topology).
        """
        if pid in self._workers or pid in self._starting_workers:
            return True
        if any(worker.pid == pid for worker in self._unstarted_workers):
            return True
        return any(worker.pid == pid for worker in self._closing_workers.values())

    async def on_worker_stop(self, pid: int, exit_code: int):
        # If the inference pool is closing, or the current worker
        # is not in this inference pool, worker stop handling
        # should be skipped
        for closing in self._closing_workers.values():
            if closing.pid == pid and closing.exitcode is None:
                closing.record_exit(exit_code)
                return
        if pid not in self._workers:
            # Registration may be waiting for the dispatcher lock. Its startup
            # task still owns this child, including waitpid's confirmed status.
            starting = self._starting_workers.get(pid)
            if starting is not None:
                starting.record_exit(exit_code)
            for worker in self._unstarted_workers:
                if worker.pid == pid:
                    worker.record_exit(exit_code)
            return

        worker = self._workers[pid]
        worker.record_exit(exit_code)
        logger.warning(
            f"Worker with PID {worker.pid} on {self.name} stopped "
            f"unexpectedly with exit code {exit_code}. "
            "Triggering worker restart..."
        )

        # Dispatcher revocation removes active ownership and can then raise
        # during socket cleanup. Retain retirement before that fallible step.
        self._closing_workers[id(worker)] = worker
        failures: list[BaseException] = []
        try:
            try:
                self._dispatcher.on_worker_stop(worker, exit_code)
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
            if self._workers.get(pid) is worker:
                del self._workers[pid]
            # Best effort execution of on worker stop hooks
            try:
                results = await asyncio.gather(
                    *[callback(worker) for callback in self._on_worker_stop],
                    return_exceptions=True,
                )
                failures.extend(r for r in results if isinstance(r, BaseException))
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
        finally:
            # The child has already been reaped by the SIGCHLD handler. Clean
            # up the parent-side queue resources before discarding the worker.
            try:
                await worker.retire_queues(self._settings.parallel_workers_timeout)
            except (Exception, asyncio.CancelledError) as error:
                # Keep the existing logged/retained result for an ordinary
                # queue-only failure. Revocation/hook failure still exposes
                # this secondary error; genuine cancellation must propagate.
                if failures or isinstance(error, asyncio.CancelledError):
                    failures.append(error)
                else:
                    self._cleanup_error = error
                logger.error(
                    f"Queue cleanup for worker {pid} remains owned until close",
                    exc_info=True,
                )
            else:
                self._closing_workers.pop(id(worker), None)

            # Restart independently of the signal handler and its inherited
            # timeout context. Keep each owner until its result is observed.
            if not self._closing:
                task = create_task_independent(self._safe_start_worker())
                self._restart_tasks.add(task)
                task.add_done_callback(self._restart_tasks.discard)
        if failures:
            self._cleanup_error = failures[0]
            if len(failures) > 1:
                setattr(failures[0], "cleanup_error", failures[1])
            self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
            raise failures[0]

    async def _safe_start_worker(self):
        """
        Fire-and-forget worker restart for use with asyncio.create_task.
        Logs failures rather than raising.
        """
        # Jitter before spawning: rate-limits rapid fork cycles and guarantees
        # a lock-free window so operator unloads can break a crash loop
        await asyncio.sleep(random.uniform(0.5, 2.0))
        if self._closing:
            return
        if self._staged or self._recovery_task is not None:
            # The dependency may outlast worker initialization's deadline.
            # Transfer the obligation to another retained owner; do not turn
            # dependency expiry into a dropped replacement or a busy retry.
            task = create_task_independent(self._finish_restart())
            self._restart_tasks.add(task)
            task.add_done_callback(self._restart_tasks.discard)
            return
        await self._finish_restart()

    async def _finish_restart(self) -> None:
        try:
            await self._start_worker()
        except Exception:
            logger.error(
                f"Failed to start replacement worker on {self.name}",
                exc_info=True,
            )

    async def _start_worker(self) -> Worker | None:
        # Wait without the pool lock: lifecycle completion may need that lock.
        # Recheck after acquisition because new preparation can arrive meanwhile.
        while not self._closing:
            if self._staged:
                await asyncio.gather(
                    *[asyncio.shield(f) for f in self._staged.values()]
                )
            await self._settle_recovery()
            async with self._operation_lock:
                if self._closing:
                    return None
                if self._staged or self._recovery_task is not None:
                    continue
                return await self._initialize_worker()
        return None

    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.worker_start_timeout
    )
    async def _initialize_worker(self) -> Worker | None:
        if self._closing:
            return None
        # Spawn may acquire queues or expose a child before returning. The
        # operation lock serializes attempts; select only this attempt's new
        # owner, leaving older failed cleanup and healthy siblings untouched.
        unstarted = self._unstarted_workers.copy()
        try:
            worker = _spawn_worker(self)
        except (Exception, asyncio.CancelledError) as primary:
            for candidate in self._unstarted_workers - unstarted:
                try:
                    await self._rollback_worker(candidate)
                except (Exception, asyncio.CancelledError) as cleanup:
                    setattr(primary, "cleanup_error", cleanup)
                    logger.error(
                        "Replacement acquisition rollback failed; owner retained",
                        exc_info=True,
                    )
            raise
        assert worker.pid is not None
        # Own the PID before registration can suspend, without making this
        # child routable or eligible for lifecycle update broadcasts.
        self._starting_workers[worker.pid] = worker
        logger.info(f"Starting new worker with PID {worker.pid} on {self.name}...")

        # Phase 1/2 replay runs inside the dispatcher lock via on_worker_start
        # to ensure no concurrent dispatch_update can reach the replacement
        # worker in an inconsistent state during initialization
        replay = self._register_and_replay(worker)
        try:
            await self._dispatcher.on_worker_start(worker, replay)
            if not self._closing:
                exit_code = worker.exitcode
                if exit_code is not None:
                    # Empty replay can complete without receiving a response.
                    # Observe a dead child instead of publishing it as ready;
                    # this also handles an exit polled before the SIGCHLD task.
                    await self.on_worker_stop(worker.pid, exit_code)
                    raise WorkerStop(exit_code)
        except (Exception, asyncio.CancelledError) as primary:
            # Replay can fail through an operation error, timeout, or
            # cancellation. In every case the replacement worker is not safe
            # to publish as ready.
            try:
                await self._rollback_worker(worker)
            except (Exception, asyncio.CancelledError) as cleanup:
                setattr(primary, "cleanup_error", cleanup)
                logger.error(
                    "Replacement rollback failed; owner retained", exc_info=True
                )
            raise
        finally:
            # A deadline while waiting for dispatcher registration may leave
            # replay unawaited. Startup has settled, so it cannot run later.
            replay.close()
            if self._starting_workers.get(worker.pid) is worker:
                self._starting_workers.pop(worker.pid, None)

        if not self._closing:
            self._dispatcher.on_worker_ready(worker)
            await self._notify_worker_ready(worker)
            logger.info(
                f"New worker with PID {worker.pid} on {self.name} is now ready."
            )
        else:
            assert worker.pid is not None
            if not worker.queues_retired:
                self._closing_workers[id(worker)] = worker
            self._dispatcher.on_worker_stop(worker, -1)
            # Revoke the pool-side registration made by
            # _register_and_replay (no-op if registration never ran):
            # the dispatcher no longer deletes from the pool's table,
            # and a closed-out replacement must not linger in _workers.
            if self._workers.get(worker.pid) is worker:
                del self._workers[worker.pid]
            return None
        return worker

    async def _rollback_worker(self, worker: Worker) -> None:
        # Transfer before revocation: it can remove the active entry and fail.
        # An unresolved socket remains owned by the transport independently.
        if not worker.queues_retired:
            self._closing_workers[id(worker)] = worker
        # Revoke the pool-side registration made by _register_and_replay
        # (under the worker-starting lock): the dispatcher's table is
        # private now, so without this delete a rolled-back replacement
        # would be treated as an unexpected death by a later
        # on_worker_stop (restart chain, queue retirement) for a child
        # this rollback already retired. No-ops for a worker whose
        # registration never ran.
        if worker.pid is not None and self._workers.get(worker.pid) is worker:
            del self._workers[worker.pid]
        # Acquisition can fail before a PID exists. Drop its earlier slot only
        # after the retirement owner above can manage those partial resources.
        self._unstarted_workers.discard(worker)
        failures: list[BaseException] = []
        try:
            self._dispatcher.on_worker_stop(worker, -1)
        except (Exception, asyncio.CancelledError) as error:
            failures.append(error)
        # Close joins accepted startup and sweeps these retained workers. If it
        # already took ownership, do not start a competing native retirement.
        if not self._closing and not worker.queues_retired:
            try:
                # Partial queue allocation has no PID. It still needs parent
                # resource cleanup, without claiming a child ever existed.
                if worker.pid is not None and worker.exitcode is None:
                    worker.kill()
                    await asyncio.to_thread(
                        worker.join, self._settings.parallel_workers_timeout
                    )
                if worker.pid is not None and worker.exitcode is None:
                    raise RuntimeError(f"Worker {worker.pid} has not confirmed exit")
                await worker.retire_queues(self._settings.parallel_workers_timeout)
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
            else:
                self._closing_workers.pop(id(worker), None)
        if failures:
            self._cleanup_error = failures[0]
            if len(failures) > 1:
                setattr(failures[0], "cleanup_error", failures[1])
            raise failures[0]

    async def _register_and_replay(self, worker: Worker):
        """Register the replacement pool-side, then run its replay.

        Runs as on_worker_start's init coroutine — i.e. under the
        worker-starting lock — so the pool's table gains the child at
        exactly the point the pre-split shared table did: after the
        lock is acquired, before any replay work. This keeps the child
        visible to tracks_pid, close()'s sweep and find_worker from
        registration until retirement, with no blind window between
        the finally popping _starting_workers and readiness. A spawn
        whose registration never runs (lock deadline, cancellation) is
        rolled back by _rollback_worker's identity-guarded delete.
        """
        assert worker.pid is not None
        self._workers[worker.pid] = worker
        return await self._replay_worker(worker)

    async def _replay_worker(self, worker: Worker):
        """
        Replay all model state to a replacement worker. Runs inside the
        dispatcher lock (via on_worker_start) so no concurrent dispatch_update
        can interleave with initialization.
        """
        # If the pool started closing while waiting for the lock, skip replay.
        if self._closing:
            return

        # Phase 1: load all committed models from the worker registry
        load_results = await asyncio.gather(
            *[
                self._dispatcher.dispatch_update_to_worker(
                    worker,
                    ModelUpdateMessage(
                        update_type=ModelUpdateType.Load,
                        model_settings=model_settings,  # type: ignore
                    ),
                )
                for model_settings in self._worker_registry.models
            ],
            return_exceptions=True,
        )
        load_failures = [r for r in load_results if isinstance(r, BaseException)]
        if load_failures:
            raise load_failures[0]

        # Phase 2: load all in-progress reload models so the replacement
        # mirrors the staged state of existing workers
        reload_results = await asyncio.gather(
            *[
                self._dispatcher.dispatch_update_to_worker(
                    worker,
                    ModelUpdateMessage(
                        update_type=ModelUpdateType.Load,
                        model_settings=model.settings,  # type: ignore
                    ),
                )
                for model in self._pending_reload.values()
            ],
            return_exceptions=True,
        )
        reload_failures = [r for r in reload_results if isinstance(r, BaseException)]
        if reload_failures:
            raise reload_failures[0]

    def _model_key(self, model_settings: ModelSettings) -> tuple[str, str]:
        return (model_settings.name, model_settings.version or "")

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def load_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        await self._settle_recovery()
        reload = self.has_model(model.settings)
        # Replay only committed fresh loads. Reloads are staged only after
        # every live worker acknowledges preparation.
        prepared: ModelUpdateMessage | None = None
        parallel_model: ParallelModel | None = None
        # Whether THIS load acquired the dispatcher's batch affinity:
        # the failure path must only reverse what it acquired, or a
        # worker error while preparing a second version would disable
        # the first version's affinity.
        affinity_acquired = False
        try:
            load_message = ModelUpdateMessage(
                update_type=ModelUpdateType.Load,
                model_settings=model.settings,  # type: ignore
                rollback=reload,
            )
            await self._dispatch_owned(load_message)
            self._track_batched(model.settings, True)
            affinity_acquired = True
            prepared = load_message
            if self._closing:
                raise InferencePoolUnavailable(self.name)
            self._retired_models.discard(model)
            parallel_model = self._wrap(model)
            if _model_load_hooks_active.get():
                # The pool hook is preparation, not frontend publication. Replay
                # waits for the registry to finish every subsequent hook as well.
                self._staged[parallel_model] = (
                    asyncio.get_running_loop().create_future()
                )
                parallel_model._on_load_complete = lambda: self._commit_model(
                    parallel_model, reload
                )
                parallel_model._on_load_failed = lambda: self._abandon_model(
                    parallel_model, load_message
                )
            else:
                self._commit_model(parallel_model, reload)
            return parallel_model
        except (Exception, asyncio.CancelledError):
            self._retire(model)
            # The load is abandoned: undo the dispatcher's batch
            # affinity only if this load acquired it — a failure before
            # the increment (e.g. a worker error during preparation of
            # a second version) must not disable a committed version's
            # affinity.
            if affinity_acquired:
                self._track_batched(model.settings, False)
            if prepared is not None and not self._closing:
                # Worker preparation succeeded, but wrapper construction or
                # frontend staging did not transfer ownership. Ordinary unload
                # could publish that abandoned worker replacement. Quarantine
                # and acknowledge rollback before any later mutation/replay.
                self._recover_prepared(prepared)
            if parallel_model is not None:
                self._release_stage(parallel_model)
            raise

    def _release_stage(self, model: ParallelModel) -> None:
        future = self._staged.pop(model, None)
        if future is not None and not future.done():
            future.set_result(None)

    def _commit_model(self, model: ParallelModel, reload: bool) -> None:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        if reload:
            self._pending_reload[self._model_key(model.settings)] = model
            # The commit is final in both paths: staged under the
            # registry's load hooks (frontend publication already ran)
            # or immediate (the supervisor drives the load with no hook
            # chain). A ready pending is what makes the retiring unload
            # PUBLISH the replacement instead of rolling it back (see
            # unload_model's rollback computation).
            model.ready = True
        else:
            self._worker_registry.add(model.settings)
        self._release_stage(model)

    def _abandon_model(self, model: ParallelModel, message: ModelUpdateMessage) -> None:
        if model not in self._staged:
            return
        self._retire(model)
        self._recover_prepared(message)
        self._release_stage(model)

    def _recover_prepared(self, message: ModelUpdateMessage) -> None:
        dispatch = asyncio.get_running_loop().create_future()
        dispatch.set_result(None)
        self._begin_recovery(dispatch, message)

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def unload_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        await self._settle_recovery()
        if model in self._retired_models:
            # A retry can target an instance whose retiring unload
            # already completed in the background: the earlier unload
            # outlived the caller's deadline (the shielded dispatch
            # kept running), so the caller re-drove the reload with a
            # NEW staged replacement. Publishing that replacement here
            # keeps the retry honest — without it, the retry would
            # report success while the PREVIOUS replacement still
            # serves on the workers.
            pending = self._pending_reload.get(self._model_key(model.settings))
            if pending is not None and pending.ready:
                self._pending_reload.pop(self._model_key(model.settings), None)
                self._worker_registry.add(pending.settings)
                # The promotion dispatch mirrors the retiring unload
                # that already ran: it retires the key's serving
                # version on the workers and promotes the staged one.
                # The instance's batch affinity was already released
                # by the earlier unload — do not decrement it again.
                unload_message = ModelUpdateMessage(
                    update_type=ModelUpdateType.Unload,
                    model_settings=model.settings,  # type: ignore
                    rollback=False,
                )
                await self._dispatch_owned(unload_message)
            return self._wrap(model)
        key = self._model_key(model.settings)
        pending_model = self._pending_reload.pop(key, None)
        rollback = pending_model is not None and not pending_model.ready
        if pending_model is not None and not rollback:
            # This is the publication point for replay. A later failure while
            # releasing the old instance must preserve the new settings.
            self._worker_registry.add(pending_model.settings)
        elif pending_model is None:
            self._worker_registry.remove(model.settings)
        self._retire(model)
        # Release batch affinity at the local commit point (with the
        # retirement above), BEFORE the dispatch: a dispatch deadline
        # arms recovery and skips any code after it, and a retry
        # returns early through _retired_models — both would otherwise
        # leak the affinity entry for every version retired this way.
        # The decrement runs exactly once here, whether the dispatch
        # confirms immediately or recovery settles it later, and never
        # touches a replacement's own affinity (acquired at its load).
        self._track_batched(model.settings, False)
        unload_message = ModelUpdateMessage(
            update_type=ModelUpdateType.Unload,
            model_settings=model.settings,  # type: ignore
            rollback=rollback,
        )
        await self._dispatch_owned(unload_message)

        # Wrap the model in the ParallelModel class so that the main process
        # does not load/unload the model (only done on workers)
        if isinstance(model, ParallelModel):
            return model
        return ParallelModel(model, self._dispatcher)

    def _wrap(self, model: MLModel) -> ParallelModel:
        return (
            model
            if isinstance(model, ParallelModel)
            else ParallelModel(model, self._dispatcher)
        )

    def _retire(self, model: MLModel) -> None:
        self._retired_models.add(model)
        if isinstance(model, ParallelModel):
            self._retired_models.add(model._model)

    async def _settle_recovery(self) -> None:
        while self._recovery_task is not None:
            task = self._recovery_task
            await asyncio.shield(task)
            if self._recovery_task is task:
                self._recovery_task = None

    async def _dispatch_owned(self, message: ModelUpdateMessage) -> None:
        # A frontend deadline cannot cancel an update already delivered to an
        # independent process. Keep the response waiter until it settles.
        dispatch = asyncio.create_task(self._dispatcher.dispatch_update(message))
        try:
            await asyncio.shield(dispatch)
        except (Exception, asyncio.CancelledError) as error:
            self._begin_recovery(dispatch, message, error)
            raise

    def _begin_recovery(
        self,
        dispatch: asyncio.Future,
        message: ModelUpdateMessage,
        error: BaseException | None = None,
    ) -> None:
        # A clean model-level error (WorkerError: the model's own
        # load()/unload() raised, reported by a live worker) is a
        # definitive per-model failure. For a failed LOAD the model was
        # never registered on any frontend (failed loads broadcast
        # ok=False, which frontends acknowledge without applying), so
        # no request can route to it — pausing the pool's routing
        # protects nothing and denies service to every UNRELATED model
        # for the rollback window (the single-process server serves
        # through failed loads). For a failed UNLOAD the model stays
        # registered while some workers may have removed it, so its
        # own requests can fail intermittently — a deliberate
        # trade-off: the blast radius stays on the failing model, and
        # the pre-fix alternative was strictly worse (an
        # unacknowledged unload's recovery preserves the failure, so
        # the quarantine would never release — the whole pool stayed
        # dark until close). The rollback unload below still runs for
        # loads: on a multi-worker pool another worker may hold a
        # staged instance. Unknown-state failures — a worker stop, a
        # transport error, a frontend deadline with the dispatch still
        # running, or the pre-dispatch staged path (error None) — keep
        # the full quarantine.
        pause_routing = error is None or not isinstance(error, WorkerError)
        if pause_routing:
            self._quarantined_workers.update(self._dispatcher._ready_workers)
            self._dispatcher._ready_workers.clear()
            self._dispatcher._reset_round_robin()
        previous = self._recovery_task

        async def recover():
            # Notify first (and promptly): frontends sharing this pool
            # must stop routing to its workers for the whole recovery.
            # Chained quarantines re-fire it; receivers treat the event
            # as idempotent.
            if pause_routing:
                await self._notify_pool_quarantined()
            if previous is not None:
                await previous
            await self._recover_update(dispatch, message)

        self._recovery_task = create_task_independent(recover())
        self._recovery_task.add_done_callback(
            functools.partial(self._recovery_done, paused=pause_routing)
        )

    async def _notify_pool_quarantined(self) -> None:
        """Fire the quarantine hooks with this pool's routing id."""
        # Hook failures are logged but never block recovery: a broken
        # observer must not prevent the pool from restoring itself.
        for hook in self._on_pool_quarantined:
            try:
                await hook(self.routing_id)
            except Exception:
                logger.exception("Pool-quarantined hook failed for %s", self.name)

    def _recovery_done(self, task: asyncio.Task, *, paused: bool) -> None:
        if task.cancelled() or task.exception() is not None:
            if paused:
                logger.error(
                    f"Recovery of {self.name} failed; pool remains "
                    "quarantined and owned until close.",
                    exc_info=None if task.cancelled() else task.exception(),
                )
            else:
                # Routing was never paused (a clean model-level failure):
                # traffic kept flowing; only the failing model's own
                # state is affected.
                logger.error(
                    f"Recovery of {self.name} failed after a clean "
                    "model-level error; routing was not paused.",
                    exc_info=None if task.cancelled() else task.exception(),
                )

    async def _recover_update(
        self,
        dispatch: asyncio.Future,
        message: ModelUpdateMessage,
    ) -> None:
        failed = None
        try:
            await dispatch
        except (Exception, asyncio.CancelledError) as error:
            failed = error
        if self._closing:
            return
        if message.update_type == ModelUpdateType.Load:
            # Settlement alone is insufficient: partial success may have left
            # staged instances on some workers. Restore the old committed state
            # with acknowledged cleanup before allowing any routing or replay.
            await self._dispatcher.dispatch_update(
                ModelUpdateMessage(
                    update_type=ModelUpdateType.Unload,
                    model_settings=message.model_settings,  # type: ignore
                    rollback=message.rollback,
                )
            )
        elif failed is not None:
            # An unacknowledged unload is unsafe to replay by name: a second
            # unload could remove the replacement which the first committed.
            # Preserve ownership and the failure instead of guessing state.
            raise failed
        if not self._closing and asyncio.current_task() is self._recovery_task:
            for pid, worker in self._quarantined_workers.items():
                if self._workers.get(pid) is worker:
                    self._dispatcher.on_worker_ready(worker)
                    await self._notify_worker_ready(worker)
            self._quarantined_workers.clear()

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._worker_registry.has_model(model_settings)

    def get_ready_worker_endpoints(self) -> dict[int, "Worker"]:
        """Public: the currently routable (ready) workers.

        The supervisor's restart replay uses this to re-announce
        worker incarnations to a replacement frontend. A pool retained
        by the registry but not yet activated (start() pending) has no
        dispatcher and no routable workers: an empty set keeps a
        concurrent frontend restart from aborting its replay on the
        dispatcher assertion — the workers announce themselves through
        the ready hook once the pool activates.
        """
        if self._owned_dispatcher is None:
            return {}
        return dict(self._dispatcher._ready_workers)

    def empty(self) -> bool:
        # Replay inventory excludes preparation, but those accepted operations
        # still own the pool. Do not auto-close it between staging and commit.
        # An abandoned update alone can retire through close(), which retains
        # recovery and confirms worker exit. Registry callers check
        # this under their pool lifecycle lock; those owner transfers do not yield.
        # The lock-holders are InferencePoolRegistry.load_model (its failure
        # path) and unload_model, each under the per-pool operation lock
        # resolved by _get_pool_lock (keyed by the pool's routing id), so no
        # concurrent owner transfer can interleave between this check and
        # _close_pool.
        # A LIVE recovery task does not block retirement: close() joins it
        # (bounded by model_operation_timeout) before clearing the worker
        # registry, and _recover_update re-checks _closing at each step, so
        # retiring with recovery in flight is a supported flow — the failed
        # fresh load path depends on it.
        return (
            len(self._worker_registry) == 0
            and not self._staged
            and not self._pending_reload
        )

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            # Dependency/lock waiters have not accepted native initialization.
            # Accepted initialization retains its existing deferred cancellation
            # boundary and completes (or times out) before this owner joins it.
            for task in list(self._restart_tasks):
                task.cancel()
            for model in list(self._staged):
                self._release_stage(model)
            self._close_task = create_task_independent(self._close())
            self._close_task.add_done_callback(self._close_done)
        await asyncio.shield(self._close_task)

    def _close_done(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            logger.error(
                f"Close of {self.name} failed; result and resources retained.",
                exc_info=None if task.cancelled() else task.exception(),
            )

    async def _close(self) -> None:
        cleanup_errors: list[BaseException] = []
        # Existing accepted operations get their configured window to settle.
        # If they stall, closure escalates to process exit rather than waiting
        # forever on a frontend lifecycle lock.
        try:
            await asyncio.wait_for(
                self._operation_lock.acquire(), self._settings.model_operation_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Lifecycle lock did not drain while closing {self.name}")
        else:
            # Release immediately after the drain: _closing is already
            # set, and every mutating path re-checks it under this lock
            # before touching pool state — so post-drain lock holders
            # can only observe _closing and back out. This invariant
            # (check-_closing-under-lock in every mutator) is what
            # makes the early release safe; a new mutator that skips
            # the re-check would race this window.
            self._operation_lock.release()

        if self._start_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._start_task),
                    self._settings.model_operation_timeout,
                )
            except (Exception, asyncio.CancelledError):
                # A settled initialization error is the primary failure, not
                # a cleanup failure. An unsettled task still owns activation.
                if not self._start_task.done():
                    cleanup_errors.append(RuntimeError("Pool activation remains owned"))

        # Best effort cleanup, including partially acquired resources.
        try:
            await self._close_workers()
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to fully cleanup workers on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            if self._response_queue is not None:
                await terminate_queue(self._responses)
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to terminate response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            if self._owned_dispatcher is not None:
                await self._dispatcher.stop(self._settings.parallel_workers_timeout)
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to stop dispatcher on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            # The dispatcher's executor may still be reading this handle.
            # Stop and join that reader before closing the queue.
            if self._response_queue is not None:
                self._responses.close()
                feeder = getattr(self._responses, "_thread", None)
                if feeder is not None:
                    await asyncio.to_thread(
                        feeder.join, self._settings.parallel_workers_timeout
                    )
                    if feeder.is_alive():
                        raise RuntimeError("Response queue writer remains owned")
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to close response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        if self._recovery_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._recovery_task),
                    self._settings.model_operation_timeout,
                )
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
        if self._restart_tasks:
            results = await asyncio.gather(*self._restart_tasks, return_exceptions=True)
            cleanup_errors.extend(
                r
                for r in results
                if isinstance(r, BaseException)
                and not isinstance(r, asyncio.CancelledError)
            )
        if self._closing_workers:
            # Startup may register after the first sweep (for example while
            # waiting for the dispatcher lock). No restart can spawn now;
            # collect transferred owners with the same bounded exit checks.
            try:
                await self._close_workers()
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)

        # The pool is terminal after close. No model state can be replayed or
        # rolled back, so release the tracking references after cleanup.
        self._worker_registry.clear()
        self._pending_reload.clear()

        if self._closing_workers or self._unstarted_workers:
            cleanup_errors.append(
                RuntimeError("Worker cleanup remains owned after restart settlement")
            )
        if cleanup_errors:
            # Track cleanup failures
            self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
            raise cleanup_errors[0]

        if self._transport_directory is not None:
            # Its endpoints are resources of the workers and dispatcher. Retain
            # their directory too if cleanup did not confirm their release.
            self._transport_directory.cleanup()

        if self._env is not None:
            # Release before the registry allows this hash to be reused.
            # On failed/uncertain close the environment remains owned above.
            self._env.close()

    async def _close_workers(self) -> None:
        cleanup_errors: list[BaseException] = []
        for worker in list(self._unstarted_workers):
            if worker.pid is not None:
                self._closing_workers[id(worker)] = worker
                self._unstarted_workers.discard(worker)
                continue
            try:
                await worker.retire_queues(self._settings.parallel_workers_timeout)
                self._unstarted_workers.discard(worker)
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
        self._closing_workers.update(
            (id(w), w) for w in self._starting_workers.values()
        )
        self._closing_workers.update((id(w), w) for w in self._workers.values())
        for worker in list(self._closing_workers.values()):
            pid = worker.pid
            if pid is None:
                # Failed acquisition can transfer real queues here before a
                # process exists. None is not a live-child exit status; release
                # parent resources without stop/kill/join or worker-stop hooks.
                try:
                    await worker.retire_queues(self._settings.parallel_workers_timeout)
                except (Exception, asyncio.CancelledError) as error:
                    cleanup_errors.append(error)
                else:
                    self._closing_workers.pop(id(worker), None)
                continue
            # Quarantine first, retaining the owner until process exit and
            # parent queue retirement are confirmed. A kill is only a request.
            try:
                if self._owned_dispatcher is not None:
                    self._dispatcher.on_worker_stop(worker, -1)
                # Revoke the pool-side registration alongside the
                # dispatcher's (its table is private): close() is
                # terminal, and a stopped worker left in _workers would
                # misreport tracking and capacity after close.
                if self._workers.get(pid) is worker:
                    del self._workers[pid]
            except (Exception, asyncio.CancelledError) as error:
                # Revocation/socket cleanup failure must not skip confirming
                # this child's exit or attempting independent worker cleanup.
                cleanup_errors.append(error)
            try:
                if worker.exitcode is None:
                    await asyncio.wait_for(
                        worker.stop(), self._settings.parallel_workers_timeout
                    )
                    await asyncio.to_thread(
                        worker.join, self._settings.parallel_workers_timeout
                    )
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
            try:
                if worker.exitcode is None:
                    worker.kill()
                    await asyncio.to_thread(
                        worker.join, self._settings.parallel_workers_timeout
                    )
                if worker.exitcode is None:
                    raise RuntimeError(f"Worker {pid} has not confirmed exit")
                await worker.retire_queues(self._settings.parallel_workers_timeout)
                self._closing_workers.pop(id(worker), None)
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
                continue

            results = await asyncio.gather(
                *[
                    asyncio.wait_for(
                        callback(worker), self._settings.parallel_workers_timeout
                    )
                    for callback in self._on_worker_stop
                ],
                return_exceptions=True,
            )
            cleanup_errors.extend(r for r in results if isinstance(r, BaseException))
        if cleanup_errors:
            raise cleanup_errors[0]
