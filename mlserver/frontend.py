# Multi-process frontend supervisor: at frontend_workers=1 (default)
# it is never used. At N>1 the supervisor owns the shared inference
# pool, spawns N frontends onto the same SO_REUSEPORT ports, and
# coordinates model lifecycle over inherited pipes. Shutdown order:
# frontends drain first, then the pool stops workers via control
# queues. Children arm PR_SET_PDEATHSIG so a crashed supervisor leaves
# no orphans.

import asyncio
import multiprocessing

import os
import signal

from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from .logging import logger
from .settings import Settings, ModelSettings
from .model import MLModel
from .utils import set_parent_death_signal, generate_uuid, consume_send_failure

from .parallel.registry import ENV_HASH_ATTR, InferencePoolRegistry
from .parallel.zmq import create_endpoint
from .parallel.messages import ModelUpdateMessage, ModelUpdateType
from .parallel.shared import (
    ApplyAck,
    ApplyLoad,
    ApplyUnload,
    CancelRequest,
    deserialise_model_settings,
    OpRequest,
    OpResult,
    PoolQuarantinedEvent,
    serialise_model_settings,
    StartupApplyFailed,
    StartupComplete,
    WorkerReadyEvent,
    WorkerStoppedEvent,
)


def _process_exited(proc: "multiprocessing.process.BaseProcess") -> bool:
    """True once the process has exited, regardless of who reaps it.

    ``is_alive()`` goes permanently blind when another reaper already
    consumed the corpse: the shared pool's SIGCHLD handler waitpids
    ``-1`` (reaping ANY child, frontends included) before filtering by
    pool membership, and ``Popen.poll()`` maps the resulting ECHILD to
    ``None`` — which ``is_alive()`` reads as still-running. The
    sentinel pipe is readable from the moment the child exits, no
    matter who reaps the corpse, so it stays a reliable exit signal.
    """
    if not proc.is_alive():
        return True
    from multiprocessing.connection import wait as connection_wait

    return bool(connection_wait([proc.sentinel], timeout=0))


def _run_frontend(
    frontend_id: int,
    settings_dict: dict,
    reply_endpoint: str,
    control: Connection,
    custom_server_settings: dict | None = None,
) -> None:
    """Entry point for each frontend process."""
    from .server import MLServer
    from .settings import Settings as _Settings
    from .parallel.shared import SharedPoolRegistry

    # Die with the supervisor even if it is SIGKILLed.
    set_parent_death_signal()

    settings = _Settings(**settings_dict)
    if custom_server_settings is not None:
        # The _custom_* attributes are plain class attributes, excluded
        # from model_dump(); the supervisor passes them alongside so a
        # programmatic embedder's custom server configuration survives
        # the spawn instead of silently resetting to None.
        settings._custom_rest_server_settings = custom_server_settings.get("rest")
        settings._custom_metrics_server_settings = custom_server_settings.get("metrics")
        settings._custom_grpc_server_settings = custom_server_settings.get("grpc")
    # Lets operators verify SO_REUSEPORT distribution from the client
    # side; absent in single-frontend mode.
    os.environ["MLSERVER_FRONTEND_ID"] = str(frontend_id)

    async def _main() -> None:
        shared_pool = SharedPoolRegistry(
            settings,
            frontend_id=frontend_id,
            reply_endpoint=reply_endpoint,
            control=control,
        )
        server = MLServer(settings, inference_pool_registry=shared_pool)
        shared_pool.attach_registry(server._model_registry)
        shared_pool.attach_data_plane(server._data_plane)
        # Frontends start empty: the supervisor loads models into the
        # shared pool and broadcasts applies, which land here.
        # Readiness gates on the empty_registry_readiness=False setting
        # (set when startup models exist) until the first model applies.
        # server.start() blocks on the gather of the server tasks —
        # it returns only when they complete (SIGTERM via uvicorn's
        # handler, or unexpected task exit). No Event().wait() after
        # it: nothing sets that event, and the supervisor would burn
        # its grace period killing an otherwise-healthy frontend.
        try:
            await server.start([])
        finally:
            # Model unloads become local no-ops: the supervisor drains the
            # shared pool itself after the frontends exit.
            shared_pool.begin_shutdown()
            # A SIGTERM-initiated stop can complete while the server
            # tasks drain: _live is False by then, and a second stop()
            # would re-run the full (idempotent but noisy) shutdown
            # path. An unexpected task exit keeps _live set, so the
            # cleanup stop still runs.
            if server._live:
                await server.stop()

    # Convert SIGTERM into KeyboardInterrupt so asyncio.run's cleanup
    # (finally blocks, server.stop()) executes on supervisor terminate().
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


class _PendingOp:
    """Tracks broadcast acknowledgement for one model operation."""

    def __init__(self, origin: int, ok: bool, error: str | None) -> None:
        self.origin = origin
        self.ok = ok
        self.error = error
        self.acks: set[int] = set()


class FrontendSupervisor:
    """Spawns N frontends and the shared inference pool behind them."""

    _live_frontends_gauge = None
    _frontend_restarts_counter = None

    def __init__(
        self,
        settings: Settings,
        models_settings: list[ModelSettings] | None = None,
    ) -> None:
        self._settings = settings
        self._models_settings = models_settings or []
        self._frontends: dict[int, multiprocessing.process.BaseProcess] = {}
        self._controls: dict[int, Connection] = {}
        self._restart_counts: dict[int, int] = {}
        self._closing = False
        self._pool_registry: InferencePoolRegistry | None = None
        # Model instances the supervisor loaded into the shared pool,
        # retained for exact-instance unload and frontend re-apply.
        self._loaded_models: dict[tuple[str, str], MLModel] = {}
        # Per-model-key commit generations: a frontend retiring a
        # superseded wrapper (same-version reload cleanup) carries an
        # older generation and must not unload the current one.
        self._generations: dict[tuple[str, str], int] = {}
        # Per-key transition locks: a model key's prepare → retire →
        # publish sequence must not interleave with another operation
        # on the same key, or both can read the same displaced
        # instance and the later inventory write strands the earlier
        # replacement without an unload. Like _generations, entries
        # live for the process lifetime (bounded by distinct keys).
        self._model_key_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._pending_ops: dict[str, _PendingOp] = {}
        self._ack_condition = asyncio.Condition()
        self._transport_directory: TemporaryDirectory | None = None
        self._listener_tasks: list[asyncio.Task] = []
        # Model operations spawned from control messages: fire-and-
        # forget by design (their results reach the origin through the
        # ack path), but stop() must cancel and settle them — an op
        # blocked on a pool call would otherwise outlive the control
        # pipes and write results to closed connections.
        self._op_tasks: set[asyncio.Task] = set()
        self._restart_tasks: set[asyncio.Task] = set()
        # Pending-ack notifications scheduled from sync death handling:
        # untracked tasks are how shutdown leaks start (a notify lost to
        # a cancelled task leaves ack collectors waiting their full
        # timeout), so every scheduled notify is settled by stop().
        self._ack_notify_tasks: set[asyncio.Task] = set()
        # The restart attempt that currently owns each slot's retry:
        # the monitor and a failing attempt's replay handler can both
        # observe the same death and schedule a retry — without a
        # single owner, both tasks read the same restart count and
        # spawn a process, and the later registration overwrites (and
        # orphans) the earlier one.
        self._restart_tasks_by_frontend: dict[int, asyncio.Task] = {}
        # Exit confirmations for partially started frontends rolled back
        # by a restart retry: the popped child is invisible to stop()'s
        # sweep over _frontends, so its confirmation must be tracked
        # and drained even when the restart task itself is cancelled
        # mid-retirement (shutdown does exactly that).
        self._retire_tasks: set[asyncio.Task] = set()
        # Live control traffic for a frontend mid-restart-replay is
        # queued and flushed before its readiness gate opens: a live
        # unload committing between the replay's snapshot and its sends
        # would otherwise be overtaken by the replayed load (the
        # replacement would serve a model the pool already removed),
        # and a live reload could pair the snapshot's old instance
        # with the new generation read at send time.
        self._replaying: set[int] = set()
        self._replay_pending: dict[int, list[object]] = {}
        # Each listener blocks a thread indefinitely on recv, so the
        # pool must cover every frontend (restarts reuse freed threads).
        self._listener_executor = ThreadPoolExecutor(
            max_workers=max(8, self._settings.frontend_workers),
            thread_name_prefix="mlserver-supervisor-ctl",
        )
        # Sends must be serialized per connection: Connection.send is
        # single-writer (POSIX writes header and payload separately for
        # larger frames, and a partial write's retry loop can interleave
        # with another thread), so concurrent model operations sharing
        # one control pipe would corrupt its framing. One writer thread
        # per connection keeps every send off the event loop AND
        # single-writer, mirroring the frontend-side send executor in
        # SharedPoolRegistry — and isolates connections from each
        # other: a frontend that stops reading blocks only its own
        # writer, never the sends to every other frontend.
        self._send_executors: dict[int, ThreadPoolExecutor] = {}

    def _register_supervisor_metrics(self) -> None:
        # Liveness and restart telemetry for the supervisor itself; the
        # gauge lets operators alert on degradation below frontend_workers
        # (including the all-restarts-exhausted zero-frontend state).
        from prometheus_client import Counter, Gauge

        if FrontendSupervisor._live_frontends_gauge is None:
            FrontendSupervisor._live_frontends_gauge = Gauge(
                "mlserver_live_frontends",
                "Number of live frontend processes",
            )
        if FrontendSupervisor._frontend_restarts_counter is None:
            FrontendSupervisor._frontend_restarts_counter = Counter(
                "mlserver_frontend_restarts_total",
                "Total frontend process restarts",
                ["frontend_id"],
            )

    def _observe_live_frontends(self) -> None:
        if FrontendSupervisor._live_frontends_gauge is not None:
            FrontendSupervisor._live_frontends_gauge.set(len(self._frontends))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _make_frontend_process(
        self, frontend_id: int
    ) -> "multiprocessing.process.BaseProcess":
        # Children keep frontend_workers in their settings: the REST, gRPC
        # and metrics servers key their SO_REUSEPORT configuration on it.
        # Only the CLI spawns a supervisor, so there is no recursion.
        settings_dict = self._settings.model_dump()
        # An explicit spawn context, independent of the global start
        # method: the default Linux method (fork) is only switched to
        # spawn later, when the pool activates. A forked frontend would
        # inherit the supervisor's ends of every control connection —
        # closing the supervisor's copy then never delivers EOF to the
        # frontend's reader, and shutdown hangs until the grace kill.
        # Spawn children get only the descriptors they are passed.
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        # Plain class attributes are excluded from model_dump(): pass
        # the custom server settings alongside so programmatic
        # embedders keep them in the spawned frontends.
        custom_server_settings = {
            "rest": self._settings._custom_rest_server_settings,
            "metrics": self._settings._custom_metrics_server_settings,
            "grpc": self._settings._custom_grpc_server_settings,
        }
        proc = ctx.Process(
            target=_run_frontend,
            args=(
                frontend_id,
                settings_dict,
                self._reply_endpoints[frontend_id],
                child_conn,
                custom_server_settings,
            ),  # noqa: E501
            name=f"mlserver-frontend-{frontend_id}",
        )
        # Retire the previous occupant's resources before overwriting
        # the slot: a failed earlier start() (or a restart racing this
        # factory) can leave an untracked connection whose fds would
        # otherwise leak, and a send writer still blocked on the old
        # connection would occupy the slot's single-thread executor —
        # every send to this replacement would then queue behind it,
        # time out, and terminate the replacement (a restart storm
        # until the budget is exhausted). Abandoning the executor
        # (wait=False, exactly as stop() does) strands at most the old
        # writer thread, which fails once the old connection's peer
        # closes.
        superseded = self._controls.pop(frontend_id, None)
        if superseded is not None:
            try:
                superseded.close()
            except Exception:
                pass
        old_executor = self._send_executors.pop(frontend_id, None)
        if old_executor is not None:
            old_executor.shutdown(wait=False)
        self._controls[frontend_id] = parent_conn
        return proc

    async def start(self) -> None:
        n = self._settings.frontend_workers
        logger.info(f"Starting {n} frontend processes...")

        # Everything that creates resources lives inside the try: a
        # Prometheus registry conflict or a failed temporary-directory
        # creation must not leak the transport directory or spawned
        # frontends — the stop() guard cleans both up.
        try:
            self._prepare_shared_resources()

            # The supervisor's pool, registry and dispatcher create
            # their own metrics; configure multiprocess mode first so
            # those land in the shared directory instead of an
            # unscraped in-process registry. This must precede the
            # spawn loop below: every spawn observes the
            # live-frontends gauge, and registering it after the
            # first observe would leave the initial count unpublished
            # (alerts on a healthy deployment would see 0).
            if self._settings.metrics_endpoint:
                from .metrics import configure_metrics

                configure_metrics(self._settings)
                self._register_supervisor_metrics()

            for i in range(n):
                proc = self._make_frontend_process(i)
                proc.start()
                self._frontends[i] = proc
                self._restart_counts[i] = 0
                self._observe_live_frontends()
                logger.info(f"Frontend {i} started (PID {proc.pid})")

            for frontend_id, control in list(self._controls.items()):
                self._listener_tasks.append(
                    asyncio.create_task(self._listen_frontend(frontend_id, control))
                )

            await self._start_shared_pool()
            await self._monitor()
        except BaseException:
            # Spawn or pool startup failure must not leave orphaned
            # frontends holding the shared ports.
            await self.stop()
            raise

    def _prepare_shared_resources(self) -> None:
        # Reply endpoints, created before any spawn.
        self._transport_directory = TemporaryDirectory(prefix="mlserver-ipc-")
        directory = Path(self._transport_directory.name)
        self._reply_endpoints = [
            create_endpoint(directory, "r")
            for _ in range(self._settings.frontend_workers)
        ]

    async def _start_shared_pool(self) -> None:
        # Warn when rotation could fragment batch windows: a rotation
        # mid-window splits the batcher's traffic across two workers.
        rotation = self._settings.parallel_routing_rotation
        if rotation > 0:
            for model_settings in self._models_settings:
                batch_time = model_settings.max_batch_time or 0
                if batch_time > 0 and rotation <= batch_time:
                    logger.warning(
                        "parallel_routing_rotation (%.1fs) is not larger "
                        "than max_batch_time (%.1fs) for model '%s': "
                        "rotation may fragment batch windows across "
                        "workers. Keep rotation at least a few times "
                        "larger than the batch window.",
                        rotation,
                        batch_time,
                        model_settings.name,
                    )
        self._pool_registry = InferencePoolRegistry(
            self._settings,
            on_worker_stop=[
                self._broadcast_worker_stopped,
                self._stop_worker_metrics,
            ],
            on_worker_ready=[self._broadcast_worker_ready],
            on_pool_quarantined=[self._broadcast_pool_quarantined],
            reply_endpoints=self._reply_endpoints,
        )
        await self._pool_registry.start()
        logger.info("Shared inference pool started.")

        # Load the startup models into the shared pool; frontends apply
        # the broadcasts (no origin, so no operation result is awaited).
        # A failed startup model fails the supervisor start — matching
        # the single-process server, whose startup gather raises and
        # tears the process down. Frontends must not report ready with
        # the default empty_registry_readiness while a startup model
        # failed to load.
        # Load concurrently, matching the single-process server's
        # startup gather: readiness latency must not scale with the
        # model count. Per-key locks serialise any same-key pair; the
        # per-key apply chains on the frontends keep the broadcast
        # applications ordered.
        startup_failures: list[str] = []
        results = await asyncio.gather(
            *(
                self._load_settings(model_settings, safe_error=True)
                for model_settings in self._models_settings
            ),
            return_exceptions=True,
        )
        for model_settings, result in zip(self._models_settings, results):
            if isinstance(result, BaseException):
                # safe_error discipline: class only (CWE-532).
                startup_failures.append(
                    f"{model_settings.name}: {type(result).__name__}"
                )
                continue
            ok, error, _ = result
            if not ok:
                # safe_error: this text lands in the RuntimeError that
                # escapes `mlserver start` to container stderr, where
                # exception messages can echo model-store credentials
                # (CWE-532). The class (and the full traceback at debug
                # level) stay available for diagnosis.
                startup_failures.append(
                    f"{model_settings.name}: {error or 'unknown error'}"
                )
        if startup_failures:
            raise RuntimeError(
                "Startup model loads failed — "
                f"{'; '.join(startup_failures)}. Enable debug logging "
                "(MLSERVER_DEBUG=1) for the full failure detail, which "
                "is kept out of this message because exception text can "
                "contain model-store credentials."
            )

        # Signal all frontends that the startup inventory is complete;
        # they release their readiness gate after receiving this.
        await self._broadcast(StartupComplete(model_count=len(self._models_settings)))

    async def _monitor(self) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
            loop.add_signal_handler(sig, stop_event.set)

        try:
            while not stop_event.is_set() and not self._closing:
                for frontend_id, proc in list(self._frontends.items()):
                    if _process_exited(proc):
                        self._handle_dead_frontend(frontend_id, proc)
                await asyncio.sleep(1.0)
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    async def _listen_frontend(self, frontend_id: int, control: Connection) -> None:
        loop = asyncio.get_running_loop()
        while not self._closing:
            try:
                message = await loop.run_in_executor(
                    self._listener_executor, control.recv
                )
            except (EOFError, OSError):
                # Frontend closed the pipe (exit or crash).
                return
            except asyncio.CancelledError:
                raise
            try:
                await self._handle_control(frontend_id, message, control)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed handling control message from frontend %s", frontend_id
                )

    async def _handle_control(
        self,
        frontend_id: int,
        message: object,
        control: "Connection | None" = None,
    ) -> None:
        # ``control`` is the connection the message arrived on: the
        # destructive handlers identity-check it against the slot so a
        # message from a frontend that died (and whose slot was since
        # restarted) cannot terminate its replacement.
        if isinstance(message, OpRequest):
            # Run the operation off the listener task: its acknowledgements
            # arrive back on this same pipe, so the listener must stay free
            # to receive them while the operation completes.
            op_task = asyncio.create_task(self._handle_op_logged(frontend_id, message))
            self._op_tasks.add(op_task)
            op_task.add_done_callback(self._op_tasks.discard)
        elif isinstance(message, ApplyAck):
            await self._handle_ack(frontend_id, message)
        elif isinstance(message, CancelRequest):
            self._handle_cancel(message)
        elif isinstance(message, StartupApplyFailed):
            self._handle_startup_apply_failed(frontend_id, message, control)
        else:
            # Type only: an unhandled message's repr could carry
            # settings payloads (see _handle_op_logged).
            logger.warning(
                "Unknown control message from frontend: %s", type(message).__name__
            )

    async def _handle_op_logged(self, frontend_id: int, message: OpRequest) -> None:
        try:
            await self._handle_op(frontend_id, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Payload-free logging: a deserialise failure's exception
            # text echoes the serialised settings (whose extra params
            # can hold credentials — CWE-532), so only the exception
            # class is logged here; the traceback goes to debug.
            logger.error(
                "Shared-pool operation from frontend %s (op %s) failed: %s",
                frontend_id,
                message.op_id,
                type(exc).__name__,
            )
            logger.debug("Shared-pool operation failure detail", exc_info=True)

    async def _handle_op(self, frontend_id: int, message: OpRequest) -> None:
        assert self._pool_registry is not None
        if message.kind == "load":
            assert message.serialised_model_settings is not None
            settings = deserialise_model_settings(message.serialised_model_settings)
            await self._load_settings(settings, origin=frontend_id, op_id=message.op_id)
        elif message.kind == "unload":
            await self._unload_settings(
                message.name or "",
                message.version,
                origin=frontend_id,
                op_id=message.op_id,
                generation=message.generation,
            )

    async def _load_settings(
        self,
        settings: ModelSettings,
        origin: int | None = None,
        op_id: str | None = None,
        safe_error: bool = False,
    ) -> tuple[bool, str | None, int]:
        """Load (or reload) a model into the shared pool."""
        # ``safe_error`` limits the returned error to the exception
        # class: the full text reaches the origin's API response
        # (matching the single-process server), but startup paths must
        # not embed it in exceptions that escape to container stderr,
        # where exception text can echo model-store credentials
        # (CWE-532).
        assert self._pool_registry is not None
        own_op = op_id if op_id is not None else generate_uuid()
        key = (settings.name, settings.version or "")
        # Serialise same-key transitions (preparation, retirement and
        # publication): two concurrent operations on one key would both
        # read the same displaced instance, and the later inventory
        # write would strand the earlier replacement in its pool
        # without an unload. Different keys stay concurrent.
        # The finally spans registration-to-completion: the ack
        # collector is registered before the broadcast (inside the
        # locked body), so a cancellation anywhere from there through
        # the ack collection must not leak the entry.
        try:
            async with self._key_lock(key):
                ok, error, generation, delivered = await self._load_settings_locked(
                    settings, origin, own_op, safe_error
                )
            # Frontend acknowledgements are awaited OUTSIDE the key lock: a
            # slow or dead origin must not delay other operations on the
            # same key.
            if origin is not None:
                await self._collect_acks(
                    own_op,
                    origin,
                    ok,
                    error,
                    generation=generation,
                    expected=delivered,
                )
        finally:
            if origin is not None:
                self._pending_ops.pop(own_op, None)
        return ok, error, generation

    def _key_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._model_key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._model_key_locks[key] = lock
        return lock

    async def _load_settings_locked(
        self,
        settings: ModelSettings,
        origin: int | None,
        own_op: str,
        safe_error: bool,
    ) -> tuple[bool, str | None, int, set[int]]:
        assert self._pool_registry is not None
        ok, error = True, None
        env_hash: str | None = None
        # Mirror the server-level guard (frontend_workers > 1 requires
        # parallel_workers >= 1): a model that opts out of parallel
        # inference needs in-process execution, which in the
        # multi-frontend topology would load one copy per frontend —
        # and its requests could never reach workers that did not load
        # it. Reject loudly at load time instead of failing at
        # inference time.
        if self._settings.frontend_workers > 1 and (
            not self._pool_registry.should_load_model(settings)
        ):
            ok = False
            error = (
                f"Model '{settings.name}' disables parallel inference "
                f"(model-level parallel_workers="
                f"{settings.parallel_workers}), which requires "
                f"in-process execution. With frontend_workers="
                f"{self._settings.frontend_workers}, every frontend "
                "process would need its own copy of the model. Use the "
                "server-level parallel_workers setting (the model-level "
                "field is deprecated), or deploy this model with "
                "frontend_workers=1."
            )
            logger.error("Shared-pool load of %s rejected: %s", settings.name, error)
        else:
            try:
                key = (settings.name, settings.version or "")
                instance = self._pool_registry.model_initialiser(settings)
                await self._pool_registry.load_model(instance)
                # The registry resolved the new instance's pool identity
                # (env hash or GID) — compare it with the displaced
                # instance's resolved identity, not the raw settings
                # GID, and only after the replacement is ready: the
                # old model keeps serving until the new one is
                # validated, and a failed reload leaves it in place.
                new_hash = getattr(instance, ENV_HASH_ATTR, None)
                displaced = self._loaded_models.get(key)
                if displaced is not None:
                    displaced_hash = getattr(displaced, ENV_HASH_ATTR, None)
                    # Compare resolved pool identities: both are
                    # ENV_HASH_ATTR values set by the registry.
                    if displaced_hash != new_hash:
                        # Cross-pool move: unload the old instance from
                        # its original pool only after the replacement
                        # committed elsewhere. A failure here is
                        # tolerated — the new pool is live; the old
                        # instance leaks until its next operation.
                        try:
                            await self._pool_registry.unload_model(displaced)
                        except (Exception, asyncio.CancelledError) as exc:
                            # The exception text can echo worker-side
                            # payload content (e.g. credentials from a
                            # storage-backed runtime's unload path), and
                            # this log fires at ERROR where operators
                            # read supervisor output: record the class,
                            # never the message or traceback (CWE-532).
                            logger.error(
                                "Failed unloading displaced pool instance "
                                "for %s (pool %s): %s",
                                settings.name,
                                displaced_hash,
                                type(exc).__name__,
                            )
                    else:
                        # Same-pool replacement: the pool's load only
                        # STAGED the new version — the workers keep the
                        # old one registered and serving until the
                        # retiring unload arrives. Commit it before
                        # reporting success: a reload that returns ok
                        # while the old code still serves is exactly the
                        # divergence this prevents. Unlike the cross-pool
                        # case, a failure MUST fail the reload (the old
                        # instance stays tracked for the retry; the
                        # pool's recovery owns the failed dispatch).
                        await self._pool_registry.unload_model(displaced)
                self._loaded_models[key] = instance
                # The registry resolves pool identity onto the loaded model;
                # frontends tag the model's routing partition with it.
                env_hash = getattr(instance, ENV_HASH_ATTR, None)
            except (Exception, asyncio.CancelledError) as exc:
                ok = False
                # The error text reaches the origin's API response
                # (matching the single-process server's semantics), but
                # the supervisor log must stay payload-free: exception
                # text can echo settings values, including credentials
                # from parameters.extra (CWE-532). Class + op id at
                # error level; the full traceback at debug only.
                error = (
                    type(exc).__name__ if safe_error else f"{type(exc).__name__}: {exc}"
                )
                logger.error(
                    "Shared-pool load of %s failed: %s (op %s)",
                    settings.name,
                    type(exc).__name__,
                    own_op,
                )
                logger.debug("Shared-pool load failure detail", exc_info=True)
        # A committed (re)load advances the model key's generation;
        # frontends attach it to their wrappers so a later retirement
        # of a superseded wrapper is not mistaken for a real unload.
        generation = 0
        if ok:
            key = (settings.name, settings.version or "")
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
        # Register the ack collector before broadcasting: the origin
        # frontend acknowledges on receipt, which can beat _collect_acks.
        if origin is not None:
            self._pending_ops[own_op] = _PendingOp(origin, ok, error)
        delivered = await self._broadcast(
            ApplyLoad(
                op_id=own_op,
                serialised_model_settings=serialise_model_settings(settings),
                ok=ok,
                error=error,
                env_hash=env_hash,
                generation=generation,
            )
        )
        return ok, error, generation, delivered

    async def _unload_settings(
        self,
        name: str,
        version: str | None,
        origin: int | None = None,
        op_id: str | None = None,
        generation: int | None = None,
    ) -> None:
        assert self._pool_registry is not None
        own_op = op_id if op_id is not None else generate_uuid()
        retired_key = (name, version or "")
        # Same-key serialisation as loads: an unload interleaving with a
        # load of the same key must not observe a half-committed
        # transition. Acknowledgements are awaited outside the lock.
        # Same registration-to-completion span as loads (see there).
        try:
            async with self._key_lock(retired_key):
                ok, error, delivered = await self._unload_settings_locked(
                    name, version, origin, own_op, generation
                )
            if origin is not None:
                await self._collect_acks(own_op, origin, ok, error, expected=delivered)
        finally:
            if origin is not None:
                self._pending_ops.pop(own_op, None)

    async def _unload_settings_locked(
        self,
        name: str,
        version: str | None,
        origin: int | None,
        own_op: str,
        generation: int | None,
    ) -> tuple[bool, str | None, set[int]]:
        assert self._pool_registry is not None
        # Every unload targets exactly one key — the caller's wrapper's
        # own key, including the unversioned key. The public name-wide
        # unload enumerates its wrappers at the frontend (one exact
        # operation per version), and a repository sync's stale cleanup
        # (which displaces exactly the old unversioned instance) must
        # not expand to the version that sync just loaded.
        retired_key = (name, version or "")
        # A retirement (reload cleanup of a displaced wrapper) carries a
        # generation that no longer matches the committed one. Generation
        # counters are monotonic per key — a successful unload keeps the
        # key's counter so the next load commits a value higher than any
        # earlier one (resetting to 1 would let a stale wrapper whose
        # peer-side unload failed collide with a fresh generation-1
        # commit and unload it).
        superseded = generation is not None and generation != self._generations.get(
            retired_key, 0
        )
        if superseded:
            # The current pool generation stays committed. Resolve the
            # origin's operation without unloading or broadcasting —
            # peers already hold the newer generation.
            logger.debug(
                "Skipping unload of superseded generation %d for %s " "(current %d)",
                generation,
                name,
                self._generations.get(retired_key, 0),
            )
            if origin is not None:
                await self._send_to(origin, OpResult(op_id=own_op, ok=True))
            return True, None, set()
        failures: list[str] = []
        # The outcome is captured when the pool call settles: the
        # inventory observed later can be changed by a concurrent load
        # interleaving at the await below.
        key_ok = True
        instance = self._loaded_models.get(retired_key)
        if instance is None:
            # Already absent (or never committed here): the desired
            # post-unload state.
            pass
        else:
            try:
                await self._pool_registry.unload_model(instance)
                # Identity-guarded removal: a concurrent load may
                # have committed a replacement while the pool call
                # was in flight — only this instance goes.
                if self._loaded_models.get(retired_key) is instance:
                    self._loaded_models.pop(retired_key, None)
            except (Exception, asyncio.CancelledError) as exc:
                key_ok = False
                # The error text reaches the origin's API response
                # (matching the single-process server's semantics), but
                # the supervisor log must stay payload-free (CWE-532).
                failures.append(f"{retired_key[1] or 'unversioned'}: {exc}")
                logger.error(
                    "Shared-pool unload of %s failed: %s (op %s)",
                    name,
                    type(exc).__name__,
                    own_op,
                )
        # Aggregate failures only in the origin's result: each key's
        # broadcast carries its own outcome, so a peer removes every
        # version whose workers actually unloaded it and keeps the ones
        # that failed (matching the pool's per-key state).
        ok = not failures
        error = "; ".join(failures) if failures else None
        # Same pre-broadcast registration as loads (see _load_settings).
        if origin is not None:
            self._pending_ops[own_op] = _PendingOp(origin, ok, error)
        # Broadcast the resolved key with its concrete version so every
        # peer exact-unloads the version it holds (a peer that missed
        # the version's apply no-ops on it). The outcome was captured
        # when the pool call settled above — never re-derived from
        # inventory a concurrent load may have changed.
        delivered = await self._broadcast(
            ApplyUnload(
                op_id=own_op,
                name=name,
                version=retired_key[1],
                ok=key_ok,
                error=None if key_ok else error,
                # The generation this unload actually removed: a
                # peer holding a newer generation must skip this
                # stale unload (the newer load committed during the
                # pool await).
                generation=generation or 0,
            )
        )
        return ok, error, delivered

    async def _collect_acks(
        self,
        op_id: str,
        origin: int,
        ok: bool,
        error: str | None,
        generation: int = 0,
        expected: set[int] | None = None,
    ) -> None:
        """Wait until every live frontend applied, then answer the origin."""
        # The pending op is registered before the broadcast; reuse it
        # so early acks are not lost.
        pending = self._pending_ops.get(op_id)
        if pending is None:
            # Defensive fallback (both callers register before the
            # broadcasts): without the registration the early acks are
            # already lost, so waiting for one ack per frontend matches
            # the pre-existing behaviour of this path.
            pending = _PendingOp(origin, ok, error)
            self._pending_ops[op_id] = pending
        try:
            # The captured recipient set from the broadcast: a
            # replacement spawning between the broadcast and a later
            # _frontends snapshot never saw the message and would
            # delay the result by the full timeout.
            expected_acks = (
                expected if expected is not None else set(self._frontends.keys())
            )
            deadline = asyncio.get_running_loop().time() + (
                self._settings.model_operation_timeout
            )
            async with self._ack_condition:
                while not expected_acks.issubset(pending.acks):
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        logger.warning(
                            "Operation %s ack collection timed out "
                            "(missing frontends: %s)",
                            op_id,
                            sorted(expected_acks - pending.acks),
                        )
                        break
                    try:
                        await asyncio.wait_for(
                            self._ack_condition.wait(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break
        finally:
            self._pending_ops.pop(op_id, None)
        # On timeout the origin gets the pool's result (ok/error), not a
        # convergence guarantee: some frontends may not have applied yet.
        # The broadcast itself is not rolled back (matching the pool's
        # deferred-cancellation semantics); late applies converge, and the
        # next operation on the same model observes the settled state.
        # Note: a peer's local apply failure after the pool succeeded is
        # logged but does not change the result sent to the origin.
        await self._send_to(
            origin, OpResult(op_id=op_id, ok=ok, error=error, generation=generation)
        )

    async def _handle_ack(self, frontend_id: int, message: ApplyAck) -> None:
        pending = self._pending_ops.get(message.op_id)
        if pending is not None:
            async with self._ack_condition:
                pending.acks.add(frontend_id)
                self._ack_condition.notify_all()

    def _handle_startup_apply_failed(
        self,
        frontend_id: int,
        message: StartupApplyFailed,
        control: "Connection | None" = None,
    ) -> None:
        """Terminate a frontend whose startup apply failed.

        The frontend keeps its readiness gate closed, but it would
        otherwise stay alive forever — still receiving SO_REUSEPORT
        traffic it cannot serve, counted healthy in the live-frontends
        gauge, and never restarted. Terminating it hands the slot to
        the monitor's restart machinery (with the restart budget).
        """
        logger.error(
            "Frontend %s failed applying startup model %s; terminating "
            "it for restart",
            frontend_id,
            message.model_name,
        )
        proc = self._frontends.get(frontend_id)
        if proc is not None and not _process_exited(proc):
            # ``control`` (the connection the report arrived on)
            # identity-guards the slot: if this frontend died right
            # after reporting and the monitor already restarted the
            # slot, the lookup above returns the REPLACEMENT — which
            # must not be terminated for its predecessor's failure.
            self._terminate_frontend_for_restart(frontend_id, proc, control)

    def _handle_cancel(self, message: CancelRequest) -> None:
        """Forward a request cancellation to the worker that holds it."""
        if self._pool_registry is None:
            return
        worker = self._pool_registry.find_worker(message.worker_pid)
        if worker is None:
            return
        try:
            worker.send_update(
                ModelUpdateMessage(
                    update_type=ModelUpdateType.Cancel,
                    model_settings=ModelSettings(
                        name=message.model_name,
                        implementation="mlserver.model.MLModel",
                        # The eviction targets this version's batcher,
                        # not the name's default version.
                        parameters=(
                            {"version": message.model_version}
                            if message.model_version is not None
                            else None
                        ),
                    ),
                    request_id=message.request_id,
                )
            )
        except Exception:
            # Accept-and-discard remains correct if the cancel is lost,
            # but the loss should be visible to operators.
            logger.warning("Failed forwarding cancel for %s", message.request_id)

    # ------------------------------------------------------------------
    # Broadcasts
    # ------------------------------------------------------------------

    async def _broadcast(self, message: object) -> set[int]:
        # Route through the per-connection send writers, NOT
        # _listener_executor: the listener pool's threads are occupied
        # by control.recv calls, and at frontend_workers >= 8 all of
        # them are blocked on recv — sends queued behind them would
        # livelock (frontends can't ack what they can't receive).
        # Each connection's writer is single-threaded:
        # Connection.send is single-writer (separate header/payload
        # writes on POSIX), so two concurrent model operations must
        # never send on one pipe at the same time. ApplyLoad carries
        # the full serialized ModelSettings (KB-sized), so synchronous
        # sends would block the loop on a full pipe buffer.
        # A frontend mid-restart-replay gets the message queued instead:
        # its replay snapshot must land before any live update.
        sends = []
        delivered: set[int] = set()
        for frontend_id, control in list(self._controls.items()):
            if frontend_id in self._replaying:
                # Queue only for the connection this broadcast
                # captured: a restart can replace the slot while an
                # earlier send yields to the executor, and the
                # replacement's replay snapshot is NEWER than this
                # message — queueing it would apply a stale generation
                # after the snapshot (e.g. restoring a superseded
                # wrapper or pool tag).
                if self._controls.get(frontend_id) is control:
                    self._replay_pending.setdefault(frontend_id, []).append(message)
                    delivered.add(frontend_id)
                continue
            sends.append(self._send_on_connection(frontend_id, control, message))
            delivered.add(frontend_id)
        # Independent enqueueing: every connection's writer takes the
        # message at once — one stalled frontend cannot starve the
        # rest (its own send is bounded by the delivery deadline
        # below, which terminates it for the monitor to restart).
        if sends:
            await asyncio.gather(*sends)
        # The captured recipient set: an ack collection must wait for
        # exactly these frontends, not a later snapshot of _frontends
        # (a replacement spawning in between never saw the message and
        # would delay the operation result by the full timeout).
        return delivered

    async def _send_to(self, frontend_id: int, message: object) -> None:
        if frontend_id in self._replaying:
            # Queue live traffic behind the replay: a live unload or
            # reload committing mid-replay must apply AFTER the
            # snapshot, not interleave with it.
            self._replay_pending.setdefault(frontend_id, []).append(message)
            return
        await self._send_direct(frontend_id, message)

    async def _send_direct(
        self, frontend_id: int, message: object, strict: bool = False
    ) -> None:
        """Send bypassing the replay queue (used by the replay itself).

        ``strict`` propagates write failures: the restart handler must
        roll back a replacement whose replay could not be delivered.
        A live-but-unreadable child would otherwise keep its slot with
        a partial inventory and a readiness gate that nothing opens —
        the monitor only detects process death, so nothing would
        restart it.
        """
        control = self._controls.get(frontend_id)
        if control is None:
            if strict:
                raise RuntimeError(
                    f"Control connection for frontend {frontend_id} is gone"
                )
            return
        await self._send_on_connection(frontend_id, control, message, strict=strict)

    def _send_executor_for(self, frontend_id: int) -> ThreadPoolExecutor:
        # Slot-scoped and reused across restarts of the same slot: the
        # id set is bounded by frontend_workers, so lazy creation
        # cannot leak.
        executor = self._send_executors.get(frontend_id)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"mlserver-supervisor-send-{frontend_id}",
            )
            self._send_executors[frontend_id] = executor
        return executor

    async def _send_on_connection(
        self,
        frontend_id: int,
        control: Connection,
        message: object,
        strict: bool = False,
    ) -> None:
        """Send on the connection's dedicated writer, deadline-bounded."""
        # A live frontend that stops reading control traffic blocks only
        # its own writer thread. Past the delivery deadline it is
        # terminated — a frontend that cannot receive can never
        # acknowledge an operation again — and the monitor's restart
        # (with its replay) replaces it. The blocked write itself fails
        # once the dead frontend's pipe closes.
        loop = asyncio.get_running_loop()
        send = loop.run_in_executor(
            self._send_executor_for(frontend_id), control.send, message
        )
        # The deadline below may abandon this send; retrieve (and
        # drop) its eventual failure so the executor does not warn.
        send.add_done_callback(consume_send_failure)
        try:
            await asyncio.wait_for(
                asyncio.shield(send), self._settings.model_operation_timeout
            )
        except TimeoutError:
            proc = self._frontends.get(frontend_id)
            if proc is not None and not _process_exited(proc):
                logger.error(
                    "Frontend %s control pipe stalled: no delivery within "
                    "%.1fs. Terminating it for restart.",
                    frontend_id,
                    self._settings.model_operation_timeout,
                )
                # Close the supervisor's end of the stalled connection
                # before terminating, then confirm the exit — a bare
                # terminate() leaves a SIGTERMed frontend hung on its
                # non-daemon control reader (no EOF is delivered while
                # this end stays open) and the slot is never restarted.
                self._terminate_frontend_for_restart(frontend_id, proc, control)
            if strict:
                raise
        except Exception:
            if strict:
                raise
            logger.debug("Send to dead frontend pipe", exc_info=True)

    async def _broadcast_worker_ready(self, worker) -> None:
        await self._broadcast(
            WorkerReadyEvent(
                pid=worker.pid,
                request_endpoint=worker._zmq_request_endpoint,
                # Workers carry their pool's env hash from spawn; the
                # frontends partition routing by it.
                env_hash=getattr(worker, "env_hash", None),
            )
        )

    async def _broadcast_worker_stopped(self, worker) -> None:
        exit_code = worker.exitcode if worker.exitcode is not None else -1
        await self._broadcast(
            WorkerStoppedEvent(
                pid=worker.pid,
                exit_code=exit_code,
                env_hash=getattr(worker, "env_hash", None),
            )
        )

    async def _broadcast_pool_quarantined(self, env_hash: str | None) -> None:
        """Tell every frontend to pause routing to a recovering pool."""
        await self._broadcast(PoolQuarantinedEvent(env_hash=env_hash))

    async def _stop_worker_metrics(self, worker) -> None:
        # Remove the dead worker's multiprocess metric files, matching
        # the single-process server's MetricsServer.on_worker_stop hook.
        if not worker.pid:
            return
        from .metrics.prometheus import stop_metrics

        await stop_metrics(self._settings, worker.pid)

    # ------------------------------------------------------------------
    # Frontend death and restart
    # ------------------------------------------------------------------

    async def _notify_pending_acks(self) -> None:
        async with self._ack_condition:
            self._ack_condition.notify_all()

    def _handle_dead_frontend(
        self, frontend_id: int, proc: multiprocessing.process.BaseProcess
    ) -> None:
        del self._frontends[frontend_id]
        self._observe_live_frontends()
        control = self._controls.pop(frontend_id, None)
        if control is not None:
            try:
                control.close()
            except Exception:
                pass
        # A dead frontend can no longer acknowledge pending operations.
        for pending in self._pending_ops.values():
            pending.acks.add(frontend_id)
        # Wake any in-flight ack collection: it may now be complete.
        # (Sync method cannot hold the condition; schedule the notify.)
        notify = asyncio.create_task(self._notify_pending_acks())
        self._ack_notify_tasks.add(notify)
        notify.add_done_callback(self._ack_notify_tasks.discard)
        self._mark_metrics_process_dead(proc)
        restarts = self._restart_counts.get(frontend_id, 0)

        if restarts >= self._settings.frontend_max_restarts:
            logger.error(
                f"Frontend {frontend_id} died and exceeded "
                f"{self._settings.frontend_max_restarts} restarts "
                f"(exit code {proc.exitcode}); "
                "giving up. Remaining frontends continue serving."
            )
            return

        logger.warning(
            f"Frontend {frontend_id} (PID {proc.pid}) died with exit "
            f"code {proc.exitcode}. Restarting "
            f"(restart {restarts + 1}/{self._settings.frontend_max_restarts})..."
        )
        self._schedule_restart(frontend_id)

    def _schedule_restart(
        self, frontend_id: int, exclude: "asyncio.Task | None" = None
    ) -> None:
        """Schedule a restart attempt for a dead frontend slot.

        The slot is absent from ``_frontends`` (removed on death and only
        re-added on a successful spawn), so the monitor loop cannot
        retrigger it — retries must be scheduled here explicitly.

        One owner per slot: an already-scheduled or running attempt
        owns the retry. ``exclude`` lets a FAILING attempt hand over to
        its successor (the incumbent is ending, so it does not count
        as the owner).
        """
        restarts = self._restart_counts.get(frontend_id, 0)
        if restarts >= self._settings.frontend_max_restarts:
            return
        existing = self._restart_tasks_by_frontend.get(frontend_id)
        if existing is not None and existing is not exclude and not existing.done():
            logger.debug(
                "Frontend %s restart already owned by a scheduled attempt",
                frontend_id,
            )
            return
        # The backoff sleep happens INSIDE _restart_frontend (at its
        # start), not here — this function only schedules the task.
        restart_task = asyncio.create_task(self._restart_frontend(frontend_id))
        self._restart_tasks_by_frontend[frontend_id] = restart_task
        self._restart_tasks.add(restart_task)
        restart_task.add_done_callback(self._restart_tasks.discard)

        def _clear_owner(task: asyncio.Task) -> None:
            if self._restart_tasks_by_frontend.get(frontend_id) is task:
                del self._restart_tasks_by_frontend[frontend_id]

        restart_task.add_done_callback(_clear_owner)
        logger.debug("Frontend %s restart attempt scheduled", frontend_id)

    def _close_unstarted_control(self, frontend_id: int) -> None:
        """Close a connection registered by a factory whose start() failed.

        The factory registers the slot's connection before the caller
        starts the process; a start() failure leaves it unread. Closing
        it now (instead of waiting for the retry's factory or stop())
        makes sends to the slot fail fast instead of buffering into
        the dead pipe's 64KB window, and releases the fds for
        determinism rather than waiting for the never-started Process
        holding the child end to be collected.
        """
        orphaned = self._controls.pop(frontend_id, None)
        if orphaned is not None:
            try:
                orphaned.close()
            except Exception:
                pass

    async def _retire_partial_frontend(
        self,
        frontend_id: int,
        listener_task: asyncio.Task | None,
        proc: "multiprocessing.process.BaseProcess | None" = None,
        control: "Connection | None" = None,
    ) -> None:
        """Roll back a partially started replacement before a retry.

        The restart path registers the child, its listener and its
        control connection as soon as the process starts; a failure in
        the replay would otherwise leave them untracked when the retry
        overwrites the slot.

        Identity-guarded: a newer restart attempt may already have
        claimed the slot (its own process, connection and listener).
        This attempt rolls back only what it registered — never a
        newer owner's replacement. ``proc``/``control`` are this
        attempt's own registrations; leaving ``proc`` unset retires
        whatever currently occupies the slot (the caller asserts it
        is theirs).
        """
        if proc is None:
            proc = self._frontends.pop(frontend_id, None)
            still_ours = True
        else:
            still_ours = self._frontends.get(frontend_id) is proc
            if still_ours:
                self._frontends.pop(frontend_id, None)
        # Quick, non-awaiting cleanups first: a cancellation of the
        # restart task (shutdown cancels restarts mid-retirement) must
        # not skip them.
        if listener_task is not None:
            listener_task.cancel()
            if listener_task in self._listener_tasks:
                self._listener_tasks.remove(listener_task)
        if control is not None:
            if self._controls.get(frontend_id) is control:
                self._controls.pop(frontend_id, None)
        elif still_ours:
            control = self._controls.pop(frontend_id, None)
        if still_ours:
            self._observe_live_frontends()
        if control is not None:
            # Our own connection (popped from the slot, or superseded
            # in it by a newer owner): close the reference either way.
            try:
                control.close()
            except Exception:
                pass
        if not still_ours:
            logger.debug(
                "Frontend %s slot already claimed by a newer restart "
                "attempt; rolling back only this attempt's process",
                frontend_id,
            )
        if proc is None:
            return
        # The popped child is invisible to stop()'s sweep over
        # _frontends: confirm its exit in a tracked task that survives
        # cancellation of this restart task, and that stop() drains
        # before its own sweep.
        confirm = asyncio.create_task(self._confirm_frontend_exit(frontend_id, proc))
        self._retire_tasks.add(confirm)
        confirm.add_done_callback(self._retire_tasks.discard)
        await asyncio.shield(confirm)

    async def _confirm_frontend_exit(
        self, frontend_id: int, proc: multiprocessing.process.BaseProcess
    ) -> None:
        """Terminate a partially started frontend and confirm its exit."""
        # _process_exited (not is_alive): a corpse already reaped by
        # the pool's SIGCHLD handler blinds is_alive() forever.
        # to_thread: join blocks, and this runs on the supervisor's
        # event loop — a stuck child must not freeze the pool's
        # SIGCHLD handler and in-flight responses for up to 10s.
        if not _process_exited(proc):
            try:
                proc.terminate()
            except ProcessLookupError:
                # Reaped between the exit check and the signal: the
                # child is already gone; nothing to signal. (A reused
                # PID cannot be caught here without pidfds — accepted
                # micro-race.)
                pass
        await asyncio.to_thread(proc.join, timeout=5)
        if not _process_exited(proc):
            logger.warning(
                f"Partial frontend {frontend_id} (PID {proc.pid}) did not "
                "exit gracefully; killing..."
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await asyncio.to_thread(proc.join, timeout=5)
        if not _process_exited(proc):
            logger.error(
                f"Partial frontend {frontend_id} (PID {proc.pid}) did not "
                "exit after SIGKILL; shutdown incomplete"
            )

    def _terminate_frontend_for_restart(
        self,
        frontend_id: int,
        proc: "multiprocessing.process.BaseProcess",
        control: "Connection | None" = None,
    ) -> None:
        """Terminate a live-but-unresponsive frontend for monitor restart.

        Same discipline as stop(): close the supervisor's control end
        BEFORE terminating. The frontend's control reader blocks in
        recv() inside an executor thread, and on Linux closing the
        local end does not interrupt it — only the peer's EOF does.
        Terminating without the close leaves the SIGTERMed frontend
        hung at interpreter exit (the non-daemon recv thread never
        finishes), the monitor — which only restarts on death — never
        sees it, and the slot is lost with a leaked PID.
        """
        if control is not None and self._controls.get(frontend_id) is not control:
            # The slot was replaced while this send was stalled: the
            # replacement's registration already retired the stalled
            # owner's process and connection.
            return
        if control is None:
            control = self._controls.get(frontend_id)
        if control is not None and self._controls.get(frontend_id) is control:
            self._controls.pop(frontend_id, None)
            try:
                control.close()
            except Exception:
                pass
        if _process_exited(proc):
            return
        # Exit confirmation runs in a tracked task (drained by stop()
        # alongside other retirements) with join deadline and SIGKILL
        # escalation; the monitor picks up the death for restart.
        confirm = asyncio.create_task(self._confirm_frontend_exit(frontend_id, proc))
        self._retire_tasks.add(confirm)
        confirm.add_done_callback(self._retire_tasks.discard)

    async def _restart_frontend(self, frontend_id: int) -> None:
        restarts = self._restart_counts.get(frontend_id, 0)
        backoff = min(2**restarts, 16)
        registered = False
        own_control: Connection | None = None
        listener_task: asyncio.Task | None = None
        replaying = False
        try:
            await asyncio.sleep(backoff)
            if self._closing:
                return
            # Count every attempt (successful or not) against the budget.
            self._restart_counts[frontend_id] = restarts + 1
            new_proc = self._make_frontend_process(frontend_id)
            new_proc.start()
            self._frontends[frontend_id] = new_proc
            # This attempt's own connection (registered by the factory
            # above): the identity-guarded rollback closes exactly
            # this reference, never a newer attempt's.
            own_control = self._controls.get(frontend_id)
            # From here the child, its listener and its control
            # connection are registered: a failure in the replay below
            # must roll them back before the retry, or the retry would
            # overwrite the slot and leave an untracked partial
            # frontend serving an incomplete inventory.
            registered = True
            self._observe_live_frontends()
            if FrontendSupervisor._frontend_restarts_counter is not None:
                FrontendSupervisor._frontend_restarts_counter.labels(
                    frontend_id=str(frontend_id)
                ).inc()
            listener_task = asyncio.create_task(
                self._listen_frontend(frontend_id, self._controls[frontend_id])
            )
            self._listener_tasks.append(listener_task)
            logger.info(f"Frontend {frontend_id} restarted (new PID {new_proc.pid})")
            # From registration on, live control traffic for this
            # frontend queues behind the replay: a live unload
            # committing between the snapshot below and its sends must
            # apply AFTER the replayed load (or the replacement would
            # serve a model the pool already removed), and a live
            # reload must not interleave with the replayed state.
            self._replaying.add(frontend_id)
            replaying = True
            # Snapshot the committed inventory in ONE synchronous pass:
            # instance and generation captured together, so a reload
            # committing mid-replay cannot pair the snapshot's old
            # instance with the new generation read at send time.
            replayed_models = [
                (
                    name,
                    version,
                    instance,
                    self._generations.get((name, version), 0),
                )
                for (name, version), instance in list(self._loaded_models.items())
            ]
            # Replay the current worker incarnations first: the
            # replacement's dispatcher starts empty and needs the
            # WorkerReadyEvents before it can route anything. Only the
            # pools' READY workers are replayed (tagged with the pool's
            # env hash) so routing partitions survive a frontend
            # restart. Quarantined workers stay in _workers with live
            # endpoints, but their model state is mid-recovery — the
            # post-recovery worker-ready events publish them instead.
            if self._pool_registry is not None:
                pools = list(self._pool_registry._pools.values()) + (
                    [self._pool_registry._default_pool]
                    if self._pool_registry._default_pool
                    else []
                )
                for pool in pools:
                    if pool is None:
                        continue
                    for pid, worker in pool.get_ready_worker_endpoints().items():
                        endpoint = worker._zmq_request_endpoint
                        if endpoint is None:
                            # Not yet routable (no zmq endpoint bound);
                            # its worker-ready event will arrive later.
                            continue
                        await self._send_direct(
                            frontend_id,
                            WorkerReadyEvent(
                                pid=pid,
                                request_endpoint=endpoint,
                                env_hash=worker.env_hash,
                            ),
                            strict=True,
                        )

            # Re-apply every model already committed to the shared
            # pool so the fresh frontend serves the full registry.
            # Unicast: surviving frontends already have these models,
            # and a broadcast would reload them (brief unavailability).
            for name, version, instance, generation in replayed_models:
                await self._send_direct(
                    frontend_id,
                    ApplyLoad(
                        op_id=generate_uuid(),
                        serialised_model_settings=serialise_model_settings(
                            instance.settings
                        ),
                        ok=True,
                        env_hash=getattr(instance, ENV_HASH_ATTR, None),
                        # The committed generation: without it the
                        # replacement holds generation 0, and its later
                        # unloads would look superseded (0 < current)
                        # and be silently skipped.
                        generation=generation,
                    ),
                    strict=True,
                )

            # Flush the live updates that committed during the replay,
            # still queueing: each pass drains the current batch, and
            # messages arriving during a pass queue for the next one.
            # The final empty check and the discard below are one
            # synchronous stretch, so nothing can queue in between —
            # everything before the discard lands in order, everything
            # after sends directly.
            while self._replay_pending.get(frontend_id):
                for message in self._replay_pending.pop(frontend_id):
                    await self._send_direct(frontend_id, message, strict=True)
            self._replaying.discard(frontend_id)
            replaying = False

            # The replacement starts with its startup readiness gate
            # closed (like every fresh frontend). The replay above
            # covered the committed inventory, so release the gate now —
            # including when the inventory is empty. Strict: a gate
            # release that cannot be delivered must roll the
            # replacement back, not leave it unreadable.
            await self._send_direct(
                frontend_id,
                StartupComplete(model_count=len(self._loaded_models)),
                strict=True,
            )
        except asyncio.CancelledError:
            # Shutdown cancels restart tasks mid-flight: after
            # registration the partial replacement must roll back here
            # (stop()'s sweep would terminate it only later, leaving a
            # frontend whose replay never finished holding the slot).
            # The shield keeps the rollback alive if a second
            # cancellation arrives while it runs; the cancellation
            # itself is then re-raised.
            if not registered:
                self._close_unstarted_control(frontend_id)
            if registered:
                await asyncio.shield(
                    self._retire_partial_frontend(
                        frontend_id,
                        listener_task,
                        proc=new_proc,
                        control=own_control,
                    )
                )
            raise
        except Exception:
            logger.exception(f"Failed to restart frontend {frontend_id}")
            if not registered:
                self._close_unstarted_control(frontend_id)
            if registered:
                # Roll back the partial replacement so the retry starts
                # from an empty slot — identity-guarded, so a newer
                # restart attempt's replacement is never retired by
                # this (older) attempt's failure.
                await self._retire_partial_frontend(
                    frontend_id,
                    listener_task,
                    proc=new_proc,
                    control=own_control,
                )
            # The slot stays absent from _frontends, so the monitor
            # cannot retrigger it: schedule the next attempt explicitly
            # while the restart budget remains. Hand over from THIS
            # task (the incumbent): an already-scheduled attempt owns
            # the retry — a second concurrent owner would read the
            # same restart count and spawn a second process.
            if not self._closing and restarts + 1 < (
                self._settings.frontend_max_restarts
            ):
                self._schedule_restart(frontend_id, exclude=asyncio.current_task())
        finally:
            if replaying:
                # Failure or cancellation mid-replay: drop the queue —
                # the frontend is being retired (or shutting down), and
                # a queued live op times out exactly as if it had been
                # sent to a dying pipe.
                self._replaying.discard(frontend_id)
                self._replay_pending.pop(frontend_id, None)

    def _mark_metrics_process_dead(
        self, proc: multiprocessing.process.BaseProcess
    ) -> None:
        """Remove a dead frontend's multiprocess metric files."""
        if not self._settings.metrics_endpoint or proc.pid is None:
            return
        try:
            from prometheus_client import multiprocess

            os.environ.setdefault(
                "PROMETHEUS_MULTIPROC_DIR", self._settings.metrics_dir
            )
            multiprocess.mark_process_dead(proc.pid)
        except Exception:
            logger.debug("Failed marking metrics process dead", exc_info=True)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        if self._closing:
            return
        self._closing = True

        logger.info("Stopping frontend processes...")

        for task in self._listener_tasks:
            task.cancel()
        for task in self._restart_tasks:
            task.cancel()
        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)
        if self._restart_tasks:
            await asyncio.gather(*self._restart_tasks, return_exceptions=True)
        self._listener_tasks.clear()
        self._restart_tasks.clear()
        self._restart_tasks_by_frontend.clear()
        # In-flight model operations: the listeners are gone, so no new
        # ones arrive. Cancel and settle them while the control pipes
        # are still open — a pool-blocked op's deferred cancellation
        # settles by its own operation deadline, bounding this wait.
        for op_task in self._op_tasks:
            op_task.cancel()
        if self._op_tasks:
            await asyncio.gather(*self._op_tasks, return_exceptions=True)
        self._op_tasks.clear()
        # Ack notifications scheduled by death handling: settle them
        # before the retire confirmations so the collectors they wake
        # can observe their (possibly now-complete) ack sets.
        if self._ack_notify_tasks:
            await asyncio.gather(*self._ack_notify_tasks, return_exceptions=True)
        self._ack_notify_tasks.clear()
        # Retirement confirmations spawned by the cancelled restart
        # tasks: their children were already popped from _frontends,
        # so the sweep below cannot see them — drain their exit
        # confirmations first.
        if self._retire_tasks:
            await asyncio.gather(*self._retire_tasks, return_exceptions=True)

        # Close the supervisor's control ends BEFORE terminating: a
        # frontend's control reader blocks in recv() inside an executor
        # thread, and on Linux closing the local end does not interrupt
        # it — only the peer's EOF does. Closing first lets every
        # frontend's reader observe EOF, finish its executor work and
        # let the process exit within the grace period instead of
        # consuming it and being killed.
        for control in self._controls.values():
            try:
                control.close()
            except Exception:
                pass

        for proc in self._frontends.values():
            if _process_exited(proc):
                continue
            try:
                proc.terminate()
            except ProcessLookupError:
                # Reaped between the exit check and the signal; the
                # child is already gone.
                pass

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.parallel_workers_timeout
        for frontend_id, proc in self._frontends.items():
            remaining = max(0, deadline - loop.time())
            # to_thread: proc.join is a blocking call — running it on
            # the loop would freeze it for up to the full timeout × N
            # frontends, stalling the pool's SIGCHLD handler and
            # in-flight ZMQ responses during shutdown.
            await asyncio.to_thread(proc.join, timeout=remaining)
            # _process_exited, not is_alive: a corpse already reaped by
            # the pool's SIGCHLD handler makes is_alive() report True
            # forever (see _process_exited).
            if _process_exited(proc):
                continue
            logger.warning(
                f"Frontend {frontend_id} (PID {proc.pid}) did not "
                f"exit gracefully; killing..."
            )
            try:
                proc.kill()
            except ProcessLookupError:
                # Reaped between the exit check and the signal.
                pass
            # Confirm the exit before releasing ownership: an
            # unconfirmed kill leaves a process that still holds
            # the shared ports while the supervisor proceeds to
            # pool and IPC cleanup.
            await asyncio.to_thread(proc.join, timeout=5)
            if not _process_exited(proc):
                logger.error(
                    f"Frontend {frontend_id} (PID {proc.pid}) did not "
                    "exit after SIGKILL; shutdown incomplete"
                )
        self._controls.clear()
        self._frontends.clear()
        logger.info("All frontend processes stopped.")

        # Frontends drained first; now stop the shared pool's workers
        # through their control queues (workers ignore signals).
        pool_stop_error: Exception | None = None
        if self._pool_registry is not None:
            try:
                await self._pool_registry.close()
                self._pool_registry = None
            except Exception as exc:
                # The registry retains its workers and IPC endpoints on
                # failed cleanup (resource ownership): keep both the
                # registry and the transport directory so an operator
                # can still inspect them, and surface the failure
                # instead of reporting a clean shutdown. The failure is
                # terminal — the registry's internal close task is in a
                # failed state and a second stop() would await the same
                # failed task, so no retry is promised.
                pool_stop_error = exc
                logger.exception("Failed stopping shared inference pool")
        self._listener_executor.shutdown(wait=False)
        for executor in self._send_executors.values():
            executor.shutdown(wait=False)
        self._send_executors.clear()

        if self._transport_directory is not None and pool_stop_error is None:
            self._transport_directory.cleanup()
            self._transport_directory = None

        if pool_stop_error is not None:
            raise pool_stop_error
