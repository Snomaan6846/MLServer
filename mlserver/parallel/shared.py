# Frontend-side access to the supervisor's shared inference pool: the
# supervisor owns the pool and model instances; each frontend owns a
# dispatcher routing to those workers. Control plane = inherited
# multiprocessing pipes; data plane = ZeroMQ.

import asyncio

import json

from contextvars import ContextVar
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import Connection
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from collections.abc import Callable, Coroutine

from ..model import MLModel
from ..registry import MultiModelRegistry
from ..registry import PUBLICATION_GUARD_ATTR
from ..registry import StalePublication
from ..registry import model_initialiser as _default_model_initialiser
from ..settings import Settings, ModelSettings
from ..utils import generate_uuid, logger, consume_send_failure

from .dispatcher import Dispatcher
from .worker import Worker
from .model import ParallelModel


@dataclass
class OpRequest:
    """Frontend → supervisor: perform a model operation on the shared pool."""

    op_id: str
    kind: str  # "load" | "unload"
    serialised_model_settings: str | None = None
    name: str | None = None
    version: str | None = None
    # Generation of the instance the caller holds, for unloads: a
    # frontend retiring a superseded wrapper (same-version reload
    # cleanup) must not unload the current pool generation.
    generation: int | None = None


@dataclass
class ApplyAck:
    """Frontend → supervisor: a broadcast apply was handled.

    ``ok`` reports whether the frontend actually converged to the
    broadcast's state: a failed local apply acknowledges with
    ``ok=False`` so the supervisor can retire the diverged frontend
    through the restart path (a restarted frontend starts empty and
    converges via the readiness-gated inventory replay) instead of
    waiting for the next operation on the model to reconcile it. A
    stale-publication abort acknowledges ``ok=True`` — the frontend
    holds NEWER state, not diverged state.
    """

    op_id: str
    ok: bool = True


@dataclass
class CancelRequest:
    """Frontend → supervisor: a dispatched request was cancelled upstream."""

    model_name: str
    request_id: str
    worker_pid: int
    # Version of the cancelled request's model: the eviction targets
    # that version's batcher, not the name's default version.
    model_version: str | None = None


@dataclass
class WorkerReadyEvent:
    """Supervisor → frontends: a worker incarnation became routable."""

    pid: int
    request_endpoint: str
    # Env hash of the pool owning the worker; None is the default pool.
    # Frontends partition routing by it so workers only serve models of
    # their own pool.
    env_hash: str | None = None


@dataclass
class WorkerStoppedEvent:
    """Supervisor → frontends: a worker incarnation stopped."""

    pid: int
    exit_code: int
    env_hash: str | None = None
    # Endpoint of the stopped incarnation. A delayed stop for an old
    # incarnation must not revoke the registry's replacement proxy
    # (same PID, new endpoint): frontends match it before revoking.
    # None means the incarnation never announced an endpoint (or the
    # event predates this field) — frontends then revoke by PID alone.
    request_endpoint: str | None = None


@dataclass
class PoolQuarantinedEvent:
    """Supervisor → frontends: pause routing to a recovering pool."""

    # Workers stay registered (in-flight responses may still arrive);
    # recovery re-announces them with WorkerReadyEvents, which re-enable
    # routing. No explicit un-quarantine message exists.
    env_hash: str | None


@dataclass
class ApplyLoad:
    """Supervisor → all frontends: apply a (successful) pool load."""

    op_id: str
    serialised_model_settings: str
    ok: bool
    error: str | None = None
    # Env hash of the pool that committed the load; frontends tag the
    # model's routing partition with it.
    env_hash: str | None = None
    # Generation of the committed instance: frontends attach it to the
    # registered wrapper so a later retirement of a superseded wrapper
    # is not mistaken for a real unload.
    generation: int = 0


@dataclass
class ApplyUnload:
    """Supervisor → all frontends: apply a pool unload."""

    op_id: str
    name: str
    version: str | None
    ok: bool
    error: str | None = None
    # The generation that was unloaded: a peer holding a newer
    # generation must skip a stale unload, and the generation-aware
    # reconciliation uses it to decide late-load eligibility.
    generation: int = 0


@dataclass
class OpResult:
    """Supervisor → origin frontend: the operation completed everywhere."""

    op_id: str
    ok: bool
    error: str | None = None
    # For loads: the generation of the committed instance, so the
    # origin attaches it to the wrapper it registers.
    generation: int = 0


GENERATION_ATTR = "__pool_generation__"


def _attach_generation(model: MLModel | None, source: object) -> None:
    """Attach the committed generation to a registered wrapper."""
    if model is None:
        return
    generation = getattr(source, "generation", None)
    if generation is not None:
        setattr(model, GENERATION_ATTR, generation)


def _attach_publication_guard(model: MLModel, guard: "Callable[[], bool]") -> None:
    """Arm the registry's publication gate for an origin load."""
    setattr(model, PUBLICATION_GUARD_ATTR, guard)


def get_generation(model: MLModel) -> int | None:
    """The pool generation of the wrapper a caller holds, if known."""
    return getattr(model, GENERATION_ATTR, None)


@dataclass
class StartupComplete:
    """Supervisor → frontends: all startup models have been committed."""

    model_count: int


@dataclass
class StartupApplyFailed:
    """Frontend → supervisor: a startup model's local apply failed.

    The frontend keeps its readiness gate closed, but it would
    otherwise stay alive forever — still receiving SO_REUSEPORT
    traffic it cannot serve and never restarted. The supervisor
    terminates it so the monitor's restart machinery replaces it.
    """

    model_name: str


def serialise_model_settings(settings: ModelSettings) -> str:
    """Serialise model settings for the control plane.

    The settings must be JSON-serialisable: a value that Python's
    encoder cannot express (e.g. a ``pathlib.Path`` placed in
    ``parameters.extra`` by a custom repository) is rejected here with
    a clear error rather than a bare TypeError — the value never
    reaches the shared pool either way. Converting such values to
    strings silently is NOT an option: the worker would receive a
    different type than the runtime expects.
    """
    as_dict = settings.model_dump(by_alias=True)
    if settings._source:
        as_dict["_source"] = settings._source
    try:
        return json.dumps(as_dict)
    except TypeError as exc:
        from ..errors import MLServerError

        raise MLServerError(
            f"Model settings for '{settings.name}' contain a value that "
            "cannot be serialised to JSON (e.g. a Path or a custom object "
            "in parameters.extra). The shared-pool control plane "
            "requires JSON-serialisable model settings."
        ) from exc


def deserialise_model_settings(serialised: str) -> ModelSettings:
    # model_validate (dict path) is required: ModelSettings overrides it
    # to restore the private `_source` attribute, which pydantic's
    # model_validate_json drops. Without `_source`, dev-mode models
    # loaded through the shared pool lose the directory their custom
    # implementation resolves from.
    return ModelSettings.model_validate(json.loads(serialised))


class WorkerProxy:
    """Frontend-side stand-in for a supervisor-owned worker incarnation."""

    def __init__(self, pid: int, request_endpoint: str) -> None:
        self.pid = pid
        self._zmq_request_endpoint = request_endpoint

    @property
    def request_endpoint(self) -> str:
        return self._zmq_request_endpoint


class SharedPoolRegistry:
    """Inference-pool registry for frontends sharing a supervisor-owned pool."""

    def __init__(
        self,
        settings: Settings,
        frontend_id: int,
        reply_endpoint: str,
        control: Connection,
    ) -> None:
        self._settings = settings
        self._frontend_id = frontend_id
        self._control = control
        self._model_registry: MultiModelRegistry | None = None
        self._data_plane: Any = None
        # Guards against forwarding locally-applied broadcasts back to
        # the supervisor (which would loop the operation). Task-scoped:
        # a ContextVar is only visible to the task (and its children)
        # that runs the broadcast apply, so a concurrent user load in
        # another task — e.g. a REST model load arriving while a peer
        # apply is mid-flight — still forwards to the supervisor
        # instead of silently registering a model no worker loaded.
        # The value is the broadcast message being applied (or
        # False): the load hook reads its generation to bind it to
        # the wrapper BEFORE the registry publishes — a wrapper
        # visible without its generation would pass another load's
        # displacement guard (registered_generation is None).
        self._applying_load: ContextVar[object] = ContextVar(
            "mlserver_applying_load", default=False
        )
        # Startup gating: readiness deferred until the supervisor sends
        # StartupComplete (after all startup models were broadcast).
        # After startup, the user's empty_registry_readiness governs.
        self._startup_done: bool = False
        # A startup model's local apply failed on this frontend: the
        # readiness gate must stay closed (with the default
        # empty_registry_readiness=True, releasing it would report ready
        # while the startup model is missing here).
        self._startup_apply_failed: bool = False
        self._startup_failed_model: str | None = None
        # Set during frontend shutdown: model unloads become local no-ops
        # because the supervisor drains the shared pool itself.
        self._draining = False
        self._closing = False
        self._op_waiters: dict[str, asyncio.Future] = {}
        # Apply messages retained on the ORIGIN path: if the caller's
        # wait expires (or is cancelled) before the supervisor resolves
        # the operation, a committed load is applied late from here so
        # this frontend converges instead of missing the model the
        # pool now serves.
        self._origin_applies: dict[str, ApplyLoad] = {}
        self._reconcile_tasks: set[asyncio.Task] = set()
        # Generation of the last operation that published on each key:
        # a late-arriving origin result reconciles only if its
        # generation is above the last published one (an older result
        # is superseded). An unload records the generation it removed —
        # not infinity — so a newer-generation load committed after it
        # still reconciles.
        self._key_generations: dict[tuple[str, str], float] = {}
        # The generation at which each key was last REMOVED locally
        # (a broadcast unload's bumped generation). An origin whose
        # delayed load result arrives after a removal must not
        # republish its wrapper — the model is gone pool-wide; the
        # resurrection guard in load_model reads this. Cleared by a
        # successful load apply (the key is served again).
        self._key_removals: dict[tuple[str, str], float] = {}
        self._proxies: dict[int, WorkerProxy] = {}
        # Peer applies run as background tasks sequenced per model key:
        # the listener must stay receptive while a slow apply (a model
        # load) runs — blocking it freezes acknowledgements, operation
        # results and worker events, and two concurrent loads of the
        # same model from different frontends can then out-wait each
        # other's ack collection.
        self._apply_chains: dict[tuple[str, str], asyncio.Task] = {}
        # Every scheduled apply task, including superseded mid-chain
        # links (the chain map only keeps the latest link per key):
        # shutdown must cancel and settle all of them, not just the
        # tips — a cancelled tip does not cancel the links it waits on.
        self._apply_tasks: set[asyncio.Task] = set()

        # The response queue exists only to carry this frontend's reply
        # endpoint to the dispatcher, exactly like the pool does for the
        # single-process server. Replies arrive over ZeroMQ, not the queue.
        self._responses: Queue = Queue()
        setattr(self._responses, "_zmq_reply_endpoint", reply_endpoint)
        self._dispatcher = Dispatcher(
            {},
            self._responses,
            model_operation_timeout=self._settings.model_operation_timeout,
            worker_start_timeout=self._settings.worker_start_timeout,
            frontend_id=frontend_id,
            routing_rotation=self._settings.parallel_routing_rotation,
            inference_timeout=self._settings.inference_timeout,
        )
        # Remote cancellation: caller disconnects evict the queued request
        # from its worker-side batcher instead of computing it anyway.
        self._dispatcher.set_cancel_handler(self.cancel_request)
        self._listener_task: asyncio.Task | None = None
        # The recv executor blocks on the pipe; sends must never queue
        # behind it or they would deadlock against the listener.
        self._recv_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mlserver-frontend-{frontend_id}-ctl"
        )
        self._send_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mlserver-frontend-{frontend_id}-tx"
        )

    def attach_registry(self, model_registry: MultiModelRegistry) -> None:
        """Bind the frontend's model registry for broadcast applies."""
        self._model_registry = model_registry

    def attach_data_plane(self, data_plane) -> None:
        """Bind the data plane for startup-complete signalling."""
        self._data_plane = data_plane

    # ------------------------------------------------------------------
    # InferencePoolRegistry-compatible surface
    # ------------------------------------------------------------------

    def model_initialiser(self, model_settings: ModelSettings) -> MLModel:
        """Same semantics as the pool registry's initialiser."""
        parameters = model_settings.parameters
        if parameters and parameters.environment_tarball:
            return MLModel(model_settings)
        return _default_model_initialiser(model_settings)

    async def start(self) -> None:
        self._dispatcher.start()
        self._listener_task = asyncio.create_task(self._listen())

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        # Retained origin applies can no longer reconcile: their
        # results will never arrive after the listener stops.
        self._origin_applies.clear()
        for task in self._reconcile_tasks:
            task.cancel()
        if self._reconcile_tasks:
            await asyncio.gather(*self._reconcile_tasks, return_exceptions=True)
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None
        # In-flight applies scheduled by the listener: cancel and
        # settle them before the remaining teardown. The superset
        # covers mid-chain links too — cancelling a chain tip does
        # not cancel the links it waits on.
        for task in self._apply_tasks:
            task.cancel()
        if self._apply_tasks:
            await asyncio.gather(*self._apply_tasks, return_exceptions=True)
        self._apply_tasks.clear()
        self._apply_chains.clear()
        for waiter in self._op_waiters.values():
            if not waiter.done():
                waiter.cancel()
        self._op_waiters.clear()
        try:
            self._control.close()
        except Exception:
            logger.debug("Control pipe already closed", exc_info=True)
        # Dispatcher shutdown can report a transport-close error after
        # stopping its response reader; the queue and both executors
        # must still close (a second close() returns immediately once
        # _closing is set, so this is the only chance).
        try:
            await self._dispatcher.stop(
                timeout=self._settings.parallel_workers_timeout,
            )
        finally:
            # The dummy response queue only carried this frontend's
            # reply endpoint; close it so its semaphore pair is not
            # leaked.
            try:
                self._responses.close()
                self._responses.cancel_join_thread()
            except Exception:
                logger.debug("Response queue already closed", exc_info=True)
            self._recv_executor.shutdown(wait=False)
            self._send_executor.shutdown(wait=False)

    def begin_shutdown(self) -> None:
        """Stop forwarding model operations; the supervisor owns draining."""
        self._draining = True

    async def load_model(self, model: MLModel) -> MLModel:
        applying = self._applying_load.get()
        if applying or self._draining or self._closing:
            # Locally-applied broadcast (or shutdown drain): wrap and
            # bind the generation NOW — the registry publishes this
            # wrapper inside _load_for_apply, and a wrapper visible
            # without its generation would pass another load's
            # displacement guard (the retirement-cleanup await after
            # publication is the window).
            wrapper = ParallelModel(model, self._dispatcher)
            generation = getattr(applying, "generation", None)
            if generation:
                _attach_generation(wrapper, applying)
            return wrapper
        op_id = generate_uuid()
        # Register the waiter before sending: the supervisor's broadcast
        # can otherwise beat the registration, making the origin apply
        # locally and deadlock on its own suspended registry lock.
        waiter = self._begin_op(op_id)
        try:
            await self._send(
                OpRequest(
                    op_id=op_id,
                    kind="load",
                    serialised_model_settings=serialise_model_settings(model.settings),
                )
            )
            await self._await_op(op_id, waiter)
        finally:
            self._op_waiters.pop(op_id, None)
        # Success: the result was consumed and the local registration
        # happened above — the retained apply is no longer needed. On
        # timeout or cancellation the retention must SURVIVE so a late
        # committed result can reconcile this frontend (see
        # _handle_op_result); it is bounded by close() clearing it.
        self._origin_applies.pop(op_id, None)
        result = getattr(waiter, "mlserver_result", None)
        if result is not None and result.generation:
            self._mark_key_generation(model.name, model.version, result.generation)
        if result is not None and result.generation:
            key = (model.name, model.version or "")
            # A newer generation may have been applied LOCALLY while
            # this load waited for its pool round (a same-key
            # operation that committed after ours — the supervisor
            # serialises pool transitions, not frontend publications).
            # Publishing our own older wrapper would displace the
            # newer registration, and the displaced wrapper's
            # retirement would unload the CURRENT pool state — a
            # stale publication. Applies RESERVE their generation
            # before their local load (see _apply_load), so this
            # check sees a newer generation the moment its apply
            # starts — no waiting on the apply's model load. The span
            # from this check through the registry's publication is
            # synchronous (get_model never yields), so no apply can
            # interleave between the check and the registration.
            if self._key_generations.get(key, 0) > result.generation:
                # EXACT-key lookup (never the default-version
                # fallback): an unversioned load must not hand back a
                # versioned sibling — the rollback would treat the
                # live sibling as this load's unpublished instance and
                # unload it. Only a READY wrapper of the exact key
                # with a STRICTLY newer generation is a convergence
                # winner.
                assert self._model_registry is not None
                newer = self._model_registry.find_registered(model.name, model.version)
                newer_generation = get_generation(newer) if newer is not None else None
                if (
                    newer is not None
                    and newer is not model
                    and getattr(newer, "ready", False)
                    and newer_generation is not None
                    and newer_generation > result.generation
                ):
                    # Hand the registry the newer registered wrapper:
                    # publishing it re-registers the same object and
                    # displaces nothing. The gate is armed for the
                    # newer generation (the same publication window
                    # applies).
                    _attach_publication_guard(
                        newer, self._publication_guard(key, newer, result.generation)
                    )
                    return newer
            removal = self._key_removals.get(key)
            if removal is not None and removal >= result.generation:
                # The model was REMOVED pool-wide while this load
                # waited (a same-key unload committed after our
                # round). Publishing our wrapper would resurrect a
                # model the pool no longer serves — requests would
                # route to workers that no longer hold it. The load
                # committed, but its state is gone; fail the caller
                # with the truth instead of a phantom registration.
                raise RuntimeError(
                    f"Model '{model.name}' was unloaded while its load "
                    f"was in flight (generation {result.generation} "
                    f"superseded by removal at {removal}); not "
                    f"republishing locally"
                )
            # Our publication re-adds the key: a stale removal mark
            # must not fail a LATER load whose generation superseded
            # it (that mark is older than our commit — the raise
            # above filtered anything at or above our generation).
            if self._key_removals.get(key, 0) < result.generation:
                self._key_removals.pop(key, None)

        wrapper = ParallelModel(model, self._dispatcher)
        _attach_generation(wrapper, getattr(waiter, "mlserver_result", None))
        if result is not None and result.generation:
            # Arm the registry's publication gate: the checks above
            # run inside this hook, but remaining hooks and model
            # preparation still await before publication — a lock-free
            # unload apply can commit a removal in that window.
            _attach_publication_guard(
                wrapper, self._publication_guard(key, wrapper, result.generation)
            )
        return wrapper

    def _publication_guard(
        self, key: "tuple[str, str]", model: MLModel, fallback_generation: int
    ) -> "Callable[[], bool]":
        """A stale check for the registry's publication gate."""

        def _live_removal_watermark() -> float:
            # The registry's convergence check reads the LIVE
            # watermark after its rollback: a removal reserved while
            # the rollback cleanup ran must invalidate a winner that
            # the abort-time snapshot still endorsed.
            return self._key_removals.get(key, 0)

        def _superseded_at_publication() -> bool:
            # Read the generation at GATE time: a swapped-in wrapper
            # may not carry its generation yet at attach time (the
            # apply attaches it after its registry call returns).
            expected_generation = get_generation(model) or fallback_generation
            # A removal at-or-after our commit: publishing would
            # resurrect a model the pool no longer serves.
            removal = self._key_removals.get(key, 0)
            if removal >= expected_generation:
                # The abort cause, read by the registry's gate when it
                # raises StalePublication (the publication_abort
                # attribute contract): "removed" carries the removal
                # watermark so the load's convergence check can reject
                # wrappers the removal is about to retire.
                setattr(
                    _superseded_at_publication,
                    "publication_abort",
                    ("removed", removal),
                )
                return True
            if self._key_generations.get(key, 0) <= expected_generation:
                return False
            # A newer generation exists: harmful only when REGISTERED
            # here — publishing ours would displace it and its
            # retirement would unload the current pool state. A
            # mid-flight newer apply publishes over ours afterwards
            # (its retirement is superseded) — self-healing, not a
            # stale publication.
            if self._model_registry is None:
                return False
            registered = self._model_registry.find_registered(key[0], key[1] or None)
            registered_generation = (
                get_generation(registered) if registered is not None else None
            )
            superseded = (
                registered_generation is not None
                and registered_generation > expected_generation
            )
            if superseded:
                setattr(
                    _superseded_at_publication,
                    "publication_abort",
                    ("superseded", None),
                )
            return superseded

        setattr(
            _superseded_at_publication, "removal_watermark", _live_removal_watermark
        )
        return _superseded_at_publication

    async def unload_model(self, model: MLModel) -> MLModel:
        if self._applying_load.get() or self._draining or self._closing:
            return model
        if get_generation(model) is None:
            # A model that never acquired a pool generation is the
            # initial (unwrapped) model of a failed load: the pool holds
            # nothing from this operation, so cleanup stays local.
            # Forwarding with no generation would be treated as
            # permission to unload the current serving instance of the
            # key — a failed reload must not take it offline.
            logger.debug(
                "Local-only cleanup of unowned model %s (never committed "
                "to the shared pool)",
                model.name,
            )
            return model
        generation = get_generation(model)
        if generation is not None and generation < self._key_generations.get(
            (model.name, model.version or ""), 0
        ):
            # A retirement (reload/unload cleanup of a displaced
            # wrapper) whose generation is older than this frontend's
            # watermark: the commit that displaced it already settled
            # the pool state (a reload's commit unloads the displaced
            # pool instance; an unload's commit removes it). Forwarding
            # would ask the supervisor to unload a generation it may
            # still be serving — the retirement of a CURRENT-looking
            # generation is exactly how a displaced wrapper's cleanup
            # can take a live model offline. Cleanup stays local.
            logger.debug(
                "Local-only retirement of superseded wrapper %s (gen %s "
                "< local gen %s)",
                model.name,
                get_generation(model),
                self._key_generations.get((model.name, model.version or ""), 0),
            )
            return model
        op_id = generate_uuid()
        waiter = self._begin_op(op_id)
        try:
            await self._send(
                OpRequest(
                    op_id=op_id,
                    kind="unload",
                    name=model.name,
                    version=model.version,
                    generation=get_generation(model),
                )
            )
            await self._await_op(op_id, waiter)
            # No generation marking here: a superseded retirement also
            # resolves ok (the supervisor returns success without
            # removing anything), and marking would prevent any
            # subsequent reconciliation for a newer committed
            # generation. Real removals are marked by the broadcast
            # listener path (_handle_apply_unload), which only fires
            # for actual unloads — superseded retirements produce no
            # broadcast, so retained applies stay eligible.
        finally:
            self._op_waiters.pop(op_id, None)
        return model

    # ------------------------------------------------------------------
    # Control-plane plumbing
    # ------------------------------------------------------------------

    async def _send(self, message: object) -> None:
        """Send a control message without blocking the event loop."""
        if self._closing:
            raise RuntimeError("Shared pool control channel is closed")
        loop = asyncio.get_running_loop()
        # Delivery deadline, mirroring the supervisor's send writers: a
        # supervisor that stops reading its control pipe must not pin
        # this frontend's applies and acknowledgements forever.
        await asyncio.wait_for(
            loop.run_in_executor(self._send_executor, self._control.send, message),
            timeout=self._settings.model_operation_timeout,
        )

    def _begin_op(self, op_id: str) -> "asyncio.Future":
        """Register the operation's result waiter before any send."""
        waiter = asyncio.get_running_loop().create_future()
        self._op_waiters[op_id] = waiter
        return waiter

    async def _await_op(self, op_id: str, waiter: "asyncio.Future") -> None:
        try:
            await asyncio.wait_for(
                waiter, timeout=self._settings.model_operation_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Model operation {op_id} timed out on the shared pool"
            ) from None
        result = waiter.result()
        if isinstance(result, OpResult) and not result.ok:
            raise RuntimeError(result.error or "Model operation failed on shared pool")
        # The waiter keeps the resolved OpResult so load_model can
        # attach the committed generation to its wrapper.
        setattr(waiter, "mlserver_result", result)

    async def _listen(self) -> None:
        """Receive control messages until the pipe closes."""
        loop = asyncio.get_running_loop()
        while not self._closing:
            try:
                message = await loop.run_in_executor(
                    self._recv_executor, self._control.recv
                )
            except (EOFError, OSError):
                # Supervisor closed the pipe (shutdown or death).
                return
            except asyncio.CancelledError:
                raise
            try:
                await self._handle(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never log the message payload or the exception text:
                # ApplyLoad carries the full serialised settings, whose
                # parameters.extra can hold model-store credentials, and
                # validation errors echo their input (CWE-532). The
                # traceback goes to debug only for the same reason.
                logger.error(
                    "Failed handling shared-pool control message %s (op %s): %s",
                    type(message).__name__,
                    getattr(message, "op_id", None),
                    type(exc).__name__,
                )
                logger.debug(
                    "Shared-pool control message failure detail",
                    exc_info=True,
                )

    def _sequence_apply(
        self,
        key: tuple[str, str],
        apply_factory: "Callable[[], Coroutine[Any, Any, None]]",
    ) -> asyncio.Task:
        """Run a peer apply off the listener, sequenced per model key.

        Per-key chaining preserves apply order (routing tags, batch
        affinity and registry state stay consistent across concurrent
        operations on one key); the acknowledgement still means
        APPLIED — it is sent when the apply completes, inside the
        task. The coroutine is CREATED only when its turn begins: a
        cancelled queued apply never leaves an unawaited coroutine
        behind (the runtime warning).
        """
        previous = self._apply_chains.get(key)

        async def _sequenced() -> None:
            try:
                if previous is not None:
                    # asyncio.wait never raises from the previous
                    # link's failure or cancellation — only this
                    # task's OWN cancellation raises here, and it must
                    # propagate: awaiting the previous task directly
                    # would swallow this task's cancellation into the
                    # failure handler and run the apply anyway, making
                    # close()'s cancel-and-settle wait out a full
                    # model load during shutdown.
                    await asyncio.wait({previous})
                await apply_factory()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Payload-free, mirroring _listen's logging discipline.
                logger.error(
                    "Frontend %s: apply task failed: %s",
                    self._frontend_id,
                    "see debug logs",
                )
                logger.debug("Apply task failure detail", exc_info=True)

        task = asyncio.create_task(_sequenced())
        self._apply_chains[key] = task
        self._apply_tasks.add(task)

        def _clear(done: asyncio.Task) -> None:
            if self._apply_chains.get(key) is done:
                del self._apply_chains[key]
            self._apply_tasks.discard(done)

        task.add_done_callback(_clear)
        return task

    async def _handle(self, message: object) -> "asyncio.Task | None":
        # Apply messages run as sequenced background tasks (see
        # _sequence_apply); the returned task lets a driver await the
        # apply's completion. Everything else is handled inline.
        if isinstance(message, WorkerReadyEvent):
            await self._handle_worker_ready(message)
        elif isinstance(message, WorkerStoppedEvent):
            self._handle_worker_stopped(message)
        elif isinstance(message, ApplyLoad):
            return await self._handle_apply_load(message)
        elif isinstance(message, ApplyUnload):
            return await self._handle_apply_unload(message)
        elif isinstance(message, OpResult):
            self._handle_op_result(message)
        elif isinstance(message, StartupComplete):
            await self._handle_startup_complete(message)
        elif isinstance(message, PoolQuarantinedEvent):
            self._handle_pool_quarantined(message)
        else:
            # Type only: an unhandled message type's repr could carry
            # settings payloads (see the handler above).
            logger.warning(
                "Unknown shared-pool control message: %s", type(message).__name__
            )
        return None

    async def _handle_worker_ready(self, message: WorkerReadyEvent) -> None:
        # Reuse the existing proxy for the same worker incarnation:
        # quarantine recovery re-announces the same PID and endpoint,
        # and replacing the proxy would orphan in-flight requests
        # dispatched to the old proxy (their response futures would
        # stay pending with no cancellation path — the old proxy's
        # stop event no longer matches the registry's entry).
        proxy = self._proxies.get(message.pid)
        if proxy is None or proxy.request_endpoint != message.request_endpoint:
            if proxy is not None:
                # Endpoint changed (worker restart on a new endpoint):
                # retire the OLD proxy first so its in-flight requests
                # are cancelled, then publish the replacement. Without
                # this, the old proxy's requests strand — the stop
                # event targets the NEW proxy (PID-only match), and
                # the old proxy is no longer in the registry to
                # receive the cancellation.
                old_proxy = self._proxies.pop(message.pid, None)
                if old_proxy is not None:
                    # exit_code=-1 marks "incarnation replaced" (not a
                    # clean exit) so the cancellation error and logs
                    # read correctly.
                    self._dispatcher.on_worker_stop(
                        cast(Worker, old_proxy), exit_code=-1
                    )
            proxy = WorkerProxy(message.pid, message.request_endpoint)
            self._proxies[message.pid] = proxy
        # Proxies deliberately duck-type the Worker surface the dispatcher
        # reads (pid + request endpoint), mirroring supervisor incarnations.
        await self._dispatcher.on_worker_start(cast(Worker, proxy))
        self._dispatcher.on_worker_ready(cast(Worker, proxy), env_hash=message.env_hash)
        logger.debug("Frontend %s routing to worker %s", self._frontend_id, message.pid)

    def _handle_worker_stopped(self, message: WorkerStoppedEvent) -> None:
        proxy = self._proxies.get(message.pid)
        if proxy is None:
            return
        if (
            message.request_endpoint is not None
            and proxy.request_endpoint != message.request_endpoint
        ):
            # A delayed stop for an OLD incarnation: the registry
            # already holds the replacement (its ready event retired
            # the old proxy). Revoking by PID alone would remove the
            # replacement and fail its in-flight requests.
            logger.debug(
                "Frontend %s ignored stale stop for worker %s at %s "
                "(current incarnation at %s)",
                self._frontend_id,
                message.pid,
                message.request_endpoint,
                proxy.request_endpoint,
            )
            return
        self._proxies.pop(message.pid, None)
        self._dispatcher.on_worker_stop(cast(Worker, proxy), message.exit_code)
        logger.debug(
            "Frontend %s revoked worker %s (exit %s)",
            self._frontend_id,
            message.pid,
            message.exit_code,
        )

    async def _handle_startup_complete(self, message: StartupComplete) -> None:
        """All startup models committed; release the readiness gate."""
        # Applies run as background tasks: the startup inventory is
        # complete only once every pending apply has settled (a failed
        # startup apply must record its gate-closing mark first).
        if self._apply_chains:
            await asyncio.gather(
                *list(self._apply_chains.values()), return_exceptions=True
            )
        self._startup_done = True
        if self._startup_apply_failed:
            # A startup model's local apply failed on this frontend:
            # keep the readiness gate closed. With the default
            # empty_registry_readiness=True, releasing it would report
            # ready while the startup model is missing here.
            logger.error(
                "Frontend %s: startup complete but %s apply(s) failed; "
                "readiness gate remains closed",
                self._frontend_id,
                "a startup model",
            )
            # Report to the supervisor and let it decide: this frontend
            # would otherwise stay alive forever — still receiving
            # SO_REUSEPORT traffic it cannot serve, never restarted.
            # The supervisor terminates it so the monitor's restart
            # machinery replaces it.
            await self._send(
                StartupApplyFailed(model_name=self._startup_failed_model or "unknown")
            )
            return
        logger.debug(
            "Frontend %s startup complete (%d models)",
            self._frontend_id,
            message.model_count,
        )
        # Release the deferred readiness: MLServer.start() skipped
        # startup_complete while models were pending.
        if self._data_plane is not None:
            self._data_plane.startup_complete()

    def _handle_pool_quarantined(self, message: PoolQuarantinedEvent) -> None:
        """Pause routing to a recovering pool; recovery re-enables it."""
        self._dispatcher.quarantine_pool(message.env_hash)
        logger.debug(
            "Frontend %s paused routing to pool %s during recovery",
            self._frontend_id,
            message.env_hash,
        )

    def _handle_op_result(self, message: OpResult) -> None:
        waiter = self._op_waiters.get(message.op_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(message)
            return
        # No live waiter: the caller's wait expired (or was cancelled)
        # before the supervisor resolved the operation. A committed
        # load would otherwise leave this frontend without the model
        # the pool now serves — apply it late so the frontend converges
        # without waiting for the next operation on the same model.
        # (An unload needs no reconciliation: the origin's registry
        # unregisters before the hook that forwards to the supervisor,
        # so a timed-out unload has already removed the local entry.)
        pending = self._origin_applies.pop(message.op_id, None)
        if message.ok and pending is not None:
            retained = pending
            task = asyncio.create_task(self._reconcile_origin_apply(retained, message))
            self._reconcile_tasks.add(task)
            task.add_done_callback(self._reconcile_tasks.discard)

    def _mark_key_generation(
        self, name: str, version: str | None, generation: float
    ) -> None:
        """Record the generation of the last operation on this key."""
        key = (name, version or "")
        self._key_generations[key] = max(self._key_generations.get(key, 0), generation)

    async def _reconcile_origin_apply(
        self, message: ApplyLoad, result: OpResult
    ) -> None:
        """Late-apply a committed load whose origin stopped waiting."""
        assert self._model_registry is not None
        settings = deserialise_model_settings(message.serialised_model_settings)
        key = (settings.name, settings.version or "")
        # A generation at or below the last published one must not
        # overwrite it: an older result is superseded, and a result
        # matching a just-unloaded generation was removed at that
        # generation. A strictly NEWER result still reconciles.
        if result.generation <= self._key_generations.get(key, 0):
            logger.debug(
                "Frontend %s skipped late apply of %s: a newer generation "
                "already published on this key",
                self._frontend_id,
                message.op_id,
            )
            return

        # The generation must be re-checked AT PUBLICATION, not only
        # here: the reconcile's model load spans lock waits and
        # native preparation, during which a newer generation or a
        # REMOVAL can apply locally (the applies are lock-free — they
        # do not queue behind this reconcile's registry locks).
        # Publishing then would displace the newer state or
        # resurrect the removed model. The registry's stale gate
        # aborts the publication when this fires.
        def _superseded() -> bool:
            return (
                self._key_generations.get(key, 0) > result.generation
                or self._key_removals.get(key, 0) >= result.generation
            )

        token = self._applying_load.set(result)
        try:
            # The lock-free apply entry point with the publication
            # gate. The reconcile is not an acknowledgement path
            # (nothing waits on it), and the gate — not the registry
            # lock — is what protects its publication against the
            # lock-free applies around it.
            registered = await self._model_registry._load_for_apply(
                settings, stale_check=_superseded
            )
            _attach_generation(registered, result)
            self._mark_key_generation(
                settings.name, settings.version, result.generation
            )
            # The key is served again: clear a stale removal mark —
            # but ONLY one strictly older than this publication. A
            # mark from a NEWER generation (an unload reserved while
            # this reconcile loaded) is live state: erasing it would
            # let a delayed older origin publish through the
            # resurrection guard and re-register a model the pool
            # removed at that newer generation.
            if self._key_removals.get(key, 0) < result.generation:
                self._key_removals.pop(key, None)
            logger.debug(
                "Frontend %s late-applied committed load %s after the "
                "caller stopped waiting",
                self._frontend_id,
                message.op_id,
            )
        except StalePublication:
            # A newer generation or a removal landed locally while the
            # reconcile loaded: the committed state it was converging
            # to has been superseded — skip (the local state stands).
            logger.debug(
                "Frontend %s skipped late apply of %s: superseded at "
                "publication by a newer local state",
                self._frontend_id,
                message.op_id,
            )
        except Exception:
            logger.exception(
                "Frontend %s failed late-applying committed load %s; "
                "the next operation on the model must reconcile it",
                self._frontend_id,
                message.op_id,
            )
        finally:
            self._applying_load.reset(token)

    async def _handle_apply_load(self, message: ApplyLoad) -> "asyncio.Task | None":
        assert self._model_registry is not None
        if not message.ok:
            # Nothing to apply anywhere, but every frontend must still
            # acknowledge: the supervisor resolves the origin's operation
            # only after all live frontends have acked.
            await self._send(ApplyAck(op_id=message.op_id))
            return None
        settings = deserialise_model_settings(message.serialised_model_settings)

        def _load_factory():
            return self._apply_load(message, settings)

        return self._sequence_apply(
            (settings.name, settings.version or ""),
            _load_factory,
        )

    async def _apply_load(self, message: ApplyLoad, settings: ModelSettings) -> None:
        assert self._model_registry is not None
        if message.generation < self._key_generations.get(
            (settings.name, settings.version or ""), 0
        ):
            # An obsolete message (a stale replay or a duplicate
            # broadcast): a newer generation is already applied on
            # this key here. Reject BEFORE any side effect — routing
            # tags and the registry may only change under the
            # generation check. Applying the routing below would
            # redirect live inference to the old pool and restore the
            # old batching flag while the newer wrapper stays
            # registered (the registry's own gates keep the wrapper;
            # the routing has none). Acknowledge success: this
            # frontend holds NEWER state, not diverged state.
            logger.debug(
                "Frontend %s skipped obsolete apply of %s (gen %d < " "held gen %d)",
                self._frontend_id,
                settings.name,
                message.generation,
                self._key_generations.get((settings.name, settings.version or ""), 0),
            )
            await self._send(ApplyAck(op_id=message.op_id))
            return
        self._track_batched(settings, settings.version)
        # Route the model version to the pool that committed the load
        # (runs on the origin too, before its early return below).
        self._dispatcher.set_model_pool(
            settings.name, message.env_hash, settings.version
        )
        if message.op_id in self._op_waiters:
            # Origin frontend: our own suspended load call completes the
            # local registration; ack receipt only. Retain the message:
            # if the caller's wait expires before the result arrives,
            # the committed load is late-applied from it — unless a
            # newer operation on the key has published by then (see
            # _handle_op_result).
            self._origin_applies[message.op_id] = message
            await self._send(ApplyAck(op_id=message.op_id))
            return
        diverged = False
        token = self._applying_load.set(message)
        try:
            # Reserve the generation BEFORE the local apply. The mark
            # is what an origin resuming mid-apply reads to refuse
            # displacing a newer registration (load_model's
            # stale-publication guard): marking after the apply would
            # leave a window spanning the whole publication and its
            # retirement cleanups during which the origin would see a
            # stale watermark. Reserving up front also means the
            # origin's guard never waits for this apply — it sees the
            # generation the moment the apply starts.
            self._mark_key_generation(
                settings.name, settings.version, message.generation
            )
            # Publish-time staleness gate: an origin (or a newer
            # chained apply) may publish a NEWER generation of this
            # key while our model loads. Publishing our older wrapper
            # would displace the newer registration and its
            # retirement would unload the CURRENT pool state. The
            # registry aborts the publication when this fires.
            key = (settings.name, settings.version or "")

            def _superseded() -> bool:
                return self._key_generations.get(key, 0) > message.generation

            # Peer apply: the lock-free registry entry point. The
            # locked load() would wait on the operation lock an origin
            # operation on this key holds across its supervisor round
            # — stalling the acknowledgement collection the origin is
            # waiting on (the ACK-collection deadlock).
            registered = await self._model_registry._load_for_apply(
                settings, stale_check=_superseded
            )
            _attach_generation(registered, message)
            # The key is served again: a later origin load delayed
            # behind this apply must not hit a STALE resurrection
            # guard — but a mark from a newer generation is live
            # state and must survive this publication.
            if self._key_removals.get(key, 0) < message.generation:
                self._key_removals.pop(key, None)
        except StalePublication:
            # The publication aborted because a NEWER generation is
            # registered here: this frontend is converged (it holds
            # newer state), not diverged — acknowledge success.
            logger.debug(
                "Frontend %s: apply of %s superseded by a newer local "
                "generation; publication aborted",
                self._frontend_id,
                settings.name,
            )
        except Exception:
            if not self._startup_done:
                # A startup model's local apply failed: this frontend
                # cannot serve the full inventory. Mark it so the
                # readiness gate stays closed when StartupComplete
                # arrives (the exception is logged by _listen; the
                # frontend stays up for diagnostics).
                self._startup_apply_failed = True
                self._startup_failed_model = settings.name
                logger.error(
                    "Frontend %s: startup apply of %s failed; "
                    "readiness gate will remain closed",
                    self._frontend_id,
                    settings.name,
                )
                diverged = True
            else:
                # A non-startup apply failed: ack with ok=False so
                # the supervisor retires this frontend through the
                # restart path — otherwise it would keep receiving
                # traffic without the model until some later
                # operation on it happens to reconcile it.
                logger.exception(
                    "Frontend %s: apply of %s failed; reporting divergence "
                    "for retirement",
                    self._frontend_id,
                    settings.name,
                )
                diverged = True
        finally:
            self._applying_load.reset(token)
        # Always ack (the supervisor resolves the operation only after
        # every live frontend acknowledged), carrying the outcome: a
        # diverged frontend is retired by the supervisor; a converged
        # one (stale-publication abort included) is left as is.
        await self._send(ApplyAck(op_id=message.op_id, ok=not diverged))

    async def _handle_apply_unload(self, message: ApplyUnload) -> "asyncio.Task | None":
        assert self._model_registry is not None
        if not message.ok:
            # Same as load: always acknowledge failures.
            await self._send(ApplyAck(op_id=message.op_id))
            return None
        return self._sequence_apply(
            (message.name, message.version or ""),
            lambda: self._apply_unload(message),
        )

    async def _apply_unload(self, message: ApplyUnload) -> None:
        assert self._model_registry is not None
        # A stale unload must not change ANY state — routing tags, batch
        # affinity, or the registry — for a generation this frontend
        # has already superseded. Check BEFORE touching routing: the
        # newer load's ApplyLoad arrived during the supervisor's pool
        # await, after the unload check passed, and its routing tags
        # must survive. This applies to the ORIGIN too: with the
        # load-first displacement ordering, the origin's routing was
        # already updated to the new pool by the ApplyLoad, and a
        # stale displaced-instance unload must not clear it.
        if message.generation < self._key_generations.get(
            (message.name, message.version or ""), 0
        ):
            logger.debug(
                "Frontend %s skipped stale unload of %s (gen %d < held gen): "
                "a newer generation is registered here",
                self._frontend_id,
                message.name,
                message.generation,
            )
            await self._send(ApplyAck(op_id=message.op_id))
            return
        # Routing and batch affinity are per (name, version): unloading
        # m/v1 must leave m/v2's affinity intact, and a plain version
        # must not inherit a batched sibling's sticky routing. Only a
        # current (non-stale) unload reaches here.
        self._dispatcher.set_model_batched(message.name, False, message.version)
        self._dispatcher.set_model_pool(message.name, None, message.version)
        if message.op_id in self._op_waiters:
            # Origin frontend: our own suspended unload call removes the
            # local registration; ack receipt only. Record the removed
            # generation (NOT infinity — a newer-generation load that
            # committed after this unload should still reconcile).
            self._mark_key_generation(message.name, message.version, message.generation)
            await self._send(ApplyAck(op_id=message.op_id))
            return
        token = self._applying_load.set(True)
        # Reserve BOTH marks before the registry call: the retirement
        # cleanups drained inside the local unload must already see
        # this generation (their superseded wrappers stay local-only —
        # forwarding them is the unload-side ACK cycle), and an origin
        # load delayed behind this apply must see the removal and not
        # republish its wrapper (the resurrection guard).
        self._mark_key_generation(message.name, message.version, message.generation)
        self._key_removals[(message.name, message.version or "")] = message.generation
        # Like _apply_load: initialised up front so a control-flow
        # change in the branches below can never reference an
        # unbound flag (a CancelledError skips the ack send
        # entirely, matching apply-cancellation semantics).
        diverged = False
        try:
            # Peer apply: the lock-free registry entry point (same
            # ACK-collection rationale as the load apply).
            await self._model_registry._unload_for_apply(
                message.name, message.version, False
            )
        except Exception:
            # A genuine local unload failure (e.g. a retained cleanup
            # that keeps failing before the unregister): the pool has
            # already unloaded the model, so this frontend diverges.
            # Report it at error level — an absent local model is the
            # desired post-unload state and never reaches here
            # (unload_exact_version returns normally for a miss) — and
            # ack ok=False so the supervisor retires this frontend.
            # (The routing tags were already cleared above, so the
            # stuck local entry receives no traffic; the divergence is
            # a resource retention, not a serving one.)
            logger.exception(
                "Broadcast unload of %s failed to apply locally; "
                "reporting divergence for retirement",
                message.name,
            )
            diverged = True
        else:
            diverged = False
        finally:
            self._applying_load.reset(token)
        await self._send(ApplyAck(op_id=message.op_id, ok=not diverged))

    @property
    def startup_models_pending(self) -> bool:
        """True while startup models are still being broadcast."""
        return not self._startup_done

    def _track_batched(
        self, settings: ModelSettings, version: str | None = None
    ) -> None:
        batched = settings.max_batch_size > 1 and settings.max_batch_time > 0
        # Set the flag explicitly on every load, including False: a
        # reload that disables batching must clear the sticky flag the
        # previous generation set, or the model keeps routing to one
        # worker after the batcher is gone.
        self._dispatcher.set_model_batched(
            settings.name, batched, version or settings.version
        )

    # ------------------------------------------------------------------
    # Request cancellation (fire-and-forget)
    # ------------------------------------------------------------------

    def cancel_request(
        self,
        model_name: str,
        request_id: str,
        worker_pid: int,
        model_version: str | None = None,
    ) -> None:
        """Best-effort eviction of a queued request from a worker batcher."""
        try:
            # Route through the send executor to avoid interleaving with
            # a concurrent OpRequest write on the same pipe.
            future = self._send_executor.submit(
                self._control.send,
                CancelRequest(
                    model_name=model_name,
                    request_id=request_id,
                    worker_pid=worker_pid,
                    model_version=model_version,
                ),
            )
            # Retrieve a late failure (a broken pipe after supervisor
            # death): an unretrieved exception would flood the log with
            # 'exception was never retrieved' during high-traffic
            # restarts.
            future.add_done_callback(consume_send_failure)
        except Exception:
            # Accept-and-discard remains correct if the cancel is lost.
            logger.debug("Failed forwarding cancel for request %s", request_id)
