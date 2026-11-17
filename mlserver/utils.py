import os
import uuid
import signal
import socket
import asyncio
import inspect
import urllib.parse
from contextvars import ContextVar, copy_context

from asyncio import Task
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future
from typing import Any, ParamSpec, TypeVar
from functools import wraps

from .logging import logger
from .types import InferenceRequest, InferenceResponse, Parameters
from .settings import ModelSettings, DEFAULT_MODEL_OPERATION_TIMEOUT
from .errors import InvalidModelURI
from .version import __version__

T = TypeVar("T")
P = ParamSpec("P")


_deferred_cancellation_active: ContextVar[bool] = ContextVar(
    "deferred_cancellation_active", default=False
)


async def get_model_uri(
    settings: ModelSettings, wellknown_filenames: list[str] = []
) -> str:
    if not settings.parameters:
        raise InvalidModelURI(settings.name)

    model_uri = settings.parameters.uri
    if not model_uri:
        raise InvalidModelURI(settings.name)

    model_uri_components = urllib.parse.urlparse(model_uri, scheme="file")
    if model_uri_components.scheme != "file":
        return model_uri

    full_model_path = to_absolute_path(settings, model_uri_components.path)
    if os.path.isfile(full_model_path):
        return full_model_path

    if os.path.isdir(full_model_path):
        # If full_model_path is a folder, search for a well-known model filename
        for fname in wellknown_filenames:
            model_path = os.path.join(full_model_path, fname)
            if os.path.isfile(model_path):
                return model_path

        # If none, return the folder
        return full_model_path

    # Otherwise, the uri is neither a file nor a folder
    raise InvalidModelURI(settings.name, full_model_path)


def to_absolute_path(model_settings: ModelSettings, uri: str) -> str:
    source = model_settings._source
    if source is None:
        # Treat path as either absolute or relative to the working directory of
        # the MLServer instance
        return uri

    parent_folder = os.path.dirname(source)
    unnormalised = os.path.join(parent_folder, uri)
    return os.path.normpath(unnormalised)


def get_wrapped_method(f: Callable) -> Callable:
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__  # type: ignore

    return f


def generate_uuid() -> str:
    return str(uuid.uuid4())


def consume_send_failure(future: Future[Any] | asyncio.Future[Any]) -> None:
    """Retrieve (and drop) a late fire-and-forget send failure."""
    # A delivery deadline (or process exit) abandons the send's future;
    # without retrieving the eventual exception the executor logs
    # 'exception was never retrieved' when a dead peer's pipe breaks.
    # Accepts both future flavours: executor submissions return
    # concurrent.futures.Future, run_in_executor wraps asyncio.Future.
    if not future.cancelled():
        future.exception()


def insert_headers(
    inference_request: InferenceRequest, headers: dict[str, str]
) -> InferenceRequest:
    # Ensure parameters are present
    if inference_request.parameters is None:
        inference_request.parameters = Parameters()

    parameters = inference_request.parameters

    if parameters.headers is not None:
        # TODO: Raise warning that headers will be replaced and shouldn't be used
        logger.warning(
            f"There are {len(parameters.headers)} entries present in the"
            "`headers` field of the request `parameters` object."
            "The `headers` field of the `parameters` object "
            "SHOULDN'T BE USED directly."
            "These entries will be replaced by the actual headers (REST) "
            "or metadata (gRPC) of the incoming request."
        )

    parameters.headers = headers
    return inference_request


def extract_headers(inference_response: InferenceResponse) -> dict[str, str] | None:
    if inference_response.parameters is None:
        return None

    parameters = inference_response.parameters
    if parameters.headers is None:
        return None

    headers = parameters.headers
    parameters.headers = None
    return headers


def _check_current_event_loop_policy() -> str:
    policy = (
        "uvloop"
        if type(asyncio.get_event_loop_policy()).__module__.startswith("uvloop")
        else "asyncio"
    )
    return policy


def install_uvloop_event_loop():
    if "uvloop" == _check_current_event_loop_policy():
        return

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        # else keep the standard asyncio loop as a fallback
        pass

    policy = _check_current_event_loop_policy()

    logger.debug(f"Using asyncio event-loop policy: {policy}")


def schedule_with_callback(coro, cb) -> Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(cb)
    return task


def get_normalized_version(version: str | None = None) -> str:
    """
    Return a public version string without local build metadata.

    Example:
    - 1.7.1+rhaiv.8 -> 1.7.1
    - 1.7.1 -> 1.7.1
    - 1.7.0.dev0 -> 1.7.0.dev0
    """
    resolved_version = version or __version__
    return resolved_version.split("+", 1)[0]


async def _defer_cancellation(
    operation: Awaitable[T], timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT
) -> T:
    """Wait for an operation to settle while deferring caller cancellation.

    Caller cancellation is suppressed during the timeout window. If timeout expires,
    the underlying operation is cancelled.

    If the operation completes during the timeout window its result is returned or
    any exception is raised. If timeout expires a ``TimeoutError`` is raised.
    """
    # wait_for requests coroutine cancellation only. Registries/pools retain
    # separate owners for preparation or worker updates that may outlive it.
    operation = asyncio.wait_for(operation, timeout)
    task = asyncio.ensure_future(operation)
    caller = asyncio.current_task()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # This helper intentionally defers caller cancellation. Catching
            # CancelledError does not clear asyncio's recorded requests, so
            # clear every pending request before the operation resumes. A
            # task can receive one CancelledError after multiple cancel() calls.
            if (
                caller is not None
                and hasattr(caller, "cancelling")
                and hasattr(caller, "uncancel")
            ):
                while caller.cancelling():
                    caller.uncancel()
            continue
    return task.result()


def create_task_independent(operation: Coroutine[Any, Any, T]) -> Task[T]:
    """Spawn owned background work with its own lifecycle timeout boundary.

    Other context (logging, tracing, etc.) is preserved. The caller must keep
    the task and observe its result. Awaited nested calls still share deadlines.
    """
    context = copy_context()
    context.run(_deferred_cancellation_active.set, False)
    return context.run(asyncio.create_task, operation)


def defer_cancellation(
    timeout: float | Callable[..., float] = DEFAULT_MODEL_OPERATION_TIMEOUT,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Apply one timeout and deferred-cancellation boundary to an operation.

    Nested decorated calls reuse the outer boundary and timeout.
    ``timeout`` may be a value or a callable resolved at invocation time.
    """

    def decorate(
        operation_method: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(operation_method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            # Only the outermost operation owns the timeout and deferred
            # cancellation. Nested decorated methods run inside that boundary.
            if _deferred_cancellation_active.get():
                return await operation_method(*args, **kwargs)

            operation_timeout = timeout
            if callable(operation_timeout):
                operation_timeout = operation_timeout(*args, **kwargs)

            active_token = _deferred_cancellation_active.set(True)
            try:
                return await _defer_cancellation(
                    operation_method(*args, **kwargs), operation_timeout
                )
            finally:
                _deferred_cancellation_active.reset(active_token)

        return wrapped

    return decorate


def with_operation_lock(
    lock_for: Callable[..., asyncio.Lock | Awaitable[asyncio.Lock]],
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Acquire an operation lock before running the decorated method.

    ``lock_for`` may return a lock directly or an awaitable.
    """

    def decorate(
        method: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            lock = lock_for(*args, **kwargs)
            if inspect.isawaitable(lock):
                lock = await lock
            async with lock:
                return await method(*args, **kwargs)

        return wrapped

    return decorate


def set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    """Ask the kernel to signal this process if its parent dies.

    Used by supervisor-spawned frontends and inference workers so that a
    crashed supervisor does not leave orphaned processes holding ports,
    sockets or GPU memory. On kernels without PR_SET_PDEATHSIG (or
    non-Linux platforms) this is a no-op. Best effort: if the parent
    already died before the call, the process exits immediately.

    Workers pass SIGKILL: they install no-op handlers for the
    termination signals (so the parent controls graceful shutdown), which
    would silently swallow a SIGTERM delivered by PR_SET_PDEATHSIG.
    SIGKILL cannot be caught or ignored, so parent death is always
    fatal for a worker.
    """
    import sys
    import ctypes

    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        # No getppid()-based parent-death check here: in containers the
        # main MLServer process IS PID 1, so a parent PID of 1 is the
        # normal case there, not a sign the parent died. PR_SET_PDEATHSIG
        # is the parent-death mechanism: the kernel delivers SIGTERM to
        # this process the moment its actual parent exits.
    except Exception:
        logger.debug("PR_SET_PDEATHSIG unavailable", exc_info=True)


def _reuse_port_sockets(host: str, port: int) -> "list[socket.socket]":
    """Bind TCP sockets with SO_REUSEPORT for uvicorn.

    uvicorn.Config has no ``reuse_port`` option; passing pre-bound
    sockets lets several frontend processes share the ports with
    kernel-level connection balancing. Every resolved address that can
    be bound gets its own listener — matching the single-frontend path
    (asyncio's create_server), which serves all of a hostname's
    addresses, not just the first.
    """
    import socket as socket_module

    # proto=IPPROTO_TCP is required: asyncio's _set_nodelay only fires for
    # TCP-proto sockets, and accepted sockets inherit the listener's
    # proto. With the default proto=0, accepted connections keep Nagle
    # enabled and small keep-alive writes stall ~40ms on the peer's
    # delayed ACK.
    # Resolve the family from the hostname (not a colon check): a
    # hostname with only an AAAA record must get an IPv6 socket, or
    # bind() fails during resolution where the single-frontend listener
    # (uvicorn resolves all families) would have started fine.
    addrinfo = socket_module.getaddrinfo(host, port, type=socket_module.SOCK_STREAM)
    # Mirror asyncio's create_server: try every resolved address in
    # turn, so a hostname whose first address cannot bind (but whose
    # second can) starts with frontend_workers=1 AND >1. The whole
    # per-address setup — socket creation (e.g. EAFNOSUPPORT for an
    # IPv6 address on a host without IPv6), options, bind and listen —
    # sits inside the failure boundary; a failed candidate is closed
    # before the next is tried.
    bound: list[socket.socket] = []
    last_error: OSError | None = None
    for family, _type, _proto, _canon, sockaddr in addrinfo:
        sock = None
        try:
            sock = socket_module.socket(
                family,
                socket_module.SOCK_STREAM,
                proto=socket_module.IPPROTO_TCP,
            )
            sock.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEPORT, 1)
            sock.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            # Match uvicorn's default accept backlog (2048): a smaller
            # queue would undercut the server's own listener settings
            # under connection storms.
            sock.listen(2048)
            bound.append(sock)
        except OSError as exc:
            last_error = exc
            if sock is not None:
                sock.close()
            continue
    if bound:
        return bound
    if last_error is not None:
        raise last_error
    raise OSError(f"no addresses resolved for {host}:{port}")
