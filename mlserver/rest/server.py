import asyncio
import socket
import uvicorn
import signal

from collections.abc import MutableSequence

from ..settings import Settings
from ..utils import _reuse_port_sockets
from ..handlers import DataPlane, ModelRepositoryHandlers, get_custom_handlers
from ..model import MLModel

from .utils import matches
from .app import create_app
from .logging import (
    logger,
    disable_health_access_logs,
    clear_uvicorn_access_log_filters,
)

# A frontend with more connections than a peer leaves a pending
# connection to the less loaded one, rechecking every 1 ms, for up to
# 5 ms (it may be busy).
_ACCEPT_DEFER_CHECKS = 5
# The load of a frontend that is not accepting (also the initial value
# of the shared array), so it is never the least loaded.
_NOT_ACCEPTING = 2**31 - 1


async def _readable(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
    """Wait until the (non-blocking) socket has a pending connection."""
    ready: "asyncio.Future[None]" = loop.create_future()

    def wake() -> None:
        if not ready.done():
            ready.set_result(None)

    loop.add_reader(sock, wake)
    try:
        await ready
    finally:
        loop.remove_reader(sock)


class _NoSignalServer(uvicorn.Server):
    """Uvicorn server that never installs its own signal handlers.

    With peer_loads (several frontends on one shared socket: a shared
    array with one slot per frontend, and this frontend's index), each
    frontend publishes its number of open connections, and a pending
    connection is accepted by the least loaded frontend. The event
    loop's own server would accept every queued connection in
    whichever frontend wakes first (the pattern proven in production
    by vLLM's NoSignalServer).
    """

    def __init__(
        self,
        config: uvicorn.Config,
        peer_loads: "tuple[MutableSequence[int], int] | None" = None,
    ):
        super().__init__(config)
        self._peer_loads = peer_loads
        self._accept_tasks: "set[asyncio.Task]" = set()
        self._connecting = 0

    def install_signal_handlers(self):
        pass

    def _publish_load(self) -> None:
        assert self._peer_loads is not None
        loads, index = self._peer_loads
        # Clamped below the _NOT_ACCEPTING sentinel: the shared array
        # is signed 32-bit — an unclamped sum reaching the sentinel
        # would alias it (starving the overloaded frontend), and one
        # beyond it would wrap negative (making it look minimally
        # loaded). The bound is unreachable in practice (2^31-2
        # connections) but the clamp makes it structural.
        loads[index] = min(
            len(self.server_state.connections) + self._connecting,
            _NOT_ACCEPTING - 1,
        )

    def _least_loaded(self) -> bool:
        assert self._peer_loads is not None
        loads, index = self._peer_loads
        return loads[index] <= min(loads)

    async def startup(self, sockets: "list[socket.socket] | None" = None) -> None:
        if not (self._peer_loads and sockets):
            return await super().startup(sockets)
        # With no sockets uvicorn starts no asyncio server; _accept
        # serves them with uvicorn's protocol instead.
        await super().startup(sockets=[])
        if not self.started:
            # Lifespan startup failed.
            return
        self._publish_load()
        for sock in sockets:
            # The supervisor prepared the shared socket fully: listening
            # (backlog 2048) and non-blocking. Every frontend dups the
            # same open file description — re-calling listen() or
            # setblocking() here would be redundant, and setblocking
            # would mutate the shared description once per frontend.
            self._track(self._accept(sock))

    def _track(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._accept_tasks.add(task)
        task.add_done_callback(self._accept_tasks.discard)

    async def _accept(self, sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await _readable(loop, sock)
            # Connections closed since the last accept; peers refresh likewise.
            self._publish_load()
            for _ in range(_ACCEPT_DEFER_CHECKS):
                if self._least_loaded():
                    break
                await asyncio.sleep(0.001)
            try:
                conn, _ = sock.accept()
            except (BlockingIOError, InterruptedError, ConnectionAbortedError):
                continue  # another frontend took it
            except OSError:
                # e.g. EMFILE; asyncio's own server also retries after 1 s.
                logger.exception("Error accepting a connection")
                await asyncio.sleep(1)
                continue
            self._connecting += 1
            self._publish_load()
            # Its own task, so a slow handshake does not hold up the
            # next accept.
            self._track(self._connect(loop, conn))

    async def _connect(
        self, loop: asyncio.AbstractEventLoop, conn: socket.socket
    ) -> None:
        config = self.config
        try:
            await loop.connect_accepted_socket(
                lambda: config.http_protocol_class(  # type: ignore[call-arg]
                    config=config,
                    server_state=self.server_state,
                    app_state=self.lifespan.state,
                    _loop=loop,
                ),
                conn,
                ssl=config.ssl,
            )
        except asyncio.CancelledError:
            conn.close()
            raise
        except Exception:
            logger.debug("Accepted connection failed to start", exc_info=True)
            conn.close()
        finally:
            self._connecting -= 1
            self._publish_load()

    async def shutdown(self, sockets: "list[socket.socket] | None" = None) -> None:
        # Before super() closes the listening sockets.
        tasks = list(self._accept_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._peer_loads is not None:
            loads, index = self._peer_loads
            loads[index] = _NOT_ACCEPTING
        await super().shutdown(sockets)


class RESTServer:
    def __init__(
        self,
        settings: Settings,
        data_plane: DataPlane,
        model_repository_handlers: ModelRepositoryHandlers,
        shared_sockets: "list[socket.socket] | None" = None,
        peer_loads: "tuple[MutableSequence[int], int] | None" = None,
    ):
        self._settings = settings
        self._data_plane = data_plane
        self._model_repository_handlers = model_repository_handlers
        # The supervisor's shared listener (one socket set + the peer
        # load array) in the supervised topology; None when this server
        # owns its socket (direct mode, or an unsupervised embedder).
        # _get_config() falls back to per-process SO_REUSEPORT sockets
        # only when no shared listener was provided.
        self._shared_port_sockets = shared_sockets
        self._peer_loads = peer_loads
        self._app = create_app(
            self._settings,
            data_plane=self._data_plane,
            model_repository_handlers=self._model_repository_handlers,
        )

    async def add_custom_handlers(self, model: MLModel) -> MLModel:
        handlers = get_custom_handlers(model)
        if not handlers:
            return model
        try:
            for custom_handler, handler_method in handlers:
                # Idempotent per instance: a superseded origin's hook
                # chain re-runs on the wrapper an apply already routed
                # (the stale-publication swap); adding again would
                # double the routes and leak one on removal (each
                # handler removes exactly one matching route).
                if any(
                    matches(route, custom_handler, handler_method)  # type: ignore
                    for route in self._app.routes
                ):
                    continue
                self._app.add_api_route(
                    custom_handler.rest_path,
                    handler_method,
                    methods=[custom_handler.rest_method],
                )
        except Exception:
            self._delete_custom_handlers(model)
            raise

        previous = model._on_load_failed

        def discard() -> None:
            # Routes call the instance directly, bypassing registry lookup.
            # Revoke them before waiting for abandoned native preparation;
            # retaining resource ownership must not retain serving eligibility.
            try:
                self._delete_custom_handlers(model)
            finally:
                previous()

        model._on_load_failed = discard

        return model

    async def delete_custom_handlers(self, model: MLModel) -> MLModel:
        self._delete_custom_handlers(model)
        return model

    def _delete_custom_handlers(self, model: MLModel) -> None:
        handlers = get_custom_handlers(model)
        if len(handlers) == 0:
            return

        # NOTE: Loop in reverse, so that it's quicker to find all the recently
        # added routes and we can remove routes on-the-fly
        for i, route in reversed(list(enumerate(self._app.routes))):
            for j, (custom_handler, handler_method) in enumerate(handlers):
                if matches(route, custom_handler, handler_method):  # type: ignore
                    self._app.routes.pop(i)
                    handlers.pop(j)

    async def start(self):
        cfg = self._get_config()
        self._server = _NoSignalServer(cfg, peer_loads=self._peer_loads)
        if self._settings.access_log and not self._settings.debug:
            disable_health_access_logs()

        logger.info(
            "HTTP server running on "
            f"http://{self._settings.host}:{self._settings.http_port}"
        )

        if self._shared_port_sockets is not None:
            await self._server.serve(sockets=self._shared_port_sockets)
        else:
            await self._server.serve()

    def _get_config(self):
        kwargs = {}

        if self._settings._custom_rest_server_settings:
            logger.warning(
                "REST custom configuration is out of support. Use as your own risk"
            )
            kwargs.update(self._settings._custom_rest_server_settings)

        kwargs.update(
            {
                "host": self._settings.host,
                "port": self._settings.http_port,
                "root_path": self._settings.root_path,
                "access_log": self._settings.access_log,
            }
        )

        if self._shared_port_sockets is None and self._settings.frontend_workers >= 1:
            # Kernel-level connection balancing across frontend processes
            # — the fallback for a frontend that owns its socket (an
            # unsupervised embedder with frontend_workers set). In the
            # supervised topology the supervisor's shared listener is
            # provided at construction and used instead.
            # uvicorn has no reuse_port config: the supported pattern (its
            # own gunicorn-worker path) is pre-bound sockets handed to
            # Server.serve(sockets=[...]). Every resolved address gets a
            # listener, matching the direct-server path.
            self._shared_port_sockets = _reuse_port_sockets(
                self._settings.host, self._settings.http_port
            )

        if self._settings.logging_settings:
            # If not None, use ours. Otherwise, let Uvicorn fall back on its
            # own config.
            kwargs.update({"log_config": self._settings.logging_settings})

        return uvicorn.Config(self._app, **kwargs)

    async def stop(self, sig: int | None = None):
        # stop() before start() (e.g. a failed startup unwinding) has
        # no server to shut down; nothing to do.
        if getattr(self, "_server", None) is None:
            return
        if sig is None:
            # `sig` is no longer optional for `handle_exit` in
            # latest `uvicorn`
            sig = signal.SIGINT

        self._server.handle_exit(sig=sig, frame=None)
        # handle_exit returns immediately; drain runs in serve(). Do NOT
        # close the shared socket here: graceful shutdown closes every
        # passed socket exactly once when serve() returns — in the
        # shared-listener topology that closes only THIS process's fd
        # dup (the supervisor and the other frontends keep the socket
        # open), and in the SO_REUSEPORT fallback it closes this
        # process's own socket. Closing it in this handler would close
        # it a second time — and the fd number can be reused in
        # between, so that second close could hit an unrelated
        # resource.
        clear_uvicorn_access_log_filters()
