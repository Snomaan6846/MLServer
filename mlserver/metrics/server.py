import socket
import uvicorn
import os
import signal

from typing import TYPE_CHECKING

from fastapi import FastAPI

from ..settings import Settings
from ..utils import _reuse_port_sockets
from .logging import logger
from .prometheus import PrometheusEndpoint, stop_metrics

if TYPE_CHECKING:
    from ..parallel import Worker


class _NoSignalServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass


class MetricsServer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._endpoint = PrometheusEndpoint(settings)
        # Pre-bound SO_REUSEPORT socket in the supervised topology
        # (frontend_workers >= 1).
        self._shared_port_sockets: "list[socket.socket] | None" = None
        self._app = self._get_app()

    def _get_app(self):
        app = FastAPI(debug=self._settings.debug)
        app.add_route(self._settings.metrics_endpoint, self._endpoint.handle_metrics)
        return app

    async def on_worker_stop(self, worker: "Worker") -> None:
        if not worker.pid:
            return

        await stop_metrics(self._settings, worker.pid)

    async def start(self):
        cfg = self._get_config()
        self._server = _NoSignalServer(cfg)

        metrics_server = f"http://{self._settings.host}:{self._settings.metrics_port}"
        logger.info(f"Metrics server running on {metrics_server}")
        logger.info(
            "Prometheus scraping endpoint can be accessed on "
            f"{metrics_server}{self._settings.metrics_endpoint}"
        )
        if self._shared_port_sockets is not None:
            await self._server.serve(sockets=self._shared_port_sockets)
        else:
            await self._server.serve()

    def _get_config(self):
        kwargs = {}

        if self._settings._custom_metrics_server_settings:
            logger.warning(
                "REST custom configuration is out of support. Use as your own risk"
            )
            kwargs.update(self._settings._custom_metrics_server_settings)

        kwargs.update(
            {
                "host": self._settings.host,
                "port": self._settings.metrics_port,
                "access_log": self._settings.access_log,
            }
        )

        if self._settings.frontend_workers >= 1:
            # Pre-bound SO_REUSEPORT sockets, handed to Server.serve()
            # below (uvicorn's supported pattern for shared ports).
            # Every resolved address gets a listener, matching the
            # direct-server path.
            self._shared_port_sockets = _reuse_port_sockets(
                self._settings.host, self._settings.metrics_port
            )

        if self._settings.logging_settings:
            # If not None, use ours. Otherwise, let Uvicorn fall back on its
            # own config.
            kwargs.update({"log_config": self._settings.logging_settings})

        return uvicorn.Config(self._app, **kwargs)

    async def stop(self, sig: int | None = None):
        if sig is None:
            # `sig` is no longer optional for `handle_exit` in
            # latest `uvicorn`
            sig = signal.SIGINT

        await stop_metrics(self._settings, os.getpid())
        self._server.handle_exit(sig=sig, frame=None)
        # Do NOT close the shared socket here: Uvicorn's graceful
        # shutdown closes the asyncio servers it built from the passed
        # sockets (the fd is closed exactly once when serve() returns).
        # Closing it in this handler would close it a second time — and
        # the fd number can be reused in between, so that second close
        # could hit an unrelated resource.
