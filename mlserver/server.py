import asyncio
import signal

from mlserver.repository.factory import ModelRepositoryFactory

from .model import MLModel
from .settings import Settings, ModelSettings, log_runtime_security_mode
from .logging import configure_logger
from .registry import MultiModelRegistry
from .handlers import DataPlane, ModelRepositoryHandlers
from .parallel import InferencePoolRegistry
from .batching import load_batching, unload_batching, warn_batching_misconfigured
from .rest import RESTServer
from .grpc import GRPCServer
from .metrics import MetricsServer
from .utils import logger

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]


class MLServer:
    def __init__(self, settings: Settings, inference_pool_registry=None):
        self._settings = settings
        self._live: bool = False
        self._stop_task: asyncio.Task | None = None

        # Response payload topology: only the multi-frontend opt-in
        # keeps NumPy payloads from the codec. Single-frontend (the
        # default, and in-process models generally) materialise lists
        # at the codec — master's exact data flow.
        from .codecs.numpy import set_ndarray_response_pipeline

        set_ndarray_response_pipeline(settings.frontend_workers > 1)

        if settings.frontend_workers > 1:
            # Only the CLI entry point spawns the supervisor; a
            # programmatic MLServer(settings) embed would otherwise run
            # single-process with the setting silently ignored.
            logger.warning(
                "frontend_workers=%s requires the 'mlserver start' CLI "
                "entry point, which spawns the frontend supervisor; a "
                "programmatically constructed MLServer runs "
                "single-process regardless of this setting.",
                settings.frontend_workers,
            )
        self._startup_model_tasks: list[asyncio.Task] = []
        self._server_tasks: list[asyncio.Task] = []
        self._add_signal_handlers()

        self._metrics_server = None
        if self._settings.metrics_endpoint:
            self._metrics_server = MetricsServer(self._settings)

        self._inference_pool_registry = None
        if inference_pool_registry is not None:
            # Shared-pool topology: the frontend dispatches to workers owned
            # by the supervisor's pool (see mlserver/frontend.py).
            self._inference_pool_registry = inference_pool_registry
        elif self._settings.parallel_workers:
            # Only load inference pool if parallel inference has been enabled
            on_worker_stop = []
            if self._metrics_server:
                on_worker_stop = [self._metrics_server.on_worker_stop]

            self._inference_pool_registry = InferencePoolRegistry(
                self._settings,
                on_worker_stop=on_worker_stop,
            )

        self._model_registry = self._create_model_registry()
        self._model_repository = ModelRepositoryFactory.resolve_model_repository(
            self._settings
        )
        self._data_plane = DataPlane(
            settings=self._settings, model_registry=self._model_registry
        )
        self._model_repository_handlers = ModelRepositoryHandlers(
            repository=self._model_repository,
            model_registry=self._model_registry,
            model_operation_timeout=self._settings.model_operation_timeout,
        )

        self._configure_logger()
        self._create_servers()

    def _create_model_registry(self) -> MultiModelRegistry:
        on_model_load = [
            self.add_custom_handlers,
            load_batching,
        ]
        on_model_unload = [
            unload_batching,
            self.remove_custom_handlers,
        ]
        if not self._inference_pool_registry:
            return MultiModelRegistry(
                on_model_load=on_model_load,
                on_model_unload=on_model_unload,
                model_operation_timeout=self._settings.model_operation_timeout,
            )

        # The adaptive batcher wraps the model at its point of execution.
        # At frontend_workers > 1 the worker's registry installs it (one
        # batcher in the shared worker sees every frontend's traffic — a
        # per-frontend batcher would only see 1/N of it). At
        # frontend_workers == 1 (the default) the batcher stays in the
        # MAIN process, wrapping the parallel wrapper exactly as before
        # the multi-frontend change: single-frontend deployments keep
        # their previous behaviour. Models that opt out of parallel
        # inference (model-level parallel_workers=0) always run
        # in-process and always get the main-process batcher.
        batching_in_main = self._settings.frontend_workers == 1

        async def _pool_load(model):
            loaded = await self._inference_pool_registry.load_model(model)
            batched = (
                model.settings.max_batch_size > 1 and model.settings.max_batch_time > 0
            )
            # The pooled paths below skip load_batching for models it
            # would not wrap; its misconfiguration warning must still
            # fire, or pooled deployments lose the diagnosis.
            warn_batching_misconfigured(model)
            if loaded is model:
                # Pool returned the model unchanged (no parallel inference
                # for this model); apply batching in-process.
                if batched:
                    return await load_batching(model)
                return loaded
            if batched and batching_in_main:
                # Single-frontend topology: wrap the parallel wrapper in
                # the main process (the previous behaviour); the worker
                # skipped its own batcher install for this topology.
                return await load_batching(loaded)
            return loaded

        on_model_load = [
            _pool_load,
            self.add_custom_handlers,
        ]
        if batching_in_main:
            # Teardown while the wrapped instance is still in hand, then
            # the standard chain (a no-op for models without a batcher).
            on_model_unload = [
                unload_batching,
                self.remove_custom_handlers,
                self._inference_pool_registry.unload_model,
            ]
        else:
            on_model_unload = [
                self.remove_custom_handlers,
                self._inference_pool_registry.unload_model,
            ]
        return MultiModelRegistry(
            on_model_load=on_model_load,
            on_model_unload=on_model_unload,
            model_initialiser=self._inference_pool_registry.model_initialiser,
            model_operation_timeout=self._settings.model_operation_timeout,
        )

    def _configure_logger(self):
        self._logger = configure_logger(self._settings)

    def _create_servers(self):
        self._rest_server = RESTServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )
        self._grpc_server = GRPCServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )

    async def start(self, models_settings: list[ModelSettings] = []):
        # Validate runtime security configuration before starting servers to prevent
        # a window where endpoints are accessible but security hasn't been verified
        if self._stop_task is not None:
            logger.info("Server shutdown currently in progress...")
            return
        if self._live:
            logger.info("Server already running...")
            return
        self._live = True

        try:
            log_runtime_security_mode()
        except Exception as exc:
            self._live = False
            logger.exception("Failed to load trusted runtimes allowlist!")
            if self._inference_pool_registry:
                try:
                    await self._inference_pool_registry.close()
                except Exception:
                    logger.exception(
                        "Failed to close inference pool after startup validation error"
                    )
            raise RuntimeError(
                "Server startup aborted: "
                "invalid trusted runtimes allowlist configuration"
            ) from exc

        # Start servers and load startup models concurrently
        try:
            if self._inference_pool_registry:
                await self._inference_pool_registry.start()
            servers = [self._rest_server.start(), self._grpc_server.start()]
            if self._metrics_server:
                servers.append(self._metrics_server.start())

            for server in servers:
                self._server_tasks.append(asyncio.create_task(server))
            for model_settings in models_settings:
                self._startup_model_tasks.append(
                    asyncio.create_task(self._model_registry.load(model_settings))
                )
            await asyncio.gather(*self._startup_model_tasks)
            self._startup_model_tasks.clear()

            # Mark startup complete only if all models loaded successfully.
            # With a shared pool, defer until the supervisor confirms all
            # startup models have been applied (StartupComplete message);
            # the SharedPoolRegistry calls startup_complete on receipt.
            if (
                hasattr(self._inference_pool_registry, "startup_models_pending")
                and self._inference_pool_registry.startup_models_pending
            ):
                logger.info(
                    "Startup models pending broadcast; deferring " "startup_complete"
                )
                startup_models_pending = True
            else:
                self._data_plane.startup_complete()
                startup_models_pending = False

            if (
                not models_settings
                and not self._settings.empty_registry_readiness
                and not startup_models_pending
            ):
                # A frontend in the shared-pool topology always starts
                # with an empty registry (models arrive by broadcast):
                # the warning only applies when nothing is coming.
                logger.warning(
                    "No models were loaded at startup and empty_registry_readiness "
                    "is False, so the server will report not ready until a model is "
                    "loaded. If models were expected, check that "
                    "load_models_at_startup is enabled, that the model repository "
                    "contains models, and the logs above for any model settings that "
                    "failed to load. To load models dynamically into an empty "
                    "server, set empty_registry_readiness to True."
                )

            # Then await the server tasks throughout the lifetime of MLServer
            await asyncio.gather(*self._server_tasks)
        except (Exception, asyncio.CancelledError) as start_error:
            if isinstance(start_error, asyncio.CancelledError):
                logger.info("Server startup was cancelled. Shutting down...")
            else:
                logger.exception(
                    "A failure occurred during server startup. Shutting down..."
                )
            try:
                if self._live:
                    await self.stop()
                elif self._stop_task is not None:
                    await self._stop_task
            except (Exception, asyncio.CancelledError):
                logger.error("Failed while shutting down after startup failure")
            raise
        finally:
            self._startup_model_tasks.clear()
            self._server_tasks.clear()

    async def add_custom_handlers(self, model: MLModel) -> MLModel:
        await self._rest_server.add_custom_handlers(model)

        # TODO: Add support for custom gRPC endpoints
        # self._grpc_server.add_custom_handlers(handlers)

        return model

    async def remove_custom_handlers(self, model: MLModel) -> MLModel:
        await self._rest_server.delete_custom_handlers(model)

        # TODO: Add support for custom gRPC endpoints
        # self._grpc_server.delete_custom_handlers(handlers)

        return model

    def _add_signal_handlers(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running yet, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for sig in HANDLED_SIGNALS:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.stop(sig=s))
            )

    async def stop(self, sig: int | None = None):
        """Request shutdown, cancelling startup if it is still running."""
        self._live = False
        if self._stop_task is not None:
            logger.info("Server shutdown already in progress...")
            return
        try:
            self._stop_task = asyncio.current_task()
            # Cancel startup model load tasks
            for model_task in self._startup_model_tasks:
                if not model_task.done():
                    model_task.cancel()
            await asyncio.gather(*self._startup_model_tasks, return_exceptions=True)

            try:
                await self._stop_resources(sig)
            except Exception:
                for server_task in self._server_tasks:
                    if not server_task.done():
                        server_task.cancel()
            await asyncio.gather(*self._server_tasks, return_exceptions=True)
        finally:
            self._stop_task = None

    async def _stop_resources(self, sig: int | None = None):
        """Stop transports and shared resources after shutdown is requested."""
        # Best effort cleanup
        stop_errors = []

        if self._grpc_server:
            try:
                await self._grpc_server.stop(sig)
            except Exception as e:
                logger.error("Failed to stop gRPC server", exc_info=True)
                stop_errors.append(e)

        if self._rest_server:
            try:
                await self._rest_server.stop(sig)
            except Exception as e:
                logger.error("Failed to stop REST server", exc_info=True)
                stop_errors.append(e)

        if self._inference_pool_registry:
            try:
                await self._inference_pool_registry.close()
            except Exception as e:
                logger.error("Failed to close inference pool registry", exc_info=True)
                stop_errors.append(e)

        if self._metrics_server:
            try:
                await self._metrics_server.stop(sig)
            except Exception as e:
                logger.error("Failed to stop metrics server", exc_info=True)
                stop_errors.append(e)

        if stop_errors:
            raise stop_errors[0]
