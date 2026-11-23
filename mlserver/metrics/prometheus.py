import asyncio
import glob
import os

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    values,
)
from prometheus_client.multiprocess import MultiProcessCollector, mark_process_dead
from fastapi import Request, Response, status

from ..settings import Settings

PROMETHEUS_MULTIPROC_DIR = "PROMETHEUS_MULTIPROC_DIR"


def configure_metrics(settings: Settings):
    if not settings.parallel_workers:
        return

    # Only configure prometheus multiprocess if metrics are enabled
    if not settings.metrics_endpoint:
        return

    # Re-set Prometheus' Value class to use the multiproc version.
    # Note that this workaround depends on initialising all metrics in a
    # lazy manner (i.e. not as global values)
    # https://github.com/prometheus/client_python/blob/781e3e1851d80a53732bb8102d5754cf9d68b3c1/prometheus_client/values.py#L126-L134
    os.makedirs(settings.metrics_dir, exist_ok=True)
    os.environ[PROMETHEUS_MULTIPROC_DIR] = settings.metrics_dir
    values.ValueClass = values.get_value_class()


def remove_process_metric_files(settings: Settings, pid: int) -> None:
    """Remove every multiprocess metric file of a dead process.

    ``prometheus_client``'s ``mark_process_dead`` only prunes live-gauge
    files — counters, histograms and plain gauges of dead processes are
    left on disk by design (its README assumes external cleanup).
    Without the full ``*_{pid}.db`` removal, every process death leaks
    its metric files into the metrics dir AND keeps the dead process's
    stale series merged into every scrape forever.
    """
    # mark_process_dead reads PROMETHEUS_MULTIPROC_DIR for its path.
    if os.environ.get(PROMETHEUS_MULTIPROC_DIR) is None:
        os.environ.setdefault(PROMETHEUS_MULTIPROC_DIR, settings.metrics_dir)
    mark_process_dead(pid)
    pattern = os.path.join(settings.metrics_dir, f"*_{pid}.db")
    # File may not exist if the process was cancelled during
    # initialization or already cleaned up by another process
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


async def stop_metrics(settings: Settings, pid: int):
    if not settings.parallel_workers:
        return
    if not os.environ.get(PROMETHEUS_MULTIPROC_DIR):
        # Multiprocess metrics were never configured in this process
        # (or were already torn down): there is no directory to mark
        # the dead process in. Tolerated so a late worker-stop hook
        # cannot crash the shutting-down process.
        return

    # Filesystem I/O off the event loop: rapid worker cycling (a
    # recovery event cycling 8 workers) runs this per stop — the
    # glob+remove must not block inference serving.
    await asyncio.get_event_loop().run_in_executor(
        None, remove_process_metric_files, settings, pid
    )


class PrometheusEndpoint:
    def __init__(self, settings: Settings):
        self._settings = settings
        configure_metrics(self._settings)

    @property
    def _registry(self) -> CollectorRegistry:
        if not self._settings.parallel_workers:
            return REGISTRY

        registry = CollectorRegistry()
        MultiProcessCollector(registry)

        return registry

    def handle_metrics(self, req: Request) -> Response:
        headers = {"Content-Type": CONTENT_TYPE_LATEST}
        return Response(
            generate_latest(self._registry),
            status_code=status.HTTP_200_OK,
            headers=headers,
        )
