import logging


class HealthEndpointFilter(logging.Filter):
    """
    Filter to suppress successful health-probe access log lines.

    Only GET requests to health/readiness paths with a 2xx status are
    suppressed; failed probes (non-2xx) are always logged.

    From:
        https://github.com/encode/starlette/issues/864#issuecomment-653076434
    """

    @staticmethod
    def _parse_status_code(args: tuple) -> int | None:
        if len(args) < 5:
            return None
        status = args[4]
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)
        return None

    @staticmethod
    def _is_health_probe_path(path: str) -> bool:
        if path in ("/v2/health/live", "/v2/health/ready"):
            return True
        return path.endswith("/ready") and path.startswith("/v2/models/")

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.args, tuple):
            return True

        if len(record.args) < 3:
            return True

        request_method = record.args[1]
        query_string = record.args[2]
        if isinstance(request_method, bytes):
            request_method = request_method.decode("latin-1")
        if isinstance(query_string, bytes):
            query_string = query_string.decode("latin-1")
        if not isinstance(request_method, str) or not isinstance(query_string, str):
            return True
        if request_method != "GET":
            return True

        if not self._is_health_probe_path(query_string):
            return True

        status = self._parse_status_code(record.args)
        if status is not None and not (200 <= status < 300):
            return True

        return False


def disable_health_access_logs() -> None:
    uvicorn_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(f, HealthEndpointFilter) for f in uvicorn_logger.filters):
        return
    uvicorn_logger.addFilter(HealthEndpointFilter())


def clear_uvicorn_access_log_filters() -> None:
    """Clear all uvicorn.access filters on REST server shutdown.

    Any filters from logging_settings are re-applied on the next server
    start via dictConfig.
    """
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.filters.clear()


loggerName = "mlserver.rest"
logger = logging.getLogger(loggerName)
