from unittest.mock import Mock

from mlserver.settings import Settings


async def test_rest_stop_before_start_is_no_op():
    """stop() before start() (e.g. a failed startup unwinding) must not
    raise AttributeError on the missing server — it is teardown noise
    that hides real failures in the logs."""
    from mlserver.rest.server import RESTServer

    server = RESTServer(Settings(), Mock(), Mock())
    await server.stop()
