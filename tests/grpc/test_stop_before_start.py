from unittest.mock import Mock

from mlserver.settings import Settings


async def test_grpc_stop_before_start_is_no_op():
    """stop() before start() must not raise AttributeError on the
    missing server."""
    from mlserver.grpc.server import GRPCServer

    server = GRPCServer(Settings(), Mock(), Mock())
    await server.stop()
