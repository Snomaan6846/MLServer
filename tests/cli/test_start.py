import asyncio
import os
import pytest
import signal
import sys

from aiohttp.client_exceptions import ClientResponseError
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from typing import Tuple

from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest

from ..utils import (
    RESTClient,
    TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
    get_available_ports,
)
from .test_start_cases import case_custom_module, case_sum_model


def _spawn_mlserver(folder: str) -> Popen:
    # Use the same interpreter as the running test env so imports resolve
    # consistently across tox and local runs.
    # This fixture depends on repository-root `conftest.py` bootstrap setup,
    # which pre-populates PYTHONPATH and trusted-runtime artifact env for
    # spawned subprocesses.
    repo_root = str(Path(__file__).resolve().parents[2])
    subprocess_env = {
        key: value for key, value in os.environ.items() if key != "PYTHONHOME"
    }
    if "PYTHONPATH" not in subprocess_env:
        raise RuntimeError("Missing PYTHONPATH test bootstrap env.")
    if TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV not in subprocess_env:
        raise RuntimeError("Missing trusted-runtimes test artifact env.")
    return Popen(
        [sys.executable, "-m", "mlserver.cli.main", "start", folder],
        cwd=repo_root,
        start_new_session=True,
        env=subprocess_env,
    )


def _stop_mlserver(process: Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process may have already exited before fixture teardown runs.
        pass
    try:
        process.wait(timeout=10)
    except TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except TimeoutExpired:
            pass  # Give up; process is unkillable (zombie or kernel issue)


@pytest.fixture
def settings(settings: Settings, free_ports: Tuple[int, int]) -> Settings:
    http_port, grpc_port, metrics_port = free_ports

    settings.http_port = http_port
    settings.grpc_port = grpc_port
    settings.metrics_port = metrics_port

    return settings


@pytest.fixture
def mlserver_start_sum_model(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Baseline scenario: importable runtime (`tests.fixtures.SumModel`).
    sum_model_folder = case_sum_model(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(sum_model_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
def mlserver_start_custom_module(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Security scenario: model-folder module (`custom.SumModel`) should not load.
    custom_module_folder = case_custom_module(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(custom_module_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
async def rest_client(settings: Settings) -> RESTClient:
    http_server = f"127.0.0.1:{settings.http_port}"
    client = RESTClient(http_server)

    yield client

    await client.close()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_live(rest_client: RESTClient):
    await rest_client.wait_until_live()
    is_live = await rest_client.live()
    assert is_live

    # Assert that the server is live, but some models are still loading
    with pytest.raises(ClientResponseError):
        await rest_client.ready()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_infer(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    await rest_client.wait_until_model_ready(sum_model_settings.name)
    response = await rest_client.infer(sum_model_settings.name, inference_request)

    assert len(response.outputs) == 1


async def test_custom_module_fails_closed(
    mlserver_start_custom_module: Popen,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # Fail closed when runtime points to a non-allowlisted model-folder module.
    # This also proves spawned workers (parallel_workers=2) are in PRODUCTION mode
    # with the allowlist enforced. If worker bootstrap failed and workers degraded
    # to DEVELOPMENT mode, custom.SumModel would load successfully.
    # See: test_spawned_workers_load_allowlisted_runtime_via_bootstrap (positive test)
    await rest_client.wait_until_live()
    with pytest.raises(ClientResponseError):
        await rest_client.wait_until_model_ready(sum_model_settings.name)


async def test_custom_module_loads_in_development_mode(
    development_mode,
    mlserver_start_custom_module: Popen,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # In DEVELOPMENT mode, custom.SumModel should load successfully
    await rest_client.wait_until_live()
    await rest_client.wait_until_model_ready(sum_model_settings.name)


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_spawned_workers_load_allowlisted_runtime_via_bootstrap(
    settings: Settings,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # This verifies the trusted-runtime bootstrap is applied inside spawned
    # worker processes (parallel_workers > 0). Without that bootstrap, workers
    # would fall back to DEVELOPMENT mode and accept any runtime.
    #
    # Positive test: allowlisted runtime loads successfully in workers.
    # For proof that workers are in PRODUCTION mode (not DEVELOPMENT fallback),
    # see: test_custom_module_fails_closed (negative test - rejects non-allowlisted)
    assert settings.parallel_workers > 0
    await rest_client.wait_until_live()
    await rest_client.wait_until_model_ready(sum_model_settings.name)


async def test_concurrent_mlserver_start_spawns_workers(
    tmp_path: str,
    settings: Settings,
    sum_model_settings: ModelSettings,
):
    # Start multiple MLServer instances concurrently to stress worker spawn
    # and trusted-runtime bootstrap application under parallel startup.
    instances: list[tuple[Popen, RESTClient]] = []
    used_ports: set[int] = set()

    try:
        for idx in range(2):
            instance_settings = settings.model_copy(deep=True)
            instance_ports = get_available_ports(3)
            if used_ports.intersection(instance_ports):
                raise AssertionError(
                    "Concurrent test allocated duplicate ports across instances."
                )
            used_ports.update(instance_ports)
            instance_settings.http_port = instance_ports[0]
            instance_settings.grpc_port = instance_ports[1]
            instance_settings.metrics_port = instance_ports[2]

            folder = os.path.join(str(tmp_path), f"instance-{idx}")
            os.makedirs(folder, exist_ok=True)
            case_sum_model(folder, instance_settings, sum_model_settings)

            process = _spawn_mlserver(folder)
            client = RESTClient(f"127.0.0.1:{instance_settings.http_port}")
            instances.append((process, client))

        await asyncio.gather(*[client.wait_until_live() for _, client in instances])
        await asyncio.gather(
            *[
                client.wait_until_model_ready(sum_model_settings.name)
                for _, client in instances
            ]
        )
    finally:
        await asyncio.gather(*[client.close() for _, client in instances])
        for process, _ in instances:
            _stop_mlserver(process)


def test_server_startup_aborts_with_corrupted_allowlist(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings, monkeypatch
):
    """
    Test that server.start() aborts when trusted-runtimes.json is corrupted.
    Verifies that system-level failures (corrupted artifact) cause immediate
    server shutdown rather than just logging errors.
    """
    # Create corrupted trusted-runtimes.json artifact
    corrupted_artifact = tmp_path / "corrupted-runtimes.json"
    corrupted_artifact.write_text("{invalid-json", encoding="utf-8")

    # Override artifact path via environment variable
    monkeypatch.setenv(TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV, str(corrupted_artifact))

    # Create model folder
    folder = case_sum_model(tmp_path, settings, sum_model_settings)

    # Spawn MLServer (should fail to start)
    process = _spawn_mlserver(folder)

    try:
        # Wait for process to exit (should fail fast during server.start())
        exit_code = process.wait(timeout=10)
        # Server should exit with non-zero code due to corrupted allowlist
        assert (
            exit_code != 0
        ), "Server should have failed to start with corrupted allowlist"
    except TimeoutExpired:
        _stop_mlserver(process)
        pytest.fail("Server did not exit within timeout - should have failed fast")
    finally:
        _stop_mlserver(process)
