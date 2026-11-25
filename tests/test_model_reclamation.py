"""Model reclamation after unload.

gc.freeze() excludes objects from cyclic collection, so it must never
cover model instances: models participate in reference cycles through
ordinary machinery (e.g. adaptive batching) that refcounting cannot
always break, and a frozen model could never be reclaimed after
unload. The subprocess isolates the process-wide GC state from the
test runner's.
"""

import asyncio
import gc
import re
import subprocess
import sys
import weakref
from pathlib import Path

from mlserver import MLServer, ModelSettings, Settings
from mlserver.model import MLModel

REPO_ROOT = Path(__file__).parent.parent


async def _wait_for_model(registry, name: str, timeout: float = 30.0) -> MLModel:
    waited = 0.0
    while waited < timeout:
        try:
            return await registry.get_model(name)
        except Exception:
            await asyncio.sleep(0.1)
            waited += 0.1
    raise TimeoutError(f"model {name} not loaded within {timeout}s")


def _settings() -> Settings:
    return Settings(
        host="127.0.0.1",
        http_port=19510,
        grpc_port=19511,
        metrics_port=19512,
        metrics_endpoint="",
        parallel_workers=0,
        load_models_at_startup=False,
        empty_registry_readiness=True,
    )


def _batched_settings() -> ModelSettings:
    # Adaptive batching installs the model<->batcher reference cycles
    # the reclaimer must be able to collect after unload.
    return ModelSettings(
        name="reclaim-model",
        implementation="tests.fixtures.SumModel",
        max_batch_size=4,
        max_batch_time=0.01,
        inputs=[
            {
                "name": "input-0",
                "datatype": "FP32",
                "shape": [1, 2],
                "parameters": {"content_type": "np"},
            }
        ],
    )


def _plain_settings() -> ModelSettings:
    return ModelSettings(
        name="reclaim-model",
        implementation="tests.fixtures.SumModel",
        inputs=[
            {
                "name": "input-0",
                "datatype": "FP32",
                "shape": [1, 2],
                "parameters": {"content_type": "np"},
            }
        ],
    )


async def _drive_to_reclaim(
    server: MLServer,
    model_settings: ModelSettings,
    startup: bool,
    freeze_before: int,
) -> int:
    """Start (or hand the already-started server) a model, unload it,
    and report whether its weak reference dies after collection."""
    if startup:
        start_task = asyncio.create_task(server.start([model_settings]))
    else:
        start_task = asyncio.create_task(server.start([]))
    try:
        if not startup:
            await _wait_for_empty_server(server)
            await server._model_registry.load(model_settings)
        model = await _wait_for_model(server._model_registry, "reclaim-model")

        if startup and model_settings.max_batch_size > 1:
            if not hasattr(model, "__adaptive_batching__"):
                print(
                    "INVALID: batcher not installed — test does not exercise cycles",
                    flush=True,
                )
                return 2

        ref = weakref.ref(model)
        await server._model_registry.unload("reclaim-model")
        del model
        # Settle the batcher's task cancellation and the unload hooks.
        for _ in range(3):
            await asyncio.sleep(0.05)
        gc.collect()
        await asyncio.sleep(0.05)
        gc.collect()

        # Python starts with ~hundreds of objects in the permanent
        # generation; OUR freeze moves the import heap (thousands of
        # objects) on top of that baseline.
        freeze_delta = gc.get_freeze_count() - freeze_before
        if ref() is not None:
            print(
                f"RETAINED (freeze_delta={freeze_delta}): model survives unload",
                flush=True,
            )
            return 1
        print(f"RECLAIMED (freeze_delta={freeze_delta})", flush=True)
        return 0
    finally:
        await server.stop()
        await start_task


async def _wait_for_empty_server(server: MLServer, timeout: float = 30.0) -> None:
    waited = 0.0
    while waited < timeout:
        if server._live:
            return
        await asyncio.sleep(0.1)
        waited += 0.1
    raise TimeoutError("server did not start")


def test_startup_model_reclaimed_after_unload():
    """A batched startup model must be reclaimable after unload.

    Regression: gc.freeze() ran after startup model loads, moving the
    models (and their batching cycles) into the GC's permanent
    generation — excluded from every future collection, so an
    unloaded startup model stayed resident.
    """
    _run_probe("startup-batched")


def test_plain_startup_model_reclaimed_after_unload():
    """A non-batched startup model (no cycle machinery) also reclaims."""
    _run_probe("startup-plain")


def test_runtime_loaded_model_reclaimed_after_unload():
    """A model loaded after startup (never eligible for the freeze)
    reclaims — pins the boundary of the freeze's scope."""
    _run_probe("runtime-loaded")


def _run_probe(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"model not reclaimed after unload (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-2000:]}"
    )
    assert "RECLAIMED" in result.stdout, result.stdout
    # The freeze must still happen (its scan-saving purpose); only its
    # placement changed. Guards against deleting the freeze as a fix:
    # our freeze moves the import heap — thousands of objects on top
    # of the interpreter's ~hundreds baseline.
    match = re.search(r"freeze_delta=(\d+)", result.stdout)
    assert match, result.stdout
    assert int(match.group(1)) > 1000, result.stdout


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    mode = sys.argv[1] if len(sys.argv) > 1 else "startup-batched"

    async def _probe() -> int:
        freeze_before = gc.get_freeze_count()
        server = MLServer(_settings())
        if mode == "startup-batched":
            return await _drive_to_reclaim(
                server, _batched_settings(), True, freeze_before
            )
        if mode == "startup-plain":
            return await _drive_to_reclaim(
                server, _plain_settings(), True, freeze_before
            )
        if mode == "runtime-loaded":
            return await _drive_to_reclaim(
                server, _batched_settings(), False, freeze_before
            )
        print(f"INVALID: unknown mode {mode}", flush=True)
        return 2

    sys.exit(asyncio.run(_probe()))
