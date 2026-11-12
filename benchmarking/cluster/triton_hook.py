#!/usr/bin/env python3
"""RHOAIENG-102092 Triton reference-measurement prepare hook.

Scales the MLServer deployments to zero (exclusive Tesla T4 use),
restarts the Triton deployment with a fresh process, waits for server
and model readiness, verifies the ONNX Runtime backend loaded on CUDA
from server logs, and checks response value parity against the MLServer
CUDA reference (tolerance: cross-build ONNX Runtime variance). Writes
receipts into BENCHMARK_TRIAL_DIR. Exit nonzero on any failure.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
import grpc
import numpy as np
from kubernetes import client as k8s
from kubernetes import config as k8s_config

NAMESPACE = os.environ.get("BENCHMARK_NAMESPACE", "mlserver-perf")
TRITON_DEPLOYMENT = "triton-rfdetr"
TRITON_SERVICE = "triton-rfdetr"
MLSERVER_DEPLOYMENTS = ("mlserver-rfdetr", "mlserver-rfdetr-cpu")
MODEL = "rfdetr-small"
CONTAINER = "triton"
HTTP_PORT = 8000
GRPC_PORT = 8001
STORE = Path("/store")
PAYLOADS = STORE / "payloads"
REFERENCES = STORE / "references"
RPC = "/inference.GRPCInferenceService/ModelInfer"
PARITY_TOLERANCE = 1e-3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for(predicate, timeout: float, description: str, interval: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s: {description}")


def selector_for(apps, name: str) -> str:
    dep = apps.read_namespaced_deployment(name, NAMESPACE)
    return ",".join(
        f"{key}={value}" for key, value in dep.spec.selector.match_labels.items()
    )


def running_pod(core, apps, name: str):
    pods = core.list_namespaced_pod(
        NAMESPACE, label_selector=selector_for(apps, name)
    ).items
    alive = [
        p
        for p in pods
        if p.status.phase == "Running" and p.metadata.deletion_timestamp is None
    ]
    return alive[0] if alive else None


def ensure_scaled_zero(apps, core, name: str) -> None:
    dep = apps.read_namespaced_deployment(name, NAMESPACE)
    if (dep.spec.replicas or 0) == 0 and not running_pod(core, apps, name):
        return
    apps.patch_namespaced_deployment(name, NAMESPACE, {"spec": {"replicas": 0}})

    def gone():
        dep = apps.read_namespaced_deployment(name, NAMESPACE)
        if (dep.spec.replicas or 0) != 0:
            return False
        return running_pod(core, apps, name) is None

    wait_for(gone, 240, f"{name} scaled to zero with no running pods")


def restart_triton(apps) -> None:
    apps.patch_namespaced_deployment(
        TRITON_DEPLOYMENT,
        NAMESPACE,
        {
            "spec": {
                "replicas": 1,
                "template": {
                    "metadata": {
                        "annotations": {"rhoaieng-102092/restarted-at": now_utc()}
                    }
                },
            }
        },
    )


def wait_rollout(apps, core):
    dep = apps.read_namespaced_deployment(TRITON_DEPLOYMENT, NAMESPACE)
    generation = dep.metadata.generation

    def rolled():
        current = apps.read_namespaced_deployment(TRITON_DEPLOYMENT, NAMESPACE)
        status = current.status
        if status.observed_generation != generation:
            return False
        if (status.ready_replicas or 0) != 1:
            return False
        pod = running_pod(core, apps, TRITON_DEPLOYMENT)
        return pod if pod is not None else False

    return wait_for(rolled, 1200, "triton rollout (image pull + model load)")


async def wait_ready() -> float:
    base = f"http://{TRITON_SERVICE}.{NAMESPACE}.svc:{HTTP_PORT}"
    began = time.monotonic()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        for path in (
            "/v2/health/ready",
            f"/v2/models/{MODEL}/ready",
        ):
            deadline = time.monotonic() + 300
            while True:
                try:
                    async with session.get(base + path) as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Triton not ready at " + path)
                await asyncio.sleep(2)
    return time.monotonic() - began


def backend_evidence(core, pod) -> list[str]:
    log = core.read_namespaced_pod_log(pod.metadata.name, NAMESPACE)
    hits = [
        line.strip()
        for line in log.splitlines()
        if "onnxruntime" in line.lower() or "cuda" in line.lower()
    ]
    if not any("onnxruntime" in line.lower() for line in hits):
        raise RuntimeError("No onnxruntime backend evidence in Triton logs")
    return hits[:8]


def arrays_from_response(body: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for item in body.get("outputs", []):
        out[item["name"]] = np.asarray(item["data"], dtype=np.float32).reshape(
            item["shape"]
        )
    return out


async def check_rest() -> dict[str, Any]:
    j0 = (PAYLOADS / "j0-0.bin").read_bytes()
    reference = json.loads((REFERENCES / "cuda-rest-response.bin").read_bytes())
    ref_arrays = arrays_from_response(reference)
    url = f"http://{TRITON_SERVICE}.{NAMESPACE}.svc:{HTTP_PORT}/v2/models/{MODEL}/infer"
    began = time.monotonic()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        async with session.post(
            url, data=j0, headers={"Content-Type": "application/json"}
        ) as response:
            body = await response.read()
            if response.status != 200:
                raise RuntimeError(f"Triton REST HTTP {response.status}")
    got = json.loads(body)
    got_arrays = arrays_from_response(got)
    if set(got_arrays) != set(ref_arrays):
        raise RuntimeError(f"Output names differ: {sorted(got_arrays)}")
    max_diff = 0.0
    for name, ref in ref_arrays.items():
        arr = got_arrays[name]
        if arr.shape != ref.shape:
            raise RuntimeError(f"Output '{name}' shape {arr.shape} != {ref.shape}")
        max_diff = max(max_diff, float(np.abs(arr - ref).max()))
    if max_diff > PARITY_TOLERANCE:
        raise RuntimeError(
            f"Value parity exceeded tolerance: max_abs_diff={max_diff:.3e}"
        )
    return {
        "rest_latency_s": time.monotonic() - began,
        "rest_response_sha256": hashlib.sha256(body).hexdigest(),
        "rest_response_bytes": len(body),
        "max_abs_diff_vs_mlserver_cuda_ref": max_diff,
        "parity_tolerance": PARITY_TOLERANCE,
    }


async def check_grpc() -> dict[str, Any]:
    g0 = (PAYLOADS / "g-0.bin").read_bytes()
    options = [
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ]
    began = time.monotonic()
    async with grpc.aio.insecure_channel(
        f"{TRITON_SERVICE}.{NAMESPACE}.svc:{GRPC_PORT}", options=options
    ) as channel:
        method = channel.unary_unary(RPC)
        response = await method(g0, timeout=300)
    if len(response) < 100_000:
        raise RuntimeError(f"Triton gRPC response suspiciously small: {len(response)}")
    return {
        "grpc_latency_s": time.monotonic() - began,
        "grpc_response_sha256": hashlib.sha256(response).hexdigest(),
        "grpc_response_bytes": len(response),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    args = parser.parse_args()
    trial_dir = Path(os.environ["BENCHMARK_TRIAL_DIR"])
    trial_dir.mkdir(parents=True, exist_ok=True)

    began = time.monotonic()
    k8s_config.load_incluster_config()
    apps = k8s.AppsV1Api()
    core = k8s.CoreV1Api()

    for deployment in MLSERVER_DEPLOYMENTS:
        ensure_scaled_zero(apps, core, deployment)

    old_pod = running_pod(core, apps, TRITON_DEPLOYMENT)
    restart_triton(apps)
    new_pod = wait_rollout(apps, core)
    if old_pod is not None and old_pod.metadata.uid == new_pod.metadata.uid:
        raise RuntimeError("Triton was not restarted: pod UID unchanged")

    ready_seconds = asyncio.run(wait_ready())
    evidence = backend_evidence(core, new_pod)
    rest = asyncio.run(check_rest())
    grpc_result = asyncio.run(check_grpc())

    applied = apps.read_namespaced_deployment(TRITON_DEPLOYMENT, NAMESPACE)
    receipts = {
        "variant": args.variant,
        "engine": "triton",
        "deployment": TRITON_DEPLOYMENT,
        "image": applied.spec.template.spec.containers[0].image,
        "pod_image_id": (new_pod.status.container_statuses or [None])[0].image_id,
        "old_pod_uid": old_pod.metadata.uid if old_pod else None,
        "pod_uid": new_pod.metadata.uid,
        "pod_name": new_pod.metadata.name,
        "pod_node": new_pod.spec.node_name,
        "pod_started_at": (
            new_pod.status.start_time.isoformat() if new_pod.status.start_time else None
        ),
        "backend_evidence": evidence,
        "model_ready_seconds": ready_seconds,
        "correctness": {"rest": rest, "grpc": grpc_result},
        "hook_utc": now_utc(),
        "hook_duration_s": time.monotonic() - began,
    }
    (trial_dir / "receipts.json").write_text(
        json.dumps(receipts, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"hook OK: engine=triton pod={new_pod.metadata.name} "
        f"ready={ready_seconds:.1f}s value_parity_max_diff="
        f"{rest['max_abs_diff_vs_mlserver_cuda_ref']:.2e} "
        f"grpc_bytes={grpc_result['grpc_response_bytes']}"
    )


if __name__ == "__main__":
    main()
