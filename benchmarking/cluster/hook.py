#!/usr/bin/env python3
"""RHOAIENG-102092 environment-specific sweep prepare hook.

Maps a variant name to a serving deployment configuration, restarts the
serving instance with a fresh process, proves the old instance is gone,
waits for model readiness, verifies the active execution providers from
server logs, and checks REST/gRPC response parity against captured
references. Writes receipts into BENCHMARK_TRIAL_DIR.

Variants may carry a "model" key (default rfdetr-small) — readiness and
parity then target that model, using model-scoped payload/reference files
(g-batched-0.bin, cuda-batched-rest-response.bin, ...). With
--capture-refs, the parity responses are SAVED as the references for that
model instead of compared — run once, deliberately, when deploying a new
model; every normal trial must compare.

Any failure exits nonzero so sweep.py marks the trial invalid.

Usage: hook.py VARIANT_NAME [--capture-refs]
Env:   BENCHMARK_TRIAL_DIR (required), BENCHMARK_VARIANT (set by sweep.py)
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
GPU_DEPLOYMENT = "mlserver-rfdetr"
CPU_DEPLOYMENT = "mlserver-rfdetr-cpu"
GPU_SERVICE = "mlserver-rfdetr"
CPU_SERVICE = "mlserver-rfdetr-cpu"
MODEL = "rfdetr-small"
CONTAINER = "mlserver"
REST_PORT = 8080
GRPC_PORT = 8081
STORE = Path("/store")
PAYLOADS = STORE / "payloads"
REFERENCES = STORE / "references"
IMAGES_FILE = STORE / "images.json"
RPC = "/inference.GRPCInferenceService/ModelInfer"

CUDA_PROVIDERS = '["CUDAExecutionProvider","CPUExecutionProvider"]'
CPU_PROVIDERS = '["CPUExecutionProvider"]'


def server_env(
    providers: str,
    workers: str,
    rest_worker_json: str,
    grpc_raw_input_buffers: str,
    transport: str,
    cuda_graph: str | None,
    cuda_blocking_sync: str | None,
) -> list[dict[str, str]]:
    env = [
        {"name": "MLSERVER_MODEL_ONNX_PROVIDERS", "value": providers},
        {"name": "MLSERVER_PARALLEL_WORKERS", "value": workers},
        {"name": "MLSERVER_REST_WORKER_JSON", "value": rest_worker_json},
        {"name": "MLSERVER_GRPC_RAW_INPUT_BUFFERS", "value": grpc_raw_input_buffers},
        {"name": "MLSERVER_PARALLEL_INFERENCE_TRANSPORT", "value": transport},
    ]
    if cuda_graph is not None:
        env.append({"name": "MLSERVER_MODEL_ONNX_CUDA_GRAPH", "value": cuda_graph})
    if cuda_blocking_sync is not None:
        env.append(
            {
                "name": "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC",
                "value": cuda_blocking_sync,
            }
        )
    return env


VARIANTS: dict[str, dict[str, Any]] = {
    "gpu-stock": {
        "deployment": GPU_DEPLOYMENT,
        "service": GPU_SERVICE,
        "device": "cuda",
        "env": server_env(
            CUDA_PROVIDERS, "1", "false", "false", "queue", "false", "false"
        ),
        "expect_cuda": True,
    },
    "gpu-opt-rest": {
        "deployment": GPU_DEPLOYMENT,
        "service": GPU_SERVICE,
        "device": "cuda",
        "env": server_env(
            CUDA_PROVIDERS, "8", "true", "false", "zmq", "false", "false"
        ),
        "expect_cuda": True,
    },
    "gpu-opt-grpc": {
        "deployment": GPU_DEPLOYMENT,
        "service": GPU_SERVICE,
        "device": "cuda",
        "env": server_env(CUDA_PROVIDERS, "2", "false", "true", "zmq", "true", "true"),
        "expect_cuda": True,
    },
    "cpu-stock": {
        "deployment": CPU_DEPLOYMENT,
        "service": CPU_SERVICE,
        "device": "cpu",
        "env": server_env(CPU_PROVIDERS, "1", "false", "false", "queue", None, None),
        "expect_cuda": False,
    },
    "cpu-opt-rest": {
        "deployment": CPU_DEPLOYMENT,
        "service": CPU_SERVICE,
        "device": "cpu",
        "env": server_env(CPU_PROVIDERS, "8", "true", "false", "zmq", None, None),
        "expect_cuda": False,
    },
    "cpu-opt-grpc": {
        "deployment": CPU_DEPLOYMENT,
        "service": CPU_SERVICE,
        "device": "cpu",
        "env": server_env(CPU_PROVIDERS, "2", "false", "true", "zmq", None, None),
        "expect_cuda": False,
    },
}


EXTRA_VARIANTS_FILE = STORE / "variants.json"


def load_variants() -> dict[str, dict[str, Any]]:
    """Merge built-in variants with data-driven ones from /store/variants.json."""
    variants = dict(VARIANTS)
    if EXTRA_VARIANTS_FILE.exists():
        extra = json.loads(EXTRA_VARIANTS_FILE.read_text())
        for name, config in extra.items():
            if name in variants:
                raise ValueError(f"Variant '{name}' shadows a built-in variant")
            entry = {
                "deployment": config["deployment"],
                "service": config["service"],
                "device": config["device"],
                "env": [
                    {"name": key, "value": value}
                    for key, value in config["env"].items()
                ],
                "expect_cuda": bool(config["expect_cuda"]),
                "model": config.get("model", MODEL),
            }
            if config.get("image"):
                entry["image"] = config["image"]
            entry["command"] = config.get("command")
            variants[name] = entry
    return variants


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for(predicate, timeout: float, description: str, interval: float = 1.0):
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
    selector = selector_for(apps, name)
    pods = core.list_namespaced_pod(NAMESPACE, label_selector=selector).items
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


def patch_target(apps, name: str, image: str, env: list, command) -> None:
    # Strategic merge with explicit deletes: env entries left over from
    # earlier variants are removed so every trial starts from exactly the
    # variant's env. A plain strategic merge would keep stale entries (the
    # merge is keyed by env name), silently contaminating "stock" baselines
    # with knobs from previous variants.
    dep = apps.read_namespaced_deployment(name, NAMESPACE)
    container = next(
        c for c in dep.spec.template.spec.containers if c.name == CONTAINER
    )
    desired = {entry["name"] for entry in env}
    env_patch = [dict(entry) for entry in env]
    for stale in {e.name for e in (container.env or [])} - desired:
        env_patch.append({"name": stale, "$patch": "delete"})
    # command=None clears any previously set command (strategic merge:
    # omitting the key would leave a stale wrapper in place).
    body = {
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {
                    "annotations": {"rhoaieng-102092/restarted-at": now_utc()}
                },
                "spec": {
                    "containers": [
                        {
                            "name": CONTAINER,
                            "image": image,
                            "env": env_patch,
                            "command": command,
                        }
                    ]
                },
            },
        }
    }
    apps.patch_namespaced_deployment(name, NAMESPACE, body)


def wait_rollout(apps, core, name: str):
    dep = apps.read_namespaced_deployment(name, NAMESPACE)
    generation = dep.metadata.generation

    def rolled():
        current = apps.read_namespaced_deployment(name, NAMESPACE)
        status = current.status
        if status.observed_generation != generation:
            return False
        if (status.ready_replicas or 0) != 1 or (status.updated_replicas or 0) != 1:
            return False
        pod = running_pod(core, apps, name)
        return pod if pod is not None else False

    return wait_for(rolled, 420, f"{name} rollout to a fresh ready pod")


async def wait_model_ready(service: str, model: str) -> float:
    url = f"http://{service}.{NAMESPACE}.svc:{REST_PORT}/v2/models/{model}/ready"
    began = time.monotonic()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return time.monotonic() - began
            except aiohttp.ClientError:
                pass
            if time.monotonic() - began > 300:
                raise TimeoutError(f"Model not ready at {url} within 300s")
            await asyncio.sleep(1)


def verify_providers(core, pod, expect_cuda: bool) -> str:
    log = core.read_namespaced_pod_log(pod.metadata.name, NAMESPACE)
    line = next(
        (entry for entry in log.splitlines() if "execution providers" in entry),
        None,
    )
    if line is None:
        raise RuntimeError("No 'execution providers' line found in server logs")
    has_cuda = "CUDAExecutionProvider" in line
    if has_cuda != expect_cuda:
        raise RuntimeError(f"Unexpected providers: {line.strip()}")
    return line.strip()


def arrays_from_response(body: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for item in body.get("outputs", []):
        out[item["name"]] = np.asarray(item["data"], dtype=np.float32).reshape(
            item["shape"]
        )
    return out


async def check_rest(
    service: str, device: str, model: str, capture: bool = False
) -> dict[str, Any]:
    j0 = (PAYLOADS / "j0-0.bin").read_bytes()
    reference_path = REFERENCES / f"{device}{model_stem(model)}-rest-response.bin"
    url = f"http://{service}.{NAMESPACE}.svc:{REST_PORT}/v2/models/{model}/infer"
    began = time.monotonic()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        async with session.post(
            url, data=j0, headers={"Content-Type": "application/json"}
        ) as response:
            body = await response.read()
            if response.status != 200:
                raise RuntimeError(f"REST correctness HTTP {response.status}")
    if capture:
        reference_path.write_bytes(body)
        source = "captured"
    else:
        reference = json.loads(reference_path.read_bytes())
        ref_arrays = arrays_from_response(reference)
        got = json.loads(body)
        got_arrays = arrays_from_response(got)
        if set(got_arrays) != set(ref_arrays):
            raise RuntimeError(
                f"Output names differ: {sorted(got_arrays)} vs {sorted(ref_arrays)}"
            )
        for name, ref in ref_arrays.items():
            arr = got_arrays[name]
            if arr.shape != ref.shape:
                raise RuntimeError(f"Output '{name}' shape {arr.shape} != {ref.shape}")
            if arr.tobytes() != ref.tobytes():
                diff = float(np.abs(arr - ref).max())
                raise RuntimeError(
                    f"Output '{name}' differs from reference (max_abs_diff={diff:.3e})"
                )
        source = "verified"
    return {
        "rest_latency_s": time.monotonic() - began,
        "rest_response_sha256": hashlib.sha256(body).hexdigest(),
        "rest_response_bytes": len(body),
        "rest_reference": source,
    }


async def check_grpc(
    service: str, device: str, model: str, capture: bool = False
) -> dict[str, Any]:
    payload_path = PAYLOADS / f"g{model_stem(model)}-0.bin"
    g0 = payload_path.read_bytes()
    reference_path = REFERENCES / f"{device}{model_stem(model)}-grpc-response.bin"
    options = [
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ]
    began = time.monotonic()
    async with grpc.aio.insecure_channel(
        f"{service}.{NAMESPACE}.svc:{GRPC_PORT}", options=options
    ) as channel:
        method = channel.unary_unary(RPC)
        response = await method(g0, timeout=300)
    if capture:
        reference_path.write_bytes(response)
        source = "captured"
    else:
        reference = reference_path.read_bytes()
        if response != reference:
            raise RuntimeError(
                "gRPC response bytes differ from reference "
                f"(got sha256 {hashlib.sha256(response).hexdigest()}, "
                f"expected {hashlib.sha256(reference).hexdigest()})"
            )
        source = "verified"
    return {
        "grpc_latency_s": time.monotonic() - began,
        "grpc_response_sha256": hashlib.sha256(response).hexdigest(),
        "grpc_response_bytes": len(response),
        "grpc_reference": source,
    }


def resolve_image(variant: str, device: str) -> str:
    mapping = json.loads(IMAGES_FILE.read_text())
    overrides = mapping.get("overrides", {})
    if variant in overrides:
        return overrides[variant]
    return mapping["defaults"][device]


def model_stem(model: str) -> str:
    """File-name stem for model-scoped payloads and references.

    The default model keeps the historical file names (g-0.bin,
    {device}-rest-response.bin). Additional models get a derived suffix,
    e.g. rfdetr-small-batched -> g-batched-0.bin and
    cuda-batched-rest-response.bin.
    """
    if model == MODEL:
        return ""
    if model.startswith(MODEL + "-"):
        return model[len(MODEL) :]
    return "-" + model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    variants = load_variants()
    parser.add_argument("variant", choices=sorted(variants))
    parser.add_argument(
        "--capture-refs",
        action="store_true",
        help=(
            "Capture the parity references for the variant's model instead "
            "of comparing against them. Run once, deliberately, when a new "
            "model is deployed; every normal trial must compare."
        ),
    )
    args = parser.parse_args()
    variant = args.variant
    config = variants[variant]
    model = config.get("model", MODEL)
    trial_dir = Path(os.environ["BENCHMARK_TRIAL_DIR"])

    began = time.monotonic()
    k8s_config.load_incluster_config()
    apps = k8s.AppsV1Api()
    core = k8s.CoreV1Api()

    target = config["deployment"]
    other = CPU_DEPLOYMENT if target == GPU_DEPLOYMENT else GPU_DEPLOYMENT
    image = config.get("image") or resolve_image(variant, config["device"])

    # 1. The sibling deployment and Triton must never coexist with the
    #    target (single GPU on the node).
    ensure_scaled_zero(apps, core, other)
    ensure_scaled_zero(apps, core, "triton-rfdetr")

    # 2. Capture the old pod, then patch image/env and force a fresh process.
    old_pod = running_pod(core, apps, target)
    patch_target(apps, target, image, config["env"], config.get("command"))

    # 3. Wait for a fresh ready pod; prove the old instance is gone.
    new_pod = wait_rollout(apps, core, target)
    if old_pod is not None and old_pod.metadata.uid == new_pod.metadata.uid:
        raise RuntimeError("Server was not restarted: pod UID unchanged")

    # 4. Model readiness through the Service.
    ready_seconds = asyncio.run(wait_model_ready(config["service"], model))

    # 5. Providers must match the variant's device.
    providers_line = verify_providers(core, new_pod, config["expect_cuda"])

    # 6. Response parity for REST and raw gRPC (or a deliberate one-time
    #    reference capture when --capture-refs is given).
    rest = asyncio.run(
        check_rest(config["service"], config["device"], model, args.capture_refs)
    )
    grpc_result = asyncio.run(
        check_grpc(config["service"], config["device"], model, args.capture_refs)
    )

    applied = apps.read_namespaced_deployment(target, NAMESPACE)
    container = next(
        c for c in applied.spec.template.spec.containers if c.name == CONTAINER
    )
    receipts = {
        "variant": variant,
        "deployment": target,
        "service": config["service"],
        "device": config["device"],
        "model": model,
        "reference_capture": args.capture_refs,
        "image": image,
        "pod_image_id": (new_pod.status.container_statuses or [None])[0].image_id,
        "env_applied": {e.name: e.value for e in container.env},
        "resources": {
            "requests": applied.spec.template.spec.containers[0].resources.requests,
            "limits": applied.spec.template.spec.containers[0].resources.limits,
        },
        "old_pod_uid": old_pod.metadata.uid if old_pod else None,
        "pod_uid": new_pod.metadata.uid,
        "pod_name": new_pod.metadata.name,
        "pod_node": new_pod.spec.node_name,
        "pod_started_at": (
            new_pod.status.start_time.isoformat() if new_pod.status.start_time else None
        ),
        "providers_line": providers_line,
        "model_ready_seconds": ready_seconds,
        "correctness": {"rest": rest, "grpc": grpc_result},
        "hook_utc": now_utc(),
        "hook_duration_s": time.monotonic() - began,
    }
    (trial_dir / "receipts.json").write_text(
        json.dumps(receipts, indent=2, allow_nan=False) + "\n"
    )
    mode = "CAPTURED references" if args.capture_refs else "parity=bit-exact"
    print(
        f"hook OK: variant={variant} model={model} pod={new_pod.metadata.name} "
        f"providers={providers_line.split(':')[-1].strip()} "
        f"ready={ready_seconds:.1f}s rest={rest['rest_reference']} "
        f"grpc={grpc_result['grpc_reference']} {mode}"
    )


if __name__ == "__main__":
    main()
