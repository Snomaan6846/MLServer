#!/usr/bin/env python3
"""Write the worker-binary campaign variants and session plans.

Campaign A (RF-DETR on the GPU deployment, image b6094ab1):
  A1 core: stock JSON vs binary frontend-parse vs binary worker-parse
  A2 reviewer regression: worker-JSON +/- binary (prefilter verification)
  A3 orthogonality: JSON requests with worker-binary on
  A4 batched fallback: batched model +/- worker-binary
  A5 CUDA graph cell
  A6 soak on the worker-binary path
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmark-results" / "worker-binary-campaign"
IMAGE = (
    "quay.io/syedali/mlserver-cuda@sha256:b6094ab1ae67ef09287e61e70e26ab4127"
    "31551dd89c5790cc4898843800030c"
)

GPU = {
    "deployment": "mlserver-rfdetr",
    "service": "mlserver-rfdetr",
    "device": "cuda",
    "expect_cuda": True,
    "image": IMAGE,
}


def env(
    binary="false",
    worker_json="false",
    worker_binary="false",
    cuda_graph="false",
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": "[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"]",
        "MLSERVER_PARALLEL_WORKERS": "1",
        "MLSERVER_REST_WORKER_JSON": worker_json,
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": worker_binary,
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": cuda_graph,
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": 2, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(
            {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            }
        ),
    }


def variant(name, e, model=None):
    v = {**GPU, "name": name, "env": e}
    if model:
        v["model"] = model
    return v


variants = {
    # A1 core comparison
    "wb-stock-rest": variant("wb-stock-rest", env()),
    "wb-binary-frontend": variant("wb-binary-frontend", env(binary="true")),
    "wb-binary-worker": variant(
        "wb-binary-worker", env(binary="true", worker_binary="true")
    ),
    # A2 reviewer regression verification
    "wb-json-worker-binoff": variant(
        "wb-json-worker-binoff", env(worker_json="true")
    ),
    "wb-json-worker-binon": variant(
        "wb-json-worker-binon", env(worker_json="true", binary="true")
    ),
    # A3 orthogonality
    "wb-json-worker-wbinon": variant(
        "wb-json-worker-wbinon",
        env(worker_json="true", binary="true", worker_binary="true"),
    ),
    # A4 batched fallback (batch-1 frames on the dynamic-batch model)
    "wb-batched-frontend": variant(
        "wb-batched-frontend", env(binary="true"), model="rfdetr-small-batched"
    ),
    "wb-batched-workerbin": variant(
        "wb-batched-workerbin",
        env(binary="true", worker_binary="true"),
        model="rfdetr-small-batched",
    ),
    # A5 CUDA graph cell
    "wb-binary-worker-graph": variant(
        "wb-binary-worker-graph",
        env(binary="true", worker_binary="true", cuda_graph="true"),
    ),
}

REST_SPEC = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer",
    "payloads": ["/store/payloads/j0-0.bin"],
    "headers": {"Content-Type": "application/json"},
    "min_response_bytes": 100000,
}

BINARY_SPEC = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer",
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}

BATCHED_BINARY_SPEC = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small-batched/infer",
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}


def spec_for(name):
    if name.startswith("wb-batched"):
        return BATCHED_BINARY_SPEC
    if "binary" in name and "json" not in name:
        return BINARY_SPEC
    return REST_SPEC


def plan(session, names, concurrency, rounds=3, seconds=60, warmup=60):
    return {
        "round_offset": 0,
        "rounds": rounds,
        "clients": 2,
        "seconds": seconds,
        "warmup_seconds": warmup,
        "warmup_requests": 32,
        "timeout_seconds": 120,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": concurrency,
        "session": session,
        "variants": [
            {
                "name": n,
                "prepare_argv": ["python3", "/store/tools/hook.py", n],
                "spec": spec_for(n),
            }
            for n in names
        ],
    }


plans = {
    "wb-A1-core": plan(
        "wb-A1-core",
        ["wb-stock-rest", "wb-binary-frontend", "wb-binary-worker"],
        [1, 4, 8, 16],
    ),
    "wb-A2-jsonworker": plan(
        "wb-A2-jsonworker",
        ["wb-json-worker-binoff", "wb-json-worker-binon"],
        [4, 8],
    ),
    "wb-A3-orthogonality": plan(
        "wb-A3-orthogonality", ["wb-json-worker-wbinon"], [4, 8]
    ),
    "wb-A4-batched": plan(
        "wb-A4-batched",
        ["wb-batched-frontend", "wb-batched-workerbin"],
        [4],
    ),
    "wb-A5-graph": plan("wb-A5-graph", ["wb-binary-worker-graph"], [8]),
    "wb-A6-soak": plan(
        "wb-A6-soak",
        ["wb-binary-worker"],
        [8],
        rounds=1,
        seconds=3600,
        warmup=120,
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variants.json").write_text(json.dumps(variants, indent=2))
    for name, p in plans.items():
        (OUT / f"plan-{name}.json").write_text(json.dumps(p, indent=2))
    total = sum(
        len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        for p in plans.values()
    )
    print(f"wrote {len(variants)} variants, {len(plans)} plans, {total} trials")
    for name, p in plans.items():
        n = len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        print(f"  {name}: {n} trials")


if __name__ == "__main__":
    main()
