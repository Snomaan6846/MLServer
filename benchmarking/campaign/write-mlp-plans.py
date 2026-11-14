#!/usr/bin/env python3
"""Write the phase-1 replication (MLP) campaign variants and plans.

Campaign B replicates the phase-1 CPU redesign study's methodology with
the current optimization stack, on the same 28-vCPU hardware class:

  B2 core   (C64, W4, zmq): 6 tracks x 3 row counts
  B3 W sweep (C64, 1-row):  4 tracks x W in {1,2,4,8,12}
  B4 C sweep (W4, 1-row):   4 tracks x C in {2,16,128}
  B5 transport (C64, W4):   3 tracks on queue
  B6 threads (C64, W4):     binary-worker at T1 (T2 is the core default)

Tracks: stock JSON, worker JSON, binary frontend-parse, binary
worker-parse, stock gRPC, gRPC raw buffers.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmark-results" / "mlp-campaign"
IMAGE = (
    "quay.io/syedali/mlserver-cuda@sha256:b6094ab1ae67ef09287e61e70e26ab4127"
    "31551dd89c5790cc4898843800030c"
)

CPU = {
    "deployment": "mlserver-rfdetr-cpu",
    "service": "mlserver-rfdetr-cpu",
    "device": "cpu",
    "expect_cuda": False,
    "image": IMAGE,
    "model": "mlp-small",
}


def env(
    workers=4,
    transport="zmq",
    worker_json="false",
    binary="false",
    worker_binary="false",
    threads=2,
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": "[\"CPUExecutionProvider\"]",
        "MLSERVER_PARALLEL_WORKERS": str(workers),
        "MLSERVER_REST_WORKER_JSON": worker_json,
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": worker_binary,
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": transport,
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": threads, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(
            {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            }
        ),
    }


def grpc_env(raw_buffers="false", **kw):
    e = env(**kw)
    e["MLSERVER_GRPC_RAW_INPUT_BUFFERS"] = raw_buffers
    e["MLSERVER_REST_WORKER_JSON"] = "false"
    e["MLSERVER_REST_BINARY_TENSOR"] = "false"
    e["MLSERVER_REST_WORKER_BINARY"] = "false"
    return e


variants = {}

# Core tracks (W4, T2, zmq)
variants["mlp-json-frontend"] = {**CPU, "env": env()}
variants["mlp-json-worker"] = {**CPU, "env": env(worker_json="true")}
variants["mlp-bin-frontend"] = {**CPU, "env": env(binary="true")}
variants["mlp-bin-worker"] = {**CPU, "env": env(binary="true", worker_binary="true")}
variants["mlp-grpc-stock"] = {**CPU, "env": grpc_env()}
variants["mlp-grpc-raw"] = {**CPU, "env": grpc_env(raw_buffers="true")}

# W sweep (T2, zmq) for four key tracks
for w in (1, 2, 8, 12):  # W4 is the core default
    variants[f"mlp-json-frontend-w{w}"] = {**CPU, "env": env(workers=w)}
    variants[f"mlp-json-worker-w{w}"] = {**CPU, "env": env(workers=w, worker_json="true")}
    variants[f"mlp-bin-worker-w{w}"] = {
        **CPU,
        "env": env(workers=w, binary="true", worker_binary="true"),
    }
    variants[f"mlp-grpc-raw-w{w}"] = {
        **CPU,
        "env": grpc_env(raw_buffers="true", workers=w),
    }

# Transport axis (queue, W4)
variants["mlp-json-frontend-queue"] = {**CPU, "env": env(transport="queue")}
variants["mlp-json-worker-queue"] = {
    **CPU,
    "env": env(transport="queue", worker_json="true"),
}
variants["mlp-bin-worker-queue"] = {
    **CPU,
    "env": env(transport="queue", binary="true", worker_binary="true"),
}

# Thread check (T1)
variants["mlp-bin-worker-t1"] = {
    **CPU,
    "env": env(binary="true", worker_binary="true", threads=1),
}

ROWS = (1, 100, 1000)
FRAMES = {1: 164, 100: 168, 1000: 170}


def rest_json_spec(rows):
    return {
        "protocol": "rest",
        "url": "http://mlserver-rfdetr-cpu.mlserver-perf.svc:8080/v2/models/mlp-small/infer",
        "payloads": [f"/store/payloads/j0-mlp-small-{rows}.bin"],
        "headers": {"Content-Type": "application/json"},
        "min_response_bytes": 20,
    }


def rest_binary_spec(rows):
    return {
        "protocol": "rest",
        "url": "http://mlserver-rfdetr-cpu.mlserver-perf.svc:8080/v2/models/mlp-small/infer",
        "payloads": [f"/store/payloads/triton-binary-mlp-small-{rows}.bin"],
        "headers": {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(FRAMES[rows]),
        },
        "min_response_bytes": 20,
    }


def grpc_spec(rows):
    return {
        "protocol": "grpc",
        "target": "mlserver-rfdetr-cpu.mlserver-perf.svc:8081",
        "payloads": [f"/store/payloads/g-mlp-small-{rows}.bin"],
        "min_response_bytes": 20,
    }


def spec_for(track, rows):
    if "grpc" in track:
        return grpc_spec(rows)
    if track.startswith("mlp-bin"):
        return rest_binary_spec(rows)
    return rest_json_spec(rows)


def entry(cell_name, hook_variant, track, rows):
    return {
        "name": cell_name,
        "prepare_argv": ["python3", "/store/tools/hook.py", hook_variant],
        "spec": spec_for(track, rows),
    }


def plan(session, entries, concurrency, rounds=3):
    return {
        "round_offset": 0,
        "rounds": rounds,
        "clients": 2,
        "seconds": 60,
        "warmup_seconds": 60,
        "warmup_requests": 256,
        "timeout_seconds": 120,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": concurrency,
        "session": session,
        "variants": entries,
    }


TRACKS = {
    "mlp-json-frontend": "mlp-json-frontend",
    "mlp-json-worker": "mlp-json-worker",
    "mlp-bin-frontend": "mlp-bin-frontend",
    "mlp-bin-worker": "mlp-bin-worker",
    "mlp-grpc-stock": "mlp-grpc-stock",
    "mlp-grpc-raw": "mlp-grpc-raw",
}

# B2 core: 6 tracks x 3 rows at C64 W4
b2 = [
    entry(f"{track}@r{rows}", hook, track, rows)
    for track, hook in TRACKS.items()
    for rows in ROWS
]
plans = {"mlp-B2-core": plan("mlp-B2-core", b2, [64])}

# B3 W sweep: 4 tracks x W{1,2,8,12} at C64 (W4 comes from B2)
b3 = []
for track in ("mlp-json-frontend", "mlp-json-worker", "mlp-bin-worker", "mlp-grpc-raw"):
    for w in (1, 2, 8, 12):
        b3.append(entry(f"{track}-w{w}@r1", f"{track}-w{w}", track, 1))
plans["mlp-B3-wsweep"] = plan("mlp-B3-wsweep", b3, [64])

# B4 C sweep: 4 tracks at C{2,16,128} (C64 from B2) — the concurrency
# axis is native to the sweep, so one entry per track
b4 = [
    entry(f"{track}@r1", track, track, 1)
    for track in ("mlp-json-frontend", "mlp-json-worker", "mlp-bin-worker", "mlp-grpc-raw")
]
plans["mlp-B4-csweep"] = plan("mlp-B4-csweep", b4, [2, 16, 128])

# B5 transport: 3 tracks on queue at C64 W4
b5 = [
    entry(f"{track}-queue@r1", f"{track}-queue", track, 1)
    for track in ("mlp-json-frontend", "mlp-json-worker", "mlp-bin-worker")
]
plans["mlp-B5-transport"] = plan("mlp-B5-transport", b5, [64])

# B6 threads: T1 at C64 W4 (T2 from B2)
b6 = [entry("mlp-bin-worker-t1@r1", "mlp-bin-worker-t1", "mlp-bin-worker", 1)]
plans["mlp-B6-threads"] = plan("mlp-B6-threads", b6, [64])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variants.json").write_text(json.dumps(variants, indent=2))
    for name, p in plans.items():
        (OUT / f"plan-{name}.json").write_text(json.dumps(p, indent=2))
    total = sum(
        len(p["variants"]) * len(p["concurrency"]) * p["rounds"] for p in plans.values()
    )
    print(f"wrote {len(variants)} variants, {len(plans)} plans, {total} trials")
    for name, p in plans.items():
        n = len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        print(f"  {name}: {n} trials")


if __name__ == "__main__":
    main()
