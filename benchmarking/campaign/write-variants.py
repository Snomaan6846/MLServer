#!/usr/bin/env python3
"""Write all variant configurations for the final benchmark campaign.

Every variant runs on the HEAD image (single variable = env config).
No stale data — writes fresh to variants.json.
"""
import json
from pathlib import Path

RES = Path(
    "/home/syedali/Workspace/mlserver-performance/MLServer-resources/benchmark-results"
)
CAMPAIGN = RES / "final-campaign"

ship_image = (CAMPAIGN / "ship-image.txt").read_text().strip()

# --- Model A: RF-DETR (unbatched, batch=1) ---

# Stock env (feature OFF — no regression baseline). Every knob is set
# explicitly so receipts prove the baseline had the binary extension OFF.
stock_env = {
    "MLSERVER_MODEL_ONNX_PROVIDERS": '["CUDAExecutionProvider","CPUExecutionProvider"]',
    "MLSERVER_PARALLEL_WORKERS": "1",
    "MLSERVER_REST_WORKER_JSON": "false",
    "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
    "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "queue",
    "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
    "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
    "MLSERVER_REST_BINARY_TENSOR": "false",
}

# Binary tensor env (feature ON)
binary_env = {
    "MLSERVER_MODEL_ONNX_PROVIDERS": '["CUDAExecutionProvider","CPUExecutionProvider"]',
    "MLSERVER_PARALLEL_WORKERS": "1",
    "MLSERVER_REST_WORKER_JSON": "false",
    "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
    "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
    "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
    "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
    "MLSERVER_REST_BINARY_TENSOR": "true",
}

# Binary + CUDA graph
graph_env = dict(binary_env)
graph_env["MLSERVER_MODEL_ONNX_CUDA_GRAPH"] = "true"

# gRPC opt-in env
grpc_env = {
    "MLSERVER_MODEL_ONNX_PROVIDERS": '["CUDAExecutionProvider","CPUExecutionProvider"]',
    "MLSERVER_PARALLEL_WORKERS": "2",
    "MLSERVER_REST_WORKER_JSON": "false",
    "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "true",
    "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
    "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "true",
    "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "true",
    "MLSERVER_REST_BINARY_TENSOR": "true",
}

# gRPC stock env
grpc_stock_env = dict(stock_env)

# --- Model B: RF-DETR batched (dynamic batch, max_batch_size=4) ---
# Same env configs but pointing at the batched model

variants = {
    # Session A: Unbatched REST
    "head-stock-rest": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": stock_env,
    },
    "head-binary-rest": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": binary_env,
    },
    "head-binary-json": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": binary_env,
        # Same env as binary, but the sweep sends JSON payloads
    },
    "head-binary-graph": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": graph_env,
    },
    # Session B: Unbatched gRPC
    "head-stock-grpc": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": grpc_stock_env,
    },
    "head-grpc-optin": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": grpc_env,
    },
    # Session C: Batched (dynamic-batch model). The "model" key points
    # hook.py readiness/parity at the batched model.
    "batched-stock": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": stock_env,
        "model": "rfdetr-small-batched",
    },
    "batched-binary": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": binary_env,
        "model": "rfdetr-small-batched",
    },
    "batched-binary-json": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": binary_env,
        "model": "rfdetr-small-batched",
    },
    "batched-grpc-raw": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "model": "rfdetr-small-batched",
        "env": {
            "MLSERVER_MODEL_ONNX_PROVIDERS": (
                '["CUDAExecutionProvider","CPUExecutionProvider"]'
            ),
            "MLSERVER_PARALLEL_WORKERS": "2",
            "MLSERVER_REST_WORKER_JSON": "false",
            "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "true",
            "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
            "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
            "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
            "MLSERVER_REST_BINARY_TENSOR": "true",
        },
    },
    # Session D: Mixed binary/JSON co-batching
    "batched-mixed": {
        "deployment": "mlserver-rfdetr",
        "service": "mlserver-rfdetr",
        "device": "cuda",
        "expect_cuda": True,
        "image": ship_image,
        "env": binary_env,
        "model": "rfdetr-small-batched",
    },
}

(RES / "variants.json").write_text(json.dumps(variants, indent=2) + "\n")
print(f"Wrote {len(variants)} variants to variants.json")
for name in sorted(variants):
    print(f"  {name}")
