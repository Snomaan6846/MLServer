#!/usr/bin/env python3
"""Write all session plans for the final benchmark campaign.

Each plan specifies: variants (in execution order), concurrency levels,
rounds, timing, and payload specs. Fresh output — no reuse.
"""
import json
from pathlib import Path

CAMPAIGN = Path(
    "/home/syedali/Workspace/mlserver-performance/MLServer-resources"
    "/benchmark-results/final-campaign"
)
PLANS = CAMPAIGN / "plans"
PLANS.mkdir(parents=True, exist_ok=True)

# Payload specs
binary_spec = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer",
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}

json_spec = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer",
    "payloads": ["/store/payloads/j0-0.bin"],
    "min_response_bytes": 100000,
}

grpc_spec = {
    "protocol": "grpc",
    "target": "mlserver-rfdetr.mlserver-perf.svc:8081",
    "payloads": ["/store/payloads/g-0.bin"],
    "min_response_bytes": 100000,
}

# Batched model endpoints (different model name)
BATCHED_URL = (
    "http://mlserver-rfdetr.mlserver-perf.svc:8080"
    "/v2/models/rfdetr-small-batched/infer"
)

batched_binary_spec = {
    "protocol": "rest",
    "url": BATCHED_URL,
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}

batched_json_spec = {
    "protocol": "rest",
    "url": BATCHED_URL,
    "payloads": ["/store/payloads/j0-0.bin"],
    "min_response_bytes": 100000,
}

# gRPC routes by the model_name embedded in the request protobuf — the
# batched model needs its own wire file (g-batched-0.bin).
batched_grpc_spec = {
    "protocol": "grpc",
    "target": "mlserver-rfdetr.mlserver-perf.svc:8081",
    "payloads": ["/store/payloads/g-batched-0.bin"],
    "min_response_bytes": 100000,
}

# Mixed spec: alternates between binary and JSON payloads. Each payload
# needs its own Content-Type/negotiation headers, so the spec carries
# per-payload header overrides (client.py merges them over the base).
mixed_spec = {
    "protocol": "rest",
    "url": BATCHED_URL,
    "payloads": [
        "/store/payloads/triton-binary-rest.bin",
        "/store/payloads/j0-0.bin",
    ],
    "payload_headers": [
        {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": "225",
        },
        {"Content-Type": "application/json"},
    ],
    "min_response_bytes": 100000,
}

# Common sweep parameters
common = {
    "round_offset": 0,
    "rounds": 3,
    "concurrency": [1, 2, 4, 8, 16],
    "clients": 2,
    "seconds": 60,
    "warmup_seconds": 60,
    "warmup_requests": 32,
    "timeout_seconds": 120,
    "ready_timeout_seconds": 300,
    "prepare_timeout_seconds": 900,
    "max_start_lateness_ms": 20,
}


def make_plan(session_name, variant_specs):
    """Build a plan from (variant_name, prepare_argv, spec) tuples."""
    return {
        **common,
        "session": session_name,
        "variants": [
            {
                "name": name,
                "prepare_argv": ["python3", "/store/tools/hook.py", name],
                "spec": spec,
            }
            for name, spec in variant_specs
        ],
    }


# Session A: Unbatched REST (4 variants, 60 trials)
plan_a = make_plan(
    "final-unbatched-rest",
    [
        ("head-stock-rest", json_spec),  # stock baseline (JSON)
        ("head-binary-rest", binary_spec),  # binary tensor (the headline)
        ("head-binary-json", json_spec),  # binary ON, JSON requests (fallback)
        ("head-binary-graph", binary_spec),  # binary + cuda_graph
    ],
)
# Session A uses different concurrency for graph variant? No — same.
# The paired analyses handle the comparisons.

# Session B: Unbatched gRPC (2 variants, 30 trials)
plan_b = make_plan(
    "final-unbatched-grpc",
    [
        ("head-stock-grpc", grpc_spec),  # stock baseline
        ("head-grpc-optin", grpc_spec),  # opt-in + converter fix
    ],
)

# Session C: Batched (4 variants, 60 trials)
# Note: the batched model needs the deployment patched to point at
# rfdetr-small-batched instead of rfdetr-small. The hook handles this
# via the model URI in the deployment env.
plan_c = make_plan(
    "final-batched",
    [
        ("batched-stock", batched_json_spec),  # stock, batched model, JSON
        ("batched-binary", batched_binary_spec),  # binary, batched model
        ("batched-binary-json", batched_json_spec),  # binary ON, JSON requests
        ("batched-grpc-raw", batched_grpc_spec),  # gRPC raw buffers, batched
    ],
)

# Session D: Mixed co-batching (1 variant, 9 trials)
# Only meaningful at higher concurrency where the batcher collects
# multiple requests within the batch window
plan_d = {
    **common,
    "concurrency": [4, 8, 16],
    "rounds": 3,
    "session": "final-mixed",
    "variants": [
        {
            "name": "batched-mixed",
            "prepare_argv": ["python3", "/store/tools/hook.py", "batched-mixed"],
            "spec": mixed_spec,
        }
    ],
}

# Write all plans
for letter, plan in [("A", plan_a), ("B", plan_b), ("C", plan_c), ("D", plan_d)]:
    path = PLANS / f"session-{letter}.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    n_trials = plan["rounds"] * len(plan["concurrency"]) * len(plan["variants"])
    n_variants = len(plan["variants"])
    print(
        f"Plan {letter}: {plan['session']} ({n_trials} trials, {n_variants} variants)"
    )

# Write paired-analysis specs
paired_specs = {
    "A": {
        "comparisons": [
            {
                "control": "head-stock-rest",
                "candidate": "head-binary-rest",
                "label": "Binary tensor vs stock",
            },
            {
                "control": "head-stock-rest",
                "candidate": "head-binary-json",
                "label": "JSON fallback vs stock",
            },
            {
                "control": "head-binary-rest",
                "candidate": "head-binary-graph",
                "label": "CUDA graph delta",
            },
        ]
    },
    "B": {
        "comparisons": [
            {
                "control": "head-stock-grpc",
                "candidate": "head-grpc-optin",
                "label": "gRPC opt-in vs stock",
            },
        ]
    },
    "C": {
        "comparisons": [
            {
                "control": "batched-stock",
                "candidate": "batched-binary",
                "label": "Batched binary vs batched stock",
            },
            {
                "control": "batched-stock",
                "candidate": "batched-binary-json",
                "label": "Batched JSON fallback vs batched stock",
            },
        ]
    },
    "D": {},
}

paired_dir = CAMPAIGN / "paired-specs"
paired_dir.mkdir(parents=True, exist_ok=True)
for letter, spec in paired_specs.items():
    path = paired_dir / f"{letter}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")

print("\nAll plans and paired-analysis specs written.")
