#!/usr/bin/env python3
"""RHOAIENG-102092 utilization sampler (workstation side, read-only).

Samples, via the cluster API (cluster-admin context, no impact on the
serving path):
  - kubelet Summary API: per-pod cumulative CPU (usageCoreNanoSeconds),
    instantaneous rate (usageNanoCores), memory working set, CPU PSI
    for the mlserver serving pods
  - DCGM exporter (service proxy): GPU_UTIL, FB_USED, GPU_TEMP,
    POWER_USAGE - already labeled per serving pod

Writes one JSON line per interval. Post-hoc alignment: each sweep case
stores start.json (utc_recorded + requested_start_ns) and result.json
(drained_ns, duration_seconds); CPU per trial = delta of cumulative
usageCoreNanoSeconds across the window; GPU% = mean of in-window samples.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

CTX = (
    "default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/"
    "htpasswd-cluster-admin-user"
)
NODE = "syedali-cluster-pool-j6qh5-gpu-t4-a-8kpk4"
SUMMARY_PATH = f"/api/v1/nodes/{NODE}/proxy/stats/summary"
DCGM_PATH = (
    "/api/v1/namespaces/nvidia-gpu-operator/services/"
    "nvidia-dcgm-exporter:9400/proxy/metrics"
)
POD_PREFIXES = ("mlserver-rfdetr", "triton-rfdetr")
DCGM_METRICS = (
    "GPU_UTIL",
    "FB_USED",
    "FB_FREE",
    "GPU_TEMP",
    "POWER_USAGE",
    "CLOCKS_THROTTLE_REASONS",
)


def oc_raw(path: str) -> bytes:
    return subprocess.check_output(
        ["oc", "--context", CTX, "get", "--raw", path], timeout=30
    )


def sample_pods() -> dict:
    data = json.loads(oc_raw(SUMMARY_PATH))
    out = {}
    for pod in data.get("pods", []):
        name = pod["podRef"]["name"]
        if not any(name.startswith(prefix) for prefix in POD_PREFIXES):
            continue
        cpu = pod.get("cpu") or {}
        mem = pod.get("memory") or {}
        out[name] = {
            "cpu_core_ns": cpu.get("usageCoreNanoSeconds"),
            "cpu_nano_cores": cpu.get("usageNanoCores"),
            "mem_working_set": mem.get("workingSetBytes"),
            "psi_some_avg10": (cpu.get("psi") or {}).get("some", {}).get("avg10"),
        }
    return out


def sample_dcgm() -> dict:
    text = oc_raw(DCGM_PATH).decode()
    out = {}
    for line in text.splitlines():
        if not line.startswith("DCGM_FI_DEV_"):
            continue
        key = line.split("{", 1)[0].strip()
        metric = key.removeprefix("DCGM_FI_DEV_")
        if metric not in DCGM_METRICS:
            continue
        labels = {}
        if "{" in line:
            label_part = line[line.index("{") + 1 : line.index("}")]
            for item in label_part.split(","):
                name, _, value = item.partition("=")
                labels[name] = value.strip('"')
        try:
            metric_value = float(line[line.rindex(" ") + 1 :])
        except ValueError:
            continue
        out[metric.lower()] = {
            "value": metric_value,
            "pod": labels.get("pod"),
            "container": labels.get("container"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=7200.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.duration
    with open(args.output, "a") as stream:
        while time.monotonic() < deadline:
            record = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "mono_ns": time.monotonic_ns(),
            }
            try:
                record["pods"] = sample_pods()
            except Exception as error:  # noqa: BLE001 - keep sampling
                record["pods_error"] = f"{type(error).__name__}: {error}"
            try:
                record["gpu"] = sample_dcgm()
            except Exception as error:  # noqa: BLE001 - keep sampling
                record["gpu_error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
