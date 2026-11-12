#!/usr/bin/env python3
"""Summarize RHOAIENG-102092 sweep sessions.

Inputs: an exported session directory (SUMMARY.json + case-*/ with client
samples and hook receipts) and the matching utilization-sampler JSONL.

Outputs:
  - <session>/metrics-cases.csv    one row per trial
  - <session>/metrics-summary.csv  aggregated per (variant, concurrency)
  - <session>/metrics-summary.json machine-readable aggregate

Pooled percentiles concatenate raw client samples (per STUDY_GUIDE:
never average percentile values). CPU per request integrates the kubelet
Summary API cumulative counters across each trial window; GPU figures
average DCGM samples attributed to the serving pod.
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np

SERVER_CORES = 28.0  # Guaranteed QoS CPU of the serving deployment


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_cases(session_dir: Path) -> list[dict[str, Any]]:
    summary = json.loads((session_dir / "SUMMARY.json").read_text())
    cases = []
    for row in summary["results"]:
        case_dir = session_dir / row["case"]
        record: dict[str, Any] = dict(row)
        start = json.loads((case_dir / "start.json").read_text())
        record["utc_start"] = parse_utc(start["utc_recorded"])
        latencies = []
        for samples in sorted(case_dir.glob("client-*/samples.jsonl")):
            for line in samples.read_text().splitlines():
                sample = json.loads(line)
                if sample["ok"]:
                    latencies.append((sample["end_ns"] - sample["start_ns"]) / 1e6)
        record["latencies"] = latencies
        if latencies:
            record["pooled"] = dict(
                zip(
                    ("p50", "p90", "p95", "p99"),
                    map(float, np.percentile(latencies, [50, 90, 95, 99])),
                )
            )
        receipts = case_dir / "receipts.json"
        record["receipts"] = (
            json.loads(receipts.read_text()) if receipts.exists() else {}
        )
        cases.append(record)
    return cases


def sampler_stats(
    samples_file: Path, pod_name: str, utc_start: datetime, duration_s: float
) -> dict[str, Any]:
    records = [json.loads(line) for line in samples_file.read_text().splitlines()]
    window_end = utc_start + timedelta(seconds=duration_s + 1.0)
    window = [
        r
        for r in records
        if parse_utc(r["utc"]) >= utc_start - timedelta(seconds=1.0)
        and parse_utc(r["utc"]) <= window_end
    ]
    out: dict[str, Any] = {
        "sampler_samples": len(window),
        "cpu_ms_per_request": None,
        "cpu_pct": None,
        "gpu_util_mean": None,
        "gpu_util_max": None,
        "gpu_fb_used_max": None,
        "mem_working_set_max": None,
    }
    pod_rows = [
        (parse_utc(r["utc"]), r["pods"][pod_name])
        for r in window
        if pod_name in r.get("pods", {})
    ]
    if len(pod_rows) >= 2:
        first, last = pod_rows[0][1], pod_rows[-1][1]
        delta_ns = (last["cpu_core_ns"] or 0) - (first["cpu_core_ns"] or 0)
        elapsed = (pod_rows[-1][0] - pod_rows[0][0]).total_seconds()
        rates = [p["cpu_nano_cores"] for _, p in pod_rows if p["cpu_nano_cores"]]
        mem_ws = [p["mem_working_set"] for _, p in pod_rows if p["mem_working_set"]]
        if mem_ws:
            out["mem_working_set_max"] = float(np.max(mem_ws))
        if elapsed > 0:
            out["cpu_pct"] = float(np.mean(rates) / (SERVER_CORES * 1e9) * 100)
        out["_cpu_delta_ns"] = delta_ns
        out["_cpu_window_s"] = elapsed
    gpu_rows = [
        r["gpu"]
        for r in window
        if r.get("gpu", {}).get("gpu_util", {}).get("pod") == pod_name
    ]
    if gpu_rows:
        utils = [g["gpu_util"]["value"] for g in gpu_rows]
        fb = [g["fb_used"]["value"] for g in gpu_rows if "fb_used" in g]
        out["gpu_util_mean"] = float(np.mean(utils))
        out["gpu_util_max"] = float(np.max(utils))
        out["gpu_fb_used_max"] = float(np.max(fb)) if fb else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.session)
    for case in cases:
        if not case.get("valid") or not case.get("receipts"):
            continue
        pod = case["receipts"].get("pod_name")
        stats = sampler_stats(
            args.sampler, pod, case["utc_start"], case.get("duration_seconds", 0)
        )
        cpu_delta = stats.pop("_cpu_delta_ns", None)
        if cpu_delta is not None and case.get("successful"):
            stats["cpu_ms_per_request"] = cpu_delta / 1e6 / case["successful"]
        case["metrics"] = stats

    header = (
        "case,round,concurrency,variant,valid,rps,successful,p50_ms,p90_ms,"
        "p95_ms,p99_ms,cpu_ms_per_req,cpu_pct,gpu_util_mean,gpu_fb_max,"
        "mem_ws_max_mb,pod"
    )
    lines = [header]
    for case in cases:
        pooled = case.get("pooled", {})
        metrics = case.get("metrics", {})
        lines.append(
            ",".join(
                str(
                    round(case.get(k, metrics.get(k) or pooled.get(k)), 4)
                    if isinstance(case.get(k, metrics.get(k) or pooled.get(k)), float)
                    else case.get(k, metrics.get(k) or pooled.get(k))
                )
                for k in (
                    "case",
                    "round",
                    "concurrency",
                    "variant",
                    "valid",
                    "rps",
                    "successful",
                )
            )
            + ","
            + ",".join(
                "null" if pooled.get(k) is None else f"{pooled[k]:.3f}"
                for k in ("p50", "p90", "p95", "p99")
            )
            + ","
            + ",".join(
                "null" if metrics.get(k) is None else f"{metrics[k]:.3f}"
                for k in (
                    "cpu_ms_per_request",
                    "cpu_pct",
                    "gpu_util_mean",
                    "gpu_fb_used_max",
                )
            )
            + ","
            + (
                "null"
                if metrics.get("mem_working_set_max") is None
                else f"{metrics['mem_working_set_max'] / 1048576:.1f}"
            )
            + ","
            + str(case.get("receipts", {}).get("pod_name"))
        )
    (args.session / "metrics-cases.csv").write_text("\n".join(lines) + "\n")

    groups: dict[tuple, list[dict]] = {}
    for case in cases:
        if case.get("valid"):
            groups.setdefault((case["variant"], case["concurrency"]), []).append(case)
    agg_rows = [
        "variant,concurrency,trials,rps_median,rps_spread_pct,p50_ms,p90_ms,"
        "p99_ms,cpu_ms_per_req,cpu_pct,gpu_util_mean,mem_ws_max_mb"
    ]
    aggregate = {}
    for (variant, concurrency), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        rps = [r["rps"] for r in rows]
        median = float(np.median(rps))
        spread = (max(rps) - min(rps)) / median * 100 if median else None
        pooled_all = [ms for r in rows for ms in r["latencies"]]
        percentiles = dict(
            zip(("p50", "p90", "p99"), np.percentile(pooled_all, [50, 90, 99]))
        )

        def collect(key: str) -> list:
            # `is not None` — gpu_util_mean can legitimately be 0.0 (GPU
            # idle while requests queue in the batcher); a truthiness test
            # would silently drop those measurements from the aggregate.
            return [
                r["metrics"][key] for r in rows if r["metrics"].get(key) is not None
            ]

        cpu_req = collect("cpu_ms_per_request")
        cpu_pct = collect("cpu_pct")
        gpu = collect("gpu_util_mean")
        mem = collect("mem_working_set_max")
        aggregate[f"{variant}@c{concurrency}"] = {
            "trials": len(rows),
            "rps": rps,
            "rps_median": median,
            "rps_spread_pct": spread,
            "p50_ms": float(percentiles["p50"]),
            "p90_ms": float(percentiles["p90"]),
            "p99_ms": float(percentiles["p99"]),
            "cpu_ms_per_request": float(np.mean(cpu_req)) if cpu_req else None,
            "cpu_pct": float(np.mean(cpu_pct)) if cpu_pct else None,
            "gpu_util_mean": float(np.mean(gpu)) if gpu else None,
            "mem_working_set_max_mb": float(np.max(mem) / 1048576) if mem else None,
        }
        agg_rows.append(
            f"{variant},{concurrency},{len(rows)},{median:.3f},"
            f"{'null' if spread is None else f'{spread:.2f}'},"
            f"{percentiles['p50']:.3f},"
            f"{percentiles['p90']:.3f},{percentiles['p99']:.3f},"
            f"{'null' if not cpu_req else f'{np.mean(cpu_req):.3f}'},"
            f"{'null' if not cpu_pct else f'{np.mean(cpu_pct):.1f}'},"
            f"{'null' if not gpu else f'{np.mean(gpu):.1f}'},"
            f"{'null' if not mem else f'{np.max(mem) / 1048576:.1f}'}"
        )
    (args.session / "metrics-summary.csv").write_text("\n".join(agg_rows) + "\n")
    (args.session / "metrics-summary.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + "\n"
    )
    print("\n".join(agg_rows))


if __name__ == "__main__":
    main()
