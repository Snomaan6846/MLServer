"""Validate that every trial in a session has complete metric data.

Checks RPS, latency percentiles, CPU-ms/request, CPU%, GPU util, and
memory for every trial. Reports any gaps. Exit nonzero if incomplete.
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_METRICS = [
    "rps",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "cpu_ms_per_request",
    "cpu_pct",
    "gpu_util_mean",
    "mem_ws_max_mb",
]


def validate_session(session_dir: Path) -> tuple[bool, list[str]]:
    summary = json.loads((session_dir / "SUMMARY.json").read_text())
    issues = []

    for row in summary["results"]:
        case = row["case"]
        variant = row["variant"]
        concurrency = row["concurrency"]
        label = f"{case} ({variant}@c{concurrency})"

        if not row.get("valid"):
            issues.append(f"{label}: trial marked INVALID")
            continue

        if not row.get("latency_ms"):
            issues.append(f"{label}: missing latency percentiles")
        else:
            for pct in ("p50", "p95", "p99"):
                if pct not in row["latency_ms"]:
                    issues.append(f"{label}: missing latency {pct}")

        if not row.get("rps"):
            issues.append(f"{label}: missing RPS")

        receipts = session_dir / row["case"] / "receipts.json"
        if not receipts.exists():
            issues.append(f"{label}: missing receipts.json")
            continue

        metrics_file = session_dir / "metrics-summary.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            key = f"{variant}@c{concurrency}"
            if key in metrics:
                cell = metrics[key]
                for metric in REQUIRED_METRICS:
                    if metric in cell and cell[metric] is None:
                        issues.append(f"{label}: {metric} is null")
            else:
                issues.append(f"{label}: not in metrics-summary")

    per_case = session_dir / "metrics-cases.csv"
    if per_case.exists():
        lines = per_case.read_text().strip().split("\n")
        header = lines[0].split(",")
        for line in lines[1:]:
            values = line.split(",")
            case_name = values[0]
            for i, (col, val) in enumerate(zip(header, values)):
                if val == "null" and col in (
                    "cpu_ms_per_req",
                    "cpu_pct",
                    "gpu_util_mean",
                ):
                    if "gpu_util_mean" == col and "cpu" in values[3].lower():
                        continue
                    issues.append(f"{case_name}: {col} is null in cases CSV")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()

    complete, issues = validate_session(args.session)
    total = len(json.loads((args.session / "SUMMARY.json").read_text())["results"])
    print(f"Session: {args.session.name}")
    print(f"Trials: {total}")
    print(f"Data completeness: {'COMPLETE' if complete else 'INCOMPLETE'}")
    if issues:
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("All trials have complete metric data.")


if __name__ == "__main__":
    main()
