#!/usr/bin/env python3
"""Distill the ONNX thread-sweep campaign into tracked evidence.

Copies, for every session under
benchmark-results/onnx-thread-sweep/sessions/:
  - plan.json, SUMMARY.json, metrics-cases.csv, metrics-summary.csv,
    metrics-summary.json
  - case-*/receipts.json (per-trial provenance)

Plus campaign-level context:
  - variants.json (variant definitions used by the hook)
  - logs/session.log (execution narrative)
  - cudnn-investigation/* (manual verification artifacts)

Bulky raw data (per-request samples, telemetry streams, client logs)
stays local-only under benchmark-results/, matching the phase-2
evidence convention.

Usage: python3 distill-thread-sweep.py
"""

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "benchmark-results" / "onnx-thread-sweep"
DST = REPO / "results" / "onnx-thread-sweep"
SESSION_FILES = [
    "plan.json",
    "SUMMARY.json",
    "metrics-cases.csv",
    "metrics-summary.csv",
    "metrics-summary.json",
]


def distill_session(session_dir: Path, dst_dir: Path) -> tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in SESSION_FILES:
        src = session_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
    receipts = 0
    for case_dir in sorted(session_dir.glob("case-*")):
        receipt = case_dir / "receipts.json"
        if receipt.exists():
            case_dst = dst_dir / case_dir.name
            case_dst.mkdir(exist_ok=True)
            shutil.copy2(receipt, case_dst / "receipts.json")
            receipts += 1
    return copied, receipts


def main() -> int:
    sessions_src = SRC / "sessions"
    if not sessions_src.exists():
        print(f"missing {sessions_src}", file=sys.stderr)
        return 1

    sessions = sorted(p for p in sessions_src.iterdir() if p.is_dir())
    print(f"distilling {len(sessions)} sessions -> {DST}")

    index = {"sessions": {}}
    for session_dir in sessions:
        summary = session_dir / "SUMMARY.json"
        if not summary.exists():
            print(f"  {session_dir.name}: NO SUMMARY.json — skipped")
            continue
        data = json.loads(summary.read_text())
        valid = sum(1 for r in data["results"] if r["valid"])
        total = len(data["results"])
        copied, receipts = distill_session(session_dir, DST / "sessions" / session_dir.name)
        index["sessions"][session_dir.name] = {
            "trials_total": total,
            "trials_valid": valid,
            "receipts": receipts,
            "files_copied": copied,
        }
        print(f"  {session_dir.name}: {valid}/{total} valid, {receipts} receipts")

    # Campaign-level context
    variants = Path("/tmp/opencode/thread-sweep/variants.json")
    if variants.exists():
        shutil.copy2(variants, DST / "variants.json")
        print(f"  variants.json copied")

    session_log = SRC / "logs" / "session.log"
    if session_log.exists():
        (DST / "logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(session_log, DST / "logs" / "session.log")
        print(f"  session.log copied")

    investigation = SRC / "cudnn-investigation"
    if investigation.exists():
        inv_dst = DST / "cudnn-investigation"
        inv_dst.mkdir(parents=True, exist_ok=True)
        for f in investigation.iterdir():
            if f.is_file():
                shutil.copy2(f, inv_dst / f.name)
        print(f"  cudnn-investigation copied")

    (DST / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"  INDEX.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
