#!/usr/bin/env python3
"""Export the benchmark models into assets/models/ and verify their SHAs.

Run inside model-preparation/export-env (the pinned CPU PyTorch/RF-DETR
stack). Produces:
  - assets/models/rfdetr-small/model.onnx         (fixed batch=1)
  - assets/models/rfdetr-small-batched/model.onnx (dynamic batch)

Both are verified against assets/MANIFEST.json — the pinned contract from
the phase-2 campaign. Exits nonzero on any SHA mismatch.
"""

import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets"
MODELS = ASSETS / "models"


def expected_sha(rel: str) -> str:
    manifest = json.loads((ASSETS / "MANIFEST.json").read_text())
    return manifest[rel]


def verify(path: Path, rel: str) -> None:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    want = expected_sha(rel)
    if sha != want:
        raise SystemExit(f"SHA mismatch for {rel}: got {sha}, expected {want}")
    print(f"verified {rel}: {sha[:16]}...")


def main() -> None:
    from rfdetr import RFDETRSmall

    MODELS.mkdir(parents=True, exist_ok=True)

    print("exporting rfdetr-small (fixed batch=1)...")
    original = RFDETRSmall()
    artifact = original.export(
        output_dir=str(MODELS / "rfdetr-small"),
        output_name="model",
        batch_size=1,
        dynamic_batch=False,
        opset_version=17,
        verbose=False,
    )
    verify(Path(artifact), "models/rfdetr-small/model.onnx")

    print("exporting rfdetr-small-batched (dynamic batch)...")
    batched = RFDETRSmall()
    artifact = batched.export(
        output_dir=str(MODELS / "rfdetr-small-batched"),
        output_name="model",
        batch_size=1,
        dynamic_batch=True,
        opset_version=17,
        verbose=False,
    )
    verify(Path(artifact), "models/rfdetr-small-batched/model.onnx")

    print("both models exported and SHA-verified")


if __name__ == "__main__":
    main()
