#!/usr/bin/env python3
"""Build the batched-model gRPC wire file from the reference g-0.bin.

Re-serializes the reference ModelInferRequest with model_name swapped to
the batched model. Every other field (raw input contents, shapes,
requested outputs) is preserved; protobuf serialization is deterministic,
so the result differs from the source only in the model_name bytes.

Usage: build-gbatched.py <g-0.bin> <output-g-batched-0.bin>
"""

import sys
from pathlib import Path

from mlserver.grpc import dataplane_pb2 as pb

DEFAULT_MODEL = "rfdetr-small"
BATCHED_MODEL = "rfdetr-small-batched"


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    req = pb.ModelInferRequest()
    req.ParseFromString(src.read_bytes())
    if req.model_name != DEFAULT_MODEL:
        raise SystemExit(f"unexpected model_name in {src}: {req.model_name}")
    if not req.raw_input_contents:
        raise SystemExit(f"{src} has no raw input contents")

    req.model_name = BATCHED_MODEL
    data = req.SerializeToString()
    dst.write_bytes(data)

    # Round-trip check: the re-serialized request parses and points at the
    # batched model with identical tensor bytes.
    check = pb.ModelInferRequest()
    check.ParseFromString(dst.read_bytes())
    assert check.model_name == BATCHED_MODEL
    assert check.raw_input_contents == req.raw_input_contents
    assert [list(i.shape) for i in check.inputs] == [list(i.shape) for i in req.inputs]

    print(
        f"wrote {dst} ({len(data)} bytes, model_name={BATCHED_MODEL}, "
        f"input shape={list(check.inputs[0].shape)})"
    )


if __name__ == "__main__":
    main()
