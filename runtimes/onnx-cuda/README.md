# ONNX CUDA runtime for MLServer

This package provides GPU-accelerated ONNX inference for MLServer using
ONNX Runtime with NVIDIA CUDA. It requires `onnxruntime-gpu` for
GPU-accelerated inference.

The Python source (`mlserver_onnx/`) is shared with `mlserver-onnx` via a
symlink. Both packages provide the same `mlserver_onnx.OnnxModel` runtime
class. Install `mlserver-onnx` for CPU or `mlserver-onnx-cuda` for GPU.

## Usage

```bash
pip install mlserver-onnx-cuda
```

> **Package availability:** This package is not published to PyPI.
> RHOAI/Konflux builds resolve it from the AIPCC private pip index;
> ODH community images build it as a local wheel via `hack/build-wheels.sh`.
> For local development, see the [Developer Setup](../onnx/README.md#developer-setup)
> section in the main ONNX runtime README.

## Configuration

For CUDA execution provider configuration, model settings, and performance
tuning, see the [GPU Acceleration](../onnx/README.md#gpu-acceleration-cuda)
section in the main ONNX runtime README.

## Runtime class

`mlserver_onnx.OnnxModel`

## Build notes

The `mlserver_onnx/` and `tests/` directories are symlinks to
`../onnx/mlserver_onnx` and `../onnx/tests`. This ensures a single source
of truth — edit source in `runtimes/onnx/mlserver_onnx/` only.

`poetry-core` cannot build with symlinks pointing outside the project root.
`hack/build-wheels.sh` handles this by building in a temporary directory
with dereferenced copies (`cp -rL`), leaving the source tree untouched.
The tox config uses `skip_install = true` + `PYTHONPATH` to avoid
triggering a build.
