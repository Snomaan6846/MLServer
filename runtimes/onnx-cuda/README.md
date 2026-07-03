# ONNX CUDA runtime for MLServer

This package provides a MLServer runtime compatible with ONNX models using
ONNX Runtime with NVIDIA CUDA GPU acceleration.

It depends on `mlserver-onnx` for the core ONNX runtime and adds
`onnxruntime-gpu` for CUDA-accelerated inference.

## Usage

```bash
pip install mlserver-onnx-cuda
```

To also install NVIDIA CUDA runtime libraries via pip (useful when system
CUDA packages are not available):

```bash
pip install "mlserver-onnx-cuda[cuda-libs]"
```

The `[cuda-libs]` extra installs the following NVIDIA CUDA runtime pip
packages so inference sessions can run without a system CUDA installation:

| Package | Version | Purpose |
|---|---|---|
| `nvidia-cuda-nvrtc-cu12` | `~=12.0` | Runtime CUDA compiler |
| `nvidia-cuda-runtime-cu12` | `~=12.0` | Core CUDA runtime (`libcudart`) |
| `nvidia-cublas-cu12` | `~=12.0` | BLAS linear algebra |
| `nvidia-cufft-cu12` | `~=11.0` | FFT operations |
| `nvidia-curand-cu12` | `~=10.0` | Random number generation |
| `nvidia-cusolver-cu12` | `~=11.0` | Matrix decomposition |
| `nvidia-cusparse-cu12` | `~=12.0` | Sparse matrix operations |
| `nvidia-nvjitlink-cu12` | `~=12.0` | JIT kernel linker |
| `nvidia-cudnn-cu12` | `~=9.0` | cuDNN deep learning primitives |

> **Note:** Container images (`Dockerfile.cuda`, `Dockerfile.cuda.konflux`)
> already ship system CUDA packages, so `[cuda-libs]` is not needed there.
> This extra is intended for bare-metal or virtualenv installs where no
> system CUDA toolkit is present.

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
