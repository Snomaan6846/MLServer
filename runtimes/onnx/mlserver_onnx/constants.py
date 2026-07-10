"""Shared constants for GPU / accelerator support (CUDA, ROCm, etc.)."""

NVIDIA_LIB_MODULES = [
    "nvidia.cublas.lib",
    "nvidia.cudnn.lib",
    "nvidia.cuda_runtime.lib",
    "nvidia.nvjitlink.lib",
    "nvidia.cufft.lib",
    "nvidia.curand.lib",
    "nvidia.cusolver.lib",
    "nvidia.cusparse.lib",
    "nvidia.cuda_nvrtc.lib",
]
