# =============================================================================
# OpenFold3 Dockerfile for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)
# =============================================================================
# Base Image: NGC PyTorch 25.01-py3 (Contains CUDA 12.8, PyTorch 2.6.0a0)
# This base image provides critical support for ARM64 and Blackwell (sm_121).

FROM nvcr.io/nvidia/pytorch:25.01-py3

# -----------------------------------------------------------------------------
# Build Configuration
# -----------------------------------------------------------------------------
ENV MAX_JOBS=16 \
    NVCC_THREADS=8 \
    OMP_NUM_THREADS=16 \
    # Force architecture to 12.0 to avoid parsing bugs with 12.1 in some tools,
    # though DeepSpeed patch below handles the runtime detection.
    TORCH_CUDA_ARCH_LIST="12.0" \
    CUTLASS_PATH=/opt/cutlass \
    KMP_AFFINITY=none \
    DS_BUILD_AIO=1 \
    DS_BUILD_CPU_ADAM=0 \
    DS_BUILD_CCL_COMM=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    hmmer kalign aria2 libxrender1 libxext6 libsm6 libxft2 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    rdkit biopython typing-extensions modelcif memory_profiler \
    func_timeout pdbeccdutils pytorch-lightning biotite==1.2.0 \
    "nvidia-cutlass<4" py-cpuinfo lmdb aria2p ijson boto3 \
    ml-collections wandb awscli tqdm pyyaml requests pandas scipy numpy

# -----------------------------------------------------------------------------
# DeepSpeed (Patched for Blackwell)
# -----------------------------------------------------------------------------
# 1. Pin to 0.15.4 to avoid Muon optimizer/Inductor conflicts with Triton nightly.
# 2. Patch builder.py to fix architecture parsing bug:
#    - Problem: DeepSpeed misparses '12.1' as '1.' causing 'compute_1.' error.
#    - Problem: NVCC 12.8 rejects 'compute_121'.
#    - Fix: Clean up version string and map '121' to '120'.
COPY patch_ds.py /opt/patch_ds.py
RUN pip install deepspeed==0.15.4 \
    && python3 /opt/patch_ds.py

# -----------------------------------------------------------------------------
# CUTLASS & OpenFold3 Source
# -----------------------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1 \
    && git clone https://github.com/aqlaboratory/openfold-3.git

# Install OpenFold3 (--no-deps prevents overwriting NGC PyTorch)
WORKDIR /opt/openfold-3
RUN pip install --no-deps -e .

# -----------------------------------------------------------------------------
# Triton Nightly (Required for sm_121)
# -----------------------------------------------------------------------------
# NGC container's Triton is too old for Blackwell. Install compatible nightly.
RUN pip install --pre triton --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# -----------------------------------------------------------------------------
# Runtime Setup
# -----------------------------------------------------------------------------
# Pre-create cache directories
RUN mkdir -p /root/.openfold3 /root/.triton/autotune

# Download model parameters
RUN bash /opt/openfold-3/openfold3/scripts/download_openfold3_params.sh --download_dir=/root/.openfold3



WORKDIR /opt/openfold-3
CMD ["/bin/bash"]
