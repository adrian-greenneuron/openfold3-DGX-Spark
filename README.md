# OpenFold3 on NVIDIA DGX Spark (Grace Blackwell / ARM64)

This repository provides a specialized Docker deployment for running implementation of **OpenFold3** on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

This build solves specific compatibility issues ("Dependency Hell") encountered on the Blackwell platform, including:
- **Triton Compatibility**: Uses `triton-nightly` to support `sm_121` (Blackwell) kernels.
- **DeepSpeed Fixes**: Patches DeepSpeed JIT compilation to correctly parse `sm_121` architecture flags (mapping to `sm_120` for NVCC compatibility).
- **ARM64 Support**: Built on the `nvcr.io/nvidia/pytorch:25.01-py3` base image for native ARM64 optimization.

## Getting Started

### 1. Build the Docker Image
The provided `Dockerfile` handles all patching and dependency resolution automatically.

```bash
docker build -t openfold3-spark:latest .
```

*Note: The build process pins DeepSpeed to 0.15.4 and applies a Python-based patch during the build to ensure the JIT builder works correctly.*

### 2. Run Inference
You can run inference using the `run_openfold` command inside the container.

**Example Command:**
```bash
docker run --gpus all --ipc=host --shm-size=32g \
    -v $(pwd)/output:/output \
    openfold3-spark:latest \
    run_openfold predict \
    --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
    --output_dir=/output
```

### 3. Verification
If successful, you should see output indicating that DeepSpeed operators were loaded and inference completed:
```text
Time to load evoformer_attn op: ... seconds
...
Total Queries Processed: 1
  - Successful Queries:  1
```

## Repository Structure
- `Dockerfile`: The complete build recipe.
- `README.md`: This guide.

## Technical Details
- **Base Image**: `nvcr.io/nvidia/pytorch:25.01-py3`
- **CUDA Version**: 12.8
- **DeepSpeed**: 0.15.4 (Pinned to avoid Muon optimizer conflicts)
- **Architecture**: Explicitly maps invalid `compute_121` flags to compatible `compute_120` flags for NVCC, while preserving `sm_121` for Triton.
