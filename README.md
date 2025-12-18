# OpenFold3 on NVIDIA DGX Spark (Grace Blackwell / ARM64)

This repository provides a specialized Docker deployment for running **OpenFold3** on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

This build solves specific compatibility issues ("Dependency Hell") encountered on the Blackwell platform, including:
- **Triton Compatibility**: Uses `triton-nightly` to support `sm_121` (Blackwell) kernels.
- **DeepSpeed Fixes**: Patches DeepSpeed JIT compilation to correctly parse `sm_121` architecture flags (mapping to `sm_120` for NVCC compatibility).
- **ARM64 Support**: Built on the `nvcr.io/nvidia/pytorch:25.01-py3` base image for native ARM64 optimization.
- **Pre-compiled Kernels**: Uses a `docker commit` workflow to bake JIT-compiled CUDA kernels into the image.

## Getting Started

### 1. Build the Docker Image

Use the provided `build.sh` script, which performs the build with GPU access to pre-compile CUDA kernels:

```bash
./build.sh
```

This script:
1. Builds the base image with all dependencies and model weights
2. Runs inference with GPU access to trigger JIT compilation
3. Commits the container (with compiled kernels) as the final image

> **Note**: The first build takes ~5 minutes due to JIT compilation. Subsequent inference runs will be much faster (~9 seconds vs ~3 minutes).

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

If successful, you should see output indicating that DeepSpeed operators were loaded quickly (pre-compiled):
```text
ninja: no work to do.
Time to load evoformer_attn op: 0.04 seconds
...
Total Queries Processed: 1
  - Successful Queries:  1
```

## Repository Structure

- `Dockerfile`: The complete build recipe.
- `build.sh`: Build script with GPU-aware pre-compilation workflow.
- `patch_ds.py`: Python script to patch DeepSpeed for Blackwell compatibility.
- `README.md`: This guide.

## Technical Details

- **Base Image**: `nvcr.io/nvidia/pytorch:25.01-py3`
- **CUDA Version**: 12.8
- **DeepSpeed**: 0.15.4 (Pinned to avoid Muon optimizer conflicts)
- **Model Weights**: Embedded in image (`of3_ft3_v1.pt`, ~2GB)
- **Architecture**: Explicitly maps invalid `compute_121` flags to compatible `compute_120` flags for NVCC, while preserving `sm_121` for Triton.

## Performance

| Metric | Cold Start | Pre-warmed Image |
|--------|------------|------------------|
| `evoformer_attn` load | ~156 seconds | ~0.05 seconds |
| Total inference time | ~3 minutes | ~9 seconds |
