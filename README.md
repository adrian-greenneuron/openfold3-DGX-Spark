# OpenFold3 for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository contains Dockerfiles optimized for building and running OpenFold3 on NVIDIA DGX Spark systems, which feature ARM64 architecture (Grace CPU) and Blackwell GPUs (GB10).

## Why This Exists

The official OpenFold3 Docker images are built for x86_64 architecture. DGX Spark uses ARM64, requiring custom builds with specific optimizations.

## Quick Start

```bash
# Clone OpenFold3 source
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3

# Copy the DGX Spark Dockerfile
curl -O https://raw.githubusercontent.com/adriancarr/openfold3-DGX-Spark/main/Dockerfile.spark
mv Dockerfile.spark docker/

# Build (takes ~15 minutes)
docker build --no-cache -t openfold3-spark:cuda13 -f docker/Dockerfile.spark .

# Test
docker run --gpus all --rm openfold3-spark:cuda13 python3 -c \
  "import openfold3; print('OpenFold3 ready')"
```

## Run Inference

```bash
docker run --gpus all --ipc=host --shm-size=16g \
    -v $(pwd):/output -w /output \
    openfold3-spark:cuda13 \
    run_openfold predict \
    --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
    --num_diffusion_samples=1 \
    --num_model_seeds=1 \
    --use_templates=false
```

## Key Optimizations

| Setting | Value | Reason |
|---------|-------|--------|
| `MAX_JOBS` | 4 | Limits parallel compilation to prevent OOM |
| `TORCH_CUDA_ARCH_LIST` | `9.0;12.1` | Targets Hopper/Blackwell only |
| `DS_BUILD_CPU_ADAM` | 0 | Disables x86-only Intel optimizations |
| `DS_BUILD_CCL_COMM` | 0 | Disables Intel oneCCL (x86-only) |

## System Requirements

- NVIDIA DGX Spark (Grace Blackwell / GB10)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build

## Notes

- PyTorch Nightly with CUDA 13.0 is used for Blackwell GPU support
- A warning about compute capability 12.1 may appear - this is expected and safe to ignore
- Model weights (~2.13 GB) are downloaded on first run

## Credits

- [OpenFold3](https://github.com/aqlaboratory/openfold-3) by OpenFold Consortium

## License

Apache 2.0 (same as OpenFold3)
