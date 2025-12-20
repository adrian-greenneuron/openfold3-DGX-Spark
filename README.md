# OpenFold3 on NVIDIA DGX Spark (Grace Blackwell / ARM64)

This repository provides a specialized Docker deployment for running **OpenFold3** on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

This build solves specific compatibility issues ("Dependency Hell") encountered on the Blackwell platform, including:
- **Triton Compatibility**: Uses `triton-nightly` to support `sm_121` (Blackwell) kernels.
- **DeepSpeed Fixes**: Patches DeepSpeed JIT compilation to correctly parse `sm_121` architecture flags (mapping to `sm_120` for NVCC compatibility).
- **ARM64 Support**: Built on the `nvcr.io/nvidia/pytorch:25.11-py3` base image (CUDA 13.0) for native ARM64 optimization.
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
docker run --gpus all --ipc=host --shm-size=64g \
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

- **Base Image**: `nvcr.io/nvidia/pytorch:25.11-py3`
- **CUDA Version**: 13.0
- **PyTorch Version**: 2.10
- **DeepSpeed**: 0.15.4 (Pinned to avoid Muon optimizer conflicts)
- **Model Weights**: Embedded in image (`of3_ft3_v1.pt`, ~2GB)
- **Architecture**: Explicitly maps invalid `compute_121` flags to compatible `compute_120` flags for NVCC, while preserving `sm_121` for Triton.

## Benchmark Results

Benchmarks run on **NVIDIA DGX Spark** (Grace Blackwell GB10, 20 CPU cores, 119GB unified RAM).

*Benchmark date: 2025-12-20*

### CUDA 13.0 Performance (NGC 25.11)

| Query | Total Time | Inference | Memory |
|-------|------------|-----------|--------|
| `query_ubiquitin.json` | 55s | 8s | 16 GB |
| `query_homomer.json` | 51s | 8s | 16 GB |
| `query_dna_ptm.json` | 47s | 6s | 15 GB |
| `query_multimer.json` | 173s | 2m 0s | 26 GB |
| `query_protein_ligand.json` | 228s | 3m 0s | 42 GB |
| `query_protein_ligand_multiple.json` | 403s | 5m 55s | **54 GB** |

> **Total Time** = Container startup + model loading + inference + cleanup  
> **Memory** = Peak unified memory (119 GB available on DGX Spark)

### CUDA 13.0 vs CUDA 12.8 Comparison

| Query | CUDA 13.0 | CUDA 12.8 | Diff | % |
|-------|-----------|-----------|------|---|
| `query_ubiquitin.json` | 55s | 49s | +6s | +12% |
| `query_homomer.json` | 51s | 43s | +8s | +19% |
| `query_dna_ptm.json` | 47s | 40s | +7s | +18% |
| `query_multimer.json` | 173s | 168s | +5s | +3% |
| `query_protein_ligand.json` | 228s | 223s | +5s | +2% |
| `query_protein_ligand_multiple.json` | 403s | 406s | -3s | **-1%** |

> **Note**: CUDA 12.8 has ~5-8s faster container overhead. GPU inference time is **identical**. For long-running queries, the overhead difference is negligible (<3%).

### Cold Start vs Pre-warmed

| Metric | Cold Start | Pre-warmed |
|--------|------------|------------|
| `evoformer_attn` compile | ~156s | ~0.5s |
| Ubiquitin total | ~3m | 55s |

### Software Versions

| Component | CUDA 13.0 (NGC 25.11) | CUDA 12.8 (NGC 25.01) |
|-----------|----------------------|----------------------|
| **CUDA** | 13.0.88 | 12.8.61 |
| **PyTorch** | 2.10.0a0 | 2.6.0a0 |
| **Triton** | 3.5.0 (native) | Nightly (cu128) |
| **DeepSpeed** | 0.15.4 (patched) | 0.15.4 (patched) |

## Custom Input

To run inference on your own proteins, create a JSON query file:

```json
{
  "name": "my_protein",
  "modelSeeds": [42],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MKTAYIAKQRQISFVKSHFSRQ..."
      }
    }
  ]
}
```

Then mount it into the container:

```bash
docker run --gpus all --ipc=host --shm-size=64g \
    -v $(pwd)/my_query.json:/input/query.json \
    -v $(pwd)/output:/output \
    openfold3-spark:latest \
    run_openfold predict \
    --query_json=/input/query.json \
    --output_dir=/output
```

See the [OpenFold3 documentation](https://github.com/aqlaboratory/openfold) for full query format details including multi-chain complexes, ligands, and DNA/RNA.

## Batch Processing

To avoid the ~40s container startup overhead per prediction, keep the container running:

```bash
# Start interactive container
docker run -it --gpus all --ipc=host --shm-size=64g \
    -v $(pwd)/queries:/queries \
    -v $(pwd)/output:/output \
    openfold3-spark:latest bash

# Inside container, run multiple predictions
for q in /queries/*.json; do
    run_openfold predict --query_json=$q --output_dir=/output
done
```

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (tested on DGX Spark with GB10)
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Disk Space**: ~30GB for the Docker image
- **Memory**: 64GB+ recommended (set via `--shm-size`)

## Troubleshooting

### "compute_121 is not recognized"
This error occurs on Blackwell GPUs due to NVCC not supporting `compute_121`. The `patch_ds.py` script in this repo fixes this by mapping to `compute_120`. Make sure you're using the pre-built image from `./build.sh`.

### "ninja: error: loading 'build.ninja'"
The CUDA kernels weren't pre-compiled. Rebuild the image using `./build.sh` which runs a warmup inference to trigger JIT compilation.

### Out of Memory (OOM)
- Increase shared memory: `--shm-size=128g`
- For very large proteins, consider reducing batch size or using CPU offloading

### Slow First Run (~3 minutes)
This is expected on a fresh build. The `build.sh` script pre-compiles kernels to avoid this. If you're seeing slow runs, the image may not have been properly warmed up.

## Resources

- [OpenFold3 GitHub](https://github.com/aqlaboratory/openfold)
- [NVIDIA NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## License

This deployment is provided for use with OpenFold3. See the [OpenFold License](https://github.com/aqlaboratory/openfold/blob/main/LICENSE) for terms.
