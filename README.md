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

- **Base Image**: `nvcr.io/nvidia/pytorch:25.01-py3`
- **CUDA Version**: 12.8
- **DeepSpeed**: 0.15.4 (Pinned to avoid Muon optimizer conflicts)
- **Model Weights**: Embedded in image (`of3_ft3_v1.pt`, ~2GB)
- **Architecture**: Explicitly maps invalid `compute_121` flags to compatible `compute_120` flags for NVCC, while preserving `sm_121` for Triton.

## Benchmark Results

Benchmarks run on **NVIDIA DGX Spark** (Grace Blackwell GB10, 20 CPU cores, 119GB RAM).

*Benchmark date: 2025-12-18*

### Cold Start vs Pre-warmed Image

| Metric | Cold Start | Pre-warmed Image |
|--------|------------|------------------|
| `evoformer_attn` load | ~156 seconds | ~0.07 seconds |
| Ubiquitin inference | ~3 minutes | **9 seconds** |

### Inference Benchmarks (Pre-warmed Image)

The table below shows **pure inference time** (GPU computation only) vs **total time** (includes Docker container startup, model loading, and cleanup):

| Example | Description | Inference | Total | Overhead |
|---------|-------------|-----------|-------|----------|
| `query_ubiquitin.json` | Simple protein (76 residues) | **9s** | 53s | ~44s |
| `query_homomer.json` | Protein homomer | **8s** | 43s | ~35s |
| `query_dna_ptm.json` | DNA with post-translational modifications | **7s** | 41s | ~34s |
| `query_multimer.json` | Protein multimer complex | **2m 11s** | 177s | ~46s |
| `query_protein_ligand.json` | Protein-ligand (MCL1) | **3m 16s** | 239s | ~43s |
| `query_protein_ligand_multiple.json` | Multiple protein-ligand (2 queries) | **6m 30s** | 429s | ~39s |

> **Note**: Container overhead (~35-45s) includes Docker startup, PyTorch/DeepSpeed initialization, and model weight loading. For batch processing, consider keeping the container running to amortize this cost.

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
