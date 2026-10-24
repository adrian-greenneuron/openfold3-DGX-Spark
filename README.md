# OpenFold3 on NVIDIA DGX Spark (Grace Blackwell / ARM64)

> **Looking for the latest model?** We recommend the [official OpenFold3 repository](https://github.com/aqlaboratory/openfold-3) for the latest model, weights and installation instructions. This repository's `main` branch preserves the original tested DGX Spark deployment (OpenFold3 v0.3.1 and its original weights). Newer OpenFold3 versions and DGX Spark optimisations are being tested separately on the [development branch](https://github.com/adrian-greenneuron/openfold3-DGX-Spark/tree/modernise/upstream-runtime).

This repository provides a specialized Docker deployment for running **OpenFold3** on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

This build solves specific compatibility issues ("Dependency Hell") encountered on the Blackwell platform, including:
- **Triton Compatibility**: NGC 25.11 includes native Triton 3.5.0 with `sm_121` (Blackwell) support.
- **DeepSpeed Fixes**: Patches DeepSpeed JIT compilation to correctly parse `sm_121` architecture flags (mapping to `sm_120` for NVCC compatibility).
- **ARM64 Support**: Built on the `nvcr.io/nvidia/pytorch:25.11-py3` base image (CUDA 13.0) for native ARM64 optimization.
- **Pre-compiled Kernels**: Uses a `docker commit` workflow to bake JIT-compiled CUDA kernels into the image.

## Getting Started

### 1. Build the Docker Image

Clone this repository, then use `build.sh` to build the image and pre-compile CUDA kernels with GPU access:

```bash
git clone https://github.com/adrian-greenneuron/openfold3-DGX-Spark.git
cd openfold3-DGX-Spark
./build.sh
```

This script:
1. Builds the base image with all dependencies and model weights
2. Runs inference with GPU access to trigger JIT compilation
3. Commits the container (with compiled kernels) as the final image

The build pins the NVIDIA base-image digest, OpenFold3 source revision, CUTLASS revision and Python package versions. It also checks the downloaded model checkpoint's SHA-256. The warm-up uses the bundled ubiquitin example without contacting an MSA server; ordinary inference retains its existing settings.

For a fresh build under a new name, preserving existing image tags:

```bash
BUILD_NO_CACHE=true ./build.sh openfold3-spark:rebuild-check
```

The tested no-cache build on 24 October 2026 took about **12 minutes**, including downloading dependencies and weights, compiling kernels and running warm-up inference. Time depends on the network and machine. `SKIP_WARMUP=true ./build.sh IMAGE:TAG` builds without GPU pre-compilation; the first inference will then compile kernels.

This is a version-pinned rebuild, not a promise of identical Docker image bytes. Ubuntu package repositories and download services remain external dependencies. See [build and verification details](docs/reproducibility.md).

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

This confirms successful execution. **Default fast inference is not guaranteed to produce identical outputs for repeated runs with the same seed.** Upstream atom-to-token accumulation and the DeepSpeed attention path can introduce numerical variation. This was observed in both the retained image and the clean rebuild.

For an opt-in deterministic comparison, use the [verification procedure](docs/reproducibility.md#deterministic-verification). It enables deterministic PyTorch operations and standard attention without changing normal `run_openfold` behaviour. The [24 October validation record](docs/rebuild_validation_241026.md) describes the checks and their limits.

## Repository Structure

- `Dockerfile`: The complete build recipe.
- `build.sh`: Build script with GPU-aware pre-compilation workflow.
- `patch_ds.py`: Python script to patch DeepSpeed for Blackwell compatibility.
- `runtime-constraints.txt`: Version constraints from the verified CUDA 13 runtime.
- `tools/deterministic_inference.py`: Opt-in deterministic inference wrapper.
- `tools/compare_predictions.py`: Exact comparison of model and confidence outputs.
- `docs/reproducibility.md`: Build identities, nondeterminism and verification commands.
- `README.md`: This guide.

## Technical Details

- **Base Image**: `nvcr.io/nvidia/pytorch:25.11-py3`
- **OpenFold3 Source**: `d8afcd3485388d28b2dfd949a202672720693668` (v0.3.1)
- **CUDA Version**: 13.0
- **PyTorch Version**: 2.10
- **DeepSpeed**: 0.15.4 (Pinned to avoid Muon optimizer conflicts)
- **Model Weights**: Embedded in image (`of3_ft3_v1.pt`, ~2GB)
- **Architecture**: Explicitly maps invalid `compute_121` flags to compatible `compute_120` flags for NVCC, while preserving `sm_121` for Triton.

## Benchmark Results

Benchmarks run on **NVIDIA DGX Spark** (Grace Blackwell GB10, 20 CPU cores, 119GB unified RAM).

*Benchmark date: 2025-12-20*

### CUDA 13.0 Performance (NGC 25.11)

| Query | Protein / complex size | Total Time | Inference | Memory |
|-------|------------------------|------------|-----------|--------|
| `query_ubiquitin.json` | One protein chain: **76 aa** | 55s | 8s | 16 GB |
| `query_homomer.json` | Leucine-zipper homodimer: 2 × 34 = **68 aa** | 51s | 8s | 16 GB |
| `query_dna_ptm.json` | One modified DNA strand: **13 nucleotides** | 47s | 6s | 15 GB |
| `query_multimer.json` | Four protein chains (`7cnx`): 2 × 253 + 2 × 42 = **590 aa** | 173s | 2m 0s | 26 GB |
| `query_protein_ligand.json` | Four MCL1 chains: 4 × 158 = **632 aa**, plus three ATP molecules and one other ligand | 228s | 3m 0s | 42 GB |
| `query_protein_ligand_multiple.json` | **Two separate queries**, each with four MCL1 chains (**632 aa**) plus four ligands | 403s | 5m 55s | **54 GB** |

Sizes were checked against the benchmark input JSON files and retained prediction outputs. **aa** means amino-acid residues, summed across all protein-chain copies in one complex.

The largest individual complex tested contained **632 amino-acid residues across four protein chains, plus four ligands**, with a reported peak system memory of **42 GB**. The **54 GB** result is for a run processing two separate complexes, not one combined 1,264-residue complex.

**Maximum supported size has not been established.** These are successful example benchmarks; we have not tested progressively larger proteins or complexes to find the point at which the Spark runs out of memory. The results therefore do not define the largest complex that fits in its 119 GB of unified memory.

> **Total Time** = Container startup + model loading + inference + cleanup  
> **Memory** = Peak system memory (unified CPU/GPU). System at rest uses ~5 GB. DGX Spark has 119 GB total.

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
>
> **Root Cause**: Nsight Systems profiling reveals that CUDA 13's `cuLibraryLoadData` function takes ~1.2s to load 46 GPU libraries, compared to near-instant loading on CUDA 12.8. This is appears inherent to CUDA 13's new lazy library loading mechanism and cannot be optimized at the application level. I think we will just need to wait for future releases of CUDA 13 and/or PyTorch.

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

> **Need CUDA 12.8?** The legacy version is available at the [`cuda-12.8` tag](https://github.com/adrian-greenneuron/openfold3-DGX-Spark/tree/cuda-12.8).

## Custom Input

To run inference on your own proteins, create a JSON query file:

```json
{
  "seeds": [42],
  "queries": {
    "my_protein": {
      "chains": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        }
      ]
    }
  }
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

See the [OpenFold3 documentation](https://github.com/aqlaboratory/openfold-3) for full query format details including multi-chain complexes, ligands, and DNA/RNA.

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
- **Host Python**: 3.11+ for the build-output checks and comparison tool
- **Network**: Access to NGC, GitHub, Ubuntu/Python package repositories and the public model checkpoint during the build. The build warm-up and quick deterministic check do not require an MSA server.
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

### Repeated runs give different predictions

Small CUDA floating-point differences can propagate through recycling and diffusion, even with a fixed seed. See [nondeterminism and deterministic verification](docs/reproducibility.md). This is distinct from a missing or different model checkpoint; the build verifies the checkpoint checksum.

## Resources

- [OpenFold3 GitHub](https://github.com/aqlaboratory/openfold-3)
- [NVIDIA NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## License

This deployment is provided for use with OpenFold3. See the [OpenFold3 License](https://github.com/aqlaboratory/openfold-3/blob/d8afcd3485388d28b2dfd949a202672720693668/LICENSE) and upstream model-parameter terms.
