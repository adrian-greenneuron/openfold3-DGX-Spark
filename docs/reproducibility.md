# Rebuilding and verifying OpenFold3 on DGX Spark

Normal inference keeps the existing OpenFold3 configuration, including DeepSpeed attention. Deterministic verification is a separate, opt-in procedure.

## Pinned build inputs

| Component | Identity |
|---|---|
| NVIDIA PyTorch base | `nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21` |
| OpenFold3 source | `d8afcd3485388d28b2dfd949a202672720693668` (v0.3.1) |
| CUTLASS | v3.6.0, checked against `bf9da7b76c766d7ee7d536afc77880a4ef1f1156` |
| DeepSpeed | 0.15.4, with the existing Blackwell architecture-parsing patch |
| Python dependencies | [runtime-constraints.txt](../runtime-constraints.txt), captured from the retained CUDA 13 runtime |
| Checkpoint | `of3_ft3_v1.pt`, 2,288,027,095 bytes |
| Checkpoint SHA-256 | `aedd8f3eb814e3926c8974ef34c9499df224443f173b7e396c97684da6e3eeb6` |

The previous unpinned source clone could follow upstream changes that moved the checkpoint-download script. The pinned revision keeps the tested script and model configuration together. The build fails if the downloaded checkpoint's checksum is wrong or the CUTLASS tag no longer resolves to the expected commit.

The constraints file is for this NVIDIA base image, not a portable environment specification. It preserves NVIDIA's existing PyTorch/Triton packages while constraining installed dependencies. It does not independently install all listed packages or provide an offline package mirror. Ubuntu packages are installed from their repositories without individual version pins. Consequently, successful rebuilds depend on external services and need not have identical image IDs, timestamps or system package versions.

## Build from scratch

From a clone of this repository:

```bash
BUILD_NO_CACHE=true ./build.sh openfold3-spark:rebuild-check
```

This uses `docker build --pull --no-cache`, downloads the public model weights, and runs an offline ubiquitin warm-up to compile GPU kernels. The script checks the successful-query summary and complete model/confidence outputs before committing the warmed container. Existing image names are preserved if you choose a new target tag. The intermediate image is named `openfold3-spark:rebuild-check-base`.

`--no-cache` reruns Docker build steps; BuildKit's package-download caches and already-downloaded immutable base layers may still be reused. A successful rebuild does not depend on an old OpenFold3 image or its compiled kernels. Network access to the pinned sources, package repositories and checkpoint is required. GPU warm-up requires a working NVIDIA Container Toolkit and sufficient free memory.

For a build without GPU access, use `SKIP_WARMUP=true`. This skips GPU validation and compilation, so a successful build alone does not establish successful inference.

## Why default predictions can differ

Two upstream computation paths were isolated in the tested runtime:

- Atom-to-token aggregation uses CUDA `scatter_add_`. Parallel floating-point accumulation can vary between calls. PyTorch documents this behaviour; enabling deterministic algorithms removed the variation in our focused test. [PyTorch documentation](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.scatter_add_.html).
- DeepSpeed Evoformer attention gave different output values for identical inputs, even with deterministic PyTorch algorithms enabled. Switching this path to standard attention removed the remaining variation in the controlled model comparison. Similar consistency problems have been reported in [OpenFold](https://github.com/aqlaboratory/openfold/issues/532) and on [H100 systems](https://github.com/deepspeedai/DeepSpeed/issues/5052); these reports do not establish an identical internal cause.

Small activation differences can propagate through the recycled trunk and diffusion sampling. A fixed seed alone therefore does not establish exact repeatability. This is separate from intentional differences between diffusion seeds or samples.

The focused kernel tests found numerically plausible DeepSpeed outputs, rather than evidence of gross attention calculation errors. On captured MCL1 inputs it was closer to an FP32 reference than ordinary BF16 attention. That does not establish that resulting structural differences are harmless or biologically equivalent. The exact internal cause and behaviour on other GPU/software combinations remain unverified. See the [validation record](rebuild_validation_241026.md).

## Deterministic verification

Use the wrapper below only when checking repeatability or comparing rebuilt images. It fixes Python hash randomisation, sets the cuBLAS workspace configuration, enables deterministic PyTorch operations, disables cuDNN benchmarking and selects standard attention for inference. The allocator setting is applied before CUDA initialisation. Normal `run_openfold` calls do not use this wrapper or change their settings.

### Quick offline comparison

This uses the upstream ubiquitin example with the MSA server and templates disabled, avoiding changes to remote inputs. It is a small reproducibility check, not an accuracy benchmark or a substitute for testing your own complexes.

```bash
mkdir -p verification/run1 verification/run2

for RUN in run1 run2; do
    docker run --gpus all --ipc=host --shm-size=64g \
        -e PYTHONHASHSEED=0 \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e OF3_VERIFICATION_RECORD=/output/verification_settings.json \
        --mount "type=bind,src=$(pwd)/verification/${RUN},dst=/output" \
        openfold3-spark:rebuild-check \
        python3 /opt/openfold3-deployment/tools/deterministic_inference.py predict \
        --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
        --output_dir=/output \
        --use_templates=false \
        --use_msa_server=false
done

python3 tools/compare_predictions.py verification/run1 verification/run2 \
    --output verification/comparison.json
```

Use fresh output directories for each comparison. A passing result reports `"identical": true` and exits zero. Different predictions exit 1; missing, failed or incomplete predictions exit 2. The tool compares exact bytes for model CIFs and both confidence JSON files for every saved sample. Timing files, logs, settings records and configuration files are excluded because paths and timings can legitimately differ. It requires matching sample-file sets and complete model/confidence triplets; empty output directories cannot pass.

To compare two images, run the first command once with each image, keeping everything else identical. For an older image without the wrapper installed, add this read-only mount to the Docker command so both images use the same verification code:

```bash
--mount "type=bind,src=$(pwd)/tools,dst=/opt/openfold3-deployment/tools,readonly"
```

### Verifying your own queries

Keep the query JSON, seeds, checkpoint, MSA/template files, other settings and container mount paths fixed. Supply saved MSAs using the query's `main_msa_file_paths` and, where relevant, `paired_msa_file_paths`, rather than requesting fresh remote searches. Keep templates disabled for the procedure above; comparisons using templates also need identical saved template inputs. Disable the MSA server after supplying the saved inputs.

Use the same GPU and software versions when testing exact repeatability. Deterministic settings do not promise identical results across hardware, PyTorch releases, all possible inputs or future OpenFold3 revisions. Unsupported deterministic operations can raise errors instead of silently producing a comparable result. The wrapper is tied to the pinned model interface.

Standard attention can take longer and use more memory than the default DeepSpeed path. The earlier instrumented MCL1 comparison took roughly eight minutes per image with standard attention, versus about four and a half minutes with deterministic operations and DeepSpeed. Those single-run timings are diagnostic evidence, not a performance benchmark. No inference defaults have been changed.

## Local checks

```bash
bash -n build.sh
python3 -m unittest discover -s tests -v
```

These CPU checks validate the comparison tool's handling of failed, missing and different outputs. GPU inference and numerical checks are recorded separately in [the validation record](rebuild_validation_241026.md).
