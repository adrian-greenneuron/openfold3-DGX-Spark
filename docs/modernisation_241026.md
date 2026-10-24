# Simpler OpenFold3 deployments for DGX Spark

24 October 2026

Two separate candidates now run on the Spark without the local DeepSpeed source patch. The original model remains the working default. All existing images were kept; these candidates use different image tags.

| | Original-model candidate | Current official-model candidate |
|---|---|---|
| Model source | Original OpenFold3 v0.3.1 revision, unchanged | Official upstream revision `a83cb119f58a86306f9d66ca528ef2a0fffad250`, reporting v0.5.0 |
| Parameters | Original `of3_ft3_v1.pt`, unchanged checksum | Current default OpenBind `of3-ob-2025-06-30-174k.pt` |
| Environment | Existing NVIDIA base and dependency versions; stock DeepSpeed 0.19.8 | Official Docker/Pixi recipe and lockfile; PyTorch 2.12.1, DeepSpeed 0.19.2, Python 3.14.6 |
| CUDA | PyTorch/toolkit 13.0 | PyTorch CUDA 13.0; locked toolkit 13.2 |
| Attention | Existing DeepSpeed default retained | Upstream defaults: Triton triangle kernels enabled; DeepSpeed disabled |
| Local source patch | None | None |
| `docker commit` needed | No | No |
| Relationship to existing results | Same model; suitable for a matched runtime comparison | Different model and algorithms; not expected to reproduce original predictions |

The current official environment recognised GB10 and ran on the existing driver, **580.178.04**. No host driver, CUDA installation or system Python was upgraded.

## Original model, fewer workarounds

```bash
./build-modern.sh legacy openfold3-spark:legacy-modern
```

This uses [Dockerfile.legacy-modern](../Dockerfile.legacy-modern). It builds directly from the pinned NVIDIA image and public sources, without depending on a retained local OpenFold3 image. It keeps the original source, checkpoint and other Python package versions, and changes DeepSpeed from 0.15.4 to stock 0.19.8.

The build removes two local workarounds:

- Current stock DeepSpeed correctly handles the Spark's native `12.1` target, so `patch_ds.py` and the `12.1 → 12.0` remapping are not used.
- DeepSpeed attention is compiled during a normal, GPU-less Docker build using its supported build options. The image therefore does not need a GPU warm-up followed by `docker commit` to contain the attention extension. Other shape-dependent JIT caches can still be populated during inference.

The recipe continues to use the official CUTLASS repository and NVIDIA's bundled Triton. It does not install a Triton nightly. The original source contains upstream compatibility helpers and historical dependency metadata; those are retained. This is a less patched deployment of the original model, rather than a complete upgrade of its historical environment.

Run normal inference as before, selecting the separate image:

```bash
mkdir -p modern-output/legacy
docker run --gpus all --ipc=host --shm-size=64g \
    --mount "type=bind,src=$(pwd)/modern-output/legacy,dst=/output" \
    openfold3-spark:legacy-modern \
    run_openfold predict \
    --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
    --output_dir=/output
```

The [opt-in deterministic verification](reproducibility.md#deterministic-verification) also works with this candidate. On the final from-base image, all **15 ubiquitin model/confidence files matched the retained image byte for byte** in that mode. Source and checkpoint checks matched; the only changed recorded Python distribution was DeepSpeed.

### Matched speed comparison

These are medians of **two timed repeats after one warm-up**, with the same original model, checkpoint, fixed saved MSAs, seed 42, disabled templates and disabled MSA server. Each timed repeat starts a new Python CLI process inside its image's container. CLI wall time includes Python startup, model loading and prediction; initial container creation and warm-up are excluded. The images were tested consecutively, not simultaneously.

| Dataset | Retained CLI time | Modern CLI time | Time change | Retained inference | Modern inference |
|---|---:|---:|---:|---:|---:|
| Ubiquitin, 76 aa | 50.18 s | **49.19 s** | **−2.0%** | 8.13 s | **7.93 s** |
| MCL1, 632 aa plus four ligands | 231.20 s | **229.47 s** | **−0.7%** | 184.86 s | **184.81 s** |

Bold marks the lower measured time; it does not denote statistical significance. The differences are small: these measurements show no evident regression, not a demonstrated substantial speedup. MCL1 inference time was effectively unchanged. Host-load snapshots and individual repeats were retained with the timing records.

The historical README reported 55 s and 228 s total time for these examples. Its timing boundary and MSA/server conditions differ, so those values are context, not the baseline used to calculate the percentages above.

Stock DeepSpeed passed all **320 synthetic attention calls** in the version-matched forward checks. It still varied on identical captured MCL1 inputs, with maximum repeat difference 0.03125, as the older kernel did. Updating the package has not established deterministic fast inference or removed the need for the verification mode. No biological accuracy claim follows from these runtime tests.

## Current official model and environment

```bash
./build-modern.sh current openfold3-spark:current-upstream
```

This checks out a fixed official revision, then uses its **unmodified** [Dockerfile.pixi](https://github.com/aqlaboratory/openfold-3/blob/a83cb119f58a86306f9d66ca528ef2a0fffad250/docker/Dockerfile.pixi), `openfold3-cuda13` environment and `pixi.lock`. The Ubuntu base digest and Pixi version are fixed through upstream-supported build arguments. The source checkout is cached under `.modern-build/`; set `MODERN_BUILD_ROOT` to choose another location.

This route replaces the hand-maintained dependency list with upstream's maintained environment. It does not inherit our old NVIDIA image, call `patch_ds.py`, modify installed package source or use `docker commit`.

Download its **separate** current parameters and run a small example:

```bash
mkdir -p modern-output/current-cache modern-output/current
docker run --gpus all --ipc=host --shm-size=64g \
    -e TORCH_CUDA_ARCH_LIST=12.1 \
    -e MAX_JOBS=16 -e OMP_NUM_THREADS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --mount "type=bind,src=$(pwd)/modern-output/current-cache,dst=/root/.openfold3" \
    --mount "type=bind,src=$(pwd)/modern-output/current,dst=/output" \
    openfold3-spark:current-upstream bash -c '
        set -e
        setup_openfold --non-interactive
        run_openfold predict \
            --query_json=/opt/openfold3/examples/example_inference_inputs/query_ubiquitin.json \
            --output_dir=/output \
            --use_templates=false --use_msa_server=false
    '
```

`TORCH_CUDA_ARCH_LIST=12.1` is a platform setting: upstream's generic Dockerfile defaults to older GPU targets. No compiler or source rewrite is needed. The setup command also prepares Biotite's chemical-component dictionary inside the container. Keep the container alive for multiple predictions if you want to reuse that setup and JIT caches. Its model weights live in the separate mounted cache and are not baked into the image. Do not point it at the original-model cache.

### What was tested

The unmodified official build completed successfully, then produced complete five-sample outputs for:

- Ubiquitin: one 76-residue protein.
- Modified DNA: the bundled 13-nucleotide example.
- MCL1: four protein chains totalling 632 residues, plus four ligands, using the same saved MSA inputs as the original-model tests.

The subsequent [separate official-model benchmark](official_model_benchmark_241026.md) measured speed and peak memory against the retained model, with one warm-up and two timed repeats per condition. Official-model CLI time was **46.3% lower on ubiquitin** and **18.4% higher on MCL1**. MCL1 CUDA tensor allocation fell **11.0%**, but sampled whole-system peak memory rose **3.6%**. The different weights and algorithms make this a comparison of complete model/runtime combinations, not a software regression or an accuracy comparison.

Upstream emitted a warning that its Triton Evoformer kernel uses an untuned configuration on this NVIDIA backend. End-to-end execution passed, but wider performance testing and ground-truth accuracy evaluation remain to be done. The original-model deterministic verification results do not qualify the new model's repeatability.

The official image's logical size was about **9.08 GB**, with approximately **2.29 GB** of model weights kept separately, versus **22.95 GB** for the retained image. These are logical sizes, not additional physical Docker storage or space freed; all images remain present.

## Recommendation

Keep the original working default for now. The original-model candidate is a practical way to remove the local DeepSpeed patch and `docker commit` while preserving the established model. The current official route is the more maintainable longer-term option, but it changes the model and needs scientific evaluation before replacing established results.

Raw compact results are in [modernisation_checks_241026.json](data/modernisation_checks_241026.json). The [official installation guide](https://github.com/aqlaboratory/openfold-3/blob/a83cb119f58a86306f9d66ca528ef2a0fffad250/docs/source/Installation.md) and [upstream environment definition](https://github.com/aqlaboratory/openfold-3/blob/a83cb119f58a86306f9d66ca528ef2a0fffad250/pixi.toml) describe the maintained platform options.
