# Original and current official models: speed and memory

24 October 2026

Extended results: the [six-example investigation](official_model_performance_241026.md) includes all original benchmark inputs and an opt-in, output-preserving cache. The measurements below remain the original two-example comparison.

The current official model was faster on ubiquitin, but slower on the MCL1 complex. Its CUDA tensor peak was lower on MCL1, while sampled whole-system memory was slightly higher. There is no consistent winner across these two examples.

**These models use different weights.** This comparison measures the complete model/runtime combinations. It does not establish prediction accuracy or isolate the effect of a software upgrade. The [same-model modernisation comparison](modernisation_241026.md#matched-speed-comparison) remains separate. The original working default is unchanged.

## Models and inputs

| | Original | Current official |
|---|---|---|
| OpenFold3 | v0.3.1, source `d8afcd3485388d28b2dfd949a202672720693668` | v0.5.0, source `a83cb119f58a86306f9d66ca528ef2a0fffad250` |
| Parameters | `of3_ft3_v1.pt` | OpenBind `of3-ob-2025-06-30-174k.pt` |
| Environment | Retained NVIDIA CUDA 13 image; PyTorch 2.10 development build; DeepSpeed 0.15.4 | Unmodified official Docker/Pixi recipe and lockfile; PyTorch 2.12.1; DeepSpeed 0.19.2 |
| Default kernels | DeepSpeed attention enabled | DeepSpeed disabled; Triton triangle kernels enabled |

Both ran on the same DGX Spark/GB10, with driver 580.178.04. Ubiquitin has **76 protein residues**. MCL1 has **four chains totalling 632 residues, plus four ligands**: three ATP molecules and one other ligand. The same query JSON and saved MSAs were reused for both models.

Each condition had **one warm-up and two timed repeats**, run consecutively. Each repeat started a new Python process inside its container. Both used seed 42, three recycles, 200 diffusion steps and five output samples. Templates and the MSA server were disabled. `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=16` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` were set for both. Other model settings followed each version's defaults; deterministic verification mode was not enabled.

## Runtime

CLI time includes Python imports, CUDA initialisation, model loading and prediction. Container creation, initial setup/downloads, output validation and warm-up are excluded. Inference time is OpenFold3's narrower internal timer. Values below are medians of the two measured repeats; changes use unrounded measurements.

| Input | Original CLI | Official CLI | Official CLI time change | Original inference | Official inference |
|---|---:|---:|---:|---:|---:|
| Ubiquitin | 50.32 s | **27.01 s** | **−46.3% (1.86× faster)** | 8.15 s | **6.28 s** |
| MCL1 complex | **230.86 s** | 273.33 s | +18.4% (slower) | **185.12 s** | 249.87 s |

Bold denotes the lower measured time, not statistical significance. The small example benefits particularly from the official runtime's shorter startup/model-loading overhead. On MCL1, prediction itself takes longer. The test does not identify which model or implementation changes cause these differences.

### Individual timed repeats

| Input | Model | Repeat 1 CLI | Repeat 2 CLI | Repeat 1 inference | Repeat 2 inference |
|---|---|---:|---:|---:|---:|
| Ubiquitin | Original | 50.164 s | 50.478 s | 8.172 s | 8.129 s |
| Ubiquitin | Official | 26.535 s | 27.483 s | 6.248 s | 6.316 s |
| MCL1 | Original | 230.941 s | 230.771 s | 184.708 s | 185.542 s |
| MCL1 | Official | 273.659 s | 272.993 s | 250.557 s | 249.189 s |

## Memory

CUDA tensor memory, CUDA reserved memory and CPU process RSS describe different allocation domains. **Do not add these columns together**, particularly on the Spark's unified-memory hardware. All sizes below are GiB (2³⁰ bytes); medians are calculated independently for each metric.

| Input | Model | Peak CUDA tensor allocation | Peak CUDA reservation | Peak main-process RSS |
|---|---|---:|---:|---:|
| Ubiquitin | Original | 1.776 | 1.795 | 6.536 |
| Ubiquitin | Official | 1.858 | 1.871 | 7.351 |
| MCL1 | Original | 23.265 | 24.410 | 6.622 |
| MCL1 | Official | 20.694 | 21.754 | 7.419 |

Relative to the original, the official model's CUDA tensor peak changed by **+4.6% on ubiquitin** and **−11.0% on MCL1**. Main-process RSS increased by **12.5%** and **12.0%**, respectively. CUDA peaks come from PyTorch's allocator, so they exclude allocations made outside that allocator. RSS is the Linux high-water mark for the main process, not a sum over workers or all GPU allocations.

### Whole-system memory on the Spark

System memory was sampled every 0.1 seconds as `MemTotal − MemAvailable`. This includes driver/context overhead and background activity, and can miss brief peaks. The increase column subtracts the baseline measured immediately before that repeat. It is a useful view of the machine's footprint, but is not an isolated model-memory measurement or a capacity guarantee.

| Input | Model | Pre-run baseline | Sampled system peak | Increase above baseline |
|---|---|---:|---:|---:|
| Ubiquitin | Original | 4.04 | 16.72 | 12.68 |
| Ubiquitin | Official | 4.95 | 21.22 | 16.27 |
| MCL1 | Original | 4.40 | 39.35 | 34.95 |
| MCL1 | Official | 4.60 | 40.78 | 36.18 |

The official model's sampled system peak was **26.9% higher on ubiquitin** and **3.6% higher on MCL1**. Its increase above the pre-run baseline was **28.3%** and **3.5%** higher. The MCL1 CUDA allocation reduction therefore did not translate into lower whole-system memory in these runs.

## Validation and limits

All **12 runs**, including warm-ups, completed successfully and produced five complete model/confidence triplets each: **180 files** in total. Coordinates and confidence values were checked for finite numbers. This checks successful execution, not biological accuracy or agreement between the two models. Their outputs are not expected to match byte for byte.

Two examples, two measured repeats and one seed give limited performance evidence. Saved MSAs exclude the cost of building alignments; warm-ups exclude first-use compilation from the reported medians. Run order was fixed, rather than randomised. The official runtime warned that its Triton Evoformer configuration is untuned for this NVIDIA backend; no kernel tuning was attempted. Wider inputs and ground-truth accuracy evaluation remain necessary before choosing a new default. Maximum complex size before running out of memory remains untested.

All **69 images present before the rebuild work**, and their original tags, remain available. No images were deleted, no original tags were moved and no host driver or system Python was upgraded.

## Repeating the measurements

The committed [resource benchmark](../tools/benchmark_resources.py) launches the normal OpenFold3 CLI through [a memory-measurement wrapper](../tools/measure_inference.py). Run it in the selected image after preparing that version's checkpoint and chemical-component dictionary. Mount the tools, a query JSON, its saved MSAs at the paths named in the JSON, and a fresh output directory. For example, inside the prepared container:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=16 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 /tools/benchmark_resources.py \
    --query-json=/query.json --output-root=/output \
    --warmups=1 --repeats=2 --interval=0.1
```

Use the [separate build profiles](modernisation_241026.md) to prepare the model environments. Keep the current model's cache separate from the original checkpoint. The tools require Linux `/proc/meminfo` and a CUDA-enabled runtime; they retain per-run logs, resource metrics and a system-memory time series. Run one condition at a time.

The [machine-readable record](data/official_model_benchmark_241026.json) includes every repeat and warm-up, query hashes, effective inference settings, image/source/checkpoint identities, instrumentation hashes and validation results. The saved-MSA payloads and prediction files remain in the local benchmark audit; they are not included in this compact repository record. Historical README timings use a different timing boundary and should not replace the measured original-model baseline above.
