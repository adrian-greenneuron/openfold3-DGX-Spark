# Six-example benchmark and official-model performance investigation

24 October 2026

The official model spends more time in diffusion than the original model on MCL1. Reusing calculations that are constant across diffusion steps reduces official-model CLI time by **11.2–13.4% on the three larger benchmark cases**, with **byte-identical prediction files** and **unchanged peak CUDA tensor allocation** on all six cases. The original remains faster on both MCL1 cases; the cached official model is fastest on the other four.

The original remains the working default. The cache is a separate, opt-in candidate for the pinned official model. All existing Docker images and tags were retained.

## Speed

Medians of two timed repeats after one warm-up per condition. CLI time includes Python startup, CUDA initialisation, model loading and prediction; it excludes initial container creation/setup, warm-up and output validation. The four completed original/official conditions from the [earlier comparison](official_model_benchmark_241026.md) were reused unchanged. Every other condition was measured anew, consecutively on the same Spark.

| Benchmark | Original CLI | Official CLI | Cached official CLI | Cache time change vs official |
|---|---:|---:|---:|---:|
| Ubiquitin: 76 aa | 50.32 s | 27.01 s | **26.54 s** | −1.7% |
| Homodimer: 68 aa | 49.59 s | 26.40 s | **25.37 s** | −3.9% |
| Modified DNA: 13 nt | 48.08 s | 24.16 s | **23.30 s** | −3.6% |
| Multimer: 590 aa | 2m 49.44s | 2m 53.09s | **2m 29.93s** | −13.4% |
| MCL1: 632 aa plus four ligands | **3m 50.86s** | 4m 33.33s | 4m 2.69s | −11.2% |
| Two separate MCL1 complexes | **6m 51.55s** | 8m 40.38s | 7m 39.02s | −11.8% |

Bold identifies the lowest median, not statistical significance. Improvements of less than a second on the small cases may be noise; those cases do not establish a substantial caching benefit. The original and official models have different parameters and implementation details, so their speed differences are not a same-model software regression test.

The narrower internal inference timers help separate startup from prediction:

| Benchmark | Original inference | Official inference | Cached official inference |
|---|---:|---:|---:|
| Ubiquitin | 8.15 s | 6.28 s | **6.08 s** |
| Homodimer | 7.84 s | 5.89 s | **5.69 s** |
| Modified DNA | 6.35 s | 4.36 s | **4.24 s** |
| Multimer | **122.70 s** | 148.04 s | 126.51 s |
| MCL1 | **185.12 s** | 249.87 s | 219.03 s |
| Two MCL1 complexes | **365.91 s** | 496.33 s | 435.44 s |

The official runtime's smaller startup/loading overhead explains much of its advantage on short examples. On larger examples, diffusion computation is more influential. For the multimer, caching reduces prediction time enough that the official runtime's lower startup overhead gives it the lowest total time.

## Memory

CUDA tensor allocation is PyTorch's allocator high-water mark, not total GPU or system memory. Values are GiB (2³⁰ bytes). Additional cached tensors are retained during diffusion; on these inputs, they fit below the peak reached elsewhere in the existing computation.

| Benchmark | Original CUDA tensor peak | Official CUDA tensor peak | Cached official CUDA tensor peak |
|---|---:|---:|---:|
| Ubiquitin | **1.776** | 1.858 | 1.858 |
| Homodimer | 1.570 | **1.569** | **1.569** |
| Modified DNA | **1.490** | 1.500 | 1.500 |
| Multimer | 16.161 | **14.520** | **14.520** |
| MCL1 | 23.265 | **20.694** | **20.694** |
| Two MCL1 complexes | 23.265 | **20.694** | **20.694** |

The cache changes measured peak CUDA tensor allocation by **0.0% in every case** relative to the official baseline. The official/cached peaks are **11.0% lower than the original** for both MCL1 cases, and **10.2% lower** for the multimer. The small cases have effectively similar CUDA peaks, except ubiquitin's **4.6% increase** in the official runtime. The homodimer's difference is only about 0.03%.

Whole-system memory was sampled every 0.1 s as `MemTotal − MemAvailable`. It includes background activity and driver/context overhead, and can miss brief peaks. It must not be added to CUDA or process-RSS measurements.

| Benchmark | Original sampled system peak | Official sampled system peak | Cached official sampled system peak |
|---|---:|---:|---:|
| Ubiquitin | **16.72** | 21.22 | 20.96 |
| Homodimer | **15.75** | 20.19 | 20.37 |
| Modified DNA | **15.40** | 19.46 | 19.77 |
| Multimer | **32.19** | 34.12 | 34.11 |
| MCL1 | **39.35** | 40.78 | 40.83 |
| Two MCL1 complexes | **40.91** | 41.54 | 41.58 |

Relative to the official baseline, cached system peaks change by **−1.3%, +0.9%, +1.6%, −0.01%, +0.1% and +0.1%**, respectively. These small system-wide differences do not establish a cache-memory reduction or increase. The official model's smaller CUDA peak on large cases does not imply lower whole-system memory. These tests still do not establish the largest complex that fits on the Spark.

## Why MCL1 is slower

One diagnostic MCL1 profile per image recorded the following CUDA-stream elapsed intervals. These include stream-idle gaps and instrumentation/first-use overhead; they are **not kernel busy-time totals or substitute benchmark replicates**.

| Stage | Original | Official |
|---|---:|---:|
| Main trunk | 90.74 s | 91.67 s |
| Diffusion sampling | 84.86 s | 154.58 s |
| Confidence heads | 7.50 s | 7.40 s |

The difference is concentrated in diffusion. The source explains a relevant implementation change: the original passes its DeepSpeed setting into diffusion, and that attention wrapper converts FP32 inputs to BF16. The official `_rollout` instead explicitly requests `use_high_precision_attention=True`; its diffusion attention runs through the ordinary FP32 tensor path. The upstream matmul precision policy is preserved in both cases. This profile and source evidence identify a strong explanation, but do not isolate every cause as a controlled precision-only ablation would.

The official trunk does use a Triton triangle kernel that warns it is untuned for NVIDIA. Nevertheless, its trunk duration is similar to the original in this profile. The warning alone does not explain the MCL1 slowdown.

## What the cache changes

The unchanged official computation repeatedly derives pair-conditioning features, normalises the conditioned pair representation and projects it into each diffusion block's attention bias. For ordinary inference, these quantities depend on the fixed trunk/batch features, not on the current noise timestep or noisy coordinates.

The [opt-in cache](../tools/cached_diffusion.py) reuses those results within one `SampleDiffusion` rollout. It continues to recompute all timestep-dependent single features and noisy-coordinate-dependent atom features. It retains the same weights, attention implementation, precision policy, three recycles, 200 diffusion steps and five samples. No model configuration fields are changed.

Each cache is cleared after its rollout, including between queries. It keeps one conditioning result, one normalisation result and at most one bias result per diffusion block. Strong input references prevent object-ID recycling from producing false cache hits; tensor version changes invalidate entries where version counters exist. Inference-mode tensors lack version counters, so this also relies on the pinned rollout's immutable-input contract. The relevant upstream source-file hashes are checked before enabling the cache; an unqualified source revision fails before prediction.

Real cache hits are checked exactly against the original functions: **26 checks per rollout**, including all 24 bias blocks. Across all six cases, **every one of the 18 cached runs**, including warm-ups, matched the official baseline's CIF and confidence files byte for byte. Both queries in the final case were checked. Model-configuration JSON files also matched exactly. The full combined study comprises **54 runs and 945 prediction/confidence files**.

These observations qualify the tested input/configuration set. They do not prove equivalence for every possible input, a future upstream version, training, or other sampling modes.

### Rejected candidate

A separate `torch.compile` experiment was faster on representative attention tests, but failed a strict numerical check on actual diffusion activations: maximum absolute difference **0.00555**, outside the preselected `rtol=1e-4`, `atol=1e-5` gate. It was rejected; no model-level timing or accuracy benefit is claimed for it. An initial cache prototype was also superseded because it retained excessive intermediate entries. The bounded cache above is the accepted candidate.

## Using the separate candidate

Use the [pinned official build](modernisation_241026.md#current-official-model-and-environment), its separate checkpoint cache and the same setup procedure. Mount this repository's `tools` directory at `/tools` in that image, then replace the prediction entry point with:

```bash
PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 /tools/cached_inference.py --cache-record=/output/cache_record.json \
    predict --query_json=/query.json --output_dir=/output/prediction \
    --use_templates=false --use_msa_server=false
```

The wrapper is opt-in. Ordinary `run_openfold`, `build.sh`, the original checkpoint and existing image tags remain unchanged. The [measurement launcher](../tools/run_model_benchmarks.py) accepts a frozen JSON condition manifest and runs Docker conditions consecutively, with an optional alternative measurement wrapper. It preserves completed records for reuse and never overwrites a condition directory.

All individual measurements, warm-ups, image/source/checkpoint identities, instrumentation hashes, profiles, exact cache checks and output hashes are in the [compact result record](data/official_model_six_benchmarks_241026.json). The final source guard and command-line entry point were added after timing; they do not change the tested cache computation. Their source check and installation were tested in the pinned image without repeating inference.

## Structural comparisons

The [MCL1 local-structure analysis](mcl1_local_structure_comparison_241026.md) explains why global RMSD alone can overstate differences in the individual chain folds. The [ubiquitin comparison](ubiquitin_structure_comparison_241026.md) uses the experimental 1UBQ structure and exact-sequence domains from the latest public AlphaFold Database versions retrieved on the analysis date. These are limited structural checks, not a broad accuracy ranking of the two trained models.
