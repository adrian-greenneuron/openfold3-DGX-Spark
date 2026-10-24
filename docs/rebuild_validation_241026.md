# Rebuild and numerical verification — 24 October 2026

Validation used an NVIDIA DGX Spark (GB10). The original retained CUDA 13 image was compared with clean builds using the pinned inputs documented in [reproducibility.md](reproducibility.md). All existing images and tags were retained. Normal inference defaults, checkpoint and model source were unchanged.

## Final repository build

- `BUILD_NO_CACHE=true ./build.sh openfold3-repo-validation-241026:verified` completed successfully in **714 seconds**.
- The checkpoint was downloaded again and its SHA-256 verified. The build used public sources and repositories; no files or kernels were copied from the retained OpenFold3 image.
- The final image matched the retained source revision, all 283 recorded Python package versions and the checkpoint checksum. The checked-out upstream model source was unmodified.
- The offline warm-up compiled DeepSpeed attention, processed the ubiquitin query successfully and produced all five model/confidence triplets. The build script checked these outputs before committing the warmed image.
- The repository's opt-in deterministic wrapper then ran the offline ubiquitin example once in each image. All **15 prediction files were byte-for-byte identical**. Diagnostic wall times were 46.89 seconds for the retained image and 46.72 seconds for the rebuilt image; these are not a repeated performance benchmark.
- Eight focused CPU tests passed, including rejection of empty output directories, failed queries, missing confidence files, changed samples and altered predictions. The comparator also confirmed the previously retained exact MCL1 comparison below.

## Earlier clean-build and whole-model controls

The pinned clean-build candidate matched the retained image's OpenFold3 source revision, checkpoint checksum and all 283 recorded Python distribution versions. All six bundled example configurations ran successfully in both images. Default fast-mode outputs were not guaranteed to match exactly; repeated predictions within the retained image also varied.

For MCL1, all 40 recorded model-input leaves and initial CPU/CUDA RNG states matched between the retained and rebuilt images. Deterministic PyTorch operations alone did not remove the remaining differences when DeepSpeed attention stayed enabled. With standard attention as well, the model tensors matched and all **15 prediction files**, covering five diffusion samples, were byte-for-byte identical. This was a controlled comparison on saved identical MSAs, with templates disabled.

## Focused forward attention check

| Check | Result |
|---|---|
| Two upstream shapes × two dtypes × three seeds | 12/12 cases passed the version-matched forward tolerance, all 120 calls |
| BF16 sequence lengths 17, 63, 64, 65 and 737, with four bias/mask conditions | 20/20 cases passed, all 200 calls |
| Captured MCL1 template-attention call, ten identical-input repeats | Ten different DeepSpeed outputs |
| Captured MCL1, DeepSpeed relative L2 error against FP32 | 0.168% |
| Same input, ordinary BF16 reference relative L2 error against FP32 | 0.202% |
| Captured MCL1, largest DeepSpeed absolute error against FP32 | 0.02166 |
| FP32 reference repeatability | Bitwise identical throughout all 38 cases |

There were 38 cases and 380 DeepSpeed forward calls, plus independent reference-validation calls. Inputs were quantized identically for both backends, with attention scaling applied once. The upstream synthetic-test limits were 0.05 for BF16 and 0.01 for FP16. Gradients were not tested.

The actual MCL1 case exceeded the borrowed BF16 limit when compared against ordinary BF16 attention: maximum difference 0.109375. However, ordinary BF16 attention itself differed from FP32 by up to 0.106296, whereas DeepSpeed differed by 0.021660. Consequently this threshold failure does not establish that the DeepSpeed answer was less accurate. Most of its measured error was consistent with BF16 output precision.

In one pair of identical-input calls, about 0.00032% of output elements changed, with maximum difference 0.03125. Making biases contiguous or removing them did not eliminate variation. An independent PyTorch SDPA math calculation agreed with the explicit reference; comparison against FP64 on four selected captured rows gave maximum reference error about 0.00006061.

The tests establish numerical plausibility and a repeatability limitation in this installed stack. They do not identify an internal kernel cause, establish biological accuracy, qualify other GPUs or guarantee repeatability for arbitrary inputs. The [DeepSpeed v0.15.4 upstream test](https://github.com/deepspeedai/DeepSpeed/blob/v0.15.4/tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py) defines the synthetic reference and tolerances used here.
