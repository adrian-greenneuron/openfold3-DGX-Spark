# Official OpenFold3: fresh build and inference verification

24 October 2026

The clean `official/openfold3-spark` branch built successfully from a new GitHub
clone and ran the official newer model on DGX Spark. Verification used repository
commit `94610d21094968193feb4ecf134b8d6be4fbb314`. Subsequent changes add this
validation record and documentation only.

## Build

```bash
OF3_NO_CACHE=true ./build.sh openfold3-official-spark:clean-241026
```

The source checkout was initially absent. The build downloaded official OpenFold3
revision `a83cb119f58a86306f9d66ca528ef2a0fffad250`, then used its unmodified
Dockerfile and locked `openfold3-cuda13` environment. Docker used `--no-cache` and
`--pull`. The build succeeded in **334.08 seconds (5 minutes 34 seconds)**.

No historical OpenFold3 image, checkpoint, compiled extension or neighbouring
source checkout was an input to this build. Docker's normal handling of the
pinned Ubuntu base image still applies; `--no-cache` rebuilds the Dockerfile steps
and does not require deleting existing base-image layers.

The resulting image was:

- Tag: `openfold3-official-spark:clean-241026`
- ID: `sha256:c5c9c9ca44369e655dca78f720d5be7a22f6da246465178632fc2afa156d8657`
- Logical size: **9.08 GB**; model parameters are stored separately.

All **350 Python source files** in the installed OpenFold3 source tree matched
the pinned upstream checkout by SHA-256.

| Runtime component | Version |
|---|---|
| OpenFold3 | 0.5.0 |
| Model | OpenBind-0 |
| Python | 3.14.6 |
| PyTorch | 2.12.1 |
| PyTorch CUDA | 13.0 |
| DeepSpeed, installed but disabled | 0.19.2 |

## Fresh model setup and inference

The launcher was tested with a new, empty model cache and output directory:

```bash
OF3_IMAGE=openfold3-official-spark:clean-241026 \
OF3_CACHE_DIR=/path/to/empty-model-cache \
OF3_OUTPUT_DIR=/path/to/new-output \
./run.sh
```

Official setup downloaded the OpenBind checkpoint again and prepared its
chemical-component dictionary. The checkpoint was **2,287,872,989 bytes** and
matched the expected SHA-256:

```text
bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4
```

The bundled ubiquitin example completed with **one successful query and zero
failures**. It produced five model CIFs and ten corresponding confidence JSON
files. Each model contained the expected 76 C-alpha residues. All atom
coordinates and numerical confidence values were finite.

The effective configuration retained upstream attention defaults: DeepSpeed
disabled, Triton triangle kernels enabled, 200 diffusion steps and five samples.
No model, checkpoint, precision or attention wrapper was applied. The example
explicitly disabled template and MSA-server requests, using the official fallback
alignment. It is therefore not the same input-alignment condition as earlier
saved-MSA performance measurements.

First setup plus prediction took **314.93 seconds (5 minutes 15 seconds)**,
including downloads, checksum verification, startup and first-use compilation.
The model's internal inference timer was **9.45 seconds**. These are single
installation-check timings, **not repeated performance benchmarks**.

The host driver was not upgraded. Existing images were retained. The upstream
Triton implementation still warned that its NVIDIA configuration is untuned;
this did not prevent successful inference.

## Scope

This verifies a fresh build, checkpoint download, launcher and complete finite
outputs for one small example. It does not establish prediction accuracy,
repeatability, maximum complex size or performance on other inputs. Source and
checkpoint identities, output hashes, configuration checks and timings are in
the [machine-readable validation record](official_build_validation_241026.json).
