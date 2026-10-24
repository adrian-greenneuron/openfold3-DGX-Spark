# Official OpenFold3 on DGX Spark

This development branch builds and runs the newer official OpenFold3 model on
NVIDIA DGX Spark. It uses the **unmodified upstream Dockerfile, Pixi lockfile and
inference code**. The model is **OpenBind-0**, supplied with OpenFold3 v0.5.0.

For current models and general instructions, start with the
[official OpenFold3 repository](https://github.com/aqlaboratory/openfold-3).
This branch provides a small, version-pinned Spark build and launch recipe.
The repository's `main` branch retains the original deployment and its original
model weights.

## Requirements

A DGX Spark with Docker, NVIDIA Container Toolkit and working GPU access from
containers. The host needs Bash and Git; Python and CUDA dependencies are
installed inside the image. Internet access is needed for the source, packages,
model parameters and chemical-component dictionary. The model parameters alone
are approximately 2.29 GB.

## Build

```bash
git clone --branch official/openfold3-spark --single-branch \
    https://github.com/adrian-greenneuron/openfold3-DGX-Spark.git
cd openfold3-DGX-Spark
./build.sh
```

The script downloads the pinned official source and builds its
`docker/Dockerfile.pixi`, using the `openfold3-cuda13` environment. It does not
use an existing local OpenFold3 image, patch DeepSpeed, or use `docker commit`.
Docker can reuse build cache by default. To rebuild all build steps:

```bash
OF3_NO_CACHE=true ./build.sh openfold3-official-spark:clean-check
```

## Run the example

```bash
./run.sh
```

This runs the bundled 76-residue ubiquitin example with MSA-server and template
queries disabled. It uses the official fallback alignment rather than contacting
an MSA service. The other model settings remain upstream defaults. The first run
downloads the model parameters and checks their SHA-256; they are kept in
`.openfold3/` for reuse. Results are saved in `output/`.

After building under a different image tag, select that tag explicitly:

```bash
OF3_IMAGE=openfold3-official-spark:clean-check ./run.sh
```

## Run your own query

Place your query JSON and any saved alignments under `input/`. Paths within the
query must use their container location under `/input/`.

```bash
./run.sh predict --query_json=/input/query.json --output_dir=/output
```

This forwards arguments to the official `run_openfold` CLI. In this example,
MSA-server and template behaviour follows the upstream defaults; use the
[official documentation](https://openfold-3.readthedocs.io/en/latest/) to configure
your analysis. You can choose other host directories:

```bash
OF3_INPUT_DIR=/path/to/inputs \
OF3_OUTPUT_DIR=/path/to/results \
OF3_CACHE_DIR=/path/to/model-cache \
./run.sh predict --query_json=/input/query.json --output_dir=/output
```

The input directory is read-only inside the container. Outputs and model cache
are writable. Docker runs as the upstream image's default user, so files may be
owned by root on the host. A new container may download the chemical-component
dictionary again; it is managed by the official setup command inside the image.

## Model and settings

The build is pinned in [versions.env](versions.env). This is a newer trained model
than the one on `main`; its predictions are not expected to reproduce the old
checkpoint. The launcher retains the official attention choices, precision,
recycles and diffusion sampling. It sets the native Spark architecture target
(`12.1`), 16 CPU threads and the expandable CUDA allocator.

No experimental diffusion cache or original-style acceleration wrapper is
included. Repeated predictions are not guaranteed to be byte-identical, and the
maximum complex size that fits in memory has not been established.

See [reproducibility and validation](docs/reproducibility.md) for the exact build
inputs and verification procedure.
