# Official-model build and verification

The recipe downloads official OpenFold3 source at
`a83cb119f58a86306f9d66ca528ef2a0fffad250` (v0.5.0) and uses its unchanged
`docker/Dockerfile.pixi`, `pixi.toml` and `pixi.lock`. The build selects
`openfold3-cuda13`, Pixi v0.80.0 and the Ubuntu 22.04 base digest in
[versions.env](../versions.env). The deployed model is OpenBind-0, checkpoint
`of3-ob-2025-06-30-174k.pt`; its SHA-256 is in
[checkpoint.sha256](../runtime/checkpoint.sha256).

The source revision, base digest and locked application dependencies are pinned.
Ubuntu package repositories, package download services and the upstream
chemical-component dictionary remain external dependencies. A successful rebuild
does not imply identical Docker image bytes or universally deterministic
predictions.

## Fresh verification

Use a fresh checkout, a new image tag and an empty model cache. This requires no
source files, checkpoints or compiled extensions from the historical deployment.

```bash
OF3_NO_CACHE=true ./build.sh openfold3-official-spark:verification
OF3_IMAGE=openfold3-official-spark:verification \
OF3_CACHE_DIR="$PWD/verification-cache" \
OF3_OUTPUT_DIR="$PWD/verification-output" \
./run.sh
```

Check the command's exit status, its zero-failed-query summary and all five model
CIFs with their corresponding confidence files. Successful execution checks
installation and output completeness; it is not a biological accuracy benchmark.
The example disables remote MSA and template requests; it does not shorten the
model's normal diffusion schedule or reduce the number of output samples.

Normal runs use the upstream inference code directly. No DeepSpeed source patch,
checkpoint conversion, precision override, diffusion cache, deterministic wrapper
or attention monkeypatch is applied. The entry point only runs official setup,
checks the downloaded checkpoint and invokes the official CLI.

To preserve a verification container for inspection, set
`OF3_KEEP_CONTAINER=true OF3_CONTAINER_NAME=your-verification-name`. Existing images
are not removed by either script. New containers are otherwise removed when they
exit; bound outputs and model parameters remain on the host.

Validation of this branch is in progress; a dated record will be added after the
fresh build and inference check finish.
