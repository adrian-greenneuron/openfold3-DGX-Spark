#!/usr/bin/env bash
# Run the official setup and CLI. No model or attention wrappers are installed.
set -euo pipefail
OF3_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: ./run.sh [run_openfold arguments]
With no arguments, predict the bundled ubiquitin example without MSA/template servers.
For custom input: ./run.sh predict --query_json=/input/query.json --output_dir=/output
Host directories: OF3_INPUT_DIR, OF3_OUTPUT_DIR, OF3_CACHE_DIR.
Image: OF3_IMAGE (default openfold3-official-spark:0.5.0).
HELP
    exit 0
fi
OF3_IMAGE="${OF3_IMAGE:-openfold3-official-spark:0.5.0}"
OF3_INPUT_DIR="${OF3_INPUT_DIR:-$OF3_REPO_DIR/input}"
OF3_OUTPUT_DIR="${OF3_OUTPUT_DIR:-$OF3_REPO_DIR/output}"
OF3_CACHE_DIR="${OF3_CACHE_DIR:-$OF3_REPO_DIR/.openfold3}"
mkdir -p "$OF3_INPUT_DIR" "$OF3_OUTPUT_DIR" "$OF3_CACHE_DIR"
OF3_INPUT_DIR="$(cd -- "$OF3_INPUT_DIR" && pwd)"
OF3_OUTPUT_DIR="$(cd -- "$OF3_OUTPUT_DIR" && pwd)"
OF3_CACHE_DIR="$(cd -- "$OF3_CACHE_DIR" && pwd)"
OF3_DOCKER_OPTIONS=(--gpus all --ipc=host --shm-size=64g)
if [[ "${OF3_KEEP_CONTAINER:-false}" != true ]]; then OF3_DOCKER_OPTIONS+=(--rm); fi
if [[ -n "${OF3_CONTAINER_NAME:-}" ]]; then OF3_DOCKER_OPTIONS+=(--name "$OF3_CONTAINER_NAME"); fi
if [[ $# == 0 ]]; then
    set -- predict \
        --query_json=/opt/openfold3/examples/example_inference_inputs/query_ubiquitin.json \
        --output_dir=/output --use_templates=false --use_msa_server=false
fi
exec docker run "${OF3_DOCKER_OPTIONS[@]}" \
    -e TORCH_CUDA_ARCH_LIST=12.1 \
    -e OMP_NUM_THREADS=16 -e MAX_JOBS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e OPENFOLD_CACHE=/root/.openfold3 \
    --mount "type=bind,src=$OF3_REPO_DIR/runtime,dst=/deployment,readonly" \
    --mount "type=bind,src=$OF3_INPUT_DIR,dst=/input,readonly" \
    --mount "type=bind,src=$OF3_OUTPUT_DIR,dst=/output" \
    --mount "type=bind,src=$OF3_CACHE_DIR,dst=/root/.openfold3" \
    "$OF3_IMAGE" bash /deployment/entrypoint.sh "$@"
