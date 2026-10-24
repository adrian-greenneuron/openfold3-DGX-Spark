#!/usr/bin/env bash
# Build the unmodified official OpenFold3 Dockerfile and locked environment.
set -euo pipefail
OF3_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$OF3_REPO_DIR/versions.env"
OF3_IMAGE="${1:-openfold3-official-spark:0.5.0}"
OF3_BUILD_ROOT="${OF3_BUILD_ROOT:-$OF3_REPO_DIR/.build}"
OF3_SOURCE_DIR="$OF3_BUILD_ROOT/openfold3-$OF3_SOURCE_COMMIT"
export DOCKER_BUILDKIT=1
mkdir -p "$OF3_BUILD_ROOT"
if [[ ! -d "$OF3_SOURCE_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$OF3_SOURCE_URL" "$OF3_SOURCE_DIR"
else
    if [[ -n "$(git -C "$OF3_SOURCE_DIR" status --porcelain --untracked-files=all)" ]]; then
        echo "Refusing to build from a modified source checkout: $OF3_SOURCE_DIR" >&2
        exit 1
    fi
fi
git -C "$OF3_SOURCE_DIR" fetch origin "$OF3_SOURCE_COMMIT"
git -C "$OF3_SOURCE_DIR" checkout --detach "$OF3_SOURCE_COMMIT"
[[ "$(git -C "$OF3_SOURCE_DIR" rev-parse HEAD)" == "$OF3_SOURCE_COMMIT" ]]
OF3_BUILD_OPTIONS=(--pull)
if [[ "${OF3_NO_CACHE:-false}" == true ]]; then OF3_BUILD_OPTIONS+=(--no-cache); fi
docker build "${OF3_BUILD_OPTIONS[@]}" \
    -f "$OF3_SOURCE_DIR/docker/Dockerfile.pixi" \
    --build-arg "BASE_IMAGE=$OF3_BASE_IMAGE" \
    --build-arg "PIXI_VERSION=$OF3_PIXI_VERSION" \
    --build-arg "PIXI_ENV=$OF3_PIXI_ENV" \
    --target devel -t "$OF3_IMAGE" "$OF3_SOURCE_DIR"
printf 'Built %s from official source %s\n' "$OF3_IMAGE" "$OF3_SOURCE_COMMIT"
