#!/usr/bin/env bash
# Separate candidates: never replace the established build.sh default.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export DOCKER_BUILDKIT=1

PROFILE="${1:-legacy}"
case "$PROFILE" in
    legacy)
        IMAGE_NAME="${2:-openfold3-spark:legacy-modern}"
        docker build --pull -f Dockerfile.legacy-modern -t "$IMAGE_NAME" .
        ;;
    current)
        IMAGE_NAME="${2:-openfold3-spark:current-upstream}"
        SOURCE_REVISION=a83cb119f58a86306f9d66ca528ef2a0fffad250
        BUILD_ROOT="${MODERN_BUILD_ROOT:-$(pwd)/.modern-build}"
        SOURCE_DIR="$BUILD_ROOT/openfold3-$SOURCE_REVISION"
        mkdir -p "$BUILD_ROOT"
        if [ ! -d "$SOURCE_DIR/.git" ]; then
            git clone https://github.com/aqlaboratory/openfold-3.git "$SOURCE_DIR"
        fi
        git -C "$SOURCE_DIR" checkout --detach "$SOURCE_REVISION"
        # All dependency resolution and activation come from upstream's lockfile
        # and Dockerfile. BASE_IMAGE is an upstream-supported build argument.
        docker build --pull \
            -f "$SOURCE_DIR/docker/Dockerfile.pixi" \
            --build-arg PIXI_ENV=openfold3-cuda13 \
            --build-arg PIXI_VERSION=v0.80.0 \
            --build-arg BASE_IMAGE=ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956 \
            --target test -t "$IMAGE_NAME" "$SOURCE_DIR"
        ;;
    -h|--help)
        echo "Usage: ./build-modern.sh {legacy|current} [IMAGE:TAG]"
        echo "legacy: original model, stock DeepSpeed, build-time attention compilation"
        echo "current: pinned official OpenFold3/Pixi Dockerfile and current model"
        exit 0
        ;;
    *)
        echo "Unknown profile: $PROFILE (expected legacy or current)" >&2
        exit 2
        ;;
esac
echo "Built $IMAGE_NAME; the established working image and default are unchanged."
