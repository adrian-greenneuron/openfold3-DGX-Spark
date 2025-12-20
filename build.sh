#!/bin/bash
# =============================================================================
# OpenFold3 Docker Build Script with JIT Pre-compilation
# =============================================================================
# This script builds the OpenFold3 Docker image with pre-compiled CUDA kernels
# by using a "docker commit" workflow that allows GPU access during the build.
#
# Usage: ./build.sh [IMAGE_NAME:TAG]
# Default: openfold3-spark:latest
#
# Environment Variables:
#   SKIP_WARMUP=true  - Skip the pre-compilation step (faster build, slower first run)
# =============================================================================

# Enable BuildKit for cache mounts (required for NGC 25.11 / CUDA 13.0)
export DOCKER_BUILDKIT=1

set -e

IMAGE_NAME="${1:-openfold3-spark:latest}"
BASE_IMAGE="${IMAGE_NAME%:*}:base"
CONTAINER_NAME="openfold3-warmup-$$"
SKIP_WARMUP="${SKIP_WARMUP:-false}"
START_TIME=$(date +%s)

# Cleanup function - ensures container is removed even on failure
cleanup() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Cleaning up container ${CONTAINER_NAME}..."
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=============================================="
echo " OpenFold3 Build with JIT Pre-compilation"
echo "=============================================="
echo "Target image: ${IMAGE_NAME}"
echo "Skip warmup:  ${SKIP_WARMUP}"
echo ""

# Step 1: Build base image (without GPU)
echo "[1/4] Building base image..."
docker build -t "${BASE_IMAGE}" .

if [ "$SKIP_WARMUP" = "true" ]; then
    echo ""
    echo "[2/4] Skipping pre-compilation (SKIP_WARMUP=true)"
    echo "[3/4] Tagging base image as final..."
    docker tag "${BASE_IMAGE}" "${IMAGE_NAME}"
    echo "[4/4] No cleanup needed."
else
    # Check GPU availability
    echo ""
    echo "[2/4] Checking GPU availability..."
    if ! docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi &>/dev/null; then
        echo "ERROR: GPU not available or nvidia-docker not configured."
        echo "       Run with SKIP_WARMUP=true to build without pre-compilation."
        exit 1
    fi
    echo "      GPU detected successfully."

    # Run warmup with GPU
    echo ""
    echo "      Running inference to pre-compile CUDA kernels..."
    echo "      (This may take 2-3 minutes on first run)"
    if ! docker run --gpus all --ipc=host --shm-size=64g \
        --name "${CONTAINER_NAME}" \
        "${BASE_IMAGE}" \
        python3 -m openfold3.run_openfold predict \
        --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
        --output_dir=/tmp/warmup_output \
        --use_templates=false; then
        echo "ERROR: Warmup inference failed!"
        exit 1
    fi

    # Commit the container to the final image
    echo ""
    echo "[3/4] Committing container with compiled kernels..."
    docker commit \
        --change 'CMD ["/bin/bash"]' \
        --change 'WORKDIR /opt/openfold-3' \
        "${CONTAINER_NAME}" "${IMAGE_NAME}"

    # Cleanup is handled by trap, but we can remove container explicitly
    echo ""
    echo "[4/4] Cleaning up..."
    docker rm "${CONTAINER_NAME}"
    # Optionally remove the base image to save space
    # docker rmi "${BASE_IMAGE}"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo " Build Complete!"
echo "=============================================="
echo "Image:    ${IMAGE_NAME}"
echo "Duration: ${DURATION} seconds"
echo ""
echo "Run inference with:"
echo "  docker run --gpus all --ipc=host --shm-size=64g \\"
echo "      -v \$(pwd)/output:/output \\"
echo "      ${IMAGE_NAME} \\"
echo "      run_openfold predict \\"
echo "      --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \\"
echo "      --output_dir=/output"
echo ""
