#!/bin/bash
# =============================================================================
# OpenFold3 Docker Build Script with JIT Pre-compilation
# =============================================================================
# This script builds the OpenFold3 Docker image with pre-compiled CUDA kernels
# by using a "docker commit" workflow that allows GPU access during the build.
#
# Usage: ./build.sh [IMAGE_NAME:TAG]
# Default: openfold3-spark:latest
# =============================================================================

set -e

IMAGE_NAME="${1:-openfold3-spark:latest}"
BASE_IMAGE="${IMAGE_NAME%:*}:base"
CONTAINER_NAME="openfold3-warmup-$$"

echo "=============================================="
echo " OpenFold3 Build with JIT Pre-compilation"
echo "=============================================="
echo "Target image: ${IMAGE_NAME}"
echo ""

# Step 1: Build base image (without GPU)
echo "[1/4] Building base image..."
docker build -t "${BASE_IMAGE}" .

# Step 2: Run container with GPU to pre-compile kernels
echo ""
echo "[2/4] Running inference to pre-compile CUDA kernels..."
echo "      (This may take 2-3 minutes on first run)"
docker run --gpus all --ipc=host --shm-size=32g \
    --name "${CONTAINER_NAME}" \
    "${BASE_IMAGE}" \
    python3 -m openfold3.run_openfold predict \
    --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
    --output_dir=/tmp/warmup_output \
    --use_templates=false

# Step 3: Commit the container to the final image
echo ""
echo "[3/4] Committing container with compiled kernels..."
docker commit \
    --change 'CMD ["/bin/bash"]' \
    --change 'WORKDIR /opt/openfold-3' \
    "${CONTAINER_NAME}" "${IMAGE_NAME}"

# Step 4: Cleanup
echo ""
echo "[4/4] Cleaning up..."
docker rm "${CONTAINER_NAME}"
# Optionally remove the base image to save space
# docker rmi "${BASE_IMAGE}"

echo ""
echo "=============================================="
echo " Build Complete!"
echo "=============================================="
echo "Image: ${IMAGE_NAME}"
echo ""
echo "Run inference with:"
echo "  docker run --gpus all --ipc=host --shm-size=32g \\"
echo "      -v \$(pwd)/output:/output \\"
echo "      ${IMAGE_NAME} \\"
echo "      run_openfold predict \\"
echo "      --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \\"
echo "      --output_dir=/output"
echo ""
