#!/bin/bash
# =============================================================================
# OpenFold3 Benchmark Script
# =============================================================================
# Runs benchmarks on CUDA 13, CUDA 12.8, or both (interleaved comparison).
#
# Usage:
#   ./benchmark_comprehensive.sh                    # Run both, 2 iterations
#   ./benchmark_comprehensive.sh cuda13             # Run CUDA 13 only, 1 iteration
#   ./benchmark_comprehensive.sh cuda12 3           # Run CUDA 12.8, 3 iterations
#   ./benchmark_comprehensive.sh both 2             # Run both, 2 iterations each
#
# Environment variables:
#   CUDA13_IMAGE  - Docker image for CUDA 13 (default: openfold3-spark:latest)
#   CUDA128_IMAGE - Docker image for CUDA 12.8 (default: openfold3-spark:cuda12)
# =============================================================================

set -e

# Configuration
QUERIES="query_ubiquitin.json query_homomer.json query_dna_ptm.json query_multimer.json query_protein_ligand.json query_protein_ligand_multiple.json"
CUDA13_IMAGE="${CUDA13_IMAGE:-openfold3-spark:latest}"
CUDA128_IMAGE="${CUDA128_IMAGE:-openfold3-spark:cuda12}"
OUTPUT_DIR="$(pwd)/output"
RESULTS_FILE="$(pwd)/benchmark_results.txt"

# Parse arguments
MODE="${1:-both}"
ITERATIONS="${2:-2}"

# Validate mode
case "$MODE" in
    cuda13|cuda12|both)
        ;;
    -h|--help|help)
        echo "Usage: $0 [MODE] [ITERATIONS]"
        echo ""
        echo "Arguments:"
        echo "  MODE        cuda13, cuda12, or both (default: both)"
        echo "  ITERATIONS  Number of rounds to run (default: 2)"
        echo ""
        echo "Examples:"
        echo "  $0                  # Run both versions, 2 iterations each"
        echo "  $0 cuda13           # Run CUDA 13 only, 1 iteration"
        echo "  $0 cuda12 3         # Run CUDA 12.8, 3 iterations"
        echo "  $0 both 4           # Run both, 4 iterations (interleaved)"
        echo ""
        echo "Environment variables:"
        echo "  CUDA13_IMAGE  - Docker image for CUDA 13 (default: openfold3-spark:latest)"
        echo "  CUDA128_IMAGE - Docker image for CUDA 12.8 (default: openfold3-spark:cuda12)"
        exit 0
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Run '$0 --help' for usage"
        exit 1
        ;;
esac

# Validate iterations
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: ITERATIONS must be a positive integer"
    exit 1
fi

# Function to run a single benchmark
run_benchmark() {
    local IMAGE=$1
    local QUERY=$2
    local LABEL=$3
    
    echo "--- $LABEL: $QUERY ---"
    
    START=$(date +%s)
    
    # Run inference and capture output
    OUTPUT=$(docker run --gpus all --ipc=host --shm-size=64g \
        -v $OUTPUT_DIR:/output \
        $IMAGE \
        run_openfold predict \
        --query_json=/opt/openfold-3/examples/example_inference_inputs/$QUERY \
        --output_dir=/output \
        --use_templates=false 2>&1)
    
    END=$(date +%s)
    TOTAL=$((END - START))
    
    # Extract inference time from output
    INFERENCE=$(echo "$OUTPUT" | grep -oP 'Predicting.*?\K\d+:\d+:\d+' | head -1 || echo "N/A")
    
    echo "  Inference: $INFERENCE"
    echo "  Total: ${TOTAL}s"
    echo ""
    
    # Log to results file
    echo "$LABEL,$QUERY,$INFERENCE,${TOTAL}s" >> $RESULTS_FILE
}

# Run CUDA 13 benchmarks
run_cuda13() {
    local ROUND=$1
    echo "=== CUDA 13.0 (Round $ROUND) ==="
    for q in $QUERIES; do
        run_benchmark $CUDA13_IMAGE $q "CUDA13-R$ROUND"
    done
}

# Run CUDA 12.8 benchmarks
run_cuda12() {
    local ROUND=$1
    echo "=== CUDA 12.8 (Round $ROUND) ==="
    for q in $QUERIES; do
        run_benchmark $CUDA128_IMAGE $q "CUDA128-R$ROUND"
    done
}

# Initialize results file
echo "Label,Query,Inference,Total" > $RESULTS_FILE

echo "=========================================="
echo " OpenFold3 Benchmark"
echo " Mode: $MODE | Iterations: $ITERATIONS"
echo " $(date)"
echo "=========================================="
echo ""
echo "CUDA 13 Image: $CUDA13_IMAGE"
echo "CUDA 12.8 Image: $CUDA128_IMAGE"
echo ""

case "$MODE" in
    cuda13)
        for i in $(seq 1 $ITERATIONS); do
            run_cuda13 $i
        done
        ;;
    cuda12)
        for i in $(seq 1 $ITERATIONS); do
            run_cuda12 $i
        done
        ;;
    both)
        # Interleaved: alternates between versions to account for thermal effects
        for i in $(seq 1 $ITERATIONS); do
            if [ $((i % 2)) -eq 1 ]; then
                run_cuda13 $i
                run_cuda12 $i
            else
                run_cuda12 $i
                run_cuda13 $i
            fi
        done
        ;;
esac

echo "=========================================="
echo " BENCHMARK COMPLETE"
echo "=========================================="
echo "Results saved to: $RESULTS_FILE"
echo ""
cat $RESULTS_FILE
