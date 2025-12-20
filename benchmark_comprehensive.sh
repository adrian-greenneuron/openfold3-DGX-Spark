#!/bin/bash
# Comprehensive CUDA 12.8 vs 13.0 Benchmark Script
# Runs all 6 tests, interleaved, with memory tracking

set -e

QUERIES="query_ubiquitin.json query_homomer.json query_dna_ptm.json query_multimer.json query_protein_ligand.json query_protein_ligand_multiple.json"
CUDA13_IMAGE="openfold3-spark:cuda13-fixed"
CUDA128_IMAGE="openfold3-spark:latest"
OUTPUT_DIR="$(pwd)/output"
RESULTS_FILE="$(pwd)/benchmark_results.txt"

# Function to run a single benchmark with memory monitoring
run_benchmark() {
    local IMAGE=$1
    local QUERY=$2
    local LABEL=$3
    
    echo "--- $LABEL: $QUERY ---"
    
    # Start GPU memory monitoring in background
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > /tmp/gpu_mem_$$.log 2>/dev/null &
    GPU_MON_PID=$!
    
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
    
    # Stop GPU monitoring
    kill $GPU_MON_PID 2>/dev/null || true
    wait $GPU_MON_PID 2>/dev/null || true
    
    # Get max GPU memory
    MAX_GPU_MEM=$(sort -n /tmp/gpu_mem_$$.log 2>/dev/null | tail -1 || echo "N/A")
    rm -f /tmp/gpu_mem_$$.log
    
    # Extract inference time from output
    INFERENCE=$(echo "$OUTPUT" | grep -oP 'Predicting.*?\K\d+:\d+:\d+' | head -1 || echo "N/A")
    
    echo "  Inference: $INFERENCE"
    echo "  Total: ${TOTAL}s"
    echo "  GPU Mem: ${MAX_GPU_MEM} MiB"
    echo ""
    
    # Log to results file
    echo "$LABEL,$QUERY,$INFERENCE,${TOTAL}s,${MAX_GPU_MEM}MiB" >> $RESULTS_FILE
}

# Clear previous results
echo "Label,Query,Inference,Total,MaxGPU" > $RESULTS_FILE

echo "=========================================="
echo " COMPREHENSIVE CUDA BENCHMARK"
echo " $(date)"
echo "=========================================="
echo ""

# Round 1: CUDA 13
echo "=== ROUND 1: CUDA 13.0 ==="
for q in $QUERIES; do
    run_benchmark $CUDA13_IMAGE $q "CUDA13-R1"
done

# Round 1: CUDA 12.8
echo "=== ROUND 1: CUDA 12.8 ==="
for q in $QUERIES; do
    run_benchmark $CUDA128_IMAGE $q "CUDA128-R1"
done

# Round 2: CUDA 12.8
echo "=== ROUND 2: CUDA 12.8 ==="
for q in $QUERIES; do
    run_benchmark $CUDA128_IMAGE $q "CUDA128-R2"
done

# Round 2: CUDA 13
echo "=== ROUND 2: CUDA 13.0 ==="
for q in $QUERIES; do
    run_benchmark $CUDA13_IMAGE $q "CUDA13-R2"
done

echo "=========================================="
echo " BENCHMARK COMPLETE"
echo "=========================================="
echo "Results saved to: $RESULTS_FILE"
cat $RESULTS_FILE
