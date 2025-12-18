#!/bin/bash
# OpenFold3 Benchmark Suite for DGX Spark

echo "=============================================="
echo " OpenFold3 Benchmark Suite - DGX Spark"
echo "=============================================="
echo "CPU Cores: $(nproc)"
echo "RAM: $(free -h | awk '/Mem:/ {print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

declare -A RESULTS

for query in query_ubiquitin.json query_homomer.json query_multimer.json query_dna_ptm.json query_protein_ligand.json query_protein_ligand_multiple.json; do
    echo "=========================================="
    echo "Running: $query"
    echo "=========================================="
    START=$(date +%s)
    
    docker run --gpus all --ipc=host --shm-size=64g \
        -e OMP_NUM_THREADS=16 \
        -e MAX_JOBS=16 \
        -v $(pwd)/output:/output \
        openfold3-spark:latest \
        run_openfold predict \
        --query_json=/opt/openfold-3/examples/example_inference_inputs/$query \
        --output_dir=/output \
        --use_templates=false 2>&1 | grep -E "(PREDICTION SUMMARY|Successful|Failed|Predicting|evoformer_attn)"
    
    END=$(date +%s)
    DURATION=$((END - START))
    RESULTS[$query]=$DURATION
    echo ">>> Duration: ${DURATION}s"
    echo ""
done

echo "=============================================="
echo " BENCHMARK RESULTS - DGX Spark (GB10)"
echo "=============================================="
printf "%-35s %10s\n" "Example" "Time (s)"
echo "----------------------------------------------"
for query in query_ubiquitin.json query_homomer.json query_multimer.json query_dna_ptm.json query_protein_ligand.json query_protein_ligand_multiple.json; do
    printf "%-35s %10s\n" "$query" "${RESULTS[$query]}"
done
echo "=============================================="
