#!/usr/bin/env bash
set -euo pipefail

cd "${RELIAMOL_OUTPUT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${RELIAMOL_PYTHON:-python}"
FRAME=outputs/bib_protocol_sensitivity_v0/protocol_sensitivity_feature_frame.csv
PLAN=outputs/diffsbdd_crossdocked_vina_gnina_100pocket_v0/reports/receptor_preparation.csv
NUM_SHARDS=24
mkdir -p outputs/bib_protocol_sensitivity_v0/logs

run_worker() {
  local gpu=$1
  for epsilon in 0.10 0.40; do
    local tag=${epsilon/./p}
    local out=outputs/bib_protocol_sensitivity_v0/epsilon_${tag}
    mkdir -p "$out"
    for ((shard=gpu; shard<NUM_SHARDS; shard+=6)); do
      "$PY" scripts/revision_r1_openmm_relaxation.py \
        --feature-frame "$FRAME" \
        --receptor-plan "$PLAN" \
        --output-dir "$out" \
        --shard-index "$shard" \
        --num-shards "$NUM_SHARDS" \
        --platform CUDA \
        --device-index "$gpu" \
        --receptor-model steric_dummy \
        --receptor-epsilon "$epsilon" \
        --max-iterations 500 \
        --tolerance 10.0 \
        --candidate-timeout-sec 180 \
        --resume \
        > "outputs/bib_protocol_sensitivity_v0/logs/eps_${tag}_shard_${shard}.log" 2>&1
    done
  done
}

for gpu in 0 1 2 3 4 5; do
  run_worker "$gpu" &
done
wait
"$PY" research_data/ReliaMol3D_Benchmark_v1.0/scripts/analyze_bib_protocol_sensitivity.py
