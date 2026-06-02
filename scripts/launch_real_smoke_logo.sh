#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
PYTHON="${PYTHON:-$ROOT/.conda_env/bin/python}"

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
GENERATORS=(NativeNear DriftGen ClashGen ScoreHackGen)
CANDIDATES="${CANDIDATES:-outputs/real_smoke_crossdocked/candidate_labels.csv}"
OUT_ROOT="${OUT_ROOT:-outputs/real_smoke_crossdocked_logo}"

mkdir -p logs "$OUT_ROOT" reports

job_id=0
for generator in "${GENERATORS[@]}"; do
  gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_real_smoke_logo.py \
    --config configs/real_smoke_crossdocked.yaml \
    --candidates "$CANDIDATES" \
    --heldout-generator "$generator" \
    --output-dir "$OUT_ROOT/$generator" \
    > "logs/logo_${generator}.log" 2>&1 &
  job_id=$((job_id + 1))
done

wait
"$PYTHON" scripts/summarize_logo.py --input "$OUT_ROOT" --output reports/real_smoke_crossdocked_logo_summary.csv

MIXED_OUT_ROOT="${MIXED_OUT_ROOT:-outputs/real_smoke_crossdocked_logo_mixed_native}"
mkdir -p "$MIXED_OUT_ROOT"
MIXED_GENERATORS=(DriftGen ClashGen ScoreHackGen)

job_id=0
for generator in "${MIXED_GENERATORS[@]}"; do
  gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_real_smoke_logo.py \
    --config configs/real_smoke_crossdocked.yaml \
    --candidates "$CANDIDATES" \
    --heldout-generator "$generator" \
    --mixed-native-anchor \
    --output-dir "$MIXED_OUT_ROOT/$generator" \
    > "logs/logo_mixed_native_${generator}.log" 2>&1 &
  job_id=$((job_id + 1))
done

wait
"$PYTHON" scripts/summarize_logo.py --input "$MIXED_OUT_ROOT" --output reports/real_smoke_crossdocked_logo_mixed_native_summary.csv
