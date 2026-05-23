#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.conda_env/bin/python}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export RELIAMOL_N_JOBS="${RELIAMOL_N_JOBS:-8}"

GPUS="${GPUS:-0,1,2,3,4,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
SEEDS=(1 2 3 4 5)
SPLITS=(random unseen_family)
GENERATORS=(Pocket2Mol TargetDiff DiffSBDD DecompDiff)

mkdir -p logs outputs/proxy_v0

job_id=0
for seed in "${SEEDS[@]}"; do
  for split in "${SPLITS[@]}"; do
    gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_proxy_experiments.py \
      --config configs/proxy.yaml --seed "$seed" --split "$split" \
      > "logs/${split}_seed${seed}.log" 2>&1 &
    job_id=$((job_id + 1))
  done
  for generator in "${GENERATORS[@]}"; do
    gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_proxy_experiments.py \
      --config configs/proxy.yaml --seed "$seed" --split unseen_generator --heldout-generator "$generator" \
      > "logs/unseen_generator_${generator}_seed${seed}.log" 2>&1 &
    job_id=$((job_id + 1))
  done
done

wait
"$PYTHON" scripts/summarize_outputs.py --input outputs/proxy_v0 --output reports/proxy_v0_summary.csv
