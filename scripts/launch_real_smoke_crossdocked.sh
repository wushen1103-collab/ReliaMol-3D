#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
PYTHON="${PYTHON:-$ROOT/.conda_env/bin/python}"

mkdir -p logs outputs/real_smoke_crossdocked reports
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_real_smoke_crossdocked.py \
  --config configs/real_smoke_crossdocked.yaml \
  > logs/real_smoke_crossdocked.log 2>&1

cp outputs/real_smoke_crossdocked/metrics.csv reports/real_smoke_crossdocked_metrics.csv
cp outputs/real_smoke_crossdocked/reranking.csv reports/real_smoke_crossdocked_reranking.csv
cp outputs/real_smoke_crossdocked/failure_diagnosis.csv reports/real_smoke_crossdocked_failure_diagnosis.csv
