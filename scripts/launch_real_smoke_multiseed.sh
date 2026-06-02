#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
PYTHON="${PYTHON:-$ROOT/.conda_env/bin/python}"

SEEDS=(${SEEDS:-11 22 33})
GPUS="${GPUS:-0,1,2}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"

mkdir -p logs configs/generated outputs/real_smoke_crossdocked_multiseed reports

job_id=0
for seed in "${SEEDS[@]}"; do
  cfg="configs/generated/real_smoke_crossdocked_seed${seed}.yaml"
  "$PYTHON" - "$seed" "$cfg" <<'PY'
import sys
import yaml

seed = int(sys.argv[1])
out = sys.argv[2]
with open("configs/real_smoke_crossdocked.yaml", "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["experiment"]["seed"] = seed
cfg["experiment"]["name"] = f"crossdocked_tensor_real_smoke_seed{seed}"
cfg["experiment"]["output_dir"] = f"outputs/real_smoke_crossdocked_multiseed/seed{seed}"
with open(out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY
  gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_real_smoke_crossdocked.py --config "$cfg" \
    > "logs/real_smoke_crossdocked_seed${seed}.log" 2>&1 &
  job_id=$((job_id + 1))
done

wait
"$PYTHON" scripts/summarize_real_smoke_multiseed.py \
  --input outputs/real_smoke_crossdocked_multiseed \
  --output reports/real_smoke_crossdocked_multiseed_summary.csv

