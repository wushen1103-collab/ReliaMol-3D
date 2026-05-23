# Reproducibility Guide

This guide gives a practical route from public inputs to the manuscript evidence blocks.

## 1. Create the Environment

```bash
conda env create -f environment.yml
conda activate reliamol3d
pip install -e .
```

For CPU-only checks, `pip install -r requirements.txt && pip install -e .` is usually enough for scripts that do not call external docking tools.

## 2. Run a Lightweight Sanity Check

The proxy experiment uses synthetic/proxy data and does not require molecular files:

```bash
python scripts/run_proxy_experiments.py --config configs/proxy.yaml
python scripts/summarize_outputs.py --input outputs/proxy_v0 --output reports/proxy_v0_summary.csv
```

This verifies the Python package, training loop, metrics, and reranking summaries.

## 3. Reproduce the DiffSBDD Exact-CrossDocked Evidence

Prepare official DiffSBDD SDF files and CrossDocked receptor `.gninatypes` files, then run:

```bash
python scripts/audit_diffsbdd_crossdocked_gninatypes.py \
  --sdf-root /path/to/diffsbdd/crossdocked_fullatom_cond \
  --archive /path/to/CrossDocked2020_receptors.tgz \
  --index /path/to/tar_index.txt \
  --extract-dir /path/to/selected_receptors \
  --output-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0 \
  --workers 24

python scripts/run_diffsbdd_crossdocked_reliamol.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports \
  --output-dir outputs/diffsbdd_crossdocked_reliamol_v0
```

Expected manuscript-level anchors:

- QED/SA top-10% reliable rate near `0.9660`;
- ReliaMol top-10% reliable rate near `0.9980--0.9993`;
- exact receptor coordinates remove the public-coordinate pocket-contact mismatch.

## 4. Reproduce Ablation and Sensitivity Blocks

```bash
python scripts/run_diffsbdd_crossdocked_feature_ablation.py \
  --feature-frame outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv \
  --output-dir outputs/diffsbdd_crossdocked_feature_ablation_v0

python scripts/run_diffsbdd_crossdocked_extra_baselines.py \
  --feature-frame outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv \
  --output-dir outputs/diffsbdd_crossdocked_extra_baselines_v0

python scripts/analyze_diffsbdd_crossdocked_calibration.py \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol_v0/reports \
  --output-dir outputs/diffsbdd_crossdocked_calibration_v0

python scripts/sweep_diffsbdd_crossdocked_thresholds.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol_v0/reports \
  --output-dir outputs/diffsbdd_crossdocked_threshold_sensitivity_v0
```

Expected manuscript-level anchors:

- internal 3D plus ligand-pocket geometry reaches about `0.9990` top-10% reliability;
- ligand-pocket geometry is the dominant reliability and clash-failure signal;
- threshold and coordinate-noise analyses keep the selected top-list reliability high.

## 5. Reproduce Physical-Validity and Docking Baselines

```bash
python scripts/run_diffsbdd_crossdocked_posebusters.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol_v0/reports \
  --output-dir outputs/diffsbdd_crossdocked_posebusters_molfast_v0

python scripts/score_diffsbdd_crossdocked_vina_gnina.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol_v0/reports \
  --posebusters-dir outputs/diffsbdd_crossdocked_posebusters_molfast_v0/reports \
  --output-dir outputs/diffsbdd_crossdocked_vina_gnina_100pocket_v0 \
  --vina /path/to/vina \
  --gnina /path/to/gnina
```

Expected manuscript-level anchors:

- PoseBusters gate plus ReliaMol can produce selected sets with physical-validity pass rate `1.0000`;
- Vina/GNINA score-only baselines are useful but do not match ReliaMol reliability selection.

## 6. Reproduce Multi-Generator Transfer

```bash
python scripts/audit_targetdiff_official_meta.py \
  --input-dir /path/to/targetdiff_sampling_results \
  --pocket-dir /path/to/crossdocked_pocket10 \
  --output-dir outputs/targetdiff_official_multigen_audit_v0

python scripts/run_targetdiff_official_multigen_reliamol.py \
  --audit-dir outputs/targetdiff_official_multigen_audit_v0/reports \
  --output-dir outputs/targetdiff_official_multigen_reliamol_v0
```

Expected manuscript-level anchors:

- pocket-CV RF/HGB ReliaMol top-10% reliability near `0.9947`;
- leave-generator-out RF ReliaMol top-10% reliability near `0.9950`;
- held-out CVAE improves from Vina-minimize `0.7868` to RF ReliaMol `0.9800`.

## 7. Reproduce External DUD-E Stress Testing

```bash
python scripts/audit_dude_public_pool.py \
  --dude-root /path/to/dude_subset \
  --output-dir outputs/dude_public_pool_v0

python scripts/run_dude_public_reliability.py \
  --input-root outputs/dude_docking_public6_v0 \
  --output-dir outputs/dude_public_reliability_external_ood_20260522 \
  --seeds 20260522 20260523 20260524
```

Expected manuscript-level anchors:

- no-leak MLP ReliaMol ROC-AUC near `0.9981`;
- top-10% reliable rate near `0.9846`;
- Vina ranking remains low on deliberately corrupted controlled-failure poses.

## 8. Verify Included Paper Tables

The repository includes compact derived result files. Start with:

```bash
ls reports
find paper_results -maxdepth 4 -type f | sort
```

Then use `docs/paper_result_map.md` to locate which files support each manuscript table or supplementary table.
