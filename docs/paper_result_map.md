# Paper Result Map

This file maps manuscript evidence blocks to tracked scripts and compact result files. Raw data, generated molecule files, and large feature/prediction tables are excluded from Git.

## Main DiffSBDD Exact-CrossDocked Results

Manuscript claim:

- QED/SA top-10% reliable rate: `0.9660`.
- ReliaMol variants top-10% reliable rate: `0.9980--0.9993`.

Scripts:

- `scripts/audit_diffsbdd_crossdocked_gninatypes.py`
- `scripts/run_diffsbdd_crossdocked_reliamol.py`

Tracked result files:

- `paper_results/outputs/diffsbdd_crossdocked_gninatypes_audit/reports/audit_report.md`
- `paper_results/outputs/diffsbdd_crossdocked_gninatypes_audit/reports/pocket_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_reliamol/reports/metric_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_reliamol/reports/reranking_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_reliamol/reports/compact_comparison.json`

## Feature-Group Ablation

Manuscript claim:

- internal 3D plus ligand-pocket geometry reaches about `0.9990` top-10% reliability;
- full score-free ReliaMol reaches about `0.9816` ROC-AUC and `0.9990` top-10% reliability.

Script:

- `scripts/run_diffsbdd_crossdocked_feature_ablation.py`

Tracked result files:

- `paper_results/outputs/diffsbdd_crossdocked_feature_ablation/reports/ablation_metric_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_feature_ablation/reports/ablation_reranking_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_feature_ablation/reports/ablation_table_top10.csv`
- `paper_results/outputs/diffsbdd_crossdocked_feature_ablation/reports/ablation_metadata.json`

## Robustness, Calibration, and Significance

Manuscript claim:

- reliability remains high under threshold sweeps, radius changes, low-data training, bootstrap, and coordinate perturbation.

Scripts:

- `scripts/analyze_diffsbdd_crossdocked_calibration.py`
- `scripts/sweep_diffsbdd_crossdocked_thresholds.py`
- `scripts/run_diffsbdd_aligned_reliamol.py`

Tracked result files:

- `paper_results/outputs/diffsbdd_crossdocked_calibration/reports/calibration_bins.csv`
- `paper_results/outputs/diffsbdd_crossdocked_calibration/reports/calibration_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_threshold_sensitivity/reports/threshold_metric_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_threshold_sensitivity/reports/threshold_reranking_summary.csv`
- `paper_results/outputs/diffsbdd_aligned_reliamol/reports/reranking_summary.csv`

## PoseBusters and Docking-Score Baselines

Manuscript claim:

- PoseBusters gate plus ReliaMol gives physical-validity pass rate `1.0000` in selected sets;
- Vina/GNINA score-only baselines underperform ReliaMol on reliability selection.

Scripts:

- `scripts/run_diffsbdd_crossdocked_posebusters.py`
- `scripts/score_diffsbdd_crossdocked_vina_gnina.py`

Tracked result files:

- `paper_results/outputs/diffsbdd_crossdocked_posebusters_molfast/reports/posebusters_metric_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_posebusters_molfast/reports/posebusters_overall.csv`
- `paper_results/outputs/diffsbdd_crossdocked_posebusters_molfast/reports/posebusters_gated_reranking_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_vina_gnina_100pocket/reports/ensemble_reranking_summary.csv`
- `paper_results/outputs/diffsbdd_crossdocked_vina_gnina_100pocket/reports/vina_gnina_report.md`

## Chemical Utility and Scaffold Diversity

Manuscript claim:

- selected reliable molecules preserve scaffold diversity and basic medicinal-chemistry rule compatibility.

Scripts:

- `scripts/run_diffsbdd_drug_alignment.py`
- `scripts/score_diffsbdd_aligned_vina.py`

Tracked result files:

- `paper_results/outputs/diffsbdd_crossdocked_drug_alignment/reports/drug_alignment_report.md`
- `paper_results/outputs/diffsbdd_crossdocked_drug_alignment/reports/druglikeness_selection_summary.csv`
- `paper_results/outputs/diffsbdd_aligned_vina/reports/vina_reranking_summary.csv`

## TargetDiff-Family Multi-Generator Transfer

Manuscript claim:

- RF/HGB ReliaMol reaches about `0.9947` top-10% reliability under pocket-CV;
- RF ReliaMol reaches about `0.9950` under leave-generator-out;
- held-out CVAE improves from Vina-minimize `0.7868` to RF ReliaMol `0.9800`.

Scripts:

- `scripts/audit_targetdiff_official_meta.py`
- `scripts/run_targetdiff_official_multigen_reliamol.py`

Tracked result files:

- `reports/targetdiff_official_multigen_reliamol_summary.md`
- `paper_results/outputs/targetdiff_official_multigen_audit/reports/audit_report.md`
- `paper_results/outputs/targetdiff_official_multigen_audit/reports/pocket_generator_summary.csv`
- `paper_results/outputs/targetdiff_official_multigen_reliamol/reports/metric_summary.csv`
- `paper_results/outputs/targetdiff_official_multigen_reliamol/reports/reranking_summary.csv`
- `paper_results/outputs/targetdiff_official_multigen_reliamol/reports/leave_generator_topk_summary.csv`
- `paper_results/outputs/targetdiff_official_multigen_reliamol/reports/official_multigen_report.md`

## External DUD-E Controlled-Failure Stress Test

Manuscript claim:

- no-leak MLP ReliaMol reaches ROC-AUC about `0.9981` and top-10% reliable rate about `0.9846`;
- Vina ranking remains low in controlled-failure separation.

Scripts:

- `scripts/audit_dude_public_pool.py`
- `scripts/run_dude_docking_smoke.py`
- `scripts/run_dude_public_reliability.py`

Tracked result files:

- `paper_results/outputs/dude_docking_public6/reports/docking_analysis.md`
- `paper_results/outputs/dude_docking_public6/reports/docking_topk_summary.csv`
- `paper_results/outputs/dude_public_reliability_external_ood/analysis.md`
- `paper_results/outputs/dude_public_reliability_external_ood/lot_metrics_summary.csv`
- `paper_results/outputs/dude_public_reliability_external_ood/lot_reranking_summary.csv`
- `paper_results/outputs/dude_public_reliability_external_ood/lot_failure_diagnosis_summary.csv`

## Provenance

The release package was built from:

- remote source: `/home/test/wsk/16ReliaMol-3D`;
- remote commit: `d78abc2`;
- file inventory: `REPRODUCIBILITY_FILELIST.txt`;
- source record: `REPRODUCIBILITY_SOURCE.txt`.
