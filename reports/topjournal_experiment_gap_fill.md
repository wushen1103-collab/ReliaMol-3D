# Top-Journal Experiment Gap-Fill Pack

This note summarizes the experiments added after the IF/TNNLS/TKDE/TPAMI-style reviewer gap analysis. The goal is to strengthen the existing DiffSBDD exact-CrossDocked evidence without mixing it with unrelated or inaccessible generator data.

## Completed Additions

| Reviewer concern | Added experiment | Output |
|---|---|---|
| Reliability labels may depend on arbitrary geometry thresholds | Exact-receptor label threshold sweep over ligand atom mode, contact threshold, clash threshold, and internal-close threshold | `outputs/diffsbdd_crossdocked_threshold_sensitivity/reports/` |
| Generated molecules may fail standard physical validity checks | PoseBusters `mol_fast` audit on all 10,000 DiffSBDD molecules | `outputs/diffsbdd_crossdocked_posebusters_molfast/reports/` |
| PoseBusters and receptor-reliability should both be satisfied | PoseBusters-gated reranking analysis | `posebusters_gated_reranking_summary.csv` |
| Method may only beat weak QED/SA baselines | Linear, unsupervised, rule-based, and audit-upper-bound baselines | `outputs/diffsbdd_crossdocked_extra_baselines/reports/` |
| Scores may be poorly calibrated or not statistically different | Brier/calibration bins and pocket-bootstrap top-k delta vs QED/SA | `outputs/diffsbdd_crossdocked_calibration/reports/` |
| Model may be sensitive to pocket radius | Fast sensitivity reruns at pocket radii 8, 10, 12, and 16 A | `outputs/diffsbdd_crossdocked_reliamol_radius*/reports/` |
| Model may need too much training data | Fast sensitivity reruns using 25% and 50% of training pockets | `outputs/diffsbdd_crossdocked_reliamol_lowdata*/reports/` |
| Multi-generator generalization is expected | Public source availability audit for TargetDiff/Pocket2Mol/DecompDiff data | `reports/public_multigenerator_availability.md` |
| Same-population docking/scoring baseline is expected | 100-pocket exact CrossDocked Vina and GNINA/CNN score-only baselines | `outputs/diffsbdd_crossdocked_vina_gnina_100pocket/reports/` |

## Main Exact-CrossDocked Result

100 pockets, 10,000 official DiffSBDD candidates, exact CrossDocked `.gninatypes` receptor coordinates, 3 seeds x 5 pocket-heldout folds.

| Method | Top-10% reliable | Clash-free | ROC-AUC |
|---|---:|---:|---:|
| QED/SA | 0.9660 | 0.9730 | 0.6455 |
| MLP pose shape | 0.9987 +/- 0.0006 | 0.9990 | 0.9724 |
| MLP ReliaMol no-Vina | 0.9980 +/- 0.0020 | 0.9990 | 0.9737 |
| RF ReliaMol no-Vina | 0.9990 +/- 0.0000 | 0.9990 | 0.9819 |
| HGB ReliaMol no-Vina | 0.9993 +/- 0.0012 | 0.9997 | 0.9845 |

## Threshold Robustness

Raw reliable-label rate varies from 0.8978 to 0.9479 across the threshold grid, with median 0.9304. Top-10% selection remains stable:

| Method | Top-10% reliable range across thresholds |
|---|---:|
| QED/SA | 0.9530-0.9740 |
| MLP pose shape | 0.9937-0.9993 |
| MLP ReliaMol no-Vina | 0.9943-0.9983 |
| RF ReliaMol no-Vina | 0.9983-0.9990 |
| HGB ReliaMol no-Vina | 0.9980-0.9993 |

Interpretation: the improvement is not an artifact of a single clash/contact threshold.

## PoseBusters Validity

PoseBusters `mol_fast` overall pass rate is 0.5732. The main failed checks are bond lengths (0.7565 pass) and bond angles (0.7040 pass), while sanitization, InChI convertibility, connectivity, and radicals are all 1.0000.

ReliaMol alone should not be claimed to fix molecule-internal PoseBusters issues. The clean manuscript strategy is a two-stage post-filter:

| Policy | Method | Top-10% PoseBusters pass | Top-10% reliable |
|---|---|---:|---:|
| Raw rerank | QED/SA | 0.7640 | 0.9660 |
| PoseBusters gate | QED/SA | 1.0000 | 0.9800 |
| Raw rerank | RF ReliaMol | 0.7333 | 0.9990 |
| PoseBusters gate | RF ReliaMol | 1.0000 | 1.0000 |
| Raw rerank | HGB ReliaMol | 0.7670 | 0.9993 |
| PoseBusters gate | HGB ReliaMol | 1.0000 | 0.9983 |

All pockets had enough PoseBusters-pass candidates to keep exactly 10 selected molecules per pocket under the hard gate.

## Extra Baselines

| Baseline | Top-10% reliable |
|---|---:|
| QED/SA + explicit clash/contact rule | 0.9930 |
| Logistic regression, ligand descriptors | 0.9813 |
| Logistic regression, pose shape | 0.9967 |
| Isolation Forest, pose shape | 0.9967 |
| Isolation Forest, ReliaMol features | 0.9950 |
| Audit-label upper bound | 1.0000 |

Interpretation: simple pose-aware and unsupervised baselines are strong, which is good reviewer-facing context. The RF/HGB ReliaMol variants still give the highest non-oracle top-10 reliability in the main 3-seed run.

## Calibration And Significance

| Method | Brier score | High-confidence reliable rate |
|---|---:|---:|
| QED/SA | 0.3145 | 0.9660 |
| MLP pose shape | 0.0280 | 0.9955 |
| MLP ReliaMol no-Vina | 0.0266 | 0.9948 |
| RF ReliaMol no-Vina | 0.0253 | 0.9972 |
| HGB ReliaMol no-Vina | 0.0218 | 0.9884 |

Pocket-bootstrap top-10 reliable-rate deltas vs QED/SA are positive for pose-aware ReliaMol variants, with 95% intervals excluding zero:

| Method | Delta vs QED/SA | 95% CI |
|---|---:|---:|
| MLP pose shape | +0.0328 | [0.0177, 0.0503] |
| MLP ReliaMol no-Vina | +0.0318 | [0.0163, 0.0493] |
| RF ReliaMol no-Vina | +0.0329 | [0.0177, 0.0503] |
| HGB ReliaMol no-Vina | +0.0334 | [0.0177, 0.0517] |

## Radius And Low-Data Sensitivity

Fast sensitivity reruns use seed 44, 5 pocket-heldout folds, 45 epochs, 120 RF trees, and 100 HGB iterations. These are robustness checks, not replacements for the main 3-seed table.

| Setting | Best non-oracle Top-10% reliable |
|---|---:|
| radius 8 A | 0.9980 |
| radius 10 A | 0.9990 |
| radius 12 A fast rerun | 1.0000 |
| radius 16 A | 1.0000 |
| 25% training pockets | 1.0000 |
| 50% training pockets | 0.9990 |

Interpretation: the conclusion survives pocket-radius changes and remains strong with substantially fewer training pockets.

## Multi-Generator Boundary

TargetDiff's official repository is reachable, and its README points to Google Drive sampling-results meta files for TargetDiff, CVAE, AR, and Pocket2Mol. From this experiment server, that Google Drive folder timed out, and Pocket2Mol/DecompDiff repositories also timed out during the availability audit. Therefore multi-generator generalization should be treated as a remaining data-access-dependent experiment, not silently claimed.

## 100-Pocket Vina/GNINA Baseline

The exact 100 CrossDocked pocket PDBs were recovered from the IF3/Pocket2Mol/TargetDiff standard test split archive. A coordinate identity audit showed that the selected pocket PDB atoms match the CrossDocked `.gninatypes` receptor coordinates to numerical precision, so these are appropriate same-population receptors for score-only docking baselines.

| Score-only method | Success rate | ROC-AUC | PR-AUC | Top-10% reliable | Top-10% PoseBusters pass |
|---|---:|---:|---:|---:|---:|
| Vina | 0.9901 | 0.8559 | 0.9803 | 0.9750 | 0.5950 |
| GNINA empirical affinity | 0.9901 | 0.8518 | 0.9791 | 0.9760 | 0.5100 |
| GNINA CNNscore | 0.9901 | 0.6354 | 0.9517 | 0.9460 | 0.5820 |
| GNINA CNNaffinity | 0.9901 | 0.5245 | 0.9275 | 0.8790 | 0.4400 |

ReliaMol remains stronger for reliability-oriented selection. The best top-10% ReliaMol + Vina ensemble reaches 0.9997 reliable proxy and 0.9997 clash-free rate, but its PoseBusters pass rate remains about 0.7083; therefore PoseBusters-gated ReliaMol is still the cleanest manuscript strategy when molecule-internal validity is required.

## Manuscript Guidance

Recommended new claims:

1. ReliaMol-style pose-aware ranking is robust to exact-receptor threshold choices on public DiffSBDD CrossDocked samples.
2. PoseBusters detects an independent molecule-internal validity problem in raw DiffSBDD outputs; applying a PoseBusters gate before ReliaMol reranking gives both physical-validity pass and high receptor-reliability.
3. The improvement over QED/SA is statistically supported by pocket-bootstrap deltas and is not explained away by simple linear or unsupervised baselines.
4. Same-population Vina/GNINA score-only baselines on exact 100-pocket CrossDocked receptors are weaker than pose-aware ReliaMol for reliability-oriented top-k selection.

Avoid claiming multi-generator generalization until the TargetDiff/Pocket2Mol/DecompDiff generated-sample files are available locally.
