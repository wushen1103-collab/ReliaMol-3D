# TargetDiff-Family Official Multi-Generator Experiment

## Source and Scope

The TargetDiff official Google Drive folder referenced by the README was not programmatically listable from either the local machine or the remote server. I therefore used the public PIDiff GitHub mirror of the same official TargetDiff sampling-result filenames and recorded the concrete file hashes in `/home/test/wsk/public_generators/targetdiff_sampling_results/SOURCE_MANIFEST.tsv`.

Evaluated files:

- `targetdiff_vina_docked.pt`
- `pocket2mol_vina_docked.pt`
- `cvae_vina_docked.pt`
- `ar_vina_docked.pt`
- `crossdocked_test_vina_docked.pt` as a native/reference audit control

All generated candidates were audited against the 100 IF3/CrossDocked pocket10 PDBs. The learned ReliaMol comparisons use the four generated methods only: TargetDiff, Pocket2Mol, CVAE, and AR.

## Audit Summary

| generator | candidates | pockets | reliable proxy | chemical fail | geometry fail | clash fail | median Vina minimize |
|---|---:|---:|---:|---:|---:|---:|---:|
| CrossDockedNative | 100 | 100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | -6.4865 |
| CVAE | 9,911 | 100 | 0.7905 | 0.0221 | 0.1825 | 0.0000 | -6.5780 |
| AR | 9,295 | 97 | 0.9826 | 0.0132 | 0.0013 | 0.0002 | -5.8790 |
| Pocket2Mol | 9,831 | 100 | 0.9861 | 0.0136 | 0.0001 | 0.0003 | -5.8220 |
| TargetDiff | 9,036 | 100 | 0.9925 | 0.0020 | 0.0002 | 0.0022 | -6.8310 |

CVAE is the intentionally useful hard case: its baseline reliability is much lower and mostly driven by geometry failures, while TargetDiff/Pocket2Mol/AR are already very clean.

## Pocket-Heldout Results

Top-10% reranking over 3 seeds x 5 pocket folds:

| method | reliable top10 | geometry pass | clash-free | median Vina minimize |
|---|---:|---:|---:|---:|
| Vina minimize | 0.9423 | 0.9488 | 1.0000 | -8.4650 |
| QED+SA+Vina | 0.9514 | 0.9587 | 1.0000 | -7.0978 |
| MLP ReliaMol | 0.9933 | 0.9982 | 0.9999 | -6.1822 |
| RF ReliaMol | 0.9947 | 0.9977 | 0.9999 | -5.4786 |
| HGB ReliaMol | 0.9947 | 0.9973 | 1.0000 | -5.7594 |

Classification metrics:

| method | ROC-AUC | PR-AUC | MCC | ECE |
|---|---:|---:|---:|---:|
| Vina minimize | 0.5549 | 0.9370 | 0.0794 | 0.4441 |
| QED+SA+Vina | 0.7310 | 0.9743 | 0.1542 | 0.4358 |
| MLP ReliaMol | 0.9970 | 0.9998 | 0.8957 | 0.0236 |
| RF ReliaMol | 0.9987 | 0.9999 | 0.9509 | 0.0187 |
| HGB ReliaMol | 0.9987 | 0.9999 | 0.9638 | 0.0031 |

## Leave-Generator-Out Results

Top-10% reranking when an entire generator is held out:

| method | reliable top10 | geometry pass | clash-free | median Vina minimize |
|---|---:|---:|---:|---:|
| Vina minimize | 0.9441 | 0.9504 | 1.0000 | -8.2923 |
| QED+SA+Vina | 0.9529 | 0.9599 | 1.0000 | -7.0970 |
| MLP ReliaMol | 0.9892 | 0.9928 | 0.9999 | -6.1555 |
| RF ReliaMol | 0.9950 | 0.9957 | 1.0000 | -5.7864 |
| HGB ReliaMol | 0.9846 | 0.9862 | 1.0000 | -6.0126 |

Classification metrics:

| method | ROC-AUC | PR-AUC | MCC | ECE |
|---|---:|---:|---:|---:|
| Vina minimize | 0.7589 | 0.9398 | 0.0373 | 0.4533 |
| QED+SA+Vina | 0.7690 | 0.9630 | 0.1271 | 0.4396 |
| MLP ReliaMol | 0.9677 | 0.9925 | 0.7090 | 0.0325 |
| RF ReliaMol | 0.9903 | 0.9981 | 0.8966 | 0.0346 |
| HGB ReliaMol | 0.9474 | 0.9852 | 0.7873 | 0.0404 |

The most informative hard split is held-out CVAE. Its raw reliability is 0.7905; Vina minimize top10 stays at 0.7868, QED+SA+Vina reaches 0.8218, MLP ReliaMol reaches 0.9573, and RF ReliaMol reaches 0.9800. This is the strongest new evidence that ReliaMol learns transferable reliability cues rather than merely copying docking score.

## Paper-Facing Conclusion

This experiment directly addresses the top-journal concern that ReliaMol might only work on DiffSBDD or a single synthetic generator family. On official TargetDiff-family generated results, ReliaMol substantially improves reliability top-k under both pocket-heldout and leave-generator-out settings, while Vina tends to select high-affinity but less reliable candidates. The CVAE held-out result is especially useful because it creates a nontrivial low-reliability target distribution where ReliaMol recovers most of the clean candidates without using Vina as a feature.

Artifacts:

- Audit: `outputs/targetdiff_official_multigen_audit/reports/`
- ReliaMol experiment: `outputs/targetdiff_official_multigen_reliamol/reports/`
- Main generated report: `outputs/targetdiff_official_multigen_reliamol/reports/official_multigen_report.md`
