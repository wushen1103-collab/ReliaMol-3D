# DiffSBDD CrossDocked Evidence Note

Date: 2026-05-08

## Scope

This note separates three evidence layers that should not be conflated in the manuscript.

1. DUD-E controlled-failure reliability benchmark:
   stress-test ablation showing whether reliability models reject synthetic pocket, clash, chemistry, and scoring failures.
2. DiffSBDD public-PDB aligned Vina baseline:
   conventional score-only reranking on 39 pockets where public RCSB coordinates are compatible enough for PDBQT/Vina preparation.
3. DiffSBDD CrossDocked exact-coordinate benchmark:
   real generated-candidate audit and reranking on all 100 sampled pockets using CrossDocked2020 `.gninatypes` receptor coordinates.

## Data Provenance

- DiffSBDD official repository: `https://github.com/arneschneuing/DiffSBDD`
- DiffSBDD sampled molecules: `https://zenodo.org/record/8239058`
- CrossDocked receptor archive: `https://bits.csb.pitt.edu/files/crossdock2020/v1.0/CrossDocked2020_receptors.tgz`

The public-RCSB approximation is useful for a transparent first pass, but it is not coordinate-equivalent to CrossDocked prepared receptors. The exact-coordinate benchmark should use CrossDocked receptor coordinates when making generated-candidate claims.

## Receptor Provenance Result

| Receptor source | Candidates | Pockets | Reliable proxy | Pocket failure | Clash failure | Median min dist | Median contacts 4.5A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public RCSB approximation | 10,000 | 100 | 0.3816 | 0.5358 | 0.0761 | 6.9927 | 0.0 |
| CrossDocked gninatypes heavy receptor | 10,000 | 100 | 0.9304 | 0.0000 | 0.0502 | 2.6063 | 23.0 |

Key interpretation: the earlier all-pocket pocket-failure rate was primarily receptor-coordinate provenance, not DiffSBDD molecule placement. The 44 public-RCSB coordinate-mismatch pockets improve from 0.0000 to 0.9341 reliable proxy with CrossDocked coordinates.

## Main Real Generated-Candidate Table

Use this table as the primary real generated-candidate reranking evidence.

| Method | ROC-AUC | PR-AUC | Top 10% reliable | Top 20% reliable | Top 50% reliable |
|---|---:|---:|---:|---:|---:|
| QED | 0.6226 +/- 0.0000 | 0.9508 +/- 0.0000 | 0.9550 +/- 0.0000 | 0.9520 +/- 0.0000 | 0.9526 +/- 0.0000 |
| QED/SA | 0.6455 +/- 0.0000 | 0.9548 +/- 0.0000 | 0.9660 +/- 0.0000 | 0.9615 +/- 0.0000 | 0.9628 +/- 0.0000 |
| MLP ligand descriptor | 0.6695 +/- 0.0027 | 0.9561 +/- 0.0011 | 0.9703 +/- 0.0025 | 0.9715 +/- 0.0038 | 0.9609 +/- 0.0008 |
| MLP pose shape | 0.9724 +/- 0.0015 | 0.9974 +/- 0.0002 | 0.9987 +/- 0.0006 | 0.9980 +/- 0.0005 | 0.9951 +/- 0.0010 |
| MLP ReliaMol no-Vina | 0.9737 +/- 0.0026 | 0.9974 +/- 0.0007 | 0.9980 +/- 0.0020 | 0.9982 +/- 0.0003 | 0.9949 +/- 0.0006 |
| RF ReliaMol no-Vina | 0.9819 +/- 0.0014 | 0.9986 +/- 0.0002 | 0.9990 +/- 0.0000 | 0.9988 +/- 0.0003 | 0.9959 +/- 0.0002 |
| HGB ReliaMol no-Vina | 0.9845 +/- 0.0020 | 0.9984 +/- 0.0008 | 0.9993 +/- 0.0012 | 0.9983 +/- 0.0003 | 0.9961 +/- 0.0004 |

Recommended claim wording:

ReliaMol-style pose-aware reliability ranking substantially improves the reliability of public DiffSBDD generated candidates under exact CrossDocked receptor coordinates. On 100 pockets and 10,000 candidates, the raw reliable proxy rate is 0.9304; QED/SA selects 0.9660 reliable candidates at top 10%, while pose-aware ReliaMol variants select 0.9980-0.9993.

## Vina Baseline Boundary

Vina score-only should be reported as a separate 39-pocket public-PDB/PDBQT-compatible baseline:

- Vina score-only top 10% reliable: 0.9923
- MLP ReliaMol no-leak top 10% reliable on the same aligned subset: 0.9974
- RF ReliaMol no-leak top 10% reliable on the same aligned subset: 1.0000

Do not compare the 39-pocket Vina subset as if it were the same evaluation population as the 100-pocket CrossDocked `.gninatypes` benchmark. The Vina subset is useful for conventional scoring context; the 100-pocket exact-coordinate table is the main generated-candidate evidence.

## Reporting Cautions

- State that `.gninatypes` records provide CrossDocked coordinates and atom types, not a Vina-ready receptor.
- State that receptor hydrogen atom types 0/1 are excluded for heavy-receptor geometry auditing.
- Do not report the earlier all-pocket public-RCSB pocket-failure rate as a generated-candidate failure.
- Keep controlled-failure results and real-generated-candidate results in separate tables.

## Reviewer-Facing Robustness Additions

Additional IF/TNNLS/TKDE/TPAMI-style robustness experiments are summarized in `reports/topjournal_experiment_gap_fill.md`.

- Threshold sweep: raw reliable-label rate ranges from 0.8978 to 0.9479 across contact/clash/internal thresholds, while pose-aware ReliaMol top-10 reliability remains 0.9937-0.9993.
- PoseBusters: raw DiffSBDD molecule-internal validity pass rate is 0.5732; a PoseBusters gate followed by RF ReliaMol reranking gives 1.0000 PoseBusters pass and 1.0000 reliable proxy at top 10%.
- Extra baselines: linear pose, unsupervised pose, and explicit QED/SA+clash/contact baselines are strong but remain below the best RF/HGB ReliaMol variants in the main 3-seed table.
- Calibration/significance: pocket-bootstrap top-10 deltas versus QED/SA are positive for pose-aware ReliaMol variants, with 95% intervals excluding zero.
- 100-pocket Vina/GNINA: exact IF3 CrossDocked pocket PDBs are coordinate-identical to the `.gninatypes` receptors. Vina and GNINA affinity score-only baselines reach 0.9750-0.9760 reliable proxy at top 10%, below the main ReliaMol top-10 range of 0.9980-0.9993.
