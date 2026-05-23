# Experiment Log

## 2026-05-05: CrossDocked Tensor Real Smoke v1

Remote path: `/home/test/wsk/16ReliaMol-3D`

Data source:

- `/home/test/wsk/TASC-VS/data/crossdocked2020/crossdocked.zip`
- Tensorized CrossDocked package with `positions`, `atom_types`, `n_atoms`, `offsets`, and `starting_fragment_mask`.
- The mask separates pocket atoms from ligand atoms. This allows a first real-coordinate smoke test without raw PDB/SDF files.

Setup:

- Train pockets: 900
- Val pockets: 300
- Test pockets: 100
- Generators simulated from real ligand poses:
  - `NativeNear`
  - `DriftGen`
  - `ClashGen`
  - `ScoreHackGen`
- Candidates per generator per pocket: 10

Main result:

| Method | ROC-AUC | PR-AUC | F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| ReliaMol MLP | 0.9998 | 0.9996 | 0.9919 | 0.9878 | 0.0174 |
| Vina rank proxy | 0.4541 | 0.2930 | 0.5156 | 0.0989 | 0.2967 |
| GNINA rank proxy | 0.8574 | 0.6796 | 0.7169 | 0.5528 | 0.1608 |
| QED+SA+Vina proxy | 0.4539 | 0.2933 | 0.5160 | 0.1000 | 0.2923 |
| RF handcrafted | 0.9999 | 0.9999 | 0.9963 | 0.9944 | 0.0143 |
| HGB handcrafted | 0.9999 | 0.9997 | 0.9967 | 0.9950 | 0.0025 |

Reranking:

| Score | Top-k | Reliable Rate | Clash-Free | Vina/GNINA Consistency |
|---|---:|---:|---:|---:|
| ReliaMol | 1% | 0.99 | 1.00 | 1.00 |
| ReliaMol | 5% | 0.99 | 1.00 | 1.00 |
| ReliaMol | 10% | 0.99 | 1.00 | 1.00 |
| Vina rank | 1% | 0.14 | 0.27 | 0.25 |
| Vina rank | 5% | 0.155 | 0.27 | 0.235 |
| Vina rank | 10% | 0.1575 | 0.27 | 0.2475 |
| QED | 1% | 0.32 | 0.62 | 0.65 |

Failure diagnosis:

- Chemical failure positives are now present in the test set: 8.55%.
- Synthetic failure positives are now present in the test set: 7.70%.
- Pocket/scoring/geometric failures are all learnable in this smoke setting.

Interpretation:

- The full pipeline is now proven end to end on real CrossDocked coordinate tensors:
  data loading -> candidate construction -> five failure labels -> reliability training -> diagnosis -> reranking.
- Vina-style ranking selects many unreliable high-score candidates, while reliability reranking strongly enriches valid candidates. This supports the paper's central story direction.

Important limitation:

- This is not yet a publishable main result. RF/HGB also achieve near-perfect scores, meaning the current smoke labels are still mostly recoverable from explicit handcrafted rules.
- The next experiment must reduce rule leakage and introduce learned structure representations, otherwise reviewers can argue the method is just a rule classifier.

Next technical step:

1. Add representation variants:
   - handcrafted-only
   - ligand geometry encoder
   - pocket geometry encoder
   - ligand-pocket interaction encoder
2. Run ablations without direct alert/count features.
3. Add real generated candidates from open-source generators or existing generated PDBQT/SDF pools.
4. Replace current synthetic/chemical proxies with RDKit/PoseBusters/AiZynthFinder-compatible labels when raw molecules are available.

## 2026-05-05: CrossDocked Tensor Real Smoke v2, No-Leak Feature Variants

Reason:

- v1 proved the end-to-end real-coordinate pipeline, but RF/HGB and MLP were nearly perfect because direct rule features leaked the label definition.
- v2 separates feature variants so the main result can distinguish genuine interaction/geometry signal from direct rule recovery.

Feature variants:

| Variant | Inputs |
|---|---|
| `mlp_score_only` | Vina/GNINA/QED/SA/SCScore proxies |
| `mlp_ligand_geometry` | ligand atom composition and internal geometry summaries |
| `mlp_interaction_shape` | ligand geometry + pocket radius/center distance + ligand-pocket distance distribution/shell features |
| `mlp_reliamol_noleak` | score + ligand geometry + pocket/interaction shape, excluding direct count/alert/leak features |
| `mlp_leak_upper_bound` | no-leak features plus direct count/alert/perturbation metadata, used only as an upper-bound diagnostic |

Main result:

| Method | ROC-AUC | PR-AUC | F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| Vina rank proxy | 0.4541 | 0.2930 | 0.5156 | 0.0989 | 0.2967 |
| GNINA rank proxy | 0.8574 | 0.6796 | 0.7169 | 0.5528 | 0.1608 |
| QED+SA+Vina proxy | 0.4539 | 0.2933 | 0.5160 | 0.1000 | 0.2923 |
| MLP score-only | 0.9036 | 0.7538 | 0.7752 | 0.6539 | 0.0874 |
| MLP ligand-geometry | 0.8345 | 0.6681 | 0.6804 | 0.4895 | 0.0774 |
| MLP interaction-shape | 0.9621 | 0.8837 | 0.8896 | 0.8314 | 0.0840 |
| MLP ReliaMol no-leak | 0.9645 | 0.8877 | 0.9108 | 0.8655 | 0.0765 |
| MLP leak upper bound | 0.9998 | 0.9996 | 0.9912 | 0.9866 | 0.0172 |
| RF no-leak | 0.9626 | 0.8756 | 0.9022 | 0.8524 | 0.0261 |
| HGB no-leak | 0.9640 | 0.8850 | 0.9123 | 0.8675 | 0.0148 |

Reranking:

| Score | Top-k | Reliable Rate | Clash-Free | Vina/GNINA Consistency |
|---|---:|---:|---:|---:|
| MLP score-only | 1% | 0.82 | 1.00 | 1.00 |
| MLP ligand-geometry | 1% | 0.82 | 0.97 | 0.97 |
| MLP interaction-shape | 1% | 0.87 | 1.00 | 1.00 |
| MLP ReliaMol no-leak | 1% | 0.90 | 1.00 | 1.00 |
| MLP leak upper bound | 1% | 0.99 | 1.00 | 1.00 |
| Vina rank | 1% | 0.14 | 0.27 | 0.25 |
| QED | 1% | 0.32 | 0.62 | 0.65 |

Failure diagnosis from `mlp_reliamol_noleak`:

| Failure | ROC-AUC | PR-AUC | Positive Rate |
|---|---:|---:|---:|
| Chemical | 0.5482 | 0.1015 | 0.0855 |
| Geometric | 0.9944 | 0.9885 | 0.2970 |
| Pocket | 0.9997 | 0.9996 | 0.3950 |
| Synthetic | 0.6793 | 0.2773 | 0.0770 |
| Scoring | 0.9985 | 0.9973 | 0.3353 |

Interpretation:

- Removing direct leak features makes the task nontrivial while preserving the core story.
- Interaction-shape features are the main contributor: `mlp_interaction_shape` beats score-only and ligand-only, especially in MCC and reranking.
- `mlp_reliamol_noleak` gives the strongest no-leak top-k enrichment among MLP variants.
- Chemical and synthetic diagnosis remain weak. This is expected because current tensor data lacks true molecular graph validity and retrosynthesis labels. These heads should be improved only after raw molecules/RDKit/PoseBusters/AiZynthFinder labels are attached.

Next step:

- Run multi-seed v2 to confirm stability.
- Add leave-one-generator-out evaluation on the four perturbation families.
- Start attaching raw generated molecule pools so chemical/synthetic failures are not proxy-only.

## 2026-05-05: CrossDocked Tensor Real Smoke v2 Multi-Seed

Seeds:

- 11
- 22
- 33

Main multi-seed result:

| Method | ROC-AUC mean | ROC-AUC std | PR-AUC mean | PR-AUC std | F1 mean | MCC mean | ECE mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vina rank proxy | 0.4462 | 0.0058 | 0.2865 | 0.0019 | 0.5111 | 0.0851 | 0.2981 |
| GNINA rank proxy | 0.8498 | 0.0051 | 0.6628 | 0.0153 | 0.7135 | 0.5489 | 0.1631 |
| MLP score-only | 0.8973 | 0.0045 | 0.7558 | 0.0146 | 0.7629 | 0.6364 | 0.0716 |
| MLP ligand-geometry | 0.8320 | 0.0032 | 0.6649 | 0.0135 | 0.6757 | 0.4847 | 0.0763 |
| MLP interaction-shape | 0.9612 | 0.0001 | 0.8798 | 0.0033 | 0.8877 | 0.8299 | 0.0772 |
| MLP ReliaMol no-leak | 0.9662 | 0.0015 | 0.8901 | 0.0114 | 0.9143 | 0.8717 | 0.0820 |
| MLP leak upper bound | 0.9998 | 0.0001 | 0.9996 | 0.0001 | 0.9921 | 0.9881 | 0.0188 |
| RF no-leak | 0.9642 | 0.0007 | 0.8862 | 0.0056 | 0.9025 | 0.8527 | 0.0257 |
| HGB no-leak | 0.9662 | 0.0006 | 0.8876 | 0.0025 | 0.9150 | 0.8727 | 0.0120 |

Top-1% reranking stability:

| Score | Reliable Rate mean | Reliable Rate std | Clash-Free mean | Vina/GNINA Consistency mean |
|---|---:|---:|---:|---:|
| MLP score-only | 0.8533 | 0.0231 | 1.0000 | 0.9967 |
| MLP ligand-geometry | 0.6933 | 0.0493 | 0.9500 | 0.9500 |
| MLP interaction-shape | 0.8933 | 0.0115 | 1.0000 | 1.0000 |
| MLP ReliaMol no-leak | 0.9067 | 0.0252 | 1.0000 | 1.0000 |
| MLP leak upper bound | 0.9900 | 0.0000 | 1.0000 | 1.0000 |
| Vina rank | 0.1567 | 0.0115 | 0.2500 | 0.2367 |
| QED | 0.3000 | 0.0100 | 0.5833 | 0.6700 |

Interpretation:

- The no-leak ReliaMol variant is stable across seeds and consistently improves over score-only, ligand-only, Vina, GNINA, and QED-style rules.
- Interaction features are the major signal source. This directly supports the method design choice to include ligand-pocket interaction encoding.
- HGB no-leak remains competitive, so the next model step should replace summary statistics with learned point/graph encoders rather than only tuning the MLP.
- Vina top-k remains sharply unreliable, supporting the paper's main motivation that docking-like scores can overselect bad poses.

Next step:

- Add leave-one-generator-out split on `NativeNear`, `DriftGen`, `ClashGen`, and `ScoreHackGen`.
- Then attach raw generated pools with RDKit/PoseBusters labels.

## 2026-05-05: Leave-One-Generator-Out Smoke

Setup:

- Candidate labels are reused from CrossDocked tensor smoke v2.
- Training uses train+val pockets while removing the held-out generator.
- Test initially uses only the held-out generator on test pockets.

Finding from held-out-only test:

- `DriftGen` and `ScoreHackGen` are useful LOGO tests.
- `ClashGen` and `NativeNear` have extreme positive rates:
  - `ClashGen`: reliable rate 0.002
  - `NativeNear`: reliable rate 0.869
- These extremes make PR-AUC and top-k results hard to compare directly.

Therefore, I added a mixed-native-anchor LOGO:

- Hold out one failure generator from training.
- Test on `NativeNear + heldout generator` for the same test pockets.
- This better matches reranking: choose reliable candidates from a pool containing an unseen bad generator and reliable-like candidates.

Mixed-native-anchor LOGO main result:

| Method | ROC-AUC mean | PR-AUC mean | F1 mean | MCC mean | ECE mean |
|---|---:|---:|---:|---:|---:|
| Vina rank proxy | 0.4879 | 0.4826 | 0.6959 | 0.1977 | 0.2808 |
| GNINA rank proxy | 0.8258 | 0.7773 | 0.8402 | 0.6128 | 0.1507 |
| MLP score-only | 0.8660 | 0.8268 | 0.8610 | 0.6697 | 0.1281 |
| MLP ligand-geometry | 0.8424 | 0.7954 | 0.8056 | 0.5803 | 0.1911 |
| MLP interaction-shape | 0.9204 | 0.8783 | 0.9118 | 0.8044 | 0.1081 |
| MLP ReliaMol no-leak | 0.9209 | 0.8799 | 0.9168 | 0.8170 | 0.0920 |
| MLP leak upper bound | 0.9943 | 0.9959 | 0.9814 | 0.9601 | 0.0329 |
| RF no-leak | 0.9232 | 0.8825 | 0.9145 | 0.8121 | 0.0438 |
| HGB no-leak | 0.9150 | 0.8692 | 0.9092 | 0.7981 | 0.0455 |

Mixed-native-anchor reranking:

| Score | Top-1% Reliable Rate | Clash-Free | Vina/GNINA Consistency |
|---|---:|---:|---:|
| MLP interaction-shape | 0.8400 | 0.9967 | 1.0000 |
| MLP ReliaMol no-leak | 0.8767 | 1.0000 | 1.0000 |
| MLP leak upper bound | 0.9867 | 1.0000 | 1.0000 |
| Vina rank | 0.4700 | 0.6300 | 0.6067 |
| QED | 0.5200 | 0.7333 | 0.7600 |

Note:

- In mixed-anchor LOGO, each pocket has 20 candidates, so top-1% and top-5% both select one candidate per pocket; top-10% selects two.

Interpretation:

- The mixed-anchor LOGO result supports generator-agnostic reranking better than heldout-only LOGO.
- ReliaMol no-leak improves substantially over Vina/QED and modestly over score-only.
- Interaction-shape features remain essential.
- However, the current "generators" are perturbation families, not actual SBDD generators. This is acceptable as a smoke/stress test but must be replaced or supplemented with Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff candidates.

Next step:

- Attach raw generated molecules/poses from existing pools or run a small open-source generator reproduction.
- Prioritize obtaining SDF/PDBQT plus receptor assets so RDKit, PoseBusters, and Vina/GNINA labels can replace proxy labels.
## 2026-05-05: Excluded Internal Unpublished Assets

Integrity note:

- A later experiment briefly used an internal asset pool discovered on the remote filesystem.
- That asset pool belongs to a separate unpublished manuscript and must not be mixed with this paper.
- All derived scripts, reports, and generated outputs have been removed from the current ReliaMol-3D evidence chain.
- The only valid current paper-facing results in this log are the proxy and CrossDocked smoke/stress experiments above.

Next step:

- Use only public, separately citable, or newly generated SBDD candidate pools for the next real-file benchmark.
- Candidate directions: CrossDocked-derived raw molecule/pose exports, or public Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff outputs regenerated specifically for this paper.

## 2026-05-07: Public DUD-E Real-File Audit and Vina Docking Smoke

Integrity boundary:

- This experiment uses only the public DUD-E subset under `/home/test/wsk/external_benchmarks/dude_subset`.
- No files from `/home/test/wsk/Finish` or the internal quarantined asset pool are used.
- This is a public candidate-pool and docking smoke, not a generated-molecule benchmark.

Scripts added:

- `scripts/audit_dude_public_pool.py`
- `scripts/run_dude_docking_smoke.py`
- `scripts/analyze_dude_docking_smoke.py`

Audit result:

- Targets checked first: `adrb2`, `bace1`, `cp3a4`, 100 actives + 100 decoys per target/class where SDF files exist.
- Raw DUD-E active/decoy SDF coordinates are not pocket-aligned:
  - Median center-to-crystal distance is roughly 34.6 to 54.0 A.
  - Pocket-contact rate is 0.0 before docking.
- Therefore raw DUD-E SDF coordinates must not be used directly for pocket-conditioned reliability labels.

Docking smoke setup:

- Complete public targets with both active and decoy SDF files:
  - `adrb2`, `bace1`, `cp3a4`, `pparg`, `src`, `try1`
- Molecules: 30 actives + 30 decoys per target, 360 total.
- Docking: AutoDock Vina 1.2.7, exhaustiveness 4, one mode, 64 parallel workers, one CPU per Vina process.
- Receptor prep:
  - DUD-E receptor PDB files lack standard element columns.
  - The script writes temporary fixed PDB files.
  - Meeko is attempted first; Open Babel receptor PDBQT fallback is used when Meeko rejects protein connectivity.
- Output:
  - `outputs/dude_docking_public6_v0/reports/docking_results.csv`
  - `outputs/dude_docking_public6_v0/reports/docking_analysis.md`
  - `outputs/dude_docking_public6_v0/reports/docking_reliability_labels.csv`

Pose quality:

| Target/Class | n | Success | Public Pose Reliable | Median Vina | Median Center-to-Crystal |
|---|---:|---:|---:|---:|---:|
| adrb2 active | 30 | 1.0000 | 0.8667 | -8.2775 | 3.5733 |
| adrb2 decoy | 30 | 1.0000 | 0.9667 | -9.6625 | 5.6835 |
| bace1 active | 30 | 1.0000 | 1.0000 | -7.4080 | 3.6185 |
| bace1 decoy | 30 | 1.0000 | 1.0000 | -7.8965 | 3.6955 |
| cp3a4 active | 30 | 1.0000 | 1.0000 | -9.0945 | 2.5354 |
| cp3a4 decoy | 30 | 1.0000 | 1.0000 | -8.2880 | 4.5387 |
| pparg active | 30 | 1.0000 | 0.9667 | -8.7670 | 3.1666 |
| pparg decoy | 30 | 1.0000 | 0.9000 | -8.4145 | 4.1381 |
| src active | 30 | 1.0000 | 1.0000 | -9.2435 | 5.1393 |
| src decoy | 30 | 1.0000 | 1.0000 | -8.2175 | 3.7089 |
| try1 active | 30 | 0.9667 | 0.9667 | -7.9730 | 2.5735 |
| try1 decoy | 30 | 1.0000 | 0.9667 | -6.2485 | 4.7686 |

Vina active/decoy separation:

| Target | Active Median | Decoy Median | Active-Decoy Median Delta | P(active better than decoy) |
|---|---:|---:|---:|---:|
| adrb2 | -8.2775 | -9.6625 | 1.3850 | 0.3956 |
| bace1 | -7.4080 | -7.8965 | 0.4885 | 0.4011 |
| cp3a4 | -9.0945 | -8.2880 | -0.8065 | 0.6844 |
| pparg | -8.7670 | -8.4145 | -0.3525 | 0.6300 |
| src | -9.2435 | -8.2175 | -1.0260 | 0.7944 |
| try1 | -7.9730 | -6.2485 | -1.7245 | 0.9644 |

Interpretation:

- The public real-file pipeline now works end to end:
  public receptor/SDF -> receptor repair -> ligand/receptor PDBQT -> Vina docking -> pose-quality labels.
- Pose quality is high enough for a first public benchmark, but the labels are heavily skewed toward reliable poses.
- Vina score is not a stable active/decoy discriminator across targets. On `adrb2` and `bace1`, decoys score better than actives by median Vina score.
- This supports using Vina as a baseline and diagnostic feature, not as the sole reliability criterion.

Next step:

- Build a no-leak public reliability dataset from these docked poses by adding controlled failure variants:
  - shifted poses for pocket failures,
  - clash-injected poses for geometry failures,
  - invalid/failed ligand prep for chemical failures,
  - score/pose inconsistency labels from Vina sensitivity runs.
- Run leave-one-target validation so the model is not just learning target-specific docking geometry.
- Keep the public DUD-E route separate from any unpublished internal assets.

## 2026-05-07: Public DUD-E Controlled-Failure Reliability v2

Purpose:

- Turn the public DUD-E docked poses into a reliability stress benchmark.
- Add controlled failure variants around each public docked pose:
  - `docked`
  - `pocket_shift`
  - `clash_inject`
  - `collapse`
  - `chemical_alert`
  - `synthetic_alert`
- This is still a controlled-failure stress test, not a generated-molecule benchmark.

Leakage correction:

- v1 included paired `native_vina_score` and `vina_delta_from_native` in model features.
- That pairing would not exist for arbitrary generated candidates, so v2 removes both from all model feature sets.
- v2 model score features use only each candidate's own `vina_score`.

Setup:

- Input: `outputs/dude_docking_public6_v0`
- Output: `outputs/dude_public_reliability_v2_multiseed`
- Public targets: `adrb2`, `bace1`, `cp3a4`, `pparg`, `src`, `try1`
- Base docked poses: 359 successful public docked poses.
- Controlled candidate rows: 2154.
- Evaluation: leave-one-target validation.
- Seeds: `20260507`, `20260508`, `20260509`.
- Main script: `scripts/run_dude_public_reliability.py`

Variant label summary:

| Variant | Reliable | Chemical Fail | Geometric Fail | Pocket Fail | Synthetic Fail | Scoring Fail |
|---|---:|---:|---:|---:|---:|---:|
| docked | 0.8830 | 0.0028 | 0.0808 | 0.0251 | 0.0084 | 0.0000 |
| pocket_shift | 0.0000 | 0.0028 | 0.5209 | 1.0000 | 0.0084 | 0.0000 |
| clash_inject | 0.0000 | 0.0028 | 1.0000 | 0.6741 | 0.0084 | 1.0000 |
| collapse | 0.0000 | 0.0028 | 1.0000 | 0.0613 | 0.0084 | 0.0000 |
| chemical_alert | 0.0000 | 1.0000 | 0.0808 | 0.0251 | 0.0084 | 0.0000 |
| synthetic_alert | 0.0000 | 0.0028 | 0.0808 | 0.0251 | 1.0000 | 0.0000 |

Three-seed leave-one-target metrics:

| Method | ROC-AUC | PR-AUC | F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| Vina rank | 0.5993 +/- 0.0292 | 0.1634 +/- 0.0084 | 0.3401 | 0.2390 | 0.3529 |
| QED+SA+Vina | 0.6411 +/- 0.0286 | 0.1794 +/- 0.0097 | 0.3665 | 0.2761 | 0.3528 |
| MLP score-only | 0.7070 +/- 0.0581 | 0.2562 +/- 0.0402 | 0.3772 | 0.2721 | 0.1261 |
| MLP ligand descriptor | 0.7121 +/- 0.0165 | 0.2348 +/- 0.0118 | 0.3688 | 0.2804 | 0.1047 |
| MLP pose shape | 0.8205 +/- 0.0149 | 0.3242 +/- 0.0128 | 0.4890 | 0.4537 | 0.0752 |
| MLP ReliaMol no-leak | 0.9978 +/- 0.0035 | 0.9812 +/- 0.0345 | 0.9777 | 0.9743 | 0.0838 |
| MLP leak upper bound | 0.9992 +/- 0.0008 | 0.9956 +/- 0.0042 | 0.9777 | 0.9741 | 0.0768 |
| RF no-leak | 0.9960 +/- 0.0026 | 0.9763 +/- 0.0184 | 0.9531 | 0.9455 | 0.0529 |
| HGB no-leak | 0.9982 +/- 0.0020 | 0.9899 +/- 0.0112 | 0.9740 | 0.9698 | 0.0234 |

Reranking:

| Score | Top-k | Reliable Rate | Clash-Free Proxy |
|---|---:|---:|---:|
| Vina rank | 10% | 0.0108 +/- 0.0216 | 0.3148 +/- 0.0820 |
| Vina rank | 20% | 0.1044 +/- 0.0245 | 0.5282 +/- 0.0570 |
| MLP score-only | 10% | 0.2485 +/- 0.0599 | 0.8812 +/- 0.1056 |
| MLP pose shape | 10% | 0.3225 +/- 0.0194 | 0.9861 +/- 0.0320 |
| MLP ReliaMol no-leak | 10% | 0.9799 +/- 0.0367 | 0.9815 +/- 0.0369 |
| MLP ReliaMol no-leak | 20% | 0.7356 +/- 0.0789 | 0.8640 +/- 0.0594 |
| MLP leak upper bound | 10% | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

Failure diagnosis with ReliaMol no-leak:

| Failure Head | ROC-AUC | PR-AUC | F1 | Positive Rate |
|---|---:|---:|---:|---:|
| Chemical | 0.9981 | 0.9967 | 0.9952 | 0.1690 |
| Geometric | 0.9945 | 0.9936 | 0.9650 | 0.4583 |
| Pocket | 0.9542 | 0.9389 | 0.8816 | 0.3031 |
| Synthetic | 0.9940 | 0.9892 | 0.9811 | 0.1736 |
| Scoring | 1.0000 | 0.9999 | 0.9986 | 0.1667 |

Interpretation:

- The clean no-leak model stays close to the leak upper bound even after removing paired native/delta score features.
- Score-only and ligand-only ablations are much weaker, while pose-shape helps but is not enough alone.
- The strongest claim supported by this experiment is not "we beat all real generators"; it is:
  public receptor/SDF docking plus controlled reliability stress labels show that multi-factor reliability reranking can reject pose, geometry, chemistry, synthesis, and scoring failures far better than Vina ranking.
- Vina ranking is actively unsafe in this stress setting: its top-10% reliable rate is about 0.011.

Limitations:

- Failure variants are controlled perturbations around public docked poses.
- Scoring failure is intentionally synthetic and easy because it is represented by clash-injected, overly favorable Vina-like scores.
- This should be reported as a stress/ablation benchmark, not as the final generated-candidate benchmark.

Next step:

- Add one more validation layer with real generated public candidates from Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff, or regenerate a small public pool specifically for this paper.
- Add a stricter scoring-failure variant based on real redocking/rescoring disagreement rather than the current controlled score hack.
- Keep the v2 results as the clean public controlled-failure ablation table.

## 2026-05-07: Official DiffSBDD Public Generated Sample Audit

Purpose:

- Add a real generated-candidate layer, separate from the DUD-E controlled stress test.
- Use official DiffSBDD sampled molecules from the public Zenodo record linked by the DiffSBDD repository.
- This is a generated SDF audit, not yet a full pocket-conditioned benchmark with exact CrossDocked receptor files.

Source:

- DiffSBDD repository: `https://github.com/arneschneuing/DiffSBDD`
- Zenodo sampled molecules: `https://zenodo.org/record/8239058`
- Downloaded file: `crossdocked_fullatom_cond.zip`
- Remote cache: `/home/test/wsk/public_generators/diffsbdd_zenodo_8239058`

Script added:

- `scripts/audit_diffsbdd_public_samples.py`

Setup:

- SDF files: official `crossdocked_fullatom_cond/*_gen.sdf`
- Valid SDF pocket files audited: 100
- Molecules per file: 100
- Total generated candidates: 10,000
- Receptors:
  - inferred from CrossDocked-style filenames, e.g. `2zen-A-rec-...`
  - downloaded as public RCSB PDB files from `https://files.rcsb.org/view/{PDB}.pdb`
  - this is an approximation because exact CrossDocked prepared receptor files are not yet available locally.

Overall audit:

| Metric | Value |
|---|---:|
| Candidates | 10,000 |
| Pockets | 100 |
| RDKit sanitize rate | 1.0000 |
| Has 3D coordinates | 1.0000 |
| Reliable proxy rate, all pockets | 0.3816 |
| Chemical failure proxy | 0.0010 |
| Geometric failure proxy | 0.0143 |
| Pocket failure proxy | 0.5358 |
| Clash failure proxy | 0.0761 |
| Synthetic failure proxy | 0.0093 |
| Median QED | 0.4753 |
| Median SA proxy | 4.0607 |

RCSB alignment strata:

| Alignment status | Pockets | Candidates | Mean reliable proxy | Mean pocket failure | Mean clash failure | Median min receptor distance | Median contacts 4.5A |
|---|---:|---:|---:|---:|---:|---:|---:|
| usable public PDB alignment | 39 | 3,900 | 0.9400 | 0.0038 | 0.0397 | 2.6349 | 23.0 |
| coordinate mismatch | 44 | 4,400 | 0.0000 | 1.0000 | 0.0000 | 54.1240 | 0.0 |
| ambiguous | 10 | 1,000 | 0.0570 | 0.9410 | 0.0010 | 7.8246 | 0.0 |
| high clash alignment | 7 | 700 | 0.1329 | 0.0029 | 0.8643 | 0.6631 | 18.0 |

Interpretation:

- The official DiffSBDD generated SDFs are chemically well-formed in this sample: RDKit sanitize rate is 1.0.
- The generated molecules also carry 3D coordinates.
- Direct RCSB receptors are not always coordinate-compatible with the CrossDocked prepared receptors used to condition DiffSBDD:
  44/100 pockets are clear coordinate mismatches.
- On the 39 pockets where public RCSB coordinates appear usable, generated candidates have a high reliable proxy rate of 0.94.
- Therefore the real generated-candidate story is promising, but exact receptor provenance matters. We must not report all-pocket pocket-failure rates without an alignment filter.

Next step:

- Find or reconstruct exact CrossDocked prepared receptor files for the DiffSBDD sampled pockets.
- Until then, use the 39 aligned public-PDB pockets as a conservative real generated SDF audit subset.
- Add Vina/PoseBusters-style checks on the aligned subset before claiming real generated benchmark performance.

## 2026-05-07: DiffSBDD Aligned Public-PDB Vina Score-Only Check

Purpose:

- Test whether a conventional physics score can act as a conservative post-generation filter on real public generated candidates.
- Keep the analysis limited to the 39 DiffSBDD pockets marked `usable_public_pdb_alignment` in the public audit, so receptor-coordinate mismatch does not dominate the result.
- Treat this as a reranking/ablation layer, not as the final exact CrossDocked benchmark.

Script added:

- `scripts/score_diffsbdd_aligned_vina.py`

Command:

```bash
/home/test/miniconda3/envs/genmol-eval2/bin/python scripts/score_diffsbdd_aligned_vina.py \
  --max-pockets 39 \
  --max-mols-per-pocket 100 \
  --workers 96 \
  --output-dir outputs/diffsbdd_aligned_vina_v0
```

Outputs:

- `outputs/diffsbdd_aligned_vina_v0/reports/vina_score_summary.csv`
- `outputs/diffsbdd_aligned_vina_v0/reports/vina_reranking_summary.csv`
- `outputs/diffsbdd_aligned_vina_v0/reports/vina_diagnostic_report.md`

Overall:

| Metric | Value |
|---|---:|
| Candidates | 3,900 |
| Pockets | 39 |
| Ligand prep success rate | 0.9903 |
| Vina score-only success rate | 0.9903 |
| All-candidate reliable proxy rate | 0.9400 |
| Median Vina affinity, reliable | -4.087 |
| Median Vina affinity, unreliable | 17.311 |

Vina reranking:

| Selection | Selected | Reliable proxy | Chemical pass | Geometry pass | Pocket pass | Clash-free |
|---|---:|---:|---:|---:|---:|---:|
| Top 10% per pocket | 390 | 0.9923 | 1.0000 | 0.9974 | 1.0000 | 1.0000 |
| Top 20% per pocket | 778 | 0.9884 | 1.0000 | 0.9923 | 0.9987 | 1.0000 |
| Top 50% per pocket | 1,938 | 0.9840 | 1.0000 | 0.9907 | 0.9974 | 0.9974 |

Diagnostic metrics, success-only:

| Metric | Value |
|---|---:|
| ROC-AUC for reliable proxy using `-vina_affinity` | 0.8658 |
| PR-AUC for reliable class | 0.9866 |
| PR-AUC for unreliable class | 0.4129 |
| Mean pocket top-10% reliable proxy | 0.9923 |
| Worst pocket top-10% reliable proxy | 0.8000 |

Interpretation:

- On the aligned public subset, official DiffSBDD samples are already strong by the current reliability proxy: 0.94 reliable.
- Vina score-only still improves selection: top 10% per pocket reaches 0.9923 reliable proxy, a +5.23 percentage-point gain over the aligned pool.
- The gain is mainly from rejecting clash/geometric outliers. Chemistry is not the bottleneck here because the sample is chemically clean.
- The unreliable median Vina affinity is strongly positive, which means Vina is heavily penalizing some bad poses rather than separating subtle near-native alternatives.
- This is useful as an external physics reranker baseline, but not enough to claim a full generated benchmark until exact CrossDocked prepared receptors are recovered or reconstructed.

Next step:

- Use this Vina score-only layer as a baseline in the real-generated-candidate section.
- Add a ReliaMol ranking pass on the same 39 aligned pockets and compare top-k reliability against Vina.
- Continue searching for exact CrossDocked receptor assets; if found, rerun the audit without the public-PDB alignment filter.

## 2026-05-07: DiffSBDD Aligned Public-PDB ReliaMol Reranking

Purpose:

- Compare ReliaMol-style no-leak reliability reranking against the Vina score-only baseline on the same aligned official DiffSBDD public subset.
- Use pocket-heldout out-of-fold predictions so every pocket is scored by models trained without that pocket.
- Avoid direct audit-threshold leakage: model features exclude `chemical_failure_proxy`, `geometric_failure_proxy`, `pocket_failure_proxy`, `clash_failure_proxy`, and `synthetic_failure_proxy`.

Script added:

- `scripts/run_diffsbdd_aligned_reliamol.py`

Command:

```bash
CUDA_VISIBLE_DEVICES=2 /home/test/miniconda3/envs/genmol-eval2/bin/python scripts/run_diffsbdd_aligned_reliamol.py \
  --seeds 11 22 33 \
  --n-folds 5 \
  --epochs 90 \
  --rf-trees 260 \
  --hgb-iter 220 \
  --rf-jobs 16 \
  --output-dir outputs/diffsbdd_aligned_reliamol_v0
```

Setup:

- Input: `outputs/diffsbdd_aligned_vina_v0/reports/vina_score_results.csv`
- Vina-success candidates: 3,862
- Pockets: 39
- Success-only reliable proxy rate: 0.9407
- Pocket geometry features use a fixed public-PDB pocket around the median generated center per pocket, rather than per-candidate label thresholds.

Out-of-fold ranking metrics:

| Method | ROC-AUC | PR-AUC | Top 10% reliable | Top 20% reliable | Top 50% reliable |
|---|---:|---:|---:|---:|---:|
| Vina score-only | 0.8658 +/- 0.0000 | 0.9866 +/- 0.0000 | 0.9923 +/- 0.0000 | 0.9884 +/- 0.0000 | 0.9840 +/- 0.0000 |
| QED/SA/Vina heuristic | 0.8705 +/- 0.0000 | 0.9879 +/- 0.0000 | 0.9949 +/- 0.0000 | 0.9884 +/- 0.0000 | 0.9814 +/- 0.0000 |
| MLP score-only | 0.8542 +/- 0.0022 | 0.9844 +/- 0.0002 | 0.9923 +/- 0.0000 | 0.9884 +/- 0.0000 | 0.9840 +/- 0.0000 |
| MLP ligand descriptor | 0.5700 +/- 0.0216 | 0.9519 +/- 0.0020 | 0.9821 +/- 0.0044 | 0.9666 +/- 0.0034 | 0.9567 +/- 0.0022 |
| MLP pose shape | 0.9379 +/- 0.0185 | 0.9949 +/- 0.0026 | 0.9957 +/- 0.0015 | 0.9927 +/- 0.0032 | 0.9880 +/- 0.0011 |
| MLP ReliaMol no-leak | 0.9422 +/- 0.0144 | 0.9953 +/- 0.0018 | 0.9974 +/- 0.0026 | 0.9953 +/- 0.0020 | 0.9907 +/- 0.0022 |
| RF ReliaMol no-leak | 0.9593 +/- 0.0054 | 0.9974 +/- 0.0004 | 1.0000 +/- 0.0000 | 0.9983 +/- 0.0007 | 0.9936 +/- 0.0006 |
| HGB ReliaMol no-leak | 0.9586 +/- 0.0149 | 0.9969 +/- 0.0016 | 0.9974 +/- 0.0044 | 0.9940 +/- 0.0039 | 0.9897 +/- 0.0010 |

Interpretation:

- ReliaMol no-leak features improve over Vina score-only on this real generated aligned subset, but the top-k margin is necessarily small because the aligned pool is already strong.
- The useful signal is mostly pose-shape and fixed-pocket distance distribution:
  MLP pose shape alone reaches 0.9379 ROC-AUC, while ligand descriptors alone are weak at 0.5700 ROC-AUC.
- Adding Vina and ligand descriptors to pose-shape gives a modest MLP gain: 0.9422 ROC-AUC and 0.9974 top-10% reliable.
- Tree ensembles on the same no-leak feature family are strongest here: RF reaches 1.0000 top-10% reliable and HGB reaches 0.9586 ROC-AUC / 0.9974 top-10% reliable.
- This supports a conservative claim: ReliaMol-style geometric reliability reranking adds value beyond Vina on public generated samples, especially by filtering residual pose/clash outliers.
- It does not yet prove broad generator superiority, because only 39 coordinate-compatible public-PDB pockets are included.

Next step:

- Add a pocket-radius sensitivity check for the fixed public-PDB pocket definition.
- Continue searching for exact CrossDocked prepared receptor assets to remove the current alignment filter.
- Once exact receptors are available, rerun both DiffSBDD audit and ReliaMol/Vina reranking on all usable official samples.

## 2026-05-07: DiffSBDD Aligned ReliaMol Pocket-Radius Sensitivity

Purpose:

- Check whether the ReliaMol reranking gain depends on the arbitrary fixed public-PDB pocket radius used to build pose-shape features.
- Run a lightweight single-seed sensitivity at radii 10A, 12A, and 16A.
- Use the same seed and reduced training budget for all three radius values, so this is a robustness check rather than the primary evidence table.

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=2 /home/test/miniconda3/envs/genmol-eval2/bin/python scripts/run_diffsbdd_aligned_reliamol.py \
  --seeds 44 \
  --n-folds 5 \
  --epochs 60 \
  --rf-trees 220 \
  --hgb-iter 180 \
  --rf-jobs 16 \
  --pocket-radius {10,12,16} \
  --output-dir outputs/diffsbdd_aligned_reliamol_radius{R}_seed44
```

Summary:

| Pocket radius | Method | ROC-AUC | PR-AUC | Top 10% reliable | Top 20% reliable |
|---:|---|---:|---:|---:|---:|
| 10A | Vina score-only | 0.8658 | 0.9866 | 0.9923 | 0.9884 |
| 10A | QED/SA/Vina heuristic | 0.8705 | 0.9879 | 0.9949 | 0.9884 |
| 10A | MLP pose shape | 0.9435 | 0.9958 | 0.9974 | 0.9936 |
| 10A | MLP ReliaMol no-leak | 0.9373 | 0.9952 | 0.9949 | 0.9936 |
| 10A | RF ReliaMol no-leak | 0.9516 | 0.9965 | 0.9974 | 0.9961 |
| 16A | Vina score-only | 0.8658 | 0.9866 | 0.9923 | 0.9884 |
| 16A | QED/SA/Vina heuristic | 0.8705 | 0.9879 | 0.9949 | 0.9884 |
| 16A | MLP pose shape | 0.9274 | 0.9890 | 0.9949 | 0.9897 |
| 16A | MLP ReliaMol no-leak | 0.9409 | 0.9951 | 0.9974 | 0.9936 |
| 16A | RF ReliaMol no-leak | 0.9546 | 0.9970 | 1.0000 | 0.9974 |

Interpretation:

- The reranking result is not fragile to the fixed-pocket radius over 10A to 16A.
- Vina and QED/SA/Vina are radius-independent baselines; ReliaMol pose features remain stronger than these baselines at the tested radii.
- MLP pose-shape alone varies more with radius than full no-leak ReliaMol or RF, which suggests the combined feature family is a safer default than a pure geometry-only model.
- The 12A default remains reasonable, but exact CrossDocked receptors are still the cleaner long-term solution.

Next step:

- Search for exact CrossDocked prepared receptor files or a reproducible reconstruction path.
- If unavailable, report the 39-pocket public-PDB aligned result transparently and avoid all-pocket claims.

## 2026-05-08: DiffSBDD CrossDocked Receptor Gninatypes Audit

Purpose:

- Replace the public RCSB receptor approximation with CrossDocked2020 receptor coordinates for the official DiffSBDD sampled molecules.
- Test whether the previous all-pocket pocket-failure problem was caused by receptor-coordinate provenance rather than generated molecule placement.
- Keep this as a receptor-provenance audit: `.gninatypes` gives exact CrossDocked atom coordinates, but not directly a Vina-ready PDBQT receptor.

Source:

- CrossDocked receptor archive: `https://bits.csb.pitt.edu/files/crossdock2020/v1.0/CrossDocked2020_receptors.tgz`
- Remote cache: `/home/test/wsk/public_generators/crossdocked2020_receptors_v1_0/CrossDocked2020_receptors.tgz`
- Archive size: 2.7GB
- Tar index entries: 27,644

Script added:

- `scripts/audit_diffsbdd_crossdocked_gninatypes.py`

Setup:

- Input SDFs: official DiffSBDD `crossdocked_fullatom_cond/*_gen.sdf`
- Receptor matching rule: sample filename receptor token `pdb-chain-rec` -> CrossDocked member basename `pdb_chain_rec_0.gninatypes`
- Matching receptors: 100/100
- Candidates: 10,000
- `.gninatypes` parser: little-endian float32 `x/y/z` plus int32 atom type records
- Receptor hydrogen types 0/1 are excluded by default for heavy-atom geometry auditing.

Command:

```bash
/home/test/miniconda3/envs/genmol-eval2/bin/python scripts/audit_diffsbdd_crossdocked_gninatypes.py \
  --max-files 101 \
  --max-mols-per-file 100 \
  --workers 32 \
  --output-dir outputs/diffsbdd_crossdocked_gninatypes_audit_v0
```

Overall comparison:

| Receptor source | Candidates | Pockets | Reliable proxy | Pocket failure | Clash failure | Median min dist | Median contacts 4.5A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public RCSB approximation | 10,000 | 100 | 0.3816 | 0.5358 | 0.0761 | 6.9927 | 0.0 |
| CrossDocked gninatypes heavy receptor | 10,000 | 100 | 0.9304 | 0.0000 | 0.0502 | 2.6063 | 23.0 |

By previous public-RCSB alignment stratum:

| Previous status | Pockets | RCSB reliable | CrossDocked reliable | Delta | RCSB pocket failure | CrossDocked pocket failure | CrossDocked clash failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| coordinate mismatch | 44 | 0.0000 | 0.9341 | +0.9341 | 1.0000 | 0.0000 | 0.0455 |
| ambiguous | 10 | 0.0570 | 0.9200 | +0.8630 | 0.9410 | 0.0000 | 0.0540 |
| high clash alignment | 7 | 0.1329 | 0.9457 | +0.8129 | 0.0029 | 0.0000 | 0.0329 |
| usable public PDB alignment | 39 | 0.9400 | 0.9262 | -0.0138 | 0.0038 | 0.0000 | 0.0577 |

Interpretation:

- The exact CrossDocked receptor coordinates resolve the core provenance problem:
  pocket failure drops from 0.5358 to 0.0000 over all 100 official DiffSBDD sampled pockets.
- The former public-RCSB `coordinate_mismatch` stratum was not a generated-molecule failure:
  it improves from 0.0000 to 0.9341 reliable proxy when using CrossDocked receptor coordinates.
- The official DiffSBDD samples remain chemically clean: sanitize rate is 1.0000 and chemical failure is 0.0010.
- Residual failures are mostly clash/geometric outliers under the current heavy-receptor proxy:
  clash failure is 0.0502 and geometric failure is 0.0143.
- The earlier 39-pocket aligned public-PDB result was directionally valid but overly conservative.
  With exact CrossDocked coordinates, we can report a 100-pocket generated-candidate audit instead of restricting to the aligned subset.

Next step:

- Build an all-100-pocket ReliaMol pose-shape reranking experiment using the CrossDocked gninatypes audit features.
- Keep Vina score-only as a 39-pocket public-PDB/PDBQT baseline unless a Vina-ready exact CrossDocked receptor format is recovered.
- If a PDB/PDBQT version of the exact CrossDocked receptors becomes available, rerun Vina and ReliaMol on the same 100-pocket exact receptor set.

## 2026-05-08: DiffSBDD CrossDocked 100-Pocket ReliaMol Reranking

Purpose:

- Run ReliaMol-style reranking on the full 100-pocket official DiffSBDD sample set after resolving receptor provenance with CrossDocked `.gninatypes`.
- Use pocket-heldout out-of-fold predictions: each pocket is scored by models trained without that pocket.
- Avoid direct proxy leakage: model features exclude direct failure flags and direct clash/contact counts.
- Since `.gninatypes` is not Vina-ready PDBQT, compare against QED and QED/SA heuristics rather than forcing a mismatched Vina baseline.

Script added:

- `scripts/run_diffsbdd_crossdocked_reliamol.py`

Command:

```bash
CUDA_VISIBLE_DEVICES=2 /home/test/miniconda3/envs/genmol-eval2/bin/python scripts/run_diffsbdd_crossdocked_reliamol.py \
  --seeds 11 22 33 \
  --n-folds 5 \
  --epochs 90 \
  --rf-trees 260 \
  --hgb-iter 220 \
  --rf-jobs 16 \
  --output-dir outputs/diffsbdd_crossdocked_reliamol_v0
```

Setup:

- Input audit: `outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports/candidate_audit.csv`
- Candidates: 10,000
- Pockets: 100
- Reliable proxy rate before reranking: 0.9304
- Pocket definition for pose-shape features: fixed 12A CrossDocked heavy-receptor neighborhood around the median generated center per pocket.

Out-of-fold metrics:

| Method | ROC-AUC | PR-AUC | MCC | ECE |
|---|---:|---:|---:|---:|
| QED | 0.6226 +/- 0.0000 | 0.9508 +/- 0.0000 | 0.1099 | 0.4308 |
| QED/SA | 0.6455 +/- 0.0000 | 0.9548 +/- 0.0000 | 0.3090 | 0.4304 |
| MLP ligand descriptor | 0.6695 +/- 0.0027 | 0.9561 +/- 0.0011 | 0.2932 | 0.1348 |
| MLP pose shape | 0.9724 +/- 0.0015 | 0.9974 +/- 0.0002 | 0.7068 | 0.0236 |
| MLP ReliaMol no-Vina | 0.9737 +/- 0.0026 | 0.9974 +/- 0.0007 | 0.7158 | 0.0185 |
| RF ReliaMol no-Vina | 0.9819 +/- 0.0014 | 0.9986 +/- 0.0002 | 0.7565 | 0.0189 |
| HGB ReliaMol no-Vina | 0.9845 +/- 0.0020 | 0.9984 +/- 0.0008 | 0.7895 | 0.0163 |

Top-k reranking:

| Method | Top 10% reliable | Top 20% reliable | Top 50% reliable | Top 10% clash-free |
|---|---:|---:|---:|---:|
| QED | 0.9550 +/- 0.0000 | 0.9520 +/- 0.0000 | 0.9526 +/- 0.0000 | 0.9630 |
| QED/SA | 0.9660 +/- 0.0000 | 0.9615 +/- 0.0000 | 0.9628 +/- 0.0000 | 0.9730 |
| MLP ligand descriptor | 0.9703 +/- 0.0025 | 0.9715 +/- 0.0038 | 0.9609 +/- 0.0008 | 0.9747 |
| MLP pose shape | 0.9987 +/- 0.0006 | 0.9980 +/- 0.0005 | 0.9951 +/- 0.0010 | 0.9990 |
| MLP ReliaMol no-Vina | 0.9980 +/- 0.0020 | 0.9982 +/- 0.0003 | 0.9949 +/- 0.0006 | 0.9990 |
| RF ReliaMol no-Vina | 0.9990 +/- 0.0000 | 0.9988 +/- 0.0003 | 0.9959 +/- 0.0002 | 0.9990 |
| HGB ReliaMol no-Vina | 0.9993 +/- 0.0012 | 0.9983 +/- 0.0003 | 0.9961 +/- 0.0004 | 0.9997 |

Interpretation:

- The full 100-pocket exact-receptor experiment strongly supports the core ReliaMol claim:
  pose-aware reliability ranking is much better than molecule-only heuristics for filtering generated candidates.
- QED/SA improves over raw pool reliability from 0.9304 to 0.9660 at top 10%, but cannot consistently avoid pocket-specific clash/geometric failures.
- Pose-shape alone is already highly predictive: 0.9724 ROC-AUC and 0.9987 top-10% reliable.
- Adding ligand descriptors gives similar MLP performance, while tree ensembles on the same no-Vina feature family are strongest:
  HGB reaches 0.9845 ROC-AUC and 0.9993 top-10% reliable.
- The worst-pocket top-10% rate for ReliaMol variants is 0.9, while QED/SA drops to 0.6 on a low-quality pocket. This is a useful reviewer-facing robustness point.

Next step:

- Treat this as the main real generated-candidate benchmark table.
- Keep the DUD-E controlled-failure experiment as the stress-test ablation table.
- Add a concise methods note explaining why Vina is reported on the 39 public-PDB/PDBQT-compatible subset, while ReliaMol no-Vina is reported on the full 100 exact CrossDocked `.gninatypes` subset.

