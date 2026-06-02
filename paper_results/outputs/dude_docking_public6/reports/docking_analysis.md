# DUD-E Public Docking Smoke Analysis

## Raw Data Table: Pose Quality By Target/Class

| target | source_class | n | success_rate | public_pose_reliable_rate | median_vina_score | iqr_vina_score | median_center_to_crystal | frac_center_within_8A | frac_any_pocket_contact_4p5 | frac_any_receptor_clash_1p2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adrb2 | active | 30 | 1.0000 | 0.8667 | -8.2775 | 3.9923 | 3.5733 | 0.8667 | 1.0000 | 0.0000 |
| adrb2 | decoy | 30 | 1.0000 | 0.9667 | -9.6625 | 0.7635 | 5.6835 | 0.9667 | 1.0000 | 0.0000 |
| bace1 | active | 30 | 1.0000 | 1.0000 | -7.4080 | 0.9680 | 3.6185 | 1.0000 | 1.0000 | 0.0000 |
| bace1 | decoy | 30 | 1.0000 | 1.0000 | -7.8965 | 1.5195 | 3.6955 | 1.0000 | 1.0000 | 0.0000 |
| cp3a4 | active | 30 | 1.0000 | 1.0000 | -9.0945 | 2.1395 | 2.5354 | 1.0000 | 1.0000 | 0.0000 |
| cp3a4 | decoy | 30 | 1.0000 | 1.0000 | -8.2880 | 1.2295 | 4.5387 | 1.0000 | 1.0000 | 0.0000 |
| pparg | active | 30 | 1.0000 | 0.9667 | -8.7670 | 1.0225 | 3.1666 | 0.9667 | 1.0000 | 0.0000 |
| pparg | decoy | 30 | 1.0000 | 0.9000 | -8.4145 | 1.6073 | 4.1381 | 0.9000 | 1.0000 | 0.0000 |
| src | active | 30 | 1.0000 | 1.0000 | -9.2435 | 0.7498 | 5.1393 | 1.0000 | 1.0000 | 0.0000 |
| src | decoy | 30 | 1.0000 | 1.0000 | -8.2175 | 1.1065 | 3.7089 | 1.0000 | 1.0000 | 0.0000 |
| try1 | active | 30 | 0.9667 | 0.9667 | -7.9730 | 1.3870 | 2.5735 | 1.0000 | 1.0000 | 0.0000 |
| try1 | decoy | 30 | 1.0000 | 0.9667 | -6.2485 | 0.5835 | 4.7686 | 1.0000 | 1.0000 | 0.0333 |

## Active/Decoy Vina Score Separation

| target | active_median_vina | decoy_median_vina | active_minus_decoy_median | p_active_scores_better_than_decoy | n_active | n_decoy |
| --- | --- | --- | --- | --- | --- | --- |
| adrb2 | -8.2775 | -9.6625 | 1.3850 | 0.3956 | 30 | 30 |
| bace1 | -7.4080 | -7.8965 | 0.4885 | 0.4011 | 30 | 30 |
| cp3a4 | -9.0945 | -8.2880 | -0.8065 | 0.6844 | 30 | 30 |
| pparg | -8.7670 | -8.4145 | -0.3525 | 0.6300 | 30 | 30 |
| src | -9.2435 | -8.2175 | -1.0260 | 0.7944 | 30 | 30 |
| try1 | -7.9730 | -6.2485 | -1.7245 | 0.9644 | 29 | 30 |

## Vina Top-k Behavior

| target | topk_frac | n_selected | active_fraction | public_pose_reliable_rate | median_vina_score |
| --- | --- | --- | --- | --- | --- |
| adrb2 | 0.1000 | 6 | 1.0000 | 1.0000 | -11.5050 |
| bace1 | 0.1000 | 6 | 0.3333 | 1.0000 | -9.3480 |
| cp3a4 | 0.1000 | 6 | 0.8333 | 1.0000 | -10.8650 |
| pparg | 0.1000 | 6 | 0.8333 | 1.0000 | -10.2200 |
| src | 0.1000 | 6 | 0.6667 | 1.0000 | -9.9975 |
| try1 | 0.1000 | 6 | 1.0000 | 1.0000 | -8.9680 |
| adrb2 | 0.2000 | 12 | 0.8333 | 1.0000 | -11.3800 |
| bace1 | 0.2000 | 12 | 0.3333 | 1.0000 | -8.9745 |
| cp3a4 | 0.2000 | 12 | 0.9167 | 1.0000 | -10.8050 |
| pparg | 0.2000 | 12 | 0.5833 | 1.0000 | -10.0900 |
| src | 0.2000 | 12 | 0.7500 | 1.0000 | -9.8605 |
| try1 | 0.2000 | 12 | 1.0000 | 1.0000 | -8.7850 |
| adrb2 | 0.5000 | 30 | 0.4000 | 1.0000 | -10.3500 |
| bace1 | 0.5000 | 30 | 0.4000 | 1.0000 | -8.3870 |
| cp3a4 | 0.5000 | 30 | 0.5667 | 1.0000 | -9.8335 |
| pparg | 0.5000 | 30 | 0.6000 | 1.0000 | -9.3465 |
| src | 0.5000 | 30 | 0.7667 | 1.0000 | -9.3535 |
| try1 | 0.5000 | 30 | 0.8667 | 1.0000 | -7.9605 |

## Key Findings

1. Docking pipeline is technically usable: mean success rate is 0.997, and mean public pose reliable rate is 0.969.
2. Pose placement is not uniformly tight: center-within-8A falls below 0.9 for adrb2/active, so the next run should use a larger sample and inspect box size/exhaustiveness sensitivity.
3. Vina score alone is weak as an active/decoy discriminator in this sample: active-better probability is below 0.6 for adrb2, bace1.

## Suggested Next Experiments

1. Expand the clean public docking run to all DUD-E targets that have both active and decoy SDF files.
2. Run box-size/exhaustiveness ablations before treating pocket failure labels as stable.
3. Train a no-leak reliability model on docked public poses, with source_class held out from features and leave-one-target validation.
4. Add a scoring-baseline section because this smoke already shows Vina may rank decoys as well as, or better than, actives.
