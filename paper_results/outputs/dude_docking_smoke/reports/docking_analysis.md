# DUD-E Public Docking Smoke Analysis

## Raw Data Table: Pose Quality By Target/Class

| target | source_class | n | success_rate | public_pose_reliable_rate | median_vina_score | iqr_vina_score | median_center_to_crystal | frac_center_within_8A | frac_any_pocket_contact_4p5 | frac_any_receptor_clash_1p2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adrb2 | active | 20 | 1.0000 | 0.8000 | -7.8790 | 1.3770 | 4.5006 | 0.8000 | 1.0000 | 0.0000 |
| adrb2 | decoy | 20 | 1.0000 | 0.9500 | -9.6625 | 0.7567 | 5.6107 | 0.9500 | 1.0000 | 0.0000 |
| bace1 | active | 20 | 1.0000 | 1.0000 | -7.4080 | 0.7955 | 3.6472 | 1.0000 | 1.0000 | 0.0000 |
| bace1 | decoy | 20 | 1.0000 | 1.0000 | -7.6525 | 1.5900 | 3.5469 | 1.0000 | 1.0000 | 0.0000 |
| cp3a4 | active | 20 | 1.0000 | 1.0000 | -8.7590 | 1.4597 | 2.4610 | 1.0000 | 1.0000 | 0.0000 |
| cp3a4 | decoy | 20 | 1.0000 | 1.0000 | -8.7335 | 1.8123 | 4.5518 | 1.0000 | 1.0000 | 0.0000 |

## Active/Decoy Vina Score Separation

| target | active_median_vina | decoy_median_vina | active_minus_decoy_median | p_active_scores_better_than_decoy | n_active | n_decoy |
| --- | --- | --- | --- | --- | --- | --- |
| adrb2 | -7.8790 | -9.6625 | 1.7835 | 0.1425 | 20 | 20 |
| bace1 | -7.4080 | -7.6525 | 0.2445 | 0.4825 | 20 | 20 |
| cp3a4 | -8.7590 | -8.7335 | -0.0255 | 0.5950 | 20 | 20 |

## Vina Top-k Behavior

| target | topk_frac | n_selected | active_fraction | public_pose_reliable_rate | median_vina_score |
| --- | --- | --- | --- | --- | --- |
| adrb2 | 0.1000 | 4 | 0.5000 | 1.0000 | -10.9900 |
| bace1 | 0.1000 | 4 | 0.5000 | 1.0000 | -9.3715 |
| cp3a4 | 0.1000 | 4 | 0.7500 | 1.0000 | -11.1250 |
| adrb2 | 0.2000 | 8 | 0.2500 | 1.0000 | -10.5550 |
| bace1 | 0.2000 | 8 | 0.3750 | 1.0000 | -9.0150 |
| cp3a4 | 0.2000 | 8 | 0.6250 | 1.0000 | -10.8350 |
| adrb2 | 0.5000 | 20 | 0.1500 | 0.9500 | -9.7275 |
| bace1 | 0.5000 | 20 | 0.4500 | 1.0000 | -8.4470 |
| cp3a4 | 0.5000 | 20 | 0.5000 | 1.0000 | -9.6885 |

## Key Findings

1. Docking pipeline is technically usable: mean success rate is 1.000, and mean public pose reliable rate is 0.958.
2. Pose placement is not uniformly tight: center-within-8A falls below 0.9 for adrb2/active, so the next run should use a larger sample and inspect box size/exhaustiveness sensitivity.
3. Vina score alone is weak as an active/decoy discriminator in this sample: active-better probability is below 0.6 for adrb2, bace1, cp3a4.

## Suggested Next Experiments

1. Expand the clean public docking run to all DUD-E targets that have both active and decoy SDF files.
2. Run box-size/exhaustiveness ablations before treating pocket failure labels as stable.
3. Train a no-leak reliability model on docked public poses, with source_class held out from features and leave-one-target validation.
4. Add a scoring-baseline section because this smoke already shows Vina may rank decoys as well as, or better than, actives.
