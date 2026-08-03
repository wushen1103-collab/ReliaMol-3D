# JCTC Priority Evidence Summary

- CrossDocked physical endpoint: 10000 candidates across 100 pockets.
- Public-PDB aligned physical endpoint: 3862 candidates across 39 pockets.
- Aligned relaxation success rate: 0.9591; final PB pass rate: 0.8951; stable endpoint rate: 0.8294.

## CrossDocked Top-10% Physical Endpoint
| source | method | coverage | n | stable_rate_mean | stable_rate_sd | relaxation_success_rate_mean | relaxation_success_rate_sd | final_pb_pass_rate_mean | final_pb_pass_rate_sd | median_ligand_rmsd_a_mean | median_ligand_rmsd_a_sd | mean_contact_retention_mean | mean_contact_retention_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audit_label_oof | qed_sa | 0.1000 | 3 | 0.8700 | 0.0000 | 0.9490 | 0.0000 | 0.9130 | 0.0000 | 0.7125 | 0.0000 | 0.7592 | 0.0000 |
| audit_label_oof | label_informed_rule | 0.1000 | 3 | 0.8680 | 0.0000 | 0.9470 | 0.0000 | 0.9090 | 0.0000 | 0.7929 | 0.0000 | 0.7404 | 0.0000 |
| audit_label_oof | hgb_full | 0.1000 | 3 | 0.8813 | 0.0146 | 0.9453 | 0.0071 | 0.9060 | 0.0087 | 0.7528 | 0.0085 | 0.7626 | 0.0051 |
| audit_label_oof | rf_full | 0.1000 | 3 | 0.8887 | 0.0029 | 0.9513 | 0.0061 | 0.9057 | 0.0031 | 0.7771 | 0.0022 | 0.7599 | 0.0022 |
| physics_endpoint_oof | phys_rf_direct_excluded | 0.1000 | 3 | 0.9367 | 0.0038 | 0.9647 | 0.0025 | 0.9440 | 0.0052 | 0.6770 | 0.0090 | 0.7916 | 0.0037 |
| physics_endpoint_oof | phys_rf_full | 0.1000 | 3 | 0.9343 | 0.0067 | 0.9650 | 0.0030 | 0.9407 | 0.0059 | 0.6578 | 0.0069 | 0.7961 | 0.0006 |

## Public-PDB Aligned Zero-Shot Top-10%
| source | method | coverage | n | stable_rate_mean | stable_rate_sd | relaxation_success_rate_mean | relaxation_success_rate_sd | final_pb_pass_rate_mean | final_pb_pass_rate_sd | median_ligand_rmsd_a_mean | median_ligand_rmsd_a_sd | mean_contact_retention_mean | mean_contact_retention_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_public_pdb_zero_shot | qed_sa | 0.1000 | 3 | 0.8846 | 0.0000 | 0.9821 | 0.0000 | 0.9513 | 0.0000 | 0.5492 | 0.0000 | 0.7906 | 0.0000 |
| aligned_public_pdb_zero_shot | label_informed_rule | 0.1000 | 3 | 0.8692 | 0.0000 | 0.9718 | 0.0000 | 0.9256 | 0.0000 | 0.5965 | 0.0000 | 0.7840 | 0.0000 |
| aligned_public_pdb_zero_shot | rf_full | 0.1000 | 3 | 0.8094 | 0.0053 | 0.9077 | 0.0128 | 0.8581 | 0.0104 | 0.6233 | 0.0091 | 0.7837 | 0.0025 |
| aligned_public_pdb_zero_shot | hgb_full | 0.1000 | 3 | 0.8991 | 0.0107 | 0.9803 | 0.0082 | 0.9427 | 0.0074 | 0.6002 | 0.0184 | 0.7954 | 0.0045 |

## Endpoint Overlap
| audit_reliable | physical_stable | count | fraction |
| --- | --- | --- | --- |
| False | False | 352 | 0.0352 |
| False | True | 344 | 0.0344 |
| True | False | 1550 | 0.1550 |
| True | True | 7754 | 0.7754 |
| all | agreement | 8106 | 0.8106 |
