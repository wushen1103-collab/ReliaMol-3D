# DiffSBDD CrossDocked PoseBusters Validity

PoseBusters config: `mol_fast`. This is a molecule-internal physical validity audit; receptor-contact checks remain in the CrossDocked gninatypes audit.

## Overall

n_candidates,n_pockets,posebusters_pass_rate,reliability_proxy_rate,mol_pred_loaded_pass_rate,sanitization_pass_rate,inchi_convertible_pass_rate,all_atoms_connected_pass_rate,no_radicals_pass_rate,bond_lengths_pass_rate,bond_angles_pass_rate,internal_steric_clash_pass_rate,aromatic_ring_flatness_pass_rate,non-aromatic_ring_non-flatness_pass_rate,double_bond_flatness_pass_rate
10000,100,0.5732,0.9304,1.0,1.0,1.0,1.0,1.0,0.7565,0.704,0.8972,0.9987,0.9876,0.9917

## Top-10% Reranking

method,topk_frac,posebusters_pass_rate_mean,posebusters_pass_rate_std,reliable_proxy_rate_mean,reliable_proxy_rate_std
qed,0.1,0.706,0.0,0.955,0.0
qed_sa,0.1,0.7639999999999999,0.0,0.966,0.0
mlp_ligand_descriptor,0.1,0.7126666666666667,0.0102632028788937,0.9703333333333334,0.0025166114784236
mlp_pose_shape,0.1,0.749,0.0079372539331937,0.9986666666666668,0.0005773502691896
mlp_reliamol_novinascore,0.1,0.7033333333333333,0.0136503968196288,0.998,0.002
rf_reliamol_novinascore,0.1,0.7333333333333334,0.010598742063723,0.999,0.0
hgb_reliamol_novinascore,0.1,0.767,0.0108166538263919,0.9993333333333334,0.0011547005383792
