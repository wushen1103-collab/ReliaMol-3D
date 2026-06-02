# DiffSBDD CrossDocked 100-Pocket Vina/GNINA Baseline

Score-only Vina and GNINA/CNN baselines on the same 100 exact CrossDocked pocket PDBs used for the gninatypes audit.

## Overall

n_candidates,n_pockets,ligand_prep_success_rate,vina_success_rate,gnina_success_rate,reliable_proxy_rate,median_vina_affinity,median_gnina_cnnscore
20,1,1.0,1.0,1.0,0.7,-2.0540000000000003,0.593895

## Score Metrics

roc_auc,pr_auc,method
0.4642857142857143,0.6998954243702142,vina
0.4761904761904762,0.704657329132119,gnina_affinity
0.5714285714285714,0.7561136284350569,gnina_cnnscore
0.5,0.7472378345277505,gnina_cnnaffinity

## Top-10% Score-Only Reranking

method,topk_frac,n_selected,reliable_proxy_rate,posebusters_pass_rate,clash_free_rate,median_vina_affinity,median_gnina_cnnscore
vina,0.1,2,0.5,0.5,0.5,-2.8015,0.641345
gnina_affinity,0.1,2,0.5,0.5,0.5,-2.7305,0.636815
gnina_cnnscore,0.1,2,1.0,1.0,1.0,-1.644,0.72671
gnina_cnnaffinity,0.1,2,1.0,0.5,1.0,-2.1790000000000003,0.483765

## Top-10% ReliaMol + Docking/CNN Ensembles

method,topk_frac,reliable_proxy_rate_mean,reliable_proxy_rate_std,posebusters_pass_rate_mean,posebusters_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,median_vina_affinity_mean,median_vina_affinity_std,median_gnina_cnnscore_mean,median_gnina_cnnscore_std
ensemble_qed_vina,0.1,1.0,0.0,0.0,0.0,1.0,0.0,,,,
ensemble_qed_gnina_cnnaffinity,0.1,1.0,0.0,0.5,0.0,1.0,0.0,,,,
ensemble_rf_reliamol_novinascore_gnina_cnnaffinity,0.1,1.0,0.0,0.6666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_qed_sa_vina,0.1,1.0,0.0,0.0,0.0,1.0,0.0,,,,
ensemble_mlp_pose_shape_gnina_cnnscore,0.1,1.0,0.0,0.8333333333333334,0.28867513459481287,1.0,0.0,,,,
ensemble_mlp_pose_shape_vina,0.1,1.0,0.0,0.16666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_mlp_pose_shape_gnina_cnnaffinity,0.1,1.0,0.0,0.16666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_mlp_reliamol_novinascore_vina,0.1,1.0,0.0,0.3333333333333333,0.2886751345948129,1.0,0.0,,,,
ensemble_mlp_reliamol_novinascore_gnina_cnnaffinity,0.1,1.0,0.0,0.16666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_mlp_reliamol_novinascore_gnina_cnnscore,0.1,1.0,0.0,0.6666666666666666,0.28867513459481287,1.0,0.0,,,,
ensemble_hgb_reliamol_novinascore_gnina_cnnscore,0.1,1.0,0.0,0.8333333333333334,0.28867513459481287,1.0,0.0,,,,
ensemble_hgb_reliamol_novinascore_vina,0.1,1.0,0.0,0.3333333333333333,0.2886751345948129,1.0,0.0,,,,
ensemble_rf_reliamol_novinascore_vina,0.1,1.0,0.0,0.3333333333333333,0.2886751345948129,1.0,0.0,,,,
ensemble_rf_reliamol_novinascore_gnina_cnnscore,0.1,1.0,0.0,0.6666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_hgb_reliamol_novinascore_gnina_cnnaffinity,0.1,1.0,0.0,0.6666666666666666,0.2886751345948129,1.0,0.0,,,,
ensemble_qed_gnina_cnnscore,0.1,0.5,0.0,0.5,0.0,0.5,0.0,,,,
ensemble_qed_sa_gnina_cnnaffinity,0.1,0.5,0.0,1.0,0.0,0.5,0.0,,,,
ensemble_qed_sa_gnina_cnnscore,0.1,0.0,0.0,1.0,0.0,0.0,0.0,,,,
