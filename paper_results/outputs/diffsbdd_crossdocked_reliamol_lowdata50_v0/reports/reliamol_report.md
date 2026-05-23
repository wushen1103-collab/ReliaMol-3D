# DiffSBDD CrossDocked ReliaMol Reranking

Pocket-heldout out-of-fold evaluation on 100 official DiffSBDD pockets audited against CrossDocked gninatypes receptor coordinates.

## Metric Summary

method,roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std,f1_mean,f1_std,mcc_mean,mcc_std,ece_mean,ece_std
qed,0.6226439190658326,,0.9507849163578584,,0.9642653389347026,,0.10991968996872666,,0.43075822417758225,
qed_sa,0.6454983828485585,,0.9547755627100725,,0.9673958333333333,,0.30896399612613973,,0.4303999999999999,
mlp_ligand_descriptor,0.6659011758630573,,0.9537381170448078,,0.9671234304173396,,0.3010924875143047,,0.13821327642910183,
mlp_pose_shape,0.9657348279321216,,0.9951585570569864,,0.9814253020384267,,0.7025978683055003,,0.08587006478682158,
mlp_reliamol_novinascore,0.9762746495142369,,0.9979275308221935,,0.9828626341359245,,0.7340158852454403,,0.08815247428137814,
rf_reliamol_novinascore,0.9775288221108707,,0.9980761572086065,,0.9806278508183526,,0.7163285454030555,,0.020621054070787972,
hgb_reliamol_novinascore,0.9843584763937893,,0.9987669307227053,,0.9851915441570984,,0.7827747501869998,,0.011687571761804745,

## Top-10% Reranking

method,topk_frac,reliable_proxy_rate_mean,reliable_proxy_rate_std,chemical_pass_rate_mean,chemical_pass_rate_std,geometry_pass_rate_mean,geometry_pass_rate_std,pocket_pass_rate_mean,pocket_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,synthetic_pass_rate_mean,synthetic_pass_rate_std,diversity_proxy_mean,diversity_proxy_std
qed,0.1,0.955,,1.0,,0.992,,1.0,,0.963,,1.0,,10.0,
qed_sa,0.1,0.966,,0.999,,0.992,,1.0,,0.973,,1.0,,10.0,
mlp_ligand_descriptor,0.1,0.964,,0.998,,0.991,,1.0,,0.972,,1.0,,10.0,
mlp_pose_shape,0.1,0.992,,0.996,,0.998,,1.0,,0.996,,1.0,,10.0,
mlp_reliamol_novinascore,0.1,0.998,,1.0,,1.0,,1.0,,0.998,,1.0,,10.0,
rf_reliamol_novinascore,0.1,0.999,,1.0,,1.0,,1.0,,0.999,,1.0,,10.0,
hgb_reliamol_novinascore,0.1,0.999,,1.0,,1.0,,1.0,,0.999,,1.0,,10.0,
