# DiffSBDD CrossDocked ReliaMol Reranking

Pocket-heldout out-of-fold evaluation on 100 official DiffSBDD pockets audited against CrossDocked gninatypes receptor coordinates.

## Metric Summary

method,roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std,f1_mean,f1_std,mcc_mean,mcc_std,ece_mean,ece_std
qed,0.6226439190658326,,0.9507849163578584,,0.9642653389347026,,0.10991968996872666,,0.43075822417758225,
qed_sa,0.6454983828485585,,0.9547755627100725,,0.9673958333333333,,0.30896399612613973,,0.4303999999999999,
mlp_ligand_descriptor,0.6667884317460788,,0.9532399749622806,,0.9674389480275517,,0.32194809840536354,,0.1452032967423089,
mlp_pose_shape,0.9756380273964478,,0.9979066228791733,,0.9824561403508771,,0.7424231511103967,,0.061783536152821034,
mlp_reliamol_novinascore,0.9783117630780482,,0.9981443982845032,,0.9835173627780445,,0.7423338532435761,,0.05998844332366568,
rf_reliamol_novinascore,0.977941448987458,,0.9980799492794111,,0.9824336590314485,,0.7275168297462169,,0.020639714618149216,
hgb_reliamol_novinascore,0.9827332947885472,,0.9982248331511172,,0.9845825553480928,,0.759661542501371,,0.008982978138755834,

## Top-10% Reranking

method,topk_frac,reliable_proxy_rate_mean,reliable_proxy_rate_std,chemical_pass_rate_mean,chemical_pass_rate_std,geometry_pass_rate_mean,geometry_pass_rate_std,pocket_pass_rate_mean,pocket_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,synthetic_pass_rate_mean,synthetic_pass_rate_std,diversity_proxy_mean,diversity_proxy_std
qed,0.1,0.955,,1.0,,0.992,,1.0,,0.963,,1.0,,10.0,
qed_sa,0.1,0.966,,0.999,,0.992,,1.0,,0.973,,1.0,,10.0,
mlp_ligand_descriptor,0.1,0.969,,1.0,,0.994,,1.0,,0.975,,1.0,,10.0,
mlp_pose_shape,0.1,0.997,,1.0,,1.0,,1.0,,0.997,,1.0,,10.0,
mlp_reliamol_novinascore,0.1,0.999,,1.0,,1.0,,1.0,,0.999,,1.0,,10.0,
rf_reliamol_novinascore,0.1,0.998,,1.0,,1.0,,1.0,,0.998,,1.0,,10.0,
hgb_reliamol_novinascore,0.1,0.998,,1.0,,1.0,,1.0,,0.998,,1.0,,10.0,
