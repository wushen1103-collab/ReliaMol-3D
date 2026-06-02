# DiffSBDD CrossDocked ReliaMol Reranking

Pocket-heldout out-of-fold evaluation on 100 official DiffSBDD pockets audited against CrossDocked gninatypes receptor coordinates.

## Metric Summary

method,roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std,f1_mean,f1_std,mcc_mean,mcc_std,ece_mean,ece_std
qed,0.6226439190658326,,0.9507849163578584,,0.9642653389347026,,0.10991968996872666,,0.43075822417758225,
qed_sa,0.6454983828485585,,0.9547755627100725,,0.9673958333333333,,0.30896399612613973,,0.4303999999999999,
mlp_ligand_descriptor,0.6481575252517765,,0.9498009485828767,,0.9663178718309126,,0.2648924259381311,,0.16049069134168095,
mlp_pose_shape,0.9640693410818235,,0.9949478690382436,,0.9810431616150798,,0.715141618380444,,0.15059329455461937,
mlp_reliamol_novinascore,0.9639478076417509,,0.9951088011429172,,0.9812834224598931,,0.7142001479406873,,0.1473524189195252,
rf_reliamol_novinascore,0.9751938975697019,,0.9978564212401618,,0.9797431626457471,,0.7083268855870363,,0.023216235896829883,
hgb_reliamol_novinascore,0.9795781353465571,,0.9982220161576483,,0.9834838847613449,,0.7468568308161918,,0.01495976888271122,

## Top-10% Reranking

method,topk_frac,reliable_proxy_rate_mean,reliable_proxy_rate_std,chemical_pass_rate_mean,chemical_pass_rate_std,geometry_pass_rate_mean,geometry_pass_rate_std,pocket_pass_rate_mean,pocket_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,synthetic_pass_rate_mean,synthetic_pass_rate_std,diversity_proxy_mean,diversity_proxy_std
qed,0.1,0.955,,1.0,,0.992,,1.0,,0.963,,1.0,,10.0,
qed_sa,0.1,0.966,,0.999,,0.992,,1.0,,0.973,,1.0,,10.0,
mlp_ligand_descriptor,0.1,0.956,,0.994,,0.989,,1.0,,0.969,,1.0,,10.0,
mlp_pose_shape,0.1,0.994,,0.997,,0.998,,1.0,,0.997,,0.999,,10.0,
mlp_reliamol_novinascore,0.1,0.996,,0.999,,0.998,,1.0,,0.999,,1.0,,10.0,
rf_reliamol_novinascore,0.1,0.997,,1.0,,1.0,,1.0,,0.997,,1.0,,10.0,
hgb_reliamol_novinascore,0.1,1.0,,1.0,,1.0,,1.0,,1.0,,1.0,,10.0,
