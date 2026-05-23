# DiffSBDD CrossDocked ReliaMol Reranking

Pocket-heldout out-of-fold evaluation on 100 official DiffSBDD pockets audited against CrossDocked gninatypes receptor coordinates.

## Metric Summary

method,roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std,f1_mean,f1_std,mcc_mean,mcc_std,ece_mean,ece_std
qed,0.6226439190658326,,0.9507849163578584,,0.9642653389347026,,0.10991968996872666,,0.43075822417758225,
qed_sa,0.6454983828485585,,0.9547755627100725,,0.9673958333333333,,0.30896399612613973,,0.4303999999999999,
mlp_ligand_descriptor,0.6615987839861239,,0.9546798176278916,,0.9671464330413017,,0.30795375023391275,,0.3794091983084526,
mlp_pose_shape,0.9479663919115249,,0.9944051918143645,,0.9775627392598895,,0.6343482335083678,,0.3843935645346602,
mlp_reliamol_novinascore,0.9454615985214616,,0.9936895298050521,,0.9769840004252379,,0.6234858732078837,,0.40769983625372513,
rf_reliamol_novinascore,0.972737207949121,,0.9970713218676204,,0.9821983914209115,,0.7367010084780535,,0.019932288611174333,
hgb_reliamol_novinascore,0.9752108072414782,,0.9972997400707162,,0.9859200171315381,,0.7881776519894658,,0.036505933746385874,

## Top-10% Reranking

method,topk_frac,reliable_proxy_rate_mean,reliable_proxy_rate_std,chemical_pass_rate_mean,chemical_pass_rate_std,geometry_pass_rate_mean,geometry_pass_rate_std,pocket_pass_rate_mean,pocket_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,synthetic_pass_rate_mean,synthetic_pass_rate_std,diversity_proxy_mean,diversity_proxy_std
qed,0.1,0.955,,1.0,,0.992,,1.0,,0.963,,1.0,,10.0,
qed_sa,0.1,0.966,,0.999,,0.992,,1.0,,0.973,,1.0,,10.0,
mlp_ligand_descriptor,0.1,0.968,,0.997,,0.993,,1.0,,0.975,,1.0,,10.0,
mlp_pose_shape,0.1,0.997,,0.998,,1.0,,1.0,,0.999,,1.0,,10.0,
mlp_reliamol_novinascore,0.1,0.993,,0.998,,0.999,,1.0,,0.996,,1.0,,10.0,
rf_reliamol_novinascore,0.1,0.996,,1.0,,0.998,,1.0,,0.998,,1.0,,10.0,
hgb_reliamol_novinascore,0.1,0.998,,1.0,,0.998,,1.0,,1.0,,1.0,,10.0,
