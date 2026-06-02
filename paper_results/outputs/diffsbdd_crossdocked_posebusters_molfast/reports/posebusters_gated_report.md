# PoseBusters-Gated Reranking

Hard-gate policy selects only PoseBusters-pass molecules; fill-k policy first takes PoseBusters-pass molecules and fills any shortfall by model score to preserve exactly top-k count.

## Top-10 Summary

method,topk_frac,policy,n_selected_mean,n_selected_std,mean_selected_per_pocket_mean,mean_selected_per_pocket_std,posebusters_pass_rate_mean,posebusters_pass_rate_std,reliable_proxy_rate_mean,reliable_proxy_rate_std,clash_free_rate_mean,clash_free_rate_std
qed,0.1,raw,1000.0,0.0,10.0,0.0,0.706,0.0,0.955,0.0,,
qed,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9739999999999999,0.0,,
qed,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9739999999999999,0.0,,
qed_sa,0.1,raw,1000.0,0.0,10.0,0.0,0.7639999999999999,0.0,0.9659999999999999,0.0,,
qed_sa,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.98,0.0,,
qed_sa,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.98,0.0,,
mlp_ligand_descriptor,0.1,raw,1000.0,0.0,10.0,0.0,0.7126666666666667,0.010263202878893762,0.9703333333333334,0.0025166114784236073,,
mlp_ligand_descriptor,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9763333333333333,0.0025166114784236407,,
mlp_ligand_descriptor,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9763333333333333,0.0025166114784236407,,
mlp_pose_shape,0.1,raw,1000.0,0.0,10.0,0.0,0.749,0.00793725393319379,0.9986666666666667,0.0005773502691896744,,
mlp_pose_shape,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9993333333333334,0.0005773502691896423,,
mlp_pose_shape,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9993333333333334,0.0005773502691896423,,
mlp_reliamol_novinascore,0.1,raw,1000.0,0.0,10.0,0.0,0.7033333333333333,0.013650396819628808,0.9979999999999999,0.0020000000000000018,,
mlp_reliamol_novinascore,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.999,0.0010000000000000286,,
mlp_reliamol_novinascore,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.999,0.0010000000000000286,,
rf_reliamol_novinascore,0.1,raw,1000.0,0.0,10.0,0.0,0.7333333333333334,0.01059874206372309,0.999,0.0,,
rf_reliamol_novinascore,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,1.0,0.0,,
rf_reliamol_novinascore,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,1.0,0.0,,
hgb_reliamol_novinascore,0.1,raw,1000.0,0.0,10.0,0.0,0.767,0.010816653826391976,0.9993333333333334,0.0011547005383792366,,
hgb_reliamol_novinascore,0.1,posebusters_gate_variable_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9983333333333334,0.0005773502691895862,,
hgb_reliamol_novinascore,0.1,posebusters_gate_fill_k,1000.0,0.0,10.0,0.0,1.0,0.0,0.9983333333333334,0.0005773502691895862,,
