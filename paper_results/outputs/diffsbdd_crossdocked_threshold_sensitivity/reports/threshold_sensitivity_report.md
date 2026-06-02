# DiffSBDD CrossDocked Threshold Sensitivity

Labels are recomputed from exact CrossDocked gninatypes receptor coordinates while sweeping ligand atom mode, contact, clash, and internal-close thresholds.

## Label Range

ligand_atom_mode,min,median,max
all,0.8978,0.9304,0.9479
heavy,0.8978,0.9304,0.9479

## Top-10% Reliable Range

method,ligand_atom_mode,min,median,max
qed_sa,all,0.953,0.966,0.974
mlp_pose_shape,all,0.9936666666666666,0.9986666666666668,0.9993333333333334
mlp_reliamol_novinascore,all,0.9943333333333334,0.998,0.9983333333333334
rf_reliamol_novinascore,all,0.9983333333333334,0.999,0.999
hgb_reliamol_novinascore,all,0.998,0.9993333333333334,0.9993333333333334
qed_sa,heavy,0.953,0.966,0.974
mlp_pose_shape,heavy,0.9936666666666666,0.9986666666666668,0.9993333333333334
mlp_reliamol_novinascore,heavy,0.9943333333333334,0.998,0.9983333333333334
rf_reliamol_novinascore,heavy,0.9983333333333334,0.999,0.999
hgb_reliamol_novinascore,heavy,0.998,0.9993333333333334,0.9993333333333334

## Strictest Heavy-Atom Label Head

method,ligand_atom_mode,contact_threshold,clash_threshold,internal_threshold,topk_frac,reliable_rate_mean,reliable_rate_std,chemical_pass_rate_mean,chemical_pass_rate_std,geometry_pass_rate_mean,geometry_pass_rate_std,pocket_pass_rate_mean,pocket_pass_rate_std,clash_free_rate_mean,clash_free_rate_std,synthetic_pass_rate_mean,synthetic_pass_rate_std
rf_reliamol_novinascore,heavy,4.0,1.5,0.65,0.1,0.9983333333333334,0.0005773502691896,1.0,0.0,1.0,0.0,1.0,0.0,0.9983333333333334,0.0005773502691896,1.0,0.0
hgb_reliamol_novinascore,heavy,4.0,1.5,0.65,0.1,0.998,0.0017320508075688,0.9996666666666668,0.0005773502691896,1.0,0.0,1.0,0.0,0.9983333333333334,0.0011547005383792,1.0,0.0
mlp_reliamol_novinascore,heavy,4.0,1.5,0.65,0.1,0.9943333333333334,0.0037859388972001,0.999,0.001,1.0,0.0,1.0,0.0,0.9953333333333332,0.0028867513459481,1.0,0.0
mlp_pose_shape,heavy,4.0,1.5,0.65,0.1,0.9936666666666666,0.0023094010767584,0.9996666666666668,0.0005773502691895,1.0,0.0,1.0,0.0,0.994,0.0017320508075688,1.0,0.0
mlp_ligand_descriptor,heavy,4.0,1.5,0.65,0.1,0.9533333333333333,0.0015275252316519,0.9996666666666668,0.0005773502691895,0.9953333333333332,0.0011547005383792,1.0,0.0,0.9576666666666666,0.0015275252316519,1.0,0.0
qed_sa,heavy,4.0,1.5,0.65,0.1,0.953,0.0,0.999,0.0,0.992,0.0,1.0,0.0,0.96,0.0,1.0,0.0
qed,heavy,4.0,1.5,0.65,0.1,0.934,0.0,1.0,0.0,0.992,0.0,1.0,0.0,0.941,0.0,1.0,0.0
