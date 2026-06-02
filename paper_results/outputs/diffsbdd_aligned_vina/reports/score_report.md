# DiffSBDD Aligned Vina Score-Only Check

Only pockets marked usable_public_pdb_alignment in the audit are included.

## Overall

n_candidates,n_pockets,ligand_prep_success_rate,vina_success_rate,reliable_proxy_rate,median_vina_affinity,median_vina_affinity_reliable,median_vina_affinity_unreliable
3900,39,0.9902564102564102,0.9902564102564102,0.94,-3.9215,-4.087,17.311

## Vina Reranking

score,topk_frac,n_selected,reliable_proxy_rate,chemical_pass_rate,geometry_pass_rate,pocket_pass_rate,clash_free_rate,median_vina_affinity
vina_score_only,0.1,390,0.9923076923076923,1.0,0.9974358974358974,1.0,1.0,-6.404
vina_score_only,0.2,778,0.9884318766066839,1.0,0.9922879177377892,0.9987146529562982,1.0,-5.907500000000001
vina_score_only,0.5,1938,0.9840041279669762,1.0,0.9907120743034056,0.9974200206398349,0.9974200206398349,-5.121
