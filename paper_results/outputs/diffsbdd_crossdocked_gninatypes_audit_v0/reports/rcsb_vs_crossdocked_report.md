# DiffSBDD CrossDocked Gninatypes Comparison

This report compares the earlier public RCSB receptor approximation with CrossDocked2020 receptor gninatypes coordinates. Receptor hydrogen types 0/1 are excluded by default.

## Overall

source,n_candidates,n_pockets,reliable_proxy_rate,pocket_failure_proxy_rate,clash_failure_proxy_rate,median_min_dist_to_receptor,median_contacts_4p5
RCSB public approximation,10000,100,0.3816,0.5358,0.0761,6.9927074909210205,0.0
CrossDocked gninatypes heavy receptor,10000,100,0.9304,0.0,0.0502,2.6063321828842163,23.0

## By Previous RCSB Alignment Status

alignment_status,n_pockets,rcsb_reliable,crossdocked_reliable,delta_reliable,rcsb_pocket_failure,crossdocked_pocket_failure,rcsb_clash_failure,crossdocked_clash_failure,crossdocked_median_min_dist,crossdocked_median_contacts
ambiguous,10,0.0569999999999999,0.92,0.8629999999999999,0.941,0.0,0.001,0.054,2.664965569972992,18.5
coordinate_mismatch,44,0.0,0.9340909090909092,0.9340909090909092,1.0,0.0,0.0,0.0454545454545454,2.599618911743164,23.75
high_clash_alignment,7,0.1328571428571428,0.9457142857142856,0.8128571428571428,0.0028571428571428,0.0,0.8642857142857142,0.0328571428571428,2.6974635124206543,27.5
usable_public_pdb_alignment,39,0.94,0.926153846153846,-0.0138461538461538,0.0038461538461538,0.0,0.0397435897435897,0.0576923076923076,2.6277852058410645,24.0

## Lowest CrossDocked Reliable Pockets

pocket_id,alignment_status,reliable_proxy_rate_rcsb,reliable_proxy_rate_crossdocked,clash_failure_proxy_rate_crossdocked,median_min_dist_to_receptor_crossdocked,median_receptor_contacts_4p5_crossdocked
1l3l-A-rec-1l3l-lae-lig-tt-min-0-pocket10_1l3l-A-rec-1l3l-lae-lig-tt-min-0,usable_public_pdb_alignment,0.78,0.51,0.44,1.3435956239700315,32.0
2rhy-A-rec-2rhy-mlz-lig-tt-min-0-pocket10_2rhy-A-rec-2rhy-mlz-lig-tt-min-0,coordinate_mismatch,0.0,0.59,0.31,1.610144555568695,18.0
5mma-A-rec-4ztf-x2p-lig-tt-min-0-pocket10_5mma-A-rec-4ztf-x2p-lig-tt-min-0,usable_public_pdb_alignment,0.82,0.61,0.39,1.4923861026763916,15.0
4p6p-A-rec-4p77-5rp-lig-tt-docked-0-pocket10_4p6p-A-rec-4p77-5rp-lig-tt-docked-0,usable_public_pdb_alignment,0.95,0.66,0.32,1.3086338639259338,22.0
1k9t-A-rec-2wlz-dio-lig-tt-min-0-pocket10_1k9t-A-rec-2wlz-dio-lig-tt-min-0,usable_public_pdb_alignment,0.92,0.69,0.29,1.89299476146698,16.5
2gns-A-rec-4qer-stl-lig-tt-min-0-pocket10_2gns-A-rec-4qer-stl-lig-tt-min-0,ambiguous,0.0,0.7,0.24,2.394042730331421,10.0
4tqr-A-rec-2xca-doc-lig-tt-min-0-pocket10_4tqr-A-rec-2xca-doc-lig-tt-min-0,coordinate_mismatch,0.0,0.72,0.28,1.803299844264984,17.0
5ngz-A-rec-5ngz-2bg-lig-tt-min-0-pocket10_5ngz-A-rec-5ngz-2bg-lig-tt-min-0,usable_public_pdb_alignment,0.79,0.79,0.19,2.601921558380127,21.0
2pc8-A-rec-1eqc-cts-lig-tt-docked-6-pocket10_2pc8-A-rec-1eqc-cts-lig-tt-docked-6,coordinate_mismatch,0.0,0.79,0.11,1.9840965270996087,23.5
5bur-A-rec-5x8f-amp-lig-tt-docked-7-pocket10_5bur-A-rec-5x8f-amp-lig-tt-docked-7,coordinate_mismatch,0.0,0.8,0.2,2.3300546407699585,26.0

## Key Findings

1. Exact CrossDocked receptor coordinates remove the all-pocket coordinate mismatch: pocket failure falls from 0.5358 to 0.0000.
2. Overall reliable proxy rises from 0.3816 under public RCSB approximation to 0.9304 with CrossDocked heavy receptor coordinates.
3. The former coordinate_mismatch stratum rises from 0.0000 to 0.9341 reliable proxy, confirming that the previous failure was receptor provenance rather than generated molecule placement.
4. Residual failures are mostly clash/geometric outliers, not chemistry: sanitize rate remains 1.0000 and clash failure is 0.0502 after excluding receptor hydrogens.