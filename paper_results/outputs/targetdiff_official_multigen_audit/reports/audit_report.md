# TargetDiff Official Sampling-Result Audit

The TargetDiff/CVAE/AR/Pocket2Mol public sampling-result metadata files are audited against the 100 CrossDocked pocket10 PDBs.

## Generator Summary

generator,n_candidates,n_pockets,pocket_pdb_found_rate,sanitize_rate,has_3d_rate,reliable_proxy_rate,chemical_failure_proxy_rate,geometric_failure_proxy_rate,pocket_failure_proxy_rate,clash_failure_proxy_rate,synthetic_failure_proxy_rate,median_qed,median_sa_proxy,median_vina_minimize_affinity,median_min_dist_to_receptor,median_receptor_contacts_4p5
CrossDockedNative,100,100,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.46756970485171556,3.8962317595668168,-6.4864999999999995,2.813472867012024,21.0
CVAE,9911,100,1.0,1.0,1.0,0.79053576833821,0.022096660276460497,0.1825244677630915,0.0,0.0,0.01967510846534154,0.38858514480792733,3.9711720464729257,-6.578,2.850167751312256,18.0
AR,9295,97,1.0,1.0,1.0,0.9825712748789672,0.013232920925228618,0.0012910166756320601,0.00021516944593867672,0.00021516944593867672,0.0029047875201721357,0.49966753501949723,3.4358117070171645,-5.879,2.817488193511963,15.0
Pocket2Mol,9831,100,1.0,1.0,1.0,0.9860644898789543,0.013630352965110365,0.00010171905197843556,0.0,0.0003051571559353067,0.0,0.5770452563785492,3.078500789049978,-5.822,2.7148425579071045,15.0
TargetDiff,9036,100,1.0,1.0,1.0,0.9924745462594068,0.00199203187250996,0.0002213368747233289,0.0,0.002213368747233289,0.0033200531208499337,0.4808451322371523,3.991030808349075,-6.831,2.7198338508605957,23.0

## Pocket-Generator Summary Head

generator,pocket_id,n_candidates,reliable_proxy_rate,clash_failure_proxy_rate,geometric_failure_proxy_rate,median_vina_minimize_affinity
CrossDockedNative,BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3,1,1.0,0.0,0.0,-8.818
CrossDockedNative,GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0,1,1.0,0.0,0.0,-7.931
CrossDockedNative,GRK4_HUMAN_1_578_0/4yhj_A_rec_4yhj_an2_lig_tt_min_0,1,1.0,0.0,0.0,-6.421
CrossDockedNative,GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0,1,1.0,0.0,0.0,-5.965
CrossDockedNative,GUX1_HYPJE_18_451_0/2v3r_A_rec_1dy4_snp_lig_tt_docked_1,1,1.0,0.0,0.0,-8.65
CrossDockedNative,HDAC8_HUMAN_1_377_0/4rn0_B_rec_4rn1_l8g_lig_tt_min_0,1,1.0,0.0,0.0,-4.914
CrossDockedNative,HDHA_ECOLI_1_255_0/1fmc_B_rec_1fmc_cho_lig_tt_docked_1,1,1.0,0.0,0.0,-9.734
CrossDockedNative,HMD_METJA_1_358_0/3daf_A_rec_3daf_feg_lig_tt_docked_0,1,1.0,0.0,0.0,-9.12
CrossDockedNative,CCPR_YEAST_69_361_0/1a2g_A_rec_4jmv_1ly_lig_tt_min_0,1,1.0,0.0,0.0,-4.388
CrossDockedNative,IPMK_HUMAN_49_416_0/5w2g_A_rec_5w2i_adp_lig_tt_min_0,1,1.0,0.0,0.0,-4.95
CrossDockedNative,CD38_HUMAN_44_300_0/3dzh_A_rec_3u4i_cvr_lig_tt_docked_0,1,1.0,0.0,0.0,-9.057
CrossDockedNative,KS6A3_HUMAN_41_357_0/3g51_A_rec_3g51_anp_lig_tt_min_0,1,1.0,0.0,0.0,-8.791
CrossDockedNative,CHOD_BREST_46_552_0/1coy_A_rec_1coy_and_lig_tt_docked_0,1,1.0,0.0,0.0,-9.478
CrossDockedNative,LAT_MYCTU_1_449_0/2jjg_A_rec_2jjg_plp_lig_tt_min_0,1,1.0,0.0,0.0,-5.793
CrossDockedNative,LMBL1_HUMAN_198_526_0/2rhy_A_rec_2rhy_mlz_lig_tt_min_0,1,1.0,0.0,0.0,-2.816
CrossDockedNative,LMBL1_HUMAN_198_526_0/2pqw_A_rec_2rhy_mlz_lig_tt_min_0,1,1.0,0.0,0.0,-4.023
CrossDockedNative,M3K14_HUMAN_321_678_0/4g3d_B_rec_4idv_13v_lig_tt_min_0,1,1.0,0.0,0.0,-5.062
CrossDockedNative,MENE_BACSU_2_486_0/5bur_A_rec_5x8f_amp_lig_tt_docked_7,1,1.0,0.0,0.0,-7.03
CrossDockedNative,NAGZ_VIBCH_1_330_0/3gs6_A_rec_2oxn_oan_lig_tt_docked_4,1,1.0,0.0,0.0,-5.782
CrossDockedNative,NEP_HUMAN_54_750_0/1r1h_A_rec_1r1h_bir_lig_tt_docked_1,1,1.0,0.0,0.0,-9.505
CrossDockedNative,NQO1_HUMAN_2_274_0/1dxo_C_rec_1gg5_e09_lig_tt_min_0,1,1.0,0.0,0.0,-3.141
CrossDockedNative,NQO1_HUMAN_2_274_0/1gg5_A_rec_1kbo_340_lig_tt_min_0,1,1.0,0.0,0.0,-5.527
CrossDockedNative,NR1H4_HUMAN_258_486_0/5q0k_A_rec_5q0q_9ld_lig_tt_docked_0,1,1.0,0.0,0.0,-9.716
CrossDockedNative,OLIAC_CANSA_1_101_0/5b08_A_rec_5b09_4mx_lig_tt_min_0,1,1.0,0.0,0.0,-5.851
CrossDockedNative,PA21B_PIG_23_146_0/2azy_A_rec_2azy_chd_lig_tt_docked_0,1,1.0,0.0,0.0,-9.879
CrossDockedNative,PAK4_HUMAN_291_591_ATP_0/5i0b_A_rec_5vef_m77_lig_tt_min_0,1,1.0,0.0,0.0,-7.694
CrossDockedNative,PHKG1_RABIT_6_296_ATPsite_0/1phk_A_rec_1phk_atp_lig_tt_min_0,1,1.0,0.0,0.0,-8.36
CrossDockedNative,PHP_SULSO_1_314_0/4keu_A_rec_4ket_pg4_lig_tt_min_0,1,1.0,0.0,0.0,-3.864
CrossDockedNative,COTA_BACSU_1_513_0/4q8b_B_rec_4q8b_sxx_lig_tt_min_0,1,1.0,0.0,0.0,-4.284
CrossDockedNative,PLCD1_RAT_134_756_0/1djy_A_rec_1djz_ip2_lig_tt_min_0,1,1.0,0.0,0.0,-5.377

## Failure Types

Counter({'CVAE': 9911, 'Pocket2Mol': 9831, 'AR': 9295, 'TargetDiff': 9036, 'CrossDockedNative': 100})