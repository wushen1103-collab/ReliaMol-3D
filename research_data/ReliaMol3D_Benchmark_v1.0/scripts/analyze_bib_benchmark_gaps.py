#!/usr/bin/env python3
"""Close benchmark-analysis gaps for the BIB manuscript.

The script uses completed candidate-level endpoints and out-of-fold results. It
does not rerun molecular preparation or OpenMM relaxation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", Path.cwd()))
DIFF_FRAME = ROOT / "outputs/bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv"
DIFF_OOF = ROOT / "outputs/bib_physical_endpoint_analysis_v0/physical_oof_predictions.csv"
DIFF_POCKET = ROOT / "outputs/bib_reviewer_gap_closure_v2/table_D_diffsbdd_full_audit_direct_per_pocket.csv"
MULTI_POCKET = ROOT / "outputs/bib_reviewer_gap_closure_v2/table_E_multigen_processable_decomposition_per_pocket.csv"
MULTI_FRAME = ROOT / "outputs/bib_multigen_physical_transfer_v2/multigen_openmm_endpoint_frame_final.csv"
OUT = ROOT / "outputs/bib_benchmark_gap_closure_v0"
TARGET = "stable_r2p0_c0p5_with_pb"
SEEDS = (11, 22, 33)


ATOM_FEATURES = [
    "atom_metal_frac", "atom_C_frac", "atom_N_frac", "atom_O_frac", "atom_S_frac",
    "atom_P_frac", "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
]
CHEMISTRY = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM_FEATURES
INTERNAL = [
    "pose_atoms", "pose_radius_shape", "nn_q05", "nn_q10", "nn_q25", "nn_q50",
    "nn_q75", "nn_q90",
]
POCKET = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90", "shell_0_2",
    "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
DIRECT_RULE_VARIABLES = {
    "rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms",
    "sa_proxy", "descriptor_ok", "pose_radius_shape",
}


def direct_excluded(columns: list[str]) -> list[str]:
    return [column for column in columns if column not in DIRECT_RULE_VARIABLES]


CHEMISTRY_DE = direct_excluded(CHEMISTRY)
INTERNAL_DE = direct_excluded(INTERNAL)
FEATURE_GROUPS = {
    "chemistry_only": CHEMISTRY_DE,
    "internal_geometry_only": INTERNAL_DE,
    "pocket_context_only": POCKET,
    "chemistry_internal": CHEMISTRY_DE + INTERNAL_DE,
    "chemistry_pocket": CHEMISTRY_DE + POCKET,
    "internal_pocket": INTERNAL_DE + POCKET,
    "all_direct_excluded": CHEMISTRY_DE + INTERNAL_DE + POCKET,
}


def fold_pockets(pockets: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(pockets).copy()
    rng.shuffle(shuffled)
    return [part for part in np.array_split(shuffled, folds) if len(part)]


def topk_pocket_rows(frame: pd.DataFrame, score: str, method: str, seed: int) -> list[dict]:
    rows = []
    for pocket_id, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(math.ceil(0.10 * len(group))))
        selected = group.nlargest(k, score)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "pocket_id": pocket_id,
                "n_selected": len(selected),
                "endpoint_positive_rate": float(selected[TARGET].mean()),
            }
        )
    return rows


def percentile_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot = np.empty(draws, dtype=float)
    for index in range(draws):
        boot[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def correlation_bootstrap(frame: pd.DataFrame, x: str, y: str, draws: int, seed: int) -> dict:
    clean = frame[[x, y]].dropna().reset_index(drop=True)
    observed = float(clean[x].corr(clean[y], method="spearman"))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        sample = clean.iloc[rng.integers(0, len(clean), len(clean))]
        value = sample[x].corr(sample[y], method="spearman")
        if np.isfinite(value):
            boot.append(float(value))
    return {
        "n_groups": int(len(clean)),
        "spearman_rho": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def paired_summary(per_pocket: pd.DataFrame, method: str, reference: str, seed: int) -> dict:
    wide = per_pocket.pivot(index="pocket_id", columns="method", values="endpoint_positive_rate")
    pair = wide[[method, reference]].dropna()
    delta = (pair[method] - pair[reference]).to_numpy(float)
    low, high = percentile_bootstrap(delta, 5000, seed)
    return {
        "method": method,
        "reference": reference,
        "n_pockets": int(len(delta)),
        "method_mean": float(pair[method].mean()),
        "reference_mean": float(pair[reference].mean()),
        "mean_delta": float(delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def feature_family_ablation(diff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    pocket_rows = []
    diff = diff.copy()
    # Match the finalized manuscript baseline and its archived OOF score.
    diff["score_qed_sa"] = diff["qed"].astype(float) - 0.25 * diff["sa_proxy"].astype(float)
    pockets = diff["pocket_id"].drop_duplicates().to_numpy()

    for seed in SEEDS:
        seed_pred = diff[["pocket_id", TARGET, "score_qed_sa"]].copy()
        for method in FEATURE_GROUPS:
            seed_pred[f"score_{method}"] = np.nan
        for heldout in fold_pockets(pockets, 5, seed):
            test_mask = diff["pocket_id"].isin(set(heldout))
            train = diff.loc[~test_mask]
            test = diff.loc[test_mask]
            y_train = train[TARGET].astype(int).to_numpy()
            for method, columns in FEATURE_GROUPS.items():
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
                model.fit(train[columns].replace([np.inf, -np.inf], np.nan), y_train)
                seed_pred.loc[test_mask, f"score_{method}"] = model.predict_proba(
                    test[columns].replace([np.inf, -np.inf], np.nan)
                )[:, 1]

        pocket_rows.extend(topk_pocket_rows(seed_pred, "score_qed_sa", "qed_sa", seed))
        for method in FEATURE_GROUPS:
            pocket_rows.extend(topk_pocket_rows(seed_pred, f"score_{method}", method, seed))
            valid = seed_pred[f"score_{method}"].notna()
            predictions.append(
                {
                    "seed": seed,
                    "method": method,
                    "roc_auc": float(
                        roc_auc_score(seed_pred.loc[valid, TARGET], seed_pred.loc[valid, f"score_{method}"])
                    ),
                    "brier": float(
                        brier_score_loss(seed_pred.loc[valid, TARGET], seed_pred.loc[valid, f"score_{method}"])
                    ),
                }
            )

    pockets_long = pd.DataFrame(pocket_rows)
    pockets_mean = (
        pockets_long.groupby(["method", "pocket_id"], as_index=False)["endpoint_positive_rate"].mean()
    )
    metrics = pd.DataFrame(predictions)
    metric_summary = metrics.groupby("method").agg(
        roc_auc=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        brier=("brier", "mean"),
        brier_sd=("brier", "std"),
    ).reset_index()

    rows = []
    qed_mean = float(pockets_mean.loc[pockets_mean["method"].eq("qed_sa"), "endpoint_positive_rate"].mean())
    for index, method in enumerate(FEATURE_GROUPS):
        paired = paired_summary(pockets_mean, method, "qed_sa", 20260815 + index)
        seed_rates = (
            pockets_long[pockets_long["method"].eq(method)]
            .groupby("seed")["endpoint_positive_rate"].mean()
        )
        paired.update(
            {
                "n_features": len(FEATURE_GROUPS[method]),
                "endpoint_positive_sd_across_seeds": float(seed_rates.std()),
                "qed_sa_mean": qed_mean,
            }
        )
        rows.append(paired)
    table = pd.DataFrame(rows).merge(metric_summary, on="method", how="left")
    return table, pockets_mean, metrics


def diff_difficulty_analysis(diff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = diff.groupby("pocket_id", as_index=False)[TARGET].mean().rename(columns={TARGET: "full_pool_endpoint_rate"})
    raw["full_pool_failure_burden"] = 1.0 - raw["full_pool_endpoint_rate"]
    selected = pd.read_csv(DIFF_POCKET)
    keep_methods = ["qed_sa", "full_audit_gate_qedsa_fixed_budget", "phys_logreg_full", "phys_rf_direct_excluded"]
    selected = selected[selected["method"].isin(keep_methods)]
    mean_selected = selected.groupby(["method", "pocket_id"], as_index=False)["endpoint_positive_rate"].mean()
    wide = mean_selected.pivot(index="pocket_id", columns="method", values="endpoint_positive_rate").reset_index()
    merged = raw.merge(wide, on="pocket_id", how="inner")
    merged["rf_gain_vs_qed_sa"] = merged["phys_rf_direct_excluded"] - merged["qed_sa"]
    merged["logistic_gain_vs_qed_sa"] = merged["phys_logreg_full"] - merged["qed_sa"]
    merged["difficulty_quartile"] = pd.qcut(
        merged["full_pool_failure_burden"], 4, labels=["Q1 easiest", "Q2", "Q3", "Q4 hardest"]
    )
    strata = merged.groupby("difficulty_quartile", observed=True).agg(
        n_pockets=("pocket_id", "size"),
        full_pool_endpoint_rate=("full_pool_endpoint_rate", "mean"),
        qed_sa=("qed_sa", "mean"),
        five_rule=("full_audit_gate_qedsa_fixed_budget", "mean"),
        logistic_endpoint=("phys_logreg_full", "mean"),
        rf_direct_excluded=("phys_rf_direct_excluded", "mean"),
        rf_gain_vs_qed_sa=("rf_gain_vs_qed_sa", "mean"),
    ).reset_index()
    correlation = correlation_bootstrap(merged, "full_pool_failure_burden", "rf_gain_vs_qed_sa", 5000, 20260815)
    correlation["analysis"] = "DiffSBDD pocket failure burden versus RF gain"
    correlation["negative_gain_pockets"] = int((merged["rf_gain_vs_qed_sa"] < -1e-12).sum())
    return merged, strata, correlation


def generator_transfer_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    selected = pd.read_csv(MULTI_POCKET)
    selected = selected[selected["method"].isin(["qed_sa", "full_audit_gate_qedsa_fixed_budget", "logreg_endpoint"])]
    selected_mean = selected.groupby(["generator", "method", "pocket_id"], as_index=False)["endpoint_positive_rate"].mean()
    rows = []
    for generator in ["AR", "CVAE", "TargetDiff", "Pocket2Mol"]:
        sub = selected_mean[selected_mean["generator"].eq(generator)].drop(columns="generator")
        row = paired_summary(sub, "logreg_endpoint", "qed_sa", 20260820 + len(rows))
        row["generator"] = generator
        rows.append(row)
    pooled = selected_mean.copy()
    pooled["pocket_id"] = pooled["generator"].astype(str) + "||" + pooled["pocket_id"].astype(str)
    pooled = pooled.drop(columns="generator")
    pooled_row = paired_summary(pooled, "logreg_endpoint", "qed_sa", 20260830)
    pooled_row["generator"] = "All target generators"
    rows.append(pooled_row)
    transfer = pd.DataFrame(rows)

    raw = pd.read_csv(MULTI_FRAME, usecols=["generator", "pocket_id", TARGET])
    raw = raw.groupby(["generator", "pocket_id"], as_index=False)[TARGET].mean().rename(
        columns={TARGET: "full_pool_endpoint_rate"}
    )
    raw["full_pool_failure_burden"] = 1.0 - raw["full_pool_endpoint_rate"]
    wide = selected_mean.pivot(index=["generator", "pocket_id"], columns="method", values="endpoint_positive_rate").reset_index()
    merged = raw.merge(wide, on=["generator", "pocket_id"], how="inner")
    merged["logistic_gain_vs_qed_sa"] = merged["logreg_endpoint"] - merged["qed_sa"]
    merged["difficulty_quartile"] = pd.qcut(
        merged["full_pool_failure_burden"], 4, labels=["Q1 easiest", "Q2", "Q3", "Q4 hardest"], duplicates="drop"
    )
    strata = merged.groupby("difficulty_quartile", observed=True).agg(
        n_groups=("pocket_id", "size"),
        full_pool_endpoint_rate=("full_pool_endpoint_rate", "mean"),
        qed_sa=("qed_sa", "mean"),
        logistic_endpoint=("logreg_endpoint", "mean"),
        gain=("logistic_gain_vs_qed_sa", "mean"),
    ).reset_index()
    correlation = correlation_bootstrap(merged, "full_pool_failure_burden", "logistic_gain_vs_qed_sa", 5000, 20260831)
    correlation["analysis"] = "Target-generator pocket failure burden versus logistic gain"
    correlation["negative_gain_groups"] = int((merged["logistic_gain_vs_qed_sa"] < -1e-12).sum())
    return transfer, merged, strata, correlation


def endpoint_component_summary(diff: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_csv(DIFF_OOF)
    selected_parts = []
    for seed, scores in oof.groupby("seed", sort=True):
        frame = diff.merge(
            scores[["candidate_uid", "score_phys_logreg_full", "score_phys_rf_direct_excluded"]],
            on="candidate_uid",
            how="inner",
        )
        frame["score_qed_sa"] = frame["qed"].astype(float) - 0.25 * frame["sa_proxy"].astype(float)
        frame["score_five_rule"] = 100.0 * frame["reliable"].astype(float) + frame["score_qed_sa"]
        score_map = {
            "qed_sa": "score_qed_sa",
            "full_audit_gate_qedsa_fixed_budget": "score_five_rule",
            "phys_logreg_full": "score_phys_logreg_full",
            "phys_rf_direct_excluded": "score_phys_rf_direct_excluded",
        }
        for method, score_col in score_map.items():
            for _, group in frame.groupby("pocket_id", sort=False):
                k = max(1, int(math.ceil(0.10 * len(group))))
                selected = group.nlargest(k, score_col).copy()
                selected["method"] = method
                selected["seed"] = seed
                selected_parts.append(selected)
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.groupby("method", as_index=False).agg(
        endpoint_positive=(TARGET, "mean"),
        relaxation_success=("relaxation_success", "mean"),
        final_posebusters_pass=("final_pb_pass", "mean"),
        rmsd_le_2p0=("rmsd_le_2p0", "mean"),
        contact_retention_ge_0p5=("retention_ge_0p5", "mean"),
        final_clash_free=("final_clash_free", "mean"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diff = pd.read_csv(DIFF_FRAME)

    ablation, ablation_pockets, ablation_metrics = feature_family_ablation(diff)
    ablation.to_csv(OUT / "table_I_feature_family_ablation.csv", index=False)
    ablation_pockets.to_csv(OUT / "table_I_feature_family_ablation_per_pocket.csv", index=False)
    ablation_metrics.to_csv(OUT / "table_I_feature_family_candidate_metrics_by_seed.csv", index=False)

    diff_pockets, diff_strata, diff_corr = diff_difficulty_analysis(diff)
    diff_pockets.to_csv(OUT / "table_J_diffsbdd_pocket_difficulty.csv", index=False)
    diff_strata.to_csv(OUT / "table_J_diffsbdd_difficulty_strata.csv", index=False)

    transfer, transfer_pockets, transfer_strata, transfer_corr = generator_transfer_analysis()
    transfer.to_csv(OUT / "table_K_generator_transfer_bootstrap.csv", index=False)
    transfer_pockets.to_csv(OUT / "table_K_generator_pocket_difficulty.csv", index=False)
    transfer_strata.to_csv(OUT / "table_K_generator_difficulty_strata.csv", index=False)

    components = endpoint_component_summary(diff)
    components.to_csv(OUT / "table_L_selected_endpoint_components.csv", index=False)

    summary = {
        "feature_groups": {name: len(columns) for name, columns in FEATURE_GROUPS.items()},
        "diffsbdd_difficulty_correlation": diff_corr,
        "generator_difficulty_correlation": transfer_corr,
    }
    (OUT / "bib_benchmark_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFeature-family ablation:\n", ablation.to_string(index=False))
    print("\nGenerator transfer bootstrap:\n", transfer.to_string(index=False))
    print("\nDiffSBDD difficulty strata:\n", diff_strata.to_string(index=False))
    print("\nGenerator difficulty strata:\n", transfer_strata.to_string(index=False))
    print("\nSelected endpoint components:\n", components.to_string(index=False))


if __name__ == "__main__":
    main()
