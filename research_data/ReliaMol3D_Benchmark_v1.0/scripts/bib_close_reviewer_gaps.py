#!/usr/bin/env python3
"""Close BIB-transfer experiment gaps raised after the JCTC rejection.

This script does not rerun OpenMM. It reuses completed candidate-level
post-relaxation endpoints and produces tables for:
1. direct use of the full five-rule audit criteria;
2. full-pipeline vs processable-subset endpoint decomposition;
3. source-generator zero-shot pocket-disjointness checks;
4. leave-one-generator plus pocket-holdout evaluation on OpenMM endpoints.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", Path.cwd()))
TARGET = "stable_r2p0_c0p5_with_pb"
FAILURE_FLAGS = ["chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure"]
ATOM_FEATURES = [
    "atom_metal_frac", "atom_C_frac", "atom_N_frac", "atom_O_frac", "atom_S_frac",
    "atom_P_frac", "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
]
CHEM_FEATURES = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM_FEATURES
INTERNAL_FEATURES = [
    "pose_atoms", "pose_radius_shape", "nn_q05", "nn_q10", "nn_q25", "nn_q50", "nn_q75", "nn_q90",
]
POCKET_FEATURES = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90",
    "shell_0_2", "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
FULL_FEATURES = CHEM_FEATURES + INTERNAL_FEATURES + POCKET_FEATURES


def fold_pockets(pockets: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pockets = np.asarray(sorted(map(str, pockets))).copy()
    rng.shuffle(pockets)
    return [part for part in np.array_split(pockets, min(folds, len(pockets))) if len(part)]


def ensure_bool(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    return out


def add_common_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "score_qed_sa" not in out:
        out["score_qed_sa"] = out["qed"].astype(float) - 0.05 * out["sa_proxy"].astype(float)
    if "score_contact_clash_rule" not in out:
        contact = out["receptor_contact_atoms_4p5"].fillna(0).astype(float)
        clash = out["receptor_clash_atoms_1p2"].fillna(0).astype(float)
        severe = out["receptor_severe_clash_atoms_0p8"].fillna(0).astype(float)
        out["score_contact_clash_rule"] = contact - 8.0 * clash - 16.0 * severe
    out = ensure_bool(out, FAILURE_FLAGS + ["reliable", "relaxation_success", "final_pb_pass", "final_clash_free", TARGET])
    out["audit_failure_count"] = out[FAILURE_FLAGS].astype(int).sum(axis=1)
    out["score_audit_failure_count_qedsa"] = -100.0 * out["audit_failure_count"].astype(float) + out["score_qed_sa"].astype(float)
    out["score_full_audit_gate_qedsa"] = 100.0 * out["reliable"].astype(float) + out["score_qed_sa"].astype(float)
    if "smiles" in out.columns:
        out["radical_precheck_ok"] = out["smiles"].map(radical_free_from_smiles).fillna(False).astype(bool)
        out["score_radical_precheck_qedsa"] = 100.0 * out["radical_precheck_ok"].astype(float) + out["score_qed_sa"].astype(float)
    if {"atom_S_frac", "atom_P_frac"}.issubset(out.columns):
        out["no_sp_element_precheck_ok"] = out["atom_S_frac"].fillna(0).eq(0) & out["atom_P_frac"].fillna(0).eq(0)
        out["score_no_sp_element_qedsa"] = 100.0 * out["no_sp_element_precheck_ok"].astype(float) + out["score_qed_sa"].astype(float)
    if "relaxation_success" in out.columns:
        out["score_processability_oracle_qedsa"] = 100.0 * out["relaxation_success"].astype(float) + out["score_qed_sa"].astype(float)
    return out


def radical_free_from_smiles(smiles: object) -> bool:
    if not isinstance(smiles, str) or not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        return sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) == 0
    except Exception:
        return False


def select_top(group: pd.DataFrame, method: str, coverage: float) -> pd.DataFrame:
    k = max(1, int(math.ceil(coverage * len(group))))
    if method == "full_audit_gate_qedsa_no_backfill":
        eligible = group[group["reliable"].astype(bool)]
        if eligible.empty:
            return eligible
        return eligible.nlargest(min(k, len(eligible)), "score_qed_sa")
    score = {
        "qed_sa": "score_qed_sa",
        "contact_clash_rule": "score_contact_clash_rule",
        "full_audit_gate_qedsa_fixed_budget": "score_full_audit_gate_qedsa",
        "audit_failure_count_qedsa": "score_audit_failure_count_qedsa",
        "radical_precheck_qedsa": "score_radical_precheck_qedsa",
        "no_sp_element_qedsa": "score_no_sp_element_qedsa",
        "processability_oracle_qedsa": "score_processability_oracle_qedsa",
        "logreg_endpoint": "score_logreg_full",
        "hgb_endpoint": "score_hgb_full",
        "phys_logreg_full": "score_phys_logreg_full",
        "phys_hgb_full": "score_phys_hgb_full",
        "phys_rf_direct_excluded": "score_phys_rf_direct_excluded",
        "phys_rf_full": "score_phys_rf_full",
    }[method]
    return group.nlargest(k, score)


def summarize_selection(selected: pd.DataFrame, nominal_k: int) -> dict[str, float | int]:
    n = len(selected)
    if n == 0:
        return {
            "n_selected": 0,
            "nominal_k": nominal_k,
            "effective_coverage_fraction": 0.0,
            "endpoint_positive_rate": np.nan,
            "failures_per_1000_selected": np.nan,
            "failures_per_1000_nominal": 1000.0,
            "relaxation_success_rate": np.nan,
            "conditional_endpoint_rate_success": np.nan,
            "conditional_pb_pass_success": np.nan,
            "conditional_rmsd_le_2p0_success": np.nan,
            "conditional_retention_ge_0p5_success": np.nan,
            "conditional_clash_free_success": np.nan,
        }
    success = selected[selected["relaxation_success"].astype(bool)]
    endpoint = selected[TARGET].astype(float)
    failures = float(n - endpoint.sum())
    out = {
        "n_selected": int(n),
        "nominal_k": int(nominal_k),
        "effective_coverage_fraction": float(n / nominal_k) if nominal_k else np.nan,
        "endpoint_positive_rate": float(endpoint.mean()),
        "failures_per_1000_selected": float((1.0 - endpoint.mean()) * 1000.0),
        "failures_per_1000_nominal": float(failures / nominal_k * 1000.0) if nominal_k else np.nan,
        "relaxation_success_rate": float(selected["relaxation_success"].mean()),
    }
    if len(success):
        out.update(
            {
                "conditional_endpoint_rate_success": float(success[TARGET].mean()),
                "conditional_pb_pass_success": float(success["final_pb_pass"].mean()),
                "conditional_rmsd_le_2p0_success": float(success["rmsd_le_2p0"].mean()) if "rmsd_le_2p0" in success else np.nan,
                "conditional_retention_ge_0p5_success": float(success["retention_ge_0p5"].mean()) if "retention_ge_0p5" in success else np.nan,
                "conditional_clash_free_success": float(success["final_clash_free"].mean()),
            }
        )
    else:
        out.update(
            {
                "conditional_endpoint_rate_success": np.nan,
                "conditional_pb_pass_success": np.nan,
                "conditional_rmsd_le_2p0_success": np.nan,
                "conditional_retention_ge_0p5_success": np.nan,
                "conditional_clash_free_success": np.nan,
            }
        )
    return out


def per_pocket_topk(frame: pd.DataFrame, methods: list[str], coverage: float, dataset: str, seed: int = 0) -> pd.DataFrame:
    rows = []
    group_cols = ["pocket_id"] if "generator" not in frame.columns else ["generator", "pocket_id"]
    for keys, group in frame.groupby(group_cols, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        nominal_k = max(1, int(math.ceil(coverage * len(group))))
        for method in methods:
            if method == "radical_precheck_qedsa" and "score_radical_precheck_qedsa" not in group.columns:
                continue
            if method == "no_sp_element_qedsa" and "score_no_sp_element_qedsa" not in group.columns:
                continue
            if method == "processability_oracle_qedsa" and "score_processability_oracle_qedsa" not in group.columns:
                continue
            if method in {"logreg_endpoint", "hgb_endpoint"} and f"score_{method.replace('_endpoint', '_full')}" not in group.columns:
                continue
            selected = select_top(group, method, coverage)
            row = {"dataset": dataset, "seed": seed, "method": method}
            row.update(dict(zip(group_cols, key_tuple)))
            row.update(summarize_selection(selected, nominal_k))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_topk(per_pocket: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        per_pocket.groupby(group_cols + ["method"], sort=False)
        .agg(
            n_groups=("endpoint_positive_rate", "size"),
            selected_count=("n_selected", "sum"),
            nominal_count=("nominal_k", "sum"),
            endpoint_positive_mean=("endpoint_positive_rate", "mean"),
            endpoint_positive_sd=("endpoint_positive_rate", "std"),
            failures_per_1000_selected=("failures_per_1000_selected", "mean"),
            failures_per_1000_nominal=("failures_per_1000_nominal", "mean"),
            relaxation_success_mean=("relaxation_success_rate", "mean"),
            conditional_endpoint_rate_success_mean=("conditional_endpoint_rate_success", "mean"),
            conditional_pb_pass_success_mean=("conditional_pb_pass_success", "mean"),
            effective_coverage_mean=("effective_coverage_fraction", "mean"),
        )
        .reset_index()
    )
    return agg.fillna({"endpoint_positive_sd": 0.0})


def paired_bootstrap(per_pocket: pd.DataFrame, group_cols: list[str], references: list[str], seed: int = 20260814) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tmp = per_pocket.copy()
    tmp["group_key"] = tmp[group_cols].astype(str).agg("||".join, axis=1)
    wide = tmp.pivot_table(index="group_key", columns="method", values="endpoint_positive_rate", aggfunc="mean")
    rows = []
    for method in wide.columns:
        for ref in references:
            if method == ref or ref not in wide.columns:
                continue
            pair = wide[[method, ref]].dropna()
            if pair.empty:
                continue
            delta = (pair[method] - pair[ref]).to_numpy(float)
            boot = np.asarray([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(3000)])
            rows.append(
                {
                    "method": method,
                    "reference": ref,
                    "n_groups": int(len(delta)),
                    "mean_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def safe_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y, score = y[mask], score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    prob = np.clip(score, 1e-6, 1.0 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, prob)) if 0 <= float(np.nanmin(score)) and float(np.nanmax(score)) <= 1 else np.nan,
    }


def impute(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    xte = imp.transform(test[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    return xtr, xte


def add_double_holdout_predictions(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    rows = []
    frame = frame.copy()
    generators = sorted(frame["generator"].dropna().unique())
    for held_generator in generators:
        target_pockets = frame.loc[frame["generator"].eq(held_generator), "pocket_id"].astype(str).unique()
        for fold_idx, test_pockets in enumerate(fold_pockets(target_pockets, folds, seed)):
            test_mask = frame["generator"].eq(held_generator) & frame["pocket_id"].astype(str).isin(test_pockets)
            train_mask = (~frame["generator"].eq(held_generator)) & (~frame["pocket_id"].astype(str).isin(test_pockets))
            train = frame[train_mask].copy()
            test = frame[test_mask].copy()
            if test.empty or train.empty:
                continue
            y = train[TARGET].astype(int).to_numpy()
            xtr, xte = impute(train, test, FULL_FEATURES)
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed + fold_idx))
            lr.fit(xtr, y)
            test["score_double_logreg_full"] = lr.predict_proba(xte)[:, 1]
            hgb = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, l2_regularization=0.03, random_state=seed + fold_idx)
            weights = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
            hgb.fit(xtr, y, sample_weight=weights)
            test["score_double_hgb_full"] = hgb.predict_proba(xte)[:, 1]
            test["held_out_generator"] = held_generator
            test["fold"] = fold_idx
            metric_rows = []
            for score_col in ["score_double_logreg_full", "score_double_hgb_full"]:
                metrics = safe_metrics(test[TARGET].astype(int).to_numpy(), test[score_col].to_numpy())
                metrics.update(
                    {
                        "held_out_generator": held_generator,
                        "fold": fold_idx,
                        "method": score_col.replace("score_double_", ""),
                        "train_candidates": int(len(train)),
                        "test_candidates": int(len(test)),
                        "train_pockets": int(train["pocket_id"].nunique()),
                        "test_pockets": int(test["pocket_id"].nunique()),
                    }
                )
                metric_rows.append(metrics)
            rows.append((test, pd.DataFrame(metric_rows)))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    pred = pd.concat([r[0] for r in rows], ignore_index=True)
    metrics = pd.concat([r[1] for r in rows], ignore_index=True)
    return pred, metrics


def evaluate_double_holdout(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = pred.copy()
    out["score_logreg_double"] = out["score_double_logreg_full"]
    out["score_hgb_double"] = out["score_double_hgb_full"]
    methods = [
        "qed_sa",
        "contact_clash_rule",
        "full_audit_gate_qedsa_fixed_budget",
        "audit_failure_count_qedsa",
        "no_sp_element_qedsa",
    ]
    per_rows = []
    for _, group in out.groupby(["held_out_generator", "fold", "pocket_id"], sort=False):
        nominal_k = max(1, int(math.ceil(0.10 * len(group))))
        for method in methods:
            selected = select_top(group, method, 0.10)
            row = {
                "held_out_generator": group["held_out_generator"].iloc[0],
                "fold": int(group["fold"].iloc[0]),
                "pocket_id": group["pocket_id"].iloc[0],
                "method": method,
            }
            row.update(summarize_selection(selected, nominal_k))
            per_rows.append(row)
        for score_col, method in [("score_double_logreg_full", "double_logreg_full"), ("score_double_hgb_full", "double_hgb_full")]:
            selected = group.nlargest(nominal_k, score_col)
            row = {
                "held_out_generator": group["held_out_generator"].iloc[0],
                "fold": int(group["fold"].iloc[0]),
                "pocket_id": group["pocket_id"].iloc[0],
                "method": method,
            }
            row.update(summarize_selection(selected, nominal_k))
            per_rows.append(row)
    per = pd.DataFrame(per_rows)
    summary = aggregate_topk(per, ["held_out_generator"])
    boot = []
    for held, sub in per.groupby("held_out_generator", sort=False):
        b = paired_bootstrap(sub, ["pocket_id"], ["qed_sa", "contact_clash_rule", "full_audit_gate_qedsa_fixed_budget"])
        b.insert(0, "held_out_generator", held)
        boot.append(b)
    return per, summary, pd.concat(boot, ignore_index=True) if boot else pd.DataFrame()


def write_disjoint_check(diff: pd.DataFrame, multi: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    diff_pockets = set(diff["pocket_id"].astype(str))
    for generator, sub in multi.groupby("generator", sort=True):
        target_pockets = set(sub["pocket_id"].astype(str))
        rows.append(
            {
                "source_training_set": "DiffSBDD OpenMM endpoint",
                "target_generator": generator,
                "source_pockets": len(diff_pockets),
                "target_pockets": len(target_pockets),
                "overlap_pockets": len(diff_pockets & target_pockets),
                "is_generator_unseen": True,
                "is_receptor_context_unseen": len(diff_pockets & target_pockets) == 0,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "table_F_zero_shot_pocket_disjoint_check.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bib_reviewer_gap_closure_v0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_raw = pd.read_csv(ROOT / "outputs/bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv")
    audit_oof = pd.read_csv(ROOT / "outputs/revision_r1_leakage_shift_audit_v0/main_oof_predictions.csv")
    diff_oof = pd.read_csv(ROOT / "outputs/bib_physical_endpoint_analysis_v0/physical_oof_predictions.csv")
    diff_by_seed = []
    for seed, pred in diff_oof.groupby("seed", sort=True):
        phys_score_cols = [c for c in pred.columns if c.startswith("score_phys_")]
        audit_pred = audit_oof[audit_oof["seed"].eq(seed)]
        audit_score_cols = [c for c in audit_pred.columns if c.startswith("score_")]
        merged = diff_raw.merge(audit_pred[["candidate_uid"] + audit_score_cols], on="candidate_uid", how="left")
        merged = merged.merge(pred[["candidate_uid"] + phys_score_cols], on="candidate_uid", how="left")
        merged = add_common_scores(merged)
        if "score_label_informed_rule" in merged.columns:
            merged["score_contact_clash_rule"] = merged["score_label_informed_rule"]
        merged["seed"] = seed
        diff_by_seed.append(merged)
    diff_scored = pd.concat(diff_by_seed, ignore_index=True)

    multi = add_common_scores(pd.read_csv(ROOT / "outputs/bib_multigen_physical_transfer_v2/zero_shot_light_processed_predictions.csv"))

    direct_methods = [
        "qed_sa",
        "contact_clash_rule",
        "full_audit_gate_qedsa_fixed_budget",
        "full_audit_gate_qedsa_no_backfill",
        "audit_failure_count_qedsa",
        "phys_logreg_full",
        "phys_hgb_full",
        "phys_rf_direct_excluded",
        "phys_rf_full",
    ]
    diff_per_seed = []
    for seed, sub in diff_scored.groupby("seed", sort=True):
        diff_per_seed.append(per_pocket_topk(sub, direct_methods, 0.10, "DiffSBDD", int(seed)))
    diff_per = pd.concat(diff_per_seed, ignore_index=True)
    diff_per.to_csv(out_dir / "table_D_diffsbdd_full_audit_direct_per_pocket.csv", index=False)
    aggregate_topk(diff_per, ["dataset", "seed"]).groupby(["dataset", "method"], sort=False).agg(
        n_seed_rows=("endpoint_positive_mean", "size"),
        endpoint_positive_mean=("endpoint_positive_mean", "mean"),
        endpoint_positive_sd=("endpoint_positive_mean", "std"),
        selected_count_mean=("selected_count", "mean"),
        nominal_count_mean=("nominal_count", "mean"),
        failures_per_1000_selected_mean=("failures_per_1000_selected", "mean"),
        relaxation_success_mean=("relaxation_success_mean", "mean"),
        conditional_endpoint_rate_success_mean=("conditional_endpoint_rate_success_mean", "mean"),
        effective_coverage_mean=("effective_coverage_mean", "mean"),
    ).reset_index().to_csv(out_dir / "table_D_diffsbdd_full_audit_direct_summary.csv", index=False)
    paired_bootstrap(diff_per, ["pocket_id"], ["qed_sa", "contact_clash_rule", "full_audit_gate_qedsa_fixed_budget"]).to_csv(
        out_dir / "table_D_diffsbdd_full_audit_direct_bootstrap.csv", index=False
    )

    multi_methods = [
        "qed_sa",
        "contact_clash_rule",
        "full_audit_gate_qedsa_fixed_budget",
        "full_audit_gate_qedsa_no_backfill",
        "audit_failure_count_qedsa",
        "radical_precheck_qedsa",
        "no_sp_element_qedsa",
        "processability_oracle_qedsa",
        "logreg_endpoint",
        "hgb_endpoint",
    ]
    multi_per = per_pocket_topk(multi, multi_methods, 0.10, "TargetDiff-family", 0)
    multi_per.to_csv(out_dir / "table_E_multigen_processable_decomposition_per_pocket.csv", index=False)
    aggregate_topk(multi_per, ["dataset", "generator"]).to_csv(
        out_dir / "table_E_multigen_processable_decomposition_by_generator.csv", index=False
    )
    aggregate_topk(multi_per, ["dataset"]).to_csv(out_dir / "table_E_multigen_processable_decomposition_overall.csv", index=False)
    boot_parts = []
    for generator, sub in multi_per.groupby("generator", sort=False):
        b = paired_bootstrap(sub, ["pocket_id"], ["qed_sa", "contact_clash_rule", "full_audit_gate_qedsa_fixed_budget", "radical_precheck_qedsa", "no_sp_element_qedsa"])
        b.insert(0, "generator", generator)
        boot_parts.append(b)
    pd.concat(boot_parts, ignore_index=True).to_csv(out_dir / "table_E_multigen_processable_decomposition_bootstrap.csv", index=False)

    write_disjoint_check(add_common_scores(diff_raw), multi, out_dir)

    double_pred, double_metrics = add_double_holdout_predictions(multi, args.folds, args.seed)
    double_pred.to_csv(out_dir / "table_G_double_holdout_predictions.csv", index=False)
    double_metrics.to_csv(out_dir / "table_G_double_holdout_candidate_metrics_by_fold.csv", index=False)
    double_per, double_summary, double_boot = evaluate_double_holdout(double_pred)
    double_per.to_csv(out_dir / "table_G_double_holdout_per_pocket.csv", index=False)
    double_summary.to_csv(out_dir / "table_G_double_holdout_top10_by_generator.csv", index=False)
    double_boot.to_csv(out_dir / "table_G_double_holdout_bootstrap.csv", index=False)

    summary_lines = [
        "# BIB reviewer-gap closure experiments",
        "",
        "Generated tables:",
        "- table_D*: full five-rule audit direct-use baselines on DiffSBDD OpenMM endpoint.",
        "- table_E*: full-pipeline versus processable-subset decomposition on 38,073 four-generator candidates.",
        "- table_F*: zero-shot pocket-disjointness check.",
        "- table_G*: leave-one-generator plus pocket-holdout OpenMM endpoint evaluation.",
        "",
        "Zero-shot pocket-disjointness:",
        pd.read_csv(out_dir / "table_F_zero_shot_pocket_disjoint_check.csv").to_csv(index=False).strip(),
        "",
    ]
    (out_dir / "bib_reviewer_gap_closure_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("done", out_dir)
    print(pd.read_csv(out_dir / "table_D_diffsbdd_full_audit_direct_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "table_E_multigen_processable_decomposition_overall.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "table_F_zero_shot_pocket_disjoint_check.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "table_G_double_holdout_top10_by_generator.csv").to_string(index=False))


if __name__ == "__main__":
    main()
