#!/usr/bin/env python3
"""Fast pocket-held-out physical endpoint modeling on public-PDB aligned coordinates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_FRAME = Path("outputs/diffsbdd_aligned_reliamol_v0/reports/feature_frame.csv")
RELAX_DIR = Path("outputs/jctc_aligned_openmm_relaxation_v0")
OOF = Path("outputs/revision_r1_leakage_shift_audit_v0/main_oof_predictions.csv")
OUT = Path("outputs/jctc_aligned_physical_cv_v1")
TARGET = "stable_r2p0_c0p5_with_pb"

ATOM_FEATURES = [
    "atom_metal_frac", "atom_C_frac", "atom_N_frac", "atom_O_frac", "atom_S_frac",
    "atom_P_frac", "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
]
CHEM_FEATURES = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM_FEATURES
INTERNAL_FEATURES = [
    "pose_atoms", "nn_q05", "nn_q10", "nn_q25", "nn_q50", "nn_q75", "nn_q90",
]
POCKET_FEATURES = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90",
    "shell_0_2", "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
FULL_FEATURES = CHEM_FEATURES + INTERNAL_FEATURES + POCKET_FEATURES
DIRECT_RULE_VARIABLES = {"rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok"}
DIRECT_EXCLUDED_FEATURES = [c for c in FULL_FEATURES if c not in DIRECT_RULE_VARIABLES]
CHEM_PRIMITIVE_FEATURES = ["mol_wt", "logp", "tpsa", "qed", "ring_count", "hetero_atoms"] + ATOM_FEATURES


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).fillna(False)


def endpoint_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    success = out["status"].eq("success")
    out["relaxation_success"] = success
    out["final_pb_pass"] = success & bool_series(out.get("final_posebusters_pass", pd.Series(False, index=out.index)))
    out["final_clash_free"] = success & out["final_clash_atoms_1p2"].fillna(1).eq(0)
    out[TARGET] = (
        success
        & out["ligand_rmsd_a"].fillna(np.inf).le(2.0)
        & out["contact_retention"].fillna(-np.inf).ge(0.5)
        & out["final_clash_atoms_1p2"].fillna(1).eq(0)
        & out["final_pb_pass"]
    )
    return out


def load_relaxation() -> pd.DataFrame:
    files = sorted(RELAX_DIR.glob("relaxation_shard_*.csv"))
    if len(files) != 8:
        raise FileNotFoundError(f"Expected 8 relaxation shards, found {len(files)}")
    return endpoint_columns(pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=False).drop_duplicates("candidate_uid", keep="last"))


def fold_pockets(pockets: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pockets = np.asarray(pockets).copy()
    rng.shuffle(pockets)
    return [part for part in np.array_split(pockets, min(folds, len(pockets))) if len(part)]


def impute(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    available = [c for c in cols if c in train.columns and c in test.columns]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[available].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    xte = imp.transform(test[available].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    return xtr, xte


def safe_metrics(y: pd.Series, score: pd.Series) -> dict[str, float]:
    mask = y.notna() & score.notna()
    yy = y[mask].astype(int).to_numpy()
    ss = score[mask].astype(float).to_numpy()
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    prob = np.clip(ss, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(yy, ss)),
        "pr_auc": float(average_precision_score(yy, ss)),
        "brier": float(brier_score_loss(yy, prob)),
    }


def fit_models(train: pd.DataFrame, test: pd.DataFrame, seed: int, jobs: int) -> dict[str, np.ndarray]:
    y = train[TARGET].astype(int).to_numpy()
    outputs = {}
    for name, cols in {
        "chem_primitive": CHEM_PRIMITIVE_FEATURES,
        "direct_excluded": DIRECT_EXCLUDED_FEATURES,
        "full": FULL_FEATURES,
    }.items():
        xtr, xte = impute(train, test, cols)
        if name in {"chem_primitive", "full"}:
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed))
            lr.fit(xtr, y)
            outputs[f"aligned_logreg_{name}"] = lr.predict_proba(xte)[:, 1]
        if name in {"direct_excluded", "full"}:
            hgb = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, l2_regularization=0.03, random_state=seed)
            weight = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
            hgb.fit(xtr, y, sample_weight=weight)
            outputs[f"aligned_hgb_{name}"] = hgb.predict_proba(xte)[:, 1]
    return outputs


def topk(frame: pd.DataFrame, score_cols: list[str], coverage: float, seed: int, source: str):
    rows, pocket_rows = [], []
    for score_col in score_cols:
        method = score_col.replace("score_", "", 1)
        selected = []
        for pocket_id, group in frame.groupby("pocket_id", sort=False):
            group = group.dropna(subset=[score_col])
            if group.empty:
                continue
            k = max(1, int(math.ceil(coverage * len(group))))
            top = group.nlargest(k, score_col)
            selected.append(top)
            pocket_rows.append({
                "seed": seed,
                "source": source,
                "method": method,
                "coverage": coverage,
                "pocket_id": pocket_id,
                "n_selected": int(len(top)),
                "stable_rate": float(top[TARGET].mean()),
                "relaxation_success_rate": float(top["relaxation_success"].mean()),
                "final_pb_pass_rate": float(top["final_pb_pass"].mean()),
                "median_ligand_rmsd_a": float(top.loc[top["relaxation_success"], "ligand_rmsd_a"].median()),
                "mean_contact_retention": float(top.loc[top["relaxation_success"], "contact_retention"].mean()),
            })
        if not selected:
            continue
        picked = pd.concat(selected, ignore_index=True)
        rows.append({
            "seed": seed,
            "source": source,
            "method": method,
            "coverage": coverage,
            "n_selected": int(len(picked)),
            "stable_rate": float(picked[TARGET].mean()),
            "relaxation_success_rate": float(picked["relaxation_success"].mean()),
            "final_pb_pass_rate": float(picked["final_pb_pass"].mean()),
            "median_ligand_rmsd_a": float(picked.loc[picked["relaxation_success"], "ligand_rmsd_a"].median()),
            "mean_contact_retention": float(picked.loc[picked["relaxation_success"], "contact_retention"].mean()),
        })
    return pd.DataFrame(rows), pd.DataFrame(pocket_rows)


def mean_table(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for group_keys, group in frame.groupby(keys, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(keys, group_keys))
        row["n"] = int(len(group))
        for value in values:
            row[f"{value}_mean"] = float(group[value].mean())
            row[f"{value}_sd"] = float(group[value].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def worst_pocket(per_pocket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, method, coverage), group in per_pocket.groupby(["source", "method", "coverage"], sort=False):
        values = group["stable_rate"].to_numpy(float)
        rows.append({
            "source": source,
            "method": method,
            "coverage": coverage,
            "n_pockets": int(len(values)),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "bottom_quartile_mean": float(np.mean(values[values <= np.quantile(values, 0.25)])),
            "perfect_pockets": int(np.sum(values >= 1.0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature = pd.read_csv(FEATURE_FRAME)
    relax = load_relaxation()
    endpoint_keep = [c for c in relax.columns if c not in feature.columns or c == "candidate_uid"]
    frame = feature.merge(relax[endpoint_keep], on="candidate_uid", how="inner")
    seeds = [13, 29, 47]
    coverages = [0.01, 0.05, 0.10, 0.20, 0.50]
    top_rows, pocket_rows, metric_rows, prediction_rows = [], [], [], []

    # Include original zero-shot audit-label scores for direct comparison.
    oof = pd.read_csv(OOF)
    zero_cols = [
        "score_qed_sa",
        "score_label_informed_rule",
        "score_logreg_full",
        "score_rf_full",
        "score_hgb_full",
    ]
    for seed, pred in oof.groupby("seed", sort=False):
        merged = frame.merge(pred[["candidate_uid"] + zero_cols], on="candidate_uid", how="left")
        for coverage in coverages:
            top, pp = topk(merged, zero_cols, coverage, int(seed), "audit_label_zero_shot")
            top_rows.append(top)
            pocket_rows.append(pp)
        for score_col in zero_cols:
            metric = safe_metrics(merged[TARGET], merged[score_col])
            metric.update({"seed": int(seed), "source": "audit_label_zero_shot", "method": score_col.replace("score_", "", 1)})
            metric_rows.append(metric)

    for seed in seeds:
        pred = frame.copy()
        learned_cols = []
        for fold_idx, heldout in enumerate(fold_pockets(frame["pocket_id"].unique(), 5, seed), start=1):
            test_mask = frame["pocket_id"].isin(set(heldout))
            train, test = frame.loc[~test_mask], frame.loc[test_mask]
            scores = fit_models(train, test, seed + fold_idx, jobs=1)
            print(f"seed={seed} fold={fold_idx}/5 train={len(train)} test={len(test)} models={len(scores)}", flush=True)
            for name, values in scores.items():
                col = f"score_{name}"
                pred.loc[test.index, col] = values
                if col not in learned_cols:
                    learned_cols.append(col)
        for coverage in coverages:
            top, pp = topk(pred, learned_cols, coverage, seed, "aligned_physical_oof")
            top_rows.append(top)
            pocket_rows.append(pp)
        prediction_rows.append(pred[["candidate_uid", "pocket_id", TARGET] + learned_cols].assign(seed=seed))
        for score_col in learned_cols:
            metric = safe_metrics(pred[TARGET], pred[score_col])
            metric.update({"seed": seed, "source": "aligned_physical_oof", "method": score_col.replace("score_", "", 1)})
            metric_rows.append(metric)

    all_top = pd.concat(top_rows, ignore_index=True)
    all_pockets = pd.concat(pocket_rows, ignore_index=True)
    all_preds = pd.concat(prediction_rows, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    all_top.to_csv(OUT / "aligned_physical_topk_by_seed.csv", index=False)
    all_pockets.to_csv(OUT / "aligned_physical_per_pocket_topk.csv", index=False)
    all_preds.to_csv(OUT / "aligned_physical_oof_predictions.csv", index=False)
    all_metrics.to_csv(OUT / "aligned_physical_metrics_by_seed.csv", index=False)
    mean_table(
        all_top,
        ["source", "method", "coverage"],
        ["stable_rate", "relaxation_success_rate", "final_pb_pass_rate", "median_ligand_rmsd_a", "mean_contact_retention"],
    ).to_csv(OUT / "aligned_physical_topk_mean.csv", index=False)
    worst_pocket(all_pockets).to_csv(OUT / "aligned_physical_worst_pockets.csv", index=False)
    metadata = {
        "candidates": int(len(frame)),
        "pockets": int(frame["pocket_id"].nunique()),
        "target": TARGET,
        "seeds": seeds,
        "coverages": coverages,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
