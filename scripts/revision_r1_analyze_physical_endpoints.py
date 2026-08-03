#!/usr/bin/env python3
"""Analyze independent relaxation endpoints and train physics-targeted rerankers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
DIRECT_RULE_VARIABLES = {"rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok", "pose_radius_shape"}
DIRECT_EXCLUDED_FEATURES = [c for c in FULL_FEATURES if c not in DIRECT_RULE_VARIABLES]
NON_LABEL_PRIMITIVE_FEATURES = ["mol_wt", "logp", "tpsa", "qed", "ring_count", "hetero_atoms"] + ATOM_FEATURES


def fold_pockets(pockets: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pockets = np.asarray(pockets).copy()
    rng.shuffle(pockets)
    return [part for part in np.array_split(pockets, min(folds, len(pockets))) if len(part)]


def safe_metrics(y: pd.Series, score: pd.Series) -> dict[str, float]:
    mask = y.notna() & score.notna()
    yy, ss = y[mask].astype(int).to_numpy(), score[mask].astype(float).to_numpy()
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece10": np.nan}
    prob = np.clip(ss, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(yy, ss)),
        "pr_auc": float(average_precision_score(yy, ss)),
        "brier": float(brier_score_loss(yy, prob)),
        "ece10": expected_calibration_error(yy, prob, bins=10),
    }


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return ece


def load_relaxation(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("relaxation_shard_*.csv"))
    if not files:
        raise FileNotFoundError(f"No relaxation shard CSV files found in {path}")
    table = pd.concat([pd.read_csv(file) for file in files], ignore_index=True, sort=False)
    return table.drop_duplicates("candidate_uid", keep="last")


def endpoint_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    success = out["status"].eq("success")
    out["relaxation_success"] = success
    for threshold in [1.0, 2.0, 3.0]:
        out[f"rmsd_le_{str(threshold).replace('.', 'p')}"] = success & out["ligand_rmsd_a"].le(threshold)
    for threshold in [0.3, 0.5, 0.7]:
        out[f"retention_ge_{str(threshold).replace('.', 'p')}"] = success & out["contact_retention"].ge(threshold)
    out["final_clash_free"] = success & out["final_clash_atoms_1p2"].eq(0)
    out["final_pb_pass"] = success & out["final_posebusters_pass"].fillna(False).astype(bool)
    for rmsd in [1.0, 2.0, 3.0]:
        for retention in [0.3, 0.5, 0.7]:
            key = f"stable_r{str(rmsd).replace('.', 'p')}_c{str(retention).replace('.', 'p')}"
            base = success & out["ligand_rmsd_a"].le(rmsd) & out["contact_retention"].ge(retention) & out["final_clash_atoms_1p2"].eq(0)
            out[f"{key}_no_pb"] = base
            out[f"{key}_with_pb"] = base & out["final_posebusters_pass"].fillna(False).astype(bool)
    return out


def impute(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    xte = imp.transform(test[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
    return xtr, xte


def fit_physics_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    seed: int,
    jobs: int,
    rf_trees: int,
) -> dict[str, np.ndarray]:
    y = train[target].astype(int).to_numpy()
    outputs = {}
    for protocol, cols in {
        "full": FULL_FEATURES,
        "direct_excluded": DIRECT_EXCLUDED_FEATURES,
        "nonlabel_primitive": NON_LABEL_PRIMITIVE_FEATURES,
    }.items():
        xtr, xte = impute(train, test, cols)
        rf = RandomForestClassifier(n_estimators=rf_trees, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=jobs, random_state=seed)
        rf.fit(xtr, y)
        outputs[f"phys_rf_{protocol}"] = rf.predict_proba(xte)[:, 1]
        if protocol == "full":
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed))
            lr.fit(xtr, y)
            outputs["phys_logreg_full"] = lr.predict_proba(xte)[:, 1]
            hgb = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, l2_regularization=0.03, random_state=seed)
            weight = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
            hgb.fit(xtr, y, sample_weight=weight)
            outputs["phys_hgb_full"] = hgb.predict_proba(xte)[:, 1]
    return outputs


def topk_table(frame: pd.DataFrame, score_cols: list[str], target: str, seed: int, source: str, frac: float = 0.1):
    rows, pocket_rows = [], []
    for score_col in score_cols:
        selected = []
        for pocket, group in frame.groupby("pocket_id", sort=False):
            k = max(1, int(math.ceil(frac * len(group))))
            top = group.nlargest(k, score_col)
            selected.append(top)
            pocket_rows.append({
                "seed": seed,
                "source": source,
                "method": score_col.replace("score_", "", 1),
                "pocket_id": pocket,
                "n_selected": len(top),
                "stable_rate": float(top[target].mean()),
                "relaxation_success_rate": float(top["relaxation_success"].mean()),
            })
        top = pd.concat(selected, ignore_index=True)
        rows.append({
            "seed": seed,
            "source": source,
            "method": score_col.replace("score_", "", 1),
            "coverage": frac,
            "n_selected": len(top),
            "stable_rate": float(top[target].mean()),
            "stable_count": int(top[target].sum()),
            "relaxation_success_rate": float(top["relaxation_success"].mean()),
            "final_pb_pass_rate": float(top["final_pb_pass"].mean()),
            "final_clash_free_rate": float(top["final_clash_free"].mean()),
            "median_ligand_rmsd_a": float(top.loc[top["relaxation_success"], "ligand_rmsd_a"].median()),
            "mean_contact_retention": float(top.loc[top["relaxation_success"], "contact_retention"].mean()),
        })
    return pd.DataFrame(rows), pd.DataFrame(pocket_rows)


def coverage_sweep(frame: pd.DataFrame, score_cols: list[str], target: str, seed: int, source: str, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for coverage in coverages:
        top, _ = topk_table(frame, score_cols, target, seed, source, coverage)
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def worst_pocket_summary(per_pocket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if per_pocket.empty:
        return pd.DataFrame(rows)
    for (source, method), group in per_pocket.groupby(["source", "method"], sort=False):
        values = group["stable_rate"].astype(float).to_numpy()
        rows.append({
            "source": source,
            "method": method,
            "n_pockets": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "bottom_quartile_mean": float(np.mean(values[values <= np.quantile(values, 0.25)])),
            "perfect_pockets": int(np.sum(values >= 1.0)),
        })
    return pd.DataFrame(rows)


def endpoint_overlap(frame: pd.DataFrame, physical_target: str) -> pd.DataFrame:
    rows = []
    if "reliable" not in frame or physical_target not in frame:
        return pd.DataFrame(rows)
    audit = frame["reliable"].astype(bool)
    physical = frame[physical_target].astype(bool)
    for audit_value in [False, True]:
        for physical_value in [False, True]:
            mask = audit.eq(audit_value) & physical.eq(physical_value)
            rows.append({
                "audit_reliable": bool(audit_value),
                "physical_stable": bool(physical_value),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
            })
    rows.append({
        "audit_reliable": "all",
        "physical_stable": "agreement",
        "count": int(audit.eq(physical).sum()),
        "fraction": float(audit.eq(physical).mean()),
    })
    return pd.DataFrame(rows)


def bootstrap(per_pocket: pd.DataFrame, references: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for source, group in per_pocket.groupby("source"):
        wide = group.groupby(["pocket_id", "method"], as_index=False)["stable_rate"].mean().pivot(index="pocket_id", columns="method", values="stable_rate")
        methods = [m for m in wide.columns if m.startswith(("phys_rf", "phys_hgb", "rf_", "hgb_"))]
        for method in methods:
            for reference in references:
                if reference not in wide:
                    continue
                pair = wide[[method, reference]].dropna()
                delta = (pair[method] - pair[reference]).to_numpy(float)
                boots = np.asarray([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(n_boot)])
                rows.append({
                    "source": source,
                    "method": method,
                    "reference": reference,
                    "n_pockets": len(delta),
                    "mean_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boots, 0.025)),
                    "ci_high": float(np.quantile(boots, 0.975)),
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.feature_frame)
    relaxation = endpoint_columns(load_relaxation(args.relaxation_dir))
    endpoint_keep = [c for c in relaxation.columns if c not in base.columns or c == "candidate_uid"]
    frame = base.merge(relaxation[endpoint_keep], on="candidate_uid", how="left")
    frame["relaxation_success"] = frame["relaxation_success"].fillna(False)
    bool_endpoint_cols = [
        c for c in frame.columns
        if c.startswith(("stable_", "rmsd_le_", "retention_ge_", "final_pb_"))
        or c in {"final_clash_free", "final_pb_pass", "final_posebusters_pass"}
    ]
    for col in bool_endpoint_cols:
        if col not in {"final_posebusters_error"}:
            frame[col] = frame[col].fillna(False)
    target = args.target
    if target not in frame:
        raise KeyError(target)

    predictions = pd.read_csv(args.oof_predictions)
    fixed_cols = [c for c in predictions.columns if c.startswith("score_")]
    old_rows, pocket_rows, metric_rows, sweep_rows, prediction_rows = [], [], [], [], []
    coverages = sorted(set(args.coverages + [args.coverage]))
    for seed, pred in predictions.groupby("seed", sort=False):
        merged = frame.merge(pred[["candidate_uid"] + fixed_cols], on="candidate_uid", how="left")
        if fixed_cols and merged[fixed_cols].notna().sum().sum() == 0:
            continue
        top, pockets = topk_table(merged, fixed_cols, target, int(seed), "audit_label_oof", args.coverage)
        old_rows.append(top)
        pocket_rows.append(pockets)
        sweep_rows.append(coverage_sweep(merged, fixed_cols, target, int(seed), "audit_label_oof", coverages))
        for score_col in fixed_cols:
            metric = safe_metrics(merged[target].astype(int), merged[score_col])
            metric.update({"seed": seed, "source": "audit_label_oof", "method": score_col.replace("score_", "", 1), "target": target})
            metric_rows.append(metric)
            audit_metric = safe_metrics(merged["reliable"].astype(int), merged[score_col])
            audit_metric.update({"seed": seed, "source": "audit_label_oof", "method": score_col.replace("score_", "", 1), "target": "audit_reliable"})
            metric_rows.append(audit_metric)

    physics_rows = []
    for seed in args.seeds:
        pred = frame.copy()
        learned_cols = set()
        for fold_idx, heldout in enumerate(fold_pockets(frame["pocket_id"].unique(), args.folds, seed), start=1):
            test_mask = frame["pocket_id"].isin(set(heldout))
            train, test = frame.loc[~test_mask], frame.loc[test_mask]
            scores = fit_physics_models(train, test, target, seed + fold_idx, args.jobs, args.rf_trees)
            for name, values in scores.items():
                col = f"score_{name}"
                pred.loc[test.index, col] = values
                learned_cols.add(col)
        score_cols = sorted(learned_cols)
        top, pockets = topk_table(pred, score_cols, target, seed, "physics_endpoint_oof", args.coverage)
        physics_rows.append(top)
        pocket_rows.append(pockets)
        sweep_rows.append(coverage_sweep(pred, score_cols, target, seed, "physics_endpoint_oof", coverages))
        prediction_rows.append(pred[["candidate_uid", "pocket_id", "reliable", target] + score_cols].assign(seed=seed, source="physics_endpoint_oof"))
        for score_col in score_cols:
            metric = safe_metrics(pred[target].astype(int), pred[score_col])
            metric.update({"seed": seed, "source": "physics_endpoint_oof", "method": score_col.replace("score_", "", 1), "target": target})
            metric_rows.append(metric)
            audit_metric = safe_metrics(pred["reliable"].astype(int), pred[score_col])
            audit_metric.update({"seed": seed, "source": "physics_endpoint_oof", "method": score_col.replace("score_", "", 1), "target": "audit_reliable"})
            metric_rows.append(audit_metric)

    all_top = pd.concat(old_rows + physics_rows, ignore_index=True)
    all_pockets = pd.concat(pocket_rows, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    all_sweeps = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    all_predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    frame.to_csv(args.output_dir / "relaxation_endpoint_frame.csv", index=False)
    all_top.to_csv(args.output_dir / "physical_top10_by_seed.csv", index=False)
    all_pockets.to_csv(args.output_dir / "physical_per_pocket_top10.csv", index=False)
    all_metrics.to_csv(args.output_dir / "physical_metrics_by_seed.csv", index=False)
    all_sweeps.to_csv(args.output_dir / "physical_coverage_sweep.csv", index=False)
    all_predictions.to_csv(args.output_dir / "physical_oof_predictions.csv", index=False)
    worst_pocket_summary(all_pockets).to_csv(args.output_dir / "physical_worst_pockets.csv", index=False)
    endpoint_overlap(frame, target).to_csv(args.output_dir / "physical_endpoint_overlap.csv", index=False)
    bootstrap(
        all_pockets,
        ["label_informed_rule", "logreg_full", "phys_logreg_full", "qed_sa"],
        args.bootstrap,
        20260630,
    ).to_csv(args.output_dir / "physical_paired_bootstrap.csv", index=False)

    prevalence_rows = []
    for col in [c for c in frame.columns if c.startswith("stable_")]:
        prevalence_rows.append({"endpoint": col, "positive_rate": float(frame[col].mean()), "n_positive": int(frame[col].sum()), "n_total": len(frame)})
    pd.DataFrame(prevalence_rows).to_csv(args.output_dir / "endpoint_sensitivity.csv", index=False)
    metadata = {
        "candidates": len(frame),
        "pockets": int(frame["pocket_id"].nunique()),
        "relaxation_success_rate": float(frame["relaxation_success"].mean()),
        "primary_target": target,
        "physics_model_training": "pocket-held-out cross-validation on independent post-relaxation endpoint",
        "old_model_evaluation": "existing audit-label OOF scores evaluated zero-shot against post-relaxation endpoint",
        "rf_trees": int(args.rf_trees),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv"))
    p.add_argument("--relaxation-dir", type=Path, default=Path("outputs/revision_r1_openmm_relaxation_v0"))
    p.add_argument("--oof-predictions", type=Path, default=Path("outputs/revision_r1_leakage_shift_audit_v0/main_oof_predictions.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/revision_r1_physical_endpoint_analysis_v0"))
    p.add_argument("--target", default="stable_r2p0_c0p5_with_pb")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--rf-trees", type=int, default=400)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--coverage", type=float, default=0.1)
    p.add_argument("--coverages", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.5])
    return p.parse_args()


if __name__ == "__main__":
    main()
