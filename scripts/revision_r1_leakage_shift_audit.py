#!/usr/bin/env python3
"""Reviewer-driven leakage, baseline, and double-holdout audits for ReliaMol-3D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
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

# Exact variables or aliases that enter the deterministic label rules.
DIRECT_RULE_VARIABLES = {
    "rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy",
    "descriptor_ok", "pose_radius_shape",
}
DIRECT_EXCLUDED_FEATURES = [c for c in FULL_FEATURES if c not in DIRECT_RULE_VARIABLES]

# Variables that do not share the chemistry, internal-distance, or receptor-distance
# primitives used to instantiate the five audit labels. This deliberately severe
# protocol quantifies how much signal remains after removing all label-source primitives.
NON_LABEL_PRIMITIVE_FEATURES = [
    "mol_wt", "logp", "tpsa", "qed", "ring_count", "hetero_atoms",
] + ATOM_FEATURES

FAILURE_TARGETS = {
    "chemical_failure": INTERNAL_FEATURES + POCKET_FEATURES,
    "synthetic_failure": INTERNAL_FEATURES + POCKET_FEATURES,
    "geometric_failure": CHEM_FEATURES + POCKET_FEATURES,
    "pocket_failure": CHEM_FEATURES + INTERNAL_FEATURES,
    "clash_failure": CHEM_FEATURES + INTERNAL_FEATURES,
}


def rank01(values: pd.Series | np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", pct=True).to_numpy(float)


def score_name(column: str) -> str:
    return column[6:] if column.startswith("score_") else column


def safe_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    prob = np.clip(score, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, prob)),
    }


def fold_pockets(pockets: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pockets = np.asarray(pockets).copy()
    rng.shuffle(pockets)
    return [x for x in np.array_split(pockets, min(n_folds, len(pockets))) if len(x)]


def matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[cols].replace([np.inf, -np.inf], np.nan))
    xte = imp.transform(test[cols].replace([np.inf, -np.inf], np.nan))
    return xtr.astype(np.float32), xte.astype(np.float32)


def fit_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    seed: int,
    jobs: int,
    prefix: str,
    include_iso: bool = True,
) -> dict[str, np.ndarray]:
    xtr, xte = matrix(train, test, cols)
    ytr = train["reliable"].to_numpy(int)
    out: dict[str, np.ndarray] = {}
    if len(np.unique(ytr)) < 2:
        base = np.full(len(test), float(np.mean(ytr)))
        for name in ["logreg", "rf", "hgb"]:
            out[f"{name}_{prefix}"] = base.copy()
        if include_iso:
            out[f"isoforest_{prefix}"] = base.copy()
        return out

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, class_weight="balanced", C=1.0, random_state=seed),
    )
    logreg.fit(xtr, ytr)
    out[f"logreg_{prefix}"] = logreg.predict_proba(xte)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=320,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=jobs,
        random_state=seed,
    )
    rf.fit(xtr, ytr)
    out[f"rf_{prefix}"] = rf.predict_proba(xte)[:, 1]

    hgb = HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.05,
        l2_regularization=0.03,
        max_leaf_nodes=31,
        random_state=seed,
    )
    sample_weight = np.where(ytr == 1, 0.5 / max(np.mean(ytr), 1e-6), 0.5 / max(1 - np.mean(ytr), 1e-6))
    hgb.fit(xtr, ytr, sample_weight=sample_weight)
    out[f"hgb_{prefix}"] = hgb.predict_proba(xte)[:, 1]

    if include_iso:
        iso = make_pipeline(
            StandardScaler(),
            IsolationForest(
                n_estimators=400,
                contamination=float(np.clip(1 - np.mean(ytr), 0.01, 0.30)),
                n_jobs=jobs,
                random_state=seed,
            ),
        )
        iso.fit(xtr)
        out[f"isoforest_{prefix}"] = rank01(iso.score_samples(xte))
    return out


def add_fixed_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["score_qed_sa"] = out["qed"].astype(float) - 0.25 * out["sa_proxy"].astype(float)
    if {"receptor_clash_atoms_1p2", "receptor_contact_atoms_4p5"}.issubset(out.columns):
        out["score_label_informed_rule"] = (
            out["qed"].astype(float)
            - 0.25 * out["sa_proxy"].astype(float)
            - 1.5 * out["receptor_clash_atoms_1p2"].astype(float).clip(0, 5)
            + 0.05 * out["receptor_contact_atoms_4p5"].astype(float).clip(0, 20)
        )
    out["score_smooth_pose_rule"] = (
        -np.abs(out["lp_q10"].astype(float) - 2.2)
        - 0.5 * out["shell_0_2"].astype(float)
        + 0.5 * out["shell_2_4"].astype(float)
        - 0.2 * out["shell_gt8"].astype(float)
        + 0.1 * out["qed"].astype(float)
    )
    return out


def topk_rows(frame: pd.DataFrame, score_cols: list[str], group_cols: list[str], frac: float = 0.1) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        selected = []
        for _, group in frame.groupby(group_cols, sort=False):
            k = max(1, int(math.ceil(frac * len(group))))
            selected.append(group.nlargest(k, score_col))
        top = pd.concat(selected, ignore_index=True)
        row = {
            "method": score_name(score_col),
            "coverage": frac,
            "n_selected": int(len(top)),
            "reliability": float(top["reliable"].mean()),
            "n_failures": int((1 - top["reliable"]).sum()),
        }
        for col in ["chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure", "posebusters_pass"]:
            if col in top.columns:
                row[col if col == "posebusters_pass" else f"{col}_rate"] = float(top[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def per_pocket_topk(frame: pd.DataFrame, score_cols: list[str], frac: float = 0.1) -> pd.DataFrame:
    rows = []
    for pocket, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(math.ceil(frac * len(group))))
        for score_col in score_cols:
            top = group.nlargest(k, score_col)
            rows.append({
                "pocket_id": pocket,
                "method": score_name(score_col),
                "reliability": float(top["reliable"].mean()),
                "n_selected": int(len(top)),
            })
    return pd.DataFrame(rows)


def bootstrap_deltas(per_pocket: pd.DataFrame, reference_methods: list[str], seed: int, n_boot: int = 5000) -> pd.DataFrame:
    wide = per_pocket.pivot(index="pocket_id", columns="method", values="reliability")
    rng = np.random.default_rng(seed)
    rows = []
    candidates = [c for c in wide.columns if c.startswith(("rf_", "hgb_"))]
    for method in candidates:
        for ref in reference_methods:
            if ref not in wide.columns:
                continue
            pair = wide[[method, ref]].dropna()
            delta = (pair[method] - pair[ref]).to_numpy(float)
            boots = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                boots[b] = np.mean(rng.choice(delta, size=len(delta), replace=True))
            rows.append({
                "method": method,
                "reference": ref,
                "n_pockets": len(delta),
                "mean_delta": float(np.mean(delta)),
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
                "win_pockets": int(np.sum(delta > 0)),
                "tie_pockets": int(np.sum(delta == 0)),
                "loss_pockets": int(np.sum(delta < 0)),
            })
    return pd.DataFrame(rows)


def merge_posebusters(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return frame
    pb = pd.read_csv(path)
    cols = [c for c in ["candidate_uid", "posebusters_pass"] if c in pb.columns]
    if len(cols) < 2:
        return frame
    pb = pb[cols].drop_duplicates("candidate_uid")
    return frame.merge(pb, on="candidate_uid", how="left")


def run_main(args: argparse.Namespace, output_dir: Path) -> None:
    frame = add_fixed_scores(pd.read_csv(args.main_frame))
    frame = merge_posebusters(frame, args.posebusters)
    base_score_cols = ["score_qed_sa", "score_smooth_pose_rule"]
    if "score_label_informed_rule" in frame:
        base_score_cols.append("score_label_informed_rule")
    metric_rows, top_rows, pocket_rows, prediction_parts = [], [], [], []

    for seed in args.seeds:
        pred = frame.copy()
        learned_cols: set[str] = set()
        for fold_idx, heldout in enumerate(fold_pockets(frame["pocket_id"].unique(), args.folds, seed), start=1):
            test_mask = frame["pocket_id"].isin(set(heldout))
            train, test = frame.loc[~test_mask], frame.loc[test_mask]
            split_scores: dict[str, np.ndarray] = {}
            split_scores.update(fit_scores(train, test, FULL_FEATURES, seed + fold_idx, args.jobs, "full"))
            split_scores.update(fit_scores(train, test, DIRECT_EXCLUDED_FEATURES, seed + 100 + fold_idx, args.jobs, "direct_excluded", include_iso=False))
            split_scores.update(fit_scores(train, test, NON_LABEL_PRIMITIVE_FEATURES, seed + 200 + fold_idx, args.jobs, "nonlabel_primitive", include_iso=False))
            for name, values in split_scores.items():
                col = f"score_{name}"
                pred.loc[test.index, col] = values
                learned_cols.add(col)

        score_cols = base_score_cols + sorted(learned_cols)
        for score_col in score_cols:
            values = pred[score_col].to_numpy(float)
            metrics = safe_metrics(pred["reliable"].to_numpy(int), rank01(values) if score_col in base_score_cols else values)
            metrics.update({"seed": seed, "method": score_name(score_col)})
            if "posebusters_pass" in pred:
                pb_ok = pred["posebusters_pass"].notna()
                pb_metrics = safe_metrics(pred.loc[pb_ok, "posebusters_pass"].astype(int).to_numpy(), rank01(pred.loc[pb_ok, score_col]))
                metrics.update({f"pb_{k}": v for k, v in pb_metrics.items()})
            metric_rows.append(metrics)
        top = topk_rows(pred, score_cols, ["pocket_id"], 0.1)
        top["seed"] = seed
        top_rows.append(top)
        pocket = per_pocket_topk(pred, score_cols, 0.1)
        pocket["seed"] = seed
        pocket_rows.append(pocket)
        keep = ["candidate_uid", "pocket_id", "reliable"] + [c for c in ["posebusters_pass"] if c in pred] + score_cols
        part = pred[keep].copy()
        part["seed"] = seed
        prediction_parts.append(part)

    metrics_df = pd.DataFrame(metric_rows)
    top_df = pd.concat(top_rows, ignore_index=True)
    pockets_df = pd.concat(pocket_rows, ignore_index=True)
    preds_df = pd.concat(prediction_parts, ignore_index=True)
    metrics_df.to_csv(output_dir / "main_metrics_by_seed.csv", index=False)
    top_df.to_csv(output_dir / "main_top10_by_seed.csv", index=False)
    pockets_df.to_csv(output_dir / "main_per_pocket_top10.csv", index=False)
    preds_df.to_csv(output_dir / "main_oof_predictions.csv", index=False)
    bootstrap_deltas(
        pockets_df.groupby(["pocket_id", "method"], as_index=False)["reliability"].mean(),
        ["label_informed_rule", "smooth_pose_rule", "logreg_full", "logreg_direct_excluded"],
        seed=20260630,
        n_boot=args.bootstrap,
    ).to_csv(output_dir / "main_paired_bootstrap.csv", index=False)

    # Cross-family prediction of each label component.
    failure_rows = []
    for target, cols in FAILURE_TARGETS.items():
        if target not in frame or frame[target].nunique() < 2:
            continue
        for seed in args.seeds:
            pred_score = pd.Series(np.nan, index=frame.index, dtype=float)
            for fold_idx, heldout in enumerate(fold_pockets(frame["pocket_id"].unique(), args.folds, seed), start=1):
                test_mask = frame["pocket_id"].isin(set(heldout))
                train, test = frame.loc[~test_mask].copy(), frame.loc[test_mask].copy()
                train["reliable"] = train[target].astype(int)
                test["reliable"] = test[target].astype(int)
                scores = fit_scores(train, test, cols, seed + 500 + fold_idx, args.jobs, f"cross_family_{target}", include_iso=False)
                pred_score.loc[test.index] = scores[f"rf_cross_family_{target}"]
            m = safe_metrics(frame[target].astype(int).to_numpy(), pred_score.to_numpy())
            m.update({"target": target, "seed": seed, "n_features": len(cols), "protocol": "same_family_excluded"})
            failure_rows.append(m)
    pd.DataFrame(failure_rows).to_csv(output_dir / "failure_cross_family_metrics.csv", index=False)


def target_split_specs(frame: pd.DataFrame, seed: int, folds: int):
    pockets = frame["pocket_id"].drop_duplicates().to_numpy()
    fold_list = fold_pockets(pockets, folds, seed)
    for fold_idx, heldout_pockets in enumerate(fold_list, start=1):
        test_mask = frame["pocket_id"].isin(set(heldout_pockets))
        yield "pocket_cv", f"fold_{fold_idx}", ~test_mask, test_mask
    for generator in sorted(frame["generator"].unique()):
        test_mask = frame["generator"].eq(generator)
        yield "leave_generator_out", generator, ~test_mask, test_mask
    for generator in sorted(frame["generator"].unique()):
        for fold_idx, heldout_pockets in enumerate(fold_list, start=1):
            test_mask = frame["generator"].eq(generator) & frame["pocket_id"].isin(set(heldout_pockets))
            train_mask = (~frame["generator"].eq(generator)) & (~frame["pocket_id"].isin(set(heldout_pockets)))
            yield "double_holdout", f"{generator}:fold_{fold_idx}", train_mask, test_mask


def run_target(args: argparse.Namespace, output_dir: Path) -> None:
    frame = add_fixed_scores(pd.read_csv(args.target_frame))
    metric_rows, top_rows = [], []
    for seed in args.seeds:
        for split_kind, split_name, train_mask, test_mask in target_split_specs(frame, seed, args.folds):
            train, test = frame.loc[train_mask].copy(), frame.loc[test_mask].copy()
            if len(train) == 0 or len(test) == 0:
                continue
            pred = test.copy()
            scores = fit_scores(train, test, FULL_FEATURES, seed + abs(hash((split_kind, split_name))) % 100000, args.jobs, "full")
            scores.update(fit_scores(train, test, DIRECT_EXCLUDED_FEATURES, seed + 100000 + abs(hash((split_kind, split_name))) % 100000, args.jobs, "direct_excluded", include_iso=False))
            for name, values in scores.items():
                pred[f"score_{name}"] = values
            score_cols = ["score_qed_sa", "score_smooth_pose_rule"]
            if "score_label_informed_rule" in pred:
                score_cols.append("score_label_informed_rule")
            for col in ["score_vina_minimize", "score_qed_sa_vina"]:
                if col in pred:
                    score_cols.append(col)
            score_cols += [f"score_{name}" for name in scores]
            for score_col in score_cols:
                values = pred[score_col].to_numpy(float)
                metric = safe_metrics(pred["reliable"].to_numpy(int), rank01(values) if score_col in score_cols[:5] else values)
                metric.update({
                    "seed": seed,
                    "split_kind": split_kind,
                    "split_name": split_name,
                    "method": score_name(score_col),
                    "n_train": len(train),
                    "n_test": len(test),
                    "test_reliability": float(test["reliable"].mean()),
                })
                metric_rows.append(metric)
            group_cols = ["generator", "pocket_id"] if split_kind == "pocket_cv" else ["pocket_id"]
            top = topk_rows(pred, score_cols, group_cols, 0.1)
            top["seed"] = seed
            top["split_kind"] = split_kind
            top["split_name"] = split_name
            top["n_train"] = len(train)
            top["n_test"] = len(test)
            top_rows.append(top)
    pd.DataFrame(metric_rows).to_csv(output_dir / "shift_metrics_all.csv", index=False)
    pd.concat(top_rows, ignore_index=True).to_csv(output_dir / "shift_top10_all.csv", index=False)


def summarize(output_dir: Path) -> None:
    summary: dict[str, object] = {}
    for name in ["main_metrics_by_seed", "main_top10_by_seed", "failure_cross_family_metrics", "shift_metrics_all", "shift_top10_all"]:
        path = output_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        group_cols = [c for c in ["split_kind", "split_name", "method", "target", "protocol", "coverage"] if c in df.columns]
        value_cols = [c for c in ["roc_auc", "pr_auc", "brier", "reliability", "n_failures", "posebusters_pass", "pb_roc_auc"] if c in df.columns]
        if group_cols and value_cols:
            agg = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
            agg.columns = ["_".join(str(x) for x in col if str(x)) for col in agg.columns.to_flat_index()]
            agg.to_csv(output_dir / f"{name}_summary.csv", index=False)
            summary[name] = {"rows": len(df), "groups": len(agg)}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv"))
    p.add_argument("--posebusters", type=Path, default=Path("outputs/diffsbdd_crossdocked_posebusters_molfast_v0/reports/posebusters_candidates.csv"))
    p.add_argument("--target-frame", type=Path, default=Path("outputs/targetdiff_official_multigen_reliamol_v0/reports/feature_frame.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/revision_r1_leakage_shift_audit_v0"))
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--main-only", action="store_true")
    p.add_argument("--target-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocols = pd.DataFrame([
        {"protocol": "full_current", "n_features": len(FULL_FEATURES), "features": ";".join(FULL_FEATURES)},
        {"protocol": "direct_excluded", "n_features": len(DIRECT_EXCLUDED_FEATURES), "features": ";".join(DIRECT_EXCLUDED_FEATURES)},
        {"protocol": "nonlabel_primitive", "n_features": len(NON_LABEL_PRIMITIVE_FEATURES), "features": ";".join(NON_LABEL_PRIMITIVE_FEATURES)},
    ])
    protocols.to_csv(args.output_dir / "feature_access_protocols.csv", index=False)
    if not args.target_only:
        run_main(args, args.output_dir)
    if not args.main_only:
        run_target(args, args.output_dir)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
