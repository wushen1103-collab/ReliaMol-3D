from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))
from reliamol3d.metrics import binary_metrics  # noqa: E402
from run_diffsbdd_crossdocked_reliamol import clean_matrix, feature_columns, fold_pockets, reranking_table  # noqa: E402


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = np.asarray(score).argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def fit_logreg(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=1)
    clf.fit(x_train, train["reliable"])
    return clf.predict_proba(x_test)[:, 1]


def fit_ridge_rank(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = RidgeClassifier(class_weight="balanced", random_state=seed)
    clf.fit(x_train, train["reliable"])
    return rank_to_prob(clf.decision_function(x_test))


def fit_isoforest(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, jobs: int) -> np.ndarray:
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    contamination = float(np.clip(1.0 - train["reliable"].mean(), 0.01, 0.25))
    iso = IsolationForest(n_estimators=400, contamination=contamination, random_state=seed, n_jobs=jobs)
    iso.fit(x_train)
    # Higher score_samples means less anomalous, hence more likely reliable.
    return rank_to_prob(iso.score_samples(x_test))


def add_rule_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["score_rule_qed_sa_clash"] = (
        out["qed"].astype(float)
        - 0.25 * out["sa_proxy"].astype(float)
        - 1.5 * out["receptor_clash_atoms_1p2"].astype(float).clip(0, 5)
        + 0.05 * out["receptor_contact_atoms_4p5"].astype(float).clip(0, 20)
    )
    out["score_rule_pose_band"] = (
        -np.abs(out["lp_q10"].astype(float) - 2.2)
        -0.5 * out["shell_0_2"].astype(float)
        +0.5 * out["shell_2_4"].astype(float)
        -0.2 * out["shell_gt8"].astype(float)
    )
    out["score_rule_audit_upper_bound"] = out["reliable"].astype(float)
    return out


def run_seed(frame: pd.DataFrame, seed: int, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pockets = frame["pocket_id"].drop_duplicates().to_numpy()
    folds = fold_pockets(pockets, args.n_folds, seed)
    pred = add_rule_scores(frame.copy())
    methods = {
        "logreg_ligand_descriptor": ("logreg", "ligand_descriptor"),
        "logreg_pose_shape": ("logreg", "pose_shape"),
        "logreg_reliamol_novinascore": ("logreg", "reliamol_novinascore"),
        "ridge_pose_shape": ("ridge", "pose_shape"),
        "isoforest_pose_shape": ("isoforest", "pose_shape"),
        "isoforest_reliamol_novinascore": ("isoforest", "reliamol_novinascore"),
    }
    for method in methods:
        pred[f"score_{method}"] = np.nan
    for fold_idx, heldout in enumerate(folds):
        test_mask = pred["pocket_id"].isin(set(heldout))
        train = pred.loc[~test_mask].copy()
        test = pred.loc[test_mask].copy()
        print(f"seed={seed} fold={fold_idx + 1}/{len(folds)} extra-baselines train={len(train)} test={len(test)}", flush=True)
        for method, (kind, variant) in methods.items():
            features = feature_columns(variant)
            run_seed_i = seed + deterministic_id(f"{fold_idx}:{method}")
            if kind == "logreg":
                score = fit_logreg(train, test, features, run_seed_i)
            elif kind == "ridge":
                score = fit_ridge_rank(train, test, features, run_seed_i)
            elif kind == "isoforest":
                score = fit_isoforest(train, test, features, run_seed_i, args.jobs)
            else:
                raise ValueError(method)
            pred.loc[test.index, f"score_{method}"] = score
    score_cols = {
        "rule_qed_sa_clash": "score_rule_qed_sa_clash",
        "rule_pose_band": "score_rule_pose_band",
        "rule_audit_upper_bound": "score_rule_audit_upper_bound",
        **{method: f"score_{method}" for method in methods},
    }
    metric_rows = []
    for method, col in score_cols.items():
        score = pred[col].to_numpy(float)
        prob = rank_to_prob(score) if method.startswith("rule_") else score
        row = binary_metrics(pred["reliable"].to_numpy(), prob)
        row.update({"seed": seed, "method": method})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    rerank = reranking_table(pred, score_cols, args.topk_fracs)
    rerank["seed"] = seed
    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(seed_dir / "extra_baseline_oof_predictions.csv", index=False)
    metrics.to_csv(seed_dir / "extra_baseline_metrics.csv", index=False)
    rerank.to_csv(seed_dir / "extra_baseline_reranking.csv", index=False)
    return pred, metrics, rerank


def summarize(metrics: pd.DataFrame, rerank: pd.DataFrame, output_dir: Path) -> None:
    metric_summary = metrics.groupby("method", sort=False)[["roc_auc", "pr_auc", "f1", "mcc", "ece"]].agg(["mean", "std"]).reset_index()
    metric_summary.columns = ["_".join([str(x) for x in col if x]) for col in metric_summary.columns.to_flat_index()]
    rerank_summary = (
        rerank.groupby(["method", "topk_frac"], sort=False)[
            ["reliable_proxy_rate", "chemical_pass_rate", "geometry_pass_rate", "pocket_pass_rate", "clash_free_rate", "synthetic_pass_rate", "diversity_proxy"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    rerank_summary.columns = ["_".join([str(x) for x in col if x]) for col in rerank_summary.columns.to_flat_index()]
    metric_summary.to_csv(output_dir / "extra_baseline_metric_summary.csv", index=False)
    rerank_summary.to_csv(output_dir / "extra_baseline_reranking_summary.csv", index=False)
    report_lines = [
        "# DiffSBDD CrossDocked Extra Baselines",
        "",
        "Additional reviewer-facing baselines: linear supervised models, unsupervised anomaly detection, simple geometry rules, and an audit-label upper bound.",
        "",
        "## Metrics",
        "",
        metric_summary.to_csv(index=False),
        "## Top-10%",
        "",
        rerank_summary[rerank_summary["topk_frac"].eq(0.1)].to_csv(index=False),
    ]
    (output_dir / "extra_baseline_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(metric_summary.to_string(index=False))
    print(rerank_summary[rerank_summary["topk_frac"].eq(0.1)].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra baselines for DiffSBDD CrossDocked ReliaMol reranking.")
    parser.add_argument("--feature-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol/reports/feature_frame.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_extra_baselines"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--jobs", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_frame)
    metrics_all = []
    rerank_all = []
    for seed in args.seeds:
        _, metrics, rerank = run_seed(frame, seed, args, report_dir)
        metrics_all.append(metrics)
        rerank_all.append(rerank)
    metrics = pd.concat(metrics_all, ignore_index=True)
    rerank = pd.concat(rerank_all, ignore_index=True)
    metrics.to_csv(report_dir / "extra_baseline_metrics_by_seed.csv", index=False)
    rerank.to_csv(report_dir / "extra_baseline_reranking_by_seed.csv", index=False)
    summarize(metrics, rerank, report_dir)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_frame": str(args.feature_frame),
                "n_candidates": int(len(frame)),
                "n_pockets": int(frame["pocket_id"].nunique()),
                "note": "rule_audit_upper_bound intentionally uses the audit label as an upper-bound sanity check and should not be positioned as a deployable baseline.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
