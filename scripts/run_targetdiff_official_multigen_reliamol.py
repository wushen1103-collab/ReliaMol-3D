from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402
from reliamol3d.torch_model import train_reliability_net  # noqa: E402


ATOM_FEATURES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "metal"]
FAILURE_COLUMNS = ["chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure"]

LIGAND_FEATURES = [
    "mol_wt",
    "logp",
    "tpsa",
    "qed",
    "rot_bonds",
    "ring_count",
    "hetero_atoms",
    "abs_formal_charge",
    "rare_element_count",
    "heavy_atoms",
    "sa_proxy",
    "descriptor_ok",
] + [f"atom_{a}_frac" for a in ATOM_FEATURES]

POSE_SHAPE_BASE = [
    "pose_atoms",
    "pose_radius_shape",
    "lp_q05",
    "lp_q10",
    "lp_q25",
    "lp_q50",
    "lp_q75",
    "lp_q90",
    "nn_q05",
    "nn_q10",
    "nn_q25",
    "nn_q50",
    "nn_q75",
    "nn_q90",
    "shell_0_2",
    "shell_2_4",
    "shell_4_6",
    "shell_6_8",
    "shell_gt8",
]


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def feature_columns(variant: str) -> list[str]:
    variants = {
        "ligand_descriptor": LIGAND_FEATURES,
        "pose_shape": POSE_SHAPE_BASE + [f"atom_{a}_frac" for a in ATOM_FEATURES],
        "reliamol_novinascore": LIGAND_FEATURES + POSE_SHAPE_BASE,
    }
    if variant not in variants:
        raise ValueError(f"Unknown feature variant: {variant}")
    return variants[variant]


def clean_matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = train[cols].replace([np.inf, -np.inf], np.nan)
    te = test[cols].replace([np.inf, -np.inf], np.nan)
    med = tr.median(numeric_only=True).fillna(0.0)
    return tr.fillna(med).to_numpy(np.float32), te.fillna(med).to_numpy(np.float32)


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(len(arr), 0.5, dtype=float)
    fill = np.nanmedian(arr[finite])
    arr = np.where(finite, arr, fill)
    order = arr.argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def model_cfg(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "focal_gamma": 1.5,
        "failure_loss_weight": 0.25,
        "calibration_loss_weight": 0.05,
    }


def fit_mlp(train: pd.DataFrame, test: pd.DataFrame, features: list[str], cfg: dict, seed: int, device: str) -> np.ndarray:
    if train["reliable"].nunique() < 2:
        return np.full(len(test), float(train["reliable"].mean()), dtype=float)
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    rel_prob, _ = train_reliability_net(
        x_train,
        train["reliable"].to_numpy(np.float32),
        train[FAILURE_COLUMNS].to_numpy(np.float32),
        x_test,
        cfg,
        device,
        seed,
    )
    return rel_prob


def build_frame(args: argparse.Namespace) -> pd.DataFrame:
    raw = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    if args.generators:
        raw = raw[raw["generator"].isin(set(args.generators))].copy()
    raw["descriptor_ok"] = raw["sanitize_ok"].astype(float)
    raw["reliable"] = raw["public_generated_reliable_proxy"].astype(int)
    raw["chemical_failure"] = raw["chemical_failure_proxy"].astype(bool)
    raw["geometric_failure"] = raw["geometric_failure_proxy"].astype(bool)
    raw["pocket_failure"] = raw["pocket_failure_proxy"].astype(bool)
    raw["clash_failure"] = raw["clash_failure_proxy"].astype(bool)
    raw["synthetic_failure"] = raw["synthetic_failure_proxy"].astype(bool)
    raw["chemotype_id"] = [deterministic_id(str(x)) % 100000 for x in raw["smiles"].fillna(raw["candidate_uid"])]
    raw["score_qed"] = raw["qed"].astype(float)
    raw["score_qed_sa"] = raw["qed"].astype(float) - 0.25 * raw["sa_proxy"].astype(float)
    raw["score_vina_score_only"] = -raw["vina_score_only_affinity"].astype(float)
    raw["score_vina_minimize"] = -raw["vina_minimize_affinity"].astype(float)
    raw["score_qed_sa_vina"] = raw["score_qed_sa"].fillna(raw["score_qed_sa"].median()) + 0.08 * raw["score_vina_minimize"].fillna(raw["score_vina_minimize"].median())
    return raw.reset_index(drop=True)


def fold_pockets(pockets: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(pockets).copy()
    rng.shuffle(shuffled)
    n_folds = min(max(2, n_folds), len(shuffled))
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold)]


def split_specs(frame: pd.DataFrame, seed: int, args: argparse.Namespace) -> list[tuple[str, str, pd.Series, pd.Series]]:
    specs: list[tuple[str, str, pd.Series, pd.Series]] = []
    if args.run_pocket_cv:
        for fold_idx, heldout in enumerate(fold_pockets(frame["pocket_id"].drop_duplicates().to_numpy(), args.n_folds, seed)):
            test = frame["pocket_id"].isin(set(heldout))
            specs.append(("pocket_cv", f"fold_{fold_idx + 1}", ~test, test))
    if args.run_leave_generator_out:
        for generator in sorted(frame["generator"].unique()):
            test = frame["generator"].eq(generator)
            specs.append(("leave_generator_out", generator, ~test, test))
    return specs


def reranking_table(frame: pd.DataFrame, score_cols: dict[str, str], topk_fracs: list[float], split_kind: str, split_name: str) -> pd.DataFrame:
    rows = []
    for method, score_col in score_cols.items():
        for frac in topk_fracs:
            selected = []
            group_cols = ["generator", "pocket_id"] if split_kind == "pocket_cv" else ["pocket_id"]
            for _, group in frame.groupby(group_cols, sort=False):
                k = max(1, int(np.ceil(len(group) * frac)))
                selected.append(group.nlargest(k, score_col))
            top = pd.concat(selected, ignore_index=True)
            rows.append(
                {
                    "split_kind": split_kind,
                    "split_name": split_name,
                    "method": method,
                    "topk_frac": frac,
                    "n_selected": int(len(top)),
                    "reliable_proxy_rate": float(top["reliable"].mean()),
                    "chemical_pass_rate": float((~top["chemical_failure"]).mean()),
                    "geometry_pass_rate": float((~top["geometric_failure"]).mean()),
                    "pocket_pass_rate": float((~top["pocket_failure"]).mean()),
                    "clash_free_rate": float((~top["clash_failure"]).mean()),
                    "synthetic_pass_rate": float((~top["synthetic_failure"]).mean()),
                    "diversity_proxy": float(top.groupby("pocket_id")["chemotype_id"].nunique().mean()),
                    "median_vina_minimize_affinity": float(top["vina_minimize_affinity"].median()),
                }
            )
    return pd.DataFrame(rows)


def score_split(train: pd.DataFrame, test: pd.DataFrame, seed: int, args: argparse.Namespace, split_kind: str, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = model_cfg(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = test.copy()
    for name in [
        "mlp_ligand_descriptor",
        "mlp_pose_shape",
        "mlp_reliamol_novinascore",
        "rf_reliamol_novinascore",
        "hgb_reliamol_novinascore",
    ]:
        pred[f"score_{name}"] = np.nan

    print(
        f"seed={seed} split={split_kind}:{split_name} train={len(train)} test={len(test)} "
        f"train_pos={train['reliable'].mean():.4f} test_pos={test['reliable'].mean():.4f} device={device}",
        flush=True,
    )
    for variant in ["ligand_descriptor", "pose_shape", "reliamol_novinascore"]:
        method = f"mlp_{variant}"
        pred[f"score_{method}"] = fit_mlp(train, test, feature_columns(variant), cfg, seed + deterministic_id(f"{split_kind}:{split_name}:{variant}"), device)

    features = feature_columns("reliamol_novinascore")
    x_train, x_test = clean_matrix(train, test, features)
    if train["reliable"].nunique() >= 2:
        rf = RandomForestClassifier(
            n_estimators=args.rf_trees,
            min_samples_leaf=3,
            n_jobs=args.rf_jobs,
            random_state=seed + deterministic_id(f"{split_kind}:{split_name}:rf"),
            class_weight="balanced",
        )
        rf.fit(x_train, train["reliable"])
        pred["score_rf_reliamol_novinascore"] = rf.predict_proba(x_test)[:, 1]
        hgb = HistGradientBoostingClassifier(
            max_iter=args.hgb_iter,
            learning_rate=0.05,
            l2_regularization=0.03,
            random_state=seed + deterministic_id(f"{split_kind}:{split_name}:hgb"),
        )
        hgb.fit(x_train, train["reliable"])
        pred["score_hgb_reliamol_novinascore"] = hgb.predict_proba(x_test)[:, 1]
    else:
        base = float(train["reliable"].mean())
        pred["score_rf_reliamol_novinascore"] = base
        pred["score_hgb_reliamol_novinascore"] = base

    score_cols = {
        "qed": "score_qed",
        "qed_sa": "score_qed_sa",
        "vina_score_only": "score_vina_score_only",
        "vina_minimize": "score_vina_minimize",
        "qed_sa_vina": "score_qed_sa_vina",
        "mlp_ligand_descriptor": "score_mlp_ligand_descriptor",
        "mlp_pose_shape": "score_mlp_pose_shape",
        "mlp_reliamol_novinascore": "score_mlp_reliamol_novinascore",
        "rf_reliamol_novinascore": "score_rf_reliamol_novinascore",
        "hgb_reliamol_novinascore": "score_hgb_reliamol_novinascore",
    }
    metric_rows = []
    for method, col in score_cols.items():
        prob = rank_to_prob(pred[col].to_numpy(float)) if method in {"qed", "qed_sa", "vina_score_only", "vina_minimize", "qed_sa_vina"} else pred[col].to_numpy(float)
        row = binary_metrics(pred["reliable"].to_numpy(), prob)
        row.update({"seed": seed, "split_kind": split_kind, "split_name": split_name, "method": method, "device": device, "n_train": len(train), "n_test": len(pred)})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    rerank = reranking_table(pred, score_cols, args.topk_fracs, split_kind, split_name)
    rerank["seed"] = seed
    return pred, metrics, rerank


def summarize(metrics: pd.DataFrame, rerank: pd.DataFrame, output_dir: Path) -> None:
    metric_summary = (
        metrics.groupby(["split_kind", "method"], sort=False)[["roc_auc", "pr_auc", "f1", "best_f1", "mcc", "ece"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metric_summary.columns = ["_".join([str(x) for x in col if x]) for col in metric_summary.columns.to_flat_index()]
    rerank_summary = (
        rerank.groupby(["split_kind", "method", "topk_frac"], sort=False)[
            [
                "reliable_proxy_rate",
                "chemical_pass_rate",
                "geometry_pass_rate",
                "pocket_pass_rate",
                "clash_free_rate",
                "synthetic_pass_rate",
                "diversity_proxy",
                "median_vina_minimize_affinity",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    rerank_summary.columns = ["_".join([str(x) for x in col if x]) for col in rerank_summary.columns.to_flat_index()]
    logo_summary = (
        rerank.groupby(["split_name", "method", "topk_frac"], sort=False)[
            ["reliable_proxy_rate", "clash_free_rate", "geometry_pass_rate", "synthetic_pass_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    logo_summary.columns = ["_".join([str(x) for x in col if x]) for col in logo_summary.columns.to_flat_index()]
    metric_summary.to_csv(output_dir / "metric_summary.csv", index=False)
    rerank_summary.to_csv(output_dir / "reranking_summary.csv", index=False)
    logo_summary.to_csv(output_dir / "leave_generator_topk_summary.csv", index=False)
    top10 = rerank_summary[rerank_summary["topk_frac"].eq(0.1)].copy()
    lines = [
        "# TargetDiff Official Multi-Generator ReliaMol Experiment",
        "",
        "Official TargetDiff-family sampling-result metadata are evaluated with pocket-heldout and leave-generator-out splits.",
        "",
        "## Metric Summary",
        "",
        metric_summary.to_csv(index=False),
        "## Top-10% Reranking",
        "",
        top10.to_csv(index=False),
        "## Leave-Generator Top-k Head",
        "",
        logo_summary[logo_summary["topk_frac"].eq(0.1)].head(80).to_csv(index=False),
    ]
    (output_dir / "official_multigen_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(metric_summary.to_string(index=False))
    print(top10.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReliaMol multi-generator evaluation on TargetDiff official sampling-result metadata.")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/targetdiff_official_multigen_audit_v0/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/targetdiff_official_multigen_reliamol_v0"))
    parser.add_argument("--generators", nargs="+", default=["TargetDiff", "Pocket2Mol", "CVAE", "AR"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.set_defaults(run_pocket_cv=True, run_leave_generator_out=True)
    parser.add_argument("--run-pocket-cv", dest="run_pocket_cv", action="store_true")
    parser.add_argument("--no-run-pocket-cv", dest="run_pocket_cv", action="store_false")
    parser.add_argument("--run-leave-generator-out", dest="run_leave_generator_out", action="store_true")
    parser.add_argument("--no-run-leave-generator-out", dest="run_leave_generator_out", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--rf-trees", type=int, default=240)
    parser.add_argument("--rf-jobs", type=int, default=20)
    parser.add_argument("--hgb-iter", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = build_frame(args)
    frame.to_csv(report_dir / "feature_frame.csv", index=False)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_dir": str(args.audit_dir),
                "generators": args.generators,
                "n_candidates": int(len(frame)),
                "n_pockets": int(frame["pocket_id"].nunique()),
                "positive_rate": float(frame["reliable"].mean()),
                "splits": {
                    "pocket_cv": bool(args.run_pocket_cv),
                    "leave_generator_out": bool(args.run_leave_generator_out),
                    "n_folds": int(args.n_folds),
                },
                "feature_note": "No direct failure proxy flags, clash/contact counts, or Vina score are used in ReliaMol feature variants; Vina remains a separate baseline.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"feature_frame candidates={len(frame)} generators={frame['generator'].nunique()} "
        f"pockets={frame['pocket_id'].nunique()} reliable_rate={frame['reliable'].mean():.4f}",
        flush=True,
    )
    pred_list = []
    metrics_list = []
    rerank_list = []
    for seed in args.seeds:
        for split_kind, split_name, train_mask, test_mask in split_specs(frame, seed, args):
            train = frame.loc[train_mask].copy()
            test = frame.loc[test_mask].copy()
            pred, metrics, rerank = score_split(train, test, seed, args, split_kind, split_name)
            split_dir = report_dir / "splits" / f"seed_{seed}" / split_kind / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(split_dir / "predictions.csv", index=False)
            metrics.to_csv(split_dir / "metrics.csv", index=False)
            rerank.to_csv(split_dir / "reranking.csv", index=False)
            pred_list.append(pred.assign(seed=seed, split_kind=split_kind, split_name=split_name))
            metrics_list.append(metrics)
            rerank_list.append(rerank)
    predictions = pd.concat(pred_list, ignore_index=True)
    metrics_all = pd.concat(metrics_list, ignore_index=True)
    rerank_all = pd.concat(rerank_list, ignore_index=True)
    predictions.to_csv(report_dir / "predictions_all.csv", index=False)
    metrics_all.to_csv(report_dir / "metrics_all.csv", index=False)
    rerank_all.to_csv(report_dir / "reranking_all.csv", index=False)
    summarize(metrics_all, rerank_all, report_dir)


if __name__ == "__main__":
    main()
