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
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402


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
] + [f"atom_{atom}_frac" for atom in ATOM_FEATURES]

INTERNAL_GEOMETRY_FEATURES = [
    "pose_atoms",
    "pose_radius_shape",
    "nn_q05",
    "nn_q10",
    "nn_q25",
    "nn_q50",
    "nn_q75",
    "nn_q90",
]

POCKET_GEOMETRY_FEATURES = [
    "lp_q05",
    "lp_q10",
    "lp_q25",
    "lp_q50",
    "lp_q75",
    "lp_q90",
    "shell_0_2",
    "shell_2_4",
    "shell_4_6",
    "shell_6_8",
    "shell_gt8",
]

FEATURE_SETS: dict[str, tuple[str, list[str]]] = {
    "rf_ligand_descriptor": ("Ligand descriptors only", LIGAND_FEATURES),
    "rf_internal_geometry": ("Internal 3D geometry only", INTERNAL_GEOMETRY_FEATURES),
    "rf_pocket_geometry": ("Ligand-pocket geometry only", POCKET_GEOMETRY_FEATURES),
    "rf_geometry_only": ("Internal + pocket geometry", INTERNAL_GEOMETRY_FEATURES + POCKET_GEOMETRY_FEATURES),
    "rf_ligand_internal": ("Ligand + internal geometry", LIGAND_FEATURES + INTERNAL_GEOMETRY_FEATURES),
    "rf_ligand_pocket": ("Ligand + pocket geometry", LIGAND_FEATURES + POCKET_GEOMETRY_FEATURES),
    "rf_full_reliamol_novinascore": (
        "Full ReliaMol without Vina",
        LIGAND_FEATURES + INTERNAL_GEOMETRY_FEATURES + POCKET_GEOMETRY_FEATURES,
    ),
}


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def clean_matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted(set(cols) - set(train.columns))
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    tr = train[cols].replace([np.inf, -np.inf], np.nan)
    te = test[cols].replace([np.inf, -np.inf], np.nan)
    med = tr.median(numeric_only=True).fillna(0.0)
    return tr.fillna(med).to_numpy(np.float32), te.fillna(med).to_numpy(np.float32)


def fold_pockets(pockets: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(pockets).copy()
    rng.shuffle(shuffled)
    n_folds = min(max(2, n_folds), len(shuffled))
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold)]


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    order = arr.argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def reranking_table(frame: pd.DataFrame, score_cols: dict[str, str], topk_fracs: list[float]) -> pd.DataFrame:
    rows = []
    for method, score_col in score_cols.items():
        for frac in topk_fracs:
            selected = []
            for _, group in frame.groupby("pocket_id", sort=False):
                k = max(1, int(math.ceil(len(group) * frac)))
                selected.append(group.nlargest(k, score_col))
            top = pd.concat(selected, ignore_index=True)
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


def prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "reliable" not in frame.columns:
        frame["reliable"] = frame["public_generated_reliable_proxy"].astype(int)
    for target, source in [
        ("chemical_failure", "chemical_failure_proxy"),
        ("geometric_failure", "geometric_failure_proxy"),
        ("pocket_failure", "pocket_failure_proxy"),
        ("clash_failure", "clash_failure_proxy"),
        ("synthetic_failure", "synthetic_failure_proxy"),
    ]:
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source].astype(bool)
    if "chemotype_id" not in frame.columns:
        frame["chemotype_id"] = [deterministic_id(f"{pid}:{idx}") % 100000 for pid, idx in zip(frame["pocket_id"], frame["mol_idx"])]
    frame["score_qed"] = frame["qed"].astype(float)
    frame["score_qed_sa"] = frame["qed"].astype(float) - 0.25 * frame["sa_proxy"].astype(float)
    return frame


def run_seed(frame: pd.DataFrame, seed: int, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pockets = frame["pocket_id"].drop_duplicates().to_numpy()
    folds = fold_pockets(pockets, args.n_folds, seed)
    pred = frame.copy()
    for method in FEATURE_SETS:
        pred[f"score_{method}"] = np.nan

    for fold_idx, heldout in enumerate(folds):
        test_mask = pred["pocket_id"].isin(set(heldout))
        train = pred.loc[~test_mask].copy()
        test = pred.loc[test_mask].copy()
        print(
            f"seed={seed} fold={fold_idx + 1}/{len(folds)} train={len(train)} test={len(test)} heldout_pockets={len(heldout)}",
            flush=True,
        )
        for method, (_, features) in FEATURE_SETS.items():
            x_train, x_test = clean_matrix(train, test, features)
            rf = RandomForestClassifier(
                n_estimators=args.rf_trees,
                min_samples_leaf=args.min_samples_leaf,
                n_jobs=args.rf_jobs,
                random_state=seed + fold_idx + deterministic_id(method),
                class_weight="balanced",
            )
            rf.fit(x_train, train["reliable"])
            pred.loc[test.index, f"score_{method}"] = rf.predict_proba(x_test)[:, 1]

    score_cols = {"qed": "score_qed", "qed_sa": "score_qed_sa"}
    score_cols.update({method: f"score_{method}" for method in FEATURE_SETS})

    metric_rows = []
    for method, col in score_cols.items():
        score = pred[col].to_numpy(float)
        prob = rank_to_prob(score) if method in {"qed", "qed_sa"} else score
        row = binary_metrics(pred["reliable"].to_numpy(), prob)
        row.update({"seed": seed, "method": method, "n_folds": len(folds)})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    rerank = reranking_table(pred, score_cols, args.topk_fracs)
    rerank["seed"] = seed

    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(seed_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(seed_dir / "metrics.csv", index=False)
    rerank.to_csv(seed_dir / "reranking.csv", index=False)
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

    feature_rows = [
        {
            "method": method,
            "feature_group": label,
            "n_features": len(features),
            "uses_vina_or_gnina": False,
            "uses_failure_label_columns": False,
            "uses_generator_identity": False,
        }
        for method, (label, features) in FEATURE_SETS.items()
    ]
    feature_rows.insert(
        0,
        {
            "method": "qed",
            "feature_group": "QED scalar baseline",
            "n_features": 1,
            "uses_vina_or_gnina": False,
            "uses_failure_label_columns": False,
            "uses_generator_identity": False,
        },
    )
    feature_rows.insert(
        1,
        {
            "method": "qed_sa",
            "feature_group": "QED and SA scalar baseline",
            "n_features": 2,
            "uses_vina_or_gnina": False,
            "uses_failure_label_columns": False,
            "uses_generator_identity": False,
        },
    )
    feature_map = pd.DataFrame(feature_rows)
    table = feature_map.merge(metric_summary, on="method", how="left")
    table = table.merge(rerank_summary[rerank_summary["topk_frac"].eq(0.1)], on="method", how="left")
    baseline = float(table.loc[table["method"].eq("qed_sa"), "reliable_proxy_rate_mean"].iloc[0])
    table["delta_top10_reliable_vs_qed_sa"] = table["reliable_proxy_rate_mean"] - baseline

    metric_summary.to_csv(output_dir / "ablation_metric_summary.csv", index=False)
    rerank_summary.to_csv(output_dir / "ablation_reranking_summary.csv", index=False)
    feature_map.to_csv(output_dir / "ablation_feature_groups.csv", index=False)
    table.to_csv(output_dir / "ablation_table_top10.csv", index=False)
    (output_dir / "ablation_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_note": "Feature-group ablation uses the existing DiffSBDD exact-CrossDocked feature frame. Learned variants use RF with identical pocket-heldout splits. No Vina/GNINA score, direct failure-label column, or generator identity is used.",
                "feature_sets": {method: {"label": label, "columns": features} for method, (label, features) in FEATURE_SETS.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-group ablation for DiffSBDD exact-CrossDocked ReliaMol reranking.")
    parser.add_argument("--feature-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol/reports/feature_frame.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_feature_ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--rf-trees", type=int, default=260)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--rf-jobs", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(args.feature_frame)
    print(f"feature_frame candidates={len(frame)} pockets={frame['pocket_id'].nunique()} reliable_rate={frame['reliable'].mean():.4f}", flush=True)
    metrics_list = []
    rerank_list = []
    for seed in args.seeds:
        _, metrics, rerank = run_seed(frame, seed, args, report_dir)
        metrics_list.append(metrics)
        rerank_list.append(rerank)
    metrics_all = pd.concat(metrics_list, ignore_index=True)
    rerank_all = pd.concat(rerank_list, ignore_index=True)
    metrics_all.to_csv(report_dir / "ablation_metrics_by_seed.csv", index=False)
    rerank_all.to_csv(report_dir / "ablation_reranking_by_seed.csv", index=False)
    summarize(metrics_all, rerank_all, report_dir)


if __name__ == "__main__":
    main()
