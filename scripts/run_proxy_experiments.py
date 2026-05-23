from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reliamol3d.metrics import FAILURE_COLUMNS, binary_metrics, failure_metrics, reranking_metrics
from reliamol3d.proxy_data import feature_columns, make_proxy_dataset
from reliamol3d.torch_model import train_reliability_net


def split_frame(frame: pd.DataFrame, mode: str, seed: int, heldout_generator: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        idx = np.arange(len(frame))
        return train_test_split(idx, test_size=0.25, random_state=seed, stratify=frame["reliable"])
    if mode == "unseen_family":
        families = np.array(sorted(frame["target_family"].unique()))
        rng = np.random.default_rng(seed)
        heldout = set(rng.choice(families, size=max(1, len(families) // 5), replace=False).tolist())
        test = frame.index[frame["target_family"].isin(heldout)].to_numpy()
        train = frame.index[~frame["target_family"].isin(heldout)].to_numpy()
        return train, test
    if mode == "unseen_generator":
        if heldout_generator is None:
            raise ValueError("heldout_generator is required for unseen_generator")
        test = frame.index[frame["generator"].eq(heldout_generator)].to_numpy()
        train = frame.index[~frame["generator"].eq(heldout_generator)].to_numpy()
        return train, test
    raise ValueError(f"unknown split mode: {mode}")


def add_baseline_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    outputs["vina_rank"] = rank_to_prob(-test["vina_score"].to_numpy())
    outputs["gnina_rank"] = rank_to_prob(-test["gnina_score"].to_numpy())
    outputs["qed_sa_vina"] = rank_to_prob(
        -test["vina_score"].to_numpy() + 0.6 * test["qed"].to_numpy() - 0.35 * test["sa"].to_numpy()
    )

    n_jobs = max(1, int(os.environ.get("RELIAMOL_N_JOBS", "8")))
    rf = RandomForestClassifier(n_estimators=220, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed, class_weight="balanced")
    rf.fit(train[features], train["reliable"])
    outputs["rf_handcrafted"] = rf.predict_proba(test[features])[:, 1]

    hgb = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, l2_regularization=0.02, random_state=seed)
    hgb.fit(train[features], train["reliable"])
    outputs["hgb_handcrafted"] = hgb.predict_proba(test[features])[:, 1]
    return outputs


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = score.argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def run_one(config: dict, seed: int, split_mode: str, heldout_generator: str | None, output_dir: Path) -> dict[str, float | str | int]:
    data_cfg = config["data"]
    frame = make_proxy_dataset(
        data_cfg["n_pockets"],
        data_cfg["candidates_per_pocket"],
        data_cfg["generators"],
        data_cfg["feature_dim"],
        data_cfg["target_family_count"],
        config["experiment"]["seed"] + seed,
    )
    features = feature_columns(frame)
    train_idx, test_idx = split_frame(frame, split_mode, seed, heldout_generator)
    train = frame.loc[train_idx].copy()
    test = frame.loc[test_idx].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy(np.float32))
    x_test = scaler.transform(test[features].to_numpy(np.float32))
    y_train = train["reliable"].to_numpy(np.float32)
    f_train = train[FAILURE_COLUMNS].to_numpy(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel_prob, fail_prob = train_reliability_net(x_train, y_train, f_train, x_test, config["model"], device, seed)
    test["reliamol_score"] = rel_prob
    for i, label in enumerate(FAILURE_COLUMNS):
        test[f"{label}_prob"] = fail_prob[:, i]

    rows = []
    for name, prob in {"reliamol": rel_prob, **add_baseline_scores(train, test, features, seed)}.items():
        metrics = binary_metrics(test["reliable"].to_numpy(), prob, prefix="")
        metrics.update({"method": name, "split": split_mode, "seed": seed, "heldout_generator": heldout_generator or "NA"})
        rows.append(metrics)

    main = pd.DataFrame(rows)
    fail = failure_metrics(test, [f"{c}_prob" for c in FAILURE_COLUMNS])
    rerank_frames = []
    for score_col in ["reliamol_score", "vina_score", "qed"]:
        work = test.copy()
        if score_col == "vina_score":
            work["vina_rank_score"] = -work["vina_score"]
            use_col = "vina_rank_score"
        else:
            use_col = score_col
        rerank_frames.append(reranking_metrics(work, use_col, config["evaluation"]["topk_fracs"]))
    rerank = pd.concat(rerank_frames, ignore_index=True)

    tag = f"{split_mode}_{heldout_generator or 'all'}_seed{seed}"
    main.to_csv(output_dir / f"metrics_{tag}.csv", index=False)
    pd.DataFrame([fail]).to_csv(output_dir / f"failure_diagnosis_{tag}.csv", index=False)
    rerank.to_csv(output_dir / f"reranking_{tag}.csv", index=False)

    rel_ap = average_precision_score(test["reliable"], rel_prob)
    return {
        "split": split_mode,
        "seed": seed,
        "heldout_generator": heldout_generator or "NA",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate_test": float(test["reliable"].mean()),
        "reliamol_pr_auc": float(rel_ap),
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proxy.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["random", "unseen_family", "unseen_generator"], default=None)
    parser.add_argument("--heldout-generator", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else config["evaluation"]["seeds"]
    plan: list[tuple[str, str | None]] = []
    if args.split:
        plan = [(args.split, args.heldout_generator)]
    else:
        plan.append(("random", None))
        plan.append(("unseen_family", None))
        for generator in config["data"]["generators"]:
            plan.append(("unseen_generator", generator))

    summaries = []
    for seed in seeds:
        for split, heldout in plan:
            summaries.append(run_one(config, seed, split, heldout, output_dir))

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
