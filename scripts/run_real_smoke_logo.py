from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from reliamol3d.metrics import FAILURE_COLUMNS, binary_metrics, failure_metrics, reranking_metrics
from reliamol3d.torch_model import train_reliability_net
from run_real_smoke_crossdocked import feature_columns, rank_to_prob


def fit_predict_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cfg: dict,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy(np.float32))
    x_test = scaler.transform(test[features].to_numpy(np.float32))
    y_train = train["reliable"].to_numpy(np.float32)
    f_train = train[FAILURE_COLUMNS].to_numpy(np.float32)
    return train_reliability_net(x_train, y_train, f_train, x_test, cfg["model"], device, seed)


def run_heldout(frame: pd.DataFrame, cfg: dict, heldout: str, output_dir: Path, test_generators: list[str] | None = None) -> None:
    train = frame[(frame["split"].isin(["train", "val"])) & (~frame["generator"].eq(heldout))].copy()
    if test_generators is None:
        test_generators = [heldout]
    test = frame[(frame["split"].eq("test")) & (frame["generator"].isin(test_generators))].copy()
    if train.empty or test.empty:
        raise SystemExit(f"Empty train/test for heldout={heldout}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_scores: dict[str, np.ndarray] = {
        "vina_rank": rank_to_prob(-test["vina_score"].to_numpy()),
        "gnina_rank": rank_to_prob(-test["gnina_score"].to_numpy()),
        "qed_sa_vina": rank_to_prob(-test["vina_score"].to_numpy() + 0.5 * test["qed"].to_numpy() - 0.3 * test["sa"].to_numpy()),
    }
    feature_variants = ["score_only", "ligand_geometry", "interaction_shape", "reliamol_noleak", "leak_upper_bound"]
    for variant in feature_variants:
        rel_prob, fail_prob = fit_predict_mlp(
            train,
            test,
            feature_columns(frame, variant),
            cfg,
            cfg["experiment"]["seed"] + len(variant) + len(heldout),
            device,
        )
        variant_scores[f"mlp_{variant}"] = rel_prob
        test[f"score_mlp_{variant}"] = rel_prob
        if variant == "reliamol_noleak":
            for i, label in enumerate(FAILURE_COLUMNS):
                test[f"{label}_prob"] = fail_prob[:, i]
            pd.DataFrame([failure_metrics(test, [f"{c}_prob" for c in FAILURE_COLUMNS])]).to_csv(output_dir / "failure_diagnosis.csv", index=False)

    lite_features = feature_columns(frame, "reliamol_noleak")
    rf = RandomForestClassifier(
        n_estimators=260,
        min_samples_leaf=4,
        n_jobs=16,
        random_state=cfg["experiment"]["seed"],
        class_weight="balanced",
    )
    rf.fit(train[lite_features], train["reliable"])
    variant_scores["rf_noleak"] = rf.predict_proba(test[lite_features])[:, 1]
    hgb = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, l2_regularization=0.03, random_state=cfg["experiment"]["seed"])
    hgb.fit(train[lite_features], train["reliable"])
    variant_scores["hgb_noleak"] = hgb.predict_proba(test[lite_features])[:, 1]

    metrics = []
    for method, prob in variant_scores.items():
        row = binary_metrics(test["reliable"].to_numpy(), prob)
        row["method"] = method
        row["heldout_generator"] = heldout
        row["n_train"] = len(train)
        row["n_test"] = len(test)
        row["device"] = device
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)

    rerank_frames = []
    for score_col in [
        "score_mlp_score_only",
        "score_mlp_ligand_geometry",
        "score_mlp_interaction_shape",
        "score_mlp_reliamol_noleak",
        "score_mlp_leak_upper_bound",
        "vina_score",
        "qed",
    ]:
        work = test.copy()
        use_col = score_col
        if score_col == "vina_score":
            work["vina_rank_score"] = -work["vina_score"]
            use_col = "vina_rank_score"
        rr = reranking_metrics(work, use_col, cfg["evaluation"]["topk_fracs"])
        rr["heldout_generator"] = heldout
        rerank_frames.append(rr)
    pd.concat(rerank_frames, ignore_index=True).to_csv(output_dir / "reranking.csv", index=False)
    test.to_csv(output_dir / "test_predictions.csv", index=False)

    print(pd.read_csv(output_dir / "metrics.csv").to_string(index=False))
    print(pd.read_csv(output_dir / "reranking.csv").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_smoke_crossdocked.yaml")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--heldout-generator", required=True)
    parser.add_argument("--mixed-native-anchor", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    candidates = args.candidates or str(Path(cfg["experiment"]["output_dir"]) / "candidate_labels.csv")
    output_dir = Path(args.output_dir or Path(cfg["experiment"]["output_dir"]) / "logo" / args.heldout_generator)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(candidates)
    if args.mixed_native_anchor and args.heldout_generator != "NativeNear":
        train_mask = (frame["split"].isin(["train", "val"])) & (~frame["generator"].eq(args.heldout_generator))
        test_mask = frame["split"].eq("test") & frame["generator"].isin(["NativeNear", args.heldout_generator])
        mixed = frame.copy()
        mixed.loc[:, "split"] = "unused"
        mixed.loc[train_mask, "split"] = "train"
        mixed.loc[test_mask, "split"] = "test"
        mixed = mixed[mixed["split"].isin(["train", "test"])].copy()
        run_heldout(mixed, cfg, args.heldout_generator, output_dir, test_generators=["NativeNear", args.heldout_generator])
    else:
        run_heldout(frame, cfg, args.heldout_generator, output_dir)


if __name__ == "__main__":
    main()
