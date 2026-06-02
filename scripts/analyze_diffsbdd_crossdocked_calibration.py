from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


SCORE_COLS = {
    "qed": "score_qed",
    "qed_sa": "score_qed_sa",
    "mlp_ligand_descriptor": "score_mlp_ligand_descriptor",
    "mlp_pose_shape": "score_mlp_pose_shape",
    "mlp_reliamol_novinascore": "score_mlp_reliamol_novinascore",
    "rf_reliamol_novinascore": "score_rf_reliamol_novinascore",
    "hgb_reliamol_novinascore": "score_hgb_reliamol_novinascore",
}


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = np.asarray(score).argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def per_pocket_topk(frame: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(np.ceil(len(group) * frac)))
        selected.append(group.nlargest(k, score_col))
    return pd.concat(selected, ignore_index=True)


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & (p < hi)
        if idx == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": int(mask.sum()),
                "confidence": float(p[mask].mean()),
                "empirical_reliable": float(y[mask].mean()),
                "abs_gap": float(abs(p[mask].mean() - y[mask].mean())),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_topk_delta(frame_a: pd.DataFrame, frame_b: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    pockets = np.asarray(sorted(set(frame_a["pocket_id"]).intersection(set(frame_b["pocket_id"]))))
    by_a = frame_a.groupby("pocket_id")["reliable"].mean()
    by_b = frame_b.groupby("pocket_id")["reliable"].mean()
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(pockets, size=len(pockets), replace=True)
        diffs.append(float(by_a.loc[sample].mean() - by_b.loc[sample].mean()))
    arr = np.asarray(diffs)
    return {
        "delta_mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "p_delta_le_0": float(np.mean(arr <= 0)),
    }


def analyze_seed(pred: pd.DataFrame, seed: int, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = pred["reliable"].astype(int).to_numpy()
    cal_rows = []
    bin_frames = []
    top_frames = {}
    for method, col in SCORE_COLS.items():
        if col not in pred.columns:
            continue
        score = pred[col].to_numpy(float)
        prob = rank_to_prob(score) if method in {"qed", "qed_sa"} else np.clip(score, 0.0, 1.0)
        cal_rows.append(
            {
                "seed": seed,
                "method": method,
                "brier": float(brier_score_loss(y, prob)),
                "mean_prob": float(np.mean(prob)),
                "positive_rate": float(np.mean(y)),
                "high_conf_n": int(np.sum(prob >= args.high_conf_threshold)),
                "high_conf_reliable_rate": float(pred.loc[prob >= args.high_conf_threshold, "reliable"].mean()) if np.any(prob >= args.high_conf_threshold) else np.nan,
            }
        )
        bins = calibration_bins(y, prob, args.bins)
        bins["seed"] = seed
        bins["method"] = method
        bin_frames.append(bins)
        top = per_pocket_topk(pred, col, args.topk_frac)
        top_frames[method] = top
    sig_rows = []
    rng = np.random.default_rng(seed + 43)
    if args.baseline_method in top_frames:
        base = top_frames[args.baseline_method]
        for method, top in top_frames.items():
            if method == args.baseline_method:
                continue
            out = bootstrap_topk_delta(top, base, rng, args.bootstrap)
            out.update({"seed": seed, "method": method, "baseline": args.baseline_method, "topk_frac": args.topk_frac})
            sig_rows.append(out)
    return pd.DataFrame(cal_rows), pd.concat(bin_frames, ignore_index=True), pd.DataFrame(sig_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration and top-k significance analysis for CrossDocked DiffSBDD reranking.")
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_calibration"))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--topk-frac", type=float, default=0.1)
    parser.add_argument("--baseline-method", default="qed_sa")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--high-conf-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    cal_all = []
    bin_all = []
    sig_all = []
    for pred_path in sorted((args.pred_dir / "seeds").glob("seed_*/oof_predictions.csv")):
        seed = int(pred_path.parent.name.split("_")[-1])
        pred = pd.read_csv(pred_path)
        cal, bins, sig = analyze_seed(pred, seed, args)
        cal_all.append(cal)
        bin_all.append(bins)
        sig_all.append(sig)
    cal_df = pd.concat(cal_all, ignore_index=True)
    bins_df = pd.concat(bin_all, ignore_index=True)
    sig_df = pd.concat(sig_all, ignore_index=True)
    cal_df.to_csv(report_dir / "calibration_by_seed.csv", index=False)
    bins_df.to_csv(report_dir / "calibration_bins.csv", index=False)
    sig_df.to_csv(report_dir / "topk_bootstrap_delta_by_seed.csv", index=False)
    cal_summary = cal_df.groupby("method", sort=False)[["brier", "mean_prob", "high_conf_reliable_rate"]].agg(["mean", "std"]).reset_index()
    cal_summary.columns = ["_".join([str(x) for x in col if x]) for col in cal_summary.columns.to_flat_index()]
    sig_summary = sig_df.groupby(["method", "baseline", "topk_frac"], sort=False)[["delta_mean", "ci95_low", "ci95_high", "p_delta_le_0"]].mean().reset_index()
    cal_summary.to_csv(report_dir / "calibration_summary.csv", index=False)
    sig_summary.to_csv(report_dir / "topk_bootstrap_delta_summary.csv", index=False)
    lines = [
        "# DiffSBDD CrossDocked Calibration and Significance",
        "",
        "Calibration diagnostics and pocket-bootstrap top-k reliable-rate deltas against QED/SA.",
        "",
        "## Calibration",
        "",
        cal_summary.to_csv(index=False),
        "## Top-k Bootstrap Delta",
        "",
        sig_summary.to_csv(index=False),
    ]
    (report_dir / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "pred_dir": str(args.pred_dir),
                "bootstrap": args.bootstrap,
                "topk_frac": args.topk_frac,
                "baseline_method": args.baseline_method,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((report_dir / "calibration_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
