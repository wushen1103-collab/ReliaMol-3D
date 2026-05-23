from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_metric_files(root: Path, filename: str, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"seed*/{filename}")):
        seed = path.parent.name.replace("seed", "")
        frame = pd.read_csv(path)
        frame["seed"] = int(seed)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No {filename} files found under {root}")
    all_rows = pd.concat(frames, ignore_index=True)
    return all_rows.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/real_smoke_crossdocked_v2_multiseed")
    parser.add_argument("--output", default="reports/real_smoke_crossdocked_v2_multiseed_summary.csv")
    args = parser.parse_args()
    root = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    metric_summary = summarize_metric_files(
        root,
        "metrics.csv",
        ["method"],
        ["roc_auc", "pr_auc", "f1", "mcc", "ece"],
    )
    metric_summary.to_csv(out, index=False)

    rerank_frames = []
    for path in sorted(root.glob("seed*/reranking.csv")):
        seed = int(path.parent.name.replace("seed", ""))
        frame = pd.read_csv(path)
        frame["seed"] = seed
        rerank_frames.append(frame)
    rerank = pd.concat(rerank_frames, ignore_index=True)
    rerank_summary = rerank.groupby(["score", "topk_frac"])[
        ["reliable_rate", "clash_free_proxy", "vina_gnina_consistency_proxy", "mean_affinity_proxy"]
    ].agg(["mean", "std"]).reset_index()
    rerank_summary.to_csv(out.with_name(out.stem + "_reranking.csv"), index=False)

    print("METRICS")
    print(metric_summary.to_string(index=False))
    print("\nRERANKING")
    print(rerank_summary.to_string(index=False))


if __name__ == "__main__":
    main()

