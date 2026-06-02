from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/real_smoke_crossdocked_logo")
    parser.add_argument("--output", default="reports/real_smoke_crossdocked_logo_summary.csv")
    args = parser.parse_args()
    root = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = []
    for path in sorted(root.glob("*/metrics.csv")):
        metrics.append(pd.read_csv(path))
    metric_frame = pd.concat(metrics, ignore_index=True)
    metric_frame.to_csv(out.with_name(out.stem + "_raw_metrics.csv"), index=False)
    metric_summary = metric_frame.groupby("method")[["roc_auc", "pr_auc", "f1", "mcc", "ece"]].agg(["mean", "std"]).reset_index()
    metric_summary.to_csv(out, index=False)

    rerank = []
    for path in sorted(root.glob("*/reranking.csv")):
        rerank.append(pd.read_csv(path))
    rerank_frame = pd.concat(rerank, ignore_index=True)
    rerank_frame.to_csv(out.with_name(out.stem + "_raw_reranking.csv"), index=False)
    rerank_summary = rerank_frame.groupby(["score", "topk_frac"])[
        ["reliable_rate", "clash_free_proxy", "vina_gnina_consistency_proxy", "mean_affinity_proxy"]
    ].agg(["mean", "std"]).reset_index()
    rerank_summary.to_csv(out.with_name(out.stem + "_reranking.csv"), index=False)

    print("LOGO METRICS")
    print(metric_frame.sort_values(["heldout_generator", "method"]).to_string(index=False))
    print("\nLOGO SUMMARY")
    print(metric_summary.to_string(index=False))
    print("\nLOGO RERANKING")
    print(rerank_summary.to_string(index=False))


if __name__ == "__main__":
    main()

