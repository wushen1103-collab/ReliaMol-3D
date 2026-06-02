from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/proxy")
    parser.add_argument("--output", default="reports/proxy_summary.csv")
    args = parser.parse_args()

    root = Path(args.input)
    metric_files = sorted(root.glob("metrics_*.csv"))
    if not metric_files:
        raise SystemExit(f"No metrics files found under {root}")
    metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
    group_cols = ["split", "heldout_generator", "method"]
    value_cols = ["roc_auc", "pr_auc", "f1", "mcc", "ece", "positive_rate"]
    summary = metrics.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

