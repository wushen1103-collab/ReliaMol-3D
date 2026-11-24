"""Summarize transfer after averaging three seed-specific rates within pocket."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    assert set(data.seed) == {11, 22, 33}
    counts = data.groupby(["evaluation", "generator", "pocket_id", "method"]).seed.nunique()
    assert counts.eq(3).all(), "Incomplete seed coverage"
    pocket = data.groupby(["evaluation", "generator", "pocket_id", "method"], as_index=False).agg(
        endpoint_positive_rate=("endpoint_positive_rate", "mean"),
        n_selected=("n_selected", "first"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pocket.to_csv(args.output / "transfer_seed_averaged_pocket_rates.csv", index=False)
    results = []
    for evaluation, block in pocket.groupby("evaluation", sort=True):
        groups = list(block.groupby("generator", sort=True)) + [("POOLED", block)]
        for generator, group in groups:
            wide = group.pivot(index=["generator", "pocket_id"], columns="method", values="endpoint_positive_rate")
            reference = wide["qed_sa"].to_numpy(float)
            # Identical pocket resamples across methods preserve pairing.
            rng = np.random.default_rng(args.bootstrap_seed)
            indices = rng.integers(0, len(wide), size=(args.resamples, len(wide)))
            for method in wide.columns:
                values = wide[method].to_numpy(float)
                delta = values - reference
                boot_delta = delta[indices].mean(axis=1)
                boot_absolute = values[indices].mean(axis=1)
                results.append({
                    "evaluation": evaluation, "generator": generator, "method": method,
                    "n_pockets": len(wide), "n_seeds": 3,
                    "mean_rate": float(values.mean()), "reference_rate": float(reference.mean()),
                    "delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boot_delta, 0.025)),
                    "ci_high": float(np.quantile(boot_delta, 0.975)),
                    "absolute_ci_low": float(np.quantile(boot_absolute, 0.025)),
                    "absolute_ci_high": float(np.quantile(boot_absolute, 0.975)),
                    "wins": int((delta > 0.01 + 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 0.01 + 1e-12).sum()),
                    "losses": int((delta < -0.01 - 1e-12).sum()),
                    "bootstrap_resamples": args.resamples, "bootstrap_seed": args.bootstrap_seed,
                })
    result = pd.DataFrame(results)
    result.to_csv(args.output / "transfer_three_seed_paired_bootstrap.csv", index=False)
    print(result[result.evaluation.eq("double_holdout") & ~result.method.eq("qed_sa")].to_string(index=False))


if __name__ == "__main__":
    main()
