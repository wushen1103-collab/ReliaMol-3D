"""Reproduce final docking and three-seed transfer summaries from recorded outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return float(center - half), float(center + half)


def check_table(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert_frame_equal(actual[expected.columns], expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def docking_summary() -> dict:
    directory = ROOT / "independent_docking_final"
    candidates = pd.read_csv(directory / "candidate_results.csv")
    recorded = json.loads((directory / "docking_validation_statistics.json").read_text(encoding="utf-8"))
    groups = []
    for name in ["endpoint_positive", "endpoint_negative"]:
        group = candidates[candidates.endpoint_group.eq(name)]
        successes = int(group.usable_output.sum())
        total = len(group)
        groups.append((successes, total, wilson(successes, total)))
    (positive, n_positive, (lp, up)), (negative, n_negative, (ln, un)) = groups
    p, n = positive / n_positive, negative / n_negative
    delta = p - n
    interval = [delta - np.hypot(p - lp, un - n), delta + np.hypot(up - p, n - ln)]
    result = {
        "n": len(candidates),
        "distinct_pockets": int(candidates.pocket_id.nunique()),
        "vina_searches_completed": int(candidates.docking_status.eq("success").sum()),
        "usable_outputs": int(candidates.usable_output.sum()),
        "usable_rate_endpoint_positive": p,
        "usable_rate_endpoint_positive_wilson_95ci": [lp, up],
        "usable_rate_endpoint_negative": n,
        "usable_rate_endpoint_negative_wilson_95ci": [ln, un],
        "risk_difference_positive_minus_negative": delta,
        "risk_difference_newcombe_95ci": interval,
        "fisher_exact_two_sided_p": float(fisher_exact(
            [[positive, n_positive - positive], [negative, n_negative - negative]],
            alternative="two-sided",
        ).pvalue),
    }
    for name, value in result.items():
        if name in recorded:
            np.testing.assert_allclose(value, recorded[name], atol=1e-12, rtol=1e-12)
    assert (positive, negative, n_positive, n_negative) == (14, 14, 15, 15)
    assert result["vina_searches_completed"] == 30
    return result


def seed_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["evaluation", "seed", "generator", "method"]
    seed = data.groupby(keys, as_index=False).agg(endpoint_positive_rate=("endpoint_positive_rate", "mean"))
    pooled = data.groupby(["evaluation", "seed", "method"], as_index=False).agg(
        endpoint_positive_rate=("endpoint_positive_rate", "mean"))
    pooled["generator"] = "POOLED"
    seed = pd.concat([seed, pooled], ignore_index=True)
    reference = seed[seed.method.eq("qed_sa")].rename(columns={"endpoint_positive_rate": "qed_sa_rate"})
    reference = reference[["evaluation", "seed", "generator", "qed_sa_rate"]]
    seed = seed.merge(reference, on=["evaluation", "seed", "generator"], validate="many_to_one")
    seed["gain_vs_qed_sa"] = seed.endpoint_positive_rate - seed.qed_sa_rate
    return seed.groupby(["evaluation", "generator", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean=("endpoint_positive_rate", "mean"),
        sample_sd=("endpoint_positive_rate", lambda x: float(np.std(x, ddof=1))),
        minimum=("endpoint_positive_rate", "min"),
        maximum=("endpoint_positive_rate", "max"),
        gain_vs_qed_sa_mean=("gain_vs_qed_sa", "mean"),
        gain_vs_qed_sa_sample_sd=("gain_vs_qed_sa", lambda x: float(np.std(x, ddof=1))),
        gain_vs_qed_sa_minimum=("gain_vs_qed_sa", "min"),
        gain_vs_qed_sa_maximum=("gain_vs_qed_sa", "max"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for regenerated tables")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    transfer = ROOT / "transfer_three_seed"
    source = transfer / "transfer_top10_rates_by_seed_and_pocket.csv.gz"
    data = pd.read_csv(source)
    assert set(data.seed) == {11, 22, 33}
    assert data.groupby(["evaluation", "generator", "pocket_id", "method"]).seed.nunique().eq(3).all()
    seed = seed_summary(data)
    check_table(seed, pd.read_csv(transfer / "transfer_seed_variability_summary.csv"),
                ["evaluation", "generator", "method"])
    seed.to_csv(args.output / "transfer_seed_variability_summary.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "summarize_transfer_bootstrap.py"),
                    "--input", str(source), "--output", str(args.output),
                    "--bootstrap-seed", "20261105", "--resamples", "5000"], check=True)
    check_table(pd.read_csv(args.output / "transfer_three_seed_paired_bootstrap.csv"),
                pd.read_csv(transfer / "transfer_three_seed_paired_bootstrap.csv"),
                ["evaluation", "generator", "method"])
    docking = docking_summary()
    (args.output / "docking_statistics_recomputed.json").write_text(json.dumps(docking, indent=2), encoding="utf-8")
    print(json.dumps({"checks_passed": True, "seed_pocket_rows": len(data),
                      "seed_summary_rows": len(seed), "docking": docking}, indent=2))


if __name__ == "__main__":
    main()
