#!/usr/bin/env python3
"""Create the fixed shortlist-union frame for endpoint protocol sensitivity."""

import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", Path.cwd()))
FRAME = ROOT / "outputs/bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv"
OOF = ROOT / "outputs/bib_physical_endpoint_analysis_v0/physical_oof_predictions.csv"
OUT = ROOT / "outputs/bib_protocol_sensitivity_v0"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FRAME)
    oof = pd.read_csv(OOF)
    scores = oof.groupby("candidate_uid", as_index=False).agg(
        score_logistic=("score_phys_logreg_full", "mean"),
        score_rf=("score_phys_rf_direct_excluded", "mean"),
    )
    frame = frame.merge(scores, on="candidate_uid", how="left")
    for seed in sorted(oof["seed"].unique()):
        seed_scores = oof[oof["seed"].eq(seed)][
            ["candidate_uid", "score_phys_logreg_full", "score_phys_rf_direct_excluded"]
        ].rename(columns={
            "score_phys_logreg_full": f"score_logistic_seed{seed}",
            "score_phys_rf_direct_excluded": f"score_rf_seed{seed}",
        })
        frame = frame.merge(seed_scores, on="candidate_uid", how="left")
    frame["score_qed_sa"] = frame["qed"].astype(float) - 0.25 * frame["sa_proxy"].astype(float)
    failure_columns = [
        "chemical_failure", "geometric_failure", "pocket_failure",
        "clash_failure", "synthetic_failure",
    ]
    frame["direct_failure_count"] = frame[failure_columns].astype(int).sum(axis=1)
    frame["score_five_rule"] = frame["direct_failure_count"].eq(0).astype(float) * 1000.0 + frame["score_qed_sa"]
    methods = {
        "qed_sa": "score_qed_sa",
        "five_rule": "score_five_rule",
        "logistic": "score_logistic",
        "rf": "score_rf",
    }
    seed_methods = {
        f"{method}_seed{seed}": f"score_{method}_seed{seed}"
        for method in ("logistic", "rf")
        for seed in sorted(oof["seed"].unique())
    }
    all_methods = {**methods, **seed_methods}
    for method in all_methods:
        frame[f"selected_{method}"] = False
    for _, group in frame.groupby("pocket_id", sort=False):
        for method, score in all_methods.items():
            chosen = group.nlargest(10, score).index
            frame.loc[chosen, f"selected_{method}"] = True
    selection_columns = [f"selected_{method}" for method in all_methods]
    subset = frame[frame[selection_columns].any(axis=1)].copy()
    subset.to_csv(OUT / "protocol_sensitivity_feature_frame.csv", index=False)
    membership = subset[["candidate_uid", "pocket_id"] + selection_columns]
    membership.to_csv(OUT / "protocol_sensitivity_membership.csv", index=False)
    counts = {
        "candidate_union": int(len(subset)),
        "pockets": int(subset["pocket_id"].nunique()),
        **{method: int(subset[f"selected_{method}"].sum()) for method in all_methods},
    }
    pd.Series(counts, name="value").to_csv(OUT / "protocol_sensitivity_subset_counts.csv")
    print(counts)


if __name__ == "__main__":
    main()
