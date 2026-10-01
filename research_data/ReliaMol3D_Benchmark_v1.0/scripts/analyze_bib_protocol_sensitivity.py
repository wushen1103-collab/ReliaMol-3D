#!/usr/bin/env python3
"""Analyze fixed-shortlist endpoint sensitivity to receptor LJ epsilon."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", Path.cwd()))
BASE = ROOT / "outputs/bib_protocol_sensitivity_v0"
ORIGINAL = ROOT / "outputs/bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv"
PRIMARY_BOOTSTRAP = ROOT / "outputs/bib_reviewer_gap_closure_v2/table_D_diffsbdd_full_audit_direct_bootstrap.csv"
METHODS = ("qed_sa", "five_rule", "logistic", "rf")
SEEDS = (11, 22, 33)


def read_jsonl_directory(path: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(path.glob("relaxation_shard_*.jsonl")):
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No relaxation records found in {path}")
    return frame.drop_duplicates("candidate_uid", keep="last")


def bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).lower() in {"true", "1", "yes"}


def endpoint_from_records(frame: pd.DataFrame) -> pd.Series:
    success = frame["status"].eq("success")
    rmsd = pd.to_numeric(frame.get("ligand_rmsd_a"), errors="coerce").le(2.0)
    retention = pd.to_numeric(frame.get("contact_retention"), errors="coerce").ge(0.5)
    clash_free = pd.to_numeric(frame.get("final_clash_atoms_1p2"), errors="coerce").eq(0)
    pb = frame.get("final_posebusters_pass", pd.Series(False, index=frame.index)).map(bool_value)
    return (success & rmsd & retention & clash_free & pb).astype(int)


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def main() -> None:
    membership = pd.read_csv(BASE / "protocol_sensitivity_membership.csv")
    original = pd.read_csv(ORIGINAL, usecols=["candidate_uid", "stable_r2p0_c0p5_with_pb"])
    original = original.rename(columns={"stable_r2p0_c0p5_with_pb": "endpoint_0.20"})
    combined = membership.merge(original, on="candidate_uid", how="left")
    records_by_epsilon = {}
    for label, directory in [("0.10", "epsilon_0p10"), ("0.40", "epsilon_0p40")]:
        records = read_jsonl_directory(BASE / directory)
        records[f"endpoint_{label}"] = endpoint_from_records(records)
        records_by_epsilon[label] = records
        combined = combined.merge(records[["candidate_uid", f"endpoint_{label}"]], on="candidate_uid", how="left")
        combined[f"endpoint_{label}"] = combined[f"endpoint_{label}"].fillna(0).astype(int)
    combined["endpoint_0.20"] = combined["endpoint_0.20"].fillna(0).astype(int)

    rows = []
    pocket_rows = []
    for epsilon in ("0.10", "0.20", "0.40"):
        endpoint = f"endpoint_{epsilon}"
        rates = {}
        for method in METHODS:
            if method in {"logistic", "rf"}:
                seed_rates = []
                for seed in SEEDS:
                    selected = combined[combined[f"selected_{method}_seed{seed}"]]
                    one_seed = selected.groupby("pocket_id")[endpoint].mean().rename(seed)
                    seed_rates.append(one_seed)
                    for pocket, value in one_seed.items():
                        pocket_rows.append({
                            "epsilon_kj_mol": float(epsilon),
                            "method": method,
                            "seed": seed,
                            "pocket_id": pocket,
                            "endpoint_positive_rate": float(value),
                        })
                per_pocket = pd.concat(seed_rates, axis=1).mean(axis=1)
            else:
                selected = combined[combined[f"selected_{method}"]]
                per_pocket = selected.groupby("pocket_id")[endpoint].mean()
                for pocket, value in per_pocket.items():
                    pocket_rows.append({
                        "epsilon_kj_mol": float(epsilon),
                        "method": method,
                        "seed": "fixed",
                        "pocket_id": pocket,
                        "endpoint_positive_rate": float(value),
                    })
            rates[method] = per_pocket
        for index, method in enumerate(METHODS):
            delta = (rates[method] - rates["qed_sa"]).dropna().to_numpy(float)
            low, high = (float("nan"), float("nan")) if method == "qed_sa" else bootstrap(delta, 20260901 + index + int(float(epsilon) * 100))
            rows.append({
                "epsilon_kj_mol": float(epsilon),
                "method": method,
                "n_pockets": int(len(rates[method])),
                "endpoint_positive_rate": float(rates[method].mean()),
                "delta_vs_qed_sa": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
            })

    agreement = []
    for epsilon in ("0.10", "0.40"):
        left = combined[f"endpoint_{epsilon}"]
        right = combined["endpoint_0.20"]
        agreement.append({
            "epsilon_kj_mol": float(epsilon),
            "union_candidates": int(len(combined)),
            "agreement_with_0p20": float((left == right).mean()),
            "positive_rate": float(left.mean()),
            "reference_0p20_positive_rate": float(right.mean()),
            "negative_to_positive": int(((right == 0) & (left == 1)).sum()),
            "positive_to_negative": int(((right == 1) & (left == 0)).sum()),
        })
    summary = pd.DataFrame(rows)
    # Reuse the prespecified primary-analysis bootstrap for the unchanged
    # 0.20 kJ/mol endpoint so one result is not reported with two Monte Carlo
    # percentile intervals generated from different random streams.
    primary = pd.read_csv(PRIMARY_BOOTSTRAP)
    primary_names = {
        "five_rule": "full_audit_gate_qedsa_fixed_budget",
        "logistic": "phys_logreg_full",
        "rf": "phys_rf_direct_excluded",
    }
    for method, primary_method in primary_names.items():
        match = primary.loc[
            primary["method"].eq(primary_method) & primary["reference"].eq("qed_sa")
        ].iloc[0]
        mask = summary["epsilon_kj_mol"].eq(0.20) & summary["method"].eq(method)
        for column in ["ci_low", "ci_high", "wins", "ties", "losses"]:
            summary.loc[mask, column] = match[column]
    summary.to_csv(BASE / "protocol_sensitivity_summary.csv", index=False)
    pd.DataFrame(pocket_rows).to_csv(BASE / "protocol_sensitivity_per_pocket.csv", index=False)
    pd.DataFrame(agreement).to_csv(BASE / "protocol_sensitivity_agreement.csv", index=False)
    combined.to_csv(BASE / "protocol_sensitivity_candidate_endpoints.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(agreement).to_string(index=False))


if __name__ == "__main__":
    main()
