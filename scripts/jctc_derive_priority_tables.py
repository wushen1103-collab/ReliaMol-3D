#!/usr/bin/env python3
"""Derive JCTC-oriented physical and coordinate-provenance evidence tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs")
OLD_PHYS = ROOT / "revision_r1_physical_endpoint_analysis_v0"
ALIGNED_RELAX = ROOT / "jctc_aligned_openmm_relaxation_v0"
OOF = ROOT / "revision_r1_leakage_shift_audit_v0" / "main_oof_predictions.csv"
OUT = ROOT / "jctc_priority_tables_v1"
TARGET = "stable_r2p0_c0p5_with_pb"
METHODS = [
    "qed_sa",
    "label_informed_rule",
    "logreg_full",
    "rf_full",
    "hgb_full",
    "rf_direct_excluded",
]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).fillna(False)


def summarize_mean_sd(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for group_keys, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(keys, group_keys))
        row["n"] = int(len(group))
        for value in values:
            if value in group:
                row[f"{value}_mean"] = float(group[value].mean())
                row[f"{value}_sd"] = float(group[value].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def topk(frame: pd.DataFrame, score_cols: list[str], target: str, coverage: float, seed: int, source: str):
    rows, pocket_rows = [], []
    for score_col in score_cols:
        selected = []
        method = score_col.replace("score_", "", 1)
        for pocket_id, group in frame.groupby("pocket_id", sort=False):
            group = group.dropna(subset=[score_col])
            if group.empty:
                continue
            k = max(1, int(math.ceil(coverage * len(group))))
            top = group.nlargest(k, score_col)
            selected.append(top)
            pocket_rows.append({
                "seed": seed,
                "source": source,
                "method": method,
                "pocket_id": pocket_id,
                "n_selected": int(len(top)),
                "stable_rate": float(top[target].mean()),
                "relaxation_success_rate": float(top["relaxation_success"].mean()),
                "final_pb_pass_rate": float(top["final_pb_pass"].mean()),
                "median_ligand_rmsd_a": float(top.loc[top["relaxation_success"], "ligand_rmsd_a"].median()),
                "mean_contact_retention": float(top.loc[top["relaxation_success"], "contact_retention"].mean()),
            })
        if not selected:
            continue
        picked = pd.concat(selected, ignore_index=True)
        rows.append({
            "seed": seed,
            "source": source,
            "method": method,
            "coverage": coverage,
            "n_selected": int(len(picked)),
            "stable_rate": float(picked[target].mean()),
            "stable_count": int(picked[target].sum()),
            "relaxation_success_rate": float(picked["relaxation_success"].mean()),
            "final_pb_pass_rate": float(picked["final_pb_pass"].mean()),
            "final_clash_free_rate": float(picked["final_clash_free"].mean()),
            "median_ligand_rmsd_a": float(picked.loc[picked["relaxation_success"], "ligand_rmsd_a"].median()),
            "mean_contact_retention": float(picked.loc[picked["relaxation_success"], "contact_retention"].mean()),
        })
    return pd.DataFrame(rows), pd.DataFrame(pocket_rows)


def worst_pocket(per_pocket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, method), group in per_pocket.groupby(["source", "method"], sort=False):
        values = group["stable_rate"].astype(float).to_numpy()
        rows.append({
            "source": source,
            "method": method,
            "n_pockets": int(len(values)),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "bottom_quartile_mean": float(np.mean(values[values <= np.quantile(values, 0.25)])),
            "perfect_pockets": int(np.sum(values >= 1.0)),
        })
    return pd.DataFrame(rows)


def endpoint_overlap(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    audit = bool_series(frame["reliable"])
    physical = bool_series(frame[target])
    rows = []
    for audit_value in [False, True]:
        for physical_value in [False, True]:
            mask = audit.eq(audit_value) & physical.eq(physical_value)
            rows.append({
                "audit_reliable": audit_value,
                "physical_stable": physical_value,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
            })
    rows.append({
        "audit_reliable": "all",
        "physical_stable": "agreement",
        "count": int(audit.eq(physical).sum()),
        "fraction": float(audit.eq(physical).mean()),
    })
    return pd.DataFrame(rows)


def endpoint_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    success = out["status"].eq("success")
    out["relaxation_success"] = success
    out["final_pb_pass"] = success & bool_series(out.get("final_posebusters_pass", pd.Series(False, index=out.index)))
    out["final_clash_free"] = success & out["final_clash_atoms_1p2"].fillna(1).eq(0)
    out[TARGET] = (
        success
        & out["ligand_rmsd_a"].fillna(np.inf).le(2.0)
        & out["contact_retention"].fillna(-np.inf).ge(0.5)
        & out["final_clash_atoms_1p2"].fillna(1).eq(0)
        & out["final_pb_pass"]
    )
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, sep] + rows)


def aligned_frame() -> pd.DataFrame:
    files = sorted(ALIGNED_RELAX.glob("relaxation_shard_*.csv"))
    if len(files) != 8:
        raise FileNotFoundError(f"Expected 8 aligned relaxation CSV shards, found {len(files)}")
    frame = pd.concat([pd.read_csv(file) for file in files], ignore_index=True, sort=False)
    return endpoint_columns(frame.drop_duplicates("candidate_uid", keep="last"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    old_top = pd.read_csv(OLD_PHYS / "physical_top10_by_seed.csv")
    old_per = pd.read_csv(OLD_PHYS / "physical_per_pocket_top10.csv")
    old_frame = pd.read_csv(OLD_PHYS / "relaxation_endpoint_frame.csv")
    old_metrics = pd.read_csv(OLD_PHYS / "physical_metrics_by_seed.csv")
    old_boot = pd.read_csv(OLD_PHYS / "physical_paired_bootstrap.csv")
    old_sens = pd.read_csv(OLD_PHYS / "endpoint_sensitivity.csv")

    old_top_mean = summarize_mean_sd(
        old_top[old_top["method"].isin(METHODS + ["phys_rf_direct_excluded", "phys_rf_full", "phys_hgb_full", "phys_logreg_full"])],
        ["source", "method", "coverage"],
        ["stable_rate", "relaxation_success_rate", "final_pb_pass_rate", "median_ligand_rmsd_a", "mean_contact_retention"],
    )
    old_top_mean.to_csv(OUT / "table_1_crossdocked_physical_top10_mean.csv", index=False)
    old_metrics.to_csv(OUT / "table_1b_crossdocked_physical_metrics_by_seed.csv", index=False)
    old_boot.to_csv(OUT / "table_1c_crossdocked_physical_bootstrap.csv", index=False)
    old_sens.to_csv(OUT / "table_1d_crossdocked_endpoint_sensitivity.csv", index=False)
    worst_pocket(old_per).to_csv(OUT / "table_2_crossdocked_worst_pockets.csv", index=False)
    endpoint_overlap(old_frame, TARGET).to_csv(OUT / "table_3_crossdocked_endpoint_overlap.csv", index=False)

    aligned = aligned_frame()
    aligned_qc = pd.DataFrame([{
        "n_candidates": int(len(aligned)),
        "n_pockets": int(aligned["pocket_id"].nunique()),
        "relaxation_success_rate": float(aligned["relaxation_success"].mean()),
        "final_pb_pass_rate": float(aligned["final_pb_pass"].mean()),
        "stable_r2p0_c0p5_with_pb_rate": float(aligned[TARGET].mean()),
        "median_ligand_rmsd_a": float(aligned.loc[aligned["relaxation_success"], "ligand_rmsd_a"].median()),
        "mean_contact_retention": float(aligned.loc[aligned["relaxation_success"], "contact_retention"].mean()),
        "parameterization_failure_rate": float(1.0 - aligned["relaxation_success"].mean()),
    }])
    aligned_qc.to_csv(OUT / "table_4_aligned_relaxation_overall_qc.csv", index=False)
    aligned.groupby("pocket_id", as_index=False).agg(
        n_candidates=("candidate_uid", "count"),
        relaxation_success_rate=("relaxation_success", "mean"),
        final_pb_pass_rate=("final_pb_pass", "mean"),
        stable_rate=(TARGET, "mean"),
        median_ligand_rmsd_a=("ligand_rmsd_a", "median"),
        mean_contact_retention=("contact_retention", "mean"),
    ).to_csv(OUT / "table_4b_aligned_relaxation_by_pocket.csv", index=False)

    preds = pd.read_csv(OOF)
    score_cols = [f"score_{m}" for m in METHODS if f"score_{m}" in preds.columns]
    rows, pockets = [], []
    for seed, pred in preds.groupby("seed", sort=False):
        merged = aligned.merge(pred[["candidate_uid"] + score_cols], on="candidate_uid", how="left")
        top, pp = topk(merged, score_cols, TARGET, 0.1, int(seed), "aligned_public_pdb_zero_shot")
        rows.append(top)
        pockets.append(pp)
    aligned_top = pd.concat(rows, ignore_index=True)
    aligned_per = pd.concat(pockets, ignore_index=True)
    aligned_top.to_csv(OUT / "table_5_aligned_zero_shot_top10_by_seed.csv", index=False)
    summarize_mean_sd(
        aligned_top,
        ["source", "method", "coverage"],
        ["stable_rate", "relaxation_success_rate", "final_pb_pass_rate", "median_ligand_rmsd_a", "mean_contact_retention"],
    ).to_csv(OUT / "table_5b_aligned_zero_shot_top10_mean.csv", index=False)
    worst_pocket(aligned_per).to_csv(OUT / "table_6_aligned_zero_shot_worst_pockets.csv", index=False)

    key_methods = ["qed_sa", "label_informed_rule", "rf_full", "hgb_full", "phys_rf_direct_excluded", "phys_rf_full"]
    old_key = old_top_mean[old_top_mean["method"].isin(key_methods)].copy()
    aligned_key = pd.read_csv(OUT / "table_5b_aligned_zero_shot_top10_mean.csv")
    aligned_key = aligned_key[aligned_key["method"].isin(["qed_sa", "label_informed_rule", "rf_full", "hgb_full"])]
    metadata = {
        "crossdocked_candidates": int(len(old_frame)),
        "crossdocked_pockets": int(old_frame["pocket_id"].nunique()),
        "aligned_candidates": int(len(aligned)),
        "aligned_pockets": int(aligned["pocket_id"].nunique()),
        "target": TARGET,
        "note": "Aligned zero-shot uses the original audit-label OOF scores and evaluates them against independent post-relaxation public-PDB endpoints.",
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# JCTC Priority Evidence Summary",
        "",
        f"- CrossDocked physical endpoint: {metadata['crossdocked_candidates']} candidates across {metadata['crossdocked_pockets']} pockets.",
        f"- Public-PDB aligned physical endpoint: {metadata['aligned_candidates']} candidates across {metadata['aligned_pockets']} pockets.",
        f"- Aligned relaxation success rate: {aligned_qc.loc[0, 'relaxation_success_rate']:.4f}; final PB pass rate: {aligned_qc.loc[0, 'final_pb_pass_rate']:.4f}; stable endpoint rate: {aligned_qc.loc[0, 'stable_r2p0_c0p5_with_pb_rate']:.4f}.",
        "",
        "## CrossDocked Top-10% Physical Endpoint",
        markdown_table(old_key),
        "",
        "## Public-PDB Aligned Zero-Shot Top-10%",
        markdown_table(aligned_key),
        "",
        "## Endpoint Overlap",
        markdown_table(pd.read_csv(OUT / "table_3_crossdocked_endpoint_overlap.csv")),
    ]
    (OUT / "jctc_priority_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
