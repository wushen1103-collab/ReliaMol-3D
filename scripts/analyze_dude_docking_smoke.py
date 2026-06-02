from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def active_better_probability(active_scores: pd.Series, decoy_scores: pd.Series) -> float:
    active = active_scores.dropna().to_numpy()
    decoy = decoy_scores.dropna().to_numpy()
    if len(active) == 0 or len(decoy) == 0:
        return math.nan
    return float((active[:, None] < decoy[None, :]).mean())


def add_pose_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["docked_success"] = out["status"].eq("success")
    out["pose_near_crystal_8A"] = out["docked_success"] & out["center_to_crystal"].le(8.0)
    out["pose_has_pocket_contact"] = out["docked_success"] & out["pocket_contact_atoms_4p5"].gt(0)
    out["pose_clash_free"] = out["docked_success"] & out["receptor_clash_atoms_1p2"].eq(0)
    out["public_pose_reliable"] = (
        out["docked_success"]
        & out["pose_near_crystal_8A"]
        & out["pose_has_pocket_contact"]
        & out["pose_clash_free"]
    )
    out["chemical_failure"] = ~out["docked_success"]
    out["geometric_failure"] = out["docked_success"] & ~out["pose_clash_free"]
    out["pocket_failure"] = out["docked_success"] & (~out["pose_near_crystal_8A"] | ~out["pose_has_pocket_contact"])
    out["synthetic_failure"] = False
    out["scoring_failure"] = False
    return out


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, source_class), group in df.groupby(["target", "source_class"], sort=False):
        success = group[group["status"] == "success"]
        rows.append(
            {
                "target": target,
                "source_class": source_class,
                "n": int(len(group)),
                "success_rate": float(group["docked_success"].mean()),
                "public_pose_reliable_rate": float(group["public_pose_reliable"].mean()),
                "median_vina_score": float(success["vina_score"].median()) if len(success) else math.nan,
                "iqr_vina_score": float(success["vina_score"].quantile(0.75) - success["vina_score"].quantile(0.25))
                if len(success)
                else math.nan,
                "median_center_to_crystal": float(success["center_to_crystal"].median()) if len(success) else math.nan,
                "frac_center_within_8A": float(success["center_to_crystal"].le(8.0).mean()) if len(success) else math.nan,
                "frac_any_pocket_contact_4p5": float(success["pocket_contact_atoms_4p5"].gt(0).mean()) if len(success) else math.nan,
                "frac_any_receptor_clash_1p2": float(success["receptor_clash_atoms_1p2"].gt(0).mean()) if len(success) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_score_separation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in df[df["status"] == "success"].groupby("target", sort=False):
        active = group[group["source_class"] == "active"]["vina_score"]
        decoy = group[group["source_class"] == "decoy"]["vina_score"]
        active_median = float(active.median()) if len(active) else math.nan
        decoy_median = float(decoy.median()) if len(decoy) else math.nan
        rows.append(
            {
                "target": target,
                "active_median_vina": active_median,
                "decoy_median_vina": decoy_median,
                "active_minus_decoy_median": active_median - decoy_median,
                "p_active_scores_better_than_decoy": active_better_probability(active, decoy),
                "n_active": int(len(active)),
                "n_decoy": int(len(decoy)),
            }
        )
    return pd.DataFrame(rows)


def summarize_topk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = df[df["status"] == "success"].copy()
    for frac in [0.1, 0.2, 0.5]:
        for target, group in scored.groupby("target", sort=False):
            k = max(1, int(np.ceil(len(group) * frac)))
            top = group.nsmallest(k, "vina_score")
            rows.append(
                {
                    "target": target,
                    "topk_frac": frac,
                    "n_selected": int(len(top)),
                    "active_fraction": float(top["source_class"].eq("active").mean()),
                    "public_pose_reliable_rate": float(top["public_pose_reliable"].mean()),
                    "median_vina_score": float(top["vina_score"].median()),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "_empty_\n"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, by_class: pd.DataFrame, separation: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# DUD-E Public Docking Smoke Analysis",
        "",
        "## Raw Data Table: Pose Quality By Target/Class",
        "",
        markdown_table(by_class),
        "## Active/Decoy Vina Score Separation",
        "",
        markdown_table(separation),
        "## Vina Top-k Behavior",
        "",
        markdown_table(topk),
        "## Key Findings",
        "",
    ]

    overall_success = by_class["success_rate"].mean()
    reliable = by_class["public_pose_reliable_rate"].mean()
    lines.append(f"1. Docking pipeline is technically usable: mean success rate is {overall_success:.3f}, and mean public pose reliable rate is {reliable:.3f}.")
    poor_targets = by_class[by_class["frac_center_within_8A"] < 0.9]
    if len(poor_targets):
        names = ", ".join(f"{row.target}/{row.source_class}" for row in poor_targets.itertuples())
        lines.append(f"2. Pose placement is not uniformly tight: center-within-8A falls below 0.9 for {names}, so the next run should use a larger sample and inspect box size/exhaustiveness sensitivity.")
    else:
        lines.append("2. Pose placement is tight in this smoke: all target/class groups have center-within-8A at least 0.9.")

    weak_score = separation[separation["p_active_scores_better_than_decoy"] < 0.6]
    if len(weak_score):
        names = ", ".join(weak_score["target"].tolist())
        lines.append(f"3. Vina score alone is weak as an active/decoy discriminator in this sample: active-better probability is below 0.6 for {names}.")
    else:
        lines.append("3. Vina score shows useful active/decoy separation in this small sample, but this still needs larger sampling and seeds.")

    lines += [
        "",
        "## Suggested Next Experiments",
        "",
        "1. Expand the clean public docking run to all DUD-E targets that have both active and decoy SDF files.",
        "2. Run box-size/exhaustiveness ablations before treating pocket failure labels as stable.",
        "3. Train a no-leak reliability model on docked public poses, with source_class held out from features and leave-one-target validation.",
        "4. Add a scoring-baseline section because this smoke already shows Vina may rank decoys as well as, or better than, actives.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze public DUD-E docking smoke outputs.")
    parser.add_argument("--results", type=Path, default=Path("outputs/dude_docking_smoke/reports/docking_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dude_docking_smoke/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    labeled = add_pose_labels(df)
    by_class = summarize_by_class(labeled)
    separation = summarize_score_separation(labeled)
    topk = summarize_topk(labeled)

    labeled.to_csv(args.output_dir / "docking_reliability_labels.csv", index=False)
    by_class.to_csv(args.output_dir / "docking_pose_quality_by_class.csv", index=False)
    separation.to_csv(args.output_dir / "docking_score_separation.csv", index=False)
    topk.to_csv(args.output_dir / "docking_topk_summary.csv", index=False)
    write_report(args.output_dir / "docking_analysis.md", by_class, separation, topk)

    print("Pose quality by target/class")
    print(by_class.to_string(index=False))
    print("\nScore separation")
    print(separation.to_string(index=False))
    print("\nTop-k")
    print(topk.to_string(index=False))
    print(f"\nReport: {args.output_dir / 'docking_analysis.md'}")


if __name__ == "__main__":
    main()
