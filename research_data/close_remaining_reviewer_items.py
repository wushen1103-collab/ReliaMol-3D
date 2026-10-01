"""Compute candidate-matched revision analyses from repository output tables.

Run from the repository root, or set RELIAMOL_OUTPUT_ROOT to its outputs directory.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", "outputs"))
SOURCE = ROOT / "bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv"
DOCKING = ROOT / "diffsbdd_crossdocked_vina_gnina_100pocket_v0/reports/vina_gnina_scores.csv"
TARGET = ROOT / "bib_multigen_physical_transfer_v2/multigen_endpoint_frame.csv"
TARGET_FINAL = ROOT / "bib_multigen_physical_transfer_v2/multigen_openmm_endpoint_frame_final.csv"
DOUBLE = ROOT / "bib_reviewer_gap_closure_v2/table_G_double_holdout_per_pocket.csv"
ENDPOINT = "stable_r2p0_c0p5_with_pb"
SEEDS = (11, 22, 33)

ATOM = [f"atom_{x}_frac" for x in ("metal", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I")]
CHEM = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM
INTERNAL = [
    "pose_atoms", "pose_radius_shape", "nn_q05", "nn_q10", "nn_q25", "nn_q50", "nn_q75", "nn_q90",
]
POCKET = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90",
    "shell_0_2", "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
FULL = CHEM + INTERNAL + POCKET
RULE_VARIABLES = {
    "rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms",
    "sa_proxy", "descriptor_ok", "pose_radius_shape",
}
DIRECT_EXCLUDED = [name for name in FULL if name not in RULE_VARIABLES]


def binary(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).to_numpy(int)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fold_pockets(pockets: np.ndarray, seed: int) -> list[np.ndarray]:
    shuffled = np.asarray(pockets).copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return list(np.array_split(shuffled, 5))


def pocket_topk(frame: pd.DataFrame, score: str, method: str, seed: int) -> list[dict]:
    rows = []
    for pocket, group in frame.groupby("pocket_id", sort=False):
        k = max(1, math.ceil(0.10 * len(group)))
        selected = group.sort_values(score, ascending=False, kind="stable").head(k)
        rows.append({
            "seed": seed,
            "pocket_id": pocket,
            "method": method,
            "n_selected": len(selected),
            "endpoint_positive": binary(selected[ENDPOINT]).mean(),
        })
    return rows


def paired_bootstrap(pocket_rates: pd.DataFrame, reference: str, methods: list[str], resamples: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(20261001)
    wide = pocket_rates.pivot_table(index="pocket_id", columns="method", values="endpoint_positive", aggfunc="mean")
    rows = []
    for method in methods:
        paired = wide[[method, reference]].dropna()
        delta = (paired[method] - paired[reference]).to_numpy(float)
        samples = rng.choice(delta, (resamples, len(delta)), replace=True).mean(axis=1)
        rows.append({
            "method": method,
            "reference": reference,
            "n_pockets": len(delta),
            "endpoint_positive": float(paired[method].mean()),
            "reference_endpoint_positive": float(paired[reference].mean()),
            "delta": float(delta.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "wins": int((delta > 0.01).sum()),
            "ties": int((np.abs(delta) <= 0.01).sum()),
            "losses": int((delta < -0.01).sum()),
        })
    return pd.DataFrame(rows)


def source_docking_comparison(source: pd.DataFrame, out: Path) -> None:
    docking = pd.read_csv(DOCKING, low_memory=False)
    assert len(source) == source.candidate_uid.nunique() == 10000
    assert len(docking) == docking.candidate_uid.nunique() == 10000
    source = source.merge(
        docking[["candidate_uid", "score_vina", "vina_status"]],
        on="candidate_uid", how="left", validate="one_to_one",
    )
    source["score_vina"] = pd.to_numeric(source.score_vina, errors="coerce")
    source["score_qed_sa"] = source.qed.astype(float) - 0.25 * source.sa_proxy.astype(float)
    n_scored = int(source.score_vina.notna().sum())
    assert n_scored == 9901, f"Docking coverage changed: {n_scored}"
    pd.DataFrame([{
        "candidates": len(source),
        "vina_scored": n_scored,
        "vina_unscored": len(source) - n_scored,
        "scored_endpoint_positive": binary(source.loc[source.score_vina.notna(), ENDPOINT]).mean(),
        "unscored_endpoint_positive": binary(source.loc[source.score_vina.isna(), ENDPOINT]).mean(),
    }]).to_csv(out / "docking_score_coverage.csv", index=False)

    rows = []
    for seed in SEEDS:
        predictions = source[["candidate_uid", "pocket_id", ENDPOINT, "score_qed_sa", "score_vina"]].copy()
        predictions["score_vina_rank"] = predictions.score_vina.fillna(-np.inf)
        predictions["score_logistic_34"] = np.nan
        predictions["score_logistic_34_vina"] = np.nan
        for heldout in fold_pockets(source.pocket_id.unique(), seed):
            test_mask = source.pocket_id.isin(heldout)
            train = source.loc[~test_mask]
            test = source.loc[test_mask]
            y_train = binary(train[ENDPOINT])
            for name, columns in (
                ("score_logistic_34", DIRECT_EXCLUDED),
                ("score_logistic_34_vina", DIRECT_EXCLUDED + ["score_vina"]),
            ):
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed),
                )
                model.fit(numeric(train, columns), y_train)
                predictions.loc[test_mask, name] = model.predict_proba(numeric(test, columns))[:, 1]
        for score, name in (
            ("score_qed_sa", "QED/SA"),
            ("score_vina_rank", "Vina score-only"),
            ("score_logistic_34", "Logistic 34-feature"),
            ("score_logistic_34_vina", "Logistic 34-feature + Vina"),
        ):
            rows.extend(pocket_topk(predictions, score, name, seed))

    rates = pd.DataFrame(rows)
    rates.to_csv(out / "docking_combined_per_pocket.csv", index=False)
    comparisons = []
    for reference, methods in (
        ("QED/SA", ["Vina score-only", "Logistic 34-feature", "Logistic 34-feature + Vina"]),
        ("Logistic 34-feature", ["Logistic 34-feature + Vina"]),
    ):
        comparisons.append(paired_bootstrap(rates, reference, methods))
    pd.concat(comparisons, ignore_index=True).to_csv(out / "docking_combined_summary.csv", index=False)
    assert abs(rates.loc[rates.method.eq("Logistic 34-feature"), "endpoint_positive"].mean() - 0.9353) < 0.001


def double_holdout_uncertainty(out: Path) -> None:
    data = pd.read_csv(DOUBLE)
    results = []
    for generator, group in data.groupby("held_out_generator", sort=False):
        pocket = group.rename(columns={"endpoint_positive_rate": "endpoint_positive"})
        paired = paired_bootstrap(pocket, "qed_sa", ["double_logreg_full", "double_hgb_full"], 5000)
        paired.insert(0, "generator", generator)
        results.append(paired)
    pd.concat(results, ignore_index=True).to_csv(out / "double_holdout_paired_ci.csv", index=False)


def generator_feature_importance(source: pd.DataFrame, out: Path) -> None:
    features = pd.read_csv(TARGET, low_memory=False).drop(columns=[ENDPOINT])
    outcomes = pd.read_csv(TARGET_FINAL, usecols=["candidate_uid", ENDPOINT])
    target = features.merge(outcomes, on="candidate_uid", how="left", validate="one_to_one")
    assert len(target) == 38073 and target.candidate_uid.nunique() == len(target)
    observed = target.groupby("generator")[ENDPOINT].mean()
    expected = {"AR": 0.8120, "CVAE": 0.6076, "Pocket2Mol": 0.9828, "TargetDiff": 0.8925}
    for generator, rate in expected.items():
        assert abs(observed[generator] - rate) < 0.0001, (generator, observed[generator], rate)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2500, class_weight="balanced", random_state=20260813),
    )
    model.fit(numeric(source, FULL), binary(source[ENDPOINT]))
    rows = []
    rng = np.random.default_rng(20261001)
    for generator, group in target.groupby("generator", sort=False):
        matrix = numeric(group, FULL)
        labels = binary(group[ENDPOINT])
        base = roc_auc_score(labels, model.predict_proba(matrix)[:, 1])
        for feature in FULL:
            drops = []
            for _ in range(3):
                permuted = matrix.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                drops.append(base - roc_auc_score(labels, model.predict_proba(permuted)[:, 1]))
            rows.append({
                "generator": generator,
                "feature": feature,
                "candidates": len(group),
                "baseline_roc_auc": base,
                "permutation_auc_drop_mean": float(np.mean(drops)),
                "permutation_auc_drop_sd": float(np.std(drops, ddof=1)),
            })
    importance = pd.DataFrame(rows).sort_values(["generator", "permutation_auc_drop_mean"], ascending=[True, False])
    importance.to_csv(out / "generator_permutation_importance.csv", index=False)
    importance.groupby("generator", sort=False).head(10).to_csv(out / "generator_importance_top10.csv", index=False)


def figure6_shortlist_distribution(source: pd.DataFrame, shortlist_path: Path, manifest_path: Path, out: Path) -> None:
    ids = pd.read_csv(shortlist_path)
    manifest = pd.read_csv(manifest_path)
    pockets = manifest.pocket_id.unique()
    selected = ids.loc[ids.pocket_id.isin(pockets) & ids.method.isin(("qed_sa", "five_rule", "rf"))]
    merged = selected.merge(
        source[["candidate_uid", "pocket_id", "mol_idx", "ligand_rmsd_a", "relaxation_success", ENDPOINT]],
        on=["candidate_uid", "pocket_id"], how="left", validate="many_to_one",
    )
    assert len(merged) == 60 and merged.ligand_rmsd_a.notna().sum() <= 60
    merged["endpoint_positive"] = binary(merged[ENDPOINT])
    merged.sort_values(["pocket_id", "method", "mol_idx"]).to_csv(out / "figure6_all_shortlist_candidates.csv", index=False)
    rows = []
    for (pocket, method), group in merged.groupby(["pocket_id", "method"]):
        assert len(group) == 10
        values = group.sort_values("mol_idx").ligand_rmsd_a
        rows.append({
            "pocket_id": pocket,
            "method": method,
            "selected": len(group),
            "rmsd_available": int(values.notna().sum()),
            "endpoint_positive": int(group.endpoint_positive.sum()),
            "median_rmsd_a": float(values.median()),
            "rmsd_values_a": ", ".join("NA" if pd.isna(v) else f"{v:.2f}" for v in values),
        })
    pd.DataFrame(rows).to_csv(out / "figure6_shortlist_distributions.csv", index=False)


def heuristic_timing(source: pd.DataFrame, out: Path) -> None:
    pockets = source.pocket_id.drop_duplicates().head(5)
    frame = source.loc[source.pocket_id.isin(pockets)].copy()
    assert len(frame) == 500
    frame["score_qed_sa"] = frame.qed.astype(float) - 0.25 * frame.sa_proxy.astype(float)
    frame["failure_count"] = sum(
        binary(frame[col]) for col in (
            "chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure"
        )
    )
    frame["gate_pass"] = (frame.failure_count == 0).astype(int)
    frame["contact_rule"] = (
        frame.receptor_contact_atoms_4p5.fillna(0)
        - 8 * frame.receptor_clash_atoms_1p2.fillna(0)
        - 16 * frame.receptor_severe_clash_atoms_0p8.fillna(0)
    )
    settings = (
        ("contact/clash rule", ["contact_rule"], [False]),
        ("five-rule gate + QED/SA", ["gate_pass", "score_qed_sa"], [False, False]),
        ("audit-failure count + QED/SA", ["failure_count", "score_qed_sa"], [True, False]),
    )
    rows = []
    for method, columns, ascending in settings:
        times = []
        for _ in range(31):
            start = time.perf_counter()
            ranked = frame.sort_values(["pocket_id"] + columns, ascending=[True] + ascending, kind="stable")
            selected = ranked.groupby("pocket_id", sort=False).head(10)
            times.append(time.perf_counter() - start)
            assert len(selected) == 50
        rows.append({
            "operation": method,
            "candidates": len(frame),
            "pockets": len(pockets),
            "repeats": len(times),
            "median_seconds": float(np.median(times)),
            "min_seconds": float(np.min(times)),
            "max_seconds": float(np.max(times)),
            "scope": "ranking on precomputed descriptors; excludes feature extraction",
        })
    pd.DataFrame(rows).to_csv(out / "heuristic_timing.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlists", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, low_memory=False)
    source_docking_comparison(source, args.output)
    double_holdout_uncertainty(args.output)
    generator_feature_importance(source, args.output)
    figure6_shortlist_distribution(source, args.shortlists, args.manifest, args.output)
    heuristic_timing(source, args.output)
    print("Wrote", len(list(args.output.glob("*.csv"))), "analysis tables to", args.output)


if __name__ == "__main__":
    main()
