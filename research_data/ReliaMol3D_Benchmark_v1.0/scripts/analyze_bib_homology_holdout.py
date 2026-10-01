#!/usr/bin/env python3
"""Evaluate endpoint triage under protein-domain and sequence-cluster holdout.

The analysis keeps every homologous receptor cluster in a single fold. It uses
the completed DiffSBDD candidate-level endpoint table and does not rerun
molecular relaxation.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Data.PDBData import protein_letters_3to1_extended
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("RELIAMOL_OUTPUT_ROOT", Path.cwd()))
FRAME_PATH = ROOT / "outputs/bib_physical_endpoint_analysis_v0/relaxation_endpoint_frame.csv"
META_PATH = ROOT / "outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports/pocket_metadata.csv"
PDB_CACHE = Path(os.environ.get("RELIAMOL_PDB_CACHE", ROOT / "data" / "rcsb_pdb_cache"))
OUT = ROOT / "outputs/bib_homology_holdout_v0"
TARGET = "stable_r2p0_c0p5_with_pb"
SEEDS = (11, 22, 33)
THRESHOLDS = (0.30, 0.40, 0.50)

ATOM_FEATURES = [
    "atom_metal_frac", "atom_C_frac", "atom_N_frac", "atom_O_frac", "atom_S_frac",
    "atom_P_frac", "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
]
CHEMISTRY = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM_FEATURES
INTERNAL = [
    "pose_atoms", "pose_radius_shape", "nn_q05", "nn_q10", "nn_q25", "nn_q50",
    "nn_q75", "nn_q90",
]
POCKET = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90", "shell_0_2",
    "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
DIRECT_RULE_VARIABLES = {
    "rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms",
    "sa_proxy", "descriptor_ok", "pose_radius_shape",
}
FEATURES = [
    column for column in CHEMISTRY + INTERNAL + POCKET
    if column not in DIRECT_RULE_VARIABLES
]


def parse_seqres(path: Path, chain: str) -> str:
    residues: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("SEQRES") and line[11:12].strip() == chain:
                residues.extend(line[19:70].split())
    sequence = "".join(protein_letters_3to1_extended.get(residue.upper(), "X") for residue in residues)
    if sequence:
        return sequence

    atom_residues: list[str] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21:22].strip() != chain:
                continue
            key = (line[22:27], line[17:20].strip())
            if key in seen:
                continue
            seen.add(key)
            atom_residues.append(key[1])
    return "".join(protein_letters_3to1_extended.get(residue.upper(), "X") for residue in atom_residues)


def sequence_identity(first: str, second: str, aligner: Align.PairwiseAligner) -> float:
    if not first or not second:
        return float("nan")
    alignment = aligner.align(first, second)[0]
    matches = 0
    for (a0, a1), (b0, b1) in zip(alignment.aligned[0], alignment.aligned[1]):
        matches += sum(x == y for x, y in zip(first[a0:a1], second[b0:b1]))
    return matches / max(len(first), len(second))


def connected_components(items: list[str], matrix: np.ndarray, threshold: float) -> dict[str, str]:
    parent = {item: item for item in items}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            if matrix[i, j] >= threshold:
                union(left, items[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[find(item)].append(item)
    ordered = sorted(groups.values(), key=lambda group: (-len(group), group[0]))
    return {item: f"seq{int(threshold * 100):02d}_{index:03d}" for index, group in enumerate(ordered, 1) for item in group}


def assign_group_folds(pocket_meta: pd.DataFrame, group_column: str, seed: int, folds: int = 5) -> dict[str, int]:
    grouped = pocket_meta.groupby(group_column)["pocket_id"].apply(list).to_dict()
    rng = np.random.default_rng(seed)
    tie = {group: float(rng.random()) for group in grouped}
    ordered = sorted(grouped, key=lambda group: (-len(grouped[group]), tie[group]))
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    fold_sizes = [0] * folds
    for group in ordered:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].append(group)
        fold_sizes[fold] += len(grouped[group])
    assignment: dict[str, int] = {}
    for fold, groups in enumerate(fold_groups):
        for group in groups:
            for pocket in grouped[group]:
                assignment[pocket] = fold
    return assignment


def build_model(model_name: str, seed: int):
    if model_name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed),
        )
    if model_name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=260,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(model_name)


def topk_rows(frame: pd.DataFrame, score: str, method: str, split: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    for pocket_id, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(math.ceil(0.10 * len(group))))
        selected = group.nlargest(k, score)
        rows.append({
            "split": split,
            "seed": seed,
            "method": method,
            "pocket_id": pocket_id,
            "n_selected": len(selected),
            "endpoint_positive_rate": float(selected[TARGET].mean()),
        })
    return rows


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def summarize(per_pocket: pd.DataFrame, candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    averaged = per_pocket.groupby(["split", "method", "pocket_id"], as_index=False)["endpoint_positive_rate"].mean()
    rows: list[dict] = []
    for split in averaged["split"].unique():
        subset = averaged[averaged["split"].eq(split)]
        wide = subset.pivot(index="pocket_id", columns="method", values="endpoint_positive_rate")
        for method in ["qed_sa", "five_rule", "logistic", "rf"]:
            if method == "qed_sa":
                method_values = wide["qed_sa"].dropna()
                reference_values = method_values
                delta = np.zeros(len(method_values), dtype=float)
                low, high = float("nan"), float("nan")
            else:
                pair = wide[[method, "qed_sa"]].dropna()
                method_values = pair[method]
                reference_values = pair["qed_sa"]
                delta = (method_values - reference_values).to_numpy(float)
                low, high = bootstrap(delta, 20260831 + len(rows))
            metric = candidate_metrics[
                candidate_metrics["split"].eq(split) & candidate_metrics["method"].eq(method)
            ]
            rows.append({
                "split": split,
                "method": method,
                "n_pockets": int(len(method_values)),
                "endpoint_positive_rate": float(method_values.mean()),
                "qed_sa_rate": float(reference_values.mean()),
                "delta_vs_qed_sa": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "roc_auc": float(metric["roc_auc"].mean()) if len(metric) else float("nan"),
                "brier": float(metric["brier"].mean()) if len(metric) else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FRAME_PATH)
    frame[TARGET] = frame[TARGET].astype(int)
    frame["score_qed_sa"] = frame["qed"].astype(float) - 0.25 * frame["sa_proxy"].astype(float)
    frame["direct_failure_count"] = frame[
        ["chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure"]
    ].astype(int).sum(axis=1)
    frame["score_five_rule"] = (frame["direct_failure_count"].eq(0).astype(float) * 1000.0) + frame["score_qed_sa"]

    metadata = pd.read_csv(META_PATH)
    metadata = metadata[["pdb_id", "chain_id", "pocket_id", "tar_member"]].drop_duplicates("pocket_id")
    metadata["crossdocked_domain"] = metadata["tar_member"].astype(str).str.split("/").str[0]
    sequences = []
    for row in metadata.itertuples(index=False):
        sequence = parse_seqres(PDB_CACHE / f"{str(row.pdb_id).lower()}.pdb", str(row.chain_id))
        sequences.append(sequence)
    metadata["sequence"] = sequences
    metadata["sequence_length"] = metadata["sequence"].str.len()
    if (metadata["sequence_length"] == 0).any():
        missing = metadata.loc[metadata["sequence_length"].eq(0), ["pdb_id", "chain_id"]].to_dict("records")
        raise RuntimeError(f"Missing receptor sequences: {missing}")

    pockets = metadata["pocket_id"].tolist()
    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    identity = np.eye(len(pockets), dtype=float)
    for i in range(len(pockets)):
        for j in range(i + 1, len(pockets)):
            value = sequence_identity(metadata.at[i, "sequence"], metadata.at[j, "sequence"], aligner)
            identity[i, j] = identity[j, i] = value

    identity_long = []
    for i in range(len(pockets)):
        for j in range(i + 1, len(pockets)):
            identity_long.append({"pocket_i": pockets[i], "pocket_j": pockets[j], "sequence_identity": identity[i, j]})
    pd.DataFrame(identity_long).to_csv(OUT / "pairwise_sequence_identity.csv", index=False)

    split_columns = {"domain_group": "crossdocked_domain"}
    cluster_rows = []
    for threshold in THRESHOLDS:
        column = f"sequence_cluster_{int(threshold * 100)}"
        metadata[column] = metadata["pocket_id"].map(connected_components(pockets, identity, threshold))
        split_columns[f"sequence_{int(threshold * 100)}"] = column
        sizes = metadata.groupby(column).size()
        cluster_rows.append({
            "split": f"sequence_{int(threshold * 100)}",
            "identity_threshold": threshold,
            "n_clusters": int(len(sizes)),
            "largest_cluster_pockets": int(sizes.max()),
            "median_cluster_pockets": float(sizes.median()),
            "singleton_clusters": int((sizes == 1).sum()),
        })

    metadata.drop(columns="sequence").to_csv(OUT / "pocket_sequence_metadata.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "sequence_cluster_summary.csv", index=False)

    frame = frame.merge(metadata[["pocket_id"] + list(split_columns.values())], on="pocket_id", how="inner")
    per_pocket_rows: list[dict] = []
    metric_rows: list[dict] = []
    assignment_rows: list[dict] = []

    for split_name, group_column in split_columns.items():
        for seed in SEEDS:
            assignment = assign_group_folds(metadata, group_column, seed)
            for pocket, fold in assignment.items():
                assignment_rows.append({
                    "split": split_name,
                    "seed": seed,
                    "pocket_id": pocket,
                    "group_id": metadata.loc[metadata["pocket_id"].eq(pocket), group_column].iloc[0],
                    "fold": fold,
                })
            predicted = frame[["pocket_id", TARGET, "score_qed_sa", "score_five_rule"]].copy()
            predicted["score_logistic"] = np.nan
            predicted["score_rf"] = np.nan
            for fold in range(5):
                test_pockets = {pocket for pocket, value in assignment.items() if value == fold}
                test_mask = frame["pocket_id"].isin(test_pockets)
                train = frame.loc[~test_mask]
                test = frame.loc[test_mask]
                for model_name in ("logistic", "rf"):
                    model = build_model(model_name, seed)
                    model.fit(train[FEATURES].replace([np.inf, -np.inf], np.nan), train[TARGET])
                    predicted.loc[test_mask, f"score_{model_name}"] = model.predict_proba(
                        test[FEATURES].replace([np.inf, -np.inf], np.nan)
                    )[:, 1]

            per_pocket_rows.extend(topk_rows(predicted, "score_qed_sa", "qed_sa", split_name, seed))
            per_pocket_rows.extend(topk_rows(predicted, "score_five_rule", "five_rule", split_name, seed))
            for model_name in ("logistic", "rf"):
                per_pocket_rows.extend(topk_rows(predicted, f"score_{model_name}", model_name, split_name, seed))
                valid = predicted[f"score_{model_name}"].notna()
                metric_rows.append({
                    "split": split_name,
                    "seed": seed,
                    "method": model_name,
                    "roc_auc": roc_auc_score(predicted.loc[valid, TARGET], predicted.loc[valid, f"score_{model_name}"]),
                    "brier": brier_score_loss(predicted.loc[valid, TARGET], predicted.loc[valid, f"score_{model_name}"]),
                })

    per_pocket = pd.DataFrame(per_pocket_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize(per_pocket, metrics)
    assignments = pd.DataFrame(assignment_rows)
    per_pocket.to_csv(OUT / "homology_holdout_per_pocket.csv", index=False)
    metrics.to_csv(OUT / "homology_holdout_candidate_metrics.csv", index=False)
    summary.to_csv(OUT / "homology_holdout_summary.csv", index=False)
    assignments.to_csv(OUT / "homology_split_assignments.csv", index=False)

    manifest = {
        "candidate_rows": int(len(frame)),
        "pockets": int(frame["pocket_id"].nunique()),
        "features": FEATURES,
        "seeds": list(SEEDS),
        "folds": 5,
        "sequence_identity_definition": "global aligned exact matches divided by the longer chain length",
        "split_columns": split_columns,
    }
    (OUT / "homology_holdout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(cluster_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
