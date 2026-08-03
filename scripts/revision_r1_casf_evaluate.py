#!/usr/bin/env python3
"""Zero-shot CASF-2016 pose-ranking evaluation for ReliaMol-3D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spyrmsd import molecule as spyrmsd_molecule
from spyrmsd import rmsd as spyrmsd_rmsd


ATOM_NAMES = ["metal", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
ATOM_FEATURES = [f"atom_{name}_frac" for name in ATOM_NAMES]
CHEM_FEATURES = [
    "mol_wt", "logp", "tpsa", "qed", "rot_bonds", "ring_count", "hetero_atoms",
    "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok",
] + ATOM_FEATURES
INTERNAL_FEATURES = [
    "pose_atoms", "pose_radius_shape", "nn_q05", "nn_q10", "nn_q25", "nn_q50", "nn_q75", "nn_q90",
]
POCKET_FEATURES = [
    "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "lp_q90",
    "shell_0_2", "shell_2_4", "shell_4_6", "shell_6_8", "shell_gt8",
]
FULL_FEATURES = CHEM_FEATURES + INTERNAL_FEATURES + POCKET_FEATURES
DIRECT_RULE_VARIABLES = {"rot_bonds", "abs_formal_charge", "rare_element_count", "heavy_atoms", "sa_proxy", "descriptor_ok", "pose_radius_shape"}
DIRECT_EXCLUDED_FEATURES = [c for c in FULL_FEATURES if c not in DIRECT_RULE_VARIABLES]
NON_LABEL_PRIMITIVE_FEATURES = ["mol_wt", "logp", "tpsa", "qed", "ring_count", "hetero_atoms"] + ATOM_FEATURES
ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
METALS = {"Zn", "Mg", "Fe", "Mn", "Ca", "Na", "K", "Al"}


def mol_coords_atoms(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    conf = mol.GetConformer()
    coords, atoms = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((p.x, p.y, p.z))
        atoms.append(atom.GetSymbol())
    return np.asarray(coords, dtype=np.float32), np.asarray(atoms, dtype=str)


def receptor_coords(path: Path) -> np.ndarray:
    coords = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element.upper() == "H":
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float32)


def fixed_pocket(receptor: np.ndarray, center: np.ndarray, radius: float = 12.0) -> np.ndarray:
    distance = np.linalg.norm(receptor - center[None, :], axis=1)
    pocket = receptor[distance <= radius]
    if len(pocket) < 20:
        pocket = receptor[distance <= radius + 4.0]
    return pocket if len(pocket) else receptor


def pairwise_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
    return distance.min(axis=1)


def internal_nearest(coords: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    distance += np.eye(len(coords), dtype=np.float32) * 999.0
    return distance.min(axis=1)


def descriptor_features(mol: Chem.Mol) -> dict[str, float]:
    work = Chem.Mol(mol)
    descriptor_ok = 1.0
    try:
        Chem.SanitizeMol(work)
    except Exception:
        descriptor_ok = 0.0
    atoms = [atom.GetSymbol() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy = float(work.GetNumHeavyAtoms())
    rot = float(Lipinski.NumRotatableBonds(work))
    rings = float(rdMolDescriptors.CalcNumRings(work))
    qed = float(QED.qed(work))
    out = {
        "mol_wt": float(Descriptors.MolWt(work)),
        "logp": float(Crippen.MolLogP(work)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(work)),
        "qed": qed,
        "rot_bonds": rot,
        "ring_count": rings,
        "hetero_atoms": float(rdMolDescriptors.CalcNumHeteroatoms(work)),
        "abs_formal_charge": float(abs(Chem.GetFormalCharge(work))),
        "rare_element_count": float(sum(a not in ALLOWED_ELEMENTS for a in atoms)),
        "heavy_atoms": heavy,
        "sa_proxy": float(1.6 + 0.035 * heavy + 0.12 * rot + 0.16 * rings + 1.2 * (1 - qed)),
        "descriptor_ok": descriptor_ok,
    }
    arr = np.asarray(atoms, dtype=str)
    out["atom_metal_frac"] = float(np.mean(np.isin(arr, list(METALS)))) if len(arr) else 0.0
    for symbol in ATOM_NAMES[1:]:
        out[f"atom_{symbol}_frac"] = float(np.mean(arr == symbol)) if len(arr) else 0.0
    return out


def pose_features(mol: Chem.Mol, pocket: np.ndarray) -> dict[str, float]:
    coords, atoms = mol_coords_atoms(mol)
    lp = pairwise_min(coords, pocket)
    nn = internal_nearest(coords)
    centered = coords - coords.mean(axis=0)
    lp_q = np.quantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    nn_q = np.quantile(nn, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    out = {
        "pose_atoms": float(len(coords)),
        "pose_radius_shape": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
        "lp_q05": float(lp_q[0]), "lp_q10": float(lp_q[1]), "lp_q25": float(lp_q[2]),
        "lp_q50": float(lp_q[3]), "lp_q75": float(lp_q[4]), "lp_q90": float(lp_q[5]),
        "nn_q05": float(nn_q[0]), "nn_q10": float(nn_q[1]), "nn_q25": float(nn_q[2]),
        "nn_q50": float(nn_q[3]), "nn_q75": float(nn_q[4]), "nn_q90": float(nn_q[5]),
        "shell_0_2": float(np.mean(lp < 2.0)),
        "shell_2_4": float(np.mean((lp >= 2.0) & (lp < 4.0))),
        "shell_4_6": float(np.mean((lp >= 4.0) & (lp < 6.0))),
        "shell_6_8": float(np.mean((lp >= 6.0) & (lp < 8.0))),
        "shell_gt8": float(np.mean(lp >= 8.0)),
        "receptor_contact_atoms_4p5": int(np.sum(lp < 4.5)),
        "receptor_clash_atoms_1p2": int(np.sum(lp < 1.2)),
    }
    return out


def symmetry_rmsd(reference: Chem.Mol, pose: Chem.Mol) -> float:
    ref = spyrmsd_molecule.Molecule.from_rdkit(Chem.RemoveHs(reference, sanitize=False))
    prb = spyrmsd_molecule.Molecule.from_rdkit(Chem.RemoveHs(pose, sanitize=False))
    return float(spyrmsd_rmsd.symmrmsd(
        ref.coordinates, prb.coordinates, ref.atomicnums, prb.atomicnums,
        ref.adjacency_matrix, prb.adjacency_matrix, center=False, minimize=False,
    ))


def clean_matrix_fit(frame: pd.DataFrame, cols: list[str]) -> tuple[SimpleImputer, np.ndarray]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    return imp, imp.fit_transform(frame[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)


def train_models(train: pd.DataFrame, seed: int, jobs: int):
    models = {}
    for name, cols in {
        "full": FULL_FEATURES,
        "direct_excluded": DIRECT_EXCLUDED_FEATURES,
        "nonlabel_primitive": NON_LABEL_PRIMITIVE_FEATURES,
    }.items():
        imp, x = clean_matrix_fit(train, cols)
        y = train["reliable"].astype(int).to_numpy()
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=jobs, random_state=seed)
        rf.fit(x, y)
        models[f"rf_{name}"] = (imp, rf, cols)
        if name == "full":
            logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed))
            logreg.fit(x, y)
            models["logreg_full"] = (imp, logreg, cols)
            hgb = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, l2_regularization=0.03, random_state=seed)
            weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
            hgb.fit(x, y, sample_weight=weight)
            models["hgb_full"] = (imp, hgb, cols)
    return models


def build_pose_table(args: argparse.Namespace) -> pd.DataFrame:
    records = []
    for pose_file in sorted((args.redock_dir / "targets").glob("*/poses.sdf")):
        target = pose_file.parent.name
        reference_path = args.redock_dir / "references" / f"{target}_ligand.sdf"
        protein_path = args.casf_dir / f"{target}_protein.pdb"
        if not reference_path.exists() or not protein_path.exists():
            continue
        reference = next((m for m in Chem.SDMolSupplier(str(reference_path), removeHs=False, sanitize=False) if m is not None), None)
        if reference is None:
            continue
        receptor = receptor_coords(protein_path)
        ref_coords, _ = mol_coords_atoms(reference)
        pocket = fixed_pocket(receptor, ref_coords.mean(axis=0), args.pocket_radius)
        chemistry = descriptor_features(reference)
        for pose_index, pose in enumerate(Chem.SDMolSupplier(str(pose_file), removeHs=False, sanitize=False)):
            if pose is None:
                continue
            try:
                record = {"target": target, "pose_index": pose_index, "rmsd_a": symmetry_rmsd(reference, pose)}
                record.update(chemistry)
                record.update(pose_features(pose, pocket))
                props = pose.GetPropsAsDict()
                record["score_vina"] = -float(props.get("minimizedAffinity", np.nan))
                record["score_gnina_cnn"] = float(props.get("CNNscore", np.nan))
                record["score_gnina_affinity"] = float(props.get("CNNaffinity", np.nan))
                record["score_label_informed_rule"] = (
                    record["qed"] - 0.25 * record["sa_proxy"]
                    - 1.5 * min(record["receptor_clash_atoms_1p2"], 5)
                    + 0.05 * min(record["receptor_contact_atoms_4p5"], 20)
                )
                record["score_smooth_pose_rule"] = (
                    -abs(record["lp_q10"] - 2.2) - 0.5 * record["shell_0_2"]
                    + 0.5 * record["shell_2_4"] - 0.2 * record["shell_gt8"] + 0.1 * record["qed"]
                )
                records.append(record)
            except Exception as exc:
                records.append({"target": target, "pose_index": pose_index, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(records)


def add_model_scores(poses: pd.DataFrame, train: pd.DataFrame, seed: int, jobs: int) -> pd.DataFrame:
    out = poses.copy()
    models = train_models(train, seed, jobs)
    for name, (imp, model, cols) in models.items():
        x = imp.transform(out[cols].replace([np.inf, -np.inf], np.nan)).astype(np.float32)
        out[f"score_{name}"] = model.predict_proba(x)[:, 1]
    return out


def ranking_summary(poses: pd.DataFrame, score_cols: list[str], bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target_rows = []
    for target, group in poses.groupby("target", sort=False):
        group = group.dropna(subset=["rmsd_a"])
        if len(group) < 2:
            continue
        oracle = bool((group["rmsd_a"] < 2.0).any())
        for score_col in score_cols:
            scored = group.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
            if scored.empty:
                continue
            rho = spearmanr(scored[score_col], -scored["rmsd_a"]).statistic if len(scored) > 2 else np.nan
            row = {
                "target": target,
                "method": score_col.replace("score_", "", 1),
                "n_poses": len(scored),
                "oracle_has_rmsd2": oracle,
                "top1_success": bool(scored.iloc[0]["rmsd_a"] < 2.0),
                "top3_success": bool((scored.head(3)["rmsd_a"] < 2.0).any()),
                "top5_success": bool((scored.head(5)["rmsd_a"] < 2.0).any()),
                "top1_rmsd_a": float(scored.iloc[0]["rmsd_a"]),
                "best_top5_rmsd_a": float(scored.head(5)["rmsd_a"].min()),
                "spearman_score_neg_rmsd": float(rho) if np.isfinite(rho) else np.nan,
                "pose_auc": np.nan,
                "pose_pr_auc": np.nan,
            }
            y = (scored["rmsd_a"] < 2.0).astype(int)
            if y.nunique() == 2:
                row["pose_auc"] = float(roc_auc_score(y, scored[score_col]))
                row["pose_pr_auc"] = float(average_precision_score(y, scored[score_col]))
            per_target_rows.append(row)
    per_target = pd.DataFrame(per_target_rows)
    rows = []
    rng = np.random.default_rng(seed)
    for method, group in per_target.groupby("method", sort=False):
        oracle_group = group[group["oracle_has_rmsd2"]]
        values = group["top1_success"].astype(float).to_numpy()
        boots = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(bootstrap)])
        rows.append({
            "method": method,
            "n_targets": len(group),
            "oracle_coverage": float(group["oracle_has_rmsd2"].mean()),
            "top1_success_all": float(group["top1_success"].mean()),
            "top1_success_ci_low": float(np.quantile(boots, 0.025)),
            "top1_success_ci_high": float(np.quantile(boots, 0.975)),
            "top1_success_oracle_positive": float(oracle_group["top1_success"].mean()) if len(oracle_group) else np.nan,
            "top3_success_all": float(group["top3_success"].mean()),
            "top5_success_all": float(group["top5_success"].mean()),
            "median_top1_rmsd_a": float(group["top1_rmsd_a"].median()),
            "median_best_top5_rmsd_a": float(group["best_top5_rmsd_a"].median()),
            "mean_target_spearman": float(group["spearman_score_neg_rmsd"].mean()),
            "mean_target_pose_auc": float(group["pose_auc"].mean()),
        })
    return pd.DataFrame(rows), per_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redock-dir", type=Path, default=Path("outputs/revision_r1_casf_redocking_v0"))
    p.add_argument("--casf-dir", type=Path, default=Path("/home/test/wsk/TASC-VS/third_party/CarsiDock/data/casf2016"))
    p.add_argument("--train-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/revision_r1_casf_pose_evaluation_v0"))
    p.add_argument("--pocket-radius", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=20260630)
    p.add_argument("--jobs", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses = build_pose_table(args)
    valid = poses.dropna(subset=["rmsd_a"]).copy()
    train = pd.read_csv(args.train_frame)
    valid = add_model_scores(valid, train, args.seed, args.jobs)
    valid.to_csv(args.output_dir / "casf_pose_scores.csv", index=False)
    score_cols = [c for c in valid.columns if c.startswith("score_")]
    summary, per_target = ranking_summary(valid, score_cols, args.bootstrap, args.seed)
    summary.to_csv(args.output_dir / "casf_ranking_summary.csv", index=False)
    per_target.to_csv(args.output_dir / "casf_per_target_results.csv", index=False)
    metadata = {
        "n_pose_rows": len(valid),
        "n_targets": int(valid["target"].nunique()),
        "train_rows": len(train),
        "evaluation": "zero-shot; models fit only on DiffSBDD/CrossDocked audit labels",
        "rmsd": "symmetry-corrected receptor-frame RMSD with center=False and minimize=False",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
