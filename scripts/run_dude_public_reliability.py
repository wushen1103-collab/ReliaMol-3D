from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import FAILURE_COLUMNS, binary_metrics, failure_metrics, reranking_metrics  # noqa: E402
from reliamol3d.torch_model import train_reliability_net  # noqa: E402


ATOM_FEATURES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "metal"]
ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
CONTROLLED_VARIANTS = ["docked", "pocket_shift", "clash_inject", "collapse", "chemical_alert", "synthetic_alert"]


def parse_pdb_coords(path: Path) -> np.ndarray:
    coords = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        coords.append((float(parts[6]), float(parts[7]), float(parts[8])))
                    except ValueError:
                        continue
    return np.asarray(coords, dtype=np.float32)


def parse_mol2_coords(path: Path) -> np.ndarray:
    coords = []
    in_atoms = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>") and in_atoms:
                break
            if not in_atoms:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                coords.append((float(parts[2]), float(parts[3]), float(parts[4])))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float32)


def parse_pdbqt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    atoms = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    xyz = (float(parts[5]), float(parts[6]), float(parts[7]))
                except ValueError:
                    continue
            token = line.split()[-1] if line.split() else line[12:16].strip()
            atom = "".join(ch for ch in token if ch.isalpha()).title()
            if len(atom) > 1 and atom not in {"Cl", "Br", "Na", "Mg", "Al", "Si", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Se"}:
                atom = atom[0]
            coords.append(xyz)
            atoms.append(atom or "C")
    return np.asarray(coords, dtype=np.float32), np.asarray(atoms, dtype=str)


def pairwise_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.asarray([math.nan], dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def internal_nearest(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.asarray([math.nan], dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist += np.eye(len(coords), dtype=np.float32) * 999.0
    return dist.min(axis=1)


def ligand_sdf_path(root: Path, row: pd.Series) -> Path | None:
    pattern = f"{row.target}_{row.source_class}_{int(row.candidate_index):05d}_*.sdf"
    hits = sorted((root / "ligands" / row.target).glob(pattern))
    return hits[0] if hits else None


def descriptor_row(sdf_path: Path | None) -> dict[str, float]:
    mol = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)[0] if sdf_path and sdf_path.exists() else None
    if mol is None:
        return {
            "mol_wt": math.nan,
            "logp": math.nan,
            "tpsa": math.nan,
            "qed": math.nan,
            "rot_bonds": math.nan,
            "ring_count": math.nan,
            "hetero_atoms": math.nan,
            "abs_formal_charge": math.nan,
            "rare_element_count": math.nan,
            "heavy_atoms": math.nan,
            "sa_proxy": math.nan,
            "descriptor_ok": 0.0,
        }
    try:
        smol = Chem.Mol(mol)
        Chem.SanitizeMol(smol)
    except Exception:  # noqa: BLE001 - descriptor failure becomes a chemical signal
        smol = mol
    elements = [atom.GetSymbol() for atom in smol.GetAtoms()]
    heavy_atoms = float(smol.GetNumHeavyAtoms())
    rot_bonds = float(Lipinski.NumRotatableBonds(smol))
    ring_count = float(rdMolDescriptors.CalcNumRings(smol))
    qed = float(QED.qed(smol)) if heavy_atoms else math.nan
    sa_proxy = 1.6 + 0.035 * heavy_atoms + 0.12 * rot_bonds + 0.16 * ring_count + 1.2 * (1.0 - (qed if not math.isnan(qed) else 0.5))
    return {
        "mol_wt": float(Descriptors.MolWt(smol)),
        "logp": float(Crippen.MolLogP(smol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(smol)),
        "qed": qed,
        "rot_bonds": rot_bonds,
        "ring_count": ring_count,
        "hetero_atoms": float(sum(1 for atom in smol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
        "abs_formal_charge": float(abs(Chem.GetFormalCharge(smol))),
        "rare_element_count": float(sum(1 for symbol in elements if symbol not in ALLOWED_ELEMENTS)),
        "heavy_atoms": heavy_atoms,
        "sa_proxy": float(sa_proxy),
        "descriptor_ok": 1.0,
    }


def unit_vector(seed: int, preferred: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(preferred)
    if norm > 1e-6:
        return preferred / norm
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=3)
    return vec / max(np.linalg.norm(vec), 1e-6)


def make_variant_coords(
    variant: str,
    coords: np.ndarray,
    crystal_center: np.ndarray,
    receptor_center: np.ndarray,
    pocket: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = coords.mean(axis=0)
    if variant in {"docked", "chemical_alert", "synthetic_alert"}:
        return coords.copy()
    if variant == "pocket_shift":
        direction = unit_vector(seed, center - receptor_center)
        return (coords + direction * rng.uniform(14.0, 20.0) + rng.normal(0.0, 0.10, coords.shape)).astype(np.float32)
    if variant == "clash_inject":
        anchor = pocket[int(seed % len(pocket))]
        return (coords - center + anchor + rng.normal(0.0, 0.18, coords.shape)).astype(np.float32)
    if variant == "collapse":
        return ((coords - center) * 0.12 + center + rng.normal(0.0, 0.03, coords.shape)).astype(np.float32)
    raise ValueError(f"unknown controlled variant: {variant}")


def score_proxy(base_vina: float, variant: str, contact_count: int, clash_count: int, center_to_crystal: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if variant == "docked":
        return float(base_vina)
    if variant == "pocket_shift":
        return float(base_vina + 2.8 + 0.12 * center_to_crystal + rng.normal(0.0, 0.25))
    if variant == "clash_inject":
        return float(base_vina - 1.8 - 0.03 * contact_count + rng.normal(0.0, 0.25))
    if variant == "collapse":
        return float(base_vina + 1.2 + 0.05 * clash_count + rng.normal(0.0, 0.25))
    if variant == "chemical_alert":
        return float(base_vina + rng.normal(0.0, 0.20))
    if variant == "synthetic_alert":
        return float(base_vina + rng.normal(0.0, 0.20))
    return float(base_vina)


def geometry_features(coords: np.ndarray, atoms: np.ndarray, receptor: np.ndarray, pocket: np.ndarray, crystal_center: np.ndarray) -> dict[str, float]:
    lp = pairwise_min(coords, pocket)
    lr = pairwise_min(coords, receptor)
    nn = internal_nearest(coords)
    center = coords.mean(axis=0)
    centered = coords - center
    center_to_crystal = float(np.linalg.norm(center - crystal_center))
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    lp_q = np.nanquantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    nn_q = np.nanquantile(nn, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    metal_frac = float(np.mean(np.isin(atoms, ["Zn", "Mg", "Fe", "Mn", "Ca", "Na", "K", "Al"]))) if len(atoms) else 0.0
    out = {
        "pose_atoms": float(len(coords)),
        "pose_radius": radius,
        "center_to_crystal": center_to_crystal,
        "min_dist_to_pocket": float(np.nanmin(lp)),
        "min_dist_to_receptor": float(np.nanmin(lr)),
        "mean_dist_to_pocket": float(np.nanmean(lp)),
        "pocket_contact_atoms_4p5": float(np.nansum(lp <= 4.5)),
        "pocket_close_atoms_3p5": float(np.nansum(lp <= 3.5)),
        "receptor_clash_atoms_1p2": float(np.nansum(lr < 1.2)),
        "receptor_severe_clash_atoms_0p8": float(np.nansum(lr < 0.8)),
        "internal_min_dist": float(np.nanmin(nn)),
        "internal_close_atoms_0p65": float(np.nansum(nn < 0.65)),
        "lp_q05": float(lp_q[0]),
        "lp_q10": float(lp_q[1]),
        "lp_q25": float(lp_q[2]),
        "lp_q50": float(lp_q[3]),
        "lp_q75": float(lp_q[4]),
        "lp_q90": float(lp_q[5]),
        "nn_q05": float(nn_q[0]),
        "nn_q10": float(nn_q[1]),
        "nn_q25": float(nn_q[2]),
        "nn_q50": float(nn_q[3]),
        "nn_q75": float(nn_q[4]),
        "nn_q90": float(nn_q[5]),
        "shell_0_2": float(np.mean(lp < 2.0)),
        "shell_2_4": float(np.mean((lp >= 2.0) & (lp < 4.0))),
        "shell_4_6": float(np.mean((lp >= 4.0) & (lp < 6.0))),
        "shell_6_8": float(np.mean((lp >= 6.0) & (lp < 8.0))),
        "shell_gt8": float(np.mean(lp >= 8.0)),
        "atom_metal_frac": metal_frac,
    }
    for atom in ATOM_FEATURES:
        if atom == "metal":
            continue
        out[f"atom_{atom}_frac"] = float(np.mean(atoms == atom)) if len(atoms) else 0.0
    return out


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def build_dataset(input_root: Path, seed: int) -> pd.DataFrame:
    labels = pd.read_csv(input_root / "reports" / "docking_reliability_labels.csv")
    labels = labels[labels["status"].eq("success")].copy()
    receptor_cache: dict[str, dict[str, np.ndarray]] = {}
    descriptor_cache: dict[tuple[str, str, int], dict[str, float]] = {}
    rows = []

    for row in labels.itertuples(index=False):
        target = str(row.target)
        if target not in receptor_cache:
            receptor = parse_pdb_coords(input_root / "work" / "receptors" / target / "receptor_fixed.pdb")
            crystal = parse_mol2_coords(input_root / "raw" / target / "crystal_ligand.mol2")
            crystal_center = crystal.mean(axis=0)
            receptor_center = receptor.mean(axis=0)
            pocket = receptor[pairwise_min(receptor, crystal) <= 6.0]
            if len(pocket) == 0:
                pocket = receptor
            receptor_cache[target] = {"receptor": receptor, "crystal_center": crystal_center, "receptor_center": receptor_center, "pocket": pocket}

        pose_path = Path(str(row.pose_pdbqt))
        coords, atoms = parse_pdbqt(pose_path)
        if len(coords) == 0:
            continue
        key = (target, str(row.source_class), int(row.candidate_index))
        if key not in descriptor_cache:
            sdf_path = ligand_sdf_path(input_root, pd.Series({"target": target, "source_class": row.source_class, "candidate_index": row.candidate_index}))
            descriptor_cache[key] = descriptor_row(sdf_path)
        desc_base = descriptor_cache[key]

        rec = receptor_cache[target]
        base_id = f"{target}_{row.source_class}_{int(row.candidate_index):05d}"
        for variant in CONTROLLED_VARIANTS:
            v_seed = seed + deterministic_id(f"{base_id}:{variant}")
            v_coords = make_variant_coords(variant, coords, rec["crystal_center"], rec["receptor_center"], rec["pocket"], v_seed)
            geom = geometry_features(v_coords, atoms, rec["receptor"], rec["pocket"], rec["crystal_center"])
            desc = dict(desc_base)
            chemical_failure = bool(desc["descriptor_ok"] < 0.5 or desc["rare_element_count"] > 0 or desc["abs_formal_charge"] > 2)
            synthetic_failure = bool(desc["sa_proxy"] > 6.8 or desc["heavy_atoms"] > 70 or desc["rot_bonds"] > 22)
            if variant == "chemical_alert":
                desc["rare_element_count"] = float(desc.get("rare_element_count", 0.0) + 2.0)
                desc["abs_formal_charge"] = max(float(desc.get("abs_formal_charge", 0.0)), 3.0)
                desc["qed"] = max(0.01, float(desc["qed"]) - 0.20)
                chemical_failure = True
            if variant == "synthetic_alert":
                desc["sa_proxy"] = float(desc["sa_proxy"] + 2.3)
                desc["rot_bonds"] = float(desc["rot_bonds"] + 8.0)
                desc["qed"] = max(0.01, float(desc["qed"]) - 0.25)
                synthetic_failure = True

            geometric_failure = bool(geom["receptor_severe_clash_atoms_0p8"] > 0 or geom["internal_close_atoms_0p65"] > 0 or geom["pose_radius"] < 0.8)
            pocket_failure = bool(geom["center_to_crystal"] > 8.0 or geom["pocket_contact_atoms_4p5"] <= 0)
            vina_score = score_proxy(float(row.vina_score), variant, int(geom["pocket_contact_atoms_4p5"]), int(geom["receptor_clash_atoms_1p2"]), geom["center_to_crystal"], v_seed)
            scoring_failure = bool((variant == "clash_inject" and vina_score < float(row.vina_score)) or (variant == "pocket_shift" and vina_score < -7.5))
            if variant == "collapse":
                geometric_failure = True
            if variant == "pocket_shift":
                pocket_failure = True
            if variant == "clash_inject":
                geometric_failure = True

            failure_flags = {
                "chemical_failure": chemical_failure,
                "geometric_failure": geometric_failure,
                "pocket_failure": pocket_failure,
                "synthetic_failure": synthetic_failure,
                "scoring_failure": scoring_failure,
            }
            out = {
                "target": target,
                "pocket_id": target,
                "base_id": base_id,
                "source_class": row.source_class,
                "candidate_index": int(row.candidate_index),
                "candidate_id": row.candidate_id,
                "controlled_variant": variant,
                "chemotype_id": deterministic_id(str(row.candidate_id)) % 100000,
                "native_vina_score": float(row.vina_score),
                "vina_score": float(vina_score),
                "vina_delta_from_native": float(vina_score - float(row.vina_score)),
                **desc,
                **geom,
                **failure_flags,
            }
            out["reliable"] = int(not any(out[col] for col in FAILURE_COLUMNS))
            rows.append(out)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    score = ["vina_score"]
    ligand = [
        "mol_wt",
        "logp",
        "tpsa",
        "qed",
        "rot_bonds",
        "ring_count",
        "hetero_atoms",
        "abs_formal_charge",
        "rare_element_count",
        "heavy_atoms",
        "sa_proxy",
        "descriptor_ok",
    ] + [f"atom_{a}_frac" for a in ATOM_FEATURES]
    pose_shape = [
        "pose_atoms",
        "pose_radius",
        "lp_q05",
        "lp_q10",
        "lp_q25",
        "lp_q50",
        "lp_q75",
        "lp_q90",
        "nn_q05",
        "nn_q10",
        "nn_q25",
        "nn_q50",
        "nn_q75",
        "nn_q90",
        "shell_0_2",
        "shell_2_4",
        "shell_4_6",
        "shell_6_8",
        "shell_gt8",
    ]
    direct = [
        "center_to_crystal",
        "min_dist_to_pocket",
        "min_dist_to_receptor",
        "mean_dist_to_pocket",
        "pocket_contact_atoms_4p5",
        "pocket_close_atoms_3p5",
        "receptor_clash_atoms_1p2",
        "receptor_severe_clash_atoms_0p8",
        "internal_min_dist",
        "internal_close_atoms_0p65",
    ]
    variants = {
        "score_only": score,
        "ligand_descriptor": ligand,
        "pose_shape": pose_shape,
        "reliamol_noleak": score + ligand + pose_shape,
        "leak_upper_bound": score + ligand + pose_shape + direct,
    }
    if variant not in variants:
        raise ValueError(f"Unknown feature variant: {variant}")
    return variants[variant]


def clean_matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = train[cols].replace([np.inf, -np.inf], np.nan)
    te = test[cols].replace([np.inf, -np.inf], np.nan)
    med = tr.median(numeric_only=True).fillna(0.0)
    return tr.fillna(med).to_numpy(np.float32), te.fillna(med).to_numpy(np.float32)


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = score.argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def model_cfg(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "focal_gamma": 1.5,
        "failure_loss_weight": 0.35,
        "calibration_loss_weight": 0.08,
    }


def fit_mlp(train: pd.DataFrame, test: pd.DataFrame, features: list[str], cfg: dict, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return train_reliability_net(
        x_train,
        train["reliable"].to_numpy(np.float32),
        train[FAILURE_COLUMNS].to_numpy(np.float32),
        x_test,
        cfg,
        device,
        seed,
    )


def eval_fold(frame: pd.DataFrame, heldout: str, seed: int, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[~frame["target"].eq(heldout)].copy()
    test = frame[frame["target"].eq(heldout)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Empty fold for heldout={heldout}")
    cfg = model_cfg(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scores: dict[str, np.ndarray] = {
        "vina_rank": rank_to_prob(-test["vina_score"].to_numpy(float)),
        "native_vina_rank": rank_to_prob(-test["native_vina_score"].to_numpy(float)),
        "qed_sa_vina": rank_to_prob(-test["vina_score"].to_numpy(float) + 0.5 * test["qed"].to_numpy(float) - 0.25 * test["sa_proxy"].to_numpy(float)),
    }
    fold_pred = test.copy()

    for variant in ["score_only", "ligand_descriptor", "pose_shape", "reliamol_noleak", "leak_upper_bound"]:
        rel_prob, fail_prob = fit_mlp(train, test, feature_columns(frame, variant), cfg, seed + deterministic_id(f"{heldout}:{variant}"), device)
        method = f"mlp_{variant}"
        scores[method] = rel_prob
        fold_pred[f"score_{method}"] = rel_prob
        if variant == "reliamol_noleak":
            for i, label in enumerate(FAILURE_COLUMNS):
                fold_pred[f"{label}_prob"] = fail_prob[:, i]

    noleak_features = feature_columns(frame, "reliamol_noleak")
    x_train, x_test = clean_matrix(train, test, noleak_features)
    rf = RandomForestClassifier(n_estimators=320, min_samples_leaf=3, n_jobs=16, random_state=seed, class_weight="balanced")
    rf.fit(x_train, train["reliable"])
    scores["rf_noleak"] = rf.predict_proba(x_test)[:, 1]
    hgb = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, l2_regularization=0.03, random_state=seed)
    hgb.fit(x_train, train["reliable"])
    scores["hgb_noleak"] = hgb.predict_proba(x_test)[:, 1]

    metric_rows = []
    for method, prob in scores.items():
        row = binary_metrics(test["reliable"].to_numpy(), prob)
        row.update(
            {
                "heldout_target": heldout,
                "seed": seed,
                "method": method,
                "n_train": len(train),
                "n_test": len(test),
                "test_positive_rate": float(test["reliable"].mean()),
                "device": device,
            }
        )
        metric_rows.append(row)

    rerank_rows = []
    score_cols = {
        "vina_rank": -fold_pred["vina_score"],
        "native_vina_rank": -fold_pred["native_vina_score"],
        "mlp_score_only": fold_pred["score_mlp_score_only"],
        "mlp_ligand_descriptor": fold_pred["score_mlp_ligand_descriptor"],
        "mlp_pose_shape": fold_pred["score_mlp_pose_shape"],
        "mlp_reliamol_noleak": fold_pred["score_mlp_reliamol_noleak"],
        "mlp_leak_upper_bound": fold_pred["score_mlp_leak_upper_bound"],
    }
    work = fold_pred.copy()
    for name, values in score_cols.items():
        work[f"rank_{name}"] = values.to_numpy(float)
        rr = reranking_metrics(work, f"rank_{name}", args.topk_fracs, group_col="target")
        rr["heldout_target"] = heldout
        rr["seed"] = seed
        rerank_rows.append(rr)

    failure_df = pd.DataFrame([failure_metrics(fold_pred, [f"{c}_prob" for c in FAILURE_COLUMNS])])
    failure_df["heldout_target"] = heldout
    failure_df["seed"] = seed

    fold_dir = output_dir / "folds" / f"seed_{seed}" / heldout
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(fold_dir / "metrics.csv", index=False)
    pd.concat(rerank_rows, ignore_index=True).to_csv(fold_dir / "reranking.csv", index=False)
    failure_df.to_csv(fold_dir / "failure_diagnosis.csv", index=False)
    fold_pred.to_csv(fold_dir / "test_predictions.csv", index=False)
    return pd.DataFrame(metric_rows), pd.concat(rerank_rows, ignore_index=True), failure_df


def summarize(metrics: pd.DataFrame, reranking: pd.DataFrame, failures: pd.DataFrame, output_dir: Path) -> None:
    metric_summary = (
        metrics.groupby("method", sort=False)[["roc_auc", "pr_auc", "f1", "mcc", "ece"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metric_summary.columns = [".".join(col).strip(".") for col in metric_summary.columns.to_flat_index()]
    rerank_focus = reranking[reranking["topk_frac"].isin([0.1, 0.2])].copy()
    rerank_summary = (
        rerank_focus.groupby(["score", "topk_frac"], sort=False)[["reliable_rate", "posebusters_pass_proxy", "clash_free_proxy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    rerank_summary.columns = [".".join(col).strip(".") for col in rerank_summary.columns.to_flat_index()]
    failure_summary = failures.mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})

    metrics.to_csv(output_dir / "lot_metrics_all.csv", index=False)
    metric_summary.to_csv(output_dir / "lot_metrics_summary.csv", index=False)
    reranking.to_csv(output_dir / "lot_reranking_all.csv", index=False)
    rerank_summary.to_csv(output_dir / "lot_reranking_summary.csv", index=False)
    failures.to_csv(output_dir / "lot_failure_diagnosis_all.csv", index=False)
    failure_summary.to_csv(output_dir / "lot_failure_diagnosis_summary.csv", index=False)

    lines = [
        "# DUD-E Public Controlled-Failure Reliability",
        "",
        "This run uses public docked DUD-E poses plus controlled failure variants. It is a stress test, not a generated-molecule benchmark.",
        "",
        "## Metrics Summary",
        "",
        metric_summary.to_csv(index=False),
        "## Reranking Summary",
        "",
        rerank_summary.to_csv(index=False),
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a public DUD-E controlled-failure reliability benchmark.")
    parser.add_argument("--input-root", type=Path, default=Path("outputs/dude_docking_public6"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dude_public_reliability"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_dataset(args.input_root, args.seeds[0])
    frame.to_csv(args.output_dir / "controlled_candidate_labels.csv", index=False)
    label_summary = frame.groupby(["target", "controlled_variant"])[["reliable"] + FAILURE_COLUMNS].mean().reset_index()
    label_summary.to_csv(args.output_dir / "label_summary.csv", index=False)
    variant_summary = frame.groupby("controlled_variant")[["reliable"] + FAILURE_COLUMNS].mean().reset_index()
    variant_summary.to_csv(args.output_dir / "variant_summary.csv", index=False)

    all_metrics = []
    all_reranking = []
    all_failures = []
    targets = sorted(frame["target"].unique())
    for seed in args.seeds:
        for heldout in targets:
            metrics, reranking, failures = eval_fold(frame, heldout, seed, args, args.output_dir)
            all_metrics.append(metrics)
            all_reranking.append(reranking)
            all_failures.append(failures)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    reranking_df = pd.concat(all_reranking, ignore_index=True)
    failures_df = pd.concat(all_failures, ignore_index=True)
    summarize(metrics_df, reranking_df, failures_df, args.output_dir)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_root": str(args.input_root),
                "seeds": args.seeds,
                "targets": targets,
                "controlled_variants": CONTROLLED_VARIANTS,
                "rows": int(len(frame)),
                "positive_rate": float(frame["reliable"].mean()),
                "note": "Public DUD-E docked poses plus controlled failure variants; no internal unpublished assets.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Variant summary")
    print(variant_summary.to_string(index=False))
    print("\nMetrics summary")
    print(pd.read_csv(args.output_dir / "lot_metrics_summary.csv").to_string(index=False))
    print("\nReranking summary")
    print(pd.read_csv(args.output_dir / "lot_reranking_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
