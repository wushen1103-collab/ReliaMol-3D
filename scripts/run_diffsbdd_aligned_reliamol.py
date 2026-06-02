from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402
from reliamol3d.torch_model import train_reliability_net  # noqa: E402


ATOM_FEATURES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "metal"]
PUBLIC_FAILURE_COLUMNS = [
    "chemical_failure",
    "geometric_failure",
    "pocket_failure",
    "clash_failure",
    "synthetic_failure",
]


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def safe_name(text: str, max_len: int = 140) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:max_len].strip("._") or "item"


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def parse_pdb_coords(path: Path, chain_id: str) -> np.ndarray:
    coords = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            if chain_id and len(line) > 21 and line[21].strip() and line[21].strip() != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
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
    if not coords and chain_id:
        return parse_pdb_coords(path, "")
    return np.asarray(coords, dtype=np.float32)


def mol_coords_atoms(sdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mol = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)[0]
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=np.float32), np.asarray([], dtype=str)
    conf = mol.GetConformer()
    coords = []
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
        atoms.append(atom.GetSymbol())
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


def fixed_pocket(receptor: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if len(receptor) == 0:
        return receptor
    dist = np.linalg.norm(receptor - center[None, :], axis=1)
    pocket = receptor[dist <= radius]
    if len(pocket) < 20:
        pocket = receptor[dist <= radius + 4.0]
    return pocket if len(pocket) else receptor


def geometry_features(coords: np.ndarray, atoms: np.ndarray, pocket: np.ndarray) -> dict[str, float]:
    if len(coords) == 0:
        return {
            key: math.nan
            for key in [
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
        }
    lp = pairwise_min(coords, pocket)
    nn = internal_nearest(coords)
    center = coords.mean(axis=0)
    centered = coords - center
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    lp_q = np.nanquantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    nn_q = np.nanquantile(nn, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    metal_frac = float(np.mean(np.isin(atoms, ["Zn", "Mg", "Fe", "Mn", "Ca", "Na", "K", "Al"]))) if len(atoms) else 0.0
    out = {
        "pose_atoms": float(len(coords)),
        "pose_radius": radius,
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


def feature_columns(variant: str) -> list[str]:
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
    variants = {
        "score_only": score,
        "ligand_descriptor": ligand,
        "pose_shape": pose_shape,
        "reliamol_noleak": score + ligand + pose_shape,
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
    order = np.asarray(score).argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def model_cfg(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "focal_gamma": 1.5,
        "failure_loss_weight": 0.25,
        "calibration_loss_weight": 0.05,
    }


def fit_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cfg: dict,
    seed: int,
    device: str,
) -> np.ndarray:
    x_train, x_test = clean_matrix(train, test, features)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    rel_prob, _ = train_reliability_net(
        x_train,
        train["reliable"].to_numpy(np.float32),
        train[PUBLIC_FAILURE_COLUMNS].to_numpy(np.float32),
        x_test,
        cfg,
        device,
        seed,
    )
    return rel_prob


def build_frame(args: argparse.Namespace) -> pd.DataFrame:
    root = Path.cwd()
    raw = pd.read_csv(args.vina_dir / "vina_score_results.csv")
    raw = raw[raw["vina_status"].eq("success")].copy()
    raw["vina_score"] = raw["vina_affinity"].astype(float)
    raw["descriptor_ok"] = raw["sanitize_ok"].astype(float)
    raw["reliable"] = raw["public_generated_reliable_proxy"].astype(int)
    raw["chemical_failure"] = raw["chemical_failure_proxy"].astype(bool)
    raw["geometric_failure"] = raw["geometric_failure_proxy"].astype(bool)
    raw["pocket_failure"] = raw["pocket_failure_proxy"].astype(bool)
    raw["clash_failure"] = raw["clash_failure_proxy"].astype(bool)
    raw["synthetic_failure"] = raw["synthetic_failure_proxy"].astype(bool)
    raw["chemotype_id"] = [deterministic_id(f"{pid}:{idx}") % 100000 for pid, idx in zip(raw["pocket_id"], raw["mol_idx"])]

    receptor_cache: dict[str, np.ndarray] = {}
    pocket_cache: dict[str, np.ndarray] = {}
    rows = []
    for pocket_id, group in raw.groupby("pocket_id", sort=False):
        pdb_id = str(group["pdb_id"].iloc[0]).lower()
        chain_id = str(group["chain_id"].iloc[0])
        receptor_key = f"{pdb_id}:{chain_id}"
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = parse_pdb_coords(args.pdb_cache / f"{pdb_id}.pdb", chain_id)
        center = group[["center_x", "center_y", "center_z"]].median().to_numpy(np.float32)
        pocket_cache[pocket_id] = fixed_pocket(receptor_cache[receptor_key], center, args.pocket_radius)

    for row in raw.itertuples(index=False):
        row_dict = row._asdict()
        sdf_path = resolve_path(str(row_dict["single_sdf"]), root)
        coords, atoms = mol_coords_atoms(sdf_path)
        geom = geometry_features(coords, atoms, pocket_cache[str(row_dict["pocket_id"])])
        row_dict.update(geom)
        rows.append(row_dict)

    frame = pd.DataFrame(rows)
    frame["candidate_uid"] = frame["pocket_id"].astype(str) + ":" + frame["mol_idx"].astype(str)
    return frame


def fold_pockets(pockets: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(pockets).copy()
    rng.shuffle(shuffled)
    n_folds = min(max(2, n_folds), len(shuffled))
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold)]


def reranking_table(frame: pd.DataFrame, score_cols: dict[str, str], topk_fracs: list[float]) -> pd.DataFrame:
    rows = []
    for method, score_col in score_cols.items():
        for frac in topk_fracs:
            selected = []
            for _, group in frame.groupby("pocket_id", sort=False):
                k = max(1, int(np.ceil(len(group) * frac)))
                selected.append(group.nlargest(k, score_col))
            top = pd.concat(selected, ignore_index=True)
            rows.append(
                {
                    "method": method,
                    "topk_frac": frac,
                    "n_selected": int(len(top)),
                    "reliable_proxy_rate": float(top["reliable"].mean()),
                    "chemical_pass_rate": float((~top["chemical_failure"]).mean()),
                    "geometry_pass_rate": float((~top["geometric_failure"]).mean()),
                    "pocket_pass_rate": float((~top["pocket_failure"]).mean()),
                    "clash_free_rate": float((~top["clash_failure"]).mean()),
                    "synthetic_pass_rate": float((~top["synthetic_failure"]).mean()),
                    "diversity_proxy": float(top.groupby("pocket_id")["chemotype_id"].nunique().mean()),
                    "median_vina_affinity": float(top["vina_score"].median()),
                }
            )
    return pd.DataFrame(rows)


def run_seed(frame: pd.DataFrame, seed: int, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = model_cfg(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pockets = frame["pocket_id"].drop_duplicates().to_numpy()
    folds = fold_pockets(pockets, args.n_folds, seed)
    pred = frame.copy()
    pred["score_vina_score_only"] = -pred["vina_score"].astype(float)
    pred["score_qed_sa_vina"] = -pred["vina_score"].astype(float) + 0.5 * pred["qed"].astype(float) - 0.25 * pred["sa_proxy"].astype(float)
    for name in ["mlp_score_only", "mlp_ligand_descriptor", "mlp_pose_shape", "mlp_reliamol_noleak", "rf_reliamol_noleak", "hgb_reliamol_noleak"]:
        pred[f"score_{name}"] = np.nan

    for fold_idx, heldout in enumerate(folds):
        test_mask = pred["pocket_id"].isin(set(heldout))
        train = pred.loc[~test_mask].copy()
        test = pred.loc[test_mask].copy()
        print(f"seed={seed} fold={fold_idx + 1}/{len(folds)} train={len(train)} test={len(test)} heldout_pockets={len(heldout)}", flush=True)

        for variant in ["score_only", "ligand_descriptor", "pose_shape", "reliamol_noleak"]:
            method = f"mlp_{variant}"
            rel_prob = fit_mlp(train, test, feature_columns(variant), cfg, seed + deterministic_id(f"{fold_idx}:{variant}"), device)
            pred.loc[test.index, f"score_{method}"] = rel_prob

        noleak_features = feature_columns("reliamol_noleak")
        x_train, x_test = clean_matrix(train, test, noleak_features)
        rf = RandomForestClassifier(
            n_estimators=args.rf_trees,
            min_samples_leaf=3,
            n_jobs=args.rf_jobs,
            random_state=seed + fold_idx,
            class_weight="balanced",
        )
        rf.fit(x_train, train["reliable"])
        pred.loc[test.index, "score_rf_reliamol_noleak"] = rf.predict_proba(x_test)[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=args.hgb_iter, learning_rate=0.05, l2_regularization=0.03, random_state=seed + fold_idx)
        hgb.fit(x_train, train["reliable"])
        pred.loc[test.index, "score_hgb_reliamol_noleak"] = hgb.predict_proba(x_test)[:, 1]

    score_cols = {
        "vina_score_only": "score_vina_score_only",
        "qed_sa_vina": "score_qed_sa_vina",
        "mlp_score_only": "score_mlp_score_only",
        "mlp_ligand_descriptor": "score_mlp_ligand_descriptor",
        "mlp_pose_shape": "score_mlp_pose_shape",
        "mlp_reliamol_noleak": "score_mlp_reliamol_noleak",
        "rf_reliamol_noleak": "score_rf_reliamol_noleak",
        "hgb_reliamol_noleak": "score_hgb_reliamol_noleak",
    }
    metric_rows = []
    for method, col in score_cols.items():
        score = pred[col].to_numpy(float)
        prob = rank_to_prob(score) if method in {"vina_score_only", "qed_sa_vina"} else score
        row = binary_metrics(pred["reliable"].to_numpy(), prob)
        row.update({"seed": seed, "method": method, "device": device, "n_folds": len(folds)})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    rerank = reranking_table(pred, score_cols, args.topk_fracs)
    rerank["seed"] = seed

    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(seed_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(seed_dir / "metrics.csv", index=False)
    rerank.to_csv(seed_dir / "reranking.csv", index=False)
    return pred, metrics, rerank


def summarize(metrics: pd.DataFrame, rerank: pd.DataFrame, output_dir: Path) -> None:
    metric_summary = (
        metrics.groupby("method", sort=False)[["roc_auc", "pr_auc", "f1", "mcc", "ece"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metric_summary.columns = ["_".join([str(x) for x in col if x]) for col in metric_summary.columns.to_flat_index()]
    rerank_summary = (
        rerank.groupby(["method", "topk_frac"], sort=False)[
            ["reliable_proxy_rate", "chemical_pass_rate", "geometry_pass_rate", "pocket_pass_rate", "clash_free_rate", "synthetic_pass_rate", "diversity_proxy"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    rerank_summary.columns = ["_".join([str(x) for x in col if x]) for col in rerank_summary.columns.to_flat_index()]
    metric_summary.to_csv(output_dir / "metric_summary.csv", index=False)
    rerank_summary.to_csv(output_dir / "reranking_summary.csv", index=False)

    top10 = rerank_summary[rerank_summary["topk_frac"].eq(0.1)].copy()
    report_lines = [
        "# DiffSBDD Aligned ReliaMol Reranking",
        "",
        "Pocket-heldout out-of-fold evaluation on official DiffSBDD public samples restricted to public-PDB aligned pockets.",
        "",
        "## Metric Summary",
        "",
        metric_summary.to_csv(index=False),
        "## Top-10% Reranking",
        "",
        top10.to_csv(index=False),
    ]
    (output_dir / "reliamol_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(metric_summary.to_string(index=False))
    print(top10.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReliaMol reranking on aligned official DiffSBDD public samples.")
    parser.add_argument("--vina-dir", type=Path, default=Path("outputs/diffsbdd_aligned_vina/reports"))
    parser.add_argument("--pdb-cache", type=Path, default=Path("/home/test/wsk/public_generators/rcsb_pdb_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_aligned_reliamol"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--pocket-radius", type=float, default=12.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--rf-trees", type=int, default=260)
    parser.add_argument("--rf-jobs", type=int, default=16)
    parser.add_argument("--hgb-iter", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_frame(args)
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "feature_frame.csv", index=False)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "vina_dir": str(args.vina_dir),
                "n_candidates": int(len(frame)),
                "n_pockets": int(frame["pocket_id"].nunique()),
                "positive_rate": float(frame["reliable"].mean()),
                "feature_note": "No direct audit threshold flags are used as features; pocket distances use a fixed public-PDB pocket around the median generated center.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"feature_frame candidates={len(frame)} pockets={frame['pocket_id'].nunique()} reliable_rate={frame['reliable'].mean():.4f}",
        flush=True,
    )

    metrics_list = []
    rerank_list = []
    for seed in args.seeds:
        _, metrics, rerank = run_seed(frame, seed, args, report_dir)
        metrics_list.append(metrics)
        rerank_list.append(rerank)
    metrics_all = pd.concat(metrics_list, ignore_index=True)
    rerank_all = pd.concat(rerank_list, ignore_index=True)
    metrics_all.to_csv(report_dir / "metrics_by_seed.csv", index=False)
    rerank_all.to_csv(report_dir / "reranking_by_seed.csv", index=False)
    summarize(metrics_all, rerank_all, report_dir)


if __name__ == "__main__":
    main()
