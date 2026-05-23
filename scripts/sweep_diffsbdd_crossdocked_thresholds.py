from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402


SCORE_COLS = {
    "qed": "score_qed",
    "qed_sa": "score_qed_sa",
    "mlp_ligand_descriptor": "score_mlp_ligand_descriptor",
    "mlp_pose_shape": "score_mlp_pose_shape",
    "mlp_reliamol_novinascore": "score_mlp_reliamol_novinascore",
    "rf_reliamol_novinascore": "score_rf_reliamol_novinascore",
    "hgb_reliamol_novinascore": "score_hgb_reliamol_novinascore",
}


def thr_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def parse_gninatypes(path: Path, include_hydrogens: bool = False) -> np.ndarray:
    arr = np.fromfile(path, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<i4")])
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if not include_hydrogens:
        arr = arr[~np.isin(arr["t"], [0, 1])]
    coords = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return coords[np.isfinite(coords).all(axis=1)]


def mol_coords(mol: Chem.Mol | None, ligand_atom_mode: str) -> np.ndarray:
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=np.float32)
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if ligand_atom_mode == "heavy" and atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    return np.asarray(coords, dtype=np.float32)


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


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = np.asarray(score).argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def compute_one_file(
    sdf_file: str,
    group: pd.DataFrame,
    root: Path,
    ligand_atom_modes: list[str],
    contact_thresholds: list[float],
    clash_thresholds: list[float],
    internal_thresholds: list[float],
    include_receptor_hydrogens: bool,
) -> list[dict]:
    receptor_path = resolve_path(str(group["receptor_gninatypes"].iloc[0]), root)
    receptor = parse_gninatypes(receptor_path, include_hydrogens=include_receptor_hydrogens)
    sdf_path = resolve_path(str(sdf_file), root)
    mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False))
    rows: list[dict] = []
    for rec in group.itertuples(index=False):
        mol_idx = int(getattr(rec, "mol_idx"))
        mol = mols[mol_idx] if mol_idx < len(mols) else None
        for mode in ligand_atom_modes:
            coords = mol_coords(mol, mode)
            lr = pairwise_min(coords, receptor)
            nn = internal_nearest(coords)
            center = coords.mean(axis=0) if len(coords) else np.asarray([math.nan, math.nan, math.nan])
            radius = float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))) if len(coords) else math.nan
            out = {
                "pocket_id": getattr(rec, "pocket_id"),
                "pdb_id": getattr(rec, "pdb_id"),
                "chain_id": getattr(rec, "chain_id"),
                "sdf_file": str(sdf_path),
                "mol_idx": mol_idx,
                "candidate_uid": f"{getattr(rec, 'pocket_id')}:{mol_idx}",
                "ligand_atom_mode": mode,
                "has_3d_sweep": float(len(coords) > 0),
                "sweep_pose_atoms": float(len(coords)),
                "sweep_pose_radius": radius,
                "sweep_min_dist_to_receptor": float(np.nanmin(lr)),
                "sweep_internal_min_dist": float(np.nanmin(nn)),
            }
            for thr in contact_thresholds:
                out[f"contact_atoms_le_{thr_key(thr)}"] = float(np.nansum(lr <= thr))
            for thr in clash_thresholds:
                out[f"clash_atoms_lt_{thr_key(thr)}"] = float(np.nansum(lr < thr))
            for thr in internal_thresholds:
                out[f"internal_atoms_lt_{thr_key(thr)}"] = float(np.nansum(nn < thr))
            rows.append(out)
    return rows


def build_distance_features(args: argparse.Namespace, report_dir: Path) -> pd.DataFrame:
    root = Path.cwd()
    raw = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                compute_one_file,
                str(sdf_file),
                group.copy(),
                root,
                args.ligand_atom_modes,
                args.contact_thresholds,
                args.clash_thresholds,
                args.internal_thresholds,
                args.include_receptor_hydrogens,
            )
            for sdf_file, group in raw.groupby("sdf_file", sort=False)
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    dist = pd.DataFrame(rows)
    dist.to_csv(report_dir / "distance_features.csv", index=False)
    return dist


def make_label_grid(raw: pd.DataFrame, dist: pd.DataFrame, args: argparse.Namespace, report_dir: Path) -> pd.DataFrame:
    base_cols = [
        "pocket_id",
        "pdb_id",
        "chain_id",
        "sdf_file",
        "mol_idx",
        "candidate_uid",
        "sanitize_ok",
        "heavy_atoms",
        "rare_element_count",
        "abs_formal_charge",
        "sa_proxy",
        "rot_bonds",
    ]
    base = raw.copy()
    base["candidate_uid"] = base["pocket_id"].astype(str) + ":" + base["mol_idx"].astype(str)
    merged = dist.merge(base[base_cols], on=["pocket_id", "pdb_id", "chain_id", "sdf_file", "mol_idx", "candidate_uid"], how="left")
    rows = []
    for rec in merged.itertuples(index=False):
        common = {
            "candidate_uid": getattr(rec, "candidate_uid"),
            "pocket_id": getattr(rec, "pocket_id"),
            "pdb_id": getattr(rec, "pdb_id"),
            "chain_id": getattr(rec, "chain_id"),
            "mol_idx": int(getattr(rec, "mol_idx")),
            "ligand_atom_mode": getattr(rec, "ligand_atom_mode"),
        }
        chemical = (
            float(getattr(rec, "sanitize_ok")) < 1.0
            or float(getattr(rec, "heavy_atoms")) < 6
            or float(getattr(rec, "rare_element_count")) > 0
            or float(getattr(rec, "abs_formal_charge")) > 2
        )
        synthetic = float(getattr(rec, "sa_proxy")) > args.synthetic_sa_threshold or float(getattr(rec, "heavy_atoms")) > 70 or float(getattr(rec, "rot_bonds")) > 22
        for contact_thr in args.contact_thresholds:
            contact = float(getattr(rec, f"contact_atoms_le_{thr_key(contact_thr)}"))
            for clash_thr in args.clash_thresholds:
                clash = float(getattr(rec, f"clash_atoms_lt_{thr_key(clash_thr)}"))
                for internal_thr in args.internal_thresholds:
                    internal = float(getattr(rec, f"internal_atoms_lt_{thr_key(internal_thr)}"))
                    geometric = (
                        float(getattr(rec, "has_3d_sweep")) < 1.0
                        or internal > 0
                        or float(getattr(rec, "sweep_pose_radius")) < 0.8
                        or float(getattr(rec, "sweep_pose_radius")) > 8.0
                    )
                    pocket = contact <= 0
                    clash_failure = clash > 0
                    rows.append(
                        {
                            **common,
                            "contact_threshold": contact_thr,
                            "clash_threshold": clash_thr,
                            "internal_threshold": internal_thr,
                            "chemical_failure": bool(chemical),
                            "geometric_failure": bool(geometric),
                            "pocket_failure": bool(pocket),
                            "clash_failure": bool(clash_failure),
                            "synthetic_failure": bool(synthetic),
                            "reliable": not (chemical or geometric or pocket or clash_failure or synthetic),
                            "contact_atoms": contact,
                            "clash_atoms": clash,
                            "internal_close_atoms": internal,
                            "min_dist_to_receptor": float(getattr(rec, "sweep_min_dist_to_receptor")),
                        }
                    )
    labels = pd.DataFrame(rows)
    labels.to_csv(report_dir / "threshold_labels.csv", index=False)
    summary = (
        labels.groupby(["ligand_atom_mode", "contact_threshold", "clash_threshold", "internal_threshold"], sort=False)[
            ["reliable", "chemical_failure", "geometric_failure", "pocket_failure", "clash_failure", "synthetic_failure"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(report_dir / "label_sensitivity_summary.csv", index=False)
    return labels


def per_pocket_topk(frame: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(np.ceil(len(group) * frac)))
        selected.append(group.nlargest(k, score_col))
    return pd.concat(selected, ignore_index=True)


def evaluate_scores(labels: pd.DataFrame, args: argparse.Namespace, report_dir: Path) -> None:
    metric_rows = []
    rerank_rows = []
    for pred_path in sorted((args.pred_dir / "seeds").glob("seed_*/oof_predictions.csv")):
        seed = int(pred_path.parent.name.split("_")[-1])
        pred = pd.read_csv(pred_path)
        keep_cols = ["candidate_uid", "pocket_id"] + [c for c in SCORE_COLS.values() if c in pred.columns]
        pred = pred[keep_cols]
        for keys, label_group in labels.groupby(["ligand_atom_mode", "contact_threshold", "clash_threshold", "internal_threshold"], sort=False):
            frame = label_group.merge(pred, on=["candidate_uid", "pocket_id"], how="left")
            y = frame["reliable"].astype(int).to_numpy()
            for method, score_col in SCORE_COLS.items():
                if score_col not in frame.columns:
                    continue
                score = frame[score_col].to_numpy(float)
                prob = rank_to_prob(score) if method in {"qed", "qed_sa"} else score
                metrics = binary_metrics(y, prob)
                metric_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "ligand_atom_mode": keys[0],
                        "contact_threshold": keys[1],
                        "clash_threshold": keys[2],
                        "internal_threshold": keys[3],
                        **metrics,
                    }
                )
                for frac in args.topk_fracs:
                    top = per_pocket_topk(frame, score_col, frac)
                    rerank_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "ligand_atom_mode": keys[0],
                            "contact_threshold": keys[1],
                            "clash_threshold": keys[2],
                            "internal_threshold": keys[3],
                            "topk_frac": frac,
                            "n_selected": int(len(top)),
                            "reliable_rate": float(top["reliable"].mean()),
                            "chemical_pass_rate": float((~top["chemical_failure"]).mean()),
                            "geometry_pass_rate": float((~top["geometric_failure"]).mean()),
                            "pocket_pass_rate": float((~top["pocket_failure"]).mean()),
                            "clash_free_rate": float((~top["clash_failure"]).mean()),
                            "synthetic_pass_rate": float((~top["synthetic_failure"]).mean()),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    rerank = pd.DataFrame(rerank_rows)
    metrics.to_csv(report_dir / "threshold_metrics_by_seed.csv", index=False)
    rerank.to_csv(report_dir / "threshold_reranking_by_seed.csv", index=False)
    metric_summary = metrics.groupby(["method", "ligand_atom_mode", "contact_threshold", "clash_threshold", "internal_threshold"], sort=False)[
        ["roc_auc", "pr_auc", "ece"]
    ].agg(["mean", "std"]).reset_index()
    metric_summary.columns = ["_".join([str(x) for x in col if x]) for col in metric_summary.columns.to_flat_index()]
    metric_summary.to_csv(report_dir / "threshold_metric_summary.csv", index=False)
    rerank_summary = rerank.groupby(["method", "ligand_atom_mode", "contact_threshold", "clash_threshold", "internal_threshold", "topk_frac"], sort=False)[
        ["reliable_rate", "chemical_pass_rate", "geometry_pass_rate", "pocket_pass_rate", "clash_free_rate", "synthetic_pass_rate"]
    ].agg(["mean", "std"]).reset_index()
    rerank_summary.columns = ["_".join([str(x) for x in col if x]) for col in rerank_summary.columns.to_flat_index()]
    rerank_summary.to_csv(report_dir / "threshold_reranking_summary.csv", index=False)


def write_report(report_dir: Path) -> None:
    labels = pd.read_csv(report_dir / "label_sensitivity_summary.csv")
    rerank = pd.read_csv(report_dir / "threshold_reranking_summary.csv")
    top10 = rerank[rerank["topk_frac"].eq(0.1)].copy()
    key_methods = ["qed_sa", "mlp_pose_shape", "mlp_reliamol_novinascore", "rf_reliamol_novinascore", "hgb_reliamol_novinascore"]
    lines = [
        "# DiffSBDD CrossDocked Threshold Sensitivity",
        "",
        "Labels are recomputed from exact CrossDocked gninatypes receptor coordinates while sweeping ligand atom mode, contact, clash, and internal-close thresholds.",
        "",
        "## Label Range",
        "",
        labels.groupby("ligand_atom_mode")["reliable"].agg(["min", "median", "max"]).reset_index().to_csv(index=False),
        "## Top-10% Reliable Range",
        "",
        top10[top10["method"].isin(key_methods)]
        .groupby(["method", "ligand_atom_mode"], sort=False)["reliable_rate_mean"]
        .agg(["min", "median", "max"])
        .reset_index()
        .to_csv(index=False),
        "## Strictest Heavy-Atom Label Head",
        "",
        top10[
            top10["ligand_atom_mode"].eq("heavy")
            & top10["contact_threshold"].eq(top10["contact_threshold"].min())
            & top10["clash_threshold"].eq(top10["clash_threshold"].max())
        ]
        .sort_values("reliable_rate_mean", ascending=False)
        .head(10)
        .to_csv(index=False),
    ]
    (report_dir / "threshold_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CrossDocked DiffSBDD reliability-label thresholds and reevaluate reranking scores.")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports"))
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_threshold_sensitivity_v0"))
    parser.add_argument("--ligand-atom-modes", nargs="+", default=["all", "heavy"], choices=["all", "heavy"])
    parser.add_argument("--contact-thresholds", nargs="+", type=float, default=[4.0, 4.5, 5.0])
    parser.add_argument("--clash-thresholds", nargs="+", type=float, default=[1.0, 1.2, 1.5])
    parser.add_argument("--internal-thresholds", nargs="+", type=float, default=[0.65])
    parser.add_argument("--synthetic-sa-threshold", type=float, default=6.8)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--include-receptor-hydrogens", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    dist_path = report_dir / "distance_features.csv"
    if dist_path.exists() and not args.force:
        dist = pd.read_csv(dist_path)
    else:
        dist = build_distance_features(args, report_dir)
    labels = make_label_grid(raw, dist, args, report_dir)
    evaluate_scores(labels, args, report_dir)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_dir": str(args.audit_dir),
                "pred_dir": str(args.pred_dir),
                "n_candidates": int(raw.shape[0]),
                "n_label_rows": int(labels.shape[0]),
                "ligand_atom_modes": args.ligand_atom_modes,
                "contact_thresholds": args.contact_thresholds,
                "clash_thresholds": args.clash_thresholds,
                "internal_thresholds": args.internal_thresholds,
                "note": "Evaluation-only label robustness: OOF scores are fixed; labels are recomputed under alternate geometry thresholds.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(report_dir)
    print((report_dir / "threshold_sensitivity_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
