from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def parse_sample_name(path: Path) -> tuple[str, str, str]:
    stem = path.name
    match = re.match(r"^([0-9a-zA-Z]{4})-([A-Za-z0-9]+)-rec-", stem)
    if not match:
        raise ValueError(f"Cannot parse receptor PDB/chain from {path.name}")
    pdb_id = match.group(1).lower()
    chain_id = match.group(2)
    pocket_id = stem.replace("_gen.sdf", "")
    return pdb_id, chain_id, pocket_id


def download_pdb(pdb_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = f"https://files.rcsb.org/view/{pdb_id.upper()}.pdb"
    tmp = out.with_suffix(".tmp")
    proc = subprocess.run(
        ["curl", "-L", "--retry", "2", "--connect-timeout", "20", "--max-time", "90", "-o", str(tmp), url],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        raise RuntimeError(f"Failed to download {pdb_id}: {proc.stderr[-500:]}")
    tmp.replace(out)
    return out


def parse_receptor_coords(path: Path, chain_id: str) -> np.ndarray:
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
                continue
    if not coords and chain_id:
        return parse_receptor_coords(path, "")
    return np.asarray(coords, dtype=np.float32)


def mol_coords(mol: Chem.Mol | None) -> np.ndarray:
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)


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


def sanitize_and_describe(mol: Chem.Mol | None) -> dict[str, float | str]:
    if mol is None:
        return {
            "sanitize_ok": 0.0,
            "sanitize_error": "missing_mol",
            "heavy_atoms": 0.0,
            "mol_wt": math.nan,
            "logp": math.nan,
            "tpsa": math.nan,
            "qed": math.nan,
            "rot_bonds": math.nan,
            "ring_count": math.nan,
            "hetero_atoms": math.nan,
            "abs_formal_charge": math.nan,
            "rare_element_count": math.nan,
            "sa_proxy": math.nan,
        }
    try:
        smol = Chem.Mol(mol)
        Chem.SanitizeMol(smol)
        sanitize_ok = 1.0
        sanitize_error = ""
    except Exception as exc:  # noqa: BLE001 - audit records RDKit failure
        smol = mol
        sanitize_ok = 0.0
        sanitize_error = str(exc).splitlines()[0][:180]

    elements = [atom.GetSymbol() for atom in smol.GetAtoms()]
    heavy_atoms = float(smol.GetNumHeavyAtoms())
    rot_bonds = float(Lipinski.NumRotatableBonds(smol)) if sanitize_ok else math.nan
    ring_count = float(rdMolDescriptors.CalcNumRings(smol)) if sanitize_ok else math.nan
    qed = float(QED.qed(smol)) if sanitize_ok and heavy_atoms else math.nan
    sa_proxy = (
        1.6
        + 0.035 * heavy_atoms
        + 0.12 * (0.0 if math.isnan(rot_bonds) else rot_bonds)
        + 0.16 * (0.0 if math.isnan(ring_count) else ring_count)
        + 1.2 * (1.0 - (qed if not math.isnan(qed) else 0.5))
    )
    return {
        "sanitize_ok": sanitize_ok,
        "sanitize_error": sanitize_error,
        "heavy_atoms": heavy_atoms,
        "mol_wt": float(Descriptors.MolWt(smol)) if sanitize_ok else math.nan,
        "logp": float(Crippen.MolLogP(smol)) if sanitize_ok else math.nan,
        "tpsa": float(rdMolDescriptors.CalcTPSA(smol)) if sanitize_ok else math.nan,
        "qed": qed,
        "rot_bonds": rot_bonds,
        "ring_count": ring_count,
        "hetero_atoms": float(sum(1 for atom in smol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
        "abs_formal_charge": float(abs(Chem.GetFormalCharge(smol))) if sanitize_ok else math.nan,
        "rare_element_count": float(sum(1 for symbol in elements if symbol not in ALLOWED_ELEMENTS)),
        "sa_proxy": float(sa_proxy),
    }


def audit_one_file(sdf_path: Path, pdb_cache: Path, max_mols: int) -> tuple[list[dict], dict]:
    pdb_id, chain_id, pocket_id = parse_sample_name(sdf_path)
    pdb_path = download_pdb(pdb_id, pdb_cache)
    receptor = parse_receptor_coords(pdb_path, chain_id)
    if len(receptor) == 0:
        raise RuntimeError(f"No receptor atoms parsed for {pdb_id} chain {chain_id}")

    rows = []
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for mol_idx, mol in enumerate(suppl):
        if mol_idx >= max_mols:
            break
        coords = mol_coords(mol)
        desc = sanitize_and_describe(mol)
        lr = pairwise_min(coords, receptor)
        nn = internal_nearest(coords)
        center = coords.mean(axis=0) if len(coords) else np.asarray([math.nan, math.nan, math.nan])
        receptor_center = receptor.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))) if len(coords) else math.nan
        rows.append(
            {
                "generator": "DiffSBDD",
                "sample_set": "crossdocked_fullatom_cond",
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "pocket_id": pocket_id,
                "sdf_file": str(sdf_path),
                "mol_idx": mol_idx,
                "mol_name": mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else f"{pocket_id}_{mol_idx:03d}",
                "has_3d": float(len(coords) > 0),
                "pose_atoms": float(len(coords)),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "center_to_receptor": float(np.linalg.norm(center - receptor_center)) if len(coords) else math.nan,
                "pose_radius": radius,
                "min_dist_to_receptor": float(np.nanmin(lr)),
                "mean_dist_to_receptor": float(np.nanmean(lr)),
                "receptor_contact_atoms_4p5": float(np.nansum(lr <= 4.5)),
                "receptor_close_atoms_3p5": float(np.nansum(lr <= 3.5)),
                "receptor_clash_atoms_1p2": float(np.nansum(lr < 1.2)),
                "receptor_severe_clash_atoms_0p8": float(np.nansum(lr < 0.8)),
                "internal_min_dist": float(np.nanmin(nn)),
                "internal_close_atoms_0p65": float(np.nansum(nn < 0.65)),
                **desc,
            }
        )
    meta = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "pocket_id": pocket_id,
        "sdf_file": str(sdf_path),
        "receptor_atoms": int(len(receptor)),
        "n_rows": len(rows),
    }
    return rows, meta


def add_failure_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["chemical_failure_proxy"] = (
        out["sanitize_ok"].lt(1.0)
        | out["heavy_atoms"].lt(6)
        | out["rare_element_count"].gt(0)
        | out["abs_formal_charge"].fillna(0).gt(2)
    )
    out["geometric_failure_proxy"] = (
        out["has_3d"].lt(1.0)
        | out["internal_close_atoms_0p65"].gt(0)
        | out["pose_radius"].lt(0.8)
        | out["pose_radius"].gt(8.0)
    )
    out["pocket_failure_proxy"] = out["receptor_contact_atoms_4p5"].le(0)
    out["clash_failure_proxy"] = out["receptor_clash_atoms_1p2"].gt(0)
    out["synthetic_failure_proxy"] = out["sa_proxy"].gt(6.8) | out["heavy_atoms"].gt(70) | out["rot_bonds"].fillna(0).gt(22)
    out["public_generated_reliable_proxy"] = ~(
        out["chemical_failure_proxy"]
        | out["geometric_failure_proxy"]
        | out["pocket_failure_proxy"]
        | out["clash_failure_proxy"]
        | out["synthetic_failure_proxy"]
    )
    return out


def summarize(df: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "n_candidates": int(len(df)),
                "n_pockets": int(df["pocket_id"].nunique()),
                "sanitize_rate": float(df["sanitize_ok"].mean()),
                "has_3d_rate": float(df["has_3d"].mean()),
                "reliable_proxy_rate": float(df["public_generated_reliable_proxy"].mean()),
                "chemical_failure_proxy_rate": float(df["chemical_failure_proxy"].mean()),
                "geometric_failure_proxy_rate": float(df["geometric_failure_proxy"].mean()),
                "pocket_failure_proxy_rate": float(df["pocket_failure_proxy"].mean()),
                "clash_failure_proxy_rate": float(df["clash_failure_proxy"].mean()),
                "synthetic_failure_proxy_rate": float(df["synthetic_failure_proxy"].mean()),
                "median_qed": float(df["qed"].median()),
                "median_sa_proxy": float(df["sa_proxy"].median()),
                "median_min_dist_to_receptor": float(df["min_dist_to_receptor"].median()),
                "median_receptor_contacts_4p5": float(df["receptor_contact_atoms_4p5"].median()),
            }
        ]
    )
    by_pocket = (
        df.groupby(["pdb_id", "chain_id", "pocket_id"], sort=False)
        .agg(
            n_candidates=("mol_idx", "count"),
            sanitize_rate=("sanitize_ok", "mean"),
            reliable_proxy_rate=("public_generated_reliable_proxy", "mean"),
            chemical_failure_proxy_rate=("chemical_failure_proxy", "mean"),
            geometric_failure_proxy_rate=("geometric_failure_proxy", "mean"),
            pocket_failure_proxy_rate=("pocket_failure_proxy", "mean"),
            clash_failure_proxy_rate=("clash_failure_proxy", "mean"),
            median_qed=("qed", "median"),
            median_sa_proxy=("sa_proxy", "median"),
            median_min_dist_to_receptor=("min_dist_to_receptor", "median"),
            median_receptor_contacts_4p5=("receptor_contact_atoms_4p5", "median"),
        )
        .reset_index()
        .merge(meta[["pdb_id", "chain_id", "pocket_id", "receptor_atoms"]], on=["pdb_id", "chain_id", "pocket_id"], how="left")
    )
    by_pocket["coordinate_mismatch"] = (by_pocket["median_receptor_contacts_4p5"] <= 0) & (by_pocket["median_min_dist_to_receptor"] > 10)
    by_pocket["high_clash_alignment"] = (by_pocket["clash_failure_proxy_rate"] > 0.5) & (~by_pocket["coordinate_mismatch"])
    by_pocket["usable_public_pdb_alignment"] = (
        (by_pocket["median_receptor_contacts_4p5"] > 0)
        & (by_pocket["median_min_dist_to_receptor"] < 6)
        & (by_pocket["clash_failure_proxy_rate"] <= 0.5)
    )
    by_pocket["alignment_status"] = "ambiguous"
    by_pocket.loc[by_pocket["coordinate_mismatch"], "alignment_status"] = "coordinate_mismatch"
    by_pocket.loc[by_pocket["high_clash_alignment"], "alignment_status"] = "high_clash_alignment"
    by_pocket.loc[by_pocket["usable_public_pdb_alignment"], "alignment_status"] = "usable_public_pdb_alignment"

    alignment = (
        by_pocket.groupby("alignment_status", sort=False)
        .agg(
            n_pockets=("pocket_id", "count"),
            n_candidates=("n_candidates", "sum"),
            mean_reliable_proxy_rate=("reliable_proxy_rate", "mean"),
            mean_pocket_failure_proxy_rate=("pocket_failure_proxy_rate", "mean"),
            mean_clash_failure_proxy_rate=("clash_failure_proxy_rate", "mean"),
            median_min_dist_to_receptor=("median_min_dist_to_receptor", "median"),
            median_receptor_contacts_4p5=("median_receptor_contacts_4p5", "median"),
        )
        .reset_index()
    )
    return overall, by_pocket, alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official DiffSBDD public sampled molecules against public RCSB receptors.")
    parser.add_argument(
        "--sdf-root",
        type=Path,
        default=Path("/home/test/wsk/public_generators/diffsbdd_zenodo_8239058/extracted/crossdocked_fullatom_cond"),
    )
    parser.add_argument("--pdb-cache", type=Path, default=Path("/home/test/wsk/public_generators/rcsb_pdb_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_public_audit_v0"))
    parser.add_argument("--max-files", type=int, default=101)
    parser.add_argument("--max-mols-per-file", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    sdf_files = [p for p in sorted(args.sdf_root.glob("*.sdf")) if not p.name.startswith("._")][: args.max_files]
    if not sdf_files:
        raise SystemExit(f"No SDF files found in {args.sdf_root}")

    rows: list[dict] = []
    meta_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(audit_one_file, path, args.pdb_cache, args.max_mols_per_file) for path in sdf_files]
        for future in as_completed(futures):
            try:
                file_rows, meta = future.result()
                rows.extend(file_rows)
                meta_rows.append(meta)
            except Exception as exc:  # noqa: BLE001 - keep public audit moving and record failed pockets
                meta_rows.append({"pdb_id": "", "chain_id": "", "pocket_id": "", "sdf_file": "", "receptor_atoms": 0, "n_rows": 0, "error": str(exc)[:500]})

    candidates = add_failure_proxies(pd.DataFrame(rows))
    meta = pd.DataFrame(meta_rows)
    overall, by_pocket, alignment = summarize(candidates, meta)
    status = by_pocket[["pocket_id", "alignment_status", "usable_public_pdb_alignment"]]
    candidates = candidates.merge(status, on="pocket_id", how="left")

    candidates.to_csv(report_dir / "candidate_audit.csv", index=False)
    meta.to_csv(report_dir / "pocket_metadata.csv", index=False)
    overall.to_csv(report_dir / "overall_summary.csv", index=False)
    by_pocket.to_csv(report_dir / "pocket_summary.csv", index=False)
    alignment.to_csv(report_dir / "alignment_summary.csv", index=False)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "DiffSBDD official Zenodo sampled molecules, crossdocked_fullatom_cond.zip",
                "sdf_root": str(args.sdf_root),
                "pdb_cache": str(args.pdb_cache),
                "max_files": args.max_files,
                "max_mols_per_file": args.max_mols_per_file,
                "n_files": len(sdf_files),
                "n_candidates": int(len(candidates)),
                "note": "Receptors are public RCSB PDB files inferred from DiffSBDD CrossDocked-style sample filenames.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DiffSBDD Public Sample Audit",
        "",
        "Official DiffSBDD sampled SDF files are evaluated against public RCSB receptor coordinates inferred from filenames.",
        "",
        "## Overall",
        "",
        overall.to_csv(index=False),
        "## Pocket Summary Head",
        "",
        by_pocket.head(25).to_csv(index=False),
        "## Alignment Summary",
        "",
        alignment.to_csv(index=False),
    ]
    (report_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(overall.to_string(index=False))
    print(by_pocket.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
