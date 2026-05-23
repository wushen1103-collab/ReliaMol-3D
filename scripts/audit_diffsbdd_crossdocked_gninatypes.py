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


def parse_sample_name(path: Path) -> tuple[str, str, str, str]:
    match = re.match(r"^([0-9a-zA-Z]{4})-([A-Za-z0-9]+)-rec-", path.name)
    if not match:
        raise ValueError(f"Cannot parse receptor PDB/chain from {path.name}")
    pdb_id = match.group(1).lower()
    chain_id = match.group(2)
    pocket_id = path.name.replace("_gen.sdf", "")
    receptor_key = f"{pdb_id}_{chain_id}_rec_0.gninatypes"
    return pdb_id, chain_id, pocket_id, receptor_key


def build_receptor_index(index_path: Path) -> dict[str, str]:
    out = {}
    with index_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            member = line.strip()
            if not member.endswith(".gninatypes"):
                continue
            out[Path(member).name] = member
    return out


def ensure_receptors_extracted(archive: Path, index_path: Path, sdf_files: list[Path], extract_dir: Path) -> pd.DataFrame:
    receptor_index = build_receptor_index(index_path)
    rows = []
    members = []
    for sdf_path in sdf_files:
        pdb_id, chain_id, pocket_id, receptor_key = parse_sample_name(sdf_path)
        member = receptor_index.get(receptor_key)
        status = "found" if member else "missing"
        local_path = extract_dir / member if member else Path("")
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "pocket_id": pocket_id,
                "sdf_file": str(sdf_path),
                "receptor_key": receptor_key,
                "tar_member": member or "",
                "receptor_gninatypes": str(local_path) if member else "",
                "receptor_status": status,
            }
        )
        if member and (not local_path.exists() or local_path.stat().st_size == 0):
            members.append(member)

    if members:
        extract_dir.mkdir(parents=True, exist_ok=True)
        list_path = extract_dir / "selected_members.txt"
        list_path.write_text("\n".join(sorted(set(members))) + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(extract_dir), "--files-from", str(list_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"tar extraction failed: {proc.stderr[-1000:]}")
    return pd.DataFrame(rows)


def parse_gninatypes(path: Path, include_hydrogens: bool) -> np.ndarray:
    arr = np.fromfile(path, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<i4")])
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if not include_hydrogens:
        # libmolgrid/GNINA index types start with Hydrogen and PolarHydrogen.
        arr = arr[~np.isin(arr["t"], [0, 1])]
    coords = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return coords[np.isfinite(coords).all(axis=1)]


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
    except Exception as exc:  # noqa: BLE001
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


def audit_one_file(meta: dict, max_mols: int, include_hydrogens: bool) -> tuple[list[dict], dict]:
    sdf_path = Path(meta["sdf_file"])
    receptor_path = Path(meta["receptor_gninatypes"])
    receptor = parse_gninatypes(receptor_path, include_hydrogens)
    if len(receptor) == 0:
        raise RuntimeError(f"No receptor atoms parsed from {receptor_path}")

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
                "receptor_source": "crossdocked2020_gninatypes",
                "pdb_id": meta["pdb_id"],
                "chain_id": meta["chain_id"],
                "pocket_id": meta["pocket_id"],
                "sdf_file": str(sdf_path),
                "receptor_gninatypes": str(receptor_path),
                "mol_idx": mol_idx,
                "mol_name": mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else f"{meta['pocket_id']}_{mol_idx:03d}",
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
    meta_out = dict(meta)
    meta_out.update({"receptor_atoms": int(len(receptor)), "n_rows": len(rows)})
    return rows, meta_out


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


def summarize(df: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            synthetic_failure_proxy_rate=("synthetic_failure_proxy", "mean"),
            median_qed=("qed", "median"),
            median_sa_proxy=("sa_proxy", "median"),
            median_min_dist_to_receptor=("min_dist_to_receptor", "median"),
            median_receptor_contacts_4p5=("receptor_contact_atoms_4p5", "median"),
        )
        .reset_index()
        .merge(
            meta[["pdb_id", "chain_id", "pocket_id", "receptor_key", "tar_member", "receptor_atoms"]],
            on=["pdb_id", "chain_id", "pocket_id"],
            how="left",
        )
    )
    return overall, by_pocket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official DiffSBDD samples against CrossDocked2020 gninatypes receptors.")
    parser.add_argument(
        "--sdf-root",
        type=Path,
        default=Path("/home/test/wsk/public_generators/diffsbdd_zenodo_8239058/extracted/crossdocked_fullatom_cond"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/home/test/wsk/public_generators/crossdocked2020_receptors_v1_0/CrossDocked2020_receptors.tgz"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("/home/test/wsk/public_generators/crossdocked2020_receptors_v1_0/tar_index.txt"),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("/home/test/wsk/public_generators/crossdocked2020_receptors_v1_0/selected_receptors"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_gninatypes_audit_v0"))
    parser.add_argument("--max-files", type=int, default=101)
    parser.add_argument("--max-mols-per-file", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--include-receptor-hydrogens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    sdf_files = [p for p in sorted(args.sdf_root.glob("*.sdf")) if not p.name.startswith("._")][: args.max_files]
    if not sdf_files:
        raise SystemExit(f"No SDF files found in {args.sdf_root}")

    receptor_plan = ensure_receptors_extracted(args.archive, args.index, sdf_files, args.extract_dir)
    runnable = receptor_plan[receptor_plan["receptor_status"].eq("found")].copy()
    if runnable.empty:
        raise SystemExit("No matching CrossDocked gninatypes receptors found")

    rows: list[dict] = []
    meta_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(audit_one_file, rec._asdict(), args.max_mols_per_file, args.include_receptor_hydrogens)
            for rec in runnable.itertuples(index=False)
        ]
        for future in as_completed(futures):
            try:
                file_rows, meta = future.result()
                rows.extend(file_rows)
                meta_rows.append(meta)
            except Exception as exc:  # noqa: BLE001
                meta_rows.append({"error": str(exc)[:500]})

    candidates = add_failure_proxies(pd.DataFrame(rows))
    meta = pd.DataFrame(meta_rows)
    overall, by_pocket = summarize(candidates, meta)
    candidates.to_csv(report_dir / "candidate_audit.csv", index=False)
    receptor_plan.to_csv(report_dir / "receptor_match_plan.csv", index=False)
    meta.to_csv(report_dir / "pocket_metadata.csv", index=False)
    overall.to_csv(report_dir / "overall_summary.csv", index=False)
    by_pocket.to_csv(report_dir / "pocket_summary.csv", index=False)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "DiffSBDD official Zenodo sampled molecules audited against CrossDocked2020 receptor gninatypes coordinates.",
                "sdf_root": str(args.sdf_root),
                "archive": str(args.archive),
                "index": str(args.index),
                "extract_dir": str(args.extract_dir),
                "max_files": args.max_files,
                "max_mols_per_file": args.max_mols_per_file,
                "include_receptor_hydrogens": bool(args.include_receptor_hydrogens),
                "n_files": len(sdf_files),
                "n_matching_receptors": int(receptor_plan["receptor_status"].eq("found").sum()),
                "n_candidates": int(len(candidates)),
                "note": "gninatypes is parsed as little-endian float32 x/y/z plus int32 atom type records; receptor hydrogen types 0/1 are excluded by default.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DiffSBDD CrossDocked Gninatypes Audit",
        "",
        "Official DiffSBDD sampled SDF files are evaluated against CrossDocked2020 receptor gninatypes coordinates.",
        "",
        "## Overall",
        "",
        overall.to_csv(index=False),
        "## Pocket Summary Head",
        "",
        by_pocket.head(25).to_csv(index=False),
    ]
    (report_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(overall.to_string(index=False))
    print(by_pocket.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
