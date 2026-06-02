from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
ATOM_FEATURES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "metal"]
POSE_SHAPE_BASE = [
    "pose_atoms",
    "pose_radius_shape",
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


SOURCE_FILES = {
    "CrossDockedNative": "crossdocked_test_vina_docked.pt",
    "CVAE": "cvae_vina_docked.pt",
    "AR": "ar_vina_docked.pt",
    "Pocket2Mol": "pocket2mol_vina_docked.pt",
    "TargetDiff": "targetdiff_vina_docked.pt",
}


def deterministic_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def normalize_pocket_id(ligand_filename: str) -> str:
    folder, name = ligand_filename.split("/", 1)
    stem = Path(name).stem
    return f"{folder}/{stem}"


def pocket_pdb_path(ligand_filename: str, pocket_root: Path) -> Path:
    folder, name = ligand_filename.split("/", 1)
    stem = Path(name).stem
    return pocket_root / folder / f"{stem}_pocket10.pdb"


def parse_pdb_coords(path: Path) -> np.ndarray:
    coords = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if atom_name.startswith("H") or element == "H":
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float32)


def mol_coords_atoms(mol: Chem.Mol | None) -> tuple[np.ndarray, np.ndarray]:
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


def pose_shape_features(coords: np.ndarray, atoms: np.ndarray, pocket: np.ndarray) -> dict[str, float]:
    if len(coords) == 0:
        out = {name: math.nan for name in POSE_SHAPE_BASE}
        for atom in ATOM_FEATURES:
            out[f"atom_{atom}_frac"] = 0.0
        return out
    lp = pairwise_min(coords, pocket)
    nn = internal_nearest(coords)
    center = coords.mean(axis=0)
    centered = coords - center
    lp_q = np.nanquantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    nn_q = np.nanquantile(nn, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    out = {
        "pose_atoms": float(len(coords)),
        "pose_radius_shape": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
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
    }
    out["atom_metal_frac"] = float(np.mean(np.isin(atoms, ["Zn", "Mg", "Fe", "Mn", "Ca", "Na", "K", "Al"]))) if len(atoms) else 0.0
    for atom in ATOM_FEATURES:
        if atom == "metal":
            continue
        out[f"atom_{atom}_frac"] = float(np.mean(atoms == atom)) if len(atoms) else 0.0
    return out


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
    except Exception as exc:  # noqa: BLE001 - keep RDKit failure message for audit.
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


def flatten_records(obj: object) -> list[tuple[int, int, dict]]:
    rows: list[tuple[int, int, dict]] = []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for group_idx, rec in enumerate(obj):
            rows.append((group_idx, 0, rec))
        return rows
    if isinstance(obj, list):
        for group_idx, group in enumerate(obj):
            records = group if isinstance(group, list) else [group]
            for mol_idx, rec in enumerate(records):
                if isinstance(rec, dict):
                    rows.append((group_idx, mol_idx, rec))
    return rows


def first_affinity(vina: dict | None, key: str) -> float:
    if not isinstance(vina, dict):
        return math.nan
    entries = vina.get(key)
    if isinstance(entries, list) and entries:
        try:
            return float(entries[0].get("affinity", math.nan))
        except (TypeError, ValueError):
            return math.nan
    return math.nan


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


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    receptor_cache: dict[Path, np.ndarray] = {}
    rows = []
    load_errors = []
    for generator, file_name in SOURCE_FILES.items():
        path = args.input_dir / file_name
        if not path.exists():
            load_errors.append({"generator": generator, "file": str(path), "error": "missing"})
            continue
        # These files contain RDKit Mol objects from the public TargetDiff sampling-result format.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        records = flatten_records(obj)
        for group_idx, mol_idx, rec in records:
            ligand_filename = str(rec.get("ligand_filename", ""))
            if "/" not in ligand_filename:
                load_errors.append({"generator": generator, "file": file_name, "error": f"bad ligand_filename: {ligand_filename}"})
                continue
            pocket_id = normalize_pocket_id(ligand_filename)
            receptor_path = pocket_pdb_path(ligand_filename, args.pocket_root)
            if receptor_path not in receptor_cache:
                receptor_cache[receptor_path] = parse_pdb_coords(receptor_path) if receptor_path.exists() else np.empty((0, 3), dtype=np.float32)
            receptor = receptor_cache[receptor_path]
            mol = rec.get("mol")
            coords, atoms = mol_coords_atoms(mol)
            desc = sanitize_and_describe(mol)
            lr = pairwise_min(coords, receptor)
            nn = internal_nearest(coords)
            center = coords.mean(axis=0) if len(coords) else np.asarray([math.nan, math.nan, math.nan], dtype=np.float32)
            receptor_center = receptor.mean(axis=0) if len(receptor) else np.asarray([math.nan, math.nan, math.nan], dtype=np.float32)
            radius = float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))) if len(coords) else math.nan
            chem_results = rec.get("chem_results") if isinstance(rec.get("chem_results"), dict) else {}
            row = {
                "generator": generator,
                "source_file": file_name,
                "group_idx": group_idx,
                "mol_idx": mol_idx,
                "candidate_uid": f"{generator}:{pocket_id}:{mol_idx}",
                "ligand_filename": ligand_filename,
                "pocket_id": pocket_id,
                "pocket_pdb": str(receptor_path),
                "pocket_pdb_found": bool(receptor_path.exists()),
                "receptor_atoms": int(len(receptor)),
                "smiles": str(rec.get("smiles", "")),
                "official_qed": float(chem_results.get("qed", math.nan)),
                "official_sa": float(chem_results.get("sa", math.nan)),
                "official_logp": float(chem_results.get("logp", math.nan)),
                "official_lipinski": float(chem_results.get("lipinski", math.nan)),
                "vina_score_only_affinity": first_affinity(rec.get("vina"), "score_only"),
                "vina_minimize_affinity": first_affinity(rec.get("vina"), "minimize"),
                "vina_dock_affinity": first_affinity(rec.get("vina"), "dock"),
                "has_3d": float(len(coords) > 0),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "center_to_receptor": float(np.linalg.norm(center - receptor_center)) if len(coords) and len(receptor) else math.nan,
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
                **pose_shape_features(coords, atoms, receptor),
            }
            row["chemotype_id"] = deterministic_id(row["smiles"] or row["candidate_uid"]) % 100000
            rows.append(row)
        print(f"{generator}: groups={len(obj) if hasattr(obj, '__len__') else 'na'} rows={len(records)}", flush=True)

    candidates = add_failure_proxies(pd.DataFrame(rows))
    by_generator = (
        candidates.groupby("generator", sort=False)
        .agg(
            n_candidates=("candidate_uid", "count"),
            n_pockets=("pocket_id", "nunique"),
            pocket_pdb_found_rate=("pocket_pdb_found", "mean"),
            sanitize_rate=("sanitize_ok", "mean"),
            has_3d_rate=("has_3d", "mean"),
            reliable_proxy_rate=("public_generated_reliable_proxy", "mean"),
            chemical_failure_proxy_rate=("chemical_failure_proxy", "mean"),
            geometric_failure_proxy_rate=("geometric_failure_proxy", "mean"),
            pocket_failure_proxy_rate=("pocket_failure_proxy", "mean"),
            clash_failure_proxy_rate=("clash_failure_proxy", "mean"),
            synthetic_failure_proxy_rate=("synthetic_failure_proxy", "mean"),
            median_qed=("qed", "median"),
            median_sa_proxy=("sa_proxy", "median"),
            median_vina_minimize_affinity=("vina_minimize_affinity", "median"),
            median_min_dist_to_receptor=("min_dist_to_receptor", "median"),
            median_receptor_contacts_4p5=("receptor_contact_atoms_4p5", "median"),
        )
        .reset_index()
    )
    by_pocket_generator = (
        candidates.groupby(["generator", "pocket_id"], sort=False)
        .agg(
            n_candidates=("candidate_uid", "count"),
            reliable_proxy_rate=("public_generated_reliable_proxy", "mean"),
            clash_failure_proxy_rate=("clash_failure_proxy", "mean"),
            geometric_failure_proxy_rate=("geometric_failure_proxy", "mean"),
            median_vina_minimize_affinity=("vina_minimize_affinity", "median"),
        )
        .reset_index()
    )
    load_error_frame = pd.DataFrame(load_errors)
    return candidates, by_generator, by_pocket_generator if load_error_frame.empty else pd.concat([by_pocket_generator, load_error_frame], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TargetDiff official sampling-result .pt files in a unified ReliaMol candidate table.")
    parser.add_argument("--input-dir", type=Path, default=Path("/home/test/wsk/public_generators/targetdiff_sampling_results"))
    parser.add_argument(
        "--pocket-root",
        type=Path,
        default=Path("/home/test/wsk/public_generators/if3_crossdocked2020/selected_pockets/crossdocked_pocket10"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/targetdiff_official_multigen_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    candidates, by_generator, by_pocket_generator = audit(args)
    candidates.to_csv(report_dir / "candidate_audit.csv", index=False)
    by_generator.to_csv(report_dir / "generator_summary.csv", index=False)
    by_pocket_generator.to_csv(report_dir / "pocket_generator_summary.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_note": "TargetDiff README official sampling-result filenames loaded from the local mirror directory; SOURCE_MANIFEST.tsv records the concrete download source.",
        "input_dir": str(args.input_dir),
        "pocket_root": str(args.pocket_root),
        "n_candidates": int(len(candidates)),
        "n_generators": int(candidates["generator"].nunique()),
        "n_pockets": int(candidates["pocket_id"].nunique()),
        "source_files": SOURCE_FILES,
        "failure_proxy_note": "Same proxy family as DiffSBDD CrossDocked audit: RDKit chemistry, internal geometry, pocket contact, receptor clash, and synthetic accessibility filters.",
    }
    (report_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TargetDiff Official Sampling-Result Audit",
        "",
        "The TargetDiff/CVAE/AR/Pocket2Mol public sampling-result metadata files are audited against the 100 CrossDocked pocket10 PDBs.",
        "",
        "## Generator Summary",
        "",
        by_generator.to_csv(index=False),
        "## Pocket-Generator Summary Head",
        "",
        by_pocket_generator.head(30).to_csv(index=False),
        "## Failure Types",
        "",
        str(Counter(candidates["generator"])),
    ]
    (report_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(by_generator.to_string(index=False))


if __name__ == "__main__":
    main()
