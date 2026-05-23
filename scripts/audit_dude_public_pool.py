from __future__ import annotations

import argparse
import gzip
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


DEFAULT_TARGETS = ["adrb2", "bace1", "cp3a4"]
ALLOWED_ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
}


def safe_extract(tar_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_dir)):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tf.extractall(out_dir)


def ensure_target_extracted(dude_root: Path, target: str, raw_dir: Path) -> Path:
    extracted = raw_dir / target
    if (extracted / "receptor.pdb").exists():
        return extracted

    tar_path = dude_root / "downloads" / f"{target}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing DUD-E tarball for {target}: {tar_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_extract(tar_path, raw_dir)
    if not (extracted / "receptor.pdb").exists():
        raise FileNotFoundError(f"Extracted target lacks receptor.pdb: {extracted}")
    return extracted


def parse_pdb_coords(path: Path) -> np.ndarray:
    coords = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
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


def mol_coords(mol: Chem.Mol | None) -> np.ndarray:
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=np.float32)
    conf = mol.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=np.float32)


def pairwise_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.asarray([math.nan], dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def internal_min_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return math.nan
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist += np.eye(len(coords), dtype=np.float32) * 999.0
    return float(dist.min())


def radius_of_gyration(coords: np.ndarray) -> float:
    if len(coords) == 0:
        return math.nan
    centered = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def sanitize_copy(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str]:
    if mol is None:
        return None, "missing_mol"
    try:
        smol = Chem.Mol(mol)
        Chem.SanitizeMol(smol)
        return smol, ""
    except Exception as exc:  # noqa: BLE001 - audit script records RDKit failure text
        return None, str(exc).splitlines()[0][:180]


def descriptor_row(mol: Chem.Mol | None) -> dict[str, float | int | str]:
    smol, sanitize_error = sanitize_copy(mol)
    if smol is None:
        return {
            "sanitize_ok": 0,
            "sanitize_error": sanitize_error,
            "mol_wt": math.nan,
            "logp": math.nan,
            "tpsa": math.nan,
            "qed": math.nan,
            "rot_bonds": math.nan,
            "ring_count": math.nan,
            "hetero_atoms": math.nan,
            "abs_formal_charge": math.nan,
            "rare_element_count": math.nan,
            "heavy_atoms": mol.GetNumHeavyAtoms() if mol is not None else 0,
        }

    elements = [atom.GetSymbol() for atom in smol.GetAtoms()]
    return {
        "sanitize_ok": 1,
        "sanitize_error": "",
        "mol_wt": float(Descriptors.MolWt(smol)),
        "logp": float(Crippen.MolLogP(smol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(smol)),
        "qed": float(QED.qed(smol)),
        "rot_bonds": int(Lipinski.NumRotatableBonds(smol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(smol)),
        "hetero_atoms": int(sum(1 for atom in smol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
        "abs_formal_charge": int(abs(Chem.GetFormalCharge(smol))),
        "rare_element_count": int(sum(1 for symbol in elements if symbol not in ALLOWED_ELEMENTS)),
        "heavy_atoms": int(smol.GetNumHeavyAtoms()),
    }


def mol_id(mol: Chem.Mol | None, fallback: str) -> str:
    if mol is None:
        return fallback
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    return name.strip() or fallback


def iter_sdf(path: Path, limit: int):
    seen = 0
    with gzip.open(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            yield idx, mol
            seen += 1
            if seen >= limit:
                break


def audit_target(target_dir: Path, target: str, max_per_class: int) -> tuple[list[dict], dict]:
    receptor = parse_pdb_coords(target_dir / "receptor.pdb")
    crystal = parse_mol2_coords(target_dir / "crystal_ligand.mol2")
    if len(receptor) == 0:
        raise RuntimeError(f"No receptor coordinates parsed for {target}")
    if len(crystal) == 0:
        raise RuntimeError(f"No crystal ligand coordinates parsed for {target}")

    crystal_center = crystal.mean(axis=0)
    receptor_center = receptor.mean(axis=0)
    pocket_atom_mask = pairwise_min(receptor, crystal) <= 6.0
    pocket = receptor[pocket_atom_mask]
    if len(pocket) == 0:
        pocket = receptor

    rows: list[dict] = []
    for class_name, sdf_name in [("active", "actives_final.sdf.gz"), ("decoy", "decoys_final.sdf.gz")]:
        sdf_path = target_dir / sdf_name
        if not sdf_path.exists():
            raise FileNotFoundError(f"Missing SDF file: {sdf_path}")

        for idx, mol in iter_sdf(sdf_path, max_per_class):
            coords = mol_coords(mol)
            center = coords.mean(axis=0) if len(coords) else np.asarray([math.nan, math.nan, math.nan])
            receptor_min = pairwise_min(coords, receptor)
            pocket_min = pairwise_min(coords, pocket)
            descriptors = descriptor_row(mol)
            rows.append(
                {
                    "target": target,
                    "source_class": class_name,
                    "source_sdf": str(sdf_path),
                    "candidate_index": idx,
                    "candidate_id": mol_id(mol, f"{target}_{class_name}_{idx:05d}"),
                    "has_3d": int(len(coords) > 0),
                    "n_atoms_with_coords": int(len(coords)),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "center_z": float(center[2]),
                    "center_to_crystal": float(np.linalg.norm(center - crystal_center)) if len(coords) else math.nan,
                    "center_to_receptor": float(np.linalg.norm(center - receptor_center)) if len(coords) else math.nan,
                    "min_dist_to_receptor": float(np.nanmin(receptor_min)),
                    "min_dist_to_pocket": float(np.nanmin(pocket_min)),
                    "pocket_contact_atoms_4p5": int(np.nansum(pocket_min <= 4.5)),
                    "receptor_clash_atoms_1p2": int(np.nansum(receptor_min < 1.2)),
                    "receptor_severe_clash_atoms_0p8": int(np.nansum(receptor_min < 0.8)),
                    "ligand_internal_min_dist": internal_min_distance(coords),
                    "ligand_radius_gyration": radius_of_gyration(coords),
                    **descriptors,
                }
            )

    target_meta = {
        "target": target,
        "receptor_atoms": int(len(receptor)),
        "crystal_ligand_atoms": int(len(crystal)),
        "pocket_atoms_within_6A": int(len(pocket)),
        "crystal_center_x": float(crystal_center[0]),
        "crystal_center_y": float(crystal_center[1]),
        "crystal_center_z": float(crystal_center[2]),
    }
    return rows, target_meta


def summarize(candidates: pd.DataFrame, target_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, source_class), group in candidates.groupby(["target", "source_class"], sort=False):
        rows.append(
            {
                "target": target,
                "source_class": source_class,
                "n": int(len(group)),
                "sanitize_rate": float(group["sanitize_ok"].mean()),
                "has_3d_rate": float(group["has_3d"].mean()),
                "median_center_to_crystal": float(group["center_to_crystal"].median()),
                "frac_center_within_8A": float((group["center_to_crystal"] <= 8.0).mean()),
                "frac_center_within_12A": float((group["center_to_crystal"] <= 12.0).mean()),
                "median_min_dist_to_pocket": float(group["min_dist_to_pocket"].median()),
                "median_pocket_contact_atoms_4p5": float(group["pocket_contact_atoms_4p5"].median()),
                "frac_any_pocket_contact_4p5": float((group["pocket_contact_atoms_4p5"] > 0).mean()),
                "frac_any_receptor_clash_1p2": float((group["receptor_clash_atoms_1p2"] > 0).mean()),
                "median_internal_min_dist": float(group["ligand_internal_min_dist"].median()),
                "median_qed": float(group["qed"].median()),
                "median_logp": float(group["logp"].median()),
                "median_tpsa": float(group["tpsa"].median()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.merge(target_meta, on="target", how="left")


def make_decision_report(summary: pd.DataFrame, targets: list[str], max_per_class: int) -> str:
    lines = [
        "# DUD-E Public Candidate Pool Audit",
        "",
        f"Targets: {', '.join(targets)}",
        f"Max molecules per class: {max_per_class}",
        "",
        "Purpose: verify whether public DUD-E SDF candidates have usable 3D coordinates near the receptor pocket before using pocket-conditioned reliability labels.",
        "",
    ]
    for _, row in summary.iterrows():
        near12 = row["frac_center_within_12A"]
        contacts = row["frac_any_pocket_contact_4p5"]
        verdict = "pocket-aligned" if near12 >= 0.7 and contacts >= 0.7 else "needs docking/alignment"
        lines.append(
            "- {target} {source_class}: n={n}, sanitize={sanitize_rate:.3f}, "
            "median center-to-crystal={median_center_to_crystal:.2f} A, "
            "within12A={frac_center_within_12A:.3f}, pocket-contact={frac_any_pocket_contact_4p5:.3f} -> {verdict}".format(
                **row.to_dict(),
                verdict=verdict,
            )
        )

    bad = summary[(summary["frac_center_within_12A"] < 0.7) | (summary["frac_any_pocket_contact_4p5"] < 0.7)]
    lines += [
        "",
        "Decision:",
    ]
    if len(bad):
        lines += [
            "- Do not use raw DUD-E SDF coordinates directly as pocket poses.",
            "- Next step should dock a small active/decoy subset into each public receptor, then compute pocket/geometric/scoring failures on docked poses.",
            "- The current audit remains useful as a clean molecule-quality and public-source traceability check.",
        ]
    else:
        lines += [
            "- Raw SDF coordinates appear pocket-aligned enough for a first public real-file smoke.",
            "- Next step can build a reliability dataset from these poses, while still validating with an external docking/PoseBusters-style check.",
        ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public DUD-E candidate SDFs before using pocket-conditioned labels.")
    parser.add_argument("--dude-root", type=Path, default=Path("/home/test/wsk/external_benchmarks/dude_subset"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dude_public_pool_v0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    meta_rows: list[dict] = []
    for target in args.targets:
        target_dir = ensure_target_extracted(args.dude_root, target, raw_dir)
        rows, meta = audit_target(target_dir, target, args.max_per_class)
        all_rows.extend(rows)
        meta_rows.append(meta)

    candidates = pd.DataFrame(all_rows)
    target_meta = pd.DataFrame(meta_rows)
    summary = summarize(candidates, target_meta)

    candidates_path = report_dir / "candidate_audit.csv"
    summary_path = report_dir / "target_summary.csv"
    decision_path = report_dir / "decision_report.md"
    metadata_path = report_dir / "run_metadata.json"

    candidates.to_csv(candidates_path, index=False)
    summary.to_csv(summary_path, index=False)
    decision_path.write_text(make_decision_report(summary, args.targets, args.max_per_class), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dude_root": str(args.dude_root),
                "targets": args.targets,
                "max_per_class": args.max_per_class,
                "outputs": {
                    "candidate_audit": str(candidates_path),
                    "target_summary": str(summary_path),
                    "decision_report": str(decision_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(candidates)} candidate rows")
    print(summary.to_string(index=False))
    print(f"Decision report: {decision_path}")


if __name__ == "__main__":
    main()
