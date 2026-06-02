from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


DEFAULT_TARGETS = ["adrb2", "bace1", "cp3a4"]
ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "K",
    "V",
    "I",
    "W",
    "Y",
    "Cl",
    "Br",
    "Na",
    "Mg",
    "Al",
    "Si",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Se",
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
            parts = line.split()
            try:
                if len(parts) >= 9 and not _is_float(parts[4]):
                    x, y, z = map(float, parts[6:9])
                else:
                    x, y, z = map(float, parts[5:8])
            except Exception:  # noqa: BLE001 - tolerate malformed public PDB rows
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
            coords.append((x, y, z))
    return np.asarray(coords, dtype=np.float32)


def parse_pdbqt_coords(path: Path) -> np.ndarray:
    coords = []
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        coords.append((float(parts[5]), float(parts[6]), float(parts[7])))
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


def pairwise_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.asarray([math.nan], dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def infer_element(record: str, atom_name: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", atom_name)
    if not letters:
        return "C"
    first = letters[0].upper()
    two = letters[:2].title()
    if record.startswith("HETATM") and two in ELEMENTS:
        return two
    if first in ELEMENTS:
        return first
    return "C"


def standardize_receptor_pdb(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with in_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            record = parts[0]
            serial = int(parts[1]) if parts[1].isdigit() else len(rows) + 1
            atom_name = parts[2]
            resname = parts[3]
            chain = "A"
            if len(parts) >= 9 and not _is_float(parts[4]):
                chain = parts[4][:1] or "A"
                resseq_token = parts[5]
                xyz = parts[6:9]
            else:
                resseq_token = parts[4]
                xyz = parts[5:8]
            try:
                resseq = int(re.sub(r"[^0-9-]", "", resseq_token) or "1")
                x, y, z = map(float, xyz)
            except ValueError:
                continue
            element = infer_element(record, atom_name)
            rows.append(
                f"{record:<6}{serial:5d} {atom_name:^4s} {resname:>3s} {chain:1s}"
                f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
                f"          {element:>2s}\n"
            )
    rows.append("END\n")
    out_path.write_text("".join(rows), encoding="utf-8")


def run_cmd(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def prepare_receptor(
    target: str,
    target_dir: Path,
    work_dir: Path,
    mk_prepare_receptor: Path,
    obabel: Path,
    box_size: float,
    timeout: int,
) -> dict:
    receptor_dir = work_dir / "receptors" / target
    receptor_dir.mkdir(parents=True, exist_ok=True)
    fixed_pdb = receptor_dir / "receptor_fixed.pdb"
    standardize_receptor_pdb(target_dir / "receptor.pdb", fixed_pdb)

    crystal = parse_mol2_coords(target_dir / "crystal_ligand.mol2")
    if len(crystal) == 0:
        raise RuntimeError(f"No crystal ligand coordinates parsed for {target}")
    center = crystal.mean(axis=0)
    out_base = receptor_dir / "receptor"
    cmd = [
        str(mk_prepare_receptor),
        "--read_pdb",
        str(fixed_pdb),
        "-o",
        str(out_base),
        "-p",
        "--box_center",
        f"{center[0]:.3f}",
        f"{center[1]:.3f}",
        f"{center[2]:.3f}",
        "--box_size",
        f"{box_size:.3f}",
        f"{box_size:.3f}",
        f"{box_size:.3f}",
    ]
    code, stdout, stderr = run_cmd(cmd, timeout)
    receptor_pdbqt = out_base.with_suffix(".pdbqt")
    if code != 0 or not receptor_pdbqt.exists() or receptor_pdbqt.stat().st_size == 0:
        fallback_pdbqt = receptor_dir / "receptor_obabel.pdbqt"
        fallback_cmd = [str(obabel), str(fixed_pdb), "-O", str(fallback_pdbqt), "-xr"]
        fallback_code, fallback_stdout, fallback_stderr = run_cmd(fallback_cmd, timeout)
        (receptor_dir / "mk_prepare_receptor.stdout.txt").write_text(stdout, encoding="utf-8")
        (receptor_dir / "mk_prepare_receptor.stderr.txt").write_text(stderr, encoding="utf-8")
        (receptor_dir / "obabel_receptor.stdout.txt").write_text(fallback_stdout, encoding="utf-8")
        (receptor_dir / "obabel_receptor.stderr.txt").write_text(fallback_stderr, encoding="utf-8")
        if fallback_code != 0 or not fallback_pdbqt.exists() or fallback_pdbqt.stat().st_size == 0:
            raise RuntimeError(f"receptor prep failed for {target}: {stderr[-800:]} {fallback_stderr[-800:]}")
        receptor_pdbqt = fallback_pdbqt
        prep_backend = "obabel_fallback"
    else:
        prep_backend = "meeko"

    receptor_coords = parse_pdb_coords(fixed_pdb)
    pocket = receptor_coords[pairwise_min(receptor_coords, crystal) <= 6.0]
    if len(pocket) == 0:
        pocket = receptor_coords
    (receptor_dir / "mk_prepare_receptor.stdout.txt").write_text(stdout, encoding="utf-8")
    (receptor_dir / "mk_prepare_receptor.stderr.txt").write_text(stderr, encoding="utf-8")
    return {
        "target": target,
        "receptor_pdbqt": receptor_pdbqt,
        "receptor_prep_backend": prep_backend,
        "fixed_pdb": fixed_pdb,
        "center": center.astype(float),
        "pocket": pocket.astype(np.float32),
        "receptor_coords": receptor_coords.astype(np.float32),
    }


def iter_sdf(path: Path, limit: int):
    seen = 0
    with gzip.open(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            if mol is None:
                continue
            yield idx, mol
            seen += 1
            if seen >= limit:
                break


def candidate_id(mol: Chem.Mol, fallback: str) -> str:
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return fallback


def write_candidate_sdfs(target_dir: Path, target: str, max_per_class: int, ligand_dir: Path) -> list[dict]:
    ligand_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_name, sdf_name in [("active", "actives_final.sdf.gz"), ("decoy", "decoys_final.sdf.gz")]:
        sdf_path = target_dir / sdf_name
        if not sdf_path.exists():
            raise FileNotFoundError(f"Missing SDF file: {sdf_path}")
        for idx, mol in iter_sdf(sdf_path, max_per_class):
            fallback = f"{target}_{class_name}_{idx:05d}"
            cid = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id(mol, fallback))[:80] or fallback
            single_sdf = ligand_dir / f"{target}_{class_name}_{idx:05d}_{cid}.sdf"
            writer = Chem.SDWriter(str(single_sdf))
            writer.write(mol)
            writer.close()
            rows.append(
                {
                    "target": target,
                    "source_class": class_name,
                    "candidate_index": idx,
                    "candidate_id": cid,
                    "sdf_path": single_sdf,
                }
            )
    return rows


def parse_vina_score(log_text: str) -> float:
    for line in log_text.splitlines():
        stripped = line.strip()
        if re.match(r"^1\s+", stripped):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return math.nan


def dock_one(task: dict) -> dict:
    work = task["work_dir"] / task["target"] / f"{task['source_class']}_{task['candidate_index']:05d}"
    work.mkdir(parents=True, exist_ok=True)
    ligand_pdbqt = work / "ligand.pdbqt"
    pose_pdbqt = work / "pose.pdbqt"

    prep_cmd = [
        str(task["mk_prepare_ligand"]),
        "-i",
        str(task["sdf_path"]),
        "-o",
        str(ligand_pdbqt),
    ]
    try:
        prep_code, prep_stdout, prep_stderr = run_cmd(prep_cmd, task["prep_timeout"])
    except subprocess.TimeoutExpired:
        return {**task["public_row"], "status": "ligand_prep_timeout", "vina_score": math.nan}
    (work / "ligand_prep.stdout.txt").write_text(prep_stdout, encoding="utf-8")
    (work / "ligand_prep.stderr.txt").write_text(prep_stderr, encoding="utf-8")
    if prep_code != 0 or not ligand_pdbqt.exists() or ligand_pdbqt.stat().st_size == 0:
        return {
            **task["public_row"],
            "status": "ligand_prep_failed",
            "vina_score": math.nan,
            "error_tail": prep_stderr[-500:],
        }

    center = task["center"]
    vina_cmd = [
        str(task["vina"]),
        "--receptor",
        str(task["receptor_pdbqt"]),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{center[0]:.3f}",
        "--center_y",
        f"{center[1]:.3f}",
        "--center_z",
        f"{center[2]:.3f}",
        "--size_x",
        f"{task['box_size']:.3f}",
        "--size_y",
        f"{task['box_size']:.3f}",
        "--size_z",
        f"{task['box_size']:.3f}",
        "--exhaustiveness",
        str(task["exhaustiveness"]),
        "--num_modes",
        "1",
        "--energy_range",
        "3",
        "--cpu",
        "1",
        "--seed",
        str(task["seed"]),
        "--out",
        str(pose_pdbqt),
    ]
    try:
        vina_code, vina_stdout, vina_stderr = run_cmd(vina_cmd, task["vina_timeout"])
    except subprocess.TimeoutExpired:
        return {**task["public_row"], "status": "vina_timeout", "vina_score": math.nan}
    (work / "vina.stdout.txt").write_text(vina_stdout, encoding="utf-8")
    (work / "vina.stderr.txt").write_text(vina_stderr, encoding="utf-8")
    log_text = vina_stdout + "\n" + vina_stderr
    (work / "vina.log.txt").write_text(log_text, encoding="utf-8")
    if vina_code != 0 or not pose_pdbqt.exists() or pose_pdbqt.stat().st_size == 0:
        return {
            **task["public_row"],
            "status": "vina_failed",
            "vina_score": parse_vina_score(log_text),
            "error_tail": (vina_stderr or log_text)[-500:],
        }

    pose = parse_pdbqt_coords(pose_pdbqt)
    pocket_min = pairwise_min(pose, task["pocket"])
    receptor_min = pairwise_min(pose, task["receptor_coords"])
    pose_center = pose.mean(axis=0) if len(pose) else np.asarray([math.nan, math.nan, math.nan])
    return {
        **task["public_row"],
        "status": "success",
        "vina_score": parse_vina_score(log_text),
        "pose_atoms": int(len(pose)),
        "pose_center_x": float(pose_center[0]),
        "pose_center_y": float(pose_center[1]),
        "pose_center_z": float(pose_center[2]),
        "center_to_crystal": float(np.linalg.norm(pose_center - center)) if len(pose) else math.nan,
        "min_dist_to_pocket": float(np.nanmin(pocket_min)),
        "pocket_contact_atoms_4p5": int(np.nansum(pocket_min <= 4.5)),
        "receptor_clash_atoms_1p2": int(np.nansum(receptor_min < 1.2)),
        "receptor_severe_clash_atoms_0p8": int(np.nansum(receptor_min < 0.8)),
        "pose_pdbqt": str(pose_pdbqt),
        "ligand_pdbqt": str(ligand_pdbqt),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, source_class), group in results.groupby(["target", "source_class"], sort=False):
        success = group[group["status"] == "success"]
        rows.append(
            {
                "target": target,
                "source_class": source_class,
                "n": int(len(group)),
                "success_rate": float((group["status"] == "success").mean()),
                "median_vina_score": float(success["vina_score"].median()) if len(success) else math.nan,
                "median_center_to_crystal": float(success["center_to_crystal"].median()) if len(success) else math.nan,
                "frac_center_within_8A": float((success["center_to_crystal"] <= 8.0).mean()) if len(success) else math.nan,
                "frac_any_pocket_contact_4p5": float((success["pocket_contact_atoms_4p5"] > 0).mean()) if len(success) else math.nan,
                "frac_any_receptor_clash_1p2": float((success["receptor_clash_atoms_1p2"] > 0).mean()) if len(success) else math.nan,
                "median_pocket_contacts_4p5": float(success["pocket_contact_atoms_4p5"].median()) if len(success) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dock a public DUD-E active/decoy smoke subset with Vina.")
    parser.add_argument("--dude-root", type=Path, default=Path("/home/test/wsk/external_benchmarks/dude_subset"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dude_docking_smoke"))
    parser.add_argument("--box-size", type=float, default=24.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prep-timeout", type=int, default=60)
    parser.add_argument("--vina-timeout", type=int, default=180)
    parser.add_argument("--vina", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_vina_screen_v2/bin/vina"))
    parser.add_argument(
        "--mk-prepare-ligand",
        type=Path,
        default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/mk_prepare_ligand.py"),
    )
    parser.add_argument(
        "--mk-prepare-receptor",
        type=Path,
        default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/mk_prepare_receptor.py"),
    )
    parser.add_argument("--obabel", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/obabel"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    ligands_dir = args.output_dir / "ligands"
    work_dir = args.output_dir / "work"
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    receptor_info = {}
    public_rows = []
    for target in args.targets:
        target_dir = ensure_target_extracted(args.dude_root, target, raw_dir)
        receptor_info[target] = prepare_receptor(
            target,
            target_dir,
            work_dir,
            args.mk_prepare_receptor,
            args.obabel,
            args.box_size,
            args.prep_timeout,
        )
        public_rows.extend(write_candidate_sdfs(target_dir, target, args.max_per_class, ligands_dir / target))

    tasks = []
    for row in public_rows:
        info = receptor_info[row["target"]]
        public_row = {k: (str(v) if isinstance(v, Path) else v) for k, v in row.items() if k != "sdf_path"}
        tasks.append(
            {
                "public_row": public_row,
                "sdf_path": row["sdf_path"],
                "target": row["target"],
                "source_class": row["source_class"],
                "candidate_index": row["candidate_index"],
                "work_dir": work_dir / "docking",
                "receptor_pdbqt": info["receptor_pdbqt"],
                "center": info["center"],
                "pocket": info["pocket"],
                "receptor_coords": info["receptor_coords"],
                "vina": args.vina,
                "mk_prepare_ligand": args.mk_prepare_ligand,
                "box_size": args.box_size,
                "exhaustiveness": args.exhaustiveness,
                "seed": args.seed + int(row["candidate_index"]),
                "prep_timeout": args.prep_timeout,
                "vina_timeout": args.vina_timeout,
            }
        )

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(dock_one, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    results_df = pd.DataFrame(results).sort_values(["target", "source_class", "candidate_index"])
    summary_df = summarize(results_df)

    results_path = report_dir / "docking_results.csv"
    summary_path = report_dir / "docking_summary.csv"
    metadata_path = report_dir / "run_metadata.json"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dude_root": str(args.dude_root),
                "targets": args.targets,
                "max_per_class": args.max_per_class,
                "box_size": args.box_size,
                "exhaustiveness": args.exhaustiveness,
                "workers": args.workers,
                "seed": args.seed,
                "tools": {
                    "vina": str(args.vina),
                    "mk_prepare_ligand": str(args.mk_prepare_ligand),
                    "mk_prepare_receptor": str(args.mk_prepare_receptor),
                    "obabel": str(args.obabel),
                },
                "outputs": {
                    "results": str(results_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Docking rows: {len(results_df)}")
    print(summary_df.to_string(index=False))
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
