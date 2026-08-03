#!/usr/bin/env python3
"""Generate an external CASF-2016 redocking pose benchmark with GNINA."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from spyrmsd import molecule as spyrmsd_molecule
from spyrmsd import rmsd as spyrmsd_rmsd


def count_valid_poses(path: Path) -> tuple[int, int, float]:
    if not path.exists():
        return 0, 0, float("nan")
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False) if m is not None]
    near_native = 0
    best = float("nan")
    if mols:
        reference_path = path.parent.parent.parent / "references" / f"{path.parent.name}_ligand.sdf"
        ref = next((m for m in Chem.SDMolSupplier(str(reference_path), removeHs=False, sanitize=False) if m is not None), None) if reference_path.exists() else None
        rmsds = []
        if ref is not None:
            ref_heavy = Chem.RemoveHs(ref, sanitize=False)
            ref_spy = spyrmsd_molecule.Molecule.from_rdkit(ref_heavy)
            for mol in mols:
                try:
                    pose_spy = spyrmsd_molecule.Molecule.from_rdkit(Chem.RemoveHs(mol, sanitize=False))
                    rmsds.append(float(spyrmsd_rmsd.symmrmsd(
                        ref_spy.coordinates,
                        pose_spy.coordinates,
                        ref_spy.atomicnums,
                        pose_spy.atomicnums,
                        ref_spy.adjacency_matrix,
                        pose_spy.adjacency_matrix,
                        center=False,
                        minimize=False,
                    )))
                except Exception:
                    continue
            if rmsds:
                best = min(rmsds)
                near_native = sum(x < 2.0 for x in rmsds)
    return len(mols), near_native, best


def dock_target(target: str, args: argparse.Namespace, target_dir: Path) -> dict[str, object]:
    protein = args.casf_dir / f"{target}_protein.pdb"
    ligand = args.casf_dir / f"{target}_ligand.sdf"
    pose_file = target_dir / "poses.sdf"
    log_file = target_dir / "gnina.log"
    ref_dir = args.output_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_copy = ref_dir / ligand.name
    if not ref_copy.exists():
        ref_copy.write_bytes(ligand.read_bytes())
    target_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and pose_file.exists() and pose_file.stat().st_size > 0:
        n_poses, near, best = count_valid_poses(pose_file)
        return {"target": target, "status": "cached", "n_poses": n_poses, "near_native_poses": near, "best_rmsd_a": best}

    cmd = [
        str(args.gnina),
        "-r", str(protein),
        "-l", str(ligand),
        "--autobox_ligand", str(ligand),
        "--autobox_add", str(args.autobox_add),
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
        "--min_rmsd_filter", str(args.min_rmsd_filter),
        "--cnn_scoring", "rescore",
        "--cpu", str(args.cpu),
        "--device", str(args.device_index),
        "--seed", str(args.seed),
        "-o", str(pose_file),
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_file.write_text(proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf-8")
    n_poses, near, best = count_valid_poses(pose_file)
    return {
        "target": target,
        "status": "success" if proc.returncode == 0 and n_poses > 0 else "failed",
        "returncode": proc.returncode,
        "n_poses": n_poses,
        "near_native_poses": near,
        "best_rmsd_a": best,
        "wall_seconds": time.time() - started,
        "protein": str(protein),
        "ligand": str(ligand),
        "pose_file": str(pose_file),
        "error_tail": " | ".join(proc.stderr.splitlines()[-5:]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(path.name.replace("_protein.pdb", "") for path in args.casf_dir.glob("*_protein.pdb"))
    targets = [t for i, t in enumerate(targets) if i % args.num_shards == args.shard_index]
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    records = []
    for index, target in enumerate(targets, start=1):
        result = dock_target(target, args, args.output_dir / "targets" / target)
        records.append(result)
        pd.DataFrame(records).to_csv(args.output_dir / f"docking_shard_{args.shard_index:02d}.csv", index=False)
        print(
            f"shard={args.shard_index} target={index}/{len(targets)} id={target} "
            f"status={result['status']} poses={result.get('n_poses', 0)} best_rmsd={result.get('best_rmsd_a')}",
            flush=True,
        )
    metadata = {
        "casf_dir": str(args.casf_dir),
        "gnina": str(args.gnina),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seed": args.seed,
        "exhaustiveness": args.exhaustiveness,
        "num_modes": args.num_modes,
        "targets": len(targets),
    }
    (args.output_dir / f"metadata_shard_{args.shard_index:02d}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--casf-dir", type=Path, default=Path("/home/test/wsk/TASC-VS/third_party/CarsiDock/data/casf2016"))
    p.add_argument("--gnina", type=Path, default=Path("/home/test/miniconda3/envs/posebench310/bin/gnina"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/revision_r1_casf_redocking_v0"))
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=5)
    p.add_argument("--device-index", default="0")
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260630)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=20)
    p.add_argument("--min-rmsd-filter", type=float, default=0.25)
    p.add_argument("--autobox-add", type=float, default=5.0)
    p.add_argument("--max-targets", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
