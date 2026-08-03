#!/usr/bin/env python3
"""Independent fixed-receptor relaxation endpoint for generated 3D molecules.

The script freezes the prepared receptor, minimizes each generated ligand with
Amber protein parameters and GAFF ligand parameters, and records continuous
post-relaxation outcomes. PoseBusters is evaluated only after minimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmm import CustomNonbondedForce, LangevinMiddleIntegrator, LocalEnergyMinimizer, NonbondedForce, Platform, unit
from openmm import app
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from posebusters import PoseBusters
except Exception:
    PoseBusters = None


class CandidateTimeout(RuntimeError):
    pass


@contextmanager
def candidate_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum, frame):
        raise CandidateTimeout(f"candidate evaluation exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    previous_alarm = signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(previous_alarm)
        signal.signal(signal.SIGALRM, previous_handler)


def clean_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def prepare_receptor(source: Path, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixer = PDBFixer(filename=str(source))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with destination.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    return destination


def heavy_indices(topology: app.Topology) -> list[int]:
    return [atom.index for atom in topology.atoms() if atom.element is not None and atom.element.symbol != "H"]


def contact_set(ligand_xyz_a: np.ndarray, receptor_xyz_a: np.ndarray, cutoff_a: float = 4.5) -> set[tuple[int, int]]:
    if ligand_xyz_a.size == 0 or receptor_xyz_a.size == 0:
        return set()
    d2 = np.sum((ligand_xyz_a[:, None, :] - receptor_xyz_a[None, :, :]) ** 2, axis=2)
    li, ri = np.where(d2 < cutoff_a * cutoff_a)
    return set(zip(li.tolist(), ri.tolist()))


def geometry_summary(initial_lig: np.ndarray, final_lig: np.ndarray, receptor: np.ndarray) -> dict[str, float]:
    displacement = final_lig - initial_lig
    rmsd = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
    center_shift = float(np.linalg.norm(final_lig.mean(axis=0) - initial_lig.mean(axis=0)))
    initial_contacts = contact_set(initial_lig, receptor, 4.5)
    final_contacts = contact_set(final_lig, receptor, 4.5)
    retention = float(len(initial_contacts & final_contacts) / len(initial_contacts)) if initial_contacts else np.nan
    final_d2 = np.sum((final_lig[:, None, :] - receptor[None, :, :]) ** 2, axis=2)
    final_min = float(np.sqrt(np.min(final_d2))) if final_d2.size else np.nan
    final_clash_atoms = int(np.any(final_d2 < 1.2**2, axis=1).sum()) if final_d2.size else 0
    return {
        "ligand_rmsd_a": rmsd,
        "center_shift_a": center_shift,
        "initial_contact_pairs_4p5": len(initial_contacts),
        "final_contact_pairs_4p5": len(final_contacts),
        "contact_retention": retention,
        "final_min_receptor_distance_a": final_min,
        "final_clash_atoms_1p2": final_clash_atoms,
    }


class ReceptorContext:
    def __init__(self, fixed_pdb: Path):
        pdb = app.PDBFile(str(fixed_pdb))
        self.topology = pdb.topology
        self.positions = pdb.positions
        self.n_atoms = self.topology.getNumAtoms()
        self.heavy = heavy_indices(self.topology)
        xyz = np.asarray(self.positions.value_in_unit(unit.angstrom), dtype=np.float64)
        self.heavy_xyz_a = xyz[self.heavy]
        self.heavy_elements = [
            atom.element.symbol if atom.element is not None else "C"
            for atom in self.topology.atoms()
            if atom.index in set(self.heavy)
        ]


VDW_RADII_A = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Mg": 1.73, "Ca": 2.31, "Mn": 1.79, "Fe": 1.72, "Zn": 1.39,
}


def rdkit_to_openff(mol: Chem.Mol) -> tuple[Chem.Mol, Molecule]:
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    if work.GetNumConformers() == 0:
        raise ValueError("input molecule has no 3D conformer")
    work = Chem.AddHs(work, addCoords=True)
    offmol = Molecule.from_rdkit(work, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    if not offmol.conformers:
        coords = np.asarray(work.GetConformer().GetPositions(), dtype=float) * unit.angstrom
        offmol.add_conformer(coords)
    # Use deterministic RDKit Gasteiger charges while preserving the generated
    # SDF conformer. Calling OpenFF charge assignment can trigger a fresh
    # conformer-generation attempt for strained generated molecules.
    AllChem.ComputeGasteigerCharges(work)
    charges = []
    for atom in work.GetAtoms():
        value = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        try:
            charge = float(value)
        except ValueError:
            charge = 0.0
        if not math.isfinite(charge):
            charge = 0.0
        charges.append(charge)
    formal_charge = float(sum(atom.GetFormalCharge() for atom in work.GetAtoms()))
    if charges:
        correction = (formal_charge - float(sum(charges))) / len(charges)
        charges = [charge + correction for charge in charges]
    offmol.partial_charges = np.asarray(charges, dtype=float) * offunit.elementary_charge
    return work, offmol


def minimize_one(
    mol: Chem.Mol,
    receptor: ReceptorContext,
    platform_name: str,
    device_index: str,
    max_iterations: int,
    tolerance: float,
    receptor_model: str,
) -> tuple[dict[str, float | int | str | bool], Chem.Mol]:
    work, offmol = rdkit_to_openff(mol)
    ligand_topology = offmol.to_topology().to_openmm()
    ligand_positions = offmol.conformers[0].to_openmm()
    gaff = GAFFTemplateGenerator(molecules=offmol, forcefield="gaff-2.11")
    if receptor_model == "protein_ff":
        modeller = app.Modeller(receptor.topology, receptor.positions)
        modeller.add(ligand_topology, ligand_positions)
        forcefield = app.ForceField("amber14/protein.ff14SB.xml")
        forcefield.registerTemplateGenerator(gaff.generator)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
        )
        for index in range(receptor.n_atoms):
            system.setParticleMass(index, 0.0 * unit.dalton)
        all_positions = modeller.positions
        ligand_offset = receptor.n_atoms
    else:
        # Pocket10 receptors are non-contiguous residue collections and cannot be
        # assigned an intact protein topology without inventing peptide termini.
        # Use GAFF intramolecular ligand energy plus a fixed generic receptor LJ
        # environment, which tests local steric relaxation without that artifact.
        forcefield = app.ForceField()
        forcefield.registerTemplateGenerator(gaff.generator)
        system = forcefield.createSystem(
            ligand_topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )
        ligand_count = system.getNumParticles()
        nb_force = next(force for force in system.getForces() if isinstance(force, NonbondedForce))
        steric = CustomNonbondedForce(
            "4*sqrt(epsilon1*epsilon2)*((0.5*(sigma1+sigma2)/r)^12-(0.5*(sigma1+sigma2)/r)^6)"
        )
        steric.addPerParticleParameter("sigma")
        steric.addPerParticleParameter("epsilon")
        ligand_indices = set(range(ligand_count))
        for idx in range(ligand_count):
            _, sigma, epsilon = nb_force.getParticleParameters(idx)
            steric.addParticle([sigma, epsilon])
        for exception_index in range(nb_force.getNumExceptions()):
            particle1, particle2, _, _, _ = nb_force.getExceptionParameters(exception_index)
            steric.addExclusion(int(particle1), int(particle2))
        receptor_indices = set()
        for offset, element in enumerate(receptor.heavy_elements):
            system.addParticle(0.0 * unit.dalton)
            nb_force.addParticle(0.0, 0.30, 0.0)
            index = ligand_count + offset
            receptor_indices.add(index)
            radius_nm = VDW_RADII_A.get(element, 1.70) * 0.1
            sigma_nm = 2.0 * radius_nm / (2.0 ** (1.0 / 6.0))
            steric.addParticle([sigma_nm, 0.20])
        steric.addInteractionGroup(ligand_indices, receptor_indices)
        steric.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
        system.addForce(steric)
        ligand_xyz_nm = np.asarray(ligand_positions.value_in_unit(unit.nanometer), dtype=float)
        receptor_xyz_nm = receptor.heavy_xyz_a * 0.1
        all_positions = unit.Quantity(np.vstack([ligand_xyz_nm, receptor_xyz_nm]), unit.nanometer)
        ligand_offset = 0

    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name.upper() in {"CUDA", "OPENCL"}:
        properties["DeviceIndex"] = str(device_index)
        if platform_name.upper() == "CUDA":
            properties["Precision"] = "mixed"
    context = None
    try:
        from openmm import Context
        context = Context(system, integrator, platform, properties)
        context.setPositions(all_positions)
        initial_state = context.getState(getEnergy=True, getPositions=True)
        initial_energy = float(initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        initial_xyz = np.asarray(initial_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom), dtype=np.float64)

        LocalEnergyMinimizer.minimize(
            context,
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations,
        )
        final_state = context.getState(getEnergy=True, getPositions=True)
        final_energy = float(final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        final_xyz = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom), dtype=np.float64)

        rdkit_atom_count = work.GetNumAtoms()
        ligand_initial_all = initial_xyz[ligand_offset : ligand_offset + rdkit_atom_count]
        ligand_final_all = final_xyz[ligand_offset : ligand_offset + rdkit_atom_count]
        ligand_heavy = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1]
        summary = geometry_summary(
            ligand_initial_all[ligand_heavy],
            ligand_final_all[ligand_heavy],
            receptor.heavy_xyz_a,
        )
        summary.update({
            "initial_energy_kj_mol": initial_energy,
            "final_energy_kj_mol": final_energy,
            "energy_delta_kj_mol": final_energy - initial_energy,
            "parameterization_success": True,
            "minimization_success": True,
        })

        conf = work.GetConformer()
        for idx, xyz in enumerate(ligand_final_all):
            conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
        final_mol = Chem.RemoveHs(work, sanitize=True)
        return summary, final_mol
    finally:
        if context is not None:
            del context
        del integrator


def posebusters_for_molecules(molecules: list[Chem.Mol]) -> list[dict[str, object]]:
    if not molecules:
        return []
    if PoseBusters is None:
        return [{"final_posebusters_pass": np.nan, "final_posebusters_error": "PoseBusters unavailable"} for _ in molecules]
    try:
        table = PoseBusters(config="mol_fast", max_workers=0).bust(molecules, full_report=False)
        bool_cols = [c for c in table.columns if pd.api.types.is_bool_dtype(table[c])]
        output = []
        for _, row in table.iterrows():
            values = {f"final_pb_{c}": bool(row[c]) for c in bool_cols}
            values["final_posebusters_pass"] = bool(row[bool_cols].all()) if bool_cols else np.nan
            values["final_posebusters_error"] = ""
            output.append(values)
        return output
    except Exception as exc:
        return [{"final_posebusters_pass": np.nan, "final_posebusters_error": f"{type(exc).__name__}: {exc}"} for _ in molecules]


def load_receptor_plan(path: Path) -> dict[str, Path]:
    table = pd.read_csv(path)
    return {str(row.pocket_id): Path(row.pocket_pdb) for row in table.itertuples(index=False)}


def load_pocket_molecules(path: Path) -> list[Chem.Mol | None]:
    return list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=True) + "\n")
        handle.flush()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(str(json.loads(line)["candidate_uid"]))
            except Exception:
                continue
    return done


def process(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "prepared_receptors"
    jsonl = args.output_dir / f"relaxation_shard_{args.shard_index:02d}.jsonl"
    final_sdf_path = args.output_dir / f"relaxed_shard_{args.shard_index:02d}.sdf"
    done = completed_ids(jsonl) if args.resume else set()
    frame = pd.read_csv(args.feature_frame)
    pockets = sorted(frame["pocket_id"].astype(str).unique())
    selected_pockets = [p for i, p in enumerate(pockets) if i % args.num_shards == args.shard_index]
    if args.max_pockets > 0:
        selected_pockets = selected_pockets[: args.max_pockets]
    receptor_plan = load_receptor_plan(args.receptor_plan)
    writer = Chem.SDWriter(str(final_sdf_path)) if args.save_final_sdf and not args.resume else None

    metadata = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "platform": args.platform,
        "device_index": args.device_index,
        "receptor_model": args.receptor_model,
        "n_pockets": len(selected_pockets),
        "max_candidates_per_pocket": args.max_candidates_per_pocket,
        "endpoint": "GAFF ligand local minimization in a fixed receptor steric environment plus post-minimization PoseBusters mol_fast",
    }
    (args.output_dir / f"metadata_shard_{args.shard_index:02d}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for pocket_no, pocket_id in enumerate(selected_pockets, start=1):
        pocket_rows = frame[frame["pocket_id"].astype(str).eq(pocket_id)].sort_values("mol_idx")
        if args.max_candidates_per_pocket > 0:
            pocket_rows = pocket_rows.head(args.max_candidates_per_pocket)
        source_pdb = receptor_plan.get(pocket_id)
        if source_pdb is None or not source_pdb.exists():
            for row in pocket_rows.itertuples(index=False):
                append_jsonl(jsonl, {"candidate_uid": row.candidate_uid, "pocket_id": pocket_id, "status": "receptor_missing"})
            continue
        try:
            if args.receptor_model == "protein_ff":
                receptor_pdb = prepare_receptor(source_pdb, cache_dir / f"{clean_id(pocket_id)}.pdb")
            else:
                receptor_pdb = source_pdb
            receptor = ReceptorContext(receptor_pdb)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for row in pocket_rows.itertuples(index=False):
                if row.candidate_uid not in done:
                    append_jsonl(jsonl, {"candidate_uid": row.candidate_uid, "pocket_id": pocket_id, "status": "receptor_prep_failed", "error": error})
            continue

        sdf_groups = list(pocket_rows.groupby("sdf_file", sort=False))
        pocket_success_records: list[dict[str, object]] = []
        pocket_success_mols: list[Chem.Mol] = []
        for sdf_file, rows in sdf_groups:
            mols = load_pocket_molecules(Path(sdf_file))
            for row in rows.itertuples(index=False):
                uid = str(row.candidate_uid)
                if uid in done:
                    continue
                record: dict[str, object] = {
                    "candidate_uid": uid,
                    "pocket_id": pocket_id,
                    "sdf_file": str(sdf_file),
                    "mol_idx": int(row.mol_idx),
                    "status": "failed",
                    "platform": args.platform,
                    "device_index": str(args.device_index),
                }
                try:
                    mol = mols[int(row.mol_idx)]
                    if mol is None:
                        raise ValueError("RDKit failed to read molecule")
                    with candidate_timeout(args.candidate_timeout_sec):
                        summary, final_mol = minimize_one(
                            mol,
                            receptor,
                            args.platform,
                            args.device_index,
                            args.max_iterations,
                            args.tolerance,
                            args.receptor_model,
                        )
                    record.update(summary)
                    record["status"] = "success"
                    final_mol.SetProp("_Name", uid)
                    final_mol.SetProp("candidate_uid", uid)
                    pocket_success_records.append(record)
                    pocket_success_mols.append(final_mol)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback_tail"] = " | ".join(traceback.format_exc().splitlines()[-4:])
                    append_jsonl(jsonl, record)

        pb_records = posebusters_for_molecules(pocket_success_mols)
        for record, final_mol, pb in zip(pocket_success_records, pocket_success_mols, pb_records):
            record.update(pb)
            stable_components = [
                record.get("ligand_rmsd_a", np.inf) <= 2.0,
                record.get("contact_retention", -np.inf) >= 0.5,
                record.get("final_clash_atoms_1p2", 1) == 0,
            ]
            if isinstance(record.get("final_posebusters_pass"), (bool, np.bool_)):
                stable_components.append(bool(record["final_posebusters_pass"]))
            record["stable_relaxation_rmsd2_retention50"] = bool(all(stable_components))
            append_jsonl(jsonl, record)
            if writer is not None:
                for key, value in record.items():
                    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                        final_mol.SetProp(str(key), str(value))
                writer.write(final_mol)
        print(
            f"shard={args.shard_index} pocket={pocket_no}/{len(selected_pockets)} id={pocket_id} "
            f"candidates={len(pocket_rows)} success={len(pocket_success_records)}",
            flush=True,
        )
    if writer is not None:
        writer.close()
    if jsonl.exists():
        records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        pd.DataFrame(records).to_csv(args.output_dir / f"relaxation_shard_{args.shard_index:02d}.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-frame", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports/feature_frame.csv"))
    p.add_argument("--receptor-plan", type=Path, default=Path("outputs/diffsbdd_crossdocked_vina_gnina_100pocket_v0/reports/receptor_preparation.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/revision_r1_openmm_relaxation_v0"))
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU", "Reference"], default="CUDA")
    p.add_argument("--receptor-model", choices=["steric_dummy", "protein_ff"], default="steric_dummy")
    p.add_argument("--device-index", default="0")
    p.add_argument("--max-pockets", type=int, default=0)
    p.add_argument("--max-candidates-per-pocket", type=int, default=0)
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--tolerance", type=float, default=10.0)
    p.add_argument("--candidate-timeout-sec", type=int, default=0)
    p.add_argument("--save-final-sdf", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
