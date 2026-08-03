#!/usr/bin/env python3
"""Build an OpenMM receptor plan for aligned public-PDB receptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.vina_receptor_plan)
    rows = []
    for row in table.itertuples(index=False):
        pdbqt = Path(str(row.receptor_pdbqt))
        chain_pdb = pdbqt.parent / "receptor_chain.pdb"
        rows.append({
            "pocket_id": str(row.pocket_id),
            "pocket_pdb": str(chain_pdb),
            "source_pdb_id": getattr(row, "pdb_id", ""),
            "source_chain_id": getattr(row, "chain_id", ""),
            "pocket_pdb_exists": chain_pdb.exists(),
        })
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out["pocket_pdb_exists"].value_counts(dropna=False).to_string())
    missing = out.loc[~out["pocket_pdb_exists"], ["pocket_id", "pocket_pdb"]]
    if not missing.empty:
        print("Missing receptor_chain.pdb records:")
        print(missing.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vina-receptor-plan", type=Path, default=Path("outputs/diffsbdd_aligned_vina_v0/reports/receptor_prep.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jctc_aligned_openmm_relaxation_v0/aligned_receptor_preparation.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    main()

