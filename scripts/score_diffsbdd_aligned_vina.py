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


def run_cmd(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def safe_name(text: str, max_len: int = 140) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:max_len].strip("._") or "item"


def filter_chain_pdb(pdb_path: Path, chain_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            if chain_id and len(line) > 21 and line[21].strip() and line[21].strip() != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            rows.append(line.rstrip("\n") + "\n")
    if not rows and chain_id:
        filter_chain_pdb(pdb_path, "", out_path)
        return
    rows.append("END\n")
    out_path.write_text("".join(rows), encoding="utf-8")


def prepare_receptor(row: pd.Series, pdb_cache: Path, work_dir: Path, mk_prepare_receptor: Path, obabel: Path, timeout: int) -> dict:
    pdb_id = str(row["pdb_id"]).lower()
    chain_id = str(row["chain_id"])
    pocket_id = str(row["pocket_id"])
    receptor_dir = work_dir / "receptors" / safe_name(pocket_id)
    receptor_dir.mkdir(parents=True, exist_ok=True)
    source_pdb = pdb_cache / f"{pdb_id}.pdb"
    if not source_pdb.exists():
        raise FileNotFoundError(f"Missing cached PDB for {pdb_id}: {source_pdb}")
    chain_pdb = receptor_dir / "receptor_chain.pdb"
    filter_chain_pdb(source_pdb, chain_id, chain_pdb)

    out_base = receptor_dir / "receptor"
    cmd = [str(mk_prepare_receptor), "--read_pdb", str(chain_pdb), "-o", str(out_base), "-p"]
    code, stdout, stderr = run_cmd(cmd, timeout)
    pdbqt = out_base.with_suffix(".pdbqt")
    backend = "meeko"
    if code != 0 or not pdbqt.exists() or pdbqt.stat().st_size == 0:
        fallback = receptor_dir / "receptor_obabel.pdbqt"
        fcmd = [str(obabel), str(chain_pdb), "-O", str(fallback), "-xr"]
        fcode, fstdout, fstderr = run_cmd(fcmd, timeout)
        (receptor_dir / "mk_prepare_receptor.stdout.txt").write_text(stdout, encoding="utf-8")
        (receptor_dir / "mk_prepare_receptor.stderr.txt").write_text(stderr, encoding="utf-8")
        (receptor_dir / "obabel_receptor.stdout.txt").write_text(fstdout, encoding="utf-8")
        (receptor_dir / "obabel_receptor.stderr.txt").write_text(fstderr, encoding="utf-8")
        if fcode != 0 or not fallback.exists() or fallback.stat().st_size == 0:
            raise RuntimeError(f"Receptor prep failed for {pocket_id}: {stderr[-500:]} {fstderr[-500:]}")
        pdbqt = fallback
        backend = "obabel_fallback"
    else:
        (receptor_dir / "mk_prepare_receptor.stdout.txt").write_text(stdout, encoding="utf-8")
        (receptor_dir / "mk_prepare_receptor.stderr.txt").write_text(stderr, encoding="utf-8")

    return {
        "pocket_id": pocket_id,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "receptor_pdbqt": str(pdbqt),
        "receptor_prep_backend": backend,
    }


def write_selected_sdfs(selected: pd.DataFrame, ligand_dir: Path) -> pd.DataFrame:
    ligand_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sdf_file, group in selected.groupby("sdf_file", sort=False):
        source = Path(sdf_file)
        needed = set(int(x) for x in group["mol_idx"])
        supplier = Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False)
        mols = {idx: mol for idx, mol in enumerate(supplier) if idx in needed and mol is not None}
        for item in group.itertuples(index=False):
            mol_idx = int(item.mol_idx)
            mol = mols.get(mol_idx)
            out = ligand_dir / safe_name(str(item.pocket_id)) / f"mol_{mol_idx:03d}.sdf"
            out.parent.mkdir(parents=True, exist_ok=True)
            status = "ok"
            if mol is None:
                status = "missing_mol"
            else:
                try:
                    work_mol = Chem.Mol(mol)
                    Chem.SanitizeMol(work_mol)
                    work_mol = Chem.AddHs(work_mol, addCoords=True)
                except Exception:
                    work_mol = mol
                writer = Chem.SDWriter(str(out))
                writer.write(work_mol)
                writer.close()
            row = item._asdict()
            row["single_sdf"] = str(out)
            row["single_sdf_status"] = status
            rows.append(row)
    return pd.DataFrame(rows)


def parse_score(text: str) -> dict[str, float]:
    out = {
        "vina_affinity": math.nan,
        "vina_inter": math.nan,
        "vina_intra": math.nan,
        "vina_torsions": math.nan,
    }

    def line_value(line: str) -> float:
        kcal_match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)\s*\(kcal/mol\)", line)
        if kcal_match:
            return float(kcal_match.group(1))
        value_match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)\b", line)
        return float(value_match.group(1)) if value_match else math.nan

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        value = line_value(stripped)
        if math.isnan(value):
            continue
        if lower.startswith("affinity:") or lower.startswith("estimated free energy of binding"):
            out["vina_affinity"] = value
        elif lower.startswith("intermolecular") or "final intermolecular energy" in lower:
            out["vina_inter"] = value
        elif lower.startswith("intramolecular") or "final total internal energy" in lower:
            out["vina_intra"] = value
        elif lower.startswith("torsional") or "torsional free energy" in lower:
            out["vina_torsions"] = value
    return out


def score_one(task: dict) -> dict:
    row = task["row"].copy()
    work = task["work_dir"] / "scoring" / safe_name(row["pocket_id"]) / f"mol_{int(row['mol_idx']):03d}"
    work.mkdir(parents=True, exist_ok=True)
    ligand_pdbqt = work / "ligand.pdbqt"

    if row.get("single_sdf_status") != "ok":
        row.update({"ligand_prep_status": "missing_mol", "vina_status": "not_run", "vina_affinity": math.nan})
        return row

    prep_cmd = [str(task["mk_prepare_ligand"]), "-i", str(row["single_sdf"]), "-o", str(ligand_pdbqt)]
    try:
        prep_code, prep_stdout, prep_stderr = run_cmd(prep_cmd, task["prep_timeout"])
    except subprocess.TimeoutExpired:
        row.update({"ligand_prep_status": "timeout", "vina_status": "not_run", "vina_affinity": math.nan})
        return row
    (work / "ligand_prep.stdout.txt").write_text(prep_stdout, encoding="utf-8")
    (work / "ligand_prep.stderr.txt").write_text(prep_stderr, encoding="utf-8")
    if prep_code != 0 or not ligand_pdbqt.exists() or ligand_pdbqt.stat().st_size == 0:
        row.update(
            {
                "ligand_prep_status": "failed",
                "vina_status": "not_run",
                "vina_affinity": math.nan,
                "error_tail": prep_stderr[-500:],
            }
        )
        return row

    receptor_pdbqt = task["receptors"][row["pocket_id"]]["receptor_pdbqt"]
    vina_cmd = [
        str(task["vina"]),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--score_only",
        "--autobox",
        "--cpu",
        "1",
    ]
    try:
        code, stdout, stderr = run_cmd(vina_cmd, task["vina_timeout"])
    except subprocess.TimeoutExpired:
        row.update({"ligand_prep_status": "ok", "vina_status": "timeout", "vina_affinity": math.nan})
        return row
    log_text = stdout + "\n" + stderr
    (work / "vina_score.stdout.txt").write_text(stdout, encoding="utf-8")
    (work / "vina_score.stderr.txt").write_text(stderr, encoding="utf-8")
    scores = parse_score(log_text)
    status = "success" if code == 0 and not math.isnan(scores["vina_affinity"]) else "failed"
    row.update(
        {
            "ligand_prep_status": "ok",
            "vina_status": status,
            "error_tail": "" if status == "success" else (stderr or stdout)[-500:],
            "ligand_pdbqt": str(ligand_pdbqt),
            "receptor_pdbqt": str(receptor_pdbqt),
            **scores,
        }
    )
    return row


def summarize(results: pd.DataFrame, topk_fracs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = results[results["vina_status"].eq("success")].copy()
    overall = pd.DataFrame(
        [
            {
                "n_candidates": int(len(results)),
                "n_pockets": int(results["pocket_id"].nunique()),
                "ligand_prep_success_rate": float(results["ligand_prep_status"].eq("ok").mean()),
                "vina_success_rate": float(results["vina_status"].eq("success").mean()),
                "reliable_proxy_rate": float(results["public_generated_reliable_proxy"].mean()),
                "median_vina_affinity": float(success["vina_affinity"].median()) if len(success) else math.nan,
                "median_vina_affinity_reliable": float(success[success["public_generated_reliable_proxy"]]["vina_affinity"].median())
                if success["public_generated_reliable_proxy"].any()
                else math.nan,
                "median_vina_affinity_unreliable": float(success[~success["public_generated_reliable_proxy"]]["vina_affinity"].median())
                if (~success["public_generated_reliable_proxy"]).any()
                else math.nan,
            }
        ]
    )
    rerank_rows = []
    for frac in topk_fracs:
        selected = []
        for _, group in success.groupby("pocket_id", sort=False):
            k = max(1, int(np.ceil(len(group) * frac)))
            selected.append(group.nsmallest(k, "vina_affinity"))
        top = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
        rerank_rows.append(
            {
                "score": "vina_score_only",
                "topk_frac": frac,
                "n_selected": int(len(top)),
                "reliable_proxy_rate": float(top["public_generated_reliable_proxy"].mean()) if len(top) else math.nan,
                "chemical_pass_rate": float((~top["chemical_failure_proxy"]).mean()) if len(top) else math.nan,
                "geometry_pass_rate": float((~top["geometric_failure_proxy"]).mean()) if len(top) else math.nan,
                "pocket_pass_rate": float((~top["pocket_failure_proxy"]).mean()) if len(top) else math.nan,
                "clash_free_rate": float((~top["clash_failure_proxy"]).mean()) if len(top) else math.nan,
                "median_vina_affinity": float(top["vina_affinity"].median()) if len(top) else math.nan,
            }
        )
    return overall, pd.DataFrame(rerank_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vina score-only check for aligned official DiffSBDD public samples.")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/diffsbdd_public_audit/reports"))
    parser.add_argument("--pdb-cache", type=Path, default=Path("/home/test/wsk/public_generators/rcsb_pdb_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_aligned_vina"))
    parser.add_argument("--max-pockets", type=int, default=39)
    parser.add_argument("--max-mols-per-pocket", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--prep-timeout", type=int, default=60)
    parser.add_argument("--vina-timeout", type=int, default=60)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
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
    report_dir = args.output_dir / "reports"
    work_dir = args.output_dir / "work"
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    pockets = pd.read_csv(args.audit_dir / "pocket_summary.csv")
    usable = pockets[pockets["usable_public_pdb_alignment"]].copy().head(args.max_pockets)
    selected = candidates[candidates["pocket_id"].isin(set(usable["pocket_id"]))].copy()
    selected = selected.groupby("pocket_id", sort=False).head(args.max_mols_per_pocket).reset_index(drop=True)
    if selected.empty:
        raise SystemExit("No aligned DiffSBDD candidates selected")

    receptor_map = {}
    for _, row in usable.iterrows():
        info = prepare_receptor(row, args.pdb_cache, work_dir, args.mk_prepare_receptor, args.obabel, args.prep_timeout)
        receptor_map[info["pocket_id"]] = info
    pd.DataFrame(receptor_map.values()).to_csv(report_dir / "receptor_prep.csv", index=False)

    selected = write_selected_sdfs(selected, work_dir / "ligands")
    tasks = [
        {
            "row": row._asdict(),
            "work_dir": work_dir,
            "receptors": receptor_map,
            "vina": args.vina,
            "mk_prepare_ligand": args.mk_prepare_ligand,
            "prep_timeout": args.prep_timeout,
            "vina_timeout": args.vina_timeout,
        }
        for row in selected.itertuples(index=False)
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(score_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())

    results = pd.DataFrame(rows).sort_values(["pocket_id", "mol_idx"])
    overall, rerank = summarize(results, args.topk_fracs)
    results.to_csv(report_dir / "vina_score_results.csv", index=False)
    overall.to_csv(report_dir / "vina_score_summary.csv", index=False)
    rerank.to_csv(report_dir / "vina_reranking_summary.csv", index=False)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_dir": str(args.audit_dir),
                "n_pockets": int(selected["pocket_id"].nunique()),
                "n_candidates": int(len(selected)),
                "note": "Vina score-only check on DiffSBDD official samples limited to public-PDB aligned pockets.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "score_report.md").write_text(
        "\n".join(
            [
                "# DiffSBDD Aligned Vina Score-Only Check",
                "",
                "Only pockets marked usable_public_pdb_alignment in the audit are included.",
                "",
                "## Overall",
                "",
                overall.to_csv(index=False),
                "## Vina Reranking",
                "",
                rerank.to_csv(index=False),
            ]
        ),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))
    print(rerank.to_string(index=False))


if __name__ == "__main__":
    main()
