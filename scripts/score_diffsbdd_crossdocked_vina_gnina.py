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
from sklearn.metrics import average_precision_score, roc_auc_score


RELIA_SCORE_COLS = {
    "qed": "score_qed",
    "qed_sa": "score_qed_sa",
    "mlp_pose_shape": "score_mlp_pose_shape",
    "mlp_reliamol_novinascore": "score_mlp_reliamol_novinascore",
    "rf_reliamol_novinascore": "score_rf_reliamol_novinascore",
    "hgb_reliamol_novinascore": "score_hgb_reliamol_novinascore",
}


def safe_name(text: str, max_len: int = 150) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:max_len].strip("._") or "item"


def run_cmd(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def diff_pocket_base(pocket_id: str) -> str:
    base = pocket_id.split("_", 1)[0]
    if base.endswith("-pocket10"):
        base = base[: -len("-pocket10")]
    return base.replace("-", "_")


def extract_if3_pockets(args: argparse.Namespace, meta: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in meta.itertuples(index=False):
        pocket_dir = str(rec.tar_member).rsplit("/", 1)[0]
        member = f"crossdocked_pocket10/{pocket_dir}/{diff_pocket_base(str(rec.pocket_id))}_pocket10.pdb"
        local = args.if3_extract_dir / member
        rows.append({"pocket_id": rec.pocket_id, "if3_member": member, "pocket_pdb": str(local)})
    plan = pd.DataFrame(rows)
    missing = plan[~plan["pocket_pdb"].map(lambda p: Path(p).exists())]
    if len(missing):
        args.if3_extract_dir.mkdir(parents=True, exist_ok=True)
        files_from = report_dir / "if3_selected_members.txt"
        files_from.write_text("\n".join("./" + x for x in missing["if3_member"]) + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["tar", "-xzf", str(args.if3_archive), "-C", str(args.if3_extract_dir), "--files-from", str(files_from)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"IF3 pocket extraction failed: {proc.stderr[-1000:]}")
    plan["pocket_pdb_exists"] = plan["pocket_pdb"].map(lambda p: Path(p).exists())
    plan.to_csv(report_dir / "if3_pocket_plan.csv", index=False)
    return plan


def prepare_receptors(plan: pd.DataFrame, args: argparse.Namespace, report_dir: Path) -> dict[str, dict]:
    rows = []
    for rec in plan.itertuples(index=False):
        pocket_id = str(rec.pocket_id)
        work = args.output_dir / "work" / "receptors" / safe_name(pocket_id)
        work.mkdir(parents=True, exist_ok=True)
        out_base = work / "receptor"
        pdbqt = out_base.with_suffix(".pdbqt")
        status = "cached" if pdbqt.exists() and pdbqt.stat().st_size else "missing"
        stderr_tail = ""
        if status == "missing":
            cmd = [str(args.mk_prepare_receptor), "--read_pdb", str(rec.pocket_pdb), "-o", str(out_base), "-p"]
            code, stdout, stderr = run_cmd(cmd, args.receptor_timeout)
            (work / "mk_prepare_receptor.stdout.txt").write_text(stdout, encoding="utf-8")
            (work / "mk_prepare_receptor.stderr.txt").write_text(stderr, encoding="utf-8")
            status = "success" if code == 0 and pdbqt.exists() and pdbqt.stat().st_size else "failed"
            stderr_tail = stderr[-500:]
            if status == "failed":
                fallback = work / "receptor_obabel.pdbqt"
                fcmd = [str(args.obabel), str(rec.pocket_pdb), "-O", str(fallback), "-xr"]
                fcode, fstdout, fstderr = run_cmd(fcmd, args.receptor_timeout)
                (work / "obabel_receptor.stdout.txt").write_text(fstdout, encoding="utf-8")
                (work / "obabel_receptor.stderr.txt").write_text(fstderr, encoding="utf-8")
                if fcode == 0 and fallback.exists() and fallback.stat().st_size:
                    pdbqt = fallback
                    status = "obabel_fallback"
                    stderr_tail = fstderr[-500:]
        rows.append(
            {
                "pocket_id": pocket_id,
                "pocket_pdb": str(rec.pocket_pdb),
                "receptor_pdbqt": str(pdbqt),
                "receptor_prep_status": status,
                "receptor_prep_error_tail": stderr_tail,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "receptor_preparation.csv", index=False)
    return {str(r["pocket_id"]): r for r in table.to_dict("records")}


def write_single_sdfs(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    ligand_dir = args.output_dir / "work" / "ligands_sdf"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    for sdf_file, group in candidates.groupby("sdf_file", sort=False):
        supplier = Chem.SDMolSupplier(str(sdf_file), sanitize=False, removeHs=False)
        needed = {int(x) for x in group["mol_idx"]}
        mols = {idx: mol for idx, mol in enumerate(supplier) if idx in needed and mol is not None}
        for row in group.itertuples(index=False):
            mol_idx = int(row.mol_idx)
            out = ligand_dir / safe_name(str(row.pocket_id)) / f"mol_{mol_idx:03d}.sdf"
            out.parent.mkdir(parents=True, exist_ok=True)
            status = "cached" if out.exists() and out.stat().st_size else "missing"
            if status == "missing":
                mol = mols.get(mol_idx)
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
                    status = "success"
            rec = row._asdict()
            rec["single_sdf"] = str(out)
            rec["single_sdf_status"] = status
            rows.append(rec)
    return pd.DataFrame(rows)


def parse_vina(text: str) -> dict[str, float]:
    out = {"vina_affinity": math.nan, "vina_inter": math.nan, "vina_intra": math.nan, "vina_torsions": math.nan}

    def value(line: str) -> float:
        match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)", line)
        return float(match.group(1)) if match else math.nan

    for line in text.splitlines():
        lower = line.strip().lower()
        val = value(line)
        if math.isnan(val):
            continue
        if lower.startswith("affinity:") or lower.startswith("estimated free energy of binding"):
            out["vina_affinity"] = val
        elif lower.startswith("intermolecular") or "final intermolecular energy" in lower:
            out["vina_inter"] = val
        elif lower.startswith("intramolecular") or "final total internal energy" in lower:
            out["vina_intra"] = val
        elif lower.startswith("torsional") or "torsional free energy" in lower:
            out["vina_torsions"] = val
    return out


def parse_gnina(text: str) -> dict[str, float]:
    out = {"gnina_affinity": math.nan, "gnina_cnnscore": math.nan, "gnina_cnnaffinity": math.nan}
    patterns = {
        "gnina_affinity": r"Affinity:\s*([-+]?\d+(?:\.\d+)?)",
        "gnina_cnnscore": r"CNNscore:\s*([-+]?\d+(?:\.\d+)?)",
        "gnina_cnnaffinity": r"CNNaffinity:\s*([-+]?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def score_one(task: dict) -> dict:
    row = task["row"].copy()
    pocket_id = str(row["pocket_id"])
    mol_idx = int(row["mol_idx"])
    receptor = task["receptors"].get(pocket_id, {})
    work = task["output_dir"] / "work" / "scoring" / safe_name(pocket_id) / f"mol_{mol_idx:03d}"
    work.mkdir(parents=True, exist_ok=True)
    ligand_pdbqt = work / "ligand.pdbqt"
    out = {
        **row,
        "ligand_pdbqt": str(ligand_pdbqt),
        "receptor_pdbqt": receptor.get("receptor_pdbqt", ""),
        "pocket_pdb": receptor.get("pocket_pdb", ""),
        "ligand_prep_status": "not_run",
        "vina_status": "not_run",
        "gnina_status": "not_run",
    }
    if row.get("single_sdf_status") not in {"success", "cached"}:
        out["ligand_prep_status"] = "missing_mol"
        return out
    if receptor.get("receptor_prep_status") not in {"success", "cached", "obabel_fallback"}:
        out["vina_status"] = "no_receptor"
    if not Path(receptor.get("pocket_pdb", "")).exists():
        out["gnina_status"] = "no_receptor"

    if not ligand_pdbqt.exists() or not ligand_pdbqt.stat().st_size:
        try:
            code, stdout, stderr = run_cmd(
                [str(task["mk_prepare_ligand"]), "-i", str(row["single_sdf"]), "-o", str(ligand_pdbqt)],
                task["prep_timeout"],
            )
        except subprocess.TimeoutExpired:
            out["ligand_prep_status"] = "timeout"
            return out
        (work / "ligand_prep.stdout.txt").write_text(stdout, encoding="utf-8")
        (work / "ligand_prep.stderr.txt").write_text(stderr, encoding="utf-8")
        if code != 0 or not ligand_pdbqt.exists() or not ligand_pdbqt.stat().st_size:
            out["ligand_prep_status"] = "failed"
            out["error_tail"] = stderr[-500:]
            return out
    out["ligand_prep_status"] = "ok"

    vina_log = work / "vina_score.stdout.txt"
    vina_err = work / "vina_score.stderr.txt"
    if out["vina_status"] != "no_receptor":
        if vina_log.exists() and vina_log.stat().st_size:
            stdout = vina_log.read_text(encoding="utf-8", errors="ignore")
            stderr = vina_err.read_text(encoding="utf-8", errors="ignore") if vina_err.exists() else ""
            scores = parse_vina(stdout + "\n" + stderr)
            out.update(scores)
            out["vina_status"] = "success" if not math.isnan(scores["vina_affinity"]) else "failed_cached"
        else:
            cmd = [
                str(task["vina"]),
                "--receptor",
                str(receptor["receptor_pdbqt"]),
                "--ligand",
                str(ligand_pdbqt),
                "--score_only",
                "--autobox",
                "--cpu",
                "1",
            ]
            try:
                code, stdout, stderr = run_cmd(cmd, task["score_timeout"])
            except subprocess.TimeoutExpired:
                out["vina_status"] = "timeout"
                stdout, stderr = "", ""
            vina_log.write_text(stdout, encoding="utf-8")
            vina_err.write_text(stderr, encoding="utf-8")
            scores = parse_vina(stdout + "\n" + stderr)
            out.update(scores)
            if out["vina_status"] != "timeout":
                out["vina_status"] = "success" if code == 0 and not math.isnan(scores["vina_affinity"]) else "failed"

    gnina_log = work / "gnina_score.stdout.txt"
    gnina_err = work / "gnina_score.stderr.txt"
    if out["gnina_status"] != "no_receptor":
        if gnina_log.exists() and gnina_log.stat().st_size:
            stdout = gnina_log.read_text(encoding="utf-8", errors="ignore")
            stderr = gnina_err.read_text(encoding="utf-8", errors="ignore") if gnina_err.exists() else ""
            scores = parse_gnina(stdout + "\n" + stderr)
            out.update(scores)
            out["gnina_status"] = "success" if not math.isnan(scores["gnina_cnnscore"]) else "failed_cached"
        else:
            cmd = [
                str(task["gnina"]),
                "--receptor",
                str(receptor["pocket_pdb"]),
                "--ligand",
                str(row["single_sdf"]),
                "--score_only",
                "--cnn_scoring",
                "rescore",
                "--cnn",
                str(task["gnina_cnn"]),
                "--no_gpu",
            ]
            try:
                code, stdout, stderr = run_cmd(cmd, task["score_timeout"])
            except subprocess.TimeoutExpired:
                out["gnina_status"] = "timeout"
                stdout, stderr = "", ""
            gnina_log.write_text(stdout, encoding="utf-8")
            gnina_err.write_text(stderr, encoding="utf-8")
            scores = parse_gnina(stdout + "\n" + stderr)
            out.update(scores)
            if out["gnina_status"] != "timeout":
                out["gnina_status"] = "success" if code == 0 and not math.isnan(scores["gnina_cnnscore"]) else "failed"
    return out


def rank01(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="average", ascending=ascending, pct=True)


def add_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["score_vina"] = -out["vina_affinity"].astype(float)
    out["score_gnina_affinity"] = -out["gnina_affinity"].astype(float)
    out["score_gnina_cnnscore"] = out["gnina_cnnscore"].astype(float)
    out["score_gnina_cnnaffinity"] = out["gnina_cnnaffinity"].astype(float)
    return out


def metric_row(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(score)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return {"roc_auc": math.nan, "pr_auc": math.nan}
    return {"roc_auc": float(roc_auc_score(y[mask], score[mask])), "pr_auc": float(average_precision_score(y[mask], score[mask]))}


def rerank(frame: pd.DataFrame, score_col: str, topk_fracs: list[float]) -> pd.DataFrame:
    rows = []
    valid = frame[np.isfinite(frame[score_col])].copy()
    for frac in topk_fracs:
        selected = []
        for _, group in valid.groupby("pocket_id", sort=False):
            k = max(1, int(np.ceil(len(group) * frac)))
            selected.append(group.nlargest(k, score_col))
        top = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
        rows.append(
            {
                "method": score_col.replace("score_", ""),
                "topk_frac": frac,
                "n_selected": int(len(top)),
                "reliable_proxy_rate": float(top["public_generated_reliable_proxy"].mean()) if len(top) else math.nan,
                "posebusters_pass_rate": float(top["posebusters_pass"].mean()) if "posebusters_pass" in top and len(top) else math.nan,
                "clash_free_rate": float((~top["clash_failure_proxy"]).mean()) if len(top) else math.nan,
                "median_vina_affinity": float(top["vina_affinity"].median()) if "vina_affinity" in top and len(top) else math.nan,
                "median_gnina_cnnscore": float(top["gnina_cnnscore"].median()) if "gnina_cnnscore" in top and len(top) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, args: argparse.Namespace, report_dir: Path) -> None:
    frame = add_score_columns(results)
    pb_path = args.posebusters_dir / "posebusters_candidates.csv"
    if pb_path.exists():
        pb = pd.read_csv(pb_path)[["candidate_uid", "pocket_id", "posebusters_pass"]]
        frame = frame.merge(pb, on=["candidate_uid", "pocket_id"], how="left")
    score_cols = ["score_vina", "score_gnina_affinity", "score_gnina_cnnscore", "score_gnina_cnnaffinity"]
    metric_rows = []
    reranks = []
    y = frame["public_generated_reliable_proxy"].astype(int).to_numpy()
    for col in score_cols:
        row = metric_row(y, frame[col].to_numpy(float))
        row.update({"method": col.replace("score_", "")})
        metric_rows.append(row)
        reranks.append(rerank(frame, col, args.topk_fracs))
    metric_summary = pd.DataFrame(metric_rows)
    rerank_summary = pd.concat(reranks, ignore_index=True)

    # Seed-aware ReliaMol + score-function rank ensembles.
    ensemble_rows = []
    if args.pred_dir.exists():
        base_cols = ["candidate_uid", "pocket_id"] + score_cols + ["public_generated_reliable_proxy", "clash_failure_proxy"]
        if "posebusters_pass" in frame.columns:
            base_cols.append("posebusters_pass")
        base = frame[base_cols].copy()
        for pred_path in sorted((args.pred_dir / "seeds").glob("seed_*/oof_predictions.csv")):
            seed = int(pred_path.parent.name.split("_")[-1])
            pred = pd.read_csv(pred_path)
            cols = ["candidate_uid", "pocket_id"] + [c for c in RELIA_SCORE_COLS.values() if c in pred.columns]
            merged = base.merge(pred[cols], on=["candidate_uid", "pocket_id"], how="inner")
            for relia_name, relia_col in RELIA_SCORE_COLS.items():
                if relia_col not in merged.columns:
                    continue
                for dock_col in ["score_vina", "score_gnina_cnnscore", "score_gnina_cnnaffinity"]:
                    tmp = merged.copy()
                    parts = []
                    for _, group in tmp.groupby("pocket_id", sort=False):
                        g = group.copy()
                        g["_relia_rank"] = rank01(g[relia_col], ascending=True)
                        g["_dock_rank"] = rank01(g[dock_col], ascending=True)
                        g["_ensemble"] = 0.5 * g["_relia_rank"] + 0.5 * g["_dock_rank"]
                        parts.append(g)
                    tmp = pd.concat(parts, ignore_index=True)
                    method = f"ensemble_{relia_name}_{dock_col.replace('score_', '')}"
                    for frac in args.topk_fracs:
                        top_parts = []
                        for _, group in tmp.groupby("pocket_id", sort=False):
                            k = max(1, int(np.ceil(len(group) * frac)))
                            top_parts.append(group.nlargest(k, "_ensemble"))
                        top = pd.concat(top_parts, ignore_index=True)
                        ensemble_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "topk_frac": frac,
                                "n_selected": int(len(top)),
                                "reliable_proxy_rate": float(top["public_generated_reliable_proxy"].mean()),
                                "posebusters_pass_rate": float(top["posebusters_pass"].mean()) if "posebusters_pass" in top else math.nan,
                                "clash_free_rate": float((~top["clash_failure_proxy"]).mean()),
                                "median_vina_affinity": float(top["vina_affinity"].median()) if "vina_affinity" in top else math.nan,
                                "median_gnina_cnnscore": float(top["gnina_cnnscore"].median()) if "gnina_cnnscore" in top else math.nan,
                            }
                        )
    ensemble = pd.DataFrame(ensemble_rows)
    if len(ensemble):
        ensemble.to_csv(report_dir / "ensemble_reranking_by_seed.csv", index=False)
        ens_sum = ensemble.groupby(["method", "topk_frac"], sort=False)[
            ["reliable_proxy_rate", "posebusters_pass_rate", "clash_free_rate", "median_vina_affinity", "median_gnina_cnnscore"]
        ].agg(["mean", "std"]).reset_index()
        ens_sum.columns = ["_".join([str(x) for x in col if x]) for col in ens_sum.columns.to_flat_index()]
        ens_sum.to_csv(report_dir / "ensemble_reranking_summary.csv", index=False)
    else:
        ens_sum = pd.DataFrame()

    overall = pd.DataFrame(
        [
            {
                "n_candidates": int(len(frame)),
                "n_pockets": int(frame["pocket_id"].nunique()),
                "ligand_prep_success_rate": float(frame["ligand_prep_status"].eq("ok").mean()),
                "vina_success_rate": float(frame["vina_status"].eq("success").mean()),
                "gnina_success_rate": float(frame["gnina_status"].eq("success").mean()),
                "reliable_proxy_rate": float(frame["public_generated_reliable_proxy"].mean()),
                "median_vina_affinity": float(frame.loc[frame["vina_status"].eq("success"), "vina_affinity"].median()),
                "median_gnina_cnnscore": float(frame.loc[frame["gnina_status"].eq("success"), "gnina_cnnscore"].median()),
            }
        ]
    )
    frame.to_csv(report_dir / "vina_gnina_scores.csv", index=False)
    overall.to_csv(report_dir / "overall_summary.csv", index=False)
    metric_summary.to_csv(report_dir / "metric_summary.csv", index=False)
    rerank_summary.to_csv(report_dir / "reranking_summary.csv", index=False)
    lines = [
        "# DiffSBDD CrossDocked 100-Pocket Vina/GNINA Baseline",
        "",
        "Score-only Vina and GNINA/CNN baselines on the same 100 exact CrossDocked pocket PDBs used for the gninatypes audit.",
        "",
        "## Overall",
        "",
        overall.to_csv(index=False),
        "## Score Metrics",
        "",
        metric_summary.to_csv(index=False),
        "## Top-10% Score-Only Reranking",
        "",
        rerank_summary[rerank_summary["topk_frac"].eq(0.1)].to_csv(index=False),
    ]
    if len(ens_sum):
        lines += [
            "## Top-10% ReliaMol + Docking/CNN Ensembles",
            "",
            ens_sum[ens_sum["topk_frac"].eq(0.1)].sort_values("reliable_proxy_rate_mean", ascending=False).head(20).to_csv(index=False),
        ]
    (report_dir / "vina_gnina_report.md").write_text("\n".join(lines), encoding="utf-8")
    print((report_dir / "vina_gnina_report.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score official DiffSBDD CrossDocked samples with 100-pocket exact Vina/GNINA baselines.")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports"))
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports"))
    parser.add_argument("--posebusters-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_posebusters_molfast_v0/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_vina_gnina_100pocket_v0"))
    parser.add_argument("--if3-archive", type=Path, default=Path("/home/test/wsk/public_generators/if3_crossdocked2020/crossdocked_pocket10.tar.gz"))
    parser.add_argument("--if3-extract-dir", type=Path, default=Path("/home/test/wsk/public_generators/if3_crossdocked2020/selected_pockets"))
    parser.add_argument("--vina", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_vina_screen_v2/bin/vina"))
    parser.add_argument("--gnina", type=Path, default=Path("/home/test/miniconda3/envs/posebench310/bin/gnina"))
    parser.add_argument("--gnina-cnn", default="crossdock_default2018")
    parser.add_argument("--mk-prepare-ligand", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/mk_prepare_ligand.py"))
    parser.add_argument("--mk-prepare-receptor", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/mk_prepare_receptor.py"))
    parser.add_argument("--obabel", type=Path, default=Path("/home/test/wsk/TASC-VS/environment/tasvs_rtm_rescore/bin/obabel"))
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--prep-timeout", type=int, default=90)
    parser.add_argument("--score-timeout", type=int, default=180)
    parser.add_argument("--receptor-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.audit_dir / "pocket_metadata.csv")
    plan = extract_if3_pockets(args, meta, report_dir)
    receptors = prepare_receptors(plan, args, report_dir)
    candidates = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    candidates["candidate_uid"] = candidates["pocket_id"].astype(str) + ":" + candidates["mol_idx"].astype(str)
    if args.max_candidates:
        candidates = candidates.head(args.max_candidates).copy()
    candidates = candidates.merge(plan[["pocket_id", "pocket_pdb"]], on="pocket_id", how="left")
    selected = write_single_sdfs(candidates, args)
    tasks = [
        {
            "row": rec._asdict(),
            "receptors": receptors,
            "output_dir": args.output_dir,
            "mk_prepare_ligand": args.mk_prepare_ligand,
            "vina": args.vina,
            "gnina": args.gnina,
            "gnina_cnn": args.gnina_cnn,
            "prep_timeout": args.prep_timeout,
            "score_timeout": args.score_timeout,
        }
        for rec in selected.itertuples(index=False)
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(score_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 500 == 0:
                print(f"scored {len(rows)}/{len(tasks)}", flush=True)
    results = pd.DataFrame(rows)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_tasks": len(tasks),
                "if3_archive": str(args.if3_archive),
                "if3_extract_dir": str(args.if3_extract_dir),
                "vina": str(args.vina),
                "gnina": str(args.gnina),
                "gnina_cnn": args.gnina_cnn,
                "note": "IF3 pocket PDBs were used only after coordinate identity to CrossDocked gninatypes was verified in if3_pocket_pdb_alignment.csv.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summarize(results, args, report_dir)


if __name__ == "__main__":
    main()
