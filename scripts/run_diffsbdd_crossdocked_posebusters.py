from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402


SCORE_COLS = {
    "qed": "score_qed",
    "qed_sa": "score_qed_sa",
    "mlp_ligand_descriptor": "score_mlp_ligand_descriptor",
    "mlp_pose_shape": "score_mlp_pose_shape",
    "mlp_reliamol_novinascore": "score_mlp_reliamol_novinascore",
    "rf_reliamol_novinascore": "score_rf_reliamol_novinascore",
    "hgb_reliamol_novinascore": "score_hgb_reliamol_novinascore",
}


def patch_posebusters_name() -> None:
    import posebusters.posebusters as pbmod

    def get_name(mol):
        if mol is None or not mol.HasProp("_Name"):
            return ""
        return str(mol.GetProp("_Name"))

    pbmod.PoseBusters._get_name = staticmethod(get_name)


def run_one_sdf(sdf_file: str, config: str, full_report: bool) -> pd.DataFrame:
    patch_posebusters_name()
    from posebusters import PoseBusters

    buster = PoseBusters(config=config, max_workers=0)
    out = buster.bust(Path(sdf_file), full_report=full_report).reset_index()
    out["sdf_file"] = str(Path(sdf_file))
    out["mol_idx"] = out["position"].astype(int)
    return out


def build_posebusters_table(args: argparse.Namespace, report_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    sdf_files = raw["sdf_file"].drop_duplicates().tolist()
    if args.max_files:
        sdf_files = sdf_files[: args.max_files]
    frames = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one_sdf, sdf_file, args.config, args.full_report): sdf_file for sdf_file in sdf_files}
        for future in as_completed(futures):
            sdf_file = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append({"sdf_file": sdf_file, "error": str(exc)[:500]})
    if failures:
        pd.DataFrame(failures).to_csv(report_dir / "posebusters_failures.csv", index=False)
    if not frames:
        raise SystemExit("PoseBusters produced no successful file-level outputs.")
    pb = pd.concat(frames, ignore_index=True)
    raw["candidate_uid"] = raw["pocket_id"].astype(str) + ":" + raw["mol_idx"].astype(str)
    meta_cols = ["sdf_file", "mol_idx", "candidate_uid", "pocket_id", "pdb_id", "chain_id", "public_generated_reliable_proxy"]
    merged = raw[meta_cols].merge(pb, on=["sdf_file", "mol_idx"], how="inner")
    bool_cols = [c for c in merged.columns if c not in set(meta_cols + ["file", "molecule", "position"]) and merged[c].dropna().isin([True, False]).all()]
    for col in bool_cols:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["posebusters_pass"] = merged[bool_cols].all(axis=1)
    merged.to_csv(report_dir / "posebusters_candidates.csv", index=False)
    return merged


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = np.asarray(score).argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def per_pocket_topk(frame: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(np.ceil(len(group) * frac)))
        selected.append(group.nlargest(k, score_col))
    return pd.concat(selected, ignore_index=True)


def summarize_posebusters(pb: pd.DataFrame, args: argparse.Namespace, report_dir: Path) -> None:
    test_cols = [
        c
        for c in pb.columns
        if c
        not in {
            "sdf_file",
            "mol_idx",
            "candidate_uid",
            "pocket_id",
            "pdb_id",
            "chain_id",
            "public_generated_reliable_proxy",
            "file",
            "molecule",
            "position",
            "posebusters_pass",
        }
        and pb[c].dropna().isin([True, False]).all()
    ]
    overall = pd.DataFrame(
        [
            {
                "n_candidates": int(len(pb)),
                "n_pockets": int(pb["pocket_id"].nunique()),
                "posebusters_pass_rate": float(pb["posebusters_pass"].mean()),
                "reliability_proxy_rate": float(pb["public_generated_reliable_proxy"].mean()),
                **{f"{col}_pass_rate": float(pb[col].mean()) for col in test_cols},
            }
        ]
    )
    by_pocket = pb.groupby(["pdb_id", "chain_id", "pocket_id"], sort=False).agg(
        n_candidates=("mol_idx", "count"),
        posebusters_pass_rate=("posebusters_pass", "mean"),
        reliability_proxy_rate=("public_generated_reliable_proxy", "mean"),
    )
    by_pocket = by_pocket.reset_index()
    overall.to_csv(report_dir / "posebusters_overall.csv", index=False)
    by_pocket.to_csv(report_dir / "posebusters_by_pocket.csv", index=False)

    metric_rows = []
    rerank_rows = []
    if args.pred_dir.exists():
        for pred_path in sorted((args.pred_dir / "seeds").glob("seed_*/oof_predictions.csv")):
            seed = int(pred_path.parent.name.split("_")[-1])
            pred = pd.read_csv(pred_path)
            keep_cols = ["candidate_uid", "pocket_id"] + [c for c in SCORE_COLS.values() if c in pred.columns]
            frame = pb.merge(pred[keep_cols], on=["candidate_uid", "pocket_id"], how="inner")
            y = frame["posebusters_pass"].astype(int).to_numpy()
            for method, col in SCORE_COLS.items():
                if col not in frame.columns:
                    continue
                score = frame[col].to_numpy(float)
                prob = rank_to_prob(score) if method in {"qed", "qed_sa"} else score
                row = binary_metrics(y, prob)
                row.update({"seed": seed, "method": method})
                metric_rows.append(row)
                for frac in args.topk_fracs:
                    top = per_pocket_topk(frame, col, frac)
                    rerank_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "topk_frac": frac,
                            "n_selected": int(len(top)),
                            "posebusters_pass_rate": float(top["posebusters_pass"].mean()),
                            "reliable_proxy_rate": float(top["public_generated_reliable_proxy"].mean()),
                        }
                    )
    if metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(report_dir / "posebusters_metrics_by_seed.csv", index=False)
        metric_summary = metrics.groupby("method", sort=False)[["roc_auc", "pr_auc", "ece"]].agg(["mean", "std"]).reset_index()
        metric_summary.columns = ["_".join([str(x) for x in col if x]) for col in metric_summary.columns.to_flat_index()]
        metric_summary.to_csv(report_dir / "posebusters_metric_summary.csv", index=False)
    if rerank_rows:
        rerank = pd.DataFrame(rerank_rows)
        rerank.to_csv(report_dir / "posebusters_reranking_by_seed.csv", index=False)
        rerank_summary = rerank.groupby(["method", "topk_frac"], sort=False)[["posebusters_pass_rate", "reliable_proxy_rate"]].agg(["mean", "std"]).reset_index()
        rerank_summary.columns = ["_".join([str(x) for x in col if x]) for col in rerank_summary.columns.to_flat_index()]
        rerank_summary.to_csv(report_dir / "posebusters_reranking_summary.csv", index=False)
    lines = [
        "# DiffSBDD CrossDocked PoseBusters Validity",
        "",
        f"PoseBusters config: `{args.config}`. This is a molecule-internal physical validity audit; receptor-contact checks remain in the CrossDocked gninatypes audit.",
        "",
        "## Overall",
        "",
        overall.to_csv(index=False),
    ]
    if (report_dir / "posebusters_reranking_summary.csv").exists():
        rerank_summary = pd.read_csv(report_dir / "posebusters_reranking_summary.csv")
        lines += [
            "## Top-10% Reranking",
            "",
            rerank_summary[rerank_summary["topk_frac"].eq(0.1)].to_csv(index=False),
        ]
    (report_dir / "posebusters_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PoseBusters molecule validity checks on DiffSBDD CrossDocked generated candidates.")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_gninatypes_audit_v0/reports"))
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_posebusters_molfast_v0"))
    parser.add_argument("--config", default="mol_fast")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--full-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pb = build_posebusters_table(args, report_dir)
    summarize_posebusters(pb, args, report_dir)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": args.config,
                "audit_dir": str(args.audit_dir),
                "pred_dir": str(args.pred_dir),
                "n_candidates": int(len(pb)),
                "n_pockets": int(pb["pocket_id"].nunique()),
                "note": "PoseBusters _get_name is monkey-patched for compatibility with the RDKit build whose GetProp does not accept autoConvert.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((report_dir / "posebusters_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
