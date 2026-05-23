from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_diffsbdd_crossdocked_reliamol import (  # noqa: E402
    clean_matrix,
    feature_columns,
    fixed_pocket,
    fold_pockets,
    mol_coords_atoms,
    parse_gninatypes,
    pose_shape_features,
    rank_to_prob,
    resolve_path,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from reliamol3d.metrics import binary_metrics  # noqa: E402


SCORE_COLS = {
    "QED/SA": "score_qed_sa",
    "RF ReliaMol": "score_rf_reliamol_novinascore",
    "HGB ReliaMol": "score_hgb_reliamol_novinascore",
}


def make_catalog(catalog_name: str) -> FilterCatalog | None:
    params = FilterCatalogParams()
    cats = FilterCatalogParams.FilterCatalogs
    value = getattr(cats, catalog_name, None)
    if value is None:
        return None
    params.AddCatalog(value)
    return FilterCatalog(params)


def catalog_hits(catalog: FilterCatalog | None, mol: Chem.Mol | None) -> tuple[int, str]:
    if catalog is None or mol is None:
        return 0, ""
    try:
        matches = list(catalog.GetMatches(mol))
    except Exception:
        return 0, ""
    desc = "; ".join(sorted({m.GetDescription() for m in matches}))[:400]
    return len(matches), desc


def safe_sanitize(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    out = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def molecule_record(mol: Chem.Mol | None, catalogs: dict[str, FilterCatalog | None]) -> dict:
    smol = safe_sanitize(mol)
    if smol is None:
        return {
            "canonical_smiles": "",
            "drug_audit_ok": False,
            "mw": math.nan,
            "logp": math.nan,
            "hbd": math.nan,
            "hba": math.nan,
            "tpsa": math.nan,
            "rot_bonds": math.nan,
            "qed_rdkit": math.nan,
            "lipinski_violations": math.nan,
            "lipinski_zero_violation": False,
            "lipinski_le1_violation": False,
            "veber_pass": False,
            "pains_hits": math.nan,
            "brenk_hits": math.nan,
            "nih_hits": math.nan,
            "structural_alert_free": False,
            "murcko_scaffold": "",
        }
    mw = float(Descriptors.MolWt(smol))
    logp = float(Crippen.MolLogP(smol))
    hbd = int(Lipinski.NumHDonors(smol))
    hba = int(Lipinski.NumHAcceptors(smol))
    tpsa = float(rdMolDescriptors.CalcTPSA(smol))
    rot = int(Lipinski.NumRotatableBonds(smol))
    lip_v = int(mw > 500.0) + int(logp > 5.0) + int(hbd > 5) + int(hba > 10)
    pains_n, pains_desc = catalog_hits(catalogs.get("PAINS"), smol)
    brenk_n, brenk_desc = catalog_hits(catalogs.get("BRENK"), smol)
    nih_n, nih_desc = catalog_hits(catalogs.get("NIH"), smol)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=smol)
    except Exception:
        scaffold = ""
    return {
        "canonical_smiles": Chem.MolToSmiles(smol, isomericSmiles=True),
        "drug_audit_ok": True,
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rot_bonds": rot,
        "qed_rdkit": float(QED.qed(smol)),
        "lipinski_violations": lip_v,
        "lipinski_zero_violation": lip_v == 0,
        "lipinski_le1_violation": lip_v <= 1,
        "veber_pass": bool(rot <= 10 and tpsa <= 140.0),
        "pains_hits": pains_n,
        "pains_desc": pains_desc,
        "brenk_hits": brenk_n,
        "brenk_desc": brenk_desc,
        "nih_hits": nih_n,
        "nih_desc": nih_desc,
        "structural_alert_free": pains_n == 0 and brenk_n == 0 and nih_n == 0,
        "murcko_scaffold": scaffold,
    }


def build_drug_table(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    catalogs = {
        "PAINS": make_catalog("PAINS"),
        "BRENK": make_catalog("BRENK"),
        "NIH": make_catalog("NIH"),
    }
    rows = []
    for sdf_file, group in frame.groupby("sdf_file", sort=False):
        sdf_path = resolve_path(str(sdf_file), root)
        mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False))
        for row in group[["candidate_uid", "pocket_id", "mol_idx"]].itertuples(index=False):
            mol_idx = int(row.mol_idx)
            mol = mols[mol_idx] if mol_idx < len(mols) else None
            rec = molecule_record(mol, catalogs)
            rec.update({"candidate_uid": row.candidate_uid, "pocket_id": row.pocket_id, "mol_idx": mol_idx})
            rows.append(rec)
    return pd.DataFrame(rows)


def per_pocket_select(frame: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("pocket_id", sort=False):
        k = max(1, int(np.ceil(len(group) * frac)))
        selected.append(group.nlargest(k, score_col))
    return pd.concat(selected, ignore_index=True)


def summarize_selection(pred_frames: dict[int, pd.DataFrame], drug: pd.DataFrame, topk_fracs: list[float]) -> pd.DataFrame:
    rows = []
    base = next(iter(pred_frames.values()))
    full = base.merge(drug, on=["candidate_uid", "pocket_id", "mol_idx"], how="left", suffixes=("", "_drug"))
    rows.append(summary_row(full, "Raw pool", 1.0, seed=-1))
    for seed, pred in pred_frames.items():
        merged = pred.merge(drug, on=["candidate_uid", "pocket_id", "mol_idx"], how="left", suffixes=("", "_drug"))
        for method, col in SCORE_COLS.items():
            if col not in merged.columns:
                continue
            for frac in topk_fracs:
                rows.append(summary_row(per_pocket_select(merged, col, frac), method, frac, seed))
    out = pd.DataFrame(rows)
    keys = ["method", "topk_frac"]
    metric_cols = [c for c in out.columns if c not in {"method", "topk_frac", "seed", "n_selected"}]
    agg = out.groupby(keys, sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns.to_flat_index()]
    nsel = out.groupby(keys, sort=False)["n_selected"].mean().reset_index(name="n_selected_mean")
    return out, agg.merge(nsel, on=keys, how="left")


def summary_row(df: pd.DataFrame, method: str, frac: float, seed: int) -> dict:
    return {
        "method": method,
        "topk_frac": frac,
        "seed": seed,
        "n_selected": int(len(df)),
        "reliable_rate": float(df["reliable"].mean()),
        "qed_mean": float(df["qed"].mean()),
        "sa_proxy_mean": float(df["sa_proxy"].mean()),
        "mw_mean": float(df["mw"].mean()),
        "logp_mean": float(df["logp"].mean()),
        "tpsa_mean": float(df["tpsa"].mean()),
        "rot_bonds_mean": float(df["rot_bonds"].mean()),
        "lipinski_zero_violation_rate": float(df["lipinski_zero_violation"].mean()),
        "lipinski_le1_violation_rate": float(df["lipinski_le1_violation"].mean()),
        "veber_pass_rate": float(df["veber_pass"].mean()),
        "pains_free_rate": float((df["pains_hits"].fillna(999) == 0).mean()),
        "brenk_free_rate": float((df["brenk_hits"].fillna(999) == 0).mean()),
        "nih_free_rate": float((df["nih_hits"].fillna(999) == 0).mean()),
        "structural_alert_free_rate": float(df["structural_alert_free"].mean()),
        "murcko_scaffolds_per_pocket": float(df.groupby("pocket_id")["murcko_scaffold"].nunique().mean()),
    }


def call_admetlab(smiles: list[str], sleep_s: float) -> tuple[pd.DataFrame, dict]:
    if not smiles:
        return pd.DataFrame(), {"status": "skipped", "reason": "no smiles"}
    payload = {"SMILES": smiles, "feature": False, "uncertain": False}
    try:
        response = requests.post("https://admetlab3.scbdd.com/api/admet", json=payload, timeout=180)
        meta = {"http_status": response.status_code, "text_head": response.text[:500]}
        if response.status_code != 200:
            return pd.DataFrame(), {"status": "failed", **meta}
        js = response.json()
        meta.update({"api_status": js.get("status"), "api_code": js.get("code")})
        data = js.get("data", {}).get("data", [])
        time.sleep(sleep_s)
        return pd.DataFrame(data), {"status": "success", **meta, "n_rows": len(data)}
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), {"status": "failed", "error": str(exc)[:500]}


def run_admetlab_subset(selected: pd.DataFrame, max_mols: int, chunk_size: int, sleep_s: float, report_dir: Path) -> None:
    unique = selected.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").copy()
    unique = unique[unique["canonical_smiles"].astype(str).str.len().gt(0)].head(max_mols)
    frames = []
    metas = []
    smiles = unique["canonical_smiles"].tolist()
    for start in range(0, len(smiles), chunk_size):
        chunk = smiles[start : start + chunk_size]
        df, meta = call_admetlab(chunk, sleep_s=sleep_s)
        meta.update({"start": start, "n_request": len(chunk)})
        metas.append(meta)
        if not df.empty:
            frames.append(df)
    pd.DataFrame(metas).to_csv(report_dir / "admetlab_api_status.csv", index=False)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(report_dir / "admetlab_raw.csv", index=False)
        summarize_admetlab(out).to_csv(report_dir / "admetlab_endpoint_summary.csv", index=False)


def summarize_admetlab(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skip = {"smiles", "structure"}
    for col in df.columns:
        if col in skip:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        rows.append(
            {
                "endpoint": col,
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "pass_decision_rate_if_0": float((vals == 0).mean()),
                "danger_decision_rate_if_1": float((vals == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def recompute_pose_frame(base: pd.DataFrame, sigma: float, mode: str, seed: int, pocket_radius: float, root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    receptor_cache = {}
    pocket_cache = {}
    for pocket_id, group in base.groupby("pocket_id", sort=False):
        receptor_path = resolve_path(str(group["receptor_gninatypes"].iloc[0]), root)
        receptor = receptor_cache.get(receptor_path)
        if receptor is None:
            receptor = parse_gninatypes(receptor_path, include_hydrogens=False)
            receptor_cache[receptor_path] = receptor
        receptor_use = receptor.copy()
        if mode in {"receptor", "both"} and sigma > 0:
            receptor_use = receptor_use + rng.normal(0.0, sigma, size=receptor_use.shape).astype(np.float32)
        center = group[["center_x", "center_y", "center_z"]].median().to_numpy(np.float32)
        pocket_cache[pocket_id] = fixed_pocket(receptor_use, center, pocket_radius)

    rows = []
    pose_cols = feature_columns("pose_shape")
    pose_cols = [c for c in pose_cols if c in base.columns or c.startswith("atom_") or c in {"pose_atoms", "pose_radius_shape"}]
    for sdf_file, group in base.groupby("sdf_file", sort=False):
        sdf_path = resolve_path(str(sdf_file), root)
        mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False))
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            mol_idx = int(row_dict["mol_idx"])
            mol = mols[mol_idx] if mol_idx < len(mols) else None
            coords, atoms = mol_coords_atoms(mol)
            if mode in {"ligand", "both"} and sigma > 0 and len(coords):
                coords = coords + rng.normal(0.0, sigma, size=coords.shape).astype(np.float32)
            row_dict.update(pose_shape_features(coords, atoms, pocket_cache[str(row_dict["pocket_id"])]))
            rows.append(row_dict)
    return pd.DataFrame(rows)


def topk_reliable(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    return float(per_pocket_select(frame, score_col, frac)["reliable"].mean())


def run_coordinate_noise(base: pd.DataFrame, sigmas: list[float], seeds: list[int], mode: str, pocket_radius: float, report_dir: Path, root: Path) -> pd.DataFrame:
    rows = []
    features = feature_columns("reliamol_novinascore")
    for sigma in sigmas:
        pert = recompute_pose_frame(base, sigma=sigma, mode=mode, seed=20260522 + int(round(sigma * 1000)), pocket_radius=pocket_radius, root=root)
        for seed in seeds:
            pockets = base["pocket_id"].drop_duplicates().to_numpy()
            folds = fold_pockets(pockets, 5, seed)
            pred = pert.copy()
            pred["score_rf_reliamol_novinascore"] = np.nan
            pred["score_qed_sa"] = pred["qed"].astype(float) - 0.25 * pred["sa_proxy"].astype(float)
            for fold_idx, heldout in enumerate(folds):
                test_mask = base["pocket_id"].isin(set(heldout))
                train = base.loc[~test_mask].copy()
                test = pert.loc[test_mask].copy()
                x_train, x_test = clean_matrix(train, test, features)
                rf = RandomForestClassifier(
                    n_estimators=260,
                    min_samples_leaf=3,
                    n_jobs=16,
                    random_state=seed + fold_idx,
                    class_weight="balanced",
                )
                rf.fit(x_train, train["reliable"])
                pred.loc[test.index, "score_rf_reliamol_novinascore"] = rf.predict_proba(x_test)[:, 1]
            y = pred["reliable"].to_numpy()
            rf_metrics = binary_metrics(y, pred["score_rf_reliamol_novinascore"].to_numpy(float))
            qsa_metrics = binary_metrics(y, rank_to_prob(pred["score_qed_sa"].to_numpy(float)))
            rows.append(
                {
                    "mode": mode,
                    "sigma_angstrom": sigma,
                    "seed": seed,
                    "method": "RF ReliaMol",
                    "top10_reliable": topk_reliable(pred, "score_rf_reliamol_novinascore", 0.1),
                    "top20_reliable": topk_reliable(pred, "score_rf_reliamol_novinascore", 0.2),
                    "roc_auc": rf_metrics["roc_auc"],
                    "pr_auc": rf_metrics["pr_auc"],
                    "ece": rf_metrics["ece"],
                }
            )
            rows.append(
                {
                    "mode": mode,
                    "sigma_angstrom": sigma,
                    "seed": seed,
                    "method": "QED/SA",
                    "top10_reliable": topk_reliable(pred, "score_qed_sa", 0.1),
                    "top20_reliable": topk_reliable(pred, "score_qed_sa", 0.2),
                    "roc_auc": qsa_metrics["roc_auc"],
                    "pr_auc": qsa_metrics["pr_auc"],
                    "ece": qsa_metrics["ece"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(report_dir / "coordinate_noise_true_by_seed.csv", index=False)
    summary = out.groupby(["mode", "sigma_angstrom", "method"], sort=False)[["top10_reliable", "top20_reliable", "roc_auc", "pr_auc", "ece"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    summary.to_csv(report_dir / "coordinate_noise_true_summary.csv", index=False)
    return summary


def make_report(report_dir: Path, drug_summary: pd.DataFrame, coord_summary: pd.DataFrame) -> None:
    lines = [
        "# Drug-discovery alignment audits",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "All calculations were run on the remote repository environment. RDKit is used for medicinal-chemistry rules, structural alerts, Murcko scaffolds, and true coordinate perturbation feature recomputation.",
        "",
        "## Selection drug-likeness summary",
        "",
        drug_summary.to_csv(index=False),
        "",
        "## True coordinate perturbation summary",
        "",
        coord_summary.to_csv(index=False),
    ]
    status = report_dir / "admetlab_api_status.csv"
    if status.exists():
        lines += ["", "## ADMETlab API status", "", pd.read_csv(status).to_csv(index=False)]
    (report_dir / "drug_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-journal drug-discovery alignment audits for ReliaMol-3D.")
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_reliamol_v0/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffsbdd_crossdocked_drug_alignment_v0"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--topk-fracs", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--noise-sigmas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--noise-mode", choices=["ligand", "receptor", "both"], default="both")
    parser.add_argument("--pocket-radius", type=float, default=12.0)
    parser.add_argument("--run-admetlab", action="store_true")
    parser.add_argument("--admetlab-max-mols", type=int, default=600)
    parser.add_argument("--admetlab-chunk-size", type=int, default=50)
    parser.add_argument("--admetlab-sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_frames = {}
    for seed in args.seeds:
        path = args.pred_dir / "seeds" / f"seed_{seed}" / "oof_predictions.csv"
        pred_frames[seed] = pd.read_csv(path)
    base = next(iter(pred_frames.values()))
    drug = build_drug_table(base, root=root)
    drug.to_csv(report_dir / "candidate_drug_alerts.csv", index=False)
    drug_by_seed, drug_summary = summarize_selection(pred_frames, drug, args.topk_fracs)
    drug_by_seed.to_csv(report_dir / "druglikeness_selection_by_seed.csv", index=False)
    drug_summary.to_csv(report_dir / "druglikeness_selection_summary.csv", index=False)

    if args.run_admetlab:
        merged = pred_frames[args.seeds[0]].merge(drug, on=["candidate_uid", "pocket_id", "mol_idx"], how="left", suffixes=("", "_drug"))
        selected = per_pocket_select(merged, "score_rf_reliamol_novinascore", 0.1)
        run_admetlab_subset(selected, args.admetlab_max_mols, args.admetlab_chunk_size, args.admetlab_sleep, report_dir)

    coord_summary = run_coordinate_noise(
        base,
        sigmas=args.noise_sigmas,
        seeds=args.seeds,
        mode=args.noise_mode,
        pocket_radius=args.pocket_radius,
        report_dir=report_dir,
        root=root,
    )
    make_report(report_dir, drug_summary, coord_summary)
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "pred_dir": str(args.pred_dir),
                "n_candidates": int(len(base)),
                "n_pockets": int(base["pocket_id"].nunique()),
                "rdkit_note": "RDKit-pypi was installed in the remote project .conda_env for this audit.",
                "noise_mode": args.noise_mode,
                "noise_sigmas": args.noise_sigmas,
                "admetlab_requested": bool(args.run_admetlab),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((report_dir / "drug_alignment_report.md").read_text(encoding="utf-8")[:4000])


if __name__ == "__main__":
    main()
