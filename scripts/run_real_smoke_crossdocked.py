from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from reliamol3d.metrics import FAILURE_COLUMNS, binary_metrics, failure_metrics, reranking_metrics
from reliamol3d.torch_model import train_reliability_net


ATOM_FEATURES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "metal"]


def load_split(zip_path: str, split: str, limit: int) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        props = json.load(zf.open(f"crossdocked/{split}_properties.json"))
        n_atoms = np.load(zf.open(f"crossdocked/{split}_n_atoms.npy"))
        offsets = np.load(zf.open(f"crossdocked/{split}_offsets.npy"))
        positions = np.load(zf.open(f"crossdocked/{split}_positions.npy"))
        atom_types = np.load(zf.open(f"crossdocked/{split}_atom_types.npy"))
        lookup = np.load(zf.open(f"crossdocked/{split}_atom_type_lookup.npy"))

    rows = []
    for idx in range(min(limit, len(props))):
        start, end = offsets[idx], offsets[idx + 1]
        coords = positions[start:end]
        atom_symbols = lookup[atom_types[start:end]]
        mask = np.array(props[idx]["starting_fragment_mask"], dtype=bool)
        if len(mask) != len(coords) or mask.all() or (~mask).sum() < 5:
            continue
        pocket = coords[mask]
        ligand = coords[~mask]
        ligand_atoms = atom_symbols[~mask]
        rows.append(
            {
                "split": split,
                "pocket_id": f"{split}_{idx:05d}",
                "target_family": props[idx]["pocket_file"].split("/")[0],
                "pocket_file": props[idx]["pocket_file"],
                "pocket": pocket.astype(np.float32),
                "ligand": ligand.astype(np.float32),
                "ligand_atoms": ligand_atoms.astype(str),
                "n_atoms": int(n_atoms[idx]),
            }
        )
    return rows


def pairwise_min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def internal_nearest_dist(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.array([99.0], dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist += np.eye(len(x)) * 99.0
    return dist.min(axis=1)


def perturb_ligand(lig: np.ndarray, pocket: np.ndarray, generator: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    lig = lig.copy()
    pocket_center = pocket.mean(axis=0)
    ligand_center = lig.mean(axis=0)
    direction = ligand_center - pocket_center
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    meta = {"shift_norm": 0.0, "scale": 1.0, "noise": 0.0}

    if generator == "NativeNear":
        noise = rng.normal(0.0, 0.18, size=lig.shape)
        lig = lig + noise
        meta.update({"noise": 0.18})
    elif generator == "DriftGen":
        shift = direction * rng.uniform(3.5, 8.5)
        lig = lig + shift + rng.normal(0.0, 0.35, size=lig.shape)
        meta.update({"shift_norm": float(np.linalg.norm(shift)), "noise": 0.35})
    elif generator == "ClashGen":
        target = pocket[rng.integers(0, len(pocket))]
        lig = lig - ligand_center + target + rng.normal(0.0, 0.25, size=lig.shape)
        meta.update({"shift_norm": float(np.linalg.norm(target - ligand_center)), "noise": 0.25})
    elif generator == "ScoreHackGen":
        scale = rng.uniform(0.45, 0.75)
        lig = (lig - ligand_center) * scale + ligand_center + rng.normal(0.0, 0.12, size=lig.shape)
        meta.update({"scale": float(scale), "noise": 0.12})
    else:
        raise ValueError(generator)
    return lig.astype(np.float32), meta


def featurize_candidate(
    pocket: np.ndarray,
    ligand: np.ndarray,
    ligand_atoms: np.ndarray,
    generator: str,
    candidate_idx: int,
    meta: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float | str | int]:
    lp_min = pairwise_min_dist(ligand, pocket)
    nn = internal_nearest_dist(ligand)
    ligand_center = ligand.mean(axis=0)
    pocket_center = pocket.mean(axis=0)
    center_dist = float(np.linalg.norm(ligand_center - pocket_center))
    lig_radius = float(np.sqrt(((ligand - ligand_center) ** 2).sum(axis=1)).mean())
    pocket_radius = float(np.sqrt(((pocket - pocket_center) ** 2).sum(axis=1)).mean())
    contact_count = int((lp_min < 4.5).sum())
    close_contact_count = int((lp_min < 3.5).sum())
    clash_count = int((lp_min < 1.2).sum())
    severe_clash_count = int((lp_min < 0.8).sum())
    internal_clash_count = int((nn < 0.65).sum())
    lp_q = np.quantile(lp_min, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    nn_q = np.quantile(nn, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    shell_0_2 = float(np.mean(lp_min < 2.0))
    shell_2_4 = float(np.mean((lp_min >= 2.0) & (lp_min < 4.0)))
    shell_4_6 = float(np.mean((lp_min >= 4.0) & (lp_min < 6.0)))
    shell_6_8 = float(np.mean((lp_min >= 6.0) & (lp_min < 8.0)))
    shell_gt8 = float(np.mean(lp_min >= 8.0))
    atom_count = len(ligand)
    hetero_frac = float(np.mean(np.isin(ligand_atoms, ["N", "O", "S", "P", "F", "Cl", "Br", "I"])))
    rare_frac = float(np.mean(~np.isin(ligand_atoms, ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"])))
    metal_frac = float(np.mean(np.isin(ligand_atoms, ["Zn", "Mg", "Fe", "Mn", "Ca", "Na", "K", "Al", "Au"])))

    # Score proxies emulate the failure mode where very close contacts look attractive to one scorer.
    vina_score = -0.18 * close_contact_count + 0.08 * center_dist + 0.45 * severe_clash_count + np.random.default_rng(candidate_idx).normal(0, 0.25)
    gnina_score = -0.14 * contact_count + 0.10 * center_dist + 0.95 * clash_count + 0.20 * internal_clash_count
    qed_proxy = np.clip(0.58 - 0.006 * abs(atom_count - 28) - 0.15 * rare_frac - 0.06 * metal_frac + 0.04 * hetero_frac + rng.normal(0, 0.06), 0.02, 0.98)
    sa_proxy = 2.35 + 0.028 * atom_count + 1.6 * rare_frac + 0.8 * metal_frac + rng.normal(0, 0.35)
    scscore_proxy = 2.65 + 0.024 * atom_count + 1.5 * rare_frac + 0.7 * metal_frac + 0.25 * hetero_frac + rng.normal(0, 0.30)
    alert_prior = {"NativeNear": 0.025, "DriftGen": 0.055, "ClashGen": 0.045, "ScoreHackGen": 0.090}[generator]
    valence_alert_proxy = int(rng.random() < alert_prior + 0.08 * rare_frac + 0.05 * metal_frac)
    charge_proxy = int(rng.choice([-3, -2, -1, 0, 1, 2, 3], p=[0.01, 0.03, 0.12, 0.62, 0.14, 0.06, 0.02]))
    route_fail_proxy = int(rng.random() < np.clip(0.04 + 0.12 * (sa_proxy > 4.0) + 0.10 * (scscore_proxy > 3.9) + 0.06 * (generator == "ScoreHackGen"), 0, 0.45))

    chemical_failure = bool(rare_frac > 0.05 or metal_frac > 0.0 or atom_count < 6 or valence_alert_proxy or abs(charge_proxy) > 2)
    geometric_failure = bool(internal_clash_count > 0 or lig_radius < 1.1 or lig_radius > 6.5)
    pocket_failure = bool(clash_count > 0 or contact_count < 3 or center_dist > pocket_radius + 4.0)
    synthetic_failure = bool(sa_proxy > 4.7 or scscore_proxy > 4.35 or atom_count > 55 or rare_frac > 0.02 or route_fail_proxy)
    scoring_failure = bool((vina_score <= np.quantile([vina_score, gnina_score], 0.10) and (gnina_score - vina_score) > 2.0) or ((vina_score < -3.0) and (geometric_failure or pocket_failure)))

    row: dict[str, float | str | int] = {
        "generator": generator,
        "candidate_idx": candidate_idx,
        "chemotype_id": int(hash((generator, atom_count, round(hetero_frac, 2), candidate_idx % 7)) % 100000),
        "atom_count": atom_count,
        "hetero_frac": hetero_frac,
        "rare_frac": rare_frac,
        "metal_frac": metal_frac,
        "center_dist": center_dist,
        "lig_radius": lig_radius,
        "pocket_radius": pocket_radius,
        "contact_count": contact_count,
        "close_contact_count": close_contact_count,
        "clash_count": clash_count,
        "severe_clash_count": severe_clash_count,
        "internal_clash_count": internal_clash_count,
        "min_lig_pocket_dist": float(lp_min.min()),
        "mean_lig_pocket_min_dist": float(lp_min.mean()),
        "min_internal_dist": float(nn.min()),
        "mean_internal_nn_dist": float(nn.mean()),
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
        "shell_0_2": shell_0_2,
        "shell_2_4": shell_2_4,
        "shell_4_6": shell_4_6,
        "shell_6_8": shell_6_8,
        "shell_gt8": shell_gt8,
        "vina_score": float(vina_score),
        "gnina_score": float(gnina_score),
        "qed": float(qed_proxy),
        "sa": float(sa_proxy),
        "scscore": float(scscore_proxy),
        "valence_alert_proxy": valence_alert_proxy,
        "charge_proxy": charge_proxy,
        "route_fail_proxy": route_fail_proxy,
        "shift_norm": float(meta["shift_norm"]),
        "scale": float(meta["scale"]),
        "noise": float(meta["noise"]),
        "chemical_failure": chemical_failure,
        "geometric_failure": geometric_failure,
        "pocket_failure": pocket_failure,
        "synthetic_failure": synthetic_failure,
        "scoring_failure": scoring_failure,
    }
    for atom in ATOM_FEATURES:
        if atom == "metal":
            row[f"atom_{atom}_frac"] = metal_frac
        else:
            row[f"atom_{atom}_frac"] = float(np.mean(ligand_atoms == atom))
    row["reliable"] = int(not any(row[c] for c in FAILURE_COLUMNS))
    return row


def build_candidate_frame(records: list[dict], generators: list[str], n_per_gen: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    candidate_idx = 0
    for rec in records:
        for generator in generators:
            for _ in range(n_per_gen):
                ligand, meta = perturb_ligand(rec["ligand"], rec["pocket"], generator, rng)
                row = featurize_candidate(rec["pocket"], ligand, rec["ligand_atoms"], generator, candidate_idx, meta, rng)
                row.update(
                    {
                        "split": rec["split"],
                        "pocket_id": rec["pocket_id"],
                        "target_family": rec["target_family"],
                        "pocket_file": rec["pocket_file"],
                    }
                )
                rows.append(row)
                candidate_idx += 1
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    score = [
        "vina_score",
        "gnina_score",
        "qed",
        "sa",
        "scscore",
    ]
    ligand_geometry = [
        "atom_count",
        "hetero_frac",
        "rare_frac",
        "metal_frac",
        "lig_radius",
        "nn_q05",
        "nn_q10",
        "nn_q25",
        "nn_q50",
        "nn_q75",
        "nn_q90",
    ] + [f"atom_{a}_frac" for a in ATOM_FEATURES]
    pocket_geometry = [
        "pocket_radius",
        "center_dist",
    ]
    interaction_shape = [
        "lp_q05",
        "lp_q10",
        "lp_q25",
        "lp_q50",
        "lp_q75",
        "lp_q90",
        "shell_0_2",
        "shell_2_4",
        "shell_4_6",
        "shell_6_8",
        "shell_gt8",
    ]
    leak_upper_bound = [
        "contact_count",
        "close_contact_count",
        "clash_count",
        "severe_clash_count",
        "internal_clash_count",
        "min_lig_pocket_dist",
        "mean_lig_pocket_min_dist",
        "min_internal_dist",
        "mean_internal_nn_dist",
        "valence_alert_proxy",
        "charge_proxy",
        "route_fail_proxy",
        "shift_norm",
        "scale",
        "noise",
    ]
    variants = {
        "score_only": score,
        "ligand_geometry": ligand_geometry,
        "interaction_shape": ligand_geometry + pocket_geometry + interaction_shape,
        "reliamol_noleak": score + ligand_geometry + pocket_geometry + interaction_shape,
        "leak_upper_bound": score + ligand_geometry + pocket_geometry + interaction_shape + leak_upper_bound,
    }
    if variant not in variants:
        raise ValueError(f"unknown feature variant: {variant}")
    return variants[variant]


def rank_to_prob(score: np.ndarray) -> np.ndarray:
    order = score.argsort().argsort().astype(float)
    return (order + 1.0) / (len(order) + 1.0)


def train_and_eval(frame: pd.DataFrame, cfg: dict, output_dir: Path) -> None:
    train = frame[frame["split"].eq("train")].copy()
    test = frame[frame["split"].eq("test")].copy()
    y_train = train["reliable"].to_numpy(np.float32)
    f_train = train[FAILURE_COLUMNS].to_numpy(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_scores: dict[str, np.ndarray] = {
        "vina_rank": rank_to_prob(-test["vina_score"].to_numpy()),
        "gnina_rank": rank_to_prob(-test["gnina_score"].to_numpy()),
        "qed_sa_vina": rank_to_prob(-test["vina_score"].to_numpy() + 0.5 * test["qed"].to_numpy() - 0.3 * test["sa"].to_numpy()),
    }

    failure_written = False
    feature_variants = ["score_only", "ligand_geometry", "interaction_shape", "reliamol_noleak", "leak_upper_bound"]
    for variant in feature_variants:
        features = feature_columns(frame, variant)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[features].to_numpy(np.float32))
        x_test = scaler.transform(test[features].to_numpy(np.float32))
        rel_prob, fail_prob = train_reliability_net(
            x_train,
            y_train,
            f_train,
            x_test,
            cfg["model"],
            device,
            cfg["experiment"]["seed"] + len(variant),
        )
        variant_scores[f"mlp_{variant}"] = rel_prob
        test[f"score_mlp_{variant}"] = rel_prob
        if variant == "reliamol_noleak":
            for i, label in enumerate(FAILURE_COLUMNS):
                test[f"{label}_prob"] = fail_prob[:, i]
            pd.DataFrame([failure_metrics(test, [f"{c}_prob" for c in FAILURE_COLUMNS])]).to_csv(output_dir / "failure_diagnosis.csv", index=False)
            failure_written = True

    lite_features = feature_columns(frame, "reliamol_noleak")
    rf = RandomForestClassifier(n_estimators=260, min_samples_leaf=4, n_jobs=16, random_state=cfg["experiment"]["seed"], class_weight="balanced")
    rf.fit(train[lite_features], train["reliable"])
    variant_scores["rf_noleak"] = rf.predict_proba(test[lite_features])[:, 1]
    hgb = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, l2_regularization=0.03, random_state=cfg["experiment"]["seed"])
    hgb.fit(train[lite_features], train["reliable"])
    variant_scores["hgb_noleak"] = hgb.predict_proba(test[lite_features])[:, 1]

    metrics = []
    for method, prob in variant_scores.items():
        row = binary_metrics(test["reliable"].to_numpy(), prob)
        row["method"] = method
        row["device"] = device
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    if not failure_written:
        pd.DataFrame().to_csv(output_dir / "failure_diagnosis.csv", index=False)

    rerank_frames = []
    rerank_columns = ["score_mlp_score_only", "score_mlp_ligand_geometry", "score_mlp_interaction_shape", "score_mlp_reliamol_noleak", "score_mlp_leak_upper_bound", "vina_score", "qed"]
    for score_col in rerank_columns:
        work = test.copy()
        use_col = score_col
        if score_col == "vina_score":
            work["vina_rank_score"] = -work["vina_score"]
            use_col = "vina_rank_score"
        rerank_frames.append(reranking_metrics(work, use_col, cfg["evaluation"]["topk_fracs"]))
    pd.concat(rerank_frames, ignore_index=True).to_csv(output_dir / "reranking.csv", index=False)
    test.drop(columns=[c for c in test.columns if c.startswith("gen_")], errors="ignore").to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_smoke_crossdocked.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train = load_split(data_cfg["zip_path"], "train", data_cfg["train_pockets"])
    val = load_split(data_cfg["zip_path"], "val", data_cfg["val_pockets"])
    test = load_split(data_cfg["zip_path"], "test", data_cfg["test_pockets"])
    frame = build_candidate_frame(
        train + val + test,
        data_cfg["generators"],
        data_cfg["candidates_per_generator"],
        cfg["experiment"]["seed"],
    )
    frame.to_csv(output_dir / "candidate_labels.csv", index=False)
    label_summary = frame.groupby(["split", "generator"])[["reliable"] + FAILURE_COLUMNS].mean().reset_index()
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    train_and_eval(frame, cfg, output_dir)
    print(label_summary.to_string(index=False))
    print(pd.read_csv(output_dir / "metrics.csv").to_string(index=False))
    print(pd.read_csv(output_dir / "reranking.csv").to_string(index=False))


if __name__ == "__main__":
    main()
