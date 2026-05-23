from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import FAILURE_COLUMNS


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_proxy_dataset(
    n_pockets: int,
    candidates_per_pocket: int,
    generators: list[str],
    feature_dim: int,
    target_family_count: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pockets * candidates_per_pocket * len(generators)
    pocket_ids = np.repeat(np.arange(n_pockets), candidates_per_pocket * len(generators))
    target_family = rng.integers(0, target_family_count, size=n_pockets)
    family = target_family[pocket_ids]
    generator = np.tile(np.repeat(generators, candidates_per_pocket), n_pockets)

    pocket_latent = rng.normal(size=(n_pockets, 8))
    family_latent = rng.normal(size=(target_family_count, 6))
    gen_bias = {
        "Pocket2Mol": np.array([0.20, 0.05, -0.05, 0.12, 0.04]),
        "TargetDiff": np.array([0.02, 0.18, 0.12, -0.02, 0.08]),
        "DiffSBDD": np.array([-0.06, 0.08, 0.16, 0.03, 0.18]),
        "DecompDiff": np.array([0.10, -0.04, 0.02, 0.22, 0.06]),
    }

    base = rng.normal(size=(n, feature_dim))
    pocket_features = pocket_latent[pocket_ids]
    family_features = family_latent[family]
    chemotype = rng.integers(0, 900, size=n)

    vina_score = -7.2 + 1.25 * base[:, 0] - 0.55 * base[:, 1] + rng.normal(0, 0.8, size=n)
    gnina_score = -6.8 + 0.9 * base[:, 0] - 0.15 * base[:, 2] + rng.normal(0, 0.9, size=n)
    qed = sigmoid(0.8 * base[:, 3] - 0.4 * base[:, 4] + rng.normal(0, 0.4, size=n))
    sa = 3.0 + 1.1 * sigmoid(base[:, 5]) + 0.7 * (generator == "DecompDiff") + rng.normal(0, 0.35, size=n)
    scscore = 3.2 + 1.0 * sigmoid(base[:, 6]) + 0.5 * (generator == "Pocket2Mol") + rng.normal(0, 0.4, size=n)
    clash_count = np.maximum(0, rng.poisson(sigmoid(base[:, 7] + pocket_features[:, 0]) * 4.5)).astype(float)
    contact_count = np.maximum(0, rng.poisson(sigmoid(-base[:, 8] + pocket_features[:, 1]) * 7.0)).astype(float)
    redock_rmsd = np.maximum(0, 1.2 + 1.0 * sigmoid(base[:, 9]) + 0.35 * clash_count + rng.normal(0, 0.45, size=n))
    mmff_rmsd = np.maximum(0, 0.7 + 0.8 * sigmoid(base[:, 10]) + rng.normal(0, 0.3, size=n))

    logits = np.zeros((n, len(FAILURE_COLUMNS)))
    gen_matrix = np.vstack([gen_bias[g] for g in generator])
    logits[:, 0] = -1.8 + 0.9 * base[:, 11] + 0.5 * np.abs(base[:, 12]) + gen_matrix[:, 0]
    logits[:, 1] = -1.4 + 0.45 * clash_count + 0.7 * mmff_rmsd + gen_matrix[:, 1]
    logits[:, 2] = -1.5 + 0.65 * clash_count - 0.25 * contact_count + 0.55 * redock_rmsd + gen_matrix[:, 2]
    logits[:, 3] = -1.7 + 0.8 * (sa - 3.5) + 0.5 * (scscore - 3.8) + gen_matrix[:, 3]
    logits[:, 4] = -1.6 + 0.55 * (vina_score < np.quantile(vina_score, 0.20)) + 0.9 * np.abs(vina_score - gnina_score) + gen_matrix[:, 4]
    logits += 0.15 * family_features[:, :5] + rng.normal(0, 0.15, size=logits.shape)

    probs = sigmoid(logits)
    failures = rng.binomial(1, probs).astype(bool)
    reliable = ~failures.any(axis=1)

    frame = pd.DataFrame(
        {
            "pocket_id": pocket_ids,
            "target_family": family,
            "generator": generator,
            "chemotype_id": chemotype,
            "vina_score": vina_score,
            "gnina_score": gnina_score,
            "qed": qed,
            "sa": sa,
            "scscore": scscore,
            "clash_count": clash_count,
            "contact_count": contact_count,
            "redock_rmsd": redock_rmsd,
            "mmff_rmsd": mmff_rmsd,
            "reliable": reliable.astype(int),
        }
    )
    for i, col in enumerate(FAILURE_COLUMNS):
        frame[col] = failures[:, i]
    for j in range(feature_dim):
        frame[f"x{j:02d}"] = base[:, j]
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    handcrafted = [
        "vina_score",
        "gnina_score",
        "qed",
        "sa",
        "scscore",
        "clash_count",
        "contact_count",
        "redock_rmsd",
        "mmff_rmsd",
    ]
    latent = [c for c in frame.columns if c.startswith("x")]
    generator_dummies = pd.get_dummies(frame["generator"], prefix="gen")
    for col in generator_dummies.columns:
        frame[col] = generator_dummies[col].astype(float)
    return handcrafted + latent + list(generator_dummies.columns)

