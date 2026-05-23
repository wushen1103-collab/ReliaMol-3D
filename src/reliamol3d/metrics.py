from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


FAILURE_COLUMNS = [
    "chemical_failure",
    "geometric_failure",
    "pocket_failure",
    "synthetic_failure",
    "scoring_failure",
]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1))
    if best >= len(thresholds):
        return 0.5, float(f1[best])
    return float(thresholds[best]), float(f1[best])


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    threshold, best_f1 = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        f"{prefix}roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
        f"{prefix}pr_auc": safe_auc(average_precision_score, y_true, y_prob),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}best_f1": best_f1,
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}ece": expected_calibration_error(y_true, y_prob),
        f"{prefix}threshold": threshold,
        f"{prefix}positive_rate": float(y_true.mean()),
    }
    return out


def safe_auc(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_prob))
    except ValueError:
        return float("nan")


def failure_metrics(frame: pd.DataFrame, prob_columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, prob in zip(FAILURE_COLUMNS, prob_columns):
        out.update(binary_metrics(frame[label].to_numpy(), frame[prob].to_numpy(), prefix=f"{label}."))
    return out


def reranking_metrics(
    frame: pd.DataFrame,
    score_col: str,
    topk_fracs: list[float],
    group_col: str = "pocket_id",
) -> pd.DataFrame:
    rows = []
    for frac in topk_fracs:
        selected = []
        for _, group in frame.groupby(group_col, sort=False):
            k = max(1, int(np.ceil(len(group) * frac)))
            selected.append(group.nlargest(k, score_col))
        top = pd.concat(selected, ignore_index=True)
        rows.append(
            {
                "score": score_col,
                "topk_frac": frac,
                "n_selected": len(top),
                "reliable_rate": float(top["reliable"].mean()),
                "posebusters_pass_proxy": float((~top["geometric_failure"]).mean()),
                "sa_pass_proxy": float((~top["synthetic_failure"]).mean()),
                "clash_free_proxy": float((~top["pocket_failure"]).mean()),
                "vina_gnina_consistency_proxy": float((~top["scoring_failure"]).mean()),
                "diversity_proxy": float(top.groupby(group_col)["chemotype_id"].nunique().mean()),
                "mean_affinity_proxy": float(top["vina_score"].mean()),
            }
        )
    return pd.DataFrame(rows)

