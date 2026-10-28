"""Per-branch VQA metrics with stable handling of degenerate inputs."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


SCORE_NAMES = ("Spatial_MOS", "Temporal_MOS", "Alignment_MOS")


def _correlation(function, target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")
    result = function(target, prediction)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError(f"Expected matching [N,3] arrays, got {predictions.shape} and {targets.shape}")
    output: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(SCORE_NAMES):
        prediction = predictions[:, index]
        target = targets[:, index]
        output[name] = {
            "SRCC": _correlation(spearmanr, target, prediction),
            "PLCC": _correlation(pearsonr, target, prediction),
            "KRCC": _correlation(kendalltau, target, prediction),
            "RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "MAE": float(np.mean(np.abs(prediction - target))),
        }
    valid_srcc = [values["SRCC"] for values in output.values() if np.isfinite(values["SRCC"])]
    output["mean"] = {"SRCC": float(np.mean(valid_srcc)) if valid_srcc else float("nan")}
    return output


def compute_spatial_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    return {
        "SRCC": _correlation(spearmanr, target, prediction),
        "PLCC": _correlation(pearsonr, target, prediction),
        "KRCC": _correlation(kendalltau, target, prediction),
        "RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "MAE": float(np.mean(np.abs(prediction - target))),
    }
