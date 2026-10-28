"""Alignment loss, metrics, and atomic output helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr


class AlignmentLoss:
    def __init__(self, rank_weight: float = 0.3, rank_margin: float = 0.05) -> None:
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prediction = scores.reshape(-1) / 100.0
        target = labels.reshape(-1) / 100.0
        regression = F.smooth_l1_loss(prediction, target)
        if len(prediction) < 2:
            ranking = prediction.new_zeros(())
        else:
            pred_diff = prediction[:, None] - prediction[None, :]
            target_diff = target[:, None] - target[None, :]
            valid = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
            valid &= target_diff.abs() > 1e-3
            ranking = (
                F.relu(self.rank_margin - target_diff.sign() * pred_diff)[valid].mean()
                if valid.any() else prediction.new_zeros(())
            )
        return regression + self.rank_weight * ranking


def _correlation(function, target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")
    result = function(target, prediction)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {
        "SRCC": _correlation(spearmanr, target, prediction),
        "PLCC": _correlation(pearsonr, target, prediction),
        "KRCC": _correlation(kendalltau, target, prediction),
        "RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "MAE": float(np.mean(np.abs(prediction - target))),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

