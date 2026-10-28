"""Regression and within-batch ordering objectives."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ._metrics import compute_spatial_metrics

__all__ = ["SpatialLoss", "ThreeBranchLoss", "compute_spatial_metrics"]


class ThreeBranchLoss:
    def __init__(self, rank_weight: float = 0.3, rank_margin: float = 0.05) -> None:
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def _ranking_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.numel() < 2:
            return prediction.new_zeros(())
        pred_diff = prediction[:, None] - prediction[None, :]
        target_diff = target[:, None] - target[None, :]
        upper = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        valid = upper & (target_diff.abs() > 1e-3)
        if not valid.any():
            return prediction.new_zeros(())
        signed_distance = target_diff.sign() * pred_diff
        return F.relu(self.rank_margin - signed_distance)[valid].mean()

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.shape != labels.shape or scores.ndim != 2 or scores.shape[1] != 3:
            raise ValueError(f"Expected matching [B,3] scores and labels, got {scores.shape} and {labels.shape}")
        prediction = scores / 100.0
        target = labels / 100.0
        regression_by_branch = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=0)
        ranking_by_branch = torch.stack(
            [self._ranking_loss(prediction[:, branch], target[:, branch]) for branch in range(3)]
        )
        total = regression_by_branch.sum() + self.rank_weight * ranking_by_branch.sum()
        names = ("spatial", "temporal", "alignment")
        details: Dict[str, float] = {"total": float(total.detach())}
        for index, name in enumerate(names):
            details[f"{name}_regression"] = float(regression_by_branch[index].detach())
            details[f"{name}_ranking"] = float(ranking_by_branch[index].detach())
        return total, details


class SpatialLoss:
    """Smooth-L1 plus pairwise ranking for one Spatial MOS output."""

    def __init__(self, rank_weight: float = 0.3, rank_margin: float = 0.05) -> None:
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        prediction = scores.reshape(-1) / 100.0
        target = labels.reshape(-1) / 100.0
        if prediction.shape != target.shape:
            raise ValueError(f"Score/label shape mismatch: {prediction.shape} vs {target.shape}")
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
        total = regression + self.rank_weight * ranking
        return total, {
            "total": float(total.detach()),
            "regression": float(regression.detach()),
            "ranking": float(ranking.detach()),
        }
