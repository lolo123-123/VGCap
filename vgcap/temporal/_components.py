"""Shared components for the VGCap temporal branch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr

def feature_path(root: str | Path, video_name: str) -> Path:
    relative = Path(str(video_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe video_name: {video_name}")
    return (Path(root) / relative).with_suffix(".pt")

def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, sequence, mask):
        logits = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.to(sequence.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1), weights

class SegmentTemporalPool(nn.Module):
    """Masked mean, attention, and adjacent-segment difference pooling."""
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = MaskedAttentionPool(dim)
        self.output = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, sequence, mask):
        values = mask.to(sequence.dtype).unsqueeze(-1)
        mean = (sequence * values).sum(dim=1) / values.sum(dim=1).clamp_min(1.0)
        attended, weights = self.attention(sequence, mask)
        pair_mask = mask[:, 1:] & mask[:, :-1]
        pair_values = pair_mask.to(sequence.dtype).unsqueeze(-1)
        adjacent = torch.abs(sequence[:, 1:] - sequence[:, :-1])
        difference = (adjacent * pair_values).sum(dim=1) / pair_values.sum(dim=1).clamp_min(1.0)
        return self.output(torch.cat([mean, attended, difference], dim=-1)), weights, difference

class TemporalLoss:
    def __init__(self, rank_weight: float = 0.3, rank_margin: float = 0.05) -> None:
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def __call__(self, scores, labels):
        prediction = scores.reshape(-1) / 100.0
        target = labels.reshape(-1) / 100.0
        regression = F.smooth_l1_loss(prediction, target)
        if len(prediction) < 2:
            ranking = prediction.new_zeros(())
        else:
            prediction_difference = prediction[:, None] - prediction[None, :]
            target_difference = target[:, None] - target[None, :]
            valid = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
            valid &= target_difference.abs() > 1e-3
            ranking = F.relu(self.rank_margin - target_difference.sign() * prediction_difference)[valid].mean() if valid.any() else prediction.new_zeros(())
        return regression + self.rank_weight * ranking

def _correlation(function, target, prediction):
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")
    result = function(target, prediction)
    return float(result.statistic if hasattr(result, "statistic") else result[0])

def compute_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {"SRCC": _correlation(spearmanr, target, prediction),
            "PLCC": _correlation(pearsonr, target, prediction),
            "KRCC": _correlation(kendalltau, target, prediction),
            "RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "MAE": float(np.mean(np.abs(prediction - target)))}
