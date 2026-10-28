"""Whole-video and segment-level VGCap temporal quality model."""

import torch
import torch.nn as nn

from ._components import SegmentTemporalPool


class ScoreHead(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(value)) * 100.0

class TemporalQualityModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        max_segments=4,
        num_heads=8,
        layers=2,
        dropout=0.1,
        global_gate_init=0.33,
        segment_gate_init=0.45,
        fusion_gate_init=0.65,
    ):
        super().__init__()
        gate_values = (
            ("global_gate_init", global_gate_init),
            ("segment_gate_init", segment_gate_init),
            ("fusion_gate_init", fusion_gate_init),
        )
        for name, value in gate_values:
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.config = {
            "hidden_dim": hidden_dim,
            "max_segments": max_segments,
            "num_heads": num_heads,
            "layers": layers,
            "dropout": dropout,
            "global_gate_init": float(global_gate_init),
            "segment_gate_init": float(segment_gate_init),
            "fusion_gate_init": float(fusion_gate_init),
        }
        self.whole_motion_projection = nn.Sequential(
            nn.LayerNorm(2304), nn.Linear(2304, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.global_caption_projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.global_agbg_logit = nn.Parameter(torch.logit(torch.tensor(global_gate_init)))
        self.segment_motion_projection = nn.Sequential(
            nn.LayerNorm(2304), nn.Linear(2304, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.segment_caption_projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.segment_agbg_logit = nn.Parameter(torch.logit(torch.tensor(segment_gate_init)))
        self.position = nn.Parameter(torch.randn(1, max_segments, hidden_dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 2, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            block, layers, norm=nn.LayerNorm(hidden_dim), enable_nested_tensor=False
        )
        self.temporal_pool = SegmentTemporalPool(hidden_dim, dropout)
        self.segment_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.final_agbg_logit = nn.Parameter(torch.logit(torch.tensor(fusion_gate_init)))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.score_head = ScoreHead(hidden_dim, dropout)

    def forward(
        self, whole, segment_motion, segment_caption, segment_mask, global_caption
    ):
        whole_feature = self.whole_motion_projection(whole)
        alpha = torch.sigmoid(self.global_agbg_logit)
        whole_fused = (
            (1.0 - alpha) * whole_feature
            + alpha * self.global_caption_projection(global_caption)
        )
        beta = torch.sigmoid(self.segment_agbg_logit)
        segments = (
            (1.0 - beta) * self.segment_motion_projection(segment_motion)
            + beta * self.segment_caption_projection(segment_caption)
        )
        segments = self.temporal_transformer(
            segments + self.position[:, : segments.shape[1]],
            src_key_padding_mask=~segment_mask,
        )
        segments = segments * segment_mask.unsqueeze(-1)
        pooled, _, _ = self.temporal_pool(segments, segment_mask)
        segment_feature = self.segment_projection(pooled)
        lambda_t = torch.sigmoid(self.final_agbg_logit)
        fused = self.output_norm((1.0 - lambda_t) * whole_fused + lambda_t * segment_feature)
        return self.score_head(fused)
