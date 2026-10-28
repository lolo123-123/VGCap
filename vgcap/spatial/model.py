"""Spatial quality branch used by VGCap."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn

@dataclass
class SpatialBranchConfig:
    spatial_dim: int = 512
    text_dim: int = 768
    hidden_dim: int = 256
    max_frames: int = 32
    num_heads: int = 8
    spatial_layers: int = 2
    dropout: float = 0.1
    use_global_caption: bool = True
    agbg_gate_init: float = 0.33

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 < self.agbg_gate_init < 1.0:
            raise ValueError("agbg_gate_init must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)

class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        logits = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.to(sequence.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1), weights

class SpatialQualityBranch(nn.Module):
    """Aggregate frame features and fuse the global-caption feature with AGBG."""
    def __init__(self, config: SpatialBranchConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.frame_projection = nn.Sequential(nn.LayerNorm(config.spatial_dim), nn.Linear(config.spatial_dim, dim))
        self.position = nn.Parameter(torch.randn(1, config.max_frames, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, config.num_heads, dim * 4, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, config.spatial_layers, enable_nested_tensor=False)
        self.frame_norm = nn.LayerNorm(dim)
        self.attention_pool = MaskedAttentionPool(dim)
        self.spatial_projection = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.caption_projection = nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, dim), nn.GELU(), nn.LayerNorm(dim)) if config.use_global_caption else None
        self.agbg_gate_logit = nn.Parameter(torch.logit(torch.tensor(config.agbg_gate_init))) if config.use_global_caption else None
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim // 2, 1))
        nn.init.normal_(self.score_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, spatial_features, spatial_mask, global_caption_features=None, **_):
        hidden = self.frame_projection(spatial_features) + self.position[:, :spatial_features.shape[1]]
        hidden = self.frame_norm(self.transformer(hidden, src_key_padding_mask=~spatial_mask))
        mask_values = spatial_mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * mask_values).sum(dim=1) / mask_values.sum(dim=1).clamp_min(1.0)
        attended, frame_weights = self.attention_pool(hidden, spatial_mask)
        deviation = (torch.abs(hidden - mean.unsqueeze(1)) * frame_weights.unsqueeze(-1)).sum(dim=1)
        visual = self.spatial_projection(torch.cat([mean, attended, deviation], dim=-1))
        gate = None
        caption = None
        fused = visual
        if self.config.use_global_caption:
            if global_caption_features is None:
                raise ValueError("global_caption_features are required")
            caption = self.caption_projection(global_caption_features)
            gate = torch.sigmoid(self.agbg_gate_logit)
            fused = (1.0 - gate) * visual + gate * caption
        return {"score": torch.sigmoid(self.score_head(fused)) * 100.0,
                "spatial_features": visual, "caption_features": caption,
                "fused_features": fused, "agbg_gate": gate, "frame_weights": frame_weights}
