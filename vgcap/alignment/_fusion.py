"""Three-input AGBG used by the VGCap alignment branch."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
import torch.nn as nn

@dataclass
class AlignmentBranchConfig:
    visual_dim: int = 768
    text_dim: int = 768
    hidden_dim: int = 256
    max_frames: int = 32
    num_heads: int = 8
    visual_layers: int = 2
    dropout: float = 0.1
    gate_init: tuple[float, float, float] = (0.55, 0.25, 0.20)
    fusion_mode: str = "agbg"

    def __post_init__(self) -> None:
        self.gate_init = tuple(float(value) for value in self.gate_init)
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if len(self.gate_init) != 3 or any(value <= 0 for value in self.gate_init):
            raise ValueError("gate_init must contain three positive V/P/C weights")
        if abs(sum(self.gate_init) - 1.0) > 1e-6:
            raise ValueError("gate_init weights must sum to 1")
        if self.fusion_mode not in {"agbg", "visual_prompt", "visual_only"}:
            raise ValueError("fusion_mode must be agbg, visual_prompt, or visual_only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AlignmentBranchConfig":
        return cls(**values)

class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        logits = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.to(sequence.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1), weights

class AlignmentFusionModel(nn.Module):
    """Aggregate frame features and apply the three-input AGBG."""
    def __init__(self, config: AlignmentBranchConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.visual_projection = nn.Sequential(nn.LayerNorm(config.visual_dim), nn.Linear(config.visual_dim, dim))
        self.position = nn.Parameter(torch.randn(1, config.max_frames, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, config.num_heads, dim * 4, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, config.visual_layers, enable_nested_tensor=False)
        self.visual_norm = nn.LayerNorm(dim)
        self.attention_pool = MaskedAttentionPool(dim)
        self.visual_pool = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.prompt_projection = self._text_projection(config.text_dim, dim)
        self.global_projection = self._text_projection(config.text_dim, dim)
        self.gate_logits = nn.Parameter(torch.log(torch.tensor(config.gate_init, dtype=torch.float32)))
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim // 2, 1))
        nn.init.normal_(self.score_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _text_projection(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def gate_weights(self) -> torch.Tensor:
        if self.config.fusion_mode == "visual_only":
            return self.gate_logits.new_tensor([1.0, 0.0, 0.0])
        if self.config.fusion_mode == "visual_prompt":
            weights = torch.softmax(self.gate_logits[:2], dim=0)
            return torch.cat([weights, weights.new_zeros(1)])
        return torch.softmax(self.gate_logits, dim=0)

    def forward(self, visual_features, visual_mask, prompt_features, global_caption_features):
        hidden = self.visual_projection(visual_features) + self.position[:, :visual_features.shape[1]]
        hidden = self.visual_norm(self.transformer(hidden, src_key_padding_mask=~visual_mask))
        mask_values = visual_mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * mask_values).sum(dim=1) / mask_values.sum(dim=1).clamp_min(1.0)
        attended, frame_weights = self.attention_pool(hidden, visual_mask)
        deviation = (torch.abs(hidden - mean.unsqueeze(1)) * frame_weights.unsqueeze(-1)).sum(dim=1)
        visual = self.visual_pool(torch.cat([mean, attended, deviation], dim=-1))
        prompt = self.prompt_projection(prompt_features)
        global_caption = self.global_projection(global_caption_features)
        weights = self.gate_weights()
        fused = weights[0] * visual + weights[1] * prompt + weights[2] * global_caption
        return {"score": torch.sigmoid(self.score_head(fused)) * 100.0, "fused_features": fused, "gate_weights": weights, "frame_weights": frame_weights}
