"""Alignment wrapper with a partially fine-tuned SigLIP2 vision encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval, SiglipVisionModel


from ._fusion import AlignmentBranchConfig, AlignmentFusionModel


class AlignmentQualityModel(nn.Module):
    def __init__(
        self,
        alignment_config: AlignmentBranchConfig,
        vision_model: str = "google/siglip2-base-patch16-224",
        blip_model: str = "Salesforce/blip-itm-base-coco",
        vision_frame_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.alignment_config = alignment_config
        self.vision_model_name = vision_model
        self.blip_model_name = blip_model
        self.vision_frame_batch_size = vision_frame_batch_size
        self.siglip2_vision = SiglipVisionModel.from_pretrained(vision_model)
        self.blip = BlipForImageTextRetrieval.from_pretrained(blip_model)
        self.alignment = AlignmentFusionModel(alignment_config)
        self.set_trainable_encoder_layers(0, 0)

    @property
    def vision_backbone(self):
        # Transformers <=4.x wrapped these modules in ``vision_model``;
        # Transformers 5.x exposes them directly on SiglipVisionModel.
        return getattr(self.siglip2_vision, "vision_model", self.siglip2_vision)

    @property
    def vision_layers(self):
        return self.vision_backbone.encoder.layers

    @property
    def blip_text_layers(self):
        return self.blip.text_encoder.encoder.layer

    def set_trainable_encoder_layers(self, vision_blocks: int, blip_layers: int) -> None:
        vision_stack = self.vision_layers
        if not 0 <= vision_blocks <= len(vision_stack):
            raise ValueError("invalid number of SigLIP2 vision blocks")
        for parameter in self.siglip2_vision.parameters():
            parameter.requires_grad = False
        self.siglip2_vision.eval()
        if vision_blocks:
            for layer in vision_stack[-vision_blocks:]:
                layer.train()
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for module_name in ("post_layernorm", "head"):
                module = getattr(self.vision_backbone, module_name, None)
                if module is not None:
                    module.train()
                    for parameter in module.parameters():
                        parameter.requires_grad = True

        text_stack = self.blip_text_layers
        if not 0 <= blip_layers <= len(text_stack):
            raise ValueError("invalid number of BLIP text layers")
        for parameter in self.blip.parameters():
            parameter.requires_grad = False
        self.blip.eval()
        if blip_layers:
            for layer in text_stack[-blip_layers:]:
                layer.train()
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def trainable_counts(self) -> dict[str, int]:
        return {
            "siglip2": sum(p.numel() for p in self.siglip2_vision.parameters() if p.requires_grad),
            "blip": sum(p.numel() for p in self.blip.text_encoder.parameters() if p.requires_grad),
            "alignment": sum(p.numel() for p in self.alignment.parameters() if p.requires_grad),
        }

    def encode_frames(self, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, frames = pixels.shape[:2]
        flat_mask = mask.reshape(-1).bool()
        valid_pixels = pixels.reshape(-1, *pixels.shape[2:])[flat_mask]
        values = []
        has_grad = self.training and any(
            parameter.requires_grad for parameter in self.siglip2_vision.parameters()
        )
        with torch.set_grad_enabled(has_grad):
            for start in range(0, len(valid_pixels), self.vision_frame_batch_size):
                output = self.siglip2_vision(
                    pixel_values=valid_pixels[start:start + self.vision_frame_batch_size],
                    return_dict=True,
                )
                if output.pooler_output is None:
                    raise RuntimeError("SigLIP2 vision model did not return pooler_output")
                values.append(output.pooler_output)
        valid = F.normalize(torch.cat(values).float(), dim=-1)
        flat = valid.new_zeros(batch * frames, valid.shape[-1])
        flat = flat.index_copy(0, flat_mask.nonzero(as_tuple=False).flatten(), valid)
        return flat.view(batch, frames, -1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        has_grad = self.training and any(
            parameter.requires_grad for parameter in self.blip.text_encoder.parameters()
        )
        with torch.set_grad_enabled(has_grad):
            hidden = self.blip.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state[:, 0]
        return F.normalize(hidden.float(), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        visual_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        global_input_ids: torch.Tensor,
        global_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_frames(pixel_values, visual_mask)
        if self.alignment_config.fusion_mode == "visual_only":
            # A strict visual-only ablation: neither text branch is executed.
            prompt = visual.new_zeros(visual.shape[0], self.alignment_config.text_dim)
            global_caption = torch.zeros_like(prompt)
        else:
            prompt = self.encode_text(prompt_input_ids, prompt_attention_mask)
            if self.alignment_config.fusion_mode == "visual_prompt":
                # The visual-plus-prompt ablation does not consume global captions.
                global_caption = torch.zeros_like(prompt)
            else:
                global_caption = self.encode_text(global_input_ids, global_attention_mask)
        return self.alignment(visual, visual_mask, prompt, global_caption)


def trainable_parameter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def restore_parameter_state(module: nn.Module, values: dict[str, torch.Tensor]) -> None:
    parameters = dict(module.named_parameters())
    missing = sorted(set(values) - set(parameters))
    if missing:
        raise ValueError(f"checkpoint parameters missing in model: {missing[:3]}")
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].copy_(value)

