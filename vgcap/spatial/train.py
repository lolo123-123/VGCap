#!/usr/bin/env python3
"""Train the VGCap spatial branch with SigLIP2 and BLIP encoders."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BlipForImageTextRetrieval,
    SiglipVisionModel,
)

from ..common.captions import canonical_text
from ..common.video import read_video_images
from .data import SpatialFeatureDataset, collate_spatial
from .model import SpatialQualityBranch, SpatialBranchConfig
from .training_utils import SpatialLoss, compute_spatial_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--feature-dir", help="Required only when --vision-mode frozen")
    parser.add_argument("--video-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--caption-source",
        choices=("prompt", "global"),
        default="global",
        help="Text input: the source prompt from Prompt or the global caption from Global_Caption",
    )
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--vision-mode", choices=("frozen", "partial"), default="partial")
    parser.add_argument("--vision-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--blip-model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-last-vision-blocks", type=int, default=2)
    parser.add_argument("--unfreeze-last-blip-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-frame-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        default=None,
        help="Learning rate for AGBG; defaults to --learning-rate",
    )
    parser.add_argument("--vision-learning-rate", type=float, default=2e-6)
    parser.add_argument("--blip-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--video-backend", choices=("auto", "decord", "opencv"), default="opencv")
    parser.add_argument("--agbg-gate-init", type=float, default=0.33)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class OnlineCaptionSpatialDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_dir: str | None,
        caption_source: str,
        visual_only: bool,
        vision_mode: str,
        video_dir: str | None,
        vision_model: str,
        max_frames: int,
        limit: int | None,
        video_backend: str,
    ) -> None:
        self.table = pd.read_csv(csv_path)
        required_columns = {"video_name", "Spatial_MOS"}
        missing = sorted(required_columns - set(self.table.columns))
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.table = self.table.head(limit).reset_index(drop=True)
        self.base = None
        if vision_mode == "frozen":
            if not feature_dir:
                raise ValueError("--feature-dir is required when --vision-mode frozen")
            self.base = SpatialFeatureDataset(csv_path, feature_dir, require_labels=True)
            self.base.table = self.table
        self.caption_source = caption_source
        self.visual_only = visual_only
        self.vision_mode = vision_mode
        self.max_frames = max_frames
        self.video_backend = video_backend
        self.video_dir = Path(video_dir) if video_dir else None
        self.processor = None
        if vision_mode == "partial":
            if self.video_dir is None:
                raise ValueError("--video-dir is required for partial SigLIP2 fine-tuning")
            self.processor = AutoImageProcessor.from_pretrained(vision_model)
        if not visual_only:
            required = "Prompt" if caption_source == "prompt" else "Global_Caption"
            if required not in self.table.columns:
                raise ValueError(f"{csv_path} is missing {required}")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        video_name = canonical_text(row["video_name"])
        label = torch.tensor(float(row["Spatial_MOS"]), dtype=torch.float32)
        if not video_name or not torch.isfinite(label) or not 0 <= float(label) <= 100:
            raise ValueError(f"Invalid spatial sample at row {index + 2}")
        sample = self.base[index] if self.base is not None else {
            "video_name": video_name,
            "label": label,
        }
        if self.visual_only:
            sample["caption_text"] = ""
        else:
            column = "Prompt" if self.caption_source == "prompt" else "Global_Caption"
            caption = str(self.table.iloc[index][column]).strip()
            if not caption or caption.lower() == "nan":
                raise ValueError(f"Empty {column} at row {index + 2}")
            sample["caption_text"] = caption
        if self.vision_mode == "partial":
            video_path = self.video_dir / Path(sample["video_name"].replace("\\", "/"))
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video: {video_path}")
            images, indices = read_video_images(video_path, self.max_frames, self.video_backend)
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
            sample["spatial_features"] = torch.zeros(len(pixels), 512, dtype=torch.float32)
            sample["spatial_mask"] = torch.ones(len(pixels), dtype=torch.bool)
            sample["pixel_values"] = pixels.float()
            sample["sampled_frame_indices"] = torch.from_numpy(indices)
        return sample


def collate_online(samples):
    batch = collate_spatial(samples)
    batch["caption_texts"] = [sample["caption_text"] for sample in samples]
    if "pixel_values" in samples[0]:
        max_frames = max(sample["pixel_values"].shape[0] for sample in samples)
        channels, height, width = samples[0]["pixel_values"].shape[1:]
        pixels = torch.zeros(len(samples), max_frames, channels, height, width)
        for index, sample in enumerate(samples):
            count = sample["pixel_values"].shape[0]
            pixels[index, :count] = sample["pixel_values"]
        batch["pixel_values"] = pixels
    return batch


def create_loader(args, csv_path, shuffle, limit):
    dataset = OnlineCaptionSpatialDataset(
        csv_path,
        args.feature_dir,
        args.caption_source,
        args.visual_only,
        args.vision_mode,
        args.video_dir,
        args.vision_model,
        args.max_frames,
        limit,
        args.video_backend,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_online,
    )


class SpatialQualityModel(nn.Module):
    def __init__(
        self, config, vision_mode, vision_model, blip_model, visual_only,
        vision_frame_batch_size,
    ):
        super().__init__()
        self.branch_config = config
        self.vision_mode = vision_mode
        self.vision_model_name = vision_model
        self.blip_model_name = blip_model
        self.visual_only = visual_only
        self.vision_frame_batch_size = vision_frame_batch_size
        self.vision_encoder = None
        self.vision_projection = None
        if vision_mode == "partial":
            self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model)
            hidden_size = int(self.vision_encoder.config.hidden_size)
            self.vision_projection = nn.Linear(hidden_size, 512, bias=False)
        self.blip_text = None
        if not visual_only:
            full_blip = BlipForImageTextRetrieval.from_pretrained(blip_model)
            self.blip_text = full_blip.text_encoder
            del full_blip
            gc.collect()
        self.spatial_branch = SpatialQualityBranch(config)
        self.set_vision_trainable_blocks(0)
        self.set_blip_trainable_layers(0)

    def _blip_layers(self):
        if self.blip_text is None:
            return []
        encoder = getattr(self.blip_text, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise AttributeError("Unable to locate text encoder layers")
        return layers

    def set_vision_trainable_blocks(self, last_n_blocks):
        if self.vision_encoder is None:
            return
        vision_root = getattr(self.vision_encoder, "vision_model", self.vision_encoder)
        layers = vision_root.encoder.layers
        if not 0 <= last_n_blocks <= len(layers):
            raise ValueError("Invalid number of SigLIP2 vision blocks")
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        if last_n_blocks:
            for layer in layers[-last_n_blocks:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in vision_root.post_layernorm.parameters():
                parameter.requires_grad = True
            for parameter in self.vision_projection.parameters():
                parameter.requires_grad = True

    def set_blip_trainable_layers(self, last_n_layers):
        if self.blip_text is None:
            return
        layers = self._blip_layers()
        if not 0 <= last_n_layers <= len(layers):
            raise ValueError("Invalid number of BLIP layers")
        for parameter in self.blip_text.parameters():
            parameter.requires_grad = False
        if last_n_layers:
            for layer in layers[-last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    @property
    def trainable_vision_parameters(self):
        if self.vision_encoder is None:
            return 0
        count = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        if self.vision_projection is not None:
            count += sum(p.numel() for p in self.vision_projection.parameters() if p.requires_grad)
        return count

    @property
    def trainable_blip_parameters(self):
        if self.blip_text is None:
            return 0
        return sum(p.numel() for p in self.blip_text.parameters() if p.requires_grad)

    def encode_frames(self, pixel_values, spatial_mask):
        batch, frames = pixel_values.shape[:2]
        flat_mask = spatial_mask.reshape(-1).bool()
        valid_pixels = pixel_values.reshape(-1, *pixel_values.shape[2:])[flat_mask]
        encoded = []
        for start in range(0, len(valid_pixels), self.vision_frame_batch_size):
            output = self.vision_encoder(
                pixel_values=valid_pixels[start : start + self.vision_frame_batch_size],
                return_dict=True,
            )
            image_embeds = getattr(output, "pooler_output", None)
            if image_embeds is None:
                image_embeds = output.last_hidden_state.mean(dim=1)
            image_embeds = self.vision_projection(image_embeds)
            encoded.append(F.normalize(image_embeds.float(), dim=-1))
        valid_embeddings = torch.cat(encoded, dim=0)
        flat_embeddings = valid_embeddings.new_zeros(batch * frames, valid_embeddings.shape[-1])
        indexes = flat_mask.nonzero(as_tuple=False).flatten()
        flat_embeddings = flat_embeddings.index_copy(0, indexes, valid_embeddings)
        return flat_embeddings.view(batch, frames, -1)

    def encode_text(self, input_ids, attention_mask):
        output = self.blip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoded = output.last_hidden_state[:, 0]
        return F.normalize(encoded.float(), dim=-1)

    def forward(self, spatial_features, spatial_mask, input_ids=None, attention_mask=None, pixel_values=None):
        if self.vision_mode == "partial":
            spatial_features = self.encode_frames(pixel_values, spatial_mask)
        caption_features = None
        if not self.visual_only:
            caption_features = self.encode_text(input_ids, attention_mask)
        return self.spatial_branch(
            spatial_features,
            spatial_mask,
            global_caption_features=caption_features,
        )

    def adapter_state(self):
        state = {"spatial_branch": {k: v.detach().cpu() for k, v in self.spatial_branch.state_dict().items()}}
        if self.vision_encoder is not None:
            state["vision"] = {
                name: parameter.detach().cpu()
                for name, parameter in self.vision_encoder.named_parameters()
                if parameter.requires_grad
            }
        if self.vision_projection is not None:
            state["vision_projection"] = {
                name: parameter.detach().cpu()
                for name, parameter in self.vision_projection.named_parameters()
                if parameter.requires_grad
            }
        if self.blip_text is not None:
            state["blip"] = {
                name: parameter.detach().cpu()
                for name, parameter in self.blip_text.named_parameters()
                if parameter.requires_grad
            }
        return state

    def load_adapter_state(self, state):
        self.spatial_branch.load_state_dict(state["spatial_branch"], strict=True)
        for key, module in (("vision", self.vision_encoder), ("blip", self.blip_text)):
            if key not in state:
                continue
            parameters = dict(module.named_parameters())
            missing = sorted(set(state[key]) - set(parameters))
            if missing:
                raise ValueError(f"Unknown {key} parameters in checkpoint: {missing[:3]}")
            with torch.no_grad():
                for name, value in state[key].items():
                    parameters[name].copy_(value)
        if "vision_projection" in state and self.vision_projection is not None:
            parameters = dict(self.vision_projection.named_parameters())
            with torch.no_grad():
                for name, value in state["vision_projection"].items():
                    parameters[name].copy_(value)


def tokenize(tokenizer, texts, device):
    tokens = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    # SigLIP's SentencePiece tokenizer may omit an attention mask because it
    # always returns a fixed-length padded sequence.
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def forward_batch(model, batch, tokenizer, device):
    spatial = batch["spatial_features"].to(device, non_blocking=True)
    mask = batch["spatial_mask"].to(device, non_blocking=True)
    input_ids = attention_mask = None
    if tokenizer is not None:
        input_ids, attention_mask = tokenize(tokenizer, batch["caption_texts"], device)
    pixels = batch.get("pixel_values")
    if pixels is not None:
        pixels = pixels.to(device, non_blocking=True)
    return model(spatial, mask, input_ids, attention_mask, pixels)


@torch.inference_mode()
def validate(model, loader, tokenizer, loss_function, device, amp_enabled):
    model.eval()
    losses, predictions, targets = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            output = forward_batch(model, batch, tokenizer, device)
            loss, _ = loss_function(output["score"], labels)
        losses.append(float(loss))
        predictions.append(output["score"].float().cpu().numpy())
        targets.append(labels.float().cpu().numpy())
    return float(np.mean(losses)), compute_spatial_metrics(
        np.concatenate(predictions).reshape(-1), np.concatenate(targets).reshape(-1)
    )


def save_checkpoint(path, model, epoch, metrics, args):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": "vgcap-spatial-1",
            "adapter_state": model.adapter_state(),
            "branch_config": model.branch_config.to_dict(),
            "vision_model": model.vision_model_name,
            "blip_model": model.blip_model_name,
            "vision_mode": model.vision_mode,
            "visual_only": model.visual_only,
            "vision_frame_batch_size": model.vision_frame_batch_size,
            "epoch": epoch,
            "metrics": metrics,
            "training_args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if not 0 <= args.freeze_epochs < args.epochs:
        raise ValueError("freeze-epochs must be in [0, epochs)")
    if args.vision_mode == "partial" and not args.video_dir:
        raise ValueError("--video-dir is required for partial SigLIP2 fine-tuning")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    train_loader = create_loader(args, args.train_csv, True, args.limit_train)
    val_loader = create_loader(args, args.val_csv, False, args.limit_val)
    config = SpatialBranchConfig(
        spatial_dim=512,
        text_dim=768,
        hidden_dim=args.hidden_dim,
        max_frames=args.max_frames,
        use_global_caption=not args.visual_only,
        agbg_gate_init=args.agbg_gate_init,
    )
    model = SpatialQualityModel(
        config,
        args.vision_mode,
        args.vision_model,
        args.blip_model,
        args.visual_only,
        args.vision_frame_batch_size,
    ).to(device)
    tokenizer = None if args.visual_only else AutoTokenizer.from_pretrained(args.blip_model)
    spatial_params = [
        parameter
        for name, parameter in model.spatial_branch.named_parameters()
        if name != "agbg_gate_logit"
    ]
    groups = [{"params": spatial_params, "lr": args.learning_rate}]
    gate_lr = args.gate_learning_rate if args.gate_learning_rate is not None else args.learning_rate
    gate_parameter = getattr(model.spatial_branch, "agbg_gate_logit", None)
    if gate_parameter is not None:
        groups.append({"params": [gate_parameter], "lr": gate_lr})
    if model.vision_encoder is not None:
        groups.append({"params": model.vision_encoder.parameters(), "lr": args.vision_learning_rate})
    if model.vision_projection is not None:
        groups.append({"params": model.vision_projection.parameters(), "lr": args.vision_learning_rate})
    if model.blip_text is not None:
        groups.append({"params": model.blip_text.parameters(), "lr": args.blip_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_function = SpatialLoss(args.rank_weight, args.rank_margin)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_srcc = -math.inf

    for epoch in range(args.epochs):
        unfreeze = epoch >= args.freeze_epochs
        vision_blocks = args.unfreeze_last_vision_blocks if (unfreeze and args.vision_mode == "partial") else 0
        blip_layers = args.unfreeze_last_blip_layers if (unfreeze and not args.visual_only) else 0
        model.set_vision_trainable_blocks(vision_blocks)
        model.set_blip_trainable_layers(blip_layers)
        model.train()
        if model.vision_encoder is not None and not vision_blocks:
            model.vision_encoder.eval()
        if model.blip_text is not None and not blip_layers:
            model.blip_text.eval()
        running_loss = 0.0
        running_entropy = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = forward_batch(model, batch, tokenizer, device)
                loss, _ = loss_function(output["score"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
        scheduler.step()
        val_loss, metrics = validate(model, val_loader, tokenizer, loss_function, device, amp_enabled)
        gate = None
        gate_parameter = getattr(model.spatial_branch, "agbg_gate_logit", None)
        if gate_parameter is not None:
            gate = float(torch.sigmoid(gate_parameter).detach().cpu())
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "val_loss": val_loss,
            "metrics": metrics,
            "trainable_vision_blocks": vision_blocks,
            "trainable_vision_parameters": model.trainable_vision_parameters,
            "trainable_blip_layers": blip_layers,
            "trainable_blip_parameters": model.trainable_blip_parameters,
            "agbg_gate": gate,
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(output_dir / "last.pt", model, epoch + 1, metrics, args)
        selection = metrics["SRCC"] if math.isfinite(metrics["SRCC"]) else -val_loss
        if selection > best_srcc:
            best_srcc = selection
            save_checkpoint(output_dir / "best.pt", model, epoch + 1, metrics, args)
        print(
            f"epoch={epoch + 1} train_loss={record['train_loss']:.5f} val_loss={val_loss:.5f} "
            f"SRCC={metrics['SRCC']:.5f} PLCC={metrics['PLCC']:.5f} "
            f"vision_blocks={vision_blocks} blip_layers={blip_layers} gate={gate}"
        )
    print(f"Training complete: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()


