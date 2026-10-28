#!/usr/bin/env python3
"""Train Alignment with SigLIP2 vision and scheduled partial fine-tuning."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


from .data import create_dataloader
from .model import (
    AlignmentBranchConfig,
    AlignmentQualityModel,
    trainable_parameter_state,
)
from .training_utils import AlignmentLoss, atomic_checkpoint, atomic_json, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-column", default="Prompt")
    parser.add_argument("--global-column", default="Global_Caption")
    parser.add_argument("--label-column", default="Alignment_MOS")
    parser.add_argument("--vision-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--blip-model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-vision-blocks", type=int, default=2)
    parser.add_argument("--unfreeze-blip-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-frame-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--vision-learning-rate", type=float, default=2e-6)
    parser.add_argument("--blip-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--fusion-mode",
        choices=("agbg", "visual_prompt", "visual_only"),
        default="agbg",
        help="VGCap fusion or an alignment input ablation reported in the paper",
    )
    parser.add_argument("--gate-init", type=float, nargs=3, default=(0.55, 0.25, 0.20))
    parser.add_argument("--rank-weight", type=float, default=0.3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_model(model, batch):
    return model(
        batch["pixel_values"], batch["visual_mask"],
        batch["prompt_input_ids"], batch["prompt_attention_mask"],
        batch["global_input_ids"], batch["global_attention_mask"],
    )


@torch.inference_mode()
def validate(model, loader, loss_fn, device, amp_enabled):
    model.eval()
    losses, predictions, targets = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            scores = run_model(model, batch)["score"]
            loss = loss_fn(scores, batch["labels"])
        losses.append(float(loss))
        predictions.append(scores.float().cpu().numpy())
        targets.append(batch["labels"].float().cpu().numpy())
    prediction = np.concatenate(predictions).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    return float(np.mean(losses)), compute_metrics(prediction, target)


def checkpoint_payload(model, epoch, metrics, args, vision_blocks, blip_layers):
    return {
        "format_version": "alignment-partial-siglip2-blip-1",
        "alignment_state": {
            key: value.detach().cpu() for key, value in model.alignment.state_dict().items()
        },
        "vision_trainable_state": trainable_parameter_state(model.siglip2_vision),
        "blip_trainable_state": trainable_parameter_state(model.blip.text_encoder),
        "alignment_config": model.alignment_config.to_dict(),
        "vision_model": model.vision_model_name,
        "blip_model": model.blip_model_name,
        "vision_frame_batch_size": model.vision_frame_batch_size,
        "vision_blocks": vision_blocks,
        "blip_layers": blip_layers,
        "epoch": epoch,
        "metrics": metrics,
        "training_args": vars(args),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.freeze_epochs < args.epochs:
        raise ValueError("freeze-epochs must be in [0, epochs)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    loaders = [
        create_dataloader(
            path, args.video_dir, args.vision_model, args.blip_model,
            args.prompt_column, args.global_column, args.label_column,
            args.max_frames, args.max_text_length, args.batch_size,
            shuffle, args.num_workers, limit,
        )
        for path, shuffle, limit in (
            (args.train_csv, True, args.limit_train),
            (args.val_csv, False, args.limit_val),
        )
    ]
    config = AlignmentBranchConfig(
        visual_dim=768,
        text_dim=768,
        hidden_dim=args.hidden_dim,
        max_frames=args.max_frames,
        gate_init=tuple(args.gate_init),
        fusion_mode=args.fusion_mode,
    )
    model = AlignmentQualityModel(
        config, args.vision_model, args.blip_model, args.vision_frame_batch_size
    ).to(device)
    groups = [
        {"params": model.alignment.parameters(), "lr": args.learning_rate},
        {"params": model.siglip2_vision.parameters(), "lr": args.vision_learning_rate},
        {"params": model.blip.text_encoder.parameters(), "lr": args.blip_learning_rate},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = AlignmentLoss(args.rank_weight, args.rank_margin)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_srcc = [], -math.inf
    for epoch in range(args.epochs):
        vision_blocks = 0 if epoch < args.freeze_epochs else args.unfreeze_vision_blocks
        blip_layers = 0 if epoch < args.freeze_epochs else args.unfreeze_blip_layers
        model.train()
        model.set_trainable_encoder_layers(vision_blocks, blip_layers)
        running = 0.0
        for batch in tqdm(loaders[0], desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                scores = run_model(model, batch)["score"]
                loss = loss_fn(scores, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        scheduler.step()
        val_loss, metrics = validate(model, loaders[1], loss_fn, device, amp_enabled)
        weights = model.alignment.gate_weights().detach().float().cpu().tolist()
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(len(loaders[0]), 1),
            "val_loss": val_loss,
            "metrics": metrics,
            "vision_blocks": vision_blocks,
            "blip_layers": blip_layers,
            "trainable_parameters": model.trainable_counts(),
            "gate_weights": {"V": weights[0], "P": weights[1], "C": weights[2]},
        }
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        payload = checkpoint_payload(
            model, epoch + 1, metrics, args, vision_blocks, blip_layers
        )
        atomic_checkpoint(output_dir / "last.pt", payload)
        selection = metrics["SRCC"] if math.isfinite(metrics["SRCC"]) else -val_loss
        if selection > best_srcc:
            best_srcc = selection
            atomic_checkpoint(output_dir / "best.pt", payload)
        print(
            f"epoch={epoch + 1} loss={record['train_loss']:.5f} val={val_loss:.5f} "
            f"SRCC={metrics['SRCC']:.5f} PLCC={metrics['PLCC']:.5f} "
            f"vision_blocks={vision_blocks} blip_layers={blip_layers} "
            f"w={weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f}"
        )
    print(f"Training complete: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

