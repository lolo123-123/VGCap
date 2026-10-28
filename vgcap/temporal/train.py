#!/usr/bin/env python3
"""Train the VGCap temporal quality branch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import create_dataloader
from .model import TemporalQualityModel
from .training_utils import TemporalLoss, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-csv", "val-csv", "whole-dir", "seg-dir", "segment-caption-dir",
        "global-train-dir", "global-val-dir", "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-gate-init", type=float, default=0.33)
    parser.add_argument("--segment-gate-init", type=float, default=0.45)
    parser.add_argument("--fusion-gate-init", type=float, default=0.65)
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predict(model: TemporalQualityModel, batch: dict) -> torch.Tensor:
    return model(
        batch["whole"], batch["seg"], batch["segment_caption"],
        batch["mask"], batch["global"],
    )


@torch.inference_mode()
def validate(model, loader, loss_function, device, amp_enabled):
    model.eval()
    losses, predictions, targets = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            scores = predict(model, batch)
            loss = loss_function(scores, batch["labels"])
        losses.append(float(loss))
        predictions.append(scores.float().cpu().numpy())
        targets.append(batch["labels"].float().cpu().numpy())
    return float(np.mean(losses)), compute_metrics(
        np.concatenate(predictions), np.concatenate(targets)
    )


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": "vgcap-temporal-1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config,
            "epoch": epoch,
            "metrics": metrics,
            "training_args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = TemporalQualityModel(
        global_gate_init=args.global_gate_init,
        segment_gate_init=args.segment_gate_init,
        fusion_gate_init=args.fusion_gate_init,
    ).to(device)
    train_loader = create_dataloader(
        args.train_csv, args.whole_dir, args.seg_dir, args.segment_caption_dir,
        args.global_train_dir, args.batch_size, True, args.num_workers,
    )
    val_loader = create_dataloader(
        args.val_csv, args.whole_dir, args.seg_dir, args.segment_caption_dir,
        args.global_val_dir, args.batch_size, False, args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_function = TemporalLoss(0.3, 0.05)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_srcc = -math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = loss_function(predict(model, batch), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())

        scheduler.step()
        val_loss, metrics = validate(model, val_loader, loss_function, device, amp_enabled)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "metrics": metrics,
            "global_gate": float(torch.sigmoid(model.global_agbg_logit).detach().cpu()),
            "segment_gate": float(torch.sigmoid(model.segment_agbg_logit).detach().cpu()),
            "fusion_gate": float(torch.sigmoid(model.final_agbg_logit).detach().cpu()),
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, args)
        selection = metrics["SRCC"] if math.isfinite(metrics["SRCC"]) else -val_loss
        if selection > best_srcc:
            best_srcc = selection
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics, args)
        print(
            f"epoch={epoch} SRCC={metrics['SRCC']:.5f} PLCC={metrics['PLCC']:.5f} "
            f"global={record['global_gate']:.5f} seg={record['segment_gate']:.5f} "
            f"final={record['fusion_gate']:.5f}"
        )

    print(f"Training complete: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
