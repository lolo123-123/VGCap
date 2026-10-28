#!/usr/bin/env python3
"""Evaluate a partially fine-tuned SigLIP2/BLIP Alignment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


from .data import create_dataloader
from .model import (
    AlignmentBranchConfig,
    AlignmentQualityModel,
    restore_parameter_state,
)
from .training_utils import atomic_json, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["training_args"]
    config = AlignmentBranchConfig.from_dict(checkpoint["alignment_config"])
    model = AlignmentQualityModel(
        config,
        checkpoint["vision_model"],
        checkpoint["blip_model"],
        checkpoint["vision_frame_batch_size"],
    )
    model.set_trainable_encoder_layers(
        checkpoint["vision_blocks"], checkpoint["blip_layers"]
    )
    model.alignment.load_state_dict(checkpoint["alignment_state"], strict=True)
    restore_parameter_state(model.siglip2_vision, checkpoint["vision_trainable_state"])
    restore_parameter_state(model.blip.text_encoder, checkpoint["blip_trainable_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    loader = create_dataloader(
        args.csv,
        args.video_dir,
        checkpoint["vision_model"],
        checkpoint["blip_model"],
        saved["prompt_column"],
        saved["global_column"],
        saved["label_column"],
        saved["max_frames"],
        saved["max_text_length"],
        args.batch_size,
        False,
        args.num_workers,
    )
    names, predictions, targets = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="SigLIP2 Alignment evaluation"):
            names.extend(batch["video_names"])
            batch = move(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(
                    batch["pixel_values"], batch["visual_mask"],
                    batch["prompt_input_ids"], batch["prompt_attention_mask"],
                    batch["global_input_ids"], batch["global_attention_mask"],
                )
            predictions.append(output["score"].float().cpu().numpy())
            targets.append(batch["labels"].float().cpu().numpy())
    prediction = np.concatenate(predictions).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame({
        "video_name": names,
        "Predicted_Alignment_MOS": prediction,
        "Target_Alignment_MOS": target,
    })
    prediction_table.to_csv(output_dir / "alignment_predictions.csv", index=False)
    metrics = compute_metrics(prediction, target)
    atomic_json(output_dir / "alignment_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

