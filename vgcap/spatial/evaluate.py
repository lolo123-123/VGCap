#!/usr/bin/env python3
"""Evaluate a partial-encoder spatial model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .train import (
    SpatialQualityModel,
    create_loader,
    forward_batch,
)
from ..common.io import safe_torch_load
from .model import SpatialBranchConfig
from .training_utils import compute_spatial_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--feature-dir", help="Required only for frozen-vision checkpoints")
    parser.add_argument("--video-dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = safe_torch_load(Path(args.checkpoint))
    training_args = argparse.Namespace(**checkpoint["training_args"])
    training_args.train_csv = args.csv
    training_args.feature_dir = args.feature_dir
    training_args.video_dir = args.video_dir
    training_args.batch_size = args.batch_size
    training_args.num_workers = args.num_workers
    model = SpatialQualityModel(
        SpatialBranchConfig(**checkpoint["branch_config"]),
        checkpoint["vision_mode"],
        checkpoint["vision_model"],
        checkpoint["blip_model"],
        checkpoint["visual_only"],
        checkpoint["vision_frame_batch_size"],
    ).to(device)
    model.load_adapter_state(checkpoint["adapter_state"])
    model.eval()
    tokenizer = None if checkpoint["visual_only"] else AutoTokenizer.from_pretrained(
        checkpoint["blip_model"]
    )
    loader = create_loader(training_args, args.csv, False, args.limit)
    names, predictions, targets = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluation"):
            names.extend(batch["video_names"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = forward_batch(model, batch, tokenizer, device)
            predictions.append(output["score"].float().cpu().numpy())
            targets.append(batch["labels"].float().numpy())
    prediction = np.concatenate(predictions).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    metrics = compute_spatial_metrics(prediction, target)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "video_name": names,
        "Predicted_Spatial_MOS": prediction,
        "Target_Spatial_MOS": target,
    }).to_csv(output_dir / "spatial_predictions.csv", index=False)
    (output_dir / "spatial_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()


