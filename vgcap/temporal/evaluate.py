#!/usr/bin/env python3
"""Evaluate a trained VGCap temporal quality model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ._components import safe_load
from .data import create_dataloader
from .model import TemporalQualityModel
from .training_utils import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "csv", "whole-dir", "seg-dir", "segment-caption-dir", "global-dir",
        "checkpoint", "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = safe_load(Path(args.checkpoint))
    model = TemporalQualityModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = create_dataloader(
        args.csv, args.whole_dir, args.seg_dir, args.segment_caption_dir,
        args.global_dir, args.batch_size, False, args.num_workers,
    )

    names, predictions, targets = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Temporal evaluation"):
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                scores = model(
                    batch["whole"].to(device),
                    batch["seg"].to(device),
                    batch["segment_caption"].to(device),
                    batch["mask"].to(device),
                    batch["global"].to(device),
                )
            names.extend(batch["video_names"])
            predictions.extend(scores.float().cpu().reshape(-1).tolist())
            targets.extend(batch["labels"].reshape(-1).tolist())

    metrics = compute_metrics(np.asarray(predictions), np.asarray(targets))
    metrics.update(
        {
            "global_gate": float(torch.sigmoid(model.global_agbg_logit).detach().cpu()),
            "segment_gate": float(torch.sigmoid(model.segment_agbg_logit).detach().cpu()),
            "fusion_gate": float(torch.sigmoid(model.final_agbg_logit).detach().cpu()),
            "best_epoch": int(checkpoint["epoch"]),
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "video_name": names,
            "Predicted_Temporal_MOS": predictions,
            "Target_Temporal_MOS": targets,
        }
    ).to_csv(output_dir / "temporal_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
