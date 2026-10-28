#!/usr/bin/env python3
"""Extract whole-video and segment-level SlowFast-R50 motion features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..common.io import atomic_torch_save, feature_path, safe_torch_load


MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--whole-output-dir", required=True)
    parser.add_argument("--segment-output-dir", required=True)
    parser.add_argument("--mode", choices=("whole", "segment", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def decode_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {path}")
    return np.stack(frames)


def parse_segment_plan(value: object) -> list[dict[str, object]]:
    plan = json.loads(str(value))
    if not isinstance(plan, list) or not 1 <= len(plan) <= 4:
        raise ValueError("segment_plan must contain 1-4 segments")
    parsed = []
    for expected_id, item in enumerate(plan):
        indices = [int(index) for index in item["source_frame_indices"]]
        if int(item.get("segment_id", expected_id)) != expected_id:
            raise ValueError("segment_id order mismatch")
        if len(indices) != 8 or indices != sorted(indices):
            raise ValueError("Each segment must contain eight ordered source_frame_indices")
        parsed.append({"segment_id": expected_id, "source_frame_indices": indices})
    return parsed


def segment_plan_hash(plan: list[dict[str, object]]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SlowFastEncoder:
    def __init__(self, device: str) -> None:
        try:
            from pytorchvideo.models.hub import slowfast_r50
        except ImportError as exc:
            raise RuntimeError("Install pytorchvideo before extracting motion features") from exc

        self.device = torch.device(device)
        self.model = slowfast_r50(pretrained=True).eval()
        head = self.model.blocks[-1]
        if not hasattr(head, "proj"):
            raise RuntimeError("Unexpected SlowFast-R50 head")
        head.proj = nn.Identity()
        if hasattr(head, "activation"):
            head.activation = nn.Identity()
        self.model.to(self.device)

    @staticmethod
    def preprocess(frames: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        height, width = tensor.shape[-2:]
        scale = 256.0 / min(height, width)
        tensor = F.interpolate(
            tensor,
            size=(max(round(height * scale), 256), max(round(width * scale), 256)),
            mode="bilinear",
            align_corners=False,
        )
        top = (tensor.shape[-2] - 256) // 2
        left = (tensor.shape[-1] - 256) // 2
        video = tensor[:, :, top : top + 256, left : left + 256].permute(1, 0, 2, 3)
        return (video - MEAN) / STD

    @torch.inference_mode()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        fast_indices = np.linspace(0, len(frames) - 1, 32).round().astype(np.int64)
        fast = self.preprocess(frames[fast_indices]).unsqueeze(0).to(self.device)
        slow_indices = torch.linspace(0, fast.shape[2] - 1, 8, device=self.device).long()
        slow = torch.index_select(fast, 2, slow_indices)
        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            feature = self.model([slow, fast])
        feature = F.normalize(feature.flatten(1).float(), dim=-1).cpu()
        if feature.shape != (1, 2304):
            raise ValueError(f"Unexpected SlowFast output: {tuple(feature.shape)}")
        return feature[0]


def whole_payload(
    encoder: SlowFastEncoder,
    frames: np.ndarray,
    video_name: str,
) -> dict[str, object]:
    indices = np.linspace(0, len(frames) - 1, 32).round().astype(np.int64)
    return {
        "format_version": "vgcap-whole-slowfast-1",
        "video_name": video_name,
        "whole_motion_feature": encoder.encode(frames),
        "sampled_frame_indices": torch.as_tensor(indices, dtype=torch.long),
        "motion_model": "pytorchvideo/slowfast_r50_k400",
    }


def segment_payload(
    encoder: SlowFastEncoder,
    frames: np.ndarray,
    video_name: str,
    plan: list[dict[str, object]],
) -> dict[str, object]:
    features = []
    for segment in plan:
        indices = segment["source_frame_indices"]
        if min(indices) < 0 or max(indices) >= len(frames):
            raise ValueError(f"Segment frame index exceeds the {len(frames)} decoded frames")
        features.append(encoder.encode(frames[indices]))
    return {
        "format_version": "vgcap-segment-slowfast-1",
        "video_name": video_name,
        "segment_motion_features": torch.stack(features),
        "source_frame_indices": [segment["source_frame_indices"] for segment in plan],
        "segment_plan_hash": segment_plan_hash(plan),
        "motion_model": "pytorchvideo/slowfast_r50_k400",
    }


def is_complete(path: Path, format_version: str, video_name: str) -> bool:
    if not path.exists():
        return False
    payload = safe_torch_load(path)
    return payload.get("format_version") == format_version and payload.get("video_name") == video_name


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard configuration")

    table = pd.read_csv(args.csv)
    required = {"video_name"}
    if args.mode in {"segment", "both"}:
        required.add("segment_plan")
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    table = table.iloc[args.shard_id :: args.num_shards]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        table = table.head(args.limit)

    encoder = SlowFastEncoder(args.device)
    video_dir = Path(args.video_dir)
    whole_dir = Path(args.whole_output_dir)
    segment_dir = Path(args.segment_output_dir)

    for _, row in tqdm(table.iterrows(), total=len(table), desc=f"SlowFast shard {args.shard_id}"):
        video_name = " ".join(str(row["video_name"]).strip().split())
        whole_path = feature_path(whole_dir, video_name)
        segment_path = feature_path(segment_dir, video_name)
        needs_whole = args.mode in {"whole", "both"} and not (
            args.resume and is_complete(whole_path, "vgcap-whole-slowfast-1", video_name)
        )
        needs_segment = args.mode in {"segment", "both"} and not (
            args.resume and is_complete(segment_path, "vgcap-segment-slowfast-1", video_name)
        )
        if not needs_whole and not needs_segment:
            continue

        frames = decode_video(video_dir / Path(video_name.replace("\\", "/")))
        if needs_whole:
            atomic_torch_save(whole_payload(encoder, frames, video_name), whole_path)
        if needs_segment:
            plan = parse_segment_plan(row["segment_plan"])
            atomic_torch_save(segment_payload(encoder, frames, video_name, plan), segment_path)


if __name__ == "__main__":
    main()
