#!/usr/bin/env python3
"""Extract frozen BLIP features for global and segment captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, BlipForImageTextRetrieval

from ..common.captions import canonical_text
from ..common.io import atomic_torch_save, feature_path, safe_torch_load, text_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--global-output-dir", required=True)
    parser.add_argument("--segment-output-dir", required=True)
    parser.add_argument("--mode", choices=("global", "segment", "both"), default="both")
    parser.add_argument("--global-column", default="Global_Caption")
    parser.add_argument("--segment-column", default="Segment_Captions")
    parser.add_argument("--model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_segment_captions(value: object) -> list[str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not 1 <= len(parsed) <= 4:
        raise ValueError("Segment_Captions must be a JSON list containing 1-4 captions")
    captions = []
    for item in parsed:
        caption = item.get("text", "") if isinstance(item, dict) else item
        caption = canonical_text(caption)
        if not caption:
            raise ValueError("Segment_Captions contains an empty caption")
        captions.append(caption)
    return captions


class BLIPTextEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = BlipForImageTextRetrieval.from_pretrained(model_name, torch_dtype=dtype)
        self.encoder = model.text_encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, captions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            text=captions,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        tokens.pop("token_type_ids", None)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        hidden = self.encoder(**tokens, return_dict=True).last_hidden_state[:, 0]
        return F.normalize(hidden.float(), dim=-1).cpu()


def valid_global(path: Path, video_name: str, caption_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = safe_torch_load(path)
        feature = torch.as_tensor(payload["global_caption_feature"])
        return (
            payload.get("video_name") == video_name
            and payload.get("global_caption_hash") == caption_hash
            and feature.shape == (768,)
            and torch.isfinite(feature).all().item()
        )
    except Exception:
        return False


def valid_segment(path: Path, video_name: str, caption_hashes: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = safe_torch_load(path)
        features = torch.as_tensor(payload["segment_caption_features"])
        return (
            payload.get("video_name") == video_name
            and payload.get("segment_caption_hashes") == caption_hashes
            and features.shape == (len(caption_hashes), 768)
            and torch.isfinite(features).all().item()
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard configuration")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    table = pd.read_csv(args.csv)
    required = {"video_name"}
    if args.mode in {"global", "both"}:
        required.add(args.global_column)
    if args.mode in {"segment", "both"}:
        required.add(args.segment_column)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if table["video_name"].duplicated().any():
        raise ValueError("video_name must be unique")
    table = table.iloc[args.shard_id :: args.num_shards]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        table = table.head(args.limit)

    global_dir = Path(args.global_output_dir)
    segment_dir = Path(args.segment_output_dir)
    pending: list[str] = []
    jobs: list[dict[str, object]] = []

    for row_index, row in table.iterrows():
        video_name = canonical_text(row["video_name"])
        if args.mode in {"global", "both"}:
            caption = canonical_text(row[args.global_column])
            if not caption:
                raise ValueError(f"Empty global caption at row {row_index + 2}")
            caption_hash = text_hash(caption)
            destination = feature_path(global_dir, video_name)
            if not (args.resume and valid_global(destination, video_name, caption_hash)):
                start = len(pending)
                pending.append(caption)
                jobs.append({
                    "kind": "global",
                    "video_name": video_name,
                    "captions": [caption],
                    "hashes": [caption_hash],
                    "path": destination,
                    "start": start,
                    "end": len(pending),
                })

        if args.mode in {"segment", "both"}:
            captions = parse_segment_captions(row[args.segment_column])
            caption_hashes = [text_hash(caption) for caption in captions]
            destination = feature_path(segment_dir, video_name)
            if not (args.resume and valid_segment(destination, video_name, caption_hashes)):
                start = len(pending)
                pending.extend(captions)
                jobs.append({
                    "kind": "segment",
                    "video_name": video_name,
                    "captions": captions,
                    "hashes": caption_hashes,
                    "path": destination,
                    "start": start,
                    "end": len(pending),
                })

    encoder = BLIPTextEncoder(args.model, args.device) if pending else None
    batches = []
    for start in tqdm(range(0, len(pending), args.batch_size), desc="BLIP captions"):
        batches.append(encoder.encode(pending[start : start + args.batch_size]))
    encoded = torch.cat(batches, dim=0) if batches else torch.empty(0, 768)

    for job in jobs:
        start, end = int(job["start"]), int(job["end"])
        features = encoded[start:end].clone()
        if job["kind"] == "global":
            payload = {
                "format_version": "vgcap-global-caption-blip-1",
                "video_name": job["video_name"],
                "global_caption": job["captions"][0],
                "global_caption_hash": job["hashes"][0],
                "global_caption_feature": features[0],
                "text_encoder": args.model,
            }
        else:
            payload = {
                "format_version": "vgcap-segment-caption-blip-1",
                "video_name": job["video_name"],
                "segment_captions": job["captions"],
                "segment_caption_hashes": job["hashes"],
                "segment_caption_features": features,
                "text_encoder": args.model,
            }
        atomic_torch_save(payload, job["path"])

    print(json.dumps({"rows": len(table), "files_written": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
