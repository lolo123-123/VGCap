"""Raw-video and raw-text dataset for end-to-end Alignment training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoProcessor


def canonical_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def uniform_indices(total: int, maximum: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("video contains no frames")
    return np.linspace(0, total - 1, min(total, maximum)).round().astype(np.int64)


def read_video(path: Path, maximum: int) -> tuple[list[Image.Image], np.ndarray]:
    # The Isambard Decord build is incompatible with its FFmpeg pix_fmts API.
    # Use OpenCV directly to avoid repeated decoder failures and worker stalls.
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uniform_indices(total, maximum)
        decoded: list[Image.Image | None] = [None] * len(indices)
        wanted = {int(frame): position for position, frame in enumerate(indices.tolist())}
        frame_number = 0
        while frame_number <= int(indices[-1]):
            ok, frame = capture.read()
            if not ok:
                break
            position = wanted.get(frame_number)
            if position is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded[position] = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            frame_number += 1
    finally:
        capture.release()
    if any(image is None for image in decoded):
        raise RuntimeError(f"OpenCV missed sampled frames in {path}")
    return [image for image in decoded if image is not None], indices


class RawAlignmentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        vision_model: str,
        blip_model: str,
        prompt_column: str = "Prompt",
        global_column: str = "Global_Caption",
        label_column: str = "Alignment_MOS",
        max_frames: int = 32,
        max_text_length: int = 64,
        limit: int | None = None,
    ) -> None:
        self.table = pd.read_csv(csv_path)
        required = {"video_name", prompt_column, global_column, label_column}
        missing = sorted(required - set(self.table.columns))
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        if limit is not None:
            self.table = self.table.head(limit).reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.prompt_column = prompt_column
        self.global_column = global_column
        self.label_column = label_column
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model)
        self.text_processor = AutoProcessor.from_pretrained(blip_model)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        video_name = canonical_text(row["video_name"])
        prompt = canonical_text(row[self.prompt_column])
        global_caption = canonical_text(row[self.global_column])
        if not video_name or not prompt or not global_caption:
            raise ValueError(f"empty visual/prompt/global-caption input at row {index + 2}")
        relative = Path(video_name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe video_name: {video_name}")
        video_path = self.video_dir / relative
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        images, indices = read_video(video_path, self.max_frames)
        pixels = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        tokens = self.text_processor(
            text=[prompt, global_caption],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        if not torch.isfinite(label) or not 0 <= float(label) <= 100:
            raise ValueError(f"invalid label at row {index + 2}")
        return {
            "video_name": video_name,
            "pixel_values": pixels.float(),
            "visual_mask": torch.ones(len(images), dtype=torch.bool),
            "sampled_frame_indices": torch.from_numpy(indices),
            "prompt_input_ids": tokens["input_ids"][0],
            "prompt_attention_mask": tokens["attention_mask"][0],
            "global_input_ids": tokens["input_ids"][1],
            "global_attention_mask": tokens["attention_mask"][1],
            "label": label,
        }


def collate_raw_alignment(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(samples)
    max_frames = max(len(sample["pixel_values"]) for sample in samples)
    shape = samples[0]["pixel_values"].shape[1:]
    pixels = torch.zeros(batch_size, max_frames, *shape)
    mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = len(sample["pixel_values"])
        pixels[index, :count] = sample["pixel_values"]
        mask[index, :count] = True
    return {
        "video_names": [sample["video_name"] for sample in samples],
        "pixel_values": pixels,
        "visual_mask": mask,
        "prompt_input_ids": torch.stack([sample["prompt_input_ids"] for sample in samples]),
        "prompt_attention_mask": torch.stack([sample["prompt_attention_mask"] for sample in samples]),
        "global_input_ids": torch.stack([sample["global_input_ids"] for sample in samples]),
        "global_attention_mask": torch.stack([sample["global_attention_mask"] for sample in samples]),
        "labels": torch.stack([sample["label"] for sample in samples]),
        "sampled_frame_indices": [sample["sampled_frame_indices"] for sample in samples],
    }


def create_dataloader(
    csv_path: str,
    video_dir: str,
    vision_model: str,
    blip_model: str,
    prompt_column: str,
    global_column: str,
    label_column: str,
    max_frames: int,
    max_text_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    limit: int | None = None,
) -> DataLoader:
    dataset = RawAlignmentDataset(
        csv_path, video_dir, vision_model, blip_model, prompt_column, global_column,
        label_column, max_frames, max_text_length, limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_raw_alignment,
    )

