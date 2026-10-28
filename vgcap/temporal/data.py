"""Data loading for the temporal quality branch."""

import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ._components import feature_path, safe_load


def canonical(value: object) -> str:
    return " ".join(str(value).strip().split())


class TemporalFeatureDataset(Dataset):
    def __init__(self, csv, whole_dir, seg_dir, segment_caption_dir, global_dir):
        self.table = pd.read_csv(csv)
        self.whole_dir = Path(whole_dir)
        self.segment_dir = Path(seg_dir)
        self.segment_caption_dir = Path(segment_caption_dir)
        self.global_dir = Path(global_dir)
        missing = sorted({"video_name", "Temporal_MOS"} - set(self.table.columns))
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        name = canonical(row.video_name)
        whole_payload = safe_load(feature_path(self.whole_dir, name))
        segment_payload = safe_load(feature_path(self.segment_dir, name))
        caption_payload = safe_load(feature_path(self.segment_caption_dir, name))
        global_payload = safe_load(feature_path(self.global_dir, name))
        whole = torch.as_tensor(whole_payload["whole_motion_feature"], dtype=torch.float32)
        segment = torch.as_tensor(segment_payload["segment_motion_features"], dtype=torch.float32)
        segment_caption = torch.as_tensor(caption_payload["segment_caption_features"], dtype=torch.float32)
        global_caption = torch.as_tensor(global_payload["global_caption_feature"], dtype=torch.float32)
        if whole.shape != (2304,) or segment.ndim != 2 or not 1 <= len(segment) <= 4:
            raise ValueError(f"Invalid motion feature shapes for {name}")
        if segment.shape[1] != 2304 or segment_caption.shape != (len(segment), 768) or global_caption.shape != (768,):
            raise ValueError(f"Invalid caption feature shapes for {name}")
        label = float(row.Temporal_MOS)
        if not math.isfinite(label):
            raise ValueError(f"Invalid Temporal_MOS for {name}")
        return {"video_name": name, "whole": whole, "seg": segment, "segment_caption": segment_caption,
                "global": global_caption, "label": torch.tensor([label], dtype=torch.float32)}


def collate(samples):
    batch_size = len(samples)
    count = max(len(sample["seg"]) for sample in samples)
    segment = torch.zeros(batch_size, count, 2304)
    segment_caption = torch.zeros(batch_size, count, 768)
    mask = torch.zeros(batch_size, count, dtype=torch.bool)
    for index, sample in enumerate(samples):
        valid = len(sample["seg"])
        segment[index, :valid] = sample["seg"]
        segment_caption[index, :valid] = sample["segment_caption"]
        mask[index, :valid] = True
    return {"video_names": [sample["video_name"] for sample in samples],
            "whole": torch.stack([sample["whole"] for sample in samples]),
            "seg": segment, "segment_caption": segment_caption, "mask": mask,
            "global": torch.stack([sample["global"] for sample in samples]),
            "labels": torch.stack([sample["label"] for sample in samples])}


def create_dataloader(csv, whole, segment, segment_caption, global_caption, batch_size, shuffle, workers):
    dataset = TemporalFeatureDataset(csv, whole, segment, segment_caption, global_caption)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      collate_fn=collate)


