"""Data loading for the VGCap spatial branch."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..common.captions import canonical_text
from ..common.io import feature_path, safe_torch_load

class SpatialFeatureDataset(Dataset):
    def __init__(self, csv_path: str, feature_dir: str, require_labels: bool = True):
        self.table = pd.read_csv(csv_path)
        self.feature_dir = Path(feature_dir)
        required = {"video_name"}
        if require_labels:
            required.add("Spatial_MOS")
        missing = sorted(required - set(self.table.columns))
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")
        self.has_labels = "Spatial_MOS" in self.table.columns

    def __len__(self) -> int:
        return len(self.table)

    def _feature_path(self, video_name: str) -> Path:
        direct = feature_path(self.feature_dir, video_name)
        candidates = [direct] + [feature_path(self.feature_dir / split, video_name) for split in ("train", "val", "test")]
        existing = [path for path in candidates if path.exists()]
        if len(existing) > 1:
            raise ValueError(f"Duplicate spatial features for '{video_name}'")
        return existing[0] if existing else direct

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        video_name = canonical_text(row["video_name"])
        path = self._feature_path(video_name)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = safe_torch_load(path)
        if canonical_text(payload.get("video_name", "")) != video_name:
            raise ValueError(f"video_name mismatch inside {path}")
        spatial = torch.as_tensor(payload["spatial_features"], dtype=torch.float32)
        mask = torch.as_tensor(payload.get("spatial_mask", torch.ones(spatial.shape[0])), dtype=torch.bool)
        if spatial.ndim != 2 or mask.shape != spatial.shape[:1] or not mask.any() or not torch.isfinite(spatial).all():
            raise ValueError(f"Invalid spatial features for '{video_name}'")
        label = torch.tensor(float(row["Spatial_MOS"]), dtype=torch.float32) if self.has_labels else torch.tensor(float("nan"))
        if self.has_labels and (not torch.isfinite(label) or not 0 <= float(label) <= 100):
            raise ValueError(f"Spatial_MOS must be in [0,100] at row {index + 2}")
        return {"video_name": video_name, "spatial_features": spatial, "spatial_mask": mask, "label": label}

def collate_spatial(samples):
    max_frames = max(sample["spatial_features"].shape[0] for sample in samples)
    feature_dim = samples[0]["spatial_features"].shape[-1]
    spatial = torch.zeros(len(samples), max_frames, feature_dim)
    mask = torch.zeros(len(samples), max_frames, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample["spatial_features"].shape[0]
        spatial[index, :count] = sample["spatial_features"]
        mask[index, :count] = sample["spatial_mask"]
    return {"video_names": [sample["video_name"] for sample in samples], "spatial_features": spatial,
            "spatial_mask": mask, "labels": torch.stack([sample["label"] for sample in samples])}

def create_spatial_dataloader(csv_path, feature_dir, batch_size, shuffle, num_workers=4, require_labels=True):
    dataset = SpatialFeatureDataset(csv_path, feature_dir, require_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      collate_fn=collate_spatial)
