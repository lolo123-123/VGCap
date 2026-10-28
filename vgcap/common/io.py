"""Shared filesystem helpers and feature-file contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import torch

from .captions import canonical_text


def text_hash(value: Any) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def feature_path(root: str | Path, video_name: str) -> Path:
    relative = Path(str(video_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe video_name: {video_name}")
    return (Path(root) / relative).with_suffix(".pt")


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
