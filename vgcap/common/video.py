"""Video decoding and uniform frame sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def uniform_indices(total: int, maximum: int) -> np.ndarray:
    """Return at most ``maximum`` indices distributed across a video."""
    if total <= 0:
        raise ValueError("Video contains no frames")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    return np.linspace(0, total - 1, min(total, maximum)).round().astype(np.int64)


def read_video_images(video_path: Path, max_frames: int, backend: str = "auto"):
    """Decode uniformly sampled RGB frames, using OpenCV as a fallback."""
    if backend not in {"auto", "decord", "opencv"}:
        raise ValueError(f"Unknown video backend: {backend}")
    decord_error = None
    if backend in {"auto", "decord"}:
        try:
            import decord

            reader = decord.VideoReader(str(video_path), num_threads=1)
            indices = uniform_indices(len(reader), max_frames)
            frames = reader.get_batch(indices.tolist()).asnumpy()
            return [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames], indices
        except Exception as error:
            if backend == "decord":
                raise
            decord_error = error
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}; Decord error: {decord_error}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uniform_indices(total, max_frames)
        wanted = {int(frame): position for position, frame in enumerate(indices)}
        decoded = [None] * len(indices)
        frame_number = 0
        while frame_number <= int(indices[-1]):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded[wanted[frame_number]] = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            frame_number += 1
    finally:
        capture.release()
    if any(frame is None for frame in decoded):
        raise RuntimeError(f"Failed to decode all sampled frames from {video_path}")
    return decoded, indices

