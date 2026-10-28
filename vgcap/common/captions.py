"""Parsing and validation for timestamped segment captions."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, List


_TIMESTAMP_PATTERN = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<end>\d+(?:\.\d+)?)\s*s?\s*[:：]\s*"
    r"(?P<text>.*?)(?=(?:\s*[|;\n]\s*)?\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\s*[:：]|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def canonical_text(value: Any) -> str:
    """Normalize user-provided text without changing its meaning."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).strip().split())


@dataclass(frozen=True)
class SegmentCaption:
    start: float | None
    end: float | None
    text: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


def _from_mapping(item: dict) -> SegmentCaption:
    text = canonical_text(
        item.get("text", item.get("caption", item.get("description", item.get("prompt", ""))))
    )
    start = item.get("start", item.get("start_time"))
    end = item.get("end", item.get("end_time"))
    return SegmentCaption(
        None if start in (None, "") else float(start),
        None if end in (None, "") else float(end),
        text,
    )


def _from_iterable(items: Iterable[Any]) -> List[SegmentCaption]:
    captions: List[SegmentCaption] = []
    for item in items:
        if isinstance(item, dict):
            captions.append(_from_mapping(item))
        elif isinstance(item, str):
            text = canonical_text(item)
            if text:
                captions.append(SegmentCaption(None, None, text))
        else:
            raise ValueError(f"Unsupported segment caption item: {type(item).__name__}")
    return captions


def parse_segment_captions(value: Any) -> List[SegmentCaption]:
    """Parse JSON, Python-list, timestamp text, or plain caption lists.

    Supported examples:
      - [{"start": 0, "end": 2, "text": "A person turns."}]
      - "0-2s: A person turns. | 2-4s: The person walks."
      - ["A person turns.", "The person walks."]
    """
    if isinstance(value, (list, tuple)):
        captions = _from_iterable(value)
    elif isinstance(value, dict):
        nested = value.get("segments", value.get("captions"))
        captions = _from_iterable(nested) if isinstance(nested, (list, tuple)) else [_from_mapping(value)]
    else:
        raw = canonical_text(value)
        if not raw:
            raise ValueError("Segment_Captions is empty")
        parsed: Any = None
        if raw[0] in "[{":
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(raw)
                    break
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
        if parsed is not None:
            if isinstance(parsed, dict):
                nested = parsed.get("segments", parsed.get("captions"))
                captions = _from_iterable(nested) if isinstance(nested, (list, tuple)) else [_from_mapping(parsed)]
            elif isinstance(parsed, (list, tuple)):
                captions = _from_iterable(parsed)
            else:
                raise ValueError("Parsed Segment_Captions must be a list or object")
        else:
            matches = list(_TIMESTAMP_PATTERN.finditer(raw))
            if matches:
                captions = [
                    SegmentCaption(float(m.group("start")), float(m.group("end")), canonical_text(m.group("text")))
                    for m in matches
                ]
            else:
                parts = [canonical_text(part) for part in re.split(r"\s*[|\n]\s*", raw)]
                captions = [SegmentCaption(None, None, part) for part in parts if part]

    captions = [caption for caption in captions if caption.text]
    if not captions:
        raise ValueError("Segment_Captions contains no non-empty captions")
    for caption in captions:
        if (caption.start is None) != (caption.end is None):
            raise ValueError("Each segment must provide both start and end, or neither")
        if caption.start is not None and (caption.start < 0 or caption.end <= caption.start):
            raise ValueError(f"Invalid segment interval: {caption.start}-{caption.end}")
    timed = [caption for caption in captions if caption.start is not None]
    if timed and len(timed) != len(captions):
        raise ValueError("Do not mix timestamped and untimestamped segment captions")
    if timed:
        captions = sorted(captions, key=lambda caption: float(caption.start))
    return captions


def select_segments(captions: List[SegmentCaption], max_segments: int) -> List[SegmentCaption]:
    """Keep temporal coverage when a sample contains too many segments."""
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if len(captions) <= max_segments:
        return captions
    if max_segments == 1:
        return [captions[len(captions) // 2]]
    indices = [round(i * (len(captions) - 1) / (max_segments - 1)) for i in range(max_segments)]
    return [captions[index] for index in indices]


def resolve_intervals(captions: List[SegmentCaption], duration: float) -> List[SegmentCaption]:
    """Clamp timestamps to the video or assign equal intervals when absent."""
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    if all(caption.start is None for caption in captions):
        width = duration / len(captions)
        return [
            SegmentCaption(i * width, duration if i == len(captions) - 1 else (i + 1) * width, caption.text)
            for i, caption in enumerate(captions)
        ]
    resolved: List[SegmentCaption] = []
    for caption in captions:
        start = min(max(float(caption.start), 0.0), duration)
        end = min(max(float(caption.end), 0.0), duration)
        if end <= start:
            raise ValueError(
                f"Segment {caption.start}-{caption.end}s lies outside a {duration:.3f}s video"
            )
        resolved.append(SegmentCaption(start, end, caption.text))
    return resolved
