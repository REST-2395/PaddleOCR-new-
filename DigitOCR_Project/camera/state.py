"""Shared camera OCR state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from . import config


CameraBox: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CameraDetection:
    """One tracked digit shown on the live camera preview."""

    text: str
    score: float
    box: CameraBox
    age_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraInferenceResult:
    """One inference result bundle produced by the background OCR thread."""

    frame_id: int
    detections: tuple[CameraDetection, ...] = field(default_factory=tuple)
    average_score: float = 0.0
    completed_at: float = 0.0
    mode: str = "digit"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    payload: object | None = None

    @property
    def combined_text(self) -> str:
        if self.mode == config.CAMERA_MODE_HAND_COUNT:
            payload = self.payload
            if payload is None:
                return ""
            if getattr(payload, "too_many_hands", False):
                warnings = tuple(getattr(payload, "warnings", ()) or ())
                return warnings[0] if warnings else "请仅保留两只手在画面中"
            items = tuple(getattr(payload, "items", ()) or ())
            total_count = int(getattr(payload, "total_count", 0))
            if not items and total_count <= 0:
                return "未检测到手"
            return f"总数：{total_count}"
        rows = group_camera_detections(self.detections)
        if not rows:
            return ""
        if self.mode == "board":
            return "\n".join(" ".join(item.text for item in row) for row in rows)
        return " ".join(item.text for row in rows for item in row)


@dataclass(slots=True)
class CameraSnapshot:
    """Immutable-ish snapshot consumed by the GUI preview loop."""

    has_new_frame: bool = False
    frame_bgr: np.ndarray | None = None
    frame_id: int = 0
    latest_result: CameraInferenceResult | None = None
    running: bool = False
    device_index: int = 0
    backend_name: str = ""
    capture_fps: float = 0.0
    ocr_fps: float = 0.0
    status_text: str = ""
    error_message: str | None = None

    @property
    def combined_text(self) -> str:
        """Return the tracked digits as one display string."""
        if self.latest_result is None:
            return ""
        return self.latest_result.combined_text


def sort_camera_detections(detections: tuple[CameraDetection, ...] | list[CameraDetection]) -> tuple[CameraDetection, ...]:
    """Sort detections in reading order with lightweight row grouping."""
    grouped_rows = group_camera_detections(detections)
    return tuple(item for row in grouped_rows for item in row)


def group_camera_detections(
    detections: tuple[CameraDetection, ...] | list[CameraDetection],
) -> tuple[tuple[CameraDetection, ...], ...]:
    """Group detections into reading-order rows."""
    if not detections:
        return ()

    ordered_items = sorted(detections, key=lambda item: (item.box[1], item.box[0]))
    median_height = sorted(max(1, item.box[3] - item.box[1]) for item in ordered_items)[len(ordered_items) // 2]
    row_threshold = max(10, int(median_height * 0.45))

    rows: list[dict[str, object]] = []
    for item in ordered_items:
        center_y = (item.box[1] + item.box[3]) / 2.0
        target_row: dict[str, object] | None = None
        for row in rows:
            if abs(center_y - float(row["center_y"])) <= row_threshold:
                target_row = row
                break

        if target_row is None:
            rows.append({"center_y": center_y, "items": [item]})
            continue

        row_items = target_row["items"]
        assert isinstance(row_items, list)
        row_items.append(item)
        target_row["center_y"] = sum((entry.box[1] + entry.box[3]) / 2.0 for entry in row_items) / float(len(row_items))

    grouped_items: list[tuple[CameraDetection, ...]] = []
    for row in sorted(rows, key=lambda item: float(item["center_y"])):
        row_items = sorted(row["items"], key=lambda item: item.box[0])
        grouped_items.append(tuple(row_items))
    return tuple(grouped_items)


def summarize_camera_detections(
    detections: tuple[CameraDetection, ...] | list[CameraDetection],
    *,
    mode: str = "digit",
) -> str:
    """Build a compact summary string for the right-hand result panel."""
    if not detections:
        return "未检测到数字"
    if mode == "board":
        return "\n".join(" ".join(item.text for item in row) for row in group_camera_detections(detections))
    return ", ".join(f"{item.text} ({item.score:.2f})" for item in sort_camera_detections(detections))
