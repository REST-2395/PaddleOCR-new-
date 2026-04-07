"""Preview overlay helpers for the live hand-count mode."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from camera.roi import camera_roi_box

from .constants import LEFT_HAND_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS, RIGHT_HAND_COLOR, ROI_ACTIVE_COLOR, ROI_IDLE_COLOR, TEXT_COLOR, WARNING_COLOR
from .types import HandCountItem, HandCountPayload


def overlay_hand_count_frame(
    frame_bgr: np.ndarray,
    payload: HandCountPayload | None,
    *,
    capture_fps: float,
    count_fps: float,
    prompt_text: str,
    roi_width_ratio: float,
    roi_height_ratio: float,
    connections: Sequence[tuple[int, int]] = (),
) -> np.ndarray:
    """Render the hand-count preview with ROI, landmarks, labels, and stats."""
    darkened = cv2.addWeighted(frame_bgr, 0.35, np.zeros_like(frame_bgr), 0.65, 0.0)
    overlay = darkened
    del prompt_text
    roi_box = camera_roi_box(
        frame_bgr.shape,
        width_ratio=roi_width_ratio,
        height_ratio=roi_height_ratio,
    )
    x0, y0, x1, y1 = roi_box
    overlay[y0:y1, x0:x1] = frame_bgr[y0:y1, x0:x1]

    has_items = payload is not None and bool(payload.items)
    border_color = ROI_ACTIVE_COLOR if has_items else ROI_IDLE_COLOR
    cv2.rectangle(overlay, (x0, y0), (x1, y1), border_color, 2 if not has_items else 3)

    if payload is not None:
        for item in payload.items:
            _draw_hand_item(overlay, item, connections=connections)
        _draw_total_label(overlay, payload)
        if payload.warnings:
            _draw_warning(overlay, "Keep only two hands in frame")
        elif not payload.items:
            _draw_warning(overlay, "No hands", color=TEXT_COLOR)

    stats_label = f"FPS {capture_fps:.1f} | Count {count_fps:.1f}"
    cv2.rectangle(overlay, (8, 8), (250, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        overlay,
        stats_label,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return overlay


def _draw_hand_item(
    overlay: np.ndarray,
    item: HandCountItem,
    *,
    connections: Sequence[tuple[int, int]],
) -> None:
    color = LEFT_HAND_COLOR if item.handedness == "Left" else RIGHT_HAND_COLOR
    x0, y0, x1, y1 = item.box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

    for start_index, end_index in connections:
        if start_index >= len(item.landmarks) or end_index >= len(item.landmarks):
            continue
        start = item.landmarks[start_index]
        end = item.landmarks[end_index]
        cv2.line(
            overlay,
            (start.x, start.y),
            (end.x, end.y),
            color,
            LANDMARK_THICKNESS,
            cv2.LINE_AA,
        )
    for point in item.landmarks:
        cv2.circle(overlay, (point.x, point.y), LANDMARK_RADIUS, color, thickness=-1)

    label = f"{item.handedness}: {item.count}"
    cv2.putText(
        overlay,
        label,
        (x0, max(24, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_total_label(overlay: np.ndarray, payload: HandCountPayload) -> None:
    if payload.too_many_hands:
        return
    label = f"Total: {payload.total_count}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    width = overlay.shape[1]
    label_x = max(12, int((width - label_size[0]) / 2))
    cv2.putText(
        overlay,
        label,
        (label_x, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _draw_warning(overlay: np.ndarray, warning: str, *, color: tuple[int, int, int] = WARNING_COLOR) -> None:
    height = overlay.shape[0]
    cv2.putText(
        overlay,
        warning,
        (16, max(42, height - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


__all__ = ["overlay_hand_count_frame"]
