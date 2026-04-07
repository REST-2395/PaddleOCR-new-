"""Preview overlay helpers for live camera OCR."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from . import config
from .roi import camera_roi_box
from .state import CameraDetection


def resize_camera_frame_for_preview(
    frame_bgr: np.ndarray,
    *,
    max_dimension: int = config.CAMERA_MAX_PREVIEW_SIZE,
) -> np.ndarray:
    """Resize a camera frame so the preview path stays lightweight."""
    height, width = frame_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return frame_bgr.copy()

    scale = max_dimension / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def overlay_camera_detections(
    frame_bgr: np.ndarray,
    detections: Sequence[CameraDetection],
    *,
    capture_fps: float,
    ocr_fps: float,
    prompt_text: str = "Place digits in scan box",
    camera_mode: str = config.CAMERA_MODE_DIGIT,
    roi_width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
    roi_height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
) -> np.ndarray:
    """Draw camera detections and runtime stats on one preview frame."""
    darkened = cv2.addWeighted(frame_bgr, 0.35, np.zeros_like(frame_bgr), 0.65, 0.0)
    overlay = darkened
    del camera_mode, prompt_text

    roi_box = camera_roi_box(
        frame_bgr.shape,
        width_ratio=roi_width_ratio,
        height_ratio=roi_height_ratio,
    )
    x0, y0, x1, y1 = roi_box
    overlay[y0:y1, x0:x1] = frame_bgr[y0:y1, x0:x1]

    border_color = (60, 210, 60) if detections else (0, 215, 255)
    border_thickness = 3 if detections else 2
    cv2.rectangle(overlay, (x0, y0), (x1, y1), border_color, border_thickness)

    for item in detections:
        item_x0, item_y0, item_x1, item_y1 = item.box
        cv2.rectangle(overlay, (item_x0, item_y0), (item_x1, item_y1), (0, 255, 0), 2)
        label = f"{item.text} ({item.score:.2f})"
        label_origin = (item_x0, max(24, item_y0 - 8))
        cv2.putText(
            overlay,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    stats_label = f"FPS {capture_fps:.1f} | OCR {ocr_fps:.1f}"
    cv2.rectangle(overlay, (8, 8), (260, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        overlay,
        stats_label,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


__all__ = [
    "overlay_camera_detections",
    "resize_camera_frame_for_preview",
]
