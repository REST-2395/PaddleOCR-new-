"""Shared camera ROI and crop helpers."""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .state import CameraBox


def camera_roi_box(
    frame_shape: tuple[int, ...],
    *,
    width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
    height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
) -> CameraBox:
    """Return the fixed centered scan box used by both preview and OCR."""
    height, width = frame_shape[:2]
    roi_width = min(width, max(48, int(round(width * width_ratio))))
    roi_height = min(height, max(48, int(round(height * height_ratio))))
    x0 = max(0, (width - roi_width) // 2)
    y0 = max(0, (height - roi_height) // 2)
    x1 = min(width, x0 + roi_width)
    y1 = min(height, y0 + roi_height)
    return x0, y0, x1, y1


def crop_camera_roi(
    frame_bgr: np.ndarray,
    roi_box: CameraBox | None = None,
    *,
    width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
    height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
) -> tuple[np.ndarray, CameraBox]:
    """Crop the shared camera ROI from one frame."""
    box = (
        camera_roi_box(frame_bgr.shape, width_ratio=width_ratio, height_ratio=height_ratio)
        if roi_box is None
        else roi_box
    )
    x0, y0, x1, y1 = box
    return frame_bgr[y0:y1, x0:x1].copy(), box


def camera_roi_foreground_ratio(frame_bgr: np.ndarray, roi_box: CameraBox | None = None) -> float:
    """Estimate whether the scan box contains enough foreground detail to justify OCR."""
    roi_frame, _ = crop_camera_roi(frame_bgr, roi_box=roi_box)
    if roi_frame.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.GaussianBlur(blurred, (0, 0), sigmaX=9.0, sigmaY=9.0)
    difference = cv2.absdiff(blurred, background)
    _, mask = cv2.threshold(difference, 16, 255, cv2.THRESH_BINARY)
    if min(mask.shape[:2]) >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return float(np.count_nonzero(mask)) / float(max(1, mask.size))


def camera_roi_has_foreground(
    frame_bgr: np.ndarray,
    *,
    roi_box: CameraBox | None = None,
    min_ratio: float = config.CAMERA_ROI_MIN_FOREGROUND_RATIO,
) -> bool:
    """Return whether the ROI likely contains digits or strokes worth sending to OCR."""
    return camera_roi_foreground_ratio(frame_bgr, roi_box=roi_box) >= max(0.0, float(min_ratio))


def _resize_for_ocr(frame: np.ndarray, *, max_side: int) -> tuple[np.ndarray, float, float]:
    """Downscale frames before OCR and return reverse scale factors."""
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame.copy(), 1.0, 1.0

    scale = max_side / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(resized_width), height / float(resized_height)


__all__ = [
    "_resize_for_ocr",
    "camera_roi_box",
    "camera_roi_foreground_ratio",
    "camera_roi_has_foreground",
    "crop_camera_roi",
]
