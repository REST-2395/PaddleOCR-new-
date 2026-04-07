"""Shared protocol objects for the live camera OCR runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .mode_profiles import get_camera_mode_profile
from .state import CameraBox, CameraDetection
from core.ocr_engine import OCRResult

PerspectiveMatrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

_DEFAULT_MODE_PROFILE = get_camera_mode_profile(config.CAMERA_MODE_DIGIT)


@dataclass(slots=True)
class CameraTrack:
    """One smoothed camera detection track."""

    track_id: int
    detection: CameraDetection
    misses: int = 0


@dataclass(slots=True)
class CameraOCRWorkerConfig:
    """Serializable settings used by the camera OCR child process."""

    dict_path: str
    worker_kind: str = "fallback"
    ocr_version: str = "PP-OCRv5"
    score_threshold: float = 0.3
    use_gpu: bool = False
    det_model_dir: str | None = None
    rec_model_dir: str | None = None
    cls_model_dir: str | None = None
    cpu_threads: int = _DEFAULT_MODE_PROFILE.cpu_threads
    camera_mode: str = _DEFAULT_MODE_PROFILE.mode
    enable_mkldnn: bool = _DEFAULT_MODE_PROFILE.enable_mkldnn
    use_textline_orientation: bool = _DEFAULT_MODE_PROFILE.use_textline_orientation
    language: str = "en"


@dataclass(slots=True)
class CameraOCRTask:
    """One OCR task sent from the runtime to the child process."""

    frame_id: int
    task_kind: str = "fallback_roi"
    ocr_frame: np.ndarray | None = None
    candidate_images: tuple[np.ndarray, ...] = ()
    candidate_boxes: tuple[CameraBox, ...] = ()
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    allow_fallback: bool = True
    generation: int = 0
    started_at: float = 0.0
    inverse_matrix: PerspectiveMatrix | None = None
    camera_mode: str = config.CAMERA_MODE_DIGIT


@dataclass(slots=True)
class CameraOCRWorkerResult:
    """One OCR result message sent back from the child process."""

    frame_id: int
    task_kind: str = "fallback_roi"
    results: tuple[OCRResult, ...] = ()
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    fallback_used: bool = False
    worker_error: str | None = None
    warnings: tuple[str, ...] = ()
    generation: int = 0
    started_at: float = 0.0
    inverse_matrix: PerspectiveMatrix | None = None
    camera_mode: str = config.CAMERA_MODE_DIGIT


@dataclass(slots=True)
class PendingFastFrame:
    """One in-flight fast-path frame awaiting all worker batches."""

    frame_id: int
    expected_batches: int
    total_candidates: int
    fallback_task: CameraOCRTask | None = None
    generation: int = 0
    started_at: float = 0.0
    results: list[OCRResult] = field(default_factory=list)
    completed_batches: int = 0


@dataclass(slots=True)
class BoardFramePlan:
    """Prepared board-mode OCR work or one gate decision."""

    task: CameraOCRTask | None = None
    status_text: str = ""
    gate_reason: str | None = None


__all__ = [
    "BoardFramePlan",
    "CameraOCRTask",
    "CameraOCRWorkerConfig",
    "CameraOCRWorkerResult",
    "CameraTrack",
    "PendingFastFrame",
    "PerspectiveMatrix",
]
