"""Shared runtime helpers for live camera OCR."""

from __future__ import annotations

import multiprocessing as mp
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import config
from .config import (
    CAMERA_EMPTY_RESET_FRAMES,
    CAMERA_MODE_BOARD,
    CAMERA_MODE_DIGIT,
    CAMERA_MODE_HAND_COUNT,
    CAMERA_PREVIEW_INTERVAL_MS,
    CAMERA_ROI_HEIGHT_RATIO,
    CAMERA_ROI_WIDTH_RATIO,
    CAMERA_SEQUENCE_HISTORY,
)
from .fast_path import (
    build_camera_detections_from_results,
    camera_detection_signature,
    camera_result_is_fresh,
    filter_camera_detections,
    stable_camera_sequence,
    stabilize_camera_detections,
)
from .mode_profiles import get_camera_mode_profile
from .overlay import overlay_camera_detections
from .protocol import (
    BoardFramePlan,
    CameraOCRTask,
    CameraOCRWorkerConfig,
    CameraOCRWorkerResult,
    CameraTrack,
    PendingFastFrame,
)
from .runtime_lifecycle import RuntimeLifecycleMixin
from .runtime_loop_facade import RuntimeLoopFacadeMixin
from .runtime_worker_control import RuntimeWorkerControlMixin
from .state import (
    CameraDetection,
    CameraInferenceResult,
    CameraSnapshot,
    group_camera_detections,
    sort_camera_detections,
    summarize_camera_detections,
)
from desktop.messages import CAMERA_IDLE_RUNTIME_STATUS

if TYPE_CHECKING:
    from core.recognition_service import DigitOCRService


class CameraOCRRuntime(RuntimeLifecycleMixin, RuntimeWorkerControlMixin, RuntimeLoopFacadeMixin):
    """Two-thread camera runtime that mirrors the lighter example workflow."""

    def __init__(
        self,
        *,
        service_factory: Callable[[], "DigitOCRService"] | None = None,
        worker_config: CameraOCRWorkerConfig | None = None,
        camera_mode: str = config.CAMERA_MODE_DIGIT,
        device_index: int = config.CAMERA_INDEX,
        capture_size: tuple[int, int] | None = None,
        preview_interval_seconds: float = config.CAMERA_PREVIEW_INTERVAL_MS / 1000.0,
        ocr_interval_seconds: float | None = None,
        startup_grace_seconds: float = config.CAMERA_STARTUP_GRACE_MS / 1000.0,
        idle_seconds: float = config.CAMERA_WORKER_IDLE_MS / 1000.0,
        retry_seconds: float = config.CAMERA_CAPTURE_RETRY_MS / 1000.0,
        target_capture_fps: float = config.CAMERA_TARGET_FPS,
        max_preview_size: int = config.CAMERA_MAX_PREVIEW_SIZE,
        max_ocr_side: int | None = None,
        roi_width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
        roi_height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
    ) -> None:
        if service_factory is None and worker_config is None:
            raise ValueError("CameraOCRRuntime needs either worker_config or service_factory.")

        resolved_mode = worker_config.camera_mode if worker_config is not None else camera_mode
        self._mode_profile = get_camera_mode_profile(resolved_mode)

        self._service_factory = service_factory
        self._worker_config = worker_config
        self._camera_mode = self._mode_profile.mode
        self._device_index = device_index
        default_capture_size = self._mode_profile.capture_size
        self._capture_size = default_capture_size if capture_size is None else capture_size
        self._preview_interval_seconds = max(preview_interval_seconds, 0.01)
        self._ocr_interval_seconds = (
            self._mode_profile.ocr_interval_seconds if ocr_interval_seconds is None else ocr_interval_seconds
        )
        self._startup_grace_seconds = max(startup_grace_seconds, 0.0)
        self._idle_seconds = idle_seconds
        self._retry_seconds = retry_seconds
        self._target_capture_fps = max(target_capture_fps, 1.0)
        self._max_preview_size = max_preview_size
        self._max_ocr_side = self._mode_profile.max_ocr_side if max_ocr_side is None else max_ocr_side
        self._roi_width_ratio = float(np.clip(roi_width_ratio, 0.20, 0.90))
        self._roi_height_ratio = float(np.clip(roi_height_ratio, 0.15, 0.85))

        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._latest_result: CameraInferenceResult | None = None
        self._worker_error: str | None = None
        self._capture_fps = 0.0
        self._ocr_fps = 0.0
        self._backend_name = ""
        self._status_text = CAMERA_IDLE_RUNTIME_STATUS
        self._inference_resume_at = 0.0
        self._ocr_process: mp.Process | None = None
        self._ocr_task_queue: "mp.Queue[CameraOCRTask | None] | None" = None
        self._ocr_result_queue: "mp.Queue[CameraOCRWorkerResult] | None" = None
        self._ocr_task_inflight = False
        self._ocr_task_started_at = 0.0
        self._ocr_task_frame_id = -1
        self._fast_processes: list[mp.Process] = []
        self._fast_task_queues: list["mp.Queue[CameraOCRTask | None]"] = []
        self._fast_result_queues: list["mp.Queue[CameraOCRWorkerResult]"] = []
        self._fast_task_inflight: list[bool] = []
        self._fast_pending_frames: dict[int, PendingFastFrame] = {}
        self._fast_last_submitted_frame_id = -1
        self._last_fallback_completed_at = 0.0
        self._roi_generation = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def camera_mode(self) -> str:
        return self._camera_mode

    @property
    def roi_width_ratio(self) -> float:
        return self._roi_width_ratio

    @property
    def roi_height_ratio(self) -> float:
        return self._roi_height_ratio


__all__ = [
    "CAMERA_EMPTY_RESET_FRAMES",
    "CAMERA_MODE_BOARD",
    "CAMERA_MODE_DIGIT",
    "CAMERA_MODE_HAND_COUNT",
    "CAMERA_PREVIEW_INTERVAL_MS",
    "CAMERA_ROI_HEIGHT_RATIO",
    "CAMERA_ROI_WIDTH_RATIO",
    "CAMERA_SEQUENCE_HISTORY",
    "CameraDetection",
    "CameraInferenceResult",
    "CameraOCRRuntime",
    "CameraOCRWorkerConfig",
    "CameraSnapshot",
    "CameraTrack",
    "build_camera_detections_from_results",
    "camera_detection_signature",
    "camera_result_is_fresh",
    "filter_camera_detections",
    "group_camera_detections",
    "overlay_camera_detections",
    "sort_camera_detections",
    "stable_camera_sequence",
    "stabilize_camera_detections",
    "summarize_camera_detections",
]
